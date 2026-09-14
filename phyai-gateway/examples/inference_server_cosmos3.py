"""Cosmos3 policy Model Server for the canonical inference protocol."""

import argparse
import json
import logging
import os
import threading
import time
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np
import torch

import phyai_gateway.bindings.model_inference_pb2 as model_inference_pb2
import phyai_gateway.bindings.model_inference_pb2_grpc as model_inference_pb2_grpc
from phyai.engine import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.cosmos3 import Cosmos3ActionRequest, pixel_to_latent_shape
from phyai.models.cosmos3.main_cosmos3_policy import Cosmos3PolicyArgs
from phyai_utils_tools.models.cosmos3 import Cosmos3PolicyProcessor

DEFAULT_CHECKPOINT_DIR = Path("/data/share/Cosmos3-Nano-Policy-DROID")
DEFAULT_LISTEN_ADDRESS = "[::]:50064"
DEFAULT_ADVERTISED_ENDPOINT = "127.0.0.1:50064"
DEFAULT_GATEWAY_REGISTRY_ADDRESS = "127.0.0.1:50052"
DEFAULT_MODEL_NAME = "cosmos3"
DEFAULT_DOMAIN_NAME = "droid_lerobot"
DEFAULT_ACTION_CHUNK_SIZE = 32
DEFAULT_RAW_ACTION_DIM = 10
DEFAULT_ACTION_DIM = 64
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832
DEFAULT_IMAGE_SIZE = 480
DEFAULT_FPS = 15.0
DEFAULT_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_FLOW_SHIFT = 10.0
DEFAULT_SEED = 42
REGISTRATION_RETRY_SECONDS = 5
IMAGE_NAMES = ("agentview", "robot0_eye_in_hand")
MAX_MESSAGE_BYTES = 100 * 1024 * 1024

class ModelRegistryReporter:
    def __init__(self, registry_address, endpoint, model_name):
        self.endpoint = endpoint
        self.model_name = model_name
        self.channel = grpc.insecure_channel(registry_address)
        self.stub = model_inference_pb2_grpc.ModelRegistryStub(self.channel)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="model-registry-reporter", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=4)
        self.channel.close()

    def _register(self):
        response = self.stub.Register(model_inference_pb2.RegisterRequest(endpoint=self.endpoint, model_name=self.model_name), timeout=3)
        if not response.accepted:
            logging.warning("Model Server registration rejected: %s", response.message)
            return None
        if not response.server_id:
            logging.warning("Gateway accepted registration without server_id")
            return None
        interval = max(1, response.heartbeat_interval_seconds)
        logging.info("Registered with Gateway: server_id=%s, heartbeat_interval=%ss", response.server_id, interval)
        return response.server_id, interval

    def _run(self):
        registration = None
        while not self.stop_event.is_set():
            if registration is None:
                try:
                    registration = self._register()
                except grpc.RpcError as error:
                    logging.warning("Model Server registration failed: %s: %s", error.code(), error.details())
                if registration is None:
                    self.stop_event.wait(REGISTRATION_RETRY_SECONDS)
                    continue
            server_id, interval = registration
            if self.stop_event.wait(interval):
                return
            try:
                response = self.stub.Heartbeat(model_inference_pb2.HeartbeatRequest(server_id=server_id), timeout=3)
            except grpc.RpcError as error:
                logging.warning("Model Server heartbeat failed: %s: %s", error.code(), error.details())
                continue
            if response.registered:
                continue
            logging.warning("Gateway no longer recognizes server_id=%s: %s; registering again", server_id, response.message)
            registration = None

class Cosmos3Runtime:
    def __init__(self, settings):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Cosmos3 inference")
        self.settings = settings
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        checkpoint = settings.checkpoint
        logging.info("Loading Cosmos3 policy checkpoint from %s", checkpoint)
        self.processor = Cosmos3PolicyProcessor(
            tokenizer_name_or_path=str(checkpoint / "text_tokenizer"),
            height=settings.height,
            width=settings.width,
            num_frames=settings.action_chunk_size + 1,
            mode="policy",
            domain_name=settings.domain_name,
            raw_action_dim=settings.raw_action_dim,
            action_chunk_size=settings.action_chunk_size,
            fps=settings.fps,
            image_size=settings.image_size,
            append_metadata=True,
            prompt_format="json",
            view_point=settings.view_point,
            negative_prompt=settings.negative_prompt,
            device=self.device,
            params_dtype=self.dtype,
        )
        self.engine = Engine(EngineArgs(
            plugin="cosmos3_policy",
            plugin_args=Cosmos3PolicyArgs(
                checkpoint_dir=checkpoint,
                flow_shift=settings.flow_shift,
                use_karras_sigmas=settings.use_karras_sigmas,
                decode_video=True,
            ),
            config=EngineConfig(
                device=DeviceConfig(target="cuda", params_dtype=self.dtype),
                runtime=RuntimeConfig(use_cuda_graph=False),
            ),
        ))
        logging.info("Cosmos3 runtime is ready: domain=%s action_chunk=%d raw_action_dim=%d", settings.domain_name, settings.action_chunk_size, settings.raw_action_dim)

    def infer(self, image, instruction, horizon):
        processed = self.processor.preprocess({"images": image, "task": instruction})
        request = Cosmos3ActionRequest(
            text_ids=processed.text_ids.to(self.device),
            text_mask=processed.text_mask.to(self.device),
            neg_text_ids=processed.neg_text_ids.to(self.device),
            neg_text_mask=processed.neg_text_mask.to(self.device),
            video_shape=pixel_to_latent_shape(*processed.video_shape),
            mode=processed.mode,
            domain_id=processed.domain_id,
            action_chunk=processed.action_chunk,
            raw_action_dim=processed.raw_action_dim,
            action_dim=self.settings.action_dim,
            cond_video_pixels=processed.pixel_values.to(device=self.device, dtype=self.dtype),
            cond_frame_indexes=processed.cond_frame_indexes,
            fps=self.settings.fps,
            num_inference_steps=self.settings.inference_steps,
            guidance_scale=self.settings.guidance_scale,
            seed=self.settings.seed,
        )
        torch.cuda.synchronize()
        start_ns = time.perf_counter_ns()
        result = self.engine.step(request)
        torch.cuda.synchronize()
        inference_time_us = (time.perf_counter_ns() - start_ns) // 1000
        actions = result.get("action") if isinstance(result, dict) else result
        if not isinstance(actions, torch.Tensor) or actions.ndim != 3 or actions.shape[0] != 1:
            shape = tuple(actions.shape) if isinstance(actions, torch.Tensor) else type(actions).__name__
            raise RuntimeError(f"unexpected Cosmos3 action shape: {shape}")
        if actions.shape[1] < horizon:
            raise RuntimeError(f"Cosmos3 returned horizon {actions.shape[1]}, requested {horizon}")
        actions = actions[0, :horizon, :self.settings.raw_action_dim].to(torch.float32).contiguous().cpu()
        if actions.shape[1] != self.settings.raw_action_dim:
            raise RuntimeError("Cosmos3 action width does not match configured raw_action_dim")
        if not torch.isfinite(actions).all():
            raise RuntimeError("Cosmos3 returned non-finite actions")
        return actions, inference_time_us

    def infer_batch(self, images, instructions, horizon):
        if len(images) != len(instructions):
            raise RuntimeError("Cosmos3 image and instruction batch sizes differ")
        outputs = []
        total_inference_time_us = 0
        for image, instruction in zip(images, instructions):
            action, elapsed_us = self.infer(image, instruction, horizon)
            outputs.append(action)
            total_inference_time_us += elapsed_us
        return torch.stack(outputs), total_inference_time_us

    def close(self):
        if self.engine is not None:
            self.engine.close()
            self.engine = None

class ModelInferenceServicer(model_inference_pb2_grpc.ModelInferenceServicer):
    def __init__(self, runtime):
        self.runtime = runtime

    def Infer(self, request, context):
        infer_start = time.perf_counter()
        images, state, instructions, is_batch = self._validate_and_decode(request, context, self.runtime.settings)
        del state
        if not context.is_active():
            context.abort(grpc.StatusCode.CANCELLED, "request was cancelled")
        try:
            actions, inference_time_us = self.runtime.infer_batch(images, instructions, request.requested_action_horizon)
        except torch.cuda.OutOfMemoryError:
            logging.exception("CUDA out of memory for request %s", request.request_id)
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "Cosmos3 inference ran out of GPU memory")
        except Exception:
            logging.exception("Cosmos3 inference failed for request %s", request.request_id)
            context.abort(grpc.StatusCode.INTERNAL, "Cosmos3 inference failed")
        if not is_batch:
            actions = actions[0]
        action_array = actions.numpy().astype("<f4", copy=False)
        return model_inference_pb2.InferenceResponse(request_id=request.request_id, actions=model_inference_pb2.Tensor(data=action_array.tobytes(order="C"), shape=list(action_array.shape), dtype=model_inference_pb2.DATA_TYPE_FLOAT32), inference_time_us=int(inference_time_us))

    @staticmethod
    def _validate_and_decode(request, context, settings):
        if not request.request_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "request_id is required")
        if len(request.images) != len(IMAGE_NAMES):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"exactly {len(IMAGE_NAMES)} images are required")
        if not 1 <= request.requested_action_horizon <= settings.action_chunk_size:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid requested_action_horizon")
        by_name = {}
        batch = None
        batch_size = None
        for image in request.images:
            if image.name not in IMAGE_NAMES or image.name in by_name:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "image names must be exactly agentview and robot0_eye_in_hand")
            if image.dtype != model_inference_pb2.DATA_TYPE_UINT8 or image.encoding != model_inference_pb2.IMAGE_ENCODING_RAW or image.layout != model_inference_pb2.IMAGE_LAYOUT_HWC:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "images must be UINT8 RAW HWC")
            shape = tuple(image.shape)
            if len(shape) == 3 and shape[2] == 3 and all(x > 0 for x in shape):
                current_batch = False
                current_size = 1
            elif len(shape) == 4 and shape[3] == 3 and all(x > 0 for x in shape):
                current_batch = True
                current_size = shape[0]
            else:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "image shape must be [H, W, 3] or [B, H, W, 3]")
            if batch is None:
                batch, batch_size = current_batch, current_size
            elif batch != current_batch or batch_size != current_size:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "all images must use the same batch shape")
            if len(image.data) != int(np.prod(shape)):
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "image data length does not match shape")
            by_name[image.name] = image
        if batch:
            try:
                ext = json.loads(request.extensions_json)
            except (TypeError, ValueError):
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "extensions_json must be valid JSON for batch inference")
            instructions = ext.get("instructions") if isinstance(ext, dict) else None
            if not isinstance(instructions, list) or len(instructions) != batch_size or any(not isinstance(x, str) or not x.strip() for x in instructions):
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"extensions_json.instructions must contain exactly {batch_size} non-empty strings")
        else:
            if not request.instruction.strip():
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "instruction is required")
            instructions = [request.instruction]
        state = request.robot_state
        expected = (batch_size, 8) if batch else (8,)
        if state.dtype != model_inference_pb2.DATA_TYPE_FLOAT32 or tuple(state.shape) != expected or len(state.data) != int(np.prod(expected)) * 4:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"robot_state must be FLOAT32 with shape {list(expected)}")
        if not np.isfinite(np.frombuffer(state.data, dtype="<f4")).all():
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "robot_state must contain finite values")
        selected = by_name[settings.image_name]
        arr = np.frombuffer(selected.data, dtype=np.uint8).reshape(tuple(selected.shape)).copy()
        images = [arr] if not batch else [arr[i] for i in range(batch_size)]
        return images, np.frombuffer(state.data, dtype="<f4").reshape(expected).copy(), instructions, batch

def parse_args():
    parser = argparse.ArgumentParser(description="Cosmos3 gRPC Model Server")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--listen", default=DEFAULT_LISTEN_ADDRESS)
    parser.add_argument("--advertised-endpoint", default=DEFAULT_ADVERTISED_ENDPOINT)
    parser.add_argument("--gateway-registry", default=DEFAULT_GATEWAY_REGISTRY_ADDRESS)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--domain-name", default=DEFAULT_DOMAIN_NAME)
    parser.add_argument("--raw-action-dim", type=int, default=DEFAULT_RAW_ACTION_DIM)
    parser.add_argument("--action-dim", type=int, default=DEFAULT_ACTION_DIM)
    parser.add_argument("--action-chunk-size", type=int, default=DEFAULT_ACTION_CHUNK_SIZE)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--image-name", default="agentview")
    parser.add_argument("--view-point", default="ego_view")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--inference-steps", type=int, default=DEFAULT_INFERENCE_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--flow-shift", type=float, default=DEFAULT_FLOW_SHIFT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--use-karras-sigmas", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    return parser.parse_args()

def serve(settings):
    if settings.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runtime = Cosmos3Runtime(settings)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=[("grpc.max_receive_message_length", MAX_MESSAGE_BYTES), ("grpc.max_send_message_length", MAX_MESSAGE_BYTES)])
    model_inference_pb2_grpc.add_ModelInferenceServicer_to_server(ModelInferenceServicer(runtime), server)
    server.add_insecure_port(settings.listen)
    registry_reporter = ModelRegistryReporter(settings.gateway_registry, settings.advertised_endpoint, settings.model_name)
    server.start()
    logging.info("Cosmos3 ModelInference Server listening on %s", settings.listen)
    registry_reporter.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Stopping Cosmos3 ModelInference Server")
    finally:
        registry_reporter.stop()
        server.stop(grace=2).wait()
        runtime.close()

if __name__ == "__main__":
    serve(parse_args())
