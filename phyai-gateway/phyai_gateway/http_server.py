import threading

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def create_http_app():
    app = FastAPI(title="Robot Gateway HTTP API")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/inference/{client_type}")
    async def receive_inference(client_type: str, request: Request):
        body = await request.body()
        print(
            f"HTTP request received: client_type={client_type}, "
            f"content_type={request.headers.get('content-type')}, "
            f"body_bytes={len(body)}",
            flush=True,
        )
        return JSONResponse(
            status_code=202,
            content={
                "status": "received",
                "client_type": client_type,
                "body_bytes": len(body),
            },
        )

    return app


class HTTPServer:
    def __init__(self, host: str, port: int):
        self._server = uvicorn.Server(
            uvicorn.Config(
                create_http_app(),
                host=host,
                port=port,
                log_level="info",
            )
        )
        self._thread = threading.Thread(
            target=self._server.run,
            name="gateway-http",
            daemon=True,
        )

    def start(self):
        self._thread.start()

    def stop(self):
        self._server.should_exit = True
        self._thread.join()
