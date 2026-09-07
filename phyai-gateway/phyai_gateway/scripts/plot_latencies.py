import argparse
import csv
from pathlib import Path

COLUMNS = (
    "pickle_encode_ms",
    "model_request_infer_ms",
    "inference_time_ms",
    "total_gateway_ms",
)


def read_latencies(path: Path) -> dict[str, list[float]]:
    values = {name: [] for name in COLUMNS}
    with path.open("r", encoding="utf-8", newline="") as file:
        for row_number, row in enumerate(csv.reader(file), start=1):
            if not row:
                continue
            if row_number == 1 and row == list(COLUMNS):
                continue
            if len(row) != len(COLUMNS):
                raise ValueError(
                    f"row {row_number} has {len(row)} columns; expected {len(COLUMNS)}"
                )
            for name, value in zip(COLUMNS, row, strict=True):
                values[name].append(float(value))

    if not values[COLUMNS[0]]:
        raise ValueError(f"no latency samples found in {path}")
    return values


def plot_latencies(values: dict[str, list[float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "pickle_encode_ms": ("#2a78d6", "-"),
        "model_request_infer_ms": ("#008300", "--"),
        "inference_time_ms": ("#e87ba4", "-."),
        "total_gateway_ms": ("#eda100", ":"),
    }

    requests = range(1, len(values[COLUMNS[0]]) + 1)
    figure, axis = plt.subplots(figsize=(12, 6))
    for name in COLUMNS:
        color, line_style = styles[name]
        axis.plot(
            requests,
            values[name],
            color=color,
            linestyle=line_style,
            linewidth=1.8,
            label=name,
        )

    axis.set_title("Gateway Latency by Request")
    axis.set_xlabel("Request")
    axis.set_ylabel("Latency (ms)")
    axis.grid(True, color="#e1e0d9", linewidth=0.8, alpha=0.8)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

def summarize_latencies(
    csv_path="Latencies.csv",
    output_path="latency_summary.xlsx",
):
    import pandas as pd
    import numpy as np

    # 读取 CSV
    df = pd.read_csv(csv_path)
    df["gateway_local_getactions_ms"] = (
        pd.to_numeric(df["total_gateway_ms"], errors="coerce")
        - pd.to_numeric(df["model_request_infer_ms"], errors="coerce")
    )

    summaries = {}

    # 对每一列分别统计
    for column in df.columns:
        samples = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna().to_numpy(dtype=np.float64)

        if len(samples) == 0:
            continue

        summaries[column] = {
            "avg": float(np.mean(samples)),
            "p50": float(np.percentile(samples, 50)),
            "p95": float(np.percentile(samples, 95)),
            "p99": float(np.percentile(samples, 99)),
            "min": float(np.min(samples)),
            "max": float(np.max(samples)),
        }

        # 输出到终端
        print(f"{column}: {summaries[column]}")

    # 转成 DataFrame
    summary_df = pd.DataFrame.from_dict(
        summaries,
        orient="index",
    )

    summary_df.index.name = "metric"

    # 保存 Excel
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_excel(
        output_path,
        index=True,
    )

    print(f"Latency summary saved to {output_path}")

    return summaries


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent / "outputs" / "latencies"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=base_dir / "Latencies.csv",
    )

    parser.add_argument(
        "--plot-output",
        type=Path,
        default=base_dir / "Latencies.png",
    )

    parser.add_argument(
        "--excel-output",
        type=Path,
        default=base_dir / "LatencySummary.xlsx",
    )

    args = parser.parse_args()

    values = read_latencies(args.input)

    plot_latencies(
        values,
        args.plot_output,
    )

    print(f"Latency plot saved to {args.plot_output}")

    summarize_latencies(
        csv_path=args.input,
        output_path=args.excel_output,
    )


if __name__ == "__main__":
    main()
