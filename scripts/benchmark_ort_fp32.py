from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX Runtime FP32 inference for ResNet18-SiLU."
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("artifacts/onnx/resnet18_silu_fp32.onnx"),
        help="Path to the FP32 ONNX model.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8],
        help="Batch sizes to benchmark.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warm-up iterations.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=200,
        help="Number of measured iterations.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="ONNX Runtime intra-op threads. Use 0 for ORT default.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/ort_fp32_benchmark.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/ort_fp32_benchmark.json"),
        help="JSON output path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dummy input generation.",
    )
    return parser.parse_args()


def create_session(onnx_path: Path, threads: int) -> ort.InferenceSession:
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path.resolve()}")

    session_options = ort.SessionOptions()

    if threads > 0:
        session_options.intra_op_num_threads = threads

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    print("ONNX Runtime providers:", session.get_providers())
    print("Input name:", session.get_inputs()[0].name)
    print("Output name:", session.get_outputs()[0].name)

    return session


def model_size_mb(onnx_path: Path) -> float:
    total_bytes = onnx_path.stat().st_size

    # torch.onnx.export(dynamo=True) may generate an external data file.
    external_data_path = Path(str(onnx_path) + ".data")
    if external_data_path.exists():
        total_bytes += external_data_path.stat().st_size

    return total_bytes / 1024 / 1024


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("values must not be empty")

    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def benchmark_one_batch_size(
    session: ort.InferenceSession,
    batch_size: int,
    warmup: int,
    runs: int,
    rng: np.random.Generator,
) -> dict[str, float | int | str]:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Dummy CIFAR-10-like input after normalization. Random data is used to
    # measure pure inference latency without DataLoader overhead.
    images = rng.standard_normal(
        size=(batch_size, 3, 32, 32),
        dtype=np.float32,
    )

    for _ in range(warmup):
        session.run([output_name], {input_name: images})

    latencies_ms: list[float] = []

    for _ in range(runs):
        start = time.perf_counter()
        session.run([output_name], {input_name: images})
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000.0)

    mean_ms = statistics.fmean(latencies_ms)
    p50_ms = percentile(latencies_ms, 50)
    p95_ms = percentile(latencies_ms, 95)
    min_ms = min(latencies_ms)
    max_ms = max(latencies_ms)

    throughput = batch_size / (mean_ms / 1000.0)

    return {
        "backend": "ONNX Runtime CPUExecutionProvider",
        "precision": "FP32",
        "batch_size": batch_size,
        "warmup_runs": warmup,
        "measured_runs": runs,
        "mean_latency_ms": mean_ms,
        "p50_latency_ms": p50_ms,
        "p95_latency_ms": p95_ms,
        "min_latency_ms": min_ms,
        "max_latency_ms": max_ms,
        "throughput_samples_per_s": throughput,
    }


def save_results(
    rows: list[dict[str, float | int | str]],
    output_csv: Path,
    output_json: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def main() -> None:
    args = parse_args()

    rng = np.random.default_rng(args.seed)

    session = create_session(args.onnx, args.threads)
    size_mb = model_size_mb(args.onnx)

    rows: list[dict[str, float | int | str]] = []

    print("=" * 60)
    print("ONNX Runtime FP32 CPU Benchmark")
    print("=" * 60)
    print(f"ONNX model: {args.onnx}")
    print(f"Model size including external data: {size_mb:.2f} MB")
    print(f"Warm-up runs: {args.warmup}")
    print(f"Measured runs: {args.runs}")

    for batch_size in args.batch_sizes:
        row = benchmark_one_batch_size(
            session=session,
            batch_size=batch_size,
            warmup=args.warmup,
            runs=args.runs,
            rng=rng,
        )
        row["model_size_mb"] = size_mb
        rows.append(row)

        print("-" * 60)
        print(f"Batch size: {batch_size}")
        print(f"Mean latency: {row['mean_latency_ms']:.4f} ms")
        print(f"P50 latency:  {row['p50_latency_ms']:.4f} ms")
        print(f"P95 latency:  {row['p95_latency_ms']:.4f} ms")
        print(f"Throughput:   {row['throughput_samples_per_s']:.2f} samples/s")

    save_results(
        rows=rows,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )

    print("=" * 60)
    print("Benchmark completed")
    print("=" * 60)
    print(f"CSV:  {args.output_csv}")
    print(f"JSON: {args.output_json}")


if __name__ == "__main__":
    main()
