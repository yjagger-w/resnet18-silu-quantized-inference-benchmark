from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an automatic Markdown/CSV benchmark summary."
    )
    parser.add_argument("--matrix-csv", type=Path, default=Path("results/quantization_matrix_minmax.csv"))
    parser.add_argument("--fp32-benchmark-csv", type=Path, default=Path("results/ort_fp32_benchmark.csv"))
    parser.add_argument("--int8-benchmark-csv", type=Path, default=Path("results/ort_int8_benchmark.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/benchmark_summary.md"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/benchmark_summary.csv"))
    parser.add_argument("--fp32-accuracy-percent", type=float, default=93.7400)
    parser.add_argument("--fp32-correct", type=int, default=9374)
    parser.add_argument("--total-samples", type=int, default=10000)
    parser.add_argument("--project-name", type=str, default="ResNet18-SiLU Quantized Inference Benchmark")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path.resolve()}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def maybe_read_csv(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def get_fp32_benchmarks(rows: list[dict[str, str]]) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for row in rows:
        batch_size = to_int(row.get("batch_size"))
        if batch_size <= 0:
            continue
        result[batch_size] = {
            "mean_latency_ms": to_float(row.get("mean_latency_ms")),
            "p50_latency_ms": to_float(row.get("p50_latency_ms")),
            "p95_latency_ms": to_float(row.get("p95_latency_ms")),
            "throughput_samples_per_s": to_float(row.get("throughput_samples_per_s")),
            "model_size_mb": to_float(row.get("model_size_mb")),
        }
    return result


def fp32_model_size(fp32_benchmarks: dict[int, dict[str, float]], default: float = 42.62) -> float:
    for item in fp32_benchmarks.values():
        size = item.get("model_size_mb", 0.0)
        if size > 0:
            return size
    return default


def get_matrix_batch_sizes(matrix_rows: list[dict[str, str]]) -> list[int]:
    batch_sizes: set[int] = set()
    for row in matrix_rows:
        for key in row.keys():
            if key.startswith("batch") and key.endswith("_mean_latency_ms"):
                middle = key.removeprefix("batch").removesuffix("_mean_latency_ms")
                if middle.isdigit():
                    batch_sizes.add(int(middle))
    return sorted(batch_sizes)


def row_label(row: dict[str, Any]) -> str:
    return (
        f"{row['calibration_method']} / "
        f"{row['weight_granularity']} / "
        f"{row['actual_calibration_samples']} samples"
    )


def summarize_matrix(
    matrix_rows: list[dict[str, str]],
    fp32_accuracy_percent: float,
    fp32_size_mb: float,
    fp32_benchmarks: dict[int, dict[str, float]],
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    batch_sizes = get_matrix_batch_sizes(matrix_rows)

    for row in matrix_rows:
        accuracy_percent = to_float(row.get("accuracy_percent"))
        model_size_mb = to_float(row.get("model_size_mb"))
        summary: dict[str, Any] = {
            "calibration_method": row.get("calibration_method", ""),
            "weight_granularity": row.get("weight_granularity", ""),
            "actual_calibration_samples": to_int(row.get("actual_calibration_samples")),
            "accuracy_percent": accuracy_percent,
            "accuracy_drop_pp": fp32_accuracy_percent - accuracy_percent,
            "model_size_mb": model_size_mb,
            "size_reduction_percent": (
                (fp32_size_mb - model_size_mb) / fp32_size_mb * 100 if fp32_size_mb > 0 else 0.0
            ),
        }

        for batch_size in batch_sizes:
            mean_key = f"batch{batch_size}_mean_latency_ms"
            p50_key = f"batch{batch_size}_p50_latency_ms"
            p95_key = f"batch{batch_size}_p95_latency_ms"
            throughput_key = f"batch{batch_size}_throughput_samples_per_s"
            mean_latency = to_float(row.get(mean_key))
            fp32_mean = fp32_benchmarks.get(batch_size, {}).get("mean_latency_ms", 0.0)
            summary[mean_key] = mean_latency
            summary[p50_key] = to_float(row.get(p50_key))
            summary[p95_key] = to_float(row.get(p95_key))
            summary[throughput_key] = to_float(row.get(throughput_key))
            summary[f"batch{batch_size}_speedup_vs_fp32"] = (
                fp32_mean / mean_latency if fp32_mean > 0 and mean_latency > 0 else 0.0
            )
        summary_rows.append(summary)
    return summary_rows


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No rows to write.")
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_best_rows(rows: list[dict[str, Any]], fp32_benchmarks: dict[int, dict[str, float]]) -> dict[str, Any]:
    best_accuracy = max(rows, key=lambda row: float(row["accuracy_percent"]))
    smallest_model = min(rows, key=lambda row: float(row["model_size_mb"]))
    best_speedup_by_batch: dict[int, dict[str, Any]] = {}
    for batch_size in sorted(fp32_benchmarks):
        speedup_key = f"batch{batch_size}_speedup_vs_fp32"
        candidates = [row for row in rows if float(row.get(speedup_key, 0.0)) > 0]
        if candidates:
            best_speedup_by_batch[batch_size] = max(candidates, key=lambda row: float(row[speedup_key]))
    return {
        "best_accuracy": best_accuracy,
        "smallest_model": smallest_model,
        "best_speedup_by_batch": best_speedup_by_batch,
    }


def write_markdown_report(
    path: Path,
    project_name: str,
    fp32_accuracy_percent: float,
    fp32_correct: int,
    total_samples: int,
    fp32_size_mb: float,
    fp32_benchmarks: dict[int, dict[str, float]],
    summary_rows: list[dict[str, Any]],
    best: dict[str, Any],
    matrix_csv: Path,
    output_csv: Path,
    int8_benchmark_rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = best["best_accuracy"]
    smallest_model = best["smallest_model"]
    best_speedup_by_batch = best["best_speedup_by_batch"]
    batch_sizes = sorted(fp32_benchmarks)

    lines: list[str] = []
    lines.append(f"# {project_name}: Benchmark Summary")
    lines.append("")
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(
        "This report summarizes FP32 and INT8 ONNX Runtime benchmarking results "
        "for ResNet18-SiLU on CIFAR-10. It is generated automatically from local CSV outputs."
    )
    lines.append("")
    lines.append("## FP32 Baseline")
    lines.append("")
    lines.append(markdown_table(["Metric", "Value"], [
        ["Accuracy", f"{fp32_accuracy_percent:.4f}%"],
        ["Correct / Total", f"{fp32_correct} / {total_samples}"],
        ["Model size", f"{fp32_size_mb:.2f} MB"],
    ]))
    lines.append("")

    if fp32_benchmarks:
        lines.append("### FP32 CPU Latency")
        lines.append("")
        fp32_latency_rows = []
        for batch_size in batch_sizes:
            item = fp32_benchmarks[batch_size]
            fp32_latency_rows.append([
                str(batch_size),
                f"{item['mean_latency_ms']:.4f} ms",
                f"{item['p50_latency_ms']:.4f} ms",
                f"{item['p95_latency_ms']:.4f} ms",
                f"{item['throughput_samples_per_s']:.2f} samples/s",
            ])
        lines.append(markdown_table(["Batch", "Mean latency", "P50 latency", "P95 latency", "Throughput"], fp32_latency_rows))
        lines.append("")

    lines.append("## Best INT8 Results from Quantization Matrix")
    lines.append("")
    lines.append(markdown_table(
        ["Selection", "Setting", "Accuracy", "Accuracy drop", "Model size"],
        [[
            "Best accuracy",
            row_label(best_accuracy),
            f"{best_accuracy['accuracy_percent']:.4f}%",
            f"{best_accuracy['accuracy_drop_pp']:.4f} pp",
            f"{best_accuracy['model_size_mb']:.2f} MB",
        ], [
            "Smallest model",
            row_label(smallest_model),
            f"{smallest_model['accuracy_percent']:.4f}%",
            f"{smallest_model['accuracy_drop_pp']:.4f} pp",
            f"{smallest_model['model_size_mb']:.2f} MB",
        ]]
    ))
    lines.append("")

    if best_speedup_by_batch:
        lines.append("### Best Latency Speedup by Batch Size")
        lines.append("")
        speedup_rows = []
        for batch_size, row in best_speedup_by_batch.items():
            speedup_key = f"batch{batch_size}_speedup_vs_fp32"
            latency_key = f"batch{batch_size}_mean_latency_ms"
            speedup_rows.append([
                str(batch_size),
                row_label(row),
                f"{row[latency_key]:.4f} ms",
                f"{row[speedup_key]:.2f}x",
            ])
        lines.append(markdown_table(["Batch", "Setting", "INT8 mean latency", "Speedup vs FP32"], speedup_rows))
        lines.append("")

    lines.append("## Quantization Matrix")
    lines.append("")
    matrix_table_rows = []
    for row in summary_rows:
        table_row = [
            str(row["calibration_method"]),
            str(row["weight_granularity"]),
            str(row["actual_calibration_samples"]),
            f"{row['accuracy_percent']:.4f}%",
            f"{row['accuracy_drop_pp']:.4f} pp",
            f"{row['model_size_mb']:.2f} MB",
            f"{row['size_reduction_percent']:.2f}%",
        ]
        for batch_size in batch_sizes:
            table_row.append(f"{row.get(f'batch{batch_size}_mean_latency_ms', 0.0):.4f} ms")
            table_row.append(f"{row.get(f'batch{batch_size}_speedup_vs_fp32', 0.0):.2f}x")
        matrix_table_rows.append(table_row)

    headers = ["Method", "Weight", "Samples", "Accuracy", "Drop", "Size", "Size reduction"]
    for batch_size in batch_sizes:
        headers.extend([f"B{batch_size} latency", f"B{batch_size} speedup"])
    lines.append(markdown_table(headers, matrix_table_rows))
    lines.append("")

    lines.append("## Key Takeaways")
    lines.append("")
    lines.append(
        f"- Best INT8 accuracy is **{best_accuracy['accuracy_percent']:.4f}%**, "
        f"with only **{best_accuracy['accuracy_drop_pp']:.4f} percentage-point** "
        "accuracy drop from the FP32 baseline."
    )
    lines.append(
        f"- The best-accuracy INT8 model size is **{best_accuracy['model_size_mb']:.2f} MB**, "
        f"compared with **{fp32_size_mb:.2f} MB** for FP32."
    )
    if best_speedup_by_batch:
        global_best_batch = None
        global_best_row = None
        global_best_speedup = -1.0
        for batch_size, row in best_speedup_by_batch.items():
            speedup = float(row.get(f"batch{batch_size}_speedup_vs_fp32", 0.0))
            if speedup > global_best_speedup:
                global_best_speedup = speedup
                global_best_batch = batch_size
                global_best_row = row
        if global_best_row is not None:
            lines.append(
                f"- Best observed latency speedup is **{global_best_speedup:.2f}x** "
                f"at batch size **{global_best_batch}**, using **{row_label(global_best_row)}**."
            )
    lines.append(
        "- Entropy and Percentile calibration are not included in this summary because "
        "Entropy calibration was unstable in the current local ONNX Runtime environment. "
        "The stable v0.7 matrix uses MinMax calibration."
    )
    if int8_benchmark_rows:
        lines.append("- A standalone INT8 benchmark CSV was detected, but the matrix result is the primary v0.8 summary.")
    lines.append("")
    lines.append("## Source Files")
    lines.append("")
    lines.append(markdown_table(["Item", "Path"], [
        ["Quantization matrix CSV", f"`{matrix_csv}`"],
        ["Generated summary CSV", f"`{output_csv}`"],
    ]))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    matrix_rows = read_csv(args.matrix_csv)
    fp32_rows = read_csv(args.fp32_benchmark_csv)
    int8_rows = maybe_read_csv(args.int8_benchmark_csv)
    fp32_benchmarks = get_fp32_benchmarks(fp32_rows)
    fp32_size_mb = fp32_model_size(fp32_benchmarks)
    summary_rows = summarize_matrix(
        matrix_rows=matrix_rows,
        fp32_accuracy_percent=args.fp32_accuracy_percent,
        fp32_size_mb=fp32_size_mb,
        fp32_benchmarks=fp32_benchmarks,
    )
    write_summary_csv(summary_rows, args.output_csv)
    best = find_best_rows(summary_rows, fp32_benchmarks)
    write_markdown_report(
        path=args.output_md,
        project_name=args.project_name,
        fp32_accuracy_percent=args.fp32_accuracy_percent,
        fp32_correct=args.fp32_correct,
        total_samples=args.total_samples,
        fp32_size_mb=fp32_size_mb,
        fp32_benchmarks=fp32_benchmarks,
        summary_rows=summary_rows,
        best=best,
        matrix_csv=args.matrix_csv,
        output_csv=args.output_csv,
        int8_benchmark_rows=int8_rows,
    )
    print("=" * 60)
    print("Benchmark report generated")
    print("=" * 60)
    print(f"Markdown report: {args.output_md}")
    print(f"Summary CSV:     {args.output_csv}")
    print("-" * 60)
    best_accuracy = best["best_accuracy"]
    print("Best accuracy setting:")
    print(
        f"{row_label(best_accuracy)} / "
        f"{best_accuracy['accuracy_percent']:.4f}% / "
        f"drop {best_accuracy['accuracy_drop_pp']:.4f} pp"
    )


if __name__ == "__main__":
    main()
