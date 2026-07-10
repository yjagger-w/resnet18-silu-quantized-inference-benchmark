from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from silu_benchmark.config import ExperimentConfig
from silu_benchmark.data import build_cifar10_loaders


class Cifar10CalibrationDataReader(CalibrationDataReader):
    def __init__(
        self,
        data_loader,
        input_name: str,
        max_batches: int,
    ) -> None:
        self.data_loader = data_loader
        self.input_name = input_name
        self.max_batches = max_batches
        self.iterator = iter(data_loader)
        self.batch_index = 0

    def get_next(self):
        if self.max_batches > 0 and self.batch_index >= self.max_batches:
            return None

        try:
            images, _ = next(self.iterator)
        except StopIteration:
            return None

        self.batch_index += 1
        return {
            self.input_name: images.detach().cpu().numpy().astype(np.float32)
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an ONNX Runtime INT8 PTQ experiment matrix over calibration "
            "method, weight granularity, and calibration sample count."
        )
    )
    parser.add_argument(
        "--model-input",
        type=Path,
        default=Path("artifacts/onnx/resnet18_silu_fp32.onnx"),
        help="Input FP32 ONNX model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/int8/matrix"),
        help="Directory for generated INT8 ONNX models.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="CIFAR-10 root directory.",
    )
    parser.add_argument(
        "--input-name",
        type=str,
        default="images",
        help="ONNX model input name.",
    )
    parser.add_argument(
        "--calibration-methods",
        nargs="+",
        choices=["MinMax", "Entropy", "Percentile"],
        default=["MinMax", "Entropy", "Percentile"],
        help="Calibration methods to evaluate.",
    )
    parser.add_argument(
        "--weight-granularities",
        nargs="+",
        choices=["per-tensor", "per-channel"],
        default=["per-tensor", "per-channel"],
        help="Weight quantization granularities to evaluate.",
    )
    parser.add_argument(
        "--calibration-samples",
        nargs="+",
        type=int,
        default=[128, 512, 1024, 2560],
        help="Calibration sample counts to evaluate.",
    )
    parser.add_argument(
        "--calibration-batch-size",
        type=int,
        default=128,
        help="Calibration batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=128,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=0,
        help="Limit evaluation batches for debugging. Use 0 for full CIFAR-10 test set.",
    )
    parser.add_argument(
        "--benchmark-batch-sizes",
        nargs="+",
        type=int,
        default=[1, 4, 8],
        help="Batch sizes for latency benchmarking.",
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=20,
        help="Warm-up runs for each latency benchmark.",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=200,
        help="Measured runs for each latency benchmark.",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip latency benchmarking and only report accuracy and model size.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="ONNX Runtime intra-op threads. Use 0 for ORT default.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse an existing INT8 model instead of quantizing again.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/quantization_matrix.csv"),
        help="CSV result path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/quantization_matrix.json"),
        help="JSON result path.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CIFAR-10 if it is not already present.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for latency benchmark input.",
    )
    return parser.parse_args()


def calibration_method_from_name(name: str) -> CalibrationMethod:
    mapping = {
        "MinMax": CalibrationMethod.MinMax,
        "Entropy": CalibrationMethod.Entropy,
        "Percentile": CalibrationMethod.Percentile,
    }
    return mapping[name]


def create_session(onnx_path: Path, threads: int) -> ort.InferenceSession:
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path.resolve()}")

    session_options = ort.SessionOptions()
    if threads > 0:
        session_options.intra_op_num_threads = threads

    return ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def safe_name(value: str) -> str:
    return value.lower().replace("-", "_")


def build_model_output_path(
    output_dir: Path,
    calibration_method: str,
    weight_granularity: str,
    calibration_samples: int,
) -> Path:
    return output_dir / (
        f"resnet18_silu_int8_"
        f"{safe_name(calibration_method)}_"
        f"{safe_name(weight_granularity)}_"
        f"calib{calibration_samples}.onnx"
    )


def quantize_model(
    model_input: Path,
    model_output: Path,
    calibration_loader,
    input_name: str,
    calibration_batches: int,
    calibration_method: str,
    weight_granularity: str,
) -> None:
    model_output.parent.mkdir(parents=True, exist_ok=True)

    calibration_reader = Cifar10CalibrationDataReader(
        data_loader=calibration_loader,
        input_name=input_name,
        max_batches=calibration_batches,
    )

    quantize_static(
        model_input=str(model_input),
        model_output=str(model_output),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=(weight_granularity == "per-channel"),
        reduce_range=False,
        calibrate_method=calibration_method_from_name(calibration_method),
    )


def evaluate_onnx_accuracy(
    onnx_path: Path,
    test_loader,
    threads: int,
    max_test_batches: int,
) -> dict[str, float | int]:
    session = create_session(onnx_path, threads)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    correct = 0
    total = 0

    for batch_index, (images, labels) in enumerate(test_loader):
        if max_test_batches > 0 and batch_index >= max_test_batches:
            break

        logits = session.run(
            [output_name],
            {input_name: images.detach().cpu().numpy().astype(np.float32)},
        )[0]
        preds = logits.argmax(axis=1)
        labels_np = labels.detach().cpu().numpy()

        correct += int((preds == labels_np).sum())
        total += int(labels_np.shape[0])

    if total == 0:
        raise RuntimeError("No samples were evaluated.")

    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
    }


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def benchmark_onnx(
    onnx_path: Path,
    precision: str,
    batch_sizes: list[int],
    warmup: int,
    runs: int,
    threads: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    session = create_session(onnx_path, threads)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    result: dict[str, float] = {}

    for batch_size in batch_sizes:
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
        throughput = batch_size / (mean_ms / 1000.0)

        prefix = f"batch{batch_size}"
        result[f"{prefix}_mean_latency_ms"] = mean_ms
        result[f"{prefix}_p50_latency_ms"] = p50_ms
        result[f"{prefix}_p95_latency_ms"] = p95_ms
        result[f"{prefix}_throughput_samples_per_s"] = throughput

    return result


def save_rows(
    rows: list[dict[str, object]],
    output_csv: Path,
    output_json: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def main() -> None:
    args = parse_args()

    if not args.model_input.exists():
        raise FileNotFoundError(
            f"Input ONNX model not found: {args.model_input.resolve()}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    total_experiments = (
        len(args.calibration_methods)
        * len(args.weight_granularities)
        * len(args.calibration_samples)
    )

    print("=" * 60)
    print("ONNX Runtime INT8 Quantization Matrix")
    print("=" * 60)
    print(f"Input FP32 model: {args.model_input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Calibration methods: {args.calibration_methods}")
    print(f"Weight granularities: {args.weight_granularities}")
    print(f"Calibration samples: {args.calibration_samples}")
    print(f"Total experiments: {total_experiments}")
    print(f"Skip benchmark: {args.skip_benchmark}")

    rows: list[dict[str, object]] = []
    experiment_index = 0

    for calibration_method in args.calibration_methods:
        for weight_granularity in args.weight_granularities:
            for calibration_samples in args.calibration_samples:
                experiment_index += 1

                calibration_batches = math.ceil(
                    calibration_samples / args.calibration_batch_size
                )
                actual_calibration_samples = (
                    calibration_batches * args.calibration_batch_size
                )

                model_output = build_model_output_path(
                    output_dir=args.output_dir,
                    calibration_method=calibration_method,
                    weight_granularity=weight_granularity,
                    calibration_samples=actual_calibration_samples,
                )

                print("-" * 60)
                print(
                    f"[{experiment_index}/{total_experiments}] "
                    f"method={calibration_method}, "
                    f"weight={weight_granularity}, "
                    f"calib_samples={actual_calibration_samples}"
                )
                print(f"Output: {model_output}")

                config = ExperimentConfig(
                    batch_size=args.calibration_batch_size,
                    test_batch_size=args.eval_batch_size,
                    calib_batches=calibration_batches,
                    data_root=args.data_root,
                    weights_path="",
                )

                _, test_set, calibration_set, calibration_loader, test_loader = (
                    build_cifar10_loaders(
                        config=config,
                        download=args.download,
                    )
                )

                if args.skip_existing and model_output.exists():
                    print("Reusing existing INT8 model.")
                else:
                    quantize_model(
                        model_input=args.model_input,
                        model_output=model_output,
                        calibration_loader=calibration_loader,
                        input_name=args.input_name,
                        calibration_batches=calibration_batches,
                        calibration_method=calibration_method,
                        weight_granularity=weight_granularity,
                    )

                accuracy_result = evaluate_onnx_accuracy(
                    onnx_path=model_output,
                    test_loader=test_loader,
                    threads=args.threads,
                    max_test_batches=args.max_test_batches,
                )

                row: dict[str, object] = {
                    "calibration_method": calibration_method,
                    "weight_granularity": weight_granularity,
                    "requested_calibration_samples": calibration_samples,
                    "actual_calibration_samples": len(calibration_set),
                    "calibration_batches": calibration_batches,
                    "model_path": str(model_output),
                    "model_size_mb": file_size_mb(model_output),
                    "correct": accuracy_result["correct"],
                    "total": accuracy_result["total"],
                    "accuracy": accuracy_result["accuracy"],
                    "accuracy_percent": accuracy_result["accuracy"] * 100,
                }

                if not args.skip_benchmark:
                    benchmark_result = benchmark_onnx(
                        onnx_path=model_output,
                        precision="INT8",
                        batch_sizes=args.benchmark_batch_sizes,
                        warmup=args.benchmark_warmup,
                        runs=args.benchmark_runs,
                        threads=args.threads,
                        rng=rng,
                    )
                    row.update(benchmark_result)

                rows.append(row)

                print(f"Accuracy: {row['accuracy_percent']:.4f}%")
                print(f"Model size: {row['model_size_mb']:.2f} MB")

                save_rows(rows, args.output_csv, args.output_json)
                print(f"Saved partial results to: {args.output_csv}")

    print("=" * 60)
    print("Quantization matrix completed")
    print("=" * 60)
    print(f"CSV:  {args.output_csv}")
    print(f"JSON: {args.output_json}")

    best_accuracy = max(rows, key=lambda item: float(item["accuracy"]))
    smallest_model = min(rows, key=lambda item: float(item["model_size_mb"]))

    print("-" * 60)
    print("Best accuracy setting:")
    print(
        f"{best_accuracy['calibration_method']} / "
        f"{best_accuracy['weight_granularity']} / "
        f"{best_accuracy['actual_calibration_samples']} samples / "
        f"{best_accuracy['accuracy_percent']:.4f}%"
    )

    print("Smallest model setting:")
    print(
        f"{smallest_model['calibration_method']} / "
        f"{smallest_model['weight_granularity']} / "
        f"{smallest_model['actual_calibration_samples']} samples / "
        f"{smallest_model['model_size_mb']:.2f} MB"
    )


if __name__ == "__main__":
    main()
