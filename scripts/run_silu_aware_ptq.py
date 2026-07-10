from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from silu_benchmark.calibration import (
    calibrate_ncnn_style,
    calibrate_silu_aware,
    insert_ncnn_quantizers,
    insert_silu_aware_quantizers,
)
from silu_benchmark.config import ExperimentConfig, QuantizationConfig
from silu_benchmark.data import build_cifar10_loaders
from silu_benchmark.models import ResNet18
from silu_benchmark.quantization.weights import (
    apply_bias_correction,
    quantize_weights_per_channel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PyTorch-side NCNN-style and SiLU-aware PTQ simulation for "
            "ResNet18-SiLU on CIFAR-10."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/resnet18_cifar10.pth"),
        help="FP32 PyTorch checkpoint.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="CIFAR-10 root directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Evaluation and calibration batch size.",
    )
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=20,
        help="Calibration batches. 20 x 128 = 2560 images.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
        help="Weight and activation quantization bit width.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["ncnn", "silu_aware"],
        default=["ncnn", "silu_aware"],
        help="PTQ methods to run.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="PyTorch device.",
    )
    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=0,
        help="Limit test batches for debugging. Use 0 for the full test set.",
    )
    parser.add_argument(
        "--layer-error-batches",
        type=int,
        default=5,
        help="Number of test batches used for layer-wise error analysis.",
    )
    parser.add_argument(
        "--disable-bias-correction",
        action="store_true",
        help="Disable bias correction for SiLU-aware PTQ.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CIFAR-10 if it is not already present.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/silu_aware_ptq.csv"),
        help="CSV result path.",
    )
    parser.add_argument(
        "--thresholds-json",
        type=Path,
        default=Path("results/silu_aware_thresholds.json"),
        help="JSON path for threshold metadata.",
    )
    parser.add_argument(
        "--layer-error-csv",
        type=Path,
        default=Path("results/silu_aware_layer_error.csv"),
        help="CSV path for layer-wise error analysis.",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("reports/silu_aware_ptq_summary.md"),
        help="Markdown report path.",
    )
    return parser.parse_args()


def load_state_dict_file(
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path.resolve()}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("Unsupported checkpoint format.")

    return {
        str(key).removeprefix("module."): value
        for key, value in checkpoint.items()
    }


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    state_dict = load_state_dict_file(checkpoint_path, device)
    incompatible = model.load_state_dict(state_dict, strict=False)

    print(f"Missing keys: {len(incompatible.missing_keys)}")
    print(f"Unexpected keys: {len(incompatible.unexpected_keys)}")

    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("Missing key names:", incompatible.missing_keys)
        print("Unexpected key names:", incompatible.unexpected_keys)
        raise RuntimeError(
            "Checkpoint does not exactly match the ResNet18-SiLU model."
        )


def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> dict[str, float | int]:
    model.eval()

    correct = 0
    total = 0

    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            correct += int((preds == labels).sum().item())
            total += int(labels.numel())

    if total == 0:
        raise RuntimeError("No samples were evaluated.")

    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
        "accuracy_percent": correct / total * 100,
    }


def json_safe_thresholds(thresholds: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for layer_name, value in thresholds.items():
        if isinstance(value, dict):
            result[layer_name] = {
                key: float(item)
                for key, item in value.items()
            }
        else:
            result[layer_name] = float(value)

    return result


def run_method(
    fp32_model: nn.Module,
    calib_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    method: str,
    bits: int,
    device: torch.device,
    max_test_batches: int,
    bias_correction: bool,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    q_model = copy.deepcopy(fp32_model).to(device)
    q_model.eval()

    quantize_weights_per_channel(
        q_model,
        bits=bits,
        use_percentile=True,
    )

    if method == "ncnn":
        thresholds = calibrate_ncnn_style(
            q_model,
            calib_loader,
            device,
            QuantizationConfig(bits=bits, strategy="ncnn"),
        )
        quantized_layers = insert_ncnn_quantizers(
            q_model,
            thresholds,
            bits=bits,
        )
        corrections = {}
    elif method == "silu_aware":
        thresholds = calibrate_silu_aware(
            q_model,
            calib_loader,
            device,
            QuantizationConfig(bits=bits, strategy="silu_aware"),
        )
        quantized_layers = insert_silu_aware_quantizers(
            q_model,
            thresholds,
            bits=bits,
        )

        if bias_correction:
            corrections = apply_bias_correction(
                fp32_model,
                q_model,
                calib_loader,
                device,
                max_batches=1,
            )
        else:
            corrections = {}
    else:
        raise ValueError(f"Unsupported method: {method}")

    accuracy = evaluate_model(
        q_model,
        test_loader,
        device,
        max_batches=max_test_batches,
    )

    metadata = {
        "method": method,
        "bits": bits,
        "num_quantized_layers": len(quantized_layers),
        "num_thresholds": len(thresholds),
        "num_bias_corrections": len(corrections),
        "quantized_layers": list(quantized_layers),
        "thresholds": json_safe_thresholds(thresholds),
    }
    metadata.update(accuracy)

    return q_model, metadata, thresholds


def get_module_by_name(model: nn.Module, name: str) -> nn.Module | None:
    modules = dict(model.named_modules())
    return modules.get(name)


def compute_layer_error(
    fp32_model: nn.Module,
    q_model: nn.Module,
    layer_names: list[str],
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int,
) -> list[dict[str, Any]]:
    fp32_model.eval()
    q_model.eval()

    stats: dict[str, dict[str, float]] = {
        name: {
            "sum_abs": 0.0,
            "sum_sq": 0.0,
            "max_abs": 0.0,
            "count": 0.0,
        }
        for name in layer_names
    }

    fp_outputs: dict[str, torch.Tensor] = {}
    q_outputs: dict[str, torch.Tensor] = {}

    handles = []

    def make_hook(storage: dict[str, torch.Tensor], layer_name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                storage[layer_name] = output.detach().cpu()
        return hook

    for name in layer_names:
        fp_module = get_module_by_name(fp32_model, name)
        q_module = get_module_by_name(q_model, name)

        if fp_module is None or q_module is None:
            continue

        handles.append(fp_module.register_forward_hook(make_hook(fp_outputs, name)))
        handles.append(q_module.register_forward_hook(make_hook(q_outputs, name)))

    with torch.inference_mode():
        for batch_index, (images, _labels) in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break

            fp_outputs.clear()
            q_outputs.clear()

            images = images.to(device)
            fp32_model(images)
            q_model(images)

            for name in layer_names:
                if name not in fp_outputs or name not in q_outputs:
                    continue

                fp_value = fp_outputs[name].float()
                q_value = q_outputs[name].float()

                if fp_value.shape != q_value.shape:
                    continue

                diff = fp_value - q_value
                abs_diff = diff.abs()

                stats[name]["sum_abs"] += float(abs_diff.sum().item())
                stats[name]["sum_sq"] += float((diff * diff).sum().item())
                stats[name]["max_abs"] = max(
                    stats[name]["max_abs"],
                    float(abs_diff.max().item()),
                )
                stats[name]["count"] += float(diff.numel())

    for handle in handles:
        handle.remove()

    rows: list[dict[str, Any]] = []
    for name, item in stats.items():
        count = item["count"]
        if count <= 0:
            continue

        rows.append(
            {
                "layer": name,
                "mae": item["sum_abs"] / count,
                "mse": item["sum_sq"] / count,
                "rmse": (item["sum_sq"] / count) ** 0.5,
                "max_abs_error": item["max_abs"],
                "num_values": int(count),
            }
        )

    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise RuntimeError(f"No rows to write for {path}")

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    result_rows: list[dict[str, Any]],
    layer_error_rows: list[dict[str, Any]],
    fp32_accuracy_percent: float,
    calibration_samples: int,
    bits: int,
    bias_correction: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    result_table = []
    for row in result_rows:
        result_table.append(
            [
                str(row["method"]),
                str(row["bits"]),
                f"{float(row['accuracy_percent']):.4f}%",
                f"{fp32_accuracy_percent - float(row['accuracy_percent']):.4f} pp",
                str(row["num_quantized_layers"]),
                str(row["num_thresholds"]),
                str(row["num_bias_corrections"]),
            ]
        )

    top_layer_rows = sorted(
        layer_error_rows,
        key=lambda item: float(item["mse"]),
        reverse=True,
    )[:10]

    layer_table = []
    for row in top_layer_rows:
        layer_table.append(
            [
                str(row["layer"]),
                f"{float(row['mae']):.8f}",
                f"{float(row['mse']):.8f}",
                f"{float(row['rmse']):.8f}",
                f"{float(row['max_abs_error']):.8f}",
            ]
        )

    lines = [
        "# SiLU-aware PTQ Summary",
        "",
        "## Configuration",
        "",
        markdown_table(
            ["Item", "Value"],
            [
                ["Model", "ResNet18-SiLU"],
                ["Dataset", "CIFAR-10"],
                ["Bits", str(bits)],
                ["Calibration samples", str(calibration_samples)],
                ["Bias correction", "enabled" if bias_correction else "disabled"],
                ["Execution", "PyTorch-side quantization simulation"],
            ],
        ),
        "",
        "## Accuracy Results",
        "",
        markdown_table(
            [
                "Method",
                "Bits",
                "Accuracy",
                "Drop vs FP32",
                "Quantized layers",
                "Thresholds",
                "Bias corrections",
            ],
            result_table,
        ),
        "",
        "## Layer-wise Error Analysis",
        "",
        "The table below reports the top layers by MSE between FP32 SiLU activations and quantized activations.",
        "",
        markdown_table(
            ["Layer", "MAE", "MSE", "RMSE", "Max abs error"],
            layer_table,
        ),
        "",
        "## Notes",
        "",
        "- This v0.9 stage integrates the custom SiLU-aware PTQ algorithm into the benchmark repository.",
        "- The current output is a PyTorch-side quantization simulation rather than a standard ONNX QDQ graph.",
        "- The SiLU-aware method uses KLD-based Vmax search, MSE-based vsplit selection, piecewise asymmetric activation quantization, per-channel weight quantization, and optional bias correction.",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable.")

    device = torch.device(args.device)

    config = ExperimentConfig(
        batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        calib_batches=args.calibration_batches,
        data_root=args.data_root,
        weights_path=str(args.checkpoint),
    )

    _, test_set, calibration_set, calibration_loader, test_loader = (
        build_cifar10_loaders(
            config=config,
            download=args.download,
        )
    )

    fp32_model = ResNet18().to(device)
    fp32_model.eval()
    load_checkpoint(fp32_model, args.checkpoint, device)

    print("=" * 60)
    print("SiLU-aware PTQ")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset size: {len(test_set)}")
    print(f"Calibration subset size: {len(calibration_set)}")
    print(f"Methods: {args.methods}")
    print(f"Bits: {args.bits}")
    print(f"Bias correction: {not args.disable_bias_correction}")

    fp32_result = evaluate_model(
        fp32_model,
        test_loader,
        device,
        max_batches=args.max_test_batches,
    )

    print("-" * 60)
    print(f"FP32 accuracy: {fp32_result['accuracy_percent']:.4f}%")

    result_rows: list[dict[str, Any]] = []
    thresholds_payload: dict[str, Any] = {}

    q_models: dict[str, nn.Module] = {}

    for method in args.methods:
        print("-" * 60)
        print(f"Running method: {method}")

        q_model, metadata, thresholds = run_method(
            fp32_model=fp32_model,
            calib_loader=calibration_loader,
            test_loader=test_loader,
            method=method,
            bits=args.bits,
            device=device,
            max_test_batches=args.max_test_batches,
            bias_correction=not args.disable_bias_correction,
        )

        q_models[method] = q_model

        row = {
            "method": method,
            "bits": args.bits,
            "calibration_samples": len(calibration_set),
            "correct": metadata["correct"],
            "total": metadata["total"],
            "accuracy": metadata["accuracy"],
            "accuracy_percent": metadata["accuracy_percent"],
            "accuracy_drop_pp": fp32_result["accuracy_percent"]
            - metadata["accuracy_percent"],
            "num_quantized_layers": metadata["num_quantized_layers"],
            "num_thresholds": metadata["num_thresholds"],
            "num_bias_corrections": metadata["num_bias_corrections"],
        }
        result_rows.append(row)

        thresholds_payload[method] = metadata["thresholds"]

        print(f"Accuracy: {metadata['accuracy_percent']:.4f}%")
        print(f"Accuracy drop: {row['accuracy_drop_pp']:.4f} pp")
        print(f"Quantized layers: {metadata['num_quantized_layers']}")
        print(f"Thresholds: {metadata['num_thresholds']}")
        print(f"Bias corrections: {metadata['num_bias_corrections']}")

    layer_error_rows: list[dict[str, Any]] = []
    if "silu_aware" in q_models:
        print("-" * 60)
        print("Running layer-wise error analysis for SiLU-aware PTQ")

        silu_thresholds = thresholds_payload.get("silu_aware", {})
        layer_names = list(silu_thresholds.keys())

        layer_error_rows = compute_layer_error(
            fp32_model=fp32_model,
            q_model=q_models["silu_aware"],
            layer_names=layer_names,
            loader=test_loader,
            device=device,
            max_batches=args.layer_error_batches,
        )

        print(f"Layer error rows: {len(layer_error_rows)}")

    write_csv(result_rows, args.output_csv)
    write_json(thresholds_payload, args.thresholds_json)

    if layer_error_rows:
        write_csv(layer_error_rows, args.layer_error_csv)

    write_report(
        path=args.report_md,
        result_rows=result_rows,
        layer_error_rows=layer_error_rows,
        fp32_accuracy_percent=float(fp32_result["accuracy_percent"]),
        calibration_samples=len(calibration_set),
        bits=args.bits,
        bias_correction=not args.disable_bias_correction,
    )

    print("=" * 60)
    print("SiLU-aware PTQ completed")
    print("=" * 60)
    print(f"Results CSV:       {args.output_csv}")
    print(f"Thresholds JSON:   {args.thresholds_json}")
    print(f"Layer error CSV:   {args.layer_error_csv}")
    print(f"Markdown report:   {args.report_md}")


if __name__ == "__main__":
    main()
