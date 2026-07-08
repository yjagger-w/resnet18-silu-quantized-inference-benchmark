from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from silu_benchmark.config import ExperimentConfig
from silu_benchmark.data import build_cifar10_loaders
from silu_benchmark.models import ResNet18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate numerical consistency and accuracy between PyTorch FP32 "
            "and ONNX Runtime FP32 for ResNet18-SiLU."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/resnet18_cifar10.pth"),
        help="Path to the FP32 PyTorch checkpoint.",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("artifacts/onnx/resnet18_silu_fp32.onnx"),
        help="Path to the exported ONNX model.",
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
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device used for PyTorch inference.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CIFAR-10 if it is not already present.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Optional limit for debugging. Use 0 for the full test set.",
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
        raise TypeError(
            "Unsupported checkpoint format. Expected a state_dict or a "
            "dictionary containing state_dict/model_state_dict/model."
        )

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

    if incompatible.missing_keys:
        print("Missing key names:", incompatible.missing_keys)

    if incompatible.unexpected_keys:
        print("Unexpected key names:", incompatible.unexpected_keys)

    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint does not exactly match the ResNet18-SiLU model."
        )


def create_ort_session(onnx_path: Path) -> ort.InferenceSession:
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path.resolve()}")

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    print("ONNX Runtime providers:", session.get_providers())
    print("ONNX input name:", session.get_inputs()[0].name)
    print("ONNX output name:", session.get_outputs()[0].name)

    return session


def validate(
    model: torch.nn.Module,
    session: ort.InferenceSession,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int,
) -> dict[str, float | int | tuple[int, ...]]:
    model.eval()

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    max_abs_diff = 0.0
    abs_diff_sum = 0.0
    abs_diff_count = 0

    torch_correct = 0
    ort_correct = 0
    agreement = 0
    total = 0

    first_input_shape = None
    first_torch_output_shape = None
    first_ort_output_shape = None

    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break

            images_on_device = images.to(device)
            labels_on_device = labels.to(device)

            torch_logits = model(images_on_device)
            torch_logits_np = torch_logits.detach().cpu().numpy().astype(np.float32)

            ort_inputs = {input_name: images.detach().cpu().numpy().astype(np.float32)}
            ort_logits_np = session.run([output_name], ort_inputs)[0].astype(np.float32)

            if first_input_shape is None:
                first_input_shape = tuple(images.shape)
                first_torch_output_shape = tuple(torch_logits_np.shape)
                first_ort_output_shape = tuple(ort_logits_np.shape)

            diff = np.abs(torch_logits_np - ort_logits_np)
            max_abs_diff = max(max_abs_diff, float(diff.max()))
            abs_diff_sum += float(diff.sum())
            abs_diff_count += int(diff.size)

            torch_pred = torch_logits_np.argmax(axis=1)
            ort_pred = ort_logits_np.argmax(axis=1)
            labels_np = labels_on_device.detach().cpu().numpy()

            torch_correct += int((torch_pred == labels_np).sum())
            ort_correct += int((ort_pred == labels_np).sum())
            agreement += int((torch_pred == ort_pred).sum())
            total += int(labels_np.shape[0])

    if total == 0 or abs_diff_count == 0:
        raise RuntimeError("No samples were evaluated.")

    return {
        "first_input_shape": first_input_shape,
        "first_torch_output_shape": first_torch_output_shape,
        "first_ort_output_shape": first_ort_output_shape,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": abs_diff_sum / abs_diff_count,
        "torch_correct": torch_correct,
        "ort_correct": ort_correct,
        "agreement_count": agreement,
        "total": total,
        "torch_accuracy": torch_correct / total,
        "ort_accuracy": ort_correct / total,
        "prediction_agreement": agreement / total,
    }


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable.")

    device = torch.device(args.device)

    config = ExperimentConfig(
        batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        data_root=args.data_root,
        weights_path=str(args.checkpoint),
    )

    _, test_set, _, _, test_loader = build_cifar10_loaders(
        config=config,
        download=args.download,
    )

    model = ResNet18().to(device)

    load_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    session = create_ort_session(args.onnx)

    metrics = validate(
        model=model,
        session=session,
        loader=test_loader,
        device=device,
        max_batches=args.max_batches,
    )

    print("=" * 60)
    print("PyTorch vs ONNX Runtime FP32 validation")
    print("=" * 60)
    print(f"PyTorch device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"ONNX model: {args.onnx}")
    print(f"Dataset size: {len(test_set)}")
    print(f"Evaluated samples: {metrics['total']}")
    print(f"Input shape: {metrics['first_input_shape']}")
    print(f"PyTorch output shape: {metrics['first_torch_output_shape']}")
    print(f"ONNX output shape: {metrics['first_ort_output_shape']}")
    print(f"Maximum absolute difference: {metrics['max_abs_diff']:.8f}")
    print(f"Mean absolute difference: {metrics['mean_abs_diff']:.8f}")
    print(f"PyTorch correct: {metrics['torch_correct']}")
    print(f"ONNX Runtime correct: {metrics['ort_correct']}")
    print(f"PyTorch accuracy: {metrics['torch_accuracy'] * 100:.4f}%")
    print(f"ONNX Runtime accuracy: {metrics['ort_accuracy'] * 100:.4f}%")
    print(f"Prediction agreement: {metrics['prediction_agreement'] * 100:.4f}%")
    print(f"Agreement count: {metrics['agreement_count']} / {metrics['total']}")


if __name__ == "__main__":
    main()
