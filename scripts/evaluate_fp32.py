from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Add the project root to Python's import path so that
# "hku_silu_ptq" can be imported when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from silu_benchmark.config import ExperimentConfig
from silu_benchmark.data import build_cifar10_loaders
from silu_benchmark.models import ResNet18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the FP32 ResNet18-SiLU checkpoint on CIFAR-10."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/resnet18_cifar10.pth"),
        help="Path to the FP32 checkpoint.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
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
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CIFAR-10 if it is not already present.",
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

    # Support either a raw state_dict or a wrapped training checkpoint.
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


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[int, int, float]:
    model.eval()

    correct = 0
    total = 0
    first_input_shape = None
    first_output_shape = None

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            if first_input_shape is None:
                first_input_shape = tuple(images.shape)
                first_output_shape = tuple(logits.shape)

            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise RuntimeError("The CIFAR-10 test loader returned no samples.")

    accuracy = correct / total

    print(f"Input shape: {first_input_shape}")
    print(f"Output shape: {first_output_shape}")

    return correct, total, accuracy


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False."
        )

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

    correct, total, accuracy = evaluate(
        model=model,
        loader=test_loader,
        device=device,
    )

    print("=" * 60)
    print("FP32 evaluation")
    print("=" * 60)
    print("Model: ResNet18-SiLU")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset size: {len(test_set)}")
    print(f"Correct: {correct}")
    print(f"Total: {total}")
    print(f"Accuracy: {accuracy * 100:.4f}%")


if __name__ == "__main__":
    main()
