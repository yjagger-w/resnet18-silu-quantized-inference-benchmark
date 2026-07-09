from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from silu_benchmark.models import ResNet18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ResNet18-SiLU FP32 checkpoint to ONNX."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/resnet18_cifar10.pth"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/onnx/resnet18_silu_fp32.onnx"),
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
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


def export_onnx(
    model: torch.nn.Module,
    output_path: Path,
    dummy_input: torch.Tensor,
    opset: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use the legacy exporter here because it correctly preserves the
    # dynamic batch dimension for both input and output in this project.
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    print("Export method: legacy torch.onnx.export")


def check_onnx_model(output_path: Path) -> None:
    import onnx

    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)

    print("ONNX checker: passed")

    graph = model.graph
    print("ONNX inputs:")
    for item in graph.input:
        print(f"  - {item.name}")

    print("ONNX outputs:")
    for item in graph.output:
        print(f"  - {item.name}")


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable.")

    device = torch.device(args.device)

    model = ResNet18().to(device)
    model.eval()

    load_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    dummy_input = torch.randn(1, 3, 32, 32, device=device)

    with torch.inference_mode():
        dummy_output = model(dummy_input)

    print(f"Dummy input shape: {tuple(dummy_input.shape)}")
    print(f"Dummy output shape: {tuple(dummy_output.shape)}")

    export_onnx(
        model=model,
        output_path=args.output,
        dummy_input=dummy_input,
        opset=args.opset,
    )

    check_onnx_model(args.output)

    size_mb = args.output.stat().st_size / 1024 / 1024

    print("=" * 60)
    print("ONNX export completed")
    print("=" * 60)
    print(f"Output path: {args.output}")
    print(f"ONNX size: {size_mb:.2f} MB")
    print(f"Opset: {args.opset}")


if __name__ == "__main__":
    main()
