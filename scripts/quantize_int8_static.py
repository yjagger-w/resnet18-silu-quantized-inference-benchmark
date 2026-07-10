from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
            self.input_name: images.detach().cpu().numpy().astype("float32")
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply ONNX Runtime static INT8 PTQ to ResNet18-SiLU."
    )
    parser.add_argument(
        "--model-input",
        type=Path,
        default=Path("artifacts/onnx/resnet18_silu_fp32.onnx"),
        help="Input FP32 ONNX model.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("artifacts/int8/resnet18_silu_int8.onnx"),
        help="Output INT8 ONNX model.",
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
        "--calibration-batch-size",
        type=int,
        default=128,
        help="Calibration batch size.",
    )
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=20,
        help="Number of calibration batches. 20 x 128 = 2560 samples.",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["MinMax", "Entropy", "Percentile"],
        default="MinMax",
        help="ONNX Runtime calibration method.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CIFAR-10 if it is not already present.",
    )
    return parser.parse_args()


def get_calibration_method(name: str) -> CalibrationMethod:
    mapping = {
        "MinMax": CalibrationMethod.MinMax,
        "Entropy": CalibrationMethod.Entropy,
        "Percentile": CalibrationMethod.Percentile,
    }
    return mapping[name]


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def main() -> None:
    args = parse_args()

    if not args.model_input.exists():
        raise FileNotFoundError(
            f"Input ONNX model not found: {args.model_input.resolve()}"
        )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig(
        batch_size=args.calibration_batch_size,
        test_batch_size=args.calibration_batch_size,
        calib_batches=args.calibration_batches,
        data_root=args.data_root,
        weights_path="",
    )

    _, _, calibration_set, calibration_loader, _ = build_cifar10_loaders(
        config=config,
        download=args.download,
    )

    expected_samples = args.calibration_batch_size * args.calibration_batches
    actual_samples = len(calibration_set)

    print("=" * 60)
    print("ONNX Runtime Static INT8 PTQ")
    print("=" * 60)
    print(f"Input model:  {args.model_input}")
    print(f"Output model: {args.model_output}")
    print(f"Calibration method: {args.calibration_method}")
    print("Quantization format: QDQ")
    print("Activation type: QUInt8")
    print("Weight type: QInt8")
    print("Weight granularity: per-channel")
    print(f"Calibration batch size: {args.calibration_batch_size}")
    print(f"Calibration batches: {args.calibration_batches}")
    print(f"Expected calibration samples: {expected_samples}")
    print(f"Actual calibration subset size: {actual_samples}")

    calibration_reader = Cifar10CalibrationDataReader(
        data_loader=calibration_loader,
        input_name=args.input_name,
        max_batches=args.calibration_batches,
    )

    quantize_static(
        model_input=str(args.model_input),
        model_output=str(args.model_output),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        calibrate_method=get_calibration_method(args.calibration_method),
    )

    print("=" * 60)
    print("INT8 quantization completed")
    print("=" * 60)
    print(f"Output model: {args.model_output}")
    print(f"FP32 model size: {file_size_mb(args.model_input):.2f} MB")
    print(f"INT8 model size: {file_size_mb(args.model_output):.2f} MB")


if __name__ == "__main__":
    main()
