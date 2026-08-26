"""Write a JSON graph/capability report for an ONNX model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from silu_benchmark.backends import inspect_onnx_model, write_inspection_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an ONNX model on CPUExecutionProvider.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    write_inspection_report(inspect_onnx_model(args.model), args.report)
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
