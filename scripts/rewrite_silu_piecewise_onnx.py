"""Rewrite proven exported SiLU patterns into the ONNX piecewise reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from silu_benchmark.backends import (
    inspect_onnx_model,
    load_site_spec_manifest,
    rewrite_silu_piecewise_model,
    save_rewrite_result,
    write_inspection_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite ResNet18-SiLU ONNX with piecewise reference subgraphs.")
    parser.add_argument("--input", type=Path, default=Path("artifacts/onnx/resnet18_silu_fp32.onnx"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/onnx/resnet18_silu_piecewise_reference.onnx"))
    parser.add_argument("--spec-manifest", type=Path, required=True, help="JSON with one valid spec for every exported site.")
    parser.add_argument("--report", type=Path, default=Path("results/silu_piecewise_onnx_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = onnx.load(str(args.input))
    result = rewrite_silu_piecewise_model(baseline, load_site_spec_manifest(args.spec_manifest))
    save_rewrite_result(result, args.output)
    report = inspect_onnx_model(args.output)
    write_inspection_report(report, args.report)
    print(f"Rewritten sites: {len(result.sites)}")
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
