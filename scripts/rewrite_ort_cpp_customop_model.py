#!/usr/bin/env python
"""Rewrite the frozen v1.1 candidate to the v1.2 custom-node contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from silu_benchmark.ort_custom_op_backend import detect_prerequisites, load_config
from silu_benchmark.ort_custom_op_rewrite import (
    build_sidecar_manifest,
    rewrite_selected_candidate,
    sha256_file,
    validate_generated_artifact,
    validate_generated_output_path,
    verify_selected_source,
)


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmarks/resnet18_silu_cifar10_v12_ort_customop_cpu.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = (ROOT / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    source_path = (ROOT / config["selected_model"]).resolve()
    receipt_path = (ROOT / config["selection_receipt"]).resolve()
    artifact_root = validate_generated_output_path(
        ROOT / config["generated_artifact_root"], ROOT
    )
    report_root = validate_generated_output_path(
        ROOT / config["generated_report_root"], ROOT
    )
    if artifact_root.exists() or report_root.exists():
        raise FileExistsError("refusing to overwrite existing v1.2 generated output")
    source, receipt = verify_selected_source(
        source_path,
        receipt_path,
        expected_model_sha256=config["selected_model_sha256"],
    )
    rewritten, islands, graph_report = rewrite_selected_candidate(
        source,
        source_model_sha256=sha256_file(source_path),
        selection_receipt_hash=receipt["selection_receipt_hash"],
    )
    artifact_root.mkdir(parents=True)
    report_root.mkdir(parents=True)
    model_path = artifact_root / "resnet18_silu_v12_ort_cpp_customop.onnx"
    manifest_path = artifact_root / "resnet18_silu_v12_ort_cpp_customop.manifest.json"
    model_path.write_bytes(rewritten.SerializeToString(deterministic=True))
    manifest = build_sidecar_manifest(
        source_path=source_path,
        rewritten_path=model_path,
        selection_receipt_path=receipt_path,
        islands=islands,
        graph_report=graph_report,
    )
    atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validate_generated_artifact(
        source_path=source_path,
        rewritten_path=model_path,
        sidecar_path=manifest_path,
        selection_receipt_path=receipt_path,
    )
    prerequisites = detect_prerequisites().to_dict()
    report = {
        "completion_status": "graph_rewrite_only",
        "source_model": str(source_path),
        "source_model_sha256": sha256_file(source_path),
        "rewritten_model": str(model_path),
        "rewritten_model_sha256": sha256_file(model_path),
        "sidecar_manifest": str(manifest_path),
        "sidecar_manifest_hash": manifest["manifest_hash"],
        "graph_report": graph_report,
        "custom_op_build_prerequisites": prerequisites,
        "dll_built": False,
        "ort_session_run": False,
        "probe_parity_run": False,
        "full_test_run": False,
        "benchmark_run": False,
        "blocker": prerequisites["blockers"],
    }
    atomic_text(
        report_root / "graph_rewrite_report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    markdown = "\n".join(
        [
            "# v1.2 custom-op graph rewrite report",
            "",
            "Status: **graph rewrite only; build and execution are separate steps**.",
            "",
            f"- Source SHA-256: `{report['source_model_sha256']}`",
            f"- Rewritten SHA-256: `{report['rewritten_model_sha256']}`",
            f"- Selected islands removed: `{graph_report['source_piecewise_islands_removed']}`",
            f"- Source island nodes removed: `{graph_report['source_piecewise_nodes_removed']}`",
            f"- Custom nodes added: `{graph_report['custom_nodes_added']}`",
            f"- Original selected-piecewise nodes retained: `{graph_report['original_selected_piecewise_nodes_retained']}`",
            "",
            "This rewrite command does not build the DLL or run ORT, parity, evaluation, or timing.",
            "",
            "Prerequisite blockers: " + (
                "; ".join(prerequisites["blockers"])
                if prerequisites["blockers"]
                else "none detected"
            ),
            "",
            prerequisites["recommendation"],
        ]
    )
    atomic_text(report_root / "graph_rewrite_report.md", markdown + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
