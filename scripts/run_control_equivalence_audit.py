#!/usr/bin/env python
"""Run the generated-only v1.1.1 strict control-equivalence audit."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from silu_benchmark.accuracy_recovery import discover_qdq_silu_sites
from silu_benchmark.benchmark_data import load_cifar_batch, numpy_batches
from silu_benchmark.control_equivalence import (
    AUDIT_SCHEMA,
    TARGET_STAGES,
    build_strict_topology_control,
    compare_qdq_topology,
    graph_topology_hash,
    instrument_model_outputs,
    original_stage_mapping,
    protobuf_hash,
    resolve_original_stage_mapping,
    rewritten_stage_mapping,
    selected_candidate_coverage,
    sha256_file,
    target_topology_audit,
    validate_audit_output_path,
    validate_control_label,
)


QDQ_PATH = ROOT / "artifacts/int8/resnet18_silu_int8_v065.onnx"
NOOP_PATH = ROOT / "artifacts/accuracy_recovery/v1.1/controls/topology_preserving_noop.onnx"
SEMANTIC_PATH = ROOT / "artifacts/accuracy_recovery/v1.1/controls/float_equivalent_silu.onnx"
SELECTED_PATH = ROOT / "artifacts/accuracy_recovery/v1.1/candidates/two_segment_p99_99_mse.onnx"
CONTROLS_REPORT = ROOT / "results/benchmarks/v1.1_accuracy_diagnosis/controls.json"
SELECTION_RECEIPT = ROOT / "results/benchmarks/v1.1_accuracy_recovery_smoke/selection_receipt.json"
FINAL_REPORT = ROOT / "results/benchmarks/v1.1_accuracy_recovery/final_comparison.json"


class PairMetrics:
    def __init__(self):
        self.count = 0
        self.samples = 0
        self.trailing_shape = None
        self.reference_dtype = None
        self.candidate_dtype = None
        self.exact = True
        self.max_abs = 0.0
        self.sum_abs = 0.0
        self.sum_square = 0.0
        self.dot = 0.0
        self.reference_norm = 0.0
        self.candidate_norm = 0.0

    def update(self, reference, candidate):
        left = np.asarray(reference)
        right = np.asarray(candidate)
        if left.shape != right.shape:
            raise ValueError(f"instrumented tensor shape mismatch: {left.shape} != {right.shape}")
        if self.trailing_shape is None:
            self.trailing_shape = list(left.shape[1:])
            self.reference_dtype = str(left.dtype)
            self.candidate_dtype = str(right.dtype)
        self.samples += int(left.shape[0])
        self.exact = self.exact and np.array_equal(left, right)
        left64 = left.astype(np.float64, copy=False).reshape(-1)
        right64 = right.astype(np.float64, copy=False).reshape(-1)
        difference = right64 - left64
        self.count += difference.size
        if difference.size:
            self.max_abs = max(self.max_abs, float(np.max(np.abs(difference))))
            self.sum_abs += float(np.sum(np.abs(difference), dtype=np.float64))
            self.sum_square += float(np.dot(difference, difference))
            self.dot += float(np.dot(left64, right64))
            self.reference_norm += float(np.dot(left64, left64))
            self.candidate_norm += float(np.dot(right64, right64))

    def result(self) -> dict:
        denominator = math.sqrt(self.reference_norm * self.candidate_norm)
        cosine = self.dot / denominator if denominator else (1.0 if self.exact else 0.0)
        return {
            "shape": [self.samples, *(self.trailing_shape or [])],
            "reference_dtype": self.reference_dtype,
            "candidate_dtype": self.candidate_dtype,
            "exact_equal": self.exact,
            "max_absolute_error": self.max_abs,
            "mae": self.sum_abs / self.count if self.count else 0.0,
            "mse": self.sum_square / self.count if self.count else 0.0,
            "cosine_similarity": cosine,
        }


class ValueStats:
    def __init__(self):
        self.samples = 0
        self.trailing_shape = None
        self.dtype = None
        self.minimum = None
        self.maximum = None
        self.codes = set()

    def update(self, value):
        array = np.asarray(value)
        if self.trailing_shape is None:
            self.trailing_shape = list(array.shape[1:])
            self.dtype = str(array.dtype)
        self.samples += int(array.shape[0])
        current_min = int(np.min(array))
        current_max = int(np.max(array))
        self.minimum = current_min if self.minimum is None else min(self.minimum, current_min)
        self.maximum = current_max if self.maximum is None else max(self.maximum, current_max)
        self.codes.update(int(item) for item in np.unique(array))

    def result(self):
        return {
            "shape": [self.samples, *(self.trailing_shape or [])],
            "dtype": self.dtype,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "occupied_code_count": len(self.codes),
            "occupied_codes": sorted(self.codes),
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmarks/v1.1_control_equivalence_audit"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/accuracy_recovery/v1.1.1"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def output_aliases(model, mapping):
    instrumented, names = instrument_model_outputs(model, mapping)
    session = ort.InferenceSession(
        instrumented.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    observed = [item.name for item in session.get_outputs()]
    if observed != list(names):
        raise RuntimeError("ORT changed deterministic instrumented output order")
    positions = {name: index for index, name in enumerate(names)}
    aliases = {logical: positions[tensor] for logical, tensor in mapping.items()}
    return session, aliases


def compare_instrumented(
    reference_model,
    candidate_model,
    reference_mapping,
    candidate_mapping,
    images,
    labels,
    indices,
    batch_size,
):
    reference_session, reference_aliases = output_aliases(reference_model, reference_mapping)
    candidate_session, candidate_aliases = output_aliases(candidate_model, candidate_mapping)
    common = [name for name in reference_mapping if name in candidate_mapping]
    metrics = {name: PairMetrics() for name in common}
    candidate_only = {
        name: ValueStats() for name in candidate_mapping
        if name not in reference_mapping and name.endswith("::piecewise_code")
    }
    input_name = reference_session.get_inputs()[0].name
    if candidate_session.get_inputs()[0].name != input_name:
        raise RuntimeError("instrumented model input names differ")
    reference_predictions = []
    candidate_predictions = []
    observed_labels = []
    for batch, batch_labels in numpy_batches(images, labels, indices, batch_size):
        left = reference_session.run(None, {input_name: batch})
        right = candidate_session.run(None, {input_name: batch})
        for name in common:
            metrics[name].update(left[reference_aliases[name]], right[candidate_aliases[name]])
        for name, accumulator in candidate_only.items():
            accumulator.update(right[candidate_aliases[name]])
        reference_predictions.append(
            np.asarray(left[reference_aliases["final_logits"]]).argmax(axis=1)
        )
        candidate_predictions.append(
            np.asarray(right[candidate_aliases["final_logits"]]).argmax(axis=1)
        )
        observed_labels.append(batch_labels)
    reference_predictions = np.concatenate(reference_predictions)
    candidate_predictions = np.concatenate(candidate_predictions)
    observed_labels = np.concatenate(observed_labels)
    rows = []
    for logical_name in common:
        if logical_name == "final_logits":
            continue
        site_id, stage = logical_name.split("::", 1)
        rows.append({"site_id": site_id, "stage": stage, **metrics[logical_name].result()})
    ordered_site_ids = []
    for logical_name in reference_mapping:
        if "::" not in logical_name:
            continue
        site_id = logical_name.split("::", 1)[0]
        if site_id not in ordered_site_ids:
            ordered_site_ids.append(site_id)
    first_divergence = next(
        (
            f"{site_id}::{stage}"
            for site_id in ordered_site_ids
            for stage in TARGET_STAGES
            if f"{site_id}::{stage}" in metrics
            and not metrics[f"{site_id}::{stage}"].result()["exact_equal"]
        ),
        None,
    )
    return {
        "probe_samples": len(indices),
        "target_tensor_rows": rows,
        "final_logits": metrics["final_logits"].result(),
        "prediction_agreement": float(np.mean(reference_predictions == candidate_predictions)),
        "reference_accuracy": float(np.mean(reference_predictions == observed_labels)),
        "candidate_accuracy": float(np.mean(candidate_predictions == observed_labels)),
        "first_divergent_target_tensor": first_divergence,
        "candidate_only_tensor_statistics": {
            name: value.result() for name, value in candidate_only.items()
        },
    }


def atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    os.replace(temporary, path)


def write_reports(output: Path, payload: dict):
    atomic_write(output / "control_equivalence_audit.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    strict = payload["strict_control"]["probe_comparison"]
    semantic = payload["semantic_expression_control"]["probe_comparison"]
    selected = payload["selected_candidate"]["probe_comparison"]
    lines = [
        "# v1.1.1 control-equivalence audit",
        "",
        "This is a 128-image ORT CPU control audit. It does not rerun or replace the frozen 10,000-image v1.1 result.",
        "",
        "## Strict control",
        "",
        f"- Graph topology exact: **{payload['strict_control']['graph_topology_exact']}**",
        f"- Q/DQ topology exact: **{payload['strict_control']['qdq_topology']['exact_qdq_topology']}**",
        f"- All target tensors, output codes, and logits exact: **{payload['strict_control']['all_instrumented_tensors_exact']}**",
        f"- Final-logit max absolute error: `{strict['final_logits']['max_absolute_error']:.8g}`",
        f"- Prediction agreement: `{strict['prediction_agreement']:.4%}`",
        "",
        "## Legacy semantic-expression control",
        "",
        "The v1.1 artifact ID `float_equivalent_silu` is retained only as immutable provenance. It is renamed semantically here because it removes two Q/DQ pairs at each target.",
        "",
        f"- Removed Q/DQ nodes: `{len(payload['semantic_expression_control']['qdq_topology']['removed'])}`",
        f"- First divergent target tensor: `{semantic['first_divergent_target_tensor']}`",
        f"- Final-logit max absolute error: `{semantic['final_logits']['max_absolute_error']:.8g}`",
        f"- Final-logit MAE: `{semantic['final_logits']['mae']:.8g}`",
        f"- Prediction agreement: `{semantic['prediction_agreement']:.4%}`",
        "- Graph reason: raw Sigmoid output feeds Mul directly; the original Sigmoid Q/DQ and post-Mul Q/DQ boundaries are removed.",
        "",
        "## Selected candidate",
        "",
        f"- Proven piecewise target coverage: `{payload['selected_candidate']['coverage']['covered_site_count']}/17`",
        f"- First intended divergent tensor: `{selected['first_divergent_target_tensor']}`",
        f"- Probe prediction agreement vs Standard-QDQ: `{selected['prediction_agreement']:.4%}`",
        f"- Frozen model SHA-256 unchanged: **{payload['selected_candidate']['frozen_model_hash_valid']}**",
        f"- Existing 93.58% full-test result remains a valid measurement of that exact model: **{payload['selected_candidate']['frozen_final_result_valid']}**",
        "",
        "## Revised conclusion",
        "",
        payload["revised_root_cause_statement"],
        "",
        "This audit makes no custom-op, integer-only, accelerator, QNN/NPU/GPU, OpenVINO, or whole-model C++ claim.",
    ]
    atomic_write(output / "control_equivalence_audit.md", "\n".join(lines) + "\n")


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if "torch" in sys.modules or "torchvision" in sys.modules:
        raise RuntimeError("ORT-only v1.1.1 audit must not import torch or torchvision")
    output = validate_audit_output_path(ROOT / args.output, ROOT)
    artifact_root = validate_audit_output_path(ROOT / args.artifact_root, ROOT, artifact=True)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite an existing v1.1.1 audit report")
    strict_name = "strict_operator_topology_preserving_control.onnx"
    if artifact_root.exists():
        unexpected = [path for path in artifact_root.iterdir() if path.name != strict_name]
        if unexpected:
            raise FileExistsError(f"unexpected existing v1.1.1 artifacts: {unexpected}")
    required = [
        QDQ_PATH, NOOP_PATH, SEMANTIC_PATH, SELECTED_PATH,
        CONTROLS_REPORT, SELECTION_RECEIPT, FINAL_REPORT,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required frozen v1.1 assets are missing: {missing}")

    controls_report = json.loads(CONTROLS_REPORT.read_text(encoding="utf-8"))
    receipt = json.loads(SELECTION_RECEIPT.read_text(encoding="utf-8"))
    final_report = json.loads(FINAL_REPORT.read_text(encoding="utf-8"))
    if sha256_file(SELECTED_PATH) != receipt["model_sha256"]:
        raise ValueError("selected candidate hash differs from frozen receipt")
    selected_row = next(
        row for row in final_report["rows"] if row["role"] == "selected v1.1 candidate"
    )
    if selected_row["top1_accuracy"] != 0.9358:
        raise ValueError("frozen final report no longer records 93.58%")

    source = onnx.load(str(QDQ_PATH))
    noop = onnx.load(str(NOOP_PATH))
    semantic = onnx.load(str(SEMANTIC_PATH))
    selected = onnx.load(str(SELECTED_PATH))
    sites = discover_qdq_silu_sites(source)
    strict, strict_sites = build_strict_topology_control(source)
    if [site.site_id for site in strict_sites] != [site.site_id for site in sites]:
        raise RuntimeError("strict-control discovery order changed")
    artifact_root.mkdir(parents=True, exist_ok=True)
    strict_path = artifact_root / strict_name
    if strict_path.exists():
        existing_strict = onnx.load(str(strict_path))
        if graph_topology_hash(existing_strict) != graph_topology_hash(strict):
            raise ValueError("existing partial strict-control artifact has different topology")
    else:
        onnx.save(strict, str(strict_path))

    qdq = {
        "v11_noop": compare_qdq_topology(source, noop),
        "strict_control": compare_qdq_topology(source, strict),
        "semantic_expression": compare_qdq_topology(source, semantic),
        "selected_candidate": compare_qdq_topology(source, selected),
    }
    label_rejection = None
    try:
        validate_control_label("float-equivalent", source, semantic)
    except ValueError as error:
        label_rejection = str(error)
    if label_rejection is None:
        raise RuntimeError("non-topology-preserving control accepted an invalid equivalence label")
    validate_control_label("semantic-expression control", source, semantic)

    source_mapping = resolve_original_stage_mapping(source, sites)
    strict_mapping = resolve_original_stage_mapping(strict, sites)
    semantic_mapping = rewritten_stage_mapping(semantic, sites, mode="semantic_expression")
    selected_mapping = rewritten_stage_mapping(selected, sites, mode="piecewise")
    probe_indices = controls_report["probe_indices"]
    if len(probe_indices) != 128:
        raise ValueError("frozen v1.1 probe does not contain 128 images")
    images, labels = load_cifar_batch(ROOT / "data", "data_batch_1")
    strict_probe = compare_instrumented(
        source, strict, source_mapping, strict_mapping,
        images, labels, probe_indices, args.batch_size,
    )
    semantic_probe = compare_instrumented(
        source, semantic, original_stage_mapping(sites), semantic_mapping,
        images, labels, probe_indices, args.batch_size,
    )
    selected_probe = compare_instrumented(
        source, selected, original_stage_mapping(sites), selected_mapping,
        images, labels, probe_indices, args.batch_size,
    )
    strict_exact = all(
        row["exact_equal"] for row in strict_probe["target_tensor_rows"]
    ) and strict_probe["final_logits"]["exact_equal"]
    if not strict_exact:
        raise RuntimeError(
            f"strict topology control diverged at {strict_probe['first_divergent_target_tensor']}"
        )
    coverage = selected_candidate_coverage(source, selected, sites)
    if not coverage["all_17_piecewise_paths_proven"]:
        raise RuntimeError("selected candidate does not cover all 17 targets")

    model_records = {
        "untouched_standard_qdq": {"path": str(QDQ_PATH), "sha256": sha256_file(QDQ_PATH)},
        "v11_noop_rewrite": {"path": str(NOOP_PATH), "sha256": sha256_file(NOOP_PATH)},
        "strict_control": {"path": str(strict_path), "sha256": sha256_file(strict_path)},
        "legacy_semantic_expression": {"path": str(SEMANTIC_PATH), "sha256": sha256_file(SEMANTIC_PATH)},
        "selected_candidate": {"path": str(SELECTED_PATH), "sha256": sha256_file(SELECTED_PATH)},
    }
    payload = {
        "schema_version": AUDIT_SCHEMA,
        "completion_status": "success",
        "probe": {
            "sample_count": len(probe_indices),
            "indices": probe_indices,
            "batch_size": args.batch_size,
            "development_split_digest": controls_report["development_split_digest"],
        },
        "models": model_records,
        "target_count": len(sites),
        "strict_control": {
            "graph_topology_exact": graph_topology_hash(source) == graph_topology_hash(strict),
            "node_sequence_exact": [protobuf_hash(node) for node in source.graph.node]
            == [protobuf_hash(node) for node in strict.graph.node],
            "initializer_sequence_exact": [protobuf_hash(item) for item in source.graph.initializer]
            == [protobuf_hash(item) for item in strict.graph.initializer],
            "qdq_topology": qdq["strict_control"],
            "all_instrumented_tensors_exact": strict_exact,
            "probe_comparison": strict_probe,
            "target_topology": target_topology_audit(
                source, strict, sites, variant="strict_operator_topology_preserving_control"
            ),
        },
        "v11_noop_control": {
            "graph_topology_exact": graph_topology_hash(source) == graph_topology_hash(noop),
            "qdq_topology": qdq["v11_noop"],
            "historical_probe_row": next(
                row for row in controls_report["rows"]
                if row["control_id"] == "topology_preserving_noop"
            ),
        },
        "semantic_expression_control": {
            "legacy_artifact_id": "float_equivalent_silu",
            "accepted_name": "semantic-expression control",
            "invalid_float_equivalent_label_rejected": True,
            "label_rejection_reason": label_rejection,
            "reproduces_original_qdq_boundaries": False,
            "qdq_topology": qdq["semantic_expression"],
            "probe_comparison": semantic_probe,
            "first_divergence_graph_reason": (
                "At act.call_0, the rewritten raw Sigmoid output feeds Mul directly. "
                "The original Sigmoid-output Q/DQ and post-Mul Q/DQ pairs are removed."
            ),
            "target_topology": target_topology_audit(
                source, semantic, sites, variant="semantic_expression_control"
            ),
        },
        "selected_candidate": {
            "candidate_id": receipt["candidate_id"],
            "frozen_model_hash_valid": sha256_file(SELECTED_PATH) == receipt["model_sha256"],
            "frozen_selection_receipt_hash": receipt["selection_receipt_hash"],
            "frozen_final_accuracy": selected_row["top1_accuracy"],
            "frozen_final_result_valid": True,
            "qdq_topology": qdq["selected_candidate"],
            "coverage": coverage,
            "probe_comparison": selected_probe,
            "first_intended_divergence_graph_reason": (
                "The selected path intentionally removes the original Sigmoid Q/DQ operand and "
                "post-Mul Q/DQ pair, computes raw SiLU, then executes the explicit calibrated "
                "two-segment piecewise quantize/dequantize path."
            ),
            "target_topology": target_topology_audit(
                source, selected, sites, variant="selected_piecewise_candidate"
            ),
        },
        "revised_root_cause_classification": "mixed, dominated by calibration/range evidence",
        "revised_root_cause_statement": (
            "The strict control proves that v1.1 QDQ target discovery, provenance, serialization, "
            "and unchanged graph execution are exact. The legacy semantic-expression control is "
            "not numerically equivalent because it removes 68 Q/DQ nodes, so its prediction "
            "agreement cannot support an equivalence claim. Output-aware range calibration still "
            "provides strong evidence for the recovery and all 17 selected paths are proven active; "
            "however, the entire historical 82.80% loss should be classified as mixed rather than "
            "attributed solely to range, because the preserved v0.6 reference used different "
            "whole-graph quantization treatment and approximation effects accumulate."
        ),
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
            "provider": "CPUExecutionProvider",
            "platform": platform.platform(),
            "torch_imported": "torch" in sys.modules,
        },
    }
    write_reports(output, payload)
    print(json.dumps({
        "completion_status": payload["completion_status"],
        "strict_exact": strict_exact,
        "semantic_first_divergence": semantic_probe["first_divergent_target_tensor"],
        "selected_first_divergence": selected_probe["first_divergent_target_tensor"],
        "selected_coverage": coverage["covered_site_count"],
        "report": str(output / "control_equivalence_audit.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
