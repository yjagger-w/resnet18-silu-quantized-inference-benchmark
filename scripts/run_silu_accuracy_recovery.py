"""Run the controlled v1.1 ORT-only SiLU accuracy-recovery experiment."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from silu_benchmark.accuracy_recovery import (
    RECOVERY_CONFIG_SCHEMA,
    build_candidate_manifest,
    build_noop_control,
    canonical_hash,
    deterministic_development_split,
    discover_qdq_silu_sites,
    instrument_outputs,
    piecewise_error_summary,
    rank_sensitivity,
    rewrite_qdq_silu,
    select_candidate,
    sha256_file,
    split_digest,
    summarize_values,
    validate_generated_output_path,
)
from silu_benchmark.benchmark_data import load_cifar_batch, normalize_cifar_images, numpy_batches
from silu_benchmark.calibration_manifest import load_manifest


DEFAULT_CONFIG = Path("configs/accuracy_recovery/silu_accuracy_recovery_v11.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="v1.1 ORT-only piecewise-SiLU accuracy recovery and controlled ablation"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, newline=""
    ) as temporary:
        temporary.write(value)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def atomic_json(path: Path, payload) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("CSV rows must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, newline=""
    ) as temporary:
        writer = csv.DictWriter(temporary, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def load_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != RECOVERY_CONFIG_SCHEMA:
        raise ValueError("invalid v1.1 accuracy-recovery config schema")
    if payload.get("provider") != "CPUExecutionProvider":
        raise ValueError("v1.1 supports only ORT CPUExecutionProvider")
    if payload.get("calibration_samples") != 2560:
        raise ValueError("v1.1 calibration must remain exactly 2,560 images")
    identities = [candidate.get("candidate_id") for candidate in payload.get("candidates", [])]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("candidate IDs must be non-empty and unique")
    return payload


def save_model(model: onnx.ModelProto, path: Path) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite generated model: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    reloaded = onnx.load(str(path))
    onnx.checker.check_model(reloaded)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    if session.get_providers()[0] != "CPUExecutionProvider":
        raise RuntimeError("generated model did not create a CPUExecutionProvider session")
    del session
    return sha256_file(path)


def session_for(path_or_model):
    source = str(path_or_model) if isinstance(path_or_model, Path) else path_or_model.SerializeToString()
    return ort.InferenceSession(source, providers=["CPUExecutionProvider"])


def evaluate(session, images, labels, indices, batch_size, *, collect_logits=False):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    correct = 0
    total = 0
    predictions = []
    logits_parts = []
    start = time.perf_counter()
    for batch, batch_labels in numpy_batches(images, labels, indices, batch_size):
        logits = session.run([output_name], {input_name: batch})[0]
        if logits.shape != (len(batch_labels), 10) or logits.dtype != np.float32:
            raise ValueError(f"unexpected logits shape/dtype: {logits.shape}/{logits.dtype}")
        if not np.all(np.isfinite(logits)):
            raise ValueError("model produced non-finite logits")
        batch_predictions = logits.argmax(axis=1)
        correct += int(np.count_nonzero(batch_predictions == batch_labels))
        total += len(batch_labels)
        predictions.append(batch_predictions.astype(np.int64))
        if collect_logits:
            logits_parts.append(np.asarray(logits, dtype=np.float32))
    return {
        "accuracy": correct / total,
        "correct": correct,
        "samples": total,
        "predictions": np.concatenate(predictions),
        "logits": np.concatenate(logits_parts) if collect_logits else None,
        "wall_time_seconds": time.perf_counter() - start,
    }


def comparison(reference, candidate) -> dict:
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return {
        "max_absolute_error": float(np.max(np.abs(difference))),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "mse": float(np.mean(difference * difference)),
    }


def graph_summary(model: onnx.ModelProto) -> dict:
    histogram = {}
    for node in model.graph.node:
        histogram[node.op_type] = histogram.get(node.op_type, 0) + 1
    return {
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "operator_histogram": dict(sorted(histogram.items())),
        "input_shapes": [
            [dimension.dim_value if dimension.HasField("dim_value") else dimension.dim_param or None
             for dimension in item.type.tensor_type.shape.dim]
            for item in model.graph.input
        ],
        "output_shapes": [
            [dimension.dim_value if dimension.HasField("dim_value") else dimension.dim_param or None
             for dimension in item.type.tensor_type.shape.dim]
            for item in model.graph.output
        ],
        "output_dtypes": [onnx.TensorProto.DataType.Name(item.type.tensor_type.elem_type)
                          for item in model.graph.output],
    }


def run_controls(qdq_path, qdq_model, noop_path, float_path, images, labels, probe_indices):
    noop = build_noop_control(qdq_model)
    float_control = rewrite_qdq_silu(qdq_model, mode="float_equivalent")
    noop_hash = save_model(noop.model, noop_path)
    float_hash = save_model(float_control.model, float_path)
    sessions = {
        "untouched_standard_qdq": session_for(qdq_path),
        "topology_preserving_noop": session_for(noop_path),
        "float_equivalent_silu": session_for(float_path),
    }
    measured = {
        name: evaluate(session, images, labels, probe_indices, len(probe_indices), collect_logits=True)
        for name, session in sessions.items()
    }
    reference = measured["untouched_standard_qdq"]
    rows = []
    for name, value in measured.items():
        metrics = comparison(reference["logits"], value["logits"])
        rows.append(
            {
                "control_id": name,
                "accuracy": value["accuracy"],
                "correct": value["correct"],
                "samples": value["samples"],
                "prediction_agreement": float(np.mean(value["predictions"] == reference["predictions"])),
                "logit_max_absolute_error": metrics["max_absolute_error"],
                "logit_mean_absolute_error": metrics["mean_absolute_error"],
                "output_shape": list(value["logits"].shape),
                "output_dtype": str(value["logits"].dtype),
                "checker_passed": True,
                "inference_passed": True,
            }
        )
    noop_row = next(row for row in rows if row["control_id"] == "topology_preserving_noop")
    if noop_row["logit_max_absolute_error"] != 0.0 or noop_row["prediction_agreement"] != 1.0:
        raise RuntimeError("topology-preserving no-op control is not bit-exact")
    return {
        "target_silu_count": len(noop.sites),
        "target_sites": [asdict(site) for site in noop.sites],
        "rewrite_mapping": noop.tensor_mappings,
        "models": {
            "untouched_standard_qdq": {"path": str(qdq_path), "sha256": sha256_file(qdq_path)},
            "topology_preserving_noop": {"path": str(noop_path), "sha256": noop_hash},
            "float_equivalent_silu": {"path": str(float_path), "sha256": float_hash},
        },
        "graph_summaries": {
            "untouched_standard_qdq": graph_summary(qdq_model),
            "topology_preserving_noop": graph_summary(noop.model),
            "float_equivalent_silu": graph_summary(float_control.model),
        },
        "probe_indices": list(probe_indices),
        "rows": rows,
    }, float_control


def collect_calibration_samples(
    float_control,
    images,
    labels,
    calibration_indices,
    batch_size,
    per_site_limit,
):
    mappings = {}
    logical_order = []
    for site in float_control.sites:
        for kind in ("pre_silu", "post_silu"):
            logical_name = f"{site.site_id}::{kind}"
            mappings[logical_name] = float_control.tensor_mappings[site.site_id][kind]
            logical_order.append(logical_name)
    instrumented = instrument_outputs(float_control.model, mappings)
    session = session_for(instrumented)
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    if output_names != list(mappings.values()):
        raise RuntimeError("instrumented output order changed")
    batch_count = int(np.ceil(len(calibration_indices) / batch_size))
    per_batch = max(1, int(np.ceil(per_site_limit / batch_count)))
    collected = {logical_name: [] for logical_name in logical_order}
    shapes = {}
    for batch, _batch_labels in numpy_batches(
        images, labels, calibration_indices, batch_size
    ):
        outputs = session.run(output_names, {input_name: batch})
        for logical_name, array in zip(logical_order, outputs):
            shapes.setdefault(logical_name, list(array.shape))
            flat = array.reshape(-1)
            take = min(per_batch, flat.size)
            indices = np.linspace(0, flat.size - 1, take, dtype=np.int64)
            collected[logical_name].append(np.asarray(flat[indices], dtype=np.float32))
    merged = {
        logical_name: np.concatenate(parts)[:per_site_limit]
        for logical_name, parts in collected.items()
    }
    site_pre = {site.site_id: merged[f"{site.site_id}::pre_silu"] for site in float_control.sites}
    site_post = {site.site_id: merged[f"{site.site_id}::post_silu"] for site in float_control.sites}
    return site_pre, site_post, shapes


def original_diagnosis(sites, pre_values, post_values, shapes, original_specs):
    rows = []
    for site in sites:
        error = piecewise_error_summary(post_values[site.site_id], original_specs[site.site_id])
        rows.append(
            {
                "site_id": site.site_id,
                "module_path": site.module_path,
                "call_index": site.call_index,
                "pre_silu_tensor": site.input_tensor,
                "post_silu_tensor": site.mul_output_tensor,
                "pre_silu_shape": shapes[f"{site.site_id}::pre_silu"],
                "post_silu_shape": shapes[f"{site.site_id}::post_silu"],
                "pre_silu_statistics": summarize_values(pre_values[site.site_id]),
                "post_silu_statistics": summarize_values(post_values[site.site_id]),
                "original_parameters": {
                    **asdict(original_specs[site.site_id]),
                    "lower_scale": original_specs[site.site_id].lower_scale,
                    "upper_scale": original_specs[site.site_id].upper_scale,
                    "lower_zero_point": original_specs[site.site_id].lower_zero_point,
                    "upper_zero_point": original_specs[site.site_id].upper_zero_point,
                },
                "original_quantization_error": error,
                "node_provenance": asdict(site),
            }
        )
    return rows


def run_sensitivity(
    qdq_model,
    sites,
    original_specs,
    images,
    labels,
    development_indices,
    batch_size,
    artifact_root,
    reference,
):
    rows = []
    for order, site in enumerate(sites):
        result = rewrite_qdq_silu(
            qdq_model,
            mode="piecewise",
            site_specs={site.site_id: original_specs[site.site_id]},
            selected_site_ids=[site.site_id],
            candidate_id="original_v06_ort_single_site",
        )
        path = artifact_root / "sensitivity" / f"{order:02d}_{site.site_id.replace('.', '_')}.onnx"
        model_hash = save_model(result.model, path)
        measured = evaluate(
            session_for(path), images, labels, development_indices, batch_size, collect_logits=True
        )
        logit_metrics = comparison(reference["logits"], measured["logits"])
        rows.append(
            {
                "site_id": site.site_id,
                "module_path": site.module_path,
                "call_index": site.call_index,
                "model_sha256": model_hash,
                "development_accuracy": measured["accuracy"],
                "accuracy_delta_vs_standard_qdq_pp": 100.0 * (measured["accuracy"] - reference["accuracy"]),
                "prediction_agreement_vs_standard_qdq": float(
                    np.mean(measured["predictions"] == reference["predictions"])
                ),
                "logit_mae": logit_metrics["mean_absolute_error"],
                "logit_max_absolute_error": logit_metrics["max_absolute_error"],
                "wall_time_seconds": measured["wall_time_seconds"],
            }
        )
    ranked = rank_sensitivity(rows)
    for rank, row in enumerate(ranked, start=1):
        row["sensitivity_rank"] = rank
    return ranked


def candidate_error_compact(manifest):
    clipping = [
        entry["quantization_error"]["clipped_above_fraction"] for entry in manifest["sites"]
    ]
    mse = [entry["quantization_error"]["mse"] for entry in manifest["sites"]]
    return {
        "maximum_site_clipped_above_fraction": float(max(clipping)),
        "mean_site_clipped_above_fraction": float(np.mean(clipping)),
        "maximum_site_mse": float(max(mse)),
        "mean_site_mse": float(np.mean(mse)),
    }


def run_candidates(
    config,
    qdq_model,
    sites,
    post_values,
    provenance,
    images,
    labels,
    development_indices,
    batch_size,
    artifact_root,
    reference,
):
    rows = []
    manifests = {}
    for order, candidate in enumerate(config["candidates"]):
        candidate_id = candidate["candidate_id"]
        row = {
            "candidate_id": candidate_id,
            "configuration_hash": canonical_hash(candidate),
            "configuration_order": order,
            "split_role": "development",
            "status": "running",
        }
        try:
            manifest, specs = build_candidate_manifest(
                candidate=candidate,
                site_values=post_values,
                sites=sites,
                provenance=provenance,
            )
            manifest_path = artifact_root / "manifests" / f"{candidate_id}.json"
            atomic_json(manifest_path, manifest)
            result = rewrite_qdq_silu(
                qdq_model,
                mode="piecewise",
                site_specs=specs,
                candidate_id=candidate_id,
            )
            model_path = artifact_root / "candidates" / f"{candidate_id}.onnx"
            model_hash = save_model(result.model, model_path)
            measured = evaluate(
                session_for(model_path), images, labels, development_indices,
                batch_size, collect_logits=True,
            )
            metrics = comparison(reference["logits"], measured["logits"])
            row.update(
                {
                    "status": "success",
                    "failure": None,
                    "manifest_path": str(manifest_path),
                    "manifest_hash": manifest["manifest_hash"],
                    "model_path": str(model_path),
                    "model_sha256": model_hash,
                    "target_node_count": len(sites),
                    "calibration_data_digest": provenance["calibration_data_digest"],
                    "development_split_digest": provenance["development_split_digest"],
                    "accuracy": measured["accuracy"],
                    "correct": measured["correct"],
                    "samples": measured["samples"],
                    "prediction_agreement_vs_standard_qdq": float(
                        np.mean(measured["predictions"] == reference["predictions"])
                    ),
                    "logit_mean_absolute_error": metrics["mean_absolute_error"],
                    "logit_max_absolute_error": metrics["max_absolute_error"],
                    "per_layer_error_summary": candidate_error_compact(manifest),
                    "wall_time_seconds": measured["wall_time_seconds"],
                }
            )
            manifests[candidate_id] = manifest
        except Exception as error:
            row.update(
                {
                    "status": "failed",
                    "failure": {"type": type(error).__name__, "message": str(error)},
                }
            )
        rows.append(row)
    successful = [row for row in rows if row["status"] == "success"]
    selected = select_candidate(successful, split_role="development")
    return rows, selected, manifests


def write_control_reports(directory: Path, payload: dict) -> None:
    atomic_json(directory / "controls.json", payload)
    atomic_csv(directory / "controls.csv", payload["rows"])
    lines = [
        "# v1.1 rewrite controls",
        "",
        "All rows use the same deterministic training-batch development probe. The no-op must be bit-exact.",
        "",
        "| Control | Accuracy | Prediction agreement | Logit max abs | Logit mean abs |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['control_id']} | {row['accuracy']:.4%} | {row['prediction_agreement']:.4%} | "
            f"{row['logit_max_absolute_error']:.8g} | {row['logit_mean_absolute_error']:.8g} |"
        )
    atomic_text(directory / "controls.md", "\n".join(lines) + "\n")


def write_diagnosis_reports(directory: Path, payload: dict) -> None:
    atomic_json(directory / "activation_statistics.json", payload)
    rows = []
    for item in payload["sites"]:
        error = item["original_quantization_error"]
        post = item["post_silu_statistics"]
        rows.append(
            {
                "site_id": item["site_id"],
                "post_min": post["min"],
                "post_max": post["max"],
                "vmax": item["original_parameters"]["vmax"],
                "clipped_above_fraction": error["clipped_above_fraction"],
                "occupied_code_count": error["occupied_code_count"],
                "lower_segment_fraction": error["lower_segment_fraction"],
                "mae": error["mae"],
                "mse": error["mse"],
                "max_absolute_error": error["max_absolute_error"],
                "cosine_similarity": error["cosine_similarity"],
            }
        )
    atomic_csv(directory / "activation_statistics.csv", rows)


def write_sensitivity_reports(directory: Path, rows: list[dict]) -> None:
    atomic_json(directory / "layer_sensitivity.json", {"rows": rows})
    atomic_csv(directory / "layer_sensitivity.csv", rows)
    lines = [
        "# v1.1 single-site sensitivity",
        "",
        "Ranking uses only the deterministic development split; rank 1 is most damaging.",
        "",
        "| Rank | Site | Development accuracy | Delta vs Standard-QDQ (pp) | Agreement | Logit MAE |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['sensitivity_rank']} | {row['site_id']} | {row['development_accuracy']:.4%} | "
            f"{row['accuracy_delta_vs_standard_qdq_pp']:.4f} | "
            f"{row['prediction_agreement_vs_standard_qdq']:.4%} | {row['logit_mae']:.8g} |"
        )
    atomic_text(directory / "layer_sensitivity.md", "\n".join(lines) + "\n")


def write_candidate_reports(directory: Path, payload: dict) -> None:
    atomic_json(directory / "candidate_matrix.json", payload)
    flattened = []
    for row in payload["rows"]:
        flattened.append(
            {
                key: value if not isinstance(value, (dict, list)) else json.dumps(value, sort_keys=True)
                for key, value in row.items()
            }
        )
    atomic_csv(directory / "candidate_matrix.csv", flattened)
    lines = [
        "# v1.1 development candidate matrix",
        "",
        "Selection uses development accuracy, then prediction agreement, then configuration order. Final-test labels were not available to selection.",
        "",
        "| Candidate | Status | Development accuracy | Agreement vs Standard-QDQ | Logit MAE |",
        "|---|---|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['candidate_id']} | {row['status']} | "
            f"{row.get('accuracy', float('nan')):.4%} | "
            f"{row.get('prediction_agreement_vs_standard_qdq', float('nan')):.4%} | "
            f"{row.get('logit_mean_absolute_error', float('nan')):.8g} |"
        )
    lines.extend(["", f"Frozen selection: `{payload['selected']['candidate_id']}`."])
    atomic_text(directory / "candidate_matrix.md", "\n".join(lines) + "\n")


def write_final_reports(directory: Path, payload: dict) -> None:
    atomic_json(directory / "final_comparison.json", payload)
    atomic_csv(directory / "final_comparison.csv", payload["rows"])
    lines = [
        "# v1.1 final 10,000-image comparison",
        "",
        "Candidate selection was frozen on the disjoint development split before this final-test evaluation.",
        "",
        "| Model | Role | Top-1 accuracy | Gap vs Standard-QDQ (pp) | Notes |",
        "|---|---|---:|---:|---|",
    ]
    for row in payload["rows"]:
        gap = "—" if row["gap_vs_standard_qdq_pp"] is None else f"{row['gap_vs_standard_qdq_pp']:.4f}"
        lines.append(
            f"| {row['model']} | {row['role']} | {row['top1_accuracy']:.4%} | {gap} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            f"Goal within 0.5 percentage points of Standard-QDQ: **{'met' if payload['goal_met'] else 'not met'}**.",
            "",
            payload["conclusion"],
            "",
            "This is ORT functional-reference evidence only, not QNN/NPU deployment or whole-model C++ performance evidence.",
        ]
    )
    atomic_text(directory / "final_comparison.md", "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    config_path = (ROOT / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    if "torch" in sys.modules or "torchvision" in sys.modules:
        raise RuntimeError("ORT-only v1.1 process must not import torch or torchvision")

    artifact_root = (ROOT / config["generated_artifact_root"]).resolve()
    diagnosis_output = validate_generated_output_path(ROOT / config["diagnosis_output"], ROOT)
    development_output = validate_generated_output_path(ROOT / config["development_output"], ROOT)
    final_output = validate_generated_output_path(ROOT / config["final_output"], ROOT)
    for destination in (artifact_root, diagnosis_output, development_output, final_output):
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite existing v1.1 output: {destination}")

    qdq_path = (ROOT / config["standard_qdq_model"]).resolve()
    fp32_path = (ROOT / config["fp32_model"]).resolve()
    original_piecewise_path = (ROOT / config["original_piecewise_model"]).resolve()
    original_manifest_path = (ROOT / config["original_piecewise_manifest"]).resolve()
    historical_path = (ROOT / config["historical_report"]).resolve()
    data_root = (ROOT / config["data_root"]).resolve()
    source_hashes = {
        "config": sha256_file(config_path),
        "runner": sha256_file(Path(__file__).resolve()),
        "accuracy_recovery_source": sha256_file(ROOT / "src/silu_benchmark/accuracy_recovery.py"),
        "benchmark_data_source": sha256_file(ROOT / "src/silu_benchmark/benchmark_data.py"),
        "standard_qdq_model": sha256_file(qdq_path),
        "fp32_model": sha256_file(fp32_path),
        "original_piecewise_model": sha256_file(original_piecewise_path),
        "original_piecewise_manifest": sha256_file(original_manifest_path),
        "historical_report": sha256_file(historical_path),
    }
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    historical_by_id = {row["variant_id"]: row for row in historical["variants"]}
    if historical_by_id["standard_static_qdq_ort_cpu"]["accuracy"] != 0.9357:
        raise ValueError("locked historical Standard-QDQ accuracy is not 93.57%")
    if historical_by_id["silu_piecewise_ort_reference_cpu"]["accuracy"] != 0.828:
        raise ValueError("locked historical original piecewise accuracy is not 82.80%")
    if historical["artifact_sha256"]["standard_static_qdq"] != source_hashes["standard_qdq_model"]:
        raise ValueError("locked Standard-QDQ model hash differs from v0.6.5 report")
    if historical["artifact_sha256"]["silu_piecewise_reference"] != source_hashes["original_piecewise_model"]:
        raise ValueError("locked original piecewise model hash differs from v0.6.5 report")

    training_images, training_labels = load_cifar_batch(data_root, config["calibration_batch"])
    calibration_indices = list(range(config["calibration_samples"]))
    development_indices = deterministic_development_split(
        total=len(training_images),
        calibration_indices=calibration_indices,
        sample_count=config["development_samples"],
        seed=config["seed"],
    )
    if set(calibration_indices) & set(development_indices):
        raise RuntimeError("calibration and development indices overlap")
    calibration_digest = split_digest(
        training_images, training_labels, calibration_indices, "calibration"
    )
    development_digest = split_digest(
        training_images, training_labels, development_indices, "development"
    )
    test_batch_digest = sha256_file(data_root / "cifar-10-batches-py/test_batch")

    qdq_model = onnx.load(str(qdq_path))
    sites = discover_qdq_silu_sites(qdq_model)
    artifact_root.mkdir(parents=True)
    controls, float_control = run_controls(
        qdq_path,
        qdq_model,
        artifact_root / "controls/topology_preserving_noop.onnx",
        artifact_root / "controls/float_equivalent_silu.onnx",
        training_images,
        training_labels,
        development_indices[: config["probe_samples"]],
    )
    controls.update(
        {
            "schema_version": "silu-accuracy-controls/v1",
            "source_hashes": source_hashes,
            "development_split_digest": development_digest,
        }
    )
    write_control_reports(diagnosis_output, controls)

    site_pre, site_post, shapes = collect_calibration_samples(
        float_control,
        training_images,
        training_labels,
        calibration_indices,
        config["development_batch_size"],
        config["per_site_calibration_values"],
    )
    _original_manifest, original_specs = load_manifest(original_manifest_path)
    if set(original_specs) != {site.site_id for site in sites}:
        raise ValueError("preserved original manifest does not match QDQ-aware sites")
    diagnosis_rows = original_diagnosis(
        sites, site_pre, site_post, shapes, original_specs
    )
    diagnosis = {
        "schema_version": "silu-accuracy-diagnosis/v1",
        "calibration_sample_count": len(calibration_indices),
        "calibration_indices": calibration_indices,
        "calibration_data_digest": calibration_digest,
        "development_split_digest": development_digest,
        "source_hashes": source_hashes,
        "sites": diagnosis_rows,
    }
    write_diagnosis_reports(diagnosis_output, diagnosis)

    standard_development = evaluate(
        session_for(qdq_path),
        training_images,
        training_labels,
        development_indices,
        config["development_batch_size"],
        collect_logits=True,
    )
    sensitivity = run_sensitivity(
        qdq_model,
        sites,
        original_specs,
        training_images,
        training_labels,
        development_indices,
        config["development_batch_size"],
        artifact_root,
        standard_development,
    )
    write_sensitivity_reports(diagnosis_output, sensitivity)

    provenance = {
        "calibration_split": "CIFAR-10 data_batch_1 indices 0..2559",
        "calibration_sample_count": len(calibration_indices),
        "calibration_data_digest": calibration_digest,
        "development_split": "deterministic subset of data_batch_1 excluding calibration indices",
        "development_sample_count": len(development_indices),
        "development_split_digest": development_digest,
        "final_test_batch_sha256": test_batch_digest,
        "source_hashes": source_hashes,
    }
    candidate_rows, selected, manifests = run_candidates(
        config,
        qdq_model,
        sites,
        site_post,
        provenance,
        training_images,
        training_labels,
        development_indices,
        config["development_batch_size"],
        artifact_root,
        standard_development,
    )
    selection_receipt = {
        "schema_version": "silu-accuracy-selection/v1",
        "frozen_before_final_test": True,
        "selection_split_role": "development",
        "selection_rule": config["selection_rule"],
        "development_split_digest": development_digest,
        "candidate_id": selected["candidate_id"],
        "configuration_hash": selected["configuration_hash"],
        "manifest_hash": selected["manifest_hash"],
        "model_sha256": selected["model_sha256"],
        "selected_development_metrics": {
            "accuracy": selected["accuracy"],
            "prediction_agreement_vs_standard_qdq": selected[
                "prediction_agreement_vs_standard_qdq"
            ],
        },
    }
    selection_receipt["selection_receipt_hash"] = canonical_hash(selection_receipt)
    atomic_json(development_output / "selection_receipt.json", selection_receipt)
    candidate_payload = {
        "schema_version": "silu-accuracy-candidate-results/v1",
        "standard_qdq_development": {
            key: value for key, value in standard_development.items()
            if key not in {"predictions", "logits"}
        },
        "calibration_data_digest": calibration_digest,
        "development_split_digest": development_digest,
        "rows": candidate_rows,
        "selected": selection_receipt,
    }
    write_candidate_reports(development_output, candidate_payload)

    # Final-test labels are loaded only after the immutable selection receipt exists.
    test_images, test_labels = load_cifar_batch(data_root, "test_batch")
    test_indices = list(range(10000))
    selected_session = session_for(Path(selected["model_path"]))
    final_standard = evaluate(
        session_for(qdq_path), test_images, test_labels, test_indices,
        config["development_batch_size"], collect_logits=False,
    )
    final_original = evaluate(
        session_for(original_piecewise_path), test_images, test_labels, test_indices,
        config["development_batch_size"], collect_logits=False,
    )
    final_selected = evaluate(
        selected_session, test_images, test_labels, test_indices,
        config["development_batch_size"], collect_logits=False,
    )
    standard_accuracy = final_standard["accuracy"]
    rows = [
        {
            "model": "FP32",
            "role": "historical/reference",
            "top1_accuracy": historical_by_id["fp32_ort_cpu"]["accuracy"],
            "gap_vs_standard_qdq_pp": None,
            "notes": "cited from locked v0.6.5 ORT report; not rerun",
        },
        {
            "model": "Standard-QDQ",
            "role": "unchanged deployment baseline",
            "top1_accuracy": standard_accuracy,
            "gap_vs_standard_qdq_pp": 0.0,
            "notes": "new ORT CPU measurement on all 10,000 test images",
        },
        {
            "model": "Original piecewise",
            "role": "preserved v0.6 reference",
            "top1_accuracy": final_original["accuracy"],
            "gap_vs_standard_qdq_pp": 100.0 * (final_original["accuracy"] - standard_accuracy),
            "notes": "preserved model/hash; new full-test ORT CPU measurement",
        },
        {
            "model": selected["candidate_id"],
            "role": "selected v1.1 candidate",
            "top1_accuracy": final_selected["accuracy"],
            "gap_vs_standard_qdq_pp": 100.0 * (final_selected["accuracy"] - standard_accuracy),
            "notes": "selection frozen on disjoint development split before test labels were loaded",
        },
    ]
    goal_met = (standard_accuracy - final_selected["accuracy"]) * 100.0 <= 0.5
    conclusion = (
        "The selected candidate met the predefined 0.5 percentage-point recovery target."
        if goal_met
        else "The selected candidate did not meet the predefined recovery target; the measured gap is reported without a recovery claim."
    )
    final_payload = {
        "schema_version": "silu-accuracy-final-results/v1",
        "completion_status": "success",
        "selection_receipt": selection_receipt,
        "selection_receipt_path": str(development_output / "selection_receipt.json"),
        "final_test_sample_count": 10000,
        "final_test_batch_sha256": test_batch_digest,
        "source_hashes": source_hashes,
        "rows": rows,
        "goal_threshold_percentage_points": 0.5,
        "goal_met": goal_met,
        "conclusion": conclusion,
        "wall_time_seconds": {
            "standard_qdq": final_standard["wall_time_seconds"],
            "original_piecewise": final_original["wall_time_seconds"],
            "selected_candidate": final_selected["wall_time_seconds"],
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
            "provider": "CPUExecutionProvider",
            "available_providers": ort.get_available_providers(),
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "torch_imported": "torch" in sys.modules,
        },
        "limitations": [
            "ORT functional reference only; no custom-op, QNN, NPU, GPU, or C++ whole-model integration.",
            "Candidate replaces proven SiLU-local QDQ islands while preserving the locked Standard-QDQ weights and all non-SiLU graph treatment.",
            "Development selection uses only a deterministic subset of CIFAR-10 training data disjoint from the 2,560-image calibration subset.",
        ],
    }
    write_final_reports(final_output, final_payload)
    audit = {
        "schema_version": "silu-accuracy-audit/v1",
        "config": config,
        "config_hash": canonical_hash(config),
        "source_hashes": source_hashes,
        "historical_measurements": {
            key: {"accuracy": value["accuracy"], "samples": value["samples"]}
            for key, value in historical_by_id.items()
        },
        "calibration_data_digest": calibration_digest,
        "development_split_digest": development_digest,
        "final_test_batch_sha256": test_batch_digest,
        "target_silu_count": len(sites),
        "generated_outputs": {
            "artifact_root": str(artifact_root),
            "diagnosis": str(diagnosis_output),
            "development": str(development_output),
            "final": str(final_output),
        },
    }
    atomic_json(final_output / "audit_provenance.json", audit)
    print(json.dumps(final_payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
