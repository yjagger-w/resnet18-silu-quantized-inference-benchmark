"""Offline comparison of local ORT references with downloaded Galaxy S22 outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .backends.qnn_aihub_backend import numerical_metrics
from .qnn_local_accuracy import accuracy_metrics, prediction_agreement, sha256_file
from .qnn_preflight import array_metadata, write_deterministic_npz


S22_REPORT_SCHEMA = "qnn-cifar10-s22-preflight-accuracy/v1.6"
S22_FULL_REPORT_SCHEMA = "qnn-cifar10-s22-full-accuracy/v1.6"
S22_MODEL_ORDER = ("fp32", "qdq_int8")


def load_npz_exact(path: Path, expected_keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive.files) != expected_keys:
            raise ValueError(
                f"{Path(path).name} keys must be {expected_keys}, got {tuple(archive.files)}"
            )
        return {name: np.asarray(archive[name]) for name in archive.files}


def _validate_labels_and_indices(
    labels: np.ndarray, original_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    original_indices = np.asarray(original_indices)
    if labels.ndim != 1 or labels.dtype != np.int64 or labels.size == 0:
        raise ValueError(f"labels must be a non-empty int64 vector, got {labels.shape}/{labels.dtype}")
    sample_count = labels.size
    if original_indices.shape != (sample_count,) or original_indices.dtype != np.int64:
        raise ValueError(
            f"original_indices must be int64 ({sample_count},), got "
            f"{original_indices.shape}/{original_indices.dtype}"
        )
    if len(np.unique(original_indices)) != sample_count or np.any(np.diff(original_indices) <= 0):
        raise ValueError("original_indices must be unique and strictly increasing")
    if np.any((labels < 0) | (labels > 9)):
        raise ValueError("labels must be CIFAR-10 class IDs in [0, 9]")
    return labels, original_indices


def compare_local_and_remote(
    labels: np.ndarray,
    original_indices: np.ndarray,
    local_logits: np.ndarray,
    remote_logits: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    labels, original_indices = _validate_labels_and_indices(labels, original_indices)
    local_logits = np.asarray(local_logits)
    remote_logits = np.asarray(remote_logits)
    expected_shape = (labels.size, 10)
    for name, logits in (("local", local_logits), ("remote", remote_logits)):
        if logits.shape != expected_shape or logits.dtype != np.float32:
            raise ValueError(
                f"{name} logits must be float32 {expected_shape}, got "
                f"{logits.shape}/{logits.dtype}"
            )
        if not np.all(np.isfinite(logits)):
            raise ValueError(f"{name} logits contain non-finite values")
    local_predictions = np.argmax(local_logits, axis=1).astype(np.int64)
    remote_predictions = np.argmax(remote_logits, axis=1).astype(np.int64)
    local_accuracy = accuracy_metrics(labels, local_predictions)
    remote_accuracy = accuracy_metrics(labels, remote_predictions)
    agreement = prediction_agreement(local_predictions, remote_predictions)
    disagreement_indices = np.asarray(agreement["disagreement_indices"], dtype=np.int64)
    result = {
        "local": {
            **local_accuracy,
            "misclassified_original_indices": original_indices[
                np.asarray(local_accuracy["misclassified_indices"], dtype=np.int64)
            ].tolist(),
        },
        "s22": {
            **remote_accuracy,
            "misclassified_original_indices": original_indices[
                np.asarray(remote_accuracy["misclassified_indices"], dtype=np.int64)
            ].tolist(),
            "accuracy_change_vs_local_pp": round(
                100.0
                * (remote_accuracy["top1_accuracy"] - local_accuracy["top1_accuracy"]),
                12,
            ),
        },
        "local_vs_s22_prediction_agreement": {
            **agreement,
            "disagreement_original_indices": original_indices[disagreement_indices].tolist(),
        },
        "local_vs_s22_logits": numerical_metrics(local_logits, remote_logits),
    }
    return result, local_predictions, remote_predictions


def build_s22_preflight_report(
    *,
    labels: np.ndarray,
    original_indices: np.ndarray,
    local_logits: Mapping[str, np.ndarray],
    remote_logits: Mapping[str, np.ndarray],
    provenance: Mapping[str, Any],
    jobs: Mapping[str, Mapping[str, str]],
    selection: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    labels, original_indices = _validate_labels_and_indices(labels, original_indices)
    if labels.shape != (1000,) or np.bincount(labels, minlength=10).tolist() != [100] * 10:
        raise ValueError("preflight labels must contain exactly 100 samples per class")
    if tuple(local_logits) != S22_MODEL_ORDER or tuple(remote_logits) != S22_MODEL_ORDER:
        raise ValueError(f"local and remote logits must use ordered model IDs {S22_MODEL_ORDER}")
    if tuple(jobs) != S22_MODEL_ORDER:
        raise ValueError(f"jobs must use ordered model IDs {S22_MODEL_ORDER}")
    models = {}
    predictions: dict[str, np.ndarray] = {
        "labels": labels,
        "original_indices": original_indices,
    }
    for model_id in S22_MODEL_ORDER:
        comparison, local_predictions, s22_predictions = compare_local_and_remote(
            labels,
            original_indices,
            local_logits[model_id],
            remote_logits[model_id],
        )
        models[model_id] = {"jobs": dict(jobs[model_id]), **comparison}
        predictions[f"{model_id}_local_predictions"] = local_predictions
        predictions[f"{model_id}_s22_predictions"] = s22_predictions

    s22_agreement = prediction_agreement(
        predictions["fp32_s22_predictions"], predictions["qdq_int8_s22_predictions"]
    )
    disagreement_indices = np.asarray(s22_agreement["disagreement_indices"], dtype=np.int64)
    qdq_gain = models["qdq_int8"]["s22"]["correct"] - models["qdq_int8"]["local"]["correct"]
    report = {
        "schema_version": S22_REPORT_SCHEMA,
        "scope": "Galaxy S22 / Android 12 QNN accuracy on the deterministic 1,000-image CIFAR-10 preflight subset",
        "statements": {
            "ai_hub_connected_during_report_generation": False,
            "ai_hub_tasks_created_during_report_generation": False,
            "is_full_cifar10_accuracy": False,
            "piecewise_reference_included": False,
        },
        "provenance": dict(provenance),
        "selection": dict(selection),
        "models": models,
        "s22_fp32_vs_qdq_int8_prediction_agreement": {
            **s22_agreement,
            "disagreement_original_indices": original_indices[disagreement_indices].tolist(),
        },
        "interpretation": {
            "qdq_s22_accuracy_change_vs_local_pp": models["qdq_int8"]["s22"][
                "accuracy_change_vs_local_pp"
            ],
            "qdq_s22_additional_correct_samples_vs_local": qdq_gain,
            "required_caveat": (
                "The S22 QDQ result is +0.2 percentage points because it classifies 2 more "
                "samples correctly on this fixed 1,000-image subset. This does not demonstrate "
                "that quantization improves generalization accuracy."
            ),
        },
        "index_convention": (
            "Selected indices are zero-based positions in the exported 1,000-image arrays; "
            "original indices are zero-based positions in official CIFAR-10 test_batch order."
        ),
    }
    return report, predictions


def build_s22_full_report(
    *,
    labels: np.ndarray,
    original_indices: np.ndarray,
    local_logits: Mapping[str, np.ndarray],
    remote_logits: Mapping[str, np.ndarray],
    provenance: Mapping[str, Any],
    jobs: Mapping[str, Mapping[str, str]],
    selection: Mapping[str, Any],
    performance: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build the frozen full-test Galaxy S22 accuracy/performance report offline."""

    labels, original_indices = _validate_labels_and_indices(labels, original_indices)
    if labels.shape != (10000,):
        raise ValueError(f"full-test labels must have shape (10000,), got {labels.shape}")
    if not np.array_equal(original_indices, np.arange(10000, dtype=np.int64)):
        raise ValueError("full-test original_indices must be exactly 0 through 9999")
    if np.bincount(labels, minlength=10).tolist() != [1000] * 10:
        raise ValueError("full-test labels must contain exactly 1000 samples per class")
    if tuple(local_logits) != S22_MODEL_ORDER or tuple(remote_logits) != S22_MODEL_ORDER:
        raise ValueError(f"local and remote logits must use ordered model IDs {S22_MODEL_ORDER}")
    if tuple(jobs) != S22_MODEL_ORDER or tuple(performance) != S22_MODEL_ORDER:
        raise ValueError(f"jobs and performance must use ordered model IDs {S22_MODEL_ORDER}")

    models: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {
        "labels": labels,
        "original_indices": original_indices,
    }
    for model_id in S22_MODEL_ORDER:
        comparison, local_predictions, s22_predictions = compare_local_and_remote(
            labels,
            original_indices,
            local_logits[model_id],
            remote_logits[model_id],
        )
        models[model_id] = {
            "jobs": dict(jobs[model_id]),
            "performance": dict(performance[model_id]),
            **comparison,
        }
        predictions[f"{model_id}_local_predictions"] = local_predictions
        predictions[f"{model_id}_s22_predictions"] = s22_predictions

    fp32_s22 = predictions["fp32_s22_predictions"]
    qdq_s22 = predictions["qdq_int8_s22_predictions"]
    s22_agreement = prediction_agreement(fp32_s22, qdq_s22)
    disagreement_indices = np.asarray(s22_agreement["disagreement_indices"], dtype=np.int64)
    fp32_correct = fp32_s22 == labels
    qdq_correct = qdq_s22 == labels
    contingency = {
        "both_correct": int(np.count_nonzero(fp32_correct & qdq_correct)),
        "fp32_only_correct": int(np.count_nonzero(fp32_correct & ~qdq_correct)),
        "qdq_int8_only_correct": int(np.count_nonzero(~fp32_correct & qdq_correct)),
        "both_wrong": int(np.count_nonzero(~fp32_correct & ~qdq_correct)),
    }
    qdq_vs_fp32_pp = round(
        100.0
        * (
            models["qdq_int8"]["s22"]["top1_accuracy"]
            - models["fp32"]["s22"]["top1_accuracy"]
        ),
        12,
    )
    qdq_local_to_s22_pp = models["qdq_int8"]["s22"]["accuracy_change_vs_local_pp"]
    report = {
        "schema_version": S22_FULL_REPORT_SCHEMA,
        "scope": (
            "Galaxy S22 / Android 12 QNN accuracy on all 10,000 images in official "
            "CIFAR-10 test_batch order, combined with the frozen v1.6 profile results"
        ),
        "statements": {
            "ai_hub_connected_during_report_generation": False,
            "ai_hub_tasks_created_during_report_generation": False,
            "is_full_cifar10_accuracy": True,
            "piecewise_reference_included": False,
            "recommended_deployment": "standard QDQ INT8",
        },
        "provenance": dict(provenance),
        "selection": dict(selection),
        "models": models,
        "s22_fp32_vs_qdq_int8": {
            "prediction_agreement": {
                **s22_agreement,
                "disagreement_original_indices": original_indices[
                    disagreement_indices
                ].tolist(),
            },
            "qdq_int8_accuracy_change_vs_fp32_pp": qdq_vs_fp32_pp,
            "correctness_contingency": contingency,
        },
        "interpretation": {
            "qdq_int8_s22_accuracy_change_vs_fp32_pp": qdq_vs_fp32_pp,
            "qdq_int8_s22_accuracy_change_vs_local_pp": qdq_local_to_s22_pp,
            "backend_prediction_drift_count": models["qdq_int8"][
                "local_vs_s22_prediction_agreement"
            ]["disagreement_count"],
            "required_boundaries": [
                "QDQ INT8 loses only 0.05 percentage points versus FP32 on Galaxy S22.",
                (
                    "The 0.11 percentage-point increase from local QDQ to S22 QDQ must not "
                    "be interpreted as quantization improving generalization accuracy."
                ),
                (
                    "The 120 local/S22 QDQ prediction changes show backend numerical drift, "
                    "while aggregate accuracy remains stable."
                ),
                (
                    "piecewise_v065 has only 82.80% local accuracy and 1.68766 ms mean "
                    "latency, so no full Galaxy S22 CIFAR-10 task was run for it."
                ),
                (
                    "piecewise_v065 is retained only as compiler-compatibility and operator-"
                    "decomposition diagnostic evidence; it is not the current best deployment."
                ),
                "The final recommended deployment is standard QDQ INT8.",
            ],
        },
        "index_convention": (
            "All indices are zero-based positions in official CIFAR-10 test_batch order; "
            "the exported full-test order is exactly 0 through 9999."
        ),
    }
    return report, predictions


def s22_accuracy_markdown(report: Mapping[str, Any]) -> str:
    labels = {"fp32": "FP32", "qdq_int8": "QDQ INT8"}
    lines = [
        "# Galaxy S22 QNN CIFAR-10 preflight accuracy",
        "",
        "This report compares already-downloaded Galaxy S22 QNN outputs with local ONNX Runtime CPU references. Report generation did not connect to AI Hub or create a remote task.",
        "",
        "The fixed preflight subset contains the first 100 samples of every class encountered in original `test_batch` order (1,000 images total). This is not full CIFAR-10 accuracy.",
        "",
        "## Accuracy and local/S22 agreement",
        "",
        "| Model | Compile job | Inference job | Local correct | Local Top-1 | S22 correct | S22 Top-1 | Change | Local/S22 agreement |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_id in S22_MODEL_ORDER:
        model = report["models"][model_id]
        lines.append(
            f"| {labels[model_id]} | `{model['jobs']['compile']}` | "
            f"`{model['jobs']['inference']}` | {model['local']['correct']} / 1000 | "
            f"{model['local']['top1_accuracy_percent']:.4f}% | {model['s22']['correct']} / 1000 | "
            f"{model['s22']['top1_accuracy_percent']:.4f}% | "
            f"{model['s22']['accuracy_change_vs_local_pp']:+.4f} pp | "
            f"{model['local_vs_s22_prediction_agreement']['agreement_percent']:.4f}% |"
        )
    cross = report["s22_fp32_vs_qdq_int8_prediction_agreement"]
    lines.extend(
        [
            "",
            f"S22 FP32/QDQ prediction agreement: {cross['agreement_count']} / 1000 "
            f"({cross['agreement_percent']:.4f}%).",
            "",
            "## Local versus S22 logit metrics",
            "",
            "| Model | Mean absolute error | Max absolute error | RMSE | Mean cosine | Min cosine |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model_id in S22_MODEL_ORDER:
        metrics = report["models"][model_id]["local_vs_s22_logits"]
        lines.append(
            f"| {labels[model_id]} | {metrics['mean_abs_error']:.9f} | "
            f"{metrics['max_abs_error']:.9f} | {metrics['rmse']:.9f} | "
            f"{metrics['mean_cosine_similarity']:.9f} | "
            f"{metrics['min_cosine_similarity']:.9f} |"
        )
    lines.extend(["", "## Prediction disagreements (original test_batch indices)", ""])
    for model_id in S22_MODEL_ORDER:
        indices = report["models"][model_id]["local_vs_s22_prediction_agreement"][
            "disagreement_original_indices"
        ]
        text = ", ".join(str(value) for value in indices) if indices else "(none)"
        lines.append(f"- {labels[model_id]} local vs S22 ({len(indices)}): `{text}`")
    cross_indices = cross["disagreement_original_indices"]
    lines.append(
        f"- S22 FP32 vs QDQ INT8 ({len(cross_indices)}): "
        f"`{', '.join(str(value) for value in cross_indices) if cross_indices else '(none)'}`"
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"]["required_caveat"],
            "",
            "`predictions.npz` contains labels, original indices, and local/S22 predictions only. It contains no image or logit arrays.",
            "",
        ]
    )
    return "\n".join(lines)


def write_s22_accuracy_outputs(
    output_dir: Path,
    report: Mapping[str, Any],
    predictions: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Path]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": output_dir / "predictions.npz",
        "json": output_dir / "preflight_accuracy.json",
        "markdown": output_dir / "preflight_accuracy_summary.md",
    }
    write_deterministic_npz(paths["predictions"], predictions)
    payload = dict(report)
    payload["predictions_artifact"] = {
        "path": paths["predictions"].name,
        "size_bytes": paths["predictions"].stat().st_size,
        "sha256": sha256_file(paths["predictions"]),
        "arrays": {name: array_metadata(value) for name, value in predictions.items()},
        "contains_images": False,
        "contains_logits": False,
    }
    paths["json"].write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    paths["markdown"].write_text(s22_accuracy_markdown(payload), encoding="utf-8")
    return payload, paths


def s22_full_accuracy_markdown(report: Mapping[str, Any]) -> str:
    labels = {"fp32": "FP32", "qdq_int8": "QDQ INT8"}
    total = report["models"]["fp32"]["local"]["total"]
    lines = [
        "# Galaxy S22 QNN CIFAR-10 full-test accuracy and performance",
        "",
        (
            "This final v1.6 report compares already-downloaded Galaxy S22 / Android 12 "
            "QNN outputs with local ONNX Runtime CPU references over all 10,000 images in "
            "official CIFAR-10 `test_batch` order. Report generation did not connect to AI "
            "Hub or create a remote task."
        ),
        "",
        "## Accuracy and local/S22 agreement",
        "",
        (
            "| Model | Compile | Profile | Full inference | Local correct | Local Top-1 | "
            "S22 correct | S22 Top-1 | Local/S22 change | Local/S22 agreement |"
        ),
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_id in S22_MODEL_ORDER:
        model = report["models"][model_id]
        lines.append(
            f"| {labels[model_id]} | `{model['jobs']['compile']}` | "
            f"`{model['jobs']['profile']}` | `{model['jobs']['inference']}` | "
            f"{model['local']['correct']} / {total} | "
            f"{model['local']['top1_accuracy_percent']:.2f}% | "
            f"{model['s22']['correct']} / {total} | "
            f"{model['s22']['top1_accuracy_percent']:.2f}% | "
            f"{model['s22']['accuracy_change_vs_local_pp']:+.2f} pp | "
            f"{model['local_vs_s22_prediction_agreement']['agreement_percent']:.2f}% |"
        )

    cross = report["s22_fp32_vs_qdq_int8"]
    agreement = cross["prediction_agreement"]
    contingency = cross["correctness_contingency"]
    lines.extend(
        [
            "",
            "## Galaxy S22 FP32 versus QDQ INT8",
            "",
            f"- Prediction agreement: {agreement['agreement_count']} / {total} "
            f"({agreement['agreement_percent']:.2f}%); disagreements: "
            f"{agreement['disagreement_count']}.",
            f"- QDQ INT8 Top-1 change versus FP32: "
            f"{cross['qdq_int8_accuracy_change_vs_fp32_pp']:+.2f} percentage points.",
            f"- Both correct: {contingency['both_correct']}; FP32 only correct: "
            f"{contingency['fp32_only_correct']}; QDQ only correct: "
            f"{contingency['qdq_int8_only_correct']}; both wrong: "
            f"{contingency['both_wrong']}.",
            "",
            "## Local versus S22 numerical comparison",
            "",
            "| Model | Changed predictions | Mean abs error | Max abs error | RMSE | Mean cosine | Min cosine |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model_id in S22_MODEL_ORDER:
        model = report["models"][model_id]
        metric = model["local_vs_s22_logits"]
        lines.append(
            f"| {labels[model_id]} | "
            f"{model['local_vs_s22_prediction_agreement']['disagreement_count']} | "
            f"{metric['mean_abs_error']:.16g} | {metric['max_abs_error']:.16g} | "
            f"{metric['rmse']:.16g} | {metric['mean_cosine_similarity']:.16g} | "
            f"{metric['min_cosine_similarity']:.16g} |"
        )

    fp32_changed = report["models"]["fp32"]["local_vs_s22_prediction_agreement"][
        "disagreement_original_indices"
    ]
    qdq_changed = report["models"]["qdq_int8"]["local_vs_s22_prediction_agreement"][
        "disagreement_original_indices"
    ]
    cross_changed = agreement["disagreement_original_indices"]
    lines.extend(
        [
            "",
            "## Prediction disagreements (original test_batch indices)",
            "",
            f"- FP32 local vs S22 ({len(fp32_changed)}): "
            f"`{', '.join(map(str, fp32_changed))}`",
            f"- QDQ INT8 local vs S22 ({len(qdq_changed)}): "
            f"`{', '.join(map(str, qdq_changed))}`",
            f"- S22 FP32 vs QDQ INT8 ({len(cross_changed)}): "
            f"`{', '.join(map(str, cross_changed))}`",
            "",
            "## Performance",
            "",
            "| Model | Mean latency | Speedup vs FP32 | Peak memory | Memory reduction | NPU nodes |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model_id in S22_MODEL_ORDER:
        perf = report["models"][model_id]["performance"]
        speedup = "baseline" if model_id == "fp32" else f"{perf['speedup_vs_fp32']:.3f}x"
        reduction = (
            "baseline"
            if model_id == "fp32"
            else f"{perf['memory_reduction_vs_fp32_percent']:.2f}%"
        )
        lines.append(
            f"| {labels[model_id]} | {perf['mean_latency_ms']:.5f} ms | {speedup} | "
            f"{perf['peak_memory_mib']:.3f} MiB | {reduction} | "
            f"{perf['npu_nodes']}/{perf['total_nodes']} |"
        )

    lines.extend(["", "## Conclusion boundaries", ""])
    lines.extend(
        f"- {statement}" for statement in report["interpretation"]["required_boundaries"]
    )
    lines.extend(
        [
            "",
            "`predictions.npz` contains labels, original indices, and four prediction arrays only. It contains no image or logit arrays.",
            "",
        ]
    )
    return "\n".join(lines)


def write_s22_full_accuracy_outputs(
    output_dir: Path,
    report: Mapping[str, Any],
    predictions: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Path]]:
    expected_keys = (
        "labels",
        "original_indices",
        "fp32_local_predictions",
        "fp32_s22_predictions",
        "qdq_int8_local_predictions",
        "qdq_int8_s22_predictions",
    )
    if tuple(predictions) != expected_keys:
        raise ValueError(f"full predictions must use exact keys {expected_keys}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": output_dir / "predictions.npz",
        "json": output_dir / "full_accuracy.json",
        "markdown": output_dir / "full_accuracy_summary.md",
    }
    write_deterministic_npz(paths["predictions"], predictions)
    payload = dict(report)
    payload["predictions_artifact"] = {
        "path": paths["predictions"].name,
        "size_bytes": paths["predictions"].stat().st_size,
        "sha256": sha256_file(paths["predictions"]),
        "arrays": {name: array_metadata(value) for name, value in predictions.items()},
        "contains_images": False,
        "contains_logits": False,
    }
    paths["json"].write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    paths["markdown"].write_text(s22_full_accuracy_markdown(payload), encoding="utf-8")
    return payload, paths
