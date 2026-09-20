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
    if labels.shape != (1000,) or labels.dtype != np.int64:
        raise ValueError(f"labels must be int64 (1000,), got {labels.shape}/{labels.dtype}")
    if original_indices.shape != (1000,) or original_indices.dtype != np.int64:
        raise ValueError(
            "original_indices must be int64 (1000,), got "
            f"{original_indices.shape}/{original_indices.dtype}"
        )
    if len(np.unique(original_indices)) != 1000 or np.any(np.diff(original_indices) <= 0):
        raise ValueError("original_indices must be unique and strictly increasing")
    if np.bincount(labels, minlength=10).tolist() != [100] * 10:
        raise ValueError("preflight labels must contain exactly 100 samples per class")
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
    for name, logits in (("local", local_logits), ("remote", remote_logits)):
        if logits.shape != (1000, 10) or logits.dtype != np.float32:
            raise ValueError(f"{name} logits must be float32 (1000, 10), got {logits.shape}/{logits.dtype}")
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
            "accuracy_change_vs_local_pp": 100.0
            * (remote_accuracy["top1_accuracy"] - local_accuracy["top1_accuracy"]),
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
