"""Deterministic local ORT accuracy evaluation for the three v1.6 QNN source graphs."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .benchmark_data import BATCH_MD5, MEAN, STD, load_cifar_batch, normalize_cifar_images


REPORT_SCHEMA = "qnn-local-cifar10-accuracy/v1.6"
DATASET_FINGERPRINT_SCHEMA = b"cifar10-ordered-samples/v1\0"
MODEL_ORDER = ("fp32", "qdq_int8", "piecewise_reference")
EXPECTED_INPUT = {"name": "images", "shape": ["batch_size", 3, 32, 32], "type": "tensor(float)"}
EXPECTED_OUTPUT = {"name": "logits", "shape": ["batch_size", 10], "type": "tensor(float)"}


def load_cifar10_test_set(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the verified official test batch used by all local QNN accuracy tools."""

    images, labels = load_cifar_batch(Path(data_root), "test_batch")
    if images.shape != (10000, 3, 32, 32) or labels.shape != (10000,):
        raise ValueError("CIFAR-10 test set must contain exactly 10,000 NCHW samples")
    return images, labels


def preprocessing_metadata(*, randomness: str) -> dict[str, Any]:
    """Describe the single shared training/history-compatible preprocessing path."""

    return {
        "source": "silu_benchmark.data.cifar10_transform / silu_benchmark.benchmark_data.normalize_cifar_images",
        "input_layout": "NCHW",
        "channel_order": "RGB",
        "source_dtype": "uint8",
        "output_dtype": "float32",
        "operations": [
            "cast uint8 to float32",
            "divide by float32(255.0)",
            "subtract per-channel float32 mean",
            "divide by per-channel float32 standard deviation",
        ],
        "mean": list(MEAN),
        "standard_deviation": list(STD),
        "random_seed": None,
        "randomness": randomness,
    }


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_sha256(path: Path, expected: str, *, label: str) -> str:
    actual = sha256_file(path)
    normalized = str(expected).upper()
    if actual != normalized:
        raise ValueError(f"{label} SHA256 mismatch: expected {normalized}, got {actual}")
    return actual


def label_sha256(labels: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(labels, dtype="<i8"))
    return sha256_bytes(canonical.tobytes(order="C"))


def input_order_sha256(images: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(images, dtype=np.uint8))
    return sha256_bytes(canonical.tobytes(order="C"))


def ordered_dataset_fingerprint(images: np.ndarray, labels: np.ndarray) -> str:
    images = np.ascontiguousarray(np.asarray(images, dtype=np.uint8))
    labels = np.ascontiguousarray(np.asarray(labels, dtype="<i8"))
    if images.ndim != 4 or images.shape[0] != labels.shape[0]:
        raise ValueError("ordered dataset fingerprint requires aligned NCHW images and labels")
    digest = hashlib.sha256()
    digest.update(DATASET_FINGERPRINT_SCHEMA)
    digest.update(np.asarray(images.shape, dtype="<i8").tobytes())
    digest.update(images.tobytes(order="C"))
    digest.update(labels.tobytes(order="C"))
    return digest.hexdigest().upper()


def _node_metadata(node: Any) -> dict[str, Any]:
    return {"name": str(node.name), "shape": list(node.shape), "type": str(node.type)}


def validate_model_io(session: Any) -> dict[str, Any]:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            f"model must expose exactly one input and one output, got {len(inputs)}/{len(outputs)}"
        )
    actual_input = _node_metadata(inputs[0])
    actual_output = _node_metadata(outputs[0])
    if actual_input != EXPECTED_INPUT:
        raise ValueError(f"unexpected model input contract: {actual_input}")
    if actual_output != EXPECTED_OUTPUT:
        raise ValueError(f"unexpected model output contract: {actual_output}")
    providers = list(session.get_providers())
    if providers != ["CPUExecutionProvider"]:
        raise ValueError(f"ORT session must use only CPUExecutionProvider, got {providers}")
    return {"inputs": [actual_input], "outputs": [actual_output], "providers": providers}


def accuracy_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape or labels.ndim != 1 or not labels.size:
        raise ValueError("labels and predictions must be aligned non-empty vectors")
    errors = np.flatnonzero(predictions != labels).astype(np.int64)
    correct = int(labels.size - errors.size)
    return {
        "correct": correct,
        "total": int(labels.size),
        "top1_accuracy": correct / int(labels.size),
        "top1_accuracy_percent": correct * 100.0 / int(labels.size),
        "misclassified_count": int(errors.size),
        "misclassified_indices": errors.tolist(),
    }


def prediction_agreement(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.int64)
    candidate = np.asarray(candidate, dtype=np.int64)
    if reference.shape != candidate.shape or reference.ndim != 1 or not reference.size:
        raise ValueError("prediction vectors must be aligned and non-empty")
    disagreements = np.flatnonzero(reference != candidate).astype(np.int64)
    agreement_count = int(reference.size - disagreements.size)
    return {
        "agreement_count": agreement_count,
        "disagreement_count": int(disagreements.size),
        "agreement": agreement_count / int(reference.size),
        "agreement_percent": agreement_count * 100.0 / int(reference.size),
        "disagreement_indices": disagreements.tolist(),
    }


def evaluate_session(
    session: Any,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
) -> tuple[dict[str, Any], np.ndarray]:
    images = np.asarray(images)
    labels = np.asarray(labels, dtype=np.int64)
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if images.shape != (len(labels), 3, 32, 32) or images.dtype != np.uint8:
        raise ValueError(f"expected uint8 NCHW CIFAR-10 images, got {images.shape}/{images.dtype}")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    predictions = np.full(len(labels), -1, dtype=np.int64)
    nonfinite_logit_count = 0
    nonfinite_sample_indices: list[int] = []
    failed_sample_indices: list[int] = []
    inference_failures: list[dict[str, Any]] = []

    for offset in range(0, len(labels), batch_size):
        stop = min(offset + batch_size, len(labels))
        inputs = normalize_cifar_images(images[offset:stop])
        try:
            logits = np.asarray(session.run([output_name], {input_name: inputs})[0])
            if logits.shape != (stop - offset, 10) or logits.dtype != np.float32:
                raise ValueError(f"unexpected logits shape/dtype: {logits.shape}/{logits.dtype}")
        except Exception as error:  # Retain counts while keeping the report credential-safe.
            failed_sample_indices.extend(range(offset, stop))
            inference_failures.append(
                {
                    "batch_start_index": offset,
                    "sample_count": stop - offset,
                    "error_type": type(error).__name__,
                }
            )
            continue

        finite = np.isfinite(logits)
        nonfinite_logit_count += int(np.size(finite) - np.count_nonzero(finite))
        finite_samples = np.all(finite, axis=1)
        bad_local = np.flatnonzero(~finite_samples)
        nonfinite_sample_indices.extend((bad_local + offset).astype(np.int64).tolist())
        if np.any(finite_samples):
            local_predictions = np.argmax(logits[finite_samples], axis=1).astype(np.int64)
            predictions[np.flatnonzero(finite_samples) + offset] = local_predictions

    metrics = accuracy_metrics(labels, predictions)
    metrics.update(
        {
            "inference_failure_batch_count": len(inference_failures),
            "inference_failure_sample_count": len(failed_sample_indices),
            "inference_failure_sample_indices": failed_sample_indices,
            "inference_failures": inference_failures,
            "nonfinite_logit_count": nonfinite_logit_count,
            "nonfinite_sample_count": len(nonfinite_sample_indices),
            "nonfinite_sample_indices": nonfinite_sample_indices,
        }
    )
    return metrics, predictions


def pairwise_comparisons(predictions: Mapping[str, np.ndarray]) -> dict[str, Any]:
    if tuple(predictions) != MODEL_ORDER:
        raise ValueError(f"predictions must use ordered model IDs {MODEL_ORDER}")
    return {
        f"{left}_vs_{right}": {
            "reference_model": left,
            "candidate_model": right,
            **prediction_agreement(predictions[left], predictions[right]),
        }
        for left, right in combinations(MODEL_ORDER, 2)
    }


def build_report(
    *,
    manifest_path: str,
    manifest_sha256: str,
    data_root: str,
    test_batch_path: str,
    test_batch_sha256: str,
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    ort_version: str,
    available_providers: Sequence[str],
    model_results: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    if tuple(model_results) != MODEL_ORDER or tuple(predictions) != MODEL_ORDER:
        raise ValueError(f"model results and predictions must use ordered model IDs {MODEL_ORDER}")
    fp32_accuracy = float(model_results["fp32"]["top1_accuracy"])
    models: dict[str, Any] = {}
    for model_id in MODEL_ORDER:
        result = dict(model_results[model_id])
        agreement = prediction_agreement(predictions["fp32"], predictions[model_id])
        result.update(
            {
                "accuracy_change_vs_fp32_pp": 100.0
                * (float(result["top1_accuracy"]) - fp32_accuracy),
                "prediction_agreement_vs_fp32": agreement["agreement"],
                "prediction_agreement_vs_fp32_percent": agreement["agreement_percent"],
                "prediction_disagreement_vs_fp32_count": agreement["disagreement_count"],
                "prediction_disagreement_vs_fp32_indices": agreement["disagreement_indices"],
            }
        )
        models[model_id] = result

    report = {
        "schema_version": REPORT_SCHEMA,
        "scope": "local ONNX Runtime CPU CIFAR-10 accuracy",
        "completion_status": "success"
        if all(
            model["inference_failure_sample_count"] == 0
            and model["nonfinite_logit_count"] == 0
            for model in models.values()
        )
        else "completed-with-invalid-outputs",
        "statements": {
            "is_galaxy_s22_qnn_accuracy": False,
            "piecewise_reference_is_fixed": True,
            "piecewise_reference_replacement_allowed": False,
            "ai_hub_tasks_created": False,
        },
        "provenance": {
            "manifest_path": manifest_path,
            "manifest_sha256": manifest_sha256,
        },
        "dataset": {
            "name": "CIFAR-10",
            "split": "test",
            "distribution": "CIFAR-10 python version (cifar-10-python.tar.gz)",
            "batch_file": "test_batch",
            "batch_path": test_batch_path,
            "batch_md5": BATCH_MD5["test_batch"],
            "batch_sha256": test_batch_sha256,
            "data_root": data_root,
            "sample_count": int(len(labels)),
            "labels_dtype_for_hash": "little-endian int64, C order",
            "labels_sha256": label_sha256(labels),
            "input_order_dtype_for_hash": "uint8 NCHW, C order",
            "input_order_sha256": input_order_sha256(images),
            "ordered_dataset_fingerprint_schema": DATASET_FINGERPRINT_SCHEMA.rstrip(b"\0").decode(),
            "ordered_dataset_fingerprint_sha256": ordered_dataset_fingerprint(images, labels),
            "order": "official test_batch record order, zero-based indices 0..9999",
        },
        "preprocessing": preprocessing_metadata(
            randomness="none; shuffle is disabled and all 10,000 records are evaluated in file order"
        ),
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "onnxruntime": str(ort_version),
            "requested_providers": ["CPUExecutionProvider"],
            "available_providers": list(available_providers),
            "batch_size": int(batch_size),
            "platform": platform.platform(),
            "byte_order": sys.byteorder,
        },
        "models": models,
        "pairwise_prediction_comparisons": pairwise_comparisons(predictions),
        "index_convention": "All sample indices are zero-based positions in official test_batch order.",
    }
    return report


def _index_text(indices: Sequence[int]) -> str:
    return ", ".join(str(index) for index in indices) if indices else "(none)"


def accuracy_markdown(report: Mapping[str, Any]) -> str:
    labels = {
        "fp32": "FP32",
        "qdq_int8": "QDQ INT8",
        "piecewise_reference": "Piecewise reference",
    }
    lines = [
        "# ResNet18-SiLU local CIFAR-10 accuracy - v1.6 QNN source models",
        "",
        "**Scope:** This is local ONNX Runtime `CPUExecutionProvider` accuracy, not Galaxy S22 QNN accuracy.",
        "",
        "`piecewise_v065` is the fixed reference graph for this evaluation. Its measured accuracy is reported as-is and the model is not silently replaced. No Qualcomm AI Hub task was created in this stage.",
        "",
        "## Accuracy and FP32 agreement",
        "",
        "| Model | Correct / total | Top-1 | Change vs FP32 (pp) | Disagree vs FP32 | Agreement vs FP32 | Inference-failed samples | Non-finite logits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_id in MODEL_ORDER:
        row = report["models"][model_id]
        lines.append(
            f"| {labels[model_id]} | {row['correct']} / {row['total']} | "
            f"{row['top1_accuracy_percent']:.4f}% | {row['accuracy_change_vs_fp32_pp']:+.4f} | "
            f"{row['prediction_disagreement_vs_fp32_count']} | "
            f"{row['prediction_agreement_vs_fp32_percent']:.4f}% | "
            f"{row['inference_failure_sample_count']} | {row['nonfinite_logit_count']} |"
        )

    dataset = report["dataset"]
    runtime = report["runtime"]
    preprocessing = report["preprocessing"]
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Dataset: {dataset['distribution']}, `{dataset['batch_file']}`, {dataset['sample_count']} samples in official file order.",
            f"- Test batch MD5: `{dataset['batch_md5']}`; SHA256: `{dataset['batch_sha256']}`.",
            f"- Labels SHA256: `{dataset['labels_sha256']}`.",
            f"- Input-order SHA256: `{dataset['input_order_sha256']}`.",
            f"- Ordered dataset fingerprint: `{dataset['ordered_dataset_fingerprint_sha256']}`.",
            f"- Preprocessing: RGB NCHW float32, divide by 255, mean `{preprocessing['mean']}`, standard deviation `{preprocessing['standard_deviation']}`.",
            "- Random seed: not applicable; no randomized operation is used.",
            f"- ONNX Runtime: `{runtime['onnxruntime']}`; requested/active provider: `CPUExecutionProvider`; batch size: `{runtime['batch_size']}`.",
            "",
            "## Model contracts",
            "",
        ]
    )
    for model_id in MODEL_ORDER:
        model = report["models"][model_id]
        io = model["io"]
        lines.append(
            f"- {labels[model_id]}: `{model['path']}`; SHA256 `{model['sha256']}`; "
            f"input `{io['inputs'][0]['name']}` `{io['inputs'][0]['shape']}` `{io['inputs'][0]['type']}`; "
            f"output `{io['outputs'][0]['name']}` `{io['outputs'][0]['shape']}` `{io['outputs'][0]['type']}`."
        )

    lines.extend(["", "## Pairwise prediction comparison", ""])
    for comparison in report["pairwise_prediction_comparisons"].values():
        lines.append(
            f"- `{comparison['reference_model']}` vs `{comparison['candidate_model']}`: "
            f"{comparison['agreement_count']} / {report['dataset']['sample_count']} agree "
            f"({comparison['agreement_percent']:.4f}%); {comparison['disagreement_count']} disagree."
        )

    lines.extend(["", "## Error sample indices", ""])
    for model_id in MODEL_ORDER:
        row = report["models"][model_id]
        lines.extend(
            [
                f"### {labels[model_id]} ({row['misclassified_count']} errors)",
                "",
                f"`{_index_text(row['misclassified_indices'])}`",
                "",
            ]
        )
    lines.extend(
        [
            "All indices above are zero-based positions in `test_batch`. Full predictions, labels, error indices, and pairwise disagreement indices are also stored in `predictions.npz`; no CIFAR-10 image is stored in the outputs.",
            "",
        ]
    )
    return "\n".join(lines)


def write_accuracy_outputs(
    output_dir: Path,
    report: Mapping[str, Any],
    predictions: Mapping[str, np.ndarray],
    labels: np.ndarray,
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "local_accuracy.json"
    markdown_path = output_dir / "local_accuracy_summary.md"
    predictions_path = output_dir / "predictions.npz"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(accuracy_markdown(report), encoding="utf-8")
    archive: dict[str, np.ndarray] = {"labels": np.asarray(labels, dtype=np.int64)}
    for model_id in MODEL_ORDER:
        values = np.asarray(predictions[model_id], dtype=np.int64)
        archive[f"{model_id}_predictions"] = values
        archive[f"{model_id}_misclassified_indices"] = np.flatnonzero(values != labels).astype(np.int64)
    for key, comparison in report["pairwise_prediction_comparisons"].items():
        archive[f"{key}_disagreement_indices"] = np.asarray(
            comparison["disagreement_indices"], dtype=np.int64
        )
    np.savez_compressed(predictions_path, **archive)
    return json_path, markdown_path, predictions_path
