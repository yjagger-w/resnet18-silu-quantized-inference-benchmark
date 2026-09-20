"""Offline CIFAR-10 preflight export for later Galaxy S22 QNN inference."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .qnn_local_accuracy import (
    accuracy_metrics,
    normalize_cifar_images,
    ordered_dataset_fingerprint,
    prediction_agreement,
    preprocessing_metadata,
    sha256_file,
)


PREFLIGHT_SCHEMA = "qnn-cifar10-s22-preflight/v1.6"
PREFLIGHT_MODEL_ORDER = ("fp32", "qdq_int8")
NPZ_FILENAMES = ("inputs.npz", "labels.npz", "local_reference.npz")
ARRAY_HASH_SCHEMA = b"canonical-ndarray/v1\0"


def canonical_array_sha256(value: np.ndarray) -> str:
    """Hash dtype, shape, and C-order bytes using a fixed little-endian representation."""

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("object arrays are not supported")
    canonical_dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
    digest = hashlib.sha256()
    digest.update(ARRAY_HASH_SCHEMA)
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest().upper()


def array_metadata(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "canonical_sha256": canonical_array_sha256(array),
        "finite": bool(np.all(np.isfinite(array))) if np.issubdtype(array.dtype, np.number) else None,
    }


def select_balanced_original_indices(
    labels: np.ndarray,
    *,
    samples_per_class: int = 100,
    class_count: int = 10,
) -> np.ndarray:
    """Select each class's first N records while preserving global file order."""

    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or not labels.size:
        raise ValueError("labels must be a non-empty vector")
    if samples_per_class <= 0 or class_count <= 0:
        raise ValueError("samples_per_class and class_count must be positive")
    if np.any(labels < 0) or np.any(labels >= class_count):
        raise ValueError(f"labels must be in [0, {class_count - 1}]")
    counts = np.zeros(class_count, dtype=np.int64)
    selected: list[int] = []
    for original_index, label in enumerate(labels):
        label = int(label)
        if counts[label] < samples_per_class:
            selected.append(original_index)
            counts[label] += 1
        if np.all(counts == samples_per_class):
            break
    if not np.all(counts == samples_per_class):
        raise ValueError(
            f"dataset cannot provide {samples_per_class} samples for every class: {counts.tolist()}"
        )
    indices = np.asarray(selected, dtype=np.int64)
    if len(np.unique(indices)) != len(indices) or np.any(np.diff(indices) <= 0):
        raise RuntimeError("balanced selection must contain unique strictly increasing indices")
    return indices


def prepare_preflight_subset(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    samples_per_class: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized inputs, labels, and source indices for the balanced subset."""

    images = np.asarray(images)
    labels = np.asarray(labels, dtype=np.int64)
    if images.shape != (len(labels), 3, 32, 32) or images.dtype != np.uint8:
        raise ValueError(f"expected aligned uint8 NCHW images, got {images.shape}/{images.dtype}")
    indices = select_balanced_original_indices(labels, samples_per_class=samples_per_class)
    selected_labels = np.ascontiguousarray(labels[indices], dtype=np.int64)
    selected_images = images[indices]
    inputs = normalize_cifar_images(selected_images)
    expected_total = samples_per_class * 10
    counts = np.bincount(selected_labels, minlength=10)
    if inputs.shape != (expected_total, 3, 32, 32) or inputs.dtype != np.float32:
        raise RuntimeError(f"unexpected normalized input contract: {inputs.shape}/{inputs.dtype}")
    if counts.tolist() != [samples_per_class] * 10:
        raise RuntimeError(f"unexpected selected class counts: {counts.tolist()}")
    if not np.array_equal(selected_labels, labels[indices]):
        raise RuntimeError("selected input and label order diverged")
    if not np.all(np.isfinite(inputs)):
        raise ValueError("preflight inputs contain non-finite values")
    return inputs, selected_labels, indices


def evaluate_preprocessed_session(
    session: Any,
    inputs: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    inputs = np.asarray(inputs)
    labels = np.asarray(labels, dtype=np.int64)
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if inputs.shape != (len(labels), 3, 32, 32) or inputs.dtype != np.float32:
        raise ValueError(f"expected aligned float32 NCHW inputs, got {inputs.shape}/{inputs.dtype}")
    if not np.all(np.isfinite(inputs)):
        raise ValueError("preflight inputs contain non-finite values")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    parts = []
    for offset in range(0, len(labels), batch_size):
        batch = np.ascontiguousarray(inputs[offset : offset + batch_size])
        logits = np.asarray(session.run([output_name], {input_name: batch})[0])
        if logits.shape != (len(batch), 10) or logits.dtype != np.float32:
            raise ValueError(f"unexpected logits shape/dtype: {logits.shape}/{logits.dtype}")
        if not np.all(np.isfinite(logits)):
            raise ValueError("model produced non-finite logits")
        parts.append(np.ascontiguousarray(logits))
    combined = np.concatenate(parts, axis=0)
    predictions = np.argmax(combined, axis=1).astype(np.int64)
    return accuracy_metrics(labels, predictions), predictions, combined


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    if not arrays:
        raise ValueError("NPZ must contain at least one array")
    destination = io.BytesIO()
    with zipfile.ZipFile(
        destination, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, value in arrays.items():
            if not name or "/" in name or "\\" in name:
                raise ValueError(f"invalid NPZ key: {name!r}")
            array = np.ascontiguousarray(np.asarray(value))
            if array.dtype.hasobject:
                raise ValueError(f"object array is not allowed: {name}")
            payload = io.BytesIO()
            np.lib.format.write_array(payload, array, allow_pickle=False)
            entry = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.create_system = 0
            archive.writestr(entry, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return destination.getvalue()


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_deterministic_npz_bytes(arrays))
    return path


def _artifact_record(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "arrays": {name: array_metadata(value) for name, value in arrays.items()},
    }


def preflight_markdown(manifest: Mapping[str, Any]) -> str:
    selection = manifest["selection"]
    comparison = manifest["prediction_comparison"]
    lines = [
        "# Galaxy S22 QNN CIFAR-10 preflight export",
        "",
        "This stage only exports deterministic local data and ONNX Runtime CPU references. It does not connect to Qualcomm AI Hub, create a remote task, or measure Galaxy S22 QNN accuracy.",
        "",
        "## Selection",
        "",
        f"- Rule: {selection['rule']}",
        f"- Samples: {selection['total_samples']} ({selection['samples_per_class']} per class).",
        "- Output order: strictly increasing zero-based positions from the official `test_batch`.",
        "",
        "## Local ORT reference",
        "",
        "| Model | Correct / total | Top-1 |",
        "|---|---:|---:|",
    ]
    for model_id in PREFLIGHT_MODEL_ORDER:
        row = manifest["models"][model_id]
        lines.append(
            f"| {row['label']} | {row['correct']} / {row['total']} | "
            f"{row['top1_accuracy_percent']:.4f}% |"
        )
    lines.extend(
        [
            "",
            f"FP32/QDQ prediction agreement: {comparison['agreement_count']} / "
            f"{selection['total_samples']} ({comparison['agreement_percent']:.4f}%).",
            "",
            "## Suggested existing compile jobs",
            "",
            f"- FP32: `{manifest['models']['fp32']['compile_job_id']}`",
            f"- QDQ INT8: `{manifest['models']['qdq_int8']['compile_job_id']}`",
            "",
            "## Generated artifacts",
            "",
            "| File | Size (bytes) | SHA256 | Keys |",
            "|---|---:|---|---|",
        ]
    )
    for artifact in manifest["artifacts"].values():
        lines.append(
            f"| `{artifact['path']}` | {artifact['size_bytes']} | `{artifact['sha256']}` | "
            f"{', '.join(artifact['arrays'])} |"
        )
    lines.extend(
        [
            "",
            "`inputs.npz` contains normalized float32 inputs only. No raw CIFAR-10 image is exported.",
            "",
        ]
    )
    return "\n".join(lines)


def write_preflight_outputs(
    output_dir: Path,
    *,
    inputs: np.ndarray,
    labels: np.ndarray,
    original_indices: np.ndarray,
    local_reference: Mapping[str, np.ndarray],
    manifest_base: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Path]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archives = {
        "inputs": {"images": np.asarray(inputs, dtype=np.float32)},
        "labels": {
            "labels": np.asarray(labels, dtype=np.int64),
            "original_indices": np.asarray(original_indices, dtype=np.int64),
        },
        "local_reference": {
            name: np.asarray(value) for name, value in local_reference.items()
        },
    }
    paths = {
        "inputs": output_dir / "inputs.npz",
        "labels": output_dir / "labels.npz",
        "local_reference": output_dir / "local_reference.npz",
        "manifest": output_dir / "preflight_manifest.json",
        "summary": output_dir / "preflight_summary.md",
    }
    for key in ("inputs", "labels", "local_reference"):
        write_deterministic_npz(paths[key], archives[key])
    manifest = dict(manifest_base)
    manifest["artifacts"] = {
        key: _artifact_record(paths[key], archives[key])
        for key in ("inputs", "labels", "local_reference")
    }
    repeated_hashes_match = all(
        hashlib.sha256(_deterministic_npz_bytes(archives[key])).hexdigest().upper()
        == manifest["artifacts"][key]["sha256"]
        for key in ("inputs", "labels", "local_reference")
    )
    validations = dict(manifest.get("validations", {}))
    validations.update(
        {
            "npz_repeat_generation_file_hashes_match": repeated_hashes_match,
            "all_exported_arrays_match_recorded_canonical_hashes": all(
                array_metadata(value)["canonical_sha256"]
                == manifest["artifacts"][key]["arrays"][name]["canonical_sha256"]
                for key, arrays in archives.items()
                for name, value in arrays.items()
            ),
            "raw_uint8_images_exported": False,
        }
    )
    if not all(value for key, value in validations.items() if key != "raw_uint8_images_exported"):
        raise RuntimeError(f"preflight validation failed: {validations}")
    manifest["validations"] = validations
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    paths["summary"].write_text(preflight_markdown(manifest), encoding="utf-8")
    return manifest, paths


def selection_manifest(
    full_images: np.ndarray,
    full_labels: np.ndarray,
    selected_labels: np.ndarray,
    original_indices: np.ndarray,
    *,
    samples_per_class: int,
) -> dict[str, Any]:
    counts = np.bincount(selected_labels, minlength=10)
    return {
        "rule": (
            "scan official test_batch in original order and select a record while its class "
            f"has fewer than {samples_per_class} selected samples"
        ),
        "samples_per_class": samples_per_class,
        "total_samples": int(len(original_indices)),
        "class_counts": {str(label): int(count) for label, count in enumerate(counts)},
        "original_indices": np.asarray(original_indices, dtype=np.int64).tolist(),
        "indices_are_unique": len(np.unique(original_indices)) == len(original_indices),
        "indices_are_strictly_increasing": bool(np.all(np.diff(original_indices) > 0)),
        "labels_match_original_order": bool(
            np.array_equal(selected_labels, np.asarray(full_labels)[original_indices])
        ),
        "full_dataset_fingerprint_sha256": ordered_dataset_fingerprint(full_images, full_labels),
        "selected_raw_dataset_fingerprint_sha256": ordered_dataset_fingerprint(
            np.asarray(full_images)[original_indices], selected_labels
        ),
    }
