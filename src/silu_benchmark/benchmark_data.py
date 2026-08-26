"""Offline CIFAR-10 arrays for the ORT benchmark; no Torch/torchvision imports."""

import hashlib
import io
import importlib
import importlib.util
import pickle
from pathlib import Path

import numpy as np


# CIFAR-10 Python batch checksums, also used by torchvision.datasets.CIFAR10.
BATCH_MD5 = {
    "data_batch_1": "c99cafc152244af753f735de768cd75f",
    "test_batch": "40351d587109b95175f43aff81a1287e",
}
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def load_cifar_batch(root: Path, name: str):
    """Read only a verified official local batch; never download or execute arbitrary pickle."""
    if name not in BATCH_MD5:
        raise ValueError(f"unsupported frozen CIFAR-10 batch: {name}")
    path = root / "cifar-10-batches-py" / name
    raw = path.read_bytes()
    if hashlib.md5(raw).hexdigest() != BATCH_MD5[name]:
        raise ValueError(f"CIFAR-10 checksum mismatch: {path}")
    # The known checksum is checked before deserializing the trusted official batch.
    payload = pickle.load(io.BytesIO(raw), encoding="latin1")
    images = np.asarray(payload["data"])
    labels = np.asarray(payload["labels"], dtype=np.int64)
    if images.dtype != np.uint8 or images.shape != (10000, 3072) or labels.shape != (10000,):
        raise ValueError(f"unexpected CIFAR-10 shape/dtype: {path}")
    return images.reshape(-1, 3, 32, 32), labels


def normalize_cifar_images(images):
    """Same float32 divide/subtract/divide order as ToTensor + Normalize."""
    values = np.asarray(images, dtype=np.float32) / np.float32(255.0)
    values -= np.asarray(MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    values /= np.asarray(STD, dtype=np.float32).reshape(1, 3, 1, 1)
    return np.ascontiguousarray(values)


def numpy_batches(images, labels, indices, batch_size):
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    for offset in range(0, len(indices), batch_size):
        selected = indices[offset:offset + batch_size]
        yield normalize_cifar_images(images[selected]), labels[selected]


def load_ort_quantization():
    """Import ORT quantization without activating its unrelated optional Torch exporter.

    ORT 1.19.2 tools/__init__.py calls find_spec('torch') and eagerly imports
    pytorch_export_helpers. QDQ calibration needs neither that helper nor Torch.
    Scope this optional-feature selection to the single-threaded import, restore
    find_spec immediately, and do not catch/suppress any real import/runtime error.
    """
    original = importlib.util.find_spec

    def optional_spec(name, *args, **kwargs):
        return None if name == "torch" else original(name, *args, **kwargs)

    try:
        importlib.util.find_spec = optional_spec
        return importlib.import_module("onnxruntime.quantization")
    finally:
        importlib.util.find_spec = original


def build_standard_qdq(base, destination, data_root, samples, batch_size):
    """Keep the existing MinMax/QUInt8/per-channel QInt8 settings, now entirely in ORT."""
    import onnx
    quantization = load_ort_quantization()

    images, labels = load_cifar_batch(data_root, "data_batch_1")
    if samples > len(images) or samples <= 0:
        raise ValueError("invalid frozen QDQ calibration sample count")
    # The legacy reader selected the first 2,560 training images but shuffled batches.
    # MinMax sees that same subset, now in a recorded deterministic order.
    batches = iter(numpy_batches(images, labels, list(range(samples)), batch_size))
    input_name = onnx.load(str(base)).graph.input[0].name

    class Reader(quantization.CalibrationDataReader):
        def get_next(self):
            batch = next(batches, None)
            return None if batch is None else {input_name: batch[0]}

    quantization.quantize_static(
        model_input=str(base), model_output=str(destination), calibration_data_reader=Reader(),
        quant_format=quantization.QuantFormat.QDQ, activation_type=quantization.QuantType.QUInt8,
        weight_type=quantization.QuantType.QInt8, per_channel=True, reduce_range=False,
        calibrate_method=quantization.CalibrationMethod.MinMax,
    )
