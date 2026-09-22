"""Quantization utilities; Torch implementations are loaded only on demand."""

from importlib import import_module

from .spec import PiecewiseQuantizationSpec


def __getattr__(name):
    modules = {
        "activation": {
            "ActivationQuantizerNCNN", "ActivationQuantizerSiLUAware", "BaseQuantizer",
            "piecewise_dequantize", "piecewise_quantize", "piecewise_quantize_dequantize",
        },
        "hardware_aware_qdq": {
            "StandardQDQSpec", "calibrate_silu_aware_standard_qdq",
            "qdq_spec_from_manifest", "standard_qdq_dequantize",
            "standard_qdq_quantize", "standard_qdq_quantize_dequantize",
        },
        "thresholds": {
            "compute_ncnn_threshold", "compute_silu_aware_thresholds",
            "mse_split_threshold", "ncnn_kld_threshold_optimized",
        },
        "weights": {"apply_bias_correction", "quantize_weights_per_channel"},
    }
    for module, names in modules.items():
        if name in names:
            value = getattr(import_module(f".{module}", __name__), name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ActivationQuantizerNCNN",
    "ActivationQuantizerSiLUAware",
    "BaseQuantizer",
    "PiecewiseQuantizationSpec",
    "StandardQDQSpec",
    "apply_bias_correction",
    "calibrate_silu_aware_standard_qdq",
    "compute_ncnn_threshold",
    "compute_silu_aware_thresholds",
    "mse_split_threshold",
    "ncnn_kld_threshold_optimized",
    "qdq_spec_from_manifest",
    "quantize_weights_per_channel",
    "piecewise_dequantize",
    "piecewise_quantize",
    "piecewise_quantize_dequantize",
    "standard_qdq_dequantize",
    "standard_qdq_quantize",
    "standard_qdq_quantize_dequantize",
]
