"""Offline SiLU-aware calibration for portable one-scale ONNX QDQ.

The piecewise SiLU range is used only to guide an offline search.  The
selected runtime contract is an ordinary per-tensor affine uint8 quantizer:
one scale, one zero-point, QuantizeLinear, and DequantizeLinear.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import numpy as np

from .spec import PiecewiseQuantizationSpec


@dataclass(frozen=True)
class StandardQDQSpec:
    """Scalar parameters accepted by standard ONNX uint8 QDQ nodes."""

    scale: float
    zero_point: int
    bits: int = 8

    def __post_init__(self) -> None:
        if self.bits < 2 or self.bits > 16:
            raise ValueError("bits must be between 2 and 16")
        if not math.isfinite(float(self.scale)) or self.scale <= 0.0:
            raise ValueError("scale must be finite and positive")
        if not isinstance(self.zero_point, (int, np.integer)):
            raise TypeError("zero_point must be an integer")
        if not self.qmin <= int(self.zero_point) <= self.qmax:
            raise ValueError("zero_point is outside the unsigned code range")

    @property
    def qmin(self) -> int:
        return 0

    @property
    def qmax(self) -> int:
        return (1 << self.bits) - 1

    @property
    def representable_min(self) -> float:
        return (self.qmin - int(self.zero_point)) * float(self.scale)

    @property
    def representable_max(self) -> float:
        return (self.qmax - int(self.zero_point)) * float(self.scale)

    def to_manifest(self) -> dict:
        return {
            **asdict(self),
            "dtype": "uint8",
            "qmin": self.qmin,
            "qmax": self.qmax,
            "representable_min": self.representable_min,
            "representable_max": self.representable_max,
        }


def standard_qdq_quantize(values: np.ndarray, spec: StandardQDQSpec) -> np.ndarray:
    """Quantize with ONNX QuantizeLinear scalar semantics and ties-to-even."""

    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("values must be numeric")
    codes = np.rint(array.astype(np.float64) / spec.scale + spec.zero_point)
    return np.clip(codes, spec.qmin, spec.qmax).astype(np.uint16 if spec.bits > 8 else np.uint8)


def standard_qdq_dequantize(codes: np.ndarray, spec: StandardQDQSpec) -> np.ndarray:
    """Dequantize unsigned codes with ONNX DequantizeLinear semantics."""

    array = np.asarray(codes)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("codes must be integers")
    if np.any(array < spec.qmin) or np.any(array > spec.qmax):
        raise ValueError("codes are outside the quantizer range")
    return (array.astype(np.float64) - spec.zero_point) * spec.scale


def standard_qdq_quantize_dequantize(
    values: np.ndarray, spec: StandardQDQSpec
) -> np.ndarray:
    return standard_qdq_dequantize(standard_qdq_quantize(values, spec), spec)


def _validate_values(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("calibration values must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("calibration values must be finite")
    if not np.any(array < 0.0) or not np.any(array > 0.0):
        raise ValueError("SiLU calibration requires negative and positive values")
    return array


def _bounded_sample(values: np.ndarray, maximum: int) -> np.ndarray:
    if maximum <= 0:
        raise ValueError("max_samples must be positive")
    if values.size <= maximum:
        return values
    indices = np.linspace(0, values.size - 1, maximum, dtype=np.int64)
    return values[indices]


def _affine_spec_for_range(lower: float, upper: float, bits: int) -> StandardQDQSpec:
    if not math.isfinite(lower) or not math.isfinite(upper) or not lower < 0.0 < upper:
        raise ValueError("candidate range must be finite and straddle zero")
    qmax = (1 << bits) - 1
    # Persist exactly the scalar precision used by ONNX QDQ initializers so
    # calibration and runtime do not silently evaluate different grids.
    scale = float(np.float32((upper - lower) / qmax))
    zero_point = int(np.clip(np.rint(-lower / scale), 0, qmax))
    return StandardQDQSpec(scale=scale, zero_point=zero_point, bits=bits)


def _error_metrics(
    values: np.ndarray,
    reconstructed: np.ndarray,
    *,
    vsplit: float,
    central_weight: float,
) -> dict:
    error = reconstructed - values
    squared = error * error
    central = values < vsplit
    weights = np.where(central, central_weight, 1.0)
    return {
        "weighted_mse": float(np.average(squared, weights=weights)),
        "mse": float(np.mean(squared)),
        "mae": float(np.mean(np.abs(error))),
        "max_abs_error": float(np.max(np.abs(error))),
        "central_mse": float(np.mean(squared[central])) if np.any(central) else 0.0,
        "tail_mse": float(np.mean(squared[~central])) if np.any(~central) else 0.0,
    }


def calibrate_silu_aware_standard_qdq(
    values: np.ndarray,
    piecewise_hint: PiecewiseQuantizationSpec,
    *,
    bits: int = 8,
    lower_steps: int = 8,
    upper_steps: int = 24,
    central_weight: float = 2.0,
    max_samples: int = 200_000,
) -> dict:
    """Select one portable affine QDQ pair using offline SiLU range hints.

    ``Vmin``, ``Vsplit``, and ``Vmax`` influence candidate construction and
    the calibration objective only.  The returned deployment parameters do
    not contain a split or a second scale.
    """

    if not isinstance(piecewise_hint, PiecewiseQuantizationSpec):
        raise TypeError("piecewise_hint must be a PiecewiseQuantizationSpec")
    if bits != piecewise_hint.bits:
        raise ValueError("bits must match the piecewise calibration hint")
    if lower_steps < 2 or upper_steps < 2:
        raise ValueError("search grids require at least two steps")
    if not math.isfinite(central_weight) or central_weight < 1.0:
        raise ValueError("central_weight must be finite and at least 1.0")

    full_values = _validate_values(values)
    sample = _bounded_sample(full_values, max_samples)

    # The lower grid preserves the characteristic SiLU negative basin while
    # permitting controlled clipping.  The upper grid starts at Vsplit and
    # progressively admits the positive tail up to Vmax.
    lower_candidates = piecewise_hint.vmin * np.linspace(0.60, 1.0, lower_steps)
    upper_candidates = piecewise_hint.vsplit + (
        piecewise_hint.vmax - piecewise_hint.vsplit
    ) * np.linspace(0.10, 1.0, upper_steps)

    candidates: list[tuple[tuple[float, ...], StandardQDQSpec, dict]] = []
    for lower in lower_candidates:
        for upper in upper_candidates:
            spec = _affine_spec_for_range(float(lower), float(upper), bits)
            reconstructed = standard_qdq_quantize_dequantize(sample, spec)
            metrics = _error_metrics(
                sample,
                reconstructed,
                vsplit=piecewise_hint.vsplit,
                central_weight=central_weight,
            )
            clipped_fraction = float(
                np.mean(
                    (sample < spec.representable_min)
                    | (sample > spec.representable_max)
                )
            )
            key = (
                metrics["weighted_mse"],
                metrics["mse"],
                clipped_fraction,
                spec.scale,
                float(spec.zero_point),
            )
            candidates.append((key, spec, {**metrics, "clipped_fraction": clipped_fraction}))

    _key, selected, sample_metrics = min(candidates, key=lambda item: item[0])
    baseline = _affine_spec_for_range(piecewise_hint.vmin, piecewise_hint.vmax, bits)
    selected_full = standard_qdq_quantize_dequantize(full_values, selected)
    baseline_full = standard_qdq_quantize_dequantize(full_values, baseline)
    selected_metrics = _error_metrics(
        full_values,
        selected_full,
        vsplit=piecewise_hint.vsplit,
        central_weight=central_weight,
    )
    baseline_metrics = _error_metrics(
        full_values,
        baseline_full,
        vsplit=piecewise_hint.vsplit,
        central_weight=central_weight,
    )

    return {
        "schema_version": "silu-aware-standard-qdq-calibration/v1",
        "runtime_contract": {
            "quantize_op": "QuantizeLinear",
            "dequantize_op": "DequantizeLinear",
            "parameter_count": {"scale": 1, "zero_point": 1},
            "custom_runtime_nodes": 0,
            "piecewise_runtime_dispatch": False,
        },
        "selected_qdq": selected.to_manifest(),
        "baseline_qdq": baseline.to_manifest(),
        "offline_calibration": {
            "piecewise_hint": {
                "vmin": piecewise_hint.vmin,
                "vsplit": piecewise_hint.vsplit,
                "vmax": piecewise_hint.vmax,
                "bits": piecewise_hint.bits,
            },
            "central_region": "[Vmin, Vsplit)",
            "positive_tail": "[Vsplit, Vmax]",
            "central_weight": float(central_weight),
            "lower_steps": int(lower_steps),
            "upper_steps": int(upper_steps),
            "candidate_count": len(candidates),
            "input_count": int(full_values.size),
            "search_sample_count": int(sample.size),
        },
        "selected_error": selected_metrics,
        "baseline_error": baseline_metrics,
        "search_sample_error": sample_metrics,
        "weighted_mse_ratio_vs_minmax": (
            selected_metrics["weighted_mse"] / baseline_metrics["weighted_mse"]
            if baseline_metrics["weighted_mse"] > 0.0
            else 1.0
        ),
    }


def qdq_spec_from_manifest(payload: Mapping) -> StandardQDQSpec:
    """Load only the standard deployment parameters from a calibration record."""

    selected = payload.get("selected_qdq")
    if not isinstance(selected, Mapping):
        raise ValueError("manifest does not contain selected_qdq")
    return StandardQDQSpec(
        scale=float(selected["scale"]),
        zero_point=int(selected["zero_point"]),
        bits=int(selected.get("bits", 8)),
    )
