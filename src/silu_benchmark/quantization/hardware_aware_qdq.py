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


def _clipped_fraction(values: np.ndarray, spec: StandardQDQSpec) -> float:
    return float(
        np.mean(
            (values < spec.representable_min)
            | (values > spec.representable_max)
        )
    )


def _evaluate_spec(
    values: np.ndarray,
    spec: StandardQDQSpec,
    *,
    vsplit: float,
    central_weight: float,
) -> dict:
    reconstructed = standard_qdq_quantize_dequantize(values, spec)
    return {
        **_error_metrics(
            values,
            reconstructed,
            vsplit=vsplit,
            central_weight=central_weight,
        ),
        "clipped_fraction": _clipped_fraction(values, spec),
    }


def _source_anchored_specs(
    source_spec: StandardQDQSpec,
    *,
    scale_ratio_min: float,
    scale_ratio_max: float,
    max_zero_point_delta: int,
    scale_steps: int,
    zero_point_steps: int,
) -> list[StandardQDQSpec]:
    ratios = np.linspace(scale_ratio_min, scale_ratio_max, scale_steps)
    deltas = np.rint(
        np.linspace(-max_zero_point_delta, max_zero_point_delta, zero_point_steps)
    ).astype(np.int64)
    ratios = np.unique(np.concatenate([ratios, np.asarray([1.0])]))
    deltas = np.unique(np.concatenate([deltas, np.asarray([0], dtype=np.int64)]))

    unique: dict[tuple[float, int], StandardQDQSpec] = {}
    for ratio in ratios:
        scale = float(np.float32(source_spec.scale * float(ratio)))
        for delta in deltas:
            zero_point = int(source_spec.zero_point + int(delta))
            if not source_spec.qmin <= zero_point <= source_spec.qmax:
                continue
            spec = StandardQDQSpec(
                scale=scale,
                zero_point=zero_point,
                bits=source_spec.bits,
            )
            unique[(spec.scale, spec.zero_point)] = spec

    # Preserve the exact float32 source encoding even when linspace rounding
    # would otherwise produce a numerically adjacent value.
    unique[(source_spec.scale, source_spec.zero_point)] = source_spec
    return list(unique.values())


def calibrate_silu_aware_standard_qdq(
    values: np.ndarray,
    piecewise_hint: PiecewiseQuantizationSpec,
    *,
    bits: int = 8,
    lower_steps: int = 8,
    upper_steps: int = 24,
    central_weight: float = 2.0,
    max_samples: int = 200_000,
    source_qdq: StandardQDQSpec | None = None,
    source_scale_ratio_min: float = 0.90,
    source_scale_ratio_max: float = 1.10,
    max_zero_point_delta: int = 4,
    minimum_relative_improvement: float = 0.005,
) -> dict:
    """Select one portable affine QDQ pair using offline SiLU range hints.

    ``Vmin``, ``Vsplit``, and ``Vmax`` influence the calibration objective
    only when ``source_qdq`` is supplied. Candidate encodings are then
    constrained around the source model's device-validated QDQ parameters.
    If no bounded candidate materially improves the full calibration
    objective, the exact source encoding is selected. The returned deployment
    parameters never contain a split or a second scale.
    """

    if not isinstance(piecewise_hint, PiecewiseQuantizationSpec):
        raise TypeError("piecewise_hint must be a PiecewiseQuantizationSpec")
    if bits != piecewise_hint.bits:
        raise ValueError("bits must match the piecewise calibration hint")
    if lower_steps < 2 or upper_steps < 2:
        raise ValueError("search grids require at least two steps")
    if not math.isfinite(central_weight) or central_weight < 1.0:
        raise ValueError("central_weight must be finite and at least 1.0")
    if source_qdq is not None:
        if not isinstance(source_qdq, StandardQDQSpec):
            raise TypeError("source_qdq must be a StandardQDQSpec")
        if source_qdq.bits != bits:
            raise ValueError("source_qdq bits must match the requested bits")
        if (
            not math.isfinite(source_scale_ratio_min)
            or not math.isfinite(source_scale_ratio_max)
            or not 0.0 < source_scale_ratio_min <= 1.0 <= source_scale_ratio_max
        ):
            raise ValueError("source scale ratio bounds must be finite and include 1.0")
        if not isinstance(max_zero_point_delta, (int, np.integer)):
            raise TypeError("max_zero_point_delta must be an integer")
        if max_zero_point_delta < 0:
            raise ValueError("max_zero_point_delta must not be negative")
        if (
            not math.isfinite(minimum_relative_improvement)
            or not 0.0 <= minimum_relative_improvement < 1.0
        ):
            raise ValueError(
                "minimum_relative_improvement must be finite and in [0, 1)"
            )

    full_values = _validate_values(values)
    sample = _bounded_sample(full_values, max_samples)

    candidates: list[tuple[tuple[float, ...], StandardQDQSpec, dict]] = []
    minmax = _affine_spec_for_range(piecewise_hint.vmin, piecewise_hint.vmax, bits)
    if source_qdq is None:
        # Backward-compatible research path. Production v1.8 orchestration
        # supplies source_qdq and never uses this unanchored candidate grid.
        lower_candidates = piecewise_hint.vmin * np.linspace(0.60, 1.0, lower_steps)
        upper_candidates = piecewise_hint.vsplit + (
            piecewise_hint.vmax - piecewise_hint.vsplit
        ) * np.linspace(0.10, 1.0, upper_steps)
        candidate_specs = [
            _affine_spec_for_range(float(lower), float(upper), bits)
            for lower in lower_candidates
            for upper in upper_candidates
        ]
        source = minmax
        search_mode = "legacy_unanchored"
    else:
        source = source_qdq
        candidate_specs = _source_anchored_specs(
            source,
            scale_ratio_min=source_scale_ratio_min,
            scale_ratio_max=source_scale_ratio_max,
            max_zero_point_delta=max_zero_point_delta,
            scale_steps=upper_steps,
            zero_point_steps=lower_steps,
        )
        search_mode = "source_anchored_bounded"

    for spec in candidate_specs:
        metrics = _evaluate_spec(
            sample,
            spec,
            vsplit=piecewise_hint.vsplit,
            central_weight=central_weight,
        )
        scale_distance = abs(math.log(spec.scale / source.scale))
        zero_point_distance = abs(spec.zero_point - source.zero_point)
        key = (
            metrics["weighted_mse"],
            metrics["mse"],
            metrics["clipped_fraction"],
            scale_distance,
            float(zero_point_distance),
            spec.scale,
            float(spec.zero_point),
        )
        candidates.append((key, spec, metrics))

    _key, proposed, proposed_sample_metrics = min(candidates, key=lambda item: item[0])
    proposed_metrics = _evaluate_spec(
        full_values,
        proposed,
        vsplit=piecewise_hint.vsplit,
        central_weight=central_weight,
    )
    source_metrics = _evaluate_spec(
        full_values,
        source,
        vsplit=piecewise_hint.vsplit,
        central_weight=central_weight,
    )
    source_weighted_mse = source_metrics["weighted_mse"]
    relative_improvement = (
        (source_weighted_mse - proposed_metrics["weighted_mse"])
        / source_weighted_mse
        if source_weighted_mse > 0.0
        else 0.0
    )
    proposed_is_source = proposed == source
    fallback_to_source = source_qdq is not None and (
        proposed_is_source or relative_improvement < minimum_relative_improvement
    )
    if fallback_to_source:
        selected = source
        selected_metrics = source_metrics
        sample_metrics = _evaluate_spec(
            sample,
            source,
            vsplit=piecewise_hint.vsplit,
            central_weight=central_weight,
        )
        selection_reason = (
            "source_already_best"
            if proposed_is_source
            else "candidate_improvement_below_minimum"
        )
    else:
        selected = proposed
        selected_metrics = proposed_metrics
        sample_metrics = proposed_sample_metrics
        selection_reason = (
            "bounded_candidate_improved_weighted_mse"
            if source_qdq is not None
            else "legacy_unanchored_objective_minimum"
        )

    minmax_metrics = _evaluate_spec(
        full_values,
        minmax,
        vsplit=piecewise_hint.vsplit,
        central_weight=central_weight,
    )
    selected_scale_ratio = selected.scale / source.scale
    selected_zero_point_delta = selected.zero_point - source.zero_point

    return {
        "schema_version": "silu-aware-standard-qdq-calibration/v2",
        "runtime_contract": {
            "quantize_op": "QuantizeLinear",
            "dequantize_op": "DequantizeLinear",
            "parameter_count": {"scale": 1, "zero_point": 1},
            "custom_runtime_nodes": 0,
            "piecewise_runtime_dispatch": False,
        },
        "selected_qdq": selected.to_manifest(),
        "source_qdq": source.to_manifest(),
        "baseline_qdq": source.to_manifest(),
        "minmax_qdq": minmax.to_manifest(),
        "selection": {
            "search_mode": search_mode,
            "fallback_to_source": fallback_to_source,
            "reason": selection_reason,
            "minimum_relative_improvement": float(minimum_relative_improvement),
            "proposed_relative_weighted_mse_improvement_vs_source": float(
                relative_improvement
            ),
            "scale_ratio_vs_source": float(selected_scale_ratio),
            "zero_point_delta_vs_source": int(selected_zero_point_delta),
        },
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
            "source_scale_ratio_min": float(source_scale_ratio_min),
            "source_scale_ratio_max": float(source_scale_ratio_max),
            "max_zero_point_delta": int(max_zero_point_delta),
            "candidate_count": len(candidates),
            "input_count": int(full_values.size),
            "search_sample_count": int(sample.size),
        },
        "selected_error": selected_metrics,
        "source_error": source_metrics,
        "baseline_error": source_metrics,
        "minmax_error": minmax_metrics,
        "search_sample_error": sample_metrics,
        "weighted_mse_ratio_vs_minmax": (
            selected_metrics["weighted_mse"] / minmax_metrics["weighted_mse"]
            if minmax_metrics["weighted_mse"] > 0.0
            else 1.0
        ),
        "weighted_mse_ratio_vs_source": (
            selected_metrics["weighted_mse"] / source_metrics["weighted_mse"]
            if source_metrics["weighted_mse"] > 0.0
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
