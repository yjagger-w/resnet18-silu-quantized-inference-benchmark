"""Activation quantizer modules and the canonical SiLU piecewise reference."""

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class PiecewiseQuantizationSpec:
    """Validated affine two-segment, unsigned integer codebook."""

    vmin: float
    vsplit: float
    vmax: float
    bits: int = 8

    def __post_init__(self):
        values = (self.vmin, self.vsplit, self.vmax)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Vmin, Vsplit, and Vmax must be finite")
        if not self.vmin < 0.0 < self.vsplit < self.vmax:
            raise ValueError("require Vmin < 0 < Vsplit < Vmax")
        if self.bits < 2:
            raise ValueError("bits must be at least 2")

    @property
    def code_count(self):
        return 1 << self.bits

    @property
    def lower_codes(self):
        return (0, self.code_count // 2 - 1)

    @property
    def upper_codes(self):
        return (self.code_count // 2, self.code_count - 1)

    @property
    def lower_scale(self):
        return (self.vsplit - self.vmin) / self.lower_codes[1]

    @property
    def upper_scale(self):
        return (self.vmax - self.vsplit) / (self.upper_codes[1] - self.upper_codes[0])

    @property
    def lower_zero_point(self):
        return int(round(-self.vmin / self.lower_scale))

    @property
    def upper_zero_point(self):
        rounded = int(round(self.upper_codes[0] - self.vsplit / self.upper_scale))
        # Independent affine rounding can invert the two decoded boundary codes.
        # Move only the upper offset toward larger values until the codebook joins.
        lower_endpoint = (self.lower_codes[1] - self.lower_zero_point) * self.lower_scale
        boundary_guard = math.floor(self.upper_codes[0] - lower_endpoint / self.upper_scale)
        return min(rounded, boundary_guard)


def _round_nearest_even(value):
    """Use NumPy/PyTorch's deterministic ties-to-even rounding policy."""
    return np.rint(value)


def piecewise_quantize(values, spec):
    """Return integer codes for scalar, NumPy, or Torch inputs."""
    if not isinstance(spec, PiecewiseQuantizationSpec):
        raise TypeError("spec must be a PiecewiseQuantizationSpec")
    is_torch = isinstance(values, torch.Tensor)
    array = values.detach().cpu().numpy() if is_torch else np.asarray(values)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("values must be numeric")
    clipped = np.clip(array.astype(np.float64), spec.vmin, spec.vmax)
    lower = clipped < spec.vsplit
    codes = np.empty(clipped.shape, dtype=np.int64)
    codes[lower] = np.clip(
        _round_nearest_even(clipped[lower] / spec.lower_scale + spec.lower_zero_point),
        *spec.lower_codes,
    )
    codes[~lower] = np.clip(
        _round_nearest_even(clipped[~lower] / spec.upper_scale + spec.upper_zero_point),
        *spec.upper_codes,
    )
    if is_torch:
        return torch.as_tensor(codes, dtype=torch.int64, device=values.device)
    return codes.item() if codes.ndim == 0 else codes


def piecewise_dequantize(codes, spec):
    """Reconstruct floating-point values from the canonical codebook."""
    if not isinstance(spec, PiecewiseQuantizationSpec):
        raise TypeError("spec must be a PiecewiseQuantizationSpec")
    is_torch = isinstance(codes, torch.Tensor)
    array = codes.detach().cpu().numpy() if is_torch else np.asarray(codes)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("codes must be numeric")
    integer_codes = np.asarray(array, dtype=np.int64)
    if np.any(integer_codes < 0) or np.any(integer_codes >= spec.code_count):
        raise ValueError("codes outside the valid code range")
    lower = integer_codes <= spec.lower_codes[1]
    values = np.empty(integer_codes.shape, dtype=np.float64)
    values[lower] = (integer_codes[lower] - spec.lower_zero_point) * spec.lower_scale
    values[~lower] = (integer_codes[~lower] - spec.upper_zero_point) * spec.upper_scale
    values = np.clip(values, spec.vmin, spec.vmax)
    if is_torch:
        return torch.as_tensor(values, dtype=torch.float32, device=codes.device)
    return values.item() if values.ndim == 0 else values


def piecewise_quantize_dequantize(values, spec):
    """Canonical reference composition used by tests and the module wrapper."""
    return piecewise_dequantize(piecewise_quantize(values, spec), spec)


class BaseQuantizer(nn.Module):
    def __init__(self, threshold, bits=8, use_ste=False, symmetric=True):
        super().__init__()
        self.bits = bits
        self.threshold = float(threshold)
        self.use_ste = use_ste
        self.symmetric = symmetric
        self.qmax = 2 ** (bits - 1) - 1 if symmetric else 2 ** bits - 1
        self.scale = self.qmax / max(self.threshold, 1e-6)

    def quantize_dequantize(self, x):
        if self.symmetric:
            clipped = torch.clamp(x, -self.threshold, self.threshold)
            quantized = torch.round(clipped * self.scale)
            quantized = torch.clamp(quantized, -self.qmax, self.qmax)
            return quantized / self.scale

        clipped = torch.clamp(x, 0, self.threshold)
        quantized = torch.round(clipped * self.scale)
        quantized = torch.clamp(quantized, 0, self.qmax)
        return quantized / self.scale

    def forward(self, x):
        if self.use_ste and self.training:
            with torch.no_grad():
                dequantized = self.quantize_dequantize(x)
            return x + (dequantized - x).detach()
        return self.quantize_dequantize(x)


class ActivationQuantizerNCNN(BaseQuantizer):
    def __init__(self, vmax, bits=8, use_ste=False):
        super().__init__(vmax, bits, use_ste, symmetric=True)


class ActivationQuantizerSiLUAware(nn.Module):
    def __init__(self, vmax, vsplit, bits=8, use_ste=False, vmin=None):
        super().__init__()
        self.bits = bits
        self.vmax = float(vmax)
        self.vsplit = float(vsplit)
        self.vmin = float(vmin if vmin is not None else -vmax)
        self.use_ste = use_ste
        self.spec = PiecewiseQuantizationSpec(self.vmin, self.vsplit, self.vmax, bits)

    def quantize_dequantize(self, x):
        return piecewise_quantize_dequantize(x, self.spec).to(dtype=x.dtype)

    def forward(self, x):
        if self.use_ste and self.training:
            with torch.no_grad():
                dequantized = self.quantize_dequantize(x)
            return x + (dequantized - x).detach()
        return self.quantize_dequantize(x)
