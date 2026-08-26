"""Canonical spec, relocated unchanged so ONNX consumers need not import Torch."""

from dataclasses import dataclass
import math


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
