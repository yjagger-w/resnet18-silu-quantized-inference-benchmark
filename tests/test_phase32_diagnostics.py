import sys
import unittest
from pathlib import Path

from silu_benchmark.quantization import PiecewiseQuantizationSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_silu_piecewise_mismatch import _affine_detail, _margin


class Phase32DiagnosticTests(unittest.TestCase):
    def test_rounding_and_split_margins(self):
        spec = PiecewiseQuantizationSpec(-2.0, 0.25, 2.0)
        tie = (128.5 - spec.upper_zero_point) * spec.upper_scale
        detail = _affine_detail(tie, spec)
        self.assertEqual(detail["segment"], "upper")
        self.assertAlmostEqual(detail["rounding_code_margin"], 0.0)
        margins = _margin([spec.vsplit, tie], spec)
        self.assertAlmostEqual(margins["min_vsplit_margin"], 0.0)
        self.assertAlmostEqual(margins["min_rounding_code_margin"], 0.0)


if __name__ == "__main__":
    unittest.main()
