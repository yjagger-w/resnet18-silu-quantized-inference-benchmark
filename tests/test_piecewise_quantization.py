import unittest

import numpy as np
import torch

from silu_benchmark.quantization.activation import (
    PiecewiseQuantizationSpec,
    piecewise_dequantize,
    piecewise_quantize,
    piecewise_quantize_dequantize,
)


class PiecewiseQuantizationTests(unittest.TestCase):
    def setUp(self):
        self.spec = PiecewiseQuantizationSpec(-2.0, 0.25, 2.0)

    def test_codebook_has_disjoint_complete_256_codes(self):
        lower = set(range(128))
        upper = set(range(128, 256))
        self.assertTrue(lower.isdisjoint(upper))
        self.assertEqual(lower | upper, set(range(256)))
        self.assertEqual(self.spec.lower_codes, (0, 127))
        self.assertEqual(self.spec.upper_codes, (128, 255))

    def test_monotonicity_dense_around_boundaries(self):
        values = np.unique(np.concatenate([np.linspace(self.spec.vmin, self.spec.vmax, 5001), self.spec.vsplit + np.linspace(-1e-7, 1e-7, 101), np.linspace(-1e-7, 1e-7, 101)]))
        reconstructed = piecewise_quantize_dequantize(values, self.spec)
        self.assertTrue(np.all(np.diff(reconstructed) >= 0))

    def test_zero_is_stable_and_deterministic(self):
        code = piecewise_quantize(0.0, self.spec)
        self.assertEqual(code, piecewise_quantize(0.0, self.spec))
        self.assertEqual(code, self.spec.lower_zero_point)
        self.assertAlmostEqual(piecewise_dequantize(code, self.spec), 0.0, delta=self.spec.lower_scale)

    def test_vsplit_boundary_owns_upper_segment(self):
        values = np.array([self.spec.vsplit - 1e-8, self.spec.vsplit, self.spec.vsplit + 1e-8])
        codes = piecewise_quantize(values, self.spec)
        self.assertLessEqual(codes[0], self.spec.lower_codes[1])
        self.assertGreaterEqual(codes[1], self.spec.upper_codes[0])
        self.assertGreaterEqual(codes[2], self.spec.upper_codes[0])
        self.assertTrue(np.all(np.diff(piecewise_dequantize(codes, self.spec)) >= 0))

    def test_endpoint_saturation(self):
        codes = piecewise_quantize(np.array([self.spec.vmin - 1, self.spec.vmin, self.spec.vmax, self.spec.vmax + 1]), self.spec)
        np.testing.assert_array_equal(codes, np.array([0, 0, 255, 255]))

    def test_rounding_is_ties_to_even(self):
        lower_half = (10.5 - self.spec.lower_zero_point) * self.spec.lower_scale
        upper_half = (200.5 - self.spec.upper_zero_point) * self.spec.upper_scale
        self.assertEqual(piecewise_quantize(lower_half, self.spec) % 2, 0)
        self.assertEqual(piecewise_quantize(upper_half, self.spec) % 2, 0)

    def test_reference_shapes_dtypes_and_error(self):
        scalar = piecewise_quantize_dequantize(0.1, self.spec)
        vector = piecewise_quantize_dequantize(np.array([-1.0, 0.0, 1.0]), self.spec)
        tensor = piecewise_quantize_dequantize(torch.tensor([-1.0, 0.0, 1.0]), self.spec)
        self.assertTrue(np.isscalar(scalar))
        self.assertEqual(vector.shape, (3,))
        self.assertEqual(tensor.shape, (3,))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(np.isfinite(vector).all())

    def test_invalid_parameters_are_rejected(self):
        for values in [(-1, -1, 1), (-1, 1, 1), (0, 0.1, 1), (float("nan"), 0.1, 1)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                PiecewiseQuantizationSpec(*values)

    def test_invalid_codes_are_rejected(self):
        with self.assertRaises(ValueError):
            piecewise_dequantize(np.array([-1, 0, 256]), self.spec)

    def test_reproducibility(self):
        first = piecewise_quantize_dequantize(np.random.default_rng(1234).normal(size=1000), self.spec)
        second = piecewise_quantize_dequantize(np.random.default_rng(1234).normal(size=1000), self.spec)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
