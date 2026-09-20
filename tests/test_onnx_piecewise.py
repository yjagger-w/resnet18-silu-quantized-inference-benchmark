import unittest

import numpy as np
import onnx

from silu_benchmark.backends.onnx_piecewise import (
    PIECEWISE_ONNX_OPSET,
    build_piecewise_qdq_model,
    create_piecewise_ort_session,
)
from silu_benchmark.quantization import (
    PiecewiseQuantizationSpec,
    piecewise_dequantize,
    piecewise_quantize,
)


FLOAT32_DEQUANTIZATION_ATOL = 2e-7


class OnnxPiecewiseQuantizationTests(unittest.TestCase):
    def setUp(self):
        self.spec = PiecewiseQuantizationSpec(-2.0, 0.25, 2.0)
        self.model = build_piecewise_qdq_model(self.spec)
        self.session = create_piecewise_ort_session(self.model)
        self.input_rank = 1

    def run_onnx(self, values):
        input_values = np.asarray(values, dtype=np.float32)
        if input_values.ndim != self.input_rank:
            self.model = build_piecewise_qdq_model(
                self.spec, input_shape=(None,) * input_values.ndim
            )
            self.session = create_piecewise_ort_session(self.model)
            self.input_rank = input_values.ndim
        return self.session.run(
            ["quantized_codes", "dequantized_output"], {"activation": input_values}
        )

    def assert_equivalent(self, values):
        input_values = np.asarray(values, dtype=np.float32)
        expected_codes = np.asarray(piecewise_quantize(input_values, self.spec), dtype=np.uint8)
        expected_dequantized = np.asarray(
            piecewise_dequantize(expected_codes, self.spec), dtype=np.float32
        )
        actual_codes, actual_dequantized = self.run_onnx(input_values)
        mismatch = np.argwhere(actual_codes != expected_codes)
        if mismatch.size:
            index = tuple(mismatch[0])
            value = input_values[index] if input_values.ndim else input_values.item()
            expected_code = expected_codes[index] if expected_codes.ndim else expected_codes.item()
            actual_code = actual_codes[index] if actual_codes.ndim else actual_codes.item()
            segment = "lower" if value < self.spec.vsplit else "upper"
            self.fail(
                "code mismatch: input={!r}, segment={}, Python code={}, ONNX code={}, "
                "Python dequantized={!r}, ONNX dequantized={!r}".format(
                    value,
                    segment,
                    expected_code,
                    actual_code,
                    expected_dequantized[index] if expected_dequantized.ndim else expected_dequantized.item(),
                    actual_dequantized[index] if actual_dequantized.ndim else actual_dequantized.item(),
                )
            )
        errors = np.abs(actual_dequantized - expected_dequantized)
        if not np.all(errors <= FLOAT32_DEQUANTIZATION_ATOL):
            index = tuple(np.unravel_index(np.argmax(errors), errors.shape)) if errors.ndim else ()
            value = input_values[index] if input_values.ndim else input_values.item()
            segment = "lower" if value < self.spec.vsplit else "upper"
            self.fail(
                "dequantization mismatch: input={!r}, segment={}, Python code={}, "
                "ONNX code={}, Python dequantized={!r}, ONNX dequantized={!r}, "
                "absolute error={!r}".format(
                    value,
                    segment,
                    expected_codes[index] if expected_codes.ndim else expected_codes.item(),
                    actual_codes[index] if actual_codes.ndim else actual_codes.item(),
                    expected_dequantized[index] if expected_dequantized.ndim else expected_dequantized.item(),
                    actual_dequantized[index] if actual_dequantized.ndim else actual_dequantized.item(),
                    errors[index] if errors.ndim else errors.item(),
                )
            )

    def test_model_passes_checker_and_declares_opset(self):
        onnx.checker.check_model(self.model)
        self.assertEqual(self.model.opset_import[0].version, PIECEWISE_ONNX_OPSET)

    def test_boundaries_and_saturation(self):
        split_below = np.nextafter(np.float32(self.spec.vsplit), np.float32(-np.inf))
        split_above = np.nextafter(np.float32(self.spec.vsplit), np.float32(np.inf))
        values = np.array(
            [
                self.spec.vmin - 1.0,
                self.spec.vmin,
                0.0,
                split_below,
                self.spec.vsplit,
                split_above,
                self.spec.vmax,
                self.spec.vmax + 1.0,
            ],
            dtype=np.float32,
        )
        codes, outputs = self.run_onnx(values)
        self.assertEqual(codes[0], 0)
        self.assertEqual(codes[1], 0)
        self.assertEqual(codes[2], self.spec.lower_zero_point)
        self.assertLessEqual(codes[3], self.spec.lower_codes[1])
        self.assertGreaterEqual(codes[4], self.spec.upper_codes[0])
        self.assertGreaterEqual(codes[5], self.spec.upper_codes[0])
        self.assertEqual(codes[6], 255)
        self.assertEqual(codes[7], 255)
        self.assertTrue(np.all(np.isfinite(outputs)))
        self.assert_equivalent(values)

    def test_ties_to_even_in_both_segments(self):
        lower_tie = (10.5 - self.spec.lower_zero_point) * self.spec.lower_scale
        upper_tie = (200.5 - self.spec.upper_zero_point) * self.spec.upper_scale
        codes, _ = self.run_onnx(np.array([lower_tie, upper_tie], dtype=np.float32))
        self.assertEqual(codes[0] % 2, 0)
        self.assertEqual(codes[1] % 2, 0)
        self.assert_equivalent([lower_tie, upper_tie])

    def test_all_codes_are_disjoint_and_reachable(self):
        lower_codes = np.arange(0, 128, dtype=np.int64)
        upper_codes = np.arange(128, 256, dtype=np.int64)
        lower_values = (lower_codes - self.spec.lower_zero_point) * self.spec.lower_scale
        upper_values = (upper_codes - self.spec.upper_zero_point) * self.spec.upper_scale
        # The decoded representation of code 128 can lie just below Vsplit;
        # it still belongs to the upper segment, so use the owned boundary as
        # its constructed input rather than re-testing that decoded value.
        upper_values = np.maximum(upper_values, self.spec.vsplit)
        values = np.concatenate([lower_values, upper_values]).astype(np.float32)
        codes, _ = self.run_onnx(values)
        self.assertEqual(set(codes.tolist()), set(range(256)))
        self.assert_equivalent(values)

    def test_scalar_vector_matrix_and_tensor_shapes(self):
        for values in (
            np.float32(0.1),
            np.array([-1.0, 0.0, 1.0], dtype=np.float32),
            np.array([[-1.0, 0.0], [0.25, 1.0]], dtype=np.float32),
            np.zeros((2, 1, 3, 4), dtype=np.float32),
        ):
            with self.subTest(shape=np.asarray(values).shape):
                codes, dequantized = self.run_onnx(values)
                self.assertEqual(codes.shape, np.asarray(values).shape)
                self.assertEqual(dequantized.shape, np.asarray(values).shape)
                self.assertEqual(codes.dtype, np.uint8)
                self.assertEqual(dequantized.dtype, np.float32)
                self.assert_equivalent(values)

    def test_fixed_seed_random_and_multiple_specs(self):
        values = np.random.default_rng(1234).normal(size=(3, 5, 7)).astype(np.float32)
        self.assert_equivalent(values)
        for spec in (
            PiecewiseQuantizationSpec(-1.0, 0.1, 1.5),
            PiecewiseQuantizationSpec(-3.75, 0.75, 6.0),
            PiecewiseQuantizationSpec(-0.125, 0.025, 0.5),
        ):
            with self.subTest(spec=spec):
                self.spec = spec
                self.model = build_piecewise_qdq_model(spec)
                self.session = create_piecewise_ort_session(self.model)
                self.input_rank = 1
                self.assert_equivalent(values)

    def test_graph_uses_piecewise_operators_not_standard_qdq(self):
        operator_types = {node.op_type for node in self.model.graph.node}
        self.assertTrue({"Less", "Where", "Round", "Clip"}.issubset(operator_types))
        self.assertNotIn("QuantizeLinear", operator_types)
        self.assertNotIn("DequantizeLinear", operator_types)


if __name__ == "__main__":
    unittest.main()
