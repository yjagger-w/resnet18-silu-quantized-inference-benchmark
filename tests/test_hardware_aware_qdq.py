import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from silu_benchmark.quantization import (
    PiecewiseQuantizationSpec,
    StandardQDQSpec,
    calibrate_silu_aware_standard_qdq,
    qdq_spec_from_manifest,
    standard_qdq_dequantize,
    standard_qdq_quantize,
    standard_qdq_quantize_dequantize,
)


ROOT = Path(__file__).resolve().parents[1]
HAS_ONNX_RUNTIME = (
    importlib.util.find_spec("onnx") is not None
    and importlib.util.find_spec("onnxruntime") is not None
)


def calibration_values() -> np.ndarray:
    rng = np.random.default_rng(20260922)
    central = rng.uniform(-0.278, 0.25, 40_000)
    tail = np.clip(rng.gamma(2.0, 0.55, 10_000), 0.25, 6.0)
    return np.concatenate([central, tail]).astype(np.float32)


class StandardQDQReferenceTests(unittest.TestCase):
    def test_spec_validation_and_unsigned_range(self):
        spec = StandardQDQSpec(scale=0.01, zero_point=28)
        self.assertEqual((spec.qmin, spec.qmax), (0, 255))
        self.assertAlmostEqual(spec.representable_min, -0.28)
        self.assertAlmostEqual(spec.representable_max, 2.27)
        with self.assertRaises(ValueError):
            StandardQDQSpec(scale=0.0, zero_point=0)
        with self.assertRaises(ValueError):
            StandardQDQSpec(scale=0.01, zero_point=256)
        with self.assertRaises(TypeError):
            StandardQDQSpec(scale=0.01, zero_point=1.5)

    def test_ties_to_even_saturation_shape_and_dtype(self):
        spec = StandardQDQSpec(scale=0.25, zero_point=4)
        values = np.array(
            [-100.0, -1.0, -0.875, -0.625, 0.0, 62.75, 100.0],
            dtype=np.float32,
        )
        codes = standard_qdq_quantize(values, spec)
        np.testing.assert_array_equal(codes, np.array([0, 0, 0, 2, 4, 255, 255], dtype=np.uint8))
        self.assertEqual(codes.shape, values.shape)
        self.assertEqual(codes.dtype, np.uint8)
        reconstructed = standard_qdq_dequantize(codes, spec)
        self.assertEqual(reconstructed.shape, values.shape)
        with self.assertRaises(ValueError):
            standard_qdq_dequantize(np.array([256]), spec)
        with self.assertRaises(TypeError):
            standard_qdq_dequantize(np.array([1.0]), spec)

    def test_manifest_loader_uses_only_standard_parameters(self):
        payload = {
            "selected_qdq": {"scale": 0.02, "zero_point": 14, "bits": 8},
            "offline_calibration": {
                "piecewise_hint": {"vmin": -0.278, "vsplit": 0.25, "vmax": 6.0}
            },
        }
        self.assertEqual(qdq_spec_from_manifest(payload), StandardQDQSpec(0.02, 14))
        with self.assertRaises(ValueError):
            qdq_spec_from_manifest({})


class SiLUAwareStandardQDQCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.values = calibration_values()
        self.hint = PiecewiseQuantizationSpec(-0.278, 0.25, 6.0)

    def test_search_is_deterministic_and_improves_its_objective(self):
        first = calibrate_silu_aware_standard_qdq(self.values, self.hint)
        second = calibrate_silu_aware_standard_qdq(self.values, self.hint)
        self.assertEqual(first, second)
        self.assertLessEqual(first["weighted_mse_ratio_vs_minmax"], 1.0)
        self.assertEqual(first["offline_calibration"]["candidate_count"], 192)
        json.dumps(first, sort_keys=True)

    def test_piecewise_values_are_offline_only(self):
        result = calibrate_silu_aware_standard_qdq(self.values, self.hint)
        runtime = result["runtime_contract"]
        self.assertEqual(runtime["quantize_op"], "QuantizeLinear")
        self.assertEqual(runtime["dequantize_op"], "DequantizeLinear")
        self.assertEqual(runtime["parameter_count"], {"scale": 1, "zero_point": 1})
        self.assertEqual(runtime["custom_runtime_nodes"], 0)
        self.assertFalse(runtime["piecewise_runtime_dispatch"])
        self.assertEqual(
            set(result["selected_qdq"]),
            {
                "scale", "zero_point", "bits", "dtype", "qmin", "qmax",
                "representable_min", "representable_max",
            },
        )
        self.assertEqual(
            result["offline_calibration"]["central_region"],
            "[Vmin, Vsplit)",
        )

    def test_bounded_sampling_is_reproducible(self):
        first = calibrate_silu_aware_standard_qdq(
            self.values, self.hint, max_samples=1024
        )
        second = calibrate_silu_aware_standard_qdq(
            self.values, self.hint, max_samples=1024
        )
        self.assertEqual(first, second)
        self.assertEqual(first["offline_calibration"]["input_count"], 50_000)
        self.assertEqual(first["offline_calibration"]["search_sample_count"], 1024)

    def test_invalid_calibration_requests_are_rejected(self):
        cases = [
            np.array([], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([-1.0, 0.0], dtype=np.float32),
            np.array([-1.0, np.nan, 1.0], dtype=np.float32),
        ]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                calibrate_silu_aware_standard_qdq(values, self.hint)
        with self.assertRaises(ValueError):
            calibrate_silu_aware_standard_qdq(self.values, self.hint, bits=7)
        with self.assertRaises(ValueError):
            calibrate_silu_aware_standard_qdq(self.values, self.hint, central_weight=0.5)

    def test_module_import_does_not_load_torch(self):
        program = r'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from silu_benchmark.quantization import calibrate_silu_aware_standard_qdq
assert calibrate_silu_aware_standard_qdq
assert "torch" not in sys.modules
assert "torchvision" not in sys.modules
'''
        result = subprocess.run(
            [sys.executable, "-c", program],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(HAS_ONNX_RUNTIME, "ONNX and ONNX Runtime are unavailable")
    def test_numpy_reference_matches_standard_onnx_qdq(self):
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper, numpy_helper

        result = calibrate_silu_aware_standard_qdq(self.values, self.hint)
        spec = qdq_spec_from_manifest(result)
        scale = np.array(spec.scale, dtype=np.float32)
        zero_point = np.array(spec.zero_point, dtype=np.uint8)
        nodes = [
            helper.make_node("QuantizeLinear", ["x", "scale", "zero_point"], ["q"]),
            helper.make_node("DequantizeLinear", ["q", "scale", "zero_point"], ["y"]),
        ]
        graph = helper.make_graph(
            nodes,
            "standard_silu_aware_qdq_test",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])],
            [
                helper.make_tensor_value_info("q", TensorProto.UINT8, [None]),
                helper.make_tensor_value_info("y", TensorProto.FLOAT, [None]),
            ],
            initializer=[
                numpy_helper.from_array(scale, name="scale"),
                numpy_helper.from_array(zero_point, name="zero_point"),
            ],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 13)],
            ir_version=8,
        )
        onnx.checker.check_model(model)
        self.assertEqual([node.op_type for node in model.graph.node], ["QuantizeLinear", "DequantizeLinear"])
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        test_values = np.array(
            [-1.0, -0.278, -0.001, 0.0, 0.25, 1.0, 6.0, 10.0],
            dtype=np.float32,
        )
        actual_codes, actual_values = session.run(None, {"x": test_values})
        expected_codes = standard_qdq_quantize(test_values, spec)
        expected_values = standard_qdq_quantize_dequantize(test_values, spec).astype(np.float32)
        np.testing.assert_array_equal(actual_codes, expected_codes)
        np.testing.assert_allclose(actual_values, expected_values, atol=1e-7, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
