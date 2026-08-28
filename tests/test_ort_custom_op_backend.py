import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from silu_benchmark.ort_custom_op_backend import (
    comparison_metrics,
    create_custom_op_session,
    detect_prerequisites,
    load_config,
    parse_custom_op_profile,
    require_exact_parity,
    require_prerequisites,
    validate_config,
)
from silu_benchmark.ort_custom_op_rewrite import CUSTOM_OP_TYPE


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/benchmarks/resnet18_silu_cifar10_v12_ort_customop_cpu.json"
DLL = ROOT / "build/ort-cpp-customop/Release/silu_ort_custom_op.dll"
MODEL = ROOT / "artifacts/accuracy_recovery/v1.2/resnet18_silu_v12_ort_cpp_customop.onnx"


class OrtCustomOpBackendTests(unittest.TestCase):
    def test_config_and_cli_contract(self):
        payload = load_config(CONFIG)
        self.assertEqual(payload["expected_site_count"], 17)
        changed = dict(payload)
        changed["provider"] = "SomeOtherProvider"
        with self.assertRaisesRegex(ValueError, "CPUExecutionProvider"):
            validate_config(changed)
        changed = dict(payload)
        changed["expected_site_count"] = 16
        with self.assertRaisesRegex(ValueError, "17"):
            validate_config(changed)

    def test_prerequisite_report_is_actionable(self):
        result = detect_prerequisites()
        self.assertEqual(result.ort_version, "1.19.2")
        self.assertIsNotNone(result.runtime_dll)
        if not result.supported:
            self.assertTrue(result.blockers)
            self.assertIn("ORT_ROOT", result.recommendation)
            with self.assertRaisesRegex(RuntimeError, "prerequisites"):
                require_prerequisites()

    def test_missing_custom_library_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "model is missing"):
            create_custom_op_session(ROOT / "missing-model.onnx", ROOT / "missing.dll")
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.onnx"
            model.write_bytes(b"placeholder")
            with self.assertRaisesRegex(FileNotFoundError, "library is missing"):
                create_custom_op_session(model, ROOT / "missing.dll")

    def test_profile_execution_evidence_requires_all_17_nodes(self):
        names = [f"custom_{index}" for index in range(17)]
        events = [
            {
                "cat": "Node",
                "name": f"{name}_kernel_time",
                "dur": index + 1,
                "args": {"op_name": CUSTOM_OP_TYPE, "node_name": name},
            }
            for index, name in enumerate(names)
        ]
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile.json"
            profile.write_text(json.dumps(events), encoding="utf-8")
            complete = parse_custom_op_profile(profile, names)
            self.assertTrue(complete["all_expected_nodes_executed"])
            profile.write_text(json.dumps(events[:-1]), encoding="utf-8")
            incomplete = parse_custom_op_profile(profile, names)
            self.assertFalse(incomplete["all_expected_nodes_executed"])
            self.assertEqual(incomplete["missing_node_names"], [names[-1]])

    def test_zero_tolerance_metrics_and_first_difference(self):
        reference = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        exact = require_exact_parity(reference, reference.copy(), tensor_name="codes")
        self.assertTrue(exact["exact_equal"])
        changed = reference.copy()
        changed[1, 0] = 9
        metrics = comparison_metrics(reference, changed)
        self.assertFalse(metrics["exact_equal"])
        self.assertEqual(metrics["max_absolute_error"], 6.0)
        with self.assertRaisesRegex(RuntimeError, "flat index 2"):
            require_exact_parity(reference, changed, tensor_name="codes")

    @unittest.skipUnless(DLL.exists() and MODEL.exists(), "real custom-op DLL/model unavailable")
    def test_guarded_real_custom_op_registration(self):
        prerequisites = require_prerequisites()
        self.assertTrue(prerequisites.supported)
        session, registered = create_custom_op_session(MODEL, DLL)
        self.assertTrue(registered)
        self.assertEqual(session.get_providers()[0], "CPUExecutionProvider")


if __name__ == "__main__":
    unittest.main()
