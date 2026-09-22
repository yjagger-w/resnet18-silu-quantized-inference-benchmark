import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "hardware_aware_qdq_runner", ROOT / "scripts/run_hardware_aware_qdq.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HardwareAwareQDQRunnerTests(unittest.TestCase):
    def test_prediction_transitions_separate_recoveries_and_regressions(self):
        runner = load_runner()
        labels = np.array([0, 1, 2, 3, 4])
        standard = np.array([9, 1, 8, 3, 4])
        candidate = np.array([0, 7, 2, 3, 6])
        transitions = runner.prediction_transitions(labels, standard, candidate)
        self.assertEqual(transitions["changed_prediction_count"], 4)
        self.assertEqual(transitions["recovered_error_indices"], [0, 2])
        self.assertEqual(transitions["introduced_error_indices"], [1, 4])
        self.assertEqual(transitions["net_correct_change"], 0)

    def test_prediction_transitions_reject_misaligned_inputs(self):
        runner = load_runner()
        with self.assertRaises(ValueError):
            runner.prediction_transitions(
                np.array([0, 1]), np.array([0]), np.array([0, 1])
            )

    def test_local_promotion_gate_accepts_only_complete_non_regressing_result(self):
        runner = load_runner()
        gate = {
            "required_samples": 10_000,
            "minimum_accuracy_delta_vs_standard_qdq_pp": 0.0,
            "minimum_prediction_agreement_vs_standard_qdq_percent": 98.0,
            "fallback_model": "source_standard_qdq_model",
        }
        accepted = runner.evaluate_local_promotion_gate(
            {
                "accuracy_delta_vs_standard_qdq_pp": 0.05,
                "prediction_agreement_vs_standard_qdq_percent": 98.49,
            },
            evaluation_samples=10_000,
            gate=gate,
        )
        self.assertEqual(accepted["status"], "accepted_for_device_preflight")
        self.assertTrue(accepted["accepted_for_device_preflight"])

        rejected = runner.evaluate_local_promotion_gate(
            {
                "accuracy_delta_vs_standard_qdq_pp": -0.18,
                "prediction_agreement_vs_standard_qdq_percent": 97.99,
            },
            evaluation_samples=10_000,
            gate=gate,
        )
        self.assertEqual(rejected["status"], "rejected_local_fallback_to_source")
        self.assertFalse(rejected["accepted_for_device_preflight"])
        self.assertFalse(rejected["checks"]["accuracy_not_regressed"])
        self.assertFalse(rejected["checks"]["prediction_agreement_sufficient"])

        incomplete = runner.evaluate_local_promotion_gate(
            {
                "accuracy_delta_vs_standard_qdq_pp": 0.50,
                "prediction_agreement_vs_standard_qdq_percent": 99.0,
            },
            evaluation_samples=128,
            gate=gate,
        )
        self.assertEqual(incomplete["status"], "incomplete")
        self.assertFalse(incomplete["accepted_for_device_preflight"])


if __name__ == "__main__":
    unittest.main()
