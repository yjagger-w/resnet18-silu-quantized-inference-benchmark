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


if __name__ == "__main__":
    unittest.main()
