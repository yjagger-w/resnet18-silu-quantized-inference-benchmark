import ast
import copy
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from silu_benchmark.benchmark_data import load_cifar_batch, normalize_cifar_images
from silu_benchmark.ort_custom_op_backend import create_custom_op_session
from silu_benchmark.ort_custom_op_tail_latency import (
    BASELINE_ID,
    alternating_schedule,
    candidate_fingerprint,
    candidate_qualifies,
    compare_profiles,
    load_config,
    screening_candidates,
    tail_statistics,
    validate_config,
    validate_exactly_one_variable,
    validate_output_path,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/runtime_profiles/resnet18_silu_cifar10_v15_ort_customop_tail_latency.json"
DLL = ROOT / "build/ort-cpp-customop/Release/silu_ort_custom_op.dll"
MODEL = ROOT / "artifacts/accuracy_recovery/v1.2/resnet18_silu_v12_ort_cpp_customop.onnx"
DATA = ROOT / "data/cifar-10-batches-py/test_batch"


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "v15_tail_runner", ROOT / "scripts/diagnose_ort_cpp_customop_tail_latency.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


class OrtCustomOpTailLatencyPureTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)
        self.candidates = screening_candidates(self.config)

    def test_runtime_candidate_schema_validation(self):
        self.assertEqual(len(self.candidates), 9)
        self.assertEqual(self.candidates[0]["candidate_id"], BASELINE_ID)
        self.assertEqual(self.config["baseline_benchmark"]["repetitions"], 10)
        changed = copy.deepcopy(self.config)
        changed["screening_benchmark"]["repetitions"] = 4
        with self.assertRaisesRegex(ValueError, "20/100/5"):
            validate_config(changed)

    def test_screening_is_exactly_one_logical_variable(self):
        validate_exactly_one_variable(self.candidates)
        changed = copy.deepcopy(self.candidates)
        changed[1]["requested_options"]["enable_mem_pattern"] = False
        with self.assertRaisesRegex(ValueError, "exactly one"):
            validate_exactly_one_variable(changed)
        for row in self.candidates:
            if row["change_group"] != "intra_op_threads":
                self.assertEqual(row["requested_options"]["intra_op_num_threads"], 4)

    def test_candidate_fingerprint_is_deterministic_and_sensitive(self):
        candidate = self.candidates[1]
        self.assertEqual(candidate_fingerprint(candidate), candidate_fingerprint(copy.deepcopy(candidate)))
        changed = copy.deepcopy(candidate)
        changed["requested_options"]["enable_cpu_mem_arena"] = True
        self.assertNotEqual(candidate_fingerprint(candidate), candidate_fingerprint(changed))

    def test_requested_and_applied_option_reporting(self):
        runner = load_runner()
        candidate = next(row for row in self.candidates if row["candidate_id"] == "graph_extended")
        _, record = runner.build_candidate_options(candidate)
        self.assertEqual(record["requested"]["graph_optimization_level"], "ORT_ENABLE_EXTENDED")
        self.assertEqual(
            record["applied"]["graph_optimization_level"],
            "GraphOptimizationLevel.ORT_ENABLE_EXTENDED",
        )
        self.assertEqual(record["applied"]["intra_op_num_threads"], 4)

    def test_raw_samples_are_retained_and_p99_reported(self):
        raw = [[float(index + repetition) for index in range(100)] for repetition in range(10)]
        result = tail_statistics(raw)
        self.assertEqual(result["sample_count"], 1000)
        self.assertIn("p99_ms", result)
        self.assertTrue(result["raw_samples_retained"])
        self.assertFalse(result["outliers_deleted"])

    def test_alternating_finalist_schedule(self):
        schedule = alternating_schedule("memory_pattern_disabled")
        self.assertEqual(len(schedule), 20)
        self.assertEqual(schedule[::2], [BASELINE_ID] * 10)
        self.assertEqual(schedule[1::2], ["memory_pattern_disabled"] * 10)

    def test_candidate_rejected_when_p95_or_cv_worsens(self):
        baseline = {
            "statistics": {
                "p50_ms": 25,
                "median_of_repetition_p50_ms": 25,
                "median_of_repetition_p95_ms": 35,
                "coefficient_of_variation": 0.3,
            }
        }
        candidate = {
            "status": "complete",
            "exactness": {"passed": True},
            "statistics": {
                "p50_ms": 24,
                "median_of_repetition_p50_ms": 24,
                "median_of_repetition_p95_ms": 40,
                "coefficient_of_variation": 0.4,
            },
        }
        result = candidate_qualifies(baseline, candidate)
        self.assertFalse(result["qualifies"])
        self.assertIn("median_repetition_p95_not_worse", result["reasons"])
        self.assertIn("coefficient_of_variation_not_worse", result["reasons"])

    def test_exactness_failure_prohibits_candidate_timing(self):
        baseline = {
            "statistics": {
                "p50_ms": 25,
                "median_of_repetition_p50_ms": 25,
                "median_of_repetition_p95_ms": 35,
                "coefficient_of_variation": 0.3,
            }
        }
        candidate = {"status": "exactness_failed", "exactness": {"passed": False}}
        result = candidate_qualifies(baseline, candidate)
        self.assertFalse(result["qualifies"])

        runner = load_runner()
        with mock.patch.object(
            runner,
            "run_exactness_gate",
            return_value={"passed": False, "first_failure": {"probe_offset": 0}},
        ), mock.patch.object(runner, "run_repetitions") as timed:
            row = runner.run_candidate(
                candidate=self.candidates[0],
                reference_path=ROOT / "reference.onnx",
                custom_path=ROOT / "custom.onnx",
                library_path=ROOT / "custom.dll",
                normalized_probe=np.zeros((128, 3, 32, 32), dtype=np.float32),
                benchmark_input=np.zeros((1, 3, 32, 32), dtype=np.float32),
                protocol=self.config["baseline_benchmark"],
            )
        self.assertEqual(row["status"], "exactness_failed")
        timed.assert_not_called()

    def test_profile_comparison_aggregates_independent_runs(self):
        def profile(profile_id, conv, custom):
            return {
                "profile_id": profile_id,
                "total_inference_wall_ms": 100,
                "analysis": {
                    "summed_node_event_time_us": 90_000,
                    "custom_op_execution": {"unique_node_count": 17},
                    "categories": [
                        {"name": "convolution", "percentage_of_node_event_time": conv},
                        {"name": "custom_activation", "percentage_of_node_event_time": custom},
                    ],
                },
            }
        result = compare_profiles([profile("a", 60, 10), profile("b", 55, 12)])
        self.assertTrue(result["both_all_17_nodes"])
        self.assertTrue(result["convolution_dominant_in_both"])
        self.assertEqual(result["custom_share_absolute_difference_pp"], 2)

    def test_output_path_isolation(self):
        allowed = validate_output_path(
            ROOT / "results/benchmarks/v1.5_ort_customop_tail_latency_test", ROOT
        )
        self.assertIn("v1.5_ort_customop_tail_latency", str(allowed))
        with self.assertRaises(ValueError):
            validate_output_path(
                ROOT / "results/benchmarks/v1.4_ort_customop_thread_preset", ROOT
            )

    def test_new_sources_do_not_import_torch(self):
        for path in (
            ROOT / "src/silu_benchmark/ort_custom_op_tail_latency.py",
            ROOT / "scripts/diagnose_ort_cpp_customop_tail_latency.py",
        ):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module or "")
            self.assertFalse(any(name == "torch" or name.startswith("torch.") for name in imports))


@unittest.skipUnless(DLL.exists() and MODEL.exists() and DATA.exists(), "v1.2 runtime assets unavailable")
class OrtCustomOpTailLatencyIntegrationTests(unittest.TestCase):
    def test_real_baseline_candidate_session(self):
        runner = load_runner()
        candidate = screening_candidates(load_config(CONFIG))[0]
        options, record = runner.build_candidate_options(candidate)
        session, _ = create_custom_op_session(MODEL, DLL, session_options=options)
        images, _ = load_cifar_batch(ROOT / "data", "test_batch")
        inputs = normalize_cifar_images(images[:1])
        logits = session.run(
            [session.get_outputs()[0].name],
            {session.get_inputs()[0].name: inputs},
        )[0]
        self.assertEqual(logits.shape, (1, 10))
        self.assertEqual(record["applied"]["intra_op_num_threads"], 4)
        self.assertEqual(session.get_providers()[0], "CPUExecutionProvider")


if __name__ == "__main__":
    unittest.main()
