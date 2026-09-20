import ast
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from silu_benchmark.benchmark_data import load_cifar_batch, normalize_cifar_images
from silu_benchmark.ort_custom_op_performance import (
    NON_ADDITIVITY_WARNING,
    benchmark_cells,
    classify_comparability,
    decide_optimization,
    load_config,
    parse_profile_events,
    sample_statistics,
    summarize_repetitions,
    validate_config,
    validate_output_path,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/benchmarks/resnet18_silu_cifar10_v13_ort_customop_performance.json"
DLL = ROOT / "build/ort-cpp-customop/Release/silu_ort_custom_op.dll"
MODEL = ROOT / "artifacts/accuracy_recovery/v1.2/resnet18_silu_v12_ort_cpp_customop.onnx"
DATA = ROOT / "data/cifar-10-batches-py/test_batch"


class OrtCustomOpPerformancePureTests(unittest.TestCase):
    def test_config_validation_and_bounded_matrix(self):
        config = load_config(CONFIG)
        cells = benchmark_cells(config)
        self.assertEqual(len(cells), 16)
        self.assertEqual({row["batch_size"] for row in cells}, {1, 8, 32})
        changed = copy.deepcopy(config)
        changed["benchmark"]["repetitions"] = 4
        with self.assertRaisesRegex(ValueError, "five"):
            validate_config(changed)
        changed = copy.deepcopy(config)
        changed["benchmark"]["timed_iterations"] = 99
        with self.assertRaisesRegex(ValueError, "100"):
            validate_config(changed)

    def test_profile_parser_accepts_unordered_and_sparse_events(self):
        events = [
            {"name": "session_initialization", "cat": "Session", "dur": 10},
            {
                "name": "custom_b_kernel_time",
                "cat": "Node",
                "dur": 7,
                "args": {"op_name": "QuantizedPiecewiseSiLU", "node_name": "custom_b"},
            },
            {"name": "incomplete_kernel_time", "cat": "Node", "args": {}},
            {
                "name": "conv_kernel_time",
                "cat": "Node",
                "dur": 90,
                "args": {"op_name": "Conv", "node_name": "conv"},
            },
            {
                "name": "custom_a_kernel_time",
                "cat": "Node",
                "dur": 3,
                "args": {"op_name": "QuantizedPiecewiseSiLU", "node_name": "custom_a"},
            },
        ]
        parsed = parse_profile_events(events, inference_wall_time_us=120)
        self.assertEqual(parsed["parsed_kernel_event_count"], 3)
        self.assertEqual(parsed["custom_op_execution"]["unique_node_count"], 2)
        self.assertEqual(parsed["custom_op_execution"]["aggregate_duration_us"], 10)
        self.assertEqual(parsed["top_20_nodes"][0]["name"], "conv")

    def test_aggregation_ranking_and_non_additivity_warning(self):
        events = [
            {
                "name": f"{name}_kernel_time",
                "cat": "Node",
                "dur": duration,
                "args": {"op_name": op, "node_name": name},
            }
            for name, op, duration in (
                ("gemm", "Gemm", 20),
                ("conv", "Conv", 80),
                ("dq", "DequantizeLinear", 10),
            )
        ]
        parsed = parse_profile_events(events, inference_wall_time_us=60)
        self.assertEqual(parsed["nodes"][0]["name"], "conv")
        self.assertEqual(parsed["categories"][0]["name"], "convolution")
        self.assertEqual(parsed["non_additivity_warning"], NON_ADDITIVITY_WARNING)
        self.assertGreater(parsed["node_sum_to_wall_ratio"], 1)

    def test_raw_statistics_keep_every_sample_and_no_outlier_deletion(self):
        raw = [[float(index + repetition) for index in range(100)] for repetition in range(5)]
        result = summarize_repetitions(raw, batch_size=1)
        self.assertEqual(result["sample_count"], 500)
        self.assertEqual(result["repetition_count"], 5)
        self.assertTrue(result["raw_samples_retained"])
        self.assertFalse(result["outliers_deleted"])
        single = sample_statistics([1, 2, 100], batch_size=2)
        self.assertEqual(single["max_ms"], 100)
        self.assertEqual(single["sample_count"], 3)

    def test_comparability_classification(self):
        protocol = {
            "provider": "CPUExecutionProvider",
            "batch_size": 1,
            "warmup_iterations": 20,
            "timed_iterations": 100,
            "repetitions": 5,
            "input_digest": "same",
            "input_reuse": "cached",
            "graph_optimization_level": "all",
            "execution_mode": "sequential",
            "intra_op_threads": 0,
            "inter_op_threads": 0,
            "profiling_enabled": False,
            "timing_boundary": "run",
            "output_selection": "logits",
            "session_lifecycle": "fresh",
            "between_invocation_work": "none",
        }
        self.assertTrue(
            classify_comparability(protocol, dict(protocol))["directly_protocol_comparable"]
        )
        changed = dict(protocol, input_digest="different", repetitions=1)
        result = classify_comparability(protocol, changed)
        self.assertFalse(result["directly_protocol_comparable"])
        self.assertEqual({row["field"] for row in result["differences"]}, {"input_digest", "repetitions"})

    def test_decision_rejects_small_custom_share_for_vector_priority(self):
        profile = {
            "custom_op_execution": {"percentage_of_node_event_time": 2.5},
            "categories": [
                {"name": "convolution", "percentage_of_node_event_time": 90.0}
            ],
        }
        cells = []
        for threads, medians, cv in ((0, [10, 10, 10, 10, 10], 0.1), (4, [9, 11, 9, 11, 9], 0.2)):
            cells.append(
                {
                    "graph": "custom_op",
                    "batch_size": 1,
                    "intra_op_threads": threads,
                    "execution_mode": "sequential",
                    "statistics": {
                        "median_of_repetition_p50_ms": float(np.median(medians)),
                        "coefficient_of_variation": cv,
                        "per_repetition": [{"p50_ms": value} for value in medians],
                    },
                }
            )
        decisions = decide_optimization(profile, cells)
        self.assertEqual(decisions["avx2_kernel"]["decision"], "not recommended")
        self.assertEqual(decisions["thread_tuning"]["decision"], "no stable improvement")
        self.assertEqual(decisions["graph_runtime_work"]["target"], "convolution")

    def test_thread_decision_prefers_stable_candidate_over_unstable_fastest(self):
        profile = {
            "custom_op_execution": {"percentage_of_node_event_time": 2.0},
            "categories": [
                {"name": "convolution", "percentage_of_node_event_time": 90.0}
            ],
        }

        def cell(threads, medians, cv):
            return {
                "graph": "custom_op",
                "batch_size": 1,
                "intra_op_threads": threads,
                "execution_mode": "sequential",
                "statistics": {
                    "median_of_repetition_p50_ms": float(np.median(medians)),
                    "coefficient_of_variation": cv,
                    "per_repetition": [{"p50_ms": value} for value in medians],
                },
            }

        decisions = decide_optimization(
            profile,
            [
                cell(0, [70, 68, 72, 69, 71], 0.32),
                cell(4, [27, 26, 28, 25, 27], 0.30),
                cell(8, [25, 24, 23, 26, 25], 0.49),
            ],
        )
        self.assertIn("intra_op_num_threads=4", decisions["thread_tuning"]["decision"])

    def test_output_path_isolation(self):
        allowed = validate_output_path(
            ROOT / "results/benchmarks/v1.3_ort_customop_performance_diagnosis_test",
            ROOT,
        )
        self.assertIn("v1.3_ort_customop_performance_diagnosis", str(allowed))
        with self.assertRaises(ValueError):
            validate_output_path(ROOT / "results/benchmarks/v1.2_ort_cpp_customop", ROOT)
        with self.assertRaises(ValueError):
            validate_output_path(ROOT / "artifacts/accuracy_recovery/v1.3", ROOT)

    def test_diagnostic_sources_have_no_torch_imports(self):
        for path in (
            ROOT / "src/silu_benchmark/ort_custom_op_performance.py",
            ROOT / "scripts/diagnose_ort_cpp_customop_performance.py",
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
class OrtCustomOpPerformanceIntegrationTests(unittest.TestCase):
    def test_real_unprofiled_diagnostic_session(self):
        spec = importlib.util.spec_from_file_location(
            "v13_diagnostic_runner", ROOT / "scripts/diagnose_ort_cpp_customop_performance.py"
        )
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        session, metadata = runner.create_session(
            MODEL, graph="custom_op", library_path=DLL
        )
        images, _ = load_cifar_batch(ROOT / "data", "test_batch")
        values = normalize_cifar_images(images[:1])
        logits = session.run(
            [session.get_outputs()[0].name],
            {session.get_inputs()[0].name: values},
        )[0]
        self.assertEqual(logits.shape, (1, 10))
        self.assertFalse(metadata["session_options"]["enable_profiling"])
        self.assertEqual(session.get_providers()[0], "CPUExecutionProvider")
        runtime = Path(metadata["runtime_dll"])
        self.assertTrue(runner.windows_file_version(runtime).startswith("1.19"))


if __name__ == "__main__":
    unittest.main()
