import ast
import copy
import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort

from silu_benchmark.benchmark_data import load_cifar_batch, normalize_cifar_images
from silu_benchmark.ort_custom_op_backend import create_custom_op_session
from silu_benchmark.ort_custom_op_thread_preset import (
    analyze_profile_events,
    benchmark_statistics,
    build_session_options,
    exact_tensor_metrics,
    load_config,
    preset_fingerprint,
    repetition_reproduced,
    validate_config,
    validate_output_path,
    validation_decisions,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/runtime_profiles/resnet18_silu_cifar10_v14_ort_customop_cpu_4threads.json"
DLL = ROOT / "build/ort-cpp-customop/Release/silu_ort_custom_op.dll"
MODEL = ROOT / "artifacts/accuracy_recovery/v1.2/resnet18_silu_v12_ort_cpp_customop.onnx"
DATA = ROOT / "data/cifar-10-batches-py/test_batch"


class OrtCustomOpThreadPresetPureTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)
        self.preset = self.config["preset"]

    def test_preset_schema_and_config_validation(self):
        self.assertEqual(self.preset["intra_op_num_threads"], 4)
        self.assertEqual(self.preset["execution_mode"], "ORT_SEQUENTIAL")
        self.assertEqual(self.preset["provider"], "CPUExecutionProvider")
        self.assertEqual(self.config["benchmark"]["repetitions"], 5)

    def test_fingerprint_is_deterministic_and_field_sensitive(self):
        first = preset_fingerprint(self.preset)
        reordered = dict(reversed(list(self.preset.items())))
        self.assertEqual(first, preset_fingerprint(reordered))
        changed = copy.deepcopy(self.preset)
        changed["scope_boundary"] += " changed"
        self.assertNotEqual(first, preset_fingerprint(changed))

    def test_preset_builds_exact_session_options(self):
        options, record = build_session_options(self.preset)
        self.assertEqual(options.intra_op_num_threads, 4)
        self.assertEqual(options.inter_op_num_threads, 0)
        self.assertEqual(options.execution_mode, ort.ExecutionMode.ORT_SEQUENTIAL)
        self.assertEqual(
            options.graph_optimization_level,
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        )
        self.assertFalse(record["enable_profiling"])

    def test_selection_is_explicit_and_not_a_global_default(self):
        untouched = ort.SessionOptions()
        self.assertEqual(untouched.intra_op_num_threads, 0)
        selected, _ = build_session_options(self.preset)
        self.assertEqual(selected.intra_op_num_threads, 4)
        self.assertEqual(ort.SessionOptions().intra_op_num_threads, 0)

    def test_invalid_threads_and_provider_are_rejected(self):
        for threads in (0, -1, 2):
            changed = copy.deepcopy(self.config)
            changed["preset"]["intra_op_num_threads"] = threads
            with self.assertRaises(ValueError):
                validate_config(changed)
        changed = copy.deepcopy(self.config)
        changed["preset"]["provider"] = "AzureExecutionProvider"
        with self.assertRaisesRegex(ValueError, "CPUExecutionProvider"):
            validate_config(changed)

    def test_report_path_isolation(self):
        allowed = validate_output_path(
            ROOT / "results/benchmarks/v1.4_ort_customop_thread_preset_test",
            ROOT,
        )
        self.assertIn("v1.4_ort_customop_thread_preset", str(allowed))
        with self.assertRaises(ValueError):
            validate_output_path(
                ROOT / "results/benchmarks/v1.3_ort_customop_performance_diagnosis",
                ROOT,
            )
        with self.assertRaises(ValueError):
            validate_output_path(ROOT / "artifacts/accuracy_recovery/v1.4", ROOT)

    def test_exactness_metrics_are_zero_tolerance(self):
        reference = np.asarray([0, 1, 255], dtype=np.uint8)
        self.assertTrue(exact_tensor_metrics(reference, reference.copy())["exact_equal"])
        changed = reference.copy()
        changed[1] = 2
        metrics = exact_tensor_metrics(reference, changed)
        self.assertFalse(metrics["exact_equal"])
        self.assertEqual(metrics["mismatch_count"], 1)

    def test_profile_parser_and_category_ranking(self):
        events = [
            {
                "name": "custom_kernel_time",
                "cat": "Node",
                "dur": 8,
                "args": {
                    "op_name": "QuantizedPiecewiseSiLU",
                    "node_name": "custom",
                },
            },
            {
                "name": "conv_kernel_time",
                "cat": "Node",
                "dur": 80,
                "args": {"op_name": "Conv", "node_name": "conv"},
            },
            {"name": "session", "cat": "Session", "dur": 200},
        ]
        result = analyze_profile_events(events, inference_wall_time_us=100)
        self.assertEqual(result["categories"][0]["name"], "convolution")
        self.assertEqual(result["top_20_nodes"][0]["name"], "conv")
        self.assertEqual(result["custom_op_execution"]["unique_node_count"], 1)

    def test_raw_benchmark_statistics_retain_all_samples(self):
        raw = [[float(index + repetition) for index in range(100)] for repetition in range(5)]
        result = benchmark_statistics(raw)
        self.assertEqual(result["sample_count"], 500)
        self.assertTrue(result["raw_samples_retained"])
        self.assertFalse(result["outliers_deleted"])

    def test_batch1_digest_matches_frozen_v13_recipe(self):
        spec = importlib.util.spec_from_file_location(
            "v14_preset_runner",
            ROOT / "scripts/validate_ort_cpp_customop_thread_preset.py",
        )
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        images, _ = load_cifar_batch(ROOT / "data", "test_batch")
        inputs = normalize_cifar_images(images[:1])
        self.assertEqual(
            runner.input_digest(inputs, [0]),
            "2170a7b88671b9771e76ccb5af60e284ef35695219588a34b6f9f2ddbfc4beef",
        )

    def test_decision_defers_vector_work_when_convolution_dominates(self):
        profile = {
            "categories": [
                {"name": "convolution", "percentage_of_node_event_time": 70.0},
                {"name": "custom_activation", "percentage_of_node_event_time": 7.0},
            ],
            "custom_op_execution": {
                "unique_node_count": 17,
                "percentage_of_node_event_time": 7.0,
            },
        }
        decisions = validation_decisions(
            exactness_passed=True,
            profile=profile,
            benchmark_completed=True,
            historical_protocol_matched=True,
            historical_typical_latency_reproduced=True,
            historical_variance_reproduced=True,
        )
        self.assertEqual(
            decisions["preset"]["decision"], "validated for this machine/protocol"
        )
        self.assertEqual(decisions["avx2"]["decision"], "deferred")
        self.assertEqual(
            decisions["convolution"]["next_target"], "Conv/runtime configuration"
        )

    def test_historical_reproduction_separates_latency_and_variance(self):
        result = repetition_reproduced(
            {
                "median_of_repetition_p50_ms": 25.5,
                "coefficient_of_variation": 0.48,
            },
            {
                "repetition_p50_range_ms": [23.3, 27.5],
                "coefficient_of_variation": 0.30,
            },
        )
        self.assertTrue(result["typical_latency_reproduced"])
        self.assertFalse(result["variance_reproduced"])
        self.assertFalse(result["reproduced"])

    def test_preset_sources_do_not_import_torch(self):
        for path in (
            ROOT / "src/silu_benchmark/ort_custom_op_thread_preset.py",
            ROOT / "scripts/validate_ort_cpp_customop_thread_preset.py",
        ):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module or "")
            self.assertFalse(
                any(name == "torch" or name.startswith("torch.") for name in imports)
            )


@unittest.skipUnless(DLL.exists() and MODEL.exists() and DATA.exists(), "v1.2 runtime assets unavailable")
class OrtCustomOpThreadPresetIntegrationTests(unittest.TestCase):
    def test_real_explicit_four_thread_custom_session(self):
        config = load_config(CONFIG)
        options, record = build_session_options(config["preset"])
        session, _ = create_custom_op_session(
            MODEL,
            DLL,
            session_options=options,
        )
        images, _ = load_cifar_batch(ROOT / "data", "test_batch")
        inputs = normalize_cifar_images(images[:1])
        logits = session.run(
            [session.get_outputs()[0].name],
            {session.get_inputs()[0].name: inputs},
        )[0]
        self.assertEqual(logits.shape, (1, 10))
        self.assertEqual(record["intra_op_num_threads"], 4)
        self.assertEqual(session.get_providers()[0], "CPUExecutionProvider")


if __name__ == "__main__":
    unittest.main()
