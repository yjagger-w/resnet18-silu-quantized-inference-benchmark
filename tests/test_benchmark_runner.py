import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("benchmark_runner", PROJECT_ROOT / "scripts" / "run_official_benchmark.py")
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def config():
    return {
        "schema_version": "benchmark/v0.6.5", "checkpoint": "checkpoints/resnet18_cifar10.pth",
        "dataset": "CIFAR-10", "data_root": "data", "evaluation_samples": 10000,
        "calibration_manifest": "configs/calibration/resnet18_silu_piecewise_v06_ort_cpu.json",
        "calibration_samples": 2560, "calibration_batch_size": 128, "seed": 20260826,
        "provider": "CPUExecutionProvider", "evaluation_batch_size": 128, "latency_batch_size": 1,
        "latency_warmup_runs": 1, "latency_timed_runs": 2, "throughput_batch_size": 2,
        "throughput_warmup_runs": 1, "throughput_timed_runs": 2, "standard_qdq": {},
        "output_directory": "results/benchmarks/v0.6.5",
    }


class BenchmarkRunnerTests(unittest.TestCase):
    def test_config_and_deterministic_selection(self):
        self.assertEqual(runner.validate_config(config())["dataset"], "CIFAR-10")
        with self.assertRaisesRegex(ValueError, "missing"):
            runner.validate_config({})
        self.assertEqual(runner.select_sample_indices(10000, 10000, False), list(range(10000)))
        self.assertEqual(runner.select_sample_indices(10000, 10000, True), list(range(128)))

    def test_fingerprint_artifact_resume_and_force(self):
        first = runner.build_fingerprint(config(), {"a": "1"}, "1.19")
        self.assertEqual(first, runner.build_fingerprint(config(), {"a": "1"}, "1.19"))
        self.assertNotEqual(first, runner.build_fingerprint(config(), {"a": "2"}, "1.19"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.onnx"; path.write_bytes(b"model")
            self.assertTrue(runner.needs_rebuild(path, first, True, False))
            runner.atomic_json(runner.metadata(path), {"fingerprint": first})
            self.assertFalse(runner.needs_rebuild(path, first, True, False))
            self.assertTrue(runner.needs_rebuild(path, first, False, False))
            self.assertTrue(runner.needs_rebuild(path, first, True, True))
            runner.atomic_text(runner.metadata(path), "not json")
            self.assertTrue(runner.needs_rebuild(path, first, True, False))

    def test_metrics_memory_and_semantics(self):
        metrics = runner.q([0.001, 0.003, 0.002])
        self.assertEqual(metrics["p50_ms"], 2.0)
        self.assertAlmostEqual(2 / 0.5, 4.0)
        available = runner.process_memory_snapshot(lambda: 42)
        self.assertEqual((available["status"], available["bytes"]), ("available", 42))
        unavailable = runner.process_memory_snapshot(lambda: (_ for _ in ()).throw(RuntimeError("blocked")))
        self.assertEqual(unavailable["status"], "unavailable")
        matched = runner.semantic_metrics(np.array([[1, 2]], np.float32), np.array([[1, 4]], np.float32))
        self.assertEqual(matched["mapping_status"], "matched")
        self.assertEqual(matched["shape"], [1, 2])
        self.assertEqual(runner.semantic_metrics(np.zeros((1,)), np.zeros((2,)))["mapping_status"], "unavailable")
        self.assertEqual(runner.unavailable_site("act.call_0", "not proven")["mapping_status"], "unavailable")

    def test_output_isolation_reports_and_partial_promotion(self):
        self.assertNotEqual(runner.output_path(config(), False), runner.output_path(config(), True))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); stage = root / "partial"; stage.mkdir()
            runner.atomic_json(stage / "run_status.json", {"status": "failed"})
            with self.assertRaisesRegex(RuntimeError, "partial-result"):
                runner.publish_stage(stage, root / "official")
            result = {
                "completion_status": "smoke-success", "fingerprint": "f", "variants": [{
                    "variant_id": "fp32_ort_cpu", "accuracy": .5, "latency": {"p50_ms": 1},
                    "throughput_images_per_second": 2, "memory": {"status": "unavailable"},
                    "graph_facts": {"quantize_linear_nodes": 0, "dequantize_linear_nodes": 0,
                    "baseline_silu_patterns": 17, "piecewise_sites": 0, "model_size_bytes": 1, "facts": {"ok": True}},
                }], "semantic_activation_site_analysis": {"site_count": 17}, "artifact_sha256": {"base": "x"},
            }
            runner.write_reports(stage, {"provider": "CPUExecutionProvider", "onnxruntime": "test", "cpu_count": 1, "thread_settings": {}}, result)
            self.assertTrue((stage / "benchmark_results.json").exists())
            self.assertTrue((stage / "benchmark_results.csv").exists())
            self.assertIn("Comparability caveat", (stage / "benchmark_report.md").read_text())
            self.assertTrue((stage / "graph_reports" / "fp32_ort_cpu.json").exists())


if __name__ == "__main__":
    unittest.main()
