import importlib.util
import json
import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx

from silu_benchmark.benchmark_data import normalize_cifar_images
from silu_benchmark.benchmark_semantics import semantic_output_mappings, instrument_model


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
            runner.atomic_json(runner.metadata(path), {"fingerprint": first, "sha256": runner.digest(path)})
            self.assertFalse(runner.needs_rebuild(path, first, True, False))
            self.assertTrue(runner.needs_rebuild(path, first, False, False))
            self.assertTrue(runner.needs_rebuild(path, first, True, True))
            path.write_bytes(b"tampered model")
            self.assertTrue(runner.needs_rebuild(path, first, True, False))
            runner.atomic_text(runner.metadata(path), "not json")
            self.assertTrue(runner.needs_rebuild(path, first, True, False))

    def test_metrics_memory_and_semantics(self):
        metrics = runner.q([0.001, 0.003, 0.002])
        self.assertEqual(metrics["p50_ms"], 2.0)
        self.assertAlmostEqual(metrics["p95_ms"], 2.9)
        available = runner.process_memory_snapshot(lambda: 42)
        self.assertEqual((available["status"], available["bytes"]), ("available", 42))
        unavailable = runner.process_memory_snapshot(lambda: (_ for _ in ()).throw(RuntimeError("blocked")))
        self.assertEqual(unavailable["status"], "unavailable")
        matched = runner.semantic_metrics(np.array([[1, 2]], np.float32), np.array([[1, 4]], np.float32))
        self.assertEqual(matched["mapping_status"], "matched")
        self.assertEqual(matched["shape"], [1, 2])
        self.assertEqual(matched["mse"], 2.0)
        self.assertEqual(matched["mae"], 1.0)
        self.assertEqual(matched["max_absolute_error"], 2.0)
        self.assertAlmostEqual(matched["cosine_similarity"], 9 / (5 * 17) ** .5)
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

    def test_clean_import_and_smoke_setup_never_import_torch(self):
        program = r'''
import importlib.abc, sys, tempfile
from pathlib import Path
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torchvision'):
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, NoTorch())
sys.path.insert(0, str(Path.cwd() / 'scripts'))
import run_official_benchmark as r
from silu_benchmark.benchmark_data import load_cifar_batch, numpy_batches, load_ort_quantization
original_find_spec = importlib.util.find_spec
load_ort_quantization()
assert importlib.util.find_spec is original_find_spec
config = r.validate_config(r.json.loads((r.ROOT / 'configs/benchmarks/resnet18_silu_cifar10_v065_ort_cpu.json').read_text()))
r.load_site_spec_manifest(r.ROOT / config['calibration_manifest'])
r.assert_ort_only()
with tempfile.TemporaryDirectory() as directory:
    config['output_directory'] = str(Path(directory) / 'official')
    stage, destination = r.setup_run_paths(config, True)
    assert stage.exists() and destination.name == 'official_smoke'
    assert r.effective_settings(config, True)['evaluation_samples'] == 128
assert 'torch' not in sys.modules and 'torchvision' not in sys.modules
'''
        result = subprocess.run([sys.executable, "-c", program], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_new_staging_paths_preserve_existing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = {**config(), "output_directory": str(Path(directory) / "official")}
            previous = Path(directory) / "official_smoke"
            previous.mkdir()
            marker = previous / "keep.txt"
            marker.write_text("untouched")
            first, destination = runner.setup_run_paths(cfg, True)
            second, _ = runner.setup_run_paths(cfg, True)
            self.assertNotEqual(first, second)
            self.assertNotEqual(destination, previous)
            self.assertEqual(marker.read_text(), "untouched")

    def test_native_and_python_worker_failures_are_terminal(self):
        for program, expected_exit in (("raise RuntimeError('intentional failure')", 1),
                                       ("import os; os._exit(3)", 3)):
            with self.subTest(program=program), tempfile.TemporaryDirectory() as directory:
                stage = Path(directory) / "partial"
                stage.mkdir()
                runner.update_status(stage, status="running", phase="semantic activation-site analysis")
                destination = Path(directory) / "official"
                result = runner.supervise_worker([sys.executable, "-c", program], stage, destination)
                status = json.loads((stage / "run_status.json").read_text())
                self.assertEqual(result, expected_exit)
                self.assertEqual(status["status"], "failed")
                self.assertEqual(status["worker_exit_code"], expected_exit)
                self.assertEqual(status["phase"], "semantic activation-site analysis")
                self.assertIn("action", status)
                self.assertFalse(destination.exists())

    def test_zero_exit_without_complete_reports_is_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory) / "partial"
            stage.mkdir()
            runner.update_status(stage, status="running", phase="test")
            self.assertEqual(runner.supervise_worker([sys.executable, "-c", "pass"], stage,
                                                    Path(directory) / "official"), 1)
            self.assertEqual(json.loads((stage / "run_status.json").read_text())["status"], "failed")

    def test_fingerprint_separates_smoke_and_official(self):
        self.assertNotEqual(runner.build_fingerprint(config(), {}, "test", True),
                            runner.build_fingerprint(config(), {}, "test", False))

    def test_numpy_preprocessing_matches_existing_transform(self):
        # Torch is used only in this compatibility test, never by the benchmark.
        from silu_benchmark.data import cifar10_transform
        from PIL import Image
        pixels = np.arange(3 * 32 * 32, dtype=np.uint16).astype(np.uint8).reshape(3, 32, 32)
        expected = cifar10_transform()(Image.fromarray(pixels.transpose(1, 2, 0))).numpy()
        actual = normalize_cifar_images(pixels[None])[0]
        np.testing.assert_array_equal(expected, actual)


class BenchmarkInstrumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = onnx.load(str(PROJECT_ROOT / "artifacts/onnx/resnet18_silu_fp32.onnx"))
        cls.manifest, specs = runner.load_manifest(
            PROJECT_ROOT / "configs/calibration/resnet18_silu_piecewise_v06_ort_cpu.json")
        cls.custom = runner.rewrite_silu_piecewise_model(cls.baseline, specs).model

    def test_ordered_17_sites_and_logits_do_not_mutate_source(self):
        before = self.baseline.SerializeToString()
        custom_before = self.custom.SerializeToString()
        fp32, custom = semantic_output_mappings(self.baseline, self.custom, self.manifest)
        expected = [entry["site_id"] for entry in self.manifest["sites"]] + ["logits"]
        self.assertEqual(list(fp32), expected)
        self.assertEqual(list(custom), expected)
        self.assertEqual(len(expected), 18)
        for model, mapping in ((self.baseline, fp32), (self.custom, custom)):
            instrumented = instrument_model(model, mapping)
            self.assertEqual([item.name for item in instrumented.graph.output], list(mapping.values()))
            self.assertEqual(len(instrumented.graph.output), 18)
        self.assertEqual(before, self.baseline.SerializeToString())
        self.assertEqual(custom_before, self.custom.SerializeToString())
        self.assertEqual((fp32, custom), semantic_output_mappings(self.baseline, self.custom, self.manifest))

    def test_unproven_or_reordered_sites_are_rejected(self):
        changed = copy.deepcopy(self.manifest)
        changed["sites"].reverse()
        with self.assertRaisesRegex(ValueError, "ordered"):
            semantic_output_mappings(self.baseline, self.custom, changed)
        changed_custom = copy.deepcopy(self.custom)
        record = next(item for item in changed_custom.metadata_props
                      if item.key == "silu_benchmark.piecewise_rewrite_sites")
        entries = json.loads(record.value)
        entries[0]["dequantized_output_tensor"] = "guessed_tensor"
        record.value = json.dumps(entries)
        with self.assertRaisesRegex(ValueError, "unproven"):
            semantic_output_mappings(self.baseline, changed_custom, self.manifest)


if __name__ == "__main__":
    unittest.main()
