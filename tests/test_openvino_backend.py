"""v0.8 pure/fixture tests; optional real CPU checks never import PyTorch."""

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper

from silu_benchmark.backends import openvino_backend as backend

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_openvino_standard_qdq_benchmark as runner


def config():
    return json.loads((ROOT / "configs/benchmarks/resnet18_silu_cifar10_v08_openvino_cpu.json").read_text())


IR_XML = '<net name="fixture" version="11"><layers><layer id="0" type="Parameter"/><layer id="1" type="FakeQuantize"/><layer id="2" type="Convert"><data element_type="i8"/></layer><layer id="3" type="Result"/></layers></net>'


class FakeCore:
    available_devices = ["CPU"]

    def read_model(self, path, weights=None):
        return path


def fake_runtime():
    def serialize(model, xml, binary, version):
        assert version == "IR_V11"
        Path(xml).write_text(IR_XML)
        Path(binary).write_bytes(b"synthetic test weights")
    return SimpleNamespace(Core=FakeCore, get_version=lambda: "fixture-version", serialize=serialize)


class OpenVINOBackendTests(unittest.TestCase):
    def test_config_and_cli_validation(self):
        self.assertEqual(backend.validate_config(config())["device"], "CPU")
        for key, value in (("device", "GPU"), ("device", "NPU"), ("tolerance", 0.01),
                           ("evaluation_samples", 128), ("latency_timed_runs", 0)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                backend.validate_config({**config(), key: value})
        with self.assertRaisesRegex(ValueError, "missing"):
            backend.validate_config({})
        args = runner.parse_args(["--smoke", "--resume", "--force-rebuild", "--model", "copy.onnx", "--output", "results/benchmarks/v0.8_test"])
        self.assertTrue(args.smoke and args.resume and args.force_rebuild)
        self.assertEqual(args.model, Path("copy.onnx"))

    def test_fingerprint_is_deterministic_and_version_sensitive(self):
        first = backend.ir_fingerprint("a" * 64, "version")
        self.assertEqual(first, backend.ir_fingerprint("a" * 64, "version", dict(reversed(list(backend.CONVERSION.items())))))
        self.assertNotEqual(first, backend.ir_fingerprint("b" * 64, "version"))
        self.assertNotEqual(first, backend.ir_fingerprint("a" * 64, "new-version"))
        with self.assertRaisesRegex(ValueError, "CPU"):
            backend.ir_fingerprint("a" * 64, "version", device="GPU")

    def test_stale_ir_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            (bundle / "model.xml").write_text(IR_XML)
            (bundle / "model.bin").write_bytes(b"test")
            metadata = {"status": "complete", "fingerprint": "fp", "xml_sha256": backend.sha256(bundle / "model.xml"),
                        "bin_sha256": backend.sha256(bundle / "model.bin")}
            backend.atomic_json(bundle / "metadata.json", metadata)
            self.assertTrue(backend.valid_ir(bundle, "fp"))
            self.assertFalse(backend.valid_ir(bundle, "changed"))
            (bundle / "model.bin").write_bytes(b"tampered")
            self.assertFalse(backend.valid_ir(bundle, "fp"))
            backend.atomic_json(bundle / "metadata.json", [])
            self.assertFalse(backend.valid_ir(bundle, "fp"))

    def test_generated_paths_and_smoke_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(runner, "ROOT", root):
                first, smoke = runner.stage_paths(config(), True)
                second, official = runner.stage_paths(config(), False)
                self.assertNotEqual(first, second)
                self.assertNotEqual(smoke, official)
                self.assertIn("smoke", smoke.name)
            for unsafe, kind in (("results/benchmarks/v0.6.5", "report"), ("configs/generated.xml", "ir"),
                                 ("artifacts/openvino/v0.8/../../outside", "ir")):
                with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                    backend.generated_path(root, unsafe, kind)

    def test_runtime_unavailable_message_is_actionable(self):
        with patch.object(backend.importlib, "import_module", side_effect=ModuleNotFoundError("openvino")):
            with self.assertRaisesRegex(backend.OpenVINOUnavailableError, "Do not install automatically") as error:
                backend.require_openvino()
        self.assertIn('pip install "openvino==2025.3.0"', str(error.exception))

    def test_ir_graph_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            xml = Path(temporary) / "model.xml"
            xml.write_text(IR_XML)
            facts = backend.ir_graph_report(xml)
            self.assertEqual(facts["operation_count"], 4)
            self.assertEqual(facts["operation_types"]["FakeQuantize"], 1)
            self.assertTrue(facts["fake_quantize_present"])
            self.assertEqual(facts["declared_element_types"], {"i8": 1})
            self.assertIn("do not prove", facts["caveat"])
            xml.write_text("<unrelated/>")
            with self.assertRaises(ValueError):
                backend.ir_graph_report(xml)

    def test_metrics_retain_disagreements(self):
        first = np.zeros((2, 10), np.float32)
        second = first.copy()
        first[0, 1], first[1, 2] = 2, 2
        second[0, 1], second[1, 3] = 2, 2
        comparison = backend.compare_outputs(first, second, np.array([1, 2]))
        self.assertEqual(comparison["prediction_agreement"], 0.5)
        self.assertEqual(comparison["prediction_disagreement_indices"], [1])
        self.assertEqual(comparison["logit_max_absolute_error"], 2.0)
        self.assertAlmostEqual(comparison["logit_mean_absolute_error"], .2)
        self.assertEqual(comparison["ort_top1_accuracy"], 1.0)
        self.assertEqual(comparison["openvino_top1_accuracy"], .5)
        self.assertEqual(comparison["tolerance"]["status"], "not_established")
        with self.assertRaises(ValueError):
            backend.compare_outputs(first, second[:1], np.array([1, 2]))
        second[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            backend.compare_outputs(first, second, np.array([1, 2]))

    def test_conversion_resume_and_force_preserve_source_and_bundles(self):
        # A fake serializer exercises orchestration only; this is not an OpenVINO result.
        with tempfile.TemporaryDirectory() as temporary, patch.object(backend, "require_openvino", return_value=fake_runtime()):
            root = Path(temporary)
            source = root / "source.onnx"
            source.write_bytes(b"synthetic ONNX fixture")
            cfg = config()
            cfg["source_sha256"] = backend.sha256(source)
            proof = {"path": str(source), "sha256": cfg["source_sha256"]}
            first = backend.convert_standard_qdq(root, cfg, proof)
            resumed = backend.convert_standard_qdq(root, cfg, proof, resume=True)
            forced = backend.convert_standard_qdq(root, cfg, proof, resume=True, force_rebuild=True)
            self.assertTrue(resumed["reused"])
            self.assertEqual(first["directory"], resumed["directory"])
            self.assertNotEqual(first["directory"], forced["directory"])
            self.assertTrue(Path(first["directory"]).exists())
            self.assertEqual(backend.sha256(source), cfg["source_sha256"])
            with self.assertRaisesRegex(ValueError, "exact verified"):
                backend.convert_standard_qdq(root, cfg, {**proof, "sha256": "0" * 64})

    def test_failed_conversion_is_recorded(self):
        runtime = fake_runtime()
        runtime.serialize = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unsupported test op"))
        with tempfile.TemporaryDirectory() as temporary, patch.object(backend, "require_openvino", return_value=runtime):
            root = Path(temporary)
            source = root / "source.onnx"
            source.write_bytes(b"fixture")
            cfg = {**config(), "source_sha256": backend.sha256(source)}
            with self.assertRaisesRegex(RuntimeError, "unsupported test op"):
                backend.convert_standard_qdq(root, cfg, {"path": str(source), "sha256": cfg["source_sha256"]})
            failures = list((root / cfg["ir_output"]).glob(".partial-*/failure.json"))
            self.assertEqual(len(failures), 1)
            self.assertEqual(json.loads(failures[0].read_text())["status"], "failed")

    def test_source_provenance_checks_report_sidecar_data_and_graph(self):
        # Self-contained provenance fixture; no ignored benchmark artifacts required.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = config()
            source = root / cfg["source_model"]
            source.parent.mkdir(parents=True)
            graph = helper.make_graph([
                helper.make_node("QuantizeLinear", ["x", "scale", "zero"], ["q"]),
                helper.make_node("DequantizeLinear", ["q", "scale", "zero"], ["y"]),
            ], "fixture", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 10])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 10])], [
                    helper.make_tensor("scale", TensorProto.FLOAT, [], [.1]),
                    helper.make_tensor("zero", TensorProto.UINT8, [], [0]),
                ])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
            onnx.save(model, str(source))
            cfg["source_sha256"] = backend.sha256(source)
            artifacts = {"standard_static_qdq": cfg["source_sha256"]}
            for key, path in (("checkpoint", cfg["checkpoint"]), ("test_batch", cfg["data_root"] + "/cifar-10-batches-py/test_batch")):
                fixture = root / path
                fixture.parent.mkdir(parents=True, exist_ok=True)
                fixture.write_bytes(b"test-only provenance fixture")
                artifacts[key] = backend.sha256(fixture)
            report_path = root / cfg["source_report"]
            report = {"completion_status": "official-success", "smoke": False, "fingerprint": "fixture",
                      "frozen_config": cfg, "artifact_sha256": artifacts,
                      "variants": [{"variant_id": backend.SOURCE_VARIANT, "accuracy": .5, "samples": 10000}]}
            backend.atomic_json(report_path, report)
            backend.atomic_json(report_path.parent / "run_status.json", {"status": "success", "smoke": False, "fingerprint": "fixture"})
            backend.atomic_json(root / cfg["source_sidecar"], {"sha256": cfg["source_sha256"], "fingerprint": "fixture"})
            proof = backend.verify_source(root, cfg)
            self.assertEqual(proof["sha256"], cfg["source_sha256"])
            self.assertEqual((proof["q_nodes"], proof["dq_nodes"]), (1, 1))
            wrong = root / "wrong.onnx"
            wrong.write_bytes(b"not the deployment artifact")
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                backend.verify_source(root, cfg, wrong)
            with self.assertRaisesRegex(ValueError, "protocol"):
                backend.verify_source(root, {**cfg, "seed": 1})
            backend.atomic_json(report_path, {**report, "smoke": True})
            with self.assertRaisesRegex(ValueError, "OFFICIAL"):
                backend.verify_source(root, cfg)

    def test_cpu_adapter_inference_and_safe_properties(self):
        class FakeCompiled:
            inputs, outputs = ["x"], ["y"]

            def output(self, index):
                return self.outputs[index]

            def get_property(self, name):
                raise RuntimeError("fixture property unavailable")

            def create_infer_request(self):
                def infer(values, share_inputs, share_outputs):
                    self_test.assertFalse(share_inputs or share_outputs)
                    self_test.assertEqual(values[0].dtype, np.float32)
                    self_test.assertTrue(values[0].flags.c_contiguous)
                    return {"y": np.zeros((len(values[0]), 10), np.float32)}
                return SimpleNamespace(infer=infer)

        class CompileCore(FakeCore):
            def compile_model(self, model, device, properties):
                self_test.assertEqual(device, "CPU")
                self_test.assertEqual(properties, {"PERFORMANCE_HINT": "LATENCY"})
                return FakeCompiled()

        self_test = self
        runtime = fake_runtime()
        runtime.Core = CompileCore
        with tempfile.TemporaryDirectory() as temporary, patch.object(backend, "require_openvino", return_value=runtime):
            root = Path(temporary)
            source = root / "fixture.onnx"
            source.write_bytes(b"fixture")
            cfg = {**config(), "source_sha256": backend.sha256(source)}
            bundle = backend.convert_standard_qdq(root, cfg, {"path": str(source), "sha256": cfg["source_sha256"]})
            adapter = backend.OpenVINOCPU(bundle, cfg["compile_properties"])
            self.assertEqual(adapter.predict(np.ones((2, 3, 32, 32), np.float64)).shape, (2, 10))
            self.assertEqual(adapter.capabilities["compiled_properties"]["NUM_STREAMS"]["status"], "unavailable")
            with self.assertRaisesRegex(ValueError, "CPU"):
                backend.OpenVINOCPU(bundle, cfg["compile_properties"], "GPU")
            with patch.object(CompileCore, "compile_model", side_effect=RuntimeError("unsupported fixture op")):
                with self.assertRaisesRegex(RuntimeError, "no device fallback"):
                    backend.OpenVINOCPU(bundle, cfg["compile_properties"])

    def test_report_formats_separate_historical_and_new_results(self):
        comparison = backend.compare_outputs(np.zeros((2, 10)), np.zeros((2, 10)), np.array([0, 1]))
        timing = {"p50_ms": 1.0, "p95_ms": 2.0, "images_per_second": 1000.0, "memory": {"status": "unavailable"}}
        result = {"completion_status": "smoke-success", "smoke": True, "config": config(), "effective_settings": {},
                  "accuracy": {"samples": 2, "correct": 1, "top1_accuracy": .5}, "validation": comparison,
                  "latency": timing, "throughput": timing, "environment": {"fixture": True},
                  "source": {"official_ort_v065": {"accuracy": .9}, "q_nodes": 1, "dq_nodes": 1},
                  "conversion": {"ir_graph": {"fixture": True}}, "capabilities": {"caveat": backend.CAPABILITY_CAVEAT}}
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            runner.write_reports(stage, result)
            saved = json.loads((stage / "benchmark_results.json").read_text())
            self.assertEqual(saved, result)
            with (stage / "benchmark_results.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["variant_id"], "standard_static_qdq_openvino_cpu")
            self.assertEqual(float(rows[0]["top1_accuracy"]), .5)
            markdown = (stage / "benchmark_report.md").read_text()
            self.assertIn("Historical ORT", markdown)
            self.assertIn("No speedup is claimed", markdown)
            for name in ("environment.json", "validation.json", "graph_report.json"):
                self.assertTrue((stage / name).exists())

    def test_new_modules_do_not_import_torch_or_openvino_eagerly(self):
        program = r'''
import importlib.abc, sys
from pathlib import Path
class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torchvision', 'openvino'):
            raise AssertionError('unexpected eager import: ' + fullname)
sys.meta_path.insert(0, BlockFrameworks())
sys.path.insert(0, str(Path.cwd()/'scripts'))
import run_openvino_standard_qdq_benchmark as r
import convert_openvino_standard_qdq
r.parse_args(['--smoke'])
r.assert_no_torch()
assert 'openvino' not in sys.modules
'''
        result = subprocess.run([sys.executable, "-c", program], cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_supervisor_records_native_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / "partial"
            stage.mkdir()
            runner.update_status(stage, status="running", phase="CPU compile")
            result = runner.supervise([sys.executable, "-c", "import os; os._exit(3)"], stage,
                                       Path(temporary) / "official", False)
            self.assertEqual(result, 3)
            self.assertEqual(json.loads((stage / "run_status.json").read_text())["status"], "failed")
            self.assertFalse((Path(temporary) / "official").exists())

    @unittest.skipUnless(importlib.util.find_spec("openvino"), "OpenVINO absent: real CPU integration not run")
    def test_optional_real_cpu_integration(self):
        program = r'''
import json, pathlib, tempfile, numpy as np
from silu_benchmark.backends.openvino_backend import verify_source, convert_standard_qdq, OpenVINOCPU
root=pathlib.Path.cwd()
cfg=json.loads((root/'configs/benchmarks/resnet18_silu_cifar10_v08_openvino_cpu.json').read_text())
source=verify_source(root,cfg)
with tempfile.TemporaryDirectory() as temporary:
    bundle=convert_standard_qdq(pathlib.Path(temporary),cfg,source)
    backend=OpenVINOCPU(bundle,cfg['compile_properties'])
    result=backend.predict(np.zeros((1,3,32,32),np.float32))
    assert result.shape==(1,10) and np.all(np.isfinite(result))
'''
        result = subprocess.run([sys.executable, "-c", program], cwd=ROOT, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
