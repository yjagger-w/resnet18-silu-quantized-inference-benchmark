import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from export_onnx import export_onnx, load_checkpoint
from silu_benchmark.backends import (
    bind_module_specs_to_sites,
    discover_silu_patterns,
    inspect_onnx_model,
    load_site_spec_manifest,
    rewrite_silu_piecewise_model,
    save_rewrite_result,
    write_inspection_report,
)
from silu_benchmark.models import ResNet18
from silu_benchmark.quantization import PiecewiseQuantizationSpec, piecewise_quantize_dequantize


FULL_MODEL_ATOL = 5e-5


class ReferencePiecewiseSiLU(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.silu = nn.SiLU()

    def forward(self, values):
        return piecewise_quantize_dequantize(self.silu(values), self.spec).to(values.dtype)


def set_module(model, path, module):
    parent = model
    parts = path.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


class FullModelPiecewiseRewriteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(1234)
        cls.input_a = torch.randn(1, 3, 32, 32)
        cls.input_b = torch.randn(1, 3, 32, 32)
        cls.model = ResNet18().eval()
        load_checkpoint(cls.model, PROJECT_ROOT / "checkpoints" / "resnet18_cifar10.pth", torch.device("cpu"))
        cls.baseline_graph = onnx.load(str(PROJECT_ROOT / "artifacts" / "onnx" / "resnet18_silu_fp32.onnx"))
        cls.sites = discover_silu_patterns(cls.baseline_graph)
        # Explicit synthetic wiring fixture only: these are not calibration
        # results and are never written to the project's research artifacts.
        # Wide ranges force ordinary smoke activations to the stable zero
        # representation, isolating graph wiring from calibration quality.
        module_specs = {
            name: PiecewiseQuantizationSpec(-1_000_000.0, 0.001 + index * 0.0001, 1_000_000.0 + index)
            for index, name in enumerate(sorted({site.module_path for site in cls.sites}))
        }
        cls.site_specs = bind_module_specs_to_sites(cls.sites, module_specs)
        cls.module_specs = module_specs

    def test_discovery_mapping_and_failure_cases(self):
        self.assertEqual(len(self.sites), 17)
        self.assertEqual(len(self.site_specs), 17)
        self.assertEqual(len({site.module_path for site in self.sites}), 9)
        with self.assertRaisesRegex(ValueError, "missing"):
            rewrite_silu_piecewise_model(self.baseline_graph, dict(list(self.site_specs.items())[1:]))
        with self.assertRaisesRegex(ValueError, "unused"):
            rewrite_silu_piecewise_model(self.baseline_graph, {**self.site_specs, "extra": next(iter(self.site_specs.values()))})
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "duplicate.json"
            manifest.write_text(json.dumps({"sites": [{"site_id": "act.call_0", "vmin": -1, "vsplit": 0.1, "vmax": 1}, {"site_id": "act.call_0", "vmin": -1, "vsplit": 0.1, "vmax": 1}]}))
            with self.assertRaisesRegex(ValueError, "legacy"):
                load_site_spec_manifest(manifest)

    def test_full_model_export_rewrite_inspection_and_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            baseline_path = directory / "baseline.onnx"
            export_onnx(self.model, baseline_path, self.input_a, opset=18)
            baseline_session = ort.InferenceSession(str(baseline_path), providers=["CPUExecutionProvider"])
            with torch.no_grad():
                torch_baseline = self.model(self.input_a).numpy()
            ort_baseline = baseline_session.run(None, {"images": self.input_a.numpy()})[0]
            self.assertLessEqual(float(np.max(np.abs(torch_baseline - ort_baseline))), FULL_MODEL_ATOL)

            exported = onnx.load(str(baseline_path))
            exported_sites = discover_silu_patterns(exported)
            result = rewrite_silu_piecewise_model(exported, self.site_specs)
            self.assertEqual(len(result.sites), len(exported_sites))
            self.assertEqual(len(discover_silu_patterns(result.model)), 0)
            onnx.checker.check_model(result.model)
            with self.assertRaisesRegex(ValueError, "already"):
                rewrite_silu_piecewise_model(result.model, self.site_specs)

            rewritten_path = directory / "rewritten.onnx"
            save_rewrite_result(result, rewritten_path)
            report_path = directory / "report.json"
            report = inspect_onnx_model(rewritten_path)
            write_inspection_report(report, report_path)
            self.assertTrue(report["checker"]["passed"])
            self.assertTrue(report["runtime"]["run_passed"])
            self.assertEqual(report["inserted_piecewise_subgraph_count"], 17)
            self.assertEqual(report["baseline_silu_pattern_count"], 0)
            self.assertEqual(report["inputs"], inspect_onnx_model(baseline_path)["inputs"])
            self.assertEqual(report["outputs"], inspect_onnx_model(baseline_path)["outputs"])
            self.assertEqual(len(json.loads(report_path.read_text())["replacement_sites"]), 17)
            self.assertTrue(all(item["quantized_codes_tensor"].startswith("silu_piecewise_") for item in report["replacement_sites"]))

            reference_model = copy.deepcopy(self.model)
            for module_path, spec in self.module_specs.items():
                set_module(reference_model, module_path, ReferencePiecewiseSiLU(spec))
            reference_model.eval()
            rewritten_session = ort.InferenceSession(str(rewritten_path), providers=["CPUExecutionProvider"])
            for values in (self.input_a, self.input_b):
                with self.subTest(seed_input=float(values.flatten()[0])):
                    with torch.no_grad():
                        expected = reference_model(values).numpy()
                    actual = rewritten_session.run(None, {"images": values.numpy()})[0]
                    error = np.abs(expected - actual)
                    worst = tuple(np.unravel_index(np.argmax(error), error.shape))
                    self.assertLessEqual(float(error[worst]), FULL_MODEL_ATOL, msg=f"worst index={worst}")


if __name__ == "__main__":
    unittest.main()
