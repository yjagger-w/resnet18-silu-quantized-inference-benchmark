import copy
import unittest

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import onnx

from silu_benchmark.calibration_manifest import (
    SCHEMA_VERSION,
    build_manifest,
    collect_silu_callsite_activations,
    validate_manifest,
)
from silu_benchmark.backends import discover_silu_patterns


class SharedSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x) + self.act(x + 0.1)


class ChangingCallOrder(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.SiLU()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        output = self.act(x)
        return output + (self.act(x) if self.calls == 1 else 0)


class CalibrationManifestTests(unittest.TestCase):
    def setUp(self):
        values = np.linspace(-0.25, 1.5, 1000, dtype=np.float32)
        self.manifest = build_manifest(
            site_activations={"act.call_0": values, "block.act.call_1": values},
            metadata={"model_architecture": "test", "checkpoint": {"logical_id": "test", "sha256": "0"}, "calibration_dataset": {"logical_id": "test"}, "calibration": {"seed": 1}},
        )

    def test_schema_positive_splits_and_derived_values(self):
        specs = validate_manifest(self.manifest)
        self.assertEqual(self.manifest["schema_version"], SCHEMA_VERSION)
        self.assertEqual(list(specs), ["act.call_0", "block.act.call_1"])
        self.assertTrue(all(spec.vsplit > 0 for spec in specs.values()))

    def test_legacy_negative_duplicate_and_derived_errors_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "legacy"):
            validate_manifest({"sites": []})
        negative = copy.deepcopy(self.manifest)
        negative["sites"][0]["vsplit"] = -0.1
        with self.assertRaises(ValueError):
            validate_manifest(negative)
        duplicate = copy.deepcopy(self.manifest)
        duplicate["sites"][1]["site_id"] = "act.call_0"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_manifest(duplicate)
        bad_scale = copy.deepcopy(self.manifest)
        bad_scale["sites"][0]["lower_scale"] *= 2
        with self.assertRaisesRegex(ValueError, "derived"):
            validate_manifest(bad_scale)

    def test_shared_module_calls_are_collected_separately(self):
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.ones(4, 1), torch.zeros(4)), batch_size=2)
        values = collect_silu_callsite_activations(SharedSiLU(), loader, torch.device("cpu"))
        self.assertEqual(list(values), ["act.call_0", "act.call_1"])
        self.assertFalse(np.array_equal(values["act.call_0"], values["act.call_1"]))

    def test_callsite_order_changes_fail(self):
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.ones(4, 1), torch.zeros(4)), batch_size=2)
        with self.assertRaisesRegex(RuntimeError, "ordering changed"):
            collect_silu_callsite_activations(ChangingCallOrder(), loader, torch.device("cpu"))

    def test_real_asset_manifest_if_available(self):
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "configs" / "calibration" / "resnet18_silu_piecewise_v06.json"
        checkpoint = root / "checkpoints" / "resnet18_cifar10.pth"
        baseline = root / "artifacts" / "onnx" / "resnet18_silu_fp32.onnx"
        data = root / "data" / "cifar-10-batches-py"
        if not all(path.exists() for path in (manifest_path, checkpoint, baseline, data)):
            self.skipTest("real integration prerequisite missing: manifest/checkpoint/baseline ONNX/CIFAR-10 train data")
        import json
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        specs = validate_manifest(payload)
        sites = discover_silu_patterns(onnx.load(str(baseline)))
        self.assertEqual(set(specs), {site.site_id for site in sites})
        self.assertEqual(len(specs), 17)


if __name__ == "__main__":
    unittest.main()
