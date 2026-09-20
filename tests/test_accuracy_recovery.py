import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from silu_benchmark.accuracy_recovery import (
    build_candidate_manifest,
    build_noop_control,
    deterministic_development_split,
    discover_qdq_silu_sites,
    piecewise_error_summary,
    rank_sensitivity,
    rewrite_qdq_silu,
    select_candidate,
    sha256_file,
    split_digest,
    summarize_values,
    validate_candidate_manifest,
    validate_generated_output_path,
)


ROOT = Path(__file__).resolve().parents[1]
QDQ = ROOT / "artifacts/int8/resnet18_silu_int8_v065.onnx"
FP32 = ROOT / "artifacts/onnx/resnet18_silu_fp32.onnx"
ORIGINAL = ROOT / "artifacts/onnx/resnet18_silu_piecewise_v065.onnx"
MANIFEST = ROOT / "configs/calibration/resnet18_silu_piecewise_v06_ort_cpu.json"
HISTORICAL = ROOT / "results/benchmarks/v0.6.5/benchmark_results.json"


@unittest.skipUnless(QDQ.exists(), "locked Standard-QDQ integration asset is unavailable")
class AccuracyRecoveryGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = onnx.load(str(QDQ))
        cls.sites = discover_qdq_silu_sites(cls.model)

    def test_exact_ordered_qdq_target_mapping(self):
        self.assertEqual(len(self.sites), 17)
        self.assertEqual(self.sites[0].site_id, "act.call_0")
        self.assertEqual(self.sites[-1].site_id, "layer4.1.act.call_1")
        self.assertEqual(len({index for site in self.sites for index in site.replaced_node_indices}), 102)

    def test_missing_and_duplicate_topology_is_rejected(self):
        missing = copy.deepcopy(self.model)
        missing.graph.node[self.sites[0].sigmoid_quantize_node_index].op_type = "Identity"
        with self.assertRaisesRegex(ValueError, "QuantizeLinear"):
            discover_qdq_silu_sites(missing)
        duplicate = copy.deepcopy(self.model)
        duplicate.graph.node.extend([copy.deepcopy(duplicate.graph.node[self.sites[0].sigmoid_node_index])])
        with self.assertRaises(ValueError):
            discover_qdq_silu_sites(duplicate)

    def test_noop_and_float_controls_construct_and_run(self):
        noop = build_noop_control(self.model)
        floating = rewrite_qdq_silu(self.model, mode="float_equivalent")
        onnx.checker.check_model(noop.model)
        onnx.checker.check_model(floating.model)
        rng = np.random.default_rng(20260828)
        values = rng.standard_normal((1, 3, 32, 32), dtype=np.float32)
        original_session = ort.InferenceSession(
            self.model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        noop_session = ort.InferenceSession(
            noop.model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        floating_session = ort.InferenceSession(
            floating.model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        original = original_session.run(None, {"images": values})[0]
        nooped = noop_session.run(None, {"images": values})[0]
        floated = floating_session.run(None, {"images": values})[0]
        np.testing.assert_array_equal(original, nooped)
        self.assertEqual(floated.shape, original.shape)
        self.assertEqual(floated.dtype, np.float32)

    def test_candidate_manifest_hash_is_stable_and_rejects_tampering(self):
        rng = np.random.default_rng(12)
        values = {
            site.site_id: np.concatenate(
                [rng.uniform(-0.278, -0.001, 256), rng.uniform(0.001, 3.0, 1024)]
            )
            for site in self.sites
        }
        candidate = {
            "candidate_id": "test_p99_9",
            "method": "two-segment-piecewise-asymmetric",
            "vmax_strategy": "percentile",
            "vmax_percentile": 99.9,
            "vsplit_strategy": "sample_mse_grid",
            "vsplit_candidate_count": 8,
            "bits": 8,
        }
        first, specs = build_candidate_manifest(
            candidate=candidate, site_values=values, sites=self.sites, provenance={"test": True}
        )
        second, _ = build_candidate_manifest(
            candidate=candidate, site_values=values, sites=self.sites, provenance={"test": True}
        )
        self.assertEqual(first, second)
        self.assertEqual(len(specs), 17)
        changed = copy.deepcopy(first)
        changed["sites"][0]["parameters"]["vmax"] += 1.0
        with self.assertRaises(ValueError):
            validate_candidate_manifest(changed, [site.site_id for site in self.sites])


class AccuracyRecoveryPureTests(unittest.TestCase):
    def test_development_split_is_deterministic_and_disjoint(self):
        calibration = list(range(2560))
        first = deterministic_development_split(
            total=10000, calibration_indices=calibration, sample_count=2000, seed=20260828
        )
        second = deterministic_development_split(
            total=10000, calibration_indices=calibration, sample_count=2000, seed=20260828
        )
        self.assertEqual(first, second)
        self.assertFalse(set(first) & set(calibration))
        self.assertEqual(len(first), len(set(first)))
        images = np.arange(40, dtype=np.uint8).reshape(10, 4)
        labels = np.arange(10, dtype=np.int64)
        self.assertEqual(
            split_digest(images, labels, [2, 4], "development"),
            split_digest(images, labels, [2, 4], "development"),
        )
        self.assertNotEqual(
            split_digest(images, labels, [2, 4], "development"),
            split_digest(images, labels, [2, 4], "final_test"),
        )

    def test_statistics_clipping_and_occupancy(self):
        from silu_benchmark.quantization import PiecewiseQuantizationSpec

        values = np.array([-2.0, -0.5, 0.0, 0.2, 0.8, 2.0])
        summary = summarize_values(values)
        self.assertEqual(summary["count"], 6)
        error = piecewise_error_summary(values, PiecewiseQuantizationSpec(-0.5, 0.2, 0.8))
        self.assertEqual(error["clipped_below_count"], 1)
        self.assertEqual(error["clipped_above_count"], 1)
        self.assertGreater(error["occupied_code_count"], 1)
        self.assertAlmostEqual(error["lower_segment_fraction"] + error["upper_segment_fraction"], 1.0)

    def test_sensitivity_ranking_and_development_only_selection(self):
        ranked = rank_sensitivity(
            [
                {"site_id": "b", "development_accuracy": 0.9, "logit_mae": 0.1},
                {"site_id": "a", "development_accuracy": 0.8, "logit_mae": 0.2},
            ]
        )
        self.assertEqual([row["site_id"] for row in ranked], ["a", "b"])
        rows = [
            {"candidate_id": "a", "split_role": "development", "status": "success", "accuracy": 0.9,
             "prediction_agreement_vs_standard_qdq": 0.95},
            {"candidate_id": "b", "split_role": "development", "status": "success", "accuracy": 0.91,
             "prediction_agreement_vs_standard_qdq": 0.90},
        ]
        self.assertEqual(select_candidate(rows, split_role="development")["candidate_id"], "b")
        with self.assertRaisesRegex(ValueError, "development"):
            select_candidate(rows, split_role="final_test")
        contaminated = [{**rows[0], "final_test_accuracy": 1.0}]
        with self.assertRaisesRegex(ValueError, "final-test"):
            select_candidate(contaminated, split_role="development")

    def test_generated_output_path_isolation(self):
        allowed = validate_generated_output_path(
            ROOT / "results/benchmarks/v1.1_accuracy_recovery_smoke", ROOT
        )
        self.assertTrue(str(allowed).endswith("v1.1_accuracy_recovery_smoke"))
        with self.assertRaises(ValueError):
            validate_generated_output_path(ROOT / "results/benchmarks/v0.6.5", ROOT)
        with self.assertRaises(ValueError):
            validate_generated_output_path(ROOT / "outside", ROOT)

    @unittest.skipUnless(all(path.exists() for path in (QDQ, FP32, ORIGINAL, MANIFEST, HISTORICAL)),
                         "locked provenance assets are unavailable")
    def test_canonical_assets_match_locked_v065_hashes(self):
        report = json.loads(HISTORICAL.read_text(encoding="utf-8"))
        self.assertEqual(sha256_file(QDQ), report["artifact_sha256"]["standard_static_qdq"])
        self.assertEqual(sha256_file(FP32), report["artifact_sha256"]["base_onnx"])
        self.assertEqual(sha256_file(ORIGINAL), report["artifact_sha256"]["silu_piecewise_reference"])
        self.assertEqual(sha256_file(MANIFEST), report["artifact_sha256"]["manifest"])

    def test_ort_only_import_does_not_import_torch(self):
        program = r'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
import silu_benchmark.accuracy_recovery
assert "torch" not in sys.modules
assert "torchvision" not in sys.modules
'''
        result = subprocess.run(
            [sys.executable, "-c", program], cwd=ROOT, capture_output=True, text=True, timeout=60
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
