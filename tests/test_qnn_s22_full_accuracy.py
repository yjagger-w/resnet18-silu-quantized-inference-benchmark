import json
import unittest
from pathlib import Path

import numpy as np

from silu_benchmark.qnn_local_accuracy import sha256_file
from silu_benchmark.qnn_s22_accuracy import S22_FULL_REPORT_SCHEMA


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/qnn/resnet18_silu_qnn_v16.json"
RESULT_DIR = ROOT / "results/benchmarks/v1.6_qnn_cifar10_s22_full_10000"
PREDICTION_KEYS = [
    "labels",
    "original_indices",
    "fp32_local_predictions",
    "fp32_s22_predictions",
    "qdq_int8_local_predictions",
    "qdq_int8_s22_predictions",
]


class QnnS22FullAccuracyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.report = json.loads(
            (RESULT_DIR / "full_accuracy.json").read_text(encoding="utf-8")
        )

    def test_manifest_adds_full_jobs_without_replacing_synthetic_jobs(self):
        self.assertEqual(
            [model["jobs"]["inference"] for model in self.manifest["models"].values()],
            ["jprl97wkp", "j5m041w9g", "jp8e8dn8p"],
        )
        full = self.manifest["full_cifar10_evaluation"]
        self.assertEqual(full["models"]["fp32"]["inference_job"], "jpyoq7nr5")
        self.assertEqual(full["models"]["qdq_int8"]["inference_job"], "jpe78l275")
        self.assertEqual(
            full["models"]["fp32"]["remote_output_sha256"],
            "3F586D571E3E6D972CC806AD4E4A217150708E0A258AF8C755506BBDE63D0078",
        )
        self.assertEqual(
            full["models"]["qdq_int8"]["remote_output_sha256"],
            "3A5ADB12524FE7BBC2193C7688210993C78B44F35F9A0D647D917FDC8901A75D",
        )

    def test_final_report_reproduces_accuracy_agreement_and_jobs(self):
        report = self.report
        self.assertEqual(report["schema_version"], S22_FULL_REPORT_SCHEMA)
        fp32 = report["models"]["fp32"]
        qdq = report["models"]["qdq_int8"]
        self.assertEqual(fp32["jobs"], {
            "compile": "jp1ndxwlg",
            "profile": "jgjr1d7vp",
            "inference": "jpyoq7nr5",
        })
        self.assertEqual(qdq["jobs"], {
            "compile": "j5qlw227p",
            "profile": "jprl9yn9p",
            "inference": "jpe78l275",
        })
        self.assertEqual((fp32["local"]["correct"], fp32["s22"]["correct"]), (9374, 9373))
        self.assertEqual(
            fp32["local_vs_s22_prediction_agreement"]["disagreement_original_indices"],
            [5565, 8861],
        )
        self.assertEqual((qdq["local"]["correct"], qdq["s22"]["correct"]), (9357, 9368))
        self.assertEqual(qdq["local_vs_s22_prediction_agreement"]["disagreement_count"], 120)
        self.assertEqual(
            report["provenance"]["remote_outputs"]["fp32"]["sha256"],
            "3F586D571E3E6D972CC806AD4E4A217150708E0A258AF8C755506BBDE63D0078",
        )
        self.assertEqual(
            report["provenance"]["remote_outputs"]["qdq_int8"]["sha256"],
            "3A5ADB12524FE7BBC2193C7688210993C78B44F35F9A0D647D917FDC8901A75D",
        )
        self.assertAlmostEqual(qdq["local_vs_s22_logits"]["mean_abs_error"], 0.2151494556091726)
        self.assertAlmostEqual(qdq["local_vs_s22_logits"]["max_abs_error"], 2.4266529083251953)
        self.assertAlmostEqual(qdq["local_vs_s22_logits"]["rmse"], 0.2944515844983891)
        self.assertAlmostEqual(
            qdq["local_vs_s22_logits"]["mean_cosine_similarity"], 0.998074252089587
        )
        self.assertAlmostEqual(
            qdq["local_vs_s22_logits"]["min_cosine_similarity"], 0.9553450473539299
        )
        cross = report["s22_fp32_vs_qdq_int8"]
        self.assertEqual(cross["prediction_agreement"]["disagreement_count"], 155)
        self.assertAlmostEqual(cross["qdq_int8_accuracy_change_vs_fp32_pp"], -0.05)
        self.assertEqual(
            cross["correctness_contingency"],
            {
                "both_correct": 9309,
                "fp32_only_correct": 64,
                "qdq_int8_only_correct": 59,
                "both_wrong": 568,
            },
        )

    def test_predictions_archive_has_only_int64_labels_indices_and_predictions(self):
        path = RESULT_DIR / "predictions.npz"
        with np.load(path, allow_pickle=False) as archive:
            self.assertEqual(archive.files, PREDICTION_KEYS)
            for key in archive.files:
                self.assertEqual(archive[key].shape, (10000,))
                self.assertEqual(archive[key].dtype, np.int64)
            self.assertTrue(np.array_equal(archive["original_indices"], np.arange(10000)))
            labels = archive["labels"]
            fp32_s22 = archive["fp32_s22_predictions"]
            qdq_s22 = archive["qdq_int8_s22_predictions"]
            self.assertEqual(int(np.count_nonzero(fp32_s22 == labels)), 9373)
            self.assertEqual(int(np.count_nonzero(qdq_s22 == labels)), 9368)
            self.assertEqual(int(np.count_nonzero(fp32_s22 == qdq_s22)), 9845)
        artifact = self.report["predictions_artifact"]
        self.assertFalse(artifact["contains_images"])
        self.assertFalse(artifact["contains_logits"])
        self.assertEqual(artifact["sha256"], sha256_file(path))

    def test_performance_and_report_boundaries_are_frozen(self):
        fp32 = self.report["models"]["fp32"]["performance"]
        qdq = self.report["models"]["qdq_int8"]["performance"]
        self.assertAlmostEqual(fp32["mean_latency_ms"], 0.82398)
        self.assertAlmostEqual(fp32["peak_memory_mib"], 156.765625)
        self.assertEqual((fp32["npu_nodes"], fp32["total_nodes"]), (68, 68))
        self.assertAlmostEqual(qdq["mean_latency_ms"], 0.40647)
        self.assertAlmostEqual(qdq["speedup_vs_fp32"], 2.027160676064654)
        self.assertAlmostEqual(qdq["peak_memory_mib"], 125.29296875)
        self.assertAlmostEqual(qdq["memory_reduction_vs_fp32_percent"], 20.076248380344865)
        self.assertEqual((qdq["npu_nodes"], qdq["total_nodes"]), (70, 70))
        markdown = (RESULT_DIR / "full_accuracy_summary.md").read_text(encoding="utf-8")
        for required in (
            "must not be interpreted as quantization improving generalization accuracy",
            "120 local/S22 QDQ prediction changes show backend numerical drift",
            "piecewise_v065 has only 82.80% local accuracy and 1.68766 ms",
            "not the current best deployment",
            "final recommended deployment is standard QDQ INT8",
            "contains no image or logit arrays",
        ):
            self.assertIn(required, markdown)


if __name__ == "__main__":
    unittest.main()
