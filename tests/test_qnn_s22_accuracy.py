import tempfile
import unittest
from pathlib import Path

import numpy as np

from silu_benchmark.qnn_s22_accuracy import (
    S22_REPORT_SCHEMA,
    build_s22_preflight_report,
    compare_local_and_remote,
    load_npz_exact,
    write_s22_accuracy_outputs,
)


def logits_for(predictions):
    result = np.zeros((len(predictions), 10), dtype=np.float32)
    result[np.arange(len(predictions)), predictions] = 1.0
    return result


class QnnS22AccuracyTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.tile(np.arange(10, dtype=np.int64), 100)
        self.original_indices = np.arange(1000, dtype=np.int64) + 50

    def test_local_remote_accuracy_agreement_metrics_and_original_indices(self):
        local_predictions = self.labels.copy()
        remote_predictions = self.labels.copy()
        remote_predictions[[3, 17]] = (remote_predictions[[3, 17]] + 1) % 10
        result, actual_local, actual_remote = compare_local_and_remote(
            self.labels,
            self.original_indices,
            logits_for(local_predictions),
            logits_for(remote_predictions),
        )
        self.assertEqual(result["local"]["correct"], 1000)
        self.assertEqual(result["s22"]["correct"], 998)
        self.assertEqual(result["local_vs_s22_prediction_agreement"]["agreement_count"], 998)
        self.assertEqual(
            result["local_vs_s22_prediction_agreement"]["disagreement_original_indices"],
            [53, 67],
        )
        self.assertGreater(result["local_vs_s22_logits"]["rmse"], 0.0)
        np.testing.assert_array_equal(actual_local, local_predictions)
        np.testing.assert_array_equal(actual_remote, remote_predictions)

    def test_report_contains_jobs_caveat_and_prediction_only_archive(self):
        fp32 = self.labels.copy()
        qdq_local = self.labels.copy()
        qdq_local[[1, 2]] = (qdq_local[[1, 2]] + 1) % 10
        report, predictions = build_s22_preflight_report(
            labels=self.labels,
            original_indices=self.original_indices,
            local_logits={"fp32": logits_for(fp32), "qdq_int8": logits_for(qdq_local)},
            remote_logits={"fp32": logits_for(fp32), "qdq_int8": logits_for(self.labels)},
            provenance={"inputs": {"sha256": "A" * 64}},
            jobs={
                "fp32": {"compile": "jp1ndxwlg", "inference": "jp0mdve2g"},
                "qdq_int8": {"compile": "j5qlw227p", "inference": "jgolo4e4g"},
            },
            selection={"rule": "first 100 per class", "total_samples": 1000},
        )
        self.assertEqual(report["schema_version"], S22_REPORT_SCHEMA)
        self.assertAlmostEqual(
            report["models"]["qdq_int8"]["s22"]["accuracy_change_vs_local_pp"], 0.2
        )
        self.assertEqual(
            report["interpretation"]["qdq_s22_additional_correct_samples_vs_local"], 2
        )
        self.assertIn("does not demonstrate", report["interpretation"]["required_caveat"])
        self.assertEqual(
            list(predictions),
            [
                "labels",
                "original_indices",
                "fp32_local_predictions",
                "fp32_s22_predictions",
                "qdq_int8_local_predictions",
                "qdq_int8_s22_predictions",
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            payload, paths = write_s22_accuracy_outputs(Path(directory), report, predictions)
            with np.load(paths["predictions"], allow_pickle=False) as archive:
                self.assertEqual(archive.files, list(predictions))
                self.assertFalse(any("logit" in key or "image" in key for key in archive.files))
            self.assertFalse(payload["predictions_artifact"]["contains_images"])
            self.assertFalse(payload["predictions_artifact"]["contains_logits"])
            self.assertIn("+0.2 percentage points", paths["markdown"].read_text(encoding="utf-8"))

    def test_npz_contract_requires_exact_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outputs.npz"
            np.savez(path, output_0=np.zeros((1000, 10), dtype=np.float32))
            loaded = load_npz_exact(path, ("output_0",))
            self.assertEqual(loaded["output_0"].shape, (1000, 10))
            with self.assertRaisesRegex(ValueError, "keys must be"):
                load_npz_exact(path, ("wrong",))


if __name__ == "__main__":
    unittest.main()
