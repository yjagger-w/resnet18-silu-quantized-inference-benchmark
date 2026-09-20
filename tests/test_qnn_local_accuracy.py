import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from silu_benchmark.qnn_local_accuracy import (
    EXPECTED_INPUT,
    EXPECTED_OUTPUT,
    MODEL_ORDER,
    accuracy_markdown,
    accuracy_metrics,
    build_report,
    evaluate_session,
    label_sha256,
    ordered_dataset_fingerprint,
    prediction_agreement,
    validate_model_io,
    verify_sha256,
    write_accuracy_outputs,
)


class FakeNode:
    def __init__(self, metadata):
        self.name = metadata["name"]
        self.shape = metadata["shape"]
        self.type = metadata["type"]


class FakeSession:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def get_inputs(self):
        return [FakeNode(EXPECTED_INPUT)]

    def get_outputs(self):
        return [FakeNode(EXPECTED_OUTPUT)]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, output_names, feeds):
        self.assert_contract(output_names, feeds)
        return [self.outputs.pop(0)]

    @staticmethod
    def assert_contract(output_names, feeds):
        if output_names != ["logits"] or list(feeds) != ["images"]:
            raise AssertionError("unexpected ORT request")


def model_result(predictions, labels):
    metrics = accuracy_metrics(labels, predictions)
    return {
        "label": "test model",
        "path": "artifacts/model.onnx",
        "sha256": "A" * 64,
        "io": {
            "inputs": [dict(EXPECTED_INPUT)],
            "outputs": [dict(EXPECTED_OUTPUT)],
            "providers": ["CPUExecutionProvider"],
        },
        **metrics,
        "inference_failure_batch_count": 0,
        "inference_failure_sample_count": 0,
        "inference_failure_sample_indices": [],
        "inference_failures": [],
        "nonfinite_logit_count": 0,
        "nonfinite_sample_count": 0,
        "nonfinite_sample_indices": [],
    }


class QnnLocalAccuracyTests(unittest.TestCase):
    def test_accuracy_calculation_records_all_error_indices(self):
        metrics = accuracy_metrics(
            np.array([0, 1, 2, 3]), np.array([0, 7, 2, 9])
        )
        self.assertEqual(metrics["correct"], 2)
        self.assertEqual(metrics["total"], 4)
        self.assertEqual(metrics["top1_accuracy"], 0.5)
        self.assertEqual(metrics["misclassified_indices"], [1, 3])

    def test_prediction_agreement_records_disagreement_indices(self):
        metrics = prediction_agreement(
            np.array([0, 1, 2, 3]), np.array([0, 1, 8, 9])
        )
        self.assertEqual(metrics["agreement_count"], 2)
        self.assertEqual(metrics["disagreement_count"], 2)
        self.assertEqual(metrics["agreement"], 0.5)
        self.assertEqual(metrics["disagreement_indices"], [2, 3])

    def test_label_and_model_hash_verification(self):
        labels = np.array([9, 1, 4], dtype=np.int64)
        expected_labels = hashlib.sha256(labels.astype("<i8").tobytes()).hexdigest().upper()
        self.assertEqual(label_sha256(labels), expected_labels)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.onnx"
            model.write_bytes(b"locked model bytes")
            expected = hashlib.sha256(model.read_bytes()).hexdigest().upper()
            self.assertEqual(verify_sha256(model, expected.lower(), label="model"), expected)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_sha256(model, "0" * 64, label="model")

    def test_dataset_fingerprint_is_stable_and_order_sensitive(self):
        images = np.arange(4 * 3 * 32 * 32, dtype=np.uint16).astype(np.uint8)
        images = images.reshape(4, 3, 32, 32)
        labels = np.array([0, 1, 2, 3], dtype=np.int64)
        first = ordered_dataset_fingerprint(images, labels)
        self.assertEqual(first, ordered_dataset_fingerprint(images.copy(), labels.copy()))
        self.assertNotEqual(first, ordered_dataset_fingerprint(images[::-1], labels[::-1]))
        self.assertNotEqual(first, ordered_dataset_fingerprint(images, labels[::-1]))

    def test_model_contract_and_nonfinite_output_detection(self):
        logits = np.zeros((2, 10), dtype=np.float32)
        logits[0, 3] = 1.0
        logits[1, 4] = np.nan
        session = FakeSession([logits])
        self.assertEqual(validate_model_io(session)["providers"], ["CPUExecutionProvider"])
        images = np.zeros((2, 3, 32, 32), dtype=np.uint8)
        labels = np.array([3, 4], dtype=np.int64)
        metrics, predictions = evaluate_session(session, images, labels, batch_size=2)
        np.testing.assert_array_equal(predictions, np.array([3, -1]))
        self.assertEqual(metrics["correct"], 1)
        self.assertEqual(metrics["nonfinite_logit_count"], 1)
        self.assertEqual(metrics["nonfinite_sample_count"], 1)
        self.assertEqual(metrics["nonfinite_sample_indices"], [1])
        self.assertEqual(metrics["inference_failure_sample_count"], 0)

    def test_json_markdown_and_prediction_archive_generation(self):
        images = np.zeros((3, 3, 32, 32), dtype=np.uint8)
        labels = np.array([0, 1, 2], dtype=np.int64)
        predictions = {
            "fp32": np.array([0, 1, 2]),
            "qdq_int8": np.array([0, 9, 2]),
            "piecewise_reference": np.array([8, 1, 2]),
        }
        results = {
            model_id: model_result(predictions[model_id], labels) for model_id in MODEL_ORDER
        }
        report = build_report(
            manifest_path="configs/qnn/manifest.json",
            manifest_sha256="B" * 64,
            data_root="data",
            test_batch_path="data/cifar-10-batches-py/test_batch",
            test_batch_sha256="C" * 64,
            images=images,
            labels=labels,
            batch_size=2,
            ort_version="test",
            available_providers=["CPUExecutionProvider"],
            model_results=results,
            predictions=predictions,
        )
        self.assertEqual(report["models"]["qdq_int8"]["accuracy_change_vs_fp32_pp"], -100 / 3)
        self.assertEqual(
            report["pairwise_prediction_comparisons"]["fp32_vs_qdq_int8"][
                "disagreement_indices"
            ],
            [1],
        )
        markdown = accuracy_markdown(report)
        self.assertIn("not Galaxy S22 QNN accuracy", markdown)
        self.assertIn("No Qualcomm AI Hub task was created", markdown)
        self.assertIn("Piecewise reference (1 errors)", markdown)

        with tempfile.TemporaryDirectory() as directory:
            paths = write_accuracy_outputs(Path(directory), report, predictions, labels)
            self.assertTrue(all(path.exists() for path in paths))
            payload = json.loads(paths[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "qnn-local-cifar10-accuracy/v1.6")
            with np.load(paths[2], allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["labels"], labels)
                np.testing.assert_array_equal(
                    archive["qdq_int8_misclassified_indices"], np.array([1])
                )
                self.assertFalse(any("image" in name for name in archive.files))


if __name__ == "__main__":
    unittest.main()
