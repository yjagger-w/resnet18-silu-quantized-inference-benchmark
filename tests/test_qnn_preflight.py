import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from silu_benchmark.qnn_local_accuracy import EXPECTED_INPUT, EXPECTED_OUTPUT
from silu_benchmark.qnn_preflight import (
    PREFLIGHT_SCHEMA,
    array_metadata,
    canonical_array_sha256,
    evaluate_preprocessed_session,
    prepare_preflight_subset,
    select_balanced_original_indices,
    write_deterministic_npz,
    write_preflight_outputs,
)


ROOT = Path(__file__).resolve().parents[1]


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

    def run(self, output_names, feeds):
        if output_names != ["logits"] or list(feeds) != ["images"]:
            raise AssertionError("unexpected ORT request")
        return [self.outputs.pop(0)]


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "qnn_preflight_runner", ROOT / "scripts/run_qnn_aihub.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QnnPreflightTests(unittest.TestCase):
    def test_balanced_selection_is_deterministic_and_preserves_original_order(self):
        labels = np.array([9, 0, 1, 0, 9, 1, 2, 2] + list(range(3, 9)) * 2)
        first = select_balanced_original_indices(labels, samples_per_class=2)
        second = select_balanced_original_indices(labels.copy(), samples_per_class=2)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(np.diff(first) > 0))
        self.assertEqual(len(np.unique(first)), len(first))
        np.testing.assert_array_equal(np.bincount(labels[first], minlength=10), np.full(10, 2))
        for label in range(10):
            expected = np.flatnonzero(labels == label)[:2]
            np.testing.assert_array_equal(first[labels[first] == label], expected)

    def test_subset_inputs_labels_and_indices_stay_aligned(self):
        labels = np.tile(np.arange(10, dtype=np.int64), 3)
        images = np.arange(len(labels) * 3 * 32 * 32, dtype=np.uint32)
        images = images.astype(np.uint8).reshape(len(labels), 3, 32, 32)
        inputs, selected_labels, indices = prepare_preflight_subset(
            images, labels, samples_per_class=2
        )
        self.assertEqual(inputs.shape, (20, 3, 32, 32))
        self.assertEqual(inputs.dtype, np.float32)
        np.testing.assert_array_equal(indices, np.arange(20))
        np.testing.assert_array_equal(selected_labels, labels[indices])
        np.testing.assert_array_equal(np.bincount(selected_labels), np.full(10, 2))
        self.assertTrue(np.all(np.isfinite(inputs)))

    def test_canonical_hash_is_repeatable_and_order_sensitive(self):
        values = np.arange(12, dtype=np.float32).reshape(3, 4)
        self.assertEqual(canonical_array_sha256(values), canonical_array_sha256(values.copy()))
        self.assertNotEqual(canonical_array_sha256(values), canonical_array_sha256(values[::-1]))
        metadata = array_metadata(values)
        self.assertEqual(metadata["shape"], [3, 4])
        self.assertEqual(metadata["dtype"], "float32")
        self.assertTrue(metadata["finite"])

    def test_local_reference_inference_collects_finite_logits(self):
        inputs = np.zeros((3, 3, 32, 32), dtype=np.float32)
        labels = np.array([1, 2, 3], dtype=np.int64)
        first = np.zeros((2, 10), dtype=np.float32)
        second = np.zeros((1, 10), dtype=np.float32)
        first[[0, 1], [1, 2]] = 1.0
        second[0, 3] = 1.0
        metrics, predictions, logits = evaluate_preprocessed_session(
            FakeSession([first, second]), inputs, labels, batch_size=2
        )
        self.assertEqual((metrics["correct"], metrics["total"]), (3, 3))
        np.testing.assert_array_equal(predictions, labels)
        self.assertEqual(logits.shape, (3, 10))
        self.assertTrue(np.all(np.isfinite(logits)))

    def test_npz_bytes_and_existing_inference_interface_are_stable(self):
        runner = load_runner()
        images = np.zeros((4, 3, 32, 32), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_deterministic_npz(root / "first.npz", {"images": images})
            second = write_deterministic_npz(root / "second.npz", {"images": images.copy()})
            self.assertEqual(first.read_bytes(), second.read_bytes())
            loaded = runner.load_inference_inputs(first)
            self.assertEqual(list(loaded), ["images"])
            self.assertEqual(len(loaded["images"]), 4)
            self.assertEqual(loaded["images"][0].shape, (1, 3, 32, 32))
            self.assertEqual(loaded["images"][0].dtype, np.float32)

    def test_report_writer_records_keys_shapes_dtypes_and_repeatability(self):
        inputs = np.zeros((2, 3, 32, 32), dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int64)
        indices = np.array([3, 7], dtype=np.int64)
        local_reference = {
            "fp32_predictions": labels.copy(),
            "fp32_logits": np.zeros((2, 10), dtype=np.float32),
            "qdq_int8_predictions": labels.copy(),
            "qdq_int8_logits": np.zeros((2, 10), dtype=np.float32),
        }
        model = {
            "label": "model",
            "correct": 2,
            "total": 2,
            "top1_accuracy_percent": 100.0,
            "compile_job_id": "jtest",
        }
        base = {
            "schema_version": PREFLIGHT_SCHEMA,
            "selection": {
                "rule": "test",
                "total_samples": 2,
                "samples_per_class": 1,
            },
            "models": {"fp32": dict(model), "qdq_int8": dict(model)},
            "prediction_comparison": {"agreement_count": 2, "agreement_percent": 100.0},
            "validations": {"all_inputs_are_finite": True, "all_logits_are_finite": True},
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest, paths = write_preflight_outputs(
                Path(directory),
                inputs=inputs,
                labels=labels,
                original_indices=indices,
                local_reference=local_reference,
                manifest_base=base,
            )
            self.assertEqual(manifest["artifacts"]["inputs"]["arrays"]["images"]["shape"], [2, 3, 32, 32])
            self.assertEqual(manifest["artifacts"]["labels"]["arrays"]["labels"]["dtype"], "int64")
            self.assertTrue(manifest["validations"]["npz_repeat_generation_file_hashes_match"])
            with np.load(paths["local_reference"], allow_pickle=False) as archive:
                self.assertEqual(
                    archive.files,
                    [
                        "fp32_predictions",
                        "fp32_logits",
                        "qdq_int8_predictions",
                        "qdq_int8_logits",
                    ],
                )
            payload = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], PREFLIGHT_SCHEMA)


if __name__ == "__main__":
    unittest.main()
