import copy
import json
import tempfile
import unittest
from pathlib import Path

import onnx

from silu_benchmark.ort_custom_op_rewrite import (
    CUSTOM_DOMAIN,
    CUSTOM_OP_TYPE,
    EXPECTED_SITE_COUNT,
    build_sidecar_manifest,
    discover_selected_piecewise_islands,
    rewrite_selected_candidate,
    sha256_file,
    validate_custom_node_contract,
    validate_generated_artifact,
    validate_generated_output_path,
    verify_selected_source,
)


ROOT = Path(__file__).resolve().parents[1]
SELECTED = ROOT / "artifacts/accuracy_recovery/v1.1/candidates/two_segment_p99_99_mse.onnx"
RECEIPT = ROOT / "results/benchmarks/v1.1_accuracy_recovery_smoke/selection_receipt.json"
EXPECTED_SHA = "0df5e7f6388d39a658a53790788d20393ff3e8e481d075b85ca84a54e5821a8d"


@unittest.skipUnless(SELECTED.exists() and RECEIPT.exists(), "frozen v1.1 assets unavailable")
class OrtCustomOpRewriteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.receipt = verify_selected_source(
            SELECTED, RECEIPT, expected_model_sha256=EXPECTED_SHA
        )
        cls.islands = discover_selected_piecewise_islands(cls.model)

    def test_selected_source_and_receipt_hash_verification(self):
        self.assertEqual(sha256_file(SELECTED), EXPECTED_SHA)
        self.assertEqual(self.receipt["model_sha256"], EXPECTED_SHA)
        with self.assertRaisesRegex(ValueError, "configuration"):
            verify_selected_source(SELECTED, RECEIPT, expected_model_sha256="0" * 64)

    def test_exact_17_island_discovery_and_explicit_boundary(self):
        self.assertEqual(len(self.islands), EXPECTED_SITE_COUNT)
        self.assertEqual(self.islands[0].site_id, "act.call_0")
        self.assertEqual(self.islands[-1].site_id, "layer4.1.act.call_1")
        for island in self.islands:
            self.assertEqual(len(island.source_node_names), 24)
            self.assertTrue(island.input_code_tensor.endswith("QuantizeLinear_Output"))
            self.assertTrue(island.input_dequantized_tensor.endswith("DequantizeLinear_Output"))
            self.assertEqual(island.bits, 8)
            self.assertGreater(island.input_scale, 0)
            self.assertGreaterEqual(island.input_zero_point, 0)
            self.assertLessEqual(island.input_zero_point, 255)

    def test_parameter_serialization_and_custom_contract_stability(self):
        first, islands, report = rewrite_selected_candidate(
            self.model,
            source_model_sha256=EXPECTED_SHA,
            selection_receipt_hash=self.receipt["selection_receipt_hash"],
        )
        second, _, second_report = rewrite_selected_candidate(
            self.model,
            source_model_sha256=EXPECTED_SHA,
            selection_receipt_hash=self.receipt["selection_receipt_hash"],
        )
        self.assertEqual(
            first.SerializeToString(deterministic=True),
            second.SerializeToString(deterministic=True),
        )
        self.assertEqual(report, second_report)
        self.assertEqual(report["source_piecewise_islands_removed"], 17)
        self.assertEqual(report["source_piecewise_nodes_removed"], 408)
        self.assertEqual(report["custom_nodes_added"], 17)
        self.assertEqual(report["original_selected_piecewise_nodes_retained"], 0)
        nodes = [node for node in first.graph.node if node.domain == CUSTOM_DOMAIN]
        self.assertEqual(len(nodes), 17)
        self.assertTrue(all(node.op_type == CUSTOM_OP_TYPE for node in nodes))
        self.assertEqual([node.name for node in nodes], [item.custom_node_name for item in islands])

    def test_manifest_validation_and_stale_artifact_rejection(self):
        rewritten, islands, report = rewrite_selected_candidate(
            self.model,
            source_model_sha256=EXPECTED_SHA,
            selection_receipt_hash=self.receipt["selection_receipt_hash"],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "custom.onnx"
            sidecar_path = root / "custom.manifest.json"
            model_path.write_bytes(rewritten.SerializeToString(deterministic=True))
            manifest = build_sidecar_manifest(
                source_path=SELECTED,
                rewritten_path=model_path,
                selection_receipt_path=RECEIPT,
                islands=islands,
                graph_report=report,
                generated_at_utc="2026-08-28T00:00:00+00:00",
            )
            sidecar_path.write_text(json.dumps(manifest), encoding="utf-8")
            validated = validate_generated_artifact(
                source_path=SELECTED,
                rewritten_path=model_path,
                sidecar_path=sidecar_path,
                selection_receipt_path=RECEIPT,
            )
            self.assertEqual(validated["manifest_hash"], manifest["manifest_hash"])
            validate_custom_node_contract(onnx.load(str(model_path)), manifest)
            model_path.write_bytes(model_path.read_bytes() + b"stale")
            with self.assertRaisesRegex(ValueError, "rewritten_model_sha256"):
                validate_generated_artifact(
                    source_path=SELECTED,
                    rewritten_path=model_path,
                    sidecar_path=sidecar_path,
                    selection_receipt_path=RECEIPT,
                )

    def test_custom_contract_tampering_is_rejected(self):
        rewritten, islands, report = rewrite_selected_candidate(
            self.model,
            source_model_sha256=EXPECTED_SHA,
            selection_receipt_hash=self.receipt["selection_receipt_hash"],
        )
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "custom.onnx"
            model_path.write_bytes(rewritten.SerializeToString(deterministic=True))
            manifest = build_sidecar_manifest(
                source_path=SELECTED,
                rewritten_path=model_path,
                selection_receipt_path=RECEIPT,
                islands=islands,
                graph_report=report,
                generated_at_utc="2026-08-28T00:00:00+00:00",
            )
            changed = copy.deepcopy(manifest)
            changed["sites"][0]["attributes"]["input_zero_point"] += 1
            with self.assertRaisesRegex(ValueError, "attributes differ"):
                validate_custom_node_contract(rewritten, changed)

    def test_generated_output_paths_are_isolated(self):
        artifact = validate_generated_output_path(
            ROOT / "artifacts/accuracy_recovery/v1.2/model.onnx", ROOT
        )
        report = validate_generated_output_path(
            ROOT / "results/benchmarks/v1.2_ort_cpp_customop/report.json", ROOT
        )
        self.assertTrue(str(artifact).endswith("model.onnx"))
        self.assertTrue(str(report).endswith("report.json"))
        with self.assertRaises(ValueError):
            validate_generated_output_path(SELECTED, ROOT)


if __name__ == "__main__":
    unittest.main()
