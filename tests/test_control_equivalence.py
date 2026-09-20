import copy
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from silu_benchmark.accuracy_recovery import discover_qdq_silu_sites
from silu_benchmark.control_equivalence import (
    build_strict_topology_control,
    compare_qdq_topology,
    graph_topology_hash,
    instrument_model_outputs,
    original_stage_mapping,
    protobuf_hash,
    selected_candidate_coverage,
    validate_audit_output_path,
    validate_control_label,
)


ROOT = Path(__file__).resolve().parents[1]
QDQ = ROOT / "artifacts/int8/resnet18_silu_int8_v065.onnx"
SEMANTIC = ROOT / "artifacts/accuracy_recovery/v1.1/controls/float_equivalent_silu.onnx"
SELECTED = ROOT / "artifacts/accuracy_recovery/v1.1/candidates/two_segment_p99_99_mse.onnx"


@unittest.skipUnless(all(path.exists() for path in (QDQ, SEMANTIC, SELECTED)),
                     "frozen v1.1 audit assets are unavailable")
class ControlEquivalenceGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = onnx.load(str(QDQ))
        cls.semantic = onnx.load(str(SEMANTIC))
        cls.selected = onnx.load(str(SELECTED))
        cls.sites = discover_qdq_silu_sites(cls.source)

    def test_strict_control_preserves_nodes_initializers_and_qdq_topology(self):
        strict, sites = build_strict_topology_control(self.source)
        self.assertEqual([site.site_id for site in sites], [site.site_id for site in self.sites])
        self.assertEqual(graph_topology_hash(strict), graph_topology_hash(self.source))
        self.assertEqual(
            [protobuf_hash(node) for node in strict.graph.node],
            [protobuf_hash(node) for node in self.source.graph.node],
        )
        self.assertEqual(
            [protobuf_hash(item) for item in strict.graph.initializer],
            [protobuf_hash(item) for item in self.source.graph.initializer],
        )
        audit = compare_qdq_topology(self.source, strict)
        self.assertTrue(audit["exact_qdq_topology"])
        self.assertFalse(audit["inserted"])
        self.assertFalse(audit["removed"])
        self.assertFalse(audit["moved_boundaries"])

    def test_inserted_removed_and_moved_qdq_are_detected(self):
        semantic = compare_qdq_topology(self.source, self.semantic)
        self.assertFalse(semantic["exact_qdq_topology"])
        self.assertEqual(len(semantic["removed"]), 68)
        self.assertFalse(semantic["inserted"])
        self.assertFalse(semantic["moved_boundaries"])
        moved = copy.deepcopy(self.source)
        first_qdq = next(node for node in moved.graph.node if node.op_type == "QuantizeLinear")
        first_qdq.input[0] = "deliberately_moved_boundary"
        changed = compare_qdq_topology(self.source, moved)
        self.assertEqual(len(changed["moved_boundaries"]), 1)

    def test_target_tensor_instrumentation_is_deterministic(self):
        strict, _ = build_strict_topology_control(self.source)
        site = self.sites[0]
        mapping = {
            "final_logits": "logits",
            f"{site.site_id}::pre_silu": site.input_tensor,
            f"{site.site_id}::post_silu": site.mul_output_tensor,
            f"{site.site_id}::activation_output": site.output_tensor,
        }
        first, first_names = instrument_model_outputs(strict, mapping)
        second, second_names = instrument_model_outputs(strict, mapping)
        self.assertEqual(first_names, second_names)
        self.assertEqual(protobuf_hash(first), protobuf_hash(second))
        rng = np.random.default_rng(20260828)
        values = rng.standard_normal((1, 3, 32, 32), dtype=np.float32)
        source_model, source_names = instrument_model_outputs(self.source, mapping)
        left_session = ort.InferenceSession(
            source_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        right_session = ort.InferenceSession(
            first.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        left = left_session.run(list(source_names), {"images": values})
        right = right_session.run(list(first_names), {"images": values})
        for expected, observed in zip(left, right):
            np.testing.assert_array_equal(expected, observed)

    def test_selected_candidate_has_all_17_unbypassed_piecewise_paths(self):
        coverage = selected_candidate_coverage(self.source, self.selected, self.sites)
        self.assertTrue(coverage["all_17_piecewise_paths_proven"])
        self.assertEqual(coverage["covered_site_count"], 17)
        self.assertTrue(all(row["no_original_silu_bypass"] for row in coverage["sites"]))
        self.assertTrue(all(row["inserted_piecewise_node_count"] == 24
                            for row in coverage["sites"]))

    def test_float_equivalent_label_is_rejected_when_qdq_differs(self):
        with self.assertRaisesRegex(ValueError, "topology differs"):
            validate_control_label("float-equivalent", self.source, self.semantic)
        validate_control_label("semantic-expression control", self.source, self.semantic)


class ControlEquivalencePureTests(unittest.TestCase):
    def test_generated_audit_paths_are_isolated(self):
        result = validate_audit_output_path(
            ROOT / "results/benchmarks/v1.1_control_equivalence_audit", ROOT
        )
        artifact = validate_audit_output_path(
            ROOT / "artifacts/accuracy_recovery/v1.1.1", ROOT, artifact=True
        )
        self.assertTrue(str(result).endswith("v1.1_control_equivalence_audit"))
        self.assertTrue(str(artifact).endswith("v1.1.1"))
        with self.assertRaises(ValueError):
            validate_audit_output_path(ROOT / "results/benchmarks/v1.1_accuracy_recovery", ROOT)
        with self.assertRaises(ValueError):
            validate_audit_output_path(ROOT / "artifacts/accuracy_recovery/v1.1", ROOT,
                                       artifact=True)

    def test_ort_only_audit_import_does_not_import_torch(self):
        program = r'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
import silu_benchmark.control_equivalence
assert "torch" not in sys.modules
assert "torchvision" not in sys.modules
'''
        result = subprocess.run(
            [sys.executable, "-c", program], cwd=ROOT, capture_output=True, text=True, timeout=60
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
