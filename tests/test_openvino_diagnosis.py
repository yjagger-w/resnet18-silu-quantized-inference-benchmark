"""Pure diagnostic contracts; no PyTorch or eager OpenVINO imports."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from silu_benchmark.backends.openvino_backend import compare_outputs
from silu_benchmark.backends.onnx_model_rewrite import discover_silu_patterns
from silu_benchmark.openvino_diagnosis import (
    array_digest, diagnostic_path, first_divergence, instrument, metrics,
    quantizer_control_model, rounding_boundary_evidence, select_probes, semantic_sites, validate_output_names, validate_probes,
)

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    nodes = [helper.make_node("Conv", ["images", "weight"], ["stem"], name="/conv1/Conv")]
    value = "stem"
    paths = ["/act"] + [f"/layer{i}/layer{i}.{j}/{act}" for i in range(1, 5) for j in range(2) for act in ("act", "act_1")]
    for path in paths:
        if path.endswith("act_1"):
            nodes.append(helper.make_node("Add", [value, value], [path + "/add"], name=path + "/Add"))
            value = path + "/add"
        sigmoid, output = path + "/sigmoid", path + "/output"
        nodes += [helper.make_node("Sigmoid", [value], [sigmoid], name=path + "/Sigmoid"),
                  helper.make_node("Mul", [value, sigmoid], [output], name=path + "/Mul")]
        value = output
    nodes += [helper.make_node("GlobalAveragePool", [value], ["pool"], name="pool"),
              helper.make_node("Flatten", ["pool"], ["logits"], name="flatten")]
    graph = helper.make_graph(nodes, "fixture", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [None, 1, 2, 2])],
                              [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, 1])],
                              [helper.make_tensor("weight", TensorProto.FLOAT, [1, 1, 1, 1], [1])])
    baseline = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model = copy.deepcopy(baseline)
    del model.graph.node[:]
    model.graph.initializer.extend([helper.make_tensor("scale", TensorProto.FLOAT, [], [.1]),
                                    helper.make_tensor("zero", TensorProto.UINT8, [], [1])])
    replacements = {}
    for node in baseline.graph.node:
        changed = copy.deepcopy(node)
        for i, name in enumerate(changed.input):
            changed.input[i] = replacements.get(name, name)
        model.graph.node.append(changed)
        if node.op_type in ("Conv", "Sigmoid", "Mul", "Add"):
            name = node.output[0]
            model.graph.node.extend([
                helper.make_node("QuantizeLinear", [name, "scale", "zero"], [name + ".q"], name=name + ".Q"),
                helper.make_node("DequantizeLinear", [name + ".q", "scale", "zero"], [name + ".dq"], name=name + ".DQ"),
            ])
            replacements[name] = name + ".dq"
    manifest = {"sites": [{"site_id": s.site_id} for s in discover_silu_patterns(baseline)]}
    return model, baseline, manifest


class OpenVINODiagnosisTests(unittest.TestCase):
    def test_deterministic_semantic_selection_and_order(self):
        model, baseline, manifest = fixture()
        sites = semantic_sites(model, baseline, manifest)
        self.assertEqual(len(sites), 17)
        probes = select_probes(model, sites)
        self.assertEqual(probes, select_probes(model, sites))
        self.assertEqual(probes[0]["source_value"], "images")
        self.assertEqual(probes[-1]["source_value"], "logits")
        self.assertEqual([p["topological_index"] for p in probes], sorted(p["topological_index"] for p in probes))
        self.assertEqual(sum(any(role.startswith("silu:") for role in p["roles"]) for p in probes), 17)
        self.assertEqual(sum("residual_add" in p["roles"] for p in probes), 8)
        narrowed = select_probes(model, sites, 0, 4)
        self.assertEqual([p["topological_index"] for p in narrowed][1:-1], list(range(5)))

    def test_semantic_provenance_rejects_missing_reordered_and_wrong_connections(self):
        model, baseline, manifest = fixture()
        manifest["sites"].reverse()
        with self.assertRaisesRegex(ValueError, "ordered"):
            semantic_sites(model, baseline, manifest)
        model, baseline, manifest = fixture()
        mul = next(n for n in model.graph.node if n.op_type == "Mul")
        mul.input[1] = "images"
        with self.assertRaisesRegex(ValueError, "connections"):
            semantic_sites(model, baseline, manifest)

    def test_instrumentation_is_append_only_and_does_not_mutate_source(self):
        model, baseline, manifest = fixture()
        before = model.SerializeToString()
        probes = select_probes(model, semantic_sites(model, baseline, manifest))
        debug, mapping = instrument(model, probes)
        self.assertEqual(model.SerializeToString(), before)
        self.assertEqual([n.SerializeToString() for n in debug.graph.node], [n.SerializeToString() for n in model.graph.node])
        self.assertEqual([w.SerializeToString() for w in debug.graph.initializer], [w.SerializeToString() for w in model.graph.initializer])
        self.assertEqual(debug.graph.output[0], model.graph.output[0])
        for probe in mapping:
            self.assertEqual(debug.graph.output[probe["instrumented_output_index"]].name, probe["source_value"])
        onnx.checker.check_model(debug)

    def test_source_and_runtime_mapping_rejection(self):
        model, baseline, manifest = fixture()
        probes = select_probes(model, semantic_sites(model, baseline, manifest))
        with self.assertRaisesRegex(ValueError, "topological"):
            validate_probes(model, list(reversed(probes)))
        wrong = copy.deepcopy(probes)
        wrong[0]["source_value"] = "missing"
        with self.assertRaisesRegex(ValueError, "missing"):
            instrument(model, wrong)
        wrong = copy.deepcopy(probes)
        wrong[1]["producer_name"] = "wrong"
        with self.assertRaisesRegex(ValueError, "unverified"):
            instrument(model, wrong)
        validate_output_names(["x", "y"], [["x"], ["y"]])
        for actual in ([["x"]], [["y"], ["x"]], [["x", "y"], ["y"]]):
            with self.assertRaises(ValueError):
                validate_output_names(["x", "y"], actual)

    def test_metrics_and_first_divergence_have_no_acceptance_threshold(self):
        exact = metrics(np.zeros(2, np.float32), np.zeros(2, np.float32))
        tiny = metrics(np.zeros(2, np.float32), np.array([0, 1e-12], np.float32))
        integer = metrics(np.array([1, 2], np.uint8), np.array([1, 3], np.uint8))
        rows = [{"source_value": str(i), "topological_index": i, "input_dependent": True, "metrics": m}
                for i, m in enumerate((exact, tiny, integer))]
        self.assertEqual(first_divergence(rows)["first_divergent"]["source_value"], "1")
        self.assertEqual(first_divergence(rows)["last_preceding_exact_match"]["source_value"], "0")
        self.assertEqual(first_divergence(rows, integer_only=True)["first_divergent"]["source_value"], "2")
        self.assertIsNone(first_divergence(rows[:1])["first_divergent"])
        self.assertEqual(integer["max_absolute_error"], 1)
        self.assertEqual(integer["mse"], .5)
        self.assertEqual(integer["different_elements"], 1)
        json.dumps(integer)
        with self.assertRaisesRegex(ValueError, "shape"):
            metrics(np.zeros(2), np.zeros(3))
        with self.assertRaisesRegex(ValueError, "finite"):
            metrics(np.zeros(2), np.array([0, np.nan]))

    def test_prediction_disagreements_are_preserved(self):
        reference = np.zeros((128, 10), np.float32)
        candidate = reference.copy()
        candidate[[22, 52, 125], 1] = 1
        result = compare_outputs(reference, candidate, np.zeros(128, np.int64))
        self.assertEqual(result["prediction_disagreement_indices"], [22, 52, 125])
        self.assertEqual(result["prediction_agreement"], 125 / 128)
        self.assertEqual(result["openvino_top1_accuracy"], 125 / 128)

    def test_quantizer_control_copies_exact_source_node_and_parameters(self):
        model, _, _ = fixture()
        before = model.SerializeToString()
        index = next(i for i, node in enumerate(model.graph.node) if node.op_type == "QuantizeLinear")
        control = quantizer_control_model(model, index, [128, 1, 2, 2])
        self.assertEqual(model.SerializeToString(), before)
        self.assertEqual(control.graph.node[0].SerializeToString(), model.graph.node[index].SerializeToString())
        initializers = {v.name: v.SerializeToString() for v in model.graph.initializer}
        for value in control.graph.initializer:
            self.assertEqual(value.SerializeToString(), initializers[value.name])
        with self.assertRaisesRegex(ValueError, "QuantizeLinear"):
            quantizer_control_model(model, 0, [128, 1, 2, 2])

    def test_rounding_evidence_keeps_same_input_code_differences(self):
        value = np.array([-.5, .499998, 0], np.float32)
        a, b = np.array([111, 111, 111], np.uint8), np.array([110, 112, 111], np.uint8)
        evidence = rounding_boundary_evidence(value, value.copy(), a, b, np.float32(1), np.uint8(111))
        self.assertEqual(evidence["code_disagreements"], 2)
        self.assertEqual(evidence["same_input_but_different_code_count"], 2)
        self.assertEqual(evidence["examples"][0]["ort_distance_to_half_integer"], 0)
        self.assertFalse(evidence["examples_truncated"])
        json.dumps(evidence)

    def test_output_path_and_array_digest_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = diagnostic_path(root, "results/benchmarks/v0.8.1_diagnosis_test")
            self.assertTrue(str(valid).startswith(str(root)))
            for path in ("results/benchmarks/v0.8_openvino_cpu_smoke", "results/benchmarks/v0.6.5", "artifacts/int8", "results/benchmarks/v0.8.1_diagnosis/../../outside"):
                with self.assertRaises(ValueError):
                    diagnostic_path(root, path)
        value = np.arange(4, dtype=np.float32)
        self.assertEqual(array_digest(value), array_digest(value.copy()))
        self.assertNotEqual(array_digest(value), array_digest(value.reshape(2, 2)))
        self.assertNotEqual(array_digest(value), array_digest(value.astype(np.int32)))

    def test_clean_import_has_no_torch_or_eager_openvino(self):
        program = r'''
import importlib.abc, pathlib, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torchvision', 'openvino'):
            raise AssertionError('unexpected import: '+fullname)
sys.meta_path.insert(0,Guard())
sys.path.insert(0,str(pathlib.Path.cwd()/'scripts'))
import diagnose_openvino_standard_qdq as d
d.no_torch()
assert d.parse_args(['--smoke']).smoke
'''
        result = subprocess.run([sys.executable, "-c", program], cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
