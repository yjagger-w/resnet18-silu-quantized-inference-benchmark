import copy
import json
import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from silu_benchmark.accuracy_recovery import discover_qdq_silu_sites
from silu_benchmark.hardware_aware_qdq_model import (
    METADATA_KEY,
    extract_standard_qdq_site_specs,
    rewrite_standard_qdq_parameters,
    validate_standard_qdq_rewrite,
)
from silu_benchmark.quantization import StandardQDQSpec


def _initializer(name, value, dtype):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def build_synthetic_qdq_model():
    site_names = ["act"]
    for layer in range(1, 5):
        for block in range(2):
            site_names.extend(
                [f"layer{layer}.{block}/act", f"layer{layer}.{block}/act_1"]
            )
    nodes = []
    initializers = []
    current = "x"
    for index, site_name in enumerate(site_names):
        prefix = site_name.replace(".", "_").replace("/", "_")
        sigmoid = f"{prefix}_sigmoid"
        sigmoid_q = f"{prefix}_sigmoid_q"
        sigmoid_dq = f"{prefix}_sigmoid_dq"
        multiplied = f"{prefix}_mul"
        output_q = f"{prefix}_output_q"
        output = f"site_{index}_output"
        sigmoid_scale = f"{prefix}_sigmoid_scale"
        sigmoid_zero = f"{prefix}_sigmoid_zero"
        output_scale = f"{prefix}_output_scale"
        output_zero = f"{prefix}_output_zero"
        initializers.extend(
            [
                _initializer(sigmoid_scale, 1.0 / 255.0, np.float32),
                _initializer(sigmoid_zero, 0, np.uint8),
                _initializer(output_scale, 0.02, np.float32),
                _initializer(output_zero, 14, np.uint8),
            ]
        )
        nodes.extend(
            [
                helper.make_node(
                    "Sigmoid", [current], [sigmoid], name=f"/{site_name}/Sigmoid"
                ),
                helper.make_node(
                    "QuantizeLinear",
                    [sigmoid, sigmoid_scale, sigmoid_zero],
                    [sigmoid_q],
                    name=f"/{site_name}/Sigmoid_QuantizeLinear",
                ),
                helper.make_node(
                    "DequantizeLinear",
                    [sigmoid_q, sigmoid_scale, sigmoid_zero],
                    [sigmoid_dq],
                    name=f"/{site_name}/Sigmoid_DequantizeLinear",
                ),
                helper.make_node(
                    "Mul", [current, sigmoid_dq], [multiplied], name=f"/{site_name}/Mul"
                ),
                helper.make_node(
                    "QuantizeLinear",
                    [multiplied, output_scale, output_zero],
                    [output_q],
                    name=f"/{site_name}/Mul_QuantizeLinear",
                ),
                helper.make_node(
                    "DequantizeLinear",
                    [output_q, output_scale, output_zero],
                    [output],
                    name=f"/{site_name}/Mul_DequantizeLinear",
                ),
            ]
        )
        current = output
    nodes.append(helper.make_node("Identity", [current], ["y"], name="output_identity"))
    graph = helper.make_graph(
        nodes,
        "synthetic_17_site_qdq",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [8])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=8,
    )
    onnx.checker.check_model(model)
    return model


class HardwareAwareStandardQDQModelTests(unittest.TestCase):
    def setUp(self):
        self.model = build_synthetic_qdq_model()
        self.sites = discover_qdq_silu_sites(self.model)
        self.specs = {
            site.site_id: StandardQDQSpec(0.01 + index * 0.0001, 28)
            for index, site in enumerate(self.sites)
        }

    def test_synthetic_model_has_exact_production_site_identity_contract(self):
        self.assertEqual(len(self.sites), 17)
        self.assertEqual(self.sites[0].site_id, "act.call_0")
        self.assertEqual(self.sites[-1].site_id, "layer4.1.act.call_1")

    def test_source_qdq_specs_are_extracted_from_existing_model(self):
        specs = extract_standard_qdq_site_specs(self.model)
        self.assertEqual(set(specs), {site.site_id for site in self.sites})
        self.assertEqual(len(specs), 17)
        for spec in specs.values():
            self.assertAlmostEqual(spec.scale, 0.02, places=7)
            self.assertEqual(spec.zero_point, 14)
            self.assertEqual(spec.bits, 8)

    def test_rewrite_preserves_nodes_and_emits_only_standard_parameters(self):
        original = self.model.SerializeToString()
        result = rewrite_standard_qdq_parameters(
            self.model, self.specs, calibration_digest="A" * 64
        )
        self.assertEqual(self.model.SerializeToString(), original)
        contract = validate_standard_qdq_rewrite(self.model, result.model)
        self.assertEqual(contract["target_qdq_pair_count"], 17)
        self.assertEqual(contract["added_node_count"], 0)
        self.assertEqual(contract["custom_runtime_nodes_added"], 0)
        self.assertFalse(contract["piecewise_runtime_dispatch"])
        self.assertEqual(
            [node.op_type for node in self.model.graph.node],
            [node.op_type for node in result.model.graph.node],
        )
        self.assertEqual(
            [node.domain for node in self.model.graph.node],
            [node.domain for node in result.model.graph.node],
        )

        metadata = {item.key: item.value for item in result.model.metadata_props}
        payload = json.loads(metadata[METADATA_KEY])
        self.assertFalse(payload["runtime_contract"]["piecewise_runtime_dispatch"])
        self.assertNotIn("vsplit", json.dumps(payload["selected_qdq"]))

    def test_rewritten_model_runs_with_onnxruntime(self):
        result = rewrite_standard_qdq_parameters(self.model, self.specs)
        session = ort.InferenceSession(
            result.model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        values = np.linspace(-2.0, 4.0, 8, dtype=np.float32)
        output = session.run(None, {"x": values})[0]
        self.assertEqual(output.shape, values.shape)
        self.assertEqual(output.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(output)))

    def test_missing_extra_and_non_uint8_specs_are_rejected(self):
        missing = dict(self.specs)
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(ValueError, "missing"):
            rewrite_standard_qdq_parameters(self.model, missing)
        extra = {**self.specs, "unknown.call_0": StandardQDQSpec(0.01, 1)}
        with self.assertRaisesRegex(ValueError, "unused"):
            rewrite_standard_qdq_parameters(self.model, extra)
        invalid = dict(self.specs)
        invalid[next(iter(invalid))] = StandardQDQSpec(0.01, 1, bits=7)
        with self.assertRaisesRegex(ValueError, "invalid"):
            rewrite_standard_qdq_parameters(self.model, invalid)

    def test_topology_mutation_is_rejected(self):
        rewritten = copy.deepcopy(self.model)
        rewritten.graph.node[0].op_type = "Tanh"
        with self.assertRaisesRegex(ValueError, "topology"):
            validate_standard_qdq_rewrite(self.model, rewritten)


if __name__ == "__main__":
    unittest.main()
