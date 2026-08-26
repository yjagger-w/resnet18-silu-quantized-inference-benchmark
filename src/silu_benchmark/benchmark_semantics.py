"""Proven, ordered semantic output mappings for debug-only ONNX copies."""

import copy
import json

import onnx

from .backends import discover_silu_patterns
from .config import SILU_ACTIVATION_MODULES


def semantic_output_mappings(baseline, custom, manifest):
    sites = discover_silu_patterns(baseline)
    if len(sites) != 17 or {s.module_path for s in sites} != set(SILU_ACTIVATION_MODULES):
        raise ValueError("expected exactly 17 runtime sites from the nine ResNet18 SiLU modules")
    identities = [site.site_id for site in sites]
    if identities != [entry["site_id"] for entry in manifest["sites"]]:
        raise ValueError("ordered ONNX sites differ from the committed manifest")
    metadata = [item.value for item in custom.metadata_props
                if item.key == "silu_benchmark.piecewise_rewrite_sites"]
    if len(metadata) != 1:
        raise ValueError("custom graph must have exactly one Phase 3 rewrite metadata record")
    entries = json.loads(metadata[0])
    if [entry["site_id"] for entry in entries] != identities:
        raise ValueError("ordered rewrite metadata sites differ from baseline")
    if len(baseline.graph.output) != 1 or len(custom.graph.output) != 1:
        raise ValueError("expected a single final logits output per graph")
    fp32_mapping = {}
    custom_mapping = {}
    for site, entry in zip(sites, entries):
        expected = {
            "module_path": site.module_path, "call_index": site.call_index,
            "input_tensor": site.input_tensor, "output_tensor": site.output_tensor,
        }
        if any(entry.get(key) != value for key, value in expected.items()):
            raise ValueError(f"unproven custom mapping for {site.site_id}: rewrite metadata differs")
        tensor = entry.get("dequantized_output_tensor")
        producers = [node for node in custom.graph.node if tensor in node.output]
        if tensor != site.output_tensor or len(producers) != 1 or producers[0].op_type != "Cast":
            raise ValueError(f"unproven dequantized output for {site.site_id}")
        fp32_mapping[site.site_id] = site.output_tensor
        custom_mapping[site.site_id] = tensor
    fp32_mapping["logits"] = baseline.graph.output[0].name
    custom_mapping["logits"] = custom.graph.output[0].name
    return fp32_mapping, custom_mapping


def instrument_model(model, mapping):
    """Return a copy whose only outputs are the 17 ordered sites, then logits."""
    if len(mapping) != 18 or list(mapping)[-1] != "logits":
        raise ValueError("instrumentation requires 17 ordered semantic sites plus final logits")
    tensors = set(name for node in model.graph.node for name in node.output)
    if not set(mapping.values()).issubset(tensors):
        raise ValueError("instrumentation tensor is not produced by the source graph")
    copied = copy.deepcopy(model)
    try:
        copied = onnx.shape_inference.infer_shapes(copied)
    except (onnx.shape_inference.InferenceError, ValueError):
        pass
    known = {item.name: item for item in list(copied.graph.value_info) + list(copied.graph.output)}
    outputs = []
    for site_id, tensor in mapping.items():
        value = known.get(tensor)
        if value is None or not value.type.tensor_type.HasField("shape"):
            rank = 2 if site_id == "logits" else 4
            value = onnx.helper.make_tensor_value_info(tensor, onnx.TensorProto.FLOAT, [None] * rank)
        outputs.append(copy.deepcopy(value))
    del copied.graph.output[:]
    copied.graph.output.extend(outputs)
    onnx.checker.check_model(copied)
    return copied
