"""Debug-only Standard-QDQ probes and numerical evidence; no acceptance tolerance."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx

from .backends.onnx_model_rewrite import discover_silu_patterns
from .backends.openvino_backend import generated_path

QDQ = {"QuantizeLinear", "DequantizeLinear"}


def diagnostic_path(root, value):
    path = generated_path(root, value, "report")
    relative = path.relative_to(Path(root).resolve() / "results/benchmarks")
    if not relative.parts[0].startswith("v0.8.1_diagnosis"):
        raise ValueError("diagnostics must use a new results/benchmarks/v0.8.1_diagnosis* directory")
    return path


def array_digest(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(values.shape), "dtype": str(values.dtype)}, sort_keys=True).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def catalog(model):
    """Every source value has a unique producer and zero-based ONNX node position."""
    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    rows = {}

    def add(name, index, output_index, node=None):
        if name in rows:
            raise ValueError("ambiguous source value: " + name)
        inputs = list(node.input) if node else []
        rows[name] = {
            "source_value": name, "instrumented_output": name,
            "topological_index": index, "producer_output_index": output_index,
            "producer_name": node.name if node else "graph_input",
            "producer_op": node.op_type if node else "Input", "producer_inputs": inputs,
            "input_dependent": node is None or any(rows.get(x, {}).get("input_dependent", False) for x in inputs),
            "qdq_context": {
                "producer_is_qdq": bool(node and node.op_type in QDQ),
                "after_qdq": any(rows.get(x, {}).get("producer_op") in QDQ for x in inputs),
                "before_qdq": any(n.op_type in QDQ for n in consumers.get(name, [])),
                "consumer_ops": [n.op_type for n in consumers.get(name, [])],
            },
        }
    for i, value in enumerate(model.graph.input):
        add(value.name, -1, i)
    for i, node in enumerate(model.graph.node):
        for j, name in enumerate(node.output):
            if name:
                add(name, i, j, node)
    return rows


def semantic_sites(model, baseline, manifest):
    """Bind original named SiLU expressions through Q/DQ, never positional guessing."""
    sites = discover_silu_patterns(baseline)
    if len(sites) != 17 or [s.site_id for s in sites] != [s["site_id"] for s in manifest["sites"]]:
        raise ValueError("17 ordered baseline sites must match the committed manifest")
    producers = {name: node for node in model.graph.node for name in node.output}

    def strip(value):
        visited = set()
        while value in producers and producers[value].op_type in QDQ:
            if value in visited:
                raise ValueError("cyclic QDQ provenance")
            visited.add(value)
            value = producers[value].input[0]
        return value

    result = {}
    for site in sites:
        mul = producers.get(site.output_tensor)
        if mul is None or mul.op_type != "Mul" or mul.name != site.mul_node_name or len(mul.input) != 2:
            raise ValueError("missing semantic Mul: " + site.site_id)
        matched = False
        for a, b in (mul.input, list(reversed(mul.input))):
            sigmoid = producers.get(strip(b))
            if (sigmoid is not None and sigmoid.op_type == "Sigmoid" and sigmoid.name == site.sigmoid_node_name
                    and strip(a) == strip(sigmoid.input[0]) == site.input_tensor):
                matched = True
        if not matched:
            raise ValueError("unproven QDQ SiLU connections: " + site.site_id)
        result[site.output_tensor] = site.site_id
    return result


def select_probes(model, sites, node_start=None, node_end=None):
    rows = catalog(model)
    selected = {}

    def take(value, role):
        if value not in rows:
            raise ValueError("missing probe value: " + value)
        selected.setdefault(value, {**rows[value], "roles": []})["roles"].append(role)

    for value in model.graph.input:
        take(value.name, "model_input")
    if node_start is not None or node_end is not None:
        if node_start is None or node_end is None or not 0 <= node_start <= node_end < len(model.graph.node):
            raise ValueError("narrowing requires a valid inclusive ONNX node range")
        for node in model.graph.node[node_start:node_end + 1]:
            for name in node.output:
                if name:
                    take(name, "narrowed_node_output")
    else:
        conv = next(n for n in model.graph.node if n.op_type == "Conv")
        take(conv.output[0], "stem_conv")
        for value, identity in sites.items():
            take(value, "silu:" + identity)
            if identity.endswith(".call_1"):
                take(value, "residual_block_output_before_qdq")
        for node in model.graph.node:
            if node.op_type == "Add":
                take(node.output[0], "residual_add")
            if node.op_type == "GlobalAveragePool":
                take(node.input[0], "pre_pool")
                take(node.output[0], "pool")
            if node.op_type in ("Flatten", "Reshape"):
                take(node.output[0], "flatten")
    for value in model.graph.output:
        take(value.name, "logits")
    return sorted(selected.values(), key=lambda r: (r["topological_index"], r["producer_output_index"]))


def validate_probes(model, probes):
    rows = catalog(model)
    seen = set()
    positions = []
    for probe in probes:
        value = probe["source_value"]
        if value not in rows or value in seen:
            raise ValueError("missing or duplicate source probe: " + value)
        if any(probe.get(key) != expected for key, expected in rows[value].items()):
            raise ValueError("unverified producer/output mapping: " + value)
        seen.add(value)
        positions.append((probe["topological_index"], probe["producer_output_index"]))
    if not probes or positions != sorted(positions):
        raise ValueError("probes must retain source topological order")


def instrument(model, probes):
    """Append existing verified tensors to outputs; retain all source nodes/weights."""
    validate_probes(model, probes)
    known_model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    known = {v.name: v for v in list(known_model.graph.input) + list(known_model.graph.output) + list(known_model.graph.value_info)}
    debug = copy.deepcopy(model)
    names = [v.name for v in debug.graph.output]
    for probe in probes:
        name = probe["source_value"]
        if name not in known or not known[name].type.tensor_type.HasField("shape"):
            raise ValueError("missing inferred probe shape/type: " + name)
        if name not in names:
            debug.graph.output.append(copy.deepcopy(known[name]))
            names.append(name)
    onnx.checker.check_model(debug)
    mapping = [{**p, "instrumented_output_index": names.index(p["source_value"])} for p in probes]
    return debug, mapping


def validate_output_names(expected, actual):
    """Do not silently accept missing, aliased-to-wrong-slot or reordered outputs."""
    if len(expected) != len(actual) or len(set(expected)) != len(expected):
        raise ValueError("output mapping count/uniqueness mismatch")
    for index, (name, aliases) in enumerate(zip(expected, actual)):
        if name not in aliases or any(other in aliases for other in expected if other != name):
            raise ValueError(f"missing/reordered/ambiguous output at {index}: expected {name}, got {aliases}")


def metrics(reference, candidate):
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape:
        raise ValueError(f"probe shape mismatch: {a.shape} vs {b.shape}")
    if not a.size or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("nonempty finite probes required")
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    difference = x - y
    maximum_index = int(np.argmax(np.abs(difference)))
    norm_a, norm_b = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    cosine = float(np.dot(x, y) / (norm_a * norm_b)) if norm_a and norm_b else float(norm_a == norm_b)
    return {
        "shape": list(a.shape), "ort_dtype": str(a.dtype), "openvino_dtype": str(b.dtype),
        "exact_equal": bool(a.dtype == b.dtype and np.array_equal(a, b)),
        "different_elements": int(np.count_nonzero(a != b)), "elements": int(a.size),
        "max_absolute_error": float(np.abs(difference).max()), "mean_absolute_error": float(np.abs(difference).mean()),
        "mse": float(np.mean(difference * difference)), "cosine_similarity": cosine,
        "max_error_index": [int(i) for i in np.unravel_index(maximum_index, a.shape)],
        "ort_at_max_error": float(x[maximum_index]), "openvino_at_max_error": float(y[maximum_index]),
    }


def first_divergence(rows, input_dependent_only=False, integer_only=False):
    """Strict observational inequality, NOT a model acceptance threshold."""
    considered = [r for r in rows if (not input_dependent_only or r["input_dependent"])
                  and (not integer_only or np.issubdtype(np.dtype(r["metrics"]["ort_dtype"]), np.integer))]
    positions = [r["topological_index"] for r in considered]
    if positions != sorted(positions):
        raise ValueError("divergence rows must be topologically ordered")
    last = None
    for row in considered:
        if not row["metrics"]["exact_equal"]:
            return {"first_divergent": row, "last_preceding_exact_match": last}
        last = row
    return {"first_divergent": None, "last_preceding_exact_match": last}


def quantizer_control_model(model, node_index, input_shape):
    """Copy one source quantizer with unchanged parameters for a same-input control."""
    node = model.graph.node[node_index]
    if node.op_type != "QuantizeLinear" or len(node.input) != 3:
        raise ValueError("control requires an explicit source QuantizeLinear")
    initializers = {v.name: v for v in model.graph.initializer}
    if not all(name in initializers for name in node.input[1:]):
        raise ValueError("control requires constant source quantization parameters")
    graph = onnx.helper.make_graph([copy.deepcopy(node)], "diagnostic_same_input_quantizer",
        [onnx.helper.make_tensor_value_info(node.input[0], onnx.TensorProto.FLOAT, list(input_shape))],
        [onnx.helper.make_tensor_value_info(node.output[0], initializers[node.input[2]].data_type, list(input_shape))],
        [copy.deepcopy(initializers[name]) for name in node.input[1:]])
    copied = copy.deepcopy(model)
    copied.graph.CopyFrom(graph)
    onnx.checker.check_model(copied)
    return copied


def rounding_boundary_evidence(a, b, qa, qb, scale, zero_point):
    """Observed code changes and distance to half-integers, not a tolerance test."""
    if a.shape != b.shape or a.shape != qa.shape or a.shape != qb.shape:
        raise ValueError("rounding evidence shape mismatch")
    scale = np.asarray(scale)
    if scale.size != 1 or float(scale.reshape(-1)[0]) <= 0:
        return {"status": "unavailable", "reason": "boundary summary currently requires positive per-tensor scale"}
    flat = np.flatnonzero(qa.ravel() != qb.ravel())
    x, y = a.ravel()[flat], b.ravel()[flat]
    ratios_a, ratios_b = x / scale.reshape(()), y / scale.reshape(())
    distance_a = np.abs(ratios_a.astype(np.float64) - (np.floor(ratios_a.astype(np.float64)) + .5))
    distance_b = np.abs(ratios_b.astype(np.float64) - (np.floor(ratios_b.astype(np.float64)) + .5))
    examples = []
    for i, position in enumerate(flat[:64]):
        examples.append({"index": [int(j) for j in np.unravel_index(position, a.shape)],
                         "ort_input": float(x[i]), "openvino_input": float(y[i]),
                         "ort_code": int(qa.ravel()[position]), "openvino_code": int(qb.ravel()[position]),
                         "ort_input_div_scale_float32": float(ratios_a[i]), "openvino_input_div_scale_float32": float(ratios_b[i]),
                         "ort_distance_to_half_integer": float(distance_a[i]), "openvino_distance_to_half_integer": float(distance_b[i])})
    return {"status": "observed", "scale": float(scale.reshape(-1)[0]), "zero_point": int(np.asarray(zero_point).reshape(())),
            "code_disagreements": int(flat.size), "same_input_but_different_code_count": int(np.count_nonzero(x == y)),
            "max_distance_to_half_integer_at_disagreements": float(max(distance_a.max(), distance_b.max())) if flat.size else None,
            "examples": examples, "example_count": len(examples), "examples_truncated": len(examples) < flat.size,
            "note": "Distances are measured evidence only; no threshold or rounding implementation is substituted into inference."}
