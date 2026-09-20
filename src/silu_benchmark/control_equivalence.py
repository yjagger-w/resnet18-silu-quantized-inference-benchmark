"""v1.1.1 strict QDQ topology and target-tensor equivalence audit helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import onnx
from onnx import helper, numpy_helper

from silu_benchmark.accuracy_recovery import (
    EXPECTED_SITE_COUNT,
    QDQSiLUSite,
    discover_qdq_silu_sites,
)


AUDIT_METADATA_KEY = "silu_benchmark.v111_control_equivalence"
AUDIT_SCHEMA = "silu-control-equivalence-audit/v1"
TARGET_STAGES = ("pre_silu", "post_silu", "activation_output")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def protobuf_hash(value) -> str:
    return sha256_bytes(value.SerializeToString(deterministic=True))


def graph_topology_hash(model: onnx.ModelProto) -> str:
    """Hash the graph only, deliberately excluding model metadata."""
    return protobuf_hash(model.graph)


def initializer_record(initializer: onnx.TensorProto) -> dict:
    array = numpy_helper.to_array(initializer)
    return {
        "name": initializer.name,
        "data_type": onnx.TensorProto.DataType.Name(initializer.data_type),
        "shape": list(initializer.dims),
        "tensor_proto_sha256": protobuf_hash(initializer),
        "value_sha256": sha256_bytes(np.ascontiguousarray(array).tobytes()),
    }


def node_record(index: int, node: onnx.NodeProto) -> dict:
    return {
        "index": index,
        "name": node.name,
        "op_type": node.op_type,
        "inputs": list(node.input),
        "outputs": list(node.output),
        "attributes_sha256": sha256_bytes(
            b"".join(item.SerializeToString(deterministic=True) for item in node.attribute)
        ),
    }


def build_strict_topology_control(model: onnx.ModelProto):
    """Copy the model while proving every graph field and target island is unchanged."""
    sites = discover_qdq_silu_sites(model)
    copied = copy.deepcopy(model)
    if graph_topology_hash(copied) != graph_topology_hash(model):
        raise RuntimeError("strict control graph changed during copy")
    if [protobuf_hash(node) for node in copied.graph.node] != [
        protobuf_hash(node) for node in model.graph.node
    ]:
        raise RuntimeError("strict control node sequence differs from source")
    if [protobuf_hash(item) for item in copied.graph.initializer] != [
        protobuf_hash(item) for item in model.graph.initializer
    ]:
        raise RuntimeError("strict control initializer sequence differs from source")
    copied.metadata_props.add(
        key=AUDIT_METADATA_KEY,
        value=json.dumps(
            {
                "schema_version": AUDIT_SCHEMA,
                "mode": "strict_operator_topology_preserving_control",
                "source_graph_topology_sha256": graph_topology_hash(model),
                "sites": [asdict(site) for site in sites],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    onnx.checker.check_model(copied)
    if graph_topology_hash(copied) != graph_topology_hash(model):
        raise RuntimeError("strict control provenance changed graph topology")
    return copied, sites


def qdq_inventory(model: onnx.ModelProto) -> dict[str, dict]:
    initializers = {item.name: item for item in model.graph.initializer}
    inventory = {}
    for index, node in enumerate(model.graph.node):
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        parameters = []
        for name in node.input[1:]:
            if name in initializers:
                parameters.append(initializer_record(initializers[name]))
        inventory[node.name] = {**node_record(index, node), "parameter_initializers": parameters}
    return inventory


def compare_qdq_topology(source: onnx.ModelProto, candidate: onnx.ModelProto) -> dict:
    before = qdq_inventory(source)
    after = qdq_inventory(candidate)
    common = sorted(set(before) & set(after))
    moved_boundaries = []
    index_shifts = []
    for name in common:
        left = before[name]
        right = after[name]
        if (
            left["op_type"], left["inputs"], left["outputs"], left["attributes_sha256"]
        ) != (
            right["op_type"], right["inputs"], right["outputs"], right["attributes_sha256"]
        ):
            moved_boundaries.append({"name": name, "source": left, "candidate": right})
        elif left["index"] != right["index"]:
            index_shifts.append(
                {"name": name, "source_index": left["index"], "candidate_index": right["index"]}
            )
    return {
        "source_qdq_count": len(before),
        "candidate_qdq_count": len(after),
        "inserted": [after[name] for name in sorted(set(after) - set(before))],
        "removed": [before[name] for name in sorted(set(before) - set(after))],
        "moved_boundaries": moved_boundaries,
        "unchanged_boundary_index_shifts": index_shifts,
        "exact_qdq_topology": before == after,
    }


def validate_control_label(label: str, source: onnx.ModelProto, candidate: onnx.ModelProto) -> None:
    """Forbid numerical-equivalence terminology when QDQ topology differs."""
    normalized = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    comparison = compare_qdq_topology(source, candidate)
    if "float_equivalent" in normalized and not comparison["exact_qdq_topology"]:
        raise ValueError(
            "float-equivalent label is invalid because QuantizeLinear/DequantizeLinear "
            "topology differs"
        )


def _site_prefix(site_id: str) -> str:
    return "silu_v11_" + re.sub(r"[^A-Za-z0-9_]", "_", site_id)


def _producer_map(model: onnx.ModelProto) -> dict[str, tuple[int, onnx.NodeProto]]:
    return {
        output: (index, node)
        for index, node in enumerate(model.graph.node)
        for output in node.output
    }


def original_stage_mapping(sites: Sequence[QDQSiLUSite]) -> dict[str, str]:
    mapping = {"final_logits": "logits"}
    for site in sites:
        mapping[f"{site.site_id}::pre_silu"] = site.input_tensor
        mapping[f"{site.site_id}::post_silu"] = site.mul_output_tensor
        mapping[f"{site.site_id}::activation_output"] = site.output_tensor
    return mapping


def resolve_original_stage_mapping(
    model: onnx.ModelProto, sites: Sequence[QDQSiLUSite]
) -> dict[str, str]:
    mapping = original_stage_mapping(sites)
    for site in sites:
        mapping[f"{site.site_id}::sigmoid_code"] = model.graph.node[
            site.sigmoid_quantize_node_index
        ].output[0]
        mapping[f"{site.site_id}::output_code"] = model.graph.node[
            site.output_quantize_node_index
        ].output[0]
    return mapping


def rewritten_stage_mapping(
    model: onnx.ModelProto,
    sites: Sequence[QDQSiLUSite],
    *,
    mode: str,
) -> dict[str, str]:
    if mode not in {"semantic_expression", "piecewise"}:
        raise ValueError("unsupported rewritten mapping mode")
    produced = _producer_map(model)
    mapping = {"final_logits": "logits"}
    for site in sites:
        prefix = _site_prefix(site.site_id)
        mapping[f"{site.site_id}::pre_silu"] = site.input_tensor
        mapping[f"{site.site_id}::post_silu"] = (
            site.output_tensor if mode == "semantic_expression" else f"{prefix}_post_silu"
        )
        mapping[f"{site.site_id}::activation_output"] = site.output_tensor
        for tensor in mapping.values():
            if tensor != "logits" and tensor not in produced:
                raise ValueError(f"rewritten instrumentation tensor is not produced: {tensor}")
        if mode == "piecewise":
            code = f"{prefix}_quantized_codes"
            if code not in produced:
                raise ValueError(f"piecewise code tensor is not produced: {code}")
            mapping[f"{site.site_id}::piecewise_code"] = code
    return mapping


def instrument_model_outputs(model: onnx.ModelProto, mapping: Mapping[str, str]):
    """Expose deterministic unique tensors and retain logical aliases separately."""
    if not mapping:
        raise ValueError("instrumentation mapping must not be empty")
    copied = copy.deepcopy(model)
    try:
        copied = onnx.shape_inference.infer_shapes(copied)
    except (onnx.shape_inference.InferenceError, ValueError):
        pass
    produced = {tensor for node in copied.graph.node for tensor in node.output}
    known = {
        item.name: item
        for item in [*copied.graph.input, *copied.graph.output, *copied.graph.value_info]
    }
    unique = []
    for tensor in mapping.values():
        if tensor not in unique:
            unique.append(tensor)
    missing = sorted(tensor for tensor in unique if tensor not in produced)
    if missing:
        raise ValueError(f"instrumentation tensors are not produced: {missing}")
    outputs = []
    for tensor in unique:
        value = known.get(tensor)
        if value is None:
            raise ValueError(f"instrumentation tensor has no inferred type: {tensor}")
        outputs.append(copy.deepcopy(value))
    del copied.graph.output[:]
    copied.graph.output.extend(outputs)
    onnx.checker.check_model(copied)
    return copied, tuple(unique)


def selected_candidate_coverage(
    source: onnx.ModelProto,
    candidate: onnx.ModelProto,
    sites: Sequence[QDQSiLUSite],
) -> dict:
    metadata = [item for item in candidate.metadata_props if item.key.endswith("v11_accuracy_recovery")]
    if len(metadata) != 1:
        raise ValueError("selected candidate has missing or duplicate v1.1 metadata")
    payload = json.loads(metadata[0].value)
    expected = [site.site_id for site in sites]
    if payload.get("mode") != "piecewise" or payload.get("selected_site_ids") != expected:
        raise ValueError("selected candidate metadata does not cover all ordered piecewise sites")
    if set(payload.get("site_specs", {})) != set(expected):
        raise ValueError("selected candidate metadata lacks a site specification")
    candidate_names = {node.name for node in candidate.graph.node}
    producer = _producer_map(candidate)
    candidate_initializers = {item.name: item for item in candidate.graph.initializer}
    rows = []
    for site in sites:
        prefix = _site_prefix(site.site_id)
        original_nodes = [source.graph.node[index] for index in site.replaced_node_indices]
        retained = sorted(node.name for node in original_nodes if node.name in candidate_names)
        inserted = [
            (index, node) for index, node in enumerate(candidate.graph.node)
            if node.name.startswith(prefix + "_")
        ]
        output_producer = producer.get(site.output_tensor)
        if retained or len(inserted) != 24 or output_producer is None:
            raise ValueError(f"incomplete piecewise replacement coverage at {site.site_id}")
        if not output_producer[1].name.startswith(prefix + "_"):
            raise ValueError(f"piecewise output is bypassed at {site.site_id}")
        post_tensor = f"{prefix}_post_silu"
        code_tensor = f"{prefix}_quantized_codes"
        if post_tensor not in producer or code_tensor not in producer:
            raise ValueError(f"piecewise path tensors are missing at {site.site_id}")
        parameter_names = [
            name for name in candidate_initializers
            if name.startswith(prefix + "_") and (
                name.endswith("scale") or name.endswith("zero_point")
            )
        ]
        rows.append(
            {
                "site_id": site.site_id,
                "all_original_six_nodes_removed": not retained,
                "inserted_piecewise_node_count": len(inserted),
                "inserted_operator_path": [node.op_type for _, node in inserted],
                "piecewise_post_tensor": post_tensor,
                "piecewise_code_tensor": code_tensor,
                "activation_output_tensor": site.output_tensor,
                "activation_output_producer": node_record(*output_producer),
                "piecewise_parameter_initializers": [
                    initializer_record(candidate_initializers[name])
                    for name in sorted(parameter_names)
                ],
                "no_original_silu_bypass": True,
            }
        )
    return {
        "candidate_id": payload.get("candidate_id"),
        "metadata_selected_site_count": len(payload["selected_site_ids"]),
        "covered_site_count": len(rows),
        "all_17_piecewise_paths_proven": len(rows) == EXPECTED_SITE_COUNT,
        "sites": rows,
    }


def target_topology_audit(
    source: onnx.ModelProto,
    candidate: onnx.ModelProto,
    sites: Sequence[QDQSiLUSite],
    *,
    variant: str,
) -> list[dict]:
    source_initializers = {item.name: item for item in source.graph.initializer}
    candidate_initializers = {item.name: item for item in candidate.graph.initializer}
    candidate_names = {node.name for node in candidate.graph.node}
    rows = []
    for site in sites:
        original_nodes = [
            node_record(index, source.graph.node[index]) for index in site.replaced_node_indices
        ]
        parameter_names = sorted(
            {
                name
                for record in original_nodes
                if record["op_type"] in {"QuantizeLinear", "DequantizeLinear"}
                for name in record["inputs"][1:]
                if name in source_initializers
            }
        )
        retained = [record["name"] for record in original_nodes if record["name"] in candidate_names]
        prefix = _site_prefix(site.site_id)
        inserted = [
            node_record(index, node) for index, node in enumerate(candidate.graph.node)
            if node.name.startswith(prefix + "_")
        ]
        rows.append(
            {
                "site_id": site.site_id,
                "variant": variant,
                "original_producer_subgraph": original_nodes,
                "original_qdq_parameter_initializers": [
                    initializer_record(source_initializers[name]) for name in parameter_names
                ],
                "retained_original_node_names": retained,
                "removed_original_node_names": [
                    record["name"] for record in original_nodes if record["name"] not in candidate_names
                ],
                "inserted_rewrite_nodes": inserted,
                "original_activation_output_tensor": site.output_tensor,
                "original_initializer_names_present_in_variant": [
                    name for name in parameter_names if name in candidate_initializers
                ],
            }
        )
    return rows


def validate_audit_output_path(path: Path, repository_root: Path, *, artifact: bool = False) -> Path:
    root = repository_root.resolve()
    resolved = path.resolve()
    base = (
        root / "artifacts/accuracy_recovery/v1.1.1"
        if artifact
        else root / "results/benchmarks/v1.1_control_equivalence_audit"
    ).resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError(f"v1.1.1 output must remain under isolated generated path: {base}")
    return resolved
