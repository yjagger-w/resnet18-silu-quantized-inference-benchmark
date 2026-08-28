"""Hash-locked v1.2 rewrite contract for the selected v1.1 SiLU candidate.

This module is source-independent: it creates and audits custom-domain ONNX
nodes, but it does not imply that an ORT custom-op library is installed.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import onnx
from onnx import helper, numpy_helper

from silu_benchmark.quantization.spec import PiecewiseQuantizationSpec


CUSTOM_DOMAIN = "com.yjagger.silu"
CUSTOM_OP_TYPE = "QuantizedPiecewiseSiLU"
CUSTOM_OPSET_VERSION = 1
CONTRACT_VERSION = 1
EXPECTED_SITE_COUNT = 17
EXPECTED_SOURCE_NODE_COUNT_PER_SITE = 24
EXPECTED_SOURCE_OPERATOR_PATH = (
    "Sigmoid", "Mul", "Cast", "Clip", "Less", "Div", "Add", "Round",
    "Clip", "Div", "Add", "Round", "Clip", "Where", "Cast", "Cast",
    "Less", "Sub", "Mul", "Sub", "Mul", "Where", "Clip", "Cast",
)
SOURCE_METADATA_KEY = "silu_benchmark.v11_accuracy_recovery"
REWRITE_METADATA_KEY = "silu_benchmark.v12_ort_cpp_customop"
MANIFEST_SCHEMA = "ort-cpp-customop-rewrite-manifest/v1"
REPORT_SCHEMA = "ort-cpp-customop-graph-report/v1"
TOOL_VERSION = "1.2.0"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_hash(payload: Mapping) -> str:
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _prefix(site_id: str) -> str:
    return "silu_v11_" + re.sub(r"[^A-Za-z0-9_]", "_", site_id)


def _custom_name(site_id: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9_]", "_", site_id)
    return f"silu_v12_{suffix}_QuantizedPiecewiseSiLU"


def _float_hex(value: float) -> str:
    return float(value).hex()


def _float32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", np.float32(value)))[0]


def _metadata(model: onnx.ModelProto, key: str) -> dict:
    values = [item.value for item in model.metadata_props if item.key == key]
    if len(values) != 1:
        raise ValueError(f"model must contain exactly one {key!r} metadata entry")
    return json.loads(values[0])


def _scalar(initializer: onnx.TensorProto):
    value = numpy_helper.to_array(initializer)
    if value.shape != ():
        raise ValueError(f"initializer must be scalar: {initializer.name}")
    return value.item()


def _node_record(index: int, node: onnx.NodeProto) -> dict:
    return {
        "index": index,
        "name": node.name,
        "op_type": node.op_type,
        "domain": node.domain,
        "inputs": list(node.input),
        "outputs": list(node.output),
        "sha256": sha256_bytes(node.SerializeToString(deterministic=True)),
    }


def _attribute_value(node: onnx.NodeProto, name: str):
    matches = [attribute for attribute in node.attribute if attribute.name == name]
    if len(matches) != 1:
        raise ValueError(f"node {node.name} must contain one {name!r} attribute")
    return helper.get_attribute_value(matches[0])


@dataclass(frozen=True)
class SelectedPiecewiseIsland:
    site_id: str
    source_node_indices: tuple[int, ...]
    source_node_names: tuple[str, ...]
    source_operator_path: tuple[str, ...]
    source_initializer_names: tuple[str, ...]
    input_code_tensor: str
    input_dequantized_tensor: str
    input_scale_initializer: str
    input_zero_point_initializer: str
    input_scale: float
    input_zero_point: int
    output_code_tensor: str
    output_dequantized_tensor: str
    vmin: float
    vsplit: float
    vmax: float
    bits: int
    lower_scale: float
    upper_scale: float
    lower_zero_point: int
    upper_zero_point: int

    @property
    def custom_node_name(self) -> str:
        return _custom_name(self.site_id)

    @property
    def custom_attributes(self) -> dict:
        return {
            "contract_version": CONTRACT_VERSION,
            "site_id": self.site_id,
            "input_scale_hex": _float_hex(self.input_scale),
            "input_scale_float32_bits": _float32_bits(self.input_scale),
            "input_zero_point": self.input_zero_point,
            "input_qmin": 0,
            "input_qmax": 255,
            "vmin_hex": _float_hex(self.vmin),
            "vsplit_hex": _float_hex(self.vsplit),
            "vmax_hex": _float_hex(self.vmax),
            "bits": self.bits,
            "lower_scale_hex": _float_hex(self.lower_scale),
            "upper_scale_hex": _float_hex(self.upper_scale),
            "lower_zero_point": self.lower_zero_point,
            "upper_zero_point": self.upper_zero_point,
            "lower_code_start": 0,
            "lower_code_end": 127,
            "upper_code_start": 128,
            "upper_code_end": 255,
            "split_ownership": "upper",
            "rounding": "nearest_even",
        }


def verify_selected_source(
    model_path: Path,
    selection_receipt_path: Path,
    *,
    expected_model_sha256: str | None = None,
) -> tuple[onnx.ModelProto, dict]:
    receipt = json.loads(selection_receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "silu-accuracy-selection/v1":
        raise ValueError("selection receipt schema is not v1.1")
    if receipt.get("candidate_id") != "two_segment_p99_99_mse":
        raise ValueError("selection receipt does not identify the frozen v1.1 candidate")
    if receipt.get("frozen_before_final_test") is not True:
        raise ValueError("selection receipt was not frozen before final test")
    actual_hash = sha256_file(model_path)
    if actual_hash != receipt.get("model_sha256"):
        raise ValueError("selected source model hash differs from selection receipt")
    if expected_model_sha256 is not None and actual_hash != expected_model_sha256:
        raise ValueError("selected source model hash differs from v1.2 configuration")
    model = onnx.load(str(model_path))
    metadata = _metadata(model, SOURCE_METADATA_KEY)
    if metadata.get("mode") != "piecewise":
        raise ValueError("selected source model is not a piecewise rewrite")
    if metadata.get("candidate_id") != receipt["candidate_id"]:
        raise ValueError("selected model metadata and receipt candidate IDs differ")
    return model, receipt


def discover_selected_piecewise_islands(
    model: onnx.ModelProto,
) -> tuple[SelectedPiecewiseIsland, ...]:
    metadata = _metadata(model, SOURCE_METADATA_KEY)
    selected_ids = metadata.get("selected_site_ids")
    site_specs = metadata.get("site_specs")
    source_sites = metadata.get("sites")
    if not isinstance(selected_ids, list) or len(selected_ids) != EXPECTED_SITE_COUNT:
        raise ValueError("selected model must list exactly 17 ordered site IDs")
    if len(set(selected_ids)) != EXPECTED_SITE_COUNT:
        raise ValueError("selected model contains duplicate site IDs")
    if set(site_specs or {}) != set(selected_ids):
        raise ValueError("selected model site specifications are incomplete")
    source_by_id = {site["site_id"]: site for site in source_sites or []}
    if set(source_by_id) != set(selected_ids):
        raise ValueError("selected model source-site metadata is incomplete")

    producers = {
        output: (index, node)
        for index, node in enumerate(model.graph.node)
        for output in node.output
    }
    initializers = {item.name: item for item in model.graph.initializer}
    consumers = {}
    for index, node in enumerate(model.graph.node):
        for tensor in node.input:
            consumers.setdefault(tensor, []).append((index, node))
    islands = []
    owned_indices = set()
    for site_id in selected_ids:
        prefix = _prefix(site_id)
        nodes = [
            (index, node)
            for index, node in enumerate(model.graph.node)
            if node.name.startswith(prefix + "_")
        ]
        if len(nodes) != EXPECTED_SOURCE_NODE_COUNT_PER_SITE:
            raise ValueError(f"{site_id} must contain exactly 24 selected piecewise nodes")
        indices = tuple(index for index, _ in nodes)
        if owned_indices.intersection(indices):
            raise ValueError("selected piecewise islands overlap")
        owned_indices.update(indices)
        operator_path = tuple(node.op_type for _, node in nodes)
        if operator_path != EXPECTED_SOURCE_OPERATOR_PATH:
            raise ValueError(f"selected piecewise operator path differs at {site_id}")
        source_site = source_by_id[site_id]
        input_dequantized = source_site["input_tensor"]
        dq = producers.get(input_dequantized)
        if dq is None or dq[1].op_type != "DequantizeLinear" or len(dq[1].input) != 3:
            raise ValueError(f"{site_id} input is not produced by scalar-parameter DQ")
        input_code, scale_name, zero_point_name = dq[1].input
        if scale_name not in initializers or zero_point_name not in initializers:
            raise ValueError(f"{site_id} DQ parameters are not initializers")
        scale_initializer = initializers[scale_name]
        zero_initializer = initializers[zero_point_name]
        scale = float(_scalar(scale_initializer))
        zero_point = int(_scalar(zero_initializer))
        if scale_initializer.data_type != onnx.TensorProto.FLOAT or not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"{site_id} input scale must be positive float32")
        if zero_initializer.data_type != onnx.TensorProto.UINT8 or not 0 <= zero_point <= 255:
            raise ValueError(f"{site_id} input zero point must be uint8")
        if not any(node.name.startswith(prefix + "_") for _, node in consumers[input_dequantized]):
            raise ValueError(f"{site_id} DQ tensor does not feed its selected island")
        spec_payload = site_specs[site_id]
        spec = PiecewiseQuantizationSpec(
            float(spec_payload["vmin"]),
            float(spec_payload["vsplit"]),
            float(spec_payload["vmax"]),
            int(spec_payload["bits"]),
        )
        referenced_initializers = tuple(
            sorted(
                {
                    name
                    for _, node in nodes
                    for name in node.input
                    if name in initializers and name.startswith(prefix + "_")
                }
            )
        )
        output_code = f"{prefix}_quantized_codes"
        output_dequantized = source_site["output_tensor"]
        if output_code not in producers or output_dequantized not in producers:
            raise ValueError(f"{site_id} selected outputs are not produced")
        if producers[output_code][0] not in indices or producers[output_dequantized][0] not in indices:
            raise ValueError(f"{site_id} selected outputs bypass the piecewise island")
        islands.append(
            SelectedPiecewiseIsland(
                site_id=site_id,
                source_node_indices=indices,
                source_node_names=tuple(node.name for _, node in nodes),
                source_operator_path=operator_path,
                source_initializer_names=referenced_initializers,
                input_code_tensor=input_code,
                input_dequantized_tensor=input_dequantized,
                input_scale_initializer=scale_name,
                input_zero_point_initializer=zero_point_name,
                input_scale=scale,
                input_zero_point=zero_point,
                output_code_tensor=output_code,
                output_dequantized_tensor=output_dequantized,
                vmin=spec.vmin,
                vsplit=spec.vsplit,
                vmax=spec.vmax,
                bits=spec.bits,
                lower_scale=spec.lower_scale,
                upper_scale=spec.upper_scale,
                lower_zero_point=spec.lower_zero_point,
                upper_zero_point=spec.upper_zero_point,
            )
        )
    if len(owned_indices) != EXPECTED_SITE_COUNT * EXPECTED_SOURCE_NODE_COUNT_PER_SITE:
        raise ValueError("selected piecewise island ownership is incomplete")
    return tuple(islands)


def _make_custom_node(island: SelectedPiecewiseIsland) -> onnx.NodeProto:
    attributes = island.custom_attributes
    return helper.make_node(
        CUSTOM_OP_TYPE,
        [island.input_code_tensor, island.input_dequantized_tensor],
        [island.output_code_tensor, island.output_dequantized_tensor],
        name=island.custom_node_name,
        domain=CUSTOM_DOMAIN,
        **attributes,
    )


def rewrite_selected_candidate(
    model: onnx.ModelProto,
    *,
    source_model_sha256: str,
    selection_receipt_hash: str,
) -> tuple[onnx.ModelProto, tuple[SelectedPiecewiseIsland, ...], dict]:
    islands = discover_selected_piecewise_islands(model)
    by_start = {island.source_node_indices[0]: island for island in islands}
    removed_indices = {
        index for island in islands for index in island.source_node_indices
    }
    nodes = []
    for index, node in enumerate(model.graph.node):
        if index in by_start:
            nodes.append(_make_custom_node(by_start[index]))
        elif index not in removed_indices:
            nodes.append(copy.deepcopy(node))
    referenced = {name for node in nodes for name in node.input}
    initializers = [
        copy.deepcopy(item) for item in model.graph.initializer if item.name in referenced
    ]
    graph = helper.make_graph(
        nodes,
        f"{model.graph.name}_v12_ort_cpp_customop",
        [copy.deepcopy(item) for item in model.graph.input],
        [copy.deepcopy(item) for item in model.graph.output],
        initializer=initializers,
        value_info=[copy.deepcopy(item) for item in model.graph.value_info],
    )
    rewritten = copy.deepcopy(model)
    rewritten.graph.CopyFrom(graph)
    if any(item.domain == CUSTOM_DOMAIN for item in rewritten.opset_import):
        raise ValueError(f"source model already imports custom domain {CUSTOM_DOMAIN}")
    rewritten.opset_import.add(domain=CUSTOM_DOMAIN, version=CUSTOM_OPSET_VERSION)
    if any(item.key == REWRITE_METADATA_KEY for item in rewritten.metadata_props):
        raise ValueError("source model already contains v1.2 rewrite metadata")
    contract_rows = [
        {
            "site_id": island.site_id,
            "custom_node_name": island.custom_node_name,
            "inputs": [island.input_code_tensor, island.input_dequantized_tensor],
            "outputs": [island.output_code_tensor, island.output_dequantized_tensor],
            "attributes": island.custom_attributes,
        }
        for island in islands
    ]
    rewritten.metadata_props.add(
        key=REWRITE_METADATA_KEY,
        value=json.dumps(
            {
                "tool_version": TOOL_VERSION,
                "source_model_sha256": source_model_sha256,
                "selection_receipt_hash": selection_receipt_hash,
                "custom_domain": CUSTOM_DOMAIN,
                "custom_op_type": CUSTOM_OP_TYPE,
                "custom_opset_version": CUSTOM_OPSET_VERSION,
                "contract_version": CONTRACT_VERSION,
                "sites": contract_rows,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    onnx.checker.check_model(rewritten, check_custom_domain=False)
    custom_nodes = [
        node for node in rewritten.graph.node
        if node.domain == CUSTOM_DOMAIN and node.op_type == CUSTOM_OP_TYPE
    ]
    retained = {
        node.name for node in rewritten.graph.node
    }.intersection({name for island in islands for name in island.source_node_names})
    report = {
        "schema_version": REPORT_SCHEMA,
        "source_piecewise_islands_removed": len(islands),
        "source_piecewise_nodes_removed": len(removed_indices),
        "custom_nodes_added": len(custom_nodes),
        "original_selected_piecewise_nodes_retained": len(retained),
        "source_node_count": len(model.graph.node),
        "rewritten_node_count": len(rewritten.graph.node),
        "source_initializer_count": len(model.graph.initializer),
        "rewritten_initializer_count": len(rewritten.graph.initializer),
        "pruned_initializer_count": len(model.graph.initializer) - len(rewritten.graph.initializer),
        "custom_domain": CUSTOM_DOMAIN,
        "custom_op_type": CUSTOM_OP_TYPE,
        "sites": contract_rows,
    }
    if (
        report["source_piecewise_islands_removed"] != EXPECTED_SITE_COUNT
        or report["custom_nodes_added"] != EXPECTED_SITE_COUNT
        or report["original_selected_piecewise_nodes_retained"] != 0
    ):
        raise RuntimeError("v1.2 graph rewrite coverage is incomplete")
    return rewritten, islands, report


def build_sidecar_manifest(
    *,
    source_path: Path,
    rewritten_path: Path,
    selection_receipt_path: Path,
    islands: Sequence[SelectedPiecewiseIsland],
    graph_report: Mapping,
    generated_at_utc: str | None = None,
) -> dict:
    receipt = json.loads(selection_receipt_path.read_text(encoding="utf-8"))
    timestamp = generated_at_utc or dt.datetime.now(dt.timezone.utc).isoformat()
    payload = {
        "schema_version": MANIFEST_SCHEMA,
        "tool_version": TOOL_VERSION,
        "generated_at_utc": timestamp,
        "source_model": str(source_path.resolve()),
        "source_model_sha256": sha256_file(source_path),
        "rewritten_model": str(rewritten_path.resolve()),
        "rewritten_model_sha256": sha256_file(rewritten_path),
        "selection_receipt": str(selection_receipt_path.resolve()),
        "selection_receipt_sha256": sha256_file(selection_receipt_path),
        "selection_receipt_hash": receipt["selection_receipt_hash"],
        "candidate_id": receipt["candidate_id"],
        "custom_domain": CUSTOM_DOMAIN,
        "custom_op_type": CUSTOM_OP_TYPE,
        "custom_opset_version": CUSTOM_OPSET_VERSION,
        "contract_version": CONTRACT_VERSION,
        "site_count": len(islands),
        "graph_report": dict(graph_report),
        "sites": [
            {
                "site_id": island.site_id,
                "source_island_nodes": list(island.source_node_names),
                "source_island_operator_path": list(island.source_operator_path),
                "source_island_initializers": list(island.source_initializer_names),
                "custom_node_name": island.custom_node_name,
                "entry_code_tensor": island.input_code_tensor,
                "entry_dequantized_tensor": island.input_dequantized_tensor,
                "exit_code_tensor": island.output_code_tensor,
                "exit_dequantized_tensor": island.output_dequantized_tensor,
                "attributes": island.custom_attributes,
            }
            for island in islands
        ],
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def validate_custom_node_contract(model: onnx.ModelProto, manifest: Mapping) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("invalid v1.2 sidecar schema")
    if manifest.get("site_count") != EXPECTED_SITE_COUNT:
        raise ValueError("v1.2 sidecar must contain exactly 17 sites")
    rows = manifest.get("sites")
    nodes = {
        node.name: node for node in model.graph.node
        if node.domain == CUSTOM_DOMAIN and node.op_type == CUSTOM_OP_TYPE
    }
    if len(nodes) != EXPECTED_SITE_COUNT or len(rows or []) != EXPECTED_SITE_COUNT:
        raise ValueError("custom-op model/sidecar does not contain exactly 17 nodes")
    for row in rows:
        node = nodes.get(row["custom_node_name"])
        if node is None:
            raise ValueError(f"missing custom node: {row['custom_node_name']}")
        if list(node.input) != [row["entry_code_tensor"], row["entry_dequantized_tensor"]]:
            raise ValueError(f"custom node inputs differ at {row['site_id']}")
        if list(node.output) != [row["exit_code_tensor"], row["exit_dequantized_tensor"]]:
            raise ValueError(f"custom node outputs differ at {row['site_id']}")
        expected = row["attributes"]
        observed = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        normalized = {
            key: value.decode("utf-8") if isinstance(value, bytes) else value
            for key, value in observed.items()
        }
        if normalized != expected:
            raise ValueError(f"custom node attributes differ at {row['site_id']}")


def validate_generated_artifact(
    *,
    source_path: Path,
    rewritten_path: Path,
    sidecar_path: Path,
    selection_receipt_path: Path,
) -> dict:
    manifest = json.loads(sidecar_path.read_text(encoding="utf-8"))
    stored_hash = manifest.get("manifest_hash")
    unhashed = dict(manifest)
    unhashed.pop("manifest_hash", None)
    if stored_hash != canonical_hash(unhashed):
        raise ValueError("v1.2 sidecar manifest hash mismatch")
    checks = {
        "source_model_sha256": sha256_file(source_path),
        "rewritten_model_sha256": sha256_file(rewritten_path),
        "selection_receipt_sha256": sha256_file(selection_receipt_path),
    }
    for key, actual in checks.items():
        if manifest.get(key) != actual:
            raise ValueError(f"stale or mismatched v1.2 artifact: {key}")
    model = onnx.load(str(rewritten_path))
    validate_custom_node_contract(model, manifest)
    return manifest


def validate_generated_output_path(path: Path, repository_root: Path) -> Path:
    root = repository_root.resolve()
    resolved = path.resolve()
    allowed = (root / "artifacts/accuracy_recovery/v1.2").resolve()
    reports = (root / "results/benchmarks/v1.2_ort_cpp_customop").resolve()
    if not any(resolved == base or base in resolved.parents for base in (allowed, reports)):
        raise ValueError("v1.2 generated output must stay under an isolated ignored path")
    return resolved
