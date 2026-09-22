"""Standard-QDQ model rewrite for offline SiLU-aware calibration."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import onnx
from onnx import numpy_helper

from .accuracy_recovery import QDQSiLUSite, discover_qdq_silu_sites
from .quantization.hardware_aware_qdq import StandardQDQSpec


METADATA_KEY = "silu_benchmark.v18_hardware_aware_standard_qdq"
REWRITE_SCHEMA = "hardware-aware-standard-qdq-rewrite/v1.8"


@dataclass(frozen=True)
class StandardQDQRewriteResult:
    model: onnx.ModelProto
    sites: tuple[QDQSiLUSite, ...]
    parameter_names: Mapping[str, Mapping[str, str]]
    contract: Mapping[str, object]


def _topology_signature(model: onnx.ModelProto) -> str:
    payload = [
        {
            "name": node.name,
            "domain": node.domain,
            "op_type": node.op_type,
            "outputs": list(node.output),
        }
        for node in model.graph.node
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _initializer_map(model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
    values = {item.name: item for item in model.graph.initializer}
    if len(values) != len(model.graph.initializer):
        raise ValueError("model contains duplicate initializer names")
    return values


def _safe_site_name(site_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", site_id)


def extract_standard_qdq_site_specs(
    model: onnx.ModelProto,
) -> dict[str, StandardQDQSpec]:
    """Read the existing scalar uint8 QDQ encoding for every post-SiLU site."""

    sites = discover_qdq_silu_sites(model)
    initializers = _initializer_map(model)
    specs: dict[str, StandardQDQSpec] = {}
    for site in sites:
        quantize = model.graph.node[site.output_quantize_node_index]
        dequantize = model.graph.node[site.output_dequantize_node_index]
        if len(quantize.input) != 3 or len(dequantize.input) != 3:
            raise ValueError(f"{site.site_id} is not an asymmetric scalar QDQ pair")
        if list(quantize.input[1:]) != list(dequantize.input[1:]):
            raise ValueError(f"{site.site_id} QuantizeLinear/DequantizeLinear parameters differ")
        scale_name, zero_name = quantize.input[1:]
        if scale_name not in initializers or zero_name not in initializers:
            raise ValueError(f"{site.site_id} QDQ parameter initializer is missing")
        scale = numpy_helper.to_array(initializers[scale_name])
        zero_point = numpy_helper.to_array(initializers[zero_name])
        if scale.shape != () or scale.dtype != np.float32 or not float(scale) > 0.0:
            raise ValueError(f"{site.site_id} scale must be a positive float32 scalar")
        if zero_point.shape != () or zero_point.dtype != np.uint8:
            raise ValueError(f"{site.site_id} zero-point must be a uint8 scalar")
        specs[site.site_id] = StandardQDQSpec(
            scale=float(scale),
            zero_point=int(zero_point),
            bits=8,
        )
    return specs


def _validate_site_parameters(
    sites: tuple[QDQSiLUSite, ...], site_specs: Mapping[str, StandardQDQSpec]
) -> None:
    expected = [site.site_id for site in sites]
    missing = sorted(set(expected) - set(site_specs))
    unused = sorted(set(site_specs) - set(expected))
    invalid = sorted(
        key
        for key, value in site_specs.items()
        if not isinstance(value, StandardQDQSpec) or value.bits != 8
    )
    if missing or unused or invalid:
        raise ValueError(
            f"standard QDQ site parameter mismatch: missing={missing}, "
            f"unused={unused}, invalid={invalid}"
        )


def validate_standard_qdq_rewrite(
    source: onnx.ModelProto, rewritten: onnx.ModelProto
) -> dict[str, object]:
    """Prove that only scalar parameters changed and no runtime branch was added."""

    if len(source.graph.node) != len(rewritten.graph.node):
        raise ValueError("standard QDQ rewrite must preserve node count")
    if _topology_signature(source) != _topology_signature(rewritten):
        raise ValueError("standard QDQ rewrite must preserve operator topology")
    source_sites = discover_qdq_silu_sites(source)
    rewritten_sites = discover_qdq_silu_sites(rewritten)
    if [site.site_id for site in source_sites] != [site.site_id for site in rewritten_sites]:
        raise ValueError("standard QDQ rewrite changed SiLU site identity or order")

    initializers = _initializer_map(rewritten)
    target_parameters = {}
    for site in rewritten_sites:
        quantize = rewritten.graph.node[site.output_quantize_node_index]
        dequantize = rewritten.graph.node[site.output_dequantize_node_index]
        if len(quantize.input) != 3 or len(dequantize.input) != 3:
            raise ValueError(f"{site.site_id} must retain scalar asymmetric QDQ inputs")
        if list(quantize.input[1:]) != list(dequantize.input[1:]):
            raise ValueError(f"{site.site_id} QuantizeLinear/DequantizeLinear parameters differ")
        scale_name, zero_name = quantize.input[1:]
        if scale_name not in initializers or zero_name not in initializers:
            raise ValueError(f"{site.site_id} QDQ parameter initializer is missing")
        scale = numpy_helper.to_array(initializers[scale_name])
        zero_point = numpy_helper.to_array(initializers[zero_name])
        if scale.shape != () or scale.dtype != np.float32 or not float(scale) > 0.0:
            raise ValueError(f"{site.site_id} scale must be a positive float32 scalar")
        if zero_point.shape != () or zero_point.dtype != np.uint8:
            raise ValueError(f"{site.site_id} zero-point must be a uint8 scalar")
        target_parameters[site.site_id] = {
            "scale_initializer": scale_name,
            "zero_point_initializer": zero_name,
            "scale": float(scale),
            "zero_point": int(zero_point),
        }

    return {
        "source_node_count": len(source.graph.node),
        "rewritten_node_count": len(rewritten.graph.node),
        "topology_sha256": _topology_signature(source),
        "target_silu_count": len(rewritten_sites),
        "target_qdq_pair_count": len(rewritten_sites),
        "added_node_count": 0,
        "custom_runtime_nodes_added": 0,
        "piecewise_runtime_dispatch": False,
        "runtime_parameters_per_site": {"scale": 1, "zero_point": 1},
        "target_parameters": target_parameters,
    }


def rewrite_standard_qdq_parameters(
    model: onnx.ModelProto,
    site_specs: Mapping[str, StandardQDQSpec],
    *,
    calibration_digest: str | None = None,
) -> StandardQDQRewriteResult:
    """Replace only the 17 post-SiLU standard QDQ scalar parameter pairs."""

    if any(item.key == METADATA_KEY for item in model.metadata_props):
        raise ValueError("model already contains v1.8 hardware-aware QDQ metadata")
    sites = discover_qdq_silu_sites(model)
    _validate_site_parameters(sites, site_specs)
    rewritten = copy.deepcopy(model)
    parameter_names: dict[str, dict[str, str]] = {}

    for site in sites:
        quantize = rewritten.graph.node[site.output_quantize_node_index]
        dequantize = rewritten.graph.node[site.output_dequantize_node_index]
        if len(quantize.input) != 3 or len(dequantize.input) != 3:
            raise ValueError(f"{site.site_id} is not an asymmetric scalar QDQ pair")
        prefix = f"silu_v18_{_safe_site_name(site.site_id)}"
        scale_name = f"{prefix}_scale"
        zero_name = f"{prefix}_zero_point"
        spec = site_specs[site.site_id]
        quantize.input[1] = scale_name
        quantize.input[2] = zero_name
        dequantize.input[1] = scale_name
        dequantize.input[2] = zero_name
        rewritten.graph.initializer.extend(
            [
                numpy_helper.from_array(np.asarray(spec.scale, dtype=np.float32), scale_name),
                numpy_helper.from_array(np.asarray(spec.zero_point, dtype=np.uint8), zero_name),
            ]
        )
        parameter_names[site.site_id] = {
            "scale": scale_name,
            "zero_point": zero_name,
        }

    referenced = {name for node in rewritten.graph.node for name in node.input}
    retained = [item for item in rewritten.graph.initializer if item.name in referenced]
    del rewritten.graph.initializer[:]
    rewritten.graph.initializer.extend(retained)

    metadata = {
        "schema_version": REWRITE_SCHEMA,
        "calibration_digest": calibration_digest,
        "target_site_ids": [site.site_id for site in sites],
        "selected_qdq": {
            site.site_id: site_specs[site.site_id].to_manifest() for site in sites
        },
        "runtime_contract": {
            "quantize_op": "QuantizeLinear",
            "dequantize_op": "DequantizeLinear",
            "custom_runtime_nodes_added": 0,
            "piecewise_runtime_dispatch": False,
        },
    }
    rewritten.metadata_props.add(
        key=METADATA_KEY,
        value=json.dumps(metadata, sort_keys=True, separators=(",", ":")),
    )
    onnx.checker.check_model(rewritten)
    contract = validate_standard_qdq_rewrite(model, rewritten)
    return StandardQDQRewriteResult(
        model=rewritten,
        sites=sites,
        parameter_names=parameter_names,
        contract=contract,
    )
