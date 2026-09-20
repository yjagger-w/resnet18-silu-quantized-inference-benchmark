"""Discovery, rewrite, and inspection utilities for the full ONNX model."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from silu_benchmark.quantization import PiecewiseQuantizationSpec

from .onnx_piecewise import build_piecewise_qdq_subgraph


_REWRITE_METADATA_KEY = "silu_benchmark.piecewise_rewrite_sites"


@dataclass(frozen=True)
class SiLUPatternSite:
    """A proven ``Mul(x, Sigmoid(x))`` realization of one SiLU call site."""

    site_id: str
    module_path: str
    call_index: int
    input_tensor: str
    output_tensor: str
    sigmoid_node_name: str
    mul_node_name: str
    sigmoid_node_index: int
    mul_node_index: int


@dataclass(frozen=True)
class RewriteResult:
    model: onnx.ModelProto
    sites: tuple[SiLUPatternSite, ...]
    inserted_outputs: Mapping[str, Mapping[str, str]]


def _site_identity(node_name: str, ordinal: int) -> tuple[str, int, str]:
    """Recover a stable module-oriented identity from this exporter naming."""
    path = node_name.strip("/").split("/")
    if path and path[-1] == "Mul":
        path.pop()
    leaf = path[-1] if path else "silu"
    match = re.fullmatch(r"(act)(?:_(\d+))?", leaf)
    call_index = int(match.group(2) or 0) if match else 0
    leaf_base = match.group(1) if match else leaf
    if len(path) >= 2 and re.fullmatch(r"layer[1-4]\.\d+", path[-2]):
        module_path = f"{path[-2]}.{leaf_base}"
    elif leaf_base == "act":
        module_path = "act"
    else:
        module_path = ".".join(path) or f"silu_{ordinal}"
    return module_path, call_index, f"{module_path}.call_{call_index}"


def discover_silu_patterns(model: onnx.ModelProto) -> tuple[SiLUPatternSite, ...]:
    """Return only exact, single-use ``Mul(x, Sigmoid(x))`` SiLU patterns."""
    consumers: dict[str, list[tuple[int, onnx.NodeProto]]] = {}
    for index, node in enumerate(model.graph.node):
        for name in node.input:
            consumers.setdefault(name, []).append((index, node))

    sites = []
    for sigmoid_index, sigmoid in enumerate(model.graph.node):
        if sigmoid.name.startswith("silu_piecewise_"):
            continue
        if sigmoid.op_type != "Sigmoid" or len(sigmoid.input) != 1 or len(sigmoid.output) != 1:
            continue
        source = sigmoid.input[0]
        sigmoid_output = sigmoid.output[0]
        users = consumers.get(sigmoid_output, [])
        matches = [
            (index, node)
            for index, node in users
            if node.op_type == "Mul"
            and len(node.input) == 2
            and sigmoid_output in node.input
            and source in node.input
        ]
        if len(users) != 1 or len(matches) != 1:
            continue
        mul_index, mul = matches[0]
        if len(mul.output) != 1:
            continue
        module_path, call_index, site_id = _site_identity(mul.name, len(sites))
        sites.append(
            SiLUPatternSite(
                site_id=site_id,
                module_path=module_path,
                call_index=call_index,
                input_tensor=source,
                output_tensor=mul.output[0],
                sigmoid_node_name=sigmoid.name,
                mul_node_name=mul.name,
                sigmoid_node_index=sigmoid_index,
                mul_node_index=mul_index,
            )
        )
    ids = [site.site_id for site in sites]
    if len(ids) != len(set(ids)):
        raise ValueError("exported SiLU pattern names do not produce unique site identifiers")
    return tuple(sites)


def bind_module_specs_to_sites(
    sites: Sequence[SiLUPatternSite],
    module_specs: Mapping[str, PiecewiseQuantizationSpec],
) -> dict[str, PiecewiseQuantizationSpec]:
    """Expand explicit module calibration into an explicit exported-site map."""
    required_modules = {site.module_path for site in sites}
    missing = sorted(required_modules - set(module_specs))
    unused = sorted(set(module_specs) - required_modules)
    if missing or unused:
        raise ValueError(f"module spec mapping mismatch: missing={missing}, unused={unused}")
    return {site.site_id: module_specs[site.module_path] for site in sites}


def _validate_site_specs(
    sites: Sequence[SiLUPatternSite],
    site_specs: Mapping[str, PiecewiseQuantizationSpec],
) -> None:
    expected = {site.site_id for site in sites}
    supplied = set(site_specs)
    missing = sorted(expected - supplied)
    unused = sorted(supplied - expected)
    invalid = sorted(key for key, value in site_specs.items() if not isinstance(value, PiecewiseQuantizationSpec))
    if missing or unused or invalid:
        raise ValueError(
            f"site spec mapping mismatch: missing={missing}, unused={unused}, invalid={invalid}"
        )


def load_site_spec_manifest(path: Union[str, Path]) -> dict[str, PiecewiseQuantizationSpec]:
    """Read only the versioned v0.6 calibration manifest contract."""
    from silu_benchmark.calibration_manifest import load_manifest

    _payload, site_specs = load_manifest(Path(path))
    return site_specs


def _rewrite_metadata(
    sites: Sequence[SiLUPatternSite],
    site_specs: Mapping[str, PiecewiseQuantizationSpec],
    inserted_outputs: Mapping[str, Mapping[str, str]],
) -> str:
    return json.dumps(
        [
            {
                **asdict(site),
                "spec": asdict(site_specs[site.site_id]),
                **inserted_outputs[site.site_id],
            }
            for site in sites
        ],
        sort_keys=True,
    )


def rewrite_silu_piecewise_model(
    model: onnx.ModelProto,
    site_specs: Mapping[str, PiecewiseQuantizationSpec],
) -> RewriteResult:
    """Replace all proven SiLU patterns with the checked piecewise subgraph."""
    if any(item.key == _REWRITE_METADATA_KEY for item in model.metadata_props):
        raise ValueError("model is already a SiLU piecewise rewrite")
    sites = discover_silu_patterns(model)
    if not sites:
        raise ValueError("no proven SiLU Sigmoid/Mul patterns were found")
    _validate_site_specs(sites, site_specs)

    by_sigmoid = {site.sigmoid_node_index: site for site in sites}
    by_mul = {site.mul_node_index: site for site in sites}
    nodes: list[onnx.NodeProto] = []
    initializers = [copy.deepcopy(item) for item in model.graph.initializer]
    inserted_outputs: dict[str, dict[str, str]] = {}

    for index, node in enumerate(model.graph.node):
        if index in by_sigmoid:
            continue
        if index not in by_mul:
            nodes.append(copy.deepcopy(node))
            continue
        site = by_mul[index]
        prefix = "silu_piecewise_" + re.sub(r"[^A-Za-z0-9_]", "_", site.site_id)
        sigmoid_output = f"{prefix}_silu_sigmoid"
        silu_output = f"{prefix}_silu_output"
        # Recreate the proven SiLU expression under the inserted prefix, then
        # quantize its output.  Quantizing the pre-SiLU convolution/addition
        # input would be a different algorithm.
        nodes.extend(
            [
                helper.make_node("Sigmoid", [site.input_tensor], [sigmoid_output], name=f"{prefix}_silu_sigmoid_node"),
                helper.make_node("Mul", [site.input_tensor, sigmoid_output], [silu_output], name=f"{prefix}_silu_mul_node"),
            ]
        )
        subgraph_nodes, subgraph_initializers, outputs = build_piecewise_qdq_subgraph(
            spec=site_specs[site.site_id],
            input_name=silu_output,
            prefix=prefix,
            dequantized_output_name=site.output_tensor,
        )
        nodes.extend(subgraph_nodes)
        initializers.extend(subgraph_initializers)
        inserted_outputs[site.site_id] = {
            "quantized_codes_tensor": outputs.quantized_codes,
            "dequantized_output_tensor": outputs.dequantized_output,
        }

    graph = helper.make_graph(
        nodes=nodes,
        name=f"{model.graph.name}_silu_piecewise",
        inputs=[copy.deepcopy(item) for item in model.graph.input],
        outputs=[copy.deepcopy(item) for item in model.graph.output],
        initializer=initializers,
        value_info=[copy.deepcopy(item) for item in model.graph.value_info],
    )
    rewritten = copy.deepcopy(model)
    rewritten.graph.CopyFrom(graph)
    rewritten.metadata_props.add(
        key=_REWRITE_METADATA_KEY,
        value=_rewrite_metadata(sites, site_specs, inserted_outputs),
    )
    try:
        rewritten = onnx.shape_inference.infer_shapes(rewritten)
    except (onnx.shape_inference.InferenceError, ValueError):
        pass
    onnx.checker.check_model(rewritten)
    return RewriteResult(rewritten, tuple(sites), inserted_outputs)


def save_rewrite_result(result: RewriteResult, path: Union[str, Path]) -> None:
    """Validate and save a rewritten model without changing the baseline model."""
    onnx.checker.check_model(result.model)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(result.model, str(destination))


def _shape(value_info: onnx.ValueInfoProto) -> list[Union[int, str, None]]:
    return [
        dimension.dim_value if dimension.HasField("dim_value") else dimension.dim_param or None
        for dimension in value_info.type.tensor_type.shape.dim
    ]


def _dtype_name(elem_type: int) -> str:
    return TensorProto.DataType.Name(elem_type)


def _smoke_inputs(model: onnx.ModelProto, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    element_types = {TensorProto.FLOAT: np.float32, TensorProto.DOUBLE: np.float64}
    result = {}
    for value_info in model.graph.input:
        elem_type = value_info.type.tensor_type.elem_type
        dtype = element_types.get(elem_type)
        if dtype is None:
            raise ValueError(f"smoke input does not support {_dtype_name(elem_type)}")
        dimensions = [dimension if isinstance(dimension, int) and dimension > 0 else 1 for dimension in _shape(value_info)]
        result[value_info.name] = rng.standard_normal(dimensions).astype(dtype)
    return result


def inspect_onnx_model(
    model_or_path: Union[onnx.ModelProto, str, Path],
    smoke_seed: int = 1234,
) -> dict[str, Any]:
    """Produce a JSON-serializable graph and CPU-runtime capability report."""
    model_path: Optional[Path] = None
    if isinstance(model_or_path, onnx.ModelProto):
        model = model_or_path
        serialized = model.SerializeToString()
    else:
        model_path = Path(model_or_path)
        serialized = model_path.read_bytes()
        model = onnx.load_from_string(serialized)
    histogram: dict[str, int] = {}
    for node in model.graph.node:
        histogram[node.op_type] = histogram.get(node.op_type, 0) + 1
    metadata = next((item.value for item in model.metadata_props if item.key == _REWRITE_METADATA_KEY), "[]")
    try:
        inserted_sites = json.loads(metadata)
    except json.JSONDecodeError:
        inserted_sites = []
    try:
        onnx.checker.check_model(model)
        checker = {"passed": True}
    except onnx.checker.ValidationError as exc:
        checker = {"passed": False, "error": str(exc)}

    runtime: dict[str, Any] = {"available_providers": ort.get_available_providers()}
    try:
        session = ort.InferenceSession(serialized, providers=["CPUExecutionProvider"])
        outputs = session.run(None, _smoke_inputs(model, smoke_seed))
        runtime.update(
            {
                "cpu_execution_provider": True,
                "session_providers": session.get_providers(),
                "run_passed": True,
                "output_shapes": [list(output.shape) for output in outputs],
            }
        )
    except Exception as exc:  # report a capability failure without hiding it
        runtime.update({"cpu_execution_provider": False, "run_passed": False, "error": str(exc)})

    portability = []
    if any(item.data_type == TensorProto.DOUBLE for item in model.graph.initializer):
        portability.append("FLOAT64 arithmetic is retained for reference fidelity.")
    if "Round" in histogram:
        portability.append("Round-based piecewise arithmetic requires backend operator support.")
    return {
        "model_path": str(model_path.resolve()) if model_path else None,
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "ir_version": model.ir_version,
        "opset": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
        "inputs": [{"name": item.name, "dtype": _dtype_name(item.type.tensor_type.elem_type), "shape": _shape(item)} for item in model.graph.input],
        "outputs": [{"name": item.name, "dtype": _dtype_name(item.type.tensor_type.elem_type), "shape": _shape(item)} for item in model.graph.output],
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "operator_histogram": dict(sorted(histogram.items())),
        "conv_count": histogram.get("Conv", 0),
        "baseline_silu_pattern_count": len(discover_silu_patterns(model)),
        "inserted_piecewise_subgraph_count": len(inserted_sites),
        "replacement_sites": inserted_sites,
        "original_silu_pattern_remains": bool(discover_silu_patterns(model)),
        "quantize_linear_present": "QuantizeLinear" in histogram,
        "dequantize_linear_present": "DequantizeLinear" in histogram,
        "checker": checker,
        "runtime": runtime,
        "portability_considerations": portability,
        "unsupported_or_unverified_claims": [
            "No OpenVINO, QNN, GPU, NPU, DSP, or INT8-kernel support was tested.",
            "This functional reference graph is not a production INT8 acceleration claim.",
        ],
    }


def write_inspection_report(report: Mapping[str, Any], path: Union[str, Path]) -> None:
    """Persist a deterministic, machine-readable inspection report."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
