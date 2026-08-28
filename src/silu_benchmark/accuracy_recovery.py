"""v1.1 QDQ-aware SiLU controls, diagnosis, and candidate selection."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import onnx
from onnx import helper

from silu_benchmark.backends.onnx_piecewise import build_piecewise_qdq_subgraph
from silu_benchmark.quantization.spec import PiecewiseQuantizationSpec


RECOVERY_METADATA_KEY = "silu_benchmark.v11_accuracy_recovery"
RECOVERY_CONFIG_SCHEMA = "silu-accuracy-recovery-config/v1"
CANDIDATE_MANIFEST_SCHEMA = "silu-accuracy-candidate-manifest/v1"
EXPECTED_SITE_COUNT = 17


def _piecewise_quantize_numpy(values, spec: PiecewiseQuantizationSpec):
    """Torch-free NumPy expression of the frozen activation.py contract."""
    array = np.asarray(values)
    clipped = np.clip(array.astype(np.float64), spec.vmin, spec.vmax)
    lower = clipped < spec.vsplit
    codes = np.empty(clipped.shape, dtype=np.int64)
    codes[lower] = np.clip(
        np.rint(clipped[lower] / spec.lower_scale + spec.lower_zero_point),
        *spec.lower_codes,
    )
    codes[~lower] = np.clip(
        np.rint(clipped[~lower] / spec.upper_scale + spec.upper_zero_point),
        *spec.upper_codes,
    )
    return codes


def _piecewise_dequantize_numpy(codes, spec: PiecewiseQuantizationSpec):
    integer_codes = np.asarray(codes, dtype=np.int64)
    lower = integer_codes <= spec.lower_codes[1]
    values = np.empty(integer_codes.shape, dtype=np.float64)
    values[lower] = (integer_codes[lower] - spec.lower_zero_point) * spec.lower_scale
    values[~lower] = (integer_codes[~lower] - spec.upper_zero_point) * spec.upper_scale
    return np.clip(values, spec.vmin, spec.vmax)


@dataclass(frozen=True)
class QDQSiLUSite:
    site_id: str
    module_path: str
    call_index: int
    input_tensor: str
    sigmoid_output_tensor: str
    sigmoid_qdq_output_tensor: str
    mul_output_tensor: str
    output_tensor: str
    sigmoid_node_name: str
    mul_node_name: str
    sigmoid_node_index: int
    sigmoid_quantize_node_index: int
    sigmoid_dequantize_node_index: int
    mul_node_index: int
    output_quantize_node_index: int
    output_dequantize_node_index: int

    @property
    def replaced_node_indices(self) -> tuple[int, ...]:
        return (
            self.sigmoid_node_index,
            self.sigmoid_quantize_node_index,
            self.sigmoid_dequantize_node_index,
            self.mul_node_index,
            self.output_quantize_node_index,
            self.output_dequantize_node_index,
        )


@dataclass(frozen=True)
class RecoveryRewriteResult:
    model: onnx.ModelProto
    sites: tuple[QDQSiLUSite, ...]
    tensor_mappings: Mapping[str, Mapping[str, str]]
    mode: str


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_hash(payload: Mapping) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def _site_identity(node_name: str) -> tuple[str, int, str]:
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
        module_path = ".".join(path)
    return module_path, call_index, f"{module_path}.call_{call_index}"


def discover_qdq_silu_sites(model: onnx.ModelProto) -> tuple[QDQSiLUSite, ...]:
    """Prove the exact six-node QDQ-wrapped SiLU topology at all 17 sites."""
    consumers: dict[str, list[tuple[int, onnx.NodeProto]]] = {}
    for index, node in enumerate(model.graph.node):
        for tensor in node.input:
            consumers.setdefault(tensor, []).append((index, node))

    def only_consumer(tensor: str, op_type: str, context: str):
        matches = consumers.get(tensor, [])
        if len(matches) != 1 or matches[0][1].op_type != op_type:
            raise ValueError(
                f"{context} must have exactly one {op_type} consumer; observed="
                f"{[(index, node.op_type) for index, node in matches]}"
            )
        return matches[0]

    sites = []
    for sigmoid_index, sigmoid in enumerate(model.graph.node):
        if sigmoid.op_type != "Sigmoid":
            continue
        if len(sigmoid.input) != 1 or len(sigmoid.output) != 1:
            raise ValueError(f"invalid Sigmoid arity at node {sigmoid.name}")
        sigmoid_q_index, sigmoid_q = only_consumer(
            sigmoid.output[0], "QuantizeLinear", sigmoid.name
        )
        sigmoid_dq_index, sigmoid_dq = only_consumer(
            sigmoid_q.output[0], "DequantizeLinear", sigmoid_q.name
        )
        mul_index, mul = only_consumer(sigmoid_dq.output[0], "Mul", sigmoid_dq.name)
        if len(mul.input) != 2 or sigmoid.input[0] not in mul.input:
            raise ValueError(f"Mul does not reuse the proven SiLU source: {mul.name}")
        output_q_index, output_q = only_consumer(mul.output[0], "QuantizeLinear", mul.name)
        output_dq_index, output_dq = only_consumer(
            output_q.output[0], "DequantizeLinear", output_q.name
        )
        if not consumers.get(output_dq.output[0]):
            raise ValueError(f"SiLU output has no downstream consumer: {output_dq.name}")
        module_path, call_index, site_id = _site_identity(mul.name)
        sites.append(
            QDQSiLUSite(
                site_id=site_id,
                module_path=module_path,
                call_index=call_index,
                input_tensor=sigmoid.input[0],
                sigmoid_output_tensor=sigmoid.output[0],
                sigmoid_qdq_output_tensor=sigmoid_dq.output[0],
                mul_output_tensor=mul.output[0],
                output_tensor=output_dq.output[0],
                sigmoid_node_name=sigmoid.name,
                mul_node_name=mul.name,
                sigmoid_node_index=sigmoid_index,
                sigmoid_quantize_node_index=sigmoid_q_index,
                sigmoid_dequantize_node_index=sigmoid_dq_index,
                mul_node_index=mul_index,
                output_quantize_node_index=output_q_index,
                output_dequantize_node_index=output_dq_index,
            )
        )
    identities = [site.site_id for site in sites]
    if len(sites) != EXPECTED_SITE_COUNT:
        raise ValueError(f"expected exactly 17 QDQ SiLU sites, observed {len(sites)}")
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate QDQ SiLU site identity")
    replaced = [index for site in sites for index in site.replaced_node_indices]
    if len(replaced) != len(set(replaced)):
        raise ValueError("QDQ SiLU sites overlap in node ownership")
    return tuple(sites)


def _add_metadata(model: onnx.ModelProto, payload: Mapping) -> None:
    if any(item.key == RECOVERY_METADATA_KEY for item in model.metadata_props):
        raise ValueError("model already contains v1.1 accuracy-recovery metadata")
    model.metadata_props.add(
        key=RECOVERY_METADATA_KEY,
        value=json.dumps(payload, sort_keys=True, separators=(",", ":")),
    )


def build_noop_control(model: onnx.ModelProto) -> RecoveryRewriteResult:
    """Validate and provenance-stamp a byte-semantics-preserving QDQ copy."""
    sites = discover_qdq_silu_sites(model)
    copied = copy.deepcopy(model)
    mappings = {
        site.site_id: {
            "pre_silu": site.input_tensor,
            "post_silu": site.mul_output_tensor,
            "activation_output": site.output_tensor,
        }
        for site in sites
    }
    _add_metadata(
        copied,
        {
            "schema_version": "silu-accuracy-rewrite/v1",
            "mode": "topology_preserving_noop",
            "sites": [asdict(site) for site in sites],
            "tensor_mappings": mappings,
        },
    )
    onnx.checker.check_model(copied)
    return RecoveryRewriteResult(copied, sites, mappings, "topology_preserving_noop")


def rewrite_qdq_silu(
    model: onnx.ModelProto,
    *,
    mode: str,
    site_specs: Mapping[str, PiecewiseQuantizationSpec] | None = None,
    selected_site_ids: Sequence[str] | None = None,
    candidate_id: str | None = None,
) -> RecoveryRewriteResult:
    """Replace proven QDQ SiLUs with exact-float or canonical piecewise outputs."""
    if mode not in {"float_equivalent", "piecewise"}:
        raise ValueError("mode must be float_equivalent or piecewise")
    sites = discover_qdq_silu_sites(model)
    selected = set(selected_site_ids or [site.site_id for site in sites])
    known = {site.site_id for site in sites}
    missing = sorted(selected - known)
    if missing:
        raise ValueError(f"selected sites are not present in the graph: {missing}")
    if mode == "piecewise":
        if site_specs is None:
            raise ValueError("piecewise rewrite requires explicit site_specs")
        absent = sorted(selected - set(site_specs))
        unused = sorted(set(site_specs) - selected)
        invalid = sorted(
            key for key, value in site_specs.items()
            if not isinstance(value, PiecewiseQuantizationSpec)
        )
        if absent or unused or invalid:
            raise ValueError(
                f"piecewise site spec mismatch: missing={absent}, unused={unused}, invalid={invalid}"
            )
    by_start = {site.sigmoid_node_index: site for site in sites if site.site_id in selected}
    removed = {
        index
        for site in sites
        if site.site_id in selected
        for index in site.replaced_node_indices
    }
    nodes: list[onnx.NodeProto] = []
    initializers = [copy.deepcopy(item) for item in model.graph.initializer]
    mappings: dict[str, dict[str, str]] = {}
    for index, node in enumerate(model.graph.node):
        if index in removed and index not in by_start:
            continue
        if index not in by_start:
            nodes.append(copy.deepcopy(node))
            continue
        site = by_start[index]
        prefix = "silu_v11_" + re.sub(r"[^A-Za-z0-9_]", "_", site.site_id)
        sigmoid_output = f"{prefix}_sigmoid"
        post_silu = f"{prefix}_post_silu"
        nodes.append(
            helper.make_node(
                "Sigmoid", [site.input_tensor], [sigmoid_output],
                name=f"{prefix}_sigmoid_node",
            )
        )
        if mode == "float_equivalent":
            nodes.append(
                helper.make_node(
                    "Mul", [site.input_tensor, sigmoid_output], [site.output_tensor],
                    name=f"{prefix}_mul_node",
                )
            )
            post_tensor = site.output_tensor
        else:
            nodes.append(
                helper.make_node(
                    "Mul", [site.input_tensor, sigmoid_output], [post_silu],
                    name=f"{prefix}_mul_node",
                )
            )
            piecewise_nodes, piecewise_initializers, _outputs = build_piecewise_qdq_subgraph(
                spec=site_specs[site.site_id],
                input_name=post_silu,
                prefix=prefix,
                dequantized_output_name=site.output_tensor,
            )
            nodes.extend(piecewise_nodes)
            initializers.extend(piecewise_initializers)
            post_tensor = post_silu
        mappings[site.site_id] = {
            "pre_silu": site.input_tensor,
            "post_silu": post_tensor,
            "activation_output": site.output_tensor,
        }
    for site in sites:
        if site.site_id not in selected:
            mappings[site.site_id] = {
                "pre_silu": site.input_tensor,
                "post_silu": site.mul_output_tensor,
                "activation_output": site.output_tensor,
            }

    referenced_initializers = {tensor for node in nodes for tensor in node.input}
    initializers = [
        item for item in initializers if item.name in referenced_initializers
    ]

    graph = helper.make_graph(
        nodes,
        f"{model.graph.name}_v11_{mode}",
        [copy.deepcopy(item) for item in model.graph.input],
        [copy.deepcopy(item) for item in model.graph.output],
        initializer=initializers,
        value_info=[copy.deepcopy(item) for item in model.graph.value_info],
    )
    rewritten = copy.deepcopy(model)
    rewritten.graph.CopyFrom(graph)
    metadata = {
        "schema_version": "silu-accuracy-rewrite/v1",
        "mode": mode,
        "candidate_id": candidate_id,
        "selected_site_ids": [site.site_id for site in sites if site.site_id in selected],
        "sites": [asdict(site) for site in sites],
        "tensor_mappings": mappings,
    }
    if site_specs is not None:
        metadata["site_specs"] = {
            key: asdict(value) for key, value in sorted(site_specs.items())
        }
    _add_metadata(rewritten, metadata)
    try:
        rewritten = onnx.shape_inference.infer_shapes(rewritten)
    except (onnx.shape_inference.InferenceError, ValueError):
        pass
    onnx.checker.check_model(rewritten)
    return RecoveryRewriteResult(rewritten, sites, mappings, mode)


def instrument_outputs(model: onnx.ModelProto, mappings: Mapping[str, str]) -> onnx.ModelProto:
    if not mappings:
        raise ValueError("instrumentation mapping must not be empty")
    produced = {tensor for node in model.graph.node for tensor in node.output}
    if not set(mappings.values()).issubset(produced):
        missing = sorted(set(mappings.values()) - produced)
        raise ValueError(f"instrumentation tensors are not produced: {missing}")
    copied = copy.deepcopy(model)
    try:
        copied = onnx.shape_inference.infer_shapes(copied)
    except (onnx.shape_inference.InferenceError, ValueError):
        pass
    known = {
        item.name: item
        for item in [*copied.graph.input, *copied.graph.output, *copied.graph.value_info]
    }
    outputs = []
    for logical_name, tensor in mappings.items():
        value = known.get(tensor)
        if value is None:
            value = helper.make_tensor_value_info(tensor, onnx.TensorProto.FLOAT, [None] * 4)
        outputs.append(copy.deepcopy(value))
    del copied.graph.output[:]
    copied.graph.output.extend(outputs)
    onnx.checker.check_model(copied)
    return copied


def deterministic_development_split(
    *, total: int, calibration_indices: Sequence[int], sample_count: int, seed: int
) -> list[int]:
    calibration = {int(index) for index in calibration_indices}
    if any(index < 0 or index >= total for index in calibration):
        raise ValueError("calibration index is outside the training batch")
    available = np.asarray([index for index in range(total) if index not in calibration], dtype=np.int64)
    if sample_count <= 0 or sample_count > len(available):
        raise ValueError("invalid development sample count")
    generator = np.random.default_rng(seed)
    selected = generator.choice(available, size=sample_count, replace=False)
    return sorted(int(index) for index in selected)


def split_digest(images: np.ndarray, labels: np.ndarray, indices: Sequence[int], role: str) -> str:
    selected = np.asarray(indices, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(role.encode("utf-8"))
    digest.update(selected.tobytes())
    digest.update(np.ascontiguousarray(images[selected]).tobytes())
    digest.update(np.ascontiguousarray(labels[selected]).tobytes())
    return digest.hexdigest()


def summarize_values(values: np.ndarray) -> dict:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("statistics values must be non-empty and finite")
    percentiles = np.percentile(array, [0.01, 0.1, 1.0, 50.0, 99.0, 99.9, 99.99])
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "percentiles": {
            key: float(value)
            for key, value in zip(
                ("p0_01", "p0_1", "p1", "p50", "p99", "p99_9", "p99_99"),
                percentiles,
            )
        },
    }


def piecewise_error_summary(values: np.ndarray, spec: PiecewiseQuantizationSpec) -> dict:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    codes = np.asarray(_piecewise_quantize_numpy(array, spec), dtype=np.int64)
    reconstructed = np.asarray(_piecewise_dequantize_numpy(codes, spec), dtype=np.float64)
    error = reconstructed - array
    denominator = float(np.linalg.norm(array) * np.linalg.norm(reconstructed))
    occupancy = np.bincount(codes, minlength=256)
    return {
        "clipped_below_count": int(np.count_nonzero(array < spec.vmin)),
        "clipped_below_fraction": float(np.mean(array < spec.vmin)),
        "clipped_above_count": int(np.count_nonzero(array > spec.vmax)),
        "clipped_above_fraction": float(np.mean(array > spec.vmax)),
        "lower_segment_fraction": float(np.mean(np.clip(array, spec.vmin, spec.vmax) < spec.vsplit)),
        "upper_segment_fraction": float(np.mean(np.clip(array, spec.vmin, spec.vmax) >= spec.vsplit)),
        "occupied_code_count": int(np.count_nonzero(occupancy)),
        "code_occupancy": occupancy.astype(int).tolist(),
        "max_absolute_error": float(np.max(np.abs(error))),
        "mae": float(np.mean(np.abs(error))),
        "mse": float(np.mean(error * error)),
        "cosine_similarity": float(np.dot(array, reconstructed) / denominator) if denominator else None,
    }


def calibrate_candidate_spec(values: np.ndarray, candidate: Mapping) -> PiecewiseQuantizationSpec:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("candidate calibration values must be non-empty and finite")
    positive = array[array > 0.0]
    negative = array[array < 0.0]
    if positive.size == 0 or negative.size == 0:
        raise ValueError("candidate calibration requires negative and positive SiLU values")
    if candidate["vmax_strategy"] == "percentile":
        vmax = float(np.percentile(positive, float(candidate["vmax_percentile"])))
    elif candidate["vmax_strategy"] == "observed_max":
        vmax = float(positive.max())
    else:
        raise ValueError("unsupported vmax_strategy")
    vmin = float(negative.min())
    count = int(candidate["vsplit_candidate_count"])
    splits = np.linspace(0.01 * vmax, 0.80 * vmax, count)
    best_spec = None
    best_mse = math.inf
    for split in splits:
        spec = PiecewiseQuantizationSpec(vmin, float(split), vmax, int(candidate["bits"]))
        reconstructed = np.asarray(
            _piecewise_dequantize_numpy(_piecewise_quantize_numpy(array, spec), spec)
        )
        mse = float(np.mean((array - reconstructed) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_spec = spec
    if best_spec is None:
        raise RuntimeError("candidate split search produced no valid spec")
    return best_spec


def build_candidate_manifest(
    *,
    candidate: Mapping,
    site_values: Mapping[str, np.ndarray],
    sites: Sequence[QDQSiLUSite],
    provenance: Mapping,
) -> tuple[dict, dict[str, PiecewiseQuantizationSpec]]:
    expected = [site.site_id for site in sites]
    if set(site_values) != set(expected):
        raise ValueError("candidate calibration values do not match discovered sites")
    configuration_hash = canonical_hash(candidate)
    specs = {}
    entries = []
    for site in sites:
        values = np.asarray(site_values[site.site_id])
        spec = calibrate_candidate_spec(values, candidate)
        specs[site.site_id] = spec
        entries.append(
            {
                "site_id": site.site_id,
                "module_path": site.module_path,
                "call_index": site.call_index,
                "parameters": {
                    **asdict(spec),
                    "lower_scale": spec.lower_scale,
                    "lower_zero_point": spec.lower_zero_point,
                    "upper_scale": spec.upper_scale,
                    "upper_zero_point": spec.upper_zero_point,
                    "lower_codes": list(spec.lower_codes),
                    "upper_codes": list(spec.upper_codes),
                },
                "calibration_statistics": summarize_values(values),
                "quantization_error": piecewise_error_summary(values, spec),
                "node_provenance": asdict(site),
            }
        )
    manifest = {
        "schema_version": CANDIDATE_MANIFEST_SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "configuration": dict(candidate),
        "configuration_hash": configuration_hash,
        "provenance": dict(provenance),
        "target_node_count": len(entries),
        "sites": entries,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    validate_candidate_manifest(manifest, expected)
    return manifest, specs


def validate_candidate_manifest(payload: Mapping, expected_site_ids: Sequence[str]) -> None:
    if payload.get("schema_version") != CANDIDATE_MANIFEST_SCHEMA:
        raise ValueError("invalid candidate manifest schema")
    sites = payload.get("sites")
    if not isinstance(sites, list) or len(sites) != EXPECTED_SITE_COUNT:
        raise ValueError("candidate manifest must contain exactly 17 sites")
    identities = [entry.get("site_id") for entry in sites]
    if identities != list(expected_site_ids) or len(set(identities)) != len(identities):
        raise ValueError("candidate manifest sites are missing, duplicated, or reordered")
    expected_configuration_hash = canonical_hash(payload["configuration"])
    if payload.get("configuration_hash") != expected_configuration_hash:
        raise ValueError("candidate configuration hash mismatch")
    without_hash = dict(payload)
    observed_hash = without_hash.pop("manifest_hash", None)
    if observed_hash != canonical_hash(without_hash):
        raise ValueError("candidate manifest hash mismatch")
    for entry in sites:
        parameters = entry["parameters"]
        spec = PiecewiseQuantizationSpec(
            parameters["vmin"], parameters["vsplit"], parameters["vmax"], parameters["bits"]
        )
        if not np.isclose(parameters["lower_scale"], spec.lower_scale, atol=1e-12, rtol=0):
            raise ValueError(f"derived lower_scale mismatch for {entry['site_id']}")
        if not np.isclose(parameters["upper_scale"], spec.upper_scale, atol=1e-12, rtol=0):
            raise ValueError(f"derived upper_scale mismatch for {entry['site_id']}")


def specs_from_candidate_manifest(payload: Mapping) -> dict[str, PiecewiseQuantizationSpec]:
    identities = [entry["site_id"] for entry in payload["sites"]]
    validate_candidate_manifest(payload, identities)
    return {
        entry["site_id"]: PiecewiseQuantizationSpec(**{
            key: entry["parameters"][key] for key in ("vmin", "vsplit", "vmax", "bits")
        })
        for entry in payload["sites"]
    }


def rank_sensitivity(rows: Sequence[Mapping]) -> list[dict]:
    required = {"site_id", "development_accuracy", "logit_mae"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError("sensitivity row is missing required metrics")
    if len({row["site_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate sensitivity site")
    return [
        dict(row)
        for row in sorted(
            rows,
            key=lambda row: (float(row["development_accuracy"]), -float(row["logit_mae"]), row["site_id"]),
        )
    ]


def select_candidate(rows: Sequence[Mapping], *, split_role: str) -> dict:
    if split_role != "development":
        raise ValueError("candidate selection may use only the development split")
    if not rows:
        raise ValueError("candidate result rows must not be empty")
    normalized = []
    for order, row in enumerate(rows):
        if row.get("split_role") != "development" or row.get("status") != "success":
            raise ValueError("all selectable candidate rows must be successful development results")
        if "final_test_accuracy" in row:
            raise ValueError("candidate selection must not receive final-test metrics")
        normalized.append({**dict(row), "configuration_order": row.get("configuration_order", order)})
    selected = max(
        normalized,
        key=lambda row: (
            float(row["accuracy"]),
            float(row["prediction_agreement_vs_standard_qdq"]),
            -int(row["configuration_order"]),
        ),
    )
    return dict(selected)


def validate_generated_output_path(path: Path, repository_root: Path) -> Path:
    root = repository_root.resolve()
    destination = path.resolve()
    allowed = (root / "results" / "benchmarks").resolve()
    try:
        relative = destination.relative_to(allowed)
    except ValueError as exc:
        raise ValueError("v1.1 output must stay under results/benchmarks") from exc
    if not relative.parts or not relative.parts[0].startswith("v1.1_accuracy_"):
        raise ValueError("v1.1 output directory must start with v1.1_accuracy_")
    return destination
