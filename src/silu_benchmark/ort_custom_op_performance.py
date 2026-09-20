"""Pure analysis helpers for the v1.3 ORT custom-op CPU diagnosis."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


CONFIG_SCHEMA = "ort-customop-performance-diagnosis/v1"
EXPECTED_SITE_COUNT = 17
AVX2_PRIORITY_SHARE_PERCENT = 10.0
NON_ADDITIVITY_WARNING = (
    "Summed ORT node-event durations are not a wall-clock decomposition: "
    "scheduler overhead and overlapping or parallel execution can make node sums differ from wall time."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output_path(path: Path, root: Path) -> Path:
    resolved = Path(path).resolve()
    allowed = (Path(root).resolve() / "results/benchmarks").resolve()
    try:
        relative = resolved.relative_to(allowed)
    except ValueError as error:
        raise ValueError("v1.3 output must stay under results/benchmarks") from error
    if not relative.parts or not relative.parts[0].startswith(
        "v1.3_ort_customop_performance_diagnosis"
    ):
        raise ValueError("v1.3 output must use its isolated diagnosis directory")
    return resolved


def validate_config(payload: Mapping) -> dict:
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("invalid v1.3 performance-diagnosis schema")
    if payload.get("provider") != "CPUExecutionProvider":
        raise ValueError("v1.3 diagnosis permits CPUExecutionProvider only")
    if payload.get("ort_version") != "1.19.2":
        raise ValueError("v1.3 is frozen to ONNX Runtime 1.19.2")
    for key in (
        "reference_model",
        "custom_model",
        "custom_library",
        "data_root",
        "probe_indices",
        "output_root",
    ):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"missing v1.3 path setting: {key}")
    for key in ("reference_model_sha256", "custom_model_sha256", "custom_library_sha256"):
        value = payload.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"invalid frozen SHA-256: {key}")
    exactness = payload.get("exactness", {})
    if exactness.get("samples") != 128 or exactness.get("batch_size") != 8:
        raise ValueError("v1.3 exactness gate must use 128 samples in batches of 8")
    profiling = payload.get("profiling", {})
    if not isinstance(profiling.get("iterations"), int) or profiling["iterations"] <= 0:
        raise ValueError("profiling iterations must be positive")
    benchmark = payload.get("benchmark", {})
    if benchmark.get("warmup_iterations") != 20:
        raise ValueError("every v1.3 benchmark repetition requires 20 warmups")
    if benchmark.get("timed_iterations") != 100:
        raise ValueError("every v1.3 benchmark repetition requires 100 timed invocations")
    if not isinstance(benchmark.get("repetitions"), int) or benchmark["repetitions"] < 5:
        raise ValueError("v1.3 requires at least five independent repetitions")
    if benchmark.get("batch_sizes") != [1, 8, 32]:
        raise ValueError("v1.3 batch controls must be exactly [1, 8, 32]")
    if benchmark.get("intra_op_threads") != [0, 1, 2, 4, 8]:
        raise ValueError("v1.3 thread controls must be default/1/2/4/8")
    if benchmark.get("execution_modes") != ["sequential", "parallel"]:
        raise ValueError("v1.3 execution controls must be sequential and parallel")
    if benchmark.get("inter_op_threads") != 0:
        raise ValueError("v1.3 keeps inter-op threads fixed at documented default 0")
    serialized = json.dumps(payload).lower()
    for prohibited in ("openvino", "cuda", "qnn", "npu", "avx2"):
        if prohibited in serialized:
            raise ValueError(f"prohibited implementation/backend appears in v1.3 config: {prohibited}")
    return dict(payload)


def load_config(path: Path) -> dict:
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


def benchmark_cells(config: Mapping) -> list[dict]:
    benchmark = config["benchmark"]
    cells = {}
    for graph in ("custom_op", "standard_operator"):
        for threads in benchmark["intra_op_threads"]:
            key = (graph, 1, threads, "sequential")
            cells[key] = {
                "graph": graph,
                "batch_size": 1,
                "intra_op_threads": threads,
                "execution_mode": "sequential",
                "purpose": "batch1_thread_sweep",
            }
        for batch_size in benchmark["batch_sizes"]:
            key = (graph, batch_size, 0, "sequential")
            cells.setdefault(
                key,
                {
                    "graph": graph,
                    "batch_size": batch_size,
                    "intra_op_threads": 0,
                    "execution_mode": "sequential",
                    "purpose": "matched_batch_sweep",
                },
            )
        key = (graph, 1, 0, "parallel")
        cells[key] = {
            "graph": graph,
            "batch_size": 1,
            "intra_op_threads": 0,
            "execution_mode": "parallel",
            "purpose": "execution_mode_control",
        }
    return [cells[key] for key in sorted(cells)]


def sample_statistics(samples_ms: Sequence[float], *, batch_size: int) -> dict:
    values = np.asarray(samples_ms, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("timing samples must be a non-empty finite vector")
    if np.any(values < 0):
        raise ValueError("timing samples must be non-negative")
    mean_ms = float(np.mean(values))
    std_ms = float(np.std(values))
    return {
        "sample_count": int(values.size),
        "mean_ms": mean_ms,
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "std_ms": std_ms,
        "coefficient_of_variation": std_ms / mean_ms if mean_ms else 0.0,
        "images_per_second": batch_size * 1000.0 / mean_ms if mean_ms else math.inf,
        "total_measured_wall_time_seconds": float(np.sum(values) / 1000.0),
        "raw_samples_retained": True,
        "outliers_deleted": False,
    }


def summarize_repetitions(raw_repetitions: Sequence[Sequence[float]], *, batch_size: int) -> dict:
    if len(raw_repetitions) < 5:
        raise ValueError("at least five raw benchmark repetitions are required")
    per_repetition = [sample_statistics(row, batch_size=batch_size) for row in raw_repetitions]
    flattened = [value for row in raw_repetitions for value in row]
    aggregate = sample_statistics(flattened, batch_size=batch_size)
    repetition_p50 = [row["p50_ms"] for row in per_repetition]
    repetition_p95 = [row["p95_ms"] for row in per_repetition]
    aggregate.update(
        {
            "repetition_count": len(raw_repetitions),
            "per_repetition": per_repetition,
            "median_of_repetition_p50_ms": float(statistics.median(repetition_p50)),
            "median_of_repetition_p95_ms": float(statistics.median(repetition_p95)),
            "repetition_p50_range_ms": [min(repetition_p50), max(repetition_p50)],
            "repetition_p95_range_ms": [min(repetition_p95), max(repetition_p95)],
            "high_p95_reproduced_in_all_repetitions": all(
                row["p95_ms"] > row["p50_ms"] * 1.25 for row in per_repetition
            ),
        }
    )
    return aggregate


def classify_comparability(left: Mapping, right: Mapping) -> dict:
    fields = (
        "provider",
        "batch_size",
        "warmup_iterations",
        "timed_iterations",
        "repetitions",
        "input_digest",
        "input_reuse",
        "graph_optimization_level",
        "execution_mode",
        "intra_op_threads",
        "inter_op_threads",
        "profiling_enabled",
        "timing_boundary",
        "output_selection",
        "session_lifecycle",
        "between_invocation_work",
    )
    differences = [
        {"field": field, "left": left.get(field), "right": right.get(field)}
        for field in fields
        if left.get(field) != right.get(field)
    ]
    return {
        "directly_protocol_comparable": not differences,
        "differences": differences,
        "classification": "matched_protocol" if not differences else "not_directly_comparable",
    }


def _event_node_name(event: Mapping) -> str:
    args = event.get("args") or {}
    explicit = args.get("node_name")
    if explicit:
        return str(explicit)
    name = str(event.get("name") or "")
    for suffix in ("_kernel_time", "_fence_before", "_fence_after"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def classify_operation(node_name: str, op_name: str) -> str:
    lower_node = node_name.lower()
    lower_op = op_name.lower()
    if lower_op == "quantizedpiecewisesilu":
        return "custom_activation"
    if lower_op in {"conv", "fusedconv"}:
        return "convolution"
    if lower_op in {"gemm", "matmul", "fusedgemm"}:
        return "gemm_matmul"
    if lower_op in {"quantizelinear", "dequantizelinear", "dynamicquantizelinear"}:
        return "quantization_related"
    if lower_node.startswith("silu_v11_"):
        return "quantization_related"
    if lower_op in {
        "memcpyfromhost",
        "memcpytohost",
        "copy",
        "transpose",
        "reshape",
        "flatten",
        "gather",
        "unsqueeze",
        "concat",
    }:
        return "data_movement_copy"
    return "other"


def parse_profile_events(events: Sequence[Mapping], *, inference_wall_time_us: float) -> dict:
    node_rows = defaultdict(lambda: {"durations_us": [], "op_name": "", "category": ""})
    op_durations = defaultdict(list)
    category_durations = defaultdict(list)
    event_names = set()
    event_categories = set()
    skipped = 0
    for event in events:
        event_names.add(str(event.get("name") or ""))
        event_categories.add(str(event.get("cat") or ""))
        args = event.get("args") or {}
        event_name = str(event.get("name") or "")
        op_name = str(args.get("op_name") or args.get("op_type") or "")
        duration = event.get("dur")
        if not event_name.endswith("_kernel_time") or not op_name or duration is None:
            skipped += 1
            continue
        try:
            duration_us = float(duration)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if not math.isfinite(duration_us) or duration_us < 0:
            skipped += 1
            continue
        node_name = _event_node_name(event)
        category = classify_operation(node_name, op_name)
        row = node_rows[node_name]
        row["op_name"] = op_name
        row["category"] = category
        row["durations_us"].append(duration_us)
        op_durations[op_name].append(duration_us)
        category_durations[category].append(duration_us)

    node_event_time_us = sum(sum(row["durations_us"]) for row in node_rows.values())

    def aggregate(name: str, values: Sequence[float], **extra) -> dict:
        return {
            **extra,
            "name": name,
            "execution_count": len(values),
            "total_duration_us": float(sum(values)),
            "mean_duration_us": float(statistics.fmean(values)),
            "median_duration_us": float(statistics.median(values)),
            "percentage_of_node_event_time": (
                100.0 * sum(values) / node_event_time_us if node_event_time_us else 0.0
            ),
        }

    nodes = [
        aggregate(
            name,
            row["durations_us"],
            op_name=row["op_name"],
            category=row["category"],
        )
        for name, row in node_rows.items()
    ]
    nodes.sort(key=lambda row: (-row["total_duration_us"], row["name"]))
    operations = [aggregate(name, values) for name, values in op_durations.items()]
    operations.sort(key=lambda row: (-row["total_duration_us"], row["name"]))
    categories = [aggregate(name, values) for name, values in category_durations.items()]
    categories.sort(key=lambda row: (-row["total_duration_us"], row["name"]))
    custom_nodes = [row for row in nodes if row["op_name"] == "QuantizedPiecewiseSiLU"]
    return {
        "inference_wall_time_us": float(inference_wall_time_us),
        "summed_node_event_time_us": float(node_event_time_us),
        "node_sum_to_wall_ratio": (
            node_event_time_us / inference_wall_time_us if inference_wall_time_us else None
        ),
        "non_additivity_warning": NON_ADDITIVITY_WARNING,
        "parsed_kernel_event_count": sum(row["execution_count"] for row in nodes),
        "skipped_event_count": skipped,
        "profile_event_names_observed": sorted(event_names),
        "profile_event_categories_observed": sorted(event_categories),
        "nodes": nodes,
        "top_20_nodes": nodes[:20],
        "operations": operations,
        "categories": categories,
        "custom_op_execution": {
            "unique_node_count": len(custom_nodes),
            "aggregate_duration_us": float(
                sum(row["total_duration_us"] for row in custom_nodes)
            ),
            "percentage_of_node_event_time": (
                100.0
                * sum(row["total_duration_us"] for row in custom_nodes)
                / node_event_time_us
                if node_event_time_us
                else 0.0
            ),
            "nodes": custom_nodes,
        },
    }


def parse_profile_file(path: Path, *, inference_wall_time_us: float) -> dict:
    return parse_profile_events(
        json.loads(Path(path).read_text(encoding="utf-8")),
        inference_wall_time_us=inference_wall_time_us,
    )


def decide_optimization(profile: Mapping, benchmark_cells_payload: Sequence[Mapping]) -> dict:
    custom_share = float(
        profile["custom_op_execution"]["percentage_of_node_event_time"]
    )
    avx2 = {
        "decision": (
            "not recommended"
            if custom_share < AVX2_PRIORITY_SHARE_PERCENT
            else "inconclusive"
        ),
        "custom_node_share_percent": custom_share,
        "reason": (
            "Custom activations are below the 10% diagnostic priority boundary; vectorizing "
            "them cannot address the measured dominant model cost."
            if custom_share < AVX2_PRIORITY_SHARE_PERCENT
            else "Custom-node share is material, but v1.3 contains no vectorized implementation evidence."
        ),
    }
    batch1_custom = [
        row
        for row in benchmark_cells_payload
        if row["graph"] == "custom_op"
        and row["batch_size"] == 1
        and row["execution_mode"] == "sequential"
    ]
    default = next((row for row in batch1_custom if row["intra_op_threads"] == 0), None)
    candidates = [row for row in batch1_custom if row["intra_op_threads"] != 0]
    stable_candidates = []
    if default:
        default_medians = [
            row["p50_ms"] for row in default["statistics"]["per_repetition"]
        ]
        for candidate in candidates:
            candidate_medians = [
                row["p50_ms"]
                for row in candidate["statistics"]["per_repetition"]
            ]
            wins = sum(
                tested < baseline
                for baseline, tested in zip(default_medians, candidate_medians)
            )
            if (
                statistics.median(candidate_medians)
                < statistics.median(default_medians)
                and wins >= 4
                and candidate["statistics"]["coefficient_of_variation"]
                <= default["statistics"]["coefficient_of_variation"]
            ):
                stable_candidates.append(candidate)
    best = min(
        stable_candidates,
        key=lambda row: row["statistics"]["median_of_repetition_p50_ms"],
        default=None,
    )
    stable = best is not None
    threading = {
        "decision": (
            f"recommended configuration: intra_op_num_threads={best['intra_op_threads']}"
            if stable
            else "no stable improvement"
        ),
        "reason": (
            "This was the lowest-median tested configuration that improved at least four of five "
            "repetition medians without increasing aggregate variance."
            if stable
            else "No tested setting improved at least four of five repetition medians while preserving or reducing aggregate variance."
        ),
    }
    categories = profile.get("categories", [])
    dominant = next(
        (row for row in categories if row["name"] != "custom_activation"),
        None,
    )
    graph_runtime = {
        "decision": "recommended target" if dominant else "inconclusive",
        "target": dominant["name"] if dominant else None,
        "reason": (
            f"{dominant['name']} is the largest measured non-custom node-event category "
            f"at {dominant['percentage_of_node_event_time']:.2f}%."
            if dominant
            else "No non-custom dominant category was measurable."
        ),
    }
    return {
        "avx2_kernel": avx2,
        "thread_tuning": threading,
        "graph_runtime_work": graph_runtime,
        "further_work": {
            "decision": (
                "Run one isolated batch-1 profile with intra_op_num_threads=4 to verify "
                f"whether {dominant['name']} remains dominant under the recommended session setting."
                if dominant and custom_share < AVX2_PRIORITY_SHARE_PERCENT
                else "stop here"
            )
        },
    }
