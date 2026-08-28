"""Pure contracts and decision helpers for the v1.5 ORT tail investigation."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


CONFIG_SCHEMA = "ort-customop-tail-latency-diagnosis/v1"
REPORT_SCHEMA = "ort-customop-tail-latency-report/v1"
BASELINE_ID = "baseline_v14"


def validate_output_path(path: Path, root: Path) -> Path:
    resolved = Path(path).resolve()
    allowed = (Path(root).resolve() / "results/benchmarks").resolve()
    try:
        relative = resolved.relative_to(allowed)
    except ValueError as error:
        raise ValueError("v1.5 output must stay under results/benchmarks") from error
    if not relative.parts or not relative.parts[0].startswith(
        "v1.5_ort_customop_tail_latency"
    ):
        raise ValueError("v1.5 output must use its isolated tail-latency directory")
    return resolved


def baseline_options() -> dict:
    return {
        "provider": "CPUExecutionProvider",
        "execution_mode": "ORT_SEQUENTIAL",
        "intra_op_num_threads": 4,
        "inter_op_num_threads": 0,
        "graph_optimization_level": "ORT_ENABLE_ALL",
        "enable_cpu_mem_arena": True,
        "enable_mem_pattern": True,
    }


def screening_candidates(config: Mapping) -> list[dict]:
    baseline = baseline_options()
    rows = [
        {
            "candidate_id": BASELINE_ID,
            "label": "frozen v1.4 baseline",
            "change_group": "baseline",
            "explanatory_control": False,
            "requested_options": baseline,
        }
    ]
    definitions = [
        ("cpu_arena_disabled", "CPU memory arena disabled", "cpu_memory_arena", {"enable_cpu_mem_arena": False}, False),
        ("memory_pattern_disabled", "memory pattern disabled", "memory_pattern", {"enable_mem_pattern": False}, False),
        ("graph_extended", "graph optimization extended", "graph_optimization", {"graph_optimization_level": "ORT_ENABLE_EXTENDED"}, False),
        ("graph_basic", "graph optimization basic", "graph_optimization", {"graph_optimization_level": "ORT_ENABLE_BASIC"}, False),
        (
            "execution_parallel_i2",
            "parallel execution with explicit inter-op 2",
            "execution_mode",
            {"execution_mode": "ORT_PARALLEL", "inter_op_num_threads": 2},
            False,
        ),
    ]
    if config.get("include_nearby_thread_controls", True):
        definitions.extend(
            (
                f"intra_threads_{threads}",
                f"nearby explanatory intra-op control: {threads} threads",
                "intra_op_threads",
                {"intra_op_num_threads": threads},
                True,
            )
            for threads in (3, 5, 6)
        )
    for candidate_id, label, group, overrides, explanatory in definitions:
        requested = {**baseline, **overrides}
        rows.append(
            {
                "candidate_id": candidate_id,
                "label": label,
                "change_group": group,
                "explanatory_control": explanatory,
                "requested_options": requested,
            }
        )
    validate_exactly_one_variable(rows)
    return rows


def validate_exactly_one_variable(candidates: Sequence[Mapping]) -> None:
    baseline = baseline_options()
    expected_ids = {
        BASELINE_ID,
        "cpu_arena_disabled",
        "memory_pattern_disabled",
        "graph_extended",
        "graph_basic",
        "execution_parallel_i2",
        "intra_threads_3",
        "intra_threads_5",
        "intra_threads_6",
    }
    ids = {row.get("candidate_id") for row in candidates}
    if ids != expected_ids:
        raise ValueError("v1.5 screening candidate set is incomplete or unbounded")
    for row in candidates:
        requested = row.get("requested_options", {})
        if requested.get("provider") != "CPUExecutionProvider":
            raise ValueError("all v1.5 candidates require CPUExecutionProvider")
        differences = {
            key for key, value in requested.items() if baseline.get(key) != value
        }
        if row["candidate_id"] == BASELINE_ID:
            if differences:
                raise ValueError("v1.5 baseline differs from frozen v1.4 options")
            continue
        allowed = {
            "cpu_memory_arena": {"enable_cpu_mem_arena"},
            "memory_pattern": {"enable_mem_pattern"},
            "graph_optimization": {"graph_optimization_level"},
            "execution_mode": {"execution_mode", "inter_op_num_threads"},
            "intra_op_threads": {"intra_op_num_threads"},
        }[row["change_group"]]
        if differences != allowed:
            raise ValueError(
                f"candidate {row['candidate_id']} is not exactly one logical variable: {sorted(differences)}"
            )
        if row["change_group"] != "intra_op_threads" and requested["intra_op_num_threads"] != 4:
            raise ValueError("non-thread screening candidates must retain four intra-op threads")


def validate_config(payload: Mapping) -> dict:
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("invalid v1.5 tail-latency schema")
    if payload.get("ort_version") != "1.19.2":
        raise ValueError("v1.5 is frozen to ORT 1.19.2")
    for key in (
        "reference_model",
        "custom_model",
        "custom_library",
        "data_root",
        "probe_indices",
        "historical_v14_report",
        "historical_v14_raw_timings",
        "output_root",
    ):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"missing v1.5 path setting: {key}")
    for key in (
        "reference_model_sha256",
        "custom_model_sha256",
        "custom_library_sha256",
    ):
        value = payload.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"invalid frozen SHA-256: {key}")
    if payload.get("exactness") != {"samples": 128, "batch_size": 8}:
        raise ValueError("every v1.5 candidate requires the 128-image exactness gate")
    baseline = payload.get("baseline_benchmark", {})
    if baseline != {
        "batch_size": 1,
        "warmup_iterations": 20,
        "timed_iterations": 100,
        "repetitions": 10,
    }:
        raise ValueError("v1.5 baseline protocol must be batch1/20/100/10")
    screening = payload.get("screening_benchmark", {})
    if screening != {
        "batch_size": 1,
        "warmup_iterations": 20,
        "timed_iterations": 100,
        "repetitions": 5,
    }:
        raise ValueError("v1.5 screening protocol must be batch1/20/100/5")
    finalist = payload.get("finalist_confirmation", {})
    if finalist != {
        "warmup_iterations": 20,
        "timed_iterations": 100,
        "repetitions_per_arm": 10,
    }:
        raise ValueError("v1.5 finalist protocol must be 10 alternating repetitions per arm")
    if payload.get("profiling") != {"iterations": 10, "baseline_profiles": 2}:
        raise ValueError("v1.5 requires two independent 10-inference baseline profiles")
    candidates = screening_candidates(payload)
    serialized = json.dumps(payload).lower()
    for prohibited in ("openvino", "cuda", "qnn", "npu", "avx2"):
        if prohibited in serialized:
            raise ValueError(f"prohibited implementation/backend in v1.5 config: {prohibited}")
    return {**dict(payload), "candidate_count": len(candidates)}


def load_config(path: Path) -> dict:
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


def candidate_fingerprint(candidate: Mapping) -> str:
    payload = {
        "candidate_id": candidate["candidate_id"],
        "change_group": candidate["change_group"],
        "explanatory_control": bool(candidate["explanatory_control"]),
        "requested_options": candidate["requested_options"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def tail_statistics(raw_repetitions: Sequence[Sequence[float]]) -> dict:
    if not raw_repetitions or any(not row for row in raw_repetitions):
        raise ValueError("tail statistics require non-empty repetitions")
    per_repetition = []
    flattened = []
    for samples in raw_repetitions:
        values = np.asarray(samples, dtype=np.float64)
        flattened.extend(float(value) for value in values)
        mean = float(np.mean(values))
        std = float(np.std(values))
        per_repetition.append(
            {
                "sample_count": len(values),
                "mean_ms": mean,
                "p50_ms": float(np.percentile(values, 50)),
                "p95_ms": float(np.percentile(values, 95)),
                "p99_ms": float(np.percentile(values, 99)),
                "min_ms": float(np.min(values)),
                "max_ms": float(np.max(values)),
                "std_ms": std,
                "coefficient_of_variation": std / mean if mean else 0.0,
                "images_per_second": 1000.0 / mean if mean else 0.0,
                "raw_samples_retained": True,
                "outliers_deleted": False,
            }
        )
    values = np.asarray(flattened, dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values))
    return {
        "sample_count": len(values),
        "repetition_count": len(raw_repetitions),
        "mean_ms": mean,
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "std_ms": std,
        "coefficient_of_variation": std / mean if mean else 0.0,
        "images_per_second": 1000.0 / mean if mean else 0.0,
        "median_of_repetition_p50_ms": float(
            statistics.median(row["p50_ms"] for row in per_repetition)
        ),
        "median_of_repetition_p95_ms": float(
            statistics.median(row["p95_ms"] for row in per_repetition)
        ),
        "median_of_repetition_p99_ms": float(
            statistics.median(row["p99_ms"] for row in per_repetition)
        ),
        "repetition_p50_range_ms": [
            min(row["p50_ms"] for row in per_repetition),
            max(row["p50_ms"] for row in per_repetition),
        ],
        "repetition_p95_range_ms": [
            min(row["p95_ms"] for row in per_repetition),
            max(row["p95_ms"] for row in per_repetition),
        ],
        "repetition_p99_range_ms": [
            min(row["p99_ms"] for row in per_repetition),
            max(row["p99_ms"] for row in per_repetition),
        ],
        "per_repetition": per_repetition,
        "raw_samples_retained": True,
        "outliers_deleted": False,
    }


def alternating_schedule(candidate_id: str, repetitions_per_arm: int = 10) -> list[str]:
    if not candidate_id or candidate_id == BASELINE_ID:
        raise ValueError("finalist schedule requires a non-baseline candidate")
    if repetitions_per_arm != 10:
        raise ValueError("v1.5 finalist schedule is frozen to 10 repetitions per arm")
    return [value for _ in range(repetitions_per_arm) for value in (BASELINE_ID, candidate_id)]


def candidate_qualifies(baseline: Mapping, candidate: Mapping) -> dict:
    if candidate.get("status") != "complete" or not candidate.get("exactness", {}).get("passed"):
        return {"qualifies": False, "reasons": ["candidate did not complete exact timing parity"]}
    base = baseline["statistics"]
    tested = candidate["statistics"]
    checks = {
        "aggregate_p50_not_worse": tested["p50_ms"] <= base["p50_ms"],
        "median_repetition_p50_not_worse": (
            tested["median_of_repetition_p50_ms"]
            <= base["median_of_repetition_p50_ms"]
        ),
        "median_repetition_p95_not_worse": (
            tested["median_of_repetition_p95_ms"]
            <= base["median_of_repetition_p95_ms"]
        ),
        "coefficient_of_variation_not_worse": (
            tested["coefficient_of_variation"]
            <= base["coefficient_of_variation"]
        ),
    }
    return {
        "qualifies": all(checks.values()),
        "checks": checks,
        "reasons": [key for key, passed in checks.items() if not passed],
    }


def select_finalist(baseline: Mapping, candidates: Sequence[Mapping]) -> dict:
    eligible = []
    evaluations = []
    for candidate in candidates:
        if candidate.get("candidate_id") == BASELINE_ID:
            continue
        result = candidate_qualifies(baseline, candidate)
        evaluations.append({"candidate_id": candidate.get("candidate_id"), **result})
        if result["qualifies"]:
            eligible.append(candidate)
    selected = min(
        eligible,
        key=lambda row: (
            row["statistics"]["median_of_repetition_p95_ms"],
            row["statistics"]["median_of_repetition_p50_ms"],
            row["statistics"]["coefficient_of_variation"],
        ),
        default=None,
    )
    return {
        "selected_candidate_id": selected["candidate_id"] if selected else None,
        "selection_rule": (
            "exactness plus non-worse aggregate/repetition p50, median repetition p95, "
            "and retained-sample CV; rank by median repetition p95"
        ),
        "evaluations": evaluations,
    }


def tail_reproducibility(current: Mapping, historical: Mapping) -> dict:
    current_max_rep_p95 = current["repetition_p95_range_ms"][1]
    historical_max_rep_p95 = historical["repetition_p95_range_ms"][1]
    current_cv = current["coefficient_of_variation"]
    historical_cv = historical["coefficient_of_variation"]
    if current_max_rep_p95 >= historical_max_rep_p95 and current_cv >= historical_cv:
        conclusion = "yes"
    elif current_max_rep_p95 < historical["p95_ms"] and current_cv < historical_cv:
        conclusion = "no"
    else:
        conclusion = "mixed"
    return {
        "conclusion": conclusion,
        "criterion": (
            "yes when current maximum repetition p95 and aggregate CV both meet or exceed v1.4; "
            "no when current maximum repetition p95 is below v1.4 aggregate p95 and CV is lower; otherwise mixed"
        ),
        "current_max_repetition_p95_ms": current_max_rep_p95,
        "historical_max_repetition_p95_ms": historical_max_rep_p95,
        "current_coefficient_of_variation": current_cv,
        "historical_coefficient_of_variation": historical_cv,
    }


def compare_profiles(profiles: Sequence[Mapping]) -> dict:
    if len(profiles) != 2:
        raise ValueError("v1.5 requires exactly two independent baseline profiles")
    rows = []
    for profile in profiles:
        analysis = profile["analysis"]
        categories = {row["name"]: row for row in analysis["categories"]}
        rows.append(
            {
                "profile_id": profile["profile_id"],
                "wall_ms": profile["total_inference_wall_ms"],
                "node_sum_ms": analysis["summed_node_event_time_us"] / 1000.0,
                "custom_nodes": analysis["custom_op_execution"]["unique_node_count"],
                "convolution_share_percent": categories.get("convolution", {}).get("percentage_of_node_event_time", 0.0),
                "quantization_related_share_percent": categories.get("quantization_related", {}).get("percentage_of_node_event_time", 0.0),
                "custom_activation_share_percent": categories.get("custom_activation", {}).get("percentage_of_node_event_time", 0.0),
                "dominant_category": analysis["categories"][0]["name"] if analysis["categories"] else None,
            }
        )
    custom_delta = abs(
        rows[0]["custom_activation_share_percent"]
        - rows[1]["custom_activation_share_percent"]
    )
    return {
        "profiles": rows,
        "both_all_17_nodes": all(row["custom_nodes"] == 17 for row in rows),
        "convolution_dominant_in_both": all(
            row["dominant_category"] == "convolution" for row in rows
        ),
        "custom_share_absolute_difference_pp": custom_delta,
        "material_custom_share_difference": custom_delta >= 5.0,
        "non_additivity_warning": (
            "Summed ORT node-event durations are not a wall-clock decomposition; "
            "scheduling and overlap can make them differ from measured wall time."
        ),
    }
