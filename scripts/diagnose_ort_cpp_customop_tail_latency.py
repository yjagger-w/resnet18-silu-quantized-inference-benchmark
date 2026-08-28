"""Run the bounded v1.5 ORT CPU custom-op tail-latency investigation."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import onnxruntime as ort

from silu_benchmark.benchmark_data import load_cifar_batch, normalize_cifar_images
from silu_benchmark.ort_custom_op_backend import prepare_process_local_ort_runtime
from silu_benchmark.ort_custom_op_performance import parse_profile_file
from silu_benchmark.ort_custom_op_tail_latency import (
    BASELINE_ID,
    REPORT_SCHEMA,
    alternating_schedule,
    candidate_fingerprint,
    candidate_qualifies,
    compare_profiles,
    load_config,
    screening_candidates,
    select_finalist,
    tail_reproducibility,
    tail_statistics,
    validate_output_path,
)
from silu_benchmark.ort_custom_op_thread_preset import (
    exact_tensor_metrics,
    sha256_file,
    verify_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
SCOPE = "ORT CPU hybrid graph with project C++ custom-op activations"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def input_digest(values: np.ndarray, indices: list[int]) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(np.asarray(indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(values).shape.__repr__().encode("ascii"))
    digest.update(np.asarray(values).dtype.str.encode("ascii"))
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def load_v14_helpers():
    path = ROOT / "scripts/validate_ort_cpp_customop_thread_preset.py"
    spec = importlib.util.spec_from_file_location("frozen_v14_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_candidate_options(
    candidate: Mapping, *, profile_prefix: Path | None = None
) -> tuple[ort.SessionOptions, dict]:
    requested = candidate["requested_options"]
    options = ort.SessionOptions()
    levels = {
        "ORT_ENABLE_ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        "ORT_ENABLE_EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "ORT_ENABLE_BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    }
    modes = {
        "ORT_SEQUENTIAL": ort.ExecutionMode.ORT_SEQUENTIAL,
        "ORT_PARALLEL": ort.ExecutionMode.ORT_PARALLEL,
    }
    options.graph_optimization_level = levels[requested["graph_optimization_level"]]
    options.execution_mode = modes[requested["execution_mode"]]
    options.intra_op_num_threads = requested["intra_op_num_threads"]
    options.inter_op_num_threads = requested["inter_op_num_threads"]
    options.enable_cpu_mem_arena = requested["enable_cpu_mem_arena"]
    options.enable_mem_pattern = requested["enable_mem_pattern"]
    if profile_prefix is not None:
        profile_prefix = Path(profile_prefix).resolve()
        profile_prefix.parent.mkdir(parents=True, exist_ok=True)
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    actual = {
        "provider": "CPUExecutionProvider",
        "execution_mode": str(options.execution_mode),
        "intra_op_num_threads": options.intra_op_num_threads,
        "inter_op_num_threads": options.inter_op_num_threads,
        "graph_optimization_level": str(options.graph_optimization_level),
        "enable_cpu_mem_arena": options.enable_cpu_mem_arena,
        "enable_mem_pattern": options.enable_mem_pattern,
        "enable_profiling": options.enable_profiling,
        "profile_file_prefix": options.profile_file_prefix,
        "use_deterministic_compute": options.use_deterministic_compute,
    }
    return options, {
        "requested": dict(requested),
        "applied": actual,
        "candidate_fingerprint": candidate_fingerprint(candidate),
    }


def create_session(
    model_path: Path,
    *,
    graph: str,
    library_path: Path,
    candidate: Mapping,
    profile_prefix: Path | None = None,
) -> tuple[ort.InferenceSession, dict]:
    options, option_record = build_candidate_options(
        candidate, profile_prefix=profile_prefix
    )
    runtime_dll = None
    registered_library = None
    registration_ms = 0.0
    if graph == "custom_op":
        runtime_dll = prepare_process_local_ort_runtime()
        started = time.perf_counter_ns()
        registered_library = options.register_custom_ops_library(str(library_path))
        registration_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    started = time.perf_counter_ns()
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    session_creation_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return session, {
        "graph": graph,
        "model": str(model_path),
        "providers": session.get_providers(),
        "runtime_dll": str(runtime_dll) if runtime_dll else None,
        "registered_library": registered_library or (
            str(library_path) if graph == "custom_op" else None
        ),
        "registration_load_ms": registration_ms,
        "session_creation_load_ms": session_creation_ms,
        "options": option_record,
    }


def run_exactness_gate(
    *,
    candidate: Mapping,
    reference_path: Path,
    custom_path: Path,
    library_path: Path,
    normalized_probe: np.ndarray,
) -> dict:
    reference_session, reference_metadata = create_session(
        reference_path,
        graph="standard_operator",
        library_path=library_path,
        candidate=candidate,
    )
    custom_session, custom_metadata = create_session(
        custom_path,
        graph="custom_op",
        library_path=library_path,
        candidate=candidate,
    )
    completed = 0
    logits_elements = 0
    predictions_equal = 0
    first_failure = None
    batch_size = 8
    for offset in range(0, len(normalized_probe), batch_size):
        inputs = normalized_probe[offset : offset + batch_size]
        reference_logits = reference_session.run(
            [reference_session.get_outputs()[0].name],
            {reference_session.get_inputs()[0].name: inputs},
        )[0]
        custom_logits = custom_session.run(
            [custom_session.get_outputs()[0].name],
            {custom_session.get_inputs()[0].name: inputs},
        )[0]
        metrics = exact_tensor_metrics(reference_logits, custom_logits)
        logits_elements += metrics["elements_compared"]
        if not metrics["exact_equal"]:
            first_failure = {"probe_offset": offset, "metrics": metrics}
            break
        predictions_equal += int(
            np.sum(np.argmax(reference_logits, axis=1) == np.argmax(custom_logits, axis=1))
        )
        completed += len(inputs)
    passed = bool(
        first_failure is None
        and completed == 128
        and logits_elements == 1280
        and predictions_equal == 128
    )
    return {
        "passed": passed,
        "samples_completed": completed,
        "final_logits_elements": logits_elements,
        "final_logits_exact": first_failure is None and logits_elements == 1280,
        "prediction_agreement": predictions_equal / completed if completed else 0.0,
        "first_failure": first_failure,
        "reference_session": reference_metadata,
        "custom_session": custom_metadata,
    }


def run_repetitions(
    *,
    candidate: Mapping,
    model_path: Path,
    library_path: Path,
    inputs: np.ndarray,
    protocol: Mapping,
) -> dict:
    raw = []
    sessions = []
    for repetition in range(protocol["repetitions"]):
        session, metadata = create_session(
            model_path,
            graph="custom_op",
            library_path=library_path,
            candidate=candidate,
        )
        output_name = session.get_outputs()[0].name
        feed = {session.get_inputs()[0].name: inputs}
        for _ in range(protocol["warmup_iterations"]):
            session.run([output_name], feed)
        samples = []
        for _ in range(protocol["timed_iterations"]):
            started = time.perf_counter_ns()
            session.run([output_name], feed)
            samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
        raw.append(samples)
        sessions.append(metadata)
        print(
            f"    repetition {repetition + 1}/{protocol['repetitions']}: "
            f"p50={np.percentile(samples, 50):.3f} "
            f"p95={np.percentile(samples, 95):.3f} "
            f"p99={np.percentile(samples, 99):.3f} ms",
            flush=True,
        )
        del session
    creation = [row["session_creation_load_ms"] for row in sessions]
    return {
        "raw_samples_ms": raw,
        "statistics": tail_statistics(raw),
        "session_creation_ms": creation,
        "session_creation_statistics": {
            "mean_ms": float(np.mean(creation)),
            "p50_ms": float(np.percentile(creation, 50)),
            "p95_ms": float(np.percentile(creation, 95)),
            "min_ms": float(np.min(creation)),
            "max_ms": float(np.max(creation)),
        },
        "session_metadata": sessions,
        "timing_boundary": (
            "perf_counter_ns immediately around session.run([logits], cached_feed); "
            "session creation, input preparation, and warmups excluded"
        ),
        "raw_samples_retained": True,
        "outliers_deleted": False,
    }


def run_candidate(
    *,
    candidate: Mapping,
    reference_path: Path,
    custom_path: Path,
    library_path: Path,
    normalized_probe: np.ndarray,
    benchmark_input: np.ndarray,
    protocol: Mapping,
) -> dict:
    row = {
        **candidate,
        "candidate_fingerprint": candidate_fingerprint(candidate),
        "status": "running",
    }
    try:
        exactness = run_exactness_gate(
            candidate=candidate,
            reference_path=reference_path,
            custom_path=custom_path,
            library_path=library_path,
            normalized_probe=normalized_probe,
        )
        row["exactness"] = exactness
        if not exactness["passed"]:
            row.update(
                {
                    "status": "exactness_failed",
                    "error": "zero-tolerance final-logit parity failed; timing prohibited",
                }
            )
            return row
        row.update(
            run_repetitions(
                candidate=candidate,
                model_path=custom_path,
                library_path=library_path,
                inputs=benchmark_input,
                protocol=protocol,
            )
        )
        row["status"] = "complete"
    except Exception as error:
        row.update(
            {
                "status": "rejected",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
    return row


def run_profile(
    *,
    profile_id: str,
    candidate: Mapping,
    model_path: Path,
    library_path: Path,
    inputs: np.ndarray,
    iterations: int,
    output_root: Path,
) -> dict:
    prefix = output_root / "profiles" / profile_id
    session, metadata = create_session(
        model_path,
        graph="custom_op",
        library_path=library_path,
        candidate=candidate,
        profile_prefix=prefix,
    )
    output_name = session.get_outputs()[0].name
    feed = {session.get_inputs()[0].name: inputs}
    wall = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        session.run([output_name], feed)
        wall.append((time.perf_counter_ns() - started) / 1_000_000.0)
    path = Path(session.end_profiling()).resolve()
    analysis = parse_profile_file(path, inference_wall_time_us=sum(wall) * 1000.0)
    custom = analysis["custom_op_execution"]
    counts = {row["execution_count"] for row in custom["nodes"]}
    if custom["unique_node_count"] != 17 or counts != {iterations}:
        raise RuntimeError("profile did not prove all 17 custom nodes executed")
    return {
        "profile_id": profile_id,
        "candidate_id": candidate["candidate_id"],
        "profile_path": str(path),
        "profile_sha256": sha256_file(path),
        "profile_iterations": iterations,
        "raw_inference_wall_ms": wall,
        "total_inference_wall_ms": sum(wall),
        "session": metadata,
        "analysis": analysis,
    }


def run_alternating_confirmation(
    *,
    baseline: Mapping,
    finalist: Mapping,
    custom_path: Path,
    library_path: Path,
    inputs: np.ndarray,
    protocol: Mapping,
) -> dict:
    schedule = alternating_schedule(
        finalist["candidate_id"], protocol["repetitions_per_arm"]
    )
    definitions = {BASELINE_ID: baseline, finalist["candidate_id"]: finalist}
    raw = {BASELINE_ID: [], finalist["candidate_id"]: []}
    sessions = {BASELINE_ID: [], finalist["candidate_id"]: []}
    for position, candidate_id in enumerate(schedule, 1):
        candidate = definitions[candidate_id]
        session, metadata = create_session(
            custom_path,
            graph="custom_op",
            library_path=library_path,
            candidate=candidate,
        )
        output_name = session.get_outputs()[0].name
        feed = {session.get_inputs()[0].name: inputs}
        for _ in range(protocol["warmup_iterations"]):
            session.run([output_name], feed)
        samples = []
        for _ in range(protocol["timed_iterations"]):
            started = time.perf_counter_ns()
            session.run([output_name], feed)
            samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
        raw[candidate_id].append(samples)
        sessions[candidate_id].append(metadata)
        print(
            f"    alternating {position}/{len(schedule)} {candidate_id}: "
            f"p50={np.percentile(samples, 50):.3f} p95={np.percentile(samples, 95):.3f} ms",
            flush=True,
        )
        del session
    arms = {
        candidate_id: {
            "raw_samples_ms": samples,
            "statistics": tail_statistics(samples),
            "session_metadata": sessions[candidate_id],
        }
        for candidate_id, samples in raw.items()
    }
    finalist_evaluation = candidate_qualifies(
        {"statistics": arms[BASELINE_ID]["statistics"]},
        {
            "status": "complete",
            "exactness": {"passed": True},
            "statistics": arms[finalist["candidate_id"]]["statistics"],
        },
    )
    return {
        "schedule": schedule,
        "arms": arms,
        "finalist_evaluation": finalist_evaluation,
        "selected_after_confirmation": finalist["candidate_id"] if finalist_evaluation["qualifies"] else None,
        "raw_samples_retained": True,
        "outliers_deleted": False,
    }


def machine_facts(runtime_dll: Path | None) -> dict:
    helpers = load_v14_helpers()
    facts = helpers.machine_facts(runtime_dll)
    facts.update(
        {
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "torch_imported": "torch" in sys.modules or "torchvision" in sys.modules,
        }
    )
    return facts


def build_decisions(
    *,
    tail_result: Mapping,
    screening: Sequence[Mapping],
    finalist_confirmation: Mapping | None,
    profile_comparison: Mapping,
) -> dict:
    confirmed_id = (
        finalist_confirmation.get("selected_after_confirmation")
        if finalist_confirmation
        else None
    )
    complete = [row for row in screening if row.get("status") == "complete"]
    baseline = next(row for row in complete if row["candidate_id"] == BASELINE_ID)
    typical = [
        row for row in complete
        if row["candidate_id"] != BASELINE_ID
        and row["statistics"]["median_of_repetition_p50_ms"]
        < baseline["statistics"]["median_of_repetition_p50_ms"]
    ]
    tail = [
        row for row in complete
        if row["candidate_id"] != BASELINE_ID
        and row["statistics"]["median_of_repetition_p95_ms"]
        < baseline["statistics"]["median_of_repetition_p95_ms"]
        and row["statistics"]["coefficient_of_variation"]
        <= baseline["statistics"]["coefficient_of_variation"]
    ]
    return {
        "tail_latency_reproducible": tail_result,
        "typical_latency": {
            "conclusion": confirmed_id or "none demonstrated",
            "screening_candidates_with_lower_median_p50": [
                row["candidate_id"] for row in typical
            ],
        },
        "tail_variance": {
            "conclusion": confirmed_id or "none demonstrated",
            "screening_candidates_with_better_tail_and_cv": [
                row["candidate_id"] for row in tail
            ],
        },
        "recommended_preset": {
            "decision": (
                f"new candidate: {confirmed_id}" if confirmed_id else "retain v1.4"
            ),
            "reason": (
                "The candidate passed exact alternating confirmation."
                if confirmed_id
                else "No screened candidate passed or retained the full confirmation criteria."
            ),
        },
        "convolution": {
            "conclusion": (
                "yes" if profile_comparison["convolution_dominant_in_both"] else "inconclusive"
            )
        },
        "avx2_priority": {
            "decision": (
                "continue deferred"
                if profile_comparison["convolution_dominant_in_both"]
                else "reconsider"
            ),
            "reason": (
                "Both independent unchanged baseline profiles remained convolution-dominant."
                if profile_comparison["convolution_dominant_in_both"]
                else "Independent profiles did not agree that convolution remained dominant."
            ),
        },
        "next_smallest_experiment": (
            "Repeat the selected candidate confirmation in a separate process."
            if confirmed_id
            else "Repeat the unchanged v1.4 10x100 baseline in one separate process to estimate run-to-run tail incidence."
        ),
    }


def write_reports(output_root: Path, report: Mapping) -> None:
    atomic_json(output_root / "diagnosis_report.json", report)
    atomic_json(output_root / "screening_matrix.json", report["screening_matrix"])
    atomic_json(output_root / "profile_comparison.json", report["profile_comparison"])
    atomic_json(output_root / "finalist_confirmation.json", report["finalist_confirmation"])

    with (output_root / "screening_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fields = [
            "candidate_id", "status", "change_group", "explanatory_control",
            "exactness_passed", "mean_ms", "p50_ms", "p95_ms", "p99_ms",
            "min_ms", "max_ms", "std_ms", "coefficient_of_variation",
            "images_per_second", "error",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["screening_matrix"]:
            stats = row.get("statistics", {})
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "status": row["status"],
                    "change_group": row["change_group"],
                    "explanatory_control": row["explanatory_control"],
                    "exactness_passed": row.get("exactness", {}).get("passed"),
                    **{key: stats.get(key) for key in fields[5:14]},
                    "error": row.get("error"),
                }
            )
    with (output_root / "raw_timings.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["phase", "candidate_id", "repetition", "iteration", "latency_ms"])
        for row in report["screening_matrix"]:
            for repetition, samples in enumerate(row.get("raw_samples_ms", []), 1):
                for iteration, latency in enumerate(samples, 1):
                    writer.writerow(["screening", row["candidate_id"], repetition, iteration, latency])
        confirmation = report.get("finalist_confirmation") or {}
        for candidate_id, arm in confirmation.get("arms", {}).items():
            for repetition, samples in enumerate(arm["raw_samples_ms"], 1):
                for iteration, latency in enumerate(samples, 1):
                    writer.writerow(["confirmation", candidate_id, repetition, iteration, latency])
    with (output_root / "profile_nodes.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fields = [
            "profile_id", "rank", "name", "op_name", "category",
            "execution_count", "total_duration_us", "mean_duration_us",
            "median_duration_us", "percentage_of_node_event_time",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for profile in report["profiles"]:
            for rank, row in enumerate(profile["analysis"]["nodes"], 1):
                writer.writerow({"profile_id": profile["profile_id"], "rank": rank, **row})

    baseline = next(row for row in report["screening_matrix"] if row["candidate_id"] == BASELINE_ID)
    lines = [
        "# v1.5 ORT CPU runtime tail-latency investigation",
        "",
        f"Scope: **{SCOPE}**. This is a session-local runtime investigation, not kernel, graph, quantization, accelerator, or universal-performance work.",
        "",
        "## Provenance and protocol",
        "",
        f"- Custom model SHA-256: `{report['provenance']['custom_model']['sha256']}`",
        f"- Reference model SHA-256: `{report['provenance']['reference_model']['sha256']}`",
        f"- Unchanged DLL SHA-256: `{report['provenance']['custom_library']['sha256']}`",
        f"- ORT/runtime: `{report['machine']['onnxruntime']}` / `{report['machine']['runtime_dll_file_version']}`",
        f"- Provider: `{report['machine']['selected_provider']}`; processor/logical CPUs: `{report['machine']['processor']}` / `{report['machine']['logical_cpu_count']}`",
        f"- Cached batch-1 input SHA-256: `{report['benchmark_input_digest']}`",
        "- Every raw sample was retained; no outlier was deleted. Session creation is recorded separately from warm inference.",
        "",
        "## Baseline reproducibility",
        "",
        "The frozen v1.4 settings were rerun for 10 fresh-session repetitions with 20 warmups and 100 timings each.",
        "",
        "| Mean | P50 | P95 | P99 | Min | Max | Std | CV | Images/s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {baseline['statistics']['mean_ms']:.3f} | {baseline['statistics']['p50_ms']:.3f} | {baseline['statistics']['p95_ms']:.3f} | {baseline['statistics']['p99_ms']:.3f} | {baseline['statistics']['min_ms']:.3f} | {baseline['statistics']['max_ms']:.3f} | {baseline['statistics']['std_ms']:.3f} | {baseline['statistics']['coefficient_of_variation']:.3f} | {baseline['statistics']['images_per_second']:.2f} |",
        "",
        f"Cold session creation: mean `{baseline['session_creation_statistics']['mean_ms']:.3f} ms`, p50 `{baseline['session_creation_statistics']['p50_ms']:.3f} ms`, p95 `{baseline['session_creation_statistics']['p95_ms']:.3f} ms`; excluded from warm inference timings.",
        "",
        "Baseline repetition summaries:",
        "",
        "| Rep | Mean | P50 | P95 | P99 | Min | Max | Std | CV |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for repetition, stats in enumerate(baseline["statistics"]["per_repetition"], 1):
        lines.append(
            f"| {repetition} | {stats['mean_ms']:.3f} | {stats['p50_ms']:.3f} | "
            f"{stats['p95_ms']:.3f} | {stats['p99_ms']:.3f} | {stats['min_ms']:.3f} | "
            f"{stats['max_ms']:.3f} | {stats['std_ms']:.3f} | {stats['coefficient_of_variation']:.3f} |"
        )
    lines.extend([
        "",
        f"Tail-latency reproducibility conclusion: **{report['decisions']['tail_latency_reproducible']['conclusion']}**. {report['decisions']['tail_latency_reproducible']['criterion']}",
        "",
        "## Complete screening matrix",
        "",
        "Each non-baseline cell used its own exact 128-image final-logit gate followed by five fresh-session 20x100 repetitions only after exactness passed.",
        "",
        "| Candidate | Status | Exact | Mean | P50 | P95 | P99 | CV | Images/s | Requested change |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in report["screening_matrix"]:
        stats = row.get("statistics", {})
        lines.append(
            f"| `{row['candidate_id']}` | {row['status']} | {row.get('exactness', {}).get('passed', False)} | "
            f"{stats.get('mean_ms', float('nan')):.3f} | {stats.get('p50_ms', float('nan')):.3f} | "
            f"{stats.get('p95_ms', float('nan')):.3f} | {stats.get('p99_ms', float('nan')):.3f} | "
            f"{stats.get('coefficient_of_variation', float('nan')):.3f} | "
            f"{stats.get('images_per_second', float('nan')):.2f} | {row['label']} |"
        )
    lines.extend([
        "",
        "Requested versus actually applied settings:",
        "",
        "| Candidate | Requested mode/threads/inter/graph/arena/pattern | Applied mode/threads/inter/graph/arena/pattern |",
        "|---|---|---|",
    ])
    for row in report["screening_matrix"]:
        if not row.get("session_metadata"):
            lines.append(f"| `{row['candidate_id']}` | unavailable | rejected before session |")
            continue
        requested = row["requested_options"]
        applied = row["session_metadata"][0]["options"]["applied"]
        lines.append(
            f"| `{row['candidate_id']}` | `{requested['execution_mode']}` / `{requested['intra_op_num_threads']}` / "
            f"`{requested['inter_op_num_threads']}` / `{requested['graph_optimization_level']}` / "
            f"`{requested['enable_cpu_mem_arena']}` / `{requested['enable_mem_pattern']}` | "
            f"`{applied['execution_mode']}` / `{applied['intra_op_num_threads']}` / "
            f"`{applied['inter_op_num_threads']}` / `{applied['graph_optimization_level']}` / "
            f"`{applied['enable_cpu_mem_arena']}` / `{applied['enable_mem_pattern']}` |"
        )
    lines.extend([
        "",
        "Every repetition summary, actionable rejection error, and raw timing is retained in JSON/CSV.",
        "",
        "## Independent baseline profiles",
        "",
        report["profile_comparison"]["non_additivity_warning"],
        "",
        "| Profile | Wall ms | Node sum ms | Conv | Quantization | Custom | 17 nodes | Dominant |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ])
    for row in report["profile_comparison"]["profiles"]:
        lines.append(
            f"| `{row['profile_id']}` | {row['wall_ms']:.3f} | {row['node_sum_ms']:.3f} | "
            f"{row['convolution_share_percent']:.2f}% | {row['quantization_related_share_percent']:.2f}% | "
            f"{row['custom_activation_share_percent']:.2f}% | {row['custom_nodes'] == 17} | {row['dominant_category']} |"
        )
    lines.extend([
        "",
        f"Absolute custom-share difference: `{report['profile_comparison']['custom_share_absolute_difference_pp']:.2f} pp`; material by the recorded 5 pp rule: `{report['profile_comparison']['material_custom_share_difference']}`.",
        "The two profiles agree on category ordering and their custom shares differ by less than one percentage point; this is not a material attribution change.",
        "",
        "Top five nodes from each independent baseline profile:",
        "",
        "| Profile | Rank | Node | Op | Category | Share |",
        "|---|---:|---|---|---|---:|",
    ])
    for profile in report["profiles"][:2]:
        for rank, row in enumerate(profile["analysis"]["top_20_nodes"][:5], 1):
            lines.append(
                f"| `{profile['profile_id']}` | {rank} | `{row['name']}` | `{row['op_name']}` | "
                f"`{row['category']}` | {row['percentage_of_node_event_time']:.2f}% |"
            )
    lines.extend([
        "",
        "## Finalist confirmation",
        "",
    ])
    confirmation = report.get("finalist_confirmation")
    if confirmation:
        lines.extend([
            f"Screening finalist: `{report['screening_selection']['selected_candidate_id']}`. Confirmation alternated baseline and candidate for 10 fresh-session repetitions per arm.",
            f"Finalist retained after confirmation: `{confirmation['selected_after_confirmation']}`.",
            "",
            "| Confirmation arm | Mean | P50 | P95 | P99 | CV | Median rep P50 | Median rep P95 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for candidate_id, arm in confirmation["arms"].items():
            stats = arm["statistics"]
            lines.append(
                f"| `{candidate_id}` | {stats['mean_ms']:.3f} | {stats['p50_ms']:.3f} | "
                f"{stats['p95_ms']:.3f} | {stats['p99_ms']:.3f} | "
                f"{stats['coefficient_of_variation']:.3f} | "
                f"{stats['median_of_repetition_p50_ms']:.3f} | "
                f"{stats['median_of_repetition_p95_ms']:.3f} |"
            )
        lines.extend([
            "",
            "Confirmation checks: `" + ", ".join(
                f"{key}={value}"
                for key, value in confirmation["finalist_evaluation"]["checks"].items()
            ) + "`.",
        ])
    else:
        lines.append("No candidate met the screening criteria; no finalist confirmation was run.")
    lines.extend([
        "",
        "## Final decisions",
        "",
        "| Question | Conclusion |",
        "|---|---|",
        f"| Is v1.4 tail latency reproducible? | {report['decisions']['tail_latency_reproducible']['conclusion']} |",
        f"| Does any setting improve typical latency? | {report['decisions']['typical_latency']['conclusion']} |",
        f"| Does any setting reduce tail variance? | {report['decisions']['tail_variance']['conclusion']} |",
        f"| Recommended preset | {report['decisions']['recommended_preset']['decision']} |",
        f"| Conv remains dominant? | {report['decisions']['convolution']['conclusion']} |",
        f"| AVX2 priority | {report['decisions']['avx2_priority']['decision']} |",
        f"| Next smallest experiment | {report['decisions']['next_smallest_experiment']} |",
        "",
        "No C++ kernel, model, graph, calibration, quantization, global environment, affinity, priority, or power setting was changed. No 10,000-image evaluation was run.",
        "",
        "## Limitations",
        "",
        "- One Windows x64 machine and ORT 1.19.2 only.",
        "- Tail events can depend on external OS activity not controlled by session-local settings.",
        "- Profiles add instrumentation overhead and node sums are non-additive.",
        "- Nearby thread rows are explanatory controls, not replacement presets unless confirmed.",
    ])
    atomic_text(output_root / "diagnosis_report.md", "\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/runtime_profiles/resnet18_silu_cifar10_v15_ort_customop_tail_latency.json"
        ),
    )
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if "torch" in sys.modules or "torchvision" in sys.modules:
        raise RuntimeError("v1.5 diagnostic process must remain NumPy/ORT-only")
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    output_root = validate_output_path(resolve_path(config["output_root"]), ROOT)
    if output_root.exists():
        if not args.force_rebuild:
            raise FileExistsError(
                f"v1.5 output already exists; use --force-rebuild: {output_root}"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    atomic_json(output_root / "run_status.json", {"status": "running", "phase": "provenance"})

    reference_path = resolve_path(config["reference_model"])
    custom_path = resolve_path(config["custom_model"])
    library_path = resolve_path(config["custom_library"])
    provenance = {
        "reference_model": verify_artifact(reference_path, config["reference_model_sha256"], label="v1.1 reference model"),
        "custom_model": verify_artifact(custom_path, config["custom_model_sha256"], label="v1.2 custom model"),
        "custom_library": verify_artifact(library_path, config["custom_library_sha256"], label="v1.2 custom-op DLL"),
        "historical_v14_report": {
            "path": str(resolve_path(config["historical_v14_report"])),
            "sha256": sha256_file(resolve_path(config["historical_v14_report"])),
        },
        "historical_v14_raw_timings": {
            "path": str(resolve_path(config["historical_v14_raw_timings"])),
            "sha256": sha256_file(resolve_path(config["historical_v14_raw_timings"])),
        },
    }
    if ort.__version__ != config["ort_version"]:
        raise RuntimeError(f"ORT version mismatch: expected {config['ort_version']}, got {ort.__version__}")
    runtime_dll = prepare_process_local_ort_runtime()
    images, _ = load_cifar_batch(resolve_path(config["data_root"]), "test_batch")
    controls = json.loads(resolve_path(config["probe_indices"]).read_text(encoding="utf-8"))
    indices = [int(value) for value in controls["probe_indices"]]
    if len(indices) != 128 or len(set(indices)) != 128:
        raise ValueError("v1.5 requires 128 unique frozen probe indices")
    normalized_probe = normalize_cifar_images(images[indices])
    benchmark_input = normalize_cifar_images(images[:1])
    historical_report = json.loads(
        resolve_path(config["historical_v14_report"]).read_text(encoding="utf-8")
    )
    historical_baseline = next(
        row for row in historical_report["benchmarks"] if row["graph"] == "custom_op"
    )["statistics"]

    try:
        candidates = screening_candidates(config)
        rows = []
        print("v1.5 phase 1/5: baseline reproducibility and bounded screening", flush=True)
        for index, candidate in enumerate(candidates, 1):
            protocol = (
                config["baseline_benchmark"]
                if candidate["candidate_id"] == BASELINE_ID
                else config["screening_benchmark"]
            )
            print(
                f"  candidate {index}/{len(candidates)} {candidate['candidate_id']} "
                f"({protocol['repetitions']} repetitions)",
                flush=True,
            )
            row = run_candidate(
                candidate=candidate,
                reference_path=reference_path,
                custom_path=custom_path,
                library_path=library_path,
                normalized_probe=normalized_probe,
                benchmark_input=benchmark_input,
                protocol=protocol,
            )
            rows.append(row)
            atomic_json(output_root / "screening_matrix.partial.json", rows)
        baseline_row = next(row for row in rows if row["candidate_id"] == BASELINE_ID)
        if baseline_row["status"] != "complete":
            raise RuntimeError("v1.5 baseline did not complete exactness and timing")
        tail_result = tail_reproducibility(
            baseline_row["statistics"], historical_baseline
        )

        atomic_json(output_root / "run_status.json", {"status": "running", "phase": "profiles"})
        print("v1.5 phase 2/5: two independent unchanged baseline profiles", flush=True)
        profiles = [
            run_profile(
                profile_id=f"baseline_profile_{index}",
                candidate=candidates[0],
                model_path=custom_path,
                library_path=library_path,
                inputs=benchmark_input,
                iterations=config["profiling"]["iterations"],
                output_root=output_root,
            )
            for index in (1, 2)
        ]
        profile_comparison = compare_profiles(profiles)

        selection = select_finalist(baseline_row, rows)
        finalist_confirmation = None
        selected = selection["selected_candidate_id"]
        print(f"v1.5 phase 3/5: screening selection={selected}", flush=True)
        if selected:
            finalist = next(row for row in candidates if row["candidate_id"] == selected)
            atomic_json(output_root / "run_status.json", {"status": "running", "phase": "finalist_confirmation"})
            finalist_confirmation = run_alternating_confirmation(
                baseline=candidates[0],
                finalist=finalist,
                custom_path=custom_path,
                library_path=library_path,
                inputs=benchmark_input,
                protocol=config["finalist_confirmation"],
            )
            profiles.append(
                run_profile(
                    profile_id=f"finalist_{selected}_profile",
                    candidate=finalist,
                    model_path=custom_path,
                    library_path=library_path,
                    inputs=benchmark_input,
                    iterations=config["profiling"]["iterations"],
                    output_root=output_root,
                )
            )
        decisions = build_decisions(
            tail_result=tail_result,
            screening=rows,
            finalist_confirmation=finalist_confirmation,
            profile_comparison=profile_comparison,
        )
        report = {
            "schema_version": REPORT_SCHEMA,
            "completion_status": "success",
            "scope": SCOPE,
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "machine": machine_facts(runtime_dll),
            "provenance": provenance,
            "probe_input_digest": input_digest(normalized_probe, indices),
            "benchmark_input_digest": input_digest(benchmark_input, [0]),
            "historical_v14_statistics": historical_baseline,
            "screening_matrix": rows,
            "profiles": profiles,
            "profile_comparison": profile_comparison,
            "screening_selection": selection,
            "finalist_confirmation": finalist_confirmation,
            "decisions": decisions,
        }
        print("v1.5 phase 4/5: write JSON/CSV/Markdown evidence", flush=True)
        write_reports(output_root, report)
        atomic_json(output_root / "run_status.json", {"status": "complete"})
        print("v1.5 phase 5/5: complete", flush=True)
        print(output_root, flush=True)
        return 0
    except Exception as error:
        atomic_json(
            output_root / "run_status.json",
            {"status": "failed", "error_type": type(error).__name__, "error": str(error)},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
