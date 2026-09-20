"""Validate the explicit v1.4 four-thread ORT custom-op CPU preset."""

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
from silu_benchmark.ort_custom_op_thread_preset import (
    EXPECTED_SITE_COUNT,
    REPORT_SCHEMA,
    benchmark_statistics,
    build_session_options,
    compare_historical_protocol,
    exact_tensor_metrics,
    load_config,
    preset_fingerprint,
    protocol_record,
    repetition_reproduced,
    sha256_file,
    validate_output_path,
    validation_decisions,
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


def load_v13_helpers():
    path = ROOT / "scripts/diagnose_ort_cpp_customop_performance.py"
    spec = importlib.util.spec_from_file_location("frozen_v13_diagnostic_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_preset_session(
    model_path: Path,
    *,
    graph: str,
    library_path: Path,
    preset: Mapping,
    profile_prefix: Path | None = None,
) -> tuple[ort.InferenceSession, dict]:
    options, settings = build_session_options(preset, profile_prefix=profile_prefix)
    registration_ms = 0.0
    registered_library = None
    runtime_dll = None
    if graph == "custom_op":
        runtime_dll = prepare_process_local_ort_runtime()
        started = time.perf_counter_ns()
        registered_library = options.register_custom_ops_library(str(library_path))
        registration_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    started = time.perf_counter_ns()
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=[preset["provider"]]
    )
    session_creation_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return session, {
        "graph": graph,
        "model": str(model_path),
        "registration_load_ms": registration_ms,
        "registered_library": registered_library or (
            str(library_path) if graph == "custom_op" else None
        ),
        "session_creation_load_ms": session_creation_ms,
        "session_options": settings,
        "providers": session.get_providers(),
        "runtime_dll": str(runtime_dll) if runtime_dll else None,
    }


def run_exactness_gate(
    *,
    reference_path: Path,
    custom_path: Path,
    library_path: Path,
    output_root: Path,
    images: np.ndarray,
    indices: list[int],
    preset: Mapping,
) -> dict:
    helpers = load_v13_helpers()
    reference_model, custom_model, islands = helpers.prepare_probe_models(
        reference_path, custom_path, output_root
    )
    reference_session, reference_metadata = create_preset_session(
        reference_model,
        graph="standard_operator",
        library_path=library_path,
        preset=preset,
    )
    custom_session, custom_metadata = create_preset_session(
        custom_model,
        graph="custom_op",
        library_path=library_path,
        preset=preset,
    )
    rows = [
        {
            "site_id": island.site_id,
            "tensor_name": island.output_code_tensor,
            "elements_compared": 0,
            "mismatch_count": 0,
            "exact_equal": True,
        }
        for island in islands
    ]
    normalized = normalize_cifar_images(images[indices])
    completed = 0
    logits_elements = 0
    predictions_equal = 0
    first_failure = None
    for offset in range(0, len(indices), 8):
        batch = normalized[offset : offset + 8]
        reference_outputs = reference_session.run(
            None, {reference_session.get_inputs()[0].name: batch}
        )
        custom_outputs = custom_session.run(
            None, {custom_session.get_inputs()[0].name: batch}
        )
        for site_index, row in enumerate(rows):
            metrics = exact_tensor_metrics(
                reference_outputs[site_index], custom_outputs[site_index]
            )
            row["elements_compared"] += metrics["elements_compared"]
            row["mismatch_count"] += metrics["mismatch_count"]
            row["exact_equal"] = row["exact_equal"] and metrics["exact_equal"]
            if not metrics["exact_equal"]:
                first_failure = {
                    "site_id": row["site_id"],
                    "probe_offset": offset,
                    "metrics": metrics,
                }
                break
        if first_failure:
            break
        logits_metrics = exact_tensor_metrics(reference_outputs[-1], custom_outputs[-1])
        logits_elements += logits_metrics["elements_compared"]
        if not logits_metrics["exact_equal"]:
            first_failure = {
                "site_id": "final_logits",
                "probe_offset": offset,
                "metrics": logits_metrics,
            }
            break
        predictions_equal += int(
            np.sum(
                np.argmax(reference_outputs[-1], axis=1)
                == np.argmax(custom_outputs[-1], axis=1)
            )
        )
        completed += len(batch)
    passed = bool(
        first_failure is None
        and completed == 128
        and len(rows) == EXPECTED_SITE_COUNT
        and all(row["exact_equal"] for row in rows)
        and logits_elements == 1280
        and predictions_equal == 128
    )
    result = {
        "schema_version": "ort-customop-thread-preset-exactness/v1",
        "passed": passed,
        "samples_required": 128,
        "samples_completed": completed,
        "input_digest": input_digest(normalized, indices),
        "preset_fingerprint": preset_fingerprint(preset),
        "site_code_parity": rows,
        "all_site_codes_exact": all(row["exact_equal"] for row in rows),
        "final_logits_exact": first_failure is None and logits_elements == 1280,
        "final_logits_elements": logits_elements,
        "prediction_agreement": predictions_equal / completed if completed else 0.0,
        "first_failure": first_failure,
        "reference_session": reference_metadata,
        "custom_session": custom_metadata,
    }
    atomic_json(output_root / "exactness_gate.json", result)
    if not passed:
        raise RuntimeError(
            "v1.4 four-thread exactness gate failed; profiling and benchmarking were not run: "
            + json.dumps(first_failure, sort_keys=True)
        )
    return result


def run_profile(
    *,
    model_path: Path,
    library_path: Path,
    inputs: np.ndarray,
    iterations: int,
    output_root: Path,
    preset: Mapping,
) -> dict:
    prefix = output_root / "profiles/custom_op_4threads_profile"
    session, metadata = create_preset_session(
        model_path,
        graph="custom_op",
        library_path=library_path,
        preset=preset,
        profile_prefix=prefix,
    )
    output_name = session.get_outputs()[0].name
    feed = {session.get_inputs()[0].name: inputs}
    wall_ms = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        session.run([output_name], feed)
        wall_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    profile_path = Path(session.end_profiling()).resolve()
    analysis = parse_profile_file(
        profile_path, inference_wall_time_us=sum(wall_ms) * 1000.0
    )
    custom = analysis["custom_op_execution"]
    counts = {row["execution_count"] for row in custom["nodes"]}
    if custom["unique_node_count"] != EXPECTED_SITE_COUNT or counts != {iterations}:
        raise RuntimeError(
            "v1.4 profile did not prove all 17 custom nodes executed once per invocation"
        )
    return {
        "profile_path": str(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "profile_iterations": iterations,
        "batch_size": len(inputs),
        "raw_inference_wall_ms": wall_ms,
        "total_inference_wall_ms": sum(wall_ms),
        "session": metadata,
        "analysis": analysis,
    }


def run_benchmark(
    *,
    graph: str,
    model_path: Path,
    library_path: Path,
    inputs: np.ndarray,
    benchmark: Mapping,
    preset: Mapping,
) -> dict:
    raw_repetitions = []
    sessions = []
    for repetition in range(benchmark["repetitions"]):
        session, metadata = create_preset_session(
            model_path,
            graph=graph,
            library_path=library_path,
            preset=preset,
        )
        output_name = session.get_outputs()[0].name
        feed = {session.get_inputs()[0].name: inputs}
        for _ in range(benchmark["warmup_iterations"]):
            session.run([output_name], feed)
        samples = []
        for _ in range(benchmark["timed_iterations"]):
            started = time.perf_counter_ns()
            session.run([output_name], feed)
            samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
        raw_repetitions.append(samples)
        sessions.append(metadata)
        print(
            f"  {graph} repetition {repetition + 1}/{benchmark['repetitions']}: "
            f"p50={np.percentile(samples, 50):.3f} ms "
            f"p95={np.percentile(samples, 95):.3f} ms",
            flush=True,
        )
        del session
    return {
        "graph": graph,
        "graph_form": (
            "v1.2 hybrid graph with 17 project C++ custom activations"
            if graph == "custom_op"
            else "v1.1 expanded standard-operator piecewise reference"
        ),
        "batch_size": 1,
        "preset_fingerprint": preset_fingerprint(preset),
        "input_digest": input_digest(inputs, list(range(len(inputs)))),
        "warmup_iterations_per_repetition": benchmark["warmup_iterations"],
        "timed_iterations_per_repetition": benchmark["timed_iterations"],
        "repetitions": benchmark["repetitions"],
        "profiling_enabled": False,
        "raw_samples_ms": raw_repetitions,
        "statistics": benchmark_statistics(raw_repetitions),
        "session_metadata": sessions,
        "timing_boundary": (
            "perf_counter_ns immediately around session.run([logits], cached_feed); "
            "input preparation and session creation excluded"
        ),
    }


def machine_facts(runtime_dll: Path | None) -> dict:
    helpers = load_v13_helpers()
    facts = helpers.machine_facts(runtime_dll)
    facts.update(
        {
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "torch_imported": "torch" in sys.modules or "torchvision" in sys.modules,
        }
    )
    return facts


def historical_comparison(
    report_path: Path, current_benchmark: Mapping
) -> dict:
    if not report_path.is_file():
        return {
            "available": False,
            "path": str(report_path),
            "protocol": {"directly_protocol_comparable": False, "differences": []},
            "reproduction": {"reproduced": False, "criterion": "historical report unavailable"},
        }
    historical_report = json.loads(report_path.read_text(encoding="utf-8"))
    historical_cell = next(
        row
        for row in historical_report["benchmark_matrix"]["cells"]
        if row["graph"] == "custom_op"
        and row["batch_size"] == 1
        and row["intra_op_threads"] == 4
        and row["execution_mode"] == "sequential"
    )
    current_protocol = protocol_record(input_digest=current_benchmark["input_digest"])
    historical_protocol = protocol_record(input_digest=historical_cell["input_digest"])
    protocol = compare_historical_protocol(current_protocol, historical_protocol)
    reproduction = repetition_reproduced(
        current_benchmark["statistics"], historical_cell["statistics"]
    )
    return {
        "available": True,
        "path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "protocol": protocol,
        "current_protocol": current_protocol,
        "historical_protocol": historical_protocol,
        "historical_statistics": historical_cell["statistics"],
        "reproduction": reproduction,
        "comparison_note": (
            "Same-machine historical evidence. Different graph forms are never presented as speedup baselines."
        ),
    }


def write_reports(output_root: Path, report: Mapping) -> None:
    atomic_json(output_root / "validation_report.json", report)
    atomic_json(output_root / "profile_analysis.json", report["profile"])
    atomic_json(output_root / "benchmark_results.json", report["benchmarks"])
    atomic_json(output_root / "historical_comparison.json", report["historical_comparison"])

    with (output_root / "benchmark_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fields = [
            "graph", "mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms",
            "std_ms", "coefficient_of_variation", "images_per_second",
            "total_measured_wall_time_seconds",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for benchmark in report["benchmarks"]:
            writer.writerow(
                {"graph": benchmark["graph"], **{
                    key: benchmark["statistics"][key] for key in fields[1:]
                }}
            )
    with (output_root / "raw_timings.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["graph", "repetition", "iteration", "latency_ms"])
        for benchmark in report["benchmarks"]:
            for repetition, samples in enumerate(benchmark["raw_samples_ms"], 1):
                for iteration, latency in enumerate(samples, 1):
                    writer.writerow([benchmark["graph"], repetition, iteration, latency])
    with (output_root / "profile_nodes.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fields = [
            "rank", "name", "op_name", "category", "execution_count",
            "total_duration_us", "mean_duration_us", "median_duration_us",
            "percentage_of_node_event_time",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(report["profile"]["analysis"]["nodes"], 1):
            writer.writerow({"rank": rank, **row})

    profile = report["profile"]
    analysis = profile["analysis"]
    custom_benchmark = next(
        row for row in report["benchmarks"] if row["graph"] == "custom_op"
    )
    standard_benchmark = next(
        row for row in report["benchmarks"] if row["graph"] == "standard_operator"
    )
    current = custom_benchmark["statistics"]
    historical = report["historical_comparison"]
    lines = [
        "# v1.4 validated four-thread ORT custom-op runtime preset",
        "",
        f"Scope: **{SCOPE}**. This is an explicit, machine-specific preset, not a global default or universal CPU recommendation.",
        "",
        "## Preset contract",
        "",
        f"- Name/version: `{report['preset']['name']}` / `{report['preset']['version']}`",
        f"- Provider: `{report['preset']['provider']}`",
        f"- Execution mode: `{report['preset']['execution_mode']}`",
        f"- Intra-op threads: `{report['preset']['intra_op_num_threads']}`",
        f"- Inter-op threads/behavior: `{report['preset']['inter_op_num_threads']}` / {report['preset']['inter_op_behavior']}",
        f"- Graph optimization: `{report['preset']['graph_optimization_level']}`",
        f"- Fingerprint: `{report['preset_fingerprint']}`",
        f"- Scope boundary: {report['preset']['scope_boundary']}",
        "",
        "## Provenance and runtime",
        "",
        f"- v1.1 reference model SHA-256: `{report['provenance']['reference_model']['sha256']}`",
        f"- v1.2 custom model SHA-256: `{report['provenance']['custom_model']['sha256']}`",
        f"- Unchanged custom-op DLL SHA-256: `{report['provenance']['custom_library']['sha256']}`",
        f"- ORT/runtime DLL version: `{report['machine']['onnxruntime']}` / `{report['machine']['runtime_dll_file_version']}`",
        f"- Provider/processor/logical CPUs: `{report['machine']['selected_provider']}` / `{report['machine']['processor']}` / `{report['machine']['logical_cpu_count']}`",
        f"- PyTorch imported by validation process: `{report['machine']['torch_imported']}`",
        "",
        "## Exactness gate",
        "",
        f"- Passed: `{report['exactness_gate']['passed']}`",
        f"- All 17 uint8 activation outputs exact: `{report['exactness_gate']['all_site_codes_exact']}`",
        f"- Final logits exact across 1,280 values: `{report['exactness_gate']['final_logits_exact']}`",
        f"- Prediction agreement: `{report['exactness_gate']['prediction_agreement']:.2%}`",
        f"- Probe input SHA-256: `{report['exactness_gate']['input_digest']}`",
        "",
        "Actual applied custom-session options:",
        "",
        "```json",
        json.dumps(report["exactness_gate"]["custom_session"]["session_options"], indent=2, sort_keys=True),
        "```",
        "",
        "## Four-thread profile",
        "",
        f"- Profiled inference wall time: `{profile['total_inference_wall_ms']:.3f} ms`",
        f"- Summed node-event time: `{analysis['summed_node_event_time_us'] / 1000.0:.3f} ms`",
        f"- All custom nodes: `{analysis['custom_op_execution']['unique_node_count']}/17`; total executions: `{sum(row['execution_count'] for row in analysis['custom_op_execution']['nodes'])}`",
        f"- Aggregate custom-op share: `{analysis['custom_op_execution']['percentage_of_node_event_time']:.3f}%`",
        "",
        analysis["non_additivity_warning"],
        "",
        "| Category | Executions | Total ms | Node-event share |",
        "|---|---:|---:|---:|",
    ]
    for row in analysis["categories"]:
        lines.append(
            f"| `{row['name']}` | {row['execution_count']} | "
            f"{row['total_duration_us'] / 1000.0:.3f} | {row['percentage_of_node_event_time']:.2f}% |"
        )
    lines.extend([
        "",
        "Top 20 nodes:",
        "",
        "| Rank | Node | Op | Category | Total ms | Share |",
        "|---:|---|---|---|---:|---:|",
    ])
    for rank, row in enumerate(analysis["top_20_nodes"], 1):
        lines.append(
            f"| {rank} | `{row['name']}` | `{row['op_name']}` | `{row['category']}` | "
            f"{row['total_duration_us'] / 1000.0:.3f} | {row['percentage_of_node_event_time']:.2f}% |"
        )
    lines.extend([
        "",
        "Profile category shares are compared with v1.3 only as ordering/attribution evidence because the session configuration differs.",
        "",
        "## Repeated uninstrumented benchmark",
        "",
        "Every graph used five fresh sessions, each with 20 warmups and 100 timed invocations. All raw samples were retained; no outlier was deleted.",
        "",
        "| Graph form | Mean ms | P50 ms | P95 ms | Min ms | Max ms | Std ms | CV | Images/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in (custom_benchmark, standard_benchmark):
        stats = row["statistics"]
        lines.append(
            f"| {row['graph_form']} | {stats['mean_ms']:.3f} | {stats['p50_ms']:.3f} | "
            f"{stats['p95_ms']:.3f} | {stats['min_ms']:.3f} | {stats['max_ms']:.3f} | "
            f"{stats['std_ms']:.3f} | {stats['coefficient_of_variation']:.3f} | "
            f"{stats['images_per_second']:.2f} |"
        )
    lines.extend([
        "",
        "Custom-op repetition summaries:",
        "",
        "| Repetition | Mean ms | P50 ms | P95 ms | Min ms | Max ms | Std ms | CV |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for repetition, stats in enumerate(current["per_repetition"], 1):
        lines.append(
            f"| {repetition} | {stats['mean_ms']:.3f} | {stats['p50_ms']:.3f} | "
            f"{stats['p95_ms']:.3f} | {stats['min_ms']:.3f} | {stats['max_ms']:.3f} | "
            f"{stats['std_ms']:.3f} | {stats['coefficient_of_variation']:.3f} |"
        )
    lines.extend([
        "",
        f"Custom repetition p50 range: `{current['repetition_p50_range_ms'][0]:.3f}` to `{current['repetition_p50_range_ms'][1]:.3f} ms`; repetition p95 range: `{current['repetition_p95_range_ms'][0]:.3f}` to `{current['repetition_p95_range_ms'][1]:.3f} ms`.",
        "The standard-operator control is a different graph form and is not a speedup baseline.",
        "",
        "## Historical reproduction and decisions",
        "",
        f"- v1.3 protocol matched: `{historical['protocol']['directly_protocol_comparable']}`",
        f"- Predeclared repetition criterion: {historical['reproduction']['criterion']}",
        f"- Typical latency reproduced: `{historical['reproduction']['typical_latency_reproduced']}`; current median repetition p50 `{historical['reproduction']['current_median_of_repetition_p50_ms']:.3f} ms` versus historical range `{historical['reproduction']['historical_repetition_p50_range_ms'][0]:.3f}`–`{historical['reproduction']['historical_repetition_p50_range_ms'][1]:.3f} ms`.",
        f"- Variance reproduced: `{historical['reproduction']['variance_reproduced']}`; current CV `{historical['reproduction']['current_coefficient_of_variation']:.3f}` versus historical `{historical['reproduction']['historical_coefficient_of_variation']:.3f}`.",
        f"- Full v1.3 four-thread distribution reproduced: `{historical['reproduction']['reproduced']}`",
        f"- Preset decision: **{report['decisions']['preset']['decision']}**",
        f"- Preset rationale: {report['decisions']['preset']['reason']}",
        f"- Convolution remains dominant: `{report['decisions']['convolution']['remains_dominant']}` at `{report['decisions']['convolution']['share_percent']:.2f}%`",
        f"- AVX2 decision: **{report['decisions']['avx2']['decision']}**; custom share `{report['decisions']['avx2']['custom_node_share_percent']:.2f}%`",
        f"- AVX2 rationale: {report['decisions']['avx2']['reason']}",
        f"- Next narrow target: **{report['decisions']['convolution']['next_target']}**",
        "",
        "No AVX2, OpenMP, custom thread pool, graph rewrite, fusion, calibration change, ONNX rewrite, or C++ kernel semantic change was implemented.",
        "",
        "## Limitations",
        "",
        "- One Windows x64 machine and ORT 1.19.2 only; no cross-machine or universal-default claim.",
        "- Profile node-event sums can overlap and are not a wall-clock decomposition.",
        "- The different standard-operator graph form is a control, not performance equivalence evidence.",
        "- No 10,000-image evaluation was run; frozen v1.2 accuracy evidence remains unchanged.",
    ])
    atomic_text(output_root / "validation_report.md", "\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/runtime_profiles/resnet18_silu_cifar10_v14_ort_customop_cpu_4threads.json"
        ),
    )
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if "torch" in sys.modules or "torchvision" in sys.modules:
        raise RuntimeError("v1.4 validation process must remain NumPy/ORT-only")
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    output_root = validate_output_path(resolve_path(config["output_root"]), ROOT)
    if output_root.exists():
        if not args.force_rebuild:
            raise FileExistsError(
                f"v1.4 output already exists; use --force-rebuild: {output_root}"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    atomic_json(output_root / "run_status.json", {"status": "running", "phase": "provenance"})

    reference_path = resolve_path(config["reference_model"])
    custom_path = resolve_path(config["custom_model"])
    library_path = resolve_path(config["custom_library"])
    provenance = {
        "reference_model": verify_artifact(
            reference_path, config["reference_model_sha256"], label="v1.1 reference model"
        ),
        "custom_model": verify_artifact(
            custom_path, config["custom_model_sha256"], label="v1.2 custom model"
        ),
        "custom_library": verify_artifact(
            library_path, config["custom_library_sha256"], label="v1.2 custom-op DLL"
        ),
    }
    if ort.__version__ != config["ort_version"]:
        raise RuntimeError(
            f"ORT version mismatch: expected {config['ort_version']}, got {ort.__version__}"
        )
    runtime_dll = prepare_process_local_ort_runtime()
    images, _ = load_cifar_batch(resolve_path(config["data_root"]), "test_batch")
    controls = json.loads(resolve_path(config["probe_indices"]).read_text(encoding="utf-8"))
    indices = [int(value) for value in controls["probe_indices"]]
    if len(indices) != 128 or len(set(indices)) != 128:
        raise ValueError("v1.4 probe requires 128 unique frozen indices")

    try:
        print("v1.4 phase 1/4: four-thread exactness gate", flush=True)
        exactness = run_exactness_gate(
            reference_path=reference_path,
            custom_path=custom_path,
            library_path=library_path,
            output_root=output_root,
            images=images,
            indices=indices,
            preset=config["preset"],
        )
        atomic_json(output_root / "run_status.json", {"status": "running", "phase": "profile"})
        benchmark_input = normalize_cifar_images(images[:1])

        print("v1.4 phase 2/4: isolated four-thread custom profile", flush=True)
        profile = run_profile(
            model_path=custom_path,
            library_path=library_path,
            inputs=benchmark_input,
            iterations=config["profiling"]["iterations"],
            output_root=output_root,
            preset=config["preset"],
        )
        atomic_json(output_root / "run_status.json", {"status": "running", "phase": "benchmark"})

        print("v1.4 phase 3/4: repeated custom and standard controls", flush=True)
        benchmarks = []
        for graph, model in (
            ("custom_op", custom_path),
            ("standard_operator", reference_path),
        ):
            benchmarks.append(
                run_benchmark(
                    graph=graph,
                    model_path=model,
                    library_path=library_path,
                    inputs=benchmark_input,
                    benchmark=config["benchmark"],
                    preset=config["preset"],
                )
            )
        custom_benchmark = benchmarks[0]
        historical = historical_comparison(
            resolve_path(config["historical_v13_report"]), custom_benchmark
        )
        decisions = validation_decisions(
            exactness_passed=exactness["passed"],
            profile=profile["analysis"],
            benchmark_completed=True,
            historical_protocol_matched=historical["protocol"]["directly_protocol_comparable"],
            historical_typical_latency_reproduced=historical["reproduction"]["typical_latency_reproduced"],
            historical_variance_reproduced=historical["reproduction"]["variance_reproduced"],
        )
        report = {
            "schema_version": REPORT_SCHEMA,
            "completion_status": "success",
            "scope": SCOPE,
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "preset": config["preset"],
            "preset_fingerprint": preset_fingerprint(config["preset"]),
            "machine": machine_facts(runtime_dll),
            "provenance": provenance,
            "exactness_gate": exactness,
            "profile": profile,
            "benchmarks": benchmarks,
            "historical_comparison": historical,
            "decisions": decisions,
        }
        print("v1.4 phase 4/4: write evidence", flush=True)
        write_reports(output_root, report)
        atomic_json(output_root / "run_status.json", {"status": "complete"})
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
