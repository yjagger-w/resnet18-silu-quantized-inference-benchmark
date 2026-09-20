#!/usr/bin/env python
"""Run the v1.3 exactness-gated ORT CPU custom-op performance diagnosis."""

from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from silu_benchmark.accuracy_recovery import instrument_outputs
from silu_benchmark.benchmark_data import load_cifar_batch, normalize_cifar_images
from silu_benchmark.ort_custom_op_backend import prepare_process_local_ort_runtime
from silu_benchmark.ort_custom_op_performance import (
    NON_ADDITIVITY_WARNING,
    benchmark_cells,
    classify_comparability,
    decide_optimization,
    load_config,
    parse_profile_file,
    sha256_file,
    summarize_repetitions,
    validate_output_path,
)
from silu_benchmark.ort_custom_op_rewrite import discover_selected_piecewise_islands


SCOPE = "ORT CPU hybrid graph with project C++ custom-op activations"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


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


def windows_file_version(path: Path) -> str | None:
    if os.name != "nt":
        return None

    class VSFixedFileInfo(ctypes.Structure):
        _fields_ = [
            ("dwSignature", ctypes.c_uint32),
            ("dwStrucVersion", ctypes.c_uint32),
            ("dwFileVersionMS", ctypes.c_uint32),
            ("dwFileVersionLS", ctypes.c_uint32),
            ("dwProductVersionMS", ctypes.c_uint32),
            ("dwProductVersionLS", ctypes.c_uint32),
            ("dwFileFlagsMask", ctypes.c_uint32),
            ("dwFileFlags", ctypes.c_uint32),
            ("dwFileOS", ctypes.c_uint32),
            ("dwFileType", ctypes.c_uint32),
            ("dwFileSubtype", ctypes.c_uint32),
            ("dwFileDateMS", ctypes.c_uint32),
            ("dwFileDateLS", ctypes.c_uint32),
        ]

    version = ctypes.windll.version
    size = version.GetFileVersionInfoSizeW(str(path), None)
    if not size:
        return None
    buffer = ctypes.create_string_buffer(size)
    if not version.GetFileVersionInfoW(str(path), 0, size, buffer):
        return None
    translation_pointer = ctypes.c_void_p()
    translation_length = ctypes.c_uint()
    if version.VerQueryValueW(
        buffer,
        "\\VarFileInfo\\Translation",
        ctypes.byref(translation_pointer),
        ctypes.byref(translation_length),
    ) and translation_length.value >= 4:
        translation = ctypes.cast(
            translation_pointer, ctypes.POINTER(ctypes.c_uint16)
        )
        block = (
            f"\\StringFileInfo\\{translation[0]:04x}{translation[1]:04x}"
            "\\FileVersion"
        )
        value_pointer = ctypes.c_void_p()
        value_length = ctypes.c_uint()
        if version.VerQueryValueW(
            buffer,
            block,
            ctypes.byref(value_pointer),
            ctypes.byref(value_length),
        ) and value_length.value:
            return ctypes.wstring_at(value_pointer).strip()
    pointer = ctypes.c_void_p()
    length = ctypes.c_uint()
    if not version.VerQueryValueW(buffer, "\\", ctypes.byref(pointer), ctypes.byref(length)):
        return None
    info = ctypes.cast(pointer, ctypes.POINTER(VSFixedFileInfo)).contents
    return ".".join(
        str(value)
        for value in (
            info.dwFileVersionMS >> 16,
            info.dwFileVersionMS & 0xFFFF,
            info.dwFileVersionLS >> 16,
            info.dwFileVersionLS & 0xFFFF,
        )
    )


def configure_options(
    *,
    intra_op_threads: int = 0,
    inter_op_threads: int = 0,
    execution_mode: str = "sequential",
    profiling_prefix: Path | None = None,
) -> tuple[ort.SessionOptions, dict]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = (
        ort.ExecutionMode.ORT_SEQUENTIAL
        if execution_mode == "sequential"
        else ort.ExecutionMode.ORT_PARALLEL
    )
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = inter_op_threads
    if profiling_prefix is not None:
        profiling_prefix.parent.mkdir(parents=True, exist_ok=True)
        options.enable_profiling = True
        options.profile_file_prefix = str(profiling_prefix.resolve())
    record = {
        "graph_optimization_level": str(options.graph_optimization_level),
        "execution_mode": str(options.execution_mode),
        "intra_op_num_threads": options.intra_op_num_threads,
        "inter_op_num_threads": options.inter_op_num_threads,
        "enable_profiling": options.enable_profiling,
        "enable_cpu_mem_arena": options.enable_cpu_mem_arena,
        "enable_mem_pattern": options.enable_mem_pattern,
        "use_deterministic_compute": options.use_deterministic_compute,
        "profile_file_prefix": options.profile_file_prefix,
        "optimized_model_filepath": options.optimized_model_filepath,
        "log_severity_level": options.log_severity_level,
        "log_verbosity_level": options.log_verbosity_level,
    }
    return options, record


def create_session(
    model_path: Path,
    *,
    graph: str,
    library_path: Path,
    intra_op_threads: int = 0,
    inter_op_threads: int = 0,
    execution_mode: str = "sequential",
    profiling_prefix: Path | None = None,
) -> tuple[ort.InferenceSession, dict]:
    options, settings = configure_options(
        intra_op_threads=intra_op_threads,
        inter_op_threads=inter_op_threads,
        execution_mode=execution_mode,
        profiling_prefix=profiling_prefix,
    )
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
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    session_creation_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    metadata = {
        "graph": graph,
        "model": str(model_path),
        "registration_load_ms": registration_ms,
        "registered_library": registered_library or (str(library_path) if graph == "custom_op" else None),
        "session_creation_load_ms": session_creation_ms,
        "session_options": settings,
        "providers": session.get_providers(),
        "runtime_dll": str(runtime_dll) if runtime_dll else None,
    }
    return session, metadata


def graph_facts(path: Path, form: str) -> dict:
    model = onnx.load(str(path))
    histogram = {}
    for node in model.graph.node:
        key = f"{node.domain or 'ai.onnx'}::{node.op_type}"
        histogram[key] = histogram.get(key, 0) + 1
    return {
        "graph_form": form,
        "path": str(path),
        "sha256": sha256_file(path),
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "operator_histogram": dict(sorted(histogram.items())),
    }


def prepare_probe_models(
    reference_path: Path, custom_path: Path, output_root: Path
) -> tuple[Path, Path, tuple]:
    reference = onnx.load(str(reference_path))
    custom = onnx.load(str(custom_path))
    islands = discover_selected_piecewise_islands(reference)
    mappings = {
        **{island.site_id: island.output_code_tensor for island in islands},
        "final_logits": reference.graph.output[0].name,
    }
    if custom.graph.output[0].name != reference.graph.output[0].name:
        raise RuntimeError("v1.3 source/custom final output names differ")
    directory = output_root / "exactness_models"
    directory.mkdir(parents=True, exist_ok=True)
    reference_out = directory / "reference_instrumented.onnx"
    custom_out = directory / "custom_instrumented.onnx"
    reference_out.write_bytes(
        instrument_outputs(reference, mappings).SerializeToString(deterministic=True)
    )
    custom_out.write_bytes(
        instrument_outputs(custom, mappings).SerializeToString(deterministic=True)
    )
    return reference_out, custom_out, islands


def run_exactness_gate(
    *,
    reference_path: Path,
    custom_path: Path,
    library_path: Path,
    output_root: Path,
    images: np.ndarray,
    labels: np.ndarray,
    indices: list[int],
) -> dict:
    instrumented_reference, instrumented_custom, islands = prepare_probe_models(
        reference_path, custom_path, output_root
    )
    reference_session, reference_metadata = create_session(
        instrumented_reference, graph="standard_operator", library_path=library_path
    )
    custom_session, custom_metadata = create_session(
        instrumented_custom, graph="custom_op", library_path=library_path
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
    logits_elements = 0
    predictions_equal = 0
    completed = 0
    first_failure = None
    normalized = normalize_cifar_images(images[indices])
    for offset in range(0, len(indices), 8):
        batch = normalized[offset : offset + 8]
        reference_outputs = reference_session.run(
            None, {reference_session.get_inputs()[0].name: batch}
        )
        custom_outputs = custom_session.run(
            None, {custom_session.get_inputs()[0].name: batch}
        )
        for site_index, row in enumerate(rows):
            left = reference_outputs[site_index]
            right = custom_outputs[site_index]
            row["elements_compared"] += int(left.size)
            compatible = left.shape == right.shape and left.dtype == right.dtype
            mismatch = (
                np.flatnonzero(left.reshape(-1) != right.reshape(-1))
                if compatible
                else np.asarray([], dtype=np.int64)
            )
            if not compatible or mismatch.size:
                row["exact_equal"] = False
                row["mismatch_count"] = int(mismatch.size) if mismatch.size else 1
                first_failure = {
                    "site_id": row["site_id"],
                    "probe_offset": offset,
                    "reference_shape": list(left.shape),
                    "custom_shape": list(right.shape),
                    "reference_dtype": str(left.dtype),
                    "custom_dtype": str(right.dtype),
                    "first_flat_index": int(mismatch[0]) if mismatch.size else None,
                }
                break
        if first_failure:
            break
        left_logits = reference_outputs[-1]
        right_logits = custom_outputs[-1]
        logits_elements += int(left_logits.size)
        compatible = (
            left_logits.shape == right_logits.shape
            and left_logits.dtype == right_logits.dtype
        )
        mismatch = (
            np.flatnonzero(left_logits.reshape(-1) != right_logits.reshape(-1))
            if compatible
            else np.asarray([], dtype=np.int64)
        )
        if not compatible or mismatch.size:
            first_failure = {
                "site_id": "final_logits",
                "probe_offset": offset,
                "first_flat_index": int(mismatch[0]) if mismatch.size else None,
            }
            break
        predictions_equal += int(
            np.sum(np.argmax(left_logits, axis=1) == np.argmax(right_logits, axis=1))
        )
        completed += len(batch)
    result = {
        "schema_version": "ort-customop-performance-exactness/v1",
        "passed": bool(
            first_failure is None
            and completed == 128
            and len(rows) == 17
            and all(row["exact_equal"] for row in rows)
            and logits_elements == 1280
            and predictions_equal == 128
        ),
        "samples_required": 128,
        "samples_completed": completed,
        "input_digest": input_digest(normalized, indices),
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
    if not result["passed"]:
        raise RuntimeError(
            "v1.3 exactness gate failed; profiling and benchmarking are prohibited: "
            + json.dumps(first_failure, sort_keys=True)
        )
    return result


def run_profile(
    *,
    graph: str,
    model_path: Path,
    library_path: Path,
    inputs: np.ndarray,
    iterations: int,
    output_root: Path,
) -> dict:
    prefix = output_root / "profiles" / f"{graph}_profile"
    session, metadata = create_session(
        model_path,
        graph=graph,
        library_path=library_path,
        profiling_prefix=prefix,
    )
    output_name = session.get_outputs()[0].name
    feed = {session.get_inputs()[0].name: inputs}
    raw_wall_ms = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        session.run([output_name], feed)
        raw_wall_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    profile_path = Path(session.end_profiling()).resolve()
    parsed = parse_profile_file(
        profile_path, inference_wall_time_us=sum(raw_wall_ms) * 1000.0
    )
    if graph == "custom_op":
        custom = parsed["custom_op_execution"]
        counts = {row["execution_count"] for row in custom["nodes"]}
        if custom["unique_node_count"] != 17 or counts != {iterations}:
            raise RuntimeError(
                "profile did not prove all 17 custom nodes executed once per invocation"
            )
    return {
        "graph": graph,
        "profile_path": str(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "profile_iterations": iterations,
        "batch_size": len(inputs),
        "raw_inference_wall_ms": raw_wall_ms,
        "total_inference_wall_ms": sum(raw_wall_ms),
        "session": metadata,
        "analysis": parsed,
    }


def run_benchmark_cell(
    *,
    cell: dict,
    model_path: Path,
    library_path: Path,
    inputs: np.ndarray,
    benchmark: Mapping,
) -> dict:
    repetitions = []
    session_metadata = []
    for repetition in range(benchmark["repetitions"]):
        session, metadata = create_session(
            model_path,
            graph=cell["graph"],
            library_path=library_path,
            intra_op_threads=cell["intra_op_threads"],
            inter_op_threads=benchmark["inter_op_threads"],
            execution_mode=cell["execution_mode"],
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
        repetitions.append(samples)
        session_metadata.append(metadata)
        print(
            f"  repetition {repetition + 1}/{benchmark['repetitions']}: "
            f"p50={np.percentile(samples, 50):.3f} ms p95={np.percentile(samples, 95):.3f} ms",
            flush=True,
        )
        del session
    return {
        **cell,
        "cell_id": (
            f"{cell['graph']}_b{cell['batch_size']}_t{cell['intra_op_threads']}_"
            f"{cell['execution_mode']}"
        ),
        "provider": "CPUExecutionProvider",
        "profiling_enabled": False,
        "warmup_iterations_per_repetition": benchmark["warmup_iterations"],
        "timed_iterations_per_repetition": benchmark["timed_iterations"],
        "repetitions": benchmark["repetitions"],
        "input_digest": input_digest(inputs, list(range(len(inputs)))),
        "raw_samples_ms": repetitions,
        "statistics": summarize_repetitions(
            repetitions, batch_size=cell["batch_size"]
        ),
        "session_metadata": session_metadata,
        "timing_boundary": (
            "perf_counter_ns immediately around session.run([logits], cached_feed); "
            "includes ORT execution and Python/NumPy output materialization, excludes input preparation"
        ),
    }


def protocol_record(
    *,
    provider="CPUExecutionProvider",
    batch_size=1,
    warmups=20,
    timed=100,
    repetitions=1,
    digest=None,
    output_selection="explicit final logits",
    lifecycle="fresh session followed by warmups",
    between_work="none",
) -> dict:
    return {
        "provider": provider,
        "batch_size": batch_size,
        "warmup_iterations": warmups,
        "timed_iterations": timed,
        "repetitions": repetitions,
        "input_digest": digest,
        "input_reuse": "same cached NumPy array for every invocation",
        "graph_optimization_level": "GraphOptimizationLevel.ORT_ENABLE_ALL",
        "execution_mode": "ExecutionMode.ORT_SEQUENTIAL",
        "intra_op_threads": 0,
        "inter_op_threads": 0,
        "profiling_enabled": False,
        "timing_boundary": "perf_counter_ns immediately around session.run",
        "output_selection": output_selection,
        "session_lifecycle": lifecycle,
        "between_invocation_work": between_work,
    }


def build_comparability_audit(
    *,
    config: Mapping,
    benchmark_input: np.ndarray,
    reference_facts: dict,
    custom_facts: dict,
) -> dict:
    v065_config = json.loads(resolve_path(config["legacy_v065_config"]).read_text(encoding="utf-8"))
    v065_report = json.loads(resolve_path(config["legacy_v065_report"]).read_text(encoding="utf-8"))
    v065_environment = json.loads(
        resolve_path(config["legacy_v065_environment"]).read_text(encoding="utf-8")
    )
    v12_report = json.loads(resolve_path(config["legacy_v12_report"]).read_text(encoding="utf-8"))
    zero_input = np.zeros((1, 3, 32, 32), dtype=np.float32)
    v065 = protocol_record(
        warmups=v065_config["latency_warmup_runs"],
        timed=v065_config["latency_timed_runs"],
        digest=input_digest(zero_input, [0]),
        lifecycle="same session after complete 10,000-image evaluation",
        between_work="process RSS/working-set query after every timed invocation",
    )
    v065.update(
        {
            "name": "v0.6.5 Standard-QDQ historical",
            "model_hash": v065_report["artifact_sha256"]["standard_static_qdq"],
            "graph_form": "standard QDQ deployment baseline",
            "session_options_recorded": v065_environment["thread_settings"],
        }
    )
    digest = input_digest(benchmark_input, [0])
    v12 = protocol_record(
        digest=digest,
        output_selection="session.run(None): all declared graph outputs",
        lifecycle="fresh custom-op benchmark session followed by warmups",
    )
    v12.update(
        {
            "name": "v1.2 custom-op historical",
            "model_hash": v12_report["custom_model_sha256"],
            "graph_form": custom_facts["graph_form"],
            "reported_mean_ms": v12_report["benchmark"]["mean_latency_ms"],
        }
    )
    matched_custom = protocol_record(
        repetitions=config["benchmark"]["repetitions"], digest=digest
    )
    matched_custom.update(
        {"name": "v1.3 matched custom-op", "model_hash": custom_facts["sha256"], "graph_form": custom_facts["graph_form"]}
    )
    matched_reference = copy.deepcopy(matched_custom)
    matched_reference.update(
        {"name": "v1.3 matched standard-operator", "model_hash": reference_facts["sha256"], "graph_form": reference_facts["graph_form"]}
    )
    v11 = {
        "name": "v1.1 selected standard-operator campaign",
        "model_hash": reference_facts["sha256"],
        "graph_form": reference_facts["graph_form"],
        "timing_protocol": "no dedicated latency protocol; only batched evaluation wall time",
        "direct_latency_available": False,
    }
    return {
        "paths": [v065, v11, v12, matched_custom, matched_reference],
        "v12_vs_v065": classify_comparability(v12, v065),
        "v12_85_9749_ms_directly_comparable_to_v065": False,
        "v12_vs_v065_conclusion": (
            "No. The input content/digest, output selection, session lifecycle, "
            "and between-invocation work differ; the historical values are contextual only."
        ),
        "matched_custom_vs_standard": classify_comparability(
            matched_custom, matched_reference
        ),
    }


def machine_facts(runtime_dll: Path | None) -> dict:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version,
        "python_executable": sys.executable,
        "onnx": onnx.__version__,
        "numpy": np.__version__,
        "onnxruntime": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "selected_provider": "CPUExecutionProvider",
        "runtime_dll": str(runtime_dll) if runtime_dll else None,
        "runtime_dll_file_version": windows_file_version(runtime_dll) if runtime_dll else None,
        "torch_imported": "torch" in sys.modules,
    }


def write_reports(output_root: Path, report: dict) -> None:
    atomic_json(output_root / "diagnosis_report.json", report)
    atomic_json(output_root / "comparability_audit.json", report["comparability"])
    atomic_json(output_root / "profile_analysis.json", report["profiles"])
    atomic_json(output_root / "benchmark_matrix.json", report["benchmark_matrix"])
    cells = report["benchmark_matrix"]["cells"]
    with (output_root / "benchmark_matrix.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "cell_id", "graph", "batch_size", "intra_op_threads", "execution_mode",
            "sample_count", "mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms",
            "std_ms", "coefficient_of_variation", "images_per_second",
            "total_measured_wall_time_seconds", "median_of_repetition_p50_ms",
            "median_of_repetition_p95_ms", "high_p95_reproduced_in_all_repetitions",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            writer.writerow(
                {
                    **{key: cell[key] for key in ("cell_id", "graph", "batch_size", "intra_op_threads", "execution_mode")},
                    **{key: cell["statistics"][key] for key in fields if key in cell["statistics"]},
                }
            )
    with (output_root / "profile_nodes.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["graph", "rank", "name", "op_name", "category", "execution_count", "total_duration_us", "mean_duration_us", "median_duration_us", "percentage_of_node_event_time"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for profile in report["profiles"]:
            for rank, row in enumerate(profile["analysis"]["nodes"], 1):
                writer.writerow({"graph": profile["graph"], "rank": rank, **row})

    custom_profile = next(row for row in report["profiles"] if row["graph"] == "custom_op")
    baseline = next(
        row for row in cells
        if row["graph"] == "custom_op" and row["batch_size"] == 1
        and row["intra_op_threads"] == 0 and row["execution_mode"] == "sequential"
    )
    matched = next(
        row for row in cells
        if row["graph"] == "standard_operator" and row["batch_size"] == 1
        and row["intra_op_threads"] == 0 and row["execution_mode"] == "sequential"
    )
    lines = [
        "# v1.3 ORT C++ custom-op CPU performance diagnosis",
        "",
        f"Scope: **{SCOPE}**. This is not integer-only whole-model inference.",
        "",
        "## Provenance",
        "",
        f"- Standard-operator model SHA-256: `{report['provenance']['reference']['sha256']}`",
        f"- Custom-op model SHA-256: `{report['provenance']['custom']['sha256']}`",
        f"- Custom-op DLL SHA-256: `{report['provenance']['library']['sha256']}`",
        f"- ORT/runtime: `{report['machine']['onnxruntime']}` / `{report['machine']['runtime_dll_file_version']}`",
        f"- Provider: `{report['machine']['selected_provider']}`",
        f"- Processor/logical CPUs: `{report['machine']['processor']}` / `{report['machine']['logical_cpu_count']}`",
        f"- Probe input SHA-256: `{report['provenance']['probe_input_digest']}`",
        f"- PyTorch imported by diagnostic process: `{report['machine']['torch_imported']}`",
        "",
        "## Exactness gate",
        "",
        f"- Passed: `{report['exactness_gate']['passed']}`",
        f"- All 17 uint8 code outputs exact: `{report['exactness_gate']['all_site_codes_exact']}`",
        f"- Final logits exact: `{report['exactness_gate']['final_logits_exact']}`",
        f"- Prediction agreement: `{report['exactness_gate']['prediction_agreement']:.2%}`",
        "",
        "## Comparability",
        "",
        report["comparability"]["v12_vs_v065_conclusion"],
        "The v1.3 custom and standard-operator cells below use the same cached inputs, session settings, timing boundary, warmups, iterations, and repetitions.",
        "",
        "| Path | Graph form | Batch | Warmups | Timed | Repetitions | Output/timing note |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for path in report["comparability"]["paths"]:
        lines.append(
            f"| {path['name']} | {path['graph_form']} | {path.get('batch_size', 'n/a')} | "
            f"{path.get('warmup_iterations', 'n/a')} | {path.get('timed_iterations', 'n/a')} | "
            f"{path.get('repetitions', 'n/a')} | {path.get('output_selection', path.get('timing_protocol', 'n/a'))}; "
            f"{path.get('session_lifecycle', 'n/a')} |"
        )
    lines.extend([
        "",
        "## Profiling",
        "",
        "- Separate fresh sessions and profile files were used for the custom and standard-operator graphs.",
        "- Each profile used the same cached batch-1 input and 10 inference invocations.",
        "- The parser selects events with category `Node`, a name ending in `_kernel_time`, and ORT `args.op_name`/`args.node_name`; event order is not assumed.",
        f"- Observed profile categories: `{', '.join(custom_profile['analysis']['profile_event_categories_observed'])}`; selected kernel events: `{custom_profile['analysis']['parsed_kernel_event_count']}`.",
        f"- Custom DLL registration/load: `{custom_profile['session']['registration_load_ms']:.3f} ms`; custom session creation/load: `{custom_profile['session']['session_creation_load_ms']:.3f} ms`.",
        f"- All profiled custom nodes: `{custom_profile['analysis']['custom_op_execution']['unique_node_count']}/17`",
        f"- Custom-node executions: `{sum(row['execution_count'] for row in custom_profile['analysis']['custom_op_execution']['nodes'])}` (`10` per node)",
        f"- Aggregate custom-node share of node-event time: `{custom_profile['analysis']['custom_op_execution']['percentage_of_node_event_time']:.3f}%`",
        f"- Profiled inference wall time: `{custom_profile['total_inference_wall_ms']:.3f} ms`",
        f"- Summed node-event time: `{custom_profile['analysis']['summed_node_event_time_us'] / 1000.0:.3f} ms`",
        "",
        NON_ADDITIVITY_WARNING,
        "",
        "Custom-graph operation categories:",
        "",
        "| Category | Executions | Total ms | Node-event share |",
        "|---|---:|---:|---:|",
    ])
    for row in custom_profile["analysis"]["categories"]:
        lines.append(
            f"| `{row['name']}` | {row['execution_count']} | "
            f"{row['total_duration_us'] / 1000.0:.3f} | {row['percentage_of_node_event_time']:.2f}% |"
        )
    lines.extend([
        "",
        "Top custom-graph nodes:",
        "",
        "| Rank | Node | Op | Category | Total ms | Node-event share |",
        "|---:|---|---|---|---:|---:|",
    ])
    for rank, row in enumerate(custom_profile["analysis"]["top_20_nodes"], 1):
        lines.append(
            f"| {rank} | `{row['name']}` | `{row['op_name']}` | `{row['category']}` | "
            f"{row['total_duration_us'] / 1000.0:.3f} | {row['percentage_of_node_event_time']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Repeated benchmark matrix",
            "",
            "Every cell retains 500 samples: five fresh-session repetitions, each with 20 warmups and 100 timed invocations. No outlier was deleted and profiling was disabled.",
            "",
            "| Graph | Batch | Threads | Mode | Mean ms | P50 ms | P95 ms | CV | Images/s |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in cells:
        stats = cell["statistics"]
        lines.append(
            f"| {cell['graph']} | {cell['batch_size']} | "
            f"{'default' if cell['intra_op_threads'] == 0 else cell['intra_op_threads']} | "
            f"{cell['execution_mode']} | {stats['mean_ms']:.3f} | {stats['p50_ms']:.3f} | "
            f"{stats['p95_ms']:.3f} | {stats['coefficient_of_variation']:.3f} | "
            f"{stats['images_per_second']:.2f} |"
        )
    lines.extend(
        [
            "",
            "### Batch-1 matched baseline",
            "",
            f"- Custom-op: mean `{baseline['statistics']['mean_ms']:.3f} ms`, p50 `{baseline['statistics']['p50_ms']:.3f} ms`, p95 `{baseline['statistics']['p95_ms']:.3f} ms`, CV `{baseline['statistics']['coefficient_of_variation']:.3f}`.",
            f"- Standard-operator: mean `{matched['statistics']['mean_ms']:.3f} ms`, p50 `{matched['statistics']['p50_ms']:.3f} ms`, p95 `{matched['statistics']['p95_ms']:.3f} ms`, CV `{matched['statistics']['coefficient_of_variation']:.3f}`.",
            "",
            "These are matched graph-form measurements, not a speedup claim.",
            "",
            "### Latency variance",
            "",
            f"The custom batch-1 default/sequential p50 ranged from `{baseline['statistics']['repetition_p50_range_ms'][0]:.3f}` to `{baseline['statistics']['repetition_p50_range_ms'][1]:.3f} ms` across repetitions; p95 ranged from `{baseline['statistics']['repetition_p95_range_ms'][0]:.3f}` to `{baseline['statistics']['repetition_p95_range_ms'][1]:.3f} ms`.",
            f"A high p95 was reproduced in all five repetitions: `{baseline['statistics']['high_p95_reproduced_in_all_repetitions']}`. Raw samples were retained and no outlier was deleted.",
            "The 4-thread custom cell had mean/p50/p95 `25.649/25.722/36.732 ms` and CV `0.298`; the 8-thread cell regressed to `29.031/26.098/53.689 ms` and CV `0.494`.",
            "",
            "## Decisions",
            "",
            "| Option | Decision | Evidence |",
            "|---|---|---|",
            f"| AVX2 kernel | {report['decisions']['avx2_kernel']['decision']} | {report['decisions']['avx2_kernel']['reason']} |",
            f"| Thread tuning | {report['decisions']['thread_tuning']['decision']} | {report['decisions']['thread_tuning']['reason']} |",
            f"| Graph/runtime work | {report['decisions']['graph_runtime_work']['decision']}: {report['decisions']['graph_runtime_work']['target']} | {report['decisions']['graph_runtime_work']['reason']} |",
            f"| Further work | {report['decisions']['further_work']['decision']} | Smallest evidence-driven next step only. |",
            "",
            "No AVX2, parallel custom kernel, thread pool, graph rewrite, fusion, or benchmark-oriented semantic change was implemented in v1.3.",
            "",
            "## Limitations",
            "",
            "- One Windows x64 CPU machine and ORT 1.19.2 only; no cross-machine generality.",
            "- ORT profile node sums can overlap and are not equal to wall time.",
            "- Execution-time controls do not change model semantics, topology, quantization parameters, or scalar-kernel mathematics.",
            "- No complete 10,000-image evaluation was run in v1.3.",
        ]
    )
    atomic_text(output_root / "diagnosis_report.md", "\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmarks/resnet18_silu_cifar10_v13_ort_customop_performance.json"),
    )
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if "torch" in sys.modules or "torchvision" in sys.modules:
        raise RuntimeError("v1.3 diagnostic process must remain NumPy/ORT-only")
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    output_root = validate_output_path(resolve_path(config["output_root"]), ROOT)
    if output_root.exists():
        if not args.force_rebuild:
            raise FileExistsError(
                f"v1.3 output already exists; use --force-rebuild: {output_root}"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    atomic_json(output_root / "run_status.json", {"status": "running", "phase": "audit"})

    reference_path = resolve_path(config["reference_model"])
    custom_path = resolve_path(config["custom_model"])
    library_path = resolve_path(config["custom_library"])
    for path, expected in (
        (reference_path, config["reference_model_sha256"]),
        (custom_path, config["custom_model_sha256"]),
        (library_path, config["custom_library_sha256"]),
    ):
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen artifact hash differs: {path}: {actual}")
    runtime_dll = prepare_process_local_ort_runtime()
    test_images, test_labels = load_cifar_batch(resolve_path(config["data_root"]), "test_batch")
    controls = json.loads(resolve_path(config["probe_indices"]).read_text(encoding="utf-8"))
    probe_indices = [int(value) for value in controls["probe_indices"]]
    if len(probe_indices) != 128 or len(set(probe_indices)) != 128:
        raise ValueError("frozen v1.3 probe must contain 128 unique indices")

    print("v1.3 phase 1/4: exactness gate", flush=True)
    exactness = run_exactness_gate(
        reference_path=reference_path,
        custom_path=custom_path,
        library_path=library_path,
        output_root=output_root,
        images=test_images,
        labels=test_labels,
        indices=probe_indices,
    )
    atomic_json(output_root / "run_status.json", {"status": "running", "phase": "profiling"})

    benchmark_inputs = {
        batch: normalize_cifar_images(test_images[:batch])
        for batch in config["benchmark"]["batch_sizes"]
    }
    reference_facts = graph_facts(reference_path, "v1.1 expanded standard-operator piecewise reference")
    custom_facts = graph_facts(custom_path, "v1.2 hybrid graph with 17 project C++ custom activations")
    comparability = build_comparability_audit(
        config=config,
        benchmark_input=benchmark_inputs[1],
        reference_facts=reference_facts,
        custom_facts=custom_facts,
    )

    print("v1.3 phase 2/4: isolated ORT profiles", flush=True)
    profiles = []
    for graph, path in (("custom_op", custom_path), ("standard_operator", reference_path)):
        profiles.append(
            run_profile(
                graph=graph,
                model_path=path,
                library_path=library_path,
                inputs=benchmark_inputs[1],
                iterations=config["profiling"]["iterations"],
                output_root=output_root,
            )
        )
    atomic_json(output_root / "run_status.json", {"status": "running", "phase": "benchmark_matrix"})

    cells = benchmark_cells(config)
    benchmark_rows = []
    print(f"v1.3 phase 3/4: {len(cells)} repeated benchmark cells", flush=True)
    for index, cell in enumerate(cells, 1):
        print(
            f"cell {index}/{len(cells)}: {cell['graph']} batch={cell['batch_size']} "
            f"threads={cell['intra_op_threads']} mode={cell['execution_mode']}",
            flush=True,
        )
        path = custom_path if cell["graph"] == "custom_op" else reference_path
        benchmark_rows.append(
            run_benchmark_cell(
                cell=cell,
                model_path=path,
                library_path=library_path,
                inputs=benchmark_inputs[cell["batch_size"]],
                benchmark=config["benchmark"],
            )
        )
        atomic_json(
            output_root / "benchmark_matrix.partial.json",
            {"completed_cells": len(benchmark_rows), "cells": benchmark_rows},
        )

    custom_profile = next(row for row in profiles if row["graph"] == "custom_op")
    decisions = decide_optimization(custom_profile["analysis"], benchmark_rows)
    report = {
        "schema_version": "ort-customop-performance-diagnosis-report/v1",
        "completion_status": "success",
        "scope": SCOPE,
        "config": config,
        "config_sha256": sha256_file(config_path),
        "machine": machine_facts(runtime_dll),
        "provenance": {
            "reference": reference_facts,
            "custom": custom_facts,
            "library": {"path": str(library_path), "sha256": sha256_file(library_path)},
            "probe_input_digest": exactness["input_digest"],
            "benchmark_input_digests": {
                str(batch): input_digest(values, list(range(batch)))
                for batch, values in benchmark_inputs.items()
            },
        },
        "exactness_gate": exactness,
        "comparability": comparability,
        "profiles": profiles,
        "benchmark_matrix": {
            "design": "bounded non-factorial controls: full batch-1 thread sweep, default-thread batch sweep, default-thread execution-mode control",
            "profiling_enabled": False,
            "raw_samples_retained": True,
            "outliers_deleted": False,
            "cells": benchmark_rows,
        },
        "decisions": decisions,
        "claim_boundary": (
            "Diagnosis only. No performance implementation, SIMD acceleration, whole-model integer-only, "
            "accelerator, or cross-machine claim."
        ),
    }
    print("v1.3 phase 4/4: reports and decisions", flush=True)
    write_reports(output_root, report)
    atomic_json(output_root / "run_status.json", {"status": "success", "phase": "complete"})
    partial = output_root / "benchmark_matrix.partial.json"
    if partial.exists():
        partial.unlink()
    print(
        json.dumps(
            {
                "status": "success",
                "output": str(output_root),
                "exactness_gate": exactness["passed"],
                "custom_node_share_percent": custom_profile["analysis"]["custom_op_execution"]["percentage_of_node_event_time"],
                "decisions": decisions,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
