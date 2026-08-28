"""Contract and evidence helpers for the explicit v1.4 ORT CPU preset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import onnxruntime as ort

from silu_benchmark.ort_custom_op_performance import (
    classify_comparability,
    parse_profile_events,
    summarize_repetitions,
)


CONFIG_SCHEMA = "ort-customop-thread-preset-validation/v1"
PRESET_SCHEMA = "ort-runtime-preset/v1"
REPORT_SCHEMA = "ort-customop-thread-preset-report/v1"
EXPECTED_SITE_COUNT = 17
EXPECTED_ORT_VERSION = "1.19.2"
CUSTOM_SHARE_PRIORITY_PERCENT = 10.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_preset(preset: Mapping) -> dict:
    return {
        "schema_version": preset.get("schema_version"),
        "name": preset.get("name"),
        "version": preset.get("version"),
        "provider": preset.get("provider"),
        "execution_mode": preset.get("execution_mode"),
        "intra_op_num_threads": preset.get("intra_op_num_threads"),
        "inter_op_num_threads": preset.get("inter_op_num_threads"),
        "inter_op_behavior": preset.get("inter_op_behavior"),
        "graph_optimization_level": preset.get("graph_optimization_level"),
        "intended_workload": preset.get("intended_workload"),
        "scope_boundary": preset.get("scope_boundary"),
    }


def validate_preset(preset: Mapping) -> dict:
    if preset.get("schema_version") != PRESET_SCHEMA:
        raise ValueError("invalid v1.4 runtime-preset schema")
    if preset.get("name") != "resnet18-silu-cifar10-ort-customop-cpu-4threads":
        raise ValueError("unexpected v1.4 preset name")
    if preset.get("version") != "1.4":
        raise ValueError("unexpected v1.4 preset version")
    if preset.get("provider") != "CPUExecutionProvider":
        raise ValueError("v1.4 preset supports CPUExecutionProvider only")
    if preset.get("execution_mode") != "ORT_SEQUENTIAL":
        raise ValueError("v1.4 preset requires ORT_SEQUENTIAL")
    threads = preset.get("intra_op_num_threads")
    if not isinstance(threads, int) or isinstance(threads, bool) or threads <= 0:
        raise ValueError("intra_op_num_threads must be a positive integer")
    if threads != 4:
        raise ValueError("v1.4 preset is frozen to four intra-op threads")
    if preset.get("inter_op_num_threads") != 0:
        raise ValueError("v1.4 keeps inter-op threads at ORT default 0")
    if preset.get("inter_op_behavior") != "ORT default; unused by ORT_SEQUENTIAL":
        raise ValueError("v1.4 inter-op behavior must be documented explicitly")
    if preset.get("graph_optimization_level") != "ORT_ENABLE_ALL":
        raise ValueError("v1.4 requires ORT_ENABLE_ALL")
    for key in ("intended_workload", "scope_boundary"):
        if not isinstance(preset.get(key), str) or not preset[key].strip():
            raise ValueError(f"missing preset field: {key}")
    return canonical_preset(preset)


def preset_fingerprint(preset: Mapping) -> str:
    validated = validate_preset(preset)
    encoded = json.dumps(
        validated, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_session_options(
    preset: Mapping,
    *,
    profile_prefix: Path | None = None,
) -> tuple[ort.SessionOptions, dict]:
    validated = validate_preset(preset)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = validated["intra_op_num_threads"]
    options.inter_op_num_threads = validated["inter_op_num_threads"]
    if profile_prefix is not None:
        profile_prefix = Path(profile_prefix).resolve()
        profile_prefix.parent.mkdir(parents=True, exist_ok=True)
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    return options, session_options_record(options)


def session_options_record(options: ort.SessionOptions) -> dict:
    return {
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


def validate_config(payload: Mapping) -> dict:
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("invalid v1.4 preset-validation schema")
    if payload.get("ort_version") != EXPECTED_ORT_VERSION:
        raise ValueError("v1.4 is frozen to ONNX Runtime 1.19.2")
    preset = validate_preset(payload.get("preset", {}))
    for key in (
        "reference_model",
        "custom_model",
        "custom_library",
        "data_root",
        "probe_indices",
        "historical_v13_report",
        "output_root",
    ):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"missing v1.4 path setting: {key}")
    for key in (
        "reference_model_sha256",
        "custom_model_sha256",
        "custom_library_sha256",
    ):
        value = payload.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"invalid frozen SHA-256: {key}")
    exactness = payload.get("exactness", {})
    if exactness != {"samples": 128, "batch_size": 8}:
        raise ValueError("v1.4 exactness gate requires 128 samples in batches of 8")
    profiling = payload.get("profiling", {})
    if profiling.get("batch_size") != 1 or profiling.get("iterations") != 10:
        raise ValueError("v1.4 profile requires 10 batch-1 inferences")
    benchmark = payload.get("benchmark", {})
    if benchmark != {
        "batch_size": 1,
        "warmup_iterations": 20,
        "timed_iterations": 100,
        "repetitions": 5,
    }:
        raise ValueError("v1.4 benchmark protocol must be batch1/20/100/5")
    serialized = json.dumps(payload).lower()
    for prohibited in ("openvino", "cuda", "qnn", "npu"):
        if prohibited in serialized:
            raise ValueError(f"prohibited backend in v1.4 config: {prohibited}")
    return {**dict(payload), "preset": preset}


def load_config(path: Path) -> dict:
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_output_path(path: Path, root: Path) -> Path:
    resolved = Path(path).resolve()
    allowed = (Path(root).resolve() / "results/benchmarks").resolve()
    try:
        relative = resolved.relative_to(allowed)
    except ValueError as error:
        raise ValueError("v1.4 output must stay under results/benchmarks") from error
    if not relative.parts or not relative.parts[0].startswith(
        "v1.4_ort_customop_thread_preset"
    ):
        raise ValueError("v1.4 output must use its isolated preset directory")
    return resolved


def verify_artifact(path: Path, expected_sha256: str, *, label: str) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} provenance mismatch: expected {expected_sha256}, got {actual}: {path}"
        )
    return {"label": label, "path": str(path), "sha256": actual}


def exact_tensor_metrics(reference, candidate) -> dict:
    left = np.asarray(reference)
    right = np.asarray(candidate)
    compatible = left.shape == right.shape and left.dtype == right.dtype
    mismatch_count = (
        int(np.count_nonzero(left.reshape(-1) != right.reshape(-1)))
        if compatible
        else max(int(left.size), int(right.size), 1)
    )
    return {
        "shape_equal": left.shape == right.shape,
        "dtype_equal": left.dtype == right.dtype,
        "exact_equal": compatible and mismatch_count == 0,
        "elements_compared": int(left.size) if compatible else 0,
        "mismatch_count": mismatch_count,
    }


def analyze_profile_events(
    events: Sequence[Mapping], *, inference_wall_time_us: float
) -> dict:
    return parse_profile_events(events, inference_wall_time_us=inference_wall_time_us)


def benchmark_statistics(raw_repetitions: Sequence[Sequence[float]]) -> dict:
    return summarize_repetitions(raw_repetitions, batch_size=1)


def protocol_record(*, input_digest: str) -> dict:
    return {
        "provider": "CPUExecutionProvider",
        "batch_size": 1,
        "warmup_iterations": 20,
        "timed_iterations": 100,
        "repetitions": 5,
        "input_digest": input_digest,
        "input_reuse": "same cached NumPy array for every invocation",
        "graph_optimization_level": "GraphOptimizationLevel.ORT_ENABLE_ALL",
        "execution_mode": "ExecutionMode.ORT_SEQUENTIAL",
        "intra_op_threads": 4,
        "inter_op_threads": 0,
        "profiling_enabled": False,
        "timing_boundary": "perf_counter_ns immediately around session.run",
        "output_selection": "explicit final logits",
        "session_lifecycle": "fresh session followed by warmups",
        "between_invocation_work": "none",
    }


def compare_historical_protocol(current: Mapping, historical: Mapping) -> dict:
    return classify_comparability(current, historical)


def repetition_reproduced(current: Mapping, historical: Mapping) -> dict:
    current_median = float(current["median_of_repetition_p50_ms"])
    historical_range = [float(value) for value in historical["repetition_p50_range_ms"]]
    typical_latency_reproduced = (
        historical_range[0] <= current_median <= historical_range[1]
    )
    current_cv = float(current["coefficient_of_variation"])
    historical_cv = float(historical["coefficient_of_variation"])
    variance_reproduced = current_cv <= historical_cv
    return {
        "criterion": (
            "typical latency requires the current median of five repetition p50 values "
            "to lie within the historical v1.3 range; variance is reproduced only when "
            "the retained-sample CV is no greater than the historical CV"
        ),
        "current_median_of_repetition_p50_ms": current_median,
        "historical_repetition_p50_range_ms": historical_range,
        "typical_latency_reproduced": typical_latency_reproduced,
        "current_coefficient_of_variation": current_cv,
        "historical_coefficient_of_variation": historical_cv,
        "variance_reproduced": variance_reproduced,
        "reproduced": typical_latency_reproduced and variance_reproduced,
    }


def validation_decisions(
    *,
    exactness_passed: bool,
    profile: Mapping,
    benchmark_completed: bool,
    historical_protocol_matched: bool,
    historical_typical_latency_reproduced: bool,
    historical_variance_reproduced: bool,
) -> dict:
    categories = profile.get("categories", [])
    dominant = categories[0] if categories else None
    custom_share = float(
        profile.get("custom_op_execution", {}).get(
            "percentage_of_node_event_time", 0.0
        )
    )
    custom_nodes = int(
        profile.get("custom_op_execution", {}).get("unique_node_count", 0)
    )
    validated_typical = bool(
        exactness_passed
        and benchmark_completed
        and custom_nodes == EXPECTED_SITE_COUNT
        and historical_protocol_matched
        and historical_typical_latency_reproduced
    )
    convolution_dominant = bool(dominant and dominant["name"] == "convolution")
    below_prior_boundary = custom_share < CUSTOM_SHARE_PRIORITY_PERCENT
    return {
        "preset": {
            "decision": (
                "validated for this machine/protocol"
                if validated_typical and historical_variance_reproduced
                else (
                    "validated with a tail-variance caveat for this machine/protocol"
                    if validated_typical
                    else "not validated"
                )
            ),
            "reason": (
                "Exactness, all-node execution, protocol match, and typical-latency reproduction passed, but retained-sample CV exceeded v1.3."
                if validated_typical and not historical_variance_reproduced
                else (
                    "Exactness, all-node execution, protocol match, typical latency, and variance reproduction passed."
                    if validated_typical
                    else "At least one exactness, execution, protocol, or typical-latency condition did not pass."
                )
            ),
        },
        "convolution": {
            "remains_dominant": convolution_dominant,
            "category": dominant["name"] if dominant else None,
            "share_percent": (
                float(dominant["percentage_of_node_event_time"])
                if dominant
                else None
            ),
            "next_target": (
                "Conv/runtime configuration"
                if convolution_dominant
                else "reassess profile attribution"
            ),
        },
        "avx2": {
            "decision": (
                "deferred"
                if convolution_dominant and below_prior_boundary
                else (
                    "deferred pending a second four-thread attribution profile"
                    if convolution_dominant
                    else "inconclusive"
                )
            ),
            "custom_node_share_percent": custom_share,
            "reason": (
                "Custom activations remain below the 10% v1.3 diagnostic priority boundary while convolution is dominant."
                if convolution_dominant and below_prior_boundary
                else (
                    "Custom share exceeded the prior 10% boundary, but convolution remains dominant; confirm the changed attribution before prioritizing kernel work."
                    if convolution_dominant
                    else "The four-thread profile materially differs from the v1.3 attribution; inspect before prioritizing implementation work."
                )
            ),
        },
    }
