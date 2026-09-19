"""Qualcomm AI Hub orchestration and offline QNN v1.6 analysis helpers.

The SDK is deliberately imported lazily.  Importing this module is safe in the
baseline ORT environment, and all network-facing operations accept an injected
client so the orchestration can be tested without contacting AI Hub.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA = "qnn-aihub-experiments/v1.6"
REPORT_SCHEMA = "qnn-aihub-toolchain/v1.6"
DEFAULT_DEVICE = "Samsung Galaxy S22 5G"
DEFAULT_OS = "12"
AUDIT_SEED = 20260919
JOB_ID_PATTERN = re.compile(r"^j[a-z0-9]+$")
SUPPORTED_DTYPES = {
    "float32",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
}


class QaiHubUnavailableError(RuntimeError):
    """Raised when an online command is used without the optional SDK."""


def require_qai_hub():
    """Import the optional SDK with an actionable, credential-safe error."""

    try:
        return importlib.import_module("qai_hub")
    except ModuleNotFoundError as error:
        if error.name != "qai_hub":
            raise
        raise QaiHubUnavailableError(
            "Qualcomm AI Hub support is optional and qai_hub is not installed. "
            "Create the separate QNN environment with "
            "`python -m pip install -r requirements-qnn.txt`; configure credentials "
            "outside the repository. Never commit token values or client.ini."
        ) from error


def sdk_version(sdk: Any) -> str:
    """Return a JSON-safe SDK version (some SDK releases use rich objects)."""

    return str(getattr(sdk, "__version__", "unknown"))


def validate_job_id(job_id: str) -> str:
    if not isinstance(job_id, str) or not JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError("AI Hub job ID must match j followed by lowercase letters/digits")
    return job_id


def parse_input_spec(value: str) -> tuple[str, tuple[tuple[int, ...], str]]:
    """Parse ``name=dim,dim,...:dtype`` into an AI Hub input spec entry."""

    try:
        name, description = value.split("=", 1)
        dimensions, dtype = description.rsplit(":", 1)
        shape = tuple(int(item) for item in dimensions.split(","))
    except (ValueError, TypeError) as error:
        raise ValueError(
            "input spec must use name=dim,dim,...:dtype, for example "
            "images=1,3,32,32:float32"
        ) from error
    if not name or not name.replace("_", "").isalnum():
        raise ValueError("input spec name must be alphanumeric (underscores allowed)")
    if not shape or any(dimension <= 0 for dimension in shape):
        raise ValueError("input spec dimensions must be positive integers")
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"unsupported AI Hub input dtype: {dtype}")
    return name, (shape, dtype)


def parse_input_specs(values: Sequence[str]) -> dict[str, tuple[tuple[int, ...], str]]:
    specs: dict[str, tuple[tuple[int, ...], str]] = {}
    for value in values:
        name, spec = parse_input_spec(value)
        if name in specs:
            raise ValueError(f"duplicate input spec: {name}")
        specs[name] = spec
    if not specs:
        raise ValueError("at least one input spec is required")
    return specs


def repository_path(
    root: Path, value: str | Path, *, must_exist: bool = False, kind: str = "path"
) -> Path:
    """Resolve a repository-relative path and reject absolute/path-traversal input."""

    root = Path(root).resolve()
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"{kind} must be repository-relative, not absolute")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{kind} must stay inside the repository") from error
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"{kind} does not exist: {candidate.as_posix()}")
    return resolved


def relative_path(root: Path, path: Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_manifest(payload)


def validate_manifest(payload: Mapping[str, Any]) -> dict:
    if payload.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA}")
    if payload.get("seed") != AUDIT_SEED:
        raise ValueError(f"QNN numerical audit seed must be {AUDIT_SEED}")
    if payload.get("device") != {"name": DEFAULT_DEVICE, "os": DEFAULT_OS}:
        raise ValueError("v1.6 baseline device must be Samsung Galaxy S22 5G / Android 12")
    environment = payload.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("manifest environment is missing")
    required_versions = {
        "python": "3.11.15",
        "numpy": "2.4.6",
        "onnx": "1.19.1",
        "onnxruntime": "1.30.0",
        "qai_hub": "0.55.0",
    }
    if dict(environment) != required_versions:
        raise ValueError("manifest environment does not match the validated QNN environment")
    models = payload.get("models")
    if not isinstance(models, Mapping) or tuple(models) != (
        "fp32",
        "qdq_int8",
        "piecewise_reference",
    ):
        raise ValueError("manifest must contain the three ordered v1.6 models")
    for model_id, model in models.items():
        if not isinstance(model, Mapping):
            raise ValueError(f"invalid model entry: {model_id}")
        for key in ("source_path", "profile_path"):
            value = model.get(key)
            if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
                raise ValueError(f"{model_id}.{key} must be a safe repository-relative path")
        digest = model.get("source_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[A-Fa-f0-9]{64}", digest):
            raise ValueError(f"invalid source digest for {model_id}")
        jobs = model.get("jobs")
        if not isinstance(jobs, Mapping) or set(jobs) != {"compile", "profile", "inference"}:
            raise ValueError(f"{model_id} must contain compile/profile/inference job IDs")
        for job_id in jobs.values():
            validate_job_id(job_id)
        output_keys = model.get("audit_output_keys")
        if not isinstance(output_keys, Mapping) or set(output_keys) != {"local", "remote"}:
            raise ValueError(f"{model_id} audit output keys are missing")
    audit = payload.get("numerical_audit")
    if not isinstance(audit, Mapping):
        raise ValueError("numerical audit paths are missing")
    for key in ("inputs", "outputs", "baseline_json", "baseline_markdown"):
        value = audit.get(key)
        if not isinstance(value, str) or Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError(f"numerical_audit.{key} must be repository-relative")
    return dict(payload)


def _status_code(status: Any) -> str:
    return str(getattr(status, "code", status))


def wait_for_success(job: Any, timeout: int | None = None) -> Any:
    status = job.wait(timeout=timeout)
    success = getattr(status, "success", _status_code(status).upper() == "SUCCESS")
    if not success:
        raise RuntimeError(f"AI Hub job {getattr(job, 'job_id', 'unknown')} finished with {_status_code(status)}")
    return status


def job_record(
    job: Any,
    *,
    operation: str,
    resumed: bool,
    sdk: Any,
    device_name: str,
    os_version: str,
    status: Any | None = None,
) -> dict:
    """Build a strict allow-list record; SDK config and credentials are excluded."""

    job_id = validate_job_id(str(job.job_id))
    record = {
        "schema_version": REPORT_SCHEMA,
        "operation": operation,
        "resumed": bool(resumed),
        "job_id": job_id,
        "job_url": f"https://workbench.aihub.qualcomm.com/jobs/{job_id}/",
        "job_type": type(job).__name__,
        "device": {"name": str(device_name), "os": str(os_version)},
        "qai_hub_version": sdk_version(sdk),
    }
    if status is not None:
        record["status"] = _status_code(status)
    return record


class QnnAiHubBackend:
    """Small adapter around the public qai_hub 0.55 client surface."""

    def __init__(self, *, sdk: Any | None = None, client: Any | None = None):
        self.sdk = sdk if sdk is not None else require_qai_hub()
        set_verbose = getattr(self.sdk, "set_verbose", None)
        if callable(set_verbose):
            set_verbose(False)
        self.client = client if client is not None else self.sdk.Client()

    def device(self, name: str = DEFAULT_DEVICE, os_version: str = DEFAULT_OS):
        if not name or not os_version:
            raise ValueError("device name and OS version are required")
        return self.sdk.Device(name=name, os=os_version)

    def restore_job(self, job_id: str, *, required_method: str | None = None):
        job = self.client.get_job(validate_job_id(job_id))
        if required_method is not None and not callable(getattr(job, required_method, None)):
            raise TypeError(
                f"job {job_id} is not compatible with required operation {required_method}"
            )
        return job

    def compile(
        self,
        *,
        model: Path | str | None = None,
        input_specs: Mapping[str, Any] | None = None,
        device_name: str = DEFAULT_DEVICE,
        os_version: str = DEFAULT_OS,
        name: str | None = None,
        options: str = "",
        job_id: str | None = None,
    ):
        if job_id:
            return self.restore_job(job_id, required_method="get_target_model")
        if model is None:
            raise ValueError("model is required when no compile job ID is supplied")
        return self.client.submit_compile_job(
            model=str(model),
            device=self.device(device_name, os_version),
            name=name,
            input_specs=dict(input_specs or {}),
            options=options,
        )

    def compiled_model(self, compile_job_id: str):
        job = self.restore_job(compile_job_id, required_method="get_target_model")
        model = job.get_target_model()
        if model is None:
            raise RuntimeError(f"compile job {compile_job_id} did not produce a target model")
        return model

    def profile(
        self,
        *,
        compile_job_id: str | None = None,
        device_name: str = DEFAULT_DEVICE,
        os_version: str = DEFAULT_OS,
        name: str | None = None,
        options: str = "",
        job_id: str | None = None,
    ):
        if job_id:
            return self.restore_job(job_id, required_method="download_profile")
        if not compile_job_id:
            raise ValueError("compile job ID is required when no profile job ID is supplied")
        return self.client.submit_profile_job(
            model=self.compiled_model(compile_job_id),
            device=self.device(device_name, os_version),
            name=name,
            options=options,
        )

    def inference(
        self,
        *,
        inputs: Any | None = None,
        compile_job_id: str | None = None,
        device_name: str = DEFAULT_DEVICE,
        os_version: str = DEFAULT_OS,
        name: str | None = None,
        options: str = "",
        job_id: str | None = None,
    ):
        if job_id:
            return self.restore_job(job_id, required_method="download_output_data")
        if not compile_job_id:
            raise ValueError("compile job ID is required when no inference job ID is supplied")
        if inputs is None:
            raise ValueError("inputs are required when no inference job ID is supplied")
        return self.client.submit_inference_job(
            model=self.compiled_model(compile_job_id),
            device=self.device(device_name, os_version),
            inputs=inputs,
            name=name,
            options=options,
        )


def percentile(values: Sequence[float], percentage: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= percentage <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_profile(profile: Mapping[str, Any]) -> dict:
    summary = profile.get("execution_summary")
    if not isinstance(summary, Mapping):
        raise ValueError("profile is missing execution_summary")
    raw_times = summary.get("all_inference_times")
    if not isinstance(raw_times, list) or not raw_times:
        raise ValueError("profile is missing raw inference times")
    times_ms = [float(value) / 1000.0 for value in raw_times]
    if any(not math.isfinite(value) or value < 0 for value in times_ms):
        raise ValueError("profile inference times must be finite and non-negative")
    details = profile.get("execution_detail", [])
    if not isinstance(details, list):
        raise ValueError("profile execution_detail must be a list")
    compute_units: dict[str, int] = {}
    non_npu_nodes = []
    for index, node in enumerate(details):
        if not isinstance(node, Mapping):
            raise ValueError("profile execution_detail contains a non-object node")
        unit = str(node.get("compute_unit", "unknown"))
        compute_units[unit] = compute_units.get(unit, 0) + 1
        if unit.upper() != "NPU":
            non_npu_nodes.append(
                {"index": index, "name": str(node.get("name", "")), "compute_unit": unit}
            )
    node_count = len(details)
    npu_count = sum(
        count for unit, count in compute_units.items() if unit.upper() == "NPU"
    )
    return {
        "sample_count": len(times_ms),
        "latency_ms": {
            "mean": statistics.fmean(times_ms),
            "p50": percentile(times_ms, 50),
            "p90": percentile(times_ms, 90),
            "p95": percentile(times_ms, 95),
            "p99": percentile(times_ms, 99),
            "min": min(times_ms),
            "max": max(times_ms),
        },
        "memory_bytes": {
            "inference_peak": summary.get("estimated_inference_peak_memory"),
            "first_load_peak": summary.get("first_load_peak_memory"),
            "warm_load_peak": summary.get("warm_load_peak_memory"),
        },
        "nodes": {
            "total": node_count,
            "npu": npu_count,
            "npu_coverage": (npu_count / node_count) if node_count else None,
            "compute_units": compute_units,
            "non_npu_count": len(non_npu_nodes),
            "non_npu_nodes": non_npu_nodes,
        },
    }


def build_profile_summary(profiles: Mapping[str, Mapping[str, Any]]) -> dict:
    required = ("fp32", "qdq_int8", "piecewise_reference")
    if tuple(profiles) != required:
        raise ValueError(f"profiles must be ordered as {required}")
    models = {model_id: summarize_profile(profiles[model_id]) for model_id in required}
    fp32_mean = models["fp32"]["latency_ms"]["mean"]
    int8_mean = models["qdq_int8"]["latency_ms"]["mean"]
    piecewise_mean = models["piecewise_reference"]["latency_ms"]["mean"]
    return {
        "schema_version": REPORT_SCHEMA,
        "models": models,
        "comparisons": {
            "qdq_int8_vs_fp32": {
                "speedup": fp32_mean / int8_mean,
                "mean_latency_change_percent": (int8_mean / fp32_mean - 1.0) * 100.0,
            },
            "piecewise_reference_vs_fp32": {
                "speedup": fp32_mean / piecewise_mean,
                "mean_latency_change_percent": (piecewise_mean / fp32_mean - 1.0) * 100.0,
            },
        },
    }


def generate_audit_inputs(seed: int = AUDIT_SEED):
    """Recreate the frozen 4 structured + 8 random stress inputs."""

    import numpy as np

    if seed != AUDIT_SEED:
        raise ValueError(f"v1.6 QNN audit seed is fixed at {AUDIT_SEED}")
    ramp = np.linspace(-2.5, 2.5, 32 * 32, dtype=np.float32).reshape(1, 32, 32)
    structured = np.concatenate(
        [
            np.zeros((1, 3, 32, 32), dtype=np.float32),
            -np.ones((1, 3, 32, 32), dtype=np.float32),
            np.ones((1, 3, 32, 32), dtype=np.float32),
            np.broadcast_to(ramp, (1, 3, 32, 32)),
        ]
    )
    random = np.random.default_rng(seed).normal(size=(8, 3, 32, 32))
    random = np.clip(random, -3.0, 3.0).astype(np.float32)
    return np.ascontiguousarray(np.concatenate([structured, random]))


def numerical_metrics(reference: Any, candidate: Any) -> dict:
    import numpy as np

    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim < 2 or not len(reference):
        raise ValueError("numerical audit tensors must have the same non-empty batched shape")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("numerical audit tensors must be finite")
    difference = candidate - reference
    flat_reference = reference.reshape(len(reference), -1)
    flat_candidate = candidate.reshape(len(candidate), -1)
    numerator = np.sum(flat_reference * flat_candidate, axis=1)
    reference_norm = np.linalg.norm(flat_reference, axis=1)
    candidate_norm = np.linalg.norm(flat_candidate, axis=1)
    denominator = reference_norm * candidate_norm
    cosine = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator != 0,
    )
    cosine[(reference_norm == 0) & (candidate_norm == 0)] = 1.0
    return {
        "max_abs_error": float(np.max(np.abs(difference))),
        "mean_abs_error": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "mean_cosine_similarity": float(np.mean(cosine)),
        "min_cosine_similarity": float(np.min(cosine)),
        "top1_agreement": float(
            np.mean(np.argmax(flat_reference, axis=1) == np.argmax(flat_candidate, axis=1))
        ),
    }


def build_numerical_audit(
    manifest: Mapping[str, Any],
    inputs: Any,
    outputs: Mapping[str, Any],
    *,
    input_sha256: str,
    outputs_sha256: str,
    qai_hub_version: Any,
) -> dict:
    import numpy as np

    validate_manifest(manifest)
    inputs = np.asarray(inputs)
    expected_inputs = generate_audit_inputs(manifest["seed"])
    if not np.array_equal(inputs, expected_inputs):
        raise ValueError("audit inputs do not match the frozen 12-input seed=20260919 set")
    models: dict[str, Any] = {}
    fp32_local = outputs[manifest["models"]["fp32"]["audit_output_keys"]["local"]]
    for model_id, model in manifest["models"].items():
        local = np.asarray(outputs[model["audit_output_keys"]["local"]])
        remote = np.asarray(outputs[model["audit_output_keys"]["remote"]])
        if local.shape[0] != len(inputs) or remote.shape[0] != len(inputs):
            raise ValueError(f"{model_id} outputs must contain all 12 audit samples")
        jobs = model["jobs"]
        row = {
            "source_path": model["source_path"],
            "source_sha256": model["source_sha256"],
            "jobs": dict(jobs),
            "local_output_shape": list(local.shape),
            "remote_output_shape": list(remote.shape),
            "local_vs_remote": numerical_metrics(local, remote),
        }
        if model_id != "fp32":
            row["local_vs_fp32_local"] = numerical_metrics(fp32_local, local)
        models[model_id] = row
    return {
        "schema_version": REPORT_SCHEMA,
        "device": dict(manifest["device"]),
        "environment": {**dict(manifest["environment"]), "qai_hub": str(qai_hub_version)},
        "input_set": {
            "kind": "4 structured and 8 seeded random synthetic stress inputs",
            "seed": AUDIT_SEED,
            "sample_count": 12,
            "shape": list(inputs.shape),
            "sha256": input_sha256,
            "is_accuracy_dataset": False,
        },
        "outputs_sha256": outputs_sha256,
        "models": models,
        "limitations": [
            "This is a compiler-semantics smoke audit, not CIFAR-10 accuracy evaluation.",
            "Top-1 agreement does not prove internal UINT8-code equality.",
            "The piecewise model requires a standalone boundary probe for ties-to-even and code-boundary validation.",
        ],
    }


def profile_summary_markdown(summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    labels = {model_id: model["label"] for model_id, model in manifest["models"].items()}
    lines = [
        "# QNN profile summary - Galaxy S22 / Android 12",
        "",
        "| Model | Mean (ms) | P50 | P90 | P95 | P99 | Min | Max | Peak memory (MiB) | NPU nodes | Non-NPU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_id, result in summary["models"].items():
        latency = result["latency_ms"]
        nodes = result["nodes"]
        peak = result["memory_bytes"]["inference_peak"]
        peak_mib = float(peak) / (1024 * 1024) if peak is not None else float("nan")
        lines.append(
            f"| {labels[model_id]} | {latency['mean']:.5f} | {latency['p50']:.5f} | "
            f"{latency['p90']:.5f} | {latency['p95']:.5f} | {latency['p99']:.5f} | "
            f"{latency['min']:.5f} | {latency['max']:.5f} | {peak_mib:.2f} | "
            f"{nodes['npu']}/{nodes['total']} | {nodes['non_npu_count']} |"
        )
    int8_mean = summary["models"]["qdq_int8"]["latency_ms"]["mean"]
    piecewise = summary["models"]["piecewise_reference"]
    piecewise_mean = piecewise["latency_ms"]["mean"]
    piecewise_nodes = piecewise["nodes"]
    speedup = summary["comparisons"]["qdq_int8_vs_fp32"]["speedup"]
    slower = summary["comparisons"]["piecewise_reference_vs_fp32"]["mean_latency_change_percent"]
    lines.extend(
        [
            "",
            "## Confirmed conclusions",
            "",
            f"- QDQ INT8 averages {int8_mean:.5f} ms and is {speedup:.3f}x faster than FP32.",
            f"- Piecewise reference averages {piecewise_mean:.5f} ms; all "
            f"{piecewise_nodes['npu']}/{piecewise_nodes['total']} nodes run on NPU, "
            f"but it is {slower:.2f}% slower than FP32.",
            "",
            "Statistics are computed from the raw inference samples in the original profile JSON files; no samples are removed.",
        ]
    )
    return "\n".join(lines) + "\n"


def numerical_audit_markdown(audit: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    lines = [
        "# QNN numerical audit - Galaxy S22 / Android 12",
        "",
        "This audit uses 12 deterministic synthetic stress inputs (4 structured and 8 random, seed 20260919). It is not a CIFAR-10 accuracy evaluation.",
        "",
        "| Model | Mean abs. error | Max abs. error | RMSE | Mean cosine | Min cosine | Top-1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_id, row in audit["models"].items():
        metrics = row["local_vs_remote"]
        lines.append(
            f"| {manifest['models'][model_id]['label']} | {metrics['mean_abs_error']:.9f} | "
            f"{metrics['max_abs_error']:.9f} | {metrics['rmse']:.9f} | "
            f"{metrics['mean_cosine_similarity']:.9f} | {metrics['min_cosine_similarity']:.9f} | "
            f"{metrics['top1_agreement']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Top-1 agreement on this synthetic set does not prove CIFAR-10 accuracy and does not prove internal UINT8 code-value equality.",
        ]
    )
    return "\n".join(lines) + "\n"


def job_markdown(record: Mapping[str, Any]) -> str:
    return (
        f"# QNN AI Hub {record['operation']}\n\n"
        f"- Job ID: `{record['job_id']}`\n"
        f"- Status: `{record.get('status', 'not-waited')}`\n"
        f"- Restored: `{str(record['resumed']).lower()}`\n"
        f"- Device: {record['device']['name']} / Android {record['device']['os']}\n"
        f"- AI Hub: [job]({record['job_url']})\n\n"
        "Credentials, tokens, and client configuration are intentionally excluded.\n"
    )


def write_reports(output_dir: Path, stem: str, payload: Mapping[str, Any], markdown: str) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    return json_path, markdown_path
