"""Prerequisite, registration, profiling, and parity helpers for v1.2.

No dependency is downloaded or installed here. A real custom-op session is
allowed only when an explicit DLL and the matching official ORT development
headers are present.
"""

from __future__ import annotations

import json
import os
import shutil
import ctypes
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import onnxruntime as ort

from silu_benchmark.ort_custom_op_rewrite import (
    CUSTOM_DOMAIN,
    CUSTOM_OP_TYPE,
    EXPECTED_SITE_COUNT,
)


CONFIG_SCHEMA = "ort-cpp-customop-benchmark-config/v1"

# Python ORT 1.19.2 is statically linked into its extension on this Windows
# installation, while a different onnxruntime.dll is visible in System32.
# Keep the exact package-local runtime and directory handles alive for the
# process so the project's import-library dependency cannot bind elsewhere.
_DLL_DIRECTORY_HANDLES = []
_RUNTIME_DLL_HANDLES = []


@dataclass(frozen=True)
class OrtCustomOpPrerequisites:
    ort_version: str
    ort_package_root: str
    runtime_dll: str | None
    c_api_header: str | None
    cxx_api_header: str | None
    import_library: str | None
    cmake: str | None
    compiler_environment: str | None
    compiler_kind: str | None
    supported: bool
    blockers: tuple[str, ...]
    recommendation: str

    def to_dict(self) -> dict:
        return asdict(self)


def _first_existing(paths: Sequence[Path]) -> Path | None:
    return next((path.resolve() for path in paths if path.is_file()), None)


def _candidate_roots(explicit_root: Path | None = None) -> list[Path]:
    roots = []
    if explicit_root is not None:
        roots.append(explicit_root)
    environment_root = os.environ.get("ORT_ROOT")
    if environment_root:
        roots.append(Path(environment_root))
    package_root = Path(ort.__file__).resolve().parent
    roots.extend([package_root, package_root / "capi", Path(os.environ.get("CONDA_PREFIX", ""))])
    unique = []
    for root in roots:
        if str(root) and root not in unique:
            unique.append(root)
    return unique


def detect_prerequisites(explicit_ort_root: Path | None = None) -> OrtCustomOpPrerequisites:
    package_root = Path(ort.__file__).resolve().parent
    roots = _candidate_roots(explicit_ort_root)
    header_relatives = (
        Path("include/onnxruntime/core/session/onnxruntime_c_api.h"),
        Path("include/onnxruntime_c_api.h"),
        Path("onnxruntime_c_api.h"),
    )
    cxx_relatives = (
        Path("include/onnxruntime/core/session/onnxruntime_cxx_api.h"),
        Path("include/onnxruntime_cxx_api.h"),
        Path("onnxruntime_cxx_api.h"),
    )
    library_relatives = (
        Path("lib/onnxruntime.lib"), Path("onnxruntime.lib"), Path("capi/onnxruntime.lib")
    )
    runtime_relatives = (
        Path("bin/onnxruntime.dll"), Path("onnxruntime.dll"), Path("capi/onnxruntime.dll")
    )
    c_api = _first_existing([root / item for root in roots for item in header_relatives])
    cxx_api = _first_existing([root / item for root in roots for item in cxx_relatives])
    import_library = _first_existing([root / item for root in roots for item in library_relatives])
    runtime = _first_existing([root / item for root in roots for item in runtime_relatives])

    cmake = shutil.which("cmake")
    cl = shutil.which("cl")
    compiler_environment = None
    compiler_kind = None
    if cl:
        compiler_environment = cl
        compiler_kind = "MSVC"
    else:
        vs_build_tools = Path(
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
        )
        vcvars = vs_build_tools / "VC/Auxiliary/Build/vcvars64.bat"
        bundled_cmake = (
            vs_build_tools
            / "Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
        )
        if vcvars.is_file():
            compiler_environment = str(vcvars.resolve())
            compiler_kind = "MSVC via vcvars64.bat"
        if cmake is None and bundled_cmake.is_file():
            cmake = str(bundled_cmake.resolve())
    if compiler_environment is None:
        for name in ("clang-cl", "g++", "clang++"):
            found = shutil.which(name)
            if found:
                compiler_environment = found
                compiler_kind = name
                break

    blockers = []
    if c_api is None:
        blockers.append("missing official onnxruntime_c_api.h")
    if cxx_api is None:
        blockers.append("missing official onnxruntime_cxx_api.h")
    if cmake is None:
        blockers.append("missing CMake executable")
    if compiler_environment is None:
        blockers.append("missing usable C++ compiler environment")
    supported = not blockers
    recommendation = (
        "Provide the official ONNX Runtime 1.19.2 x64 development package "
        "(matching the installed Python runtime), including its include directory, "
        "and set ORT_ROOT to that extracted package. Do not install or upgrade Python packages."
        if blockers
        else "All source prerequisites are discoverable; a project custom-op source may be built."
    )
    return OrtCustomOpPrerequisites(
        ort_version=ort.__version__,
        ort_package_root=str(package_root),
        runtime_dll=str(runtime) if runtime else None,
        c_api_header=str(c_api) if c_api else None,
        cxx_api_header=str(cxx_api) if cxx_api else None,
        import_library=str(import_library) if import_library else None,
        cmake=cmake,
        compiler_environment=compiler_environment,
        compiler_kind=compiler_kind,
        supported=supported,
        blockers=tuple(blockers),
        recommendation=recommendation,
    )


def require_prerequisites(explicit_ort_root: Path | None = None) -> OrtCustomOpPrerequisites:
    result = detect_prerequisites(explicit_ort_root)
    if not result.supported:
        raise RuntimeError(
            "ORT C++ custom-op build prerequisites are unavailable: "
            + "; ".join(result.blockers)
            + ". "
            + result.recommendation
        )
    return result


def validate_config(payload: Mapping) -> None:
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("invalid v1.2 benchmark configuration schema")
    if payload.get("provider") != "CPUExecutionProvider":
        raise ValueError("v1.2 permits only ORT CPUExecutionProvider")
    if payload.get("custom_domain") != CUSTOM_DOMAIN:
        raise ValueError("v1.2 custom domain differs from the frozen contract")
    if payload.get("custom_op_type") != CUSTOM_OP_TYPE:
        raise ValueError("v1.2 custom op type differs from the frozen contract")
    if payload.get("expected_site_count") != EXPECTED_SITE_COUNT:
        raise ValueError("v1.2 must target exactly 17 custom nodes")
    expected_hash = payload.get("selected_model_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("v1.2 selected model SHA-256 is invalid")
    if payload.get("probe_samples") != 128 or payload.get("final_test_samples") != 10000:
        raise ValueError("v1.2 probe/final sample counts differ from protocol")
    benchmark = payload.get("benchmark", {})
    for key in ("warmup_iterations", "timed_iterations", "batch_size"):
        if not isinstance(benchmark.get(key), int) or benchmark[key] <= 0:
            raise ValueError(f"benchmark {key} must be a positive integer")
    serialized = json.dumps(payload).lower()
    for prohibited in ("qnn", "openvino", "pytorch", "cuda", "npu"):
        if prohibited in serialized:
            raise ValueError(f"prohibited dependency or backend in v1.2 config: {prohibited}")


def load_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_config(payload)
    return payload


def prepare_process_local_ort_runtime() -> Path | None:
    if os.name != "nt":
        return None
    capi_directory = Path(ort.__file__).resolve().parent / "capi"
    runtime_dll = capi_directory / "onnxruntime.dll"
    if not runtime_dll.is_file():
        raise FileNotFoundError(
            f"Python ORT runtime DLL is missing from its package: {runtime_dll}"
        )
    if not _RUNTIME_DLL_HANDLES:
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(capi_directory)))
        _RUNTIME_DLL_HANDLES.append(ctypes.WinDLL(str(runtime_dll.resolve())))
    return runtime_dll.resolve()


def create_custom_op_session(
    model_path: Path,
    library_path: Path,
    *,
    enable_profiling: bool = False,
    profile_prefix: Path | None = None,
):
    if not model_path.is_file():
        raise FileNotFoundError(f"custom-op ONNX model is missing: {model_path}")
    if not library_path.is_file():
        raise FileNotFoundError(f"custom-op library is missing: {library_path}")
    prepare_process_local_ort_runtime()
    options = ort.SessionOptions()
    if enable_profiling:
        options.enable_profiling = True
        if profile_prefix is not None:
            profile_prefix.parent.mkdir(parents=True, exist_ok=True)
            options.profile_file_prefix = str(profile_prefix.resolve())
    try:
        registered_path = options.register_custom_ops_library(str(library_path.resolve()))
    except Exception as error:
        raise RuntimeError(f"failed to register ORT custom-op library: {error}") from error
    session = ort.InferenceSession(
        str(model_path.resolve()), sess_options=options, providers=["CPUExecutionProvider"]
    )
    return session, registered_path or str(library_path.resolve())


def parse_custom_op_profile(profile_path: Path, expected_node_names: Sequence[str]) -> dict:
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    expected = set(expected_node_names)
    observed = set()
    execution_by_node = {}
    for event in events:
        args = event.get("args") or {}
        op_name = args.get("op_name") or args.get("op_type")
        node_name = args.get("node_name") or event.get("name", "")
        matched = str(node_name) if str(node_name) in expected else next(
            (name for name in sorted(expected, key=len, reverse=True) if name in str(node_name)),
            None,
        )
        event_name = str(event.get("name", ""))
        if (
            op_name == CUSTOM_OP_TYPE
            and matched is not None
            and event_name.endswith("_kernel_time")
        ):
            observed.add(matched)
            row = execution_by_node.setdefault(
                matched,
                {
                    "node_name": matched,
                    "kernel_execution_count": 0,
                    "total_duration_us": 0,
                },
            )
            row["kernel_execution_count"] += 1
            row["total_duration_us"] += int(event.get("dur") or 0)
    missing = sorted(expected - observed)
    return {
        "expected_node_count": len(expected),
        "executed_node_count": len(observed),
        "all_expected_nodes_executed": not missing and len(expected) == EXPECTED_SITE_COUNT,
        "missing_node_names": missing,
        "kernel_execution_evidence": [
            execution_by_node[name] for name in sorted(execution_by_node)
        ],
    }


def comparison_metrics(reference, candidate) -> dict:
    left = np.asarray(reference)
    right = np.asarray(candidate)
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("parity tensors differ in shape or dtype")
    difference = right.astype(np.float64) - left.astype(np.float64)
    left64 = left.astype(np.float64).reshape(-1)
    right64 = right.astype(np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left64) * np.linalg.norm(right64))
    exact = bool(np.array_equal(left, right))
    return {
        "shape": list(left.shape),
        "dtype": str(left.dtype),
        "exact_equal": exact,
        "max_absolute_error": float(np.max(np.abs(difference))) if difference.size else 0.0,
        "mae": float(np.mean(np.abs(difference))) if difference.size else 0.0,
        "mse": float(np.mean(difference * difference)) if difference.size else 0.0,
        "cosine_similarity": (
            float(np.dot(left64, right64) / denominator)
            if denominator else (1.0 if exact else 0.0)
        ),
    }


def require_exact_parity(reference, candidate, *, tensor_name: str) -> dict:
    metrics = comparison_metrics(reference, candidate)
    if not metrics["exact_equal"]:
        left = np.asarray(reference).reshape(-1)
        right = np.asarray(candidate).reshape(-1)
        first = int(np.flatnonzero(left != right)[0])
        raise RuntimeError(
            f"zero-tolerance parity failed at {tensor_name}, flat index {first}: "
            f"reference={left[first]!r}, candidate={right[first]!r}"
        )
    return metrics
