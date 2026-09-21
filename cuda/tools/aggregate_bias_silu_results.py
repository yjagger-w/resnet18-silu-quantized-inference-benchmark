#!/usr/bin/env python3
"""Aggregate repeated CUDA Bias-SiLU benchmark JSON reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

METRIC_FIELDS = (
    "mean_ms",
    "p50_ms",
    "p90_ms",
    "p95_ms",
    "p99_ms",
    "min_ms",
    "max_ms",
    "effective_tensor_gbps",
)
REQUIRED_IMPLEMENTATIONS = {"scalar", "float4", "auto"}
REQUIRED_EXECUTION_ORDER = (
    "interleaved round-robin with per-round rotation"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def run_sort_key(path: Path) -> tuple[int, str]:
    match = re.fullmatch(r"run_(\d+)\.json", path.name)
    if match is None:
        return (2**31 - 1, path.name)
    return (int(match.group(1)), path.name)


def require_finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{context} must be finite")
    return numeric


def canonical_key(result: dict[str, Any]) -> tuple[str, str, str]:
    shape = result.get("shape")
    if not isinstance(shape, dict) or not isinstance(shape.get("name"), str):
        raise ValueError("each result must contain shape.name")
    mode = result.get("mode")
    implementation = result.get("implementation")
    if not isinstance(mode, str) or not isinstance(implementation, str):
        raise ValueError("each result must contain string mode/implementation")
    return (shape["name"], mode, implementation)


def read_run(path: Path, max_abs_error: float) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read benchmark JSON {path}: {error}") from error

    for field in ("schema_version", "device", "build", "protocol", "results"):
        if field not in payload:
            raise ValueError(f"{path.name} is missing {field}")

    protocol = payload["protocol"]
    if not isinstance(protocol, dict):
        raise ValueError(f"{path.name} protocol must be an object")
    if protocol.get("iteration_scope") != "per implementation":
        raise ValueError(f"{path.name} has an unsupported iteration scope")
    if protocol.get("execution_order") != REQUIRED_EXECUTION_ORDER:
        raise ValueError(f"{path.name} is not an interleaved benchmark")

    results = payload["results"]
    if not isinstance(results, list) or not results:
        raise ValueError(f"{path.name} results must be a non-empty list")

    seen: set[tuple[str, str, str]] = set()
    for result in results:
        if not isinstance(result, dict):
            raise ValueError(f"{path.name} contains a non-object result")
        key = canonical_key(result)
        if key in seen:
            raise ValueError(f"{path.name} contains duplicate result {key}")
        seen.add(key)

        selected_path = result.get("selected_kernel_path")
        if not isinstance(selected_path, str) or not selected_path:
            raise ValueError(f"{path.name} result {key} has no kernel path")

        for metric in METRIC_FIELDS:
            require_finite_number(
                result.get(metric),
                f"{path.name} result {key} {metric}",
            )
        error = require_finite_number(
            result.get("max_abs_error"),
            f"{path.name} result {key} max_abs_error",
        )
        if error < 0.0 or error > max_abs_error:
            raise ValueError(
                f"{path.name} result {key} max_abs_error "
                f"{error} exceeds {max_abs_error}"
            )

    return payload


def metric_summary(values: Iterable[float]) -> dict[str, float]:
    materialized = [float(value) for value in values]
    return {
        "median": float(statistics.median(materialized)),
        "min": float(min(materialized)),
        "max": float(max(materialized)),
    }


def aggregate_runs(
    paths: Sequence[Path],
    expected_runs: int,
    max_abs_error: float,
) -> dict[str, Any]:
    if len(paths) != expected_runs:
        raise ValueError(
            f"expected {expected_runs} run files, found {len(paths)}"
        )

    runs = [read_run(path, max_abs_error) for path in paths]
    baseline = runs[0]
    for index, run in enumerate(runs[1:], start=2):
        for field in ("schema_version", "device", "build", "protocol"):
            if run[field] != baseline[field]:
                raise ValueError(
                    f"run {index} has inconsistent {field}"
                )

    ordered_keys = [
        canonical_key(result) for result in baseline["results"]
    ]
    expected_key_set = set(ordered_keys)
    indexed_runs: list[dict[tuple[str, str, str], dict[str, Any]]] = []
    for index, run in enumerate(runs, start=1):
        indexed = {
            canonical_key(result): result
            for result in run["results"]
        }
        if set(indexed) != expected_key_set:
            raise ValueError(f"run {index} has inconsistent result keys")
        indexed_runs.append(indexed)

    groups: dict[tuple[str, str], set[str]] = {}
    group_order: list[tuple[str, str]] = []
    for shape_name, mode, implementation in ordered_keys:
        group = (shape_name, mode)
        if group not in groups:
            groups[group] = set()
            group_order.append(group)
        groups[group].add(implementation)
    for group, implementations in groups.items():
        if implementations != REQUIRED_IMPLEMENTATIONS:
            raise ValueError(
                f"group {group} must contain scalar, float4, and auto"
            )

    aggregated_results: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key in ordered_keys:
        source_results = [run[key] for run in indexed_runs]
        paths_seen = {
            result["selected_kernel_path"]
            for result in source_results
        }
        if len(paths_seen) != 1:
            raise ValueError(
                f"result {key} has inconsistent selected kernel paths"
            )

        item = {
            "shape": source_results[0]["shape"],
            "mode": key[1],
            "implementation": key[2],
            "selected_kernel_path": next(iter(paths_seen)),
            "run_count": len(source_results),
            "metrics": {
                metric: metric_summary(
                    require_finite_number(
                        result[metric],
                        f"result {key} {metric}",
                    )
                    for result in source_results
                )
                for metric in METRIC_FIELDS
            },
            "max_abs_error": {
                "maximum": max(
                    float(result["max_abs_error"])
                    for result in source_results
                )
            },
        }
        aggregated_results.append(item)
        by_key[key] = item

    comparisons: list[dict[str, Any]] = []
    for shape_name, mode in group_order:
        scalar = by_key[(shape_name, mode, "scalar")]
        float4 = by_key[(shape_name, mode, "float4")]
        auto = by_key[(shape_name, mode, "auto")]
        scalar_p50 = scalar["metrics"]["p50_ms"]["median"]
        float4_p50 = float4["metrics"]["p50_ms"]["median"]
        auto_p50 = auto["metrics"]["p50_ms"]["median"]
        best_explicit = (
            "scalar" if scalar_p50 <= float4_p50 else "float4"
        )
        best_p50 = min(scalar_p50, float4_p50)

        comparisons.append({
            "shape": scalar["shape"],
            "mode": mode,
            "scalar_median_p50_ms": scalar_p50,
            "float4_median_p50_ms": float4_p50,
            "auto_median_p50_ms": auto_p50,
            "auto_selected_kernel_path": auto["selected_kernel_path"],
            "best_explicit_implementation": best_explicit,
            "float4_speedup_vs_scalar_p50": (
                scalar_p50 / float4_p50
            ),
            "auto_speedup_vs_scalar_p50": scalar_p50 / auto_p50,
            "auto_speedup_vs_best_explicit_p50": best_p50 / auto_p50,
            "maximum_abs_error": max(
                scalar["max_abs_error"]["maximum"],
                float4["max_abs_error"]["maximum"],
                auto["max_abs_error"]["maximum"],
            ),
        })

    return {
        "schema_version": 1,
        "kind": "cuda_bias_silu_repeated_benchmark_aggregate",
        "source_runs": [
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ],
        "device": baseline["device"],
        "build": baseline["build"],
        "protocol": baseline["protocol"],
        "aggregation": {
            "run_count": len(paths),
            "method": "median of each per-run summary metric",
            "p50_range": "minimum and maximum per-run P50",
            "max_abs_error_limit": max_abs_error,
        },
        "results": aggregated_results,
        "comparisons": comparisons,
    }


def format_microseconds(milliseconds: float) -> str:
    return f"{milliseconds * 1000.0:.3f}"


def render_markdown(report: dict[str, Any]) -> str:
    device = report["device"]
    build = report["build"]
    protocol = report["protocol"]
    lines = [
        "# Tesla T4 CUDA Bias-SiLU repeated benchmark",
        "",
        "This report aggregates five interleaved benchmark runs. "
        "Each configured iteration count applies independently to scalar, "
        "float4, and automatic dispatch.",
        "",
        "## Environment",
        "",
        f"- Device: {device['name']}",
        f"- Compute capability: {device['compute_capability']}",
        f"- CUDA compiler: {build['cuda_compiler_version']}",
        f"- Host compiler: {build['host_compiler']}",
        f"- Warm-up iterations per implementation: "
        f"{protocol['warmup_iterations']}",
        f"- Measured iterations per implementation: "
        f"{protocol['measured_iterations']}",
        f"- Execution order: {protocol['execution_order']}",
        "",
        "## Median P50 latency",
        "",
        "| Shape | Mode | Scalar (us) | Float4 (us) | Auto (us) | "
        "Auto path | Auto/scalar | Auto/best |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]

    for item in report["comparisons"]:
        lines.append(
            f"| {item['shape']['name']} "
            f"| {item['mode']} "
            f"| {format_microseconds(item['scalar_median_p50_ms'])} "
            f"| {format_microseconds(item['float4_median_p50_ms'])} "
            f"| {format_microseconds(item['auto_median_p50_ms'])} "
            f"| {item['auto_selected_kernel_path']} "
            f"| {item['auto_speedup_vs_scalar_p50']:.4f}x "
            f"| {item['auto_speedup_vs_best_explicit_p50']:.4f}x |"
        )

    lines.extend([
        "",
        "## Source runs",
        "",
        "| File | Bytes | SHA256 |",
        "|---|---:|---|",
    ])
    for source in report["source_runs"]:
        lines.append(
            f"| {source['filename']} "
            f"| {source['size_bytes']} "
            f"| `{source['sha256']}` |"
        )

    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "- Latencies are CUDA Event measurements on the recorded Tesla T4.",
        "- Allocation and host transfers are excluded as stated in the "
        "protocol.",
        "- Auto/best values near 1.0 indicate that adaptive dispatch tracks "
        "the best explicit implementation within run-to-run variation.",
        "- The 65,536-element automatic-dispatch threshold is specific to "
        "this T4/CUDA 12.4 evidence and is not a universal GPU threshold.",
        "",
    ])
    return "\n".join(lines)


def write_outputs(
    report: dict[str, Any],
    output_json: Path,
    output_markdown: Path,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    output_markdown.write_text(
        render_markdown(report),
        encoding="utf-8",
        newline="\n",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate repeated interleaved CUDA Bias-SiLU benchmarks"
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=5)
    parser.add_argument("--max-abs-error", type=float, default=2.0e-6)
    args = parser.parse_args(argv)
    if args.expected_runs <= 0:
        parser.error("--expected-runs must be positive")
    if not math.isfinite(args.max_abs_error) or args.max_abs_error < 0.0:
        parser.error("--max-abs-error must be finite and non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = sorted(
        args.input_dir.glob("run_*.json"),
        key=run_sort_key,
    )
    report = aggregate_runs(
        paths,
        expected_runs=args.expected_runs,
        max_abs_error=args.max_abs_error,
    )
    write_outputs(report, args.output_json, args.output_markdown)
    print(f"Wrote JSON: {args.output_json}")
    print(f"JSON SHA256: {sha256_file(args.output_json)}")
    print(f"Wrote Markdown: {args.output_markdown}")
    print(f"Markdown SHA256: {sha256_file(args.output_markdown)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
