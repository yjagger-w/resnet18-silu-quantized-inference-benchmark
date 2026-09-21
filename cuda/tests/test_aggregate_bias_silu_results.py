#!/usr/bin/env python3
"""Offline tests for the CUDA Bias-SiLU benchmark aggregator."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import aggregate_bias_silu_results as aggregate  # noqa: E402


def make_run(run_index: int) -> dict:
    protocol = {
        "shape_set": "test shapes",
        "warmup_iterations": 5,
        "measured_iterations": 10,
        "iteration_scope": "per implementation",
        "execution_order": (
            "interleaved round-robin with per-round rotation"
        ),
        "timer": "CUDA events on the default stream",
        "excluded": "allocation and transfers",
        "effective_bandwidth_definition": "test definition",
    }
    results = []
    specifications = (
        ("scalar", "scalar", 0.0060),
        ("float4", "float4", 0.0050),
        ("auto", "float4", 0.0051),
    )
    for implementation, selected_path, base_p50 in specifications:
        p50 = base_p50 + run_index * 0.00001
        results.append({
            "shape": {
                "name": "stem",
                "batch": 1,
                "channels": 64,
                "height": 32,
                "width": 32,
                "elements": 65536,
            },
            "mode": "out-of-place",
            "implementation": implementation,
            "selected_kernel_path": selected_path,
            "mean_ms": p50 + 0.0001,
            "p50_ms": p50,
            "p90_ms": p50 + 0.0002,
            "p95_ms": p50 + 0.0003,
            "p99_ms": p50 + 0.0004,
            "min_ms": p50 - 0.0002,
            "max_ms": p50 + 0.0005,
            "effective_tensor_gbps": 100.0,
            "speedup_vs_scalar": 1.0,
            "max_abs_error": 1.0e-6,
        })
    return {
        "schema_version": 1,
        "device": {
            "name": "Tesla T4",
            "compute_capability": "7.5",
        },
        "build": {
            "cuda_compiler_version": "12.4.131",
            "host_compiler": "GNU 11.4.0",
        },
        "protocol": protocol,
        "results": results,
    }


class AggregateBiasSiluResultsTests(unittest.TestCase):
    def write_runs(self, directory: Path) -> list[Path]:
        paths = []
        for run_index in range(1, 6):
            path = directory / f"run_{run_index}.json"
            path.write_text(
                json.dumps(make_run(run_index), indent=2) + "\n",
                encoding="utf-8",
            )
            paths.append(path)
        return paths

    def test_aggregate_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.write_runs(root)
            report = aggregate.aggregate_runs(paths, 5, 2.0e-6)

            first_json = root / "first.json"
            first_md = root / "first.md"
            second_json = root / "second.json"
            second_md = root / "second.md"
            aggregate.write_outputs(report, first_json, first_md)
            aggregate.write_outputs(report, second_json, second_md)

            self.assertEqual(first_json.read_bytes(), second_json.read_bytes())
            self.assertEqual(first_md.read_bytes(), second_md.read_bytes())
            self.assertEqual(report["aggregation"]["run_count"], 5)
            comparison = report["comparisons"][0]
            self.assertEqual(
                comparison["auto_selected_kernel_path"],
                "float4",
            )
            self.assertEqual(
                comparison["best_explicit_implementation"],
                "float4",
            )

    def test_rejects_inconsistent_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.write_runs(root)
            payload = json.loads(paths[1].read_text(encoding="utf-8"))
            payload["protocol"]["measured_iterations"] = 11
            paths[1].write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "inconsistent protocol"):
                aggregate.aggregate_runs(paths, 5, 2.0e-6)

    def test_rejects_inconsistent_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.write_runs(root)
            payload = json.loads(paths[2].read_text(encoding="utf-8"))
            payload["results"][2]["selected_kernel_path"] = "scalar"
            paths[2].write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "inconsistent selected kernel paths",
            ):
                aggregate.aggregate_runs(paths, 5, 2.0e-6)

    def test_rejects_excessive_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.write_runs(root)
            payload = copy.deepcopy(make_run(4))
            payload["results"][0]["max_abs_error"] = 3.0e-6
            paths[3].write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "exceeds"):
                aggregate.aggregate_runs(paths, 5, 2.0e-6)


if __name__ == "__main__":
    unittest.main()
