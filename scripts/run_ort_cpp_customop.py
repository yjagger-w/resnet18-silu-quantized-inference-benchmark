#!/usr/bin/env python
"""Prove v1.2 custom-op parity, then evaluate and benchmark the hybrid graph."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
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
from silu_benchmark.benchmark_data import load_cifar_batch, numpy_batches, normalize_cifar_images
from silu_benchmark.ort_custom_op_backend import (
    create_custom_op_session,
    load_config,
    parse_custom_op_profile,
)
from silu_benchmark.ort_custom_op_rewrite import (
    discover_selected_piecewise_islands,
    sha256_file,
    validate_generated_artifact,
    validate_generated_output_path,
    verify_selected_source,
)


RUN_SCHEMA = "ort-cpp-customop-execution/v1"
SCOPE_LABEL = "ORT CPU hybrid graph with project C++ custom-op activations"


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_from_root(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def instrument_probe_models(
    source_model: onnx.ModelProto,
    custom_model: onnx.ModelProto,
    islands,
    artifact_root: Path,
) -> tuple[Path, Path, list[str]]:
    output_tensors = [item.output_code_tensor for item in islands]
    source_logits = source_model.graph.output[0].name
    custom_logits = custom_model.graph.output[0].name
    if source_logits != custom_logits:
        raise RuntimeError("source and custom models expose different final logits")
    mappings = {
        **{item.site_id: item.output_code_tensor for item in islands},
        "final_logits": source_logits,
    }
    source_instrumented = instrument_outputs(source_model, mappings)
    custom_instrumented = instrument_outputs(custom_model, mappings)
    source_path = artifact_root / "probe_reference_instrumented.onnx"
    custom_path = artifact_root / "probe_customop_instrumented.onnx"
    source_path.write_bytes(source_instrumented.SerializeToString(deterministic=True))
    custom_path.write_bytes(custom_instrumented.SerializeToString(deterministic=True))
    return source_path, custom_path, output_tensors


def first_mismatch(left: np.ndarray, right: np.ndarray) -> dict | None:
    if left.shape != right.shape or left.dtype != right.dtype:
        return {
            "kind": "shape_or_dtype",
            "reference_shape": list(left.shape),
            "candidate_shape": list(right.shape),
            "reference_dtype": str(left.dtype),
            "candidate_dtype": str(right.dtype),
        }
    indices = np.flatnonzero(left.reshape(-1) != right.reshape(-1))
    if indices.size == 0:
        return None
    flat_index = int(indices[0])
    return {
        "kind": "value",
        "flat_index": flat_index,
        "reference": left.reshape(-1)[flat_index].item(),
        "candidate": right.reshape(-1)[flat_index].item(),
    }


def run_probe(
    *,
    source_path: Path,
    custom_path: Path,
    library_path: Path,
    artifact_root: Path,
    report_root: Path,
    data_root: Path,
    probe_indices: list[int],
    islands,
) -> dict:
    source_model = onnx.load(str(source_path))
    custom_model = onnx.load(str(custom_path))
    instrumented_source, instrumented_custom, code_tensors = instrument_probe_models(
        source_model, custom_model, islands, artifact_root
    )
    reference_session = ort.InferenceSession(
        str(instrumented_source), providers=["CPUExecutionProvider"]
    )
    custom_session, registered_library = create_custom_op_session(
        instrumented_custom,
        library_path,
        enable_profiling=True,
        profile_prefix=report_root / "ort_customop_profile",
    )
    images, labels = load_cifar_batch(data_root, "test_batch")
    site_rows = {
        item.site_id: {
            "site_id": item.site_id,
            "custom_node_name": item.custom_node_name,
            "tensor_name": item.output_code_tensor,
            "dtype": "uint8",
            "elements_compared": 0,
            "mismatch_count": 0,
            "exact_equal": True,
        }
        for item in islands
    }
    logits_elements = 0
    correct = 0
    predictions_equal = 0
    completed = 0
    failure = None
    try:
        for inputs, expected_labels in numpy_batches(
            images, labels, probe_indices, batch_size=8
        ):
            reference_outputs = reference_session.run(
                None, {reference_session.get_inputs()[0].name: inputs}
            )
            custom_outputs = custom_session.run(
                None, {custom_session.get_inputs()[0].name: inputs}
            )
            for index, island in enumerate(islands):
                reference_codes = reference_outputs[index]
                custom_codes = custom_outputs[index]
                mismatch = first_mismatch(reference_codes, custom_codes)
                row = site_rows[island.site_id]
                row["elements_compared"] += int(reference_codes.size)
                if mismatch is not None:
                    row["exact_equal"] = False
                    row["mismatch_count"] = 1
                    failure = {
                        "first_divergent_site": island.site_id,
                        "site_order": index,
                        "probe_batch_offset": completed,
                        **mismatch,
                    }
                    break
            if failure is not None:
                break
            reference_logits = reference_outputs[-1]
            custom_logits = custom_outputs[-1]
            mismatch = first_mismatch(reference_logits, custom_logits)
            logits_elements += int(reference_logits.size)
            if mismatch is not None:
                failure = {
                    "first_divergent_site": "final_logits",
                    "site_order": len(islands),
                    "probe_batch_offset": completed,
                    **mismatch,
                }
                break
            reference_predictions = np.argmax(reference_logits, axis=1)
            custom_predictions = np.argmax(custom_logits, axis=1)
            predictions_equal += int(np.sum(reference_predictions == custom_predictions))
            correct += int(np.sum(custom_predictions == expected_labels))
            completed += len(expected_labels)
    finally:
        profile_path = Path(custom_session.end_profiling()).resolve()

    execution = parse_custom_op_profile(
        profile_path, [item.custom_node_name for item in islands]
    )
    result = {
        "schema_version": "ort-cpp-customop-probe/v1",
        "scope": SCOPE_LABEL,
        "probe_samples_required": 128,
        "probe_samples_completed": completed,
        "probe_indices": probe_indices,
        "reference_model": str(source_path),
        "reference_model_sha256": sha256_file(source_path),
        "custom_model": str(custom_path),
        "custom_model_sha256": sha256_file(custom_path),
        "registered_library": registered_library,
        "profile_path": str(profile_path),
        "profile_sha256": file_sha256(profile_path),
        "execution_proof": execution,
        "site_code_parity": list(site_rows.values()),
        "all_site_codes_exact": all(row["exact_equal"] for row in site_rows.values()),
        "final_logits": {
            "dtype": "float32",
            "elements_compared": logits_elements,
            "exact_equal": failure is None,
        },
        "prediction_agreement": (
            predictions_equal / completed if completed else 0.0
        ),
        "probe_custom_accuracy": correct / completed if completed else 0.0,
        "failure": failure,
    }
    result["exact_probe_parity"] = bool(
        failure is None
        and completed == 128
        and len(probe_indices) == 128
        and result["all_site_codes_exact"]
        and result["final_logits"]["exact_equal"]
        and result["prediction_agreement"] == 1.0
        and execution["all_expected_nodes_executed"]
    )
    atomic_json(report_root / "probe_parity.json", result)
    if not result["exact_probe_parity"]:
        raise RuntimeError(
            "v1.2 exact probe parity failed; full evaluation and benchmark are prohibited: "
            + json.dumps(failure or execution, sort_keys=True)
        )
    return result


def run_full_evaluation(
    model_path: Path, library_path: Path, data_root: Path, sample_count: int
) -> dict:
    session, registered_library = create_custom_op_session(model_path, library_path)
    images, labels = load_cifar_batch(data_root, "test_batch")
    indices = list(range(sample_count))
    correct = 0
    seen = 0
    started = time.perf_counter()
    for inputs, expected_labels in numpy_batches(images, labels, indices, batch_size=128):
        logits = session.run(None, {session.get_inputs()[0].name: inputs})[0]
        predictions = np.argmax(logits, axis=1)
        correct += int(np.sum(predictions == expected_labels))
        seen += len(expected_labels)
    elapsed = time.perf_counter() - started
    return {
        "schema_version": "ort-cpp-customop-full-evaluation/v1",
        "scope": SCOPE_LABEL,
        "sample_count": seen,
        "correct_predictions": correct,
        "top1_accuracy": correct / seen,
        "elapsed_seconds": elapsed,
        "evaluation_images_per_second": seen / elapsed,
        "batch_size": 128,
        "registered_library": registered_library,
        "provider": session.get_providers()[0],
    }


def run_uninstrumented_benchmark(
    model_path: Path,
    library_path: Path,
    data_root: Path,
    *,
    warmup_iterations: int,
    timed_iterations: int,
    batch_size: int,
) -> dict:
    session, registered_library = create_custom_op_session(model_path, library_path)
    images, _ = load_cifar_batch(data_root, "test_batch")
    inputs = normalize_cifar_images(images[:batch_size])
    feed = {session.get_inputs()[0].name: inputs}
    for _ in range(warmup_iterations):
        session.run(None, feed)
    latencies_ms = []
    for _ in range(timed_iterations):
        started = time.perf_counter_ns()
        session.run(None, feed)
        latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    values = np.asarray(latencies_ms, dtype=np.float64)
    mean_ms = statistics.fmean(latencies_ms)
    return {
        "schema_version": "ort-cpp-customop-benchmark/v1",
        "scope": SCOPE_LABEL,
        "model_instrumented": False,
        "provider": session.get_providers()[0],
        "batch_size": batch_size,
        "warmup_iterations": warmup_iterations,
        "timed_iterations": timed_iterations,
        "mean_latency_ms": mean_ms,
        "p50_latency_ms": float(np.percentile(values, 50)),
        "p95_latency_ms": float(np.percentile(values, 95)),
        "min_latency_ms": float(np.min(values)),
        "max_latency_ms": float(np.max(values)),
        "throughput_images_per_second": batch_size * 1000.0 / mean_ms,
        "registered_library": registered_library,
        "latencies_ms": latencies_ms,
        "claim_boundary": (
            "Hybrid ORT CPU graph measurement; not integer-only whole-model inference."
        ),
    }


def write_summary(report_root: Path, report: dict) -> None:
    atomic_json(report_root / "execution_report.json", report)
    benchmark = report.get("benchmark")
    if benchmark:
        with (report_root / "benchmark_results.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "scope",
                    "provider",
                    "batch_size",
                    "warmup_iterations",
                    "timed_iterations",
                    "mean_latency_ms",
                    "p50_latency_ms",
                    "p95_latency_ms",
                    "throughput_images_per_second",
                ]
            )
            writer.writerow(
                [
                    benchmark["scope"],
                    benchmark["provider"],
                    benchmark["batch_size"],
                    benchmark["warmup_iterations"],
                    benchmark["timed_iterations"],
                    benchmark["mean_latency_ms"],
                    benchmark["p50_latency_ms"],
                    benchmark["p95_latency_ms"],
                    benchmark["throughput_images_per_second"],
                ]
            )
    lines = [
        "# v1.2 ORT C++ custom-op execution report",
        "",
        f"Scope: **{SCOPE_LABEL}**.",
        "",
        f"- DLL SHA-256: `{report['library_sha256']}`",
        f"- All 17 custom nodes executed: `{report['probe']['execution_proof']['all_expected_nodes_executed']}`",
        f"- All 17 uint8 code tensors exact: `{report['probe']['all_site_codes_exact']}`",
        f"- Final logits exact: `{report['probe']['final_logits']['exact_equal']}`",
        f"- Probe prediction agreement: `{report['probe']['prediction_agreement']:.4f}`",
    ]
    if report.get("full_evaluation"):
        lines.append(
            f"- Full 10,000-image accuracy: `{report['full_evaluation']['top1_accuracy']:.4%}`"
        )
    if benchmark:
        lines.extend(
            [
                f"- Mean latency: `{benchmark['mean_latency_ms']:.4f} ms`",
                f"- P50 / P95: `{benchmark['p50_latency_ms']:.4f} / {benchmark['p95_latency_ms']:.4f} ms`",
                f"- Throughput: `{benchmark['throughput_images_per_second']:.2f} images/s`",
                "",
                "This is not integer-only whole-model inference.",
            ]
        )
    (report_root / "execution_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmarks/resnet18_silu_cifar10_v12_ort_customop_cpu.json"),
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=Path("build/ort-cpp-customop/Release/silu_ort_custom_op.dll"),
    )
    parser.add_argument("--probe-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(resolve_from_root(args.config))
    library_path = resolve_from_root(args.library)
    if not library_path.is_file():
        raise FileNotFoundError(f"built custom-op DLL is missing: {library_path}")
    source_path = resolve_from_root(config["selected_model"])
    receipt_path = resolve_from_root(config["selection_receipt"])
    artifact_root = validate_generated_output_path(
        resolve_from_root(config["generated_artifact_root"]), ROOT
    )
    report_root = validate_generated_output_path(
        resolve_from_root(config["generated_report_root"]), ROOT
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    model_path = artifact_root / "resnet18_silu_v12_ort_cpp_customop.onnx"
    manifest_path = artifact_root / "resnet18_silu_v12_ort_cpp_customop.manifest.json"
    validate_generated_artifact(
        source_path=source_path,
        rewritten_path=model_path,
        sidecar_path=manifest_path,
        selection_receipt_path=receipt_path,
    )
    source_model, _ = verify_selected_source(
        source_path,
        receipt_path,
        expected_model_sha256=config["selected_model_sha256"],
    )
    islands = discover_selected_piecewise_islands(source_model)
    controls = json.loads(
        (ROOT / "results/benchmarks/v1.1_accuracy_diagnosis/controls.json").read_text(
            encoding="utf-8"
        )
    )
    probe_indices = [int(value) for value in controls["probe_indices"]]
    if len(probe_indices) != config["probe_samples"] or len(set(probe_indices)) != 128:
        raise RuntimeError("frozen deterministic probe must contain 128 unique indices")
    data_root = resolve_from_root(config["data_root"])
    probe = run_probe(
        source_path=source_path,
        custom_path=model_path,
        library_path=library_path,
        artifact_root=artifact_root,
        report_root=report_root,
        data_root=data_root,
        probe_indices=probe_indices,
        islands=islands,
    )
    report = {
        "schema_version": RUN_SCHEMA,
        "scope": SCOPE_LABEL,
        "ort_version": ort.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "source_model": str(source_path),
        "source_model_sha256": sha256_file(source_path),
        "custom_model": str(model_path),
        "custom_model_sha256": sha256_file(model_path),
        "library": str(library_path),
        "library_sha256": file_sha256(library_path),
        "probe": probe,
        "full_evaluation": None,
        "benchmark": None,
    }
    if not args.probe_only:
        report["full_evaluation"] = run_full_evaluation(
            model_path,
            library_path,
            data_root,
            config["final_test_samples"],
        )
        atomic_json(report_root / "full_evaluation.json", report["full_evaluation"])
        settings = config["benchmark"]
        report["benchmark"] = run_uninstrumented_benchmark(
            model_path,
            library_path,
            data_root,
            warmup_iterations=settings["warmup_iterations"],
            timed_iterations=settings["timed_iterations"],
            batch_size=settings["batch_size"],
        )
        atomic_json(report_root / "benchmark_results.json", report["benchmark"])
    write_summary(report_root, report)
    print(
        json.dumps(
            {
                "scope": report["scope"],
                "library_sha256": report["library_sha256"],
                "exact_probe_parity": report["probe"]["exact_probe_parity"],
                "executed_custom_nodes": report["probe"]["execution_proof"][
                    "executed_node_count"
                ],
                "full_evaluation": report["full_evaluation"],
                "benchmark": report["benchmark"],
                "execution_report": str(report_root / "execution_report.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
