"""v0.8 Standard-QDQ OpenVINO CPU conversion, validation and benchmark CLI."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import onnxruntime as ort

from silu_benchmark.backends.openvino_backend import (
    CAPABILITY_CAVEAT, OpenVINOCPU, OpenVINOUnavailableError, atomic_json,
    compare_outputs, convert_standard_qdq, generated_path, require_openvino,
    sha256, validate_config, verify_source,
)
from silu_benchmark.benchmark_data import load_cifar_batch, numpy_batches
# Reuse the frozen protocol's pure helpers without editing its source/fingerprints.
from run_official_benchmark import atomic_csv, atomic_text, process_memory_snapshot, q


def assert_no_torch():
    if any(name.split(".")[0] in ("torch", "torchvision") for name in sys.modules):
        raise RuntimeError("v0.8 benchmark must not import PyTorch/torchvision")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmarks/resnet18_silu_cifar10_v08_openvino_cpu.json")
    parser.add_argument("--model", type=Path, help="ONNX path; its SHA-256 must match the official Standard-QDQ source")
    parser.add_argument("--output", type=Path, help="report directory, restricted to results/benchmarks/v0.8*")
    parser.add_argument("--ir-output", type=Path, help="IR bundle root, restricted to artifacts/openvino/v0.8")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def stage_paths(config, smoke, conversion_only=False):
    destination = generated_path(ROOT, config["output_directory"], "report")
    suffix = ("_conversion" if conversion_only else "") + ("_smoke" if smoke else "")
    destination = destination.with_name(destination.name + suffix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination = destination.with_name(destination.name + "-" + uuid.uuid4().hex[:12])
    stage = Path(tempfile.mkdtemp(prefix=destination.name + ".partial-", dir=destination.parent))
    return stage, destination


def update_status(stage, **fields):
    path = stage / "run_status.json"
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    record.update(fields)
    atomic_json(path, record)


def measure(predict, values, warmups, runs):
    for _ in range(warmups):
        predict(values)
    baseline = process_memory_snapshot()
    observed = [baseline["bytes"]] if baseline["status"] == "available" else []
    durations = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        predict(values)
        durations.append((time.perf_counter_ns() - start) / 1e9)
        sample = process_memory_snapshot()
        if sample["status"] == "available":
            observed.append(sample["bytes"])
    memory = {
        "status": baseline["status"], "scope": "process RSS/working set, not OpenVINO tensor-allocator memory",
        "sampling": "baseline and after each timed synchronous inference; peak observed, not a continuous high-water mark",
    }
    if baseline["status"] == "available":
        memory.update(baseline_bytes=baseline["bytes"], peak_observed_bytes=max(observed), source=baseline["source"])
    else:
        memory["reason"] = baseline["reason"]
    return {**q(durations), "images_per_second": float(len(values) / np.mean(durations)), "memory": memory}


def write_reports(stage, result):
    atomic_json(stage / "benchmark_results.json", result)
    atomic_json(stage / "environment.json", result["environment"])
    atomic_json(stage / "validation.json", result["validation"])
    atomic_json(stage / "graph_report.json", {
        "onnx_source": result["source"], "ir": result["conversion"]["ir_graph"],
        "compiled": result["capabilities"], "caveat": CAPABILITY_CAVEAT,
    })
    latency, throughput = result["latency"], result["throughput"]
    atomic_csv(stage / "benchmark_results.csv", [{
        "variant_id": "standard_static_qdq_openvino_cpu", "smoke": result["smoke"],
        "samples": result["accuracy"]["samples"], "correct": result["accuracy"]["correct"],
        "top1_accuracy": result["accuracy"]["top1_accuracy"],
        "latency_p50_ms": latency["p50_ms"], "latency_p95_ms": latency["p95_ms"],
        "throughput_images_per_second": throughput["images_per_second"],
        "memory_status": latency["memory"]["status"],
        "memory_peak_observed_bytes": latency["memory"].get("peak_observed_bytes"),
        **{key: result["validation"][key] for key in ("prediction_agreement", "logit_max_absolute_error", "logit_mean_absolute_error")},
    }])
    lines = ["# v0.8 Standard-QDQ OpenVINO CPU", "", "**" + result["completion_status"] + "**", "",
             "Only Standard-QDQ ONNX was converted. " + CAPABILITY_CAVEAT, "",
             "Smoke is a 128-image development result, never an official 10,000-image result." if result["smoke"] else "Official 10,000-image OpenVINO run.",
             "", "## New OpenVINO measurements", "",
             "| Samples | Top-1 | p50 ms (batch 1) | p95 ms | Synchronous images/s |",
             "|---:|---:|---:|---:|---:|",
             f"| {result['accuracy']['samples']} | {result['accuracy']['top1_accuracy']:.4%} | {latency['p50_ms']:.4f} | {latency['p95_ms']:.4f} | {throughput['images_per_second']:.4f} |",
             "", "Process-level memory (timing excludes this query):", "", "```json", json.dumps(latency["memory"], indent=2), "```",
             "", "## ORT versus OpenVINO: deterministic validation batch", "",
             "Every prediction disagreement is retained below. No numerical acceptance tolerance has yet been established.",
             "", "```json", json.dumps(result["validation"], indent=2), "```",
             "", "## Historical ORT v0.6.5 (separate experiment)", "",
             "```json", json.dumps(result["source"]["official_ort_v065"], indent=2), "```",
             "", "Historical ORT results are not OpenVINO measurements. No speedup is claimed.",
             "", "## Conversion, IR and CPU capabilities", "",
             "```json", json.dumps({"conversion": result["conversion"], "source_q_nodes": result["source"]["q_nodes"],
                                     "source_dq_nodes": result["source"]["dq_nodes"], "capabilities": result["capabilities"]}, indent=2), "```",
             "", "## Protocol and environment", "", "```json",
             json.dumps({"config": result["config"], "effective_settings": result["effective_settings"], "environment": result["environment"]}, indent=2), "```"]
    atomic_text(stage / "benchmark_report.md", "\n".join(lines) + "\n")


def worker(request, stage):
    assert_no_torch()
    config = validate_config(request["config"])
    update_status(stage, phase="OpenVINO availability")
    ov = require_openvino()
    update_status(stage, phase="source provenance")
    source = verify_source(ROOT, config, request.get("model"))
    update_status(stage, phase="ONNX-to-IR conversion")
    print("v0.8 phase: Standard-QDQ conversion", flush=True)
    bundle = convert_standard_qdq(ROOT, config, source, resume=request["resume"], force_rebuild=request["force_rebuild"])
    atomic_json(stage / "conversion.json", {"source": source, **bundle})
    assert_no_torch()
    if request["conversion_only"]:
        update_status(stage, status="reports_ready", completion_status="conversion-success")
        return
    update_status(stage, phase="CPU compilation")
    backend = OpenVINOCPU(bundle, config["compile_properties"], config["device"])
    images, labels = load_cifar_batch(ROOT / config["data_root"], "test_batch")
    count = 128 if request["smoke"] else config["evaluation_samples"]
    indices = list(range(count))
    validation_images, validation_labels = next(numpy_batches(images, labels, indices, config["validation_batch_size"]))
    update_status(stage, phase="ORT versus OpenVINO validation")
    session = ort.InferenceSession(source["path"], providers=["CPUExecutionProvider"])
    ort_logits = session.run(None, {session.get_inputs()[0].name: validation_images})[0]
    ov_logits = backend.predict(validation_images)
    validation = compare_outputs(ort_logits, ov_logits, validation_labels)
    del session
    update_status(stage, phase="OpenVINO accuracy")
    correct = total = 0
    for batch, target in numpy_batches(images, labels, indices, config["evaluation_batch_size"]):
        logits = backend.predict(batch)
        if logits.shape != (len(target), 10) or not np.all(np.isfinite(logits)):
            raise ValueError("OpenVINO returned invalid logits during accuracy evaluation")
        correct += int(np.sum(logits.argmax(1) == target))
        total += len(target)
        print(f"v0.8 accuracy: {total}/{count}", flush=True)
    settings = {
        "evaluation_samples": count, "validation_samples": 128,
        "latency_warmups": 2 if request["smoke"] else config["latency_warmup_runs"],
        "latency_runs": 5 if request["smoke"] else config["latency_timed_runs"],
        "throughput_warmups": 1 if request["smoke"] else config["throughput_warmup_runs"],
        "throughput_runs": 3 if request["smoke"] else config["throughput_timed_runs"],
    }
    update_status(stage, phase="OpenVINO timing")
    latency = measure(backend.predict, np.zeros((1, 3, 32, 32), np.float32), settings["latency_warmups"], settings["latency_runs"])
    throughput = measure(backend.predict, validation_images[:config["throughput_batch_size"]], settings["throughput_warmups"], settings["throughput_runs"])
    assert_no_torch()
    result = {
        "schema_version": "benchmark-results/v0.8-openvino-cpu", "smoke": request["smoke"],
        "completion_status": "smoke-success" if request["smoke"] else "official-success",
        "config": config, "effective_settings": settings, "source": source, "conversion": bundle,
        "accuracy": {"samples": total, "correct": correct, "top1_accuracy": correct / total},
        "latency": latency, "throughput": throughput, "validation": validation, "capabilities": backend.capabilities,
        "environment": {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
                        "cpu": platform.processor(), "cpu_count": os.cpu_count(), "device": "CPU",
                        "openvino": ov.get_version(), "onnxruntime": ort.__version__, "numpy": np.__version__,
                        "torch_imported": False, "runner_sha256": sha256(Path(__file__))},
    }
    write_reports(stage, result)
    update_status(stage, status="reports_ready", completion_status=result["completion_status"])


def supervise(command, stage, destination, conversion_only):
    process = None
    try:
        with (stage / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, errors="replace", bufsize=1)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
            process.stdout.close()
        status = json.loads((stage / "run_status.json").read_text(encoding="utf-8"))
        if code:
            update_status(stage, status="failed", worker_exit_code=code,
                          error=status.get("error", "Native worker failure; inspect worker.log"),
                          diagnostic_tail=(stage / "worker.log").read_text(encoding="utf-8")[-6000:])
            return code if code > 0 else 1
        required = ("conversion.json",) if conversion_only else ("conversion.json", "benchmark_results.json", "benchmark_results.csv", "benchmark_report.md", "environment.json", "validation.json", "graph_report.json")
        if status.get("status") != "reports_ready" or not all((stage / item).exists() for item in required):
            raise RuntimeError("Incomplete worker reports; nothing is promoted")
        if destination.exists():
            raise FileExistsError("Refusing to replace existing output: " + str(destination))
        update_status(stage, status="success", phase="complete", worker_exit_code=0)
        stage.replace(destination)
        print(json.dumps({"status": status["completion_status"], "output": str(destination)}, indent=2), flush=True)
        return 0
    except BaseException as exc:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        update_status(stage, status="failed", error=str(exc), traceback=traceback.format_exc())
        return 1


def main(argv=None, conversion_only=False):
    args = parse_args(argv)
    if args.worker_request:
        stage = args.worker_request.resolve().parent
        try:
            worker(json.loads(args.worker_request.read_text(encoding="utf-8")), stage)
            return 0
        except BaseException as exc:
            update_status(stage, status="failed", error=str(exc), traceback=traceback.format_exc(),
                          failure_kind="dependency_unavailable" if isinstance(exc, OpenVINOUnavailableError) else "execution_failure")
            traceback.print_exc()
            return 4 if isinstance(exc, OpenVINOUnavailableError) else 1
    config = validate_config(json.loads(args.config.read_text(encoding="utf-8")))
    if args.output:
        config["output_directory"] = str(args.output)
    if args.ir_output:
        config["ir_output"] = str(args.ir_output)
    generated_path(ROOT, config["ir_output"], "ir")
    stage, destination = stage_paths(config, args.smoke, conversion_only)
    update_status(stage, status="running", phase="initializing", smoke=args.smoke)
    request = {"config": config, "smoke": args.smoke, "resume": args.resume,
               "force_rebuild": args.force_rebuild, "conversion_only": conversion_only,
               "model": str(args.model) if args.model else None}
    request_path = stage / "worker_request.json"
    try:
        atomic_json(request_path, request)
        print("v0.8 staging output: " + str(stage), flush=True)
    except BaseException as exc:
        update_status(stage, status="failed", error=str(exc), traceback=traceback.format_exc())
        return 1
    return supervise([sys.executable, "-u", str(Path(__file__).resolve()), "--worker-request", str(request_path)],
                     stage, destination, conversion_only)


if __name__ == "__main__":
    raise SystemExit(main())
