"""v0.8.1: isolated 128-image CPU numerical diagnosis, never a full benchmark."""

import argparse
import json
import subprocess
import sys
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import onnx
import onnxruntime as ort

from silu_benchmark.backends.openvino_backend import (
    atomic_json, compare_outputs, ir_fingerprint, ir_graph_report, require_openvino,
    safe_properties, sha256, valid_ir, validate_config, verify_source,
)
from silu_benchmark.benchmark_data import load_cifar_batch, numpy_batches
from silu_benchmark.openvino_diagnosis import (
    array_digest, diagnostic_path, first_divergence, instrument, metrics,
    quantizer_control_model, rounding_boundary_evidence, select_probes, semantic_sites, validate_output_names,
)
from run_official_benchmark import atomic_text


def no_torch():
    if any(name.split(".")[0] in ("torch", "torchvision") for name in sys.modules):
        raise RuntimeError("diagnosis must never import PyTorch")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmarks/resnet18_silu_cifar10_v08_openvino_cpu.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/calibration/resnet18_silu_piecewise_v06_ort_cpu.json")
    parser.add_argument("--output", default="results/benchmarks/v0.8.1_diagnosis")
    parser.add_argument("--smoke", action="store_true", help="128 images only; also the default, no full mode exists")
    parser.add_argument("--node-start", type=int, help="optional extra inclusive source-node window")
    parser.add_argument("--node-end", type=int)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def worker(request):
    no_torch()
    destination = Path(request["output"])
    images = np.load(request["inputs"], allow_pickle=False)
    model_path = Path(request["model"])
    expected = request["outputs"]
    result = {"backend": request["backend"], "model_sha256": sha256(model_path)}
    if request["backend"] == "ort":
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        names = [[p.name] for p in session.get_outputs()]
        validate_output_names(expected, names)
        outputs = session.run(expected, {session.get_inputs()[0].name: images})
        result["version"] = ort.__version__
        for index, output in enumerate(outputs):
            np.save(destination / f"{index:03d}.npy", output, allow_pickle=False)
        del outputs, session
    else:
        ov = require_openvino()
        core = ov.Core()
        if model_path.suffix == ".onnx":
            # Only the isolated instrumented copy is converted, never canonical IR.
            model = core.read_model(str(model_path))
            xml = destination / "debug.xml"
            ov.serialize(model, str(xml), str(xml.with_suffix(".bin")), version="IR_V11")
            del model
            model_path = xml
        model = core.read_model(str(model_path), str(model_path.with_suffix(".bin")))
        compiled = core.compile_model(model, "CPU", request["compile_properties"])
        names = [sorted(p.get_names()) for p in compiled.outputs]
        validate_output_names(expected, names)
        result.update(version=ov.get_version(), xml_sha256=sha256(model_path), bin_sha256=sha256(model_path.with_suffix(".bin")),
                      ir_graph=ir_graph_report(model_path),
                      compiled_properties=safe_properties(compiled.get_property, ("EXECUTION_DEVICES", "INFERENCE_PRECISION_HINT", "PERFORMANCE_HINT")))
        runtime = compiled.get_runtime_model()
        result["runtime_ops"] = [{"name": op.get_friendly_name(), "op": op.get_type_name(),
                                  "info": {str(k): str(v.value) for k, v in op.get_rt_info().items()}}
                                 for op in runtime.get_ordered_ops()]
        del runtime
        infer = compiled.create_infer_request()
        outputs = infer.infer({0: images}, share_inputs=False, share_outputs=False)
        for index in range(len(expected)):
            np.save(destination / f"{index:03d}.npy", np.asarray(outputs[compiled.output(index)]), allow_pickle=False)
        del outputs, infer, compiled, model, core
    result["output_mapping"] = [{"source_value": name, "output_index": i, "runtime_names": names[i], "array_file": f"{i:03d}.npy"}
                                for i, name in enumerate(expected)]
    no_torch()
    atomic_json(destination / "execution.json", result)


def execute(stage, name, backend, model, output_names, config, inputs=None):
    destination = stage / name
    destination.mkdir()
    request = {"output": str(destination), "backend": backend, "model": str(model),
               "outputs": output_names, "inputs": str(inputs or stage / "inputs.npy"), "compile_properties": config["compile_properties"]}
    atomic_json(destination / "request.json", request)
    print("v0.8.1 phase: " + name, flush=True)
    with (destination / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(destination / "request.json")],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=240)
    if process.returncode:
        raise RuntimeError(f"{name}: worker exit {process.returncode}; inspect {destination / 'worker.log'}")
    return json.loads((destination / "execution.json").read_text()), destination


def load_output(directory, index):
    # Copy closes any npy mapping before comparisons or later directory cleanup.
    return np.load(directory / f"{index:03d}.npy", allow_pickle=False)


def paired_run(stage, name, model, probes, config, labels, baseline_logits):
    debug, mapping = instrument(model, probes)
    source = stage / (name + ".onnx")
    onnx.save(debug, str(source))
    expected = [v.name for v in debug.graph.output]
    ort_info, ort_dir = execute(stage, name + "_ort", "ort", source, expected, config)
    ov_info, ov_dir = execute(stage, name + "_openvino", "openvino", source, expected, config)
    rows = []
    for probe in mapping:
        index = probe["instrumented_output_index"]
        rows.append({**probe, "openvino_output": ov_info["output_mapping"][index],
                     "metrics": metrics(load_output(ort_dir, index), load_output(ov_dir, index))})
    logits_index = expected.index(model.graph.output[0].name)
    ort_logits, ov_logits = load_output(ort_dir, logits_index), load_output(ov_dir, logits_index)
    return {"name": name, "instrumented_onnx": str(source), "instrumented_onnx_sha256": sha256(source),
            "ort_execution": ort_info, "openvino_execution": ov_info, "probes": rows,
            "first_difference": first_divergence(rows),
            "first_input_dependent_difference": first_divergence(rows, input_dependent_only=True),
            "first_integer_difference": first_divergence(rows, input_dependent_only=True, integer_only=True),
            "logit_comparison": compare_outputs(ort_logits, ov_logits, labels),
            "instrumentation_effect": {
                "ort": metrics(baseline_logits[0], ort_logits), "openvino": metrics(baseline_logits[1], ov_logits),
                "ort_predictions": compare_outputs(baseline_logits[0], ort_logits, labels),
                "openvino_predictions": compare_outputs(baseline_logits[1], ov_logits, labels),
            }}


def write_report(stage, result):
    atomic_json(stage / "diagnosis.json", result)
    lines = ["# v0.8.1 OpenVINO CPU numerical-divergence diagnosis", "",
             "128 deterministic CIFAR-10 test images; Standard-QDQ only. No tolerance, repair, or backend-equivalence claim.", "",
             "## Conclusion", "", result["conclusion"], "",
             "## Localization and same-input quantizer control", "", "```json",
             json.dumps({"localization": result["localization"], "first_integer_difference": result["first_integer_difference"],
                         "quantizer_control": result.get("quantizer_control")}, indent=2), "```", "",
             "Exact inequality is an observational locator, not an acceptance threshold. The report separately identifies integer-code changes.", "",
             "## Uninstrumented logits", "", "```json", json.dumps(result["baseline_comparison"], indent=2), "```", "",
             "## Provenance", "", "```json", json.dumps(result["provenance"], indent=2), "```"]
    for run in result["runs"]:
        lines += ["", "## " + run["name"], "",
                  "| ONNX index (0-based) | Source value / producer | Shape | Max abs | MAE | MSE | Cosine | Exact |",
                  "|---:|---|---|---:|---:|---:|---:|---|"]
        for p in run["probes"]:
            m = p["metrics"]
            lines.append(f"| {p['topological_index']} | {p['source_value']} / {p['producer_op']} | {m['shape']} | {m['max_absolute_error']:.9g} | {m['mean_absolute_error']:.9g} | {m['mse']:.9g} | {m['cosine_similarity']:.9g} | {m['exact_equal']} |")
        lines += ["", "Instrumented logits, all disagreeing predictions, and same-backend instrumentation effects:", "", "```json",
                  json.dumps({"comparison": run["logit_comparison"], "instrumentation_effect": run["instrumentation_effect"]}, indent=2), "```"]
    lines += ["", "Full verified source/output mappings, QDQ adjacency, data types, source topological indices, execution metadata and hashes are in diagnosis.json.", "",
              "Appending outputs can prevent fusion or change execution precision. These debug observations alone cannot identify an unobserved canonical intermediate.",
              "No unsupported-op or mapping failure was suppressed; such failures terminate the run. No full benchmark was run."]
    atomic_text(stage / "diagnosis.md", "\n".join(lines) + "\n")


def quantizer_control(stage, model, run, config):
    first = run["first_integer_difference"]["first_divergent"]
    if first is None or first["producer_op"] != "QuantizeLinear":
        return {"status": "unavailable", "reason": "no differing quantizer code in the narrowed window"}
    node = model.graph.node[first["topological_index"]]
    inputs = [p for p in run["probes"] if p["source_value"] == node.input[0]]
    if len(inputs) != 1:
        return {"status": "unavailable", "reason": "quantizer input is not exposed in this pass"}
    input_index, output_index = inputs[0]["instrumented_output_index"], first["instrumented_output_index"]
    ort_dir, ov_dir = stage / (run["name"] + "_ort"), stage / (run["name"] + "_openvino")
    a, b = load_output(ort_dir, input_index), load_output(ov_dir, input_index)
    qa, qb = load_output(ort_dir, output_index), load_output(ov_dir, output_index)
    constants = {p.name: onnx.numpy_helper.to_array(p) for p in model.graph.initializer}
    boundary = rounding_boundary_evidence(a, b, qa, qb, constants[node.input[1]], constants[node.input[2]])
    control = quantizer_control_model(model, first["topological_index"], a.shape)
    control_path = stage / "same_input_quantizer.onnx"
    onnx.save(control, str(control_path))
    # Both runtimes get the SAME saved ORT Conv output, not their own outputs.
    common_input = ort_dir / f"{input_index:03d}.npy"
    ort_info, control_ort = execute(stage, "quantizer_control_ort", "ort", control_path, list(node.output), config, common_input)
    ov_info, control_ov = execute(stage, "quantizer_control_openvino", "openvino", control_path, list(node.output), config, common_input)
    ca, cb = load_output(control_ort, 0), load_output(control_ov, 0)
    return {"status": "observed", "source_node": first["producer_name"], "source_node_index": first["topological_index"],
            "control_onnx_sha256": sha256(control_path), "identical_input_digest": array_digest(a),
            "source_boundary_evidence": boundary, "same_input_metrics": metrics(ca, cb),
            "same_input_boundary_evidence": rounding_boundary_evidence(a, a, ca, cb, constants[node.input[1]], constants[node.input[2]]),
            "ort_execution": ort_info, "openvino_execution": ov_info,
            "scope": "Isolated copy of the source quantizer with unchanged attributes/scales/zero-point. No model repair."}


def diagnose(args, stage):
    no_torch()
    config = validate_config(json.loads(args.config.read_text()))
    proof = verify_source(ROOT, config)
    model = onnx.load(proof["path"])
    official = json.loads(Path(proof["source_report"]).read_text())
    manifest = json.loads(args.manifest.read_text())
    base_path = ROOT / manifest["baseline_onnx_model"]["logical_id"]
    if sha256(args.manifest) != official["artifact_sha256"]["manifest"] or sha256(base_path) != official["artifact_sha256"]["base_onnx"]:
        raise ValueError("semantic provenance differs from v0.6.5")
    sites = semantic_sites(model, onnx.load(str(base_path)), manifest)
    ov = require_openvino()
    fingerprint = ir_fingerprint(proof["sha256"], ov.get_version(), config["conversion"])
    bundles = [p for p in sorted((ROOT / config["ir_output"]).glob("*")) if valid_ir(p, fingerprint)]
    if not bundles:
        raise ValueError("no verified canonical v0.8 IR available; diagnosis will not create/overwrite canonical IR")
    canonical_ir = bundles[-1] / "model.xml"
    canonical_hashes = (sha256(canonical_ir), sha256(canonical_ir.with_suffix(".bin")))
    images, labels = load_cifar_batch(ROOT / config["data_root"], "test_batch")
    images, labels = next(numpy_batches(images, labels, list(range(128)), 128))
    np.save(stage / "inputs.npy", images, allow_pickle=False)
    np.save(stage / "labels.npy", labels, allow_pickle=False)
    output_names = [v.name for v in model.graph.output]
    ort_info, ort_dir = execute(stage, "baseline_ort", "ort", proof["path"], output_names, config)
    ov_info, ov_dir = execute(stage, "baseline_openvino", "openvino", canonical_ir, output_names, config)
    baseline_logits = (load_output(ort_dir, 0), load_output(ov_dir, 0))
    result = {"schema": "openvino-diagnosis/v0.8.1", "acceptance_tolerance": None,
              "provenance": {"source": proof, "canonical_ir": str(canonical_ir), "xml_sha256": canonical_hashes[0],
                             "bin_sha256": canonical_hashes[1], "manifest_sha256": sha256(args.manifest),
                             "openvino": ov.get_version(), "onnxruntime": ort.__version__, "onnx": onnx.__version__,
                             "device": "CPU", "compile_properties": config["compile_properties"],
                             "test_batch_sha256": sha256(ROOT / config["data_root"] / "cifar-10-batches-py/test_batch"),
                             "input_digest": array_digest(images), "label_digest": array_digest(labels),
                             "input_shape": list(images.shape), "indices": list(range(128)),
                             "script_sha256": sha256(Path(__file__)),
                             "diagnostic_source_sha256": sha256(ROOT / "src/silu_benchmark/openvino_diagnosis.py")},
              "baseline_execution": {"ort": ort_info, "openvino": ov_info},
              "baseline_comparison": compare_outputs(*baseline_logits, labels), "runs": []}
    coarse = paired_run(stage, "checkpoints", model, select_probes(model, sites), config, labels, baseline_logits)
    result["runs"].append(coarse)
    first = coarse["first_input_dependent_difference"]["first_divergent"]
    if first is not None:
        end = min(first["topological_index"] + 8, len(model.graph.node) - 1)
        result["runs"].append(paired_run(stage, "narrowed_prefix", model, select_probes(model, sites, 0, end), config, labels, baseline_logits))
    if args.node_start is not None or args.node_end is not None:
        probes = select_probes(model, sites, args.node_start, args.node_end)
        result["runs"].append(paired_run(stage, "requested_window", model, probes, config, labels, baseline_logits))
    narrowed = result["runs"][1] if len(result["runs"]) > 1 else coarse
    result["localization"] = narrowed["first_input_dependent_difference"]
    result["first_integer_difference"] = narrowed["first_integer_difference"]
    result["quantizer_control"] = quantizer_control(stage, model, narrowed, config)
    first = result["localization"]["first_divergent"]
    if first:
        control = result["quantizer_control"]
        control_differs = control["status"] == "observed" and not control["same_input_metrics"]["exact_equal"]
        likely = ("The identical-input control also produces different integer codes with unchanged source scales/zero-point. "
                  "Evidence points to backend QuantizeLinear/FakeQuantize lowering/rounding differences, not solely preceding Conv arithmetic. "
                  if control_differs else "The cause of the quantization discrepancy remains undetermined. ")
        result["conclusion"] = (f"First observed input-dependent difference in the narrowed debug graph: {first['source_value']} "
                                f"({first['producer_op']}, zero-based ONNX node {first['topological_index']}). " + likely +
                                "The exact contribution to the original end-to-end discrepancy remains undetermined: "
                                "instrumentation changes OpenVINO fusion/output behavior, so debug localization is not proof of the "
                                "first differing hidden value in the uninstrumented compiled graph. No mapping/unsupported-op failure "
                                "was observed, no acceptance tolerance is set, and backend equivalence is not claimed.")
    else:
        result["conclusion"] = "No difference at the selected instrumented checkpoints; original discrepancy cause undetermined."
    if (sha256(proof["path"]) != proof["sha256"] or canonical_hashes != (sha256(canonical_ir), sha256(canonical_ir.with_suffix(".bin")))):
        raise RuntimeError("canonical model changed during diagnosis")
    no_torch()
    result["torch_imported"] = False
    write_report(stage, result)


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        worker(json.loads(args.worker.read_text()))
        return 0
    root = diagnostic_path(ROOT, args.output)
    stage = root.with_name(root.name + "-" + uuid.uuid4().hex[:12])
    stage.mkdir(parents=True, exist_ok=False)
    atomic_json(stage / "run_status.json", {"status": "running", "samples": 128})
    print("Diagnostic output: " + str(stage), flush=True)
    try:
        diagnose(args, stage)
    except BaseException as exc:
        atomic_json(stage / "run_status.json", {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        raise
    atomic_json(stage / "run_status.json", {"status": "success", "completion_status": "diagnostic-smoke-success", "samples": 128})
    print("Diagnosis complete: " + str(stage / "diagnosis.md"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
