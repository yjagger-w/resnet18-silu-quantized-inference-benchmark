"""Optional OpenVINO CPU adapter for the proven v0.6.5 Standard-QDQ artifact only.

OpenVINO is imported only when conversion/inference is requested. No Torch,
recalibration, custom-piecewise conversion or device fallback is performed.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import tempfile
import uuid
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np


INSTALL_VERSION = "2025.3.0"  # Published CPython 3.9 Windows x86-64 wheel.
SOURCE_VARIANT = "standard_static_qdq_ort_cpu"
CONVERSION = {
    "frontend": "Core.read_model", "serializer": "serialize",
    "ir_version": "IR_V11", "compress_to_fp16": False,
}
QDQ = {"activation_type": "QUInt8", "calibration": "MinMax", "format": "QDQ",
       "per_channel": True, "weight_type": "QInt8"}
PROTOCOL_KEYS = (
    "model_architecture", "checkpoint", "dataset", "data_root", "evaluation_split",
    "preprocessing", "evaluation_samples", "evaluation_batch_size", "seed",
    "calibration_samples", "calibration_batch_size", "standard_qdq",
    "latency_batch_size", "latency_warmup_runs", "latency_timed_runs",
    "throughput_batch_size", "throughput_warmup_runs", "throughput_timed_runs",
)
CAPABILITY_CAVEAT = (
    "IR quantization nodes and compiled properties do not prove that every operation "
    "executes as INT8 on this CPU. No GPU, NPU, QNN or custom-piecewise deployment is tested."
)


class OpenVINOUnavailableError(RuntimeError):
    pass


def installation_command():
    return f'& \'{sys.executable}\' -m pip install "openvino=={INSTALL_VERSION}"'


def require_openvino():
    try:
        return importlib.import_module("openvino")
    except ImportError as exc:
        raise OpenVINOUnavailableError(
            "OpenVINO is unavailable in this interpreter; no conversion or measurement was made. "
            "Do not install automatically. After manual approval, use PowerShell: "
            + installation_command() + f". Import detail: {exc}"
        ) from exc


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def generated_path(root, value, kind):
    """Constrain CLI overrides to ignored v0.8 outputs, resolving '..' and symlinks."""
    root = Path(root).resolve()
    path = (root / value).resolve()
    base = root / ("artifacts/openvino/v0.8" if kind == "ir" else "results/benchmarks")
    try:
        relative = path.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"{kind} output must stay under {base}") from exc
    if kind == "report" and (not relative.parts or not relative.parts[0].startswith("v0.8")):
        raise ValueError("report output must be a v0.8 directory, never v0.6.5")
    return path


def validate_config(config):
    required = set(PROTOCOL_KEYS) | {
        "schema_version", "device", "source_model", "source_sha256", "source_report",
        "source_sidecar", "ir_output", "output_directory", "conversion", "compile_properties",
        "validation_batch_size", "tolerance",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("v0.8 config missing: " + ", ".join(missing))
    if config["schema_version"] != "benchmark/v0.8-openvino-cpu" or config["device"] != "CPU":
        raise ValueError("v0.8 supports only OpenVINO CPU")
    if config["model_architecture"] != "ResNet18-SiLU-CIFAR10" or config["dataset"] != "CIFAR-10":
        raise ValueError("v0.8 requires ResNet18-SiLU/CIFAR-10")
    if config["standard_qdq"] != QDQ or config["calibration_samples"] != 2560:
        raise ValueError("source must be static MinMax QUInt8/per-channel QInt8 with 2,560 calibration images")
    if config["evaluation_split"] != "test" or config["evaluation_samples"] != 10000:
        raise ValueError("official protocol must use all 10,000 CIFAR-10 test images")
    if config["conversion"] != CONVERSION:
        raise ValueError("v0.8 conversion settings are frozen; no extra weight compression")
    if config["compile_properties"] != {"PERFORMANCE_HINT": "LATENCY"}:
        raise ValueError("v0.8 uses synchronous inference with PERFORMANCE_HINT=LATENCY")
    if config["tolerance"] is not None:
        raise ValueError("No acceptance tolerance is established: observe real outputs before defining one")
    for key in ("evaluation_batch_size", "calibration_batch_size", "validation_batch_size",
                "latency_batch_size", "latency_timed_runs", "throughput_batch_size", "throughput_timed_runs"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("latency_warmup_runs", "throughput_warmup_runs"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if config["latency_batch_size"] != 1 or config["validation_batch_size"] != 128:
        raise ValueError("latency batch must be 1; deterministic validation batch must be 128")
    if config["throughput_batch_size"] > config["evaluation_batch_size"]:
        raise ValueError("throughput batch exceeds the evaluation batch")
    source_hash = config["source_sha256"]
    if not isinstance(source_hash, str) or len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash):
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
    return dict(config)


def verify_source(root, config, model_path=None):
    """Prove source identity from the completed official report, sidecar and graph."""
    import onnx

    root = Path(root)
    source = (root / (model_path or config["source_model"])).resolve()
    report_path = root / config["source_report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    status = json.loads((report_path.parent / "run_status.json").read_text(encoding="utf-8"))
    if (report.get("completion_status") != "official-success" or report.get("smoke") is not False
            or status.get("status") != "success" or status.get("smoke") is not False
            or status.get("fingerprint") != report.get("fingerprint")):
        raise ValueError("source provenance must be a completed OFFICIAL v0.6.5 report")
    expected = config["source_sha256"]
    sidecar = json.loads((root / config["source_sidecar"]).read_text(encoding="utf-8"))
    if (sha256(source) != expected or report["artifact_sha256"]["standard_static_qdq"] != expected
            or sidecar.get("sha256") != expected or sidecar.get("fingerprint") != report["fingerprint"]):
        raise ValueError("Standard-QDQ source/report/sidecar provenance mismatch; do not regenerate or substitute models")
    for key in PROTOCOL_KEYS:
        if config[key] != report["frozen_config"][key]:
            raise ValueError(f"v0.8 must preserve the v0.6.5 protocol: {key}")
    if sha256(root / config["checkpoint"]) != report["artifact_sha256"]["checkpoint"]:
        raise ValueError("checkpoint hash differs from v0.6.5")
    if sha256(root / config["data_root"] / "cifar-10-batches-py/test_batch") != report["artifact_sha256"]["test_batch"]:
        raise ValueError("CIFAR-10 test data differs from v0.6.5")
    model = onnx.load(str(source))
    if any(prop.key == "silu_benchmark.piecewise_rewrite_sites" for prop in model.metadata_props):
        raise ValueError("custom SiLU piecewise conversion is outside v0.8 scope")
    onnx.checker.check_model(model)
    counts = dict(sorted(Counter(node.op_type for node in model.graph.node).items()))
    if not counts.get("QuantizeLinear") or not counts.get("DequantizeLinear"):
        raise ValueError("source is not a Standard-QDQ ONNX graph")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("expected one CIFAR-10 input and one logits output")
    official = next(row for row in report["variants"] if row["variant_id"] == SOURCE_VARIANT)
    return {
        "path": str(source), "sha256": expected, "model_size_bytes": source.stat().st_size,
        "operator_histogram": counts, "q_nodes": counts["QuantizeLinear"], "dq_nodes": counts["DequantizeLinear"],
        "source_report": str(report_path), "source_report_sha256": sha256(report_path),
        "official_ort_v065": {"variant_id": SOURCE_VARIANT, "accuracy": official["accuracy"],
                              "samples": official["samples"], "label": "historical official ORT; not a new OpenVINO measurement"},
    }


def ir_fingerprint(source_sha256, version, conversion=CONVERSION, device="CPU"):
    if device != "CPU":
        raise ValueError("v0.8 supports only CPU")
    payload = {"schema": "openvino-ir/v0.8", "source_sha256": source_sha256,
               "openvino_version": version, "device": device, "conversion": conversion,
               "adapter_sha256": sha256(Path(__file__))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def valid_ir(bundle, fingerprint):
    try:
        bundle = Path(bundle)
        metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
        return (isinstance(metadata, dict) and metadata.get("status") == "complete"
                and metadata.get("fingerprint") == fingerprint
                and metadata.get("xml_sha256") == sha256(bundle / "model.xml")
                and metadata.get("bin_sha256") == sha256(bundle / "model.bin"))
    except (OSError, ValueError, TypeError):
        return False


def ir_graph_report(xml_path):
    graph = ET.parse(xml_path).getroot()
    if graph.tag != "net" or graph.find("layers") is None:
        raise ValueError("not an OpenVINO IR XML graph")
    layers = graph.findall("./layers/layer")
    counts = dict(sorted(Counter(layer.get("type", "unknown") for layer in layers).items()))
    types = Counter()
    for layer in layers:
        data = layer.find("data")
        if data is not None and data.get("element_type"):
            types[data.get("element_type")] += 1
    related = {key: value for key, value in counts.items()
               if "quantiz" in key.lower() or key in ("Convert", "ConvertLike")}
    return {"ir_version": graph.get("version"), "operation_count": len(layers),
            "operation_types": counts, "fake_quantize_present": counts.get("FakeQuantize", 0) > 0,
            "quantization_related_operations": related, "declared_element_types": dict(types),
            "caveat": CAPABILITY_CAVEAT}


def convert_standard_qdq(root, config, source, output_dir=None, resume=False, force_rebuild=False):
    """Convert into a new immutable bundle; never overwrite original ONNX or earlier IR."""
    if config["device"] != "CPU" or config["conversion"] != CONVERSION:
        raise ValueError("only the frozen CPU conversion is supported")
    if source["sha256"] != config["source_sha256"] or sha256(source["path"]) != config["source_sha256"]:
        raise ValueError("only the exact verified Standard-QDQ source may be converted")
    ov = require_openvino()
    version = ov.get_version()
    core = ov.Core()
    available = list(core.available_devices)
    if "CPU" not in available:
        raise RuntimeError(f"OpenVINO CPU plugin unavailable; devices={available}. Check the CPU runtime installation.")
    fingerprint = ir_fingerprint(source["sha256"], version, config["conversion"])
    target = generated_path(root, output_dir or config["ir_output"], "ir")
    if resume and not force_rebuild and target.exists():
        for candidate in sorted(target.glob(fingerprint[:16] + "-*"), reverse=True):
            if valid_ir(candidate, fingerprint):
                return {"directory": str(candidate), "reused": True,
                        **json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))}
    target.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".partial-", dir=target))
    try:
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError("ONNX source changed after provenance verification")
        # Native ONNX frontend: no PyTorch/framework converter or Python hooks.
        model = core.read_model(source["path"])
        ov.serialize(model, str(stage / "model.xml"), str(stage / "model.bin"), version="IR_V11")
        facts = ir_graph_report(stage / "model.xml")
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError("ONNX source changed during conversion")
        payload = {
            "status": "complete", "fingerprint": fingerprint, "source_onnx_sha256": source["sha256"],
            "conversion": config["conversion"], "openvino_version": version, "device": "CPU",
            "available_devices": available, "xml_sha256": sha256(stage / "model.xml"),
            "bin_sha256": sha256(stage / "model.bin"), "ir_graph": facts,
        }
        atomic_json(stage / "metadata.json", payload)
        destination = target / (fingerprint[:16] + "-" + uuid.uuid4().hex[:12])
        stage.replace(destination)
        return {"directory": str(destination), "reused": False, **payload}
    except Exception as exc:
        atomic_json(stage / "failure.json", {"status": "failed", "phase": "ONNX-to-IR conversion", "error": str(exc)})
        raise RuntimeError(f"Standard-QDQ ONNX-to-IR conversion failed: {exc}. Source is unchanged; inspect {stage}.") from exc


def safe_properties(getter, names):
    values = {}
    for name in names:
        try:
            value = getter(name)
            values[name] = {"status": "available", "value": value if isinstance(value, (str, int, float, bool, type(None))) else str(value)}
        except Exception as exc:
            values[name] = {"status": "unavailable", "reason": str(exc)}
    return values


class OpenVINOCPU:
    def __init__(self, bundle, properties, device="CPU"):
        if device != "CPU":
            raise ValueError("v0.8 supports only CPU")
        ov = require_openvino()
        directory = Path(bundle["directory"])
        if not valid_ir(directory, bundle["fingerprint"]):
            raise ValueError("IR is stale or incomplete; rerun conversion with --force-rebuild")
        core = ov.Core()
        try:
            model = core.read_model(str(directory / "model.xml"), str(directory / "model.bin"))
            self.compiled = core.compile_model(model, "CPU", properties)
            if len(self.compiled.inputs) != 1 or len(self.compiled.outputs) != 1:
                raise ValueError("expected one input and one output")
            self.request = self.compiled.create_infer_request()
        except Exception as exc:
            raise RuntimeError(f"OpenVINO CPU compile failed: {exc}. Check CPU plugin/operator support; no device fallback is used.") from exc
        self.capabilities = {
            "device": "CPU", "available_devices": list(core.available_devices), "requested_properties": properties,
            "plugin_versions": safe_properties(lambda _name: core.get_versions("CPU"), ("CPU",)),
            "device_properties": safe_properties(lambda name: core.get_property("CPU", name),
                                                  ("FULL_DEVICE_NAME", "OPTIMIZATION_CAPABILITIES", "SUPPORTED_PROPERTIES")),
            "compiled_properties": safe_properties(self.compiled.get_property,
                ("EXECUTION_DEVICES", "PERFORMANCE_HINT", "INFERENCE_PRECISION_HINT", "NUM_STREAMS", "INFERENCE_NUM_THREADS", "OPTIMAL_NUMBER_OF_INFER_REQUESTS")),
            "caveat": CAPABILITY_CAVEAT,
        }

    def predict(self, images):
        images = np.ascontiguousarray(images, dtype=np.float32)
        try:
            outputs = self.request.infer({0: images}, share_inputs=False, share_outputs=False)
            return np.asarray(outputs[self.compiled.output(0)])
        except Exception as exc:
            raise RuntimeError(f"OpenVINO CPU inference failed for shape {images.shape}: {exc}") from exc


def compare_outputs(ort_logits, ov_logits, labels):
    ort_logits, ov_logits, labels = np.asarray(ort_logits), np.asarray(ov_logits), np.asarray(labels)
    if ort_logits.shape != ov_logits.shape or ort_logits.ndim != 2 or ort_logits.shape[1] != 10:
        raise ValueError("ORT/OpenVINO outputs must have compatible [N,10] logits")
    if not len(labels) or labels.shape != (len(ort_logits),) or not np.all((labels >= 0) & (labels < 10)):
        raise ValueError("labels must match the nonempty validation batch")
    if not np.all(np.isfinite(ort_logits)) or not np.all(np.isfinite(ov_logits)):
        raise ValueError("non-finite validation logits")
    error = np.abs(ort_logits.astype(np.float64) - ov_logits.astype(np.float64))
    ort_pred, ov_pred = ort_logits.argmax(1), ov_logits.argmax(1)
    disagreements = np.flatnonzero(ort_pred != ov_pred).tolist()
    return {
        "samples": len(labels), "logit_max_absolute_error": float(error.max()),
        "logit_mean_absolute_error": float(error.mean()), "prediction_agreement": float(np.mean(ort_pred == ov_pred)),
        "prediction_disagreement_indices": disagreements, "ort_predictions": ort_pred.tolist(),
        "openvino_predictions": ov_pred.tolist(), "ort_top1_accuracy": float(np.mean(ort_pred == labels)),
        "openvino_top1_accuracy": float(np.mean(ov_pred == labels)),
        "tolerance": {"status": "not_established", "reason": "Observational comparison; no bit-exact or acceptance-threshold claim."},
    }
