"""Calibrate, rewrite, and evaluate the v1.8 portable SiLU-aware QDQ model."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from silu_benchmark.accuracy_recovery import (
    calibrate_candidate_spec,
    canonical_hash,
    discover_qdq_silu_sites,
    instrument_outputs,
)
from silu_benchmark.benchmark_data import (
    load_cifar_batch,
    normalize_cifar_images,
)
from silu_benchmark.hardware_aware_qdq_model import (
    rewrite_standard_qdq_parameters,
    validate_standard_qdq_rewrite,
)
from silu_benchmark.qnn_local_accuracy import (
    evaluate_session,
    prediction_agreement,
    sha256_file,
    validate_model_io,
)
from silu_benchmark.quantization.hardware_aware_qdq import (
    calibrate_silu_aware_standard_qdq,
    qdq_spec_from_manifest,
)


CONFIG_SCHEMA = "hardware-aware-standard-qdq-config/v1.8"
REPORT_SCHEMA = "hardware-aware-standard-qdq-local-evaluation/v1.8"
DEFAULT_CONFIG = Path("configs/calibration/resnet18_silu_aware_standard_qdq_v18.json")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate one standard uint8 QDQ pair per SiLU output and evaluate "
            "the unchanged ONNX operator topology with ORT CPU."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--evaluation-samples",
        type=int,
        default=0,
        help="Use the first N test samples; 0 evaluates all 10,000.",
    )
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing generated outputs.",
    )
    return parser.parse_args(argv)


def load_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("invalid v1.8 hardware-aware QDQ config schema")
    if payload.get("provider") != "CPUExecutionProvider":
        raise ValueError("v1.8 local calibration requires CPUExecutionProvider")
    if payload.get("calibration_samples") != 2560:
        raise ValueError("v1.8 calibration must use exactly 2,560 training images")
    if payload.get("calibration_batch") != "data_batch_1":
        raise ValueError("v1.8 calibration must use the verified data_batch_1")
    if payload.get("evaluation_batch") != "test_batch":
        raise ValueError("v1.8 evaluation must use the verified test_batch")
    search = payload.get("standard_qdq_search", {})
    if search.get("bits") != 8 or float(search.get("central_weight", 0.0)) < 1.0:
        raise ValueError("v1.8 requires uint8 QDQ and central_weight >= 1")
    return payload


def _atomic_text(path: Path, value: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to replace existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, newline=""
    ) as temporary:
        temporary.write(value)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _atomic_json(path: Path, payload: dict, *, force: bool) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n", force=force)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to replace existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _preflight_outputs(paths: list[Path], *, force: bool) -> None:
    if len(set(paths)) != len(paths):
        raise ValueError("generated output paths must be unique")
    existing = [str(path) for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError(f"refusing to replace existing outputs: {existing}")


def _save_model(model: onnx.ModelProto, path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to replace existing model: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as temporary:
        temporary.write(model.SerializeToString())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)
    onnx.checker.check_model(onnx.load(str(path)))


def _session(source) -> ort.InferenceSession:
    value = str(source) if isinstance(source, Path) else source.SerializeToString()
    session = ort.InferenceSession(value, providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("ORT session did not stay on CPUExecutionProvider")
    return session


def collect_post_silu_values(
    model: onnx.ModelProto,
    images: np.ndarray,
    *,
    sample_count: int,
    batch_size: int,
    per_site_limit: int,
) -> tuple[tuple, dict[str, np.ndarray]]:
    """Collect deterministic pre-output-QDQ SiLU values from training data."""

    if not 0 < sample_count <= len(images):
        raise ValueError("invalid calibration sample count")
    if batch_size <= 0 or per_site_limit <= 0:
        raise ValueError("calibration batch size and per-site limit must be positive")
    sites = discover_qdq_silu_sites(model)
    mappings = {site.site_id: site.mul_output_tensor for site in sites}
    instrumented = instrument_outputs(model, mappings)
    session = _session(instrumented)
    input_name = session.get_inputs()[0].name
    output_names = [item.name for item in session.get_outputs()]
    if output_names != list(mappings.values()):
        raise RuntimeError("instrumented SiLU output order changed")

    batch_count = int(np.ceil(sample_count / batch_size))
    per_batch = max(1, int(np.ceil(per_site_limit / batch_count)))
    collected: dict[str, list[np.ndarray]] = {site.site_id: [] for site in sites}
    for offset in range(0, sample_count, batch_size):
        stop = min(offset + batch_size, sample_count)
        batch = normalize_cifar_images(images[offset:stop])
        outputs = session.run(output_names, {input_name: batch})
        for site, values in zip(sites, outputs):
            flat = np.asarray(values, dtype=np.float32).reshape(-1)
            take = min(per_batch, flat.size)
            indices = np.linspace(0, flat.size - 1, take, dtype=np.int64)
            collected[site.site_id].append(flat[indices])
    merged = {
        site.site_id: np.concatenate(collected[site.site_id])[:per_site_limit]
        for site in sites
    }
    if any(values.size == 0 for values in merged.values()):
        raise RuntimeError("one or more SiLU sites produced no calibration values")
    return sites, merged


def calibrate_sites(
    sites,
    site_values: dict[str, np.ndarray],
    config: dict,
) -> tuple[dict, dict]:
    hint_config = config["piecewise_hint"]
    search = config["standard_qdq_search"]
    records = []
    specs = {}
    for site in sites:
        values = site_values[site.site_id]
        hint = calibrate_candidate_spec(values, hint_config)
        result = calibrate_silu_aware_standard_qdq(
            values,
            hint,
            bits=int(search["bits"]),
            lower_steps=int(search["lower_steps"]),
            upper_steps=int(search["upper_steps"]),
            central_weight=float(search["central_weight"]),
            max_samples=int(search["max_samples"]),
        )
        specs[site.site_id] = qdq_spec_from_manifest(result)
        records.append(
            {
                "site_id": site.site_id,
                "module_path": site.module_path,
                "call_index": site.call_index,
                "post_silu_tensor": site.mul_output_tensor,
                "piecewise_hint": asdict(hint),
                "standard_qdq_calibration": result,
            }
        )
    digest = canonical_hash({"sites": records})
    return {
        "schema_version": "hardware-aware-standard-qdq-calibration-manifest/v1.8",
        "calibration_digest": digest,
        "target_site_count": len(records),
        "sites": records,
    }, specs


def _model_evaluation(path: Path, images: np.ndarray, labels: np.ndarray, batch_size: int):
    session = _session(path)
    contract = validate_model_io(session)
    metrics, predictions = evaluate_session(session, images, labels, batch_size=batch_size)
    if metrics["inference_failure_sample_count"] or metrics["nonfinite_sample_count"]:
        raise RuntimeError(f"model evaluation failed integrity checks: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "io": contract,
        **metrics,
    }, predictions


def _markdown(report: dict) -> str:
    lines = [
        "# v1.8 hardware-aware standard QDQ local evaluation",
        "",
        "The piecewise SiLU ranges are calibration hints only. The emitted model keeps the original standard ONNX QDQ operator topology.",
        "",
        "| Model | Correct | Samples | Top-1 | Delta vs standard QDQ | Agreement vs standard QDQ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_id, result in report["models"].items():
        comparison = report["comparisons"][model_id]
        lines.append(
            f"| {model_id} | {result['correct']} | {result['total']} | "
            f"{result['top1_accuracy_percent']:.4f}% | "
            f"{comparison['accuracy_delta_vs_standard_qdq_pp']:+.4f} pp | "
            f"{comparison['prediction_agreement_vs_standard_qdq_percent']:.4f}% |"
        )
    lines.extend(
        [
            "",
            "## Runtime contract",
            "",
            f"- Target SiLU QDQ pairs: {report['graph_contract']['target_qdq_pair_count']}",
            f"- Added runtime nodes: {report['graph_contract']['added_node_count']}",
            f"- Added custom runtime nodes: {report['graph_contract']['custom_runtime_nodes_added']}",
            "- Runtime parameters per target: one float32 scale and one uint8 zero-point",
            "- QNN compile/profile results are not claimed by this local ORT report.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict:
    config_path = (ROOT / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    source_path = (ROOT / config["source_standard_qdq_model"]).resolve()
    fp32_path = (ROOT / config["fp32_model"]).resolve()
    data_root = (ROOT / config["data_root"]).resolve()
    model_output = (ROOT / (args.model_output or Path(config["model_output"]))).resolve()
    manifest_output = (ROOT / (args.manifest_output or Path(config["manifest_output"]))).resolve()
    report_root = (ROOT / (args.report_output or Path(config["report_output"]))).resolve()
    if not source_path.exists() or not fp32_path.exists():
        raise FileNotFoundError("v1.8 requires the frozen FP32 and standard QDQ source models")
    source_hash = sha256_file(source_path)
    fp32_hash = sha256_file(fp32_path)
    if source_hash != config["source_standard_qdq_sha256"]:
        raise ValueError("frozen standard QDQ source SHA256 mismatch")
    if fp32_hash != config["fp32_model_sha256"]:
        raise ValueError("frozen FP32 source SHA256 mismatch")
    generated_paths = [
        model_output,
        manifest_output,
        report_root / "evaluation.json",
        report_root / "evaluation_summary.md",
        report_root / "predictions.npz",
    ]
    _preflight_outputs(generated_paths, force=args.force)

    training_images, _training_labels = load_cifar_batch(data_root, config["calibration_batch"])
    source_model = onnx.load(str(source_path))
    sites, values = collect_post_silu_values(
        source_model,
        training_images,
        sample_count=int(config["calibration_samples"]),
        batch_size=int(config["calibration_batch_size"]),
        per_site_limit=int(config["per_site_calibration_values"]),
    )
    manifest, specs = calibrate_sites(sites, values, config)
    manifest.update(
        {
            "configuration": config,
            "configuration_sha256": canonical_hash(config),
            "source_standard_qdq_model": {
                "path": str(source_path.relative_to(ROOT)),
                "sha256": source_hash,
            },
            "calibration_data": {
                "batch": config["calibration_batch"],
                "samples": config["calibration_samples"],
                "selection": "first 2,560 official CIFAR-10 data_batch_1 samples",
                "raw_batch_sha256": sha256_file(data_root / "cifar-10-batches-py" / config["calibration_batch"]),
            },
        }
    )
    rewritten = rewrite_standard_qdq_parameters(
        source_model, specs, calibration_digest=manifest["calibration_digest"]
    )
    _save_model(rewritten.model, model_output, force=args.force)
    graph_contract = validate_standard_qdq_rewrite(source_model, onnx.load(str(model_output)))
    manifest["generated_model"] = {
        "path": str(model_output.relative_to(ROOT)),
        "sha256": sha256_file(model_output),
        "graph_contract": graph_contract,
    }
    _atomic_json(manifest_output, manifest, force=args.force)

    test_images, test_labels = load_cifar_batch(data_root, config["evaluation_batch"])
    if args.evaluation_samples < 0 or args.evaluation_samples > len(test_images):
        raise ValueError("evaluation-samples must be between 0 and 10,000")
    count = args.evaluation_samples or len(test_images)
    test_images = test_images[:count]
    test_labels = test_labels[:count]
    paths = {
        "fp32": fp32_path,
        "standard_qdq": source_path,
        "silu_aware_standard_qdq": model_output,
    }
    model_results = {}
    predictions = {}
    for model_id, path in paths.items():
        model_results[model_id], predictions[model_id] = _model_evaluation(
            path, test_images, test_labels, int(config["evaluation_batch_size"])
        )
    reference = model_results["standard_qdq"]
    comparisons = {}
    for model_id, result in model_results.items():
        agreement = prediction_agreement(predictions["standard_qdq"], predictions[model_id])
        comparisons[model_id] = {
            "accuracy_delta_vs_standard_qdq_pp": 100.0
            * (result["top1_accuracy"] - reference["top1_accuracy"]),
            "prediction_agreement_vs_standard_qdq": agreement["agreement"],
            "prediction_agreement_vs_standard_qdq_percent": agreement["agreement_percent"],
            "prediction_disagreement_indices": agreement["disagreement_indices"],
        }
    report = {
        "schema_version": REPORT_SCHEMA,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "provider": "CPUExecutionProvider",
        },
        "evaluation": {
            "dataset": "official CIFAR-10 test_batch",
            "samples": count,
            "selection": f"original ordered indices 0 through {count - 1}",
            "batch_size": int(config["evaluation_batch_size"]),
        },
        "calibration_manifest": {
            "path": str(manifest_output.relative_to(ROOT)),
            "sha256": sha256_file(manifest_output),
            "calibration_digest": manifest["calibration_digest"],
        },
        "graph_contract": graph_contract,
        "models": model_results,
        "comparisons": comparisons,
        "limitations": [
            "This report is local ONNX Runtime CPU evidence, not QNN device evidence.",
            "The piecewise ranges guide offline calibration only and are absent from runtime dispatch.",
            "The selected thresholds are specific to the frozen model and calibration set.",
        ],
    }
    _atomic_json(report_root / "evaluation.json", report, force=args.force)
    _atomic_text(report_root / "evaluation_summary.md", _markdown(report), force=args.force)
    prediction_path = report_root / "predictions.npz"
    _atomic_npz(
        prediction_path,
        {"labels": test_labels, **predictions},
        force=args.force,
    )
    return report


def main() -> None:
    report = run(parse_args())
    result = report["models"]["silu_aware_standard_qdq"]
    comparison = report["comparisons"]["silu_aware_standard_qdq"]
    print(
        json.dumps(
            {
                "samples": result["total"],
                "top1_accuracy_percent": result["top1_accuracy_percent"],
                "accuracy_delta_vs_standard_qdq_pp": comparison[
                    "accuracy_delta_vs_standard_qdq_pp"
                ],
                "prediction_agreement_vs_standard_qdq_percent": comparison[
                    "prediction_agreement_vs_standard_qdq_percent"
                ],
                "added_runtime_nodes": report["graph_contract"]["added_node_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
