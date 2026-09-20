"""Run or resume v1.6 Qualcomm AI Hub jobs and reproduce offline reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from silu_benchmark.backends.qnn_aihub_backend import (  # noqa: E402
    DEFAULT_DEVICE,
    DEFAULT_OS,
    QaiHubUnavailableError,
    QnnAiHubBackend,
    build_numerical_audit,
    build_profile_summary,
    job_markdown,
    job_record,
    load_manifest,
    numerical_audit_markdown,
    parse_input_specs,
    profile_summary_markdown,
    relative_path,
    repository_path,
    sha256_file,
    summarize_profile,
    validate_job_id,
    wait_for_success,
    write_reports,
)
from silu_benchmark.qnn_local_accuracy import (  # noqa: E402
    MODEL_ORDER,
    build_report as build_local_accuracy_report,
    evaluate_session as evaluate_local_accuracy_session,
    load_cifar10_test_set,
    ordered_dataset_fingerprint,
    prediction_agreement,
    preprocessing_metadata,
    sha256_file as local_sha256_file,
    validate_model_io,
    verify_sha256,
    write_accuracy_outputs,
)
from silu_benchmark.qnn_preflight import (  # noqa: E402
    FULL_EXPORT_SCHEMA,
    PREFLIGHT_SCHEMA,
    PREFLIGHT_MODEL_ORDER,
    evaluate_preprocessed_session,
    full_export_markdown,
    full_selection_manifest,
    prepare_full_test_set,
    prepare_preflight_subset,
    selection_manifest,
    write_preflight_outputs,
)
from silu_benchmark.qnn_s22_accuracy import (  # noqa: E402
    S22_MODEL_ORDER,
    build_s22_preflight_report,
    load_npz_exact,
    write_s22_accuracy_outputs,
)


DEFAULT_MANIFEST = "configs/qnn/resnet18_silu_qnn_v16.json"


def _common_online(parser: argparse.ArgumentParser, output_name: str) -> None:
    parser.add_argument("--job-id", help="resume an existing job instead of submitting")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--os", dest="os_version", default=DEFAULT_OS)
    parser.add_argument("--name", help="optional AI Hub job name")
    parser.add_argument("--options", default="", help="SDK job option string")
    parser.add_argument("--timeout", type=int, help="client-side wait timeout in seconds")
    parser.add_argument("--output-dir", default=f"out/qnn/v1.6/{output_name}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducible QNN AI Hub v1.6 orchestration and offline analysis"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile", help="submit or resume a compile job")
    _common_online(compile_parser, "compile")
    compile_parser.add_argument("--model", help="repository-relative ONNX model path")
    compile_parser.add_argument(
        "--input-spec",
        action="append",
        default=None,
        help="name=dim,dim,...:dtype (repeatable; default images=1,3,32,32:float32)",
    )
    compile_parser.set_defaults(options="--target_runtime qnn_dlc")

    profile_parser = subparsers.add_parser("profile", help="submit or resume a profile job")
    _common_online(profile_parser, "profile")
    profile_parser.add_argument("--compile-job-id", help="successful compile job to profile")

    inference_parser = subparsers.add_parser(
        "inference", help="submit or resume an inference job"
    )
    _common_online(inference_parser, "inference")
    inference_parser.add_argument("--compile-job-id", help="successful compile job to run")
    inference_parser.add_argument("--inputs", help="repository-relative NPZ input path")

    audit_parser = subparsers.add_parser(
        "numerical-audit", help="recompute the frozen offline numerical audit"
    )
    audit_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    audit_parser.add_argument("--inputs", help="override manifest input NPZ")
    audit_parser.add_argument("--outputs", help="override manifest output NPZ")
    audit_parser.add_argument("--output-dir", default="out/qnn/v1.6/numerical-audit")

    summary_parser = subparsers.add_parser(
        "profile-summary", help="summarize the three offline QNN profiles"
    )
    summary_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    summary_parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help="override a profile with model_id=repository/relative/profile.json",
    )
    summary_parser.add_argument("--output-dir", default="out/qnn/v1.6/profile-summary")

    accuracy_parser = subparsers.add_parser(
        "cifar10-accuracy",
        help="evaluate the three frozen source models locally with ORT CPU",
    )
    accuracy_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    accuracy_parser.add_argument("--data-root", default="data")
    accuracy_parser.add_argument("--batch-size", type=int, default=128)
    accuracy_parser.add_argument(
        "--output-dir",
        default="results/benchmarks/v1.6_qnn_cifar10_local_accuracy",
    )

    preflight_parser = subparsers.add_parser(
        "cifar10-preflight",
        help="export a deterministic balanced CIFAR-10 subset and local ORT references",
    )
    preflight_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    preflight_parser.add_argument("--data-root", default="data")
    preflight_parser.add_argument("--batch-size", type=int, default=128)
    preflight_parser.add_argument("--samples-per-class", type=int, default=100)
    preflight_parser.add_argument(
        "--output-dir",
        default="out/qnn/v1.6/cifar10-s22-preflight-1000",
    )

    s22_report_parser = subparsers.add_parser(
        "cifar10-s22-report",
        help="compare downloaded S22 outputs with the deterministic local preflight reference",
    )
    s22_report_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    s22_report_parser.add_argument(
        "--preflight-dir", default="out/qnn/v1.6/cifar10-s22-preflight-1000"
    )
    s22_report_parser.add_argument(
        "--fp32-remote", default="out/qnn/v1.6/cifar10-s22-preflight-1000/fp32_remote/outputs.npz"
    )
    s22_report_parser.add_argument(
        "--qdq-int8-remote",
        default="out/qnn/v1.6/cifar10-s22-preflight-1000/qdq_int8_remote/outputs.npz",
    )
    s22_report_parser.add_argument("--fp32-inference-job-id", default="jp0mdve2g")
    s22_report_parser.add_argument("--qdq-int8-inference-job-id", default="jgolo4e4g")
    s22_report_parser.add_argument(
        "--output-dir",
        default="results/benchmarks/v1.6_qnn_cifar10_s22_preflight_1000",
    )

    full_parser = subparsers.add_parser(
        "cifar10-full-export",
        help="export all 10,000 CIFAR-10 test inputs and local FP32/QDQ ORT references",
    )
    full_parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    full_parser.add_argument("--data-root", default="data")
    full_parser.add_argument("--batch-size", type=int, default=128)
    full_parser.add_argument(
        "--output-dir", default="out/qnn/v1.6/cifar10-s22-full-10000"
    )
    return parser.parse_args(argv)


def _output_dir(value: str) -> Path:
    return repository_path(ROOT, value, kind="output directory")


def _write_job(job: Any, args: argparse.Namespace, operation: str, extra: Mapping[str, Any] | None = None) -> dict:
    status = wait_for_success(job, args.timeout)
    backend_sdk = getattr(args, "_backend_sdk")
    record = job_record(
        job,
        operation=operation,
        resumed=bool(args.job_id),
        sdk=backend_sdk,
        device_name=args.device,
        os_version=args.os_version,
        status=status,
    )
    if extra:
        record.update(extra)
    output_dir = _output_dir(args.output_dir)
    write_reports(output_dir, "job", record, job_markdown(record))
    return record


def command_compile(args: argparse.Namespace) -> dict:
    if not args.job_id and not args.model:
        raise ValueError("--model is required unless --job-id resumes a compile job")
    model_path = None
    model_relative = None
    if args.model:
        model_path = repository_path(ROOT, args.model, must_exist=True, kind="model")
        model_relative = relative_path(ROOT, model_path)
    specs = parse_input_specs(args.input_spec or ["images=1,3,32,32:float32"])
    backend = QnnAiHubBackend()
    args._backend_sdk = backend.sdk
    job = backend.compile(
        model=model_path,
        input_specs=specs,
        device_name=args.device,
        os_version=args.os_version,
        name=args.name,
        options=args.options,
        job_id=args.job_id,
    )
    serializable_specs = {
        name: {"shape": list(spec[0]), "dtype": spec[1]} for name, spec in specs.items()
    }
    return _write_job(
        job,
        args,
        "compile",
        {"model": model_relative, "input_specs": serializable_specs},
    )


def command_profile(args: argparse.Namespace) -> dict:
    if not args.job_id and not args.compile_job_id:
        raise ValueError("--compile-job-id is required unless --job-id resumes a profile job")
    backend = QnnAiHubBackend()
    args._backend_sdk = backend.sdk
    job = backend.profile(
        compile_job_id=args.compile_job_id,
        device_name=args.device,
        os_version=args.os_version,
        name=args.name,
        options=args.options,
        job_id=args.job_id,
    )
    record = _write_job(job, args, "profile")
    profile = job.download_profile()
    if not isinstance(profile, Mapping):
        raise RuntimeError("profile job did not return an in-memory profile mapping")
    record["profile_summary"] = summarize_profile(profile)
    output_dir = _output_dir(args.output_dir)
    (output_dir / "profile.json").write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_reports(output_dir, "job", record, job_markdown(record))
    return record


def load_inference_inputs(path: Path) -> dict[str, list[Any]]:
    import numpy as np

    inputs: dict[str, list[Any]] = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            values = np.asarray(archive[name])
            if values.ndim < 1 or not len(values):
                raise ValueError(f"input {name} must have a non-empty sample dimension")
            inputs[name] = [np.ascontiguousarray(values[index : index + 1]) for index in range(len(values))]
    if not inputs:
        raise ValueError("input NPZ has no arrays")
    sample_counts = {len(values) for values in inputs.values()}
    if len(sample_counts) != 1:
        raise ValueError("all inference inputs must have the same number of samples")
    return inputs


def save_inference_outputs(path: Path, outputs: Mapping[str, Any]) -> dict:
    import numpy as np

    arrays = {}
    metadata = {}
    for name, value in outputs.items():
        if isinstance(value, (list, tuple)):
            pieces = [np.asarray(piece) for piece in value]
            if not pieces:
                raise ValueError(f"output {name} is empty")
            array = np.concatenate(pieces, axis=0)
        else:
            array = np.asarray(value)
        arrays[str(name)] = array
        metadata[str(name)] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    if not arrays:
        raise ValueError("inference job returned no output tensors")
    np.savez_compressed(path, **arrays)
    return metadata


def command_inference(args: argparse.Namespace) -> dict:
    if not args.job_id and (not args.compile_job_id or not args.inputs):
        raise ValueError(
            "--compile-job-id and --inputs are required unless --job-id resumes an inference job"
        )
    input_path = None
    inputs = None
    if args.inputs:
        input_path = repository_path(ROOT, args.inputs, must_exist=True, kind="input NPZ")
        inputs = load_inference_inputs(input_path)
    backend = QnnAiHubBackend()
    args._backend_sdk = backend.sdk
    job = backend.inference(
        inputs=inputs,
        compile_job_id=args.compile_job_id,
        device_name=args.device,
        os_version=args.os_version,
        name=args.name,
        options=args.options,
        job_id=args.job_id,
    )
    record = _write_job(
        job,
        args,
        "inference",
        {"inputs": relative_path(ROOT, input_path) if input_path else None},
    )
    outputs = job.download_output_data()
    if not isinstance(outputs, Mapping):
        raise RuntimeError("inference job did not return output tensors")
    output_dir = _output_dir(args.output_dir)
    record["outputs"] = save_inference_outputs(output_dir / "outputs.npz", outputs)
    write_reports(output_dir, "job", record, job_markdown(record))
    return record


def _load_npz(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _load_manifest_arg(value: str) -> tuple[Path, dict]:
    path = repository_path(ROOT, value, must_exist=True, kind="manifest")
    return path, load_manifest(path)


def command_numerical_audit(args: argparse.Namespace) -> dict:
    _, manifest = _load_manifest_arg(args.manifest)
    input_path = repository_path(
        ROOT,
        args.inputs or manifest["numerical_audit"]["inputs"],
        must_exist=True,
        kind="audit input NPZ",
    )
    output_path = repository_path(
        ROOT,
        args.outputs or manifest["numerical_audit"]["outputs"],
        must_exist=True,
        kind="audit output NPZ",
    )
    inputs_archive = _load_npz(input_path)
    if set(inputs_archive) != {"images"}:
        raise ValueError("audit input NPZ must contain only the images tensor")
    outputs = _load_npz(output_path)
    audit = build_numerical_audit(
        manifest,
        inputs_archive["images"],
        outputs,
        input_sha256=sha256_file(input_path).upper(),
        outputs_sha256=sha256_file(output_path).upper(),
        qai_hub_version=str(manifest["environment"]["qai_hub"]),
    )
    write_reports(
        _output_dir(args.output_dir),
        "numerical_audit",
        audit,
        numerical_audit_markdown(audit, manifest),
    )
    return audit


def _profile_overrides(values: list[str]) -> dict[str, str]:
    overrides = {}
    for value in values:
        try:
            model_id, path = value.split("=", 1)
        except ValueError as error:
            raise ValueError("--profile must use model_id=repository/relative/profile.json") from error
        if model_id in overrides:
            raise ValueError(f"duplicate profile override: {model_id}")
        overrides[model_id] = path
    return overrides


def command_profile_summary(args: argparse.Namespace) -> dict:
    _, manifest = _load_manifest_arg(args.manifest)
    overrides = _profile_overrides(args.profile)
    unknown = set(overrides) - set(manifest["models"])
    if unknown:
        raise ValueError(f"unknown profile model IDs: {sorted(unknown)}")
    profiles = {}
    for model_id, model in manifest["models"].items():
        path = repository_path(
            ROOT,
            overrides.get(model_id, model["profile_path"]),
            must_exist=True,
            kind=f"{model_id} profile",
        )
        profiles[model_id] = json.loads(path.read_text(encoding="utf-8"))
    summary = build_profile_summary(profiles)
    summary["device"] = dict(manifest["device"])
    summary["environment"] = dict(manifest["environment"])
    write_reports(
        _output_dir(args.output_dir),
        "profile_summary",
        summary,
        profile_summary_markdown(summary, manifest),
    )
    return summary


def command_cifar10_accuracy(args: argparse.Namespace) -> dict:
    import onnxruntime as ort

    manifest_path, manifest = _load_manifest_arg(args.manifest)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    data_root = repository_path(ROOT, args.data_root, must_exist=True, kind="data root")
    test_batch_path = data_root / "cifar-10-batches-py" / "test_batch"
    images, labels = load_cifar10_test_set(data_root)
    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CPUExecutionProvider is unavailable")

    model_results = {}
    predictions = {}
    for model_id in MODEL_ORDER:
        model_config = manifest["models"][model_id]
        model_path = repository_path(
            ROOT, model_config["source_path"], must_exist=True, kind=f"{model_id} model"
        )
        digest = verify_sha256(
            model_path, model_config["source_sha256"], label=f"{model_id} model"
        )
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        io_contract = validate_model_io(session)
        metrics, model_predictions = evaluate_local_accuracy_session(
            session, images, labels, batch_size=args.batch_size
        )
        model_results[model_id] = {
            "label": model_config["label"],
            "path": relative_path(ROOT, model_path),
            "sha256": digest,
            "io": io_contract,
            **metrics,
        }
        predictions[model_id] = model_predictions
        del session

    report = build_local_accuracy_report(
        manifest_path=relative_path(ROOT, manifest_path),
        manifest_sha256=local_sha256_file(manifest_path),
        data_root=relative_path(ROOT, data_root),
        test_batch_path=relative_path(ROOT, test_batch_path),
        test_batch_sha256=local_sha256_file(test_batch_path),
        images=images,
        labels=labels,
        batch_size=args.batch_size,
        ort_version=ort.__version__,
        available_providers=ort.get_available_providers(),
        model_results=model_results,
        predictions=predictions,
    )
    write_accuracy_outputs(_output_dir(args.output_dir), report, predictions, labels)
    return report


def _evaluate_export_models(experiment, inputs, labels, batch_size, ort):
    model_results = {}
    reference_arrays = {}
    for model_id in PREFLIGHT_MODEL_ORDER:
        model_config = experiment["models"][model_id]
        model_path = repository_path(
            ROOT, model_config["source_path"], must_exist=True, kind=f"{model_id} model"
        )
        digest = verify_sha256(
            model_path, model_config["source_sha256"], label=f"{model_id} model"
        )
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        io_contract = validate_model_io(session)
        metrics, predictions, logits = evaluate_preprocessed_session(
            session, inputs, labels, batch_size=batch_size
        )
        model_results[model_id] = {
            "label": model_config["label"],
            "path": relative_path(ROOT, model_path),
            "sha256": digest,
            "compile_job_id": model_config["jobs"]["compile"],
            "io": io_contract,
            **metrics,
        }
        reference_arrays[f"{model_id}_predictions"] = predictions
        reference_arrays[f"{model_id}_logits"] = logits
        del session
    return model_results, reference_arrays


def command_cifar10_preflight(args: argparse.Namespace) -> dict:
    import numpy as np
    import onnxruntime as ort

    manifest_path, experiment = _load_manifest_arg(args.manifest)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.samples_per_class != 100:
        raise ValueError("v1.6 Galaxy S22 preflight requires exactly 100 samples per class")
    data_root = repository_path(ROOT, args.data_root, must_exist=True, kind="data root")
    test_batch_path = data_root / "cifar-10-batches-py" / "test_batch"
    full_images, full_labels = load_cifar10_test_set(data_root)
    inputs, labels, original_indices = prepare_preflight_subset(
        full_images, full_labels, samples_per_class=args.samples_per_class
    )
    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CPUExecutionProvider is unavailable")

    model_results, reference_arrays = _evaluate_export_models(
        experiment, inputs, labels, args.batch_size, ort
    )

    comparison = prediction_agreement(
        reference_arrays["fp32_predictions"], reference_arrays["qdq_int8_predictions"]
    )
    selection = selection_manifest(
        full_images,
        full_labels,
        labels,
        original_indices,
        samples_per_class=args.samples_per_class,
    )
    validations = {
        "each_class_has_exactly_100_samples": selection["class_counts"]
        == {str(label): 100 for label in range(10)},
        "original_indices_are_unique": selection["indices_are_unique"],
        "original_indices_are_strictly_increasing": selection[
            "indices_are_strictly_increasing"
        ],
        "input_and_label_order_matches_test_batch": selection["labels_match_original_order"],
        "all_inputs_are_finite": bool(np.all(np.isfinite(inputs))),
        "all_logits_are_finite": all(
            bool(np.all(np.isfinite(reference_arrays[f"{model_id}_logits"])))
            for model_id in PREFLIGHT_MODEL_ORDER
        ),
    }
    manifest_base = {
        "schema_version": PREFLIGHT_SCHEMA,
        "scope": "offline data export and local ONNX Runtime CPU reference",
        "statements": {
            "ai_hub_connected": False,
            "ai_hub_tasks_created": False,
            "is_galaxy_s22_qnn_accuracy": False,
            "piecewise_reference_included": False,
        },
        "provenance": {
            "experiment_manifest_path": relative_path(ROOT, manifest_path),
            "experiment_manifest_sha256": local_sha256_file(manifest_path),
            "test_batch_path": relative_path(ROOT, test_batch_path),
            "test_batch_sha256": local_sha256_file(test_batch_path),
        },
        "dataset": {
            "name": "CIFAR-10",
            "split": "test",
            "sample_count": int(len(full_labels)),
            "ordered_dataset_fingerprint_sha256": ordered_dataset_fingerprint(
                full_images, full_labels
            ),
        },
        "selection": selection,
        "preprocessing": preprocessing_metadata(
            randomness="none; deterministic balanced prefix selection in official test_batch order"
        ),
        "runtime": {
            "numpy": np.__version__,
            "onnxruntime": ort.__version__,
            "requested_providers": ["CPUExecutionProvider"],
            "available_providers": ort.get_available_providers(),
            "batch_size": args.batch_size,
        },
        "models": model_results,
        "prediction_comparison": comparison,
        "validations": validations,
    }
    report, _ = write_preflight_outputs(
        _output_dir(args.output_dir),
        inputs=inputs,
        labels=labels,
        original_indices=original_indices,
        local_reference=reference_arrays,
        manifest_base=manifest_base,
    )
    return report


def command_cifar10_s22_report(args: argparse.Namespace) -> dict:
    import numpy as np

    manifest_path, experiment = _load_manifest_arg(args.manifest)
    preflight_dir = repository_path(
        ROOT, args.preflight_dir, must_exist=True, kind="preflight directory"
    )
    preflight_manifest_path = preflight_dir / "preflight_manifest.json"
    if not preflight_manifest_path.is_file():
        raise FileNotFoundError("preflight manifest does not exist")
    preflight = json.loads(preflight_manifest_path.read_text(encoding="utf-8"))
    labels_path = preflight_dir / "labels.npz"
    local_reference_path = preflight_dir / "local_reference.npz"
    inputs_path = preflight_dir / "inputs.npz"
    for key, path in (
        ("inputs", inputs_path),
        ("labels", labels_path),
        ("local_reference", local_reference_path),
    ):
        verify_sha256(path, preflight["artifacts"][key]["sha256"], label=key)

    labels_archive = load_npz_exact(labels_path, ("labels", "original_indices"))
    reference = load_npz_exact(
        local_reference_path,
        (
            "fp32_predictions",
            "fp32_logits",
            "qdq_int8_predictions",
            "qdq_int8_logits",
        ),
    )
    fp32_remote_path = repository_path(
        ROOT, args.fp32_remote, must_exist=True, kind="FP32 remote output"
    )
    qdq_remote_path = repository_path(
        ROOT, args.qdq_int8_remote, must_exist=True, kind="QDQ INT8 remote output"
    )
    fp32_remote = load_npz_exact(fp32_remote_path, ("output_0",))["output_0"]
    qdq_remote = load_npz_exact(qdq_remote_path, ("output_0",))["output_0"]
    local_logits = {
        "fp32": reference["fp32_logits"],
        "qdq_int8": reference["qdq_int8_logits"],
    }
    for model_id in S22_MODEL_ORDER:
        expected = np.argmax(local_logits[model_id], axis=1).astype(np.int64)
        if not np.array_equal(reference[f"{model_id}_predictions"], expected):
            raise ValueError(f"stored {model_id} local predictions do not match local logits")
    remote_logits = {"fp32": fp32_remote, "qdq_int8": qdq_remote}
    jobs = {
        "fp32": {
            "compile": experiment["models"]["fp32"]["jobs"]["compile"],
            "inference": validate_job_id(args.fp32_inference_job_id),
        },
        "qdq_int8": {
            "compile": experiment["models"]["qdq_int8"]["jobs"]["compile"],
            "inference": validate_job_id(args.qdq_int8_inference_job_id),
        },
    }
    provenance = {
        "experiment_manifest": {
            "path": relative_path(ROOT, manifest_path),
            "sha256": local_sha256_file(manifest_path),
        },
        "preflight_manifest": {
            "path": relative_path(ROOT, preflight_manifest_path),
            "sha256": local_sha256_file(preflight_manifest_path),
        },
        "inputs": {
            "path": relative_path(ROOT, inputs_path),
            "sha256": local_sha256_file(inputs_path),
        },
        "labels": {
            "path": relative_path(ROOT, labels_path),
            "sha256": local_sha256_file(labels_path),
        },
        "local_reference": {
            "path": relative_path(ROOT, local_reference_path),
            "sha256": local_sha256_file(local_reference_path),
        },
        "fp32_remote_output": {
            "path": relative_path(ROOT, fp32_remote_path),
            "sha256": local_sha256_file(fp32_remote_path),
        },
        "qdq_int8_remote_output": {
            "path": relative_path(ROOT, qdq_remote_path),
            "sha256": local_sha256_file(qdq_remote_path),
        },
        "models": {
            model_id: {
                "path": experiment["models"][model_id]["source_path"],
                "sha256": experiment["models"][model_id]["source_sha256"],
            }
            for model_id in S22_MODEL_ORDER
        },
    }
    report, predictions = build_s22_preflight_report(
        labels=labels_archive["labels"],
        original_indices=labels_archive["original_indices"],
        local_logits=local_logits,
        remote_logits=remote_logits,
        provenance=provenance,
        jobs=jobs,
        selection=preflight["selection"],
    )
    payload, _ = write_s22_accuracy_outputs(
        _output_dir(args.output_dir), report, predictions
    )
    return payload


def command_cifar10_full_export(args: argparse.Namespace) -> dict:
    import numpy as np
    import onnxruntime as ort

    manifest_path, experiment = _load_manifest_arg(args.manifest)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    data_root = repository_path(ROOT, args.data_root, must_exist=True, kind="data root")
    test_batch_path = data_root / "cifar-10-batches-py" / "test_batch"
    full_images, full_labels = load_cifar10_test_set(data_root)
    inputs, labels, original_indices = prepare_full_test_set(full_images, full_labels)
    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CPUExecutionProvider is unavailable")
    model_results, reference_arrays = _evaluate_export_models(
        experiment, inputs, labels, args.batch_size, ort
    )
    comparison = prediction_agreement(
        reference_arrays["fp32_predictions"], reference_arrays["qdq_int8_predictions"]
    )
    expected_correct = {"fp32": 9374, "qdq_int8": 9357}
    for model_id, expected in expected_correct.items():
        if model_results[model_id]["correct"] != expected:
            raise RuntimeError(
                f"{model_id} full-test correct count must reproduce {expected}, got "
                f"{model_results[model_id]['correct']}"
            )
    if comparison["agreement_count"] != 9839:
        raise RuntimeError(
            "FP32/QDQ full-test agreement must reproduce 9839/10000, got "
            f"{comparison['agreement_count']}/10000"
        )
    selection = full_selection_manifest(full_images, full_labels)
    validations = {
        "original_indices_are_exactly_0_through_9999": bool(
            np.array_equal(original_indices, np.arange(10000, dtype=np.int64))
        ),
        "input_and_label_order_matches_test_batch": bool(np.array_equal(labels, full_labels)),
        "all_inputs_are_finite": bool(np.all(np.isfinite(inputs))),
        "all_logits_are_finite": all(
            bool(np.all(np.isfinite(reference_arrays[f"{model_id}_logits"])))
            for model_id in PREFLIGHT_MODEL_ORDER
        ),
        "fp32_accuracy_reproduces_9374_of_10000": model_results["fp32"]["correct"]
        == 9374,
        "qdq_int8_accuracy_reproduces_9357_of_10000": model_results["qdq_int8"][
            "correct"
        ]
        == 9357,
        "fp32_qdq_agreement_reproduces_9839_of_10000": comparison["agreement_count"]
        == 9839,
    }
    manifest_base = {
        "schema_version": FULL_EXPORT_SCHEMA,
        "scope": "offline full CIFAR-10 data export and local ONNX Runtime CPU reference",
        "statements": {
            "ai_hub_connected": False,
            "ai_hub_tasks_created": False,
            "is_galaxy_s22_qnn_accuracy": False,
            "piecewise_reference_included": False,
        },
        "provenance": {
            "experiment_manifest_path": relative_path(ROOT, manifest_path),
            "experiment_manifest_sha256": local_sha256_file(manifest_path),
            "test_batch_path": relative_path(ROOT, test_batch_path),
            "test_batch_sha256": local_sha256_file(test_batch_path),
        },
        "dataset": {
            "name": "CIFAR-10",
            "split": "test",
            "sample_count": 10000,
            "ordered_dataset_fingerprint_sha256": ordered_dataset_fingerprint(
                full_images, full_labels
            ),
        },
        "selection": selection,
        "preprocessing": preprocessing_metadata(
            randomness="none; all official test_batch records are retained in original order"
        ),
        "runtime": {
            "numpy": np.__version__,
            "onnxruntime": ort.__version__,
            "requested_providers": ["CPUExecutionProvider"],
            "available_providers": ort.get_available_providers(),
            "batch_size": args.batch_size,
        },
        "models": model_results,
        "prediction_comparison": comparison,
        "validations": validations,
    }
    report, _ = write_preflight_outputs(
        _output_dir(args.output_dir),
        inputs=inputs,
        labels=labels,
        original_indices=original_indices,
        local_reference=reference_arrays,
        manifest_base=manifest_base,
        manifest_filename="full_manifest.json",
        summary_filename="full_summary.md",
        markdown_renderer=full_export_markdown,
    )
    return report


COMMANDS = {
    "compile": command_compile,
    "profile": command_profile,
    "inference": command_inference,
    "numerical-audit": command_numerical_audit,
    "profile-summary": command_profile_summary,
    "cifar10-accuracy": command_cifar10_accuracy,
    "cifar10-preflight": command_cifar10_preflight,
    "cifar10-s22-report": command_cifar10_s22_report,
    "cifar10-full-export": command_cifar10_full_export,
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        COMMANDS[args.command](args)
        if args.command == "cifar10-accuracy":
            print(f"Wrote JSON, Markdown, and predictions NPZ under {Path(args.output_dir).as_posix()}")
        elif args.command == "cifar10-preflight":
            print(f"Wrote preflight NPZ, JSON, and Markdown under {Path(args.output_dir).as_posix()}")
        elif args.command == "cifar10-s22-report":
            print(f"Wrote S22 accuracy JSON, Markdown, and predictions under {Path(args.output_dir).as_posix()}")
        elif args.command == "cifar10-full-export":
            print(f"Wrote full-test NPZ, JSON, and Markdown under {Path(args.output_dir).as_posix()}")
        else:
            print(f"Wrote JSON and Markdown under {Path(args.output_dir).as_posix()}")
        return 0
    except (ValueError, FileNotFoundError, QaiHubUnavailableError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # Do not echo SDK internals that may include account data.
        print(
            f"error: {type(error).__name__}: QNN command failed; credentials and SDK internals were not logged",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
