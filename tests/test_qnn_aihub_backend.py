import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from silu_benchmark.backends import qnn_aihub_backend as qnn


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/qnn/resnet18_silu_qnn_v16.json"
AUDIT_ROOT = ROOT / "results/benchmarks/v1.6_qnn_numerical_audit_s22_android12"


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "qnn_aihub_runner", ROOT / "scripts/run_qnn_aihub.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VersionObject:
    def __str__(self):
        return "0.55.0"


class FakeDevice:
    def __init__(self, name="", os=""):
        self.name = name
        self.os = os


class FakeStatus:
    code = "SUCCESS"
    success = True


class FakeJob:
    def __init__(self, job_id, kind):
        self.job_id = job_id
        self.kind = kind
        self.url = f"https://workbench.aihub.qualcomm.com/jobs/{job_id}/"
        self.wait_timeouts = []

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        return FakeStatus()

    def get_target_model(self):
        if self.kind != "compile":
            raise AssertionError("not a compile job")
        return "fake-target-model"

    def download_profile(self):
        if self.kind != "profile":
            raise AssertionError("not a profile job")
        return {
            "execution_summary": {
                "all_inference_times": [400, 500],
                "estimated_inference_peak_memory": 1024,
            },
            "execution_detail": [
                {"name": "conv", "compute_unit": "NPU"},
                {"name": "fallback", "compute_unit": "CPU"},
            ],
        }

    def download_output_data(self):
        if self.kind != "inference":
            raise AssertionError("not an inference job")
        return {"output_0": [np.zeros((1, 10), dtype=np.float32)]}


class FakeClient:
    def __init__(self):
        self.jobs = {
            "jc0mpile": FakeJob("jc0mpile", "compile"),
            "jpr0file": FakeJob("jpr0file", "profile"),
            "j1nference": FakeJob("j1nference", "inference"),
        }
        self.calls = []

    def get_job(self, job_id):
        self.calls.append(("get_job", job_id))
        return self.jobs[job_id]

    def submit_compile_job(self, **kwargs):
        self.calls.append(("submit_compile_job", kwargs))
        return FakeJob("jnewcompile", "compile")

    def submit_profile_job(self, **kwargs):
        self.calls.append(("submit_profile_job", kwargs))
        return FakeJob("jnewprofile", "profile")

    def submit_inference_job(self, **kwargs):
        self.calls.append(("submit_inference_job", kwargs))
        return FakeJob("jnewinference", "inference")


FAKE_SDK = SimpleNamespace(__version__=VersionObject(), Device=FakeDevice)


class QnnAiHubBackendTests(unittest.TestCase):
    def test_manifest_contains_frozen_jobs_and_environment(self):
        manifest = qnn.load_manifest(MANIFEST)
        self.assertEqual(manifest["seed"], 20260919)
        self.assertEqual(manifest["device"], {"name": "Samsung Galaxy S22 5G", "os": "12"})
        self.assertEqual(
            [model["jobs"] for model in manifest["models"].values()],
            [
                {"compile": "jp1ndxwlg", "profile": "jgjr1d7vp", "inference": "jprl97wkp"},
                {"compile": "j5qlw227p", "profile": "jprl9yn9p", "inference": "j5m041w9g"},
                {"compile": "j568v3r7g", "profile": "jprl9dv0p", "inference": "jp8e8dn8p"},
            ],
        )
        self.assertEqual(manifest["environment"]["qai_hub"], "0.55.0")
        changed = json.loads(MANIFEST.read_text(encoding="utf-8"))
        changed["seed"] = 1
        with self.assertRaisesRegex(ValueError, "20260919"):
            qnn.validate_manifest(changed)

    def test_input_spec_parser_and_repository_paths(self):
        self.assertEqual(
            qnn.parse_input_specs(["images=1,3,32,32:float32"]),
            {"images": ((1, 3, 32, 32), "float32")},
        )
        for invalid in (
            "images=1,3,32,0:float32",
            "images=1,3,32,32:float64",
            "images:1,3,32,32",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                qnn.parse_input_specs([invalid])
        self.assertEqual(qnn.repository_path(ROOT, "configs/qnn").name, "qnn")
        with self.assertRaisesRegex(ValueError, "relative"):
            qnn.repository_path(ROOT, ROOT / "configs")
        with self.assertRaisesRegex(ValueError, "inside"):
            qnn.repository_path(ROOT, "../outside")

    def test_optional_dependency_error_is_actionable_and_import_is_lazy(self):
        with mock.patch.object(
            qnn.importlib, "import_module", side_effect=ModuleNotFoundError("No module", name="qai_hub")
        ):
            with self.assertRaisesRegex(qnn.QaiHubUnavailableError, "requirements-qnn.txt") as caught:
                qnn.require_qai_hub()
        self.assertIn("Never commit token values or client.ini", str(caught.exception))

        # The already imported module proves qai_hub is not a module-level dependency.
        self.assertNotIn("qai_hub", qnn.__dict__)

    def test_fake_client_submit_and_resume_never_contacts_network(self):
        client = FakeClient()
        backend = qnn.QnnAiHubBackend(sdk=FAKE_SDK, client=client)
        compile_job = backend.compile(
            model=Path("model.onnx"),
            input_specs={"images": ((1, 3, 32, 32), "float32")},
            options="--target_runtime qnn_dlc",
        )
        self.assertEqual(compile_job.job_id, "jnewcompile")
        submit = client.calls[-1]
        self.assertEqual(submit[0], "submit_compile_job")
        self.assertEqual((submit[1]["device"].name, submit[1]["device"].os), (qnn.DEFAULT_DEVICE, "12"))

        resumed = backend.compile(job_id="jc0mpile")
        self.assertIs(resumed, client.jobs["jc0mpile"])
        self.assertEqual(client.calls[-1], ("get_job", "jc0mpile"))
        self.assertFalse(any(call[0] == "submit_compile_job" for call in client.calls[-1:]))

        profile = backend.profile(compile_job_id="jc0mpile")
        self.assertEqual(profile.job_id, "jnewprofile")
        self.assertEqual(client.calls[-1][1]["model"], "fake-target-model")
        inference = backend.inference(
            compile_job_id="jc0mpile", inputs={"images": [np.zeros((1, 3, 32, 32))]}
        )
        self.assertEqual(inference.job_id, "jnewinference")
        self.assertEqual(client.calls[-1][1]["model"], "fake-target-model")

    def test_job_record_is_allow_listed_and_version_is_string(self):
        job = FakeJob("jc0mpile", "compile")
        record = qnn.job_record(
            job,
            operation="compile",
            resumed=True,
            sdk=FAKE_SDK,
            device_name=qnn.DEFAULT_DEVICE,
            os_version="12",
            status=FakeStatus(),
        )
        self.assertEqual(record["qai_hub_version"], "0.55.0")
        self.assertIsInstance(record["qai_hub_version"], str)
        serialized = json.dumps(record).lower()
        self.assertNotIn("token", serialized)
        self.assertNotIn("client.ini", serialized)
        self.assertEqual(qnn.wait_for_success(job, 30).code, "SUCCESS")
        self.assertEqual(job.wait_timeouts, [30])

    def test_profile_parser_exact_statistics_and_compute_units(self):
        paths = {
            "fp32": ROOT / "results/benchmarks/v1.6_qnn_fp32_s22_android12/profile.json",
            "qdq_int8": ROOT / "results/benchmarks/v1.6_qnn_int8_s22_android12/profile.json",
            "piecewise_reference": ROOT
            / "results/benchmarks/v1.6_qnn_piecewise_reference_s22_android12/profile.json",
        }
        profiles = {
            key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()
        }
        summary = qnn.build_profile_summary(profiles)
        self.assertAlmostEqual(summary["models"]["fp32"]["latency_ms"]["mean"], 0.82398)
        self.assertAlmostEqual(summary["models"]["qdq_int8"]["latency_ms"]["mean"], 0.40647)
        self.assertAlmostEqual(
            summary["comparisons"]["qdq_int8_vs_fp32"]["speedup"], 2.02716068
        )
        piecewise = summary["models"]["piecewise_reference"]
        self.assertAlmostEqual(piecewise["latency_ms"]["mean"], 1.68766)
        self.assertEqual((piecewise["nodes"]["npu"], piecewise["nodes"]["total"]), (374, 374))
        self.assertEqual(piecewise["nodes"]["non_npu_count"], 0)
        self.assertAlmostEqual(
            summary["comparisons"]["piecewise_reference_vs_fp32"]["mean_latency_change_percent"],
            104.81808,
            places=5,
        )

        synthetic = qnn.summarize_profile(FakeJob("jpr0file", "profile").download_profile())
        self.assertEqual(synthetic["nodes"]["compute_units"], {"NPU": 1, "CPU": 1})
        self.assertEqual(synthetic["nodes"]["non_npu_nodes"][0]["name"], "fallback")

    def test_frozen_stress_inputs_and_numerical_metrics_reproduce_baseline(self):
        with np.load(AUDIT_ROOT / "audit_inputs.npz", allow_pickle=False) as archive:
            existing_inputs = archive["images"]
        self.assertTrue(np.array_equal(qnn.generate_audit_inputs(), existing_inputs))
        self.assertEqual(existing_inputs.shape, (12, 3, 32, 32))

        with np.load(AUDIT_ROOT / "audit_outputs.npz", allow_pickle=False) as archive:
            outputs = {key: archive[key] for key in archive.files}
        manifest = qnn.load_manifest(MANIFEST)
        audit = qnn.build_numerical_audit(
            manifest,
            existing_inputs,
            outputs,
            input_sha256=qnn.sha256_file(AUDIT_ROOT / "audit_inputs.npz").upper(),
            outputs_sha256=qnn.sha256_file(AUDIT_ROOT / "audit_outputs.npz").upper(),
            qai_hub_version=VersionObject(),
        )
        baseline = json.loads((AUDIT_ROOT / "numerical_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["environment"]["qai_hub"], "0.55.0")
        self.assertIsInstance(audit["environment"]["qai_hub"], str)
        for model_id in manifest["models"]:
            self.assertEqual(
                audit["models"][model_id]["local_vs_remote"],
                baseline["models"][model_id]["local_vs_remote"],
            )
        self.assertFalse(audit["input_set"]["is_accuracy_dataset"])
        self.assertIn("UINT8", " ".join(audit["limitations"]))

    def test_metrics_cover_errors_cosine_and_top1(self):
        reference = np.array([[1.0, 0.0], [0.0, 2.0]])
        candidate = np.array([[1.0, 0.0], [3.0, 0.0]])
        metrics = qnn.numerical_metrics(reference, candidate)
        self.assertEqual(metrics["max_abs_error"], 3.0)
        self.assertEqual(metrics["mean_abs_error"], 1.25)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(13 / 4))
        self.assertEqual(metrics["mean_cosine_similarity"], 0.5)
        self.assertEqual(metrics["min_cosine_similarity"], 0.0)
        self.assertEqual(metrics["top1_agreement"], 0.5)

    def test_cli_has_all_commands_and_defaults_to_qnn_dlc(self):
        runner = load_runner()
        compile_args = runner.parse_args(["compile", "--job-id", "jc0mpile"])
        self.assertEqual(compile_args.options, "--target_runtime qnn_dlc")
        self.assertEqual((compile_args.device, compile_args.os_version), (qnn.DEFAULT_DEVICE, "12"))
        for command in (
            "profile",
            "inference",
            "numerical-audit",
            "profile-summary",
            "cifar10-accuracy",
            "cifar10-preflight",
        ):
            args = runner.parse_args([command, "--job-id", "jtest"] if command in {"profile", "inference"} else [command])
            self.assertEqual(args.command, command)

    def test_cli_input_and_output_npz_helpers_are_offline(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "inputs.npz"
            np.savez_compressed(source, images=np.zeros((2, 3, 32, 32), dtype=np.float32))
            loaded = runner.load_inference_inputs(source)
            self.assertEqual(len(loaded["images"]), 2)
            self.assertEqual(loaded["images"][0].shape, (1, 3, 32, 32))
            destination = root / "outputs.npz"
            metadata = runner.save_inference_outputs(
                destination,
                {"output_0": [np.zeros((1, 10)), np.ones((1, 10))]},
            )
            self.assertEqual(metadata["output_0"]["shape"], [2, 10])
            with np.load(destination, allow_pickle=False) as archive:
                self.assertEqual(archive["output_0"].shape, (2, 10))


if __name__ == "__main__":
    unittest.main()
