# v1.6 Qualcomm AI Hub QNN backend

## Scope

v1.6 turns the existing hand-run Qualcomm AI Hub experiments into a reusable backend and CLI. It does not resubmit the completed jobs, modify their original profile JSON, or replace the frozen numerical-audit artifacts. The backend supports new compile, profile, and inference submissions when explicitly requested, and it can recover every operation from an existing job ID.

The validated target is **Samsung Galaxy S22 5G / Android 12**. Compilation explicitly requests the `qnn_dlc` target. AI Hub is an optional online dependency; importing the project and running offline analysis does not import `qai_hub` or contact the network.

## Environment isolation

Keep QNN tooling separate from the original PyTorch/ORT baseline environment:

```powershell
python -m venv .venv-qnn
.\.venv-qnn\Scripts\Activate.ps1
python -m pip install -r requirements-qnn.txt
```

The validated QNN environment is:

| Package | Version |
|---|---:|
| Python | 3.11.15 |
| NumPy | 2.4.6 |
| ONNX | 1.19.1 |
| ONNX Runtime | 1.30.0 |
| qai-hub | 0.55.0 |

Configure AI Hub credentials using the vendor instructions and keep the SDK configuration outside this repository. Never copy a token or `client.ini` into the project. The backend records only a job ID, job URL, operation, status, target device, and the string form of `qai_hub.__version__`; it does not serialize client configuration or account information.

Generated DLCs, `client.ini`, `.qai_hub/`, `artifacts/qnn/`, and `artifacts/onnx/qnn_dlc/` are ignored. Use `out/qnn/` for generated reports and downloaded outputs.

## Frozen experiment manifest

[`configs/qnn/resnet18_silu_qnn_v16.json`](../configs/qnn/resnet18_silu_qnn_v16.json) is the single source of experiment identity:

| Model | Compile | Profile | Inference |
|---|---|---|---|
| FP32 source | `jp1ndxwlg` | `jgjr1d7vp` | `jprl97wkp` |
| QDQ INT8 | `j5qlw227p` | `jprl9yn9p` | `j5m041w9g` |
| Piecewise reference | `j568v3r7g` | `jprl9dv0p` | `jp8e8dn8p` |

The manifest uses repository-relative paths only. It links the three original profiles and the frozen `audit_inputs.npz`, `audit_outputs.npz`, JSON, and Markdown audit evidence. Those baseline result files remain unchanged.

## Offline reproduction

These commands do not load `qai_hub`, connect to AI Hub, or create a job:

```powershell
python scripts/run_qnn_aihub.py profile-summary `
  --output-dir out/qnn/v1.6/profile-summary

python scripts/run_qnn_aihub.py numerical-audit `
  --output-dir out/qnn/v1.6/numerical-audit
```

Each command writes JSON and Markdown. Profile analysis reads all 100 retained timing samples per model and computes mean, P50, P90, P95, P99, minimum, and maximum. It also reports inference peak memory, per-compute-unit node counts, NPU coverage, and every non-NPU node.

The full local CIFAR-10 accuracy baseline is also an offline command. It verifies the three source-model hashes from the manifest, verifies their `images`/`logits` float32 contracts, reads the official local `test_batch` in file order, and runs all 10,000 images through ONNX Runtime CPU:

```powershell
python scripts/run_qnn_aihub.py cifar10-accuracy `
  --data-root data `
  --batch-size 128 `
  --output-dir results/benchmarks/v1.6_qnn_cifar10_local_accuracy
```

This writes `local_accuracy.json`, `local_accuracy_summary.md`, and the compact `predictions.npz`. The archive contains labels, predictions, error indices, and pairwise disagreement indices only; it does not contain CIFAR-10 images. The command does not import the AI Hub SDK, contact AI Hub, or create a remote task. Its results are local ORT CPU accuracy, not Galaxy S22 QNN accuracy. The frozen `piecewise_v065` graph is always evaluated as-is even if its accuracy is lower.

The recorded local baseline in [`results/benchmarks/v1.6_qnn_cifar10_local_accuracy`](../results/benchmarks/v1.6_qnn_cifar10_local_accuracy/local_accuracy_summary.md) used ONNX Runtime 1.30.0 and `CPUExecutionProvider`:

| Model | Correct / total | Top-1 | Change vs FP32 | Agreement vs FP32 |
|---|---:|---:|---:|---:|
| FP32 | 9374 / 10000 | 93.7400% | +0.0000 pp | 100.0000% |
| QDQ INT8 | 9357 / 10000 | 93.5700% | -0.1700 pp | 98.3900% |
| Piecewise reference | 8280 / 10000 | 82.8000% | -10.9400 pp | 84.7700% |

All three runs completed with zero inference-failed samples and zero non-finite logits. The JSON and NPZ retain the complete zero-based error and disagreement index lists.

### Galaxy S22 CIFAR-10 preflight export

Prepare a deterministic balanced 1,000-image input package before any device inference:

```powershell
python scripts/run_qnn_aihub.py cifar10-preflight `
  --data-root data `
  --batch-size 128 `
  --output-dir out/qnn/v1.6/cifar10-s22-preflight-1000
```

The offline command scans the official test batch in original order, retaining a record while its class has fewer than 100 selected samples. The resulting indices remain strictly increasing, every class has exactly 100 samples, and the same shared `qnn_local_accuracy` loader and preprocessing implementation is used. It evaluates only FP32 and QDQ INT8 with local ONNX Runtime CPU; the piecewise graph is excluded.

The ignored output directory contains:

- `inputs.npz`: `images` float32 `[1000, 3, 32, 32]`, directly accepted by the existing `inference --inputs` interface;
- `labels.npz`: int64 `labels` and `original_indices`;
- `local_reference.npz`: FP32/QDQ predictions and logits;
- `preflight_manifest.json`: selection provenance, hashes, model/job identities, runtime, metrics, array contracts, and validation gates;
- `preflight_summary.md`: a human-readable local preflight summary.

The NPZ writer fixes archive metadata and records canonical dtype/shape/content hashes, so repeated exports are byte-stable as well as array-stable. No raw uint8 CIFAR-10 image, credential, or remote output is written. This command does not import the AI Hub SDK, connect to AI Hub, or create a task.

The confirmed comparison is:

| Model | Mean latency | Inference peak memory | NPU coverage | Comparison with FP32 |
|---|---:|---:|---:|---:|
| FP32 | 0.82398 ms | 156.77 MiB | 68/68 | baseline |
| QDQ INT8 | 0.40647 ms | 125.29 MiB | 70/70 | 2.027x faster |
| Piecewise reference | 1.68766 ms | 173.98 MiB | 374/374 | 104.82% slower |

## Resume completed jobs

Use `--job-id` to recover an existing job. This path calls `get_job` and never uploads the model or creates a replacement task:

```powershell
python scripts/run_qnn_aihub.py compile `
  --job-id jp1ndxwlg `
  --output-dir out/qnn/v1.6/fp32-compile

python scripts/run_qnn_aihub.py profile `
  --job-id jgjr1d7vp `
  --output-dir out/qnn/v1.6/fp32-profile

python scripts/run_qnn_aihub.py inference `
  --job-id jprl97wkp `
  --output-dir out/qnn/v1.6/fp32-inference
```

Profile recovery downloads `profile.json`; inference recovery downloads `outputs.npz`. Job metadata is written as `job.json` and `job.md`. The CLI never prints raw SDK configuration or exception details that could expose account data.

## Submit new work

The following examples create real AI Hub jobs and require configured credentials. Do not use them to repeat the frozen jobs above.

```powershell
python scripts/run_qnn_aihub.py compile `
  --model artifacts/onnx/resnet18_silu_fp32.onnx `
  --device "Samsung Galaxy S22 5G" `
  --os 12 `
  --input-spec images=1,3,32,32:float32 `
  --options="--target_runtime qnn_dlc" `
  --output-dir out/qnn/v1.6/new-compile
```

Use the returned compile job ID for downstream tasks:

```powershell
python scripts/run_qnn_aihub.py profile `
  --compile-job-id COMPILE_JOB_ID `
  --device "Samsung Galaxy S22 5G" `
  --os 12 `
  --output-dir out/qnn/v1.6/new-profile

python scripts/run_qnn_aihub.py inference `
  --compile-job-id COMPILE_JOB_ID `
  --inputs results/benchmarks/v1.6_qnn_numerical_audit_s22_android12/audit_inputs.npz `
  --device "Samsung Galaxy S22 5G" `
  --os 12 `
  --output-dir out/qnn/v1.6/new-inference
```

All CLI path arguments must be repository-relative and may not escape the repository. A missing optional SDK produces an actionable error referring to `requirements-qnn.txt`.

## Numerical-audit contract

The input generator is frozen at seed `20260919` and exactly reproduces the committed 12 samples:

- four structured probes: all-zero, all-negative-one, all-positive-one, and a `[-2.5, 2.5]` spatial ramp;
- eight seeded normal random probes clipped to `[-3, 3]`.

For each model, the audit calculates maximum and mean absolute error, RMSE, mean and minimum per-sample cosine similarity, and Top-1 agreement between local ONNX Runtime logits and device QNN logits. It also retains local-to-FP32 comparisons for QDQ INT8 and the piecewise reference.

This is a deterministic compiler-semantics stress audit, **not** CIFAR-10 accuracy evaluation. Top-1 agreement over 12 synthetic inputs does not demonstrate dataset accuracy, and close reconstructed logits do not prove equality of internal UINT8 codes or rounding-boundary behavior.

## Offline test boundary

Tests inject a fake SDK/client and cover submission arguments, job recovery, parsers, manifest validation, optional-dependency behavior, profile statistics, NPU/non-NPU accounting, frozen input reproduction, numerical metrics, and NPZ conversion. No test imports credentials, contacts AI Hub, or creates a real task.
