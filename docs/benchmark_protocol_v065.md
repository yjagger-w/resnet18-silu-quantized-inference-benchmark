# v0.6.5 ORT CPU benchmark protocol

This protocol freezes a reproducible three-variant CIFAR-10 benchmark for the
ResNet18-SiLU checkpoint.  It measures the same 10,000 test images for every
official variant, using the committed checkpoint and the committed ORT-native
piecewise manifest.  Calibration remains fixed at 2,560 images (20 batches of
128) for the official Standard-QDQ build.

## Variants

`fp32_ort_cpu` is the unmodified FP32 ONNX baseline running on ONNX Runtime's
CPUExecutionProvider.  `standard_static_qdq_ort_cpu` is ONNX Runtime static
MinMax QDQ PTQ with QUInt8 activations, per-channel symmetric QInt8 weights,
and the frozen calibration protocol.  `silu_piecewise_ort_reference_cpu` is
the Phase 3 graph rewrite using the ORT-native 17-site SiLU calibration
manifest; it is a functional reference rather than an accelerator deployment
claim.

Standard static QDQ is the future OpenVINO/QNN deployment baseline.  The SiLU
piecewise graph is an ONNX Runtime functional reference and is not a fair
all-INT8 deployment comparison unless graph facts prove equal non-SiLU and
weight-quantization treatment.  The report records those facts rather than
assuming equivalence.

## Official and smoke runs

An official run evaluates all 10,000 CIFAR-10 test images and uses all 2,560
calibration images.  A smoke run is deliberately non-official: it evaluates a
deterministic first 128-image subset, calibrates the temporary QDQ artifact
with one batch, and writes only to `results/benchmarks/v0.6.5_smoke` (or a new
uniquely suffixed smoke directory if that destination already exists). Smoke
timing uses 2/5 latency warmup/timed runs, 1/3 throughput warmup/timed runs,
and semantic analysis on the first test image. The effective settings are
recorded separately from the frozen official config.

The complete benchmark worker, including QDQ calibration, uses NumPy, ONNX,
and ONNX Runtime; it does not import Torch or torchvision. The offline NumPy
CIFAR reader verifies the official batch checksums and performs the same
float32 divide-by-255, subtract-mean, divide-by-standard-deviation operations.
Official QDQ calibration retains the previous reader's first 2,560 training
images, now in deterministic sequential order rather than unseeded shuffled
batches. The committed piecewise calibration manifest is not regenerated or
modified; its own sample provenance remains distinct and is preserved.

Run the isolated smoke check first:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python scripts\run_official_benchmark.py --config configs\benchmarks\resnet18_silu_cifar10_v065_ort_cpu.json --smoke --force-rebuild
```

Run the official benchmark locally from the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path

python scripts/run_official_benchmark.py `
  --config configs\benchmarks\resnet18_silu_cifar10_v065_ort_cpu.json `
  --resume
```

Use `--force-rebuild` to rebuild benchmark artifacts even when their sidecar
fingerprint matches.  The fingerprint covers the frozen config, checkpoint,
FP32 ONNX model, calibration manifest, data files, runner/data/semantic source,
smoke mode, provider, and ONNX Runtime version. The ORT-only runner revision
invalidates sidecars made by the old mixed-runtime runner.
`--resume` accepts an artifact only when the fingerprint sidecar matches;
missing, malformed, stale sidecars or mismatched artifact SHA-256 force a
rebuild. Successful artifacts can be reused even when a previous attempt failed.

## Measurements

Accuracy uses the frozen CIFAR-10 transform and deterministic test ordering.
Latency uses batch size 1 after warmups; throughput uses the configured batch
size after separate warmups.  The runner reports mean, p50, p95, min, max, and
standard deviation latency.  Process memory is sampled during timed execution:
RSS via `psutil` when installed, otherwise Windows working set via `ctypes`.
It is explicitly process-level memory, not ORT tensor-allocator memory.

The semantic activation-site analysis uses debug-only ONNX copies.  It compares
FP32 with the piecewise reference at all 17 Phase 3 SiLU output identities and
compares compatible final logits for both quantized variants.  Standard-QDQ
intermediate sites are marked unavailable unless their mapping is proven; the
runner never guesses tensor correspondence.

Both the 17-site order and the custom dequantized tensor mapping are checked
against the committed manifest and Phase 3 rewrite metadata. Instrumented
models have exactly those 17 outputs followed by final logits. They live only
in the new run's `debug_models/` directory; they never replace the source ONNX
artifacts. Timing uses separate, uninstrumented sessions because exposing
graph outputs can change optimization even if only logits are requested.
Official semantic analysis uses the first evaluation batch. JSON and the
additional `semantic_activation_sites.csv` contain shapes, MSE, MAE, maximum
absolute error, cosine similarity, and mapping status/reason. All accuracy
results come from ORT CPU. No PyTorch/ORT cross-backend equality is claimed.

## Outputs and failure handling

Only after every phase succeeds, the runner atomically promotes a staging
directory to the official output path.  A completed directory contains:

```text
results/benchmarks/v0.6.5/
  environment.json
  benchmark_results.json
  benchmark_results.csv
  benchmark_report.md
  semantic_activation_sites.csv
  run_status.json
  worker.log
  worker_request.json
  debug_models/
  graph_reports/
    fp32_ort_cpu.json
    standard_static_qdq_ort_cpu.json
    silu_piecewise_ort_reference_cpu.json
```

Every attempt creates a fresh sibling `.partial-<unique-id>` directory; existing
results are never overwritten, deleted, or renamed. A supervisor launches the
ORT-only worker, relays its progress, records its log and waits for its exit.
Python exceptions and native crashes (including OpenMP exit code 3) become
terminal `status: "failed"` records containing the last phase, exit code,
diagnostic log tail and a next action. Worker success alone is insufficient:
the supervisor checks report readiness before promoting this new attempt.
Interrupted workers are terminated by the supervisor and marked failed.
An external hard kill of the supervisor itself cannot be handled by Python;
an abandoned partial directory is never evidence of completion.

## OpenMP failure diagnosis

On the audited Conda environment, importing Torch then calling NumPy's
`linalg.norm` reproduces native exit code 3 without a model. Torch loads
`site-packages/torch/lib/libiomp5md.dll`; NumPy's BLAS loads
`Library/bin/libiomp5md.dll`. NumPy alone and ORT + NumPy succeed.
The old runner's semantic cosine metric called `np.linalg.norm`/`np.dot`
after Torch had been imported directly and through the eager quantization
package. Moving additional ORT sessions did not remove that dependency.

ORT 1.19.2 also imports Torch indirectly through
`quantization -> shape_inference -> tools -> pytorch_export_helpers` when
`tools` discovers the optional Torch package. The benchmark disables only
that unused exporter feature probe during the quantizer import, then restores
`importlib.util.find_spec` immediately. Actual import/runtime errors still
propagate. The clean-process regression test rejects any real Torch import,
including transitive imports, throughout runner and calibration setup.

The canonical spec is relocated unchanged to a Torch-free module; the old
import paths remain compatible through re-exports/lazy imports. Torch-only
calibration hooks load Torch only when explicitly invoked. The benchmark
never invokes them. No duplicate-runtime override, DLL-path change, package
installation, or error suppression is used. An inherited duplicate-runtime
override is rejected with an actionable failure rather than accepted.

The Standard-QDQ ONNX artifact is intentionally retained as the input baseline
for later OpenVINO/QNN work.  Those later backend tests are outside this ORT
CPU protocol and must report their own graph and kernel facts.
