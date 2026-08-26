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
with one batch, and writes only to `results/benchmarks/v0.6.5_smoke`.

Run the official benchmark locally from the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path

python scripts/run_official_benchmark.py `
  --config configs\benchmarks\resnet18_silu_cifar10_v065_ort_cpu.json `
  --resume
```

Use `--force-rebuild` to rebuild benchmark artifacts even when their sidecar
fingerprint matches.  The fingerprint covers the frozen config, checkpoint,
FP32 ONNX model, calibration manifest, provider, and ONNX Runtime version.
`--resume` accepts an artifact only when the fingerprint sidecar matches;
missing, malformed, or stale sidecars force a rebuild.

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

## Outputs and failure handling

Only after every phase succeeds, the runner atomically promotes a staging
directory to the official output path.  A completed directory contains:

```text
results/benchmarks/v0.6.5/
  environment.json
  benchmark_results.json
  benchmark_results.csv
  benchmark_report.md
  graph_reports/
    fp32_ort_cpu.json
    standard_static_qdq_ort_cpu.json
    silu_piecewise_ort_reference_cpu.json
```

Failed or interrupted work remains in a sibling `.partial-<fingerprint>`
directory with machine-readable `run_status.json`; it is never promoted as an
official result.  The same layout is used for smoke output under the distinctly
named `v0.6.5_smoke` path.

The Standard-QDQ ONNX artifact is intentionally retained as the input baseline
for later OpenVINO/QNN work.  Those later backend tests are outside this ORT
CPU protocol and must report their own graph and kernel facts.
