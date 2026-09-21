# ResNet18-SiLU Quantized Inference Benchmark

The standalone v1.0 portable C++17 quantized SiLU kernel, exact golden validation, and kernel-only microbenchmark are documented in [`docs/cpp_quantized_silu_kernel_v10.md`](docs/cpp_quantized_silu_kernel_v10.md). This is not a full C++ inference runtime or backend deployment claim.

The v1.1 controlled accuracy-recovery experiment is documented in [`docs/silu_accuracy_recovery_v11.md`](docs/silu_accuracy_recovery_v11.md). Its selected ORT functional reference reached 93.58% CIFAR-10 top-1 accuracy, versus 93.57% for the locked Standard-QDQ baseline; this is an accuracy result, not an accelerated-kernel or deployment result.

The v1.2 ORT C++ custom-op implementation and graph contract are documented in [`docs/ort_cpp_customop_v12.md`](docs/ort_cpp_customop_v12.md). Its Windows x64 DLL reuses the v1.0 scalar kernel, executes all 17 selected activations in ORT CPU, matches the v1.1 reference exactly on the frozen 128-image probe, and reaches 93.58% on the complete CIFAR-10 test set. This is an ORT CPU hybrid graph, not integer-only whole-model inference.

The v1.3 CPU performance diagnosis is documented in [`docs/ort_cpp_customop_performance_v13.md`](docs/ort_cpp_customop_performance_v13.md). It preserves v1.2 semantics, profiles both graph forms, retains 8,000 raw timings across a bounded repeated matrix, and recommends four ORT intra-op threads on this machine. It implements no performance optimization and makes no speedup or cross-machine claim.

The explicit v1.4 four-thread runtime preset and its validation evidence are documented in [`docs/ort_cpp_customop_thread_preset_v14.md`](docs/ort_cpp_customop_thread_preset_v14.md). It is an opt-in, machine-specific ORT CPU session configuration—not a global default—and preserves exact v1.2 outputs. Typical v1.3 latency was reproduced, with a recorded tail-variance caveat.

The v1.6 Qualcomm AI Hub QNN toolchain is documented in [`docs/qnn_aihub_backend_v16.md`](docs/qnn_aihub_backend_v16.md). It organizes the existing Galaxy S22 / Android 12 compile, profile, inference, and numerical-audit evidence behind a resumable CLI and an offline-tested backend. QDQ INT8 averages 0.40647 ms (2.027x faster than FP32); the piecewise reference averages 1.68766 ms with 374/374 NPU nodes, but is 104.82% slower than FP32. The 12-input synthetic audit is not a CIFAR-10 accuracy evaluation and does not prove internal UINT8 code equality.

The same v1.6 CLI also provides a fully offline `cifar10-accuracy` command for reproducible local ONNX Runtime CPU evaluation of the three frozen source models on the complete 10,000-image CIFAR-10 test set. These local results establish a baseline for later device work; they are not Galaxy S22 QNN accuracy and the command creates no AI Hub task.

For Galaxy S22 inference, `cifar10-preflight` deterministically exports an ignored, class-balanced 1,000-image normalized NPZ plus FP32/QDQ local ORT references. It uses the same verified CIFAR-10 loader and preprocessing as the full local baseline and performs no network operation.

`cifar10-s22-report` converts already-downloaded device outputs into a permanent accuracy/numerical comparison without contacting AI Hub, while `cifar10-full-export` prepares the corresponding complete 10,000-image FP32/QDQ input and local-reference package in the ignored `out/` tree. The offline `cifar10-s22-full-report` command freezes the [completed full-test result](results/benchmarks/v1.6_qnn_cifar10_s22_full_10000/full_accuracy_summary.md): FP32 reaches 93.73% on Galaxy S22 and standard QDQ INT8 reaches 93.68% (-0.05 percentage points) with 0.40647 ms mean latency, a 2.027x speedup over FP32. Standard QDQ INT8 is the recommended deployment; the lower-accuracy, slower piecewise graph remains diagnostic evidence only.

The v1.7 CUDA module is documented in [`cuda/README.md`](cuda/README.md). It provides standalone CUDA vector-add and reduction baselines, fused FP32 scalar/`float4` Bias+SiLU kernels, fused FP16 scalar/`half2` Bias+SiLU kernels, adaptive dispatch, CUDA Event benchmarks, deterministic result aggregation, and a framework-independent integration example.

Reviewed Tesla T4 / CUDA 12.4 stem results are summarized below. Values are median P50 latency from repeated interleaved benchmark runs.

| Precision | Mode | Scalar P50 | Vectorized P50 | Auto P50 | Auto path | Auto/scalar |
|---|---|---:|---:|---:|---|---:|
| FP32 | Out-of-place | 6.016 us | 5.440 us | 5.472 us | `float4` | 1.0994x |
| FP32 | In-place | 5.632 us | 5.024 us | 5.008 us | `float4` | 1.1246x |
| FP16 | Out-of-place | 6.144 us | 5.568 us | 5.536 us | `half2` | 1.1098x |
| FP16 | In-place | 5.952 us | 5.184 us | 5.280 us | `half2` | 1.1273x |

All 13 CUDA CTest cases passed on the Tesla T4. Compute Sanitizer reported zero memcheck and synccheck errors, and reduction racecheck reported zero hazards. The `65,536`-element adaptive-dispatch threshold is specific to the recorded Tesla T4 evidence and is not presented as a universal GPU threshold. See the [v1.7.0 release](https://github.com/yjagger-w/resnet18-silu-quantized-inference-benchmark/releases/tag/v1.7.0), [FP16 results](results/benchmarks/v1.7_cuda_bias_silu_fp16_t4/summary.md), and [FP32 results](results/benchmarks/v1.7_cuda_bias_silu_t4/aggregate_summary.md). This release does not claim complete framework integration or end-to-end ResNet acceleration.

A lightweight quantization and deployment benchmark for **ResNet18-SiLU on CIFAR-10**, covering PyTorch FP32 evaluation, ONNX export, ONNX Runtime validation, INT8 post-training quantization, CPU latency benchmarking, quantization matrix evaluation, automatic report generation, and custom **SiLU-aware PTQ** simulation.

This project is designed as a reproducible MVP for AI model quantization, model deployment, and inference performance evaluation.

## Highlights

- Verified PyTorch FP32 baseline accuracy on CIFAR-10.
- Exported ResNet18-SiLU from PyTorch to ONNX.
- Validated ONNX Runtime FP32 output against PyTorch FP32.
- Benchmarked ONNX Runtime FP32 and INT8 CPU inference.
- Applied ONNX Runtime static INT8 PTQ using QDQ format.
- Built a MinMax INT8 quantization matrix across weight granularity and calibration sample counts.
- Generated automatic Markdown/CSV benchmark reports.
- Integrated a custom PyTorch-side **SiLU-aware PTQ simulation** with:
  - KLD-based `Vmax` search
  - MSE-based `vsplit` selection
  - Piecewise asymmetric activation quantization
  - Per-channel weight quantization
  - Bias correction
  - Layer-wise error analysis

## Model and Dataset

| Item | Setting |
|---|---|
| Model | ResNet18-SiLU |
| Dataset | CIFAR-10 |
| Input shape | 3 × 32 × 32 |
| Classes | 10 |
| FP32 checkpoint | `checkpoints/resnet18_cifar10.pth` |
| FP32 ONNX model | `artifacts/onnx/resnet18_silu_fp32.onnx` |
| Standard INT8 ONNX model | `artifacts/int8/resnet18_silu_int8.onnx` |
| Backend | ONNX Runtime `CPUExecutionProvider` |
| Custom PTQ mode | PyTorch-side quantization simulation |

Large files such as datasets, checkpoints, ONNX models, generated INT8 models, and raw benchmark CSV/JSON files are intentionally excluded from Git.

## Results Summary

### FP32 Baseline

| Metric | Result |
|---|---:|
| PyTorch FP32 accuracy | 93.7400% |
| Correct / Total | 9374 / 10000 |
| FP32 ONNX Runtime accuracy | 93.7400% |
| Prediction agreement | 100.0000% |
| Maximum absolute difference | 0.00003624 |
| Mean absolute difference | 0.00000331 |
| FP32 ONNX model size | 42.62 MB |

The exported FP32 ONNX model is numerically consistent with the PyTorch checkpoint.

### Standard ONNX Runtime INT8 PTQ

The standard INT8 pipeline uses ONNX Runtime static quantization.

| Item | Setting |
|---|---|
| Quantization format | QDQ |
| Activation type | QUInt8 |
| Weight type | QInt8 |
| Weight granularity | Per-channel |
| Calibration method | MinMax |
| Calibration samples | 2560 |

| Metric | FP32 ONNX Runtime | INT8 ONNX Runtime | Change |
|---|---:|---:|---:|
| Accuracy | 93.7400% | 93.5700% | -0.1700 pp |
| Correct / Total | 9374 / 10000 | 9357 / 10000 | -17 samples |
| Model size | 42.62 MB | 10.80 MB | -74.7% |
| Prediction agreement vs PyTorch | 100.0000% | 98.3900% | -1.61 pp |

### MinMax INT8 Quantization Matrix

The MinMax matrix evaluates standard ONNX Runtime INT8 PTQ across:

| Dimension | Values |
|---|---|
| Weight granularity | `per-tensor`, `per-channel` |
| Calibration samples | 128, 512, 1024, 2560 |
| Calibration method | MinMax |

Best accuracy setting:

| Setting | Accuracy | Drop vs FP32 | Model size |
|---|---:|---:|---:|
| MinMax / per-channel / 1024 samples | 93.6500% | 0.0900 pp | 10.80 MB |

Smallest model setting:

| Setting | Accuracy | Drop vs FP32 | Model size |
|---|---:|---:|---:|
| MinMax / per-tensor / 1024 samples | 93.4300% | 0.3100 pp | 10.74 MB |

The matrix improved the best standard INT8 result compared with the original 2560-sample setting:

| Setting | Accuracy |
|---|---:|
| MinMax / per-channel / 2560 samples | 93.5700% |
| MinMax / per-channel / 1024 samples | 93.6500% |

### ONNX Runtime CPU Benchmark

The following benchmark was measured on a local Windows CPU environment using ONNX Runtime `CPUExecutionProvider`.

| Batch size | FP32 mean latency | INT8 mean latency | Speedup | FP32 throughput | INT8 throughput |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.4832 ms | 1.3693 ms | 4.73× | 154.25 samples/s | 730.28 samples/s |
| 4 | 41.8719 ms | 3.7490 ms | 11.17× | 95.53 samples/s | 1066.95 samples/s |
| 8 | 68.2313 ms | 30.8052 ms | 2.22× | 117.25 samples/s | 259.70 samples/s |

CPU benchmark results may vary across machines, thread settings, power modes, thermal throttling, and background workloads.

### Custom SiLU-aware PTQ Simulation

The custom SiLU-aware PTQ pipeline is implemented as a PyTorch-side quantization simulation. It is designed to evaluate algorithmic quantization behavior for SiLU-based CNNs.

| Method | Accuracy | Drop vs FP32 | Quantized layers | Thresholds | Bias corrections |
|---|---:|---:|---:|---:|---:|
| FP32 | 93.7400% | 0.0000 pp | — | — | — |
| NCNN-style PTQ simulation | 91.8800% | 1.8600 pp | 9 | 9 | 0 |
| SiLU-aware PTQ simulation | 92.9700% | 0.7700 pp | 9 | 9 | 1 |

Key observation:

| Comparison | Improvement |
|---|---:|
| SiLU-aware PTQ vs NCNN-style PTQ | +1.0900 pp |
| Accuracy drop reduction | 1.8600 pp → 0.7700 pp |

The SiLU-aware method improves over the NCNN-style activation quantization simulation, but the current PyTorch-side simulation is not directly equivalent to the ONNX Runtime QDQ INT8 graph. Therefore, standard ORT PTQ and SiLU-aware PTQ are reported as separate evaluation tracks.

The repository also contains a standalone ONNX standard-operator reference subgraph for the locked SiLU piecewise contract. It returns uint8 codes and float32 reconstructed values, but is not standard single-scale QDQ and does not claim INT8 acceleration. See `docs/quantization_spec.md` for its contract and validation command.

## SiLU Piecewise Full-Model Reference

The Phase 3 path exports the FP32 ResNet18-SiLU model, discovers only exact `Mul(x, Sigmoid(x))` SiLU patterns, binds a valid `PiecewiseQuantizationSpec` to every exported call site, rewrites each pattern with the verified reference subgraph, checks it, runs it on ONNX Runtime CPU, and writes a JSON capability report. In the current exporter there are 17 call sites: the nine configured SiLU modules include eight residual-block modules invoked twice.

```powershell
python scripts/export_onnx.py --checkpoint checkpoints/resnet18_cifar10.pth --output artifacts/onnx/resnet18_silu_fp32.onnx --opset 18
python scripts/rewrite_silu_piecewise_onnx.py --spec-manifest path\to\site_specs.json
python scripts/inspect_onnx_model.py --model artifacts/onnx/resnet18_silu_piecewise_reference.onnx --report results/silu_piecewise_onnx_report.json
```

The manifest must contain a `sites` list with exactly one `{site_id, vmin, vsplit, vmax, bits}` entry for every discovered site. It is an explicit calibration artifact; the historical threshold JSON is not a valid Phase 3 manifest because it predates the locked positive-`Vsplit` contract. Generated ONNX models under `artifacts/onnx/` and JSON reports under `results/` are intentionally untracked.

This is a functional ONNX Runtime reference: it retains float64 internal arithmetic to preserve reference rounding, remains separate from the standard QDQ baseline, and is not evidence of OpenVINO/QNN portability or end-to-end INT8 performance.

### Phase 3.1 calibrated manifest

`configs/calibration/resnet18_silu_piecewise_v06.json` is the versioned v0.6 calibration manifest. It records 17 runtime call sites (nine modules, with shared block activations represented by invocation ordinal), the real checkpoint digest, deterministic CIFAR-10 training indices, and the canonical positive-`Vsplit` parameters plus auditable derived fields. Generate it with the real local assets:

```powershell
python scripts/generate_silu_piecewise_manifest.py `
  --checkpoint checkpoints\resnet18_cifar10.pth `
  --data-root data `
  --batch-size 128 `
  --num-calibration-batches 20 `
  --seed 20260826 `
  --output configs\calibration\resnet18_silu_piecewise_v06.json
```

The legacy `results/silu_aware_thresholds.json` is retained unchanged as historical evidence, but is deliberately rejected as a v0.6 manifest: it was produced before the locked contract and stores negative `Vsplit` values. The real closure command uses a separately selected CIFAR-10 test image for numerical comparison only; it does not measure accuracy:

```powershell
python scripts/validate_silu_piecewise_full_model.py --manifest configs\calibration\resnet18_silu_piecewise_v06.json
```

### Phase 3.2 boundary diagnostic

Use `scripts/diagnose_silu_piecewise_mismatch.py` to create an ignored JSON report with per-call-site pre-SiLU, SiLU-expression, code, reconstruction, and rounding-margin diagnostics. The real fixed-input result localizes the first code difference to `layer1.0.act.call_1`: a `2.38e-7` PyTorch/ORT pre-activation difference crosses an upper-segment half-integer rounding boundary, producing codes 128 and 129. This is retained as boundary-sensitivity evidence, not hidden by a tolerance change. Future work must explicitly choose a new validation/reference policy (for example an ORT-native reference or a formally specified boundary-stability policy); the locked v0.6 contract is unchanged.

## Project Structure

```text
resnet18-silu-quantized-inference-benchmark/
├── src/
│   └── silu_benchmark/
│       ├── models/
│       │   └── resnet_silu.py
│       ├── quantization/
│       │   ├── activation.py
│       │   ├── thresholds.py
│       │   └── weights.py
│       ├── activation_collection.py
│       ├── calibration.py
│       ├── config.py
│       ├── constants.py
│       ├── data.py
│       └── evaluation.py
├── scripts/
│   ├── evaluate_fp32.py
│   ├── export_onnx.py
│   ├── validate_onnx.py
│   ├── benchmark_ort_fp32.py
│   ├── benchmark_ort.py
│   ├── quantize_int8_static.py
│   ├── run_quantization_matrix.py
│   ├── generate_report.py
│   └── run_silu_aware_ptq.py
├── checkpoints/
├── data/
├── artifacts/
│   ├── onnx/
│   └── int8/
├── results/
├── reports/
│   ├── benchmark_summary.md
│   └── silu_aware_ptq_summary.md
└── tests/
```

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

Recommended `requirements.txt`:

```text
torch
torchvision
numpy
pandas
onnx
onnxruntime
onnxscript
```

## Reproduction

### 1. Evaluate PyTorch FP32 Baseline

```bash
python scripts/evaluate_fp32.py --checkpoint checkpoints/resnet18_cifar10.pth --data-root data --batch-size 128 --device cpu
```

Expected result:

```text
Accuracy: 93.7400%
Correct: 9374
Total: 10000
```

### 2. Export FP32 Model to ONNX

```bash
python scripts/export_onnx.py --checkpoint checkpoints/resnet18_cifar10.pth --output artifacts/onnx/resnet18_silu_fp32.onnx --opset 18 --device cpu
```

Expected result:

```text
Export method: legacy torch.onnx.export
ONNX checker: passed
ONNX size: 42.62 MB
```

The legacy exporter is used because it cleanly preserves the dynamic batch dimension for this project.

### 3. Validate PyTorch vs ONNX Runtime FP32

```bash
python scripts/validate_onnx.py --checkpoint checkpoints/resnet18_cifar10.pth --onnx artifacts/onnx/resnet18_silu_fp32.onnx --data-root data --batch-size 128 --device cpu
```

Expected result:

```text
PyTorch accuracy: 93.7400%
ONNX Runtime accuracy: 93.7400%
Prediction agreement: 100.0000%
Maximum absolute difference: 0.00003624
Mean absolute difference: 0.00000331
```

### 4. Benchmark ONNX Runtime FP32

```bash
python scripts/benchmark_ort.py --onnx artifacts/onnx/resnet18_silu_fp32.onnx --precision FP32 --batch-sizes 1 4 8 --warmup 20 --runs 200 --output-csv results/ort_fp32_benchmark.csv --output-json results/ort_fp32_benchmark.json
```

### 5. Apply ONNX Runtime Static INT8 PTQ

```bash
python scripts/quantize_int8_static.py --model-input artifacts/onnx/resnet18_silu_fp32.onnx --model-output artifacts/int8/resnet18_silu_int8.onnx --data-root data --calibration-batch-size 128 --calibration-batches 20 --calibration-method MinMax
```

Expected result:

```text
INT8 quantization completed
FP32 model size: 42.62 MB
INT8 model size: 10.80 MB
```

### 6. Validate INT8 Accuracy

```bash
python scripts/validate_onnx.py --checkpoint checkpoints/resnet18_cifar10.pth --onnx artifacts/int8/resnet18_silu_int8.onnx --data-root data --batch-size 128 --device cpu
```

Expected result:

```text
PyTorch accuracy: 93.7400%
ONNX Runtime accuracy: 93.5700%
Prediction agreement: 98.3900%
Maximum absolute difference: 2.68642616
Mean absolute difference: 0.30417237
```

The larger numerical difference is expected because the ONNX model is quantized to INT8.

### 7. Benchmark ONNX Runtime INT8

```bash
python scripts/benchmark_ort.py --onnx artifacts/int8/resnet18_silu_int8.onnx --precision INT8 --batch-sizes 1 4 8 --warmup 20 --runs 200 --output-csv results/ort_int8_benchmark.csv --output-json results/ort_int8_benchmark.json
```

### 8. Run MinMax Quantization Matrix

```bash
python scripts/run_quantization_matrix.py --model-input artifacts/onnx/resnet18_silu_fp32.onnx --output-dir artifacts/int8/matrix --data-root data --calibration-methods MinMax --weight-granularities per-tensor per-channel --calibration-samples 128 512 1024 2560 --benchmark-batch-sizes 1 4 8 --benchmark-warmup 20 --benchmark-runs 200 --skip-existing --output-csv results/quantization_matrix_minmax.csv --output-json results/quantization_matrix_minmax.json
```

Expected best setting:

```text
Best accuracy setting:
MinMax / per-channel / 1024 samples / 93.6500%
```

### 9. Generate Automatic Benchmark Report

```bash
python scripts/generate_report.py --matrix-csv results/quantization_matrix_minmax.csv --fp32-benchmark-csv results/ort_fp32_benchmark.csv --int8-benchmark-csv results/ort_int8_benchmark.csv --output-md reports/benchmark_summary.md --output-csv results/benchmark_summary.csv
```

Expected result:

```text
Benchmark report generated
Best accuracy setting:
MinMax / per-channel / 1024 samples / 93.6500% / drop 0.0900 pp
```

### 10. Run Custom SiLU-aware PTQ Simulation

```bash
python scripts/run_silu_aware_ptq.py --checkpoint checkpoints/resnet18_cifar10.pth --data-root data --batch-size 128 --calibration-batches 20 --methods ncnn silu_aware --device cpu --layer-error-batches 5
```

Expected result:

```text
FP32 accuracy: 93.7400%

Running method: ncnn
Accuracy: 91.8800%
Accuracy drop: 1.8600 pp
Quantized layers: 9
Thresholds: 9

Running method: silu_aware
Accuracy: 92.9700%
Accuracy drop: 0.7700 pp
Quantized layers: 9
Thresholds: 9
Bias corrections: 1

Layer error rows: 9
```

Generated files:

```text
results/silu_aware_ptq.csv
results/silu_aware_thresholds.json
results/silu_aware_layer_error.csv
reports/silu_aware_ptq_summary.md
```

## Version Tags

| Tag | Description |
|---|---|
| `v0.1.0-fp32-baseline` | Verified PyTorch FP32 baseline |
| `v0.2.0-onnx-export` | Exported FP32 model to ONNX |
| `v0.3.0-onnx-validation` | Validated PyTorch and ONNX Runtime FP32 consistency |
| `v0.4.0-ort-fp32-benchmark` | Benchmarked ONNX Runtime FP32 CPU inference |
| `v0.5.0-ort-int8-ptq` | Applied static INT8 PTQ and benchmarked INT8 model |
| `v0.6.0-readme-results` | Documented FP32 and INT8 benchmark results |
| `v0.7.0-quantization-matrix` | Added MinMax INT8 quantization matrix |
| `v0.8.0-auto-report` | Added automatic benchmark report generation |
| `v0.9.0-silu-aware-ptq` | Integrated custom SiLU-aware PTQ simulation and layer-wise error analysis |

## Current Limitations

- Benchmark results are measured on a local Windows CPU environment and may vary on other machines.
- ONNX Runtime INT8 PTQ and PyTorch-side SiLU-aware PTQ simulation are separate evaluation tracks.
- The current SiLU-aware PTQ implementation is an algorithm simulation and is not yet exported as a standard ONNX QDQ INT8 graph.
- Entropy calibration was unstable in the current local ONNX Runtime environment, so the stable quantization matrix currently uses MinMax calibration.
- Calibration uses a fixed 2560-image CIFAR-10 subset for the main PTQ experiments.
- GPU and TensorRT benchmarking are not included in the current version.
- Quantization pre-processing and graph optimization can be further improved.

## Roadmap

- [x] PyTorch FP32 baseline evaluation
- [x] ONNX FP32 export
- [x] PyTorch vs ONNX Runtime FP32 validation
- [x] ONNX Runtime FP32 CPU benchmark
- [x] ONNX Runtime static INT8 PTQ
- [x] ONNX Runtime INT8 CPU benchmark
- [x] MinMax quantization experiment matrix
- [x] Automatic benchmark report generation
- [x] Custom SiLU-aware PTQ simulation
- [x] Layer-wise error analysis
- [ ] Stabilize Entropy and Percentile calibration
- [ ] Add fixed-thread CPU benchmark configuration
- [ ] Add graph pre-processing and quantization optimization
- [ ] Export custom SiLU-aware PTQ to a deployable ONNX-compatible representation
- [ ] Add TensorRT FP16 / INT8 benchmark on NVIDIA GPU
- [ ] Extend the benchmark framework to DistilBERT or CLIP image encoder
- [ ] Extend the benchmark framework to ST-Mamba traffic forecasting models

## Example Resume Bullets

Built an ONNX Runtime quantized inference benchmark for ResNet18-SiLU on CIFAR-10, covering PyTorch-to-ONNX export, FP32/INT8 validation, static INT8 PTQ, CPU latency profiling, quantization matrix evaluation, and automatic benchmark reporting.

Reduced model size from 42.62 MB to 10.80 MB with only 0.09 percentage-point accuracy drop under the best MinMax INT8 configuration, while achieving up to 11.17× CPU latency speedup in ONNX Runtime benchmarks.

Integrated a custom SiLU-aware PTQ simulation with KLD-based `Vmax` search, MSE-based `vsplit` selection, piecewise asymmetric activation quantization, per-channel weight quantization, bias correction, and layer-wise error analysis.

Improved ResNet18-SiLU INT8 simulation accuracy from 91.88% with NCNN-style activation quantization to 92.97% using SiLU-aware PTQ, reducing accuracy drop from 1.86 pp to 0.77 pp.

## License

## ORT-native calibration and closure

`configs/calibration/resnet18_silu_piecewise_v06_ort_cpu.json` is separate from the PyTorch-origin manifest. It records activation statistics collected from actual baseline ONNX `Sigmoid → Mul` SiLU outputs on `CPUExecutionProvider`, including model digest, provider/runtime versions, and the ordered 17-site identity.

```powershell
python scripts/generate_silu_piecewise_ort_manifest.py --output configs\calibration\resnet18_silu_piecewise_v06_ort_cpu.json
python scripts/validate_silu_piecewise_ort_native.py --manifest configs\calibration\resnet18_silu_piecewise_v06_ort_cpu.json
```

Validation applies the canonical Python Q/DQ reference to the exact ORT-produced pre-Q/DQ SiLU tensors and compares each result with the embedded ONNX subgraph. This is same-backend functional closure only; it does not remove the documented PyTorch-versus-ORT boundary sensitivity, prove portability, or claim INT8 acceleration.

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

