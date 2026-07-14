# ResNet18-SiLU Quantized Inference Benchmark

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

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

