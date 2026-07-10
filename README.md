# ResNet18-SiLU Quantized Inference Benchmark

A lightweight deployment benchmark for ResNet18-SiLU on CIFAR-10, covering the full path from a PyTorch FP32 checkpoint to ONNX export, ONNX Runtime validation, CPU latency benchmarking, and static INT8 post-training quantization.

This project is designed as a minimal, reproducible MVP for AI model quantization and deployment evaluation.

## Highlights

- Verified PyTorch FP32 baseline accuracy on CIFAR-10.
- Exported ResNet18-SiLU from PyTorch to ONNX.
- Validated ONNX Runtime FP32 output against PyTorch FP32.
- Benchmarked ONNX Runtime FP32 CPU inference latency and throughput.
- Applied ONNX Runtime static INT8 PTQ using QDQ format.
- Compared FP32 and INT8 models in accuracy, model size, latency, and throughput.

## Model and Dataset

| Item | Setting |
|---|---|
| Model | ResNet18-SiLU |
| Dataset | CIFAR-10 |
| Input shape | 3 × 32 × 32 |
| Classes | 10 |
| FP32 checkpoint | `checkpoints/resnet18_cifar10.pth` |
| FP32 ONNX model | `artifacts/onnx/resnet18_silu_fp32.onnx` |
| INT8 ONNX model | `artifacts/int8/resnet18_silu_int8.onnx` |
| Backend | ONNX Runtime CPUExecutionProvider |

Large files such as datasets, checkpoints, ONNX models, and benchmark outputs are intentionally excluded from Git.

## Results Summary

### Accuracy and Model Size

| Metric | FP32 ONNX Runtime | INT8 ONNX Runtime | Change |
|---|---:|---:|---:|
| Accuracy | 93.7400% | 93.5700% | -0.1700 percentage point |
| Correct / Total | 9374 / 10000 | 9357 / 10000 | -17 samples |
| Model size | 42.62 MB | 10.80 MB | -74.7% |
| Prediction agreement vs PyTorch | 100.0000% | 98.3900% | -1.61 percentage points |

### PyTorch vs ONNX Runtime FP32 Validation

| Metric | Result |
|---|---:|
| PyTorch accuracy | 93.7400% |
| ONNX Runtime accuracy | 93.7400% |
| Prediction agreement | 100.0000% |
| Maximum absolute difference | 0.00003624 |
| Mean absolute difference | 0.00000331 |

The FP32 ONNX model is numerically consistent with the PyTorch checkpoint.

### ONNX Runtime CPU Benchmark

The following benchmark was measured on a local Windows CPU environment using ONNX Runtime `CPUExecutionProvider`.

| Batch size | FP32 mean latency | INT8 mean latency | Speedup | FP32 throughput | INT8 throughput |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.4832 ms | 1.3693 ms | 4.73× | 154.25 samples/s | 730.28 samples/s |
| 4 | 41.8719 ms | 3.7490 ms | 11.17× | 95.53 samples/s | 1066.95 samples/s |
| 8 | 68.2313 ms | 30.8052 ms | 2.22× | 117.25 samples/s | 259.70 samples/s |

CPU benchmark results may vary across machines, thread settings, power modes, and background workloads.

## Project Structure

```text
resnet18-silu-quantized-inference-benchmark/
├── src/
│   └── silu_benchmark/
│       ├── models/
│       │   └── resnet_silu.py
│       ├── config.py
│       ├── data.py
│       └── evaluation.py
├── scripts/
│   ├── evaluate_fp32.py
│   ├── export_onnx.py
│   ├── validate_onnx.py
│   ├── benchmark_ort_fp32.py
│   ├── benchmark_ort.py
│   └── quantize_int8_static.py
├── checkpoints/
├── data/
├── artifacts/
│   ├── onnx/
│   └── int8/
├── results/
├── reports/
└── tests/
```

## Environment

Tested with:

```text
Python
PyTorch
TorchVision
ONNX
ONNX Runtime
NumPy
Pandas
```

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
python scripts/evaluate_fp32.py \
  --checkpoint checkpoints/resnet18_cifar10.pth \
  --data-root data \
  --batch-size 128 \
  --device cpu
```

Expected result:

```text
Accuracy: 93.7400%
Correct: 9374
Total: 10000
```

### 2. Export FP32 Model to ONNX

```bash
python scripts/export_onnx.py \
  --checkpoint checkpoints/resnet18_cifar10.pth \
  --output artifacts/onnx/resnet18_silu_fp32.onnx \
  --opset 18 \
  --device cpu
```

Expected result:

```text
Export method: legacy torch.onnx.export
ONNX checker: passed
ONNX size: 42.62 MB
```

The legacy exporter is used here because it preserves the dynamic batch dimension cleanly for this project.

### 3. Validate PyTorch vs ONNX Runtime FP32

```bash
python scripts/validate_onnx.py \
  --checkpoint checkpoints/resnet18_cifar10.pth \
  --onnx artifacts/onnx/resnet18_silu_fp32.onnx \
  --data-root data \
  --batch-size 128 \
  --device cpu
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
python scripts/benchmark_ort.py \
  --onnx artifacts/onnx/resnet18_silu_fp32.onnx \
  --precision FP32 \
  --batch-sizes 1 4 8 \
  --warmup 20 \
  --runs 200 \
  --output-csv results/ort_fp32_benchmark.csv \
  --output-json results/ort_fp32_benchmark.json
```

### 5. Apply ONNX Runtime Static INT8 PTQ

```bash
python scripts/quantize_int8_static.py \
  --model-input artifacts/onnx/resnet18_silu_fp32.onnx \
  --model-output artifacts/int8/resnet18_silu_int8.onnx \
  --data-root data \
  --calibration-batch-size 128 \
  --calibration-batches 20 \
  --calibration-method MinMax
```

Quantization configuration:

| Item | Setting |
|---|---|
| Quantization format | QDQ |
| Activation type | QUInt8 |
| Weight type | QInt8 |
| Weight granularity | Per-channel |
| Calibration method | MinMax |
| Calibration samples | 2560 |

Expected result:

```text
INT8 quantization completed
FP32 model size: 42.62 MB
INT8 model size: 10.80 MB
```

### 6. Validate INT8 Accuracy

```bash
python scripts/validate_onnx.py \
  --checkpoint checkpoints/resnet18_cifar10.pth \
  --onnx artifacts/int8/resnet18_silu_int8.onnx \
  --data-root data \
  --batch-size 128 \
  --device cpu
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
python scripts/benchmark_ort.py \
  --onnx artifacts/int8/resnet18_silu_int8.onnx \
  --precision INT8 \
  --batch-sizes 1 4 8 \
  --warmup 20 \
  --runs 200 \
  --output-csv results/ort_int8_benchmark.csv \
  --output-json results/ort_int8_benchmark.json
```

## Version Tags

| Tag | Description |
|---|---|
| `v0.1.0-fp32-baseline` | Verified PyTorch FP32 baseline |
| `v0.2.0-onnx-export` | Exported FP32 model to ONNX |
| `v0.3.0-onnx-validation` | Validated PyTorch and ONNX Runtime FP32 consistency |
| `v0.4.0-ort-fp32-benchmark` | Benchmarked ONNX Runtime FP32 CPU inference |
| `v0.5.0-ort-int8-ptq` | Applied static INT8 PTQ and benchmarked INT8 model |

## Current Limitations

- Benchmark results are measured on a local Windows CPU environment and may vary on other machines.
- The current INT8 pipeline uses standard ONNX Runtime static PTQ rather than custom SiLU-aware quantization.
- Calibration uses a fixed 2560-image CIFAR-10 subset.
- GPU and TensorRT benchmarking are not included in the current MVP.
- Quantization pre-processing and graph optimization can be further improved.

## Roadmap

- [x] PyTorch FP32 baseline evaluation
- [x] ONNX FP32 export
- [x] PyTorch vs ONNX Runtime FP32 validation
- [x] ONNX Runtime FP32 CPU benchmark
- [x] ONNX Runtime static INT8 PTQ
- [x] ONNX Runtime INT8 CPU benchmark
- [ ] Add automatic FP32 vs INT8 comparison report
- [ ] Add fixed-thread benchmark configuration for more stable CPU results
- [ ] Add quantization pre-processing and graph optimization
- [ ] Add TensorRT FP16 / INT8 benchmark on NVIDIA GPU
- [ ] Extend the benchmark framework to additional models or datasets

## Example Resume Bullet

Built an ONNX Runtime quantized inference benchmark for ResNet18-SiLU on CIFAR-10, covering PyTorch-to-ONNX export, FP32/INT8 validation, static INT8 post-training quantization, and CPU latency profiling. Reduced model size from 42.62 MB to 10.80 MB with only 0.17 percentage-point accuracy drop, achieving up to 11.17× CPU latency speedup.
