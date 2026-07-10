# ResNet18-SiLU Quantized Inference Benchmark: Benchmark Summary

Generated at: 2026-07-10 15:20:18

## Overview

This report summarizes FP32 and INT8 ONNX Runtime benchmarking results for ResNet18-SiLU on CIFAR-10. It is generated automatically from local CSV outputs.

## FP32 Baseline

| Metric | Value |
| --- | --- |
| Accuracy | 93.7400% |
| Correct / Total | 9374 / 10000 |
| Model size | 42.62 MB |

### FP32 CPU Latency

| Batch | Mean latency | P50 latency | P95 latency | Throughput |
| --- | --- | --- | --- | --- |
| 1 | 6.4832 ms | 3.9934 ms | 17.2286 ms | 154.25 samples/s |
| 4 | 41.8719 ms | 32.1705 ms | 115.1472 ms | 95.53 samples/s |
| 8 | 68.2313 ms | 57.3472 ms | 135.6764 ms | 117.25 samples/s |

## Best INT8 Results from Quantization Matrix

| Selection | Setting | Accuracy | Accuracy drop | Model size |
| --- | --- | --- | --- | --- |
| Best accuracy | MinMax / per-channel / 1024 samples | 93.6500% | 0.0900 pp | 10.80 MB |
| Smallest model | MinMax / per-tensor / 1024 samples | 93.4300% | 0.3100 pp | 10.74 MB |

### Best Latency Speedup by Batch Size

| Batch | Setting | INT8 mean latency | Speedup vs FP32 |
| --- | --- | --- | --- |
| 1 | MinMax / per-tensor / 128 samples | 5.1128 ms | 1.27x |
| 4 | MinMax / per-tensor / 128 samples | 8.2045 ms | 5.10x |
| 8 | MinMax / per-tensor / 128 samples | 13.3760 ms | 5.10x |

## Quantization Matrix

| Method | Weight | Samples | Accuracy | Drop | Size | Size reduction | B1 latency | B1 speedup | B4 latency | B4 speedup | B8 latency | B8 speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MinMax | per-tensor | 128 | 93.3300% | 0.4100 pp | 10.74 MB | 74.81% | 5.1128 ms | 1.27x | 8.2045 ms | 5.10x | 13.3760 ms | 5.10x |
| MinMax | per-tensor | 512 | 93.5200% | 0.2200 pp | 10.74 MB | 74.81% | 6.8644 ms | 0.94x | 16.6818 ms | 2.51x | 23.0725 ms | 2.96x |
| MinMax | per-tensor | 1024 | 93.4300% | 0.3100 pp | 10.74 MB | 74.81% | 7.2369 ms | 0.90x | 21.6970 ms | 1.93x | 41.0753 ms | 1.66x |
| MinMax | per-tensor | 2560 | 93.5600% | 0.1800 pp | 10.74 MB | 74.81% | 7.2201 ms | 0.90x | 30.3084 ms | 1.38x | 28.4708 ms | 2.40x |
| MinMax | per-channel | 128 | 93.4600% | 0.2800 pp | 10.80 MB | 74.67% | 8.4590 ms | 0.77x | 23.5617 ms | 1.78x | 38.6415 ms | 1.77x |
| MinMax | per-channel | 512 | 93.4500% | 0.2900 pp | 10.80 MB | 74.67% | 9.3270 ms | 0.70x | 21.9527 ms | 1.91x | 37.4824 ms | 1.82x |
| MinMax | per-channel | 1024 | 93.6500% | 0.0900 pp | 10.80 MB | 74.67% | 10.0638 ms | 0.64x | 26.9603 ms | 1.55x | 16.3772 ms | 4.17x |
| MinMax | per-channel | 2560 | 93.5700% | 0.1700 pp | 10.80 MB | 74.67% | 10.3084 ms | 0.63x | 11.3220 ms | 3.70x | 28.1656 ms | 2.42x |

## Key Takeaways

- Best INT8 accuracy is **93.6500%**, with only **0.0900 percentage-point** accuracy drop from the FP32 baseline.
- The best-accuracy INT8 model size is **10.80 MB**, compared with **42.62 MB** for FP32.
- Best observed latency speedup is **5.10x** at batch size **4**, using **MinMax / per-tensor / 128 samples**.
- Entropy and Percentile calibration are not included in this summary because Entropy calibration was unstable in the current local ONNX Runtime environment. The stable v0.7 matrix uses MinMax calibration.
- A standalone INT8 benchmark CSV was detected, but the matrix result is the primary v0.8 summary.

## Source Files

| Item | Path |
| --- | --- |
| Quantization matrix CSV | `results\quantization_matrix_minmax.csv` |
| Generated summary CSV | `results\benchmark_summary.csv` |
