# SiLU-aware PTQ Summary

## Configuration

| Item | Value |
| --- | --- |
| Model | ResNet18-SiLU |
| Dataset | CIFAR-10 |
| Bits | 8 |
| Calibration samples | 2560 |
| Bias correction | enabled |
| Execution | PyTorch-side quantization simulation |

## Accuracy Results

| Method | Bits | Accuracy | Drop vs FP32 | Quantized layers | Thresholds | Bias corrections |
| --- | --- | --- | --- | --- | --- | --- |
| ncnn | 8 | 91.8800% | 1.8600 pp | 9 | 9 | 0 |
| silu_aware | 8 | 92.9700% | 0.7700 pp | 9 | 9 | 1 |

## Layer-wise Error Analysis

The table below reports the top layers by MSE between FP32 SiLU activations and quantized activations.

| Layer | MAE | MSE | RMSE | Max abs error |
| --- | --- | --- | --- | --- |
| layer4.1.act | 0.01667045 | 0.01368807 | 0.11699604 | 9.70385742 |
| layer4.0.act | 0.01294856 | 0.00657328 | 0.08107578 | 7.26910353 |
| layer1.1.act | 0.00815533 | 0.00205477 | 0.04532958 | 6.36315155 |
| layer1.0.act | 0.00512826 | 0.00146607 | 0.03828936 | 6.02688599 |
| layer3.1.act | 0.00760749 | 0.00116766 | 0.03417105 | 8.42497540 |
| layer2.0.act | 0.01170463 | 0.00093264 | 0.03053911 | 4.95789814 |
| layer2.1.act | 0.00886964 | 0.00088287 | 0.02971307 | 5.34544373 |
| layer3.0.act | 0.00902915 | 0.00080848 | 0.02843370 | 5.37824059 |
| act | 0.00124493 | 0.00033414 | 0.01827944 | 4.15013599 |

## Notes

- This v0.9 stage integrates the custom SiLU-aware PTQ algorithm into the benchmark repository.
- The current output is a PyTorch-side quantization simulation rather than a standard ONNX QDQ graph.
- The SiLU-aware method uses KLD-based Vmax search, MSE-based vsplit selection, piecewise asymmetric activation quantization, per-channel weight quantization, and optional bias correction.