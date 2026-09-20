# Galaxy S22 QNN CIFAR-10 preflight accuracy

This report compares already-downloaded Galaxy S22 QNN outputs with local ONNX Runtime CPU references. Report generation did not connect to AI Hub or create a remote task.

The fixed preflight subset contains the first 100 samples of every class encountered in original `test_batch` order (1,000 images total). This is not full CIFAR-10 accuracy.

## Accuracy and local/S22 agreement

| Model | Compile job | Inference job | Local correct | Local Top-1 | S22 correct | S22 Top-1 | Change | Local/S22 agreement |
|---|---|---|---:|---:|---:|---:|---:|---:|
| FP32 | `jp1ndxwlg` | `jp0mdve2g` | 936 / 1000 | 93.6000% | 936 / 1000 | 93.6000% | +0.0000 pp | 100.0000% |
| QDQ INT8 | `j5qlw227p` | `jgolo4e4g` | 936 / 1000 | 93.6000% | 938 / 1000 | 93.8000% | +0.2000 pp | 98.9000% |

S22 FP32/QDQ prediction agreement: 982 / 1000 (98.2000%).

## Local versus S22 logit metrics

| Model | Mean absolute error | Max absolute error | RMSE | Mean cosine | Min cosine |
|---|---:|---:|---:|---:|---:|
| FP32 | 0.003750443 | 0.031660080 | 0.005086553 | 0.999999435 | 0.999992989 |
| QDQ INT8 | 0.217270350 | 1.698657036 | 0.299199981 | 0.997899171 | 0.965285544 |

## Prediction disagreements (original test_batch indices)

- FP32 local vs S22 (0): `(none)`
- QDQ INT8 local vs S22 (11): `52, 165, 275, 309, 313, 796, 799, 893, 925, 953, 1049`
- S22 FP32 vs QDQ INT8 (18): `37, 125, 147, 162, 165, 309, 412, 508, 551, 680, 734, 751, 793, 796, 893, 925, 956, 1049`

## Interpretation

The S22 QDQ result is +0.2 percentage points because it classifies 2 more samples correctly on this fixed 1,000-image subset. This does not demonstrate that quantization improves generalization accuracy.

`predictions.npz` contains labels, original indices, and local/S22 predictions only. It contains no image or logit arrays.
