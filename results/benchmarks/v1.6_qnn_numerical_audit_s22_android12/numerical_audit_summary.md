# QNN numerical audit - Galaxy S22 / Android 12

This audit compares local ONNX Runtime outputs with the corresponding QNN DLC outputs on a Galaxy S22 using 12 deterministic synthetic stress inputs. It is not a CIFAR-10 accuracy evaluation.

| Model | Mean abs. error | Max abs. error | RMSE | Mean cosine | Min cosine | Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| FP32 source | 0.004700019 | 0.024322033 | 0.006093357 | 0.999998808 | 0.999992255 | 1.0000 |
| QDQ INT8 | 0.373097847 | 3.882644296 | 0.674006158 | 0.978718048 | 0.836676481 | 1.0000 |
| Piecewise reference | 0.018818396 | 0.188420892 | 0.034407796 | 0.999853911 | 0.998941306 | 1.0000 |

## Interpretation

- All three compiled models preserve Top-1 predictions on all 12 stress inputs.
- FP32 has the smallest local-to-device numerical deviation.
- The piecewise reference remains substantially closer to its local reference than the standard QDQ INT8 model.
- The QDQ INT8 model has the best latency, but its larger logit drift requires evaluation on the real CIFAR-10 test set.
- This audit does not establish internal UINT8-code or rounding-boundary equality.

## AI Hub jobs

- FP32 source: [compile](https://workbench.aihub.qualcomm.com/jobs/jp1ndxwlg/), [profile](https://workbench.aihub.qualcomm.com/jobs/jgjr1d7vp/), [inference](https://workbench.aihub.qualcomm.com/jobs/jprl97wkp/)
- QDQ INT8: [compile](https://workbench.aihub.qualcomm.com/jobs/j5qlw227p/), [profile](https://workbench.aihub.qualcomm.com/jobs/jprl9yn9p/), [inference](https://workbench.aihub.qualcomm.com/jobs/j5m041w9g/)
- Piecewise reference: [compile](https://workbench.aihub.qualcomm.com/jobs/j568v3r7g/), [profile](https://workbench.aihub.qualcomm.com/jobs/jprl9dv0p/), [inference](https://workbench.aihub.qualcomm.com/jobs/jp8e8dn8p/)
