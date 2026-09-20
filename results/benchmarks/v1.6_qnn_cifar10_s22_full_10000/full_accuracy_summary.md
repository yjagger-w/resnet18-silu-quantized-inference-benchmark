# Galaxy S22 QNN CIFAR-10 full-test accuracy and performance

This final v1.6 report compares already-downloaded Galaxy S22 / Android 12 QNN outputs with local ONNX Runtime CPU references over all 10,000 images in official CIFAR-10 `test_batch` order. Report generation did not connect to AI Hub or create a remote task.

## Accuracy and local/S22 agreement

| Model | Compile | Profile | Full inference | Local correct | Local Top-1 | S22 correct | S22 Top-1 | Local/S22 change | Local/S22 agreement |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| FP32 | `jp1ndxwlg` | `jgjr1d7vp` | `jpyoq7nr5` | 9374 / 10000 | 93.74% | 9373 / 10000 | 93.73% | -0.01 pp | 99.98% |
| QDQ INT8 | `j5qlw227p` | `jprl9yn9p` | `jpe78l275` | 9357 / 10000 | 93.57% | 9368 / 10000 | 93.68% | +0.11 pp | 98.80% |

## Galaxy S22 FP32 versus QDQ INT8

- Prediction agreement: 9845 / 10000 (98.45%); disagreements: 155.
- QDQ INT8 Top-1 change versus FP32: -0.05 percentage points.
- Both correct: 9309; FP32 only correct: 64; QDQ only correct: 59; both wrong: 568.

## Local versus S22 numerical comparison

| Model | Changed predictions | Mean abs error | Max abs error | RMSE | Mean cosine | Min cosine |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 2 | 0.003795087837080427 | 0.03838348388671875 | 0.005124231574212088 | 0.9999994564592338 | 0.9999895330541682 |
| QDQ INT8 | 120 | 0.2151494556091726 | 2.426652908325195 | 0.2944515844983891 | 0.998074252089587 | 0.9553450473539299 |

## Prediction disagreements (original test_batch indices)

- FP32 local vs S22 (2): `5565, 8861`
- QDQ INT8 local vs S22 (120): `52, 165, 275, 309, 313, 796, 799, 893, 925, 953, 994, 1049, 1181, 1217, 1391, 1552, 1683, 1845, 1924, 2010, 2032, 2046, 2159, 2248, 2331, 2581, 2650, 2705, 2760, 2779, 2844, 2845, 2884, 2905, 3041, 3107, 3158, 3180, 3208, 3211, 3235, 3297, 3336, 3400, 3422, 3514, 3622, 3636, 3708, 3887, 3995, 4000, 4002, 4208, 4223, 4309, 4355, 4476, 4630, 4696, 4718, 4740, 4903, 4985, 4986, 4995, 5006, 5098, 5155, 5162, 5213, 5271, 5354, 5369, 5690, 5718, 5830, 6035, 6180, 6393, 6422, 6434, 6438, 6535, 6574, 6646, 6656, 6825, 6900, 7002, 7082, 7203, 7497, 7756, 7813, 8048, 8276, 8281, 8396, 8480, 8491, 8521, 8642, 8720, 8898, 8906, 8976, 8983, 9237, 9375, 9434, 9518, 9587, 9633, 9665, 9741, 9753, 9787, 9794, 9839`
- S22 FP32 vs QDQ INT8 (155): `37, 125, 147, 162, 165, 309, 412, 508, 551, 680, 734, 751, 793, 796, 893, 925, 956, 994, 1049, 1181, 1280, 1470, 1561, 1683, 1714, 1862, 1924, 1939, 1944, 2005, 2032, 2059, 2061, 2159, 2331, 2511, 2650, 2690, 2705, 2760, 2770, 2779, 2845, 2854, 2905, 2988, 3002, 3041, 3059, 3107, 3180, 3336, 3497, 3646, 3704, 3962, 3995, 4012, 4208, 4251, 4266, 4286, 4352, 4355, 4404, 4532, 4546, 4555, 4630, 4676, 4718, 4740, 4776, 4779, 4985, 4986, 4990, 4995, 5093, 5176, 5213, 5354, 5424, 5441, 5558, 5560, 5565, 5603, 5690, 5718, 5826, 5830, 5835, 5856, 5882, 6008, 6125, 6135, 6197, 6218, 6305, 6419, 6422, 6535, 6569, 6574, 6594, 6631, 6646, 6750, 6753, 6825, 6862, 6985, 7174, 7265, 7375, 7399, 7497, 7524, 7625, 7756, 7909, 8022, 8177, 8199, 8276, 8364, 8480, 8484, 8507, 8521, 8529, 8546, 8642, 8720, 8818, 8861, 8898, 8976, 9039, 9185, 9237, 9255, 9302, 9375, 9431, 9434, 9633, 9741, 9753, 9764, 9786, 9794, 9880`

## Performance

| Model | Mean latency | Speedup vs FP32 | Peak memory | Memory reduction | NPU nodes |
|---|---:|---:|---:|---:|---:|
| FP32 | 0.82398 ms | baseline | 156.766 MiB | baseline | 68/68 |
| QDQ INT8 | 0.40647 ms | 2.027x | 125.293 MiB | 20.08% | 70/70 |

## Conclusion boundaries

- QDQ INT8 loses only 0.05 percentage points versus FP32 on Galaxy S22.
- The 0.11 percentage-point increase from local QDQ to S22 QDQ must not be interpreted as quantization improving generalization accuracy.
- The 120 local/S22 QDQ prediction changes show backend numerical drift, while aggregate accuracy remains stable.
- piecewise_v065 has only 82.80% local accuracy and 1.68766 ms mean latency, so no full Galaxy S22 CIFAR-10 task was run for it.
- piecewise_v065 is retained only as compiler-compatibility and operator-decomposition diagnostic evidence; it is not the current best deployment.
- The final recommended deployment is standard QDQ INT8.

`predictions.npz` contains labels, original indices, and four prediction arrays only. It contains no image or logit arrays.
