# v1.3 ORT C++ custom-op CPU performance diagnosis

## Scope

This is a diagnostic of an **ORT CPU hybrid graph with project C++ custom-op activations**. It is not integer-only whole-model inference. v1.3 changes neither the v1.0 scalar-kernel mathematics nor the v1.2 custom-op contract, model topology, attributes, rounding, saturation, split ownership, calibration values, or inputs. No vectorized kernel, custom parallelism, thread pool, graph rewrite, fusion, or benchmark-oriented semantic change was implemented.

The run used Windows `10.0.26200`, an `Intel64 Family 6 Model 197 Stepping 2` processor with 16 logical CPUs, Python `3.9.25`, ONNX Runtime `1.19.2`, and `CPUExecutionProvider`. The package runtime DLL reported file version `1.19.20240830.4.ffceed9`. The diagnostic runner is NumPy-only and did not import PyTorch.

## Frozen provenance and exactness

| Artifact | SHA-256 |
|---|---|
| v1.1 standard-operator model | `0df5e7f6388d39a658a53790788d20393ff3e8e481d075b85ca84a54e5821a8d` |
| v1.2 custom-op model | `18e51fe02b1e9d5fbe6cd56f1221d052c6753b13bfc4e553755d9252b2ca09ba` |
| `silu_ort_custom_op.dll` | `24be06487486adb9494973b60210a7b723117d2bf8e67324e6e5eaac347dedd2` |
| Deterministic 128-image probe | `716fd224a92f97edc2af532191716b8f8ff80ed9017beade3a2a359dc84d2524` |

The exactness gate ran first with 128 images in sixteen batches of eight. All exposed uint8 activation-code tensors were exactly equal at all 17 sites, all 1,280 final logit values were bit-exact, and prediction agreement was 100%. Benchmarking proceeded only after this gate passed.

## Timing comparability

The historical v1.2 mean of `85.9749 ms` is not directly protocol-comparable with the v0.6.5 Standard-QDQ latency. Although both used batch 1, 20 warmups, 100 timings, default ORT threads, and `ORT_SEQUENTIAL`, they differed in input content/digest, selected outputs, session lifecycle, and between-invocation work. v0.6.5 timed an explicit logit output on a session already used for the complete evaluation and queried process memory after each invocation; v1.2 used a fresh session and requested every declared output. Their graph forms also differ. The old values therefore remain context only.

The v1.3 custom and v1.3 standard-operator cells are protocol-matched: the same cached inputs, explicit logit output, provider, session settings, timing boundary, fresh-session repetitions, warmups, and iteration counts. They are measurements of different graph forms, not a speedup claim. v1.1 itself had no dedicated latency benchmark; it recorded only batched evaluation wall time.

## Profiling evidence

The two graphs were profiled in separate fresh sessions with the same cached batch-1 input and 10 invocations. The parser accepts unordered and sparse traces and selects only events whose category is `Node`, whose event name ends in `_kernel_time`, and whose ORT arguments expose `op_name` and `node_name`. Observed trace categories were `Node` and `Session`. ORT scheduling and parallel overlap mean summed node-event durations are not a wall-clock decomposition.

For the custom graph, DLL registration/load took `2.074 ms`, session creation/load took `110.798 ms`, ten profiled inferences took `831.162 ms` wall-clock, and selected node events summed to `763.709 ms`. All 17 named `QuantizedPiecewiseSiLU` nodes appeared and each executed 10 times, for 170 profile executions total.

| Custom-graph category | Total node-event time | Share |
|---|---:|---:|
| Convolution | 490.686 ms | 64.25% |
| Quantization-related | 191.772 ms | 25.11% |
| Project custom activations | 64.795 ms | 8.48% |
| Other | 14.109 ms | 1.85% |
| Data movement/copy | 2.347 ms | 0.31% |

The five largest individual nodes were convolution nodes:

| Rank | Node | Share of custom-graph node-event time |
|---:|---|---:|
| 1 | `/layer3/layer3.1/conv2/Conv` | 6.39% |
| 2 | `/layer2/layer2.1/conv1/Conv` | 5.72% |
| 3 | `/layer4/layer4.0/conv2/Conv` | 5.64% |
| 4 | `/layer3/layer3.0/conv2/Conv` | 5.49% |
| 5 | `/layer3/layer3.1/conv1/Conv` | 4.99% |

The ignored JSON and CSV reports contain the full per-node/per-op aggregates and top 20 ranking. In the standard-operator profile, quantization-related expanded activation work accounted for 59.55% of node-event time and convolution for 38.67%; this attribution is graph-form-specific.

## Repeated benchmark matrix

Each cell used five fresh sessions, 20 warmups per session, and 100 timed invocations per session. Profiling was disabled, precise monotonic timing surrounded only `session.run`, all 500 raw timings per cell were retained, and no outlier was deleted. Across 16 cells, 8,000 timings were retained.

| Graph | Batch | Intra-op threads | Mode | Mean ms | P50 ms | P95 ms | CV | Images/s |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Custom | 1 | default | sequential | 71.063 | 68.243 | 112.062 | 0.316 | 14.07 |
| Standard operator | 1 | default | sequential | 105.788 | 108.464 | 136.568 | 0.216 | 9.45 |
| Custom | 1 | 1 | sequential | 46.166 | 47.275 | 66.351 | 0.282 | 21.66 |
| Custom | 1 | 2 | sequential | 33.137 | 33.217 | 45.982 | 0.240 | 30.18 |
| Custom | 1 | 4 | sequential | 25.649 | 25.722 | 36.732 | 0.298 | 38.99 |
| Custom | 1 | 8 | sequential | 29.031 | 26.098 | 53.689 | 0.494 | 34.45 |
| Standard operator | 1 | 1 | sequential | 67.797 | 69.684 | 93.765 | 0.277 | 14.75 |
| Standard operator | 1 | 2 | sequential | 51.676 | 52.186 | 72.629 | 0.257 | 19.35 |
| Standard operator | 1 | 4 | sequential | 41.515 | 40.183 | 59.927 | 0.256 | 24.09 |
| Standard operator | 1 | 8 | sequential | 47.981 | 42.703 | 81.319 | 0.528 | 20.84 |
| Custom | 1 | default | parallel | 93.179 | 86.949 | 165.476 | 0.458 | 10.73 |
| Standard operator | 1 | default | parallel | 132.540 | 131.964 | 184.680 | 0.245 | 7.54 |
| Custom | 8 | default | sequential | 506.911 | 463.541 | 799.930 | 0.351 | 15.78 |
| Custom | 32 | default | sequential | 2015.051 | 1976.693 | 2622.372 | 0.192 | 15.88 |
| Standard operator | 8 | default | sequential | 1069.707 | 947.697 | 1838.980 | 0.344 | 7.48 |
| Standard operator | 32 | default | sequential | 3770.778 | 3584.619 | 5483.706 | 0.219 | 8.49 |

For the default/sequential custom baseline, per-repetition p50 ranged from `64.389` to `70.393 ms` and p95 ranged from `92.906` to `135.236 ms`. The high p95 was reproduced in every repetition, so the variance was not a single-run artifact. The four-thread result improved at least four of five repetition medians without increasing aggregate CV relative to default. The eight-thread setting had a slightly lower aggregate p50 but worse mean, p95, and CV, so it was rejected as unstable. Parallel execution mode was also worse for this batch-1 control.

Batch 8 and 32 rows are throughput controls and must not be compared as batch-1 latency. Their custom-graph throughput plateaued around `15.8 images/s` with default threads on this machine.

## Decisions

| Option | Decision | Evidence |
|---|---|---|
| Vectorized custom kernel | Not recommended as the next priority | All 17 custom activations contributed only 8.48% of selected node-event time; the dominant measured category was convolution at 64.25%. |
| ORT thread tuning | Recommend `intra_op_num_threads=4` for this measured machine/protocol | Stable repeated-median improvement over default; eight threads and parallel mode worsened aggregate latency and/or variance. |
| Graph/runtime work | Investigate convolution/runtime configuration, not a new graph rewrite | Convolution is the largest non-custom category. The diagnosis does not establish that changing graph semantics is justified. |
| Smallest next experiment | One isolated batch-1 profile at four intra-op threads | Verify whether convolution remains dominant under the recommended session setting before considering any implementation work. |

## Reproduction and outputs

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
$env:ORT_ROOT = 'E:\sdk\onnxruntime-win-x64-1.19.2'

& 'E:\ProgramData\Anaconda_envs\envs\mamba_ptq\python.exe' `
  scripts\diagnose_ort_cpp_customop_performance.py `
  --config configs\benchmarks\resnet18_silu_cifar10_v13_ort_customop_performance.json `
  --force-rebuild
```

Ignored outputs are written under `results/benchmarks/v1.3_ort_customop_performance_diagnosis/`, including `diagnosis_report.json`, `diagnosis_report.md`, `comparability_audit.json`, `profile_analysis.json`, `benchmark_matrix.json`, `benchmark_matrix.csv`, `profile_nodes.csv`, `exactness_gate.json`, and the raw ORT profiles.

## Limitations

- The measurements cover one Windows x64 CPU machine and ORT 1.19.2; they do not establish cross-machine behavior.
- Profile node sums can overlap and do not equal wall time.
- Node-event category shares depend on graph form and ORT profiling overhead.
- The matrix is deliberately bounded and is not a full factorial thread/batch/mode search.
- No new 10,000-image evaluation was run. v1.2's frozen 93.58% result remains the accuracy evidence.
- No performance improvement, SIMD acceleration, whole-model integer-only execution, or accelerator deployment claim is made.
