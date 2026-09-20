# v1.4 validated four-thread ORT custom-op runtime preset

## Scope and contract

v1.4 adds an explicitly selected runtime preset for the **ORT CPU hybrid graph with project C++ custom-op activations**. It does not change the v1.0 scalar kernel, the v1.2 DLL, either ONNX graph, quantization parameters, calibration, rounding, saturation, topology, or model inputs. It is not a whole-model integer-only inference claim.

| Preset field | Value |
|---|---|
| Name | `resnet18-silu-cifar10-ort-customop-cpu-4threads` |
| Version | `1.4` |
| Provider | `CPUExecutionProvider` |
| Execution mode | `ORT_SEQUENTIAL` |
| Intra-op threads | `4` |
| Inter-op threads | `0`—ORT default and unused by sequential execution |
| Graph optimization | `ORT_ENABLE_ALL` |
| Intended workload | Batch-1 ResNet18-SiLU CIFAR-10 CPU inference with the frozen v1.2 graph |
| Preset fingerprint | `4a9c7e7fcb5ca54cff2ad1d680560486e2835f75f93850aac2cbe7f7dd5c1709` |

The preset is opt-in through the v1.4 config/runner. The unchanged backend default still creates an ordinary `SessionOptions` object with ORT's default intra-op setting. Validation rejects non-positive or non-four thread values, unsupported providers, wrong ORT version, invalid output roots, and model/DLL hash mismatches.

This recommendation is scoped to Windows `10.0.26200`, an `Intel64 Family 6 Model 197 Stepping 2` processor with 16 logical CPUs, Python `3.9.25`, and ONNX Runtime `1.19.2` using `CPUExecutionProvider`. The package runtime DLL reported file version `1.19.20240830.4.ffceed9`. The process-local DLL loader was reused; no DLL was copied and no global path was changed.

## Frozen provenance

| Artifact | SHA-256 |
|---|---|
| v1.1 selected standard-operator model | `0df5e7f6388d39a658a53790788d20393ff3e8e481d075b85ca84a54e5821a8d` |
| v1.2 custom-op model | `18e51fe02b1e9d5fbe6cd56f1221d052c6753b13bfc4e553755d9252b2ca09ba` |
| Unchanged v1.2 custom-op DLL | `24be06487486adb9494973b60210a7b723117d2bf8e67324e6e5eaac347dedd2` |
| Deterministic 128-image probe | `716fd224a92f97edc2af532191716b8f8ff80ed9017beade3a2a359dc84d2524` |

The NumPy/ORT-only validation process did not import PyTorch.

## Exactness gate

The exactness gate ran before profiling or timing. It used the frozen 128 indices in sixteen batches of eight and applied the four-thread preset to both sessions.

- All 17 exposed uint8 activation-code tensors were exactly equal.
- All 1,280 final logits were exactly equal.
- Prediction agreement was 100%.
- The custom session actually reported `ExecutionMode.ORT_SEQUENTIAL`, four intra-op threads, inter-op value zero, `GraphOptimizationLevel.ORT_ENABLE_ALL`, profiling disabled, and `CPUExecutionProvider`.

No 10,000-image evaluation was run. The frozen v1.2 result of 93.58% remains the accuracy evidence.

## Four-thread profile

One isolated custom-graph profile retained 10 batch-1 invocations of the same cached NumPy input. DLL registration/load took `2.333 ms`, session creation/load took `87.022 ms`, inference wall time was `400.644 ms`, and selected node-event durations summed to `362.244 ms`. Node-event sums can overlap and are not a wall-clock decomposition.

All 17 `QuantizedPiecewiseSiLU` nodes appeared in the trace and each executed 10 times, for 170 executions total.

| Category | Total node-event time | Share |
|---|---:|---:|
| Convolution | 191.328 ms | 52.82% |
| Quantization-related | 113.009 ms | 31.20% |
| Custom activations | 46.413 ms | 12.81% |
| Other | 9.374 ms | 2.59% |
| Data movement/copy | 2.120 ms | 0.59% |

Convolution remains the largest category. The five largest nodes were:

| Rank | Node | Category | Share |
|---:|---|---|---:|
| 1 | `/layer3/layer3.0/conv2/Conv` | Convolution | 4.45% |
| 2 | `onnx::Conv_266_DequantizeLinear` | Quantization-related | 4.31% |
| 3 | `/layer4/layer4.0/conv2/Conv` | Convolution | 4.22% |
| 4 | `/layer3/layer3.1/conv1/Conv` | Convolution | 4.11% |
| 5 | `onnx::Conv_275_DequantizeLinear` | Quantization-related | 4.10% |

The full top 20 and raw trace are retained in the ignored JSON/CSV outputs. v1.3 used a different thread configuration, so profile durations are not compared directly. Category ordering is consistent—convolution remains dominant—but custom share increased from 8.48% to 12.81%. That exceeds the prior 10% diagnostic boundary and is reported as materially different attribution evidence.

## Repeated batch-1 validation

Both graph controls used the same cached input, explicit logit output, four-thread preset, five fresh sessions, 20 warmups per session, and 100 timed invocations per session. Profiling was disabled, all raw samples were retained, and no outlier was deleted.

| Graph form | Mean | P50 | P95 | Min | Max | Std | CV | Images/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1.2 hybrid custom-op graph | 29.607 ms | 26.412 ms | 53.548 ms | 14.685 ms | 143.203 ms | 14.111 ms | 0.477 | 33.77 |
| v1.1 expanded standard-operator graph | 50.581 ms | 50.037 ms | 59.822 ms | 38.110 ms | 122.467 ms | 6.455 ms | 0.128 | 19.77 |

The standard-operator row is a different graph form and is not a speedup baseline.

Custom-op repetition summaries:

| Repetition | Mean | P50 | P95 | CV |
|---:|---:|---:|---:|---:|
| 1 | 27.256 ms | 26.792 ms | 30.756 ms | 0.084 |
| 2 | 26.626 ms | 26.236 ms | 34.035 ms | 0.142 |
| 3 | 27.079 ms | 25.989 ms | 36.908 ms | 0.197 |
| 4 | 44.442 ms | 34.017 ms | 101.500 ms | 0.570 |
| 5 | 22.663 ms | 22.120 ms | 29.066 ms | 0.180 |

The v1.4 protocol is machine-readably identical to the historical v1.3 four-thread cell. The current median of the five repetition p50 values was `26.236 ms`, inside the v1.3 repetition-p50 range of `23.388`–`27.418 ms`; typical latency was reproduced. The full distribution was not reproduced: current retained-sample CV was `0.477` versus historical `0.298`, driven by the fully retained fourth repetition.

## Decisions

| Question | Decision | Evidence |
|---|---|---|
| Adopt explicit four-thread preset | Validated with a tail-variance caveat for this machine/protocol | Exactness and all-node execution passed; typical historical latency reproduced; variance did not. The preset remains explicit and machine-scoped. |
| Convolution/runtime work | Remains the next narrow target | Convolution remains the largest category at 52.82%. |
| AVX2 | Deferred pending a second four-thread attribution profile | Custom share increased to 12.81%, so the old below-10% rationale is no longer sufficient; one more isolated attribution profile should confirm the change before kernel work is prioritized. |

No AVX2, OpenMP, custom thread pool, graph rewrite, model fusion, calibration change, ONNX rewrite, or C++ kernel semantic change was implemented.

## Reproduction and ignored reports

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
$env:ORT_ROOT = 'E:\sdk\onnxruntime-win-x64-1.19.2'

& 'E:\ProgramData\Anaconda_envs\envs\mamba_ptq\python.exe' `
  scripts\validate_ort_cpp_customop_thread_preset.py `
  --config configs\runtime_profiles\resnet18_silu_cifar10_v14_ort_customop_cpu_4threads.json `
  --force-rebuild
```

Generated evidence is ignored under `results/benchmarks/v1.4_ort_customop_thread_preset/`: `validation_report.json`, `validation_report.md`, `exactness_gate.json`, `profile_analysis.json`, `benchmark_results.json`, `benchmark_summary.csv`, `raw_timings.csv`, `profile_nodes.csv`, `historical_comparison.json`, the instrumented probe models, and the raw ORT profile.

## Limitations

- This is one Windows x64 machine and one ORT version; no universal CPU or cross-machine claim follows.
- Tail latency was not fully reproduced, so the preset carries an explicit variance caveat.
- Profile category shares depend on runtime configuration and profiling overhead.
- Different graph forms are not presented as speedup evidence.
- No performance optimization or complete accuracy evaluation was performed in v1.4.
