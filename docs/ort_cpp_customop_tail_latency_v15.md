# v1.5 ORT CPU runtime tail-latency investigation

## Scope and non-goals

v1.5 investigates batch-1 tail latency for the **ORT CPU hybrid graph with project C++ custom-op activations**. It changes only session-local ONNX Runtime options in a bounded experiment. The v1.0 scalar kernel, v1.2 DLL, custom-op attributes, calibration, quantization mathematics, model topology, ONNX artifacts, process priority, affinity, power settings, registry, global environment, and package versions remain unchanged.

This is one Windows x64 machine with ONNX Runtime `1.19.2` and `CPUExecutionProvider`. It is not a universal CPU recommendation, accelerator result, whole-model integer-only claim, graph rewrite, or AVX2 implementation.

## Frozen provenance

| Artifact | SHA-256 |
|---|---|
| v1.1 selected standard-operator model | `0df5e7f6388d39a658a53790788d20393ff3e8e481d075b85ca84a54e5821a8d` |
| v1.2 custom-op model | `18e51fe02b1e9d5fbe6cd56f1221d052c6753b13bfc4e553755d9252b2ca09ba` |
| Unchanged v1.2 custom-op DLL | `24be06487486adb9494973b60210a7b723117d2bf8e67324e6e5eaac347dedd2` |
| Deterministic 128-image probe | `716fd224a92f97edc2af532191716b8f8ff80ed9017beade3a2a359dc84d2524` |
| Cached batch-1 input | `2170a7b88671b9771e76ccb5af60e284ef35695219588a34b6f9f2ddbfc4beef` |

The runtime was ORT `1.19.2`, package DLL file version `1.19.20240830.4.ffceed9`, on Windows `10.0.26200`, an `Intel64 Family 6 Model 197 Stepping 2` processor, and 16 logical CPUs. The diagnostic process was NumPy/ORT-only and did not import PyTorch. The existing process-local DLL loader was reused.

## Protocol

The frozen v1.4 baseline retained:

```text
CPUExecutionProvider
ORT_SEQUENTIAL
intra_op_num_threads = 4
inter_op_num_threads = 0
ORT_ENABLE_ALL
CPU memory arena enabled
memory pattern enabled
```

The baseline used 10 fresh sessions. Each session had 20 warmups followed by 100 timed invocations. Each screening candidate first passed an exact 128-image final-logit gate, then used five fresh 20×100 repetitions. Timing used `perf_counter_ns` immediately around `session.run` with a cached input and explicit logit output. Cold session creation was measured separately. All raw samples and worse candidates were retained; no outlier was removed.

The matrix changed one logical setting at a time. Parallel execution required an explicit inter-op value of two as part of that single logical control. The 3/5/6-thread rows are nearby explanatory controls and not implicit replacements for the four-thread preset.

## Exactness

All nine tested rows—including baseline—passed the deterministic 128-image gate:

- All 1,280 final logits were exactly equal to the frozen v1.1 standard-operator candidate.
- Prediction agreement was 100% for every row.
- Timing was not permitted before the corresponding exactness result passed.
- Requested and actually applied options are preserved in `diagnosis_report.json`; ORT accepted every requested setting.

No 10,000-image evaluation was run. Frozen v1.2 accuracy evidence remains unchanged.

## Baseline reproducibility

The new 10×100 baseline result was:

| Mean | P50 | P95 | P99 | Min | Max | Std | CV | Images/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 27.479 ms | 26.213 ms | 35.160 ms | 46.849 ms | 22.247 ms | 55.891 ms | 4.296 ms | 0.156 | 36.39 |

Cold session creation was separate: mean `70.075 ms`, p50 `70.370 ms`, p95 `75.969 ms`.

| Repetition | Mean | P50 | P95 | P99 | CV |
|---:|---:|---:|---:|---:|---:|
| 1 | 27.943 | 26.071 | 36.380 | 46.849 | 0.166 |
| 2 | 26.624 | 26.091 | 32.096 | 34.002 | 0.103 |
| 3 | 28.520 | 26.950 | 40.662 | 49.046 | 0.193 |
| 4 | 31.197 | 29.561 | 46.112 | 54.092 | 0.209 |
| 5 | 25.620 | 25.417 | 27.677 | 29.459 | 0.047 |
| 6 | 25.879 | 25.417 | 28.305 | 32.431 | 0.073 |
| 7 | 26.388 | 25.896 | 30.470 | 33.920 | 0.080 |
| 8 | 26.768 | 26.036 | 33.003 | 38.595 | 0.113 |
| 9 | 27.700 | 26.906 | 35.018 | 36.218 | 0.138 |
| 10 | 28.207 | 26.676 | 39.673 | 48.645 | 0.179 |

The v1.4 run had aggregate p95 `53.548 ms`, CV `0.477`, and one repetition p95 `101.500 ms`. In v1.5, maximum repetition p95 was `46.112 ms` and CV was `0.156`. Under the predeclared rule, the extreme v1.4 tail behavior was **not reproduced**. This does not prove it cannot recur; it shows it was not persistent across this fresh 10-repetition process.

## Complete screening matrix

| Candidate | Changed setting | Mean | P50 | P95 | P99 | CV | Images/s | Exact |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `baseline_v14` | None; frozen v1.4 | 27.479 | 26.213 | 35.160 | 46.849 | 0.156 | 36.39 | Yes |
| `cpu_arena_disabled` | CPU arena off | 34.747 | 33.831 | 41.276 | 48.932 | 0.116 | 28.78 | Yes |
| `memory_pattern_disabled` | Memory pattern off | 26.866 | 25.461 | 36.257 | 48.824 | 0.183 | 37.22 | Yes |
| `graph_extended` | `ORT_ENABLE_EXTENDED` | 25.523 | 24.128 | 31.092 | 52.222 | 0.240 | 39.18 | Yes |
| `graph_basic` | `ORT_ENABLE_BASIC` | 28.197 | 26.611 | 36.839 | 54.620 | 0.198 | 35.46 | Yes |
| `execution_parallel_i2` | `ORT_PARALLEL`, inter-op 2 | 32.045 | 30.090 | 45.594 | 59.499 | 0.232 | 31.21 | Yes |
| `intra_threads_3` | Intra-op 3 | 29.438 | 28.834 | 33.576 | 38.267 | 0.081 | 33.97 | Yes |
| `intra_threads_5` | Intra-op 5 | 25.619 | 25.097 | 29.381 | 35.360 | 0.086 | 39.03 | Yes |
| `intra_threads_6` | Intra-op 6 | 32.567 | 31.552 | 46.207 | 53.882 | 0.228 | 30.71 | Yes |

Disabling the CPU arena, basic graph optimization, parallel execution, three threads, and six threads were worse in one or more important measures. Disabling memory pattern and extended graph optimization did not satisfy the full no-worse-tail/CV rule. Five threads was the only screening row that met all predeclared screening checks, so it—not the lowest isolated minimum—entered confirmation.

## Alternating finalist confirmation

The confirmation alternated baseline and five threads in fixed order for 10 fresh-session repetitions per arm, each 20×100. No sample was removed.

| Arm | Mean | P50 | P95 | P99 | CV | Median repetition P50 | Median repetition P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Four-thread v1.4 baseline | 26.056 | 25.318 | 30.269 | 36.024 | 0.108 | 25.213 | 29.422 |
| Five-thread finalist | 27.194 | 25.820 | 35.431 | 41.121 | 0.166 | 25.437 | 31.101 |

The finalist failed all confirmation checks: aggregate p50, median repetition p50, median repetition p95, and CV were all worse. Therefore, no new preset was validated and five threads was rejected. This illustrates why a clean screening row is insufficient evidence by itself.

## Independent profiles

Two separate fresh-session profiles used the unchanged v1.4 baseline, with 10 inferences each. Every profile contained all 17 custom nodes, each executed 10 times—170 executions per profile.

| Profile | Wall time | Node-event sum | Convolution | Quantization-related | Custom activations |
|---|---:|---:|---:|---:|---:|
| Baseline profile 1 | 333.707 ms | 298.575 ms | 49.74% | 34.39% | 12.48% |
| Baseline profile 2 | 353.543 ms | 317.956 ms | 51.78% | 32.97% | 11.98% |

Convolution was the largest category in both. Custom share differed by only `0.50` percentage points, below the recorded five-point material-difference rule. The two profiles therefore agree on category ordering and broadly agree on attribution. Node-event sums are not a wall-time decomposition because scheduling and overlap can make them non-additive.

The rejected five-thread finalist was separately profiled: wall `388.123 ms`, node-event sum `348.940 ms`, convolution `51.98%`, quantization-related `32.03%`, and custom activations `12.90%`. It also executed all 17 custom nodes 10 times.

## Final decisions

| Question | Conclusion |
|---|---|
| Is v1.4 tail latency reproducible? | **No in this fresh 10×100 process**; maximum repetition p95 and aggregate CV were both below v1.4. This is not proof that an OS-level tail cannot recur. |
| Does any ORT setting improve typical latency? | **None demonstrated after confirmation.** |
| Does any setting reduce tail variance? | **None demonstrated after confirmation.** |
| Recommended preset | **Retain v1.4**—sequential, four intra-op threads, inter-op zero, `ORT_ENABLE_ALL`, arena and memory pattern enabled. |
| Conv remains dominant? | **Yes**, independently measured twice. |
| AVX2 priority | **Continue deferred**; custom activation share was stable near 12%, while convolution remained about 50% in both profiles. |
| Next smallest experiment | Repeat the unchanged v1.4 10×100 baseline in one separate process to estimate run-to-run tail incidence. |

No README recommendation was changed because no replacement configuration survived confirmation.

## Reproduction and ignored reports

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
$env:ORT_ROOT = 'E:\sdk\onnxruntime-win-x64-1.19.2'

& 'E:\ProgramData\Anaconda_envs\envs\mamba_ptq\python.exe' `
  scripts\diagnose_ort_cpp_customop_tail_latency.py `
  --config configs\runtime_profiles\resnet18_silu_cifar10_v15_ort_customop_tail_latency.json `
  --force-rebuild
```

Generated evidence is ignored under `results/benchmarks/v1.5_ort_customop_tail_latency/`, including `diagnosis_report.json`, `diagnosis_report.md`, `screening_matrix.json`, `screening_matrix.csv`, `raw_timings.csv`, `profile_comparison.json`, `profile_nodes.csv`, `finalist_confirmation.json`, partial progress, and raw ORT profiles.

## Limitations

- Results cover one Windows x64 machine and ORT 1.19.2 only.
- External OS scheduling and background activity were not manipulated and can affect tail latency.
- Screening used five repetitions per candidate; only the finalist received 10-arm alternating confirmation.
- Profile instrumentation changes execution overhead, so only cautious category comparisons are made.
- Nearby thread settings were explanatory controls; no replacement preset was created.
- No full CIFAR-10 evaluation or implementation optimization was performed.
