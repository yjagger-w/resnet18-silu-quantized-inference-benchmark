# v1.2 ONNX Runtime C++ custom-op integration

## Status

v1.2 is **implemented and verified** as a Windows x64 ONNX Runtime custom-op DLL. The frozen v1.1 model is rewritten reproducibly to 17 project-domain custom nodes. The DLL uses the existing v1.0 portable scalar uint8 lookup kernel and exports the standard `RegisterCustomOps` entry point.

The required zero-tolerance 128-image gate passed before full evaluation: every one of the 17 uint8 activation-code outputs and all 1,280 final logits were exactly equal, prediction agreement was 100%, and ORT profiling recorded 16 kernel executions for every custom node. The complete custom-op graph then produced 9,358/10,000 correct predictions (93.58%).

## Prerequisite audit

| Component | Result |
|---|---|
| Python ORT | 1.19.2 |
| Runtime DLL | `E:\ProgramData\Anaconda_envs\envs\mamba_ptq\Lib\site-packages\onnxruntime\capi\onnxruntime.dll` |
| Official SDK root | `E:\sdk\onnxruntime-win-x64-1.19.2` |
| `onnxruntime_c_api.h` | available |
| `onnxruntime_cxx_api.h` | available |
| ORT import library | `E:\sdk\onnxruntime-win-x64-1.19.2\lib\onnxruntime.lib` |
| Visual Studio Build Tools | 17.14.39 |
| Compiler environment | MSVC through `vcvars64.bat` |
| CMake | Visual Studio bundled CMake |

The SDK intentionally has no runtime DLL. The runner adds the installed Python package's `capi` directory with `os.add_dll_directory` and preloads its absolute ORT 1.19.2 DLL for the current process before registering the project DLL. It does not copy DLLs or change global `PATH`.

Check without modifying the environment:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
$env:ORT_ROOT = 'E:\sdk\onnxruntime-win-x64-1.19.2'
python scripts\check_ort_cpp_customop_prerequisites.py --require
```

## Frozen source

| Item | Value |
|---|---|
| Candidate | `two_segment_p99_99_mse` |
| Source model SHA-256 | `0df5e7f6388d39a658a53790788d20393ff3e8e481d075b85ca84a54e5821a8d` |
| Selection receipt hash | `bd524f3ae909bfb27df211b6d37b95495aadb994b7f10e975a1e87aaf8385254` |
| Frozen test accuracy | 93.58% (v1.1 standard-operator reference only) |
| Target sites | 17 |

The rewrite rejects any source model or selection receipt whose hashes differ.

## Quantization-boundary audit

The v1.0 kernel is a canonical uint8-to-uint8 lookup kernel. The selected v1.1 graph, however, receives each pre-SiLU activation through a retained `DequantizeLinear` and emits both an internal piecewise uint8 code and a reconstructed float activation.

To preserve every surrounding Q/DQ node while allowing the custom op to call the shared kernel, the v1.2 node contract is:

```text
inputs:
  0: upstream uint8 activation code
  1: the existing float32 DequantizeLinear output

outputs:
  0: piecewise uint8 activation code
  1: reconstructed float32 activation used by the unchanged downstream graph
```

The code input is an additional consumer of the existing upstream quantized tensor. The original DQ node and its float output remain present and wired to the custom node. This makes the transport boundary explicit and avoids describing the implementation as a float-only SiLU.

Each node contains deterministic attributes for:

- site identity and contract version;
- exact float32 input scale bits and input zero point;
- `Vmin`, `Vsplit`, and `Vmax` encoded as hexadecimal floating-point strings;
- lower/upper derived scales and zero points;
- 8-bit code ranges, upper-segment split ownership, and nearest-even rounding.

There are no hidden global parameter tables or environment-dependent per-layer state.

## Generated graph rewrite

Run:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python scripts\rewrite_ort_cpp_customop_model.py `
  --config configs\benchmarks\resnet18_silu_cifar10_v12_ort_customop_cpu.json
```

The generated model requires the matching registered project DLL.

| Graph property | Result |
|---|---:|
| Selected piecewise islands removed | 17 |
| Standard-operator island nodes removed | 408 |
| `com.yjagger.silu::QuantizedPiecewiseSiLU` nodes added | 17 |
| Original selected-island nodes retained | 0 |
| Source initializers | 375 |
| Rewritten initializers | 188 |
| Unused island initializers pruned | 187 |
| Source nodes | 552 |
| Rewritten nodes | 161 |

Generated provenance from this run:

- rewritten model SHA-256: `18e51fe02b1e9d5fbe6cd56f1221d052c6753b13bfc4e553755d9252b2ca09ba`
- sidecar manifest hash: `1d153bd13b3c4f8bd9871e09a1e7d0604d2d49d13ae546dc22265b38810b9ea2`

The sidecar records the source and rewritten hashes, selection receipt, timestamp/tool version, all source island nodes and initializers, custom node name, entry/exit tensors, and exact attributes. Validation rejects stale or tampered models, sidecars, receipts, contracts, and source models.

Generated files remain ignored under:

- `artifacts/accuracy_recovery/v1.2/`
- `results/benchmarks/v1.2_ort_cpp_customop/`
- `build/ort-cpp-customop/`

## Build and execution

```powershell
$env:ORT_ROOT = 'E:\sdk\onnxruntime-win-x64-1.19.2'
$env:PYTHONPATH = (Resolve-Path .\src).Path
cmake -S cpp -B build\ort-cpp-customop -DCMAKE_BUILD_TYPE=Release -DORT_ROOT="$env:ORT_ROOT"
cmake --build build\ort-cpp-customop --config Release
ctest --test-dir build\ort-cpp-customop -C Release --output-on-failure
python scripts\run_ort_cpp_customop.py
```

The generated DLL is `build/ort-cpp-customop/Release/silu_ort_custom_op.dll`. The run command first performs the frozen exact probe and stops on the first divergence. Full evaluation and the separate uninstrumented benchmark run only after that gate passes.

## Verified results

| Evidence | Result |
|---|---:|
| Profiled custom nodes | 17/17 |
| Kernel executions per node in probe | 16 |
| Per-site uint8 code equality | exact at all 17 sites |
| Final-logit equality | exact, 1,280/1,280 values |
| Probe prediction agreement | 100% |
| Full CIFAR-10 top-1 | 93.58% (9,358/10,000) |
| Uninstrumented batch-1 mean latency | 85.9749 ms |
| Uninstrumented batch-1 p50 / p95 | 72.9944 / 181.8344 ms |
| Uninstrumented batch-1 throughput | 11.63 images/s |

The timing values are one local CPU run with 20 warmups and 100 timed iterations. They are measurements of an **ORT CPU hybrid graph with project C++ custom-op activations**. They are not integer-only whole-model inference or a speedup claim.

## Allowed wording

Allowed claim:

> A deterministic, hash-locked ONNX rewrite maps the selected v1.1 reference to 17 project C++ custom-op activations. On the frozen 128-image probe, ORT profiling confirms all 17 nodes execute and all activation codes and final logits match exactly. The resulting ORT CPU hybrid graph reaches 93.58% on the complete CIFAR-10 test set.

Not allowed: integer-only whole-model inference, accelerator deployment, or speedup claims.
