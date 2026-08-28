# v1.2 ONNX Runtime C++ custom-op integration

## Status

v1.2 is currently **graph-contract complete and C++ integration blocked**. The frozen v1.1 model can be rewritten reproducibly to 17 project-domain custom nodes, but this environment does not contain the official ONNX Runtime C/C++ custom-op headers required to implement and build the DLL safely.

No DLL was built. No custom-op ORT session, 128-image parity run, execution profile, 10,000-image evaluation, or timing benchmark was run. The existing v1.1 93.58% result is unchanged and must not be presented as a v1.2 custom-op result.

## Prerequisite audit

| Component | Result |
|---|---|
| Python ORT | 1.19.2 |
| Runtime DLL | `E:\ProgramData\Anaconda_envs\envs\mamba_ptq\Lib\site-packages\onnxruntime\capi\onnxruntime.dll` |
| `onnxruntime_c_api.h` | missing |
| `onnxruntime_cxx_api.h` | missing |
| ORT import library | not present in the Python wheel |
| Visual Studio Build Tools | 17.14.39 |
| Compiler environment | MSVC through `vcvars64.bat` |
| CMake | Visual Studio bundled CMake |

The minimal safe unblock is to provide the official ONNX Runtime **1.19.2 x64 development package** matching the installed Python runtime, including its `include` directory, and set `ORT_ROOT` to the extracted package. Do not install or upgrade Python packages merely to obtain headers. A matching import library should also be retained if the final implementation/link strategy requires it.

Check without modifying the environment:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python scripts\check_ort_cpp_customop_prerequisites.py
python scripts\check_ort_cpp_customop_prerequisites.py --require
```

The second command returns a non-zero status while required headers are absent.

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

To preserve every surrounding Q/DQ node while allowing the future custom op to call the shared kernel, the v1.2 node contract is:

```text
inputs:
  0: upstream uint8 activation code
  1: the existing float32 DequantizeLinear output

outputs:
  0: piecewise uint8 activation code
  1: reconstructed float32 activation used by the unchanged downstream graph
```

The code input is an additional consumer of the existing upstream quantized tensor. The original DQ node and its float output remain present and wired to the custom node. This makes the transport boundary explicit and avoids describing the future implementation as a float-only SiLU.

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

The generated model is not directly runnable without a matching registered DLL.

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

## Work deferred by the blocker

The following are intentionally not implemented or claimed until official headers are supplied:

- the C++ custom-op wrapper and standard ORT registration entry point;
- CMake custom-op shared-library target;
- source-level proof that the wrapper calls `QuantizedSiluScalar`;
- DLL registration with `SessionOptions.register_custom_ops_library`;
- shape/type execution validation in ORT;
- profiling proof that all 17 nodes executed;
- exact per-site code and final-logit parity;
- complete CIFAR-10 accuracy and CPU timing benchmark.

Once unblocked, the required order is: implement the wrapper using the existing kernel, build and run its tests, register the DLL, prove 17-node execution and zero-tolerance parity on the frozen 128-image probe, and only then run the full 10,000-image evaluation and separate uninstrumented benchmark.

## Allowed wording

Current allowed claim:

> A deterministic, hash-locked ONNX rewrite maps the selected v1.1 functional reference to 17 explicit project-domain custom-node contracts. C++ ORT execution remains blocked by missing official development headers.

Not allowed: a working ORT custom-op DLL, C++ execution parity, deployment, integer-only whole-model inference, acceleration, or speedup.
