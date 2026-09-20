# v1.0: portable C++17 quantized SiLU kernel

## Scope

This milestone is a standalone `uint8` SiLU activation kernel and microbenchmark. It is not a C++ ResNet18 inference engine, an ONNX custom operator, a model-accuracy result, or evidence for OpenVINO, QNN, GPU, NPU, DSP, or Hexagon deployment. The kernel is not inserted into either canonical ONNX model.

## Frozen Python relationship

The canonical output contract remains `src/silu_benchmark/quantization/activation.py::piecewise_quantize`, parameterized by `PiecewiseQuantizationSpec` in `src/silu_benchmark/quantization/spec.py`. The frozen v1.0 fixture uses the Python-origin `act.call_0` entry from `configs/calibration/resnet18_silu_piecewise_v06.json`; the fixture records the manifest SHA-256 and all derived fields.

That Python function receives an already-computed SiLU value. It does not define a quantized pre-SiLU input. The C++ API therefore keeps two parameter groups explicit:

- an ordinary uniform `uint8` input scale and zero point used to reconstruct the pre-SiLU float32 value; and
- the unchanged canonical piecewise output parameters `Vmin`, `Vsplit`, `Vmax`, and 8-bit code allocation.

The fixture input scale `0.0625` and zero point `128` are deterministic interface test parameters, not model-calibrated parameters and not a change to the project calibration algorithm. A deployment integration would have to supply the real producer tensor's input quantization parameters. The output parameters remain KLD/MSE-derived data from the frozen manifest: this milestone does not recalibrate them.

## Mathematical and numeric contract

For input code `q_in`, input scale `s_in`, and input zero point `z_in`, setup evaluates:

```text
x_float32 = float32((q_in - z_in) * s_in)
y_float32 = float32(x_float32 * sigmoid(x_float32))
```

The implementation uses stable positive and negative branches for SiLU. It creates a 256-entry lookup table once. The timed scalar kernel is then exactly:

```text
q_out[i] = lookup_table[q_in[i]]
```

For the output, let `L=127`, `U0=128`, and `U1=255`:

```text
s_lower = (Vsplit - Vmin) / L
s_upper = (Vmax - Vsplit) / (U1 - U0)
z_lower = round_even(-Vmin / s_lower)
z_upper = min(
    round_even(U0 - Vsplit / s_upper),
    floor(U0 - decoded_lower_endpoint / s_upper)
)
```

The post-SiLU value is first clipped to `[Vmin,Vmax]`. Values strictly below `Vsplit` use the lower affine segment and are explicitly saturated to codes 0–127. `Vsplit` and values above it use the upper segment and are explicitly saturated to 128–255. Rounding is deterministic nearest with ties to even and does not depend on the process floating-point rounding mode.

Both input and output dtypes are explicitly validated as `uint8`; `int8`, non-8-bit output, invalid ranges, non-finite or non-positive scales, invalid zero points, null non-empty buffers, and partially overlapping buffers produce actionable `std::invalid_argument` messages. Exact in-place operation is supported because each source code is read before its destination byte is written.

## API example

```cpp
#include "silu_benchmark/quantized_silu_kernel.h"

using namespace silu_benchmark;

const auto params = MakeQuantizedSiluKernelParams(
    UniformInputQuantizationParams{QuantizedDataType::kUInt8, 0.0625, 128, 0, 255},
    PiecewiseOutputQuantizationParams{
        QuantizedDataType::kUInt8,
        -0.27846449613571167,
        0.0970680373116061,
        0.8900823637959547,
        8,
    });

QuantizedSiluScalar(input_codes, output_codes, element_count, params);
```

The unchecked entry point exists only so the microbenchmark can exclude validation and alias checks after a checked call has already passed. Normal callers should use `QuantizedSiluScalar`.

## Golden fixture provenance and coverage

Run the generator without modifying any model or calibration artifact:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
& 'E:\ProgramData\Anaconda_envs\envs\mamba_ptq\python.exe' scripts\generate_cpp_silu_golden.py
```

The tracked JSON fixture and generated C++ header are under `cpp/tests/data/`. They contain parameters, quantized inputs, exact expected output codes, source provenance, the complete 0–255 input codebook, deterministic random codes, calibrated endpoint/split neighbors, both lower and upper half-step ties, saturation values, and deterministic random post-SiLU probes. C++ tests require exact code equality and report mismatch count, first index, input, expected code, and actual code. The Python source test regenerates both files in a temporary directory and requires byte equality.

## Build and test

With CMake and a C++17 compiler available:

```powershell
cmake -S cpp -B build\cpp-silu-kernel -DCMAKE_BUILD_TYPE=Release
cmake --build build\cpp-silu-kernel --config Release
ctest --test-dir build\cpp-silu-kernel -C Release --output-on-failure
```

The static kernel library has no PyTorch, ONNX Runtime, OpenVINO, or QNN dependency. A Python test scans the library include/source tree to preserve that boundary.

## Standalone microbenchmark

The benchmark requires every setting explicitly and validates the frozen fixture plus generated input once before timing. Timed calls use the uninstrumented scalar LUT loop. For example:

```powershell
build\cpp-silu-kernel\Release\silu_kernel_benchmark.exe `
  --fixture cpp\tests\data\act_call_0_golden.json `
  --size 1000000 `
  --warmup 20 `
  --iterations 100 `
  --seed 20260828 `
  --implementation scalar `
  --output-dir results\benchmarks\v1.0_silu_kernel_scalar
```

The output directory must remain under the ignored `results/benchmarks/v1.0_silu_kernel*` namespace and must be empty. The executable writes `benchmark_results.json`, `benchmark_results.csv`, and `benchmark_report.md`, reporting latency per invocation, effective elements/s, selected implementation, compiler/build data, host CPU description when exposed by the OS, input size, and seed.

Every report labels the result as a microbenchmark measurement for the standalone SiLU kernel only. It must not be compared with ORT/OpenVINO whole-model latency or used to claim a model speedup.

## Limitations and deferred work

- Only portable scalar `uint8` lookup is implemented. AVX2 is intentionally deferred; requesting it through the benchmark CLI fails clearly.
- The fixture input affine parameters are interface test data, not extracted full-model producer parameters.
- No full graph integration, ResNet18 accuracy evaluation, model conversion, or backend deployment was performed.
- LUT setup uses the documented float32 SiLU representation. The exact frozen fixture test is the portability guard for compiler/libm differences.
