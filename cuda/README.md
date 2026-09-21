# CUDA kernel experiments

This directory contains the CUDA portion of the ResNet18-SiLU inference
benchmark. It is intentionally independent from the existing `cpp/` project
so the CPU and CUDA baselines can be configured and tested separately.

The milestones establish a vector-add scaffold, atomic and hierarchical
reductions, reproducible CUDA Event benchmarking, fused FP32 NCHW
Bias+SiLU scalar and float4 kernels, and FP16 scalar and half2 correctness
baselines. Performance claims are made
only from recorded GPU benchmark artifacts, not from CTest process duration.

## Requirements

- CMake 3.22 or newer
- CUDA Toolkit 12.x
- A C++17-compatible host compiler
- NVIDIA T4 for runtime validation and profiling

The default CUDA architecture is `75`, which targets the NVIDIA T4. Override
it explicitly when building for another GPU.

## Configure and build

```bash
cmake \
  -S cuda \
  -B build/cuda-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75

cmake \
  --build build/cuda-release \
  --parallel "$(nproc)"
```

## Test

```bash
ctest \
  --test-dir build/cuda-release \
  --output-on-failure
```

When the CUDA Toolkit is installed but no GPU is attached, configuration and
compilation remain valid. Runtime correctness tests perform host-side contract
checks and report a CTest skip using exit code 77. Benchmark CLI contract tests
do not require a GPU. With a T4 attached, device tests compare kernel results
with CPU references.

## Reduction implementations

The atomic baseline loads two elements per thread, combines values within each
block through shared memory and `__syncthreads()`, and performs one global
`atomicAdd` per block.

The hierarchical implementation writes one partial sum per block and repeatedly
reduces those partial sums until a single result remains. It accepts reusable
caller-provided workspace and avoids contention on one global output address.

The correctness test compares both implementations against the same CPU
reference over boundary sizes and workloads up to 16M elements.

## Reduction benchmark

The benchmark preallocates device memory and hierarchical workspace, uploads
the input once, performs warm-up launches, and measures each iteration with
CUDA Events on the default stream. Allocation, workspace creation, and the
host-to-device input copy are outside the timed interval. The atomic
implementation's required output clear remains inside its timed operation.

Run the default 16M-element comparison:

```bash
mkdir -p out/cuda/v1.7

./build/cuda-release/cuda_reduction_benchmark \
  --elements 16777216 \
  --warmup 20 \
  --iterations 200 \
  --implementation both \
  | tee out/cuda/v1.7/reduction_t4_16m.json
```

The JSON report records device, driver, CUDA runtime, compiler versions, the
benchmark protocol, mean/P50/P90/P95/P99/min/max latency, validated output, and
effective input bandwidth. Effective input bandwidth is input bytes divided by
mean elapsed time; it is not a claim about total DRAM traffic.

Use `--implementation atomic` or `--implementation hierarchical` for an
isolated run. Use `--help` to inspect the complete CLI.

## Fused Bias+SiLU baseline

`launch_bias_silu_nchw` computes the following FP32 operation in one kernel:

```text
output[n, c, h, w] =
    silu(input[n, c, h, w] + bias[c])
```

The interface accepts flattened contiguous NCHW storage, a channel bias vector,
batch size, channel count, and flattened spatial size. It supports in-place
execution when `output == input`.

The scalar entry point preserves the original element-wise baseline. The
explicit vectorized entry point processes four contiguous spatial values per
thread through aligned `float4` loads and stores whenever layout permits, so
benchmarks and callers can still force that implementation.

The backward-compatible `launch_bias_silu_nchw` entry point uses adaptive
dispatch. It selects `float4` only when both data pointers are 16-byte aligned,
every channel plane contains a multiple of four elements, and the tensor has at
least 65,536 elements. Smaller or ineligible tensors use the scalar kernel.
The threshold comes from five repeated Tesla T4 / CUDA 12.4 benchmark runs:
the 64x32x32 stem consistently benefited from float4, while the smaller stage
outputs were tied or faster with scalar. It is a T4-specific policy rather than
a universal cross-GPU tuning claim.

The correctness test compares scalar, explicit float4, and adaptive results
with one CPU reference, covers both sides of the dispatch threshold, exercises
non-multiple-of-four shapes, forces a misaligned-pointer fallback, checks
non-finite outputs, and covers in-place execution.

## FP16 Bias+SiLU correctness baseline

The independent FP16 API mirrors the FP32 NCHW contract with half-precision
input, channel bias, and output storage. The scalar kernel processes one FP16
element per thread. The vectorized kernel uses aligned `half2` loads and stores
for two contiguous spatial values per thread, converts the pair to FP32 for the
Bias+SiLU calculation, and rounds the outputs back to FP16.

`launch_bias_silu_nchw_fp16` uses adaptive dispatch. It selects half2 only
when both data pointers satisfy four-byte alignment, each channel plane has an
even spatial size, and the tensor contains at least 65,536 elements. Smaller or
ineligible tensors use scalar. Three repeated Tesla T4 / CUDA 12.4 runs showed
about 1.18x stem speedup, a stage2 tie, and no reliable benefit for stage3 or
stage4. Static resource inspection reported 24 registers for scalar and 26 for
half2, with zero stack, shared, and local memory for both kernels. Explicit
scalar and vectorized entry points remain available, all paths support in-place
execution, and the threshold is T4-specific rather than universal.

The FP16 test covers both sides of the automatic-dispatch threshold, aligned
half2, odd spatial sizes, deliberately misaligned pointers, shape overflow,
zero-sized no-ops, in-place execution, finite outputs, and comparison against
an FP32 CPU reference.

## FP16 Bias+SiLU benchmark

The separate FP16 benchmark compares scalar, explicit half2, and automatic
dispatch for the same four CIFAR-10 ResNet18 stage-output shapes and both
out-of-place and in-place execution. It uses FP16 input, bias, and output
storage while the kernels evaluate Bias+SiLU in FP32 before rounding back to
FP16. Results use the same interleaved CUDA Event protocol as the FP32
benchmark, distinguish layout and threshold fallbacks, and include numerical
validation against an FP32 CPU reference.

Run one complete comparison:

```bash
mkdir -p out/cuda/v1.7/fp16_bias_silu

./build/cuda-release/cuda_bias_silu_fp16_benchmark \
  --warmup 50 \
  --iterations 1000 \
  --implementation all \
  --mode both \
  > out/cuda/v1.7/fp16_bias_silu/run_1.json

python3 -m json.tool \
  out/cuda/v1.7/fp16_bias_silu/run_1.json \
  >/dev/null
```

Use `--implementation scalar|half2|auto` or
`--mode out-of-place|in-place` for isolated runs. For the final T4 evidence,
run the command three times and report the median P50 values; keep only the
reviewed final JSON and Markdown artifacts.

## Bias+SiLU benchmark

The Bias+SiLU benchmark covers the four CIFAR-10 ResNet18 stage-output shapes
from 64x32x32 through 512x4x4. It compares explicit scalar, explicit float4,
and automatic dispatch in both out-of-place and in-place modes. Each result
records mean/P50/P90/P95/P99/min/max latency, the selected kernel path,
effective tensor bandwidth, maximum absolute error, and speedup relative to
the matching scalar mode. Automatic results distinguish a layout fallback
from the benchmark-derived small-tensor threshold fallback.

For in-place measurements, an untimed device-to-device copy restores the input
before every launch. CUDA Events therefore measure only the Bias+SiLU kernel,
while all implementations receive identical input values.

Within each shape and mode, warm-up and measured launches are interleaved using
a deterministic round-robin schedule. The starting implementation rotates
every round, so scalar, float4, and automatic dispatch collect samples under
comparable GPU clock and thermal conditions instead of running as three long
sequential blocks. The JSON protocol records this execution order, and the
configured iteration counts apply independently to every selected
implementation.

Run the complete comparison:

```bash
mkdir -p out/cuda/v1.7

./build/cuda-release/cuda_bias_silu_benchmark \
  --warmup 20 \
  --iterations 200 \
  --implementation all \
  --mode both \
  | tee out/cuda/v1.7/bias_silu_t4_resnet18.json
```

Use `--implementation scalar|float4|auto` or
`--mode out-of-place|in-place` for isolated runs. A speedup is emitted only
when the matching scalar result is part of the same run.

## Repeated benchmark aggregation

The standard-library-only aggregation tool validates that all repeated reports
share the same schema, device, build, protocol, result keys, and selected kernel
paths. It rejects non-finite metrics and numerical errors above the configured
limit, records every source file SHA256, and emits deterministic JSON and
Markdown without timestamps or absolute paths.

Aggregate five interleaved runs:

```bash
python3 cuda/tools/aggregate_bias_silu_results.py \
  --input-dir out/cuda/v1.7/interleaved_bias_silu \
  --output-json out/cuda/v1.7/interleaved_bias_silu/aggregate.json \
  --output-markdown out/cuda/v1.7/interleaved_bias_silu/aggregate_summary.md \
  --expected-runs 5
```

The comparison table reports the median of each run's P50 latency, automatic
dispatch relative to scalar, and automatic dispatch relative to the faster
explicit implementation. Generated files remain under the ignored `out/`
directory until a reviewed formal result is intentionally promoted.

Profiler evidence remains optional when the host permits GPU performance
counters. End-to-end ResNet integration belongs to a later milestone.