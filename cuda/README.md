# CUDA kernel experiments

This directory contains the CUDA portion of the ResNet18-SiLU inference
benchmark. It is intentionally independent from the existing `cpp/` project
so the CPU and CUDA baselines can be configured and tested separately.

The first milestone establishes a minimal vector-add kernel and checked host
launch interface. The second adds a shared-memory sum-reduction baseline. The
third adds a hierarchical multi-pass reduction with reusable caller-provided
workspace. The fourth adds a CUDA Event benchmark for reproducible comparison
of the two reduction implementations.

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
compilation remain valid. Each runtime correctness test performs its host-side
contract checks and reports a CTest skip using exit code 77. The benchmark CLI
contract tests do not require a GPU. With a T4 attached, the correctness tests
execute their kernels and compare device results with CPU references.

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

Vectorized loads, FP16, fused Bias+SiLU kernels, Nsight Compute profiling, and
end-to-end ResNet integration belong to later milestones.
