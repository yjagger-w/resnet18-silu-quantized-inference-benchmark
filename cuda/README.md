# CUDA kernel experiments

This directory contains the CUDA portion of the ResNet18-SiLU inference
benchmark. It is intentionally independent from the existing `cpp/` project
so the CPU and CUDA baselines can be configured and tested separately.

The first milestone establishes a minimal vector-add kernel, a checked host
launch interface, and a correctness test. The second milestone adds a
shared-memory sum-reduction baseline. Neither milestone claims performance
results.

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
compilation remain valid. Each runtime test performs its host-side contract
checks and then reports a CTest skip using exit code 77. With a T4 attached,
the tests execute their kernels and compare the device results with CPU
references.

## Reduction baseline

The reduction kernel loads two input elements per thread, combines values
within each block through shared memory and `__syncthreads()`, and uses one
global `atomicAdd` per block. This is intentionally a learning baseline rather
than the final performance implementation. A later milestone will compare it
with hierarchical and two-pass reductions that avoid the global atomic
bottleneck.

Performance measurements, CUDA Event timing, vectorized loads, FP16, and fused
Bias+SiLU kernels belong to later milestones.
