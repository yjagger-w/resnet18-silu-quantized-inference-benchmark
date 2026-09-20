# CUDA kernel experiments

This directory contains the CUDA portion of the ResNet18-SiLU inference
benchmark. It is intentionally independent from the existing `cpp/` project
so the CPU and CUDA baselines can be configured and tested separately.

The first milestone establishes a minimal vector-add kernel and checked host
launch interface. The second adds a shared-memory sum-reduction baseline. The
third adds a hierarchical multi-pass reduction with reusable caller-provided
workspace. These milestones establish correctness and do not claim performance
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

## Reduction implementations

The atomic baseline loads two elements per thread, combines values within each
block through shared memory and `__syncthreads()`, and performs one global
`atomicAdd` per block.

The hierarchical implementation writes one partial sum per block and repeatedly
reduces those partial sums until a single result remains. It accepts reusable
caller-provided workspace, so later CUDA Event measurements can exclude memory
allocation from kernel timing. It avoids the atomic baseline's contention on
one global output address.

The correctness test compares both implementations against the same CPU
reference over boundary sizes and workloads up to 16M elements.

Performance measurements, CUDA Event timing, vectorized loads, FP16, and fused
Bias+SiLU kernels belong to later milestones.
