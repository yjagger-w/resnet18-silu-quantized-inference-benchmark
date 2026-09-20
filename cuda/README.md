# CUDA kernel experiments

This directory contains the CUDA portion of the ResNet18-SiLU inference
benchmark. It is intentionally independent from the existing `cpp/` project
so the CPU and CUDA baselines can be configured and tested separately.

The first milestone establishes a minimal vector-add kernel, a checked host
launch interface, and a correctness test. It does not claim performance
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
compilation remain valid. The runtime test performs its host-side contract
checks and then reports a CTest skip using exit code 77. With a T4 attached,
the same test executes the kernel and verifies every output element against a
CPU reference.

Performance measurements, CUDA Event timing, vectorized loads, FP16, and fused
Bias+SiLU kernels belong to later milestones.
