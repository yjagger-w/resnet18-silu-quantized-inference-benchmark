#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace silu_cuda {

enum class BiasSiluKernelPath {
    kScalar,
    kFloat4,
};

// T4/CUDA 12.4 repeated benchmarks showed that float4 becomes beneficial at
// the 64x32x32 stem tensor (65,536 elements), while smaller ResNet18 stage
// tensors are launch-latency dominated. Explicit vectorized launches remain
// available independently of this automatic-dispatch threshold.
inline constexpr std::size_t kBiasSiluFloat4MinimumElements = 65536;

BiasSiluKernelPath select_bias_silu_nchw_kernel_path(
    const float* input,
    const float* output,
    std::size_t spatial_size
) noexcept;

BiasSiluKernelPath select_bias_silu_nchw_auto_path(
    const float* input,
    const float* output,
    std::size_t batch_size,
    std::size_t channel_count,
    std::size_t spatial_size
) noexcept;

cudaError_t launch_bias_silu_nchw_scalar(
    const float* input,
    const float* bias,
    float* output,
    std::size_t batch_size,
    std::size_t channel_count,
    std::size_t spatial_size,
    cudaStream_t stream = nullptr
) noexcept;

cudaError_t launch_bias_silu_nchw_vectorized(
    const float* input,
    const float* bias,
    float* output,
    std::size_t batch_size,
    std::size_t channel_count,
    std::size_t spatial_size,
    cudaStream_t stream = nullptr
) noexcept;

// Backward-compatible adaptive entry point. It selects float4 only when the
// input/output pointers are 16-byte aligned, every channel plane contains a
// multiple of four elements, and the tensor meets the benchmark-derived
// minimum element count. Other cases use the scalar kernel.
cudaError_t launch_bias_silu_nchw(
    const float* input,
    const float* bias,
    float* output,
    std::size_t batch_size,
    std::size_t channel_count,
    std::size_t spatial_size,
    cudaStream_t stream = nullptr
) noexcept;

}  // namespace silu_cuda
