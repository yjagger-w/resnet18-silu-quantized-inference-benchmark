#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace silu_cuda {

enum class BiasSiluKernelPath {
    kScalar,
    kFloat4,
};

BiasSiluKernelPath select_bias_silu_nchw_kernel_path(
    const float* input,
    const float* output,
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

// Backward-compatible auto-dispatch entry point. It selects the float4 path
// only when the input/output pointers are 16-byte aligned and each NCHW
// channel plane contains a multiple of four elements.
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
