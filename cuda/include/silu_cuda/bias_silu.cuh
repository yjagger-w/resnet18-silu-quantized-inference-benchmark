#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace silu_cuda {

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
