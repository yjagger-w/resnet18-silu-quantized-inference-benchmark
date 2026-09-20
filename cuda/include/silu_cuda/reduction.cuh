#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace silu_cuda {

cudaError_t launch_reduce_sum_atomic(
    const float* input,
    float* output,
    std::size_t element_count,
    cudaStream_t stream = nullptr
) noexcept;

std::size_t hierarchical_reduce_workspace_size_bytes(
    std::size_t element_count
) noexcept;

cudaError_t launch_reduce_sum_hierarchical(
    const float* input,
    float* output,
    std::size_t element_count,
    void* workspace,
    std::size_t workspace_size_bytes,
    cudaStream_t stream = nullptr
) noexcept;

}  // namespace silu_cuda
