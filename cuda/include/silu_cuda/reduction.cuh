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

}  // namespace silu_cuda
