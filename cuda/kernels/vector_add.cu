#include "silu_cuda/vector_add.cuh"

#include <cuda_runtime.h>

#include <limits>

namespace {

constexpr unsigned int kThreadsPerBlock = 256;

__global__ void vector_add_kernel(
    const float* lhs,
    const float* rhs,
    float* output,
    std::size_t element_count
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index < element_count) {
        output[index] = lhs[index] + rhs[index];
    }
}

}  // namespace

namespace silu_cuda {

cudaError_t launch_vector_add(
    const float* lhs,
    const float* rhs,
    float* output,
    std::size_t element_count,
    cudaStream_t stream
) noexcept {
    if (element_count == 0) {
        return cudaSuccess;
    }

    if (lhs == nullptr || rhs == nullptr || output == nullptr) {
        return cudaErrorInvalidValue;
    }

    const std::size_t block_count =
        ((element_count - 1) / kThreadsPerBlock) + 1;

    if (block_count >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return cudaErrorInvalidConfiguration;
    }

    vector_add_kernel<<<
        static_cast<unsigned int>(block_count),
        kThreadsPerBlock,
        0,
        stream
    >>>(lhs, rhs, output, element_count);

    return cudaGetLastError();
}

}  // namespace silu_cuda
