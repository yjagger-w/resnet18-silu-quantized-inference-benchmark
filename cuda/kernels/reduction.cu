#include "silu_cuda/reduction.cuh"

#include <cuda_runtime.h>

#include <limits>

namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr unsigned int kElementsPerThread = 2;

__global__ void reduce_sum_atomic_kernel(
    const float* input,
    float* output,
    std::size_t element_count
) {
    extern __shared__ float shared_values[];

    const unsigned int thread_index = threadIdx.x;
    const std::size_t block_start =
        static_cast<std::size_t>(blockIdx.x)
        * blockDim.x
        * kElementsPerThread;
    const std::size_t first_index = block_start + thread_index;
    const std::size_t second_index = first_index + blockDim.x;

    float thread_sum = 0.0F;
    if (first_index < element_count) {
        thread_sum += input[first_index];
    }
    if (second_index < element_count) {
        thread_sum += input[second_index];
    }

    shared_values[thread_index] = thread_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2) {
        if (thread_index < stride) {
            shared_values[thread_index] +=
                shared_values[thread_index + stride];
        }
        __syncthreads();
    }

    if (thread_index == 0) {
        atomicAdd(output, shared_values[0]);
    }
}

}  // namespace

namespace silu_cuda {

cudaError_t launch_reduce_sum_atomic(
    const float* input,
    float* output,
    std::size_t element_count,
    cudaStream_t stream
) noexcept {
    if (element_count == 0) {
        return cudaSuccess;
    }

    if (input == nullptr || output == nullptr) {
        return cudaErrorInvalidValue;
    }

    const std::size_t elements_per_block =
        static_cast<std::size_t>(kThreadsPerBlock)
        * kElementsPerThread;
    const std::size_t block_count =
        ((element_count - 1) / elements_per_block) + 1;

    if (block_count >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return cudaErrorInvalidConfiguration;
    }

    const cudaError_t clear_status =
        cudaMemsetAsync(output, 0, sizeof(float), stream);
    if (clear_status != cudaSuccess) {
        return clear_status;
    }

    reduce_sum_atomic_kernel<<<
        static_cast<unsigned int>(block_count),
        kThreadsPerBlock,
        kThreadsPerBlock * sizeof(float),
        stream
    >>>(input, output, element_count);

    return cudaGetLastError();
}

}  // namespace silu_cuda
