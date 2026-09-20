#include "silu_cuda/reduction.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>

namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr unsigned int kElementsPerThread = 2;
constexpr std::size_t kElementsPerBlock =
    static_cast<std::size_t>(kThreadsPerBlock)
    * kElementsPerThread;

std::size_t block_count_for(std::size_t element_count) noexcept {
    if (element_count == 0) {
        return 0;
    }
    return ((element_count - 1) / kElementsPerBlock) + 1;
}

bool workspace_size_for(
    std::size_t element_count,
    std::size_t* workspace_size_bytes
) noexcept {
    const std::size_t first_pass_blocks =
        block_count_for(element_count);

    if (first_pass_blocks <= 1) {
        *workspace_size_bytes = 0;
        return true;
    }

    const std::size_t second_pass_blocks =
        block_count_for(first_pass_blocks);
    const std::size_t maximum_size =
        std::numeric_limits<std::size_t>::max();

    if (first_pass_blocks > maximum_size - second_pass_blocks) {
        return false;
    }

    const std::size_t workspace_elements =
        first_pass_blocks + second_pass_blocks;
    if (workspace_elements > maximum_size / sizeof(float)) {
        return false;
    }

    *workspace_size_bytes = workspace_elements * sizeof(float);
    return true;
}

__device__ float load_thread_sum(
    const float* input,
    std::size_t element_count
) {
    const std::size_t block_start =
        static_cast<std::size_t>(blockIdx.x)
        * blockDim.x
        * kElementsPerThread;
    const std::size_t first_index =
        block_start + threadIdx.x;
    const std::size_t second_index =
        first_index + blockDim.x;

    float thread_sum = 0.0F;
    if (first_index < element_count) {
        thread_sum += input[first_index];
    }
    if (second_index < element_count) {
        thread_sum += input[second_index];
    }
    return thread_sum;
}

__device__ float reduce_shared_block(
    float thread_sum,
    float* shared_values
) {
    const unsigned int thread_index = threadIdx.x;
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

    return shared_values[0];
}

__global__ void reduce_sum_atomic_kernel(
    const float* input,
    float* output,
    std::size_t element_count
) {
    extern __shared__ float shared_values[];
    const float block_sum = reduce_shared_block(
        load_thread_sum(input, element_count),
        shared_values
    );

    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum);
    }
}

__global__ void reduce_sum_partials_kernel(
    const float* input,
    float* partial_sums,
    std::size_t element_count
) {
    extern __shared__ float shared_values[];
    const float block_sum = reduce_shared_block(
        load_thread_sum(input, element_count),
        shared_values
    );

    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = block_sum;
    }
}

cudaError_t launch_partial_pass(
    const float* input,
    float* partial_sums,
    std::size_t element_count,
    std::size_t block_count,
    cudaStream_t stream
) noexcept {
    if (block_count >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return cudaErrorInvalidConfiguration;
    }

    reduce_sum_partials_kernel<<<
        static_cast<unsigned int>(block_count),
        kThreadsPerBlock,
        kThreadsPerBlock * sizeof(float),
        stream
    >>>(input, partial_sums, element_count);

    return cudaGetLastError();
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

    const std::size_t block_count =
        block_count_for(element_count);

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

std::size_t hierarchical_reduce_workspace_size_bytes(
    std::size_t element_count
) noexcept {
    std::size_t workspace_size_bytes = 0;
    if (!workspace_size_for(
            element_count,
            &workspace_size_bytes
        )) {
        return std::numeric_limits<std::size_t>::max();
    }
    return workspace_size_bytes;
}

cudaError_t launch_reduce_sum_hierarchical(
    const float* input,
    float* output,
    std::size_t element_count,
    void* workspace,
    std::size_t workspace_size_bytes,
    cudaStream_t stream
) noexcept {
    if (element_count == 0) {
        return cudaSuccess;
    }

    if (input == nullptr || output == nullptr) {
        return cudaErrorInvalidValue;
    }

    std::size_t required_workspace_size = 0;
    if (!workspace_size_for(
            element_count,
            &required_workspace_size
        )) {
        return cudaErrorInvalidValue;
    }

    if (required_workspace_size > 0
        && (workspace == nullptr
            || workspace_size_bytes < required_workspace_size)) {
        return cudaErrorInvalidValue;
    }

    const std::size_t first_pass_blocks =
        block_count_for(element_count);
    if (first_pass_blocks <= 1) {
        return launch_partial_pass(
            input,
            output,
            element_count,
            first_pass_blocks,
            stream
        );
    }

    float* const first_workspace =
        static_cast<float*>(workspace);
    float* const second_workspace =
        first_workspace + first_pass_blocks;

    const float* current_input = input;
    std::size_t current_count = element_count;
    bool write_first_workspace = true;

    while (current_count > 1) {
        const std::size_t block_count =
            block_count_for(current_count);
        float* pass_output = nullptr;

        if (block_count == 1) {
            pass_output = output;
        } else if (write_first_workspace) {
            pass_output = first_workspace;
        } else {
            pass_output = second_workspace;
        }

        const cudaError_t launch_status = launch_partial_pass(
            current_input,
            pass_output,
            current_count,
            block_count,
            stream
        );
        if (launch_status != cudaSuccess) {
            return launch_status;
        }

        current_input = pass_output;
        current_count = block_count;
        write_first_workspace = !write_first_workspace;
    }

    return cudaSuccess;
}

}  // namespace silu_cuda
