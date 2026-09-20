#include "silu_cuda/bias_silu.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>

namespace {

constexpr unsigned int kThreadsPerBlock = 256;

bool checked_multiply(
    std::size_t left,
    std::size_t right,
    std::size_t* result
) noexcept {
    if (left == 0 || right == 0) {
        *result = 0;
        return true;
    }

    if (left > std::numeric_limits<std::size_t>::max() / right) {
        return false;
    }

    *result = left * right;
    return true;
}

__global__ void bias_silu_nchw_kernel(
    const float* input,
    const float* bias,
    float* output,
    std::size_t element_count,
    std::size_t channel_count,
    std::size_t spatial_size
) {
    const std::size_t first_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    const std::size_t grid_stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (std::size_t index = first_index;
         index < element_count;
         index += grid_stride) {
        const std::size_t channel =
            (index / spatial_size) % channel_count;
        const float biased_value = input[index] + bias[channel];
        output[index] = biased_value
            / (1.0F + expf(-biased_value));
    }
}

}  // namespace

namespace silu_cuda {

cudaError_t launch_bias_silu_nchw(
    const float* input,
    const float* bias,
    float* output,
    std::size_t batch_size,
    std::size_t channel_count,
    std::size_t spatial_size,
    cudaStream_t stream
) noexcept {
    if (batch_size == 0
        || channel_count == 0
        || spatial_size == 0) {
        return cudaSuccess;
    }

    if (input == nullptr || bias == nullptr || output == nullptr) {
        return cudaErrorInvalidValue;
    }

    std::size_t batch_channels = 0;
    std::size_t element_count = 0;
    if (!checked_multiply(
            batch_size,
            channel_count,
            &batch_channels
        )
        || !checked_multiply(
            batch_channels,
            spatial_size,
            &element_count
        )) {
        return cudaErrorInvalidValue;
    }

    const std::size_t block_count =
        ((element_count - 1) / kThreadsPerBlock) + 1;
    const std::size_t maximum_grid_x =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    const unsigned int launch_blocks = static_cast<unsigned int>(
        block_count > maximum_grid_x
        ? maximum_grid_x
        : block_count
    );

    bias_silu_nchw_kernel<<<
        launch_blocks,
        kThreadsPerBlock,
        0,
        stream
    >>>(
        input,
        bias,
        output,
        element_count,
        channel_count,
        spatial_size
    );

    return cudaGetLastError();
}

}  // namespace silu_cuda
