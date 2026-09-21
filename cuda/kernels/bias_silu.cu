#include "silu_cuda/bias_silu.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
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

cudaError_t validate_arguments(
    const float* input,
    const float* bias,
    float* output,
    std::size_t batch_size,
    std::size_t channel_count,
    std::size_t spatial_size,
    std::size_t* element_count
) noexcept {
    if (batch_size == 0
        || channel_count == 0
        || spatial_size == 0) {
        *element_count = 0;
        return cudaSuccess;
    }

    if (input == nullptr || bias == nullptr || output == nullptr) {
        return cudaErrorInvalidValue;
    }

    std::size_t batch_channels = 0;
    if (!checked_multiply(
            batch_size,
            channel_count,
            &batch_channels
        )
        || !checked_multiply(
            batch_channels,
            spatial_size,
            element_count
        )) {
        return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}

unsigned int launch_block_count(std::size_t work_items) noexcept {
    const std::size_t block_count =
        ((work_items - 1) / kThreadsPerBlock) + 1;
    const std::size_t maximum_grid_x =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    return static_cast<unsigned int>(
        block_count > maximum_grid_x
        ? maximum_grid_x
        : block_count
    );
}

__device__ __forceinline__ float silu(float value) {
    return value / (1.0F + expf(-value));
}

__global__ void bias_silu_nchw_scalar_kernel(
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
        output[index] = silu(input[index] + bias[channel]);
    }
}

__global__ void bias_silu_nchw_float4_kernel(
    const float4* input,
    const float* bias,
    float4* output,
    std::size_t vector_count,
    std::size_t channel_count,
    std::size_t vectors_per_channel
) {
    const std::size_t first_vector =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    const std::size_t grid_stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (std::size_t vector_index = first_vector;
         vector_index < vector_count;
         vector_index += grid_stride) {
        const std::size_t channel =
            (vector_index / vectors_per_channel) % channel_count;
        const float channel_bias = bias[channel];
        float4 values = input[vector_index];
        values.x = silu(values.x + channel_bias);
        values.y = silu(values.y + channel_bias);
        values.z = silu(values.z + channel_bias);
        values.w = silu(values.w + channel_bias);
        output[vector_index] = values;
    }
}

cudaError_t launch_scalar_validated(
    const float* input,
    const float* bias,
    float* output,
    std::size_t element_count,
    std::size_t channel_count,
    std::size_t spatial_size,
    cudaStream_t stream
) noexcept {
    bias_silu_nchw_scalar_kernel<<<
        launch_block_count(element_count),
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

}  // namespace

namespace silu_cuda {

BiasSiluKernelPath select_bias_silu_nchw_kernel_path(
    const float* input,
    const float* output,
    std::size_t spatial_size
) noexcept {
    if (input == nullptr
        || output == nullptr
        || spatial_size == 0
        || spatial_size % 4 != 0) {
        return BiasSiluKernelPath::kScalar;
    }

    constexpr std::uintptr_t kFloat4Alignment = alignof(float4);
    const bool input_is_aligned =
        reinterpret_cast<std::uintptr_t>(input) % kFloat4Alignment == 0;
    const bool output_is_aligned =
        reinterpret_cast<std::uintptr_t>(output) % kFloat4Alignment == 0;

    return input_is_aligned && output_is_aligned
        ? BiasSiluKernelPath::kFloat4
        : BiasSiluKernelPath::kScalar;
}

cudaError_t launch_bias_silu_nchw_scalar(
    const float* input,
    const float* bias,
    float* output,
    std::size_t batch_size,
    std::size_t channel_count,
    std::size_t spatial_size,
    cudaStream_t stream
) noexcept {
    std::size_t element_count = 0;
    const cudaError_t validation_status = validate_arguments(
        input,
        bias,
        output,
        batch_size,
        channel_count,
        spatial_size,
        &element_count
    );
    if (validation_status != cudaSuccess || element_count == 0) {
        return validation_status;
    }

    return launch_scalar_validated(
        input,
        bias,
        output,
        element_count,
        channel_count,
        spatial_size,
        stream
    );
}

cudaError_t launch_bias_silu_nchw_vectorized(
    const float* input,
    const float* bias,
    float* output,
    std::size_t batch_size,
    std::size_t channel_count,
    std::size_t spatial_size,
    cudaStream_t stream
) noexcept {
    std::size_t element_count = 0;
    const cudaError_t validation_status = validate_arguments(
        input,
        bias,
        output,
        batch_size,
        channel_count,
        spatial_size,
        &element_count
    );
    if (validation_status != cudaSuccess || element_count == 0) {
        return validation_status;
    }

    if (select_bias_silu_nchw_kernel_path(
            input,
            output,
            spatial_size
        ) != BiasSiluKernelPath::kFloat4) {
        return launch_scalar_validated(
            input,
            bias,
            output,
            element_count,
            channel_count,
            spatial_size,
            stream
        );
    }

    const std::size_t vector_count = element_count / 4;
    const std::size_t vectors_per_channel = spatial_size / 4;
    bias_silu_nchw_float4_kernel<<<
        launch_block_count(vector_count),
        kThreadsPerBlock,
        0,
        stream
    >>>(
        reinterpret_cast<const float4*>(input),
        bias,
        reinterpret_cast<float4*>(output),
        vector_count,
        channel_count,
        vectors_per_channel
    );
    return cudaGetLastError();
}

cudaError_t launch_bias_silu_nchw(
    const float* input,
    const float* bias,
    float* output,
    std::size_t batch_size,
    std::size_t channel_count,
    std::size_t spatial_size,
    cudaStream_t stream
) noexcept {
    return launch_bias_silu_nchw_vectorized(
        input,
        bias,
        output,
        batch_size,
        channel_count,
        spatial_size,
        stream
    );
}

}  // namespace silu_cuda
