#include "silu_cuda/bias_silu_fp16.cuh"

#include <cuda_fp16.h>
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
    const __half* input,
    const __half* bias,
    __half* output,
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

__device__ __forceinline__ float silu_float(float value) {
    return value / (1.0F + expf(-value));
}

__global__ void bias_silu_nchw_fp16_scalar_kernel(
    const __half* input,
    const __half* bias,
    __half* output,
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
        const float value =
            __half2float(input[index])
            + __half2float(bias[channel]);
        output[index] = __float2half_rn(silu_float(value));
    }
}

__global__ void bias_silu_nchw_fp16_half2_kernel(
    const __half2* input,
    const __half* bias,
    __half2* output,
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
        const float channel_bias = __half2float(bias[channel]);
        float2 values = __half22float2(input[vector_index]);
        values.x = silu_float(values.x + channel_bias);
        values.y = silu_float(values.y + channel_bias);
        output[vector_index] = __floats2half2_rn(
            values.x,
            values.y
        );
    }
}

cudaError_t launch_scalar_validated(
    const __half* input,
    const __half* bias,
    __half* output,
    std::size_t element_count,
    std::size_t channel_count,
    std::size_t spatial_size,
    cudaStream_t stream
) noexcept {
    bias_silu_nchw_fp16_scalar_kernel<<<
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

cudaError_t launch_half2_validated(
    const __half* input,
    const __half* bias,
    __half* output,
    std::size_t element_count,
    std::size_t channel_count,
    std::size_t spatial_size,
    cudaStream_t stream
) noexcept {
    const std::size_t vector_count = element_count / 2;
    const std::size_t vectors_per_channel = spatial_size / 2;
    bias_silu_nchw_fp16_half2_kernel<<<
        launch_block_count(vector_count),
        kThreadsPerBlock,
        0,
        stream
    >>>(
        reinterpret_cast<const __half2*>(input),
        bias,
        reinterpret_cast<__half2*>(output),
        vector_count,
        channel_count,
        vectors_per_channel
    );
    return cudaGetLastError();
}

}  // namespace

namespace silu_cuda {

BiasSiluFp16KernelPath select_bias_silu_nchw_fp16_kernel_path(
    const __half* input,
    const __half* output,
    std::size_t spatial_size
) noexcept {
    if (input == nullptr
        || output == nullptr
        || spatial_size == 0
        || spatial_size % 2 != 0) {
        return BiasSiluFp16KernelPath::kScalar;
    }

    constexpr std::uintptr_t kHalf2Alignment = alignof(__half2);
    const bool input_is_aligned =
        reinterpret_cast<std::uintptr_t>(input) % kHalf2Alignment == 0;
    const bool output_is_aligned =
        reinterpret_cast<std::uintptr_t>(output) % kHalf2Alignment == 0;
    return input_is_aligned && output_is_aligned
        ? BiasSiluFp16KernelPath::kHalf2
        : BiasSiluFp16KernelPath::kScalar;
}

cudaError_t launch_bias_silu_nchw_fp16_scalar(
    const __half* input,
    const __half* bias,
    __half* output,
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

cudaError_t launch_bias_silu_nchw_fp16_vectorized(
    const __half* input,
    const __half* bias,
    __half* output,
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

    if (select_bias_silu_nchw_fp16_kernel_path(
            input,
            output,
            spatial_size
        ) != BiasSiluFp16KernelPath::kHalf2) {
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

    return launch_half2_validated(
        input,
        bias,
        output,
        element_count,
        channel_count,
        spatial_size,
        stream
    );
}

cudaError_t launch_bias_silu_nchw_fp16(
    const __half* input,
    const __half* bias,
    __half* output,
    std::size_t batch_size,
    std::size_t channel_count,
    std::size_t spatial_size,
    cudaStream_t stream
) noexcept {
    return launch_bias_silu_nchw_fp16_vectorized(
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
