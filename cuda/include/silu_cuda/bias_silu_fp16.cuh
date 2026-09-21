#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>

namespace silu_cuda {

enum class BiasSiluFp16KernelPath {
    kScalar,
    kHalf2,
};

BiasSiluFp16KernelPath select_bias_silu_nchw_fp16_kernel_path(
    const __half* input,
    const __half* output,
    std::size_t spatial_size
) noexcept;

cudaError_t launch_bias_silu_nchw_fp16_scalar(
    const __half* input,
    const __half* bias,
    __half* output,
    std::size_t batch_size,
    std::size_t channel_count,
    std::size_t spatial_size,
    cudaStream_t stream = nullptr
) noexcept;

cudaError_t launch_bias_silu_nchw_fp16_vectorized(
    const __half* input,
    const __half* bias,
    __half* output,
    std::size_t batch_size,
    std::size_t channel_count,
    std::size_t spatial_size,
    cudaStream_t stream = nullptr
) noexcept;

// Automatic FP16 entry point. It selects half2 for four-byte-aligned input and
// output pointers when every NCHW channel plane contains an even number of
// elements. Other layouts fall back to the scalar FP16 kernel.
cudaError_t launch_bias_silu_nchw_fp16(
    const __half* input,
    const __half* bias,
    __half* output,
    std::size_t batch_size,
    std::size_t channel_count,
    std::size_t spatial_size,
    cudaStream_t stream = nullptr
) noexcept;

}  // namespace silu_cuda
