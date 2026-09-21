#pragma once

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>

namespace silu_cuda {

inline void check_cuda(
    cudaError_t status,
    const char* expression,
    const char* file,
    int line
) {
    if (status == cudaSuccess) {
        return;
    }

    std::ostringstream message;
    message
        << "CUDA call failed: " << expression
        << " at " << file << ':' << line
        << " (" << cudaGetErrorName(status)
        << ": " << cudaGetErrorString(status) << ')';
    throw std::runtime_error(message.str());
}

}  // namespace silu_cuda

#define SILU_CUDA_CHECK(expression) \
    ::silu_cuda::check_cuda((expression), #expression, __FILE__, __LINE__)
