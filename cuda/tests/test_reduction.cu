#include "silu_cuda/cuda_check.h"
#include "silu_cuda/reduction.cuh"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kCTestSkipReturnCode = 77;
constexpr float kTolerance = 1.0e-4F;

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t element_count) {
        SILU_CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&data_),
            element_count * sizeof(float)
        ));
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* get() noexcept {
        return data_;
    }

private:
    float* data_ = nullptr;
};

bool host_contract_is_valid() {
    if (silu_cuda::launch_reduce_sum_atomic(
            nullptr,
            nullptr,
            0
        ) != cudaSuccess) {
        std::cerr << "Zero-length reduction must be a successful no-op.\n";
        return false;
    }

    if (silu_cuda::launch_reduce_sum_atomic(
            nullptr,
            nullptr,
            1
        ) != cudaErrorInvalidValue) {
        std::cerr << "Non-empty reduction must reject null pointers.\n";
        return false;
    }

    return true;
}

bool run_reduction_case(std::size_t element_count) {
    std::vector<float> input(element_count);

    double expected_double = 0.0;
    for (std::size_t index = 0; index < element_count; ++index) {
        const float value =
            static_cast<float>((index % 31) + 1) * 0.03125F;
        input[index] = value;
        expected_double += static_cast<double>(value);
    }
    const float expected = static_cast<float>(expected_double);

    DeviceBuffer device_input(element_count);
    DeviceBuffer device_output(1);

    SILU_CUDA_CHECK(cudaMemcpy(
        device_input.get(),
        input.data(),
        element_count * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    SILU_CUDA_CHECK(silu_cuda::launch_reduce_sum_atomic(
        device_input.get(),
        device_output.get(),
        element_count
    ));
    SILU_CUDA_CHECK(cudaDeviceSynchronize());

    float actual = 0.0F;
    SILU_CUDA_CHECK(cudaMemcpy(
        &actual,
        device_output.get(),
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    const float absolute_error = std::abs(actual - expected);
    if (absolute_error > kTolerance) {
        std::cerr
            << "Reduction mismatch for element_count=" << element_count
            << ": expected=" << expected
            << ", actual=" << actual
            << ", abs_error=" << absolute_error
            << '\n';
        return false;
    }

    return true;
}

}  // namespace

int main() {
    if (!host_contract_is_valid()) {
        return 1;
    }

    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);

    if (device_status == cudaErrorNoDevice || device_count == 0) {
        std::cout
            << "SKIP: CUDA Toolkit is available, but no GPU is attached.\n";
        return kCTestSkipReturnCode;
    }

    if (device_status != cudaSuccess) {
        std::cerr
            << "cudaGetDeviceCount failed: "
            << cudaGetErrorName(device_status)
            << ": "
            << cudaGetErrorString(device_status)
            << '\n';
        return 1;
    }

    try {
        constexpr std::array<std::size_t, 8> element_counts = {
            1,
            2,
            255,
            256,
            257,
            511,
            512,
            65537,
        };

        for (const std::size_t element_count : element_counts) {
            if (!run_reduction_case(element_count)) {
                return 1;
            }
        }

        std::cout
            << "PASS: atomic shared-memory reduction matched the CPU "
            << "reference for "
            << element_counts.size()
            << " boundary-focused cases.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
