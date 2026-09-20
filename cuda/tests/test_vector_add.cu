#include "silu_cuda/cuda_check.h"
#include "silu_cuda/vector_add.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kCTestSkipReturnCode = 77;
constexpr std::size_t kElementCount = 4096;
constexpr float kTolerance = 1.0e-6F;

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
    if (silu_cuda::launch_vector_add(
            nullptr,
            nullptr,
            nullptr,
            0
        ) != cudaSuccess) {
        std::cerr << "Zero-length launch must be a successful no-op.\n";
        return false;
    }

    if (silu_cuda::launch_vector_add(
            nullptr,
            nullptr,
            nullptr,
            1
        ) != cudaErrorInvalidValue) {
        std::cerr << "Non-empty launch must reject null pointers.\n";
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
        std::vector<float> lhs(kElementCount);
        std::vector<float> rhs(kElementCount);
        std::vector<float> expected(kElementCount);
        std::vector<float> actual(kElementCount);

        for (std::size_t index = 0; index < kElementCount; ++index) {
            const int signed_index = static_cast<int>(index);
            lhs[index] =
                static_cast<float>((signed_index % 97) - 48) * 0.125F;
            rhs[index] =
                static_cast<float>(((signed_index * 17) % 89) - 44)
                * 0.0625F;
            expected[index] = lhs[index] + rhs[index];
        }

        DeviceBuffer device_lhs(kElementCount);
        DeviceBuffer device_rhs(kElementCount);
        DeviceBuffer device_output(kElementCount);

        SILU_CUDA_CHECK(cudaMemcpy(
            device_lhs.get(),
            lhs.data(),
            kElementCount * sizeof(float),
            cudaMemcpyHostToDevice
        ));
        SILU_CUDA_CHECK(cudaMemcpy(
            device_rhs.get(),
            rhs.data(),
            kElementCount * sizeof(float),
            cudaMemcpyHostToDevice
        ));

        SILU_CUDA_CHECK(silu_cuda::launch_vector_add(
            device_lhs.get(),
            device_rhs.get(),
            device_output.get(),
            kElementCount
        ));
        SILU_CUDA_CHECK(cudaDeviceSynchronize());

        SILU_CUDA_CHECK(cudaMemcpy(
            actual.data(),
            device_output.get(),
            kElementCount * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        for (std::size_t index = 0; index < kElementCount; ++index) {
            const float absolute_error =
                std::abs(actual[index] - expected[index]);
            if (absolute_error > kTolerance) {
                std::cerr
                    << "Mismatch at index " << index
                    << ": expected=" << expected[index]
                    << ", actual=" << actual[index]
                    << ", abs_error=" << absolute_error
                    << '\n';
                return 1;
            }
        }

        std::cout
            << "PASS: vector_add matched the CPU reference for "
            << kElementCount
            << " elements.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
