#include "silu_cuda/cuda_check.h"
#include "silu_cuda/reduction.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kCTestSkipReturnCode = 77;
constexpr float kInputValue = 0.03125F;
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
        std::cerr << "Atomic zero-length reduction must be a no-op.\n";
        return false;
    }

    if (silu_cuda::launch_reduce_sum_atomic(
            nullptr,
            nullptr,
            1
        ) != cudaErrorInvalidValue) {
        std::cerr << "Atomic reduction must reject null pointers.\n";
        return false;
    }

    if (silu_cuda::launch_reduce_sum_hierarchical(
            nullptr,
            nullptr,
            0,
            nullptr,
            0
        ) != cudaSuccess) {
        std::cerr << "Hierarchical zero-length reduction must be a no-op.\n";
        return false;
    }

    if (silu_cuda::launch_reduce_sum_hierarchical(
            nullptr,
            nullptr,
            1,
            nullptr,
            0
        ) != cudaErrorInvalidValue) {
        std::cerr << "Hierarchical reduction must reject null pointers.\n";
        return false;
    }

    if (silu_cuda::hierarchical_reduce_workspace_size_bytes(512) != 0) {
        std::cerr << "A single-block reduction must not require workspace.\n";
        return false;
    }

    constexpr std::size_t expected_workspace_bytes =
        3 * sizeof(float);
    const std::size_t required_workspace_bytes =
        silu_cuda::hierarchical_reduce_workspace_size_bytes(513);

    if (required_workspace_bytes != expected_workspace_bytes) {
        std::cerr << "Unexpected hierarchical workspace size.\n";
        return false;
    }

    float dummy = 0.0F;
    if (silu_cuda::launch_reduce_sum_hierarchical(
            &dummy,
            &dummy,
            513,
            nullptr,
            0
        ) != cudaErrorInvalidValue) {
        std::cerr << "Missing hierarchical workspace must be rejected.\n";
        return false;
    }

    if (silu_cuda::launch_reduce_sum_hierarchical(
            &dummy,
            &dummy,
            513,
            &dummy,
            required_workspace_bytes - 1
        ) != cudaErrorInvalidValue) {
        std::cerr << "Undersized hierarchical workspace must be rejected.\n";
        return false;
    }

    return true;
}

bool output_matches(
    const char* implementation,
    std::size_t element_count,
    float expected,
    float actual
) {
    const float absolute_error = std::abs(actual - expected);
    if (absolute_error <= kTolerance) {
        return true;
    }

    std::cerr
        << implementation
        << " mismatch for element_count=" << element_count
        << ": expected=" << expected
        << ", actual=" << actual
        << ", abs_error=" << absolute_error
        << '\n';
    return false;
}

bool run_reduction_case(std::size_t element_count) {
    const std::vector<float> input(element_count, kInputValue);
    const float expected = static_cast<float>(
        static_cast<double>(element_count)
        * static_cast<double>(kInputValue)
    );

    DeviceBuffer device_input(element_count);
    DeviceBuffer atomic_output(1);
    DeviceBuffer hierarchical_output(1);

    const std::size_t workspace_size_bytes =
        silu_cuda::hierarchical_reduce_workspace_size_bytes(
            element_count
        );
    const std::size_t workspace_elements = std::max<std::size_t>(
        1,
        (workspace_size_bytes + sizeof(float) - 1)
            / sizeof(float)
    );
    DeviceBuffer workspace(workspace_elements);

    SILU_CUDA_CHECK(cudaMemcpy(
        device_input.get(),
        input.data(),
        element_count * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    SILU_CUDA_CHECK(silu_cuda::launch_reduce_sum_atomic(
        device_input.get(),
        atomic_output.get(),
        element_count
    ));
    SILU_CUDA_CHECK(silu_cuda::launch_reduce_sum_hierarchical(
        device_input.get(),
        hierarchical_output.get(),
        element_count,
        workspace.get(),
        workspace_size_bytes
    ));
    SILU_CUDA_CHECK(cudaDeviceSynchronize());

    float atomic_actual = 0.0F;
    float hierarchical_actual = 0.0F;
    SILU_CUDA_CHECK(cudaMemcpy(
        &atomic_actual,
        atomic_output.get(),
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    SILU_CUDA_CHECK(cudaMemcpy(
        &hierarchical_actual,
        hierarchical_output.get(),
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    return output_matches(
               "Atomic reduction",
               element_count,
               expected,
               atomic_actual
           )
        && output_matches(
               "Hierarchical reduction",
               element_count,
               expected,
               hierarchical_actual
           )
        && output_matches(
               "Atomic/hierarchical comparison",
               element_count,
               atomic_actual,
               hierarchical_actual
           );
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
        constexpr std::array<std::size_t, 12> element_counts = {
            1,
            2,
            255,
            256,
            257,
            511,
            512,
            513,
            1024,
            65537,
            1U << 20,
            1U << 24,
        };

        for (const std::size_t element_count : element_counts) {
            if (!run_reduction_case(element_count)) {
                return 1;
            }
        }

        std::cout
            << "PASS: atomic and hierarchical reductions matched the CPU "
            << "reference for "
            << element_counts.size()
            << " boundary and scale cases.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
