#include "silu_cuda/bias_silu.cuh"
#include "silu_cuda/cuda_check.h"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kCTestSkipReturnCode = 77;
constexpr float kAbsoluteTolerance = 2.0e-6F;
constexpr float kRelativeTolerance = 2.0e-6F;

struct ShapeCase {
    std::size_t batch_size;
    std::size_t channel_count;
    std::size_t spatial_size;
    silu_cuda::BiasSiluKernelPath expected_layout_path;
    silu_cuda::BiasSiluKernelPath expected_auto_path;
};

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
    if (silu_cuda::launch_bias_silu_nchw(
            nullptr,
            nullptr,
            nullptr,
            0,
            3,
            32
        ) != cudaSuccess
        || silu_cuda::launch_bias_silu_nchw_scalar(
            nullptr,
            nullptr,
            nullptr,
            1,
            0,
            32
        ) != cudaSuccess
        || silu_cuda::launch_bias_silu_nchw_vectorized(
            nullptr,
            nullptr,
            nullptr,
            1,
            3,
            0
        ) != cudaSuccess) {
        std::cerr << "Zero-sized Bias+SiLU must be a no-op.\n";
        return false;
    }

    float dummy = 0.0F;
    if (silu_cuda::launch_bias_silu_nchw(
            nullptr,
            &dummy,
            &dummy,
            1,
            1,
            1
        ) != cudaErrorInvalidValue) {
        std::cerr << "Bias+SiLU must reject a null input.\n";
        return false;
    }

    if (silu_cuda::launch_bias_silu_nchw(
            &dummy,
            nullptr,
            &dummy,
            1,
            1,
            1
        ) != cudaErrorInvalidValue) {
        std::cerr << "Bias+SiLU must reject a null bias.\n";
        return false;
    }

    if (silu_cuda::launch_bias_silu_nchw(
            &dummy,
            &dummy,
            nullptr,
            1,
            1,
            1
        ) != cudaErrorInvalidValue) {
        std::cerr << "Bias+SiLU must reject a null output.\n";
        return false;
    }

    if (silu_cuda::launch_bias_silu_nchw_vectorized(
            &dummy,
            &dummy,
            &dummy,
            std::numeric_limits<std::size_t>::max(),
            2,
            1
        ) != cudaErrorInvalidValue) {
        std::cerr << "Bias+SiLU must reject shape overflow.\n";
        return false;
    }

    if (silu_cuda::select_bias_silu_nchw_kernel_path(
            nullptr,
            nullptr,
            4
        ) != silu_cuda::BiasSiluKernelPath::kScalar) {
        std::cerr << "Null pointers must select the scalar fallback.\n";
        return false;
    }

    return true;
}

std::vector<float> make_input(std::size_t element_count) {
    std::vector<float> input(element_count);
    for (std::size_t index = 0; index < element_count; ++index) {
        const int centered =
            static_cast<int>(index % 257) - 128;
        input[index] =
            static_cast<float>(centered) * 0.0625F;
    }
    return input;
}

std::vector<float> make_bias(std::size_t channel_count) {
    std::vector<float> bias(channel_count);
    for (std::size_t channel = 0;
         channel < channel_count;
         ++channel) {
        const int centered =
            static_cast<int>(channel % 17) - 8;
        bias[channel] =
            static_cast<float>(centered) * 0.03125F;
    }
    return bias;
}

std::vector<float> make_reference(
    const std::vector<float>& input,
    const std::vector<float>& bias,
    const ShapeCase& shape
) {
    std::vector<float> reference(input.size());

    for (std::size_t index = 0; index < input.size(); ++index) {
        const std::size_t channel =
            (index / shape.spatial_size)
            % shape.channel_count;
        const double value =
            static_cast<double>(input[index])
            + static_cast<double>(bias[channel]);
        reference[index] = static_cast<float>(
            value / (1.0 + std::exp(-value))
        );
    }

    return reference;
}

bool outputs_match(
    const char* mode,
    const std::vector<float>& reference,
    const std::vector<float>& actual
) {
    for (std::size_t index = 0; index < reference.size(); ++index) {
        if (!std::isfinite(actual[index])) {
            std::cerr
                << mode
                << " produced a non-finite value at index "
                << index << ".\n";
            return false;
        }

        const float absolute_error =
            std::abs(actual[index] - reference[index]);
        const float allowed_error =
            kAbsoluteTolerance
            + kRelativeTolerance * std::abs(reference[index]);

        if (absolute_error > allowed_error) {
            std::cerr
                << mode
                << " mismatch at index " << index
                << ": expected=" << reference[index]
                << ", actual=" << actual[index]
                << ", abs_error=" << absolute_error
                << ", allowed_error=" << allowed_error
                << '\n';
            return false;
        }
    }

    return true;
}

void copy_to_device(
    float* destination,
    const std::vector<float>& source
) {
    SILU_CUDA_CHECK(cudaMemcpy(
        destination,
        source.data(),
        source.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));
}

bool copy_and_compare(
    const char* mode,
    const float* device_output,
    const std::vector<float>& reference
) {
    std::vector<float> actual(reference.size());
    SILU_CUDA_CHECK(cudaDeviceSynchronize());
    SILU_CUDA_CHECK(cudaMemcpy(
        actual.data(),
        device_output,
        actual.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    return outputs_match(mode, reference, actual);
}

bool run_shape_case(const ShapeCase& shape) {
    const std::size_t element_count =
        shape.batch_size
        * shape.channel_count
        * shape.spatial_size;

    const std::vector<float> input = make_input(element_count);
    const std::vector<float> bias = make_bias(shape.channel_count);
    const std::vector<float> reference =
        make_reference(input, bias, shape);

    DeviceBuffer device_input(element_count);
    DeviceBuffer device_bias(shape.channel_count);
    DeviceBuffer device_output(element_count);
    copy_to_device(device_input.get(), input);
    copy_to_device(device_bias.get(), bias);

    const silu_cuda::BiasSiluKernelPath layout_path =
        silu_cuda::select_bias_silu_nchw_kernel_path(
            device_input.get(),
            device_output.get(),
            shape.spatial_size
        );
    if (layout_path != shape.expected_layout_path) {
        std::cerr << "Bias+SiLU selected an unexpected layout path.\n";
        return false;
    }

    const silu_cuda::BiasSiluKernelPath auto_path =
        silu_cuda::select_bias_silu_nchw_auto_path(
            device_input.get(),
            device_output.get(),
            shape.batch_size,
            shape.channel_count,
            shape.spatial_size
        );
    if (auto_path != shape.expected_auto_path) {
        std::cerr << "Bias+SiLU selected an unexpected automatic path.\n";
        return false;
    }

    SILU_CUDA_CHECK(silu_cuda::launch_bias_silu_nchw_scalar(
        device_input.get(),
        device_bias.get(),
        device_output.get(),
        shape.batch_size,
        shape.channel_count,
        shape.spatial_size
    ));
    if (!copy_and_compare(
            "Scalar out-of-place Bias+SiLU",
            device_output.get(),
            reference
        )) {
        return false;
    }

    SILU_CUDA_CHECK(silu_cuda::launch_bias_silu_nchw(
        device_input.get(),
        device_bias.get(),
        device_output.get(),
        shape.batch_size,
        shape.channel_count,
        shape.spatial_size
    ));
    if (!copy_and_compare(
            "Auto-dispatch out-of-place Bias+SiLU",
            device_output.get(),
            reference
        )) {
        return false;
    }

    copy_to_device(device_input.get(), input);
    SILU_CUDA_CHECK(silu_cuda::launch_bias_silu_nchw_vectorized(
        device_input.get(),
        device_bias.get(),
        device_input.get(),
        shape.batch_size,
        shape.channel_count,
        shape.spatial_size
    ));
    return copy_and_compare(
        "Vectorized-entry in-place Bias+SiLU",
        device_input.get(),
        reference
    );
}

bool run_misaligned_fallback_case() {
    constexpr ShapeCase shape = {
        1,
        3,
        8,
        silu_cuda::BiasSiluKernelPath::kScalar,
    };
    const std::size_t element_count =
        shape.batch_size
        * shape.channel_count
        * shape.spatial_size;
    const std::vector<float> input = make_input(element_count);
    const std::vector<float> bias = make_bias(shape.channel_count);
    const std::vector<float> reference =
        make_reference(input, bias, shape);

    DeviceBuffer input_storage(element_count + 1);
    DeviceBuffer output_storage(element_count + 1);
    DeviceBuffer device_bias(shape.channel_count);
    float* const misaligned_input = input_storage.get() + 1;
    float* const misaligned_output = output_storage.get() + 1;
    copy_to_device(misaligned_input, input);
    copy_to_device(device_bias.get(), bias);

    if (silu_cuda::select_bias_silu_nchw_kernel_path(
            misaligned_input,
            misaligned_output,
            shape.spatial_size
        ) != silu_cuda::BiasSiluKernelPath::kScalar) {
        std::cerr << "Misaligned pointers must select scalar fallback.\n";
        return false;
    }

    SILU_CUDA_CHECK(silu_cuda::launch_bias_silu_nchw_vectorized(
        misaligned_input,
        device_bias.get(),
        misaligned_output,
        shape.batch_size,
        shape.channel_count,
        shape.spatial_size
    ));
    return copy_and_compare(
        "Misaligned scalar-fallback Bias+SiLU",
        misaligned_output,
        reference
    );
}

}  // namespace

int main() {
    if (!host_contract_is_valid()) {
        return 1;
    }

    int device_count = 0;
    const cudaError_t device_status =
        cudaGetDeviceCount(&device_count);

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
        constexpr std::array<ShapeCase, 6> shapes = {{
            {1, 1, 1,
             silu_cuda::BiasSiluKernelPath::kScalar,
             silu_cuda::BiasSiluKernelPath::kScalar},
            {1, 64, 32 * 32,
             silu_cuda::BiasSiluKernelPath::kFloat4,
             silu_cuda::BiasSiluKernelPath::kFloat4},
            {1, 128, 16 * 16,
             silu_cuda::BiasSiluKernelPath::kFloat4,
             silu_cuda::BiasSiluKernelPath::kScalar},
            {2, 17, 36,
             silu_cuda::BiasSiluKernelPath::kFloat4,
             silu_cuda::BiasSiluKernelPath::kScalar},
            {2, 17, 37,
             silu_cuda::BiasSiluKernelPath::kScalar,
             silu_cuda::BiasSiluKernelPath::kScalar},
            {3, 64, 7 * 7,
             silu_cuda::BiasSiluKernelPath::kScalar,
             silu_cuda::BiasSiluKernelPath::kScalar},
        }};

        for (const ShapeCase& shape : shapes) {
            if (!run_shape_case(shape)) {
                return 1;
            }
        }

        if (!run_misaligned_fallback_case()) {
            return 1;
        }

        std::cout
            << "PASS: scalar, explicit float4, adaptive dispatch, and "
            << "in-place NCHW Bias+SiLU matched the CPU reference.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
