#include "silu_cuda/bias_silu_fp16.cuh"
#include "silu_cuda/cuda_check.h"

#include <cuda_fp16.h>
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
constexpr float kAbsoluteTolerance = 5.0e-3F;
constexpr float kRelativeTolerance = 2.0e-3F;

struct ShapeCase {
    std::size_t batch_size;
    std::size_t channel_count;
    std::size_t spatial_size;
    silu_cuda::BiasSiluFp16KernelPath expected_layout_path;
    silu_cuda::BiasSiluFp16KernelPath expected_auto_path;
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t element_count) {
        SILU_CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&data_),
            element_count * sizeof(__half)
        ));
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    __half* get() noexcept {
        return data_;
    }

private:
    __half* data_ = nullptr;
};

bool host_contract_is_valid() {
    if (silu_cuda::launch_bias_silu_nchw_fp16(
            nullptr,
            nullptr,
            nullptr,
            0,
            3,
            32
        ) != cudaSuccess
        || silu_cuda::launch_bias_silu_nchw_fp16_scalar(
            nullptr,
            nullptr,
            nullptr,
            1,
            0,
            32
        ) != cudaSuccess
        || silu_cuda::launch_bias_silu_nchw_fp16_vectorized(
            nullptr,
            nullptr,
            nullptr,
            1,
            3,
            0
        ) != cudaSuccess) {
        std::cerr << "Zero-sized FP16 Bias+SiLU must be a no-op.\n";
        return false;
    }

    __half dummy = __float2half(0.0F);
    if (silu_cuda::launch_bias_silu_nchw_fp16(
            nullptr,
            &dummy,
            &dummy,
            1,
            1,
            1
        ) != cudaErrorInvalidValue
        || silu_cuda::launch_bias_silu_nchw_fp16(
            &dummy,
            nullptr,
            &dummy,
            1,
            1,
            1
        ) != cudaErrorInvalidValue
        || silu_cuda::launch_bias_silu_nchw_fp16(
            &dummy,
            &dummy,
            nullptr,
            1,
            1,
            1
        ) != cudaErrorInvalidValue) {
        std::cerr << "FP16 Bias+SiLU must reject null data pointers.\n";
        return false;
    }

    if (silu_cuda::launch_bias_silu_nchw_fp16_vectorized(
            &dummy,
            &dummy,
            &dummy,
            std::numeric_limits<std::size_t>::max(),
            2,
            1
        ) != cudaErrorInvalidValue) {
        std::cerr << "FP16 Bias+SiLU must reject shape overflow.\n";
        return false;
    }

    if (silu_cuda::select_bias_silu_nchw_fp16_kernel_path(
            nullptr,
            nullptr,
            2
        ) != silu_cuda::BiasSiluFp16KernelPath::kScalar) {
        std::cerr << "Null FP16 pointers must select scalar fallback.\n";
        return false;
    }

    alignas(4) __half aligned_input[2] = {};
    alignas(4) __half aligned_output[2] = {};
    if (silu_cuda::select_bias_silu_nchw_fp16_auto_path(
            aligned_input,
            aligned_output,
            1,
            64,
            1024
        ) != silu_cuda::BiasSiluFp16KernelPath::kHalf2
        || silu_cuda::select_bias_silu_nchw_fp16_auto_path(
            aligned_input,
            aligned_output,
            1,
            64,
            1022
        ) != silu_cuda::BiasSiluFp16KernelPath::kScalar) {
        std::cerr
            << "FP16 automatic dispatch must honor the 65,536-element "
            << "threshold.\n";
        return false;
    }
    return true;
}

std::vector<__half> make_input(std::size_t element_count) {
    std::vector<__half> input(element_count);
    for (std::size_t index = 0; index < element_count; ++index) {
        const int centered = static_cast<int>(index % 257) - 128;
        input[index] = __float2half(
            static_cast<float>(centered) * 0.0625F
        );
    }
    return input;
}

std::vector<__half> make_bias(std::size_t channel_count) {
    std::vector<__half> bias(channel_count);
    for (std::size_t channel = 0;
         channel < channel_count;
         ++channel) {
        const int centered = static_cast<int>(channel % 17) - 8;
        bias[channel] = __float2half(
            static_cast<float>(centered) * 0.03125F
        );
    }
    return bias;
}

std::vector<float> make_reference(
    const std::vector<__half>& input,
    const std::vector<__half>& bias,
    const ShapeCase& shape
) {
    std::vector<float> reference(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        const std::size_t channel =
            (index / shape.spatial_size) % shape.channel_count;
        const float value =
            __half2float(input[index])
            + __half2float(bias[channel]);
        reference[index] =
            value / (1.0F + std::exp(-value));
    }
    return reference;
}

void copy_to_device(
    __half* destination,
    const std::vector<__half>& source
) {
    SILU_CUDA_CHECK(cudaMemcpy(
        destination,
        source.data(),
        source.size() * sizeof(__half),
        cudaMemcpyHostToDevice
    ));
}

bool copy_and_compare(
    const char* mode,
    const __half* device_output,
    const std::vector<float>& reference
) {
    std::vector<__half> output(reference.size());
    SILU_CUDA_CHECK(cudaDeviceSynchronize());
    SILU_CUDA_CHECK(cudaMemcpy(
        output.data(),
        device_output,
        output.size() * sizeof(__half),
        cudaMemcpyDeviceToHost
    ));

    for (std::size_t index = 0; index < output.size(); ++index) {
        const float actual = __half2float(output[index]);
        if (!std::isfinite(actual)) {
            std::cerr
                << mode << " produced non-finite output at index "
                << index << ".\n";
            return false;
        }
        const float absolute_error =
            std::abs(actual - reference[index]);
        const float allowed_error =
            kAbsoluteTolerance
            + kRelativeTolerance * std::abs(reference[index]);
        if (absolute_error > allowed_error) {
            std::cerr
                << mode << " mismatch at index " << index
                << ": expected=" << reference[index]
                << ", actual=" << actual
                << ", abs_error=" << absolute_error
                << ", allowed_error=" << allowed_error
                << '\n';
            return false;
        }
    }
    return true;
}

bool run_shape_case(const ShapeCase& shape) {
    const std::size_t element_count =
        shape.batch_size * shape.channel_count * shape.spatial_size;
    const std::vector<__half> input = make_input(element_count);
    const std::vector<__half> bias = make_bias(shape.channel_count);
    const std::vector<float> reference =
        make_reference(input, bias, shape);

    DeviceBuffer device_input(element_count);
    DeviceBuffer device_bias(shape.channel_count);
    DeviceBuffer device_output(element_count);
    copy_to_device(device_input.get(), input);
    copy_to_device(device_bias.get(), bias);

    const auto selected_path =
        silu_cuda::select_bias_silu_nchw_fp16_kernel_path(
            device_input.get(),
            device_output.get(),
            shape.spatial_size
        );
    if (selected_path != shape.expected_layout_path) {
        std::cerr
            << "FP16 Bias+SiLU selected an unexpected layout path.\n";
        return false;
    }

    const auto auto_path =
        silu_cuda::select_bias_silu_nchw_fp16_auto_path(
            device_input.get(),
            device_output.get(),
            shape.batch_size,
            shape.channel_count,
            shape.spatial_size
        );
    if (auto_path != shape.expected_auto_path) {
        std::cerr
            << "FP16 Bias+SiLU selected an unexpected automatic path.\n";
        return false;
    }

    SILU_CUDA_CHECK(silu_cuda::launch_bias_silu_nchw_fp16_scalar(
        device_input.get(),
        device_bias.get(),
        device_output.get(),
        shape.batch_size,
        shape.channel_count,
        shape.spatial_size
    ));
    if (!copy_and_compare(
            "FP16 scalar out-of-place",
            device_output.get(),
            reference
        )) {
        return false;
    }

    SILU_CUDA_CHECK(
        silu_cuda::launch_bias_silu_nchw_fp16_vectorized(
            device_input.get(),
            device_bias.get(),
            device_output.get(),
            shape.batch_size,
            shape.channel_count,
            shape.spatial_size
        )
    );
    if (!copy_and_compare(
            "FP16 explicit half2 out-of-place",
            device_output.get(),
            reference
        )) {
        return false;
    }

    SILU_CUDA_CHECK(silu_cuda::launch_bias_silu_nchw_fp16(
        device_input.get(),
        device_bias.get(),
        device_output.get(),
        shape.batch_size,
        shape.channel_count,
        shape.spatial_size
    ));
    if (!copy_and_compare(
            "FP16 auto out-of-place",
            device_output.get(),
            reference
        )) {
        return false;
    }

    copy_to_device(device_input.get(), input);
    SILU_CUDA_CHECK(silu_cuda::launch_bias_silu_nchw_fp16(
        device_input.get(),
        device_bias.get(),
        device_input.get(),
        shape.batch_size,
        shape.channel_count,
        shape.spatial_size
    ));
    return copy_and_compare(
        "FP16 auto in-place",
        device_input.get(),
        reference
    );
}

bool run_misaligned_fallback_case() {
    const ShapeCase shape = {
        1,
        3,
        32,
        silu_cuda::BiasSiluFp16KernelPath::kHalf2,
        silu_cuda::BiasSiluFp16KernelPath::kScalar,
    };
    const std::size_t element_count =
        shape.batch_size * shape.channel_count * shape.spatial_size;
    const std::vector<__half> input = make_input(element_count);
    const std::vector<__half> bias = make_bias(shape.channel_count);
    const std::vector<float> reference =
        make_reference(input, bias, shape);

    DeviceBuffer input_storage(element_count + 1);
    DeviceBuffer output_storage(element_count + 1);
    DeviceBuffer device_bias(shape.channel_count);
    __half* const misaligned_input = input_storage.get() + 1;
    __half* const misaligned_output = output_storage.get() + 1;
    copy_to_device(misaligned_input, input);
    copy_to_device(device_bias.get(), bias);

    if (silu_cuda::select_bias_silu_nchw_fp16_kernel_path(
            misaligned_input,
            misaligned_output,
            shape.spatial_size
        ) != silu_cuda::BiasSiluFp16KernelPath::kScalar
        || silu_cuda::select_bias_silu_nchw_fp16_auto_path(
            misaligned_input,
            misaligned_output,
            shape.batch_size,
            shape.channel_count,
            shape.spatial_size
        ) != silu_cuda::BiasSiluFp16KernelPath::kScalar) {
        std::cerr
            << "Misaligned FP16 pointers must select scalar fallback.\n";
        return false;
    }

    SILU_CUDA_CHECK(
        silu_cuda::launch_bias_silu_nchw_fp16_vectorized(
            misaligned_input,
            device_bias.get(),
            misaligned_output,
            shape.batch_size,
            shape.channel_count,
            shape.spatial_size
        )
    );
    return copy_and_compare(
        "Misaligned FP16 scalar fallback",
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
        constexpr std::array<ShapeCase, 5> shapes = {{
            {
                1, 1, 1,
                silu_cuda::BiasSiluFp16KernelPath::kScalar,
                silu_cuda::BiasSiluFp16KernelPath::kScalar,
            },
            {
                1, 64, 32 * 32,
                silu_cuda::BiasSiluFp16KernelPath::kHalf2,
                silu_cuda::BiasSiluFp16KernelPath::kHalf2,
            },
            {
                2, 17, 36,
                silu_cuda::BiasSiluFp16KernelPath::kHalf2,
                silu_cuda::BiasSiluFp16KernelPath::kScalar,
            },
            {
                2, 17, 37,
                silu_cuda::BiasSiluFp16KernelPath::kScalar,
                silu_cuda::BiasSiluFp16KernelPath::kScalar,
            },
            {
                3, 64, 7 * 7,
                silu_cuda::BiasSiluFp16KernelPath::kScalar,
                silu_cuda::BiasSiluFp16KernelPath::kScalar,
            },
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
            << "PASS: FP16 scalar, half2, adaptive dispatch, and "
            << "in-place NCHW Bias+SiLU matched the FP32 CPU reference.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
