#include "silu_cuda/bias_silu.cuh"
#include "silu_cuda/bias_silu_fp16.cuh"
#include "silu_cuda/cuda_check.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kCTestSkipReturnCode = 77;

struct Options {
    std::string precision = "fp16";
    std::string stage = "stem";
    bool in_place = false;
    bool show_help = false;
};

struct ShapeSpec {
    const char* cli_name;
    const char* report_name;
    std::size_t batch_size;
    std::size_t channel_count;
    std::size_t height;
    std::size_t width;

    constexpr std::size_t spatial_size() const noexcept {
        return height * width;
    }

    constexpr std::size_t element_count() const noexcept {
        return batch_size * channel_count * spatial_size();
    }
};

constexpr std::array<ShapeSpec, 4> kShapes = {{
    {"stem", "stem_64x32x32", 1, 64, 32, 32},
    {"stage2", "stage2_128x16x16", 1, 128, 16, 16},
    {"stage3", "stage3_256x8x8", 1, 256, 8, 8},
    {"stage4", "stage4_512x4x4", 1, 512, 4, 4},
}};

struct RunSummary {
    std::string selected_kernel_path;
    double checksum = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t element_count) {
        SILU_CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&data_),
            element_count * sizeof(T)
        ));
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() noexcept {
        return data_;
    }

    const T* get() const noexcept {
        return data_;
    }

private:
    T* data_ = nullptr;
};

void print_usage(std::ostream& stream, const char* program) {
    stream
        << "Usage: " << program << " [options]\n"
        << "  --precision NAME    fp32 or fp16; default fp16\n"
        << "  --stage NAME        stem, stage2, stage3, or stage4\n"
        << "  --in-place          Reuse the input buffer for output\n"
        << "  --help              Show this message\n";
}

Options parse_options(int argc, char** argv) {
    Options options;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];

        if (argument == "--help") {
            options.show_help = true;
        } else if (argument == "--in-place") {
            options.in_place = true;
        } else if (argument == "--precision" || argument == "--stage") {
            if (index + 1 >= argc) {
                throw std::invalid_argument(
                    "Missing value for option: " + argument
                );
            }
            const std::string value = argv[++index];
            if (argument == "--precision") {
                if (value != "fp32" && value != "fp16") {
                    throw std::invalid_argument(
                        "--precision must be fp32 or fp16"
                    );
                }
                options.precision = value;
            } else {
                options.stage = value;
            }
        } else {
            throw std::invalid_argument(
                "Unknown option: " + argument
            );
        }
    }
    return options;
}

const ShapeSpec& find_shape(const std::string& name) {
    for (const ShapeSpec& shape : kShapes) {
        if (name == shape.cli_name) {
            return shape;
        }
    }
    throw std::invalid_argument(
        "--stage must be stem, stage2, stage3, or stage4"
    );
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

template <typename T>
void copy_to_device(
    T* destination,
    const std::vector<T>& source
) {
    SILU_CUDA_CHECK(cudaMemcpy(
        destination,
        source.data(),
        source.size() * sizeof(T),
        cudaMemcpyHostToDevice
    ));
}

template <typename T>
std::vector<T> copy_to_host(
    const T* source,
    std::size_t element_count
) {
    std::vector<T> output(element_count);
    SILU_CUDA_CHECK(cudaMemcpy(
        output.data(),
        source,
        output.size() * sizeof(T),
        cudaMemcpyDeviceToHost
    ));
    return output;
}

template <typename Converter>
RunSummary summarize_values(
    std::size_t element_count,
    Converter convert,
    std::string selected_kernel_path
) {
    RunSummary summary;
    summary.selected_kernel_path =
        std::move(selected_kernel_path);
    summary.minimum = std::numeric_limits<double>::infinity();
    summary.maximum = -std::numeric_limits<double>::infinity();

    for (std::size_t index = 0; index < element_count; ++index) {
        const double value = convert(index);
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "Bias+SiLU produced a non-finite output"
            );
        }
        summary.checksum += value;
        summary.minimum = std::min(summary.minimum, value);
        summary.maximum = std::max(summary.maximum, value);
    }
    return summary;
}

RunSummary run_fp32(
    const Options& options,
    const ShapeSpec& shape
) {
    const std::vector<float> host_input =
        make_input(shape.element_count());
    const std::vector<float> host_bias =
        make_bias(shape.channel_count);

    DeviceBuffer<float> device_source(shape.element_count());
    DeviceBuffer<float> device_working(shape.element_count());
    DeviceBuffer<float> device_output(shape.element_count());
    DeviceBuffer<float> device_bias(shape.channel_count);

    copy_to_device(device_source.get(), host_input);
    copy_to_device(device_bias.get(), host_bias);

    const std::size_t tensor_bytes =
        shape.element_count() * sizeof(float);
    if (options.in_place) {
        SILU_CUDA_CHECK(cudaMemcpy(
            device_working.get(),
            device_source.get(),
            tensor_bytes,
            cudaMemcpyDeviceToDevice
        ));
    }

    const float* const input =
        options.in_place
        ? device_working.get()
        : device_source.get();
    float* const output =
        options.in_place
        ? device_working.get()
        : device_output.get();

    const silu_cuda::BiasSiluKernelPath selected_path =
        silu_cuda::select_bias_silu_nchw_auto_path(
            input,
            output,
            shape.batch_size,
            shape.channel_count,
            shape.spatial_size()
        );

    SILU_CUDA_CHECK(silu_cuda::launch_bias_silu_nchw(
        input,
        device_bias.get(),
        output,
        shape.batch_size,
        shape.channel_count,
        shape.spatial_size()
    ));
    SILU_CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<float> host_output =
        copy_to_host(output, shape.element_count());
    return summarize_values(
        host_output.size(),
        [&](std::size_t index) {
            return static_cast<double>(host_output[index]);
        },
        selected_path == silu_cuda::BiasSiluKernelPath::kFloat4
            ? "float4"
            : "scalar"
    );
}

RunSummary run_fp16(
    const Options& options,
    const ShapeSpec& shape
) {
    const std::vector<float> float_input =
        make_input(shape.element_count());
    const std::vector<float> float_bias =
        make_bias(shape.channel_count);
    std::vector<__half> host_input(float_input.size());
    std::vector<__half> host_bias(float_bias.size());

    for (std::size_t index = 0;
         index < host_input.size();
         ++index) {
        host_input[index] = __float2half_rn(float_input[index]);
    }
    for (std::size_t index = 0;
         index < host_bias.size();
         ++index) {
        host_bias[index] = __float2half_rn(float_bias[index]);
    }

    DeviceBuffer<__half> device_source(shape.element_count());
    DeviceBuffer<__half> device_working(shape.element_count());
    DeviceBuffer<__half> device_output(shape.element_count());
    DeviceBuffer<__half> device_bias(shape.channel_count);

    copy_to_device(device_source.get(), host_input);
    copy_to_device(device_bias.get(), host_bias);

    const std::size_t tensor_bytes =
        shape.element_count() * sizeof(__half);
    if (options.in_place) {
        SILU_CUDA_CHECK(cudaMemcpy(
            device_working.get(),
            device_source.get(),
            tensor_bytes,
            cudaMemcpyDeviceToDevice
        ));
    }

    const __half* const input =
        options.in_place
        ? device_working.get()
        : device_source.get();
    __half* const output =
        options.in_place
        ? device_working.get()
        : device_output.get();

    const silu_cuda::BiasSiluFp16KernelPath selected_path =
        silu_cuda::select_bias_silu_nchw_fp16_auto_path(
            input,
            output,
            shape.batch_size,
            shape.channel_count,
            shape.spatial_size()
        );

    SILU_CUDA_CHECK(silu_cuda::launch_bias_silu_nchw_fp16(
        input,
        device_bias.get(),
        output,
        shape.batch_size,
        shape.channel_count,
        shape.spatial_size()
    ));
    SILU_CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<__half> host_output =
        copy_to_host(output, shape.element_count());
    return summarize_values(
        host_output.size(),
        [&](std::size_t index) {
            return static_cast<double>(
                __half2float(host_output[index])
            );
        },
        selected_path
                == silu_cuda::BiasSiluFp16KernelPath::kHalf2
            ? "half2"
            : "scalar"
    );
}

void print_json(
    const Options& options,
    const ShapeSpec& shape,
    const RunSummary& summary
) {
    std::cout
        << std::fixed
        << std::setprecision(9)
        << "{\n"
        << "  \"precision\": \"" << options.precision << "\",\n"
        << "  \"stage\": \"" << shape.report_name << "\",\n"
        << "  \"shape\": ["
        << shape.batch_size << ", "
        << shape.channel_count << ", "
        << shape.height << ", "
        << shape.width << "],\n"
        << "  \"elements\": " << shape.element_count() << ",\n"
        << "  \"mode\": \""
        << (options.in_place ? "in-place" : "out-of-place")
        << "\",\n"
        << "  \"selected_kernel_path\": \""
        << summary.selected_kernel_path << "\",\n"
        << "  \"finite\": true,\n"
        << "  \"checksum\": " << summary.checksum << ",\n"
        << "  \"minimum\": " << summary.minimum << ",\n"
        << "  \"maximum\": " << summary.maximum << "\n"
        << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.show_help) {
            print_usage(std::cout, argv[0]);
            return 0;
        }
        const ShapeSpec& shape = find_shape(options.stage);

        int device_count = 0;
        const cudaError_t status = cudaGetDeviceCount(&device_count);
        if (status == cudaErrorNoDevice || device_count == 0) {
            std::cerr
                << "SKIP: CUDA Toolkit is available, "
                << "but no GPU is attached.\n";
            return kCTestSkipReturnCode;
        }
        SILU_CUDA_CHECK(status);
        SILU_CUDA_CHECK(cudaSetDevice(0));

        const RunSummary summary =
            options.precision == "fp32"
            ? run_fp32(options, shape)
            : run_fp16(options, shape);
        print_json(options, shape, summary);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        print_usage(std::cerr, argv[0]);
        return 2;
    }
}
