#include "silu_cuda/bias_silu_fp16.cuh"
#include "silu_cuda/cuda_check.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef SILU_CUDA_COMPILER_VERSION
#define SILU_CUDA_COMPILER_VERSION "unknown"
#endif

#ifndef SILU_CXX_COMPILER_ID
#define SILU_CXX_COMPILER_ID "unknown"
#endif

#ifndef SILU_CXX_COMPILER_VERSION
#define SILU_CXX_COMPILER_VERSION "unknown"
#endif

namespace {

constexpr int kCTestSkipReturnCode = 77;
constexpr float kAbsoluteTolerance = 5.0e-3F;
constexpr float kRelativeTolerance = 2.0e-3F;

struct Options {
    int warmup_iterations = 20;
    int measured_iterations = 200;
    std::string implementation = "all";
    std::string mode = "both";
    bool show_help = false;
};

struct ShapeSpec {
    const char* name;
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

constexpr std::array<ShapeSpec, 4> kResNet18Shapes = {{
    {"stem_64x32x32", 1, 64, 32, 32},
    {"stage2_128x16x16", 1, 128, 16, 16},
    {"stage3_256x8x8", 1, 256, 8, 8},
    {"stage4_512x4x4", 1, 512, 4, 4},
}};

struct Statistics {
    double mean_ms = 0.0;
    double p50_ms = 0.0;
    double p90_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double effective_tensor_gbps = 0.0;
};

struct BenchmarkResult {
    ShapeSpec shape;
    std::string mode;
    std::string implementation;
    std::string selected_kernel_path;
    Statistics statistics;
    double speedup_vs_scalar =
        std::numeric_limits<double>::quiet_NaN();
    double max_abs_error = 0.0;
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

class CudaEvent {
public:
    CudaEvent() {
        SILU_CUDA_CHECK(cudaEventCreate(&event_));
    }

    ~CudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t get() const noexcept {
        return event_;
    }

private:
    cudaEvent_t event_ = nullptr;
};

void print_usage(std::ostream& stream, const char* program) {
    stream
        << "Usage: " << program << " [options]\n"
        << "  --warmup N               Warm-up launches; default 20\n"
        << "  --iterations N           Measured launches; default 200\n"
        << "  --implementation NAME    scalar, half2, auto, or all\n"
        << "  --mode NAME              out-of-place, in-place, or both\n"
        << "  --help                    Show this message\n";
}

int parse_count(
    const std::string& text,
    const char* option,
    bool allow_zero
) {
    std::size_t consumed = 0;
    const long long parsed = std::stoll(text, &consumed);
    const long long minimum = allow_zero ? 0 : 1;
    if (consumed != text.size()
        || parsed < minimum
        || parsed > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            std::string(option)
            + (allow_zero
                ? " requires a non-negative integer"
                : " requires a positive integer")
        );
    }
    return static_cast<int>(parsed);
}

Options parse_options(int argc, char** argv) {
    Options options;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];

        if (argument == "--help") {
            options.show_help = true;
            continue;
        }

        if (index + 1 >= argc) {
            throw std::invalid_argument(
                "Missing value for option: " + argument
            );
        }

        const std::string value = argv[++index];
        if (argument == "--warmup") {
            options.warmup_iterations =
                parse_count(value, "--warmup", true);
        } else if (argument == "--iterations") {
            options.measured_iterations =
                parse_count(value, "--iterations", false);
        } else if (argument == "--implementation") {
            if (value != "scalar"
                && value != "half2"
                && value != "auto"
                && value != "all") {
                throw std::invalid_argument(
                    "--implementation must be scalar, half2, auto, or all"
                );
            }
            options.implementation = value;
        } else if (argument == "--mode") {
            if (value != "out-of-place"
                && value != "in-place"
                && value != "both") {
                throw std::invalid_argument(
                    "--mode must be out-of-place, in-place, or both"
                );
            }
            options.mode = value;
        } else {
            throw std::invalid_argument(
                "Unknown option: " + argument
            );
        }
    }

    return options;
}

std::vector<std::string> selected_implementations(
    const Options& options
) {
    if (options.implementation == "all") {
        return {"scalar", "half2", "auto"};
    }
    return {options.implementation};
}

std::vector<std::string> selected_modes(const Options& options) {
    if (options.mode == "both") {
        return {"out-of-place", "in-place"};
    }
    return {options.mode};
}

std::vector<__half> make_input(std::size_t element_count) {
    std::vector<__half> input(element_count);
    for (std::size_t index = 0; index < element_count; ++index) {
        const int centered =
            static_cast<int>(index % 257) - 128;
        input[index] = __float2half_rn(
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
        const int centered =
            static_cast<int>(channel % 17) - 8;
        bias[channel] = __float2half_rn(
            static_cast<float>(centered) * 0.03125F
        );
    }
    return bias;
}

std::vector<float> make_reference(
    const std::vector<__half>& input,
    const std::vector<__half>& bias,
    const ShapeSpec& shape
) {
    std::vector<float> reference(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        const std::size_t channel =
            (index / shape.spatial_size())
            % shape.channel_count;
        const double value =
            static_cast<double>(__half2float(input[index]))
            + static_cast<double>(__half2float(bias[channel]));
        reference[index] = static_cast<float>(
            value / (1.0 + std::exp(-value))
        );
    }
    return reference;
}

cudaError_t launch_implementation(
    const std::string& implementation,
    const __half* input,
    const __half* bias,
    __half* output,
    const ShapeSpec& shape
) {
    if (implementation == "scalar") {
        return silu_cuda::launch_bias_silu_nchw_fp16_scalar(
            input,
            bias,
            output,
            shape.batch_size,
            shape.channel_count,
            shape.spatial_size()
        );
    }
    if (implementation == "half2") {
        return silu_cuda::launch_bias_silu_nchw_fp16_vectorized(
            input,
            bias,
            output,
            shape.batch_size,
            shape.channel_count,
            shape.spatial_size()
        );
    }
    return silu_cuda::launch_bias_silu_nchw_fp16(
        input,
        bias,
        output,
        shape.batch_size,
        shape.channel_count,
        shape.spatial_size()
    );
}

std::string selected_path_name(
    const std::string& implementation,
    const __half* input,
    const __half* output,
    const ShapeSpec& shape
) {
    if (implementation == "scalar") {
        return "scalar";
    }

    return silu_cuda::select_bias_silu_nchw_fp16_kernel_path(
        input,
        output,
        shape.spatial_size()
    ) == silu_cuda::BiasSiluFp16KernelPath::kHalf2
        ? "half2"
        : "scalar_layout_fallback";
}

double validate_output(
    const __half* device_output,
    const std::vector<float>& reference
) {
    std::vector<__half> output_storage(reference.size());
    SILU_CUDA_CHECK(cudaDeviceSynchronize());
    SILU_CUDA_CHECK(cudaMemcpy(
        output_storage.data(),
        device_output,
        output_storage.size() * sizeof(__half),
        cudaMemcpyDeviceToHost
    ));

    double maximum_error = 0.0;
    for (std::size_t index = 0;
         index < output_storage.size();
         ++index) {
        const float actual = __half2float(output_storage[index]);
        if (!std::isfinite(actual)) {
            throw std::runtime_error(
                "FP16 Bias+SiLU produced a non-finite output"
            );
        }
        const double absolute_error = std::abs(
            static_cast<double>(actual)
            - static_cast<double>(reference[index])
        );
        const double allowed_error =
            static_cast<double>(kAbsoluteTolerance)
            + static_cast<double>(kRelativeTolerance)
                * std::abs(static_cast<double>(reference[index]));
        if (absolute_error > allowed_error) {
            throw std::runtime_error(
                "FP16 Bias+SiLU output failed correctness validation"
            );
        }
        maximum_error = std::max(maximum_error, absolute_error);
    }
    return maximum_error;
}

double percentile(
    const std::vector<float>& sorted_samples,
    double quantile
) {
    const double position =
        static_cast<double>(sorted_samples.size() - 1)
        * quantile;
    const std::size_t lower =
        static_cast<std::size_t>(std::floor(position));
    const std::size_t upper =
        std::min(lower + 1, sorted_samples.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return static_cast<double>(sorted_samples[lower])
        * (1.0 - fraction)
        + static_cast<double>(sorted_samples[upper])
        * fraction;
}

Statistics summarize(
    std::vector<float> samples_ms,
    std::size_t effective_tensor_bytes
) {
    std::sort(samples_ms.begin(), samples_ms.end());

    Statistics result;
    result.mean_ms = std::accumulate(
        samples_ms.begin(),
        samples_ms.end(),
        0.0
    ) / static_cast<double>(samples_ms.size());
    result.p50_ms = percentile(samples_ms, 0.50);
    result.p90_ms = percentile(samples_ms, 0.90);
    result.p95_ms = percentile(samples_ms, 0.95);
    result.p99_ms = percentile(samples_ms, 0.99);
    result.min_ms = samples_ms.front();
    result.max_ms = samples_ms.back();
    result.effective_tensor_gbps =
        result.mean_ms > 0.0
        ? static_cast<double>(effective_tensor_bytes)
            / result.mean_ms
            / 1.0e6
        : 0.0;
    return result;
}

std::vector<BenchmarkResult> benchmark_mode_interleaved(
    const Options& options,
    const ShapeSpec& shape,
    const std::string& mode,
    const std::vector<std::string>& implementations,
    const std::vector<float>& reference,
    __half* device_source,
    __half* device_working,
    __half* device_bias,
    __half* device_output
) {
    const std::size_t tensor_bytes =
        shape.element_count() * sizeof(__half);
    const bool in_place = mode == "in-place";
    __half* const launch_input =
        in_place ? device_working : device_source;
    __half* const launch_output =
        in_place ? device_working : device_output;

    const auto prepare = [&]() -> cudaError_t {
        if (!in_place) {
            return cudaSuccess;
        }
        return cudaMemcpyAsync(
            device_working,
            device_source,
            tensor_bytes,
            cudaMemcpyDeviceToDevice
        );
    };
    const auto launch =
        [&](std::size_t implementation_index) -> cudaError_t {
            return launch_implementation(
                implementations[implementation_index],
                launch_input,
                device_bias,
                launch_output,
                shape
            );
        };
    const auto for_each_rotated =
        [&](int round, const auto& operation) {
            for (std::size_t offset = 0;
                 offset < implementations.size();
                 ++offset) {
                const std::size_t implementation_index =
                    (static_cast<std::size_t>(round) + offset)
                    % implementations.size();
                operation(implementation_index);
            }
        };

    std::vector<double> maximum_errors(
        implementations.size(),
        0.0
    );
    for (std::size_t implementation_index = 0;
         implementation_index < implementations.size();
         ++implementation_index) {
        SILU_CUDA_CHECK(prepare());
        SILU_CUDA_CHECK(launch(implementation_index));
        maximum_errors[implementation_index] =
            validate_output(launch_output, reference);
    }

    for (int round = 0;
         round < options.warmup_iterations;
         ++round) {
        for_each_rotated(
            round,
            [&](std::size_t implementation_index) {
                SILU_CUDA_CHECK(prepare());
                SILU_CUDA_CHECK(launch(implementation_index));
            }
        );
    }
    SILU_CUDA_CHECK(cudaDeviceSynchronize());

    CudaEvent start_event;
    CudaEvent stop_event;
    std::vector<std::vector<float>> samples_ms(
        implementations.size()
    );
    for (std::vector<float>& samples : samples_ms) {
        samples.reserve(
            static_cast<std::size_t>(options.measured_iterations)
        );
    }

    for (int round = 0;
         round < options.measured_iterations;
         ++round) {
        for_each_rotated(
            round,
            [&](std::size_t implementation_index) {
                SILU_CUDA_CHECK(prepare());
                SILU_CUDA_CHECK(cudaEventRecord(start_event.get()));
                SILU_CUDA_CHECK(launch(implementation_index));
                SILU_CUDA_CHECK(cudaEventRecord(stop_event.get()));
                SILU_CUDA_CHECK(
                    cudaEventSynchronize(stop_event.get())
                );

                float elapsed_ms = 0.0F;
                SILU_CUDA_CHECK(cudaEventElapsedTime(
                    &elapsed_ms,
                    start_event.get(),
                    stop_event.get()
                ));
                samples_ms[implementation_index].push_back(
                    elapsed_ms
                );
            }
        );
    }

    std::vector<BenchmarkResult> results;
    results.reserve(implementations.size());
    for (std::size_t implementation_index = 0;
         implementation_index < implementations.size();
         ++implementation_index) {
        results.push_back({
            shape,
            mode,
            implementations[implementation_index],
            selected_path_name(
                implementations[implementation_index],
                launch_input,
                launch_output,
                shape
            ),
            summarize(
                std::move(samples_ms[implementation_index]),
                tensor_bytes * 2
            ),
            std::numeric_limits<double>::quiet_NaN(),
            maximum_errors[implementation_index],
        });
    }
    return results;
}

void add_scalar_speedups(std::vector<BenchmarkResult>* results) {
    for (BenchmarkResult& result : *results) {
        const auto scalar = std::find_if(
            results->begin(),
            results->end(),
            [&](const BenchmarkResult& candidate) {
                return std::string(candidate.shape.name)
                        == result.shape.name
                    && candidate.mode == result.mode
                    && candidate.implementation == "scalar";
            }
        );
        if (scalar != results->end()
            && result.statistics.mean_ms > 0.0) {
            result.speedup_vs_scalar =
                scalar->statistics.mean_ms
                / result.statistics.mean_ms;
        }
    }
}

std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char character : value) {
        if (character == '"' || character == '\\') {
            escaped.push_back('\\');
            escaped.push_back(character);
        } else if (character == '\n') {
            escaped += "\\n";
        } else if (character == '\r') {
            escaped += "\\r";
        } else if (character == '\t') {
            escaped += "\\t";
        } else {
            escaped.push_back(character);
        }
    }
    return escaped;
}

std::string cuda_version_string(int encoded_version) {
    return std::to_string(encoded_version / 1000)
        + "."
        + std::to_string((encoded_version % 1000) / 10);
}

void print_json(
    const Options& options,
    const cudaDeviceProp& device,
    int driver_version,
    int runtime_version,
    const std::vector<BenchmarkResult>& results
) {
    std::cout
        << std::fixed
        << std::setprecision(6)
        << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"data_type\": \"fp16\",\n"
        << "  \"device\": {\n"
        << "    \"name\": \""
        << json_escape(device.name) << "\",\n"
        << "    \"compute_capability\": \""
        << device.major << '.' << device.minor << "\",\n"
        << "    \"multiprocessor_count\": "
        << device.multiProcessorCount << ",\n"
        << "    \"total_global_memory_bytes\": "
        << device.totalGlobalMem << ",\n"
        << "    \"driver_version\": \""
        << cuda_version_string(driver_version) << "\",\n"
        << "    \"runtime_version\": \""
        << cuda_version_string(runtime_version) << "\"\n"
        << "  },\n"
        << "  \"build\": {\n"
        << "    \"cuda_compiler_version\": \""
        << json_escape(SILU_CUDA_COMPILER_VERSION) << "\",\n"
        << "    \"host_compiler\": \""
        << json_escape(SILU_CXX_COMPILER_ID)
        << ' '
        << json_escape(SILU_CXX_COMPILER_VERSION)
        << "\"\n"
        << "  },\n"
        << "  \"protocol\": {\n"
        << "    \"shape_set\": \"CIFAR-10 ResNet18 stage outputs\",\n"
        << "    \"arithmetic\": \"FP16 storage with FP32 Bias+SiLU math\",\n"
        << "    \"warmup_iterations\": "
        << options.warmup_iterations << ",\n"
        << "    \"measured_iterations\": "
        << options.measured_iterations << ",\n"
        << "    \"iteration_scope\": \"per implementation\",\n"
        << "    \"execution_order\": \"interleaved round-robin with "
        << "per-round rotation\",\n"
        << "    \"timer\": \"CUDA events on the default stream\",\n"
        << "    \"excluded\": \"allocation, host transfers, and "
        << "in-place device reset copies\",\n"
        << "    \"effective_bandwidth_definition\": \"input plus "
        << "output tensor bytes divided by mean elapsed time; bias "
        << "traffic excluded\"\n"
        << "  },\n"
        << "  \"results\": [\n";

    for (std::size_t index = 0; index < results.size(); ++index) {
        const BenchmarkResult& result = results[index];
        const Statistics& stats = result.statistics;
        std::cout
            << "    {\n"
            << "      \"shape\": {\n"
            << "        \"name\": \""
            << json_escape(result.shape.name) << "\",\n"
            << "        \"batch\": "
            << result.shape.batch_size << ",\n"
            << "        \"channels\": "
            << result.shape.channel_count << ",\n"
            << "        \"height\": "
            << result.shape.height << ",\n"
            << "        \"width\": "
            << result.shape.width << ",\n"
            << "        \"elements\": "
            << result.shape.element_count() << "\n"
            << "      },\n"
            << "      \"mode\": \""
            << json_escape(result.mode) << "\",\n"
            << "      \"implementation\": \""
            << json_escape(result.implementation) << "\",\n"
            << "      \"selected_kernel_path\": \""
            << json_escape(result.selected_kernel_path) << "\",\n"
            << "      \"mean_ms\": " << stats.mean_ms << ",\n"
            << "      \"p50_ms\": " << stats.p50_ms << ",\n"
            << "      \"p90_ms\": " << stats.p90_ms << ",\n"
            << "      \"p95_ms\": " << stats.p95_ms << ",\n"
            << "      \"p99_ms\": " << stats.p99_ms << ",\n"
            << "      \"min_ms\": " << stats.min_ms << ",\n"
            << "      \"max_ms\": " << stats.max_ms << ",\n"
            << "      \"effective_tensor_gbps\": "
            << stats.effective_tensor_gbps << ",\n"
            << "      \"speedup_vs_scalar\": ";
        if (std::isfinite(result.speedup_vs_scalar)) {
            std::cout << result.speedup_vs_scalar;
        } else {
            std::cout << "null";
        }
        std::cout
            << ",\n"
            << "      \"max_abs_error\": "
            << result.max_abs_error << "\n"
            << "    }"
            << (index + 1 == results.size() ? "\n" : ",\n");
    }

    std::cout
        << "  ]\n"
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

        int device_count = 0;
        const cudaError_t device_status =
            cudaGetDeviceCount(&device_count);
        if (device_status == cudaErrorNoDevice || device_count == 0) {
            std::cerr
                << "SKIP: CUDA Toolkit is available, "
                << "but no GPU is attached.\n";
            return kCTestSkipReturnCode;
        }
        SILU_CUDA_CHECK(device_status);
        SILU_CUDA_CHECK(cudaSetDevice(0));

        cudaDeviceProp device{};
        SILU_CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
        int driver_version = 0;
        int runtime_version = 0;
        SILU_CUDA_CHECK(cudaDriverGetVersion(&driver_version));
        SILU_CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));

        const std::vector<std::string> implementations =
            selected_implementations(options);
        const std::vector<std::string> modes =
            selected_modes(options);
        std::vector<BenchmarkResult> results;

        for (const ShapeSpec& shape : kResNet18Shapes) {
            const std::vector<__half> host_input =
                make_input(shape.element_count());
            const std::vector<__half> host_bias =
                make_bias(shape.channel_count);
            const std::vector<float> reference =
                make_reference(host_input, host_bias, shape);
            const std::size_t tensor_bytes =
                shape.element_count() * sizeof(__half);
            const std::size_t bias_bytes =
                shape.channel_count * sizeof(__half);

            DeviceBuffer device_source(shape.element_count());
            DeviceBuffer device_working(shape.element_count());
            DeviceBuffer device_output(shape.element_count());
            DeviceBuffer device_bias(shape.channel_count);
            SILU_CUDA_CHECK(cudaMemcpy(
                device_source.get(),
                host_input.data(),
                tensor_bytes,
                cudaMemcpyHostToDevice
            ));
            SILU_CUDA_CHECK(cudaMemcpy(
                device_bias.get(),
                host_bias.data(),
                bias_bytes,
                cudaMemcpyHostToDevice
            ));

            for (const std::string& mode : modes) {
                std::vector<BenchmarkResult> mode_results =
                    benchmark_mode_interleaved(
                        options,
                        shape,
                        mode,
                        implementations,
                        reference,
                        device_source.get(),
                        device_working.get(),
                        device_bias.get(),
                        device_output.get()
                    );
                results.insert(
                    results.end(),
                    std::make_move_iterator(mode_results.begin()),
                    std::make_move_iterator(mode_results.end())
                );
            }
        }

        add_scalar_speedups(&results);
        print_json(
            options,
            device,
            driver_version,
            runtime_version,
            results
        );
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        print_usage(std::cerr, argv[0]);
        return 2;
    }
}