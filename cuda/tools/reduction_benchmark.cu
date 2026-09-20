#include "silu_cuda/cuda_check.h"
#include "silu_cuda/reduction.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
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
constexpr float kInputValue = 0.03125F;
constexpr float kCorrectnessTolerance = 1.0e-4F;

struct Options {
    std::size_t element_count = 1U << 24;
    int warmup_iterations = 20;
    int measured_iterations = 200;
    std::string implementation = "both";
    bool show_help = false;
};

struct Statistics {
    double mean_ms = 0.0;
    double p50_ms = 0.0;
    double p90_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double effective_input_gbps = 0.0;
};

struct BenchmarkResult {
    std::string implementation;
    Statistics statistics;
    float output = 0.0F;
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
        << "  --elements N             Input elements; default 16777216\n"
        << "  --warmup N               Warm-up launches; default 20\n"
        << "  --iterations N           Measured launches; default 200\n"
        << "  --implementation NAME    atomic, hierarchical, or both\n"
        << "  --help                    Show this message\n";
}

std::size_t parse_size(const std::string& text, const char* option) {
    std::size_t consumed = 0;
    const unsigned long long parsed = std::stoull(text, &consumed);
    if (consumed != text.size()
        || parsed == 0
        || parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(
            std::string(option) + " requires a positive integer"
        );
    }
    return static_cast<std::size_t>(parsed);
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
        if (argument == "--elements") {
            options.element_count =
                parse_size(value, "--elements");
        } else if (argument == "--warmup") {
            options.warmup_iterations =
                parse_count(value, "--warmup", true);
        } else if (argument == "--iterations") {
            options.measured_iterations =
                parse_count(value, "--iterations", false);
        } else if (argument == "--implementation") {
            if (value != "atomic"
                && value != "hierarchical"
                && value != "both") {
                throw std::invalid_argument(
                    "--implementation must be atomic, "
                    "hierarchical, or both"
                );
            }
            options.implementation = value;
        } else {
            throw std::invalid_argument(
                "Unknown option: " + argument
            );
        }
    }

    return options;
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
    std::size_t input_bytes
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
    result.effective_input_gbps =
        result.mean_ms > 0.0
        ? static_cast<double>(input_bytes)
            / result.mean_ms
            / 1.0e6
        : 0.0;
    return result;
}

BenchmarkResult benchmark_implementation(
    const std::string& name,
    const Options& options,
    std::size_t input_bytes,
    float* device_output,
    const std::function<cudaError_t()>& launch
) {
    for (int iteration = 0;
         iteration < options.warmup_iterations;
         ++iteration) {
        SILU_CUDA_CHECK(launch());
    }
    SILU_CUDA_CHECK(cudaDeviceSynchronize());

    CudaEvent start;
    CudaEvent stop;
    std::vector<float> samples_ms;
    samples_ms.reserve(
        static_cast<std::size_t>(options.measured_iterations)
    );

    for (int iteration = 0;
         iteration < options.measured_iterations;
         ++iteration) {
        SILU_CUDA_CHECK(cudaEventRecord(start.get()));
        SILU_CUDA_CHECK(launch());
        SILU_CUDA_CHECK(cudaEventRecord(stop.get()));
        SILU_CUDA_CHECK(cudaEventSynchronize(stop.get()));

        float elapsed_ms = 0.0F;
        SILU_CUDA_CHECK(cudaEventElapsedTime(
            &elapsed_ms,
            start.get(),
            stop.get()
        ));
        samples_ms.push_back(elapsed_ms);
    }

    float output = 0.0F;
    SILU_CUDA_CHECK(cudaMemcpy(
        &output,
        device_output,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    return {
        name,
        summarize(std::move(samples_ms), input_bytes),
        output,
    };
}

void validate_output(
    const BenchmarkResult& result,
    float expected
) {
    const float absolute_error =
        std::abs(result.output - expected);
    if (absolute_error > kCorrectnessTolerance) {
        throw std::runtime_error(
            result.implementation
            + " output failed correctness validation"
        );
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
    const std::size_t input_bytes =
        options.element_count * sizeof(float);

    std::cout
        << std::fixed
        << std::setprecision(6)
        << "{\n"
        << "  \"schema_version\": 1,\n"
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
        << "    \"elements\": " << options.element_count << ",\n"
        << "    \"input_bytes\": " << input_bytes << ",\n"
        << "    \"warmup_iterations\": "
        << options.warmup_iterations << ",\n"
        << "    \"measured_iterations\": "
        << options.measured_iterations << ",\n"
        << "    \"timer\": \"CUDA events on the default stream\",\n"
        << "    \"excluded\": "
        << "\"allocation, workspace creation, and host-to-device copy\",\n"
        << "    \"effective_bandwidth_definition\": "
        << "\"input bytes divided by mean elapsed time\"\n"
        << "  },\n"
        << "  \"results\": [\n";

    for (std::size_t index = 0; index < results.size(); ++index) {
        const BenchmarkResult& result = results[index];
        const Statistics& stats = result.statistics;

        std::cout
            << "    {\n"
            << "      \"implementation\": \""
            << json_escape(result.implementation) << "\",\n"
            << "      \"mean_ms\": " << stats.mean_ms << ",\n"
            << "      \"p50_ms\": " << stats.p50_ms << ",\n"
            << "      \"p90_ms\": " << stats.p90_ms << ",\n"
            << "      \"p95_ms\": " << stats.p95_ms << ",\n"
            << "      \"p99_ms\": " << stats.p99_ms << ",\n"
            << "      \"min_ms\": " << stats.min_ms << ",\n"
            << "      \"max_ms\": " << stats.max_ms << ",\n"
            << "      \"effective_input_gbps\": "
            << stats.effective_input_gbps << ",\n"
            << "      \"validated_output\": "
            << result.output << "\n"
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

        if (options.element_count
            > std::numeric_limits<std::size_t>::max()
                / sizeof(float)) {
            throw std::invalid_argument(
                "--elements exceeds addressable input size"
            );
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

        const std::size_t input_bytes =
            options.element_count * sizeof(float);
        const std::vector<float> host_input(
            options.element_count,
            kInputValue
        );
        const float expected = static_cast<float>(
            static_cast<double>(options.element_count)
            * static_cast<double>(kInputValue)
        );

        DeviceBuffer device_input(options.element_count);
        DeviceBuffer atomic_output(1);
        DeviceBuffer hierarchical_output(1);

        const std::size_t workspace_size_bytes =
            silu_cuda::hierarchical_reduce_workspace_size_bytes(
                options.element_count
            );
        if (workspace_size_bytes
            == std::numeric_limits<std::size_t>::max()) {
            throw std::invalid_argument(
                "Hierarchical workspace size overflow"
            );
        }

        const std::size_t workspace_elements =
            std::max<std::size_t>(
                1,
                (workspace_size_bytes + sizeof(float) - 1)
                    / sizeof(float)
            );
        DeviceBuffer workspace(workspace_elements);

        SILU_CUDA_CHECK(cudaMemcpy(
            device_input.get(),
            host_input.data(),
            input_bytes,
            cudaMemcpyHostToDevice
        ));

        std::vector<BenchmarkResult> results;

        if (options.implementation == "atomic"
            || options.implementation == "both") {
            BenchmarkResult result = benchmark_implementation(
                "atomic_shared_memory",
                options,
                input_bytes,
                atomic_output.get(),
                [&]() {
                    return silu_cuda::launch_reduce_sum_atomic(
                        device_input.get(),
                        atomic_output.get(),
                        options.element_count
                    );
                }
            );
            validate_output(result, expected);
            results.push_back(std::move(result));
        }

        if (options.implementation == "hierarchical"
            || options.implementation == "both") {
            BenchmarkResult result = benchmark_implementation(
                "hierarchical_multi_pass",
                options,
                input_bytes,
                hierarchical_output.get(),
                [&]() {
                    return silu_cuda::launch_reduce_sum_hierarchical(
                        device_input.get(),
                        hierarchical_output.get(),
                        options.element_count,
                        workspace.get(),
                        workspace_size_bytes
                    );
                }
            );
            validate_output(result, expected);
            results.push_back(std::move(result));
        }

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
