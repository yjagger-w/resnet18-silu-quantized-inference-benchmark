#include "silu_benchmark/quantized_silu_kernel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef SILU_KERNEL_COMPILER_ID
#define SILU_KERNEL_COMPILER_ID "unknown"
#endif
#ifndef SILU_KERNEL_COMPILER_VERSION
#define SILU_KERNEL_COMPILER_VERSION "unknown"
#endif
#ifndef SILU_KERNEL_BUILD_TYPE
#define SILU_KERNEL_BUILD_TYPE "unknown"
#endif
#ifndef SILU_KERNEL_LIBRARY_FLAGS
#define SILU_KERNEL_LIBRARY_FLAGS "unknown"
#endif

namespace fs = std::filesystem;

namespace {

using silu_benchmark::MakeQuantizedSiluKernelParams;
using silu_benchmark::PiecewiseOutputQuantizationParams;
using silu_benchmark::QuantizedDataType;
using silu_benchmark::QuantizedSiluScalar;
using silu_benchmark::QuantizedSiluScalarUnchecked;
using silu_benchmark::UniformInputQuantizationParams;

struct Options {
  fs::path fixture;
  fs::path output_dir;
  std::size_t input_size = 0;
  std::size_t warmup = 0;
  std::size_t iterations = 0;
  std::uint32_t seed = 0;
  std::string implementation;
};

struct Fixture {
  UniformInputQuantizationParams input;
  PiecewiseOutputQuantizationParams output;
  std::vector<std::uint8_t> input_codes;
  std::vector<std::uint8_t> expected_codes;
};

std::string ReadText(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::invalid_argument("fixture does not exist or is not readable: " + path.string());
  }
  return std::string(std::istreambuf_iterator<char>(stream),
                     std::istreambuf_iterator<char>());
}

double ExtractNumber(const std::string& text, const std::string& key) {
  const std::regex pattern(
      "\\\"" + key + "\\\"\\s*:\\s*([-+0-9.eE]+)");
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) {
    throw std::invalid_argument("fixture is missing numeric field: " + key);
  }
  const double value = std::stod(match[1].str());
  if (!std::isfinite(value)) {
    throw std::invalid_argument("fixture numeric field must be finite: " + key);
  }
  return value;
}

std::vector<std::uint8_t> ExtractByteArray(
    const std::string& text, const std::string& key) {
  const std::regex pattern(
      "\\\"" + key + "\\\"\\s*:\\s*\\[([\\s\\S]*?)\\]");
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) {
    throw std::invalid_argument("fixture is missing byte array: " + key);
  }
  std::vector<std::uint8_t> values;
  const std::regex integer_pattern("[0-9]+");
  const std::string body = match[1].str();
  for (auto iterator = std::sregex_iterator(body.begin(), body.end(), integer_pattern);
       iterator != std::sregex_iterator(); ++iterator) {
    const auto parsed = std::stoul(iterator->str());
    if (parsed > 255) {
      throw std::invalid_argument("fixture byte array value is outside [0, 255]: " + key);
    }
    values.push_back(static_cast<std::uint8_t>(parsed));
  }
  if (values.empty()) {
    throw std::invalid_argument("fixture byte array must not be empty: " + key);
  }
  return values;
}

Fixture LoadFixture(const fs::path& path) {
  const std::string text = ReadText(path);
  if (text.find("quantized-silu-kernel-golden/v1") == std::string::npos) {
    throw std::invalid_argument(
        "fixture schema_version must be quantized-silu-kernel-golden/v1");
  }
  Fixture fixture;
  fixture.input = {
      QuantizedDataType::kUInt8,
      ExtractNumber(text, "scale"),
      static_cast<std::int32_t>(ExtractNumber(text, "zero_point")),
      static_cast<std::int32_t>(ExtractNumber(text, "qmin")),
      static_cast<std::int32_t>(ExtractNumber(text, "qmax")),
  };
  fixture.output = {
      QuantizedDataType::kUInt8,
      ExtractNumber(text, "vmin"),
      ExtractNumber(text, "vsplit"),
      ExtractNumber(text, "vmax"),
      static_cast<std::int32_t>(ExtractNumber(text, "bits")),
  };
  fixture.input_codes = ExtractByteArray(text, "quantized_input_codes");
  fixture.expected_codes = ExtractByteArray(text, "expected_output_codes");
  if (fixture.input_codes.size() != fixture.expected_codes.size()) {
    throw std::invalid_argument("fixture input and expected arrays must have equal length");
  }
  return fixture;
}

std::size_t ParseSize(const std::string& value, const char* option, bool allow_zero) {
  std::size_t consumed = 0;
  const auto parsed = std::stoull(value, &consumed);
  if (consumed != value.size() || (!allow_zero && parsed == 0)) {
    throw std::invalid_argument(std::string(option) +
                                (allow_zero ? " must be a non-negative integer"
                                            : " must be a positive integer"));
  }
  return static_cast<std::size_t>(parsed);
}

void PrintHelp() {
  std::cout
      << "Standalone quantized SiLU kernel microbenchmark only.\n"
      << "Required: --fixture PATH --size N --warmup N --iterations N --seed N "
         "--implementation auto|scalar --output-dir PATH\n";
}

Options ParseArgs(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (option == "--help") {
      PrintHelp();
      std::exit(0);
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for " + option);
    }
    const std::string value = argv[++index];
    if (option == "--fixture") {
      options.fixture = value;
    } else if (option == "--output-dir") {
      options.output_dir = value;
    } else if (option == "--size") {
      options.input_size = ParseSize(value, "--size", false);
    } else if (option == "--warmup") {
      options.warmup = ParseSize(value, "--warmup", true);
    } else if (option == "--iterations") {
      options.iterations = ParseSize(value, "--iterations", false);
    } else if (option == "--seed") {
      const auto parsed = ParseSize(value, "--seed", true);
      if (parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("--seed must fit in uint32");
      }
      options.seed = static_cast<std::uint32_t>(parsed);
    } else if (option == "--implementation") {
      options.implementation = value;
    } else {
      throw std::invalid_argument("unknown option: " + option);
    }
  }
  if (options.fixture.empty() || options.output_dir.empty() ||
      options.input_size == 0 || options.iterations == 0 ||
      options.implementation.empty()) {
    throw std::invalid_argument("all required benchmark options must be provided; use --help");
  }
  if (options.implementation == "avx2") {
    throw std::invalid_argument(
        "AVX2 is intentionally deferred in v1.0; use --implementation scalar or auto");
  }
  if (options.implementation != "scalar" && options.implementation != "auto") {
    throw std::invalid_argument("--implementation must be auto or scalar");
  }
  return options;
}

fs::path FindRepositoryRoot(fs::path current) {
  current = fs::absolute(current).lexically_normal();
  while (!current.empty()) {
    if (fs::exists(current / ".git") && fs::exists(current / "cpp/CMakeLists.txt")) {
      return current;
    }
    const fs::path parent = current.parent_path();
    if (parent == current) {
      break;
    }
    current = parent;
  }
  throw std::runtime_error("cannot locate repository root from current working directory");
}

fs::path ValidateOutputDirectory(const fs::path& requested, const fs::path& root) {
  const fs::path output = fs::absolute(requested).lexically_normal();
  const fs::path allowed_root = (root / "results/benchmarks").lexically_normal();
  const fs::path relative = output.lexically_relative(allowed_root);
  const std::string first = relative.empty() ? "" : (*relative.begin()).string();
  if (relative.empty() || first == ".." || first == "." ||
      first.rfind("v1.0_silu_kernel", 0) != 0) {
    throw std::invalid_argument(
        "--output-dir must be inside results/benchmarks/v1.0_silu_kernel*");
  }
  if (fs::exists(output) && !fs::is_empty(output)) {
    throw std::invalid_argument("--output-dir already exists and is not empty: " + output.string());
  }
  return output;
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream result;
  for (char character : value) {
    if (character == '\\' || character == '"') {
      result << '\\';
    }
    result << character;
  }
  return result.str();
}

std::string CsvEscape(const std::string& value) {
  if (value.find_first_of(",\"\r\n") == std::string::npos) {
    return value;
  }
  std::string escaped = "\"";
  for (char character : value) {
    if (character == '"') {
      escaped += '"';
    }
    escaped += character;
  }
  escaped += '"';
  return escaped;
}

std::string CpuDescription() {
  const char* description = std::getenv("PROCESSOR_IDENTIFIER");
  return description ? description : "unavailable";
}

double Percentile(std::vector<double> values, double percentile) {
  std::sort(values.begin(), values.end());
  const double position = percentile * static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(std::floor(position));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
  const double weight = position - static_cast<double>(lower);
  return values[lower] * (1.0 - weight) + values[upper] * weight;
}

void WriteReports(const fs::path& output_dir,
                  const Options& options,
                  const fs::path& fixture_path,
                  double mean_ns,
                  double p50_ns,
                  double p95_ns,
                  double min_ns,
                  double max_ns) {
  const double elements_per_second =
      static_cast<double>(options.input_size) / (mean_ns / 1.0e9);
  const std::string implementation = silu_benchmark::ScalarImplementationName();
  const std::string cpu = CpuDescription();
  const std::string compiler =
      std::string(SILU_KERNEL_COMPILER_ID) + " " + SILU_KERNEL_COMPILER_VERSION;
  const std::string flags =
      std::string("C++17; ") + SILU_KERNEL_LIBRARY_FLAGS;

  std::ofstream json(output_dir / "benchmark_results.json");
  json << std::setprecision(17)
       << "{\n"
       << "  \"schema_version\": \"silu-kernel-microbenchmark/v1\",\n"
       << "  \"scope\": \"microbenchmark measurements for the standalone SiLU kernel only\",\n"
       << "  \"implementation\": \"" << implementation << "\",\n"
       << "  \"input_size_elements\": " << options.input_size << ",\n"
       << "  \"warmup_iterations\": " << options.warmup << ",\n"
       << "  \"timed_iterations\": " << options.iterations << ",\n"
       << "  \"seed\": " << options.seed << ",\n"
       << "  \"fixture\": \"" << JsonEscape(fixture_path.string()) << "\",\n"
       << "  \"mean_latency_ns\": " << mean_ns << ",\n"
       << "  \"p50_latency_ns\": " << p50_ns << ",\n"
       << "  \"p95_latency_ns\": " << p95_ns << ",\n"
       << "  \"min_latency_ns\": " << min_ns << ",\n"
       << "  \"max_latency_ns\": " << max_ns << ",\n"
       << "  \"effective_elements_per_second\": " << elements_per_second << ",\n"
       << "  \"compiler\": \"" << JsonEscape(compiler) << "\",\n"
       << "  \"build_flags\": \"" << JsonEscape(flags) << "\",\n"
       << "  \"host_cpu\": \"" << JsonEscape(cpu) << "\"\n"
       << "}\n";

  std::ofstream csv(output_dir / "benchmark_results.csv");
  csv << "scope,implementation,input_size_elements,warmup_iterations,timed_iterations,seed,"
         "mean_latency_ns,p50_latency_ns,p95_latency_ns,min_latency_ns,max_latency_ns,"
         "effective_elements_per_second,compiler,build_type,host_cpu\n";
  csv << std::setprecision(17)
      << "standalone SiLU kernel microbenchmark," << implementation << ','
      << options.input_size << ',' << options.warmup << ',' << options.iterations << ','
      << options.seed << ',' << mean_ns << ',' << p50_ns << ',' << p95_ns << ','
      << min_ns << ',' << max_ns << ',' << elements_per_second << ','
      << CsvEscape(compiler) << ',' << CsvEscape(SILU_KERNEL_BUILD_TYPE) << ','
      << CsvEscape(cpu) << '\n';

  std::ofstream markdown(output_dir / "benchmark_report.md");
  markdown << "# Standalone quantized SiLU kernel microbenchmark\n\n"
           << "These are microbenchmark measurements for the standalone SiLU kernel only. "
              "They are not whole-model latency, accuracy, backend, or hardware-acceleration evidence.\n\n"
           << "| Field | Measurement |\n|---|---:|\n"
           << "| Implementation | " << implementation << " |\n"
           << "| Input elements | " << options.input_size << " |\n"
           << "| Warm-up / timed iterations | " << options.warmup << " / "
           << options.iterations << " |\n"
           << "| Mean latency | " << std::fixed << std::setprecision(2) << mean_ns
           << " ns |\n"
           << "| P50 / P95 latency | " << p50_ns << " / " << p95_ns << " ns |\n"
           << "| Effective elements/s | " << elements_per_second << " |\n"
           << "| Compiler | " << compiler << " |\n"
           << "| Build | " << SILU_KERNEL_BUILD_TYPE << " |\n"
           << "| Host CPU | " << cpu << " |\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseArgs(argc, argv);
    const fs::path root = FindRepositoryRoot(fs::current_path());
    const fs::path output_dir = ValidateOutputDirectory(options.output_dir, root);
    const fs::path fixture_path = fs::absolute(options.fixture).lexically_normal();
    const Fixture fixture = LoadFixture(fixture_path);
    const auto params = MakeQuantizedSiluKernelParams(fixture.input, fixture.output);

    std::vector<std::uint8_t> fixture_actual(fixture.input_codes.size());
    QuantizedSiluScalar(fixture.input_codes.data(), fixture_actual.data(),
                        fixture_actual.size(), params);
    if (fixture_actual != fixture.expected_codes) {
      const auto mismatch = std::mismatch(
          fixture_actual.begin(), fixture_actual.end(), fixture.expected_codes.begin());
      const auto index = static_cast<std::size_t>(mismatch.first - fixture_actual.begin());
      throw std::runtime_error(
          "golden validation failed before timing at index " + std::to_string(index) +
          ": expected=" + std::to_string(fixture.expected_codes[index]) +
          ", actual=" + std::to_string(fixture_actual[index]));
    }

    std::mt19937 generator(options.seed);
    std::uniform_int_distribution<int> distribution(0, 255);
    std::vector<std::uint8_t> input(options.input_size);
    std::vector<std::uint8_t> output(options.input_size);
    for (auto& value : input) {
      value = static_cast<std::uint8_t>(distribution(generator));
    }
    QuantizedSiluScalar(input.data(), output.data(), output.size(), params);
    for (std::size_t index = 0; index < output.size(); ++index) {
      if (output[index] != params.lookup_table[input[index]]) {
        throw std::runtime_error("input validation failed before timing at index " +
                                 std::to_string(index));
      }
    }

    for (std::size_t iteration = 0; iteration < options.warmup; ++iteration) {
      QuantizedSiluScalarUnchecked(input.data(), output.data(), output.size(), params);
    }
    std::vector<double> latencies_ns;
    latencies_ns.reserve(options.iterations);
    for (std::size_t iteration = 0; iteration < options.iterations; ++iteration) {
      const auto start = std::chrono::steady_clock::now();
      QuantizedSiluScalarUnchecked(input.data(), output.data(), output.size(), params);
      const auto end = std::chrono::steady_clock::now();
      latencies_ns.push_back(static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()));
    }
    const double mean_ns =
        std::accumulate(latencies_ns.begin(), latencies_ns.end(), 0.0) /
        static_cast<double>(latencies_ns.size());
    const double p50_ns = Percentile(latencies_ns, 0.50);
    const double p95_ns = Percentile(latencies_ns, 0.95);
    const auto [minimum, maximum] =
        std::minmax_element(latencies_ns.begin(), latencies_ns.end());

    fs::create_directories(output_dir);
    WriteReports(output_dir, options, fixture_path, mean_ns, p50_ns, p95_ns,
                 *minimum, *maximum);
    std::cout << "Standalone SiLU kernel microbenchmark completed\n"
              << "implementation=" << silu_benchmark::ScalarImplementationName() << '\n'
              << "input_size_elements=" << options.input_size << '\n'
              << "mean_latency_ns=" << mean_ns << '\n'
              << "effective_elements_per_second="
              << static_cast<double>(options.input_size) / (mean_ns / 1.0e9) << '\n'
              << "output_dir=" << output_dir.string() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
