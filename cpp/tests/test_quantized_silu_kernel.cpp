#include "silu_benchmark/quantized_silu_kernel.h"

#include "data/act_call_0_golden.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using silu_benchmark::MakeQuantizedSiluKernelParams;
using silu_benchmark::DequantizePiecewiseCodeReference;
using silu_benchmark::PiecewiseOutputQuantizationParams;
using silu_benchmark::QuantizedDataType;
using silu_benchmark::QuantizedSiluScalar;
using silu_benchmark::UniformInputQuantizationParams;

int failures = 0;
int tests = 0;

void Check(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Function>
void ExpectInvalid(Function function, const std::string& expected) {
  try {
    function();
  } catch (const std::invalid_argument& error) {
    Check(std::string(error.what()).find(expected) != std::string::npos,
          "unexpected error: " + std::string(error.what()));
    return;
  }
  throw std::runtime_error("expected invalid_argument containing: " + expected);
}

auto GoldenParams() {
  return MakeQuantizedSiluKernelParams(
      UniformInputQuantizationParams{
          QuantizedDataType::kUInt8,
          silu_benchmark::golden::kInputScale,
          silu_benchmark::golden::kInputZeroPoint,
          0,
          255,
      },
      PiecewiseOutputQuantizationParams{
          QuantizedDataType::kUInt8,
          silu_benchmark::golden::kVmin,
          silu_benchmark::golden::kVsplit,
          silu_benchmark::golden::kVmax,
          8,
      });
}

void Run(const char* name, const std::function<void()>& function) {
  ++tests;
  try {
    function();
    std::cout << "PASS " << name << '\n';
  } catch (const std::exception& error) {
    ++failures;
    std::cerr << "FAIL " << name << ": " << error.what() << '\n';
  }
}

void TestGoldenExactEquality() {
  const auto params = GoldenParams();
  std::vector<std::uint8_t> actual(silu_benchmark::golden::kInputCodes.size());
  QuantizedSiluScalar(silu_benchmark::golden::kInputCodes.data(), actual.data(),
                      actual.size(), params);
  std::vector<std::size_t> mismatch_indices;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (actual[index] != silu_benchmark::golden::kExpectedOutputCodes[index]) {
      mismatch_indices.push_back(index);
    }
  }
  if (!mismatch_indices.empty()) {
    const auto index = mismatch_indices.front();
    throw std::runtime_error(
        "golden mismatch count=" + std::to_string(mismatch_indices.size()) +
        ", first index=" + std::to_string(index) +
        ", input=" + std::to_string(silu_benchmark::golden::kInputCodes[index]) +
        ", expected=" +
        std::to_string(silu_benchmark::golden::kExpectedOutputCodes[index]) +
        ", actual=" + std::to_string(actual[index]));
  }
  Check(params.lower_scale == silu_benchmark::golden::kLowerScale,
        "lower scale differs from frozen manifest");
  Check(params.upper_scale == silu_benchmark::golden::kUpperScale,
        "upper scale differs from frozen manifest");
  Check(params.lower_zero_point == silu_benchmark::golden::kLowerZeroPoint,
        "lower zero point differs from frozen manifest");
  Check(params.upper_zero_point == silu_benchmark::golden::kUpperZeroPoint,
        "upper zero point differs from frozen manifest");
}

void TestPostSiluReferenceProbes() {
  const auto params = GoldenParams();
  for (std::size_t index = 0;
       index < silu_benchmark::golden::kPostSiluReferenceValues.size(); ++index) {
    const auto actual = silu_benchmark::QuantizePostSiluReference(
        silu_benchmark::golden::kPostSiluReferenceValues[index], params);
    Check(actual == silu_benchmark::golden::kPostSiluExpectedCodes[index],
          "post-SiLU probe mismatch at index " + std::to_string(index));
  }
  Check(silu_benchmark::QuantizePostSiluReference(params.output.vmin - 1.0, params) == 0,
        "lower saturation must produce code 0");
  Check(silu_benchmark::QuantizePostSiluReference(params.output.vmax + 1.0, params) == 255,
        "upper saturation must produce code 255");
  Check(silu_benchmark::QuantizePostSiluReference(params.output.vsplit, params) >= 128,
        "Vsplit must belong to upper segment");
}

void TestPiecewiseDequantization() {
  const auto params = GoldenParams();
  for (std::int32_t code = 0; code <= 255; ++code) {
    const double reconstructed =
        code <= 127
            ? static_cast<double>(code - params.lower_zero_point) * params.lower_scale
            : static_cast<double>(code - params.upper_zero_point) * params.upper_scale;
    const float expected = static_cast<float>(std::max(
        params.output.vmin, std::min(reconstructed, params.output.vmax)));
    const float actual = DequantizePiecewiseCodeReference(
        static_cast<std::uint8_t>(code), params);
    Check(actual == expected,
          "piecewise dequantization mismatch at code " + std::to_string(code));
  }
}

void TestDeterminismAndInPlace() {
  const auto params = GoldenParams();
  std::vector<std::uint8_t> first(silu_benchmark::golden::kInputCodes.size());
  std::vector<std::uint8_t> second(first.size());
  QuantizedSiluScalar(silu_benchmark::golden::kInputCodes.data(), first.data(), first.size(), params);
  QuantizedSiluScalar(silu_benchmark::golden::kInputCodes.data(), second.data(), second.size(), params);
  Check(first == second, "two scalar calls must be bit-exact");
  auto in_place = std::vector<std::uint8_t>(silu_benchmark::golden::kInputCodes.begin(),
                                           silu_benchmark::golden::kInputCodes.end());
  QuantizedSiluScalar(in_place.data(), in_place.data(), in_place.size(), params);
  Check(in_place == first, "in-place result must match out-of-place result");

  std::array<std::uint8_t, 4> overlap{1, 2, 3, 4};
  ExpectInvalid(
      [&] { QuantizedSiluScalar(overlap.data(), overlap.data() + 1, 3, params); },
      "partially overlap");
}

void TestValidation() {
  const auto output = PiecewiseOutputQuantizationParams{
      QuantizedDataType::kUInt8,
      silu_benchmark::golden::kVmin,
      silu_benchmark::golden::kVsplit,
      silu_benchmark::golden::kVmax,
      8,
  };
  auto input = UniformInputQuantizationParams{
      QuantizedDataType::kUInt8,
      silu_benchmark::golden::kInputScale,
      silu_benchmark::golden::kInputZeroPoint,
      0,
      255,
  };
  input.scale = 0.0;
  ExpectInvalid([&] { MakeQuantizedSiluKernelParams(input, output); }, "input scale");
  input.scale = silu_benchmark::golden::kInputScale;
  input.dtype = QuantizedDataType::kInt8;
  ExpectInvalid([&] { MakeQuantizedSiluKernelParams(input, output); }, "input dtype");
  input.dtype = QuantizedDataType::kUInt8;
  auto invalid_output = output;
  invalid_output.bits = 7;
  ExpectInvalid([&] { MakeQuantizedSiluKernelParams(input, invalid_output); }, "bits");
  invalid_output = output;
  invalid_output.vsplit = 0.0;
  ExpectInvalid([&] { MakeQuantizedSiluKernelParams(input, invalid_output); }, "Vmin");

  auto params = GoldenParams();
  params.lookup_table[0] ^= 1;
  ExpectInvalid(
      [&] { silu_benchmark::ValidateQuantizedSiluKernelParams(params); },
      "lookup_table");
  params = GoldenParams();
  std::uint8_t value = 0;
  ExpectInvalid([&] { QuantizedSiluScalar(nullptr, &value, 1, params); }, "input buffer");
  ExpectInvalid([&] { QuantizedSiluScalar(&value, nullptr, 1, params); }, "output buffer");
  QuantizedSiluScalar(nullptr, nullptr, 0, params);
}

}  // namespace

int main() {
  Run("golden exact code equality", TestGoldenExactEquality);
  Run("post-SiLU boundaries ties and saturation", TestPostSiluReferenceProbes);
  Run("piecewise dequantization exact equality", TestPiecewiseDequantization);
  Run("determinism and aliasing", TestDeterminismAndInPlace);
  Run("parameter and buffer validation", TestValidation);
  std::cout << (tests - failures) << "/" << tests << " tests passed\n";
  return failures == 0 ? 0 : 1;
}
