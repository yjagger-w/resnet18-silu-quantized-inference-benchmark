#include "silu_benchmark/quantized_silu_kernel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace silu_benchmark {
namespace {

constexpr std::int32_t kLowerCodeStart = 0;
constexpr std::int32_t kLowerCodeEnd = 127;
constexpr std::int32_t kUpperCodeStart = 128;
constexpr std::int32_t kUpperCodeEnd = 255;

void Require(bool condition, const char* message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

std::int32_t RoundNearestEven(double value) {
  Require(std::isfinite(value), "rounding input must be finite");
  const double lower = std::floor(value);
  const double fraction = value - lower;
  if (fraction < 0.5) {
    return static_cast<std::int32_t>(lower);
  }
  if (fraction > 0.5) {
    return static_cast<std::int32_t>(lower + 1.0);
  }
  const auto lower_integer = static_cast<std::int64_t>(lower);
  return static_cast<std::int32_t>(
      (lower_integer % 2 == 0) ? lower_integer : lower_integer + 1);
}

float SiluFloat32(float value) {
  if (value >= 0.0F) {
    return value / (1.0F + std::exp(-value));
  }
  const float exp_value = std::exp(value);
  return value * exp_value / (1.0F + exp_value);
}

bool RangesOverlap(const std::uint8_t* input,
                   const std::uint8_t* output,
                   std::size_t element_count) {
  if (element_count == 0 || input == output) {
    return false;
  }
  const auto input_start = reinterpret_cast<std::uintptr_t>(input);
  const auto output_start = reinterpret_cast<std::uintptr_t>(output);
  const auto address_max = std::numeric_limits<std::uintptr_t>::max();
  Require(element_count <= address_max - input_start,
          "input buffer address range overflows uintptr_t");
  Require(element_count <= address_max - output_start,
          "output buffer address range overflows uintptr_t");
  const auto input_end = input_start + element_count;
  const auto output_end = output_start + element_count;
  return input_start < output_end && output_start < input_end;
}

void ValidateInput(const UniformInputQuantizationParams& input) {
  Require(input.dtype == QuantizedDataType::kUInt8,
          "input dtype must be uint8; int8 is unsupported in v1.0");
  Require(std::isfinite(input.scale) && input.scale > 0.0,
          "input scale must be finite and greater than zero");
  Require(input.qmin == 0 && input.qmax == 255,
          "input code range must be exactly [0, 255]");
  Require(input.zero_point >= input.qmin && input.zero_point <= input.qmax,
          "input zero_point must lie in [qmin, qmax]");
  const double low = static_cast<double>(input.qmin - input.zero_point) * input.scale;
  const double high = static_cast<double>(input.qmax - input.zero_point) * input.scale;
  Require(std::isfinite(low) && std::isfinite(high) &&
              low >= -std::numeric_limits<float>::max() &&
              high <= std::numeric_limits<float>::max(),
          "input dequantized range must be representable as float32");
}

void ValidateOutput(const PiecewiseOutputQuantizationParams& output) {
  Require(output.dtype == QuantizedDataType::kUInt8,
          "output dtype must be uint8; int8 is unsupported in v1.0");
  Require(output.bits == 8,
          "output bits must be exactly 8 for uint8 transport");
  Require(std::isfinite(output.vmin) && std::isfinite(output.vsplit) &&
              std::isfinite(output.vmax),
          "Vmin, Vsplit, and Vmax must be finite");
  Require(output.vmin < 0.0 && output.vsplit > 0.0 &&
              output.vsplit < output.vmax,
          "require Vmin < 0 < Vsplit < Vmax");
}

}  // namespace

QuantizedSiluKernelParams MakeQuantizedSiluKernelParams(
    const UniformInputQuantizationParams& input,
    const PiecewiseOutputQuantizationParams& output) {
  ValidateInput(input);
  ValidateOutput(output);

  QuantizedSiluKernelParams params;
  params.input = input;
  params.output = output;
  params.lower_scale =
      (output.vsplit - output.vmin) / static_cast<double>(kLowerCodeEnd);
  params.upper_scale =
      (output.vmax - output.vsplit) /
      static_cast<double>(kUpperCodeEnd - kUpperCodeStart);
  Require(std::isfinite(params.lower_scale) && params.lower_scale > 0.0,
          "derived lower scale must be finite and greater than zero");
  Require(std::isfinite(params.upper_scale) && params.upper_scale > 0.0,
          "derived upper scale must be finite and greater than zero");

  params.lower_zero_point =
      RoundNearestEven(-output.vmin / params.lower_scale);
  const auto rounded_upper = RoundNearestEven(
      static_cast<double>(kUpperCodeStart) - output.vsplit / params.upper_scale);
  const double lower_endpoint =
      static_cast<double>(kLowerCodeEnd - params.lower_zero_point) *
      params.lower_scale;
  const auto boundary_guard = static_cast<std::int32_t>(std::floor(
      static_cast<double>(kUpperCodeStart) - lower_endpoint / params.upper_scale));
  params.upper_zero_point = std::min(rounded_upper, boundary_guard);

  for (std::size_t code = 0; code < params.lookup_table.size(); ++code) {
    const float dequantized = static_cast<float>(
        (static_cast<std::int32_t>(code) - input.zero_point) * input.scale);
    const float post_silu = SiluFloat32(dequantized);
    params.lookup_table[code] =
        QuantizePostSiluReference(static_cast<double>(post_silu), params);
  }
  ValidateQuantizedSiluKernelParams(params);
  return params;
}

void ValidateQuantizedSiluKernelParams(const QuantizedSiluKernelParams& params) {
  ValidateInput(params.input);
  ValidateOutput(params.output);
  const double expected_lower =
      (params.output.vsplit - params.output.vmin) /
      static_cast<double>(kLowerCodeEnd);
  const double expected_upper =
      (params.output.vmax - params.output.vsplit) /
      static_cast<double>(kUpperCodeEnd - kUpperCodeStart);
  Require(params.lower_scale == expected_lower,
          "lower_scale does not match the canonical derived value");
  Require(params.upper_scale == expected_upper,
          "upper_scale does not match the canonical derived value");
  Require(params.lower_zero_point ==
              RoundNearestEven(-params.output.vmin / expected_lower),
          "lower_zero_point does not match the canonical derived value");
  const auto rounded_upper = RoundNearestEven(
      static_cast<double>(kUpperCodeStart) - params.output.vsplit / expected_upper);
  const double lower_endpoint =
      static_cast<double>(kLowerCodeEnd - params.lower_zero_point) * expected_lower;
  const auto boundary_guard = static_cast<std::int32_t>(std::floor(
      static_cast<double>(kUpperCodeStart) - lower_endpoint / expected_upper));
  Require(params.upper_zero_point == std::min(rounded_upper, boundary_guard),
          "upper_zero_point does not match the canonical boundary-guarded value");
  for (std::size_t code = 0; code < params.lookup_table.size(); ++code) {
    const float dequantized = static_cast<float>(
        (static_cast<std::int32_t>(code) - params.input.zero_point) *
        params.input.scale);
    const auto expected_code = QuantizePostSiluReference(
        static_cast<double>(SiluFloat32(dequantized)), params);
    if (params.lookup_table[code] != expected_code) {
      throw std::invalid_argument(
          "lookup_table does not match canonical parameters at input code " +
          std::to_string(code));
    }
  }
}

std::uint8_t QuantizePostSiluReference(
    double post_silu_value,
    const QuantizedSiluKernelParams& params) {
  Require(std::isfinite(post_silu_value),
          "post-SiLU reference value must be finite");
  const double clipped = std::max(
      params.output.vmin, std::min(post_silu_value, params.output.vmax));
  if (clipped < params.output.vsplit) {
    const auto rounded = RoundNearestEven(
        clipped / params.lower_scale + params.lower_zero_point);
    return static_cast<std::uint8_t>(
        std::max(kLowerCodeStart, std::min(rounded, kLowerCodeEnd)));
  }
  const auto rounded = RoundNearestEven(
      clipped / params.upper_scale + params.upper_zero_point);
  return static_cast<std::uint8_t>(
      std::max(kUpperCodeStart, std::min(rounded, kUpperCodeEnd)));
}

float DequantizePiecewiseCodeReference(
    std::uint8_t code,
    const QuantizedSiluKernelParams& params) noexcept {
  const auto integer_code = static_cast<std::int32_t>(code);
  const double reconstructed =
      integer_code <= kLowerCodeEnd
          ? static_cast<double>(integer_code - params.lower_zero_point) *
                params.lower_scale
          : static_cast<double>(integer_code - params.upper_zero_point) *
                params.upper_scale;
  const double clipped = std::max(
      params.output.vmin, std::min(reconstructed, params.output.vmax));
  return static_cast<float>(clipped);
}

void QuantizedSiluScalar(
    const std::uint8_t* input,
    std::uint8_t* output,
    std::size_t element_count,
    const QuantizedSiluKernelParams& params) {
  ValidateQuantizedSiluKernelParams(params);
  if (element_count == 0) {
    return;
  }
  Require(input != nullptr, "input buffer must not be null when element_count > 0");
  Require(output != nullptr, "output buffer must not be null when element_count > 0");
  Require(!RangesOverlap(input, output, element_count),
          "input/output buffers may be identical for in-place use but must not partially overlap");
  QuantizedSiluScalarUnchecked(input, output, element_count, params);
}

void QuantizedSiluScalarUnchecked(
    const std::uint8_t* input,
    std::uint8_t* output,
    std::size_t element_count,
    const QuantizedSiluKernelParams& params) noexcept {
  for (std::size_t index = 0; index < element_count; ++index) {
    output[index] = params.lookup_table[input[index]];
  }
}

const char* ScalarImplementationName() noexcept {
  return "portable_scalar_uint8_lut";
}

}  // namespace silu_benchmark
