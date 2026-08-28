#ifndef SILU_BENCHMARK_QUANTIZED_SILU_KERNEL_H_
#define SILU_BENCHMARK_QUANTIZED_SILU_KERNEL_H_

#include <array>
#include <cstddef>
#include <cstdint>

namespace silu_benchmark {

enum class QuantizedDataType {
  kUInt8,
  kInt8,
};

struct UniformInputQuantizationParams {
  QuantizedDataType dtype = QuantizedDataType::kUInt8;
  double scale = 0.0;
  std::int32_t zero_point = 0;
  std::int32_t qmin = 0;
  std::int32_t qmax = 255;
};

struct PiecewiseOutputQuantizationParams {
  QuantizedDataType dtype = QuantizedDataType::kUInt8;
  double vmin = 0.0;
  double vsplit = 0.0;
  double vmax = 0.0;
  std::int32_t bits = 8;
};

struct QuantizedSiluKernelParams {
  UniformInputQuantizationParams input;
  PiecewiseOutputQuantizationParams output;
  double lower_scale = 0.0;
  double upper_scale = 0.0;
  std::int32_t lower_zero_point = 0;
  std::int32_t upper_zero_point = 0;
  std::array<std::uint8_t, 256> lookup_table{};
};

// Validates and freezes all derived parameters and the 256-entry scalar SiLU
// lookup table. Input dequantization and SiLU are evaluated once during setup;
// timed kernel calls contain only code lookup and explicit uint8 output writes.
QuantizedSiluKernelParams MakeQuantizedSiluKernelParams(
    const UniformInputQuantizationParams& input,
    const PiecewiseOutputQuantizationParams& output);

void ValidateQuantizedSiluKernelParams(const QuantizedSiluKernelParams& params);

// Canonical post-SiLU reference quantizer. Values are clipped to [vmin, vmax],
// Vsplit belongs to the upper segment, and ties round to nearest even.
std::uint8_t QuantizePostSiluReference(
    double post_silu_value,
    const QuantizedSiluKernelParams& params);

// Portable scalar uint8 -> uint8 kernel. Exact in-place operation is supported;
// any other overlapping input/output ranges are rejected.
void QuantizedSiluScalar(
    const std::uint8_t* input,
    std::uint8_t* output,
    std::size_t element_count,
    const QuantizedSiluKernelParams& params);

// Timing-only entry point for already validated, non-null, non-overlapping
// buffers. The benchmark validates once before invoking this uninstrumented
// loop. Application code should normally call QuantizedSiluScalar.
void QuantizedSiluScalarUnchecked(
    const std::uint8_t* input,
    std::uint8_t* output,
    std::size_t element_count,
    const QuantizedSiluKernelParams& params) noexcept;

const char* ScalarImplementationName() noexcept;

}  // namespace silu_benchmark

#endif  // SILU_BENCHMARK_QUANTIZED_SILU_KERNEL_H_
