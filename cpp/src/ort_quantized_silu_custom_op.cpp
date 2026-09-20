#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include "silu_benchmark/quantized_silu_kernel.h"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char* kDomain = "com.yjagger.silu";
constexpr const char* kOpName = "QuantizedPiecewiseSiLU";
constexpr const char* kOrtVersion = "1.19.2";

[[noreturn]] void Invalid(const std::string& message) {
  throw std::invalid_argument("QuantizedPiecewiseSiLU: " + message);
}

double ParseHexDouble(const Ort::ConstKernelInfo& info, const char* name) {
  const std::string text = info.GetAttribute<std::string>(name);
  errno = 0;
  char* end = nullptr;
  const double value = std::strtod(text.c_str(), &end);
  if (errno == ERANGE || end == text.c_str() || end == nullptr || *end != '\0' ||
      !std::isfinite(value)) {
    Invalid(std::string(name) + " is not a finite hexadecimal float");
  }
  return value;
}

std::int64_t IntAttribute(const Ort::ConstKernelInfo& info, const char* name) {
  return info.GetAttribute<std::int64_t>(name);
}

void RequireInt(const Ort::ConstKernelInfo& info,
                const char* name,
                std::int64_t expected) {
  if (IntAttribute(info, name) != expected) {
    Invalid(std::string(name) + " differs from the frozen v1.2 contract");
  }
}

void RequireString(const Ort::ConstKernelInfo& info,
                   const char* name,
                   const char* expected) {
  if (info.GetAttribute<std::string>(name) != expected) {
    Invalid(std::string(name) + " differs from the frozen v1.2 contract");
  }
}

OrtStatusPtr StatusFromCurrentException() noexcept {
  try {
    throw;
  } catch (const Ort::Exception& error) {
    return Ort::GetApi().CreateStatus(error.GetOrtErrorCode(), error.what());
  } catch (const std::exception& error) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, error.what());
  } catch (...) {
    return Ort::GetApi().CreateStatus(ORT_FAIL, "unknown custom-op failure");
  }
}

class QuantizedPiecewiseSiluKernel {
 public:
  explicit QuantizedPiecewiseSiluKernel(const OrtKernelInfo* raw_info) {
    const Ort::ConstKernelInfo info{raw_info};
    RequireInt(info, "contract_version", 1);
    RequireInt(info, "bits", 8);
    RequireInt(info, "input_qmin", 0);
    RequireInt(info, "input_qmax", 255);
    RequireInt(info, "lower_code_start", 0);
    RequireInt(info, "lower_code_end", 127);
    RequireInt(info, "upper_code_start", 128);
    RequireInt(info, "upper_code_end", 255);
    RequireString(info, "split_ownership", "upper");
    RequireString(info, "rounding", "nearest_even");
    site_id_ = info.GetAttribute<std::string>("site_id");
    if (site_id_.empty()) {
      Invalid("site_id must not be empty");
    }

    const double input_scale = ParseHexDouble(info, "input_scale_hex");
    const auto input_zero_point = IntAttribute(info, "input_zero_point");
    if (input_zero_point < 0 || input_zero_point > 255) {
      Invalid("input_zero_point is outside uint8");
    }
    const auto input_scale_bits = IntAttribute(info, "input_scale_float32_bits");
    const float input_scale_float = static_cast<float>(input_scale);
    std::uint32_t actual_scale_bits = 0;
    static_assert(sizeof(actual_scale_bits) == sizeof(input_scale_float));
    std::memcpy(&actual_scale_bits, &input_scale_float, sizeof(actual_scale_bits));
    if (input_scale_bits < 0 ||
        static_cast<std::uint64_t>(input_scale_bits) != actual_scale_bits) {
      Invalid("input_scale hexadecimal value and float32 bits disagree");
    }

    params_ = silu_benchmark::MakeQuantizedSiluKernelParams(
        silu_benchmark::UniformInputQuantizationParams{
            silu_benchmark::QuantizedDataType::kUInt8,
            input_scale,
            static_cast<std::int32_t>(input_zero_point),
            0,
            255,
        },
        silu_benchmark::PiecewiseOutputQuantizationParams{
            silu_benchmark::QuantizedDataType::kUInt8,
            ParseHexDouble(info, "vmin_hex"),
            ParseHexDouble(info, "vsplit_hex"),
            ParseHexDouble(info, "vmax_hex"),
            8,
        });
    if (ParseHexDouble(info, "lower_scale_hex") != params_.lower_scale ||
        ParseHexDouble(info, "upper_scale_hex") != params_.upper_scale ||
        IntAttribute(info, "lower_zero_point") != params_.lower_zero_point ||
        IntAttribute(info, "upper_zero_point") != params_.upper_zero_point) {
      Invalid("serialized derived parameters differ from the shared v1.0 kernel");
    }
  }

  OrtStatusPtr ComputeV2(OrtKernelContext* raw_context) noexcept {
    try {
      Ort::KernelContext context{raw_context};
      const Ort::ConstValue code_input = context.GetInput(0);
      const Ort::ConstValue float_input = context.GetInput(1);
      const auto code_info = code_input.GetTensorTypeAndShapeInfo();
      const auto float_info = float_input.GetTensorTypeAndShapeInfo();
      const std::vector<std::int64_t> code_shape = code_info.GetShape();
      const std::vector<std::int64_t> float_shape = float_info.GetShape();
      if (code_shape != float_shape ||
          code_info.GetElementCount() != float_info.GetElementCount()) {
        Invalid("uint8 and float input shapes must match exactly");
      }
      const std::size_t element_count = code_info.GetElementCount();
      const std::uint8_t* input_codes = code_input.GetTensorData<std::uint8_t>();
      Ort::UnownedValue code_output = context.GetOutput(0, code_shape);
      Ort::UnownedValue float_output = context.GetOutput(1, float_shape);
      std::uint8_t* output_codes = code_output.GetTensorMutableData<std::uint8_t>();
      float* output_values = float_output.GetTensorMutableData<float>();
      // Parameters were fully validated and the LUT frozen in the constructor;
      // ORT owns distinct input/output tensors, so the validated unchecked
      // entry point avoids repeating setup work for every activation tensor.
      silu_benchmark::QuantizedSiluScalarUnchecked(
          input_codes, output_codes, element_count, params_);
      for (std::size_t index = 0; index < element_count; ++index) {
        output_values[index] = silu_benchmark::DequantizePiecewiseCodeReference(
            output_codes[index], params_);
      }
      return nullptr;
    } catch (...) {
      return StatusFromCurrentException();
    }
  }

 private:
  std::string site_id_;
  silu_benchmark::QuantizedSiluKernelParams params_;
};

struct QuantizedPiecewiseSiluOp
    : Ort::CustomOpBase<QuantizedPiecewiseSiluOp,
                        QuantizedPiecewiseSiluKernel,
                        true> {
  OrtStatusPtr CreateKernelV2(const OrtApi&,
                              const OrtKernelInfo* info,
                              void** kernel) const noexcept {
    try {
      *kernel = new QuantizedPiecewiseSiluKernel(info);
      return nullptr;
    } catch (...) {
      return StatusFromCurrentException();
    }
  }

  const char* GetName() const noexcept { return kOpName; }
  const char* GetExecutionProviderType() const noexcept {
    return "CPUExecutionProvider";
  }
  std::size_t GetInputTypeCount() const noexcept { return 2; }
  ONNXTensorElementDataType GetInputType(std::size_t index) const {
    if (index == 0) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    }
    if (index == 1) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  std::size_t GetOutputTypeCount() const noexcept { return 2; }
  ONNXTensorElementDataType GetOutputType(std::size_t index) const {
    return GetInputType(index);
  }

  static Ort::Status InferOutputShape(Ort::ShapeInferContext& context) {
    if (context.GetInputCount() != 2) {
      return Ort::Status("QuantizedPiecewiseSiLU requires exactly two inputs",
                         ORT_INVALID_ARGUMENT);
    }
    Ort::Status status = context.SetOutputShape(
        0, context.GetInputShape(0), ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    if (status.IsOK()) {
      status = context.SetOutputShape(
          1, context.GetInputShape(1), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    }
    return status;
  }
};

}  // namespace

extern "C" __declspec(dllexport) OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options,
    const OrtApiBase* api_base) noexcept {
  if (options == nullptr || api_base == nullptr) {
    return nullptr;
  }
  const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
  if (api == nullptr) {
    return nullptr;
  }
  Ort::InitApi(api);
  try {
    // This call intentionally keeps an import-library dependency on the exact
    // official ORT SDK used for the build. Python adds its local capi directory
    // to the process DLL search path before loading this library.
    const OrtApiBase* linked_api_base = OrtGetApiBase();
    const char* runtime_version = linked_api_base->GetVersionString();
    if (runtime_version == nullptr || std::string(runtime_version) != kOrtVersion) {
      throw std::runtime_error(
          "silu_ort_custom_op requires ONNX Runtime 1.19.2");
    }
    static QuantizedPiecewiseSiluOp op;
    static Ort::CustomOpDomain domain{kDomain};
    static const bool op_added = [] {
      domain.Add(&op);
      return true;
    }();
    (void)op_added;
    Ort::UnownedSessionOptions session_options{options};
    session_options.Add(domain);
    return nullptr;
  } catch (...) {
    return StatusFromCurrentException();
  }
}
