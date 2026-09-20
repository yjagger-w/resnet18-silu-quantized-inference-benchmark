# SiLU Piecewise Quantization Specification

This is the canonical contract for the trusted Python reference. It describes the research path, not the standard ONNX Runtime QDQ baseline.

## Terminology and Inputs

x is an activation value; Vmin < 0 < Vsplit < Vmax are finite calibration parameters. q is an unsigned integer code with bits bits (8 bits by default), and x_hat is the reconstructed floating-point value. The default code range is [0, 255].

## Segment Definition

There are exactly two segments. The lower segment is [Vmin, Vsplit) and owns codes [0, 127]; it contains negative, zero, and low-positive values. The upper segment is [Vsplit, Vmax] and owns codes [128, 255]. The split belongs to the upper segment. Values outside the calibrated interval are clipped before quantization.

For N = 2^bits, the lower range has N/2 codes and the upper range has N/2 codes. The implementation requires bits >= 2.

## Codebook Allocation

The lower range is exactly 0..N/2-1; the upper range is exactly N/2..N-1. These ranges are disjoint, contiguous, complete, and in-range. There is no hidden or duplicate boundary code.

## Quantization Formula

Let L = N/2 - 1, U0 = N/2, and U1 = N-1.

* Lower scale: s_l = (Vsplit - Vmin) / L; lower zero-point: z_l = round_to_nearest_even(-Vmin / s_l).
* Upper scale: s_u = (Vmax - Vsplit) / (U1 - U0); upper zero-point: z_u = round_to_nearest_even(U0 - Vsplit / s_u), reduced when necessary so the decoded upper endpoint is not below the decoded lower endpoint. This deterministic boundary guard handles one-code affine rounding inversions.
* For x < Vsplit, q = clip(round_to_nearest_even(x / s_l + z_l), 0, L).
* For x >= Vsplit, q = clip(round_to_nearest_even(x / s_u + z_u), U0, U1).

round_to_nearest_even is NumPy/PyTorch ties-to-even rounding. The output code dtype of the reference is signed int64 because it is an arithmetic reference; the represented code values are unsigned-range values and can be serialized as uint8 for 8-bit transport.

## Dequantization Formula

Codes q <= L decode as x_hat = (q - z_l) * s_l. Codes q >= U0 decode as x_hat = (q - z_u) * s_u. The result is clipped to [Vmin, Vmax]. Segment selection is determined by code ownership, not by re-testing a reconstructed float.

## Boundary Contract

x < Vmin saturates to code 0; x = Vmin also maps to the lower endpoint. Values immediately below and above zero use the lower segment; x = 0 maps deterministically to z_l, whose reconstruction is zero up to one lower step. Values below Vsplit use lower codes, while x = Vsplit and values above it use upper codes. x = Vmax maps to 255; x > Vmax saturates to 255.

## Required Invariants

The reference enforces finite, strictly ordered parameters; finite positive scales; complete, disjoint code ranges; deterministic output; monotonic quantize/dequantize output; stable zero mapping; endpoint saturation; valid code inputs; shape preservation; and a documented conversion from arithmetic int64 codes to transport uint8.

## Backend Separation

Standard ONNX QDQ, with one scale and zero-point per tensor, is the portable deployment baseline. It must not be reported as implementing this custom piecewise method. v0.6 Phase 2 expresses the verified reference with ordinary ONNX operators and compares it with ONNX Runtime; C++ custom-operator work remains a future v0.7 backend path.

## ONNX Standard-Operator Reference (v0.6 Phase 2)

`silu_benchmark.quantization.piecewise_quantize` and `piecewise_dequantize` remain the canonical Python reference. `silu_benchmark.backends.onnx_piecewise.build_piecewise_qdq_model` expresses those same two affine segments as an ONNX subgraph using ordinary `Cast`, `Clip`, `Less`, `Where`, `Div`, `Add`, `Sub`, `Mul`, and `Round` nodes. It uses ONNX opset 13, which is supported by the project's ONNX Runtime 1.19.2 environment.

The graph accepts a float32 tensor named `activation` and publishes `quantized_codes` as uint8 plus `dequantized_output` as float32. The builder takes an `input_shape` argument so callers can choose a scalar, vector, matrix, or higher practical tensor rank; dimensions may be dynamic, though ONNX requires the rank itself to be declared. Arithmetic through code selection is float64 so the graph consumes the validated `PiecewiseQuantizationSpec` scalars without independently re-deriving them; the published dequantized float32 value is the final cast of the canonical reconstruction. `x < Vsplit` selects the lower branch, so the split itself belongs to the upper branch. The input is clipped before quantization, each branch is clipped to its own disjoint code range, and decoded values are clipped to `[Vmin, Vmax]`.

Validation requires exact uint8 code equality. The float32 reconstruction comparison uses an absolute tolerance of `2e-7`, which is deliberately strict for the final float32 cast; test failures report the input, selected segment, both codes, both reconstructed values, and the absolute error.

This is a functional ONNX expression of the custom piecewise method, not an ordinary one-scale `QuantizeLinear`/`DequantizeLinear` graph and not a claim of hardware INT8 acceleration. The standard QDQ deployment baseline stays separate. Current limitations are that this validates only the standalone subgraph on `CPUExecutionProvider`; replacing SiLU nodes in the ResNet18 model and a C++ custom-operator/runtime backend remain future work.

Run the validation with:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m unittest discover -s tests -v
```

## Audit Decisions and Contradictions

The dissertation source uses the intended affine allocation (0..127 and 128..255), y < Vsplit, a positive MSE-selected Vsplit, per-segment scales, zero-points, clipping, and ties-to-even rounding. The package previously used two independent 0..255 scales split at zero, while its MSE helper searched a negative cutoff and did not expose Vmin. Those definitions could not satisfy the requested contract. The class and MSE helper now delegate to this reference; Vmin is carried from calibration, Vmax remains KLD-selected, and Vsplit remains MSE-selected.

Historical benchmark tracks remain separate: controlled 2,560-image results, full ORT results, and standalone RTL feasibility are not reconciled here. The existing v0.6.0-readme-results tag is preserved.
