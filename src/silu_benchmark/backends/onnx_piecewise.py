"""Standard-operator ONNX expression of the SiLU piecewise reference.

This module intentionally does not use QuantizeLinear or DequantizeLinear:
the canonical SiLU-aware contract has two affine segments, whereas standard
ONNX QDQ describes a single affine quantizer per node.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from silu_benchmark.quantization import PiecewiseQuantizationSpec


# Opset 13 is supported by ONNX Runtime 1.19.2 and provides Clip with dynamic
# scalar bounds. The graph's IR version is pinned for that runtime release.
PIECEWISE_ONNX_OPSET = 13
_ORT_COMPATIBLE_IR_VERSION = 10


def _float64_initializer(name: str, value: float) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.float64), name)


def build_piecewise_qdq_model(
    spec: PiecewiseQuantizationSpec,
    input_shape: Sequence[Optional[int]] = (None,),
) -> onnx.ModelProto:
    """Build a checked ONNX model for one validated spec and tensor rank.

    ``activation`` is FLOAT (float32).  The graph converts it to FLOAT64 for
    the affine calculation so that its rounding inputs and code decisions use
    the exact scalar values exposed by ``PiecewiseQuantizationSpec``.  The
    transport codes are UINT8 and ``dequantized_output`` is FLOAT (float32).
    ``input_shape`` declares the tensor rank; use ``()`` for a scalar and
    ``None`` dimensions for dynamic sizes at a practical fixed rank.
    """
    if not isinstance(spec, PiecewiseQuantizationSpec):
        raise TypeError("spec must be a PiecewiseQuantizationSpec")
    if spec.bits != 8:
        raise ValueError("the ONNX UINT8 transport graph requires an 8-bit spec")
    input_shape = tuple(input_shape)

    lower_start, lower_end = spec.lower_codes
    upper_start, upper_end = spec.upper_codes
    initializers = [
        _float64_initializer("vmin", spec.vmin),
        _float64_initializer("vsplit", spec.vsplit),
        _float64_initializer("vmax", spec.vmax),
        _float64_initializer("lower_scale", spec.lower_scale),
        _float64_initializer("upper_scale", spec.upper_scale),
        _float64_initializer("lower_zero_point", spec.lower_zero_point),
        _float64_initializer("upper_zero_point", spec.upper_zero_point),
        _float64_initializer("lower_code_start", lower_start),
        _float64_initializer("lower_code_end", lower_end),
        _float64_initializer("upper_code_start", upper_start),
        _float64_initializer("upper_code_end", upper_end),
    ]

    nodes = [
        helper.make_node("Cast", ["activation"], ["activation_f64"], to=TensorProto.DOUBLE),
        helper.make_node("Clip", ["activation_f64", "vmin", "vmax"], ["clipped"]),
        helper.make_node("Less", ["clipped", "vsplit"], ["is_lower_input"]),
        helper.make_node("Div", ["clipped", "lower_scale"], ["lower_scaled"]),
        helper.make_node("Add", ["lower_scaled", "lower_zero_point"], ["lower_affine"]),
        helper.make_node("Round", ["lower_affine"], ["lower_rounded"]),
        helper.make_node(
            "Clip",
            ["lower_rounded", "lower_code_start", "lower_code_end"],
            ["lower_codes_f64"],
        ),
        helper.make_node("Div", ["clipped", "upper_scale"], ["upper_scaled"]),
        helper.make_node("Add", ["upper_scaled", "upper_zero_point"], ["upper_affine"]),
        helper.make_node("Round", ["upper_affine"], ["upper_rounded"]),
        helper.make_node(
            "Clip",
            ["upper_rounded", "upper_code_start", "upper_code_end"],
            ["upper_codes_f64"],
        ),
        helper.make_node(
            "Where",
            ["is_lower_input", "lower_codes_f64", "upper_codes_f64"],
            ["selected_codes_f64"],
        ),
        helper.make_node(
            "Cast", ["selected_codes_f64"], ["quantized_codes"], to=TensorProto.UINT8
        ),
        helper.make_node("Cast", ["quantized_codes"], ["codes_f64"], to=TensorProto.DOUBLE),
        helper.make_node("Less", ["codes_f64", "upper_code_start"], ["is_lower_code"]),
        helper.make_node("Sub", ["codes_f64", "lower_zero_point"], ["lower_unoffset"]),
        helper.make_node("Mul", ["lower_unoffset", "lower_scale"], ["lower_dequantized"]),
        helper.make_node("Sub", ["codes_f64", "upper_zero_point"], ["upper_unoffset"]),
        helper.make_node("Mul", ["upper_unoffset", "upper_scale"], ["upper_dequantized"]),
        helper.make_node(
            "Where",
            ["is_lower_code", "lower_dequantized", "upper_dequantized"],
            ["selected_dequantized"],
        ),
        helper.make_node(
            "Clip",
            ["selected_dequantized", "vmin", "vmax"],
            ["clipped_dequantized"],
        ),
        helper.make_node(
            "Cast", ["clipped_dequantized"], ["dequantized_output"], to=TensorProto.FLOAT
        ),
    ]
    graph = helper.make_graph(
        nodes=nodes,
        name="silu_piecewise_qdq_reference",
        inputs=[
            helper.make_tensor_value_info("activation", TensorProto.FLOAT, input_shape)
        ],
        outputs=[
            helper.make_tensor_value_info("quantized_codes", TensorProto.UINT8, input_shape),
            helper.make_tensor_value_info(
                "dequantized_output", TensorProto.FLOAT, input_shape
            ),
        ],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="silu_benchmark",
        opset_imports=[helper.make_opsetid("", PIECEWISE_ONNX_OPSET)],
    )
    model.ir_version = _ORT_COMPATIBLE_IR_VERSION
    onnx.checker.check_model(model)
    return model


def create_piecewise_ort_session(
    model_or_path: Union[onnx.ModelProto, str, Path],
) -> ort.InferenceSession:
    """Create a CPU-only ONNX Runtime session for the reference graph."""
    if isinstance(model_or_path, onnx.ModelProto):
        model_or_path = model_or_path.SerializeToString()
    return ort.InferenceSession(model_or_path, providers=["CPUExecutionProvider"])
