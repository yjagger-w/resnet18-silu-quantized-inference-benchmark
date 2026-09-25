#!/usr/bin/env python3
"""Small TVM TensorIR vs CUDA Bias+SiLU experiment on a CUDA device.

This intentionally benchmarks out-of-place FP32 only.  The existing CUDA
benchmark is the comparison baseline; use its out-of-place rows on the same
GPU with the same warm-up/iteration settings.  Both use the same host inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Shape:
    name: str
    batch: int
    channels: int
    height: int
    width: int

    @property
    def spatial(self) -> int:
        return self.height * self.width

    @property
    def elements(self) -> int:
        return self.batch * self.channels * self.spatial


SHAPES = (
    Shape("stem_64x32x32", 1, 64, 32, 32),
    Shape("stage2_128x16x16", 1, 128, 16, 16),
    Shape("stage3_256x8x8", 1, 256, 8, 8),
    Shape("stage4_512x4x4", 1, 512, 4, 4),
)


def make_data(shape: Shape) -> tuple[np.ndarray, np.ndarray]:
    """Match cuda/tools/bias_silu_benchmark.cu exactly (float32 inputs)."""
    indices = np.arange(shape.elements, dtype=np.int32)
    channels = np.arange(shape.channels, dtype=np.int32)
    x = ((indices % 257) - 128).astype(np.float32) * np.float32(0.0625)
    bias = ((channels % 17) - 8).astype(np.float32) * np.float32(0.03125)
    return x, bias


def reference(x: np.ndarray, bias: np.ndarray, shape: Shape) -> np.ndarray:
    # The original CUDA benchmark uses double for its host reference.
    channel = (np.arange(shape.elements) // shape.spatial) % shape.channels
    values = x.astype(np.float64) + bias[channel].astype(np.float64)
    return (values / (1.0 + np.exp(-values))).astype(np.float32)


def build_kernel(tvm, shape: Shape, threads: int):
    """Express two TE stages; inline bias_add with an S-TIR schedule."""
    from tvm import te

    x = te.placeholder((shape.elements,), name="input", dtype="float32")
    bias = te.placeholder((shape.channels,), name="bias", dtype="float32")
    added = te.compute(
        (shape.elements,),
        lambda i: x[i] + bias[(i // shape.spatial) % shape.channels],
        name="bias_add",
    )
    output = te.compute(
        (shape.elements,),
        lambda i: added[i] / (te.const(1.0, "float32") + te.exp(-added[i])),
        name="bias_silu",
    )
    prim_func = te.create_prim_func([x, bias, output]).with_attr(
        {"global_symbol": "main"}
    )
    sch = tvm.s_tir.Schedule(tvm.IRModule({"main": prim_func}))
    sch.compute_inline(sch.get_sblock("bias_add"))
    (axis,) = sch.get_loops(sch.get_sblock("bias_silu"))
    blocks, lanes = sch.split(axis, factors=[None, threads])
    sch.bind(blocks, "blockIdx.x")
    sch.bind(lanes, "threadIdx.x")
    return tvm.tirx.build(sch.mod, target="cuda"), sch.mod


def run_shape(tvm, shape: Shape, args, device) -> dict:
    x, bias = make_data(shape)
    expected = reference(x, bias, shape)
    module, scheduled_ir = build_kernel(tvm, shape, args.threads)
    if args.dump_ir:
        ir_path = args.dump_ir / f"{shape.name}.tir"
        ir_path.write_text(scheduled_ir.script(), encoding="utf-8")

    x_dev = tvm.runtime.tensor(x, device=device)
    bias_dev = tvm.runtime.tensor(bias, device=device)
    out_dev = tvm.runtime.tensor(np.zeros_like(x), device=device)
    module(x_dev, bias_dev, out_dev)
    device.sync()
    actual = out_dev.numpy()
    max_error = float(np.max(np.abs(actual.astype(np.float64) - expected)))
    if not np.all(np.isfinite(actual)) or not np.allclose(
        actual, expected, rtol=1e-5, atol=1e-5
    ):
        raise RuntimeError(f"{shape.name}: TVM output failed CUDA CPU reference; max error={max_error}")

    for _ in range(args.warmup):
        module(x_dev, bias_dev, out_dev)
    device.sync()

    # One launch per sample. Allocation, copies, compile and warm-up are outside.
    evaluator = module.time_evaluator(
        "main", device, number=1, repeat=args.iterations, min_repeat_ms=0
    )
    timings_us = [float(seconds * 1e6) for seconds in evaluator(x_dev, bias_dev, out_dev).results]
    if len(timings_us) != args.iterations or not all(
        math.isfinite(t) and t > 0 for t in timings_us
    ):
        raise RuntimeError(f"{shape.name}: invalid TVM timing samples")
    return {
        "shape": shape.name,
        "shape_nchw": [shape.batch, shape.channels, shape.height, shape.width],
        "mode": "out-of-place",
        "implementation": "tvm_te_s_tir_bias_inline",
        "elements": shape.elements,
        "mean_us": statistics.mean(timings_us),
        "p50_us": statistics.median(timings_us),
        "min_us": min(timings_us),
        "max_us": max(timings_us),
        "max_abs_error": max_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=[s.name for s in SHAPES] + ["all"], default="all")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--dump-ir", type=Path, help="Write the scheduled TIR to this directory")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0 or not 1 <= args.threads <= 1024:
        parser.error("require warmup >= 0, iterations > 0, threads in [1, 1024]")

    try:
        import tvm
    except ImportError as exc:
        parser.error(f"Apache TVM is required in this Python environment: {exc}")

    device = tvm.cuda(0)
    if not device.exist:
        parser.error("CUDA device 0 is unavailable to this Apache TVM build")
    if args.dump_ir:
        args.dump_ir.mkdir(parents=True, exist_ok=True)
    selected = SHAPES if args.shape == "all" else tuple(s for s in SHAPES if s.name == args.shape)
    results = [run_shape(tvm, shape, args, device) for shape in selected]
    print(json.dumps({
        "schema_version": 1,
        "tvm_version": tvm.__version__,
        "device": str(device),
        "protocol": {
            "warmup_iterations": args.warmup,
            "measured_iterations": args.iterations,
            "threads_per_block": args.threads,
            "timer": "TVM runtime time_evaluator on CUDA, one invocation per sample",
            "excluded": "compile, allocation, host transfers and warm-up",
            "comparison": "compare only out-of-place FP32 CUDA benchmark rows on the same device",
        },
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
