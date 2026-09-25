# TVM FP32 Bias+SiLU comparison

This experiment starts from two tensor-expression stages (`bias_add` and
`bias_silu`). The S-TIR schedule inlines `bias_add` into the output stage,
then binds the output loop to CUDA blocks and threads. It targets the same
contiguous NCHW shapes, bias values and FP32 input values as
`cuda/tools/bias_silu_benchmark.cu`. It validates the output against a
float64 CPU reference before timing GPU invocations.

The first experiment supports **out-of-place FP32 only**. It does not import
the complete ONNX ResNet18, tune a kernel, or claim automated model-wide
fusion. Compare only with out-of-place FP32 CUDA results taken in the same
T4 session. TVM's runtime `time_evaluator` and the C++ benchmark's CUDA
Events use different timing interfaces, so small differences need a
matched-timer confirmation before drawing performance conclusions.

## On the Tesla T4 machine

Use a Python environment with a CUDA-enabled Apache TVM build. Inspect the
actual installed version and device before starting. TVM's current [official
installation guide](https://tvm.apache.org/docs/install/index.html) gives
source-build and Docker routes; avoid assuming that a generic pip package
includes the CUDA build needed here.

```bash
python3 -c 'import tvm; print(tvm.__version__, tvm.cuda(0).exist)'
python3 tvm_experiments/bias_silu_fusion.py --help

mkdir -p out/tvm/bias_silu_fusion
python3 tvm_experiments/bias_silu_fusion.py \
  --shape stem_64x32x32 --warmup 10 --iterations 20 \
  --dump-ir out/tvm/bias_silu_fusion/ir \
  > out/tvm/bias_silu_fusion/stem_smoke.json
python3 -m json.tool out/tvm/bias_silu_fusion/stem_smoke.json > /dev/null

python3 tvm_experiments/bias_silu_fusion.py \
  --shape all --warmup 50 --iterations 1000 \
  --dump-ir out/tvm/bias_silu_fusion/ir \
  > out/tvm/bias_silu_fusion/tvm_fp32.json

./build/cuda-release/cuda_bias_silu_benchmark \
  --warmup 50 --iterations 1000 \
  --implementation all --mode out-of-place \
  > out/tvm/bias_silu_fusion/cuda_fp32.json
```

Inspect the scheduled `.tir` files to confirm that the intermediate
`bias_add` buffer has been inlined. Both benchmark commands exclude
compilation, allocation and host transfers. Keep the original output JSON
with the GPU, toolkit and TVM build information when reporting results.
If correctness fails, do not report latency as a successful comparison.
