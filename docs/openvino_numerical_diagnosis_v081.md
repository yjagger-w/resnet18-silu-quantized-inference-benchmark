# v0.8.1: OpenVINO CPU numerical-divergence diagnosis

This is a diagnostic, not a repair or an official benchmark. Only the verified v0.6.5 Standard-QDQ ONNX and the corresponding v0.8 CPU IR are executed. The original source, calibration, weights, scales, IR, compile properties and artifact fingerprints are unchanged. No PyTorch, GPU, NPU, QNN or custom-piecewise inference is involved.

## Run

From the repository root, using the existing environment:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
& 'E:\ProgramData\Anaconda_envs\envs\mamba_ptq\python.exe' scripts\diagnose_openvino_standard_qdq.py --smoke
```

Exactly the first 128 CIFAR-10 test images are normalized through the existing NumPy helper, in the same order and one batch as the v0.8 smoke. There is no full-dataset mode or acceptance-tolerance option. A verified canonical IR must already exist; the diagnostic never builds or overwrites it.

Optional narrowing, with inclusive zero-based **original ONNX** node indices:

```powershell
& 'E:\ProgramData\Anaconda_envs\envs\mamba_ptq\python.exe' scripts\diagnose_openvino_standard_qdq.py `
  --smoke --node-start 46 --node-end 55
```

Every invocation creates a new `results/benchmarks/v0.8.1_diagnosis-<unique-id>/` directory. `--output` may select a different `v0.8.1_diagnosis*` root under `results/benchmarks/`; it cannot target the v0.6.5 report, v0.8 smoke or canonical models. The existing narrow v0.8 ignore rule covers these outputs, so no `.gitignore` change is needed.

## Probe and output identity

The initial checkpoints cover input, stem Conv, all 17 SiLU Mul semantic outputs, eight residual adds/block outputs, pre-pool, pool, flatten and logits. SiLU identities are bound from the verified FP32 source graph and committed manifest, then checked against the QDQ graph's producer names and Sigmoid/Mul connections through inserted Q/DQ operations. The manifest is used only for identity/provenance; its custom quantization parameters are not applied.

Each probe records the source tensor, producer name/op, producer output index, original topological index, input dependence, QDQ adjacency and semantic roles. Instrumentation appends existing tensors as graph outputs on a deep copy; source nodes, attributes, initializers and original outputs remain unchanged. ONNX shape/type inference supplies verified output descriptions, and checker validation is mandatory.

The instrumented ONNX is executed by ORT CPU and converted separately through OpenVINO's native ONNX reader to debug IR. Serialized debug IR is reread and compiled for CPU with the unchanged v0.8 compile properties. Source names, graph output indices and OpenVINO port aliases must agree: missing, reordered, ambiguous or shape-incompatible probes fail instead of being silently paired.

The initial pass is followed automatically by a prefix through eight nodes beyond the first input-dependent differing checkpoint, capped at the last source node. This includes parameter/constant outputs in that prefix so they can be distinguished from activation differences. An optional requested node window is an additional independent pass, not silently merged into another instrumented graph's observations.

When that prefix exposes a differing QuantizeLinear and its input, a supplementary **identical-input quantizer control** copies that one source node and its original constant parameters into an isolated debug model. Both backends receive the exact same saved ORT pre-quantization tensor. This removes upstream Conv arithmetic as a confounder without changing any source quantization parameter. Control ONNX/IR hashes, input digest, output mappings and metrics are recorded separately; it is not a replacement model. Boundary summaries report measured distances to half-integers, not a chosen tolerance. All tensors remain available; at most 64 code-disagreement examples are printed, with total counts and an explicit truncation flag.

## Reading the evidence

For every pair, JSON records shape, both dtypes, maximum/mean absolute error, MSE, cosine similarity, exact equality, differing-element count and the values/location at maximum error. Zero/zero cosine is defined as 1; one zero norm as 0. All values must be finite.

The first **exact numerical inequality** locates an observation; it does not establish that a difference is meaningful for accuracy. There is no epsilon or acceptance threshold. A separate first integer-code discrepancy identifies a discrete quantization consequence without inventing a floating-point tolerance. The last preceding exactly matching checkpoint is reported in source topological order, which need not mean it is the immediate dataflow parent. Constant/parameter and input-dependent differences are explicitly separated.

All logits comparisons retain prediction arrays, every disagreeing sample index, agreement and both top-1 accuracies. The uninstrumented baseline reproduces the original smoke comparison. Each debug pass also compares its logits with the uninstrumented logits **within each backend**, because exposing intermediates can prevent fusion or change precision choices. If instrumentation changes the result, debug localization cannot by itself establish the precise first difference inside the uninstrumented optimized graph.

OpenVINO IR operation counts and compiled execution metadata (`originalLayersNames`, `runtimePrecision`, `primitiveType`, when provided) are evidence about lowering/fusion, not proof of all-INT8 execution. No unsupported operation, conversion failure or mapping error is ignored. A causal explanation must distinguish these facts from hypotheses; otherwise the conclusion remains **undetermined**.

## Artifacts and lifecycle

`diagnosis.json` and `diagnosis.md` contain the diagnosis and all comparisons. Provenance includes canonical ONNX/XML/BIN SHA-256, debug ONNX/IR hashes, library versions, manifest hash, raw test-batch SHA, normalized input and label digests, sample indices, source-code hashes and output mappings. `run_status.json` records terminal success/failure.

Debug ONNX/IR, NumPy inputs/outputs and worker logs are retained only in that isolated ignored diagnostic directory for reproducibility; they are not canonical/deployment artifacts. ORT and OpenVINO run in separate supervised subprocesses with finite timeouts. Inference objects are explicitly released and each process exits before the parent reads outputs, avoiding Windows IR-handle lifetime issues. Existing generated artifacts are preserved, including failed diagnostic attempts. There are no performance or latency claims for instrumented models.

## Primary API references

- [OpenVINO tensor names and output ports](https://docs.openvino.ai/2025/openvino-workflow/running-inference/model-input-output.html)
- [OpenVINO execution-model metadata](https://docs.openvino.ai/2025/api/c_cpp_api/group__ov__dev__exec__model.html)
- [ONNX QuantizeLinear rounding semantics](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html#quantizelinear-13)

These describe the mapping and execution metadata API, not the cause of this model's numerical discrepancy. The generated diagnosis is the source of measured evidence.
