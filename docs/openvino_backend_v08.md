# v0.8: Standard-QDQ OpenVINO CPU adapter

## Scope and current evidence

This milestone supports **only the existing Standard static-QDQ deployment baseline on CPU**. It does not convert the custom SiLU piecewise ORT functional reference. There is no GPU, NPU or QNN implementation or deployment claim. The next portability target after OpenVINO is QNN, using Standard-QDQ ONNX.

At implementation time, the local `mamba_ptq` environment has Python 3.9.25 and no OpenVINO installation. Unit fixtures test orchestration and reporting without OpenVINO; they are not conversion, inference or performance evidence. Real IR operation counts, CPU plugin support, prediction agreement, accuracy and timing remain unmeasured. No numerical acceptance tolerance or speedup is claimed.

The locked source is `artifacts/int8/resnet18_silu_int8_v065.onnx`, SHA-256:

```text
fba46162c6685d7d23bd6da0a94d14bfa09f11babfc776fd5597a9e39c41b3ac
```

It is 11,319,715 bytes and contains 66 `QuantizeLinear` and 108 `DequantizeLinear` nodes. Provenance is the successful, non-smoke v0.6.5 report plus its artifact sidecar. The official Standard-QDQ ORT CPU top-1 accuracy is 93.57% on 10,000 images: this is historical ORT evidence, **not an OpenVINO result**.

## Data and source invariants

`configs/benchmarks/resnet18_silu_cifar10_v08_openvino_cpu.json` preserves:

- ResNet18-SiLU, CIFAR-10 test split and the v0.6.5 checkpoint;
- static MinMax QDQ, QUInt8 activations, per-channel QInt8 weights, 2,560 calibration images;
- the original preprocessing, evaluation batch size 128, seed and timing conventions.

Every invocation validates the source SHA against the config, official report and sidecar; checks the official run status/fingerprint, checkpoint and test-batch hashes; and checks protocol fields against the frozen v0.6.5 configuration. `--model` allows a byte-identical copy, not a different model. Smoke-calibrated ONNX, FP32, custom piecewise and modified models are rejected. Missing source provenance is an error: this adapter never recalibrates, regenerates or edits the source.

Inputs come from the existing offline NumPy CIFAR-10 helper. No downloads, PyTorch imports, hooks or duplicate-OpenMP workarounds are used. The checkpoint is hashed, not loaded into PyTorch. Clean-process import tests prohibit Torch/torchvision imports. Existing legacy tests elsewhere in the repository retain their original framework dependencies.

## Conversion and reuse

`Core.read_model` reads Standard-QDQ ONNX with the native ONNX frontend; `openvino.serialize(..., version="IR_V11")` saves XML/BIN. This deliberately avoids the extra FP16 weight compression enabled by default in `save_model`. The source quantization mathematics are unchanged. These are frontend/serialization settings, not a promise about the CPU plugin's execution precision.

Each immutable bundle contains `model.xml`, `model.bin` and `metadata.json`. Metadata records source SHA, OpenVINO version, conversion settings, CPU target, available devices, output hashes and parsed IR facts. The fingerprint includes source SHA, runtime version, settings, device and adapter source SHA. XML and BIN hashes must also match before reuse or compilation.

- `--resume` reuses only a complete, hash-valid, matching IR bundle; validation and timing run again into a new report directory.
- Stale, incomplete, changed-version or corrupted IR is not reused. Conversion creates a new bundle and preserves the old one.
- `--force-rebuild` overrides `--resume` and always creates another bundle; it does not delete or overwrite previous bundles.
- Generated IR is constrained to `artifacts/openvino/v0.8/`; reports to `results/benchmarks/v0.8*/`. Resolved path checks reject escapes and the v0.6.5 report path.

Conversion and benchmark workers are supervised. Python errors and native child-process failures leave terminal `failed` status, exit code/error detail and `worker.log` in the partial run directory. Conversion failures also retain `failure.json` in their partial IR bundle. Success promotes only a complete report set. Existing report directories are preserved by using a new suffixed directory. An OS kill of the supervisor itself or loss of write access cannot be made crash-proof by this mechanism.

## Validation, timing and reports

Validation compares ORT `CPUExecutionProvider` and OpenVINO `CPU` on the first 128 CIFAR-10 test images, in original order. It reports logit max absolute error, mean absolute error, prediction agreement, all disagreement indices and predictions, and top-1 accuracy for each backend on that same batch. Disagreement indices are zero-based test-set indices. `tolerance` is deliberately `null` / `not_established`: successful execution means the measurements completed, not that cross-backend equality or numerical acceptance was established. A tolerance must be justified from real output behavior before changing this policy.

Accuracy uses only OpenVINO for the new backend row. Smoke evaluates exactly the first 128 images, with 2/5 latency warmup/timed calls and 1/3 throughput warmup/timed calls. The official protocol supports 10,000 images, 20/100 latency calls and 10/30 throughput calls, but **no full OpenVINO benchmark is authorized in this task**.

The uninstrumented IR is compiled explicitly for `CPU` with `PERFORMANCE_HINT=LATENCY`. Timing uses synchronous `InferRequest.infer`, NumPy FP32 contiguous inputs and no shared output buffers. Latency uses batch 1 zeros; throughput uses the first normalized batch of 128 real images. As in v0.6.5, measurements use `perf_counter_ns`, p50/p95 latency and batch size divided by mean duration for throughput. API/dispatch/copy overhead is included; data loading and RSS queries are excluded. No automatic device fallback or asynchronous throughput optimization is used.

Memory is sampled process RSS/Windows working set, not tensor-allocator memory. The reported maximum is the peak **observed at sampling points**, not a continuous high-water mark. It includes the Python process, runtime allocations and any retained memory from validation. It does not isolate model-only memory or guarantee comparability to historical ORT runs.

Successful benchmark reports contain:

- `benchmark_results.json`, `benchmark_results.csv`, `benchmark_report.md`;
- `conversion.json`, `validation.json`, `environment.json`, `graph_report.json`;
- terminal `run_status.json`, the invocation record `worker_request.json` and `worker.log`.

JSON/Markdown separate historical official ORT accuracy from new OpenVINO measurements; CSV has only the new OpenVINO row and same-batch comparison metrics. IR reports include operation-type counts, `FakeQuantize` presence, quantization-related conversions and declared element types. Device/plugin and compiled properties are queried defensively, recording unavailable properties instead of inventing values. **IR quantization nodes and capability properties do not prove all operations execute as INT8 on a particular CPU.**

## Manual environment setup and smoke commands

No installation was performed. For the existing Python 3.9 Windows x86-64 environment, the recommended pinned package is OpenVINO 2025.3.0, which publishes a CPython 3.9 Windows wheel. Review the environment change before running:

```powershell
& 'E:\ProgramData\Anaconda_envs\envs\mamba_ptq\python.exe' -m pip install "openvino==2025.3.0"
```

After manual installation, from the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
& 'E:\ProgramData\Anaconda_envs\envs\mamba_ptq\python.exe' scripts\run_openvino_standard_qdq_benchmark.py `
  --config configs\benchmarks\resnet18_silu_cifar10_v08_openvino_cpu.json `
  --smoke `
  --force-rebuild
```

Default smoke report destination: `results/benchmarks/v0.8_openvino_cpu_smoke/` (a unique suffix is added when it exists). IR bundles are under `artifacts/openvino/v0.8/standard_qdq_cpu/`. Failed attempts stay in distinct `.partial-*` directories. `--output` and `--ir-output` accept explicit paths within the corresponding allowed generated roots.

Conversion only, with safe reuse:

```powershell
& 'E:\ProgramData\Anaconda_envs\envs\mamba_ptq\python.exe' scripts\convert_openvino_standard_qdq.py `
  --config configs\benchmarks\resnet18_silu_cifar10_v08_openvino_cpu.json `
  --resume
```

This writes a separate conversion receipt, not a benchmark report or accuracy claim. Conversion is identical for smoke and official runs because both reuse the fully calibrated official source.

## Verification and primary references

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
& 'E:\ProgramData\Anaconda_envs\envs\mamba_ptq\python.exe' -m unittest discover -s tests -v
& 'E:\ProgramData\Anaconda_envs\envs\mamba_ptq\python.exe' -m compileall -q src tests
git diff --check
```

The optional real CPU integration test is skipped when OpenVINO is absent. Its isolated subprocess avoids importing legacy framework tests into the OpenVINO process. Fixture conversion/report tests do not establish runtime compatibility.

- [OpenVINO Core: native reading and explicit device compilation](https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.Core.html)
- [IR serialization](https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.serialize.html)
- [save_model default compression behavior](https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.save_model.html)
- [Synchronous inference and buffer sharing](https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.InferRequest.html)
- [OpenVINO 2025.3.0 published wheels](https://pypi.org/project/openvino/2025.3.0/#files)
