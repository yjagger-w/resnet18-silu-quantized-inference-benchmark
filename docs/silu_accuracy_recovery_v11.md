# v1.1 SiLU piecewise accuracy recovery

## Scope and conclusion

v1.1 diagnoses the preserved v0.6 SiLU-piecewise ONNX reference and evaluates a bounded set of two-segment recovery candidates. It does not integrate the v1.0 C++ kernel, add a custom ORT operator, or make a whole-model performance or deployment claim.

The selected candidate, `two_segment_p99_99_mse`, achieved **93.58%** top-1 accuracy on the complete CIFAR-10 test set. The locked Standard-QDQ model achieved **93.57%** in the same new ORT CPU evaluation, so the selected candidate is +0.01 percentage points relative to that baseline and meets the predefined within-0.5-pp recovery goal. The preserved original piecewise model remained at **82.80%**.

## Controlled protocol

The three data roles are strictly separated:

1. Calibration uses the first 2,560 images of CIFAR-10 `data_batch_1`. It supplies activation statistics and parameters only.
2. Candidate selection uses a deterministic 2,000-image development subset drawn from the remaining CIFAR-10 training images with seed `20260828`. It is disjoint from calibration and from the final test set. Selection ranks development top-1 accuracy, then prediction agreement with Standard-QDQ, then declared configuration order.
3. The complete 10,000-image CIFAR-10 test set is loaded only after the selected candidate and a hash-bearing selection receipt have been frozen. Test labels do not participate in calibration, sensitivity ranking, or selection.

The development subset is training-derived and is consequently easy for this trained checkpoint; its near-perfect accuracy must not be interpreted as generalization performance. Its purpose is deterministic, leakage-free candidate ranking. The final test measurement is the only reported generalization result for v1.1.

## Rewrite controls

The QDQ-aware discovery pass found 17 runtime SiLU sites: the initial activation plus two calls for each of eight shared residual-block activation modules. Each site is the strict six-node island `Sigmoid -> QuantizeLinear -> DequantizeLinear -> Mul -> QuantizeLinear -> DequantizeLinear`. Missing or duplicate structure is rejected.

On a fixed 128-image probe batch:

| Control | Checker/inference | Logit max abs. error | Logit mean abs. error | Prediction agreement | Smoke accuracy |
|---|---|---:|---:|---:|---:|
| Untouched Standard-QDQ | pass | 0 | 0 | 100% | 100% |
| Topology-preserving no-op | pass | 0 | 0 | 100% | 100% |
| Semantic-expression control (legacy artifact ID `float_equivalent_silu`) | pass | 1.455992 | 0.269491 | 100% | 100% |

The no-op rewrite is bit-exact. The third row must not be called float-equivalent: it removes the Sigmoid-output and post-Mul Q/DQ pairs at all 17 sites and connects raw `Sigmoid` directly to `Mul`. Its 100% probe prediction agreement does not demonstrate numerical equivalence.

### v1.1.1 strict control-equivalence audit

The follow-up audit constructs an independent strict control that retains the original six nodes, their ordering, all input/output names, and all scale and zero-point initializers at every site. On the same fixed 128 images it proves:

- identical graph-topology hash, node sequence, initializer sequence, and all 174 whole-model Q/DQ node records;
- exact equality for 34 exposed uint8 target code tensors, 51 pre/post/activation tensors, and final logits;
- zero maximum/mean logit error and 100% prediction agreement;
- no inserted, removed, or moved Q/DQ boundary in the strict control.

The semantic-expression control removes 68 Q/DQ nodes in total. Its first numeric divergence is `act.call_0::post_silu`, where the raw-Sigmoid multiplication differs from the original quantized-Sigmoid operand by a maximum `0.00816345`, MAE `0.000132045`, and MSE `9.52488e-8`. The final-logit maximum error is `1.455992`, so this row is useful as a semantic re-expression ablation but not as an equivalence control.

The selected candidate intentionally has the same first raw-SiLU divergence, then executes its calibrated piecewise path. Graph provenance proves all 17 original six-node islands are absent, each target has exactly 24 candidate-prefixed functional nodes, every original activation-output tensor is produced by that path, and all 17 piecewise uint8 code tensors execute on the probe. No target is retained or bypassed.

## Diagnosis

The updated evidence classifies the historical failure as **mixed, dominated by calibration/range evidence**. Graph wiring in the v1.1 QDQ-aware path is exact under the strict control; the old semantic-expression control is non-identical under QDQ; and calibration/range plus accumulated approximation effects remain material:

- All 17 original piecewise sites clip observed calibration activations.
- Mean total clipping fraction is 0.5381%; the maximum is 1.5625% at `layer3.0.act.call_0` (`Vmax=0.37168`, observed maximum `3.65920`, p99.99 `1.62332`).
- The highest local MSE is 0.017372 at `layer4.1.act.call_1` (`Vmax=3.45355`, observed maximum `14.06790`, p99.99 `8.63275`, maximum absolute error `10.61435`).
- Original code occupancy is often sparse; for example, the worst-clipping site occupies only 61 of 256 codes.

One-site-at-a-time development sensitivity, with every other SiLU left exact, ranks these sites highest:

| Rank | Site | Development change vs Standard-QDQ | Prediction agreement |
|---:|---|---:|---:|
| 1 | `layer3.0.act.call_0` | -0.95 pp | 99.05% |
| 2 | `layer3.1.act.call_0` | -0.70 pp | 99.30% |
| 3 | `layer2.1.act.call_1` | -0.70 pp | 99.30% |
| 4 | `layer4.0.act.call_0` | -0.65 pp | 99.35% |
| 5 | `layer1.1.act.call_1` | -0.30 pp | 99.70% |

The preserved original reference was produced by rewriting the FP32 graph, whereas Standard-QDQ also quantizes weights and non-SiLU activations. The v1.1 candidates therefore start from the locked Standard-QDQ graph, preserve its weights and all non-SiLU treatment, and replace only the proven SiLU-local QDQ islands. This makes candidate deltas against Standard-QDQ controlled; the original 82.80% model remains a historical functional-reference row rather than an apples-to-apples deployment baseline. Consequently, the entire historical 10.77-pp gap cannot be attributed solely to range selection even though the clipping, sensitivity, and successful output-aware recovery provide strong range evidence.

## Candidate matrix and frozen selection

All candidates retain the canonical two-segment 8-bit functional semantics and 128/128 code allocation. They use output-aware post-SiLU calibration, an MSE-searched split, and differ only in the robust upper range:

| Candidate | Upper range | Development accuracy | Agreement vs Standard-QDQ | Max clip fraction | Mean local MSE | Selected |
|---|---|---:|---:|---:|---:|---|
| `two_segment_p99_9_mse` | per-site p99.9 | 99.90% | 99.95% | 0.0435% | 0.00008462 | no |
| `two_segment_p99_99_mse` | per-site p99.99 | 100.00% | 99.95% | 0.0050% | 0.00001463 | yes |
| `two_segment_observed_max_mse` | per-site observed maximum | 100.00% | 99.95% | 0% | 0.00000231 | no; configuration-order tiebreak |

The selected candidate was frozen before final-test loading with:

- configuration hash: `ddd245857fdfd1ba3db49c0286673d3e0c1ea06524154706a9cf205420638a54`
- calibration digest: `f3c7d509885f2b1b98905cb3f6af79fa50e3804ab95cca660e4bff2c1a833dad`
- development split digest: `894289d808b95c6c25f990236533151e64dc317c00167feebbba5575f04dab60`
- model SHA-256: `0df5e7f6388d39a658a53790788d20393ff3e8e481d075b85ca84a54e5821a8d`
- manifest hash: `ae9fd05e779c530e11ec11d632075a7b5dca767640b39aa021f7b8cffa9df9f4`
- selection receipt hash: `bd524f3ae909bfb27df211b6d37b95495aadb994b7f10e975a1e87aaf8385254`

## Final full-test comparison

All newly measured rows use ONNX Runtime 1.19.2 `CPUExecutionProvider` on all 10,000 test images. FP32 is cited from the locked v0.6.5 report and was not rerun.

| Model | Role | Top-1 accuracy | Gap vs Standard-QDQ | Notes |
|---|---|---:|---:|---|
| FP32 | historical/reference | 93.74% | — | locked v0.6.5 provenance |
| Standard-QDQ | unchanged deployment baseline | 93.57% | 0 | new full-test ORT CPU measurement |
| Original piecewise | preserved v0.6 reference | 82.80% | -10.77 pp | preserved model and hash |
| `two_segment_p99_99_mse` | selected v1.1 candidate | 93.58% | +0.01 pp | selection frozen on development data |

## Reproduction and generated reports

Use the repository environment without importing PyTorch in the ORT-only process:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python scripts\run_silu_accuracy_recovery.py --config configs\accuracy_recovery\silu_accuracy_recovery_v11.json
python scripts\run_control_equivalence_audit.py --batch-size 4
```

Generated models and reports are intentionally ignored:

- `artifacts/accuracy_recovery/v1.1/`
- `results/benchmarks/v1.1_accuracy_diagnosis/`
- `results/benchmarks/v1.1_accuracy_recovery_smoke/`
- `results/benchmarks/v1.1_accuracy_recovery/`
- `artifacts/accuracy_recovery/v1.1.1/`
- `results/benchmarks/v1.1_control_equivalence_audit/`

The report sets include JSON, CSV, and Markdown controls, activation statistics, layer sensitivity, candidate results, selection receipt, audit provenance, and final comparison.

## Limitations and recommendation

This result validates an ORT standard-operator functional piecewise-SiLU reference. It does not demonstrate integer-only execution, lower latency, a custom-op integration, the v1.0 C++ kernel in a full graph, or QNN/NPU/GPU portability. The selected graph is slower than Standard-QDQ because functional reference arithmetic is expanded into standard operators.

The next step is to freeze this candidate contract and verify the v1.0 C++ kernel against its generated per-site fixtures before any whole-model custom-op integration. Performance claims should wait for an integrated backend benchmark with numerical parity and unchanged full-test accuracy.
