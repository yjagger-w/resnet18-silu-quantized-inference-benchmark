# v1.8 source-anchored SiLU-aware standard QDQ

## Goal

v1.8 tests whether SiLU activation statistics can improve ordinary per-tensor uint8 QDQ calibration without adding custom operators, runtime branches, or a second activation scale. Piecewise SiLU ranges remain offline calibration hints only. The emitted graph still uses standard ONNX `QuantizeLinear` and `DequantizeLinear` nodes.

This is an incremental deployment experiment, not a claim that the earlier piecewise quantizer maps directly to mobile hardware.

## Why the first candidate was rejected

The first v1.8 candidate preserved the 246-node ONNX topology, compiled successfully for the Samsung Galaxy S22 5G, profiled at 0.41227 ms mean latency, and placed all 70 profiled nodes on the NPU. Nevertheless, its balanced 1,000-image device preflight reached only 75.1% top-1 accuracy, versus 94.0% in local ONNX Runtime and 93.8% for the previous standard-QDQ device baseline. Local/device prediction agreement was 76.8%.

The failure was caused by a calibration contract error rather than an unsupported graph. The search called a newly computed min-max encoding the baseline and was not anchored to the actual scale and zero-point values from the QNN-validated source model. Some of the 17 post-SiLU encodings therefore changed aggressively even though the operator topology stayed standard.

This result establishes an important boundary: successful compilation, full NPU placement, and plausible latency do not prove numerical portability.

The diagnostic AI Hub jobs were:

- Compile: `jprlnox9p` (`SUCCESS`)
- Profile: `jpxld3q3p` (`SUCCESS`)
- Balanced 1,000-image inference: `jgol44z1g` (`SUCCESS`, numerically rejected)

## Corrected calibration contract

For each discovered post-SiLU QDQ pair, the corrected workflow:

1. Reads the exact float32 scale and uint8 zero-point from the frozen source model.
2. Includes that source encoding as a mandatory candidate and deployment baseline.
3. Restricts scale search to 0.90–1.10 times the source scale.
4. Restricts the zero-point change to at most four integer codes.
5. Accepts a changed encoding only when it improves full-calibration weighted MSE by at least 0.5%.
6. Otherwise selects the exact source encoding and records the fallback reason.

The manifest records `source_qdq`, `selected_qdq`, the selected scale ratio, zero-point delta, objective improvement, and source-fallback status for every site. The rewrite continues to require an identical operator topology and zero added runtime nodes.

These bounds are conservative engineering defaults for this frozen model. They are not universal quantization constants.

## Final candidate decisions

| Candidate | Local top-1 | Delta vs standard QDQ | Local agreement | Galaxy S22 evidence | Decision |
|---|---:|---:|---:|---|---|
| Frozen standard QDQ | 93.57% | baseline | 100.00% | 93.68% on the complete 10,000-image test set | deployment fallback |
| Aggressive SiLU-aware calibration | 93.62% | +0.05 pp | 98.49% | 75.1% on the balanced 1,000-image preflight | rejected on device |
| Source-anchored bounded calibration | 93.39% | -0.18 pp | 97.99% | not submitted after the local gate failed | rejected locally |

The source-anchored candidate changed all 17 target QDQ pairs even though every change stayed within the configured scale and zero-point bounds. Its generated model SHA256 was `2F0CDF262AB3FEBCE2B731871A45F6285F10897AF97980E61318580BDDBC27EE`. The complete local gate correctly recorded `rejected_local_fallback_to_source`, `deployable=false`, and `accepted_for_device_preflight=false`.

Twenty focused tests covered deterministic calibration, ONNX QDQ semantics, source-parameter extraction, topology preservation, model execution, and the local promotion decision. All passed. This confirms implementation integrity; it does not override the failed accuracy gates.

No v1.8 candidate is promoted as a deployment model. The frozen v1.6 standard-QDQ model remains the final industrial recommendation.

## Deployment gate

The corrected model is not deployment-approved by a local accuracy result. Validation proceeds in this order:

1. Run unit, ONNX Runtime, topology, and deterministic-calibration tests.
2. Evaluate the complete 10,000-image local CIFAR-10 test set.
3. Require no top-1 regression against standard QDQ and at least 98% prediction agreement. A failed local gate records the source QDQ model as the deployment fallback and stops the workflow.
4. Compile for the target QNN device.
5. Run the fixed balanced 1,000-image Galaxy S22 preflight and compare it with both local output and the frozen standard-QDQ device baseline.
6. Stop if accuracy, prediction agreement, or finite-output checks regress materially.
7. Profile the passing candidate and reject CPU/GPU fallback or material latency regression.
8. Run the full 10,000-image device evaluation only after the accuracy and profile gates pass.

On any failed device gate, the deployment recommendation remains the frozen v1.6 standard-QDQ model. A calibration candidate is never promoted solely because it improves an offline error proxy or local top-1 accuracy.

## Limitations and industrial interpretation

- Weighted activation reconstruction error is only a search proxy; it is not end-to-end task accuracy.
- Standard ONNX QDQ syntax does not guarantee identical numerical lowering across inference backends.
- Calibration parameters remain model-, dataset-, backend-, and device-dependent.
- Custom silicon is not required for this workflow. Its practical value is learning how to express activation-aware calibration through standard QDQ parameters that existing NPUs already consume.
- A genuinely piecewise runtime encoding would require explicit backend/compiler support or a custom kernel and is therefore kept separate from the portable deployment path.
- The main industrial lesson is to co-design calibration with the target compiler contract, retain a known-good fallback, and treat device accuracy as a release gate rather than a post-release observation.

The realistic research contribution is therefore not “SiLU needs a custom chip.” It is a reproducible demonstration of where algorithm-only calibration stops, where backend behavior begins, and how to convert that boundary into a safer deployment workflow.
