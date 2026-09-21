# Tesla T4 CUDA Bias-SiLU repeated benchmark

This report aggregates five interleaved benchmark runs. Each configured iteration count applies independently to scalar, float4, and automatic dispatch.

## Environment

- Device: Tesla T4
- Compute capability: 7.5
- CUDA compiler: 12.4.131
- Host compiler: GNU 11.4.0
- Warm-up iterations per implementation: 50
- Measured iterations per implementation: 1000
- Execution order: interleaved round-robin with per-round rotation

## Median P50 latency

| Shape | Mode | Scalar (us) | Float4 (us) | Auto (us) | Auto path | Auto/scalar | Auto/best |
|---|---|---:|---:|---:|---|---:|---:|
| stem_64x32x32 | out-of-place | 6.016 | 5.440 | 5.472 | float4 | 1.0994x | 0.9942x |
| stem_64x32x32 | in-place | 5.632 | 5.024 | 5.008 | float4 | 1.1246x | 1.0032x |
| stage2_128x16x16 | out-of-place | 4.896 | 4.960 | 4.928 | scalar_threshold_fallback | 0.9935x | 0.9935x |
| stage2_128x16x16 | in-place | 4.768 | 4.768 | 4.768 | scalar_threshold_fallback | 1.0000x | 1.0000x |
| stage3_256x8x8 | out-of-place | 4.640 | 4.960 | 4.672 | scalar_threshold_fallback | 0.9932x | 0.9932x |
| stage3_256x8x8 | in-place | 4.544 | 4.704 | 4.544 | scalar_threshold_fallback | 1.0000x | 1.0000x |
| stage4_512x4x4 | out-of-place | 4.512 | 4.864 | 4.512 | scalar_threshold_fallback | 1.0000x | 1.0000x |
| stage4_512x4x4 | in-place | 4.480 | 4.736 | 4.448 | scalar_threshold_fallback | 1.0072x | 1.0072x |

## Source runs

| File | Bytes | SHA256 |
|---|---:|---|
| run_1.json | 14919 | `CE76BFF3C26251E8DA6954A921195FC7A6F210323688AEEC6DE5AEA79D80C145` |
| run_2.json | 14919 | `F0AC4CD40D0FC36C0519B00D9F5FE5F9135C1B00665B97F92714993EAB3015C5` |
| run_3.json | 14919 | `4EB71FB15955B8D3AB86C7AFF35DDFE585E0E57AFF2F5C8F32E8316A997873D6` |
| run_4.json | 14921 | `DB9D00BB137742EE717621B343D71B43AEA3E53842417BE85FC0E093D623E103` |
| run_5.json | 14920 | `7B41150B3EB8D45A91EB0FA77E7E593C12B9BD7A52F986545F72AC67F48827D4` |

## Interpretation boundary

- Latencies are CUDA Event measurements on the recorded Tesla T4.
- Allocation and host transfers are excluded as stated in the protocol.
- Auto/best values near 1.0 indicate that adaptive dispatch tracks the best explicit implementation within run-to-run variation.
- The 65,536-element automatic-dispatch threshold is specific to this T4/CUDA 12.4 evidence and is not a universal GPU threshold.
