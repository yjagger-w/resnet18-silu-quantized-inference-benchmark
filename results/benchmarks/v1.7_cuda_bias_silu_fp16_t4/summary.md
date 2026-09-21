# Tesla T4 FP16 CUDA Bias-SiLU results

- Implementation commit: `0745b5f6a6183a890abe0fde4d100a08ef1f309c`
- Dispatch threshold: `65,536` elements
- CTest: `11/11 passed`
- Compute Sanitizer memcheck: `0 errors`
- Maximum absolute error: `0.00261`
- Scalar / half2 registers: `24 / 26`
- Stack, shared and local memory: `0` for both kernels

| Shape | Mode | Scalar P50 | Half2 P50 | Auto P50 | Auto path | Auto/scalar |
|---|---|---:|---:|---:|---|---:|
| stem_64x32x32 | out-of-place | 6.144 us | 5.568 us | 5.536 us | half2 | 1.1098x |
| stem_64x32x32 | in-place | 5.952 us | 5.184 us | 5.280 us | half2 | 1.1273x |
| stage2_128x16x16 | out-of-place | 5.184 us | 5.056 us | 5.216 us | scalar_threshold_fallback | 0.9939x |
| stage2_128x16x16 | in-place | 5.120 us | 4.864 us | 4.960 us | scalar_threshold_fallback | 1.0323x |
| stage3_256x8x8 | out-of-place | 4.832 us | 4.864 us | 4.928 us | scalar_threshold_fallback | 0.9805x |
| stage3_256x8x8 | in-place | 4.544 us | 4.544 us | 4.576 us | scalar_threshold_fallback | 0.9930x |
| stage4_512x4x4 | out-of-place | 4.512 us | 4.608 us | 4.512 us | scalar_threshold_fallback | 1.0000x |
| stage4_512x4x4 | in-place | 4.416 us | 4.576 us | 4.448 us | scalar_threshold_fallback | 0.9928x |

## Conclusion

The adaptive policy selects half2 for the 64x32x32 stem tensor and scalar for the smaller stages. The stem improves by about 1.11x–1.13x while preserving the validated FP16 numerical tolerance.

The 65,536-element threshold is supported by three repeated Tesla T4 runs and static CUDA resource analysis. It is not presented as a universal cross-GPU threshold.

Below the threshold, automatic dispatch and explicit scalar execute the same scalar kernel. Small differences around 1.0x therefore represent measurement variation, not a separate automatic-dispatch optimization.
