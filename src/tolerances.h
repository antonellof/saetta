// Numerical tolerances for engine tests. Each value records the measurement behind it.
#ifndef SAETTA_TOLERANCES_H
#define SAETTA_TOLERANCES_H

// Quantized matvec, GPU against the scalar f64-accumulated reference, max error
// relative to the largest output. Measured 1.5e-6 (4-bit) and 1.7e-6 (8-bit) on
// layer 0 of the MLX pack, 2026-11-14; f32 summation order only.
#define TOL_QMATVEC_REL 1e-5f // f32

// Residual streams after each layer, engine (f32) against a numpy reference (f64), max error
// relative to the largest |value| of that vector. Measured 1.1e-6 to 5.9e-6 over the 48
// layers on the golden prompt, 2026-11-14 (CPU small ops, GPU f32 matvecs); about 8x headroom.
#define TOL_LAYER_REL 5e-5f // f32
// Logits at the dumped positions, same measure. Measured 1.2e-6.
#define TOL_LOGITS_REL 5e-5f // f32
// Greedy argmax may differ only where the reference's top-1 minus top-2 logit is below this.
#define TOL_ARGMAX_MARGIN 0.05 // f64

// Prefill matmul (qmm), GPU against the scalar f64 reference per token, max error relative
// to the largest output. Measured 1.8e-6 (4-bit, in 2560), 2.6e-6 (8-bit), 1.3e-5 on the
// 10240-wide hyper-connection inject (4 outputs; qmatvec 3.4e-6 there: qmm rounds each
// dequantized weight to f32 before the product), 2026-11-14.
#define TOL_QMM_REL 5e-5f // f32

// Accelerator prefill (half inputs to the 4-bit matmuls), against the numpy reference on the golden
// prompt. The gate is the output distribution: KL(ref || engine) of the last token, nats
// (measured 2.6e-5, 2026-11-15), plus equal top-1. The per-layer residual bound only catches
// gross bugs: half-precision drift grows with depth and peaks at layers 30-31 (measured 7.7e-2;
// layer 0 at 4.1e-4), the same layers where the exact path has its largest error.
#define TOL_PREFILL_FAST_KL 1e-3 // f64
#define TOL_PREFILL_FAST_LAYER_REL 0.2f // f32

// Run-time LoRA (src/kernels/lora.metal: down t = A x, up y += scale * B t, f16 weights, f32
// accumulation), GPU against the scalar f64 reference, max error relative to the largest output.
// Measured 2026-11-20 on random f16 weights in [-1, 1): down 2.2e-7 (T <= 8, tree sum) and
// 4.7e-6 (lora_mm, in 17408 summed in one accumulator), up 5.3e-7; f32 summation order only.
// The same bound as TOL_QMM_REL (about 10x headroom on the matmul path).
#define TOL_LORA_REL 5e-5f // f32

// DeepSeek-V4-Flash routed experts (src/kernels/dsv4_moe.metal), GPU against src/dsv4_ref.c.
// The reference rounds g, u, h and each expert's down output to bf16 (and h to e4m3 after
// act_quant); a different f32 summation order breaks a rounding tie the other way now and then,
// which moves that value by one bf16 ulp (2^-7 relative at most). Each stage is checked on the
// GPU's own input (src/test/dsv4_moe.c), so one flip stays one flip:
// - STAGE_REL: max |error| relative to the largest |value| of the vector, for h and the summed
//   down output y: two bf16 ulps of the largest value (2^-6). Measured 2026-11-20, random FP4
//   experts, T = 1, 3, 8: 3.2e-3 (h), 1.9e-3 (y).
// - DIFF_FRAC: share of values not bit-identical. Measured 2.7e-3 (h at T = 1; most are values
//   below 2^-118 that the GPU flushes to zero, which quantize to the same e4m3 zero), 6e-4 (y).
// - E2E_REL: y end to end, where an e4m3 code of hq may also differ (2 of 135168 codes in the
//   measurement): measured 1.9e-3; bound 3e-2.
// act_quant (x and h) has no tolerance: it is bit-exact.
#define TOL_DSV4_STAGE_REL 0.015625 // f64
#define TOL_DSV4_DIFF_FRAC 1e-2     // f64
#define TOL_DSV4_E2E_REL 3e-2       // f64

#endif
