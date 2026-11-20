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

// DeepSeek-V4-Flash attention kernels (src/kernels/dsv4_attn.metal) against the scalar
// references (src/dsv4_ref_attn.c) on random data, 2026-11-20 (src/test/dsv4_attn.c). BF16
// outputs; the references sum in f64 where the kernels sum in f32 in another order, so a value
// can sit on the other side of a rounding boundary.
// Query tail (per-head norm, each op BF16, RoPE): every value within one BF16 step, at most this
// fraction not identical. Measured 0 (4 x 64 heads x 512 at 8 position/theta settings).
#define TOL_DSV4_ATTN_Q_FRAC 1e-3 // f64
// KV row (RMSNorm, RoPE, FP8 qdq of the nope dims): fraction not identical, and the largest
// difference relative to the largest value of its 64-block, one e4m3 step (2^-3) when a norm
// output flips across an FP8 rounding or scale boundary. Measured 0 and 0.
#define TOL_DSV4_ATTN_KV_FRAC 1e-3 // f64
#define TOL_DSV4_ATTN_KV_REL 0.125 // f64
// Sparse attention output o (after de-rotation), per row: the largest difference relative to
// the largest |o| of its head, and the fraction of BF16 values not identical. An f32 ulp of a
// score can flip one BF16-rounded probability (2^-8 of that key's weight) and an output
// rounding is one BF16 step (2^-8 relative at the head's max). Measured 5.9e-3 and 8.3e-3 over
// 23 cases (window only, HCA and CSA lists, chunks, logits up to about 180, sinks).
#define TOL_DSV4_ATTN_O_REL 1.5e-2 // f64
#define TOL_DSV4_ATTN_O_FRAC 3e-2 // f64
// RMS of (GPU - reference) over RMS of (plain f64 softmax - reference): the GPU must reproduce
// the blockwise BF16 probability rounding, not just softmax. Measured at most 0.20.
#define TOL_DSV4_ATTN_O_RMS_RATIO 0.5 // f64
// wo_a (BF16 grouped matvec, 4096-wide dots): every value within one BF16 step, at most this
// fraction not identical. Measured 2.4e-4.
#define TOL_DSV4_ATTN_WOA_FRAC 2e-3 // f64

#endif
