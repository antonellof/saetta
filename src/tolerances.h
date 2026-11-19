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

#endif
