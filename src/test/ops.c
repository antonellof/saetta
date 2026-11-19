#include <math.h>
#include <stdlib.h>

#include "../gpu.h"
#include "../ops.h"
#include "../qgen.h"
#include "../test.h"
#include "../tolerances.h"

// Zig's "{e}" for an f32: shortest round-trip digits, d.ddde<exp> (e.g. 1.5e-6, 3e0).
static const char *fmt_e(char *out, size_t cap, float x) {
    char tmp[48];
    for (int p = 0; p < 9; p++) {
        snprintf(tmp, sizeof tmp, "%.*e", p, (double)x);
        if ((float)strtod(tmp, NULL) == x) break;
    }
    char *e = strchr(tmp, 'e');
    if (!e) {
        snprintf(out, cap, "%s", tmp); // inf, nan
        return out;
    }
    *e = 0;
    snprintf(out, cap, "%se%d", tmp, atoi(e + 1));
    return out;
}

static const char *const mods[] = {"language_model.model.layers.0.mlp.shared_expert.gate_proj",
                                   "language_model.model.layers.0.mlp.gate",
                                   "language_model.model.layers.0.attn_hyper_connection.block_inject_weight"};

TEST(ops_qmatvec_gpu_matches_the_scalar_reference_on_real_pack_tensors) {
    const char *dir = getenv("SAETTA_PACK");
    if (!dir) SKIP("SAETTA_PACK not set");
    Arena arena = {0};
    Pack pack;
    EXPECT_OK(pack_open(&arena, dir, &pack));
    Gpu g;
    EXPECT_OK(gpu_init(&arena, &pack, &g));
    Ops ops;
    EXPECT_OK(ops_init(&g, &ops));
    // A 4-bit/32 and an 8-bit/64 linear from layer 0.
    for (int mi = 0; mi < 2; mi++) {
        Linear lin;
        EXPECT_OK(pack_linear(&pack, str(mods[mi]), &lin));
        sx_buffer *xb, *yb;
        EXPECT_OK(gpu_new_buffer(&g, (size_t)lin.in_features * 4, &xb));
        EXPECT_OK(gpu_new_buffer(&g, (size_t)lin.out_features * 4, &yb));
        float *x = gpu_floats(xb);
        QgenPrng rng;
        qgen_prng_init(&rng, 42);
        for (size_t i = 0; i < gpu_nfloats(xb); i++) x[i] = qgen_prng_float(&rng) * 2 - 1;
        EXPECT_OK(ops_qmatvec(&ops, &g, &lin, xb, yb));
        float *ref = ARENA_ARRAY(&arena, float, lin.out_features);
        ops_qmatvec_ref(&pack, &lin, x, ref);
        float e = ops_max_rel_err(gpu_floats(yb), ref, (size_t)lin.out_features);
        char eb[32];
        dbg("qmatvec %u-bit %s: max rel err %s\n", lin.bits, mods[mi], fmt_e(eb, sizeof eb, e));
        EXPECT(e < TOL_QMATVEC_REL);
    }
    gpu_deinit(&g);
    pack_close(&pack);
    arena_reset(&arena);
}

TEST(ops_qmm_gpu_matches_the_scalar_reference_per_token_on_real_pack_tensors) {
    const char *dir = getenv("SAETTA_PACK");
    if (!dir) SKIP("SAETTA_PACK not set");
    Arena arena = {0};
    Pack pack;
    EXPECT_OK(pack_open(&arena, dir, &pack));
    Gpu g;
    EXPECT_OK(gpu_init(&arena, &pack, &g));
    Ops ops;
    EXPECT_OK(ops_init(&g, &ops));
    const uint32_t T = 37; // not a multiple of the 32-token tile
    for (int mi = 0; mi < 3; mi++) {
        Linear lin;
        EXPECT_OK(pack_linear(&pack, str(mods[mi]), &lin));
        size_t in = (size_t)lin.in_features, out = (size_t)lin.out_features;
        sx_buffer *xb, *yb;
        EXPECT_OK(gpu_new_buffer(&g, T * in * 4, &xb));
        EXPECT_OK(gpu_new_buffer(&g, T * out * 4, &yb));
        float *x = gpu_floats(xb);
        QgenPrng rng;
        qgen_prng_init(&rng, 7);
        for (size_t i = 0; i < gpu_nfloats(xb); i++) x[i] = qgen_prng_float(&rng) * 2 - 1;
        EXPECT_OK(ops_qmm(&ops, &g, &lin, xb, yb, T));
        float *ref = ARENA_ARRAY(&arena, float, out);
        float worst = 0;
        for (size_t t = 0; t < T; t++) {
            ops_qmatvec_ref(&pack, &lin, x + t * in, ref);
            worst = fmaxf(worst, ops_max_rel_err(gpu_floats(yb) + t * out, ref, out));
        }
        // the decode kernel on the same rows, for the f32 summation-order floor
        float worst_mv = 0;
        sx_buffer *x1, *y1;
        EXPECT_OK(gpu_new_buffer(&g, in * 4, &x1));
        EXPECT_OK(gpu_new_buffer(&g, out * 4, &y1));
        for (size_t t = 0; t < T; t++) {
            memcpy(gpu_floats(x1), x + t * in, in * 4);
            EXPECT_OK(ops_qmatvec(&ops, &g, &lin, x1, y1));
            ops_qmatvec_ref(&pack, &lin, x + t * in, ref);
            worst_mv = fmaxf(worst_mv, ops_max_rel_err(gpu_floats(y1), ref, out));
        }
        char e1[32], e2[32];
        dbg("qmm %u-bit %s T=%u: max rel err %s (qmatvec on the same rows %s)\n", lin.bits, mods[mi], T,
            fmt_e(e1, sizeof e1, worst), fmt_e(e2, sizeof e2, worst_mv));
        EXPECT(worst < TOL_QMM_REL);
    }
    gpu_deinit(&g);
    pack_close(&pack);
    arena_reset(&arena);
}
