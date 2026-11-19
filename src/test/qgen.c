#include <math.h>
#include <stdlib.h>

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
        snprintf(out, cap, "%s", tmp);
        return out;
    }
    *e = 0;
    snprintf(out, cap, "%se%d", tmp, atoi(e + 1));
    return out;
}

static float dot_ref(const float *w, const float *x, size_t n) {
    double acc = 0;
    for (size_t i = 0; i < n; i++) acc += (double)w[i] * (double)x[i];
    return (float)acc;
}

static float silu(float v) { return v / (1 + expf(-v)); }

#define NCONFIGS 9
static const uint32_t configs[NCONFIGS][2] = {{4, 32}, {4, 64}, {5, 32}, {5, 64}, {5, 128}, {6, 64}, {6, 128}, {8, 64}, {8, 128}};

static float *alloc_f(QgenCtx *x, size_t n) { return ARENA_ARRAY(x->arena, float, n); }

// false (and the message) when the error is not below bound.
static bool check(const char *name, const uint32_t cfg[2], const float *got, const float *ref, size_t n, float bound, float *e) {
    *e = ops_max_rel_err(got, ref, n);
    if (!(*e < bound)) {
        char a[32], b[32];
        dbg("%s %u/%u: max rel err %s (bound %s)\n", name, cfg[0], cfg[1], fmt_e(a, sizeof a, *e), fmt_e(b, sizeof b, bound));
        return false;
    }
    return true;
}
#define CHECK(worst, name, cfg, got, ref, n, bound)                                  \
    do { float e_; if (!check(name, cfg, got, ref, n, bound, &e_)) { *failed_ = 1; return; } \
        worst = fmaxf(worst, e_); } while (0)

static void report(float w, const char *name) {
    char a[32];
    dbg("qgen %s: worst max rel err %s over %d configs\n", name, fmt_e(a, sizeof a, w), NCONFIGS);
}

#define BYTES(p) gpu_bytes_arg((p), sizeof(*(p)))
#define BUF(b) gpu_buf_arg(b)
#define OFFS(name, ci, k) uint32_t name[3]; qgen_offs_for(ci, k, name)
#define MAT(m, E, out, in, cfg, offs) QgenMat m; EXPECT_OK(qgen_ctx_mat(&x, E, out, in, cfg, offs, &m))
#define FLOATS(b, n, lo, hi) sx_buffer *b; EXPECT_OK(qgen_ctx_floats(&x, n, lo, hi, &b))
#define NEWBUF(b, n) sx_buffer *b; EXPECT_OK(gpu_new_buffer(&x.g, n, &b))
#define RUN(src, name, args, gx, gy, th)                                                         \
    do { sx_pipeline *p_; EXPECT_OK(qgen_ctx_pipe(&x, src, name, &p_));                        \
        EXPECT_OK(qgen_ctx_run(&x, p_, args, COUNT(args), gx, gy, th)); } while (0)

TEST(qgen_decode_matvecs) {
    Arena arena = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&arena, &x));
    float w_mv = 0, w_ex = 0, w_dn = 0, w_rt = 0;
    for (size_t ci = 0; ci < NCONFIGS; ci++) {
        const uint32_t *cfg = configs[ci];
        const uint32_t *cfg2 = configs[(ci + 4) % NCONFIGS]; // the second matrix of two-matrix kernels
        { // qmatvec_gen
            enum { in = 1024, out = 72 };
            OFFS(o, ci, 0);
            MAT(m, 1, out, in, cfg, o);
            FLOATS(xb, in, -1, 1);
            NEWBUF(yb, out * 4);
            sx_arg a[3];
            qgen_mat_args(&m, a);
            uint32_t d[2] = {in, out}, q[2] = {m.bits, m.group};
            sx_arg args[] = {a[0], a[1], a[2], BUF(xb), BUF(yb), gpu_u32_arg(&d[0]), gpu_u32_arg(&d[1]), BYTES(&m.sh), BYTES(&q)};
            RUN(qgen_src, "qmatvec_gen", args, out, 1, 32);
            float *ref = alloc_f(&x, out), *wr = alloc_f(&x, in);
            for (size_t r = 0; r < out; r++) {
                qgen_mat_row(&m, 0, r, wr);
                ref[r] = dot_ref(wr, gpu_floats(xb), in);
            }
            CHECK(w_mv, "qmatvec_gen", cfg, gpu_floats(yb), ref, out, TOL_QMATVEC_REL);
            qgen_mat_deinit(&m);
        }
        { // qmatvec_expert_gen: 4 slots of 5 experts, slot s reads x row s / 2
            enum { in = 512, out = 40, E = 5 };
            OFFS(o, ci, 1);
            MAT(m, E, out, in, cfg, o);
            FLOATS(xb, 2 * in, -1, 1);
            NEWBUF(yb, 4 * out * 4);
            NEWBUF(idx, 16);
            const uint32_t ids[4] = {3, 0, 4, 2};
            memcpy(sx_buffer_contents(idx), ids, sizeof ids);
            sx_arg a[3];
            qgen_mat_args(&m, a);
            uint32_t d[2] = {in, out};
            uint32_t st[4] = {(uint32_t)qgen_mat_w_stride(&m), (uint32_t)qgen_mat_s_stride(&m), in, 2};
            uint32_t q[2] = {m.bits, m.group};
            sx_arg args[] = {a[0], a[1], a[2], BUF(xb), BUF(yb), gpu_u32_arg(&d[0]), gpu_u32_arg(&d[1]), BYTES(&m.sh), BUF(idx), BYTES(&st), BYTES(&q)};
            RUN(qgen_src, "qmatvec_expert_gen", args, out, 4, 32);
            float *ref = alloc_f(&x, 4 * out), *wr = alloc_f(&x, in);
            for (size_t s = 0; s < 4; s++)
                for (size_t r = 0; r < out; r++) {
                    qgen_mat_row(&m, ids[s], r, wr);
                    ref[s * out + r] = dot_ref(wr, gpu_floats(xb) + (s / 2) * in, in);
                }
            CHECK(w_ex, "qmatvec_expert_gen", cfg, gpu_floats(yb), ref, 4 * out, TOL_QMATVEC_REL);
            qgen_mat_deinit(&m);
        }
        { // qmv_down_sh_gen: 3 routed slots (cfg) and the shared expert (cfg2)
            enum { out = 48, E = 4, topk = 3, kin = 768 }; // kin: multiple of 32, 64, 128
            OFFS(o2, ci, 2);
            MAT(m, E, out, kin, cfg, o2);
            OFFS(o3, ci, 3);
            MAT(sm, 1, out, kin, cfg2, o3);
            FLOATS(xb, (topk + 1) * kin, -1, 1);
            NEWBUF(yb, (topk + 1) * out * 4);
            NEWBUF(idx, 16);
            const uint32_t ids[3] = {1, 3, 0};
            memcpy(sx_buffer_contents(idx), ids, sizeof ids);
            sx_arg a[3], sa[3];
            qgen_mat_args(&m, a);
            qgen_mat_args(&sm, sa);
            uint32_t d[4] = {kin, out, topk, 0};
            uint32_t st[2] = {(uint32_t)qgen_mat_w_stride(&m), (uint32_t)qgen_mat_s_stride(&m)};
            uint32_t q[4] = {cfg[0], cfg[1], cfg2[0], cfg2[1]};
            sx_arg args[] = {a[0], a[1], a[2], BUF(xb), BUF(yb), BYTES(&d), BYTES(&m.sh), BUF(idx), BYTES(&st), sa[0], sa[1], sa[2], BYTES(&sm.sh), BYTES(&q)};
            RUN(qgen_src, "qmv_down_sh_gen", args, out, topk + 1, 32);
            float *ref = alloc_f(&x, (topk + 1) * out), *wr = alloc_f(&x, kin);
            for (size_t s = 0; s < topk + 1; s++)
                for (size_t r = 0; r < out; r++) {
                    if (s < topk) qgen_mat_row(&m, ids[s], r, wr);
                    else qgen_mat_row(&sm, 0, r, wr);
                    ref[s * out + r] = dot_ref(wr, gpu_floats(xb) + s * kin, kin);
                }
            CHECK(w_dn, "qmv_down_sh_gen", cfg, gpu_floats(yb), ref, (topk + 1) * out, TOL_QMATVEC_REL);
            qgen_mat_deinit(&sm);
            qgen_mat_deinit(&m);
        }
        { // qmv_router_sg_gen: 64 router rows (cfg), 1 shared-gate row (cfg2)
            enum { in = 1024, out = 64 };
            OFFS(o4, ci, 4);
            MAT(m, 1, out, in, cfg, o4);
            OFFS(o5, ci, 5);
            MAT(gm, 1, 1, in, cfg2, o5);
            FLOATS(xb, in, -1, 1);
            NEWBUF(y0, out * 4);
            NEWBUF(y1, 16);
            sx_arg a[3], ga[3];
            qgen_mat_args(&m, a);
            qgen_mat_args(&gm, ga);
            uint32_t d[2] = {in, out};
            uint32_t q[4] = {cfg[0], cfg[1], cfg2[0], cfg2[1]};
            sx_arg args[] = {a[0], a[1], a[2], BUF(xb), BUF(y0), BYTES(&d), BYTES(&m.sh), ga[0], ga[1], ga[2], BUF(y1), BYTES(&gm.sh), BYTES(&q)};
            RUN(qgen_src, "qmv_router_sg_gen", args, out + 1, 1, 32);
            float *ref = alloc_f(&x, out + 1), *got = alloc_f(&x, out + 1), *wr = alloc_f(&x, in);
            for (size_t r = 0; r < out + 1; r++) {
                if (r < out) qgen_mat_row(&m, 0, r, wr);
                else qgen_mat_row(&gm, 0, 0, wr);
                ref[r] = dot_ref(wr, gpu_floats(xb), in);
            }
            memcpy(got, gpu_floats(y0), out * 4);
            got[out] = gpu_floats(y1)[0];
            CHECK(w_rt, "qmv_router_sg_gen", cfg, got, ref, out + 1, TOL_QMATVEC_REL);
            qgen_mat_deinit(&gm);
            qgen_mat_deinit(&m);
        }
    }
    report(w_mv, "qmatvec_gen");
    report(w_ex, "qmatvec_expert_gen");
    report(w_dn, "qmv_down_sh_gen");
    report(w_rt, "qmv_router_sg_gen");
    arena_reset(&arena);
}

TEST(qgen_lanes_per_row_matvecs) {
    Arena arena = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&arena, &x));
    float w_multi = 0, w_gu = 0, w_gush = 0, w_mix = 0;
    static const uint32_t Ls[3] = {8, 32, 256};
    for (size_t ci = 0; ci < NCONFIGS; ci++) {
        const uint32_t *cfg = configs[ci];
        // qmv_multi: 3 matrices with their own quantization, 2 tokens, with and without the input activation
        for (int li = 0; li < 3; li++)
            for (int ai = 0; ai < 2; ai++) {
                const uint32_t L = Ls[li];
                const bool act = ai == 1;
                enum { in = 1024, T = 2 };
                const size_t rows[3] = {40, 17, 300};
                const uint32_t *cf[3] = {cfg, configs[(ci + 3) % NCONFIGS], configs[(ci + 7) % NCONFIGS]};
                QgenMat ms[3];
                for (size_t k = 0; k < 3; k++) {
                    OFFS(o, ci, 6 + k);
                    EXPECT_OK(qgen_ctx_mat(&x, 1, rows[k], in, cf[k], o, &ms[k]));
                }
                FLOATS(xb, T * in, -1, 1);
                sx_buffer *yb[3];
                for (size_t k = 0; k < 3; k++) EXPECT_OK(gpu_new_buffer(&x.g, T * rows[k] * 4, &yb[k]));
                const float act_scale = 0.7f;
                MvGP p = {.in = in, .nmat = 3, .act = act ? 1 : 0, .act_scale = act_scale};
                sx_arg args[18];
                const uint32_t rpt = 256 / L;
                for (size_t k = 0; k < 4; k++) {
                    const QgenMat *m = &ms[MIN(k, (size_t)2)];
                    sx_arg a[3];
                    qgen_mat_args(m, a);
                    for (size_t j = 0; j < 3; j++) {
                        args[3 * k + j] = a[j];
                        p.shift[3 * k + j] = m->sh[j];
                    }
                    args[13 + k] = BUF(yb[MIN(k, (size_t)2)]);
                    if (k < 3) {
                        p.rows[k] = (uint32_t)rows[k];
                        p.tg0[k + 1] = p.tg0[k] + (p.rows[k] + rpt - 1) / rpt;
                        p.bits[k] = cf[k][0];
                        p.group[k] = cf[k][1];
                    }
                }
                args[12] = BUF(xb);
                args[17] = BYTES(&p);
                char name[32];
                snprintf(name, sizeof name, "qmv_multi_l%u_gen", L);
                RUN(qgen_src, name, args, p.tg0[3], T, 256);
                float *xin = alloc_f(&x, T * in);
                for (size_t i = 0; i < T * in; i++) {
                    float v = gpu_floats(xb)[i];
                    xin[i] = act ? silu(v * act_scale) : v;
                }
                for (size_t k = 0; k < 3; k++) {
                    float *ref = alloc_f(&x, T * rows[k]), *wr = alloc_f(&x, in);
                    for (size_t r = 0; r < rows[k]; r++) {
                        qgen_mat_row(&ms[k], 0, r, wr);
                        for (size_t t = 0; t < T; t++) ref[t * rows[k] + r] = dot_ref(wr, xin + t * in, in);
                    }
                    CHECK(w_multi, name, cf[k], gpu_floats(yb[k]), ref, T * rows[k], TOL_QMATVEC_REL);
                }
                for (size_t k = 0; k < 3; k++) qgen_mat_deinit(&ms[k]);
            }
        { // qmv_gate_up_gen: routed (idx, 2 tokens x 3 slots, div 3) and shared (st.z = 0)
            enum { in = 768, out = 40, E = 4 };
            for (int ri = 0; ri < 2; ri++) {
                const bool routed = ri == 0;
                OFFS(o9, ci, 9);
                MAT(mg, routed ? E : 1, out, in, cfg, o9);
                OFFS(o10, ci, 10);
                MAT(mu, routed ? E : 1, out, in, cfg, o10);
                const size_t slots = routed ? 6 : 1;
                FLOATS(xb, 2 * in, -1, 1);
                NEWBUF(hb, slots * out * 4);
                NEWBUF(idx, 32);
                const uint32_t ids[6] = {2, 0, 3, 3, 1, 0};
                memcpy(sx_buffer_contents(idx), ids, sizeof ids);
                sx_arg ga[3], ua[3];
                qgen_mat_args(&mg, ga);
                qgen_mat_args(&mu, ua);
                uint32_t d[4] = {in, out, in, 3};
                uint32_t st[4] = {routed ? (uint32_t)qgen_mat_w_stride(&mg) : 0, routed ? (uint32_t)qgen_mat_s_stride(&mg) : 0, routed ? 1 : 0, 0};
                uint32_t sh[6] = {mg.sh[0], mg.sh[1], mg.sh[2], mu.sh[0], mu.sh[1], mu.sh[2]};
                uint32_t q[2] = {mg.bits, mg.group};
                sx_arg args[] = {ga[0], ga[1], ga[2], ua[0], ua[1], ua[2], BUF(xb), BUF(hb), BUF(idx), BYTES(&d), BYTES(&st), BYTES(&sh), BYTES(&q)};
                RUN(qgen_src, "qmv_gate_up_gen", args, out, slots, 32);
                float *ref = alloc_f(&x, slots * out), *wg = alloc_f(&x, in), *wu = alloc_f(&x, in);
                for (size_t s = 0; s < slots; s++)
                    for (size_t r = 0; r < out; r++) {
                        size_t e = routed ? ids[s] : 0;
                        qgen_mat_row(&mg, e, r, wg);
                        qgen_mat_row(&mu, e, r, wu);
                        const float *xs = gpu_floats(xb) + (s / 3) * in;
                        ref[s * out + r] = silu(dot_ref(wg, xs, in)) * dot_ref(wu, xs, in);
                    }
                CHECK(w_gu, "qmv_gate_up_gen", cfg, gpu_floats(hb), ref, slots * out, TOL_QMATVEC_REL);
                qgen_mat_deinit(&mu);
                qgen_mat_deinit(&mg);
            }
        }
        { // qmv_gate_up_sh_gen: 3 routed slots (cfg) + the shared expert (cfg2)
            enum { in = 768, out = 40, E = 4, topk = 3 };
            const uint32_t *cfg2 = configs[(ci + 5) % NCONFIGS];
            OFFS(o11, ci, 11);
            MAT(mg, E, out, in, cfg, o11);
            OFFS(o12, ci, 12);
            MAT(mu, E, out, in, cfg, o12);
            OFFS(o13, ci, 13);
            MAT(sg, 1, out, in, cfg2, o13);
            OFFS(o14, ci, 14);
            MAT(su, 1, out, in, cfg2, o14);
            FLOATS(xb, in, -1, 1);
            NEWBUF(hb, (topk + 1) * out * 4);
            NEWBUF(idx, 16);
            const uint32_t ids[3] = {3, 1, 2};
            memcpy(sx_buffer_contents(idx), ids, sizeof ids);
            sx_arg ga[3], ua[3], sga[3], sua[3];
            qgen_mat_args(&mg, ga);
            qgen_mat_args(&mu, ua);
            qgen_mat_args(&sg, sga);
            qgen_mat_args(&su, sua);
            uint32_t d[4] = {in, out, topk, 0};
            uint32_t st[4] = {(uint32_t)qgen_mat_w_stride(&mg), (uint32_t)qgen_mat_s_stride(&mg), 0, 0};
            uint32_t sh[6] = {mg.sh[0], mg.sh[1], mg.sh[2], mu.sh[0], mu.sh[1], mu.sh[2]};
            uint32_t sh2[6] = {sg.sh[0], sg.sh[1], sg.sh[2], su.sh[0], su.sh[1], su.sh[2]};
            uint32_t q[4] = {cfg[0], cfg[1], cfg2[0], cfg2[1]};
            sx_arg args[] = {ga[0], ga[1], ga[2], ua[0], ua[1], ua[2], BUF(xb), BUF(hb), BUF(idx), BYTES(&d), BYTES(&st), BYTES(&sh),
                             sga[0], sga[1], sga[2], sua[0], sua[1], sua[2], BYTES(&sh2), BYTES(&q)};
            RUN(qgen_src, "qmv_gate_up_sh_gen", args, out, topk + 1, 32);
            float *ref = alloc_f(&x, (topk + 1) * out), *wg = alloc_f(&x, in), *wu = alloc_f(&x, in);
            for (size_t s = 0; s < topk + 1; s++)
                for (size_t r = 0; r < out; r++) {
                    if (s < topk) {
                        qgen_mat_row(&mg, ids[s], r, wg);
                        qgen_mat_row(&mu, ids[s], r, wu);
                    } else {
                        qgen_mat_row(&sg, 0, r, wg);
                        qgen_mat_row(&su, 0, r, wu);
                    }
                    const float *xs = gpu_floats(xb);
                    ref[s * out + r] = silu(dot_ref(wg, xs, in)) * dot_ref(wu, xs, in);
                }
            CHECK(w_gush, "qmv_gate_up_sh_gen", cfg, gpu_floats(hb), ref, (topk + 1) * out, TOL_QMATVEC_REL);
            qgen_mat_deinit(&su);
            qgen_mat_deinit(&sg);
            qgen_mat_deinit(&mu);
            qgen_mat_deinit(&mg);
        }
        { // qmv_up_mix_gen: H = 16, HC = 4 streams, in 256, 2 tokens
            enum { in = 256, H = 16, HC = 4, T = 2 };
            OFFS(o15, ci, 15);
            MAT(m, 1, HC * H, in, cfg, o15);
            FLOATS(low, T * in, -2, 2);
            FLOATS(xn, T * HC * H, -1, 1);
            NEWBUF(ab, T * H * 4);
            sx_arg a[3];
            qgen_mat_args(&m, a);
            uint32_t p[4] = {in, H, HC, 0};
            const float act_scale = 0.25f;
            uint32_t q[2] = {m.bits, m.group};
            sx_arg args[] = {a[0], a[1], a[2], BUF(low), BUF(xn), BUF(ab), BYTES(&p), BYTES(&act_scale), BYTES(&m.sh), BYTES(&q)};
            RUN(qgen_src, "qmv_up_mix_gen", args, H / 8, T, 256);
            float *ref = alloc_f(&x, T * H), *wr = alloc_f(&x, in), *xin = alloc_f(&x, in);
            for (size_t t = 0; t < T; t++) {
                for (size_t i = 0; i < in; i++) xin[i] = silu(gpu_floats(low)[t * in + i] * act_scale);
                for (size_t j = 0; j < H; j++) {
                    float s = 0;
                    for (size_t k = 0; k < HC; k++) {
                        qgen_mat_row(&m, 0, k * H + j, wr);
                        s += 1 / (1 + expf(-dot_ref(wr, xin, in))) * gpu_floats(xn)[t * HC * H + k * H + j];
                    }
                    ref[t * H + j] = s / HC;
                }
            }
            CHECK(w_mix, "qmv_up_mix_gen", cfg, gpu_floats(ab), ref, T * H, TOL_QMATVEC_REL);
            qgen_mat_deinit(&m);
        }
    }
    report(w_multi, "qmv_multi_l{8,32,256}_gen");
    report(w_gu, "qmv_gate_up_gen");
    report(w_gush, "qmv_gate_up_sh_gen");
    report(w_mix, "qmv_up_mix_gen");
    arena_reset(&arena);
}

TEST(qgen_verification_chunk_rows) {
    Arena arena = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&arena, &x));
    float w1 = 0, w4 = 0;
    for (size_t ci = 0; ci < NCONFIGS; ci++) {
        const uint32_t *cfg = configs[ci];
        enum { in = 1280, out = 44 }; // in not a multiple of 256: the shared-input layout's last lane step is partial
        OFFS(o, ci, 16);
        MAT(m, 1, out, in, cfg, o);
        FLOATS(xb, 8 * in, -1, 1);
        NEWBUF(yb, 8 * out * 4);
        float *ref = alloc_f(&x, 8 * out), *wr = alloc_f(&x, in);
        for (size_t r = 0; r < out; r++) {
            qgen_mat_row(&m, 0, r, wr);
            for (size_t t = 0; t < 8; t++) ref[t * out + r] = dot_ref(wr, gpu_floats(xb) + t * in, in);
        }
        sx_arg a[3];
        qgen_mat_args(&m, a);
        uint32_t q[2] = {m.bits, m.group};
        for (uint32_t T = 1; T < 9; T++) {
            uint32_t d[3] = {in, out, T};
            sx_arg args[] = {a[0], a[1], a[2], BUF(xb), BUF(yb), BYTES(&d), BYTES(&m.sh), BYTES(&q)};
            char name[32];
            snprintf(name, sizeof name, "qmv_rows1_gen_%u", T);
            RUN(qgen_src, name, args, out, 1, 32);
            CHECK(w1, "qmv_rows1_gen", cfg, gpu_floats(yb), ref, T * out, TOL_QMATVEC_REL);
            memset(gpu_floats(yb), 0, gpu_nfloats(yb) * 4);
            snprintf(name, sizeof name, "qmv_rows_gen_%u", T);
            RUN(qgen_src, name, args, (out + 7) / 8, 1, 64);
            CHECK(w4, "qmv_rows_gen", cfg, gpu_floats(yb), ref, T * out, TOL_QMATVEC_REL);
            memset(gpu_floats(yb), 0, gpu_nfloats(yb) * 4);
        }
        qgen_mat_deinit(&m);
    }
    report(w1, "qmv_rows1_gen_1..8");
    report(w4, "qmv_rows_gen_1..8");
    arena_reset(&arena);
}

TEST(qgen_exact_prefill_matmuls) {
    Arena arena = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&arena, &x));
    float wd = 0, wg = 0;
    for (size_t ci = 0; ci < NCONFIGS; ci++) {
        const uint32_t *cfg = configs[ci];
        { // dense: T = 37 tokens (not a multiple of the 32-token tile), 72 rows
            enum { in = 256, out = 72, T = 37 };
            OFFS(o, ci, 17);
            MAT(m, 1, out, in, cfg, o);
            FLOATS(xb, T * in, -1, 1);
            NEWBUF(yb, T * out * 4);
            sx_arg a[3];
            qgen_mat_args(&m, a);
            uint32_t d[3] = {in, out, T};
            uint32_t q[2] = {m.bits, m.group};
            sx_arg args[] = {a[0], a[1], a[2], BUF(xb), BUF(yb), BYTES(&d), BYTES(&m.sh), BYTES(&q)};
            RUN(qgen_src, "qmm_gen", args, (out + 31) / 32, (T + 31) / 32, 128);
            float *ref = alloc_f(&x, T * out), *wr = alloc_f(&x, in);
            for (size_t r = 0; r < out; r++) {
                qgen_mat_row(&m, 0, r, wr);
                for (size_t t = 0; t < T; t++) ref[t * out + r] = dot_ref(wr, gpu_floats(xb) + t * in, in);
            }
            CHECK(wd, "qmm_gen", cfg, gpu_floats(yb), ref, T * out, TOL_QMM_REL);
            qgen_mat_deinit(&m);
        }
        { // grouped: 3 experts, tiles of at most 32 rows, one grid row past ntiles
            enum { in = 256, out = 40, E = 3, M = 50 };
            OFFS(o, ci, 18);
            MAT(m, E, out, in, cfg, o);
            FLOATS(xb, M * in, -1, 1);
            NEWBUF(yb, M * out * 4);
            const uint32_t tiles[3][4] = {{2, 0, 20, 0}, {0, 20, 37, 0}, {1, 37, 50, 0}};
            NEWBUF(tb, sizeof tiles);
            memcpy(sx_buffer_contents(tb), tiles, sizeof tiles);
            NEWBUF(nt, 4);
            *(uint32_t *)sx_buffer_contents(nt) = 3;
            sx_arg a[3];
            qgen_mat_args(&m, a);
            uint32_t d[3] = {in, out, 0};
            uint32_t st[2] = {(uint32_t)qgen_mat_w_stride(&m), (uint32_t)qgen_mat_s_stride(&m)};
            uint32_t q[2] = {m.bits, m.group};
            sx_arg args[] = {a[0], a[1], a[2], BUF(xb), BUF(yb), BYTES(&d), BYTES(&m.sh), BUF(tb), BUF(nt), BYTES(&st), BYTES(&q)};
            RUN(qgen_src, "qmm_grouped_gen", args, (out + 31) / 32, 3 + 1, 128);
            float *ref = alloc_f(&x, M * out), *wr = alloc_f(&x, in);
            for (size_t ti = 0; ti < 3; ti++)
                for (size_t r = 0; r < out; r++) {
                    qgen_mat_row(&m, tiles[ti][0], r, wr);
                    for (size_t row = tiles[ti][1]; row < tiles[ti][2]; row++)
                        ref[row * out + r] = dot_ref(wr, gpu_floats(xb) + row * in, in);
                }
            CHECK(wg, "qmm_grouped_gen", cfg, gpu_floats(yb), ref, M * out, TOL_QMM_REL);
            qgen_mat_deinit(&m);
        }
    }
    report(wd, "qmm_gen");
    report(wg, "qmm_grouped_gen");
    arena_reset(&arena);
}

TEST(qgen_accelerator_prompt_kernels) {
    Arena arena = {0};
    QgenCtx x;
    EXPECT_OK(qgen_ctx_init(&arena, &x));
    // Two errors per kernel: against the f32 reference on the same half inputs (what half weights
    // cost, reported like the existing half kernels), and against a reference that rounds the
    // dequantized weights like the kernel (half fma), which must meet qmm_rel.
    size_t dq_ulp = 0;
    float dq_err = 0, dn_f32 = 0, dn_h = 0, gr_f32 = 0, gr_h = 0;
    sx_pipeline *dq;
    Err pe = qgen_ctx_pipe(&x, qgen_mpp_src, "dq_f16_gen", &dq);
    if (pe) {
        dbg("qgen: accelerator kernels do not compile here (%s), skipped\n", pe);
        SKIP("accelerator kernels do not compile");
    }
    for (size_t ci = 0; ci < NCONFIGS; ci++) {
        const uint32_t *cfg = configs[ci];
        { // dq_f16_gen: K = 384, N = 50
            enum { K = 384, N = 50 };
            OFFS(o, ci, 19);
            MAT(m, 1, N, K, cfg, o);
            NEWBUF(wh, N * K * 2);
            sx_arg a[3];
            qgen_mat_args(&m, a);
            uint32_t d[2] = {K, N};
            uint32_t q[2] = {m.bits, m.group};
            sx_arg args[] = {a[0], a[1], a[2], BUF(wh), BYTES(&d), BYTES(&m.sh), BYTES(&q)};
            EXPECT_OK(qgen_ctx_run(&x, dq, args, COUNT(args), (N * K / 8 + 255) / 256, 1, 256));
            const _Float16 *got = sx_buffer_contents(wh);
            float *wr = alloc_f(&x, K), *we = alloc_f(&x, K), *gf = alloc_f(&x, N * K), *ef = alloc_f(&x, N * K);
            for (size_t n = 0; n < N; n++) {
                qgen_mat_row_half(&m, 0, n, wr);
                qgen_mat_row(&m, 0, n, we);
                for (size_t k = 0; k < K; k++) {
                    float v = (float)got[n * K + k];
                    if (v != wr[k]) dq_ulp++;
                    gf[n * K + k] = v;
                    ef[n * K + k] = we[k];
                }
            }
            dq_err = fmaxf(dq_err, ops_max_rel_err(gf, ef, N * K));
            qgen_mat_deinit(&m);
        }
        { // qmmh_gen: K = 256, N = 100, M = 70
            enum { K = 256, N = 100, M = 70 };
            OFFS(o, ci, 20);
            MAT(m, 1, N, K, cfg, o);
            sx_buffer *xb;
            EXPECT_OK(qgen_ctx_halfs(&x, M * K, &xb));
            NEWBUF(yb, M * N * 4);
            sx_arg a[3];
            qgen_mat_args(&m, a);
            uint32_t d[3] = {K, N, M};
            uint32_t q[2] = {m.bits, m.group};
            sx_arg args[] = {a[0], a[1], a[2], BUF(xb), BUF(yb), BYTES(&d), BYTES(&m.sh), BYTES(&q)};
            RUN(qgen_mpp_src, "qmmh_gen", args, (N + 63) / 64, (M + 63) / 64, 128);
            const _Float16 *xh = sx_buffer_contents(xb);
            float *xf = alloc_f(&x, M * K);
            for (size_t i = 0; i < M * K; i++) xf[i] = (float)xh[i];
            float *r32 = alloc_f(&x, M * N), *rh = alloc_f(&x, M * N), *wr = alloc_f(&x, K);
            for (size_t n = 0; n < N; n++) {
                qgen_mat_row(&m, 0, n, wr);
                for (size_t t = 0; t < M; t++) r32[t * N + n] = dot_ref(wr, xf + t * K, K);
                qgen_mat_row_half(&m, 0, n, wr);
                for (size_t t = 0; t < M; t++) rh[t * N + n] = dot_ref(wr, xf + t * K, K);
            }
            dn_f32 = fmaxf(dn_f32, ops_max_rel_err(gpu_floats(yb), r32, M * N));
            CHECK(dn_h, "qmmh_gen", cfg, gpu_floats(yb), rh, M * N, TOL_QMM_REL);
            qgen_mat_deinit(&m);
        }
        { // qmmh_g_gen: 3 experts, tiles of at most 64 rows, K = 256, N = 128
            enum { K = 256, N = 128, E = 3, M = 100 };
            OFFS(o, ci, 21);
            MAT(m, E, N, K, cfg, o);
            sx_buffer *xb;
            EXPECT_OK(qgen_ctx_halfs(&x, M * K, &xb));
            NEWBUF(yb, M * N * 4);
            const uint32_t tiles[3][4] = {{2, 0, 40, 0}, {0, 40, 70, 0}, {1, 70, 100, 0}};
            NEWBUF(tb, sizeof tiles);
            memcpy(sx_buffer_contents(tb), tiles, sizeof tiles);
            NEWBUF(nt, 4);
            *(uint32_t *)sx_buffer_contents(nt) = 3;
            sx_arg a[3];
            qgen_mat_args(&m, a);
            uint32_t d[3] = {K, N, M};
            uint32_t st[2] = {(uint32_t)qgen_mat_w_stride(&m), (uint32_t)qgen_mat_s_stride(&m)};
            uint32_t q[2] = {m.bits, m.group};
            sx_arg args[] = {a[0], a[1], a[2], BUF(xb), BUF(yb), BYTES(&d), BYTES(&m.sh), BUF(tb), BUF(nt), BYTES(&st), BYTES(&q)};
            RUN(qgen_mpp_src, "qmmh_g_gen", args, N / 64, 3 + 1, 128);
            const _Float16 *xh = sx_buffer_contents(xb);
            float *xf = alloc_f(&x, M * K);
            for (size_t i = 0; i < M * K; i++) xf[i] = (float)xh[i];
            float *r32 = alloc_f(&x, M * N), *rh = alloc_f(&x, M * N), *wr = alloc_f(&x, K);
            for (size_t ti = 0; ti < 3; ti++)
                for (size_t n = 0; n < N; n++) {
                    qgen_mat_row(&m, tiles[ti][0], n, wr);
                    for (size_t row = tiles[ti][1]; row < tiles[ti][2]; row++) r32[row * N + n] = dot_ref(wr, xf + row * K, K);
                    qgen_mat_row_half(&m, tiles[ti][0], n, wr);
                    for (size_t row = tiles[ti][1]; row < tiles[ti][2]; row++) rh[row * N + n] = dot_ref(wr, xf + row * K, K);
                }
            gr_f32 = fmaxf(gr_f32, ops_max_rel_err(gpu_floats(yb), r32, M * N));
            CHECK(gr_h, "qmmh_g_gen", cfg, gpu_floats(yb), rh, M * N, TOL_QMM_REL);
            qgen_mat_deinit(&m);
        }
    }
    char a[32], b[32];
    dbg("qgen dq_f16_gen: %zu weights differ from half fma rounding; max rel err vs f32 weights %s\n", dq_ulp, fmt_e(a, sizeof a, dq_err));
    dbg("qgen qmmh_gen: max rel err %s vs f32 weights, %s vs half-rounded weights\n", fmt_e(a, sizeof a, dn_f32), fmt_e(b, sizeof b, dn_h));
    dbg("qgen qmmh_g_gen: max rel err %s vs f32 weights, %s vs half-rounded weights\n", fmt_e(a, sizeof a, gr_f32), fmt_e(b, sizeof b, gr_h));
    arena_reset(&arena);
}
