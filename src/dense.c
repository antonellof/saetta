// Qwen3.8-27B forward pass (see model.h and dense.h). Exports dense_arch (model_arch.h).
// Formulas follow the Hugging Face transformers qwen3_5 model (M27, its modeling file) on an
// MLX pack (norm weights stored as 1 + w by mlx-lm's qwen3_5 conversion) and are checked
// against a numpy reference by `saetta verify`. The encoder helpers, the GDN and the
// session code follow src/model.c; the hyper-connections, MoE, PLE and the QSA indexer are
// not in this model.
#include "dense.h"
#define ARCH_PFX dense_
#include "model_rename.h"

#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "attn27.h"
#include "json.h"
#include "lora.h"
#include "ops.h"
#include "qgen.h"
#include "safetensors.h"

#define H CONFIG_HIDDEN_SIZE        // 5120
#define FFN CONFIG_INTERMEDIATE_SIZE // 17408
#define EPS CONFIG_RMS_NORM_EPS
// GDN
#define NK CONFIG_LINEAR_NUM_KEY_HEADS     // 16
#define NV CONFIG_LINEAR_NUM_VALUE_HEADS   // 48
#define DK CONFIG_LINEAR_KEY_HEAD_DIM      // 128
#define DV CONFIG_LINEAR_VALUE_HEAD_DIM    // 128
#define KD (NK * DK)                       // 2048
#define VD (NV * DV)                       // 6144
#define CONV (KD * 2 + VD)                 // 10240
#define KCONV CONFIG_LINEAR_CONV_KERNEL_DIM // 4
// attention
#define NQ CONFIG_NUM_ATTENTION_HEADS  // 24
#define NKV CONFIG_NUM_KEY_VALUE_HEADS // 4
#define HD CONFIG_HEAD_DIM             // 256
#define ROT ((size_t)((float)HD * CONFIG_PARTIAL_ROTARY_FACTOR)) // 64
#define VOCAB CONFIG_VOCAB_SIZE
#define NLAYERS CONFIG_NUM_HIDDEN_LAYERS
#define SMALL MODEL_SMALL
static_assert(NQ / NKV == 6 && HD == 256); // attn_full_6 shape (src/kernels/dense.metal)
static_assert(NQ == ATTN27_NQ && NKV == ATTN27_NKV && HD == ATTN27_HD); // attn_fl shape (src/kernels/dense_attn.metal)

// Attention splits: decode and verification chunks up to NSPLIT_DEC (few queries, long rows),
// prompt chunks NSPLIT_PRE over slices of AB queries.
#define NSPLIT_DEC 32
#define NSPLIT_PRE 8
#define AB 512

// n_kv: KV planes per session, n_attn plus one for the MTP head when model_load loaded it
// (sessions are created after the model, which they follow).
static size_t n_gdn, n_attn, n_kv;
__attribute__((constructor)) static void count_layers(void) {
    size_t n = 0;
    for (size_t i = 0; i < NLAYERS; i++) n += config_layer_kinds[i] == LAYER_KIND_GDN;
    n_gdn = n;
    n_attn = NLAYERS - n;
    n_kv = n_attn;
}

static Err params_take(Params *p, size_t n, Vec *out) {
    float *all = gpu_floats(p->buf);
    if (p->used + n > gpu_nfloats(p->buf)) return "ParamsFull";
    *out = (Vec){p->used, all + p->used, n};
    p->used += (n + 63) & ~(size_t)63;
    return OK;
}

// Decode scratch (floats).
constexpr size_t S_X = 0;
constexpr size_t S_a = S_X + H;
constexpr size_t S_y = S_a + H;
constexpr size_t S_qkv = S_y + H;
constexpr size_t S_z = S_qkv + CONV;
constexpr size_t S_gb = S_z + VD;
constexpr size_t S_ga = S_gb + 64;
constexpr size_t S_conv = S_ga + 64;
constexpr size_t S_o = S_conv + CONV;
constexpr size_t S_qg = S_o + VD;
constexpr size_t S_qn = S_qg + NQ * 2 * HD;
constexpr size_t S_kx = S_qn + NQ * HD;
constexpr size_t S_vx = S_kx + NKV * HD;
constexpr size_t S_att = S_vx + NKV * HD;
constexpr size_t S_g = S_att + NQ * HD;
constexpr size_t S_u = S_g + FFN;
constexpr size_t S_hh = S_u + FFN;
constexpr size_t S_h = S_hh + FFN;
constexpr size_t S_lt = S_h + H; // LoRA t = A x, [LORA_MAX_RT]
#define LORA_MAX_RT (3 * LORA_MAX_R)
constexpr size_t S_total = S_lt + LORA_MAX_RT;

// Prompt chunk scratch (floats).
constexpr size_t B_X = 0;
constexpr size_t B_a = B_X + CH * H;
constexpr size_t B_y = B_a + CH * H;
constexpr size_t B_qkv = B_y + CH * H;
constexpr size_t B_z = B_qkv + CH * CONV;
constexpr size_t B_gb = B_z + CH * VD;
constexpr size_t B_ga = B_gb + CH * NV;
constexpr size_t B_conv = B_ga + CH * NV;
constexpr size_t B_o = B_conv + CH * CONV;
constexpr size_t B_qg = B_o + CH * VD;
constexpr size_t B_qn = B_qg + CH * NQ * 2 * HD;
constexpr size_t B_kx = B_qn + CH * NQ * HD;
constexpr size_t B_vx = B_kx + CH * NKV * HD;
constexpr size_t B_att = B_vx + CH * NKV * HD;
constexpr size_t B_g = B_att + CH * NQ * HD;
constexpr size_t B_u = B_g + CH * FFN;
constexpr size_t B_hh = B_u + CH * FFN;
constexpr size_t PART_ROWS = MAX((size_t)AB * NQ * NSPLIT_PRE, (size_t)KMAX * NQ * NSPLIT_DEC);
constexpr size_t B_part = B_hh + CH * FFN; // attention partials [rows][HD+2]
constexpr size_t B_xh = B_part + PART_ROWS * (HD + 2); // half copy of a matmul input
constexpr size_t B_wh = B_xh + CH * FFN / 2 + 16;     // f16 copy of one weight matrix
constexpr size_t B_lt = B_wh + FFN * H / 2 + 16; // LoRA t = A x, [CH][LORA_MAX_RT]
constexpr size_t B_total = B_lt + CH * LORA_MAX_RT;
// attn_fl's permuted half queries [NKV][T rounded up to 16][6][HD] live in B_xh
static_assert((size_t)NKV * (CH + 16) * (NQ / NKV) * HD <= CH * FFN);

static const char kernel_src[] = {
#embed "kernels/qmatvec.metal"
    , '\n',
#embed "kernels/decode.metal"
    , '\n',
#embed "kernels/qmm.metal"
    , '\n',
#embed "kernels/dense.metal"
    , 0};

static const char mpp_src[] = {
#embed "kernels/qmm_mpp.metal"
    , 0};

static Err gen_load(Gpu *g, Gen *r) {
    const char *src = qgen_src;
    TRY(gpu_pipeline(g, src, "qmatvec_gen", &r->mv));
    TRY(gpu_pipeline(g, src, "qmv_multi_l8_gen", &r->multi[0]));
    TRY(gpu_pipeline(g, src, "qmv_multi_l32_gen", &r->multi[1]));
    TRY(gpu_pipeline(g, src, "qmv_multi_l256_gen", &r->multi[2]));
    for (int t = 1; t <= SMALL; t++) {
        char nm[64];
        snprintf(nm, sizeof nm, "qmv_rows1_gen_%d", t);
        TRY(gpu_pipeline(g, src, nm, &r->rows1[t]));
        snprintf(nm, sizeof nm, "qmv_rows_gen_%d", t);
        TRY(gpu_pipeline(g, src, nm, &r->rows[t]));
    }
    r->rows1[0] = r->rows1[1];
    r->rows[0] = r->rows[1];
    TRY(gpu_pipeline(g, src, "qmm_gen", &r->qmm));
    if (gpu_pipeline(g, qgen_mpp_src, "dq_f16_gen", &r->dq)) r->dq = NULL;
    if (gpu_pipeline(g, qgen_mpp_src, "qmmh_gen", &r->qmmh)) r->qmmh = NULL;
    return OK;
}

static bool fast_q(const Linear *l) {
    return (l->bits == 4 && l->group_size == 32) || (l->bits == 8 && l->group_size == 64);
}
static bool is4_32(const Linear *l) { return l->bits == 4 && l->group_size == 32; }

static Err lin(Arena *a, const Pack *pack, const char *p, const char *name, Linear *out);
static Err vec(Arena *a, const Pack *pack, Params *P, const char *p, const char *name, Vec *out);
static char *cat(Arena *a, const char *x, const char *y) { return arena_printf(a, "%s%s", x, y); }
static void dequant_row(const Pack *pack, const Linear *l, size_t r, float *out);
static Err load_mtp(Arena *a, const Pack *pack, Params *P, Mtp *mt);

static const char *const pipe_kernels[] = {
#define MODEL_PIPE_KNAME(f, k) k,
    MODEL_PIPES(MODEL_PIPE_KNAME)
#undef MODEL_PIPE_KNAME
};
static const char *const pipe_names[] = {
#define MODEL_PIPE_FNAME(f, k) #f,
    MODEL_PIPES(MODEL_PIPE_FNAME)
#undef MODEL_PIPE_FNAME
};
#define NPIPES COUNT(pipe_names)
static_assert(sizeof(Pipes) == NPIPES * sizeof(sx_pipeline *));

Err model_load(Arena *arena, const Pack *pack, Gpu *g, Model **out) {
    Pipes pipes;
    sx_pipeline **pp = (sx_pipeline **)&pipes;
    for (size_t i = 0; i < NPIPES; i++) TRY(gpu_pipeline(g, kernel_src, pipe_kernels[i], &pp[i]));
    Mpp *mpp = NULL;
    {
        Mpp mp = {0};
        if (!gpu_pipeline(g, mpp_src, "to_half", &mp.to_half) && !gpu_pipeline(g, mpp_src, "mmh_dense", &mp.mmh)) {
            if (gpu_pipeline(g, mpp_src, "qmmh_4_32", &mp.dense)) mp.dense = NULL;
            if (gpu_pipeline(g, mpp_src, "dq_f16_4_32", &mp.dq)) mp.dq = NULL;
            if (gpu_pipeline(g, mpp_src, "dq_bf16_f16", &mp.dq_bf16)) mp.dq_bf16 = NULL;
            mp.attn_qb = 8;
            if (getenv("SAETTA_ATTN27_SIMD") == NULL &&
                (gpu_pipeline(g, attn27_src, "attn_fl_q6h", &mp.attn_q6h) || gpu_pipeline(g, attn27_src, "attn_fl_6_q8", &mp.attn_fl))) {
                dbg("model: accelerator prompt attention unavailable, attn_full_6h serves every chunk\n");
                mp.attn_q6h = mp.attn_fl = NULL;
            }
            mpp = ARENA_NEW(arena, Mpp);
            *mpp = mp;
        }
    }
    if (mpp == NULL) dbg("model: accelerator matmul kernels unavailable, prefill stays f32\n");
    Model *m = ARENA_NEW(arena, Model);
    *m = (Model){0};
    m->arch = &dense_arch;
    *out = m;
    for (int t = 1; t <= SMALL; t++) {
        char nm[64];
        snprintf(nm, sizeof nm, "qmv_rows_4_32_%d", t);
        TRY(gpu_pipeline(g, kernel_src, nm, &m->rows_p[0][t]));
        snprintf(nm, sizeof nm, "qmv_rows_8_64_%d", t);
        TRY(gpu_pipeline(g, kernel_src, nm, &m->rows_p[1][t]));
        snprintf(nm, sizeof nm, "qmv_rows1_4_32_%d", t);
        TRY(gpu_pipeline(g, kernel_src, nm, &m->rows1_p[0][t]));
        snprintf(nm, sizeof nm, "qmv_rows1_8_64_%d", t);
        TRY(gpu_pipeline(g, kernel_src, nm, &m->rows1_p[1][t]));
    }
    for (int i = 0; i < 2; i++) {
        m->rows_p[i][0] = m->rows_p[i][1];
        m->rows1_p[i][0] = m->rows1_p[i][1];
    }
    m->pack = pack;
    m->g = g;
    m->p = pipes;
    m->mpp = mpp;
    m->fast_prefill = mpp != NULL && getenv("SAETTA_PREFILL_F32") == NULL;
    m->lora_on = -1;
    // the packs are 4-bit and 8-bit with group 64: the run-time (bits, group) kernels serve
    // every matrix outside the 4/32 and 8/64 templated ones
    Gen *gn = ARENA_NEW(arena, Gen);
    TRY(gen_load(g, gn));
    m->gen = gn;
    m->layers = ARENA_ARRAY(arena, Layer, NLAYERS);
    TRY(gpu_new_buffer(g, (size_t)64 << 20, &m->params.buf));
    TRY(gpu_new_buffer(g, S_total * 4, &m->scratch));
    TRY(gpu_new_buffer(g, B_total * 4, &m->bscratch));
    TRY(gpu_new_buffer(g, VOCAB * 4, &m->logits));
    TRY(gpu_new_buffer(g, KMAX * VOCAB * 4, &m->vlogits));
    TRY(gpu_new_buffer(g, CH * H * 4, &m->stage));
    TRY(pack_linear(pack, S(WEIGHTS_PREFIX "embed_tokens"), &m->embed));
    TRY(pack_linear(pack, S("language_model.lm_head"), &m->lm_head));
    TRY(vec(arena, pack, &m->params, WEIGHTS_PREFIX, "norm.weight", &m->final_norm));
    size_t gi = 0, ai = 0;
    for (size_t i = 0; i < NLAYERS; i++) {
        Layer *L = &m->layers[i];
        const char *p = arena_printf(arena, "%slayers.%zu.", WEIGHTS_PREFIX, i);
        Params *P = &m->params;
        TRY(vec(arena, pack, P, p, "input_layernorm.weight", &L->in_norm));
        TRY(vec(arena, pack, P, p, "post_attention_layernorm.weight", &L->post_norm));
        L->gdn = NULL;
        L->attn = NULL;
        switch (config_layer_kinds[i]) {
        case LAYER_KIND_GDN: {
            Gdn *G = ARENA_NEW(arena, Gdn);
            G->slot = gi++;
            TRY(lin(arena, pack, p, "linear_attn.in_proj_qkv", &G->qkv));
            TRY(lin(arena, pack, p, "linear_attn.in_proj_z", &G->z));
            TRY(lin(arena, pack, p, "linear_attn.in_proj_a", &G->a));
            TRY(lin(arena, pack, p, "linear_attn.in_proj_b", &G->b));
            TRY(lin(arena, pack, p, "linear_attn.out_proj", &G->out));
            TRY(vec(arena, pack, P, p, "linear_attn.conv1d.weight", &G->conv));
            TRY(vec(arena, pack, P, p, "linear_attn.A_log", &G->a_log));
            TRY(vec(arena, pack, P, p, "linear_attn.dt_bias", &G->dt_bias));
            TRY(vec(arena, pack, P, p, "linear_attn.norm.weight", &G->norm));
            L->gdn = G;
            break;
        }
        case LAYER_KIND_ATTENTION: {
            Attn *A = ARENA_NEW(arena, Attn);
            A->slot = ai++;
            TRY(lin(arena, pack, p, "self_attn.q_proj", &A->q));
            TRY(lin(arena, pack, p, "self_attn.k_proj", &A->k));
            TRY(lin(arena, pack, p, "self_attn.v_proj", &A->v));
            TRY(lin(arena, pack, p, "self_attn.o_proj", &A->o));
            TRY(vec(arena, pack, P, p, "self_attn.q_norm.weight", &A->q_norm));
            TRY(vec(arena, pack, P, p, "self_attn.k_norm.weight", &A->k_norm));
            L->attn = A;
            break;
        }
        }
        TRY(lin(arena, pack, p, "mlp.gate_proj", &L->mlp.gate));
        TRY(lin(arena, pack, p, "mlp.up_proj", &L->mlp.up));
        TRY(lin(arena, pack, p, "mlp.down_proj", &L->mlp.down));
    }
    // the MTP draft head, when the pack has it (SAETTA_MTP=0 leaves it off)
    const char *mv = getenv("SAETTA_MTP");
    if ((mv ? mv[0] != '0' : true) && pack_get(pack, S("mtp.fc.weight")) != NULL) {
        Mtp *mt = ARENA_NEW(arena, Mtp);
        TRY(load_mtp(arena, pack, &m->params, mt));
        TRY(gpu_new_buffer(g, (CH + 1) * H * 4, &m->mtp_hid));
        TRY(gpu_new_buffer(g, VOCAB * 4, &m->mtp_logits));
        m->mtp = mt;
        n_kv = n_attn + 1;
        dbg("model: MTP head loaded (fc %u-bit group %u)\n", mt->fc.bits, mt->fc.group_size);
    } else
        n_kv = n_attn;
    // the shapes are compile-time constants here: a pack that differs is refused, not run
    FILE *quiet = fopen("/dev/null", "w");
    size_t bad = model_check_pack(arena, pack, quiet ? quiet : stderr);
    if (quiet) fclose(quiet);
    if (bad) {
        (void)model_check_pack(arena, pack, stderr);
        return "PackShapeMismatch";
    }
    return OK;
}

static inline sx_arg barg(sx_buffer *b, size_t off_floats) { return (sx_arg){b, off_floats * 4, NULL, 0}; }
static inline sx_arg sarg(Model *m, size_t off) { return barg(m->scratch, off); }
static inline sx_arg offa(sx_arg a, size_t floats) {
    a.offset += floats * 4;
    return a;
}
#define BYTES(x) gpu_bytes_arg(&(x), sizeof(x))
#define TARG(t, sh) gpu_tensor_arg(m->g, &(t), (sh))

// ---- profiling (SAETTA_PROFILE=1: every dispatch in its own command buffer) -----------------

static struct {
    bool on;
    double secs[NPIPES];
    uint64_t count[NPIPES];
    double other_secs;
    uint64_t other_count;
} prof;

void model_prof_init(void) { prof.on = getenv("SAETTA_PROFILE") != NULL; }

void model_prof_reset(void) {
    memset(prof.secs, 0, sizeof prof.secs);
    memset(prof.count, 0, sizeof prof.count);
    prof.other_secs = 0;
    prof.other_count = 0;
}

static void prof_add(Model *m, sx_pipeline *p, double s) {
    sx_pipeline **pp = (sx_pipeline **)&m->p;
    for (size_t i = 0; i < NPIPES; i++)
        if (pp[i] == p) {
            prof.secs[i] += s;
            prof.count[i] += 1;
            return;
        }
    prof.other_secs += s;
    prof.other_count += 1;
}

void model_prof_report(size_t tokens) {
    if (!prof.on) return;
    double total = prof.other_secs;
    for (size_t i = 0; i < NPIPES; i++) total += prof.secs[i];
    char a[64];
    fmt_fixed(a, sizeof a, total * 1e3 / (double)tokens, 2);
    dbg("profile: GPU time per token by kernel (%zu tokens), total %s ms/token\n", tokens, a);
    for (size_t i = 0; i < NPIPES; i++)
        if (prof.count[i] > 0) {
            fmt_fixed(a, sizeof a, prof.secs[i] * 1e3 / (double)tokens, 3);
            dbg("  %14s %8s ms/token  %5llu dispatches/token\n", pipe_names[i], a,
                (unsigned long long)(prof.count[i] / tokens));
        }
    if (prof.other_count > 0) {
        fmt_fixed(a, sizeof a, prof.other_secs * 1e3 / (double)tokens, 3);
        dbg("  other (matmuls) %7s ms/token  %5llu dispatches/token\n", a, (unsigned long long)(prof.other_count / tokens));
    }
}

// ---- encoder --------------------------------------------------------------------------------

typedef struct {
    uint32_t in, nmat, act;
    float act_scale;
    uint32_t rows[4], tg0[5], shift[12];
} MvP;
static_assert(sizeof(MvP) == offsetof(MvGP, bits));

typedef struct {
    uint32_t n, xs, os, ws, plus1, wmod;
    float eps;
} NormP;
typedef struct {
    uint32_t nk, nv, dk, dv, key_dim, conv_dim, T;
    float eps;
    uint32_t save, slot;
} GdnP;
typedef struct {
    uint32_t pos0, budget_blocks, ratio, head_dim, kv_stride, q_per_kv, gate_stride, nq, nsplit, maxsel;
    float scale;
} AttnSP;

static Enc enc_begin(Model *m) { return (Enc){.m = m, .cmd = sx_cmd_begin(m->g->dev)}; }

static Err enc_finish(Enc *e) {
    char err[512];
    if (sx_cmd_finish(e->cmd, err, sizeof err) != 0) {
        dbg("command buffer: %s\n", err);
        return "GpuFailed";
    }
    return OK;
}

static bool is_accel(const Model *m, const sx_pipeline *p) {
    const Mpp *mp = m->mpp;
    if (!mp) return false;
    return p == mp->dense || p == mp->to_half || p == mp->dq || p == mp->mmh || p == mp->dq_bf16 || p == m->gen->dq ||
           p == m->gen->qmmh;
}

static void prof_cut(Enc *e, sx_pipeline *p) {
    double secs = 0;
    char err[256];
    (void)sx_cmd_finish_timed(e->cmd, &secs, err, sizeof err);
    e->cmd = sx_cmd_begin(e->m->g->dev);
    prof_add(e->m, p, secs);
}

static void enc_d(Enc *e, sx_pipeline *p, const sx_arg *args, size_t nargs, size_t gx, size_t gy, size_t threads) {
    if (!is_accel(e->m, p)) e->xh_set = false;
    sx_cmd_dispatch(e->cmd, p, args, nargs, gx, gy, threads);
    if (prof.on) prof_cut(e, p);
}
static void enc_d3(Enc *e, sx_pipeline *p, const sx_arg *args, size_t nargs, size_t gx, size_t gy, size_t gz,
                   size_t threads) {
    e->xh_set = false;
    sx_cmd_dispatch3(e->cmd, p, args, nargs, gx, gy, gz, threads);
    if (prof.on) prof_cut(e, p);
}
static void enc_d1(Enc *e, sx_pipeline *p, const sx_arg *args, size_t nargs, size_t n) {
    enc_d(e, p, args, nargs, (n + 255) / 256, 1, 256);
}
#define D(e, p, a, gx, gy, th) enc_d((e), (p), (a), COUNT(a), (gx), (gy), (th))
#define D1(e, p, a, n) enc_d1((e), (p), (a), COUNT(a), (n))

static void enc_mv_bf16(Enc *e, const Linear *l, sx_arg x, sx_arg y, size_t T) {
    Model *m = e->m;
    uint32_t sh = 0;
    uint32_t dd[3] = {(uint32_t)l->in_features, (uint32_t)l->out_features, (uint32_t)T};
    sx_arg a[] = {TARG(l->weight, &sh), x, y, gpu_bytes_arg(dd, 12), BYTES(sh)};
    D(e, m->p.mv_bf16, a, dd[1], T, 32);
}

static void enc_mv(Enc *e, const Linear *l, sx_arg x, sx_arg y) {
    Model *m = e->m;
    uint32_t in = (uint32_t)l->in_features;
    uint32_t out = (uint32_t)l->out_features;
    if (l->bits == 16) {
        enc_mv_bf16(e, l, x, y, 1);
        return;
    }
    if (!fast_q(l)) {
        uint32_t q[2] = {l->bits, l->group_size};
        uint32_t shg[3];
        sx_arg ga[] = {TARG(l->weight, &shg[0]), TARG(l->scales, &shg[1]), TARG(l->biases, &shg[2]), x, y, BYTES(in), BYTES(out),
                       gpu_bytes_arg(shg, 12), BYTES(q)};
        D(e, m->gen->mv, ga, out, 1, 32);
        return;
    }
    uint32_t groups = in / 32;
    uint32_t lanes = l->bits != 4 ? 32 : (out <= 512 && groups >= 64) ? 256 : groups <= 16 ? 8 : 32;
    sx_pipeline *p = l->bits == 8 ? m->p.q8 : lanes == 256 ? m->p.qmv4_l256 : lanes == 8 ? m->p.qmv4_l8 : m->p.q4;
    uint32_t sh[3];
    sx_arg args[] = {TARG(l->weight, &sh[0]), TARG(l->scales, &sh[1]), TARG(l->biases, &sh[2]), x, y, BYTES(in), BYTES(out),
                     gpu_bytes_arg(sh, 12)};
    if (lanes == 32)
        D(e, p, args, out, 1, 32);
    else
        D(e, p, args, (out + 256 / lanes - 1) / (256 / lanes), 1, 256);
}

// Up to 4 linears on the same input in one dispatch (lanes per row 8, 32 or 256), over T input
// rows (x row stride = in, output k row stride = its rows).
static void enc_mv_multi_t(Enc *e, const Linear *const *lins, size_t nl, sx_arg x, const sx_arg *ys, size_t ny,
                           uint32_t lanes, size_t T) {
    Model *m = e->m;
    bool gen = false;
    for (size_t i = 0; i < nl; i++) gen = gen || !is4_32(lins[i]);
    MvGP p = {.in = (uint32_t)lins[0]->in_features, .nmat = (uint32_t)nl, .act = 0, .act_scale = 1.0f,
              .bits = {4, 4, 4, 4}, .group = {32, 32, 32, 32}};
    sx_arg args[18];
    uint32_t rpt = 256 / lanes;
    for (size_t k = 0; k < 4; k++) {
        const Linear *l = lins[MIN(k, nl - 1)];
        p.bits[k] = l->bits;
        p.group[k] = l->group_size;
        args[3 * k] = TARG(l->weight, &p.shift[3 * k]);
        args[3 * k + 1] = TARG(l->scales, &p.shift[3 * k + 1]);
        args[3 * k + 2] = TARG(l->biases, &p.shift[3 * k + 2]);
        args[13 + k] = ys[MIN(k, ny - 1)];
        if (k < nl) {
            p.rows[k] = (uint32_t)l->out_features;
            p.tg0[k + 1] = p.tg0[k] + (p.rows[k] + rpt - 1) / rpt;
        }
    }
    args[12] = x;
    args[17] = gen ? BYTES(p) : gpu_bytes_arg(&p, sizeof(MvP));
    size_t li = lanes == 8 ? 0 : lanes == 32 ? 1 : 2;
    sx_pipeline *pl = gen ? m->gen->multi[li]
                          : (lanes == 8 ? m->p.qmv_multi_l8 : lanes == 32 ? m->p.qmv_multi_l32 : m->p.qmv_multi_l256);
    D(e, pl, args, p.tg0[nl], T, 256);
}
#define LINS(...) (const Linear *const[]){__VA_ARGS__}, COUNT(((const Linear *const[]){__VA_ARGS__}))
#define ARGS(...) (const sx_arg[]){__VA_ARGS__}, COUNT(((const sx_arg[]){__VA_ARGS__}))

// Shared input (one bit width and group for all): one multi dispatch; else one matvec each.
static bool same_quant_all(const Linear *const *lins, size_t n) {
    for (size_t i = 1; i < n; i++)
        if (lins[i]->bits != lins[0]->bits || lins[i]->group_size != lins[0]->group_size) return false;
    return lins[0]->bits != 16;
}

static void enc_norm(Enc *e, sx_arg x, Vec w, sx_arg out, size_t groups, size_t n, size_t xs, size_t os, size_t ws,
                     bool plus1) {
    NormP p = {(uint32_t)n, (uint32_t)xs, (uint32_t)os, (uint32_t)ws, plus1, 0, EPS};
    sx_arg a[] = {x, barg(e->m->params.buf, w.off), out, BYTES(p)};
    D(e, e->m->p.rmsnorm, a, groups, 1, 256);
}
// x += y over T rows of H, then out = rms(x) * w (the next block's input norm).
static void enc_add_norm(Enc *e, sx_arg x, sx_arg y, Vec w, sx_arg out, size_t T) {
    uint32_t n = (uint32_t)H;
    float eps = EPS;
    sx_arg a[] = {x, y, barg(e->m->params.buf, w.off), out, BYTES(n), BYTES(eps)};
    D(e, e->m->p.add_rmsnorm, a, T, 1, 1024);
}
static void enc_copy(Enc *e, sx_arg src, sx_arg dst, size_t n) {
    uint32_t nn = (uint32_t)n;
    sx_arg a[] = {src, dst, BYTES(nn)};
    D1(e, e->m->p.copy_f32, a, n);
}
static void enc_copy_rows(Enc *e, sx_arg src, sx_arg dst, size_t n, size_t rows, size_t ss, size_t ds) {
    uint32_t p[4] = {(uint32_t)n, (uint32_t)rows, (uint32_t)ss, (uint32_t)ds};
    sx_arg a[] = {src, dst, gpu_bytes_arg(p, 16)};
    D1(e, e->m->p.copy_rows, a, n * rows);
}
static void enc_rope_t(Enc *e, sx_arg x, size_t heads, size_t stride, size_t T, sx_arg cs) {
    uint32_t p[3] = {(uint32_t)stride, (uint32_t)(ROT / 2), (uint32_t)heads};
    sx_arg a[] = {x, gpu_bytes_arg(p, 12), cs};
    D(e, e->m->p.rope, a, heads * T, 1, 32);
}

static sx_arg kv_arg(Session *s, size_t elems) { return (sx_arg){s->kv, elems * session_kv_elem(s), NULL, 0}; }

static void enc_kv_store(Enc *e, Session *s, sx_arg src, size_t dst_elems, size_t n, size_t rows) {
    if (s->kv_half) {
        uint32_t p[4] = {(uint32_t)n, (uint32_t)rows, (uint32_t)n, (uint32_t)n};
        sx_arg a[] = {src, kv_arg(s, dst_elems), gpu_bytes_arg(p, 16)};
        D1(e, e->m->p.copy_rows_h, a, n * rows);
    } else
        enc_copy_rows(e, src, kv_arg(s, dst_elems), n, rows, n, n);
}

static void enc_to_half(Enc *e, sx_arg x, size_t n) {
    if (e->xh_set && e->xh_buf == x.buffer && e->xh_off == x.offset && e->xh_n == n) return;
    uint32_t nn = (uint32_t)n;
    sx_arg a[] = {x, barg(e->m->bscratch, B_xh), BYTES(nn)};
    D1(e, e->m->mpp->to_half, a, n);
    e->xh_set = true;
    e->xh_buf = x.buffer;
    e->xh_off = x.offset;
    e->xh_n = n;
}
static void enc_out_writes_range(Enc *e, sx_arg y, size_t n) {
    if (e->xh_set && e->xh_buf == y.buffer && y.offset < e->xh_off + e->xh_n * 4 && e->xh_off < y.offset + n * 4)
        e->xh_set = false;
}

// Y[T][out] = X[T][in] W^T
static void enc_mm(Enc *e, const Linear *l, sx_arg x, sx_arg y, size_t T) {
    Model *m = e->m;
    uint32_t dims[3] = {(uint32_t)l->in_features, (uint32_t)l->out_features, (uint32_t)T};
    if (l->bits == 16) {
        enc_mv_bf16(e, l, x, y, T);
        return;
    }
    if (T <= SMALL) {
        // verification-sized chunk: weights read once, applied to each row
        uint32_t sh[3];
        sx_arg args[8] = {TARG(l->weight, &sh[0]), TARG(l->scales, &sh[1]), TARG(l->biases, &sh[2]), x, y,
                          gpu_bytes_arg(dims, 12), gpu_bytes_arg(sh, 12)};
        if (!fast_q(l)) {
            uint32_t q[2] = {l->bits, l->group_size};
            args[7] = BYTES(q);
            if (T >= 3 && dims[1] >= 2048)
                enc_d(e, m->gen->rows[T], args, 8, (dims[1] + 7) / 8, 1, 64);
            else
                enc_d(e, m->gen->rows1[T], args, 8, dims[1], 1, 32);
            return;
        }
        size_t bi = l->bits == 8 ? 1 : 0;
        if (T >= 3 && dims[1] >= 2048)
            enc_d(e, m->rows_p[bi][T], args, 7, (dims[1] + 7) / 8, 1, 64);
        else
            enc_d(e, m->rows1_p[bi][T], args, 7, dims[1], 1, 32);
        return;
    }
    if (m->fast_prefill) {
        enc_to_half(e, x, T * (size_t)l->in_features);
        uint32_t sh[3];
        sx_arg args[8] = {TARG(l->weight, &sh[0]), TARG(l->scales, &sh[1]), TARG(l->biases, &sh[2]),
                          barg(m->bscratch, B_xh), y, gpu_bytes_arg(dims, 12), gpu_bytes_arg(sh, 12)};
        const Mpp *mp = m->mpp;
        uint32_t q[2] = {l->bits, l->group_size};
        bool gen = !is4_32(l);
        sx_pipeline *dq = gen ? m->gen->dq : mp->dq;
        if (mp->mmh && dq && (size_t)dims[0] * dims[1] <= (size_t)FFN * H) {
            // unpack W to f16 in B.wh, then the f16 GEMM
            uint32_t d2[2] = {dims[0], dims[1]};
            if (gen) {
                sx_arg a[] = {args[0], args[1], args[2], barg(m->bscratch, B_wh), gpu_bytes_arg(d2, 8), args[6], BYTES(q)};
                D1(e, dq, a, (size_t)dims[1] * dims[0] / 8);
            } else {
                sx_arg a[] = {args[0], args[1], args[2], barg(m->bscratch, B_wh), gpu_bytes_arg(d2, 8), args[6]};
                D1(e, dq, a, (size_t)dims[1] * dims[0] / 8);
            }
            sx_arg b[] = {barg(m->bscratch, B_xh), barg(m->bscratch, B_wh), y, gpu_bytes_arg(dims, 12)};
            D(e, mp->mmh, b, (dims[1] + 31) / 32, (T + 63) / 64, 32);
            enc_out_writes_range(e, y, T * dims[1]);
            return;
        }
        if (gen && m->gen->qmmh) {
            args[7] = BYTES(q);
            enc_d(e, m->gen->qmmh, args, 8, (dims[1] + 63) / 64, (T + 63) / 64, 128);
            enc_out_writes_range(e, y, T * dims[1]);
            return;
        }
        if (!gen && mp->dense) {
            enc_d(e, mp->dense, args, 7, (dims[1] + 63) / 64, (T + 63) / 64, 128);
            enc_out_writes_range(e, y, T * dims[1]);
            return;
        }
    }
    uint32_t sh[3];
    if (!fast_q(l)) {
        uint32_t q[2] = {l->bits, l->group_size};
        sx_arg ga[] = {TARG(l->weight, &sh[0]), TARG(l->scales, &sh[1]), TARG(l->biases, &sh[2]), x, y,
                       gpu_bytes_arg(dims, 12), gpu_bytes_arg(sh, 12), BYTES(q)};
        D(e, m->gen->qmm, ga, (dims[1] + 31) / 32, (T + 31) / 32, 128);
        return;
    }
    sx_pipeline *p = l->bits == 8 ? m->p.mm8 : m->p.mm4;
    sx_arg args[] = {TARG(l->weight, &sh[0]), TARG(l->scales, &sh[1]), TARG(l->biases, &sh[2]), x, y,
                     gpu_bytes_arg(dims, 12), gpu_bytes_arg(sh, 12)};
    D(e, p, args, (dims[1] + 31) / 32, (T + 31) / 32, 128);
}

// ---- LoRA -----------------------------------------------------------------------------------

// The selected adapter's group k of layer li, or NULL.
static const LoraGroup *lora_group(const Model *m, size_t li, int k) {
    if (m->lora_on < 0 || li >= NLAYERS) return NULL; // li NLAYERS: the MTP layer
    const LoraGroup *G = &m->lora[m->lora_on].g[li][k];
    return G->on ? G : NULL;
}

// y_i += scale * B_i (A x) for the linears of group G over T rows: x [T][in], y_i [T][out_i].
static void enc_lora(Enc *e, const LoraGroup *G, sx_arg x, size_t T, const sx_arg *ys, bool dec) {
    if (!G) return;
    Model *m = e->m;
    const Adapter *ad = &m->lora[m->lora_on];
    sx_arg t = dec ? sarg(m, S_lt) : barg(m->bscratch, B_lt);
    lora_encode_down(&m->lp, e->cmd, (sx_arg){ad->buf, G->a, NULL, 0}, x, t, (uint32_t)T, G->in, G->in, G->rt);
    LoraUp ups[3];
    for (uint32_t i = 0; i < G->n; i++)
        ups[i] = (LoraUp){.B = {ad->buf, G->up[i].b, NULL, 0}, .y = ys[G->up[i].slot], .out = G->up[i].out, .r0 = G->up[i].r0,
                          .ldy = G->up[i].out};
    lora_encode_up(&m->lp, e->cmd, t, (uint32_t)T, G->rt, ad->r, ad->scale, ups, G->n);
    e->xh_set = false; // y may be the source of the cached half copy
}

// ---- blocks ---------------------------------------------------------------------------------

static sx_arg src_gdn(const Session *s, size_t slot) {
    size_t n = NV * DK * DV;
    return s->roll != SESSION_NO_ROLL ? barg(s->sp_gdn, s->roll * n_gdn * n + slot * n) : barg(s->gdn_state, slot * n);
}
static sx_arg src_conv(const Session *s, size_t slot) {
    size_t n = (KCONV - 1) * CONV;
    return s->roll != SESSION_NO_ROLL ? barg(s->sp_conv, s->roll * n_gdn * n + slot * n) : barg(s->gdn_conv, slot * n);
}

// Gated DeltaNet over T tokens (M27:500-656): x at xa ([T][H], normed), output at y.
static void enc_gdn_t(Enc *e, const Gdn *G, Session *s, size_t T, sx_arg xa, sx_arg y, size_t base, bool dec) {
    Model *m = e->m;
    sx_buffer *b = dec ? m->scratch : m->bscratch;
    size_t o_qkv = dec ? S_qkv : B_qkv, o_z = dec ? S_z : B_z, o_gb = dec ? S_gb : B_gb, o_ga = dec ? S_ga : B_ga;
    size_t o_conv = dec ? S_conv : B_conv, o_o = dec ? S_o : B_o;
    (void)base;
    if (T == 1 && same_quant_all(LINS(&G->qkv, &G->z, &G->b, &G->a))) {
        enc_mv_multi_t(e, LINS(&G->qkv, &G->z, &G->b, &G->a), xa, ARGS(barg(b, o_qkv), barg(b, o_z), barg(b, o_gb), barg(b, o_ga)),
                       32, 1);
    } else {
        enc_mm(e, &G->qkv, xa, barg(b, o_qkv), T);
        enc_mm(e, &G->z, xa, barg(b, o_z), T);
        enc_mm(e, &G->b, xa, barg(b, o_gb), T);
        enc_mm(e, &G->a, xa, barg(b, o_ga), T);
    }
    uint32_t conv_slot = (uint32_t)(n_gdn * (KCONV - 1) * CONV);
    uint32_t n[4] = {(uint32_t)CONV, (uint32_t)T, s->spec, conv_slot};
    sx_arg a[] = {barg(b, o_qkv), barg(s->gdn_conv, G->slot * (KCONV - 1) * CONV), barg(m->params.buf, G->conv.off),
                  barg(b, o_conv), gpu_bytes_arg(n, 16), barg(s->sp_conv, G->slot * (KCONV - 1) * CONV), src_conv(s, G->slot)};
    D1(e, m->p.gdn_conv, a, CONV);
    uint32_t qk[4] = {(uint32_t)NK, (uint32_t)DK, (uint32_t)KD, (uint32_t)CONV};
    sx_arg c[] = {barg(b, o_conv), gpu_bytes_arg(qk, 16)};
    D(e, m->p.gdn_qknorm, c, 2 * NK * T, 1, DK);
    GdnP gp = {(uint32_t)NK, (uint32_t)NV, (uint32_t)DK, (uint32_t)DV, (uint32_t)KD, (uint32_t)CONV, (uint32_t)T, EPS,
               s->spec, (uint32_t)(n_gdn * NV * DK * DV)};
    sx_arg d[] = {barg(b, o_conv),
                  barg(b, o_ga),
                  barg(b, o_gb),
                  barg(m->params.buf, G->a_log.off),
                  barg(m->params.buf, G->dt_bias.off),
                  barg(s->gdn_state, G->slot * NV * DK * DV),
                  barg(b, o_o),
                  BYTES(gp),
                  barg(s->sp_gdn, G->slot * NV * DK * DV),
                  src_gdn(s, G->slot)};
    D(e, m->p.gdn_step, d, DV / 16, NV, 64);
    sx_arg f[] = {barg(b, o_o), barg(b, o_z), barg(m->params.buf, G->norm.off), BYTES(gp)};
    D(e, m->p.gdn_gnorm, f, T * NV, 1, DV);
    if (T == 1)
        enc_mv(e, &G->out, barg(b, o_o), y);
    else
        enc_mm(e, &G->out, barg(b, o_o), y, T);
}

// Dense causal attention of T queries at pos0.. over the KV cache (all visible tokens), gated
// by sigmoid(gate) (M27:742-815): q [T][NQ][HD] normed and roped, gate at gate (row stride
// NQ*2*HD, head stride 2*HD), output att [T][NQ][HD].
static void enc_attn_full(Enc *e, Session *s, sx_arg q, sx_arg kc, sx_arg vc, sx_arg gate, sx_arg att, size_t pos0,
                          size_t T) {
    Model *m = e->m;
    const Mpp *mp = m->mpp;
    if (T > KMAX && s->kv_half && m->fast_prefill && mp->attn_fl) {
        // prompt chunks: every query block of a kv head shares its K / V tiles, on the
        // accelerators (src/kernels/dense_attn.metal), the gate fused; queries to half in B_xh
        Attn27P p = {(uint32_t)pos0, (uint32_t)T, attn27_tq(T, mp->attn_qb)};
        sx_arg qh = barg(m->bscratch, B_xh);
        sx_arg a[] = {q, qh, BYTES(p)};
        D1(e, mp->attn_q6h, a, (size_t)NKV * p.tq * (NQ / NKV) * HD / 4);
        sx_arg b[] = {qh, kc, vc, gate, att, BYTES(p)};
        D(e, mp->attn_fl, b, NKV, p.tq / mp->attn_qb, mp->attn_qb * 32);
        return;
    }
    size_t nsplit = T <= KMAX ? NSPLIT_DEC : NSPLIT_PRE;
    size_t slice = T <= KMAX ? T : AB;
    AttnSP p = {(uint32_t)pos0, 0, 1, (uint32_t)HD, (uint32_t)(NKV * HD), (uint32_t)(NQ / NKV), (uint32_t)(2 * HD),
                (uint32_t)NQ, (uint32_t)nsplit, 0, 1.0f / sqrtf((float)HD)};
    sx_arg part = barg(m->bscratch, B_part);
    for (size_t q0 = 0; q0 < T; q0 += slice) {
        size_t tb = MIN(slice, T - q0);
        AttnSP pq = p;
        pq.pos0 = (uint32_t)(pos0 + q0);
        sx_arg a[] = {offa(q, q0 * NQ * HD), kc, vc, part, BYTES(pq)};
        enc_d3(e, s->kv_half ? m->p.attn_full_h : m->p.attn_full, a, COUNT(a), NKV, nsplit, tb, 256);
        sx_arg b[] = {part, offa(gate, q0 * NQ * 2 * HD), offa(att, q0 * NQ * HD), BYTES(pq)};
        D(e, m->p.attn_merge, b, NQ, tb, 256);
    }
}

// Gated attention over T tokens at pos0 (input xa normed, output y).
static void enc_attn_t(Enc *e, const Attn *A, Session *s, size_t T, size_t pos0, sx_arg xa, sx_arg y, bool dec,
                       size_t li) {
    Model *m = e->m;
    sx_buffer *b = dec ? m->scratch : m->bscratch;
    size_t o_qg = dec ? S_qg : B_qg, o_qn = dec ? S_qn : B_qn, o_kx = dec ? S_kx : B_kx, o_vx = dec ? S_vx : B_vx;
    size_t o_att = dec ? S_att : B_att;
    size_t kv_row = NKV * HD;
    size_t kbase = A->slot * 2 * s->cap * kv_row;
    size_t vbase = kbase + s->cap * kv_row;
    if (T == 1 && same_quant_all(LINS(&A->q, &A->k, &A->v)))
        enc_mv_multi_t(e, LINS(&A->q, &A->k, &A->v), xa, ARGS(barg(b, o_qg), barg(b, o_kx), barg(b, o_vx)), 32, 1);
    else {
        enc_mm(e, &A->q, xa, barg(b, o_qg), T);
        enc_mm(e, &A->k, xa, barg(b, o_kx), T);
        enc_mm(e, &A->v, xa, barg(b, o_vx), T);
    }
    enc_lora(e, lora_group(m, li, LORA_QKV), xa, T, (const sx_arg[]){barg(b, o_qg), barg(b, o_kx), barg(b, o_vx)}, dec);
    // q / k norms (plain weights) and RoPE; per head the projection holds [query | gate]
    enc_norm(e, barg(b, o_qg), A->q_norm, barg(b, o_qn), T * NQ, HD, 2 * HD, HD, 0, false);
    enc_rope_t(e, barg(b, o_qn), NQ, HD, T, barg(s->rope, pos0 * ROT));
    enc_norm(e, barg(b, o_kx), A->k_norm, barg(b, o_kx), T * NKV, HD, HD, HD, 0, false);
    enc_rope_t(e, barg(b, o_kx), NKV, HD, T, barg(s->rope, pos0 * ROT));
    enc_kv_store(e, s, barg(b, o_kx), kbase + pos0 * kv_row, kv_row, T);
    enc_kv_store(e, s, barg(b, o_vx), vbase + pos0 * kv_row, kv_row, T);
    enc_attn_full(e, s, barg(b, o_qn), kv_arg(s, kbase), kv_arg(s, vbase), barg(b, o_qg + HD), barg(b, o_att), pos0, T);
    if (T == 1)
        enc_mv(e, &A->o, barg(b, o_att), y);
    else
        enc_mm(e, &A->o, barg(b, o_att), y, T);
    enc_lora(e, lora_group(m, li, LORA_O), barg(b, o_att), T, (const sx_arg[]){y}, dec);
}

// Dense SwiGLU MLP over T tokens (M27:818-831): down(silu(gate x) * up x).
static void enc_mlp_t(Enc *e, const Mlp *M, size_t T, sx_arg xa, sx_arg y, bool dec, size_t li) {
    Model *m = e->m;
    sx_buffer *b = dec ? m->scratch : m->bscratch;
    size_t o_g = dec ? S_g : B_g, o_u = dec ? S_u : B_u, o_hh = dec ? S_hh : B_hh;
    if (T == 1 && same_quant_all(LINS(&M->gate, &M->up)))
        enc_mv_multi_t(e, LINS(&M->gate, &M->up), xa, ARGS(barg(b, o_g), barg(b, o_u)), 32, 1);
    else {
        enc_mm(e, &M->gate, xa, barg(b, o_g), T);
        enc_mm(e, &M->up, xa, barg(b, o_u), T);
    }
    enc_lora(e, lora_group(m, li, LORA_GU), xa, T, (const sx_arg[]){barg(b, o_g), barg(b, o_u)}, dec);
    uint32_t n = (uint32_t)(T * FFN);
    sx_arg c[] = {barg(b, o_g), barg(b, o_u), barg(b, o_hh), BYTES(n)};
    D1(e, m->p.swiglu, c, T * FFN);
    if (T == 1)
        enc_mv(e, &M->down, barg(b, o_hh), y);
    else
        enc_mm(e, &M->down, barg(b, o_hh), y, T);
    enc_lora(e, lora_group(m, li, LORA_DOWN), barg(b, o_hh), T, (const sx_arg[]){y}, dec);
}

// The layers over T rows whose residual stream is at x ([T][H]) and whose first input norm is
// already in xa; leaves the final-normed rows in xa. dec: the decode scratch (T == 1).
static void enc_layers(Enc *e, Session *s, size_t T, size_t pos0, sx_arg x, sx_arg xa, sx_arg y, sx_buffer *trace,
                       bool dec) {
    Model *m = e->m;
    for (size_t li = 0; li < NLAYERS; li++) {
        const Layer *L = &m->layers[li];
        if (L->gdn)
            enc_gdn_t(e, L->gdn, s, T, xa, y, 0, dec);
        else
            enc_attn_t(e, L->attn, s, T, pos0, xa, y, dec, li);
        enc_add_norm(e, x, y, L->post_norm, xa, T);
        enc_mlp_t(e, &L->mlp, T, xa, y, dec, li);
        enc_add_norm(e, x, y, li + 1 < NLAYERS ? m->layers[li + 1].in_norm : m->final_norm, xa, T);
        if (trace) enc_copy(e, x, barg(trace, li * T * HCW), T * HCW);
    }
}

// MTP rows (V27:159-190): row i pairs the token embedding emb[i] ([T][H]) with the hidden
// hid[i] before it, at position p0 + i of that hidden (the head's KV and RoPE position, as
// vLLM's drafter: the hidden's position, so the head's KV has no gap at 0). As vLLM, hid is
// the trunk's output after the final norm, or the head's own output after mtp.norm when
// chaining (greedy acceptance at fixed depth 3 was higher on three prompts than with the
// streams before those norms). Leaves the head's outputs after mtp.norm in a ([T][H]) and,
// with logits, the last row's logits in mtp_logits. dec: the decode scratch (T == 1), else
// the chunk scratch.
static sx_arg mtp_x(Model *m, bool dec) { return dec ? sarg(m, S_X) : barg(m->bscratch, B_X); }

static void enc_mtp_rows(Enc *e, Session *s, sx_arg emb, sx_arg hid, size_t T, size_t p0, bool dec, bool logits) {
    Model *m = e->m;
    const Mtp *mt = m->mtp;
    sx_buffer *b = dec ? m->scratch : m->bscratch;
    // [rms_e(emb) | rms_h(hid)] per row, in the GDN qkv scratch (CONV == 2H floats a row)
    static_assert(CONV == 2 * H);
    sx_arg cat2 = barg(b, dec ? S_qkv : B_qkv);
    sx_arg x = mtp_x(m, dec), xa = barg(b, dec ? S_a : B_a), y = barg(b, dec ? S_y : B_y);
    enc_norm(e, emb, mt->n_e, cat2, T, H, H, 2 * H, 0, false);
    enc_norm(e, hid, mt->n_h, offa(cat2, H), T, H, H, 2 * H, 0, false);
    if (dec)
        enc_mv(e, &mt->fc, cat2, x);
    else
        enc_mm(e, &mt->fc, cat2, x, T);
    const Layer *L = &mt->layer;
    enc_norm(e, x, L->in_norm, xa, T, H, H, H, 0, false);
    enc_attn_t(e, L->attn, s, T, p0, xa, y, dec, NLAYERS);
    enc_add_norm(e, x, y, L->post_norm, xa, T);
    enc_mlp_t(e, &L->mlp, T, xa, y, dec, NLAYERS);
    enc_add_norm(e, x, y, mt->norm, xa, T);
    if (logits) enc_mv(e, &m->lm_head, offa(xa, (T - 1) * H), barg(m->mtp_logits, 0));
}

// ---- host math ------------------------------------------------------------------------------

// std.math.pow(f64) as the Zig engine computed it (see model.c), for the RoPE table.
static double zig_pow(double x, double y) {
    if (y == 0 || x == 1) return 1;
    if (isnan(x) || isnan(y)) return NAN;
    if (y == 1) return x;
    if (x == 0 || isinf(y) || isinf(x)) return pow(x, y);
    if (y == 0.5) return sqrt(x);
    if (y == -0.5) return 1 / sqrt(x);
    double yi;
    double yf = modf(fabs(y), &yi);
    if (yf != 0 && x < 0) return NAN;
    if (yi >= 9223372036854775808.0) return exp(y * log(fabs(x)));
    double a1 = 1.0;
    int ae = 0;
    if (yf != 0) {
        if (yf > 0.5) {
            yf -= 1;
            yi += 1;
        }
        a1 = exp(yf * log(x));
    }
    int xe;
    double x1 = frexp(x, &xe);
    for (int64_t i = (int64_t)yi; i != 0; i >>= 1) {
        if (xe < -(1 << 12) || (1 << 12) < xe) {
            ae += xe;
            break;
        }
        if ((i & 1) == 1) {
            a1 *= x1;
            ae += xe;
        }
        x1 *= x1;
        xe <<= 1;
        if (x1 < 0.5) {
            x1 += x1;
            xe -= 1;
        }
    }
    if (y < 0) {
        a1 = 1 / a1;
        ae = -ae;
    }
    return scalbn(a1, ae);
}

static void rope_row(float *cs, size_t pos) {
    for (size_t i = 0; i < ROT / 2; i++) {
        double inv = zig_pow((double)CONFIG_ROPE_THETA, -(double)(2 * i) / (double)ROT);
        double ang = (double)pos * inv;
        cs[2 * i] = (float)cos(ang);
        cs[2 * i + 1] = (float)sin(ang);
    }
}

// ---- Model passes ---------------------------------------------------------------------------

Err model_step(Model *m, Session *s, uint32_t token, float **logits) {
    if (s->pos >= s->cap) return "ContextFull";
    float *sc = gpu_floats(m->scratch);
    dequant_row(m->pack, &m->embed, token, sc + S_X);
    Enc e = enc_begin(m);
    enc_norm(&e, sarg(m, S_X), m->layers[0].in_norm, sarg(m, S_a), 1, H, H, H, 0, false);
    sx_buffer *trace = s->trace ? s->trace_buf : NULL;
    enc_layers(&e, s, 1, s->pos, sarg(m, S_X), sarg(m, S_a), sarg(m, S_y), trace, true);
    s->roll = SESSION_NO_ROLL; // the step's start state is encoded; it ends in the live state
    enc_mv(&e, &m->lm_head, sarg(m, S_a), barg(m->logits, 0));
    if (m->mtp) {
        enc_copy(&e, sarg(m, S_a), barg(m->mtp_hid, 0), H);
        s->mtp_n = 1;
    }
    TRY(enc_finish(&e));
    if (s->trace) memcpy(s->trace, gpu_floats(s->trace_buf), s->trace_len * sizeof(float));
    s->pos += 1;
    *logits = gpu_floats(m->logits);
    return OK;
}

// A chunk of T tokens at s->pos: all, the logits of every row into vlogits; else of the last row
// into logits.
static Err chunk_logits(Model *m, Session *s, const uint32_t *toks, size_t T, sx_buffer *trace, bool all,
                        float **logits) {
    if (s->pos + T > s->cap) return "ContextFull";
    float *st = gpu_floats(m->stage);
    for (size_t t = 0; t < T; t++) dequant_row(m->pack, &m->embed, toks[t], st + t * H);
    sx_buffer *b = m->bscratch;
    Enc e = enc_begin(m);
    enc_copy(&e, barg(m->stage, 0), barg(b, B_X), T * H);
    enc_norm(&e, barg(b, B_X), m->layers[0].in_norm, barg(b, B_a), T, H, H, H, 0, false);
    enc_layers(&e, s, T, s->pos, barg(b, B_X), barg(b, B_a), barg(b, B_y), trace, false);
    s->roll = SESSION_NO_ROLL;
    if (m->mtp) enc_copy(&e, barg(b, B_a), barg(m->mtp_hid, all ? 0 : H), T * H);
    if (all)
        enc_mm(&e, &m->lm_head, barg(b, B_a), barg(m->vlogits, 0), T);
    else
        enc_mv(&e, &m->lm_head, barg(b, B_a + (T - 1) * H), barg(m->logits, 0));
    if (m->mtp && !all) {
        // the head's history for the prompt chunk: rows q in [pos0, pos0 + T) pair the trunk
        // output at q - 1 (mtp_hid row q - pos0; row 0 the previous chunk's last) with toks
        // q - pos0 (still in stage); the first row is skipped at position 0 or when that
        // stream is unknown. Then the chunk's last stream moves to row 0.
        size_t skip0 = (s->pos == 0 || !s->mtp_ok) ? 1 : 0;
        if (T > skip0)
            enc_mtp_rows(&e, s, barg(m->stage, skip0 * H), barg(m->mtp_hid, skip0 * H), T - skip0, s->pos + skip0 - 1,
                         false, false);
        enc_copy(&e, barg(m->mtp_hid, T * H), barg(m->mtp_hid, 0), H);
        s->mtp_ok = true;
        s->mtp_n = 1;
    }
    TRY(enc_finish(&e));
    s->pos += T;
    *logits = gpu_floats(all ? m->vlogits : m->logits);
    return OK;
}

Err model_prefill(Model *m, Session *s, const uint32_t *tokens, size_t n, sx_buffer *trace, float **logits) {
    return model_prefill_with(m, s, tokens, n, trace, NULL, logits);
}

static size_t size_at(size_t i, size_t ae, size_t eq, size_t rp) { return ae == 0 ? eq : i < ae ? PT_EVERY : rp; }

Err model_prefill_with(Model *m, Session *s, const uint32_t *tokens, size_t n, sx_buffer *trace,
                       const PrefillHook *hook, float **logits) {
    // fewest chunks of at most CH tokens, of equal size; aligned on prefix points as model.c
    size_t nchunks = (n + CH - 1) / CH;
    size_t per = (n + nchunks - 1) / MAX(nchunks, (size_t)1);
    size_t pos0 = s->pos;
    bool aligned = hook && hook->aligned ? hook->aligned(hook->ctx) : false;
    size_t align_end = (aligned && pos0 % PT_EVERY == 0 && PT_EVERY <= CH)
                           ? MIN(n, (MAX(pos0, (size_t)(PT_MAX * PT_EVERY)) - pos0) / PT_EVERY * PT_EVERY)
                           : 0;
    size_t rest = n - align_end;
    size_t r_chunks = (rest + CH - 1) / CH;
    size_t r_per = (rest + r_chunks - 1) / MAX(r_chunks, (size_t)1);
    size_t i = 0;
    float *lg = NULL;
    while (i < n) {
        size_t T = MIN(size_at(i, align_end, per, r_per), n - i);
        TRY(chunk_logits(m, s, tokens + i, T, trace, false, &lg));
        i += T;
        if (hook && hook->chunk_done) TRY(hook->chunk_done(hook->ctx, s->pos));
    }
    *logits = lg;
    return OK;
}

Err model_verify_chunk(Model *m, Session *s, const uint32_t *toks, size_t n, float **logits) {
    if (!(n >= 1 && n <= KMAX)) die("model_verify_chunk: %zu tokens\n", n);
    float *lg;
    TRY(chunk_logits(m, s, toks, n, NULL, true, &lg));
    *logits = gpu_floats(m->vlogits);
    return OK;
}

static uint32_t greedy_pick(void *ctx, const float *row, bool has_draft, uint32_t draft) {
    (void)ctx, (void)has_draft, (void)draft;
    return model_argmax(row, VOCAB);
}

Err model_spec_greedy(Model *m, Session *s, uint32_t x, const uint32_t *drafts, size_t nd, SpecResult *out) {
    return model_spec_verify(m, s, x, drafts, nd, greedy_pick, NULL, out);
}

Err model_spec_verify(Model *m, Session *s, uint32_t x, const uint32_t *drafts, size_t nd, SpecPick pick, void *ctx,
                      SpecResult *out) {
    if (!(nd >= 1 && nd < KMAX)) die("model_spec_verify: %zu drafts\n", nd);
    uint32_t toks[KMAX];
    toks[0] = x;
    memcpy(toks + 1, drafts, nd * sizeof(uint32_t));
    size_t T = 1 + nd;
    size_t pos0 = s->pos;
    s->spec = true;
    s->chunk_pos0 = pos0;
    s->chunk_t = T;
    float *lg;
    Err err = model_verify_chunk(m, s, toks, T, &lg);
    if (err) {
        s->spec = false;
        return err;
    }
    size_t j = 0;
    uint32_t next = pick(ctx, lg, true, drafts[0]);
    while (j < nd && next == drafts[j]) {
        j += 1;
        next = pick(ctx, lg + j * VOCAB, j < nd, j < nd ? drafts[j] : 0);
    }
    if (m->mtp) {
        // the chunk left its trunk outputs in mtp_hid: rows 0..j pair with the accepted
        // drafts and then next
        memcpy(s->mtp_tok, drafts, j * sizeof(uint32_t));
        s->mtp_n = j + 1;
    }
    if (j + 1 < T) {
        // back to the state after token j of the chunk (its slot)
        s->roll = j;
        s->pos = pos0 + j + 1;
    }
    s->spec = false;
    *out = (SpecResult){j, next};
    return OK;
}

// As model.c: the pending verified rows first (streams at s->pos - n .. s->pos - 1, paired with
// mtp_tok and x), rewriting the head's KV there with the trunk's outputs; then each further
// draft feeds the head's own output and the previous draft.
Err model_draft_mtp(Model *m, Session *s, uint32_t x, size_t k, uint32_t *out, size_t *ndraft) {
    *ndraft = 0;
    if (!m->mtp || !s->mtp_ok || s->mtp_n == 0 || k == 0) return OK;
    size_t n = s->mtp_n;
    if (s->pos + k + 1 > s->cap || s->pos < n) return OK;
    bool dec = n == 1;
    float *emb = dec ? gpu_floats(m->scratch) + S_a : gpu_floats(m->bscratch) + B_a;
    for (size_t j = 0; j < n; j++) dequant_row(m->pack, &m->embed, j + 1 < n ? s->mtp_tok[j] : x, emb + j * H);
    Enc e = enc_begin(m);
    enc_mtp_rows(&e, s, dec ? sarg(m, S_a) : barg(m->bscratch, B_a), barg(m->mtp_hid, 0), n, s->pos - n, dec, true);
    TRY(enc_finish(&e));
    // the head's last output after mtp.norm: the next row's h (B.a is left alone by the
    // decode-scratch passes; S.a is read before it is rewritten)
    sx_arg h = dec ? sarg(m, S_a) : barg(m->bscratch, B_a + (n - 1) * H);
    out[0] = model_argmax(gpu_floats(m->mtp_logits), VOCAB);
    size_t nd = 1;
    for (; nd < k; nd++) {
        dequant_row(m->pack, &m->embed, out[nd - 1], gpu_floats(m->scratch) + S_y);
        Enc e2 = enc_begin(m);
        enc_mtp_rows(&e2, s, sarg(m, S_y), h, 1, s->pos - 1 + nd, true, true);
        TRY(enc_finish(&e2));
        h = sarg(m, S_a);
        out[nd] = model_argmax(gpu_floats(m->mtp_logits), VOCAB);
    }
    *ndraft = nd;
    return OK;
}

// ---- Session ---------------------------------------------------------------------------------

size_t session_kv_elem(const Session *s) { return s->kv_half ? 2 : 4; }

Err session_create(Gpu *g, Session **out) { return session_create_kv(g, getenv("SAETTA_KV_F32") == NULL, out); }

Err session_create_kv(Gpu *g, bool kv_half, Session **out) {
    size_t cap = 32768;
    const char *v = getenv("SAETTA_CTX");
    if (v) {
        uint64_t c;
        TRY(parse_u64(str(v), &c));
        cap = (size_t)c;
    }
    return session_create_cap(g, kv_half, cap, cap, out);
}

#define GDN_STATE_N (n_gdn * NV * DK * DV)
#define GDN_CONV_N (n_gdn * (KCONV - 1) * CONV)

static Err new_state(Gpu *g, sx_buffer **gdn, sx_buffer **conv) {
    TRY(gpu_new_buffer(g, GDN_STATE_N * 4, gdn));
    TRY(gpu_new_buffer(g, GDN_CONV_N * 4, conv));
    return OK;
}

static size_t kv_bytes(size_t cap, size_t elem) { return n_kv * 2 * cap * NKV * HD * elem; }

Err session_create_cap(Gpu *g, bool kv_half, size_t cap, size_t max_cap, Session **out) {
    Session *s = xcalloc(1, sizeof *s);
    s->arch = &dense_arch;
    s->cap = cap;
    s->max_cap = max_cap;
    s->roll = SESSION_NO_ROLL;
    s->kv_half = kv_half;
    TRY(gpu_new_buffer(g, GDN_STATE_N * 4, &s->gdn_state));
    TRY(gpu_new_buffer(g, GDN_CONV_N * 4, &s->gdn_conv));
    TRY(gpu_new_buffer(g, kv_bytes(cap, kv_half ? 2 : 4), &s->kv));
    TRY(gpu_new_buffer(g, max_cap * ROT * 4, &s->rope));
    TRY(gpu_new_buffer(g, NLAYERS * HCW * 4, &s->trace_buf));
    TRY(gpu_new_buffer(g, KMAX * GDN_STATE_N * 4, &s->sp_gdn));
    TRY(gpu_new_buffer(g, KMAX * GDN_CONV_N * 4, &s->sp_conv));
    for (int i = 0; i < 2; i++) TRY(new_state(g, &s->ck[i].gdn, &s->ck[i].conv));
    float *tab = gpu_floats(s->rope);
    for (size_t pos = 0; pos < max_cap; pos++) rope_row(tab + pos * ROT, pos);
    *out = s;
    return OK;
}

size_t session_kv_bytes(const Session *s, size_t cap) { return kv_bytes(cap, session_kv_elem(s)); }

Err session_grow(Session *s, Gpu *g, size_t cap) {
    if (!(cap >= s->cap && cap <= s->max_cap && cap % 4 == 0)) die("session_grow: bad capacity %zu\n", cap);
    if (cap == s->cap) return OK;
    size_t old = s->cap;
    size_t kvrow = NKV * HD * session_kv_elem(s);
    sx_buffer *kv;
    TRY(gpu_new_buffer(g, kv_bytes(cap, session_kv_elem(s)), &kv));
    uint8_t *dkv = sx_buffer_contents(kv);
    const uint8_t *skv = sx_buffer_contents(s->kv);
    for (size_t plane = 0; plane < n_kv * 2; plane++) memcpy(dkv + plane * cap * kvrow, skv + plane * old * kvrow, old * kvrow);
    sx_buffer_release(s->kv);
    s->kv = kv;
    s->cap = cap;
    return OK;
}

static const float *live(const Session *s, sx_buffer *cur, sx_buffer *slots, size_t n) {
    return s->roll != SESSION_NO_ROLL ? gpu_floats(slots) + s->roll * n : gpu_floats(cur);
}

static void vec_set(VecU32 *v, const uint32_t *p, size_t n) {
    v->len = 0;
    vec_reserve(v, n);
    if (n) memcpy(v->ptr, p, n * sizeof(uint32_t));
    v->len = n;
}

static void save_state(const Session *s, sx_buffer *gdn, sx_buffer *conv) {
    memcpy(gpu_floats(gdn), live(s, s->gdn_state, s->sp_gdn, GDN_STATE_N), GDN_STATE_N * 4);
    memcpy(gpu_floats(conv), live(s, s->gdn_conv, s->sp_conv, GDN_CONV_N), GDN_CONV_N * 4);
}

Err session_save_checkpoint(Session *s, const uint32_t *toks, size_t n) {
    return session_save_checkpoint_at(s, 0, toks, n);
}

Err session_save_checkpoint_at(Session *s, size_t slot, const uint32_t *toks, size_t n) {
    if (n != s->pos) die("session_save_checkpoint_at: %zu tokens at position %zu\n", n, s->pos);
    Ckpt *k = &s->ck[slot];
    save_state(s, k->gdn, k->conv);
    vec_set(&k->tokens, toks, n);
    return OK;
}

size_t session_reusable(Session *s, const uint32_t *prompt, size_t n) {
    Reusable r = session_reusable_in(s->ck, prompt, n);
    s->ck_pick = r.pick;
    return r.n;
}

size_t session_conv_bytes(const Session *s, size_t cap) {
    return session_kv_bytes(s, cap) + 2 * (GDN_STATE_N * 4 + GDN_CONV_N * 4);
}

Err session_new_conv(const Session *s, Gpu *g, size_t cap, Conv *out) {
    Conv cv = {0};
    cv.cap = cap;
    TRY(gpu_new_buffer(g, kv_bytes(cap, session_kv_elem(s)), &cv.kv));
    for (int i = 0; i < 2; i++) TRY(new_state(g, &cv.ck[i].gdn, &cv.ck[i].conv));
    *out = cv;
    return OK;
}

Err session_conv_shrink(const Session *s, Gpu *g, Conv *cv, size_t cap) {
    Conv small;
    TRY(session_new_conv(s, g, cap, &small));
    sx_buffer_release(cv->kv);
    cv->cap = small.cap;
    cv->kv = small.kv;
    for (int k = 0; k < 2; k++) {
        sx_buffer_release(small.ck[k].gdn);
        sx_buffer_release(small.ck[k].conv);
    }
    for (int k = 0; k < 2; k++) cv->ck[k].tokens.len = 0;
    cv->ptoks.len = 0;
    return OK;
}

#define SWAP(T, a, b) do { T t_ = (a); (a) = (b); (b) = t_; } while (0)

void session_swap_conv(Session *s, Conv *cv) {
    SWAP(size_t, s->cap, cv->cap);
    SWAP(sx_buffer *, s->kv, cv->kv);
    for (int i = 0; i < 2; i++) SWAP(Ckpt, s->ck[i], cv->ck[i]);
    for (int i = 0; i < PT_MAX; i++) SWAP(Point, s->pts[i], cv->pts[i]);
    SWAP(VecU32, s->ptoks, cv->ptoks);
    s->roll = SESSION_NO_ROLL;
    s->mtp_ok = false;
    s->mtp_n = 0;
}

bool session_roll_to(Session *s, size_t p, const uint32_t *toks) {
    (void)toks;
    if (s->chunk_t == 0 || p <= s->chunk_pos0 || p > s->chunk_pos0 + s->chunk_t || p > s->pos) return false;
    s->roll = p - s->chunk_pos0 - 1;
    s->pos = p;
    s->mtp_ok = false;
    s->mtp_n = 0;
    return true;
}


Err session_save_point(Session *s, Gpu *g, const uint32_t *toks, size_t n) {
    if (!(s->pos % PT_EVERY == 0 && n == s->pos)) die("session_save_point: %zu tokens at position %zu\n", n, s->pos);
    size_t k = s->pos / PT_EVERY - 1;
    if (k >= PT_MAX) return OK;
    Point *pt = &s->pts[k];
    if (pt->gdn == NULL) TRY(new_state(g, &pt->gdn, &pt->conv));
    save_state(s, pt->gdn, pt->conv);
    vec_set(&s->ptoks, toks, n);
    return OK;
}

static void start_at(Session *s, const Point *pt, size_t n) {
    s->roll = SESSION_NO_ROLL;
    memcpy(gpu_floats(s->gdn_state), gpu_floats(pt->gdn), GDN_STATE_N * 4);
    memcpy(gpu_floats(s->gdn_conv), gpu_floats(pt->conv), GDN_CONV_N * 4);
    s->pos = n;
    s->mtp_ok = false;
    s->mtp_n = 0;
}

void session_start_at_own_point(Session *s, size_t n) {
    start_at(s, &s->pts[n / PT_EVERY - 1], n);
    session_keep_points(s, n);
}

Err session_start_from(Session *s, Gpu *g, const Conv *src, size_t n) {
    if (!(n % PT_EVERY == 0 && n > 0)) die("session_start_from: position %zu\n", n);
    if (!(n <= s->cap && n <= src->cap)) die("session_start_from: position %zu past capacity\n", n);
    size_t kvrow = NKV * HD * session_kv_elem(s);
    uint8_t *dkv = sx_buffer_contents(s->kv);
    const uint8_t *skv = sx_buffer_contents(src->kv);
    for (size_t plane = 0; plane < n_kv * 2; plane++) memcpy(dkv + plane * s->cap * kvrow, skv + plane * src->cap * kvrow, n * kvrow);
    size_t k = n / PT_EVERY - 1;
    start_at(s, &src->pts[k], n);
    for (size_t i = 0; i < k + 1; i++) {
        const Point *sp = &src->pts[i];
        if (sp->gdn == NULL) continue;
        Point *d = &s->pts[i];
        if (d->gdn == NULL) TRY(new_state(g, &d->gdn, &d->conv));
        memcpy(gpu_floats(d->gdn), gpu_floats(sp->gdn), GDN_STATE_N * 4);
        memcpy(gpu_floats(d->conv), gpu_floats(sp->conv), GDN_CONV_N * 4);
    }
    vec_set(&s->ptoks, src->ptoks.ptr, n);
    return OK;
}

void session_restore(Session *s) {
    s->roll = SESSION_NO_ROLL;
    Ckpt *k = &s->ck[s->ck_pick];
    memcpy(gpu_floats(s->gdn_state), gpu_floats(k->gdn), GDN_STATE_N * 4);
    memcpy(gpu_floats(s->gdn_conv), gpu_floats(k->conv), GDN_CONV_N * 4);
    s->pos = k->tokens.len;
    s->mtp_ok = false; // the trunk stream before the checkpoint is not kept
    s->mtp_n = 0;
}

void session_push_history(Session *s, uint32_t token) { (void)s, (void)token; }

void session_reset(Session *s) {
    s->roll = SESSION_NO_ROLL;
    s->pos = 0;
    s->mtp_ok = false;
    s->mtp_n = 0;
    memset(gpu_floats(s->gdn_state), 0, GDN_STATE_N * 4);
    memset(gpu_floats(s->gdn_conv), 0, GDN_CONV_N * 4);
}

// ---- LoRA adapters ------------------------------------------------------------------------------

int model_lora_find(const Model *m, const char *name) {
    for (size_t i = 0; i < m->n_lora; i++)
        if (!strcmp(m->lora[i].name, name)) return (int)i;
    return -1;
}
void model_lora_select(Model *m, int idx) { m->lora_on = idx >= 0 && (size_t)idx < m->n_lora ? idx : -1; }
int model_lora_selected(const Model *m) { return m->lora_on; }
size_t model_lora_count(const Model *m) { return m->n_lora; }
const char *model_lora_name(const Model *m, size_t i) { return i < m->n_lora ? m->lora[i].name : NULL; }

// Target modules: PEFT name, group, slot in the group, whether the module is in attention layers.
static const struct {
    const char *mod;
    int group, slot;
    bool attn;
} lora_targets[] = {
    {"self_attn.q_proj", LORA_QKV, 0, true},  {"self_attn.k_proj", LORA_QKV, 1, true}, {"self_attn.v_proj", LORA_QKV, 2, true},
    {"self_attn.o_proj", LORA_O, 0, true},    {"mlp.gate_proj", LORA_GU, 0, false},    {"mlp.up_proj", LORA_GU, 1, false},
    {"mlp.down_proj", LORA_DOWN, 0, false},
};

static const Linear *base_linear(const Layer *L, size_t k) {
    switch (k) {
    case 0: return L->attn ? &L->attn->q : NULL;
    case 1: return L->attn ? &L->attn->k : NULL;
    case 2: return L->attn ? &L->attn->v : NULL;
    case 3: return L->attn ? &L->attn->o : NULL;
    case 4: return &L->mlp.gate;
    case 5: return &L->mlp.up;
    default: return &L->mlp.down;
    }
}

static const JValue *jget(const JValue *o, const char *k) { return o ? json_get(o, k) : NULL; }

static void put_half(uint8_t *dst, const uint8_t *src_f32, size_t n) {
    _Float16 *d = (_Float16 *)dst;
    for (size_t i = 0; i < n; i++) {
        float f;
        memcpy(&f, src_f32 + i * 4, 4);
        d[i] = (_Float16)f;
    }
}

Err model_lora_load(Arena *a, Model *m, const char *name, const char *dir) {
    if (m->n_lora == MODEL_MAX_LORA) return "TooManyAdapters";
    if (!m->lp_ok) {
        TRY(lora_init(m->g, &m->lp));
        m->lp_ok = true;
    }
    // adapter_config.json: PEFT LoRA without DoRA, bias or saved modules
    char *text;
    size_t len;
    TRY(read_file(path_join(a, dir, "adapter_config.json"), 1u << 20, &text, &len));
    JValue *cfg;
    Err e = json_parse(a, str_n(text, len), true, &cfg);
    free(text);
    TRY(e);
    const JValue *jr = jget(cfg, "r"), *ja = jget(cfg, "lora_alpha"), *jp = jget(cfg, "peft_type");
    if (!jr || !ja || jr->kind != J_INTEGER || (ja->kind != J_INTEGER && ja->kind != J_FLOAT)) return "BadAdapterConfig";
    if (jp && !(jp->kind == J_STRING && str_eq(jp->s, S("LORA")))) return "BadAdapterConfig";
    const JValue *jd = jget(cfg, "use_dora"), *jb = jget(cfg, "bias"), *js = jget(cfg, "modules_to_save");
    if ((jd && jd->kind == J_BOOL && jd->b) || (jb && jb->kind == J_STRING && !str_eq(jb->s, S("none"))) ||
        (js && js->kind == J_ARRAY && js->arr.len > 0)) {
        dbg("lora %s: DoRA, bias and modules_to_save are not supported\n", name);
        return "BadAdapterConfig";
    }
    const JValue *jrs = jget(cfg, "use_rslora");
    bool rslora = jrs && jrs->kind == J_BOOL && jrs->b;
    const JValue *rp = jget(cfg, "rank_pattern"), *ap = jget(cfg, "alpha_pattern");
    if ((rp && rp->kind == J_OBJECT && rp->obj.len > 0) || (ap && ap->kind == J_OBJECT && ap->obj.len > 0)) {
        dbg("lora %s: rank_pattern and alpha_pattern are not supported\n", name);
        return "BadAdapterConfig";
    }
    uint32_t r = (uint32_t)jr->i;
    double alpha = ja->kind == J_INTEGER ? (double)ja->i : ja->f;
    if (r == 0 || r % 8 != 0 || r > LORA_MAX_R) {
        dbg("lora %s: rank %u (must be a multiple of 8, at most %u)\n", name, r, LORA_MAX_R);
        return "BadAdapterConfig";
    }
    // the tensors, by the part of the name from "layers." on
    const char *wpath = path_join(a, dir, "adapter_model.safetensors");
    size_t flen;
    const uint8_t *bytes = sx_map_file(wpath, &flen);
    if (!bytes) return "MapFailed";
    Err err = OK;
    StrMap raw = {0}, by = {0};
    if ((err = safetensors_parse(a, bytes, flen, 0, &raw))) goto out;
    Str k;
    void *v;
    size_t ntens = 0;
    for (size_t it = 0; strmap_next(&raw, &it, &k, &v);) {
        const char *p = strstr(arena_strndup(a, k.ptr, k.len), "layers.");
        if (!p) {
            dbg("lora %s: tensor %.*s is not a layer weight\n", name, (int)k.len, k.ptr);
            err = "LoraTargetUnsupported";
            goto out;
        }
        strmap_put(&by, str(p), v);
        ntens++;
    }
    Adapter *ad = &m->lora[m->n_lora];
    *ad = (Adapter){.name = arena_strndup(a, name, strlen(name)), .r = r,
                    .scale = (float)(rslora ? alpha / sqrt((double)r) : alpha / (double)r)};
    // pass 0 sizes the buffer, pass 1 fills it
    size_t used = 0, nused = 0;
    uint8_t *dst = NULL;
    for (int pass = 0; pass < 2; pass++) {
        size_t at = 0;
        nused = 0;
        for (size_t li = 0; li < NLAYERS; li++) {
            const Layer *L = &m->layers[li];
            for (int gk = 0; gk < LORA_NGROUPS; gk++) {
                LoraGroup G = {0};
                // A rows of the group's present members, concatenated, then each member's B
                const TensorInfo *ta[3] = {0}, *tb[3] = {0};
                const Linear *bl[3] = {0};
                uint32_t slot[3] = {0};
                for (size_t ti = 0; ti < COUNT(lora_targets); ti++) {
                    if (lora_targets[ti].group != gk) continue;
                    char an[160], bn[160];
                    snprintf(an, sizeof an, "layers.%zu.%s.lora_A.weight", li, lora_targets[ti].mod);
                    snprintf(bn, sizeof bn, "layers.%zu.%s.lora_B.weight", li, lora_targets[ti].mod);
                    void **pa = strmap_get(&by, str(an)), **pb = strmap_get(&by, str(bn));
                    if (!pa && !pb) continue;
                    if (!pa || !pb) {
                        err = "LoraUnmatchedTensor";
                        goto out;
                    }
                    const Linear *base = base_linear(L, ti);
                    if (!base) {
                        dbg("lora %s: %s has no %s\n", name, an, lora_targets[ti].mod);
                        err = "LoraTargetUnsupported";
                        goto out;
                    }
                    const TensorInfo *A = *pa, *B = *pb;
                    if (A->dtype != DTYPE_F32 || B->dtype != DTYPE_F32 || A->rank != 2 || B->rank != 2 || A->shape[0] != r ||
                        A->shape[1] != base->in_features || B->shape[0] != base->out_features || B->shape[1] != r) {
                        dbg("lora %s: %s shape or dtype does not match the base linear\n", name, an);
                        err = "LoraShapeMismatch";
                        goto out;
                    }
                    int sl = (int)G.n;
                    ta[sl] = A;
                    tb[sl] = B;
                    bl[sl] = base;
                    slot[sl] = (uint32_t)lora_targets[ti].slot;
                    G.n++;
                    nused += 2;
                }
                if (G.n == 0) continue;
                G.on = true;
                G.in = (uint32_t)bl[0]->in_features;
                G.rt = G.n * r;
                G.a = at;
                for (uint32_t i = 0; i < G.n; i++) {
                    if (pass) put_half(dst + at, bytes + ta[i]->offset, (size_t)r * G.in);
                    at = ALIGN_UP(at + (size_t)r * G.in * 2, 16);
                }
                for (uint32_t i = 0; i < G.n; i++) {
                    G.up[i].b = at;
                    G.up[i].out = (uint32_t)bl[i]->out_features;
                    G.up[i].r0 = i * r;
                    G.up[i].slot = slot[i];
                    if (pass) put_half(dst + at, bytes + tb[i]->offset, (size_t)G.up[i].out * r);
                    at = ALIGN_UP(at + (size_t)G.up[i].out * r * 2, 16);
                }
                if (pass) ad->g[li][gk] = G;
            }
        }
        if (nused != ntens) {
            for (size_t it = 0; strmap_next(&by, &it, &k, &v);) {
                (void)v;
                dbg("lora %s: tensor %.*s not applied\n", name, (int)k.len, k.ptr);
                break;
            }
            err = "LoraUnmatchedTensor";
            goto out;
        }
        if (pass == 0) {
            used = at;
            if ((err = gpu_new_buffer(m->g, MAX(used, (size_t)16), &ad->buf))) goto out;
            dst = sx_buffer_contents(ad->buf);
        }
    }
out:
    sx_unmap_file(bytes, flen);
    strmap_free(&by);
    strmap_free(&raw);
    if (err) {
        if (m->lora[m->n_lora].buf) sx_buffer_release(m->lora[m->n_lora].buf);
        m->lora[m->n_lora] = (Adapter){0};
        return err;
    }
    char mb[32];
    fmt_fixed(mb, sizeof mb, (double)used / 1e6, 1);
    dbg("model: LoRA %s loaded: %zu tensors, r %u, scale %g, %s MB f16\n", ad->name, nused, r, (double)ad->scale, mb);
    m->n_lora++;
    return OK;
}

// ---- loading helpers ---------------------------------------------------------------------------

static Err lin(Arena *a, const Pack *pack, const char *p, const char *name, Linear *out) {
    return pack_linear(pack, str(cat(a, p, name)), out);
}

// A BF16 or F32 tensor as f32 in the GPU parameter buffer.
static Err vec(Arena *a, const Pack *pack, Params *P, const char *p, const char *name, Vec *out) {
    const char *full = cat(a, p, name);
    const TensorInfo *t = pack_get(pack, str(full));
    if (!t) {
        dbg("missing tensor %s\n", full);
        return "MissingTensor";
    }
    if (t->dtype != DTYPE_BF16 && t->dtype != DTYPE_F32) return "UnexpectedDType";
    const uint8_t *raw = pack_data(pack, t);
    TRY(params_take(P, (size_t)tensor_numel(t), out));
    if (t->dtype == DTYPE_F32)
        memcpy(out->s, raw, out->n * 4);
    else
        for (size_t i = 0; i < out->n; i++) out->s[i] = ops_bf16_to_f32((uint16_t)(raw[i * 2] | (raw[i * 2 + 1] << 8)));
    return OK;
}

// The MTP head (tensor names as HF's, under mtp.), matrices at whatever (bits, group) the pack
// stores; the layer's attention takes the KV plane after the trunk's.
static Err load_mtp(Arena *a, const Pack *pack, Params *P, Mtp *mt) {
    *mt = (Mtp){0};
    TRY(pack_linear(pack, S("mtp.fc"), &mt->fc));
    TRY(vec(a, pack, P, "mtp.", "pre_fc_norm_embedding.weight", &mt->n_e));
    TRY(vec(a, pack, P, "mtp.", "pre_fc_norm_hidden.weight", &mt->n_h));
    TRY(vec(a, pack, P, "mtp.", "norm.weight", &mt->norm));
    const char *p = "mtp.layers.0.";
    Layer *L = &mt->layer;
    TRY(vec(a, pack, P, p, "input_layernorm.weight", &L->in_norm));
    TRY(vec(a, pack, P, p, "post_attention_layernorm.weight", &L->post_norm));
    Attn *A = ARENA_NEW(a, Attn);
    A->slot = n_attn;
    TRY(lin(a, pack, p, "self_attn.q_proj", &A->q));
    TRY(lin(a, pack, p, "self_attn.k_proj", &A->k));
    TRY(lin(a, pack, p, "self_attn.v_proj", &A->v));
    TRY(lin(a, pack, p, "self_attn.o_proj", &A->o));
    TRY(vec(a, pack, P, p, "self_attn.q_norm.weight", &A->q_norm));
    TRY(vec(a, pack, P, p, "self_attn.k_norm.weight", &A->k_norm));
    L->attn = A;
    L->gdn = NULL;
    TRY(lin(a, pack, p, "mlp.gate_proj", &L->mlp.gate));
    TRY(lin(a, pack, p, "mlp.up_proj", &L->mlp.up));
    TRY(lin(a, pack, p, "mlp.down_proj", &L->mlp.down));
    return OK;
}

// Dequantizes row r of an affine quantized matrix into out[in_features] (see model.c).
static void dequant_row(const Pack *pack, const Linear *l, size_t r, float *out) {
    size_t bits = l->bits;
    uint32_t mask = (1u << bits) - 1;
    size_t in = (size_t)l->in_features;
    size_t groups = in / l->group_size;
    const uint8_t *w = pack_data(pack, &l->weight);
    const uint8_t *s = pack_data(pack, &l->scales);
    const uint8_t *b = pack_data(pack, &l->biases);
    const uint8_t *row = w + r * (in * bits / 8);
    for (size_t i = 0; i < in; i++) {
        size_t bit = i * bits;
        uint32_t lo = row[bit / 8];
        uint32_t hi = bit / 8 + 1 < in * bits / 8 ? row[bit / 8 + 1] : 0;
        uint32_t q = ((lo | (hi << 8)) >> (bit % 8)) & mask;
        size_t g = r * groups + i / l->group_size;
        float sc = ops_bf16_to_f32((uint16_t)(s[g * 2] | (s[g * 2 + 1] << 8)));
        float bi = ops_bf16_to_f32((uint16_t)(b[g * 2] | (b[g * 2 + 1] << 8)));
        out[i] = (float)q * sc + bi;
    }
}


// ---- the table model_api.c dispatches through ----------------------------------------------

bool model_has_mtp(const Model *m) { return m->mtp != NULL; }
bool model_fast_prefill(const Model *m) { return m->fast_prefill; }
bool model_fast_prefill_available(const Model *m) { return m->mpp != NULL; }
void model_set_fast_prefill(Model *m, bool on) { m->fast_prefill = on && m->mpp != NULL; }

size_t model_check_pack(Arena *a, const Pack *pack, FILE *out) {
    fprintf(out, "model: Qwen3.8-27B (%s), %zu layers, hidden %zu, dense MLP %zu\n", CONFIG_MODEL_TYPE, NLAYERS, H, FFN);
    size_t n = 0, bad = 0;
    for (size_t l = 0; l <= NLAYERS; l++) {
        struct {
            const char *name;
            uint64_t out, in;
        } want[8];
        size_t k = 0;
        if (l == NLAYERS) {
            want[k++] = (typeof(want[0])){"language_model.lm_head", VOCAB, H};
            want[k++] = (typeof(want[0])){WEIGHTS_PREFIX "embed_tokens", VOCAB, H};
        } else {
            if (config_layer_kinds[l] == LAYER_KIND_GDN) {
                want[k++] = (typeof(want[0])){"linear_attn.in_proj_qkv", CONV, H};
                want[k++] = (typeof(want[0])){"linear_attn.in_proj_z", VD, H};
                want[k++] = (typeof(want[0])){"linear_attn.in_proj_a", NV, H};
                want[k++] = (typeof(want[0])){"linear_attn.in_proj_b", NV, H};
                want[k++] = (typeof(want[0])){"linear_attn.out_proj", H, VD};
            } else {
                want[k++] = (typeof(want[0])){"self_attn.q_proj", 2 * NQ * HD, H};
                want[k++] = (typeof(want[0])){"self_attn.k_proj", NKV * HD, H};
                want[k++] = (typeof(want[0])){"self_attn.v_proj", NKV * HD, H};
                want[k++] = (typeof(want[0])){"self_attn.o_proj", H, NQ * HD};
            }
            want[k++] = (typeof(want[0])){"mlp.gate_proj", FFN, H};
            want[k++] = (typeof(want[0])){"mlp.up_proj", FFN, H};
            want[k++] = (typeof(want[0])){"mlp.down_proj", H, FFN};
        }
        for (size_t i = 0; i < k; i++) {
            const char *mod = l == NLAYERS ? want[i].name : arena_printf(a, "%slayers.%zu.%s", WEIGHTS_PREFIX, l, want[i].name);
            Linear lin;
            Err err = pack_linear(pack, str(mod), &lin);
            n++;
            if (err) {
                fprintf(out, "  MISSING %s: %s\n", mod, err);
                bad++;
            } else if (lin.out_features != want[i].out || lin.in_features != want[i].in) {
                fprintf(out, "  SHAPE %s: got %llux%llu, want %llux%llu\n", mod, (unsigned long long)lin.out_features,
                        (unsigned long long)lin.in_features, (unsigned long long)want[i].out, (unsigned long long)want[i].in);
                bad++;
            }
        }
    }
    if (pack_get(pack, S("mtp.fc.weight")) != NULL) {
        static const struct {
            const char *name;
            uint64_t out, in;
        } mw[] = {
            {"mtp.fc", H, 2 * H},
            {"mtp.layers.0.self_attn.q_proj", 2 * NQ * HD, H},
            {"mtp.layers.0.self_attn.k_proj", NKV * HD, H},
            {"mtp.layers.0.self_attn.v_proj", NKV * HD, H},
            {"mtp.layers.0.self_attn.o_proj", H, NQ * HD},
            {"mtp.layers.0.mlp.gate_proj", FFN, H},
            {"mtp.layers.0.mlp.up_proj", FFN, H},
            {"mtp.layers.0.mlp.down_proj", H, FFN},
        };
        for (size_t i = 0; i < COUNT(mw); i++) {
            Linear lin;
            Err err = pack_linear(pack, str(mw[i].name), &lin);
            n++;
            if (err) {
                fprintf(out, "  MISSING %s: %s\n", mw[i].name, err);
                bad++;
            } else if (lin.out_features != mw[i].out || lin.in_features != mw[i].in) {
                fprintf(out, "  SHAPE %s: got %llux%llu, want %llux%llu\n", mw[i].name,
                        (unsigned long long)lin.out_features, (unsigned long long)lin.in_features,
                        (unsigned long long)mw[i].out, (unsigned long long)mw[i].in);
                bad++;
            }
        }
    }
    fprintf(out, "linears: %zu checked, %zu problems\n", n, bad);
    return bad;
}

const ModelArch dense_arch = MODEL_ARCH_TABLE({
    .model_type = CONFIG_MODEL_TYPE,
    .id = "qwen3.8-27b",
    .name = "Qwen3.8-27B",
    .layers = NLAYERS,
    .width = HCW,
    .vocab = VOCAB,
    .hidden = H,
    .eos = config_eos_token_ids,
    .n_eos = CONFIG_N_EOS_TOKEN_IDS,
});
