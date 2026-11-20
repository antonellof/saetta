// Qwen3.8-Flash-Next forward pass (see model.h; internals model_flash.h, constants
// config_flash.h). Exports flash_arch (model_arch.h).
#include "model_flash.h"
#define ARCH_PFX flash_
#include "model_rename.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>

#include "ops.h"
#include "qgen.h"

#define H CONFIG_HIDDEN_SIZE // 2560
#define HC CONFIG_HC_COUNT   // 4
#define LOW CONFIG_HC_LOWRANK // 320
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
#define NQ CONFIG_NUM_ATTENTION_HEADS // 24
#define NKV CONFIG_NUM_KEY_VALUE_HEADS // 2
#define HD CONFIG_HEAD_DIM // 256
#define ROT ((size_t)((float)HD * CONFIG_PARTIAL_ROTARY_FACTOR)) // 64
#define IXH CONFIG_INDEXER_N_HEADS // 4
#define IXD CONFIG_INDEXER_HEAD_DIM // 128
#define BLOCK CONFIG_INDEXER_COMPRESS_RATIO // 4
#define TOPB (CONFIG_INDEXER_BUDGET / BLOCK) // 512
#define MAXSEL (TOPB * BLOCK + BLOCK - 1) // 2051
// MoE
#define NE CONFIG_NUM_EXPERTS
#define TOPK CONFIG_NUM_EXPERTS_PER_TOK
#define FF CONFIG_MOE_INTERMEDIATE_SIZE
#define SFF CONFIG_SHARED_EXPERT_INTERMEDIATE_SIZE
// PLE
#define NG_HEADS MODEL_NG_HEADS // 16
#define NG_DIM (CONFIG_PLE_EMBED_DIM / NG_HEADS) // 160
#define PLE_CONV_STATE ((CONFIG_PLE_CONV_KERNEL_SIZE - 1) * CONFIG_NGRAM_SIZE) // 9
#define EOS_PLE 248044u // text_config eos_token_id (config.json), used for n-gram history
#define VOCAB CONFIG_VOCAB_SIZE
#define NLAYERS CONFIG_NUM_HIDDEN_LAYERS

static size_t n_gdn, n_attn;
__attribute__((constructor)) static void count_layers(void) {
    size_t n = 0;
    for (size_t i = 0; i < NLAYERS; i++) n += config_layer_kinds[i] == LAYER_KIND_GDN;
    n_gdn = n;
    n_attn = NLAYERS - n;
}

static Err params_take(Params *p, size_t n, Vec *out) {
    float *all = gpu_floats(p->buf);
    if (p->used + n > gpu_nfloats(p->buf)) return "ParamsFull";
    *out = (Vec){p->used, all + p->used, n};
    p->used += (n + 63) & ~(size_t)63;
    return OK;
}

// Scratch activations, offsets in floats into one GPU buffer.
constexpr size_t S_X = 0;
constexpr size_t S_xn = S_X + HCW;
constexpr size_t S_low = S_xn + HCW;
constexpr size_t S_mix = S_low + LOW;
constexpr size_t S_a = S_mix + HCW;
constexpr size_t S_r = S_a + H;
constexpr size_t S_y = S_r + 64;
constexpr size_t S_qkv = S_y + H;
constexpr size_t S_z = S_qkv + CONV;
constexpr size_t S_gb = S_z + VD;
constexpr size_t S_ga = S_gb + 64;
constexpr size_t S_conv = S_ga + 64;
constexpr size_t S_o = S_conv + CONV;
constexpr size_t S_qg = S_o + VD;
constexpr size_t S_qn = S_qg + NQ * 2 * HD;
constexpr size_t S_ix = S_qn + NQ * HD;
constexpr size_t S_iq = S_ix + (IXH + 1) * IXD;
constexpr size_t S_att = S_iq + IXH * IXD;
constexpr size_t S_rl = S_att + NQ * HD;
constexpr size_t S_topi = S_rl + NE;
constexpr size_t S_topw = S_topi + 64;
constexpr size_t S_eg = S_topw + 64;
constexpr size_t S_eu = S_eg + TOPK * FF;
constexpr size_t S_eh = S_eu + TOPK * FF;
constexpr size_t S_ed = S_eh + (TOPK + 1) * FF; // slot TOPK: the shared expert (moe, folded launches)
constexpr size_t S_shg = S_ed + (TOPK + 1) * H;
constexpr size_t S_shu = S_shg + SFF;
constexpr size_t S_shh = S_shu + SFF;
constexpr size_t S_sh = S_shh + SFF;
constexpr size_t S_sgs = S_sh + H;
constexpr size_t S_pe = S_sgs + 64;
constexpr size_t S_kp = S_pe + CONFIG_PLE_EMBED_DIM;
constexpr size_t S_kn = S_kp + HCW;
constexpr size_t S_val = S_kn + HCW;
constexpr size_t S_qp = S_val + H;
constexpr size_t S_gv = S_qp + HCW;
constexpr size_t S_gvn = S_gv + HCW;
constexpr size_t S_sel = S_gvn + HCW;
constexpr size_t S_cnt = S_sel + MAXSEL + 13;
constexpr size_t S_kx = S_cnt + 16; // this token's keys, values before they go to the KV cache
constexpr size_t S_vx = S_kx + NKV * HD;
constexpr size_t S_h = S_vx + NKV * HD;
constexpr size_t S_total = S_h + H;

// Chunks of at most this many tokens use the decode-style kernels (weights read once,
// applied per row) instead of the prefill matmuls, whose per-tile cost dominates there.
#define SMALL MODEL_SMALL
// Attention and QSA scoring run over a chunk in slices of AB queries, so their partials
// and score rows are sized for AB, not CH (a 64K prompt swapped with CH-sized buffers).
#define AB 512
#define NSPLIT ((MAXSEL + 63) / 64) // decode.metal SPL
// Prefill chunk scratch layout (floats) in one GPU buffer.
constexpr size_t B_X = 0;
constexpr size_t B_xn = B_X + CH * HCW;
constexpr size_t B_low = B_xn + CH * HCW;
constexpr size_t B_mix = B_low + CH * LOW;
constexpr size_t B_a = B_mix + CH * HCW;
constexpr size_t B_r = B_a + CH * H;
constexpr size_t B_y = B_r + CH * HC;
constexpr size_t B_qkv = B_y + CH * H;
constexpr size_t B_z = B_qkv + CH * CONV;
constexpr size_t B_gb = B_z + CH * VD;
constexpr size_t B_ga = B_gb + CH * NV;
constexpr size_t B_conv = B_ga + CH * NV;
constexpr size_t B_o = B_conv + CH * CONV;
constexpr size_t B_qg = B_o + CH * VD;
constexpr size_t B_qn = B_qg + CH * NQ * 2 * HD;
constexpr size_t B_ix = B_qn + CH * NQ * HD;
constexpr size_t B_att = B_ix + CH * (IXH + 1) * IXD;
constexpr size_t B_rl = B_att + CH * NQ * HD;
constexpr size_t B_topi = B_rl + CH * NE;
constexpr size_t B_topw = B_topi + CH * TOPK;
constexpr size_t B_eg = B_topw + CH * TOPK;
constexpr size_t B_eu = B_eg + CH * TOPK * FF;
constexpr size_t B_eh = B_eu + CH * TOPK * FF;
constexpr size_t B_ed = B_eh + CH * TOPK * FF;
constexpr size_t B_shg = B_ed + CH * TOPK * H;
constexpr size_t B_shu = B_shg + CH * SFF;
constexpr size_t B_shh = B_shu + CH * SFF;
constexpr size_t B_sh = B_shh + CH * SFF;
constexpr size_t B_sgs = B_sh + CH * H;
constexpr size_t B_pe = B_sgs + CH;
constexpr size_t B_kp = B_pe + CH * CONFIG_PLE_EMBED_DIM;
constexpr size_t B_kn = B_kp + CH * HCW;
constexpr size_t B_val = B_kn + CH * HCW;
constexpr size_t B_qp = B_val + CH * H;
constexpr size_t B_gv = B_qp + CH * HCW;
constexpr size_t B_gvn = B_gv + CH * HCW;
constexpr size_t B_xg = B_gvn + CH * HCW; // expert-grouped inputs [CH*TOPK][H]
constexpr size_t B_gtok = B_xg + CH * TOPK * H; // u32 [CH*TOPK]
constexpr size_t B_ginv = B_gtok + CH * TOPK; // u32 [CH*TOPK]
constexpr size_t B_gtiles = B_ginv + CH * TOPK; // uint4 [CH*TOPK/32 + NE]
constexpr size_t B_gnt = B_gtiles + 4 * (CH * TOPK / 32 + NE);
constexpr size_t B_sel = B_gnt + 16; // u32 [CH][MAXSEL] per-token QSA lists
constexpr size_t B_cnt = B_sel + CH * (MAXSEL + 13); // u32 [CH]
constexpr size_t B_part = B_cnt + CH; // attention partials [AB][NQ][NSPLIT][HD+2]
constexpr size_t B_kx = B_part + AB * NQ * NSPLIT * (HD + 2); // chunk keys, values before the KV cache
constexpr size_t B_vx = B_kx + CH * NKV * HD;
constexpr size_t B_iqs = B_vx + CH * NKV * HD; // indexer queries [CH][IXH][IXD], normed, roped
constexpr size_t B_xh = B_iqs + CH * IXH * IXD; // half copy of a matmul input
constexpr size_t B_qh = B_xh + MAX(CH * HCW, CH * TOPK * H) / 2 + 16; // half copy of the chunk's queries
constexpr size_t B_wh = B_qh + CH * NQ * HD / 2 + 16; // f16 copy of one dense weight matrix (prompt matmuls)
constexpr size_t B_total = B_wh + 12288 * 2560 / 2 + 16;

static const char kernel_src[] = {
#embed "kernels/qmatvec.metal"
    , '\n',
#embed "kernels/decode.metal"
    , '\n',
#embed "kernels/qmm.metal"
    , 0};

static const char mpp_src[] = {
#embed "kernels/qmm_mpp.metal"
    , 0};

static Err gen_load(Gpu *g, Gen *r) {
    const char *src = qgen_src;
    TRY(gpu_pipeline(g, src, "qmatvec_gen", &r->mv));
    TRY(gpu_pipeline(g, src, "qmatvec_expert_gen", &r->expert));
    TRY(gpu_pipeline(g, src, "qmv_multi_l8_gen", &r->multi[0]));
    TRY(gpu_pipeline(g, src, "qmv_multi_l32_gen", &r->multi[1]));
    TRY(gpu_pipeline(g, src, "qmv_multi_l256_gen", &r->multi[2]));
    TRY(gpu_pipeline(g, src, "qmv_gate_up_gen", &r->gate_up));
    TRY(gpu_pipeline(g, src, "qmv_gate_up_sh_gen", &r->gate_up_sh));
    TRY(gpu_pipeline(g, src, "qmv_down_sh_gen", &r->down_sh));
    TRY(gpu_pipeline(g, src, "qmv_up_mix_gen", &r->up_mix));
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
    TRY(gpu_pipeline(g, src, "qmm_grouped_gen", &r->qmm_grouped));
    if (gpu_pipeline(g, qgen_mpp_src, "dq_f16_gen", &r->dq)) r->dq = NULL;
    if (gpu_pipeline(g, qgen_mpp_src, "qmmh_gen", &r->qmmh)) r->qmmh = NULL;
    if (gpu_pipeline(g, qgen_mpp_src, "qmmh_g_gen", &r->qmmh_g)) r->qmmh_g = NULL;
    return OK;
}

// The 4-bit group-32 / 8-bit group-64 templated kernels serve this linear.
static bool fast_q(const Linear *l) {
    return (l->bits == 4 && l->group_size == 32) || (l->bits == 8 && l->group_size == 64);
}
static bool same_quant(const Linear *a, const Linear *b) {
    return a->bits == b->bits && a->group_size == b->group_size;
}
static bool is4_32(const Linear *l) { return l->bits == 4 && l->group_size == 32; }

static Err load_residual(Arena *a, const Pack *pack, Params *P, const char *p, bool inject, Residual *out);
static Err lin(Arena *a, const Pack *pack, const char *p, const char *name, Linear *out);
static Err vec(Arena *a, const Pack *pack, Params *P, const char *p, const char *name, Vec *out);
static Err rq_lin(Arena *arena, const Pack *pack, Gpu *g, const char *name, uint32_t bits, Linear *out);
static Err load_mtp(Arena *arena, const Pack *pack, Gpu *g, Params *P, Mtp *mt);
static Err load_ple(Arena *a, const Pack *pack, Params *P, const char *p, Ple *L);
static char *cat(Arena *a, const char *x, const char *y) { return arena_printf(a, "%s%s", x, y); }
static void dequant_row(const Pack *pack, const Linear *l, size_t r, float *out);

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

static bool quant_fast(Quant q) { return (q.bits == 4 && q.group_size == 32) || (q.bits == 8 && q.group_size == 64); }

Err model_load(Arena *arena, const Pack *pack, Gpu *g, Model **out) {
    Pipes pipes;
    sx_pipeline **pp = (sx_pipeline **)&pipes;
    for (size_t i = 0; i < NPIPES; i++) TRY(gpu_pipeline(g, kernel_src, pipe_kernels[i], &pp[i]));
    Mpp *mpp = NULL;
    {
        sx_pipeline *dn;
        if (!gpu_pipeline(g, mpp_src, "qmmh_4_32", &dn)) {
            sx_pipeline *qsa;
            Err e = gpu_pipeline(g, mpp_src, "qsa_flash_tr", &qsa);
            if (e) {
                dbg("model: accelerator attention unavailable (%s)\n", e);
                qsa = NULL;
            }
            Mpp mp = {.dense = dn};
            TRY(gpu_pipeline(g, mpp_src, "qmmh_g_4_32", &mp.grouped));
            TRY(gpu_pipeline(g, mpp_src, "to_half", &mp.to_half));
            mp.qsa = getenv("SAETTA_QSA_SIMD") != NULL ? NULL : qsa;
            if (getenv("SAETTA_DENSE_FUSED") == NULL) {
                if (gpu_pipeline(g, mpp_src, "dq_f16_4_32", &mp.dq)) mp.dq = NULL;
                if (gpu_pipeline(g, mpp_src, "dq_bf16_f16", &mp.dq_bf16)) mp.dq_bf16 = NULL;
                if (gpu_pipeline(g, mpp_src, "mmh_dense", &mp.mmh)) mp.mmh = NULL;
                if (mp.dq == NULL || mp.mmh == NULL) {
                    dbg("model: f16 GEMM path unavailable, dense prompt matmuls stay fused\n");
                    mp.dq = NULL;
                    mp.mmh = NULL;
                }
            }
            mpp = ARENA_NEW(arena, Mpp);
            *mpp = mp;
        }
    }
    if (mpp == NULL) dbg("model: accelerator matmul kernels unavailable, prefill stays f32\n");
    sx_pipeline *rows_p[2][SMALL + 1], *rows1_p[2][SMALL + 1];
    for (int t = 1; t <= SMALL; t++) {
        char nm[64];
        snprintf(nm, sizeof nm, "qmv_rows_4_32_%d", t);
        TRY(gpu_pipeline(g, kernel_src, nm, &rows_p[0][t]));
        snprintf(nm, sizeof nm, "qmv_rows_8_64_%d", t);
        TRY(gpu_pipeline(g, kernel_src, nm, &rows_p[1][t]));
        snprintf(nm, sizeof nm, "qmv_rows1_4_32_%d", t);
        TRY(gpu_pipeline(g, kernel_src, nm, &rows1_p[0][t]));
        snprintf(nm, sizeof nm, "qmv_rows1_8_64_%d", t);
        TRY(gpu_pipeline(g, kernel_src, nm, &rows1_p[1][t]));
    }
    for (int i = 0; i < 2; i++) {
        rows_p[i][0] = rows_p[i][1];
        rows1_p[i][0] = rows1_p[i][1];
    }
    Model *m = ARENA_NEW(arena, Model);
    *m = (Model){0};
    m->arch = &flash_arch;
    *out = m;
    memcpy(m->rows_p, rows_p, sizeof rows_p);
    memcpy(m->rows1_p, rows1_p, sizeof rows1_p);
    m->pack = pack;
    m->g = g;
    m->p = pipes;
    m->mpp = mpp;
    m->fast_prefill = mpp != NULL && getenv("SAETTA_PREFILL_F32") == NULL;
    m->layers = ARENA_ARRAY(arena, Layer, NLAYERS);
    TRY(gpu_new_buffer(g, (size_t)64 << 20, &m->params.buf));
    TRY(gpu_new_buffer(g, S_total * 4, &m->scratch));
    TRY(gpu_new_buffer(g, B_total * 4, &m->bscratch));
    TRY(gpu_new_buffer(g, VOCAB * 4, &m->logits));
    TRY(gpu_new_buffer(g, KMAX * VOCAB * 4, &m->vlogits));
    TRY(gpu_new_buffer(g, (CH + 1) * HCW * 4, &m->mtp_hid));
    TRY(gpu_new_buffer(g, HCW * 4, &m->mtp_chain));
    TRY(gpu_new_buffer(g, VOCAB * 4, &m->mtp_logits));
    TRY(gpu_new_buffer(g, CH * (H + CONFIG_PLE_EMBED_DIM) * 4, &m->stage[0]));
    TRY(gpu_new_buffer(g, CH * (H + CONFIG_PLE_EMBED_DIM) * 4, &m->stage[1]));
    TRY(pack_linear(pack, S(WEIGHTS_PREFIX "embed_tokens"), &m->embed));
    TRY(pack_linear(pack, S("language_model.lm_head"), &m->lm_head));
    TRY(load_residual(arena, pack, &m->params, WEIGHTS_PREFIX "hyper_connection_mixer", false, &m->mixer));
    size_t gi = 0, ai = 0;
    for (size_t i = 0; i < NLAYERS; i++) {
        Layer *L = &m->layers[i];
        const char *p = arena_printf(arena, "%slayers.%zu.", WEIGHTS_PREFIX, i);
        Params *P = &m->params;
        TRY(load_residual(arena, pack, P, cat(arena, p, "attn_hyper_connection"), true, &L->attn_hc));
        TRY(load_residual(arena, pack, P, cat(arena, p, "mlp_hyper_connection"), true, &L->mlp_hc));
        L->gdn = NULL;
        L->attn = NULL;
        switch (config_layer_kinds[i]) {
        case LAYER_KIND_GDN: {
            Gdn *G = ARENA_NEW(arena, Gdn);
            G->slot = gi;
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
            gi++;
            break;
        }
        case LAYER_KIND_ATTENTION: {
            Attn *A = ARENA_NEW(arena, Attn);
            A->slot = ai;
            TRY(lin(arena, pack, p, "self_attn.q_proj", &A->q));
            TRY(lin(arena, pack, p, "self_attn.k_proj", &A->k));
            TRY(lin(arena, pack, p, "self_attn.v_proj", &A->v));
            TRY(lin(arena, pack, p, "self_attn.o_proj", &A->o));
            TRY(vec(arena, pack, P, p, "self_attn.q_norm.weight", &A->q_norm));
            TRY(vec(arena, pack, P, p, "self_attn.k_norm.weight", &A->k_norm));
            TRY(lin(arena, pack, p, "self_attn.indexer.index_qk_proj", &A->ix));
            TRY(vec(arena, pack, P, p, "self_attn.indexer.q_layernorm.weight", &A->ix_qn));
            TRY(vec(arena, pack, P, p, "self_attn.indexer.k_layernorm.weight", &A->ix_kn));
            L->attn = A;
            ai++;
            break;
        }
        }
        Moe *E = &L->moe;
        E->ne = NE;
        TRY(rq_lin(arena, pack, g, cat(arena, p, "mlp.gate"), 16, &E->router)); // BF16 kept exact
        TRY(rq_lin(arena, pack, g, cat(arena, p, "mlp.shared_expert_gate"), 8, &E->shared_gate));
        TRY(lin(arena, pack, p, "mlp.shared_expert.gate_proj", &E->sg));
        TRY(lin(arena, pack, p, "mlp.shared_expert.up_proj", &E->su));
        TRY(lin(arena, pack, p, "mlp.shared_expert.down_proj", &E->sd));
        TRY(lin(arena, pack, p, "mlp.switch_mlp.gate_proj", &E->eg));
        TRY(lin(arena, pack, p, "mlp.switch_mlp.up_proj", &E->eu));
        TRY(lin(arena, pack, p, "mlp.switch_mlp.down_proj", &E->ed));
        E->ne = (uint32_t)E->router.out_features;
        if (E->ne > NE) return "TooManyExperts";
        L->ple = NULL;
        for (size_t k = 0; k < CONFIG_N_PLE_LAYERS; k++)
            if (config_ple_layers[k] == i) {
                Ple *pl = ARENA_NEW(arena, Ple);
                TRY(load_ple(arena, pack, P, p, pl));
                L->ple = pl;
            }
    }
    // generic-quantization kernels when the pack has tensors outside the fast paths
    bool needs_gen = false;
    if (pack->has_quant_default && !quant_fast(pack->quant_default)) needs_gen = true;
    if (!needs_gen) {
        Str k;
        void *v;
        for (size_t it = 0; strmap_next(&pack->quant, &it, &k, &v);)
            if (!quant_fast(*(const Quant *)v)) {
                needs_gen = true;
                break;
            }
    }
    if (needs_gen) {
        Gen *gn = ARENA_NEW(arena, Gen);
        TRY(gen_load(g, gn));
        m->gen = gn;
        dbg("model: mixed quantization, run-time (bits, group) kernels loaded\n");
    }
    const char *mv = getenv("SAETTA_MTP");
    bool mtp_on = mv ? mv[0] != '0' : true;
    if (mtp_on && pack_get(pack, S("mtp.fc_embedding.weight")) != NULL) {
        Mtp *mt = ARENA_NEW(arena, Mtp);
        TRY(load_mtp(arena, pack, g, &m->params, mt));
        m->mtp = mt;
    }
    return OK;
}

static float *sf(Model *m) { return gpu_floats(m->scratch); }
static float *bf(Model *m) { return gpu_floats(m->bscratch); }

static inline sx_arg barg(sx_buffer *b, size_t off_floats) { return (sx_arg){b, off_floats * 4, NULL, 0}; }
static inline sx_arg sarg(Model *m, size_t off) { return barg(m->scratch, off); }
// The same buffer argument, floats further on.
static inline sx_arg offa(sx_arg a, size_t floats) {
    a.offset += floats * 4;
    return a;
}
#define BYTES(x) gpu_bytes_arg(&(x), sizeof(x))
#define TARG(t, sh) gpu_tensor_arg(m->g, &(t), (sh))

// ---- environment switches and profiling ----------------------------------------------------

// Encode the verification chunk during the last draft step (SAETTA_PREP_VERIFY=0: after it).
static bool prep_verify(void) {
    static int v = -1;
    if (v < 0) {
        const char *x = getenv("SAETTA_PREP_VERIFY");
        v = x ? x[0] != '0' : 1;
    }
    return v;
}

// SAETTA_SKIP=attn,gdn,experts,dense,ix (measurement only, wrong output): leaves those prefill
// parts out, to read their real cost from the time saved (profile mode distorts it).
static struct {
    bool attn, gdn, experts, dense, ix, read;
} skip;
static void skip_init(void) {
    if (skip.read) return;
    skip.read = true;
    const char *v = getenv("SAETTA_SKIP");
    if (!v) return;
    skip.attn = strstr(v, "attn") != NULL;
    skip.gdn = strstr(v, "gdn") != NULL;
    skip.experts = strstr(v, "experts") != NULL;
    skip.dense = strstr(v, "dense") != NULL;
    skip.ix = strstr(v, "ix") != NULL;
}

#define NGEN 14
static const char *const gen_names[NGEN] = {"mv",   "expert", "multi", "gate_up",     "gate_up_sh", "down_sh", "up_mix",
                                            "rows1", "rows",  "qmm",   "qmm_grouped", "dq",         "qmmh",    "qmmh_g"};
// Field index of p in Gen, or -1.
static int gen_index(const Gen *g, const sx_pipeline *p) {
    if (g->mv == p) return 0;
    if (g->expert == p) return 1;
    for (int i = 0; i < 3; i++)
        if (g->multi[i] == p) return 2;
    if (g->gate_up == p) return 3;
    if (g->gate_up_sh == p) return 4;
    if (g->down_sh == p) return 5;
    if (g->up_mix == p) return 6;
    for (int i = 0; i <= SMALL; i++)
        if (g->rows1[i] == p) return 7;
    for (int i = 0; i <= SMALL; i++)
        if (g->rows[i] == p) return 8;
    if (g->qmm == p) return 9;
    if (g->qmm_grouped == p) return 10;
    if (g->dq == p) return 11;
    if (g->qmmh == p) return 12;
    if (g->qmmh_g == p) return 13;
    return -1;
}

static struct {
    bool on;
    double secs[NPIPES];
    double gen_secs[NGEN];
    uint64_t gen_count[NGEN];
    double other_secs;
    uint64_t other_count;
    // Weight bytes the next dispatch reads (set by the dense matvec paths), per pipeline.
    uint64_t pend;
    uint64_t bytes_of[NPIPES];
    uint64_t gen_bytes[NGEN];
    uint64_t count[NPIPES];
    double mpp_secs[3];
    uint64_t mpp_count[3];
    uint64_t other_bytes;
} prof;

// k experts' slices of a stacked expert linear (counted as distinct; decode reads each once).
static uint64_t prof_expert_bytes(const Linear *l, size_t k) {
    uint64_t all = l->weight.nbytes + l->scales.nbytes + l->biases.nbytes;
    return all / l->weight.shape[0] * k;
}
static void prof_weight_bytes(const Linear *const *lins, size_t n) {
    if (!prof.on) return;
    uint64_t b = 0;
    for (size_t i = 0; i < n; i++)
        b += lins[i]->weight.nbytes + (lins[i]->bits == 16 ? 0 : lins[i]->scales.nbytes + lins[i]->biases.nbytes);
    prof.pend = b;
}
#define PROF_WB(...) prof_weight_bytes((const Linear *const[]){__VA_ARGS__}, COUNT(((const Linear *const[]){__VA_ARGS__})))

void model_prof_init(void) { prof.on = getenv("SAETTA_PROFILE") != NULL; }

void model_prof_reset(void) {
    memset(prof.secs, 0, sizeof prof.secs);
    memset(prof.count, 0, sizeof prof.count);
    memset(prof.gen_secs, 0, sizeof prof.gen_secs);
    memset(prof.gen_count, 0, sizeof prof.gen_count);
    memset(prof.bytes_of, 0, sizeof prof.bytes_of);
    memset(prof.gen_bytes, 0, sizeof prof.gen_bytes);
    memset(prof.mpp_secs, 0, sizeof prof.mpp_secs);
    memset(prof.mpp_count, 0, sizeof prof.mpp_count);
    prof.other_secs = 0;
    prof.other_count = 0;
    prof.other_bytes = 0;
}

static void prof_add(Model *m, sx_pipeline *p, double s) {
    if (m->mpp) {
        const Mpp *mp = m->mpp;
        int k = (p == mp->dense || p == mp->mmh) ? 0 : p == mp->grouped ? 1 : (p == mp->to_half || p == mp->dq) ? 2 : -1;
        if (k >= 0) {
            prof.mpp_secs[k] += s;
            prof.mpp_count[k] += 1;
            return;
        }
    }
    if (m->gen) {
        int i = gen_index(m->gen, p);
        if (i >= 0) {
            prof.gen_secs[i] += s;
            prof.gen_count[i] += 1;
            prof.gen_bytes[i] += prof.pend;
            prof.pend = 0;
            return;
        }
    }
    sx_pipeline **pp = (sx_pipeline **)&m->p;
    for (size_t i = 0; i < NPIPES; i++)
        if (pp[i] == p) {
            prof.secs[i] += s;
            prof.count[i] += 1;
            prof.bytes_of[i] += prof.pend;
            prof.pend = 0;
            return;
        }
    prof.other_bytes += prof.pend;
    prof.pend = 0;
    prof.other_secs += s;
    prof.other_count += 1;
}

static const char *prof_gbs(char *buf, size_t cap, uint64_t b, double t) {
    if (b == 0 || t == 0) return "";
    char v[64];
    fmt_fixed(v, sizeof v, (double)b / t / 1e9, 1);
    snprintf(buf, cap, "  weights %6s GB/s", v);
    return buf;
}

void model_prof_report(size_t tokens) {
    if (!prof.on) return;
    double total = prof.mpp_secs[0] + prof.mpp_secs[1] + prof.mpp_secs[2] + prof.other_secs;
    for (size_t i = 0; i < NPIPES; i++) total += prof.secs[i];
    for (size_t i = 0; i < NGEN; i++) total += prof.gen_secs[i];
    char a[64], gb[96];
    fmt_fixed(a, sizeof a, total * 1e3 / (double)tokens, 2);
    dbg("profile: GPU time per token by kernel (%zu tokens), total %s ms/token\n", tokens, a);
    static const char *const mpp_names[3] = {"mpp dense", "mpp grouped", "to_half"};
    for (int i = 0; i < 3; i++)
        if (prof.mpp_count[i] > 0) {
            fmt_fixed(a, sizeof a, prof.mpp_secs[i] * 1e3 / (double)tokens, 3);
            dbg("  %14s %8s ms/token  %5llu dispatches/token\n", mpp_names[i], a,
                (unsigned long long)(prof.mpp_count[i] / tokens));
        }
    for (size_t i = 0; i < NPIPES; i++)
        if (prof.count[i] > 0) {
            fmt_fixed(a, sizeof a, prof.secs[i] * 1e3 / (double)tokens, 3);
            dbg("  %14s %8s ms/token  %5llu dispatches/token%s\n", pipe_names[i], a,
                (unsigned long long)(prof.count[i] / tokens), prof_gbs(gb, sizeof gb, prof.bytes_of[i], prof.secs[i]));
        }
    for (size_t i = 0; i < NGEN; i++)
        if (prof.gen_count[i] > 0) {
            fmt_fixed(a, sizeof a, prof.gen_secs[i] * 1e3 / (double)tokens, 3);
            dbg("  gen_%10s %8s ms/token  %5llu dispatches/token%s\n", gen_names[i], a,
                (unsigned long long)(prof.gen_count[i] / tokens),
                prof_gbs(gb, sizeof gb, prof.gen_bytes[i], prof.gen_secs[i]));
        }
    if (prof.other_count > 0) {
        fmt_fixed(a, sizeof a, prof.other_secs * 1e3 / (double)tokens, 3);
        dbg("  other          %8s ms/token  %5llu dispatches/token%s\n", a,
            (unsigned long long)(prof.other_count / tokens), prof_gbs(gb, sizeof gb, prof.other_bytes, prof.other_secs));
    }
}

// SAETTA_TIMING=1: per decode step, host time before the commit (embedding, n-gram rows,
// encoding) and time waiting for the GPU, averaged and printed every 100 steps.
static struct {
    int on; // -1 unread
    uint64_t encoded, host, wait, n;
} timing = {.on = -1};

static uint64_t now_mono_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void step_timing(uint64_t t0) {
    if (timing.on < 0) timing.on = getenv("SAETTA_TIMING") != NULL;
    if (!timing.on) return;
    uint64_t t1 = now_mono_ns();
    timing.host += timing.encoded - t0;
    timing.wait += t1 - timing.encoded;
    timing.n += 1;
    if (timing.n % 100 == 0) {
        char a[64], b[64];
        fmt_fixed(a, sizeof a, (double)timing.host / 1e6 / (double)timing.n, 3);
        fmt_fixed(b, sizeof b, (double)timing.wait / 1e6 / (double)timing.n, 3);
        dbg("step timing: host (embed, n-gram, encode) %s ms, GPU wait %s ms per step\n", a, b);
    }
}

// Chunks up to this many tokens run the GDN projections as one multi-matvec with a token
// grid dimension (SAETTA_GDN_MULTI). Default 1: a 1-token chunk 13.51 -> 13.23 ms, but at 2+
// tokens re-reading the 13 MB qkv matrix per token costs more (T=5 33.6 -> 34.8 ms).
static size_t gdn_multi_max(void) {
    const char *v = getenv("SAETTA_GDN_MULTI");
    if (!v) return 1;
    uint64_t n;
    return parse_u64(str(v), &n) ? 0 : (size_t)n;
}

// Decode folds the shared expert into the routed launches when both are 4-bit group 32 with
// the same widths (SAETTA_FOLD_SHARED=0 keeps separate launches).
static bool fold_shared(const Moe *E) {
    const char *v = getenv("SAETTA_FOLD_SHARED");
    if (v && v[0] == '0') return false;
    // the shapes must match; the 4/32 kernels need every matrix at 4/32, the generic ones one
    // (bits, group) for routed gate and up and one for shared gate and up
    bool shapes = E->sg.out_features == E->eg.out_features && E->sd.in_features == E->ed.in_features &&
                  E->sd.out_features == E->ed.out_features;
    return shapes && same_quant(&E->sg, &E->su) && same_quant(&E->eg, &E->eu) && E->sg.bits != 16;
}

static bool small_fused(void) {
    const char *v = getenv("SAETTA_SMALL_FUSED");
    if (!v) return true;
    return v[0] != '0';
}

// ---- encoder --------------------------------------------------------------------------------

typedef struct {
    uint32_t in, nmat, act;
    float act_scale;
    uint32_t rows[4], tg0[5], shift[12];
} MvP;
static_assert(sizeof(MvP) == offsetof(MvGP, bits));

typedef struct {
    uint32_t nq, nkv, hd, half_rot, ixd, ix_off;
    float eps;
} AttnPrepP;
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

static void enc_d(Enc *e, sx_pipeline *p, const sx_arg *args, size_t nargs, size_t gx, size_t gy, size_t threads) {
    const Mpp *mp = e->m->mpp;
    if (mp) {
        const Gen *gn = e->m->gen;
        bool gk = gn ? (p == gn->dq || p == gn->qmmh || p == gn->qmmh_g) : false;
        if (p != mp->dense && p != mp->grouped && p != mp->to_half && p != mp->dq && p != mp->mmh && p != mp->dq_bf16 &&
            !gk)
            e->xh_set = false;
    } else
        e->xh_set = false;
    sx_cmd_dispatch(e->cmd, p, args, nargs, gx, gy, threads);
    if (prof.on) {
        double secs = 0;
        char err[256];
        (void)sx_cmd_finish_timed(e->cmd, &secs, err, sizeof err);
        e->cmd = sx_cmd_begin(e->m->g->dev);
        prof_add(e->m, p, secs);
    }
}
static void enc_d3(Enc *e, sx_pipeline *p, const sx_arg *args, size_t nargs, size_t gx, size_t gy, size_t gz,
                   size_t threads) {
    e->xh_set = false;
    sx_cmd_dispatch3(e->cmd, p, args, nargs, gx, gy, gz, threads);
    if (prof.on) {
        double secs = 0;
        char err[256];
        (void)sx_cmd_finish_timed(e->cmd, &secs, err, sizeof err);
        e->cmd = sx_cmd_begin(e->m->g->dev);
        prof_add(e->m, p, secs);
    }
}
static void enc_d1(Enc *e, sx_pipeline *p, const sx_arg *args, size_t nargs, size_t n) {
    enc_d(e, p, args, nargs, (n + 255) / 256, 1, 256);
}
// Dispatch helpers over an args array a (its count from the array).
#define D(e, p, a, gx, gy, th) enc_d((e), (p), (a), COUNT(a), (gx), (gy), (th))
#define D1(e, p, a, n) enc_d1((e), (p), (a), COUNT(a), (n))

static void enc_mv_bf16(Enc *e, const Linear *l, sx_arg x, sx_arg y, size_t T);

static void enc_mv(Enc *e, const Linear *l, sx_arg x, sx_arg y) {
    if (skip.dense) return;
    Model *m = e->m;
    uint32_t in = (uint32_t)l->in_features;
    uint32_t out = (uint32_t)l->out_features;
    PROF_WB(l);
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
    // lanes per row by shape (mv-bench): 256 for few rows of long
    // input, 8 for short rows, else one simdgroup per row
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

// The KV cache at element index elems (f32 or f16 by session).
static sx_arg kv_arg(Session *s, size_t elems) { return (sx_arg){s->kv, elems * session_kv_elem(s), NULL, 0}; }

static void enc_copy_rows(Enc *e, sx_arg src, sx_arg dst, size_t n, size_t rows, size_t ss, size_t ds);

// rows x n floats from src into the KV cache at element dst_elems (row stride n).
static void enc_kv_store(Enc *e, Session *s, sx_arg src, size_t dst_elems, size_t n, size_t rows) {
    if (s->kv_half) {
        uint32_t p[4] = {(uint32_t)n, (uint32_t)rows, (uint32_t)n, (uint32_t)n};
        sx_arg a[] = {src, kv_arg(s, dst_elems), gpu_bytes_arg(p, 16)};
        D1(e, e->m->p.copy_rows_h, a, n * rows);
    } else
        enc_copy_rows(e, src, kv_arg(s, dst_elems), n, rows, n, n);
}

static void enc_mv_bf16(Enc *e, const Linear *l, sx_arg x, sx_arg y, size_t T) {
    Model *m = e->m;
    uint32_t sh = 0;
    uint32_t dd[3] = {(uint32_t)l->in_features, (uint32_t)l->out_features, (uint32_t)T};
    sx_arg a[] = {TARG(l->weight, &sh), x, y, gpu_bytes_arg(dd, 12), BYTES(sh)};
    D(e, m->p.mv_bf16, a, dd[1], T, 32);
}

// Up to 4 linears (4-bit, group 32, or the generic kernels) on the same input in one
// dispatch, lanes lanes per row (8, 32 or 256); act: the input goes through silu(x *
// act_scale) as it is read. Over T input rows (x row stride = in, output k row stride = its rows).
static void enc_mv_multi_t(Enc *e, const Linear *const *lins, size_t nl, sx_arg x, const sx_arg *ys, size_t ny,
                           uint32_t lanes, bool act, float act_scale, size_t T) {
    if (skip.dense) return;
    Model *m = e->m;
    prof_weight_bytes(lins, nl);
    bool gen = false;
    for (size_t i = 0; i < nl; i++) gen = gen || !is4_32(lins[i]);
    MvGP p = {.in = (uint32_t)lins[0]->in_features, .nmat = (uint32_t)nl, .act = act, .act_scale = act_scale,
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
    // the templated kernels read MvP, a prefix of MvGP
    args[17] = gen ? BYTES(p) : gpu_bytes_arg(&p, sizeof(MvP));
    size_t li = lanes == 8 ? 0 : lanes == 32 ? 1 : 2;
    sx_pipeline *pl = gen ? m->gen->multi[li]
                          : (lanes == 8 ? m->p.qmv_multi_l8 : lanes == 32 ? m->p.qmv_multi_l32 : m->p.qmv_multi_l256);
    D(e, pl, args, p.tg0[nl], T, 256);
}
static void enc_mv_multi(Enc *e, const Linear *const *lins, size_t nl, sx_arg x, const sx_arg *ys, size_t ny,
                         uint32_t lanes, bool act, float act_scale) {
    enc_mv_multi_t(e, lins, nl, x, ys, ny, lanes, act, act_scale, 1);
}
#define LINS(...) (const Linear *const[]){__VA_ARGS__}, COUNT(((const Linear *const[]){__VA_ARGS__}))
#define ARGS(...) (const sx_arg[]){__VA_ARGS__}, COUNT(((const sx_arg[]){__VA_ARGS__}))

// h = silu(gate x) * (up x): routed experts idx[slot] (T tokens x TOPK slots) or, with
// routed false, the shared expert (one slot).
static void enc_gate_up(Enc *e, const Linear *gate, const Linear *up, sx_arg x, sx_arg h, bool routed, sx_arg idx,
                        size_t slots) {
    if (skip.experts && routed) return;
    Model *m = e->m;
    if (prof.on) {
        if (routed)
            prof.pend = prof_expert_bytes(gate, slots) + prof_expert_bytes(up, slots);
        else
            PROF_WB(gate, up);
    }
    uint32_t sh[6];
    uint32_t dm[4] = {(uint32_t)gate->in_features, (uint32_t)gate->out_features, (uint32_t)H, routed ? (uint32_t)TOPK : 1};
    uint32_t es[4] = {routed ? (uint32_t)(gate->weight.nbytes / gate->weight.shape[0]) : 0,
                      routed ? (uint32_t)(gate->scales.nbytes / gate->scales.shape[0]) : 0, routed, 0};
    sx_arg args[13] = {TARG(gate->weight, &sh[0]), TARG(gate->scales, &sh[1]), TARG(gate->biases, &sh[2]),
                       TARG(up->weight, &sh[3]),   TARG(up->scales, &sh[4]),   TARG(up->biases, &sh[5]),
                       x,                          h,                          idx,
                       gpu_bytes_arg(dm, 16),      gpu_bytes_arg(es, 16),      gpu_bytes_arg(sh, 24)};
    if (!is4_32(gate) || !is4_32(up)) {
        // one (bits, group) for both in qmv_gate_up_gen
        uint32_t q[2] = {gate->bits, gate->group_size};
        args[12] = BYTES(q);
        enc_d(e, m->gen->gate_up, args, 13, dm[1], slots, 32);
        return;
    }
    enc_d(e, m->p.qmv_gate_up, args, 12, dm[1], slots, 32);
}

static void enc_gate_up_sh(Enc *e, const Moe *E, sx_arg x, sx_arg h) {
    if (skip.experts) return;
    Model *m = e->m;
    if (prof.on) {
        PROF_WB(&E->sg, &E->su);
        prof.pend += prof_expert_bytes(&E->eg, TOPK) + prof_expert_bytes(&E->eu, TOPK);
    }
    uint32_t sh[6], sh2[6];
    uint32_t dm[4] = {(uint32_t)E->eg.in_features, (uint32_t)E->eg.out_features, (uint32_t)TOPK, 0};
    uint32_t es[4] = {(uint32_t)(E->eg.weight.nbytes / E->eg.weight.shape[0]),
                      (uint32_t)(E->eg.scales.nbytes / E->eg.scales.shape[0]), 0, 0};
    sx_arg args[20] = {
        TARG(E->eg.weight, &sh[0]),  TARG(E->eg.scales, &sh[1]),  TARG(E->eg.biases, &sh[2]),  TARG(E->eu.weight, &sh[3]),
        TARG(E->eu.scales, &sh[4]),  TARG(E->eu.biases, &sh[5]),  x,                           h,
        sarg(m, S_topi),             gpu_bytes_arg(dm, 16),       gpu_bytes_arg(es, 16),       gpu_bytes_arg(sh, 24),
        TARG(E->sg.weight, &sh2[0]), TARG(E->sg.scales, &sh2[1]), TARG(E->sg.biases, &sh2[2]), TARG(E->su.weight, &sh2[3]),
        TARG(E->su.scales, &sh2[4]), TARG(E->su.biases, &sh2[5]), gpu_bytes_arg(sh2, 24),
    };
    if (!(is4_32(&E->eg) && is4_32(&E->sg))) {
        uint32_t q[4] = {E->eg.bits, E->eg.group_size, E->sg.bits, E->sg.group_size};
        args[19] = gpu_bytes_arg(q, 16);
        enc_d(e, m->gen->gate_up_sh, args, 20, dm[1], TOPK + 1, 32);
        return;
    }
    enc_d(e, m->p.qmv_gate_up_sh, args, 19, dm[1], TOPK + 1, 32);
}

static void enc_down_sh(Enc *e, const Moe *E, sx_arg x, sx_arg y) {
    if (skip.experts) return;
    Model *m = e->m;
    if (prof.on) {
        PROF_WB(&E->sd);
        prof.pend += prof_expert_bytes(&E->ed, TOPK);
    }
    uint32_t sh[3], sh2[3];
    uint32_t dm[4] = {(uint32_t)E->ed.in_features, (uint32_t)E->ed.out_features, (uint32_t)TOPK, 0};
    uint32_t es[2] = {(uint32_t)(E->ed.weight.nbytes / E->ed.weight.shape[0]),
                      (uint32_t)(E->ed.scales.nbytes / E->ed.scales.shape[0])};
    sx_arg args[14] = {
        TARG(E->ed.weight, &sh[0]), TARG(E->ed.scales, &sh[1]),  TARG(E->ed.biases, &sh[2]),  x,
        y,                          gpu_bytes_arg(dm, 16),       gpu_bytes_arg(sh, 12),       sarg(m, S_topi),
        gpu_bytes_arg(es, 8),       TARG(E->sd.weight, &sh2[0]), TARG(E->sd.scales, &sh2[1]), TARG(E->sd.biases, &sh2[2]),
        gpu_bytes_arg(sh2, 12),
    };
    if (!(is4_32(&E->ed) && is4_32(&E->sd))) {
        uint32_t q[4] = {E->ed.bits, E->ed.group_size, E->sd.bits, E->sd.group_size};
        args[13] = gpu_bytes_arg(q, 16);
        enc_d(e, m->gen->down_sh, args, 14, dm[1], TOPK + 1, 32);
        return;
    }
    enc_d(e, m->p.qmv_down_sh, args, 13, dm[1], TOPK + 1, 32);
}

// Routed experts idx[0..TOPK] (T tokens x TOPK slots) of a rank-3 linear; slot s reads x +
// (s / div) * x_stride.
static void enc_mv_expert_idx(Enc *e, const Linear *l, sx_arg x, uint32_t x_stride, uint32_t div, sx_arg y, size_t T,
                              sx_arg idx) {
    if (skip.experts) return;
    Model *m = e->m;
    uint32_t in = (uint32_t)l->in_features;
    uint32_t out = (uint32_t)l->out_features;
    uint32_t sh[3];
    if (prof.on) prof.pend = prof_expert_bytes(l, TOPK * T);
    uint32_t stride[4] = {(uint32_t)(l->weight.nbytes / l->weight.shape[0]),
                          (uint32_t)(l->scales.nbytes / l->scales.shape[0]), x_stride, div};
    sx_arg args[11] = {TARG(l->weight, &sh[0]), TARG(l->scales, &sh[1]), TARG(l->biases, &sh[2]), x, y, BYTES(in),
                       BYTES(out), gpu_bytes_arg(sh, 12), idx, gpu_bytes_arg(stride, 16)};
    if (!is4_32(l)) {
        uint32_t q[2] = {l->bits, l->group_size};
        args[10] = BYTES(q);
        enc_d(e, m->gen->expert, args, 11, out, TOPK * T, 32);
        return;
    }
    enc_d(e, m->p.qe4, args, 10, out, TOPK * T, 32);
}
static void enc_mv_expert_t(Enc *e, const Linear *l, sx_arg x, uint32_t x_stride, uint32_t div, sx_arg y, size_t T) {
    enc_mv_expert_idx(e, l, x, x_stride, div, y, T, sarg(e->m, S_topi));
}
static void enc_mv_expert(Enc *e, const Linear *l, sx_arg x, uint32_t x_stride, sx_arg y) {
    enc_mv_expert_t(e, l, x, x_stride, 1, y, 1);
}
static void enc_mv_expert_ti(Enc *e, const Linear *l, sx_arg x, uint32_t x_stride, uint32_t div, sx_arg y, size_t T) {
    enc_mv_expert_idx(e, l, x, x_stride, div, y, T, barg(e->m->bscratch, B_topi));
}

// As norm; the weight row is (group % wmod) when wmod > 0 (one row per stream, T tokens).
static void enc_normw(Enc *e, sx_arg x, Vec w, sx_arg out, size_t groups, size_t n, size_t xs, size_t os, size_t ws,
                      bool plus1, size_t wmod) {
    NormP p = {(uint32_t)n, (uint32_t)xs, (uint32_t)os, (uint32_t)ws, plus1, (uint32_t)wmod, EPS};
    sx_arg a[] = {x, barg(e->m->params.buf, w.off), out, BYTES(p)};
    D(e, e->m->p.rmsnorm, a, groups, 1, 256);
}
static void enc_norm(Enc *e, sx_arg x, Vec w, sx_arg out, size_t groups, size_t n, size_t xs, size_t os, size_t ws,
                     bool plus1) {
    enc_normw(e, x, w, out, groups, n, xs, os, ws, plus1, 0);
}
static void enc_copy(Enc *e, sx_arg src, sx_arg dst, size_t n) {
    uint32_t nn = (uint32_t)n;
    sx_arg a[] = {src, dst, BYTES(nn)};
    D1(e, e->m->p.copy_f32, a, n);
}
// T tokens, heads per token, angles from the table cs ([T][ROT/2] pairs).
static void enc_rope_t(Enc *e, sx_arg x, size_t heads, size_t stride, size_t T, sx_arg cs) {
    uint32_t p[3] = {(uint32_t)stride, (uint32_t)(ROT / 2), (uint32_t)heads};
    sx_arg a[] = {x, gpu_bytes_arg(p, 12), cs};
    D(e, e->m->p.rope, a, heads * T, 1, 32);
}
static void enc_rope(Enc *e, Session *s, sx_arg x, size_t heads, size_t stride, size_t pos) {
    enc_rope_t(e, x, heads, stride, 1, barg(s->rope, pos * ROT));
}
// Pooled, normed, roped keys of blocks b0 .. b0+nb-1 of attention layer A.
static void enc_ix_blocks(Enc *e, const Attn *A, Session *s, size_t b0, size_t nb) {
    if (nb == 0) return;
    uint32_t p[4] = {(uint32_t)b0, (uint32_t)IXD, (uint32_t)BLOCK, (uint32_t)(ROT / 2)};
    float eps = EPS;
    sx_arg a[] = {barg(s->ixk, A->slot * s->cap * IXD), barg(e->m->params.buf, A->ix_kn.off), barg(s->rope, 0),
                  barg(s->bk, A->slot * (s->cap / BLOCK) * IXD), gpu_bytes_arg(p, 16), BYTES(eps)};
    D(e, e->m->p.ix_block, a, nb, 1, IXD);
}
// QSA selection for a query whose indexer part (IXH x IXD, before norm) is at ixq, at
// position pos with n visible tokens: token list in sel, its length in cnt.
static void enc_ix_select(Enc *e, const Attn *A, Session *s, sx_arg ixq, size_t pos, size_t n, sx_arg sel, sx_arg cnt) {
    Model *m = e->m;
    enc_norm(e, ixq, A->ix_qn, sarg(m, S_iq), IXH, IXD, IXD, IXD, 0, true);
    enc_rope(e, s, sarg(m, S_iq), IXH, IXD, pos);
    size_t nb = n / BLOCK;
    uint32_t sp[3] = {(uint32_t)nb, (uint32_t)IXH, (uint32_t)IXD};
    bool fast = getenv("SAETTA_IX_OLD") == NULL && IXD == 128 && nb <= 1024 * 64;
    sx_arg a[] = {sarg(m, S_iq), barg(s->bk, A->slot * (s->cap / BLOCK) * IXD), barg(s->scores, 0), gpu_bytes_arg(sp, 12)};
    if (fast)
        D(e, m->p.ix_score_sg, a, nb, 1, 32);
    else
        D1(e, m->p.ix_score, a, nb);
    uint32_t pp[4] = {(uint32_t)nb, (uint32_t)TOPB, (uint32_t)BLOCK, (uint32_t)n};
    sx_arg b[] = {barg(s->scores, 0), sel, cnt, gpu_bytes_arg(pp, 16)};
    D(e, fast ? m->p.ix_select_rx : m->p.ix_select, b, 1, 1, 1024);
}
static void enc_copy_rows(Enc *e, sx_arg src, sx_arg dst, size_t n, size_t rows, size_t ss, size_t ds) {
    uint32_t p[4] = {(uint32_t)n, (uint32_t)rows, (uint32_t)ss, (uint32_t)ds};
    sx_arg a[] = {src, dst, gpu_bytes_arg(p, 16)};
    D1(e, e->m->p.copy_rows, a, n * rows);
}
// Attention for T queries at positions pos0.. over the KV cache, into att; per-token
// selection lists (when past the budget) at sel + t*(MAXSEL+13), counts at cnt.
static void enc_attn_batch(Enc *e, Session *s, sx_arg q, sx_arg kc, sx_arg vc, sx_arg sel, sx_arg cnt, sx_arg gate,
                           sx_arg att, size_t pos0, size_t T) {
    Model *m = e->m;
    AttnSP p = {(uint32_t)pos0, (uint32_t)TOPB, (uint32_t)BLOCK, (uint32_t)HD, (uint32_t)(NKV * HD),
                (uint32_t)(NQ / NKV), (uint32_t)(2 * HD), (uint32_t)NQ, (uint32_t)NSPLIT, (uint32_t)(MAXSEL + 13),
                1.0f / sqrtf((float)HD)};
    sx_arg part = barg(m->bscratch, B_part);
    static_assert(NQ / NKV == 12 && HD == 256); // attn_split_g shape
    for (size_t q0 = 0; q0 < T; q0 += AB) {
        size_t tb = MIN((size_t)AB, T - q0);
        AttnSP pq = p;
        pq.pos0 = (uint32_t)(pos0 + q0);
        sx_arg qa = offa(q, q0 * NQ * HD);
        sx_arg sa = offa(sel, q0 * (MAXSEL + 13));
        sx_arg ca = offa(cnt, q0);
        sx_arg a[] = {qa, kc, vc, sa, ca, part, BYTES(pq)};
        enc_d3(e, s->kv_half ? m->p.attn_split_gh : m->p.attn_split_g, a, COUNT(a), NKV, NSPLIT, tb, 256);
        sx_arg b[] = {part, offa(gate, q0 * NQ * 2 * HD), offa(att, q0 * NQ * HD), BYTES(pq)};
        D(e, m->p.attn_merge, b, NQ, tb, 256);
    }
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
// An accelerator matmul writing n floats at y clears the cache if that overlaps its source.
static void enc_out_writes_range(Enc *e, sx_arg y, size_t n) {
    if (e->xh_set && e->xh_buf == y.buffer && y.offset < e->xh_off + e->xh_n * 4 && e->xh_off < y.offset + n * 4)
        e->xh_set = false;
}

// Grouped expert matmul over the tile table of moe_group (T tokens, TOPK slots each).
static void enc_mm_grouped(Enc *e, const Linear *l, sx_arg x, sx_arg y, size_t T) {
    if (skip.experts) return;
    Model *m = e->m;
    sx_buffer *b = m->bscratch;
    if (m->fast_prefill) {
        enc_to_half(e, x, T * TOPK * (size_t)l->in_features);
        uint32_t dims[3] = {(uint32_t)l->in_features, (uint32_t)l->out_features, (uint32_t)(T * TOPK)};
        uint32_t stride[2] = {(uint32_t)(l->weight.nbytes / l->weight.shape[0]),
                              (uint32_t)(l->scales.nbytes / l->scales.shape[0])};
        uint32_t sh[3];
        sx_arg args[11] = {TARG(l->weight, &sh[0]), TARG(l->scales, &sh[1]), TARG(l->biases, &sh[2]), barg(b, B_xh), y,
                           gpu_bytes_arg(dims, 12), gpu_bytes_arg(sh, 12), barg(b, B_gtiles), barg(b, B_gnt),
                           gpu_bytes_arg(stride, 8)};
        if (!is4_32(l)) {
            uint32_t q[2] = {l->bits, l->group_size};
            args[10] = BYTES(q);
            enc_d(e, m->gen->qmmh_g, args, 11, dims[1] / 64, (T * TOPK + 63) / 64 + NE, 128);
        } else
            enc_d(e, m->mpp->grouped, args, 10, dims[1] / 64, (T * TOPK + 63) / 64 + NE, 128);
        enc_out_writes_range(e, y, T * TOPK * dims[1]);
        return;
    }
    uint32_t dims[3] = {(uint32_t)l->in_features, (uint32_t)l->out_features, 0};
    uint32_t stride[2] = {(uint32_t)(l->weight.nbytes / l->weight.shape[0]), (uint32_t)(l->scales.nbytes / l->scales.shape[0])};
    uint32_t sh[3];
    sx_arg args[11] = {TARG(l->weight, &sh[0]), TARG(l->scales, &sh[1]), TARG(l->biases, &sh[2]), x, y,
                       gpu_bytes_arg(dims, 12), gpu_bytes_arg(sh, 12), barg(b, B_gtiles), barg(b, B_gnt),
                       gpu_bytes_arg(stride, 8)};
    if (!is4_32(l)) {
        uint32_t q[2] = {l->bits, l->group_size};
        args[10] = BYTES(q);
        enc_d(e, m->gen->qmm_grouped, args, 11, (dims[1] + 31) / 32, (T * TOPK + 31) / 32 + NE, 128);
        return;
    }
    enc_d(e, m->p.mmg4, args, 10, (dims[1] + 31) / 32, (T * TOPK + 31) / 32 + NE, 128);
}

// Y[T][out] = X[T][in] W^T
static void enc_mm(Enc *e, const Linear *l, sx_arg x, sx_arg y, size_t T) {
    if (skip.dense) return;
    Model *m = e->m;
    if (T <= SMALL) PROF_WB(l);
    if (l->bits == 16) {
        const Mpp *mp = m->mpp;
        if (T > SMALL && m->fast_prefill && mp != NULL && mp->mmh != NULL && mp->dq_bf16 != NULL) {
            // prompt chunks: BF16 to f16 once, then the f16 GEMM
            enc_to_half(e, x, T * (size_t)l->in_features);
            uint32_t dims[3] = {(uint32_t)l->in_features, (uint32_t)l->out_features, (uint32_t)T};
            uint32_t sh = 0;
            uint32_t d2[2] = {dims[0], dims[1]};
            sx_arg a[] = {TARG(l->weight, &sh), barg(m->bscratch, B_wh), gpu_bytes_arg(d2, 8), BYTES(sh)};
            D1(e, mp->dq_bf16, a, (size_t)dims[0] * dims[1]);
            sx_arg b[] = {barg(m->bscratch, B_xh), barg(m->bscratch, B_wh), y, gpu_bytes_arg(dims, 12)};
            D(e, mp->mmh, b, (dims[1] + 31) / 32, (T + 63) / 64, 32);
            enc_out_writes_range(e, y, T * dims[1]);
            return;
        }
        enc_mv_bf16(e, l, x, y, T); // small chunks and the exact path
        return;
    }
    if (T <= SMALL) {
        // verification-sized chunk: weights read once, applied to each row (decode-like)
        uint32_t dims[3] = {(uint32_t)l->in_features, (uint32_t)l->out_features, (uint32_t)T};
        uint32_t sh[3];
        sx_arg args[8] = {TARG(l->weight, &sh[0]), TARG(l->scales, &sh[1]), TARG(l->biases, &sh[2]), x, y,
                          gpu_bytes_arg(dims, 12), gpu_bytes_arg(sh, 12)};
        size_t bi = l->bits == 8 ? 1 : 0;
        if (!fast_q(l)) {
            uint32_t q[2] = {l->bits, l->group_size};
            args[7] = BYTES(q);
            if (T >= 3 && dims[1] >= 2048)
                enc_d(e, m->gen->rows[T], args, 8, (dims[1] + 7) / 8, 1, 64);
            else
                enc_d(e, m->gen->rows1[T], args, 8, dims[1], 1, 32);
            return;
        }
        if (T >= 3 && dims[1] >= 2048)
            enc_d(e, m->rows_p[bi][T], args, 7, (dims[1] + 7) / 8, 1, 64); // 2 simdgroups x 4 rows (qmatvec.metal ROWS_R)
        else
            enc_d(e, m->rows1_p[bi][T], args, 7, dims[1], 1, 32);
        return;
    }
    // accelerator path for every matrix except the router-sized 8-bit ones (the full pack's
    // routers and shared-expert gates stay f32); the Coder's 8-bit dense layers go here too
    bool router_like = l->bits == 8 && l->out_features <= 512;
    if (m->fast_prefill && !router_like) {
        enc_to_half(e, x, T * (size_t)l->in_features);
        uint32_t dims[3] = {(uint32_t)l->in_features, (uint32_t)l->out_features, (uint32_t)T};
        uint32_t sh[3];
        sx_arg args[8] = {TARG(l->weight, &sh[0]), TARG(l->scales, &sh[1]), TARG(l->biases, &sh[2]),
                          barg(m->bscratch, B_xh), y, gpu_bytes_arg(dims, 12), gpu_bytes_arg(sh, 12)};
        const Mpp *mp = m->mpp;
        uint32_t q[2] = {l->bits, l->group_size};
        bool gen = !is4_32(l);
        if (mp->mmh && (size_t)dims[0] * dims[1] <= 12288 * 2560 && (!gen || m->gen->dq != NULL)) {
            // unpack W to f16 in B.wh (one thread per 8 weights), then the f16 GEMM
            uint32_t dq[2] = {dims[0], dims[1]};
            if (gen) {
                sx_arg a[] = {args[0], args[1], args[2], barg(m->bscratch, B_wh), gpu_bytes_arg(dq, 8), args[6], BYTES(q)};
                D1(e, m->gen->dq, a, (size_t)dims[1] * dims[0] / 8);
            } else {
                sx_arg a[] = {args[0], args[1], args[2], barg(m->bscratch, B_wh), gpu_bytes_arg(dq, 8), args[6]};
                D1(e, mp->dq, a, (size_t)dims[1] * dims[0] / 8);
            }
            sx_arg b[] = {barg(m->bscratch, B_xh), barg(m->bscratch, B_wh), y, gpu_bytes_arg(dims, 12)};
            D(e, mp->mmh, b, (dims[1] + 31) / 32, (T + 63) / 64, 32);
            enc_out_writes_range(e, y, T * dims[1]);
            return;
        }
        if (gen) {
            args[7] = BYTES(q);
            enc_d(e, m->gen->qmmh, args, 8, (dims[1] + 63) / 64, (T + 63) / 64, 128);
        } else
            enc_d(e, mp->dense, args, 7, (dims[1] + 63) / 64, (T + 63) / 64, 128);
        enc_out_writes_range(e, y, T * dims[1]);
        return;
    }
    if (!fast_q(l)) {
        // exact path, mixed quantization
        uint32_t dims[3] = {(uint32_t)l->in_features, (uint32_t)l->out_features, (uint32_t)T};
        uint32_t q[2] = {l->bits, l->group_size};
        uint32_t shg[3];
        sx_arg ga[] = {TARG(l->weight, &shg[0]), TARG(l->scales, &shg[1]), TARG(l->biases, &shg[2]), x, y,
                       gpu_bytes_arg(dims, 12), gpu_bytes_arg(shg, 12), BYTES(q)};
        D(e, m->gen->qmm, ga, (dims[1] + 31) / 32, (T + 31) / 32, 128);
        return;
    }
    sx_pipeline *p = l->bits == 8 ? m->p.mm8 : m->p.mm4;
    uint32_t dims[3] = {(uint32_t)l->in_features, (uint32_t)l->out_features, (uint32_t)T};
    uint32_t sh[3];
    sx_arg args[] = {TARG(l->weight, &sh[0]), TARG(l->scales, &sh[1]), TARG(l->biases, &sh[2]), x, y,
                     gpu_bytes_arg(dims, 12), gpu_bytes_arg(sh, 12)};
    D(e, p, args, (dims[1] + 31) / 32, (T + 31) / 32, 128);
}

// ---- prefill (T tokens) ----
static void enc_residual_t(Enc *e, const Residual *R, size_t T) {
    sx_buffer *b = e->m->bscratch;
    enc_normw(e, barg(b, B_X), R->norm, barg(b, B_xn), T * HC, H, H, H, H, true, HC);
    enc_mm(e, &R->down, barg(b, B_xn), barg(b, B_low), T);
    // inject reads xn too: right after down, while its half copy is still cached
    if (R->has_inject) enc_mm(e, &R->inject, barg(b, B_xn), barg(b, B_r), T);
    uint32_t n = (uint32_t)(T * LOW);
    float scale = 1.0f / (float)HC;
    sx_arg a[] = {barg(b, B_low), BYTES(n), BYTES(scale)};
    D1(e, e->m->p.silu_scale, a, T * LOW);
    enc_mm(e, &R->up, barg(b, B_low), barg(b, B_mix), T);
    uint32_t p[3] = {(uint32_t)H, (uint32_t)HC, (uint32_t)T};
    sx_arg c[] = {barg(b, B_mix), barg(b, B_xn), barg(b, B_a), gpu_bytes_arg(p, 12)};
    D1(e, e->m->p.hc_mix, c, T * H);
}
static void enc_hc_add_t(Enc *e, size_t T) {
    sx_buffer *b = e->m->bscratch;
    uint32_t p[3] = {(uint32_t)H, (uint32_t)HC, (uint32_t)T};
    sx_arg a[] = {barg(b, B_X), barg(b, B_y), barg(b, B_r), gpu_bytes_arg(p, 12)};
    D1(e, e->m->p.hc_add, a, T * HCW);
}

// Start-state arguments for the GDN layer with index slot and for the PLE history.
static sx_arg src_gdn(const Session *s, size_t slot) {
    size_t n = NV * DK * DV;
    return s->roll != SESSION_NO_ROLL ? barg(s->sp_gdn, s->roll * n_gdn * n + slot * n) : barg(s->gdn_state, slot * n);
}
static sx_arg src_conv(const Session *s, size_t slot) {
    size_t n = (KCONV - 1) * CONV;
    return s->roll != SESSION_NO_ROLL ? barg(s->sp_conv, s->roll * n_gdn * n + slot * n) : barg(s->gdn_conv, slot * n);
}
static sx_arg src_ple(const Session *s) {
    return s->roll != SESSION_NO_ROLL ? barg(s->sp_ple, s->roll * PLE_CONV_STATE * HCW) : barg(s->ple_hist, 0);
}

static void enc_gdn_t(Enc *e, const Gdn *G, Session *s, size_t T) {
    Model *m = e->m;
    sx_buffer *b = m->bscratch;
    if (T <= gdn_multi_max()) {
        // the 4 projections in one dispatch, one row of the grid per token (as decode)
        enc_mv_multi_t(e, LINS(&G->qkv, &G->z, &G->b, &G->a), barg(b, B_a),
                       ARGS(barg(b, B_qkv), barg(b, B_z), barg(b, B_gb), barg(b, B_ga)), 32, false, 1.0f, T);
    } else {
        enc_mm(e, &G->qkv, barg(b, B_a), barg(b, B_qkv), T);
        enc_mm(e, &G->z, barg(b, B_a), barg(b, B_z), T);
        enc_mm(e, &G->b, barg(b, B_a), barg(b, B_gb), T);
        enc_mm(e, &G->a, barg(b, B_a), barg(b, B_ga), T);
    }
    uint32_t conv_slot = (uint32_t)(n_gdn * (KCONV - 1) * CONV);
    uint32_t n[4] = {(uint32_t)CONV, (uint32_t)T, s->spec, conv_slot};
    sx_arg a[] = {barg(b, B_qkv), barg(s->gdn_conv, G->slot * (KCONV - 1) * CONV), barg(m->params.buf, G->conv.off),
                  barg(b, B_conv), gpu_bytes_arg(n, 16), barg(s->sp_conv, G->slot * (KCONV - 1) * CONV), src_conv(s, G->slot)};
    D1(e, m->p.gdn_conv, a, CONV);
    uint32_t qk[4] = {(uint32_t)NK, (uint32_t)DK, (uint32_t)KD, (uint32_t)CONV};
    sx_arg c[] = {barg(b, B_conv), gpu_bytes_arg(qk, 16)};
    D(e, m->p.gdn_qknorm, c, 2 * NK * T, 1, DK);
    GdnP gp = {(uint32_t)NK, (uint32_t)NV, (uint32_t)DK, (uint32_t)DV, (uint32_t)KD, (uint32_t)CONV, (uint32_t)T, EPS,
               s->spec, (uint32_t)(n_gdn * NV * DK * DV)};
    if (!skip.gdn) {
        sx_arg d[] = {barg(b, B_conv),
                      barg(b, B_ga),
                      barg(b, B_gb),
                      barg(m->params.buf, G->a_log.off),
                      barg(m->params.buf, G->dt_bias.off),
                      barg(s->gdn_state, G->slot * NV * DK * DV),
                      barg(b, B_o),
                      BYTES(gp),
                      barg(s->sp_gdn, G->slot * NV * DK * DV),
                      src_gdn(s, G->slot)};
        D(e, m->p.gdn_step, d, DV / 16, NV, 64);
    }
    sx_arg f[] = {barg(b, B_o), barg(b, B_z), barg(m->params.buf, G->norm.off), BYTES(gp)};
    D(e, m->p.gdn_gnorm, f, T * NV, 1, DV);
    enc_mm(e, &G->out, barg(b, B_o), barg(b, B_y), T);
}

static void enc_residual(Enc *e, const Residual *R, size_t out);
static void enc_moe_t(Enc *e, const Moe *E, size_t T);

static void enc_attn_t_at(Enc *e, const Attn *A, Session *s, size_t T, size_t pos0) {
    Model *m = e->m;
    sx_buffer *b = m->bscratch;
    size_t kv_row = NKV * HD;
    size_t kbase = A->slot * 2 * s->cap * kv_row;
    size_t vbase = kbase + s->cap * kv_row;
    enc_mm(e, &A->q, barg(b, B_a), barg(b, B_qg), T);
    enc_mm(e, &A->k, barg(b, B_a), barg(b, B_kx), T);
    enc_mm(e, &A->v, barg(b, B_a), barg(b, B_vx), T);
    enc_mm(e, &A->ix, barg(b, B_a), barg(b, B_ix), T);
    enc_norm(e, barg(b, B_qg), A->q_norm, barg(b, B_qn), T * NQ, HD, 2 * HD, HD, 0, true);
    enc_rope_t(e, barg(b, B_qn), NQ, HD, T, barg(s->rope, pos0 * ROT));
    enc_norm(e, barg(b, B_kx), A->k_norm, barg(b, B_kx), T * NKV, HD, HD, HD, 0, true);
    enc_rope_t(e, barg(b, B_kx), NKV, HD, T, barg(s->rope, pos0 * ROT));
    enc_kv_store(e, s, barg(b, B_kx), kbase + pos0 * kv_row, kv_row, T);
    enc_kv_store(e, s, barg(b, B_vx), vbase + pos0 * kv_row, kv_row, T);
    enc_copy_rows(e, barg(b, B_ix + IXH * IXD), barg(s->ixk, A->slot * s->cap * IXD + pos0 * IXD), IXD, T,
                  (IXH + 1) * IXD, IXD);
    enc_ix_blocks(e, A, s, pos0 / BLOCK, (pos0 + T) / BLOCK - pos0 / BLOCK);
    // QSA selection for the chunk's tokens past the budget: batched scores, then one
    // selection threadgroup per token (session scores buffer holds [T][stride])
    size_t nbs = (pos0 + T) / BLOCK;
    if (nbs > TOPB && !skip.ix) {
        enc_copy_rows(e, barg(b, B_ix), barg(b, B_iqs), IXH * IXD, T, (IXH + 1) * IXD, IXH * IXD);
        enc_norm(e, barg(b, B_iqs), A->ix_qn, barg(b, B_iqs), T * IXH, IXD, IXD, IXD, 0, true);
        enc_rope_t(e, barg(b, B_iqs), IXH, IXD, T, barg(s->rope, pos0 * ROT));
        size_t stride = s->cap / BLOCK;
        for (size_t q0 = 0; q0 < T; q0 += AB) {
            size_t tb = MIN((size_t)AB, T - q0);
            size_t nb_end = (pos0 + q0 + tb) / BLOCK;
            if (nb_end <= TOPB) continue;
            uint32_t sp[3] = {(uint32_t)tb, (uint32_t)nb_end, (uint32_t)stride};
            sx_arg a[] = {barg(b, B_iqs + q0 * IXH * IXD), barg(s->bk, A->slot * (s->cap / BLOCK) * IXD), barg(s->scores, 0),
                          gpu_bytes_arg(sp, 12)};
            D(e, m->p.ix_score_t, a, (nb_end + 31) / 32, (tb + 31) / 32, 128);
            uint32_t pp[4] = {(uint32_t)(pos0 + q0), (uint32_t)TOPB, (uint32_t)BLOCK, (uint32_t)stride};
            uint32_t ms = MAXSEL + 13;
            sx_arg c[] = {barg(s->scores, 0), barg(b, B_sel + q0 * (MAXSEL + 13)), barg(b, B_cnt + q0), gpu_bytes_arg(pp, 16),
                          BYTES(ms)};
            D(e, m->p.ix_select_t, c, tb, 1, 1024);
        }
    }
    sx_pipeline *tr = m->mpp ? m->mpp->qsa : NULL;
    if (skip.attn) {
    } else if (tr != NULL && m->fast_prefill && s->kv_half && T > SMALL) {
        // prompt chunks: attention on the accelerators (qmm_mpp.metal qsa_flash_tr), gate fused
        uint32_t nq = (uint32_t)(T * NQ * HD);
        sx_arg a[] = {barg(b, B_qn), barg(b, B_qh), BYTES(nq)};
        D1(e, m->mpp->to_half, a, T * NQ * HD);
        uint32_t tp[4] = {(uint32_t)pos0, (uint32_t)TOPB, (uint32_t)BLOCK, (uint32_t)(MAXSEL + 13)};
        sx_arg c[] = {barg(b, B_qh), kv_arg(s, kbase), kv_arg(s, vbase), barg(b, B_sel), barg(b, B_cnt), barg(b, B_att),
                      gpu_bytes_arg(tp, 16), barg(b, B_qg)};
        D(e, tr, c, NKV, T, 256);
    } else
        enc_attn_batch(e, s, barg(b, B_qn), kv_arg(s, kbase), kv_arg(s, vbase), barg(b, B_sel), barg(b, B_cnt),
                       barg(b, B_qg + HD), barg(b, B_att), pos0, T);
    enc_mm(e, &A->o, barg(b, B_att), barg(b, B_y), T);
}
static void enc_attn_t(Enc *e, const Attn *A, Session *s, size_t T) { enc_attn_t_at(e, A, s, T, s->pos); }

// MTP rows: B.a holds the T token embeddings, hid the T trunk (or head) streams.
static void enc_mtp_rows(Enc *e, const Mtp *mt, Session *s, sx_arg hid, size_t T, size_t pos0, bool logits) {
    Model *m = e->m;
    sx_buffer *b = m->bscratch;
    enc_norm(e, barg(b, B_a), mt->n_e, barg(b, B_a), T, H, H, H, 0, true);
    enc_mm(e, &mt->fc_e, barg(b, B_a), barg(b, B_y), T);
    enc_norm(e, hid, mt->n_h, barg(b, B_xn), T, HCW, HCW, HCW, 0, true);
    enc_mm(e, &mt->fc_h, barg(b, B_xn), barg(b, B_X), T * HC);
    uint32_t ap[3] = {(uint32_t)(T * HCW), (uint32_t)H, (uint32_t)HC};
    sx_arg a[] = {barg(b, B_X), barg(b, B_y), gpu_bytes_arg(ap, 12)};
    D1(e, m->p.add_streams, a, T * HCW);
    const Layer *L = &mt->layer;
    enc_residual_t(e, &L->attn_hc, T);
    enc_attn_t_at(e, L->attn, s, T, pos0);
    enc_hc_add_t(e, T);
    enc_residual_t(e, &L->mlp_hc, T);
    enc_moe_t(e, &L->moe, T);
    enc_hc_add_t(e, T);
    if (logits) {
        // the head's mixer over the last row only, then the shared lm_head
        enc_copy(e, barg(b, B_X + (T - 1) * HCW), sarg(m, S_X), HCW);
        enc_residual(e, &mt->mixer, S_h);
        enc_mv(e, &m->lm_head, sarg(m, S_h), barg(m->mtp_logits, 0));
    }
}

static void enc_moe_t(Enc *e, const Moe *E, size_t T) {
    Model *m = e->m;
    sx_buffer *b = m->bscratch;
    enc_mm(e, &E->router, barg(b, B_a), barg(b, B_rl), T);
    enc_mm(e, &E->shared_gate, barg(b, B_a), barg(b, B_sgs), T);
    uint32_t tp[2] = {E->ne, (uint32_t)TOPK};
    sx_arg a[] = {barg(b, B_rl), barg(b, B_topi), barg(b, B_topw), gpu_bytes_arg(tp, 8)};
    D(e, m->p.moe_topk, a, T, 1, E->ne);
    if (T <= SMALL) {
        // routed experts per (token, slot) with the decode kernels: each slot reads its expert
        enc_gate_up(e, &E->eg, &E->eu, barg(b, B_a), barg(b, B_eh), true, barg(b, B_topi), T * TOPK);
        enc_mv_expert_ti(e, &E->ed, barg(b, B_eh), FF, 1, barg(b, B_ed), T);
        if (small_fused() && E->sg.bits == 4 && E->sg.group_size == 32) {
            // shared expert gate + up + SwiGLU in one dispatch, one slot per token
            enc_gate_up(e, &E->sg, &E->su, barg(b, B_a), barg(b, B_shh), false, barg(b, B_topi), T);
        } else {
            enc_mm(e, &E->sg, barg(b, B_a), barg(b, B_shg), T);
            enc_mm(e, &E->su, barg(b, B_a), barg(b, B_shu), T);
            uint32_t ns0 = (uint32_t)(T * SFF);
            sx_arg c[] = {barg(b, B_shg), barg(b, B_shu), barg(b, B_shh), BYTES(ns0)};
            D1(e, m->p.swiglu, c, T * SFF);
        }
        enc_mm(e, &E->sd, barg(b, B_shh), barg(b, B_sh), T);
        uint32_t cp0[3] = {(uint32_t)H, (uint32_t)TOPK, (uint32_t)T};
        sx_arg c[] = {barg(b, B_ed), barg(b, B_topw), barg(b, B_sh), barg(b, B_sgs), barg(b, B_y), gpu_bytes_arg(cp0, 12)};
        D1(e, m->p.moe_combine, c, T * H);
        return;
    }
    uint32_t gp[4] = {(uint32_t)(T * TOPK), (uint32_t)TOPK, E->ne, m->fast_prefill ? 64u : 32u};
    sx_arg g1[] = {barg(b, B_topi), barg(b, B_gtok), barg(b, B_ginv), barg(b, B_gtiles), barg(b, B_gnt), gpu_bytes_arg(gp, 16)};
    D(e, m->p.moe_group, g1, 1, 1, 1024);
    uint32_t rp[2] = {(uint32_t)H, (uint32_t)(T * TOPK)};
    if (m->fast_prefill) {
        // straight to the half input of the accelerator matmul (the cache says B.xg is in B.xh)
        sx_arg c[] = {barg(b, B_a), barg(b, B_gtok), barg(b, B_xh), gpu_bytes_arg(rp, 8)};
        D1(e, m->p.gather_rows_h, c, T * TOPK * H);
        e->xh_set = true;
        e->xh_buf = b;
        e->xh_off = B_xg * 4;
        e->xh_n = T * TOPK * H;
    } else {
        sx_arg c[] = {barg(b, B_a), barg(b, B_gtok), barg(b, B_xg), gpu_bytes_arg(rp, 8)};
        D1(e, m->p.gather_rows, c, T * TOPK * H);
    }
    enc_mm_grouped(e, &E->eg, barg(b, B_xg), barg(b, B_eg), T);
    enc_mm_grouped(e, &E->eu, barg(b, B_xg), barg(b, B_eu), T);
    uint32_t ne = (uint32_t)(T * TOPK * FF);
    if (m->fast_prefill) {
        sx_arg c[] = {barg(b, B_eg), barg(b, B_eu), barg(b, B_xh), BYTES(ne)};
        D1(e, m->p.swiglu_h, c, T * TOPK * FF);
        e->xh_set = true;
        e->xh_buf = b;
        e->xh_off = B_eh * 4;
        e->xh_n = T * TOPK * FF;
    } else {
        sx_arg c[] = {barg(b, B_eg), barg(b, B_eu), barg(b, B_eh), BYTES(ne)};
        D1(e, m->p.swiglu, c, T * TOPK * FF);
    }
    enc_mm_grouped(e, &E->ed, barg(b, B_eh), barg(b, B_ed), T);
    enc_mm(e, &E->sg, barg(b, B_a), barg(b, B_shg), T);
    enc_mm(e, &E->su, barg(b, B_a), barg(b, B_shu), T);
    uint32_t ns = (uint32_t)(T * SFF);
    sx_arg c[] = {barg(b, B_shg), barg(b, B_shu), barg(b, B_shh), BYTES(ns)};
    D1(e, m->p.swiglu, c, T * SFF);
    enc_mm(e, &E->sd, barg(b, B_shh), barg(b, B_sh), T);
    uint32_t cp[3] = {(uint32_t)H, (uint32_t)TOPK, (uint32_t)T};
    sx_arg f[] = {barg(b, B_ed), barg(b, B_topw), barg(b, B_sh),          barg(b, B_sgs),
                  barg(b, B_y),  gpu_bytes_arg(cp, 12), barg(b, B_ginv)};
    D1(e, m->p.moe_combine_g, f, T * H);
}

static void enc_ple_t(Enc *e, const Ple *P, Session *s, size_t T) {
    Model *m = e->m;
    sx_buffer *b = m->bscratch;
    enc_mm(e, &P->key, barg(b, B_pe), barg(b, B_kp), T);
    enc_normw(e, barg(b, B_kp), P->norm_key, barg(b, B_kn), T * HC, H, H, H, H, true, HC);
    enc_mm(e, &P->value, barg(b, B_pe), barg(b, B_val), T);
    enc_normw(e, barg(b, B_X), P->norm_query, barg(b, B_qp), T * HC, H, H, H, H, true, HC);
    uint32_t hh[2] = {(uint32_t)H, (uint32_t)HC};
    sx_arg a[] = {barg(b, B_kn), barg(b, B_qp), barg(b, B_val), barg(b, B_gv), gpu_bytes_arg(hh, 8)};
    D(e, m->p.ple_gate, a, T * HC, 1, 256);
    enc_normw(e, barg(b, B_gv), P->norm_conv, barg(b, B_gvn), T * HC, H, H, H, H, true, HC);
    uint32_t n[4] = {(uint32_t)HCW, (uint32_t)T, s->spec, 0};
    sx_arg c[] = {barg(b, B_gv), barg(b, B_gvn), barg(s->ple_hist, 0), barg(m->params.buf, P->conv.off), barg(b, B_X),
                  gpu_bytes_arg(n, 16), barg(s->sp_ple, 0), src_ple(s)};
    D1(e, m->p.ple_conv, c, HCW);
}

// norm_done: S.xn already holds the normed streams (hc_add_norm did it).
static void enc_residual_n(Enc *e, const Residual *R, size_t out, bool norm_done) {
    Model *m = e->m;
    if (!norm_done) enc_norm(e, sarg(m, S_X), R->norm, sarg(m, S_xn), HC, H, H, H, H, true);
    // down and inject share xn: one dispatch; up reads silu(low / HC) directly
    if (R->has_inject)
        enc_mv_multi(e, LINS(&R->down, &R->inject), sarg(m, S_xn), ARGS(sarg(m, S_low), sarg(m, S_r)), 256, false, 1.0f);
    else
        enc_mv_multi(e, LINS(&R->down), sarg(m, S_xn), ARGS(sarg(m, S_low)), 256, false, 1.0f);
    // up projection and the stream mix in one dispatch
    uint32_t pu[4] = {(uint32_t)LOW, (uint32_t)H, (uint32_t)HC, 0};
    float scale = 1.0f / (float)HC;
    uint32_t sh[3];
    sx_arg args[10] = {TARG(R->up.weight, &sh[0]), TARG(R->up.scales, &sh[1]), TARG(R->up.biases, &sh[2]),
                       sarg(m, S_low),             sarg(m, S_xn),              sarg(m, out),
                       gpu_bytes_arg(pu, 16),      BYTES(scale),               gpu_bytes_arg(sh, 12)};
    if (!is4_32(&R->up)) {
        uint32_t q[2] = {R->up.bits, R->up.group_size};
        args[9] = BYTES(q);
        enc_d(e, m->gen->up_mix, args, 10, H / 8, 1, 256);
    } else
        enc_d(e, m->p.qmv_up_mix, args, 9, H / 8, 1, 256);
}
// Gated residual (M:999-1027) into scratch[out]; the inject logits go to S.r.
static void enc_residual(Enc *e, const Residual *R, size_t out) { enc_residual_n(e, R, out, false); }

// residual_n over a small chunk of T tokens in the prefill scratch (verification chunks, MTP
// rows): the decode kernels with a token grid dimension (hyper-connection matrices are small
// enough to be re-read per token from cache).
static void enc_residual_s(Enc *e, const Residual *R, size_t T, bool norm_done) {
    Model *m = e->m;
    sx_buffer *b = m->bscratch;
    if (!norm_done) enc_normw(e, barg(b, B_X), R->norm, barg(b, B_xn), T * HC, H, H, H, H, true, HC);
    if (R->has_inject)
        enc_mv_multi_t(e, LINS(&R->down, &R->inject), barg(b, B_xn), ARGS(barg(b, B_low), barg(b, B_r)), 256, false, 1.0f, T);
    else
        enc_mv_multi_t(e, LINS(&R->down), barg(b, B_xn), ARGS(barg(b, B_low)), 256, false, 1.0f, T);
    uint32_t pu[4] = {(uint32_t)LOW, (uint32_t)H, (uint32_t)HC, 0};
    float scale = 1.0f / (float)HC;
    uint32_t sh[3];
    sx_arg args[10] = {TARG(R->up.weight, &sh[0]), TARG(R->up.scales, &sh[1]), TARG(R->up.biases, &sh[2]),
                       barg(b, B_low),             barg(b, B_xn),              barg(b, B_a),
                       gpu_bytes_arg(pu, 16),      BYTES(scale),               gpu_bytes_arg(sh, 12)};
    if (!is4_32(&R->up)) {
        uint32_t q[2] = {R->up.bits, R->up.group_size};
        args[9] = BYTES(q);
        enc_d(e, m->gen->up_mix, args, 10, H / 8, T, 256);
    } else
        enc_d(e, m->p.qmv_up_mix, args, 9, H / 8, T, 256);
}
static void enc_hc_add_norm_s(Enc *e, const Residual *next, size_t T) {
    sx_buffer *b = e->m->bscratch;
    uint32_t p[2] = {(uint32_t)H, (uint32_t)HC};
    float eps = EPS;
    sx_arg a[] = {barg(b, B_X), barg(b, B_y), barg(b, B_r), barg(e->m->params.buf, next->norm.off), barg(b, B_xn),
                  gpu_bytes_arg(p, 8), BYTES(eps)};
    D(e, e->m->p.hc_add_norm, a, HC, T, 1024);
}
// hc_add fused with the next block's norm into S.xn.
static void enc_hc_add_norm(Enc *e, const Residual *next) {
    Model *m = e->m;
    uint32_t p[2] = {(uint32_t)H, (uint32_t)HC};
    float eps = EPS;
    sx_arg a[] = {sarg(m, S_X), sarg(m, S_y), sarg(m, S_r), barg(m->params.buf, next->norm.off), sarg(m, S_xn),
                  gpu_bytes_arg(p, 8), BYTES(eps)};
    D(e, m->p.hc_add_norm, a, HC, 1, 1024);
}
static void enc_hc_add(Enc *e) {
    Model *m = e->m;
    uint32_t p[3] = {(uint32_t)H, (uint32_t)HC, 1};
    sx_arg a[] = {sarg(m, S_X), sarg(m, S_y), sarg(m, S_r), gpu_bytes_arg(p, 12)};
    D1(e, m->p.hc_add, a, HCW);
}

// Gated DeltaNet, recurrent form (M:395-622).
static void enc_gdn(Enc *e, const Gdn *G, Session *s) {
    Model *m = e->m;
    enc_mv_multi(e, LINS(&G->qkv, &G->z, &G->b, &G->a), sarg(m, S_a),
                 ARGS(sarg(m, S_qkv), sarg(m, S_z), sarg(m, S_gb), sarg(m, S_ga)), 32, false, 1.0f);
    // conv and the q / k norms in one dispatch (CONV / DK threadgroups of DK channels)
    static_assert(CONV % DK == 0);
    uint32_t qk[4] = {(uint32_t)NK, (uint32_t)DK, (uint32_t)KD, (uint32_t)CONV};
    sx_arg a[] = {sarg(m, S_qkv), barg(s->gdn_conv, G->slot * (KCONV - 1) * CONV), barg(m->params.buf, G->conv.off),
                  sarg(m, S_conv), gpu_bytes_arg(qk, 16), src_conv(s, G->slot)};
    D(e, m->p.gdn_conv_norm, a, CONV / DK, 1, DK);
    GdnP gp = {(uint32_t)NK, (uint32_t)NV, (uint32_t)DK, (uint32_t)DV, (uint32_t)KD, (uint32_t)CONV, 1, EPS, 0, 0};
    if (!skip.gdn) {
        sx_arg c[] = {sarg(m, S_conv),
                      sarg(m, S_ga),
                      sarg(m, S_gb),
                      barg(m->params.buf, G->a_log.off),
                      barg(m->params.buf, G->dt_bias.off),
                      barg(s->gdn_state, G->slot * NV * DK * DV),
                      sarg(m, S_o),
                      BYTES(gp),
                      barg(s->sp_gdn, 0),
                      src_gdn(s, G->slot)};
        D(e, m->p.gdn_step, c, DV / 16, NV, 64);
    }
    sx_arg f[] = {sarg(m, S_o), sarg(m, S_z), barg(m->params.buf, G->norm.off), BYTES(gp)};
    D(e, m->p.gdn_gnorm, f, NV, 1, DV);
    enc_mv(e, &G->out, sarg(m, S_o), sarg(m, S_y));
}

// QSA attention (M:670-897): dense over the indexer's selection.
static void enc_attn(Enc *e, const Attn *A, Session *s) {
    Model *m = e->m;
    size_t pos = s->pos;
    size_t kv_row = NKV * HD;
    size_t kbase = A->slot * 2 * s->cap * kv_row; // keys, then values, per attention layer
    size_t vbase = kbase + s->cap * kv_row;
    bool multi_ok = (is4_32(&A->q) && is4_32(&A->k) && is4_32(&A->v) && is4_32(&A->ix)) ||
                    (m->gen != NULL && A->q.bits != 16 && A->k.bits != 16 && A->v.bits != 16 && A->ix.bits != 16);
    if (multi_ok && getenv("SAETTA_ATTN_MULTI") == NULL) {
        // the four projections of the same input in one dispatch
        enc_mv_multi(e, LINS(&A->q, &A->k, &A->v, &A->ix), sarg(m, S_a),
                     ARGS(sarg(m, S_qg), sarg(m, S_kx), sarg(m, S_vx), sarg(m, S_ix)), 32, false, 1.0f);
    } else {
        enc_mv(e, &A->q, sarg(m, S_a), sarg(m, S_qg));
        enc_mv(e, &A->k, sarg(m, S_a), sarg(m, S_kx));
        enc_mv(e, &A->v, sarg(m, S_a), sarg(m, S_vx));
        enc_mv(e, &A->ix, sarg(m, S_a), sarg(m, S_ix));
    }
    if (s->kv_half && getenv("SAETTA_ATTN_PREP") == NULL) {
        // norms, RoPE, KV and indexer-key stores in one dispatch
        AttnPrepP pp = {(uint32_t)NQ, (uint32_t)NKV, (uint32_t)HD, (uint32_t)(ROT / 2), (uint32_t)IXD, (uint32_t)(IXH * IXD), EPS};
        sx_arg a[] = {sarg(m, S_qg),
                      sarg(m, S_kx),
                      sarg(m, S_vx),
                      sarg(m, S_ix),
                      barg(m->params.buf, A->q_norm.off),
                      barg(m->params.buf, A->k_norm.off),
                      barg(s->rope, pos * ROT),
                      sarg(m, S_qn),
                      kv_arg(s, kbase + pos * kv_row),
                      kv_arg(s, vbase + pos * kv_row),
                      barg(s->ixk, A->slot * s->cap * IXD + pos * IXD),
                      BYTES(pp)};
        D(e, m->p.attn_prep_dec, a, NQ + NKV + 1, 1, HD);
    } else {
        enc_norm(e, sarg(m, S_qg), A->q_norm, sarg(m, S_qn), NQ, HD, 2 * HD, HD, 0, true);
        enc_rope(e, s, sarg(m, S_qn), NQ, HD, pos);
        enc_norm(e, sarg(m, S_kx), A->k_norm, sarg(m, S_kx), NKV, HD, HD, HD, 0, true);
        enc_rope(e, s, sarg(m, S_kx), NKV, HD, pos);
        enc_kv_store(e, s, sarg(m, S_kx), kbase + pos * kv_row, kv_row, 1);
        enc_kv_store(e, s, sarg(m, S_vx), vbase + pos * kv_row, kv_row, 1);
        enc_copy(e, sarg(m, S_ix + IXH * IXD), barg(s->ixk, A->slot * s->cap * IXD + pos * IXD), IXD);
    }
    size_t n = pos + 1;
    if (n % BLOCK == 0) enc_ix_blocks(e, A, s, n / BLOCK - 1, 1);
    if (n / BLOCK > TOPB) enc_ix_select(e, A, s, sarg(m, S_ix), pos, n, sarg(m, S_sel), sarg(m, S_cnt));
    if (!skip.attn)
        enc_attn_batch(e, s, sarg(m, S_qn), kv_arg(s, kbase), kv_arg(s, vbase), sarg(m, S_sel), sarg(m, S_cnt),
                       sarg(m, S_qg + HD), sarg(m, S_att), pos, 1);
    enc_mv(e, &A->o, sarg(m, S_att), sarg(m, S_y));
}

// MoE (M:900-996): softmax router, top-10 renormalized, SwiGLU experts, gated shared expert.
static void enc_moe(Enc *e, const Moe *E) {
    Model *m = e->m;
    if (E->router.bits == 8 && E->router.group_size == 64 && E->shared_gate.bits == 8 && E->shared_gate.group_size == 64 &&
        !skip.dense) {
        // router and shared gate in one dispatch
        uint32_t sh[3], sh2[3];
        uint32_t dm[2] = {(uint32_t)E->router.in_features, (uint32_t)E->router.out_features};
        sx_arg args[] = {TARG(E->router.weight, &sh[0]),       TARG(E->router.scales, &sh[1]),
                         TARG(E->router.biases, &sh[2]),       sarg(m, S_a),
                         sarg(m, S_rl),                        gpu_bytes_arg(dm, 8),
                         gpu_bytes_arg(sh, 12),                TARG(E->shared_gate.weight, &sh2[0]),
                         TARG(E->shared_gate.scales, &sh2[1]), TARG(E->shared_gate.biases, &sh2[2]),
                         sarg(m, S_sgs),                       gpu_bytes_arg(sh2, 12)};
        D(e, m->p.qmv8_router_sg, args, dm[1] + 1, 1, 32);
    } else {
        enc_mv(e, &E->router, sarg(m, S_a), sarg(m, S_rl));
        enc_mv(e, &E->shared_gate, sarg(m, S_a), sarg(m, S_sgs));
    }
    uint32_t tp[2] = {E->ne, (uint32_t)TOPK};
    sx_arg a[] = {sarg(m, S_rl), sarg(m, S_topi), sarg(m, S_topw), gpu_bytes_arg(tp, 8)};
    D(e, m->p.moe_topk, a, 1, 1, E->ne);
    if (fold_shared(E)) {
        // the shared expert as slot TOPK of the routed launches (6 dispatches instead of 8)
        enc_gate_up_sh(e, E, sarg(m, S_a), sarg(m, S_eh));
        enc_down_sh(e, E, sarg(m, S_eh), sarg(m, S_ed));
        uint32_t cp[3] = {(uint32_t)H, (uint32_t)TOPK, 1};
        sx_arg c[] = {sarg(m, S_ed), sarg(m, S_topw), sarg(m, S_ed + TOPK * H), sarg(m, S_sgs), sarg(m, S_y),
                      gpu_bytes_arg(cp, 12)};
        D1(e, m->p.moe_combine, c, H);
        return;
    }
    enc_gate_up(e, &E->eg, &E->eu, sarg(m, S_a), sarg(m, S_eh), true, sarg(m, S_topi), TOPK);
    enc_mv_expert(e, &E->ed, sarg(m, S_eh), FF, sarg(m, S_ed));
    if (same_quant(&E->sg, &E->su))
        enc_gate_up(e, &E->sg, &E->su, sarg(m, S_a), sarg(m, S_shh), false, sarg(m, S_topi), 1);
    else {
        // gate and up at different bit widths (the Coder pack): two matvecs and SwiGLU
        enc_mv(e, &E->sg, sarg(m, S_a), sarg(m, S_shg));
        enc_mv(e, &E->su, sarg(m, S_a), sarg(m, S_shu));
        uint32_t ns = SFF;
        sx_arg c[] = {sarg(m, S_shg), sarg(m, S_shu), sarg(m, S_shh), BYTES(ns)};
        D1(e, m->p.swiglu, c, SFF);
    }
    enc_mv(e, &E->sd, sarg(m, S_shh), sarg(m, S_sh));
    uint32_t cp[3] = {(uint32_t)H, (uint32_t)TOPK, 1};
    sx_arg c[] = {sarg(m, S_ed), sarg(m, S_topw), sarg(m, S_sh), sarg(m, S_sgs), sarg(m, S_y), gpu_bytes_arg(cp, 12)};
    D1(e, m->p.moe_combine, c, H);
}

// PLE (M:1180-1253), added to X before its layer; the n-gram rows are in S.pe.
static void enc_ple(Enc *e, const Ple *P, Session *s) {
    Model *m = e->m;
    enc_mv(e, &P->key, sarg(m, S_pe), sarg(m, S_kp));
    enc_norm(e, sarg(m, S_kp), P->norm_key, sarg(m, S_kn), HC, H, H, H, H, true);
    enc_mv(e, &P->value, sarg(m, S_pe), sarg(m, S_val));
    enc_norm(e, sarg(m, S_X), P->norm_query, sarg(m, S_qp), HC, H, H, H, H, true);
    uint32_t hh[2] = {(uint32_t)H, (uint32_t)HC};
    sx_arg a[] = {sarg(m, S_kn), sarg(m, S_qp), sarg(m, S_val), sarg(m, S_gv), gpu_bytes_arg(hh, 8)};
    D(e, m->p.ple_gate, a, HC, 1, 256);
    enc_norm(e, sarg(m, S_gv), P->norm_conv, sarg(m, S_gvn), HC, H, H, H, H, true);
    uint32_t n[4] = {(uint32_t)HCW, 1, 0, 0};
    sx_arg c[] = {sarg(m, S_gv), sarg(m, S_gvn), barg(s->ple_hist, 0), barg(m->params.buf, P->conv.off), sarg(m, S_X),
                  gpu_bytes_arg(n, 16), barg(s->sp_ple, 0), src_ple(s)};
    D1(e, m->p.ple_conv, c, HCW);
}

// ---- host math ------------------------------------------------------------------------------

// std.math.pow(f64) as the Zig engine computed it (a port of Go's math.Pow), so the RoPE
// table is bit for bit the same (libm pow rounds differently).
static double zig_pow(double x, double y) {
    if (y == 0 || x == 1) return 1;
    if (isnan(x) || isnan(y)) return NAN;
    if (y == 1) return x;
    if (x == 0 || isinf(y) || isinf(x)) return pow(x, y); // special cases agree with C99
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

// The token's 16 n-gram table rows (global row indices, M:1030-1178), and the history update.
static void ple_row_ids(const Ple *P, Session *s, uint32_t token, uint64_t rows[NG_HEADS]) {
    int64_t t0 = token;
    int64_t t1 = s->since_eos >= 1 ? s->prev[1] : EOS_PLE;
    int64_t t2 = s->since_eos >= 2 ? s->prev[0] : EOS_PLE;
    // wrapping multiplies (Zig *%)
    int64_t mixed2 = (int64_t)((uint64_t)t0 * (uint64_t)P->mult[0]) ^ (int64_t)((uint64_t)t1 * (uint64_t)P->mult[1]);
    int64_t mixed3 = mixed2 ^ (int64_t)((uint64_t)t2 * (uint64_t)P->mult[2]);
    for (size_t hd = 0; hd < NG_HEADS; hd++) {
        int64_t mixed = hd < CONFIG_HEADS_PER_NGRAM ? mixed2 : mixed3;
        int64_t r = mixed % P->vocab[hd]; // @mod: floored, vocab > 0
        if (r < 0) r += P->vocab[hd];
        rows[hd] = (uint64_t)(r + P->offsets[hd]);
    }
    s->prev[0] = s->prev[1];
    s->prev[1] = token;
    s->since_eos = token == EOS_PLE ? 0 : s->since_eos + 1;
}

typedef struct {
    Linear lin;
    size_t local;
} PleRow;

static PleRow ple_row_linear(const Ple *P, uint64_t row) {
    uint64_t rows_per_shard = P->shard_w[0].shape[0];
    size_t shard = (size_t)(row / rows_per_shard);
    // bits and group from the shard shapes (u32 words per row, groups per row): 4/32 in the
    // full pack, 5/32 in the Coder pack
    uint32_t bits = (uint32_t)(P->shard_w[0].shape[1] * 32 / NG_DIM);
    uint32_t group = (uint32_t)(NG_DIM / P->shard_s[0].shape[1]);
    return (PleRow){.lin = {.weight = P->shard_w[shard], .scales = P->shard_s[shard], .biases = P->shard_b[shard],
                            .bits = bits, .group_size = group, .out_features = rows_per_shard, .in_features = NG_DIM},
                    .local = (size_t)(row % rows_per_shard)};
}

// Starts reading the weight, scale and bias bytes of an n-gram row (the table is demand
// paged, not warmed).
static void ple_row_prefetch(const Pack *pack, const Ple *P, uint64_t row) {
    PleRow r = ple_row_linear(P, row);
    size_t wb = NG_DIM * r.lin.bits / 8;
    size_t sb = NG_DIM / r.lin.group_size * 2; // bf16 per group
    sx_prefetch(pack_data(pack, &r.lin.weight) + r.local * wb, wb);
    sx_prefetch(pack_data(pack, &r.lin.scales) + r.local * sb, sb);
    sx_prefetch(pack_data(pack, &r.lin.biases) + r.local * sb, sb);
}

static void ple_rows_read(const Pack *pack, const Ple *P, const uint64_t rows[NG_HEADS], float *out) {
    for (size_t hd = 0; hd < NG_HEADS; hd++) {
        PleRow r = ple_row_linear(P, rows[hd]);
        dequant_row(pack, &r.lin, r.local, out + hd * NG_DIM);
    }
}

// One token: row ids, prefetch all 16 rows, then read them.
static void ple_rows(const Pack *pack, const Ple *P, Session *s, uint32_t token, float *out) {
    uint64_t rows[NG_HEADS];
    ple_row_ids(P, s, token, rows);
    for (size_t i = 0; i < NG_HEADS; i++) ple_row_prefetch(pack, P, rows[i]);
    ple_rows_read(pack, P, rows, out);
}

// ---- Model passes ---------------------------------------------------------------------------

static const Ple *first_ple(const Model *m) {
    for (size_t i = 0; i < NLAYERS; i++)
        if (m->layers[i].ple) return m->layers[i].ple;
    return NULL;
}

static void drop_pending(Model *m) {
    if (m->has_pend) sx_cmd_discard(m->pend.e.cmd);
    m->has_pend = false;
}

// Host inputs of a decode step: the token's embedding row into every stream, its n-gram
// rows for PLE (in the decode scratch the encoded step reads).
static void step_inputs(Model *m, Session *s, uint32_t token) {
    float *sc = sf(m);
    dequant_row(m->pack, &m->embed, token, sc + S_X);
    for (size_t k = 1; k < HC; k++) memcpy(sc + S_X + k * H, sc + S_X, H * sizeof(float));
    for (size_t i = 0; i < NLAYERS; i++)
        if (m->layers[i].ple) ple_rows(m->pack, m->layers[i].ple, s, token, sc + S_pe);
}

static Err step_encode(Model *m, Session *s, Enc *out) {
    Enc e = enc_begin(m);
    // each block's residual add also norms the streams for the block after it, except
    // before a PLE layer (PLE reads the streams first)
    bool normed = false;
    for (size_t li = 0; li < NLAYERS; li++) {
        const Layer *L = &m->layers[li];
        if (L->ple) enc_ple(&e, L->ple, s);
        enc_residual_n(&e, &L->attn_hc, S_a, normed);
        if (L->gdn)
            enc_gdn(&e, L->gdn, s);
        else
            enc_attn(&e, L->attn, s);
        enc_hc_add_norm(&e, &L->mlp_hc);
        enc_residual_n(&e, &L->mlp_hc, S_a, true);
        enc_moe(&e, &L->moe);
        const Residual *nxt = li + 1 < NLAYERS ? (m->layers[li + 1].ple == NULL ? &m->layers[li + 1].attn_hc : NULL) : &m->mixer;
        if (nxt)
            enc_hc_add_norm(&e, nxt);
        else
            enc_hc_add(&e);
        normed = nxt != NULL;
        if (s->trace != NULL) enc_copy(&e, sarg(m, S_X), barg(s->trace_buf, li * HCW), HCW);
    }
    s->roll = SESSION_NO_ROLL; // the step's start state is encoded; it ends in the live state
    enc_residual_n(&e, &m->mixer, S_h, normed);
    enc_mv(&e, &m->lm_head, sarg(m, S_h), barg(m->logits, 0));
    if (m->mtp != NULL) {
        enc_copy(&e, sarg(m, S_X), barg(m->mtp_hid, 0), HCW);
        s->mtp_n = 1;
    }
    *out = e;
    return OK;
}

Err model_step(Model *m, Session *s, uint32_t token, float **logits) {
    skip_init();
    if (s->pos >= s->cap) return "ContextFull";
    uint64_t t_host0 = now_mono_ns();
    drop_pending(m);
    step_inputs(m, s, token);
    Enc e;
    Err err = step_encode(m, s, &e);
    if (!err) {
        timing.encoded = now_mono_ns();
        err = enc_finish(&e);
    }
    if (!err) {
        if (s->trace) memcpy(s->trace, gpu_floats(s->trace_buf), s->trace_len * sizeof(float));
        s->pos += 1;
        *logits = gpu_floats(m->logits);
    }
    step_timing(t_host0);
    return err;
}

// Host inputs of a chunk into stage buffer stg: n-gram ids (and their prefetch, so the page
// faults overlap), the embeddings, then the n-gram rows.
static void prep_chunk(const Model *m, const Ple *P, Session *s, const uint32_t *toks, size_t n,
                       uint64_t (*rows)[NG_HEADS], sx_buffer *stg) {
    float *sf_ = gpu_floats(stg);
    if (P)
        for (size_t t = 0; t < n; t++) {
            ple_row_ids(P, s, toks[t], rows[t]);
            for (size_t k = 0; k < NG_HEADS; k++) ple_row_prefetch(m->pack, P, rows[t][k]);
        }
    for (size_t t = 0; t < n; t++) dequant_row(m->pack, &m->embed, toks[t], sf_ + t * H);
    if (P)
        for (size_t t = 0; t < n; t++) ple_rows_read(m->pack, P, rows[t], sf_ + CH * H + t * CONFIG_PLE_EMBED_DIM);
}

// Arguments of the thread that prepares the next chunk.
typedef struct {
    const Model *m;
    const Ple *P;
    Session *s;
    const uint32_t *toks;
    size_t n;
    uint64_t (*rows)[NG_HEADS];
    sx_buffer *stg;
} PrepArgs;

static void *prep_thread(void *arg) {
    PrepArgs *a = arg;
    prep_chunk(a->m, a->P, a->s, a->toks, a->n, a->rows, a->stg);
    return NULL;
}

// The next chunk's preparation thread (prefill).
typedef struct {
    bool on;
    pthread_t th;
    PrepArgs args;
} NextThread;

static void next_join(NextThread *t) {
    if (t->on) pthread_join(t->th, NULL);
    t->on = false;
}

// The dispatches of a chunk of T tokens whose inputs are (or will be, before commit) in
// stage buffer stg; reads the session's start state, leaves s unchanged.
static Err chunk_encode(Model *m, Session *s, size_t T, sx_buffer *trace, bool all, sx_buffer *stg, Enc *out) {
    const Ple *ple = first_ple(m);
    Enc e = enc_begin(m);
    Enc *ep = &e;
    uint32_t bp[3] = {(uint32_t)(T * HCW), (uint32_t)H, (uint32_t)HC};
    sx_arg a[] = {barg(m->bscratch, B_X), barg(stg, 0), gpu_bytes_arg(bp, 12)};
    D1(ep, m->p.bcast_streams, a, T * HCW);
    if (ple != NULL) enc_copy(ep, barg(stg, CH * H), barg(m->bscratch, B_pe), T * CONFIG_PLE_EMBED_DIM);
    // small chunks (verification, SAETTA_SMALL_FUSED=0 to compare): the decode fusions
    bool small = T <= SMALL && trace == NULL && small_fused();
    bool normed = false;
    for (size_t li = 0; li < NLAYERS; li++) {
        const Layer *L = &m->layers[li];
        if (L->ple) enc_ple_t(ep, L->ple, s, T);
        if (small)
            enc_residual_s(ep, &L->attn_hc, T, normed);
        else
            enc_residual_t(ep, &L->attn_hc, T);
        if (L->gdn)
            enc_gdn_t(ep, L->gdn, s, T);
        else
            enc_attn_t(ep, L->attn, s, T);
        if (small) {
            enc_hc_add_norm_s(ep, &L->mlp_hc, T);
            enc_residual_s(ep, &L->mlp_hc, T, true);
        } else {
            enc_hc_add_t(ep, T);
            enc_residual_t(ep, &L->mlp_hc, T);
        }
        enc_moe_t(ep, &L->moe, T);
        if (small) {
            const Residual *nxt =
                li + 1 < NLAYERS ? (m->layers[li + 1].ple == NULL ? &m->layers[li + 1].attn_hc : NULL) : NULL;
            if (nxt)
                enc_hc_add_norm_s(ep, nxt, T);
            else
                enc_hc_add_t(ep, T);
            normed = nxt != NULL;
        } else
            enc_hc_add_t(ep, T);
        if (trace) enc_copy(ep, barg(m->bscratch, B_X), barg(trace, li * T * HCW), T * HCW);
    }
    if (m->mtp != NULL) enc_copy(ep, barg(m->bscratch, B_X), barg(m->mtp_hid, all ? 0 : HCW), T * HCW);
    if (all) {
        // every position: final mixer over the chunk, lm_head as one matmul
        enc_residual_t(ep, &m->mixer, T);
        enc_mm(ep, &m->lm_head, barg(m->bscratch, B_a), barg(m->vlogits, 0), T);
        *out = e;
        return OK;
    }
    // last token: final mixer and lm_head through the decode scratch
    enc_copy(ep, barg(m->bscratch, B_X + (T - 1) * HCW), sarg(m, S_X), HCW);
    enc_residual(ep, &m->mixer, S_h);
    enc_mv(ep, &m->lm_head, sarg(m, S_h), barg(m->logits, 0));
    *out = e;
    return OK;
}

// MTP history for a prompt chunk at pos0: rows at positions q in [pos0, pos0 + T) pair the
// trunk streams at q - 1 (row 0: the previous chunk's last, kept in mtp_hid row 0) with
// toks[q - pos0]; the first row is skipped at position 0 or when that stream is unknown.
static Err mtp_prefill(Model *m, const Mtp *mt, Session *s, const uint32_t *toks, size_t T, size_t pos0,
                       sx_buffer *stg) {
    (void)toks;
    size_t skip0 = (pos0 == 0 || !s->mtp_ok) ? 1 : 0;
    if (T > skip0) {
        // the chunk's embeddings are still in its stage buffer
        Enc e = enc_begin(m);
        enc_copy(&e, barg(stg, skip0 * H), barg(m->bscratch, B_a), (T - skip0) * H);
        enc_mtp_rows(&e, mt, s, barg(m->mtp_hid, skip0 * HCW), T - skip0, pos0 + skip0, false);
        enc_copy(&e, barg(m->mtp_hid, T * HCW), barg(m->mtp_hid, 0), HCW);
        TRY(enc_finish(&e));
    } else {
        Enc e = enc_begin(m);
        enc_copy(&e, barg(m->mtp_hid, T * HCW), barg(m->mtp_hid, 0), HCW);
        TRY(enc_finish(&e));
    }
    s->mtp_ok = true;
    s->mtp_n = 1;
    return OK;
}

// next: the following chunk's tokens (nn of them), whose n-gram ids a thread computes (into
// s->ple_rows_b, returned in thread) once this chunk no longer needs s->prev.
static Err chunk_logits_next(Model *m, Session *s, const uint32_t *toks, size_t T, sx_buffer *trace, bool all,
                             const uint32_t *next, size_t nn, NextThread *thread, float **logits) {
    drop_pending(m);
    if (s->pos + T > s->cap) return "ContextFull";
    const Ple *ple = first_ple(m);
    // host inputs (already prepared by the previous chunk's thread when ids_ready)
    sx_buffer *stg = m->stage[s->stage_cur];
    if (!s->ids_ready) prep_chunk(m, ple, s, toks, T, s->ple_rows, stg);
    s->ids_ready = false;
    if (next) {
        thread->args = (PrepArgs){m, ple, s, next, nn, s->ple_rows_b, m->stage[1 - s->stage_cur]};
        thread->on = pthread_create(&thread->th, NULL, prep_thread, &thread->args) == 0;
    }
    Enc e;
    TRY(chunk_encode(m, s, T, trace, all, stg, &e));
    s->roll = SESSION_NO_ROLL; // the chunk's start state is encoded; it ends in the live state
    if (all) {
        TRY(enc_finish(&e));
        s->pos += T;
        *logits = gpu_floats(m->vlogits);
        return OK;
    }
    TRY(enc_finish(&e));
    size_t pos0 = s->pos;
    s->pos += T;
    if (m->mtp) TRY(mtp_prefill(m, m->mtp, s, toks, T, pos0, stg));
    *logits = gpu_floats(m->logits);
    return OK;
}

static Err chunk_logits(Model *m, Session *s, const uint32_t *toks, size_t T, sx_buffer *trace, bool all,
                        float **logits) {
    return chunk_logits_next(m, s, toks, T, trace, all, NULL, 0, NULL, logits);
}

Err model_prefill(Model *m, Session *s, const uint32_t *tokens, size_t n, sx_buffer *trace, float **logits) {
    return model_prefill_with(m, s, tokens, n, trace, NULL, logits);
}

static size_t size_at(size_t i, size_t ae, size_t eq, size_t rp) { return ae == 0 ? eq : i < ae ? PT_EVERY : rp; }

Err model_prefill_with(Model *m, Session *s, const uint32_t *tokens, size_t n, sx_buffer *trace,
                       const PrefillHook *hook, float **logits) {
    skip_init();
    // fewest chunks of at most CH tokens, of equal size: a short tail chunk would pay the
    // whole per-chunk expert cost for a few tokens
    size_t nchunks = (n + CH - 1) / CH;
    size_t per = (n + nchunks - 1) / MAX(nchunks, (size_t)1);
    // aligned: PT_EVERY-token chunks up to the last point boundary, then the equal split
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
    // the n-gram row ids of chunk c + 1 (and their prefetch, one syscall per row: about
    // 200 ms per 4,096-token chunk) are computed on a thread while the GPU runs chunk c
    NextThread next = {0};
    s->ids_ready = false;
    while (i < n) {
        size_t T = MIN(size_at(i, align_end, per, r_per), n - i);
        if (next.on) {
            next_join(&next);
            uint64_t (*tmp)[NG_HEADS] = s->ple_rows;
            s->ple_rows = s->ple_rows_b;
            s->ple_rows_b = tmp;
            s->stage_cur = 1 - s->stage_cur;
            s->ids_ready = true;
        }
        size_t nx = i + T;
        size_t nT = MIN(size_at(nx, align_end, per, r_per), n - nx);
        bool has_next = nT > 0 && trace == NULL;
        Err err = chunk_logits_next(m, s, tokens + i, T, trace, false, has_next ? tokens + nx : NULL, has_next ? nT : 0,
                                    &next, &lg);
        if (!err) {
            i += T;
            if (hook && hook->chunk_done) err = hook->chunk_done(hook->ctx, s->pos);
        }
        if (err) {
            next_join(&next);
            return err;
        }
    }
    next_join(&next);
    s->ids_ready = false;
    *logits = lg;
    return OK;
}

Err model_verify_chunk(Model *m, Session *s, const uint32_t *toks, size_t n, float **logits) {
    if (!(n >= 1 && n <= KMAX)) die("model_verify_chunk: %zu tokens\n", n);
    if (m->has_pend && m->pend.T == n && m->pend.pos == s->pos && m->pend.s == s) {
        // encoded during the last draft step: write the inputs, then commit
        Enc e = m->pend.e;
        m->has_pend = false;
        const Ple *ple = first_ple(m);
        sx_buffer *stg = m->stage[s->stage_cur];
        prep_chunk(m, ple, s, toks, n, s->ple_rows, stg);
        s->ids_ready = false;
        s->roll = SESSION_NO_ROLL;
        TRY(enc_finish(&e));
        s->pos += n;
        *logits = gpu_floats(m->vlogits);
        return OK;
    }
    float *lg;
    TRY(chunk_logits(m, s, toks, n, NULL, true, &lg));
    *logits = gpu_floats(m->vlogits);
    return OK;
}

// Commits e, encodes the verification chunk of T tokens that will follow (inputs written by
// model_verify_chunk) while the GPU runs e, then waits for e.
static Err commit_preparing_verify(Model *m, Session *s, Enc *e, size_t T) {
    sx_cmd_commit(e->cmd);
    drop_pending(m);
    if (T <= KMAX && s->pos + T <= s->cap && prep_verify()) {
        bool spec0 = s->spec;
        s->spec = true;
        Enc ve;
        TRY(chunk_encode(m, s, T, NULL, true, m->stage[s->stage_cur], &ve));
        s->spec = spec0;
        m->pend = (PendVerify){.e = ve, .T = T, .pos = s->pos, .s = s};
        m->has_pend = true;
    }
    char err[512];
    if (sx_cmd_wait(e->cmd, err, sizeof err) != 0) {
        dbg("command buffer: %s\n", err);
        return "GpuFailed";
    }
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
    uint32_t prev0[2] = {s->prev[0], s->prev[1]};
    size_t since0 = s->since_eos;
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
    if (m->mtp != NULL) {
        // the verify chunk left the trunk streams of its rows in mtp_hid: rows 0..j pair
        // with the accepted drafts and then next
        memcpy(s->mtp_tok, drafts, j * sizeof(uint32_t));
        s->mtp_n = j + 1;
    }
    if (j + 1 < T) {
        // back to the state after token j of the chunk: the next pass starts from slot j
        // (Session.roll) instead of copying 113 MB back
        s->roll = j;
        s->pos = pos0 + j + 1;
        s->prev[0] = prev0[0];
        s->prev[1] = prev0[1];
        s->since_eos = since0;
        for (size_t k = 0; k < j + 1; k++) session_push_history(s, toks[k]);
    }
    s->spec = false;
    *out = (SpecResult){j, next};
    return OK;
}

Err model_draft_mtp(Model *m, Session *s, uint32_t x, size_t k, uint32_t *out, size_t *ndraft) {
    *ndraft = 0;
    const Mtp *mt = m->mtp;
    if (!mt) return OK;
    if (!s->mtp_ok || s->mtp_n == 0 || k == 0) return OK;
    size_t n = s->mtp_n;
    if (s->pos + k + 1 > s->cap || s->pos + 1 < n) return OK;
    float *bs = bf(m);
    for (size_t j = 0; j < n; j++) dequant_row(m->pack, &m->embed, j + 1 < n ? s->mtp_tok[j] : x, bs + B_a + j * H);
    Enc e = enc_begin(m);
    enc_mtp_rows(&e, mt, s, barg(m->mtp_hid, 0), n, s->pos + 1 - n, true);
    enc_copy(&e, barg(m->bscratch, B_X + (n - 1) * HCW), barg(m->mtp_chain, 0), HCW);
    if (k == 1)
        TRY(commit_preparing_verify(m, s, &e, 2));
    else
        TRY(enc_finish(&e));
    size_t nd = 0;
    out[0] = model_argmax(gpu_floats(m->mtp_logits), VOCAB);
    nd = 1;
    for (; nd < k; nd++) {
        dequant_row(m->pack, &m->embed, out[nd - 1], bs + B_a);
        Enc e2 = enc_begin(m);
        enc_mtp_rows(&e2, mt, s, barg(m->mtp_chain, 0), 1, s->pos + nd, true);
        enc_copy(&e2, barg(m->bscratch, B_X), barg(m->mtp_chain, 0), HCW);
        if (nd + 1 == k)
            TRY(commit_preparing_verify(m, s, &e2, k + 1));
        else
            TRY(enc_finish(&e2));
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
#define PLE_HIST_N (PLE_CONV_STATE * HCW)

static Err new_state(Gpu *g, sx_buffer **gdn, sx_buffer **conv, sx_buffer **ple) {
    TRY(gpu_new_buffer(g, GDN_STATE_N * 4, gdn));
    TRY(gpu_new_buffer(g, GDN_CONV_N * 4, conv));
    TRY(gpu_new_buffer(g, PLE_HIST_N * 4, ple));
    return OK;
}

Err session_create_cap(Gpu *g, bool kv_half, size_t cap, size_t max_cap, Session **out) {
    Session *s = xcalloc(1, sizeof *s);
    s->arch = &flash_arch;
    s->cap = cap;
    s->max_cap = max_cap;
    s->prev[0] = s->prev[1] = EOS_PLE;
    s->roll = SESSION_NO_ROLL;
    TRY(gpu_new_buffer(g, GDN_STATE_N * 4, &s->gdn_state));
    TRY(gpu_new_buffer(g, GDN_CONV_N * 4, &s->gdn_conv));
    s->kv_half = kv_half;
    TRY(gpu_new_buffer(g, (n_attn + 1) * 2 * cap * NKV * HD * (size_t)(kv_half ? 2 : 4), &s->kv));
    TRY(gpu_new_buffer(g, (n_attn + 1) * cap * IXD * 4, &s->ixk));
    TRY(gpu_new_buffer(g, PLE_HIST_N * 4, &s->ple_hist));
    TRY(gpu_new_buffer(g, max_cap * ROT * 4, &s->rope));
    TRY(gpu_new_buffer(g, (n_attn + 1) * (cap / BLOCK) * IXD * 4, &s->bk));
    TRY(gpu_new_buffer(g, (max_cap / BLOCK + 1) * AB * 4, &s->scores)); // [AB][cap/BLOCK] in prefill
    TRY(gpu_new_buffer(g, NLAYERS * HCW * 4, &s->trace_buf));
    s->ple_rows = xmalloc(CH * sizeof *s->ple_rows);
    s->ple_rows_b = xmalloc(CH * sizeof *s->ple_rows_b);
    TRY(gpu_new_buffer(g, KMAX * GDN_STATE_N * 4, &s->sp_gdn));
    TRY(gpu_new_buffer(g, KMAX * GDN_CONV_N * 4, &s->sp_conv));
    TRY(gpu_new_buffer(g, KMAX * PLE_HIST_N * 4, &s->sp_ple));
    for (int i = 0; i < 2; i++) {
        TRY(new_state(g, &s->ck[i].gdn, &s->ck[i].conv, &s->ck[i].ple));
        s->ck[i].prev[0] = s->ck[i].prev[1] = EOS_PLE;
    }
    float *tab = gpu_floats(s->rope);
    for (size_t pos = 0; pos < max_cap; pos++) rope_row(tab + pos * ROT, pos);
    *out = s;
    return OK;
}

size_t session_kv_bytes(const Session *s, size_t cap) {
    return (n_attn + 1) * 2 * cap * NKV * HD * session_kv_elem(s) + (n_attn + 1) * cap * IXD * 4 +
           (n_attn + 1) * (cap / BLOCK) * IXD * 4;
}

Err session_grow(Session *s, Gpu *g, size_t cap) {
    if (!(cap >= s->cap && cap <= s->max_cap && cap % BLOCK == 0)) die("session_grow: bad capacity %zu\n", cap);
    if (cap == s->cap) return OK;
    size_t old = s->cap;
    size_t kvrow = NKV * HD * session_kv_elem(s);
    sx_buffer *kv, *ixk, *bk;
    TRY(gpu_new_buffer(g, (n_attn + 1) * 2 * cap * kvrow, &kv));
    TRY(gpu_new_buffer(g, (n_attn + 1) * cap * IXD * 4, &ixk));
    TRY(gpu_new_buffer(g, (n_attn + 1) * (cap / BLOCK) * IXD * 4, &bk));
    uint8_t *dkv = sx_buffer_contents(kv);
    const uint8_t *skv = sx_buffer_contents(s->kv);
    for (size_t plane = 0; plane < (n_attn + 1) * 2; plane++)
        memcpy(dkv + plane * cap * kvrow, skv + plane * old * kvrow, old * kvrow);
    float *dix = gpu_floats(ixk), *six = gpu_floats(s->ixk);
    float *dbk = gpu_floats(bk), *sbk = gpu_floats(s->bk);
    for (size_t l = 0; l < n_attn + 1; l++) {
        memcpy(dix + l * cap * IXD, six + l * old * IXD, old * IXD * 4);
        memcpy(dbk + l * (cap / BLOCK) * IXD, sbk + l * (old / BLOCK) * IXD, (old / BLOCK) * IXD * 4);
    }
    sx_buffer_release(s->kv);
    sx_buffer_release(s->ixk);
    sx_buffer_release(s->bk);
    s->kv = kv;
    s->ixk = ixk;
    s->bk = bk;
    s->cap = cap;
    return OK;
}

// The live copy of a recurrent state (n floats): cur, or its slot after a rollback.
static const float *live(const Session *s, sx_buffer *cur, sx_buffer *slots, size_t n) {
    return s->roll != SESSION_NO_ROLL ? gpu_floats(slots) + s->roll * n : gpu_floats(cur);
}

static void vec_set(VecU32 *v, const uint32_t *p, size_t n) {
    v->len = 0;
    vec_reserve(v, n);
    if (n) memcpy(v->ptr, p, n * sizeof(uint32_t));
    v->len = n;
}

Err session_save_checkpoint(Session *s, const uint32_t *toks, size_t n) {
    return session_save_checkpoint_at(s, 0, toks, n);
}

Err session_save_checkpoint_at(Session *s, size_t slot, const uint32_t *toks, size_t n) {
    if (n != s->pos) die("session_save_checkpoint_at: %zu tokens at position %zu\n", n, s->pos);
    Ckpt *k = &s->ck[slot];
    memcpy(gpu_floats(k->gdn), live(s, s->gdn_state, s->sp_gdn, GDN_STATE_N), GDN_STATE_N * 4);
    memcpy(gpu_floats(k->conv), live(s, s->gdn_conv, s->sp_conv, GDN_CONV_N), GDN_CONV_N * 4);
    memcpy(gpu_floats(k->ple), live(s, s->ple_hist, s->sp_ple, PLE_HIST_N), PLE_HIST_N * 4);
    k->prev[0] = s->prev[0];
    k->prev[1] = s->prev[1];
    k->since_eos = s->since_eos;
    vec_set(&k->tokens, toks, n);
    return OK;
}

size_t session_reusable(Session *s, const uint32_t *prompt, size_t n) {
    Reusable r = session_reusable_in(s->ck, prompt, n);
    s->ck_pick = r.pick;
    return r.n;
}

size_t session_conv_bytes(const Session *s, size_t cap) {
    size_t ckb = GDN_STATE_N * 4 + GDN_CONV_N * 4 + PLE_HIST_N * 4;
    return session_kv_bytes(s, cap) + 2 * ckb;
}

Err session_new_conv(const Session *s, Gpu *g, size_t cap, Conv *out) {
    Conv cv = {0};
    cv.cap = cap;
    TRY(gpu_new_buffer(g, (n_attn + 1) * 2 * cap * NKV * HD * session_kv_elem(s), &cv.kv));
    TRY(gpu_new_buffer(g, (n_attn + 1) * cap * IXD * 4, &cv.ixk));
    TRY(gpu_new_buffer(g, (n_attn + 1) * (cap / BLOCK) * IXD * 4, &cv.bk));
    for (int i = 0; i < 2; i++) {
        TRY(new_state(g, &cv.ck[i].gdn, &cv.ck[i].conv, &cv.ck[i].ple));
        cv.ck[i].prev[0] = cv.ck[i].prev[1] = EOS_PLE;
    }
    *out = cv;
    return OK;
}

Err session_conv_shrink(const Session *s, Gpu *g, Conv *cv, size_t cap) {
    Conv small;
    TRY(session_new_conv(s, g, cap, &small));
    sx_buffer_release(cv->kv);
    sx_buffer_release(cv->ixk);
    sx_buffer_release(cv->bk);
    cv->cap = small.cap;
    cv->kv = small.kv;
    cv->ixk = small.ixk;
    cv->bk = small.bk;
    for (int k = 0; k < 2; k++) {
        sx_buffer_release(small.ck[k].gdn);
        sx_buffer_release(small.ck[k].conv);
        sx_buffer_release(small.ck[k].ple);
    }
    for (int k = 0; k < 2; k++) cv->ck[k].tokens.len = 0;
    cv->ptoks.len = 0;
    return OK;
}

#define SWAP(T, a, b) do { T t_ = (a); (a) = (b); (b) = t_; } while (0)

void session_swap_conv(Session *s, Conv *cv) {
    SWAP(size_t, s->cap, cv->cap);
    SWAP(sx_buffer *, s->kv, cv->kv);
    SWAP(sx_buffer *, s->ixk, cv->ixk);
    SWAP(sx_buffer *, s->bk, cv->bk);
    for (int i = 0; i < 2; i++) SWAP(Ckpt, s->ck[i], cv->ck[i]);
    for (int i = 0; i < PT_MAX; i++) SWAP(Point, s->pts[i], cv->pts[i]);
    SWAP(VecU32, s->ptoks, cv->ptoks);
    s->roll = SESSION_NO_ROLL;
    s->mtp_ok = false;
    s->mtp_n = 0;
}

bool session_roll_to(Session *s, size_t p, const uint32_t *toks) {
    if (s->chunk_t == 0 || p <= s->chunk_pos0 || p > s->chunk_pos0 + s->chunk_t || p > s->pos) return false;
    s->roll = p - s->chunk_pos0 - 1;
    s->pos = p;
    s->prev[0] = p >= 2 ? toks[p - 2] : EOS_PLE;
    s->prev[1] = toks[p - 1];
    size_t since = 0;
    for (size_t i = p; i > 0 && toks[i - 1] != EOS_PLE; i--) since += 1;
    s->since_eos = since;
    s->mtp_ok = false;
    s->mtp_n = 0;
    return true;
}


Err session_save_point(Session *s, Gpu *g, const uint32_t *toks, size_t n) {
    if (!(s->pos % PT_EVERY == 0 && n == s->pos)) die("session_save_point: %zu tokens at position %zu\n", n, s->pos);
    size_t k = s->pos / PT_EVERY - 1;
    if (k >= PT_MAX) return OK;
    Point *pt = &s->pts[k];
    if (pt->gdn == NULL) {
        Point np = {.prev = {EOS_PLE, EOS_PLE}};
        TRY(new_state(g, &np.gdn, &np.conv, &np.ple));
        *pt = np;
    }
    memcpy(gpu_floats(pt->gdn), live(s, s->gdn_state, s->sp_gdn, GDN_STATE_N), GDN_STATE_N * 4);
    memcpy(gpu_floats(pt->conv), live(s, s->gdn_conv, s->sp_conv, GDN_CONV_N), GDN_CONV_N * 4);
    memcpy(gpu_floats(pt->ple), live(s, s->ple_hist, s->sp_ple, PLE_HIST_N), PLE_HIST_N * 4);
    pt->prev[0] = s->prev[0];
    pt->prev[1] = s->prev[1];
    pt->since_eos = s->since_eos;
    vec_set(&s->ptoks, toks, n);
    return OK;
}

// The recurrent state and history of point pt into the live state, at position n.
static void start_at(Session *s, const Point *pt, size_t n) {
    s->roll = SESSION_NO_ROLL;
    memcpy(gpu_floats(s->gdn_state), gpu_floats(pt->gdn), GDN_STATE_N * 4);
    memcpy(gpu_floats(s->gdn_conv), gpu_floats(pt->conv), GDN_CONV_N * 4);
    memcpy(gpu_floats(s->ple_hist), gpu_floats(pt->ple), PLE_HIST_N * 4);
    s->prev[0] = pt->prev[0];
    s->prev[1] = pt->prev[1];
    s->since_eos = pt->since_eos;
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
    size_t kvrow = NKV * HD * session_kv_elem(s);
    uint8_t *dkv = sx_buffer_contents(s->kv);
    const uint8_t *skv = sx_buffer_contents(src->kv);
    if (!(n <= s->cap && n <= src->cap)) die("session_start_from: position %zu past capacity\n", n);
    for (size_t plane = 0; plane < (n_attn + 1) * 2; plane++)
        memcpy(dkv + plane * s->cap * kvrow, skv + plane * src->cap * kvrow, n * kvrow);
    float *dix = gpu_floats(s->ixk), *six = gpu_floats(src->ixk);
    float *dbk = gpu_floats(s->bk), *sbk = gpu_floats(src->bk);
    for (size_t l = 0; l < n_attn + 1; l++) {
        memcpy(dix + l * s->cap * IXD, six + l * src->cap * IXD, n * IXD * 4);
        size_t nb = n / BLOCK;
        memcpy(dbk + l * (s->cap / BLOCK) * IXD, sbk + l * (src->cap / BLOCK) * IXD, nb * IXD * 4);
    }
    size_t k = n / PT_EVERY - 1;
    start_at(s, &src->pts[k], n);
    // this conversation's own points up to n, so it can be shared from in turn
    for (size_t i = 0; i < k + 1; i++) {
        const Point *sp = &src->pts[i];
        if (sp->gdn == NULL) continue;
        Point *d = &s->pts[i];
        if (d->gdn == NULL) {
            Point np = {.prev = {EOS_PLE, EOS_PLE}};
            TRY(new_state(g, &np.gdn, &np.conv, &np.ple));
            *d = np;
        }
        memcpy(gpu_floats(d->gdn), gpu_floats(sp->gdn), GDN_STATE_N * 4);
        memcpy(gpu_floats(d->conv), gpu_floats(sp->conv), GDN_CONV_N * 4);
        memcpy(gpu_floats(d->ple), gpu_floats(sp->ple), PLE_HIST_N * 4);
        d->prev[0] = sp->prev[0];
        d->prev[1] = sp->prev[1];
        d->since_eos = sp->since_eos;
    }
    vec_set(&s->ptoks, src->ptoks.ptr, n);
    return OK;
}

void session_restore(Session *s) {
    s->roll = SESSION_NO_ROLL;
    Ckpt *k = &s->ck[s->ck_pick];
    memcpy(gpu_floats(s->gdn_state), gpu_floats(k->gdn), GDN_STATE_N * 4);
    memcpy(gpu_floats(s->gdn_conv), gpu_floats(k->conv), GDN_CONV_N * 4);
    memcpy(gpu_floats(s->ple_hist), gpu_floats(k->ple), PLE_HIST_N * 4);
    s->prev[0] = k->prev[0];
    s->prev[1] = k->prev[1];
    s->since_eos = k->since_eos;
    s->pos = k->tokens.len;
    s->mtp_ok = false; // the trunk stream before the checkpoint is not kept
    s->mtp_n = 0;
}

void session_push_history(Session *s, uint32_t token) {
    s->prev[0] = s->prev[1];
    s->prev[1] = token;
    s->since_eos = token == EOS_PLE ? 0 : s->since_eos + 1;
}

void session_reset(Session *s) {
    s->roll = SESSION_NO_ROLL;
    s->pos = 0;
    s->prev[0] = s->prev[1] = EOS_PLE;
    s->since_eos = 0;
    s->mtp_ok = false;
    s->mtp_n = 0;
    memset(gpu_floats(s->gdn_state), 0, GDN_STATE_N * 4);
    memset(gpu_floats(s->gdn_conv), 0, GDN_CONV_N * 4);
    memset(gpu_floats(s->ple_hist), 0, PLE_HIST_N * 4);
}

// ---- loading helpers ---------------------------------------------------------------------------

static Err lin(Arena *a, const Pack *pack, const char *p, const char *name, Linear *out) {
    return pack_linear(pack, str(cat(a, p, name)), out);
}

// A BF16 tensor as f32 in the GPU parameter buffer.
static Err vec(Arena *a, const Pack *pack, Params *P, const char *p, const char *name, Vec *out) {
    const char *full = cat(a, p, name);
    const TensorInfo *t = pack_get(pack, str(full));
    if (!t) {
        dbg("missing tensor %s\n", full);
        return "MissingTensor";
    }
    if (t->dtype != DTYPE_BF16) return "UnexpectedDType";
    const uint8_t *raw = pack_data(pack, t);
    TRY(params_take(P, (size_t)tensor_numel(t), out));
    for (size_t i = 0; i < out->n; i++) out->s[i] = ops_bf16_to_f32((uint16_t)(raw[i * 2] | (raw[i * 2 + 1] << 8)));
    return OK;
}

// Bits of the MTP head's requantized dense matrices: 4 (runs on the accelerators in prompt
// reading). The 8-bit and exact BF16 variants were removed: 8-bit crashed on the full pack,
// BF16 changed greedy output at 4K and accepted no more drafts at 16K.
#define MTP_BITS 4u

// Load-time requantization arena: BF16 matrices the kernels cannot read (the MTP head's dense
// weights, the Coder's routers) quantized into one GPU buffer registered as a "file".
static struct {
    bool set;
    uint16_t file;
    uint8_t *dst;
    size_t len;
    uint64_t at;
} rq_cur;

// name as a Linear: the pack's own quantized tensor when it has one (.scales present), else
// its BF16 matrix quantized here to bits (group 64 for 8-bit, 32 for 4-bit).
static Err rq_lin(Arena *arena, const Pack *pack, Gpu *g, const char *name, uint32_t bits, Linear *out) {
    char nb[256];
    if ((size_t)snprintf(nb, sizeof nb, "%s.scales", name) >= sizeof nb) return "NoSpaceLeft";
    if (pack_get(pack, str(nb)) != NULL) return pack_linear(pack, str(name), out);
    if ((size_t)snprintf(nb, sizeof nb, "%s.weight", name) >= sizeof nb) return "NoSpaceLeft";
    const TensorInfo *t = pack_get(pack, str(nb));
    if (!t) return "MissingTensor";
    if (t->dtype != DTYPE_BF16 || t->rank != 2) return "UnexpectedDType";
    if (!rq_cur.set) {
        size_t size = (size_t)256 << 20;
        sx_buffer *buf;
        TRY(gpu_new_buffer(g, size, &buf));
        uint16_t file;
        TRY(gpu_add_file(g, arena, buf, &file));
        rq_cur.set = true;
        rq_cur.file = file;
        rq_cur.dst = sx_buffer_contents(buf);
        rq_cur.len = size;
        rq_cur.at = 0;
    }
    if (bits == 16) {
        // kept exact: bits 16 marks a plain BF16 matrix, copied here 16-byte aligned so
        // mv_bf16 reads 8 values per load (pack tensors can start at odd offsets)
        const uint8_t *raw = pack_data(pack, t);
        size_t rlen = (size_t)t->nbytes;
        if (rq_cur.at + rlen > rq_cur.len) return "RequantBufferFull";
        memcpy(rq_cur.dst + rq_cur.at, raw, rlen);
        TensorInfo tt = *t;
        tt.file = rq_cur.file;
        tt.offset = rq_cur.at;
        rq_cur.at = ALIGN_UP(rq_cur.at + rlen, 16);
        *out = (Linear){.weight = tt, .scales = tt, .biases = tt, .bits = 16, .group_size = 0,
                        .out_features = t->shape[0], .in_features = t->shape[1]};
        return OK;
    }
    Quantized r = weights_quantize(pack_data(pack, t), t->shape[0], t->shape[1], bits, bits == 8 ? 64 : 32, rq_cur.dst,
                                   rq_cur.at, rq_cur.file);
    rq_cur.at = ALIGN_UP(rq_cur.at + r.used, 16);
    if (rq_cur.at > rq_cur.len) return "RequantBufferFull";
    *out = r.lin;
    return OK;
}

static Err load_mtp(Arena *arena, const Pack *pack, Gpu *g, Params *P, Mtp *mt) {
    uint64_t at0 = rq_cur.set ? rq_cur.at : 0;
#define MTPQ(name, out) TRY(rq_lin(arena, pack, g, (name), MTP_BITS, (out)))
#define MP "mtp.layers.0."
    *mt = (Mtp){0};
    MTPQ("mtp.fc_embedding", &mt->fc_e);
    MTPQ("mtp.fc_hidden", &mt->fc_h);
    TRY(vec(arena, pack, P, "mtp.", "pre_fc_norm_embedding.weight", &mt->n_e));
    TRY(vec(arena, pack, P, "mtp.", "pre_fc_norm_hidden.weight", &mt->n_h));
    TRY(load_residual(arena, pack, P, "mtp.hyper_connection_mixer", false, &mt->mixer));
    Layer *L = &mt->layer;
    TRY(load_residual(arena, pack, P, cat(arena, MP, "attn_hyper_connection"), true, &L->attn_hc));
    TRY(load_residual(arena, pack, P, cat(arena, MP, "mlp_hyper_connection"), true, &L->mlp_hc));
    L->gdn = NULL;
    L->ple = NULL;
    Attn *A = ARENA_NEW(arena, Attn);
    A->slot = n_attn;
    MTPQ(MP "self_attn.q_proj", &A->q);
    MTPQ(MP "self_attn.k_proj", &A->k);
    MTPQ(MP "self_attn.v_proj", &A->v);
    MTPQ(MP "self_attn.o_proj", &A->o);
    TRY(vec(arena, pack, P, MP, "self_attn.q_norm.weight", &A->q_norm));
    TRY(vec(arena, pack, P, MP, "self_attn.k_norm.weight", &A->k_norm));
    MTPQ(MP "self_attn.indexer.index_qk_proj", &A->ix);
    TRY(vec(arena, pack, P, MP, "self_attn.indexer.q_layernorm.weight", &A->ix_qn));
    TRY(vec(arena, pack, P, MP, "self_attn.indexer.k_layernorm.weight", &A->ix_kn));
    L->attn = A;
    Moe *E = &L->moe;
    E->ne = NE;
    TRY(rq_lin(arena, pack, g, MP "mlp.gate", 8, &E->router)); // 8-bit like the trunk's routers
    TRY(rq_lin(arena, pack, g, MP "mlp.shared_expert_gate", 8, &E->shared_gate));
    MTPQ(MP "mlp.shared_expert.gate_proj", &E->sg);
    MTPQ(MP "mlp.shared_expert.up_proj", &E->su);
    MTPQ(MP "mlp.shared_expert.down_proj", &E->sd);
    TRY(lin(arena, pack, MP, "mlp.switch_mlp.gate_proj", &E->eg));
    TRY(lin(arena, pack, MP, "mlp.switch_mlp.up_proj", &E->eu));
    TRY(lin(arena, pack, MP, "mlp.switch_mlp.down_proj", &E->ed));
    E->ne = (uint32_t)E->router.out_features;
#undef MTPQ
#undef MP
    uint64_t at = (rq_cur.set ? rq_cur.at : 0) - at0;
    char mb[64];
    fmt_fixed(mb, sizeof mb, (double)at / 1e6, 1);
    dbg("model: MTP head loaded, %s MB requantized to %u-bit\n", mb, MTP_BITS);
    return OK;
}

static Err load_residual(Arena *a, const Pack *pack, Params *P, const char *p, bool inject, Residual *out) {
    *out = (Residual){0};
    TRY(vec(a, pack, P, p, ".hc_norm.weight", &out->norm));
    TRY(lin(a, pack, p, ".input_mix_weight_down", &out->down));
    TRY(lin(a, pack, p, ".input_mix_weight_up", &out->up));
    if (inject) {
        TRY(lin(a, pack, p, ".block_inject_weight", &out->inject));
        out->has_inject = true;
    }
    return OK;
}

static Err read_i64(const Pack *pack, const char *name, int64_t *out, size_t n) {
    const TensorInfo *t = pack_get(pack, str(name));
    if (!t) return "MissingTensor";
    if (t->dtype != DTYPE_I64 || tensor_numel(t) != n) return "UnexpectedDType";
    const uint8_t *raw = pack_data(pack, t);
    for (size_t i = 0; i < n; i++) memcpy(&out[i], raw + i * 8, 8); // little-endian host
    return OK;
}

static Err load_ple(Arena *a, const Pack *pack, Params *P, const char *p, Ple *L) {
    TRY(lin(a, pack, p, "ple.key_proj", &L->key));
    TRY(lin(a, pack, p, "ple.value_proj", &L->value));
    TRY(vec(a, pack, P, p, "ple.norm_key.weight", &L->norm_key));
    TRY(vec(a, pack, P, p, "ple.norm_query.weight", &L->norm_query));
    TRY(vec(a, pack, P, p, "ple.norm_conv.weight", &L->norm_conv));
    TRY(vec(a, pack, P, p, "ple.conv1d.weight", &L->conv));
    TRY(read_i64(pack, cat(a, p, "ple.ple_embedding.layer_multipliers"), L->mult, COUNT(L->mult)));
    TRY(read_i64(pack, cat(a, p, "ple.ple_embedding.ngram_heads_offsets"), L->offsets, COUNT(L->offsets)));
    TRY(read_i64(pack, cat(a, p, "ple.ple_embedding.ngram_heads_vocab_sizes"), L->vocab, COUNT(L->vocab)));
    for (size_t i = 0; i < 128; i++) {
        const char *base = arena_printf(a, "%sple.ple_embedding.ngram_embedding.shards.%zu", p, i);
        const TensorInfo *t;
        if (!(t = pack_get(pack, str(cat(a, base, ".weight"))))) return "MissingTensor";
        L->shard_w[i] = *t;
        if (!(t = pack_get(pack, str(cat(a, base, ".scales"))))) return "MissingTensor";
        L->shard_s[i] = *t;
        if (!(t = pack_get(pack, str(cat(a, base, ".biases"))))) return "MissingTensor";
        L->shard_b[i] = *t;
    }
    return OK;
}

// Dequantizes row r of an affine quantized matrix into out[in_features].
static void dequant_row(const Pack *pack, const Linear *l, size_t r, float *out) {
    // MLX affine packing is a little-endian bitstream per row: value i at bit i * bits, for any
    // bits in 2..8 (5 and 6 cross bytes; checked against mx.dequantize)
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

// ---- LoRA (27B only, docs/dense-lora.md) -------------------------------------------------------

Err model_lora_load(Arena *a, Model *m, const char *name, const char *dir) {
    (void)a, (void)m, (void)name, (void)dir;
    return "LoraUnsupported";
}
int model_lora_find(const Model *m, const char *name) {
    (void)m, (void)name;
    return -1;
}
void model_lora_select(Model *m, int idx) { (void)m, (void)idx; }
int model_lora_selected(const Model *m) {
    (void)m;
    return -1;
}
size_t model_lora_count(const Model *m) {
    (void)m;
    return 0;
}
const char *model_lora_name(const Model *m, size_t i) {
    (void)m, (void)i;
    return NULL;
}


// ---- the table model_api.c dispatches through ----------------------------------------------

bool model_has_mtp(const Model *m) { return m->mtp != NULL; }
bool model_fast_prefill(const Model *m) { return m->fast_prefill; }
bool model_fast_prefill_available(const Model *m) { return m->mpp != NULL; }
void model_set_fast_prefill(Model *m, bool on) { m->fast_prefill = on && m->mpp != NULL; }

typedef struct {
    const char *module;
    uint64_t out, in;
} Expect;
typedef VEC(Expect) VecExpect;

static void add_lin(VecExpect *b, Arena *a, size_t l, const char *name, uint64_t o, uint64_t i) {
    vec_push(b, ((Expect){arena_printf(a, "%slayers.%zu.%s", WEIGHTS_PREFIX, l, name), o, i}));
}

static void expected_linears(VecExpect *buf, Arena *a, size_t layer) {
    const uint64_t h = CONFIG_HIDDEN_SIZE;
    const uint64_t hcw = CONFIG_HC_COUNT * h;
    const char *hcs[2] = {"attn_hyper_connection", "mlp_hyper_connection"};
    for (int k = 0; k < 2; k++) {
        add_lin(buf, a, layer, arena_printf(a, "%s.input_mix_weight_down", hcs[k]), CONFIG_HC_LOWRANK, hcw);
        add_lin(buf, a, layer, arena_printf(a, "%s.input_mix_weight_up", hcs[k]), hcw, CONFIG_HC_LOWRANK);
        add_lin(buf, a, layer, arena_printf(a, "%s.block_inject_weight", hcs[k]), CONFIG_HC_COUNT, hcw);
    }
    if (config_layer_kinds[layer] == LAYER_KIND_GDN) {
        const uint64_t kd = CONFIG_LINEAR_NUM_KEY_HEADS * CONFIG_LINEAR_KEY_HEAD_DIM;
        const uint64_t vd = CONFIG_LINEAR_NUM_VALUE_HEADS * CONFIG_LINEAR_VALUE_HEAD_DIM;
        add_lin(buf, a, layer, "linear_attn.in_proj_qkv", 2 * kd + vd, h);
        add_lin(buf, a, layer, "linear_attn.in_proj_z", vd, h);
        add_lin(buf, a, layer, "linear_attn.in_proj_a", CONFIG_LINEAR_NUM_VALUE_HEADS, h);
        add_lin(buf, a, layer, "linear_attn.in_proj_b", CONFIG_LINEAR_NUM_VALUE_HEADS, h);
        add_lin(buf, a, layer, "linear_attn.out_proj", h, vd);
    } else {
        const uint64_t qd = CONFIG_NUM_ATTENTION_HEADS * CONFIG_HEAD_DIM;
        const uint64_t kvd = CONFIG_NUM_KEY_VALUE_HEADS * CONFIG_HEAD_DIM;
        add_lin(buf, a, layer, "self_attn.q_proj", 2 * qd, h);
        add_lin(buf, a, layer, "self_attn.k_proj", kvd, h);
        add_lin(buf, a, layer, "self_attn.v_proj", kvd, h);
        add_lin(buf, a, layer, "self_attn.o_proj", h, qd);
        add_lin(buf, a, layer, "self_attn.indexer.index_qk_proj",
                (CONFIG_INDEXER_N_HEADS + CONFIG_INDEXER_KV_HEADS) * CONFIG_INDEXER_HEAD_DIM, h);
    }
    add_lin(buf, a, layer, "mlp.gate", CONFIG_NUM_EXPERTS, h);
    add_lin(buf, a, layer, "mlp.shared_expert_gate", 1, h);
    add_lin(buf, a, layer, "mlp.shared_expert.gate_proj", CONFIG_SHARED_EXPERT_INTERMEDIATE_SIZE, h);
    add_lin(buf, a, layer, "mlp.shared_expert.up_proj", CONFIG_SHARED_EXPERT_INTERMEDIATE_SIZE, h);
    add_lin(buf, a, layer, "mlp.shared_expert.down_proj", h, CONFIG_SHARED_EXPERT_INTERMEDIATE_SIZE);
    add_lin(buf, a, layer, "mlp.switch_mlp.gate_proj", CONFIG_MOE_INTERMEDIATE_SIZE, h);
    add_lin(buf, a, layer, "mlp.switch_mlp.up_proj", CONFIG_MOE_INTERMEDIATE_SIZE, h);
    add_lin(buf, a, layer, "mlp.switch_mlp.down_proj", h, CONFIG_MOE_INTERMEDIATE_SIZE);
    for (size_t i = 0; i < CONFIG_N_PLE_LAYERS; i++)
        if (config_ple_layers[i] == layer) {
            add_lin(buf, a, layer, "ple.key_proj", hcw, h);
            add_lin(buf, a, layer, "ple.value_proj", h, h);
        }
}

size_t model_check_pack(Arena *a, const Pack *pack, FILE *out) {
    fprintf(out, "model: Qwen3.8-Flash-Next (%s), %zu layers, hidden %zu, %zu experts top-%zu\n", CONFIG_MODEL_TYPE,
            CONFIG_NUM_HIDDEN_LAYERS, CONFIG_HIDDEN_SIZE, CONFIG_NUM_EXPERTS, CONFIG_NUM_EXPERTS_PER_TOK);
    VecExpect exp = {0};
    for (size_t l = 0; l < CONFIG_NUM_HIDDEN_LAYERS; l++) expected_linears(&exp, a, l);
    vec_push(&exp, ((Expect){"language_model.lm_head", CONFIG_VOCAB_SIZE, CONFIG_HIDDEN_SIZE}));
    vec_push(&exp, ((Expect){WEIGHTS_PREFIX "embed_tokens", CONFIG_VOCAB_SIZE, CONFIG_HIDDEN_SIZE}));
    size_t bad = 0;
    for (size_t i = 0; i < exp.len; i++) {
        Expect e = exp.ptr[i];
        Linear lin;
        Err err = pack_linear(pack, str(e.module), &lin);
        if (err) {
            // a plain BF16 matrix (the Coder routers) is requantized at load: accept it
            const TensorInfo *t = pack_get(pack, str(arena_printf(a, "%s.weight", e.module)));
            if (t && t->dtype == DTYPE_BF16 && t->rank == 2) continue;
            fprintf(out, "  MISSING %s: %s\n", e.module, err);
            bad += 1;
            continue;
        }
        if (lin.out_features != e.out || lin.in_features != e.in) {
            fprintf(out, "  SHAPE %s: got %llux%llu, want %llux%llu\n", e.module, (unsigned long long)lin.out_features,
                    (unsigned long long)lin.in_features, (unsigned long long)e.out, (unsigned long long)e.in);
            bad += 1;
        }
    }
    fprintf(out, "linears: %zu checked, %zu problems\n", exp.len, bad);
    vec_free(&exp);
    return bad;
}

const ModelArch flash_arch = MODEL_ARCH_TABLE({
    .model_type = CONFIG_MODEL_TYPE,
    .id = "qwen3.8-flash-next",
    .name = "Qwen3.8-Flash-Next",
    .layers = CONFIG_NUM_HIDDEN_LAYERS,
    .width = HCW,
    .vocab = CONFIG_VOCAB_SIZE,
    .hidden = CONFIG_HIDDEN_SIZE,
    .eos = config_eos_token_ids,
    .n_eos = CONFIG_N_EOS_TOKEN_IDS,
});
