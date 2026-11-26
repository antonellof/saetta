// Inkling 2 Mini forward pass (HF thinkingmachines/Inkling-2-Mini b85c3cf, config.json
// model_type inkling2_loop) on the MLX pack mlx-community/Inkling-2-Mini-4bit (dense linears
// affine 4-bit group 64, experts NVFP4 on layers 0-19 and MXFP8 on 20-23). Exports inkling_arch
// (model_arch.h). Formulas follow transformers be48e7a models/inkling2_loop/
// modeling_inkling2_loop.py (M) and are checked against a numpy float64 reference (itself checked
// against transformers on a random model) by `saetta verify`. Constants: src/config_inkling.h.
//
// 24 layers, 48 visits per token: layers 0-5, the loop core 6-17 three times, then 18-23
// (M:1350-1440). Each visit has its own KV cache and its own four conv states (the passes use
// copies of the core's cache layers, M:1354-1357): the 8 visits of the full layers (5, 11, 17, 23)
// keep every position, the 40 sliding visits a ring of the window's 1024 rows. A chunk's keys and
// values go to the ring only after its attention (src/kernels/inkling_attn.metal), so chunks
// longer than the window never read rows overwritten by their own later tokens.
// The attention residuals keep up to 5 block states per token, within the token's pass only.
// One command buffer per chunk (decode: T = 1); all activations f32; the dense linears run on
// the run-time affine kernels (src/qgen.h), the rest on src/kernels/inkling_*.metal.
#include "model.h"

#include "config_inkling.h"
#define ARCH_PFX inkling_
#include "model_rename.h"

#include <math.h>
#include <stdlib.h>

#include "inkling.h"
#include "ops.h"
#include "qgen.h"
#include "tolerances.h"

#define H CONFIG_HIDDEN_SIZE
#define NL CONFIG_NUM_HIDDEN_LAYERS
#define NH CONFIG_NUM_ATTENTION_HEADS
#define NKV CONFIG_NUM_KEY_VALUE_HEADS
#define HD CONFIG_HEAD_DIM
#define QD (NH * HD)
#define KVD (NKV * HD)
#define DREL CONFIG_D_REL
#define EXT CONFIG_REL_EXTENT
#define WIN CONFIG_SLIDING_WINDOW_SIZE
#define NE CONFIG_N_ROUTED_EXPERTS
#define TOPK CONFIG_NUM_EXPERTS_PER_TOK
#define FF CONFIG_INTERMEDIATE_SIZE
#define VOCAB CONFIG_UNPADDED_VOCAB_SIZE
#define LS CONFIG_LOOP_CORE_START
#define LE CONFIG_LOOP_CORE_END
#define NP CONFIG_LOOP_NUM_PASSES
#define RANK CONFIG_LOOP_GATE_RANK
#define NBLK CONFIG_N_LAYERS_PER_ATTN_RES_BLOCK
#define EPS CONFIG_RMS_NORM_EPS
#define NVIS (LS + (LE - LS) * NP + (NL - LE)) // 48
#define NLOOP (LE - LS)
#define MAXBS 5 // block states: embedding and 3 more, plus one appended inside a boundary layer
static_assert(CONFIG_SWA_HEAD_DIM == HD && CONFIG_SWA_NUM_ATTENTION_HEADS == NH && CONFIG_SWA_NUM_KEY_VALUE_HEADS == NKV);
static_assert(HD == 128 && DREL <= 32 && CONFIG_SCONV_KERNEL_SIZE == 4 && NE <= 128 && TOPK <= 16);
static_assert(WIN == EXT); // sliding layers use the window as their relative extent (M:557)
static_assert(H % 256 == 0 && FF % 32 == 0 && LS % NBLK == 0 && LE % NBLK == 0);
// The loop as the forward pass assumes it (enc_forward): the core starts on a block boundary
// after the first block, holds exactly two blocks (so a later pass's messages read one block
// state and the partial, M:1400, 1407), and the block states never exceed MAXBS, nor the mix
// kernel's IK_MAXS (6: MAXBS states and the partial) or IK_MAXQ (4 message queries).
static_assert(LS == NBLK && (LE - LS) == 2 * NBLK && NP >= 1);
static_assert(NL / NBLK + 1 <= MAXBS && MAXBS + 1 <= 6);
static_assert(CONFIG_N_SHARED_EXPERTS == 0 && CONFIG_USE_GATE_BIAS && NE % 32 == 0 && RANK % 8 == 0);

// Prompt chunk: tokens per command buffer.
#define PCH 256
// Attention splits: decode up to NSPLIT_DEC, chunks NSPLIT_PRE.
#define NSPLIT_DEC 64
#define NSPLIT_PRE 4

// ---- the visit schedule -------------------------------------------------------------------
static uint8_t vis_layer[NVIS], vis_pass[NVIS], vis_kv[NVIS]; // kv: index among full / sliding visits
static size_t n_full_vis, n_swa_vis;
static bool is_full(size_t li) { return config_layer_kinds[li] == LAYER_KIND_GLOBAL; }
__attribute__((constructor)) static void schedule(void) {
    size_t v = 0;
    for (size_t li = 0; li < LS; li++) vis_layer[v] = (uint8_t)li, vis_pass[v++] = 0;
    for (size_t p = 0; p < NP; p++)
        for (size_t li = LS; li < LE; li++) vis_layer[v] = (uint8_t)li, vis_pass[v++] = (uint8_t)p;
    for (size_t li = LE; li < NL; li++) vis_layer[v] = (uint8_t)li, vis_pass[v++] = 0;
    for (v = 0; v < NVIS; v++) vis_kv[v] = (uint8_t)(is_full(vis_layer[v]) ? n_full_vis++ : n_swa_vis++);
}

typedef struct {
    bool full, mx;
    Linear q, k, v, o, g, sink, r, kd, vd; // kd, vd: the dynamic conv projections (full layers)
    sx_arg n_in, n_q, n_k, n_v, n_post;     // RMSNorm weights (sliding: n_in; full: n_q, n_k, n_v)
    sx_arg kw, vw, aw, mw;                  // conv taps [C][4], lag order (0: the token itself)
    sx_arg qscale, rscale, proj;            // [NH][HD], [NH][DREL], [DREL][EXT]
    sx_arg aq, mq;                          // attention_input.query [1 or 3][H] (none on layer 0), mlp_input.query [H]
    sx_arg rw, rb;                          // router [NE][H], bias [NE]
    sx_arg gfu, gfd;                        // NVFP4 per-expert factors global_scale / 2688 [NE]
    TensorInfo guw, gus, dnw, dns;          // experts: gate_up and down weights, scales
    sx_arg md[4], mf[4], mg[4];             // message gates q, k, v, mlp: down bf16 [RANK][2H], forget, gain f32 [H][RANK]
} ILayer;

typedef struct {
    sx_pipeline *mix, *norm, *msg_down, *msg_out, *qr, *kv_conv, *conv_res, *conv_state, *router, *moe_sum, *softcap, *copy,
        *kv_store, *attn_part, *attn_merge, *gu_nv, *dn_nv, *gu_mx, *dn_mx, *plan, *gu_nv_g, *gu_mx_g, *dn_nv_g,
        *dn_mx_g, *router_lg;
    sx_pipeline *mv, *mm; // qgen: qmatvec_gen, qmm_gen
} Pipes;

// Scratch regions, floats per token (each region holds PCH rows).
enum { R_H, R_HN, R_BS, R_P0, R_MS, R_MSG, R_MIX, R_G, R_XN, R_Q, R_K, R_V, R_GT, R_SK, R_RS, R_KD, R_VD, R_KN, R_VN, R_ATT, R_A,
       R_R, R_HM, R_XM, R_D, R_IDX, R_W, R_HACT, R_YS, R_YSUM, R_PAIRS, R_TILES, R_NT, R_LG, R_N };
static const size_t region_w[R_N] = {
    [R_H] = H, [R_HN] = H, [R_BS] = MAXBS * H, [R_P0] = H, [R_MS] = 2 * H, [R_MSG] = 4 * H, [R_MIX] = 3 * H, [R_G] = 3 * H,
    [R_XN] = 3 * H, [R_Q] = QD, [R_K] = KVD, [R_V] = KVD, [R_GT] = QD, [R_SK] = NH, [R_RS] = NH * DREL, [R_KD] = NKV * 4,
    [R_VD] = NKV * 4, [R_KN] = KVD, [R_VN] = KVD, [R_ATT] = QD, [R_A] = H, [R_R] = H, [R_HM] = H, [R_XM] = H, [R_D] = RANK,
    [R_IDX] = TOPK, [R_W] = TOPK, [R_HACT] = TOPK * FF, [R_YS] = TOPK * H, [R_YSUM] = H,
    [R_PAIRS] = TOPK, [R_TILES] = 4, [R_NT] = 1, [R_LG] = NE};
// the expert tiles of a chunk (ink_moe_plan): at most T * TOPK / 32 + NE, 4 words each
static_assert(PCH * TOPK / 32 + NE <= PCH);
static size_t region_off[R_N + 1]; // floats
__attribute__((constructor)) static void regions(void) {
    size_t o = 0;
    for (size_t i = 0; i < R_N; i++) {
        region_off[i] = o;
        o += ALIGN_UP(region_w[i] * PCH, 64);
    }
    region_off[R_N] = o;
}

static const char *kernel_names[sizeof(Pipes) / sizeof(sx_pipeline *)];
static const Model *prof_model;

struct Model {
    const ModelArch *arch; // first: model_api.c dispatches on it
    const Pack *pack;
    Gpu *g;
    Pipes p;
    ILayer L[NL];
    Linear embed, lm_head;
    sx_arg embed_norm, final_norm, final_q, msg_q; // msg_q: message_input.query [NLOOP][4][H]
    sx_buffer *params, *scratch, *part, *stage, *logits;
    // Prompt chunks of PCH tokens (fast_prefill, the default), else one token at a time, which
    // gives decode's streams bit for bit (verify's exact prefill).
    bool fast_prefill;
    // SAETTA_INK_ROUTES=<golden dir>, read at load (verify, src/inkling.h): the reference's
    // experts of every token at every visit and their margins, so the routers' own choices are
    // checked against them and the engine can be run on them: f32 rounding flips the router at
    // near ties (an 8th / 9th score gap of 3.2e-7 at one token of the 1815-token golden), and one
    // flipped expert moves every later token. Positions past the reference route on their own.
    sx_buffer *routes;  // [NVIS][routes_rows][TOPK] uint32
    sx_buffer *margins; // [NVIS][routes_rows] f32
    sx_buffer *rstats;  // InkRouteStats
    size_t routes_rows;
    bool replay;        // INK_ROUTES_REPLAY
    bool route_ref;     // this chunk lies within the reference
};

// The per-conversation state lives in the base Session's buffers, so conversations swap as the
// other models' do (session_swap_conv):
//   kv   the full visits' keys and values [n_full_vis][2][cap][KVD] f32;
//   ixk  the sliding visits' rings [n_swa_vis][2][WIN][KVD] f32 (row: position % WIN);
//   bk   the conv states per visit: k [3][KVD], v [3][KVD], attention out [3][H], MLP out [3][H].
// The rings and conv states hold only the latest positions, so a conversation can continue only
// from where it stands: its one restore point is its live state (ck[0] or ck[1], whichever has
// s->pos tokens), and a prompt that extends those tokens reuses them; anything else starts over.
typedef Session InkSession;
#define CONV_VIS (6 * KVD + 6 * H) // floats per visit
#define RING_BYTES ((size_t)n_swa_vis * 2 * WIN * KVD * 4)
#define CONV_BYTES ((size_t)NVIS * CONV_VIS * 4)
static size_t kvf_bytes(size_t cap) { return n_full_vis * 2 * cap * KVD * 4; }

static inline sx_arg B(sx_buffer *b) { return (sx_arg){b, 0, NULL, 0}; }
static inline sx_arg AT(sx_buffer *b, size_t off) { return (sx_arg){b, off, NULL, 0}; }
static inline sx_arg OFF(sx_arg a, size_t floats) { return (sx_arg){a.buffer, a.offset + floats * 4, NULL, 0}; }
#define BYTES(v) gpu_bytes_arg(&(v), sizeof(v))
// The command buffer being encoded. SAETTA_PROFILE=1: every dispatch in its own command buffer,
// GPU time summed per kernel (model_prof_report).
typedef struct {
    sx_cmd *cmd;
    Gpu *g;
} Enc;
static struct {
    bool on;
    size_t n;
    const sx_pipeline *p[64];
    const char *name[64];
    double s[64];
    size_t calls[64];
} prof;
static const char *pipe_name(const sx_pipeline *p);
static void enc_dispatch(Enc *c, sx_pipeline *p, const sx_arg *a, size_t na, size_t gx, size_t gy, size_t gz, size_t th) {
    sx_cmd_dispatch3(c->cmd, p, a, na, gx, gy, gz, th);
    if (!prof.on) return;
    double secs = 0;
    char err[256];
    (void)sx_cmd_finish_timed(c->cmd, &secs, err, sizeof err);
    c->cmd = sx_cmd_begin(c->g->dev);
    size_t i = 0;
    while (i < prof.n && prof.p[i] != p) i++;
    if (i == prof.n && i < 64) {
        prof.p[i] = p;
        prof.name[i] = pipe_name(p);
        prof.n++;
    }
    if (i < 64) {
        prof.s[i] += secs;
        prof.calls[i]++;
    }
}
#define DISP(c, p, gx, gy, gz, th, ...)                                     \
    do {                                                                   \
        sx_arg a_[] = {__VA_ARGS__};                                       \
        enc_dispatch((c), (p), a_, COUNT(a_), (gx), (gy), (gz), (th));     \
    } while (0)

static sx_arg reg(const Model *m, int r) { return AT(m->scratch, region_off[r] * 4); }
// row t of region r (width region_w[r])
static sx_arg regt(const Model *m, int r, size_t t) { return OFF(reg(m, r), t * region_w[r]); }

// ---- load -----------------------------------------------------------------------------------

#define PFX "model.language_model."

// Small tensors converted to f32 (or copied as bf16) into Model.params at 16-byte offsets; a first
// pass with base NULL only measures.
typedef struct {
    const Pack *pack;
    sx_buffer *buf;
    uint8_t *base;
    size_t used;
    Err err;
} Prm;

static const TensorInfo *tensor(Prm *P, const char *name, uint64_t numel) {
    const TensorInfo *t = pack_get(P->pack, str(name));
    if (!t || tensor_numel(t) != numel || (t->dtype != DTYPE_BF16 && t->dtype != DTYPE_F32)) {
        if (!P->err) dbg("model: %s missing, or not %llu bf16/f32 values\n", name, (unsigned long long)numel);
        P->err = "MissingTensor";
        return NULL;
    }
    return t;
}
static float tval(const Pack *pack, const TensorInfo *t, size_t i) {
    const uint8_t *d = pack_data(pack, t);
    if (t->dtype == DTYPE_F32) {
        float f;
        memcpy(&f, d + 4 * i, 4);
        return f;
    }
    uint16_t b;
    memcpy(&b, d + 2 * i, 2);
    return ops_bf16_to_f32(b);
}
static size_t prm_take(Prm *P, size_t bytes) {
    size_t off = ALIGN_UP(P->used, 16);
    P->used = off + bytes;
    return off;
}
// f32 copy of a tensor of numel values
static sx_arg prm_f32(Prm *P, const char *name, uint64_t numel) {
    const TensorInfo *t = tensor(P, name, numel);
    size_t off = prm_take(P, numel * 4);
    if (t && P->base)
        for (size_t i = 0; i < numel; i++) ((float *)(P->base + off))[i] = tval(P->pack, t, i);
    return AT(P->buf, off);
}
// the raw bf16 values
static sx_arg prm_bf16(Prm *P, const char *name, uint64_t numel) {
    const TensorInfo *t = tensor(P, name, numel);
    if (t && t->dtype != DTYPE_BF16) P->err = "UnexpectedDType";
    size_t off = prm_take(P, numel * 2);
    if (t && P->base && t->dtype == DTYPE_BF16) memcpy(P->base + off, pack_data(P->pack, t), numel * 2);
    return AT(P->buf, off);
}
// conv taps in lag order [C][4]: a torch conv1d weight (MLX [C][4][1], tap 3 the token itself),
// or the dynamic conv's static weight [4][C] (tap 0 the token itself, M:410-411, 432)
static sx_arg prm_taps(Prm *P, const char *name, size_t C, bool lag_first) {
    const TensorInfo *t = tensor(P, name, C * 4);
    size_t off = prm_take(P, C * 16);
    if (t && P->base) {
        float *o = (float *)(P->base + off);
        for (size_t c = 0; c < C; c++)
            for (size_t l = 0; l < 4; l++) o[c * 4 + l] = lag_first ? tval(P->pack, t, l * C + c) : tval(P->pack, t, c * 4 + (3 - l));
    }
    return AT(P->buf, off);
}
// global_scale / 2688 per expert (nvfp4); a dummy for mxfp8 layers
static sx_arg prm_gfac(Prm *P, const char *name, bool *present) {
    const TensorInfo *t = pack_get(P->pack, str(name));
    *present = t != NULL;
    size_t off = prm_take(P, NE * 4);
    if (t && (t->dtype != DTYPE_F32 || tensor_numel(t) != NE)) P->err = "UnexpectedDType";
    if (t && P->base)
        for (size_t e = 0; e < NE; e++) ((float *)(P->base + off))[e] = tval(P->pack, t, e) / 2688.0f;
    return AT(P->buf, off);
}

static Err lin(const Pack *pack, const char *pfx, const char *name, uint64_t out, uint64_t in, Linear *l) {
    char nm[256];
    snprintf(nm, sizeof nm, "%s%s", pfx, name);
    Err e = pack_linear(pack, str(nm), l);
    if (e) {
        dbg("model: %s: %s\n", nm, e);
        return e;
    }
    if (l->out_features != out || l->in_features != in) {
        dbg("model: %s is [%llu][%llu], expected [%llu][%llu]\n", nm, (unsigned long long)l->out_features,
            (unsigned long long)l->in_features, (unsigned long long)out, (unsigned long long)in);
        return "PackShapeMismatch";
    }
    return OK;
}

static const TensorInfo *expert_t(const Pack *pack, size_t li, const char *which, const char *what, uint64_t d1, uint64_t d2,
                                  Err *err) {
    char nm[160];
    snprintf(nm, sizeof nm, PFX "layers.%zu.mlp.experts.%s.%s", li, which, what);
    const TensorInfo *t = pack_get(pack, str(nm));
    if (!t || t->rank != 3 || t->shape[0] != NE || t->shape[1] != d1 || t->shape[2] * dtype_size(t->dtype) != d2) {
        if (!*err) dbg("model: %s missing or not [%zu][%llu][%llu bytes]\n", nm, (size_t)NE, (unsigned long long)d1, (unsigned long long)d2);
        *err = "PackShapeMismatch";
        return NULL;
    }
    return t;
}

// The small tensors of every layer (both passes of Prm) and the linears and experts.
static Err load_layers(Model *m, Prm *P) {
    const Pack *pack = m->pack;
    char nm[192];
#define NM(...) (snprintf(nm, sizeof nm, __VA_ARGS__), nm)
    for (size_t li = 0; li < NL; li++) {
        ILayer *L = &m->L[li];
        const char *pf = NM(PFX "layers.%zu.", li);
        char p[64];
        snprintf(p, sizeof p, "%s", pf);
        L->full = is_full(li);
        if (L->full) {
            L->n_q = prm_f32(P, NM("%sq_input_layernorm.weight", p), H);
            L->n_k = prm_f32(P, NM("%sk_input_layernorm.weight", p), H);
            L->n_v = prm_f32(P, NM("%sv_input_layernorm.weight", p), H);
            L->kw = prm_taps(P, NM("%sself_attn.k_sconv.static_weight", p), KVD, true);
            L->vw = prm_taps(P, NM("%sself_attn.v_sconv.static_weight", p), KVD, true);
        } else {
            L->n_in = prm_f32(P, NM("%sinput_layernorm.weight", p), H);
            L->kw = prm_taps(P, NM("%sself_attn.k_sconv.conv1d.weight", p), KVD, false);
            L->vw = prm_taps(P, NM("%sself_attn.v_sconv.conv1d.weight", p), KVD, false);
        }
        L->n_post = prm_f32(P, NM("%spost_attention_layernorm.weight", p), H);
        L->aw = prm_taps(P, NM("%spost_attention_convolution.conv1d.weight", p), H, false);
        L->mw = prm_taps(P, NM("%spost_mlp_convolution.conv1d.weight", p), H, false);
        L->qscale = prm_f32(P, NM("%sself_attn.q_l2_clip.scale", p), QD);
        L->rscale = prm_f32(P, NM("%sself_attn.rel_l2_clip.scale", p), NH * DREL);
        L->proj = prm_f32(P, NM("%sself_attn.rel_logits_proj.proj", p), DREL * EXT);
        if (li > 0) L->aq = prm_f32(P, NM("%sattention_input.query", p), (L->full ? 3 : 1) * H);
        L->mq = prm_f32(P, NM("%smlp_input.query", p), H);
        L->rw = prm_f32(P, NM("%smlp.gate.weight", p), (size_t)NE * H);
        L->rb = prm_f32(P, NM("%smlp.gate.e_score_correction_bias", p), NE);
        bool nv_u, nv_d;
        L->gfu = prm_gfac(P, NM("%smlp.experts.gate_up_proj.global_scale", p), &nv_u);
        L->gfd = prm_gfac(P, NM("%smlp.experts.down_proj.global_scale", p), &nv_d);
        if (nv_u != nv_d) P->err = "PackShapeMismatch";
        L->mx = !nv_u;
        if (li >= LS && li < LE) {
            static const char *const gn[4] = {"q", "k", "v", "mlp"};
            for (size_t k = 0; k < 4; k++) {
                L->md[k] = prm_bf16(P, NM("%s%s_message_gate.down_proj.weight", p, gn[k]), (size_t)RANK * 2 * H);
                L->mf[k] = prm_f32(P, NM("%s%s_message_gate.forget_gate_proj.weight", p, gn[k]), (size_t)H * RANK);
                L->mg[k] = prm_f32(P, NM("%s%s_message_gate.message_gate_proj.weight", p, gn[k]), (size_t)H * RANK);
            }
        }
        if (P->base) continue; // linears and experts once, on the measuring pass
        TRY(lin(pack, p, "self_attn.q_proj", QD, H, &L->q));
        TRY(lin(pack, p, "self_attn.k_proj", KVD, H, &L->k));
        TRY(lin(pack, p, "self_attn.v_proj", KVD, H, &L->v));
        TRY(lin(pack, p, "self_attn.o_proj", H, QD, &L->o));
        TRY(lin(pack, p, "self_attn.gate_proj", QD, H, &L->g));
        TRY(lin(pack, p, "self_attn.sink_proj", NH, H, &L->sink));
        TRY(lin(pack, p, "self_attn.r_proj", NH * DREL, H, &L->r));
        if (L->full) {
            TRY(lin(pack, p, "self_attn.k_sconv.weight_proj", NKV * 4, H, &L->kd));
            TRY(lin(pack, p, "self_attn.v_sconv.weight_proj", NKV * 4, H, &L->vd));
        }
        Err e = OK;
        const TensorInfo *gw = expert_t(pack, li, "gate_up_proj", "weight", 2 * FF, L->mx ? H : H / 2, &e);
        const TensorInfo *gs = expert_t(pack, li, "gate_up_proj", "scales", 2 * FF, L->mx ? H / 32 : H / 16, &e);
        const TensorInfo *dw = expert_t(pack, li, "down_proj", "weight", H, L->mx ? FF : FF / 2, &e);
        const TensorInfo *ds = expert_t(pack, li, "down_proj", "scales", H, L->mx ? FF / 32 : FF / 16, &e);
        if (e) return e;
        L->guw = *gw, L->gus = *gs, L->dnw = *dw, L->dns = *ds;
    }
#undef NM
    return OK;
}

static const char ink_src[] = {
#embed "kernels/inkling_ops.metal"
    , '\n',
#embed "kernels/inkling_attn.metal"
    , '\n',
#embed "kernels/inkling_moe.metal"
    , 0};

Err model_load(Arena *arena, const Pack *pack, Gpu *g, Model **out) {
    Model *m = ARENA_NEW(arena, Model);
    *m = (Model){0};
    m->arch = &inkling_arch;
    m->pack = pack;
    m->g = g;
    m->fast_prefill = true;
    *out = m;
    Pipes *pp = &m->p;
    static const struct {
        size_t off;
        const char *name;
    } kn[] = {
#define K(f, n) {offsetof(Pipes, f), n}
        K(mix, "ink_mix"), K(norm, "ink_rmsnorm"), K(msg_down, "ink_msg_down"), K(msg_out, "ink_msg_out"), K(qr, "ink_qr_prep"),
        K(kv_conv, "ink_kv_conv"), K(conv_res, "ink_conv_res"), K(conv_state, "ink_conv_state"), K(router, "ink_router"), K(router_lg, "ink_router_logits"),
        K(moe_sum, "ink_moe_sum"), K(softcap, "ink_softcap"), K(copy, "ink_copy"), K(kv_store, "ink_kv_store"),
        K(attn_part, "ink_attn_part"), K(attn_merge, "ink_attn_merge"), K(gu_nv, "ink_moe_gu_nv"), K(dn_nv, "ink_moe_dn_nv"),
        K(gu_mx, "ink_moe_gu_mx"), K(dn_mx, "ink_moe_dn_mx"), K(plan, "ink_moe_plan"),
        K(gu_nv_g, "ink_moe_gu_nv_g"), K(gu_mx_g, "ink_moe_gu_mx_g"), K(dn_nv_g, "ink_moe_dn_nv_g"), K(dn_mx_g, "ink_moe_dn_mx_g"),
#undef K
    };
    for (size_t i = 0; i < COUNT(kn); i++) {
        TRY(gpu_pipeline(g, ink_src, kn[i].name, (sx_pipeline **)((char *)pp + kn[i].off)));
        kernel_names[kn[i].off / sizeof(sx_pipeline *)] = kn[i].name;
    }
    kernel_names[offsetof(Pipes, mv) / sizeof(sx_pipeline *)] = "qmatvec_gen";
    kernel_names[offsetof(Pipes, mm) / sizeof(sx_pipeline *)] = "qmm_gen";
    prof_model = m;
    TRY(gpu_pipeline(g, qgen_src, "qmatvec_gen", &pp->mv));
    TRY(gpu_pipeline(g, qgen_src, "qmm_gen", &pp->mm));

    TRY(lin(pack, PFX, "embed_tokens", CONFIG_VOCAB_SIZE, H, &m->embed));
    TRY(lin(pack, "", "lm_head", CONFIG_VOCAB_SIZE, H, &m->lm_head));
    // two passes over the small tensors: measure, then fill
    Prm P = {.pack = pack};
    for (int pass = 0; pass < 2; pass++) {
        P.used = 0;
        m->embed_norm = prm_f32(&P, PFX "embed_norm.weight", H);
        m->final_norm = prm_f32(&P, PFX "norm.weight", H);
        m->final_q = prm_f32(&P, PFX "final_input.query", H);
        m->msg_q = prm_f32(&P, PFX "message_input.query", (size_t)NLOOP * 4 * H);
        TRY(load_layers(m, &P));
        if (P.err) return P.err;
        if (pass == 0) {
            TRY(gpu_new_buffer(g, ALIGN_UP(P.used, 16), &m->params));
            P.buf = m->params;
            P.base = sx_buffer_contents(m->params);
        }
    }
    TRY(gpu_new_buffer(g, region_off[R_N] * 4, &m->scratch));
    TRY(gpu_new_buffer(g, (size_t)MAX(PCH * NSPLIT_PRE, NSPLIT_DEC) * NH * (HD + 2) * 4, &m->part));
    TRY(gpu_new_buffer(g, (size_t)PCH * H * 4, &m->stage));
    TRY(gpu_new_buffer(g, (size_t)KMAX * VOCAB * 4, &m->logits));
    // SAETTA_INK_ROUTES=<golden dir> (verify), read once
    const char *rd = getenv("SAETTA_INK_ROUTES");
    if (rd && rd[0]) {
        char *rb, *mb;
        size_t rl, ml;
        TRY(read_file(path_join(arena, rd, "routes.u8"), (size_t)1 << 30, &rb, &rl));
        Err e = read_file(path_join(arena, rd, "margins.f32"), (size_t)1 << 30, &mb, &ml);
        if (e) {
            free(rb);
            return e;
        }
        const size_t n = rl / ((size_t)NVIS * TOPK);
        e = n > 0 && n * NVIS * TOPK == rl && ml == (size_t)NVIS * n * 4 ? OK : "BadRoutesFile";
        if (!e) e = gpu_new_buffer(g, rl * 4, &m->routes);
        if (!e) e = gpu_new_buffer(g, ml, &m->margins);
        if (!e) e = gpu_new_buffer(g, 16, &m->rstats);
        if (!e) {
            uint32_t *r = sx_buffer_contents(m->routes);
            for (size_t i = 0; i < rl; i++) r[i] = (uint8_t)rb[i];
            memcpy(sx_buffer_contents(m->margins), mb, ml);
            m->routes_rows = n;
            (void)inkling_route_stats(m);
        }
        free(rb);
        free(mb);
        TRY(e);
    }
    dbg("model: Inkling 2 Mini, %zu visits (%zu full, %zu sliding), params %.1f MB\n", (size_t)NVIS, n_full_vis, n_swa_vis,
        (double)P.used / 1e6);
    return OK;
}

size_t model_check_pack(Arena *a, const Pack *pack, FILE *out) {
    (void)a;
    Model m = {.pack = pack};
    Prm P = {.pack = pack};
    Err e = load_layers(&m, &P);
    if (!e) e = P.err;
    Linear l;
    if (!e) e = lin(pack, PFX, "embed_tokens", CONFIG_VOCAB_SIZE, H, &l);
    if (!e) e = lin(pack, "", "lm_head", CONFIG_VOCAB_SIZE, H, &l);
    if (e) {
        fprintf(out, "  Inkling 2 Mini: %s\n", e);
        return 1;
    }
    return 0;
}

// ---- encoder --------------------------------------------------------------------------------

typedef struct { uint32_t n, nq, h, sstride, pstride, ostride; float eps; } MixP;
typedef struct { uint32_t n, xs, os, hasw; float eps; } NormP;
typedef struct { uint32_t h, r, xs, ms; float eps; } MsgP;
typedef struct { uint32_t NH_, HD_, DREL_, pos0, logscale; float thr_q, thr_r, alpha, nfloor; } QrP;
typedef struct { uint32_t C, HD_, dyn, mode, update; float den, thr, eps; } KvP;
typedef struct { uint32_t C, boundary, update; float den; } CrP;
typedef struct { uint32_t C, T; } CsP;
typedef struct { uint32_t e, h, k; float scale; uint32_t forced, check, pos0; float thr; } RtP;
typedef struct { uint32_t c, k; } SumP;
typedef struct { uint32_t n; float cap; } CapP;
typedef struct { uint32_t C, pos0, ring, T; } StP;
typedef struct { uint32_t NH_, NKV_, HD_, DREL_, EXT_, pos0, win, ring, nsplit, span; float scale; } AttP;
typedef struct { uint32_t h, i, k, sh_w, sh_s; uint64_t wsE, ssE; float gcap, ucap; } MoeP;
typedef struct { uint32_t e, k, t; } PlanP;
static_assert(sizeof(MoeP) == 48);

// y = W x over T rows: the run-time affine kernels (decode: one simdgroup per row; chunks: f32
// tiles). out rows of W (lm_head: the unpadded vocabulary).
static void enc_lin(Model *m, Enc *c, const Linear *l, sx_arg x, sx_arg y, uint32_t T, uint32_t out) {
    uint32_t sh[3], q[2] = {l->bits, l->group_size}, in = (uint32_t)l->in_features;
    sx_arg w = gpu_tensor_arg(m->g, &l->weight, &sh[0]), s = gpu_tensor_arg(m->g, &l->scales, &sh[1]),
           b = gpu_tensor_arg(m->g, &l->biases, &sh[2]);
    if (T == 1) {
        DISP(c, m->p.mv, out, 1, 1, 32, w, s, b, x, y, BYTES(in), BYTES(out), gpu_bytes_arg(sh, 12), BYTES(q));
    } else {
        uint32_t dims[3] = {in, out, T};
        DISP(c, m->p.mm, (out + 31) / 32, (T + 31) / 32, 1, 128, w, s, b, x, y, gpu_bytes_arg(dims, 12), gpu_bytes_arg(sh, 12),
             BYTES(q));
    }
}

static void enc_norm(Model *m, Enc *c, sx_arg x, uint32_t xs, sx_arg w, bool hasw, sx_arg out, uint32_t T) {
    NormP p = {H, xs, H, hasw, EPS};
    DISP(c, m->p.norm, T, 1, 1, 256, x, w, out, BYTES(p));
}

// out [T][nq][H] (rows ostride apart) = mix of block states 0..n-1 and partial with the queries.
static void enc_mix(Model *m, Enc *c, uint32_t n, sx_arg partial, sx_arg query, uint32_t nq, sx_arg outp, uint32_t ostride,
                    uint32_t T, sx_arg states, uint32_t sstride) {
    MixP p = {n, nq, H, sstride, H, ostride, EPS};
    DISP(c, m->p.mix, T, 1, 1, 1024, states, partial, query, outp, BYTES(p));
}

// message gate k of layer L: out [T][H] = gate(x [rows xs apart], msg [rows ms apart])
static void enc_gate(Model *m, Enc *c, const ILayer *L, int k, sx_arg x, uint32_t xs, sx_arg msg, uint32_t ms, sx_arg out,
                     uint32_t T) {
    MsgP p = {H, RANK, xs, ms, EPS};
    DISP(c, m->p.msg_down, RANK / 8, T, 1, 256, x, msg, L->md[k], reg(m, R_D), BYTES(p));
    DISP(c, m->p.msg_out, H / 256, T, 1, 256, x, msg, reg(m, R_D), L->mf[k], L->mg[k], out, BYTES(p));
}

static void enc_copy(Model *m, Enc *c, sx_arg src, sx_arg dst, uint32_t n) {
    DISP(c, m->p.copy, (n + 255) / 256, 1, 1, 256, src, dst, BYTES(n));
}

static void enc_conv_state(Model *m, Enc *c, sx_arg raw, sx_arg state, uint32_t C, uint32_t T) {
    CsP p = {C, T};
    DISP(c, m->p.conv_state, (C + 255) / 256, 1, 1, 256, raw, state, BYTES(p));
}

// One visit v of T tokens at positions pos0.. with n block states.
static void enc_visit(Model *m, InkSession *d, Enc *c, size_t v, uint32_t T, uint32_t pos0, uint32_t n) {
    const size_t li = vis_layer[v];
    const ILayer *L = &m->L[li];
    const bool msg = vis_pass[v] > 0, boundary = li % NBLK == 0, full = L->full;
    const uint32_t PH = PCH * H; // stride of the block-state planes
    const sx_arg h = reg(m, R_H), bs = reg(m, R_BS), conv = AT(d->bk, v * CONV_VIS * 4);
    // the messages of this loop layer from the previous pass's states (M:1407, 145-173)
    if (msg) {
        sx_arg q = OFF(m->msg_q, (li - LS) * 4 * H);
        enc_mix(m, c, 1, OFF(reg(m, R_MS), PH), q, 4, reg(m, R_MSG), 4 * H, T, reg(m, R_MS), PH);
    }
    // attention inputs (M:886-898, 957-967, 1032-1044, 1111-1119)
    sx_arg xq, xk, xv;
    uint32_t xs = 3 * H;
    if (li == 0) {
        xq = xk = xv = h;
        xs = H;
    } else {
        enc_mix(m, c, n, h, L->aq, full ? 3 : 1, reg(m, R_MIX), 3 * H, T, bs, PH);
        xq = reg(m, R_MIX);
        xk = full ? OFF(xq, H) : xq;
        xv = full ? OFF(xq, 2 * H) : xq;
    }
    if (msg) {
        const sx_arg G = reg(m, R_G);
        enc_gate(m, c, L, 0, xq, xs, reg(m, R_MSG), 4 * H, G, T);
        enc_gate(m, c, L, 1, xk, xs, OFF(reg(m, R_MSG), H), 4 * H, OFF(G, PH), T);
        enc_gate(m, c, L, 2, xv, xs, OFF(reg(m, R_MSG), 2 * H), 4 * H, OFF(G, 2 * PH), T);
        xq = G, xk = OFF(G, PH), xv = OFF(G, 2 * PH);
        xs = H;
    }
    const sx_arg XN = reg(m, R_XN);
    sx_arg nq = XN, nk = OFF(XN, PH), nv = OFF(XN, 2 * PH);
    if (full) {
        enc_norm(m, c, xq, xs, L->n_q, true, nq, T);
        enc_norm(m, c, xk, xs, L->n_k, true, nk, T);
        enc_norm(m, c, xv, xs, L->n_v, true, nv, T);
    } else if (msg) {
        enc_norm(m, c, xq, xs, L->n_in, true, nq, T);
        enc_norm(m, c, xk, xs, L->n_in, true, nk, T);
        enc_norm(m, c, xv, xs, L->n_in, true, nv, T);
    } else {
        enc_norm(m, c, xq, xs, L->n_in, true, nq, T);
        nk = nv = nq;
    }
    // projections
    enc_lin(m, c, &L->q, nq, reg(m, R_Q), T, QD);
    enc_lin(m, c, &L->g, nq, reg(m, R_GT), T, QD);
    enc_lin(m, c, &L->sink, nq, reg(m, R_SK), T, NH);
    enc_lin(m, c, &L->r, nq, reg(m, R_RS), T, NH * DREL);
    enc_lin(m, c, &L->k, nk, reg(m, R_K), T, KVD);
    enc_lin(m, c, &L->v, nv, reg(m, R_V), T, KVD);
    if (full) {
        enc_lin(m, c, &L->kd, nk, reg(m, R_KD), T, NKV * 4);
        enc_lin(m, c, &L->vd, nv, reg(m, R_VD), T, NKV * 4);
    }
    {
        QrP p = {NH, HD, DREL, pos0, full, CONFIG_QK_L2_CLIP_THRESHOLD, CONFIG_REL_L2_CLIP_THRESHOLD, CONFIG_LOG_SCALING_ALPHA,
                 (float)CONFIG_LOG_SCALING_N_FLOOR};
        DISP(c, m->p.qr, NH, T, 1, HD, reg(m, R_Q), reg(m, R_RS), L->qscale, L->rscale, BYTES(p));
    }
    // keys and values: conv, clip / norm (M:478-528)
    {
        const float den = full ? sqrtf(9.0f) : sqrtf(5.0f);
        KvP pk = {KVD, HD, full, 0, T == 1, den, CONFIG_QK_L2_CLIP_THRESHOLD, EPS};
        KvP pv = {KVD, HD, full, 1, T == 1, den, CONFIG_QK_L2_CLIP_THRESHOLD, EPS};
        const sx_arg ks = conv, vs = OFF(conv, 3 * KVD);
        DISP(c, m->p.kv_conv, NKV, T, 1, HD, reg(m, R_K), ks, L->kw, reg(m, R_KD), reg(m, R_KN), BYTES(pk));
        DISP(c, m->p.kv_conv, NKV, T, 1, HD, reg(m, R_V), vs, L->vw, reg(m, R_VD), reg(m, R_VN), BYTES(pv));
        if (T > 1) {
            enc_conv_state(m, c, reg(m, R_K), ks, KVD, T);
            enc_conv_state(m, c, reg(m, R_V), vs, KVD, T);
        }
    }
    // attention (M:111-142, 611-636)
    {
        sx_arg kc, vc;
        uint32_t ring;
        if (full) {
            const size_t plane = (size_t)d->cap * KVD * 4;
            kc = AT(d->kv, vis_kv[v] * 2 * plane);
            vc = AT(d->kv, (vis_kv[v] * 2 + 1) * plane);
            ring = 0;
        } else {
            const size_t plane = (size_t)WIN * KVD * 4;
            kc = AT(d->ixk, vis_kv[v] * 2 * plane);
            vc = AT(d->ixk, (vis_kv[v] * 2 + 1) * plane);
            ring = WIN;
        }
        // keys of the last query: its whole range; the splits cover the longest range of the chunk
        const uint32_t last = pos0 + T - 1, range = full ? last + 1 : MIN(last + 1, (uint32_t)WIN);
        uint32_t nsplit = T == 1 ? MIN((uint32_t)NSPLIT_DEC, MAX(1u, (range + 127) / 128)) : MIN((uint32_t)NSPLIT_PRE, MAX(1u, (range + 255) / 256));
        const uint32_t span = (range + nsplit - 1) / nsplit;
        AttP p = {NH, NKV, HD, DREL, EXT, pos0, full ? 0 : WIN, ring, nsplit, span, 1.0f / sqrtf((float)HD)};
        DISP(c, m->p.attn_part, NH, nsplit, T, 128, reg(m, R_Q), reg(m, R_RS), L->proj, kc, vc, reg(m, R_KN), reg(m, R_VN), B(m->part),
             BYTES(p));
        DISP(c, m->p.attn_merge, NH, T, 1, HD, B(m->part), reg(m, R_SK), reg(m, R_GT), reg(m, R_ATT), BYTES(p));
        StP sp = {KVD, pos0, ring, T};
        DISP(c, m->p.kv_store, KVD / 256, T, 1, 256, reg(m, R_KN), reg(m, R_VN), kc, vc, BYTES(sp));
    }
    enc_lin(m, c, &L->o, reg(m, R_ATT), reg(m, R_A), T, H);
    // post-attention conv and the residual (M:899-905)
    {
        CrP p = {H, boundary, T == 1, sqrtf(5.0f)};
        const sx_arg as = OFF(conv, 6 * KVD);
        DISP(c, m->p.conv_res, H / 256, T, 1, 256, reg(m, R_A), as, L->aw, h, reg(m, R_R), OFF(bs, n * PH), BYTES(p));
        if (T > 1) enc_conv_state(m, c, reg(m, R_A), as, H, T);
    }
    const uint32_t n2 = n + boundary;
    // MLP input (M:907-908, 990-993)
    enc_mix(m, c, n2, reg(m, R_R), L->mq, 1, reg(m, R_HM), H, T, bs, PH);
    sx_arg hm = reg(m, R_HM);
    if (msg) {
        enc_gate(m, c, L, 3, hm, H, OFF(reg(m, R_MSG), 3 * H), 4 * H, reg(m, R_G), T);
        hm = reg(m, R_G);
    }
    enc_norm(m, c, hm, H, L->n_post, true, reg(m, R_XM), T);
    // router and experts (M:814-825, 1152-1172)
    {
        const bool ref = m->route_ref;
        const sx_arg rt = ref ? AT(m->routes, ((size_t)v * m->routes_rows + pos0) * TOPK * 4) : reg(m, R_IDX);
        const sx_arg mg = ref ? AT(m->margins, ((size_t)v * m->routes_rows + pos0) * 4) : reg(m, R_W);
        const sx_arg st = ref ? B(m->rstats) : reg(m, R_NT);
        RtP rp = {NE, H, TOPK, CONFIG_ROUTE_SCALE, ref && m->replay, ref, pos0, TOL_INK_ROUTE_MARGIN};
        uint32_t lgp[2] = {NE, H};
        DISP(c, m->p.router_lg, NE / 8, T, 1, 256, reg(m, R_XM), L->rw, reg(m, R_LG), gpu_bytes_arg(lgp, 8));
        DISP(c, m->p.router, T, 1, 1, NE, reg(m, R_LG), L->rb, reg(m, R_IDX), reg(m, R_W), BYTES(rp), rt, mg, st);
        uint32_t shw, shs;
        sx_arg gw = gpu_tensor_arg(m->g, &L->guw, &shw), gs = gpu_tensor_arg(m->g, &L->gus, &shs);
        MoeP mp = {H, FF, TOPK, shw, shs, L->guw.nbytes / NE, L->gus.nbytes / NE, CONFIG_MLP_GATE_SOFTCAP, CONFIG_MLP_UP_SOFTCAP};
        sx_arg dw = gpu_tensor_arg(m->g, &L->dnw, &shw), ds = gpu_tensor_arg(m->g, &L->dns, &shs);
        MoeP dp = {H, FF, TOPK, shw, shs, L->dnw.nbytes / NE, L->dns.nbytes / NE, CONFIG_MLP_GATE_SOFTCAP, CONFIG_MLP_UP_SOFTCAP};
        if (T == 1) {
            // one token: each expert's rows read once, f32 dot products per row
            DISP(c, L->mx ? m->p.gu_mx : m->p.gu_nv, FF / 8, TOPK, T, 256, gw, gs, L->gfu, reg(m, R_XM), reg(m, R_IDX),
                 reg(m, R_HACT), BYTES(mp));
            DISP(c, L->mx ? m->p.dn_mx : m->p.dn_nv, H / 32, TOPK, T, 256, dw, ds, L->gfd, reg(m, R_HACT), reg(m, R_IDX),
                 reg(m, R_W), reg(m, R_YS), BYTES(dp));
        } else {
            // a chunk: the (token, slot) pairs grouped by expert, tiles of 32 pairs, so each expert's
            // weights are read once per chunk
            PlanP pp = {NE, TOPK, T};
            DISP(c, m->p.plan, 1, 1, 1, 128, reg(m, R_IDX), reg(m, R_PAIRS), reg(m, R_TILES), reg(m, R_NT), BYTES(pp));
            const uint32_t maxt = T * TOPK / 32 + NE;
            DISP(c, L->mx ? m->p.gu_mx_g : m->p.gu_nv_g, 2 * FF / 64, maxt, 1, 128, gw, gs, L->gfu, reg(m, R_XM), reg(m, R_PAIRS),
                 reg(m, R_TILES), reg(m, R_NT), reg(m, R_HACT), BYTES(mp));
            DISP(c, L->mx ? m->p.dn_mx_g : m->p.dn_nv_g, H / 64, maxt, 1, 128, dw, ds, L->gfd, reg(m, R_HACT), reg(m, R_PAIRS),
                 reg(m, R_TILES), reg(m, R_NT), reg(m, R_W), reg(m, R_YS), BYTES(dp));
        }
        SumP sp = {H, TOPK};
        DISP(c, m->p.moe_sum, H / 256, T, 1, 256, reg(m, R_YS), reg(m, R_YSUM), BYTES(sp));
    }
    // post-MLP conv and the layer output (M:909-910)
    {
        CrP p = {H, 0, T == 1, sqrtf(5.0f)};
        const sx_arg ms = OFF(conv, 6 * KVD + 3 * H);
        DISP(c, m->p.conv_res, H / 256, T, 1, 256, reg(m, R_YSUM), ms, L->mw, reg(m, R_R), reg(m, R_HN), reg(m, R_HN), BYTES(p));
        if (T > 1) enc_conv_state(m, c, reg(m, R_YSUM), ms, H, T);
    }
}

// The whole forward pass of T tokens (embeddings in Model.stage) at s->pos: hidden states after
// each visit to trace rows (v * tstride + trow) when trace is set; logits of the last row (all
// rows when all_logits) into Model.logits.
static void enc_forward(Model *m, InkSession *d, Enc *c, uint32_t T, sx_buffer *trace, size_t trow, size_t tstride,
                        bool all_logits) {
    const uint32_t pos0 = (uint32_t)d->pos, PH = PCH * H;
    const sx_arg h = reg(m, R_H), bs = reg(m, R_BS);
    enc_norm(m, c, B(m->stage), H, m->embed_norm, true, h, T); // M:1240-1246
    uint32_t n = 0, npre = 0;
    for (size_t v = 0; v < NVIS; v++) {
        const size_t li = vis_layer[v];
        if (v == LS) { // the pre-loop states (M:1378-1379)
            enc_copy(m, c, h, reg(m, R_P0), T * H);
            npre = n;
        }
        if (vis_pass[v] > 0 && li == LS) { // a later pass restarts from them (M:1398-1402)
            enc_copy(m, c, OFF(bs, (npre + 1) * PH), reg(m, R_MS), T * H);
            enc_copy(m, c, h, OFF(reg(m, R_MS), PH), T * H);
            enc_copy(m, c, reg(m, R_P0), h, T * H);
            n = npre;
        }
        enc_visit(m, d, c, v, T, pos0, n);
        n += li % NBLK == 0;
        if (trace) {
            uint32_t nn = T * H;
            DISP(c, m->p.copy, (nn + 255) / 256, 1, 1, 256, reg(m, R_HN), AT(trace, (v * tstride + trow) * H * 4), BYTES(nn));
        }
        enc_copy(m, c, reg(m, R_HN), h, T * H);
    }
    // final mix, norm, lm_head, softcap (M:1442-1443, 1533-1535)
    enc_mix(m, c, n, h, m->final_q, 1, reg(m, R_HM), H, T, bs, PH);
    enc_norm(m, c, reg(m, R_HM), H, m->final_norm, true, reg(m, R_XM), T);
    const uint32_t nl = all_logits ? T : 1;
    const sx_arg xl = all_logits ? reg(m, R_XM) : regt(m, R_XM, T - 1);
    enc_lin(m, c, &m->lm_head, xl, B(m->logits), nl, VOCAB);
    CapP cp = {nl * VOCAB, CONFIG_FINAL_LOGIT_SOFTCAPPING};
    DISP(c, m->p.softcap, (cp.n + 255) / 256, 1, 1, 256, B(m->logits), BYTES(cp));
}

// Embedding rows of tokens (affine 4-bit, dequantized on the host as the kernels do) into stage.
static Err stage_tokens(Model *m, const uint32_t *toks, size_t T) {
    const Linear *l = &m->embed;
    const size_t in = H, groups = in / l->group_size;
    const uint8_t *w = pack_data(m->pack, &l->weight), *s = pack_data(m->pack, &l->scales), *b = pack_data(m->pack, &l->biases);
    float *out = gpu_floats(m->stage);
    for (size_t t = 0; t < T; t++) {
        const size_t r = toks[t];
        if (r >= CONFIG_VOCAB_SIZE) return "BadToken";
        const uint8_t *row = w + r * (in / 2);
        for (size_t i = 0; i < in; i++) {
            const uint32_t q = (row[i / 2] >> (4 * (i & 1))) & 15;
            const size_t g = r * groups + i / l->group_size;
            const float sc = ops_bf16_to_f32((uint16_t)(s[g * 2] | (s[g * 2 + 1] << 8)));
            const float bi = ops_bf16_to_f32((uint16_t)(b[g * 2] | (b[g * 2 + 1] << 8)));
            out[t * H + i] = (float)q * sc + bi;
        }
    }
    return OK;
}

// One chunk of T <= PCH tokens, one command buffer.
static Err run_chunk(Model *m, Session *s, const uint32_t *toks, uint32_t T, sx_buffer *trace, size_t trow, size_t tstride,
                     bool all_logits) {
    InkSession *d = s;
    if (s->pos + T > s->cap) return "ContextFull";
    // the reference's experts where it has them (verify)
    m->route_ref = m->routes && s->pos + T <= m->routes_rows;
    TRY(stage_tokens(m, toks, T));
    Enc enc = {sx_cmd_begin(m->g->dev), m->g}, *c = &enc;
    if (!c->cmd) return "GpuFailed";
    enc_forward(m, d, c, T, trace, trow, tstride, all_logits);
    char err[512];
    if (sx_cmd_finish(c->cmd, err, sizeof err) != 0) {
        dbg("command buffer: %s\n", err);
        return "GpuFailed";
    }
    s->pos += T;
    // a restore point is usable only while it is the live state
    for (int k = 0; k < 2; k++)
        if (s->ck[k].tokens.len != s->pos) s->ck[k].tokens.len = 0;
    return OK;
}

Err model_step(Model *m, Session *s, uint32_t token, float **logits) {
    TRY(run_chunk(m, s, &token, 1, s->trace ? s->trace_buf : NULL, 0, 1, false));
    if (s->trace) memcpy(s->trace, gpu_floats(s->trace_buf), s->trace_len * sizeof(float));
    *logits = gpu_floats(m->logits);
    return OK;
}

// Prompt chunks of PCH tokens (SAETTA_INK_PCH: smaller chunks, a diagnostic); trace
// [visits][n][H]. The hook's chunk_done runs after each chunk.
static Err prefill(Model *m, Session *s, const uint32_t *tokens, size_t n, sx_buffer *trace, const PrefillHook *hook) {
    size_t ch = m->fast_prefill ? PCH : 1;
    const char *e = getenv("SAETTA_INK_PCH");
    if (e && atoi(e) > 0) ch = MIN((size_t)atoi(e), (size_t)PCH);
    for (size_t t0 = 0; t0 < n; t0 += ch) {
        const uint32_t T = (uint32_t)MIN(ch, n - t0);
        TRY(run_chunk(m, s, tokens + t0, T, trace, t0, n, false));
        if (hook && hook->chunk_done && ch > 1) TRY(hook->chunk_done(hook->ctx, s->pos));
    }
    return OK;
}

Err model_prefill(Model *m, Session *s, const uint32_t *tokens, size_t n, sx_buffer *trace, float **logits) {
    TRY(prefill(m, s, tokens, n, trace, NULL));
    *logits = gpu_floats(m->logits);
    return OK;
}

Err model_prefill_with(Model *m, Session *s, const uint32_t *tokens, size_t n, sx_buffer *trace, const PrefillHook *hook,
                       float **logits) {
    TRY(prefill(m, s, tokens, n, trace, hook));
    // token by token (not fast_prefill, no SAETTA_INK_PCH): one call at the end
    const char *e = getenv("SAETTA_INK_PCH");
    if (hook && hook->chunk_done && !m->fast_prefill && !(e && atoi(e) > 1)) TRY(hook->chunk_done(hook->ctx, s->pos));
    *logits = gpu_floats(m->logits);
    return OK;
}

Err model_verify_chunk(Model *m, Session *s, const uint32_t *toks, size_t n, float **logits) {
    if (n == 0 || n > KMAX) return "BadChunk";
    TRY(run_chunk(m, s, toks, (uint32_t)n, NULL, 0, 1, true));
    *logits = gpu_floats(m->logits);
    return OK;
}

// ---- sessions -----------------------------------------------------------------------------

// KV in f32 whatever kv_half asks (no f16 attention kernels yet); session_kv_elem says so, and
// the server's memory fit counts it.
Err session_create_cap(Gpu *g, bool kv_half, size_t cap, size_t max_cap, Session **out) {
    (void)kv_half;
    cap = ALIGN_UP(MIN(MAX(cap, (size_t)4), max_cap), 4);
    Session *s = xcalloc(1, sizeof *s);
    s->arch = &inkling_arch;
    s->cap = cap;
    s->max_cap = max_cap;
    s->kv_half = false;
    s->roll = SESSION_NO_ROLL;
    TRY(gpu_new_buffer(g, kvf_bytes(cap), &s->kv));
    TRY(gpu_new_buffer(g, RING_BYTES, &s->ixk));
    TRY(gpu_new_buffer(g, CONV_BYTES, &s->bk));
    TRY(gpu_new_buffer(g, (size_t)NVIS * H * 4, &s->trace_buf));
    *out = s;
    return OK;
}

Err session_create_kv(Gpu *g, bool kv_half, Session **out) {
    size_t cap;
    TRY(session_ctx_parse(getenv("SAETTA_CTX"), CONFIG_MODEL_MAX_LENGTH, &cap));
    return session_create_cap(g, kv_half, cap, cap, out);
}
Err session_create(Gpu *g, Session **out) { return session_create_kv(g, false, out); }

void session_reset(Session *s) {
    s->pos = 0;
    s->roll = SESSION_NO_ROLL;
    for (int k = 0; k < 2; k++) s->ck[k].tokens.len = 0;
    memset(sx_buffer_contents(s->bk), 0, sx_buffer_length(s->bk));
}

size_t session_kv_elem(const Session *s) { (void)s; return 4; }
// The bytes that grow with the capacity (the full visits' KV); the rings and conv states are
// fixed (session_conv_bytes).
size_t session_kv_bytes(const Session *s, size_t cap) {
    (void)s;
    return kvf_bytes(cap);
}
size_t session_conv_bytes(const Session *s, size_t cap) { return session_kv_bytes(s, cap) + RING_BYTES + CONV_BYTES; }

// The full visits' rows below s->pos copied into new buffers of capacity cap.
static Err regrow(Gpu *g, sx_buffer **kv, size_t old_cap, size_t cap, size_t rows) {
    sx_buffer *nb;
    TRY(gpu_new_buffer(g, kvf_bytes(cap), &nb));
    const size_t row = KVD * 4;
    const uint8_t *src = sx_buffer_contents(*kv);
    uint8_t *dst = sx_buffer_contents(nb);
    for (size_t plane = 0; plane < n_full_vis * 2; plane++) memcpy(dst + plane * cap * row, src + plane * old_cap * row, rows * row);
    sx_buffer_release(*kv);
    *kv = nb;
    return OK;
}

Err session_grow(Session *s, Gpu *g, size_t cap) {
    if (!(cap >= s->cap && cap <= s->max_cap && cap % 4 == 0)) die("session_grow: bad capacity %zu\n", cap);
    if (cap == s->cap) return OK;
    TRY(regrow(g, &s->kv, s->cap, cap, s->pos));
    s->cap = cap;
    return OK;
}

// Restore points: only the live state (the tokens consumed so far) is kept, in slot `slot`.
Err session_save_checkpoint(Session *s, const uint32_t *toks, size_t n) { return session_save_checkpoint_at(s, 0, toks, n); }
Err session_save_checkpoint_at(Session *s, size_t slot, const uint32_t *toks, size_t n) {
    if (n != s->pos) die("session_save_checkpoint_at: %zu tokens at position %zu\n", n, s->pos);
    VecU32 *v = &s->ck[slot].tokens;
    v->len = 0;
    vec_reserve(v, n);
    if (n) memcpy(v->ptr, toks, n * sizeof(uint32_t));
    v->len = n;
    s->ck[1 - slot].tokens.len = 0;
    return OK;
}
size_t session_reusable(Session *s, const uint32_t *prompt, size_t n) {
    Reusable r = session_reusable_in(s->ck, prompt, n);
    if (s->ck[r.pick].tokens.len != s->pos) return 0;
    s->ck_pick = r.pick;
    return r.n;
}
// Continues the live state (session_reusable checked it extends the prompt).
void session_restore(Session *s) {
    if (s->ck[s->ck_pick].tokens.len != s->pos) die("session_restore: restore point is not the live state\n");
}

Err session_new_conv(const Session *s, Gpu *g, size_t cap, Conv *out) {
    (void)s;
    Conv cv = {0};
    cv.cap = cap;
    TRY(gpu_new_buffer(g, kvf_bytes(cap), &cv.kv));
    TRY(gpu_new_buffer(g, RING_BYTES, &cv.ixk));
    TRY(gpu_new_buffer(g, CONV_BYTES, &cv.bk));
    *out = cv;
    return OK;
}
Err session_conv_shrink(const Session *s, Gpu *g, Conv *cv, size_t cap) {
    (void)s;
    sx_buffer *nb;
    TRY(gpu_new_buffer(g, kvf_bytes(cap), &nb));
    sx_buffer_release(cv->kv);
    cv->kv = nb;
    cv->cap = cap;
    for (int k = 0; k < 2; k++) cv->ck[k].tokens.len = 0;
    cv->ptoks.len = 0;
    return OK;
}
#define SWAP(T, a, b) do { T t_ = (a); (a) = (b); (b) = t_; } while (0)
// The held conversation's live position is its restore point's length (session_restore keeps
// it; a conversation without one starts over).
void session_swap_conv(Session *s, Conv *cv) {
    SWAP(size_t, s->cap, cv->cap);
    SWAP(sx_buffer *, s->kv, cv->kv);
    SWAP(sx_buffer *, s->ixk, cv->ixk);
    SWAP(sx_buffer *, s->bk, cv->bk);
    for (int i = 0; i < 2; i++) SWAP(Ckpt, s->ck[i], cv->ck[i]);
    SWAP(VecU32, s->ptoks, cv->ptoks);
    s->pos = MAX(s->ck[0].tokens.len, s->ck[1].tokens.len);
    s->roll = SESSION_NO_ROLL;
}

// ---- not implemented yet -------------------------------------------------------------------

#define NI "NotImplemented"
static const char *pipe_name(const sx_pipeline *p) {
    const Model *m = prof_model;
    if (!m) return "?";
    const sx_pipeline *const *pp = (const sx_pipeline *const *)&m->p;
    for (size_t i = 0; i < sizeof m->p / sizeof(sx_pipeline *); i++)
        if (pp[i] == p) return kernel_names[i];
    return "?";
}
void model_prof_init(void) { prof.on = getenv("SAETTA_PROFILE") != NULL; }
void model_prof_reset(void) {
    for (size_t i = 0; i < prof.n; i++) prof.s[i] = 0, prof.calls[i] = 0;
}
void model_prof_report(size_t tokens) {
    if (!prof.on || tokens == 0) return;
    double tot = 0;
    for (size_t i = 0; i < prof.n; i++) tot += prof.s[i];
    dbg("profile (GPU ms per token, %zu tokens; total %.3f):\n", tokens, tot * 1e3 / (double)tokens);
    for (size_t i = 0; i < prof.n; i++)
        dbg("  %-18s %8.3f ms  %5.1f%%  %6.1f calls\n", prof.name[i], prof.s[i] * 1e3 / (double)tokens, 100 * prof.s[i] / tot,
            (double)prof.calls[i] / (double)tokens);
}
bool model_has_mtp(const Model *m) { (void)m; return false; }
bool model_fast_prefill(const Model *m) { return m->fast_prefill; }
bool model_fast_prefill_available(const Model *m) { (void)m; return true; }
void model_set_fast_prefill(Model *m, bool on) { m->fast_prefill = on; }
static uint32_t greedy_pick(void *ctx, const float *row, bool has_draft, uint32_t draft) {
    (void)ctx, (void)has_draft, (void)draft;
    return model_argmax(row, VOCAB);
}
Err model_spec_verify(Model *m, Session *s, uint32_t x, const uint32_t *drafts, size_t nd, SpecPick pick, void *ctx,
                      SpecResult *out) {
    (void)m, (void)s, (void)x, (void)drafts, (void)nd, (void)pick, (void)ctx, (void)out;
    return NI;
}
Err model_spec_greedy(Model *m, Session *s, uint32_t x, const uint32_t *drafts, size_t nd, SpecResult *out) {
    return model_spec_verify(m, s, x, drafts, nd, greedy_pick, NULL, out);
}
Err model_draft_mtp(Model *m, Session *s, uint32_t x, size_t k, uint32_t *out, size_t *nd) {
    (void)m, (void)s, (void)x, (void)k, (void)out;
    *nd = 0;
    return OK;
}
Err model_lora_load(Arena *a, Model *m, const char *name, const char *dir) { (void)a, (void)m, (void)name, (void)dir; return NI; }
int model_lora_find(const Model *m, const char *name) { (void)m, (void)name; return -1; }
void model_lora_select(Model *m, int idx) { (void)m, (void)idx; }
int model_lora_selected(const Model *m) { (void)m; return -1; }
size_t model_lora_count(const Model *m) { (void)m; return 0; }
const char *model_lora_name(const Model *m, size_t i) { (void)m, (void)i; return NULL; }
bool session_roll_to(Session *s, size_t p, const uint32_t *toks) { (void)s, (void)p, (void)toks; return false; }
Err session_save_point(Session *s, Gpu *g, const uint32_t *toks, size_t n) { (void)s, (void)g, (void)toks, (void)n; return OK; }
void session_start_at_own_point(Session *s, size_t n) { (void)s, (void)n; }
Err session_start_from(Session *s, Gpu *g, const Conv *src, size_t n) { (void)s, (void)g, (void)src, (void)n; return NI; }
void session_push_history(Session *s, uint32_t token) { (void)s, (void)token; }
// Not yet: its restore point is the live state (full-attention KV, the sliding rings, the conv
// states), which needs its own layout; the state store reports StateUnsupported for it.
bool session_state_layout(const Session *s, size_t slot, size_t n, StLayout *out) { (void)s, (void)slot, (void)n, (void)out; return false; }

const ModelArch inkling_arch = MODEL_ARCH_TABLE({
    .model_type = CONFIG_MODEL_TYPE,
    .id = "inkling-2-mini",
    .name = "Inkling 2 Mini",
    .layers = NVIS,
    .width = H,
    .vocab = VOCAB,
    .hidden = H,
    .eos = config_eos_token_ids,
    .n_eos = CONFIG_N_EOS_TOKEN_IDS,
    .max_positions = CONFIG_MODEL_MAX_LENGTH,
    .no_spec = true,
    .chat = CHAT_INKLING,
});

// ---- verification hooks (src/inkling.h) ---------------------------------------------------

bool inkling_has_routes(const Model *m) { return m->arch == &inkling_arch && m->routes != NULL; }
void inkling_set_routes(Model *m, InkRouteMode mode) { m->replay = mode == INK_ROUTES_REPLAY; }
InkRouteStats inkling_route_stats(Model *m) {
    InkRouteStats r = {0, 0, UINT32_MAX};
    if (!m->rstats) return r;
    uint32_t *st = sx_buffer_contents(m->rstats);
    r = (InkRouteStats){st[0], st[1], st[2]};
    st[0] = st[1] = 0;
    st[2] = UINT32_MAX;
    return r;
}
