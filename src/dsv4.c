// DeepSeek-V4-Flash: the pack check, the loader and the decode forward pass, one token per
// step in one command buffer (prefill runs the prompt as successive steps).
//
// The official checkpoint is read in place: dense weights FP8 e4m3 with e8m0 128 x 128 block
// scales (plus BF16 and F32 tensors), routed experts FP4 e2m1 (two per I8 byte) with an e8m0
// scale per 32 values. One tensor per expert matrix; an expert's three weights lie back to
// back (w1, w2, w3) and so do its three scales, so the expert cache reads an expert as two
// ranges. The routed experts are always streamed (src/xstream.h): the pack does not fit.
//
// Per layer (R:695-707, R = the checkpoint's inference/model.py, MIT, Copyright (c) DeepSeek):
// hc_pre and attn_norm; attention (q and kv projections, the compressor and, on CSA layers,
// the indexer; sparse attention over the 128-slot window and the compressed list, with the
// sink; wo_a, wo_b); hc_post; hc_pre and ffn_norm; MoE (shared expert, router, routed experts
// summed in ascending expert id); hc_post. Hash layers 0-2 take their experts from the token
// (tid2eid), mapped to cache slots before the pass; the others stop the command buffer after
// the router until the host has mapped them. After the last layer: hc_head, norm, head.
// Speculation, the draft head, LoRA, restore and prefix points are not implemented.
#include "model.h"

#include <math.h>
#include <stdlib.h>

#include "config_dsv4.h"
#define ARCH_PFX dsv4_
#include "model_rename.h"
#include "dsv4_cmp.h"
#include "dsv4_fp8.h"
#include "dsv4_glue.h"
#include "dsv4_moe.h"
#include "dsv4_ref_cmp.h"
#include "weights.h"
#include "xcache.h"
#include "xstream.h"

#define H CONFIG_HIDDEN_SIZE
#define NLAYERS CONFIG_NUM_HIDDEN_LAYERS
#define VOCAB CONFIG_VOCAB_SIZE
#define NE CONFIG_N_ROUTED_EXPERTS
#define FF CONFIG_MOE_INTERMEDIATE_SIZE
#define HC CONFIG_HC_MULT

#define DSV4_DEFAULT_CACHE 80000000000ull

static const uint32_t dsv4_eos[1] = {(uint32_t)CONFIG_EOS_TOKEN_ID};

// ---- pack check -------------------------------------------------------------------------

typedef struct {
    const char *name; // after "layers.N." (or the full name at top level)
    DType dtype;
    uint64_t d0, d1; // shape; d1 0 for rank 1
} Want;

static size_t check_one(const Pack *pack, const char *name, DType dt, uint64_t d0, uint64_t d1, FILE *out) {
    const TensorInfo *t = pack_get(pack, str(name));
    if (!t) {
        fprintf(out, "  MISSING %s\n", name);
        return 1;
    }
    bool rank_ok = d1 ? t->rank == 2 && t->shape[1] == d1 : t->rank == 1;
    if (t->dtype != dt || !rank_ok || t->shape[0] != d0) {
        fprintf(out, "  MISMATCH %s: %s rank %u [%llu, %llu], want %s [%llu, %llu]\n", name, dtype_name(t->dtype),
                t->rank, (unsigned long long)t->shape[0], (unsigned long long)(t->rank > 1 ? t->shape[1] : 0),
                dtype_name(dt), (unsigned long long)d0, (unsigned long long)d1);
        return 1;
    }
    return 0;
}

// Expert e of layer l: its two ranges (weights w1 w2 w3, scales w1 w2 w3), or false with a
// message when the six tensors are missing or not back to back in one file.
static bool expert_ranges(Arena *a, const Pack *pack, size_t l, size_t e, XcRange r[2], FILE *out) {
    static const char *const mat[3] = {"w1", "w2", "w3"};
    static const uint64_t wlen[3] = {FF * H / 2, H * FF / 2, FF * H / 2};
    static const uint64_t slen[3] = {FF * H / 32, H * FF / 32, FF * H / 32};
    for (int k = 0; k < 2; k++) {
        uint64_t next = 0;
        uint16_t file = 0;
        for (int i = 0; i < 3; i++) {
            const char *nm = arena_printf(a, "layers.%zu.ffn.experts.%zu.%s.%s", l, e, mat[i], k ? "scale" : "weight");
            const TensorInfo *t = pack_get(pack, str(nm));
            uint64_t want = k ? slen[i] : wlen[i];
            if (!t || t->nbytes != want || t->dtype != (k ? DTYPE_F8_E8M0 : DTYPE_I8)) {
                if (out) fprintf(out, "  EXPERT %s: missing or wrong size\n", nm);
                return false;
            }
            if (i == 0) {
                r[k].off = t->offset;
                file = t->file;
            } else if (t->offset != next || t->file != file) {
                if (out) fprintf(out, "  EXPERT layers.%zu.ffn.experts.%zu: %s not contiguous\n", l, e, k ? "scales" : "weights");
                return false;
            }
            next = t->offset + t->nbytes;
        }
        r[k].len = next - r[k].off;
        r[k].file = file;
    }
    return true;
}

size_t model_check_pack(Arena *a, const Pack *pack, FILE *out) {
    const DType B = DTYPE_BF16, F = DTYPE_F32, Q = DTYPE_F8_E4M3, S = DTYPE_F8_E8M0;
    size_t bad = 0, n = 0;
    const Want top[] = {
        {"embed.weight", B, VOCAB, H}, {"head.weight", B, VOCAB, H}, {"norm.weight", B, H, 0},
        {"hc_head_fn", F, HC, HC * H}, {"hc_head_base", F, HC, 0},  {"hc_head_scale", F, 1, 0},
    };
    for (size_t i = 0; i < COUNT(top); i++, n++) bad += check_one(pack, top[i].name, top[i].dtype, top[i].d0, top[i].d1, out);
    const size_t QL = CONFIG_Q_LORA_RANK, NH = CONFIG_NUM_ATTENTION_HEADS, HD = CONFIG_HEAD_DIM, OL = CONFIG_O_LORA_RANK;
    const size_t OG = CONFIG_O_GROUPS, IH = CONFIG_INDEX_N_HEADS, ID = CONFIG_INDEX_HEAD_DIM, MIX = (2 + HC) * HC;
    const Want every[] = {
        {"attn.wq_a.weight", Q, QL, H},          {"attn.wq_a.scale", S, QL / 128, H / 128},
        {"attn.wq_b.weight", Q, NH * HD, QL},    {"attn.wq_b.scale", S, NH * HD / 128, QL / 128},
        {"attn.wkv.weight", Q, HD, H},           {"attn.wkv.scale", S, HD / 128, H / 128},
        {"attn.wo_a.weight", Q, OG * OL, NH * HD / OG}, {"attn.wo_a.scale", S, OG * OL / 128, NH * HD / OG / 128},
        {"attn.wo_b.weight", Q, H, OG * OL},     {"attn.wo_b.scale", S, H / 128, OG * OL / 128},
        {"attn.q_norm.weight", B, QL, 0},        {"attn.kv_norm.weight", B, HD, 0},
        {"attn.attn_sink", F, NH, 0},            {"attn_norm.weight", B, H, 0},
        {"ffn_norm.weight", B, H, 0},            {"ffn.gate.weight", B, NE, H},
        {"ffn.shared_experts.w1.weight", Q, FF, H}, {"ffn.shared_experts.w1.scale", S, FF / 128, H / 128},
        {"ffn.shared_experts.w3.weight", Q, FF, H}, {"ffn.shared_experts.w3.scale", S, FF / 128, H / 128},
        {"ffn.shared_experts.w2.weight", Q, H, FF}, {"ffn.shared_experts.w2.scale", S, H / 128, FF / 128},
        {"hc_attn_fn", F, MIX, HC * H},          {"hc_attn_base", F, MIX, 0}, {"hc_attn_scale", F, 3, 0},
        {"hc_ffn_fn", F, MIX, HC * H},           {"hc_ffn_base", F, MIX, 0},  {"hc_ffn_scale", F, 3, 0},
    };
    const Want csa[] = {
        {"attn.compressor.wkv.weight", B, 2 * HD, H}, {"attn.compressor.wgate.weight", B, 2 * HD, H},
        {"attn.compressor.ape", F, 4, 2 * HD},        {"attn.compressor.norm.weight", B, HD, 0},
        {"attn.indexer.compressor.wkv.weight", B, 2 * ID, H}, {"attn.indexer.compressor.wgate.weight", B, 2 * ID, H},
        {"attn.indexer.compressor.ape", F, 4, 2 * ID},        {"attn.indexer.compressor.norm.weight", B, ID, 0},
        {"attn.indexer.wq_b.weight", Q, IH * ID, QL},         {"attn.indexer.wq_b.scale", S, IH * ID / 128, QL / 128},
        {"attn.indexer.weights_proj.weight", B, IH, H},
    };
    const Want hca[] = {
        {"attn.compressor.wkv.weight", B, HD, H}, {"attn.compressor.wgate.weight", B, HD, H},
        {"attn.compressor.ape", F, 128, HD},      {"attn.compressor.norm.weight", B, HD, 0},
    };
    for (size_t l = 0; l < NLAYERS; l++) {
        const uint32_t ratio = config_compress_ratios[l];
        const Want *sets[2] = {every, ratio == 4 ? csa : ratio == 128 ? hca : NULL};
        size_t counts[2] = {COUNT(every), ratio == 4 ? COUNT(csa) : ratio == 128 ? COUNT(hca) : 0};
        for (int s = 0; s < 2; s++)
            for (size_t i = 0; i < counts[s]; i++, n++)
                bad += check_one(pack, arena_printf(a, "layers.%zu.%s", l, sets[s][i].name), sets[s][i].dtype, sets[s][i].d0,
                                 sets[s][i].d1, out);
        n++;
        if (l < CONFIG_NUM_HASH_LAYERS)
            bad += check_one(pack, arena_printf(a, "layers.%zu.ffn.gate.tid2eid", l), DTYPE_I64, VOCAB, CONFIG_NUM_EXPERTS_PER_TOK, out);
        else
            bad += check_one(pack, arena_printf(a, "layers.%zu.ffn.gate.bias", l), F, NE, 0, out);
        for (size_t e = 0; e < NE; e++, n++) {
            XcRange r[2];
            if (!expert_ranges(a, pack, l, e, r, out)) bad++;
        }
    }
    fprintf(out, "tensors and experts: %zu checked, %zu problems\n", n, bad);
    return bad;
}

// ---- model --------------------------------------------------------------------------------

#define QL CONFIG_Q_LORA_RANK      // 1024
#define NH CONFIG_NUM_ATTENTION_HEADS // 64
#define HD CONFIG_HEAD_DIM         // 512
#define OL CONFIG_O_LORA_RANK      // 1024
#define OG CONFIG_O_GROUPS         // 8
#define IH CONFIG_INDEX_N_HEADS    // 64
#define ID CONFIG_INDEX_HEAD_DIM   // 128
#define WIN CONFIG_SLIDING_WINDOW  // 128
#define ITOPK CONFIG_INDEX_TOPK    // 512
#define TOPK CONFIG_NUM_EXPERTS_PER_TOK // 6
#define NHASH CONFIG_NUM_HASH_LAYERS    // 3
#define MIX ((2 + HC) * HC)        // 24
#define HCW (HC * H)               // 16384, the 4 streams of a token
#define NPAIR 32                   // RoPE pairs (64 rope dims)
#define SLICES 8                   // dsv4_hc_mix slices per row
#define MAX_STOPS 48               // stops per pass (40 in a decode step)

// Constant structs of the kernels (same layouts as the .metal files).
typedef struct {
    uint32_t nmix, T;
} MixP; // DsMixP
typedef struct {
    uint32_t T;
} HcP; // DsHcP
typedef struct {
    uint32_t n, T;
} NormP; // DsNormP
typedef struct {
    uint32_t pos0, T, rope_pos0, cap, ls;
} AttnP; // DsAttnP

// A BF16 matrix in the pack, for dsv4_bf16_mv: the binding (16-byte aligned) and the byte shift.
typedef struct {
    sx_arg a;
    uint32_t shift;
} Mat;

typedef struct {
    uint32_t ratio; // 0 window only, 4 CSA, 128 HCA
    Dsv4Fp8Lin wq_a, wq_b, wkv, wo_a, wo_b, sw1, sw2, sw3, iwq_b;
    // copies in Model.params (the pack's small tensors are not 16-byte aligned)
    sx_arg q_norm, kv_norm, attn_norm, ffn_norm, sink;
    sx_arg hca_fn, hca_base, hca_scale, hcf_fn, hcf_base, hcf_scale;
    sx_arg gate_w, gate_bias;
    sx_arg c_ape, c_norm, i_ape, i_norm;
    Mat c_wkv, c_wgate, i_wkv, i_wgate, i_wp;
    const uint8_t *tid2eid; // hash layers: the pack's int64 [vocab][6], read on the host
} DsLayer;

typedef struct {
    sx_pipeline *hc_mix, *hc_split_norm, *hc_head_norm, *hc_post, *rmsnorm, *r_logits, *r_select, *r_hash;
    sx_pipeline *a_q, *a_kv, *a_store, *a_scores, *a_pv;
    sx_pipeline *c_step[3]; // CSA, HCA, the indexer's compressor
    sx_pipeline *i_q, *i_score, *i_topk;
    sx_pipeline *m_aq, *m_gu, *m_dn;
    sx_pipeline *mv, *round, *to_bf, *from_bf, *sact, *madd, *copy4;
} Pipes;

struct Model {
    const ModelArch *arch; // first: model_api.c dispatches on it
    const Pack *pack;
    Gpu *g;
    Pipes p;
    Dsv4Fp8Pipes f8;
    DsLayer L[NLAYERS];
    sx_buffer *params;
    sx_arg hc_head_fn, hc_head_base, hc_head_scale, norm_w;
    Mat head;
    const uint8_t *embed; // BF16 [vocab][4096] in the pack, read on the host
    // one token's scratch (f32 unless noted)
    sx_buffer *st;               // [4][4096] the residual streams (BF16 values)
    sx_buffer *part, *post, *comb; // mixer partials, post [4], comb [4][4]
    sx_buffer *X;                // [4096] block input after its norm
    sx_buffer *xc, *xsc;         // act_quant codes (half [8192]) and scales [64]
    sx_buffer *qa, *qr, *qb, *q; // wq_a out, q_norm out [1024]; wq_b out [32768], as BF16 bits
    sx_buffer *kva, *kvb, *kvrow; // wkv out [512], as BF16 bits, the token's KV row (BF16 bits)
    sx_buffer *ckv, *csc;        // compressor kv and score [1024]
    sx_buffer *iq, *wp, *ikv, *isc, *qx, *iw, *ilist; // indexer: q [8192], weights_proj [64], its
                                 // compressor's kv and score [256], rotated q, head weights, top-512
    sx_buffer *o, *of, *woa, *f; // attention out (BF16 bits, then f32) [32768], wo_a out [8192], block out
    sx_buffer *rlog, *xq, *hq, *yr, *g1, *u3, *sh, *ys; // router logits, routed and shared expert
    sx_buffer *rids, *rw, *slots, *hids, *zero; // per layer [43][8]: router ids, weights, cache
                                 // slots; hash layers' sorted ids; a zero token index
    sx_buffer *cnt;              // compressed list lengths [4]: 0, CSA, HCA
    sx_buffer *dummy, *logits;
    Dsv4MoeP moe;
    // SAETTA_DSV4_FORCE=<file> (diagnostic): reference streams [T][layers][16384] f32; layer l of
    // the token at position t < force_n starts from row (t, l - 1) instead of the engine's own
    // streams, so each layer's error is measured on exact inputs.
    sx_buffer *force;
    size_t force_n;
    XStream xs;
    XSvc stops[MAX_STOPS];
    uint32_t nstops;
    bool failed; // a stop could not be encoded
};

// The per-session state: the base Session first (the engine sees only it).
typedef struct {
    Session s;
    sx_buffer *ring;   // [43][128][512] BF16 bits, the window
    sx_buffer *ccache; // compressed entries of each compressed layer [cap / ratio + 1][512] BF16 bits
    sx_buffer *icache; // indexer keys of each CSA layer [cap / 4 + 1][128] BF16 bits
    sx_buffer *cst;    // compressor states (kv, score) of each compressed layer and indexer, f32
    sx_buffer *rope_w, *rope_c; // [cap][32] (cos, sin): theta 10000; compressed (YaRN)
    sx_buffer *iscore; // indexer scores [cap / 4 + 32]
    sx_buffer *ident;  // 0, 1, ...: the HCA index list
    sx_buffer *scores; // attention scores [64][ls]
    size_t cc[NLAYERS], ic[NLAYERS], kvs[NLAYERS], scs[NLAYERS], ikvs[NLAYERS], iscs[NLAYERS]; // byte offsets
    uint32_t ls;
} DsSession;

static DsSession *dss(Session *s) { return (DsSession *)s; }

static XCache *ds_report; // for model_prof_report (one model per process)

static inline sx_arg B(sx_buffer *b) { return (sx_arg){b, 0, NULL, 0}; }
static inline sx_arg AT(sx_buffer *b, size_t off) { return (sx_arg){b, off, NULL, 0}; }
#define BYTES(v) gpu_bytes_arg(&(v), sizeof(v))
#define DISP(c, p, gx, gy, gz, th, ...)                                            \
    do {                                                                          \
        sx_arg a_[] = {__VA_ARGS__};                                              \
        sx_cmd_dispatch3((c), (p), a_, COUNT(a_), (gx), (gy), (gz), (th));        \
    } while (0)

// ---- load ---------------------------------------------------------------------------------

// The pack's small tensors copied into one GPU buffer at 16-byte offsets. A first pass with
// base NULL only measures.
typedef struct {
    Arena *a;
    const Pack *pack;
    sx_buffer *buf;
    uint8_t *base;
    size_t used;
    Err err;
} Prm;

static sx_arg prm(Prm *P, const char *name, DType dt) {
    const TensorInfo *t = pack_get(P->pack, str(name));
    if (!t || t->dtype != dt) {
        if (!P->err) dbg("model: %s missing or not %s\n", name, dtype_name(dt));
        P->err = "MissingTensor";
        return (sx_arg){0};
    }
    size_t off = ALIGN_UP(P->used, 16);
    if (P->base) memcpy(P->base + off, pack_data(P->pack, t), t->nbytes);
    P->used = off + t->nbytes;
    return AT(P->buf, off);
}
static sx_arg prml(Prm *P, size_t l, const char *name, DType dt) {
    return prm(P, arena_printf(P->a, "layers.%zu.%s", l, name), dt);
}

static Err fp8(const Gpu *g, const Pack *pack, const char *pfx, uint32_t out, uint32_t in, Dsv4Fp8Lin *l) {
    char nm[160];
    snprintf(nm, sizeof nm, "%s.weight", pfx);
    const TensorInfo *w = pack_get(pack, str(nm));
    snprintf(nm, sizeof nm, "%s.scale", pfx);
    const TensorInfo *s = pack_get(pack, str(nm));
    if (!w || !s) return "MissingTensor";
    l->w = gpu_tensor_arg(g, w, &l->wshift);
    l->s = gpu_tensor_arg(g, s, &l->sshift);
    l->out = out;
    l->in = in;
    return OK;
}

static Err mat(const Gpu *g, const Pack *pack, const char *name, Mat *m) {
    const TensorInfo *t = pack_get(pack, str(name));
    if (!t || t->dtype != DTYPE_BF16) return "MissingTensor";
    m->a = gpu_tensor_arg(g, t, &m->shift);
    if (m->shift % 4) return "UnalignedTensor"; // dsv4_bf16_mv reads 32-bit words
    return OK;
}

// Everything but the routed experts. With P->base NULL only the params size is measured.
static Err load_layers(Model *m, Prm *P) {
    const Pack *pack = m->pack;
    const DType B16 = DTYPE_BF16, F = DTYPE_F32;
    m->hc_head_fn = prm(P, "hc_head_fn", F);
    m->hc_head_base = prm(P, "hc_head_base", F);
    m->hc_head_scale = prm(P, "hc_head_scale", F);
    m->norm_w = prm(P, "norm.weight", B16);
    for (size_t l = 0; l < NLAYERS; l++) {
        DsLayer *L = &m->L[l];
        L->ratio = config_compress_ratios[l];
        L->q_norm = prml(P, l, "attn.q_norm.weight", B16);
        L->kv_norm = prml(P, l, "attn.kv_norm.weight", B16);
        L->attn_norm = prml(P, l, "attn_norm.weight", B16);
        L->ffn_norm = prml(P, l, "ffn_norm.weight", B16);
        L->sink = prml(P, l, "attn.attn_sink", F);
        L->hca_fn = prml(P, l, "hc_attn_fn", F);
        L->hca_base = prml(P, l, "hc_attn_base", F);
        L->hca_scale = prml(P, l, "hc_attn_scale", F);
        L->hcf_fn = prml(P, l, "hc_ffn_fn", F);
        L->hcf_base = prml(P, l, "hc_ffn_base", F);
        L->hcf_scale = prml(P, l, "hc_ffn_scale", F);
        L->gate_w = prml(P, l, "ffn.gate.weight", B16);
        if (l >= NHASH) L->gate_bias = prml(P, l, "ffn.gate.bias", F);
        if (L->ratio) {
            L->c_ape = prml(P, l, "attn.compressor.ape", F);
            L->c_norm = prml(P, l, "attn.compressor.norm.weight", B16);
        }
        if (L->ratio == 4) {
            L->i_ape = prml(P, l, "attn.indexer.compressor.ape", F);
            L->i_norm = prml(P, l, "attn.indexer.compressor.norm.weight", B16);
        }
        if (P->err) return P->err;
        if (!P->base) continue;
        // pack bindings (once)
        const char *pf = arena_printf(P->a, "layers.%zu.", l);
#define PF(s) arena_printf(P->a, "%s%s", pf, s)
        TRY(fp8(m->g, pack, PF("attn.wq_a"), QL, H, &L->wq_a));
        TRY(fp8(m->g, pack, PF("attn.wq_b"), NH * HD, QL, &L->wq_b));
        TRY(fp8(m->g, pack, PF("attn.wkv"), HD, H, &L->wkv));
        TRY(fp8(m->g, pack, PF("attn.wo_a"), OG * OL, NH * HD / OG, &L->wo_a));
        TRY(fp8(m->g, pack, PF("attn.wo_b"), H, OG * OL, &L->wo_b));
        TRY(fp8(m->g, pack, PF("ffn.shared_experts.w1"), FF, H, &L->sw1));
        TRY(fp8(m->g, pack, PF("ffn.shared_experts.w3"), FF, H, &L->sw3));
        TRY(fp8(m->g, pack, PF("ffn.shared_experts.w2"), H, FF, &L->sw2));
        if (L->ratio) {
            TRY(mat(m->g, pack, PF("attn.compressor.wkv.weight"), &L->c_wkv));
            TRY(mat(m->g, pack, PF("attn.compressor.wgate.weight"), &L->c_wgate));
        }
        if (L->ratio == 4) {
            TRY(fp8(m->g, pack, PF("attn.indexer.wq_b"), IH * ID, QL, &L->iwq_b));
            TRY(mat(m->g, pack, PF("attn.indexer.compressor.wkv.weight"), &L->i_wkv));
            TRY(mat(m->g, pack, PF("attn.indexer.compressor.wgate.weight"), &L->i_wgate));
            TRY(mat(m->g, pack, PF("attn.indexer.weights_proj.weight"), &L->i_wp));
        }
        if (l < NHASH) {
            const TensorInfo *t = pack_get(pack, str(PF("ffn.gate.tid2eid")));
            if (!t) return "MissingTensor";
            L->tid2eid = pack_data(pack, t);
        }
#undef PF
    }
    return OK;
}

static Err mkpipe(Gpu *g, const char *src, const char *name, sx_pipeline **out) { return gpu_pipeline(g, src, name, out); }

static Err load_pipes(Gpu *g, Pipes *p) {
    const char *hc = dsv4_hc_lib_src, *at = dsv4_attn_lib_src, *cm = dsv4_cmp_src, *mo = dsv4_moe_src, *gl = dsv4_glue_src;
    TRY(mkpipe(g, hc, "dsv4_hc_mix", &p->hc_mix));
    TRY(mkpipe(g, hc, "dsv4_hc_split_norm", &p->hc_split_norm));
    TRY(mkpipe(g, hc, "dsv4_hc_head_norm", &p->hc_head_norm));
    TRY(mkpipe(g, hc, "dsv4_hc_post", &p->hc_post));
    TRY(mkpipe(g, hc, "dsv4_rmsnorm", &p->rmsnorm));
    TRY(mkpipe(g, hc, "dsv4_router_logits", &p->r_logits));
    TRY(mkpipe(g, hc, "dsv4_router_select", &p->r_select));
    TRY(mkpipe(g, hc, "dsv4_router_hash", &p->r_hash));
    TRY(mkpipe(g, at, "dsv4_attn_q", &p->a_q));
    TRY(mkpipe(g, at, "dsv4_attn_kv", &p->a_kv));
    TRY(mkpipe(g, at, "dsv4_attn_kv_store", &p->a_store));
    TRY(mkpipe(g, at, "dsv4_attn_scores", &p->a_scores));
    TRY(mkpipe(g, at, "dsv4_attn_pv", &p->a_pv));
    TRY(mkpipe(g, cm, "dsv4_cmp_step_csa", &p->c_step[0]));
    TRY(mkpipe(g, cm, "dsv4_cmp_step_hca", &p->c_step[1]));
    TRY(mkpipe(g, cm, "dsv4_cmp_step_idx", &p->c_step[2]));
    TRY(mkpipe(g, cm, "dsv4_idx_q", &p->i_q));
    TRY(mkpipe(g, cm, "dsv4_idx_score", &p->i_score));
    TRY(mkpipe(g, cm, "dsv4_idx_topk", &p->i_topk));
    TRY(mkpipe(g, mo, "dsv4_act_quant", &p->m_aq));
    TRY(mkpipe(g, mo, "dsv4_moe_gate_up", &p->m_gu));
    TRY(mkpipe(g, mo, "dsv4_moe_down", &p->m_dn));
    TRY(mkpipe(g, gl, "dsv4_bf16_mv", &p->mv));
    TRY(mkpipe(g, gl, "dsv4_round_bf16", &p->round));
    TRY(mkpipe(g, gl, "dsv4_to_bf16", &p->to_bf));
    TRY(mkpipe(g, gl, "dsv4_from_bf16", &p->from_bf));
    TRY(mkpipe(g, gl, "dsv4_shared_act", &p->sact));
    TRY(mkpipe(g, gl, "dsv4_moe_add", &p->madd));
    TRY(mkpipe(g, gl, "dsv4_copy4", &p->copy4));
    return OK;
}

static Err newbuf(Gpu *g, size_t bytes, sx_buffer **out) { return gpu_new_buffer(g, bytes, out); }

static Err load_scratch(Model *m) {
    Gpu *g = m->g;
    struct {
        sx_buffer **b;
        size_t bytes;
    } bs[] = {
        {&m->st, HCW * 4},           {&m->part, SLICES * (MIX + 1) * 4}, {&m->post, 16 * 4}, {&m->comb, 16 * 4},
        {&m->X, H * 4},             {&m->xc, OG * OL * 2},              {&m->xsc, 64 * 4},   {&m->qa, QL * 4},
        {&m->qr, QL * 4},           {&m->qb, NH * HD * 4},              {&m->q, NH * HD * 2}, {&m->kva, HD * 4},
        {&m->kvb, HD * 2},          {&m->kvrow, HD * 2},                {&m->ckv, 2 * HD * 4}, {&m->csc, 2 * HD * 4},
        {&m->iq, IH * ID * 4},      {&m->wp, IH * 4},                   {&m->ikv, 2 * ID * 4}, {&m->isc, 2 * ID * 4},
        {&m->qx, IH * ID * 4},      {&m->iw, IH * 4},                   {&m->ilist, ITOPK * 4}, {&m->o, NH * HD * 2},
        {&m->of, NH * HD * 4},      {&m->woa, OG * OL * 4},             {&m->f, H * 4},     {&m->rlog, NE * 4},
        {&m->xq, H * 4},            {&m->hq, TOPK * FF * 4},            {&m->yr, H * 4},    {&m->g1, FF * 4},
        {&m->u3, FF * 4},           {&m->sh, FF * 4},                   {&m->ys, H * 4},    {&m->rids, NLAYERS * 8 * 4},
        {&m->rw, NLAYERS * 8 * 4},  {&m->slots, NLAYERS * 8 * 4},       {&m->hids, NHASH * 8 * 4}, {&m->zero, 16},
        {&m->cnt, 16},              {&m->dummy, 1u << 16},              {&m->logits, VOCAB * 4},
    };
    for (size_t i = 0; i < COUNT(bs); i++) TRY(newbuf(g, bs[i].bytes, bs[i].b));
    return OK;
}

Err model_load(Arena *a, const Pack *pack, Gpu *g, Model **out) {
    FILE *devnull = fopen("/dev/null", "w");
    size_t bad = model_check_pack(a, pack, devnull ? devnull : stderr);
    if (devnull) fclose(devnull);
    if (bad) {
        (void)model_check_pack(a, pack, stderr);
        return "BadPack";
    }
    if (!g->segs) return "NotStreamed"; // gpu_init wraps a DeepSeek-V4 pack as segments
    // the expert cache layout: two ranges per expert (weights, scales), validated above
    XcLayer *xl = ARENA_ARRAY(a, XcLayer, NLAYERS);
    if (!xl) return "OutOfMemory";
    for (size_t l = 0; l < NLAYERS; l++) {
        XcRange *r = ARENA_ARRAY(a, XcRange, NE * 2);
        if (!r) return "OutOfMemory";
        for (size_t e = 0; e < NE; e++)
            if (!expert_ranges(a, pack, l, e, &r[e * 2], NULL)) return "BadPack";
        xl[l] = (XcLayer){.ne = (uint32_t)NE, .ranges = r};
    }
    Model *m = ARENA_NEW(a, Model);
    if (!m) return "OutOfMemory";
    memset(m, 0, sizeof *m);
    m->arch = &dsv4_arch;
    m->pack = pack;
    m->g = g;
    TRY(load_pipes(g, &m->p));
    TRY(dsv4_fp8_init(g, &m->f8));
    Prm P = {.a = a, .pack = pack};
    TRY(load_layers(m, &P));
    TRY(newbuf(g, P.used + 16, &P.buf));
    m->params = P.buf;
    P.base = sx_buffer_contents(P.buf);
    P.used = 0;
    TRY(load_layers(m, &P));
    TRY(mat(g, pack, "head.weight", &m->head));
    m->embed = pack_data(pack, pack_get(pack, S("embed.weight")));
    TRY(load_scratch(m));
    const char *fp = getenv("SAETTA_DSV4_FORCE");
    if (fp) {
        char *fb;
        size_t fl;
        TRY(read_file(fp, (size_t)1 << 32, &fb, &fl));
        m->force_n = fl / (NLAYERS * HCW * 4);
        TRY(newbuf(g, MAX(fl, (size_t)16), &m->force));
        memcpy(sx_buffer_contents(m->force), fb, fl);
        free(fb);
        dbg("model: SAETTA_DSV4_FORCE: layer inputs from %s for the first %zu positions\n", fp, m->force_n);
    }
    // the expert cache: SAETTA_EXPERT_CACHE_GB, else the largest that fits less a margin, at
    // most DSV4_DEFAULT_CACHE: a cache close to the whole working set pushes the resident weights
    // and the rest of the system out of memory (a 110 GiB cache swapped on a 128 GB machine)
    uint64_t keep = 0;
    Str k;
    void *v;
    for (size_t it = 0; strmap_next(&pack->tensors, &it, &k, &v);)
        if (!gpu_streamed_tensor(k) && !str_starts(k, S("mtp."))) keep += ((const TensorInfo *)v)->nbytes;
    uint64_t xbytes = xc_requested_bytes();
    if (!xbytes) {
        const uint64_t margin = 4000000000ull;
        xbytes = xs_fit_bytes(g, keep, margin);
        if (xbytes > DSV4_DEFAULT_CACHE) xbytes = DSV4_DEFAULT_CACHE;
        dbg("model: SAETTA_EXPERT_CACHE_GB unset: expert cache of %.1f GB (at most 80, and what fits next to %.1f GB "
            "resident, less %.1f GB)\n",
            (double)xbytes / 1e9, (double)keep / 1e9, (double)margin / 1e9);
    }
    TRY(xs_fits(g, xbytes, keep));
    TRY(xs_open(a, pack, g, xl, NLAYERS, 2, xbytes, &m->xs));
    m->moe = (Dsv4MoeP){.T = 1, .w_stride = xc_stride(m->xs.xc, 0), .s_stride = xc_stride(m->xs.xc, 1), .keep_h = 0,
                        .w_off = {DSV4_W1_OFF, DSV4_W2_OFF, DSV4_W3_OFF, 0}, .s_off = {DSV4_S1_OFF, DSV4_S2_OFF, DSV4_S3_OFF, 0}};
    ds_report = m->xs.xc;
    dbg("model: DeepSeek-V4-Flash, %.2f GB of params copied, streamed experts: cache of %u slots (%.1f GB)\n",
        (double)P.used / 1e9, xc_slots(m->xs.xc), (double)xc_bytes(m->xs.xc) / 1e9);
    *out = m;
    return OK;
}

// ---- encoding -----------------------------------------------------------------------------

static void ew(sx_cmd *c, sx_pipeline *p, sx_arg x, sx_arg y, uint32_t n) {
    DISP(c, p, (n + 255) / 256, 1, 1, 256, x, y, gpu_u32_arg(&n));
}
static void enc_round(Model *m, sx_cmd *c, sx_buffer *x, uint32_t n) {
    DISP(c, m->p.round, (n + 255) / 256, 1, 1, 256, B(x), gpu_u32_arg(&n));
}
static void enc_mv(Model *m, sx_cmd *c, const Mat *w, uint32_t K, uint32_t N, sx_buffer *x, sx_buffer *y, bool round) {
    Dsv4GlueMvP p = {K, N, w->shift, round};
    DISP(c, m->p.mv, (N + 7) / 8, 1, 1, 256, w->a, B(x), B(y), BYTES(p));
}
static void enc_aq(Model *m, sx_cmd *c, sx_buffer *x, uint32_t K) {
    dsv4_encode_act_quant(&m->f8, c, B(x), K, B(m->xc), B(m->xsc), 1, K);
}
static void enc_lin(Model *m, sx_cmd *c, const Dsv4Fp8Lin *l, sx_buffer *y) {
    dsv4_encode_linear(&m->f8, c, l, B(m->xc), B(m->xsc), B(y), 1, l->out);
}
// One hyper-connection mixer: the block input X (with its norm) and post, comb.
static void enc_hc_pre(Model *m, sx_cmd *c, sx_arg fn, sx_arg base, sx_arg scale, sx_arg norm) {
    MixP mp = {MIX, 1};
    DISP(c, m->p.hc_mix, MIX, SLICES, 1, 128, fn, B(m->st), B(m->part), BYTES(mp));
    DISP(c, m->p.hc_split_norm, 1, 1, 1, 256, B(m->part), scale, base, B(m->st), B(m->X), B(m->post), B(m->comb), norm);
}
static void enc_hc_post(Model *m, sx_cmd *c) {
    HcP hp = {1};
    DISP(c, m->p.hc_post, H / 4 / 256, 1, 1, 256, B(m->f), B(m->st), B(m->post), B(m->comb), B(m->st), BYTES(hp));
}

// Attention of layer l for the token at pos (R:497-547), from the streams to hc_post.
static void enc_attn(Model *m, DsSession *d, sx_cmd *c, size_t l, uint32_t pos) {
    const DsLayer *L = &m->L[l];
    const uint32_t ratio = L->ratio;
    enc_hc_pre(m, c, L->hca_fn, L->hca_base, L->hca_scale, L->attn_norm);
    // q and kv projections
    enc_aq(m, c, m->X, H);
    enc_lin(m, c, &L->wq_a, m->qa);
    enc_lin(m, c, &L->wkv, m->kva);
    enc_round(m, c, m->qa, QL);
    NormP np = {QL, 1};
    DISP(c, m->p.rmsnorm, 1, 1, 1, 256, B(m->qa), L->q_norm, B(m->qr), BYTES(np));
    enc_aq(m, c, m->qr, QL);
    enc_lin(m, c, &L->wq_b, m->qb);
    if (ratio == 4) enc_lin(m, c, &L->iwq_b, m->iq);
    ew(c, m->p.to_bf, B(m->qb), B(m->q), NH * HD);
    AttnP ap = {pos, 1, 0, ITOPK, d->ls};
    sx_arg cs = B(ratio ? d->rope_c : d->rope_w);
    DISP(c, m->p.a_q, NH, 1, 1, 128, B(m->q), cs, BYTES(ap));
    ew(c, m->p.to_bf, B(m->kva), B(m->kvb), HD);
    DISP(c, m->p.a_kv, 1, 1, 1, 128, B(m->kvb), L->kv_norm, cs, B(m->kvrow), BYTES(ap));
    // compressor, and for CSA the indexer first (R:517, R:423)
    uint32_t n = 0;
    sx_arg comp = B(m->dummy), list = B(m->dummy), cnt = AT(m->cnt, 0);
    if (ratio) {
        const uint32_t cd = ratio == 4 ? 2 * HD : HD;
        enc_mv(m, c, &L->c_wkv, H, cd, m->X, m->ckv, false);
        enc_mv(m, c, &L->c_wgate, H, cd, m->X, m->csc, false);
        if (ratio == 4) {
            enc_round(m, c, m->iq, IH * ID);
            enc_mv(m, c, &L->i_wp, H, IH, m->X, m->wp, true);
            enc_mv(m, c, &L->i_wkv, H, 2 * ID, m->X, m->ikv, false);
            enc_mv(m, c, &L->i_wgate, H, 2 * ID, m->X, m->isc, false);
            const bool close = (pos + 1) % 4 == 0;
            Dsv4CmpP cp = {pos, 1, pos / 4, close, 0, {0}};
            const size_t crow = close ? pos + 1 - 4 : 0;
            DISP(c, m->p.c_step[2], 1, 1, 1, ID, B(m->ikv), B(m->isc), L->i_ape, AT(d->cst, d->ikvs[l]),
                 AT(d->cst, d->iscs[l]), L->i_norm, AT(d->rope_c, crow * NPAIR * 8), AT(d->icache, d->ic[l]), B(m->dummy),
                 BYTES(cp));
            DISP(c, m->p.i_q, IH, 1, 1, ID, B(m->iq), B(m->wp), AT(d->rope_c, (size_t)pos * NPAIR * 8), B(m->qx), B(m->iw));
            const uint32_t E = (pos + 1) / 4;
            if (E) {
                Dsv4IdxP ip = {E, E, 0, 0};
                DISP(c, m->p.i_score, (E + 31) / 32, 1, 1, 128, AT(d->icache, d->ic[l]), B(m->qx), B(m->iw), B(d->iscore),
                     BYTES(ip));
                DISP(c, m->p.i_topk, 1, 1, 1, 1024, B(d->iscore), B(m->ilist), BYTES(ip));
            }
            n = MIN((uint32_t)ITOPK, E);
            list = B(m->ilist);
            cnt = AT(m->cnt, 4);
        } else {
            n = (pos + 1) / 128;
            list = B(d->ident);
            cnt = AT(m->cnt, 8);
        }
        const bool close = (pos + 1) % ratio == 0;
        Dsv4CmpP cp = {pos, 1, pos / ratio, close, 0, {0}};
        const size_t crow = close ? pos + 1 - ratio : 0;
        DISP(c, m->p.c_step[ratio == 4 ? 0 : 1], 1, 1, 1, HD, B(m->ckv), B(m->csc), L->c_ape, AT(d->cst, d->kvs[l]),
             AT(d->cst, d->scs[l]), L->c_norm, AT(d->rope_c, crow * NPAIR * 8), AT(d->ccache, d->cc[l]), B(m->dummy),
             BYTES(cp));
        comp = AT(d->ccache, d->cc[l]);
    }
    // sparse attention over [window | compressed list] with the sink, then the window write
    const sx_arg ring = AT(d->ring, l * WIN * HD * 2);
    const uint32_t nb = (WIN + n + 63) / 64;
    DISP(c, m->p.a_scores, nb, NH / 16, 1, 128, B(m->q), ring, B(m->kvrow), comp, list, cnt, B(d->scores), BYTES(ap));
    DISP(c, m->p.a_pv, HD / 64, NH / 16, 1, 128, B(d->scores), ring, B(m->kvrow), comp, list, cnt, L->sink, cs, B(m->o),
         BYTES(ap));
    DISP(c, m->p.a_store, 1, 1, 1, 128, B(m->kvrow), ring, BYTES(ap));
    // output projection (R:542-547)
    ew(c, m->p.from_bf, B(m->o), B(m->of), NH * HD);
    dsv4_encode_grouped(&m->f8, c, &L->wo_a, OG, B(m->of), B(m->woa), 1, OG * OL);
    enc_round(m, c, m->woa, OG * OL);
    enc_aq(m, c, m->woa, OG * OL);
    enc_lin(m, c, &L->wo_b, m->f);
    enc_round(m, c, m->f, H);
    enc_hc_post(m, c);
}

// MoE of layer l (R:569-649): shared expert, router (a stop on score-routed layers), routed experts.
static void enc_ffn(Model *m, sx_cmd *c, size_t l) {
    const DsLayer *L = &m->L[l];
    enc_hc_pre(m, c, L->hcf_fn, L->hcf_base, L->hcf_scale, L->ffn_norm);
    // shared expert
    enc_aq(m, c, m->X, H);
    enc_lin(m, c, &L->sw1, m->g1);
    enc_lin(m, c, &L->sw3, m->u3);
    {
        uint32_t nf = FF;
        DISP(c, m->p.sact, FF / 256, 1, 1, 256, B(m->g1), B(m->u3), B(m->sh), gpu_u32_arg(&nf));
    }
    enc_aq(m, c, m->sh, FF);
    enc_lin(m, c, &L->sw2, m->ys);
    // router
    const sx_arg ids = AT(m->rids, l * 32), rw = AT(m->rw, l * 32), slots = AT(m->slots, l * 32);
    if (l < NHASH) {
        // the ids (sorted) and their slots were written before commit: only the weights here
        DISP(c, m->p.r_hash, 1, 1, 1, 32 * TOPK, L->gate_w, B(m->X), AT(m->hids, l * 32), B(m->zero), ids, rw);
    } else {
        DISP(c, m->p.r_logits, NE / 8, 1, 1, 256, L->gate_w, B(m->X), B(m->rlog));
        DISP(c, m->p.r_select, 1, 1, 1, 256, B(m->rlog), L->gate_bias, ids, rw);
        if (!xs_stop(&m->xs, c, m->stops, &m->nstops, MAX_STOPS, 0, (uint32_t)l, TOPK)) m->failed = true;
    }
    // routed experts, pairs in ascending expert id
    {
        uint32_t K = H;
        DISP(c, m->p.m_aq, H / 128, 1, 1, 32, B(m->X), B(m->xq), gpu_u32_arg(&K));
    }
    const sx_arg W = B(xc_region(m->xs.xc, 0)), S = B(xc_region(m->xs.xc, 1));
    DISP(c, m->p.m_gu, FF / 128, TOPK, 1, 256, W, S, B(m->xq), slots, rw, B(m->hq), B(m->dummy), BYTES(m->moe));
    DISP(c, m->p.m_dn, H / 16, 1, 1, 256, W, S, B(m->hq), slots, B(m->yr), BYTES(m->moe));
    {
        uint32_t nh = H;
        DISP(c, m->p.madd, H / 256, 1, 1, 256, B(m->yr), B(m->ys), B(m->f), gpu_u32_arg(&nh));
    }
    enc_hc_post(m, c);
}

// pairs (id, weight) of one row, ascending id (insertion sort of 6)
static void sort_pairs(uint32_t *e, float *w) {
    for (int i = 1; i < (int)TOPK; i++)
        for (int j = i; j > 0 && e[j - 1] > e[j]; j--) {
            uint32_t te = e[j];
            e[j] = e[j - 1];
            e[j - 1] = te;
            float tw = w[j];
            w[j] = w[j - 1];
            w[j - 1] = tw;
        }
}

// The service of a score-routed layer's stop: the router's (id, weight) pairs sorted by id,
// the ids mapped to cache slots (reading the missing experts).
static Err ds_xmap(void *ctx, XCache *xc, const XSvc *s) {
    Model *m = ctx;
    const uint32_t l = s->layer;
    uint32_t *ids = (uint32_t *)sx_buffer_contents(m->rids) + l * 8;
    float *w = gpu_floats(m->rw) + l * 8;
    uint32_t *sl = (uint32_t *)sx_buffer_contents(m->slots) + l * 8;
    uint32_t e[TOPK];
    float ww[TOPK];
    memcpy(e, ids, sizeof e);
    memcpy(ww, w, sizeof ww);
    sort_pairs(e, ww);
    Err err = OK;
    for (size_t k = 0; k < TOPK; k++)
        if (e[k] >= NE) err = "BadExpertId";
    if (!err) err = xc_map(xc, l, e, TOPK, sl);
    if (err) memset(sl, 0, TOPK * sizeof *sl);
    memcpy(w, ww, sizeof ww);
    return err;
}

// The hash layers' experts from the token (tid2eid), sorted and mapped before the pass.
static Err map_hash(Model *m, uint32_t token) {
    for (size_t l = 0; l < NHASH; l++) {
        int64_t row[TOPK];
        memcpy(row, m->L[l].tid2eid + (size_t)token * TOPK * 8, sizeof row);
        uint32_t e[TOPK];
        float w[TOPK] = {0};
        for (size_t k = 0; k < TOPK; k++) {
            if (row[k] < 0 || row[k] >= (int64_t)NE) return "BadExpertId";
            e[k] = (uint32_t)row[k];
        }
        sort_pairs(e, w);
        memcpy((uint32_t *)sx_buffer_contents(m->hids) + l * 8, e, sizeof e);
        TRY(xc_map(m->xs.xc, (uint32_t)l, e, TOPK, (uint32_t *)sx_buffer_contents(m->slots) + l * 8));
    }
    return OK;
}

// One token at s->pos, one command buffer. trace: the streams after each layer l go to row
// (l * tstride + trow) of it ([.][16384] floats).
static Err step(Model *m, Session *s, uint32_t token, sx_buffer *trace, size_t trow, size_t tstride) {
    DsSession *d = dss(s);
    const size_t pos = s->pos;
    if (pos >= s->cap) return "ContextFull";
    if (token >= VOCAB) return "BadToken";
    if (m->xs.fault) return "StreamingFault";
    // host inputs: the embedding in the 4 streams, the list lengths, the hash layers' experts
    {
        float *h = gpu_floats(m->st);
        const uint16_t *row = (const uint16_t *)(m->embed + (size_t)token * H * 2);
        for (size_t i = 0; i < H; i++) {
            uint32_t u = (uint32_t)row[i] << 16;
            float v;
            memcpy(&v, &u, 4);
            for (size_t k = 0; k < HC; k++) h[k * H + i] = v;
        }
        uint32_t *cn = sx_buffer_contents(m->cnt);
        cn[0] = 0;
        cn[1] = (uint32_t)MIN((size_t)ITOPK, (pos + 1) / 4);
        cn[2] = (uint32_t)((pos + 1) / 128);
    }
    TRY(map_hash(m, token));
    sx_cmd *c = sx_cmd_begin(m->g->dev);
    if (!c) return "GpuFailed";
    m->nstops = 0;
    m->failed = false;
    for (size_t l = 0; l < NLAYERS; l++) {
        if (m->force && l > 0 && pos < m->force_n) {
            uint32_t n4 = HCW / 4;
            DISP(c, m->p.copy4, n4 / 256, 1, 1, 256, AT(m->force, (pos * NLAYERS + l - 1) * HCW * 4), B(m->st),
                 gpu_u32_arg(&n4));
        }
        enc_attn(m, d, c, l, (uint32_t)pos);
        enc_ffn(m, c, l);
        if (trace) {
            uint32_t n4 = HCW / 4;
            DISP(c, m->p.copy4, n4 / 256, 1, 1, 256, B(m->st), AT(trace, (l * tstride + trow) * HCW * 4), gpu_u32_arg(&n4));
        }
    }
    // head (R:709-735)
    {
        MixP mp = {HC, 1};
        DISP(c, m->p.hc_mix, HC, SLICES, 1, 128, m->hc_head_fn, B(m->st), B(m->part), BYTES(mp));
        DISP(c, m->p.hc_head_norm, 1, 1, 1, 256, B(m->part), m->hc_head_scale, m->hc_head_base, B(m->st), B(m->X), m->norm_w);
        enc_mv(m, c, &m->head, H, VOCAB, m->X, m->logits, false);
    }
    if (m->failed) {
        sx_cmd_discard(c);
        m->nstops = 0;
        return m->xs.fault ? "StreamingFault" : "TooManyStops";
    }
    sx_cmd_commit(c);
    Err xe = m->nstops ? xs_service(&m->xs, c, m->stops, &m->nstops, ds_xmap, m) : OK;
    char err[512];
    if (sx_cmd_wait(c, err, sizeof err) != 0) {
        dbg("command buffer: %s\n", err);
        return "GpuFailed";
    }
    TRY(xe);
    s->pos = pos + 1;
    return OK;
}

Err model_step(Model *m, Session *s, uint32_t token, float **logits) {
    TRY(step(m, s, token, s->trace ? s->trace_buf : NULL, 0, 1));
    if (s->trace) memcpy(s->trace, gpu_floats(s->trace_buf), s->trace_len * sizeof(float));
    *logits = gpu_floats(m->logits);
    return OK;
}

// Prefill as successive decode steps (the decode path's semantics); trace [layers][n][16384].
Err model_prefill(Model *m, Session *s, const uint32_t *tokens, size_t n, sx_buffer *trace, float **logits) {
    for (size_t t = 0; t < n; t++) TRY(step(m, s, tokens[t], trace, t, n));
    *logits = gpu_floats(m->logits);
    return OK;
}

Err model_prefill_with(Model *m, Session *s, const uint32_t *tokens, size_t n, sx_buffer *trace, const PrefillHook *hook,
                       float **logits) {
    TRY(model_prefill(m, s, tokens, n, trace, logits));
    if (hook && hook->chunk_done) TRY(hook->chunk_done(hook->ctx, s->pos));
    return OK;
}

void model_prof_init(void) {}
void model_prof_reset(void) {}
void model_prof_report(size_t tokens) {
    if (ds_report) xs_report(ds_report, tokens);
}

// ---- sessions -----------------------------------------------------------------------------

// The 32 RoPE frequencies of the window-only layers: theta 10000, no YaRN (R:483-485, R:226).
static void rope_freqs_window(float f[NPAIR]) {
    for (int i = 0; i < NPAIR; i++) f[i] = 1.0f / (float)pow(CONFIG_ROPE_THETA, (double)((float)(2 * i) / 64.0f));
}

// Compressor states as R starts them: kv 0, score -inf.
static void reset_states(DsSession *d) {
    uint8_t *base = sx_buffer_contents(d->cst);
    memset(base, 0, sx_buffer_length(d->cst));
    for (size_t l = 0; l < NLAYERS; l++) {
        const uint32_t r = config_compress_ratios[l];
        if (!r) continue;
        const size_t n = r == 4 ? 8 * 2 * HD : 128 * HD;
        float *sc = (float *)(base + d->scs[l]);
        for (size_t i = 0; i < n; i++) sc[i] = -INFINITY;
        if (r == 4) {
            float *isc = (float *)(base + d->iscs[l]);
            for (size_t i = 0; i < 8 * 2 * ID; i++) isc[i] = -INFINITY;
        }
    }
}

Err session_create_cap(Gpu *g, bool kv_half, size_t cap, size_t max_cap, Session **out) {
    (void)kv_half;
    cap = MAX(cap, max_cap);
    if (cap < 128) cap = 128;
    DsSession *d = xcalloc(1, sizeof *d);
    Session *s = &d->s;
    s->arch = &dsv4_arch;
    s->cap = s->max_cap = cap;
    s->roll = SESSION_NO_ROLL;
    size_t cc = 0, ic = 0, st = 0;
    for (size_t l = 0; l < NLAYERS; l++) {
        const uint32_t r = config_compress_ratios[l];
        if (!r) continue;
        d->cc[l] = cc;
        cc += ALIGN_UP((cap / r + 1) * HD * 2, 16);
        const size_t sn = (r == 4 ? 8 * 2 * HD : 128 * HD) * 4;
        d->kvs[l] = st;
        d->scs[l] = st + sn;
        st += 2 * sn;
        if (r == 4) {
            d->ic[l] = ic;
            ic += ALIGN_UP((cap / 4 + 1) * ID * 2, 16);
            d->ikvs[l] = st;
            d->iscs[l] = st + 8 * 2 * ID * 4;
            st += 2 * 8 * 2 * ID * 4;
        }
    }
    d->ls = (uint32_t)ALIGN_UP(WIN + MAX((size_t)ITOPK, cap / 128 + 1), 64);
    TRY(gpu_new_buffer(g, NLAYERS * WIN * HD * 2, &d->ring));
    TRY(gpu_new_buffer(g, cc, &d->ccache));
    TRY(gpu_new_buffer(g, ic, &d->icache));
    TRY(gpu_new_buffer(g, st, &d->cst));
    TRY(gpu_new_buffer(g, cap * NPAIR * 8, &d->rope_w));
    TRY(gpu_new_buffer(g, cap * NPAIR * 8, &d->rope_c));
    TRY(gpu_new_buffer(g, (cap / 4 + 32) * 4, &d->iscore));
    TRY(gpu_new_buffer(g, (cap / 128 + 1) * 4, &d->ident));
    TRY(gpu_new_buffer(g, (size_t)NH * d->ls * 4, &d->scores));
    TRY(gpu_new_buffer(g, NLAYERS * HCW * 4, &s->trace_buf));
    float fw[NPAIR], fc[NPAIR];
    rope_freqs_window(fw);
    dsv4_cmp_rope_freqs(fc);
    float *tw = gpu_floats(d->rope_w), *tc = gpu_floats(d->rope_c);
    for (size_t p = 0; p < cap; p++) {
        dsv4_cmp_rope_cs(fw, (uint32_t)p, tw + p * NPAIR * 2);
        dsv4_cmp_rope_cs(fc, (uint32_t)p, tc + p * NPAIR * 2);
    }
    uint32_t *id = sx_buffer_contents(d->ident);
    for (size_t i = 0; i <= cap / 128; i++) id[i] = (uint32_t)i;
    reset_states(d);
    *out = s;
    return OK;
}

Err session_create_kv(Gpu *g, bool kv_half, Session **out) {
    size_t cap;
    TRY(session_ctx_parse(getenv("SAETTA_CTX"), &cap));
    return session_create_cap(g, kv_half, cap, cap, out);
}
Err session_create(Gpu *g, Session **out) { return session_create_kv(g, false, out); }

void session_reset(Session *s) {
    s->pos = 0;
    s->roll = SESSION_NO_ROLL;
    reset_states(dss(s));
}

// ---- not implemented yet -------------------------------------------------------------------

#define NI "NotImplemented"
bool model_has_mtp(const Model *m) { (void)m; return false; }
bool model_fast_prefill(const Model *m) { (void)m; return false; }
bool model_fast_prefill_available(const Model *m) { (void)m; return false; }
void model_set_fast_prefill(Model *m, bool on) { (void)m, (void)on; }
Err model_verify_chunk(Model *m, Session *s, const uint32_t *toks, size_t n, float **logits) {
    (void)m, (void)s, (void)toks, (void)n, (void)logits;
    return NI;
}
Err model_spec_greedy(Model *m, Session *s, uint32_t x, const uint32_t *drafts, size_t nd, SpecResult *out) {
    (void)m, (void)s, (void)x, (void)drafts, (void)nd, (void)out;
    return NI;
}
Err model_spec_verify(Model *m, Session *s, uint32_t x, const uint32_t *drafts, size_t nd, SpecPick pick, void *ctx,
                      SpecResult *out) {
    (void)m, (void)s, (void)x, (void)drafts, (void)nd, (void)pick, (void)ctx, (void)out;
    return NI;
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
size_t session_kv_elem(const Session *s) { (void)s; return 2; }
size_t session_kv_bytes(const Session *s, size_t cap) { (void)s, (void)cap; return 0; }
Err session_grow(Session *s, Gpu *g, size_t cap) { (void)g; return cap <= s->cap ? OK : NI; }
Err session_save_checkpoint(Session *s, const uint32_t *toks, size_t n) { (void)s, (void)toks, (void)n; return NI; }
Err session_save_checkpoint_at(Session *s, size_t slot, const uint32_t *toks, size_t n) {
    (void)s, (void)slot, (void)toks, (void)n;
    return NI;
}
size_t session_reusable(Session *s, const uint32_t *prompt, size_t n) { (void)s, (void)prompt, (void)n; return 0; }
size_t session_conv_bytes(const Session *s, size_t cap) { (void)s, (void)cap; return 0; }
Err session_new_conv(const Session *s, Gpu *g, size_t cap, Conv *out) { (void)s, (void)g, (void)cap, (void)out; return NI; }
Err session_conv_shrink(const Session *s, Gpu *g, Conv *cv, size_t cap) { (void)s, (void)g, (void)cv, (void)cap; return NI; }
void session_swap_conv(Session *s, Conv *cv) { (void)s, (void)cv; }
bool session_roll_to(Session *s, size_t p, const uint32_t *toks) { (void)s, (void)p, (void)toks; return false; }
Err session_save_point(Session *s, Gpu *g, const uint32_t *toks, size_t n) { (void)s, (void)g, (void)toks, (void)n; return NI; }
void session_start_at_own_point(Session *s, size_t n) { (void)s, (void)n; }
Err session_start_from(Session *s, Gpu *g, const Conv *src, size_t n) { (void)s, (void)g, (void)src, (void)n; return NI; }
void session_restore(Session *s) { (void)s; }
void session_push_history(Session *s, uint32_t token) { (void)s, (void)token; }

const ModelArch dsv4_arch = MODEL_ARCH_TABLE({
    .model_type = CONFIG_MODEL_TYPE,
    .id = "deepseek-v4-flash",
    .name = "DeepSeek-V4-Flash",
    .layers = NLAYERS,
    .width = HC * H,
    .vocab = VOCAB,
    .hidden = H,
    .eos = dsv4_eos,
    .n_eos = 1,
    .no_spec = true,
    .no_serve = true,
});
