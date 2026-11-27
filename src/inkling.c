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
// One command buffer per chunk (decode: T = 1). Decode, verification chunks and, without tensor
// ops (or with SAETTA_PREFILL_F32 / SAETTA_NO_ACCEL), prompt chunks are f32: the dense linears on
// the run-time affine kernels (src/qgen.h), the rest on src/kernels/inkling_*.metal. Where the
// tensor ops run, prompt chunks take half activations with f32 sums (src/kernels/inkling_mpp.metal,
// mmh_dense): the experts, dense projections, message gates and, on an f16 cache, attention.
#include "model.h"

#include "config_inkling.h"
#define ARCH_PFX inkling_
#include "model_rename.h"

#include <math.h>
#include <stdlib.h>
#include <unistd.h>

#include "download.h"

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
// Element offsets in the kernels are 32-bit: every position of the longest context times one row.
static_assert((uint64_t)CONFIG_MODEL_MAX_LENGTH * KVD <= UINT32_MAX && CONFIG_MODEL_MAX_LENGTH == SESSION_CTX_INKLING_MAX);
static_assert(CONFIG_N_SHARED_EXPERTS == 0 && CONFIG_USE_GATE_BIAS && NE % 32 == 0 && RANK % 8 == 0);

// Prompt chunk: tokens per command buffer. Each chunk reads every routed expert's weights about
// once per visit (29 GB a chunk), so chunks are long: 4K prompt 3.79 s at 1024, 3.60 s at 2048
// and at 4096 (scratch about 1 GB at 2048).
#define PCH 2048
// Attention splits of decode and verification chunks (T <= KMAX): up to NSPLIT_DEC per query;
// prompt chunks use the tiled kernel, no splits.
#define NSPLIT_DEC 256

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
        *kv_store, *kv_store_h, *attn_part, *attn_part_h, *attn_flash, *attn_flash_h, *attn_merge, *gu_nv, *dn_nv, *gu_mx, *dn_mx, *plan, *gu_nv_g, *gu_mx_g, *dn_nv_g,
        *dn_mx_g, *router_lg;
    // the DSpark drafter (inkling_dspark.metal)
    sx_pipeline *ds_mv, *ds_mix, *ds_rope, *ds_attn, *ds_attn_h, *ds_addn, *ds_silu, *ds_step, *ds_fold, *ds_argmax, *ds_fold_t,
        *ds_ring;
    sx_pipeline *mv, *mm; // qgen: qmatvec_gen, qmm_gen
    // prompt chunks on the tensor ops (inkling_mpp.metal; NULL where they do not run)
    sx_pipeline *im_plan, *im_gather, *im_gu_nv, *im_gu_mx, *im_dn_nv, *im_dn_mx;
    sx_pipeline *im_half, *im_dq, *im_mm;
    sx_pipeline *im_attq, *im_attn, *im_mcat, *im_mcomb; // to_half, dq_f16_gen, mmh_dense (qmm_mpp.metal, qgen_mpp.metal)
} Pipes;

// Scratch regions, floats per token (each region holds PCH rows).
enum { R_H, R_HN, R_BS, R_P0, R_MS, R_MSG, R_MIX, R_G, R_XN, R_Q, R_K, R_V, R_GT, R_SK, R_RS, R_KD, R_VD, R_KN, R_VN, R_ATT, R_A,
       R_R, R_HM, R_XM, R_D, R_IDX, R_W, R_HACT, R_YS, R_YSUM, R_PAIRS, R_TILES, R_NT, R_LG, R_TAP, R_N };
static const size_t region_w[R_N] = {
    [R_H] = H, [R_HN] = H, [R_BS] = MAXBS * H, [R_P0] = H, [R_MS] = 2 * H, [R_MSG] = 4 * H, [R_MIX] = 3 * H, [R_G] = 3 * H,
    [R_XN] = 3 * H, [R_Q] = QD, [R_K] = KVD, [R_V] = KVD, [R_GT] = QD, [R_SK] = NH, [R_RS] = NH * DREL, [R_KD] = NKV * 4,
    [R_VD] = NKV * 4, [R_KN] = KVD, [R_VN] = KVD, [R_ATT] = QD, [R_A] = H, [R_R] = H, [R_HM] = H, [R_XM] = H, [R_D] = RANK,
    [R_IDX] = TOPK, [R_W] = TOPK, [R_HACT] = TOPK * FF, [R_YS] = TOPK * H, [R_YSUM] = H,
    [R_PAIRS] = TOPK, [R_TILES] = 4, [R_NT] = 1, [R_LG] = NE,
    [R_TAP] = 4 * H}; // R_TAP: 4 planes of [PCH][H], the drafter's trunk features
// the expert tiles of a chunk (ink_moe_plan): at most T * TOPK / 32 + NE, 4 words each
static_assert(PCH * TOPK / 32 + NE <= PCH);
// the tensor-op experts' sorted rows (each expert's padded to 32) and tiles of at most 128 rows
#define MPP_ROWS (PCH * TOPK + 31 * NE)
#define MPP_TILES (MPP_ROWS / 128 + NE)
#define MSGW_HALVES ((size_t)4 * H * RANK) // one message gate's f16 weights: D, then F and G
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

// ---- the DSpark drafter (thinkingmachines' mtp.safetensors, BF16) ---------------------------
#define DS_L CONFIG_MTP_NUM_HIDDEN_LAYERS   // 3
#define DS_FF CONFIG_MTP_INTERMEDIATE_SIZE  // 9216
#define DS_BLK CONFIG_MTP_BLOCK_SIZE        // 8: the anchor and 7 drafts
#define DS_R CONFIG_MTP_MARKOV_RANK         // 256
static_assert(CONFIG_MTP_HIDDEN_SIZE == H && CONFIG_MTP_NUM_ATTENTION_HEADS == NH && CONFIG_MTP_NUM_KEY_VALUE_HEADS == NKV &&
              CONFIG_MTP_HEAD_DIM == HD && CONFIG_MTP_SLIDING_WINDOW_SIZE == WIN && DS_BLK == KMAX && DS_R == 256 &&
              CONFIG_MTP_N_TARGET_LAYERS == 4 && DS_L == 3);
// the kernels' shapes (inkling_dspark.metal): a block of 8 rows (ds_attn_part's p0 + 8), the
// Markov state of 256 in one threadgroup, rnn_proj rows of [state, w1, hidden, f_rnn]; the
// drafter's norms use the trunk's epsilon (enc_norm)
static_assert(DS_BLK == 8 && DS_R == 256 && 2 * DS_R + 2 * CONFIG_MTP_HIDDEN_SIZE == 5120 && CONFIG_MTP_RMS_NORM_EPS == CONFIG_RMS_NORM_EPS);
// the drafter's layers: local ones in rings, global ones in full planes (ds_planes)
static size_t ds_n_global(void) {
    size_t n = 0;
    for (size_t l = 0; l < DS_L; l++) n += config_mtp_layer_kinds[l] == LAYER_KIND_GLOBAL;
    return n;
}
// Activations (floats, offsets into Ds.a).
enum { DA_RES, DA_XN, DA_Q, DA_KB, DA_VB, DA_ATT, DA_O, DA_G, DA_U, DA_HM, DA_CTX, DA_KC, DA_VC, DA_INP, DA_MCTX, DA_STATE,
       DA_LAT, DA_PREV, DA_DRAFT, DA_PART, DA_FRNN, DA_FTAP, DA_BASE, DA_GATE1, DA_SINK, DA_N };
typedef struct {
    bool on;
    sx_buffer *w; // BF16 matrices, then f32 vectors, 16-byte aligned
    size_t q[DS_L], k[DS_L], v[DS_L], kc[DS_L], vc[DS_L], o[DS_L], gt[DS_L], up[DS_L], dn[DS_L], w1, rnn; // byte offsets (bf16)
    size_t qn[DS_L], kn[DS_L], kcn[DS_L], iln[DS_L], pln[DS_L], dfl[DS_L], mdfl, norm, mask, rnnb, alpha; // byte offsets (f32)
    sx_buffer *fold, *foldb; // lm_head . hidden_out: f16 [VOCAB][DS_R]; lm_head . hidden_out bias [VOCAB]
    sx_buffer *a;            // activations
    sx_buffer *cs;           // RoPE (cos, sin) rows [PCH + DS_BLK][HD / 2][2], host-written in double
    size_t off[DA_N + 1];
    uint32_t tap_t, tap_pos0; // the rows R_TAP holds (the last chunk's): SAETTA_DS_DUMP
    bool dumped;
} Ds;

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
    // Prompt chunks (T > KMAX) on the tensor ops with half activations (accel: their kernels
    // loaded; half_prefill: in use, the default where they load; SAETTA_PREFILL_F32 and verify's
    // exact checks turn it off): the experts' rows sorted by expert (msrt, mtiles, mmeta: the
    // plan), their half inputs xs_h [rows][H] and activations hs_h [rows][FF].
    bool accel, half_prefill;
    sx_buffer *xs_h, *hs_h, *msrt, *mtiles, *mmeta;
    // the dense projections: their input in half (xh [PCH][QD]), the weights unpacked to f16 (wh)
    sx_buffer *xh, *wh;
    sx_arg xh_of; // the input xh holds (none: {0}); reset at each visit and around o_proj
    // the prompt attention: queries in half per head (qh [NH][PCH][HD]), the chunk's keys and
    // values in half (kvh [2][PCH][KVD]), the relative bias table per layer (projt [NL][EXT + 1][16]
    // f32: proj transposed, row EXT zeros)
    sx_buffer *qh, *kvh, *projt;
    // the message gates: f16 copies of each gate's down D [RANK][2H] then [F; G] [2H][RANK]
    // (msgw, per loop layer and gate), the normalized inputs xc [PCH][2H] half, d in half
    // (dh [PCH][RANK]), [f, g] (fg [PCH][2H] f32)
    sx_buffer *msgw, *xc, *dh, *fg;
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
    Ds ds;
    // A verification chunk (model_spec_verify) commits nothing while it runs: each visit's rows of
    // the conv inputs (raw K, raw V, attention out, MLP out) and of the keys and values stay in spk;
    // the commit then moves the accepted rows into the conv states and caches, after saving the
    // conv states and the ring rows it replaces in spb (session_roll_to puts them back).
    bool spec_chunk;
    sx_buffer *spk, *spb;
};
#define SPK_W (4 * KVD + 2 * H) // floats per visit and row

static const Model *ds_model;
static Err ds_load(Arena *arena, Model *m, Gpu *g);
static void ds_rope_rows(Model *m, size_t row0, size_t pos0, size_t T);
struct Enc_;
static void enc_ds_ctx(Model *m, Session *d, struct Enc_ *c, uint32_t T, uint32_t pos0);

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
// With the DSpark drafter loaded the sessions hold its context keys and values too: its global
// layer as one more full plane pair (index n_full_vis), its two local layers as two more rings
// (n_swa_vis, n_swa_vis + 1); set by model_load before any session exists.
static size_t n_kvf_planes, n_ring_planes;
#define RING_BYTES(elem) ((size_t)n_ring_planes * 2 * WIN * KVD * (elem))
#define CONV_BYTES ((size_t)NVIS * CONV_VIS * 4)
static size_t kvf_bytes(size_t cap, size_t elem) { return n_kvf_planes * 2 * cap * KVD * elem; }

static inline sx_arg B(sx_buffer *b) { return (sx_arg){b, 0, NULL, 0}; }
static inline sx_arg AT(sx_buffer *b, size_t off) { return (sx_arg){b, off, NULL, 0}; }
static inline sx_arg OFF(sx_arg a, size_t floats) { return (sx_arg){a.buffer, a.offset + floats * 4, NULL, 0}; }
#define BYTES(v) gpu_bytes_arg(&(v), sizeof(v))
// The command buffer being encoded. SAETTA_PROFILE=1: every dispatch in its own command buffer,
// GPU time summed per kernel (model_prof_report).
typedef struct Enc_ {
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
    , '\n',
#embed "kernels/inkling_dspark.metal"
    , 0};
static const char ink_mpp_src[] = {
#embed "kernels/inkling_mpp.metal"
    , 0};
static const char qmm_mpp_src[] = {
#embed "kernels/qmm_mpp.metal"
    , 0};

// The tensor-op prompt kernels, all or none (gpu_pipeline refuses them without neural
// accelerators or with SAETTA_NO_ACCEL=1): true when they loaded.
static bool load_mpp(Gpu *g, Pipes *pp) {
    static const struct {
        size_t off;
        const char *name;
    } kn[] = {{offsetof(Pipes, im_plan), "ink_mpp_plan"},   {offsetof(Pipes, im_gather), "ink_mpp_gather"},
              {offsetof(Pipes, im_gu_nv), "ink_mpp_gu_nv"}, {offsetof(Pipes, im_gu_mx), "ink_mpp_gu_mx"},
              {offsetof(Pipes, im_dn_nv), "ink_mpp_dn_nv"}, {offsetof(Pipes, im_dn_mx), "ink_mpp_dn_mx"},
              {offsetof(Pipes, im_attq), "ink_mpp_attn_q"}, {offsetof(Pipes, im_attn), "ink_mpp_attn"},
              {offsetof(Pipes, im_mcat), "ink_mpp_msg_cat"}, {offsetof(Pipes, im_mcomb), "ink_mpp_msg_comb"}};
    bool ok = true;
    for (size_t i = 0; i < COUNT(kn) && ok; i++) {
        ok = gpu_pipeline(g, ink_mpp_src, kn[i].name, (sx_pipeline **)((char *)pp + kn[i].off)) == OK;
        kernel_names[kn[i].off / sizeof(sx_pipeline *)] = kn[i].name;
    }
    // ink_mpp_plan runs as one threadgroup of 1024 threads
    if (ok && gpu_require_threads(pp->im_plan, 1024) != OK) {
        dbg("model: ink_mpp_plan cannot run 1024 threads a threadgroup\n");
        ok = false;
    }
    if (ok) {
        ok = gpu_pipeline(g, qmm_mpp_src, "to_half", &pp->im_half) == OK && gpu_pipeline(g, qmm_mpp_src, "mmh_dense", &pp->im_mm) == OK &&
             gpu_pipeline(g, qgen_mpp_src, "dq_f16_gen", &pp->im_dq) == OK;
        kernel_names[offsetof(Pipes, im_half) / sizeof(sx_pipeline *)] = "to_half";
        kernel_names[offsetof(Pipes, im_mm) / sizeof(sx_pipeline *)] = "mmh_dense";
        kernel_names[offsetof(Pipes, im_dq) / sizeof(sx_pipeline *)] = "dq_f16_gen";
    }
    if (!ok)
        for (size_t i = 0; i < COUNT(kn); i++) {
            sx_pipeline **q = (sx_pipeline **)((char *)pp + kn[i].off);
            if (*q) sx_pipeline_release(*q);
            *q = NULL;
        }
    if (!ok) {
        sx_pipeline **q[3] = {&pp->im_half, &pp->im_mm, &pp->im_dq};
        for (size_t i = 0; i < 3; i++) {
            if (*q[i]) sx_pipeline_release(*q[i]);
            *q[i] = NULL;
        }
    }
    return ok;
}

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
        K(ds_mv, "ds_mv"), K(ds_mix, "ds_mix_norm"), K(ds_rope, "ds_norm_rope"), K(ds_attn, "ds_attn_part"), K(ds_attn_h, "ds_attn_part_h"),
        K(ds_addn, "ds_add_norm"), K(ds_silu, "ds_silu_mul"), K(ds_step, "ds_markov_step"), K(ds_fold, "ds_fold_argmax"),
        K(ds_argmax, "ds_argmax_final"), K(ds_fold_t, "ds_fold_t"), K(ds_ring, "ds_ring_copy"),
        K(moe_sum, "ink_moe_sum"), K(softcap, "ink_softcap"), K(copy, "ink_copy"), K(kv_store, "ink_kv_store"),
        K(attn_part, "ink_attn_part"), K(attn_part_h, "ink_attn_part_h"), K(attn_flash, "ink_attn_flash"),
        K(attn_flash_h, "ink_attn_flash_mma"), K(kv_store_h, "ink_kv_store_h"), K(attn_merge, "ink_attn_merge"), K(gu_nv, "ink_moe_gu_nv"), K(dn_nv, "ink_moe_dn_nv"),
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
    {
        // qmatvec_gen over T rows (grid y): decode and verification chunks share this kernel, so a
        // token's projections are the same bits whichever of the two reads it
        static const char tail[] =
            "\nkernel void ink_qmv_rows(device const uchar *w [[buffer(0)]], device const uchar *scales [[buffer(1)]],\n"
            "    device const uchar *biases [[buffer(2)]], device const float *x [[buffer(3)]], device float *y [[buffer(4)]],\n"
            "    constant uint &in [[buffer(5)]], constant uint &out [[buffer(6)]], constant uint *shift [[buffer(7)]],\n"
            "    constant uint2 &q [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {\n"
            "    if (tg.x < out) qg_row(q.x, q.y, w + shift[0], scales + shift[1], biases + shift[2], x + tg.y * in, y + tg.y * out, in, tg.x, lane);\n"
            "}\n";
        char *src = arena_printf(arena, "%s%s", qgen_src, tail);
        TRY(gpu_pipeline(g, src, "ink_qmv_rows", &pp->mv));
    }
    TRY(gpu_pipeline(g, qgen_src, "qmm_gen", &pp->mm));
    m->accel = load_mpp(g, pp);
    m->half_prefill = m->accel && getenv("SAETTA_PREFILL_F32") == NULL;
    if (m->accel) {
        TRY(gpu_new_buffer(g, (size_t)MPP_ROWS * H * 2, &m->xs_h));
        TRY(gpu_new_buffer(g, (size_t)MPP_ROWS * FF * 2, &m->hs_h));
        TRY(gpu_new_buffer(g, (size_t)MPP_ROWS * 4, &m->msrt));
        TRY(gpu_new_buffer(g, (size_t)MPP_TILES * 16, &m->mtiles));
        TRY(gpu_new_buffer(g, 16, &m->mmeta));
        TRY(gpu_new_buffer(g, (size_t)PCH * QD * 2, &m->xh));
        TRY(gpu_new_buffer(g, (size_t)QD * QD * 2, &m->wh));
        TRY(gpu_new_buffer(g, (size_t)NH * PCH * HD * 2, &m->qh));
        TRY(gpu_new_buffer(g, (size_t)2 * PCH * KVD * 2, &m->kvh));
        TRY(gpu_new_buffer(g, (size_t)PCH * 2 * H * 2, &m->xc));
        TRY(gpu_new_buffer(g, (size_t)PCH * RANK * 2, &m->dh));
        TRY(gpu_new_buffer(g, (size_t)PCH * 2 * H * 4, &m->fg));
    } else
        dbg("model: accelerator kernels unavailable, Inkling prompt chunks on f32 tiles\n");
    if (m->half_prefill) dbg("model: Inkling prompt chunks on the tensor ops (half activations, f32 sums)\n");

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
    if (m->accel) {
        static_assert(DREL <= 16);
        TRY(gpu_new_buffer(g, (size_t)NL * (EXT + 1) * 16 * 4, &m->projt));
        float *pt = gpu_floats(m->projt);
        memset(pt, 0, (size_t)NL * (EXT + 1) * 16 * 4);
        for (size_t li = 0; li < NL; li++) {
            const float *pj = (const float *)((const uint8_t *)sx_buffer_contents(m->params) + m->L[li].proj.offset);
            for (size_t i = 0; i < DREL; i++)
                for (size_t dd = 0; dd < EXT; dd++) pt[(li * (EXT + 1) + dd) * 16 + i] = pj[i * EXT + dd];
        }
    }
    if (m->accel) {
        TRY(gpu_new_buffer(g, (size_t)NLOOP * 4 * MSGW_HALVES * 2, &m->msgw));
        _Float16 *w16 = sx_buffer_contents(m->msgw);
        const uint8_t *pb = sx_buffer_contents(m->params);
        for (size_t li = LS; li < LE; li++)
            for (size_t k = 0; k < 4; k++) {
                _Float16 *o = w16 + ((li - LS) * 4 + k) * MSGW_HALVES;
                const uint16_t *D = (const uint16_t *)(pb + m->L[li].md[k].offset);
                const float *F = (const float *)(pb + m->L[li].mf[k].offset), *G = (const float *)(pb + m->L[li].mg[k].offset);
                for (size_t i = 0; i < (size_t)RANK * 2 * H; i++) o[i] = (_Float16)ops_bf16_to_f32(D[i]);
                o += (size_t)RANK * 2 * H;
                for (size_t i = 0; i < (size_t)H * RANK; i++) o[i] = (_Float16)F[i], o[H * RANK + i] = (_Float16)G[i];
            }
    }
    TRY(gpu_new_buffer(g, region_off[R_N] * 4, &m->scratch));
    TRY(gpu_new_buffer(g, (size_t)KMAX * NSPLIT_DEC * NH * (HD + 2) * 4, &m->part));
    TRY(gpu_new_buffer(g, (size_t)PCH * H * 4, &m->stage));
    TRY(gpu_new_buffer(g, (size_t)KMAX * VOCAB * 4, &m->logits));
    // the DSpark drafter, when downloaded: its planes in every session, the verification buffers
    TRY(ds_load(arena, m, g));
    n_kvf_planes = n_full_vis + (m->ds.on ? ds_n_global() : 0);
    n_ring_planes = n_swa_vis + (m->ds.on ? DS_L - ds_n_global() : 0);
    TRY(gpu_new_buffer(g, (size_t)NVIS * KMAX * SPK_W * 4, &m->spk));
    TRY(gpu_new_buffer(g, (size_t)n_swa_vis * 2 * KMAX * KVD * 4 + CONV_BYTES, &m->spb));
    ds_model = m;
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
        if (!e) e = gpu_new_buffer(g, 32, &m->rstats);
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
typedef struct { uint32_t NH_, NKV_, HD_, DREL_, EXT_, pos0, win, ring, nsplit, span, T; float scale; } AttP;
typedef struct { uint32_t h, i, k, sh_w, sh_s; uint64_t wsE, ssE; float gcap, ucap; } MoeP;
typedef struct { uint32_t e, k, t; } PlanP;
static_assert(sizeof(MoeP) == 48);

// y = W x over T rows: the run-time affine kernels (decode and verification chunks, T <= KMAX:
// one simdgroup per output row and token, the same bits for a token in either; prompt chunks: f32
// tiles). out rows of W (lm_head: the unpadded vocabulary).
// The input of the half matmuls: x [T][in] (f32) to half in Model.xh, once for the projections
// that share it (Model.xh_of: the input it holds).
static void enc_xh(Model *m, Enc *c, sx_arg x, uint32_t n) {
    if (m->xh_of.buffer == x.buffer && m->xh_of.offset == x.offset) return;
    DISP(c, m->p.im_half, (n + 255) / 256, 1, 1, 256, x, B(m->xh), BYTES(n));
    m->xh_of = x;
}
static void enc_lin(Model *m, Enc *c, const Linear *l, sx_arg x, sx_arg y, uint32_t T, uint32_t out) {
    uint32_t sh[3], q[2] = {l->bits, l->group_size}, in = (uint32_t)l->in_features;
    sx_arg w = gpu_tensor_arg(m->g, &l->weight, &sh[0]), s = gpu_tensor_arg(m->g, &l->scales, &sh[1]),
           b = gpu_tensor_arg(m->g, &l->biases, &sh[2]);
    if (T > KMAX && m->half_prefill && out == l->out_features && (size_t)in * out <= (size_t)QD * QD) {
        // a prompt chunk on the tensor ops (after qwental / Strata, src/kernels/qmm_mpp.metal):
        // the weights unpacked to f16, then the f16 GEMM with f32 sums; the input in half
        enc_xh(m, c, x, T * in);
        uint32_t d2[2] = {in, out}, dims[3] = {in, out, T};
        DISP(c, m->p.im_dq, ((size_t)out * in / 8 + 255) / 256, 1, 1, 256, w, s, b, B(m->wh), gpu_bytes_arg(d2, 8), gpu_bytes_arg(sh, 12),
             BYTES(q));
        DISP(c, m->p.im_mm, (out + 31) / 32, (T + 63) / 64, 1, 32, B(m->xh), B(m->wh), y, gpu_bytes_arg(dims, 12));
        return;
    }
    if (T <= KMAX) {
        DISP(c, m->p.mv, out, T, 1, 32, w, s, b, x, y, BYTES(in), BYTES(out), gpu_bytes_arg(sh, 12), BYTES(q));
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
    if (T > KMAX && m->half_prefill) {
        // a prompt chunk on the tensor ops: the two GEMMs in half with f32 sums
        const size_t li = (size_t)(L - m->L);
        const sx_arg w = AT(m->msgw, ((li - LS) * 4 + (size_t)k) * MSGW_HALVES * 2), wfg = OFF(w, (size_t)RANK * H); // 2 RANK H halves
        uint32_t d1[3] = {2 * H, RANK, T}, d2[3] = {RANK, 2 * H, T}, nd = T * RANK;
        DISP(c, m->p.im_mcat, T, 1, 1, 256, x, msg, B(m->xc), BYTES(p));
        DISP(c, m->p.im_mm, RANK / 32, (T + 63) / 64, 1, 32, B(m->xc), w, reg(m, R_D), gpu_bytes_arg(d1, 12));
        DISP(c, m->p.im_half, (nd + 255) / 256, 1, 1, 256, reg(m, R_D), B(m->dh), BYTES(nd));
        DISP(c, m->p.im_mm, 2 * H / 32, (T + 63) / 64, 1, 32, B(m->dh), wfg, B(m->fg), gpu_bytes_arg(d2, 12));
        DISP(c, m->p.im_mcomb, H / 256, T, 1, 256, x, msg, B(m->fg), out, BYTES(p));
        return;
    }
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
    m->xh_of = (sx_arg){0};
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
        const bool upd = T == 1 && !m->spec_chunk;
        KvP pk = {KVD, HD, full, 0, upd, den, CONFIG_QK_L2_CLIP_THRESHOLD, EPS};
        KvP pv = {KVD, HD, full, 1, upd, den, CONFIG_QK_L2_CLIP_THRESHOLD, EPS};
        const sx_arg ks = conv, vs = OFF(conv, 3 * KVD);
        DISP(c, m->p.kv_conv, NKV, T, 1, HD, reg(m, R_K), ks, L->kw, reg(m, R_KD), reg(m, R_KN), BYTES(pk));
        DISP(c, m->p.kv_conv, NKV, T, 1, HD, reg(m, R_V), vs, L->vw, reg(m, R_VD), reg(m, R_VN), BYTES(pv));
        if (T > 1 && !m->spec_chunk) {
            enc_conv_state(m, c, reg(m, R_K), ks, KVD, T);
            enc_conv_state(m, c, reg(m, R_V), vs, KVD, T);
        }
    }
    // attention (M:111-142, 611-636)
    {
        sx_arg kc, vc;
        uint32_t ring;
        const size_t elem = d->kv_half ? 2 : 4;
        if (full) {
            const size_t plane = (size_t)d->cap * KVD * elem;
            kc = AT(d->kv, vis_kv[v] * 2 * plane);
            vc = AT(d->kv, (vis_kv[v] * 2 + 1) * plane);
            ring = 0;
        } else {
            const size_t plane = (size_t)WIN * KVD * elem;
            kc = AT(d->ixk, vis_kv[v] * 2 * plane);
            vc = AT(d->ixk, (vis_kv[v] * 2 + 1) * plane);
            ring = WIN;
        }
        // keys of the last query: its whole range
        const uint32_t last = pos0 + T - 1, range = full ? last + 1 : MIN(last + 1, (uint32_t)WIN);
        const float scale = 1.0f / sqrtf((float)HD);
        if (T <= KMAX) {
            // decode and verification chunks: split-K over the range (enough threadgroups at
            // 100K-1M keys: one per KV head and split), then the merge with the sink and gate
            // each query splits its own range as decode would (the kernel derives it); the grid
            // covers the last query's, the longest
            const uint32_t nsplit = MIN((uint32_t)NSPLIT_DEC, MAX(1u, (range + 255) / 256));
            const uint32_t span = (range + nsplit - 1) / nsplit;
            AttP p = {NH, NKV, HD, DREL, EXT, pos0, full ? 0 : WIN, ring, nsplit, span, T, scale};
            DISP(c, d->kv_half ? m->p.attn_part_h : m->p.attn_part, NKV, nsplit, T, 128, reg(m, R_Q), reg(m, R_RS), L->proj, kc, vc,
                 reg(m, R_KN), reg(m, R_VN), B(m->part), BYTES(p));
            DISP(c, m->p.attn_merge, NH, T, 1, HD, B(m->part), reg(m, R_SK), reg(m, R_GT), reg(m, R_ATT), BYTES(p));
        } else if (m->half_prefill && d->kv_half) {
            // prompt chunks on the tensor ops (f16 cache): queries per head and the chunk's keys
            // and values in half, 32 queries a threadgroup
            AttP p = {NH, NKV, HD, DREL, EXT, pos0, full ? 0 : WIN, ring, 1, 0, T, scale};
            const uint32_t rows = PCH, tq = (T + 31) / 32, n = T * KVD;
            DISP(c, m->p.im_attq, tq * 32, NH, 1, HD, reg(m, R_Q), B(m->qh), BYTES(p), BYTES(rows));
            DISP(c, m->p.im_half, (n + 255) / 256, 1, 1, 256, reg(m, R_KN), B(m->kvh), BYTES(n));
            DISP(c, m->p.im_half, (n + 255) / 256, 1, 1, 256, reg(m, R_VN), AT(m->kvh, (size_t)PCH * KVD * 2), BYTES(n));
            DISP(c, m->p.im_attn, tq, NH, 1, 128, B(m->qh), reg(m, R_RS), AT(m->projt, li * (EXT + 1) * 16 * 4), kc, vc, B(m->kvh),
                 AT(m->kvh, (size_t)PCH * KVD * 2), reg(m, R_SK), reg(m, R_GT), reg(m, R_ATT), BYTES(p), BYTES(rows));
        } else {
            // prompt chunks: key tiles shared by blocks of 16 queries
            AttP p = {NH, NKV, HD, DREL, EXT, pos0, full ? 0 : WIN, ring, 1, 0, T, scale};
            // (f16 cache: simdgroup matrices, 6 simdgroups; f32: the exact scalar tiles, 8)
            DISP(c, d->kv_half ? m->p.attn_flash_h : m->p.attn_flash, NKV, (T + 15) / 16, 1, d->kv_half ? 192 : 256, reg(m, R_Q),
                 reg(m, R_RS), L->proj, kc, vc, reg(m, R_KN), reg(m, R_VN), reg(m, R_SK), reg(m, R_GT), reg(m, R_ATT), BYTES(p));
        }
        if (!m->spec_chunk) {
            StP sp = {KVD, pos0, ring, T};
            DISP(c, d->kv_half ? m->p.kv_store_h : m->p.kv_store, KVD / 256, T, 1, 256, reg(m, R_KN), reg(m, R_VN), kc, vc, BYTES(sp));
        }
    }
    m->xh_of = (sx_arg){0};
    enc_lin(m, c, &L->o, reg(m, R_ATT), reg(m, R_A), T, H);
    m->xh_of = (sx_arg){0};
    // post-attention conv and the residual (M:899-905)
    {
        CrP p = {H, boundary, T == 1 && !m->spec_chunk, sqrtf(5.0f)};
        const sx_arg as = OFF(conv, 6 * KVD);
        DISP(c, m->p.conv_res, H / 256, T, 1, 256, reg(m, R_A), as, L->aw, h, reg(m, R_R), OFF(bs, n * PH), BYTES(p));
        if (T > 1 && !m->spec_chunk) enc_conv_state(m, c, reg(m, R_A), as, H, T);
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
        if (T <= KMAX) {
            // decode and verification chunks: per token, each expert's rows read once, f32 dot
            // products per row (the same bits for a token in either)
            DISP(c, L->mx ? m->p.gu_mx : m->p.gu_nv, FF / 8, TOPK, T, 256, gw, gs, L->gfu, reg(m, R_XM), reg(m, R_IDX),
                 reg(m, R_HACT), BYTES(mp));
            DISP(c, L->mx ? m->p.dn_mx : m->p.dn_nv, H / 32, TOPK, T, 256, dw, ds, L->gfd, reg(m, R_HACT), reg(m, R_IDX),
                 reg(m, R_W), reg(m, R_YS), BYTES(dp));
        } else if (m->half_prefill) {
            // a prompt chunk on the tensor ops: the pairs' rows sorted by expert, their inputs in
            // half, tiles of up to 128 rows sharing each decoded weight tile
            uint32_t pp[2] = {NE, T * TOPK}, gp[2] = {H, TOPK};
            DISP(c, m->p.im_plan, 1, 1, 1, 1024, reg(m, R_IDX), B(m->msrt), B(m->mtiles), B(m->mmeta), gpu_bytes_arg(pp, 8));
            const uint32_t rows = T * TOPK + 31 * NE, maxt = rows / 128 + NE;
            DISP(c, m->p.im_gather, rows, 1, 1, 256, reg(m, R_XM), B(m->msrt), B(m->mmeta), B(m->xs_h), gpu_bytes_arg(gp, 8));
            DISP(c, L->mx ? m->p.im_gu_mx : m->p.im_gu_nv, 2 * FF / 64, maxt, 1, 128, gw, gs, L->gfu, B(m->xs_h), B(m->mtiles),
                 B(m->mmeta), B(m->hs_h), BYTES(mp));
            DISP(c, L->mx ? m->p.im_dn_mx : m->p.im_dn_nv, H / 64, maxt, 1, 128, dw, ds, L->gfd, B(m->hs_h), B(m->mtiles), B(m->mmeta),
                 B(m->msrt), reg(m, R_W), reg(m, R_YS), BYTES(dp));
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
        CrP p = {H, 0, T == 1 && !m->spec_chunk, sqrtf(5.0f)};
        const sx_arg ms = OFF(conv, 6 * KVD + 3 * H);
        DISP(c, m->p.conv_res, H / 256, T, 1, 256, reg(m, R_YSUM), ms, L->mw, reg(m, R_R), reg(m, R_HN), reg(m, R_HN), BYTES(p));
        if (T > 1 && !m->spec_chunk) enc_conv_state(m, c, reg(m, R_YSUM), ms, H, T);
    }
    // a verification chunk keeps this visit's rows for its commit
    if (m->spec_chunk) {
        const sx_arg sk = AT(m->spk, v * KMAX * SPK_W * 4);
        enc_copy(m, c, reg(m, R_K), sk, T * KVD);
        enc_copy(m, c, reg(m, R_V), OFF(sk, KMAX * KVD), T * KVD);
        enc_copy(m, c, reg(m, R_KN), OFF(sk, 2 * KMAX * KVD), T * KVD);
        enc_copy(m, c, reg(m, R_VN), OFF(sk, 3 * KMAX * KVD), T * KVD);
        enc_copy(m, c, reg(m, R_A), OFF(sk, 4 * KMAX * KVD), T * H);
        enc_copy(m, c, reg(m, R_YSUM), OFF(sk, 4 * KMAX * KVD + KMAX * H), T * H);
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
        // the drafter's features: the partial block at the input of visits 5, 23 and 41 (vLLM
        // ef687f6 nvidia/loop.py:342-345, 375-376)
        if (m->ds.on)
            for (uint32_t k = 0; k < 3; k++)
                if (v == config_mtp_target_layers[k]) enc_copy(m, c, h, OFF(reg(m, R_TAP), k * PH), T * H);
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
    if (m->ds.on) enc_copy(m, c, reg(m, R_XM), OFF(reg(m, R_TAP), 3 * PH), T * H); // and the final norm's output (visit 48)
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
    const bool ctx = m->ds.on && !m->spec_chunk;
    if (ctx) ds_rope_rows(m, 0, s->pos, T);
    Enc enc = {sx_cmd_begin(m->g->dev), m->g}, *c = &enc;
    if (!c->cmd) return "GpuFailed";
    enc_forward(m, d, c, T, trace, trow, tstride, all_logits);
    if (ctx) {
        enc_ds_ctx(m, d, c, T, (uint32_t)s->pos);
        m->ds.tap_t = T;
        m->ds.tap_pos0 = (uint32_t)s->pos;
    }
    char err[512];
    if (sx_cmd_finish(c->cmd, err, sizeof err) != 0) {
        dbg("command buffer: %s\n", err);
        return "GpuFailed";
    }
    if (m->spec_chunk) return OK; // model_spec_verify commits
    s->pos += T;
    s->chunk_t = 0; // no verification chunk to roll back into
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

// ---- the DSpark drafter --------------------------------------------------------------------
//
// Thinking Machines' block drafter (config.json mtp_config; mtp.safetensors of the official
// release, BF16), run as vLLM ef687f6 runs it (vllm/models/inkling2/nvidia/dspark.py, D):
// - every committed token leaves its context keys and values in the drafter's three layers: each
//   layer mixes the trunk's four features (the partial block at the input of visits 5, 23 and 41,
//   and the final norm's output) with softmax(fuse_weight[layer]), RMS-normalizes them with its
//   dflare_norm, and projects K (k_norm_ctx, RoPE at the token's position) and V (D:249-265, 403-408);
// - a draft is one block of 8 rows at positions p .. p + 7 (p the next token's position): the
//   anchor (that token's normalized embedding) and 7 mask_embed rows, through 3 pre-norm layers
//   (q/k RMSNorm per head and NeoX RoPE with theta 1e6, attention over the context keys and all 8
//   block keys, no causal mask, the local layers 0 and 2 the last 1024 keys up to p + 7, no sink;
//   SwiGLU 9216) and the
//   final norm (D:195-246, 303-318, 433-439);
// - the Markov head turns rows 1..7 into 7 drafts one after another: its 256-wide state starts at
//   0 and the previous token at the anchor; raw = rnn_proj [state, markov_w1[prev], h_i, f_rnn] +
//   bias (f_rnn: the last committed token's features mixed with fuse_weight[3]); the state gates
//   on raw's thirds; the logits are lm_head h_i + (lm_head hidden_out) latent + lm_head b, and the
//   draft their argmax (the softcap is monotone) (D:330-385, 544-561).
// The lm_head is the pack's (affine 4-bit, where the reference has BF16) and lm_head . hidden_out
// is formed once at load (f16). Drafts only decide what is verified: the output is the trunk's.

// ds_model: session_roll_to has no model argument (one model per process)
static const TensorInfo *st_get(StrMap *m, const char *name) {
    void **v = strmap_get(m, str(name));
    return v ? (const TensorInfo *)*v : NULL;
}

static sx_arg dsa(const Model *m, int r) { return AT(m->ds.a, m->ds.off[r] * 4); }
static sx_arg dsw(const Model *m, size_t off) { return AT(m->ds.w, off); }

// y [T][out] = W x [T][in] for W bf16 at byte offset w (rows ldw values, from column col0).
static void ds_lin(Model *m, Enc *c, size_t w, uint32_t ldw, uint32_t col0, uint32_t in, uint32_t out, sx_arg x, uint32_t xs, sx_arg y,
                   uint32_t ys, uint32_t T) {
    struct { uint32_t in, out, T, ldw, col0, xs, ys; } p = {in, out, T, ldw, col0, xs, ys};
    DISP(c, m->p.ds_mv, (out + 7) / 8, (T + 7) / 8, 1, 256, dsw(m, w), x, y, BYTES(p));
}

// The drafter's planes in the session's caches: its layer l's keys / values.
static void ds_planes(const Session *d, size_t l, sx_arg *kc, sx_arg *vc, uint32_t *ring) {
    const size_t elem = d->kv_half ? 2 : 4;
    const bool global = config_mtp_layer_kinds[l] == LAYER_KIND_GLOBAL;
    size_t r = 0; // this layer's place among the drafter's layers of its kind
    for (size_t i = 0; i < l; i++) r += (config_mtp_layer_kinds[i] == LAYER_KIND_GLOBAL) == global;
    if (global) {
        const size_t plane = (size_t)d->cap * KVD * elem;
        *kc = AT(d->kv, (n_full_vis + r) * 2 * plane);
        *vc = AT(d->kv, ((n_full_vis + r) * 2 + 1) * plane);
        *ring = 0;
    } else {
        const size_t plane = (size_t)WIN * KVD * elem;
        *kc = AT(d->ixk, (n_swa_vis + r) * 2 * plane);
        *vc = AT(d->ixk, ((n_swa_vis + r) * 2 + 1) * plane);
        *ring = WIN;
    }
}

// (cos, sin) of positions pos0 .. pos0 + T - 1 into rope rows row0.. (double precision angles).
static void ds_rope_rows(Model *m, size_t row0, size_t pos0, size_t T) {
    float *cs = gpu_floats(m->ds.cs);
    for (size_t t = 0; t < T; t++)
        for (size_t i = 0; i < HD / 2; i++) {
            const double inv = pow(CONFIG_MTP_ROPE_THETA, -(double)(2 * i) / (double)HD), a = (double)(pos0 + t) * inv;
            cs[((row0 + t) * (HD / 2) + i) * 2] = (float)cos(a);
            cs[((row0 + t) * (HD / 2) + i) * 2 + 1] = (float)sin(a);
        }
}

// Context keys and values of chunk rows 0..T-1 (their taps in R_TAP) at positions pos0.., and the
// last row's taps kept for f_rnn. The host wrote their RoPE rows 0..T-1.
static void enc_ds_ctx(Model *m, Session *d, Enc *c, uint32_t T, uint32_t pos0) {
    const Ds *ds = &m->ds;
    const uint32_t PH = PCH * H;
    for (size_t l = 0; l < DS_L; l++) {
        struct { uint32_t h, ts; float eps; } mp = {H, PH, EPS};
        DISP(c, m->p.ds_mix, T, 1, 1, 256, reg(m, R_TAP), dsw(m, ds->alpha + l * 16), dsw(m, ds->dfl[l]), dsa(m, DA_CTX), BYTES(mp));
        ds_lin(m, c, ds->kc[l], H, 0, H, KVD, dsa(m, DA_CTX), H, dsa(m, DA_KC), KVD, T);
        ds_lin(m, c, ds->vc[l], H, 0, H, KVD, dsa(m, DA_CTX), H, dsa(m, DA_VC), KVD, T);
        struct { uint32_t heads, hd, xs, norm; float eps; } rp = {NKV, HD, KVD, 1, EPS};
        DISP(c, m->p.ds_rope, NKV, T, 1, HD / 2, dsa(m, DA_KC), dsw(m, ds->kcn[l]), B(ds->cs), BYTES(rp));
        sx_arg kc, vc;
        uint32_t ring;
        ds_planes(d, l, &kc, &vc, &ring);
        StP sp = {KVD, pos0, ring, T};
        DISP(c, d->kv_half ? m->p.kv_store_h : m->p.kv_store, KVD / 256, T, 1, 256, dsa(m, DA_KC), dsa(m, DA_VC), kc, vc, BYTES(sp));
    }
    for (uint32_t k = 0; k < 4; k++) enc_copy(m, c, OFF(reg(m, R_TAP), k * PH + (T - 1) * H), OFF(dsa(m, DA_FTAP), k * H), H);
}

// SAETTA_DS_DUMP=<dir>: the first draft's inputs and outputs as raw files (for checking the
// drafter against a reference): the taps of the last chunk, the drafter's caches, f_rnn's taps,
// the block's final normalized rows, the Markov head's inputs, the fold and the drafts.
static void ds_dump(Model *m, Session *d, uint32_t x, const uint32_t *drafts, size_t nd) {
    Ds *ds = &m->ds;
    ds->dumped = true;
    const char *dir = getenv("SAETTA_DS_DUMP");
    char path[1024];
    FILE *f;
#define DUMP(name, ptr, bytes)                                                                     \
    do {                                                                                           \
        snprintf(path, sizeof path, "%s/%s", dir, name);                                           \
        if ((f = fopen(path, "wb"))) {                                                             \
            fwrite(ptr, 1, bytes, f);                                                              \
            fclose(f);                                                                             \
        }                                                                                          \
    } while (0)
    const float *af = gpu_floats(ds->a);
    DUMP("taps.f32", gpu_floats(m->scratch) + region_off[R_TAP], (size_t)4 * PCH * H * 4);
    DUMP("ftap.f32", af + ds->off[DA_FTAP], (size_t)4 * H * 4);
    DUMP("xn.f32", af + ds->off[DA_XN], (size_t)DS_BLK * H * 4);
    DUMP("base.f32", af + ds->off[DA_BASE], (size_t)(DS_BLK - 1) * VOCAB * 4);
    DUMP("mctx.f32", af + ds->off[DA_MCTX], (size_t)(DS_BLK - 1) * 3 * DS_R * 4);
    DUMP("frnn.f32", af + ds->off[DA_FRNN], (size_t)H * 4);
    DUMP("lat.f32", af + ds->off[DA_LAT], (size_t)DS_R * 4);
    DUMP("state.f32", af + ds->off[DA_STATE], (size_t)DS_R * 4);
    DUMP("fold.f16", sx_buffer_contents(ds->fold), (size_t)VOCAB * DS_R * 2);
    DUMP("foldb.f32", sx_buffer_contents(ds->foldb), (size_t)VOCAB * 4);
    const size_t elem = d->kv_half ? 2 : 4;
    for (size_t l = 0; l < DS_L; l++) {
        sx_arg kc, vc;
        uint32_t ring;
        ds_planes(d, l, &kc, &vc, &ring);
        const size_t rows = ring ? WIN : d->pos;
        char nm[32];
        snprintf(nm, sizeof nm, "k%zu.bin", l);
        DUMP(nm, (const uint8_t *)sx_buffer_contents(kc.buffer) + kc.offset, rows * KVD * elem);
        snprintf(nm, sizeof nm, "v%zu.bin", l);
        DUMP(nm, (const uint8_t *)sx_buffer_contents(vc.buffer) + vc.offset, rows * KVD * elem);
    }
    snprintf(path, sizeof path, "%s/meta.json", dir);
    if ((f = fopen(path, "w"))) {
        fprintf(f, "{\"p0\": %zu, \"x\": %u, \"kv_half\": %d, \"tap_t\": %u, \"tap_pos0\": %u, \"pch\": %d, \"drafts\": [", d->pos, x,
                d->kv_half ? 1 : 0, ds->tap_t, ds->tap_pos0, PCH);
        for (size_t i = 0; i < nd; i++) fprintf(f, "%s%u", i ? ", " : "", drafts[i]);
        fprintf(f, "]}\n");
        fclose(f);
    }
#undef DUMP
    dbg("model: DSpark dump in %s\n", dir);
}

// The drafter's block and Markov head after the trunk's last committed token: up to k drafts
// for the token x at position s->pos.
static Err ds_draft(Model *m, Session *d, uint32_t x, size_t k, uint32_t *out, size_t *nd) {
    Ds *ds = &m->ds;
    const uint32_t p0 = (uint32_t)d->pos, nk = (uint32_t)MIN(k, (size_t)DS_BLK - 1);
    *nd = 0;
    if (nk == 0 || p0 == 0) return OK;
    TRY(stage_tokens(m, &x, 1));
    ds_rope_rows(m, PCH, p0, DS_BLK);
    ((uint32_t *)sx_buffer_contents(ds->a))[ds->off[DA_PREV]] = x;
    memset(gpu_floats(ds->a) + ds->off[DA_STATE], 0, DS_R * 4);
    Enc enc = {sx_cmd_begin(m->g->dev), m->g}, *c = &enc;
    if (!c->cmd) return "GpuFailed";
    const sx_arg res = dsa(m, DA_RES), xn = dsa(m, DA_XN), cs = AT(ds->cs, PCH * (HD / 2) * 2 * 4);
    // the block: the anchor's normalized embedding, then mask rows
    enc_norm(m, c, B(m->stage), H, m->embed_norm, true, res, 1);
    for (uint32_t r = 1; r < DS_BLK; r++) enc_copy(m, c, dsw(m, ds->mask), OFF(res, r * H), H);
    for (size_t l = 0; l < DS_L; l++) {
        if (l == 0) enc_norm(m, c, res, H, dsw(m, ds->iln[0]), true, xn, DS_BLK);
        else {
            struct { uint32_t h; float eps; } ap = {H, EPS};
            DISP(c, m->p.ds_addn, DS_BLK, 1, 1, 256, res, dsa(m, DA_O), dsw(m, ds->iln[l]), xn, BYTES(ap));
        }
        ds_lin(m, c, ds->q[l], H, 0, H, QD, xn, H, dsa(m, DA_Q), QD, DS_BLK);
        ds_lin(m, c, ds->k[l], H, 0, H, KVD, xn, H, dsa(m, DA_KB), KVD, DS_BLK);
        ds_lin(m, c, ds->v[l], H, 0, H, KVD, xn, H, dsa(m, DA_VB), KVD, DS_BLK);
        struct { uint32_t heads, hd, xs, norm; float eps; } rq = {NH, HD, QD, 1, EPS}, rk = {NKV, HD, KVD, 1, EPS};
        DISP(c, m->p.ds_rope, NH, DS_BLK, 1, HD / 2, dsa(m, DA_Q), dsw(m, ds->qn[l]), cs, BYTES(rq));
        DISP(c, m->p.ds_rope, NKV, DS_BLK, 1, HD / 2, dsa(m, DA_KB), dsw(m, ds->kn[l]), cs, BYTES(rk));
        sx_arg kc, vc;
        uint32_t ring;
        ds_planes(d, l, &kc, &vc, &ring);
        // every block row sits at the block's last position p0 + 7 (vLLM ef687f6
        // ops/attention.py:903-917): a local layer sees the 1024 keys up to it
        const bool local = config_mtp_layer_kinds[l] == LAYER_KIND_LOCAL;
        const uint32_t lo = !local ? 0 : (p0 + DS_BLK > WIN ? p0 + DS_BLK - WIN : 0), range = p0 + DS_BLK - lo;
        const uint32_t nsplit = MIN((uint32_t)NSPLIT_DEC, MAX(1u, (range + 255) / 256)), span = (range + nsplit - 1) / nsplit;
        struct { uint32_t nh, nkv, hd, p0, lo, ring, nsplit, span; float scale; } ap = {NH, NKV, HD, p0, lo, ring, nsplit, span,
                                                                                         1.0f / sqrtf((float)HD)};
        DISP(c, d->kv_half ? m->p.ds_attn_h : m->p.ds_attn, NKV, nsplit, DS_BLK, 128, dsa(m, DA_Q), kc, vc, dsa(m, DA_KB), dsa(m, DA_VB),
             B(m->part), BYTES(ap));
        AttP mp = {NH, NKV, HD, DREL, EXT, 0, 0, 0, nsplit, span, DS_BLK, 1.0f};
        DISP(c, m->p.attn_merge, NH, DS_BLK, 1, HD, B(m->part), dsa(m, DA_SINK), dsa(m, DA_GATE1), dsa(m, DA_ATT), BYTES(mp));
        ds_lin(m, c, ds->o[l], QD, 0, QD, H, dsa(m, DA_ATT), QD, dsa(m, DA_O), H, DS_BLK);
        struct { uint32_t h; float eps; } ap2 = {H, EPS};
        DISP(c, m->p.ds_addn, DS_BLK, 1, 1, 256, res, dsa(m, DA_O), dsw(m, ds->pln[l]), xn, BYTES(ap2));
        ds_lin(m, c, ds->gt[l], H, 0, H, DS_FF, xn, H, dsa(m, DA_G), DS_FF, DS_BLK);
        ds_lin(m, c, ds->up[l], H, 0, H, DS_FF, xn, H, dsa(m, DA_U), DS_FF, DS_BLK);
        uint32_t nn = DS_BLK * DS_FF;
        DISP(c, m->p.ds_silu, (nn + 255) / 256, 1, 1, 256, dsa(m, DA_G), dsa(m, DA_U), dsa(m, DA_HM), BYTES(nn));
        ds_lin(m, c, ds->dn[l], DS_FF, 0, DS_FF, H, dsa(m, DA_HM), DS_FF, dsa(m, DA_O), H, DS_BLK);
    }
    {
        struct { uint32_t h; float eps; } ap = {H, EPS};
        DISP(c, m->p.ds_addn, DS_BLK, 1, 1, 256, res, dsa(m, DA_O), dsw(m, ds->norm), xn, BYTES(ap));
    }
    // the Markov head's inputs: f_rnn, [h_i, f_rnn] for rows 1..7, base logits of rows 1..7
    {
        struct { uint32_t h, ts; float eps; } mp = {H, H, EPS};
        DISP(c, m->p.ds_mix, 1, 1, 1, 256, dsa(m, DA_FTAP), dsw(m, ds->alpha + 3 * 16), dsw(m, ds->mdfl), dsa(m, DA_FRNN), BYTES(mp));
    }
    for (uint32_t i = 0; i < DS_BLK - 1; i++) {
        enc_copy(m, c, OFF(xn, (i + 1) * H), OFF(dsa(m, DA_INP), i * 2 * H), H);
        enc_copy(m, c, dsa(m, DA_FRNN), OFF(dsa(m, DA_INP), i * 2 * H + H), H);
    }
    ds_lin(m, c, ds->rnn, 2 * DS_R + 2 * H, 2 * DS_R, 2 * H, 3 * DS_R, dsa(m, DA_INP), 2 * H, dsa(m, DA_MCTX), 3 * DS_R, nk);
    enc_lin(m, c, &m->lm_head, OFF(xn, H), dsa(m, DA_BASE), nk, VOCAB);
    const uint32_t ng = (VOCAB + 255) / 256, ldr = 2 * DS_R + 2 * H;
    for (uint32_t i = 0; i < nk; i++) {
        DISP(c, m->p.ds_step, 1, 1, 1, 256, OFF(dsa(m, DA_MCTX), i * 3 * DS_R), dsw(m, ds->rnnb), dsa(m, DA_STATE), dsa(m, DA_PREV),
             dsw(m, ds->w1), dsw(m, ds->rnn), dsa(m, DA_LAT), BYTES(ldr));
        uint32_t vv = VOCAB;
        DISP(c, m->p.ds_fold, ng, 1, 1, 256, OFF(dsa(m, DA_BASE), (size_t)i * VOCAB), B(ds->foldb), B(ds->fold), dsa(m, DA_LAT),
             dsa(m, DA_PART), BYTES(vv));
        DISP(c, m->p.ds_argmax, 1, 1, 1, 256, dsa(m, DA_PART), BYTES(ng), dsa(m, DA_DRAFT), dsa(m, DA_PREV), BYTES(i));
    }
    char err[512];
    if (sx_cmd_finish(c->cmd, err, sizeof err) != 0) {
        dbg("command buffer: %s\n", err);
        return "GpuFailed";
    }
    const uint32_t *dr = (const uint32_t *)sx_buffer_contents(ds->a) + ds->off[DA_DRAFT];
    for (uint32_t i = 0; i < nk; i++) {
        if (dr[i] >= VOCAB) break;
        out[(*nd)++] = dr[i];
    }
    if (getenv("SAETTA_DS_DUMP") && !ds->dumped) ds_dump(m, d, x, out, *nd);
    return OK;
}

static Err ds_load_file(Arena *arena, Model *m, Gpu *g, const char **pathp) {
    Ds *ds = &m->ds;
    const char *dv = getenv("SAETTA_DSPARK");
    const char *cand[3] = {dv && strcmp(dv, "1") != 0 ? dv : NULL, path_join(arena, m->pack->dir, "mtp.safetensors"), NULL};
    {
        const DownloadPack *dp = download_find("inkling-dspark");
        char *dir;
        if (dp && !download_pack_dir(arena, dp, &dir)) cand[2] = path_join(arena, dir, "mtp.safetensors");
    }
    const char *path = NULL;
    for (int i = 0; i < 3 && !path; i++)
        if (cand[i] && access(cand[i], R_OK) == 0) path = cand[i];
    if (!path) return OK;
    *pathp = path;
    size_t flen;
    const uint8_t *bytes = sx_map_file(path, &flen);
    if (!bytes) return "MapFailed";
    StrMap tm = {0};
    Err e = safetensors_parse(arena, bytes, flen, 0, &tm);
    // layout: bf16 matrices first, then f32 vectors (norms, mask, bias, fuse softmax)
    size_t used = 0;
    const char *names_m[] = {"q_proj", "k_proj", "v_proj", "k_proj_ctx", "v_proj_ctx", "o_proj"};
    struct { const char *name; size_t *off; uint64_t numel; } mats[64];
    size_t nm = 0;
    char nb[128];
#define DSN(...) (snprintf(nb, sizeof nb, __VA_ARGS__), arena_strdup(arena, nb))
    for (size_t l = 0; l < DS_L; l++) {
        size_t *offs[6] = {&ds->q[l], &ds->k[l], &ds->v[l], &ds->kc[l], &ds->vc[l], &ds->o[l]};
        const uint64_t ne[6] = {(uint64_t)QD * H, (uint64_t)KVD * H, (uint64_t)KVD * H, (uint64_t)KVD * H, (uint64_t)KVD * H, (uint64_t)H * QD};
        for (int i = 0; i < 6; i++) mats[nm++] = (typeof(mats[0])){DSN("layers.%zu.self_attn.%s.weight", l, names_m[i]), offs[i], ne[i]};
        mats[nm++] = (typeof(mats[0])){DSN("layers.%zu.mlp.gate_proj.weight", l), &ds->gt[l], (uint64_t)DS_FF * H};
        mats[nm++] = (typeof(mats[0])){DSN("layers.%zu.mlp.up_proj.weight", l), &ds->up[l], (uint64_t)DS_FF * H};
        mats[nm++] = (typeof(mats[0])){DSN("layers.%zu.mlp.down_proj.weight", l), &ds->dn[l], (uint64_t)H * DS_FF};
    }
    mats[nm++] = (typeof(mats[0])){"markov_head.markov_w1.weight", &ds->w1, (uint64_t)CONFIG_VOCAB_SIZE * DS_R};
    mats[nm++] = (typeof(mats[0])){"markov_head.rnn_proj.weight", &ds->rnn, (uint64_t)3 * DS_R * (2 * DS_R + 2 * H)};
    struct { const char *name; size_t *off; uint64_t numel; } vecs[64];
    size_t nv = 0;
    for (size_t l = 0; l < DS_L; l++) {
        vecs[nv++] = (typeof(vecs[0])){DSN("layers.%zu.self_attn.q_norm.weight", l), &ds->qn[l], HD};
        vecs[nv++] = (typeof(vecs[0])){DSN("layers.%zu.self_attn.k_norm.weight", l), &ds->kn[l], HD};
        vecs[nv++] = (typeof(vecs[0])){DSN("layers.%zu.self_attn.k_norm_ctx.weight", l), &ds->kcn[l], HD};
        vecs[nv++] = (typeof(vecs[0])){DSN("layers.%zu.input_layernorm.weight", l), &ds->iln[l], H};
        vecs[nv++] = (typeof(vecs[0])){DSN("layers.%zu.post_attention_layernorm.weight", l), &ds->pln[l], H};
        vecs[nv++] = (typeof(vecs[0])){DSN("layers.%zu.dflare_norm.weight", l), &ds->dfl[l], H};
    }
    vecs[nv++] = (typeof(vecs[0])){"markov_head.dflare_norm.weight", &ds->mdfl, H};
    vecs[nv++] = (typeof(vecs[0])){"norm.weight", &ds->norm, H};
    vecs[nv++] = (typeof(vecs[0])){"mask_embed", &ds->mask, H};
    vecs[nv++] = (typeof(vecs[0])){"markov_head.rnn_proj.bias", &ds->rnnb, 3 * DS_R};
#undef DSN
    const TensorInfo *tw[64], *tv[64];
    for (size_t i = 0; i < nm && !e; i++) {
        tw[i] = st_get(&tm, mats[i].name);
        if (!tw[i] || tw[i]->dtype != DTYPE_BF16 || tensor_numel(tw[i]) != mats[i].numel || tw[i]->offset + mats[i].numel * 2 > flen)
            e = "DSparkTensor";
        *mats[i].off = used;
        used = ALIGN_UP(used + mats[i].numel * 2, 16);
    }
    for (size_t i = 0; i < nv && !e; i++) {
        tv[i] = st_get(&tm, vecs[i].name);
        if (!tv[i] || tv[i]->dtype != DTYPE_BF16 || tensor_numel(tv[i]) != vecs[i].numel || tv[i]->offset + vecs[i].numel * 2 > flen)
            e = "DSparkTensor";
        *vecs[i].off = used;
        used = ALIGN_UP(used + vecs[i].numel * 4, 16);
    }
    const TensorInfo *fw = e ? NULL : st_get(&tm, "fuse_weight"), *ho = e ? NULL : st_get(&tm, "markov_head.hidden_out.weight"),
                     *hb = e ? NULL : st_get(&tm, "markov_head.hidden_out.bias");
    if (!e && (!fw || fw->dtype != DTYPE_BF16 || tensor_numel(fw) != 16 || fw->offset + 32 > flen || !ho || ho->dtype != DTYPE_BF16 ||
               tensor_numel(ho) != (uint64_t)H * DS_R || ho->offset + (uint64_t)H * DS_R * 2 > flen || !hb || hb->dtype != DTYPE_BF16 ||
               tensor_numel(hb) != H || hb->offset + H * 2 > flen))
        e = "DSparkTensor";
    ds->alpha = used;
    used = ALIGN_UP(used + 4 * 16, 16);
    if (!e) e = gpu_new_buffer(g, used, &ds->w);
    if (e) {
        sx_unmap_file(bytes, flen);
        return e;
    }
    uint8_t *base = sx_buffer_contents(ds->w);
    for (size_t i = 0; i < nm; i++) memcpy(base + *mats[i].off, bytes + tw[i]->offset, mats[i].numel * 2);
    for (size_t i = 0; i < nv; i++) {
        float *o = (float *)(base + *vecs[i].off);
        const uint8_t *src = bytes + tv[i]->offset;
        for (size_t j = 0; j < vecs[i].numel; j++) {
            uint16_t b;
            memcpy(&b, src + 2 * j, 2);
            o[j] = ops_bf16_to_f32(b);
        }
    }
    {
        // softmax over the 4 taps of each fuse_weight row (D:506-508: refresh_derived_weights)
        float *al = (float *)(base + ds->alpha);
        for (size_t r = 0; r < 4; r++) {
            double z[4], mx = -INFINITY, sum = 0;
            for (size_t t = 0; t < 4; t++) {
                uint16_t b;
                memcpy(&b, bytes + fw->offset + 2 * (r * 4 + t), 2);
                z[t] = ops_bf16_to_f32(b);
                mx = fmax(mx, z[t]);
            }
            for (size_t t = 0; t < 4; t++) sum += z[t] = exp(z[t] - mx);
            for (size_t t = 0; t < 4; t++) al[r * 4 + t] = (float)(z[t] / sum);
        }
    }
    // lm_head . hidden_out and lm_head . hidden_out_bias on the GPU (qmm over the 256 columns of
    // hidden_out and the bias, then transposed to [vocab][256] f16)
    sx_buffer *xin = NULL, *prod = NULL;
    e = gpu_new_buffer(g, (size_t)(DS_R + 1) * H * 4, &xin);
    if (!e) {
        float *xi = gpu_floats(xin);
        for (size_t r = 0; r < DS_R; r++)
            for (size_t i = 0; i < H; i++) {
                uint16_t b;
                memcpy(&b, bytes + ho->offset + 2 * (i * DS_R + r), 2);
                xi[r * H + i] = ops_bf16_to_f32(b);
            }
        for (size_t i = 0; i < H; i++) {
            uint16_t b;
            memcpy(&b, bytes + hb->offset + 2 * i, 2);
            xi[DS_R * H + i] = ops_bf16_to_f32(b);
        }
    }
    sx_unmap_file(bytes, flen);
    if (!e) e = gpu_new_buffer(g, (size_t)(DS_R + 1) * VOCAB * 4, &prod);
    if (!e) e = gpu_new_buffer(g, (size_t)VOCAB * DS_R * 2, &ds->fold);
    if (!e) e = gpu_new_buffer(g, (size_t)VOCAB * 4, &ds->foldb);
    Enc fenc = {e ? NULL : sx_cmd_begin(g->dev), g};
    if (!e && !fenc.cmd) e = "GpuFailed";
    if (!e) {
        Enc *c = &fenc;
        enc_lin(m, c, &m->lm_head, B(xin), B(prod), DS_R + 1, VOCAB);
        uint32_t fp[2] = {VOCAB, DS_R};
        DISP(c, m->p.ds_fold_t, (VOCAB + 255) / 256, DS_R, 1, 256, B(prod), B(ds->fold), gpu_bytes_arg(fp, 8));
        uint32_t nb = VOCAB;
        DISP(c, m->p.copy, (nb + 255) / 256, 1, 1, 256, AT(prod, (size_t)DS_R * VOCAB * 4), B(ds->foldb), BYTES(nb));
        char err[512];
        if (sx_cmd_finish(c->cmd, err, sizeof err) != 0) {
            dbg("command buffer: %s\n", err);
            e = "GpuFailed";
        }
    }
    if (xin) sx_buffer_release(xin);
    if (prod) sx_buffer_release(prod);
    if (e) return e;
    // activations
    static const size_t aw[DA_N] = {[DA_RES] = DS_BLK * H, [DA_XN] = DS_BLK * H, [DA_Q] = DS_BLK * QD, [DA_KB] = DS_BLK * KVD,
                                    [DA_VB] = DS_BLK * KVD, [DA_ATT] = DS_BLK * QD, [DA_O] = DS_BLK * H, [DA_G] = DS_BLK * DS_FF,
                                    [DA_U] = DS_BLK * DS_FF, [DA_HM] = DS_BLK * DS_FF, [DA_CTX] = PCH * H, [DA_KC] = PCH * KVD,
                                    [DA_VC] = PCH * KVD, [DA_INP] = (DS_BLK - 1) * 2 * H, [DA_MCTX] = (DS_BLK - 1) * 3 * DS_R,
                                    [DA_STATE] = DS_R, [DA_LAT] = DS_R, [DA_PREV] = 4, [DA_DRAFT] = DS_BLK,
                                    [DA_PART] = 2 * ((VOCAB + 255) / 256), [DA_FRNN] = H, [DA_FTAP] = 4 * H,
                                    [DA_BASE] = (DS_BLK - 1) * VOCAB, [DA_GATE1] = DS_BLK * QD, [DA_SINK] = DS_BLK * NH};
    size_t o = 0;
    for (size_t i = 0; i < DA_N; i++) {
        ds->off[i] = o;
        o += ALIGN_UP(aw[i], 64);
    }
    ds->off[DA_N] = o;
    TRY(gpu_new_buffer(g, o * 4, &ds->a)); // ds_load releases what was made on an error
    float *af = gpu_floats(ds->a);
    for (size_t i = 0; i < DS_BLK * QD; i++) af[ds->off[DA_GATE1] + i] = 1e4f;      // sigmoid: 1
    for (size_t i = 0; i < DS_BLK * NH; i++) af[ds->off[DA_SINK] + i] = -INFINITY;  // no sink
    TRY(gpu_new_buffer(g, (size_t)(PCH + DS_BLK) * (HD / 2) * 2 * 4, &ds->cs));
    ds->on = true;
    dbg("model: DSpark drafter loaded (%s, %.0f MB)\n", path, (double)used / 1e6);
    return OK;
}

// Loads the DSpark drafter only when asked: SAETTA_DSPARK=1 (the pack folder's mtp.safetensors,
// else the inkling-dspark download's) or SAETTA_DSPARK=<file>; never with SAETTA_MTP=0 or
// SAETTA_SPEC=0. Off by default: at 4K, prompt lookup alone decodes faster than lookup with the
// drafter (an 8-row verify pass costs about 3 single steps). A missing file leaves it off; a file
// that does not load leaves it off with a warning (the model runs without it).
static Err ds_load(Arena *arena, Model *m, Gpu *g) {
    const char *mv = getenv("SAETTA_MTP"), *sv = getenv("SAETTA_SPEC"), *dv = getenv("SAETTA_DSPARK");
    if (!dv || !dv[0] || dv[0] == '0') return OK; // opt-in
    if ((mv && mv[0] == '0') || (sv && sv[0] == '0')) return OK; // no drafts: no drafter context per token either
    const char *path = "?";
    Err e = ds_load_file(arena, m, g, &path);
    if (e) {
        dbg("saetta: warning: the DSpark drafter %s did not load (%s); Inkling 2 Mini runs without it\n", path, e);
        Ds *ds = &m->ds;
        sx_buffer *bufs[] = {ds->w, ds->fold, ds->foldb, ds->a, ds->cs};
        for (size_t i = 0; i < COUNT(bufs); i++)
            if (bufs[i]) sx_buffer_release(bufs[i]);
        *ds = (Ds){0};
    }
    return OK;
}

// The commit of a verification chunk's first n rows (accepted drafts and the token before them)
// at positions pos0..: conv states and caches from the saved rows, the drafter's context.
static void enc_commit(Model *m, Session *d, Enc *c, uint32_t pos0, uint32_t n) {
    for (size_t v = 0; v < NVIS; v++) {
        const sx_arg conv = AT(d->bk, v * CONV_VIS * 4), sk = AT(m->spk, v * KMAX * SPK_W * 4);
        enc_conv_state(m, c, sk, conv, KVD, n);
        enc_conv_state(m, c, OFF(sk, KMAX * KVD), OFF(conv, 3 * KVD), KVD, n);
        enc_conv_state(m, c, OFF(sk, 4 * KMAX * KVD), OFF(conv, 6 * KVD), H, n);
        enc_conv_state(m, c, OFF(sk, 4 * KMAX * KVD + KMAX * H), OFF(conv, 6 * KVD + 3 * H), H, n);
        const bool full = is_full(vis_layer[v]);
        const size_t elem = d->kv_half ? 2 : 4;
        sx_arg kc, vc;
        uint32_t ring;
        if (full) {
            const size_t plane = (size_t)d->cap * KVD * elem;
            kc = AT(d->kv, vis_kv[v] * 2 * plane);
            vc = AT(d->kv, (vis_kv[v] * 2 + 1) * plane);
            ring = 0;
        } else {
            const size_t plane = (size_t)WIN * KVD * elem;
            kc = AT(d->ixk, vis_kv[v] * 2 * plane);
            vc = AT(d->ixk, (vis_kv[v] * 2 + 1) * plane);
            ring = WIN;
        }
        StP sp = {KVD, pos0, ring, n};
        DISP(c, d->kv_half ? m->p.kv_store_h : m->p.kv_store, KVD / 256, n, 1, 256, OFF(sk, 2 * KMAX * KVD), OFF(sk, 3 * KMAX * KVD), kc,
             vc, BYTES(sp));
    }
    if (m->ds.on) enc_ds_ctx(m, d, c, n, pos0);
}

// The sliding rings' rows at pos0..pos0 + T - 1 to (dir 0) or from (1) the backup, and the conv
// states likewise: what a commit replaces.
static void enc_backup(Model *m, Session *d, Enc *c, uint32_t pos0, uint32_t T, uint32_t dir) {
    const size_t elem = d->kv_half ? 2 : 4;
    struct { uint32_t words, pos0, ring, kmax, dir; } rp = {(uint32_t)(KVD * elem / 4), pos0, WIN, KMAX, dir};
    DISP(c, m->p.ds_ring, (rp.words + 255) / 256, T, n_swa_vis * 2, 256, B(d->ixk), B(m->spb), BYTES(rp));
    const uint32_t nc = (uint32_t)(CONV_BYTES / 4);
    const sx_arg cb = AT(m->spb, (size_t)n_swa_vis * 2 * KMAX * KVD * 4);
    if (dir == 0) DISP(c, m->p.copy, (nc + 255) / 256, 1, 1, 256, B(d->bk), cb, BYTES(nc));
    else DISP(c, m->p.copy, (nc + 255) / 256, 1, 1, 256, cb, B(d->bk), BYTES(nc));
}

// ---- sessions -----------------------------------------------------------------------------

// KV in f16 when kv_half (the server's default; SAETTA_KV_F32 and verify's exact checks ask f32),
// the full visits' and the rings alike.
Err session_create_cap(Gpu *g, bool kv_half, size_t cap, size_t max_cap, Session **out) {
    cap = ALIGN_UP(MIN(MAX(cap, (size_t)4), max_cap), 4);
    Session *s = xcalloc(1, sizeof *s);
    s->arch = &inkling_arch;
    s->cap = cap;
    s->max_cap = max_cap;
    s->kv_half = kv_half;
    s->roll = SESSION_NO_ROLL;
    const size_t elem = kv_half ? 2 : 4;
    TRY(gpu_new_buffer(g, kvf_bytes(cap, elem), &s->kv));
    TRY(gpu_new_buffer(g, RING_BYTES(elem), &s->ixk));
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
Err session_create(Gpu *g, Session **out) { return session_create_kv(g, getenv("SAETTA_KV_F32") == NULL, out); }

void session_reset(Session *s) {
    s->pos = 0;
    s->roll = SESSION_NO_ROLL;
    for (int k = 0; k < 2; k++) s->ck[k].tokens.len = 0;
    memset(sx_buffer_contents(s->bk), 0, sx_buffer_length(s->bk));
}

size_t session_kv_elem(const Session *s) { return s->kv_half ? 2 : 4; }
// The bytes that grow with the capacity (the full visits' KV); the rings and conv states are
// fixed (session_conv_bytes).
size_t session_kv_bytes(const Session *s, size_t cap) {
    return kvf_bytes(cap, session_kv_elem(s));
}
size_t session_conv_bytes(const Session *s, size_t cap) {
    return session_kv_bytes(s, cap) + RING_BYTES(session_kv_elem(s)) + CONV_BYTES;
}

// The full visits' rows below s->pos copied into new buffers of capacity cap.
// The peak is both buffers (session_conv_bytes doubled in the server's fit).
static Err regrow(Gpu *g, sx_buffer **kv, size_t old_cap, size_t cap, size_t rows, size_t elem) {
    sx_buffer *nb;
    TRY(gpu_new_buffer(g, kvf_bytes(cap, elem), &nb));
    const size_t row = KVD * elem;
    const uint8_t *src = sx_buffer_contents(*kv);
    uint8_t *dst = sx_buffer_contents(nb);
    for (size_t plane = 0; plane < n_kvf_planes * 2; plane++) memcpy(dst + plane * cap * row, src + plane * old_cap * row, rows * row);
    sx_buffer_release(*kv);
    *kv = nb;
    return OK;
}

Err session_grow(Session *s, Gpu *g, size_t cap) {
    if (!(cap >= s->cap && cap <= s->max_cap && cap % 4 == 0)) die("session_grow: bad capacity %zu\n", cap);
    if (cap == s->cap) return OK;
    TRY(regrow(g, &s->kv, s->cap, cap, s->pos, session_kv_elem(s)));
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
    Conv cv = {0};
    cv.cap = cap;
    TRY(gpu_new_buffer(g, kvf_bytes(cap, session_kv_elem(s)), &cv.kv));
    TRY(gpu_new_buffer(g, RING_BYTES(session_kv_elem(s)), &cv.ixk));
    TRY(gpu_new_buffer(g, CONV_BYTES, &cv.bk));
    *out = cv;
    return OK;
}
Err session_conv_shrink(const Session *s, Gpu *g, Conv *cv, size_t cap) {
    sx_buffer *nb;
    TRY(gpu_new_buffer(g, kvf_bytes(cap, session_kv_elem(s)), &nb));
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
bool model_has_mtp(const Model *m) { return m->ds.on; }
bool model_fast_prefill(const Model *m) { return m->fast_prefill; }
bool model_fast_prefill_available(const Model *m) { (void)m; return true; }
void model_set_fast_prefill(Model *m, bool on) { m->fast_prefill = on; }
static uint32_t greedy_pick(void *ctx, const float *row, bool has_draft, uint32_t draft) {
    (void)ctx, (void)has_draft, (void)draft;
    return model_argmax(row, VOCAB);
}
// Verification: x and the drafts as one chunk through the 48 visits (the same bits per token as
// decode: src/kernels and enc_lin), nothing committed while it runs; the longest prefix of drafts
// the rule agrees with is accepted, and only x and those rows are committed (conv states, rings,
// caches, the drafter's context), after the backup of what the commit replaces
// (session_roll_to). Rejected rows never reach a ring or a conv state.
Err model_spec_verify(Model *m, Session *s, uint32_t x, const uint32_t *drafts, size_t nd, SpecPick pick, void *ctx,
                      SpecResult *out) {
    if (!(nd >= 1 && nd < KMAX)) die("model_spec_verify: %zu drafts\n", nd);
    uint32_t toks[KMAX];
    toks[0] = x;
    memcpy(toks + 1, drafts, nd * sizeof(uint32_t));
    const uint32_t T = (uint32_t)(1 + nd), pos0 = (uint32_t)s->pos;
    m->spec_chunk = true;
    Err err = run_chunk(m, s, toks, T, NULL, 0, 1, true);
    m->spec_chunk = false;
    TRY(err);
    const float *lg = gpu_floats(m->logits);
    size_t j = 0;
    uint32_t next = pick(ctx, lg, true, drafts[0]);
    while (j < nd && next == drafts[j]) {
        j += 1;
        next = pick(ctx, lg + j * VOCAB, j < nd, j < nd ? drafts[j] : 0);
    }
    const uint32_t n = (uint32_t)j + 1;
    if (m->ds.on) ds_rope_rows(m, 0, pos0, n);
    Enc enc = {sx_cmd_begin(m->g->dev), m->g}, *c = &enc;
    if (!c->cmd) return "GpuFailed";
    enc_backup(m, s, c, pos0, n, 0);
    enc_commit(m, s, c, pos0, n);
    char e2[512];
    if (sx_cmd_finish(c->cmd, e2, sizeof e2) != 0) {
        dbg("command buffer: %s\n", e2);
        return "GpuFailed";
    }
    s->pos = pos0 + n;
    s->chunk_pos0 = pos0;
    s->chunk_t = n;
    for (int k = 0; k < 2; k++)
        if (s->ck[k].tokens.len != s->pos) s->ck[k].tokens.len = 0;
    *out = (SpecResult){j, next};
    return OK;
}
Err model_spec_greedy(Model *m, Session *s, uint32_t x, const uint32_t *drafts, size_t nd, SpecResult *out) {
    return model_spec_verify(m, s, x, drafts, nd, greedy_pick, NULL, out);
}
Err model_draft_mtp(Model *m, Session *s, uint32_t x, size_t k, uint32_t *out, size_t *nd) {
    *nd = 0;
    if (!m->ds.on) return OK;
    return ds_draft(m, s, x, k, out, nd);
}
Err model_lora_load(Arena *a, Model *m, const char *name, const char *dir) { (void)a, (void)m, (void)name, (void)dir; return NI; }
int model_lora_find(const Model *m, const char *name) { (void)m, (void)name; return -1; }
void model_lora_select(Model *m, int idx) { (void)m, (void)idx; }
int model_lora_selected(const Model *m) { (void)m; return -1; }
size_t model_lora_count(const Model *m) { (void)m; return 0; }
const char *model_lora_name(const Model *m, size_t i) { (void)m, (void)i; return NULL; }
// Back to just after token p - 1 inside the last verification chunk (a reply that ended with
// accepted drafts still queued): the conv states and ring rows its commit replaced come back
// from the backup, then the chunk's first p - chunk_pos0 rows are committed again.
bool session_roll_to(Session *s, size_t p, const uint32_t *toks) {
    (void)toks;
    Model *m = (Model *)ds_model;
    if (!m || s->chunk_t == 0 || p <= s->chunk_pos0 || p > s->chunk_pos0 + s->chunk_t) return false;
    if (p == s->pos) return true;
    const uint32_t pos0 = (uint32_t)s->chunk_pos0, n = (uint32_t)(p - s->chunk_pos0);
    if (m->ds.on) ds_rope_rows(m, 0, pos0, n);
    Enc enc = {sx_cmd_begin(m->g->dev), m->g}, *c = &enc;
    if (!c->cmd) return false;
    enc_backup(m, s, c, pos0, (uint32_t)s->chunk_t, 1);
    enc_commit(m, s, c, pos0, n);
    char err[512];
    if (sx_cmd_finish(c->cmd, err, sizeof err) != 0) return false;
    s->pos = p;
    s->chunk_t = n;
    return true;
}
Err session_save_point(Session *s, Gpu *g, const uint32_t *toks, size_t n, bool may_alloc) {
    (void)s, (void)g, (void)toks, (void)n, (void)may_alloc;
    return OK;
}
void session_start_at_own_point(Session *s, size_t n) { (void)s, (void)n; }
Err session_start_from(Session *s, Gpu *g, const Conv *src, size_t n, size_t alloc_max) {
    (void)s, (void)g, (void)src, (void)n, (void)alloc_max;
    return NI;
}
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
    .chat = CHAT_INKLING,
});

// ---- verification hooks (src/inkling.h) ---------------------------------------------------

bool inkling_half_available(const Model *m) { return m->arch == &inkling_arch && m->accel; }
void inkling_set_half(Model *m, bool on) { m->half_prefill = on && m->accel; }
bool inkling_half(const Model *m) { return m->half_prefill; }
size_t inkling_prompt_chunk(void) { return PCH; }
bool inkling_has_routes(const Model *m) { return m->arch == &inkling_arch && m->routes != NULL; }
void inkling_set_routes(Model *m, InkRouteMode mode) { m->replay = mode == INK_ROUTES_REPLAY; }
InkRouteStats inkling_route_stats(Model *m) {
    InkRouteStats r = {0, 0, UINT32_MAX, 0, UINT32_MAX};
    if (!m->rstats) return r;
    uint32_t *st = sx_buffer_contents(m->rstats);
    float mg;
    memcpy(&mg, &st[3], 4);
    r = (InkRouteStats){st[0], st[1], st[2], mg, st[4]};
    st[0] = st[1] = st[3] = 0;
    st[2] = st[4] = UINT32_MAX;
    return r;
}
