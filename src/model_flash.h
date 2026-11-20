// Qwen3.8-Flash-Next internals (src/model.c), compiled against src/config_flash.h. Only
// model.c includes this; the rest of the engine sees model.h's opaque Model.
#ifndef SAETTA_MODEL_FLASH_H
#define SAETTA_MODEL_FLASH_H

#include "config_flash.h"
#include "model.h"
#include "model_arch.h"
#include "xcache.h"

// Width of the hyper-connection streams: hc_count x hidden_size (10240).
#define HCW (CONFIG_HC_COUNT * CONFIG_HIDDEN_SIZE)
// n-gram heads of the PLE layer (16).
#define MODEL_NG_HEADS ((CONFIG_NGRAM_SIZE - 1) * CONFIG_HEADS_PER_NGRAM)

// An f32 parameter (norm weight, conv weight, ...) in the GPU parameter buffer: float offset
// off, n values at s.
typedef struct {
    size_t off;
    float *s;
    size_t n;
} Vec;

typedef struct {
    sx_buffer *buf;
    size_t used;
} Params;

typedef struct {
    Vec norm;
    Linear down, up;
    bool has_inject;
    Linear inject;
} Residual;

typedef struct {
    size_t slot;
    Linear qkv, z, a, b, out;
    Vec conv; // [CONV][KCONV]
    Vec a_log, dt_bias;
    Vec norm; // [DV], plain weight
} Gdn;

typedef struct {
    size_t slot;
    Linear q, k, v, o;
    Vec q_norm, k_norm;
    Linear ix;
    Vec ix_qn, ix_kn;
} Attn;

typedef struct {
    // experts of this layer (the router's rows): 512 in the full model, 256 in the pruned
    // Coder trunk, 512 in its MTP head; buffers are sized for num_experts (the most)
    uint32_t ne;
    Linear router, shared_gate, sg, su, sd, eg, eu, ed;
    uint32_t xl1; // streamed: 1 + this layer's index in the expert cache; 0 when resident
} Moe;

typedef struct {
    Linear key, value;
    Vec norm_key, norm_query, norm_conv;
    Vec conv; // [HCW][4]
    int64_t mult[CONFIG_NGRAM_SIZE];
    int64_t offsets[MODEL_NG_HEADS];
    int64_t vocab[MODEL_NG_HEADS];
    TensorInfo shard_w[128], shard_s[128], shard_b[128];
} Ple;

typedef struct {
    Residual attn_hc, mlp_hc;
    Gdn *gdn;   // NULL unless a GDN layer
    Attn *attn; // NULL unless an attention layer
    Moe moe;
    Ple *ple;   // NULL unless the layer carries the n-gram embedding
} Layer;

// The MTP draft head (wiring as in sushi src/qwen4/qwen4_forward.zig:1220-1308, MIT): row r
// pairs the trunk's last-layer streams at position r (before the mixer) with the embedding of
// token r+1 and runs at position r+1:
//   X_k = fc_hidden(rms_10240(H) * (1 + n_h))_k + fc_embedding(rms(embed(tok)) * (1 + n_e)), k < 4,
// then one QSA + MoE layer with its own KV slot, its own mixer and the trunk's lm_head. The
// pack keeps its dense matrices in BF16; they are requantized at load to 4-bit, group 32
// (routers 8-bit).
typedef struct {
    Linear fc_e, fc_h;
    Vec n_e, n_h;
    Layer layer;
    Residual mixer;
} Mtp;

// The templated pipelines: field name, kernel name (in this order; the profiler reports them so).
#define MODEL_PIPES(X)                                                                          \
    X(q4, "qmatvec_4_32") X(q8, "qmatvec_8_64") X(qmv4_l8, "qmv4_l8") X(qmv4_l256, "qmv4_l256")    \
    X(qmv_multi_l8, "qmv_multi_l8") X(qmv_multi_l32, "qmv_multi_l32")                            \
    X(qmv_multi_l256, "qmv_multi_l256") X(qmv_gate_up, "qmv_gate_up") X(qmv_up_mix, "qmv_up_mix") \
    X(qmv_gate_up_sh, "qmv_gate_up_sh") X(qmv8_router_sg, "qmv8_router_sg")                      \
    X(gdn_conv_norm, "gdn_conv_norm") X(attn_prep_dec, "attn_prep_dec") X(mv_bf16, "mv_bf16")      \
    X(ix_score_sg, "ix_score_sg") X(ix_select_rx, "ix_select_rx") X(qmv_down_sh, "qmv_down_sh")    \
    X(add_streams, "add_streams") X(bcast_streams, "bcast_streams") X(hc_add_norm, "hc_add_norm")  \
    X(qe4, "qmatvec_expert_4_32") X(rmsnorm, "rmsnorm") X(silu_scale, "silu_scale")                \
    X(hc_mix, "hc_mix") X(hc_add, "hc_add") X(gdn_conv, "gdn_conv") X(gdn_qknorm, "gdn_qknorm")    \
    X(gdn_step, "gdn_step") X(gdn_gnorm, "gdn_gnorm") X(rope, "rope") X(copy_f32, "copy_f32")      \
    X(attn_decode, "attn_decode") X(moe_topk, "moe_topk") X(swiglu, "swiglu")                      \
    X(moe_combine, "moe_combine") X(ple_gate, "ple_gate") X(ple_conv, "ple_conv")                  \
    X(copy_rows, "copy_rows") X(ix_block, "ix_block") X(ix_score, "ix_score")                      \
    X(ix_select, "ix_select") X(ix_score_t, "ix_score_t") X(ix_select_t, "ix_select_t")            \
    X(moe_group, "moe_group") X(gather_rows, "gather_rows") X(gather_rows_h, "gather_rows_h")      \
    X(swiglu_h, "swiglu_h") X(moe_combine_g, "moe_combine_g") X(mmg4, "qmm_grouped_4_32")         \
    X(attn_split, "attn_split") X(attn_split_g, "attn_split_g") X(attn_split_gh, "attn_split_gh")  \
    X(copy_rows_h, "copy_rows_h") X(attn_merge, "attn_merge") X(mm4, "qmm_4_32") X(mm8, "qmm_8_64")

typedef struct {
#define MODEL_PIPE_FIELD(f, k) sx_pipeline *f;
    MODEL_PIPES(MODEL_PIPE_FIELD)
#undef MODEL_PIPE_FIELD
} Pipes;

// Prefill matmuls on the neural accelerators (half inputs): src/kernels/qmm_mpp.metal.
// qsa: QSA attention for prompt chunks on the accelerators (f16 KV), NULL if it does not compile.
// dq / mmh: dense prompt matmuls as unpack-to-f16 then an f16 GEMM (SAETTA_DENSE_FUSED=1 keeps
// the fused tile-dequant kernel); NULL if they do not compile.
typedef struct {
    sx_pipeline *dense, *grouped, *to_half, *qsa, *dq, *mmh, *dq_bf16;
} Mpp;

// Run-time (bits, group) kernels (src/qgen.h) for tensors outside the 4/32 and 8/64 fast
// paths (the Coder pack mixes 4/5/6/8 bits and groups 32/64/128); compiled only when needed.
typedef struct {
    sx_pipeline *mv, *expert;
    sx_pipeline *multi[3]; // lanes 8, 32, 256
    sx_pipeline *gate_up, *gate_up_sh, *down_sh, *up_mix;
    sx_pipeline *rows1[MODEL_SMALL + 1];
    sx_pipeline *rows[MODEL_SMALL + 1];
    sx_pipeline *qmm, *qmm_grouped;
    sx_pipeline *dq, *qmmh, *qmmh_g; // NULL when they do not compile
} Gen;


// A point where the command buffer stops until the host has mapped one layer's expert ids to
// cache slots (streamed packs): the GPU signals sig once the ids are written, the host remaps
// them in place, loads the missing experts and sets sig + 1.
typedef enum { XS_DECODE, XS_ROWS, XS_TILES } XsKind;
typedef struct {
    uint8_t kind;
    uint32_t layer, n; // cache layer; ids to map (XS_DECODE, XS_ROWS)
    uint64_t sig;
} XSvc;
#define MODEL_MAX_XSVC 160

// Encoder for one token: dispatches go into one serial command buffer.
typedef struct {
    Model *m;
    sx_cmd *cmd;
    bool failed;
    XSvc xs[MODEL_MAX_XSVC]; // stops encoded in cmd, in order
    uint32_t nxs;
    // B.xh holds the half copy of this (buffer, offset, count); any dispatch other than an
    // accelerator matmul clears it, since it may have rewritten the source.
    bool xh_set;
    sx_buffer *xh_buf;
    size_t xh_off, xh_n;
} Enc;

// A verification chunk encoded during the last draft step.
typedef struct {
    Enc e;
    size_t T, pos;
    const Session *s;
} PendVerify;

struct Model {
    const ModelArch *arch; // first: model_api.c dispatches on it
    Gen *gen; // NULL unless the pack mixes quantizations
    const Pack *pack;
    Gpu *g;
    Pipes p;
    Mpp *mpp; // NULL when the accelerator kernels do not compile
    // qmv_rows kernels per chunk size 1..SMALL: [0] 4-bit group 32, [1] 8-bit group 64; rows1_p
    // one row per simdgroup, rows_p 4 rows per simdgroup sharing the inputs
    sx_pipeline *rows_p[2][MODEL_SMALL + 1];
    sx_pipeline *rows1_p[2][MODEL_SMALL + 1];
    // Prefill 4-bit matmuls on the accelerators in half precision (8-bit router and gates stay
    // f32). Off with SAETTA_PREFILL_F32=1 (exact f32 path) or when the kernels do not compile.
    bool fast_prefill;
    Layer *layers; // num_hidden_layers
    Residual mixer;
    Linear embed, lm_head;
    Params params;
    sx_buffer *scratch, *bscratch, *logits;
    sx_buffer *vlogits; // [KMAX][vocab], verification chunks
    // MTP head (NULL when the pack has none or SAETTA_MTP=0).
    Mtp *mtp;
    sx_buffer *mtp_hid;    // [CH + 1][HCW] trunk streams before the mixer, rows paired with the next tokens
    sx_buffer *mtp_chain;  // [HCW] the head's own stream, fed to the next draft step
    sx_buffer *mtp_logits; // [vocab]
    // Prefill inputs of a chunk, prepared on the host (by a thread for the next chunk while
    // the GPU runs the current one): [CH][H] embeddings, then [CH][ple_embed_dim] n-gram rows.
    sx_buffer *stage[2];
    // A verification chunk encoded while the MTP head's last draft step ran on the GPU
    // (draft_mtp), committed by verify_chunk once the drafts are known; any other call drops it.
    bool has_pend;
    PendVerify pend;
    // Streamed experts (SAETTA_EXPERT_CACHE_GB): the cache, the event the command buffers stop
    // on, and its last value handed out. NULL when every expert is resident.
    XCache *xc;
    sx_event *xev;
    uint64_t xev_val;
};

static_assert(MODEL_NG_HEADS == SESSION_NG_HEADS);

#endif
