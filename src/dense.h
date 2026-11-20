// Qwen3.8-27B internals, included by model.h in the 27B build (CONFIG_ARCH_27B). The public API
// is model.h's; the forward is src/dense.c. Pre-norm decoder with plain residuals: 64 layers,
// three Gated DeltaNet blocks then one gated attention block, each followed by a dense SwiGLU
// MLP (docs/model.md, "Qwen3.8-27B").
#ifndef SAETTA_DENSE_H
#define SAETTA_DENSE_H

// Residual width: one stream of hidden_size (Flash-Next has 4); verify's golden shapes use it.
#define HCW CONFIG_HIDDEN_SIZE

#include "lora.h"

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
    size_t slot;
    Linear qkv, z, a, b, out;
    Vec conv; // [CONV][KCONV]
    Vec a_log, dt_bias;
    Vec norm; // [DV]
} Gdn;

typedef struct {
    size_t slot;
    Linear q, k, v, o;
    Vec q_norm, k_norm;
} Attn;

typedef struct {
    Linear gate, up, down;
} Mlp;

typedef struct {
    Vec in_norm, post_norm; // plain RMSNorm weights (the pack stores 1 + w)
    Gdn *gdn;               // NULL unless a GDN layer
    Attn *attn;             // NULL unless an attention layer
    Mlp mlp;
} Layer;

#define MODEL_PIPES(X)                                                                                         \
    X(q4, "qmatvec_4_32") X(q8, "qmatvec_8_64") X(qmv4_l8, "qmv4_l8") X(qmv4_l256, "qmv4_l256")                   \
    X(qmv_multi_l8, "qmv_multi_l8") X(qmv_multi_l32, "qmv_multi_l32") X(qmv_multi_l256, "qmv_multi_l256")          \
    X(mv_bf16, "mv_bf16") X(rmsnorm, "rmsnorm") X(add_rmsnorm, "add_rmsnorm") X(gdn_conv, "gdn_conv")               \
    X(gdn_conv_norm, "gdn_conv_norm") X(gdn_qknorm, "gdn_qknorm") X(gdn_step, "gdn_step")                          \
    X(gdn_gnorm, "gdn_gnorm_silu") X(rope, "rope") X(copy_f32, "copy_f32") X(copy_rows, "copy_rows")               \
    X(copy_rows_h, "copy_rows_h") X(swiglu, "swiglu") X(attn_full, "attn_full_6") X(attn_full_h, "attn_full_6h")   \
    X(attn_merge, "attn_merge") X(mm4, "qmm_4_32") X(mm8, "qmm_8_64")

typedef struct {
#define MODEL_PIPE_FIELD(f, k) sx_pipeline *f;
    MODEL_PIPES(MODEL_PIPE_FIELD)
#undef MODEL_PIPE_FIELD
} Pipes;

// Prompt matmuls on the neural accelerators (half inputs), src/kernels/qmm_mpp.metal; NULL
// fields when a kernel does not compile.
typedef struct {
    sx_pipeline *dense, *to_half, *dq, *mmh, *dq_bf16;
} Mpp;

// Run-time (bits, group) kernels (src/qgen.h): the 27B packs are 4-bit group 64 and 8-bit
// group 64, outside the 4/32 templated paths.
typedef struct {
    sx_pipeline *mv;
    sx_pipeline *multi[3]; // lanes 8, 32, 256
    sx_pipeline *rows1[MODEL_SMALL + 1];
    sx_pipeline *rows[MODEL_SMALL + 1];
    sx_pipeline *qmm;
    sx_pipeline *dq, *qmmh; // NULL when they do not compile
} Gen;

// A LoRA adapter (PEFT format) applied at run time: per target linear y += scale * B (A x).
// Linears sharing an input form one group: their A rows concatenated ([rt][in], rt = n * r) and
// their B matrices applied to slices of t = A x. Groups per layer: q/k/v, o, gate/up, down.
// A and B are f16 in one GPU buffer (byte offsets).
enum { LORA_QKV, LORA_O, LORA_GU, LORA_DOWN, LORA_NGROUPS };
typedef struct {
    bool on;
    uint32_t in, rt, n;
    size_t a;
    struct {
        size_t b;
        uint32_t out, r0;
    } up[3];
} LoraGroup;

typedef struct {
    const char *name;
    sx_buffer *buf;
    float scale;
    uint32_t r;
    LoraGroup g[CONFIG_NUM_HIDDEN_LAYERS][LORA_NGROUPS];
} Adapter;

#define MODEL_MAX_LORA 8

typedef struct Model Model;
typedef struct Session Session;

typedef struct {
    Model *m;
    sx_cmd *cmd;
    bool failed;
    bool xh_set;
    sx_buffer *xh_buf;
    size_t xh_off, xh_n;
} Enc;

struct Model {
    Gen *gen;
    const Pack *pack;
    Gpu *g;
    Pipes p;
    Mpp *mpp;
    sx_pipeline *rows_p[2][MODEL_SMALL + 1];
    sx_pipeline *rows1_p[2][MODEL_SMALL + 1];
    // Prompt matmuls on the accelerators in half precision; off with SAETTA_PREFILL_F32=1.
    bool fast_prefill;
    Layer *layers; // num_hidden_layers
    Vec final_norm;
    Linear embed, lm_head;
    Params params;
    sx_buffer *scratch, *bscratch, *logits;
    sx_buffer *vlogits; // [KMAX][vocab], verification chunks
    void *mtp;          // always NULL: the 27B MTP head is not loaded yet (docs/dense-lora.md)
    // LoRA adapters (model_lora_load) and the one the next passes apply (-1: none).
    LoraPipes lp;
    bool lp_ok;
    Adapter lora[MODEL_MAX_LORA];
    size_t n_lora;
    int lora_on;
    sx_buffer *stage;   // [CH][H] chunk embeddings, written by the host
};

// Recurrent state of a restore point.
typedef struct {
    sx_buffer *gdn, *conv;
    VecU32 tokens;
} Ckpt;

// A prefix point; absent while gdn is NULL.
typedef struct {
    sx_buffer *gdn, *conv;
} Point;

// The per-conversation part of a session: KV cache and restore points (see the Flash-Next
// Conv in model.h for the sharing scheme).
typedef struct {
    size_t cap;
    sx_buffer *kv;
    Ckpt ck[2];
    Point pts[PT_MAX];
    VecU32 ptoks;
} Conv;

struct Session {
    size_t pos;
    size_t cap, max_cap;
    bool kv_half;
    sx_buffer *gdn_state; // [n_gdn][NV][DV][DK]
    sx_buffer *gdn_conv;  // [n_gdn][3][CONV]
    sx_buffer *kv;        // [n_attn][2][cap][NKV*HD], f16 or f32
    sx_buffer *rope;      // [max_cap][ROT/2] (cos, sin)
    float *trace;
    size_t trace_len;
    sx_buffer *trace_buf;
    bool spec;
    sx_buffer *sp_gdn;  // [KMAX][n_gdn][NV][DV][DK]
    sx_buffer *sp_conv; // [KMAX][n_gdn][3][CONV]
    size_t roll;
    Ckpt ck[2];
    size_t ck_pick;
    Point pts[PT_MAX];
    VecU32 ptoks;
    size_t chunk_pos0, chunk_t;
};

#endif
