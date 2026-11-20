// Qwen3.8-27B internals (src/dense.c), compiled against src/config_27b.h. Only dense.c
// includes this; the public API is model.h's. Pre-norm decoder with plain residuals: 64 layers,
// three Gated DeltaNet blocks then one gated attention block, each followed by a dense SwiGLU
// MLP.
#ifndef SAETTA_DENSE_H
#define SAETTA_DENSE_H

#include "config_27b.h"
#include "model.h"
#include "model_arch.h"

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

// The MTP draft head (V27:67-190): x = fc([rms_e(embed(t)) | rms_h(h)]), one gated-attention
// decoder layer with its own KV plane (slot n_attn), then norm and the shared lm_head; its
// output after norm is the next draft's h (as vLLM's drafter).
typedef struct {
    Linear fc;            // [H][2H], input columns [embedding | hidden]
    Vec n_e, n_h, norm;   // pre_fc_norm_embedding, pre_fc_norm_hidden, mtp.norm
    Layer layer;          // layer.attn->slot == n_attn
} Mtp;

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
// attn_q6h / attn_fl: prompt-chunk attention (src/kernels/dense_attn.metal), NULL when it does
// not compile or SAETTA_ATTN27_SIMD is set (then attn_full_6h serves every chunk); attn_qb is
// the query tokens per threadgroup of attn_fl.
typedef struct {
    sx_pipeline *dense, *to_half, *dq, *mmh, *dq_bf16;
    sx_pipeline *attn_q6h, *attn_fl;
    uint32_t attn_qb;
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
        uint32_t slot; // the member's output in the group: q 0, k 1, v 2; gate 0, up 1
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


typedef struct {
    Model *m;
    sx_cmd *cmd;
    bool failed;
    bool xh_set;
    sx_buffer *xh_buf;
    size_t xh_off, xh_n;
} Enc;

struct Model {
    const ModelArch *arch; // first: model_api.c dispatches on it
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
    Mtp *mtp;           // NULL without the head (no mtp.fc in the pack, or SAETTA_MTP=0)
    // MTP: trunk outputs after the final norm ([CH + 1][H]: row 0 the one before the
    // current chunk, then the chunk's rows) and the draft logits.
    sx_buffer *mtp_hid, *mtp_logits;
    // LoRA adapters (model_lora_load) and the one the next passes apply (-1: none).
    LoraPipes lp;
    bool lp_ok;
    Adapter lora[MODEL_MAX_LORA];
    size_t n_lora;
    int lora_on;
    sx_buffer *stage;   // [CH][H] chunk embeddings, written by the host
};

#endif
