// The table each model exports (flash_arch in src/model.c, dense_arch in src/dense.c) and
// src/model_api.c dispatches through: one binary holds both models, each compiled against its
// own compile-time constants (src/config_flash.h, src/config_27b.h). Model starts with a
// `const ModelArch *arch` field in both, which model_api.c reads; Session carries one too.
#ifndef SAETTA_MODEL_ARCH_H
#define SAETTA_MODEL_ARCH_H

#include "model.h"

struct ModelArch {
    ModelInfo info;
    Err (*load)(Arena *a, const Pack *pack, Gpu *g, Model **out);
    size_t (*check_pack)(Arena *a, const Pack *pack, FILE *out);
    bool (*has_mtp)(const Model *m);
    bool (*fast_prefill)(const Model *m);
    bool (*fast_prefill_available)(const Model *m);
    void (*set_fast_prefill)(Model *m, bool on);
    Err (*step)(Model *m, Session *s, uint32_t token, float **logits);
    Err (*prefill)(Model *m, Session *s, const uint32_t *tokens, size_t n, sx_buffer *trace, float **logits);
    Err (*prefill_with)(Model *m, Session *s, const uint32_t *tokens, size_t n, sx_buffer *trace, const PrefillHook *hook,
                        float **logits);
    Err (*verify_chunk)(Model *m, Session *s, const uint32_t *toks, size_t n, float **logits);
    Err (*spec_greedy)(Model *m, Session *s, uint32_t x, const uint32_t *drafts, size_t nd, SpecResult *out);
    Err (*spec_verify)(Model *m, Session *s, uint32_t x, const uint32_t *drafts, size_t nd, SpecPick pick, void *ctx,
                       SpecResult *out);
    Err (*draft_mtp)(Model *m, Session *s, uint32_t x, size_t k, uint32_t *out, size_t *nd);
    Err (*lora_load)(Arena *a, Model *m, const char *name, const char *dir);
    int (*lora_find)(const Model *m, const char *name);
    void (*lora_select)(Model *m, int idx);
    int (*lora_selected)(const Model *m);
    size_t (*lora_count)(const Model *m);
    const char *(*lora_name)(const Model *m, size_t i);
    void (*prof_init)(void);
    void (*prof_reset)(void);
    void (*prof_report)(size_t tokens);
    Err (*s_create)(Gpu *g, Session **out);
    Err (*s_create_kv)(Gpu *g, bool kv_half, Session **out);
    Err (*s_create_cap)(Gpu *g, bool kv_half, size_t cap, size_t max_cap, Session **out);
    size_t (*s_kv_elem)(const Session *s);
    size_t (*s_kv_bytes)(const Session *s, size_t cap);
    Err (*s_grow)(Session *s, Gpu *g, size_t cap);
    Err (*s_save_checkpoint)(Session *s, const uint32_t *toks, size_t n);
    Err (*s_save_checkpoint_at)(Session *s, size_t slot, const uint32_t *toks, size_t n);
    size_t (*s_reusable)(Session *s, const uint32_t *prompt, size_t n);
    size_t (*s_conv_bytes)(const Session *s, size_t cap);
    Err (*s_new_conv)(const Session *s, Gpu *g, size_t cap, Conv *out);
    Err (*s_conv_shrink)(const Session *s, Gpu *g, Conv *cv, size_t cap);
    void (*s_swap_conv)(Session *s, Conv *cv);
    bool (*s_roll_to)(Session *s, size_t p, const uint32_t *toks);
    Err (*s_save_point)(Session *s, Gpu *g, const uint32_t *toks, size_t n);
    void (*s_start_at_own_point)(Session *s, size_t n);
    Err (*s_start_from)(Session *s, Gpu *g, const Conv *src, size_t n);
    void (*s_restore)(Session *s);
    void (*s_push_history)(Session *s, uint32_t token);
    void (*s_reset)(Session *s);
};

extern const ModelArch flash_arch, dense_arch, dsv4_arch;

#endif
