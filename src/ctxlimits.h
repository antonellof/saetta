// Context capacity limits shared by the sessions (src/model.h, session_ctx_parse) and
// `saetta start --ctx` (src/launch.c, which cannot include model.h: its Session type clashes).
#ifndef SAETTA_CTXLIMITS_H
#define SAETTA_CTXLIMITS_H

// Default capacity, the multiple a capacity is rounded down to (session_grow wants a multiple
// of the indexer block, 4; DeepSeek-V4-Flash's compressed rows come every 128 tokens), and the
// smallest one accepted.
enum { SESSION_CTX_DEFAULT = 32768, SESSION_CTX_ALIGN = 128, SESSION_CTX_MIN = 1024 };
// max_position_embeddings of both Qwen models (config_flash.h, config_27b.h check it): the
// largest capacity `saetta start` accepts; the sessions use the loaded model's own value.
#define SESSION_CTX_QWEN_MAX ((size_t)262144)
// Inkling 2 Mini's model_max_length (config_inkling.h): its sessions' and `saetta start`'s limit
// for an Inkling pack.
#define SESSION_CTX_INKLING_MAX ((size_t)1048576)
// Bound on a request's max_tokens (any context is smaller).
#define SESSION_CTX_MAX ((size_t)1 << 30)

#endif
