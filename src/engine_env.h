// Every environment switch the engine reads (src/*.c but Saetta Code's own code_*.c, and the
// Metal shim), in one of two lists: the ones that can change the numbers the engine computes
// into a conversation (a kernel, a precision, prompt chunk boundaries, speculation, which tokens
// are consumed), whose values the state store hashes into a saved state's shape (server.c), so a
// state is only read back by an engine computing the same way; and the ones that cannot (paths,
// ports, capacity, logging, benchmarks and checks of their own). src/test/engine_env.c fails
// when a switch read in those files is in neither list.
#ifndef SAETTA_ENGINE_ENV_H
#define SAETTA_ENGINE_ENV_H

static const char *const engine_env_rows[] = {
    "SAETTA_ACCEL",          "SAETTA_NO_ACCEL",       "SAETTA_ATTN_MULTI",     "SAETTA_ATTN_PREP",
    "SAETTA_ATTN_SG_OLD",    "SAETTA_ATTN27_SIMD",    "SAETTA_BENCH_GREEDY",   "SAETTA_DENSE_FUSED",
    "SAETTA_EXPERT_CACHE_GB", "SAETTA_FOLD_SHARED",   "SAETTA_GDN_MULTI",      "SAETTA_INK_PCH",
    "SAETTA_IX_OLD",         "SAETTA_KV_F32",         "SAETTA_METAL_MATH",     "SAETTA_METAL_MAX_THREADS_CAP",
    "SAETTA_MTP",            "SAETTA_MV_ROWS",        "SAETTA_POINTS",         "SAETTA_PREFILL_F32",
    "SAETTA_PREP_VERIFY",    "SAETTA_QMMP",           "SAETTA_QSA_SIMD",       "SAETTA_SKIP",
    "SAETTA_SMALL_FUSED",    "SAETTA_SPEC",
};

// Read by the engine's files, and changing nothing a state holds.
static const char *const engine_env_other[] = {
    "HF_TOKEN",          "HOME",               "PATH",                "SAETTA_AUTOSTOP",     "SAETTA_CHUNK_COST",
    "SAETTA_CONVERSATIONS", "SAETTA_CTX",      "SAETTA_DSV4_FORCE",   "SAETTA_DUMP_LOGITS",  "SAETTA_IGNORE_MEMORY",
    "SAETTA_INK_ROUTES", "SAETTA_INPROC_LOCK_FD", "SAETTA_LORA",       "SAETTA_MODELS",       "SAETTA_MTP_K",
    "SAETTA_NLL",        "SAETTA_NLL_STEP",    "SAETTA_PACK",         "SAETTA_PORT",         "SAETTA_PROFILE",
    "SAETTA_QGEN_BENCH", "SAETTA_RAM_GB",      "SAETTA_SPIN",         "SAETTA_TIMING",       "SAETTA_VERIFY_DUMP",
    "SAETTA_VERIFY_FREE_RUN", "SAETTA_VERIFY_LAYERS",
};

#endif
