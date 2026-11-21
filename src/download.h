// `saetta download [coder|full|all|27b-8bit|27b-4bit|27b-pi|27b-turbo|27b-turbo-8bit]`: fetches a
// model pack (or the pi LoRA adapter for Qwen3.8-27B) from Hugging Face at its pinned
// revision into $SAETTA_MODELS (default ~/Models) as <org>/<repo>, so other apps can share
// the files. Lists the revision's files through the Hub's tree API, downloads several at a
// time, resumes partial files (`<name>.part`, HTTP Range), and checks each file against the
// hash the Hub publishes (SHA-256 for LFS files, the git blob SHA-1 for the others) before
// renaming it into place. A file already present with the right size is kept.
// HF_TOKEN (optional, higher rate limits) from the environment or ~/.cache/huggingface/token.
#ifndef SAETTA_DOWNLOAD_H
#define SAETTA_DOWNLOAD_H

#include "base.h"

// A downloadable pack (download.Pack in the Zig engine; renamed so it does not clash with
// the weights Pack).
// A GGUF pack (gguf_file set): only the files in `only` come from repo (the model's own
// config and tokenizer), then gguf_file from gguf_repo at gguf_rev, which `saetta download`
// imports (gguf.h) into `dir` and deletes.
typedef struct {
    const char *name;
    const char *repo;
    const char *rev;
    const char *note;
    const char *dir; // local folder under the models folder (default: repo)
    const char *const *only;
    const char *gguf_repo, *gguf_rev, *gguf_file;
    unsigned min_ram_gb; // the smallest Mac memory it runs on (fit.h: weights + FIT_RESERVE_GB), 0 for adapters
} DownloadPack;

#define DOWNLOAD_NPACKS 7
// The first DOWNLOAD_NFLASH are the Flash-Next packs ("all", and saetta start's default).
#define DOWNLOAD_NFLASH 2
extern const DownloadPack download_packs[DOWNLOAD_NPACKS]; // coder, full, 27b-8bit, 27b-4bit, 27b-pi, 27b-turbo, 27b-turbo-8bit

// The pack named `name`, or NULL.
const DownloadPack *download_find(const char *name);
// Models folder: $SAETTA_MODELS, else ~/Models (error "NoHome").
Err download_models_dir(Arena *a, char **out);
// Local folder of a pack.
Err download_pack_dir(Arena *a, const DownloadPack *p, char **out);
const char *download_usage(void);
// args: the arguments after "download". Prints progress on stderr. Errors: "UnknownPack",
// "DownloadFailed", "ListFailed", "BadListing", parse errors of --jobs, HTTP transport
// errors (their message).
Err download_run(const char *const *args, size_t n);

#endif
