// `saetta download [coder|full|all|27b-8bit|27b-4bit|27b-pi|27b-turbo|27b-turbo-8bit|27b-coder390|...]`: fetches a
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
// imports (gguf.h) into `dir` and deletes. `subdir` / `gguf_subdir`: a folder of the repo that
// holds those files; they are stored at the pack folder's root.
typedef struct {
    const char *name;
    const char *repo;
    const char *rev;
    const char *note;
    const char *title; // the model's name as Saetta Code lists it
    const char *dir; // local folder under the models folder (default: repo)
    const char *const *only; // names relative to subdir
    const char *subdir;      // a folder of repo, or NULL for its root
    const char *gguf_repo, *gguf_rev, *gguf_file, *gguf_subdir;
    unsigned min_ram_gb; // the smallest Mac memory it runs on (fit.h: weights + FIT_RESERVE_GB), 0 for adapters
    unsigned disk_gb;    // free disk the download needs at its peak (a GGUF import holds the GGUF and the pack)
    // a LoRA adapter (min_ram_gb 0): the name it is served under, and the base model it was
    // trained on (as its adapter_config.json base_model_name_or_path names it)
    const char *lora_name, *lora_base;
} DownloadPack;

#define DOWNLOAD_NPACKS 12
// The first DOWNLOAD_NFLASH are the Flash-Next packs ("all"); saetta start's default is the
// first downloaded pack that fits this Mac (launch.c pack_for).
#define DOWNLOAD_NFLASH 2
extern const DownloadPack download_packs[DOWNLOAD_NPACKS]; // coder, full, 27b-8bit, 27b-4bit, 27b-pi, 27b-turbo, 27b-turbo-8bit,
// 27b-coder390, 27b-coder390-8bit, 27b-fable5, 27b-fable5-8bit, inkling

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

// ---- LoRA adapters: saetta serve --lora-modules, saetta code and saetta start --lora-modules,
// Saetta Code's /model --------------------------------------------------------------------------

// The "name=path" adapters given (--lora-modules, --lora), in order.
typedef struct {
    const char **v;
    size_t n, cap;
} DownloadLoras;
// vLLM's --lora-modules values: the arguments after args[*i] up to the next one that starts with
// '-', each name=path or a comma-separated list of them, appended to *out; *i ends on the last one
// taken. Errors (the argument printed): "BadLoraModule" (not name=path with both parts non-empty),
// "NoLoraModules" (no value followed).
Err download_lora_args(Arena *a, const char *const *args, size_t n, size_t *i, DownloadLoras *out);
// One value as --lora gives it (name=path, or a comma-separated list), appended to *out.
// Errors: "BadLoraModule" (printed).
Err download_lora_add(Arena *a, const char *spec, DownloadLoras *out);
// The folder of an adapter path as --lora-modules takes it: an existing folder as given, a
// download name (27b-pi), else a Hugging Face id (org/repo) under the models folder.
Err download_lora_dir(Arena *a, const char *path, char **out);
// The name a download name or folder is served under when none is given (/model): the
// download's lora_name, else the folder's last component in lower case.
const char *download_lora_name(Arena *a, const char *path);
// The base model an adapter folder names (adapter_config.json base_model_name_or_path), or NULL.
const char *download_lora_base(Arena *a, const char *dir);
// Whether base (a base_model_name_or_path; NULL: unknown, taken as fitting) names the model id
// (ModelInfo.id, such as qwen3.8-27b): its last path component, in lower case, contains the id.
bool download_lora_base_fits(const char *base, const char *model_id);
// Whether the adapter in dir fits the base model model_id (ModelInfo.id) as far as its
// adapter_config.json tells: NULL when it does, else why not (a sentence, without the adapter's
// name). Checks the base model it names, PEFT LoRA without DoRA, bias or saved modules, a rank
// that is a multiple of 8 up to 128 without rank or alpha patterns, and target modules among the
// attention and MLP projections. The tensors are checked when it loads (model_lora_load).
const char *download_lora_check(Arena *a, const char *dir, const char *model_id);
// The model of the pack in dir (model.h ModelInfo id and name), NULLs for an unknown one.
void download_model_of(Arena *a, const char *dir, const char **id, const char **name);
// What a failed model_lora_load means, a sentence (NULL for an error that is not the adapter's).
const char *download_lora_error(Err e);

// The LoRA adapters /model lists: the download list's, downloaded or not, then the other folders
// of the models folder (org/repo) that hold an adapter_config.json.
typedef struct {
    const char *key;   // what --lora-modules takes for it: the download name, else the folder
    const char *name;  // the name it is served under (download_lora_name)
    const char *title; // for /model: the download's title, else the folder's name
    const char *dir;
    const char *base;  // the base model it names, or NULL
    const DownloadPack *dl; // NULL: found in the models folder
    bool have;              // its adapter_config.json is there
    unsigned disk_gb;
} DownloadAdapter;
size_t download_adapters(Arena *a, DownloadAdapter **out);

#endif
