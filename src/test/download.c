// Tests for the LoRA adapter helpers of src/download.c: --lora-modules parsing (saetta serve,
// saetta code, saetta start), adapter folders, the adapter_config.json checks, and the adapters
// /model finds under the models folder.
#include "../download.h"
#include "../test.h"

#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static char *tmp_models(Arena *a) {
    const char *t = getenv("TMPDIR");
    char *d = arena_printf(a, "%s/saetta-lora-XXXXXX", t && *t ? t : "/tmp");
    return mkdtemp(d);
}

static void put(Arena *a, const char *dir, const char *text) {
    mkdir_p(dir);
    (void)write_file(path_join(a, dir, "adapter_config.json"), text, strlen(text));
}

TEST(lora_args_vllm_forms) {
    Arena a = {0};
    DownloadLoras l = {0};
    // vLLM's: values until the next option, each name=path or a comma list
    const char *args[] = {"serve", "--lora-modules", "a=x", "b=y,c=/p/z", "--port", "1"};
    size_t i = 1;
    EXPECT_OK(download_lora_args(&a, args, COUNT(args), &i, &l));
    EXPECT_EQ(i, 3);
    EXPECT_EQ(l.n, 3);
    EXPECT(!strcmp(l.v[0], "a=x") && !strcmp(l.v[1], "b=y") && !strcmp(l.v[2], "c=/p/z"));
    // --lora adds one more (a list too); more than the first capacity
    EXPECT_OK(download_lora_add(&a, "d=27b-pi,e=org/repo", &l));
    EXPECT_EQ(l.n, 5);
    EXPECT(!strcmp(l.v[4], "e=org/repo"));
    // the original argument is left as it was
    EXPECT(!strcmp(args[3], "b=y,c=/p/z"));
    // errors: no value, not name=path, an empty part
    const char *none[] = {"--lora-modules", "--port"};
    i = 0;
    EXPECT_ERR(download_lora_args(&a, none, COUNT(none), &i, &l), "NoLoraModules");
    const char *bad[] = {"--lora-modules", "x"};
    i = 0;
    EXPECT_ERR(download_lora_args(&a, bad, COUNT(bad), &i, &l), "BadLoraModule");
    EXPECT_ERR(download_lora_add(&a, "=p", &l), "BadLoraModule");
    EXPECT_ERR(download_lora_add(&a, "n=", &l), "BadLoraModule");
    EXPECT_ERR(download_lora_add(&a, ",", &l), "BadLoraModule");
    arena_reset(&a);
}

TEST(lora_dir_name_base) {
    Arena a = {0};
    char *md = tmp_models(&a);
    EXPECT(md);
    const char *old = getenv("SAETTA_MODELS");
    char *keep = old ? strdup(old) : NULL;
    setenv("SAETTA_MODELS", md, 1);
    char *d;
    // a download name, a Hugging Face id, a folder as given
    EXPECT_OK(download_lora_dir(&a, "27b-pi", &d));
    EXPECT(!strcmp(d, path_join(&a, md, "harness-hub/Qwen3.8-27B-pi-adapted")));
    EXPECT_OK(download_lora_dir(&a, "org/Repo", &d));
    EXPECT(!strcmp(d, path_join(&a, md, "org/Repo")));
    EXPECT_OK(download_lora_dir(&a, md, &d));
    EXPECT(!strcmp(d, md));
    EXPECT(!strcmp(download_lora_name(&a, "27b-pi"), "qwen3.8-pi"));
    EXPECT(!strcmp(download_lora_name(&a, "/x/My-Adapter/"), "my-adapter"));
    // the base model it names, against a model id
    EXPECT(download_lora_base_fits("Qwen/Qwen3.8-27B", "qwen3.8-27b"));
    EXPECT(download_lora_base_fits("unsloth/Qwen3.8-27B-Instruct", "qwen3.8-27b"));
    EXPECT(!download_lora_base_fits("Qwen/Qwen3.8-27B", "qwen3.8-flash-next"));
    EXPECT(!download_lora_base_fits("Qwen/Qwen3.8-27B", NULL));
    EXPECT(download_lora_base_fits(NULL, "qwen3.8-27b"));
    if (keep) setenv("SAETTA_MODELS", keep, 1);
    else unsetenv("SAETTA_MODELS");
    free(keep);
    arena_reset(&a);
}

TEST(lora_check_config) {
    Arena a = {0};
    char *md = tmp_models(&a);
    EXPECT(md);
    char *d = path_join(&a, md, "x");
    EXPECT(download_lora_check(&a, d, "qwen3.8-27b") != NULL); // no adapter_config.json
    put(&a, d, "{\"base_model_name_or_path\": \"Qwen/Qwen3.8-27B\", \"peft_type\": \"LORA\", \"r\": 32, \"lora_alpha\": 64, "
               "\"target_modules\": [\"q_proj\", \"model.layers.0.mlp.down_proj\", \"gate_proj\"], \"bias\": \"none\", "
               "\"rank_pattern\": {}, \"use_dora\": false}");
    EXPECT(download_lora_check(&a, d, "qwen3.8-27b") == NULL);
    EXPECT(!strcmp(download_lora_base(&a, d), "Qwen/Qwen3.8-27B"));
    const char *why = download_lora_check(&a, d, "qwen3.8-flash-next");
    EXPECT(why && strstr(why, "trained on Qwen/Qwen3.8-27B"));
    static const struct {
        const char *cfg, *why;
    } bad[] = {
        {"{\"r\": 12, \"lora_alpha\": 16}", "rank 12"},
        {"{\"r\": 256, \"lora_alpha\": 16}", "rank 256"},
        {"{\"lora_alpha\": 16}", "no rank"},
        {"{\"r\": 8, \"lora_alpha\": 16, \"use_dora\": true}", "DoRA"},
        {"{\"r\": 8, \"lora_alpha\": 16, \"bias\": \"all\"}", "bias"},
        {"{\"r\": 8, \"lora_alpha\": 16, \"modules_to_save\": [\"lm_head\"]}", "modules_to_save"},
        {"{\"r\": 8, \"lora_alpha\": 16, \"rank_pattern\": {\"q_proj\": 16}}", "rank_pattern"},
        {"{\"r\": 8, \"lora_alpha\": 16, \"target_modules\": [\"q_proj\", \"embed_tokens\"]}", "embed_tokens"},
        {"{\"r\": 8, \"lora_alpha\": 16, \"peft_type\": \"IA3\"}", "not a LoRA"},
        {"{\"r\": 8,", "does not parse"},
    };
    for (size_t i = 0; i < COUNT(bad); i++) {
        put(&a, d, bad[i].cfg);
        why = download_lora_check(&a, d, "qwen3.8-27b");
        if (!why || !strstr(why, bad[i].why)) dbg("case %zu: %s\n", i, why ? why : "(fits)");
        EXPECT(why && strstr(why, bad[i].why));
    }
    // the load's errors, explained
    EXPECT(download_lora_error("LoraShapeMismatch") != NULL);
    EXPECT(download_lora_error("LoraUnsupported") != NULL);
    EXPECT(download_lora_error("NoModel") == NULL);
    arena_reset(&a);
}

TEST(lora_adapters_found) {
    Arena a = {0};
    char *md = tmp_models(&a);
    EXPECT(md);
    const char *old = getenv("SAETTA_MODELS");
    char *keep = old ? strdup(old) : NULL;
    setenv("SAETTA_MODELS", md, 1);
    // the download list's pi adapter (not downloaded), then two found ones in folder order; a
    // model pack (config.json) is not an adapter
    put(&a, path_join(&a, md, "zz/b-lora"), "{\"base_model_name_or_path\": \"Qwen/Qwen3.8-27B\", \"r\": 8, \"lora_alpha\": 8}");
    put(&a, path_join(&a, md, "aa/A-Lora"), "{\"r\": 8, \"lora_alpha\": 8}");
    mkdir_p(path_join(&a, md, "mlx-community/Qwen3.8-27B-8bit"));
    (void)write_file(path_join(&a, md, "mlx-community/Qwen3.8-27B-8bit/config.json"), "{}", 2);
    DownloadAdapter *ads;
    size_t n = download_adapters(&a, &ads);
    EXPECT_EQ(n, 3);
    EXPECT(ads[0].dl && !strcmp(ads[0].key, "27b-pi") && !ads[0].have && !strcmp(ads[0].name, "qwen3.8-pi"));
    EXPECT(!strcmp(ads[0].base, "Qwen/Qwen3.8-27B") && ads[0].disk_gb == 1);
    EXPECT(!ads[1].dl && ads[1].have && !strcmp(ads[1].name, "a-lora") && !strcmp(ads[1].title, "A-Lora") && !ads[1].base);
    EXPECT(!strcmp(ads[2].name, "b-lora") && !strcmp(ads[2].key, path_join(&a, md, "zz/b-lora")));
    // the pi adapter downloaded: found as the download, once
    put(&a, path_join(&a, md, "harness-hub/Qwen3.8-27B-pi-adapted"), "{\"base_model_name_or_path\": \"Qwen/Qwen3.8-27B\", \"r\": 32, \"lora_alpha\": 64}");
    n = download_adapters(&a, &ads);
    EXPECT_EQ(n, 3);
    EXPECT(ads[0].have && !strcmp(ads[0].key, "27b-pi"));
    if (keep) setenv("SAETTA_MODELS", keep, 1);
    else unsetenv("SAETTA_MODELS");
    free(keep);
    arena_reset(&a);
}
