// Tests for src/code_cfg.c: settings.json and models.json changed in place (what launch.c
// pi_setup and /model write), pi's resolveConfigValue, the footer's pack names.
#include "../code_cfg.h"
#include "../test.h"

#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

static char *tmp_dir(Arena *a) {
    const char *t = getenv("TMPDIR");
    char *d = arena_printf(a, "%s/saetta-cfg-XXXXXX", t && *t ? t : "/tmp");
    return mkdtemp(d);
}

static Str slurp(Arena *a, const char *path) {
    char *text;
    size_t len;
    if (read_file(path, 1 << 20, &text, &len)) return S("");
    char *c = arena_strndup(a, text, len);
    free(text);
    return str_n(c, len);
}

static void put(const char *path, const char *text) { (void)write_file(path, text, strlen(text)); }

TEST(code_cfg_settings_keep_other_keys) {
    Arena a = {0};
    char *d = tmp_dir(&a);
    EXPECT(d);
    char *p = path_join(&a, d, "settings.json");
    put(p, "{\"theme\":\"saetta\",\"defaultProvider\":\"x\",\"lastChangelogVersion\":\"1.5.2\",\"enabledModels\":[\"a/b\"],\"n\":1.50}");
    Buf why = {0};
    EXPECT(cfg_set_default_model(d, "saetta", "qwen3.8-flash-next", &why));
    EXPECT_STR(slurp(&a, p), S("{\n  \"theme\": \"saetta\",\n  \"defaultProvider\": \"saetta\",\n  \"lastChangelogVersion\": \"1.5.2\",\n"
                               "  \"enabledModels\": [\n    \"a/b\"\n  ],\n  \"n\": 1.50,\n  \"defaultModel\": \"qwen3.8-flash-next\"\n}"));
    EXPECT(!file_exists(arena_printf(&a, "%s.lock", p)));
    // a file that does not parse is never overwritten
    put(p, "{\"theme\": \"saetta\",");
    EXPECT(!cfg_set_default_model(d, "saetta", "m", &why));
    EXPECT(why.len > 0);
    EXPECT_STR(slurp(&a, p), S("{\"theme\": \"saetta\","));
    // a missing file is created
    unlink(p);
    EXPECT(cfg_set_default_model(d, "saetta", "m", NULL));
    EXPECT_STR(slurp(&a, p), S("{\n  \"defaultProvider\": \"saetta\",\n  \"defaultModel\": \"m\"\n}"));
    buf_free(&why);
    unlink(p);
    rmdir(d);
    arena_reset(&a);
}

TEST(code_cfg_models_provider_only) {
    Arena a = {0};
    char *d = tmp_dir(&a);
    EXPECT(d);
    char *p = path_join(&a, d, "models.json");
    put(p, "{\"$schema\":\"s\",\"providers\":{\"ollama\":{\"baseUrl\":\"http://h/v1\",\"api\":\"openai-completions\",\"apiKey\":\"$K\","
           "\"models\":[{\"id\":\"q\",\"contextWindow\":32000}]},\"saetta\":{\"baseUrl\":\"old\"},\"z\":{\"baseUrl\":\"u\"}}}");
    JValue *e;
    EXPECT_OK(json_parse(&a, S("{\"baseUrl\":\"http://127.0.0.1:8080/v1\",\"models\":[{\"id\":\"m\",\"contextWindow\":204800}]}"), false, &e));
    EXPECT(cfg_put_provider(d, "saetta", NULL, e, NULL));
    JValue *m = cfg_read_models(&a, d);
    const JValue *pv = json_get(m, "providers");
    EXPECT(json_is(pv, J_OBJECT) && pv->obj.len == 3);
    EXPECT(str_eqz(pv->obj.items[0].key, "ollama") && str_eqz(pv->obj.items[1].key, "saetta") && str_eqz(pv->obj.items[2].key, "z"));
    EXPECT_STR(json_strs(json_get(json_get(pv, "saetta"), "baseUrl")), S("http://127.0.0.1:8080/v1"));
    EXPECT_STR(json_strs(json_get(json_get(pv, "ollama"), "apiKey")), S("$K"));
    EXPECT_STR(json_strs(json_get(m, "$schema")), S("s"));
    // edit (rename, keep the kept model's fields, add one), then remove
    JValue *ed = cfg_openai_provider(&a, json_get(pv, "ollama"), " http://h2/v1 ", "", "q, r q");
    EXPECT(cfg_put_provider(d, "local", "ollama", ed, NULL));
    m = cfg_read_models(&a, d);
    pv = json_get(m, "providers");
    EXPECT(str_eqz(pv->obj.items[0].key, "local"));
    const JValue *l = json_get(pv, "local");
    EXPECT_STR(json_strs(json_get(l, "baseUrl")), S("http://h2/v1"));
    EXPECT_STR(json_strs(json_get(l, "api")), S("openai-completions"));
    EXPECT_STR(json_strs(json_get(l, "apiKey")), S("none"));
    const JValue *ms = json_get(l, "models");
    EXPECT(json_is(ms, J_ARRAY) && ms->arr.len == 2);
    EXPECT_STR(json_get(ms->arr.items[0], "contextWindow")->s, S("32000"));
    EXPECT_STR(json_strs(json_get(ms->arr.items[1], "name")), S("r"));
    EXPECT(cfg_put_provider(d, "local", NULL, NULL, NULL));
    m = cfg_read_models(&a, d);
    EXPECT(json_get(json_get(m, "providers"), "local") == NULL && json_get(m, "providers")->obj.len == 2);
    // not parsing: left alone
    put(p, "{oops");
    Buf why = {0};
    EXPECT(!cfg_put_provider(d, "saetta", NULL, e, &why));
    EXPECT_STR(slurp(&a, p), S("{oops"));
    buf_free(&why);
    unlink(p);
    rmdir(d);
    arena_reset(&a);
}

TEST(code_cfg_resolve_value) {
    Arena a = {0};
    setenv("SAETTA_CFG_T", "k1", 1);
    unsetenv("SAETTA_CFG_UNSET");
    EXPECT_STR(str(cfg_resolve_value(&a, "saetta")), S("saetta"));
    EXPECT_STR(str(cfg_resolve_value(&a, "$SAETTA_CFG_T")), S("k1"));
    EXPECT_STR(str(cfg_resolve_value(&a, "Bearer ${SAETTA_CFG_T}-x")), S("Bearer k1-x"));
    EXPECT_STR(str(cfg_resolve_value(&a, "a$$b$!c$")), S("a$b!c$"));
    EXPECT(cfg_resolve_value(&a, "$SAETTA_CFG_UNSET") == NULL);
    EXPECT_STR(str(cfg_resolve_value(&a, "!echo ' hi '")), S("hi"));
    EXPECT(cfg_resolve_value(&a, "!false") == NULL);
    arena_reset(&a);
}

TEST(code_cfg_pack_label) {
    Arena a = {0};
    char *d = tmp_dir(&a);
    EXPECT(d);
    put(path_join(&a, d, "config.json"),
        "{\"quantization\":{\"group_size\":64,\"bits\":4,\"mode\":\"affine\",\"a\":{\"bits\":8,\"group_size\":64},"
        "\"b\":{\"bits\":5,\"group_size\":32},\"c\":{\"bits\":6,\"group_size\":128},\"d\":{\"bits\":4,\"group_size\":64}}}");
    const char *full, *brief;
    cfg_pack_label(&a, "Qwen3.8-Flash-Next Coder", NULL, d, &full, &brief);
    EXPECT_STR(str(full), S("Qwen3.8-Flash-Next Coder 4-bit (mixed 4/5/6/8)"));
    EXPECT_STR(str(brief), S("Qwen3.8-Flash-Next Coder 4-bit"));
    cfg_pack_label(&a, "Qwen3.8-27B TURBO 8-bit", "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q8_0.gguf", d, &full, &brief);
    EXPECT_STR(str(full), S("Qwen3.8-27B TURBO Q8_0"));
    cfg_pack_label(&a, "Qwen3.8-27B Fable-5", "Qwen3.8-27B-Fable-5-Coding-Distilled.i1-Q4_K_M.gguf", d, &full, &brief);
    EXPECT_STR(str(full), S("Qwen3.8-27B Fable-5 Q4_K_M"));
    // a GGUF name without a type token: the pack's config
    cfg_pack_label(&a, "Qwen3.8-27B Coder390", "Qwen3.8-27B-Coder390-EfficientThink-Q4LynnStyle-Q8MTP.gguf", d, &full, &brief);
    EXPECT_STR(str(full), S("Qwen3.8-27B Coder390 4-bit (mixed 4/5/6/8)"));
    put(path_join(&a, d, "config.json"), "{\"quantization\":{\"group_size\":64,\"bits\":8,\"mode\":\"affine\"}}");
    cfg_pack_label(&a, "Qwen3.8-27B 8-bit", NULL, d, &full, &brief);
    EXPECT_STR(str(full), S("Qwen3.8-27B 8-bit"));
    EXPECT(full == brief);
    unlink(path_join(&a, d, "config.json"));
    cfg_pack_label(&a, "Qwen3.8-27B 8-bit", NULL, d, &full, &brief);
    EXPECT_STR(str(full), S("Qwen3.8-27B 8-bit"));
    cfg_pack_label(&a, "X", NULL, d, &full, &brief);
    EXPECT(full == NULL);
    rmdir(d);
    arena_reset(&a);
}
