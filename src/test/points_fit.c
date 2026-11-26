// Checkpoints of every conversation stay inside what the context fit reserves for them
// (session_points_reserve: PT_MAX for the active conversation, PT_HELD for each other one): an
// inactive conversation keeps its PT_HELD most recent ones. With a model: SAETTA_POINTS_FIT_PACK
// (the 27B 4-bit: its 157 MB checkpoints), as on a 32 GB Mac (SAETTA_RAM_GB=32), 4 conversations,
// each fed a prompt with many message starts.
#include "../model.h"
#include "../server.h"
#include "../test.h"

#include <stdlib.h>

static int sink_ok(void *ctx, int status, const void *p, size_t n) {
    (void)ctx, (void)status, (void)p, (void)n;
    return 0;
}

// a conversation of 18 user/assistant pairs of about 1,200 tokens each, distinct per seed
static char *long_body(uint32_t seed) {
    Buf b = {0};
    buf_adds(&b, "{\"messages\":[");
    uint64_t x = 0x9E3779B97F4A7C15ull ^ seed;
    static const char *const w[] = {"river", "stone", "lamp", "orbit", "cedar", "quartz", "harbor", "ember"};
    for (int m = 0; m < 18; m++) {
        buf_printf(&b, "%s{\"role\":\"%s\",\"content\":\"Part %u.%d.", m ? "," : "", m % 2 ? "assistant" : "user", seed, m);
        for (int i = 0; i < 220; i++) {
            x ^= x << 13, x ^= x >> 7, x ^= x << 17;
            buf_printf(&b, " %s%llu", w[x % 8], (unsigned long long)(x >> 52));
        }
        buf_adds(&b, "\"}");
    }
    buf_adds(&b, ",{\"role\":\"user\",\"content\":\"One word.\"}],\"max_tokens\":2,\"temperature\":0,"
                 "\"chat_template_kwargs\":{\"enable_thinking\":false}}");
    char *s = strdup(b.ptr);
    buf_free(&b);
    return s;
}

TEST(points_fit_four_conversations) {
    const char *pack = getenv("SAETTA_POINTS_FIT_PACK");
    if (!pack) SKIP("SAETTA_POINTS_FIT_PACK not set (loads a model)");
    setenv("SAETTA_RAM_GB", "32", 1);
    setenv("SAETTA_CONVERSATIONS", "4", 1);
    setenv("SAETTA_CTX", "65536", 1);
    Arena a = {0};
    Engine *e;
    EXPECT_OK(server_engine_open(&a, pack, NULL, 0, NULL, &e));
    uint64_t reserved = 0;
    size_t act = 0, most = 0;
    server_engine_point_slots(e, &act, &reserved);
    // the 27B: 157 MB per checkpoint, 16 + 3 x 3 reserved
    dbg("points_fit: reserved %.2f GB, context %zu\n", (double)reserved / 1e9, server_engine_context(e));
    for (uint32_t c = 0; c < 4; c++) {
        char *b = long_body(100 + c);
        EXPECT_OK(server_engine_chat(e, str(b), sink_ok, NULL));
        free(b);
        size_t all = server_engine_point_slots(e, &act, NULL);
        dbg("points_fit: after conversation %u: %zu checkpoint slots (%zu active)\n", c + 1, all, act);
        EXPECT(act > PT_HELD); // the prompt left several
        EXPECT(all <= PT_MAX + (size_t)c * PT_HELD);
        most = all > most ? all : most;
    }
    // and back to the first: its kept checkpoints, the others trimmed to PT_HELD
    char *b = long_body(100);
    EXPECT_OK(server_engine_chat(e, str(b), sink_ok, NULL));
    free(b);
    size_t all = server_engine_point_slots(e, &act, NULL);
    EXPECT(all <= PT_MAX + 3 * PT_HELD);
    EXPECT((uint64_t)most * (reserved / (PT_MAX + 3 * PT_HELD)) <= reserved);
    unsetenv("SAETTA_RAM_GB");
}
