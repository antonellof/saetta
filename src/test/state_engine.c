// The conversation state store through the in-process engine (src/server.c), with a model: two
// threads, two conversations. A request that restores a saved state into the conversation
// another thread's last request used, and then fails before it runs (its stream refused), leaves
// that thread's end-of-turn save with nothing to save (state_restore counts the conversation as
// used); and the restore sets only its own conversation's speculation controller. Needs
// SAETTA_STATE_TEST_PACK (a Qwen3.8 pack with an MTP head, so speculation learns: the Coder; loads
// it, runs prompts on the GPU).
#include "../server.h"
#include "../test.h"

#include <dirent.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

// A user message of about `words` words, distinct per seed from its first word.
static char *body_for(uint32_t seed, size_t words, size_t max_tokens, bool stream) {
    Buf b = {0};
    buf_adds(&b, "{\"messages\":[{\"role\":\"user\",\"content\":\"");
    uint64_t x = 0x9E3779B97F4A7C15ull ^ seed;
    static const char *const w[] = {"river", "stone", "lamp", "orbit", "cedar", "quartz", "harbor", "ember", "violet", "anchor",
                                    "meadow", "signal", "copper", "lantern", "glacier", "thimble"};
    buf_printf(&b, "Topic %u.", seed);
    for (size_t i = 0; i < words; i++) {
        x ^= x << 13, x ^= x >> 7, x ^= x << 17;
        buf_printf(&b, " %s%llu", w[x % 16], (unsigned long long)(x >> 50));
    }
    buf_printf(&b, "\"}],\"max_tokens\":%zu,\"temperature\":0,\"stream\":%s}", max_tokens, stream ? "true" : "false");
    char *s = strdup(b.ptr);
    buf_free(&b);
    return s;
}

static int sink_ok(void *ctx, int status, const void *p, size_t n) {
    (void)ctx, (void)status, (void)p, (void)n;
    return 0;
}
// refuses every write: the stream fails at its first event, before the request runs
static int sink_refuse(void *ctx, int status, const void *p, size_t n) {
    (void)ctx, (void)status;
    return p != NULL && n > 0;
}

static size_t count_states(const char *dir) {
    size_t k = 0;
    DIR *d = opendir(dir);
    struct dirent *de;
    while (d && (de = readdir(d))) {
        if (de->d_name[0] == '.') continue;
        char p[1024];
        snprintf(p, sizeof p, "%s/%s", dir, de->d_name);
        struct stat sb;
        if (stat(p, &sb)) continue;
        if (S_ISDIR(sb.st_mode)) k += count_states(p);
        else if (strlen(de->d_name) == 35 && !strcmp(de->d_name + 32, ".st")) k++;
    }
    if (d) closedir(d);
    return k;
}

typedef struct {
    Engine *e;
    const char *body;
    int (*sink)(void *, int, const void *, size_t);
} Job;

static void *run_job(void *arg) {
    Job *j = arg;
    (void)server_engine_chat(j->e, str(j->body), j->sink, NULL);
    return NULL;
}

// on another thread (its own last request), waited for
static void on_thread(Engine *e, const char *body, int (*sink)(void *, int, const void *, size_t)) {
    Job j = {e, body, sink};
    pthread_t t;
    pthread_create(&t, NULL, run_job, &j);
    pthread_join(t, NULL);
}

TEST(state_engine_restore_into_another_threads_conversation) {
    const char *pack = getenv("SAETTA_STATE_TEST_PACK");
    if (!pack) SKIP("SAETTA_STATE_TEST_PACK not set (loads a model)");
    setenv("SAETTA_CONVERSATIONS", "2", 1);
    setenv("SAETTA_CTX", "32768", 1);
    char root[] = "/tmp/saetta-state-engine-XXXXXX";
    EXPECT(mkdtemp(root) != NULL);
    Arena a = {0};
    Engine *e;
    EXPECT_OK(server_engine_open(&a, pack, NULL, 0, NULL, &e));
    EXPECT_OK(server_engine_state_enable(e, root, "test"));
    // A: one token generated and none consumed, so its state is its prompt but the last token;
    // B: the same prompt again (it extends that state), streamed to a sink that refuses it
    char *A = body_for(1, 3000, 1, false), *B = body_for(1, 3000, 1, true);
    char *C = body_for(2, 600, 32, false), *E = body_for(3, 600, 1, false);
    // this thread: A (conversation 0), saved at its end
    EXPECT_OK(server_engine_chat(e, str(A), sink_ok, NULL));
    server_engine_state_save(e);
    server_engine_state_flush(e);
    EXPECT_EQ(count_states(root), 1);
    // C opens conversation 1 (32 tokens: its speculation learns), E takes conversation 0 (the
    // least recently used): this thread's last request is in conversation 0
    EXPECT_OK(server_engine_chat(e, str(C), sink_ok, NULL));
    EXPECT_OK(server_engine_chat(e, str(E), sink_ok, NULL));
    // another thread sends C again (conversation 1): conversation 0 is the least recently used
    on_thread(e, C, sink_ok);
    Gate g1, init_g = GATE_INIT;
    Depth d1, init_d = DEPTH_INIT;
    EXPECT(server_engine_controller(e, 1, &g1, &d1));
    EXPECT(memcmp(&g1, &init_g, sizeof g1) || memcmp(&d1, &init_d, sizeof d1)); // it learned
    // and then B: A's state is restored into conversation 0, and the request fails at its first
    // write, before it runs
    on_thread(e, B, sink_refuse);
    // the other conversation's speculation controller is untouched by the restore
    Gate h1;
    Depth k1;
    EXPECT(server_engine_controller(e, 1, &h1, &k1));
    EXPECT(!memcmp(&g1, &h1, sizeof g1) && !memcmp(&d1, &k1, sizeof d1));
    // this thread's save: conversation 0 is no longer E's (it holds A's restored rows): nothing
    server_engine_state_save(e);
    server_engine_state_flush(e);
    EXPECT_EQ(count_states(root), 1);
    free(A), free(B), free(C), free(E);
    char cmd[200];
    snprintf(cmd, sizeof cmd, "rm -rf '%s'", root);
    (void)system(cmd);
}
