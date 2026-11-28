// HTTP server (see server.h).
#include "server.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

#include "anthropic.h"
#include "chat.h"
#include "download.h"
#include "gpu.h"
#include "fit.h"
#include "http.h"
#include "instance.h"
#include "json.h"
#include "metal/shim.h"
#include "model.h"
#include "spec.h"
#include "hash.h"
#include "state_store.h"
#include "engine_id.h"
#include "engine_env.h"
#include "split.h"
#include "grammar.h"
#include "stats.h"
#include "tokenizer.h"
#include "toolcall.h"
#include "turns.h"
#include "hash.h"
#include "weights.h"

// The served model id (the loaded model's, model_info) and the one the current request's
// response names: the base id, or the LoRA adapter's name the request selected (one request at
// a time).
static const char *base_id = "saetta";
static const char *reply_id = "saetta";

#define MAX_CONV 8
// Capacity a conversation starts with (tokens); it grows in steps of CAP_STEP as needed.
#define CAP0 ((size_t)32768)
#define CAP_STEP ((size_t)32768)
// Room kept for the speculative chunk and the last token past a reply.
#define CAP_SLACK ((size_t)(KMAX + 8))

// Zig's "{d:.N}" of an f64, in one of 8 rotating buffers.
static _Thread_local char fx_bufs[8][400];
static _Thread_local int fx_i;
static const char *fx(double x, int prec) {
    char *b = fx_bufs[fx_i++ & 7];
    fmt_fixed(b, 400, x, prec);
    return b;
}

// Speculation fed by the agent (server_engine_feed): per agent session, the texts it fed (a file's
// content, keyed by its path: fed again, the new text replaces the old), as the token ids of their
// JSON-escaped form (the form an edit's oldText and newText take in the model's tool call: an array
// argument is written as JSON, so a file's lines appear joined by \n, which the file's own tokens
// in a tool result never match), oldest first, and one lookup index over them in that order with a
// SPEC_SEP between texts. Bounded: FEED_TEXT_TOKENS per text (its head), FEED_TOKENS per session
// (the oldest texts go first), FEED_SESSIONS sessions (the least recently fed goes). The index is
// immutable once built: a feed rebuilds it under feed_mu (a few ms for a full session) and swaps it
// in; a request takes a reference at its start and drops it at its end (feed_mu at both, never per
// token), so feeding neither waits for the engine's turn nor stops a read ahead.
#define FEED_SESSIONS 4
#define FEED_TOKENS ((size_t)131072)
#define FEED_TEXT_TOKENS ((size_t)65536)
typedef struct {
    uint64_t key; // hash of the text's key (its path)
    VecU32 ids;
} FeedText;
typedef struct {
    Lookup lk;
    size_t refs; // under feed_mu: the feed's own and each request's
} FeedIx;
typedef struct {
    char session[96]; // "": a free slot
    uint64_t used;    // the feed clock at its last change
    VEC(FeedText) texts;
    size_t ntok;
    FeedIx *ix; // NULL: nothing fed
} Feed;

typedef struct Engine {
    Gpu *g;
    Model *m;
    Session *s;
    const Tokenizer *tok;
    QgenPrng rng;
    uint32_t think_end;
    // the id that marks message starts, where prompts get their checkpoints (session_point_cuts;
    // UINT32_MAX: none, for a chat format other than Qwen's)
    uint32_t im_start;
    // Speculation statistics kept across requests: a short reply then starts from what the
    // session has learned instead of the optimistic defaults (deep drafts with poor
    // acceptance cost most on short replies).
    // (one per conversation: a conversation's tokens depend only on its own requests, so one
    // restored from the state store, with its controller, continues as it would have; e.g. a
    // subagent's speculation does not change the main agent's numbers)
    Gate gate[MAX_CONV];
    Depth depth[MAX_CONV];
    Stats *st;
    // Conversations: the active one's buffers are in s; held[i] keeps conversation i's
    // while it is inactive (has[i] false for the active one). A request goes to the
    // conversation it continues (Conv in src/model.h).
    Conv held[MAX_CONV];
    bool has[MAX_CONV];
    uint64_t used[MAX_CONV];
    int64_t last_s[MAX_CONV];
    size_t n_conv;
    size_t active;
    uint64_t clock;
    size_t max_conv;
    // LoRA adapter (model_lora_find index, -1: none) whose state each conversation holds: its
    // KV cache, restore and prefix points depend on it, so reuse stays within one adapter
    int conv_lora[MAX_CONV];
    // --default-chat-template-kwargs: an object whose keys fill in what a request leaves out
    JValue *kwargs_default;
    uint64_t opened_ns; // when engine_open began (the load time the server reports)
    const char *dir;    // the pack
    // conversation state on the SSD (server_engine_state_enable; Saetta Code in process only)
    StateStore *store;
    // each conversation's lineage (src/state_store.h): the computation that writes its chunks, set
    // at its first save, never taken from a restored state; inh: the keys of the chunks it was
    // restored from (read, and listed by its states, below the restore point); lin_hw: the most
    // tokens saved or restored in it. A conversation that starts again from 0, from another's rows,
    // or below lin_hw gets a new lineage and keeps only the inherited chunks below its start.
    uint8_t lin[MAX_CONV][16];
    bool lin_set[MAX_CONV];
    size_t lin_hw[MAX_CONV];
    uint8_t (*inh[MAX_CONV])[32];
    size_t ninh[MAX_CONV];
    // a count of the requests each conversation ran (apply): an end-of-turn save checks that the
    // conversation its caller's last request used has not been used by another request since
    uint64_t conv_gen[MAX_CONV];
    // kept[i]: the last request conversation i ran asked for a fork point (saetta_fork_point: Saetta
    // Code's agent with subagents, whose next request continues it while its children run in other
    // conversations): route, evict_one and drop_held_points take another one while one is left
    bool kept[MAX_CONV];
    // One request at a time, in arrival order: the in-process caller (server_engine_chat, from
    // any thread: Saetta Code's agent and its subagents) and the connections of the Saetta Codes
    // that share this process's engine (server_engine_connection, one thread each) take tickets
    // (turn_take) and run when theirs is served (turn_wait); a waiter that leaves gives its turn
    // up (turn_leave). Only the engine work runs in a turn: requests are read before.
    Turns turns;
    // server_engine_set_token: the token every request of a shared connection must carry
    char token[80];
    // Prefill ahead (server_engine_ahead): the agent session of the last in-process request each
    // conversation ran, with its conv_gen then (an ahead read continues only that session's
    // conversation, and only while no other request has used it since).
    char owner[MAX_CONV][96];
    uint64_t owner_gen[MAX_CONV];
    // live: the session's live state is the one an ahead read left in conversation live_conv at
    // live_pos, over live_ids[0..live_pos), and nothing ran on the model since (every request's
    // apply and every switch of conversation clear it). A prompt that starts with those ids
    // continues there in place (START_LIVE): no restore, so the MTP head's row at live_pos (the
    // pair of the stream at live_pos - 1, kept by the model, and the prompt's token there) is
    // written by the next chunk as a prompt read without a stop writes it.
    bool live;
    size_t live_conv, live_pos;
    VecU32 live_ids;
    // speculation fed by the agent (Feed above), under feed_mu
    pthread_mutex_t feed_mu;
    Feed feed[FEED_SESSIONS];
    uint64_t feed_clock;
    size_t last_fed; // the last request's passes that drafted from fed texts (for tests)
    size_t last_replaced; // the last request's tokens the tool-call grammar replaced (for tests)
} Engine;

static uint64_t turn_take(Engine *e) { return turns_take(&e->turns); }
static bool turn_wait(Engine *e, uint64_t t, int ms) { return turns_wait(&e->turns, t, ms); }
static void turn_done(Engine *e) { turns_done(&e->turns); }
static void turn_leave(Engine *e, uint64_t t) { turns_leave(&e->turns, t); }

// ---- speculation fed by the agent (Feed; server_engine_feed) ------------------------------------

// SAETTA_FEED=0 turns it off (fed texts are dropped, nothing is drafted from them).
static bool feed_on(void) {
    const char *v = getenv("SAETTA_FEED");
    return !(v && v[0] == '0');
}

// The fed index of agent session `session` with a reference taken, or NULL (feed_unref drops it).
static FeedIx *feed_ref(Engine *e, const char *session) {
    FeedIx *ix = NULL;
    pthread_mutex_lock(&e->feed_mu);
    for (size_t i = 0; i < FEED_SESSIONS; i++)
        if (e->feed[i].session[0] && !strcmp(e->feed[i].session, session) && e->feed[i].ix) {
            ix = e->feed[i].ix;
            ix->refs++;
        }
    pthread_mutex_unlock(&e->feed_mu);
    return ix;
}

// Under feed_mu.
static void feed_unref_locked(FeedIx *ix) {
    if (!ix || --ix->refs > 0) return;
    lookup_free(&ix->lk);
    free(ix);
}

static void feed_unref(Engine *e, FeedIx *ix) {
    if (!ix) return;
    pthread_mutex_lock(&e->feed_mu);
    feed_unref_locked(ix);
    pthread_mutex_unlock(&e->feed_mu);
}

// Under feed_mu.
static void feed_release(Feed *f) {
    for (size_t i = 0; i < f->texts.len; i++) vec_free(&f->texts.ptr[i].ids);
    vec_free(&f->texts);
    feed_unref_locked(f->ix);
    *f = (Feed){0};
}

// A new index over f's texts, oldest first (a later occurrence of a trigram wins: the newest
// text), in place of the old one (under feed_mu; a request holding the old keeps it).
static void feed_index(Feed *f) {
    feed_unref_locked(f->ix);
    f->ix = NULL;
    if (f->ntok == 0) return;
    FeedIx *ix = xcalloc(1, sizeof *ix);
    ix->refs = 1;
    vec_reserve(&ix->lk.hist, f->ntok + f->texts.len);
    for (size_t i = 0; i < f->texts.len; i++) {
        const VecU32 *v = &f->texts.ptr[i].ids;
        for (size_t k = 0; k < v->len; k++) lookup_push(&ix->lk, v->ptr[k]);
        lookup_push_sep(&ix->lk);
    }
    f->ix = ix;
}

// Makes conversation i the active one.
// The conversation going inactive keeps PT_HELD checkpoints (inside the fit's reserve).
static void activate(Engine *e, size_t i) {
    if (i == e->active) return;
    e->live = false;
    session_swap_conv(e->s, &e->held[i]);
    e->held[e->active] = e->held[i];
    e->has[e->active] = true;
    (void)session_conv_trim_points(&e->held[e->active], PT_HELD);
    memset(&e->held[i], 0, sizeof(Conv));
    e->has[i] = false;
    e->active = i;
}

// When memory is short: the checkpoints of the least recently used inactive conversation that
// holds some go (its restore points stay). False when none holds any.
static bool drop_held_points(Engine *e) {
    bool found = false;
    size_t pick = 0;
    for (int pass = 0; pass < 2 && !found; pass++)
        for (size_t i = 0; i < e->n_conv; i++) {
            if (!e->has[i] || (pass == 0 && e->kept[i])) continue;
            bool any = false;
            for (size_t k = 0; k < PT_MAX; k++) any |= e->held[i].pts[k].gdn != NULL;
            if (any && (!found || e->used[i] < e->used[pick])) {
                pick = i;
                found = true;
            }
        }
    if (found) (void)session_conv_trim_points(&e->held[pick], 0);
    return found;
}

static const Ckpt *ck_of(Engine *e, size_t i) { return i == e->active ? e->s->ck : e->held[i].ck; }

static bool conv_empty(Engine *e, size_t i) {
    const Ckpt *ck = ck_of(e, i);
    return ck[0].tokens.len == 0 && ck[1].tokens.len == 0;
}

// GPU bytes free under the working-set limit.
static uint64_t free_bytes(Engine *e) {
    uint64_t ws = sx_device_recommended_working_set(e->g->dev), al = sx_device_allocated_size(e->g->dev);
    return ws > al ? ws - al : 0;
}

// Room for bytes more: under the GPU working-set limit with gpu_margin left, or in the
// machine's free RAM with 8 GB left (with the full pack the weights alone pass the
// working-set limit, while RAM has room; inactive pages are not counted, they hold the
// mapped weights).
static bool room_for(Engine *e, uint64_t bytes, uint64_t gpu_margin) {
    if (free_bytes(e) >= bytes + gpu_margin) return true;
    uint32_t free_pages = 0;
    uint64_t page = 16384;
    size_t len = sizeof(uint32_t);
    if (sysctlbyname("vm.page_free_count", &free_pages, &len, NULL, 0) != 0) return false;
    len = sizeof(uint64_t);
    (void)sysctlbyname("hw.pagesize", &page, &len, NULL, 0);
    return (uint64_t)free_pages * page >= bytes + (8ull << 30);
}

// The checkpoint a prompt of nids tokens may start from in a conversation (session_shared_point),
// with PT_TAIL tokens or more left before its last one: a shorter rest would be read on the
// decode-style kernels, not as the same tokens read inside a prompt chunk (a checkpoint can lie
// within PT_TAIL of a later prompt's end since the fork point, session_point_cuts_add).
static size_t start_point(const VecU32 *ptoks, const Point pts[PT_MAX], const uint32_t *ids, size_t nids) {
    return nids > PT_TAIL ? session_shared_point(ptoks->ptr, ptoks->len, pts, ids, nids - PT_TAIL) : 0;
}

// The conversation this thread's last request ran in (the in-process callers run their requests
// on their own threads: Saetta Code's agent, each subagent), for its end-of-turn save and for
// route.
typedef struct {
    const Engine *e;
    size_t conv;
    uint64_t gen;
    bool set;
    char session[96]; // the agent session of that request (server_engine_chat_session), "" none
} LastConv;
static _Thread_local LastConv last_conv;
// The agent session of the request this thread is running (server_engine_chat_session), "" none.
static _Thread_local char req_session[96];

// Picks the conversation for a prompt: the one it continues; else, for an in-process caller
// (inproc: server_engine_chat, where each caller runs on its own thread: Saetta Code's agent, each
// subagent), the one this thread's last request used (and no other request since) when one of its
// checkpoints starts the prompt (a retry, an edit, a /fork of the caller's own conversation
// restarts there without copying its rows; another caller's prompt that shares a prefix, e.g. a
// subagent's, gets its own conversation and copies them; over HTTP and shared connections one
// thread serves every client, so a thread is no session there and this never applies); else an
// empty one, a new one while GPU memory allows (4 GB kept free under the working-set limit), or
// the least recently used.
static void route(Engine *e, const uint32_t *ids, size_t nids, int lora, bool inproc) {
    e->clock += 1;
    size_t best = e->active, best_n = 0;
    for (size_t i = 0; i < e->n_conv; i++) {
        if (e->conv_lora[i] != lora) continue;
        size_t n = session_reusable_in(ck_of(e, i), ids, nids).n;
        if (n > best_n) {
            best_n = n;
            best = i;
        }
    }
    if (best_n == 0 && inproc && req_session[0] && !strcmp(req_session, last_conv.session) && last_conv.set && last_conv.e == e &&
        last_conv.conv < e->n_conv && e->conv_lora[last_conv.conv] == lora && e->conv_gen[last_conv.conv] == last_conv.gen) {
        size_t i = last_conv.conv;
        const Session *s = e->s;
        size_t p = i == e->active ? start_point(&s->ptoks, s->pts, ids, nids) : start_point(&e->held[i].ptoks, e->held[i].pts, ids, nids);
        if (p > 0) {
            best = i;
            goto picked;
        }
    }
    if (best_n == 0) {
        if (conv_empty(e, e->active)) {
            best = e->active;
            goto picked;
        }
        for (size_t i = 0; i < e->n_conv; i++)
            if (conv_empty(e, i)) {
                best = i;
                goto picked;
            }
        if (e->n_conv < e->max_conv) {
            if (room_for(e, session_conv_bytes(e->s, MIN(CAP0, e->s->max_cap)), 4ull << 30)) {
                Conv cv;
                if (!session_new_conv(e->s, e->g, MIN(CAP0, e->s->max_cap), &cv)) {
                    e->held[e->n_conv] = cv;
                    e->has[e->n_conv] = true;
                    e->conv_lora[e->n_conv] = lora;
                    e->n_conv += 1;
                    dbg("saetta: conversation %zu opened (%s GB)\n", e->n_conv,
                        fx((double)session_conv_bytes(e->s, MIN(CAP0, e->s->max_cap)) / (double)(1 << 30), 1));
                    best = e->n_conv - 1;
                    goto picked;
                }
            }
        }
        // the least recently used, a kept one (a parent's, while its subagents run) only when all are
        size_t lru = SIZE_MAX;
        for (int pass = 0; pass < 2 && lru == SIZE_MAX; pass++)
            for (size_t i = 0; i < e->n_conv; i++)
                if (!(pass == 0 && e->kept[i]) && (lru == SIZE_MAX || e->used[i] < e->used[lru])) lru = i;
        best = lru;
    }
picked:
    activate(e, best);
    if (e->conv_lora[best] != lora) {
        // its state was built with another adapter: start it over
        e->live = false;
        session_invalidate_from(e->s, 0);
        e->conv_lora[best] = lora;
    }
    model_lora_select(e->m, lora);
    e->used[best] = e->clock;
    e->last_s[best] = (int64_t)(wall_ns() / 1000000000ull);
    atomic_store_explicit(&e->st->convs, (uint32_t)e->n_conv, memory_order_release);
}

typedef enum { START_FRESH, START_CKPT, START_OWN, START_SHARED, START_LIVE } StartKind;
typedef struct {
    size_t n;
    StartKind kind;
    size_t src, common;
} Start;

// Whether a prompt of nids tokens continues the live state an ahead read left in the active
// conversation (Engine.live): it starts with the ids read, and more than MODEL_SMALL + 1 tokens are
// left before its last one (a shorter rest would run on the small-chunk kernels, whose rounding
// differs from a prompt chunk's, and so would the MTP head's rows of a chunk that skips its first
// row: a prompt read without the stop reads those tokens in a prompt chunk).
static bool live_continues(const Engine *e, const uint32_t *ids, size_t nids) {
    size_t p = e->live_pos;
    return e->live && e->live_conv == e->active && e->s->pos == p && p > 0 && nids >= p + MODEL_SMALL + 3 &&
           e->live_ids.len >= p && memcmp(ids, e->live_ids.ptr, p * sizeof(uint32_t)) == 0;
}

// Where the active conversation starts reading ids: the longest of the live state of an ahead read
// it continues, a restore point it extends, its own checkpoints and another conversation's
// checkpoint (KV rows copied), preferring them in that order when they tie; else 0. (Another
// conversation's longer checkpoint wins over an own one: e.g. a subagent's conversation, taken
// again for the parent's next batch of subagents, holds its first task's start, while the parent's
// conversation holds where the new tasks leave it.) share false: own starts only (an ahead read,
// which continues this conversation in place).
static Start plan(Engine *e, const uint32_t *ids, size_t nids, bool share) {
    size_t r = session_reusable(e->s, ids, nids);
    size_t own = start_point(&e->s->ptoks, e->s->pts, ids, nids);
    Start best = {0, START_FRESH, 0, 0};
    for (size_t i = 0; share && i < e->n_conv; i++) {
        if (!e->has[i] || e->conv_lora[i] != e->conv_lora[e->active]) continue;
        const Conv *cv = &e->held[i];
        size_t n = start_point(&cv->ptoks, cv->pts, ids, nids);
        if (n > best.n) {
            size_t lcp = 0;
            while (lcp < MIN(cv->ptoks.len, nids) && cv->ptoks.ptr[lcp] == ids[lcp]) lcp++;
            best = (Start){n, START_SHARED, i, lcp};
        }
    }
    if (live_continues(e, ids, nids) && e->live_pos > MAX(r, own) && e->live_pos >= best.n) return (Start){e->live_pos, START_LIVE, 0, 0};
    if (r > 0 && r >= own && r >= best.n) return (Start){r, START_CKPT, 0, 0};
    if (own > 0 && own >= best.n) return (Start){own, START_OWN, 0, 0};
    return best;
}

// How many more checkpoints' buffers memory allows now (point_room's rule), at most PT_MAX.
static size_t point_budget(Engine *e) {
    size_t k = 0, pb = session_point_bytes(e->s);
    while (k < PT_MAX && room_for(e, (uint64_t)(k + 1) * pb, 2ull << 30)) k++;
    return k;
}

// request: a request's start (counted in conv_gen); false for an ahead read, which continues the
// same session's conversation on its behalf.
static Err apply(Engine *e, Start st, bool request) {
    // rows below st.n stay (from 0 when another conversation's rows are copied in): a background
    // save that still reads this buffer past that point finishes first
    if (e->store) state_store_fence(e->store, sx_buffer_contents(e->s->kv), st.kind == START_SHARED ? 0 : st.n);
    // the live state is this start's (START_LIVE) or replaced now
    e->live = false;
    size_t ci = e->active;
    if (request) e->conv_gen[ci] += 1;
    if (st.kind == START_FRESH || st.kind == START_SHARED) {
        e->ninh[ci] = 0;
        // a conversation that starts over (or from another's rows) learns its own speculation
        e->gate[ci] = GATE_INIT;
        e->depth[ci] = DEPTH_INIT;
    }
    if (st.kind == START_FRESH || st.kind == START_SHARED || st.n < e->lin_hw[ci]) {
        e->lin_set[ci] = false;
        e->lin_hw[ci] = 0;
        e->ninh[ci] = MIN(e->ninh[ci], st.n / ST_CHUNK);
    }
    switch (st.kind) {
    case START_CKPT: session_restore(e->s); break;
    case START_OWN: session_start_at_own_point(e->s, st.n); break;
    case START_SHARED:
        TRY(session_start_from(e->s, e->g, &e->held[st.src], st.n, point_budget(e)));
        dbg("saetta: conversation %zu starts from the first %zu tokens of conversation %zu (%zu in common)\n",
            e->active + 1, st.n, st.src + 1, st.common);
        break;
    case START_FRESH: session_reset(e->s); break;
    case START_LIVE: break; // in place: the live state is at st.n
    }
    session_invalidate_from(e->s, st.n);
    return OK;
}

// Empties the least recently used inactive conversation that holds more than the starting
// capacity (its buffers shrink back to CAP0, its restore and prefix points are dropped).
static bool evict_one(Engine *e) {
    bool found = false;
    size_t pick = 0;
    for (int pass = 0; pass < 2 && !found; pass++)
        for (size_t i = 0; i < e->n_conv; i++) {
            if (!e->has[i] || (pass == 0 && e->kept[i])) continue;
            if (e->held[i].cap > MIN(CAP0, e->s->max_cap) && (!found || e->used[i] < e->used[pick])) {
                pick = i;
                found = true;
            }
        }
    if (!found) return false;
    size_t i = pick;
    Conv *cv = &e->held[i];
    if (session_conv_shrink(e->s, e->g, cv, MIN(CAP0, e->s->max_cap))) return false;
    dbg("saetta: conversation %zu emptied to make room\n", i + 1);
    return true;
}

// Grows the active conversation to hold need tokens (in steps of CAP_STEP, at most
// SAETTA_CTX), emptying other conversations first when GPU memory is short. The active
// conversation always gets its SAETTA_CTX, as when it was preallocated: the working-set
// limit only decides whether the others keep their memory (with the full pack, the
// weights alone already pass it).
static Err ensure(Engine *e, size_t need) {
    if (need <= e->s->cap) return OK;
    size_t cap = MIN(e->s->max_cap, ALIGN_UP(need, CAP_STEP));
    // inactive conversations' checkpoints go first, then whole conversations
    while (!room_for(e, session_kv_bytes(e->s, cap), 2ull << 30))
        if (!drop_held_points(e) && !evict_one(e)) break;
    return session_grow(e->s, e->g, cap);
}

// The conversations' sizes for the monitor.
static void publish_convs(Engine *e) {
    ConvInfo info[MAX_CONV];
    for (size_t i = 0; i < e->n_conv; i++) {
        const Ckpt *ck = ck_of(e, i);
        info[i] = (ConvInfo){
            .tokens = (uint32_t)MAX(ck[0].tokens.len, ck[1].tokens.len),
            .cap = (uint32_t)(i == e->active ? e->s->cap : e->held[i].cap),
            .last_s = e->last_s[i],
            .active = i == e->active,
        };
    }
    stats_set_convs(e->st, info, e->n_conv);
}

// Room for one more checkpoint's buffers (2 GB kept free under the GPU working-set limit), after
// dropping inactive conversations' checkpoints when short.
static bool point_room(Engine *e) {
    while (!room_for(e, session_point_bytes(e->s), 2ull << 30))
        if (!drop_held_points(e)) return false;
    return true;
}

static Err json_respond(Conn *c, Req *h, int status, Str body) {
    return http_respond(c, h, status, body, "application/json");
}

// ---- request fields ----------------------------------------------------------------------------

// A number field (any JSON number spelling), or false when absent or not a number.
static bool num(const JValue *v, double *out) {
    if (!v) return false;
    switch (v->kind) {
    case J_INTEGER: *out = (double)v->i; return true;
    case J_FLOAT: *out = v->f; return true;
    case J_NUMBER_STRING: {
        // the parser NUL terminates number literals in the arena, so strtod reads in place
        // (any length: a long literal is still a number, as in std.json)
        char *end;
        double d = strtod(v->s.ptr, &end);
        if (end == v->s.ptr || end != v->s.ptr + v->s.len) return false;
        *out = d;
        return true;
    }
    default: return false;
    }
}

static JValue *field(const JValue *req, const char *key) {
    JValue *v = json_get(req, key);
    return (v && v->kind == J_NULL) ? NULL : v;
}

// std.json ObjectMap.orderedRemove.
static void json_remove(JValue *obj, const char *key) {
    if (!obj || obj->kind != J_OBJECT) return;
    for (size_t i = 0; i < obj->obj.len; i++)
        if (str_eqz(obj->obj.items[i].key, key)) {
            memmove(obj->obj.items + i, obj->obj.items + i + 1, (obj->obj.len - i - 1) * sizeof(JMember));
            obj->obj.len -= 1;
            return;
        }
}

// ---- sampling ---------------------------------------------------------------------------------

size_t server_top_k(const float *logits, size_t len, size_t k, uint32_t *cand) {
    // Groups of 16 whose maximum cannot enter the list once it is full are skipped (most of
    // the vocabulary): the same comparisons in the same order, so the same result as the
    // plain loop.
    enum { L = 16 };
    size_t n = 0, i = 0;
    while (i < len) {
        if (n == k && i + L <= len) {
            float g = logits[i];
            for (size_t j = 1; j < L; j++) g = logits[i + j] > g ? logits[i + j] : g;
            if (g <= logits[cand[k - 1]]) {
                i += L;
                continue;
            }
        }
        size_t end = MIN(len, i + L);
        for (; i < end; i++) {
            float v = logits[i];
            if (n < k) {
                cand[n] = (uint32_t)i;
                n += 1;
                size_t j = n - 1;
                for (; j > 0 && logits[cand[j]] > logits[cand[j - 1]]; j--) {
                    uint32_t t = cand[j];
                    cand[j] = cand[j - 1];
                    cand[j - 1] = t;
                }
            } else if (v > logits[cand[k - 1]]) {
                cand[k - 1] = (uint32_t)i;
                size_t j = k - 1;
                for (; j > 0 && logits[cand[j]] > logits[cand[j - 1]]; j--) {
                    uint32_t t = cand[j];
                    cand[j] = cand[j - 1];
                    cand[j - 1] = t;
                }
            }
        }
    }
    return n;
}

// The sampling distribution of logits: top-k by partial selection into cand (k is small),
// softmax at the temperature, then top-p. Fills cand[0..n] and their weights p[0..n]
// (unnormalized) and returns n.
static size_t dist(const float *logits, size_t len, Sampling sp, uint32_t *cand, float p[256]) {
    size_t k = MIN(sp.top_k, (size_t)256);
    size_t n = server_top_k(logits, len, k, cand);
    float mx = logits[cand[0]];
    float sum = 0;
    for (size_t i = 0; i < n; i++) {
        p[i] = expf((logits[cand[i]] - mx) / sp.temperature);
        sum += p[i];
    }
    size_t keep = 0;
    float cum = 0;
    while (keep < n) {
        cum += p[keep] / sum;
        keep += 1;
        if (cum >= sp.top_p) break;
    }
    return keep;
}

// A draw from weights p[0..n] (total tot) over cand.
static uint32_t draw(QgenPrng *rng, const uint32_t *cand, const float *p, size_t n, float tot) {
    float r = qgen_prng_float(rng) * tot;
    for (size_t i = 0; i < n; i++) {
        r -= p[i];
        if (p[i] > 0 && r <= 0) return cand[i];
    }
    size_t last = n - 1;
    while (last > 0 && p[last] <= 0) last -= 1;
    return cand[last];
}

static uint32_t sample(Engine *e, const float *logits, size_t len, Sampling sp, uint32_t *cand) {
    if (sp.temperature <= 0) return model_argmax(logits, len); // vectorized, first index on ties
    float p[256];
    size_t n = dist(logits, len, sp, cand, p);
    float tot = 0;
    for (size_t i = 0; i < n; i++) tot += p[i];
    return draw(&e->rng, cand, p, n, tot);
}

uint32_t server_sample_pick(void *ctx, const float *row, bool has_draft, uint32_t d) {
    SampleChooser *c = ctx;
    float p[256];
    size_t n = dist(row, c->n, c->sp, c->cand, p);
    float tot = 0;
    for (size_t i = 0; i < n; i++) tot += p[i];
    if (has_draft) {
        for (size_t i = 0; i < n; i++)
            if (c->cand[i] == d) {
                if (qgen_prng_float(c->rng) * tot < p[i]) return d;
                tot -= p[i];
                p[i] = 0;
                break;
            }
        if (tot <= 0) return d;
    }
    return draw(c->rng, c->cand, p, n, tot);
}

// ---- the tool-call grammar (SAETTA_GRAMMAR=1, src/grammar.h) -----------------------------------
//
// A token the grammar refuses is replaced, and only then: greedy takes the model's argmax whenever
// it is valid (so a reply whose calls are well formed is unchanged), else the valid token with the
// highest logit; sampling draws as without the grammar and, when the draw is refused, draws again
// from the distribution restricted to valid tokens (the two steps together sample that restricted
// distribution exactly). No allocation: the state is copied on the stack for each check.

// The first replacements of a request, for its log line (refused token, the one taken, the reply's
// token index), kept without allocation.
typedef struct {
    size_t n;
    uint32_t refused[4], taken[4];
    size_t at[4];
    size_t gen; // tokens emitted so far (the index of the next one)
    size_t em;  // replacements inside an edit's oldText held to a file (SAETTA_EDIT_MATCH)
} GramLog;

static void gram_note(GramLog *l, size_t ahead, uint32_t refused, uint32_t taken) {
    if (!l || l->n >= 4) return;
    l->refused[l->n] = refused;
    l->taken[l->n] = taken;
    l->at[l->n] = l->gen + ahead;
    l->n++;
}

static bool grammar_switch(void) {
    const char *v = getenv("SAETTA_GRAMMAR");
    return v && v[0] && v[0] != '0';
}

// The valid token with the highest logit of row, first (the model's own choice) when it is valid:
// the top 256 in order, then the whole vocabulary.
static uint32_t gram_best_of(const Grammar *g, const GState *st, const float *row, size_t len, uint32_t first, uint32_t *cand) {
    size_t n = server_top_k(row, len, 256, cand);
    for (size_t i = 0; i < n; i++)
        if (grammar_ok(g, st, cand[i])) return cand[i];
    uint32_t best = first;
    bool any = false;
    for (size_t i = 0; i < len; i++)
        if ((!any || row[i] > row[best]) && grammar_ok(g, st, (uint32_t)i)) {
            best = (uint32_t)i;
            any = true;
        }
    return best;
}

static uint32_t gram_best(const Grammar *g, const GState *st, const float *row, size_t len, uint32_t first, uint32_t *cand,
                          size_t *masked, GramLog *lg, size_t ahead) {
    if (grammar_ok(g, st, first)) return first;
    *masked += 1;
    if (lg && grammar_in_edit(st)) lg->em++;
    uint32_t t = gram_best_of(g, st, row, len, first, cand);
    gram_note(lg, ahead, first, t);
    return t;
}

// A sampled token t refused by the grammar: a draw from the distribution of row restricted to
// valid tokens (the greedy best when none of its candidates is valid).
static uint32_t gram_redraw(QgenPrng *rng, const Grammar *g, const GState *st, const float *row, size_t len, Sampling sp,
                            uint32_t t, uint32_t *cand, size_t *masked, GramLog *lg, size_t ahead) {
    float p[256];
    size_t n = dist(row, len, sp, cand, p);
    float tot = 0;
    for (size_t i = 0; i < n; i++) {
        if (!grammar_ok(g, st, cand[i])) p[i] = 0;
        tot += p[i];
    }
    if (tot > 0) {
        *masked += 1;
        if (lg && grammar_in_edit(st)) lg->em++;
        uint32_t d = draw(rng, cand, p, n, tot);
        gram_note(lg, ahead, t, d);
        return d;
    }
    return gram_best(g, st, row, len, t, cand, masked, lg, ahead);
}

static uint32_t gram_sample(Engine *e, const Grammar *g, const GState *st, const float *logits, size_t len, Sampling sp,
                            uint32_t *cand, size_t *masked, GramLog *lg) {
    uint32_t t = sample(e, logits, len, sp, cand);
    if (grammar_ok(g, st, t)) return t;
    if (sp.temperature <= 0) return gram_best(g, st, logits, len, t, cand, masked, lg, 0);
    return gram_redraw(&e->rng, g, st, logits, len, sp, t, cand, masked, lg, 0);
}

// The token rule of a speculation round under the grammar (a SpecPick): the rule of the round
// without it (argmax, or SampleChooser's) when its token is valid, else as gram_sample; st walks
// the tokens picked, from the state after the round's first token.
typedef struct {
    const Grammar *g;
    GState st;
    QgenPrng *rng;
    Sampling sp;
    uint32_t *cand;
    size_t vocab;
    size_t masked;
    GramLog *log;
    size_t row; // rows picked so far (the token index past the round's first token)
} GramPick;

static uint32_t gram_pick(void *ctx, const float *row, bool has_draft, uint32_t d) {
    GramPick *p = ctx;
    uint32_t t;
    if (p->sp.temperature <= 0)
        t = gram_best(p->g, &p->st, row, p->vocab, model_argmax(row, p->vocab), p->cand, &p->masked, p->log, p->row);
    else {
        SampleChooser c = {p->rng, p->sp, p->cand, p->vocab};
        t = server_sample_pick(&c, row, has_draft, d);
        if (!grammar_ok(p->g, &p->st, t))
            t = gram_redraw(p->rng, p->g, &p->st, row, p->vocab, p->sp, t, p->cand, &p->masked, p->log, p->row);
    }
    grammar_feed(p->g, &p->st, t);
    p->row++;
    return t;
}

// ---- requests ---------------------------------------------------------------------------------

typedef enum { FMT_OPENAI, FMT_ANTHROPIC } Fmt;

// A request body larger than this is refused (413).
#define BODY_LIMIT ((uint64_t)64 << 20)

void server_error_json(Buf *o, bool anthropic, int status, Str msg) {
    const char *type = status >= 500 ? (anthropic ? "api_error" : "server_error")
                       : status == 413 && anthropic ? "request_too_large"
                                                    : "invalid_request_error";
    if (anthropic) {
        buf_printf(o, "{\"type\":\"error\",\"error\":{\"type\":\"%s\",\"message\":", type);
        chat_json_string(o, msg);
        buf_adds(o, "}}");
    } else {
        buf_adds(o, "{\"error\":{\"message\":");
        chat_json_string(o, msg);
        buf_printf(o, ",\"type\":\"%s\"}}", type);
    }
}
static void error_json(Buf *o, Fmt fmt, int status, Str msg) { server_error_json(o, fmt == FMT_ANTHROPIC, status, msg); }

static Err respond_error(Conn *c, Req *h, int status, Fmt fmt, Err err) {
    Buf msg = {0};
    error_json(&msg, fmt, status, str(err));
    Err e = json_respond(c, h, status, buf_str(&msg));
    buf_free(&msg);
    return e;
}

// Request body -> (OpenAI-shaped request, the model name to echo).
static Err parse_request(Arena *a, Str body, Fmt fmt, JValue **r, Str *model) {
    JValue *raw;
    TRY(json_parse(a, body, false, &raw));
    if (raw->kind != J_OBJECT) return "BadRequest";
    JValue *m = field(raw, "model");
    *model = (m && m->kind == J_STRING) ? m->s : str(base_id);
    if (fmt == FMT_ANTHROPIC) return anthropic_to_openai(a, raw, r);
    *r = raw;
    return OK;
}

// POST /v1/messages/count_tokens: the rendered prompt's length, without running the model.
static Err count_tokens(Engine *e, Arena *a, Conn *c, Req *h, Str body, int *status) {
    JValue *r;
    Str model, prompt;
    TRY(parse_request(a, body, FMT_ANTHROPIC, &r, &model));
    TRY(chat_render_as(a, model_info(e->m)->chat, r, &prompt));
    VecU32 ids = {0};
    Err err = tokenizer_encode(e->tok, prompt, &ids);
    size_t n = ids.len;
    vec_free(&ids);
    if (err) {
        *status = 500; // not caused by the request, as in handle()
        return err;
    }
    return json_respond(c, h, 200, str(arena_printf(a, "{\"input_tokens\":%zu}", n)));
}

typedef enum { BLOCK_NONE, BLOCK_THINKING, BLOCK_TEXT } Block;

// One streamed response in either API's event format.
typedef struct {
    Arena *a;
    Fmt fmt;
    Conn *c;
    const char *id;
    int64_t created;
    Str model;
    uint64_t last;
    // Anthropic: the open content block and the index of the next one.
    Block block;
    size_t index;
    Buf ev; // the body bytes not yet sent
} Out;

// Sends the pending body bytes as one chunk and flushes the connection.
static Err out_flush(Out *o) {
    if (o->ev.len > 0 && o->c->sink) { // in process: the event bytes, no chunk framing
        Err err = conn_write(o->c, o->ev.ptr, o->ev.len);
        o->ev.len = 0;
        TRY(err);
    }
    if (o->ev.len > 0) {
        char hd[32];
        int n = snprintf(hd, sizeof hd, "%zx\r\n", o->ev.len);
        Err err = conn_write(o->c, hd, (size_t)n);
        if (!err) err = conn_write(o->c, o->ev.ptr, o->ev.len);
        if (!err) err = conn_writes(o->c, "\r\n");
        o->ev.len = 0;
        TRY(err);
    }
    TRY(conn_flush(o->c));
    o->last = now_ns();
    return OK;
}

// BodyWriter.end: the rest of the body and the last chunk.
static Err out_end(Out *o) {
    TRY(out_flush(o));
    if (o->c->sink) return conn_flush(o->c);
    TRY(conn_writes(o->c, "0\r\n\r\n"));
    return conn_flush(o->c);
}

// An error after the stream has started (the 200 and its headers are out): an error event
// in the request's format (OpenAI: a data line with the error object, then [DONE];
// Anthropic: event: error), then the end of the chunked body.
static Err out_error(Out *o, int status, Err err) {
    Buf msg = {0};
    error_json(&msg, o->fmt, status, str(err));
    if (o->fmt == FMT_OPENAI) buf_printf(&o->ev, "data: %.*s\n\ndata: [DONE]\n\n", (int)msg.len, msg.ptr);
    else buf_printf(&o->ev, "event: error\ndata: %.*s\n\n", (int)msg.len, msg.ptr);
    buf_free(&msg);
    return out_end(o);
}

static Err out_event(Out *o, const char *name, Str data) {
    buf_printf(&o->ev, "event: %s\ndata: %.*s\n\n", name, (int)data.len, data.ptr);
    return out_flush(o);
}

static Err out_chunk(Out *o, Str delta, const char *finish) {
    buf_printf(&o->ev,
               "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%lld,\"model\":\"%s"
               "\",\"choices\":[{\"index\":0,\"delta\":%.*s,\"finish_reason\":",
               o->id, (long long)o->created, reply_id, (int)delta.len, delta.ptr);
    if (finish) buf_printf(&o->ev, "\"%s\"}]}\n\n", finish);
    else buf_adds(&o->ev, "null}]}\n\n");
    return out_flush(o);
}

static Err out_begin(Out *o, size_t input, size_t cached) {
    if (o->fmt == FMT_OPENAI) return out_chunk(o, S("{\"role\":\"assistant\",\"content\":\"\"}"), NULL);
    Buf d = {0};
    buf_printf(&d, "{\"type\":\"message_start\",\"message\":{\"id\":\"%s\",\"type\":\"message\",\"role\":\"assistant\",\"model\":",
               o->id);
    chat_json_string(&d, o->model);
    buf_printf(&d,
               ",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":%zu,\"cache_creation_"
               "input_tokens\":0,\"cache_read_input_tokens\":%zu,\"output_tokens\":0}}}",
               input - cached, cached);
    Err err = out_event(o, "message_start", buf_str(&d));
    buf_free(&d);
    return err;
}

// A keep-alive if the stream has been quiet for ms.
static Err out_keep_alive(Out *o, int64_t ms) {
    if ((int64_t)(now_ns() - o->last) < ms * 1000000) return OK;
    if (o->fmt == FMT_OPENAI) buf_adds(&o->ev, ": keep-alive\n\n");
    else buf_adds(&o->ev, "event: ping\ndata: {\"type\":\"ping\"}\n\n");
    return out_flush(o);
}

static Err out_close(Out *o) {
    if (o->block == BLOCK_NONE) return OK;
    TRY(out_event(o, "content_block_stop", str(arena_printf(o->a, "{\"type\":\"content_block_stop\",\"index\":%zu}", o->index))));
    o->block = BLOCK_NONE;
    o->index += 1;
    return OK;
}

static Err out_open(Out *o, Block kind) {
    if (o->block == kind) return OK;
    TRY(out_close(o));
    const char *cb = kind == BLOCK_THINKING ? "{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}"
                                            : "{\"type\":\"text\",\"text\":\"\"}";
    TRY(out_event(o, "content_block_start",
                  str(arena_printf(o->a, "{\"type\":\"content_block_start\",\"index\":%zu,\"content_block\":%s}", o->index, cb))));
    o->block = kind;
    return OK;
}

static Err out_text(Out *o, Block kind, Str t) {
    if (t.len == 0) return OK;
    Buf d = {0};
    Err err;
    if (o->fmt == FMT_OPENAI) {
        buf_adds(&d, kind == BLOCK_THINKING ? "{\"reasoning_content\":" : "{\"content\":");
        chat_json_string(&d, t);
        buf_addc(&d, '}');
        err = out_chunk(o, buf_str(&d), NULL);
    } else {
        err = out_open(o, kind);
        if (!err) {
            buf_printf(&d, "{\"type\":\"content_block_delta\",\"index\":%zu,\"delta\":{\"type\":\"%s\",\"%s\":", o->index,
                       kind == BLOCK_THINKING ? "thinking_delta" : "text_delta", kind == BLOCK_THINKING ? "thinking" : "text");
            chat_json_string(&d, t);
            buf_adds(&d, "}}");
            err = out_event(o, "content_block_delta", buf_str(&d));
        }
    }
    buf_free(&d);
    return err;
}

// Anthropic stop_reason for a finish reason.
static const char *stop_reason(const char *finish, bool by_stop) {
    if (by_stop) return "stop_sequence";
    if (!strcmp(finish, "tool_calls")) return "tool_use";
    if (!strcmp(finish, "length")) return "max_tokens";
    return "end_turn";
}

// tool_choice of an OpenAI request: "none", "required", or a function's name ({"type":
// "function", "function": {"name": ...}}); TC_AUTO for auto.
typedef enum { TC_AUTO, TC_NONE, TC_REQUIRED, TC_NAMED } ToolChoiceKind;
static ToolChoiceKind tool_choice(const JValue *r, Str *named) {
    JValue *tc = field(r, "tool_choice");
    if (!tc) return TC_AUTO;
    if (tc->kind == J_STRING) {
        if (str_eqz(tc->s, "none")) return TC_NONE;
        if (str_eqz(tc->s, "required")) return TC_REQUIRED;
        return TC_AUTO;
    }
    if (tc->kind == J_OBJECT) {
        JValue *f = field(tc, "function");
        if (!f) return TC_AUTO;
        JValue *name = field(f, "name");
        if (!name || name->kind != J_STRING) return TC_AUTO;
        *named = name->s;
        return TC_NAMED;
    }
    return TC_AUTO;
}

static float round_ms(uint64_t t0) { return (float)(now_ns() - t0) / 1e6f; }

// SAETTA_BENCH_GREEDY (benchmarks only): see handle.
static bool bench_greedy(void) {
    const char *v = getenv("SAETTA_BENCH_GREEDY");
    return v && v[0] && v[0] != '0';
}

// The time of a speculation round as the depth controller learns it: measured, or under
// SAETTA_BENCH_GREEDY its current estimate for depth k, so the depths it picks depend only on the
// acceptance (a verification chunk's logits differ slightly from single steps', so depths that
// follow the clock make greedy output differ between runs).
static float depth_ms(const Depth *d, size_t k, uint64_t t0, bool fixed) {
    if (fixed && k < COUNT(d->ms)) return d->ms[k];
    return round_ms(t0);
}

// Per-request allocations outside the arena, freed by the caller of handle.
typedef struct {
    VecU32 ids;
    Lookup lk;
    FeedIx *fed; // the session's fed texts for prompt lookup (server_engine_feed), or NULL
    Splitter split;
    Out out;
    bool streaming;
    // the status a failure is answered with: 400 while the request is checked, 500 once it
    // runs (memory, the GPU: the server's fault, not the request's)
    int status;
} Work;

// A non-streamed request whose client has disconnected stops: checked after each prefill
// chunk and every GONE_EVERY decode steps from this loop (one non-blocking recv peek, never
// inside the model's per-token path), so it does not run on to max_tokens. A stream finds a
// gone client by its failing writes (keep-alives during prefill), and is not peeked: a
// client that half-closes after its request must still get the whole stream.
#define GONE_EVERY 16

// Whether the request asks for a fork point ("saetta_fork_point": true): a checkpoint at its
// generation prompt, where the prompts of its subagents (its messages, then their task) leave it.
static bool fork_point_asked(const JValue *r) {
    const JValue *fp = field((JValue *)r, "saetta_fork_point");
    return fp && fp->kind == J_BOOL && fp->b;
}

// The longest prefix ids[0..n) shares with another conversation's tokens (its restore points' and
// its checkpoints' prefix tokens), and that conversation in *src; 0 if none.
static size_t shared_elsewhere(const Engine *e, const uint32_t *ids, size_t n, size_t *src) {
    size_t best = 0;
    for (size_t i = 0; i < e->n_conv; i++) {
        if (!e->has[i] || e->conv_lora[i] != e->conv_lora[e->active]) continue;
        const Conv *cv = &e->held[i];
        const VecU32 *lists[3] = {&cv->ck[0].tokens, &cv->ck[1].tokens, &cv->ptoks};
        for (size_t k = 0; k < 3; k++) {
            size_t m = MIN(n, lists[k]->len), l = 0;
            if (m <= best) continue;
            while (l < m && lists[k]->ptr[l] == ids[l]) l++;
            if (l > best) {
                best = l;
                *src = i;
            }
        }
    }
    return best;
}

static bool has_cut(const size_t *cuts, size_t n, size_t x) {
    for (size_t i = 0; i < n; i++)
        if (cuts[i] == x) return true;
    return false;
}

// Prefill hook: a checkpoint at each cut (session_point_cuts), the monitor's progress, a stream's
// keep-alive, the client's disconnect.
typedef struct {
    Engine *e;
    Out *out;
    Conn *c;
    const uint32_t *ids;
    const size_t *cuts;
    size_t ncuts, ci;
} Hook;

static Err hook_chunk_done(void *ctx, size_t pos) {
    Hook *h = ctx;
    while (h->ci < h->ncuts && h->cuts[h->ci] < pos) h->ci++;
    if (h->ci < h->ncuts && h->cuts[h->ci] == pos) {
        h->ci++;
        // new buffers only while memory allows; else an empty slot's or a thinned checkpoint's
        TRY(session_save_point(h->e->s, h->e->g, h->ids, pos, point_room(h->e)));
    }
    atomic_store_explicit(&h->e->st->prompt_done, (uint32_t)pos, memory_order_release);
    if (h->out) TRY(out_keep_alive(h->out, 0));
    if (!h->out && conn_peer_gone(h->c)) return "ClientGone";
    return OK;
}

static Err anthropic_end(Arena *a, Conn *c, Req *h, Out *out, Splitter *sp, Parsed parsed, char **call_ids, const char *id,
                         Str name, size_t n, size_t reused, size_t n_gen, const char *stop, const Str *stop_seq);

// The generation after the prompt is routed: prefill, decode, the response.
static Err generate_reply(Engine *e, Arena *a, Conn *c, Req *h, Work *w, JValue *r, Str model_name, Fmt fmt, Str forced,
                          bool thinking, Sampling sp, const Str *stops, size_t n_stops, bool stream, const JValue *tools,
                          const char *id, int64_t created, Start start, size_t max_tokens) {
    const uint32_t *ids = w->ids.ptr;
    size_t n = w->ids.len;
    size_t reused = start.n;
    Out *out = NULL;
    if (stream) {
        if (c->sink) c->status = 200; // in process: no head
        else if (!c->head_sent) {
            TRY(http_write_expect_continue(c, h));
            TRY(conn_writes(c, "HTTP/1.1 200 OK\r\nconnection: close\r\ntransfer-encoding: chunked\r\ncontent-type: "
                               "text/event-stream\r\ncache-control: no-cache\r\n\r\n"));
        }
        w->out = (Out){.a = a, .fmt = fmt, .c = c, .id = id, .created = created, .model = model_name, .last = now_ns()};
        w->streaming = true;
        out = &w->out;
        TRY(out_begin(out, n, reused));
    }

    // prefill
    uint64_t t0 = now_ns();
    // prefix reuse: restore the checkpoint when this prompt extends it, then prefill the rest
    // up to the last token, checkpoint there, and run the last token for the logits. The
    // prompt is read in one prefill call (its n-gram preparation overlaps the GPU across
    // chunks; separate calls each paid it). After each chunk the hook keeps a prefix point
    // every PT_EVERY tokens (which another conversation sharing the prefix can start from),
    // updates the monitor's progress, and sends a stream's keep-alive.
    TRY(apply(e, start, true));
    const char *pv = getenv("SAETTA_POINTS");
    bool points_on = pv ? pv[0] != '0' : true;
    if (n - 1 > reused) {
        // the checkpoints of this prompt: one at its start (where the conversation's history ends
        // and the new messages begin: a retry, an edit or a /fork of this request restarts there; no
        // cut, the state is in place), and the cuts, where chunks end (message starts and multiples
        // of PT_EVERY, a function of the prompt and its start; and where it leaves another
        // conversation's tokens or, when asked, its generation prompt: chunks of PT_TAIL or more
        // either way, so where they end does not change what is computed)
        size_t *cuts = NULL, ncuts = 0;
        if (points_on && model_info(e->m)->chat == CHAT_QWEN) {
            if (reused > 0) {
                TRY(session_save_point(e->s, e->g, ids, reused, point_room(e)));
                int k = session_point_at(e->s->pts, reused);
                if (k >= 0) e->s->pts[k].start = true;
            }
            size_t cap = (n - 1 - reused) / PT_GAP + 6;
            cuts = ARENA_ARRAY(a, size_t, cap);
            ncuts = session_point_cuts(ids, n, reused, e->im_start, cuts, cap);
            // where the prompt leaves another conversation's tokens (its siblings' start: other
            // sessions sharing its system prompt and tools, other subagents of its parent), and,
            // when the caller asks (saetta_fork_point: Saetta Code's agent with subagents, whose
            // prompts are this one less its generation prompt), just before its generation prompt
            size_t src = 0, left = shared_elsewhere(e, ids, n, &src);
            if (left > 0) {
                size_t k0 = ncuts;
                ncuts = session_point_cuts_add(cuts, ncuts, cap, reused, n, left - 1);
                if (ncuts != k0 || has_cut(cuts, ncuts, left - 1))
                    dbg("saetta: conversation %zu leaves conversation %zu's tokens at %zu: a checkpoint there\n", e->active + 1,
                        src + 1, left - 1);
            }
            if (fork_point_asked(r)) {
                // at the generation prompt's <|im_start|>, at most n - 1 - PT_TAIL (a later one: the
                // few tokens of the last message between are read again by the children)
                size_t x = n - 1 - MIN(n - 1, PT_TAIL), im = n;
                while (im > reused && ids[im - 1] != e->im_start) im--;
                if (im > reused && im - 1 < x) x = im - 1;
                ncuts = session_point_cuts_add(cuts, ncuts, cap, reused, n, x);
            }
        }
        Hook hk = {e, out, c, ids, cuts, ncuts, 0};
        PrefillHook ph = {&hk, cuts, ncuts, hook_chunk_done};
        float *lg;
        TRY(model_prefill_with(e->m, e->s, ids + reused, n - 1 - reused, NULL, &ph, &lg));
    }
    if (n - 1 != reused) {
        TRY(session_save_checkpoint(e->s, ids, n - 1));
        e->s->ck[0].next = ids[n - 1]; // the prompt's last token, which its MTP row at n - 1 pairs
        e->s->ck[0].has_next = true;
    }
    float *logits;
    TRY(model_prefill(e->m, e->s, ids + n - 1, 1, NULL, &logits));
    uint64_t t1 = now_ns();
    atomic_store_explicit(&e->st->prompt_done, (uint32_t)n, memory_order_release);
    atomic_store_explicit(&e->st->decode_start_ns, stats_now_ns(), memory_order_release);
    atomic_store_explicit(&e->st->state, STATE_GENERATING, memory_order_release);

    Splitter *spl = &w->split;
    spl->thinking = thinking;
    spl->inkling = model_info(e->m)->chat == CHAT_INKLING;
    spl->arena = a;
    spl->stops = stops;
    spl->n_stops = n_stops;
    buf_addstr(&spl->text, forced); // the forced opening is part of the answer
    uint32_t cand[256];
    size_t n_gen = 0;
    const char *finish = "length";
    const Str *by_stop = NULL; // stop strings are slices: they may contain NUL
    // greedy requests speculate with prompt lookup (src/spec.h); the accepted drafts are
    // already consumed by the model and come out of queue without another pass
    const char *sv = getenv("SAETTA_SPEC");
    bool spec_on = (sv ? sv[0] != '0' : true) && !model_info(e->m)->no_spec;
    bool fixed = bench_greedy(); // read once per request, not per round
    // speculation at any temperature: greedy verifies by argmax, sampling by SampleChooser
    bool greedy = sp.temperature <= 0;
    Lookup *lk = &w->lk;
    for (size_t i = 0; i < n; i++) lookup_push(lk, ids[i]);
    // the texts this agent session fed (server_engine_feed), a second source for prompt lookup
    // (a reference held to the end of the request, dropped where Work is: the index stays as it was)
    e->last_fed = 0;
    if (spec_on && c->sink && req_session[0] && feed_on()) w->fed = feed_ref(e, req_session);
    const Lookup *flk = w->fed ? &w->fed->lk : NULL;
    Gate *gate = &e->gate[e->active];
    Depth *depth = &e->depth[e->active];
    uint32_t queue[KMAX];
    size_t qn = 0, qi = 0;
    // SAETTA_GRAMMAR=1: tool calls decoded under the grammar of the request's tools (Qwen chat
    // format only); gs is the state after the tokens emitted so far
    Grammar *gr = NULL;
    GState gs = {0};
    size_t g_masked = 0;
    GramLog glog = {0};
    const size_t vocab = model_info(e->m)->vocab;
    e->last_replaced = 0;
    // "saetta_edit_match" (Saetta Code with SAETTA_EDIT_MATCH=1): an edit's oldText held to the
    // files the agent knows, on the grammar (which the field turns on for this request)
    const JValue *emv = field(r, "saetta_edit_match");
    bool em_on = emv && emv->kind == J_OBJECT;
    if ((grammar_switch() || em_on) && tools && model_info(e->m)->chat == CHAT_QWEN) {
        gr = grammar_build(a, tools, e->tok, model_info(e->m)->eos, model_info(e->m)->n_eos);
        if (em_on) grammar_edit_files(gr, a, emv);
        if (gr->n_tools > 0) gs = grammar_start(gr, thinking, forced);
        else gr = NULL; // no tool a call could name: nothing to hold the reply to
    }
    uint32_t next = gr ? gram_sample(e, gr, &gs, logits, vocab, sp, cand, &g_masked, &glog) : sample(e, logits, vocab, sp, cand);
    size_t passes = 0;
    uint32_t *gen_ids = ARENA_ARRAY(a, uint32_t, max_tokens + 1);
    size_t n_gen_ids = 0;
    // speculation counters of this request (the "spec:" log line): passes, drafted, accepted
    size_t lk_p = 0, lk_d = 0, lk_a = 0, mtp_p = 0, mtp_d = 0, mtp_a = 0, fed_p = 0;
    while (n_gen < max_tokens) {
        uint32_t t = next;
        n_gen += 1;
        atomic_store_explicit(&e->st->gen, (uint32_t)n_gen, memory_order_release);
        gen_ids[n_gen_ids++] = t;
        if (gr) {
            grammar_feed(gr, &gs, t);
            glog.gen = n_gen;
        }
        bool eos = false;
        const ModelInfo *mi = model_info(e->m);
        for (size_t i = 0; i < mi->n_eos; i++)
            if (t == mi->eos[i]) eos = true;
        if (eos) {
            finish = "stop";
            break;
        }
        tokenizer_decode_append(e->tok, t, &spl->text);
        by_stop = split_cut_stop(spl);
        if (by_stop) {
            finish = "stop";
            break;
        }
        if (out) {
            SplitPoll d = split_poll(spl, false);
            TRY(out_text(out, BLOCK_THINKING, d.reasoning));
            TRY(out_text(out, BLOCK_TEXT, d.content));
            TRY(out_keep_alive(out, 5000));
        }
        if (n_gen >= max_tokens) break;
        if (!out && n_gen % GONE_EVERY == 0 && conn_peer_gone(c)) return "ClientGone";
        lookup_push(lk, t);
        if (qi < qn) {
            next = queue[qi];
            qi += 1;
            continue;
        }
        passes += 1;
        atomic_store_explicit(&e->st->passes, (uint32_t)passes, memory_order_release);
        uint32_t drafts[KMAX - 1];
        // a long prompt-lookup copy first (no head pass), else the MTP head at the depth the
        // controller picks (Depth in src/spec.h)
        uint64_t tr0 = now_ns();
        size_t nd = 0;
        bool from_fed = false;
        if (spec_on && !flk) nd = lookup_draft(lk, drafts, gate_depth(gate, KMAX - 1));
        else if (spec_on) {
            // the history's draft, or the fed texts' when its occurrence shares a longer context
            // with the tokens so far, or when the history has too short a draft
            size_t dep = gate_depth(gate, KMAX - 1), hm, fm;
            uint32_t fd[KMAX - 1];
            nd = lookup_draft_in(lk, lk->hist.ptr, lk->hist.len, drafts, dep, &hm);
            size_t nf = lookup_draft_in(flk, lk->hist.ptr, lk->hist.len, fd, dep, &fm);
            if (nf > 0 && (fm > hm || (nd < 3 && nf > nd))) {
                memcpy(drafts, fd, nf * sizeof *fd);
                nd = nf;
                from_fed = true;
            }
        }
        bool from_mtp = false;
        if (spec_on && nd < (mi->mtp_no_lookup ? 1u : 3u) && model_has_mtp(e->m)) {
            TRY(model_draft_mtp(e->m, e->s, t, depth_pick(depth, mi->mtp_kmax ? mi->mtp_kmax : KMAX - 1), drafts, &nd));
            from_mtp = nd > 0;
        } else if (nd > 0 && !gate_allow(gate)) nd = 0;
        if (nd > 0) {
            SpecResult sr;
            if (gr) {
                GramPick gp = {gr, gs, &e->rng, sp, cand, vocab, 0, &glog, 0};
                TRY(model_spec_verify(e->m, e->s, t, drafts, nd, gram_pick, &gp, &sr));
                g_masked += gp.masked;
            } else if (greedy) TRY(model_spec_greedy(e->m, e->s, t, drafts, nd, &sr));
            else {
                SampleChooser ch = {&e->rng, sp, cand, model_info(e->m)->vocab};
                TRY(model_spec_verify(e->m, e->s, t, drafts, nd, server_sample_pick, &ch, &sr));
            }
            if (from_mtp) depth_record(depth, nd, sr.accepted, depth_ms(depth, nd, tr0, fixed));
            else gate_record(gate, sr.accepted + 1);
            if (from_mtp) mtp_p++, mtp_d += nd, mtp_a += sr.accepted;
            else lk_p++, lk_d += nd, lk_a += sr.accepted, fed_p += from_fed;
            memcpy(queue, drafts, sr.accepted * sizeof(uint32_t));
            queue[sr.accepted] = sr.next;
            qn = sr.accepted + 1;
            next = queue[0];
            qi = 1;
        } else {
            TRY(model_step(e->m, e->s, t, &logits));
            next = gr ? gram_sample(e, gr, &gs, logits, vocab, sp, cand, &g_masked, &glog) : sample(e, logits, vocab, sp, cand);
            if (spec_on && model_has_mtp(e->m)) depth_record(depth, 0, 0, depth_ms(depth, 0, tr0, fixed));
        }
    }
    uint64_t t2 = now_ns();
    // decode-end restore point: the state after prompt + the generated tokens the model has
    // consumed, for the next turn whose prompt repeats this reply. When the reply ended with
    // accepted drafts still queued, the model consumed tokens beyond the emitted ones: the
    // state rolls back to the emitted ones (the last verification chunk keeps each token's
    // state), so the point is saved whatever the speculation did.
    size_t consumed = e->s->pos - n;
    if (consumed > n_gen_ids) {
        uint32_t *all = ARENA_ARRAY(a, uint32_t, n + n_gen_ids);
        memcpy(all, ids, n * sizeof(uint32_t));
        memcpy(all + n, gen_ids, n_gen_ids * sizeof(uint32_t));
        consumed = session_roll_to(e->s, n + n_gen_ids, all) ? n_gen_ids : 0;
    }
    if (consumed > 0) {
        uint32_t *all = ARENA_ARRAY(a, uint32_t, n + consumed);
        memcpy(all, ids, n * sizeof(uint32_t));
        memcpy(all + n, gen_ids, consumed * sizeof(uint32_t));
        (void)session_save_checkpoint_at(e->s, 1, all, n + consumed);
        // the first generated token not consumed (the next prompt repeats it after the reply); after
        // a rollback to every emitted token, none is known and the point is reused on its ids alone
        if (consumed < n_gen_ids) {
            e->s->ck[1].next = gen_ids[consumed];
            e->s->ck[1].has_next = true;
        }
    }
    int64_t pre_ms = (int64_t)(t1 - t0) / 1000000;
    int64_t dec_ms = (int64_t)(t2 - t1) / 1000000;
    Parsed parsed;
    if (spl->inkling) {
        // Inkling 2: the JSON calls of its special-token blocks (toolcall_inkling)
        split_finish(spl);
        parsed = (Parsed){str_trim(buf_str(&spl->ink.content)), spl->ink.calls, spl->ink.n_calls};
    } else
        parsed = toolcall_parse(a, str_trim(split_answer(spl)), tools);
    if (parsed.n_calls > 0) finish = "tool_calls";
    dbg("request: %zu prompt tokens (%zu reused) in %lld ms, %zu generated in %lld ms (%s tok/s, %zu passes), finish %s%s, "
        "conversation %zu/%zu\n",
        n, reused, (long long)pre_ms, n_gen, (long long)dec_ms, fx((double)n_gen * 1000.0 / (double)MAX(dec_ms, (int64_t)1), 2),
        passes, finish, fmt == FMT_ANTHROPIC ? " (anthropic)" : "", e->active + 1, e->n_conv);
    e->last_fed = fed_p;
    e->last_replaced = g_masked;
    dbg("spec: lookup %zu passes %zu drafted %zu accepted, mtp %zu passes %zu drafted %zu accepted, fed %zu\n", lk_p, lk_d, lk_a,
        mtp_p, mtp_d, mtp_a, fed_p);
    if (gr) {
        // the first replacements: the refused token's text, the one taken, and where (token index)
        char line[640];
        size_t o = (size_t)snprintf(line, sizeof line, "grammar: %zu tokens replaced", g_masked);
        if (em_on) o += (size_t)snprintf(line + o, sizeof line - o, " (edit match: %zu files, %zu in an oldText)", gr->n_files, glog.em);
        for (size_t i = 0; i < glog.n && o < sizeof line - 160; i++) {
            char rb[48], tb[48];
            size_t rn = tokenizer_token_bytes(e->tok, glog.refused[i], rb, sizeof rb - 1);
            size_t tn = tokenizer_token_bytes(e->tok, glog.taken[i], tb, sizeof tb - 1);
            Buf q = {0};
            chat_json_string(&q, str_n(rb, MIN(rn, sizeof rb - 1)));
            buf_adds(&q, " -> ");
            chat_json_string(&q, str_n(tb, MIN(tn, sizeof tb - 1)));
            o += (size_t)snprintf(line + o, sizeof line - o, "%s %.*s at %zu", i ? "," : ":", (int)MIN(q.len, (size_t)120), q.ptr,
                                  glog.at[i]);
            buf_free(&q);
        }
        dbg("%s\n", line);
    }
    {
        Done d = {0};
        d.anthropic = fmt == FMT_ANTHROPIC;
        d.prompt = (uint32_t)n;
        d.reused = (uint32_t)reused;
        d.output = (uint32_t)n_gen;
        d.passes = (uint32_t)passes;
        d.prefill_ms = (uint32_t)pre_ms;
        d.decode_ms = (uint32_t)dec_ms;
        memcpy(d.finish, finish, MIN(strlen(finish), sizeof d.finish));
        atomic_store_explicit(&e->st->ck0, (uint32_t)e->s->ck[0].tokens.len, memory_order_release);
        atomic_store_explicit(&e->st->ck1, (uint32_t)e->s->ck[1].tokens.len, memory_order_release);
        stats_finish(e->st, d);
        publish_convs(e);
    }
    char **call_ids = ARENA_ARRAY(a, char *, parsed.n_calls + 1);
    for (size_t i = 0; i < parsed.n_calls; i++)
        call_ids[i] = arena_printf(a, "%s_%016llx", fmt == FMT_ANTHROPIC ? "toolu" : "call",
                                   (unsigned long long)qgen_prng_next(&e->rng));

    if (fmt == FMT_ANTHROPIC)
        return anthropic_end(a, c, h, out, spl, parsed, call_ids, id, model_name, n, reused, n_gen,
                             stop_reason(finish, by_stop != NULL), by_stop);

    Buf calls = {0};
    buf_addc(&calls, '[');
    for (size_t i = 0; i < parsed.n_calls; i++) {
        if (i > 0) buf_addc(&calls, ',');
        buf_printf(&calls, "{\"index\":%zu,\"id\":\"%s\",\"type\":\"function\",\"function\":{\"name\":", i, call_ids[i]);
        chat_json_string(&calls, parsed.calls[i].name);
        buf_adds(&calls, ",\"arguments\":");
        chat_json_string(&calls, parsed.calls[i].arguments);
        buf_adds(&calls, "}}");
    }
    buf_addc(&calls, ']');
    const char *calls_json = arena_strdup(a, calls.ptr);
    buf_free(&calls);
    const char *usage = arena_printf(
        a, "{\"prompt_tokens\":%zu,\"completion_tokens\":%zu,\"total_tokens\":%zu,\"prompt_tokens_details\":{\"cached_tokens\":%zu}}",
        n, n_gen, n + n_gen, reused);

    if (out) {
        SplitPoll d = split_poll(spl, true);
        TRY(out_text(out, BLOCK_THINKING, d.reasoning));
        TRY(out_text(out, BLOCK_TEXT, d.content));
        if (parsed.n_calls > 0) TRY(out_chunk(out, str(arena_printf(a, "{\"tool_calls\":%s}", calls_json)), NULL));
        TRY(out_chunk(out, S("{}"), finish));
        buf_printf(&out->ev,
                   "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%lld,\"model\":\"%s"
                   "\",\"choices\":[],\"usage\":%s}\n\ndata: [DONE]\n\n",
                   id, (long long)created, reply_id, usage);
        return out_end(out);
    }
    Buf res = {0};
    buf_printf(&res,
               "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%lld,\"model\":\"%s"
               "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":",
               id, (long long)created, reply_id);
    if (parsed.content.len == 0 && parsed.n_calls > 0) buf_adds(&res, "null");
    else chat_json_string(&res, parsed.content);
    Str rc = split_reasoning(spl);
    if (rc.len > 0) {
        buf_adds(&res, ",\"reasoning_content\":");
        chat_json_string(&res, rc);
    }
    if (parsed.n_calls > 0) buf_printf(&res, ",\"tool_calls\":%s", calls_json);
    buf_printf(&res, "},\"finish_reason\":\"%s\"}],\"usage\":%s}", finish, usage);
    Err err = json_respond(c, h, 200, buf_str(&res));
    buf_free(&res);
    return err;
}

// The end of an Anthropic response: the rest of the text, the tool_use blocks, message_delta
// and message_stop when streaming; the whole message object otherwise.
static Err anthropic_end(Arena *a, Conn *c, Req *h, Out *o, Splitter *sp, Parsed parsed, char **call_ids, const char *id,
                         Str name, size_t n, size_t reused, size_t n_gen, const char *stop, const Str *stop_seq) {
    Buf seqb = {0};
    if (stop_seq) chat_json_string(&seqb, *stop_seq);
    else buf_adds(&seqb, "null");
    const char *seq = arena_strdup(a, seqb.ptr);
    buf_free(&seqb);
    if (o) {
        SplitPoll d = split_poll(sp, true);
        TRY(out_text(o, BLOCK_THINKING, d.reasoning));
        TRY(out_text(o, BLOCK_TEXT, d.content));
        TRY(out_close(o));
        for (size_t i = 0; i < parsed.n_calls; i++) {
            Call cl = parsed.calls[i];
            Buf st = {0};
            buf_printf(&st, "{\"type\":\"content_block_start\",\"index\":%zu,\"content_block\":{\"type\":\"tool_use\",\"id\":\"%s\",\"name\":",
                       o->index, call_ids[i]);
            chat_json_string(&st, cl.name);
            buf_adds(&st, ",\"input\":{}}}");
            Err err = out_event(o, "content_block_start", buf_str(&st));
            buf_free(&st);
            TRY(err);
            Buf dl = {0};
            buf_printf(&dl, "{\"type\":\"content_block_delta\",\"index\":%zu,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":",
                       o->index);
            chat_json_string(&dl, cl.arguments);
            buf_adds(&dl, "}}");
            err = out_event(o, "content_block_delta", buf_str(&dl));
            buf_free(&dl);
            TRY(err);
            TRY(out_event(o, "content_block_stop", str(arena_printf(a, "{\"type\":\"content_block_stop\",\"index\":%zu}", o->index))));
            o->index += 1;
        }
        TRY(out_event(o, "message_delta",
                      str(arena_printf(a,
                                       "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"%s\",\"stop_sequence\":%s},"
                                       "\"usage\":{\"output_tokens\":%zu}}",
                                       stop, seq, n_gen))));
        TRY(out_event(o, "message_stop", S("{\"type\":\"message_stop\"}")));
        return out_end(o);
    }
    Buf res = {0};
    buf_printf(&res, "{\"id\":\"%s\",\"type\":\"message\",\"role\":\"assistant\",\"model\":", id);
    chat_json_string(&res, name);
    buf_adds(&res, ",\"content\":[");
    bool first = true;
    Str rc = split_reasoning(sp);
    if (rc.len > 0) {
        buf_adds(&res, "{\"type\":\"thinking\",\"thinking\":");
        chat_json_string(&res, rc);
        buf_adds(&res, ",\"signature\":\"\"}");
        first = false;
    }
    if (parsed.content.len > 0) {
        if (!first) buf_addc(&res, ',');
        buf_adds(&res, "{\"type\":\"text\",\"text\":");
        chat_json_string(&res, parsed.content);
        buf_addc(&res, '}');
        first = false;
    }
    for (size_t i = 0; i < parsed.n_calls; i++) {
        Call cl = parsed.calls[i];
        if (!first) buf_addc(&res, ',');
        first = false;
        buf_printf(&res, "{\"type\":\"tool_use\",\"id\":\"%s\",\"name\":", call_ids[i]);
        chat_json_string(&res, cl.name);
        Str args = cl.arguments.len > 0 ? cl.arguments : S("{}");
        buf_printf(&res, ",\"input\":%.*s}", (int)args.len, args.ptr);
    }
    buf_printf(&res,
               "],\"stop_reason\":\"%s\",\"stop_sequence\":%s,\"usage\":{\"input_tokens\":%zu,\"cache_creation_input_tokens\":0,"
               "\"cache_read_input_tokens\":%zu,\"output_tokens\":%zu}}",
               stop, seq, n - reused, reused, n_gen);
    Err err = json_respond(c, h, 200, buf_str(&res));
    buf_free(&res);
    return err;
}

Err server_request_numbers(const JValue *r, ReqNumbers *out) {
    // every number is checked before its conversion: a cast of a negative, huge or NaN
    // double to an integer is undefined
    double v;
    out->max_tokens = 4096;
    if (num(field(r, "max_completion_tokens"), &v) || num(field(r, "max_tokens"), &v)) {
        if (!(v >= 1 && v <= (double)SESSION_CTX_MAX) || v != floor(v)) return "BadMaxTokens";
        out->max_tokens = (size_t)v;
    }
    out->sp.temperature = 1.0f;
    if (num(field(r, "temperature"), &v)) {
        if (!(v >= 0 && v <= 1e6)) return "BadTemperature";
        out->sp.temperature = (float)v;
    }
    out->sp.top_p = 0.95f;
    if (num(field(r, "top_p"), &v)) {
        if (!(v >= 0 && v <= 1)) return "BadTopP";
        out->sp.top_p = (float)v;
    }
    // top_k below 1 (0 conventionally means no limit) samples from the 256 kept candidates
    double tk = num(field(r, "top_k"), &v) ? v : 20;
    if (isnan(tk)) return "BadTopK";
    out->sp.top_k = tk < 1 ? 256 : (size_t)fmin(256, tk);
    out->has_seed = num(field(r, "seed"), &v);
    out->seed = 0;
    if (out->has_seed) {
        // any integer an i64 holds (a negative one wraps to its u64 bits)
        if (!(v >= -0x1p63 && v < 0x1p63) || v != floor(v)) return "BadSeed";
        out->seed = (uint64_t)(int64_t)v;
    }
    return OK;
}

// ---- conversation state on the SSD (src/state_store.h) ------------------------------------------

static void release_buf(void *b) { sx_buffer_release(b); }


// A saved state that covers more of the prompt than the conversations here (by at least
// RESTORE_GAIN tokens, about what reading it costs in prefill time) is read into the active
// conversation's buffers (sized for the prompt by ensure): its snapshot at n as restore point 1 and
// its second one (the checkpoint before n) as restore point 0, each when the prompt extends it,
// which plan then takes as for a resident conversation.
#define RESTORE_GAIN ((size_t)2048)
// The speculation controller saved with a state: a process resumed from it drafts as the one that
// saved it would have (its learned acceptance and depth), so the resumed conversation continues
// with the same tokens.
typedef struct {
    uint32_t magic;
    Gate gate;
    Depth depth;
} SpecSaved;
#define SPEC_SAVED_MAGIC 0x53504331u // "SPC1"

static void state_restore(Engine *e, const uint32_t *ids, size_t n) {
    Start cur = plan(e, ids, n, true);
    char name[33];
    size_t at[ST_MAX_SNAP];
    size_t got = state_store_find(e->store, ids, n, cur.n ? cur.n + RESTORE_GAIN : 0, name, at);
    if (!got) return;
    uint64_t t0 = now_ns();
    state_store_fence(e->store, sx_buffer_contents(e->s->kv), 0);
    size_t i = e->active;
    // snapshot 0 into ck[1] (with the MTP rows at its position), snapshot 1 into ck[0]
    StLayout lay, l0;
    if (!session_state_layout(e->s, 1, at[0] ? at[0] : got, &lay)) return;
    if (!at[0])
        for (size_t r = 0; r < ST_MAX_REC; r++) lay.rec[r] = NULL;
    if (at[1]) {
        if (!session_state_layout(e->s, 0, at[1], &l0)) return;
        memcpy(lay.xrec, l0.rec, sizeof lay.xrec);
    }
    size_t nn = 0, nkeys = 0;
    StMeta meta;
    uint8_t (*keys)[32] = NULL;
    uint64_t bytes = 0;
    bool touched = false;
    Err err = state_store_load(e->store, name, ids, n, &lay, &nn, &meta, &keys, &nkeys, &bytes, &touched);
    double ms = (double)(now_ns() - t0) / 1e6;
    // the conversation's contents change (or may have): another thread's end-of-turn save for its
    // earlier request in this conversation is no longer its conversation, even if this request
    // fails before apply counts it
    if (touched) {
        e->conv_gen[i] += 1;
        e->live = false;
    }
    if (!err && (nn != got || nn >= n)) err = "StateMismatch";
    // the snapshots read are the ones found (the layout's MTP rows were placed at their positions)
    for (size_t s = 0; !err && s < ST_MAX_SNAP; s++)
        if (s < meta.nsnap && meta.snap[s].loaded != (at[s] != 0)) err = "StateMismatch";
        else if (s < meta.nsnap && meta.snap[s].loaded && meta.snap[s].pos != at[s]) err = "StateMismatch";
        else if (s >= meta.nsnap && at[s]) err = "StateMismatch";
    if (err) {
        free(keys);
        dbg("state: dropped %s (%s); the prompt is read again\n", name, err);
        // the conversation stays as it was unless the load wrote into its buffers
        if (touched) {
            session_invalidate_from(e->s, 0);
            e->lin_set[i] = false;
            e->lin_hw[i] = 0;
            e->ninh[i] = 0;
        }
        return;
    }
    session_invalidate_from(e->s, 0); // its restore points and prefix points were for other rows
    for (size_t s = 0; s < meta.nsnap; s++) {
        if (!meta.snap[s].loaded) continue;
        Ckpt *k = &e->s->ck[s == 0 ? 1 : 0];
        size_t p = (size_t)meta.snap[s].pos;
        k->prev[0] = meta.snap[s].prev[0];
        k->prev[1] = meta.snap[s].prev[1];
        k->since_eos = (size_t)meta.snap[s].since_eos;
        // the token after it: snapshot 0's as saved, snapshot 1's the prompt's (the find checked it)
        k->has_next = s == 0 ? meta.snap[s].has_next : true;
        k->next = s == 0 ? meta.snap[s].next : ids[p];
        k->tokens.len = 0;
        vec_reserve(&k->tokens, p);
        memcpy(k->tokens.ptr, ids, p * sizeof(uint32_t));
        k->tokens.len = p;
    }
    // new chunks get a lineage of this conversation's own; the restored ones keep their keys (a
    // start below the state's n, at its second snapshot, keeps only those below it: apply)
    e->lin_set[i] = false;
    e->lin_hw[i] = (size_t)meta.snap[0].pos;
    free(e->inh[i]);
    e->inh[i] = keys;
    e->ninh[i] = nkeys;
    SpecSaved sv;
    if (meta.extra_len == sizeof sv) {
        memcpy(&sv, meta.extra, sizeof sv);
        if (sv.magic == SPEC_SAVED_MAGIC) {
            e->gate[i] = sv.gate;
            e->depth[i] = sv.depth;
        }
    }
    dbg("state: restored %zu of %zu prompt tokens%s (%s GB in %s ms, %s GB/s)\n", nn, n,
        meta.nsnap > 1 && meta.snap[1].loaded && !meta.snap[0].loaded ? " at the state's second snapshot" : "", fx((double)bytes / 1e9, 2), fx(ms, 0),
        fx((double)bytes / 1e6 / (ms > 0 ? ms : 1), 1));
}

// sha256 of n bytes of fd at off (fewer at the end of the file) into h.
static void hash_range(Sha256 *h, int fd, uint64_t off, size_t n) {
    uint8_t *b = xmalloc(n ? n : 1);
    ssize_t r = pread(fd, b, n, (off_t)off);
    if (r > 0) sha256_update(h, b, (size_t)r);
    free(b);
}

// The pack's identity: its resolved folder; config.json, the tokenizer files, the chat template
// and the weight index by content; every weight file by name, size, mtime, its safetensors
// header (the JSON of names, dtypes, shapes, offsets) and its first and last MB (hashing 15 to 80
// GB of weights at each start is left out).
static void pack_identity(const char *dir, uint8_t out[32]) {
    Sha256 h;
    sha256_init(&h);
    char *real = realpath(dir, NULL);
    sha256_update(&h, real ? real : dir, strlen(real ? real : dir) + 1);
    free(real);
    DIR *d = opendir(dir);
    char **names = NULL;
    size_t nn = 0, cap = 0;
    struct dirent *de;
    while (d && (de = readdir(d))) {
        if (de->d_name[0] == '.') continue;
        if (nn == cap) names = xrealloc(names, (cap = cap ? cap * 2 : 64) * sizeof *names);
        names[nn++] = strdup(de->d_name);
    }
    if (d) closedir(d);
    for (size_t i = 1; i < nn; i++) // insertion sort: a few dozen names
        for (size_t j = i; j > 0 && strcmp(names[j - 1], names[j]) > 0; j--) {
            char *t = names[j];
            names[j] = names[j - 1];
            names[j - 1] = t;
        }
    static const char *const by_content[] = {"config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                                             "chat_template.json", "generation_config.json", "special_tokens_map.json"};
    for (size_t i = 0; i < nn; i++) {
        char p[2048];
        snprintf(p, sizeof p, "%s/%s", dir, names[i]);
        bool content = str_ends(str(names[i]), S(".index.json"));
        for (size_t k = 0; k < COUNT(by_content); k++) content |= !strcmp(names[i], by_content[k]);
        struct stat sb;
        if (content) {
            char *text;
            size_t len;
            if (!read_file(p, 64u << 20, &text, &len)) {
                sha256_update(&h, names[i], strlen(names[i]) + 1);
                sha256_update(&h, text, len);
                free(text);
            }
        } else if (str_ends(str(names[i]), S(".safetensors")) && !stat(p, &sb)) {
            char line[1200];
            int k = snprintf(line, sizeof line, "%s %lld %lld.%09ld\n", names[i], (long long)sb.st_size, (long long)sb.st_mtimespec.tv_sec,
                             sb.st_mtimespec.tv_nsec);
            sha256_update(&h, line, (size_t)k);
            int fd = open(p, O_RDONLY);
            if (fd >= 0) {
                uint64_t hl = 0, size = (uint64_t)sb.st_size, mb = 1u << 20;
                if (pread(fd, &hl, 8, 0) == 8 && hl <= size - 8 && hl <= (64u << 20)) hash_range(&h, fd, 8, (size_t)hl);
                hash_range(&h, fd, 0, (size_t)MIN(mb, size));
                hash_range(&h, fd, size > mb ? size - mb : 0, (size_t)MIN(mb, size));
                close(fd);
            }
        }
        free(names[i]);
    }
    free(names);
    sha256_final(&h, out);
}

Err server_engine_state_enable(Engine *e, const char *root, const char *version) {
    _Static_assert(sizeof(SpecSaved) <= ST_EXTRA_MAX, "controller blob");
    StLayout lay;
    if (!session_state_layout(e->s, 1, 0, &lay)) return "StateUnsupported";
    uint8_t ident[32], shape[32];
    pack_identity(e->dir, ident);
    // the state's shape: the engine build (a hash of its sources and flags, SAETTA_ENGINE_ID from
    // the Makefile: a state is this build's own numbers), the format, the version, and the shapes
    // the rows and the recurrent state are written with; it names the store's folder
    Sha256 h;
    sha256_init(&h);
    char line[512];
    int k = snprintf(line, sizeof line, "saetta state %u %s %s %s %zu %zu\n", ST_FORMAT, SAETTA_ENGINE_ID, version,
                     model_info(e->m)->model_type, lay.nreg, lay.nrec);
    sha256_update(&h, line, (size_t)k);
    for (size_t i = 0; i < lay.nreg; i++) {
        k = snprintf(line, sizeof line, "%zu %zu %zu\n", lay.reg[i].planes, lay.reg[i].row_bytes, lay.reg[i].per_row);
        sha256_update(&h, line, (size_t)k);
    }
    for (size_t i = 0; i < lay.nrec; i++) {
        k = snprintf(line, sizeof line, "%zu\n", lay.rec_bytes[i]);
        sha256_update(&h, line, (size_t)k);
    }
    // the runtime choices that change the numbers: every switch src/engine_env.h lists as one,
    // whether the accelerator prompt path is in use (it falls back to the simdgroup kernels when it
    // does not compile), the MTP head, the macOS build (its Metal compiler), the chip and the GPU
    // (its name and family)
    for (size_t i = 0; i < COUNT(engine_env_rows); i++) {
        const char *v = getenv(engine_env_rows[i]);
        k = snprintf(line, sizeof line, "%s=%s\n", engine_env_rows[i], v ? v : "");
        sha256_update(&h, line, (size_t)k);
    }
    char osv[64] = "";
    size_t ol = sizeof osv - 1;
    if (sysctlbyname("kern.osversion", osv, &ol, NULL, 0) != 0) osv[0] = 0;
    char chip[256] = "", gpu[256] = "";
    size_t cl = sizeof chip - 1;
    if (sysctlbyname("machdep.cpu.brand_string", chip, &cl, NULL, 0) != 0) chip[0] = 0;
    sx_device_name(e->g->dev, gpu, sizeof gpu);
    k = snprintf(line, sizeof line, "accel %d mtp %d os %s chip %s gpu %s apple10 %d\n", model_fast_prefill(e->m) ? 1 : 0,
                 model_has_mtp(e->m) ? 1 : 0, osv, chip, gpu, sx_device_supports_apple10(e->g->dev));
    sha256_update(&h, line, (size_t)k);
    sha256_final(&h, shape);
    Err err = state_store_open(root, ident, shape, &e->store);
    if (!err) dbg("state: store %s (engine %s)\n", state_store_dir(e->store), SAETTA_ENGINE_ID);
    return err;
}

static void state_save_active(Engine *e);

void server_engine_state_save(Engine *e) {
    if (!e->store || !last_conv.set || last_conv.e != e) return;
    // in the engine's turn (other threads run requests: subagents, shared sessions), on the
    // conversation this caller's last request used, if no request used it since
    uint64_t t = turn_take(e);
    while (!turn_wait(e, t, 1000)) {}
    if (last_conv.conv < e->n_conv && e->conv_gen[last_conv.conv] == last_conv.gen) {
        activate(e, last_conv.conv);
        state_save_active(e);
    }
    turn_done(e);
}

// The save of the active conversation (in the engine's turn).
static void state_save_active(Engine *e) {
    if (e->conv_lora[e->active] != -1) return;
    Session *s = e->s;
    size_t slot = s->ck[1].tokens.len >= s->ck[0].tokens.len ? 1 : 0;
    const Ckpt *k = &s->ck[slot];
    if (k->tokens.len == 0) return;
    StLayout lay;
    if (!session_state_layout(s, slot, k->tokens.len, &lay)) return;
    void *hold[3];
    size_t nh = 0;
    hold[nh++] = sx_buffer_retain(s->kv);
    if (s->ixk) hold[nh++] = sx_buffer_retain(s->ixk);
    if (s->bk) hold[nh++] = sx_buffer_retain(s->bk);
    size_t i = e->active;
    if (!e->lin_set[i]) {
        arc4random_buf(e->lin[i], 16);
        e->lin_set[i] = true;
    }
    if (k->tokens.len > e->lin_hw[i]) e->lin_hw[i] = k->tokens.len;
    size_t n = k->tokens.len;
    StMeta meta = {.nsnap = 1, .extra_len = sizeof(SpecSaved)};
    meta.snap[0] = (StSnap){n, {k->prev[0], k->prev[1]}, k->since_eos, false, k->has_next, k->next};
    // the second snapshot: the conversation's last checkpoint before n on the same rows (the other
    // restore point when its tokens are a prefix of these, or a checkpoint within the prefix ptoks
    // shares with them)
    size_t best = 0, bslot = 0;
    const Ckpt *o = &s->ck[1 - slot];
    if (o->tokens.len > 0 && o->tokens.len < n && !memcmp(o->tokens.ptr, k->tokens.ptr, o->tokens.len * sizeof(uint32_t))) {
        best = o->tokens.len;
        bslot = 1 - slot;
    }
    size_t lcp = 0, lim = MIN(s->ptoks.len, n);
    while (lcp < lim && s->ptoks.ptr[lcp] == k->tokens.ptr[lcp]) lcp++;
    for (size_t q = 0; q < PT_MAX; q++) {
        const Point *pt = &s->pts[q];
        if (pt->gdn && pt->valid && pt->pos <= lcp && pt->pos < n && pt->pos > best) {
            best = pt->pos;
            bslot = 2 + q;
        }
    }
    StLayout l2;
    if (best > 0 && session_state_layout(s, bslot, best, &l2)) {
        memcpy(lay.xrec, l2.rec, sizeof lay.xrec);
        const uint32_t *pv = bslot < 2 ? s->ck[bslot].prev : s->pts[bslot - 2].prev;
        size_t se = bslot < 2 ? s->ck[bslot].since_eos : s->pts[bslot - 2].since_eos;
        meta.snap[1] = (StSnap){best, {pv[0], pv[1]}, se, false, true, k->tokens.ptr[best]};
        meta.nsnap = 2;
    }
    memcpy(meta.lineage, e->lin[i], 16);
    SpecSaved sv = {SPEC_SAVED_MAGIC, e->gate[i], e->depth[i]};
    memcpy(meta.extra, &sv, sizeof sv);
    (void)state_store_save(e->store, &lay, k->tokens.ptr, k->tokens.len, &meta, (const uint8_t (*)[32])e->inh[i], e->ninh[i], hold, nh,
                           release_buf);
}

void server_engine_state_flush(Engine *e) {
    if (e->store) state_store_wait(e->store);
}

size_t server_engine_point_slots(const Engine *e, size_t *active, uint64_t *reserved) {
    size_t all = 0, act = 0;
    for (size_t k = 0; k < PT_MAX; k++) act += e->s->pts[k].gdn != NULL;
    all = act;
    for (size_t i = 0; i < e->n_conv; i++)
        if (e->has[i])
            for (size_t k = 0; k < PT_MAX; k++) all += e->held[i].pts[k].gdn != NULL;
    if (active) *active = act;
    if (reserved) *reserved = session_points_reserve(e->s, e->max_conv);
    return all;
}

bool server_engine_controller(const Engine *e, size_t i, Gate *gate, Depth *depth) {
    if (i >= e->n_conv) return false;
    *gate = e->gate[i];
    *depth = e->depth[i];
    return true;
}

// The request object a prompt is rendered from: body parsed, tool_choice none leaving the tools out,
// --default-chat-template-kwargs filling in what it leaves out (handle; server_engine_ahead renders
// the next request's prefix through this too, so both render the same text).
static Err request_object(Engine *e, Arena *a, Str body, Fmt fmt, JValue **out, Str *model_name, ToolChoiceKind *choice_out,
                          Str *named_out) {
    JValue *r;
    TRY(parse_request(a, body, fmt, &r, model_name));
    // tool_choice: none leaves the tools out of the prompt; required or a named function
    // starts the answer with the tool call's opening (after an empty thinking block), so
    // the model continues a call (as Strata does: no grammar)
    Str named = {0};
    ToolChoiceKind choice = field(r, "tools") != NULL ? tool_choice(r, &named) : TC_AUTO;
    if (choice == TC_NONE) json_remove(r, "tools");
    if (e->kwargs_default) {
        // a key the request sets in chat_template_kwargs or at the top level wins
        JValue *kw = field(r, "chat_template_kwargs");
        for (size_t i = 0; i < e->kwargs_default->obj.len; i++) {
            const JMember *d = &e->kwargs_default->obj.items[i];
            const char *k = arena_strndup(a, d->key.ptr, d->key.len);
            if ((kw && json_get(kw, k)) || field(r, k)) continue;
            if (!kw || kw->kind != J_OBJECT) {
                kw = json_new(a, J_OBJECT);
                json_put(a, r, S("chat_template_kwargs"), kw);
            }
            json_put(a, kw, d->key, d->val);
        }
    }
    *out = r;
    *choice_out = choice;
    *named_out = named;
    return OK;
}

static Err handle(Engine *e, Arena *a, Conn *c, Req *h, Str body, Fmt fmt, Work *w) {
    JValue *r;
    Str model_name, named;
    ToolChoiceKind choice;
    TRY(request_object(e, a, body, fmt, &r, &model_name, &choice, &named));
    Str prompt;
    TRY(chat_render_as(a, model_info(e->m)->chat, r, &prompt));
    Str forced = S("");
    // a required tool call is forced in Qwen's XML only (Inkling's calls are left to the model)
    if ((choice == TC_REQUIRED || choice == TC_NAMED) && model_info(e->m)->chat == CHAT_QWEN) {
        if (str_ends(prompt, S("<think>\n"))) prompt = str(arena_printf(a, "%.*s\n</think>\n\n", (int)prompt.len, prompt.ptr));
        // with one declared tool, "required" means that tool
        JValue *tl = field(r, "tools");
        Str only = {0};
        bool has_only = false;
        if (tl->kind == J_ARRAY && tl->arr.len == 1) {
            JValue *f = field(tl->arr.items[0], "function");
            if (!f) f = tl->arr.items[0];
            JValue *nm = field(f, "name");
            if (nm && nm->kind == J_STRING) {
                only = nm->s;
                has_only = true;
            }
        }
        if (choice == TC_NAMED) forced = str(arena_printf(a, "<tool_call>\n<function=%.*s>\n", (int)named.len, named.ptr));
        else if (has_only) forced = str(arena_printf(a, "<tool_call>\n<function=%.*s>\n", (int)only.len, only.ptr));
        else forced = S("<tool_call>\n");
        prompt = str(arena_printf(a, "%.*s%.*s", (int)prompt.len, prompt.ptr, (int)forced.len, forced.ptr));
    }
    // the only tokenizer error (TokenNotInVocab: byte-level BPE covers every byte) is the
    // tokenizer's, not the request's
    w->status = 500;
    TRY(tokenizer_encode(e->tok, prompt, &w->ids));
    w->status = 400;
    bool thinking = forced.len == 0 && str_ends(prompt, S("<think>\n")) && !str_ends(prompt, S("</think>\n\n"));
    ReqNumbers rn;
    TRY(server_request_numbers(r, &rn));
    size_t asked = rn.max_tokens;
    Sampling sp = rn.sp;
    // SAETTA_BENCH_GREEDY=1, for benchmarks only (tools/code/session_bench.py): greedy decoding
    // whatever the request asks, and the MTP depth controller fed no measured times (depth_ms), so a
    // scripted session repeats token for token across runs
    if (bench_greedy()) sp.temperature = 0;
    if (rn.has_seed) qgen_prng_init(&e->rng, rn.seed);
    Str *stops = NULL;
    size_t n_stops = 0;
    JValue *stv = field(r, "stop");
    if (stv && stv->kind == J_STRING) {
        stops = ARENA_ARRAY(a, Str, 1);
        stops[n_stops++] = stv->s;
    } else if (stv && stv->kind == J_ARRAY) {
        stops = ARENA_ARRAY(a, Str, stv->arr.len + 1);
        for (size_t i = 0; i < stv->arr.len; i++)
            if (stv->arr.items[i]->kind == J_STRING) stops[n_stops++] = stv->arr.items[i]->s;
    }
    JValue *sv = field(r, "stream");
    bool stream = sv && sv->kind == J_BOOL && sv->b;
    const JValue *tools = field(r, "tools");
    const char *id;
    if (fmt == FMT_ANTHROPIC) {
        uint64_t hi = qgen_prng_next(&e->rng);
        uint32_t lo = (uint32_t)qgen_prng_next(&e->rng);
        id = arena_printf(a, "msg_%016llx%08x", (unsigned long long)hi, lo);
    } else {
        id = arena_printf(a, "chatcmpl-%llx", (unsigned long long)qgen_prng_next(&e->rng));
    }
    int64_t created = (int64_t)(wall_ns() / 1000000000ull);
    size_t n = w->ids.len;
    // the reply fits in the context: SAETTA_CTX less the prompt and the speculation slack
    if (n + CAP_SLACK >= e->s->max_cap) return "PromptTooLong";
    size_t max_tokens = MIN(asked, e->s->max_cap - n - CAP_SLACK);
    // the request's model: a LoRA adapter's name selects it, anything else the base model
    int lora = model_name.len > 0 ? model_lora_find(e->m, arena_strndup(a, model_name.ptr, model_name.len)) : -1;
    w->status = 500;
    route(e, w->ids.ptr, n, lora, c->sink != NULL);
    reply_id = lora >= 0 ? model_lora_name(e->m, (size_t)lora) : base_id;
    TRY(ensure(e, n + max_tokens + CAP_SLACK));
    if (e->store && lora < 0) state_restore(e, w->ids.ptr, n);
    Start start = plan(e, w->ids.ptr, n, true);
    stats_begin(e->st, fmt == FMT_ANTHROPIC, n, start.n, max_tokens);
    Err err = generate_reply(e, a, c, h, w, r, model_name, fmt, forced, thinking, sp, stops, n_stops, stream, tools, id, created,
                             start, max_tokens);
    last_conv = (LastConv){e, e->active, e->conv_gen[e->active], true, {0}};
    e->kept[e->active] = fork_point_asked(r);
    if (c->sink) snprintf(last_conv.session, sizeof last_conv.session, "%s", req_session);
    // the conversation an ahead read of this session may continue (server_engine_ahead)
    snprintf(e->owner[e->active], sizeof e->owner[0], "%s", c->sink ? req_session : "");
    e->owner_gen[e->active] = e->conv_gen[e->active];
    if (err) atomic_store_explicit(&e->st->state, STATE_IDLE, memory_order_release);
    return err;
}

// A header of the request head (case-insensitive name), empty when absent.
static Str head_field(const Conn *c, const char *name) {
    Str hd = str_n(c->head, c->head_len);
    size_t nl = strlen(name);
    for (size_t i = 0; i < hd.len;) {
        ptrdiff_t e = str_find_char(hd, '\n', i);
        size_t j = e >= 0 ? (size_t)e : hd.len;
        Str line = str_sub(hd, i, j);
        if (line.len > nl && line.ptr[nl] == ':' && !strncasecmp(line.ptr, name, nl)) return str_trim(str_sub(line, nl + 1, line.len));
        i = j + 1;
    }
    return S("");
}

// Equal strings, in a time that does not depend on where they differ.
static bool same_secret(Str a, Str b) {
    unsigned char d = a.len != b.len;
    for (size_t i = 0; i < a.len && i < b.len; i++) d |= (unsigned char)(a.ptr[i] ^ b.ptr[i]);
    return d == 0;
}

// The body asks for a stream ("stream": true at its top level).
static bool body_streams(Arena *a, Str body) {
    JValue *v;
    if (json_parse(a, body, false, &v) || v->kind != J_OBJECT) return false;
    const JValue *sv = json_get(v, "stream");
    return sv && sv->kind == J_BOOL && sv->b;
}

// Waits for turn t while the client waits too: a streamed request gets its head and a keep-alive
// comment every 5 s (so the client's idle timeout does not fire while others run), a non-streamed
// one is checked for a client that left. False when the client left (the turn is given up).
static bool wait_turn_keeping_alive(Engine *e, Arena *a, uint64_t t, Conn *c, Req *h, Fmt fmt, Str body, bool count) {
    if (turn_wait(e, t, 0)) return true; // not contended (server_serve, mostly): nothing more to do
    // count_tokens answers with one JSON body: never a stream head before it
    bool stream = !count && body_streams(a, body);
    while (!turn_wait(e, t, 5000)) {
        bool left;
        if (stream) {
            Err err = OK;
            if (!c->head_sent) {
                err = conn_writes(c, "HTTP/1.1 200 OK\r\nconnection: close\r\ntransfer-encoding: chunked\r\ncontent-type: "
                                     "text/event-stream\r\ncache-control: no-cache\r\n\r\n");
                c->head_sent = true;
            }
            const char *ka = fmt == FMT_OPENAI ? ": queued\n\n" : "event: ping\ndata: {\"type\":\"ping\"}\n\n";
            char hd[16];
            snprintf(hd, sizeof hd, "%zx\r\n", strlen(ka));
            if (!err) err = conn_writes(c, hd);
            if (!err) err = conn_writes(c, ka);
            if (!err) err = conn_writes(c, "\r\n");
            if (!err) err = conn_flush(c);
            left = err != OK;
        } else left = conn_peer_gone(c);
        (void)h;
        if (left) {
            turn_leave(e, t);
            return false;
        }
    }
    return true;
}

// The challenge a shared engine answers without the token (code_share.c verifies its holder before
// it sends one): GET /saetta/whoami?n=<nonce> -> {"hmac": HMAC-SHA256(token, nonce) in hex}.
static Err whoami(Engine *e, Conn *c, Req *h) {
    Str target = h->target;
    ptrdiff_t q = str_find(target, S("?n="), 0);
    Str n = q >= 0 ? str_sub(target, (size_t)q + 3, target.len) : S("");
    if (n.len < 32 || n.len > 128) return json_respond(c, h, 400, S("{\"error\":{\"message\":\"bad nonce\"}}"));
    const char *key = e->token + 7; // after "Bearer "
    uint8_t d[SHA256_LEN];
    hmac_sha256(key, strlen(key), n.ptr, n.len, d);
    char hex[2 * SHA256_LEN + 1];
    hash_hex(d, SHA256_LEN, hex);
    char body[96];
    snprintf(body, sizeof body, "{\"hmac\":\"%s\"}", hex);
    return json_respond(c, h, 200, str(body));
}

static Err connection(Engine *e, int fd, bool guard, void (*authed)(void *), void *actx) {
    Conn *c = xmalloc(sizeof(Conn));
    conn_init(c, fd);
    // a shared connection (no guard) has 5 s for its head: until its token is checked it holds
    // nothing, and a slow one is dropped soon
    c->deadline_ns = now_ns() + (uint64_t)(guard ? HTTP_REQUEST_DEADLINE_MS : 5000u) * 1000000ull;
    if (guard) http_guard_attach(fd);
    Err ret = OK;
    Req h;
    Err he = http_receive_head(c, &h);
    if (he) {
        // a malformed head is answered (431 / 400 / 408), a client that left is not
        int status = http_error_status(he);
        if (status) {
            Req none = {.method = M_GET};
            // drained like a refused body (1 MB, 1 s at most), so the client sees the answer
            if (!respond_error(c, &none, status, FMT_OPENAI, he)) http_drain(c, 1u << 20, 1000);
        }
        ret = status ? he : OK;
        goto done;
    }
    if (e->token[0] && h.method == M_GET && str_starts(h.target, S("/saetta/whoami"))) {
        ret = whoami(e, c, &h);
        goto done;
    }
    if (e->token[0]) {
        // a shared engine (code_share.c): no browser page, and only the holder's sessions (the
        // token of its record) are answered, before any body is read
        Req none = {.method = M_GET};
        int status = head_field(c, "origin").len ? 403
                     : !same_secret(head_field(c, "authorization"), str(e->token)) ? 401 : 0;
        if (status) {
            if (!respond_error(c, &none, status, FMT_OPENAI, status == 403 ? "Forbidden" : "Unauthorized")) http_drain(c, 1u << 20, 1000);
            ret = status == 403 ? "Forbidden" : "Unauthorized";
            goto done;
        }
        c->deadline_ns = now_ns() + (uint64_t)HTTP_REQUEST_DEADLINE_MS * 1000000ull; // the body's, as server_serve's
        if (authed) authed(actx);
    }
    {
        Arena a = {0};
        Str target = h.target;
        ptrdiff_t q = str_find_char(target, '?', 0);
        Str path = q >= 0 ? str_sub(target, 0, (size_t)q) : target;
        if (h.method == M_GET && str_eqz(path, "/health")) {
            ret = json_respond(c, &h, 200, S("{\"status\":\"ok\"}"));
        } else if (h.method == M_GET && str_eqz(path, "/v1/models")) {
            Buf mb = {0};
            buf_printf(&mb, "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\",\"owned_by\":\"saetta\",\"max_model_len\":%zu}",
                       base_id, e->s->max_cap);
            for (size_t i = 0; i < model_lora_count(e->m); i++) {
                buf_adds(&mb, ",{\"id\":");
                json_write_string(&mb, str(model_lora_name(e->m, i)));
                buf_printf(&mb, ",\"object\":\"model\",\"owned_by\":\"saetta\",\"parent\":\"%s\"}", base_id);
            }
            buf_adds(&mb, "]}");
            ret = json_respond(c, &h, 200, buf_str(&mb));
            buf_free(&mb);
        } else if (h.method == M_HEAD && str_eqz(path, "/api/hello")) {
            ret = http_respond(c, &h, 200, S(""), NULL); // Claude Code's startup probe
        } else if (h.method == M_POST && (str_eqz(path, "/v1/chat/completions") || str_eqz(path, "/v1/messages") ||
                                          str_eqz(path, "/v1/messages/count_tokens"))) {
            Fmt fmt = str_starts(path, S("/v1/messages")) ? FMT_ANTHROPIC : FMT_OPENAI;
            Buf body = {0};
            // a client that waits for 100 Continue gets it before its body is read (not when
            // the body is refused anyway, or for an expectation other than 100-continue)
            Err be = OK;
            if (h.has_expect && !str_eqz(h.expect, "100-continue")) be = "HttpExpectationFailed";
            else if (h.has_length && h.content_length > BODY_LIMIT) be = "BodyTooLarge";
            else if (h.has_expect) {
                be = http_write_expect_continue(c, &h);
                if (!be) be = conn_flush(c);
            }
            if (!be) be = http_read_body(c, &h, BODY_LIMIT, &body);
            c->deadline_ns = 0; // the request is in; generation and the answer have no deadline
            if (be) {
                dbg("request: %s\n", be);
                int status = http_error_status(be);
                h.has_expect = false;
                if (status) {
                    // the rest of a refused body is drained (1 MB, 1 s at most) so the client sees the answer
                    if (!respond_error(c, &h, status, fmt, be)) http_drain(c, 1u << 20, 1000);
                }
                buf_free(&body);
                arena_reset(&a);
                ret = be;
                goto done;
            }
            Str bs = str_n(arena_dup(&a, body.ptr ? body.ptr : "", body.len + 1), body.len);
            buf_free(&body);
            Work w = {.status = 400};
            // the engine's turn: only now, with the whole request read
            uint64_t t = turn_take(e);
            if (!wait_turn_keeping_alive(e, &a, t, c, &h, fmt, bs, str_ends(path, S("/count_tokens")))) {
                vec_free(&w.ids);
                arena_reset(&a);
                ret = "ClientGone";
                goto done;
            }
            Err err = str_ends(path, S("/count_tokens")) ? count_tokens(e, &a, c, &h, bs, &w.status) : handle(e, &a, c, &h, bs, fmt, &w);
            turn_done(e);
            if (err) {
                dbg("request failed: %s\n", err);
                // a client that is gone gets nothing; a started stream ends with an error event
                if (!strcmp(err, "WriteFailed") || !strcmp(err, "ClientGone")) ret = err;
                else if (w.streaming) (void)out_error(&w.out, w.status, err);
                else if (c->head_sent) { // keep-alives went out while it waited: the error is a stream event
                    Out o = {.a = &a, .fmt = fmt, .c = c, .last = now_ns()};
                    (void)out_error(&o, w.status, err);
                    buf_free(&o.ev);
                } else (void)respond_error(c, &h, w.status, fmt, err);
            }
            vec_free(&w.ids);
            lookup_free(&w.lk);
            feed_unref(e, w.fed);
            split_free(&w.split);
            buf_free(&w.out.ev);
        } else {
            ret = json_respond(c, &h, 404, S("{\"error\":{\"message\":\"not found\"}}"));
        }
        arena_reset(&a);
    }
    // one request per connection (connection: close): agent sessions sharing the server
    // take turns, and an idle keep-alive connection of one never holds up the others
done:
    if (guard) http_guard_detach();
    free(c);
    return ret;
}

static const char *basename_of(Arena *a, const char *dir) {
    size_t n = strlen(dir);
    while (n > 1 && dir[n - 1] == '/') n--;
    size_t b = n;
    while (b > 0 && dir[b - 1] != '/') b--;
    return arena_strndup(a, dir + b, n - b);
}

// The engine of a server (see server.h): the pack, the model, its LoRA adapters, the tokenizer,
// the conversations sized to this Mac's memory, sampling state. Shared by server_serve and the
// in-process callers (Saetta Code).
static Err engine_open(Arena *arena, const char *dir, const char *const *loras, size_t nloras, JValue *kwd, uint16_t port,
                       Engine **out) {
    uint64_t t0 = now_ns();
    Pack *pack = ARENA_NEW(arena, Pack);
    TRY(pack_open(arena, dir, pack));
    {
        const ModelInfo *mi = model_info_for(arena, pack);
        if (mi && mi->no_serve) {
            dbg("serve: %s cannot be served yet (no tokenizer and chat template for it); use saetta generate\n", mi->name);
            return "NotServable";
        }
    }
    Gpu *g = ARENA_NEW(arena, Gpu);
    TRY(gpu_init(arena, pack, g));
    // first GPU touch of the mapped pack (not the n-gram table), here instead of on the first request
    TRY(gpu_warm_except(g, pack, "ngram_embedding"));
    Model *m;
    TRY(model_load(arena, pack, g, &m));
    base_id = reply_id = model_info(m)->id;
    // --lora-modules name=dir: each adapter, served under its name next to the base model
    for (size_t i = 0; i < nloras; i++) {
        const char *eq = strchr(loras[i], '=');
        if (!eq || eq == loras[i] || !eq[1]) {
            dbg("saetta: --lora-modules wants name=path, got %s\n", loras[i]);
            return "BadLoraModule";
        }
        const char *name = arena_strndup(arena, loras[i], (size_t)(eq - loras[i]));
        if (!strcmp(name, model_info(m)->id) || model_lora_find(m, name) >= 0) {
            dbg("saetta: LoRA name %s is taken (the base model or another adapter)\n", name);
            return "BadLoraModule";
        }
        char *path;
        TRY(download_lora_dir(arena, eq + 1, &path));
        // what its adapter_config.json tells (base model, rank, targets), before the tensors
        const char *why = download_lora_check(arena, path, model_info(m)->id);
        if (why) {
            dbg("saetta: LoRA %s from %s: %s\n", name, path, why);
            return "LoraDoesNotFit";
        }
        Err le = model_lora_load(arena, m, name, path);
        if (le) {
            const char *what = download_lora_error(le);
            dbg("saetta: LoRA %s from %s: %s%s%s\n", name, path, le, what ? ": " : "", what ? what : "");
            return le;
        }
    }
    Tokenizer *tok = ARENA_NEW(arena, Tokenizer);
    TRY(tokenizer_load(arena, dir, tok));
    Stats *st = ARENA_NEW(arena, Stats);
    stats_init(st);
    st->model = base_id;
    st->pack = basename_of(arena, dir);
    st->port = port;
    st->started_s = (int64_t)(wall_ns() / 1000000000ull);
    st->dev = g->dev;
    Engine *e = ARENA_NEW(arena, Engine);
    e->st = st;
    e->g = g;
    e->max_conv = 4;
    const char *cv = getenv("SAETTA_CONVERSATIONS");
    if (cv) {
        uint64_t x;
        e->max_conv = parse_u64(str(cv), &x) ? 4 : (size_t)CLAMP(x, (uint64_t)1, (uint64_t)MAX_CONV);
    }
    e->m = m;
    {
        // conversations start small and grow (ensure); SAETTA_CTX is the maximum
        // SAETTA_CTX rounded down to a multiple of SESSION_CTX_ALIGN, at least SESSION_CTX_MIN
        size_t max_cap;
        const char *ctx = getenv("SAETTA_CTX");
        size_t max_pos = model_info(m)->max_positions;
        Err ce = session_ctx_parse(ctx, max_pos, &max_cap);
        if (ce) {
            dbg("saetta: SAETTA_CTX=%s: %s (a number of tokens from %d to %zu, the model's maximum)\n", ctx, ce,
                SESSION_CTX_MIN, max_pos);
            return ce;
        }
        if (ctx && strtoull(ctx, NULL, 10) != max_cap)
            dbg("saetta: SAETTA_CTX=%s rounded down to %zu (a multiple of %d)\n", ctx, max_cap, SESSION_CTX_ALIGN);
        TRY(session_create_cap(m, g, getenv("SAETTA_KV_F32") == NULL, MIN(CAP0, max_cap), max_cap, &e->s));
        // the context that fits this Mac's memory (fit.h): RAM less FIT_KV_RESERVE_GB less what
        // the GPU holds now (weights, scratch, the first conversation)
        // (less the n-gram table: clean file pages the CPU reads by row, fit_pack_paged_bytes)
        const uint64_t gib = 1ull << 30, ram = fit_ram_bytes(), alloc = sx_device_allocated_size(g->dev),
                       paged = fit_pack_paged_bytes(pack), used = alloc > paged ? alloc - paged : alloc;
        const char *ig = getenv("SAETTA_IGNORE_MEMORY");
        if (ram && !(ig && strcmp(ig, "0") != 0)) {
            const uint64_t budget = ram > used + (uint64_t)FIT_KV_RESERVE_GB * gib ? ram - used - (uint64_t)FIT_KV_RESERVE_GB * gib : 0;
            const size_t c0 = MIN(CAP0, max_cap), have = session_conv_bytes(e->s, c0);
            // growing a conversation allocates the new buffers before it frees the old ones
            // (session_grow), so the last step holds about twice the final size; the checkpoints
            // (PT_MAX for it, PT_HELD for each other conversation; 118 MB each on Flash-Next, 157 MB
            // on the 27B) come out of the budget
            const uint64_t pts = model_info(m)->chat == CHAT_QWEN ? session_points_reserve(e->s, e->max_conv) : 0;
            size_t fit = max_cap;
            while (fit > c0 && 2 * session_conv_bytes(e->s, fit) - have + pts > budget) fit -= SESSION_CTX_ALIGN;
            if (budget == 0)
                dbg("saetta: warning: this Mac's memory (%llu GB) is short even for a %zu-token context; expect swapping\n",
                    (unsigned long long)(ram / gib), c0);
            if (fit < max_cap) {
                dbg("saetta: context limited to %zu tokens by this Mac's memory (%llu GB; SAETTA_CTX asked %zu)\n", fit,
                    (unsigned long long)(ram / gib), max_cap);
                e->s->max_cap = fit;
            }
        }
    }
    e->tok = tok;
    qgen_prng_init(&e->rng, (uint64_t)(int64_t)wall_ns());
    if (model_info(m)->chat == CHAT_QWEN && !tokenizer_token_id(tok, S("</think>"), &e->think_end)) return "BadTokenizer";
    // message starts (src/chat.c writes "<|im_start|>" before each message): the checkpoints' candidates
    e->im_start = UINT32_MAX;
    if (model_info(m)->chat == CHAT_QWEN && !tokenizer_token_id(tok, S("<|im_start|>"), &e->im_start)) e->im_start = UINT32_MAX;
    for (size_t i = 0; i < MAX_CONV; i++) {
        e->gate[i] = GATE_INIT;
        e->depth[i] = DEPTH_INIT;
        if (model_info(m)->depth_ms) memcpy(e->depth[i].ms, model_info(m)->depth_ms, sizeof e->depth[i].ms);
    }
    e->n_conv = 1;
    e->active = 0;
    for (size_t i = 0; i < MAX_CONV; i++) e->conv_lora[i] = -1;
    e->kwargs_default = kwd;
    turns_init(&e->turns);
    pthread_mutex_init(&e->feed_mu, NULL);
    st->ctx = e->s->max_cap;
    e->opened_ns = t0;
    e->dir = arena_strdup(arena, dir);
    *out = e;
    return OK;
}

Err server_engine_open(Arena *arena, const char *dir, const char *const *loras, size_t nloras, const char *kwargs_default,
                       Engine **out) {
    // one engine per machine: not beside a server or another in-process engine
    Server srv;
    if (instance_running(&srv)) {
        dbg("saetta: a server is running (pid %d, port %u); one model runs at a time: use it (saetta code without "
            "--pack, or saetta start code), or stop it first\n", srv.pid, srv.port);
        return "EngineBusy";
    }
    int lock_fd;
    Err le = instance_inproc_acquire(dir, &lock_fd);
    if (le) {
        int pid = 0;
        instance_inproc_running(&pid);
        dbg("saetta: another Saetta Code (pid %d) is running the model in its process; one model runs at a time\n", pid);
        return le;
    }
    JValue *kwd = NULL;
    if (kwargs_default && (json_parse(arena, str(kwargs_default), true, &kwd) || kwd->kind != J_OBJECT)) return "BadKwargs";
    return engine_open(arena, dir, loras, nloras, kwd, 0, out);
}

const char *server_engine_model(const Engine *e) { return base_id; }

size_t server_engine_context(const Engine *e) { return e->s->max_cap; }

Stats *server_engine_stats(Engine *e) { return e->st; }

Err server_engine_chat(Engine *e, Str body, int (*sink)(void *ctx, int status, const void *p, size_t n), void *ctx) {
    return server_engine_chat_session(e, body, NULL, sink, ctx);
}

Err server_engine_chat_session(Engine *e, Str body, const char *session,
                               int (*sink)(void *ctx, int status, const void *p, size_t n), void *ctx) {
    // a longer id than the key holds counts as none (no in-place restart): never a false match
    if (session && strlen(session) < sizeof req_session) snprintf(req_session, sizeof req_session, "%s", session);
    else req_session[0] = 0;
    Conn *c = xcalloc(1, sizeof(Conn));
    conn_init(c, -1);
    c->sink = sink;
    c->sink_ctx = ctx;
    c->status = 200;
    Req h = {.method = M_POST, .target = S("/v1/chat/completions")};
    Arena a = {0};
    Str bs = str_n(arena_dup(&a, body.ptr ? body.ptr : "", body.len + 1), body.len);
    Work w = {.status = 400};
    // the engine's turn; a caller that leaves meanwhile (Esc) gives it up
    uint64_t t = turn_take(e);
    while (!turn_wait(e, t, 100))
        if (sink(ctx, 0, NULL, 0)) {
            turn_leave(e, t);
            arena_reset(&a);
            free(c);
            return OK;
        }
    Err err = handle(e, &a, c, &h, bs, FMT_OPENAI, &w);
    turn_done(e);
    Err ret = OK;
    if (err) {
        dbg("request failed: %s\n", err);
        // the caller that left gets nothing; a started stream ends with an error event
        if (!strcmp(err, "WriteFailed") || !strcmp(err, "ClientGone")) ret = OK;
        else if (w.streaming) (void)out_error(&w.out, w.status, err);
        else (void)respond_error(c, &h, w.status, FMT_OPENAI, err);
    }
    vec_free(&w.ids);
    lookup_free(&w.lk);
    feed_unref(e, w.fed);
    split_free(&w.split);
    buf_free(&w.out.ev);
    arena_reset(&a);
    free(c);
    return ret;
}

// ---- prefill ahead ------------------------------------------------------------------------------

// Tokens per chunk of an ahead read: a request that arrives waits for one at most (about 0.8 s on
// the Coder). Prefill is bitwise the same whatever the chunk boundaries, for chunks of more than
// MODEL_SMALL tokens (src/test/ahead.c), so these chunks change no computed row.
#define AHEAD_CHUNK ((size_t)2048)
// The fewest new tokens an ahead read starts for.
#define AHEAD_MIN ((size_t)64)
// Stable tokens left unread at the end of what the partial message gives: its last pre-token may
// still merge with the text that follows (a newline with the next line's indentation).
#define AHEAD_MARGIN ((size_t)16)

bool server_engine_feeds(const Engine *e) {
    return model_info(e->m)->chat == CHAT_QWEN && e->im_start != UINT32_MAX && !model_info(e->m)->no_spec && feed_on();
}

Err server_engine_feed(Engine *e, const char *session, Str key, Str text) {
    if (model_info(e->m)->chat != CHAT_QWEN || e->im_start == UINT32_MAX || model_info(e->m)->no_spec) return "FeedUnsupported";
    if (!feed_on() || !session || !session[0] || strlen(session) >= sizeof e->feed[0].session) return OK;
    // the JSON-escaped text (chat_json_string, as the template writes a string inside an argument
    // written as JSON), without its quotes, tokenized on the caller's thread
    Buf esc = {0};
    chat_json_string(&esc, text);
    VecU32 ids = {0};
    Err err = esc.len >= 2 ? tokenizer_encode_text(e->tok, str_n(esc.ptr + 1, esc.len - 2), &ids) : OK;
    buf_free(&esc);
    if (err) {
        vec_free(&ids);
        return err;
    }
    if (ids.len > FEED_TEXT_TOKENS) { // its head, in memory of its size
        ids.ptr = xrealloc(ids.ptr, FEED_TEXT_TOKENS * sizeof *ids.ptr);
        ids.len = ids.cap = FEED_TEXT_TOKENS;
    }
    uint64_t k = hash_bytes(key.ptr, key.len);
    pthread_mutex_lock(&e->feed_mu);
    Feed *f = NULL;
    for (size_t i = 0; i < FEED_SESSIONS && !f; i++)
        if (e->feed[i].session[0] && !strcmp(e->feed[i].session, session)) f = &e->feed[i];
    if (!f) { // a free slot, else the least recently fed session's
        f = &e->feed[0];
        for (size_t i = 0; i < FEED_SESSIONS; i++) {
            if (!e->feed[i].session[0]) {
                f = &e->feed[i];
                break;
            }
            if (e->feed[i].used < f->used) f = &e->feed[i];
        }
        feed_release(f);
        snprintf(f->session, sizeof f->session, "%s", session);
    }
    f->used = ++e->feed_clock;
    // the key's old text goes, the new one is the newest, and the oldest go while over the bound
    for (size_t i = 0; i < f->texts.len; i++)
        if (f->texts.ptr[i].key == k) {
            f->ntok -= f->texts.ptr[i].ids.len;
            vec_free(&f->texts.ptr[i].ids);
            memmove(f->texts.ptr + i, f->texts.ptr + i + 1, (f->texts.len - i - 1) * sizeof *f->texts.ptr);
            f->texts.len--;
            break;
        }
    size_t drop = 0;
    while (drop < f->texts.len && f->ntok + ids.len > FEED_TOKENS) f->ntok -= f->texts.ptr[drop++].ids.len;
    for (size_t i = 0; i < drop; i++) vec_free(&f->texts.ptr[i].ids);
    if (drop > 0) memmove(f->texts.ptr, f->texts.ptr + drop, (f->texts.len - drop) * sizeof *f->texts.ptr);
    f->texts.len -= drop;
    if (ids.len > 0) {
        vec_push(&f->texts, ((FeedText){k, ids}));
        f->ntok += ids.len;
    } else vec_free(&ids);
    feed_index(f);
    pthread_mutex_unlock(&e->feed_mu);
    return OK;
}

void server_engine_feed_clear(Engine *e, const char *session) {
    if (!session || !session[0]) return;
    pthread_mutex_lock(&e->feed_mu);
    for (size_t i = 0; i < FEED_SESSIONS; i++)
        if (e->feed[i].session[0] && !strcmp(e->feed[i].session, session)) feed_release(&e->feed[i]);
    pthread_mutex_unlock(&e->feed_mu);
}

size_t server_engine_feed_drafts(const Engine *e) { return e->last_fed; }
size_t server_engine_grammar_replaced(const Engine *e) { return e->last_replaced; }

size_t server_engine_feed_tokens(Engine *e, const char *session, size_t *texts, size_t *bytes) {
    pthread_mutex_lock(&e->feed_mu);
    size_t n = 0, nt = 0, b = 0;
    for (size_t i = 0; i < FEED_SESSIONS; i++) {
        const Feed *f = &e->feed[i];
        if (session && f->session[0] && !strcmp(f->session, session)) n = f->ntok, nt = f->texts.len;
        for (size_t j = 0; j < f->texts.len; j++) b += f->texts.ptr[j].ids.cap * sizeof(uint32_t);
        b += f->texts.cap * sizeof(FeedText);
        if (f->ix) b += sizeof *f->ix + f->ix->lk.hist.cap * sizeof(uint32_t) + f->ix->lk.cap * (sizeof(uint64_t) + sizeof(uint32_t) + 1);
    }
    pthread_mutex_unlock(&e->feed_mu);
    if (texts) *texts = nt;
    if (bytes) *bytes = b;
    return n;
}

static bool ahead_on(void) {
    const char *v = getenv("SAETTA_AHEAD");
    if (v && v[0] == '0') return false;
    const char *pv = getenv("SAETTA_POINTS");
    return !(pv && pv[0] == '0');
}

// The tool message of body's request whose tool_call_id is id (the last one), or NULL.
static JValue *partial_message(JValue *r, Str id) {
    JValue *msgs = field(r, "messages");
    if (!msgs || msgs->kind != J_ARRAY) return NULL;
    for (size_t i = msgs->arr.len; i-- > 0;) {
        JValue *m = msgs->arr.items[i];
        JValue *ro = field(m, "role"), *tc = field(m, "tool_call_id");
        if (ro && ro->kind == J_STRING && str_eqz(ro->s, "tool") && tc && tc->kind == J_STRING && str_eq(tc->s, id)) return m;
    }
    return NULL;
}

Err server_engine_ahead(Engine *e, Str body, const char *session, const char *partial_id, volatile int *stop, size_t *read) {
    *read = 0;
    if (model_info(e->m)->chat != CHAT_QWEN || e->im_start == UINT32_MAX) return "AheadUnsupported";
    if (!ahead_on() || !session || !session[0] || strlen(session) >= sizeof e->owner[0] || !partial_id) return OK;
    uint64_t t0 = now_ns();
    Arena a = {0};
    VecU32 ids = {0}, ids2 = {0};
    Err err = OK;
    const char *why = NULL;
    // the prompt with the partial message as it is, and with one more character in it: their common
    // tokens are the ones no continuation of that message changes (as far as two of them tell; the
    // request decides in the end), less a margin
    JValue *r, *m;
    Str model_name, named, p1, p2;
    ToolChoiceKind choice;
    if (request_object(e, &a, body, FMT_OPENAI, &r, &model_name, &choice, &named)) goto out;
    if (model_name.len > 0 && model_lora_find(e->m, arena_strndup(&a, model_name.ptr, model_name.len)) >= 0) goto out;
    m = partial_message(r, str(partial_id));
    JValue *ct = m ? field(m, "content") : NULL;
    if (!ct || ct->kind != J_STRING) goto out;
    if (chat_render_as(&a, CHAT_QWEN, r, &p1)) goto out;
    Str more = str(arena_printf(&a, "%.*s\x01", (int)ct->s.len, ct->s.ptr));
    json_put(&a, m, S("content"), json_string(&a, more));
    if (chat_render_as(&a, CHAT_QWEN, r, &p2)) goto out;
    if (tokenizer_encode(e->tok, p1, &ids) || tokenizer_encode(e->tok, p2, &ids2)) goto out;
    size_t k = 0;
    while (k < MIN(ids.len, ids2.len) && ids.ptr[k] == ids2.ptr[k]) k++;
    if (k <= AHEAD_MARGIN + AHEAD_MIN) goto out;
    size_t target = k - AHEAD_MARGIN;
    uint64_t t1 = now_ns();

    // the engine's turn, given up when the caller stops meanwhile
    uint64_t t = turn_take(e);
    while (!turn_wait(e, t, 50))
        if (stop && *stop) {
            turn_leave(e, t);
            goto out;
        }
    // a request waiting for its turn goes first
    if ((stop && *stop) || turns_waiting(&e->turns)) {
        why = stop && *stop ? "stopped" : "a request waits";
        goto done;
    }
    // the conversation this session's last request ran in, unused by any other request since, with
    // one of its restore points (that request's prompt, or the prompt and its reply) a prefix of the
    // prompt: a snapshot taken before that request ended continues neither and is refused
    // (the active one only: activating another would trim the active one's checkpoints, taking from
    // a conversation the read does not serve)
    size_t ci = e->active;
    if (!(ci < e->n_conv && !strcmp(e->owner[ci], session) && e->owner_gen[ci] == e->conv_gen[ci] && e->conv_lora[ci] == -1)) {
        why = "the conversation in use is not this session's, or was used since";
        goto done;
    }
    const Ckpt *ck = ck_of(e, ci);
    Reusable ru = session_reusable_in(ck, ids.ptr, k);
    if (ru.n == 0) {
        why = "the prompt does not continue the last request";
        goto done;
    }
    model_lora_select(e->m, -1);
    Session *s = e->s;
    // room in the conversation's capacity, grown only while memory allows without emptying another
    if (target + CAP_SLACK > s->cap) {
        size_t cap = MIN(s->max_cap, ALIGN_UP(target + CAP_SLACK, CAP_STEP));
        if (cap > s->cap && room_for(e, session_kv_bytes(s, cap), 2ull << 30)) (void)session_grow(s, e->g, cap);
        if (target + CAP_SLACK > s->cap) target = s->cap > CAP_SLACK ? s->cap - CAP_SLACK : 0;
    }
    Start st = plan(e, ids.ptr, k, false);
    if (st.kind == START_FRESH || st.kind == START_SHARED || st.n < ru.n || target < st.n + AHEAD_MIN) {
        why = target < st.n + AHEAD_MIN ? NULL : "no start";
        goto done;
    }
    size_t from = st.n;
    e->live_ids.len = 0;
    vec_reserve(&e->live_ids, target);
    memcpy(e->live_ids.ptr, ids.ptr, target * sizeof(uint32_t));
    e->live_ids.len = target;
    if (apply(e, st, false)) goto done;
    // the live state is at from, restored or continued
    e->live = true;
    e->live_conv = ci;
    e->live_pos = from;
    // the checkpoints the request would keep: at its start (restore point 1, the reply's end) and at
    // the message starts it reads (session_point_cuts; new buffers only while memory allows, without
    // taking other conversations' checkpoints)
    size_t pb = session_point_bytes(s);
    if (st.kind == START_CKPT) {
        if (session_save_point(s, e->g, ids.ptr, from, room_for(e, pb, 2ull << 30))) goto broken;
        int kp = session_point_at(s->pts, from);
        if (kp >= 0) s->pts[kp].start = true;
    }
    size_t cap = (k - from) / PT_GAP + 4;
    size_t *cuts = ARENA_ARRAY(&a, size_t, cap);
    size_t nc = session_point_cuts(ids.ptr, k, from, e->im_start, cuts, cap);
    // (each segment at least AHEAD_MIN long, also when the capacity lowered the target)
    while (nc > 0 && cuts[nc - 1] + AHEAD_MIN > target) nc--;
    // a checkpoint saved at a cut holds only once the chunk after it ran: that chunk writes the MTP
    // head's row at the cut (the stream before it with the token at it), which a restart there does
    // not rewrite; until then it is kept invalid (a read that stops at the cut leaves the live state)
    int pending = -1;
    size_t pos = from, c = 0;
    float *lg;
    while (pos < target) {
        if ((stop && *stop) || turns_waiting(&e->turns)) break;
        // the segment to the next cut, in equal chunks of at most AHEAD_CHUNK (none of MODEL_SMALL
        // tokens or fewer: segments are at least PT_GAP - AHEAD_MARGIN long, or AHEAD_MIN)
        size_t end = c < nc ? cuts[c] : target, len = end - pos, chunks = (len + AHEAD_CHUNK - 1) / AHEAD_CHUNK;
        size_t T = (len + chunks - 1) / chunks;
        if (model_prefill_with(e->m, s, ids.ptr + pos, T, NULL, NULL, &lg)) goto broken;
        if (pending >= 0) s->pts[pending].valid = true;
        pending = -1;
        pos += T;
        e->live_pos = pos;
        if (pos == end && c < nc) {
            c++;
            if (session_save_point(s, e->g, ids.ptr, pos, room_for(e, pb, 2ull << 30))) goto broken;
            pending = session_point_at(s->pts, pos);
            if (pending >= 0) s->pts[pending].valid = false;
        }
    }
    *read = pos - from;
    dbg("ahead: conversation %zu read %zu tokens (%zu to %zu of %zu known) in %lld ms (render %lld ms)%s\n", ci + 1, pos - from, from,
        pos, k, (long long)((now_ns() - t1) / 1000000), (long long)((t1 - t0) / 1000000), pos < target ? ", yielded" : "");
    goto done;
broken:
    // a failed read leaves no live state (the next request restores as usual)
    e->live = false;
    dbg("ahead: failed\n");
done:
    turn_done(e);
    if (why) dbg("ahead: nothing read (%s)\n", why);
out:
    vec_free(&ids);
    vec_free(&ids2);
    arena_reset(&a);
    return err;
}

Err server_engine_connection(Engine *e, int fd, void (*authed)(void *ctx), void *ctx) { return connection(e, fd, false, authed, ctx); }

void server_engine_set_token(Engine *e, const char *token) {
    snprintf(e->token, sizeof e->token, "Bearer %s", token ? token : "");
    if (!token || !*token) e->token[0] = 0;
}

Err server_serve(Arena *arena, const char *dir, uint16_t port, const char *const *loras, size_t nloras,
                 const char *kwargs_default) {
    JValue *kwd = NULL;
    if (kwargs_default) {
        if (json_parse(arena, str(kwargs_default), true, &kwd) || kwd->kind != J_OBJECT) {
            dbg("saetta: --default-chat-template-kwargs wants a JSON object, got %s\n", kwargs_default);
            return "BadKwargs";
        }
    }
    // one engine per machine: not beside a Saetta Code that runs the model in its process
    int ipid = 0;
    if (instance_inproc_running(&ipid)) {
        dbg("saetta: Saetta Code (pid %d) is running the model in its own process; one model runs at a time.\n"
            "Close it, or start it after the server (saetta start code joins a running server).\n", ipid);
        return "EngineBusy";
    }
    // one server per machine (src/instance.h); the lock is held until this process ends
    int lock_fd;
    TRY(instance_acquire(port, &lock_fd));
    // the port is bound before the load, so a port in use fails at once instead of after
    // loading the weights; listen comes after the load, so until then connections are refused
    int ls = socket(AF_INET, SOCK_STREAM, 0);
    if (ls < 0) return err_errno(errno);
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in sa = {0};
    sa.sin_family = AF_INET;
    sa.sin_port = htons(port);
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(ls, (struct sockaddr *)&sa, sizeof sa) != 0) {
        Err err = err_errno(errno);
        close(ls);
        return err;
    }
    Engine *e;
    Err oe = engine_open(arena, dir, loras, nloras, kwd, port, &e);
    if (oe) {
        close(ls);
        return oe;
    }
    uint64_t t0 = e->opened_ns;
    TRY(stats_spawn(e->st));
    TRY(http_guard_spawn());
    if (listen(ls, 128) != 0) {
        Err err = err_errno(errno);
        close(ls);
        return err;
    }
    dbg("saetta: serving %s on http://127.0.0.1:%u (loaded in %llu ms)\n", base_id, (unsigned)port,
        (unsigned long long)((now_ns() - t0) / 1000000));
    for (;;) {
        int fd = accept(ls, NULL, NULL);
        if (fd < 0) {
            if (errno == EINTR) continue;
            dbg("accept: %s\n", err_errno(errno));
            continue;
        }
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
        Err err = connection(e, fd, true, NULL, NULL);
        if (err) dbg("connection: %s\n", err);
        close(fd);
    }
}
