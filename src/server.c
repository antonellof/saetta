// HTTP server (see server.h).
#include "server.h"

#include <errno.h>
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
#include "split.h"
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

typedef struct Engine {
    Gpu *g;
    Model *m;
    Session *s;
    const Tokenizer *tok;
    QgenPrng rng;
    uint32_t think_end;
    // Speculation statistics kept across requests: a short reply then starts from what the
    // session has learned instead of the optimistic defaults (deep drafts with poor
    // acceptance cost most on short replies).
    Gate gate;
    Depth depth;
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
    // One request at a time, in arrival order: the in-process caller (server_engine_chat, from
    // any thread: Saetta Code's agent and its subagents) and the connections of the Saetta Codes
    // that share this process's engine (server_engine_connection, one thread each) take tickets
    // (turn_take) and run when theirs is served (turn_wait); a waiter that leaves gives its turn
    // up (turn_leave). Only the engine work runs in a turn: requests are read before.
    Turns turns;
    // server_engine_set_token: the token every request of a shared connection must carry
    char token[80];
} Engine;

static uint64_t turn_take(Engine *e) { return turns_take(&e->turns); }
static bool turn_wait(Engine *e, uint64_t t, int ms) { return turns_wait(&e->turns, t, ms); }
static void turn_done(Engine *e) { turns_done(&e->turns); }
static void turn_leave(Engine *e, uint64_t t) { turns_leave(&e->turns, t); }
// Makes conversation i the active one.
static void activate(Engine *e, size_t i) {
    if (i == e->active) return;
    session_swap_conv(e->s, &e->held[i]);
    e->held[e->active] = e->held[i];
    e->has[e->active] = true;
    memset(&e->held[i], 0, sizeof(Conv));
    e->has[i] = false;
    e->active = i;
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

// Picks the conversation for a prompt: the one it continues; else an empty one, a new
// one while GPU memory allows (4 GB kept free under the working-set limit), or the least
// recently used.
static void route(Engine *e, const uint32_t *ids, size_t nids, int lora) {
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
        size_t lru = 0;
        for (size_t i = 0; i < e->n_conv; i++)
            if (e->used[i] < e->used[lru]) lru = i;
        best = lru;
    }
picked:
    activate(e, best);
    if (e->conv_lora[best] != lora) {
        // its state was built with another adapter: start it over
        session_invalidate_from(e->s, 0);
        e->conv_lora[best] = lora;
    }
    model_lora_select(e->m, lora);
    e->used[best] = e->clock;
    e->last_s[best] = (int64_t)(wall_ns() / 1000000000ull);
    atomic_store_explicit(&e->st->convs, (uint32_t)e->n_conv, memory_order_release);
}

typedef enum { START_FRESH, START_CKPT, START_OWN, START_SHARED } StartKind;
typedef struct {
    size_t n;
    StartKind kind;
    size_t src, common;
} Start;

// Where the active conversation starts reading ids: a restore point it extends, else its
// own prefix point, else another conversation's prefix point (KV rows copied), else 0.
static Start plan(Engine *e, const uint32_t *ids, size_t nids) {
    size_t r = session_reusable(e->s, ids, nids);
    if (r > 0) return (Start){r, START_CKPT, 0, 0};
    size_t own = session_shared_point(e->s->ptoks.ptr, e->s->ptoks.len, e->s->pts, ids, nids);
    Start best = {own, own > 0 ? START_OWN : START_FRESH, 0, 0};
    for (size_t i = 0; i < e->n_conv; i++) {
        if (!e->has[i] || e->conv_lora[i] != e->conv_lora[e->active]) continue;
        const Conv *cv = &e->held[i];
        size_t n = session_shared_point(cv->ptoks.ptr, cv->ptoks.len, cv->pts, ids, nids);
        if (n > best.n) {
            size_t lcp = 0;
            while (lcp < MIN(cv->ptoks.len, nids) && cv->ptoks.ptr[lcp] == ids[lcp]) lcp++;
            best = (Start){n, START_SHARED, i, lcp};
        }
    }
    return best;
}

static Err apply(Engine *e, Start st) {
    switch (st.kind) {
    case START_CKPT: session_restore(e->s); break;
    case START_OWN: session_start_at_own_point(e->s, st.n); break;
    case START_SHARED:
        TRY(session_start_from(e->s, e->g, &e->held[st.src], st.n));
        dbg("saetta: conversation %zu starts from the first %zu tokens of conversation %zu (%zu in common)\n",
            e->active + 1, st.n, st.src + 1, st.common);
        break;
    case START_FRESH: session_reset(e->s); break;
    }
    session_invalidate_from(e->s, st.n);
    return OK;
}

// Empties the least recently used inactive conversation that holds more than the starting
// capacity (its buffers shrink back to CAP0, its restore and prefix points are dropped).
static bool evict_one(Engine *e) {
    bool found = false;
    size_t pick = 0;
    for (size_t i = 0; i < e->n_conv; i++) {
        if (!e->has[i]) continue;
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
    while (!room_for(e, session_kv_bytes(e->s, cap), 2ull << 30))
        if (!evict_one(e)) break;
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

// Room for one more prefix point (2 GB kept free under the GPU working-set limit).
static bool point_room(Engine *e) { return room_for(e, 128ull << 20, 2ull << 30); }

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
    TRY(chat_render(a, r, &prompt));
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

// Per-request allocations outside the arena, freed by the caller of handle.
typedef struct {
    VecU32 ids;
    Lookup lk;
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

// Prefill hook: a prefix point every PT_EVERY tokens, the monitor's progress, a stream's
// keep-alive, the client's disconnect.
typedef struct {
    Engine *e;
    Out *out;
    Conn *c;
    const uint32_t *ids;
    bool points;
} Hook;

static bool hook_aligned(void *ctx) { return ((Hook *)ctx)->points; }
static Err hook_chunk_done(void *ctx, size_t pos) {
    Hook *h = ctx;
    if (h->points && pos % PT_EVERY == 0 && pos / PT_EVERY <= PT_MAX) {
        size_t k = pos / PT_EVERY - 1;
        if (h->e->s->pts[k].gdn != NULL || point_room(h->e)) TRY(session_save_point(h->e->s, h->e->g, h->ids, pos));
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
    (void)r;
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
    TRY(apply(e, start));
    const char *pv = getenv("SAETTA_POINTS");
    bool points_on = pv ? pv[0] != '0' : true;
    if (n - 1 > reused) {
        Hook hk = {e, out, c, ids, points_on};
        PrefillHook ph = {&hk, hook_aligned, hook_chunk_done};
        float *lg;
        TRY(model_prefill_with(e->m, e->s, ids + reused, n - 1 - reused, NULL, &ph, &lg));
    }
    if (n - 1 != reused) TRY(session_save_checkpoint(e->s, ids, n - 1));
    float *logits;
    TRY(model_prefill(e->m, e->s, ids + n - 1, 1, NULL, &logits));
    uint64_t t1 = now_ns();
    atomic_store_explicit(&e->st->prompt_done, (uint32_t)n, memory_order_release);
    atomic_store_explicit(&e->st->decode_start_ns, stats_now_ns(), memory_order_release);
    atomic_store_explicit(&e->st->state, STATE_GENERATING, memory_order_release);

    Splitter *spl = &w->split;
    spl->thinking = thinking;
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
    bool spec_on = sv ? sv[0] != '0' : true;
    // speculation at any temperature: greedy verifies by argmax, sampling by SampleChooser
    bool greedy = sp.temperature <= 0;
    Lookup *lk = &w->lk;
    for (size_t i = 0; i < n; i++) lookup_push(lk, ids[i]);
    Gate *gate = &e->gate;
    Depth *depth = &e->depth;
    uint32_t queue[KMAX];
    size_t qn = 0, qi = 0;
    uint32_t next = sample(e, logits, model_info(e->m)->vocab, sp, cand);
    size_t passes = 0;
    uint32_t *gen_ids = ARENA_ARRAY(a, uint32_t, max_tokens + 1);
    size_t n_gen_ids = 0;
    while (n_gen < max_tokens) {
        uint32_t t = next;
        n_gen += 1;
        atomic_store_explicit(&e->st->gen, (uint32_t)n_gen, memory_order_release);
        gen_ids[n_gen_ids++] = t;
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
        size_t nd = spec_on ? lookup_draft(lk, drafts, gate_depth(gate, KMAX - 1)) : 0;
        bool from_mtp = false;
        if (spec_on && nd < 3 && model_has_mtp(e->m)) {
            TRY(model_draft_mtp(e->m, e->s, t, depth_pick(depth, KMAX - 1), drafts, &nd));
            from_mtp = nd > 0;
        } else if (nd > 0 && !gate_allow(gate)) nd = 0;
        if (nd > 0) {
            SpecResult sr;
            if (greedy) TRY(model_spec_greedy(e->m, e->s, t, drafts, nd, &sr));
            else {
                SampleChooser ch = {&e->rng, sp, cand, model_info(e->m)->vocab};
                TRY(model_spec_verify(e->m, e->s, t, drafts, nd, server_sample_pick, &ch, &sr));
            }
            if (from_mtp) depth_record(depth, nd, sr.accepted, round_ms(tr0));
            else gate_record(gate, sr.accepted + 1);
            memcpy(queue, drafts, sr.accepted * sizeof(uint32_t));
            queue[sr.accepted] = sr.next;
            qn = sr.accepted + 1;
            next = queue[0];
            qi = 1;
        } else {
            TRY(model_step(e->m, e->s, t, &logits));
            next = sample(e, logits, model_info(e->m)->vocab, sp, cand);
            if (spec_on && model_has_mtp(e->m)) depth_record(depth, 0, 0, round_ms(tr0));
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
    }
    int64_t pre_ms = (int64_t)(t1 - t0) / 1000000;
    int64_t dec_ms = (int64_t)(t2 - t1) / 1000000;
    Parsed parsed = toolcall_parse(a, str_trim(split_answer(spl)), tools);
    if (parsed.n_calls > 0) finish = "tool_calls";
    dbg("request: %zu prompt tokens (%zu reused) in %lld ms, %zu generated in %lld ms (%s tok/s, %zu passes), finish %s%s, "
        "conversation %zu/%zu\n",
        n, reused, (long long)pre_ms, n_gen, (long long)dec_ms, fx((double)n_gen * 1000.0 / (double)MAX(dec_ms, (int64_t)1), 2),
        passes, finish, fmt == FMT_ANTHROPIC ? " (anthropic)" : "", e->active + 1, e->n_conv);
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

static Err handle(Engine *e, Arena *a, Conn *c, Req *h, Str body, Fmt fmt, Work *w) {
    JValue *r;
    Str model_name;
    TRY(parse_request(a, body, fmt, &r, &model_name));
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
    Str prompt;
    TRY(chat_render(a, r, &prompt));
    Str forced = S("");
    if (choice == TC_REQUIRED || choice == TC_NAMED) {
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
    route(e, w->ids.ptr, n, lora);
    reply_id = lora >= 0 ? model_lora_name(e->m, (size_t)lora) : base_id;
    TRY(ensure(e, n + max_tokens + CAP_SLACK));
    Start start = plan(e, w->ids.ptr, n);
    stats_begin(e->st, fmt == FMT_ANTHROPIC, n, start.n, max_tokens);
    Err err = generate_reply(e, a, c, h, w, r, model_name, fmt, forced, thinking, sp, stops, n_stops, stream, tools, id, created,
                             start, max_tokens);
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
            buf_free(&w.split.text);
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
        const char *path = eq + 1;
        struct stat lst;
        if (stat(path, &lst) != 0) {
            // a `saetta download` name (27b-pi), else a Hugging Face id (org/repo, as vLLM
            // takes it): its folder under the models folder
            const DownloadPack *dp = download_find(path);
            char *dir;
            if (dp && !download_pack_dir(arena, dp, &dir)) path = dir;
            else {
                char *md;
                TRY(download_models_dir(arena, &md));
                path = path_join(arena, md, path);
            }
        }
        Err le = model_lora_load(arena, m, name, path);
        if (le) {
            dbg("saetta: LoRA %s from %s: %s\n", name, path, le);
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
            // (session_grow), so the last step holds about twice the final size
            size_t fit = max_cap;
            while (fit > c0 && 2 * session_conv_bytes(e->s, fit) - have > budget) fit -= SESSION_CTX_ALIGN;
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
    if (!tokenizer_token_id(tok, S("</think>"), &e->think_end)) return "BadTokenizer";
    e->gate = GATE_INIT;
    e->depth = DEPTH_INIT;
    e->n_conv = 1;
    e->active = 0;
    for (size_t i = 0; i < MAX_CONV; i++) e->conv_lora[i] = -1;
    e->kwargs_default = kwd;
    turns_init(&e->turns);
    st->ctx = e->s->max_cap;
    e->opened_ns = t0;
    *out = e;
    return OK;
}

Err server_engine_open(Arena *arena, const char *dir, const char *kwargs_default, Engine **out) {
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
    return engine_open(arena, dir, NULL, 0, kwd, 0, out);
}

const char *server_engine_model(const Engine *e) { return base_id; }

size_t server_engine_context(const Engine *e) { return e->s->max_cap; }

Stats *server_engine_stats(Engine *e) { return e->st; }

Err server_engine_chat(Engine *e, Str body, int (*sink)(void *ctx, int status, const void *p, size_t n), void *ctx) {
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
    buf_free(&w.split.text);
    buf_free(&w.out.ev);
    arena_reset(&a);
    free(c);
    return ret;
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
