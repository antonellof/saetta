// Saetta Code's rpc mode (see code_rpc.h): pi's command set over JSON lines, the responses and the
// session's events as pi writes them.
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi v1.5.2,
// b4afddd): packages/coding-agent/src/modes/rpc/rpc-mode.ts (runRpcMode, handleCommand,
// handleInputLine), modes/rpc/jsonl.ts (attachJsonlLineReader, serializeJsonLine),
// core/agent-session.ts (prompt's queueing, steer, followUp, _queueSteer, _queueFollowUp,
// clearQueue, _emitQueueUpdate, the queue's removal in _handleAgentEvent, abort, setModel,
// _cycleAvailableModel, setThinkingLevel, cycleThinkingLevel, getAvailableThinkingLevels,
// setSteeringMode, setFollowUpMode, compact, executeBash, recordBashResult, abortBash,
// setSessionName, getUserMessagesForForking, getSessionStats, getContextUsage,
// getLastAssistantText), core/bash-executor.ts (executeBashWithOperations), utils/ansi.ts
// (stripAnsi, splitIncompleteAnsiSuffix), utils/shell.ts (sanitizeBinaryOutput),
// core/session-manager.ts (getEntries, getTree), core/provider-composer.ts (modelFromJson,
// mergeCompat), packages/ai/src/models.ts (getSupportedThinkingLevels, clampThinkingLevel).
//
// Not ported: extensions (Saetta Code runs none), so no extension_ui_request is ever written and an
// extension_ui_response is read and dropped, as pi does with no extension asking; images (no vision
// input); prompt templates (Saetta Code has none: get_commands lists its /mcp command and the
// skills).
#include "code_rpc.h"
#include "code_autocompact.h"
#include "code_cfg.h"
#include "code_cmds.h"
#include "code_durable.h"
#include "code_oai.h"
#include "code_res.h"
#include "code_sessions.h"

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

enum { JOB_NONE, JOB_PROMPT, JOB_COMPACT };

typedef VEC(char *) Texts;

static struct {
    bool on;
    const CodeRpcOps *ops;
    CodeUiHost *h;
    pthread_mutex_t lock, out;
    pthread_cond_t job_cv, idle_cv;
    int job;
    char *job_text;
    JValue *job_id;   // the prompt's or compact command's id (kept in persist)
    bool job_started; // the prompt's response is written
    bool streaming, compacting;
    Texts steer, follow;           // the agent's queues (agent.steer / agent.followUp)
    Texts disp_steer, disp_follow; // AgentSession._steeringMessages / _followUpMessages
    bool steer_all, follow_all;    // steeringMode / followUpMode "all"
    VEC(volatile int *) bash_aborts; // the running bash commands' abort flags
    VEC(JValue *) pending_bash;      // bash messages recorded during a run (_pendingBashMessages)
    Arena persist;                   // what a command keeps past its line (never reset)
} R = {.lock = PTHREAD_MUTEX_INITIALIZER, .out = PTHREAD_MUTEX_INITIALIZER, .job_cv = PTHREAD_COND_INITIALIZER,
       .idle_cv = PTHREAD_COND_INITIALIZER};

void code_rpc_unlock(void) {
    if (R.on) pthread_mutex_unlock(&R.lock);
}

void code_rpc_lock(void) {
    if (R.on) pthread_mutex_lock(&R.lock);
}

// ---- output (serializeJsonLine, writeRawStdout) ------------------------------------------------------

static void out_line(const JValue *v) {
    Buf b = {0};
    json_write(&b, v);
    buf_addc(&b, '\n');
    pthread_mutex_lock(&R.out);
    for (size_t off = 0; off < b.len;) {
        ssize_t w = write(1, b.ptr + off, b.len - off);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) break;
        off += (size_t)w;
    }
    pthread_mutex_unlock(&R.out);
    buf_free(&b);
}

static JValue *jstr(Arena *a, const char *s) { return json_string(a, str(s)); }

// A JavaScript number as JSON.stringify writes it.
static JValue *jnum(Arena *a, double x) {
    if (x == floor(x) && fabs(x) < 9007199254740992.0) return json_int(a, (int64_t)x);
    char t[64];
    for (int p = 1; p <= 17; p++) {
        snprintf(t, sizeof t, "%.*g", p, x);
        if (strtod(t, NULL) == x) break;
    }
    JValue *v = json_new(a, J_NUMBER_STRING);
    // %g writes 1e-05 where JavaScript writes 0.00001; the values here (percentages) stay in the
    // plain range, written by fmt_f64
    if (strchr(t, 'e') && fabs(x) >= 1e-6 && fabs(x) < 1e21) fmt_f64(t, sizeof t, x);
    v->s = str(arena_strdup(a, t));
    return v;
}

// { id, type: "response", command, success: true, data } (data NULL: none)
static void ok(Arena *a, const JValue *id, const char *cmd, JValue *data) {
    JValue *o = json_new(a, J_OBJECT);
    if (id) json_put(a, o, S("id"), (JValue *)id);
    json_put(a, o, S("type"), jstr(a, "response"));
    if (cmd) json_put(a, o, S("command"), jstr(a, cmd));
    json_put(a, o, S("success"), json_bool(a, true));
    if (data) json_put(a, o, S("data"), data);
    out_line(o);
}

static void fail(Arena *a, const JValue *id, const char *cmd, const char *msg) {
    JValue *o = json_new(a, J_OBJECT);
    if (id) json_put(a, o, S("id"), (JValue *)id);
    json_put(a, o, S("type"), jstr(a, "response"));
    if (cmd) json_put(a, o, S("command"), jstr(a, cmd));
    json_put(a, o, S("success"), json_bool(a, false));
    json_put(a, o, S("error"), jstr(a, msg));
    out_line(o);
}

static void event(JValue *ev) { out_line(ev); }

static JValue *texts_json(Arena *a, const Texts *t) {
    JValue *arr = json_new(a, J_ARRAY);
    for (size_t i = 0; i < t->len; i++) json_push(a, arr, jstr(a, t->ptr[i]));
    return arr;
}

// _emitQueueUpdate
static void queue_update(Arena *a) {
    JValue *e = json_new(a, J_OBJECT);
    json_put(a, e, S("type"), jstr(a, "queue_update"));
    json_put(a, e, S("steering"), texts_json(a, &R.disp_steer));
    json_put(a, e, S("followUp"), texts_json(a, &R.disp_follow));
    event(e);
}

static void texts_clear(Texts *t) {
    for (size_t i = 0; i < t->len; i++) free(t->ptr[i]);
    t->len = 0;
}

static bool texts_remove(Texts *t, Str text) {
    for (size_t i = 0; i < t->len; i++)
        if (str_eqz(text, t->ptr[i])) {
            free(t->ptr[i]);
            memmove(&t->ptr[i], &t->ptr[i + 1], (t->len - i - 1) * sizeof *t->ptr);
            t->len--;
            return true;
        }
    return false;
}

// contentText(content, ""): the text blocks joined
static void content_text(Buf *b, const JValue *content) {
    if (json_is(content, J_STRING)) {
        buf_addstr(b, content->s);
        return;
    }
    for (size_t i = 0; json_is(content, J_ARRAY) && i < content->arr.len; i++) {
        const JValue *c = content->arr.items[i];
        if (str_eqz(json_strs(json_get(c, "type")), "text")) buf_addstr(b, json_strs(json_get(c, "text")));
    }
}

void code_rpc_event(const JValue *ev) {
    // _handleAgentEvent: a user message that starts leaves the queue it came from first
    if (str_eqz(json_strs(json_get(ev, "type")), "message_start")) {
        const JValue *m = json_get(ev, "message");
        if (str_eqz(json_strs(json_get(m, "role")), "user")) {
            Buf t = {0};
            content_text(&t, json_get(m, "content"));
            Str text = t.ptr ? buf_str(&t) : S("");
            if (texts_remove(&R.disp_steer, text) || texts_remove(&R.disp_follow, text)) {
                Arena a = {0};
                queue_update(&a);
                arena_reset(&a);
            }
            buf_free(&t);
        }
    }
    out_line(ev);
}

char *code_rpc_take_queued(bool steering, bool more) {
    Texts *q = steering ? &R.steer : &R.follow;
    bool all = steering ? R.steer_all : R.follow_all;
    if ((more && !all) || !q->len) return NULL;
    char *t = q->ptr[0];
    memmove(&q->ptr[0], &q->ptr[1], (q->len - 1) * sizeof *q->ptr);
    q->len--;
    return t;
}

bool code_rpc_has_queued(void) { return R.steer.len || R.follow.len; }

static void flush_pending_bash(void);

// _startRun: the bash messages held while idle go first, then onStarted
void code_rpc_prompt_started(void) {
    if (R.job != JOB_PROMPT || R.job_started) return;
    R.job_started = true;
    flush_pending_bash();
    Arena a = {0};
    JValue *d = json_new(&a, J_OBJECT);
    json_put(&a, d, S("disposition"), jstr(&a, "started"));
    ok(&a, R.job_id, "prompt", d);
    arena_reset(&a);
}

// ---- the model (provider-composer.ts modelFromJson, models.ts thinking levels) ---------------------

static const char *const LEVELS[] = {"off", "minimal", "low", "medium", "high", "xhigh", "max"};

static JValue *read_models(Arena *a) { return cfg_read_models(a, R.ops->env(R.h)->agent_dir); }

static const JValue *model_api(const JValue *prov, const JValue *m) {
    const JValue *api = json_get(m, "api");
    return api ? api : json_get(prov, "api");
}

// mergeCompat(provider compat, model compat): the model's keys over the provider's
static JValue *merge_compat(Arena *a, const JValue *base, const JValue *over) {
    if (!json_is(over, J_OBJECT)) return (JValue *)base;
    JValue *o = json_new(a, J_OBJECT);
    for (size_t i = 0; json_is(base, J_OBJECT) && i < base->obj.len; i++) json_put(a, o, base->obj.items[i].key, base->obj.items[i].val);
    for (size_t i = 0; i < over->obj.len; i++) json_put(a, o, over->obj.items[i].key, over->obj.items[i].val);
    return o;
}

static void put_if(Arena *a, JValue *o, const char *k, const JValue *v) {
    if (v) json_put(a, o, str(k), (JValue *)v);
}

// modelFromJson
static JValue *model_from_json(Arena *a, Str provider, const JValue *prov, const JValue *m) {
    JValue *o = json_new(a, J_OBJECT);
    const JValue *id = json_get(m, "id");
    json_put(a, o, S("id"), (JValue *)id);
    json_put(a, o, S("name"), json_get(m, "name") ? json_get(m, "name") : (JValue *)id);
    json_put(a, o, S("api"), (JValue *)model_api(prov, m));
    json_put(a, o, S("provider"), json_string(a, provider));
    json_put(a, o, S("baseUrl"), json_get(m, "baseUrl") ? json_get(m, "baseUrl") : json_get(prov, "baseUrl"));
    json_put(a, o, S("reasoning"), json_get(m, "reasoning") ? json_get(m, "reasoning") : json_bool(a, false));
    put_if(a, o, "thinkingLevelMap", json_get(m, "thinkingLevelMap"));
    JValue *input = json_get(m, "input");
    if (!input) {
        input = json_new(a, J_ARRAY);
        json_push(a, input, jstr(a, "text"));
    }
    json_put(a, o, S("input"), input);
    put_if(a, o, "inputLimits", json_get(m, "inputLimits"));
    JValue *cost = json_get(m, "cost");
    if (!cost) {
        cost = json_new(a, J_OBJECT);
        json_put(a, cost, S("input"), json_int(a, 0));
        json_put(a, cost, S("output"), json_int(a, 0));
        json_put(a, cost, S("cacheRead"), json_int(a, 0));
        json_put(a, cost, S("cacheWrite"), json_int(a, 0));
    }
    json_put(a, o, S("cost"), cost);
    put_if(a, o, "promptCache", json_get(m, "promptCache"));
    json_put(a, o, S("contextWindow"), json_get(m, "contextWindow") ? json_get(m, "contextWindow") : json_int(a, 128000));
    json_put(a, o, S("maxTokens"), json_get(m, "maxTokens") ? json_get(m, "maxTokens") : json_int(a, 16384));
    put_if(a, o, "samplingParams", json_get(m, "samplingParams"));
    put_if(a, o, "samplingParamsByThinkingLevel", json_get(m, "samplingParamsByThinkingLevel"));
    put_if(a, o, "compat", merge_compat(a, json_get(prov, "compat"), json_get(m, "compat")));
    return o;
}

// The model run in this process (--pack), described as `saetta start pi` describes the server.
static JValue *engine_model(Arena *a, const CodeModel *cm) {
    JValue *o = json_new(a, J_OBJECT);
    json_put(a, o, S("id"), jstr(a, cm->id));
    json_put(a, o, S("name"), jstr(a, cm->name ? cm->name : cm->id));
    json_put(a, o, S("api"), jstr(a, "openai-completions"));
    json_put(a, o, S("provider"), jstr(a, cm->provider));
    json_put(a, o, S("baseUrl"), jstr(a, cm->base_url));
    json_put(a, o, S("reasoning"), json_bool(a, cm->reasoning));
    JValue *input = json_new(a, J_ARRAY);
    json_push(a, input, jstr(a, "text"));
    json_put(a, o, S("input"), input);
    JValue *cost = json_new(a, J_OBJECT);
    json_put(a, cost, S("input"), json_int(a, 0));
    json_put(a, cost, S("output"), json_int(a, 0));
    json_put(a, cost, S("cacheRead"), json_int(a, 0));
    json_put(a, cost, S("cacheWrite"), json_int(a, 0));
    json_put(a, o, S("cost"), cost);
    json_put(a, o, S("contextWindow"), json_int(a, (int64_t)cm->context_window));
    json_put(a, o, S("maxTokens"), json_int(a, (int64_t)cm->max_tokens));
    if (cm->compat) json_put(a, o, S("compat"), (JValue *)cm->compat);
    return o;
}

// getAvailableSnapshot: the openai-completions models of models.json whose provider has an API key
// (pi lists a provider only with auth configured), and the in-process model first when there is one.
static JValue *available_models(Arena *a) {
    JValue *out = json_new(a, J_ARRAY);
    const CodeModel *cm = R.ops->model(R.h);
    if (cm->engine) json_push(a, out, engine_model(a, cm));
    const JValue *providers = json_get(read_models(a), "providers");
    for (size_t i = 0; json_is(providers, J_OBJECT) && i < providers->obj.len; i++) {
        const JMember *p = &providers->obj.items[i];
        if (!json_str(json_get(p->val, "apiKey"))) continue;
        const JValue *list = json_get(p->val, "models");
        for (size_t k = 0; json_is(list, J_ARRAY) && k < list->arr.len; k++) {
            const JValue *m = list->arr.items[k];
            if (!json_str(json_get(m, "id")) || !str_eqz(json_strs(model_api(p->val, m)), "openai-completions")) continue;
            json_push(a, out, model_from_json(a, p->key, p->val, m));
        }
    }
    return out;
}

static JValue *find_model(Arena *a, const char *provider, const char *id) {
    JValue *all = available_models(a);
    for (size_t i = 0; i < all->arr.len; i++) {
        const JValue *m = all->arr.items[i];
        if (str_eqz(json_strs(json_get(m, "provider")), provider) && str_eqz(json_strs(json_get(m, "id")), id)) return (JValue *)m;
    }
    return NULL;
}

// session.model
static JValue *current_model(Arena *a) {
    const CodeModel *cm = R.ops->model(R.h);
    if (!cm->id) return NULL;
    if (cm->engine) return engine_model(a, cm);
    JValue *m = find_model(a, cm->provider, cm->id);
    if (m) return m;
    // a model without an API key in models.json: as modelFromJson describes it all the same
    const JValue *providers = json_get(read_models(a), "providers");
    const JValue *p = json_get(providers, cm->provider);
    const JValue *list = json_get(p, "models");
    for (size_t k = 0; json_is(list, J_ARRAY) && k < list->arr.len; k++)
        if (str_eqz(json_strs(json_get(list->arr.items[k], "id")), cm->id)) return model_from_json(a, str(cm->provider), p, list->arr.items[k]);
    return engine_model(a, cm);
}

// getSupportedThinkingLevels
static size_t supported_levels(const JValue *model, const char **out) {
    size_t n = 0;
    if (!model || !(json_is(json_get(model, "reasoning"), J_BOOL) && json_get(model, "reasoning")->b)) {
        out[n++] = "off";
        return n;
    }
    const JValue *map = json_get(model, "thinkingLevelMap");
    for (size_t i = 0; i < COUNT(LEVELS); i++) {
        const JValue *mapped = json_get(map, LEVELS[i]);
        if (mapped && mapped->kind == J_NULL) continue;
        if ((!strcmp(LEVELS[i], "xhigh") || !strcmp(LEVELS[i], "max")) && !mapped) continue;
        out[n++] = LEVELS[i];
    }
    return n;
}

static bool has_level(const char *const *v, size_t n, const char *l) {
    for (size_t i = 0; i < n; i++)
        if (!strcmp(v[i], l)) return true;
    return false;
}

// clampThinkingLevel
static const char *clamp_level(const JValue *model, const char *level) {
    const char *avail[COUNT(LEVELS)];
    size_t n = supported_levels(model, avail);
    if (has_level(avail, n, level)) return level;
    int req = -1;
    for (size_t i = 0; i < COUNT(LEVELS); i++)
        if (!strcmp(LEVELS[i], level)) req = (int)i;
    if (req < 0) return n ? avail[0] : "off";
    for (int i = req; i < (int)COUNT(LEVELS); i++)
        if (has_level(avail, n, LEVELS[i])) return LEVELS[i];
    for (int i = req - 1; i >= 0; i--)
        if (has_level(avail, n, LEVELS[i])) return LEVELS[i];
    return n ? avail[0] : "off";
}

static const char *thinking_now(void) {
    const CodeModel *cm = R.ops->model(R.h);
    return cm->thinking ? cm->thinking : "off";
}

static void thinking_changed(Arena *a, const char *level) {
    JValue *e = json_new(a, J_OBJECT);
    json_put(a, e, S("type"), jstr(a, "thinking_level_changed"));
    json_put(a, e, S("level"), jstr(a, level));
    event(e);
}

// setThinkingLevel: clamped to the model; recorded and announced when it changes from before.
static void set_thinking(Arena *a, const JValue *model, const char *level, const char *before) {
    const char *eff = clamp_level(model, level);
    const char *prev = before ? before : thinking_now();
    bool changing = strcmp(eff, prev) != 0;
    R.ops->set_thinking(R.h, eff, prev);
    if (changing) thinking_changed(a, eff);
}

// settings.json: one top-level key set (settings-manager.ts setSteeringMode, ...), or one key of an
// object member (compaction.enabled, retry.enabled).
typedef struct {
    const char *key, *sub;
    JValue *val;
} SetKey;

static bool set_key(Arena *a, JValue *root, void *ctx) {
    SetKey *k = ctx;
    if (!k->sub) {
        json_put(a, root, str(k->key), k->val);
        return true;
    }
    JValue *o = json_get(root, k->key);
    if (!json_is(o, J_OBJECT)) {
        o = json_new(a, J_OBJECT);
        json_put(a, root, str(k->key), o);
    }
    json_put(a, o, str(k->sub), k->val);
    return true;
}

static void save_setting(const char *key, const char *sub, JValue *val) {
    Arena a = {0};
    Buf why = {0};
    SetKey k = {key, sub, val};
    (void)cfg_json_update(path_join(&a, R.ops->env(R.h)->agent_dir, "settings.json"), set_key, &k, &why);
    buf_free(&why);
    arena_reset(&a);
}

// pi's model switch (setModel, _cycleAvailableModel): the model, a model_change entry, then the
// thinking level for it (settings defaultThinkingLevel, else the current one), clamped.
static bool switch_to(Arena *a, const JValue *model) {
    const char *before = arena_strdup(a, thinking_now());
    if (!R.ops->ui->set_model(R.h, json_str(json_get(model, "provider")), json_str(json_get(model, "id")))) return false;
    JValue *settings = code_settings_merged(a, R.ops->env(R.h));
    const char *dflt = json_str(json_get(settings, "defaultThinkingLevel"));
    set_thinking(a, model, dflt ? dflt : before, before);
    return true;
}

// ---- the agent thread ----------------------------------------------------------------------------

static void flush_pending_bash(void);

static void compaction_job(Arena *a, const char *instructions, const JValue *id);

static void *agent_main(void *arg) {
    (void)arg;
    pthread_mutex_lock(&R.lock);
    for (;;) {
        while (R.job == JOB_NONE) pthread_cond_wait(&R.job_cv, &R.lock);
        int job = R.job;
        char *text = R.job_text;
        R.job_text = NULL;
        if (job == JOB_PROMPT) {
            Err e = R.ops->prompt(R.h, text);
            // a failure before the run started is the prompt's error response (.catch)
            if (e && !R.job_started) {
                Arena a = {0};
                fail(&a, R.job_id, "prompt", e);
                arena_reset(&a);
            }
            flush_pending_bash();
            R.streaming = false;
        } else {
            Arena a = {0};
            compaction_job(&a, text, R.job_id);
            arena_reset(&a);
            R.compacting = false;
        }
        free(text);
        R.job = JOB_NONE;
        pthread_cond_broadcast(&R.idle_cv);
    }
    return NULL;
}

static void wait_idle(void) {
    while (R.streaming || R.compacting) pthread_cond_wait(&R.idle_cv, &R.lock);
}

// a command's id, kept past its line (the response of a job comes later)
static JValue *keep_id(const JValue *id) {
    if (!id) return NULL;
    Buf b = {0};
    json_write(&b, id);
    JValue *v = NULL;
    if (json_parse(&R.persist, buf_str(&b), false, &v)) v = NULL;
    buf_free(&b);
    return v;
}

static void start_job(int job, const char *text, const JValue *id) {
    R.job = job;
    R.job_id = keep_id(id);
    R.job_started = false;
    R.job_text = text ? strdup(text) : NULL;
    if (job == JOB_PROMPT) R.streaming = true;
    else R.compacting = true;
    pthread_cond_signal(&R.job_cv);
}

// ---- compact (AgentSession.compact, manual) --------------------------------------------------------

static void compaction_job(Arena *a, const char *instructions, const JValue *id) {
    CodeEnv *env = R.ops->env(R.h);
    CodeModel *m = R.ops->model(R.h);
    CodeSession *s = R.ops->ui->session(R.h);
    volatile int *ab = env->abort;
    if (ab) *ab = 0;
    JValue *ev = json_new(a, J_OBJECT);
    json_put(a, ev, S("type"), jstr(a, "compaction_start"));
    json_put(a, ev, S("reason"), jstr(a, "manual"));
    event(ev);
    int64_t keep = code_compaction_settings(code_settings_merged(a, env), m->provider, m->id).keep_recent_tokens;
    const JValue *entry = NULL;
    Arena *ra = R.ops->arena(R.h);
    const char *err = code_sessions_compact(ra, s, m, env, instructions && *instructions ? instructions : NULL, keep, &entry);
    bool aborted = (ab && *ab) || (err && !strcmp(err, "aborted"));
    JValue *end = json_new(a, J_OBJECT);
    json_put(a, end, S("type"), jstr(a, "compaction_end"));
    json_put(a, end, S("reason"), jstr(a, "manual"));
    JValue *res = NULL;
    if (!err && entry) {
        res = json_new(a, J_OBJECT);
        json_put(a, res, S("summary"), (JValue *)json_get(entry, "summary"));
        json_put(a, res, S("firstKeptEntryId"), (JValue *)json_get(entry, "firstKeptEntryId"));
        json_put(a, res, S("tokensBefore"), (JValue *)json_get(entry, "tokensBefore"));
        // estimateMessagesTokens(buildSessionProjection().messages)
        JValue *proj = code_sessions_projection(a, code_session_entries(s), code_session_leaf(s));
        int64_t after = 0;
        for (size_t i = 0; i < proj->arr.len; i++) after += code_sessions_estimate(proj->arr.items[i]);
        json_put(a, res, S("estimatedTokensAfter"), json_int(a, after));
        if (json_get(entry, "usage")) json_put(a, res, S("usage"), (JValue *)json_get(entry, "usage"));
        json_put(a, res, S("details"), (JValue *)json_get(entry, "details"));
        json_put(a, end, S("result"), res);
    }
    json_put(a, end, S("aborted"), json_bool(a, !entry && aborted));
    json_put(a, end, S("willRetry"), json_bool(a, false));
    if (err && !aborted) json_put(a, end, S("errorMessage"), jstr(a, arena_printf(a, "Compaction failed: %s", err)));
    event(end);
    if (res) ok(a, id, "compact", res);
    else fail(a, id, "compact", aborted ? "Compaction cancelled" : err ? err : "Compaction failed");
}

// ---- bash (executeBash, executeBashWithOperations, recordBashResult) --------------------------------

// stripAnsi's sequences: the length of the one at s[i], else 0 (ansiRegex: OSC up to its first
// string terminator, or CSI with its parameters and final byte).
static size_t c1_at(Str s, size_t i, uint8_t c) { return i + 1 < s.len && (uint8_t)s.ptr[i] == 0xc2 && (uint8_t)s.ptr[i + 1] == c ? 2 : 0; }

static bool csi_final(uint8_t c) {
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'P') || (c >= 'R' && c <= 'T') || c == 'Z' || c == 'c' || (c >= 'f' && c <= 'n') ||
           (c >= 'q' && c <= 'u') || c == 'y' || c == '=' || c == '>' || c == '<' || c == '~';
}

// CSI_START then its parameters from s[i]: the end, or 0 when no introducer is there; *intro true
// for a real CSI introducer.
static size_t csi_body(Str s, size_t i) {
    size_t j;
    if (i + 1 < s.len && s.ptr[i] == 0x1b && s.ptr[i + 1] == '[' && i + 2 < s.len && strchr("<=>", s.ptr[i + 2]) && s.ptr[i + 2]) j = i + 3;
    else if (c1_at(s, i, 0x9b) && i + 2 < s.len && s.ptr[i + 2] && strchr("<=>", s.ptr[i + 2])) j = i + 3;
    else if (s.ptr[i] == 0x1b || c1_at(s, i, 0x9b)) {
        j = i + (s.ptr[i] == 0x1b ? 1 : 2);
        while (j < s.len && s.ptr[j] && strchr("[]()#;?", s.ptr[j])) j++;
    } else return 0;
    size_t k = j, d = 0;
    while (k < s.len && d < 4 && s.ptr[k] >= '0' && s.ptr[k] <= '9') k++, d++;
    if (d) {
        while (k < s.len && (s.ptr[k] == ';' || s.ptr[k] == ':')) {
            k++;
            for (d = 0; k < s.len && d < 4 && s.ptr[k] >= '0' && s.ptr[k] <= '9'; d++) k++;
        }
        j = k;
    }
    return j;
}

static size_t ansi_at(Str s, size_t i) {
    if (s.ptr[i] != 0x1b && !c1_at(s, i, 0x9b)) return 0;
    if (s.ptr[i] == 0x1b && i + 1 < s.len && s.ptr[i + 1] == ']') {
        for (size_t k = i + 2; k < s.len; k++) {
            if (s.ptr[k] == 0x07) return k + 1 - i;
            if (s.ptr[k] == 0x1b && k + 1 < s.len && s.ptr[k + 1] == '\\') return k + 2 - i;
            if (c1_at(s, k, 0x9c)) return k + 2 - i;
        }
    }
    size_t j = csi_body(s, i);
    if (j && j < s.len && csi_final((uint8_t)s.ptr[j])) return j + 1 - i;
    return 0;
}

// splitIncompleteAnsiSuffix: where an unfinished sequence at the end of s starts (s.len: none).
static size_t unfinished_ansi(Str s) {
    size_t from = s.len > 256 ? s.len - 256 : 0;
    for (size_t i = from; i < s.len; i++) {
        if (s.ptr[i] != 0x1b && !c1_at(s, i, 0x9b)) continue;
        if (s.ptr[i] == 0x1b && i + 1 < s.len && s.ptr[i + 1] == ']') {
            bool term = false;
            for (size_t k = i + 2; k < s.len && !term; k++)
                term = s.ptr[k] == 0x07 || c1_at(s, k, 0x9c) || (s.ptr[k] == 0x1b && k + 1 < s.len && s.ptr[k + 1] == '\\');
            if (!term) return i;
        }
        if (csi_body(s, i) == s.len) return i;
    }
    return s.len;
}

typedef struct {
    Arena *a;
    const JValue *id;
    Buf pend;   // bytes of an unfinished UTF-8 sequence or escape sequence
    Buf all;    // the sanitized text (pi's outputChunks, before the rolling cut)
    size_t total_raw;
    bool discarded;
    char *tmp_path;
    FILE *tmp;
} BashRun;

// a complete UTF-8 prefix of b (TextDecoder's stream mode holds a split character back)
static size_t utf8_complete(const char *p, size_t n) {
    size_t i = n, back = 0;
    while (i > 0 && back < 3 && ((uint8_t)p[i - 1] & 0xc0) == 0x80) i--, back++;
    if (i == 0) return n;
    uint8_t c = (uint8_t)p[i - 1];
    size_t need = c >= 0xf0 ? 4 : c >= 0xe0 ? 3 : c >= 0xc0 ? 2 : 1;
    return need > back + 1 ? i - 1 : n;
}

// appendText: stripAnsi, sanitizeBinaryOutput, \r removed; the rolling buffer, the temp file, the
// bash_execution_update event
static void bash_text(BashRun *b, Str raw) {
    Buf t = {0};
    for (size_t i = 0; i < raw.len;) {
        size_t n = ansi_at(raw, i);
        if (n) {
            i += n;
            continue;
        }
        uint8_t c = (uint8_t)raw.ptr[i];
        bool drop = (c <= 0x08) || c == 0x0b || c == 0x0c || (c >= 0x0e && c <= 0x1f) || c == '\r';
        // U+FFF9..U+FFFB (EF BF B9..BB)
        if (c == 0xef && i + 2 < raw.len && (uint8_t)raw.ptr[i + 1] == 0xbf && (uint8_t)raw.ptr[i + 2] >= 0xb9 && (uint8_t)raw.ptr[i + 2] <= 0xbb) {
            i += 3;
            continue;
        }
        if (!drop) buf_addc(&t, (char)c);
        i++;
    }
    if (!t.len) {
        buf_free(&t);
        return;
    }
    if (b->total_raw > 50 * 1024 && !b->tmp) {
        uint8_t r[8];
        arc4random_buf(r, sizeof r);
        const char *td = getenv("TMPDIR");
        if (!td || !*td) td = "/tmp";
        size_t tl = strlen(td);
        b->tmp_path = arena_printf(b->a, "%.*s/pi-bash-%02x%02x%02x%02x%02x%02x%02x%02x.log", (int)(tl > 1 && td[tl - 1] == '/' ? tl - 1 : tl), td,
                                   r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
        b->tmp = fopen(b->tmp_path, "we");
        if (b->tmp && b->all.len) fwrite(b->all.ptr, 1, b->all.len, b->tmp);
    }
    if (b->tmp) fwrite(t.ptr, 1, t.len, b->tmp);
    buf_add(&b->all, t.ptr, t.len);
    // the rolling buffer keeps about twice the output limit
    const size_t roll = 100 * 1024;
    if (b->all.len > roll * 2) {
        size_t cut = b->all.len - roll;
        while (cut < b->all.len && ((uint8_t)b->all.ptr[cut] & 0xc0) == 0x80) cut++;
        memmove(b->all.ptr, b->all.ptr + cut, b->all.len - cut);
        b->all.len -= cut;
        b->discarded = true;
    }
    Arena a = {0};
    JValue *e = json_new(&a, J_OBJECT);
    json_put(&a, e, S("type"), jstr(&a, "bash_execution_update"));
    if (b->id) json_put(&a, e, S("id"), (JValue *)b->id);
    json_put(&a, e, S("delta"), json_string(&a, code_utf8_clean(&a, buf_str(&t))));
    event(e);
    arena_reset(&a);
    buf_free(&t);
}

static void bash_data(void *ctx, const void *p, size_t n) {
    BashRun *b = ctx;
    b->total_raw += n;
    buf_add(&b->pend, p, n);
    size_t done = utf8_complete(b->pend.ptr, b->pend.len);
    Str all = str_n(b->pend.ptr, done);
    size_t cut = unfinished_ansi(all);
    Str text = str_n(b->pend.ptr, cut);
    Str clean = code_utf8_clean(b->a, text);
    bash_text(b, clean);
    memmove(b->pend.ptr, b->pend.ptr + cut, b->pend.len - cut);
    b->pend.len -= cut;
}

typedef struct {
    char *command;
    JValue *id;
    bool exclude, has_exclude;
} BashJob;

static void record_bash(JValue *msg) {
    CodeSession *s = R.ops->ui->session(R.h);
    Arena *a = R.ops->arena(R.h);
    JValue *f = json_new(a, J_OBJECT);
    json_put(a, f, S("message"), msg);
    (void)code_session_append(s, "message", f);
}

static void flush_pending_bash(void) {
    for (size_t i = 0; i < R.pending_bash.len; i++) record_bash(R.pending_bash.ptr[i]);
    R.pending_bash.len = 0;
}

static void *bash_main(void *arg) {
    BashJob *j = arg;
    Arena a = {0};
    BashRun b = {.a = &a, .id = j->id};
    volatile int abort_flag = 0;
    pthread_mutex_lock(&R.lock);
    vec_push(&R.bash_aborts, &abort_flag);
    CodeEnv env = *R.ops->env(R.h);
    env.cwd = arena_strdup(&a, env.cwd);
    env.session_id = NULL; // getShellEnv: no PI_* session variables
    // settings shellCommandPrefix
    const char *prefix = json_str(json_get(code_settings_merged(&a, &env), "shellCommandPrefix"));
    pthread_mutex_unlock(&R.lock);
    const char *cmd = prefix ? arena_printf(&a, "%s\n%s", prefix, j->command) : j->command;
    bool aborted;
    const char *thrown;
    int code = code_tools_exec(&a, &env, cmd, bash_data, &b, &abort_flag, &aborted, &thrown);
    // flushOutput
    if (b.pend.len) {
        Str rest = code_utf8_clean(&a, buf_str(&b.pend));
        bash_text(&b, rest);
    }
    pthread_mutex_lock(&R.lock);
    for (size_t i = 0; i < R.bash_aborts.len; i++)
        if (R.bash_aborts.ptr[i] == &abort_flag) {
            memmove(&R.bash_aborts.ptr[i], &R.bash_aborts.ptr[i + 1], (R.bash_aborts.len - i - 1) * sizeof *R.bash_aborts.ptr);
            R.bash_aborts.len--;
            break;
        }
    if (thrown && !aborted) {
        fail(&a, j->id, "bash", thrown);
    } else {
        Str full = b.all.ptr ? buf_str(&b.all) : S("");
        bool truncated;
        Str out = code_tools_truncate_tail(full, &truncated);
        if (truncated && !b.tmp_path) { // ensureTempFile
            uint8_t r[8];
            arc4random_buf(r, sizeof r);
            const char *td = getenv("TMPDIR");
            if (!td || !*td) td = "/tmp";
            size_t tl = strlen(td);
            b.tmp_path = arena_printf(&a, "%.*s/pi-bash-%02x%02x%02x%02x%02x%02x%02x%02x.log", (int)(tl > 1 && td[tl - 1] == '/' ? tl - 1 : tl), td,
                                      r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
            b.tmp = fopen(b.tmp_path, "we");
            if (b.tmp) fwrite(full.ptr, 1, full.len, b.tmp);
        }
        if (b.tmp) fclose(b.tmp);
        Arena *ra = R.ops->arena(R.h);
        Str text = truncated ? out : full;
        // BashResult
        JValue *res = json_new(&a, J_OBJECT);
        json_put(&a, res, S("output"), json_string(&a, text));
        if (!aborted) json_put(&a, res, S("exitCode"), json_int(&a, code));
        json_put(&a, res, S("cancelled"), json_bool(&a, aborted));
        json_put(&a, res, S("truncated"), json_bool(&a, b.discarded || truncated));
        if (b.tmp_path) json_put(&a, res, S("fullOutputPath"), jstr(&a, b.tmp_path));
        // recordBashResult: the BashExecutionMessage, kept for the run's end while one streams
        JValue *msg = json_new(ra, J_OBJECT);
        json_put(ra, msg, S("role"), jstr(ra, "bashExecution"));
        json_put(ra, msg, S("command"), jstr(ra, j->command));
        json_put(ra, msg, S("output"), json_string(ra, text));
        if (!aborted) json_put(ra, msg, S("exitCode"), json_int(ra, code));
        json_put(ra, msg, S("cancelled"), json_bool(ra, aborted));
        json_put(ra, msg, S("truncated"), json_bool(ra, b.discarded || truncated));
        if (b.tmp_path) json_put(ra, msg, S("fullOutputPath"), jstr(ra, b.tmp_path));
        json_put(ra, msg, S("timestamp"), json_int(ra, (int64_t)(wall_ns() / 1000000)));
        if (j->has_exclude) json_put(ra, msg, S("excludeFromContext"), json_bool(ra, j->exclude));
        if (R.streaming) vec_push(&R.pending_bash, msg);
        else record_bash(msg);
        ok(&a, j->id, "bash", res);
    }
    pthread_mutex_unlock(&R.lock);
    buf_free(&b.pend);
    buf_free(&b.all);
    arena_reset(&a);
    free(j->command);
    free(j);
    return NULL;
}

// ---- state (get_state, get_session_stats, get_tree) ------------------------------------------------

static JValue *state_json(Arena *a) {
    CodeSession *s = R.ops->ui->session(R.h);
    const CodeModel *cm = R.ops->model(R.h);
    const JValue *entries = code_session_entries(s);
    JValue *o = json_new(a, J_OBJECT);
    JValue *model = current_model(a);
    if (model) json_put(a, o, S("model"), model);
    json_put(a, o, S("thinkingLevel"), jstr(a, thinking_now()));
    json_put(a, o, S("isStreaming"), json_bool(a, R.streaming));
    json_put(a, o, S("isCompacting"), json_bool(a, R.compacting));
    json_put(a, o, S("steeringMode"), jstr(a, R.steer_all ? "all" : "one-at-a-time"));
    json_put(a, o, S("followUpMode"), jstr(a, R.follow_all ? "all" : "one-at-a-time"));
    const char *file = code_session_path(s);
    if (file) json_put(a, o, S("sessionFile"), jstr(a, file));
    json_put(a, o, S("sessionId"), (JValue *)json_get(code_session_header(s), "id"));
    const char *name = code_sessions_name(a, entries);
    if (name) json_put(a, o, S("sessionName"), jstr(a, name));
    CodeCompactionSettings cs = code_compaction_settings(code_settings_merged(a, R.ops->env(R.h)), cm->provider, cm->id);
    json_put(a, o, S("autoCompactionEnabled"), json_bool(a, cs.enabled));
    JValue *msgs = code_sessions_projection(a, entries, code_session_leaf(s));
    json_put(a, o, S("messageCount"), json_int(a, (int64_t)msgs->arr.len));
    json_put(a, o, S("pendingMessageCount"), json_int(a, (int64_t)(R.disp_steer.len + R.disp_follow.len)));
    return o;
}

static JValue *stats_json(Arena *a) {
    CodeSession *s = R.ops->ui->session(R.h);
    const CodeModel *cm = R.ops->model(R.h);
    const JValue *entries = code_session_entries(s);
    CodeSessionStats st;
    code_sessions_stats(entries, &st);
    JValue *o = json_new(a, J_OBJECT);
    const char *file = code_session_path(s);
    if (file) json_put(a, o, S("sessionFile"), jstr(a, file));
    json_put(a, o, S("sessionId"), (JValue *)json_get(code_session_header(s), "id"));
    json_put(a, o, S("userMessages"), json_int(a, (int64_t)st.user));
    json_put(a, o, S("assistantMessages"), json_int(a, (int64_t)st.assistant));
    json_put(a, o, S("toolCalls"), json_int(a, (int64_t)st.tool_calls));
    json_put(a, o, S("toolResults"), json_int(a, (int64_t)st.tool_results));
    json_put(a, o, S("totalMessages"), json_int(a, (int64_t)st.total));
    JValue *t = json_new(a, J_OBJECT);
    json_put(a, t, S("input"), jnum(a, st.input));
    json_put(a, t, S("output"), jnum(a, st.output));
    json_put(a, t, S("cacheRead"), jnum(a, st.cache_read));
    json_put(a, t, S("cacheWrite"), jnum(a, st.cache_write));
    json_put(a, t, S("total"), jnum(a, st.input + st.output + st.cache_read + st.cache_write));
    json_put(a, o, S("tokens"), t);
    json_put(a, o, S("cost"), jnum(a, st.cost));
    // getContextUsage
    if (cm->context_window > 0) {
        JValue *u = json_new(a, J_OBJECT);
        const char *leaf = code_session_leaf(s);
        if (code_sessions_context_unknown(a, entries, leaf)) {
            json_put(a, u, S("tokens"), json_new(a, J_NULL));
            json_put(a, u, S("contextWindow"), json_int(a, (int64_t)cm->context_window));
            json_put(a, u, S("percent"), json_new(a, J_NULL));
        } else {
            double tok = code_sessions_projected_tokens(a, entries, leaf);
            json_put(a, u, S("tokens"), jnum(a, tok));
            json_put(a, u, S("contextWindow"), json_int(a, (int64_t)cm->context_window));
            json_put(a, u, S("percent"), jnum(a, tok / (double)cm->context_window * 100));
        }
        json_put(a, o, S("contextUsage"), u);
    }
    return o;
}

static int64_t iso_ms(Str t) {
    struct tm tm = {0};
    char z[40];
    snprintf(z, sizeof z, "%.*s", (int)MIN(t.len, sizeof z - 1), t.ptr);
    int ms = 0;
    if (sscanf(z, "%d-%d-%dT%d:%d:%d.%dZ", &tm.tm_year, &tm.tm_mon, &tm.tm_mday, &tm.tm_hour, &tm.tm_min, &tm.tm_sec, &ms) < 6) return 0;
    tm.tm_year -= 1900;
    tm.tm_mon -= 1;
    return (int64_t)timegm(&tm) * 1000 + ms;
}

typedef struct {
    const JValue *entry;
    JValue *node, *children;
    int64_t ts;
} TreeNode;

static int by_time(const void *x, const void *y) {
    const TreeNode *a = *(TreeNode *const *)x, *b = *(TreeNode *const *)y;
    return a->ts < b->ts ? -1 : a->ts > b->ts;
}

// SessionManager.getTree: a node per entry (its resolved label), children under their parent
// (orphans as roots), each node's children by time (a stable sort, as Array.prototype.sort).
static JValue *tree_json(Arena *a, const JValue *entries) {
    size_t n = entries->arr.len;
    TreeNode *nodes = arena_alloc(a, (n + 1) * sizeof *nodes);
    size_t k = 0;
    for (size_t i = 0; i < n; i++) {
        const JValue *e = entries->arr.items[i];
        if (str_eqz(json_strs(json_get(e, "type")), "session")) continue;
        TreeNode *t = &nodes[k++];
        t->entry = e;
        t->ts = iso_ms(json_strs(json_get(e, "timestamp")));
        t->node = json_new(a, J_OBJECT);
        t->children = json_new(a, J_ARRAY);
        json_put(a, t->node, S("entry"), (JValue *)e);
        json_put(a, t->node, S("children"), t->children);
    }
    // labels: the latest label entry per target (an empty one removes it)
    for (size_t i = 0; i < k; i++) {
        const JValue *e = nodes[i].entry;
        if (!str_eqz(json_strs(json_get(e, "type")), "label")) continue;
        Str target = json_strs(json_get(e, "targetId"));
        for (size_t j = 0; j < k; j++) {
            if (!str_eq(json_strs(json_get(nodes[j].entry, "id")), target)) continue;
            JValue *nd = nodes[j].node;
            JValue *fresh = json_new(a, J_OBJECT);
            json_put(a, fresh, S("entry"), json_get(nd, "entry"));
            json_put(a, fresh, S("children"), json_get(nd, "children"));
            if (json_strs(json_get(e, "label")).len) {
                json_put(a, fresh, S("label"), json_get(e, "label"));
                json_put(a, fresh, S("labelTimestamp"), json_get(e, "timestamp"));
            }
            *nd = *fresh;
        }
    }
    JValue *roots = json_new(a, J_ARRAY);
    VEC(TreeNode *) *kids = arena_alloc(a, (k + 1) * sizeof *kids);
    memset(kids, 0, (k + 1) * sizeof *kids);
    VEC(TreeNode *) rootv = {0};
    for (size_t i = 0; i < k; i++) {
        const JValue *pid = json_get(nodes[i].entry, "parentId");
        Str p = json_strs(pid);
        ptrdiff_t parent = -1;
        if (json_is(pid, J_STRING) && !str_eq(p, json_strs(json_get(nodes[i].entry, "id"))))
            for (size_t j = 0; j < k && parent < 0; j++)
                if (str_eq(json_strs(json_get(nodes[j].entry, "id")), p)) parent = (ptrdiff_t)j;
        if (parent >= 0) vec_push(&kids[parent], &nodes[i]);
        else vec_push(&rootv, &nodes[i]);
    }
    for (size_t i = 0; i < k; i++) {
        if (kids[i].len > 1) {
            // insertion sort: stable
            for (size_t x = 1; x < kids[i].len; x++)
                for (size_t y = x; y > 0 && by_time(&kids[i].ptr[y - 1], &kids[i].ptr[y]) > 0; y--) {
                    TreeNode *t = kids[i].ptr[y];
                    kids[i].ptr[y] = kids[i].ptr[y - 1];
                    kids[i].ptr[y - 1] = t;
                }
        }
        for (size_t x = 0; x < kids[i].len; x++) json_push(a, nodes[i].children, kids[i].ptr[x]->node);
        vec_free(&kids[i]);
    }
    for (size_t i = 0; i < rootv.len; i++) json_push(a, roots, rootv.ptr[i]->node);
    vec_free(&rootv);
    return roots;
}

static JValue *leaf_json(Arena *a, CodeSession *s) {
    const char *leaf = code_session_leaf(s);
    return leaf ? jstr(a, leaf) : json_new(a, J_NULL);
}

// getLastAssistantText
static const char *last_assistant_text(Arena *a, const JValue *msgs) {
    for (size_t i = msgs->arr.len; i-- > 0;) {
        const JValue *m = msgs->arr.items[i];
        if (!str_eqz(json_strs(json_get(m, "role")), "assistant")) continue;
        const JValue *c = json_get(m, "content");
        if (str_eqz(json_strs(json_get(m, "stopReason")), "aborted") && (!json_is(c, J_ARRAY) || c->arr.len == 0)) continue;
        Buf t = {0};
        for (size_t k = 0; json_is(c, J_ARRAY) && k < c->arr.len; k++)
            if (str_eqz(json_strs(json_get(c->arr.items[k], "type")), "text")) buf_addstr(&t, json_strs(json_get(c->arr.items[k], "text")));
        Str tr = code_js_trim(t.ptr ? buf_str(&t) : S(""));
        const char *out = tr.len ? arena_strndup(a, tr.ptr, tr.len) : NULL;
        buf_free(&t);
        return out;
    }
    return NULL;
}

// get_commands: the extension commands (Saetta Code's built-in /mcp, as pi's built-in mcp
// extension registers it), then the skills.
static JValue *commands_json(Arena *a) {
    JValue *list = json_new(a, J_ARRAY);
    JValue *mcp = json_new(a, J_OBJECT);
    json_put(a, mcp, S("name"), jstr(a, "mcp"));
    json_put(a, mcp, S("description"), jstr(a, "Manage MCP servers: sign in, reconnect, enable or disable, and change exposure"));
    json_put(a, mcp, S("source"), jstr(a, "extension"));
    JValue *si = json_new(a, J_OBJECT);
    json_put(a, si, S("path"), jstr(a, "builtin:mcp"));
    json_put(a, si, S("source"), jstr(a, "builtin"));
    json_put(a, si, S("scope"), jstr(a, "temporary"));
    json_put(a, si, S("origin"), jstr(a, "top-level"));
    json_put(a, mcp, S("sourceInfo"), si);
    json_push(a, list, mcp);
    const CodeResources *res = code_resources(R.ops->env(R.h));
    for (size_t i = 0; res && i < res->nskills; i++) {
        const CodeSkill *k = &res->skills[i];
        JValue *c = json_new(a, J_OBJECT);
        json_put(a, c, S("name"), jstr(a, arena_printf(a, "skill:%.*s", (int)k->name.len, k->name.ptr)));
        json_put(a, c, S("description"), json_string(a, k->description));
        json_put(a, c, S("source"), jstr(a, "skill"));
        JValue *s = json_new(a, J_OBJECT);
        json_put(a, s, S("path"), jstr(a, k->file));
        if (k->si.source) json_put(a, s, S("source"), jstr(a, k->si.source));
        if (k->si.scope) json_put(a, s, S("scope"), jstr(a, k->si.scope));
        if (k->si.origin) json_put(a, s, S("origin"), jstr(a, k->si.origin));
        if (k->si.base_dir) json_put(a, s, S("baseDir"), jstr(a, k->si.base_dir));
        json_put(a, c, S("sourceInfo"), s);
        json_push(a, list, c);
    }
    JValue *o = json_new(a, J_OBJECT);
    json_put(a, o, S("commands"), list);
    return o;
}

// ---- commands (handleCommand) ------------------------------------------------------------------------

static const char *arg_str(const JValue *c, const char *k) { return json_str(json_get(c, k)); }

static bool arg_true(const JValue *c, const char *k) {
    const JValue *v = json_get(c, k);
    return json_is(v, J_BOOL) && v->b;
}

// _queueSteer / _queueFollowUp: the expanded text in both lists, then queue_update
static void queue(Arena *a, bool steering, const char *text) {
    const char *t = code_skill_expand(R.ops->arena(R.h), R.ops->env(R.h), text, NULL);
    vec_push(steering ? &R.disp_steer : &R.disp_follow, strdup(t));
    vec_push(steering ? &R.steer : &R.follow, strdup(t));
    code_durable_queue(steering, t); // the run's journal keeps it (code_durable.c)
    queue_update(a);
}

// the run (or compaction) stopped and waited for (teardownCurrent's session.abort())
static void abort_and_wait(void) {
    if (R.streaming || R.compacting) {
        R.ops->ui->abort(R.h);
        wait_idle();
    }
}

static JValue *cancelled_false(Arena *a) {
    JValue *d = json_new(a, J_OBJECT);
    json_put(a, d, S("cancelled"), json_bool(a, false));
    return d;
}

static void handle(Arena *a, const JValue *c) {
    const JValue *id = json_get(c, "id");
    const JValue *tv = json_get(c, "type");
    const char *type = json_str(tv);
    CodeSession *s = R.ops->ui->session(R.h);
    if (!type) {
        // `command.type` undefined: "Unknown command: undefined", no command member
        fail(a, id, NULL, "Unknown command: undefined");
        return;
    }
#define IS(x) (!strcmp(type, x))
    if (IS("prompt")) {
        const char *msg = arg_str(c, "message");
        const char *behavior = arg_str(c, "streamingBehavior");
        if (R.compacting) {
            fail(a, id, type, "Cannot submit a prompt while compaction is in progress. Wait for compaction to finish and retry.");
            return;
        }
        if (R.streaming) {
            if (!behavior) {
                fail(a, id, type, "Agent is already processing. Specify streamingBehavior ('steer' or 'followUp') to queue the message.");
                return;
            }
            queue(a, strcmp(behavior, "followUp") != 0, msg ? msg : "");
            JValue *d = json_new(a, J_OBJECT);
            json_put(a, d, S("disposition"), jstr(a, "queued"));
            ok(a, id, type, d);
            return;
        }
        // the response comes from the agent thread once the run starts (code_rpc_prompt_started)
        start_job(JOB_PROMPT, msg ? msg : "", id);
    } else if (IS("steer") || IS("follow_up")) {
        queue(a, IS("steer"), arg_str(c, "message") ? arg_str(c, "message") : "");
        JValue *d = json_new(a, J_OBJECT);
        json_put(a, d, S("disposition"), jstr(a, "queued"));
        ok(a, id, type, d);
    } else if (IS("abort")) {
        abort_and_wait();
        ok(a, id, type, NULL);
    } else if (IS("clear_queue")) {
        JValue *d = json_new(a, J_OBJECT);
        json_put(a, d, S("steering"), texts_json(a, &R.disp_steer));
        json_put(a, d, S("followUp"), texts_json(a, &R.disp_follow));
        texts_clear(&R.disp_steer);
        texts_clear(&R.disp_follow);
        texts_clear(&R.steer);
        texts_clear(&R.follow);
        queue_update(a);
        ok(a, id, type, d);
    } else if (IS("new_session")) {
        abort_and_wait();
        Err e = R.ops->new_session(R.h, arg_str(c, "parentSession"));
        if (e) fail(a, id, type, e);
        else ok(a, id, type, cancelled_false(a));
    } else if (IS("get_state")) {
        ok(a, id, type, state_json(a));
    } else if (IS("set_model")) {
        const char *p = arg_str(c, "provider"), *m = arg_str(c, "modelId");
        JValue *model = find_model(a, p ? p : "", m ? m : "");
        if (!model || !switch_to(a, model)) {
            fail(a, id, type, arena_printf(a, "Model not found: %s/%s", p ? p : "undefined", m ? m : "undefined"));
            return;
        }
        ok(a, id, type, model);
    } else if (IS("cycle_model")) {
        // _cycleAvailableModel (getNextCycleIndex: none with fewer than two models)
        JValue *all = available_models(a);
        const CodeModel *cm = R.ops->model(R.h);
        ptrdiff_t cur = -1;
        for (size_t i = 0; i < all->arr.len && cur < 0; i++)
            if (str_eqz(json_strs(json_get(all->arr.items[i], "provider")), cm->provider) && str_eqz(json_strs(json_get(all->arr.items[i], "id")), cm->id))
                cur = (ptrdiff_t)i;
        if (all->arr.len < 2) {
            ok(a, id, type, json_new(a, J_NULL));
            return;
        }
        JValue *next = all->arr.items[cur < 0 ? 0 : (size_t)(cur + 1) % all->arr.len];
        if (!switch_to(a, next)) {
            ok(a, id, type, json_new(a, J_NULL));
            return;
        }
        JValue *d = json_new(a, J_OBJECT);
        json_put(a, d, S("model"), next);
        json_put(a, d, S("thinkingLevel"), jstr(a, thinking_now()));
        json_put(a, d, S("isScoped"), json_bool(a, false));
        ok(a, id, type, d);
    } else if (IS("get_available_models")) {
        JValue *d = json_new(a, J_OBJECT);
        json_put(a, d, S("models"), available_models(a));
        ok(a, id, type, d);
    } else if (IS("set_thinking_level")) {
        const char *l = arg_str(c, "level");
        set_thinking(a, current_model(a), l ? l : "", NULL);
        ok(a, id, type, NULL);
    } else if (IS("cycle_thinking_level")) {
        JValue *model = current_model(a);
        if (!model || !(json_is(json_get(model, "reasoning"), J_BOOL) && json_get(model, "reasoning")->b)) {
            ok(a, id, type, json_new(a, J_NULL));
            return;
        }
        const char *avail[COUNT(LEVELS)];
        size_t n = supported_levels(model, avail);
        size_t cur = 0;
        bool found = false;
        for (size_t i = 0; i < n && !found; i++)
            if (!strcmp(avail[i], thinking_now())) cur = i, found = true;
        const char *next = avail[found ? (cur + 1) % n : 0];
        set_thinking(a, model, next, NULL);
        JValue *d = json_new(a, J_OBJECT);
        json_put(a, d, S("level"), jstr(a, next));
        ok(a, id, type, d);
    } else if (IS("get_available_thinking_levels")) {
        const char *avail[COUNT(LEVELS)];
        size_t n = supported_levels(current_model(a), avail);
        JValue *d = json_new(a, J_OBJECT), *l = json_new(a, J_ARRAY);
        for (size_t i = 0; i < n; i++) json_push(a, l, jstr(a, avail[i]));
        json_put(a, d, S("levels"), l);
        ok(a, id, type, d);
    } else if (IS("set_steering_mode") || IS("set_follow_up_mode")) {
        const char *mode = arg_str(c, "mode");
        bool all = mode && !strcmp(mode, "all");
        if (IS("set_steering_mode")) R.steer_all = all;
        else R.follow_all = all;
        save_setting(IS("set_steering_mode") ? "steeringMode" : "followUpMode", NULL, (JValue *)json_get(c, "mode"));
        ok(a, id, type, NULL);
    } else if (IS("compact")) {
        abort_and_wait();
        start_job(JOB_COMPACT, arg_str(c, "customInstructions"), id);
    } else if (IS("set_auto_compaction")) {
        code_compaction_set_enabled(R.ops->env(R.h)->agent_dir, arg_true(c, "enabled"));
        ok(a, id, type, NULL);
    } else if (IS("set_auto_retry")) {
        R.ops->set_auto_retry(R.h, arg_true(c, "enabled"));
        save_setting("retry", "enabled", json_bool(a, arg_true(c, "enabled")));
        ok(a, id, type, NULL);
    } else if (IS("abort_retry")) {
        R.ops->abort_retry(R.h);
        ok(a, id, type, NULL);
    } else if (IS("bash")) {
        BashJob *j = calloc(1, sizeof *j);
        j->command = strdup(arg_str(c, "command") ? arg_str(c, "command") : "");
        if (id) {
            Buf b = {0};
            json_write(&b, id);
            json_parse(&R.persist, buf_str(&b), false, &j->id);
            buf_free(&b);
        }
        const JValue *ex = json_get(c, "excludeFromContext");
        j->has_exclude = json_is(ex, J_BOOL);
        j->exclude = j->has_exclude && ex->b;
        pthread_t t;
        pthread_attr_t at;
        pthread_attr_init(&at);
        pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
        if (pthread_create(&t, &at, bash_main, j)) {
            fail(a, id, type, "Cannot start the command");
            free(j->command);
            free(j);
        }
        pthread_attr_destroy(&at);
    } else if (IS("abort_bash")) {
        for (size_t i = 0; i < R.bash_aborts.len; i++) *R.bash_aborts.ptr[i] = 1;
        ok(a, id, type, NULL);
    } else if (IS("get_session_stats")) {
        ok(a, id, type, stats_json(a));
    } else if (IS("export_html")) {
        CmdsSession cs = {0};
        const char *leaf = NULL;
        cs.entries = R.ops->ui->entries(R.h, &leaf);
        cs.leaf = leaf;
        cs.session_file = R.ops->ui->session_file(R.h);
        cs.system_prompt = R.ops->ui->system_prompt(R.h);
        cs.tools = R.ops->ui->tool_decls(R.h);
        // settings theme, else the theme pi has without a terminal (no colors reported)
        const char *theme = json_str(json_get(code_settings_merged(a, R.ops->env(R.h)), "theme"));
        CmdsTheme t = {.name = theme ? theme : "system"};
        char *written = NULL, *err = NULL;
        if (cmds_export_html(&cs, &t, arg_str(c, "outputPath"), R.ops->env(R.h)->cwd, &written, &err)) {
            JValue *d = json_new(a, J_OBJECT);
            json_put(a, d, S("path"), jstr(a, written));
            ok(a, id, type, d);
        } else fail(a, id, type, err ? err : "Export failed");
        free(written);
        free(err);
    } else if (IS("switch_session")) {
        abort_and_wait();
        const char *err = R.ops->switch_session(R.h, arg_str(c, "sessionPath") ? arg_str(c, "sessionPath") : "");
        if (err) fail(a, id, type, err);
        else ok(a, id, type, cancelled_false(a));
    } else if (IS("fork") || IS("clone")) {
        const char *entry = IS("fork") ? arg_str(c, "entryId") : code_session_leaf(s);
        if (IS("clone") && !entry) {
            fail(a, id, type, "Cannot clone session: no current entry selected");
            return;
        }
        // a missing entry fails before the run is stopped (runtime.fork's getEntry)
        abort_and_wait();
        char *selected = NULL; // in the run's arena
        const char *err = R.ops->ui->fork(R.h, entry ? entry : "", IS("clone"), &selected);
        if (err) fail(a, id, type, err);
        else if (IS("clone")) ok(a, id, type, cancelled_false(a));
        else {
            JValue *d = json_new(a, J_OBJECT);
            if (selected) json_put(a, d, S("text"), jstr(a, selected));
            json_put(a, d, S("cancelled"), json_bool(a, false));
            ok(a, id, type, d);
        }
    } else if (IS("get_fork_messages")) {
        JValue *d = json_new(a, J_OBJECT);
        json_put(a, d, S("messages"), code_sessions_fork_messages(a, code_session_entries(s)));
        ok(a, id, type, d);
    } else if (IS("get_entries")) {
        const JValue *all = code_session_entries(s);
        const char *since = arg_str(c, "since");
        JValue *list = json_new(a, J_ARRAY);
        bool found = !json_get(c, "since");
        for (size_t i = 0; i < all->arr.len; i++) {
            const JValue *e = all->arr.items[i];
            if (str_eqz(json_strs(json_get(e, "type")), "session")) continue;
            if (!found) {
                if (since && str_eqz(json_strs(json_get(e, "id")), since)) found = true;
                continue;
            }
            json_push(a, list, (JValue *)e);
        }
        if (!found) {
            fail(a, id, type, arena_printf(a, "Entry not found: %s", since ? since : "undefined"));
            return;
        }
        JValue *d = json_new(a, J_OBJECT);
        json_put(a, d, S("entries"), list);
        json_put(a, d, S("leafId"), leaf_json(a, s));
        ok(a, id, type, d);
    } else if (IS("get_tree")) {
        JValue *d = json_new(a, J_OBJECT);
        json_put(a, d, S("tree"), tree_json(a, code_session_entries(s)));
        json_put(a, d, S("leafId"), leaf_json(a, s));
        ok(a, id, type, d);
    } else if (IS("get_last_assistant_text")) {
        const char *t = last_assistant_text(a, code_sessions_projection(a, code_session_entries(s), code_session_leaf(s)));
        JValue *d = json_new(a, J_OBJECT);
        if (t) json_put(a, d, S("text"), jstr(a, t));
        ok(a, id, type, d);
    } else if (IS("set_session_name")) {
        Str name = code_js_trim(json_strs(json_get(c, "name")));
        if (!name.len) {
            fail(a, id, type, "Session name cannot be empty");
            return;
        }
        (void)code_sessions_set_name(s, arena_strndup(a, name.ptr, name.len));
        JValue *e = json_new(a, J_OBJECT);
        json_put(a, e, S("type"), jstr(a, "session_info_changed"));
        const char *now = code_sessions_name(a, code_session_entries(s));
        if (now) json_put(a, e, S("name"), jstr(a, now));
        event(e);
        ok(a, id, type, NULL);
    } else if (IS("get_messages")) {
        JValue *d = json_new(a, J_OBJECT);
        json_put(a, d, S("messages"), code_sessions_projection(a, code_session_entries(s), code_session_leaf(s)));
        ok(a, id, type, d);
    } else if (IS("get_commands")) {
        ok(a, id, type, commands_json(a));
    } else {
        fail(a, id, type, arena_printf(a, "Unknown command: %s", type));
    }
#undef IS
}

// handleInputLine: JSON.parse (pi's error text on failure), an extension UI response dropped (no
// extension asks), else the command.
static void input_line(Str line) {
    Arena a = {0};
    char *perr = NULL;
    JValue *c = code_oai_js_parse(&a, line, &perr);
    if (!c) {
        pthread_mutex_lock(&R.lock);
        fail(&a, NULL, "parse", arena_printf(&a, "Failed to parse command: %s", perr ? perr : "Unexpected end of JSON input"));
        pthread_mutex_unlock(&R.lock);
        arena_reset(&a);
        return;
    }
    if (json_is(c, J_OBJECT) && str_eqz(json_strs(json_get(c, "type")), "extension_ui_response")) {
        arena_reset(&a);
        return;
    }
    // abort and abort_bash reach a busy agent without the lock (it holds it while it works)
    const char *type = json_is(c, J_OBJECT) ? json_str(json_get(c, "type")) : NULL;
    if (type && !strcmp(type, "abort") && (R.streaming || R.compacting)) R.ops->ui->abort(R.h);
    pthread_mutex_lock(&R.lock);
    if (json_is(c, J_OBJECT)) handle(&a, c);
    else fail(&a, NULL, NULL, "Unknown command: undefined");
    pthread_mutex_unlock(&R.lock);
    arena_reset(&a);
}

Err code_rpc_run(const CodeRpcOps *ops, CodeUiHost *h) {
    R.on = true;
    R.ops = ops;
    R.h = h;
    // the queue modes from the settings (syncQueueModesFromSettings)
    {
        Arena a = {0};
        JValue *st = code_settings_merged(&a, ops->env(h));
        R.steer_all = str_eqz(json_strs(json_get(st, "steeringMode")), "all");
        R.follow_all = str_eqz(json_strs(json_get(st, "followUpMode")), "all");
        arena_reset(&a);
    }
    pthread_mutex_lock(&R.lock);
    pthread_t agent;
    if (pthread_create(&agent, NULL, agent_main, NULL)) {
        pthread_mutex_unlock(&R.lock);
        return "NoThread";
    }
    pthread_detach(agent);
    pthread_mutex_unlock(&R.lock);
    // attachJsonlLineReader: LF framing only, a trailing CR dropped, the rest at the end
    Buf in = {0};
    char chunk[65536];
    for (;;) {
        ssize_t got = read(0, chunk, sizeof chunk);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) break;
        buf_add(&in, chunk, (size_t)got);
        size_t start = 0;
        for (size_t i = 0; i < in.len; i++) {
            if (in.ptr[i] != '\n') continue;
            size_t end = i;
            if (end > start && in.ptr[end - 1] == '\r') end--;
            input_line(str_n(in.ptr + start, end - start));
            start = i + 1;
        }
        memmove(in.ptr, in.ptr + start, in.len - start);
        in.len -= start;
    }
    if (in.len) {
        size_t end = in.len;
        if (in.ptr[end - 1] == '\r') end--;
        input_line(str_n(in.ptr, end));
    }
    buf_free(&in);
    // shutdown: runtimeHost.dispose() (the run aborted and settled), then exit 0
    pthread_mutex_lock(&R.lock);
    abort_and_wait();
    pthread_mutex_unlock(&R.lock);
    return OK;
}
