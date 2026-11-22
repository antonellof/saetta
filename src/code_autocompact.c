// Saetta Code's automatic compaction (see code_autocompact.h).
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi v1.5.2,
// b4afddd): packages/coding-agent/src/core/agent-session.ts, core/settings-manager.ts,
// core/compaction/compaction.ts, packages/ai/src/utils/overflow.ts. Each function names the pi
// function it ports.
#include "code_autocompact.h"

#include "code_cfg.h"
#include "code_sessions.h"

#include <math.h>
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// ---- helpers -------------------------------------------------------------------------------------

static double jnum(const JValue *v) {
    if (json_is(v, J_INTEGER)) return (double)v->i;
    if (json_is(v, J_FLOAT)) return v->f;
    if (json_is(v, J_NUMBER_STRING)) return strtod(v->s.ptr, NULL);
    return 0;
}

static Str type_of(const JValue *e) { return json_strs(json_get(e, "type")); }
static Str role_of(const JValue *m) { return json_strs(json_get(m, "role")); }
static Str stop_of(const JValue *m) { return json_strs(json_get(m, "stopReason")); }

static JValue *ev_new(Arena *a, const char *type) {
    JValue *e = json_new(a, J_OBJECT);
    json_put(a, e, S("type"), json_string(a, str(type)));
    return e;
}

// new Date(iso).getTime()
static double iso_ms(Str t) {
    char buf[40];
    size_t n = t.len < sizeof buf - 1 ? t.len : sizeof buf - 1;
    memcpy(buf, t.ptr, n);
    buf[n] = 0;
    struct tm tm = {0};
    int ms = 0;
    int got = sscanf(buf, "%4d-%2d-%2dT%2d:%2d:%2d.%3dZ", &tm.tm_year, &tm.tm_mon, &tm.tm_mday, &tm.tm_hour, &tm.tm_min, &tm.tm_sec, &ms);
    if (got < 6) return NAN;
    tm.tm_year -= 1900;
    tm.tm_mon -= 1;
    return (double)timegm(&tm) * 1000 + ms;
}

// ---- settings (settings-manager.ts) ------------------------------------------------------------------

// A non-negative safe integer setting, or -1 when unset or invalid (pi throws on an invalid one;
// here the next source is used).
static int64_t token_setting(const JValue *v) {
    if (json_is(v, J_INTEGER)) return v->i >= 0 ? v->i : -1;
    if (json_is(v, J_NUMBER_STRING) || json_is(v, J_FLOAT)) {
        double d = jnum(v);
        return d >= 0 && d == floor(d) && d <= 9007199254740991.0 ? (int64_t)d : -1;
    }
    return -1;
}

// getCompactionSettings (getCompactionEnabled, getCompactionTokenSetting)
CodeCompactionSettings code_compaction_settings(const JValue *settings, const char *provider, const char *id) {
    const JValue *c = json_get(settings, "compaction");
    CodeCompactionSettings s = {true, 16384, 20000};
    const JValue *en = json_get(c, "enabled");
    if (json_is(en, J_BOOL)) s.enabled = en->b;
    const JValue *ov = NULL;
    if (provider && id) {
        Arena t = {0};
        ov = json_get(json_get(c, "modelOverrides"), arena_printf(&t, "%s/%s", provider, id));
        arena_reset(&t);
    }
    static const char *const fields[] = {"reserveTokens", "keepRecentTokens"};
    for (int k = 0; k < 2; k++) {
        int64_t v = token_setting(json_get(ov, fields[k]));
        if (v < 0) v = token_setting(json_get(c, fields[k]));
        if (v >= 0) *(k ? &s.keep_recent_tokens : &s.reserve_tokens) = v;
    }
    return s;
}

static bool set_enabled(Arena *a, JValue *root, void *ctx) {
    JValue *c = (JValue *)json_get(root, "compaction");
    if (!json_is(c, J_OBJECT)) {
        c = json_new(a, J_OBJECT);
        json_put(a, root, S("compaction"), c);
    }
    json_put(a, c, S("enabled"), json_bool(a, *(bool *)ctx));
    return true;
}

// setCompactionEnabled
void code_compaction_set_enabled(const char *agent_dir, bool enabled) {
    Arena a = {0};
    Buf why = {0};
    (void)cfg_json_update(path_join(&a, agent_dir, "settings.json"), set_enabled, &enabled, &why);
    buf_free(&why);
    arena_reset(&a);
}

// shouldCompact
static bool should_compact(double tokens, uint64_t window, const CodeCompactionSettings *s) {
    if (!s->enabled) return false;
    return tokens > (double)window - (double)s->reserve_tokens;
}

// ---- overflow.ts -------------------------------------------------------------------------------------

// OVERFLOW_PATTERNS, NON_OVERFLOW_PATTERNS and CEREBRAS_BODYLESS_OVERFLOW_PATTERN in POSIX
// extended syntax (case-insensitive): (?: ) as ( ), \d as [0-9], \s as [[:space:]], \w as
// [[:alnum:]_].
static const char *const OVERFLOW_PATTERNS[] = {
    "prompt (is )?too long",
    "prompt exceeds max length",
    "request_too_large",
    "input is too long for requested model",
    "exceeds the context window",
    "exceeds (the )?(model'?s )?maximum context length( of [0-9,]+ tokens?|[[:space:]]*\\([0-9,]+\\))",
    "context window exceeded",
    "input token count.*exceeds the maximum",
    "maximum prompt length is [0-9]+",
    "reduce the length of the messages",
    "maximum context length is [0-9]+ tokens",
    "exceeds (the )?maximum allowed input length of [0-9,]+ tokens?",
    "input \\([0-9]+ tokens\\) is longer than the model'?s context length \\([0-9]+ tokens\\)",
    "exceeds the limit of [0-9]+",
    "exceeds the available context size",
    "greater than the context length",
    "context window exceeds limit",
    "exceeded model token limit",
    "too large for model with [0-9]+ maximum context length",
    "prompt has [0-9,]+ tokens?, but the configured context size is [0-9,]+ tokens?",
    "model_context_window_exceeded",
    "prompt too long; exceeded (max )?context length",
    "range of input length should be",
    "context[_ ]length[_ ]exceeded",
    "too many tokens",
    "token limit exceeded",
    "^([[:alnum:]_ -]+ API error \\(413\\):|413([[:space:]]|:|$))",
};
static const char *const NON_OVERFLOW_PATTERNS[] = {"^(Throttling error|Service unavailable):", "rate limit", "too many requests"};
static const char CEREBRAS_BODYLESS_OVERFLOW_PATTERN[] = "^4(00|13)[[:space:]]*(status code)?[[:space:]]*\\(no body\\)";

static bool re_test(const char *pattern, const char *text) {
    regex_t re;
    if (regcomp(&re, pattern, REG_EXTENDED | REG_ICASE | REG_NOSUB)) return false;
    bool m = regexec(&re, text, 0, NULL, 0) == 0;
    regfree(&re);
    return m;
}

// isContextOverflow
bool code_context_overflow(const JValue *msg, uint64_t window) {
    Str stop = stop_of(msg);
    const char *em = json_str(json_get(msg, "errorMessage"));
    if (str_eqz(stop, "error") && em && *em) {
        bool non = false;
        for (size_t i = 0; i < COUNT(NON_OVERFLOW_PATTERNS) && !non; i++) non = re_test(NON_OVERFLOW_PATTERNS[i], em);
        if (!non) {
            for (size_t i = 0; i < COUNT(OVERFLOW_PATTERNS); i++)
                if (re_test(OVERFLOW_PATTERNS[i], em)) return true;
            if (str_eqz(json_strs(json_get(msg, "provider")), "cerebras") && re_test(CEREBRAS_BODYLESS_OVERFLOW_PATTERN, em)) return true;
        }
    }
    const JValue *u = json_get(msg, "usage");
    double input = jnum(json_get(u, "input")) + jnum(json_get(u, "cacheRead"));
    if (window && str_eqz(stop, "stop") && input > (double)window) return true;
    if (window && str_eqz(stop, "length") && jnum(json_get(u, "output")) == 0 && input >= (double)window * 0.99) return true;
    return false;
}

// isRecoverableLength
bool code_recoverable_length(const JValue *msg, uint64_t desired) {
    return str_eqz(stop_of(msg), "length") && desired > 0 && jnum(json_get(json_get(msg, "usage"), "output")) < (double)desired;
}

// ---- token estimates (compaction.ts) --------------------------------------------------------------

// calculateContextTokens
static double context_tokens(const JValue *u) {
    double t = jnum(json_get(u, "totalTokens"));
    return t ? t : jnum(json_get(u, "input")) + jnum(json_get(u, "output")) + jnum(json_get(u, "cacheRead")) + jnum(json_get(u, "cacheWrite"));
}

// getAssistantUsage
static const JValue *assistant_usage(const JValue *m) {
    if (!str_eqz(role_of(m), "assistant") || !json_get(m, "usage")) return NULL;
    Str stop = stop_of(m);
    if (str_eqz(stop, "aborted") || str_eqz(stop, "error")) return NULL;
    const JValue *u = json_get(m, "usage");
    return json_is(u, J_OBJECT) && context_tokens(u) > 0 ? u : NULL;
}

// estimateContextTokens: *last_usage the index of the message whose usage it starts from, or -1.
static double estimate_context(const JValue *msgs, ptrdiff_t *last_usage) {
    *last_usage = -1;
    for (size_t i = msgs->arr.len; i-- > 0 && *last_usage < 0;)
        if (assistant_usage(msgs->arr.items[i])) *last_usage = (ptrdiff_t)i;
    double tokens = 0;
    size_t from = 0;
    if (*last_usage >= 0) {
        tokens = context_tokens(assistant_usage(msgs->arr.items[*last_usage]));
        from = (size_t)*last_usage + 1;
    }
    for (size_t i = from; i < msgs->arr.len; i++) tokens += (double)code_sessions_estimate(msgs->arr.items[i]);
    return tokens;
}

// estimateMessagesTokens (agent-session.ts)
static double messages_tokens(const JValue *msgs) {
    double t = 0;
    for (size_t i = 0; i < msgs->arr.len; i++) t += (double)code_sessions_estimate(msgs->arr.items[i]);
    return t;
}

// ---- the session ------------------------------------------------------------------------------------

static CodeCompactionSettings load_settings(const CodeAutoCompact *c) {
    Arena t = {0};
    char *data = NULL;
    size_t n = 0;
    JValue *root = NULL;
    if (!read_file(path_join(&t, c->env->agent_dir, "settings.json"), 1 << 24, &data, &n)) (void)json_parse(&t, str_n(data, n), true, &root);
    free(data);
    CodeCompactionSettings s = code_compaction_settings(root, c->model->provider, c->model->id);
    arena_reset(&t);
    return s;
}

// getLatestCompactionEntry
static const JValue *latest_compaction(const JValue *branch) {
    for (size_t i = branch->arr.len; i-- > 0;)
        if (str_eqz(type_of(branch->arr.items[i]), "compaction")) return branch->arr.items[i];
    return NULL;
}

static ptrdiff_t index_of(const JValue *branch, const char *id) {
    for (size_t i = 0; id && i < branch->arr.len; i++)
        if (str_eqz(json_strs(json_get(branch->arr.items[i], "id")), id)) return (ptrdiff_t)i;
    return -1;
}

// _findPersistedMessageEntryId: the message entry of the branch holding m (the same object, or an
// assistant message of the same time and response).
static const char *entry_id_of(const JValue *branch, const JValue *m) {
    for (size_t i = branch->arr.len; i-- > 0;) {
        const JValue *e = branch->arr.items[i];
        if (!str_eqz(type_of(e), "message")) continue;
        const JValue *em = json_get(e, "message");
        if (em == m) return json_str(json_get(e, "id"));
    }
    for (size_t i = branch->arr.len; i-- > 0;) {
        const JValue *e = branch->arr.items[i];
        const JValue *em = json_get(e, "message");
        if (str_eqz(type_of(e), "message") && str_eq(role_of(em), role_of(m)) && jnum(json_get(em, "timestamp")) == jnum(json_get(m, "timestamp")) &&
            str_eq(json_strs(json_get(em, "responseId")), json_strs(json_get(m, "responseId"))))
            return json_str(json_get(e, "id"));
    }
    return NULL;
}

// Whether entry id is in the projection with an assistant message (the context entries, not
// dropped by a context edit).
static bool projected_assistant(const JValue *ctx, const char *id) {
    const JValue *entry = NULL, *edit = NULL;
    for (size_t i = 0; i < ctx->arr.len; i++) {
        const JValue *e = ctx->arr.items[i];
        if (str_eqz(json_strs(json_get(e, "id")), id)) entry = e;
        if (str_eqz(type_of(e), "context_edit") && str_eqz(json_strs(json_get(e, "targetId")), id)) edit = e;
    }
    if (!entry || !str_eqz(type_of(entry), "message") || !str_eqz(role_of(json_get(entry, "message")), "assistant")) return false;
    const JValue *rep = edit ? json_get(edit, "replacement") : NULL;
    return !edit || (rep && rep->kind != J_NULL);
}

static Err append_edit(CodeAutoCompact *c, const char *target) {
    Arena *a = c->a;
    JValue *f = json_new(a, J_OBJECT);
    json_put(a, f, S("targetId"), json_string(a, str(target)));
    json_put(a, f, S("replacement"), json_new(a, J_NULL));
    TRY(code_session_append(*c->session, "context_edit", f));
    JValue *ea = ev_new(a, "entry_appended");
    json_put(a, ea, S("entry"), (JValue *)code_session_entry_last(*c->session));
    c->emit(c->ctx, ea);
    return OK;
}

// _omitRecoveryAttempt: the failed attempt and its tool results left out of the model's context
// (a context_edit each, entry_appended emitted).
static void omit_attempt(CodeAutoCompact *c, const JValue *msg, const char *id) {
    Arena *a = c->a;
    JValue *branch = code_sessions_branch(a, code_session_entries(*c->session), code_session_leaf(*c->session));
    if (!id) id = entry_id_of(branch, msg);
    ptrdiff_t at = index_of(branch, id);
    VEC(const char *) targets = {0};
    if (id) vec_push(&targets, id);
    const JValue *content = json_get(msg, "content");
    for (size_t i = at < 0 ? branch->arr.len : (size_t)at + 1; i < branch->arr.len; i++) {
        const JValue *e = branch->arr.items[i];
        if (!str_eqz(type_of(e), "message")) continue;
        const JValue *m = json_get(e, "message");
        if (str_eqz(role_of(m), "assistant")) break;
        if (!str_eqz(role_of(m), "toolResult")) continue;
        Str call = json_strs(json_get(m, "toolCallId"));
        bool ours = false;
        for (size_t k = 0; json_is(content, J_ARRAY) && k < content->arr.len && !ours; k++) {
            const JValue *b = content->arr.items[k];
            ours = str_eqz(type_of(b), "toolCall") && str_eq(json_strs(json_get(b, "id")), call);
        }
        if (ours) vec_push(&targets, json_str(json_get(e, "id")));
    }
    for (size_t i = 0; i < targets.len; i++) (void)append_edit(c, targets.ptr[i]);
    vec_free(&targets);
}

typedef struct {
    CodeAutoCompact *c;
    const char *reason;
    bool started;
} Start;

static void on_started(void *ctx) {
    Start *st = ctx;
    st->started = true;
    JValue *ev = ev_new(st->c->a, "compaction_start");
    json_put(st->c->a, ev, S("reason"), json_string(st->c->a, str(st->reason)));
    st->c->emit(st->c->ctx, ev);
}

// _runAutoCompaction: whether the run goes on (willRetry, or queued messages to deliver).
static bool run_auto(CodeAutoCompact *c, const char *reason, bool will_retry) {
    Arena *a = c->a;
    CodeCompactionSettings set = load_settings(c);
    Start st = {c, reason, false};
    const JValue *entry = NULL;
    const char *err = code_sessions_compact_cb(a, *c->session, c->model, c->env, NULL, set.keep_recent_tokens, &entry, on_started, &st);
    bool aborted = c->env->abort && *c->env->abort;
    if (!st.started) return false; // prepareCompaction found nothing to compact
    JValue *end = ev_new(a, "compaction_end");
    json_put(a, end, S("reason"), json_string(a, str(reason)));
    if (!err && entry) {
        // The entry's time is in ms, and a message is "from before the compaction" when its time is
        // not after it: the next message must not share that ms (pi's runtime never runs that fast).
        double t = iso_ms(json_strs(json_get(entry, "timestamp")));
        while ((double)(wall_ns() / 1000000) <= t) {
            struct timespec ts = {0, 200000};
            nanosleep(&ts, NULL);
        }
        JValue *res = json_new(a, J_OBJECT);
        json_put(a, res, S("summary"), (JValue *)json_get(entry, "summary"));
        json_put(a, res, S("firstKeptEntryId"), (JValue *)json_get(entry, "firstKeptEntryId"));
        json_put(a, res, S("tokensBefore"), (JValue *)json_get(entry, "tokensBefore"));
        double after = messages_tokens(code_sessions_projection(a, code_session_entries(*c->session), code_session_leaf(*c->session)));
        json_put(a, res, S("estimatedTokensAfter"), json_int(a, (int64_t)after));
        if (json_get(entry, "usage")) json_put(a, res, S("usage"), (JValue *)json_get(entry, "usage"));
        json_put(a, res, S("details"), (JValue *)json_get(entry, "details"));
        json_put(a, end, S("result"), res);
        json_put(a, end, S("aborted"), json_bool(a, false));
        json_put(a, end, S("willRetry"), json_bool(a, will_retry));
        c->emit(c->ctx, end);
        if (will_retry) return true;
        return c->has_queued && c->has_queued(c->ctx);
    }
    aborted = aborted || (err && !strcmp(err, "aborted"));
    json_put(a, end, S("aborted"), json_bool(a, aborted));
    json_put(a, end, S("willRetry"), json_bool(a, false));
    if (!aborted) {
        const char *m = err ? err : "compaction failed";
        json_put(a, end, S("errorMessage"),
                 json_string(a, str(arena_printf(a, !strcmp(reason, "overflow") ? "Context overflow recovery failed: %s" : "Auto-compaction failed: %s", m))));
    }
    c->emit(c->ctx, end);
    // abortCompaction stops the compaction only, not the run
    if (aborted && c->env->abort) *c->env->abort = 0;
    return false;
}

// _checkCompaction
bool code_autocompact_check(CodeAutoCompact *c, const JValue *msg, const char *id, bool skip_aborted) {
    Arena *a = c->a;
    CodeCompactionSettings set = load_settings(c);
    if (!set.enabled) return false;
    if (skip_aborted && str_eqz(stop_of(msg), "aborted")) return false;
    // _modelForMessage: the current model when it produced the message
    bool same = str_eqz(json_strs(json_get(msg, "provider")), c->model->provider) && str_eqz(json_strs(json_get(msg, "model")), c->model->id);
    uint64_t window = c->model->context_window;
    CodeSession *s = *c->session;
    JValue *branch = code_sessions_branch(a, code_session_entries(s), code_session_leaf(s));
    const JValue *comp = latest_compaction(branch);
    double comp_ms = comp ? iso_ms(json_strs(json_get(comp, "timestamp"))) : NAN;
    if (comp && jnum(json_get(msg, "timestamp")) <= comp_ms) return false;

    // Automatic cases 1 and 2: context overflow
    if (!id) id = entry_id_of(branch, msg);
    JValue *ctx = code_sessions_context_entries(a, code_session_entries(s), code_session_leaf(s));
    bool projected = !id || projected_assistant(ctx, id);
    ptrdiff_t at = index_of(branch, id);
    bool post_edit = false, post_compaction = false;
    const JValue *latest_edit = NULL;
    for (size_t i = at < 0 ? branch->arr.len : (size_t)at + 1; i < branch->arr.len; i++) {
        const JValue *e = branch->arr.items[i];
        if (str_eqz(type_of(e), "context_edit")) {
            post_edit = true;
            if (str_eqz(json_strs(json_get(e, "targetId")), id)) latest_edit = e;
        }
        if (str_eqz(type_of(e), "compaction")) post_compaction = true;
    }
    const JValue *rep = latest_edit ? json_get(latest_edit, "replacement") : NULL;
    bool retained = !id || (!post_compaction && (!latest_edit || (rep && rep->kind != J_NULL)));
    bool usage_matches = projected && !post_edit;
    bool explicit = str_eqz(stop_of(msg), "error") && code_context_overflow(msg, 0);
    bool overflow = same && ((explicit && retained) || (usage_matches && code_context_overflow(msg, window)));
    bool recoverable = same && projected && code_recoverable_length(msg, c->model->max_tokens);
    if (overflow || recoverable) {
        bool will_retry = !str_eqz(stop_of(msg), "stop");
        // Case 2: the response completed; compact without a retry
        if (!will_retry) return run_auto(c, "overflow", false);
        if (c->overflow_attempted) {
            JValue *end = ev_new(a, "compaction_end");
            json_put(a, end, S("reason"), json_string(a, S("overflow")));
            json_put(a, end, S("aborted"), json_bool(a, false));
            json_put(a, end, S("willRetry"), json_bool(a, false));
            json_put(a, end, S("errorMessage"),
                     json_string(a, overflow ? S("Context overflow recovery failed after one compact-and-retry attempt. Try reducing context or switching to a larger-context model.")
                                             : S("Truncated response recovery failed after one compact-and-retry attempt.")));
            c->emit(c->ctx, end);
            return false;
        }
        c->overflow_attempted = true;
        omit_attempt(c, msg, id);
        return run_auto(c, "overflow", true);
    }

    // Case 3: the threshold, without a retry
    bool has_edits = false;
    for (size_t i = 0; i < ctx->arr.len && !has_edits; i++) has_edits = str_eqz(type_of(ctx->arr.items[i]), "context_edit");
    const JValue *u = json_get(msg, "usage");
    double direct = u ? context_tokens(u) : 0, tokens;
    if (has_edits) tokens = code_sessions_projected_tokens(a, code_session_entries(s), code_session_leaf(s));
    else if (str_eqz(stop_of(msg), "error") || direct == 0) {
        const JValue *msgs = code_sessions_projection(a, code_session_entries(s), code_session_leaf(s));
        ptrdiff_t ui;
        tokens = estimate_context(msgs, &ui);
        // usage from before the latest compaction is stale
        if (ui >= 0 && comp && str_eqz(role_of(msgs->arr.items[ui]), "assistant") && jnum(json_get(msgs->arr.items[ui], "timestamp")) <= comp_ms) return false;
    } else tokens = direct;
    if (should_compact(tokens, window, &set)) return run_auto(c, "threshold", false);
    return false;
}

// _prepareRun: _findLastAssistantMessage, then _checkCompaction(lastAssistant, false)
void code_autocompact_before_prompt(CodeAutoCompact *c) {
    const JValue *msgs = code_session_messages(*c->session);
    for (size_t i = msgs->arr.len; i-- > 0;)
        if (str_eqz(role_of(msgs->arr.items[i]), "assistant")) {
            (void)code_autocompact_check(c, msgs->arr.items[i], NULL, false);
            return;
        }
}

// _compactBeforeNextAssistantResponse (_exceedsCompactionThreshold)
void code_autocompact_before_turn(CodeAutoCompact *c) {
    if (c->model->context_window <= 0) return;
    CodeCompactionSettings set = load_settings(c);
    CodeSession *s = *c->session;
    double tokens = code_sessions_projected_tokens(c->a, code_session_entries(s), code_session_leaf(s));
    if (!should_compact(tokens, c->model->context_window, &set)) return;
    (void)run_auto(c, "threshold", false);
}

void code_autocompact_observe(CodeAutoCompact *c, const JValue *m) {
    Str role = role_of(m), stop = stop_of(m);
    if (str_eqz(role, "user") || (str_eqz(role, "assistant") && !str_eqz(stop, "error") && !str_eqz(stop, "length"))) c->overflow_attempted = false;
}
