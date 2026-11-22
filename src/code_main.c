// `saetta code` (see code.h): the command line, print and json modes, and the agent loop.
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi v1.5.2:
// packages/agent/src/agent-loop.ts, packages/coding-agent/src/main.ts, cli/args.ts,
// core/agent-session.ts, modes/print-mode.ts, modes/json-event.ts, packages/ai/src/utils/transcript.ts).
#include "code.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static const char *const default_tools[] = {"read", "bash", "edit", "write"};

typedef struct {
    const char *provider, *model, *thinking, *mode, *system_prompt, *tools;
    bool print, cont, no_session, no_tools;
    VEC(const char *) prompts;
    VEC(const char *) append;
} Args;

typedef struct {
    Arena *a;
    CodeEnv env;
    CodeModel model;
    CodeSession *session;
    const char **tools;
    size_t ntools;
    bool json;
    JValue *new_messages; // this run's messages, for agent_end
} Run;

static double wall_ms(void) { return (double)(wall_ns() / 1000000); }

// ---- output --------------------------------------------------------------------------------------

static void emit(Run *r, JValue *ev) {
    if (!r->json) return;
    Buf b = {0};
    json_write(&b, ev);
    buf_addc(&b, '\n');
    fwrite(b.ptr, 1, b.len, stdout);
    fflush(stdout);
    buf_free(&b);
}

static JValue *ev_new(Arena *a, const char *type) {
    JValue *e = json_new(a, J_OBJECT);
    json_put(a, e, S("type"), json_string(a, str(type)));
    return e;
}

static void emit_message(Run *r, const char *type, const JValue *m) {
    JValue *e = ev_new(r->a, type);
    json_put(r->a, e, S("message"), (JValue *)m);
    emit(r, e);
}

// ---- arguments (cli/args.ts, the subset print mode needs) -----------------------------------------

static Err parse_args(const char *const *v, size_t n, Args *o) {
    for (size_t i = 0; i < n; i++) {
        const char *x = v[i];
        bool more = i + 1 < n;
        if (!strcmp(x, "--provider") && more) o->provider = v[++i];
        else if (!strcmp(x, "--model") && more) o->model = v[++i];
        else if (!strcmp(x, "--thinking") && more) o->thinking = v[++i];
        else if (!strcmp(x, "--mode") && more) o->mode = v[++i];
        else if (!strcmp(x, "--print") || !strcmp(x, "-p")) o->print = true;
        else if (!strcmp(x, "--continue") || !strcmp(x, "-c")) o->cont = true;
        else if (!strcmp(x, "--no-session")) o->no_session = true;
        else if (!strcmp(x, "--system-prompt") && more) o->system_prompt = v[++i];
        else if (!strcmp(x, "--append-system-prompt") && more) vec_push(&o->append, v[++i]);
        else if (!strcmp(x, "--tools") && more) o->tools = v[++i];
        else if (!strcmp(x, "--no-tools") || !strcmp(x, "-nt")) o->no_tools = true;
        else if (x[0] == '-' && x[1]) {
            dbg("saetta code: unknown option %s\n", x);
            return "UnknownOption";
        } else vec_push(&o->prompts, x);
    }
    if (o->mode && strcmp(o->mode, "text") && strcmp(o->mode, "json")) return "UnknownMode";
    if (o->thinking) {
        static const char *const levels[] = {"off", "minimal", "low", "medium", "high", "xhigh", "max"};
        bool ok = false;
        for (size_t k = 0; k < COUNT(levels); k++) ok |= !strcmp(o->thinking, levels[k]);
        if (!ok) {
            dbg("Invalid thinking level \"%s\". Valid values: off, minimal, low, medium, high, xhigh, max\n", o->thinking);
            return "BadThinkingLevel";
        }
    }
    return OK;
}

// ---- config: models.json, settings.json (core/model-config.ts, settings-manager.ts) --------------

static JValue *read_json(Arena *a, const char *path) {
    char *text;
    size_t len;
    if (read_file(path, (size_t)16 << 20, &text, &len)) return NULL;
    JValue *v = NULL;
    if (json_parse(a, str_n(text, len), true, &v)) v = NULL;
    free(text);
    return v;
}

static bool jbool(const JValue *v, bool dflt) { return json_is(v, J_BOOL) ? v->b : dflt; }

static uint64_t juint(const JValue *v, uint64_t dflt) { return json_is(v, J_INTEGER) && v->i >= 0 ? (uint64_t)v->i : dflt; }

// resolveConfigValue: an environment variable of that name, else the literal.
static const char *config_value(const char *v) {
    if (!v) return NULL;
    const char *e = getenv(v);
    return e && e[0] ? e : v;
}

static Err find_model(Run *r, const Args *args, const JValue *settings) {
    Arena *a = r->a;
    JValue *models = read_json(a, path_join(a, r->env.agent_dir, "models.json"));
    const JValue *providers = json_get(models, "providers");
    const char *want_p = args->provider ? args->provider : json_str(json_get(settings, "defaultProvider"));
    const char *want_m = args->model ? args->model : json_str(json_get(settings, "defaultModel"));
    const char *level_suffix = NULL;
    if (want_m) {
        // "provider/id" and ":thinking"
        const char *slash = strchr(want_m, '/');
        if (slash && !args->provider) {
            want_p = arena_strndup(a, want_m, (size_t)(slash - want_m));
            want_m = slash + 1;
        }
        const char *colon = strrchr(want_m, ':');
        if (colon) {
            level_suffix = colon + 1;
            want_m = arena_strndup(a, want_m, (size_t)(colon - want_m));
        }
    }
    if (!json_is(providers, J_OBJECT)) return "NoModels";
    for (size_t i = 0; i < providers->obj.len; i++) {
        const JMember *p = &providers->obj.items[i];
        if (want_p && !str_eqz(p->key, want_p)) continue;
        const JValue *list = json_get(p->val, "models");
        if (!json_is(list, J_ARRAY)) continue;
        for (size_t k = 0; k < list->arr.len; k++) {
            const JValue *m = list->arr.items[k];
            if (want_m && !str_eqz(json_strs(json_get(m, "id")), want_m)) continue;
            if (!str_eqz(json_strs(json_get(p->val, "api")), "openai-completions") &&
                !str_eqz(json_strs(json_get(m, "api")), "openai-completions"))
                return "UnsupportedApi"; // Saetta Code speaks OpenAI chat completions only
            const JValue *compat = json_get(p->val, "compat");
            CodeModel *cm = &r->model;
            cm->provider = str_z(a, p->key);
            cm->id = json_str(json_get(m, "id"));
            cm->name = json_str(json_get(m, "name"));
            cm->base_url = json_str(json_get(m, "baseUrl")) ? json_str(json_get(m, "baseUrl")) : json_str(json_get(p->val, "baseUrl"));
            cm->api_key = config_value(json_str(json_get(p->val, "apiKey")));
            cm->reasoning = jbool(json_get(m, "reasoning"), false);
            cm->context_window = juint(json_get(m, "contextWindow"), 128000);
            cm->max_tokens = juint(json_get(m, "maxTokens"), 16384);
            cm->compat = compat;
            cm->supports_developer_role = jbool(json_get(compat, "supportsDeveloperRole"), true);
            cm->supports_store = jbool(json_get(compat, "supportsStore"), true);
            cm->supports_reasoning_effort = jbool(json_get(compat, "supportsReasoningEffort"), true);
            cm->max_tokens_field = json_str(json_get(compat, "maxTokensField")) ? json_str(json_get(compat, "maxTokensField"))
                                                                                  : "max_completion_tokens";
            cm->thinking_format = json_str(json_get(compat, "thinkingFormat"));
            if (level_suffix) cm->thinking = level_suffix;
            return OK;
        }
    }
    return "NoModel";
}

// ---- the system message and tools (transcript.ts, agent-session.ts) -------------------------------

// getCurrentSystemMessage: every system message replayed into one.
static JValue *current_system(Arena *a, const JValue *messages) {
    JValue *sections = json_new(a, J_OBJECT);
    JValue *tools = json_new(a, J_ARRAY);
    Buf content = {0};
    const JValue *ts = NULL;
    for (size_t i = 0; messages && i < messages->arr.len; i++) {
        const JValue *m = messages->arr.items[i];
        if (!str_eqz(json_strs(json_get(m, "role")), "system")) continue;
        if (!ts) ts = json_get(m, "timestamp");
        Str text = json_strs(json_get(m, "content"));
        if (text.len) {
            if (content.len) buf_adds(&content, "\n\n");
            buf_addstr(&content, text);
        }
        const JValue *sec = json_get(m, "sections");
        for (size_t k = 0; sec && json_is(sec, J_OBJECT) && k < sec->obj.len; k++) {
            const JMember *s = &sec->obj.items[k];
            if (s->val->kind == J_NULL) {
                // Map.delete: remove, keeping the others' order
                for (size_t j = 0; j < sections->obj.len; j++)
                    if (str_eq(sections->obj.items[j].key, s->key)) {
                        memmove(&sections->obj.items[j], &sections->obj.items[j + 1], (sections->obj.len - j - 1) * sizeof(JMember));
                        sections->obj.len--;
                        break;
                    }
            } else json_put(a, sections, s->key, s->val);
        }
        const JValue *rm = json_get(m, "toolsRemoved");
        for (size_t k = 0; rm && json_is(rm, J_ARRAY) && k < rm->arr.len; k++) {
            Str name = json_strs(json_get(rm->arr.items[k], "name"));
            for (size_t j = 0; j < tools->arr.len; j++)
                if (str_eq(json_strs(json_get(tools->arr.items[j], "name")), name)) {
                    memmove(&tools->arr.items[j], &tools->arr.items[j + 1], (tools->arr.len - j - 1) * sizeof(JValue *));
                    tools->arr.len--;
                    break;
                }
        }
        const JValue *add = json_get(m, "toolsAdded");
        for (size_t k = 0; add && json_is(add, J_ARRAY) && k < add->arr.len; k++) {
            Str name = json_strs(json_get(add->arr.items[k], "name"));
            bool replaced = false;
            for (size_t j = 0; j < tools->arr.len && !replaced; j++)
                if (str_eq(json_strs(json_get(tools->arr.items[j], "name")), name)) {
                    tools->arr.items[j] = add->arr.items[k];
                    replaced = true;
                }
            if (!replaced) json_push(a, tools, add->arr.items[k]);
        }
    }
    if (!ts && tools->arr.len == 0) {
        buf_free(&content);
        return NULL;
    }
    JValue *s = json_new(a, J_OBJECT);
    json_put(a, s, S("role"), json_string(a, S("system")));
    json_put(a, s, S("content"), json_string(a, str_n(arena_strndup(a, content.ptr ? content.ptr : "", content.len), content.len)));
    if (sections->obj.len) json_put(a, s, S("sections"), sections);
    if (tools->arr.len) json_put(a, s, S("toolsAdded"), tools);
    json_put(a, s, S("timestamp"), ts ? (JValue *)ts : json_int(a, 0));
    buf_free(&content);
    return s;
}

// pi's tool declaration from an OpenAI tools entry: name, description, parameters,
// constrainedSampling (the built-in tools prefer JSON-schema constrained sampling).
static JValue *declaration(Arena *a, const JValue *oai) {
    const JValue *f = json_get(oai, "function");
    JValue *d = json_new(a, J_OBJECT);
    json_put(a, d, S("name"), (JValue *)json_get(f, "name"));
    json_put(a, d, S("description"), (JValue *)json_get(f, "description"));
    json_put(a, d, S("parameters"), (JValue *)json_get(f, "parameters"));
    JValue *cs = json_new(a, J_OBJECT);
    json_put(a, cs, S("type"), json_string(a, S("json_schema")));
    json_put(a, cs, S("strict"), json_string(a, S("prefer")));
    json_put(a, d, S("constrainedSampling"), cs);
    return d;
}

// The system message to add before this prompt, or NULL when the transcript already has it
// (diffSystemPromptSections and the tool declarations).
static Err pending_system(Run *r, const JValue *messages, JValue **out) {
    Arena *a = r->a;
    *out = NULL;
    JValue *want;
    TRY(code_system_message(a, &r->env, r->tools, r->ntools, &want));
    const JValue *cur = current_system(a, messages);
    const JValue *prev = cur ? json_get(cur, "sections") : NULL;
    const JValue *now = json_get(want, "sections");
    JValue *patch = json_new(a, J_OBJECT);
    for (size_t i = 0; now && i < now->obj.len; i++) {
        const JMember *m = &now->obj.items[i];
        const JValue *p = prev ? json_get(prev, str_z(a, m->key)) : NULL;
        if (!p || !str_eq(json_strs(p), json_strs(m->val))) json_put(a, patch, m->key, m->val);
    }
    for (size_t i = 0; prev && i < prev->obj.len; i++)
        if (!now || !json_get(now, str_z(a, prev->obj.items[i].key))) json_put(a, patch, prev->obj.items[i].key, json_new(a, J_NULL));
    // tools: the declarations not yet in the transcript
    JValue *schemas = code_tool_schemas(a, r->tools, r->ntools);
    const JValue *have = cur ? json_get(cur, "toolsAdded") : NULL;
    JValue *added = json_new(a, J_ARRAY);
    for (size_t i = 0; i < schemas->arr.len; i++) {
        JValue *d = declaration(a, schemas->arr.items[i]);
        Buf x = {0}, y = {0};
        json_write(&x, d);
        bool same = false;
        for (size_t k = 0; have && k < have->arr.len && !same; k++) {
            y.len = 0;
            json_write(&y, have->arr.items[k]);
            same = str_eq(buf_str(&x), buf_str(&y));
        }
        buf_free(&x);
        buf_free(&y);
        if (!same) json_push(a, added, d);
    }
    if (patch->obj.len == 0 && added->arr.len == 0) return OK;
    JValue *s = json_new(a, J_OBJECT);
    json_put(a, s, S("role"), json_string(a, S("system")));
    json_put(a, s, S("content"), json_string(a, S("")));
    if (patch->obj.len) json_put(a, s, S("sections"), patch);
    json_put(a, s, S("timestamp"), json_int(a, (int64_t)wall_ms()));
    if (added->arr.len) json_put(a, s, S("toolsAdded"), added);
    *out = s;
    return OK;
}

// The OpenAI tools for the tools the transcript declares, in declaration order.
static JValue *current_tools(Arena *a, const JValue *system) {
    const JValue *decl = system ? json_get(system, "toolsAdded") : NULL;
    size_t n = decl && json_is(decl, J_ARRAY) ? decl->arr.len : 0;
    const char **names = arena_alloc(a, (n + 1) * sizeof *names);
    for (size_t i = 0; i < n; i++) names[i] = json_str(json_get(decl->arr.items[i], "name"));
    return code_tool_schemas(a, names, n);
}

// ---- the agent loop (agent-loop.ts runLoop, streamAssistantResponse, executeToolCalls) ------------

static Err add_message(Run *r, JValue *m) {
    emit_message(r, "message_start", m);
    emit_message(r, "message_end", m);
    json_push(r->a, r->new_messages, m);
    return code_session_append(r->session, "message", ({
                                   JValue *f = json_new(r->a, J_OBJECT);
                                   json_put(r->a, f, S("message"), m);
                                   f;
                               }));
}

typedef struct {
    Run *r;
    bool started;
} Stream;

// pi's message_update, as json mode prints it: usage, and the event without its partial
// (toolcall_start gains the call's id and name).
static void on_event(void *ctx, const JValue *event, const JValue *partial) {
    Stream *st = ctx;
    Run *r = st->r;
    Arena *a = r->a;
    Str type = json_strs(json_get(event, "type"));
    if (str_eqz(type, "start")) {
        st->started = true;
        emit_message(r, "message_start", partial);
        return;
    }
    if (str_eqz(type, "done") || str_eqz(type, "error")) return;
    JValue *e = ev_new(a, "message_update");
    json_put(a, e, S("usage"), (JValue *)json_get(partial, "usage"));
    JValue *ame = json_new(a, J_OBJECT);
    for (size_t i = 0; i < event->obj.len; i++)
        if (!str_eqz(event->obj.items[i].key, "partial")) json_put(a, ame, event->obj.items[i].key, event->obj.items[i].val);
    if (str_eqz(type, "toolcall_start")) {
        const JValue *idx = json_get(event, "contentIndex");
        const JValue *content = json_get(partial, "content");
        const JValue *tc = json_is(idx, J_INTEGER) && content && (size_t)idx->i < content->arr.len ? content->arr.items[idx->i] : NULL;
        json_put(a, ame, S("id"), tc ? (JValue *)json_get(tc, "id") : json_string(a, S("")));
        json_put(a, ame, S("toolName"), tc ? (JValue *)json_get(tc, "name") : json_string(a, S("")));
    }
    json_put(a, e, S("assistantMessageEvent"), ame);
    emit(r, e);
}

static const char *level_of(const Run *r) { return r->model.thinking ? r->model.thinking : "off"; }

static Err stream_assistant(Run *r, JValue **out) {
    Arena *a = r->a;
    const JValue *all = code_session_messages(r->session);
    JValue *system = current_system(a, all);
    JValue *conv = json_new(a, J_ARRAY);
    for (size_t i = 0; i < all->arr.len; i++)
        if (!str_eqz(json_strs(json_get(all->arr.items[i], "role")), "system")) json_push(a, conv, all->arr.items[i]);
    Buf body = {0};
    TRY(code_oai_body(a, &r->model, system, conv, current_tools(a, system), &body));
    Stream st = {.r = r};
    CodeSink sink = {.on_event = on_event, .ctx = &st};
    r->env.now_ms = wall_ms();
    JValue *msg;
    Err e = code_oai_complete(a, &r->model, &r->env, buf_str(&body), &sink, &msg);
    buf_free(&body);
    TRY(e);
    json_put(a, msg, S("thinkingLevel"), json_string(a, str(level_of(r))));
    if (!st.started) emit_message(r, "message_start", msg);
    emit_message(r, "message_end", msg);
    json_push(a, r->new_messages, msg);
    JValue *f = json_new(a, J_OBJECT);
    json_put(a, f, S("message"), msg);
    TRY(code_session_append(r->session, "message", f));
    *out = msg;
    return OK;
}

static Err run_tools(Run *r, const JValue *assistant, JValue *results) {
    Arena *a = r->a;
    const JValue *content = json_get(assistant, "content");
    VEC(const JValue *) calls = {0};
    for (size_t i = 0; content && i < content->arr.len; i++)
        if (str_eqz(json_strs(json_get(content->arr.items[i], "type")), "toolCall")) vec_push(&calls, content->arr.items[i]);
    for (size_t i = 0; i < calls.len; i++) {
        JValue *e = ev_new(a, "tool_execution_start");
        json_put(a, e, S("toolCallId"), (JValue *)json_get(calls.ptr[i], "id"));
        json_put(a, e, S("toolName"), (JValue *)json_get(calls.ptr[i], "name"));
        json_put(a, e, S("args"), (JValue *)json_get(calls.ptr[i], "arguments"));
        emit(r, e);
    }
    JValue **msgs = arena_alloc(a, (calls.len + 1) * sizeof *msgs);
    Err err = OK;
    for (size_t i = 0; i < calls.len && !err; i++) {
        r->env.now_ms = wall_ms();
        JValue *raw = NULL;
        err = code_tool_run(a, &r->env, calls.ptr[i], &msgs[i], &raw);
        if (err) break;
        JValue *e = ev_new(a, "tool_execution_end");
        json_put(a, e, S("toolCallId"), (JValue *)json_get(calls.ptr[i], "id"));
        json_put(a, e, S("toolName"), (JValue *)json_get(calls.ptr[i], "name"));
        json_put(a, e, S("result"), raw ? raw : json_new(a, J_OBJECT));
        json_put(a, e, S("isError"), (JValue *)json_get(msgs[i], "isError"));
        const JValue *d = json_get(msgs[i], "durationMs");
        if (d) json_put(a, e, S("durationMs"), (JValue *)d);
        emit(r, e);
    }
    for (size_t i = 0; i < calls.len && !err; i++) {
        err = add_message(r, msgs[i]);
        json_push(a, results, msgs[i]);
    }
    vec_free(&calls);
    return err;
}

// One prompt: agent_start, the turns, agent_end (agentLoop + AgentSession.prompt).
static Err prompt(Run *r, const char *text) {
    Arena *a = r->a;
    r->new_messages = json_new(a, J_ARRAY);
    emit(r, ev_new(a, "agent_start"));
    emit(r, ev_new(a, "turn_start"));
    JValue *sys;
    TRY(pending_system(r, code_session_messages(r->session), &sys));
    if (sys) TRY(add_message(r, sys));
    JValue *user = json_new(a, J_OBJECT);
    json_put(a, user, S("role"), json_string(a, S("user")));
    JValue *content = json_new(a, J_ARRAY);
    JValue *block = json_new(a, J_OBJECT);
    json_put(a, block, S("type"), json_string(a, S("text")));
    json_put(a, block, S("text"), json_string(a, str(text)));
    json_push(a, content, block);
    json_put(a, user, S("content"), content);
    json_put(a, user, S("timestamp"), json_int(a, (int64_t)wall_ms()));
    TRY(add_message(r, user));
    for (bool first = true;; first = false) {
        if (!first) emit(r, ev_new(a, "turn_start"));
        JValue *msg;
        TRY(stream_assistant(r, &msg));
        Str stop = json_strs(json_get(msg, "stopReason"));
        JValue *results = json_new(a, J_ARRAY);
        bool more = false;
        if (!str_eqz(stop, "error") && !str_eqz(stop, "aborted")) {
            const JValue *c = json_get(msg, "content");
            for (size_t i = 0; c && i < c->arr.len; i++) more |= str_eqz(json_strs(json_get(c->arr.items[i], "type")), "toolCall");
            if (more) TRY(run_tools(r, msg, results));
        }
        JValue *te = ev_new(a, "turn_end");
        json_put(a, te, S("message"), msg);
        json_put(a, te, S("toolResults"), results);
        emit(r, te);
        if (!more) break;
    }
    JValue *ae = ev_new(a, "agent_end");
    json_put(a, ae, S("messages"), r->new_messages);
    json_put(a, ae, S("willRetry"), json_bool(a, false));
    emit(r, ae);
    JValue *settled = ev_new(a, "agent_settled");
    json_put(a, settled, S("aborted"), json_bool(a, false));
    emit(r, settled);
    return OK;
}

// ---- entry ---------------------------------------------------------------------------------------

// pi's package folder for the <docs> section: $SAETTA_CODE_PI_DOCS, else the managed install the
// `pi` on PATH belongs to (<agent>/bin/pi, <agent>/install/current-version), else none.
static const char *find_pi_docs(Arena *a) {
    const char *e = getenv("SAETTA_CODE_PI_DOCS");
    if (e) return e[0] ? e : NULL;
    const char *path = getenv("PATH");
    if (!path) return NULL;
    for (Str rest = str(path); rest.len;) {
        ptrdiff_t c = str_find_char(rest, ':', 0);
        Str dir = c >= 0 ? str_sub(rest, 0, (size_t)c) : rest;
        rest = c >= 0 ? str_sub(rest, (size_t)c + 1, rest.len) : S("");
        char *cand = path_join(a, str_z(a, dir), "pi");
        char real[PATH_MAX];
        if (!realpath(cand, real)) continue;
        char *bin = strrchr(real, '/');
        if (!bin || strcmp(bin, "/pi")) return NULL;
        *bin = 0;
        char *agent = strrchr(real, '/');
        if (!agent || strcmp(agent, "/bin")) return NULL;
        *agent = 0;
        char *install = path_join(a, real, "install");
        char *ver;
        size_t len;
        if (read_file(path_join(a, install, "current-version"), 256, &ver, &len)) return NULL;
        Str v = str_trim(str_n(ver, len));
        char *docs = arena_printf(a, "%s/releases/%.*s/node_modules/@earendil-works/pi-coding-agent", install, (int)v.len, v.ptr);
        free(ver);
        return file_exists(path_join(a, docs, "README.md")) ? docs : NULL;
    }
    return NULL;
}

// --tools a,b,c (pi's order as given), --no-tools, else the defaults
static void pick_tools(Arena *a, const Args *o, Run *r) {
    if (o->no_tools) {
        r->tools = NULL;
        r->ntools = 0;
        return;
    }
    if (!o->tools) {
        r->tools = (const char **)default_tools;
        r->ntools = COUNT(default_tools);
        return;
    }
    size_t cap = 1;
    for (const char *p = o->tools; *p; p++) cap += *p == ',';
    r->tools = arena_alloc(a, cap * sizeof *r->tools);
    for (Str rest = str(o->tools); rest.len;) {
        ptrdiff_t c = str_find_char(rest, ',', 0);
        Str t = str_trim(c >= 0 ? str_sub(rest, 0, (size_t)c) : rest);
        rest = c >= 0 ? str_sub(rest, (size_t)c + 1, rest.len) : S("");
        if (t.len) r->tools[r->ntools++] = str_z(a, t);
    }
}

Err code_run(const char *const *args, size_t n, int *code) {
    *code = 1;
    Arena arena = {0};
    Arena *a = &arena;
    Args o = {0};
    Err e = parse_args(args, n, &o);
    if (e) goto done;
    Run r = {.a = a, .json = o.mode && !strcmp(o.mode, "json")};
    char cwd[PATH_MAX];
    if (!getcwd(cwd, sizeof cwd)) {
        e = "NoCwd";
        goto done;
    }
    r.env.cwd = arena_strdup(a, cwd);
    r.env.home = getenv("HOME") ? getenv("HOME") : "/";
    const char *ad = getenv("PI_CODING_AGENT_DIR");
    if (ad && ad[0] == '~') ad = arena_printf(a, "%s%s", r.env.home, ad + 1);
    r.env.agent_dir = ad && ad[0] ? ad : path_join(a, path_join(a, r.env.home, ".pi"), "agent");
    r.env.now_ms = wall_ms();
    r.env.pi_docs = find_pi_docs(a);
    r.env.system_prompt = o.system_prompt;
    r.env.append_system_prompt = o.append.ptr;
    r.env.nappend_system_prompt = o.append.len;
    pick_tools(a, &o, &r);
    JValue *settings = read_json(a, path_join(a, r.env.agent_dir, "settings.json"));
    e = find_model(&r, &o, settings);
    if (e) {
        dbg("saetta code: no model %s%s%s in %s/models.json (Saetta Code runs openai-completions providers)\n",
            o.provider ? o.provider : "", o.provider ? "/" : "", o.model ? o.model : "", r.env.agent_dir);
        goto done;
    }
    e = code_session_open(a, &r.env, o.cont, o.no_session, &r.session);
    if (e) goto done;
    // thinking level: the flag, else the session's, else the settings' default, else medium
    const JValue *recorded = code_session_last(r.session, "thinking_level_change", "thinkingLevel");
    const char *level = o.thinking            ? o.thinking
                        : r.model.thinking    ? r.model.thinking
                        : json_str(recorded)  ? json_str(recorded)
                        : json_str(json_get(settings, "defaultThinkingLevel")) ? json_str(json_get(settings, "defaultThinkingLevel"))
                                                                               : "medium";
    if (!r.model.reasoning) level = "off";
    r.model.thinking = level;
    if (code_session_is_new(r.session)) {
        JValue *f = json_new(a, J_OBJECT);
        json_put(a, f, S("provider"), json_string(a, str(r.model.provider)));
        json_put(a, f, S("modelId"), json_string(a, str(r.model.id)));
        if ((e = code_session_append(r.session, "model_change", f))) goto done;
        f = json_new(a, J_OBJECT);
        json_put(a, f, S("thinkingLevel"), json_string(a, str(level)));
        if ((e = code_session_append(r.session, "thinking_level_change", f))) goto done;
    } else if (!json_str(recorded) || strcmp(json_str(recorded), level)) {
        JValue *f = json_new(a, J_OBJECT);
        json_put(a, f, S("thinkingLevel"), json_string(a, str(level)));
        if ((e = code_session_append(r.session, "thinking_level_change", f))) goto done;
    }
    if (r.json) emit(&r, (JValue *)code_session_header(r.session));
    for (size_t i = 0; i < o.prompts.len && !e; i++) e = prompt(&r, o.prompts.ptr[i]);
    if (e) goto done;
    *code = 0;
    if (!r.json) {
        const JValue *msgs = code_session_messages(r.session);
        const JValue *last = msgs->arr.len ? msgs->arr.items[msgs->arr.len - 1] : NULL;
        if (last && str_eqz(json_strs(json_get(last, "role")), "assistant")) {
            Str stop = json_strs(json_get(last, "stopReason"));
            if (str_eqz(stop, "error") || str_eqz(stop, "aborted")) {
                const char *m = json_str(json_get(last, "errorMessage"));
                fprintf(stderr, "%s\n", m ? m : arena_printf(a, "Request %.*s", (int)stop.len, stop.ptr));
                *code = 1;
            } else {
                const JValue *c = json_get(last, "content");
                for (size_t i = 0; c && i < c->arr.len; i++)
                    if (str_eqz(json_strs(json_get(c->arr.items[i], "type")), "text")) {
                        Str t = json_strs(json_get(c->arr.items[i], "text"));
                        fwrite(t.ptr, 1, t.len, stdout);
                        fputc('\n', stdout);
                    }
                if (str_eqz(stop, "length")) {
                    fprintf(stderr, "Response was truncated: the model reached its output token limit.\n");
                    *code = 1;
                }
            }
        }
    }
done:
    vec_free(&o.prompts);
    vec_free(&o.append);
    arena_reset(a);
    return e;
}
