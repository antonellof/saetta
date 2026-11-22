// `saetta code` (see code.h): the command line, print and json modes, the interactive prompt,
// the agent loop and its retries, and the in-process engine (--pack).
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi v1.5.2,
// b4afddd): packages/agent/src/agent-loop.ts (runLoop, streamAssistantResponse,
// executeToolCalls*, failToolCallsFromTruncatedMessage, declareToolChanges),
// packages/coding-agent/src/main.ts (readPipedStdin, the app mode), cli/args.ts,
// cli/initial-message.ts, core/agent-session.ts (the system prompt delta, _prepareRetry,
// _handlePostAgentRun, _omitRecoveryAttempt), modes/print-mode.ts (signals), modes/json-event.ts,
// utils/shell.ts (killTrackedDetachedChildren), packages/ai/src/utils/transcript.ts
// (getCurrentSystemMessage, getToolStateChanges), utils/retry.ts (the retryable patterns).
#include "code.h"
#include "code_autocompact.h"
#include "code_cfg.h"
#include "code_mcp.h"
#include "code_oai.h"
#include "code_res.h"
#include "code_sessions.h"
#include "code_tui.h"
#include "code_monitor.h"
#include "download.h"
#include "instance.h"
#include "server.h"

#include <ctype.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>
#include <mach-o/dyld.h>
#include <errno.h>
#include "fit.h"
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <sys/mount.h>
#include <sys/socket.h>
#include <sys/stat.h>

static const char *const default_tools[] = {"read", "bash", "edit", "write"};

typedef struct {
    const char *provider, *model, *thinking, *mode, *system_prompt, *tools, *pack, *ctx, *session;
    bool print, cont, no_session, no_tools, no_skills, no_mcp;
    int approve; // --approve 1, --no-approve 0, else -1
    VEC(const char *) skills;
    VEC(const char *) prompts;
    VEC(const char *) append;
} Args;

typedef struct {
    Arena *a;
    CodeEnv env;
    CodeModel model;
    CodeSession *session;
    Arena *session_arena; // the arena of a session loaded after the start (/resume, /import), else NULL
    const char **tools;
    size_t ntools;
    const char **base_tools; // the built-in ones (--tools); tools adds the MCP tools (code_mcp.c)
    size_t nbase_tools;
    bool json;
    bool ui; // the interactive mode prints the stream as it arrives
    int64_t retry_attempt; // the retries of the current prompt so far (pi's _retryAttempt)
    JValue *new_messages; // this run's messages, for agent_end
    const char *pack;     // --pack as given (a download name or a folder), NULL over HTTP
    const char *const *argv; // the command line after "code", for a restart on another model
    size_t argc;
    const JValue *call;   // the tool call running (for its partial results)
    char *restart_provider, *restart_id; // set by the interactive mode: start again on this model
    CodeAutoCompact ac;   // automatic compaction (code_autocompact.c)
    bool deliver_queued;  // the run continues to deliver queued messages (agent.continue())
} Run;

static double wall_ms(void) { return (double)(wall_ns() / 1000000); }

// ---- output --------------------------------------------------------------------------------------

static void emit(Run *r, JValue *ev) {
    if (r->ui) {
        code_ui_event(ev);
        return;
    }
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
        else if (!strcmp(x, "--pack") && more) o->pack = v[++i];
        else if (!strcmp(x, "--ctx") && more) o->ctx = v[++i];
        else if (!strcmp(x, "--session") && more) o->session = v[++i];
        else if (!strcmp(x, "--no-tools") || !strcmp(x, "-nt")) o->no_tools = true;
        else if (!strcmp(x, "--skill") && more) vec_push(&o->skills, v[++i]);
        else if (!strcmp(x, "--no-skills") || !strcmp(x, "-ns")) o->no_skills = true;
        else if (!strcmp(x, "--approve") || !strcmp(x, "-a")) o->approve = 1;
        else if (!strcmp(x, "--no-approve") || !strcmp(x, "-na")) o->approve = 0;
        else if (!strcmp(x, "--no-mcp")) o->no_mcp = true;
        else if (!strcmp(x, "--help") || !strcmp(x, "-h")) {
            printf("usage: saetta code [options] [prompt ...]\n"
                   "\n"
                   "Saetta Code, a coding agent compatible with pi 1.5.2: the same tools (read, bash,\n"
                   "edit, write; grep, find, ls with --tools), system prompt, AGENTS.md / CLAUDE.md files, skills and\n"
                   "session files, so pi and Saetta Code resume each other's sessions. It talks to saetta or any\n"
                   "OpenAI-compatible chat completions server named in pi's models.json ($PI_CODING_AGENT_DIR, else\n"
                   "~/.pi/agent). `saetta start code` starts the server and Saetta Code together.\n"
                   "\n"
                   "  -p, --print              answer the prompt and exit (text output)\n"
                   "  --mode json              print pi's JSON event stream\n"
                   "  -c, --continue           continue the most recent session in this folder\n"
                   "  --session FILE           continue this session file\n"
                   "  --no-session             do not save the session\n"
                   "  --provider P --model M   the model (M may be provider/id and end in :thinking)\n"
                   "  --thinking LEVEL         off, minimal, low, medium, high, xhigh, max\n"
                   "  --tools a,b,c | --no-tools   the tool set\n"
                   "  --no-mcp                 no MCP servers (pi's mcp.json) and no MCP tools\n"
                   "  --system-prompt T, --append-system-prompt T   text or a file\n"
                   "  --skill PATH             load a skill file or folder (can be repeated)\n"
                   "  --no-skills, -ns         no skill discovery (--skill paths still load)\n"
                   "  --approve, -a            trust this project for the session (--no-approve, -na: do not)\n"
                   "  --pack P                 run the model in this process (a download name such as coder, or a\n"
                   "                           folder): no server, no HTTP; the requests are the same\n"
                   "  --ctx N                  with --pack: the context in tokens (default 204800, as saetta start;\n"
                   "                           less when this Mac's memory holds less)\n"
                   "\n"
                   "Without -p in a terminal it opens the interactive screen: Enter sends, Shift+Enter (or a\n"
                   "trailing \\) starts a new line, \"/\" lists the commands, \"@\" completes file names, Escape stops a\n"
                   "turn, Ctrl-C clears, Ctrl-C twice or Ctrl-D quits.\n");
            exit(0);
        } else if (x[0] == '-' && x[1]) {
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

// resolveConfigValue (resolve-config-value.ts): "$NAME" / "${NAME}" interpolated, "!command"
// run, the rest literal (code_cfg.c).
static const char *config_value(Arena *a, const char *v) { return cfg_resolve_value(a, v); }

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
            cm->api_key = config_value(a, json_str(json_get(p->val, "apiKey")));
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
// constrainedSampling for the tools that declare it.
static JValue *declaration(Arena *a, const JValue *oai) {
    const JValue *f = json_get(oai, "function");
    JValue *d = json_new(a, J_OBJECT);
    json_put(a, d, S("name"), (JValue *)json_get(f, "name"));
    json_put(a, d, S("description"), (JValue *)json_get(f, "description"));
    json_put(a, d, S("parameters"), (JValue *)json_get(f, "parameters"));
    // read, bash, edit and write prefer JSON-schema constrained sampling (their definitions in
    // core/tools/*.ts); grep, find and ls declare none
    Str name = json_strs(json_get(f, "name"));
    if (str_eqz(name, "read") || str_eqz(name, "bash") || str_eqz(name, "edit") || str_eqz(name, "write")) {
        JValue *cs = json_new(a, J_OBJECT);
        json_put(a, cs, S("type"), json_string(a, S("json_schema")));
        json_put(a, cs, S("strict"), json_string(a, S("prefer")));
        json_put(a, d, S("constrainedSampling"), cs);
    }
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
    // tools (getToolStateChanges): the run's declarations the transcript lacks or holds
    // differently are added; the transcript's tools this run lacks or declares differently are
    // removed, by name
    JValue *schemas = code_tool_schemas(a, r->tools, r->ntools);
    const JValue *have = cur ? json_get(cur, "toolsAdded") : NULL;
    JValue *added = json_new(a, J_ARRAY), *removed = json_new(a, J_ARRAY);
    JValue **want_d = arena_alloc(a, (schemas->arr.len + 1) * sizeof *want_d);
    for (size_t i = 0; i < schemas->arr.len; i++) want_d[i] = declaration(a, schemas->arr.items[i]);
    Buf x = {0}, y = {0};
    for (size_t i = 0; i < schemas->arr.len; i++) {
        x.len = 0;
        json_write(&x, want_d[i]);
        bool same = false;
        for (size_t k = 0; have && k < have->arr.len && !same; k++) {
            y.len = 0;
            json_write(&y, have->arr.items[k]);
            same = str_eq(buf_str(&x), buf_str(&y));
        }
        if (!same) json_push(a, added, want_d[i]);
    }
    for (size_t k = 0; have && k < have->arr.len; k++) {
        y.len = 0;
        json_write(&y, have->arr.items[k]);
        bool same = false;
        for (size_t i = 0; i < schemas->arr.len && !same; i++) {
            x.len = 0;
            json_write(&x, want_d[i]);
            same = str_eq(buf_str(&x), buf_str(&y));
        }
        if (!same) {
            JValue *n = json_new(a, J_OBJECT);
            json_put(a, n, S("name"), (JValue *)json_get(have->arr.items[k], "name"));
            json_push(a, removed, n);
        }
    }
    buf_free(&x);
    buf_free(&y);
    if (patch->obj.len == 0 && added->arr.len == 0 && removed->arr.len == 0) return OK;
    JValue *s = json_new(a, J_OBJECT);
    json_put(a, s, S("role"), json_string(a, S("system")));
    json_put(a, s, S("content"), json_string(a, S("")));
    if (patch->obj.len) json_put(a, s, S("sections"), patch);
    json_put(a, s, S("timestamp"), json_int(a, (int64_t)wall_ms()));
    if (added->arr.len) json_put(a, s, S("toolsAdded"), added);
    if (removed->arr.len) json_put(a, s, S("toolsRemoved"), removed);
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
    code_autocompact_observe(&r->ac, m);
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

// ---- the interactive mode's state ------------------------------------------------------------------

const char *code_version = "dev";

static volatile int g_abort;

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
    // the interactive mode: the deltas go into the UI's ring as they are (no JSON, no allocation:
    // the per-token path); the UI rebuilds the partial message. The *_end events carry nothing
    // it shows (the final message comes with message_end).
    if (r->ui) {
        const JValue *idx = json_get(event, "contentIndex");
        long k = -1;
        if (json_is(idx, J_INTEGER)) k = (long)idx->i;
        else if (json_is(idx, J_FLOAT)) k = (long)idx->f;
        else if (json_is(idx, J_NUMBER_STRING)) {
            k = 0;
            for (size_t i = 0; i < idx->s.len && idx->s.ptr[i] >= '0' && idx->s.ptr[i] <= '9'; i++) k = k * 10 + (idx->s.ptr[i] - '0');
        }
        Str delta = json_strs(json_get(event, "delta"));
        if (str_eqz(type, "text_start")) code_ui_stream(UI_S_TEXT_START, (int)k, S(""), S(""));
        else if (str_eqz(type, "text_delta")) code_ui_stream(UI_S_TEXT, (int)k, delta, S(""));
        else if (str_eqz(type, "thinking_start")) code_ui_stream(UI_S_THINK_START, (int)k, S(""), S(""));
        else if (str_eqz(type, "thinking_delta")) code_ui_stream(UI_S_THINK, (int)k, delta, S(""));
        else if (str_eqz(type, "toolcall_start")) {
            const JValue *content = json_get(partial, "content");
            const JValue *tc = k >= 0 && json_is(content, J_ARRAY) && (size_t)k < content->arr.len ? content->arr.items[k] : NULL;
            code_ui_stream(UI_S_TOOL_START, (int)k, json_strs(json_get(tc, "id")), json_strs(json_get(tc, "name")));
        } else if (str_eqz(type, "toolcall_delta")) code_ui_stream(UI_S_TOOL_ARGS, (int)k, delta, S(""));
        return;
    }
    JValue *e = ev_new(a, "message_update");
    json_put(a, e, S("usage"), (JValue *)json_get(partial, "usage"));
    JValue *ame = json_new(a, J_OBJECT);
    for (size_t i = 0; i < event->obj.len; i++)
        if (!str_eqz(event->obj.items[i].key, "partial")) json_put(a, ame, event->obj.items[i].key, event->obj.items[i].val);
    if (str_eqz(type, "toolcall_start")) {
        const JValue *idx = json_get(event, "contentIndex");
        const JValue *content = json_get(partial, "content");
        // the client writes numbers as JavaScript does (J_NUMBER_STRING)
        long k = json_is(idx, J_INTEGER) ? (long)idx->i : json_is(idx, J_NUMBER_STRING) ? strtol(str_z(a, idx->s), NULL, 10) : -1;
        const JValue *tc = k >= 0 && content && (size_t)k < content->arr.len ? content->arr.items[k] : NULL;
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
    code_autocompact_observe(&r->ac, msg);
    if (!st.started) emit_message(r, "message_start", msg);
    emit_message(r, "message_end", msg);
    // a successful answer ends the retries at once (AgentSession._handleAgentEvent)
    if (r->retry_attempt > 0 && !str_eqz(json_strs(json_get(msg, "stopReason")), "error")) {
        JValue *end = ev_new(a, "auto_retry_end");
        json_put(a, end, S("success"), json_bool(a, true));
        json_put(a, end, S("attempt"), json_int(a, r->retry_attempt));
        emit(r, end);
        r->retry_attempt = 0;
    }
    json_push(a, r->new_messages, msg);
    JValue *f = json_new(a, J_OBJECT);
    json_put(a, f, S("message"), msg);
    TRY(code_session_append(r->session, "message", f));
    *out = msg;
    return OK;
}

// The part of the edit tool's prepareArguments (edit.ts prepareEditArguments) that rewrites the
// call's own arguments object in place: edits sent as a JSON string, or a single edit object,
// become an array. pi's session line was written before, so the file keeps the original while
// later requests of this process send the rewritten call; a resumed session sends the original.
static bool single_edit(const JValue *v) {
    return json_is(v, J_OBJECT) && json_is(json_get(v, "oldText"), J_STRING) && json_is(json_get(v, "newText"), J_STRING);
}

static void prepare_edit_in_place(Arena *a, const JValue *call) {
    if (!str_eqz(json_strs(json_get(call, "name")), "edit")) return;
    JValue *args = (JValue *)json_get(call, "arguments");
    if (!json_is(args, J_OBJECT)) return;
    JValue *edits = (JValue *)json_get(args, "edits");
    JValue *repl = NULL;
    if (json_is(edits, J_STRING)) {
        char *err = NULL;
        JValue *parsed = code_oai_js_parse(a, edits->s, &err);
        if (parsed && json_is(parsed, J_ARRAY)) repl = parsed;
        else if (parsed && single_edit(parsed)) {
            repl = json_new(a, J_ARRAY);
            json_push(a, repl, parsed);
        }
    } else if (single_edit(edits)) {
        repl = json_new(a, J_ARRAY);
        json_push(a, repl, edits);
    }
    if (repl) json_put(a, args, S("edits"), repl);
}

static void emit_tool_end(Run *r, const JValue *call, JValue *msg, JValue *raw) {
    Arena *a = r->a;
    JValue *e = ev_new(a, "tool_execution_end");
    json_put(a, e, S("toolCallId"), (JValue *)json_get(call, "id"));
    json_put(a, e, S("toolName"), (JValue *)json_get(call, "name"));
    json_put(a, e, S("result"), raw ? raw : json_new(a, J_OBJECT));
    json_put(a, e, S("isError"), (JValue *)json_get(msg, "isError"));
    const JValue *d = json_get(msg, "durationMs");
    if (d) json_put(a, e, S("durationMs"), (JValue *)d);
    emit(r, e);
}

// createErrorToolResult + createToolResultMessage for a call that is not run.
static void error_result(Arena *a, const JValue *call, const char *text, JValue **msg, JValue **raw) {
    JValue *content = json_new(a, J_ARRAY);
    JValue *block = json_new(a, J_OBJECT);
    json_put(a, block, S("type"), json_string(a, S("text")));
    json_put(a, block, S("text"), json_string(a, str(text)));
    json_push(a, content, block);
    *raw = json_new(a, J_OBJECT);
    json_put(a, *raw, S("content"), content);
    json_put(a, *raw, S("details"), json_new(a, J_OBJECT));
    JValue *m = json_new(a, J_OBJECT);
    json_put(a, m, S("role"), json_string(a, S("toolResult")));
    json_put(a, m, S("toolCallId"), (JValue *)json_get(call, "id"));
    json_put(a, m, S("toolName"), (JValue *)json_get(call, "name"));
    json_put(a, m, S("content"), content);
    json_put(a, m, S("details"), json_new(a, J_OBJECT));
    json_put(a, m, S("isError"), json_bool(a, true));
    json_put(a, m, S("timestamp"), json_int(a, (int64_t)wall_ms()));
    *msg = m;
}

// tool_execution_update (emitToolExecutionUpdate): a tool's partial result, built in a throwaway
// arena (a long command sends one every 100 ms)
static void on_tool_partial(void *ctx, const JValue *partial) {
    Run *r = ctx;
    if (!r->call || (!r->ui && !r->json)) return;
    Arena t = {0};
    JValue *e = ev_new(&t, "tool_execution_update");
    json_put(&t, e, S("toolCallId"), (JValue *)json_get(r->call, "id"));
    json_put(&t, e, S("toolName"), (JValue *)json_get(r->call, "name"));
    json_put(&t, e, S("args"), (JValue *)json_get(r->call, "arguments"));
    json_put(&t, e, S("partialResult"), (JValue *)partial);
    emit(r, e);
    arena_reset(&t);
}

static void emit_tool_start(Run *r, const JValue *call) {
    JValue *e = ev_new(r->a, "tool_execution_start");
    json_put(r->a, e, S("toolCallId"), (JValue *)json_get(call, "id"));
    json_put(r->a, e, S("toolName"), (JValue *)json_get(call, "name"));
    json_put(r->a, e, S("args"), (JValue *)json_get(call, "arguments"));
    emit(r, e);
}

// failToolCallsFromTruncatedMessage: a message cut by the output limit runs none of its calls.
static Err fail_truncated(Run *r, const JValue *assistant, JValue *results) {
    Arena *a = r->a;
    const JValue *content = json_get(assistant, "content");
    for (size_t i = 0; content && i < content->arr.len; i++) {
        const JValue *call = content->arr.items[i];
        if (!str_eqz(json_strs(json_get(call, "type")), "toolCall")) continue;
        emit_tool_start(r, call);
        Str name = json_strs(json_get(call, "name"));
        JValue *msg, *raw;
        error_result(a, call,
                     arena_printf(a,
                                  "Tool call \"%.*s\" was not executed: the response hit the output token limit, so its arguments may be "
                                  "truncated. Re-issue the tool call with complete arguments.",
                                  (int)name.len, name.ptr),
                     &msg, &raw);
        emit_tool_end(r, call, msg, raw);
        TRY(add_message(r, msg));
        json_push(a, results, msg);
    }
    return OK;
}

// The run's tool set (pi looks a call up in context.tools: a call to any other is "Tool <name>
// not found", whatever the tools module knows or an older part of the transcript declared).
static bool declared(const Run *r, Str name) {
    for (size_t i = 0; i < r->ntools; i++)
        if (str_eqz(name, r->tools[i])) return true;
    return false;
}

// executeToolCallsParallel: each call's tool_execution_start in order; a call that fails before
// running (unknown tool, bad arguments: its result has no durationMs) ends at once, the executed
// ones end after every start (failures first); then the results, in call order.
static Err run_tools(Run *r, const JValue *assistant, JValue *results) {
    Arena *a = r->a;
    const JValue *content = json_get(assistant, "content");
    VEC(const JValue *) calls = {0};
    for (size_t i = 0; content && i < content->arr.len; i++)
        if (str_eqz(json_strs(json_get(content->arr.items[i], "type")), "toolCall")) vec_push(&calls, content->arr.items[i]);
    JValue **msgs = arena_alloc(a, (calls.len + 1) * sizeof *msgs);
    JValue **raws = arena_alloc(a, (calls.len + 1) * sizeof *raws);
    Err err = OK;
    size_t started = 0;
    for (size_t i = 0; i < calls.len && !err; i++) {
        // aborted: no further call starts (executeToolCallsParallel breaks out of its loop)
        if (g_abort && i > 0) break;
        started = i + 1;
        emit_tool_start(r, calls.ptr[i]);
        r->env.now_ms = wall_ms();
        Str name = json_strs(json_get(calls.ptr[i], "name"));
        if (!declared(r, name)) {
            error_result(a, calls.ptr[i], arena_printf(a, "Tool %.*s not found", (int)name.len, name.ptr), &msgs[i], &raws[i]);
            emit_tool_end(r, calls.ptr[i], msgs[i], raws[i]);
            continue;
        }
        prepare_edit_in_place(a, calls.ptr[i]);
        r->call = calls.ptr[i];
        err = code_tool_run(a, &r->env, calls.ptr[i], &msgs[i], &raws[i]);
        r->call = NULL;
        if (!err && !json_get(msgs[i], "durationMs")) emit_tool_end(r, calls.ptr[i], msgs[i], raws[i]);
    }
    // pi runs the executed calls concurrently and ends each as it settles; a call that throws
    // (a missing file, say) settles before one that completes its I/O, so errors end first
    calls.len = started;
    for (int pass = 0; pass < 2; pass++)
        for (size_t i = 0; i < calls.len && !err; i++) {
            if (!json_get(msgs[i], "durationMs")) continue;
            bool failed = jbool(json_get(msgs[i], "isError"), false);
            if (failed == (pass == 0)) emit_tool_end(r, calls.ptr[i], msgs[i], raws[i]);
        }
    for (size_t i = 0; i < calls.len && !err; i++) {
        err = add_message(r, msgs[i]);
        json_push(a, results, msgs[i]);
    }
    vec_free(&calls);
    return err;
}

// ---- retries (ai/src/utils/retry.ts isRetryableAssistantError, agent-session.ts _prepareRetry) -----

// A case-insensitive match of pat anywhere in hay, where pat has literal characters, '.' (any
// character) and '?' after an item (optional): the subset of regex pi's patterns use.
static bool pm_here(const char *h, const char *p) {
    if (!*p) return true;
    bool opt = p[1] == '?';
    const char *next = p + (opt ? 2 : 1);
    bool one = *h && (*p == '.' || tolower((unsigned char)*h) == tolower((unsigned char)*p));
    if (one && pm_here(h + 1, next)) return true;
    return opt && pm_here(h, next);
}

static bool pm(const char *hay, const char *pat) {
    for (const char *h = hay;; h++) {
        if (pm_here(h, pat)) return true;
        if (!*h) return false;
    }
}

// an HTTP status code not touching a letter or digit: (?<![a-z0-9])(?:429|500|...)(?![a-z0-9])
static bool status_code(const char *m) {
    static const char *const codes[] = {"429", "500", "502", "503", "504", "520", "524"};
    for (size_t k = 0; k < COUNT(codes); k++)
        for (const char *p = strstr(m, codes[k]); p; p = strstr(p + 1, codes[k])) {
            bool before = p > m && isalnum((unsigned char)p[-1]);
            bool after = isalnum((unsigned char)p[3]);
            if (!before && !after) return true;
        }
    return false;
}

// agent-session.ts _isRetryableError: a context overflow goes to compaction, never to a retry;
// else isRetryableAssistantError. window: the model's contextWindow (_modelForMessage ?? model).
static bool retryable(const JValue *msg, uint64_t window) {
    if (code_context_overflow(msg, window)) return false;
    if (!str_eqz(json_strs(json_get(msg, "stopReason")), "error")) return false;
    const char *m = json_str(json_get(msg, "errorMessage"));
    if (!m || !*m) return false;
    static const char *const never[] = {"GoUsageLimitError", "FreeUsageLimitError", "Monthly usage limit reached", "available balance",
                                        "insufficient_quota", "out of budget", "quota exceeded", "billing",
                                        "subscription_sharing_usage_limit_exceeded",
                                        "subscription_sharing_workplace_individual_usage_limit_exceeded"};
    for (size_t k = 0; k < COUNT(never); k++)
        if (pm(m, never[k])) return false;
    // '"code"\s*:\s*"?1308(?![0-9])'
    for (const char *p = strcasestr(m, "\"code\""); p; p = strcasestr(p + 1, "\"code\"")) {
        const char *t = p + 6;
        while (isspace((unsigned char)*t)) t++;
        if (*t != ':') continue;
        t++;
        while (isspace((unsigned char)*t)) t++;
        if (*t == '"') t++;
        if (!strncmp(t, "1308", 4) && !isdigit((unsigned char)t[4])) return false;
    }
    if (status_code(m)) return true;
    static const char *const yes[] = {
        "overloaded", "server_busy", "servers are currently busy", "currently experiencing high demand", "model is at capacity",
        "rate.?limit", "too many requests", "service.?unavailable", "service.?is.?unavailable", "service.?temporarily.?unavailable",
        "service.?is.?temporarily.?unavailable", "server.?error", "internal.?error", "provider.?returned.?error",
        "provider.?returned.?an.?error", "exceeded request buffer limit while retrying upstream", "unable to forward upstream",
        "upstream.?request.?failed", "upstream_stream_read_error", "stream was interrupted", "http/?2 stream failed",
        "network.?error", "connection.?error", "connection.?refused", "connection.?lost", "other side closed", "fetch failed",
        "getaddrinfo", "ENOTFOUND", "EAI_AGAIN", "upstream.?connect", "reset before headers", "socket hang up",
        "socket connection was closed", "timed? out", "timeout", "terminated", "websocket.?closed", "websocket.?error",
        "ended without", "stream ended before message_stop", "stream ended before a terminal response event",
        "http2 request did not get a response", "stream disconnected before completion", "cut short after the reply had begun",
        "pending stream has been canceled", "retry delay", "you can retry your request", "try your request again",
        "please retry your request", "ResourceExhausted", "subscription_sharing_usage_unavailable",
        "subscription_sharing_user_unavailable"};
    for (size_t k = 0; k < COUNT(yes); k++)
        if (pm(m, yes[k])) return true;
    // "request to \S+ failed, reason:"
    const char *q = strcasestr(m, "request to ");
    if (q) {
        const char *t = q + 11;
        if (*t && !isspace((unsigned char)*t)) {
            while (*t && !isspace((unsigned char)*t)) t++;
            if (!strncasecmp(t, " failed, reason:", 16)) return true;
        }
    }
    return false;
}

// settings.retry with pi's defaults (settings-defaults.ts): enabled, 3 retries, 2 s doubling, 60 s cap
typedef struct {
    bool enabled;
    int64_t max_retries, base_ms, cap_ms;
} RetryPolicy;

static RetryPolicy g_retry = {true, 3, 2000, 60000};

static Err run_agent(Run *r, bool started);
static bool finish_agent(Run *r, bool will_retry, const JValue *last, const char *last_id);
static Err run_turns(Run *r, JValue **last_out, const char **last_id);

static void sleep_abortable(int64_t ms) {
    for (int64_t t = 0; t < ms && !g_abort; t += 50) {
        struct timespec ts = {0, 50 * 1000000L};
        nanosleep(&ts, NULL);
    }
}

// A user message of this text, as pi records it.
static JValue *user_message(Arena *a, const char *text) {
    JValue *user = json_new(a, J_OBJECT);
    json_put(a, user, S("role"), json_string(a, S("user")));
    JValue *content = json_new(a, J_ARRAY);
    JValue *block = json_new(a, J_OBJECT);
    json_put(a, block, S("type"), json_string(a, S("text")));
    json_put(a, block, S("text"), json_string(a, code_utf8_clean(a, str(text))));
    json_push(a, content, block);
    json_put(a, user, S("content"), content);
    json_put(a, user, S("timestamp"), json_int(a, (int64_t)wall_ms()));
    return user;
}

// The tools of this prompt: the built-in ones, then the MCP tools declared now (the first prompt
// waits for their servers first, pi's before_agent_start).
static void mcp_tools(Run *r) {
    code_mcp_before_prompt();
    const char **m;
    size_t nm = code_mcp_declared(r->a, &m);
    const char **all = arena_alloc(r->a, (r->nbase_tools + nm + 1) * sizeof *all);
    for (size_t i = 0; i < r->nbase_tools; i++) all[i] = r->base_tools[i];
    for (size_t i = 0; i < nm; i++) all[r->nbase_tools + i] = m[i];
    r->tools = all;
    r->ntools = r->nbase_tools + nm;
}

// The automatic compaction's view of the run (code_autocompact.c).
static void ac_emit(void *ctx, JValue *ev) { emit(ctx, ev); }

static bool ac_has_queued(void *ctx) { return ((Run *)ctx)->ui && code_ui_has_queued(); }

static void ac_bind(Run *r) {
    r->ac.a = r->a;
    r->ac.session = &r->session;
    r->ac.model = &r->model;
    r->ac.env = &r->env;
    r->ac.emit = ac_emit;
    r->ac.has_queued = ac_has_queued;
    r->ac.ctx = r;
}

// One prompt: agent_start, the turns, agent_end (agentLoop + AgentSession.prompt).
static Err prompt(Run *r, const char *text) {
    Arena *a = r->a;
    ac_bind(r);
    // _prepareRun: compact first when needed (catches aborted responses)
    code_autocompact_before_prompt(&r->ac);
    mcp_tools(r);
    r->new_messages = json_new(a, J_ARRAY);
    emit(r, ev_new(a, "agent_start"));
    emit(r, ev_new(a, "turn_start"));
    JValue *sys;
    TRY(pending_system(r, code_session_messages(r->session), &sys));
    if (sys) TRY(add_message(r, sys));
    TRY(add_message(r, user_message(a, code_skill_expand(a, &r->env, text, NULL)))); // /skill:name
    return run_agent(r, true);
}

// The agent loop until it ends, then the retry policy: a retryable error is kept in the file,
// omitted from the model's context with a context_edit entry, and the turn runs again after
// the backoff (agent_start .. agent_end each time, as pi's continue does).
static Err run_agent(Run *r, bool started) {
    Arena *a = r->a;
    r->retry_attempt = 0;
    for (;;) {
        if (!started) {
            r->new_messages = json_new(a, J_ARRAY);
            emit(r, ev_new(a, "agent_start"));
            emit(r, ev_new(a, "turn_start"));
            // agent.continue() after an auto-compaction with messages queued: they start the run
            char *queued = r->deliver_queued ? code_ui_take_queued(true) : NULL;
            if (r->deliver_queued && !queued) queued = code_ui_take_queued(false);
            r->deliver_queued = false;
            if (queued) {
                JValue *user = user_message(a, code_skill_expand(a, &r->env, queued, NULL));
                free(queued);
                TRY(add_message(r, user));
            }
        }
        started = false;
        JValue *last = NULL;
        const char *last_id = NULL;
        TRY(run_turns(r, &last, &last_id));
        bool again = false;
        if (last && retryable(last, r->model.context_window) && g_retry.enabled && !g_abort && r->retry_attempt < g_retry.max_retries) {
            int64_t attempt = ++r->retry_attempt;
            // baseDelayMs * 2^(attempt-1), capped (and never shifted past the cap)
            int64_t delay = g_retry.base_ms;
            for (int64_t k = 1; k < attempt && delay < g_retry.cap_ms; k++) delay = delay > INT64_MAX / 2 ? INT64_MAX : delay * 2;
            if (delay > g_retry.cap_ms) delay = g_retry.cap_ms;
            JValue *e = ev_new(a, "auto_retry_start");
            json_put(a, e, S("attempt"), json_int(a, attempt));
            json_put(a, e, S("maxAttempts"), json_int(a, g_retry.max_retries));
            json_put(a, e, S("delayMs"), json_int(a, delay));
            const char *em = json_str(json_get(last, "errorMessage"));
            json_put(a, e, S("errorMessage"), json_string(a, str(em ? em : "Unknown error")));
            // agent_end (willRetry) is emitted before the retry starts
            finish_agent(r, true, NULL, NULL);
            emit(r, e);
            JValue *f = json_new(a, J_OBJECT);
            json_put(a, f, S("targetId"), json_string(a, str(last_id)));
            json_put(a, f, S("replacement"), json_new(a, J_NULL));
            TRY(code_session_append(r->session, "context_edit", f));
            JValue *ea = ev_new(a, "entry_appended");
            json_put(a, ea, S("entry"), (JValue *)code_session_entry_last(r->session));
            emit(r, ea);
            sleep_abortable(delay);
            again = !g_abort;
            if (!again) {
                JValue *end = ev_new(a, "auto_retry_end");
                json_put(a, end, S("success"), json_bool(a, false));
                json_put(a, end, S("attempt"), json_int(a, r->retry_attempt));
                json_put(a, end, S("finalError"), json_string(a, S("Retry cancelled")));
                emit(r, end);
                JValue *settled = ev_new(a, "agent_settled");
                json_put(a, settled, S("aborted"), json_bool(a, true));
                emit(r, settled);
                return OK;
            }
            continue;
        }
        if (finish_agent(r, false, last, last_id)) continue; // the overflow retry, or queued messages
        (void)again;
        return OK;
    }
}

// agent_end, then _handlePostAgentRun's end of the retries and _checkCompaction: true when the
// run continues (agent.continue()), else agent_settled.
static bool finish_agent(Run *r, bool will_retry, const JValue *last, const char *last_id) {
    Arena *a = r->a;
    JValue *ae = ev_new(a, "agent_end");
    json_put(a, ae, S("messages"), r->new_messages);
    json_put(a, ae, S("willRetry"), json_bool(a, will_retry));
    emit(r, ae);
    if (will_retry) return false;
    // retries used up on an error (_handlePostAgentRun)
    if (r->retry_attempt > 0 && last && str_eqz(json_strs(json_get(last, "stopReason")), "error")) {
        JValue *end = ev_new(a, "auto_retry_end");
        json_put(a, end, S("success"), json_bool(a, false));
        json_put(a, end, S("attempt"), json_int(a, r->retry_attempt));
        json_put(a, end, S("finalError"), (JValue *)json_get(last, "errorMessage"));
        emit(r, end);
        r->retry_attempt = 0;
    }
    if (last && !g_abort && code_autocompact_check(&r->ac, last, last_id, true)) {
        // the overflow retry continues from the context (the failed answer left out of it);
        // after an answer, the queued messages start the run
        const JValue *msgs = code_session_messages(r->session);
        r->deliver_queued = msgs->arr.len && str_eqz(json_strs(json_get(msgs->arr.items[msgs->arr.len - 1], "role")), "assistant");
        return true;
    }
    JValue *settled = ev_new(a, "agent_settled");
    json_put(a, settled, S("aborted"), json_bool(a, g_abort != 0));
    emit(r, settled);
    return false;
}

// The turns of one agent run (runLoop): *last the final assistant message and its entry id.
static bool switch_model(Run *r, const char *provider, const char *id);
static void apply_thinking(Run *r, const char *level);

// What the user chose in the interactive mode while the run went on: applied here, between turns,
// on this thread (pi's setModel takes effect at the next request).
static void ui_requests(Run *r) {
    if (!r->ui) return;
    char *p, *m;
    if (code_ui_take_model(&p, &m)) {
        if (!switch_model(r, p, m)) {
            Arena t = {0};
            JValue *ev = ev_new(&t, "ui_error");
            json_put(&t, ev, S("message"), json_string(&t, str(arena_printf(&t, "Could not switch to %s/%s", p, m))));
            code_ui_event(ev);
            arena_reset(&t);
        }
        free(p);
        free(m);
    }
    char *lv = code_ui_take_thinking();
    if (lv) {
        apply_thinking(r, lv);
        free(lv);
    }
}

static Err run_turns(Run *r, JValue **last_out, const char **last_id) {
    Arena *a = r->a;
    bool turn_started = false;
    for (bool first = true;; first = false) {
        if (!first && !turn_started) emit(r, ev_new(a, "turn_start"));
        turn_started = false;
        ui_requests(r);
        JValue *msg;
        TRY(stream_assistant(r, &msg));
        *last_out = msg;
        *last_id = code_session_leaf(r->session);
        Str stop = json_strs(json_get(msg, "stopReason"));
        JValue *results = json_new(a, J_ARRAY);
        bool more = false;
        if (!str_eqz(stop, "error") && !str_eqz(stop, "aborted")) {
            const JValue *c = json_get(msg, "content");
            for (size_t i = 0; c && i < c->arr.len; i++) more |= str_eqz(json_strs(json_get(c->arr.items[i], "type")), "toolCall");
            if (more && str_eqz(stop, "length")) TRY(fail_truncated(r, msg, results));
            else if (more) TRY(run_tools(r, msg, results));
        }
        JValue *te = ev_new(a, "turn_end");
        json_put(a, te, S("message"), msg);
        json_put(a, te, S("toolResults"), results);
        emit(r, te);
        if (g_abort || str_eqz(stop, "error") || str_eqz(stop, "aborted")) break;
        // the interactive mode's queues (runLoop's getSteeringMessages, then getFollowUpMessages
        // when the run would end): the message joins the run before its next turn
        char *queued = r->ui ? code_ui_take_queued(true) : NULL;
        if (!queued && !more && r->ui) queued = code_ui_take_queued(false);
        if (!more && !queued) break;
        // prepareNextTurn: _compactBeforeNextAssistantResponse
        code_autocompact_before_turn(&r->ac);
        if (queued) {
            emit(r, ev_new(a, "turn_start"));
            JValue *user = user_message(a, code_skill_expand(a, &r->env, queued, NULL));
            free(queued);
            TRY(add_message(r, user));
            turn_started = true;
            continue;
        }
    }
    return OK;
}

// ---- interactive mode (code_ui.c draws pi's screen; this is the agent's side) ----------------------

static struct termios g_cooked;
static volatile sig_atomic_t g_tty_saved;

// SIGTERM / SIGHUP: the terminal back as it was, the running bash commands stopped, then the exit
// pi makes: 0 from the interactive mode (interactive-mode.ts), 143 / 129 from print and json
// mode (print-mode.ts)
static volatile sig_atomic_t g_interactive;

static void on_term(int sig) {
    // everything the screen turned on (mouse, keyboard protocols, the alternate screen, raw mode)
    screen_restore_signal_safe();
    if (g_tty_saved) tcsetattr(0, TCSANOW, &g_cooked);
    code_tools_kill_detached();
    _exit(g_interactive ? 0 : sig == SIGHUP ? 129 : 143);
}

struct CodeUiHost {
    Run *r;
};

static void host_prompt(CodeUiHost *h, const char *text) {
    Run *r = h->r;
    g_abort = 0;
    Err e = prompt(r, text);
    if (e) {
        Arena a = {0};
        JValue *ev = ev_new(&a, "ui_error");
        json_put(&a, ev, S("message"), json_string(&a, str(arena_printf(&a, "saetta code: %s", e))));
        code_ui_event(ev);
        arena_reset(&a);
    }
}

static void host_new_session(CodeUiHost *h) {
    Run *r = h->r;
    Arena *a = r->a;
    if (code_session_open(a, &r->env, false, false, &r->session)) return;
    r->env.session_id = json_str(json_get(code_session_header(r->session), "id"));
    r->env.session_file = code_session_path(r->session);
    JValue *f = json_new(a, J_OBJECT);
    json_put(a, f, S("provider"), json_string(a, str(r->model.provider)));
    json_put(a, f, S("modelId"), json_string(a, str(r->model.id)));
    (void)code_session_append(r->session, "model_change", f);
    f = json_new(a, J_OBJECT);
    json_put(a, f, S("thinkingLevel"), json_string(a, str(r->model.thinking)));
    (void)code_session_append(r->session, "thinking_level_change", f);
}

static void host_set_thinking(CodeUiHost *h, const char *level) { apply_thinking(h->r, level); }

static void apply_thinking(Run *r, const char *level) {
    Arena *a = r->a;
    if (r->model.thinking && !strcmp(r->model.thinking, level)) return;
    r->model.thinking = arena_strdup(a, level);
    r->env.thinking = r->model.thinking;
    JValue *f = json_new(a, J_OBJECT);
    json_put(a, f, S("thinkingLevel"), json_string(a, str(level)));
    (void)code_session_append(r->session, "thinking_level_change", f);
}

static const JValue *host_entries(CodeUiHost *h, const char **leaf) {
    *leaf = code_session_leaf(h->r->session);
    return code_session_entries(h->r->session);
}

static const JValue *host_messages(CodeUiHost *h) { return code_session_messages(h->r->session); }

static void host_set_leaf(CodeUiHost *h, const char *id) { code_session_set_leaf(h->r->session, id); }

static bool host_set_model(CodeUiHost *h, const char *provider, const char *id) { return switch_model(h->r, provider, id); }

// ---- the session commands' side (code_sessions_ui.c draws them; code_sessions.c does the work) ----

static CodeSession *host_session(CodeUiHost *h) { return h->r->session; }

// createAgentSession for a session made current (/fork, /clone, /resume): its thinking level
// restored from the branch (or recorded when the branch has none), the model and thinking level
// recorded in a session without messages.
static void use_session(Run *r, CodeSession *s) {
    Arena *a = r->a;
    r->session = s;
    r->env.session_id = json_str(json_get(code_session_header(s), "id"));
    r->env.session_file = code_session_path(s);
    bool existing = code_session_messages(s)->arr.len > 0;
    const char *recorded = json_str(code_session_last(s, "thinking_level_change", "thinkingLevel"));
    const char *level = r->model.thinking ? r->model.thinking : "medium";
    if (existing && recorded) level = recorded;
    else if (existing) {
        JValue *settings = read_json(a, path_join(a, r->env.agent_dir, "settings.json"));
        const char *d = json_str(json_get(settings, "defaultThinkingLevel"));
        level = d ? d : "medium";
    }
    if (!r->model.reasoning) level = "off";
    level = arena_strdup(a, level);
    if (!existing) {
        JValue *f = json_new(a, J_OBJECT);
        json_put(a, f, S("provider"), json_string(a, str(r->model.provider)));
        json_put(a, f, S("modelId"), json_string(a, str(r->model.id)));
        (void)code_session_append(s, "model_change", f);
    }
    if (!existing || !recorded) {
        JValue *f = json_new(a, J_OBJECT);
        json_put(a, f, S("thinkingLevel"), json_string(a, str(level)));
        (void)code_session_append(s, "thinking_level_change", f);
    }
    r->model.thinking = level;
    r->env.thinking = level;
}

// A session loaded after the start lives in an arena of its own, freed when the next one replaces
// it (a long session that resumes and imports many keeps one tree in memory, not all of them).
static Arena *session_arena_new(void) { return xcalloc(1, sizeof(Arena)); }

static void session_arena_drop(Arena *sa) {
    if (!sa) return;
    arena_reset(sa);
    free(sa);
}

// The session in sa made current; the previous loaded one's arena freed.
static void session_arena_use(Run *r, Arena *sa) {
    Arena *old = r->session_arena;
    r->session_arena = sa;
    if (old != sa) session_arena_drop(old);
}

// A fork may share its parent's entries: it stays in the run's arena, and the parent's arena (a
// loaded session's) is kept until the next /resume or /import replaces them both.
static const char *host_fork(CodeUiHost *h, const char *entry_id, bool at, char **selected) {
    Run *r = h->r;
    CodeSession *s;
    const char *err = code_sessions_fork(r->a, r->session, json_str(json_get(code_session_header(r->session), "cwd")), entry_id, at, &s, selected);
    if (!err) use_session(r, s);
    return err;
}

// AgentSessionRuntime.switchSession: the session file opened; the process moves to its folder, as
// pi's runtime does, unless it stays here (cwdOverride: the session's folder is gone).
static const char *host_resume(CodeUiHost *h, const char *path, bool stay) {
    Run *r = h->r;
    CodeSession *s;
    Arena *sa = session_arena_new();
    Err e = code_session_open_path(sa, path, &s);
    if (e) {
        const char *m = arena_printf(r->a, "Failed to resume session: %s", e);
        session_arena_drop(sa);
        return m;
    }
    const char *cwd = json_str(json_get(code_session_header(s), "cwd"));
    if (!stay && cwd && *cwd && strcmp(cwd, r->env.cwd)) {
        if (chdir(cwd) != 0) {
            const char *m = arena_printf(r->a, "Failed to resume session: %s", strerror(errno));
            session_arena_drop(sa);
            return m;
        }
        r->env.cwd = arena_strdup(r->a, cwd);
        // createRuntime in another folder: its own project trust, then its resources under it
        (void)code_trust_trusted(&r->env);
        code_resources_reload(&r->env);
    }
    use_session(r, s);
    session_arena_use(r, sa);
    return NULL;
}

// AgentSession.compact (manual), on the agent thread: compaction_start, then compaction_end with
// the result, the error or the abort, as pi emits them.
static void host_compact(CodeUiHost *h, const char *instructions) {
    Run *r = h->r;
    Arena *a = r->a;
    g_abort = 0;
    JValue *ev = ev_new(a, "compaction_start");
    json_put(a, ev, S("reason"), json_string(a, S("manual")));
    code_ui_event(ev);
    // settingsManager.getCompactionKeepRecentTokens(model): the merged settings, the model's override
    int64_t keep = code_compaction_settings(code_settings_merged(a, &r->env), r->model.provider, r->model.id).keep_recent_tokens;
    const JValue *entry = NULL;
    const char *err = code_sessions_compact(a, r->session, &r->model, &r->env, instructions && *instructions ? instructions : NULL, keep, &entry);
    bool aborted = g_abort || (err && !strcmp(err, "aborted"));
    JValue *end = ev_new(a, "compaction_end");
    json_put(a, end, S("reason"), json_string(a, S("manual")));
    if (!err && entry) {
        JValue *res = json_new(a, J_OBJECT);
        json_put(a, res, S("summary"), json_get(entry, "summary"));
        json_put(a, res, S("firstKeptEntryId"), json_get(entry, "firstKeptEntryId"));
        json_put(a, res, S("tokensBefore"), json_get(entry, "tokensBefore"));
        if (json_get(entry, "usage")) json_put(a, res, S("usage"), json_get(entry, "usage"));
        json_put(a, res, S("details"), json_get(entry, "details"));
        json_put(a, end, S("result"), res);
    }
    json_put(a, end, S("aborted"), json_bool(a, !entry && aborted));
    json_put(a, end, S("willRetry"), json_bool(a, false));
    if (err && !aborted) json_put(a, end, S("errorMessage"), json_string(a, str(arena_printf(a, "Compaction failed: %s", err))));
    code_ui_event(end);
}

static bool switch_model(Run *r, const char *provider, const char *id) {
    if (r->model.engine) return !strcmp(provider, r->model.provider) && !strcmp(id, r->model.id);
    Arena *a = r->a;
    CodeModel keep = r->model;
    Args o = {.provider = arena_strdup(a, provider), .model = arena_strdup(a, id)};
    JValue *settings = read_json(a, path_join(a, r->env.agent_dir, "settings.json"));
    if (find_model(r, &o, settings)) {
        r->model = keep;
        return false;
    }
    r->model.thinking = r->model.reasoning ? keep.thinking : "off";
    r->model.engine = NULL;
    r->env.provider = r->model.provider;
    r->env.model = r->model.id;
    r->env.thinking = r->model.thinking;
    code_monitor_set_model(NULL, r->model.provider, r->model.base_url);
    JValue *f = json_new(a, J_OBJECT);
    json_put(a, f, S("provider"), json_string(a, str(r->model.provider)));
    json_put(a, f, S("modelId"), json_string(a, str(r->model.id)));
    (void)code_session_append(r->session, "model_change", f);
    return true;
}

// What /model lists: the model packs this Mac can run in process (downloaded or not; the ones that
// need more memory than it has marked), then the openai-completions models of pi's models.json.
static bool pack_is_current(Arena *a, const Run *r, const DownloadPack *p) {
    if (!r->pack) return false;
    if (!strcmp(r->pack, p->name)) return true;
    char *dir;
    return !download_pack_dir(a, p, &dir) && !strcmp(dir, r->pack);
}

static CodeUiModel *list_models(Arena *a, const Run *r, size_t *n, CodeUiModel *current) {
    *n = 0;
    JValue *models = read_json(a, path_join(a, r->env.agent_dir, "models.json"));
    const JValue *providers = json_get(models, "providers");
    size_t cap = DOWNLOAD_NPACKS + 1;
    for (size_t i = 0; providers && json_is(providers, J_OBJECT) && i < providers->obj.len; i++) {
        const JValue *list = json_get(providers->obj.items[i].val, "models");
        cap += json_is(list, J_ARRAY) ? list->arr.len : 0;
    }
    CodeUiModel *out = arena_alloc(a, (cap + 1) * sizeof *out);
    const uint64_t ram = fit_ram_bytes();
    for (size_t i = 0; i < DOWNLOAD_NPACKS; i++) {
        const DownloadPack *p = &download_packs[i];
        if (!p->min_ram_gb) continue; // adapters are not models
        char *dir;
        bool have = !download_pack_dir(a, p, &dir) && file_exists(path_join(a, dir, "config.json"));
        CodeUiModel m = {.provider = "saetta", .id = p->name, .name = p->title ? p->title : p->name, .reasoning = true,
                         .context_window = 204800, .pack = true, .unavailable = !have,
                         .too_big = ram && ((uint64_t)p->min_ram_gb << 30) > ram, .disk_gb = p->disk_gb, .min_ram_gb = p->min_ram_gb};
        if (have) cfg_pack_label(a, m.name, p->gguf_file, dir, &m.label, &m.label_brief);
        if (pack_is_current(a, r, p)) {
            m.unavailable = false;
            m.context_window = r->model.context_window;
            *current = m;
        }
        out[(*n)++] = m;
    }
    if (r->pack && !current->id) { // a pack folder that is not a known download
        CodeUiModel m = {.provider = "saetta", .id = r->pack, .name = r->model.id, .reasoning = true,
                         .context_window = r->model.context_window, .pack = true};
        cfg_pack_label(a, NULL, NULL, r->pack, &m.label, &m.label_brief);
        *current = m;
        out[(*n)++] = m;
    }
    for (size_t i = 0; providers && json_is(providers, J_OBJECT) && i < providers->obj.len; i++) {
        const JMember *p = &providers->obj.items[i];
        const JValue *list = json_get(p->val, "models");
        for (size_t k = 0; json_is(list, J_ARRAY) && k < list->arr.len; k++) {
            const JValue *m = list->arr.items[k];
            const char *api = json_str(json_get(m, "api")) ? json_str(json_get(m, "api")) : json_str(json_get(p->val, "api"));
            if (!api || strcmp(api, "openai-completions") || !json_str(json_get(m, "id"))) continue;
            out[(*n)++] = (CodeUiModel){.provider = str_z(a, p->key), .id = json_str(json_get(m, "id")), .name = json_str(json_get(m, "name")),
                                        .reasoning = jbool(json_get(m, "reasoning"), false),
                                        .context_window = juint(json_get(m, "contextWindow"), 128000)};
        }
    }
    return out;
}

// The interactive mode chose a model this process cannot switch to in place (another pack, or
// away from the engine): saetta code starts again on it once the UI has ended, the session
// continued (-c), the other options kept.
static void host_restart(CodeUiHost *h, const char *provider, const char *id) {
    Run *r = h->r;
    free(r->restart_provider);
    free(r->restart_id);
    r->restart_provider = strdup(provider);
    r->restart_id = strdup(id);
}

static void restart_on(Run *r) {
    Arena *a = r->a;
    char self[PATH_MAX];
    uint32_t sz = sizeof self;
    if (_NSGetExecutablePath(self, &sz) != 0) return;
    VEC(const char *) v = {0};
    vec_push(&v, self);
    vec_push(&v, "code");
    for (size_t i = 0; i < r->argc; i++) {
        const char *x = r->argv[i];
        bool more = i + 1 < r->argc;
        bool takes = !strcmp(x, "--provider") || !strcmp(x, "--model") || !strcmp(x, "--pack") || !strcmp(x, "--thinking") ||
                     !strcmp(x, "--mode") || !strcmp(x, "--system-prompt") || !strcmp(x, "--append-system-prompt") ||
                     !strcmp(x, "--tools") || !strcmp(x, "--ctx") || !strcmp(x, "--session") || !strcmp(x, "--skill");
        if (takes && more) {
            if (strcmp(x, "--provider") && strcmp(x, "--model") && strcmp(x, "--pack") && strcmp(x, "--session")) {
                vec_push(&v, x);
                vec_push(&v, r->argv[i + 1]);
            }
            i++;
        } else if (x[0] == '-' && strcmp(x, "-c") && strcmp(x, "--continue")) vec_push(&v, x);
        // prompts are not run again
    }
    if (!strcmp(r->restart_provider, "saetta")) {
        vec_push(&v, "--pack");
        vec_push(&v, r->restart_id);
    } else {
        vec_push(&v, "--provider");
        vec_push(&v, r->restart_provider);
        vec_push(&v, "--model");
        vec_push(&v, r->restart_id);
    }
    // the current folder's project trust goes on, for that folder only (never --approve: the
    // next process resolves every other folder by its own decision, as pi's projectTrustByCwd)
    code_trust_export_restart(&r->env);
    // the conversation goes on in this session's own file (never "the latest in the folder"); one
    // not written yet (no conversation) starts fresh; --no-session stays as it was
    if (code_session_written(r->session)) {
        vec_push(&v, "--session");
        vec_push(&v, code_session_path(r->session));
    }
    vec_push(&v, NULL);
    code_tools_kill_detached();
    fflush(NULL);
    execv(self, (char *const *)v.ptr);
    int e = errno;
    code_trust_cancel_restart();
    errno = e;
    fprintf(stderr, "saetta code: cannot start again on %s/%s: %s\n", r->restart_provider, r->restart_id, strerror(errno));
    (void)a;
    vec_free(&v);
}

// /model's list after /model added, edited or removed a provider of models.json: built in an arena
// of its own (the UI thread calls it, maybe while the agent thread uses r->a). Two arenas take
// turns: the list before stays valid (the UI may still hold its strings until it takes the new
// one), the one before that is freed.
static const CodeUiModel *host_reload_models(CodeUiHost *h, size_t *n) {
    static Arena lists[2];
    static int shown = 1; // the arena of the list the UI holds now
    int next = shown ^ 1;
    arena_reset(&lists[next]);
    CodeUiModel cur = {0};
    const CodeUiModel *m = list_models(&lists[next], h->r, n, &cur);
    if (m) shown = next;
    return m;
}

// Each Esc counts (any nonzero value aborts): an automatic compaction gives back only the presses
// that stopped it (code_autocompact.c run_auto), so a later one still stops the run.
static void host_abort(CodeUiHost *h) {
    (void)h;
    __atomic_fetch_add(&g_abort, 1, __ATOMIC_SEQ_CST);
}

// ---- for /export, /share and /import (code_cmds.c) ---------------------------------------------------

static const char *host_session_file(CodeUiHost *h) { return code_session_path(h->r->session); }

// getSessionDir: the folder of the session file, else the default one for the cwd
// (agent_dir/sessions/--<cwd with / and : as ->--)
static const char *host_session_dir(CodeUiHost *h) {
    Run *r = h->r;
    const char *p = code_session_path(r->session);
    if (p && strrchr(p, '/')) return arena_strndup(r->a, p, (size_t)(strrchr(p, '/') - p));
    const char *c = r->env.cwd;
    while (*c == '/') c++;
    char *safe = arena_printf(r->a, "--%s--", c);
    for (char *q = safe + 2; *q; q++)
        if (*q == '/' || *q == '\\' || *q == ':') *q = '-';
    return path_join(r->a, path_join(r->a, r->env.agent_dir, "sessions"), safe);
}

static const char *host_session_cwd(CodeUiHost *h) { return h->r->env.cwd; }

// state.systemPrompt: the transcript's system prompt, else the one the next prompt starts with
// (pi's agent holds it from its creation)
static const char *host_system_prompt(CodeUiHost *h) {
    Run *r = h->r;
    const JValue *msgs = code_session_messages(r->session);
    bool have = false;
    for (size_t i = 0; msgs && i < msgs->arr.len && !have; i++)
        have = str_eqz(json_strs(json_get(msgs->arr.items[i], "role")), "system");
    JValue *sys = NULL;
    if (have) sys = current_system(r->a, msgs);
    else if (code_system_message(r->a, &r->env, r->tools, r->ntools, &sys)) return NULL;
    return str_z(r->a, code_system_text(r->a, sys));
}

// state.tools as the export lists them: name, description, parameters
static const JValue *host_tool_decls(CodeUiHost *h) {
    Run *r = h->r;
    Arena *a = r->a;
    JValue *schemas = code_tool_schemas(a, r->tools, r->ntools), *out = json_new(a, J_ARRAY);
    for (size_t i = 0; i < schemas->arr.len; i++) {
        const JValue *f = json_get(schemas->arr.items[i], "function");
        JValue *d = json_new(a, J_OBJECT);
        json_put(a, d, S("name"), (JValue *)json_get(f, "name"));
        json_put(a, d, S("description"), (JValue *)json_get(f, "description"));
        json_put(a, d, S("parameters"), (JValue *)json_get(f, "parameters"));
        json_push(a, out, d);
    }
    return out;
}

// The agent on another session file (AgentSessionRuntime createRuntime for a resume): its entries,
// id and file, run in cwd; the thinking level it recorded, when the model reasons.
static Err host_open_session(CodeUiHost *h, const char *path, const char *cwd) {
    Run *r = h->r;
    Arena *a = r->a;
    CodeSession *s;
    Arena *sa = session_arena_new();
    Err e = code_session_open_path(sa, path, &s);
    if (e) {
        e = arena_strdup(a, e);
        session_arena_drop(sa);
        return e;
    }
    r->session = s;
    session_arena_use(r, sa);
    bool moved = strcmp(cwd, r->env.cwd) != 0;
    r->env.cwd = arena_strdup(a, cwd);
    r->env.session_id = json_str(json_get(code_session_header(s), "id"));
    r->env.session_file = code_session_path(s);
    if (moved) {
        // createRuntime in another folder: its own project trust, then its resources under it
        (void)code_trust_trusted(&r->env);
        code_resources_reload(&r->env);
    }
    // the session's thinking level when it recorded one; else the current level, recorded (as the
    // start of a resumed session does)
    const char *lv = json_str(code_session_last(s, "thinking_level_change", "thinkingLevel"));
    if (lv && r->model.reasoning) {
        r->model.thinking = arena_strdup(a, lv);
        r->env.thinking = r->model.thinking;
    } else if (!lv) {
        JValue *f = json_new(a, J_OBJECT);
        json_put(a, f, S("thinkingLevel"), json_string(a, str(level_of(r))));
        (void)code_session_append(s, "thinking_level_change", f);
    }
    return OK;
}

// fd for the "@" completion: on PATH, else pi's managed copy in the agent folder
static const char *find_fd(Arena *a, const char *agent_dir) {
    const char *path = getenv("PATH");
    for (Str rest = str(path ? path : ""); rest.len;) {
        ptrdiff_t c = str_find_char(rest, ':', 0);
        Str dir = c >= 0 ? str_sub(rest, 0, (size_t)c) : rest;
        rest = c >= 0 ? str_sub(rest, (size_t)c + 1, rest.len) : S("");
        char *cand = path_join(a, str_z(a, dir), "fd");
        if (access(cand, X_OK) == 0) return cand;
    }
    char *managed = path_join(a, path_join(a, agent_dir, "bin"), "fd");
    return access(managed, X_OK) == 0 ? managed : NULL;
}

static const char *self_exe(Arena *a) {
    char self[PATH_MAX];
    uint32_t sz = sizeof self;
    return _NSGetExecutablePath(self, &sz) == 0 ? arena_strdup(a, self) : NULL;
}

static Err interactive(Run *r, const Args *o, const JValue *settings) {
    Arena *a = r->a;
    r->ui = true;
    tcgetattr(0, &g_cooked);
    g_tty_saved = 1;
    g_interactive = 1;
    r->env.abort = &g_abort;
    signal(SIGINT, SIG_IGN); // Ctrl-C is a key in raw mode
    const JValue *compaction = json_get(settings, "compaction");
    size_t nmodels;
    CodeUiModel cur = {0};
    CodeUiModel *models = list_models(a, r, &nmodels, &cur);
    CodeUiConfig cfg = {
        .cwd = r->env.cwd,
        .home = r->env.home,
        .agent_dir = r->env.agent_dir,
        .model = cur.id ? cur
                        : (CodeUiModel){.provider = r->model.provider, .id = r->model.id, .name = r->model.name, .reasoning = r->model.reasoning,
                                        .context_window = r->model.context_window},
        .thinking = r->model.thinking,
        .auto_compact = jbool(json_get(compaction, "enabled"), true),
        .fd_path = find_fd(a, r->env.agent_dir),
        .models = models,
        .nmodels = nmodels,
        .default_provider = json_str(json_get(settings, "defaultProvider")),
        .default_model = json_str(json_get(settings, "defaultModel")),
        .settings = settings,
        .resumed = (o->cont || o->session) && code_session_messages(r->session)->arr.len > 0,
        .version = code_version,
    };
    cfg.engine = r->model.engine != NULL;
    cfg.self_exe = self_exe(a);
    CodeUiOps ops = {host_prompt, host_new_session, host_set_thinking, host_entries, host_messages,
                     host_set_leaf, host_set_model, host_abort, host_restart, host_reload_models,
                     .session_file = host_session_file, .session_dir = host_session_dir, .session_cwd = host_session_cwd,
                     .system_prompt = host_system_prompt, .tool_decls = host_tool_decls, .open_session = host_open_session,
                     .session = host_session, .fork = host_fork, .resume = host_resume, .compact = host_compact};
    CodeUiHost host = {r};
    // a stand-in for a slow model load, for the loader's test scenario: SAETTA_CODE_LOAD_DEMO_MS
    const char *demo = getenv("SAETTA_CODE_LOAD_DEMO_MS");
    if (demo && atoi(demo) > 0) {
        Splash *sp = splash_start(arena_printf(a, "Loading %s ...", r->model.name ? r->model.name : r->model.id));
        struct timespec ts = {atoi(demo) / 1000, (long)(atoi(demo) % 1000) * 1000000L};
        nanosleep(&ts, NULL);
        splash_stop(sp);
    }
    code_monitor_set_model(r->model.engine, r->model.provider, r->model.base_url);
    Err e = code_ui_run(&cfg, &ops, &host, o->prompts.ptr, o->prompts.len);
    g_tty_saved = 0;
    if (e && !strcmp(e, "AgentBusy")) {
        // the run ignored the abort: leave without touching what it uses
        code_tools_kill_detached();
        _exit(0);
    }
    if (!e && r->restart_id) restart_on(r);
    return e;
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

// --pack: the engine in this process, described as `saetta start pi` describes the server to pi
// (launch.c pi_setup), so the requests stay byte for byte the same. The engine's own log lines
// go to ~/.saetta/code.log.
static Err open_engine(Arena *a, Run *r, const char *arg, const char *ctx, bool splash) {
    const DownloadPack *pk = download_find(arg);
    char *dir = (char *)arg;
    if (pk) TRY(download_pack_dir(a, pk, &dir));
    // a folder given by path (saetta start code passes one): the known pack in it, for its name
    for (size_t i = 0; !pk && i < DOWNLOAD_NPACKS; i++) {
        char *d;
        if (!download_pack_dir(a, &download_packs[i], &d) && !strcmp(d, dir)) pk = &download_packs[i];
    }
    if (!file_exists(path_join(a, dir, "config.json"))) {
        dbg("saetta code: no model at %s%s%s\n", dir, pk ? "; download it with: saetta download " : "", pk ? arg : "");
        return "NoModel";
    }
    char *logdir = path_join(a, r->env.home, ".saetta");
    mkdir_p(logdir);
    FILE *log = fopen(path_join(a, logdir, "code.log"), "ae");
    if (log) {
        setvbuf(log, NULL, _IOLBF, 0);
        dbg_out = log;
    }
    // the context saetta start gives a server (204800 tokens), unless --ctx or SAETTA_CTX says
    // otherwise; the engine lowers it to what this Mac's memory holds
    if (ctx) setenv("SAETTA_CTX", ctx, 1);
    else if (!getenv("SAETTA_CTX")) setenv("SAETTA_CTX", "204800", 1);
    uint64_t t0 = now_ns();
    // the loader while the engine loads (interactive runs only); the load itself runs here at full speed
    Splash *sp = splash ? splash_start(arena_printf(a, "Loading %s ...", pk && pk->title ? pk->title : (strrchr(arg, '/') && strrchr(arg, '/')[1] ? strrchr(arg, '/') + 1 : arg))) : NULL;
    if (!sp && isatty(2)) fprintf(stderr, "saetta code: loading %s ...\r", pk ? arg : dir);
    Engine *e;
    Err err = server_engine_open(a, dir, NULL, &e);
    splash_stop(sp);
    if (!sp && isatty(2)) fprintf(stderr, "\x1b[K");
    if (err && !strcmp(err, "EngineBusy")) {
        Server srv;
        int pid = 0;
        if (instance_running(&srv))
            fprintf(stderr, "saetta code: a saetta server is running (pid %d, port %u); one model runs at a time.\n"
                            "Use it: saetta start code (or saetta code without --pack), or stop it first (saetta stop).\n",
                    srv.pid, srv.port);
        else if (instance_inproc_running(&pid))
            fprintf(stderr, "saetta code: another Saetta Code (pid %d) is running the model; one model runs at a time.\n", pid);
        return err;
    }
    if (err) {
        fprintf(stderr, "saetta code: cannot load %s: %s (log: ~/.saetta/code.log)\n", dir, err);
        return err;
    }
    dbg("saetta code: %s loaded in process in %llu ms\n", dir, (unsigned long long)((now_ns() - t0) / 1000000));
    CodeModel *m = &r->model;
    m->engine = e;
    m->provider = "saetta";
    m->id = server_engine_model(e);
    m->name = m->id;
    m->base_url = "in-process";
    m->api_key = "saetta";
    m->reasoning = true;
    m->context_window = server_engine_context(e);
    m->max_tokens = 32768;
    m->supports_developer_role = m->supports_store = m->supports_reasoning_effort = false;
    m->max_tokens_field = "max_tokens";
    m->thinking_format = "qwen-chat-template";
    JValue *compat = json_new(a, J_OBJECT);
    json_put(a, compat, S("supportsDeveloperRole"), json_bool(a, false));
    json_put(a, compat, S("supportsStore"), json_bool(a, false));
    json_put(a, compat, S("supportsReasoningEffort"), json_bool(a, false));
    json_put(a, compat, S("maxTokensField"), json_string(a, S("max_tokens")));
    json_put(a, compat, S("thinkingFormat"), json_string(a, S("qwen-chat-template")));
    m->compat = compat;
    return OK;
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

// ---- the first run (first-time-setup.ts, Saetta's way: a model pack on this Mac, or an
// OpenAI-compatible server in pi's models.json) ----------------------------------------------------

static bool ask_yes(const char *q, bool dflt) {
    printf("%s %s ", q, dflt ? "[Y/n]" : "[y/N]");
    fflush(stdout);
    char line[64];
    if (!fgets(line, sizeof line, stdin)) return false;
    Str t = str_trim(str(line));
    if (!t.len) return dflt;
    return t.ptr[0] == 'y' || t.ptr[0] == 'Y';
}

static bool pack_here(Arena *a, const DownloadPack *p) {
    char *dir;
    return !download_pack_dir(a, p, &dir) && file_exists(path_join(a, dir, "config.json"));
}

// The best downloaded pack that fits this Mac (saetta start's choice), or NULL.
static const DownloadPack *downloaded_pack(Arena *a) {
    uint64_t ram = fit_ram_bytes();
    for (size_t i = 0; i < DOWNLOAD_NPACKS; i++) {
        const DownloadPack *p = &download_packs[i];
        if (!p->min_ram_gb || (ram && ((uint64_t)p->min_ram_gb << 30) > ram)) continue;
        if (pack_here(a, p)) return p;
    }
    return NULL;
}

// Whether the server of base_url accepts a connection (1 s).
static bool server_answers(const char *base_url) {
    if (!base_url) return false;
    const char *h = strstr(base_url, "://");
    h = h ? h + 3 : base_url;
    char host[256] = {0};
    size_t n = strcspn(h, ":/");
    snprintf(host, sizeof host, "%.*s", (int)MIN(n, sizeof host - 1), h);
    int port = strncmp(base_url, "https", 5) ? 80 : 443;
    if (h[n] == ':') port = atoi(h + n + 1);
    char ps[16];
    snprintf(ps, sizeof ps, "%d", port);
    struct addrinfo hints = {.ai_socktype = SOCK_STREAM}, *res = NULL;
    if (getaddrinfo(host, ps, &hints, &res) || !res) return false;
    bool ok = false;
    for (struct addrinfo *ai = res; ai && !ok; ai = ai->ai_next) {
        int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        fcntl(fd, F_SETFL, O_NONBLOCK);
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) ok = true;
        else if (errno == EINPROGRESS) {
            struct pollfd pf = {fd, POLLOUT, 0};
            int err = 0;
            socklen_t el = sizeof err;
            if (poll(&pf, 1, 1000) == 1 && !getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &el) && !err) ok = true;
        }
        close(fd);
    }
    freeaddrinfo(res);
    return ok;
}

// No model to talk to: guide the user to one (in a terminal only). Sets *pack to a download name
// to run in process, or returns false to stop (the reason printed).
// The nearest existing folder's free space (the models folder may not exist yet).
static uint64_t free_bytes(const char *dir) {
    char probe[PATH_MAX];
    snprintf(probe, sizeof probe, "%s", dir);
    struct statfs st;
    while (statfs(probe, &st) != 0) {
        char *sl = strrchr(probe, '/');
        if (!sl || sl == probe) return 0;
        *sl = 0;
    }
    return (uint64_t)st.f_bavail * st.f_bsize;
}

static bool first_run(Arena *a, const char *agent_dir, bool server_down, const char *base_url, const char **pack) {
    *pack = NULL;
    // the models.json this run reads, ~ for the home folder
    const char *home = getenv("HOME");
    const char *mj = path_join(a, agent_dir, "models.json");
    if (home && *home && !strncmp(mj, home, strlen(home)) && mj[strlen(home)] == '/') mj = arena_printf(a, "~%s", mj + strlen(home));
    const DownloadPack *have = downloaded_pack(a);
    printf("\x1b[1m\xe2\x86\xaf Saetta Code\x1b[0m\n\n");
    if (server_down) printf("The server at %s does not answer.\n", base_url);
    else printf("No model is set up yet. Saetta Code runs a model on this Mac, or talks to an OpenAI-compatible\nserver listed in %s.\n", mj);
    if (have) {
        printf("\n%s is downloaded on this Mac.\n", have->title ? have->title : have->name);
        if (ask_yes(arena_printf(a, "Run %s here?", have->title ? have->title : have->name), true)) {
            *pack = have->name;
            return true;
        }
        return false;
    }
    const char *rec = fit_default_pack();
    const DownloadPack *p = rec ? download_find(rec) : NULL;
    if (!p) {
        printf("\nThis Mac has %llu GB of memory: no saetta model fits it (they need 32 GB or more).\n",
               (unsigned long long)(fit_ram_bytes() >> 30));
        printf("Add an OpenAI-compatible server to %s to use Saetta Code here.\n", mj);
        return false;
    }
    char *mdir = NULL;
    uint64_t freeb = !download_models_dir(a, &mdir) ? free_bytes(mdir) : 0;
    if (getenv("SAETTA_CODE_FREE_GB")) freeb = (uint64_t)atoll(getenv("SAETTA_CODE_FREE_GB")) * 1000000000ull; // tests
    printf("\nFor this Mac (%llu GB of memory): %s, about %u GB to download (%llu GB free on that disk).\n",
           (unsigned long long)(fit_ram_bytes() >> 30), p->title ? p->title : p->name, p->disk_gb, (unsigned long long)(freeb / 1000000000ull));
    if (freeb && freeb < (uint64_t)p->disk_gb * 1000000000ull) {
        printf("Not enough free disk for it. Free some space, or add a server to %s.\n", mj);
        return false;
    }
    if (!ask_yes("Download it now (resumable)?", true)) {
        printf("Later: saetta download %s, then saetta code.\n", p->name);
        return false;
    }
    const char *test = getenv("SAETTA_CODE_DOWNLOAD"); // a stand-in for the tests
    Err de;
    if (test && *test) {
        char *cmd = arena_printf(a, "'%s' %s", test, p->name);
        de = system(cmd) == 0 ? OK : "DownloadFailed";
    } else {
        const char *dargs[] = {p->name};
        de = download_run(dargs, 1);
    }
    if (de) {
        printf("The download stopped (%s); run saetta download %s to resume it.\n", de, p->name);
        return false;
    }
    *pack = p->name;
    return true;
}

Err code_run(const char *const *args, size_t n, int *code) {
    *code = 1;
    Arena arena = {0};
    Arena *a = &arena;
    Args o = {.approve = -1};
    Err e = parse_args(args, n, &o);
    if (e) goto done;
    Run r = {.a = a, .json = o.mode && !strcmp(o.mode, "json"), .pack = o.pack, .argv = args, .argc = n};
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
    {
        struct sigaction st = {0};
        st.sa_handler = on_term;
        sigaction(SIGTERM, &st, NULL);
        sigaction(SIGHUP, &st, NULL);
    }
    r.env.pi_docs = find_pi_docs(a);
    r.env.system_prompt = o.system_prompt;
    r.env.append_system_prompt = o.append.ptr;
    r.env.nappend_system_prompt = o.append.len;
    pick_tools(a, &o, &r);
    r.env.on_partial = on_tool_partial;
    r.env.partial_ctx = &r;
    JValue *settings = read_json(a, path_join(a, r.env.agent_dir, "settings.json"));
    const JValue *rs = json_get(settings, "retry");
    g_retry.enabled = jbool(json_get(rs, "enabled"), true);
    g_retry.max_retries = (int64_t)juint(json_get(rs, "maxRetries"), 3);
    g_retry.base_ms = (int64_t)juint(json_get(rs, "baseDelayMs"), 2000);
    g_retry.cap_ms = (int64_t)juint(json_get(rs, "maxAgentDelayMs"), 60000);
    // piped stdin (readPipedStdin, buildInitialMessage): trimmed, joined to the first prompt with
    // nothing between, and it makes the run a print run
    bool piped = false;
    if (!isatty(0)) {
        Buf in = {0};
        char chunk[65536];
        ssize_t got;
        while ((got = read(0, chunk, sizeof chunk)) > 0) buf_add(&in, chunk, (size_t)got);
        Str t = str_trim(buf_str(&in));
        if (t.len) {
            piped = true;
            Str first = o.prompts.len ? str(o.prompts.ptr[0]) : S("");
            char *joined = arena_printf(a, "%.*s%.*s", (int)t.len, t.ptr, (int)first.len, first.ptr);
            if (o.prompts.len) o.prompts.ptr[0] = joined;
            else vec_push(&o.prompts, joined);
        }
        buf_free(&in);
    }
    bool tty = !o.print && !piped && !r.json && isatty(0) && isatty(1);
    if (!o.pack) {
        e = find_model(&r, &o, settings);
        // the first run: nothing configured, or a server that does not answer
        bool down = !e && tty && r.model.base_url && !server_answers(r.model.base_url);
        if (tty && (e || down)) {
            const char *p;
            if (first_run(a, r.env.agent_dir, down, r.model.base_url, &p)) {
                o.pack = p;
                r.pack = p;
                r.model = (CodeModel){0};
                e = OK;
            } else if (e) {
                *code = 0;
                e = OK;
                goto done;
            } else e = OK; // the server stays (pi's behavior: the request reports the error)
        }
    }
    if (o.pack) {
        e = open_engine(a, &r, o.pack, o.ctx, tty);
        if (e) goto done;
        if (o.thinking) r.model.thinking = o.thinking;
    }
    if (e && !o.pack) {
        dbg("saetta code: no model %s%s%s in %s/models.json (Saetta Code runs openai-completions providers)\n",
            o.provider ? o.provider : "", o.provider ? "/" : "", o.model ? o.model : "", r.env.agent_dir);
        goto done;
    }
    e = o.session ? code_session_open_path(a, o.session, &r.session) : code_session_open(a, &r.env, o.cont, o.no_session, &r.session);
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
    r.env.session_id = json_str(json_get(code_session_header(r.session), "id"));
    r.env.session_file = code_session_path(r.session);
    r.env.provider = r.model.provider;
    r.env.model = r.model.id;
    r.env.thinking = level;
    // project trust (the startup prompt in the interactive mode), then the skills and context files
    code_skills_options(o.no_skills, o.skills.ptr, o.skills.len);
    code_trust_override(o.approve);
    code_trust_resolve(&r.env, tty);
    (void)code_resources(&r.env);
    r.base_tools = r.tools;
    r.nbase_tools = r.ntools;
    code_mcp_start(&r.env, !o.no_mcp, r.tools, r.ntools, o.tools || o.no_tools);
    if (!o.print && !piped && !r.json && isatty(0) && isatty(1)) {
        e = interactive(&r, &o, settings);
        if (!e) *code = 0;
        goto done;
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
    code_mcp_shutdown();
    code_tools_kill_detached();
    vec_free(&o.prompts);
    vec_free(&o.append);
    vec_free(&o.skills);
    arena_reset(a);
    return e;
}

// getCurrentSystemMessage for code_sessions.c (compaction's system message and estimates).
JValue *code_current_system(Arena *a, const JValue *messages) { return current_system(a, messages); }
