// `saetta code` (see code.h): the command line, print and json modes, and the agent loop.
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi v1.5.2:
// packages/agent/src/agent-loop.ts, packages/coding-agent/src/main.ts, cli/args.ts,
// core/agent-session.ts, modes/print-mode.ts, modes/json-event.ts, packages/ai/src/utils/transcript.ts).
#include "code.h"
#include "code_oai.h"

#include <ctype.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <termios.h>
#include <time.h>
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
    bool ui; // the interactive mode prints the stream as it arrives
    int64_t retry_attempt; // the retries of the current prompt so far (pi's _retryAttempt)
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
        else if (!strcmp(x, "--help") || !strcmp(x, "-h")) {
            printf("usage: saetta code [options] [prompt ...]\n"
                   "\n"
                   "Saetta Code, a minimal coding agent that behaves as pi 1.5.2 does: the same tools (read, bash,\n"
                   "edit, write; grep, find, ls with --tools), system prompt, AGENTS.md / CLAUDE.md files, skills and\n"
                   "session files, so pi and Saetta Code resume each other's sessions. It talks to saetta or any\n"
                   "OpenAI-compatible chat completions server named in pi's models.json ($PI_CODING_AGENT_DIR, else\n"
                   "~/.pi/agent). `saetta start code` starts the server and Saetta Code together.\n"
                   "\n"
                   "  -p, --print              answer the prompt and exit (text output)\n"
                   "  --mode json              print pi's JSON event stream\n"
                   "  -c, --continue           continue the most recent session in this folder\n"
                   "  --no-session             do not save the session\n"
                   "  --provider P --model M   the model (M may be provider/id and end in :thinking)\n"
                   "  --thinking LEVEL         off, minimal, low, medium, high, xhigh, max\n"
                   "  --tools a,b,c | --no-tools   the tool set\n"
                   "  --system-prompt T, --append-system-prompt T   text or a file\n"
                   "\n"
                   "Without -p in a terminal it opens a prompt: Enter sends, a trailing \\ continues on a new line,\n"
                   "Ctrl-C stops a turn, /new starts a session, /exit quits.\n");
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

// ---- the interactive display (a minimal stand-in for pi's TUI) ------------------------------------

static volatile int g_abort;
static const char *g_cwd;
static int g_ui_state; // 0 nothing, 1 thinking, 2 text

#define DIM "\x1b[2m"
#define BOLD "\x1b[1m"
#define CYAN "\x1b[36m"
#define RED "\x1b[31m"
#define RESET "\x1b[0m"

// Text from the model or a tool, shown without terminal control: escape sequences (CSI and the
// rest) and other C0 controls but newline and tab are dropped.
static void ui_write(Str t) {
    size_t i = 0, run = 0;
    while (i < t.len) {
        unsigned char c = (unsigned char)t.ptr[i];
        if (c >= 0x20 && c != 0x7f) {
            i++;
            continue;
        }
        if (c == '\n' || c == '\t') {
            i++;
            continue;
        }
        fwrite(t.ptr + run, 1, i - run, stdout);
        if (c == 0x1b && i + 1 < t.len && t.ptr[i + 1] == '[') { // CSI: to its final byte
            i += 2;
            while (i < t.len && !((unsigned char)t.ptr[i] >= 0x40 && (unsigned char)t.ptr[i] <= 0x7e)) i++;
            if (i < t.len) i++;
        } else if (c == 0x1b && i + 1 < t.len && t.ptr[i + 1] == ']') { // OSC: to BEL or ST
            i += 2;
            while (i < t.len && t.ptr[i] != 0x07 && !(t.ptr[i] == 0x1b && i + 1 < t.len && t.ptr[i + 1] == '\\')) i++;
            i += i < t.len && t.ptr[i] == 0x07 ? 1 : 2;
            if (i > t.len) i = t.len;
        } else i++;
        run = i;
    }
    fwrite(t.ptr + run, 1, i - run, stdout);
}

static void ui_event(Str type, const JValue *event) {
    Str delta = json_strs(json_get(event, "delta"));
    if (str_eqz(type, "thinking_start")) {
        printf("%s%s", g_ui_state ? "\n" : "", DIM);
        g_ui_state = 1;
    } else if (str_eqz(type, "thinking_delta")) {
        ui_write(delta);
    } else if (str_eqz(type, "thinking_end")) {
        printf("%s\n", RESET);
        g_ui_state = 0;
    } else if (str_eqz(type, "text_start")) {
        if (g_ui_state == 1) printf("%s\n", RESET);
        g_ui_state = 2;
    } else if (str_eqz(type, "text_delta")) {
        ui_write(delta);
    } else if (str_eqz(type, "text_end")) {
        printf("\n");
        g_ui_state = 0;
    }
    fflush(stdout);
}

// "read calc.py", "bash ls -la", "edit stats.py": the call's main argument, one line
static void ui_tool_start(const JValue *call) {
    const JValue *args = json_get(call, "arguments");
    Str name = json_strs(json_get(call, "name"));
    Str arg = json_strs(json_get(args, "path"));
    if (!arg.len) arg = json_strs(json_get(args, "command"));
    if (!arg.len) arg = json_strs(json_get(args, "pattern"));
    // paths under the working directory are shown relative to it
    Str cwd = str(g_cwd);
    if (cwd.len && arg.len > cwd.len + 1 && str_starts(arg, cwd) && arg.ptr[cwd.len] == '/') arg = str_sub(arg, cwd.len + 1, arg.len);
    size_t nl = 0;
    while (nl < arg.len && arg.ptr[nl] != '\n') nl++;
    if (g_ui_state) printf("%s\n", RESET);
    g_ui_state = 0;
    printf("%s%s\xe2\x97\x8f ", CYAN, BOLD);
    ui_write(name);
    printf("%s ", RESET);
    ui_write(str_n(arg.ptr, nl > 120 ? 120 : nl));
    printf("%s%s\n", nl < arg.len || nl > 120 ? " ..." : "", RESET);
    fflush(stdout);
}

// the first lines of the result, dimmed (red when it failed)
static void ui_tool_end(const JValue *msg) {
    bool err = json_is(json_get(msg, "isError"), J_BOOL) && json_get(msg, "isError")->b;
    const JValue *c = json_get(msg, "content");
    Str t = c && c->arr.len ? json_strs(json_get(c->arr.items[0], "text")) : S("");
    size_t lines = 0, i = 0;
    printf("%s", err ? RED : DIM);
    while (i < t.len && lines < 6) {
        size_t j = i;
        while (j < t.len && t.ptr[j] != '\n') j++;
        printf("  ");
        ui_write(str_n(t.ptr + i, j - i > 200 ? 200 : j - i));
        printf("\n");
        lines++;
        i = j + 1;
    }
    if (i < t.len) {
        size_t more = 0;
        for (size_t k = i; k < t.len; k++) more += t.ptr[k] == '\n';
        printf("  ... %zu more lines\n", more + 1);
    }
    printf("%s", RESET);
    fflush(stdout);
}

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
    if (r->ui) ui_event(type, event);
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
        if (r->ui) ui_tool_end(msg);
        TRY(add_message(r, msg));
        json_push(a, results, msg);
    }
    return OK;
}

// The tools the transcript declares (pi runs only the tools in the context: a call to any
// other is "Tool <name> not found", whatever the tools module knows).
static bool declared(Arena *a, const JValue *system, Str name) {
    const JValue *decl = system ? json_get(system, "toolsAdded") : NULL;
    for (size_t i = 0; decl && json_is(decl, J_ARRAY) && i < decl->arr.len; i++)
        if (str_eq(json_strs(json_get(decl->arr.items[i], "name")), name)) return true;
    (void)a;
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
    const JValue *system = current_system(a, code_session_messages(r->session));
    Err err = OK;
    for (size_t i = 0; i < calls.len && !err; i++) {
        emit_tool_start(r, calls.ptr[i]);
        if (r->ui) ui_tool_start(calls.ptr[i]);
        r->env.now_ms = wall_ms();
        Str name = json_strs(json_get(calls.ptr[i], "name"));
        if (!declared(a, system, name)) {
            error_result(a, calls.ptr[i], arena_printf(a, "Tool %.*s not found", (int)name.len, name.ptr), &msgs[i], &raws[i]);
            emit_tool_end(r, calls.ptr[i], msgs[i], raws[i]);
            continue;
        }
        prepare_edit_in_place(a, calls.ptr[i]);
        err = code_tool_run(a, &r->env, calls.ptr[i], &msgs[i], &raws[i]);
        if (!err && !json_get(msgs[i], "durationMs")) emit_tool_end(r, calls.ptr[i], msgs[i], raws[i]);
    }
    // pi runs the executed calls concurrently and ends each as it settles; a call that throws
    // (a missing file, say) settles before one that completes its I/O, so errors end first
    for (int pass = 0; pass < 2; pass++)
        for (size_t i = 0; i < calls.len && !err; i++) {
            if (!json_get(msgs[i], "durationMs")) continue;
            bool failed = jbool(json_get(msgs[i], "isError"), false);
            if (failed == (pass == 0)) emit_tool_end(r, calls.ptr[i], msgs[i], raws[i]);
        }
    for (size_t i = 0; i < calls.len && !err; i++) {
        if (r->ui) ui_tool_end(msgs[i]);
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

static bool retryable(const JValue *msg) {
    if (!str_eqz(json_strs(json_get(msg, "stopReason")), "error")) return false;
    const char *m = json_str(json_get(msg, "errorMessage"));
    if (!m || !*m) return false;
    static const char *const never[] = {"GoUsageLimitError", "FreeUsageLimitError", "Monthly usage limit reached", "available balance",
                                        "insufficient_quota", "out of budget", "quota exceeded", "billing",
                                        "subscription_sharing_usage_limit_exceeded",
                                        "subscription_sharing_workplace_individual_usage_limit_exceeded"};
    for (size_t k = 0; k < COUNT(never); k++)
        if (pm(m, never[k])) return false;
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
static void finish_agent(Run *r, bool will_retry, const JValue *last);
static Err run_turns(Run *r, JValue **last_out, const char **last_id);

static void sleep_abortable(int64_t ms) {
    for (int64_t t = 0; t < ms && !g_abort; t += 50) {
        struct timespec ts = {0, 50 * 1000000L};
        nanosleep(&ts, NULL);
    }
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
    json_put(a, block, S("text"), json_string(a, code_utf8_clean(a, str(text))));
    json_push(a, content, block);
    json_put(a, user, S("content"), content);
    json_put(a, user, S("timestamp"), json_int(a, (int64_t)wall_ms()));
    TRY(add_message(r, user));
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
        }
        started = false;
        JValue *last = NULL;
        const char *last_id = NULL;
        TRY(run_turns(r, &last, &last_id));
        bool again = false;
        if (last && retryable(last) && g_retry.enabled && !g_abort && r->retry_attempt < g_retry.max_retries) {
            int64_t attempt = ++r->retry_attempt;
            int64_t delay = g_retry.base_ms << (attempt - 1);
            if (delay > g_retry.cap_ms) delay = g_retry.cap_ms;
            JValue *e = ev_new(a, "auto_retry_start");
            json_put(a, e, S("attempt"), json_int(a, attempt));
            json_put(a, e, S("maxAttempts"), json_int(a, g_retry.max_retries));
            json_put(a, e, S("delayMs"), json_int(a, delay));
            const char *em = json_str(json_get(last, "errorMessage"));
            json_put(a, e, S("errorMessage"), json_string(a, str(em ? em : "Unknown error")));
            if (r->ui) printf("%sretrying in %lld s (%lld of %lld)%s\n", DIM, (long long)(delay / 1000), (long long)attempt, (long long)g_retry.max_retries, RESET);
            // agent_end (willRetry) is emitted before the retry starts
            finish_agent(r, true, NULL);
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
                return OK;
            }
            continue;
        }
        finish_agent(r, false, last);
        (void)again;
        return OK;
    }
}

static void finish_agent(Run *r, bool will_retry, const JValue *last) {
    Arena *a = r->a;
    JValue *ae = ev_new(a, "agent_end");
    json_put(a, ae, S("messages"), r->new_messages);
    json_put(a, ae, S("willRetry"), json_bool(a, will_retry));
    emit(r, ae);
    if (will_retry) return;
    // retries used up on an error (_handlePostAgentRun)
    if (r->retry_attempt > 0 && last && str_eqz(json_strs(json_get(last, "stopReason")), "error")) {
        JValue *end = ev_new(a, "auto_retry_end");
        json_put(a, end, S("success"), json_bool(a, false));
        json_put(a, end, S("attempt"), json_int(a, r->retry_attempt));
        json_put(a, end, S("finalError"), (JValue *)json_get(last, "errorMessage"));
        emit(r, end);
        r->retry_attempt = 0;
    }
    JValue *settled = ev_new(a, "agent_settled");
    json_put(a, settled, S("aborted"), json_bool(a, g_abort != 0));
    emit(r, settled);
}

// The turns of one agent run (runLoop): *last the final assistant message and its entry id.
static Err run_turns(Run *r, JValue **last_out, const char **last_id) {
    Arena *a = r->a;
    for (bool first = true;; first = false) {
        if (!first) emit(r, ev_new(a, "turn_start"));
        JValue *msg;
        TRY(stream_assistant(r, &msg));
        *last_out = msg;
        *last_id = code_session_leaf(r->session);
        Str stop = json_strs(json_get(msg, "stopReason"));
        if (r->ui && (str_eqz(stop, "error") || str_eqz(stop, "aborted"))) {
            printf("%s%s%s%s\n", g_ui_state ? "\n" : "", RED, json_str(json_get(msg, "errorMessage")) ? json_str(json_get(msg, "errorMessage")) : "error", RESET);
            g_ui_state = 0;
        }
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
        if (!more || g_abort) break;
    }
    return OK;
}

// ---- interactive mode: a prompt loop with a small line editor -------------------------------------

static void on_sigint(int sig) {
    (void)sig;
    g_abort = 1;
}

static struct termios g_cooked;

static void raw_on(void) {
    struct termios t = g_cooked;
    t.c_lflag &= ~(tcflag_t)(ICANON | ECHO | ISIG);
    t.c_iflag &= ~(tcflag_t)(IXON | ICRNL);
    t.c_cc[VMIN] = 1;
    t.c_cc[VTIME] = 0;
    tcsetattr(0, TCSANOW, &t); // keep what was typed or pasted ahead
    fputs("\x1b[?2004h", stdout); // bracketed paste: a pasted newline does not send
    fflush(stdout);
}

static void raw_off(void) {
    fputs("\x1b[?2004l", stdout);
    fflush(stdout);
    tcsetattr(0, TCSANOW, &g_cooked);
}

// the tools module's (code_tools.c) replaces this when it tracks background jobs
__attribute__((weak)) void code_tools_kill_detached(void) {}

static volatile sig_atomic_t g_tty_saved;

// SIGTERM / SIGHUP: the terminal back as it was, the bash jobs stopped, exit as pi does (143, 129)
static void on_term(int sig) {
    if (g_tty_saved) {
        static const char off[] = "\x1b[?2004l\x1b[0m\r\n";
        (void)!write(1, off, sizeof off - 1);
        tcsetattr(0, TCSANOW, &g_cooked);
    }
    code_tools_kill_detached();
    _exit(sig == SIGHUP ? 129 : 143);
}

// Code points in s (the cursor moves by those; wide characters are not measured).
static size_t cps(const char *s, size_t n) {
    size_t c = 0;
    for (size_t i = 0; i < n; i++) c += ((unsigned char)s[i] & 0xC0) != 0x80;
    return c;
}

static void redraw(const char *prompt, const Buf *line, size_t cur) {
    printf("\r\x1b[K%s%.*s", prompt, (int)line->len, line->ptr ? line->ptr : "");
    size_t back = cps(line->ptr + cur, line->len - cur);
    if (back) printf("\x1b[%zuD", back);
    fflush(stdout);
}

// One prompt from the user: Enter sends, a trailing backslash continues on a new line, arrows
// move and walk the history, Ctrl-A / Ctrl-E jump, Ctrl-U clears, Ctrl-C clears (or quits on an
// empty line), Ctrl-D quits on an empty line. Returns false to quit.
static bool read_prompt(Arena *a, VecStr *history, char **out) {
    Buf text = {0}, line = {0};
    size_t cur = 0, hist = history->len;
    bool pasting = false;
    const char *prompt = BOLD "> " RESET;
    raw_on();
    redraw(prompt, &line, cur);
    bool ok = true;
    for (;;) {
        unsigned char c;
        if (read(0, &c, 1) != 1) {
            ok = false;
            break;
        }
        if ((c == '\r' || c == '\n') && pasting) { // a pasted line break: keep going
            buf_addstr(&text, buf_str(&line));
            buf_addc(&text, '\n');
            line.len = cur = 0;
            printf("\r\n");
            prompt = "  ";
            redraw(prompt, &line, cur);
            continue;
        }
        if (c == '\r' || c == '\n') {
            if (line.len && line.ptr[line.len - 1] == '\\') { // continue
                line.len--;
                buf_addstr(&text, buf_str(&line));
                buf_addc(&text, '\n');
                line.len = cur = 0;
                printf("\r\n");
                prompt = "  ";
                redraw(prompt, &line, cur);
                continue;
            }
            buf_addstr(&text, buf_str(&line));
            printf("\r\n");
            break;
        }
        if (c == 3) { // Ctrl-C
            if (!line.len && !text.len) {
                ok = false;
                printf("\r\n");
                break;
            }
            line.len = cur = text.len = 0;
            prompt = BOLD "> " RESET;
        } else if (c == 4) { // Ctrl-D
            if (!line.len && !text.len) {
                ok = false;
                printf("\r\n");
                break;
            }
        } else if (c == 127 || c == 8) {
            if (cur > 0) {
                size_t k = cur - 1;
                while (k > 0 && ((unsigned char)line.ptr[k] & 0xC0) == 0x80) k--;
                memmove(line.ptr + k, line.ptr + cur, line.len - cur);
                line.len -= cur - k;
                cur = k;
            }
        } else if (c == 1) cur = 0;
        else if (c == 5) cur = line.len;
        else if (c == 21) line.len = cur = 0;
        else if (c == 27) {
            // an escape sequence, read to its final byte (CSI parameters, then 0x40..0x7e)
            unsigned char seq[2], params[16];
            size_t np = 0;
            if (read(0, &seq[0], 1) != 1) continue;
            if (seq[0] != '[' && seq[0] != 'O') continue; // Alt+key and the like: ignored
            for (;;) {
                unsigned char x;
                if (read(0, &x, 1) != 1) break;
                if (x >= 0x40 && x <= 0x7e) {
                    seq[1] = x;
                    break;
                }
                if (np < sizeof params) params[np++] = x;
            }
            if (seq[1] == '~' && np == 3 && !memcmp(params, "200", 3)) pasting = true;
            else if (seq[1] == '~' && np == 3 && !memcmp(params, "201", 3)) pasting = false;
            else if (np) {
                // other parameterized keys (Home, End, Delete, modified arrows): ignored
            } else if (seq[1] == 'D' && cur > 0) {
                cur--;
                while (cur > 0 && ((unsigned char)line.ptr[cur] & 0xC0) == 0x80) cur--;
            } else if (seq[1] == 'C' && cur < line.len) {
                cur++;
                while (cur < line.len && ((unsigned char)line.ptr[cur] & 0xC0) == 0x80) cur++;
            } else if ((seq[1] == 'A' && hist > 0) || (seq[1] == 'B' && hist < history->len)) {
                hist += seq[1] == 'A' ? (size_t)-1 : 1;
                line.len = 0;
                if (hist < history->len) buf_addstr(&line, history->ptr[hist]);
                cur = line.len;
            }
        } else if (c >= 32 || c == '\t') {
            buf_reserve(&line, 1);
            memmove(line.ptr + cur + 1, line.ptr + cur, line.len - cur);
            line.ptr[cur++] = (char)c;
            line.len++;
        }
        redraw(prompt, &line, cur);
    }
    raw_off();
    *out = ok ? str_z(a, buf_str(&text)) : NULL;
    if (ok && text.len) vec_push(history, str(*out));
    buf_free(&text);
    buf_free(&line);
    return ok;
}

static Err interactive(Run *r, const Args *o) {
    Arena *a = r->a;
    r->ui = true;
    g_cwd = r->env.cwd;
    tcgetattr(0, &g_cooked);
    g_tty_saved = 1;
    struct sigaction st = {0};
    st.sa_handler = on_term;
    sigaction(SIGTERM, &st, NULL);
    sigaction(SIGHUP, &st, NULL);
    struct sigaction sa = {0};
    sa.sa_handler = on_sigint;
    sigaction(SIGINT, &sa, NULL);
    r->env.abort = &g_abort;
    printf("%sSaetta Code%s %s%s/%s, thinking %s%s\n", BOLD, RESET, DIM, r->model.provider, r->model.id, r->model.thinking, RESET);
    printf("%s%s  (Enter sends, \\ then Enter for a new line, Ctrl-C stops a turn, /help)%s\n\n", DIM, r->env.cwd, RESET);
    VecStr history = {0};
    for (size_t i = 0; i < o->prompts.len; i++) {
        g_abort = 0;
        TRY(prompt(r, o->prompts.ptr[i]));
    }
    for (;;) {
        char *text;
        if (!read_prompt(a, &history, &text)) break;
        Str t = str_trim(str(text));
        if (!t.len) continue;
        if (str_eqz(t, "/exit") || str_eqz(t, "/quit")) break;
        if (str_eqz(t, "/help")) {
            printf("%s/new      a new session\n/exit     quit (also Ctrl-D)\nSessions are pi's (~/.pi/agent/sessions or "
                   "$PI_CODING_AGENT_DIR): saetta code -c continues the last one, and so does pi -c.%s\n\n",
                   DIM, RESET);
            continue;
        }
        if (str_eqz(t, "/new")) {
            TRY(code_session_open(a, &r->env, false, false, &r->session));
            r->env.session_id = json_str(json_get(code_session_header(r->session), "id"));
            r->env.session_file = code_session_path(r->session);
            JValue *f = json_new(a, J_OBJECT);
            json_put(a, f, S("provider"), json_string(a, str(r->model.provider)));
            json_put(a, f, S("modelId"), json_string(a, str(r->model.id)));
            TRY(code_session_append(r->session, "model_change", f));
            f = json_new(a, J_OBJECT);
            json_put(a, f, S("thinkingLevel"), json_string(a, str(r->model.thinking)));
            TRY(code_session_append(r->session, "thinking_level_change", f));
            printf("%snew session%s\n\n", DIM, RESET);
            continue;
        }
        g_abort = 0;
        uint64_t t0 = now_ns();
        Err e = prompt(r, text);
        if (e) printf("%ssaetta code: %s%s\n", RED, e, RESET);
        printf("%s%.1f s%s\n\n", DIM, (double)(now_ns() - t0) / 1e9, RESET);
    }
    vec_free(&history);
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
    r.env.session_id = json_str(json_get(code_session_header(r.session), "id"));
    r.env.session_file = code_session_path(r.session);
    r.env.provider = r.model.provider;
    r.env.model = r.model.id;
    r.env.thinking = level;
    if (!o.print && !piped && !r.json && isatty(0) && isatty(1)) {
        e = interactive(&r, &o);
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
    code_tools_kill_detached();
    vec_free(&o.prompts);
    vec_free(&o.append);
    arena_reset(a);
    return e;
}
