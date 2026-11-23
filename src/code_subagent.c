// Subagents (see code_subagent.h): the tool, the children's agent loop on their own threads, and
// their results. The loop is pi's runLoop as code_main.c runs it (agent-loop.ts: the request, the
// tool calls of the answer, their results, until an answer without calls), without what only the
// top-level agent does (retries, compaction, queued messages, events per token).
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi v1.5.2,
// b4afddd): packages/coding-agent/examples/extensions/subagent/index.ts (getFinalOutput,
// getDisplayItems, the output cap, the details kept for the views), packages/agent/src/agent-loop.ts.
#include "code_subagent.h"
#include "code_oai.h"

#include <pthread.h>
#include <stdlib.h>

static bool g_on;

void code_subagent_enable(bool on) { g_on = on; }
bool code_subagent_enabled(void) { return g_on; }

static const char SCHEMA[] =
    "{\"type\":\"function\",\"function\":{\"name\":\"subagent\",\"description\":\"Launch a subagent to carry out a "
    "self-contained task, such as searching the code, reading several files or checking one part of a change. The "
    "subagent sees this conversation so far, has the same tools (except subagent), works on its own and returns its "
    "final message as the result. Several subagent calls in one message run at the same time (up to 3 at once): use "
    "them for independent tasks, never for a step that needs another step's result. Give the subagent everything it "
    "needs in the prompt, and say what it should report back.\",\"parameters\":{\"type\":\"object\",\"required\":["
    "\"description\",\"prompt\"],\"properties\":{\"description\":{\"type\":\"string\",\"description\":\"A short (3-5 "
    "word) description of the task\"},\"prompt\":{\"type\":\"string\",\"description\":\"The task for the subagent to "
    "perform, with what it should report back\"},\"tools\":{\"type\":\"array\",\"items\":{\"type\":\"string\"},"
    "\"description\":\"The tools the subagent may use (default: all of yours except subagent)\"}}}}}";

JValue *code_subagent_schema(Arena *a) {
    JValue *v = NULL;
    json_parse(a, S(SCHEMA), true, &v);
    return v;
}

// ---- the batch and its children --------------------------------------------------------------------

typedef struct SubagentBatch SubagentBatch;

typedef struct {
    SubagentBatch *b;
    Arena a; // the child's own: nothing of it is shared
    Arena *out; // the parent's, for the result
    JValue **msg, **raw;
    char *call_json;
    const JValue *call;
    Str id, description, prompt;
    const char **allow;
    size_t nallow;
    JValue *messages; // the child's messages: its task, then its answers and tool results
    int64_t turns;
    double input, output;
    const char *status; // "running", "done", "error", "aborted"
    const char *error;
    Str final;
    uint64_t t0, t1;
    pthread_t th;
    bool started;
} Child;

struct SubagentBatch {
    SubagentParent p;
    // The model and env as they were when the children started (copied on the parent's thread,
    // under the rpc lock): rpc's set_model and set_thinking_level change the parent's meanwhile.
    CodeModel model;
    CodeEnv env;
    char *system_json, *tools_json, *prefix_json;
    double now_ms;
    VEC(Child *) kids;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int running;
};

// One progress event at a time (the parent's output is single-producer); edit and write calls of
// children one at a time (two children editing one file must not interleave read and write).
static pthread_mutex_t g_emit_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_files_mu = PTHREAD_MUTEX_INITIALIZER;

static char *to_json(const JValue *v) {
    Buf b = {0};
    if (v) json_write(&b, v);
    else buf_adds(&b, "null");
    return b.ptr;
}

static JValue *from_json(Arena *a, const char *text) {
    JValue *v = NULL;
    if (!text || json_parse(a, str(text), true, &v)) return NULL;
    return v;
}

static double wall_ms(void) { return (double)(wall_ns() / 1000000); }

static double num(const JValue *v) {
    if (json_is(v, J_INTEGER)) return (double)v->i;
    if (json_is(v, J_FLOAT)) return v->f;
    if (json_is(v, J_NUMBER_STRING)) {
        char t[64];
        snprintf(t, sizeof t, "%.*s", (int)MIN(v->s.len, sizeof t - 1), v->s.ptr);
        return strtod(t, NULL);
    }
    return 0;
}

SubagentBatch *code_subagent_begin(const SubagentParent *p) {
    SubagentBatch *b = xcalloc(1, sizeof *b);
    b->p = *p;
    b->model = *p->model;
    b->env = *p->env;
    b->p.model = &b->model;
    b->p.env = &b->env;
    b->system_json = to_json(p->system);
    b->tools_json = to_json(p->tools);
    b->prefix_json = to_json(p->prefix);
    b->now_ms = p->env->now_ms;
    pthread_mutex_init(&b->mu, NULL);
    pthread_cond_init(&b->cv, NULL);
    return b;
}

void code_subagent_add(SubagentBatch *b, Arena *a, const JValue *call, JValue **msg, JValue **raw) {
    Child *c = xcalloc(1, sizeof *c);
    c->b = b;
    c->out = a;
    c->msg = msg;
    c->raw = raw;
    c->call_json = to_json(call);
    c->status = "running";
    vec_push(&b->kids, c);
}

// ---- the child's messages -----------------------------------------------------------------------------

static JValue *text_content(Arena *a, Str text) {
    JValue *content = json_new(a, J_ARRAY), *block = json_new(a, J_OBJECT);
    json_put(a, block, S("type"), json_string(a, S("text")));
    json_put(a, block, S("text"), json_string(a, text));
    json_push(a, content, block);
    return content;
}

// The task as the child's user message: what it is, the tools it may use, the task.
static JValue *task_message(Child *c) {
    Arena *a = &c->a;
    Buf t = {0};
    buf_adds(&t, "You are a subagent. The agent working on this conversation handed you the task below and may run other "
                 "subagents at the same time. Work on this task only, with your tools. Your final message is returned to "
                 "that agent as the result: make it complete and self-contained.\n");
    if (c->allow) {
        buf_adds(&t, "You may use only these tools: ");
        for (size_t i = 0; i < c->nallow; i++) buf_printf(&t, "%s%s", i ? ", " : "", c->allow[i]);
        if (!c->nallow) buf_adds(&t, "none");
        buf_adds(&t, ".\n");
    }
    buf_printf(&t, "\nTask: %.*s\n\n%.*s", (int)c->description.len, c->description.ptr, (int)c->prompt.len, c->prompt.ptr);
    JValue *u = json_new(a, J_OBJECT);
    json_put(a, u, S("role"), json_string(a, S("user")));
    json_put(a, u, S("content"), text_content(a, code_utf8_clean(a, buf_str(&t))));
    json_put(a, u, S("timestamp"), json_int(a, (int64_t)wall_ms()));
    buf_free(&t);
    return u;
}

// createErrorToolResult + createToolResultMessage for a call that is not run.
static JValue *error_result(Arena *a, const JValue *call, Str text, double now) {
    JValue *m = json_new(a, J_OBJECT);
    json_put(a, m, S("role"), json_string(a, S("toolResult")));
    json_put(a, m, S("toolCallId"), (JValue *)json_get(call, "id"));
    json_put(a, m, S("toolName"), (JValue *)json_get(call, "name"));
    json_put(a, m, S("content"), text_content(a, text));
    json_put(a, m, S("details"), json_new(a, J_OBJECT));
    json_put(a, m, S("isError"), json_bool(a, true));
    json_put(a, m, S("timestamp"), json_int(a, (int64_t)now));
    return m;
}

// getFinalOutput: the first text part of the last assistant message that has one.
static Str final_output(const JValue *messages) {
    for (size_t i = messages ? messages->arr.len : 0; i-- > 0;) {
        const JValue *m = messages->arr.items[i];
        if (!str_eqz(json_strs(json_get(m, "role")), "assistant")) continue;
        const JValue *c = json_get(m, "content");
        for (size_t k = 0; c && json_is(c, J_ARRAY) && k < c->arr.len; k++)
            if (str_eqz(json_strs(json_get(c->arr.items[k], "type")), "text")) return json_strs(json_get(c->arr.items[k], "text"));
    }
    return S("");
}

// The details kept with the result and sent with each update: the task, the state, the usage and the
// messages (a progress update carries the answers only, not the tool outputs, which can be large).
static JValue *details(Arena *a, Child *c, bool all) {
    JValue *d = json_new(a, J_OBJECT);
    json_put(a, d, S("description"), json_string(a, c->description));
    json_put(a, d, S("prompt"), json_string(a, c->prompt));
    json_put(a, d, S("status"), json_string(a, str(c->status)));
    if (c->error) json_put(a, d, S("errorMessage"), json_string(a, str(c->error)));
    json_put(a, d, S("turns"), json_int(a, c->turns));
    JValue *u = json_new(a, J_OBJECT);
    json_put(a, u, S("input"), json_int(a, (int64_t)c->input));
    json_put(a, u, S("output"), json_int(a, (int64_t)c->output));
    json_put(a, d, S("usage"), u);
    JValue *ms = json_new(a, J_ARRAY);
    for (size_t i = 0; c->messages && i < c->messages->arr.len; i++) {
        const JValue *m = c->messages->arr.items[i];
        if (all || str_eqz(json_strs(json_get(m, "role")), "assistant")) json_push(a, ms, (JValue *)m);
    }
    json_put(a, d, S("messages"), ms);
    return d;
}

// tool_execution_update for the parent's call: the child's state so far.
static void progress(Child *c) {
    if (!c->b->p.emit) return;
    Arena t = {0};
    JValue *e = json_new(&t, J_OBJECT);
    json_put(&t, e, S("type"), json_string(&t, S("tool_execution_update")));
    json_put(&t, e, S("toolCallId"), json_string(&t, c->id));
    json_put(&t, e, S("toolName"), json_string(&t, S(SUBAGENT_TOOL)));
    json_put(&t, e, S("args"), (JValue *)json_get(c->call, "arguments"));
    JValue *pr = json_new(&t, J_OBJECT);
    json_put(&t, pr, S("content"), text_content(&t, final_output(c->messages)));
    json_put(&t, pr, S("details"), details(&t, c, false));
    json_put(&t, e, S("partialResult"), pr);
    pthread_mutex_lock(&g_emit_mu);
    c->b->p.emit(c->b->p.emit_ctx, e);
    pthread_mutex_unlock(&g_emit_mu);
    arena_reset(&t);
}

static bool allowed(const Child *c, Str name) {
    if (str_eqz(name, SUBAGENT_TOOL)) return false;
    const char *const *names = c->allow ? (const char *const *)c->allow : c->b->p.names;
    size_t n = c->allow ? c->nallow : c->b->p.nnames;
    for (size_t i = 0; i < n; i++)
        if (str_eqz(name, names[i])) return true;
    return false;
}

// The child's agent loop (runLoop): until an answer without tool calls, an error or an abort.
static void *child_main(void *arg) {
    Child *c = arg;
    SubagentBatch *b = c->b;
    Arena *a = &c->a;
    const SubagentParent *p = &b->p;
    JValue *system = from_json(a, b->system_json), *tools = from_json(a, b->tools_json), *prefix = from_json(a, b->prefix_json);
    if (tools && tools->kind == J_NULL) tools = NULL;
    CodeEnv env = *p->env;
    env.on_partial = NULL; // a child's bash output is not streamed; its progress is per message
    env.partial_ctx = NULL;
    c->messages = json_new(a, J_ARRAY);
    json_push(a, c->messages, task_message(c));
    progress(c);
    for (;;) {
        if (env.abort && *env.abort) {
            c->status = "aborted";
            c->error = "Subagent was aborted";
            break;
        }
        JValue *conv = json_new(a, J_ARRAY);
        for (size_t i = 0; prefix && json_is(prefix, J_ARRAY) && i < prefix->arr.len; i++) json_push(a, conv, prefix->arr.items[i]);
        for (size_t i = 0; i < c->messages->arr.len; i++) json_push(a, conv, c->messages->arr.items[i]);
        Buf body = {0};
        Err e = code_oai_body(a, p->model, system, conv, tools, &body);
        JValue *msg = NULL;
        env.now_ms = wall_ms();
        if (!e) e = code_oai_complete(a, p->model, &env, buf_str(&body), NULL, &msg);
        buf_free(&body);
        if (e) {
            c->status = "error";
            c->error = arena_printf(a, "saetta code: %s", e);
            break;
        }
        json_put(a, msg, S("thinkingLevel"), json_string(a, str(p->model->thinking ? p->model->thinking : "off")));
        json_push(a, c->messages, msg);
        c->turns += 1;
        const JValue *u = json_get(msg, "usage");
        c->input += num(json_get(u, "input"));
        c->output += num(json_get(u, "output"));
        Str stop = json_strs(json_get(msg, "stopReason"));
        if (str_eqz(stop, "error") || str_eqz(stop, "aborted")) {
            c->status = str_eqz(stop, "aborted") ? "aborted" : "error";
            const char *em = json_str(json_get(msg, "errorMessage"));
            c->error = em ? em : str_eqz(stop, "aborted") ? "Request was aborted" : "Unknown error";
            break;
        }
        const JValue *content = json_get(msg, "content");
        size_t ncalls = 0;
        for (size_t i = 0; content && i < content->arr.len; i++)
            ncalls += str_eqz(json_strs(json_get(content->arr.items[i], "type")), "toolCall");
        if (!ncalls) {
            c->status = "done";
            break;
        }
        progress(c);
        for (size_t i = 0; content && i < content->arr.len; i++) {
            const JValue *call = content->arr.items[i];
            if (!str_eqz(json_strs(json_get(call, "type")), "toolCall")) continue;
            Str name = json_strs(json_get(call, "name"));
            JValue *res = NULL, *raw = NULL;
            env.now_ms = wall_ms();
            if (str_eqz(stop, "length"))
                res = error_result(a, call,
                                   str(arena_printf(a,
                                                    "Tool call \"%.*s\" was not executed: the response hit the output token limit, so "
                                                    "its arguments may be truncated. Re-issue the tool call with complete arguments.",
                                                    (int)name.len, name.ptr)),
                                   env.now_ms);
            else if (env.abort && *env.abort) res = error_result(a, call, S("Operation aborted"), env.now_ms);
            else if (str_eqz(name, SUBAGENT_TOOL)) res = error_result(a, call, S("A subagent cannot start subagents"), env.now_ms);
            else if (!allowed(c, name)) res = error_result(a, call, str(arena_printf(a, "Tool %.*s not found", (int)name.len, name.ptr)), env.now_ms);
            else {
                bool files = str_eqz(name, "edit") || str_eqz(name, "write");
                if (files) pthread_mutex_lock(&g_files_mu);
                Err te = code_tool_run(a, &env, call, &res, &raw);
                if (files) pthread_mutex_unlock(&g_files_mu);
                if (te) res = error_result(a, call, str(arena_printf(a, "saetta code: %s", te)), env.now_ms);
            }
            json_push(a, c->messages, res);
        }
        progress(c);
    }
    c->final = final_output(c->messages);
    c->t1 = now_ns();
    pthread_mutex_lock(&b->mu);
    b->running -= 1;
    pthread_cond_broadcast(&b->cv);
    pthread_mutex_unlock(&b->mu);
    return NULL;
}

static void prepare(Child *c) {
    Arena *a = &c->a;
    c->call = from_json(a, c->call_json);
    c->id = json_strs(json_get(c->call, "id"));
    const JValue *args = json_get(c->call, "arguments");
    c->description = json_strs(json_get(args, "description"));
    c->prompt = json_strs(json_get(args, "prompt"));
    const JValue *tl = json_get(args, "tools");
    if (json_is(tl, J_ARRAY)) {
        c->allow = arena_alloc(a, (tl->arr.len + 1) * sizeof *c->allow);
        for (size_t i = 0; i < tl->arr.len; i++) {
            Str t = json_strs(tl->arr.items[i]);
            bool known = false;
            for (size_t k = 0; k < c->b->p.nnames && !known; k++) known = str_eqz(t, c->b->p.names[k]);
            if (known && !str_eqz(t, SUBAGENT_TOOL)) c->allow[c->nallow++] = str_z(a, t);
        }
    }
    c->t0 = now_ns();
}

// The transcript kept in the details: at most DETAILS_CAP bytes of JSON; past it the oldest tool
// outputs are replaced by a note (the answers stay: they are what the views show).
#define DETAILS_CAP (1u << 20)
static void cap_transcript(Arena *a, JValue *messages) {
    Buf b = {0};
    json_write(&b, messages);
    size_t size = b.len;
    buf_free(&b);
    for (size_t i = 0; size > DETAILS_CAP && i < messages->arr.len; i++) {
        JValue *m = messages->arr.items[i];
        if (!str_eqz(json_strs(json_get(m, "role")), "toolResult")) continue;
        Buf c = {0};
        json_write(&c, json_get(m, "content"));
        size_t was = c.len;
        buf_free(&c);
        JValue *note = text_content(a, str(arena_printf(a, "[Tool output omitted from the stored transcript: %zu bytes]", was)));
        json_put(a, m, S("content"), note);
        size -= MIN(size, was);
    }
}

static void finish(Child *c) {
    Arena *a = c->out;
    bool failed = strcmp(c->status, "done") != 0;
    Str text;
    if (failed) {
        // getFailureEvidence after "Agent <status>: ": the error, then what the child said
        Str said = c->final;
        text = str(arena_printf(a, "Subagent %s: Error: %s%s%.*s", c->status, c->error ? c->error : "Subagent failed",
                                said.len ? "\n\nAssistant output: " : "", (int)said.len, said.ptr));
    } else text = c->final.len ? c->final : S("(no output)");
    text = code_tool_truncate_model_output(a, text);
    cap_transcript(&c->a, c->messages);
    JValue *content = text_content(a, text);
    Buf dj = {0};
    json_write(&dj, details(&c->a, c, true));
    JValue *d = from_json(a, dj.ptr);
    buf_free(&dj);
    JValue *m = json_new(a, J_OBJECT);
    json_put(a, m, S("role"), json_string(a, S("toolResult")));
    json_put(a, m, S("toolCallId"), json_string(a, c->id));
    json_put(a, m, S("toolName"), json_string(a, S(SUBAGENT_TOOL)));
    json_put(a, m, S("content"), content);
    json_put(a, m, S("details"), d);
    json_put(a, m, S("isError"), json_bool(a, failed));
    json_put(a, m, S("durationMs"), json_int(a, (int64_t)((c->t1 - c->t0) / 1000000)));
    json_put(a, m, S("timestamp"), json_int(a, (int64_t)c->b->now_ms));
    JValue *raw = json_new(a, J_OBJECT);
    json_put(a, raw, S("content"), content);
    json_put(a, raw, S("details"), d);
    *c->msg = m;
    *c->raw = raw;
}

Err code_subagent_wait(SubagentBatch *b) {
    // started in call order, at most SUBAGENT_PARALLEL at once
    for (size_t i = 0; i < b->kids.len; i++) {
        Child *c = b->kids.ptr[i];
        prepare(c);
        pthread_mutex_lock(&b->mu);
        while (b->running >= SUBAGENT_PARALLEL) pthread_cond_wait(&b->cv, &b->mu);
        b->running += 1;
        pthread_mutex_unlock(&b->mu);
        pthread_attr_t at;
        pthread_attr_init(&at);
        pthread_attr_setstacksize(&at, 8u << 20); // the JSON and tool code recurse as the agent thread does
        c->started = pthread_create(&c->th, &at, child_main, c) == 0;
        pthread_attr_destroy(&at);
        if (!c->started) child_main(c); // no thread: in this one
    }
    for (size_t i = 0; i < b->kids.len; i++)
        if (b->kids.ptr[i]->started) pthread_join(b->kids.ptr[i]->th, NULL);
    for (size_t i = 0; i < b->kids.len; i++) {
        Child *c = b->kids.ptr[i];
        finish(c);
        arena_reset(&c->a);
        free(c->call_json);
        free(c);
    }
    vec_free(&b->kids);
    free(b->system_json);
    free(b->tools_json);
    free(b->prefix_json);
    pthread_mutex_destroy(&b->mu);
    pthread_cond_destroy(&b->cv);
    free(b);
    return OK;
}
