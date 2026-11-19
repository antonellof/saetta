// Anthropic Messages API to OpenAI chat request mapping (see anthropic.h).
// After Strata serve/frontend.py (MIT, Copyright (c) 2026 Niko1221 and the Strata contributors; 13eedf5).
#include "anthropic.h"

#include <stdlib.h>

typedef struct {
    size_t at;
    JValue *msg;
} Result;

static JValue *get(const JValue *v, const char *key) {
    if (!v || v->kind != J_OBJECT) return NULL;
    JValue *r = json_get(v, key);
    return r && r->kind != J_NULL ? r : NULL;
}

// String contents, or has = false.
static bool str_of(const JValue *x, Str *out) {
    if (!x || x->kind != J_STRING) return false;
    *out = x->s;
    return true;
}

static Str str_or(const JValue *x, Str dflt) {
    Str s;
    return str_of(x, &s) ? s : dflt;
}

static JValue *s(Arena *a, Str x) { return json_string(a, x); }
static JValue *sz(Arena *a, const char *x) { return json_string(a, str(x)); }

static void put(Arena *a, JValue *obj, const char *key, JValue *v) { json_put(a, obj, str(key), v); }

static Str buf_to_arena(Arena *a, const Buf *b) {
    return str_n(arena_dup(a, b->ptr ? b->ptr : "", b->len + 1), b->len);
}

static const char billing[] = "x-anthropic-billing-header:";

Str anthropic_pin_billing_stamp(Arena *a, Str sys) {
    if (!str_starts(sys, S(billing))) return sys;
    char *o = arena_alloc(a, sys.len + 1); // zero filled: NUL terminated
    memcpy(o, sys.ptr, sys.len);
    Str os = str_n(o, sys.len);
    ptrdiff_t c = str_find(os, S("cch="), sizeof(billing) - 1);
    if (c >= 0 && c <= 160) {
        size_t v = (size_t)c + 4;
        ptrdiff_t end = str_find_char(os, ';', v);
        if (end >= 0 && (size_t)end > v && (size_t)end - v <= 16) memset(o + v, 'f', (size_t)end - v);
    }
    c = str_find(os, S("cc_version="), 0);
    if (c >= 0 && c <= 160) {
        size_t v = (size_t)c + sizeof("cc_version=") - 1;
        ptrdiff_t end = str_find_char(os, ';', v);
        if (end >= 0 && (size_t)end > v && (size_t)end - v <= 64) {
            // keep the first three dot-separated parts
            size_t dots = 0;
            for (size_t k = v; k < (size_t)end; k++)
                if (o[k] == '.') {
                    dots += 1;
                    if (dots == 3) {
                        memset(o + k + 1, 'f', (size_t)end - (k + 1));
                        break;
                    }
                }
        }
    }
    return os;
}

static const char media_note[] = "[image or document omitted: this model server takes text only]";

// Text of a content value: a string, or the text of its blocks (media replaced by a note).
static Err text_of(Arena *a, const JValue *v, Str *out) {
    if (!v) {
        *out = S("");
        return OK;
    }
    if (v->kind == J_STRING) {
        *out = v->s;
        return OK;
    }
    if (v->kind != J_ARRAY) return "BadRequest";
    Buf o = {0};
    for (size_t i = 0; i < v->arr.len; i++) {
        const JValue *b = v->arr.items[i];
        Str ty, n;
        if (!str_of(get(b, "type"), &ty)) continue;
        if (str_eqz(ty, "text")) {
            buf_addstr(&o, str_or(get(b, "text"), S("")));
        } else if (str_eqz(ty, "image") || str_eqz(ty, "document")) {
            if (o.len > 0) buf_addc(&o, '\n');
            buf_adds(&o, media_note);
        } else if (str_eqz(ty, "tool_reference")) {
            // a tool search result: every declared tool is already in the prompt
            if (str_of(get(b, "tool_name"), &n)) {
                if (o.len > 0) buf_addc(&o, '\n');
                buf_adds(&o, "Tool ");
                buf_addstr(&o, n);
                buf_adds(&o, " is available.");
            }
        }
    }
    *out = buf_to_arena(a, &o);
    buf_free(&o);
    return OK;
}

static JValue *function_obj(Arena *a, JValue *f) {
    JValue *o = json_new(a, J_OBJECT);
    put(a, o, "type", sz(a, "function"));
    put(a, o, "function", f);
    return o;
}

static Err messages(Arena *a, const JValue *in, JValue *msgs) {
    // tool_use ids of the latest assistant turn, to put its results back in call order
    VecStr call_ids = {0};
    VEC(Result) results = {0};
    Buf text = {0}, reasoning = {0};
    Err err = OK;
    for (size_t mi = 0; mi < in->arr.len && !err; mi++) {
        const JValue *m = in->arr.items[mi];
        Str role;
        if (!str_of(get(m, "role"), &role)) { err = "BadRequest"; break; }
        bool sys = str_eqz(role, "system");
        const JValue *content = get(m, "content");
        if (!content || content->kind == J_STRING) {
            JValue *o = json_new(a, J_OBJECT);
            put(a, o, "role", sys ? sz(a, "user") : s(a, role));
            put(a, o, "content", s(a, content ? content->s : S("")));
            json_push(a, msgs, o);
            continue;
        }
        if (content->kind != J_ARRAY) { err = "BadRequest"; break; }
        text.len = 0;
        reasoning.len = 0;
        results.len = 0;
        JValue *calls = json_new(a, J_ARRAY);
        for (size_t bi = 0; bi < content->arr.len; bi++) {
            const JValue *b = content->arr.items[bi];
            Str ty;
            if (!str_of(get(b, "type"), &ty)) continue;
            if (str_eqz(ty, "text")) {
                buf_addstr(&text, str_or(get(b, "text"), S("")));
            } else if (str_eqz(ty, "image") || str_eqz(ty, "document")) {
                if (text.len > 0) buf_addc(&text, '\n');
                buf_adds(&text, media_note);
            } else if (str_eqz(ty, "thinking")) {
                buf_addstr(&reasoning, str_or(get(b, "thinking"), S("")));
            } else if (str_eqz(ty, "tool_use")) {
                Str name;
                if (!str_of(get(b, "name"), &name)) { err = "BadRequest"; break; }
                JValue *f = json_new(a, J_OBJECT);
                put(a, f, "name", s(a, name));
                JValue *input = get(b, "input");
                put(a, f, "arguments", input ? input : json_new(a, J_OBJECT));
                json_push(a, calls, function_obj(a, f));
            } else if (str_eqz(ty, "tool_result")) {
                Str t;
                if ((err = text_of(a, get(b, "content"), &t))) break;
                const JValue *e = get(b, "is_error");
                if (e && e->kind == J_BOOL && e->b) {
                    Buf eb = {0};
                    buf_adds(&eb, "Error: ");
                    buf_addstr(&eb, t);
                    t = buf_to_arena(a, &eb);
                    buf_free(&eb);
                }
                JValue *o = json_new(a, J_OBJECT);
                put(a, o, "role", sz(a, "tool"));
                put(a, o, "content", s(a, t));
                Str id = str_or(get(b, "tool_use_id"), S(""));
                size_t at = call_ids.len + results.len; // unknown ids keep their order, last
                for (size_t k = 0; k < call_ids.len; k++)
                    if (str_eq(call_ids.ptr[k], id)) {
                        at = k;
                        break;
                    }
                vec_push(&results, ((Result){at, o}));
            }
            // redacted_thinking, server tool blocks and others carry nothing the template renders
        }
        if (err) break;
        // a client may send parallel results in the order its tools finished (stable sort)
        for (size_t k = 1; k < results.len; k++) {
            Result x = results.ptr[k];
            size_t j = k;
            while (j > 0 && x.at < results.ptr[j - 1].at) {
                results.ptr[j] = results.ptr[j - 1];
                j--;
            }
            results.ptr[j] = x;
        }
        for (size_t k = 0; k < results.len; k++) json_push(a, msgs, results.ptr[k].msg);
        if (str_eqz(role, "assistant")) {
            call_ids.len = 0;
            for (size_t bi = 0; bi < content->arr.len; bi++) {
                const JValue *b = content->arr.items[bi];
                if (str_eqz(str_or(get(b, "type"), S("")), "tool_use")) vec_push(&call_ids, str_or(get(b, "id"), S("")));
            }
        }
        if (text.len > 0 || reasoning.len > 0 || calls->arr.len > 0) {
            JValue *o = json_new(a, J_OBJECT);
            put(a, o, "role", sys ? sz(a, "user") : s(a, role));
            put(a, o, "content", s(a, buf_to_arena(a, &text)));
            if (reasoning.len > 0) put(a, o, "reasoning_content", s(a, buf_to_arena(a, &reasoning)));
            if (calls->arr.len > 0) put(a, o, "tool_calls", calls);
            json_push(a, msgs, o);
        }
    }
    vec_free(&call_ids);
    vec_free(&results);
    buf_free(&text);
    buf_free(&reasoning);
    return err;
}

Err anthropic_to_openai(Arena *a, const JValue *r, JValue **out_) {
    if (!r || r->kind != J_OBJECT) return "BadRequest";
    JValue *out = json_new(a, J_OBJECT);
    JValue *msgs = json_new(a, J_ARRAY);

    const JValue *sys = get(r, "system");
    if (sys) {
        Str t;
        TRY(text_of(a, sys, &t));
        t = anthropic_pin_billing_stamp(a, t);
        if (t.len > 0) {
            JValue *m = json_new(a, J_OBJECT);
            put(a, m, "role", sz(a, "system"));
            put(a, m, "content", s(a, t));
            json_push(a, msgs, m);
        }
    }

    const JValue *in = get(r, "messages");
    if (!in || in->kind != J_ARRAY || in->arr.len == 0) return "NoMessages";
    TRY(messages(a, in, msgs));
    put(a, out, "messages", msgs);

    const JValue *ts = get(r, "tools");
    if (ts && ts->kind == J_ARRAY && ts->arr.len > 0) {
        JValue *tools = json_new(a, J_ARRAY);
        for (size_t i = 0; i < ts->arr.len; i++) {
            const JValue *t = ts->arr.items[i];
            Str name;
            if (!str_of(get(t, "name"), &name)) continue;
            // server tools (web search and the like) have a type and no input_schema
            JValue *schema = get(t, "input_schema");
            if (!schema) continue;
            JValue *f = json_new(a, J_OBJECT);
            put(a, f, "name", s(a, name));
            put(a, f, "description", s(a, str_or(get(t, "description"), S(""))));
            put(a, f, "parameters", schema);
            json_push(a, tools, function_obj(a, f));
        }
        if (tools->arr.len > 0) put(a, out, "tools", tools);
    }

    // tool_choice: auto (default), any -> required, {type: tool, name} -> that function, none
    const JValue *tc = get(r, "tool_choice");
    Str ty, name;
    if (tc && str_of(get(tc, "type"), &ty)) {
        if (str_eqz(ty, "any")) {
            put(a, out, "tool_choice", sz(a, "required"));
        } else if (str_eqz(ty, "none")) {
            put(a, out, "tool_choice", sz(a, "none"));
        } else if (str_eqz(ty, "tool") && str_of(get(tc, "name"), &name)) {
            JValue *f = json_new(a, J_OBJECT);
            put(a, f, "name", s(a, name));
            put(a, out, "tool_choice", function_obj(a, f));
        }
    }

    JValue *kw = json_new(a, J_OBJECT);
    const JValue *th = get(r, "thinking");
    if (th && str_eqz(str_or(get(th, "type"), S("")), "disabled")) put(a, kw, "enable_thinking", json_bool(a, false));
    const JValue *oc = get(r, "output_config");
    Str eff;
    if (oc && str_of(get(oc, "effort"), &eff)) {
        // the template knows low, medium and xhigh (its default)
        const char *e = str_eqz(eff, "low") ? "low" : str_eqz(eff, "medium") ? "medium" : "xhigh";
        put(a, kw, "reasoning_effort", sz(a, e));
    }
    if (kw->obj.len > 0) put(a, out, "chat_template_kwargs", kw);

    static const char *const carried[] = {"max_tokens", "temperature", "top_p", "top_k", "stream"};
    for (size_t k = 0; k < COUNT(carried); k++) {
        JValue *v = get(r, carried[k]);
        if (v) put(a, out, carried[k], v);
    }
    JValue *st = get(r, "stop_sequences");
    if (st) put(a, out, "stop", st);
    *out_ = out;
    return OK;
}
