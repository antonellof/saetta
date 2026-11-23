// XML tool call parser (see toolcall.h).
// Adapted from sushi src/tool_calls.zig (MIT, Copyright (c) 2026 Theinruj Toranavikrai, David Dalcu; d5e876e).
#include "toolcall.h"

#include <stdlib.h>

#include "chat.h"

static const Str open_tag = S("<tool_call>");
static const Str fn_tag = S("<function=");
static const Str p_open = S("<parameter=");
static const Str p_close = S("</parameter>");
static const Str fn_close = S("</function>");

// std.ascii.isWhitespace
static bool is_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x0b || c == 0x0c; }

static bool in_set(char c, const char *set) { return strchr(set, c) != NULL && c != 0; }

static Str trim_set(Str s, const char *set) {
    size_t a = 0, b = s.len;
    while (a < b && in_set(s.ptr[a], set)) a++;
    while (b > a && in_set(s.ptr[b - 1], set)) b--;
    return str_sub(s, a, b);
}

ptrdiff_t toolcall_start(Str text) { return str_find(text, open_tag, 0); }

static bool starts_at(Str text, size_t p, Str prefix) { return p <= text.len && str_starts(str_sub(text, p, text.len), prefix); }

// The value starting at vs ends at the last `</parameter>` before the next `<parameter=`
// whose closer is followed by another parameter or `</function>`.
static ptrdiff_t value_end(Str text, size_t vs) {
    ptrdiff_t bf = str_find(text, p_open, vs);
    size_t bound = bf >= 0 ? (size_t)bf : text.len;
    Str head = str_sub(text, 0, bound);
    ptrdiff_t best = -1, last = -1;
    size_t i = vs;
    ptrdiff_t c;
    while (i <= bound && (c = str_find(head, p_close, i)) >= 0) {
        last = c;
        size_t k = (size_t)c + p_close.len;
        while (k < bound && is_ws(text.ptr[k])) k += 1;
        if (k == bound || starts_at(text, k, fn_close)) best = c;
        if (k < bound && starts_at(text, k, fn_close)) break;
        i = (size_t)c + 1;
    }
    return best >= 0 ? best : last;
}

static Str unframe(Str v) {
    if (str_starts(v, S("\r\n"))) v = str_sub(v, 2, v.len);
    else if (str_starts(v, S("\n"))) v = str_sub(v, 1, v.len);
    if (str_ends(v, S("\r\n"))) v.len -= 2;
    else if (str_ends(v, S("\n"))) v.len -= 1;
    return v;
}

static bool plausible_name(Str n) {
    if (n.len == 0 || n.len > 128) return false;
    for (size_t i = 0; i < n.len; i++) {
        char ch = n.ptr[i];
        if (ch == '<' || ch == '>' || ch == '=' || ch == '\n' || ch == '"') return false;
    }
    return true;
}

typedef enum { KIND_STRING, KIND_JSON, KIND_UNKNOWN } Kind;

static bool is_str(const JValue *v, const char *s) { return v && v->kind == J_STRING && str_eqz(v->s, s); }

// The declared JSON Schema type of parameter pname of tool name.
static Kind param_type(const JValue *ts, Str name, Str pname) {
    if (!ts || ts->kind != J_ARRAY) return KIND_UNKNOWN;
    for (size_t i = 0; i < ts->arr.len; i++) {
        const JValue *t = ts->arr.items[i];
        if (t->kind != J_OBJECT) continue;
        const JValue *f = json_get(t, "function");
        if (!f) f = t;
        if (f->kind != J_OBJECT) continue;
        const JValue *n = json_get(f, "name");
        if (!n) continue;
        if (n->kind != J_STRING || !str_eq(n->s, name)) continue;
        const JValue *params = json_get(f, "parameters");
        if (!params || params->kind != J_OBJECT) return KIND_UNKNOWN;
        const JValue *props = json_get(params, "properties");
        if (!props || props->kind != J_OBJECT) return KIND_UNKNOWN;
        const JValue *p = NULL;
        // look the key up by Str: names are not NUL terminated
        for (size_t k = 0; k < props->obj.len; k++)
            if (str_eq(props->obj.items[k].key, pname)) p = props->obj.items[k].val;
        if (!p || p->kind != J_OBJECT) return KIND_UNKNOWN;
        const JValue *ty = json_get(p, "type");
        if (!ty) return KIND_UNKNOWN;
        if (ty->kind == J_STRING) return str_eqz(ty->s, "string") ? KIND_STRING : KIND_JSON;
        if (ty->kind == J_ARRAY) {
            for (size_t k = 0; k < ty->arr.len; k++)
                if (is_str(ty->arr.items[k], "string")) return KIND_UNKNOWN;
            return KIND_JSON;
        }
        return KIND_UNKNOWN;
    }
    return KIND_UNKNOWN;
}

static void typed_value(Arena *a, Buf *out, Str v, Kind kind) {
    if (kind != KIND_STRING) {
        Str t = trim_set(v, " \t\r\n");
        JValue *parsed;
        if (!json_parse(a, t, false, &parsed) && (kind == KIND_JSON || parsed->kind != J_STRING)) {
            buf_addstr(out, t);
            return;
        }
    }
    chat_json_string(out, v);
}

Parsed toolcall_parse(Arena *a, Str text, const JValue *tools) {
    ptrdiff_t s0 = toolcall_start(text);
    if (s0 < 0) return (Parsed){text, NULL, 0};
    VEC(Call) calls = {0};
    VecStr seen = {0};
    Buf args = {0};
    size_t pos = (size_t)s0;
    ptrdiff_t f;
    while ((f = str_find(text, fn_tag, pos)) >= 0) {
        size_t name_start = (size_t)f + fn_tag.len;
        size_t name_end = name_start;
        while (name_end < text.len && text.ptr[name_end] != '>' && text.ptr[name_end] != '\n') name_end++;
        if (name_end >= text.len) break;
        Str name = trim_set(str_sub(text, name_start, name_end), " \t\r");
        size_t p = text.ptr[name_end] == '>' ? name_end + 1 : name_end;
        args.len = 0;
        seen.len = 0;
        buf_addc(&args, '{');
        for (;;) {
            while (p < text.len && is_ws(text.ptr[p])) p += 1;
            if (!starts_at(text, p, p_open)) break;
            size_t pn_start = p + p_open.len;
            ptrdiff_t pn_end = str_find_char(text, '>', pn_start);
            if (pn_end < 0) break;
            Str pname = trim_set(str_sub(text, pn_start, (size_t)pn_end), " \t\r\n");
            size_t vs = (size_t)pn_end + 1;
            ptrdiff_t ve = value_end(text, vs);
            if (ve < 0) break;
            p = (size_t)ve + p_close.len;
            if (!plausible_name(pname)) continue;
            bool dup = false;
            for (size_t k = 0; k < seen.len; k++) dup = dup || str_eq(seen.ptr[k], pname);
            if (dup) continue;
            vec_push(&seen, pname);
            if (seen.len > 1) buf_addc(&args, ',');
            chat_json_string(&args, pname);
            buf_addc(&args, ':');
            typed_value(a, &args, unframe(str_sub(text, vs, (size_t)ve)), param_type(tools, name, pname));
        }
        buf_addc(&args, '}');
        if (name.len > 0) vec_push(&calls, ((Call){name, str_n(arena_dup(a, args.ptr, args.len + 1), args.len)}));
        ptrdiff_t e = str_find(text, fn_close, p);
        pos = e >= 0 ? (size_t)e + fn_close.len : text.len;
    }
    Str content = str_sub(text, 0, (size_t)s0);
    while (content.len > 0 && is_ws(content.ptr[content.len - 1])) content.len -= 1;
    Parsed r = {content, NULL, calls.len};
    if (calls.len) r.calls = arena_dup(a, calls.ptr, calls.len * sizeof(Call));
    vec_free(&calls);
    vec_free(&seen);
    buf_free(&args);
    return r;
}

// ---- Inkling 2 ---------------------------------------------------------------------------

static const Str ink_model = S("<|message_model|>");
static const Str ink_thinking = S("<|content_thinking|>");
static const Str ink_text = S("<|content_text|>");
static const Str ink_tool_error = S("<|content_tool_error|>");
static const Str ink_tool_text = S("<|content_invoke_tool_text|>");
static const Str ink_tool_json = S("<|content_invoke_tool_json|>");
static const Str ink_end = S("<|end_message|>");
static const Str ink_end_sampling = S("<|content_model_end_sampling|>");

typedef enum { IM_IDLE, IM_HEADER, IM_TEXT, IM_REASONING, IM_TOOL } InkMode;

// The marker at p (one of the reply's framing tokens), or an empty Str.
static Str ink_marker_at(Str t, size_t p) {
    static const Str *const all[] = {&ink_model, &ink_thinking, &ink_text, &ink_tool_error, &ink_tool_text,
                                     &ink_tool_json, &ink_end, &ink_end_sampling};
    if (p >= t.len || t.ptr[p] != '<') return S("");
    for (size_t k = 0; k < COUNT(all); k++)
        if (starts_at(t, p, *all[k])) return *all[k];
    return S("");
}

// Whether t[p..] (to the end) is a proper prefix of a marker: a later token may complete it.
static bool ink_partial_marker(Str t, size_t p) {
    static const Str *const all[] = {&ink_model, &ink_thinking, &ink_text, &ink_tool_error, &ink_tool_text,
                                     &ink_tool_json, &ink_end, &ink_end_sampling};
    size_t n = t.len - p;
    for (size_t k = 0; k < COUNT(all); k++)
        if (n < all[k]->len && memcmp(t.ptr + p, all[k]->ptr, n) == 0) return true;
    return false;
}

// Length of s without a trailing partial UTF-8 character.
static size_t utf8_whole(Str s) {
    size_t n = s.len, back = 0;
    while (n > 0 && back < 3 && ((uint8_t)s.ptr[n - 1] & 0xC0) == 0x80) n--, back++;
    if (n == 0) return s.len;
    uint8_t b = (uint8_t)s.ptr[n - 1];
    size_t need = b >= 0xF0 ? 4 : b >= 0xE0 ? 3 : b >= 0xC0 ? 2 : 1;
    return need > back + 1 ? n - 1 : s.len;
}

// End of the JSON object starting at p (just past its '}'), or 0 when the text ends first.
// Strings and escapes are skipped, so a brace or a marker spelled inside a string is data.
static size_t json_object_end(Str t, size_t p) {
    int depth = 0;
    bool in_str = false;
    for (size_t i = p; i < t.len; i++) {
        char c = t.ptr[i];
        if (in_str) {
            if (c == '\\') i++;
            else if (c == '"') in_str = false;
        } else if (c == '"') {
            in_str = true;
        } else if (c == '{' || c == '[') {
            depth++;
        } else if (c == '}' || c == ']') {
            if (--depth == 0) return i + 1;
        }
    }
    return 0;
}

// A tool call payload {"name": string, "args": object}; false when it is not one.
static bool ink_call(Arena *a, Str payload, Call *out) {
    JValue *v;
    if (json_parse(a, payload, false, &v) || v->kind != J_OBJECT) return false;
    const JValue *n = json_get(v, "name"), *args = json_get(v, "args");
    if (!n || n->kind != J_STRING || n->s.len == 0) return false;
    if (args && args->kind != J_OBJECT) return false;
    Buf b = {0};
    if (args) json_write(&b, args);
    else buf_adds(&b, "{}");
    *out = (Call){n->s, str_n(arena_dup(a, b.ptr, b.len + 1), b.len)};
    buf_free(&b);
    return true;
}

// A NUL-terminated arena copy of b.
static Str arena_str(Arena *a, const Buf *b) {
    char *d = arena_alloc(a, b->len + 1);
    if (b->len) memcpy(d, b->ptr, b->len);
    d[b->len] = 0;
    return str_n(d, b->len);
}

InkParsed toolcall_inkling(Arena *a, Str t, bool final) {
    InkParsed r = {0};
    Buf reasoning = {0}, content = {0};
    VEC(Call) calls = {0};
    InkMode mode = IM_IDLE;
    size_t hdr = 0; // start of the pending header (IM_HEADER)
    size_t p = 0;
    while (p < t.len) {
        if (mode == IM_TOOL) {
            // {"name":...,"args":{...}}, then optional whitespace and the block end; read
            // JSON-aware, so a brace or a marker spelled inside a string is data
            size_t q = p, oe = 0, e = 0;
            while (q < t.len && is_ws(t.ptr[q])) q++;
            bool open = q == t.len; // the text may still turn into a call
            Str m = S("");
            if (!open && t.ptr[q] == '{') {
                oe = json_object_end(t, q);
                if (!oe) open = true;
                else {
                    e = oe;
                    while (e < t.len && is_ws(t.ptr[e])) e++;
                    m = ink_marker_at(t, e);
                    if (e == t.len || (m.len == 0 && ink_partial_marker(t, e))) open = true;
                }
            }
            // final: a whole object at the very end is a call too (the reply was cut after it)
            bool closed = (m.len && (str_eq(m, ink_end) || str_eq(m, ink_end_sampling))) || (final && oe && e == t.len);
            Call c;
            if (closed && ink_call(a, str_sub(t, q, oe), &c)) {
                vec_push(&calls, c);
                if (str_eq(m, ink_end_sampling)) r.ended = true;
                p = e + m.len;
                mode = IM_IDLE;
                continue;
            }
            // not a call: its raw text is content, up to the block end
            size_t end = t.len, after = t.len;
            if (closed) {
                end = e;
                after = e + m.len;
                if (str_eq(m, ink_end_sampling)) r.ended = true;
            } else {
                ptrdiff_t be = str_find(t, ink_end, oe ? oe : p), bs = str_find(t, ink_end_sampling, oe ? oe : p);
                ptrdiff_t bx = be < 0 ? bs : bs < 0 ? be : MIN(be, bs);
                if (bx >= 0 && !open) {
                    end = (size_t)bx;
                    after = end + (bx == be ? ink_end.len : ink_end_sampling.len);
                    if (bx != be) r.ended = true;
                } else if (!final) {
                    r.in_call = true; // still being written, or its end is not known yet
                    break;
                } else if (bx >= 0) {
                    end = (size_t)bx; // cut off: the first block end
                    after = end + (bx == be ? ink_end.len : ink_end_sampling.len);
                    if (bx != be) r.ended = true;
                }
            }
            buf_add(&content, t.ptr + p, end - p);
            r.bad_call = true;
            p = after;
            mode = IM_IDLE;
            continue;
        }
        Str m = ink_marker_at(t, p);
        if (m.len == 0 && !final && ink_partial_marker(t, p)) break; // wait for the next token
        bool block_end = m.len && (str_eq(m, ink_end) || str_eq(m, ink_end_sampling));
        if (block_end && str_eq(m, ink_end_sampling)) r.ended = true;
        if (mode == IM_TEXT || mode == IM_REASONING) {
            if (block_end) {
                mode = IM_IDLE;
                p += m.len;
                continue;
            }
            // text up to the next possible marker start
            size_t e = p + 1;
            while (e < t.len && t.ptr[e] != '<') e++;
            buf_add(mode == IM_TEXT ? &content : &reasoning, t.ptr + p, e - p);
            p = e;
            continue;
        }
        // IM_IDLE or IM_HEADER: a marker opens, closes or starts a message
        if (m.len) {
            if (mode == IM_HEADER && (block_end || str_eq(m, ink_model)))
                buf_add(&content, t.ptr + hdr, p - hdr); // a header with no content is text
            p += m.len;
            if (str_eq(m, ink_model)) {
                mode = IM_HEADER;
                hdr = p;
            } else if (block_end) {
                mode = IM_IDLE;
            } else if (str_eq(m, ink_thinking)) {
                mode = IM_REASONING;
            } else if (str_eq(m, ink_tool_json)) {
                mode = IM_TOOL;
            } else {
                mode = IM_TEXT; // content_text, content_tool_error, content_invoke_tool_text
            }
            continue;
        }
        size_t e = p + 1;
        while (e < t.len && t.ptr[e] != '<') e++;
        if (mode == IM_IDLE) buf_add(&content, t.ptr + p, e - p);
        p = e; // header bytes stay in t until the header resolves
    }
    if (mode == IM_HEADER && final) buf_add(&content, t.ptr + hdr, t.len - hdr);
    if (!final) {
        reasoning.len = utf8_whole(buf_str(&reasoning));
        content.len = utf8_whole(buf_str(&content));
    }
    r.reasoning = arena_str(a, &reasoning);
    r.content = arena_str(a, &content);
    r.n_calls = calls.len;
    if (calls.len) r.calls = arena_dup(a, calls.ptr, calls.len * sizeof(Call));
    vec_free(&calls);
    buf_free(&reasoning);
    buf_free(&content);
    return r;
}
