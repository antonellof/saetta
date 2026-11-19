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
