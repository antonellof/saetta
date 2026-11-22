// Saetta Code OpenAI chat completions client (see code.h, code_oai.h): the request body pi
// sends, the SSE stream, pi's assistant message and its streaming events.
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at
// v1.5.2, b4afddd): packages/ai api/openai-completions.ts (stream, buildParams,
// convertMessages, convertTools, parseChunkUsage, mapStopReason, detectCompat, getCompat),
// api/transform-messages.ts, api/simple-options.ts (clampMaxTokensToContext, thinking
// budgets), utils/estimate.ts, utils/streaming-json.ts (StreamingJsonParser),
// utils/json-parse.ts (repairJson, parseToolCallArguments), utils/error-body.ts,
// utils/event-stream.ts (durationMs), utils/hash.ts;
// and the openai SDK 7.19.0 pi bundles (Apache-2.0, Copyright 2026 OpenAI): core/streaming.js
// (SSE decoding), core/error.js (APIError messages), client.js (status errors).
//
// Not ported, because saetta's models never use them: images (the model is text only, so
// images become pi's placeholders), grammar (custom) tools, strict JSON-schema tools
// (supportsStrictMode adds "strict":false only), Anthropic cache_control, prompt_cache_key
// (needs pi's session id), sampling parameters and thinkingLevelMap (not in models.json for
// saetta), model cost rates (cost is 0), OpenRouter reported cost, session affinity headers.
#include "code_oai.h"
#include "metal/shim.h"
#include "server.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

Str (*code_oai_system_text)(Arena *a, const JValue *system) = code_system_text;

// ---- JavaScript value semantics ---------------------------------------------------------

static JValue *jlit(Arena *a, const char *s) {
    JValue *v = json_new(a, J_STRING);
    v->s = str(s);
    return v;
}
static JValue *jstrs(Arena *a, Str s) { return json_string(a, s); }
static JValue *jobj(Arena *a) { return json_new(a, J_OBJECT); }
static JValue *jarr(Arena *a) { return json_new(a, J_ARRAY); }
static void put(Arena *a, JValue *o, const char *k, JValue *v) { json_put(a, o, str(k), v); }

// Number.prototype.toString (ECMA-262 Number::toString, radix 10), "null" for non-finite as
// JSON.stringify writes it.
static int js_num_fmt(char *out, size_t cap, double x) {
    if (!isfinite(x)) return snprintf(out, cap, "null");
    if (x == 0) return snprintf(out, cap, "0");
    char e[40];
    int prec = 1;
    for (; prec <= 17; prec++) {
        snprintf(e, sizeof e, "%.*e", prec - 1, x);
        if (strtod(e, NULL) == x) break;
    }
    // e: [-]d[.ddd]e[+-]XX
    const char *p = e;
    bool neg = *p == '-';
    if (neg) p++;
    char dig[24];
    int k = 0;
    for (; *p && *p != 'e'; p++)
        if (*p != '.') dig[k++] = *p;
    while (k > 1 && dig[k - 1] == '0') k--;
    dig[k] = 0;
    int n = atoi(p + 1) + 1;
    char *o = out;
    size_t left = cap;
#define PUT(...) do { int w_ = snprintf(o, left, __VA_ARGS__); if (w_ < 0 || (size_t)w_ >= left) return -1; o += w_; left -= (size_t)w_; } while (0)
    if (neg) PUT("-");
    if (k <= n && n <= 21) {
        PUT("%s", dig);
        for (int i = 0; i < n - k; i++) PUT("0");
    } else if (0 < n && n <= 21) {
        PUT("%.*s.%s", n, dig, dig + n);
    } else if (-6 < n && n <= 0) {
        PUT("0.");
        for (int i = 0; i < -n; i++) PUT("0");
        PUT("%s", dig);
    } else {
        PUT("%c", dig[0]);
        if (k > 1) PUT(".%s", dig + 1);
        PUT("e%c%d", n - 1 >= 0 ? '+' : '-', abs(n - 1));
    }
#undef PUT
    return (int)(o - out);
}

static JValue *jnum(Arena *a, double x) {
    char tmp[64];
    int n = js_num_fmt(tmp, sizeof tmp, x);
    JValue *v = json_new(a, J_NUMBER_STRING);
    v->s = (Str){arena_strndup(a, tmp, (size_t)n), (size_t)n};
    return v;
}

static double num_of(const JValue *v) {
    if (!v) return 0;
    switch (v->kind) {
    case J_INTEGER: return (double)v->i;
    case J_FLOAT: return v->f;
    case J_NUMBER_STRING: {
        char z[400];
        size_t n = v->s.len < sizeof z - 1 ? v->s.len : sizeof z - 1;
        memcpy(z, v->s.ptr, n);
        z[n] = 0;
        return strtod(z, NULL);
    }
    default: return 0;
    }
}

static bool is_num(const JValue *v) {
    return v && (v->kind == J_INTEGER || v->kind == J_FLOAT || v->kind == J_NUMBER_STRING);
}

// JavaScript truthiness of a JSON value (NULL is undefined).
static bool truthy(const JValue *v) {
    if (!v) return false;
    switch (v->kind) {
    case J_NULL: return false;
    case J_BOOL: return v->b;
    case J_STRING: return v->s.len > 0;
    case J_INTEGER: case J_FLOAT: case J_NUMBER_STRING: {
        double d = num_of(v);
        return d == d && d != 0;
    }
    default: return true;
    }
}

static const JValue *nullish(const JValue *v) { return v && v->kind != J_NULL ? v : NULL; }

// UTF-16 length of UTF-8 text (JavaScript's String.length).
static size_t u16len(Str s) {
    size_t n = 0;
    for (size_t i = 0; i < s.len; i++) {
        uint8_t c = (uint8_t)s.ptr[i];
        if ((c & 0xC0) == 0x80) continue;
        n += c >= 0xF0 ? 2 : 1;
    }
    return n;
}

// Byte offset of UTF-16 index u (clamped to the end).
static size_t u16off(Str s, size_t u) {
    size_t n = 0, i = 0;
    while (i < s.len && n < u) {
        uint8_t c = (uint8_t)s.ptr[i];
        size_t len = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
        n += len == 4 ? 2 : 1;
        i += len;
    }
    return i > s.len ? s.len : i;
}

static uint32_t utf8_at(Str s, size_t i, size_t *len) {
    uint8_t c = (uint8_t)s.ptr[i];
    size_t l = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
    if (i + l > s.len) l = 1;
    uint32_t cp = l == 1 ? c : c & (0x7F >> l);
    for (size_t k = 1; k < l; k++) cp = cp << 6 | ((uint8_t)s.ptr[i + k] & 0x3F);
    *len = l;
    return cp;
}

// String.prototype.trim's WhiteSpace and LineTerminator.
static bool js_space(uint32_t c) {
    return c == 9 || c == 10 || c == 11 || c == 12 || c == 13 || c == 32 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F ||
           c == 0x3000 || c == 0xFEFF;
}

static Str js_trim(Str s) {
    size_t b = 0, e = s.len;
    while (b < e) {
        size_t l;
        if (!js_space(utf8_at(s, b, &l))) break;
        b += l;
    }
    while (e > b) {
        size_t k = e - 1;
        while (k > b && ((uint8_t)s.ptr[k] & 0xC0) == 0x80) k--;
        size_t l;
        if (!js_space(utf8_at(s, k, &l))) break;
        e = k;
    }
    return str_n(s.ptr + b, e - b);
}

static bool js_blank(Str s) { return js_trim(s).len == 0; }

// A canonical array index ("0".."4294967294"): JavaScript orders these keys first, ascending.
static bool array_index(Str k, uint64_t *out) {
    if (k.len == 0 || k.len > 10) return false;
    if (k.len > 1 && k.ptr[0] == '0') return false;
    uint64_t v = 0;
    for (size_t i = 0; i < k.len; i++) {
        if (k.ptr[i] < '0' || k.ptr[i] > '9') return false;
        v = v * 10 + (uint64_t)(k.ptr[i] - '0');
    }
    if (v >= 4294967295u) return false;
    *out = v;
    return true;
}

// obj[key] = v with JavaScript's property order.
static void js_put(Arena *a, JValue *o, Str key, JValue *v) {
    for (size_t i = 0; i < o->obj.len; i++)
        if (str_eq(o->obj.items[i].key, key)) {
            o->obj.items[i].val = v;
            return;
        }
    json_put(a, o, key, v); // appended
    uint64_t idx;
    if (!array_index(key, &idx)) return;
    size_t at = 0;
    for (; at + 1 < o->obj.len; at++) {
        uint64_t j;
        if (!array_index(o->obj.items[at].key, &j) || j > idx) break;
    }
    JMember m = o->obj.items[o->obj.len - 1];
    memmove(&o->obj.items[at + 1], &o->obj.items[at], (o->obj.len - 1 - at) * sizeof(JMember));
    o->obj.items[at] = m;
}

static void obj_remove(JValue *o, Str key) {
    for (size_t i = 0; i < o->obj.len; i++)
        if (str_eq(o->obj.items[i].key, key)) {
            memmove(&o->obj.items[i], &o->obj.items[i + 1], (o->obj.len - 1 - i) * sizeof(JMember));
            o->obj.len--;
            return;
        }
}

static void obj_insert_at(Arena *a, JValue *o, size_t at, Str key, JValue *v) {
    json_put(a, o, key, v);
    if (at >= o->obj.len - 1) return;
    JMember m = o->obj.items[o->obj.len - 1];
    memmove(&o->obj.items[at + 1], &o->obj.items[at], (o->obj.len - 1 - at) * sizeof(JMember));
    o->obj.items[at] = m;
}

static size_t obj_index(const JValue *o, const char *key) {
    Str k = str(key);
    for (size_t i = 0; i < o->obj.len; i++)
        if (str_eq(o->obj.items[i].key, k)) return i;
    return o->obj.len;
}

void code_oai_js_write(Buf *out, const JValue *v) {
    char tmp[64];
    switch (v->kind) {
    case J_INTEGER:
        if (v->i <= (1LL << 53) && v->i >= -(1LL << 53)) buf_printf(out, "%lld", (long long)v->i);
        else buf_add(out, tmp, (size_t)js_num_fmt(tmp, sizeof tmp, (double)v->i));
        break;
    case J_FLOAT: buf_add(out, tmp, (size_t)js_num_fmt(tmp, sizeof tmp, v->f)); break;
    case J_NUMBER_STRING: {
        char z[400];
        size_t n = v->s.len < sizeof z - 1 ? v->s.len : sizeof z - 1;
        memcpy(z, v->s.ptr, n);
        z[n] = 0;
        buf_add(out, tmp, (size_t)js_num_fmt(tmp, sizeof tmp, strtod(z, NULL)));
        break;
    }
    case J_ARRAY:
        buf_addc(out, '[');
        for (size_t i = 0; i < v->arr.len; i++) {
            if (i) buf_addc(out, ',');
            code_oai_js_write(out, v->arr.items[i]);
        }
        buf_addc(out, ']');
        break;
    case J_OBJECT:
        buf_addc(out, '{');
        for (size_t i = 0; i < v->obj.len; i++) {
            if (i) buf_addc(out, ',');
            json_write_string(out, v->obj.items[i].key);
            buf_addc(out, ':');
            code_oai_js_write(out, v->obj.items[i].val);
        }
        buf_addc(out, '}');
        break;
    default: json_write(out, v);
    }
}

static Str js_stringify(Arena *a, const JValue *v) {
    Buf b = {0};
    code_oai_js_write(&b, v);
    Str s = {arena_strndup(a, b.ptr ? b.ptr : "", b.len), b.len};
    buf_free(&b);
    return s;
}

static void buf_utf8(Buf *b, uint32_t cp) {
    if (cp < 0x80) buf_addc(b, (char)cp);
    else if (cp < 0x800) {
        buf_addc(b, (char)(0xC0 | cp >> 6));
        buf_addc(b, (char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        buf_addc(b, (char)(0xE0 | cp >> 12));
        buf_addc(b, (char)(0x80 | (cp >> 6 & 0x3F)));
        buf_addc(b, (char)(0x80 | (cp & 0x3F)));
    } else {
        buf_addc(b, (char)(0xF0 | cp >> 18));
        buf_addc(b, (char)(0x80 | (cp >> 12 & 0x3F)));
        buf_addc(b, (char)(0x80 | (cp >> 6 & 0x3F)));
        buf_addc(b, (char)(0x80 | (cp & 0x3F)));
    }
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// ---- JSON.parse (V8's JsonParser, its error messages included) --------------------------

typedef struct {
    Arena *a;
    Str s;
    size_t i;
    char *err;
} JP;

static void jp_where(JP *p, size_t at, size_t *pos, size_t *line, size_t *col) {
    *pos = u16len(str_n(p->s.ptr, at));
    size_t ln = 1, start = 0;
    for (size_t i = 0; i < at; i++) {
        char c = p->s.ptr[i];
        if (c == '\n' || c == '\r') {
            if (c == '\r' && i + 1 < at && p->s.ptr[i + 1] == '\n') i++;
            ln++;
            start = i + 1;
        }
    }
    *line = ln;
    *col = u16len(str_n(p->s.ptr + start, at - start)) + 1;
}

static void *jp_fail(JP *p, const char *what, size_t at) {
    if (p->err) return NULL;
    size_t pos, line, col;
    jp_where(p, at, &pos, &line, &col);
    // V8 says "after JSON at position", every other message "in JSON at position"
    bool after = strstr(what, "after JSON") != NULL;
    p->err = arena_printf(p->a, "%s%s at position %zu (line %zu column %zu)", what, after ? "" : " in JSON", pos, line, col);
    return NULL;
}

// ReportUnexpectedToken without an explicit message.
static void *jp_unexpected(JP *p, size_t at) {
    if (p->err) return NULL;
    if (at >= p->s.len) {
        p->err = arena_strdup(p->a, "Unexpected end of JSON input");
        return NULL;
    }
    Str s = p->s;
    if (str_eqz(s, "undefined") || str_eqz(s, "NaN") || str_eqz(s, "Infinity") || str_eqz(s, "[object Object]")) {
        p->err = arena_printf(p->a, "\"%.*s\" is not valid JSON", (int)s.len, s.ptr);
        return NULL;
    }
    // the token class of the character: a number or a string has its own message
    char c0 = s.ptr[at];
    if (c0 == '-' || (c0 >= '0' && c0 <= '9')) return jp_fail(p, "Unexpected number", at);
    if (c0 == '"') return jp_fail(p, "Unexpected string", at);
    size_t l;
    utf8_at(s, at, &l);
    Str tok = str_n(s.ptr + at, l);
    size_t len = u16len(s), pos = u16len(str_n(s.ptr, at));
    enum { CTX = 10 };
    if (len < 2 * CTX + 1) {
        p->err = arena_printf(p->a, "Unexpected token '%.*s', \"%.*s\" is not valid JSON", (int)tok.len, tok.ptr,
                              (int)s.len, s.ptr);
        return NULL;
    }
    size_t from = 0, to = len;
    const char *pre = "", *post = "";
    if (pos < CTX) {
        to = pos + CTX;
        post = "...";
    } else if (pos < len - CTX) {
        from = pos - CTX;
        to = pos + CTX;
        pre = post = "...";
    } else {
        from = pos - CTX;
        pre = "...";
    }
    size_t b0 = u16off(s, from), b1 = u16off(s, to);
    p->err = arena_printf(p->a, "Unexpected token '%.*s', %s\"%.*s\"%s is not valid JSON", (int)tok.len, tok.ptr, pre,
                          (int)(b1 - b0), s.ptr + b0, post);
    return NULL;
}

static void jp_ws(JP *p) {
    while (p->i < p->s.len) {
        char c = p->s.ptr[p->i];
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
        p->i++;
    }
}

static bool jp_string(JP *p, Str *out) {
    Buf b = {0};
    p->i++; // opening quote
    for (;;) {
        if (p->i >= p->s.len) {
            buf_free(&b);
            jp_fail(p, "Unterminated string", p->s.len);
            return false;
        }
        uint8_t c = (uint8_t)p->s.ptr[p->i];
        if (c == '"') {
            p->i++;
            break;
        }
        if (c < 0x20) {
            buf_free(&b);
            jp_fail(p, "Bad control character in string literal", p->i);
            return false;
        }
        if (c != '\\') {
            buf_addc(&b, (char)c);
            p->i++;
            continue;
        }
        p->i++;
        if (p->i >= p->s.len) {
            buf_free(&b);
            jp_unexpected(p, p->s.len);
            return false;
        }
        char e = p->s.ptr[p->i];
        const char *simple = strchr("\"\\/bfnrt", e);
        if (simple && e) {
            static const char dec[] = "\"\\/\b\f\n\r\t";
            buf_addc(&b, dec[simple - "\"\\/bfnrt"]);
            p->i++;
            continue;
        }
        if (e != 'u') {
            buf_free(&b);
            jp_fail(p, "Bad escaped character", p->i);
            return false;
        }
        uint32_t u = 0;
        for (int k = 1; k <= 4; k++) {
            if (p->i + (size_t)k >= p->s.len) {
                buf_free(&b);
                jp_fail(p, "Bad Unicode escape", p->s.len);
                return false;
            }
            int h = hexval(p->s.ptr[p->i + (size_t)k]);
            if (h < 0) {
                buf_free(&b);
                jp_fail(p, "Bad Unicode escape", p->i + (size_t)k);
                return false;
            }
            u = u << 4 | (uint32_t)h;
        }
        p->i += 5;
        if (u >= 0xD800 && u <= 0xDBFF && p->i + 5 < p->s.len && p->s.ptr[p->i] == '\\' && p->s.ptr[p->i + 1] == 'u') {
            uint32_t lo = 0;
            bool ok = true;
            for (int k = 2; k <= 5; k++) {
                int h = hexval(p->s.ptr[p->i + (size_t)k]);
                if (h < 0) ok = false;
                lo = lo << 4 | (uint32_t)(h < 0 ? 0 : h);
            }
            if (ok && lo >= 0xDC00 && lo <= 0xDFFF) {
                p->i += 6;
                u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
            }
        }
        if (u >= 0xD800 && u <= 0xDFFF) u = 0xFFFD; // a lone surrogate has no UTF-8 form
        buf_utf8(&b, u);
    }
    *out = (Str){arena_strndup(p->a, b.ptr ? b.ptr : "", b.len), b.len};
    buf_free(&b);
    return true;
}

static bool isdig(char c) { return c >= '0' && c <= '9'; }

static JValue *jp_value(JP *p, int depth) {
    if (depth > 512) return jp_fail(p, "Maximum nesting depth exceeded", p->i);
    jp_ws(p);
    if (p->i >= p->s.len) return jp_unexpected(p, p->i);
    char c = p->s.ptr[p->i];
    if (c == '{') {
        JValue *o = jobj(p->a);
        p->i++;
        jp_ws(p);
        if (p->i < p->s.len && p->s.ptr[p->i] == '}') {
            p->i++;
            return o;
        }
        bool first = true;
        for (;;) {
            jp_ws(p);
            if (p->i >= p->s.len || p->s.ptr[p->i] != '"')
                return jp_fail(p, first ? "Expected property name or '}'" : "Expected double-quoted property name", p->i);
            Str key;
            if (!jp_string(p, &key)) return NULL;
            jp_ws(p);
            if (p->i >= p->s.len || p->s.ptr[p->i] != ':') return jp_fail(p, "Expected ':' after property name", p->i);
            p->i++;
            JValue *v = jp_value(p, depth + 1);
            if (!v) return NULL;
            js_put(p->a, o, key, v);
            jp_ws(p);
            if (p->i < p->s.len && p->s.ptr[p->i] == ',') {
                p->i++;
                first = false;
                continue;
            }
            if (p->i < p->s.len && p->s.ptr[p->i] == '}') {
                p->i++;
                return o;
            }
            return jp_fail(p, "Expected ',' or '}' after property value", p->i);
        }
    }
    if (c == '[') {
        JValue *arr = jarr(p->a);
        p->i++;
        jp_ws(p);
        if (p->i < p->s.len && p->s.ptr[p->i] == ']') {
            p->i++;
            return arr;
        }
        for (;;) {
            JValue *v = jp_value(p, depth + 1);
            if (!v) return NULL;
            json_push(p->a, arr, v);
            jp_ws(p);
            if (p->i < p->s.len && p->s.ptr[p->i] == ',') {
                p->i++;
                continue;
            }
            if (p->i < p->s.len && p->s.ptr[p->i] == ']') {
                p->i++;
                return arr;
            }
            return jp_fail(p, "Expected ',' or ']' after array element", p->i);
        }
    }
    if (c == '"') {
        Str s;
        if (!jp_string(p, &s)) return NULL;
        JValue *v = json_new(p->a, J_STRING);
        v->s = s;
        return v;
    }
    if (c == '-' || isdig(c)) {
        size_t start = p->i;
        if (c == '-') {
            p->i++;
            if (p->i >= p->s.len || !isdig(p->s.ptr[p->i])) return jp_fail(p, "No number after minus sign", p->i);
        }
        if (p->s.ptr[p->i] == '0') {
            p->i++;
            if (p->i < p->s.len && isdig(p->s.ptr[p->i])) return jp_fail(p, "Unexpected number", p->i);
        } else
            while (p->i < p->s.len && isdig(p->s.ptr[p->i])) p->i++;
        if (p->i < p->s.len && p->s.ptr[p->i] == '.') {
            p->i++;
            if (p->i >= p->s.len || !isdig(p->s.ptr[p->i])) return jp_fail(p, "Unterminated fractional number", p->i);
            while (p->i < p->s.len && isdig(p->s.ptr[p->i])) p->i++;
        }
        if (p->i < p->s.len && (p->s.ptr[p->i] == 'e' || p->s.ptr[p->i] == 'E')) {
            p->i++;
            if (p->i < p->s.len && (p->s.ptr[p->i] == '+' || p->s.ptr[p->i] == '-')) p->i++;
            if (p->i >= p->s.len || !isdig(p->s.ptr[p->i])) return jp_fail(p, "Exponent part is missing a number", p->i);
            while (p->i < p->s.len && isdig(p->s.ptr[p->i])) p->i++;
        }
        char *z = arena_strndup(p->a, p->s.ptr + start, p->i - start);
        return jnum(p->a, strtod(z, NULL));
    }
    static const char *const lits[] = {"true", "false", "null"};
    for (int k = 0; k < 3; k++) {
        if (c != lits[k][0]) continue;
        size_t n = strlen(lits[k]);
        for (size_t j = 1; j < n; j++)
            if (p->i + j >= p->s.len || p->s.ptr[p->i + j] != lits[k][j]) return jp_unexpected(p, p->i + j);
        p->i += n;
        if (k == 2) return json_new(p->a, J_NULL);
        return json_bool(p->a, k == 0);
    }
    return jp_unexpected(p, p->i);
}

JValue *code_oai_js_parse(Arena *a, Str text, char **err) {
    JP p = {.a = a, .s = text};
    JValue *v = jp_value(&p, 0);
    if (v) {
        jp_ws(&p);
        if (p.i < p.s.len) v = jp_fail(&p, "Unexpected non-whitespace character after JSON", p.i);
    }
    if (!v && err) *err = p.err ? p.err : arena_strdup(a, "Unexpected end of JSON input");
    return v;
}

// pi's repairJson: escapes raw control characters in strings and doubles the backslash of an
// invalid escape.
static Str repair_json(Arena *a, Str s) {
    Buf b = {0};
    bool in = false;
    for (size_t i = 0; i < s.len; i++) {
        char c = s.ptr[i];
        if (!in) {
            buf_addc(&b, c);
            if (c == '"') in = true;
            continue;
        }
        if (c == '"') {
            buf_addc(&b, c);
            in = false;
            continue;
        }
        if (c == '\\') {
            if (i + 1 >= s.len) {
                buf_adds(&b, "\\\\");
                continue;
            }
            char nx = s.ptr[i + 1];
            if (nx == 'u' && i + 5 < s.len) {
                bool hex = true;
                for (int k = 2; hex && k <= 5; k++) hex = hexval(s.ptr[i + (size_t)k]) >= 0;
                if (hex) {
                    buf_add(&b, s.ptr + i, 6);
                    i += 5;
                    continue;
                }
            }
            if (nx && strchr("\"\\/bfnrt", nx)) {
                buf_addc(&b, '\\');
                buf_addc(&b, nx);
                i++;
                continue;
            }
            buf_adds(&b, "\\\\");
            continue;
        }
        if ((uint8_t)c < 0x20) {
            switch (c) {
            case '\b': buf_adds(&b, "\\b"); break;
            case '\f': buf_adds(&b, "\\f"); break;
            case '\n': buf_adds(&b, "\\n"); break;
            case '\r': buf_adds(&b, "\\r"); break;
            case '\t': buf_adds(&b, "\\t"); break;
            default: buf_printf(&b, "\\u%04x", (unsigned)(uint8_t)c);
            }
            continue;
        }
        buf_addc(&b, c);
    }
    Str r = {arena_strndup(a, b.ptr ? b.ptr : "", b.len), b.len};
    buf_free(&b);
    return r;
}

// parseToolCallArguments: *arguments always set, *error (argumentsError) when malformed.
static void parse_tool_args(Arena *a, Str text, JValue **args, char **error) {
    *args = jobj(a);
    *error = NULL;
    if (js_blank(text)) return;
    char *msg = NULL;
    JValue *v = code_oai_js_parse(a, text, &msg);
    if (!v) {
        Str r = repair_json(a, text);
        if (!str_eq(r, text)) {
            msg = NULL;
            v = code_oai_js_parse(a, r, &msg);
        }
    }
    if (v && v->kind != J_OBJECT) {
        v = NULL;
        msg = "Tool arguments must be a JSON object";
    }
    if (v) {
        *args = v;
        return;
    }
    *error = arena_printf(a, "%s\n\nReceived arguments:\n%.*s", msg, (int)text.len, text.ptr);
}

// ---- growable strings referenced by live JValues ------------------------------------------

typedef struct {
    JValue *node; // J_STRING whose s points at p
    char *p;
    size_t len, cap;
} GStr;

static void gstr_init(Arena *a, GStr *g, JValue *node) {
    g->node = node;
    g->cap = 32;
    g->p = arena_alloc(a, g->cap);
    g->len = 0;
    node->s = (Str){g->p, 0};
}

static void gstr_add(Arena *a, GStr *g, Str s) {
    if (g->len + s.len + 1 > g->cap) {
        size_t nc = g->cap * 2;
        while (nc < g->len + s.len + 1) nc *= 2;
        char *np = arena_alloc(a, nc);
        memcpy(np, g->p, g->len);
        g->p = np;
        g->cap = nc;
    }
    memcpy(g->p + g->len, s.ptr, s.len);
    g->len += s.len;
    g->p[g->len] = 0;
    g->node->s = (Str){g->p, g->len};
}

// ---- StreamingJsonParser (partial tool-call arguments) ------------------------------------
// The value lives in one mutable tree that the partial message shows; pi copies the open
// path into immutable snapshots instead. The copy-credit pacing of pi's parser is not kept:
// the partial is always current.

typedef enum { SL_NONE, SL_ROOT, SL_MEMBER, SL_ELEMENT } SlotKind;
typedef struct {
    SlotKind kind;
    Str key;
    size_t index;
} Slot;
typedef enum { PH_KEY, PH_COLON, PH_VALUE, PH_COMMA } Phase;
typedef struct {
    JValue *c;
    Slot slot;
    bool has_key;
    Str key;
    Phase phase;
} Frame;

typedef struct {
    Arena *a;
    JValue *root;
    bool done;
    VEC(Frame) stack;
    // open string
    bool in_str, str_key;
    Slot str_slot;
    GStr text;
    Buf keybuf;
    char esc[8];
    int esc_len;
    uint32_t hi; // a high surrogate just appended (as U+FFFD), for the low one that may follow
    // unterminated number or literal
    Buf scalar;
    Slot scalar_slot;
    // what the pending scalar replaced in the live tree
    bool pend_on;
    JValue *pend_prev;
    JValue *pend_root;
} SJP;

static Slot value_slot(SJP *p) {
    if (p->stack.len == 0) return (Slot){p->root ? SL_NONE : SL_ROOT, {0}, 0};
    Frame *f = &p->stack.ptr[p->stack.len - 1];
    if (f->c->kind == J_ARRAY) return (Slot){SL_ELEMENT, {0}, f->c->arr.len};
    if ((f->phase == PH_VALUE || f->phase == PH_COLON) && f->has_key) return (Slot){SL_MEMBER, f->key, 0};
    return (Slot){SL_NONE, {0}, 0};
}

static void sjp_place(SJP *p, Slot s, JValue *v) {
    if (s.kind == SL_NONE) return;
    if (s.kind == SL_ROOT) {
        p->root = v;
        return;
    }
    JValue *c = p->stack.ptr[p->stack.len - 1].c;
    if (s.kind == SL_ELEMENT) {
        if (s.index == c->arr.len) json_push(p->a, c, v);
        else c->arr.items[s.index] = v;
        return;
    }
    js_put(p->a, c, s.key, v);
}

static void value_ended(SJP *p) {
    if (p->stack.len == 0) {
        if (p->root) p->done = true;
        return;
    }
    p->stack.ptr[p->stack.len - 1].phase = PH_COMMA;
}

static const struct {
    const char *word;
    int kind; // 0 true, 1 false, 2 null
} LITS[] = {{"true", 0}, {"false", 1}, {"null", 2}};

static JValue *lit_value(SJP *p, int k) {
    return k == 2 ? json_new(p->a, J_NULL) : json_bool(p->a, k == 0);
}

// NUMBER = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$/
static bool number_re(Str t) {
    size_t i = 0;
    if (i < t.len && t.ptr[i] == '-') i++;
    if (i >= t.len) return false;
    if (t.ptr[i] == '0') i++;
    else if (t.ptr[i] >= '1' && t.ptr[i] <= '9')
        while (i < t.len && isdig(t.ptr[i])) i++;
    else return false;
    if (i < t.len && t.ptr[i] == '.') {
        i++;
        size_t d = i;
        while (i < t.len && isdig(t.ptr[i])) i++;
        if (i == d) return false;
    }
    if (i < t.len && (t.ptr[i] == 'e' || t.ptr[i] == 'E')) {
        i++;
        if (i < t.len && (t.ptr[i] == '+' || t.ptr[i] == '-')) i++;
        size_t d = i;
        while (i < t.len && isdig(t.ptr[i])) i++;
        if (i == d) return false;
    }
    return i == t.len;
}

static JValue *scalar_value(SJP *p, Str t, bool pending) {
    for (int k = 0; k < 3; k++) {
        Str w = str(LITS[k].word);
        if (pending ? (t.len <= w.len && memcmp(w.ptr, t.ptr, t.len) == 0) : str_eq(t, w)) return lit_value(p, k);
    }
    if (number_re(t)) return jnum(p->a, strtod(str_z(p->a, t), NULL));
    if (!pending) return NULL;
    // NUMBER_WITHOUT_EXPONENT_DIGITS = /^(-?(?:0|[1-9]\d*)(?:\.\d+)?)[eE][+-]?$/
    size_t e = t.len;
    if (e && (t.ptr[e - 1] == '+' || t.ptr[e - 1] == '-')) e--;
    if (e && (t.ptr[e - 1] == 'e' || t.ptr[e - 1] == 'E')) {
        Str m = str_n(t.ptr, e - 1);
        bool no_exp = m.len && !memchr(m.ptr, 'e', m.len) && !memchr(m.ptr, 'E', m.len);
        if (no_exp && number_re(m)) return jnum(p->a, strtod(str_z(p->a, m), NULL));
    }
    return NULL;
}

// The value at slot s of the innermost frame (NULL when absent).
static JValue *slot_get(SJP *p, Slot s) {
    if (s.kind == SL_ROOT) return p->root;
    if (s.kind == SL_NONE || p->stack.len == 0) return NULL;
    JValue *c = p->stack.ptr[p->stack.len - 1].c;
    if (s.kind == SL_ELEMENT) return s.index < c->arr.len ? c->arr.items[s.index] : NULL;
    for (size_t i = 0; i < c->obj.len; i++)
        if (str_eq(c->obj.items[i].key, s.key)) return c->obj.items[i].val;
    return NULL;
}

static void pend_clear(SJP *p) {
    if (!p->pend_on) return;
    p->pend_on = false;
    Slot s = p->scalar_slot;
    if (s.kind == SL_ROOT) {
        p->pend_root = NULL;
        return;
    }
    JValue *c = p->stack.ptr[p->stack.len - 1].c;
    if (s.kind == SL_ELEMENT) {
        if (p->pend_prev) c->arr.items[s.index] = p->pend_prev;
        else if (s.index < c->arr.len) c->arr.len--;
        return;
    }
    if (p->pend_prev) js_put(p->a, c, s.key, p->pend_prev);
    else obj_remove(c, s.key);
}

static void pend_show(SJP *p) {
    pend_clear(p);
    Slot s = p->scalar_slot;
    if (s.kind == SL_NONE || p->scalar.len == 0 || p->scalar.len > 64) return;
    JValue *v = scalar_value(p, buf_str(&p->scalar), true);
    if (!v) return;
    p->pend_on = true;
    if (s.kind == SL_ROOT) {
        p->pend_root = v;
        return;
    }
    p->pend_prev = slot_get(p, s);
    sjp_place(p, s, v);
}

static void end_scalar(SJP *p) {
    pend_clear(p);
    Slot s = p->scalar_slot;
    JValue *v = scalar_value(p, buf_str(&p->scalar), false);
    p->scalar.len = 0;
    p->scalar_slot = (Slot){SL_NONE, {0}, 0};
    if (v) sjp_place(p, s, v);
    value_ended(p);
}

static void str_append(SJP *p, Str t) {
    p->hi = 0;
    if (p->str_key) {
        buf_addstr(&p->keybuf, t);
        return;
    }
    if (p->str_slot.kind != SL_NONE || p->text.node) gstr_add(p->a, &p->text, t);
}

static void start_string(SJP *p) {
    p->in_str = true;
    p->esc_len = 0;
    p->hi = 0;
    Frame *f = p->stack.len ? &p->stack.ptr[p->stack.len - 1] : NULL;
    if (f && f->c->kind == J_OBJECT && (f->phase == PH_KEY || f->phase == PH_COMMA)) {
        p->str_key = true;
        p->keybuf.len = 0;
        return;
    }
    p->str_key = false;
    p->str_slot = value_slot(p);
    JValue *node = json_new(p->a, J_STRING);
    gstr_init(p->a, &p->text, node);
    sjp_place(p, p->str_slot, node);
}

static void end_string(SJP *p) {
    p->in_str = false;
    Frame *f = p->stack.len ? &p->stack.ptr[p->stack.len - 1] : NULL;
    if (p->str_key) {
        if (!f) return;
        f->key = (Str){arena_strndup(p->a, p->keybuf.ptr ? p->keybuf.ptr : "", p->keybuf.len), p->keybuf.len};
        f->has_key = true;
        f->phase = PH_COLON;
        return;
    }
    p->text.node = NULL;
    value_ended(p);
}

// One escape character; false when c is not part of the escape and is read normally.
static bool cont_escape(SJP *p, char c) {
    if (p->esc_len == 1) {
        const char *simple = c ? strchr("\"\\/bfnrt", c) : NULL;
        if (simple) {
            static const char dec[] = "\"\\/\b\f\n\r\t";
            p->esc_len = 0;
            str_append(p, str_n(&dec[simple - "\"\\/bfnrt"], 1));
            return true;
        }
        if (c == 'u') {
            p->esc[p->esc_len++] = 'u';
            return true;
        }
        p->esc_len = 0;
        str_append(p, S("\\"));
        return false;
    }
    if (hexval(c) >= 0) {
        p->esc[p->esc_len++] = c;
        if (p->esc_len < 6) return true;
        uint32_t u = 0;
        for (int k = 2; k < 6; k++) u = u << 4 | (uint32_t)hexval(p->esc[k]);
        p->esc_len = 0;
        uint32_t hi = p->hi;
        if (hi && u >= 0xDC00 && u <= 0xDFFF) {
            // the pair completes: replace the U+FFFD shown for the high half
            if (p->str_key) p->keybuf.len -= 3;
            else if (p->text.node) {
                p->text.len -= 3;
                p->text.p[p->text.len] = 0;
            }
            u = 0x10000 + ((hi - 0xD800) << 10) + (u - 0xDC00);
        }
        Buf b = {0};
        buf_utf8(&b, u >= 0xD800 && u <= 0xDFFF ? 0xFFFD : u);
        str_append(p, buf_str(&b));
        buf_free(&b);
        if (u >= 0xD800 && u <= 0xDBFF) p->hi = u;
        return true;
    }
    Str lit = str_n(p->esc, (size_t)p->esc_len);
    char tmp[8];
    tmp[0] = '\\';
    memcpy(tmp + 1, lit.ptr + 1, lit.len - 1);
    p->esc_len = 0;
    str_append(p, str_n(tmp, lit.len));
    return false;
}

static void open_container(SJP *p, JValue *c) {
    Slot s = value_slot(p);
    sjp_place(p, s, c);
    Frame f = {.c = c, .slot = s, .phase = c->kind == J_ARRAY ? PH_VALUE : PH_KEY};
    vec_push(&p->stack, f);
}

static void sjp_append(SJP *p, Str t) {
    size_t i = 0;
    while (i < t.len && !p->done) {
        char c = t.ptr[i];
        if (p->in_str) {
            if (p->esc_len) {
                if (cont_escape(p, c)) {
                    i++;
                    continue;
                }
            }
            size_t run = i;
            while (i < t.len && t.ptr[i] != '"' && t.ptr[i] != '\\') i++;
            if (i > run) str_append(p, str_n(t.ptr + run, i - run));
            if (i >= t.len) break;
            if (t.ptr[i] == '"') end_string(p);
            else {
                p->esc[0] = '\\';
                p->esc_len = 1;
            }
            i++;
            continue;
        }
        if (p->scalar.len) {
            size_t run = i;
            while (i < t.len && !strchr(" \t\n\r,:{}[]\"", t.ptr[i])) i++;
            if (i > run) buf_add(&p->scalar, t.ptr + run, i - run);
            if (i < t.len) end_scalar(p);
            continue;
        }
        i++;
        switch (c) {
        case ' ': case '\t': case '\n': case '\r': break;
        case '"': start_string(p); break;
        case '{': open_container(p, jobj(p->a)); break;
        case '[': open_container(p, jarr(p->a)); break;
        case '}': case ']':
            if (p->stack.len) {
                p->stack.len--;
                value_ended(p);
            }
            break;
        case ':': {
            Frame *f = p->stack.len ? &p->stack.ptr[p->stack.len - 1] : NULL;
            if (f && f->c->kind == J_OBJECT && f->phase == PH_COLON) f->phase = PH_VALUE;
            break;
        }
        case ',': {
            Frame *f = p->stack.len ? &p->stack.ptr[p->stack.len - 1] : NULL;
            if (f) f->phase = f->c->kind == J_ARRAY ? PH_VALUE : PH_KEY;
            break;
        }
        default:
            buf_addc(&p->scalar, c);
            p->scalar_slot = value_slot(p);
        }
    }
    if (p->scalar.len) pend_show(p);
}

// The partial arguments: the root when it is an object, else {}.
static JValue *sjp_args(SJP *p, JValue *empty) {
    JValue *r = p->stack.len == 0 && p->pend_root ? p->pend_root : p->root;
    return r && r->kind == J_OBJECT ? r : empty;
}

// The parser on its own, for the interactive mode: it rebuilds the partial arguments of a
// streamed tool call on the UI thread from the raw deltas.
struct CodeArgsParser {
    Arena a;
    SJP p;
    JValue *empty;
};

CodeArgsParser *code_args_parser_new(void) {
    CodeArgsParser *c = xcalloc(1, sizeof *c);
    c->p.a = &c->a;
    c->empty = jobj(&c->a);
    return c;
}

void code_args_parser_feed(CodeArgsParser *c, Str text) {
    c->p.a = &c->a;
    sjp_append(&c->p, text);
}

const JValue *code_args_parser_value(CodeArgsParser *c) { return sjp_args(&c->p, c->empty); }

void code_args_parser_free(CodeArgsParser *c) {
    if (!c) return;
    vec_free(&c->p.stack);
    buf_free(&c->p.keybuf);
    buf_free(&c->p.scalar);
    arena_reset(&c->a);
    free(c);
}

// ---- compat (getCompat over detectCompat) ------------------------------------------------

typedef struct {
    bool store, developer, reasoning_effort, usage_streaming, finish_reason;
    const char *max_tokens_field;
    bool tool_result_name, assistant_after_tool_result, non_null_assistant, tools_with_history;
    bool thinking_as_text, reasoning_content_on_assistant;
    const char *thinking_format;
    const JValue *chat_template_kwargs, *chat_template_args;
    bool zai_tool_stream, thinking_budget;
    const char *budget_field;
    bool strict, mid_system;
    bool long_cache;
    const JValue *vllm_priority;
} Compat;

static bool contains(const char *h, const char *n) { return h && strstr(h, n); }

static void lower(char *s) {
    for (; *s; s++)
        if (*s >= 'A' && *s <= 'Z') *s += 32;
}

static Compat get_compat(const CodeModel *m) {
    const char *p = m->provider ? m->provider : "", *u = m->base_url ? m->base_url : "";
    char ul[1024];
    snprintf(ul, sizeof ul, "%s", u);
    lower(ul);
    bool zai = !strcmp(p, "zai") || !strcmp(p, "zai-coding-cn") || contains(u, "api.z.ai") || contains(u, "open.bigmodel.cn");
    bool together = !strcmp(p, "together") || contains(u, "api.together.ai") || contains(u, "api.together.xyz");
    bool moonshot = !strcmp(p, "moonshotai") || !strcmp(p, "moonshotai-cn") || contains(u, "api.moonshot.");
    bool openrouter = !strcmp(p, "openrouter") || contains(u, "openrouter.ai");
    bool cfw = !strcmp(p, "cloudflare-workers-ai") || contains(u, "api.cloudflare.com");
    bool cfg = !strcmp(p, "cloudflare-ai-gateway") || contains(u, "gateway.ai.cloudflare.com");
    bool nvidia = !strcmp(p, "nvidia") || contains(u, "integrate.api.nvidia.com");
    bool antling = !strcmp(p, "ant-ling") || contains(u, "api.ant-ling.com");
    bool cerebras = !strcmp(p, "cerebras") || contains(u, "cerebras.ai");
    bool deepseek = !strcmp(p, "deepseek") || contains(ul, "deepseek.com");
    bool grok = !strcmp(p, "xai") || contains(u, "api.x.ai");
    bool nonstd = nvidia || cerebras || grok || together || contains(u, "chutes.ai") || deepseek || zai || moonshot ||
                  !strcmp(p, "opencode") || contains(u, "opencode.ai") || cfw || cfg || antling;
    bool use_max = contains(u, "chutes.ai") || deepseek || moonshot || cfg || together || nvidia || antling || zai;
    const char *id = m->id ? m->id : "";
    bool or_dev = openrouter && (!strncmp(id, "anthropic/", 10) || !strncmp(id, "openai/", 7));
    Compat c = {
        .store = !nonstd,
        .developer = or_dev || (!nonstd && !openrouter),
        .reasoning_effort = !grok && !zai && !moonshot && !together && !cfg && !nvidia && !antling,
        .usage_streaming = true,
        .finish_reason = true,
        .max_tokens_field = use_max ? "max_tokens" : "max_completion_tokens",
        .reasoning_content_on_assistant = deepseek,
        .thinking_format = deepseek ? "deepseek" : zai ? "zai" : together ? "together" : antling ? "ant-ling"
                           : openrouter ? "openrouter" : "openai",
        .long_cache = !(together || cfw || cfg || nvidia || antling),
    };
    const JValue *x = m->compat;
    if (!x) {
        c.developer = m->supports_developer_role;
        c.store = m->supports_store;
        c.reasoning_effort = m->supports_reasoning_effort;
        if (m->max_tokens_field) c.max_tokens_field = m->max_tokens_field;
        if (m->thinking_format) c.thinking_format = m->thinking_format;
        return c;
    }
#define B(field, key) do { const JValue *v_ = nullish(json_get(x, key)); if (v_) c.field = truthy(v_); } while (0)
#define SS(field, key) do { const JValue *v_ = nullish(json_get(x, key)); if (v_ && v_->kind == J_STRING) c.field = v_->s.ptr; } while (0)
    B(store, "supportsStore");
    B(developer, "supportsDeveloperRole");
    B(reasoning_effort, "supportsReasoningEffort");
    B(usage_streaming, "supportsUsageInStreaming");
    B(finish_reason, "supportsFinishReason");
    SS(max_tokens_field, "maxTokensField");
    B(tool_result_name, "requiresToolResultName");
    B(assistant_after_tool_result, "requiresAssistantAfterToolResult");
    B(non_null_assistant, "requiresNonNullAssistantContent");
    B(tools_with_history, "requiresToolsWithToolHistory");
    B(thinking_as_text, "requiresThinkingAsText");
    B(reasoning_content_on_assistant, "requiresReasoningContentOnAssistantMessages");
    SS(thinking_format, "thinkingFormat");
    c.chat_template_kwargs = nullish(json_get(x, "chatTemplateKwargs"));
    c.chat_template_args = nullish(json_get(x, "chatTemplateArgs"));
    B(zai_tool_stream, "zaiToolStream");
    B(thinking_budget, "supportsThinkingTokenBudget");
    SS(budget_field, "thinkingTokenBudgetField");
    B(strict, "supportsStrictMode");
    B(mid_system, "supportsMidConvoSystemMessages");
    B(long_cache, "supportsLongCacheRetention");
    c.vllm_priority = nullish(json_get(x, "vllmPriority"));
#undef B
#undef SS
    return c;
}

// The reasoning effort pi requests for the session's thinking level (clampThinkingLevel with
// no thinkingLevelMap), NULL when off.
static const char *effort_of(const CodeModel *m) {
    const char *t = m->thinking;
    if (!m->reasoning || !t || !strcmp(t, "off")) return NULL;
    if (!strcmp(t, "xhigh") || !strcmp(t, "max")) return "high";
    if (!strcmp(t, "minimal") || !strcmp(t, "low") || !strcmp(t, "medium") || !strcmp(t, "high")) return t;
    return NULL; // an unknown level clamps to the first available one: "off"
}

static int64_t thinking_budget(const char *effort) {
    if (!strcmp(effort, "minimal")) return 1024;
    if (!strcmp(effort, "low")) return 2048;
    if (!strcmp(effort, "medium")) return 8192;
    return 16384;
}

// ---- transform-messages.ts ----------------------------------------------------------------

static const char *USER_IMAGE = "(image omitted: model does not support images)";
static const char *TOOL_IMAGE = "(tool image omitted: model does not support images)";

static JValue *shallow(Arena *a, const JValue *o) {
    JValue *c = jobj(a);
    for (size_t i = 0; i < o->obj.len; i++) json_put(a, c, o->obj.items[i].key, o->obj.items[i].val);
    return c;
}

static bool is_type(const JValue *block, const char *t) {
    const char *s = json_str(json_get(block, "type"));
    return s && !strcmp(s, t);
}

static Str sget(const JValue *o, const char *k) { return json_strs(json_get(o, k)); }

// replaceImagesWithPlaceholder (the model is text only).
static JValue *downgrade(Arena *a, const JValue *content, const char *placeholder) {
    if (!json_is(content, J_ARRAY)) return (JValue *)content;
    bool any = false;
    for (size_t i = 0; i < content->arr.len; i++) any |= is_type(content->arr.items[i], "image");
    if (!any) return (JValue *)content;
    JValue *r = jarr(a);
    bool prev = false;
    for (size_t i = 0; i < content->arr.len; i++) {
        JValue *b = content->arr.items[i];
        if (is_type(b, "image")) {
            if (!prev) {
                JValue *t = jobj(a);
                put(a, t, "type", jlit(a, "text"));
                put(a, t, "text", jlit(a, placeholder));
                json_push(a, r, t);
            }
            prev = true;
            continue;
        }
        json_push(a, r, b);
        prev = str_eqz(sget(b, "text"), placeholder);
    }
    return r;
}

static char *b36(Arena *a, uint32_t v) {
    char tmp[16];
    int n = 0;
    do {
        tmp[n++] = "0123456789abcdefghijklmnopqrstuvwxyz"[v % 36];
        v /= 36;
    } while (v);
    char *o = arena_alloc(a, (size_t)n + 1);
    for (int i = 0; i < n; i++) o[i] = tmp[n - 1 - i];
    return o;
}

static char *short_hash(Arena *a, Str s) {
    uint32_t h1 = 0xdeadbeef, h2 = 0x41c6ce57;
    for (size_t i = 0; i < s.len;) {
        size_t l;
        uint32_t cp = utf8_at(s, i, &l);
        i += l;
        uint32_t units[2] = {cp, 0};
        int nu = 1;
        if (cp >= 0x10000) {
            units[0] = 0xD800 + ((cp - 0x10000) >> 10);
            units[1] = 0xDC00 + ((cp - 0x10000) & 0x3FF);
            nu = 2;
        }
        for (int k = 0; k < nu; k++) {
            h1 = (h1 ^ units[k]) * 2654435761u;
            h2 = (h2 ^ units[k]) * 1597334677u;
        }
    }
    uint32_t n1 = ((h1 ^ (h1 >> 16)) * 2246822507u) ^ ((h2 ^ (h2 >> 13)) * 3266489909u);
    uint32_t n2 = ((h2 ^ (h2 >> 16)) * 2246822507u) ^ ((n1 ^ (n1 >> 13)) * 3266489909u);
    return arena_printf(a, "%s%s", b36(a, n2), b36(a, n1));
}

static Str sanitize_id(Arena *a, Str s) {
    char *o = arena_alloc(a, s.len + 1);
    size_t n = 0;
    for (size_t i = 0; i < s.len;) {
        size_t l;
        uint32_t cp = utf8_at(s, i, &l);
        i += l;
        bool ok = (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') || (cp >= '0' && cp <= '9') || cp == '_' || cp == '-';
        o[n++] = ok ? (char)cp : '_';
        if (cp >= 0x10000) o[n++] = '_'; // two UTF-16 units, two replacements
    }
    return str_n(o, n);
}

// convertMessages' normalizeToolCallId.
static Str normalize_id(Arena *a, const CodeModel *m, Str id) {
    ptrdiff_t bar = str_find_char(id, '|', 0);
    if (bar >= 0) {
        Str call = sanitize_id(a, str_n(id.ptr, (size_t)bar));
        Str item = sanitize_id(a, str_n(id.ptr + bar + 1, id.len - (size_t)bar - 1));
        Str comb = item.len ? str(arena_printf(a, "%.*s_%.*s", (int)call.len, call.ptr, (int)item.len, item.ptr)) : call;
        if (comb.len <= 40) return comb;
        char *h = short_hash(a, id);
        h[8 < strlen(h) ? 8 : strlen(h)] = 0;
        size_t keep = 40 - strlen(h) - 1;
        if (keep < 1) keep = 1;
        if (keep > call.len) keep = call.len;
        return str(arena_printf(a, "%.*s_%s", (int)keep, call.ptr, h));
    }
    if (m->provider && !strcmp(m->provider, "openai") && u16len(id) > 40) return str_n(id.ptr, u16off(id, 40));
    return id;
}

typedef VEC(JValue *) VecJ;

static JValue *synthetic_result(Arena *a, const JValue *tc, double now) {
    JValue *r = jobj(a);
    put(a, r, "role", jlit(a, "toolResult"));
    put(a, r, "toolCallId", json_get(tc, "id"));
    put(a, r, "toolName", json_get(tc, "name"));
    JValue *content = jarr(a), *t = jobj(a);
    put(a, t, "type", jlit(a, "text"));
    put(a, t, "text", jlit(a, "No result provided"));
    json_push(a, content, t);
    put(a, r, "content", content);
    put(a, r, "isError", json_bool(a, true));
    put(a, r, "timestamp", jnum(a, now));
    return r;
}

static void transform(Arena *a, const CodeModel *m, const JValue *messages, VecJ *out) {
    StrMap idmap = {0};
    VecJ first = {0};
    size_t n = json_is(messages, J_ARRAY) ? messages->arr.len : 0;
    for (size_t i = 0; i < n; i++) {
        JValue *msg = messages->arr.items[i];
        const char *role = json_str(json_get(msg, "role"));
        if (!role) {
            vec_push(&first, msg);
            continue;
        }
        JValue *content = json_get(msg, "content");
        if (!content || content->kind == J_NULL) {
            msg = shallow(a, msg);
            json_put(a, msg, S("content"), jarr(a));
            content = json_get(msg, "content");
        }
        if (!strcmp(role, "user") || !strcmp(role, "toolResult")) {
            JValue *d = downgrade(a, content, !strcmp(role, "user") ? USER_IMAGE : TOOL_IMAGE);
            if (d != content) {
                msg = shallow(a, msg);
                json_put(a, msg, S("content"), d);
            }
        }
        if (!strcmp(role, "toolResult")) {
            void **mapped = strmap_get(&idmap, sget(msg, "toolCallId"));
            if (mapped && !str_eq(*(Str *)*mapped, sget(msg, "toolCallId"))) {
                msg = shallow(a, msg);
                json_put(a, msg, S("toolCallId"), jstrs(a, *(Str *)*mapped));
            }
        } else if (!strcmp(role, "assistant")) {
            bool same = str_eqz(sget(msg, "provider"), m->provider ? m->provider : "") &&
                        str_eqz(sget(msg, "api"), "openai-completions") && str_eqz(sget(msg, "model"), m->id ? m->id : "");
            JValue *nc = jarr(a);
            for (size_t k = 0; json_is(content, J_ARRAY) && k < content->arr.len; k++) {
                JValue *b = content->arr.items[k];
                if (is_type(b, "thinking")) {
                    if (!same) continue;
                    if (truthy(json_get(b, "redacted")) || truthy(json_get(b, "thinkingSignature"))) {
                        json_push(a, nc, b);
                        continue;
                    }
                    if (js_blank(sget(b, "thinking"))) continue;
                    json_push(a, nc, b);
                } else if (is_type(b, "text")) {
                    if (same) json_push(a, nc, b);
                    else {
                        JValue *t = jobj(a);
                        put(a, t, "type", jlit(a, "text"));
                        put(a, t, "text", json_get(b, "text"));
                        json_push(a, nc, t);
                    }
                } else if (is_type(b, "toolCall")) {
                    JValue *tc = b;
                    if (!same && json_get(tc, "thoughtSignature") && truthy(json_get(tc, "thoughtSignature"))) {
                        tc = shallow(a, tc);
                        obj_remove(tc, S("thoughtSignature"));
                    }
                    if (!same) {
                        Str id = sget(tc, "id"), nid = normalize_id(a, m, id);
                        if (!str_eq(nid, id)) {
                            Str *keep = ARENA_NEW(a, Str);
                            *keep = nid;
                            strmap_put(&idmap, id, keep);
                            if (tc == b) tc = shallow(a, tc);
                            json_put(a, tc, S("id"), jstrs(a, nid));
                        }
                    }
                    json_push(a, nc, tc);
                } else
                    json_push(a, nc, b);
            }
            msg = shallow(a, msg);
            json_put(a, msg, S("content"), nc);
        }
        vec_push(&first, msg);
    }
    strmap_free(&idmap);

    VecJ pending = {0}, held = {0};
    StrMap have = {0};
    double now = (double)(wall_ns() / 1000000);
#define CLOSE() do { \
        for (size_t q = 0; q < pending.len; q++) \
            if (!strmap_has(&have, sget(pending.ptr[q], "id"))) vec_push(out, synthetic_result(a, pending.ptr[q], now)); \
        pending.len = 0; strmap_free(&have); \
        for (size_t q = 0; q < held.len; q++) vec_push(out, held.ptr[q]); \
        held.len = 0; \
    } while (0)
    for (size_t i = 0; i < first.len; i++) {
        JValue *msg = first.ptr[i];
        const char *role = json_str(json_get(msg, "role"));
        role = role ? role : "";
        if (!strcmp(role, "assistant")) {
            CLOSE();
            const char *sr = json_str(json_get(msg, "stopReason"));
            if (sr && (!strcmp(sr, "error") || !strcmp(sr, "aborted"))) continue;
            JValue *content = json_get(msg, "content");
            for (size_t k = 0; json_is(content, J_ARRAY) && k < content->arr.len; k++)
                if (is_type(content->arr.items[k], "toolCall")) vec_push(&pending, content->arr.items[k]);
            vec_push(out, msg);
        } else if (!strcmp(role, "toolResult")) {
            strmap_put(&have, sget(msg, "toolCallId"), msg);
            vec_push(out, msg);
        } else if (!strcmp(role, "system")) {
            if (pending.len) vec_push(&held, msg);
            else vec_push(out, msg);
        } else if (!strcmp(role, "user")) {
            CLOSE();
            vec_push(out, msg);
        } else
            vec_push(out, msg);
    }
    CLOSE();
#undef CLOSE
    vec_free(&pending);
    vec_free(&held);
    vec_free(&first);
}

// ---- reasoning_details ----------------------------------------------------------------------

static bool valid_detail(const JValue *d) {
    if (!json_is(d, J_OBJECT)) return false;
    const JValue *id = json_get(d, "id"), *fmt = json_get(d, "format"), *idx = json_get(d, "index");
    if (id && id->kind != J_NULL && id->kind != J_STRING) return false;
    if (fmt && fmt->kind != J_STRING) return false;
    if (idx && !is_num(idx)) return false;
    const char *t = json_str(json_get(d, "type"));
    if (!t) return false;
    if (!strcmp(t, "reasoning.summary")) return json_is(json_get(d, "summary"), J_STRING);
    if (!strcmp(t, "reasoning.encrypted")) return json_is(json_get(d, "data"), J_STRING);
    if (!strcmp(t, "reasoning.text")) {
        const JValue *sig = json_get(d, "signature");
        return json_is(json_get(d, "text"), J_STRING) && (!sig || sig->kind == J_NULL || sig->kind == J_STRING);
    }
    return false;
}

static JValue *parse_details(Arena *a, Str sig) {
    if (!sig.len) return NULL;
    JValue *v = code_oai_js_parse(a, sig, NULL);
    if (!json_is(v, J_ARRAY) || v->arr.len == 0) return NULL;
    for (size_t i = 0; i < v->arr.len; i++)
        if (!valid_detail(v->arr.items[i])) return NULL;
    return v;
}

static JValue *parse_legacy_detail(Arena *a, Str sig) {
    if (!sig.len) return NULL;
    JValue *v = code_oai_js_parse(a, sig, NULL);
    if (!valid_detail(v) || !str_eqz(sget(v, "type"), "reasoning.encrypted")) return NULL;
    const JValue *id = json_get(v, "id");
    if (!json_is(id, J_STRING) || !id->s.len || !sget(v, "data").len) return NULL;
    return v;
}

// ---- request body -----------------------------------------------------------------------------

static void push_text_part(Arena *a, JValue *arr, JValue *text) {
    JValue *p = jobj(a);
    put(a, p, "type", jlit(a, "text"));
    put(a, p, "text", text);
    json_push(a, arr, p);
}

static Str join_texts(Arena *a, const JValue *content, const char *sep, bool skip_blank) {
    Buf b = {0};
    bool firstp = true;
    for (size_t i = 0; json_is(content, J_ARRAY) && i < content->arr.len; i++) {
        const JValue *blk = content->arr.items[i];
        if (!is_type(blk, "text")) continue;
        Str t = sget(blk, "text");
        if (skip_blank && js_blank(t)) continue;
        if (!firstp) buf_adds(&b, sep);
        firstp = false;
        buf_addstr(&b, t);
    }
    Str s = {arena_strndup(a, b.ptr ? b.ptr : "", b.len), b.len};
    buf_free(&b);
    return s;
}

static JValue *msg_of(Arena *a, const char *role, JValue *content) {
    JValue *o = jobj(a);
    put(a, o, "role", jlit(a, role));
    put(a, o, "content", content);
    return o;
}

static JValue *convert_assistant(Arena *a, const CodeModel *m, const Compat *c, const JValue *msg) {
    JValue *am = jobj(a);
    put(a, am, "role", jlit(a, "assistant"));
    put(a, am, "content", c->assistant_after_tool_result || c->non_null_assistant ? jlit(a, "") : json_new(a, J_NULL));
    const JValue *content = json_get(msg, "content");
    JValue *text_parts = jarr(a);
    for (size_t i = 0; i < content->arr.len; i++) {
        const JValue *b = content->arr.items[i];
        if (is_type(b, "text") && !js_blank(sget(b, "text"))) push_text_part(a, text_parts, json_get(b, "text"));
    }
    Str text = join_texts(a, content, "", true);
    VecJ thinking = {0}, calls = {0};
    for (size_t i = 0; i < content->arr.len; i++) {
        if (is_type(content->arr.items[i], "thinking")) vec_push(&thinking, content->arr.items[i]);
        if (is_type(content->arr.items[i], "toolCall")) vec_push(&calls, content->arr.items[i]);
    }
    JValue *preserved = NULL;
    for (size_t i = 0; i < thinking.len && !preserved; i++) preserved = parse_details(a, sget(thinking.ptr[i], "thinkingSignature"));
    if (!preserved) {
        JValue *legacy = jarr(a);
        for (size_t i = 0; i < calls.len; i++) {
            JValue *d = parse_legacy_detail(a, sget(calls.ptr[i], "thoughtSignature"));
            if (d) json_push(a, legacy, d);
        }
        if (legacy->arr.len) preserved = legacy;
    }
    VecJ nonempty = {0};
    for (size_t i = 0; i < thinking.len; i++)
        if (!js_blank(sget(thinking.ptr[i], "thinking"))) vec_push(&nonempty, thinking.ptr[i]);
    if (nonempty.len) {
        if (c->thinking_as_text) {
            Buf b = {0};
            for (size_t i = 0; i < nonempty.len; i++) {
                if (i) buf_adds(&b, "\n\n");
                buf_addstr(&b, sget(nonempty.ptr[i], "thinking"));
            }
            JValue *arr = jarr(a);
            push_text_part(a, arr, jstrs(a, buf_str(&b)));
            buf_free(&b);
            for (size_t i = 0; i < text_parts->arr.len; i++) json_push(a, arr, text_parts->arr.items[i]);
            put(a, am, "content", arr);
        } else {
            if (text.len) put(a, am, "content", jstrs(a, text));
            if (!preserved) {
                Str sig = sget(nonempty.ptr[0], "thinkingSignature");
                if (m->provider && !strcmp(m->provider, "opencode-go") && str_eqz(sig, "reasoning")) sig = S("reasoning_content");
                if (str_eqz(sig, "reasoning") || str_eqz(sig, "reasoning_content") || str_eqz(sig, "reasoning_text")) {
                    Buf b = {0};
                    for (size_t i = 0; i < nonempty.len; i++) {
                        if (i) buf_addc(&b, '\n');
                        buf_addstr(&b, sget(nonempty.ptr[i], "thinking"));
                    }
                    json_put(a, am, (Str){arena_strndup(a, sig.ptr, sig.len), sig.len}, jstrs(a, buf_str(&b)));
                    buf_free(&b);
                }
            }
        }
    } else if (text.len)
        put(a, am, "content", jstrs(a, text));
    if (calls.len) {
        JValue *tcs = jarr(a);
        for (size_t i = 0; i < calls.len; i++) {
            JValue *tc = jobj(a), *fn = jobj(a);
            put(a, tc, "id", json_get(calls.ptr[i], "id"));
            put(a, tc, "type", jlit(a, "function"));
            put(a, fn, "name", json_get(calls.ptr[i], "name"));
            const JValue *args = json_get(calls.ptr[i], "arguments");
            put(a, fn, "arguments", jstrs(a, args ? js_stringify(a, args) : S("undefined")));
            put(a, tc, "function", fn);
            json_push(a, tcs, tc);
        }
        put(a, am, "tool_calls", tcs);
    }
    if (preserved) put(a, am, "reasoning_details", preserved);
    if (c->reasoning_content_on_assistant && m->reasoning && !json_get(am, "reasoning_content"))
        put(a, am, "reasoning_content", jlit(a, ""));
    vec_free(&thinking);
    vec_free(&calls);
    vec_free(&nonempty);
    const JValue *ct = json_get(am, "content");
    bool has = ct->kind == J_STRING ? ct->s.len > 0 : ct->kind == J_ARRAY ? ct->arr.len > 0 : false;
    if (!has && !json_get(am, "tool_calls")) return NULL;
    return am;
}

// renderSystemMessageUpdate, for a system message inside the conversation.
static Str system_update_text(Arena *a, const JValue *msg) {
    Buf b = {0};
    const JValue *content = json_get(msg, "content");
    Str text = json_is(content, J_STRING) ? content->s : join_texts(a, content, "\n", false);
    bool any = false;
    if (text.len) {
        buf_addstr(&b, text);
        any = true;
    }
    const JValue *sec = json_get(msg, "sections");
    for (size_t i = 0; json_is(sec, J_OBJECT) && i < sec->obj.len; i++) {
        if (any) buf_adds(&b, "\n\n");
        any = true;
        Str name = sec->obj.items[i].key;
        const JValue *v = sec->obj.items[i].val;
        if (v->kind == J_NULL) buf_printf(&b, "Removed system prompt section \"%.*s\".", (int)name.len, name.ptr);
        else buf_printf(&b, "Updated system prompt section \"%.*s\":\n\n%.*s", (int)name.len, name.ptr, (int)v->s.len, v->s.ptr);
    }
    Str s = {arena_strndup(a, b.ptr ? b.ptr : "", b.len), b.len};
    buf_free(&b);
    return s;
}

static JValue *convert_messages(Arena *a, const CodeModel *m, const Compat *c, const JValue *system, const JValue *messages) {
    JValue *params = jarr(a);
    const char *irole = m->reasoning && c->developer ? "developer" : "system";
    if (system) {
        Str text = code_oai_system_text(a, system);
        if (text.len) json_push(a, params, msg_of(a, irole, jstrs(a, text)));
    }
    VecJ tm = {0};
    transform(a, m, messages, &tm);
    const char *last = NULL;
    for (size_t i = 0; i < tm.len; i++) {
        const JValue *msg = tm.ptr[i];
        const char *role = json_str(json_get(msg, "role"));
        role = role ? role : "";
        if (c->assistant_after_tool_result && last && !strcmp(last, "toolResult") && !strcmp(role, "user"))
            json_push(a, params, msg_of(a, "assistant", jlit(a, "I have processed the tool results.")));
        if (!strcmp(role, "system")) {
            // A later system message: in place when the server takes them, else the caller
            // already folded it into `system` (pi's collapseSystemMessages).
            if (c->mid_system) {
                Str text = system_update_text(a, msg);
                if (text.len) json_push(a, params, msg_of(a, irole, jstrs(a, text)));
            }
        } else if (!strcmp(role, "user")) {
            const JValue *content = json_get(msg, "content");
            if (json_is(content, J_STRING)) json_push(a, params, msg_of(a, "user", (JValue *)content));
            else {
                JValue *parts = jarr(a);
                for (size_t k = 0; k < content->arr.len; k++) {
                    const JValue *b = content->arr.items[k];
                    if (is_type(b, "text")) {
                        if (sget(b, "text").len) push_text_part(a, parts, json_get(b, "text"));
                    }
                }
                if (parts->arr.len == 0) continue;
                json_push(a, params, msg_of(a, "user", parts));
            }
        } else if (!strcmp(role, "assistant")) {
            JValue *am = convert_assistant(a, m, c, msg);
            if (!am) continue;
            json_push(a, params, am);
        } else if (!strcmp(role, "toolResult")) {
            size_t j = i;
            for (; j < tm.len && str_eqz(sget(tm.ptr[j], "role"), "toolResult"); j++) {
                const JValue *tr = tm.ptr[j];
                Str text = join_texts(a, json_get(tr, "content"), "\n", false);
                JValue *o = msg_of(a, "tool", jstrs(a, js_blank(text) ? S("(no tool output)") : text));
                put(a, o, "tool_call_id", json_get(tr, "toolCallId"));
                if (c->tool_result_name && truthy(json_get(tr, "toolName"))) put(a, o, "name", json_get(tr, "toolName"));
                json_push(a, params, o);
            }
            i = j - 1;
            last = "toolResult";
            continue;
        }
        last = role;
    }
    vec_free(&tm);
    return params;
}

// estimateContextTokens over the system message and the conversation.
static int64_t ceil_chars(size_t chars) { return (int64_t)((2 * chars + 6) / 7); } // ceil(chars / 3.5)

static size_t content_chars(const JValue *content) {
    if (json_is(content, J_STRING)) return u16len(content->s);
    size_t n = 0;
    for (size_t i = 0; json_is(content, J_ARRAY) && i < content->arr.len; i++) {
        const JValue *b = content->arr.items[i];
        n += is_type(b, "text") ? u16len(sget(b, "text")) : 4800;
    }
    return n;
}

static int64_t message_tokens(Arena *a, const JValue *msg) {
    const char *role = json_str(json_get(msg, "role"));
    role = role ? role : "";
    if (!strcmp(role, "system")) {
        int64_t t = ceil_chars(u16len(code_oai_system_text(a, msg)));
        const JValue *ta = json_get(msg, "toolsAdded"), *tr = json_get(msg, "toolsRemoved");
        if (json_is(ta, J_ARRAY) && ta->arr.len) t += ceil_chars(u16len(js_stringify(a, ta)));
        if (json_is(tr, J_ARRAY) && tr->arr.len) t += ceil_chars(u16len(js_stringify(a, tr)));
        return t;
    }
    const JValue *content = json_get(msg, "content");
    if (!strcmp(role, "user") || !strcmp(role, "toolResult")) return ceil_chars(content_chars(content));
    size_t chars = 0;
    for (size_t i = 0; json_is(content, J_ARRAY) && i < content->arr.len; i++) {
        const JValue *b = content->arr.items[i];
        if (is_type(b, "text")) chars += u16len(sget(b, "text"));
        else if (is_type(b, "thinking")) chars += u16len(sget(b, "thinking"));
        else {
            const JValue *args = json_get(b, "arguments");
            chars += u16len(sget(b, "name")) + (args ? u16len(js_stringify(a, args)) : 9);
        }
    }
    return ceil_chars(chars);
}

static double usage_tokens(const JValue *u) {
    double t = num_of(json_get(u, "totalTokens"));
    if (t != 0 && t == t) return t;
    return num_of(json_get(u, "input")) + num_of(json_get(u, "output")) + num_of(json_get(u, "cacheRead")) +
           num_of(json_get(u, "cacheWrite"));
}

static double estimate_tokens(Arena *a, const JValue *system, const JValue *messages) {
    VecJ all = {0};
    if (system) vec_push(&all, (JValue *)system);
    for (size_t i = 0; json_is(messages, J_ARRAY) && i < messages->arr.len; i++) vec_push(&all, messages->arr.items[i]);
    double latest = -INFINITY;
    ptrdiff_t at = -1;
    for (size_t i = 0; i < all.len; i++) {
        const JValue *msg = all.ptr[i];
        double ts = num_of(json_get(msg, "timestamp"));
        if (str_eqz(sget(msg, "role"), "assistant")) {
            const char *sr = json_str(json_get(msg, "stopReason"));
            bool bad = sr && (!strcmp(sr, "aborted") || !strcmp(sr, "error"));
            if (ts >= latest && !bad && usage_tokens(json_get(msg, "usage")) > 0) at = (ptrdiff_t)i;
        }
        if (ts > latest) latest = ts;
    }
    double tokens = 0;
    if (at >= 0) tokens = usage_tokens(json_get(all.ptr[at], "usage"));
    for (size_t i = (size_t)(at + 1); i < all.len; i++) tokens += (double)message_tokens(a, all.ptr[i]);
    vec_free(&all);
    return tokens;
}

static double max_tokens_for(Arena *a, const CodeModel *m, const JValue *system, const JValue *messages) {
    double max = (double)m->max_tokens, cw = (double)m->context_window;
    if (cw <= 0) return max > 1 ? max : 1;
    double input = estimate_tokens(a, system, messages);
    double margin = fmax(4096, ceil(input * 0.15));
    double room = cw - input - margin;
    if (room < 32768) room = fmin(32768, cw - input - 4096);
    return fmin(max, fmax(1, room));
}

static JValue *template_values(Arena *a, const char *effort, const JValue *values, int64_t budget) {
    if (!json_is(values, J_OBJECT)) return NULL;
    JValue *r = jobj(a);
    for (size_t i = 0; i < values->obj.len; i++) {
        JValue *v = values->obj.items[i].val, *res = v;
        if (v->kind == J_OBJECT) {
            if (!effort && truthy(json_get(v, "omitWhenOff"))) continue;
            const char *var = json_str(json_get(v, "$var"));
            if (var && !strcmp(var, "thinking.enabled")) res = json_bool(a, effort != NULL);
            else if (var && !strcmp(var, "thinking.budget")) res = budget > 0 ? jnum(a, (double)budget) : NULL;
            else res = effort ? jlit(a, effort) : NULL;
        }
        if (res) json_put(a, r, values->obj.items[i].key, res);
    }
    return r->obj.len ? r : NULL;
}

static bool has_tool_history(const JValue *messages) {
    for (size_t i = 0; json_is(messages, J_ARRAY) && i < messages->arr.len; i++) {
        const JValue *msg = messages->arr.items[i];
        if (str_eqz(sget(msg, "role"), "toolResult")) return true;
        if (str_eqz(sget(msg, "role"), "assistant")) {
            const JValue *ct = json_get(msg, "content");
            for (size_t k = 0; json_is(ct, J_ARRAY) && k < ct->arr.len; k++)
                if (is_type(ct->arr.items[k], "toolCall")) return true;
        }
    }
    return false;
}

// The OpenAI tools from pi's tool declarations (convertTools without grammar and strict).
static JValue *tools_from_system(Arena *a, const JValue *system) {
    const JValue *ta = system ? json_get(system, "toolsAdded") : NULL;
    if (!json_is(ta, J_ARRAY) || ta->arr.len == 0) return NULL;
    JValue *r = jarr(a);
    for (size_t i = 0; i < ta->arr.len; i++) {
        const JValue *t = ta->arr.items[i];
        JValue *o = jobj(a), *fn = jobj(a);
        put(a, o, "type", jlit(a, "function"));
        put(a, fn, "name", json_get(t, "name"));
        put(a, fn, "description", json_get(t, "description"));
        put(a, fn, "parameters", json_get(t, "parameters"));
        put(a, o, "function", fn);
        json_push(a, r, o);
    }
    return r;
}

Err code_oai_body(Arena *a, const CodeModel *m, const JValue *system, const JValue *messages, const JValue *tools, Buf *out) {
    Compat c = get_compat(m);
    const char *effort = effort_of(m);
    JValue *p = jobj(a);
    put(a, p, "model", jlit(a, m->id ? m->id : ""));
    put(a, p, "messages", convert_messages(a, m, &c, system, messages));
    put(a, p, "stream", json_bool(a, true));
    const char *ret = getenv("PI_CACHE_RETENTION");
    if (ret && !strcmp(ret, "long") && c.long_cache) put(a, p, "prompt_cache_retention", jlit(a, "24h"));
    if (c.usage_streaming) {
        JValue *so = jobj(a);
        put(a, so, "include_usage", json_bool(a, true));
        put(a, p, "stream_options", so);
    }
    if (c.store) put(a, p, "store", json_bool(a, false));
    double maxt = max_tokens_for(a, m, system, messages);
    bool has_max = maxt != 0;
    if (has_max) put(a, p, !strcmp(c.max_tokens_field, "max_tokens") ? "max_tokens" : "max_completion_tokens", jnum(a, maxt));
    JValue *tl = tools ? (JValue *)tools : tools_from_system(a, system);
    if (json_is(tl, J_ARRAY) && tl->arr.len) {
        if (c.strict) {
            JValue *copy = jarr(a);
            for (size_t i = 0; i < tl->arr.len; i++) {
                JValue *t = shallow(a, tl->arr.items[i]), *fn = json_get(t, "function");
                if (json_is(fn, J_OBJECT)) {
                    fn = shallow(a, fn);
                    put(a, fn, "strict", json_bool(a, false));
                    put(a, t, "function", fn);
                }
                json_push(a, copy, t);
            }
            tl = copy;
        }
        put(a, p, "tools", tl);
        if (c.zai_tool_stream) put(a, p, "tool_stream", json_bool(a, true));
    } else if (c.tools_with_history && has_tool_history(messages))
        put(a, p, "tools", jarr(a));
    if (c.vllm_priority) put(a, p, "priority", (JValue *)c.vllm_priority);

    const char *budget_field = c.budget_field ? c.budget_field : c.thinking_budget ? "thinking_token_budget" : NULL;
    int64_t budget = 0;
    if (effort && m->reasoning) {
        double ceiling = has_max ? maxt : (double)m->max_tokens;
        double b = fmin((double)thinking_budget(effort), fmax(0, ceiling - 1024));
        budget = (int64_t)b;
    }
    const char *f = c.thinking_format ? c.thinking_format : "openai";
    bool sre = c.reasoning_effort;
    if (!m->reasoning) {
    } else if (!strcmp(f, "zai")) {
        JValue *t = jobj(a);
        put(a, t, "type", jlit(a, effort ? "enabled" : "disabled"));
        if (effort) put(a, t, "clear_thinking", json_bool(a, false));
        put(a, p, "thinking", t);
        if (effort && sre) put(a, p, "reasoning_effort", jlit(a, effort));
    } else if (!strcmp(f, "qwen")) {
        put(a, p, "enable_thinking", json_bool(a, effort != NULL));
        if (effort && sre) put(a, p, "reasoning_effort", jlit(a, effort));
    } else if (!strcmp(f, "qwen-chat-template")) {
        JValue *k = jobj(a);
        put(a, k, "enable_thinking", json_bool(a, effort != NULL));
        put(a, k, "preserve_thinking", json_bool(a, true));
        put(a, p, "chat_template_kwargs", k);
    } else if (!strcmp(f, "chat-template")) {
        JValue *k = template_values(a, effort, c.chat_template_kwargs, budget);
        if (k) put(a, p, "chat_template_kwargs", k);
    } else if (!strcmp(f, "baseten")) {
        JValue *k = template_values(a, effort, c.chat_template_args, budget);
        if (k) put(a, p, "chat_template_args", k);
        if (sre && effort) put(a, p, "reasoning_effort", jlit(a, effort));
    } else if (!strcmp(f, "deepseek")) {
        JValue *t = jobj(a);
        put(a, t, "type", jlit(a, effort ? "enabled" : "disabled"));
        put(a, p, "thinking", t);
        if (effort && sre) put(a, p, "reasoning_effort", jlit(a, effort));
    } else if (!strcmp(f, "openrouter")) {
        JValue *r = jobj(a);
        put(a, r, "effort", jlit(a, effort ? effort : "none"));
        put(a, p, "reasoning", r);
    } else if (!strcmp(f, "ant-ling")) {
        // needs thinkingLevelMap
    } else if (!strcmp(f, "together")) {
        JValue *r = jobj(a);
        put(a, r, "enabled", json_bool(a, effort != NULL));
        put(a, p, "reasoning", r);
        if (effort && sre) put(a, p, "reasoning_effort", jlit(a, effort));
    } else if (!strcmp(f, "string-thinking")) {
        put(a, p, "thinking", jlit(a, effort ? effort : "none"));
    } else if (effort && sre) {
        put(a, p, "reasoning_effort", jlit(a, effort));
    }
    if (budget_field && budget > 0) json_put(a, p, str(budget_field), jnum(a, (double)budget));
    const JValue *orr = m->compat ? json_get(m->compat, "openRouterRouting") : NULL;
    if (truthy(orr)) put(a, p, "provider", (JValue *)orr);
    const JValue *vg = m->compat ? json_get(m->compat, "vercelGatewayRouting") : NULL;
    if (truthy(vg) && (truthy(json_get(vg, "only")) || truthy(json_get(vg, "order")))) {
        JValue *g = jobj(a), *po = jobj(a);
        if (truthy(json_get(vg, "only"))) put(a, g, "only", json_get(vg, "only"));
        if (truthy(json_get(vg, "order"))) put(a, g, "order", json_get(vg, "order"));
        put(a, po, "gateway", g);
        put(a, p, "providerOptions", po);
    }
    code_oai_js_write(out, p);
    return OK;
}

// ---- the stream -------------------------------------------------------------------------------

typedef enum { BK_TEXT, BK_THINKING, BK_TOOL } BlockKind;
typedef struct {
    BlockKind kind;
    JValue *node;
    GStr text;    // text / thinking
    SJP args;     // tool call: partial arguments
    Buf argtext;  // tool call: the raw arguments text
    JValue *empty; // tool call: the {} shown while the arguments are no object yet
    bool has_index;
    int64_t index;
} Block;

struct CodeOaiStream {
    Arena *a;
    const CodeModel *m;
    Compat c;
    const CodeSink *sink;
    uint64_t t0;
    JValue *out, *content;
    VEC(Block *) blocks;
    Block *text_block, *thinking_block;
    JValue *details; // streamed reasoning_details (array) or NULL
    bool has_finish, sentinel, failed, started;
    char *error;
    // SSE decoding
    Buf line;
    bool prev_cr;
    Buf data;
    bool has_data, has_event;
    Buf event;
    // event delivery
    bool busy;
    VEC(JValue *) queue;
};

static void deliver(CodeOaiStream *s, JValue *ev) {
    if (!s->sink || !s->sink->on_event) return;
    if (!s->busy) {
        s->busy = true;
        s->sink->on_event(s->sink->ctx, ev, s->out);
        return;
    }
    vec_push(&s->queue, ev);
}

static void flush(CodeOaiStream *s) {
    for (size_t i = 0; i < s->queue.len; i++)
        if (s->sink && s->sink->on_event) s->sink->on_event(s->sink->ctx, s->queue.ptr[i], s->out);
    s->queue.len = 0;
    s->busy = false;
}

static JValue *ev_new(CodeOaiStream *s, const char *type, ptrdiff_t index) {
    JValue *e = jobj(s->a);
    put(s->a, e, "type", jlit(s->a, type));
    if (index >= 0) put(s->a, e, "contentIndex", jnum(s->a, (double)index));
    return e;
}

static ptrdiff_t index_of(CodeOaiStream *s, Block *b) {
    for (size_t i = 0; i < s->blocks.len; i++)
        if (s->blocks.ptr[i] == b) return (ptrdiff_t)i;
    return -1;
}

static JValue *zero_usage(Arena *a, bool full) {
    JValue *u = jobj(a), *c = jobj(a);
    put(a, u, "input", jnum(a, 0));
    put(a, u, "output", jnum(a, 0));
    put(a, u, "cacheRead", jnum(a, 0));
    put(a, u, "cacheWrite", jnum(a, 0));
    if (full) put(a, u, "reasoning", jnum(a, 0));
    put(a, u, "totalTokens", jnum(a, 0));
    put(a, c, "input", jnum(a, 0));
    put(a, c, "output", jnum(a, 0));
    put(a, c, "cacheRead", jnum(a, 0));
    put(a, c, "cacheWrite", jnum(a, 0));
    put(a, c, "total", jnum(a, 0));
    put(a, u, "cost", c);
    return u;
}

CodeOaiStream *code_oai_stream_new(Arena *a, const CodeModel *m, const CodeEnv *env, const CodeSink *sink) {
    CodeOaiStream *s = ARENA_NEW(a, CodeOaiStream);
    s->a = a;
    s->m = m;
    s->c = get_compat(m);
    s->sink = sink;
    s->t0 = now_ns();
    JValue *o = jobj(a);
    put(a, o, "role", jlit(a, "assistant"));
    s->content = jarr(a);
    put(a, o, "content", s->content);
    put(a, o, "api", jlit(a, "openai-completions"));
    put(a, o, "provider", jlit(a, m->provider ? m->provider : ""));
    put(a, o, "model", jlit(a, m->id ? m->id : ""));
    put(a, o, "usage", zero_usage(a, false));
    put(a, o, "stopReason", jlit(a, "pending"));
    put(a, o, "timestamp", jnum(a, env ? env->now_ms : 0));
    s->out = o;
    return s;
}

void code_oai_stream_start(CodeOaiStream *s) {
    if (s->started) return;
    s->started = true;
    deliver(s, ev_new(s, "start", -1));
    flush(s);
}

static void set_field(CodeOaiStream *s, const char *key, JValue *v) { json_put(s->a, s->out, str(key), v); }

static void fail(CodeOaiStream *s, char *msg) {
    if (s->failed) return;
    s->failed = true;
    s->error = msg;
}

// APIError(undefined, error, undefined): its message; then pi's formatting (no status: the message).
static char *stream_api_error(Arena *a, const JValue *error) {
    const JValue *msg = json_get(error, "message");
    Str text;
    if (json_is(error, J_OBJECT) && truthy(msg)) text = msg->kind == J_STRING ? msg->s : js_stringify(a, msg);
    else if (truthy(error)) text = js_stringify(a, error);
    else text = S("(no status code or body)");
    if (!text.len) text = S("(no status code or body)");
    char *r = arena_strndup(a, text.ptr, text.len);
    const JValue *raw = json_get(json_get(error, "metadata"), "raw");
    if (truthy(raw)) {
        Str rs = raw->kind == J_STRING ? raw->s : is_num(raw) ? js_stringify(a, raw) : raw->kind == J_BOOL ? str(raw->b ? "true" : "false") : S("[object Object]");
        if (str_find(str(r), rs, 0) < 0) r = arena_printf(a, "%s\n%.*s", r, (int)rs.len, rs.ptr);
    }
    return r;
}

static JValue *parse_usage(Arena *a, const JValue *raw) {
    const JValue *ptd = json_get(raw, "prompt_tokens_details");
    double prompt = truthy(json_get(raw, "prompt_tokens")) ? num_of(json_get(raw, "prompt_tokens")) : 0;
    const JValue *cr = nullish(json_get(ptd, "cached_tokens"));
    if (!cr) cr = nullish(json_get(raw, "prompt_cache_hit_tokens"));
    if (!cr) cr = nullish(json_get(raw, "cached_tokens"));
    double cache_read = cr ? num_of(cr) : 0;
    double cache_write = truthy(json_get(ptd, "cache_write_tokens")) ? num_of(json_get(ptd, "cache_write_tokens")) : 0;
    double input = fmax(0, prompt - cache_read - cache_write);
    double output = truthy(json_get(raw, "completion_tokens")) ? num_of(json_get(raw, "completion_tokens")) : 0;
    const JValue *rt = json_get(json_get(raw, "completion_tokens_details"), "reasoning_tokens");
    double reasoning = truthy(rt) ? num_of(rt) : 0;
    JValue *u = jobj(a), *c = jobj(a);
    put(a, u, "input", jnum(a, input));
    put(a, u, "output", jnum(a, output));
    put(a, u, "cacheRead", jnum(a, cache_read));
    put(a, u, "cacheWrite", jnum(a, cache_write));
    put(a, u, "reasoning", jnum(a, reasoning));
    put(a, u, "totalTokens", jnum(a, input + output + cache_read + cache_write));
    // calculateCost with saetta's zero rates
    put(a, c, "input", jnum(a, 0));
    put(a, c, "output", jnum(a, 0));
    put(a, c, "cacheRead", jnum(a, 0));
    put(a, c, "cacheWrite", jnum(a, 0));
    put(a, c, "total", jnum(a, 0));
    put(a, c, "totalFromProvider", json_bool(a, false));
    put(a, u, "cost", c);
    return u;
}

static Block *new_block(CodeOaiStream *s, BlockKind k) {
    Block *b = ARENA_NEW(s->a, Block);
    b->kind = k;
    b->node = jobj(s->a);
    vec_push(&s->blocks, b);
    json_push(s->a, s->content, b->node);
    return b;
}

static Block *ensure_thinking(CodeOaiStream *s, const char *sig) {
    if (s->thinking_block) return s->thinking_block;
    Block *b = new_block(s, BK_THINKING);
    put(s->a, b->node, "type", jlit(s->a, "thinking"));
    JValue *t = json_new(s->a, J_STRING);
    gstr_init(s->a, &b->text, t);
    put(s->a, b->node, "thinking", t);
    put(s->a, b->node, "thinkingSignature", jlit(s->a, sig));
    s->thinking_block = b;
    deliver(s, ev_new(s, "thinking_start", index_of(s, b)));
    return b;
}

static Block *ensure_text(CodeOaiStream *s) {
    if (s->text_block) return s->text_block;
    Block *b = new_block(s, BK_TEXT);
    put(s->a, b->node, "type", jlit(s->a, "text"));
    JValue *t = json_new(s->a, J_STRING);
    gstr_init(s->a, &b->text, t);
    put(s->a, b->node, "text", t);
    s->text_block = b;
    deliver(s, ev_new(s, "text_start", index_of(s, b)));
    return b;
}

static Block *find_tool(CodeOaiStream *s, bool has_index, int64_t index, Str id) {
    if (has_index)
        for (size_t i = 0; i < s->blocks.len; i++)
            if (s->blocks.ptr[i]->kind == BK_TOOL && s->blocks.ptr[i]->has_index && s->blocks.ptr[i]->index == index)
                return s->blocks.ptr[i];
    if (id.len)
        for (size_t i = 0; i < s->blocks.len; i++)
            if (s->blocks.ptr[i]->kind == BK_TOOL && str_eq(sget(s->blocks.ptr[i]->node, "id"), id)) return s->blocks.ptr[i];
    return NULL;
}

static void append_detail(CodeOaiStream *s, JValue *d) {
    Arena *a = s->a;
    if (!s->details) s->details = jarr(a);
    JValue *last = s->details->arr.len ? s->details->arr.items[s->details->arr.len - 1] : NULL;
    const char *t = json_str(json_get(d, "type")), *lt = last ? json_str(json_get(last, "type")) : NULL;
    bool merge_text = lt && !strcmp(t, "reasoning.text") && !strcmp(lt, "reasoning.text");
    bool merge_sum = lt && !strcmp(t, "reasoning.summary") && !strcmp(lt, "reasoning.summary");
    if (merge_text || merge_sum) {
        const char *k = merge_text ? "text" : "summary";
        Str x = sget(last, k), y = sget(d, k);
        json_put(a, last, str(k), jstrs(a, str(arena_printf(a, "%.*s%.*s", (int)x.len, x.ptr, (int)y.len, y.ptr))));
        if (merge_text && !truthy(json_get(last, "signature")) && json_get(d, "signature"))
            json_put(a, last, S("signature"), json_get(d, "signature"));
        if (!nullish(json_get(last, "id")) && json_get(d, "id")) json_put(a, last, S("id"), json_get(d, "id"));
        if (!truthy(json_get(last, "format")) && json_get(d, "format")) json_put(a, last, S("format"), json_get(d, "format"));
        if (!nullish(json_get(last, "index")) && json_get(d, "index")) json_put(a, last, S("index"), json_get(d, "index"));
        return;
    }
    json_push(a, s->details, shallow(a, d));
}

static void tool_delta(CodeOaiStream *s, const JValue *tc) {
    Arena *a = s->a;
    const JValue *idx = json_get(tc, "index"), *fn = json_get(tc, "function"), *custom = json_get(tc, "custom");
    bool has_index = is_num(idx);
    int64_t index = has_index ? (int64_t)num_of(idx) : 0;
    const JValue *idv = json_get(tc, "id");
    Str id = truthy(idv) && idv->kind == J_STRING ? idv->s : S("");
    const JValue *nm = nullish(json_get(fn, "name"));
    if (!nm) nm = nullish(json_get(custom, "name"));
    Str name = json_is(nm, J_STRING) ? nm->s : S("");
    Block *b = find_tool(s, has_index, index, id);
    if (!b) {
        b = new_block(s, BK_TOOL);
        put(a, b->node, "type", jlit(a, "toolCall"));
        put(a, b->node, "id", jstrs(a, id));
        put(a, b->node, "name", jstrs(a, name));
        b->empty = jobj(a);
        put(a, b->node, "arguments", b->empty);
        b->args.a = a;
        b->has_index = has_index;
        b->index = index;
        deliver(s, ev_new(s, "toolcall_start", index_of(s, b)));
    }
    if (has_index && !b->has_index) {
        b->has_index = true;
        b->index = index;
    }
    if (!sget(b->node, "name").len && name.len) json_put(a, b->node, S("name"), jstrs(a, name));
    if (!sget(b->node, "id").len && id.len) json_put(a, b->node, S("id"), jstrs(a, id));
    Str delta = S("");
    const JValue *args = json_get(fn, "arguments");
    if (truthy(args) && args->kind == J_STRING) {
        delta = args->s;
        buf_addstr(&b->argtext, delta);
        sjp_append(&b->args, delta);
        json_put(a, b->node, S("arguments"), sjp_args(&b->args, b->empty));
    }
    JValue *e = ev_new(s, "toolcall_delta", index_of(s, b));
    put(a, e, "delta", jstrs(a, delta));
    deliver(s, e);
}

static void map_stop(CodeOaiStream *s, const JValue *reason) {
    Arena *a = s->a;
    set_field(s, "rawStopReason", (JValue *)reason);
    Str r = reason->kind == J_STRING ? reason->s : js_stringify(a, reason);
    const char *stop = "error";
    char *msg = NULL;
    if (str_eqz(r, "stop") || str_eqz(r, "end")) stop = "stop";
    else if (str_eqz(r, "length")) stop = "length";
    else if (str_eqz(r, "function_call") || str_eqz(r, "tool_calls")) stop = "toolUse";
    else msg = arena_printf(a, "Provider finish_reason: %.*s", (int)r.len, r.ptr);
    set_field(s, "stopReason", jlit(a, stop));
    if (msg) set_field(s, "errorMessage", jstrs(a, str(msg)));
    s->has_finish = true;
}

static void on_chunk(CodeOaiStream *s, const JValue *chunk) {
    Arena *a = s->a;
    if (!json_is(chunk, J_OBJECT)) return;
    const JValue *id = json_get(chunk, "id");
    if (!truthy(json_get(s->out, "responseId")) && truthy(id)) {
        if (obj_index(s->out, "responseId") < s->out->obj.len) set_field(s, "responseId", (JValue *)id);
        else obj_insert_at(a, s->out, obj_index(s->out, "timestamp") + 1, S("responseId"), (JValue *)id);
    }
    const JValue *model = json_get(chunk, "model");
    if (json_is(model, J_STRING) && model->s.len && !str_eqz(model->s, s->m->id ? s->m->id : "") &&
        !truthy(json_get(s->out, "responseModel")))
        set_field(s, "responseModel", (JValue *)model);
    const JValue *usage = json_get(chunk, "usage");
    if (truthy(usage)) set_field(s, "usage", parse_usage(a, usage));
    const JValue *choices = json_get(chunk, "choices");
    const JValue *choice = json_is(choices, J_ARRAY) && choices->arr.len ? choices->arr.items[0] : NULL;
    if (!truthy(choice)) return;
    if (!truthy(usage) && truthy(json_get(choice, "usage"))) set_field(s, "usage", parse_usage(a, json_get(choice, "usage")));
    const JValue *fin = json_get(choice, "finish_reason");
    if (truthy(fin)) map_stop(s, fin);
    const JValue *delta = json_get(choice, "delta");
    if (!truthy(delta)) return;
    static const char *const rf[] = {"reasoning_content", "reasoning", "reasoning_text"};
    for (int k = 0; k < 3; k++) {
        const JValue *v = json_get(delta, rf[k]);
        if (!json_is(v, J_STRING) || !v->s.len) continue;
        const char *sig = s->m->provider && !strcmp(s->m->provider, "opencode-go") && k == 1 ? "reasoning_content" : rf[k];
        Block *b = ensure_thinking(s, sig);
        gstr_add(a, &b->text, v->s);
        JValue *e = ev_new(s, "thinking_delta", index_of(s, b));
        put(a, e, "delta", jstrs(a, v->s));
        deliver(s, e);
        break;
    }
    const JValue *rd = json_get(delta, "reasoning_details");
    for (size_t i = 0; json_is(rd, J_ARRAY) && i < rd->arr.len; i++) {
        if (!valid_detail(rd->arr.items[i])) continue;
        ensure_thinking(s, "");
        append_detail(s, rd->arr.items[i]);
    }
    const JValue *ct = json_get(delta, "content");
    if (json_is(ct, J_STRING) && ct->s.len) {
        Block *b = ensure_text(s);
        gstr_add(a, &b->text, ct->s);
        JValue *e = ev_new(s, "text_delta", index_of(s, b));
        put(a, e, "delta", jstrs(a, ct->s));
        deliver(s, e);
    }
    const JValue *tcs = json_get(delta, "tool_calls");
    if (truthy(tcs) && json_is(tcs, J_ARRAY))
        for (size_t i = 0; i < tcs->arr.len; i++) tool_delta(s, tcs->arr.items[i]);
}

static void on_sse(CodeOaiStream *s) {
    // pi decodes the stream with TextDecoder: ill-formed UTF-8 becomes U+FFFD
    Str data = code_utf8_clean(s->a, buf_str(&s->data)), event = buf_str(&s->event);
    bool has_event = s->has_event;
    if (str_eqz(data, "[DONE]")) {
        s->sentinel = true;
        return;
    }
    char *perr = NULL;
    JValue *v = code_oai_js_parse(s->a, data, &perr);
    if (!v) {
        fail(s, arena_strdup(s->a, "Error reading response: malformed server-sent event JSON."));
        return;
    }
    if (has_event && str_starts(event, S("thread."))) return;
    if (has_event && str_eqz(event, "error")) {
        const JValue *e = nullish(json_get(v, "error"));
        fail(s, stream_api_error(s->a, e ? e : v));
        return;
    }
    if (truthy(v) && truthy(json_get(v, "error"))) {
        fail(s, stream_api_error(s->a, json_get(v, "error")));
        return;
    }
    on_chunk(s, v);
}

static void sse_line(CodeOaiStream *s, Str line) {
    if (line.len && line.ptr[line.len - 1] == '\r') line.len--;
    if (!line.len) {
        if (!(s->has_event && s->event.len) && !s->has_data) return;
        on_sse(s);
        s->has_event = s->has_data = false;
        s->event.len = s->data.len = 0;
        return;
    }
    if (line.ptr[0] == ':') return;
    ptrdiff_t colon = str_find_char(line, ':', 0);
    Str field = colon >= 0 ? str_n(line.ptr, (size_t)colon) : line;
    Str value = colon >= 0 ? str_n(line.ptr + colon + 1, line.len - (size_t)colon - 1) : S("");
    if (value.len && value.ptr[0] == ' ') value = str_n(value.ptr + 1, value.len - 1);
    if (str_eqz(field, "event")) {
        s->event.len = 0;
        buf_addstr(&s->event, value);
        s->has_event = true;
    } else if (str_eqz(field, "data")) {
        if (s->has_data) buf_addc(&s->data, '\n');
        buf_addstr(&s->data, value);
        s->has_data = true;
    }
}

bool code_oai_stream_feed(CodeOaiStream *s, const void *p, size_t n) {
    if (!s->started) code_oai_stream_start(s);
    const char *c = p;
    for (size_t i = 0; i < n && !s->failed && !s->sentinel; i++) {
        if (c[i] == '\n' && s->prev_cr) {
            s->prev_cr = false;
            continue;
        }
        s->prev_cr = c[i] == '\r';
        if (c[i] == '\n' || c[i] == '\r') {
            sse_line(s, buf_str(&s->line));
            s->line.len = 0;
            continue;
        }
        buf_addc(&s->line, c[i]);
    }
    flush(s);
    return !s->failed;
}

static void finish_block(CodeOaiStream *s, Block *b) {
    Arena *a = s->a;
    ptrdiff_t idx = index_of(s, b);
    JValue *e;
    if (b->kind == BK_TEXT) {
        e = ev_new(s, "text_end", idx);
        put(a, e, "content", jstrs(a, b->text.node->s));
    } else if (b->kind == BK_THINKING) {
        if (s->details) json_put(a, b->node, S("thinkingSignature"), jstrs(a, js_stringify(a, s->details)));
        e = ev_new(s, "thinking_end", idx);
        put(a, e, "content", jstrs(a, b->text.node->s));
    } else {
        JValue *args;
        char *err;
        parse_tool_args(a, buf_str(&b->argtext), &args, &err);
        json_put(a, b->node, S("arguments"), args);
        if (err) json_put(a, b->node, S("argumentsError"), jstrs(a, str(err)));
        e = ev_new(s, "toolcall_end", idx);
        put(a, e, "toolCall", b->node);
    }
    deliver(s, e);
}

static void set_duration(CodeOaiStream *s) {
    double ms = round((double)(now_ns() - s->t0) / 1e6);
    set_field(s, "durationMs", jnum(s->a, ms < 0 ? 0 : ms));
}

JValue *code_oai_stream_end(CodeOaiStream *s, const char *error) {
    Arena *a = s->a;
    if (!error && !s->failed) {
        // the line decoder's and the SSE decoder's flush at the end of the body
        if (!s->sentinel) {
            if (s->line.len) {
                sse_line(s, buf_str(&s->line));
                s->line.len = 0;
            }
            if (!s->failed && !s->sentinel) sse_line(s, S(""));
        }
    }
    if (!error && !s->failed) {
        for (size_t i = 0; i < s->blocks.len; i++) finish_block(s, s->blocks.ptr[i]);
        const char *sr = json_str(json_get(s->out, "stopReason"));
        if (!s->has_finish && !s->c.finish_reason) {
            bool tool = false;
            for (size_t i = 0; i < s->blocks.len; i++) tool |= s->blocks.ptr[i]->kind == BK_TOOL;
            set_field(s, "stopReason", jlit(a, tool ? "toolUse" : "stop"));
            sr = tool ? "toolUse" : "stop";
        }
        if (!strcmp(sr, "error")) {
            const char *em = json_str(json_get(s->out, "errorMessage"));
            fail(s, arena_strdup(a, em && *em ? em : "Provider returned an error stop reason"));
        } else if ((s->c.finish_reason && !s->has_finish) || !strcmp(sr, "pending"))
            fail(s, arena_strdup(a, "Stream ended without finish_reason"));
        if (!s->failed) {
            set_duration(s);
            JValue *e = ev_new(s, "done", -1);
            put(a, e, "reason", json_get(s->out, "stopReason"));
            put(a, e, "message", s->out);
            deliver(s, e);
            flush(s);
            return s->out;
        }
    }
    if (error) fail(s, (char *)error);
    if (s->thinking_block && s->details)
        json_put(a, s->thinking_block->node, S("thinkingSignature"), jstrs(a, js_stringify(a, s->details)));
    set_field(s, "stopReason", jlit(a, "error"));
    set_field(s, "errorMessage", jstrs(a, str(s->error)));
    set_duration(s);
    JValue *e = ev_new(s, "error", -1);
    put(a, e, "reason", jlit(a, "error"));
    put(a, e, "error", s->out);
    deliver(s, e);
    flush(s);
    return s->out;
}

// ---- HTTP -------------------------------------------------------------------------------------

char *code_oai_http_error(Arena *a, int status, Str body) {
    // openai client.js makeStatusError(status, safeJSON(errText), errJSON ? undefined : errText)
    JValue *ej = code_oai_js_parse(a, body, NULL);
    const char *emsg = truthy(ej) ? NULL : arena_strndup(a, body.ptr, body.len);
    const JValue *error = json_is(ej, J_OBJECT) ? json_get(ej, "error") : NULL;
    // APIError.makeMessage
    Str msg = {0};
    bool have = false;
    const JValue *em = json_is(error, J_OBJECT) ? json_get(error, "message") : NULL;
    if (truthy(em)) {
        msg = em->kind == J_STRING ? em->s : js_stringify(a, em);
        have = true;
    } else if (truthy(error)) {
        msg = js_stringify(a, error);
        have = true;
    } else if (emsg) {
        msg = str(emsg);
        have = true;
    }
    char *message = have && msg.len ? arena_printf(a, "%d %.*s", status, (int)msg.len, msg.ptr)
                                     : arena_printf(a, "%d status code (no body)", status);
    // normalizeProviderError + formatProviderError
    char *r = message;
    if (json_is(error, J_OBJECT) && error->obj.len) {
        Str b = js_trim(js_stringify(a, error));
        if (b.len) {
            size_t n16 = u16len(b);
            if (n16 > 4000) b = str(arena_printf(a, "%.*s... [truncated %zu chars]", (int)u16off(b, 4000), b.ptr, n16 - 4000));
            if (str_find(str(message), b, 0) < 0) r = arena_printf(a, "%d: %.*s", status, (int)b.len, b.ptr);
        }
    }
    const JValue *raw = json_get(json_get(error, "metadata"), "raw");
    if (truthy(raw)) {
        Str rs = raw->kind == J_STRING ? raw->s : is_num(raw) ? js_stringify(a, raw) : raw->kind == J_BOOL ? str(raw->b ? "true" : "false") : S("[object Object]");
        if (str_find(str(r), rs, 0) < 0) r = arena_printf(a, "%s\n%.*s", r, (int)rs.len, rs.ptr);
    }
    return r;
}

typedef struct {
    CodeOaiStream *s;
    bool ok_status, seen;
    int status; // the last status seen (the error answer's)
    Buf errbody;
    volatile int *abort; // CodeEnv.abort
} HttpCtx;

static int on_http(void *ctx, int status, const void *p, size_t n) {
    HttpCtx *h = ctx;
    h->seen = true;
    h->status = status;
    if (status < 200 || status >= 300) {
        if (h->errbody.len < (1u << 20)) buf_add(&h->errbody, p, n);
        return 0;
    }
    h->ok_status = true;
    if (h->abort && *h->abort) return 1; // Esc / Ctrl-C: stop the transfer (the server stops generating)
    return code_oai_stream_feed(h->s, p, n) ? 0 : 1;
}

// The in-process transport: the server's request path (server_engine_chat) writes the response
// body straight into on_http, as the socket would; p NULL asks whether the caller has left.
static int on_engine(void *ctx, int status, const void *p, size_t n) {
    HttpCtx *h = ctx;
    if (!p) return h->abort && *h->abort;
    return on_http(ctx, status, p, n);
}

static Err complete_in_process(Arena *a, const CodeModel *m, CodeOaiStream *s, const CodeEnv *env, Str body, JValue **assistant) {
    HttpCtx hc = {.s = s, .abort = env ? env->abort : NULL};
    TRY(server_engine_chat(m->engine, body, on_engine, &hc));
    if (s->failed) *assistant = code_oai_stream_end(s, NULL);
    // Ctrl-C: the error path, as over HTTP (a cancelled transfer) and in pi's catch: open blocks
    // are not finished and partial tool arguments are not parsed
    else if (hc.abort && *hc.abort) *assistant = code_oai_stream_end(s, "Request was aborted");
    else if (!hc.ok_status) *assistant = code_oai_stream_end(s, code_oai_http_error(a, hc.status ? hc.status : 500, code_utf8_clean(a, buf_str(&hc.errbody))));
    else {
        code_oai_stream_start(s);
        *assistant = code_oai_stream_end(s, NULL);
    }
    buf_free(&hc.errbody);
    if (hc.abort && *hc.abort) {
        json_put(a, *assistant, S("stopReason"), json_string(a, S("aborted")));
        json_put(a, *assistant, S("errorMessage"), json_string(a, S("Request was aborted")));
    }
    return OK;
}

Err code_oai_complete(Arena *a, const CodeModel *m, const CodeEnv *env, Str body, const CodeSink *sink, JValue **assistant) {
    CodeOaiStream *s = code_oai_stream_new(a, m, env, sink);
    if (m->engine) return complete_in_process(a, m, s, env, body, assistant);
    if (!m->api_key || !*m->api_key) {
        *assistant = code_oai_stream_end(s, arena_printf(a, "No API key for provider: %s", m->provider ? m->provider : ""));
        return OK;
    }
    const char *base = m->base_url ? m->base_url : "";
    size_t bl = strlen(base);
    char *url = arena_printf(a, "%.*s/chat/completions", (int)(bl && base[bl - 1] == '/' ? bl - 1 : bl), base);
    sx_header h[8];
    size_t nh = 0;
    h[nh++] = (sx_header){"Accept", "application/json"};
    h[nh++] = (sx_header){"Content-Type", "application/json"};
    h[nh++] = (sx_header){"Authorization", arena_printf(a, "Bearer %s", m->api_key)};
    h[nh++] = (sx_header){"User-Agent", "saetta-code (darwin; arm64)"};
    HttpCtx hc = {.s = s, .abort = env ? env->abort : NULL};
    int status = 0;
    char err[512] = "";
    // 300 s without a byte: pi's timeout (http-dispatcher.ts, passed as the SDK timeout in sdk.ts)
    int rc = sx_http_cancellable("POST", url, h, nh, body.ptr, body.len, 300, on_http, &hc, &status, err, sizeof err, hc.abort);
    if (s->failed) {
        *assistant = code_oai_stream_end(s, NULL);
    } else if (rc != 0 && hc.ok_status && hc.abort && *hc.abort) {
        // pi: the OpenAI SDK's stream returns quietly on an AbortError (openai-node Stream: "if
        // (isAbortError(e)) return"), so the loop's end finishes the open blocks (thinking_end,
        // text_end, toolcall_end) before "Request was aborted" is thrown
        for (size_t i = 0; i < s->blocks.len; i++) finish_block(s, s->blocks.ptr[i]);
        *assistant = code_oai_stream_end(s, "Request was aborted");
    } else if (rc != 0) {
        const char *msg;
        if (hc.ok_status) msg = arena_printf(a, "terminated: %s", err);
        else if (strstr(err, "timed out")) msg = "Request timed out.";
        else msg = arena_printf(a, "Connection error.: %s", err);
        *assistant = code_oai_stream_end(s, msg);
    } else if (status < 200 || status >= 300) {
        *assistant = code_oai_stream_end(s, code_oai_http_error(a, status, code_utf8_clean(a, buf_str(&hc.errbody))));
    } else {
        code_oai_stream_start(s);
        *assistant = code_oai_stream_end(s, NULL);
    }
    buf_free(&hc.errbody);
    if (hc.abort && *hc.abort) { // pi: stopReason "aborted", errorMessage "Request was aborted"
        json_put(a, *assistant, S("stopReason"), json_string(a, S("aborted")));
        json_put(a, *assistant, S("errorMessage"), json_string(a, S("Request was aborted")));
    }
    return OK;
}
