// Implementation of json.h, following std.json (Scanner, dynamic Value, Stringify).
#include "json.h"

#include <math.h>
#include <stdlib.h>

#define JSON_MAX_DEPTH 1024

// --- builders and accessors --------------------------------------------------

JValue *json_new(Arena *a, JKind k) {
    JValue *v = ARENA_NEW(a, JValue);
    v->kind = k;
    return v;
}

JValue *json_string(Arena *a, Str s) {
    JValue *v = json_new(a, J_STRING);
    v->s = (Str){arena_strndup(a, s.ptr ? s.ptr : "", s.len), s.len};
    return v;
}

JValue *json_bool(Arena *a, bool b) {
    JValue *v = json_new(a, J_BOOL);
    v->b = b;
    return v;
}

JValue *json_int(Arena *a, int64_t i) {
    JValue *v = json_new(a, J_INTEGER);
    v->i = i;
    return v;
}

void json_push(Arena *a, JValue *arr, JValue *v) {
    if (arr->arr.len == arr->arr.cap) {
        size_t nc = arr->arr.cap ? arr->arr.cap * 2 : 4;
        JValue **items = ARENA_ARRAY(a, JValue *, nc);
        if (arr->arr.len) memcpy(items, arr->arr.items, arr->arr.len * sizeof(JValue *));
        arr->arr.items = items;
        arr->arr.cap = nc;
    }
    arr->arr.items[arr->arr.len++] = v;
}

static void obj_append(Arena *a, JValue *obj, Str key, JValue *v) {
    if (obj->obj.len == obj->obj.cap) {
        size_t nc = obj->obj.cap ? obj->obj.cap * 2 : 4;
        JMember *items = ARENA_ARRAY(a, JMember, nc);
        if (obj->obj.len) memcpy(items, obj->obj.items, obj->obj.len * sizeof(JMember));
        obj->obj.items = items;
        obj->obj.cap = nc;
    }
    obj->obj.items[obj->obj.len++] = (JMember){key, v};
}

void json_put(Arena *a, JValue *obj, Str key, JValue *v) {
    for (size_t i = 0; i < obj->obj.len; i++) {
        if (str_eq(obj->obj.items[i].key, key)) {
            obj->obj.items[i].val = v;
            return;
        }
    }
    obj_append(a, obj, key, v);
}

JValue *json_get(const JValue *obj, const char *key) {
    if (!obj || obj->kind != J_OBJECT) return NULL;
    Str k = str(key);
    for (size_t i = 0; i < obj->obj.len; i++)
        if (str_eq(obj->obj.items[i].key, k)) return obj->obj.items[i].val;
    return NULL;
}

const char *json_str(const JValue *v) { return v && v->kind == J_STRING ? v->s.ptr : NULL; }
Str json_strs(const JValue *v) { return v && v->kind == J_STRING ? v->s : S(""); }
bool json_is(const JValue *v, JKind k) { return v && v->kind == k; }

// --- parser ----------------------------------------------------------------

typedef struct {
    const unsigned char *p;
    size_t n, i;
    Arena *a;
    Buf tmp; // scratch for string decoding
} Parser;

static void skip_ws(Parser *ps) {
    while (ps->i < ps->n) {
        unsigned char c = ps->p[ps->i];
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
        ps->i++;
    }
}

static int hexval(unsigned char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static Err read_u4(Parser *ps, uint32_t *out) {
    if (ps->n - ps->i < 4) return "UnexpectedEndOfInput";
    uint32_t v = 0;
    for (int k = 0; k < 4; k++) {
        int h = hexval(ps->p[ps->i + k]);
        if (h < 0) return "SyntaxError";
        v = v << 4 | (uint32_t)h;
    }
    ps->i += 4;
    *out = v;
    return OK;
}

static void put_utf8(Buf *b, uint32_t cp) {
    char u[4];
    size_t n;
    if (cp < 0x80) { u[0] = (char)cp; n = 1; }
    else if (cp < 0x800) { u[0] = (char)(0xC0 | cp >> 6); u[1] = (char)(0x80 | (cp & 0x3F)); n = 2; }
    else if (cp < 0x10000) {
        u[0] = (char)(0xE0 | cp >> 12); u[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        u[2] = (char)(0x80 | (cp & 0x3F)); n = 3;
    } else {
        u[0] = (char)(0xF0 | cp >> 18); u[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        u[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); u[3] = (char)(0x80 | (cp & 0x3F)); n = 4;
    }
    buf_add(b, u, n);
}

// Length of a valid UTF-8 sequence at s (std.json's validation: no overlong forms, no
// surrogates, nothing above U+10FFFF), or 0.
static size_t utf8_len(const unsigned char *s, size_t avail) {
    unsigned char c = s[0];
    unsigned char lo = 0x80, hi = 0xBF;
    size_t n;
    if (c >= 0xC2 && c <= 0xDF) n = 2;
    else if (c == 0xE0) { n = 3; lo = 0xA0; }
    else if ((c >= 0xE1 && c <= 0xEC) || c == 0xEE || c == 0xEF) n = 3;
    else if (c == 0xED) { n = 3; hi = 0x9F; }
    else if (c == 0xF0) { n = 4; lo = 0x90; }
    else if (c >= 0xF1 && c <= 0xF3) n = 4;
    else if (c == 0xF4) { n = 4; hi = 0x8F; }
    else return 0;
    if (avail < n) return (size_t)-1;
    if (s[1] < lo || s[1] > hi) return 0;
    for (size_t k = 2; k < n; k++)
        if (s[k] < 0x80 || s[k] > 0xBF) return 0;
    return n;
}

// Parses a string starting after the opening quote into the arena.
static Err parse_string(Parser *ps, Str *out) {
    ps->tmp.len = 0;
    size_t run = ps->i;
    for (;;) {
        if (ps->i >= ps->n) return "UnexpectedEndOfInput";
        unsigned char c = ps->p[ps->i];
        if (c == '"') {
            buf_add(&ps->tmp, ps->p + run, ps->i - run);
            ps->i++;
            break;
        }
        if (c < 0x20) return "SyntaxError";
        if (c < 0x80 && c != '\\') {
            ps->i++;
            continue;
        }
        if (c >= 0x80) {
            size_t l = utf8_len(ps->p + ps->i, ps->n - ps->i);
            if (l == (size_t)-1) return "UnexpectedEndOfInput";
            if (l == 0) return "SyntaxError";
            ps->i += l;
            continue;
        }
        // backslash
        buf_add(&ps->tmp, ps->p + run, ps->i - run);
        ps->i++;
        if (ps->i >= ps->n) return "UnexpectedEndOfInput";
        unsigned char e = ps->p[ps->i++];
        switch (e) {
        case '"': case '\\': case '/': buf_addc(&ps->tmp, (char)e); break;
        case 'b': buf_addc(&ps->tmp, 0x08); break;
        case 'f': buf_addc(&ps->tmp, 0x0C); break;
        case 'n': buf_addc(&ps->tmp, '\n'); break;
        case 'r': buf_addc(&ps->tmp, '\r'); break;
        case 't': buf_addc(&ps->tmp, '\t'); break;
        case 'u': {
            uint32_t u;
            TRY(read_u4(ps, &u));
            if (u >= 0xDC00 && u <= 0xDFFF) return "SyntaxError"; // lone low surrogate
            if (u >= 0xD800 && u <= 0xDBFF) {
                // A high surrogate must be followed by \u and a low surrogate.
                if (ps->i >= ps->n) return "UnexpectedEndOfInput";
                if (ps->p[ps->i] != '\\') return "SyntaxError";
                ps->i++;
                if (ps->i >= ps->n) return "UnexpectedEndOfInput";
                if (ps->p[ps->i] != 'u') return "SyntaxError";
                ps->i++;
                uint32_t lo;
                TRY(read_u4(ps, &lo));
                if (lo < 0xDC00 || lo > 0xDFFF) return "SyntaxError";
                u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
            }
            put_utf8(&ps->tmp, u);
            break;
        }
        default: return "SyntaxError";
        }
        run = ps->i;
    }
    Str s = buf_str(&ps->tmp);
    *out = (Str){arena_strndup(ps->a, s.ptr, s.len), s.len};
    return OK;
}

static bool is_digit(unsigned char c) { return c >= '0' && c <= '9'; }

// RFC 8259 number grammar; returns the literal.
static Err scan_number(Parser *ps, Str *lit) {
    size_t s = ps->i;
    const unsigned char *p = ps->p;
    size_t n = ps->n, i = ps->i;
#define NEED() do { if (i >= n) return "UnexpectedEndOfInput"; } while (0)
    if (p[i] == '-') { i++; NEED(); }
    if (p[i] == '0') i++;
    else if (p[i] >= '1' && p[i] <= '9') { while (i < n && is_digit(p[i])) i++; }
    else return "SyntaxError";
    if (i < n && p[i] == '.') {
        i++;
        NEED();
        if (!is_digit(p[i])) return "SyntaxError";
        while (i < n && is_digit(p[i])) i++;
    }
    if (i < n && (p[i] == 'e' || p[i] == 'E')) {
        i++;
        NEED();
        if (p[i] == '+' || p[i] == '-') { i++; NEED(); }
        if (!is_digit(p[i])) return "SyntaxError";
        while (i < n && is_digit(p[i])) i++;
    }
#undef NEED
    ps->i = i;
    *lit = (Str){(const char *)p + s, i - s};
    return OK;
}

static JValue *number_value(Parser *ps, Str lit, bool parse_numbers) {
    char *z = arena_strndup(ps->a, lit.ptr, lit.len);
    Str zs = {z, lit.len};
    JValue *v = json_new(ps->a, J_NUMBER_STRING);
    v->s = zs;
    if (!parse_numbers) return v;
    bool intlike = !str_eqz(zs, "-0") && !memchr(z, '.', lit.len) && !memchr(z, 'e', lit.len) &&
                   !memchr(z, 'E', lit.len);
    if (intlike) {
        int64_t i;
        if (!parse_i64(zs, &i)) {
            v->kind = J_INTEGER;
            v->i = i;
        }
        return v; // overflow stays a number string
    }
    double f = strtod(z, NULL);
    if (isfinite(f)) {
        v->kind = J_FLOAT;
        v->f = f;
    }
    return v;
}

static Err parse_literal(Parser *ps, const char *word) {
    size_t l = strlen(word);
    for (size_t k = 0; k < l; k++) {
        if (ps->i + k >= ps->n) return "UnexpectedEndOfInput";
        if (ps->p[ps->i + k] != (unsigned char)word[k]) return "SyntaxError";
    }
    ps->i += l;
    return OK;
}

// std.json's default duplicate_field_behavior for Value is .error.
typedef struct {
    JValue *v;
    Str key;     // pending key while the container is an object
    StrMap seen; // keys of a large object (from 16 members on), for the duplicate check
} Frame;

// Adds key: value to the object of frame f. A key already present is "DuplicateField" at
// this point, as std.json's Value parser reports it (when the duplicate's value is complete).
static Err obj_put_checked(Arena *a, Frame *f, JValue *v) {
    JValue *obj = f->v;
    size_t n = obj->obj.len;
    if (n < 16) {
        for (size_t j = 0; j < n; j++)
            if (str_eq(obj->obj.items[j].key, f->key)) return "DuplicateField";
    } else {
        if (f->seen.len == 0)
            for (size_t j = 0; j < n; j++) strmap_put(&f->seen, obj->obj.items[j].key, NULL);
        if (strmap_has(&f->seen, f->key)) return "DuplicateField";
        strmap_put(&f->seen, f->key, NULL);
    }
    obj_append(a, obj, f->key, v);
    return OK;
}

static Err parse_doc(Parser *ps, bool parse_numbers, Frame *stack, int *hi, JValue **out) {
    int depth = 0;
    for (;;) {
        // Expecting a value.
        skip_ws(ps);
        if (ps->i >= ps->n) return "UnexpectedEndOfInput";
        unsigned char c = ps->p[ps->i];
        JValue *v = NULL;
        if (c == '{' || c == '[') {
            if (depth == JSON_MAX_DEPTH) return "NestingTooDeep";
            ps->i++;
            JValue *cont = json_new(ps->a, c == '{' ? J_OBJECT : J_ARRAY);
            skip_ws(ps);
            if (ps->i >= ps->n) return "UnexpectedEndOfInput";
            if (ps->p[ps->i] == (c == '{' ? '}' : ']')) {
                ps->i++;
                v = cont;
            } else {
                if (depth == *hi) (*hi)++;
                strmap_free(&stack[depth].seen);
                stack[depth++] = (Frame){cont, {0}, {0}};
                if (c == '{') {
                    if (ps->p[ps->i] != '"') return "SyntaxError";
                    ps->i++;
                    TRY(parse_string(ps, &stack[depth - 1].key));
                    skip_ws(ps);
                    if (ps->i >= ps->n) return "UnexpectedEndOfInput";
                    if (ps->p[ps->i] != ':') return "SyntaxError";
                    ps->i++;
                }
                continue;
            }
        } else if (c == '"') {
            ps->i++;
            v = json_new(ps->a, J_STRING);
            TRY(parse_string(ps, &v->s));
        } else if (c == '-' || is_digit(c)) {
            Str lit;
            TRY(scan_number(ps, &lit));
            v = number_value(ps, lit, parse_numbers);
        } else if (c == 't') {
            TRY(parse_literal(ps, "true"));
            v = json_bool(ps->a, true);
        } else if (c == 'f') {
            TRY(parse_literal(ps, "false"));
            v = json_bool(ps->a, false);
        } else if (c == 'n') {
            TRY(parse_literal(ps, "null"));
            v = json_new(ps->a, J_NULL);
        } else {
            return "SyntaxError";
        }
        // A complete value: attach it and close finished containers.
        for (;;) {
            if (depth == 0) {
                skip_ws(ps);
                if (ps->i != ps->n) return "SyntaxError";
                *out = v;
                return OK;
            }
            Frame *f = &stack[depth - 1];
            if (f->v->kind == J_OBJECT) TRY(obj_put_checked(ps->a, f, v));
            else json_push(ps->a, f->v, v);
            skip_ws(ps);
            if (ps->i >= ps->n) return "UnexpectedEndOfInput";
            unsigned char d = ps->p[ps->i++];
            if (d == ',') {
                if (f->v->kind == J_OBJECT) {
                    skip_ws(ps);
                    if (ps->i >= ps->n) return "UnexpectedEndOfInput";
                    if (ps->p[ps->i] != '"') return "SyntaxError";
                    ps->i++;
                    TRY(parse_string(ps, &f->key));
                    skip_ws(ps);
                    if (ps->i >= ps->n) return "UnexpectedEndOfInput";
                    if (ps->p[ps->i] != ':') return "SyntaxError";
                    ps->i++;
                }
                break; // next value
            }
            if (d != (f->v->kind == J_OBJECT ? '}' : ']')) return "SyntaxError";
            v = f->v;
            depth--;
        }
    }
}

Err json_parse(Arena *a, Str text, bool parse_numbers, JValue **out) {
    Parser ps = {(const unsigned char *)text.ptr, text.len, 0, a, {0}};
    Frame *stack = xcalloc(JSON_MAX_DEPTH, sizeof(Frame));
    int hi = 0;
    Err e = parse_doc(&ps, parse_numbers, stack, &hi, out);
    for (int i = 0; i < hi; i++) strmap_free(&stack[i].seen);
    free(stack);
    buf_free(&ps.tmp);
    return e;
}

// --- writer ----------------------------------------------------------------

void json_write_string(Buf *out, Str s) {
    static const char hx[] = "0123456789abcdef";
    buf_reserve(out, s.len + 2);
    buf_addc(out, '"');
    size_t run = 0;
    for (size_t i = 0; i < s.len; i++) {
        unsigned char c = (unsigned char)s.ptr[i];
        if (c >= 0x20 && c != '"' && c != '\\') continue;
        buf_add(out, s.ptr + run, i - run);
        run = i + 1;
        switch (c) {
        case '"': buf_adds(out, "\\\""); break;
        case '\\': buf_adds(out, "\\\\"); break;
        case 0x08: buf_adds(out, "\\b"); break;
        case 0x0C: buf_adds(out, "\\f"); break;
        case '\n': buf_adds(out, "\\n"); break;
        case '\r': buf_adds(out, "\\r"); break;
        case '\t': buf_adds(out, "\\t"); break;
        default: {
            char u[6] = {'\\', 'u', '0', '0', hx[c >> 4], hx[c & 15]};
            buf_add(out, u, 6);
        }
        }
    }
    buf_add(out, s.ptr + run, s.len - run);
    buf_addc(out, '"');
}

void json_write(Buf *out, const JValue *v) {
    switch (v->kind) {
    case J_NULL: buf_adds(out, "null"); break;
    case J_BOOL: buf_adds(out, v->b ? "true" : "false"); break;
    case J_INTEGER: buf_printf(out, "%lld", (long long)v->i); break;
    case J_FLOAT: {
        char tmp[400];
        int n = fmt_f64(tmp, sizeof tmp, v->f);
        // Stringify quotes a float that does not equal itself as f64 (NaN).
        if (isnan(v->f)) buf_addc(out, '"');
        buf_add(out, tmp, (size_t)n);
        if (isnan(v->f)) buf_addc(out, '"');
        break;
    }
    case J_NUMBER_STRING: buf_addstr(out, v->s); break;
    case J_STRING: json_write_string(out, v->s); break;
    case J_ARRAY:
        buf_addc(out, '[');
        for (size_t i = 0; i < v->arr.len; i++) {
            if (i) buf_addc(out, ',');
            json_write(out, v->arr.items[i]);
        }
        buf_addc(out, ']');
        break;
    case J_OBJECT:
        buf_addc(out, '{');
        for (size_t i = 0; i < v->obj.len; i++) {
            if (i) buf_addc(out, ',');
            json_write_string(out, v->obj.items[i].key);
            buf_addc(out, ':');
            json_write(out, v->obj.items[i].val);
        }
        buf_addc(out, '}');
        break;
    }
}
