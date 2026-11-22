// Saetta Code tools (see code.h): read, bash, edit, write, grep, find, ls, and the per-call
// path of pi's agent loop around them (unknown tool, argument preparation, validation and
// coercion, thrown errors, the toolResult message).
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at
// v1.5.2, b4afddd): packages/coding-agent/src/core/tools/{read,bash,edit,edit-diff,write,grep,
// find,ls,truncate,path-utils,output-accumulator}.ts, coding-agent/src/utils/{paths,shell,
// child-process,output-files,mime,tools-manager}.ts, agent/src/agent-loop.ts,
// ai/src/utils/validation.ts and schema/src/{coerce,explain}.ts (whose conversions follow
// TypeBox 1.3.27, MIT, Copyright (c) 2017-2025 Haydn Paterson).
// The line diff is a port of jsdiff 8.0.4 (BSD-3-Clause, Copyright (c) 2009-2015 Kevin Decker):
// src/diff/base.js, src/diff/line.js, src/patch/create.js.
//
// Strings are UTF-8 bytes where pi has UTF-16 strings: offsets and lengths only ever slice the
// same text, so the results agree; the few places where pi shows a UTF-16 length (grep's 500
// character lines) count UTF-16 units. Decoding follows TextDecoder (invalid bytes become
// U+FFFD), so byte counts match Buffer.byteLength of pi's decoded text.
#include "code.h"
#include "code_mcp.h"
#include "metal/shim.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <sys/event.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

enum { MAX_LINES = 2000, MAX_BYTES = 50 * 1024, GREP_MAX_LINE = 500 };
#define MAX_SAFE_INTEGER 9007199254740991.0

// ---------------------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------------------

static Str take(Arena *a, Buf *b) { // the buffer's text, copied into the arena; frees b
    Str s = {arena_strndup(a, b->ptr ? b->ptr : "", b->len), b->len};
    buf_free(b);
    return s;
}

static Str sfmt(Arena *a, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static Str sfmt(Arena *a, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char *s = arena_vprintf(a, fmt, ap);
    va_end(ap);
    return str(s);
}

static ptrdiff_t sfind(Str hay, Str needle, size_t pos) {
    if (pos > hay.len) return -1;
    if (needle.len == 0) return (ptrdiff_t)pos;
    const char *p = memmem(hay.ptr + pos, hay.len - pos, needle.ptr, needle.len);
    return p ? p - hay.ptr : -1;
}

static ptrdiff_t schr(Str s, char c, size_t pos) {
    if (pos >= s.len) return -1;
    const char *p = memchr(s.ptr + pos, c, s.len - pos);
    return p ? p - s.ptr : -1;
}

// lastIndexOf(c, pos): the last c at an index <= pos
static ptrdiff_t srchr(Str s, char c, ptrdiff_t pos) {
    if (s.len == 0) return -1;
    if (pos >= (ptrdiff_t)s.len) pos = (ptrdiff_t)s.len - 1;
    for (ptrdiff_t i = pos; i >= 0; i--)
        if (s.ptr[i] == c) return i;
    return -1;
}

static size_t count_char(Str s, char c, size_t from, size_t to) {
    size_t n = 0;
    for (ptrdiff_t i = schr(s, c, from); i != -1 && (size_t)i < to; i = schr(s, c, (size_t)i + 1)) n++;
    return n;
}

// Splits on '\n' (text.split("\n")): n + 1 pieces for n line feeds.
static void split_lines(Str s, VecStr *out) {
    size_t start = 0;
    for (;;) {
        ptrdiff_t nl = schr(s, '\n', start);
        if (nl == -1) {
            vec_push(out, str_n(s.ptr + start, s.len - start));
            return;
        }
        vec_push(out, str_n(s.ptr + start, (size_t)nl - start));
        start = (size_t)nl + 1;
    }
}

// The text from the start of piece i to the end of piece j - 1 of a split (pieces of one string).
static Str span(const VecStr *v, size_t i, size_t j) {
    if (j <= i) return (Str){"", 0};
    return str_n(v->ptr[i].ptr, (size_t)(v->ptr[j - 1].ptr + v->ptr[j - 1].len - v->ptr[i].ptr));
}

// JavaScript's Number.prototype.toString().
static const char *js_num(Arena *a, double x) {
    if (isnan(x)) return "NaN";
    if (isinf(x)) return x > 0 ? "Infinity" : "-Infinity";
    if (x == 0) return "0";
    double ax = fabs(x);
    if (ax >= 1e-6 && ax < 1e21) {
        char t[400];
        fmt_f64(t, sizeof t, x);
        return arena_strdup(a, t);
    }
    char t[64];
    for (int p = 1; p <= 17; p++) {
        snprintf(t, sizeof t, "%.*e", p - 1, x);
        if (strtod(t, NULL) == x) break;
    }
    char *e = strchr(t, 'e');
    *e = 0;
    int ex = atoi(e + 1);
    return arena_printf(a, "%se%c%d", t, ex < 0 ? '-' : '+', abs(ex));
}

// truncate.ts formatSize: toFixed(1) rounds half up on these exact binary fractions
static const char *format_size(Arena *a, double bytes) {
    if (bytes < 1024) return arena_printf(a, "%sB", js_num(a, bytes));
    double unit = bytes < 1024 * 1024 ? 1024 : 1024 * 1024;
    double n = floor(bytes / unit * 10 + 0.5);
    return arena_printf(a, "%.0f.%.0f%s", floor(n / 10), fmod(n, 10), unit == 1024 ? "KB" : "MB");
}

// ---- UTF-8 (TextDecoder, WHATWG decoder with replacement) ----

typedef struct {
    uint32_t cp;
    int needed, seen;
    uint8_t lo, hi;
    bool ignore_bom, started;
} Utf8Dec;

static void u8_put(Buf *o, uint32_t cp) {
    char b[4];
    size_t n;
    if (cp < 0x80) { b[0] = (char)cp; n = 1; }
    else if (cp < 0x800) { b[0] = (char)(0xc0 | (cp >> 6)); b[1] = (char)(0x80 | (cp & 0x3f)); n = 2; }
    else if (cp < 0x10000) {
        b[0] = (char)(0xe0 | (cp >> 12)); b[1] = (char)(0x80 | ((cp >> 6) & 0x3f)); b[2] = (char)(0x80 | (cp & 0x3f)); n = 3;
    } else {
        b[0] = (char)(0xf0 | (cp >> 18)); b[1] = (char)(0x80 | ((cp >> 12) & 0x3f));
        b[2] = (char)(0x80 | ((cp >> 6) & 0x3f)); b[3] = (char)(0x80 | (cp & 0x3f)); n = 4;
    }
    buf_add(o, b, n);
}

static void u8_emit(Utf8Dec *d, Buf *o, uint32_t cp) {
    bool first = !d->started;
    d->started = true;
    if (first && cp == 0xfeff && !d->ignore_bom) return;
    u8_put(o, cp);
}

static void u8_decode(Utf8Dec *d, Buf *o, const void *data, size_t n, bool flush) {
    const uint8_t *p = data;
    if (d->lo == 0) { d->lo = 0x80; d->hi = 0xbf; }
    for (size_t i = 0; i < n; i++) {
        uint8_t b = p[i];
        if (d->needed == 0) {
            if (b < 0x80) {
                if (d->started) { // fast path for ASCII runs
                    size_t j = i;
                    while (j < n && p[j] < 0x80) j++;
                    buf_add(o, p + i, j - i);
                    i = j - 1;
                } else u8_emit(d, o, b);
            } else if (b >= 0xc2 && b <= 0xdf) { d->needed = 1; d->cp = b & 0x1f; }
            else if (b >= 0xe0 && b <= 0xef) {
                if (b == 0xe0) d->lo = 0xa0;
                if (b == 0xed) d->hi = 0x9f;
                d->needed = 2; d->cp = b & 0xf;
            } else if (b >= 0xf0 && b <= 0xf4) {
                if (b == 0xf0) d->lo = 0x90;
                if (b == 0xf4) d->hi = 0x8f;
                d->needed = 3; d->cp = b & 7;
            } else u8_emit(d, o, 0xfffd);
            continue;
        }
        if (b < d->lo || b > d->hi) {
            d->cp = 0; d->needed = d->seen = 0; d->lo = 0x80; d->hi = 0xbf;
            u8_emit(d, o, 0xfffd);
            i--;
            continue;
        }
        d->lo = 0x80; d->hi = 0xbf;
        d->cp = (d->cp << 6) | (b & 0x3f);
        if (++d->seen == d->needed) {
            u8_emit(d, o, d->cp);
            d->cp = 0; d->needed = d->seen = 0;
        }
    }
    if (flush && d->needed) {
        d->cp = 0; d->needed = d->seen = 0; d->lo = 0x80; d->hi = 0xbf;
        u8_emit(d, o, 0xfffd);
    }
}

static Str u8_text(Arena *a, const void *p, size_t n, bool ignore_bom) {
    Utf8Dec d = {.ignore_bom = ignore_bom};
    Buf b = {0};
    buf_reserve(&b, n + 1);
    u8_decode(&d, &b, p, n, true);
    return take(a, &b);
}

// The code point at s[*i], advancing *i. Bounded by s.len: an invalid or truncated sequence is
// U+FFFD and consumes its maximal valid prefix (at least one byte), as TextDecoder does.
static uint32_t u8_next(Str s, size_t *i) {
    const uint8_t *p = (const uint8_t *)s.ptr;
    size_t at = *i;
    uint8_t b = p[at];
    if (b < 0x80) { *i = at + 1; return b; }
    int n;
    uint8_t lo = 0x80, hi = 0xbf;
    if (b >= 0xc2 && b <= 0xdf) n = 1;
    else if (b >= 0xe0 && b <= 0xef) { n = 2; if (b == 0xe0) lo = 0xa0; else if (b == 0xed) hi = 0x9f; }
    else if (b >= 0xf0 && b <= 0xf4) { n = 3; if (b == 0xf0) lo = 0x90; else if (b == 0xf4) hi = 0x8f; }
    else { *i = at + 1; return 0xfffd; }
    uint32_t cp = b & (0x3f >> n);
    for (int k = 1; k <= n; k++) {
        size_t q = at + (size_t)k;
        if (q >= s.len || p[q] < lo || p[q] > hi) { *i = q; return 0xfffd; }
        cp = (cp << 6) | (p[q] & 0x3f);
        lo = 0x80, hi = 0xbf;
    }
    *i = at + (size_t)n + 1;
    return cp;
}

static size_t utf16_len(Str s) {
    size_t n = 0;
    for (size_t i = 0; i < s.len;) n += u8_next(s, &i) >= 0x10000 ? 2 : 1;
    return n;
}

// sliceHead: at most max UTF-16 units, never half a surrogate pair
static Str utf16_head(Str s, size_t max) {
    size_t units = 0, i = 0;
    while (i < s.len) {
        size_t j = i;
        size_t w = u8_next(s, &j) >= 0x10000 ? 2 : 1;
        if (units + w > max) break;
        units += w;
        i = j;
    }
    return str_n(s.ptr, i);
}

// Foundation normalization (shim), on valid UTF-8
static Str normalize_form(Arena *a, Str s, bool nfkc) {
    bool ascii = true;
    for (size_t i = 0; i < s.len; i++)
        if ((uint8_t)s.ptr[i] >= 0x80) { ascii = false; break; }
    if (ascii || s.len == 0) return s;
    size_t n = nfkc ? sx_nfkc(s.ptr, s.len, NULL, 0) : sx_nfd(s.ptr, s.len, NULL, 0);
    if (n == (size_t)-1) return s;
    char *o = arena_alloc(a, n + 1);
    if (nfkc) sx_nfkc(s.ptr, s.len, o, n);
    else sx_nfd(s.ptr, s.len, o, n);
    return str_n(o, n);
}

// ---- JSON ----

static void jput(Arena *a, JValue *o, const char *k, JValue *v) { json_put(a, o, str(k), v); }
static JValue *jstr(Arena *a, Str s) { return json_string(a, s); }
static JValue *jnum(Arena *a, double x) {
    JValue *v = json_new(a, J_NUMBER_STRING);
    v->s = str(js_num(a, x));
    return v;
}

static bool jnumber(const JValue *v, double *out) {
    if (!v) return false;
    switch (v->kind) {
    case J_INTEGER: *out = (double)v->i; return true;
    case J_FLOAT: *out = v->f; return true;
    case J_NUMBER_STRING: {
        char t[512];
        snprintf(t, sizeof t, "%.*s", (int)MIN(v->s.len, sizeof t - 1), v->s.ptr);
        *out = strtod(t, NULL);
        return true;
    }
    default: return false;
    }
}

static JValue *text_block(Arena *a, Str text) {
    JValue *b = json_new(a, J_OBJECT);
    jput(a, b, "type", jstr(a, S("text")));
    jput(a, b, "text", jstr(a, text));
    return b;
}

static JValue *text_content(Arena *a, Str text) {
    JValue *c = json_new(a, J_ARRAY);
    json_push(a, c, text_block(a, text));
    return c;
}

static void put_number_js(Buf *b, const JValue *v) {
    Arena t = {0};
    double x = 0;
    if (v->kind == J_INTEGER) buf_printf(b, "%lld", (long long)v->i);
    else {
        jnumber(v, &x);
        buf_adds(b, isfinite(x) ? js_num(&t, x) : "null");
    }
    arena_reset(&t);
}

// JSON.stringify(v, null, 2)
static void json_pretty(Buf *b, const JValue *v, int ind) {
    switch (v->kind) {
    case J_NULL: buf_adds(b, "null"); break;
    case J_BOOL: buf_adds(b, v->b ? "true" : "false"); break;
    case J_INTEGER: case J_FLOAT: case J_NUMBER_STRING: put_number_js(b, v); break;
    case J_STRING: json_write_string(b, v->s); break;
    case J_ARRAY:
        if (!v->arr.len) { buf_adds(b, "[]"); break; }
        buf_adds(b, "[\n");
        for (size_t i = 0; i < v->arr.len; i++) {
            buf_printf(b, "%*s", ind + 2, "");
            json_pretty(b, v->arr.items[i], ind + 2);
            buf_adds(b, i + 1 < v->arr.len ? ",\n" : "\n");
        }
        buf_printf(b, "%*s]", ind, "");
        break;
    case J_OBJECT:
        if (!v->obj.len) { buf_adds(b, "{}"); break; }
        buf_adds(b, "{\n");
        for (size_t i = 0; i < v->obj.len; i++) {
            buf_printf(b, "%*s", ind + 2, "");
            json_write_string(b, v->obj.items[i].key);
            buf_adds(b, ": ");
            json_pretty(b, v->obj.items[i].val, ind + 2);
            buf_adds(b, i + 1 < v->obj.len ? ",\n" : "\n");
        }
        buf_printf(b, "%*s}", ind, "");
        break;
    }
}

static JValue *obj_copy(Arena *a, const JValue *o) {
    JValue *c = json_new(a, J_OBJECT);
    for (size_t i = 0; i < o->obj.len; i++) json_put(a, c, o->obj.items[i].key, o->obj.items[i].val);
    return c;
}

static JValue *arr_copy(Arena *a, const JValue *o) {
    JValue *c = json_new(a, J_ARRAY);
    for (size_t i = 0; i < o->arr.len; i++) json_push(a, c, o->arr.items[i]);
    return c;
}

// ---------------------------------------------------------------------------------------
// Node's error messages (libuv names and texts)
// ---------------------------------------------------------------------------------------

static const char *errno_code(int e) {
    switch (e) {
    case ENOENT: return "ENOENT";
    case EACCES: return "EACCES";
    case EISDIR: return "EISDIR";
    case ENOTDIR: return "ENOTDIR";
    case EEXIST: return "EEXIST";
    case EPERM: return "EPERM";
    case ELOOP: return "ELOOP";
    case ENAMETOOLONG: return "ENAMETOOLONG";
    case EROFS: return "EROFS";
    case ENOSPC: return "ENOSPC";
    case EMFILE: return "EMFILE";
    case ENFILE: return "ENFILE";
    case EINVAL: return "EINVAL";
    case EBUSY: return "EBUSY";
    case ETXTBSY: return "ETXTBSY";
    case EFBIG: return "EFBIG";
    case EIO: return "EIO";
    case ENOTEMPTY: return "ENOTEMPTY";
    case EAGAIN: return "EAGAIN";
    case EBADF: return "EBADF";
    case ENOMEM: return "ENOMEM";
    case ENXIO: return "ENXIO";
    case ENODEV: return "ENODEV";
    case EXDEV: return "EXDEV";
    case EDQUOT: return "EDQUOT";
    case E2BIG: return "E2BIG";
    default: return "UNKNOWN";
    }
}

static const char *errno_desc(int e) {
    switch (e) {
    case ENOENT: return "no such file or directory";
    case EACCES: return "permission denied";
    case EISDIR: return "illegal operation on a directory";
    case ENOTDIR: return "not a directory";
    case EEXIST: return "file already exists";
    case EPERM: return "operation not permitted";
    case ELOOP: return "too many symbolic links encountered";
    case ENAMETOOLONG: return "name too long";
    case EROFS: return "read-only file system";
    case ENOSPC: return "no space left on device";
    case EMFILE: return "too many open files";
    case ENFILE: return "file table overflow";
    case EINVAL: return "invalid argument";
    case EBUSY: return "resource busy or locked";
    case ETXTBSY: return "text file is busy";
    case EFBIG: return "file too large";
    case EIO: return "i/o error";
    case ENOTEMPTY: return "directory not empty";
    case EAGAIN: return "resource temporarily unavailable";
    case EBADF: return "bad file descriptor";
    case ENOMEM: return "not enough memory";
    case ENXIO: return "no such device or address";
    case ENODEV: return "no such device";
    case EXDEV: return "cross-device link not permitted";
    case EDQUOT: return "disk quota exceeded";
    case E2BIG: return "argument list too long";
    default: return "unknown error";
    }
}

// "ENOENT: no such file or directory, access '/x'"
static const char *node_err(Arena *a, int e, const char *syscall, const char *path) {
    if (path) return arena_printf(a, "%s: %s, %s '%s'", errno_code(e), errno_desc(e), syscall, path);
    return arena_printf(a, "%s: %s, %s", errno_code(e), errno_desc(e), syscall);
}

// ---------------------------------------------------------------------------------------
// paths (node:path posix, utils/paths.ts, tools/path-utils.ts)
// ---------------------------------------------------------------------------------------

// path.resolve(base, p) for an absolute base (or p absolute): normalized, no trailing slash
static char *path_resolve(Arena *a, const char *base, Str p) {
    Buf in = {0};
    if (!(p.len && p.ptr[0] == '/')) {
        buf_adds(&in, base);
        buf_addc(&in, '/');
    }
    buf_addstr(&in, p);
    VecStr segs = {0};
    Str s = buf_str(&in);
    size_t i = 0;
    while (i <= s.len) {
        size_t j = i;
        while (j < s.len && s.ptr[j] != '/') j++;
        Str seg = str_n(s.ptr + i, j - i);
        if (seg.len == 0 || str_eqz(seg, ".")) {
        } else if (str_eqz(seg, "..")) {
            if (segs.len) segs.len--;
        } else vec_push(&segs, seg);
        i = j + 1;
    }
    Buf o = {0};
    for (size_t k = 0; k < segs.len; k++) {
        buf_addc(&o, '/');
        buf_addstr(&o, segs.ptr[k]);
    }
    if (!o.len) buf_addc(&o, '/');
    vec_free(&segs);
    Str r = take(a, &o);
    buf_free(&in);
    return (char *)r.ptr;
}

// path.relative(from, to) for absolute normalized paths
static char *path_relative(Arena *a, const char *from, const char *to) {
    from = path_resolve(a, "/", str(from));
    to = path_resolve(a, "/", str(to));
    if (!strcmp(from, to)) return "";
    VecStr f = {0}, t = {0};
    for (const char *p = from + 1; *p;) {
        const char *q = strchr(p, '/');
        size_t n = q ? (size_t)(q - p) : strlen(p);
        vec_push(&f, str_n(p, n));
        p += n + (q ? 1 : 0);
    }
    for (const char *p = to + 1; *p;) {
        const char *q = strchr(p, '/');
        size_t n = q ? (size_t)(q - p) : strlen(p);
        vec_push(&t, str_n(p, n));
        p += n + (q ? 1 : 0);
    }
    size_t c = 0;
    while (c < f.len && c < t.len && str_eq(f.ptr[c], t.ptr[c])) c++;
    Buf o = {0};
    for (size_t k = c; k < f.len; k++) buf_adds(&o, o.len ? "/.." : "..");
    for (size_t k = c; k < t.len; k++) {
        if (o.len) buf_addc(&o, '/');
        buf_addstr(&o, t.ptr[k]);
    }
    vec_free(&f);
    vec_free(&t);
    return (char *)take(a, &o).ptr;
}

static char *path_dirname(Arena *a, const char *p) {
    const char *s = strrchr(p, '/');
    if (!s) return ".";
    if (s == p) return "/";
    return arena_strndup(a, p, (size_t)(s - p));
}

static Str path_basename(Str p) {
    while (p.len > 1 && p.ptr[p.len - 1] == '/') p.len--;
    ptrdiff_t s = srchr(p, '/', (ptrdiff_t)p.len - 1);
    return s < 0 ? p : str_n(p.ptr + s + 1, p.len - (size_t)s - 1);
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// normalizePath(input, {normalizeUnicodeSpaces, stripAtPrefix}) followed by resolvePath
static char *resolve_to_cwd(Arena *a, const CodeEnv *env, Str p) {
    Buf b = {0};
    for (size_t i = 0; i < p.len;) {
        size_t j = i;
        uint32_t cp = u8_next(p, &j);
        if (cp == 0xa0 || (cp >= 0x2000 && cp <= 0x200a) || cp == 0x202f || cp == 0x205f || cp == 0x3000) buf_addc(&b, ' ');
        else buf_add(&b, p.ptr + i, j - i);
        i = j;
    }
    Str s = take(a, &b);
    if (s.len && s.ptr[0] == '@') s = str_n(s.ptr + 1, s.len - 1);
    if (str_eqz(s, "~")) return path_resolve(a, "/", str(env->home));
    if (str_starts(s, S("~/"))) return path_resolve(a, env->home, str_n(s.ptr + 2, s.len - 2));
    if (str_starts(s, S("file://"))) {
        Str r = str_n(s.ptr + 7, s.len - 7);
        if (str_starts(r, S("localhost/"))) r = str_n(r.ptr + 9, r.len - 9);
        Buf d = {0};
        for (size_t i = 0; i < r.len; i++) {
            if (r.ptr[i] == '%' && i + 2 < r.len + 0 && hexval(r.ptr[i + 1]) >= 0 && hexval(r.ptr[i + 2]) >= 0) {
                buf_addc(&d, (char)(hexval(r.ptr[i + 1]) * 16 + hexval(r.ptr[i + 2])));
                i += 2;
            } else buf_addc(&d, r.ptr[i]);
        }
        s = take(a, &d);
    }
    return path_resolve(a, env->cwd, s);
}

static bool exists(const char *p) { return access(p, F_OK) == 0; }

// resolveReadPathAsync: the path, or a macOS spelling of it that exists
static char *resolve_read_path(Arena *a, const CodeEnv *env, Str p) {
    char *r = resolve_to_cwd(a, env, p);
    if (exists(r)) return r;
    Str rs = str(r);
    // " AM." / " PM." (any case) with a narrow no-break space
    Buf am = {0};
    for (size_t i = 0; i < rs.len; i++) {
        char c1 = i + 3 < rs.len ? (char)(rs.ptr[i + 1] | 0x20) : 0;
        if (rs.ptr[i] == ' ' && (c1 == 'a' || c1 == 'p') && (rs.ptr[i + 2] | 0x20) == 'm' && rs.ptr[i + 3] == '.') {
            buf_adds(&am, "\xe2\x80\xaf");
            continue;
        }
        buf_addc(&am, rs.ptr[i]);
    }
    Str ams = take(a, &am);
    if (!str_eq(ams, rs) && exists(ams.ptr)) return (char *)ams.ptr;
    Str nfd = normalize_form(a, rs, false);
    nfd = (Str){str_z(a, nfd), nfd.len};
    if (!str_eq(nfd, rs) && exists(nfd.ptr)) return (char *)nfd.ptr;
    Buf cq = {0}, ncq = {0};
    for (size_t i = 0; i < rs.len; i++) {
        if (rs.ptr[i] == '\'') buf_adds(&cq, "\xe2\x80\x99");
        else buf_addc(&cq, rs.ptr[i]);
    }
    for (size_t i = 0; i < nfd.len; i++) {
        if (nfd.ptr[i] == '\'') buf_adds(&ncq, "\xe2\x80\x99");
        else buf_addc(&ncq, nfd.ptr[i]);
    }
    Str c = take(a, &cq), nc = take(a, &ncq);
    if (!str_eq(c, rs) && exists(c.ptr)) return (char *)c.ptr;
    if (!str_eq(nc, rs) && exists(nc.ptr)) return (char *)nc.ptr;
    return r;
}

// ---------------------------------------------------------------------------------------
// truncation (truncate.ts)
// ---------------------------------------------------------------------------------------

typedef struct {
    Str content;
    bool truncated;
    const char *by; // "lines", "bytes" or NULL
    double total_lines, total_bytes, output_lines, output_bytes;
    bool last_partial, first_exceeds;
    double max_lines, max_bytes;
} Trunc;

static JValue *trunc_json(Arena *a, const Trunc *t) {
    JValue *o = json_new(a, J_OBJECT);
    jput(a, o, "content", jstr(a, t->content));
    jput(a, o, "truncated", json_bool(a, t->truncated));
    jput(a, o, "truncatedBy", t->by ? jstr(a, str(t->by)) : json_new(a, J_NULL));
    jput(a, o, "totalLines", jnum(a, t->total_lines));
    jput(a, o, "totalBytes", jnum(a, t->total_bytes));
    jput(a, o, "outputLines", jnum(a, t->output_lines));
    jput(a, o, "outputBytes", jnum(a, t->output_bytes));
    jput(a, o, "lastLinePartial", json_bool(a, t->last_partial));
    jput(a, o, "firstLineExceedsLimit", json_bool(a, t->first_exceeds));
    jput(a, o, "maxLines", jnum(a, t->max_lines));
    jput(a, o, "maxBytes", jnum(a, t->max_bytes));
    return o;
}

static void lines_for_counting(Str s, VecStr *v) {
    if (!s.len) return;
    split_lines(s, v);
    if (s.ptr[s.len - 1] == '\n') v->len--;
}

static Trunc truncate_head(Str content, double max_lines, double max_bytes) {
    Trunc t = {.content = content, .max_lines = max_lines, .max_bytes = max_bytes};
    VecStr l = {0};
    lines_for_counting(content, &l);
    t.total_bytes = (double)content.len;
    t.total_lines = (double)l.len;
    if (t.total_lines <= max_lines && t.total_bytes <= max_bytes) {
        t.output_lines = t.total_lines;
        t.output_bytes = t.total_bytes;
        vec_free(&l);
        return t;
    }
    t.truncated = true;
    if ((double)l.ptr[0].len > max_bytes) {
        t.content = (Str){"", 0};
        t.by = "bytes";
        t.first_exceeds = true;
        vec_free(&l);
        return t;
    }
    size_t n = 0;
    double bytes = 0;
    t.by = "lines";
    for (size_t i = 0; i < l.len && (double)i < max_lines; i++) {
        double lb = (double)l.ptr[i].len + (i > 0 ? 1 : 0);
        if (bytes + lb > max_bytes) { t.by = "bytes"; break; }
        n++;
        bytes += lb;
    }
    if ((double)n >= max_lines && bytes <= max_bytes) t.by = "lines";
    t.content = span(&l, 0, n);
    t.output_lines = (double)n;
    t.output_bytes = (double)t.content.len;
    vec_free(&l);
    return t;
}

static Trunc truncate_tail(Str content, double max_lines, double max_bytes) {
    Trunc t = {.content = content, .max_lines = max_lines, .max_bytes = max_bytes};
    VecStr l = {0};
    lines_for_counting(content, &l);
    t.total_bytes = (double)content.len;
    t.total_lines = (double)l.len;
    if (t.total_lines <= max_lines && t.total_bytes <= max_bytes) {
        t.output_lines = t.total_lines;
        t.output_bytes = t.total_bytes;
        vec_free(&l);
        return t;
    }
    t.truncated = true;
    t.by = "lines";
    size_t n = 0; // lines taken from the end
    double bytes = 0;
    Str partial = {0};
    for (ptrdiff_t i = (ptrdiff_t)l.len - 1; i >= 0 && (double)n < max_lines; i--) {
        Str line = l.ptr[i];
        double lb = (double)line.len + (n > 0 ? 1 : 0);
        if (bytes + lb > max_bytes) {
            t.by = "bytes";
            if (n == 0) {
                size_t start = line.len > (size_t)max_bytes ? line.len - (size_t)max_bytes : 0;
                while (start < line.len && ((uint8_t)line.ptr[start] & 0xc0) == 0x80) start++;
                partial = str_n(line.ptr + start, line.len - start);
                bytes = (double)partial.len;
                t.last_partial = true;
                n = 1;
            }
            break;
        }
        n++;
        bytes += lb;
    }
    if ((double)n >= max_lines && bytes <= max_bytes) t.by = "lines";
    t.content = t.last_partial ? partial : span(&l, l.len - n, l.len);
    t.output_lines = (double)n;
    t.output_bytes = (double)t.content.len;
    vec_free(&l);
    return t;
}

// ---------------------------------------------------------------------------------------
// processes
// ---------------------------------------------------------------------------------------

static bool on_path(const char *name) {
    const char *path = getenv("PATH");
    if (!path) return false;
    for (const char *p = path; *p;) {
        const char *q = strchr(p, ':');
        size_t n = q ? (size_t)(q - p) : strlen(p);
        if (n) {
            char full[4096];
            snprintf(full, sizeof full, "%.*s/%s", (int)n, p, name);
            struct stat st;
            if (stat(full, &st) == 0 && S_ISREG(st.st_mode) && access(full, X_OK) == 0) return true;
        }
        if (!q) break;
        p = q + 1;
    }
    return false;
}

// tools-manager.ts getToolPath: the agent's bin folder first, then PATH
static const char *tool_path(Arena *a, const CodeEnv *env, const char *name, const char *alt) {
    const char *agent = env->agent_dir ? env->agent_dir : arena_printf(a, "%s/.pi/agent", env->home);
    char *local = arena_printf(a, "%s/bin/%s", agent, name);
    if (exists(local)) return local;
    if (on_path(name)) return name;
    if (alt && on_path(alt)) return alt;
    return NULL;
}

typedef struct {
    bool (*on_line)(void *ctx, Str line); // returns true to stop the child (SIGTERM)
    void *ctx;
    Buf err;
    bool exited_code; // false: killed by a signal (Node's code null)
    int code;
    bool killed;
    int wait_err; // errno of a failed poll: the child was killed and reaped, its output is partial
    volatile int *abort; // CodeEnv.abort: when set, the child is killed and aborted is true
    bool aborted;
} Proc;

// Runs argv (PATH lookup) in cwd (NULL: ours) with stdin /dev/null; stdout lines to on_line,
// stderr collected. Returns errno of a failed spawn, else 0 (check pr->wait_err).
static int run_proc(const char *const *argv, const char *cwd, Proc *pr) {
    int out[2], err[2];
    if (pipe(out)) return errno;
    if (pipe(err)) { close(out[0]); close(out[1]); return errno; }
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_adddup2(&fa, out[1], 1);
    posix_spawn_file_actions_adddup2(&fa, err[1], 2);
    if (cwd) posix_spawn_file_actions_addchdir(&fa, cwd);
    posix_spawnattr_t at;
    posix_spawnattr_init(&at);
    sigset_t none, all;
    sigemptyset(&none);
    sigfillset(&all);
    posix_spawnattr_setsigmask(&at, &none);
    posix_spawnattr_setsigdefault(&at, &all);
    posix_spawnattr_setflags(&at, POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_CLOEXEC_DEFAULT);
    pid_t pid;
    int rc = posix_spawnp(&pid, argv[0], &fa, &at, (char *const *)argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    posix_spawnattr_destroy(&at);
    close(out[1]);
    close(err[1]);
    if (rc) { close(out[0]); close(err[0]); return rc; }
    Buf line = {0};
    bool stop = false;
    struct pollfd fds[2] = {{out[0], POLLIN, 0}, {err[0], POLLIN, 0}};
    int open = 2;
    char buf[65536];
    while (open > 0) {
        if (pr->abort && *pr->abort && !pr->aborted) { // grep.ts / find.ts: the signal kills the child
            pr->aborted = true;
            kill(pid, SIGKILL);
        }
        int ready = poll(fds, 2, pr->abort ? 50 : -1);
        if (ready == 0) continue;
        if (ready < 0) {
            if (errno == EINTR) continue;
            pr->wait_err = errno;
            kill(pid, SIGKILL);
            for (int k = 0; k < 2; k++)
                if (fds[k].fd >= 0) { close(fds[k].fd); fds[k].fd = -1; }
            open = 0;
            stop = true;
            break;
        }
        for (int k = 0; k < 2; k++) {
            if (fds[k].fd < 0 || !(fds[k].revents & (POLLIN | POLLHUP | POLLERR))) continue;
            ssize_t n = read(fds[k].fd, buf, sizeof buf);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) {
                close(fds[k].fd);
                fds[k].fd = -1;
                open--;
                continue;
            }
            if (k == 1) { buf_add(&pr->err, buf, (size_t)n); continue; }
            buf_add(&line, buf, (size_t)n);
            size_t start = 0;
            for (;;) {
                ptrdiff_t nl = schr(buf_str(&line), '\n', start);
                if (nl == -1) break;
                Str l = str_n(line.ptr + start, (size_t)nl - start);
                if (l.len && l.ptr[l.len - 1] == '\r') l.len--;
                if (!stop && pr->on_line(pr->ctx, l)) {
                    stop = true;
                    if (!pr->killed) { pr->killed = true; kill(pid, SIGTERM); }
                }
                start = (size_t)nl + 1;
            }
            memmove(line.ptr, line.ptr + start, line.len - start);
            line.len -= start;
        }
    }
    if (line.len && !stop) {
        Str l = buf_str(&line);
        if (l.ptr[l.len - 1] == '\r') l.len--;
        pr->on_line(pr->ctx, l);
    }
    buf_free(&line);
    int st = 0;
    pid_t w;
    while ((w = waitpid(pid, &st, 0)) < 0 && errno == EINTR) {}
    if (w < 0 && !pr->wait_err) pr->wait_err = errno;
    pr->exited_code = WIFEXITED(st);
    pr->code = WIFEXITED(st) ? WEXITSTATUS(st) : 0;
    return 0;
}

// ---------------------------------------------------------------------------------------
// tool outputs
// ---------------------------------------------------------------------------------------

typedef struct {
    JValue *content;    // array of blocks
    JValue *details;    // NULL: undefined
    JValue *structured; // NULL: absent
    bool is_error;      // the result's own isError (bash)
    const char *thrown; // the tool threw this message
} ToolOut;

static void out_text(Arena *a, ToolOut *o, Str text, JValue *details) {
    o->content = text_content(a, text);
    o->details = details;
}

// ---------------------------------------------------------------------------------------
// read (read.ts)
// ---------------------------------------------------------------------------------------

static uint32_t rd32be(const uint8_t *p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }
static uint32_t rd32le(const uint8_t *p) { return (uint32_t)p[3] << 24 | (uint32_t)p[2] << 16 | (uint32_t)p[1] << 8 | p[0]; }
static uint32_t rd16le(const uint8_t *p) { return (uint32_t)p[1] << 8 | p[0]; }

// utils/mime.ts detectSupportedImageMimeType
static const char *image_mime(const uint8_t *b, size_t n) {
    if (n >= 3 && b[0] == 0xff && b[1] == 0xd8 && b[2] == 0xff) return n > 3 && b[3] == 0xf7 ? NULL : "image/jpeg";
    static const uint8_t png[8] = {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
    if (n >= 8 && !memcmp(b, png, 8)) {
        if (!(n >= 16 && rd32be(b + 8) == 13 && !memcmp(b + 12, "IHDR", 4))) return NULL;
        size_t off = 8;
        while (off + 8 <= n) {
            if (!memcmp(b + off + 4, "acTL", 4)) return NULL;
            if (!memcmp(b + off + 4, "IDAT", 4)) break;
            size_t next = off + 8 + rd32be(b + off) + 4;
            if (next <= off || next > n) break;
            off = next;
        }
        return "image/png";
    }
    if (n >= 6 && (!memcmp(b, "GIF87a", 6) || !memcmp(b, "GIF89a", 6))) return "image/gif";
    if (n >= 12 && !memcmp(b, "RIFF", 4) && !memcmp(b + 8, "WEBP", 4)) return "image/webp";
    if (n >= 2 && !memcmp(b, "BM", 2)) {
        if (n < 26) return NULL;
        uint32_t size = rd32le(b + 2), pix = rd32le(b + 10), dib = rd32le(b + 14);
        if (size != 0 && size < 26) return NULL;
        if (pix < 14 + dib) return NULL;
        if (size != 0 && pix >= size) return NULL;
        uint32_t planes, bpp;
        if (dib == 12) { planes = rd16le(b + 22); bpp = rd16le(b + 24); }
        else if (dib >= 40 && dib <= 124) {
            if (n < 30) return NULL;
            planes = rd16le(b + 26);
            bpp = rd16le(b + 28);
        } else return NULL;
        return planes == 1 && (bpp == 1 || bpp == 4 || bpp == 8 || bpp == 16 || bpp == 24 || bpp == 32) ? "image/bmp" : NULL;
    }
    return NULL;
}

static Str base64(Arena *a, const uint8_t *p, size_t n) {
    static const char tab[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    char *o = arena_alloc(a, (n + 2) / 3 * 4 + 1);
    size_t k = 0;
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = (uint32_t)p[i] << 16 | (i + 1 < n ? (uint32_t)p[i + 1] << 8 : 0) | (i + 2 < n ? p[i + 2] : 0);
        o[k++] = tab[v >> 18];
        o[k++] = tab[(v >> 12) & 63];
        o[k++] = i + 1 < n ? tab[(v >> 6) & 63] : '=';
        o[k++] = i + 2 < n ? tab[v & 63] : '=';
    }
    return str_n(o, k);
}

static int read_all(int fd, Buf *b) {
    char tmp[65536];
    for (;;) {
        ssize_t n = read(fd, tmp, sizeof tmp);
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) return errno;
        if (n == 0) return 0;
        buf_add(b, tmp, (size_t)n);
    }
}

// Array.prototype.slice index
static size_t slice_idx(double x, size_t len) {
    if (isnan(x)) return 0;
    x = trunc(x);
    if (x < 0) return x + (double)len < 0 ? 0 : (size_t)(x + (double)len);
    return x > (double)len ? len : (size_t)x;
}

static bool arg_num(const JValue *args, const char *k, double *out) { return jnumber(json_get(args, k), out); }

static void tool_read(Arena *a, const CodeEnv *env, const JValue *args, ToolOut *o) {
    Str path = json_strs(json_get(args, "path"));
    double offset = 0, limit = 0;
    bool has_off = arg_num(args, "offset", &offset), has_lim = arg_num(args, "limit", &limit);
    char *abs = resolve_read_path(a, env, path);
    if (access(abs, R_OK)) { o->thrown = node_err(a, errno, "access", abs); return; }
    int fd = open(abs, O_RDONLY);
    if (fd < 0) { o->thrown = node_err(a, errno, "open", abs); return; }
    uint8_t head[4100];
    ssize_t hn;
    while ((hn = pread(fd, head, sizeof head, 0)) < 0 && errno == EINTR) {}
    if (hn < 0) { o->thrown = node_err(a, errno, "read", NULL); close(fd); return; }
    const char *mime = image_mime(head, (size_t)hn);
    Buf data = {0};
    int e = read_all(fd, &data);
    close(fd);
    if (e) { buf_free(&data); o->thrown = node_err(a, e, "read", NULL); return; }
    if (mime) {
        // processImage without resizing or conversion; the models saetta serves read text only
        Str note = sfmt(a, "Read image file [%s]\n[Current model does not support images. The image will be omitted from this request.]", mime);
        o->content = text_content(a, note);
        JValue *img = json_new(a, J_OBJECT);
        jput(a, img, "type", jstr(a, S("image")));
        jput(a, img, "data", jstr(a, base64(a, (const uint8_t *)data.ptr, data.len)));
        jput(a, img, "mimeType", jstr(a, str(mime)));
        json_push(a, o->content, img);
        buf_free(&data);
        return;
    }
    Str text = u8_text(a, data.ptr, data.len, true);
    buf_free(&data);
    VecStr lines = {0};
    split_lines(text, &lines);
    double total = (double)lines.len;
    double start = (has_off && offset != 0) ? fmax(0, offset - 1) : 0;
    double start_disp = start + 1;
    double end = has_lim ? fmin(start + limit, total) : total;
    size_t s = slice_idx(start, lines.len), en = slice_idx(end, lines.len);
    Trunc t = truncate_head(span(&lines, s, en), MAX_LINES, MAX_BYTES);
    double first_bytes = t.first_exceeds ? (double)lines.ptr[s].len : 0;
    vec_free(&lines);
    if (start >= total) {
        o->thrown = arena_printf(a, "Offset %s is beyond end of file (%s lines total)", has_off ? js_num(a, offset) : "undefined", js_num(a, total));
        return;
    }
    bool has_user = has_lim;
    double user_lines = has_lim ? fmin(start + limit, total) - start : 0;
    Str out;
    JValue *details = NULL;
    if (t.first_exceeds) {
        out = sfmt(a, "[Line %s is %s, exceeds %s limit. Use bash: sed -n '%sp' %.*s | head -c %d]", js_num(a, start_disp),
                   format_size(a, first_bytes), format_size(a, MAX_BYTES), js_num(a, start_disp), (int)path.len, path.ptr, MAX_BYTES);
        details = json_new(a, J_OBJECT);
        jput(a, details, "truncation", trunc_json(a, &t));
    } else if (t.truncated) {
        double end_disp = start_disp + t.output_lines - 1;
        if (!strcmp(t.by, "lines"))
            out = sfmt(a, "%.*s\n\n[Showing lines %s-%s of %s. Use offset=%s to continue.]", (int)t.content.len, t.content.ptr,
                       js_num(a, start_disp), js_num(a, end_disp), js_num(a, total), js_num(a, end_disp + 1));
        else
            out = sfmt(a, "%.*s\n\n[Showing lines %s-%s of %s (%s limit). Use offset=%s to continue.]", (int)t.content.len, t.content.ptr,
                       js_num(a, start_disp), js_num(a, end_disp), js_num(a, total), format_size(a, MAX_BYTES), js_num(a, end_disp + 1));
        details = json_new(a, J_OBJECT);
        jput(a, details, "truncation", trunc_json(a, &t));
    } else if (has_user && start + user_lines < total) {
        out = sfmt(a, "%.*s\n\n[%s more lines in file. Use offset=%s to continue.]", (int)t.content.len, t.content.ptr,
                   js_num(a, total - (start + user_lines)), js_num(a, start + user_lines + 1));
    } else out = t.content;
    out_text(a, o, out, details);
    o->structured = jstr(a, out);
}

// ---------------------------------------------------------------------------------------
// bash (bash.ts, output-accumulator.ts, utils/shell.ts, utils/child-process.ts)
// ---------------------------------------------------------------------------------------

typedef struct {
    Utf8Dec dec;
    Buf raw, tail;
    bool tail_at_line;
    uint64_t total_raw, total_dec, completed, total_lines, cur_line_bytes;
    bool open_line;
    char *tmp_path;
    int tmp_fd;
    Arena *a;
} Acc;

static const char *tmp_dir(Arena *a) {
    const char *t = getenv("TMPDIR");
    if (!t || !*t) t = getenv("TMP");
    if (!t || !*t) t = getenv("TEMP");
    if (!t || !*t) t = "/tmp";
    size_t n = strlen(t);
    if (n > 1 && t[n - 1] == '/') return arena_strndup(a, t, n - 1);
    return t;
}

static bool acc_should_file(const Acc *c) {
    return c->total_raw > MAX_BYTES || c->total_dec > MAX_BYTES || c->total_lines > MAX_LINES;
}

static void write_full(int fd, const void *p, size_t n) {
    const char *q = p;
    while (n > 0 && fd >= 0) {
        ssize_t w = write(fd, q, n);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return;
        q += w;
        n -= (size_t)w;
    }
}

static void acc_ensure_file(Acc *c) {
    if (c->tmp_path) return;
    uint8_t r[8];
    arc4random_buf(r, sizeof r);
    c->tmp_path = arena_printf(c->a, "%s/pi-bash-%02x%02x%02x%02x%02x%02x%02x%02x.log", tmp_dir(c->a), r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
    c->tmp_fd = open(c->tmp_path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    write_full(c->tmp_fd, c->raw.ptr, c->raw.len);
    buf_free(&c->raw);
}

static void acc_decoded(Acc *c, Str text) {
    if (!text.len) return;
    c->total_dec += text.len;
    buf_addstr(&c->tail, text);
    const size_t roll = MAX_BYTES * 2;
    if (c->tail.len > roll * 2) {
        size_t start = c->tail.len - roll;
        while (start < c->tail.len && ((uint8_t)c->tail.ptr[start] & 0xc0) == 0x80) start++;
        c->tail_at_line = start == 0 ? c->tail_at_line : c->tail.ptr[start - 1] == '\n';
        memmove(c->tail.ptr, c->tail.ptr + start, c->tail.len - start);
        c->tail.len -= start;
        c->tail.ptr[c->tail.len] = 0;
    }
    size_t nl = 0;
    ptrdiff_t last = -1;
    for (ptrdiff_t i = schr(text, '\n', 0); i != -1; i = schr(text, '\n', (size_t)i + 1)) { nl++; last = i; }
    if (!nl) {
        c->cur_line_bytes += text.len;
        c->open_line = true;
    } else {
        c->completed += nl;
        c->cur_line_bytes = text.len - (size_t)last - 1;
        c->open_line = c->cur_line_bytes > 0;
    }
    c->total_lines = c->completed + (c->open_line ? 1 : 0);
}

static void acc_append(Acc *c, const void *p, size_t n) {
    c->total_raw += n;
    Buf d = {0};
    u8_decode(&c->dec, &d, p, n, false);
    acc_decoded(c, buf_str(&d));
    buf_free(&d);
    if (c->tmp_path || acc_should_file(c)) {
        acc_ensure_file(c);
        write_full(c->tmp_fd, p, n);
    } else buf_add(&c->raw, p, n);
}

static void acc_finish(Acc *c) {
    Buf d = {0};
    u8_decode(&c->dec, &d, NULL, 0, true);
    acc_decoded(c, buf_str(&d));
    buf_free(&d);
    if (acc_should_file(c)) acc_ensure_file(c);
}

static Trunc acc_snapshot(Arena *a, Acc *c) {
    Str text = buf_str(&c->tail);
    if (!c->tail_at_line) {
        ptrdiff_t nl = schr(text, '\n', 0);
        if (nl != -1) text = str_n(text.ptr + nl + 1, text.len - (size_t)nl - 1);
    }
    Trunc t = truncate_tail(text, MAX_LINES, MAX_BYTES);
    t.content = (Str){str_z(a, t.content), t.content.len};
    bool truncated = c->total_lines > MAX_LINES || c->total_dec > MAX_BYTES;
    t.by = truncated ? (t.by ? t.by : c->total_dec > MAX_BYTES ? "bytes" : "lines") : NULL;
    t.truncated = truncated;
    t.total_lines = (double)c->total_lines;
    t.total_bytes = (double)c->total_dec;
    if (truncated) acc_ensure_file(c);
    return t;
}

// readFullOutput(1 MB) after the temp file is closed
static Str acc_full(Arena *a, Acc *c, bool *truncated) {
    const size_t max = 1024 * 1024;
    *truncated = false;
    if (!c->tmp_path) return u8_text(a, c->raw.ptr, c->raw.len, false);
    int fd = open(c->tmp_path, O_RDONLY);
    Buf all = {0};
    struct stat st;
    if (fd < 0 || fstat(fd, &st)) {
        if (fd >= 0) close(fd);
        return (Str){"", 0};
    }
    size_t size = (size_t)st.st_size;
    if (size <= max) {
        read_all(fd, &all);
        close(fd);
        Str s = u8_text(a, all.ptr, all.len, false);
        buf_free(&all);
        return s;
    }
    size_t hb = max / 2, tb = max - hb;
    uint8_t *h = xmalloc(hb), *t = xmalloc(tb);
    ssize_t hr = pread(fd, h, hb, 0), tr = pread(fd, t, tb, (off_t)(size - tb));
    close(fd);
    if (hr < 0) hr = 0;
    if (tr < 0) tr = 0;
    Utf8Dec hd = {0};
    Buf o = {0};
    u8_decode(&hd, &o, h, (size_t)hr, false);
    size_t ts = 0;
    while (ts < (size_t)tr && (t[ts] & 0xc0) == 0x80) ts++;
    buf_printf(&o, "\n\n[... %zu bytes omitted ...]\n\n", size - hb - tb);
    Utf8Dec td = {0};
    u8_decode(&td, &o, t + ts, (size_t)tr - ts, true);
    free(h);
    free(t);
    *truncated = true;
    return take(a, &o);
}

static void kill_tree(pid_t pid) {
    kill(-pid, SIGSTOP);
    if (kill(-pid, SIGKILL)) kill(pid, SIGKILL);
}

// How often the bash wait loop looks at env->abort when nothing else wakes it.
#define ABORT_POLL_MS 50.0

static bool reap(pid_t pid, int *st) {
    pid_t w;
    while ((w = waitpid(pid, st, 0)) < 0 && errno == EINTR) {}
    return w == pid;
}

// pi's trackedDetachedChildPids (utils/shell.ts): the bash process groups while their command
// runs (tracked at spawn, untracked once the wait is over), killed by code_tools_kill_detached.
// Lock-free atomics, so a signal handler may call it.
#define TRACK_MAX 32
static _Atomic pid_t tracked_groups[TRACK_MAX];

static void track_group(pid_t pid) {
    for (size_t i = 0; i < TRACK_MAX; i++) {
        pid_t z = 0;
        if (atomic_compare_exchange_strong(&tracked_groups[i], &z, pid)) return;
    }
}

static void untrack_group(pid_t pid) {
    for (size_t i = 0; i < TRACK_MAX; i++) {
        pid_t p = pid;
        if (atomic_compare_exchange_strong(&tracked_groups[i], &p, 0)) return;
    }
}

void code_tools_kill_detached(void) {
    for (size_t i = 0; i < TRACK_MAX; i++) {
        pid_t p = atomic_exchange(&tracked_groups[i], 0);
        if (p > 0) kill_tree(p);
    }
    code_mcp_kill_signal_safe(); // the MCP servers' process groups too (stdio.ts's exit hook)
}

static double ms_since(uint64_t t0) { return (double)(now_ns() - t0) / 1e6; }

static char **shell_env(Arena *a, const CodeEnv *env) {
    size_t n = 0;
    while (environ[n]) n++;
    char **e = ARENA_ARRAY(a, char *, n + 8);
    size_t k = 0;
    const char *path = "";
    for (size_t i = 0; i < n; i++) {
        const char *v = environ[i];
        if (!strncmp(v, "PATH=", 5)) { path = v + 5; continue; }
        if (!strncmp(v, "PI_SESSION_ID=", 14) || !strncmp(v, "PI_SESSION_FILE=", 16) || !strncmp(v, "PI_PROVIDER=", 12) ||
            !strncmp(v, "PI_MODEL=", 9) || !strncmp(v, "PI_REASONING_LEVEL=", 19))
            continue;
        e[k++] = (char *)v;
    }
    const char *agent = env->agent_dir ? env->agent_dir : arena_printf(a, "%s/.pi/agent", env->home);
    const char *bin = arena_printf(a, "%s/bin", agent);
    bool has = false;
    for (const char *p = path; *p;) {
        const char *q = strchr(p, ':');
        size_t m = q ? (size_t)(q - p) : strlen(p);
        if (m == strlen(bin) && !strncmp(p, bin, m)) has = true;
        if (!q) break;
        p = q + 1;
    }
    e[k++] = has ? arena_printf(a, "PATH=%s", path) : *path ? arena_printf(a, "PATH=%s:%s", bin, path) : arena_printf(a, "PATH=%s", bin);
    if (env->session_id) e[k++] = arena_printf(a, "PI_SESSION_ID=%s", env->session_id);
    if (env->session_id && env->session_file) e[k++] = arena_printf(a, "PI_SESSION_FILE=%s", env->session_file);
    if (env->session_id && env->provider && env->model) {
        e[k++] = arena_printf(a, "PI_PROVIDER=%s", env->provider);
        e[k++] = arena_printf(a, "PI_MODEL=%s", env->model);
    }
    if (env->session_id && env->thinking) e[k++] = arena_printf(a, "PI_REASONING_LEVEL=%s", env->thinking);
    e[k] = NULL;
    return e;
}

static const char *shell_path(void) {
    if (exists("/bin/bash")) return "/bin/bash";
    if (on_path("bash")) return "bash";
    return "sh";
}

// emitOutputUpdate: the output so far as a partial result (a throwaway arena: one per update).
static void bash_update(const CodeEnv *env, Acc *acc, bool empty) {
    if (!env->on_partial) return;
    Arena t = {0};
    JValue *r = json_new(&t, J_OBJECT);
    JValue *content = json_new(&t, J_ARRAY);
    if (!empty) {
        Arena *keep = acc->a;
        acc->a = &t; // the snapshot's text lives in the throwaway arena (the temp file's path stays in the run's)
        bool had_path = acc->tmp_path != NULL;
        Trunc tr = acc_snapshot(&t, acc);
        acc->a = keep;
        if (!had_path && acc->tmp_path) acc->tmp_path = arena_strdup(keep, acc->tmp_path);
        JValue *b = json_new(&t, J_OBJECT);
        jput(&t, b, "type", jstr(&t, S("text")));
        jput(&t, b, "text", jstr(&t, tr.content));
        json_push(&t, content, b);
        jput(&t, r, "content", content);
        JValue *d = json_new(&t, J_OBJECT);
        if (tr.truncated) jput(&t, d, "truncation", trunc_json(&t, &tr));
        if (acc->tmp_path) jput(&t, d, "fullOutputPath", jstr(&t, str(acc->tmp_path)));
        jput(&t, r, "details", d);
    } else jput(&t, r, "content", content);
    env->on_partial(env->partial_ctx, r);
    arena_reset(&t);
}

static void tool_bash(Arena *a, const CodeEnv *env, const JValue *args, ToolOut *o) {
    Str command = json_strs(json_get(args, "command"));
    double timeout = 0;
    bool has_to = arg_num(args, "timeout", &timeout);
    Acc acc = {.tail_at_line = true, .tmp_fd = -1, .a = a};
    uint64_t started = now_ns();
    const char *thrown = NULL;
    bool timed_out = false, aborted = false;
    volatile int *ab = env->abort;
    double shell_exit_before = -1; // ms after start when the shell exited, if before the deadline
    int exit_code = 0;
    double timeout_ms = 0;
    if (has_to) {
        if (!isfinite(timeout) || timeout <= 0) thrown = "Invalid timeout: must be a finite number of seconds";
        else if ((timeout_ms = timeout * 1000) > 2147483647.0) thrown = "Invalid timeout: maximum is 2147483.647 seconds";
    }
    if (!thrown && access(env->cwd, F_OK)) thrown = arena_printf(a, "Working directory does not exist: %s\nCannot execute bash commands.", env->cwd);
    if (!thrown) {
        int out[2] = {-1, -1}, err[2] = {-1, -1};
        pid_t pid = -1;
        int rc = pipe(out) ? errno : pipe(err) ? errno : 0;
        if (!rc) {
            fcntl(out[0], F_SETFD, FD_CLOEXEC);
            fcntl(err[0], F_SETFD, FD_CLOEXEC);
            posix_spawn_file_actions_t fa;
            posix_spawn_file_actions_init(&fa);
            posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
            posix_spawn_file_actions_adddup2(&fa, out[1], 1);
            posix_spawn_file_actions_adddup2(&fa, err[1], 2);
            posix_spawn_file_actions_addchdir(&fa, env->cwd);
            posix_spawnattr_t at;
            posix_spawnattr_init(&at);
            sigset_t none, all;
            sigemptyset(&none);
            sigfillset(&all);
            posix_spawnattr_setsigmask(&at, &none);
            posix_spawnattr_setsigdefault(&at, &all);
            posix_spawnattr_setflags(&at, POSIX_SPAWN_SETSID | POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_CLOEXEC_DEFAULT);
            const char *sh = shell_path();
            const char *argv[] = {sh, "-c", str_z(a, command), NULL};
            rc = posix_spawnp(&pid, sh, &fa, &at, (char *const *)argv, shell_env(a, env));
            posix_spawn_file_actions_destroy(&fa);
            posix_spawnattr_destroy(&at);
        }
        if (out[1] >= 0) close(out[1]);
        if (err[1] >= 0) close(err[1]);
        if (rc) {
            if (out[0] >= 0) close(out[0]);
            if (err[0] >= 0) close(err[0]);
            thrown = arena_printf(a, "spawn %s %s", shell_path(), errno_code(rc));
        } else {
            track_group(pid);
            const char *werr = NULL; // waiting failed: the group is killed and reaped, no result
            bool exited = false;
            int st = 0;
            int kq = kqueue();
            if (kq < 0) werr = arena_printf(a, "kqueue %s", errno_code(errno));
            else {
                struct kevent ch[3];
                EV_SET(&ch[0], out[0], EVFILT_READ, EV_ADD, 0, 0, NULL);
                EV_SET(&ch[1], err[0], EVFILT_READ, EV_ADD, 0, 0, NULL);
                EV_SET(&ch[2], pid, EVFILT_PROC, EV_ADD | EV_ONESHOT, NOTE_EXIT, 0, NULL);
                if (kevent(kq, ch, 2, NULL, 0, NULL) < 0) werr = arena_printf(a, "kevent %s", errno_code(errno));
                else if (kevent(kq, &ch[2], 1, NULL, 0, NULL) < 0) {
                    if (errno != ESRCH) werr = arena_printf(a, "kevent %s", errno_code(errno));
                    else { // already gone
                        if (!reap(pid, &st)) werr = arena_printf(a, "waitpid %s", errno_code(errno));
                        exited = true;
                    }
                }
            }
            int fds[2] = {out[0], err[0]};
            int nopen = 2;
            uint64_t t0 = now_ns();
            // the live output (scheduleOutputUpdate): an empty update at once, then at most every 100 ms
            bash_update(env, &acc, true);
            uint64_t last_update = 0;
            bool update_dirty = false;
            double exited_at = exited ? 0 : -1, idle_until = exited ? 100 : -1;
            char buf[65536];
            while (!werr) {
                if (exited && nopen == 0) break;
                double now = ms_since(t0);
                if (ab && *ab && !aborted) { // pi's onAbort: kill the whole tree, keep waiting for it
                    aborted = true;
                    kill_tree(pid);
                }
                if (has_to && !timed_out && now >= timeout_ms) {
                    timed_out = true;
                    shell_exit_before = exited ? exited_at : -1;
                    kill_tree(pid);
                }
                if (exited && now >= idle_until) break;
                double wait = -1;
                if (has_to && !timed_out) wait = timeout_ms - now;
                if (exited && (wait < 0 || idle_until - now < wait)) wait = idle_until - now;
                if (ab && !aborted && (wait < 0 || wait > ABORT_POLL_MS)) wait = ABORT_POLL_MS;
                if (update_dirty) {
                    double due = 100 - (double)(now_ns() - last_update) / 1e6;
                    if (due <= 0) {
                        bash_update(env, &acc, false);
                        last_update = now_ns();
                        update_dirty = false;
                    } else if (wait < 0 || wait > due) wait = due;
                }
                struct timespec ts, *tsp = NULL;
                if (wait >= 0) {
                    ts.tv_sec = (time_t)(wait / 1000);
                    ts.tv_nsec = (long)(fmod(wait, 1000) * 1e6);
                    tsp = &ts;
                }
                struct kevent ev[4];
                int n = kevent(kq, NULL, 0, ev, 4, tsp);
                if (n < 0) {
                    if (errno != EINTR) werr = arena_printf(a, "kevent %s", errno_code(errno));
                    continue;
                }
                for (int i = 0; i < n; i++) {
                    if (ev[i].filter == EVFILT_PROC) {
                        if (!reap(pid, &st)) werr = arena_printf(a, "waitpid %s", errno_code(errno));
                        exited = true;
                        exited_at = ms_since(t0);
                        idle_until = exited_at + 100;
                        continue;
                    }
                    int fd = (int)ev[i].ident;
                    ssize_t r = read(fd, buf, sizeof buf);
                    if (r < 0 && (errno == EINTR || errno == EAGAIN)) continue;
                    if (r <= 0) {
                        struct kevent del;
                        EV_SET(&del, fd, EVFILT_READ, EV_DELETE, 0, 0, NULL);
                        kevent(kq, &del, 1, NULL, 0, NULL);
                        close(fd);
                        for (int k = 0; k < 2; k++)
                            if (fds[k] == fd) fds[k] = -1;
                        nopen--;
                        continue;
                    }
                    acc_append(&acc, buf, (size_t)r);
                    update_dirty = env->on_partial != NULL;
                    if (exited) idle_until = ms_since(t0) + 100;
                }
            }
            if (update_dirty) bash_update(env, &acc, false); // finishOutput's last update
            if (!exited) { // the wait failed: never leave the group running or a zombie behind
                kill_tree(pid);
                if (!reap(pid, &st) && !werr) werr = arena_printf(a, "waitpid %s", errno_code(errno));
            }
            for (int k = 0; k < 2; k++)
                if (fds[k] >= 0) close(fds[k]);
            if (kq >= 0) close(kq);
            untrack_group(pid);
            if (ab && *ab) aborted = true; // pi checks signal.aborted once the wait is over
            if (werr && !aborted) thrown = arena_printf(a, "Failed to wait for %s: %s", shell_path(), werr);
            exit_code = WIFEXITED(st) ? WEXITSTATUS(st) : WIFSIGNALED(st) ? 128 + WTERMSIG(st) : 1;
        }
    }
    // finishOutput + formatOutput
    acc_finish(&acc);
    Trunc t = acc_snapshot(a, &acc);
    if (acc.tmp_fd >= 0) { close(acc.tmp_fd); acc.tmp_fd = -1; }
    bool thrown_path = thrown || timed_out || aborted;
    Str text = t.content.len ? t.content : thrown_path ? (Str){"", 0} : S("(no output)");
    JValue *details = NULL;
    if (t.truncated) {
        details = json_new(a, J_OBJECT);
        jput(a, details, "truncation", trunc_json(a, &t));
        jput(a, details, "fullOutputPath", jstr(a, str(acc.tmp_path)));
        double sl = t.total_lines - t.output_lines + 1, el = t.total_lines;
        if (t.last_partial)
            text = sfmt(a, "%.*s\n\n[Showing last %s of line %s (line is %s). Full output: %s]", (int)text.len, text.ptr, format_size(a, t.output_bytes),
                        js_num(a, el), format_size(a, (double)acc.cur_line_bytes), acc.tmp_path);
        else if (!strcmp(t.by, "lines"))
            text = sfmt(a, "%.*s\n\n[Showing lines %s-%s of %s. Full output: %s]", (int)text.len, text.ptr, js_num(a, sl), js_num(a, el),
                        js_num(a, t.total_lines), acc.tmp_path);
        else
            text = sfmt(a, "%.*s\n\n[Showing lines %s-%s of %s (%s limit). Full output: %s]", (int)text.len, text.ptr, js_num(a, sl),
                        js_num(a, el), js_num(a, t.total_lines), format_size(a, MAX_BYTES), acc.tmp_path);
    }
    if (thrown_path) {
        if (aborted) thrown = text.len ? arena_printf(a, "%.*s\n\nCommand aborted", (int)text.len, text.ptr) : "Command aborted";
        else if (timed_out) {
            Str status = sfmt(a, "Command timed out after %s seconds", js_num(a, timeout));
            if (shell_exit_before >= 0)
                status = sfmt(a, "%.*s. The shell exited with code %d after %s seconds, but background processes it started kept writing output until the timeout.",
                              (int)status.len, status.ptr, exit_code, js_num(a, round(shell_exit_before / 100) / 10));
            thrown = text.len ? arena_printf(a, "%.*s\n\n%.*s", (int)text.len, text.ptr, (int)status.len, status.ptr) : str_z(a, status);
        }
        o->thrown = thrown;
        buf_free(&acc.raw);
        buf_free(&acc.tail);
        return;
    }
    double wall = round(ms_since(started) / 100) / 10;
    bool full_trunc;
    Str full = acc_full(a, &acc, &full_trunc);
    JValue *sc = json_new(a, J_OBJECT);
    jput(a, sc, "output", jstr(a, full));
    jput(a, sc, "truncated", json_bool(a, full_trunc));
    if (full_trunc && acc.tmp_path) jput(a, sc, "full_output_path", jstr(a, str(acc.tmp_path)));
    jput(a, sc, "exit_code", jnum(a, exit_code));
    jput(a, sc, "wall_time_seconds", jnum(a, wall));
    if (exit_code != 0) {
        text = sfmt(a, "%.*s\n\nCommand exited with code %d", (int)text.len, text.ptr, exit_code);
        o->is_error = true;
    }
    out_text(a, o, text, details);
    o->structured = sc;
    buf_free(&acc.raw);
    buf_free(&acc.tail);
}

// The user's own commands (rpc mode's bash, AgentSession.executeBash): pi's
// createLocalShellOperations().exec (core/tools/bash.ts) without a timeout, the loop of tool_bash.
int code_tools_exec(Arena *a, const CodeEnv *env, const char *command, void (*on_data)(void *ctx, const void *p, size_t n), void *ctx,
                    volatile int *abort_flag, bool *aborted, const char **thrown) {
    *aborted = false;
    *thrown = NULL;
    if (access(env->cwd, F_OK)) {
        *thrown = arena_printf(a, "Working directory does not exist: %s\nCannot execute bash commands.", env->cwd);
        return 0;
    }
    int out[2] = {-1, -1}, err[2] = {-1, -1};
    pid_t pid = -1;
    int rc = pipe(out) ? errno : pipe(err) ? errno : 0;
    if (!rc) {
        fcntl(out[0], F_SETFD, FD_CLOEXEC);
        fcntl(err[0], F_SETFD, FD_CLOEXEC);
        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
        posix_spawn_file_actions_adddup2(&fa, out[1], 1);
        posix_spawn_file_actions_adddup2(&fa, err[1], 2);
        posix_spawn_file_actions_addchdir(&fa, env->cwd);
        posix_spawnattr_t at;
        posix_spawnattr_init(&at);
        sigset_t none, all;
        sigemptyset(&none);
        sigfillset(&all);
        posix_spawnattr_setsigmask(&at, &none);
        posix_spawnattr_setsigdefault(&at, &all);
        posix_spawnattr_setflags(&at, POSIX_SPAWN_SETSID | POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_CLOEXEC_DEFAULT);
        const char *sh = shell_path();
        const char *argv[] = {sh, "-c", command, NULL};
        rc = posix_spawnp(&pid, sh, &fa, &at, (char *const *)argv, shell_env(a, env));
        posix_spawn_file_actions_destroy(&fa);
        posix_spawnattr_destroy(&at);
    }
    if (out[1] >= 0) close(out[1]);
    if (err[1] >= 0) close(err[1]);
    if (rc) {
        if (out[0] >= 0) close(out[0]);
        if (err[0] >= 0) close(err[0]);
        *thrown = arena_printf(a, "spawn %s %s", shell_path(), errno_code(rc));
        return 0;
    }
    track_group(pid);
    int st = 0;
    bool exited = false, killed = false;
    struct pollfd pf[2] = {{out[0], POLLIN, 0}, {err[0], POLLIN, 0}};
    int nopen = 2;
    uint64_t t0 = now_ns();
    double idle_until = -1;
    char buf[65536];
    for (;;) {
        if (!exited) {
            pid_t w = waitpid(pid, &st, WNOHANG);
            if (w == pid) {
                exited = true;
                idle_until = ms_since(t0) + 100;
            }
        }
        if (exited && (nopen == 0 || ms_since(t0) >= idle_until)) break;
        if (abort_flag && *abort_flag && !killed) { // the abort signal: the whole tree killed
            killed = true;
            kill_tree(pid);
        }
        int n = poll(pf, 2, (int)ABORT_POLL_MS);
        if (n < 0 && errno != EINTR) break;
        for (int k = 0; k < 2 && n > 0; k++) {
            if (pf[k].fd < 0 || !(pf[k].revents & (POLLIN | POLLHUP | POLLERR))) continue;
            ssize_t r = read(pf[k].fd, buf, sizeof buf);
            if (r < 0 && (errno == EINTR || errno == EAGAIN)) continue;
            if (r <= 0) {
                close(pf[k].fd);
                pf[k].fd = -1;
                nopen--;
                continue;
            }
            if (on_data) on_data(ctx, buf, (size_t)r);
            if (exited) idle_until = ms_since(t0) + 100;
        }
    }
    if (!exited) {
        kill_tree(pid);
        reap(pid, &st);
    }
    for (int k = 0; k < 2; k++)
        if (pf[k].fd >= 0) close(pf[k].fd);
    untrack_group(pid);
    if (abort_flag && *abort_flag) *aborted = true;
    return WIFEXITED(st) ? WEXITSTATUS(st) : WIFSIGNALED(st) ? 128 + WTERMSIG(st) : 1;
}

// truncateTail with the defaults (2000 lines, 50 KB).
Str code_tools_truncate_tail(Str s, bool *truncated) {
    Trunc t = truncate_tail(s, MAX_LINES, MAX_BYTES);
    *truncated = t.truncated;
    return t.content;
}

// ---------------------------------------------------------------------------------------
// line diff (jsdiff 8.0.4 diffLines, structuredPatch, formatPatch)
// ---------------------------------------------------------------------------------------

typedef struct {
    Str value;
    size_t count;
    bool added, removed;
} Change;
typedef VEC(Change) VecChange;

typedef struct Comp {
    size_t count;
    bool added, removed;
    struct Comp *prev;
} Comp;

typedef struct {
    ptrdiff_t old_pos;
    Comp *last;
} DPath;

static void tokenize_lines(Str s, VecStr *out) {
    size_t start = 0;
    while (start < s.len) {
        ptrdiff_t nl = schr(s, '\n', start);
        size_t end = nl == -1 ? s.len : (size_t)nl + 1;
        vec_push(out, str_n(s.ptr + start, end - start));
        start = end;
    }
}

static DPath *dpath(Arena *a, ptrdiff_t old_pos, Comp *last) {
    DPath *p = ARENA_NEW(a, DPath);
    p->old_pos = old_pos;
    p->last = last;
    return p;
}

static Comp *comp(Arena *a, size_t count, bool added, bool removed, Comp *prev) {
    Comp *c = ARENA_NEW(a, Comp);
    c->count = count;
    c->added = added;
    c->removed = removed;
    c->prev = prev;
    return c;
}

static DPath *add_to_path(Arena *a, DPath *p, bool added, bool removed, ptrdiff_t inc) {
    Comp *last = p->last;
    if (last && last->added == added && last->removed == removed)
        return dpath(a, p->old_pos + inc, comp(a, last->count + 1, added, removed, last->prev));
    return dpath(a, p->old_pos + inc, comp(a, 1, added, removed, last));
}

static ptrdiff_t extract_common(Arena *a, DPath *b, const VecStr *nt, const VecStr *ot, ptrdiff_t diag) {
    ptrdiff_t nl = (ptrdiff_t)nt->len, ol = (ptrdiff_t)ot->len;
    ptrdiff_t op = b->old_pos, np = op - diag;
    size_t common = 0;
    while (np + 1 < nl && op + 1 < ol && str_eq(ot->ptr[op + 1], nt->ptr[np + 1])) { np++; op++; common++; }
    if (common) b->last = comp(a, common, false, false, b->last);
    b->old_pos = op;
    return np;
}

static void build_values(Comp *last, const VecStr *nt, const VecStr *ot, VecChange *out) {
    size_t n = 0;
    for (Comp *c = last; c; c = c->prev) n++;
    Comp **list = xmalloc((n ? n : 1) * sizeof *list);
    size_t k = n;
    for (Comp *c = last; c; c = c->prev) list[--k] = c;
    size_t np = 0, op = 0;
    for (size_t i = 0; i < n; i++) {
        Comp *c = list[i];
        Change ch = {.count = c->count, .added = c->added, .removed = c->removed};
        if (!c->removed) {
            ch.value = span(nt, np, np + c->count);
            np += c->count;
            if (!c->added) op += c->count;
        } else {
            ch.value = span(ot, op, op + c->count);
            op += c->count;
        }
        vec_push(out, ch);
    }
    free(list);
}

static void diff_lines(Arena *a, Str olds, Str news, VecChange *out) {
    VecStr ot = {0}, nt = {0};
    tokenize_lines(olds, &ot);
    tokenize_lines(news, &nt);
    ptrdiff_t nl = (ptrdiff_t)nt.len, ol = (ptrdiff_t)ot.len;
    ptrdiff_t max_edit = nl + ol;
    ptrdiff_t off = max_edit + 2;
    DPath **best = xcalloc((size_t)(2 * off + 1), sizeof *best);
    best[off] = dpath(a, -1, NULL);
    ptrdiff_t np = extract_common(a, best[off], &nt, &ot, 0);
    if (best[off]->old_pos + 1 >= ol && np + 1 >= nl) {
        build_values(best[off]->last, &nt, &ot, out);
        goto done;
    }
    ptrdiff_t min_d = PTRDIFF_MIN / 2, max_d = PTRDIFF_MAX / 2;
    for (ptrdiff_t el = 1; el <= max_edit; el++) {
        for (ptrdiff_t d = MAX(min_d, -el); d <= MIN(max_d, el); d += 2) {
            DPath *rem = best[off + d - 1], *add = best[off + d + 1];
            if (rem) best[off + d - 1] = NULL;
            bool can_add = false;
            if (add) {
                ptrdiff_t anp = add->old_pos - d;
                can_add = 0 <= anp && anp < nl;
            }
            bool can_rem = rem && rem->old_pos + 1 < ol;
            if (!can_add && !can_rem) { best[off + d] = NULL; continue; }
            DPath *base;
            if (!can_rem || (can_add && rem->old_pos < add->old_pos)) base = add_to_path(a, add, true, false, 0);
            else base = add_to_path(a, rem, false, true, 1);
            np = extract_common(a, base, &nt, &ot, d);
            if (base->old_pos + 1 >= ol && np + 1 >= nl) {
                build_values(base->last, &nt, &ot, out);
                goto done;
            }
            best[off + d] = base;
            if (base->old_pos + 1 >= ol) max_d = MIN(max_d, d - 1);
            if (np + 1 >= nl) min_d = MAX(min_d, d + 1);
        }
    }
done:
    free(best);
    vec_free(&ot);
    vec_free(&nt);
}

// value.split("\n") without a final empty piece
static void raw_lines(Str v, VecStr *out) {
    split_lines(v, out);
    if (out->len && out->ptr[out->len - 1].len == 0) out->len--;
}

static Str pad_num(Arena *a, size_t n, int width) { return sfmt(a, "%*zu", width, n); }

// edit-diff.ts formatDisplayDiff
static void format_display(Arena *a, const VecChange *parts, size_t old_line, size_t new_line, int width, size_t ctx, VecStr *out, size_t *first) {
    bool last_change = false;
    Str skip = sfmt(a, " %*s ...", width, "");
    for (size_t i = 0; i < parts->len; i++) {
        const Change *p = &parts->ptr[i];
        VecStr raw = {0};
        raw_lines(p->value, &raw);
        if (p->added || p->removed) {
            if (!*first) *first = new_line;
            for (size_t k = 0; k < raw.len; k++) {
                if (p->added) vec_push(out, sfmt(a, "+%s %.*s", pad_num(a, new_line++, width).ptr, (int)raw.ptr[k].len, raw.ptr[k].ptr));
                else vec_push(out, sfmt(a, "-%s %.*s", pad_num(a, old_line++, width).ptr, (int)raw.ptr[k].len, raw.ptr[k].ptr));
            }
            last_change = true;
        } else {
            bool next_change = i + 1 < parts->len && (parts->ptr[i + 1].added || parts->ptr[i + 1].removed);
#define CTXLINE(k) do { vec_push(out, sfmt(a, " %s %.*s", pad_num(a, old_line, width).ptr, (int)raw.ptr[k].len, raw.ptr[k].ptr)); old_line++; new_line++; } while (0)
            if (last_change && next_change) {
                if (raw.len <= ctx * 2) {
                    for (size_t k = 0; k < raw.len; k++) CTXLINE(k);
                } else {
                    for (size_t k = 0; k < ctx; k++) CTXLINE(k);
                    vec_push(out, skip);
                    size_t skipped = raw.len - 2 * ctx;
                    old_line += skipped;
                    new_line += skipped;
                    for (size_t k = raw.len - ctx; k < raw.len; k++) CTXLINE(k);
                }
            } else if (last_change) {
                size_t shown = MIN(ctx, raw.len);
                for (size_t k = 0; k < shown; k++) CTXLINE(k);
                if (raw.len > shown) {
                    vec_push(out, skip);
                    old_line += raw.len - shown;
                    new_line += raw.len - shown;
                }
            } else if (next_change) {
                size_t skipped = raw.len > ctx ? raw.len - ctx : 0;
                if (skipped) {
                    vec_push(out, skip);
                    old_line += skipped;
                    new_line += skipped;
                }
                for (size_t k = skipped; k < raw.len; k++) CTXLINE(k);
            } else {
                old_line += raw.len;
                new_line += raw.len;
            }
#undef CTXLINE
            last_change = false;
        }
        vec_free(&raw);
    }
}

typedef struct {
    ptrdiff_t old_start, old_lines, new_start, new_lines;
    VecStr lines;
} Hunk;
typedef VEC(Hunk) VecHunk;

// create.js splitLines: lines with their "\n"
static void patch_lines(Str v, VecStr *out) {
    bool trailing = v.len && v.ptr[v.len - 1] == '\n';
    size_t start = 0;
    for (;;) {
        ptrdiff_t nl = schr(v, '\n', start);
        if (nl == -1) {
            if (!trailing) vec_push(out, str_n(v.ptr + start, v.len - start));
            return;
        }
        vec_push(out, str_n(v.ptr + start, (size_t)nl + 1 - start));
        start = (size_t)nl + 1;
    }
}

static void structured_patch(Arena *a, Str olds, Str news, ptrdiff_t ctx, VecHunk *hunks) {
    VecChange diff = {0};
    diff_lines(a, olds, news, &diff);
    vec_push(&diff, ((Change){.value = {"", 0}}));
    VecStr *lines = xcalloc(diff.len, sizeof *lines);
    for (size_t i = 0; i + 1 < diff.len; i++) patch_lines(diff.ptr[i].value, &lines[i]);
    ptrdiff_t ors = 0, nrs = 0, ol = 1, nl = 1;
    VecStr cur = {0};
    for (size_t i = 0; i < diff.len; i++) {
        const Change *c = &diff.ptr[i];
        VecStr *ls = &lines[i];
        if (c->added || c->removed) {
            if (!ors) {
                ors = ol;
                nrs = nl;
                if (i > 0) {
                    VecStr *pl = &lines[i - 1];
                    size_t from = ctx > 0 ? (pl->len > (size_t)ctx ? pl->len - (size_t)ctx : 0) : pl->len;
                    for (size_t k = from; k < pl->len; k++) vec_push(&cur, sfmt(a, " %.*s", (int)pl->ptr[k].len, pl->ptr[k].ptr));
                    ors -= (ptrdiff_t)(pl->len - from);
                    nrs -= (ptrdiff_t)(pl->len - from);
                }
            }
            for (size_t k = 0; k < ls->len; k++) vec_push(&cur, sfmt(a, "%c%.*s", c->added ? '+' : '-', (int)ls->ptr[k].len, ls->ptr[k].ptr));
            if (c->added) nl += (ptrdiff_t)ls->len;
            else ol += (ptrdiff_t)ls->len;
        } else {
            if (ors) {
                if ((ptrdiff_t)ls->len <= ctx * 2 && i + 2 < diff.len) {
                    for (size_t k = 0; k < ls->len; k++) vec_push(&cur, sfmt(a, " %.*s", (int)ls->ptr[k].len, ls->ptr[k].ptr));
                } else {
                    size_t cs = MIN(ls->len, (size_t)ctx);
                    for (size_t k = 0; k < cs; k++) vec_push(&cur, sfmt(a, " %.*s", (int)ls->ptr[k].len, ls->ptr[k].ptr));
                    Hunk h = {ors, ol - ors + (ptrdiff_t)cs, nrs, nl - nrs + (ptrdiff_t)cs, cur};
                    vec_push(hunks, h);
                    ors = nrs = 0;
                    cur = (VecStr){0};
                }
            }
            ol += (ptrdiff_t)ls->len;
            nl += (ptrdiff_t)ls->len;
        }
    }
    vec_free(&cur);
    for (size_t i = 0; i < diff.len; i++) vec_free(&lines[i]);
    free(lines);
    vec_free(&diff);
    for (size_t h = 0; h < hunks->len; h++) {
        VecStr *hl = &hunks->ptr[h].lines;
        VecStr fixed = {0};
        for (size_t i = 0; i < hl->len; i++) {
            Str l = hl->ptr[i];
            if (l.len && l.ptr[l.len - 1] == '\n') vec_push(&fixed, str_n(l.ptr, l.len - 1));
            else {
                vec_push(&fixed, l);
                vec_push(&fixed, S("\\ No newline at end of file"));
            }
        }
        vec_free(hl);
        *hl = fixed;
    }
}

// ---------------------------------------------------------------------------------------
// edit (edit.ts, edit-diff.ts)
// ---------------------------------------------------------------------------------------

typedef struct {
    size_t edit_index, index, len;
    Str new_text;
} Repl;
typedef VEC(Repl) VecRepl;

static Str normalize_lf(Arena *a, Str s) {
    if (!memchr(s.ptr, '\r', s.len)) return s;
    Buf b = {0};
    for (size_t i = 0; i < s.len; i++) {
        if (s.ptr[i] == '\r') {
            buf_addc(&b, '\n');
            if (i + 1 < s.len && s.ptr[i + 1] == '\n') i++;
        } else buf_addc(&b, s.ptr[i]);
    }
    return take(a, &b);
}

static bool js_trim_space(uint32_t cp) {
    return cp == 9 || cp == 10 || cp == 11 || cp == 12 || cp == 13 || cp == 32 || cp == 0xa0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200a) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202f || cp == 0x205f || cp == 0x3000 || cp == 0xfeff;
}

// normalizeForFuzzyMatch
static Str fuzzy_norm(Arena *a, Str s) {
    s = normalize_form(a, s, true);
    Buf b = {0};
    size_t start = 0;
    for (;;) {
        ptrdiff_t nl = schr(s, '\n', start);
        Str line = str_n(s.ptr + start, (nl == -1 ? s.len : (size_t)nl) - start);
        // trimEnd: drop trailing whitespace code points
        size_t end = 0, i = 0;
        while (i < line.len) {
            size_t j = i;
            uint32_t cp = u8_next(line, &j);
            if (!js_trim_space(cp)) end = j;
            i = j;
        }
        for (size_t k = 0; k < end;) {
            size_t j = k;
            uint32_t cp = u8_next(line, &j);
            if (cp >= 0x2018 && cp <= 0x201b) buf_addc(&b, '\'');
            else if (cp >= 0x201c && cp <= 0x201f) buf_addc(&b, '"');
            else if ((cp >= 0x2010 && cp <= 0x2015) || cp == 0x2212) buf_addc(&b, '-');
            else if (cp == 0xa0 || (cp >= 0x2002 && cp <= 0x200a) || cp == 0x202f || cp == 0x205f || cp == 0x3000) buf_addc(&b, ' ');
            else buf_add(&b, line.ptr + k, j - k);
            k = j;
        }
        if (nl == -1) break;
        buf_addc(&b, '\n');
        start = (size_t)nl + 1;
    }
    return take(a, &b);
}

typedef struct {
    bool found, fuzzy;
    size_t index, len;
} Match;

static Match fuzzy_find(Arena *a, Str content, Str old) {
    ptrdiff_t i = sfind(content, old, 0);
    if (i != -1) return (Match){true, false, (size_t)i, old.len};
    Str fc = fuzzy_norm(a, content), fo = fuzzy_norm(a, old);
    i = sfind(fc, fo, 0);
    if (i == -1) return (Match){0};
    return (Match){true, true, (size_t)i, fo.len};
}

static ptrdiff_t count_occurrences(Arena *a, Str content, Str old) {
    Str fc = fuzzy_norm(a, content), fo = fuzzy_norm(a, old);
    if (!fo.len) return fc.len ? (ptrdiff_t)utf16_len(fc) - 1 : -1;
    ptrdiff_t n = 0;
    for (ptrdiff_t i = sfind(fc, fo, 0); i != -1; i = sfind(fc, fo, (size_t)i + fo.len)) n++;
    return n;
}

typedef struct {
    size_t start, end;
} Span;
typedef VEC(Span) VecSpan;

static void line_spans(Str s, VecSpan *out) {
    size_t start = 0;
    while (start < s.len) {
        ptrdiff_t nl = schr(s, '\n', start);
        size_t end = nl == -1 ? s.len : (size_t)nl + 1;
        vec_push(out, ((Span){start, end}));
        start = end;
    }
}

static Str apply_repls(Arena *a, Str content, const Repl *r, size_t n, size_t offset) {
    Buf b = {0};
    size_t pos = 0;
    for (size_t i = 0; i < n; i++) { // replacements are sorted and disjoint: forward equals reverse
        size_t at = r[i].index - offset;
        buf_add(&b, content.ptr + pos, at - pos);
        buf_addstr(&b, r[i].new_text);
        pos = at + r[i].len;
    }
    buf_add(&b, content.ptr + pos, content.len - pos);
    return take(a, &b);
}

static int repl_cmp(const void *x, const void *y) {
    const Repl *a = x, *b = y;
    if (a->index != b->index) return a->index < b->index ? -1 : 1;
    return a->edit_index < b->edit_index ? -1 : a->edit_index > b->edit_index;
}

// widenReplacementsToLines; NULL message on success
static const char *widen(Arena *a, Str orig, Str base, VecRepl *r) {
    VecSpan ol = {0}, bl = {0};
    line_spans(orig, &ol);
    line_spans(base, &bl);
    if (ol.len != bl.len) { vec_free(&ol); vec_free(&bl); return "Cannot preserve unchanged lines because the base content has a different line count."; }
    VecRepl out = {0};
    size_t gs = 0, ge = 0, gfirst = 0;
    bool have = false;
    for (size_t i = 0; i <= r->len; i++) {
        size_t s = 0, e = 0;
        if (i < r->len) {
            size_t rs = r->ptr[i].index, re = r->ptr[i].index + r->ptr[i].len;
            ptrdiff_t sl = -1;
            for (size_t k = 0; k < bl.len; k++)
                if (rs >= bl.ptr[k].start && rs < bl.ptr[k].end) { sl = (ptrdiff_t)k; break; }
            if (sl == -1) { vec_free(&ol); vec_free(&bl); vec_free(&out); return "Replacement range is outside the base content."; }
            size_t el = (size_t)sl;
            while (el < bl.len && bl.ptr[el].end < re) el++;
            if (el >= bl.len) { vec_free(&ol); vec_free(&bl); vec_free(&out); return "Replacement range is outside the base content."; }
            s = (size_t)sl;
            e = el + 1;
            if (have && s < ge) {
                ge = MAX(ge, e);
                continue;
            }
        }
        if (have) {
            size_t go = bl.ptr[gs].start, gend = bl.ptr[ge - 1].end;
            Repl w = {.index = ol.ptr[gs].start, .len = ol.ptr[ge - 1].end - ol.ptr[gs].start};
            w.new_text = apply_repls(a, str_n(base.ptr + go, gend - go), r->ptr + gfirst, i - gfirst, go);
            vec_push(&out, w);
        }
        have = true;
        gs = s;
        ge = e;
        gfirst = i;
    }
    vec_free(&ol);
    vec_free(&bl);
    vec_free(r);
    *r = out;
    return NULL;
}

// applyEditsToNormalizedContent: NULL message on success
static const char *apply_edits(Arena *a, Str content, const JValue *edits, Str path, VecRepl *repls, Str *new_content) {
    size_t n = edits->arr.len;
    Str *olds = ARENA_ARRAY(a, Str, n), *news = ARENA_ARRAY(a, Str, n);
    for (size_t i = 0; i < n; i++) {
        olds[i] = normalize_lf(a, json_strs(json_get(edits->arr.items[i], "oldText")));
        news[i] = normalize_lf(a, json_strs(json_get(edits->arr.items[i], "newText")));
    }
    for (size_t i = 0; i < n; i++)
        if (!olds[i].len)
            return n == 1 ? arena_printf(a, "oldText must not be empty in %.*s.", (int)path.len, path.ptr)
                          : arena_printf(a, "edits[%zu].oldText must not be empty in %.*s.", i, (int)path.len, path.ptr);
    bool fuzzy = false;
    for (size_t i = 0; i < n; i++)
        if (fuzzy_find(a, content, olds[i]).fuzzy) fuzzy = true;
    Str base = fuzzy ? fuzzy_norm(a, content) : content;
    for (size_t i = 0; i < n; i++) {
        Match m = fuzzy_find(a, base, olds[i]);
        if (!m.found)
            return n == 1 ? arena_printf(a, "Could not find the exact text in %.*s. The old text must match exactly including all whitespace and newlines.", (int)path.len, path.ptr)
                          : arena_printf(a, "Could not find edits[%zu] in %.*s. The oldText must match exactly including all whitespace and newlines.", i, (int)path.len, path.ptr);
        ptrdiff_t occ = count_occurrences(a, base, olds[i]);
        if (occ > 1)
            return n == 1 ? arena_printf(a, "Found %td occurrences of the text in %.*s. The text must be unique. Please provide more context to make it unique.", occ, (int)path.len, path.ptr)
                          : arena_printf(a, "Found %td occurrences of edits[%zu] in %.*s. Each oldText must be unique. Please provide more context to make it unique.", occ, i, (int)path.len, path.ptr);
        vec_push(repls, ((Repl){i, m.index, m.len, news[i]}));
    }
    qsort(repls->ptr, repls->len, sizeof(Repl), repl_cmp);
    for (size_t i = 1; i < repls->len; i++) {
        Repl *p = &repls->ptr[i - 1], *c = &repls->ptr[i];
        if (p->index + p->len > c->index)
            return arena_printf(a, "edits[%zu] and edits[%zu] overlap in %.*s. Merge them into one edit or target disjoint regions.", p->edit_index, c->edit_index, (int)path.len, path.ptr);
    }
    if (fuzzy) {
        const char *e = widen(a, content, base, repls);
        if (e) return e;
    }
    *new_content = apply_repls(a, content, repls->ptr, repls->len, 0);
    if (str_eq(content, *new_content))
        return n == 1 ? arena_printf(a, "No changes made to %.*s. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected.", (int)path.len, path.ptr)
                      : arena_printf(a, "No changes made to %.*s. The replacements produced identical content.", (int)path.len, path.ptr);
    return NULL;
}

static size_t line_start_before(Str t, size_t off) { return off == 0 ? 0 : (size_t)(srchr(t, '\n', (ptrdiff_t)off - 1) + 1); }

typedef struct {
    size_t old_start, old_end, old_line, new_start, new_end, new_line;
} Window;

static JValue *edit_diff(Arena *a, Str path, Str base, Str news, const VecRepl *r) {
    enum { CTX = 4, MARGIN = 32 };
    size_t lines = MAX(count_char(base, '\n', 0, base.len), count_char(news, '\n', 0, news.len)) + 1;
    int width = snprintf(NULL, 0, "%zu", lines);
    Str skip = sfmt(a, " %*s ...", width, "");
    // getDiffWindows
    VEC(Window) wins = {0};
    ptrdiff_t off_delta = 0, line_delta = 0;
    size_t cur_off = 0, cur_line = 0;
    for (size_t i = 0; i < r->len; i++) {
        const Repl *rp = &r->ptr[i];
        size_t me = rp->index + rp->len;
        size_t start = line_start_before(base, rp->index);
        for (int k = 0; k < MARGIN && start > 0; k++) start = line_start_before(base, start - 1);
        size_t end = me;
        for (int k = 0; k <= MARGIN && end < base.len; k++) {
            ptrdiff_t lf = schr(base, '\n', k == 0 ? MAX(me > 0 ? me - 1 : 0, rp->index) : end);
            end = lf == -1 ? base.len : (size_t)lf + 1;
        }
        ptrdiff_t rod = (ptrdiff_t)rp->new_text.len - (ptrdiff_t)rp->len;
        ptrdiff_t rld = (ptrdiff_t)count_char(rp->new_text, '\n', 0, rp->new_text.len) - (ptrdiff_t)count_char(base, '\n', rp->index, me);
        if (wins.len && start <= wins.ptr[wins.len - 1].old_end) {
            Window *pw = &wins.ptr[wins.len - 1];
            pw->old_end = MAX(pw->old_end, end);
            pw->new_end = (size_t)((ptrdiff_t)pw->old_end + off_delta + rod);
        } else {
            cur_line += count_char(base, '\n', cur_off, start);
            cur_off = start;
            Window w = {start, end, cur_line, (size_t)((ptrdiff_t)start + off_delta), (size_t)((ptrdiff_t)end + off_delta + rod),
                        (size_t)((ptrdiff_t)cur_line + line_delta)};
            vec_push(&wins, w);
        }
        off_delta += rod;
        line_delta += rld;
    }
    VecStr out = {0};
    VecHunk hunks = {0};
    size_t first = 0;
    for (size_t i = 0; i < wins.len; i++) {
        Window *w = &wins.ptr[i];
        Str ot = str_n(base.ptr + w->old_start, w->old_end - w->old_start);
        Str nt = str_n(news.ptr + w->new_start, w->new_end - w->new_start);
        VecChange parts = {0};
        diff_lines(a, ot, nt, &parts);
        VecStr shown = {0};
        size_t f = 0;
        format_display(a, &parts, w->old_line + 1, w->new_line + 1, width, CTX, &shown, &f);
        if (!first) first = f;
        size_t from = 0;
        if (out.len && str_eq(out.ptr[out.len - 1], skip) && shown.len && str_eq(shown.ptr[0], skip)) from = 1;
        for (size_t k = from; k < shown.len; k++) vec_push(&out, shown.ptr[k]);
        vec_free(&shown);
        vec_free(&parts);
        VecHunk hs = {0};
        structured_patch(a, ot, nt, CTX, &hs);
        for (size_t k = 0; k < hs.len; k++) {
            hs.ptr[k].old_start += (ptrdiff_t)w->old_line;
            hs.ptr[k].new_start += (ptrdiff_t)w->new_line;
            vec_push(&hunks, hs.ptr[k]);
        }
        vec_free(&hs);
    }
    Buf d = {0};
    for (size_t i = 0; i < out.len; i++) {
        if (i) buf_addc(&d, '\n');
        buf_addstr(&d, out.ptr[i]);
    }
    Buf p = {0};
    buf_printf(&p, "--- %.*s\n+++ %.*s", (int)path.len, path.ptr, (int)path.len, path.ptr);
    for (size_t i = 0; i < hunks.len; i++) {
        Hunk *h = &hunks.ptr[i];
        if (h->old_lines == 0) h->old_start--;
        if (h->new_lines == 0) h->new_start--;
        buf_printf(&p, "\n@@ -%td,%td +%td,%td @@", h->old_start, h->old_lines, h->new_start, h->new_lines);
        for (size_t k = 0; k < h->lines.len; k++) {
            buf_addc(&p, '\n');
            buf_addstr(&p, h->lines.ptr[k]);
        }
        vec_free(&h->lines);
    }
    buf_addc(&p, '\n');
    vec_free(&hunks);
    vec_free(&out);
    vec_free(&wins);
    JValue *o = json_new(a, J_OBJECT);
    jput(a, o, "diff", jstr(a, take(a, &d)));
    jput(a, o, "patch", jstr(a, take(a, &p)));
    if (first) jput(a, o, "firstChangedLine", jnum(a, (double)first));
    return o;
}

static const char *line_ending_at(Str c, size_t off) {
    ptrdiff_t lf = schr(c, '\n', off);
    if (lf == -1) lf = srchr(c, '\n', (ptrdiff_t)off);
    if (lf == -1) return memchr(c.ptr, '\r', c.len) ? "\r" : "\n";
    return lf > 0 && c.ptr[lf - 1] == '\r' ? "\r\n" : "\n";
}

// applyReplacementsToOriginal
static Str apply_to_original(Arena *a, Str orig, const VecRepl *r) {
    size_t crlf_before = 0;
    ptrdiff_t next = sfind(orig, S("\r\n"), 0);
    Buf b = {0};
    bool ends_cr = false;
#define TO_ORIG(o) ({ size_t o_ = (o); while (next != -1 && (size_t)next - crlf_before < o_) { crlf_before++; next = sfind(orig, S("\r\n"), (size_t)next + 2); } o_ + crlf_before; })
#define PUSH(s) do { Str s_ = (s); if (s_.len) { if (ends_cr && s_.ptr[0] == '\n') buf_addc(&b, '\n'); buf_addstr(&b, s_); ends_cr = s_.ptr[s_.len - 1] == '\r'; } } while (0)
    size_t pos = 0;
    for (size_t i = 0; i < r->len; i++) {
        size_t start = TO_ORIG(r->ptr[i].index);
        PUSH(str_n(orig.ptr + pos, start - pos));
        const char *le = line_ending_at(orig, start);
        if (!strcmp(le, "\n")) PUSH(r->ptr[i].new_text);
        else {
            Buf t = {0};
            Str nt = r->ptr[i].new_text;
            for (size_t k = 0; k < nt.len; k++) {
                if (nt.ptr[k] == '\n') buf_adds(&t, le);
                else buf_addc(&t, nt.ptr[k]);
            }
            PUSH(take(a, &t));
        }
        pos = TO_ORIG(r->ptr[i].index + r->ptr[i].len);
    }
    PUSH(str_n(orig.ptr + pos, orig.len - pos));
#undef TO_ORIG
#undef PUSH
    return take(a, &b);
}

static const char *write_text(Arena *a, const char *abs, Str content) {
    int fd = open(abs, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
    if (fd < 0) return node_err(a, errno, "open", abs);
    const char *p = content.ptr;
    size_t n = content.len;
    while (n) {
        ssize_t w = write(fd, p, n);
        if (w < 0 && errno == EINTR) continue;
        if (w < 0) {
            int e = errno;
            close(fd);
            return node_err(a, e, "write", NULL);
        }
        p += w;
        n -= (size_t)w;
    }
    close(fd);
    return NULL;
}

static void tool_edit(Arena *a, const CodeEnv *env, const JValue *args, ToolOut *o) {
    Str path = json_strs(json_get(args, "path"));
    const JValue *edits = json_get(args, "edits");
    if (!json_is(edits, J_ARRAY) || edits->arr.len == 0) { o->thrown = "Edit tool input is invalid. edits must contain at least one replacement."; return; }
    char *abs = resolve_to_cwd(a, env, path);
    if (access(abs, R_OK | W_OK)) {
        o->thrown = arena_printf(a, "Could not edit file: %.*s. Error code: %s.", (int)path.len, path.ptr, errno_code(errno));
        return;
    }
    int fd = open(abs, O_RDONLY | O_CLOEXEC);
    if (fd < 0) { o->thrown = node_err(a, errno, "open", abs); return; }
    Buf data = {0};
    int e = read_all(fd, &data);
    close(fd);
    if (e) { buf_free(&data); o->thrown = node_err(a, e, "read", abs); return; }
    Str raw = u8_text(a, data.ptr, data.len, true);
    buf_free(&data);
    Str bom = {"", 0}, content = raw;
    if (str_starts(raw, S("\xef\xbb\xbf"))) { bom = str_n(raw.ptr, 3); content = str_n(raw.ptr + 3, raw.len - 3); }
    Str norm = normalize_lf(a, content);
    VecRepl r = {0};
    Str news = {0};
    const char *err = apply_edits(a, norm, edits, path, &r, &news);
    if (err) { vec_free(&r); o->thrown = err; return; }
    JValue *details = edit_diff(a, path, norm, news, &r);
    Str final = str_eq(content, norm) ? news : apply_to_original(a, content, &r);
    vec_free(&r);
    Buf fb = {0};
    buf_addstr(&fb, bom);
    buf_addstr(&fb, final);
    err = write_text(a, abs, buf_str(&fb));
    buf_free(&fb);
    if (err) { o->thrown = err; return; }
    out_text(a, o, sfmt(a, "Successfully replaced %zu block(s) in %.*s.", edits->arr.len, (int)path.len, path.ptr), details);
}

// prepareEditArguments: edits as a JSON string, a single edit object, or top-level oldText/newText
static bool single_edit(const JValue *v) {
    return json_is(v, J_OBJECT) && json_is(json_get(v, "oldText"), J_STRING) && json_is(json_get(v, "newText"), J_STRING);
}

static JValue *prepare_edit(Arena *a, JValue *args) {
    if (!json_is(args, J_OBJECT)) return args;
    JValue *edits = json_get(args, "edits");
    JValue *res = args;
    if (json_is(edits, J_STRING)) {
        JValue *parsed;
        if (!json_parse(a, edits->s, true, &parsed)) {
            if (json_is(parsed, J_ARRAY)) { res = obj_copy(a, args); json_put(a, res, S("edits"), parsed); }
            else if (single_edit(parsed)) {
                JValue *arr = json_new(a, J_ARRAY);
                json_push(a, arr, parsed);
                res = obj_copy(a, args);
                json_put(a, res, S("edits"), arr);
            }
        }
    } else if (single_edit(edits)) {
        JValue *arr = json_new(a, J_ARRAY);
        json_push(a, arr, edits);
        res = obj_copy(a, args);
        json_put(a, res, S("edits"), arr);
    }
    JValue *ot = json_get(res, "oldText"), *nt = json_get(res, "newText");
    if (!json_is(ot, J_STRING) || !json_is(nt, J_STRING)) return res;
    JValue *list = json_is(json_get(res, "edits"), J_ARRAY) ? arr_copy(a, json_get(res, "edits")) : json_new(a, J_ARRAY);
    JValue *one = json_new(a, J_OBJECT);
    jput(a, one, "oldText", ot);
    jput(a, one, "newText", nt);
    json_push(a, list, one);
    JValue *out = json_new(a, J_OBJECT);
    for (size_t i = 0; i < res->obj.len; i++) {
        Str k = res->obj.items[i].key;
        if (str_eqz(k, "oldText") || str_eqz(k, "newText")) continue;
        json_put(a, out, k, res->obj.items[i].val);
    }
    json_put(a, out, S("edits"), list);
    return out;
}

// ---------------------------------------------------------------------------------------
// write (write.ts)
// ---------------------------------------------------------------------------------------

// fs.mkdir(dir, {recursive: true})
static const char *mkdir_rec(Arena *a, const char *dir) {
    struct stat st;
    if (stat(dir, &st) == 0) return S_ISDIR(st.st_mode) ? NULL : node_err(a, EEXIST, "mkdir", dir);
    if (errno == ENOTDIR) return node_err(a, ENOTDIR, "mkdir", dir);
    const char *parent = path_dirname(a, dir);
    if (strcmp(parent, dir)) {
        const char *e = mkdir_rec(a, parent);
        if (e) return node_err(a, strstr(e, "ENOTDIR") || strstr(e, "EEXIST") ? ENOTDIR : EACCES, "mkdir", dir);
    }
    if (mkdir(dir, 0777) && errno != EEXIST) return node_err(a, errno, "mkdir", dir);
    return NULL;
}

static void tool_write(Arena *a, const CodeEnv *env, const JValue *args, ToolOut *o) {
    Str path = json_strs(json_get(args, "path"));
    Str content = json_strs(json_get(args, "content"));
    char *abs = resolve_to_cwd(a, env, path);
    const char *e = mkdir_rec(a, path_dirname(a, abs));
    if (!e) e = write_text(a, abs, content);
    if (e) { o->thrown = e; return; }
    out_text(a, o, sfmt(a, "Successfully wrote to %.*s", (int)path.len, path.ptr), NULL);
}

// ---------------------------------------------------------------------------------------
// grep (grep.ts)
// ---------------------------------------------------------------------------------------

static Str rg_text(Arena *a, const JValue *d) {
    const JValue *t = json_get(d, "text");
    if (json_is(t, J_STRING)) return t->s;
    const JValue *b = json_get(d, "bytes");
    if (!json_is(b, J_STRING)) return (Str){"", 0};
    Buf raw = {0};
    uint32_t acc = 0;
    int bits = 0;
    for (size_t i = 0; i < b->s.len; i++) {
        char c = b->s.ptr[i];
        int v = c >= 'A' && c <= 'Z' ? c - 'A' : c >= 'a' && c <= 'z' ? c - 'a' + 26 : c >= '0' && c <= '9' ? c - '0' + 52 : c == '+' ? 62 : c == '/' ? 63 : -1;
        if (v < 0) continue;
        acc = acc << 6 | (uint32_t)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            buf_addc(&raw, (char)(acc >> bits));
        }
    }
    Str s = u8_text(a, raw.ptr, raw.len, true);
    buf_free(&raw);
    return s;
}

typedef struct {
    Arena *a;
    const char *search;
    bool is_dir;
    double ctx, limit;
    VecStr out;
    double out_bytes;
    size_t out_count, match_count;
    bool limit_hit, match_limit, lines_trunc, done;
    double trailing_end;
    Str last_file;
} Grep;

static void grep_push(Grep *g, Str file, double line_no, Str text, bool is_match) {
    Arena *a = g->a;
    g->last_file = file;
    Str rel;
    if (g->is_dir) {
        const char *r = path_relative(a, g->search, path_resolve(a, g->search, file));
        rel = *r && strncmp(r, "..", 2) ? str(r) : path_basename(file);
    } else rel = path_basename(file);
    if (text.len && text.ptr[text.len - 1] == '\n') text.len--;
    if (text.len && text.ptr[text.len - 1] == '\r') text.len--;
    Str shown = text;
    if (utf16_len(text) > GREP_MAX_LINE) {
        Str h = utf16_head(text, GREP_MAX_LINE);
        shown = sfmt(a, "%.*s... [truncated]", (int)h.len, h.ptr);
        g->lines_trunc = true;
    }
    Str line = sfmt(a, "%.*s%c%s%c %.*s", (int)rel.len, rel.ptr, is_match ? ':' : '-', js_num(a, line_no), is_match ? ':' : '-', (int)shown.len, shown.ptr);
    if (g->out_bytes - 1 <= MAX_BYTES) vec_push(&g->out, line);
    g->out_bytes += (double)line.len + 1;
    g->out_count++;
}

static bool grep_line(void *ctx, Str raw) {
    Grep *g = ctx;
    if (g->done || !str_trim(raw).len) return false;
    JValue *ev;
    Arena *a = g->a;
    if (json_parse(a, raw, true, &ev) || !json_is(ev, J_OBJECT)) return false;
    Str type = json_strs(json_get(ev, "type"));
    bool is_match = str_eqz(type, "match");
    if (!is_match && !str_eqz(type, "context")) return false;
    const JValue *d = json_get(ev, "data");
    Str file = rg_text(a, json_get(d, "path"));
    double ln;
    if (!file.len || !jnumber(json_get(d, "line_number"), &ln)) return false;
    Str text = rg_text(a, json_get(d, "lines"));
    if (!g->limit_hit) {
        grep_push(g, file, ln, text, is_match);
        if (!is_match) return false;
        g->match_count++;
        if ((double)g->match_count >= g->limit) {
            g->limit_hit = true;
            g->trailing_end = ln + g->ctx;
        }
        return false;
    }
    bool in_window = str_eq(file, g->last_file) && ln <= g->trailing_end;
    if (in_window) grep_push(g, file, ln, text, false);
    if (is_match) g->match_limit = true;
    if (g->match_limit && !(in_window && ln < g->trailing_end)) {
        g->done = true;
        return true;
    }
    return false;
}

static void tool_grep(Arena *a, const CodeEnv *env, const JValue *args, ToolOut *o) {
    const char *rg = tool_path(a, env, "rg", NULL);
    if (!rg) { o->thrown = "ripgrep (rg) is not available and could not be downloaded"; return; }
    Str pattern = json_strs(json_get(args, "pattern"));
    const JValue *pv = json_get(args, "path"), *gv = json_get(args, "glob");
    Str dir = json_strs(pv);
    const char *search = resolve_to_cwd(a, env, dir.len ? dir : S("."));
    struct stat st;
    if (stat(search, &st)) { o->thrown = arena_printf(a, "Path not found: %s", search); return; }
    bool is_dir = S_ISDIR(st.st_mode);
    const JValue *ic = json_get(args, "ignoreCase"), *lit = json_get(args, "literal");
    double context = 0, limit = 100;
    arg_num(args, "context", &context);
    arg_num(args, "limit", &limit);
    double ctx = context > 0 ? fmin(floor(context), MAX_SAFE_INTEGER) : 0;
    double eff = fmax(1, floor(limit));
    const char *argv[24];
    int n = 0;
    argv[n++] = rg;
    argv[n++] = "--no-config";
    argv[n++] = "--json";
    argv[n++] = "--line-number";
    argv[n++] = "--color=never";
    argv[n++] = "--hidden";
    if (json_is(ic, J_BOOL) && ic->b) argv[n++] = "--ignore-case";
    if (json_is(lit, J_BOOL) && lit->b) argv[n++] = "--fixed-strings";
    Str glob = json_strs(gv);
    if (glob.len) {
        size_t i = 0;
        bool neg = glob.ptr[0] == '!';
        if (neg) i = 1;
        size_t j = i;
        while (j + 1 < glob.len && glob.ptr[j] == '.' && glob.ptr[j + 1] == '/') j += 2;
        argv[n++] = "--glob";
        argv[n++] = j > i ? arena_printf(a, "%s%.*s", neg ? "!" : "", (int)(glob.len - j), glob.ptr + j) : str_z(a, glob);
    }
    if (ctx > 0) {
        argv[n++] = "--context";
        argv[n++] = js_num(a, ctx);
    }
    argv[n++] = "--";
    argv[n++] = str_z(a, pattern);
    argv[n++] = is_dir ? "." : search;
    argv[n] = NULL;
    Grep g = {.a = a, .search = search, .is_dir = is_dir, .ctx = ctx, .limit = eff};
    Proc pr = {.on_line = grep_line, .ctx = &g, .abort = env->abort};
    int rc = run_proc(argv, is_dir ? search : NULL, &pr);
    if (pr.aborted) { o->thrown = "Operation aborted"; goto out; }
    if (rc) { o->thrown = arena_printf(a, "Failed to run ripgrep: spawn %s %s", rg, errno_code(rc)); goto out; }
    if (pr.wait_err) { o->thrown = arena_printf(a, "Failed to run ripgrep: %s", errno_code(pr.wait_err)); goto out; }
    if (!pr.killed && !(pr.exited_code && (pr.code == 0 || pr.code == 1))) {
        Str e = str_trim(u8_text(a, pr.err.ptr, pr.err.len, false));
        o->thrown = e.len ? str_z(a, e) : pr.exited_code ? arena_printf(a, "ripgrep exited with code %d", pr.code) : "ripgrep exited with code null";
        goto out;
    }
    if (!g.match_count) { out_text(a, o, S("No matches found"), NULL); goto out; }
    {
        Buf j = {0};
        for (size_t i = 0; i < g.out.len; i++) {
            if (i) buf_addc(&j, '\n');
            buf_addstr(&j, g.out.ptr[i]);
        }
        Trunc t = truncate_head(take(a, &j), MAX_SAFE_INTEGER, MAX_BYTES);
        t.total_lines = (double)g.out_count;
        t.total_bytes = g.out_bytes - 1;
        JValue *details = json_new(a, J_OBJECT);
        Buf notes = {0};
        if (g.match_limit) {
            buf_printf(&notes, "%s matches limit reached. Use limit=%s for more, or refine pattern", js_num(a, eff), js_num(a, eff * 2));
            jput(a, details, "matchLimitReached", jnum(a, eff));
        }
        if (t.truncated) {
            buf_printf(&notes, "%s%s limit reached", notes.len ? ". " : "", format_size(a, MAX_BYTES));
            jput(a, details, "truncation", trunc_json(a, &t));
        }
        if (g.lines_trunc) {
            buf_printf(&notes, "%sSome lines truncated to %d chars. Use read tool to see full lines", notes.len ? ". " : "", GREP_MAX_LINE);
            jput(a, details, "linesTruncated", json_bool(a, true));
        }
        Str text = t.content;
        if (notes.len) text = sfmt(a, "%.*s\n\n[%.*s]", (int)t.content.len, t.content.ptr, (int)notes.len, notes.ptr);
        buf_free(&notes);
        out_text(a, o, text, details->obj.len ? details : NULL);
    }
out:
    vec_free(&g.out);
    buf_free(&pr.err);
}

// ---------------------------------------------------------------------------------------
// find (find.ts)
// ---------------------------------------------------------------------------------------

typedef struct {
    Arena *a;
    const char *search;
    double limit;
    VecStr out;
    double out_bytes;
    size_t count;
    bool limit_reached;
} Find;

static bool find_line(void *ctx, Str raw) {
    Find *f = ctx;
    Str line = str_trim(raw);
    if (!line.len) return false;
    if ((double)f->count == f->limit) {
        f->limit_reached = true;
        return false;
    }
    bool trailing = line.ptr[line.len - 1] == '/';
    Str rel = line;
    if (line.ptr[0] == '/') rel = str(path_relative(f->a, f->search, str_z(f->a, line)));
    if (trailing && !(rel.len && rel.ptr[rel.len - 1] == '/')) rel = sfmt(f->a, "%.*s/", (int)rel.len, rel.ptr);
    f->count++;
    if (f->out_bytes - 1 <= MAX_BYTES) vec_push(&f->out, rel);
    f->out_bytes += (double)rel.len + 1;
    return false;
}

static void tool_find(Arena *a, const CodeEnv *env, const JValue *args, ToolOut *o) {
    Str pattern = json_strs(json_get(args, "pattern"));
    Str dir = json_strs(json_get(args, "path"));
    double limit = 1000;
    arg_num(args, "limit", &limit);
    const char *search = resolve_to_cwd(a, env, dir.len ? dir : S("."));
    double eff = fmax(1, floor(limit));
    const char *fd = tool_path(a, env, "fd", "fdfind");
    if (!fd) { o->thrown = "fd is not available and could not be downloaded"; return; }
    const char *argv[16];
    int n = 0;
    argv[n++] = fd;
    argv[n++] = "--color=never";
    argv[n++] = "--hidden";
    bool git = false;
    for (const char *cur = search;;) {
        if (exists(arena_printf(a, "%s/.git", strcmp(cur, "/") ? cur : ""))) { git = true; break; }
        const char *parent = path_dirname(a, cur);
        if (!strcmp(parent, cur)) break;
        cur = parent;
    }
    if (!git) argv[n++] = "--no-require-git";
    bool path_pat = memchr(pattern.ptr, '/', pattern.len) != NULL;
    Str eff_pat = pattern;
    if (path_pat && pattern.ptr[0] != '/' && !str_starts(pattern, S("**/")) && !str_eqz(pattern, "**")) eff_pat = sfmt(a, "**/%.*s", (int)pattern.len, pattern.ptr);
    argv[n++] = "--glob";
    argv[n++] = "--max-results";
    argv[n++] = js_num(a, eff + 1);
    if (path_pat) argv[n++] = "--full-path";
    argv[n++] = "--";
    argv[n++] = str_z(a, eff_pat);
    argv[n++] = search;
    argv[n] = NULL;
    Find f = {.a = a, .search = search, .limit = eff};
    Proc pr = {.on_line = find_line, .ctx = &f, .abort = env->abort};
    int rc = run_proc(argv, NULL, &pr);
    if (pr.aborted) { o->thrown = "Operation aborted"; goto out; }
    if (rc) { o->thrown = arena_printf(a, "Failed to run fd: spawn %s %s", fd, errno_code(rc)); goto out; }
    if (pr.wait_err) { o->thrown = arena_printf(a, "Failed to run fd: %s", errno_code(pr.wait_err)); goto out; }
    if (!(pr.exited_code && pr.code == 0) && f.count == 0) {
        Str e = str_trim(u8_text(a, pr.err.ptr, pr.err.len, false));
        o->thrown = e.len ? str_z(a, e) : pr.exited_code ? arena_printf(a, "fd exited with code %d", pr.code) : "fd exited with code null";
        goto out;
    }
    if (!f.count) { out_text(a, o, S("No files found matching pattern"), NULL); goto out; }
    {
        Buf j = {0};
        for (size_t i = 0; i < f.out.len; i++) {
            if (i) buf_addc(&j, '\n');
            buf_addstr(&j, f.out.ptr[i]);
        }
        Trunc t = truncate_head(take(a, &j), MAX_SAFE_INTEGER, MAX_BYTES);
        t.total_lines = (double)f.count;
        t.total_bytes = f.out_bytes - 1;
        JValue *details = json_new(a, J_OBJECT);
        Buf notes = {0};
        if (f.limit_reached) {
            buf_printf(&notes, "%s results limit reached. Use limit=%s for more, or refine pattern", js_num(a, eff), js_num(a, eff * 2));
            jput(a, details, "resultLimitReached", jnum(a, eff));
        }
        if (t.truncated) {
            buf_printf(&notes, "%s%s limit reached", notes.len ? ". " : "", format_size(a, MAX_BYTES));
            jput(a, details, "truncation", trunc_json(a, &t));
        }
        Str text = t.content;
        if (notes.len) text = sfmt(a, "%.*s\n\n[%.*s]", (int)t.content.len, t.content.ptr, (int)notes.len, notes.ptr);
        buf_free(&notes);
        out_text(a, o, text, details->obj.len ? details : NULL);
    }
out:
    vec_free(&f.out);
    buf_free(&pr.err);
}

// ---------------------------------------------------------------------------------------
// ls (ls.ts)
// ---------------------------------------------------------------------------------------

static int name_cmp(const void *x, const void *y) {
    const Str *a = x, *b = y;
    int c = memcmp(a->ptr, b->ptr, MIN(a->len, b->len));
    return c ? c : (a->len < b->len ? -1 : a->len > b->len);
}

// Stable merge sort by sx_locale_compare_lower, O(n log n) comparisons; runs already in order
// cost one comparison to merge.
static void locale_merge_sort(Str *v, Str *tmp, size_t n) {
    if (n < 2) return;
    size_t h = n / 2;
    locale_merge_sort(v, tmp, h);
    locale_merge_sort(v + h, tmp, n - h);
    if (sx_locale_compare_lower(v[h - 1].ptr, v[h - 1].len, v[h].ptr, v[h].len) <= 0) return;
    memcpy(tmp, v, h * sizeof *v);
    size_t i = 0, j = h, k = 0;
    while (i < h && j < n) v[k++] = sx_locale_compare_lower(tmp[i].ptr, tmp[i].len, v[j].ptr, v[j].len) <= 0 ? tmp[i++] : v[j++];
    while (i < h) v[k++] = tmp[i++];
}

static void tool_ls(Arena *a, const CodeEnv *env, const JValue *args, ToolOut *o) {
    Str p = json_strs(json_get(args, "path"));
    double limit = 500;
    arg_num(args, "limit", &limit);
    const char *dir = resolve_to_cwd(a, env, p.len ? p : S("."));
    double eff = fmax(1, floor(limit));
    if (!exists(dir)) { o->thrown = arena_printf(a, "Path not found: %s", dir); return; }
    struct stat st;
    if (stat(dir, &st)) { o->thrown = node_err(a, errno, "stat", dir); return; }
    if (!S_ISDIR(st.st_mode)) { o->thrown = arena_printf(a, "Not a directory: %s", dir); return; }
    DIR *d = opendir(dir);
    if (!d) { o->thrown = arena_printf(a, "Cannot read directory: %s", node_err(a, errno, "scandir", dir)); return; }
    VecStr names = {0};
    for (struct dirent *de; (de = readdir(d));) {
        if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
        vec_push(&names, str(arena_strdup(a, de->d_name)));
    }
    closedir(d);
    qsort(names.ptr, names.len, sizeof(Str), name_cmp); // libuv scandir order
    // stable by the locale order (Array.prototype.sort is stable): a merge sort
    if (names.len > 1) {
        Str *tmp = ARENA_ARRAY(a, Str, names.len);
        locale_merge_sort(names.ptr, tmp, names.len);
    }
    Buf out = {0};
    size_t count = 0;
    bool limit_reached = false;
    for (size_t i = 0; i < names.len; i++) {
        if ((double)count >= eff) { limit_reached = true; break; }
        struct stat es;
        if (stat(arena_printf(a, "%s/%s", strcmp(dir, "/") ? dir : "", names.ptr[i].ptr), &es)) continue;
        if (count) buf_addc(&out, '\n');
        buf_addstr(&out, names.ptr[i]);
        if (S_ISDIR(es.st_mode)) buf_addc(&out, '/');
        count++;
    }
    vec_free(&names);
    if (!count) { buf_free(&out); out_text(a, o, S("(empty directory)"), NULL); return; }
    Trunc t = truncate_head(take(a, &out), MAX_SAFE_INTEGER, MAX_BYTES);
    JValue *details = json_new(a, J_OBJECT);
    Buf notes = {0};
    if (limit_reached) {
        buf_printf(&notes, "%s entries limit reached. Use limit=%s for more", js_num(a, eff), js_num(a, eff * 2));
        jput(a, details, "entryLimitReached", jnum(a, eff));
    }
    if (t.truncated) {
        buf_printf(&notes, "%s%s limit reached", notes.len ? ". " : "", format_size(a, MAX_BYTES));
        jput(a, details, "truncation", trunc_json(a, &t));
    }
    Str text = t.content;
    if (notes.len) text = sfmt(a, "%.*s\n\n[%.*s]", (int)t.content.len, t.content.ptr, (int)notes.len, notes.ptr);
    buf_free(&notes);
    out_text(a, o, text, details->obj.len ? details : NULL);
}

// ---------------------------------------------------------------------------------------
// schemas, coercion and validation (pi-ai validation.ts, pi-schema coerce.ts / explain.ts)
// ---------------------------------------------------------------------------------------

typedef void (*ToolFn)(Arena *a, const CodeEnv *env, const JValue *args, ToolOut *o);

static const struct {
    const char *name, *schema;
    ToolFn run;
} TOOLS[] = {
    {"read", "{\"type\":\"function\",\"function\":{\"name\":\"read\",\"description\":\"Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.\",\"parameters\":{\"type\":\"object\",\"required\":[\"path\"],\"properties\":{\"path\":{\"type\":\"string\",\"description\":\"Path to the file to read (relative or absolute)\"},\"offset\":{\"type\":\"number\",\"description\":\"Line number to start reading from (1-indexed)\"},\"limit\":{\"type\":\"number\",\"description\":\"Maximum number of lines to read\"}}}}}", tool_read},
    {"bash", "{\"type\":\"function\",\"function\":{\"name\":\"bash\",\"description\":\"Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to last 2000 lines or 50KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.\",\"parameters\":{\"type\":\"object\",\"required\":[\"command\"],\"properties\":{\"command\":{\"type\":\"string\",\"description\":\"Shell command to execute\"},\"timeout\":{\"type\":\"number\",\"description\":\"Timeout in seconds (optional, no default timeout)\"}}}}}", tool_bash},
    {"edit", "{\"type\":\"function\",\"function\":{\"name\":\"edit\",\"description\":\"Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.\",\"parameters\":{\"type\":\"object\",\"required\":[\"path\",\"edits\"],\"properties\":{\"path\":{\"type\":\"string\",\"description\":\"Path to the file to edit (relative or absolute)\"},\"edits\":{\"type\":\"array\",\"items\":{\"type\":\"object\",\"required\":[\"oldText\",\"newText\"],\"properties\":{\"oldText\":{\"type\":\"string\",\"description\":\"Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call.\"},\"newText\":{\"type\":\"string\",\"description\":\"Replacement text for this targeted edit.\"}}},\"description\":\"One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead.\"}}}}}", tool_edit},
    {"write", "{\"type\":\"function\",\"function\":{\"name\":\"write\",\"description\":\"Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.\",\"parameters\":{\"type\":\"object\",\"required\":[\"path\",\"content\"],\"properties\":{\"path\":{\"type\":\"string\",\"description\":\"Path to the file to write (relative or absolute)\"},\"content\":{\"type\":\"string\",\"description\":\"Content to write to the file\"}}}}}", tool_write},
    {"grep", "{\"type\":\"function\",\"function\":{\"name\":\"grep\",\"description\":\"Search file contents for a pattern. Returns matching lines with file paths and line numbers. Respects .gitignore. Output is truncated to 100 matches or 50KB (whichever is hit first). Long lines are truncated to 500 chars.\",\"parameters\":{\"type\":\"object\",\"required\":[\"pattern\"],\"properties\":{\"pattern\":{\"type\":\"string\",\"description\":\"Search pattern (regex or literal string)\"},\"path\":{\"type\":\"string\",\"description\":\"Directory or file to search (default: current directory)\"},\"glob\":{\"type\":\"string\",\"description\":\"Filter files by glob pattern, relative to path, e.g. '*.ts', 'src/**/*.ts' or '**/*.spec.ts'. A pattern without '/' matches file names at any depth\"},\"ignoreCase\":{\"type\":\"boolean\",\"description\":\"Case-insensitive search (default: false)\"},\"literal\":{\"type\":\"boolean\",\"description\":\"Treat pattern as literal string instead of regex (default: false)\"},\"context\":{\"type\":\"number\",\"description\":\"Number of lines to show before and after each match (default: 0)\"},\"limit\":{\"type\":\"number\",\"description\":\"Maximum number of matches to return (default: 100)\"}}}}}", tool_grep},
    {"find", "{\"type\":\"function\",\"function\":{\"name\":\"find\",\"description\":\"Search for files by glob pattern. Returns matching file paths relative to the search directory. Respects .gitignore. Output is truncated to 1000 results or 50KB (whichever is hit first).\",\"parameters\":{\"type\":\"object\",\"required\":[\"pattern\"],\"properties\":{\"pattern\":{\"type\":\"string\",\"description\":\"Glob pattern to match files, e.g. '*.ts', '**/*.json', or 'src/**/*.spec.ts'\"},\"path\":{\"type\":\"string\",\"description\":\"Directory to search in (default: current directory)\"},\"limit\":{\"type\":\"number\",\"description\":\"Maximum number of results (default: 1000)\"}}}}}", tool_find},
    {"ls", "{\"type\":\"function\",\"function\":{\"name\":\"ls\",\"description\":\"List directory contents. Returns entries sorted alphabetically, with '/' suffix for directories. Includes dotfiles. Output is truncated to 500 entries or 50KB (whichever is hit first).\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\",\"description\":\"Directory to list (default: current directory)\"},\"limit\":{\"type\":\"number\",\"description\":\"Maximum number of entries to return (default: 500)\"}}}}}", tool_ls},
};

static JValue *tool_schema(Arena *a, size_t i) {
    JValue *v = NULL;
    json_parse(a, str(TOOLS[i].schema), true, &v);
    return v;
}

static ptrdiff_t tool_index(Str name) {
    for (size_t i = 0; i < COUNT(TOOLS); i++)
        if (str_eqz(name, TOOLS[i].name)) return (ptrdiff_t)i;
    return -1;
}

JValue *code_tool_schemas(Arena *a, const char *const *tools, size_t ntools) {
    JValue *arr = json_new(a, J_ARRAY), *mcp;
    for (size_t i = 0; i < ntools; i++) {
        ptrdiff_t k = tool_index(str(tools[i]));
        if (k >= 0) json_push(a, arr, tool_schema(a, (size_t)k));
        else if ((mcp = code_mcp_tool_schema(a, str(tools[i])))) json_push(a, arr, mcp); // an MCP tool (code_mcp.c)
    }
    return arr;
}

static bool is_num(const JValue *v) { return v && (v->kind == J_INTEGER || v->kind == J_FLOAT || v->kind == J_NUMBER_STRING); }

static JValue *num_val(Arena *a, double x) {
    if (x == trunc(x) && fabs(x) < 9e15) return json_int(a, (int64_t)x);
    JValue *v = json_new(a, J_FLOAT);
    v->f = x;
    return v;
}

static bool js_ws(uint32_t cp) { return js_trim_space(cp); }

static Str js_trim(Str s) {
    size_t b = 0, e = 0, i = 0;
    bool started = false;
    while (i < s.len) {
        size_t j = i;
        uint32_t cp = u8_next(s, &j);
        if (!js_ws(cp)) {
            if (!started) { b = i; started = true; }
            e = j;
        }
        i = j;
    }
    return started ? str_n(s.ptr + b, e - b) : (Str){"", 0};
}

// JavaScript's Number(string): NaN as a failure
static bool js_string_number(Str s, double *out) {
    s = js_trim(s);
    if (!s.len) { *out = 0; return true; }
    Arena t = {0};
    char *z = str_z(&t, s);
    bool ok = false;
    if (s.len > 2 && z[0] == '0' && strchr("xXoObB", z[1])) {
        int base = (z[1] | 0x20) == 'x' ? 16 : (z[1] | 0x20) == 'o' ? 8 : 2;
        double v = 0;
        ok = true;
        for (size_t i = 2; i < s.len; i++) {
            int d = hexval(z[i]);
            if (d < 0 || d >= base) { ok = false; break; }
            v = v * base + d;
        }
        *out = v;
    } else {
        const char *p = z;
        if (*p == '+' || *p == '-') p++;
        if (!strcmp(p, "Infinity")) { *out = z[0] == '-' ? -INFINITY : INFINITY; ok = true; }
        else {
            bool digits = false;
            const char *q = p;
            while (*q >= '0' && *q <= '9') { q++; digits = true; }
            if (*q == '.') { q++; while (*q >= '0' && *q <= '9') { q++; digits = true; } }
            if (digits && (*q == 'e' || *q == 'E')) {
                const char *r = q + 1;
                if (*r == '+' || *r == '-') r++;
                if (*r >= '0' && *r <= '9') { while (*r >= '0' && *r <= '9') r++; q = r; }
            }
            if (digits && !*q) { *out = strtod(z, NULL); ok = true; }
        }
    }
    arena_reset(&t);
    return ok;
}

static Str lower_ascii(Arena *a, Str s) {
    char *o = arena_strndup(a, s.ptr, s.len);
    for (size_t i = 0; i < s.len; i++) if (o[i] >= 'A' && o[i] <= 'Z') o[i] += 32;
    return str_n(o, s.len);
}

// pass 2 (TypeBox Convert): the converted value, or v itself
static JValue *convert(Arena *a, const JValue *sch, JValue *v) {
    Str type = json_strs(json_get(sch, "type"));
    if (str_eqz(type, "string")) { // tryString
        if (!v) return json_string(a, (Str){"", 0});
        if (v->kind == J_BOOL) return json_string(a, v->b ? S("true") : S("false"));
        if (is_num(v)) { double x; jnumber(v, &x); return isfinite(x) ? json_string(a, str(js_num(a, x))) : v; }
        if (v->kind == J_NULL) return json_string(a, S("null"));
        return v;
    }
    if (str_eqz(type, "number")) { // tryNumber
        if (!v || v->kind == J_NULL) return num_val(a, 0);
        if (v->kind == J_BOOL) return num_val(a, v->b ? 1 : 0);
        if (is_num(v)) return v;
        if (v->kind != J_STRING) return v;
        double x;
        if (js_string_number(v->s, &x) && isfinite(x)) return num_val(a, x);
        Str l = lower_ascii(a, v->s);
        if (str_eqz(l, "false")) return num_val(a, 0);
        if (str_eqz(l, "true")) return num_val(a, 1);
        // bigint-like strings: 12n, 12.5 (truncated), 12
        Str s = v->s;
        size_t i = s.len && s.ptr[0] == '-' ? 1 : 0, j = i;
        while (j < s.len && s.ptr[j] >= '0' && s.ptr[j] <= '9') j++;
        if (j > i && (s.ptr[i] != '0' || j == i + 1)) {
            Str digits = str_n(s.ptr, j);
            bool ok = j == s.len || (j + 1 == s.len && s.ptr[j] == 'n');
            if (!ok && s.ptr[j] == '.') {
                size_t k = j + 1;
                while (k < s.len && s.ptr[k] >= '0' && s.ptr[k] <= '9') k++;
                ok = k > j + 1 && k == s.len;
            }
            if (ok) {
                double y = strtod(str_z(a, digits), NULL);
                if (fabs(y) <= MAX_SAFE_INTEGER) return num_val(a, y);
            }
        }
        return v;
    }
    if (str_eqz(type, "boolean")) { // tryBoolean
        if (!v || v->kind == J_NULL) return json_bool(a, false);
        if (v->kind == J_BOOL) return v;
        if (is_num(v)) { double x; jnumber(v, &x); return x == 0 ? json_bool(a, false) : x == 1 ? json_bool(a, true) : v; }
        if (v->kind != J_STRING) return v;
        Str l = lower_ascii(a, v->s);
        if (str_eqz(l, "false") || str_eqz(v->s, "0")) return json_bool(a, false);
        if (str_eqz(l, "true") || str_eqz(v->s, "1")) return json_bool(a, true);
        return v;
    }
    if (str_eqz(type, "array")) {
        const JValue *items = json_get(sch, "items");
        if (json_is(v, J_ARRAY)) {
            if (!items) return v;
            JValue *o = json_new(a, J_ARRAY);
            for (size_t i = 0; i < v->arr.len; i++) json_push(a, o, convert(a, items, v->arr.items[i]));
            return o;
        }
        JValue *o = json_new(a, J_ARRAY);
        json_push(a, o, items ? convert(a, items, v) : v);
        return o;
    }
    if (str_eqz(type, "object")) {
        if (!json_is(v, J_OBJECT)) return v;
        const JValue *props = json_get(sch, "properties");
        JValue *o = json_new(a, J_OBJECT);
        for (size_t i = 0; i < v->obj.len; i++) {
            JValue *p = props ? json_get(props, str_z(a, v->obj.items[i].key)) : NULL;
            json_put(a, o, v->obj.items[i].key, p ? convert(a, p, v->obj.items[i].val) : v->obj.items[i].val);
        }
        return o;
    }
    return v;
}

static bool required_has(const JValue *sch, Str k) {
    const JValue *r = json_get(sch, "required");
    if (!json_is(r, J_ARRAY)) return false;
    for (size_t i = 0; i < r->arr.len; i++)
        if (json_is(r->arr.items[i], J_STRING) && str_eq(r->arr.items[i]->s, k)) return true;
    return false;
}

// pass 1: optional properties set to null are dropped
static JValue *drop_nulls(Arena *a, const JValue *sch, JValue *v) {
    if (json_is(v, J_ARRAY)) {
        const JValue *items = json_get(sch, "items");
        if (!items) return v;
        JValue *o = json_new(a, J_ARRAY);
        for (size_t i = 0; i < v->arr.len; i++) json_push(a, o, drop_nulls(a, items, v->arr.items[i]));
        return o;
    }
    const JValue *props = json_get(sch, "properties");
    if (!json_is(v, J_OBJECT) || !json_is(props, J_OBJECT)) return v;
    JValue *o = json_new(a, J_OBJECT);
    for (size_t i = 0; i < v->obj.len; i++) {
        Str k = v->obj.items[i].key;
        JValue *val = v->obj.items[i].val;
        JValue *p = json_get(props, str_z(a, k));
        if (p) {
            if (val->kind == J_NULL && !required_has(sch, k) && !str_eqz(json_strs(json_get(p, "type")), "null")) continue;
            val = drop_nulls(a, p, val);
        }
        json_put(a, o, k, val);
    }
    return o;
}

// pass 3 (Ajv coerceTypes)
static JValue *coerce_types(Arena *a, const JValue *sch, JValue *v) {
    Str type = json_strs(json_get(sch, "type"));
    if (str_eqz(type, "number")) {
        if (v->kind == J_NULL) v = num_val(a, 0);
        else if (v->kind == J_STRING && js_trim(v->s).len) {
            double x;
            if (js_string_number(v->s, &x) && isfinite(x)) v = num_val(a, x);
        } else if (v->kind == J_BOOL) v = num_val(a, v->b ? 1 : 0);
    } else if (str_eqz(type, "boolean")) {
        double x;
        if (v->kind == J_NULL) v = json_bool(a, false);
        else if (v->kind == J_STRING && str_eqz(v->s, "true")) v = json_bool(a, true);
        else if (v->kind == J_STRING && str_eqz(v->s, "false")) v = json_bool(a, false);
        else if (is_num(v) && jnumber(v, &x) && (x == 0 || x == 1)) v = json_bool(a, x == 1);
    } else if (str_eqz(type, "string")) {
        if (v->kind == J_NULL) v = json_string(a, (Str){"", 0});
        else if (is_num(v)) { double x; jnumber(v, &x); v = json_string(a, str(js_num(a, x))); }
        else if (v->kind == J_BOOL) v = json_string(a, v->b ? S("true") : S("false"));
    } else if (str_eqz(type, "object") && json_is(v, J_OBJECT)) {
        const JValue *props = json_get(sch, "properties");
        if (props) {
            JValue *o = obj_copy(a, v);
            for (size_t i = 0; i < props->obj.len; i++) {
                JValue *cur = json_get(o, str_z(a, props->obj.items[i].key));
                if (cur) json_put(a, o, props->obj.items[i].key, coerce_types(a, props->obj.items[i].val, cur));
            }
            v = o;
        }
    } else if (str_eqz(type, "array") && json_is(v, J_ARRAY)) {
        const JValue *items = json_get(sch, "items");
        if (items) {
            JValue *o = json_new(a, J_ARRAY);
            for (size_t i = 0; i < v->arr.len; i++) json_push(a, o, coerce_types(a, items, v->arr.items[i]));
            v = o;
        }
    }
    return v;
}

static const char *type_name(const JValue *v) {
    switch (v->kind) {
    case J_NULL: return "null";
    case J_BOOL: return "boolean";
    case J_STRING: return "string";
    case J_ARRAY: return "array";
    case J_OBJECT: return "object";
    default: {
        double x = 0;
        jnumber(v, &x);
        return x == trunc(x) ? "integer" : "number";
    }
    }
}

static bool type_ok(Str type, const JValue *v) {
    if (str_eqz(type, "string")) return v->kind == J_STRING;
    if (str_eqz(type, "number")) { double x; return is_num(v) && jnumber(v, &x) && isfinite(x); }
    if (str_eqz(type, "integer")) { double x; return is_num(v) && jnumber(v, &x) && x == trunc(x); }
    if (str_eqz(type, "boolean")) return v->kind == J_BOOL;
    if (str_eqz(type, "array")) return v->kind == J_ARRAY;
    if (str_eqz(type, "object")) return v->kind == J_OBJECT;
    if (str_eqz(type, "null")) return v->kind == J_NULL;
    return true;
}

static Str sub_path(Arena *a, Str path, Str k) { return path.len ? sfmt(a, "%.*s.%.*s", (int)path.len, path.ptr, (int)k.len, k.ptr) : k; }

static void explain(Arena *a, const JValue *sch, const JValue *v, Str path, Buf *errs) {
    Str type = json_strs(json_get(sch, "type"));
    if (type.len && !type_ok(type, v)) {
        buf_printf(errs, "\n  - %.*s: expected %.*s, got %s", path.len ? (int)path.len : 4, path.len ? path.ptr : "root", (int)type.len, type.ptr, type_name(v));
        return;
    }
    if (v->kind == J_OBJECT) {
        const JValue *req = json_get(sch, "required"), *props = json_get(sch, "properties");
        if (json_is(req, J_ARRAY))
            for (size_t i = 0; i < req->arr.len; i++) {
                if (!json_is(req->arr.items[i], J_STRING)) continue;
                Str k = req->arr.items[i]->s;
                if (!json_get(v, str_z(a, k))) {
                    Str p = sub_path(a, path, k);
                    buf_printf(errs, "\n  - %.*s: missing", (int)p.len, p.ptr);
                }
            }
        if (json_is(props, J_OBJECT))
            for (size_t i = 0; i < props->obj.len; i++) {
                Str k = props->obj.items[i].key;
                const JValue *val = json_get(v, str_z(a, k));
                if (val) explain(a, props->obj.items[i].val, val, sub_path(a, path, k), errs);
            }
    }
    if (v->kind == J_ARRAY) {
        const JValue *items = json_get(sch, "items");
        if (items)
            for (size_t i = 0; i < v->arr.len; i++) explain(a, items, v->arr.items[i], sub_path(a, path, sfmt(a, "%zu", i)), errs);
    }
}

// validateToolArguments: the coerced arguments, or NULL with *err set
static JValue *validate(Arena *a, const char *name, const JValue *params, JValue *args, const char **err) {
    JValue *v = coerce_types(a, params, convert(a, params, drop_nulls(a, params, args)));
    Buf errs = {0};
    explain(a, params, v, (Str){"", 0}, &errs);
    if (!errs.len) return v;
    Buf m = {0};
    buf_printf(&m, "Validation failed for tool \"%s\":", name);
    buf_addstr(&m, buf_str(&errs));
    buf_adds(&m, "\n\nReceived arguments:\n");
    json_pretty(&m, args, 0);
    buf_free(&errs);
    *err = take(a, &m).ptr;
    return NULL;
}

// ---------------------------------------------------------------------------------------
// the call (agent-loop.ts prepareToolCall, executePreparedToolCall, createToolResultMessage)
// ---------------------------------------------------------------------------------------

Err code_tool_run(Arena *a, const CodeEnv *env, const JValue *call, JValue **result, JValue **raw) {
    Str id = json_strs(json_get(call, "id")), name = json_strs(json_get(call, "name"));
    JValue *args = json_get(call, "arguments");
    if (!args) args = json_new(a, J_OBJECT);
    ToolOut o = {0};
    bool executed = false;
    double duration = 0;
    ptrdiff_t k = tool_index(name);
    JValue *mcp_params = k < 0 ? code_mcp_tool_parameters(a, name) : NULL; // an MCP tool (code_mcp.c)
    if (k < 0 && !mcp_params) o.thrown = arena_printf(a, "Tool %.*s not found", (int)name.len, name.ptr);
    else {
        const char *tname = k >= 0 ? TOOLS[k].name : str_z(a, name);
        JValue *prepared = !strcmp(tname, "edit") ? prepare_edit(a, args) : args;
        const JValue *ae = json_get(call, "argumentsError");
        JValue *valid = NULL;
        if (json_is(ae, J_STRING)) o.thrown = arena_printf(a, "Invalid arguments for tool \"%s\": %.*s", tname, (int)ae->s.len, ae->s.ptr);
        else valid = validate(a, tname, k < 0 ? mcp_params : json_get(json_get(tool_schema(a, (size_t)k), "function"), "parameters"), prepared, &o.thrown);
        if (valid && env->abort && *env->abort) o.thrown = "Operation aborted"; // prepareToolCall's signal check
        else if (valid) {
            executed = true;
            uint64_t t0 = now_ns();
            if (k >= 0) TOOLS[k].run(a, env, valid, &o);
            else code_mcp_tool_call(a, env, name, valid, &o.content, &o.details, &o.structured, &o.is_error, &o.thrown);
            duration = round(ms_since(t0));
        }
    }
    if (o.thrown) {
        o.content = text_content(a, str(o.thrown));
        o.details = json_new(a, J_OBJECT);
        o.structured = NULL;
        o.is_error = true;
    }
    JValue *m = json_new(a, J_OBJECT);
    jput(a, m, "role", jstr(a, S("toolResult")));
    jput(a, m, "toolCallId", jstr(a, id));
    jput(a, m, "toolName", jstr(a, name));
    jput(a, m, "content", o.content);
    if (o.details) jput(a, m, "details", o.details);
    jput(a, m, "isError", json_bool(a, o.is_error));
    if (executed) jput(a, m, "durationMs", jnum(a, duration));
    jput(a, m, "timestamp", jnum(a, env->now_ms));
    *result = m;
    if (raw) {
        JValue *r = json_new(a, J_OBJECT);
        jput(a, r, "content", o.content);
        if (o.details) jput(a, r, "details", o.details);
        if (o.structured) jput(a, r, "structuredContent", o.structured);
        if (!o.thrown && o.is_error) jput(a, r, "isError", json_bool(a, true));
        *raw = r;
    }
    return OK;
}
