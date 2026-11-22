// Saetta Code TUI: lines, text width, wrapping and truncation with ANSI styles, the theme, the
// basic components (Text, Spacer, Container) and the select list (see code_tui.h).
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at v1.5.2,
// b4afddd): packages/tui/src/utils.ts (visibleWidth, wrapTextWithAnsi, truncateToWidth,
// sliceWithWidth, AnsiCodeTracker), components/text.ts, spacer.ts, select-list.ts, and
// packages/coding-agent/src/modes/interactive/theme/ (theme.ts, system-theme.ts indexedColors).
#include "code_tui.h"
#include "code_tui_theme.h"
#include "unicode.h"

#include <stdlib.h>

// ---- lines -----------------------------------------------------------------------------------

void lines_push(Lines *l, char *owned) {
    if (l->n == l->cap) {
        l->cap = l->cap ? l->cap * 2 : 16;
        l->v = xrealloc(l->v, l->cap * sizeof *l->v);
    }
    l->v[l->n++] = owned;
}

void lines_add(Lines *l, Str s) {
    char *p = xmalloc(s.len + 1);
    memcpy(p, s.ptr, s.len);
    p[s.len] = 0;
    lines_push(l, p);
}

void lines_addz(Lines *l, const char *s) { lines_add(l, str(s)); }

void lines_addb(Lines *l, Buf *b) {
    if (!b->ptr) {
        lines_addz(l, "");
        return;
    }
    lines_push(l, b->ptr);
    b->ptr = NULL;
    b->len = b->cap = 0;
}

void lines_append(Lines *dst, Lines *src) {
    for (size_t i = 0; i < src->n; i++) lines_push(dst, src->v[i]);
    free(src->v);
    src->v = NULL;
    src->n = src->cap = 0;
}

void lines_free(Lines *l) {
    for (size_t i = 0; i < l->n; i++) free(l->v[i]);
    free(l->v);
    l->v = NULL;
    l->n = l->cap = 0;
}

// ---- measuring ----------------------------------------------------------------------------------

size_t tui_ansi_len(Str s, size_t i) {
    if (i + 1 >= s.len || s.ptr[i] != 0x1b) return 0;
    char next = s.ptr[i + 1];
    if (next == '[') {
        for (size_t j = i + 2; j < s.len; j++) {
            char c = s.ptr[j];
            if (c == 'm' || c == 'G' || c == 'K' || c == 'H' || c == 'J') return j + 1 - i;
        }
        return 0;
    }
    if (next == ']' || next == '_') {
        for (size_t j = i + 2; j < s.len; j++) {
            if (s.ptr[j] == 0x07) return j + 1 - i;
            if (s.ptr[j] == 0x1b && j + 1 < s.len && s.ptr[j + 1] == '\\') return j + 2 - i;
        }
        return 0;
    }
    return 0;
}

uint32_t tui_cp(Str s, size_t *i) {
    const unsigned char *p = (const unsigned char *)s.ptr;
    size_t k = *i;
    unsigned c = p[k];
    if (c < 0x80) {
        *i = k + 1;
        return c;
    }
    int n = c >= 0xf8 ? -1 : c >= 0xf0 ? 3 : c >= 0xe0 ? 2 : c >= 0xc0 ? 1 : -1;
    if (n < 0) {
        *i = k + 1;
        return 0xfffd;
    }
    uint32_t cp = c & (0x3fu >> n);
    for (int j = 1; j <= n; j++) {
        if (k + (size_t)j >= s.len || (p[k + (size_t)j] & 0xc0) != 0x80) {
            *i = k + (size_t)j;
            return 0xfffd;
        }
        cp = cp << 6 | (p[k + (size_t)j] & 0x3f);
    }
    *i = k + (size_t)n + 1;
    return cp;
}

static bool in(uint32_t cp, uint32_t lo, uint32_t hi) { return cp >= lo && cp <= hi; }

// East Asian Wide and Fullwidth (and the emoji ranges terminals draw in two cells), as
// get-east-asian-width and pi's RGI emoji check give them for the common blocks.
static bool wide(uint32_t cp) {
    return in(cp, 0x1100, 0x115f) || in(cp, 0x231a, 0x231b) || in(cp, 0x2329, 0x232a) || in(cp, 0x23e9, 0x23ec) ||
           cp == 0x23f0 || cp == 0x23f3 || in(cp, 0x25fd, 0x25fe) || in(cp, 0x2614, 0x2615) || in(cp, 0x2648, 0x2653) ||
           cp == 0x267f || cp == 0x2693 || cp == 0x26a1 || in(cp, 0x26aa, 0x26ab) || in(cp, 0x26bd, 0x26be) ||
           in(cp, 0x26c4, 0x26c5) || cp == 0x26ce || cp == 0x26d4 || cp == 0x26ea || in(cp, 0x26f2, 0x26f3) || cp == 0x26f5 ||
           cp == 0x26fa || cp == 0x26fd || cp == 0x2705 || in(cp, 0x270a, 0x270b) || cp == 0x2728 || cp == 0x274c ||
           cp == 0x274e || in(cp, 0x2753, 0x2755) || cp == 0x2757 || in(cp, 0x2795, 0x2797) || cp == 0x27b0 || cp == 0x27bf ||
           in(cp, 0x2b1b, 0x2b1c) || cp == 0x2b50 || cp == 0x2b55 || in(cp, 0x2e80, 0x303e) || in(cp, 0x3041, 0x33ff) ||
           in(cp, 0x3400, 0x4dbf) || in(cp, 0x4e00, 0x9fff) || in(cp, 0xa000, 0xa4cf) || in(cp, 0xa960, 0xa97f) ||
           in(cp, 0xac00, 0xd7a3) || in(cp, 0xf900, 0xfaff) || in(cp, 0xfe10, 0xfe19) || in(cp, 0xfe30, 0xfe6f) ||
           in(cp, 0xff00, 0xff60) || in(cp, 0xffe0, 0xffe6) || in(cp, 0x16fe0, 0x18cff) || in(cp, 0x1b000, 0x1b2ff) ||
           in(cp, 0x1f004, 0x1f004) || cp == 0x1f0cf || cp == 0x1f18e || in(cp, 0x1f191, 0x1f19a) || in(cp, 0x1f1e6, 0x1f1ff) ||
           in(cp, 0x1f200, 0x1f251) || in(cp, 0x1f300, 0x1f64f) || in(cp, 0x1f680, 0x1f6ff) || in(cp, 0x1f7e0, 0x1f7eb) ||
           in(cp, 0x1f90c, 0x1f9ff) || in(cp, 0x1fa70, 0x1faff) || in(cp, 0x20000, 0x3fffd);
}

int tui_cp_width(uint32_t cp) {
    if (cp >= 0x20 && cp < 0x7f) return 1;
    if (cp == '\t') return 3;
    if (cp < 0x20 || in(cp, 0x7f, 0x9f)) return 0;
    if (in(cp, 0x200b, 0x200f) || in(cp, 0x2028, 0x202e) || in(cp, 0x2060, 0x206f) || in(cp, 0xfe00, 0xfe0f) || cp == 0xfeff ||
        cp == 0xad || in(cp, 0xe0000, 0xe0fff))
        return 0;
    if (unicode_class(cp) == UNICODE_MARK) return 0;
    return wide(cp) ? 2 : 1;
}

size_t tui_grapheme(Str s, size_t i) {
    size_t k = i;
    uint32_t cp = tui_cp(s, &k);
    if (cp == '\r' && k < s.len && s.ptr[k] == '\n') return k + 1 - i;
    for (;;) {
        if (k >= s.len) break;
        size_t j = k;
        uint32_t next = tui_cp(s, &j);
        bool join = (next >= 0x80 && unicode_class(next) == UNICODE_MARK) || in(next, 0xfe00, 0xfe0f) || next == 0x200d ||
                    in(next, 0x1f3fb, 0x1f3ff) || in(next, 0xe0020, 0xe007f);
        if (!join) break;
        k = j;
        if (next == 0x200d && k < s.len) tui_cp(s, &k); // the joined character
    }
    return k - i;
}

// Width of one grapheme cluster (graphemeWidth): its first visible code point, an emoji
// presentation selector making it wide.
static int grapheme_width(Str g) {
    size_t i = 0;
    int w = 0;
    bool first = true;
    while (i < g.len) {
        uint32_t cp = tui_cp(g, &i);
        if (first) {
            int cw = tui_cp_width(cp);
            if (cw == 0) continue; // leading non-printing characters
            w = cw;
            first = false;
        } else if (cp == 0xfe0f && w == 1) w = 2;
    }
    return w;
}

int tui_width(Str s) {
    int w = 0;
    size_t i = 0;
    while (i < s.len) {
        unsigned char c = (unsigned char)s.ptr[i];
        if (c >= 0x20 && c < 0x7f) {
            w++;
            i++;
            continue;
        }
        if (c == '\t') {
            w += 3;
            i++;
            continue;
        }
        if (c == 0x1b) {
            size_t n = tui_ansi_len(s, i);
            if (n) {
                i += n;
                continue;
            }
            i++;
            continue;
        }
        size_t n = tui_grapheme(s, i);
        w += grapheme_width(str_n(s.ptr + i, n));
        i += n;
    }
    return w;
}

size_t tui_js_len(Str s) {
    size_t n = 0, i = 0;
    while (i < s.len) {
        uint32_t cp = tui_cp(s, &i);
        n += cp >= 0x10000 ? 2 : 1;
    }
    return n;
}

int tui_utf8(char *out, uint32_t cp) {
    if (cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) cp = 0xfffd;
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char)(0xc0 | cp >> 6);
        out[1] = (char)(0x80 | (cp & 0x3f));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xe0 | cp >> 12);
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3f));
        out[2] = (char)(0x80 | (cp & 0x3f));
        return 3;
    }
    out[0] = (char)(0xf0 | cp >> 18);
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3f));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3f));
    out[3] = (char)(0x80 | (cp & 0x3f));
    return 4;
}

bool tui_is_space(uint32_t cp) { return unicode_space(cp) || cp == 0xfeff; }

void tui_spaces(Buf *b, int n) {
    if (n <= 0) return;
    buf_reserve(b, (size_t)n);
    memset(b->ptr + b->len, ' ', (size_t)n);
    b->len += (size_t)n;
    b->ptr[b->len] = 0;
}

void tui_repeat(Buf *b, const char *s, int n) {
    for (int i = 0; i < n; i++) buf_adds(b, s);
}

void tui_strip(Buf *out, Str s) {
    size_t i = 0;
    while (i < s.len) {
        size_t n = tui_ansi_len(s, i);
        if (n) {
            i += n;
            continue;
        }
        buf_addc(out, s.ptr[i++]);
    }
}

// ---- the SGR tracker (AnsiCodeTracker) -----------------------------------------------------------

typedef struct {
    bool bold, dim, italic, underline, blink, inverse, hidden, strike;
    char fg[32], bg[32];
    char link[512]; // the open OSC 8 sequence, or empty
} Tracker;

static void tr_reset(Tracker *t) {
    char link[512];
    memcpy(link, t->link, sizeof link);
    memset(t, 0, sizeof *t);
    memcpy(t->link, link, sizeof link);
}

static void tr_process(Tracker *t, Str code) {
    if (code.len >= 4 && !memcmp(code.ptr, "\x1b]8;", 4)) {
        // \x1b]8;params;url<term>: open when url is not empty
        size_t semi = 4;
        while (semi < code.len && code.ptr[semi] != ';') semi++;
        size_t term = code.ptr[code.len - 1] == 0x07 ? code.len - 1 : code.len - 2;
        if (semi + 1 >= term) t->link[0] = 0;
        else if (code.len < sizeof t->link) {
            memcpy(t->link, code.ptr, code.len);
            t->link[code.len] = 0;
        }
        return;
    }
    if (code.len < 3 || code.ptr[1] != '[' || code.ptr[code.len - 1] != 'm') return;
    Str params = str_n(code.ptr + 2, code.len - 3);
    for (size_t k = 0; k < params.len; k++)
        if (!(params.ptr[k] == ';' || (params.ptr[k] >= '0' && params.ptr[k] <= '9'))) return;
    if (params.len == 0 || (params.len == 1 && params.ptr[0] == '0')) {
        tr_reset(t);
        return;
    }
    // split on ';'
    long parts[32];
    size_t np = 0;
    Str rest = params;
    for (;;) {
        ptrdiff_t c = str_find_char(rest, ';', 0);
        Str p = c >= 0 ? str_sub(rest, 0, (size_t)c) : rest;
        long v = 0;
        bool nan = p.len == 0;
        for (size_t k = 0; k < p.len; k++) v = v * 10 + (p.ptr[k] - '0');
        if (np < 32) parts[np++] = nan ? -1 : v;
        if (c < 0) break;
        rest = str_sub(rest, (size_t)c + 1, rest.len);
    }
    for (size_t i = 0; i < np;) {
        long code_ = parts[i];
        if ((code_ == 38 || code_ == 48) && i + 1 < np) {
            char *dst = code_ == 38 ? t->fg : t->bg;
            if (parts[i + 1] == 5 && i + 2 < np) {
                snprintf(dst, 32, "%ld;5;%ld", code_, parts[i + 2]);
                i += 3;
                continue;
            }
            if (parts[i + 1] == 2 && i + 4 < np) {
                snprintf(dst, 32, "%ld;2;%ld;%ld;%ld", code_, parts[i + 2], parts[i + 3], parts[i + 4]);
                i += 5;
                continue;
            }
        }
        switch (code_) {
        case 0: tr_reset(t); break;
        case 1: t->bold = true; break;
        case 2: t->dim = true; break;
        case 3: t->italic = true; break;
        case 4: t->underline = true; break;
        case 5: t->blink = true; break;
        case 7: t->inverse = true; break;
        case 8: t->hidden = true; break;
        case 9: t->strike = true; break;
        case 21: t->bold = false; break;
        case 22: t->bold = t->dim = false; break;
        case 23: t->italic = false; break;
        case 24: t->underline = false; break;
        case 25: t->blink = false; break;
        case 27: t->inverse = false; break;
        case 28: t->hidden = false; break;
        case 29: t->strike = false; break;
        case 39: t->fg[0] = 0; break;
        case 49: t->bg[0] = 0; break;
        default:
            if ((code_ >= 30 && code_ <= 37) || (code_ >= 90 && code_ <= 97)) snprintf(t->fg, 32, "%ld", code_);
            else if ((code_ >= 40 && code_ <= 47) || (code_ >= 100 && code_ <= 107)) snprintf(t->bg, 32, "%ld", code_);
        }
        i++;
    }
}

static void tr_active(const Tracker *t, Buf *out) {
    char codes[128];
    size_t n = 0;
    codes[0] = 0;
#define ADD(s)                                                                    \
    do {                                                                          \
        n += (size_t)snprintf(codes + n, sizeof codes - n, "%s%s", n ? ";" : "", s); \
    } while (0)
    if (t->bold) ADD("1");
    if (t->dim) ADD("2");
    if (t->italic) ADD("3");
    if (t->underline) ADD("4");
    if (t->blink) ADD("5");
    if (t->inverse) ADD("7");
    if (t->hidden) ADD("8");
    if (t->strike) ADD("9");
    if (t->fg[0]) ADD(t->fg);
    if (t->bg[0]) ADD(t->bg);
#undef ADD
    if (n) buf_printf(out, "\x1b[%sm", codes);
    if (t->link[0]) buf_adds(out, t->link);
}

static void tr_line_end(const Tracker *t, Buf *out) {
    if (t->underline) buf_adds(out, "\x1b[24m");
    if (t->strike) buf_adds(out, "\x1b[29m");
    if (t->link[0]) buf_adds(out, t->link[strlen(t->link) - 1] == 0x07 ? "\x1b]8;;\x07" : "\x1b]8;;\x1b\\");
}

static void tr_feed(Tracker *t, Str s) {
    for (size_t i = 0; i < s.len;) {
        size_t n = s.ptr[i] == 0x1b ? tui_ansi_len(s, i) : 0;
        if (n) {
            tr_process(t, str_n(s.ptr + i, n));
            i += n;
        } else i++;
    }
}

// ---- wrapping (wrapTextWithAnsi) ---------------------------------------------------------------

static bool cjk(uint32_t cp) {
    return in(cp, 0x1100, 0x11ff) || in(cp, 0x2e80, 0x2fdf) || in(cp, 0x3000, 0x303f) || in(cp, 0x3040, 0x30ff) ||
           in(cp, 0x3100, 0x312f) || in(cp, 0x3130, 0x318f) || in(cp, 0x31a0, 0x31ff) || in(cp, 0x3200, 0x32ff) ||
           in(cp, 0x3300, 0x33ff) || in(cp, 0x3400, 0x4dbf) || in(cp, 0x4e00, 0x9fff) || in(cp, 0xa960, 0xa97f) ||
           in(cp, 0xac00, 0xd7ff) || in(cp, 0xf900, 0xfaff) || in(cp, 0xfe30, 0xfe4f) || in(cp, 0xff00, 0xffef) ||
           in(cp, 0x20000, 0x3ffff);
}

typedef VEC(Str) Tokens;

// splitIntoTokensWithAnsi: runs of spaces and runs of other text, escapes attached to the next
// visible character; CJK characters stand alone.
static void tokens_split(Arena *a, Str text, Tokens *out) {
    Buf cur = {0}, pending = {0};
    int kind = 0; // 0 none, 1 space, 2 word
#define FLUSH()                                                                                 \
    do {                                                                                        \
        if (cur.len) {                                                                          \
            vec_push(out, str_n(arena_strndup(a, cur.ptr, cur.len), cur.len));                  \
            cur.len = 0;                                                                        \
            kind = 0;                                                                           \
        }                                                                                       \
    } while (0)
    size_t i = 0;
    while (i < text.len) {
        size_t n = text.ptr[i] == 0x1b ? tui_ansi_len(text, i) : 0;
        if (n) {
            buf_add(&pending, text.ptr + i, n);
            i += n;
            continue;
        }
        size_t g = text.ptr[i] == 0x1b ? 1 : tui_grapheme(text, i);
        Str seg = str_n(text.ptr + i, g);
        bool space = g == 1 && seg.ptr[0] == ' ';
        size_t k = 0;
        uint32_t cp = tui_cp(seg, &k);
        if (!space && cp >= 0x80 && cjk(cp)) {
            FLUSH();
            buf_addstr(&pending, seg);
            vec_push(out, str_n(arena_strndup(a, pending.ptr, pending.len), pending.len));
            pending.len = 0;
            i += g;
            continue;
        }
        int sk = space ? 1 : 2;
        if (cur.len && kind != sk) FLUSH();
        if (pending.len) {
            buf_addstr(&cur, buf_str(&pending));
            pending.len = 0;
        }
        kind = sk;
        buf_addstr(&cur, seg);
        i += g;
    }
    if (pending.len) {
        if (cur.len) buf_addstr(&cur, buf_str(&pending));
        else if (out->len) {
            Str last = out->ptr[out->len - 1];
            char *p = arena_printf(a, "%.*s%.*s", (int)last.len, last.ptr, (int)pending.len, pending.ptr);
            out->ptr[out->len - 1] = str_n(p, last.len + pending.len);
        } else buf_addstr(&cur, buf_str(&pending));
    }
    if (cur.len) vec_push(out, str_n(arena_strndup(a, cur.ptr, cur.len), cur.len));
#undef FLUSH
    buf_free(&cur);
    buf_free(&pending);
}

static bool whitespace_token(Str t) {
    size_t i = 0;
    while (i < t.len) {
        unsigned char c = (unsigned char)t.ptr[i];
        if (c == 0x1b) {
            size_t n = tui_ansi_len(t, i);
            if (!n) return false;
            i += n;
        } else if (c == ' ' || (c >= 9 && c <= 13)) i++;
        else if (c < 0x80) return false;
        else {
            uint32_t cp = tui_cp(t, &i);
            if (!tui_is_space(cp)) return false;
        }
    }
    return true;
}

static void extract_codes(Buf *out, Str s) {
    for (size_t i = 0; i < s.len;) {
        size_t n = s.ptr[i] == 0x1b ? tui_ansi_len(s, i) : 0;
        if (n) {
            buf_add(out, s.ptr + i, n);
            i += n;
        } else i++;
    }
}

// String.prototype.trimEnd on a buffer.
static void trim_end(Buf *b) {
    while (b->len) {
        unsigned char c = (unsigned char)b->ptr[b->len - 1];
        if (c == ' ' || (c >= 9 && c <= 13)) {
            b->len--;
            continue;
        }
        if (c >= 0x80) {
            size_t k = b->len - 1;
            while (k > 0 && ((unsigned char)b->ptr[k] & 0xc0) == 0x80) k--;
            size_t j = k;
            uint32_t cp = tui_cp(buf_str(b), &j);
            if (tui_is_space(cp)) {
                b->len = k;
                continue;
            }
        }
        break;
    }
    if (b->ptr) b->ptr[b->len] = 0;
}

static void push_trimmed(Lines *out, Buf *b) {
    trim_end(b);
    lines_add(out, buf_str(b));
}

static void break_long_word(Str word, int width, Tracker *t, Lines *lines) {
    Buf cur = {0};
    tr_active(t, &cur);
    int w = 0;
    size_t i = 0;
    while (i < word.len) {
        size_t n = word.ptr[i] == 0x1b ? tui_ansi_len(word, i) : 0;
        if (n) {
            buf_add(&cur, word.ptr + i, n);
            tr_process(t, str_n(word.ptr + i, n));
            i += n;
            continue;
        }
        size_t g = tui_grapheme(word, i);
        int gw = tui_width(str_n(word.ptr + i, g));
        if (w + gw > width) {
            tr_line_end(t, &cur);
            lines_add(lines, buf_str(&cur));
            cur.len = 0;
            if (cur.ptr) cur.ptr[0] = 0;
            tr_active(t, &cur);
            w = 0;
        }
        buf_add(&cur, word.ptr + i, g);
        w += gw;
        i += g;
    }
    if (cur.len) lines_add(lines, buf_str(&cur));
    if (lines->n == 0) lines_addz(lines, "");
    buf_free(&cur);
}

static void wrap_single(Str line, int width, Lines *out) {
    if (!line.len) {
        lines_addz(out, "");
        return;
    }
    if (tui_width(line) <= width) {
        lines_add(out, line);
        return;
    }
    Arena a = {0};
    Tokens toks = {0};
    tokens_split(&a, line, &toks);
    Tracker t = {0};
    Buf cur = {0};
    int cw = 0;
    Lines wrapped = {0};
    for (size_t k = 0; k < toks.len; k++) {
        Str tok = toks.ptr[k];
        int tw = tui_width(tok);
        bool ws = whitespace_token(tok);
        if (tw > width && !ws) {
            if (cur.len) {
                trim_end(&cur);
                tr_line_end(&t, &cur);
                lines_add(&wrapped, buf_str(&cur));
                cur.len = 0;
                cw = 0;
            }
            Lines broken = {0};
            break_long_word(tok, width, &t, &broken);
            for (size_t j = 0; j + 1 < broken.n; j++) lines_addz(&wrapped, broken.v[j]);
            cur.len = 0;
            buf_adds(&cur, broken.v[broken.n - 1]);
            cw = tui_width(buf_str(&cur));
            lines_free(&broken);
            continue;
        }
        int need = cw + tw;
        if (need > width && cw > 0) {
            Buf lw = {0};
            buf_addstr(&lw, buf_str(&cur));
            trim_end(&lw);
            if (ws) {
                extract_codes(&lw, tok);
                tr_feed(&t, tok);
            }
            tr_line_end(&t, &lw);
            lines_add(&wrapped, buf_str(&lw));
            buf_free(&lw);
            cur.len = 0;
            if (cur.ptr) cur.ptr[0] = 0;
            tr_active(&t, &cur);
            if (ws) {
                cw = 0;
                continue;
            }
            buf_addstr(&cur, tok);
            cw = tw;
        } else if (ws && need > width && memchr(tok.ptr, 0x1b, tok.len)) {
            extract_codes(&cur, tok);
        } else {
            buf_addstr(&cur, tok);
            cw += tw;
        }
        tr_feed(&t, tok);
    }
    if (cur.len) lines_add(&wrapped, buf_str(&cur));
    if (wrapped.n == 0) lines_addz(out, "");
    for (size_t j = 0; j < wrapped.n; j++) {
        Buf b = {0};
        buf_adds(&b, wrapped.v[j]);
        push_trimmed(out, &b);
        buf_free(&b);
    }
    lines_free(&wrapped);
    buf_free(&cur);
    vec_free(&toks);
    arena_reset(&a);
}

void tui_wrap(Str text, int width, Lines *out) {
    if (!text.len) {
        lines_addz(out, "");
        return;
    }
    Tracker t = {0};
    size_t before = out->n;
    size_t i = 0;
    for (;;) {
        size_t j = i;
        while (j < text.len && text.ptr[j] != '\n' && text.ptr[j] != '\r') j++;
        Str line = str_n(text.ptr + i, j - i);
        Buf pre = {0};
        if (out->n > before) tr_active(&t, &pre);
        buf_addstr(&pre, line);
        wrap_single(buf_str(&pre), width, out);
        buf_free(&pre);
        tr_feed(&t, line);
        if (j >= text.len) break;
        if (text.ptr[j] == '\r' && j + 1 < text.len && text.ptr[j + 1] == '\n') j++;
        i = j + 1;
    }
    if (out->n == before) lines_addz(out, "");
}

// ---- truncation (truncateToWidth) -----------------------------------------------------------------

static void finalize_trunc(Buf *out, Str prefix, int prefix_w, Str ellipsis, int ell_w, int maxw, bool pad) {
    buf_addstr(out, prefix);
    // an open hyperlink in the kept part is closed
    Tracker t = {0};
    tr_feed(&t, prefix);
    if (t.link[0]) buf_adds(out, t.link[strlen(t.link) - 1] == 0x07 ? "\x1b]8;;\x07" : "\x1b]8;;\x1b\\");
    buf_adds(out, "\x1b[0m");
    if (ellipsis.len) {
        buf_addstr(out, ellipsis);
        buf_adds(out, "\x1b[0m");
    }
    if (pad) tui_spaces(out, maxw - (prefix_w + ell_w));
}

void tui_truncate(Buf *out, Str text, int maxw, const char *ellipsis_z, bool pad) {
    if (maxw <= 0) return;
    Str ellipsis = str(ellipsis_z ? ellipsis_z : "...");
    if (!text.len) {
        if (pad) tui_spaces(out, maxw);
        return;
    }
    int ell_w = tui_width(ellipsis);
    int text_w = tui_width(text);
    if (ell_w >= maxw) {
        if (text_w <= maxw) {
            buf_addstr(out, text);
            if (pad) tui_spaces(out, maxw - text_w);
            return;
        }
        Buf clipped = {0};
        tui_slice(&clipped, ellipsis, 0, maxw, true);
        int cw = tui_width(buf_str(&clipped));
        if (cw == 0) {
            if (pad) tui_spaces(out, maxw);
        } else finalize_trunc(out, S(""), 0, buf_str(&clipped), cw, maxw, pad);
        buf_free(&clipped);
        return;
    }
    if (text_w <= maxw) {
        buf_addstr(out, text);
        if (pad) tui_spaces(out, maxw - text_w);
        return;
    }
    int target = maxw - ell_w;
    Buf res = {0}, pending = {0};
    int kept = 0;
    size_t i = 0;
    while (i < text.len) {
        size_t n = text.ptr[i] == 0x1b ? tui_ansi_len(text, i) : 0;
        if (n) {
            buf_add(&pending, text.ptr + i, n);
            i += n;
            continue;
        }
        size_t g = text.ptr[i] == '\t' ? 1 : tui_grapheme(text, i);
        int gw = text.ptr[i] == '\t' ? 3 : tui_width(str_n(text.ptr + i, g));
        if (kept + gw > target) break;
        if (pending.len) {
            buf_addstr(&res, buf_str(&pending));
            pending.len = 0;
        }
        buf_add(&res, text.ptr + i, g);
        kept += gw;
        i += g;
    }
    finalize_trunc(out, buf_str(&res), kept, ellipsis, ell_w, maxw, pad);
    buf_free(&res);
    buf_free(&pending);
}

char *tui_truncatez(Str text, int width, const char *ellipsis, bool pad) {
    Buf b = {0};
    tui_truncate(&b, text, width, ellipsis, pad);
    if (!b.ptr) return calloc(1, 1);
    return b.ptr;
}

void tui_slice(Buf *out, Str line, int start, int len, bool strict) {
    if (len <= 0) return;
    int end = start + len, col = 0;
    Buf pending = {0};
    size_t i = 0;
    while (i < line.len) {
        size_t n = line.ptr[i] == 0x1b ? tui_ansi_len(line, i) : 0;
        if (n) {
            if (col >= start && col < end) {
                buf_addstr(out, buf_str(&pending));
                pending.len = 0;
                buf_add(out, line.ptr + i, n);
            } else if (col < start && line.ptr[i + 1] != '_') buf_add(&pending, line.ptr + i, n);
            i += n;
            continue;
        }
        size_t g = tui_grapheme(line, i);
        int w = grapheme_width(str_n(line.ptr + i, g));
        if (line.ptr[i] == '\t') w = 3;
        bool inr = col >= start && col < end;
        bool fits = !strict || col + w <= end;
        if (inr && fits) {
            buf_addstr(out, buf_str(&pending));
            pending.len = 0;
            buf_add(out, line.ptr + i, g);
        }
        col += w;
        i += g;
        if (col >= end) break;
    }
    buf_free(&pending);
}

// ---- theme ----------------------------------------------------------------------------------------

// pi's system theme (code_tui_theme.c): generated from the colors the terminal reports, the
// indexed palette tier when it reports none; grayscale while the reply is pending.
static const ThemeFg theme_map[T_COUNT] = {
    [T_ACCENT] = TH_FG_ACCENT, [T_BORDER] = TH_FG_BORDER, [T_BORDER_ACCENT] = TH_FG_BORDER_ACCENT,
    [T_BORDER_MUTED] = TH_FG_BORDER_MUTED, [T_SUCCESS] = TH_FG_SUCCESS, [T_ERROR] = TH_FG_ERROR, [T_WARNING] = TH_FG_WARNING,
    [T_MUTED] = TH_FG_MUTED, [T_DIM] = TH_FG_DIM, [T_TEXT] = TH_FG_TEXT, [T_THINKING_TEXT] = TH_FG_THINKING_TEXT,
    [T_USER_TEXT] = TH_FG_USER_MESSAGE_TEXT, [T_TOOL_TITLE] = TH_FG_TOOL_TITLE, [T_TOOL_OUTPUT] = TH_FG_TOOL_OUTPUT,
    [T_MD_HEADING] = TH_FG_MD_HEADING, [T_MD_LINK] = TH_FG_MD_LINK, [T_MD_LINK_URL] = TH_FG_MD_LINK_URL,
    [T_MD_CODE] = TH_FG_MD_CODE, [T_MD_CODE_BLOCK] = TH_FG_MD_CODE_BLOCK, [T_MD_CODE_BLOCK_BORDER] = TH_FG_MD_CODE_BLOCK_BORDER,
    [T_MD_QUOTE] = TH_FG_MD_QUOTE, [T_MD_QUOTE_BORDER] = TH_FG_MD_QUOTE_BORDER, [T_MD_HR] = TH_FG_MD_HR,
    [T_MD_LIST_BULLET] = TH_FG_MD_LIST_BULLET, [T_DIFF_ADDED] = TH_FG_TOOL_DIFF_ADDED, [T_DIFF_REMOVED] = TH_FG_TOOL_DIFF_REMOVED,
    [T_DIFF_CONTEXT] = TH_FG_TOOL_DIFF_CONTEXT, [T_SYN_COMMENT] = TH_FG_SYNTAX_COMMENT, [T_SYN_KEYWORD] = TH_FG_SYNTAX_KEYWORD,
    [T_SYN_FUNCTION] = TH_FG_SYNTAX_FUNCTION, [T_SYN_VARIABLE] = TH_FG_SYNTAX_VARIABLE, [T_SYN_STRING] = TH_FG_SYNTAX_STRING,
    [T_SYN_NUMBER] = TH_FG_SYNTAX_NUMBER, [T_SYN_TYPE] = TH_FG_SYNTAX_TYPE, [T_SYN_OPERATOR] = TH_FG_SYNTAX_OPERATOR,
    [T_SYN_PUNCTUATION] = TH_FG_SYNTAX_PUNCTUATION, [T_THINK_OFF] = TH_FG_THINKING_OFF, [T_THINK_MINIMAL] = TH_FG_THINKING_MINIMAL,
    [T_THINK_LOW] = TH_FG_THINKING_LOW, [T_THINK_MEDIUM] = TH_FG_THINKING_MEDIUM, [T_THINK_HIGH] = TH_FG_THINKING_HIGH,
    [T_THINK_XHIGH] = TH_FG_THINKING_XHIGH, [T_THINK_MAX] = TH_FG_THINKING_MAX, [T_BASH_MODE] = TH_FG_BASH_MODE,
    [T_CUSTOM_LABEL] = TH_FG_CUSTOM_MESSAGE_LABEL, [T_CUSTOM_TEXT] = TH_FG_CUSTOM_MESSAGE_TEXT,
    [T_SCROLL_TRACK] = TH_FG_SCROLLBAR_TRACK, [T_SCROLL_THUMB] = TH_FG_SCROLLBAR_THUMB, [T_SEARCH_TEXT] = TH_FG_SEARCH_MATCH_TEXT,
};

static ThemeSgr g_fg[THEME_NFG], g_bg[THEME_NBG];
static bool g_theme_ready;
int g_th_appearance; // 0 the terminal reported no colors, 1 dark, 2 light

void theme_set(const TermColors *tc, bool pending) {
    bool light = false;
    theme_system_ex(tc, theme_color_mode(), pending, g_fg, g_bg, NULL, NULL, &light);
    g_theme_ready = true;
    g_th_appearance = tc && tc->has_bg && !pending ? (light ? 2 : 1) : 0;
}

static void theme_init(void) {
    if (!g_theme_ready) theme_set(NULL, false);
}

int th_appearance(void) { return g_th_appearance; }

const char *th_open(ThemeColor c) {
    theme_init();
    return g_fg[theme_map[c]].open;
}

void th_fg(Buf *b, ThemeColor c, Str text) {
    theme_init();
    buf_adds(b, g_fg[theme_map[c]].open);
    buf_addstr(b, text);
    buf_adds(b, g_fg[theme_map[c]].close);
}

void th_bg(Buf *b, ThemeBgToken t, Str text) {
    theme_init();
    buf_adds(b, g_bg[t].open);
    buf_addstr(b, text);
    buf_adds(b, g_bg[t].close);
}

void th_bg_line(Buf *out, ThemeBgToken t, Str line, int width) {
    Buf p = {0};
    buf_addstr(&p, line);
    tui_spaces(&p, width - tui_width(line));
    th_bg(out, t, buf_str(&p));
    buf_free(&p);
}

void th_fgz(Buf *b, ThemeColor c, const char *text) { th_fg(b, c, str(text)); }

void th_bold(Buf *b, Str text) {
    buf_adds(b, "\x1b[1m");
    buf_addstr(b, text);
    buf_adds(b, "\x1b[22m");
}

void th_italic(Buf *b, Str text) {
    buf_adds(b, "\x1b[3m");
    buf_addstr(b, text);
    buf_adds(b, "\x1b[23m");
}

void th_underline(Buf *b, Str text) {
    buf_adds(b, "\x1b[4m");
    buf_addstr(b, text);
    buf_adds(b, "\x1b[24m");
}

void th_inverse(Buf *b, Str text) {
    buf_adds(b, "\x1b[7m");
    buf_addstr(b, text);
    buf_adds(b, "\x1b[27m");
}

void th_strike(Buf *b, Str text) {
    buf_adds(b, "\x1b[9m");
    buf_addstr(b, text);
    buf_adds(b, "\x1b[29m");
}

ThemeColor th_thinking_color(const char *level) {
    if (!level) return T_THINK_OFF;
    if (!strcmp(level, "minimal")) return T_THINK_MINIMAL;
    if (!strcmp(level, "low")) return T_THINK_LOW;
    if (!strcmp(level, "medium")) return T_THINK_MEDIUM;
    if (!strcmp(level, "high")) return T_THINK_HIGH;
    if (!strcmp(level, "xhigh")) return T_THINK_XHIGH;
    if (!strcmp(level, "max")) return T_THINK_MAX;
    return T_THINK_OFF;
}

// ---- basic components -----------------------------------------------------------------------------

const Lines *comp_lines(Comp *c, int width) {
    if (c->vt->live) {
        lines_free(&c->cache);
        c->vt->render(c, width, &c->cache);
        return &c->cache;
    }
    if (!c->cached || c->cache_w != width) {
        lines_free(&c->cache);
        c->vt->render(c, width, &c->cache);
        c->cache_w = width;
        c->cached = true;
    }
    return &c->cache;
}

void comp_render(Comp *c, int width, Lines *out) {
    if (c->vt->live) {
        c->vt->render(c, width, out);
        return;
    }
    const Lines *l = comp_lines(c, width);
    for (size_t i = 0; i < l->n; i++) lines_addz(out, l->v[i]);
}

void comp_invalidate(Comp *c) {
    if (c) c->cached = false;
}

void comp_free(Comp *c) {
    if (!c) return;
    lines_free(&c->cache);
    c->vt->destroy(c);
}

typedef struct {
    Comp base;
    Buf text;
    int pad_x, pad_y;
} TextComp;

static void text_render(Comp *c, int width, Lines *out) {
    TextComp *t = (TextComp *)c;
    Str s = buf_str(&t->text);
    if (!str_trim(s).len) return;
    Buf norm = {0};
    for (size_t i = 0; i < s.len; i++)
        if (s.ptr[i] == '\t') buf_adds(&norm, "   ");
        else buf_addc(&norm, s.ptr[i]);
    int px = MIN(t->pad_x, MAX(0, (width - 1) / 2));
    int cw = MAX(1, width - px * 2);
    Lines wrapped = {0};
    tui_wrap(buf_str(&norm), cw, &wrapped);
    for (int k = 0; k < t->pad_y; k++) {
        Buf e = {0};
        tui_spaces(&e, width);
        lines_addb(out, &e);
    }
    for (size_t k = 0; k < wrapped.n; k++) {
        Buf l = {0};
        tui_spaces(&l, px);
        buf_adds(&l, wrapped.v[k]);
        tui_spaces(&l, px);
        tui_spaces(&l, width - tui_width(buf_str(&l)));
        lines_addb(out, &l);
    }
    for (int k = 0; k < t->pad_y; k++) {
        Buf e = {0};
        tui_spaces(&e, width);
        lines_addb(out, &e);
    }
    lines_free(&wrapped);
    buf_free(&norm);
}

static void text_destroy(Comp *c) {
    buf_free(&((TextComp *)c)->text);
    free(c);
}

static const CompVt text_vt = {text_render, text_destroy, false};

Comp *comp_text(Str text, int pad_x, int pad_y) {
    TextComp *t = xcalloc(1, sizeof *t);
    t->base.vt = &text_vt;
    buf_addstr(&t->text, text);
    t->pad_x = pad_x;
    t->pad_y = pad_y;
    return &t->base;
}

void comp_text_set(Comp *c, Str text) {
    TextComp *t = (TextComp *)c;
    t->text.len = 0;
    buf_addstr(&t->text, text);
    comp_invalidate(c);
}

typedef struct {
    Comp base;
    int n;
} SpacerComp;

static void spacer_render(Comp *c, int width, Lines *out) {
    for (int k = 0; k < ((SpacerComp *)c)->n; k++) lines_addz(out, "");
}

static void plain_destroy(Comp *c) { free(c); }

static const CompVt spacer_vt = {spacer_render, plain_destroy, false};

Comp *comp_spacer(int n) {
    SpacerComp *s = xcalloc(1, sizeof *s);
    s->base.vt = &spacer_vt;
    s->n = n;
    return &s->base;
}

typedef struct {
    Comp base;
    CompVec kids;
} ContainerComp;

static void container_render(Comp *c, int width, Lines *out) {
    ContainerComp *k = (ContainerComp *)c;
    for (size_t i = 0; i < k->kids.len; i++) comp_render(k->kids.ptr[i], width, out);
}

static void container_destroy(Comp *c) {
    comp_clear(c);
    vec_free(&((ContainerComp *)c)->kids);
    free(c);
}

static const CompVt container_vt = {container_render, container_destroy, true};

Comp *comp_container(void) {
    ContainerComp *k = xcalloc(1, sizeof *k);
    k->base.vt = &container_vt;
    return &k->base;
}

CompVec *comp_children(Comp *c) { return &((ContainerComp *)c)->kids; }

void comp_add(Comp *container, Comp *child) { vec_push(&((ContainerComp *)container)->kids, child); }

void comp_clear(Comp *container) {
    ContainerComp *k = (ContainerComp *)container;
    for (size_t i = 0; i < k->kids.len; i++) comp_free(k->kids.ptr[i]);
    k->kids.len = 0;
}

// TruncatedText(text, paddingX, 0): the first line, cut to the width.
static void trunc_render(Comp *c, int width, Lines *out) {
    TextComp *t = (TextComp *)c;
    Str s = buf_str(&t->text);
    ptrdiff_t nl = str_find_char(s, '\n', 0);
    if (nl >= 0) s.len = (size_t)nl;
    Buf l = {0};
    tui_spaces(&l, t->pad_x);
    tui_truncate(&l, s, MAX(1, width - t->pad_x * 2), "...", false);
    tui_spaces(&l, t->pad_x);
    tui_spaces(&l, width - tui_width(buf_str(&l)));
    lines_addb(out, &l);
}

static const CompVt trunc_vt = {trunc_render, text_destroy, false};

Comp *comp_truncated(Str text, int pad_x) {
    TextComp *t = xcalloc(1, sizeof *t);
    t->base.vt = &trunc_vt;
    buf_addstr(&t->text, text);
    t->pad_x = pad_x;
    return &t->base;
}

typedef struct {
    Comp base;
    void (*fn)(void *ctx, int width, Lines *out);
    void *ctx;
} FnComp;

static void fn_render(Comp *c, int width, Lines *out) {
    FnComp *f = (FnComp *)c;
    f->fn(f->ctx, width, out);
}

static const CompVt fn_vt = {fn_render, plain_destroy, true};

Comp *comp_fn(void (*render)(void *ctx, int width, Lines *out), void *ctx) {
    FnComp *f = xcalloc(1, sizeof *f);
    f->base.vt = &fn_vt;
    f->fn = render;
    f->ctx = ctx;
    return &f->base;
}

// ---- select list -----------------------------------------------------------------------------------

void sel_init(SelectList *s, int max_visible, int min_primary, int max_primary) {
    memset(s, 0, sizeof *s);
    s->max_visible = max_visible;
    s->min_primary = min_primary;
    s->max_primary = max_primary;
}

static char *dupz(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

void sel_add(SelectList *s, const char *value, const char *label, const char *description) {
    s->items = xrealloc(s->items, (s->n + 1) * sizeof *s->items);
    s->items[s->n++] = (SelItem){dupz(value), dupz(label), description && *description ? dupz(description) : NULL};
}

void sel_free(SelectList *s) {
    for (size_t i = 0; i < s->n; i++) {
        free(s->items[i].value);
        free(s->items[i].label);
        free(s->items[i].description);
    }
    free(s->items);
    s->items = NULL;
    s->n = 0;
}

const SelItem *sel_current(const SelectList *s) { return s->selected < s->n ? &s->items[s->selected] : NULL; }

void sel_up(SelectList *s) {
    if (!s->n) return;
    s->selected = s->selected == 0 ? s->n - 1 : s->selected - 1;
}

void sel_down(SelectList *s) {
    if (!s->n) return;
    s->selected = s->selected == s->n - 1 ? 0 : s->selected + 1;
}

static const char *display_value(const SelItem *it) { return it->label && *it->label ? it->label : it->value; }

// The description on one line: line breaks as spaces, trimmed.
static void single_line(Buf *b, const char *d) {
    for (const char *p = d; *p; p++) {
        if (*p == '\r' || *p == '\n') {
            while (p[1] == '\r' || p[1] == '\n') p++;
            buf_addc(b, ' ');
        } else buf_addc(b, *p);
    }
    Str t = str_trim(buf_str(b));
    memmove(b->ptr, t.ptr, t.len);
    b->len = t.len;
    b->ptr[b->len] = 0;
}

void sel_render(SelectList *s, int width, Lines *out) {
    if (!s->n) {
        Buf b = {0};
        th_fgz(&b, T_MUTED, "  No matching commands");
        lines_addb(out, &b);
        return;
    }
    int lo = MAX(1, MIN(s->min_primary, s->max_primary)), hi = MAX(1, MAX(s->min_primary, s->max_primary));
    int widest = 0;
    for (size_t i = 0; i < s->n; i++) widest = MAX(widest, tui_width(str(display_value(&s->items[i]))) + 2);
    int primary = CLAMP(widest, lo, hi);
    long start = (long)s->selected - s->max_visible / 2;
    long maxstart = (long)s->n - s->max_visible;
    if (start > maxstart) start = maxstart;
    if (start < 0) start = 0;
    size_t end = MIN((size_t)start + (size_t)s->max_visible, s->n);
    for (size_t i = (size_t)start; i < end; i++) {
        const SelItem *it = &s->items[i];
        bool selected = i == s->selected;
        const char *prefix = selected ? "\xe2\x86\x92 " : "  ";
        int pw = 2;
        Buf line = {0};
        bool done = false;
        if (it->description && width > 40) {
            Buf desc = {0};
            single_line(&desc, it->description);
            if (desc.len) {
                int eff = MAX(1, MIN(primary, width - pw - 4));
                int maxpw = MAX(1, eff - 2);
                Buf tv = {0};
                tui_truncate(&tv, str(display_value(it)), maxpw, "", false);
                int tvw = tui_width(buf_str(&tv));
                int spacing = MAX(1, eff - tvw);
                int remaining = width - (pw + tvw + spacing) - 2;
                if (remaining > 10) {
                    Buf td = {0};
                    tui_truncate(&td, buf_str(&desc), remaining, "", false);
                    if (selected) {
                        Buf all = {0};
                        buf_adds(&all, prefix);
                        buf_addstr(&all, buf_str(&tv));
                        tui_spaces(&all, spacing);
                        buf_addstr(&all, buf_str(&td));
                        th_fg(&line, T_ACCENT, buf_str(&all));
                        buf_free(&all);
                    } else {
                        buf_adds(&line, prefix);
                        buf_addstr(&line, buf_str(&tv));
                        Buf d = {0};
                        tui_spaces(&d, spacing);
                        buf_addstr(&d, buf_str(&td));
                        th_fg(&line, T_MUTED, buf_str(&d));
                        buf_free(&d);
                    }
                    buf_free(&td);
                    done = true;
                }
                buf_free(&tv);
            }
            buf_free(&desc);
        }
        if (!done) {
            Buf tv = {0};
            tui_truncate(&tv, str(display_value(it)), width - pw - 2, "", false);
            if (selected) {
                Buf all = {0};
                buf_adds(&all, prefix);
                buf_addstr(&all, buf_str(&tv));
                th_fg(&line, T_ACCENT, buf_str(&all));
                buf_free(&all);
            } else {
                buf_adds(&line, prefix);
                buf_addstr(&line, buf_str(&tv));
            }
            buf_free(&tv);
        }
        lines_addb(out, &line);
    }
    if (start > 0 || end < s->n) {
        char info[64];
        snprintf(info, sizeof info, "  (%zu/%zu)", s->selected + 1, s->n);
        Buf t = {0}, b = {0};
        tui_truncate(&t, str(info), width - 2, "", false);
        th_fg(&b, T_MUTED, buf_str(&t));
        lines_addb(out, &b);
        buf_free(&t);
    }
}
