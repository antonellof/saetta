// Saetta Code TUI: markdown rendering (headings, emphasis, code spans and blocks with syntax
// highlighting, lists, tables, quotes, links, rules) and the code highlighter (see code_tui.h).
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at v1.5.2,
// b4afddd): packages/tui/src/components/markdown.ts (the rendering of each token), with a block
// and inline lexer written after the rules of marked 18 (the parser pi uses), and
// packages/coding-agent/src/utils/syntax-highlight.ts with modes/interactive/theme/theme.ts
// (highlightCode, the highlight.js scopes and their theme colors). The highlighter is a small
// lexer per language family, not highlight.js: it gives the scopes highlight.js gives for
// keywords, literals, built-ins, strings, numbers, comments and function signatures.
// The block and inline rules follow marked (MIT, Copyright (c) 2018+, MarkedJS; Copyright (c)
// 2011-2018, Christopher Jeffrey). The keyword, literal and built-in lists of the highlighter follow
// highlight.js 10.7.3's language definitions (BSD-3-Clause, Copyright (c) 2006, Ivan Sagalaev).
#include "code_tui.h"

#include <ctype.h>
#include <stdlib.h>

// ---- chalk-style attributes (theme.bold and friends nest as chalk does) ---------------------------

static void chalk(Buf *b, const char *open, const char *close, Str text) {
    Buf t = {0};
    size_t cl = strlen(close);
    for (size_t i = 0; i < text.len;) {
        if (text.ptr[i] == 0x1b && i + cl <= text.len && !memcmp(text.ptr + i, close, cl)) {
            buf_adds(&t, close);
            buf_adds(&t, open);
            i += cl;
        } else if (text.ptr[i] == '\n' || (text.ptr[i] == '\r' && i + 1 < text.len && text.ptr[i + 1] == '\n')) {
            buf_adds(&t, close);
            if (text.ptr[i] == '\r') buf_addc(&t, text.ptr[i++]);
            buf_addc(&t, text.ptr[i++]);
            buf_adds(&t, open);
        } else buf_addc(&t, text.ptr[i++]);
    }
    buf_adds(b, open);
    buf_addstr(b, buf_str(&t));
    buf_adds(b, close);
    buf_free(&t);
}

static void c_bold(Buf *b, Str t) { chalk(b, "\x1b[1m", "\x1b[22m", t); }
static void c_italic(Buf *b, Str t) { chalk(b, "\x1b[3m", "\x1b[23m", t); }
static void c_underline(Buf *b, Str t) { chalk(b, "\x1b[4m", "\x1b[24m", t); }
static void c_strike(Buf *b, Str t) { chalk(b, "\x1b[9m", "\x1b[29m", t); }

// ---- tokens -----------------------------------------------------------------------------------------

typedef enum { IN_TEXT, IN_ESCAPE, IN_STRONG, IN_EM, IN_CODE, IN_LINK, IN_DEL, IN_BR, IN_HTML, IN_IMAGE } InKind;
typedef struct Inline Inline;
struct Inline {
    InKind kind;
    Str raw, text, href;
    Inline *kids;
    size_t nkids;
};

typedef enum { B_SPACE, B_PARAGRAPH, B_HEADING, B_CODE, B_HR, B_QUOTE, B_LIST, B_TABLE, B_HTML, B_TEXT } BKind;
typedef struct Block Block;
typedef struct {
    Block *blocks;
    size_t n;
    Str raw;
    bool task, checked;
} ListItem;
struct Block {
    BKind kind;
    Str raw, text, lang;
    int depth;
    Inline *in;
    size_t nin;
    Block *kids; // blockquote content
    size_t nkids;
    ListItem *items;
    size_t nitems;
    bool ordered, loose;
    long start;
    // table: header cells then rows
    Inline **cells;
    size_t *ncells;
    size_t cols, rows; // rows excludes the header
};

typedef VEC(Block) BlockVec;
typedef VEC(Inline) InlineVec;

// ---- inline lexer ---------------------------------------------------------------------------------

static bool is_punct(uint32_t c) {
    return c < 0x80 ? (ispunct((int)c) != 0) : (c >= 0x2000 && c <= 0x206f) || (c >= 0x3000 && c <= 0x303f) || (c >= 0xff00 && c <= 0xff0f);
}

static bool is_space_cp(uint32_t c) { return tui_is_space(c); }

static uint32_t cp_before(Str s, size_t i) {
    if (i == 0) return '\n';
    size_t k = i - 1;
    while (k > 0 && ((unsigned char)s.ptr[k] & 0xc0) == 0x80) k--;
    return tui_cp(s, &k);
}

static uint32_t cp_at(Str s, size_t i) {
    if (i >= s.len) return '\n';
    size_t k = i;
    return tui_cp(s, &k);
}

static void lex_inline(Arena *a, Str s, InlineVec *out);

// Nesting is bounded (emphasis in links in emphasis..., quotes in lists in quotes...): past these
// depths the rest is plain text, so deep input cannot exhaust the stack.
#define MAX_INLINE_DEPTH 24
#define MAX_BLOCK_DEPTH 32
static int g_inline_depth, g_block_depth;

static Inline *inline_list(Arena *a, Str s, size_t *n) {
    InlineVec v = {0};
    if (g_inline_depth >= MAX_INLINE_DEPTH) {
        if (s.len) vec_push(&v, ((Inline){.kind = IN_TEXT, .raw = s, .text = s}));
    } else {
        g_inline_depth++;
        lex_inline(a, s, &v);
        g_inline_depth--;
    }
    Inline *arr = arena_alloc(a, (v.len + 1) * sizeof *arr);
    memcpy(arr, v.ptr, v.len * sizeof *arr);
    *n = v.len;
    vec_free(&v);
    return arr;
}

static void push_text(Arena *a, InlineVec *out, Str t) {
    if (!t.len) return;
    if (out->len && out->ptr[out->len - 1].kind == IN_TEXT && out->ptr[out->len - 1].raw.ptr + out->ptr[out->len - 1].raw.len == t.ptr) {
        Inline *l = &out->ptr[out->len - 1];
        l->raw.len += t.len;
        l->text = l->raw;
        return;
    }
    (void)a;
    vec_push(out, ((Inline){.kind = IN_TEXT, .raw = t, .text = t}));
}

// A delimiter run of c at s[i] that can open / close (CommonMark flanking rules).
static size_t run_len(Str s, size_t i, char c) {
    size_t j = i;
    while (j < s.len && s.ptr[j] == c) j++;
    return j - i;
}

static bool left_flanking(Str s, size_t i, size_t n) {
    uint32_t after = cp_at(s, i + n), before = cp_before(s, i);
    if (is_space_cp(after)) return false;
    if (is_punct(after) && !(is_space_cp(before) || is_punct(before))) return false;
    return true;
}

static bool right_flanking(Str s, size_t i, size_t n) {
    uint32_t after = cp_at(s, i + n), before = cp_before(s, i);
    if (is_space_cp(before)) return false;
    if (is_punct(before) && !(is_space_cp(after) || is_punct(after))) return false;
    return true;
}

// What one lex_inline call learned about its text, so an unmatched delimiter does not make every
// later one scan to the end again (the searches below are each linear; with these, the whole lexer
// stays near linear): the first position from which a search was seen to fail.
#define MEMO_RUNS 33
typedef struct {
    size_t code_fail[MEMO_RUNS]; // a closing backtick run of this length
    size_t em_fail[2][4];        // a closing '*' / '_' run of this size
    size_t del_fail;             // a closing "~~"
    size_t *bracket;             // the matching ']' of each '[' (SIZE_MAX: none)
} Memo;

static void memo_init(Memo *m) {
    for (int i = 0; i < MEMO_RUNS; i++) m->code_fail[i] = SIZE_MAX;
    for (int i = 0; i < 2; i++)
        for (int k = 0; k < 4; k++) m->em_fail[i][k] = SIZE_MAX;
    m->del_fail = SIZE_MAX;
    m->bracket = NULL;
}

// The closing backtick run of length r for a code span whose content starts at from, or -1.
static long code_close(Str s, size_t from, size_t r, Memo *m) {
    if (r < MEMO_RUNS && from >= m->code_fail[r]) return -1;
    size_t j = from;
    while (j < s.len) {
        if (s.ptr[j] == '`') {
            size_t r2 = run_len(s, j, '`');
            if (r2 == r) return (long)j;
            j += r2;
        } else j++;
    }
    if (r < MEMO_RUNS && from < m->code_fail[r]) m->code_fail[r] = from;
    return -1;
}

// The closing run for an emphasis opened at i with n delimiters (n = 1 or 2), or -1.
static long find_close(Str s, size_t from, char c, size_t n, Memo *m) {
    size_t i = from;
    while (i < s.len) {
        char ch = s.ptr[i];
        if (ch == '\\' && i + 1 < s.len) {
            i += 2;
            continue;
        }
        if (ch == '`') {
            size_t r = run_len(s, i, '`');
            // skip a code span
            long cl = code_close(s, i + r, r, m);
            i = cl >= 0 ? (size_t)cl + r : i + r;
            continue;
        }
        if (ch == c) {
            size_t r = run_len(s, i, c);
            bool closes = right_flanking(s, i, r) && (c == '*' || !left_flanking(s, i, r) || is_punct(cp_at(s, i + r)));
            if (closes && r >= n) {
                if (r == n || (n == 2 && r == 3) || (n == 1 && r == 3)) return (long)(i + (r - n == 1 && n == 2 ? 0 : 0));
                return (long)i;
            }
            // a nested opener of the other size: skip its run
            i += r;
            continue;
        }
        i++;
    }
    return -1;
}

// A C0 or C1 control or DEL at s[i]: never part of a link destination.
static bool ctl_byte(Str s, size_t i) {
    unsigned char c = (unsigned char)s.ptr[i];
    return c < 0x20 || c == 0x7f || (c == 0xc2 && i + 1 < s.len && (unsigned char)s.ptr[i + 1] >= 0x80 && (unsigned char)s.ptr[i + 1] <= 0x9f);
}

static bool has_ctl(Str s) {
    for (size_t i = 0; i < s.len; i++)
        if (ctl_byte(s, i)) return true;
    return false;
}

// s without its controls (a link destination as written in an OSC 8 sequence or shown).
static void no_ctl(Buf *b, Str s) {
    for (size_t i = 0; i < s.len; i++) {
        if (ctl_byte(s, i)) {
            if ((unsigned char)s.ptr[i] == 0xc2) i++;
            continue;
        }
        buf_addc(b, s.ptr[i]);
    }
    if (!b->ptr) buf_adds(b, "");
}

static bool scheme_char(char c) { return isalnum((unsigned char)c) || c == '+' || c == '.' || c == '-'; }

// The matching ']' of every '[' (escapes and code spans skipped), in one pass.
static void match_brackets(Arena *a, Str s, Memo *m) {
    m->bracket = arena_alloc(a, (s.len + 1) * sizeof *m->bracket);
    VEC(size_t) stack = {0};
    for (size_t j = 0; j < s.len; j++) {
        m->bracket[j] = SIZE_MAX;
        char ch = s.ptr[j];
        if (ch == '\\') {
            if (j + 1 < s.len) m->bracket[j + 1] = SIZE_MAX;
            j++;
            continue;
        }
        if (ch == '`') {
            size_t r = run_len(s, j, '`');
            long cl = code_close(s, j + r, r, m);
            size_t end = cl >= 0 ? (size_t)cl + r : j + r;
            for (size_t q = j + 1; q < end && q < s.len; q++) m->bracket[q] = SIZE_MAX;
            j = end - 1;
            continue;
        }
        if (ch == '[') vec_push(&stack, j);
        else if (ch == ']' && stack.len) m->bracket[stack.ptr[--stack.len]] = j;
    }
    vec_free(&stack);
}

static void lex_inline(Arena *a, Str s, InlineVec *out) {
    size_t i = 0, text_start = 0;
    Memo memo, *mm = &memo;
    memo_init(mm);
#define FLUSH_TEXT()                                                       \
    do {                                                                   \
        if (i > text_start) push_text(a, out, str_sub(s, text_start, i)); \
    } while (0)
    while (i < s.len) {
        char c = s.ptr[i];
        // escape
        if (c == '\\' && i + 1 < s.len && ispunct((unsigned char)s.ptr[i + 1])) {
            FLUSH_TEXT();
            vec_push(out, ((Inline){.kind = IN_ESCAPE, .raw = str_sub(s, i, i + 2), .text = str_sub(s, i + 1, i + 2)}));
            i += 2;
            text_start = i;
            continue;
        }
        // hard break: two spaces or a backslash before a newline
        if (c == '\n' && i >= 2 && s.ptr[i - 1] == ' ' && s.ptr[i - 2] == ' ' && i + 1 < s.len) {
            size_t k = i;
            while (k > text_start && s.ptr[k - 1] == ' ') k--;
            if (k > text_start) push_text(a, out, str_sub(s, text_start, k));
            vec_push(out, ((Inline){.kind = IN_BR, .raw = str_sub(s, k, i + 1)}));
            i++;
            text_start = i;
            continue;
        }
        if (c == '\\' && i + 1 < s.len && s.ptr[i + 1] == '\n' && i + 2 < s.len) {
            FLUSH_TEXT();
            vec_push(out, ((Inline){.kind = IN_BR, .raw = str_sub(s, i, i + 2)}));
            i += 2;
            text_start = i;
            continue;
        }
        // code span
        if (c == '`') {
            size_t r = run_len(s, i, '`');
            long close = code_close(s, i + r, r, mm);
            if (close >= 0) {
                FLUSH_TEXT();
                Str inner = str_sub(s, i + r, (size_t)close);
                // newlines as spaces; one space stripped from both ends when both have one
                char *t = arena_strndup(a, inner.ptr, inner.len);
                for (size_t k = 0; k < inner.len; k++)
                    if (t[k] == '\n') t[k] = ' ';
                Str ts = str_n(t, inner.len);
                bool all_space = true;
                for (size_t k = 0; k < ts.len; k++) all_space &= ts.ptr[k] == ' ';
                if (!all_space && ts.len >= 2 && ts.ptr[0] == ' ' && ts.ptr[ts.len - 1] == ' ') ts = str_sub(ts, 1, ts.len - 1);
                vec_push(out, ((Inline){.kind = IN_CODE, .raw = str_sub(s, i, (size_t)close + r), .text = ts}));
                i = (size_t)close + r;
                text_start = i;
                continue;
            }
            i += r;
            continue;
        }
        // autolink <scheme:...> or <email>, or inline html
        if (c == '<') {
            size_t j = i + 1;
            while (j < s.len && scheme_char(s.ptr[j])) j++;
            if (j < s.len && s.ptr[j] == ':' && j - i - 1 >= 2 && isalpha((unsigned char)s.ptr[i + 1])) {
                size_t k = j + 1;
                while (k < s.len && s.ptr[k] != '>' && s.ptr[k] != '<' && !isspace((unsigned char)s.ptr[k]) && !ctl_byte(s, k)) k++;
                if (k < s.len && s.ptr[k] == '>') {
                    FLUSH_TEXT();
                    Str url = str_sub(s, i + 1, k);
                    Inline l = {.kind = IN_LINK, .raw = str_sub(s, i, k + 1), .text = url, .href = url};
                    l.kids = arena_alloc(a, sizeof(Inline));
                    l.kids[0] = (Inline){.kind = IN_TEXT, .raw = url, .text = url};
                    l.nkids = 1;
                    vec_push(out, l);
                    i = k + 1;
                    text_start = i;
                    continue;
                }
            }
            if (i + 1 < s.len && (isalpha((unsigned char)s.ptr[i + 1]) || s.ptr[i + 1] == '/' || s.ptr[i + 1] == '!')) {
                size_t k = i + 1;
                while (k < s.len && s.ptr[k] != '>' && s.ptr[k] != '<' && s.ptr[k] != '\n') k++;
                if (k < s.len && s.ptr[k] == '>' && s.ptr[i + 1] != ' ') {
                    FLUSH_TEXT();
                    vec_push(out, ((Inline){.kind = IN_HTML, .raw = str_sub(s, i, k + 1), .text = str_sub(s, i, k + 1)}));
                    i = k + 1;
                    text_start = i;
                    continue;
                }
            }
        }
        // link and image: [label](href "title")
        if (c == '[' || (c == '!' && i + 1 < s.len && s.ptr[i + 1] == '[')) {
            size_t lb = c == '!' ? i + 1 : i;
            if (!mm->bracket) match_brackets(a, s, mm);
            long rb = mm->bracket[lb] == SIZE_MAX ? -1 : (long)mm->bracket[lb];
            if (rb >= 0 && (size_t)rb + 1 < s.len && s.ptr[rb + 1] == '(') {
                size_t k = (size_t)rb + 2;
                while (k < s.len && (s.ptr[k] == ' ' || s.ptr[k] == '\t')) k++;
                size_t hs = k, he;
                bool angle = k < s.len && s.ptr[k] == '<';
                int par = 0;
                if (angle) {
                    while (k < s.len && s.ptr[k] != '>' && s.ptr[k] != '\n') k++;
                    he = k;
                    hs++;
                    if (k < s.len) k++;
                } else {
                    while (k < s.len && !isspace((unsigned char)s.ptr[k]) && (s.ptr[k] != ')' || par > 0)) {
                        if (s.ptr[k] == '(') par++;
                        else if (s.ptr[k] == ')') par--;
                        else if (s.ptr[k] == '\\') k++;
                        k++;
                    }
                    he = k;
                }
                while (k < s.len && (s.ptr[k] == ' ' || s.ptr[k] == '\t' || s.ptr[k] == '\n')) k++;
                if (k < s.len && (s.ptr[k] == '"' || s.ptr[k] == '\'' || s.ptr[k] == '(')) {
                    char q = s.ptr[k] == '(' ? ')' : s.ptr[k];
                    k++;
                    while (k < s.len && s.ptr[k] != q) k++;
                    if (k < s.len) k++;
                    while (k < s.len && (s.ptr[k] == ' ' || s.ptr[k] == '\t')) k++;
                }
                if (k < s.len && s.ptr[k] == ')') {
                    FLUSH_TEXT();
                    Str label = str_sub(s, lb + 1, (size_t)rb);
                    Inline l = {.kind = c == '!' ? IN_IMAGE : IN_LINK, .raw = str_sub(s, i, k + 1), .text = label,
                                .href = str_sub(s, hs, he)};
                    l.kids = inline_list(a, label, &l.nkids);
                    vec_push(out, l);
                    i = k + 1;
                    text_start = i;
                    continue;
                }
            }
        }
        // bare URLs (GFM): http(s)://, ftp://, www.
        if ((c == 'h' || c == 'H' || c == 'w' || c == 'W' || c == 'f' || c == 'F') &&
            (i == 0 || !isalnum((unsigned char)s.ptr[i - 1]))) {
            Str rest = str_sub(s, i, s.len);
            size_t pre = 0;
            if (rest.len > 8 && !strncasecmp(rest.ptr, "https://", 8)) pre = 8;
            else if (rest.len > 7 && !strncasecmp(rest.ptr, "http://", 7)) pre = 7;
            else if (rest.len > 6 && !strncasecmp(rest.ptr, "ftp://", 6)) pre = 6;
            else if (rest.len > 4 && !strncasecmp(rest.ptr, "www.", 4)) pre = 4;
            if (pre) {
                size_t k = i + pre;
                while (k < s.len && !isspace((unsigned char)s.ptr[k]) && s.ptr[k] != '<' && !ctl_byte(s, k)) k++;
                // trailing punctuation is not part of the URL
                while (k > i + pre && strchr("?!.,:*_~'\"", s.ptr[k - 1])) k--;
                if (k > i + pre && s.ptr[k - 1] == ')') {
                    int bal = 0;
                    for (size_t q = i; q < k; q++) bal += s.ptr[q] == '(' ? 1 : s.ptr[q] == ')' ? -1 : 0;
                    if (bal < 0) k--;
                }
                if (k > i + pre) {
                    FLUSH_TEXT();
                    Str url = str_sub(s, i, k);
                    Str href = url;
                    if (pre == 4) href = str(arena_printf(a, "http://%.*s", (int)url.len, url.ptr));
                    Inline l = {.kind = IN_LINK, .raw = url, .text = url, .href = href};
                    l.kids = arena_alloc(a, sizeof(Inline));
                    l.kids[0] = (Inline){.kind = IN_TEXT, .raw = url, .text = url};
                    l.nkids = 1;
                    vec_push(out, l);
                    i = k;
                    text_start = i;
                    continue;
                }
            }
        }
        // strikethrough ~~x~~ (strict: no space inside the delimiters)
        if (c == '~' && i + 2 < s.len && s.ptr[i + 1] == '~' && s.ptr[i + 2] != '~' && !isspace((unsigned char)s.ptr[i + 2])) {
            size_t j = i + 2;
            long close = -1;
            if (j >= mm->del_fail) j = s.len; // known: no closing "~~" from here
            size_t from = j;
            while (j + 1 < s.len) {
                if (s.ptr[j] == '\\') {
                    j += 2;
                    continue;
                }
                if (s.ptr[j] == '~' && s.ptr[j + 1] == '~' && !isspace((unsigned char)s.ptr[j - 1]) && s.ptr[j - 1] != '~' &&
                    (j + 2 >= s.len || s.ptr[j + 2] != '~')) {
                    close = (long)j;
                    break;
                }
                j++;
            }
            if (close < 0 && from < mm->del_fail) mm->del_fail = from;
            if (close >= 0) {
                FLUSH_TEXT();
                Str inner = str_sub(s, i + 2, (size_t)close);
                Inline d = {.kind = IN_DEL, .raw = str_sub(s, i, (size_t)close + 2), .text = inner};
                d.kids = inline_list(a, inner, &d.nkids);
                vec_push(out, d);
                i = (size_t)close + 2;
                text_start = i;
                continue;
            }
        }
        // emphasis and strong
        if (c == '*' || c == '_') {
            size_t r = run_len(s, i, c);
            bool opens = left_flanking(s, i, r) && (c == '*' || !right_flanking(s, i, r) || is_punct(cp_before(s, i)));
            if (opens) {
                size_t n = r >= 2 ? 2 : 1;
                if (r >= 3) n = 3;
                bool done = false;
                for (size_t want = n; want >= 1 && !done; want--) {
                    size_t use = want == 3 ? 3 : want;
                    long close = -1;
                    // search a closing run of at least `use`
                    size_t j = i + use;
                    size_t *fail = &mm->em_fail[c == '*' ? 0 : 1][use];
                    if (j >= *fail) j = s.len;
                    while (j < s.len) {
                        long cpos = find_close(s, j, c, use == 3 ? 2 : use, mm);
                        if (cpos < 0) {
                            if (j < *fail) *fail = j;
                            break;
                        }
                        size_t cr = run_len(s, (size_t)cpos, c);
                        if (cr >= use) {
                            close = cpos;
                            break;
                        }
                        j = (size_t)cpos + cr;
                    }
                    if (close >= 0 && (size_t)close > i + use) {
                        FLUSH_TEXT();
                        Str inner = str_sub(s, i + use, (size_t)close);
                        if (use == 3) {
                            Inline em = {.kind = IN_EM, .raw = str_sub(s, i, (size_t)close + 3), .text = inner};
                            em.kids = arena_alloc(a, sizeof(Inline));
                            Inline st = {.kind = IN_STRONG, .raw = str_sub(s, i + 1, (size_t)close + 2), .text = inner};
                            st.kids = inline_list(a, inner, &st.nkids);
                            em.kids[0] = st;
                            em.nkids = 1;
                            vec_push(out, em);
                        } else {
                            Inline e = {.kind = use == 2 ? IN_STRONG : IN_EM, .raw = str_sub(s, i, (size_t)close + use), .text = inner};
                            e.kids = inline_list(a, inner, &e.nkids);
                            vec_push(out, e);
                        }
                        i = (size_t)close + use;
                        text_start = i;
                        done = true;
                    }
                    if (want == 1) break;
                }
                if (done) continue;
            }
            i += r;
            continue;
        }
        i++;
    }
    FLUSH_TEXT();
#undef FLUSH_TEXT
}

// ---- block lexer ----------------------------------------------------------------------------------

typedef struct {
    Str s;
    size_t pos;
} Src;

static Str line_at(Str s, size_t pos, size_t *next) {
    size_t j = pos;
    while (j < s.len && s.ptr[j] != '\n') j++;
    *next = j < s.len ? j + 1 : j;
    return str_sub(s, pos, j);
}

static bool blank(Str l) {
    for (size_t i = 0; i < l.len; i++)
        if (l.ptr[i] != ' ' && l.ptr[i] != '\t') return false;
    return true;
}

static size_t indent_of(Str l) {
    size_t n = 0;
    while (n < l.len && l.ptr[n] == ' ') n++;
    return n;
}

static bool is_hr(Str l) {
    size_t i = indent_of(l);
    if (i > 3 || i >= l.len) return false;
    char c = l.ptr[i];
    if (c != '-' && c != '_' && c != '*') return false;
    int n = 0;
    for (; i < l.len; i++) {
        if (l.ptr[i] == c) n++;
        else if (l.ptr[i] != ' ' && l.ptr[i] != '\t') return false;
    }
    return n >= 3;
}

static int heading_level(Str l) {
    size_t i = indent_of(l);
    if (i > 3) return 0;
    int n = 0;
    while (i < l.len && l.ptr[i] == '#') {
        n++;
        i++;
    }
    if (n < 1 || n > 6) return 0;
    if (i < l.len && l.ptr[i] != ' ' && l.ptr[i] != '\t') return 0;
    return n;
}

static bool is_fence(Str l, char *ch, size_t *n) {
    size_t i = indent_of(l);
    if (i > 3 || i >= l.len) return false;
    char c = l.ptr[i];
    if (c != '`' && c != '~') return false;
    size_t r = 0;
    while (i + r < l.len && l.ptr[i + r] == c) r++;
    if (r < 3) return false;
    if (c == '`' && memchr(l.ptr + i + r, '`', l.len - i - r)) return false;
    *ch = c;
    *n = r;
    return true;
}

// A list item marker: returns the width of "marker + spaces" (0 if none); ordered, its number.
static size_t list_marker(Str l, bool *ordered, long *num, char *bullet) {
    size_t i = indent_of(l);
    if (i > 3 + 0 && false) return 0;
    if (i >= l.len) return 0;
    char c = l.ptr[i];
    size_t m = i;
    if (c == '-' || c == '+' || c == '*') {
        *ordered = false;
        *bullet = c;
        m = i + 1;
    } else if (isdigit((unsigned char)c)) {
        size_t d = i;
        long v = 0;
        while (d < l.len && isdigit((unsigned char)l.ptr[d]) && d - i < 9) v = v * 10 + (l.ptr[d++] - '0');
        if (d >= l.len || (l.ptr[d] != '.' && l.ptr[d] != ')')) return 0;
        *ordered = true;
        *num = v;
        *bullet = l.ptr[d];
        m = d + 1;
    } else return 0;
    if (m < l.len && l.ptr[m] != ' ' && l.ptr[m] != '\t') return 0;
    size_t sp = m;
    while (sp < l.len && (l.ptr[sp] == ' ' || l.ptr[sp] == '\t') && sp - m < 4) sp++;
    if (sp >= l.len) sp = m + 1 <= l.len ? m + (m < l.len) : m; // an empty item
    else if (sp - m > 4) sp = m + 1;
    return sp;
}

static bool table_delim(Str l) {
    size_t i = indent_of(l);
    if (i > 3) return false;
    bool dash = false;
    for (; i < l.len; i++) {
        char c = l.ptr[i];
        if (c == '-') dash = true;
        else if (c != '|' && c != ':' && c != ' ' && c != '\t') return false;
    }
    return dash;
}

static void split_cells(Arena *a, Str row, VecStr *cells) {
    Str r = str_trim(row);
    if (r.len && r.ptr[0] == '|') r = str_sub(r, 1, r.len);
    if (r.len && r.ptr[r.len - 1] == '|' && !(r.len >= 2 && r.ptr[r.len - 2] == '\\')) r.len--;
    size_t s = 0;
    for (size_t i = 0; i <= r.len; i++) {
        if (i == r.len || (r.ptr[i] == '|' && (i == 0 || r.ptr[i - 1] != '\\'))) {
            Str cell = str_trim(str_sub(r, s, i));
            // \| becomes |
            Buf b = {0};
            for (size_t k = 0; k < cell.len; k++) {
                if (cell.ptr[k] == '\\' && k + 1 < cell.len && cell.ptr[k + 1] == '|') continue;
                buf_addc(&b, cell.ptr[k]);
            }
            vec_push(cells, str_n(arena_strndup(a, b.ptr ? b.ptr : "", b.len), b.len));
            buf_free(&b);
            s = i + 1;
        }
    }
}

static void lex_blocks(Arena *a, Str s, BlockVec *out, bool top);

// Lines that interrupt a paragraph.
static bool interrupts(Str l) {
    char fc;
    size_t fn;
    bool o;
    long num;
    char bl;
    if (is_hr(l) || heading_level(l) || is_fence(l, &fc, &fn)) return true;
    size_t i = indent_of(l);
    if (i <= 3 && i < l.len && l.ptr[i] == '>') return true;
    size_t m = list_marker(l, &o, &num, &bl);
    if (m && indent_of(l) <= 3) {
        // a list interrupts a paragraph only with content, and an ordered one only from 1
        Str rest = str_sub(l, m, l.len);
        if (blank(rest)) return false;
        if (o && num != 1) return false;
        return true;
    }
    if (i <= 3 && i + 1 < l.len && l.ptr[i] == '<' && (isalpha((unsigned char)l.ptr[i + 1]) || l.ptr[i + 1] == '/' || l.ptr[i + 1] == '!'))
        return true;
    return false;
}

static Block mk(BKind k, Str raw) { return (Block){.kind = k, .raw = raw}; }

// The blocks inside a quote or a list item; past MAX_BLOCK_DEPTH one paragraph of the text.
static void nested_blocks(Arena *a, Str text, BlockVec *kids) {
    if (g_block_depth >= MAX_BLOCK_DEPTH) {
        Block b = mk(B_PARAGRAPH, text);
        b.text = text;
        b.in = inline_list(a, text, &b.nin);
        vec_push(kids, b);
        return;
    }
    g_block_depth++;
    lex_blocks(a, text, kids, false);
    g_block_depth--;
}

static void lex_list(Arena *a, Str s, size_t *pos, BlockVec *out) {
    size_t start = *pos;
    bool ordered;
    long num = 1;
    char bullet;
    size_t nx;
    Str first = line_at(s, *pos, &nx);
    size_t mw = list_marker(first, &ordered, &num, &bullet);
    Block b = mk(B_LIST, S(""));
    b.ordered = ordered;
    b.start = num;
    VEC(ListItem) items = {0};
    bool loose = false, blank_between = false;
    size_t p = *pos;
    while (p < s.len) {
        Str l = line_at(s, p, &nx);
        bool o2;
        long n2;
        char b2;
        size_t m2 = list_marker(l, &o2, &n2, &b2);
        if (!m2 || o2 != ordered || b2 != bullet || indent_of(l) > 3) break;
        if (items.len && blank_between) loose = true;
        // the item: its first line, then lines indented to its content, blank lines inside
        size_t content_indent = m2;
        Buf body = {0};
        Str rest = str_sub(l, m2, l.len);
        buf_addstr(&body, rest);
        size_t item_start = p;
        p = nx;
        blank_between = false;
        size_t blank_start = p;
        bool item_blank = false;
        while (p < s.len) {
            Str ln = line_at(s, p, &nx);
            if (blank(ln)) {
                // a blank line continues the item only if what follows is indented
                size_t q = nx;
                while (q < s.len) {
                    size_t qn;
                    Str ql = line_at(s, q, &qn);
                    if (!blank(ql)) break;
                    q = qn;
                }
                size_t qn;
                Str ql = q < s.len ? line_at(s, q, &qn) : S("");
                if (q < s.len && indent_of(ql) >= content_indent) {
                    buf_addc(&body, '\n');
                    item_blank = true;
                    p = nx;
                    continue;
                }
                blank_between = true;
                blank_start = p;
                p = nx;
                // blank lines before the next item stay with the list
                while (p < s.len) {
                    size_t pn;
                    Str pl = line_at(s, p, &pn);
                    if (!blank(pl)) break;
                    p = pn;
                }
                break;
            }
            if (indent_of(ln) >= content_indent) {
                buf_addc(&body, '\n');
                buf_addstr(&body, str_sub(ln, content_indent, ln.len));
                p = nx;
                continue;
            }
            // a lazy continuation of the item's paragraph
            bool o3;
            long n3;
            char b3;
            if (list_marker(ln, &o3, &n3, &b3) || interrupts(ln)) break;
            buf_addc(&body, '\n');
            buf_addstr(&body, str_trim(ln));
            p = nx;
        }
        ListItem it = {0};
        it.raw = str_sub(s, item_start, p);
        Str text = str_n(arena_strndup(a, body.ptr ? body.ptr : "", body.len), body.len);
        // task item
        if (text.len >= 3 && text.ptr[0] == '[' && (text.ptr[1] == ' ' || text.ptr[1] == 'x' || text.ptr[1] == 'X') && text.ptr[2] == ']' &&
            (text.len == 3 || text.ptr[3] == ' ')) {
            it.task = true;
            it.checked = text.ptr[1] != ' ';
            text = str_sub(text, text.len > 3 ? 4 : 3, text.len);
        }
        BlockVec kids = {0};
        nested_blocks(a, text, &kids);
        // a blank line between two blocks of the item makes the list loose
        if (item_blank) {
            for (size_t k = 0; k + 1 < kids.len; k++)
                if (kids.ptr[k].kind == B_SPACE) loose = true;
        }
        // drop space tokens inside items (marked keeps them; the renderer prints them as blank lines)
        it.blocks = arena_alloc(a, (kids.len + 1) * sizeof(Block));
        memcpy(it.blocks, kids.ptr, kids.len * sizeof(Block));
        it.n = kids.len;
        vec_free(&kids);
        vec_push(&items, it);
        buf_free(&body);
        if (blank_between) {
            size_t pn;
            Str pl = p < s.len ? line_at(s, p, &pn) : S("");
            bool o4;
            long n4;
            char b4;
            if (p >= s.len || !list_marker(pl, &o4, &n4, &b4) || o4 != ordered || b4 != bullet || indent_of(pl) > 3) {
                // the blank lines end the list: they and the last line's newline become a space token
                p = blank_start > 0 && s.ptr[blank_start - 1] == '\n' ? blank_start - 1 : blank_start;
                blank_between = false;
                break;
            }
        }
    }
    // tight items: paragraphs become text
    b.loose = loose;
    if (!loose)
        for (size_t i = 0; i < items.len; i++)
            for (size_t k = 0; k < items.ptr[i].n; k++)
                if (items.ptr[i].blocks[k].kind == B_PARAGRAPH) items.ptr[i].blocks[k].kind = B_TEXT;
    b.items = arena_alloc(a, (items.len + 1) * sizeof(ListItem));
    memcpy(b.items, items.ptr, items.len * sizeof(ListItem));
    b.nitems = items.len;
    vec_free(&items);
    b.raw = str_sub(s, start, p);
    *pos = p;
    (void)mw;
    vec_push(out, b);
}

// marked's block.html start: 1 (script, pre, style, textarea: to its closing tag), 2 (a comment: to
// "-->"), 6 (a block-level tag) and 7 (a line that is only an open or a closing tag): to a blank line.
// 0: not an HTML block.
static int html_block(Str l) {
    size_t i = indent_of(l);
    if (i > 3 || i >= l.len || l.ptr[i] != '<') return 0;
    Str r = str_sub(l, i + 1, l.len);
    if (str_starts(r, S("!--"))) return 2;
    static const char *const raw_tags[] = {"script", "pre", "style", "textarea"};
    for (size_t k = 0; k < sizeof raw_tags / sizeof *raw_tags; k++) {
        size_t n = strlen(raw_tags[k]);
        if (r.len >= n && !strncasecmp(r.ptr, raw_tags[k], n) && (r.len == n || isspace((unsigned char)r.ptr[n]) || r.ptr[n] == '>')) return 1;
    }
    bool closing = r.len && r.ptr[0] == '/';
    size_t j = closing ? 1 : 0;
    size_t ns = j;
    if (j >= r.len || !isalpha((unsigned char)r.ptr[j])) return 0;
    while (j < r.len && (isalnum((unsigned char)r.ptr[j]) || r.ptr[j] == '-' || r.ptr[j] == '_')) j++;
    Str name = str_sub(r, ns, j);
    static const char *const block_tags[] = {
        "address", "article", "aside", "base", "basefont", "blockquote", "body", "caption", "center", "col", "colgroup", "dd",
        "details", "dialog", "dir", "div", "dl", "dt", "fieldset", "figcaption", "figure", "footer", "form", "frame", "frameset",
        "h1", "h2", "h3", "h4", "h5", "h6", "head", "header", "hr", "html", "iframe", "legend", "li", "link", "main", "menu",
        "menuitem", "meta", "nav", "noframes", "ol", "optgroup", "option", "p", "param", "search", "section", "summary", "table",
        "tbody", "td", "tfoot", "th", "thead", "title", "tr", "track", "ul"};
    for (size_t k = 0; k < sizeof block_tags / sizeof *block_tags; k++)
        if (name.len == strlen(block_tags[k]) && !strncasecmp(name.ptr, block_tags[k], name.len)) {
            if (j == r.len || r.ptr[j] == ' ' || r.ptr[j] == '>' || (r.ptr[j] == '/' && j + 1 < r.len && r.ptr[j + 1] == '>')) return 6;
            break;
        }
    // 7: attributes (an open tag), then "/>" or ">", then only blanks
    if (!closing)
        while (j < r.len && (r.ptr[j] == ' ' || r.ptr[j] == '\t')) {
            size_t a0 = j;
            while (j < r.len && (r.ptr[j] == ' ' || r.ptr[j] == '\t')) j++;
            if (j >= r.len || !(isalpha((unsigned char)r.ptr[j]) || r.ptr[j] == '_' || r.ptr[j] == ':')) {
                j = a0;
                break;
            }
            while (j < r.len && (isalnum((unsigned char)r.ptr[j]) || strchr("_.:-", r.ptr[j]))) j++;
            size_t e = j;
            while (e < r.len && r.ptr[e] == ' ') e++;
            if (e < r.len && r.ptr[e] == '=') {
                e++;
                while (e < r.len && r.ptr[e] == ' ') e++;
                if (e < r.len && (r.ptr[e] == '"' || r.ptr[e] == '\'')) {
                    char q = r.ptr[e++];
                    while (e < r.len && r.ptr[e] != q) e++;
                    if (e >= r.len) return 0;
                    e++;
                } else {
                    size_t v0 = e;
                    while (e < r.len && !isspace((unsigned char)r.ptr[e]) && !strchr("\"'=<>`", r.ptr[e])) e++;
                    if (e == v0) return 0;
                }
                j = e;
            }
        }
    while (j < r.len && r.ptr[j] == ' ') j++;
    if (!closing && j < r.len && r.ptr[j] == '/') j++;
    if (j >= r.len || r.ptr[j] != '>') return 0;
    return blank(str_sub(r, j + 1, r.len)) ? 7 : 0;
}

static void lex_blocks(Arena *a, Str s, BlockVec *out, bool top) {
    size_t pos = 0;
    while (pos < s.len) {
        size_t nx;
        Str l = line_at(s, pos, &nx);
        // blank lines
        if (blank(l)) {
            size_t p = pos;
            while (p < s.len) {
                size_t pn;
                Str bl = line_at(s, p, &pn);
                if (!blank(bl)) break;
                if (pn == p) break;
                p = pn;
                if (p >= s.len) break;
            }
            Str raw = str_sub(s, pos, p == pos ? nx : p);
            if (raw.len == 1 && out->len) {
                // a single newline ends the previous block
            } else vec_push(out, mk(B_SPACE, raw));
            pos = p == pos ? nx : p;
            continue;
        }
        char fc;
        size_t fn;
        // fenced code
        if (is_fence(l, &fc, &fn)) {
            size_t ind = indent_of(l);
            Str info = str_trim(str_sub(l, ind + fn, l.len));
            size_t sp = 0;
            while (sp < info.len && !isspace((unsigned char)info.ptr[sp])) sp++;
            Str lang = str_sub(info, 0, sp);
            size_t p = nx;
            Buf body = {0};
            bool closed = false;
            size_t end = s.len;
            bool firstl = true;
            while (p < s.len) {
                size_t pn;
                Str cl = line_at(s, p, &pn);
                size_t ci = indent_of(cl);
                size_t r = 0;
                while (ci + r < cl.len && cl.ptr[ci + r] == fc) r++;
                if (ci <= 3 && r >= fn && blank(str_sub(cl, ci + r, cl.len))) {
                    closed = true;
                    end = p + cl.len;
                    break;
                }
                if (!firstl) buf_addc(&body, '\n');
                firstl = false;
                // the fence's indentation is removed from each line
                size_t rm = MIN(ind, indent_of(cl));
                buf_addstr(&body, str_sub(cl, rm, cl.len));
                p = pn;
            }
            Block b = mk(B_CODE, str_sub(s, pos, closed ? end : s.len));
            b.lang = lang;
            b.text = str_n(arena_strndup(a, body.ptr ? body.ptr : "", body.len), body.len);
            buf_free(&body);
            vec_push(out, b);
            pos = closed ? end : s.len;
            continue;
        }
        // indented code (not inside a paragraph)
        if (indent_of(l) >= 4) {
            size_t p = pos;
            Buf body = {0};
            size_t lastnb = pos;
            Buf tmp = {0};
            while (p < s.len) {
                size_t pn;
                Str cl = line_at(s, p, &pn);
                if (blank(cl)) {
                    buf_addc(&tmp, '\n');
                    p = pn;
                    continue;
                }
                if (indent_of(cl) < 4) break;
                if (tmp.len) {
                    buf_addstr(&body, buf_str(&tmp));
                    tmp.len = 0;
                }
                if (body.len) buf_addc(&body, '\n');
                buf_addstr(&body, str_sub(cl, 4, cl.len));
                p = pn;
                lastnb = p;
            }
            Block b = mk(B_CODE, str_sub(s, pos, lastnb));
            b.text = str_n(arena_strndup(a, body.ptr ? body.ptr : "", body.len), body.len);
            buf_free(&body);
            buf_free(&tmp);
            vec_push(out, b);
            pos = lastnb;
            continue;
        }
        // heading
        int hl = heading_level(l);
        if (hl) {
            size_t i = indent_of(l) + (size_t)hl;
            Str t = str_trim(str_sub(l, i, l.len));
            // closing #s
            size_t e = t.len;
            while (e > 0 && t.ptr[e - 1] == '#') e--;
            if (e == 0) t.len = 0;
            else if (e < t.len && (t.ptr[e - 1] == ' ' || t.ptr[e - 1] == '\t')) t = str_trim(str_sub(t, 0, e));
            size_t p = nx;
            while (p < s.len && s.ptr[p] == '\n') p++;
            Block b = mk(B_HEADING, str_sub(s, pos, p));
            b.depth = hl;
            b.text = t;
            b.in = inline_list(a, t, &b.nin);
            vec_push(out, b);
            pos = p;
            continue;
        }
        if (is_hr(l)) {
            size_t p = nx;
            while (p < s.len && s.ptr[p] == '\n') p++;
            vec_push(out, mk(B_HR, str_sub(s, pos, p)));
            pos = p;
            continue;
        }
        // blockquote
        {
            size_t i = indent_of(l);
            if (i <= 3 && i < l.len && l.ptr[i] == '>') {
                size_t p = pos;
                Buf body = {0};
                while (p < s.len) {
                    size_t pn;
                    Str ql = line_at(s, p, &pn);
                    size_t qi = indent_of(ql);
                    if (qi <= 3 && qi < ql.len && ql.ptr[qi] == '>') {
                        size_t c = qi + 1;
                        if (c < ql.len && ql.ptr[c] == ' ') c++;
                        if (body.len) buf_addc(&body, '\n');
                        buf_addstr(&body, str_sub(ql, c, ql.len));
                        p = pn;
                    } else if (!blank(ql) && body.len && !interrupts(ql)) {
                        buf_addc(&body, '\n');
                        buf_addstr(&body, ql); // lazy continuation
                        p = pn;
                    } else break;
                }
                Block b = mk(B_QUOTE, str_sub(s, pos, p));
                Str text = str_n(arena_strndup(a, body.ptr ? body.ptr : "", body.len), body.len);
                BlockVec kids = {0};
                nested_blocks(a, text, &kids);
                b.kids = arena_alloc(a, (kids.len + 1) * sizeof(Block));
                memcpy(b.kids, kids.ptr, kids.len * sizeof(Block));
                b.nkids = kids.len;
                vec_free(&kids);
                buf_free(&body);
                vec_push(out, b);
                pos = p;
                continue;
            }
        }
        // list
        {
            bool o;
            long num;
            char bl;
            if (indent_of(l) <= 3 && list_marker(l, &o, &num, &bl)) {
                lex_list(a, s, &pos, out);
                continue;
            }
        }
        // an HTML block, drawn as its text (the blank lines after it are a space)
        int hb = html_block(l);
        if (hb) {
            size_t p = pos, end = pos;
            while (p < s.len) {
                size_t pn;
                Str hl2 = line_at(s, p, &pn);
                if ((hb == 6 || hb == 7) && blank(hl2)) break;
                end = p + hl2.len;
                p = pn;
                if (hb == 2 && str_find(hl2, S("-->"), 0) >= 0) break;
                if (hb == 1 && (strcasestr(str_z(a, hl2), "</script>") || strcasestr(str_z(a, hl2), "</pre>") ||
                                strcasestr(str_z(a, hl2), "</style>") || strcasestr(str_z(a, hl2), "</textarea>")))
                    break;
            }
            vec_push(out, mk(B_HTML, str_sub(s, pos, end)));
            pos = end;
            continue;
        }
        // table: a header row and a delimiter row
        if (nx < s.len) {
            size_t nn;
            Str l2 = line_at(s, nx, &nn);
            if ((memchr(l.ptr, '|', l.len) || memchr(l2.ptr, '|', l2.len)) && table_delim(l2) && memchr(l2.ptr, '-', l2.len)) {
                VecStr head = {0}, delim = {0};
                split_cells(a, l, &head);
                split_cells(a, l2, &delim);
                if (head.len == delim.len && head.len > 0) {
                    VEC(VecStr) rows = {0};
                    size_t p = nn;
                    while (p < s.len) {
                        size_t pn;
                        Str rl = line_at(s, p, &pn);
                        if (blank(rl) || interrupts(rl)) break;
                        VecStr cells = {0};
                        split_cells(a, rl, &cells);
                        vec_push(&rows, cells);
                        p = pn;
                    }
                    while (p < s.len && s.ptr[p] == '\n') p++;
                    Block b = mk(B_TABLE, str_sub(s, pos, p));
                    b.cols = head.len;
                    b.rows = rows.len;
                    b.cells = arena_alloc(a, (rows.len + 1) * head.len * sizeof(Inline *));
                    b.ncells = arena_alloc(a, (rows.len + 1) * head.len * sizeof(size_t));
                    for (size_t c = 0; c < head.len; c++) b.cells[c] = inline_list(a, head.ptr[c], &b.ncells[c]);
                    for (size_t r = 0; r < rows.len; r++)
                        for (size_t c = 0; c < head.len; c++) {
                            size_t k = (r + 1) * head.len + c;
                            Str cell = c < rows.ptr[r].len ? rows.ptr[r].ptr[c] : S("");
                            b.cells[k] = inline_list(a, cell, &b.ncells[k]);
                        }
                    for (size_t r = 0; r < rows.len; r++) vec_free(&rows.ptr[r]);
                    vec_free(&rows);
                    vec_free(&head);
                    vec_free(&delim);
                    vec_push(out, b);
                    pos = p;
                    continue;
                }
                vec_free(&head);
                vec_free(&delim);
            }
        }
        // paragraph (or a setext heading)
        {
            size_t p = nx;
            size_t end = pos + l.len;
            int setext = 0;
            while (p < s.len) {
                size_t pn;
                Str pl = line_at(s, p, &pn);
                if (blank(pl)) break;
                Str t = str_trim(pl);
                bool eq = t.len > 0, dash = t.len > 0;
                for (size_t k = 0; k < t.len; k++) {
                    eq &= t.ptr[k] == '=';
                    dash &= t.ptr[k] == '-';
                }
                if (indent_of(pl) <= 3 && (eq || dash)) {
                    setext = eq ? 1 : 2;
                    end = p + pl.len;
                    p = pn;
                    break;
                }
                if (interrupts(pl)) break;
                // a table interrupts a paragraph (GFM)
                if (pn < s.len) {
                    size_t pn2;
                    Str d = line_at(s, pn, &pn2);
                    if ((memchr(pl.ptr, '|', pl.len) || memchr(d.ptr, '|', d.len)) && table_delim(d) && memchr(d.ptr, '-', d.len)) break;
                }
                end = p + pl.len;
                p = pn;
            }
            if (setext) {
                Str t = str_sub(s, pos, end);
                size_t lastnl = t.len;
                while (lastnl > 0 && t.ptr[lastnl - 1] != '\n') lastnl--;
                Str text = str_trim(str_sub(t, 0, lastnl ? lastnl - 1 : 0));
                while (p < s.len && s.ptr[p] == '\n') p++;
                Block b = mk(B_HEADING, str_sub(s, pos, p));
                b.depth = setext;
                b.text = text;
                b.in = inline_list(a, text, &b.nin);
                vec_push(out, b);
                pos = p;
                continue;
            }
            Str raw = str_sub(s, pos, end);
            Block b = mk(B_PARAGRAPH, raw);
            // leading whitespace of each line is not part of the text
            Buf t = {0};
            for (size_t i = 0; i < raw.len;) {
                size_t j = i;
                while (j < raw.len && raw.ptr[j] != '\n') j++;
                Str ln = str_sub(raw, i, j);
                size_t k = 0;
                while (k < ln.len && (ln.ptr[k] == ' ' || ln.ptr[k] == '\t')) k++;
                if (t.len) buf_addc(&t, '\n');
                buf_addstr(&t, str_sub(ln, k, ln.len));
                i = j + 1;
            }
            Str text = str_n(arena_strndup(a, t.ptr ? t.ptr : "", t.len), t.len);
            buf_free(&t);
            // trailing spaces of the last line go
            while (text.len && (text.ptr[text.len - 1] == ' ' || text.ptr[text.len - 1] == '\t')) text.len--;
            b.text = text;
            b.in = inline_list(a, text, &b.nin);
            vec_push(out, b);
            pos = end;
            continue;
        }
    }
    (void)top;
}

// ---- rendering --------------------------------------------------------------------------------------

typedef enum { ST_DEFAULT, ST_HEADING1, ST_HEADING, ST_PLAIN } StyleKind;
typedef struct {
    StyleKind kind;
    const MdStyle *def;
    char prefix[64];
} Ctx;

typedef struct {
    const MdStyle *style;
    bool preserve_markers, preserve_escapes;
    bool hyperlinks;
    const char *indent;
} Md;

static void apply_style(const Ctx *c, Buf *b, Str text) {
    switch (c->kind) {
    case ST_PLAIN: buf_addstr(b, text); return;
    case ST_HEADING1:
    case ST_HEADING: {
        Buf in = {0};
        if (c->kind == ST_HEADING1) {
            Buf u = {0};
            c_underline(&u, text);
            c_bold(&in, buf_str(&u));
            buf_free(&u);
        } else c_bold(&in, text);
        th_fg(b, T_MD_HEADING, buf_str(&in));
        buf_free(&in);
        return;
    }
    case ST_DEFAULT:
        if (!c->def) {
            buf_addstr(b, text);
            return;
        }
        Buf t = {0};
        if (c->def->has_color) th_fg(&t, c->def->color, text);
        else buf_addstr(&t, text);
        if (c->def->italic) {
            Buf i = {0};
            c_italic(&i, buf_str(&t));
            buf_free(&t);
            t = i;
        }
        buf_addstr(b, buf_str(&t));
        buf_free(&t);
        return;
    }
}

static void ctx_init(Ctx *c, StyleKind k, const MdStyle *def) {
    c->kind = k;
    c->def = def;
    c->prefix[0] = 0;
    if (k == ST_DEFAULT && !def) return;
    if (k == ST_PLAIN) return;
    Buf b = {0};
    apply_style(c, &b, S("\x01"));
    char *p = b.ptr ? memchr(b.ptr, 1, b.len) : NULL;
    if (p && (size_t)(p - b.ptr) < sizeof c->prefix) {
        memcpy(c->prefix, b.ptr, (size_t)(p - b.ptr));
        c->prefix[p - b.ptr] = 0;
    }
    buf_free(&b);
}

static void apply_lines(const Ctx *c, Buf *b, Str text) {
    for (size_t i = 0;;) {
        size_t j = i;
        while (j < text.len && text.ptr[j] != '\n') j++;
        apply_style(c, b, str_sub(text, i, j));
        if (j >= text.len) break;
        buf_addc(b, '\n');
        i = j + 1;
    }
}

static void render_inlines(const Md *m, const Inline *t, size_t n, const Ctx *c, Buf *out) {
    Buf r = {0};
    size_t pl = strlen(c->prefix);
    for (size_t i = 0; i < n; i++) {
        const Inline *x = &t[i];
        switch (x->kind) {
        case IN_ESCAPE: apply_lines(c, &r, m->preserve_escapes ? x->raw : x->text); break;
        case IN_TEXT: apply_lines(c, &r, x->text); break;
        case IN_STRONG: {
            Buf in = {0};
            render_inlines(m, x->kids, x->nkids, c, &in);
            c_bold(&r, buf_str(&in));
            buf_adds(&r, c->prefix);
            buf_free(&in);
            break;
        }
        case IN_EM: {
            Buf in = {0};
            render_inlines(m, x->kids, x->nkids, c, &in);
            c_italic(&r, buf_str(&in));
            buf_adds(&r, c->prefix);
            buf_free(&in);
            break;
        }
        case IN_DEL: {
            Buf in = {0};
            render_inlines(m, x->kids, x->nkids, c, &in);
            c_strike(&r, buf_str(&in));
            buf_adds(&r, c->prefix);
            buf_free(&in);
            break;
        }
        case IN_CODE:
            th_fg(&r, T_MD_CODE, x->text);
            buf_adds(&r, c->prefix);
            break;
        case IN_LINK: {
            Buf in = {0}, u = {0};
            render_inlines(m, x->kids, x->nkids, c, &in);
            c_underline(&u, buf_str(&in));
            Buf styled = {0};
            th_fg(&styled, T_MD_LINK, buf_str(&u));
            // a destination or a label with control characters never goes into an OSC 8 sequence
            Buf hb = {0};
            no_ctl(&hb, x->href);
            Str href = buf_str(&hb);
            if (m->hyperlinks && !has_ctl(x->href) && !has_ctl(x->text)) {
                buf_printf(&r, "\x1b]8;;%.*s\x07", (int)href.len, href.ptr);
                buf_addstr(&r, buf_str(&styled));
                buf_adds(&r, "\x1b]8;;\x07");
            } else {
                Str hc = str_starts(href, S("mailto:")) ? str_sub(href, 7, href.len) : href;
                buf_addstr(&r, buf_str(&styled));
                if (!str_eq(x->text, href) && !str_eq(x->text, hc)) {
                    Buf url = {0};
                    buf_printf(&url, " (%.*s)", (int)href.len, href.ptr);
                    th_fg(&r, T_MD_LINK_URL, buf_str(&url));
                    buf_free(&url);
                }
            }
            buf_adds(&r, c->prefix);
            buf_free(&hb);
            buf_free(&in);
            buf_free(&u);
            buf_free(&styled);
            break;
        }
        case IN_IMAGE: apply_lines(c, &r, x->text); break;
        case IN_BR: buf_addc(&r, '\n'); break;
        case IN_HTML:
            if (!str_starts(x->raw, S("<!--"))) apply_lines(c, &r, x->raw);
            break;
        }
    }
    // trailing style prefixes go
    while (pl && r.len >= pl && !memcmp(r.ptr + r.len - pl, c->prefix, pl)) r.len -= pl;
    if (r.ptr) r.ptr[r.len] = 0;
    buf_addstr(out, buf_str(&r));
    buf_free(&r);
}

static void render_block(const Md *m, const Block *t, int width, int next, const Ctx *c, Lines *out);

static void render_list(const Md *m, const Block *t, int depth, int width, const Ctx *c, Lines *out) {
    for (size_t i = 0; i < t->nitems; i++) {
        const ListItem *it = &t->items[i];
        bool last = i + 1 == t->nitems;
        char bullet[32];
        if (t->ordered) {
            snprintf(bullet, sizeof bullet, "%ld. ", t->start + (long)i);
            if (m->preserve_markers) {
                // the source marker ("3)" stays "3)")
                Str raw = it->raw;
                size_t k = 0;
                while (k < raw.len && raw.ptr[k] == ' ' && k < 3) k++;
                size_t d = k;
                while (d < raw.len && isdigit((unsigned char)raw.ptr[d]) && d - k < 9) d++;
                if (d > k && d < raw.len && (raw.ptr[d] == '.' || raw.ptr[d] == ')') && d + 1 < raw.len &&
                    (raw.ptr[d + 1] == ' ' || raw.ptr[d + 1] == '\t'))
                    snprintf(bullet, sizeof bullet, "%.*s ", (int)(d + 1 - k), raw.ptr + k);
            }
        } else {
            snprintf(bullet, sizeof bullet, "- ");
            if (m->preserve_markers) {
                Str raw = it->raw;
                size_t k = 0;
                while (k < raw.len && raw.ptr[k] == ' ' && k < 3) k++;
                if (k < raw.len && strchr("-+*", raw.ptr[k])) snprintf(bullet, sizeof bullet, "%c ", raw.ptr[k]);
            }
        }
        Buf marker = {0};
        buf_adds(&marker, bullet);
        if (it->task) buf_adds(&marker, it->checked ? "[x] " : "[ ] ");
        Buf first = {0}, cont = {0};
        tui_spaces(&first, depth * 4);
        th_fg(&first, T_MD_LIST_BULLET, buf_str(&marker));
        tui_spaces(&cont, depth * 4 + tui_width(buf_str(&marker)));
        int iw = MAX(1, width - tui_width(buf_str(&first)));
        bool any = false;
        for (size_t k = 0; k < it->n; k++) {
            const Block *b = &it->blocks[k];
            if (b->kind == B_LIST) {
                render_list(m, b, depth + 1, width, c, out);
                any = true;
                continue;
            }
            Lines il = {0};
            render_block(m, b, iw, -1, c, &il);
            for (size_t q = 0; q < il.n; q++) {
                Lines w = {0};
                tui_wrap(str(il.v[q]), iw, &w);
                for (size_t z = 0; z < w.n; z++) {
                    Buf l = {0};
                    buf_addstr(&l, any ? buf_str(&cont) : buf_str(&first));
                    buf_adds(&l, w.v[z]);
                    lines_addb(out, &l);
                    any = true;
                }
                lines_free(&w);
            }
            lines_free(&il);
        }
        if (!any) lines_add(out, buf_str(&first));
        if (t->loose && !last) lines_addz(out, "");
        buf_free(&marker);
        buf_free(&first);
        buf_free(&cont);
    }
}

static int longest_word(Str text, int cap) {
    int longest = 0;
    for (size_t i = 0; i < text.len;) {
        while (i < text.len && isspace((unsigned char)text.ptr[i])) i++;
        size_t j = i;
        while (j < text.len && !isspace((unsigned char)text.ptr[j])) j++;
        if (j > i) longest = MAX(longest, tui_width(str_sub(text, i, j)));
        i = j;
    }
    return cap > 0 ? MIN(longest, cap) : longest;
}

static void wrap_cell(Str text, int w, const char *prefix, Lines *out) {
    Lines l = {0};
    tui_wrap(text, MAX(1, w), &l);
    for (size_t i = 0; i < l.n; i++) {
        Buf b = {0};
        buf_adds(&b, l.v[i]);
        if (i + 1 < l.n) buf_adds(&b, "\x1b[22;23;24;25;27;28;29;39m");
        buf_adds(&b, prefix);
        lines_addb(out, &b);
    }
    lines_free(&l);
}

static void render_table(const Md *m, const Block *t, int avail, int next, const Ctx *c, Lines *out) {
    size_t nc = t->cols;
    if (!nc) return;
    int overhead = 3 * (int)nc + 1;
    int for_cells = avail - overhead;
    size_t nr = t->rows + 1;
    char **text = xcalloc(nr * nc, sizeof *text);
    for (size_t k = 0; k < nr * nc; k++) {
        Buf b = {0};
        render_inlines(m, t->cells[k], t->ncells[k], c, &b);
        text[k] = b.ptr ? b.ptr : strdup("");
    }
    if (for_cells < (int)nc) {
        Lines fb = {0};
        tui_wrap(t->raw, avail, &fb);
        lines_append(out, &fb);
        if (next >= 0 && next != B_SPACE) lines_addz(out, "");
        for (size_t k = 0; k < nr * nc; k++) free(text[k]);
        free(text);
        return;
    }
    int *nat = xcalloc(nc, sizeof(int)), *minw = xcalloc(nc, sizeof(int)), *cw = xcalloc(nc, sizeof(int));
    for (size_t col = 0; col < nc; col++) {
        nat[col] = tui_width(str(text[col]));
        minw[col] = MAX(1, longest_word(str(text[col]), 30));
    }
    for (size_t r = 1; r < nr; r++)
        for (size_t col = 0; col < nc; col++) {
            nat[col] = MAX(nat[col], tui_width(str(text[r * nc + col])));
            minw[col] = MAX(minw[col], longest_word(str(text[r * nc + col]), 30));
        }
    int *mincol = xcalloc(nc, sizeof(int));
    int mincells = 0;
    for (size_t col = 0; col < nc; col++) {
        mincol[col] = minw[col];
        mincells += mincol[col];
    }
    if (mincells > for_cells) {
        int remaining = for_cells - (int)nc;
        for (size_t col = 0; col < nc; col++) mincol[col] = 1;
        if (remaining > 0) {
            int tw = 0;
            for (size_t col = 0; col < nc; col++) tw += MAX(0, minw[col] - 1);
            int alloc = 0;
            for (size_t col = 0; col < nc; col++) {
                int g = tw > 0 ? (int)((double)MAX(0, minw[col] - 1) / tw * remaining) : 0;
                mincol[col] += g;
                alloc += g;
            }
            int left = remaining - alloc;
            for (size_t col = 0; left > 0 && col < nc; col++, left--) mincol[col]++;
        }
        mincells = 0;
        for (size_t col = 0; col < nc; col++) mincells += mincol[col];
    }
    int total_nat = overhead;
    for (size_t col = 0; col < nc; col++) total_nat += nat[col];
    if (total_nat <= avail) {
        for (size_t col = 0; col < nc; col++) cw[col] = MAX(nat[col], mincol[col]);
    } else {
        int grow_pot = 0;
        for (size_t col = 0; col < nc; col++) grow_pot += MAX(0, nat[col] - mincol[col]);
        int extra = MAX(0, for_cells - mincells);
        int alloc = 0;
        for (size_t col = 0; col < nc; col++) {
            int g = grow_pot > 0 ? (int)((double)MAX(0, nat[col] - mincol[col]) / grow_pot * extra) : 0;
            cw[col] = mincol[col] + g;
            alloc += cw[col];
        }
        int rem = for_cells - alloc;
        while (rem > 0) {
            bool grew = false;
            for (size_t col = 0; col < nc && rem > 0; col++)
                if (cw[col] < nat[col]) {
                    cw[col]++;
                    rem--;
                    grew = true;
                }
            if (!grew) break;
        }
    }
    const char *H = "\xe2\x94\x80";
    Buf b = {0};
    buf_adds(&b, "\xe2\x94\x8c");
    buf_adds(&b, H);
    for (size_t col = 0; col < nc; col++) {
        if (col) {
            buf_adds(&b, H);
            buf_adds(&b, "\xe2\x94\xac");
            buf_adds(&b, H);
        }
        tui_repeat(&b, H, cw[col]);
    }
    buf_adds(&b, H);
    buf_adds(&b, "\xe2\x94\x90");
    lines_addb(out, &b);
    Buf sep = {0};
    buf_adds(&sep, "\xe2\x94\x9c");
    buf_adds(&sep, H);
    for (size_t col = 0; col < nc; col++) {
        if (col) {
            buf_adds(&sep, H);
            buf_adds(&sep, "\xe2\x94\xbc");
            buf_adds(&sep, H);
        }
        tui_repeat(&sep, H, cw[col]);
    }
    buf_adds(&sep, H);
    buf_adds(&sep, "\xe2\x94\xa4");
    for (size_t r = 0; r < nr; r++) {
        Lines *cl = xcalloc(nc, sizeof(Lines));
        size_t rows = 0;
        for (size_t col = 0; col < nc; col++) {
            wrap_cell(str(text[r * nc + col]), cw[col], c->prefix, &cl[col]);
            rows = MAX(rows, cl[col].n);
        }
        for (size_t li = 0; li < rows; li++) {
            Buf l = {0};
            buf_adds(&l, "\xe2\x94\x82 ");
            for (size_t col = 0; col < nc; col++) {
                if (col) buf_adds(&l, " \xe2\x94\x82 ");
                const char *tx = li < cl[col].n ? cl[col].v[li] : "";
                Buf pad = {0};
                buf_adds(&pad, tx);
                tui_spaces(&pad, cw[col] - tui_width(str(tx)));
                if (r == 0) c_bold(&l, buf_str(&pad));
                else buf_addstr(&l, buf_str(&pad));
                buf_free(&pad);
            }
            buf_adds(&l, " \xe2\x94\x82");
            lines_addb(out, &l);
        }
        for (size_t col = 0; col < nc; col++) lines_free(&cl[col]);
        free(cl);
        if (r == 0 || r + 1 < nr) lines_add(out, buf_str(&sep));
    }
    buf_adds(&b, "\xe2\x94\x94");
    buf_adds(&b, H);
    for (size_t col = 0; col < nc; col++) {
        if (col) {
            buf_adds(&b, H);
            buf_adds(&b, "\xe2\x94\xb4");
            buf_adds(&b, H);
        }
        tui_repeat(&b, H, cw[col]);
    }
    buf_adds(&b, H);
    buf_adds(&b, "\xe2\x94\x98");
    lines_addb(out, &b);
    if (next >= 0 && next != B_SPACE) lines_addz(out, "");
    buf_free(&sep);
    for (size_t k = 0; k < nr * nc; k++) free(text[k]);
    free(text);
    free(nat);
    free(minw);
    free(cw);
    free(mincol);
}

static void render_block(const Md *m, const Block *t, int width, int next, const Ctx *c, Lines *out) {
    switch (t->kind) {
    case B_HEADING: {
        Ctx hc;
        ctx_init(&hc, t->depth == 1 ? ST_HEADING1 : ST_HEADING, NULL);
        Buf b = {0};
        if (t->depth >= 3) {
            Buf pre = {0};
            for (int k = 0; k < t->depth; k++) buf_addc(&pre, '#');
            buf_addc(&pre, ' ');
            apply_style(&hc, &b, buf_str(&pre));
            buf_free(&pre);
        }
        render_inlines(m, t->in, t->nin, &hc, &b);
        lines_addb(out, &b);
        if (next >= 0 && next != B_SPACE) lines_addz(out, "");
        break;
    }
    case B_PARAGRAPH: {
        Buf b = {0};
        render_inlines(m, t->in, t->nin, c, &b);
        lines_addb(out, &b);
        if (next >= 0 && next != B_LIST && next != B_SPACE) lines_addz(out, "");
        break;
    }
    case B_TEXT: {
        Buf b = {0};
        render_inlines(m, t->in, t->nin, c, &b);
        lines_addb(out, &b);
        break;
    }
    case B_CODE: {
        Buf b = {0};
        Buf border = {0};
        buf_adds(&border, "```");
        buf_addstr(&border, t->lang);
        th_fg(&b, T_MD_CODE_BLOCK_BORDER, buf_str(&border));
        lines_addb(out, &b);
        Lines hl = {0};
        md_highlight(t->text, t->lang, &hl);
        for (size_t i = 0; i < hl.n; i++) {
            Buf l = {0};
            buf_adds(&l, m->indent);
            buf_adds(&l, hl.v[i]);
            lines_addb(out, &l);
        }
        lines_free(&hl);
        th_fgz(&b, T_MD_CODE_BLOCK_BORDER, "```");
        lines_addb(out, &b);
        if (next >= 0 && next != B_SPACE) lines_addz(out, "");
        buf_free(&border);
        break;
    }
    case B_LIST: render_list(m, t, 0, width, c, out); break;
    case B_TABLE: render_table(m, t, width, next, c, out); break;
    case B_QUOTE: {
        Ctx qc;
        qc.kind = ST_PLAIN;
        qc.def = NULL;
        // the quote style's prefix: muted then italic
        Buf pre = {0};
        Buf s1 = {0};
        c_italic(&s1, S("\x01"));
        th_fg(&pre, T_MD_QUOTE, buf_str(&s1));
        char *p = memchr(pre.ptr, 1, pre.len);
        size_t pl = p ? (size_t)(p - pre.ptr) : 0;
        memcpy(qc.prefix, pre.ptr, MIN(pl, sizeof qc.prefix - 1));
        qc.prefix[MIN(pl, sizeof qc.prefix - 1)] = 0;
        buf_free(&pre);
        buf_free(&s1);
        if (width < 3) {
            for (size_t i = 0; i < t->nkids; i++) render_block(m, &t->kids[i], width, i + 1 < t->nkids ? (int)t->kids[i + 1].kind : -1, c, out);
            break;
        }
        int qw = width - 2;
        Lines ql = {0};
        for (size_t i = 0; i < t->nkids; i++) render_block(m, &t->kids[i], qw, i + 1 < t->nkids ? (int)t->kids[i + 1].kind : -1, &qc, &ql);
        while (ql.n && !ql.v[ql.n - 1][0]) free(ql.v[--ql.n]);
        for (size_t i = 0; i < ql.n; i++) {
            // the quote style re-applied after every reset
            Buf re = {0};
            Str l = str(ql.v[i]);
            for (size_t k = 0; k < l.len;) {
                if (l.len - k >= 4 && !memcmp(l.ptr + k, "\x1b[0m", 4)) {
                    buf_adds(&re, "\x1b[0m");
                    buf_adds(&re, qc.prefix);
                    k += 4;
                } else buf_addc(&re, l.ptr[k++]);
            }
            Buf it = {0}, styled = {0};
            c_italic(&it, buf_str(&re));
            th_fg(&styled, T_MD_QUOTE, buf_str(&it));
            Lines w = {0};
            tui_wrap(buf_str(&styled), qw, &w);
            for (size_t z = 0; z < w.n; z++) {
                Buf o = {0};
                th_fgz(&o, T_MD_QUOTE_BORDER, "\xe2\x94\x82 ");
                buf_adds(&o, w.v[z]);
                lines_addb(out, &o);
            }
            lines_free(&w);
            buf_free(&re);
            buf_free(&it);
            buf_free(&styled);
        }
        lines_free(&ql);
        if (next >= 0 && next != B_SPACE) lines_addz(out, "");
        break;
    }
    case B_HR: {
        Buf b = {0}, r = {0};
        tui_repeat(&r, "\xe2\x94\x80", MIN(width, 80));
        th_fg(&b, T_MD_HR, buf_str(&r));
        lines_addb(out, &b);
        buf_free(&r);
        if (next >= 0 && next != B_SPACE) lines_addz(out, "");
        break;
    }
    case B_HTML: {
        Buf b = {0};
        apply_style(c, &b, str_trim(t->raw));
        lines_addb(out, &b);
        break;
    }
    case B_SPACE: lines_addz(out, ""); break;
    }
}

static bool hyperlinks_supported(void) {
    const char *o = getenv("PI_HYPERLINKS");
    if (o && !strcmp(o, "1")) return true;
    if (o && !strcmp(o, "0")) return false;
    const char *tp = getenv("TERM_PROGRAM"), *term = getenv("TERM");
    if (getenv("TMUX") || (term && !strncmp(term, "tmux", 4)) || (term && !strncmp(term, "screen", 6))) return false;
    if (getenv("KITTY_WINDOW_ID") || getenv("GHOSTTY_RESOURCES_DIR") || getenv("WEZTERM_PANE") || getenv("ITERM_SESSION_ID") ||
        getenv("WARP_SESSION_ID") || getenv("ALACRITTY_WINDOW_ID") || getenv("WT_SESSION"))
        return true;
    if (tp && (!strcasecmp(tp, "kitty") || !strcasecmp(tp, "ghostty") || !strcasecmp(tp, "wezterm") || !strcasecmp(tp, "iTerm.app") ||
               !strcasecmp(tp, "WarpTerminal") || !strcasecmp(tp, "vscode") || !strcasecmp(tp, "zed") || !strcasecmp(tp, "alacritty")))
        return true;
    return false;
}

bool tui_hyperlinks(void) { return hyperlinks_supported(); }

void md_render(Str text, int pad_x, int pad_y, const MdStyle *style, int width, Lines *out) {
    if (!str_trim(text).len) return;
    int cw = MAX(1, width - pad_x * 2);
    Buf norm = {0};
    for (size_t i = 0; i < text.len; i++)
        if (text.ptr[i] == '\t') buf_adds(&norm, "   ");
        else buf_addc(&norm, text.ptr[i]);
    Arena a = {0};
    BlockVec blocks = {0};
    lex_blocks(&a, buf_str(&norm), &blocks, true);
    static int hl = -1;
    if (hl < 0) hl = hyperlinks_supported();
    Md m = {.style = style, .preserve_markers = style && style->user, .preserve_escapes = style && style->user, .hyperlinks = hl,
            .indent = "  "};
    Ctx c;
    ctx_init(&c, ST_DEFAULT, style && (style->has_color || style->italic) ? style : NULL);
    size_t first = out->n;
    for (int k = 0; k < pad_y; k++) {
        Buf e = {0};
        tui_spaces(&e, width);
        lines_addb(out, &e);
    }
    for (size_t i = 0; i < blocks.len; i++) {
        Lines bl = {0};
        render_block(&m, &blocks.ptr[i], cw, i + 1 < blocks.len ? (int)blocks.ptr[i + 1].kind : -1, &c, &bl);
        for (size_t k = 0; k < bl.n; k++) {
            Lines w = {0};
            tui_wrap(str(bl.v[k]), cw, &w);
            for (size_t z = 0; z < w.n; z++) {
                Buf l = {0};
                tui_spaces(&l, pad_x);
                buf_adds(&l, w.v[z]);
                tui_spaces(&l, pad_x);
                tui_spaces(&l, width - tui_width(buf_str(&l)));
                lines_addb(out, &l);
            }
            lines_free(&w);
        }
        lines_free(&bl);
    }
    for (int k = 0; k < pad_y; k++) {
        Buf e = {0};
        tui_spaces(&e, width);
        lines_addb(out, &e);
    }
    // the panel's background (bgColor) on every line, padding rows included
    if (style && style->has_bg)
        for (size_t k = first; k < out->n; k++) {
            Buf l = {0};
            th_bg_line(&l, style->bg, str(out->v[k]), width);
            free(out->v[k]);
            out->v[k] = l.ptr ? l.ptr : strdup("");
        }
    vec_free(&blocks);
    arena_reset(&a);
    buf_free(&norm);
}

typedef struct {
    Comp base;
    Buf text;
    int pad_x, pad_y;
    MdStyle style;
} MdComp;

static void mdc_render(Comp *c, int width, Lines *out) {
    MdComp *m = (MdComp *)c;
    md_render(buf_str(&m->text), m->pad_x, m->pad_y, &m->style, width, out);
}

static void mdc_destroy(Comp *c) {
    buf_free(&((MdComp *)c)->text);
    free(c);
}

static const CompVt md_vt = {mdc_render, mdc_destroy, false};

Comp *comp_markdown(Str text, int pad_x, int pad_y, const MdStyle *style) {
    MdComp *m = xcalloc(1, sizeof *m);
    m->base.vt = &md_vt;
    buf_addstr(&m->text, text);
    m->pad_x = pad_x;
    m->pad_y = pad_y;
    if (style) m->style = *style;
    return &m->base;
}

// ---- syntax highlighting ----------------------------------------------------------------------------

typedef enum { L_NONE, L_PY, L_JS, L_C, L_GO, L_RUST, L_SH, L_JSON, L_RUBY, L_JAVA, L_LUA, L_PHP, L_OTHER } LangFam;

static LangFam lang_family(Str lang) {
    static const struct {
        const char *name;
        LangFam f;
    } map[] = {
        {"python", L_PY}, {"py", L_PY}, {"gyp", L_PY}, {"javascript", L_JS}, {"js", L_JS}, {"jsx", L_JS}, {"mjs", L_JS}, {"cjs", L_JS},
        {"typescript", L_JS}, {"ts", L_JS}, {"tsx", L_JS}, {"c", L_C}, {"h", L_C}, {"cpp", L_C}, {"c++", L_C}, {"cc", L_C},
        {"hpp", L_C}, {"cxx", L_C}, {"csharp", L_JAVA}, {"cs", L_JAVA}, {"c#", L_JAVA}, {"go", L_GO}, {"golang", L_GO},
        {"rust", L_RUST}, {"rs", L_RUST}, {"bash", L_SH}, {"sh", L_SH}, {"zsh", L_SH}, {"shell", L_SH}, {"json", L_JSON},
        {"ruby", L_RUBY}, {"rb", L_RUBY}, {"java", L_JAVA}, {"jsp", L_JAVA}, {"kotlin", L_JAVA}, {"kt", L_JAVA}, {"swift", L_JAVA},
        {"scala", L_JAVA}, {"dart", L_JAVA}, {"groovy", L_JAVA}, {"lua", L_LUA}, {"php", L_PHP}, {"perl", L_SH}, {"pl", L_SH},
        {"nix", L_SH},
    };
    for (size_t i = 0; i < COUNT(map); i++)
        if (lang.len == strlen(map[i].name) && !strncasecmp(lang.ptr, map[i].name, lang.len)) return map[i].f;
    return L_NONE;
}

// Languages highlight.js knows (with every grammar loaded, as pi has once it started) that this
// highlighter does not color: their code shows in the default color, as highlight.js's scopes for
// them (sections, tags, selectors) have no theme color.
static bool lang_plain(Str lang) {
    static const char *const names[] = {"markdown", "md", "mkdown", "mkd", "yaml", "yml", "toml", "ini", "sql", "html", "xml",
                                        "xhtml", "svg", "rss", "atom", "plist", "css", "scss", "less", "dockerfile", "docker",
                                        "makefile", "mk", "mak", "make", "cmake", "powershell", "ps", "ps1", "r", "clojure", "clj",
                                        "elixir", "ex", "exs", "erlang", "erl", "haskell", "hs", "ocaml", "ml", "vim", "protobuf",
                                        "diff", "patch", "plaintext", "text", "txt", "objectivec", "objc", "mm", "x86asm",
                                        "fortran", "julia", "matlab", "nginx", "apache", "properties", "latex", "tex", "zig",
                                        "elm", "fsharp", "fs", "vbnet", "vb", "pgsql", "postgres", "graphql", "csv", NULL};
    for (size_t i = 0; names[i]; i++)
        if (lang.len == strlen(names[i]) && !strncasecmp(lang.ptr, names[i], lang.len)) return true;
    return false;
}

bool md_lang_supported(Str lang) { return lang_family(lang) != L_NONE || lang_plain(lang); }

static bool in_list(const char *const *list, Str w) {
    for (size_t i = 0; list[i]; i++)
        if (str_eqz(w, list[i])) return true;
    return false;
}

static const char *const py_kw[] = {"and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif", "else",
                                    "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal",
                                    "not", "or", "pass", "raise", "return", "try", "while", "with", "yield", "match", "case", NULL};
static const char *const py_lit[] = {"True", "False", "None", "__debug__", "Ellipsis", "NotImplemented", NULL};
static const char *const py_bi[] = {"print", "len", "range", "int", "str", "float", "list", "dict", "set", "tuple", "open", "isinstance",
                                    "super", "self", "enumerate", "zip", "map", "filter", "sorted", "sum", "min", "max", "abs", "any",
                                    "all", "type", "object", "bool", "bytes", "repr", "iter", "next", "getattr", "setattr", "hasattr",
                                    "input", "round", "Exception", "ValueError", "TypeError", "KeyError", NULL};
static const char *const js_kw[] = {"as", "in", "of", "if", "for", "while", "finally", "var", "new", "function", "do", "return", "void",
                                    "else", "break", "catch", "instanceof", "with", "throw", "case", "default", "try", "switch",
                                    "continue", "typeof", "delete", "let", "yield", "const", "class", "debugger", "async", "await",
                                    "static", "import", "from", "export", "extends", "interface", "type", "enum", "implements",
                                    "public", "private", "protected", "readonly", "declare", "namespace", "abstract", "keyof", NULL};
static const char *const js_lit[] = {"true", "false", "null", "undefined", "NaN", "Infinity", NULL};
static const char *const js_bi[] = {"console", "window", "document", "Object", "Array", "String", "Number", "Boolean", "Promise",
                                    "Math", "JSON", "Map", "Set", "Error", "require", "module", "process", "this", "super", "string",
                                    "number", "boolean", "any", "void", "never", "unknown", NULL};
static const char *const c_kw[] = {"int", "float", "while", "private", "char", "catch", "import", "module", "export", "virtual",
                                   "operator", "sizeof", "typedef", "const", "struct", "for", "union", "unsigned", "long", "volatile",
                                   "static", "protected", "bool", "template", "mutable", "if", "public", "friend", "do", "goto",
                                   "auto", "void", "enum", "else", "break", "extern", "using", "class", "case", "double", "short",
                                   "default", "register", "explicit", "signed", "typename", "try", "this", "switch", "continue",
                                   "inline", "delete", "return", "namespace", "throw", "new", "restrict", "noexcept", "constexpr",
                                   "nullptr_t", "size_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t", "int8_t", "int16_t",
                                   "int32_t", "int64_t", "package", "func", "go", "defer", "chan", "select", "range", "type",
                                   "interface", "map", "var", "fallthrough", "fn", "let", "mut", "impl", "trait", "pub", "use", "mod",
                                   "crate", "match", "loop", "where", "move", "ref", "dyn", "unsafe", "async", "await", "final",
                                   "abstract", "extends", "implements", "instanceof", "throws", "synchronized", "transient",
                                   "native", "strictfp", "val", "fun", "object", "override", "open", "data", "sealed", "when", "is",
                                   "as", "in", "out", "lateinit", "companion", "init", "def", "trait", "lazy", "implicit", "yield",
                                   "func", "guard", "let", "struct", "protocol", "extension", "self", "Self", "string", "bool",
                                   "byte", "rune", "error", "float32", "float64", "int32", "int64", "uint", "uintptr", NULL};
static const char *const c_lit[] = {"true", "false", "NULL", "nullptr", "nil", "null", "None", "Some", "Ok", "Err", "iota", NULL};
static const char *const sh_kw[] = {"if", "then", "else", "elif", "fi", "for", "while", "in", "do", "done", "case", "esac", "function",
                                    "until", "select", "return", "local", "export", "declare", "readonly", "unset", NULL};
static const char *const sh_bi[] = {"echo", "cd", "pwd", "ls", "cat", "grep", "sed", "awk", "printf", "read", "exit", "source", "eval",
                                    "exec", "set", "shift", "test", "trap", "true", "false", "kill", "alias", "type", "command", NULL};
static const char *const rb_kw[] = {"and", "then", "defined", "module", "in", "return", "redo", "if", "BEGIN", "retry", "end", "for",
                                    "self", "when", "next", "until", "do", "begin", "unless", "END", "rescue", "else", "break",
                                    "undef", "not", "super", "class", "case", "require", "yield", "alias", "while", "ensure", "elsif",
                                    "or", "include", "attr_reader", "attr_writer", "attr_accessor", "def", "puts", NULL};
static const char *const lua_kw[] = {"and", "break", "do", "else", "elseif", "end", "for", "goto", "if", "in", "local", "not", "or",
                                     "repeat", "return", "then", "until", "while", "function", NULL};

static void emit(Buf *line, ThemeColor *c, Str text) {
    if (!text.len) return;
    if (c) th_fg(line, *c, text);
    else buf_addstr(line, text);
}

static bool ident_start(char c) { return isalpha((unsigned char)c) || c == '_' || c == '$' || (unsigned char)c >= 0x80; }
static bool ident_char(char c) { return isalnum((unsigned char)c) || c == '_' || c == '$' || (unsigned char)c >= 0x80; }

// Highlights one language family. The output is split at newlines by the caller, styles closed
// per line as pi's renderHighlightedHtml does.
static void highlight_fam(Str code, LangFam f, Lines *out) {
    Buf line = {0};
    ThemeColor KW = T_SYN_KEYWORD, LIT = T_SYN_NUMBER, NUM = T_SYN_NUMBER, STR_ = T_SYN_STRING, COM = T_SYN_COMMENT,
               FN = T_SYN_FUNCTION, BI = T_SYN_TYPE, PAR = T_SYN_VARIABLE, META = T_MUTED, ATTR = T_SYN_VARIABLE;
    const char *const *kw = f == L_PY ? py_kw : f == L_JS ? js_kw : f == L_SH ? sh_kw : f == L_RUBY ? rb_kw : f == L_LUA ? lua_kw : c_kw;
    const char *const *lit = f == L_PY ? py_lit : f == L_JS ? js_lit : c_lit;
    const char *const *bi = f == L_PY ? py_bi : f == L_JS ? js_bi : f == L_SH ? sh_bi : NULL;
    bool hash_comment = f == L_PY || f == L_SH || f == L_RUBY;
    bool slash_comment = f == L_JS || f == L_C || f == L_GO || f == L_RUST || f == L_JAVA || f == L_PHP;
    // a function signature: from its keyword to its end (":" in python, "{" or ";" elsewhere)
    bool in_sig = false, in_params = false;
    int par_depth = 0;
#define OUT(color, s)                                                    \
    do {                                                                 \
        Str s_ = (s);                                                    \
        for (size_t q_ = 0; q_ < s_.len;) {                              \
            size_t e_ = q_;                                              \
            while (e_ < s_.len && s_.ptr[e_] != '\n') e_++;              \
            emit(&line, (color), str_sub(s_, q_, e_));                   \
            if (e_ >= s_.len) break;                                     \
            lines_addb(out, &line);                                      \
            q_ = e_ + 1;                                                 \
        }                                                                \
    } while (0)
    size_t i = 0;
    while (i < code.len) {
        char c = code.ptr[i];
        ThemeColor *ctx = in_params ? &PAR : in_sig ? &FN : NULL;
        if (c == '\n') {
            lines_addb(out, &line);
            i++;
            continue;
        }
        // comments
        if ((hash_comment && c == '#') || (slash_comment && c == '/' && i + 1 < code.len && code.ptr[i + 1] == '/') ||
            (f == L_LUA && c == '-' && i + 1 < code.len && code.ptr[i + 1] == '-')) {
            size_t j = i;
            while (j < code.len && code.ptr[j] != '\n') j++;
            OUT(&COM, str_sub(code, i, j));
            i = j;
            continue;
        }
        if (slash_comment && c == '/' && i + 1 < code.len && code.ptr[i + 1] == '*') {
            ptrdiff_t e = str_find(code, S("*/"), i + 2);
            size_t j = e < 0 ? code.len : (size_t)e + 2;
            OUT(&COM, str_sub(code, i, j));
            i = j;
            continue;
        }
        if (f == L_C && c == '#' && (i == 0 || code.ptr[i - 1] == '\n')) {
            size_t j = i;
            while (j < code.len && code.ptr[j] != '\n') j++;
            OUT(&META, str_sub(code, i, j));
            i = j;
            continue;
        }
        if (f == L_PY && c == '@' && (i == 0 || code.ptr[i - 1] == '\n' || code.ptr[i - 1] == ' ')) {
            size_t j = i + 1;
            while (j < code.len && (ident_char(code.ptr[j]) || code.ptr[j] == '.')) j++;
            OUT(&META, str_sub(code, i, j));
            i = j;
            continue;
        }
        // strings
        if (c == '"' || c == '\'' || (c == '`' && f == L_JS)) {
            size_t j = i + 1;
            bool triple = f == L_PY && i + 2 < code.len && code.ptr[i + 1] == c && code.ptr[i + 2] == c;
            if (triple) {
                ptrdiff_t e = str_find(code, str_n(code.ptr + i, 3), i + 3);
                j = e < 0 ? code.len : (size_t)e + 3;
            } else {
                while (j < code.len && code.ptr[j] != c && (code.ptr[j] != '\n' || c == '`')) {
                    if (code.ptr[j] == '\\') j++;
                    j++;
                }
                if (j < code.len && code.ptr[j] == c) j++;
                if (j > code.len) j = code.len;
            }
            if (f == L_JSON && c == '"') {
                size_t k = j;
                while (k < code.len && (code.ptr[k] == ' ' || code.ptr[k] == '\t')) k++;
                if (k < code.len && code.ptr[k] == ':') {
                    OUT(&ATTR, str_sub(code, i, j));
                    i = j;
                    continue;
                }
            }
            OUT(&STR_, str_sub(code, i, j));
            i = j;
            continue;
        }
        // numbers
        if (isdigit((unsigned char)c) && (i == 0 || !ident_char(code.ptr[i - 1]))) {
            size_t j = i;
            while (j < code.len && (isalnum((unsigned char)code.ptr[j]) || code.ptr[j] == '.' || code.ptr[j] == '_')) j++;
            OUT(&NUM, str_sub(code, i, j));
            i = j;
            continue;
        }
        // shell variables
        if (f == L_SH && c == '$') {
            size_t j = i + 1;
            if (j < code.len && code.ptr[j] == '{') {
                while (j < code.len && code.ptr[j] != '}') j++;
                if (j < code.len) j++;
            } else
                while (j < code.len && (ident_char(code.ptr[j]) || (j == i + 1 && strchr("@#?$!*-0123456789", code.ptr[j])))) j++;
            OUT(&PAR, str_sub(code, i, j));
            i = j;
            continue;
        }
        if (ident_start(c) && c != '$') {
            size_t j = i;
            while (j < code.len && ident_char(code.ptr[j])) j++;
            Str w = str_sub(code, i, j);
            if (!in_params && in_list(kw, w)) {
                bool sig = (f == L_PY && (str_eqz(w, "def") || str_eqz(w, "class"))) ||
                           (f == L_JS && (str_eqz(w, "function") || str_eqz(w, "class"))) ||
                           ((f == L_GO || f == L_C) && (str_eqz(w, "func") || str_eqz(w, "fn"))) ||
                           (f == L_RUST && str_eqz(w, "fn")) || (f == L_RUBY && str_eqz(w, "def")) ||
                           (f == L_LUA && str_eqz(w, "function"));
                OUT(&KW, w);
                if (sig && !in_sig) {
                    in_sig = true;
                    par_depth = 0;
                }
            } else if (!in_params && in_list(lit, w)) OUT(&LIT, w);
            else if (!in_params && !in_sig && bi && in_list(bi, w)) OUT(&BI, w);
            else if (!in_params && !in_sig && j < code.len && code.ptr[j] == '(' && f != L_SH && f != L_JSON &&
                     (f == L_PY || f == L_JS || f == L_RUBY || f == L_LUA)) {
                // a call: no scope in highlight.js
                OUT(NULL, w);
            } else OUT(ctx, w);
            i = j;
            continue;
        }
        if (in_sig) {
            if (c == '(') {
                par_depth++;
                if (par_depth == 1) {
                    OUT(&FN, S("("));
                    in_params = true;
                    i++;
                    continue;
                }
            } else if (c == ')') {
                par_depth--;
                if (par_depth == 0) {
                    in_params = false;
                    OUT(&FN, S(")"));
                    i++;
                    continue;
                }
            } else if (par_depth == 0 && ((f == L_PY && c == ':') || ((f != L_PY) && (c == '{' || c == ';')) || (f == L_RUBY && c == '\n'))) {
                if (f == L_PY) OUT(&FN, S(":"));
                else OUT(NULL, str_n(code.ptr + i, 1));
                in_sig = false;
                i++;
                continue;
            }
        }
        // a run of other characters in the current scope
        size_t j = i + 1;
        while (j < code.len && !ident_start(code.ptr[j]) && !isdigit((unsigned char)code.ptr[j]) && !strchr("\"'`#/\n$@-(){};:", code.ptr[j])) j++;
        ThemeColor *cc = in_params ? &PAR : in_sig ? &FN : NULL;
        OUT(cc, str_sub(code, i, j));
        i = j;
    }
#undef OUT
    lines_addb(out, &line);
}

void md_highlight(Str code, Str lang, Lines *out) {
    LangFam f = lang_family(lang);
    if (f == L_NONE && lang_plain(lang)) {
        for (size_t i = 0;;) {
            size_t j = i;
            while (j < code.len && code.ptr[j] != '\n') j++;
            lines_add(out, str_sub(code, i, j));
            if (j >= code.len) break;
            i = j + 1;
        }
        return;
    }
    if (f == L_NONE) {
        for (size_t i = 0;;) {
            size_t j = i;
            while (j < code.len && code.ptr[j] != '\n') j++;
            Buf b = {0};
            th_fg(&b, T_MD_CODE_BLOCK, str_sub(code, i, j));
            lines_addb(out, &b);
            if (j >= code.len) break;
            i = j + 1;
        }
        return;
    }
    highlight_fam(code, f, out);
}
