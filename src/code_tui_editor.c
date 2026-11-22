// Saetta Code TUI: the prompt editor with its autocomplete ("/" commands, "@" files, paths).
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at v1.5.2,
// b4afddd): packages/tui/src/components/editor.ts (layout, word wrap, cursor, history, paste
// markers, the autocomplete triggers), autocomplete.ts (CombinedAutocompleteProvider), fuzzy.ts,
// and packages/coding-agent/src/modes/interactive/components/custom-editor.ts (the working status
// in the top border); the kill ring, undo and character jump of editor.ts with kill-ring.ts and
// undo-stack.ts (code_tui_ring.c).
#include "code_tui.h"
#include "code_tui_ring.h"

#include <ctype.h>
#include <math.h>
#include <dirent.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <spawn.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

typedef VEC(Buf) BufVec;
typedef VEC(char *) StrVec;

struct Editor {
    Comp base;
    BufVec lines;
    size_t cl, cc; // cursor line and byte column
    ThemeColor border;
    int rows, last_width;
    size_t scroll;
    // autocomplete
    bool ac_open, ac_force;
    SelectList ac;
    char *ac_prefix;
    SlashCmd *cmds;
    size_t ncmds;
    EditorArgFn arg_fn;
    void *arg_ctx;
    char *cwd, *fd;
    // history (newest first)
    StrVec hist;
    long hist_idx;
    char *draft;
    // paste markers
    StrVec pastes;
    // the text joined
    Buf joined;
    // the working status in the top border (custom-editor.ts)
    bool working;
    int frame;
    char *work_msg;
    // the "@" lookup (fd) on its own thread, after a debounce, a stale one cancelled
    int waker;
    uint64_t req_id, debounce_ns;
    bool req_pending, req_force;
    struct AcJob *jobs;
    // the kill ring (Emacs-style kill and yank), what the last action was, the character jump
    // awaiting its character, and the undo snapshots (editor.ts)
    KillRing kill;
    enum { LA_NONE, LA_KILL, LA_YANK, LA_TYPE_WORD } last_action;
    int jump; // 1 forward, -1 backward, 0 none
    UndoStack undo;
};

// One "@" lookup: what it searches, and its result once done.
typedef struct AcJob {
    struct AcJob *next;
    pthread_t th;
    uint64_t id;
    char *cwd, *fd, *prefix;
    bool quoted;
    // the editor's state it was made for (a result for another is dropped)
    char *snap_text;
    size_t snap_line, snap_col;
    SelectList list;
    _Atomic bool done, cancel;
    _Atomic int pid;
    int waker;
} AcJob;

static void ac_cancel(Editor *e);
static void job_free(AcJob *j);
static Str before_cursor(const Editor *e);

static char *dupz(const char *s) {
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static Buf *cur_line(Editor *e) { return &e->lines.ptr[e->cl]; }
static Str line_at(const Editor *e, size_t i) { return buf_str(&e->lines.ptr[i]); }

static void editor_comp_render(Comp *c, int width, Lines *out) { editor_render((Editor *)c, width, out); }
static void editor_comp_destroy(Comp *c) { (void)c; } // owned by the interactive mode
static const CompVt editor_vt = {editor_comp_render, editor_comp_destroy, true};

Comp *editor_comp(Editor *e) { return &e->base; }

static void lines_reset(Editor *e, Str text) {
    for (size_t i = 0; i < e->lines.len; i++) buf_free(&e->lines.ptr[i]);
    e->lines.len = 0;
    size_t i = 0;
    for (;;) {
        size_t j = i;
        while (j < text.len && text.ptr[j] != '\n') j++;
        Buf b = {0};
        buf_add(&b, text.ptr + i, j - i);
        if (!b.ptr) buf_adds(&b, "");
        vec_push(&e->lines, b);
        if (j >= text.len) break;
        i = j + 1;
    }
}

Editor *editor_new(const char *cwd, const char *fd_path) {
    Editor *e = xcalloc(1, sizeof *e);
    e->base.vt = &editor_vt;
    lines_reset(e, S(""));
    e->border = T_BORDER_MUTED;
    e->rows = 24;
    e->last_width = 80;
    e->cwd = dupz(cwd);
    e->fd = fd_path ? dupz(fd_path) : NULL;
    e->hist_idx = -1;
    e->waker = -1;
    return e;
}

static void ac_close(Editor *e) {
    if (e->ac_open) sel_free(&e->ac);
    e->ac_open = false;
    free(e->ac_prefix);
    e->ac_prefix = NULL;
}

void editor_free(Editor *e) {
    if (!e) return;
    ac_cancel(e);
    while (e->jobs) {
        AcJob *j = e->jobs;
        e->jobs = j->next;
        job_free(j);
    }
    ac_close(e);
    for (size_t i = 0; i < e->lines.len; i++) buf_free(&e->lines.ptr[i]);
    vec_free(&e->lines);
    for (size_t i = 0; i < e->hist.len; i++) free(e->hist.ptr[i]);
    vec_free(&e->hist);
    for (size_t i = 0; i < e->pastes.len; i++) free(e->pastes.ptr[i]);
    vec_free(&e->pastes);
    free(e->cmds);
    free(e->cwd);
    free(e->fd);
    free(e->draft);
    free(e->work_msg);
    buf_free(&e->joined);
    kill_ring_free(&e->kill);
    undo_clear(&e->undo);
    free(e->undo.v);
    free(e);
}

void editor_set_commands(Editor *e, const SlashCmd *cmds, size_t n) {
    free(e->cmds);
    e->cmds = xmalloc((n + 1) * sizeof *cmds);
    memcpy(e->cmds, cmds, n * sizeof *cmds);
    e->ncmds = n;
}

void editor_set_arg_completer(Editor *e, EditorArgFn fn, void *ctx) {
    e->arg_fn = fn;
    e->arg_ctx = ctx;
}

void editor_set_border(Editor *e, ThemeColor c) { e->border = c; }
void editor_set_rows(Editor *e, int rows) { e->rows = rows; }

void editor_set_working(Editor *e, const char *msg) {
    free(e->work_msg);
    e->work_msg = msg ? dupz(msg) : NULL;
    e->working = msg != NULL;
    e->frame = 0;
}

void editor_spin(Editor *e) { e->frame = (e->frame + 1) % 10; }

Str editor_text(const Editor *e) {
    Editor *m = (Editor *)e;
    m->joined.len = 0;
    for (size_t i = 0; i < e->lines.len; i++) {
        if (i) buf_addc(&m->joined, '\n');
        buf_addstr(&m->joined, line_at(e, i));
    }
    if (!m->joined.ptr) buf_adds(&m->joined, "");
    return buf_str(&m->joined);
}

bool editor_empty(const Editor *e) { return e->lines.len == 1 && e->lines.ptr[0].len == 0; }

static void push_undo(Editor *e);

void editor_set_text(Editor *e, Str text) {
    ac_close(e);
    e->last_action = LA_NONE;
    // setText: a snapshot when the content differs (programmatic changes are undoable)
    if (!str_eq(editor_text(e), text)) push_undo(e);
    lines_reset(e, text);
    e->cl = e->lines.len - 1;
    e->cc = e->lines.ptr[e->cl].len;
    e->scroll = 0;
    e->hist_idx = -1;
}

bool editor_autocomplete_open(const Editor *e) { return e->ac_open; }

void editor_add_history(Editor *e, Str text) {
    Str t = str_trim(text);
    if (!t.len) return;
    if (e->hist.len && str_eqz(t, e->hist.ptr[0])) return;
    char *p = xmalloc(t.len + 1);
    memcpy(p, t.ptr, t.len);
    p[t.len] = 0;
    vec_push(&e->hist, NULL);
    memmove(e->hist.ptr + 1, e->hist.ptr, (e->hist.len - 1) * sizeof *e->hist.ptr);
    e->hist.ptr[0] = p;
    if (e->hist.len > 100) free(e->hist.ptr[--e->hist.len]);
}

// ---- layout (layoutText, wordWrapLine) -------------------------------------------------------------

typedef struct {
    size_t start, end; // byte range in the logical line
} Chunk;
typedef VEC(Chunk) ChunkVec;

static bool is_ws_cp(uint32_t cp) { return tui_is_space(cp); }

static void word_wrap(Str line, int maxw, ChunkVec *out) {
    if (!line.len || maxw <= 0 || tui_width(line) <= maxw) {
        vec_push(out, ((Chunk){0, line.len}));
        return;
    }
    int curw = 0;
    size_t start = 0;
    long opp = -1;
    int oppw = 0;
    size_t i = 0;
    while (i < line.len) {
        size_t g = tui_grapheme(line, i);
        Str gs = str_n(line.ptr + i, g);
        int gw = tui_width(gs);
        size_t k = 0;
        bool ws = is_ws_cp(tui_cp(gs, &k));
        if (curw + gw > maxw) {
            if (opp >= 0 && curw - oppw + gw <= maxw) {
                vec_push(out, ((Chunk){start, (size_t)opp}));
                start = (size_t)opp;
                curw -= oppw;
            } else if (start < i) {
                vec_push(out, ((Chunk){start, i}));
                start = i;
                curw = 0;
            }
            opp = -1;
        }
        curw += gw;
        size_t nx = i + g;
        if (nx < line.len) {
            size_t kk = nx;
            uint32_t ncp = tui_cp(line, &kk);
            bool nws = is_ws_cp(ncp);
            size_t k2 = 0;
            uint32_t gcp = tui_cp(gs, &k2);
            if (ws && !nws) {
                opp = (long)nx;
                oppw = curw;
            } else if (!ws && !nws && (tui_cp_width(gcp) == 2 || tui_cp_width(ncp) == 2)) {
                opp = (long)nx;
                oppw = curw;
            }
        }
        i = nx;
    }
    vec_push(out, ((Chunk){start, line.len}));
}

typedef struct {
    size_t line, start, end;
    bool has_cursor;
    size_t cursor; // byte offset in the chunk
} LayoutLine;
typedef VEC(LayoutLine) LayoutVec;

static void layout(Editor *e, int width, LayoutVec *out) {
    if (editor_empty(e)) {
        vec_push(out, ((LayoutLine){0, 0, 0, true, 0}));
        return;
    }
    for (size_t i = 0; i < e->lines.len; i++) {
        Str line = line_at(e, i);
        bool cur = i == e->cl;
        if (tui_width(line) <= width) {
            vec_push(out, ((LayoutLine){i, 0, line.len, cur, cur ? e->cc : 0}));
            continue;
        }
        ChunkVec ch = {0};
        word_wrap(line, width, &ch);
        for (size_t k = 0; k < ch.len; k++) {
            Chunk c = ch.ptr[k];
            bool last = k + 1 == ch.len;
            bool has = false;
            size_t pos = 0;
            if (cur) {
                if (last) {
                    has = e->cc >= c.start;
                    pos = e->cc - c.start;
                } else {
                    has = e->cc >= c.start && e->cc < c.end;
                    if (has) pos = MIN(e->cc - c.start, c.end - c.start);
                }
            }
            vec_push(out, ((LayoutLine){i, c.start, c.end, has, pos}));
        }
        vec_free(&ch);
    }
}

static void border_line(Buf *b, ThemeColor c, int width, const char *dir, size_t hidden) {
    Buf s = {0};
    if (hidden > 0) {
        char label[64];
        snprintf(label, sizeof label, " %s %zu more ", dir, hidden);
        int lw = tui_width(str(label));
        if (lw + 2 <= width) {
            int left = (width - lw) / 2;
            tui_repeat(&s, "\xe2\x94\x80", left);
            buf_adds(&s, label);
            tui_repeat(&s, "\xe2\x94\x80", width - left - lw);
        } else {
            char ind[64];
            snprintf(ind, sizeof ind, "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80 %s %zu more ", dir, hidden);
            int rem = width - tui_width(str(ind));
            if (rem >= 0) {
                buf_adds(&s, ind);
                tui_repeat(&s, "\xe2\x94\x80", rem);
            } else {
                int el = MIN(3, MAX(0, width));
                tui_slice(&s, str(ind), 0, width - el, true);
                for (int k = 0; k < el; k++) buf_addc(&s, '.');
            }
        }
    } else tui_repeat(&s, "\xe2\x94\x80", width);
    th_fg(b, c, buf_str(&s));
    buf_free(&s);
}

static const char *const spinner[] = {"\xe2\xa0\x8b", "\xe2\xa0\x99", "\xe2\xa0\xb9", "\xe2\xa0\xb8", "\xe2\xa0\xbc",
                                      "\xe2\xa0\xb4", "\xe2\xa0\xa6", "\xe2\xa0\xa7", "\xe2\xa0\x87", "\xe2\xa0\x8f"};

// The top border with the working status in it (CustomEditor.renderTopBorder).
static void top_border(Editor *e, Buf *b, int width, size_t hidden) {
    if (!e->working || width <= 0) {
        border_line(b, e->border, width, "\xe2\x86\x91", hidden);
        return;
    }
    // the Loader's text: spinner and message in the border color, cut to width - 5
    Buf st = {0};
    th_fgz(&st, e->border, spinner[e->frame]);
    buf_addc(&st, ' ');
    th_fgz(&st, e->border, e->work_msg ? e->work_msg : "Working");
    Buf status = {0};
    tui_truncate(&status, buf_str(&st), MAX(1, width - 5), "", false);
    int sw = tui_width(buf_str(&status));
    if (width >= sw + 5) {
        th_fgz(b, e->border, "\xe2\x94\x80\xe2\x94\x80 ");
        buf_addstr(b, buf_str(&status));
        Buf rest = {0};
        buf_addc(&rest, ' ');
        tui_repeat(&rest, "\xe2\x94\x80", width - sw - 4);
        th_fg(b, e->border, buf_str(&rest));
        buf_free(&rest);
    } else border_line(b, e->border, width, "\xe2\x86\x91", hidden);
    buf_free(&st);
    buf_free(&status);
}

void editor_render(Editor *e, int width, Lines *out) {
    int padx = 0;
    int cw = MAX(1, width - padx * 2);
    int lw = MAX(1, cw - (padx ? 0 : 1));
    e->last_width = lw;
    LayoutVec ll = {0};
    layout(e, lw, &ll);
    int maxvis = MAX(5, e->rows * 3 / 10);
    size_t ci = 0;
    for (size_t i = 0; i < ll.len; i++)
        if (ll.ptr[i].has_cursor) {
            ci = i;
            break;
        }
    if (ci < e->scroll) e->scroll = ci;
    else if (ci >= e->scroll + (size_t)maxvis) e->scroll = ci - (size_t)maxvis + 1;
    size_t maxscroll = ll.len > (size_t)maxvis ? ll.len - (size_t)maxvis : 0;
    if (e->scroll > maxscroll) e->scroll = maxscroll;
    size_t vis_end = MIN(e->scroll + (size_t)maxvis, ll.len);
    Buf b = {0};
    top_border(e, &b, width, e->scroll);
    lines_addb(out, &b);
    for (size_t i = e->scroll; i < vis_end; i++) {
        LayoutLine *l = &ll.ptr[i];
        Str text = str_n(e->lines.ptr[l->line].ptr ? e->lines.ptr[l->line].ptr + l->start : "", l->end - l->start);
        int vw = tui_width(text);
        Buf d = {0};
        if (l->has_cursor) {
            size_t pos = MIN(l->cursor, text.len);
            buf_add(&d, text.ptr, pos);
            if (pos < text.len) {
                size_t g = tui_grapheme(text, pos);
                buf_adds(&d, "\x1b[7m");
                buf_add(&d, text.ptr + pos, g);
                buf_adds(&d, "\x1b[27m");
                buf_add(&d, text.ptr + pos + g, text.len - pos - g);
            } else {
                buf_adds(&d, "\x1b[7m \x1b[27m");
                vw++;
            }
        } else buf_addstr(&d, text);
        tui_spaces(&d, cw - vw);
        lines_addb(out, &d);
    }
    size_t below = ll.len - vis_end;
    border_line(&b, e->border, width, "\xe2\x86\x93", below);
    lines_addb(out, &b);
    if (e->ac_open) {
        Lines al = {0};
        sel_render(&e->ac, cw, &al);
        for (size_t i = 0; i < al.n; i++) {
            Buf x = {0};
            buf_adds(&x, al.v[i]);
            tui_spaces(&x, cw - tui_width(str(al.v[i])));
            lines_addb(out, &x);
        }
        lines_free(&al);
    }
    vec_free(&ll);
}

// ---- fuzzy matching (fuzzy.ts) -------------------------------------------------------------------

static double score_positions(Str lower, const size_t *pos, size_t n) {
    double score = 0;
    long last = -1;
    int consec = 0;
    for (size_t k = 0; k < n; k++) {
        long i = (long)pos[k];
        if (last == i - 1) {
            consec++;
            score -= consec * 5;
        } else {
            consec = 0;
            if (last >= 0) score += (double)(i - last - 1) * 2;
        }
        if (i == 0 || strchr(" \t\n-_./:", lower.ptr[i - 1])) score -= 10;
        score += (double)i * 0.1;
        last = i;
    }
    return score;
}

static void lower_ascii(Buf *b, Str s) {
    for (size_t i = 0; i < s.len; i++) {
        char c = s.ptr[i];
        buf_addc(b, c >= 'A' && c <= 'Z' ? (char)(c + 32) : c);
    }
    if (!b->ptr) buf_adds(b, "");
}

static bool match_query(Str q, Str t, double *score) {
    if (!q.len) {
        *score = 0;
        return true;
    }
    if (q.len > t.len) return false;
    size_t *pos = xmalloc(q.len * sizeof *pos);
    long last = -1;
    for (size_t k = 0; k < q.len; k++) {
        const char *f = memchr(t.ptr + last + 1, q.ptr[k], t.len - (size_t)(last + 1));
        if (!f) {
            free(pos);
            return false;
        }
        pos[k] = (size_t)(f - t.ptr);
        last = (long)pos[k];
    }
    double s = score_positions(t, pos, q.len);
    for (ptrdiff_t st = str_find(t, q, 0); st >= 0; st = str_find(t, q, (size_t)st + 1)) {
        for (size_t k = 0; k < q.len; k++) pos[k] = (size_t)st + k;
        double s2 = score_positions(t, pos, q.len);
        if (s2 < s) s = s2;
    }
    if (str_eq(q, t)) s -= 100;
    free(pos);
    *score = s;
    return true;
}

static bool fuzzy_match(Str query, Str text, double *score) {
    Buf q = {0}, t = {0};
    lower_ascii(&q, query);
    lower_ascii(&t, text);
    bool ok = match_query(buf_str(&q), buf_str(&t), score);
    if (!ok) {
        // letters then digits, or digits then letters, swapped
        size_t i = 0;
        Str qs = buf_str(&q);
        bool alpha_first = qs.len && qs.ptr[0] >= 'a' && qs.ptr[0] <= 'z';
        bool digit_first = qs.len && qs.ptr[0] >= '0' && qs.ptr[0] <= '9';
        if (alpha_first) while (i < qs.len && qs.ptr[i] >= 'a' && qs.ptr[i] <= 'z') i++;
        else if (digit_first) while (i < qs.len && qs.ptr[i] >= '0' && qs.ptr[i] <= '9') i++;
        size_t j = i;
        if (alpha_first) while (j < qs.len && qs.ptr[j] >= '0' && qs.ptr[j] <= '9') j++;
        else if (digit_first) while (j < qs.len && qs.ptr[j] >= 'a' && qs.ptr[j] <= 'z') j++;
        if ((alpha_first || digit_first) && i > 0 && i < qs.len && j == qs.len) {
            Buf sw = {0};
            buf_add(&sw, qs.ptr + i, qs.len - i);
            buf_add(&sw, qs.ptr, i);
            double s2;
            if (match_query(buf_str(&sw), buf_str(&t), &s2)) {
                ok = true;
                *score = s2 + 5;
            }
            buf_free(&sw);
        }
    }
    buf_free(&q);
    buf_free(&t);
    return ok;
}

typedef struct {
    size_t idx;
    double score;
} Scored;

static int scored_cmp(const void *a, const void *b) {
    const Scored *x = a, *y = b;
    if (x->score < y->score) return -1;
    if (x->score > y->score) return 1;
    return x->idx < y->idx ? -1 : x->idx > y->idx; // stable
}

// fuzzyFilter: indexes of the matching texts, best first. Tokens split on whitespace and "/".
static size_t fuzzy_filter(const char *const *texts, size_t n, Str query, size_t *out) {
    Str q = str_trim(query);
    if (!q.len) {
        for (size_t i = 0; i < n; i++) out[i] = i;
        return n;
    }
    Scored *sc = xmalloc((n + 1) * sizeof *sc);
    size_t m = 0;
    for (size_t i = 0; i < n; i++) {
        double total = 0;
        bool all = true, any = false;
        for (size_t p = 0; p < q.len && all;) {
            while (p < q.len && (q.ptr[p] == ' ' || q.ptr[p] == '\t' || q.ptr[p] == '/')) p++;
            size_t s = p;
            while (p < q.len && !(q.ptr[p] == ' ' || q.ptr[p] == '\t' || q.ptr[p] == '/')) p++;
            if (p == s) break;
            any = true;
            double sc1;
            if (fuzzy_match(str_n(q.ptr + s, p - s), str(texts[i]), &sc1)) total += sc1;
            else all = false;
        }
        if (!any) {
            for (size_t k = 0; k < n; k++) out[k] = k;
            free(sc);
            return n;
        }
        if (all) sc[m++] = (Scored){i, total};
    }
    qsort(sc, m, sizeof *sc, scored_cmp);
    for (size_t k = 0; k < m; k++) out[k] = sc[k].idx;
    free(sc);
    return m;
}

// ---- autocomplete (autocomplete.ts) ----------------------------------------------------------------

static bool path_delim(uint32_t cp) { return cp == ' ' || cp == '\t' || cp == '"' || cp == '\'' || cp == '=' || tui_is_space(cp); }

// The token after the last path delimiter, without opening wrappers that are not closed in it.
static Str last_token(Str text) {
    size_t start = 0, i = 0;
    while (i < text.len) {
        uint32_t cp = tui_cp(text, &i);
        if (path_delim(cp)) start = i;
    }
    Str t = str_sub(text, start, text.len);
    while (t.len) {
        char o = t.ptr[0];
        char c = o == '(' ? ')' : o == '[' ? ']' : o == '{' ? '}' : o == '<' ? '>' : o == '`' ? '`' : 0;
        if (!c || memchr(t.ptr + 1, c, t.len - 1)) break;
        t = str_sub(t, 1, t.len);
    }
    return t;
}

// findUnclosedQuoteStart and extractQuotedPrefix: the text from an unclosed quote (with its @).
static bool token_start(Str text, size_t idx) {
    size_t s = idx;
    while (s > 0 && strchr("([{<`", text.ptr[s - 1])) s--;
    if (s == 0) return true;
    char p = text.ptr[s - 1];
    return p == ' ' || p == '\t' || p == '"' || p == '\'' || p == '=' || p == '\n';
}

static bool quoted_prefix(Str text, Str *out) {
    bool inq = false;
    long qs = -1;
    for (size_t i = 0; i < text.len; i++)
        if (text.ptr[i] == '"') {
            inq = !inq;
            if (inq) qs = (long)i;
        }
    if (!inq) return false;
    if (qs > 0 && text.ptr[qs - 1] == '@') {
        if (!token_start(text, (size_t)qs - 1)) return false;
        *out = str_sub(text, (size_t)qs - 1, text.len);
        return true;
    }
    if (!token_start(text, (size_t)qs)) return false;
    *out = str_sub(text, (size_t)qs, text.len);
    return true;
}

static bool at_prefix(Str text, Str *out) {
    Str q;
    if (quoted_prefix(text, &q) && q.len >= 2 && q.ptr[0] == '@' && q.ptr[1] == '"') {
        *out = q;
        return true;
    }
    Str t = last_token(text);
    if (t.len && t.ptr[0] == '@') {
        *out = t;
        return true;
    }
    return false;
}

// The trigger pattern: the text before the cursor ends in a token that starts with @ or #
// (after wrappers) at a boundary, or in @" with the quote open.
static bool trigger_match(Str text) {
    Str q;
    if (quoted_prefix(text, &q) && q.len >= 2 && q.ptr[0] == '@' && q.ptr[1] == '"') return true;
    size_t start = 0, i = 0;
    while (i < text.len) {
        uint32_t cp = tui_cp(text, &i);
        if (tui_is_space(cp)) start = i;
    }
    Str t = str_sub(text, start, text.len);
    while (t.len && strchr("([{<`", t.ptr[0])) t = str_sub(t, 1, t.len);
    return t.len && (t.ptr[0] == '@' || t.ptr[0] == '#');
}

typedef struct {
    char *path;
    bool dir;
    int score;
    size_t order;
} Entry;
typedef VEC(Entry) EntryVec;

// fd as pi runs it (walkDirectoryWithFd): paths relative to base, directories ending in "/".
static void fd_walk(const char *fd, const char *base, Str query, int max_depth, EntryVec *out, AcJob *job) {
    if (!fd || (job && atomic_load(&job->cancel))) return;
    char depth[16];
    snprintf(depth, sizeof depth, "%d", max_depth);
    // buildFdPathQuery: a regex of the query, path segments joined by [\\/]
    Buf pat = {0};
    bool has_slash = memchr(query.ptr, '/', query.len) != NULL;
    if (!has_slash) {
        for (size_t i = 0; i < query.len; i++) {
            if (strchr(".*+?^${}()|[]\\", query.ptr[i])) buf_addc(&pat, '\\');
            buf_addc(&pat, query.ptr[i]);
        }
    } else {
        bool trailing = query.len && query.ptr[query.len - 1] == '/';
        Str t = query;
        while (t.len && t.ptr[0] == '/') t = str_sub(t, 1, t.len);
        while (t.len && t.ptr[t.len - 1] == '/') t.len--;
        if (!t.len) buf_addstr(&pat, query);
        else {
            bool first = true;
            for (size_t i = 0; i < t.len;) {
                size_t j = i;
                while (j < t.len && t.ptr[j] != '/') j++;
                if (j > i) {
                    if (!first) buf_adds(&pat, "[\\\\/]");
                    first = false;
                    for (size_t k = i; k < j; k++) {
                        if (strchr(".*+?^${}()|[]\\", t.ptr[k])) buf_addc(&pat, '\\');
                        buf_addc(&pat, t.ptr[k]);
                    }
                }
                i = j + 1;
            }
            if (trailing) buf_adds(&pat, "[\\\\/]");
        }
    }
    const char *argv[32];
    int n = 0;
    argv[n++] = fd;
    argv[n++] = "--base-directory";
    argv[n++] = base;
    argv[n++] = "--max-results";
    argv[n++] = "100";
    argv[n++] = "--type";
    argv[n++] = "f";
    argv[n++] = "--type";
    argv[n++] = "d";
    argv[n++] = "--follow";
    argv[n++] = "--hidden";
    argv[n++] = "--exclude";
    argv[n++] = ".git";
    argv[n++] = "--exclude";
    argv[n++] = ".git/*";
    argv[n++] = "--exclude";
    argv[n++] = ".git/**";
    if (max_depth > 0) {
        argv[n++] = "--max-depth";
        argv[n++] = depth;
    }
    if (has_slash) argv[n++] = "--full-path";
    if (query.len) argv[n++] = pat.ptr;
    argv[n] = NULL;
    int pfd[2];
    if (tui_pipe(pfd)) {
        buf_free(&pat);
        return;
    }
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, pfd[1], 1);
    posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", 0x0001, 0);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", 0, 0);
    pid_t pid;
    int rc = tui_spawn(&pid, (char *const *)argv, &fa, false, false);
    posix_spawn_file_actions_destroy(&fa);
    close(pfd[1]);
    if (rc == 0 && job) {
        atomic_store(&job->pid, pid);
        if (atomic_load(&job->cancel)) kill(pid, SIGKILL); // cancelled while it started
    }
    Buf o = {0};
    if (rc == 0) {
        char chunk[8192];
        ssize_t got;
        while ((got = read(pfd[0], chunk, sizeof chunk)) > 0) buf_add(&o, chunk, (size_t)got);
    }
    close(pfd[0]);
    int status = 1;
    if (rc == 0) waitpid(pid, &status, 0);
    if (job) atomic_store(&job->pid, 0);
    if (job && atomic_load(&job->cancel)) status = 1;
    if (rc == 0 && WIFEXITED(status) && WEXITSTATUS(status) == 0 && o.len) {
        Str s = str_trim(buf_str(&o));
        for (size_t i = 0; i < s.len;) {
            size_t j = i;
            while (j < s.len && s.ptr[j] != '\n') j++;
            Str l = str_n(s.ptr + i, j - i);
            if (l.len) {
                bool dir = l.ptr[l.len - 1] == '/';
                Str np = dir ? str_n(l.ptr, l.len - 1) : l;
                bool git = str_eqz(np, ".git") || str_starts(np, S(".git/")) || str_find(np, S("/.git/"), 0) >= 0;
                if (!git) {
                    char *p = xmalloc(l.len + 1);
                    memcpy(p, l.ptr, l.len);
                    p[l.len] = 0;
                    vec_push(out, ((Entry){p, dir, 0, out->len}));
                }
            }
            i = j + 1;
        }
    }
    buf_free(&o);
    buf_free(&pat);
}

static Str base_name(Str p) {
    while (p.len > 1 && p.ptr[p.len - 1] == '/') p.len--;
    size_t i = p.len;
    while (i > 0 && p.ptr[i - 1] != '/') i--;
    return str_sub(p, i, p.len);
}

static bool contains_ci(Str hay, Str needle) {
    if (!needle.len) return true;
    for (size_t i = 0; i + needle.len <= hay.len; i++) {
        size_t k = 0;
        while (k < needle.len) {
            char a = hay.ptr[i + k], b = needle.ptr[k];
            if (a >= 'A' && a <= 'Z') a += 32;
            if (b >= 'A' && b <= 'Z') b += 32;
            if (a != b) break;
            k++;
        }
        if (k == needle.len) return true;
    }
    return false;
}

static bool starts_ci(Str s, Str p) { return s.len >= p.len && contains_ci(str_n(s.ptr, p.len), p); }

static int score_entry(Str path, Str query, bool dir) {
    Str name = base_name(path);
    int score = 0;
    if (name.len == query.len && starts_ci(name, query)) score = 100;
    else if (starts_ci(name, query)) score = 80;
    else if (contains_ci(name, query)) score = 50;
    else if (contains_ci(path, query)) score = 30;
    if (dir && score > 0) score += 10;
    return score;
}

static size_t depth_of(Str p) {
    size_t d = 0;
    for (size_t i = 0; i < p.len;) {
        size_t j = i;
        while (j < p.len && p.ptr[j] != '/') j++;
        if (j > i) d++;
        i = j + 1;
    }
    return d;
}

static int entry_cmp(const void *a, const void *b) {
    const Entry *x = a, *y = b;
    if (x->score != y->score) return y->score - x->score;
    size_t dx = depth_of(str(x->path)), dy = depth_of(str(y->path));
    if (dx != dy) return dx < dy ? -1 : 1;
    size_t lx = tui_js_len(str(x->path)), ly = tui_js_len(str(y->path));
    if (lx != ly) return lx < ly ? -1 : 1;
    int c = strcasecmp(x->path, y->path);
    if (c) return c;
    c = strcmp(x->path, y->path);
    if (c) return c;
    return x->order < y->order ? -1 : x->order > y->order;
}

static bool is_dir(const char *p) {
    struct stat st;
    return stat(p, &st) == 0 && S_ISDIR(st.st_mode);
}

// A path that needs quotes: it has whitespace in it.
static bool needs_quotes(Str p) {
    for (size_t i = 0; i < p.len;) {
        uint32_t cp = tui_cp(p, &i);
        if (tui_is_space(cp)) return true;
    }
    return false;
}

static void completion_value(Buf *b, Str path, bool at, bool quoted) {
    bool q = quoted || needs_quotes(path);
    if (at) buf_addc(b, '@');
    if (q) buf_addc(b, '"');
    buf_addstr(b, path);
    if (q) buf_addc(b, '"');
}

// getFuzzyFileSuggestions: the "@" completion through fd.
static void fuzzy_files(const char *cwd, const char *fdp, AcJob *job, Str query, bool quoted, SelectList *list) {
    // resolveScopedFuzzyQuery: a query with a "/" searches below its folder
    char *base = (char *)cwd;
    char *scoped_base = NULL;
    Str display_base = S("");
    Str fq = query;
    ptrdiff_t slash = -1;
    for (size_t i = 0; i < query.len; i++)
        if (query.ptr[i] == '/') slash = (ptrdiff_t)i;
    if (slash >= 0) {
        Str db = str_sub(query, 0, (size_t)slash + 1);
        Buf bd = {0};
        if (str_starts(db, S("~/"))) {
            buf_adds(&bd, getenv("HOME") ? getenv("HOME") : "");
            buf_addstr(&bd, str_sub(db, 1, db.len));
        } else if (db.ptr[0] == '/') buf_addstr(&bd, db);
        else {
            buf_adds(&bd, cwd);
            buf_addc(&bd, '/');
            buf_addstr(&bd, db);
        }
        if (is_dir(bd.ptr)) {
            scoped_base = bd.ptr;
            base = scoped_base;
            display_base = db;
            fq = str_sub(query, (size_t)slash + 1, query.len);
        } else buf_free(&bd);
    }
    EntryVec all = {0}, rec = {0};
    fd_walk(fdp, base, fq, 1, &all, job);
    fd_walk(fdp, base, fq, 0, &rec, job);
    for (size_t i = 0; i < rec.len; i++) {
        bool seen = false;
        for (size_t k = 0; k < all.len && !seen; k++) seen = !strcmp(all.ptr[k].path, rec.ptr[i].path);
        if (seen) free(rec.ptr[i].path);
        else {
            rec.ptr[i].order = all.len;
            vec_push(&all, rec.ptr[i]);
        }
    }
    vec_free(&rec);
    size_t m = 0;
    for (size_t i = 0; i < all.len; i++) {
        all.ptr[i].score = fq.len ? score_entry(str(all.ptr[i].path), fq, all.ptr[i].dir) : 1;
        if (all.ptr[i].score > 0) all.ptr[m++] = all.ptr[i];
        else free(all.ptr[i].path);
    }
    all.len = m;
    qsort(all.ptr, all.len, sizeof *all.ptr, entry_cmp);
    for (size_t i = 0; i < all.len && i < 20; i++) {
        Entry *en = &all.ptr[i];
        Str p = str(en->path);
        Str no_slash = en->dir ? str_n(p.ptr, p.len - 1) : p;
        Buf disp = {0}, val = {0}, comp = {0}, label = {0};
        buf_addstr(&disp, display_base);
        buf_addstr(&disp, no_slash);
        buf_addstr(&comp, buf_str(&disp));
        if (en->dir) buf_addc(&comp, '/');
        completion_value(&val, buf_str(&comp), true, quoted);
        buf_addstr(&label, base_name(no_slash));
        if (en->dir) buf_addc(&label, '/');
        sel_add(list, val.ptr, label.ptr, disp.ptr);
        buf_free(&disp);
        buf_free(&val);
        buf_free(&comp);
        buf_free(&label);
    }
    for (size_t i = 0; i < all.len; i++) free(all.ptr[i].path);
    vec_free(&all);
    free(scoped_base);
}

// extractPathPrefix
static bool path_prefix(Str text, bool force, Str *out) {
    Str q;
    if (quoted_prefix(text, &q)) {
        *out = q;
        return true;
    }
    Str p = last_token(text);
    if (force) {
        *out = p;
        return true;
    }
    if (memchr(p.ptr, '/', p.len) || (p.len && p.ptr[0] == '.') || str_starts(p, S("~/"))) {
        *out = p;
        return true;
    }
    if (!p.len && text.len) {
        size_t k = text.len - 1;
        while (k > 0 && ((unsigned char)text.ptr[k] & 0xc0) == 0x80) k--;
        size_t j = k;
        if (tui_is_space(tui_cp(text, &j))) {
            *out = p;
            return true;
        }
    }
    return false;
}

typedef struct {
    char *name;
    bool dir;
} DirEnt;

static int dirent_cmp(const void *a, const void *b) {
    const DirEnt *x = a, *y = b;
    if (x->dir != y->dir) return x->dir ? -1 : 1;
    int c = strcasecmp(x->name, y->name);
    return c ? c : strcmp(x->name, y->name);
}

// getFileSuggestions: the entries of a folder that start with the prefix's last part.
static void file_suggestions(Editor *e, Str prefix, SelectList *list) {
    bool at = false, quoted = false;
    Str raw = prefix;
    if (str_starts(prefix, S("@\""))) {
        raw = str_sub(prefix, 2, prefix.len);
        at = quoted = true;
    } else if (str_starts(prefix, S("\""))) {
        raw = str_sub(prefix, 1, prefix.len);
        quoted = true;
    } else if (str_starts(prefix, S("@"))) {
        raw = str_sub(prefix, 1, prefix.len);
        at = true;
    }
    const char *home = getenv("HOME") ? getenv("HOME") : "";
    Buf exp = {0};
    if (str_starts(raw, S("~/"))) {
        buf_adds(&exp, home);
        buf_addstr(&exp, str_sub(raw, 1, raw.len));
    } else if (str_eqz(raw, "~")) buf_adds(&exp, home);
    else buf_addstr(&exp, raw);
    if (!exp.ptr) buf_adds(&exp, "");
    Str ex = buf_str(&exp);
    bool absolute = (raw.len && raw.ptr[0] == '~') || (ex.len && ex.ptr[0] == '/');
    Buf dir = {0};
    Str search = S("");
    bool root = !raw.len || str_eqz(raw, "./") || str_eqz(raw, "../") || str_eqz(raw, "~") || str_eqz(raw, "~/") ||
                str_eqz(raw, "/");
    if (root || (raw.len && raw.ptr[raw.len - 1] == '/')) {
        if (absolute) buf_addstr(&dir, ex);
        else {
            buf_adds(&dir, e->cwd);
            buf_addc(&dir, '/');
            buf_addstr(&dir, ex);
        }
    } else {
        size_t sl = ex.len;
        while (sl > 0 && ex.ptr[sl - 1] != '/') sl--;
        Str d = sl ? str_n(ex.ptr, sl) : S(".");
        search = str_sub(ex, sl, ex.len);
        if (absolute) buf_addstr(&dir, d);
        else {
            buf_adds(&dir, e->cwd);
            buf_addc(&dir, '/');
            buf_addstr(&dir, d);
        }
    }
    DIR *dp = opendir(dir.ptr ? dir.ptr : ".");
    VEC(DirEnt) ents = {0};
    if (dp) {
        struct dirent *de;
        while ((de = readdir(dp))) {
            if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
            if (!starts_ci(str(de->d_name), search)) continue;
            Buf full = {0};
            buf_addstr(&full, buf_str(&dir));
            buf_addc(&full, '/');
            buf_adds(&full, de->d_name);
            bool d = de->d_type == DT_DIR || (de->d_type == DT_LNK && is_dir(full.ptr));
            buf_free(&full);
            vec_push(&ents, ((DirEnt){dupz(de->d_name), d}));
        }
        closedir(dp);
    }
    qsort(ents.ptr, ents.len, sizeof *ents.ptr, dirent_cmp);
    for (size_t i = 0; i < ents.len; i++) {
        Buf rel = {0}, val = {0}, label = {0};
        const char *name = ents.ptr[i].name;
        if (raw.len && raw.ptr[raw.len - 1] == '/') {
            buf_addstr(&rel, raw);
            buf_adds(&rel, name);
        } else if (memchr(raw.ptr, '/', raw.len)) {
            size_t sl = raw.len;
            while (sl > 0 && raw.ptr[sl - 1] != '/') sl--;
            buf_add(&rel, raw.ptr, sl);
            buf_adds(&rel, name);
        } else if (raw.len && raw.ptr[0] == '~') {
            buf_adds(&rel, "~/");
            buf_adds(&rel, name);
        } else buf_adds(&rel, name);
        if (ents.ptr[i].dir) buf_addc(&rel, '/');
        completion_value(&val, buf_str(&rel), at, quoted);
        buf_adds(&label, name);
        if (ents.ptr[i].dir) buf_addc(&label, '/');
        sel_add(list, val.ptr, label.ptr, NULL);
        buf_free(&rel);
        buf_free(&val);
        buf_free(&label);
        free(ents.ptr[i].name);
    }
    vec_free(&ents);
    buf_free(&dir);
    buf_free(&exp);
}

// getSuggestions: fills list and returns the prefix, or NULL for none.
static char *suggestions(Editor *e, bool force, SelectList *list) {
    Str line = line_at(e, e->cl);
    Str before = str_n(line.ptr, e->cc);
    Str ap;
    if (at_prefix(before, &ap)) {
        Str raw = ap;
        bool quoted = false;
        if (str_starts(ap, S("@\""))) {
            raw = str_sub(ap, 2, ap.len);
            quoted = true;
        } else raw = str_sub(ap, 1, ap.len);
        fuzzy_files(e->cwd, e->fd, NULL, raw, quoted, list);
        if (!list->n) return NULL;
        return strndup(ap.ptr, ap.len);
    }
    Str cmd = before;
    while (cmd.len && (cmd.ptr[0] == ' ' || cmd.ptr[0] == '\t')) cmd = str_sub(cmd, 1, cmd.len);
    if (!force && cmd.len && cmd.ptr[0] == '/') {
        ptrdiff_t sp = str_find_char(cmd, ' ', 0);
        if (sp < 0) {
            Str q = str_sub(cmd, 1, cmd.len);
            const char **names = xmalloc((e->ncmds + 1) * sizeof *names);
            for (size_t i = 0; i < e->ncmds; i++) names[i] = e->cmds[i].name;
            size_t *idx = xmalloc((e->ncmds + 1) * sizeof *idx);
            size_t m = fuzzy_filter(names, e->ncmds, q, idx);
            for (size_t k = 0; k < m; k++) {
                const SlashCmd *c = &e->cmds[idx[k]];
                Buf d = {0};
                if (c->hint && c->description && *c->description) buf_printf(&d, "%s \xe2\x80\x94 %s", c->hint, c->description);
                else if (c->hint) buf_adds(&d, c->hint);
                else if (c->description) buf_adds(&d, c->description);
                sel_add(list, c->name, c->name, d.ptr);
                buf_free(&d);
            }
            free(names);
            free(idx);
            if (!list->n) return NULL;
            return strndup(cmd.ptr, cmd.len);
        }
        // the command's argument completions
        Str name = str_sub(cmd, 1, (size_t)sp);
        Str arg = str_sub(cmd, (size_t)sp + 1, cmd.len);
        if (!e->arg_fn) return NULL;
        char *cname = strndup(name.ptr, name.len);
        SelectList all;
        sel_init(&all, 5, 32, 32);
        Lines search = {0};
        bool has = e->arg_fn(e->arg_ctx, cname, &all, &search);
        free(cname);
        if (has && all.n) {
            size_t *idx = xmalloc((all.n + 1) * sizeof *idx);
            size_t m = fuzzy_filter((const char *const *)search.v, all.n, arg, idx);
            for (size_t k = 0; k < m; k++) sel_add(list, all.items[idx[k]].value, all.items[idx[k]].label, all.items[idx[k]].description);
            free(idx);
        }
        sel_free(&all);
        lines_free(&search);
        if (!list->n) return NULL;
        return strndup(arg.ptr, arg.len);
    }
    Str pp;
    if (!path_prefix(before, force, &pp)) return NULL;
    file_suggestions(e, pp, list);
    if (!list->n) return NULL;
    return strndup(pp.ptr, pp.len);
}

static void ac_show(Editor *e, SelectList list, char *prefix, bool force);

// The "@" completion (pi's async provider): after ATTACHMENT_AUTOCOMPLETE_DEBOUNCE_MS (20 ms; at
// once when forced), fd runs on its own thread; a newer request cancels it (its fd is killed).
static bool request_at(Editor *e, bool force) {
    Str before = before_cursor(e);
    Str ap;
    if (!at_prefix(before, &ap)) return false;
    ac_cancel(e);
    e->req_id++;
    e->req_pending = true;
    e->req_force = force;
    e->debounce_ns = now_ns() + (force ? 0 : 20000000ull);
    return true;
}

static void request_ac(Editor *e, bool force) {
    if (request_at(e, force)) return;
    ac_cancel(e);
    SelectList list;
    bool slash = false;
    {
        Str line = line_at(e, e->cl);
        Str before = str_n(line.ptr, e->cc);
        Str cmd = before;
        while (cmd.len && (cmd.ptr[0] == ' ' || cmd.ptr[0] == '\t')) cmd = str_sub(cmd, 1, cmd.len);
        Str ap;
        slash = !at_prefix(before, &ap) && cmd.len && cmd.ptr[0] == '/' && !memchr(cmd.ptr, ' ', cmd.len);
    }
    sel_init(&list, 5, slash ? 12 : 32, 32);
    char *prefix = suggestions(e, force, &list);
    ac_show(e, list, prefix, force);
}

static void ac_show(Editor *e, SelectList list, char *prefix, bool force) {
    ac_close(e);
    if (!prefix) {
        sel_free(&list);
        return;
    }
    // the best match: an exact value, else the first that starts with the prefix
    long best = -1;
    for (size_t i = 0; i < list.n; i++) {
        if (!strcmp(list.items[i].value, prefix)) {
            best = (long)i;
            break;
        }
        if (best < 0 && str_starts(str(list.items[i].value), str(prefix))) best = (long)i;
    }
    if (best >= 0) list.selected = (size_t)best;
    e->ac = list;
    e->ac_open = true;
    e->ac_force = force;
    e->ac_prefix = prefix;
}

// applyCompletion
static void apply_completion(Editor *e, const SelItem *it, const char *prefix_z) {
    push_undo(e);
    e->last_action = LA_NONE;
    Str prefix = str(prefix_z);
    Buf *line = cur_line(e);
    Str l = buf_str(line);
    size_t bp = e->cc >= prefix.len ? e->cc - prefix.len : 0;
    Str before = str_n(l.ptr, bp), after = str_sub(l, e->cc, l.len);
    bool qp = str_starts(prefix, S("\"")) || str_starts(prefix, S("@\""));
    Str value = str(it->value);
    if (qp && str_ends(value, S("\"")) && str_starts(after, S("\""))) after = str_sub(after, 1, after.len);
    Buf nl = {0};
    size_t cc;
    bool slash = str_starts(prefix, S("/")) && !str_trim(before).len && !memchr(prefix.ptr + 1, '/', prefix.len - 1);
    bool dir = str_ends(str(it->label), S("/"));
    bool tq = str_ends(value, S("\""));
    if (slash) {
        buf_addstr(&nl, before);
        buf_addc(&nl, '/');
        buf_addstr(&nl, value);
        buf_addc(&nl, ' ');
        buf_addstr(&nl, after);
        cc = before.len + value.len + 2;
    } else if (str_starts(prefix, S("@"))) {
        const char *suffix = dir ? "" : " ";
        buf_addstr(&nl, before);
        buf_addstr(&nl, value);
        buf_adds(&nl, suffix);
        buf_addstr(&nl, after);
        cc = before.len + (dir && tq ? value.len - 1 : value.len) + strlen(suffix);
    } else {
        buf_addstr(&nl, before);
        buf_addstr(&nl, value);
        buf_addstr(&nl, after);
        cc = before.len + (dir && tq ? value.len - 1 : value.len);
    }
    buf_free(line);
    *line = nl;
    if (!line->ptr) buf_adds(line, "");
    e->cc = cc;
}

// ---- editing ------------------------------------------------------------------------------------------

static bool slash_allowed(const Editor *e) { return e->cl == 0; }

static bool at_start_of_message(const Editor *e) {
    if (!slash_allowed(e)) return false;
    Str b = str_trim(str_n(e->lines.ptr[e->cl].ptr, e->cc));
    return !b.len || str_eqz(b, "/");
}

static bool slash_context(const Editor *e, Str before) {
    while (before.len && tui_is_space((unsigned char)before.ptr[0])) before = str_sub(before, 1, before.len);
    return slash_allowed(e) && before.len && before.ptr[0] == '/';
}

static Str before_cursor(const Editor *e) { return str_n(e->lines.ptr[e->cl].ptr ? e->lines.ptr[e->cl].ptr : "", e->cc); }

static void insert_text_raw(Editor *e, Str text) {
    // splits on newlines
    Buf *line = cur_line(e);
    Str l = buf_str(line);
    Buf after = {0};
    buf_addstr(&after, str_sub(l, e->cc, l.len));
    line->len = e->cc;
    line->ptr[line->len] = 0;
    size_t i = 0;
    for (;;) {
        size_t j = i;
        while (j < text.len && text.ptr[j] != '\n') j++;
        buf_add(cur_line(e), text.ptr + i, j - i);
        if (j >= text.len) break;
        Buf nb = {0};
        buf_adds(&nb, "");
        vec_push(&e->lines, nb);
        memmove(e->lines.ptr + e->cl + 2, e->lines.ptr + e->cl + 1, (e->lines.len - e->cl - 2) * sizeof(Buf));
        e->lines.ptr[e->cl + 1] = nb;
        e->cl++;
        i = j + 1;
    }
    e->cc = cur_line(e)->len;
    buf_addstr(cur_line(e), buf_str(&after));
    buf_free(&after);
}

static void insert_char(Editor *e, uint32_t cp) {
    e->hist_idx = -1;
    // undo coalescing (fish-style): a run of word characters is one unit; a space keeps the
    // state before it
    if (tui_is_space(cp) || e->last_action != LA_TYPE_WORD) push_undo(e);
    e->last_action = LA_TYPE_WORD;
    char enc[5];
    int n = cp < 0x80 ? 1 : cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4;
    if (n == 1) enc[0] = (char)cp;
    else if (n == 2) {
        enc[0] = (char)(0xc0 | cp >> 6);
        enc[1] = (char)(0x80 | (cp & 0x3f));
    } else if (n == 3) {
        enc[0] = (char)(0xe0 | cp >> 12);
        enc[1] = (char)(0x80 | ((cp >> 6) & 0x3f));
        enc[2] = (char)(0x80 | (cp & 0x3f));
    } else {
        enc[0] = (char)(0xf0 | cp >> 18);
        enc[1] = (char)(0x80 | ((cp >> 12) & 0x3f));
        enc[2] = (char)(0x80 | ((cp >> 6) & 0x3f));
        enc[3] = (char)(0x80 | (cp & 0x3f));
    }
    insert_text_raw(e, str_n(enc, (size_t)n));
    if (!e->ac_open) {
        Str before = before_cursor(e);
        if (cp == '/' && at_start_of_message(e)) request_ac(e, false);
        else if (cp == '@' || cp == '#') {
            if (trigger_match(before)) request_ac(e, false);
        } else if ((cp < 0x80 && (isalnum((int)cp) || cp == '.' || cp == '-' || cp == '_')) || tui_cp_width(cp) == 2) {
            if (slash_context(e, before)) request_ac(e, false);
            else if (trigger_match(before)) request_ac(e, false);
        }
    } else request_ac(e, e->ac_force);
}

static size_t prev_grapheme(Str l, size_t at) {
    size_t i = 0, last = 0;
    while (i < at) {
        last = i;
        i += tui_grapheme(l, i);
    }
    return last;
}

static void after_delete(Editor *e) {
    if (e->ac_open) request_ac(e, e->ac_force);
    else {
        Str before = before_cursor(e);
        if (slash_context(e, before)) request_ac(e, false);
        else if (trigger_match(before)) request_ac(e, false);
    }
}

static void backspace(Editor *e) {
    e->hist_idx = -1;
    e->last_action = LA_NONE;
    if (e->cc > 0 || e->cl > 0) push_undo(e);
    if (e->cc > 0) {
        Buf *line = cur_line(e);
        size_t p = prev_grapheme(buf_str(line), e->cc);
        memmove(line->ptr + p, line->ptr + e->cc, line->len - e->cc + 1);
        line->len -= e->cc - p;
        e->cc = p;
    } else if (e->cl > 0) {
        Buf *prev = &e->lines.ptr[e->cl - 1];
        size_t pl = prev->len;
        buf_addstr(prev, buf_str(cur_line(e)));
        buf_free(cur_line(e));
        memmove(e->lines.ptr + e->cl, e->lines.ptr + e->cl + 1, (e->lines.len - e->cl - 1) * sizeof(Buf));
        e->lines.len--;
        e->cl--;
        e->cc = pl;
    }
    after_delete(e);
}

static void forward_delete(Editor *e) {
    e->hist_idx = -1;
    e->last_action = LA_NONE;
    Buf *line = cur_line(e);
    if (e->cc < line->len || e->cl + 1 < e->lines.len) push_undo(e);
    if (e->cc < line->len) {
        size_t g = tui_grapheme(buf_str(line), e->cc);
        memmove(line->ptr + e->cc, line->ptr + e->cc + g, line->len - e->cc - g + 1);
        line->len -= g;
    } else if (e->cl + 1 < e->lines.len) {
        buf_addstr(line, buf_str(&e->lines.ptr[e->cl + 1]));
        buf_free(&e->lines.ptr[e->cl + 1]);
        memmove(e->lines.ptr + e->cl + 1, e->lines.ptr + e->cl + 2, (e->lines.len - e->cl - 2) * sizeof(Buf));
        e->lines.len--;
    }
    after_delete(e);
}

static void new_line(Editor *e) {
    ac_close(e);
    e->hist_idx = -1;
    e->last_action = LA_NONE;
    push_undo(e);
    insert_text_raw(e, S("\n"));
}

static bool word_char(uint32_t cp) { return !tui_is_space(cp) && !(cp < 0x80 && strchr("(){}[]<>.,;:'\"!?+-=*/\\|&%^$#@~`", (int)cp)); }

static void word_left(Editor *e) {
    e->last_action = LA_NONE;
    if (e->cc == 0) {
        if (e->cl > 0) {
            e->cl--;
            e->cc = cur_line(e)->len;
        }
        return;
    }
    Str l = buf_str(cur_line(e));
    size_t p = e->cc;
    // skip whitespace, then a run of word or of punctuation characters
    while (p > 0) {
        size_t q = prev_grapheme(l, p), k = q;
        if (!tui_is_space(tui_cp(l, &k))) break;
        p = q;
    }
    if (p > 0) {
        size_t q = prev_grapheme(l, p), k = q;
        bool w = word_char(tui_cp(l, &k));
        while (p > 0) {
            q = prev_grapheme(l, p);
            k = q;
            uint32_t cp = tui_cp(l, &k);
            if (tui_is_space(cp) || word_char(cp) != w) break;
            p = q;
        }
    }
    e->cc = p;
}

static void word_right(Editor *e) {
    e->last_action = LA_NONE;
    Str l = buf_str(cur_line(e));
    if (e->cc >= l.len) {
        if (e->cl + 1 < e->lines.len) {
            e->cl++;
            e->cc = 0;
        }
        return;
    }
    size_t p = e->cc;
    while (p < l.len) {
        size_t k = p;
        if (!tui_is_space(tui_cp(l, &k))) break;
        p = k;
    }
    if (p < l.len) {
        size_t k = p;
        bool w = word_char(tui_cp(l, &k));
        while (p < l.len) {
            k = p;
            uint32_t cp = tui_cp(l, &k);
            if (tui_is_space(cp) || word_char(cp) != w) break;
            p = k;
        }
    }
    e->cc = p;
}

// The current line joined to the one above (the cursor at the join) or the one below.
static void merge_up(Editor *e) {
    Buf *prev = &e->lines.ptr[e->cl - 1];
    size_t pl = prev->len;
    buf_addstr(prev, buf_str(cur_line(e)));
    buf_free(cur_line(e));
    memmove(e->lines.ptr + e->cl, e->lines.ptr + e->cl + 1, (e->lines.len - e->cl - 1) * sizeof(Buf));
    e->lines.len--;
    e->cl--;
    e->cc = pl;
}

static void merge_down(Editor *e) {
    Buf *line = cur_line(e);
    buf_addstr(line, buf_str(&e->lines.ptr[e->cl + 1]));
    buf_free(&e->lines.ptr[e->cl + 1]);
    memmove(e->lines.ptr + e->cl + 1, e->lines.ptr + e->cl + 2, (e->lines.len - e->cl - 2) * sizeof(Buf));
    e->lines.len--;
}

// deleteWordBackwards: the word before the cursor (or, at a line start, the line break) killed
static void delete_word_back(Editor *e) {
    e->hist_idx = -1;
    if (e->cc == 0) {
        if (e->cl > 0) {
            push_undo(e);
            kill_ring_push(&e->kill, S("\n"), true, e->last_action == LA_KILL);
            e->last_action = LA_KILL;
            merge_up(e);
        }
        after_delete(e);
        return;
    }
    push_undo(e);
    bool was_kill = e->last_action == LA_KILL; // word_left resets it
    size_t end = e->cc;
    word_left(e);
    Buf *line = cur_line(e);
    kill_ring_push(&e->kill, str_n(line->ptr + e->cc, end - e->cc), true, was_kill);
    e->last_action = LA_KILL;
    memmove(line->ptr + e->cc, line->ptr + end, line->len - end + 1);
    line->len -= end - e->cc;
    after_delete(e);
}

// deleteWordForward
static void delete_word_forward(Editor *e) {
    e->hist_idx = -1;
    Buf *line = cur_line(e);
    if (e->cc >= line->len) {
        if (e->cl + 1 < e->lines.len) {
            push_undo(e);
            kill_ring_push(&e->kill, S("\n"), false, e->last_action == LA_KILL);
            e->last_action = LA_KILL;
            merge_down(e);
        }
        return;
    }
    push_undo(e);
    bool was_kill = e->last_action == LA_KILL; // word_right resets it
    size_t start = e->cc;
    word_right(e);
    line = cur_line(e);
    kill_ring_push(&e->kill, str_n(line->ptr + start, e->cc - start), false, was_kill);
    e->last_action = LA_KILL;
    memmove(line->ptr + start, line->ptr + e->cc, line->len - e->cc + 1);
    line->len -= e->cc - start;
    e->cc = start;
}

// deleteToEndOfLine: the rest of the line (or, at its end, the line break) killed
static void delete_to_end(Editor *e) {
    e->hist_idx = -1;
    Buf *line = cur_line(e);
    if (e->cc < line->len) {
        push_undo(e);
        kill_ring_push(&e->kill, str_n(line->ptr + e->cc, line->len - e->cc), false, e->last_action == LA_KILL);
        e->last_action = LA_KILL;
        line->len = e->cc;
        line->ptr[line->len] = 0;
    } else if (e->cl + 1 < e->lines.len) {
        push_undo(e);
        kill_ring_push(&e->kill, S("\n"), false, e->last_action == LA_KILL);
        e->last_action = LA_KILL;
        merge_down(e);
    }
}

// deleteToStartOfLine
static void delete_to_start(Editor *e) {
    e->hist_idx = -1;
    Buf *line = cur_line(e);
    if (e->cc > 0) {
        push_undo(e);
        kill_ring_push(&e->kill, str_n(line->ptr, e->cc), true, e->last_action == LA_KILL);
        e->last_action = LA_KILL;
        memmove(line->ptr, line->ptr + e->cc, line->len - e->cc + 1);
        line->len -= e->cc;
        e->cc = 0;
    } else if (e->cl > 0) {
        push_undo(e);
        kill_ring_push(&e->kill, S("\n"), true, e->last_action == LA_KILL);
        e->last_action = LA_KILL;
        merge_up(e);
    }
    after_delete(e);
}

// pushUndoSnapshot
static void push_undo(Editor *e) { undo_push(&e->undo, editor_text(e), e->cl, e->cc, e->pastes.ptr, e->pastes.len); }

// undo: the last snapshot restored
static void undo(Editor *e) {
    e->hist_idx = -1;
    free(e->draft);
    e->draft = NULL;
    UndoSnap u;
    if (!undo_pop(&e->undo, &u)) return;
    lines_reset(e, str(u.text));
    e->cl = u.cl < e->lines.len ? u.cl : e->lines.len - 1;
    e->cc = u.cc <= cur_line(e)->len ? u.cc : cur_line(e)->len;
    for (size_t i = 0; i < e->pastes.len; i++) free(e->pastes.ptr[i]);
    e->pastes.len = 0;
    for (size_t i = 0; i < u.npastes; i++) vec_push(&e->pastes, strdup(u.pastes[i]));
    undo_snap_free(&u);
    e->last_action = LA_NONE;
}

// insertYankedText: text at the cursor, the cursor after it
static void insert_yanked(Editor *e, const char *text) {
    e->hist_idx = -1;
    insert_text_raw(e, str(text));
}

// deleteYankedText: the yanked text (the ring's last entry) before the cursor removed
static void delete_yanked(Editor *e) {
    const char *y = kill_ring_peek(&e->kill);
    if (!y) return;
    size_t nl = 0;
    for (const char *p = y; *p; p++) nl += *p == '\n';
    if (!nl) {
        size_t n = strlen(y);
        Buf *line = cur_line(e);
        size_t from = e->cc >= n ? e->cc - n : 0;
        memmove(line->ptr + from, line->ptr + e->cc, line->len - e->cc + 1);
        line->len -= e->cc - from;
        e->cc = from;
        return;
    }
    // multi-line: the cursor is at the end of the last yanked line
    size_t first_len = (size_t)(strchr(y, '\n') - y);
    size_t start = e->cl - nl;
    Buf *sl = &e->lines.ptr[start];
    size_t start_col = sl->len >= first_len ? sl->len - first_len : 0;
    Buf merged = {0};
    buf_add(&merged, sl->ptr, start_col);
    Buf *cur = cur_line(e);
    buf_add(&merged, cur->ptr + e->cc, cur->len - e->cc);
    if (!merged.ptr) buf_adds(&merged, "");
    for (size_t i = start; i <= e->cl; i++) buf_free(&e->lines.ptr[i]);
    memmove(e->lines.ptr + start + 1, e->lines.ptr + e->cl + 1, (e->lines.len - e->cl - 1) * sizeof(Buf));
    e->lines.len -= nl;
    e->lines.ptr[start] = merged;
    e->cl = start;
    e->cc = start_col;
}

// yank: the most recent kill at the cursor
static void yank(Editor *e) {
    const char *t = kill_ring_peek(&e->kill);
    if (!t) return;
    push_undo(e);
    insert_yanked(e, t);
    e->last_action = LA_YANK;
}

// yankPop: right after a yank, the yanked text replaced with the entry before it in the ring
static void yank_pop(Editor *e) {
    if (e->last_action != LA_YANK || e->kill.n <= 1) return;
    push_undo(e);
    delete_yanked(e);
    kill_ring_rotate(&e->kill);
    insert_yanked(e, kill_ring_peek(&e->kill));
    e->last_action = LA_YANK;
}

// jumpToChar: the first occurrence of the character after the cursor (forward) or before it,
// across lines; the cursor stays when there is none
static void jump_to_char(Editor *e, uint32_t cp, int dir) {
    e->last_action = LA_NONE;
    char enc[5];
    int n = tui_utf8(enc, cp);
    for (long li = (long)e->cl; li >= 0 && li < (long)e->lines.len; li += dir) {
        Str l = line_at(e, (size_t)li);
        bool cur = (size_t)li == e->cl;
        long found = -1;
        if (dir > 0) {
            size_t from = cur ? e->cc + 1 : 0;
            for (size_t i = from; i + (size_t)n <= l.len && found < 0; i++)
                if (!memcmp(l.ptr + i, enc, (size_t)n)) found = (long)i;
        } else {
            // lastIndexOf(char, cursorCol - 1): a negative start searches from 0
            long from = cur ? (e->cc ? (long)e->cc - 1 : 0) : (long)l.len;
            for (long i = from; i >= 0 && found < 0; i--)
                if ((size_t)i + (size_t)n <= l.len && !memcmp(l.ptr + i, enc, (size_t)n)) found = i;
        }
        if (found >= 0) {
            e->cl = (size_t)li;
            e->cc = (size_t)found;
            return;
        }
    }
}

void editor_insert_text(Editor *e, Str text) {
    if (!text.len) return;
    ac_close(e);
    push_undo(e);
    e->last_action = LA_NONE;
    e->hist_idx = -1;
    // normalizeText: line endings as \n, tabs as four spaces
    Buf t = {0};
    for (size_t i = 0; i < text.len; i++) {
        if (text.ptr[i] == '\r') {
            buf_addc(&t, '\n');
            if (i + 1 < text.len && text.ptr[i + 1] == '\n') i++;
        } else if (text.ptr[i] == '\t') buf_adds(&t, "    ");
        else buf_addc(&t, text.ptr[i]);
    }
    insert_text_raw(e, buf_str(&t));
    buf_free(&t);
}

bool editor_cursor(const Editor *e, size_t *line, size_t *col) {
    *line = e->cl;
    *col = e->cc;
    return true;
}

// The visual lines (wrapped chunks) and the one holding the cursor.
static size_t visual_lines(Editor *e, LayoutVec *ll) {
    // every line laid out with cursor flags relative to the current line
    layout(e, e->last_width, ll);
    for (size_t i = 0; i < ll->len; i++)
        if (ll->ptr[i].has_cursor) return i;
    return 0;
}

// moveToVisualLine: the cursor dir visual lines away (clamped to the text), at its column.
static void move_vertical(Editor *e, long dir) {
    LayoutVec ll = {0};
    size_t cur = visual_lines(e, &ll);
    long target = (long)cur + dir;
    if (target < 0) target = 0;
    if (target >= (long)ll.len) target = (long)ll.len - 1;
    if (target == (long)cur) {
        vec_free(&ll);
        return;
    }
    dir = target - (long)cur;
    LayoutLine *from = &ll.ptr[cur];
    Str fl = line_at(e, from->line);
    int col = tui_width(str_n(fl.ptr + from->start, e->cc - from->start));
    LayoutLine *to = &ll.ptr[(size_t)((long)cur + dir)];
    Str tl = line_at(e, to->line);
    size_t p = to->start;
    int w = 0;
    bool last_chunk = (size_t)((long)cur + dir) + 1 == ll.len || ll.ptr[(size_t)((long)cur + dir) + 1].line != to->line;
    size_t limit = last_chunk ? to->end : to->end;
    while (p < limit) {
        size_t g = tui_grapheme(tl, p);
        int gw = tui_width(str_n(tl.ptr + p, g));
        if (w + gw > col) break;
        w += gw;
        p += g;
    }
    if (!last_chunk && p >= to->end && to->end > to->start) p = prev_grapheme(tl, to->end);
    e->cl = to->line;
    e->cc = p;
    vec_free(&ll);
}

static bool on_first_visual(Editor *e) {
    LayoutVec ll = {0};
    size_t cur = visual_lines(e, &ll);
    vec_free(&ll);
    return cur == 0;
}

static bool on_last_visual(Editor *e) {
    LayoutVec ll = {0};
    size_t cur = visual_lines(e, &ll);
    bool last = cur + 1 >= ll.len;
    vec_free(&ll);
    return last;
}

static void history(Editor *e, int dir) {
    e->last_action = LA_NONE;
    if (!e->hist.len) return;
    long ni = e->hist_idx - dir;
    if (ni < -1 || ni >= (long)e->hist.len) return;
    if (e->hist_idx == -1 && ni >= 0) {
        push_undo(e);
        free(e->draft);
        Str t = editor_text(e);
        e->draft = strndup(t.ptr, t.len);
    }
    e->hist_idx = ni;
    if (ni == -1) {
        lines_reset(e, str(e->draft ? e->draft : ""));
        free(e->draft);
        e->draft = NULL;
        e->cl = e->lines.len - 1;
        e->cc = cur_line(e)->len;
    } else {
        lines_reset(e, str(e->hist.ptr[ni]));
        if (dir < 0) {
            e->cl = 0;
            e->cc = 0;
        } else {
            e->cl = e->lines.len - 1;
            e->cc = cur_line(e)->len;
        }
    }
    e->scroll = 0;
}

// The text with the paste markers expanded, trimmed.
static char *submit_text(Editor *e) {
    Str t = editor_text(e);
    Buf out = {0};
    for (size_t i = 0; i < t.len;) {
        if (t.ptr[i] == '[' && str_starts(str_sub(t, i, t.len), S("[paste #"))) {
            size_t j = i + 8;
            size_t id = 0;
            while (j < t.len && t.ptr[j] >= '0' && t.ptr[j] <= '9') id = id * 10 + (size_t)(t.ptr[j++] - '0');
            size_t close = j;
            while (close < t.len && t.ptr[close] != ']') close++;
            if (close < t.len && id >= 1 && id <= e->pastes.len) {
                buf_adds(&out, e->pastes.ptr[id - 1]);
                i = close + 1;
                continue;
            }
        }
        buf_addc(&out, t.ptr[i++]);
    }
    Str tr = str_trim(buf_str(&out));
    char *r = strndup(tr.ptr ? tr.ptr : "", tr.len);
    buf_free(&out);
    return r;
}

static void submit(Editor *e, char **out) {
    ac_close(e);
    undo_clear(&e->undo);
    e->last_action = LA_NONE;
    *out = submit_text(e);
    lines_reset(e, S(""));
    e->cl = e->cc = 0;
    for (size_t i = 0; i < e->pastes.len; i++) free(e->pastes.ptr[i]);
    e->pastes.len = 0;
    e->hist_idx = -1;
    e->scroll = 0;
}

static void paste(Editor *e, const char *text) {
    ac_close(e);
    e->hist_idx = -1;
    e->last_action = LA_NONE;
    push_undo(e);
    Buf clean = {0};
    for (const char *p = text; *p; p++) {
        if (*p == '\r') {
            buf_addc(&clean, '\n');
            if (p[1] == '\n') p++;
        } else if (*p == '\t') buf_adds(&clean, "    ");
        else if (*p == '\n' || (unsigned char)*p >= 32) buf_addc(&clean, *p);
    }
    if (!clean.ptr) buf_adds(&clean, "");
    if (clean.len && strchr("/~.", clean.ptr[0]) && e->cc > 0) {
        char c = cur_line(e)->ptr[e->cc - 1];
        if (isalnum((unsigned char)c) || c == '_') {
            Buf s = {0};
            buf_addc(&s, ' ');
            buf_addstr(&s, buf_str(&clean));
            buf_free(&clean);
            clean = s;
        }
    }
    size_t nl = 1;
    for (size_t i = 0; i < clean.len; i++) nl += clean.ptr[i] == '\n';
    size_t chars = tui_js_len(buf_str(&clean));
    if (nl > 10 || chars > 1000) {
        vec_push(&e->pastes, strndup(clean.ptr, clean.len));
        char marker[96];
        if (nl > 10) snprintf(marker, sizeof marker, "[paste #%zu +%zu lines]", e->pastes.len, nl);
        else snprintf(marker, sizeof marker, "[paste #%zu %zu chars]", e->pastes.len, chars);
        insert_text_raw(e, str(marker));
    } else insert_text_raw(e, buf_str(&clean));
    buf_free(&clean);
}

bool editor_key(Editor *e, const Key *k, char **submitted) {
    *submitted = NULL;
    bool jump_fwd = k->kind == K_CTRL && k->ch == ']', jump_back = k->kind == K_ALT && k->ch == 0x1d;
    // the character jump waiting for its character: the hotkey again cancels it, a printable
    // character jumps, anything else cancels and is handled as usual
    if (e->jump) {
        if (jump_fwd || jump_back) {
            e->jump = 0;
            return false;
        }
        if (k->kind == K_CHAR) {
            int dir = e->jump;
            e->jump = 0;
            jump_to_char(e, k->ch, dir);
            return false;
        }
        e->jump = 0;
    }
    if (k->kind == K_PASTE) {
        paste(e, k->paste);
        return false;
    }
    if (k->kind == K_CTRL && k->ch == 'c') return false; // the app clears
    if (k->kind == K_CTRL && (k->ch == '-' || k->ch == '_')) { // tui.editor.undo (ctrl+-, legacy ctrl+_)
        undo(e);
        return false;
    }
    if (e->ac_open) {
        if (k->kind == K_ESC || (k->kind == K_CTRL && k->ch == 'c')) {
            ac_close(e);
            return false;
        }
        if (k->kind == K_UP) {
            sel_up(&e->ac);
            return false;
        }
        if (k->kind == K_DOWN) {
            sel_down(&e->ac);
            return false;
        }
        if (k->kind == K_TAB || k->kind == K_ENTER) {
            const SelItem *it = sel_current(&e->ac);
            if (it) {
                char *prefix = strdup(e->ac_prefix);
                SelItem copy = {strdup(it->value), strdup(it->label), NULL};
                apply_completion(e, &copy, prefix);
                free(copy.value);
                free(copy.label);
                bool slash = prefix[0] == '/';
                free(prefix);
                ac_close(e);
                if (k->kind == K_TAB || !slash) return false;
                // Enter on a "/" command: submit it
                submit(e, submitted);
                return true;
            }
        }
    }
    switch (k->kind) {
    case K_TAB: {
        Str before = before_cursor(e);
        Str t = before;
        while (t.len && tui_is_space((unsigned char)t.ptr[0])) t = str_sub(t, 1, t.len);
        if (slash_context(e, before) && !memchr(t.ptr, ' ', t.len)) request_ac(e, false);
        else {
            // shouldTriggerFileCompletion
            Str tt = str_trim(before);
            if (tt.len && tt.ptr[0] == '/' && !memchr(tt.ptr, ' ', tt.len)) return false;
            request_ac(e, true);
            if (e->ac_open && e->ac.n == 1) { // a single match is applied at once
                char *prefix = strdup(e->ac_prefix);
                SelItem copy = {strdup(e->ac.items[0].value), strdup(e->ac.items[0].label), NULL};
                ac_close(e);
                apply_completion(e, &copy, prefix);
                free(copy.value);
                free(copy.label);
                free(prefix);
            }
        }
        return false;
    }
    case K_CTRL:
        if (k->ch != 'k' && k->ch != 'u' && k->ch != 'w' && k->ch != 'd' && k->ch != 'y' && k->ch != ']') e->last_action = LA_NONE;
        switch (k->ch) {
        case 'k': delete_to_end(e); return false;
        case 'u': delete_to_start(e); return false;
        case 'w': delete_word_back(e); return false;
        case 'y': yank(e); return false;
        case ']': e->jump = 1; return false;
        case 'a': e->cc = 0; return false;
        case 'e': e->cc = cur_line(e)->len; return false;
        case 'b':
            if (e->cc > 0) e->cc = prev_grapheme(buf_str(cur_line(e)), e->cc);
            else if (e->cl > 0) {
                e->cl--;
                e->cc = cur_line(e)->len;
            }
            return false;
        case 'f':
            if (e->cc < cur_line(e)->len) e->cc += tui_grapheme(buf_str(cur_line(e)), e->cc);
            else if (e->cl + 1 < e->lines.len) {
                e->cl++;
                e->cc = 0;
            }
            return false;
        case 'd': forward_delete(e); return false;
        case 'j': new_line(e); return false;
        default: return false;
        }
    case K_ALT:
        if (k->ch == 'd') delete_word_forward(e);
        else if (k->ch == 'y') yank_pop(e);
        else if (k->ch == 0x1d) e->jump = -1;
        return false;
    case K_ALT_BACKSPACE:
    case K_CTRL_BACKSPACE: delete_word_back(e); return false;
    case K_BACKSPACE: backspace(e); return false;
    case K_DELETE: forward_delete(e); return false;
    case K_ALT_DELETE: delete_word_forward(e); return false;
    case K_CTRL_PGUP:
    case K_CTRL_PGDN: { // pageScroll: max(5, 30% of the terminal) visual lines
        e->last_action = LA_NONE;
        long page = (long)floor(e->rows * 0.3) > 5 ? (long)floor(e->rows * 0.3) : 5;
        move_vertical(e, k->kind == K_CTRL_PGUP ? -page : page);
        return false;
    }
    case K_HOME: e->last_action = LA_NONE; e->cc = 0; return false;
    case K_END: e->last_action = LA_NONE; e->cc = cur_line(e)->len; return false;
    case K_ALT_LEFT:
    case K_CTRL_LEFT: word_left(e); return false;
    case K_ALT_RIGHT:
    case K_CTRL_RIGHT: word_right(e); return false;
    case K_SHIFT_ENTER:
    case K_ALT_ENTER: new_line(e); return false;
    case K_ENTER: {
        Buf *line = cur_line(e);
        if (e->cc > 0 && line->ptr[e->cc - 1] == '\\') {
            backspace(e);
            new_line(e);
            return false;
        }
        submit(e, submitted);
        return true;
    }
    case K_UP:
        e->last_action = LA_NONE;
        if (on_first_visual(e) && (editor_empty(e) || e->hist_idx > -1 || e->cc == 0)) history(e, -1);
        else if (on_first_visual(e)) e->cc = 0;
        else move_vertical(e, -1);
        return false;
    case K_DOWN:
        e->last_action = LA_NONE;
        if (e->hist_idx > -1 && on_last_visual(e)) history(e, 1);
        else if (on_last_visual(e)) e->cc = cur_line(e)->len;
        else move_vertical(e, 1);
        return false;
    case K_RIGHT:
        e->last_action = LA_NONE;
        if (e->cc < cur_line(e)->len) e->cc += tui_grapheme(buf_str(cur_line(e)), e->cc);
        else if (e->cl + 1 < e->lines.len) {
            e->cl++;
            e->cc = 0;
        }
        return false;
    case K_LEFT:
        e->last_action = LA_NONE;
        if (e->cc > 0) e->cc = prev_grapheme(buf_str(cur_line(e)), e->cc);
        else if (e->cl > 0) {
            e->cl--;
            e->cc = cur_line(e)->len;
        }
        return false;
    case K_CHAR: insert_char(e, k->ch); return false;
    default: return false;
    }
}

static void *ac_worker(void *arg) {
    AcJob *j = arg;
    sel_init(&j->list, 5, 32, 32);
    Str ap = str(j->prefix), raw;
    if (str_starts(ap, S("@\""))) raw = str_sub(ap, 2, ap.len);
    else raw = str_sub(ap, 1, ap.len);
    fuzzy_files(j->cwd, j->fd, j, raw, j->quoted, &j->list);
    atomic_store(&j->done, true);
    if (j->waker >= 0) (void)!write(j->waker, "a", 1);
    return NULL;
}

static void job_free(AcJob *j) {
    pthread_join(j->th, NULL);
    sel_free(&j->list);
    free(j->cwd);
    free(j->fd);
    free(j->prefix);
    free(j->snap_text);
    free(j);
}

// Stops the lookups in flight (their results will be dropped) and the debounce.
static void ac_cancel(Editor *e) {
    e->req_pending = false;
    for (AcJob *j = e->jobs; j; j = j->next) {
        atomic_store(&j->cancel, true);
        int pid = atomic_load(&j->pid);
        if (pid > 0) kill(pid, SIGKILL);
    }
}

bool editor_busy(const Editor *e) { return e->req_pending || e->jobs; }

void editor_set_waker(Editor *e, int fd) { e->waker = fd; }

bool editor_tick(Editor *e) {
    bool changed = false;
    if (e->req_pending && now_ns() >= e->debounce_ns) {
        e->req_pending = false;
        Str ap;
        if (at_prefix(before_cursor(e), &ap)) {
            AcJob *j = xcalloc(1, sizeof *j);
            j->id = e->req_id;
            j->cwd = dupz(e->cwd);
            j->fd = e->fd ? dupz(e->fd) : NULL;
            j->prefix = strndup(ap.ptr, ap.len);
            j->quoted = str_starts(ap, S("@\""));
            Str t = editor_text(e);
            j->snap_text = strndup(t.ptr, t.len);
            j->snap_line = e->cl;
            j->snap_col = e->cc;
            j->waker = e->waker;
            if (pthread_create(&j->th, NULL, ac_worker, j)) {
                free(j->cwd);
                free(j->fd);
                free(j->prefix);
                free(j->snap_text);
                free(j);
            } else {
                j->next = e->jobs;
                e->jobs = j;
            }
        }
    }
    for (AcJob **pp = &e->jobs; *pp;) {
        AcJob *j = *pp;
        if (!atomic_load(&j->done)) {
            pp = &j->next;
            continue;
        }
        *pp = j->next;
        Str now_text = editor_text(e);
        bool current = !atomic_load(&j->cancel) && j->id == e->req_id && e->cl == j->snap_line && e->cc == j->snap_col &&
                       str_eqz(now_text, j->snap_text);
        if (current) {
            SelectList list = j->list;
            j->list = (SelectList){0};
            char *prefix = list.n ? strdup(j->prefix) : NULL;
            if (e->req_force && list.n == 1) { // Tab with a single match applies it
                ac_close(e);
                apply_completion(e, &list.items[0], prefix);
                sel_free(&list);
                free(prefix);
            } else ac_show(e, list, prefix, e->req_force);
            changed = true;
        }
        job_free(j);
    }
    return changed;
}
