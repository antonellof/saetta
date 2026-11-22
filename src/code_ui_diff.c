// Saetta Code TUI: the word diff that marks the changed words of an edited line (see code_tui.h).
// Adapted from jsdiff (BSD-3-Clause, Copyright (c) 2009-2015, Kevin Decker; version 8.0.4, the
// copy pi 1.5.2 ships): libesm/diff/base.js (Myers' diff with its path components and
// buildValues), libesm/diff/word.js (diffWords: tokens with their surrounding whitespace, equality
// of trimmed tokens, the whitespace dedupe of postProcess), libesm/util/string.js. pi's
// renderIntraLineDiff (MIT, Copyright (c) 2025 Mario Zechner) calls it.
#include "code_tui.h"

#include <stdlib.h>

static bool word_cp(uint32_t c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == 0xad ||
           (c >= 0xc0 && c <= 0xd6) || (c >= 0xd8 && c <= 0xf6) || (c >= 0xf8 && c <= 0x2c6) || (c >= 0x2c8 && c <= 0x2d7) ||
           (c >= 0x2de && c <= 0x2ff) || (c >= 0x1e00 && c <= 0x1eff);
}

typedef VEC(Str) StrList;

// tokenize: runs of word characters, runs of whitespace, single other characters; whitespace is
// stitched onto the neighbouring tokens.
static void tokenize(Arena *a, Str s, StrList *out) {
    StrList parts = {0};
    for (size_t i = 0; i < s.len;) {
        size_t k = i;
        uint32_t c = tui_cp(s, &k);
        size_t j = k;
        if (word_cp(c)) {
            while (j < s.len) {
                size_t q = j;
                if (!word_cp(tui_cp(s, &q))) break;
                j = q;
            }
        } else if (tui_is_space(c)) {
            while (j < s.len) {
                size_t q = j;
                if (!tui_is_space(tui_cp(s, &q))) break;
                j = q;
            }
        }
        vec_push(&parts, str_sub(s, i, j));
        i = j;
    }
    Str prev = {0};
    bool have_prev = false;
    for (size_t i = 0; i < parts.len; i++) {
        Str p = parts.ptr[i];
        size_t k = 0;
        bool ws = tui_is_space(tui_cp(p, &k));
        if (ws) {
            if (!have_prev) vec_push(out, p);
            else {
                Str l = out->ptr[out->len - 1];
                out->ptr[out->len - 1] = str_n(l.ptr, (size_t)(p.ptr + p.len - l.ptr)); // contiguous
            }
        } else if (have_prev) {
            size_t q = 0;
            bool pws = tui_is_space(tui_cp(prev, &q));
            if (pws) {
                Str l = out->ptr[out->len - 1];
                if (l.ptr == prev.ptr && l.len == prev.len) out->ptr[out->len - 1] = str_n(l.ptr, (size_t)(p.ptr + p.len - l.ptr));
                else vec_push(out, str_n(prev.ptr, prev.len + p.len));
            } else vec_push(out, p);
        } else vec_push(out, p);
        prev = p;
        have_prev = true;
    }
    vec_free(&parts);
    (void)a;
}

static Str trim_ws(Str s) {
    size_t a = 0, b = s.len;
    while (a < b) {
        size_t k = a;
        if (!tui_is_space(tui_cp(s, &k))) break;
        a = k;
    }
    while (b > a) {
        size_t k = b - 1;
        while (k > a && ((unsigned char)s.ptr[k] & 0xc0) == 0x80) k--;
        size_t q = k;
        if (!tui_is_space(tui_cp(s, &q))) break;
        b = k;
    }
    return str_sub(s, a, b);
}

static bool tok_eq(Str x, Str y) { return str_eq(trim_ws(x), trim_ws(y)); }

typedef struct Comp_ Comp_;
struct Comp_ {
    size_t count;
    bool added, removed;
    Comp_ *prev;
};

typedef struct {
    long old_pos;
    Comp_ *last;
    bool set;
} Path;

static Path add_to_path(Arena *a, Path p, bool added, bool removed, long inc) {
    Comp_ *last = p.last;
    Comp_ *c = arena_alloc(a, sizeof *c);
    if (last && last->added == added && last->removed == removed) {
        *c = (Comp_){last->count + 1, added, removed, last->prev};
    } else *c = (Comp_){1, added, removed, last};
    return (Path){p.old_pos + inc, c, true};
}

static long extract_common(Arena *a, Path *p, const StrList *nt, const StrList *ot, long diag) {
    long nl = (long)nt->len, ol = (long)ot->len;
    long op = p->old_pos, np = op - diag, cc = 0;
    while (np + 1 < nl && op + 1 < ol && tok_eq(ot->ptr[op + 1], nt->ptr[np + 1])) {
        np++;
        op++;
        cc++;
    }
    if (cc) {
        Comp_ *c = arena_alloc(a, sizeof *c);
        *c = (Comp_){(size_t)cc, false, false, p->last};
        p->last = c;
    }
    p->old_pos = op;
    return np;
}

// join: the first token whole, the others without their leading whitespace.
static void join(Buf *b, const StrList *t, size_t from, size_t n) {
    for (size_t i = 0; i < n; i++) {
        Str s = t->ptr[from + i];
        if (i) {
            size_t k = 0;
            while (k < s.len) {
                size_t q = k;
                if (!tui_is_space(tui_cp(s, &q))) break;
                k = q;
            }
            s = str_sub(s, k, s.len);
        }
        buf_addstr(b, s);
    }
}

typedef struct {
    bool added, removed;
    Buf value;
} Part;
typedef VEC(Part) PartList;

static void build_values(Comp_ *last, const StrList *nt, const StrList *ot, PartList *out) {
    VEC(Comp_ *) cs = {0};
    for (Comp_ *c = last; c; c = c->prev) vec_push(&cs, c);
    size_t np = 0, op = 0;
    for (size_t i = cs.len; i-- > 0;) {
        Comp_ *c = cs.ptr[i];
        Part p = {c->added, c->removed, {0}};
        if (!c->removed) {
            join(&p.value, nt, np, c->count);
            np += c->count;
            if (!c->added) op += c->count;
        } else {
            join(&p.value, ot, op, c->count);
            op += c->count;
        }
        if (!p.value.ptr) buf_adds(&p.value, "");
        vec_push(out, p);
    }
    vec_free(&cs);
}

static Str lead_ws(Str s) {
    size_t k = 0;
    while (k < s.len) {
        size_t q = k;
        if (!tui_is_space(tui_cp(s, &q))) break;
        k = q;
    }
    return str_n(s.ptr, k);
}

static Str trail_ws(Str s) {
    size_t b = s.len;
    while (b > 0) {
        size_t k = b - 1;
        while (k > 0 && ((unsigned char)s.ptr[k] & 0xc0) == 0x80) k--;
        size_t q = k;
        if (!tui_is_space(tui_cp(s, &q))) break;
        b = k;
    }
    return str_sub(s, b, s.len);
}

static size_t common_prefix(Str a, Str b) {
    size_t i = 0;
    while (i < a.len && i < b.len && a.ptr[i] == b.ptr[i]) i++;
    return i;
}

static size_t common_suffix(Str a, Str b) {
    size_t i = 0;
    while (i < a.len && i < b.len && a.ptr[a.len - 1 - i] == b.ptr[b.len - 1 - i]) i++;
    return i;
}

static void buf_set(Buf *b, Str s) {
    Buf n = {0};
    buf_addstr(&n, s);
    if (!n.ptr) buf_adds(&n, "");
    buf_free(b);
    *b = n;
}

// overlapCount: the longest suffix of a that is a prefix of b.
static size_t overlap(Str a, Str b) {
    size_t best = 0;
    for (size_t n = 1; n <= a.len && n <= b.len; n++)
        if (!memcmp(a.ptr + a.len - n, b.ptr, n)) best = n;
    return best;
}

static void dedupe(Part *sk, Part *del, Part *ins, Part *ek) {
    if (del && ins) {
        Str ow = lead_ws(buf_str(&del->value)), nw = lead_ws(buf_str(&ins->value));
        if (sk) {
            size_t cp = common_prefix(ow, nw);
            // startKeep: its suffix nw becomes the common part
            Str v = buf_str(&sk->value);
            Buf n = {0};
            buf_add(&n, v.ptr, v.len >= nw.len ? v.len - nw.len : v.len);
            buf_add(&n, nw.ptr, cp);
            if (!n.ptr) buf_adds(&n, "");
            buf_free(&sk->value);
            sk->value = n;
            buf_set(&del->value, str_sub(buf_str(&del->value), cp, del->value.len));
            buf_set(&ins->value, str_sub(buf_str(&ins->value), cp, ins->value.len));
        }
        if (ek) {
            Str os = trail_ws(buf_str(&del->value)), ns = trail_ws(buf_str(&ins->value));
            size_t cs = common_suffix(os, ns);
            Str v = buf_str(&ek->value);
            Buf n = {0};
            buf_add(&n, ns.ptr + ns.len - cs, cs);
            buf_addstr(&n, str_sub(v, MIN(ns.len, v.len), v.len));
            if (!n.ptr) buf_adds(&n, "");
            buf_free(&ek->value);
            ek->value = n;
            buf_set(&del->value, str_n(del->value.ptr, del->value.len - cs));
            buf_set(&ins->value, str_n(ins->value.ptr, ins->value.len - cs));
        }
    } else if (ins) {
        if (sk) {
            Str w = lead_ws(buf_str(&ins->value));
            buf_set(&ins->value, str_sub(buf_str(&ins->value), w.len, ins->value.len));
        }
        if (ek) {
            Str w = lead_ws(buf_str(&ek->value));
            buf_set(&ek->value, str_sub(buf_str(&ek->value), w.len, ek->value.len));
        }
    } else if (del && sk && ek) {
        Buf nwf_b = {0};
        buf_addstr(&nwf_b, lead_ws(buf_str(&ek->value)));
        Str nwf = buf_str(&nwf_b);
        Buf de_b = {0};
        buf_addstr(&de_b, trail_ws(buf_str(&del->value)));
        Str ds = lead_ws(buf_str(&del->value));
        size_t ns = common_prefix(nwf, ds);
        buf_set(&del->value, str_sub(buf_str(&del->value), ns, del->value.len));
        Str rest = str_sub(nwf, ns, nwf.len);
        size_t ne = common_suffix(rest, buf_str(&de_b));
        if (ne > del->value.len) ne = del->value.len;
        buf_set(&del->value, str_n(del->value.ptr, del->value.len - ne));
        Buf kept = {0};
        buf_addstr(&kept, str_sub(nwf, nwf.len - ne, nwf.len));
        buf_addstr(&kept, str_sub(buf_str(&ek->value), nwf.len, ek->value.len));
        if (!kept.ptr) buf_adds(&kept, "");
        // startKeep: its trailing nwf (when it has it) replaced by what endKeep did not take
        Buf start = {0};
        Str sv = buf_str(&sk->value);
        if (nwf.len && str_ends(sv, nwf)) sv.len -= nwf.len;
        buf_addstr(&start, sv);
        buf_addstr(&start, str_n(nwf.ptr, nwf.len - ne));
        if (!start.ptr) buf_adds(&start, "");
        buf_free(&ek->value);
        ek->value = kept;
        buf_free(&sk->value);
        sk->value = start;
        buf_free(&nwf_b);
        buf_free(&de_b);
    } else if (del && ek) {
        Str ekp = lead_ws(buf_str(&ek->value)), dsx = trail_ws(buf_str(&del->value));
        size_t ov = overlap(dsx, ekp);
        buf_set(&del->value, str_n(del->value.ptr, del->value.len - ov));
    } else if (del && sk) {
        Str sks = trail_ws(buf_str(&sk->value)), dp = lead_ws(buf_str(&del->value));
        size_t ov = overlap(sks, dp);
        buf_set(&del->value, str_sub(buf_str(&del->value), ov, del->value.len));
    }
}

// diffWords(old, new): the parts in order, each kept, removed or added.
static void diff_words(Str old, Str nw, PartList *out) {
    Arena a = {0};
    StrList ot = {0}, nt = {0};
    tokenize(&a, old, &ot);
    tokenize(&a, nw, &nt);
    long nl = (long)nt.len, ol = (long)ot.len;
    long maxe = nl + ol;
    Path *best = arena_alloc(&a, (size_t)(2 * maxe + 3) * sizeof *best);
    Path *bp = best + maxe + 1; // diagonal index -maxe-1 .. maxe+1
    bp[0] = (Path){-1, NULL, true};
    long np = extract_common(&a, &bp[0], &nt, &ot, 0);
    Comp_ *result = NULL;
    bool done = false;
    if (bp[0].old_pos + 1 >= ol && np + 1 >= nl) {
        result = bp[0].last;
        done = true;
    }
    long mind = -maxe - 1, maxd = maxe + 1;
    for (long e = 1; !done && e <= maxe; e++) {
        for (long d = MAX(mind, -e); d <= MIN(maxd, e); d += 2) {
            Path rp = bp[d - 1], ap = bp[d + 1];
            if (rp.set) bp[d - 1].set = false;
            bool can_add = false;
            if (ap.set) {
                long anp = ap.old_pos - d;
                can_add = anp >= 0 && anp < nl;
            }
            bool can_rem = rp.set && rp.old_pos + 1 < ol;
            if (!can_add && !can_rem) {
                bp[d].set = false;
                continue;
            }
            Path base;
            if (!can_rem || (can_add && rp.old_pos < ap.old_pos)) base = add_to_path(&a, ap, true, false, 0);
            else base = add_to_path(&a, rp, false, true, 1);
            np = extract_common(&a, &base, &nt, &ot, d);
            if (base.old_pos + 1 >= ol && np + 1 >= nl) {
                result = base.last;
                done = true;
                break;
            }
            bp[d] = base;
            if (base.old_pos + 1 >= ol) maxd = MIN(maxd, d - 1);
            if (np + 1 >= nl) mind = MAX(mind, d + 1);
        }
    }
    build_values(result, &nt, &ot, out);
    // postProcess: the whitespace between changes and kept text, deduped
    long last_keep = -1, ins = -1, del = -1;
    for (size_t i = 0; i < out->len; i++) {
        Part *p = &out->ptr[i];
        if (p->added) ins = (long)i;
        else if (p->removed) del = (long)i;
        else {
            if (ins >= 0 || del >= 0)
                dedupe(last_keep >= 0 ? &out->ptr[last_keep] : NULL, del >= 0 ? &out->ptr[del] : NULL, ins >= 0 ? &out->ptr[ins] : NULL, p);
            last_keep = (long)i;
            ins = del = -1;
        }
    }
    if (ins >= 0 || del >= 0)
        dedupe(last_keep >= 0 ? &out->ptr[last_keep] : NULL, del >= 0 ? &out->ptr[del] : NULL, ins >= 0 ? &out->ptr[ins] : NULL, NULL);
    vec_free(&ot);
    vec_free(&nt);
    arena_reset(&a);
}

void tui_intra_line(Str old, Str nw, Buf *rem, Buf *add) {
    // the common indentation stays out of the diff
    size_t ci = 0;
    while (ci < old.len && ci < nw.len && old.ptr[ci] == nw.ptr[ci] && (old.ptr[ci] == ' ' || old.ptr[ci] == '\t')) ci++;
    buf_addstr(rem, str_sub(old, 0, ci));
    buf_addstr(add, str_sub(nw, 0, ci));
    PartList parts = {0};
    diff_words(str_sub(old, ci, old.len), str_sub(nw, ci, nw.len), &parts);
    bool first_rem = true, first_add = true;
    for (size_t i = 0; i < parts.len; i++) {
        Part *p = &parts.ptr[i];
        Str v = buf_str(&p->value);
        if (p->removed || p->added) {
            Buf *dst = p->removed ? rem : add;
            bool *first = p->removed ? &first_rem : &first_add;
            if (*first) {
                Str w = lead_ws(v);
                buf_addstr(dst, w);
                v = str_sub(v, w.len, v.len);
                *first = false;
            }
            if (v.len) th_inverse(dst, v);
        } else {
            buf_addstr(rem, v);
            buf_addstr(add, v);
        }
        buf_free(&p->value);
    }
    vec_free(&parts);
}
