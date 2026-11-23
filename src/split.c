// Splitting a reply into reasoning and answer for streaming (see split.h).
#include "split.h"

#include "toolcall.h"

static Str sp_text(const Splitter *sp) { return buf_str(&sp->text); }

Str split_reasoning(const Splitter *sp) {
    if (sp->inkling) return str_trim(buf_str(&sp->ink.reasoning));
    if (!sp->thinking) return S("");
    Str t = sp_text(sp);
    ptrdiff_t end = str_find(t, S("</think>"), 0);
    return str_trim(str_sub(t, 0, end < 0 ? t.len : (size_t)end));
}

Str split_answer(const Splitter *sp) {
    Str t = sp_text(sp);
    if (sp->inkling) return buf_str(&sp->ink.content);
    if (!sp->thinking) return t;
    ptrdiff_t e = str_find(t, S("</think>"), 0);
    if (e < 0) return S("");
    return str_sub(t, (size_t)e + 8, t.len);
}

void split_finish(Splitter *sp) {
    if (sp->inkling) toolcall_inkling_scan(sp->arena, &sp->ink, sp_text(sp), true);
}

void split_free(Splitter *sp) {
    buf_free(&sp->text);
    toolcall_inkling_free(&sp->ink);
}

// Inkling: stop strings in the answer (the scan's content), searched in its new bytes only; a
// hit cuts the content there and ends the scan.
static const Str *ink_cut_stop(Splitter *sp) {
    if (sp->ink.done || sp->n_stops == 0) return NULL;
    toolcall_inkling_scan(sp->arena, &sp->ink, sp_text(sp), false);
    Str c = buf_str(&sp->ink.content);
    size_t longest = 0;
    for (size_t i = 0; i < sp->n_stops; i++) longest = MAX(longest, sp->stops[i].len);
    const size_t from = sp->stop_from > longest ? sp->stop_from - longest : 0;
    const Str *hit = NULL;
    size_t at = c.len;
    for (size_t i = 0; i < sp->n_stops; i++) {
        if (sp->stops[i].len == 0) continue;
        ptrdiff_t k = str_find(c, sp->stops[i], from);
        if (k >= 0 && (size_t)k < at) {
            at = (size_t)k;
            hit = &sp->stops[i];
        }
    }
    sp->stop_from = c.len;
    if (hit) {
        sp->ink.content.len = at;
        sp->ink.done = true;
    }
    return hit;
}

const Str *split_cut_stop(Splitter *sp) {
    if (sp->inkling) return ink_cut_stop(sp);
    for (size_t i = 0; i < sp->n_stops; i++) {
        Str st = sp->stops[i];
        Str ans = split_answer(sp);
        ptrdiff_t cut = st.len > 0 ? str_find(ans, st, 0) : -1;
        if (cut >= 0) {
            sp->text.len -= ans.len - (size_t)cut;
            sp->text.ptr[sp->text.len] = 0;
            return &sp->stops[i];
        }
    }
    return NULL;
}

static bool ascii_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x0B || c == 0x0C; }
static bool trim_ws(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

static size_t lead_ws(Str s) {
    size_t i = 0;
    while (i < s.len && trim_ws(s.ptr[i])) i++;
    return i;
}
static size_t trim_end_len(Str s) {
    size_t n = s.len;
    while (n > 0 && trim_ws(s.ptr[n - 1])) n--;
    return n;
}

// The longest tail of s (shorter than p) that is a prefix of p.
static size_t prefix_tail(Str s, Str p) {
    if (p.len == 0) return 0;
    for (size_t k = MIN(p.len - 1, s.len); k > 0; k--)
        if (memcmp(s.ptr + s.len - k, p.ptr, k) == 0) return k;
    return 0;
}

size_t split_stable(Str s, Str tag, const Str *stops, size_t n_stops) {
    size_t hold = prefix_tail(s, tag);
    for (size_t i = 0; i < n_stops; i++) hold = MAX(hold, prefix_tail(s, stops[i]));
    size_t n = s.len - hold;
    size_t cut = n, back = 0;
    while (cut > 0 && back < 4 && ((unsigned char)s.ptr[cut - 1] & 0xC0) == 0x80) {
        back += 1;
        cut -= 1;
    }
    if (cut > 0 && (unsigned char)s.ptr[cut - 1] >= 0xC0) {
        unsigned char b = (unsigned char)s.ptr[cut - 1];
        size_t len = b < 0xE0 ? 2 : b < 0xF0 ? 3 : b < 0xF8 ? 4 : 1;
        if (back + 1 < len) n = cut - 1;
    }
    while (n > 0 && ascii_ws(s.ptr[n - 1])) n -= 1;
    return n;
}

SplitPoll split_poll(Splitter *sp, bool final) {
    Str t = sp_text(sp);
    SplitPoll r = {S(""), S("")};
    if (sp->inkling) {
        // the scan of a growing reply only appends; the reasoning's leading whitespace is left
        // out as in the Qwen split
        toolcall_inkling_scan(sp->arena, &sp->ink, t, final);
        final = final || sp->ink.done;
        Str ir = buf_str(&sp->ink.reasoning), ic = buf_str(&sp->ink.content);
        Str rs = str_sub(ir, lead_ws(ir), ir.len);
        size_t nr = final ? trim_end_len(rs) : split_stable(rs, S(""), NULL, 0);
        if (nr > sp->sent_reasoning) {
            r.reasoning = str_sub(rs, sp->sent_reasoning, nr);
            sp->sent_reasoning = nr;
        }
        Str cs = str_sub(ic, lead_ws(ic), ic.len);
        size_t hold = 0;
        for (size_t i = 0; i < sp->n_stops && !final; i++) hold = MAX(hold, prefix_tail(cs, sp->stops[i]));
        size_t nc = final ? trim_end_len(cs) : split_stable(str_sub(cs, 0, cs.len - hold), S(""), NULL, 0);
        if (nc > sp->sent_content) {
            r.content = str_sub(cs, sp->sent_content, nc);
            sp->sent_content = nc;
        }
        return r;
    }
    if (sp->thinking) {
        ptrdiff_t closed = str_find(t, S("</think>"), 0);
        Str raw = str_sub(t, 0, closed < 0 ? t.len : (size_t)closed);
        Str body = str_sub(raw, lead_ws(raw), raw.len);
        size_t n = (closed >= 0 || final) ? trim_end_len(body) : split_stable(body, S("</think>"), NULL, 0);
        if (n > sp->sent_reasoning) {
            r.reasoning = str_sub(body, sp->sent_reasoning, n);
            sp->sent_reasoning = n;
        }
        if (closed < 0) return r;
    }
    Str ans = split_answer(sp);
    size_t lead = lead_ws(ans);
    Str body = str_sub(ans, lead, ans.len);
    ptrdiff_t tool = toolcall_start(body);
    size_t upto = tool >= 0 ? (size_t)tool : body.len;
    size_t n;
    if (tool >= 0 || final) n = trim_end_len(str_sub(body, 0, upto));
    else {
        // a possible stop string start is held back in the raw answer, where split_cut_stop
        // looks for it (a stop string may begin with the whitespace body leaves out)
        size_t hold = 0;
        for (size_t i = 0; i < sp->n_stops; i++) hold = MAX(hold, prefix_tail(ans, sp->stops[i]));
        size_t keep = ans.len - hold > lead ? ans.len - hold - lead : 0;
        n = split_stable(str_sub(body, 0, keep), S("<tool_call>"), NULL, 0);
    }
    if (n > sp->sent_content) {
        r.content = str_sub(body, sp->sent_content, n);
        sp->sent_content = n;
    }
    return r;
}
