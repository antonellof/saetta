// The tool view of a subagent call (see code_subagent.h): the call, then the child's progress or
// result, drawn inside code_ui.c's tool box.
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi v1.5.2,
// b4afddd): packages/coding-agent/examples/extensions/subagent/index.ts (renderCall and renderResult
// in single mode, formatToolCall, formatUsageStats, formatTokens, getDisplayItems), and
// core/tools/render-utils.ts (formatReadLineRange).
#include "code_tui.h"
#include "code_subagent.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define COLLAPSED_ITEM_COUNT 10

static void text(Str t, int width, Lines *out) {
    Comp *c = comp_text(t, 0, 0);
    comp_render(c, width, out);
    comp_free(c);
}

static void bold_title(Buf *b, Str t) {
    Buf x = {0};
    th_bold(&x, t);
    th_fg(b, T_TOOL_TITLE, buf_str(&x));
    buf_free(&x);
}

// Model-written text on one line: escapes dropped, control characters made spaces.
static void one_line(Buf *out, Str s) {
    Buf t = {0};
    tui_strip(&t, s);
    for (size_t i = 0; i < t.len; i++) buf_addc(out, (unsigned char)t.ptr[i] < 0x20 ? ' ' : t.ptr[i]);
    buf_free(&t);
}

// The first n characters (code points) of s, then "..." when s is longer.
static void preview(Buf *out, Str s, size_t n) {
    size_t i = 0, k = 0;
    while (i < s.len && k < n) {
        i++;
        while (i < s.len && ((unsigned char)s.ptr[i] & 0xC0) == 0x80) i++;
        k++;
    }
    Buf t = {0};
    one_line(&t, str_sub(s, 0, i));
    buf_addstr(out, buf_str(&t));
    if (i < s.len) buf_adds(out, "...");
    buf_free(&t);
}

static double jnum(const JValue *v, bool *ok) {
    *ok = true;
    if (json_is(v, J_INTEGER)) return (double)v->i;
    if (json_is(v, J_FLOAT)) return v->f;
    if (json_is(v, J_NUMBER_STRING)) {
        char t[64];
        snprintf(t, sizeof t, "%.*s", (int)MIN(v->s.len, sizeof t - 1), v->s.ptr);
        return strtod(t, NULL);
    }
    *ok = false;
    return 0;
}

static void shorten(Buf *b, Str p) {
    const char *home = getenv("HOME");
    size_t hl = home ? strlen(home) : 0;
    Buf t = {0};
    if (hl && p.len >= hl && !memcmp(p.ptr, home, hl)) {
        buf_addc(&t, '~');
        buf_addstr(&t, str_sub(p, hl, p.len));
    } else buf_addstr(&t, p);
    one_line(b, buf_str(&t));
    buf_free(&t);
}

// formatToolCall
static void format_call(Buf *b, Str name, const JValue *args) {
    Buf t = {0};
    Str path = json_strs(json_get(args, "file_path"));
    if (!path.len) path = json_strs(json_get(args, "path"));
    if (str_eqz(name, "bash")) {
        Str cmd = json_strs(json_get(args, "command"));
        th_fgz(b, T_MUTED, "$ ");
        preview(&t, cmd.len ? cmd : S("..."), 60);
        th_fg(b, T_TOOL_OUTPUT, buf_str(&t));
    } else if (str_eqz(name, "read")) {
        th_fgz(b, T_MUTED, "read ");
        shorten(&t, path.len ? path : S("..."));
        th_fg(b, T_ACCENT, buf_str(&t));
        // formatReadLineRange
        bool ho, hl;
        double off = jnum(json_get(args, "offset"), &ho), lim = jnum(json_get(args, "limit"), &hl);
        if (ho || hl) {
            double start = ho ? off : 1;
            Buf r = {0};
            if (start >= 1 && start == floor(start)) {
                if (!hl) buf_printf(&r, ":%.0f", start);
                else if (lim >= 1 && lim == floor(lim)) buf_printf(&r, ":%.0f-%.0f", start, start + lim - 1);
            }
            if (r.len) th_fg(b, T_WARNING, buf_str(&r));
            buf_free(&r);
        }
    } else if (str_eqz(name, "write")) {
        th_fgz(b, T_MUTED, "write ");
        shorten(&t, path.len ? path : S("..."));
        th_fg(b, T_ACCENT, buf_str(&t));
        Str content = json_strs(json_get(args, "content"));
        size_t lines = 1;
        for (size_t i = 0; i < content.len; i++) lines += content.ptr[i] == '\n';
        if (lines > 1) {
            Buf n = {0};
            buf_printf(&n, " (%zu lines)", lines);
            th_fg(b, T_DIM, buf_str(&n));
            buf_free(&n);
        }
    } else if (str_eqz(name, "edit")) {
        th_fgz(b, T_MUTED, "edit ");
        shorten(&t, path.len ? path : S("..."));
        th_fg(b, T_ACCENT, buf_str(&t));
    } else if (str_eqz(name, "ls")) {
        th_fgz(b, T_MUTED, "ls ");
        shorten(&t, path.len ? path : S("."));
        th_fg(b, T_ACCENT, buf_str(&t));
    } else if (str_eqz(name, "find") || str_eqz(name, "grep")) {
        Str pat = json_strs(json_get(args, "pattern"));
        bool grep = str_eqz(name, "grep");
        th_fgz(b, T_MUTED, grep ? "grep " : "find ");
        if (grep) buf_adds(&t, "/");
        one_line(&t, pat.len || grep ? pat : S("*"));
        if (grep) buf_adds(&t, "/");
        th_fg(b, T_ACCENT, buf_str(&t));
        Buf in = {0};
        buf_adds(&in, " in ");
        shorten(&in, path.len ? path : S("."));
        th_fg(b, T_DIM, buf_str(&in));
        buf_free(&in);
    } else {
        Buf j = {0}, n = {0};
        json_write(&j, args ? args : &(JValue){.kind = J_OBJECT});
        one_line(&n, name);
        th_fg(b, T_ACCENT, buf_str(&n));
        buf_adds(&t, " ");
        preview(&t, buf_str(&j), 50);
        th_fg(b, T_DIM, buf_str(&t));
        buf_free(&j);
        buf_free(&n);
    }
    buf_free(&t);
}

// formatTokens
static void tokens(Buf *b, double n) {
    if (n < 1000) buf_printf(b, "%.0f", n);
    else if (n < 10000) buf_printf(b, "%.1fk", n / 1000);
    else if (n < 1000000) buf_printf(b, "%.0fk", round(n / 1000));
    else buf_printf(b, "%.1fM", n / 1000000);
}

// formatUsageStats: "N turns ↑in ↓out"
static void usage(Buf *b, const JValue *d) {
    bool ok;
    double turns = jnum(json_get(d, "turns"), &ok), in = jnum(json_get(json_get(d, "usage"), "input"), &ok),
           out = jnum(json_get(json_get(d, "usage"), "output"), &ok);
    if (turns) buf_printf(b, "%.0f turn%s", turns, turns > 1 ? "s" : "");
    if (in) {
        buf_adds(b, b->len ? " \xe2\x86\x91" : "\xe2\x86\x91");
        tokens(b, in);
    }
    if (out) {
        buf_adds(b, b->len ? " \xe2\x86\x93" : "\xe2\x86\x93");
        tokens(b, out);
    }
}

typedef struct {
    bool call;
    Str text, name;
    const JValue *args;
} Item;

// getDisplayItems: the texts and tool calls of the child's answers, in order.
static size_t items_of(const JValue *messages, Item **out) {
    size_t n = 0, cap = 8;
    Item *v = xmalloc(cap * sizeof *v);
    for (size_t i = 0; messages && json_is(messages, J_ARRAY) && i < messages->arr.len; i++) {
        const JValue *m = messages->arr.items[i];
        if (!str_eqz(json_strs(json_get(m, "role")), "assistant")) continue;
        const JValue *c = json_get(m, "content");
        for (size_t k = 0; c && json_is(c, J_ARRAY) && k < c->arr.len; k++) {
            const JValue *p = c->arr.items[k];
            Str t = json_strs(json_get(p, "type"));
            Item it = {0};
            if (str_eqz(t, "text")) it = (Item){.text = json_strs(json_get(p, "text"))};
            else if (str_eqz(t, "toolCall")) it = (Item){.call = true, .name = json_strs(json_get(p, "name")), .args = json_get(p, "arguments")};
            else continue;
            if (n == cap) v = xrealloc(v, (cap *= 2) * sizeof *v);
            v[n++] = it;
        }
    }
    *out = v;
    return n;
}

// A text item: its first 3 lines collapsed, all of it expanded.
static void text_item(Buf *b, Str t, bool expanded) {
    Buf x = {0};
    tui_strip(&x, t);
    Str s = buf_str(&x);
    size_t end = s.len;
    if (!expanded) {
        size_t lines = 0;
        for (size_t i = 0; i < s.len; i++)
            if (s.ptr[i] == '\n' && ++lines == 3) {
                end = i;
                break;
            }
    }
    th_fg(b, T_TOOL_OUTPUT, str_sub(s, 0, end));
    buf_free(&x);
}

void code_subagent_ui_render(const JValue *args, const JValue *result, bool has_result, bool partial, bool is_error,
                             bool expanded, int width, Lines *out) {
    // renderCall: "subagent <description>", then the task's first 60 characters
    Buf call = {0}, t = {0};
    bold_title(&call, S("subagent "));
    Str desc = json_strs(json_get(args, "description"));
    one_line(&t, desc.len ? desc : S("..."));
    th_fg(&call, T_ACCENT, buf_str(&t));
    t.len = 0;
    Str prompt = json_strs(json_get(args, "prompt"));
    buf_adds(&call, "\n  ");
    preview(&t, prompt.len ? prompt : S("..."), 60);
    th_fg(&call, T_DIM, buf_str(&t));
    text(buf_str(&call), width, out);
    buf_free(&call);
    buf_free(&t);
    if (!has_result) return;
    const JValue *d = json_get(result, "details");
    Str status = json_strs(json_get(d, "status"));
    bool running = partial || str_eqz(status, "running");
    bool failed = !running && (is_error || str_eqz(status, "error") || str_eqz(status, "aborted"));
    Item *items;
    size_t n = items_of(json_get(d, "messages"), &items);
    Str final = {0};
    for (size_t i = n; i-- > 0;)
        if (!items[i].call) {
            final = items[i].text;
            break;
        }
    Buf b = {0};
    lines_addz(out, "");
    // the header: ⏳ running, ✗ failed, ✓ done
    th_fgz(&b, running ? T_WARNING : failed ? T_ERROR : T_SUCCESS, running ? "\xe2\x8f\xb3" : failed ? "\xe2\x9c\x97" : "\xe2\x9c\x93");
    buf_addc(&b, ' ');
    Buf h = {0};
    one_line(&h, desc.len ? desc : S("subagent"));
    bold_title(&b, buf_str(&h));
    buf_free(&h);
    if (running) th_fgz(&b, T_WARNING, " [running]");
    else if (failed) {
        Buf s = {0};
        buf_printf(&s, " [%.*s]", (int)(status.len ? status.len : 5), status.len ? status.ptr : "error");
        th_fg(&b, T_ERROR, buf_str(&s));
        buf_free(&s);
    }
    Str em = json_strs(json_get(d, "errorMessage"));
    if (failed && !em.len) {
        // a result without details (a call refused before it ran): its text is the error
        const JValue *c = json_get(result, "content");
        if (c && json_is(c, J_ARRAY) && c->arr.len) em = json_strs(json_get(c->arr.items[0], "text"));
    }
    if (expanded) {
        if (failed && em.len) {
            buf_addc(&b, '\n');
            Buf e = {0};
            buf_adds(&e, "Error: ");
            one_line(&e, em);
            th_fg(&b, T_ERROR, buf_str(&e));
            buf_free(&e);
        }
        buf_adds(&b, "\n\n");
        th_fgz(&b, T_MUTED, "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80 Task \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80");
        buf_addc(&b, '\n');
        Buf p = {0};
        tui_strip(&p, prompt);
        th_fg(&b, T_DIM, buf_str(&p));
        buf_free(&p);
        buf_adds(&b, "\n\n");
        th_fgz(&b, T_MUTED, "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80 Output \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80");
        if (!n) {
            buf_addc(&b, '\n');
            th_fgz(&b, T_MUTED, "(no output)");
        }
        for (size_t i = 0; i < n; i++) {
            if (!items[i].call) continue;
            buf_addc(&b, '\n');
            th_fgz(&b, T_MUTED, "\xe2\x86\x92 ");
            format_call(&b, items[i].name, items[i].args);
        }
        if (final.len) {
            buf_adds(&b, "\n\n");
            text_item(&b, str_trim(final), true);
        }
        Buf u = {0};
        usage(&u, d);
        if (u.len) {
            buf_adds(&b, "\n\n");
            th_fg(&b, T_DIM, buf_str(&u));
        }
        buf_free(&u);
    } else {
        if (failed && em.len) {
            buf_addc(&b, '\n');
            Buf e = {0};
            buf_adds(&e, "Error: ");
            one_line(&e, em);
            th_fg(&b, T_ERROR, buf_str(&e));
            buf_free(&e);
        } else if (!n) {
            buf_addc(&b, '\n');
            th_fgz(&b, T_MUTED, "(no output)");
        } else {
            // renderDisplayItems(items, COLLAPSED_ITEM_COUNT): the last ten
            size_t from = n > COLLAPSED_ITEM_COUNT ? n - COLLAPSED_ITEM_COUNT : 0;
            if (from) {
                Buf s = {0};
                buf_printf(&s, "... %zu earlier items", from);
                buf_addc(&b, '\n');
                th_fg(&b, T_MUTED, buf_str(&s));
                buf_free(&s);
            }
            for (size_t i = from; i < n; i++) {
                buf_addc(&b, '\n');
                if (items[i].call) {
                    th_fgz(&b, T_MUTED, "\xe2\x86\x92 ");
                    format_call(&b, items[i].name, items[i].args);
                } else text_item(&b, str_trim(items[i].text), false);
            }
            if (n > COLLAPSED_ITEM_COUNT) {
                buf_addc(&b, '\n');
                th_fgz(&b, T_MUTED, "(Ctrl+O to expand)");
            }
        }
        Buf u = {0};
        usage(&u, d);
        if (u.len) {
            buf_addc(&b, '\n');
            th_fg(&b, T_DIM, buf_str(&u));
        }
        buf_free(&u);
    }
    text(buf_str(&b), width, out);
    buf_free(&b);
    free(items);
}
