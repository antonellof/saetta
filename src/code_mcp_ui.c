// Saetta Code's /mcp manager and the views of MCP tool calls (see code_mcp.h).
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at v1.5.2,
// b4afddd): packages/coding-agent/src/extensions/mcp/ui.ts (McpManagerView, frame), index.ts
// (manage, runAction, the command's handler and argument completions), tools.ts
// (createMcpToolRenderers), core/tools/render-utils.ts (formatToolCallWithArgs, replaceTabs),
// modes/interactive/components/visual-truncate.ts (VisualLinePreview), keybinding-hints.ts,
// auth-url.ts (AuthUrlComponent); packages/tui/src/components/input.ts (the paste prompt).
#include "code_mcp_ui.h"
#include "code_cmds.h"
#include "code_oai.h"

#include <stdlib.h>

// ---- the frame (ui.ts frame, DynamicBorder, keyHint) ------------------------------------------------

static void border(int width, Lines *out) {
    Buf b = {0}, r = {0};
    tui_repeat(&r, "\xe2\x94\x80", MAX(1, width));
    th_fg(&b, T_ACCENT, buf_str(&r));
    lines_addb(out, &b);
    buf_free(&r);
}

static void text(Str t, int px, int width, Lines *out) {
    Comp *c = comp_text(t, px, 0);
    comp_render(c, width, out);
    comp_free(c);
}

static void themed_text(ThemeColor color, const char *t, int width, Lines *out) {
    Buf b = {0};
    th_fgz(&b, color, t);
    text(buf_str(&b), 1, width, out);
    buf_free(&b);
}

// keyHint(binding, description): the keys dim, the description muted
static void key_hint(Buf *b, const char *keys, const char *desc) {
    th_fgz(b, T_DIM, keys);
    Buf d = {0};
    buf_printf(&d, " %s", desc);
    th_fg(b, T_MUTED, buf_str(&d));
    buf_free(&d);
}

#define CONFIRM_KEYS "enter"
#define CANCEL_KEYS "escape/ctrl+c"

// frame(): border, the title, the body, the footer (dim) after a spacer, border
static void frame_open(const char *title, int width, Lines *out) {
    border(width, out);
    Buf t = {0}, b = {0};
    th_bold(&t, str(title));
    th_fg(&b, T_ACCENT, buf_str(&t));
    text(buf_str(&b), 1, width, out);
    buf_free(&t);
    buf_free(&b);
}

static void frame_close(Str footer, int width, Lines *out) {
    if (footer.len) {
        lines_addz(out, "");
        Buf b = {0};
        th_fg(&b, T_DIM, footer);
        text(buf_str(&b), 1, width, out);
        buf_free(&b);
    }
    border(width, out);
}

// ---- the sign-in screen (McpManagerView.status and redirectUrl, AuthUrlComponent) -------------------

static struct {
    char *url;     // the URL the prompt is for (a new one starts it empty)
    Buf value;     // the pasted redirect URL (pi-tui Input)
    size_t cursor;
    char *hint;    // the copy hint after ctrl+x: NULL, or the result (copied, or the error)
    bool hint_ok;
} SI;

static void si_sync(const char *url) {
    if (url && SI.url && !strcmp(url, SI.url)) return;
    free(SI.url);
    SI.url = url ? strdup(url) : NULL;
    SI.value.len = 0;
    if (SI.value.ptr) SI.value.ptr[0] = 0;
    SI.cursor = 0;
    free(SI.hint);
    SI.hint = NULL;
}

// a URL from the authorization server, without control characters
static char *clean_url(const char *u) {
    Buf b = {0};
    for (const unsigned char *p = (const unsigned char *)u; *p; p++)
        if (*p >= 0x20 && *p != 0x7f) buf_addc(&b, (char)*p);
    return b.ptr ? b.ptr : strdup("");
}

static void hyperlink(Buf *b, const char *text, const char *url) { buf_printf(b, "\x1b]8;;%s\x1b\\%s\x1b]8;;\x1b\\", url, text); }

// Input.render: "> ", the value (scrolled to keep the cursor in view) with the fake cursor,
// padded to the width
static void input_render(int width, Lines *out) {
    Buf b = {0}, vis = {0};
    buf_adds(&b, "> ");
    int avail = width - 2;
    if (avail <= 0) {
        lines_addb(out, &b);
        return;
    }
    Str v = buf_str(&SI.value);
    int total = tui_width(v);
    size_t cur = MIN(SI.cursor, v.len);
    Str shown = v;
    size_t cd = cur; // the cursor's byte offset in what is shown
    if (total >= avail) {
        int scroll = cur == v.len ? avail - 1 : avail;
        int ccol = tui_width(str_n(v.ptr, cur)), half = scroll / 2;
        int start = ccol < half ? 0 : ccol > total - half ? MAX(0, total - scroll) : MAX(0, ccol - half);
        if (scroll > 0) {
            tui_slice(&vis, v, start, scroll, true);
            Buf before = {0};
            tui_slice(&before, v, start, MAX(0, ccol - start), true);
            cd = before.len;
            buf_free(&before);
        } else cd = 0;
        shown = buf_str(&vis);
    }
    buf_add(&b, shown.ptr, MIN(cd, shown.len));
    size_t g = cd < shown.len ? tui_grapheme(shown, cd) : 0;
    buf_adds(&b, "\x1b[7m");
    if (g) buf_add(&b, shown.ptr + cd, g);
    else buf_addc(&b, ' ');
    buf_adds(&b, "\x1b[27m");
    if (cd + g < shown.len) buf_add(&b, shown.ptr + cd + g, shown.len - cd - g);
    int w = tui_width(buf_str(&b)) - 2;
    tui_spaces(&b, MAX(0, avail - w));
    buf_free(&vis);
    lines_addb(out, &b);
}

static void input_key(const Key *k) {
    Buf *v = &SI.value;
    if (!v->ptr) buf_adds(v, "");
    if (k->kind == K_CHAR && k->ch >= 0x20) {
        char enc[4];
        int n = tui_utf8(enc, k->ch);
        buf_reserve(v, (size_t)n);
        memmove(v->ptr + SI.cursor + n, v->ptr + SI.cursor, v->len - SI.cursor + 1);
        memcpy(v->ptr + SI.cursor, enc, (size_t)n);
        v->len += (size_t)n;
        SI.cursor += (size_t)n;
    } else if (k->kind == K_PASTE && k->paste) {
        for (const char *p = k->paste; *p; p++)
            if ((unsigned char)*p >= 0x20) {
                buf_reserve(v, 1);
                memmove(v->ptr + SI.cursor + 1, v->ptr + SI.cursor, v->len - SI.cursor + 1);
                v->ptr[SI.cursor++] = *p;
                v->len++;
            }
    } else if (k->kind == K_BACKSPACE && SI.cursor > 0) {
        size_t p = SI.cursor - 1;
        while (p > 0 && ((unsigned char)v->ptr[p] & 0xc0) == 0x80) p--;
        memmove(v->ptr + p, v->ptr + SI.cursor, v->len - SI.cursor + 1);
        v->len -= SI.cursor - p;
        SI.cursor = p;
    } else if (k->kind == K_LEFT && SI.cursor > 0) {
        SI.cursor--;
        while (SI.cursor > 0 && ((unsigned char)v->ptr[SI.cursor] & 0xc0) == 0x80) SI.cursor--;
    } else if (k->kind == K_RIGHT && SI.cursor < v->len) SI.cursor += tui_grapheme(buf_str(v), SI.cursor);
    else if (k->kind == K_HOME || (k->kind == K_CTRL && k->ch == 'a')) SI.cursor = 0;
    else if (k->kind == K_END || (k->kind == K_CTRL && k->ch == 'e')) SI.cursor = v->len;
    else if (k->kind == K_CTRL && k->ch == 'u') {
        memmove(v->ptr, v->ptr + SI.cursor, v->len - SI.cursor + 1);
        v->len -= SI.cursor;
        SI.cursor = 0;
    }
}

static void signin_render(int width, Lines *out) {
    char *title, *msg, *url;
    if (!code_mcp_signin_screen(&title, &msg, &url)) {
        title = strdup("MCP servers");
        msg = strdup("Loading\xe2\x80\xa6");
        url = NULL;
    }
    char *u = url ? clean_url(url) : NULL;
    si_sync(u);
    Lines ls = {0};
    frame_open(title ? title : "", width, &ls);
    Buf footer = {0};
    lines_addz(&ls, "");
    if (u) {
        themed_text(T_MUTED, "Approve access in your browser. If it did not open, visit:", width, &ls);
        // AuthUrlComponent: the link, then the click and copy hints
        Buf l = {0}, h = {0}, t = {0};
        hyperlink(&t, u, u);
        th_fg(&l, T_ACCENT, buf_str(&t));
        text(buf_str(&l), 1, width, &ls);
        t.len = 0;
        hyperlink(&t, "Cmd+click to open", u);
        th_fg(&h, T_DIM, buf_str(&t));
        buf_addc(&h, ' ');
        th_fgz(&h, T_DIM, "\xe2\x80\xa2");
        buf_addc(&h, ' ');
        if (SI.hint) th_fgz(&h, SI.hint_ok ? T_SUCCESS : T_ERROR, SI.hint);
        else key_hint(&h, "ctrl+x", "to copy");
        text(buf_str(&h), 1, width, &ls);
        buf_free(&l);
        buf_free(&h);
        buf_free(&t);
        lines_addz(&ls, "");
        themed_text(T_MUTED, "If the browser runs on another machine, paste the URL it was redirected to:", width, &ls);
        input_render(width, &ls);
        key_hint(&footer, CONFIRM_KEYS, "submit");
        buf_adds(&footer, " \xe2\x80\xa2 ");
        key_hint(&footer, CANCEL_KEYS, "cancel");
    } else {
        themed_text(T_MUTED, msg ? msg : "", width, &ls);
        key_hint(&footer, CANCEL_KEYS, "cancel");
    }
    frame_close(buf_str(&footer), width, &ls);
    buf_free(&footer);
    for (size_t i = 0; i < ls.n; i++) {
        if (tui_width(str(ls.v[i])) > width) {
            Buf c = {0};
            tui_truncate(&c, str(ls.v[i]), width, "", false);
            lines_addb(out, &c);
        } else lines_addz(out, ls.v[i]);
    }
    lines_free(&ls);
    free(title);
    free(msg);
    free(url);
    free(u);
}

static void signin_key(const Key *k) {
    char *title, *msg, *url;
    if (!code_mcp_signin_screen(&title, &msg, &url)) return;
    bool cancel = k->kind == K_ESC || (k->kind == K_CTRL && k->ch == 'c');
    if (!url) {
        if (cancel) code_mcp_signin_cancel(); // status(): the cancel key calls onCancel
    } else if (k->kind == K_ENTER) {
        Str v = str_trim(buf_str(&SI.value));
        if (v.len) {
            char *z = strndup(v.ptr, v.len);
            code_mcp_signin_submit(z);
            free(z);
        }
    } else if (cancel) code_mcp_signin_cancel();
    else if (k->kind == K_CTRL && k->ch == 'x') {
        // AuthUrlComponent.copy
        char *u = clean_url(url);
        const char *err = cmds_copy_text(str(u));
        free(u);
        free(SI.hint);
        SI.hint = strdup(err ? err : "Copied URL to clipboard");
        SI.hint_ok = !err;
    } else input_key(k);
    free(title);
    free(msg);
    free(url);
}

// ---- the manager (McpManagerView.menu and index.ts manage) ------------------------------------------

enum { MAX_VISIBLE_ITEMS = 12 };

static void menu_list(const McpMenu *m, size_t sel, SelectList *l) {
    sel_init(l, (int)MIN(m->n, (size_t)MAX_VISIBLE_ITEMS), 32, 32);
    for (size_t i = 0; i < m->n; i++) sel_add(l, m->items[i].value, m->items[i].label, m->items[i].description);
    l->selected = sel;
}

static char *mfmt_pick(const char *action, const char *name) {
    size_t n = strlen(action) + strlen(name) + 2;
    char *s = xmalloc(n);
    snprintf(s, n, "%s %s", action, name);
    return s;
}

static bool picking(void);

bool code_mcp_ui_open(void) { return picking() || code_mcp_view(NULL) != MCP_VIEW_NONE; }

// ctx.ui.select (ExtensionSelectorComponent): the server for a subcommand that names none
static struct {
    char **opts;
    size_t n, sel;
    char *action;
} P;

static void pick_close(void) {
    for (size_t i = 0; i < P.n; i++) free(P.opts[i]);
    free(P.opts);
    free(P.action);
    memset(&P, 0, sizeof P);
}

void code_mcp_ui_command(const char *args) {
    Str a = str_trim(str(args ? args : ""));
    if (a.len) {
        char *z = strndup(a.ptr, a.len);
        size_t n = 0;
        char **pick = code_mcp_subcommand(z, &n);
        if (pick) {
            pick_close();
            P.opts = pick;
            P.n = n;
            size_t al = strcspn(z, " \t");
            P.action = strndup(z, al);
        }
        free(z);
        return;
    }
    code_mcp_view_set(MCP_VIEW_SERVERS, NULL);
}

static void dyn_border(int width, Lines *out) {
    Buf b = {0}, r = {0};
    tui_repeat(&r, "\xe2\x94\x80", MAX(1, width));
    th_fg(&b, T_BORDER, buf_str(&r));
    lines_addb(out, &b);
    buf_free(&r);
}

static void pick_render(int width, Lines *out) {
    dyn_border(width, out);
    lines_addz(out, "");
    Buf t = {0}, b = {0};
    th_bold(&t, S("MCP server"));
    th_fg(&b, T_ACCENT, buf_str(&t));
    text(buf_str(&b), 1, width, out);
    buf_free(&t);
    buf_free(&b);
    lines_addz(out, "");
    const size_t max_visible = 10;
    long start = (long)P.sel - (long)max_visible / 2;
    if (start > (long)P.n - (long)max_visible) start = (long)P.n - (long)max_visible;
    if (start < 0) start = 0;
    size_t end = MIN((size_t)start + max_visible, P.n);
    for (size_t i = (size_t)start; i < end; i++) {
        Buf l = {0};
        if (i == P.sel) {
            th_fgz(&l, T_ACCENT, "\xe2\x86\x92 ");
            th_fgz(&l, T_ACCENT, P.opts[i]);
        } else {
            buf_adds(&l, "  ");
            th_fgz(&l, T_TEXT, P.opts[i]);
        }
        text(buf_str(&l), 1, width, out);
        buf_free(&l);
    }
    if (start > 0 || end < P.n) {
        Buf l = {0}, m = {0};
        buf_printf(&m, "  (%zu/%zu)", P.sel + 1, P.n);
        th_fg(&l, T_MUTED, buf_str(&m));
        text(buf_str(&l), 0, width, out);
        buf_free(&l);
        buf_free(&m);
    }
    lines_addz(out, "");
    Buf h = {0};
    key_hint(&h, "\xe2\x86\x91\xe2\x86\x93", "navigate");
    buf_adds(&h, "  ");
    key_hint(&h, CONFIRM_KEYS, "select");
    buf_adds(&h, "  ");
    key_hint(&h, CANCEL_KEYS, "cancel");
    text(buf_str(&h), 1, width, out);
    buf_free(&h);
    lines_addz(out, "");
    dyn_border(width, out);
}

static void pick_key(const Key *k) {
    if (k->kind == K_UP || (k->kind == K_CHAR && k->ch == 'k')) {
        if (P.sel > 0) P.sel--;
    } else if (k->kind == K_DOWN || (k->kind == K_CHAR && k->ch == 'j')) {
        if (P.sel + 1 < P.n) P.sel++;
    } else if (k->kind == K_ENTER) {
        char *cmd = mfmt_pick(P.action, P.opts[P.sel]);
        pick_close();
        size_t n = 0;
        char **again = code_mcp_subcommand(cmd, &n);
        for (size_t i = 0; i < n; i++) free(again[i]);
        free(again);
        free(cmd);
    } else if (k->kind == K_ESC || (k->kind == K_CTRL && k->ch == 'c')) pick_close();
}

static bool picking(void) { return P.n > 0; }

void code_mcp_ui_render(int width, Lines *out) {
    if (picking()) {
        pick_render(width, out);
        return;
    }
    if (code_mcp_view(NULL) == MCP_VIEW_SIGNIN) {
        signin_render(width, out);
        return;
    }
    McpMenu m;
    size_t sel = code_mcp_view_menu(&m);
    Lines ls = {0};
    frame_open(m.title ? m.title : "", width, &ls);
    if (m.details) themed_text(T_MUTED, m.details, width, &ls);
    if (m.error) themed_text(T_ERROR, m.error, width, &ls);
    lines_addz(&ls, "");
    Buf footer = {0};
    if (!m.n) {
        themed_text(T_MUTED, m.empty ? m.empty : "Nothing to show.", width, &ls);
        key_hint(&footer, CANCEL_KEYS, m.cancel_label ? m.cancel_label : "");
    } else {
        SelectList l;
        menu_list(&m, sel, &l);
        sel_render(&l, width, &ls);
        sel_free(&l);
        key_hint(&footer, CONFIRM_KEYS, m.confirm_label ? m.confirm_label : "");
        buf_adds(&footer, " \xe2\x80\xa2 ");
        key_hint(&footer, CANCEL_KEYS, m.cancel_label ? m.cancel_label : "");
    }
    frame_close(buf_str(&footer), width, &ls);
    buf_free(&footer);
    // render(): lines wider than the view cut
    for (size_t i = 0; i < ls.n; i++) {
        if (tui_width(str(ls.v[i])) > width) {
            Buf c = {0};
            tui_truncate(&c, str(ls.v[i]), width, "", false);
            lines_addb(out, &c);
        } else lines_addz(out, ls.v[i]);
    }
    lines_free(&ls);
    code_mcp_menu_free(&m);
}

// The menu's choice (onSelect) or its cancel (onCancel), as manage's loops go on from them.
static void menu_done(McpView kind, const char *server, const char *value) {
    switch (kind) {
    case MCP_VIEW_SERVERS:
        if (!value) code_mcp_view_set(MCP_VIEW_NONE, NULL);
        else code_mcp_view_set(MCP_VIEW_SERVER, value);
        break;
    case MCP_VIEW_SERVER:
        if (!value) code_mcp_view_set(MCP_VIEW_SERVERS, NULL);
        else if (!strcmp(value, "tools")) code_mcp_view_set(MCP_VIEW_TOOLS, server);
        else if (!strcmp(value, "exposure")) code_mcp_view_set(MCP_VIEW_EXPOSURE, server);
        else {
            // runAction, then the server menu again (a new menu: its first item selected)
            code_mcp_view_set(MCP_VIEW_SERVER, server);
            code_mcp_run_action(server, value);
        }
        break;
    case MCP_VIEW_TOOLS:
        code_mcp_view_set(MCP_VIEW_SERVER, server);
        code_mcp_choose_exposure(server, NULL); // runAction: the message cleared
        break;
    case MCP_VIEW_EXPOSURE:
        code_mcp_view_set(MCP_VIEW_SERVER, server);
        code_mcp_choose_exposure(server, value);
        break;
    default: break;
    }
}

void code_mcp_ui_key(const Key *k) {
    if (picking()) {
        pick_key(k);
        return;
    }
    if (code_mcp_view(NULL) == MCP_VIEW_SIGNIN) {
        signin_key(k);
        return;
    }
    char *server = NULL;
    McpView kind = code_mcp_view(&server);
    McpMenu m;
    size_t sel = code_mcp_view_menu(&m);
    bool cancel = k->kind == K_ESC || (k->kind == K_CTRL && k->ch == 'c');
    if (!m.n) {
        if (cancel) menu_done(kind, server, NULL);
    } else {
        SelectList l;
        menu_list(&m, sel, &l);
        if (k->kind == K_UP || k->kind == K_DOWN) {
            if (k->kind == K_UP) sel_up(&l);
            else sel_down(&l);
            const SelItem *it = sel_current(&l);
            code_mcp_view_select(it ? it->value : NULL);
        } else if (k->kind == K_ENTER) {
            const SelItem *it = sel_current(&l);
            char *v = it ? strdup(it->value) : NULL;
            if (v) menu_done(kind, server, v);
            free(v);
        } else if (cancel) menu_done(kind, server, NULL);
        sel_free(&l);
    }
    code_mcp_menu_free(&m);
    free(server);
}

bool code_mcp_ui_arg_items(SelectList *all, Lines *search) {
    static const char *const actions[] = {"login", "logout", "reconnect"};
    for (size_t i = 0; i < COUNT(actions); i++) {
        char v[32];
        snprintf(v, sizeof v, "%s ", actions[i]);
        sel_add(all, v, actions[i], NULL);
        lines_addz(search, actions[i]);
    }
    return true;
}

// ---- the views of MCP tool calls (createMcpToolRenderers) -------------------------------------------

static void replace_tabs(Buf *b, Str s) {
    for (size_t i = 0; i < s.len; i++)
        if (s.ptr[i] == '\t') buf_adds(b, "   ");
        else buf_addc(b, s.ptr[i]);
}

// at most max UTF-16 units of s, at a code point boundary
static Str head_units(Str s, size_t max) {
    size_t units = 0, i = 0;
    while (i < s.len) {
        unsigned char c = (unsigned char)s.ptr[i];
        size_t n = c < 0x80 ? 1 : c >= 0xf0 ? 4 : c >= 0xe0 ? 3 : c >= 0xc0 ? 2 : 1;
        size_t u = n == 4 ? 2 : 1;
        if (units + u > max) break;
        units += u;
        i += n;
    }
    return str_sub(s, 0, i);
}

// JSON.stringify(value, null, 2)
static void pretty(Buf *b, const JValue *v, int level) {
    if (v->kind == J_ARRAY || v->kind == J_OBJECT) {
        bool arr = v->kind == J_ARRAY;
        size_t n = arr ? v->arr.len : v->obj.len;
        if (!n) {
            buf_adds(b, arr ? "[]" : "{}");
            return;
        }
        buf_adds(b, arr ? "[\n" : "{\n");
        for (size_t i = 0; i < n; i++) {
            tui_spaces(b, 2 * (level + 1));
            if (arr) pretty(b, v->arr.items[i], level + 1);
            else {
                json_write_string(b, v->obj.items[i].key);
                buf_adds(b, ": ");
                pretty(b, v->obj.items[i].val, level + 1);
            }
            buf_adds(b, i + 1 < n ? ",\n" : "\n");
        }
        tui_spaces(b, 2 * level);
        buf_addc(b, arr ? ']' : '}');
        return;
    }
    code_oai_js_write(b, v);
}

// formatToolCallWithArgs: the title, then key=value pairs on its line (collapsed, cut to 100
// characters), or a "key: value" line each (expanded)
static void call_text(Buf *out, const char *title, const JValue *args, bool expanded) {
    Buf bt = {0};
    th_bold(&bt, str(title));
    th_fg(out, T_TOOL_TITLE, buf_str(&bt));
    buf_free(&bt);
    if (!args || json_is(args, J_NULL)) return;
    size_t n = json_is(args, J_OBJECT) ? args->obj.len : 1;
    if (!n) return;
    Buf body = {0};
    for (size_t i = 0; i < n; i++) {
        Str key = json_is(args, J_OBJECT) ? args->obj.items[i].key : S("args");
        const JValue *val = json_is(args, J_OBJECT) ? args->obj.items[i].val : args;
        if (expanded) {
            Buf v = {0}, t = {0};
            if (json_is(val, J_STRING)) tui_strip(&v, val->s); // escape sequences never reach the terminal
            else pretty(&v, val, 0);
            replace_tabs(&t, buf_str(&v));
            if (i) buf_addc(&body, '\n');
            buf_adds(&body, "  ");
            buf_addstr(&body, key);
            buf_adds(&body, ": ");
            Str ts = buf_str(&t);
            for (size_t k = 0; k < ts.len; k++) {
                if (ts.ptr[k] == '\r') continue;
                if (ts.ptr[k] == '\n') buf_adds(&body, "\n    ");
                else buf_addc(&body, ts.ptr[k]);
            }
            buf_free(&v);
            buf_free(&t);
        } else {
            if (i) buf_addc(&body, ' ');
            buf_addstr(&body, key);
            buf_addc(&body, '=');
            code_oai_js_write(&body, val);
        }
    }
    if (expanded) {
        buf_addc(out, '\n');
        th_fg(out, T_MUTED, buf_str(&body));
    } else {
        Str pairs = buf_str(&body);
        Buf pv = {0};
        if (tui_js_len(pairs) > 100) {
            buf_addstr(&pv, head_units(pairs, 97));
            buf_adds(&pv, "...");
        } else buf_addstr(&pv, pairs);
        buf_addc(out, ' ');
        th_fg(out, T_MUTED, buf_str(&pv));
        buf_free(&pv);
    }
    buf_free(&body);
}

void code_mcp_ui_render_tool(const char *name, const JValue *args, const JValue *result, Str output, bool has_result, bool is_error,
                             bool expanded, int width, Lines *out) {
    char *label = code_mcp_tool_label(name);
    Buf call = {0};
    call_text(&call, label ? label : name, args, expanded);
    free(label);
    text(buf_str(&call), 0, width, out);
    buf_free(&call);
    if (!has_result) return;
    Str o = str_trim(output);
    if (!o.len) return;
    Buf rt = {0}, styled = {0};
    replace_tabs(&rt, o);
    Str r = buf_str(&rt);
    for (size_t i = 0;;) {
        size_t j = i;
        while (j < r.len && r.ptr[j] != '\n') j++;
        if (i) buf_addc(&styled, '\n');
        th_fg(&styled, is_error ? T_ERROR : T_TOOL_OUTPUT, str_sub(r, i, j));
        if (j >= r.len) break;
        i = j + 1;
    }
    lines_addz(out, ""); // Spacer(1)
    if (expanded) text(buf_str(&styled), 0, width, out);
    else {
        // VisualLinePreview: the first 5 wrapped lines, then the hint
        Lines vis = {0};
        text(buf_str(&styled), 0, width, &vis);
        const size_t keep = 5;
        size_t shown = MIN(vis.n, keep);
        for (size_t i = 0; i < shown; i++) lines_addz(out, vis.v[i]);
        if (vis.n > keep) {
            Buf h = {0}, m = {0}, cut = {0};
            buf_printf(&m, "... (%zu more lines,", vis.n - keep);
            th_fg(&h, T_MUTED, buf_str(&m));
            buf_addc(&h, ' ');
            key_hint(&h, "ctrl+o", "to expand");
            th_fgz(&h, T_MUTED, ")");
            tui_truncate(&cut, buf_str(&h), width, "...", false);
            lines_addb(out, &cut);
            buf_free(&h);
            buf_free(&m);
        }
        lines_free(&vis);
        const char *full = json_str(json_get(json_get(result, "details"), "fullOutputPath"));
        if (full) {
            Buf f = {0}, fm = {0};
            buf_printf(&fm, "Full output: %s", full);
            th_fg(&f, T_MUTED, buf_str(&fm));
            text(buf_str(&f), 0, width, out);
            buf_free(&f);
            buf_free(&fm);
        }
    }
    buf_free(&rt);
    buf_free(&styled);
}
