// Saetta Code's interactive mode: pi's screen (header, transcript, editor, footer), its message
// and tool views, the selectors, and the event loop that ties the terminal to the agent thread
// (see code_tui.h).
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at v1.5.2,
// b4afddd): packages/coding-agent/src/modes/interactive/interactive-mode.ts (layout, handleEvent,
// the submit handler and slash commands, key actions), chat-viewport.ts, components/
// (user-message, assistant-message, tool-execution, footer, thinking-selector, dynamic-border,
// status-indicator, pi-logo, keybinding-hints), core/tools/renderers/ (read, bash, edit, write,
// grep, find, ls), core/tools/render-utils.ts, components/diff.ts and visual-truncate.ts,
// core/compaction/compaction.ts (the context estimate the footer shows).
#include "code_tui.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

// ---- the event queue (agent thread -> UI thread) -----------------------------------------------

typedef struct QItem {
    char *json;
    bool update; // a message_update: a newer one replaces it
    struct QItem *next;
} QItem;

static pthread_mutex_t g_qlock = PTHREAD_MUTEX_INITIALIZER;
static QItem *g_qhead, *g_qtail;
static int g_wake[2] = {-1, -1};

static void q_push(char *json, bool update) {
    pthread_mutex_lock(&g_qlock);
    if (update && g_qtail && g_qtail->update) {
        free(g_qtail->json);
        g_qtail->json = json;
    } else {
        QItem *it = xcalloc(1, sizeof *it);
        it->json = json;
        it->update = update;
        if (g_qtail) g_qtail->next = it;
        else g_qhead = it;
        g_qtail = it;
    }
    pthread_mutex_unlock(&g_qlock);
    if (g_wake[1] >= 0) (void)!write(g_wake[1], "x", 1);
}

static QItem *q_take_all(void) {
    pthread_mutex_lock(&g_qlock);
    QItem *h = g_qhead;
    g_qhead = g_qtail = NULL;
    pthread_mutex_unlock(&g_qlock);
    return h;
}

void code_ui_event(const JValue *event) {
    Buf b = {0};
    json_write(&b, event);
    const char *t = json_str(json_get(event, "type"));
    q_push(b.ptr ? b.ptr : strdup("{}"), t && !strcmp(t, "message_update"));
}

static void q_post(const char *json) { q_push(strdup(json), false); }

// ---- small helpers ---------------------------------------------------------------------------------

static char *dupz(const char *s) { return s ? strdup(s) : NULL; }

// A JSON value copied into an arena (through its text).
static JValue *jcopy(Arena *a, const JValue *v) {
    if (!v) return NULL;
    Buf b = {0};
    json_write(&b, v);
    JValue *out = NULL;
    if (json_parse(a, buf_str(&b), true, &out)) out = NULL;
    buf_free(&b);
    return out;
}

static bool jtrue(const JValue *v) { return json_is(v, J_BOOL) && v->b; }

static double jnum(const JValue *v) {
    if (json_is(v, J_INTEGER)) return (double)v->i;
    if (json_is(v, J_FLOAT)) return v->f;
    if (json_is(v, J_NUMBER_STRING)) {
        char tmp[64];
        size_t n = MIN(v->s.len, sizeof tmp - 1);
        memcpy(tmp, v->s.ptr, n);
        tmp[n] = 0;
        return strtod(tmp, NULL);
    }
    return 0;
}

static bool jhas_num(const JValue *v) { return json_is(v, J_INTEGER) || json_is(v, J_FLOAT) || json_is(v, J_NUMBER_STRING); }

static void blank_line(Lines *out, int width) {
    Buf b = {0};
    tui_spaces(&b, width);
    lines_addb(out, &b);
}

// ---- the state -------------------------------------------------------------------------------------

typedef struct {
    char role; // 's' system, 'u' user, 'a' assistant, 't' tool result
    int64_t est;
    bool valid_usage;
    double input, output, cache_read, cache_write, total, cost;
} MsgStat;

typedef struct ToolView ToolView;
typedef struct AsstView AsstView;

typedef enum { SEL_NONE, SEL_THINKING, SEL_MODEL, SEL_SETTINGS, SEL_TREE } SelKind;

typedef struct {
    Buf value;
    size_t cursor;
} Input;

typedef struct {
    const CodeUiConfig *cfg;
    const CodeUiOps *ops;
    CodeUiHost *host;
    Screen scr;
    Comp *header, *chat, *pending, *status;
    Editor *ed;
    bool header_expanded, tools_expanded, hide_thinking;
    char *thinking;
    VEC(MsgStat) stats;
    // the running turn
    bool busy, working;
    AsstView *streaming;
    VEC(ToolView *) pending_tools;
    bool bug_hint_shown;
    uint64_t last_ctrl_c_ns, last_esc_ns;
    bool quit;
    // the status line at the end of the chat (showStatus)
    Comp *last_status_spacer, *last_status_text;
    // a selector in place of the editor
    SelKind sel;
    SelectList list;
    Input search;
    void *sel_data;
    // the agent thread
    pthread_t thread;
    pthread_mutex_t lock;
    pthread_cond_t cond;
    char *job;
    bool thread_quit;
    VEC(char *) queued; // prompts typed while a turn runs
    char *default_thinking;
} Ui;

static Ui *U;

static void request_render(void);

// ---- text components (pi's Text and ThemedText) ---------------------------------------------------

static void chat_add(Comp *c) { comp_add(U->chat, c); }

static Comp *themed(ThemeColor c, const char *text, int px, int py) {
    Buf b = {0};
    th_fgz(&b, c, text);
    Comp *t = comp_text(buf_str(&b), px, py);
    buf_free(&b);
    return t;
}

static void show_status(const char *msg) {
    CompVec *kids = comp_children(U->chat);
    Buf b = {0};
    th_fgz(&b, T_DIM, msg);
    if (kids->len >= 2 && kids->ptr[kids->len - 1] == U->last_status_text && kids->ptr[kids->len - 2] == U->last_status_spacer) {
        comp_text_set(U->last_status_text, buf_str(&b));
    } else {
        U->last_status_spacer = comp_spacer(1);
        U->last_status_text = comp_text(buf_str(&b), 1, 0);
        chat_add(U->last_status_spacer);
        chat_add(U->last_status_text);
    }
    buf_free(&b);
}

static void show_error(const char *msg) {
    chat_add(comp_spacer(1));
    Buf b = {0};
    buf_printf(&b, "Error: %s", msg);
    chat_add(themed(T_ERROR, b.ptr, 1, 0));
    buf_free(&b);
}

static void show_warning(const char *msg) {
    chat_add(comp_spacer(1));
    Buf b = {0};
    buf_printf(&b, "Warning: %s", msg);
    chat_add(themed(T_WARNING, b.ptr, 1, 0));
    buf_free(&b);
}

// ---- header (BuiltInHeader, pi-logo.ts) --------------------------------------------------------------

static const char *const app_version = "1.5.2";

static void key_hint(Buf *b, const char *key, const char *desc) {
    th_fgz(b, T_DIM, key);
    Buf d = {0};
    buf_printf(&d, " %s", desc);
    th_fg(b, T_MUTED, buf_str(&d));
    buf_free(&d);
}

static void header_render(void *ctx, int width, Lines *out) {
    (void)ctx;
    Buf t = {0};
    // the logo: coral, blue and yellow half blocks
    buf_adds(&t, "\x1b[38;2;228;138;122m\x1b[48;2;79;142;179m\xe2\x96\x80\x1b[0m\x1b[38;2;228;138;122m\xe2\x96\x80\xe2\x96\x88\x1b[0m ");
    buf_addc(&t, ' ');
    Buf v = {0};
    buf_printf(&v, "v%s", app_version);
    th_fg(&t, T_DIM, buf_str(&v));
    buf_free(&v);
    buf_addc(&t, '\n');
    buf_adds(&t, "\x1b[38;2;79;142;179m\xe2\x96\x88\xe2\x96\x80\x1b[0m \x1b[38;2;234;182;93m\xe2\x96\x88\x1b[0m ");
    if (!U->header_expanded) {
        Buf sep = {0};
        th_fgz(&sep, T_MUTED, " \xc2\xb7 ");
        key_hint(&t, "escape", "interrupt");
        buf_addstr(&t, buf_str(&sep));
        key_hint(&t, "ctrl+c/ctrl+d", "clear/exit");
        buf_addstr(&t, buf_str(&sep));
        key_hint(&t, "/", "commands");
        buf_addstr(&t, buf_str(&sep));
        key_hint(&t, "!", "bash");
        buf_addstr(&t, buf_str(&sep));
        key_hint(&t, "ctrl+o", "more");
        buf_free(&sep);
        buf_addc(&t, '\n');
        th_fgz(&t, T_DIM, "Press ctrl+o to show full startup help and loaded resources.");
        buf_adds(&t, "\n\n");
    } else {
        static const char *const hints[][2] = {
            {"escape", "to interrupt"}, {"ctrl+c", "to clear"}, {"ctrl+c twice", "to exit"}, {"ctrl+d", "to exit (empty)"},
            {"ctrl+z", "to suspend"}, {"ctrl+k", "to delete to end"}, {"shift+tab", "to cycle thinking level"},
            {"ctrl+p/shift+ctrl+p", "to cycle models"}, {"ctrl+l", "to select model"}, {"ctrl+o", "to expand tools"},
            {"ctrl+t", "to expand thinking"}, {"ctrl+g", "for external editor"}, {"/", "for commands"}, {"!", "to run bash"},
            {"!!", "to run bash (no context)"}, {"alt+enter", "to queue follow-up"}, {"alt+up", "to edit all queued messages"},
            {"ctrl+v", "to paste files on macOS, images, or text"}, {"drop files", "to attach"},
        };
        for (size_t i = 0; i < COUNT(hints); i++) {
            if (i) buf_addc(&t, '\n');
            key_hint(&t, hints[i][0], hints[i][1]);
        }
        buf_adds(&t, "\n\n");
    }
    th_fgz(&t, T_DIM, "Pi can explain its own features and look up its docs. Ask it how to use or extend Pi.");
    Comp *c = comp_text(buf_str(&t), 1, 0);
    lines_addz(out, "");
    comp_render(c, width, out);
    lines_addz(out, "");
    comp_free(c);
    buf_free(&t);
}

// ---- assistant messages (assistant-message.ts) ------------------------------------------------------

static const MdStyle md_plain = {0};
static const MdStyle md_thinking = {.has_color = true, .color = T_THINKING_TEXT, .italic = true};
static const MdStyle md_user = {.has_color = true, .color = T_USER_TEXT, .user = true};

struct AsstView {
    Comp base;
    Arena a;
    JValue *msg;
};

static void asst_render(Comp *c, int width, Lines *out) {
    AsstView *v = (AsstView *)c;
    const JValue *content = json_get(v->msg, "content");
    bool need_spacer = true, tools = false;
    size_t n = json_is(content, J_ARRAY) ? content->arr.len : 0;
    for (size_t i = 0; i < n; i++) {
        const JValue *b = content->arr.items[i];
        Str type = json_strs(json_get(b, "type"));
        if (str_eqz(type, "toolCall")) tools = true;
        if (str_eqz(type, "text")) {
            Str t = str_trim(json_strs(json_get(b, "text")));
            if (!t.len) continue;
            if (need_spacer) lines_addz(out, "");
            need_spacer = false;
            md_render(t, 1, 0, &md_plain, width, out);
        } else if (str_eqz(type, "thinking")) {
            Buf joined = {0};
            for (; i < n; i++) {
                const JValue *tb = content->arr.items[i];
                if (!str_eqz(json_strs(json_get(tb, "type")), "thinking")) break;
                Str t = str_trim(json_strs(json_get(tb, "thinking")));
                if (!t.len) continue;
                if (joined.len) buf_adds(&joined, "\n\n");
                buf_addstr(&joined, t);
            }
            i--;
            if (!joined.len) continue;
            if (need_spacer) lines_addz(out, "");
            need_spacer = false;
            if (U->hide_thinking) {
                Buf l = {0}, it = {0};
                th_fgz(&it, T_THINKING_TEXT, "Thinking...");
                buf_adds(&l, "\x1b[3m");
                buf_addstr(&l, buf_str(&it));
                buf_adds(&l, "\x1b[23m");
                Comp *tc = comp_text(buf_str(&l), 1, 0);
                comp_render(tc, width, out);
                comp_free(tc);
                buf_free(&l);
                buf_free(&it);
            } else md_render(buf_str(&joined), 1, 0, &md_thinking, width, out);
            need_spacer = true;
            buf_free(&joined);
        }
    }
    Str stop = json_strs(json_get(v->msg, "stopReason"));
    const char *err = NULL;
    Buf e = {0};
    if (str_eqz(stop, "length")) err = "Response was truncated before completion.";
    else if (!tools && str_eqz(stop, "aborted")) {
        const char *m = json_str(json_get(v->msg, "errorMessage"));
        err = m && strcmp(m, "Request was aborted") ? m : "Operation aborted";
    } else if (!tools && str_eqz(stop, "error")) {
        const char *m = json_str(json_get(v->msg, "errorMessage"));
        buf_printf(&e, "Error: %s", m && *m ? m : "Unknown error");
        err = e.ptr;
    }
    if (err) {
        lines_addz(out, "");
        Comp *t = themed(T_ERROR, err, 1, 0);
        comp_render(t, width, out);
        comp_free(t);
    }
    buf_free(&e);
}

static void asst_destroy(Comp *c) {
    arena_reset(&((AsstView *)c)->a);
    free(c);
}

static const CompVt asst_vt = {asst_render, asst_destroy};

static void asst_set(AsstView *v, const JValue *msg) {
    arena_reset(&v->a);
    v->msg = jcopy(&v->a, msg);
}

static AsstView *asst_new(const JValue *msg) {
    AsstView *v = xcalloc(1, sizeof *v);
    v->base.vt = &asst_vt;
    asst_set(v, msg);
    return v;
}

// ---- tool views (tool-execution.ts and the built-in renderers) --------------------------------------

struct ToolView {
    Comp base;
    Arena a;
    char *name, *id;
    JValue *args, *result; // result: {"content":[...],"details":{...}}
    bool has_result, partial, is_error, started, args_complete, expanded;
    bool has_duration;
    double duration_ms;
};

static Str text_output(const JValue *result) {
    // the text blocks joined by newlines, \r removed, ANSI escapes stripped
    static Buf b;
    b.len = 0;
    const JValue *c = json_get(result, "content");
    bool first = true;
    for (size_t i = 0; c && json_is(c, J_ARRAY) && i < c->arr.len; i++) {
        const JValue *bl = c->arr.items[i];
        if (!str_eqz(json_strs(json_get(bl, "type")), "text")) continue;
        if (!first) buf_addc(&b, '\n');
        first = false;
        Buf s = {0};
        tui_strip(&s, json_strs(json_get(bl, "text")));
        for (size_t k = 0; k < s.len; k++) {
            unsigned char ch = (unsigned char)s.ptr[k];
            if (ch == '\r') continue;
            // sanitizeBinaryOutput: control characters other than tab and newline go
            if (ch < 0x20 && ch != '\t' && ch != '\n') continue;
            buf_addc(&b, (char)ch);
        }
        buf_free(&s);
    }
    if (!b.ptr) buf_adds(&b, "");
    return buf_str(&b);
}

static void shorten_path(Buf *b, Str p) {
    const char *home = U->cfg->home;
    size_t hl = home ? strlen(home) : 0;
    if (hl && p.len >= hl && !memcmp(p.ptr, home, hl) && (p.len == hl || p.ptr[hl] == '/')) {
        buf_addc(b, '~');
        buf_addstr(b, str_sub(p, hl, p.len));
    } else buf_addstr(b, p);
}

// renderToolPath: the path in the accent color (an invalid argument, an empty one as "...").
static void tool_path(Buf *b, const JValue *v, const char *empty_fallback) {
    if (v && !json_is(v, J_STRING) && !json_is(v, J_NULL)) {
        th_fgz(b, T_ERROR, "[invalid arg]");
        return;
    }
    Str p = json_strs(v);
    if (!p.len && empty_fallback) p = str(empty_fallback);
    if (!p.len) {
        th_fgz(b, T_TOOL_OUTPUT, "...");
        return;
    }
    Buf s = {0};
    shorten_path(&s, p);
    th_fg(b, T_ACCENT, buf_str(&s));
    buf_free(&s);
}

static void title(Buf *b, const char *t) {
    Buf x = {0};
    buf_adds(&x, "\x1b[1m");
    buf_adds(&x, t);
    buf_adds(&x, "\x1b[22m");
    th_fg(b, T_TOOL_TITLE, buf_str(&x));
    buf_free(&x);
}

static void expand_hint(Buf *b, const char *what) {
    th_fgz(b, T_DIM, "ctrl+o");
    Buf d = {0};
    buf_printf(&d, " %s", what);
    th_fg(b, T_MUTED, buf_str(&d));
    buf_free(&d);
}

static const char *lang_from_path(Str p) {
    ptrdiff_t dot = -1;
    for (size_t i = 0; i < p.len; i++)
        if (p.ptr[i] == '.') dot = (ptrdiff_t)i;
        else if (p.ptr[i] == '/') dot = -1;
    if (dot < 0) return NULL;
    Str ext = str_sub(p, (size_t)dot + 1, p.len);
    static const char *const map[][2] = {
        {"ts", "typescript"}, {"tsx", "typescript"}, {"js", "javascript"}, {"jsx", "javascript"}, {"mjs", "javascript"},
        {"cjs", "javascript"}, {"py", "python"}, {"rb", "ruby"}, {"rs", "rust"}, {"go", "go"}, {"java", "java"},
        {"kt", "kotlin"}, {"swift", "swift"}, {"dart", "dart"}, {"groovy", "groovy"}, {"gradle", "groovy"}, {"nix", "nix"},
        {"c", "c"}, {"h", "c"}, {"cpp", "cpp"}, {"cc", "cpp"}, {"cxx", "cpp"}, {"hpp", "cpp"}, {"cs", "csharp"},
        {"php", "php"}, {"sh", "bash"}, {"bash", "bash"}, {"zsh", "bash"}, {"lua", "lua"}, {"perl", "perl"}, {"scala", "scala"},
        {"json", "json"}, {"fish", "fish"}, {"ps1", "powershell"}, {"sql", "sql"}, {"html", "html"}, {"htm", "html"},
        {"css", "css"}, {"scss", "scss"}, {"sass", "sass"}, {"less", "less"}, {"yaml", "yaml"}, {"yml", "yaml"},
        {"toml", "toml"}, {"xml", "xml"}, {"md", "markdown"}, {"markdown", "markdown"}, {"dockerfile", "dockerfile"},
        {"makefile", "makefile"}, {"cmake", "cmake"}, {"r", "r"}, {"clj", "clojure"}, {"ex", "elixir"}, {"exs", "elixir"},
        {"erl", "erlang"}, {"hs", "haskell"}, {"ml", "ocaml"}, {"vim", "vim"}, {"graphql", "graphql"}, {"proto", "protobuf"},
        {"tf", "hcl"}, {"hcl", "hcl"},
    };
    for (size_t i = 0; i < COUNT(map); i++)
        if (ext.len == strlen(map[i][0]) && !strncasecmp(ext.ptr, map[i][0], ext.len)) return map[i][1];
    return NULL;
}

static void replace_tabs(Buf *b, Str s) {
    for (size_t i = 0; i < s.len; i++)
        if (s.ptr[i] == '\t') buf_adds(b, "   ");
        else buf_addc(b, s.ptr[i]);
}

// A Text(text, 0, 0) rendered into out.
static void text_lines(Str text, int px, int width, Lines *out) {
    Comp *t = comp_text(text, px, 0);
    comp_render(t, width, out);
    comp_free(t);
}

static void read_call(ToolView *v, Buf *b) {
    const JValue *path = json_get(v->args, "file_path");
    if (!path) path = json_get(v->args, "path");
    Str p = json_strs(path);
    // compact forms for skills and context files (getCompactReadClassification)
    if (!v->expanded && p.len) {
        Str base = p;
        for (size_t i = 0; i < p.len; i++)
            if (p.ptr[i] == '/') base = str_sub(p, i + 1, p.len);
        bool skill = str_eqz(base, "SKILL.md");
        bool resource = str_eqz(base, "AGENTS.md") || str_eqz(base, "AGENTS.MD") || str_eqz(base, "AGENTS.override.md") ||
                        str_eqz(base, "CLAUDE.md") || str_eqz(base, "CLAUDE.MD");
        if (skill || resource) {
            Buf abs = {0};
            if (p.ptr[0] == '/') buf_addstr(&abs, p);
            else if (p.ptr[0] == '~' && (p.len == 1 || p.ptr[1] == '/')) {
                buf_adds(&abs, U->cfg->home);
                buf_addstr(&abs, str_sub(p, 1, p.len));
            } else {
                buf_adds(&abs, U->cfg->cwd);
                buf_addc(&abs, '/');
                buf_addstr(&abs, p);
            }
            if (skill) {
                // the folder's name
                Str a = buf_str(&abs);
                size_t e = a.len;
                while (e > 0 && a.ptr[e - 1] != '/') e--;
                size_t s = e ? e - 1 : 0;
                while (s > 0 && a.ptr[s - 1] != '/') s--;
                Str label = e > 1 ? str_sub(a, s, e - 1) : S("SKILL.md");
                th_fgz(b, T_CUSTOM_LABEL, "\x1b[1m[skill]\x1b[22m ");
                th_fg(b, T_CUSTOM_TEXT, label);
            } else {
                title(b, "read resource");
                buf_addc(b, ' ');
                Str a = buf_str(&abs);
                size_t cl = strlen(U->cfg->cwd);
                Str label = a.len > cl && !memcmp(a.ptr, U->cfg->cwd, cl) && a.ptr[cl] == '/' ? str_sub(a, cl + 1, a.len) : a;
                th_fg(b, T_ACCENT, label);
            }
            buf_free(&abs);
            goto range;
        }
    }
    title(b, "read");
    buf_addc(b, ' ');
    tool_path(b, path, NULL);
range:;
    // formatReadLineRange
    const JValue *off = json_get(v->args, "offset"), *lim = json_get(v->args, "limit");
    bool off_null = !off || json_is(off, J_NULL), lim_null = !lim || json_is(lim, J_NULL);
    if (!(off_null && lim_null)) {
        bool ok = true;
        double start = off_null ? 1 : jhas_num(off) ? jnum(off) : (ok = false, 0);
        if (ok) {
            Buf r = {0};
            if (lim_null) buf_printf(&r, ":%g", start);
            else if (jhas_num(lim)) {
                double end = start + jnum(lim) - 1;
                if (end) buf_printf(&r, ":%g-%g", start, end);
                else buf_printf(&r, ":%g", start);
            }
            if (r.len) th_fg(b, T_WARNING, buf_str(&r));
            buf_free(&r);
        }
    }
    if (!v->expanded && (str_ends(p, S("SKILL.md")) || str_ends(p, S("AGENTS.md")) || str_ends(p, S("CLAUDE.md")))) {
        Buf h = {0};
        buf_adds(&h, " (ctrl+o to expand)");
        th_fg(b, T_DIM, buf_str(&h));
        buf_free(&h);
    }
}

static void read_result(ToolView *v, Buf *b) {
    if (!v->expanded && !v->is_error) return;
    const JValue *path = json_get(v->args, "file_path");
    if (!path) path = json_get(v->args, "path");
    Str out = text_output(v->result);
    const char *lang = !v->is_error && json_str(path) ? lang_from_path(json_strs(path)) : NULL;
    Lines ls = {0};
    if (lang) {
        Buf rt = {0};
        replace_tabs(&rt, out);
        md_highlight(buf_str(&rt), str(lang), &ls);
        buf_free(&rt);
    } else {
        for (size_t i = 0;;) {
            size_t j = i;
            while (j < out.len && out.ptr[j] != '\n') j++;
            lines_add(&ls, str_sub(out, i, j));
            if (j >= out.len) break;
            i = j + 1;
        }
    }
    while (ls.n && !ls.v[ls.n - 1][0]) free(ls.v[--ls.n]);
    size_t maxl = v->expanded ? ls.n : 10;
    buf_addc(b, '\n');
    for (size_t i = 0; i < ls.n && i < maxl; i++) {
        if (i) buf_addc(b, '\n');
        Buf rt = {0};
        replace_tabs(&rt, str(ls.v[i]));
        if (lang) buf_addstr(b, buf_str(&rt));
        else th_fg(b, T_TOOL_OUTPUT, buf_str(&rt));
        buf_free(&rt);
    }
    if (ls.n > maxl) {
        Buf m = {0};
        buf_printf(&m, "\n... (%zu more lines,", ls.n - maxl);
        th_fg(b, T_MUTED, buf_str(&m));
        buf_addc(b, ' ');
        expand_hint(b, "to expand");
        th_fgz(b, T_MUTED, ")");
        buf_free(&m);
    }
    lines_free(&ls);
    const JValue *tr = json_get(json_get(v->result, "details"), "truncation");
    if (jtrue(json_get(tr, "truncated"))) {
        Buf w = {0};
        if (jtrue(json_get(tr, "firstLineExceedsLimit"))) buf_printf(&w, "[First line exceeds %gKB limit]", jnum(json_get(tr, "maxBytes")) / 1024);
        else if (str_eqz(json_strs(json_get(tr, "truncatedBy")), "lines"))
            buf_printf(&w, "[Truncated: showing %g of %g lines (%g line limit)]", jnum(json_get(tr, "outputLines")),
                       jnum(json_get(tr, "totalLines")), jhas_num(json_get(tr, "maxLines")) ? jnum(json_get(tr, "maxLines")) : 2000);
        else buf_printf(&w, "[Truncated: %g lines shown (%gKB limit)]", jnum(json_get(tr, "outputLines")),
                        (jhas_num(json_get(tr, "maxBytes")) ? jnum(json_get(tr, "maxBytes")) : 51200) / 1024);
        buf_addc(b, '\n');
        th_fg(b, T_WARNING, buf_str(&w));
        buf_free(&w);
    }
}

static void fmt_duration(Buf *b, double ms) {
    double s = ms / 1000;
    if (s < 60) {
        buf_printf(b, "%.1fs", s);
        return;
    }
    long total = (long)floor(s), m = total / 60, r = total % 60;
    if (m < 60) buf_printf(b, "%ldm %lds", m, r);
    else buf_printf(b, "%ldh %ldm %lds", m / 60, m % 60, r);
}

static void bash_render_body(ToolView *v, int width, Lines *out) {
    // the call
    Buf call = {0};
    {
        const JValue *cmd = json_get(v->args, "command");
        Buf t = {0};
        buf_adds(&t, "$ ");
        if (cmd && !json_is(cmd, J_STRING) && !json_is(cmd, J_NULL)) th_fgz(&t, T_ERROR, "[invalid arg]");
        else if (json_strs(cmd).len) buf_addstr(&t, json_strs(cmd));
        else th_fgz(&t, T_TOOL_OUTPUT, "...");
        title(&call, t.ptr);
        buf_free(&t);
        const JValue *to = json_get(v->args, "timeout");
        if (jhas_num(to) && jnum(to) != 0) {
            Buf s = {0};
            buf_printf(&s, " (timeout %gs)", jnum(to));
            th_fg(&call, T_MUTED, buf_str(&s));
            buf_free(&s);
        }
    }
    text_lines(buf_str(&call), 0, width, out);
    buf_free(&call);
    if (!v->has_result) return;
    Str o = str_trim(text_output(v->result));
    const JValue *det = json_get(v->result, "details");
    const JValue *tr = json_get(det, "truncation");
    const char *full = json_str(json_get(det, "fullOutputPath"));
    Buf output = {0};
    buf_addstr(&output, o);
    if (!v->partial && jtrue(json_get(tr, "truncated")) && full && str_ends(buf_str(&output), S("]"))) {
        Str os = buf_str(&output);
        ptrdiff_t fs = -1;
        for (ptrdiff_t k = (ptrdiff_t)os.len - 3; k >= 0; k--)
            if (os.ptr[k] == '\n' && os.ptr[k + 1] == '\n' && os.ptr[k + 2] == '[') {
                fs = k;
                break;
            }
        if (fs >= 0 && str_find(str_sub(os, (size_t)fs, os.len), str(full), 0) >= 0) {
            output.len = (size_t)fs;
            while (output.len && isspace((unsigned char)output.ptr[output.len - 1])) output.len--;
            output.ptr[output.len] = 0;
        }
    }
    if (output.len) {
        Buf styled = {0};
        Str os = buf_str(&output);
        for (size_t i = 0;;) {
            size_t j = i;
            while (j < os.len && os.ptr[j] != '\n') j++;
            if (i) buf_addc(&styled, '\n');
            th_fg(&styled, T_TOOL_OUTPUT, str_sub(os, i, j));
            if (j >= os.len) break;
            i = j + 1;
        }
        if (v->expanded) {
            Buf t = {0};
            buf_addc(&t, '\n');
            buf_addstr(&t, buf_str(&styled));
            text_lines(buf_str(&t), 0, width, out);
            buf_free(&t);
        } else {
            lines_addz(out, "");
            Lines vis = {0};
            text_lines(buf_str(&styled), 0, width, &vis);
            size_t keep = 5;
            if (vis.n > keep) {
                Buf h = {0}, m = {0}, hint = {0};
                buf_printf(&m, "... (%zu earlier lines,", vis.n - keep);
                th_fg(&h, T_MUTED, buf_str(&m));
                buf_addc(&h, ' ');
                expand_hint(&h, "to expand");
                th_fgz(&h, T_MUTED, ")");
                tui_truncate(&hint, buf_str(&h), width, "...", false);
                lines_add(out, buf_str(&hint));
                for (size_t i = vis.n - keep; i < vis.n; i++) lines_addz(out, vis.v[i]);
                buf_free(&h);
                buf_free(&m);
                buf_free(&hint);
            } else
                for (size_t i = 0; i < vis.n; i++) lines_addz(out, vis.v[i]);
            lines_free(&vis);
        }
        buf_free(&styled);
    }
    if (jtrue(json_get(tr, "truncated")) || full) {
        Buf w = {0};
        if (full) buf_printf(&w, "Full output: %s", full);
        if (jtrue(json_get(tr, "truncated"))) {
            if (w.len) buf_adds(&w, ". ");
            if (str_eqz(json_strs(json_get(tr, "truncatedBy")), "lines"))
                buf_printf(&w, "Truncated: showing %g of %g lines", jnum(json_get(tr, "outputLines")), jnum(json_get(tr, "totalLines")));
            else
                buf_printf(&w, "Truncated: %g lines shown (%gKB limit)", jnum(json_get(tr, "outputLines")),
                           (jhas_num(json_get(tr, "maxBytes")) ? jnum(json_get(tr, "maxBytes")) : 51200) / 1024);
        }
        Buf t = {0}, br = {0};
        buf_printf(&br, "[%s]", w.ptr);
        buf_addc(&t, '\n');
        th_fg(&t, T_WARNING, buf_str(&br));
        text_lines(buf_str(&t), 0, width, out);
        buf_free(&w);
        buf_free(&t);
        buf_free(&br);
    }
    if (!v->partial && v->has_duration) {
        Buf d = {0}, t = {0};
        buf_adds(&d, "Took ");
        fmt_duration(&d, v->duration_ms);
        buf_addc(&t, '\n');
        th_fg(&t, T_MUTED, buf_str(&d));
        text_lines(buf_str(&t), 0, width, out);
        buf_free(&d);
        buf_free(&t);
    }
    buf_free(&output);
}

static void write_call(ToolView *v, Buf *b) {
    const JValue *path = json_get(v->args, "file_path");
    if (!path) path = json_get(v->args, "path");
    const JValue *content = json_get(v->args, "content");
    title(b, "write");
    buf_addc(b, ' ');
    tool_path(b, path, NULL);
    if (content && !json_is(content, J_STRING) && !json_is(content, J_NULL)) {
        buf_adds(b, "\n\n");
        th_fgz(b, T_ERROR, "[invalid content arg - expected string]");
        return;
    }
    Str c = json_strs(content);
    if (!c.len) return;
    const char *lang = json_str(path) ? lang_from_path(json_strs(path)) : NULL;
    Buf norm = {0};
    for (size_t i = 0; i < c.len; i++)
        if (c.ptr[i] != '\r') buf_addc(&norm, c.ptr[i]);
    Lines ls = {0};
    if (lang) {
        Buf rt = {0};
        replace_tabs(&rt, buf_str(&norm));
        md_highlight(buf_str(&rt), str(lang), &ls);
        buf_free(&rt);
    } else {
        Str s = buf_str(&norm);
        for (size_t i = 0;;) {
            size_t j = i;
            while (j < s.len && s.ptr[j] != '\n') j++;
            lines_add(&ls, str_sub(s, i, j));
            if (j >= s.len) break;
            i = j + 1;
        }
    }
    while (ls.n && !ls.v[ls.n - 1][0]) free(ls.v[--ls.n]);
    size_t maxl = v->expanded ? ls.n : 10;
    buf_adds(b, "\n\n");
    for (size_t i = 0; i < ls.n && i < maxl; i++) {
        if (i) buf_addc(b, '\n');
        if (lang) buf_adds(b, ls.v[i]);
        else {
            Buf rt = {0};
            replace_tabs(&rt, str(ls.v[i]));
            th_fg(b, T_TOOL_OUTPUT, buf_str(&rt));
            buf_free(&rt);
        }
    }
    if (ls.n > maxl) {
        Buf m = {0};
        buf_printf(&m, "\n... (%zu more lines, %zu total,", ls.n - maxl, ls.n);
        th_fg(b, T_MUTED, buf_str(&m));
        buf_addc(b, ' ');
        expand_hint(b, "to expand");
        th_fgz(b, T_MUTED, ")");
        buf_free(&m);
    }
    lines_free(&ls);
    buf_free(&norm);
}

// renderDiff (diff.ts): word-level changes in inverse for a single changed line.
static bool parse_diff_line(Str l, char *prefix, Str *num, Str *content) {
    if (!l.len) return false;
    char p = l.ptr[0];
    if (p != '+' && p != '-' && !isspace((unsigned char)p)) return false;
    size_t i = 1;
    size_t ns = i;
    while (i < l.len && isspace((unsigned char)l.ptr[i]) && l.ptr[i] != '\n') i++;
    while (i < l.len && isdigit((unsigned char)l.ptr[i])) i++;
    // (\s*\d*)\s(.*): the number group may be followed by exactly one whitespace
    if (i >= l.len || !isspace((unsigned char)l.ptr[i])) {
        // backtrack: \s*\d* may give one space back
        if (i > ns && isspace((unsigned char)l.ptr[i - 1])) i--;
        else return false;
    }
    *prefix = p;
    *num = str_sub(l, ns, i);
    *content = str_sub(l, i + 1, l.len);
    return true;
}

typedef struct {
    Str s;
    bool ws;
} Word;
typedef VEC(Word) WordVec;

static void split_words(Str s, WordVec *out) {
    for (size_t i = 0; i < s.len;) {
        size_t j = i;
        bool ws = isspace((unsigned char)s.ptr[i]);
        bool word = isalnum((unsigned char)s.ptr[i]) || s.ptr[i] == '_' || (unsigned char)s.ptr[i] >= 0x80;
        if (ws)
            while (j < s.len && isspace((unsigned char)s.ptr[j])) j++;
        else if (word)
            while (j < s.len && (isalnum((unsigned char)s.ptr[j]) || s.ptr[j] == '_' || (unsigned char)s.ptr[j] >= 0x80)) j++;
        else j++;
        vec_push(out, ((Word){str_sub(s, i, j), ws}));
        i = j;
    }
}

static void intra_line(Str old, Str nw, Buf *rem, Buf *add) {
    size_t ci = 0;
    while (ci < old.len && ci < nw.len && old.ptr[ci] == nw.ptr[ci] && isspace((unsigned char)old.ptr[ci])) ci++;
    buf_addstr(rem, str_sub(old, 0, ci));
    buf_addstr(add, str_sub(nw, 0, ci));
    WordVec a = {0}, b = {0};
    split_words(str_sub(old, ci, old.len), &a);
    split_words(str_sub(nw, ci, nw.len), &b);
    // longest common subsequence of words
    size_t n = a.len, m = b.len;
    uint16_t *dp = xcalloc((n + 1) * (m + 1), sizeof *dp);
    for (size_t i = n; i-- > 0;)
        for (size_t j = m; j-- > 0;)
            dp[i * (m + 1) + j] = str_eq(a.ptr[i].s, b.ptr[j].s) ? dp[(i + 1) * (m + 1) + j + 1] + 1
                                                                  : MAX(dp[(i + 1) * (m + 1) + j], dp[i * (m + 1) + j + 1]);
    bool first_rem = true, first_add = true;
    size_t i = 0, j = 0;
    Buf rr = {0}, aa = {0};
#define FLUSH_REM()                                                                    \
    do {                                                                               \
        if (rr.len) {                                                                  \
            Str v_ = buf_str(&rr);                                                     \
            if (first_rem) {                                                           \
                size_t k_ = 0;                                                         \
                while (k_ < v_.len && isspace((unsigned char)v_.ptr[k_])) k_++;        \
                buf_addstr(rem, str_sub(v_, 0, k_));                                   \
                v_ = str_sub(v_, k_, v_.len);                                          \
                first_rem = false;                                                     \
            }                                                                          \
            if (v_.len) th_inverse(rem, v_);                                           \
            rr.len = 0;                                                                \
        }                                                                              \
    } while (0)
#define FLUSH_ADD()                                                                    \
    do {                                                                               \
        if (aa.len) {                                                                  \
            Str v_ = buf_str(&aa);                                                     \
            if (first_add) {                                                           \
                size_t k_ = 0;                                                         \
                while (k_ < v_.len && isspace((unsigned char)v_.ptr[k_])) k_++;        \
                buf_addstr(add, str_sub(v_, 0, k_));                                   \
                v_ = str_sub(v_, k_, v_.len);                                          \
                first_add = false;                                                     \
            }                                                                          \
            if (v_.len) th_inverse(add, v_);                                           \
            aa.len = 0;                                                                \
        }                                                                              \
    } while (0)
    while (i < n || j < m) {
        if (i < n && j < m && str_eq(a.ptr[i].s, b.ptr[j].s)) {
            FLUSH_REM();
            FLUSH_ADD();
            buf_addstr(rem, a.ptr[i].s);
            buf_addstr(add, b.ptr[j].s);
            i++;
            j++;
        } else if (j < m && (i >= n || dp[i * (m + 1) + j + 1] >= dp[(i + 1) * (m + 1) + j])) {
            buf_addstr(&aa, b.ptr[j++].s);
        } else {
            buf_addstr(&rr, a.ptr[i++].s);
        }
    }
    FLUSH_REM();
    FLUSH_ADD();
#undef FLUSH_REM
#undef FLUSH_ADD
    free(dp);
    vec_free(&a);
    vec_free(&b);
    buf_free(&rr);
    buf_free(&aa);
}

static void render_diff(Str diff, Buf *out) {
    VEC(Str) ls = {0};
    for (size_t i = 0;;) {
        size_t j = i;
        while (j < diff.len && diff.ptr[j] != '\n') j++;
        vec_push(&ls, str_sub(diff, i, j));
        if (j >= diff.len) break;
        i = j + 1;
    }
    bool first = true;
#define NL()                         \
    do {                             \
        if (!first) buf_addc(out, '\n'); \
        first = false;               \
    } while (0)
    for (size_t i = 0; i < ls.len;) {
        char p;
        Str num, content;
        if (!parse_diff_line(ls.ptr[i], &p, &num, &content)) {
            NL();
            th_fg(out, T_DIFF_CONTEXT, ls.ptr[i]);
            i++;
            continue;
        }
        if (p == '-') {
            VEC(Str) rn = {0}, rc = {0}, an = {0}, ac = {0};
            while (i < ls.len && parse_diff_line(ls.ptr[i], &p, &num, &content) && p == '-') {
                vec_push(&rn, num);
                vec_push(&rc, content);
                i++;
            }
            while (i < ls.len && parse_diff_line(ls.ptr[i], &p, &num, &content) && p == '+') {
                vec_push(&an, num);
                vec_push(&ac, content);
                i++;
            }
            if (rn.len == 1 && an.len == 1) {
                Buf ro = {0}, ao = {0}, rt = {0}, at = {0}, rl = {0}, al = {0};
                replace_tabs(&ro, rc.ptr[0]);
                replace_tabs(&ao, ac.ptr[0]);
                intra_line(buf_str(&ro), buf_str(&ao), &rt, &at);
                buf_printf(&rl, "-%.*s ", (int)rn.ptr[0].len, rn.ptr[0].ptr);
                buf_addstr(&rl, buf_str(&rt));
                buf_printf(&al, "+%.*s ", (int)an.ptr[0].len, an.ptr[0].ptr);
                buf_addstr(&al, buf_str(&at));
                NL();
                th_fg(out, T_DIFF_REMOVED, buf_str(&rl));
                NL();
                th_fg(out, T_DIFF_ADDED, buf_str(&al));
                buf_free(&ro);
                buf_free(&ao);
                buf_free(&rt);
                buf_free(&at);
                buf_free(&rl);
                buf_free(&al);
            } else {
                for (size_t k = 0; k < rn.len; k++) {
                    Buf l = {0};
                    buf_printf(&l, "-%.*s ", (int)rn.ptr[k].len, rn.ptr[k].ptr);
                    replace_tabs(&l, rc.ptr[k]);
                    NL();
                    th_fg(out, T_DIFF_REMOVED, buf_str(&l));
                    buf_free(&l);
                }
                for (size_t k = 0; k < an.len; k++) {
                    Buf l = {0};
                    buf_printf(&l, "+%.*s ", (int)an.ptr[k].len, an.ptr[k].ptr);
                    replace_tabs(&l, ac.ptr[k]);
                    NL();
                    th_fg(out, T_DIFF_ADDED, buf_str(&l));
                    buf_free(&l);
                }
            }
            vec_free(&rn);
            vec_free(&rc);
            vec_free(&an);
            vec_free(&ac);
        } else {
            Buf l = {0};
            buf_printf(&l, "%c%.*s ", p == '+' ? '+' : ' ', (int)num.len, num.ptr);
            replace_tabs(&l, content);
            NL();
            th_fg(out, p == '+' ? T_DIFF_ADDED : T_DIFF_CONTEXT, buf_str(&l));
            buf_free(&l);
            i++;
        }
    }
#undef NL
    vec_free(&ls);
}

// The list renderers (grep, find, ls): the output's first lines, then the truncation notes.
static void list_result(ToolView *v, Buf *b, size_t max_collapsed, const char *limit_key, const char *limit_label, bool lines_trunc) {
    Str o = str_trim(text_output(v->result));
    if (o.len) {
        VEC(Str) ls = {0};
        for (size_t i = 0;;) {
            size_t j = i;
            while (j < o.len && o.ptr[j] != '\n') j++;
            vec_push(&ls, str_sub(o, i, j));
            if (j >= o.len) break;
            i = j + 1;
        }
        size_t maxl = v->expanded ? ls.len : max_collapsed;
        buf_addc(b, '\n');
        for (size_t i = 0; i < ls.len && i < maxl; i++) {
            if (i) buf_addc(b, '\n');
            th_fg(b, T_TOOL_OUTPUT, ls.ptr[i]);
        }
        if (ls.len > maxl) {
            Buf m = {0};
            buf_printf(&m, "\n... (%zu more lines,", ls.len - maxl);
            th_fg(b, T_MUTED, buf_str(&m));
            buf_addc(b, ' ');
            expand_hint(b, "to expand");
            th_fgz(b, T_MUTED, ")");
            buf_free(&m);
        }
        vec_free(&ls);
    }
    const JValue *det = json_get(v->result, "details");
    const JValue *lim = json_get(det, limit_key);
    const JValue *tr = json_get(det, "truncation");
    bool lt = lines_trunc && jtrue(json_get(det, "linesTruncated"));
    if ((lim && !json_is(lim, J_NULL) && !(json_is(lim, J_BOOL) && !lim->b)) || jtrue(json_get(tr, "truncated")) || lt) {
        Buf w = {0};
        if (lim && jhas_num(lim)) buf_printf(&w, "%g %s", jnum(lim), limit_label);
        if (jtrue(json_get(tr, "truncated"))) {
            if (w.len) buf_adds(&w, ", ");
            buf_printf(&w, "%gKB limit", (jhas_num(json_get(tr, "maxBytes")) ? jnum(json_get(tr, "maxBytes")) : 51200) / 1024);
        }
        if (lt) {
            if (w.len) buf_adds(&w, ", ");
            buf_adds(&w, "some lines truncated");
        }
        Buf t = {0};
        buf_printf(&t, "[Truncated: %s]", w.ptr ? w.ptr : "");
        buf_addc(b, '\n');
        th_fg(b, T_WARNING, buf_str(&t));
        buf_free(&w);
        buf_free(&t);
    }
}

static void box_lines(Lines *kids, int width, int pad_x, int pad_y, Lines *out) {
    if (!kids->n) return;
    for (int k = 0; k < pad_y; k++) blank_line(out, width);
    for (size_t i = 0; i < kids->n; i++) {
        Buf l = {0};
        tui_spaces(&l, pad_x);
        buf_adds(&l, kids->v[i]);
        tui_spaces(&l, width - tui_width(buf_str(&l)));
        lines_addb(out, &l);
    }
    for (int k = 0; k < pad_y; k++) blank_line(out, width);
}

static void tool_render(Comp *c, int width, Lines *out) {
    ToolView *v = (ToolView *)c;
    const char *n = v->name;
    int cw = MAX(1, width - 2);
    Lines kids = {0};
    if (!strcmp(n, "edit")) {
        // renderShell "self": a blank line, then the call box (header, the diff) and the result
        Buf head = {0};
        title(&head, "edit");
        buf_addc(&head, ' ');
        const JValue *path = json_get(v->args, "file_path");
        if (!path) path = json_get(v->args, "path");
        tool_path(&head, path, NULL);
        text_lines(buf_str(&head), 0, cw, &kids);
        buf_free(&head);
        const char *diff = v->has_result && !v->is_error ? json_str(json_get(json_get(v->result, "details"), "diff")) : NULL;
        if (diff) {
            lines_addz(&kids, "");
            Buf d = {0};
            render_diff(str(diff), &d);
            text_lines(buf_str(&d), 0, cw, &kids);
            buf_free(&d);
        }
        lines_addz(out, "");
        box_lines(&kids, width, 1, 1, out);
        lines_free(&kids);
        if (v->has_result && v->is_error) {
            Str err = text_output(v->result);
            if (err.len) {
                lines_addz(out, "");
                Buf e = {0};
                th_fg(&e, T_ERROR, err);
                text_lines(buf_str(&e), 1, width, out);
                buf_free(&e);
            }
        }
        return;
    }
    bool known = !strcmp(n, "read") || !strcmp(n, "bash") || !strcmp(n, "write") || !strcmp(n, "grep") || !strcmp(n, "find") ||
                 !strcmp(n, "ls");
    lines_addz(out, "");
    if (!known) {
        // no renderer: the name, the arguments as JSON, the output (formatToolExecution)
        Buf t = {0};
        title(&t, n);
        if (v->args) {
            buf_adds(&t, "\n\n");
            // JSON.stringify(args, null, 2)
            Buf j = {0};
            json_write(&j, v->args);
            // indent by re-parsing is costly: a compact form is close enough for unknown tools
            buf_addstr(&t, buf_str(&j));
            buf_free(&j);
        }
        Str o = text_output(v->result);
        if (v->has_result && o.len) {
            buf_addc(&t, '\n');
            buf_addstr(&t, o);
        }
        Comp *tc = comp_text(buf_str(&t), 1, 1);
        comp_render(tc, width, out);
        comp_free(tc);
        buf_free(&t);
        return;
    }
    if (!strcmp(n, "bash")) bash_render_body(v, cw, &kids);
    else {
        Buf call = {0}, res = {0};
        if (!strcmp(n, "read")) {
            read_call(v, &call);
            if (v->has_result) read_result(v, &res);
        } else if (!strcmp(n, "write")) {
            write_call(v, &call);
            if (v->has_result && v->is_error) {
                Str o = text_output(v->result);
                if (o.len) {
                    buf_addc(&res, '\n');
                    th_fg(&res, T_ERROR, o);
                }
            }
        } else if (!strcmp(n, "grep")) {
            const JValue *pat = json_get(v->args, "pattern"), *path = json_get(v->args, "path");
            title(&call, "grep");
            buf_addc(&call, ' ');
            if (pat && !json_is(pat, J_STRING)) th_fgz(&call, T_ERROR, "[invalid arg]");
            else {
                Buf p = {0};
                buf_printf(&p, "/%.*s/", (int)json_strs(pat).len, json_strs(pat).ptr);
                th_fg(&call, T_ACCENT, buf_str(&p));
                buf_free(&p);
            }
            Buf in = {0};
            buf_adds(&in, " in ");
            shorten_path(&in, json_strs(path).len ? json_strs(path) : S("."));
            th_fg(&call, T_TOOL_OUTPUT, buf_str(&in));
            buf_free(&in);
            if (json_strs(json_get(v->args, "glob")).len) {
                Buf g = {0};
                buf_printf(&g, " (%s)", json_str(json_get(v->args, "glob")));
                th_fg(&call, T_TOOL_OUTPUT, buf_str(&g));
                buf_free(&g);
            }
            if (jhas_num(json_get(v->args, "limit"))) {
                Buf g = {0};
                buf_printf(&g, " limit %g", jnum(json_get(v->args, "limit")));
                th_fg(&call, T_TOOL_OUTPUT, buf_str(&g));
                buf_free(&g);
            }
            if (v->has_result) list_result(v, &res, 15, "matchLimitReached", "matches limit", true);
        } else if (!strcmp(n, "find")) {
            const JValue *pat = json_get(v->args, "pattern"), *path = json_get(v->args, "path");
            title(&call, "find");
            buf_addc(&call, ' ');
            th_fg(&call, T_ACCENT, json_strs(pat));
            Buf in = {0};
            buf_adds(&in, " in ");
            shorten_path(&in, json_strs(path).len ? json_strs(path) : S("."));
            th_fg(&call, T_TOOL_OUTPUT, buf_str(&in));
            buf_free(&in);
            if (jhas_num(json_get(v->args, "limit"))) {
                Buf g = {0};
                buf_printf(&g, " (limit %g)", jnum(json_get(v->args, "limit")));
                th_fg(&call, T_TOOL_OUTPUT, buf_str(&g));
                buf_free(&g);
            }
            if (v->has_result) list_result(v, &res, 20, "resultLimitReached", "results limit", false);
        } else {
            title(&call, "ls");
            buf_addc(&call, ' ');
            tool_path(&call, json_get(v->args, "path"), ".");
            if (jhas_num(json_get(v->args, "limit"))) {
                Buf g = {0};
                buf_printf(&g, " (limit %g)", jnum(json_get(v->args, "limit")));
                th_fg(&call, T_TOOL_OUTPUT, buf_str(&g));
                buf_free(&g);
            }
            if (v->has_result) list_result(v, &res, 20, "entryLimitReached", "entries limit", false);
        }
        text_lines(buf_str(&call), 0, cw, &kids);
        if (res.len) text_lines(buf_str(&res), 0, cw, &kids);
        buf_free(&call);
        buf_free(&res);
    }
    box_lines(&kids, width, 1, 1, out);
    lines_free(&kids);
}

static void tool_destroy(Comp *c) {
    ToolView *v = (ToolView *)c;
    arena_reset(&v->a);
    free(v->name);
    free(v->id);
    free(v);
}

static const CompVt tool_vt = {tool_render, tool_destroy};

static ToolView *tool_new(const char *name, const char *id, const JValue *args) {
    ToolView *v = xcalloc(1, sizeof *v);
    v->base.vt = &tool_vt;
    v->name = dupz(name ? name : "");
    v->id = dupz(id ? id : "");
    v->args = jcopy(&v->a, args);
    v->partial = true;
    v->expanded = U->tools_expanded;
    return v;
}

static void tool_set_args(ToolView *v, const JValue *args) {
    // the arguments and the result share the arena: keep the result's text
    JValue *res = v->result;
    Arena old = v->a;
    v->a = (Arena){0};
    v->args = jcopy(&v->a, args);
    v->result = jcopy(&v->a, res);
    arena_reset(&old);
}

static void tool_set_result(ToolView *v, const JValue *result, bool is_error, bool partial, const JValue *duration) {
    JValue *args = v->args;
    Arena old = v->a;
    v->a = (Arena){0};
    v->args = jcopy(&v->a, args);
    v->result = jcopy(&v->a, result);
    arena_reset(&old);
    v->has_result = true;
    v->is_error = is_error;
    v->partial = partial;
    v->has_duration = duration && jhas_num(duration);
    v->duration_ms = v->has_duration ? jnum(duration) : 0;
}

static ToolView *find_pending(const char *id) {
    for (size_t i = 0; i < U->pending_tools.len; i++)
        if (!strcmp(U->pending_tools.ptr[i]->id, id)) return U->pending_tools.ptr[i];
    return NULL;
}

static void drop_pending(ToolView *v) {
    for (size_t i = 0; i < U->pending_tools.len; i++)
        if (U->pending_tools.ptr[i] == v) {
            memmove(U->pending_tools.ptr + i, U->pending_tools.ptr + i + 1, (U->pending_tools.len - i - 1) * sizeof(ToolView *));
            U->pending_tools.len--;
            return;
        }
}

// ---- footer (footer.ts) ---------------------------------------------------------------------------------

static void fmt_tokens(Buf *b, double n) {
    if (n < 1000) buf_printf(b, "%.0f", n);
    else if (n < 10000) buf_printf(b, "%.1fk", n / 1000);
    else if (n < 1000000) buf_printf(b, "%.0fk", round(n / 1000));
    else if (n < 10000000) buf_printf(b, "%.1fM", n / 1000000);
    else buf_printf(b, "%.0fM", round(n / 1000000));
}

static void footer_render(void *ctx, int width, Lines *out) {
    (void)ctx;
    const CodeUiConfig *cfg = U->cfg;
    double in = 0, outp = 0, cr = 0, cwr = 0, cost = 0;
    double hit = -1;
    long last_valid = -1;
    for (size_t i = 0; i < U->stats.len; i++) {
        MsgStat *s = &U->stats.ptr[i];
        if (s->role != 'a') continue;
        in += s->input;
        outp += s->output;
        cr += s->cache_read;
        cwr += s->cache_write;
        cost += s->cost;
        double prompt = s->input + s->cache_read + s->cache_write;
        hit = prompt > 0 ? s->cache_read / prompt * 100 : -1;
        if (s->valid_usage) last_valid = (long)i;
    }
    double tokens = 0;
    if (last_valid < 0)
        for (size_t i = 0; i < U->stats.len; i++) tokens += (double)U->stats.ptr[i].est;
    else {
        MsgStat *s = &U->stats.ptr[last_valid];
        tokens = s->total ? s->total : s->input + s->output + s->cache_read + s->cache_write;
        for (size_t i = (size_t)last_valid + 1; i < U->stats.len; i++) tokens += (double)U->stats.ptr[i].est;
    }
    double window = (double)cfg->context_window;
    double pct = window > 0 ? tokens / window * 100 : 0;
    // the working directory, ~ for the home folder
    Buf pwd = {0};
    size_t hl = cfg->home ? strlen(cfg->home) : 0;
    if (hl && !strncmp(cfg->cwd, cfg->home, hl) && (cfg->cwd[hl] == 0 || cfg->cwd[hl] == '/') && strcmp(cfg->home, "/")) {
        buf_addc(&pwd, '~');
        buf_adds(&pwd, cfg->cwd + hl);
    } else buf_adds(&pwd, cfg->cwd);
    // git branch
    {
        char head[4096];
        snprintf(head, sizeof head, "%s/.git/HEAD", cfg->cwd);
        FILE *f = fopen(head, "r");
        if (f) {
            char line[512];
            if (fgets(line, sizeof line, f)) {
                Str l = str_trim(str(line));
                if (str_starts(l, S("ref: refs/heads/"))) buf_printf(&pwd, " (%.*s)", (int)(l.len - 16), l.ptr + 16);
                else if (l.len >= 7) buf_adds(&pwd, " (detached)");
            }
            fclose(f);
        }
    }
    Buf stats = {0};
    Buf t = {0};
#define PART(...)                            \
    do {                                     \
        if (stats.len) buf_addc(&stats, ' '); \
        buf_printf(&stats, __VA_ARGS__);     \
    } while (0)
    if (in) {
        t.len = 0;
        fmt_tokens(&t, in);
        PART("\xe2\x86\x91%s", t.ptr);
    }
    if (outp) {
        t.len = 0;
        fmt_tokens(&t, outp);
        PART("\xe2\x86\x93%s", t.ptr);
    }
    if (cr) {
        t.len = 0;
        fmt_tokens(&t, cr);
        PART("R%s", t.ptr);
    }
    if (cwr) {
        t.len = 0;
        fmt_tokens(&t, cwr);
        PART("W%s", t.ptr);
    }
    if ((cr > 0 || cwr > 0) && hit >= 0) PART("CH%.1f%%", hit);
    if (cost) PART("$%.3f", cost);
    t.len = 0;
    fmt_tokens(&t, window);
    Buf ctxs = {0};
    buf_printf(&ctxs, "%.1f%%/%s%s", pct, t.ptr, cfg->auto_compact ? " (auto)" : "");
    if (stats.len) buf_addc(&stats, ' ');
    if (pct > 90) th_fg(&stats, T_ERROR, buf_str(&ctxs));
    else if (pct > 70) th_fg(&stats, T_WARNING, buf_str(&ctxs));
    else buf_addstr(&stats, buf_str(&ctxs));
#undef PART
    int sw = tui_width(buf_str(&stats));
    if (sw > width) {
        Buf cut = {0};
        tui_truncate(&cut, buf_str(&stats), width, "...", false);
        buf_free(&stats);
        stats = cut;
        sw = tui_width(buf_str(&stats));
    }
    Buf right = {0};
    const char *model = cfg->model_id ? cfg->model_id : "no-model";
    if (cfg->reasoning) {
        const char *lv = U->thinking ? U->thinking : "off";
        if (!strcmp(lv, "off")) buf_printf(&right, "%s \xe2\x80\xa2 thinking off", model);
        else buf_printf(&right, "%s \xe2\x80\xa2 %s", model, lv);
    } else buf_adds(&right, model);
    int rw = tui_width(buf_str(&right));
    Buf line = {0};
    Buf rem = {0};
    if (sw + 2 + rw <= width) {
        tui_spaces(&rem, width - sw - rw);
        buf_addstr(&rem, buf_str(&right));
    } else if (width - sw - 2 > 0) {
        Buf tr = {0};
        tui_truncate(&tr, buf_str(&right), width - sw - 2, "", false);
        tui_spaces(&rem, MAX(0, width - sw - tui_width(buf_str(&tr))));
        buf_addstr(&rem, buf_str(&tr));
        buf_free(&tr);
    }
    th_fg(&line, T_DIM, buf_str(&stats));
    th_fg(&line, T_DIM, buf_str(&rem));
    Buf p = {0}, ell = {0}, pl = {0};
    th_fg(&p, T_DIM, buf_str(&pwd));
    th_fgz(&ell, T_DIM, "...");
    tui_truncate(&pl, buf_str(&p), width, ell.ptr, false);
    lines_addb(out, &pl);
    lines_addb(out, &line);
    buf_free(&pwd);
    buf_free(&stats);
    buf_free(&t);
    buf_free(&ctxs);
    buf_free(&right);
    buf_free(&rem);
    buf_free(&p);
    buf_free(&ell);
}

// ---- message accounting for the footer ---------------------------------------------------------------

static int64_t est_chars(const JValue *content) {
    if (json_is(content, J_STRING)) return (int64_t)tui_js_len(content->s);
    int64_t n = 0;
    for (size_t i = 0; content && json_is(content, J_ARRAY) && i < content->arr.len; i++) {
        const JValue *b = content->arr.items[i];
        Str type = json_strs(json_get(b, "type"));
        if (str_eqz(type, "text")) n += (int64_t)tui_js_len(json_strs(json_get(b, "text")));
        else if (str_eqz(type, "image")) n += 4800;
    }
    return n;
}

static int64_t ceil4(int64_t c) { return (c + 3) / 4; }

static void account(const JValue *m) {
    Str role = json_strs(json_get(m, "role"));
    MsgStat s = {0};
    if (str_eqz(role, "system")) {
        s.role = 's';
        int64_t c = est_chars(json_get(m, "content"));
        const JValue *sec = json_get(m, "sections");
        for (size_t i = 0; sec && json_is(sec, J_OBJECT) && i < sec->obj.len; i++)
            if (json_is(sec->obj.items[i].val, J_STRING)) c += (int64_t)tui_js_len(sec->obj.items[i].val->s);
        const JValue *ta = json_get(m, "toolsAdded");
        if (ta) {
            Buf b = {0};
            json_write(&b, ta);
            c += (int64_t)tui_js_len(buf_str(&b));
            buf_free(&b);
        }
        s.est = ceil4(c);
    } else if (str_eqz(role, "user")) {
        s.role = 'u';
        s.est = ceil4(est_chars(json_get(m, "content")));
    } else if (str_eqz(role, "assistant")) {
        s.role = 'a';
        int64_t c = 0;
        const JValue *content = json_get(m, "content");
        for (size_t i = 0; content && json_is(content, J_ARRAY) && i < content->arr.len; i++) {
            const JValue *b = content->arr.items[i];
            Str type = json_strs(json_get(b, "type"));
            if (str_eqz(type, "text")) c += (int64_t)tui_js_len(json_strs(json_get(b, "text")));
            else if (str_eqz(type, "thinking")) c += (int64_t)tui_js_len(json_strs(json_get(b, "thinking")));
            else if (str_eqz(type, "toolCall")) {
                Buf j = {0};
                json_write(&j, json_get(b, "arguments"));
                c += (int64_t)(tui_js_len(json_strs(json_get(b, "name"))) + tui_js_len(buf_str(&j)));
                buf_free(&j);
            }
        }
        s.est = ceil4(c);
        const JValue *u = json_get(m, "usage");
        s.input = jnum(json_get(u, "input"));
        s.output = jnum(json_get(u, "output"));
        s.cache_read = jnum(json_get(u, "cacheRead"));
        s.cache_write = jnum(json_get(u, "cacheWrite"));
        s.total = jnum(json_get(u, "totalTokens"));
        s.cost = jnum(json_get(json_get(u, "cost"), "total"));
        Str stop = json_strs(json_get(m, "stopReason"));
        double ctx = s.total ? s.total : s.input + s.output + s.cache_read + s.cache_write;
        s.valid_usage = !str_eqz(stop, "aborted") && !str_eqz(stop, "error") && ctx > 0;
    } else if (str_eqz(role, "toolResult")) {
        s.role = 't';
        s.est = ceil4(est_chars(json_get(m, "content")));
    } else return;
    vec_push(&U->stats, s);
}

// ---- the selectors ----------------------------------------------------------------------------------------

static void border_line(Lines *out, int width, ThemeColor c) {
    Buf b = {0}, r = {0};
    tui_repeat(&r, "\xe2\x94\x80", MAX(1, width));
    th_fg(&b, c, buf_str(&r));
    lines_addb(out, &b);
    buf_free(&r);
}

static void input_render(Input *in, int width, Lines *out) {
    Buf b = {0};
    buf_adds(&b, "> ");
    Str v = buf_str(&in->value);
    int avail = width - 2;
    buf_add(&b, v.ptr, MIN(in->cursor, v.len));
    if (in->cursor < v.len) {
        size_t g = tui_grapheme(v, in->cursor);
        buf_adds(&b, "\x1b[7m");
        buf_add(&b, v.ptr + in->cursor, g);
        buf_adds(&b, "\x1b[27m");
        buf_add(&b, v.ptr + in->cursor + g, v.len - in->cursor - g);
    } else buf_adds(&b, "\x1b[7m \x1b[27m");
    int w = tui_width(buf_str(&b));
    tui_spaces(&b, avail + 2 - w);
    lines_addb(out, &b);
}

// Edits the search input; true when the value changed.
static bool input_key(Input *in, const Key *k) {
    Buf *v = &in->value;
    if (k->kind == K_CHAR) {
        char enc[5];
        uint32_t cp = k->ch;
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
        buf_reserve(v, (size_t)n);
        memmove(v->ptr + in->cursor + n, v->ptr + in->cursor, v->len - in->cursor + 1);
        memcpy(v->ptr + in->cursor, enc, (size_t)n);
        v->len += (size_t)n;
        in->cursor += (size_t)n;
        return true;
    }
    if (k->kind == K_BACKSPACE && in->cursor > 0) {
        size_t p = in->cursor - 1;
        while (p > 0 && ((unsigned char)v->ptr[p] & 0xc0) == 0x80) p--;
        memmove(v->ptr + p, v->ptr + in->cursor, v->len - in->cursor + 1);
        v->len -= in->cursor - p;
        in->cursor = p;
        return true;
    }
    if (k->kind == K_LEFT && in->cursor > 0) {
        in->cursor--;
        while (in->cursor > 0 && ((unsigned char)v->ptr[in->cursor] & 0xc0) == 0x80) in->cursor--;
    } else if (k->kind == K_RIGHT && in->cursor < v->len) in->cursor += tui_grapheme(buf_str(v), in->cursor);
    else if (k->kind == K_HOME || (k->kind == K_CTRL && k->ch == 'a')) in->cursor = 0;
    else if (k->kind == K_END || (k->kind == K_CTRL && k->ch == 'e')) in->cursor = v->len;
    else if (k->kind == K_CTRL && k->ch == 'u') {
        v->len = 0;
        in->cursor = 0;
        if (v->ptr) v->ptr[0] = 0;
        return true;
    }
    return false;
}

static const char *const level_names[] = {"off", "minimal", "low", "medium", "high", "xhigh", "max"};
static const char *const level_desc[] = {"No reasoning", "Very brief reasoning (~1k tokens)", "Light reasoning (~2k tokens)",
                                         "Moderate reasoning (~8k tokens)", "Deep reasoning (~16k tokens)",
                                         "Extra-high reasoning (~32k tokens)", "Maximum reasoning"};

static size_t available_levels(void) { return U->cfg->reasoning ? 5 : 1; }

static void thinking_build(const char *filter, const char *keep) {
    sel_free(&U->list);
    sel_init(&U->list, (int)MAX(1, available_levels()), 12, 32);
    const char *cur = U->thinking ? U->thinking : "off";
    const char *dflt = U->default_thinking ? U->default_thinking : "medium";
    size_t n = available_levels();
    // fuzzy filter over "value description"
    for (size_t i = 0; i < n; i++) {
        Buf label = {0}, desc = {0}, hay = {0};
        buf_printf(&label, "%s%s", !strcmp(level_names[i], cur) ? "\xe2\x9c\x93 " : "  ", level_names[i]);
        buf_adds(&desc, level_desc[i]);
        if (!strcmp(level_names[i], dflt)) buf_adds(&desc, " \xc2\xb7 default");
        buf_printf(&hay, "%s %s", level_names[i], desc.ptr);
        bool ok = true;
        if (filter && *filter) {
            // every query token must appear in order (fuzzyMatch)
            const char *h = hay.ptr;
            for (const char *q = filter; *q && ok; q++) {
                if (*q == ' ') continue;
                char lc = (char)tolower((unsigned char)*q);
                while (*h && tolower((unsigned char)*h) != lc) h++;
                if (!*h) ok = false;
                else h++;
            }
        }
        if (ok) sel_add(&U->list, level_names[i], label.ptr, desc.ptr);
        buf_free(&label);
        buf_free(&desc);
        buf_free(&hay);
    }
    U->list.max_visible = (int)MAX(1, U->list.n);
    const char *pre = keep ? keep : cur;
    for (size_t i = 0; i < U->list.n; i++)
        if (!strcmp(U->list.items[i].value, pre)) U->list.selected = i;
}

static void thinking_render(int width, Lines *out) {
    border_line(out, width, T_BORDER);
    lines_addz(out, "");
    text_lines(S("Thinking Level"), 0, width, out);
    lines_addz(out, "");
    text_lines(S("Shift+Tab cycles thinking levels in-session"), 0, width, out);
    lines_addz(out, "");
    input_render(&U->search, width, out);
    lines_addz(out, "");
    sel_render(&U->list, width, out);
    lines_addz(out, "");
    Buf h = {0};
    th_fgz(&h, T_DIM, "  Enter to select \xc2\xb7 Ctrl+S to set as default \xc2\xb7 Escape/Ctrl+C to cancel");
    text_lines(buf_str(&h), 0, width, out);
    buf_free(&h);
    border_line(out, width, T_BORDER);
}

static void close_selector(void) {
    U->sel = SEL_NONE;
    sel_free(&U->list);
    buf_free(&U->search.value);
    U->search.cursor = 0;
}

static void set_thinking(const char *level) {
    free(U->thinking);
    U->thinking = dupz(level);
    editor_set_border(U->ed, th_thinking_color(level));
    if (U->ops->set_thinking) U->ops->set_thinking(U->host, level);
}

static void open_thinking(void) {
    close_selector();
    U->sel = SEL_THINKING;
    thinking_build(NULL, NULL);
}

// ---- layout and drawing --------------------------------------------------------------------------------

static void render_dock(int width, Lines *out) {
    comp_render(U->pending, width, out);
    comp_render(U->status, width, out);
    lines_addz(out, ""); // the widgets above the editor: a spacer when there are none
    if (U->sel == SEL_THINKING) thinking_render(width, out);
    else editor_render(U->ed, width, out);
    footer_render(NULL, width, out);
}

static void draw(bool force) {
    Screen *s = &U->scr;
    int w = s->cols, h = s->rows;
    editor_set_rows(U->ed, h);
    Lines dock = {0}, doc = {0}, frame = {0};
    render_dock(w, &dock);
    int th = MAX(1, h - (int)dock.n);
    comp_render(U->header, w, &doc);
    comp_render(U->chat, w, &doc);
    size_t start = doc.n > (size_t)th ? doc.n - (size_t)th : 0;
    for (size_t i = start; i < doc.n && frame.n < (size_t)th; i++) lines_addz(&frame, doc.v[i]);
    while (frame.n < (size_t)th) lines_addz(&frame, "");
    for (size_t i = 0; i < dock.n && frame.n < (size_t)h; i++) lines_addz(&frame, dock.v[i]);
    screen_draw(s, &frame, force);
    lines_free(&dock);
    lines_free(&doc);
    lines_free(&frame);
}

static bool g_dirty;
static void request_render(void) { g_dirty = true; }

// ---- the agent thread -------------------------------------------------------------------------------------

static void *agent_main(void *arg) {
    (void)arg;
    pthread_mutex_lock(&U->lock);
    for (;;) {
        while (!U->job && !U->thread_quit) pthread_cond_wait(&U->cond, &U->lock);
        if (U->thread_quit) break;
        char *job = U->job;
        U->job = NULL;
        pthread_mutex_unlock(&U->lock);
        U->ops->prompt(U->host, job);
        free(job);
        q_post("{\"type\":\"ui_idle\"}");
        pthread_mutex_lock(&U->lock);
    }
    pthread_mutex_unlock(&U->lock);
    return NULL;
}

static void start_prompt(const char *text) {
    U->busy = true;
    pthread_mutex_lock(&U->lock);
    U->job = strdup(text);
    pthread_cond_signal(&U->cond);
    pthread_mutex_unlock(&U->lock);
}

// ---- events (handleEvent) -------------------------------------------------------------------------------------

static void set_working(bool on) {
    U->working = on;
    editor_set_working(U->ed, on ? "Working" : NULL);
}

static void add_user(const JValue *m) {
    const JValue *c = json_get(m, "content");
    Buf t = {0};
    if (json_is(c, J_STRING)) buf_addstr(&t, c->s);
    else
        for (size_t i = 0; c && json_is(c, J_ARRAY) && i < c->arr.len; i++)
            if (str_eqz(json_strs(json_get(c->arr.items[i], "type")), "text")) buf_addstr(&t, json_strs(json_get(c->arr.items[i], "text")));
    if (t.len) {
        if (comp_children(U->chat)->len) chat_add(comp_spacer(1));
        chat_add(comp_markdown(buf_str(&t), 1, 1, &md_user));
    }
    buf_free(&t);
}

static void sync_tools(const JValue *msg) {
    const JValue *content = json_get(msg, "content");
    for (size_t i = 0; content && json_is(content, J_ARRAY) && i < content->arr.len; i++) {
        const JValue *b = content->arr.items[i];
        if (!str_eqz(json_strs(json_get(b, "type")), "toolCall")) continue;
        const char *id = json_str(json_get(b, "id"));
        if (!id) continue;
        ToolView *v = find_pending(id);
        if (!v) {
            v = tool_new(json_str(json_get(b, "name")), id, json_get(b, "arguments"));
            chat_add(&v->base);
            vec_push(&U->pending_tools, v);
        } else tool_set_args(v, json_get(b, "arguments"));
    }
}

static bool retryable_error(const char *m) {
    // the transient errors pi retries (a short list: the full one is in code_main.c)
    if (!m) return false;
    static const char *const pats[] = {"429", "500", "502", "503", "504", "overloaded", "rate limit", "timeout", "timed out",
                                       "connection", "network", "fetch failed", "terminated", "socket hang up", "ECONNRESET"};
    for (size_t i = 0; i < COUNT(pats); i++)
        if (strcasestr(m, pats[i])) return true;
    return false;
}

static void handle_event(const JValue *ev) {
    Str type = json_strs(json_get(ev, "type"));
    if (str_eqz(type, "ui_idle")) {
        U->busy = false;
        set_working(false);
        if (U->queued.len) {
            char *next = U->queued.ptr[0];
            memmove(U->queued.ptr, U->queued.ptr + 1, (U->queued.len - 1) * sizeof(char *));
            U->queued.len--;
            comp_clear(U->pending);
            start_prompt(next);
            free(next);
        }
        return;
    }
    if (str_eqz(type, "ui_error")) {
        show_error(json_str(json_get(ev, "message")) ? json_str(json_get(ev, "message")) : "error");
        return;
    }
    if (str_eqz(type, "agent_start")) {
        U->pending_tools.len = 0;
    } else if (str_eqz(type, "turn_start")) {
        if (!U->working) set_working(true);
    } else if (str_eqz(type, "message_start")) {
        const JValue *m = json_get(ev, "message");
        Str role = json_strs(json_get(m, "role"));
        if (str_eqz(role, "user")) add_user(m);
        else if (str_eqz(role, "assistant")) {
            U->streaming = asst_new(m);
            chat_add(&U->streaming->base);
        }
    } else if (str_eqz(type, "message_update")) {
        const JValue *m = json_get(ev, "message");
        if (U->streaming && m) {
            asst_set(U->streaming, m);
            sync_tools(m);
        }
    } else if (str_eqz(type, "message_end")) {
        const JValue *m = json_get(ev, "message");
        Str role = json_strs(json_get(m, "role"));
        account(m);
        if (str_eqz(role, "assistant") && U->streaming) {
            Str stop = json_strs(json_get(m, "stopReason"));
            JValue *copy = NULL;
            Arena tmp = {0};
            copy = jcopy(&tmp, m);
            if (str_eqz(stop, "aborted")) json_put(&tmp, copy, S("errorMessage"), json_string(&tmp, S("Operation aborted")));
            asst_set(U->streaming, copy);
            sync_tools(copy);
            if (str_eqz(stop, "aborted") || str_eqz(stop, "error")) {
                const char *em = json_str(json_get(copy, "errorMessage"));
                Arena ra = {0};
                JValue *res = json_new(&ra, J_OBJECT);
                JValue *content = json_new(&ra, J_ARRAY);
                JValue *blk = json_new(&ra, J_OBJECT);
                json_put(&ra, blk, S("type"), json_string(&ra, S("text")));
                json_put(&ra, blk, S("text"), json_string(&ra, str(em ? em : "Error")));
                json_push(&ra, content, blk);
                json_put(&ra, res, S("content"), content);
                for (size_t i = 0; i < U->pending_tools.len; i++) tool_set_result(U->pending_tools.ptr[i], res, true, false, NULL);
                U->pending_tools.len = 0;
                arena_reset(&ra);
                // maybeSuggestBugReport
                if (str_eqz(stop, "error") && !retryable_error(em) && !(em && (strcasestr(em, "abort") || strcasestr(em, "cancel"))) &&
                    !U->bug_hint_shown) {
                    U->bug_hint_shown = true;
                    chat_add(themed(T_MUTED, "If this looks like a pi bug, /bug sends a report to the developers.", 1, 0));
                }
            } else
                for (size_t i = 0; i < U->pending_tools.len; i++) U->pending_tools.ptr[i]->args_complete = true;
            arena_reset(&tmp);
            U->streaming = NULL;
        }
    } else if (str_eqz(type, "tool_execution_start")) {
        const char *id = json_str(json_get(ev, "toolCallId"));
        ToolView *v = id ? find_pending(id) : NULL;
        if (!v) {
            v = tool_new(json_str(json_get(ev, "toolName")), id, json_get(ev, "args"));
            chat_add(&v->base);
            vec_push(&U->pending_tools, v);
        }
        v->started = true;
    } else if (str_eqz(type, "tool_execution_update")) {
        const char *id = json_str(json_get(ev, "toolCallId"));
        ToolView *v = id ? find_pending(id) : NULL;
        if (v) tool_set_result(v, json_get(ev, "partialResult"), false, true, NULL);
    } else if (str_eqz(type, "tool_execution_end")) {
        const char *id = json_str(json_get(ev, "toolCallId"));
        ToolView *v = id ? find_pending(id) : NULL;
        if (v) {
            tool_set_result(v, json_get(ev, "result"), jtrue(json_get(ev, "isError")), false, json_get(ev, "durationMs"));
            drop_pending(v);
        }
    } else if (str_eqz(type, "agent_end")) {
        set_working(false);
        if (U->streaming) {
            // a message that never ended is dropped
            CompVec *kids = comp_children(U->chat);
            for (size_t i = 0; i < kids->len; i++)
                if (kids->ptr[i] == &U->streaming->base) {
                    comp_free(kids->ptr[i]);
                    memmove(kids->ptr + i, kids->ptr + i + 1, (kids->len - i - 1) * sizeof(Comp *));
                    kids->len--;
                    break;
                }
            U->streaming = NULL;
        }
        U->pending_tools.len = 0;
    } else if (str_eqz(type, "auto_retry_start")) {
        // RetryStatusIndicator: the retry countdown in the editor's border
        Buf m = {0};
        buf_printf(&m, "Retrying (%g/%g) in %.0fs... (escape to cancel)", jnum(json_get(ev, "attempt")), jnum(json_get(ev, "maxAttempts")),
                   ceil(jnum(json_get(ev, "delayMs")) / 1000));
        editor_set_working(U->ed, m.ptr);
        U->working = true;
        buf_free(&m);
    } else if (str_eqz(type, "auto_retry_end")) {
        if (!jtrue(json_get(ev, "success"))) {
            const char *fe = json_str(json_get(ev, "finalError"));
            Buf m = {0};
            buf_printf(&m, "Retry failed after %g attempts: %s", jnum(json_get(ev, "attempt")), fe ? fe : "Unknown error");
            show_error(m.ptr);
            buf_free(&m);
        }
    }
    request_render();
}

// ---- commands and keys --------------------------------------------------------------------------------------

static const SlashCmd builtin_cmds[] = {
    {"settings", "Open settings menu", NULL},
    {"model", "Select model (opens selector UI)", "<provider/model>"},
    {"tree", "Navigate session tree (switch branches)", NULL},
    {"thinking", "Set thinking level", "<level>"},
    {"scoped-models", "Enable/disable models for Ctrl+P cycling", NULL},
    {"export", "Export session (HTML default, or specify path: .html/.jsonl)", NULL},
    {"import", "Import a JSONL session, preserving its ID and working directory", NULL},
    {"share", "Share session via a Radius artifact or a GitHub gist", NULL},
    {"bug", "Report a bug to the Pi developers", "<description>"},
    {"copy", "Copy last agent message to clipboard", NULL},
    {"name", "Set session display name", NULL},
    {"session", "Show session info and stats", NULL},
    {"changelog", "Show changelog entries", NULL},
    {"hotkeys", "Show all keyboard shortcuts", NULL},
    {"fork", "Create a new fork from a previous user message", NULL},
    {"clone", "Duplicate the current session at the current position", NULL},
    {"trust", "Save project trust decision for future sessions", NULL},
    {"login", "Configure provider authentication", "<provider>"},
    {"logout", "Remove provider authentication", NULL},
    {"new", "Start a new session", NULL},
    {"compact", "Manually compact the session context", NULL},
    {"resume", "Resume another session", NULL},
    {"reload", "Reload keybindings, extensions, skills, prompts, themes, and context files", NULL},
    {"quit", "Quit pi", NULL},
    {"llama", "Manage llama.cpp router models", NULL},
    {"mcp", "Manage MCP servers: sign in, reconnect, enable or disable, and change exposure", NULL},
};

static void new_session(void) {
    if (U->busy) return;
    if (U->ops->new_session) U->ops->new_session(U->host);
    comp_clear(U->chat);
    U->stats.len = 0;
    U->bug_hint_shown = false;
    U->last_status_text = U->last_status_spacer = NULL;
    chat_add(comp_spacer(1));
    chat_add(themed(T_ACCENT, "\xe2\x9c\x93 New session started", 1, 1));
}

static void unsupported(const char *cmd) {
    Buf b = {0};
    buf_printf(&b, "/%s is not available in Saetta Code yet.", cmd);
    show_warning(b.ptr);
    buf_free(&b);
}

static void submit(char *text) {
    Str t = str_trim(str(text));
    if (!t.len) return;
    char *s = strndup(t.ptr, t.len);
    if (s[0] == '/') {
        const char *c = s + 1;
        if (!strcmp(c, "thinking") || !strncmp(c, "thinking ", 9)) {
            const char *arg = !strncmp(c, "thinking ", 9) ? c + 9 : NULL;
            if (arg && *arg) {
                bool ok = false;
                for (size_t i = 0; i < available_levels(); i++)
                    if (!strcmp(arg, level_names[i])) ok = true;
                if (ok) {
                    set_thinking(arg);
                    Buf m = {0};
                    buf_printf(&m, "Thinking level: %s", arg);
                    show_status(m.ptr);
                    buf_free(&m);
                } else {
                    Buf m = {0};
                    buf_printf(&m, "Invalid thinking level \"%s\".", arg);
                    show_error(m.ptr);
                    buf_free(&m);
                }
            } else open_thinking();
            free(s);
            return;
        }
        if (!strcmp(c, "new")) {
            new_session();
            free(s);
            return;
        }
        if (!strcmp(c, "quit")) {
            U->quit = true;
            free(s);
            return;
        }
        static const char *const known[] = {"settings", "model", "tree", "scoped-models", "export", "import", "share", "bug", "copy",
                                            "name", "session", "changelog", "hotkeys", "fork", "clone", "trust", "login", "logout",
                                            "compact", "resume", "reload", "llama", "mcp"};
        for (size_t i = 0; i < COUNT(known); i++) {
            size_t kl = strlen(known[i]);
            if (!strncmp(c, known[i], kl) && (c[kl] == 0 || c[kl] == ' ')) {
                unsupported(known[i]);
                free(s);
                return;
            }
        }
    }
    editor_add_history(U->ed, str(s));
    if (U->busy) {
        vec_push(&U->queued, strdup(s));
        comp_clear(U->pending);
        comp_add(U->pending, comp_spacer(1));
        for (size_t i = 0; i < U->queued.len; i++) {
            Buf m = {0}, l = {0};
            buf_printf(&m, "Follow-up: %s", U->queued.ptr[i]);
            th_fg(&l, T_DIM, buf_str(&m));
            char *line = tui_truncatez(buf_str(&l), U->scr.cols - 2, "...", false);
            comp_add(U->pending, comp_text(str(line), 1, 0));
            free(line);
            buf_free(&m);
            buf_free(&l);
        }
    } else start_prompt(s);
    free(s);
}

static void toggle_tools(void) {
    U->tools_expanded = !U->tools_expanded;
    U->header_expanded = U->tools_expanded;
    CompVec *kids = comp_children(U->chat);
    for (size_t i = 0; i < kids->len; i++)
        if (kids->ptr[i]->vt == &tool_vt) ((ToolView *)kids->ptr[i])->expanded = U->tools_expanded;
    show_status(U->tools_expanded ? "Tool output: expanded" : "Tool output: collapsed");
}

static void cycle_thinking(void) {
    if (!U->cfg->reasoning) {
        show_status("Current model does not support thinking");
        return;
    }
    size_t n = available_levels(), cur = 0;
    for (size_t i = 0; i < n; i++)
        if (U->thinking && !strcmp(U->thinking, level_names[i])) cur = i;
    const char *next = level_names[(cur + 1) % n];
    set_thinking(next);
    Buf m = {0};
    buf_printf(&m, "Thinking level: %s", next);
    show_status(m.ptr);
    buf_free(&m);
}

static void selector_key(const Key *k) {
    if (U->sel == SEL_THINKING) {
        if (k->kind == K_UP) sel_up(&U->list);
        else if (k->kind == K_DOWN) sel_down(&U->list);
        else if (k->kind == K_ENTER) {
            const SelItem *it = sel_current(&U->list);
            char *lv = it ? strdup(it->value) : NULL;
            close_selector();
            if (lv) set_thinking(lv);
            free(lv);
        } else if (k->kind == K_ESC || (k->kind == K_CTRL && k->ch == 'c')) close_selector();
        else if (k->kind == K_CTRL && k->ch == 's') {
            const SelItem *it = sel_current(&U->list);
            char *lv = it ? strdup(it->value) : NULL;
            close_selector();
            if (lv) {
                set_thinking(lv);
                free(U->default_thinking);
                U->default_thinking = lv;
            }
        } else if (input_key(&U->search, k)) {
            const SelItem *it = sel_current(&U->list);
            char *keep = it ? strdup(it->value) : NULL;
            thinking_build(U->search.value.ptr ? U->search.value.ptr : "", keep);
            free(keep);
        }
    }
}

static void handle_key(const Key *k) {
    if (U->sel != SEL_NONE) {
        selector_key(k);
        return;
    }
    // app actions (custom-editor.ts)
    if (k->kind == K_ESC && !editor_autocomplete_open(U->ed)) {
        if (U->busy) {
            // restoreQueuedMessagesToEditor({abort: true})
            if (U->queued.len) {
                Buf all = {0};
                for (size_t i = 0; i < U->queued.len; i++) {
                    if (all.len) buf_adds(&all, "\n\n");
                    buf_adds(&all, U->queued.ptr[i]);
                    free(U->queued.ptr[i]);
                }
                U->queued.len = 0;
                Str cur = editor_text(U->ed);
                if (str_trim(cur).len) {
                    buf_adds(&all, "\n\n");
                    buf_addstr(&all, cur);
                }
                editor_set_text(U->ed, buf_str(&all));
                buf_free(&all);
                comp_clear(U->pending);
            }
            U->ops->abort(U->host);
        }
        return;
    }
    if (k->kind == K_CTRL && k->ch == 'c') {
        uint64_t now = now_ns();
        if (U->last_ctrl_c_ns && now - U->last_ctrl_c_ns < 500000000ull) U->quit = true;
        else {
            editor_set_text(U->ed, S(""));
            U->last_ctrl_c_ns = now;
        }
        return;
    }
    if (k->kind == K_CTRL && k->ch == 'd' && editor_empty(U->ed)) {
        U->quit = true;
        return;
    }
    if (k->kind == K_CTRL && k->ch == 'o') {
        toggle_tools();
        return;
    }
    if (k->kind == K_CTRL && k->ch == 't') {
        U->hide_thinking = !U->hide_thinking;
        return;
    }
    if (k->kind == K_SHIFT_TAB) {
        cycle_thinking();
        return;
    }
    if (k->kind == K_ALT_ENTER) {
        // app.message.followUp: queue the text after the running turn
        Str t = str_trim(editor_text(U->ed));
        if (t.len) {
            char *s = strndup(t.ptr, t.len);
            editor_set_text(U->ed, S(""));
            submit(s);
            free(s);
        }
        return;
    }
    char *text = NULL;
    if (editor_key(U->ed, k, &text) && text) {
        submit(text);
        free(text);
    }
}

// ---- the loop -----------------------------------------------------------------------------------------------

static volatile sig_atomic_t g_winch;
static void on_winch(int sig) {
    (void)sig;
    g_winch = 1;
}

Err code_ui_run(const CodeUiConfig *cfg, const CodeUiOps *ops, CodeUiHost *host, const char *const *initial, size_t ninitial) {
    Ui ui = {0};
    U = &ui;
    ui.cfg = cfg;
    ui.ops = ops;
    ui.host = host;
    ui.thinking = dupz(cfg->thinking ? cfg->thinking : "off");
    ui.default_thinking = dupz("medium");
    ui.header = comp_fn(header_render, NULL);
    ui.chat = comp_container();
    ui.pending = comp_container();
    ui.status = comp_container();
    ui.ed = editor_new(cfg->cwd, cfg->fd_path);
    editor_set_commands(ui.ed, builtin_cmds, COUNT(builtin_cmds));
    editor_set_border(ui.ed, th_thinking_color(ui.thinking));
    sel_init(&ui.list, 5, 32, 32);
    pthread_mutex_init(&ui.lock, NULL);
    pthread_cond_init(&ui.cond, NULL);
    if (pipe(g_wake) == 0) {
        fcntl(g_wake[0], F_SETFL, O_NONBLOCK);
        fcntl(g_wake[1], F_SETFL, O_NONBLOCK);
    }
    pthread_create(&ui.thread, NULL, agent_main, NULL);
    struct sigaction sa = {0};
    sa.sa_handler = on_winch;
    sigaction(SIGWINCH, &sa, NULL);
    screen_start(&ui.scr);
    draw(true);
    for (size_t i = 0; i < ninitial; i++) submit((char *)initial[i]);
    char in[65536];
    size_t inlen = 0;
    uint64_t last_draw = 0, last_spin = now_ns();
    g_dirty = true;
    while (!ui.quit) {
        struct pollfd fds[2] = {{0, POLLIN, 0}, {g_wake[0], POLLIN, 0}};
        int timeout = inlen ? 25 : ui.working ? 80 : 1000;
        if (g_dirty) timeout = 8;
        int r = poll(fds, 2, timeout);
        if (r < 0 && errno != EINTR) break;
        if (g_winch) {
            g_winch = 0;
            screen_size(&ui.scr);
            draw(true);
        }
        if (r > 0 && (fds[1].revents & POLLIN)) {
            char junk[256];
            while (read(g_wake[0], junk, sizeof junk) > 0) {}
        }
        // events from the agent
        for (QItem *it = q_take_all(); it;) {
            Arena a = {0};
            JValue *ev = NULL;
            if (!json_parse(&a, str(it->json), true, &ev) && ev) handle_event(ev);
            arena_reset(&a);
            QItem *next = it->next;
            free(it->json);
            free(it);
            it = next;
        }
        // keys
        bool got = false;
        if (r > 0 && (fds[0].revents & (POLLIN | POLLHUP))) {
            ssize_t n = read(0, in + inlen, sizeof in - inlen);
            if (n <= 0) {
                if (n == 0) break;
            } else {
                inlen += (size_t)n;
                got = true;
            }
        }
        size_t off = 0;
        for (;;) {
            Key k;
            size_t used = key_parse(in + off, inlen - off, !got, &k);
            if (!used) break;
            off += used;
            if (k.kind != K_NONE) handle_key(&k);
            free(k.paste);
            request_render();
            if (ui.quit) break;
        }
        memmove(in, in + off, inlen - off);
        inlen -= off;
        if (inlen == sizeof in) inlen = 0; // a runaway sequence
        uint64_t now = now_ns();
        if (ui.working && now - last_spin >= 80000000ull) {
            editor_spin(ui.ed);
            last_spin = now;
            g_dirty = true;
        }
        if (g_dirty && now - last_draw >= 8000000ull) {
            draw(false);
            last_draw = now;
            g_dirty = false;
        }
    }
    if (ui.busy) {
        // the run stops at its next read; wait for it (up to 5 s)
        ops->abort(host);
        uint64_t until = now_ns() + 5000000000ull;
        while (ui.busy && now_ns() < until) {
            struct pollfd pf = {g_wake[0], POLLIN, 0};
            poll(&pf, 1, 50);
            char junk[256];
            while (read(g_wake[0], junk, sizeof junk) > 0) {}
            for (QItem *it = q_take_all(); it;) {
                if (strstr(it->json, "\"ui_idle\"")) ui.busy = false;
                QItem *next = it->next;
                free(it->json);
                free(it);
                it = next;
            }
        }
    }
    pthread_mutex_lock(&ui.lock);
    ui.thread_quit = true;
    pthread_cond_signal(&ui.cond);
    pthread_mutex_unlock(&ui.lock);
    screen_stop(&ui.scr);
    if (!ui.busy) pthread_join(ui.thread, NULL);
    // what is still queued is dropped
    for (QItem *it = q_take_all(); it;) {
        QItem *next = it->next;
        free(it->json);
        free(it);
        it = next;
    }
    editor_free(ui.ed);
    comp_free(ui.header);
    comp_free(ui.chat);
    comp_free(ui.pending);
    comp_free(ui.status);
    sel_free(&ui.list);
    buf_free(&ui.search.value);
    vec_free(&ui.stats);
    vec_free(&ui.pending_tools);
    for (size_t i = 0; i < ui.queued.len; i++) free(ui.queued.ptr[i]);
    vec_free(&ui.queued);
    free(ui.thinking);
    free(ui.default_thinking);
    free(ui.job);
    U = NULL;
    return OK;
}
