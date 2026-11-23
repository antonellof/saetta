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
#include "code_cfg.h"
#include "code_mcp_ui.h"
#include "code_oai.h"
#include "code_monitor.h"
#include "code_res_ui.h"
#include "code_cmds.h"
#include "code_update.h"
#include "code_ui_ext.h"
#include "code_subagent.h"
#include "code_share.h"
#include "code_autocompact.h"
#include "code_ui_keys.h"
#include "download.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <spawn.h>
#include <sys/wait.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <time.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <unistd.h>

// ---- the event ring (agent thread -> UI thread) ------------------------------------------------
//
// One producer (the agent thread) and one consumer (the UI thread). The per-token path (the
// streamed deltas from the decode thread) copies its text into a preallocated byte ring and writes
// one fixed-size record: no allocation, no lock, no system call, except the single write that
// wakes the UI thread when it sleeps (at most once per UI wake-up, never per token). The event
// JSON is built on the UI thread. The rare events (message start and end, tool events, the end of
// a run) carry their JSON in a heap string.

typedef enum { R_JSON, R_STREAM } RecKind;
typedef struct {
    uint8_t kind;
    uint8_t skind; // CodeUiStreamKind
    int32_t index;
    uint64_t off;  // into the byte ring (monotonic)
    uint32_t len, len2;
    char *json;    // R_JSON
    bool update;
} Rec;

#define RING_RECS 16384u
#define RING_BYTES (8u << 20)
static Rec g_recs[RING_RECS];
static char *g_bytes;
static _Atomic uint64_t g_rhead, g_rtail, g_bhead, g_btail;
static _Atomic int g_ui_sleeping;
static int g_wake[2] = {-1, -1};
static pthread_t g_ui_thread;
static bool g_queue_dirty; // the UI thread took a queued message itself

static void ring_init(void) {
    if (!g_bytes) g_bytes = xmalloc(RING_BYTES);
    atomic_store(&g_rhead, 0);
    atomic_store(&g_rtail, 0);
    atomic_store(&g_bhead, 0);
    atomic_store(&g_btail, 0);
}

static void ring_wake(void) {
    if (atomic_exchange(&g_ui_sleeping, 0) && g_wake[1] >= 0) (void)!write(g_wake[1], "x", 1);
}

static void ring_copy_in(uint64_t at, const char *p, size_t n) {
    size_t o = (size_t)(at % RING_BYTES), first = MIN(n, RING_BYTES - o);
    memcpy(g_bytes + o, p, first);
    if (n > first) memcpy(g_bytes, p + first, n - first);
}

static void ring_copy_out(uint64_t at, size_t n, Buf *out) {
    size_t o = (size_t)(at % RING_BYTES), first = MIN(n, RING_BYTES - o);
    buf_add(out, g_bytes + o, first);
    if (n > first) buf_add(out, g_bytes, n - first);
}

static void ring_push(Rec r, Str a, Str b) {
    if (!g_bytes) return;
    // a single delta never exceeds a quarter of the ring (it is cut, which no real stream reaches)
    a.len = MIN(a.len, RING_BYTES / 4);
    b.len = MIN(b.len, RING_BYTES / 4);
    size_t need = a.len + b.len;
    // backpressure: the UI thread drains on every wake-up; wait for room (never on the normal path)
    for (;;) {
        uint64_t rh = atomic_load_explicit(&g_rhead, memory_order_relaxed), rt = atomic_load_explicit(&g_rtail, memory_order_acquire);
        uint64_t bh = atomic_load_explicit(&g_bhead, memory_order_relaxed), bt = atomic_load_explicit(&g_btail, memory_order_acquire);
        if (rh - rt < RING_RECS && bh + need - bt <= RING_BYTES) break;
        ring_wake();
        struct timespec ts = {0, 1000000};
        nanosleep(&ts, NULL);
    }
    uint64_t bh = atomic_load_explicit(&g_bhead, memory_order_relaxed);
    ring_copy_in(bh, a.ptr, a.len);
    ring_copy_in(bh + a.len, b.ptr, b.len);
    r.off = bh;
    r.len = (uint32_t)a.len;
    r.len2 = (uint32_t)b.len;
    atomic_store_explicit(&g_bhead, bh + need, memory_order_relaxed);
    uint64_t rh = atomic_load_explicit(&g_rhead, memory_order_relaxed);
    g_recs[rh % RING_RECS] = r;
    atomic_store_explicit(&g_rhead, rh + 1, memory_order_release);
    ring_wake();
}

void code_ui_stream(CodeUiStreamKind kind, int index, Str a, Str b) { ring_push((Rec){.kind = R_STREAM, .skind = (uint8_t)kind, .index = index}, a, b); }

static void q_push(char *json, bool update) { ring_push((Rec){.kind = R_JSON, .json = json, .update = update}, S(""), S("")); }

void code_ui_event(const JValue *event) {
    Buf b = {0};
    json_write(&b, event);
    const char *t = json_str(json_get(event, "type"));
    q_push(b.ptr ? b.ptr : strdup("{}"), t && !strcmp(t, "message_update"));
}

static void q_post(const char *json) {
    if (pthread_equal(pthread_self(), g_ui_thread)) g_queue_dirty = true; // the UI thread never produces into the ring
    else q_push(strdup(json), false);
}

// Requests the UI makes while a run goes on, which the agent thread applies at its next turn
// boundary (it alone touches the run's state then): a model or a thinking level.
static pthread_mutex_t g_req_lock = PTHREAD_MUTEX_INITIALIZER;
static char *g_req_provider, *g_req_model, *g_req_thinking;

bool code_ui_take_model(char **provider, char **id) {
    pthread_mutex_lock(&g_req_lock);
    bool have = g_req_model != NULL;
    *provider = g_req_provider;
    *id = g_req_model;
    g_req_provider = g_req_model = NULL;
    pthread_mutex_unlock(&g_req_lock);
    return have;
}

char *code_ui_take_thinking(void) {
    pthread_mutex_lock(&g_req_lock);
    char *t = g_req_thinking;
    g_req_thinking = NULL;
    pthread_mutex_unlock(&g_req_lock);
    return t;
}

static void request_model(const char *provider, const char *id) {
    pthread_mutex_lock(&g_req_lock);
    free(g_req_provider);
    free(g_req_model);
    g_req_provider = strdup(provider);
    g_req_model = strdup(id);
    pthread_mutex_unlock(&g_req_lock);
}

static void request_thinking(const char *level) {
    pthread_mutex_lock(&g_req_lock);
    free(g_req_thinking);
    g_req_thinking = strdup(level);
    pthread_mutex_unlock(&g_req_lock);
}

// The messages typed while a run goes on (pi's steering and follow-up queues): the agent thread
// takes them at its turn boundaries (code_ui_take_queued).
typedef struct {
    char *text;
    bool steer;
} Queued;
static pthread_mutex_t g_mlock = PTHREAD_MUTEX_INITIALIZER;
static VEC(Queued) g_msgq;

char *code_ui_take_queued(bool steering_only) {
    char *t = NULL;
    pthread_mutex_lock(&g_mlock);
    for (size_t i = 0; i < g_msgq.len; i++)
        if (g_msgq.ptr[i].steer || !steering_only) {
            t = g_msgq.ptr[i].text;
            memmove(g_msgq.ptr + i, g_msgq.ptr + i + 1, (g_msgq.len - i - 1) * sizeof(Queued));
            g_msgq.len--;
            break;
        }
    pthread_mutex_unlock(&g_mlock);
    if (t) q_post("{\"type\":\"queue_update\"}");
    return t;
}

// agent.hasQueuedMessages(): a steering or follow-up message waiting (the auto-compaction's continue)
bool code_ui_has_queued(void) {
    pthread_mutex_lock(&g_mlock);
    bool any = g_msgq.len > 0;
    pthread_mutex_unlock(&g_mlock);
    return any;
}

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


// ---- the state -------------------------------------------------------------------------------------

typedef struct {
    char role; // 's' system, 'u' user, 'a' assistant, 't' tool result
    int64_t est;
    bool valid_usage;
    double input, output, cache_read, cache_write, total, cost;
} MsgStat;

typedef struct ToolView ToolView;
typedef struct AsstView AsstView;

typedef enum { SEL_NONE, SEL_THINKING, SEL_MODEL, SEL_SETTINGS, SEL_TREE, SEL_CONFIRM, SEL_THEME, SEL_PANEL, SEL_EXT } SelKind;

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
    void (*sel_free)(void *);
    // the agent thread
    pthread_t thread;
    char *default_thinking;
    // the prompts from the command line, run one after another
    char **initial;
    size_t ninitial, ninitial_next;
    char *re_provider, *re_id; // a model to start again on (/model)
    // a pack download (/model)
    char *dl_name, *dl_title, *dl_models_dir;
    unsigned dl_disk, dl_free_gb;
    pid_t dl_pid;
    int dl_fd;
    Buf dl_line;
    bool dl_cancelled;
    uint64_t dl_cancel_ns;
    // a switch to another pack (or away from the engine) waiting for its confirmation
    bool confirm_switch;
    char *sw_provider, *sw_id, *sw_title;
    size_t sw_others; // sessions sharing this process's model when the switch was asked (code_share.c)
    // the theme: the setting (NULL until chosen), the terminal's colors once reported, macOS's
    // appearance (1 dark, 2 light), and what /theme previews over
    char *theme;
    bool have_colors;
    TermColors colors;
    int mac_appearance;
    char *theme_before;
    bool theme_first;
    // the current model (/model changes it) and the settings the UI uses
    CodeUiModel model;
    bool auto_compact;
    char *default_provider, *default_model;
} Ui;

static Ui *U;
static uint64_t g_last_spin;

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

// ---- header (BuiltInHeader's layout; Saetta Code's own logo and wording) ---------------------------------

static void key_hint(Buf *b, const char *key, const char *desc) {
    th_fgz(b, T_DIM, key);
    Buf d = {0};
    buf_printf(&d, " %s", desc);
    th_fg(b, T_MUTED, buf_str(&d));
    buf_free(&d);
}


// The bolt glyph (U+21AF, one cell wide in monospace fonts) in a warm amber, the theme's accent
// when the terminal reports no colors.
static void bolt_color(Buf *b) {
    int ap = th_appearance();
    if (!ap) buf_adds(b, th_open(T_ACCENT));
    else if (ap == 2) buf_adds(b, "\x1b[38;2;179;100;0m");
    else buf_adds(b, "\x1b[38;2;255;176;32m");
}

static void header_render(void *ctx, int width, Lines *out) {
    (void)ctx;
    Buf t = {0};
    // the title row: the bolt, the name, the version (pi's logo row)
    bolt_color(&t);
    buf_adds(&t, "\x1b[1m\xe2\x86\xaf Saetta Code\x1b[22m\x1b[39m");
    Buf v = {0};
    buf_printf(&v, " v%s", U->cfg->version ? U->cfg->version : "dev");
    th_fg(&t, T_DIM, buf_str(&v));
    buf_free(&v);
    buf_addc(&t, '\n');
    if (!U->header_expanded) {
        Buf sep = {0};
        th_fgz(&sep, T_MUTED, " \xc2\xb7 ");
        key_hint(&t, "escape", "interrupt");
        buf_addstr(&t, buf_str(&sep));
        key_hint(&t, "ctrl+c/ctrl+d", "clear/exit");
        buf_addstr(&t, buf_str(&sep));
        key_hint(&t, "/", "commands");
        buf_addstr(&t, buf_str(&sep));
        key_hint(&t, "@", "files");
        buf_addstr(&t, buf_str(&sep));
        key_hint(&t, "ctrl+o", "more");
        buf_free(&sep);
        buf_addc(&t, '\n');
        th_fgz(&t, T_DIM, "Press ctrl+o to show full startup help.");
    } else {
        static const char *const hints[][2] = {
            {"escape", "to interrupt"}, {"ctrl+c", "to clear"}, {"ctrl+c twice", "to exit"}, {"ctrl+d", "to exit (empty)"},
            {"ctrl+z", "to suspend"}, {"ctrl+k", "to delete to end"}, {"shift+tab", "to cycle thinking level"},
            {"ctrl+p/shift+ctrl+p", "to cycle models"}, {"ctrl+l", "to select model"}, {"ctrl+o", "to expand tools"},
            {"ctrl+t", "to expand thinking"}, {"ctrl+g", "for external editor"}, {"/", "for commands"},
            {"@", "to attach files"}, {"shift+enter", "for a new line"}, {"alt+enter", "to queue follow-up"},
            {"alt+up", "to edit all queued messages"}, {"page up/down, wheel", "to scroll the transcript"},
            {"ctrl+home/ctrl+end", "to jump to the top / the latest message"},
        };
        for (size_t i = 0; i < COUNT(hints); i++) {
            if (i) buf_addc(&t, '\n');
            key_hint(&t, hints[i][0], hints[i][1]);
        }
    }
    buf_adds(&t, "\n\n");
    th_fgz(&t, T_DIM, "Ask about your code, or how to use Saetta Code.");
    Comp *c = comp_text(buf_str(&t), 1, 0);
    lines_addz(out, "");
    comp_render(c, width, out);
    lines_addz(out, "");
    comp_free(c);
    buf_free(&t);
    code_res_render(width, U->header_expanded, out); // the loaded resources (code_skills.c)
    cmds_header_notice(width, out); // the model scope (/scoped-models)
}

// ---- assistant messages (assistant-message.ts) ------------------------------------------------------

static const MdStyle md_plain = {0};
static const MdStyle md_thinking = {.has_color = true, .color = T_THINKING_TEXT, .italic = true};
static const MdStyle md_user = {.has_color = true, .color = T_USER_TEXT, .user = true, .has_bg = true, .bg = TH_BG_USER_MESSAGE};

struct AsstView {
    Comp base;
    Arena a;
    JValue *msg;
};

static void stream_render(int width, Lines *out);
static bool stream_live(const AsstView *v);

static void asst_render(Comp *c, int width, Lines *out) {
    AsstView *v = (AsstView *)c;
    if (stream_live(v)) { // the message in flight: drawn from the streamed blocks themselves
        stream_render(width, out);
        return;
    }
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

static const CompVt asst_vt = {asst_render, asst_destroy, false};

static void asst_set(AsstView *v, const JValue *msg) {
    arena_reset(&v->a);
    v->msg = jcopy(&v->a, msg);
    comp_invalidate(&v->base);
}

static AsstView *asst_new(const JValue *msg) {
    AsstView *v = xcalloc(1, sizeof *v);
    v->base.vt = &asst_vt;
    asst_set(v, msg);
    return v;
}

// Every component rendered again (the theme changed).
static void invalidate_all(void) {
    Comp *cs[] = {U->chat, U->pending, U->status};
    for (size_t k = 0; k < COUNT(cs); k++) {
        CompVec *kids = comp_children(cs[k]);
        for (size_t i = 0; i < kids->len; i++) comp_invalidate(kids->ptr[i]);
    }
}

// Every component of the chat with this vtable rendered again (a setting they read changed).
static void invalidate_kind(const CompVt *vt) {
    CompVec *kids = comp_children(U->chat);
    for (size_t i = 0; i < kids->len; i++)
        if (kids->ptr[i]->vt == vt) comp_invalidate(kids->ptr[i]);
}

// ---- tool views (tool-execution.ts and the built-in renderers) --------------------------------------

struct ToolView {
    Comp base;
    Arena a;
    char *name, *id;
    char *diff; // the edit's rendered diff (computed once per result)
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

// Model-written text shown in a tool view (arguments, diffs): escape sequences dropped and every
// run of other control characters made one space (pi's replaceControlCharacters), newlines and
// tabs kept where the view shows several lines. Nothing the model writes reaches the terminal as
// a sequence.
static Str clean(Buf *out, Str s, bool multiline) {
    out->len = 0;
    Buf t = {0};
    tui_strip(&t, s);
    bool run = false;
    for (size_t i = 0; i < t.len;) {
        unsigned char c = (unsigned char)t.ptr[i];
        bool ctl = c < 0x20 || c == 0x7f;
        size_t n = 1;
        if (c == 0xc2 && i + 1 < t.len && (unsigned char)t.ptr[i + 1] >= 0x80 && (unsigned char)t.ptr[i + 1] <= 0x9f) {
            ctl = true; // C1
            n = 2;
        }
        if (ctl && multiline && (c == '\n' || c == '\t')) ctl = false;
        if (ctl) {
            if (!run) buf_addc(out, ' ');
            run = true;
        } else {
            buf_add(out, t.ptr + i, n);
            run = false;
        }
        i += n;
    }
    buf_free(&t);
    if (!out->ptr) buf_adds(out, "");
    return buf_str(out);
}

static void shorten_path(Buf *b, Str p) {
    Buf c = {0};
    p = clean(&c, p, false);
    const char *home = U->cfg->home;
    size_t hl = home ? strlen(home) : 0;
    if (hl && p.len >= hl && !memcmp(p.ptr, home, hl) && (p.len == hl || p.ptr[hl] == '/')) {
        buf_addc(b, '~');
        buf_addstr(b, str_sub(p, hl, p.len));
    } else buf_addstr(b, p);
    buf_free(&c);
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
    Buf pc = {0};
    Str p = clean(&pc, json_strs(path), false);
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
    buf_free(&pc);
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
        else if (json_strs(cmd).len) {
            Buf c = {0};
            buf_addstr(&t, clean(&c, json_strs(cmd), true));
            buf_free(&c);
        }
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
    Buf cleaned = {0};
    Str c = clean(&cleaned, json_strs(content), true);
    if (!c.len) {
        buf_free(&cleaned);
        return;
    }
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
    buf_free(&cleaned);
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
                tui_intra_line(buf_str(&ro), buf_str(&ao), &rt, &at);
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

// Box(paddingX, paddingY, bg): the lines padded and on the panel's background.
static void box_lines(Lines *kids, int width, int pad_x, int pad_y, ThemeBgToken bg, Lines *out) {
    if (!kids->n) return;
    for (int k = 0; k < pad_y; k++) {
        Buf l = {0};
        th_bg_line(&l, bg, S(""), width);
        lines_addb(out, &l);
    }
    for (size_t i = 0; i < kids->n; i++) {
        Buf p = {0}, l = {0};
        tui_spaces(&p, pad_x);
        buf_adds(&p, kids->v[i]);
        th_bg_line(&l, bg, buf_str(&p), width);
        lines_addb(out, &l);
        buf_free(&p);
    }
    for (int k = 0; k < pad_y; k++) {
        Buf l = {0};
        th_bg_line(&l, bg, S(""), width);
        lines_addb(out, &l);
    }
}

static ThemeBgToken tool_bg(const ToolView *v) {
    return v->partial || !v->has_result ? TH_BG_TOOL_PENDING : v->is_error ? TH_BG_TOOL_ERROR : TH_BG_TOOL_SUCCESS;
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
        const JValue *dv = v->has_result && !v->is_error ? json_get(json_get(v->result, "details"), "diff") : NULL;
        if (json_str(dv)) {
            lines_addz(&kids, "");
            if (!v->diff) {
                Buf d = {0}, c = {0};
                render_diff(clean(&c, json_strs(dv), true), &d);
                v->diff = d.ptr ? d.ptr : strdup("");
                buf_free(&c);
            }
            text_lines(str(v->diff), 0, cw, &kids);
        }
        lines_addz(out, "");
        box_lines(&kids, width, 1, 1, tool_bg(v), out);
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
    if (!strcmp(n, SUBAGENT_TOOL) && json_get(v->args, "prompt")) { // a subagent (code_subagent_ui.c)
        lines_addz(out, "");
        code_subagent_ui_render(v->args, v->result, v->has_result, v->partial, v->is_error, v->expanded, cw, &kids);
        box_lines(&kids, width, 1, 1, tool_bg(v), out);
        lines_free(&kids);
        return;
    }
    if (!strncmp(n, "mcp__", 5)) { // an MCP tool (code_mcp_ui.c)
        lines_addz(out, "");
        code_mcp_ui_render_tool(n, v->args, v->result, text_output(v->result), v->has_result, v->is_error, v->expanded, cw, &kids);
        box_lines(&kids, width, 1, 1, tool_bg(v), out);
        lines_free(&kids);
        return;
    }
    bool known = !strcmp(n, "read") || !strcmp(n, "bash") || !strcmp(n, "write") || !strcmp(n, "grep") || !strcmp(n, "find") ||
                 !strcmp(n, "ls");
    lines_addz(out, "");
    if (!known) {
        // no renderer: the name, the arguments as JSON, the output (formatToolExecution)
        Buf t = {0}, cn = {0};
        title(&t, clean(&cn, str(n), false).ptr);
        buf_free(&cn);
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
        Lines tl = {0};
        comp_render(tc, width, &tl);
        for (size_t i = 0; i < tl.n; i++) {
            Buf l = {0};
            th_bg_line(&l, tool_bg(v), str(tl.v[i]), width);
            lines_addb(out, &l);
        }
        lines_free(&tl);
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
                Buf p = {0}, cp = {0};
                Str ps = clean(&cp, json_strs(pat), false);
                buf_printf(&p, "/%.*s/", (int)ps.len, ps.ptr);
                buf_free(&cp);
                th_fg(&call, T_ACCENT, buf_str(&p));
                buf_free(&p);
            }
            Buf in = {0};
            buf_adds(&in, " in ");
            shorten_path(&in, json_strs(path).len ? json_strs(path) : S("."));
            th_fg(&call, T_TOOL_OUTPUT, buf_str(&in));
            buf_free(&in);
            if (json_strs(json_get(v->args, "glob")).len) {
                Buf g = {0}, cg = {0};
                Str gs = clean(&cg, json_strs(json_get(v->args, "glob")), false);
                buf_printf(&g, " (%.*s)", (int)gs.len, gs.ptr);
                buf_free(&cg);
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
            Buf cp = {0};
            th_fg(&call, T_ACCENT, clean(&cp, json_strs(pat), false));
            buf_free(&cp);
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
    box_lines(&kids, width, 1, 1, tool_bg(v), out);
    lines_free(&kids);
}

static void tool_destroy(Comp *c) {
    ToolView *v = (ToolView *)c;
    arena_reset(&v->a);
    free(v->name);
    free(v->id);
    free(v->diff);
    free(v);
}

static const CompVt tool_vt = {tool_render, tool_destroy, false};

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
    comp_invalidate(&v->base);
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
    free(v->diff);
    v->diff = NULL; // rendered again from the new result
    comp_invalidate(&v->base);
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

// ---- the footer's first line: a starship-style prompt (Saetta Code's own) -----------------------------
//
// "~/path/to/folder on ⎇ branch [=$!+»✘?⇡1⇣2] took 13s": the git state comes from `git status --porcelain=v2
// --branch` and the stash count, run on their own thread with a 1 s limit, at start, after the tools
// that change files and at the end of each run; the UI never waits for them.

typedef struct {
    bool repo;
    char branch[128];
    int conflicted, stashed, modified, staged, renamed, deleted, untracked, ahead, behind;
} GitState;

static pthread_mutex_t g_git_lock = PTHREAD_MUTEX_INITIALIZER;
static GitState g_git;
static _Atomic bool g_git_running, g_git_again, g_git_changed, g_git_quit;
static char g_git_cwd[4096];  // the git thread's own
static char g_git_next[4096]; // the folder of the next look (g_git_lock)
static _Atomic int g_git_wake = -1;
static pthread_t g_git_th;
static bool g_git_started;

// footer-data-provider.ts findGitPaths: the HEAD file of the repository holding cwd (a .git
// folder, or a worktree's .git file "gitdir: <path>"), walking up; false when there is none.
static bool find_git_head(const char *cwd, char *head, size_t cap) {
    char dir[4096];
    snprintf(dir, sizeof dir, "%s", cwd);
    for (;;) {
        char gp[4200];
        snprintf(gp, sizeof gp, "%s/.git", dir[1] ? dir : "");
        struct stat st;
        if (stat(gp, &st) == 0) {
            if (S_ISREG(st.st_mode)) {
                char buf[4200];
                FILE *f = fopen(gp, "r");
                if (!f) return false;
                size_t n = fread(buf, 1, sizeof buf - 1, f);
                fclose(f);
                buf[n] = 0;
                char *t = buf;
                while (*t && isspace((unsigned char)*t)) t++;
                size_t l = strlen(t);
                while (l && isspace((unsigned char)t[l - 1])) t[--l] = 0;
                if (!strncmp(t, "gitdir: ", 8)) {
                    char *g = t + 8;
                    while (*g && isspace((unsigned char)*g)) g++;
                    // resolve(dir, gitdir) + "/HEAD"
                    if (g[0] == '/') snprintf(head, cap, "%s/HEAD", g);
                    else snprintf(head, cap, "%s/%s/HEAD", dir[1] ? dir : "", g);
                    return access(head, F_OK) == 0;
                }
            } else if (S_ISDIR(st.st_mode)) {
                snprintf(head, cap, "%s/HEAD", gp);
                return access(head, F_OK) == 0;
            }
        }
        char *sl = strrchr(dir, '/');
        if (!sl) return false;
        if (sl == dir) {
            if (!dir[1]) return false;
            dir[1] = 0;
            continue;
        }
        *sl = 0;
    }
}

// argv's stdout in the workspace, up to 1 s (the child is stopped after); false if it failed or timed out.
static bool run_out(const char *const *argv, Buf *out) { return tui_run((char *const *)argv, g_git_cwd, S(""), out, 1000); }

// The read-only calls never run what a repository's own config names: no fsmonitor, hooks,
// pager, signature check (gpg.program) or submodule walk, whatever .git/config says (the user's
// global config still applies).
#define GIT_SAFE "git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", "-c", \
                 "core.pager=cat", "-c", "log.showSignature=false", "-c", "status.submoduleSummary=false"

// footer-data-provider.ts resolveGitBranchSync: "ref: refs/heads/<branch>" is the branch (the
// reftable placeholder ".invalid" asks `git symbolic-ref`, which runs no filter or hook); any other
// HEAD is detached, shown as its commit's short id when HEAD holds one, else "detached".
static void head_branch(const char *head, GitState *g) {
    char buf[512] = {0};
    // a regular file only: a FIFO or a link to a device (the repository's choice) would block or
    // read the terminal
    int fd = open(head, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) return;
    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        close(fd);
        return;
    }
    ssize_t n = read(fd, buf, sizeof buf - 1);
    close(fd);
    if (n < 0) return;
    buf[n] = 0;
    char *t = buf;
    while (*t && isspace((unsigned char)*t)) t++;
    size_t l = strlen(t);
    while (l && isspace((unsigned char)t[l - 1])) t[--l] = 0;
    // the file is the repository's, not git's checked output: a control character (an escape
    // sequence for the terminal) is never drawn
    // also the C1 controls (U+0080..U+009F, bytes C2 80..9F) and the bidi overrides and
    // isolates (U+202A..U+202E, U+2066..U+2069)
    for (size_t i = 0; i < l; i++)
        if ((unsigned char)t[i] < 0x20 || t[i] == 0x7f ||
            ((unsigned char)t[i] == 0xC2 && i + 1 < l && (unsigned char)t[i + 1] >= 0x80 && (unsigned char)t[i + 1] <= 0x9F) ||
            ((unsigned char)t[i] == 0xE2 && i + 2 < l &&
             (((unsigned char)t[i + 1] == 0x80 && (unsigned char)t[i + 2] >= 0xAA && (unsigned char)t[i + 2] <= 0xAE) ||
              ((unsigned char)t[i + 1] == 0x81 && (unsigned char)t[i + 2] >= 0xA6 && (unsigned char)t[i + 2] <= 0xA9)))) {
            snprintf(g->branch, sizeof g->branch, "detached");
            return;
        }
    if (!strncmp(t, "ref: refs/heads/", 16)) {
        if (strcmp(t + 16, ".invalid")) {
            snprintf(g->branch, sizeof g->branch, "%s", t + 16);
            return;
        }
        static const char *const sr[] = {GIT_SAFE, "symbolic-ref", "--quiet", "--short", "HEAD", NULL};
        Buf o = {0};
        if (run_out(sr, &o)) {
            Str b = str_trim(buf_str(&o));
            if (b.len) snprintf(g->branch, sizeof g->branch, "%.*s", (int)b.len, b.ptr);
        }
        buf_free(&o);
        if (!g->branch[0]) snprintf(g->branch, sizeof g->branch, "detached");
        return;
    }
    size_t hex = 0;
    while (isxdigit((unsigned char)t[hex])) hex++;
    if (hex >= 40 && !t[hex]) snprintf(g->branch, sizeof g->branch, "%.7s", t);
    else snprintf(g->branch, sizeof g->branch, "detached");
}

static void *git_main(void *arg) {
    (void)arg;
    for (;;) {
        do {
            atomic_store(&g_git_again, false);
            pthread_mutex_lock(&g_git_lock);
            snprintf(g_git_cwd, sizeof g_git_cwd, "%s", g_git_next);
            pthread_mutex_unlock(&g_git_lock);
            GitState g = {0};
            Buf o = {0};
            static const char *const st[] = {GIT_SAFE, "status", "--porcelain=v2", "--branch", "--ignore-submodules=all", NULL};
            char head[4300];
            bool found = find_git_head(g_git_cwd, head, sizeof head);
            // git status refreshes the index, which runs the repository's own filter drivers
            // (.git/config, .gitattributes): only in a folder trusted by an explicit decision
            // (code_trust.c). Elsewhere the branch alone, read from HEAD as pi's footer does.
            bool repo = found && code_trust_explicit(g_git_cwd);
            if (found && !repo) {
                head_branch(head, &g);
                g.repo = true;
            }
            bool ran = repo && run_out(st, &o);
            if (repo && !ran) { // a timeout or a failure: what is known stays
                buf_free(&o);
                continue;
            }
            if (ran) {
                g.repo = true;
                char oid[16] = {0};
                Str all = buf_str(&o);
                for (size_t i = 0; i < all.len;) {
                    size_t j = i;
                    while (j < all.len && all.ptr[j] != '\n') j++;
                    Str l = str_sub(all, i, j);
                    i = j + 1;
                    if (str_starts(l, S("# branch.head "))) {
                        Str h = str_sub(l, 14, l.len);
                        if (!str_eqz(h, "(detached)")) snprintf(g.branch, sizeof g.branch, "%.*s", (int)h.len, h.ptr);
                    } else if (str_starts(l, S("# branch.oid "))) snprintf(oid, sizeof oid, "%.7s", l.ptr + 13);
                    else if (str_starts(l, S("# branch.ab "))) sscanf(l.ptr + 12, "+%d -%d", &g.ahead, &g.behind);
                    else if (l.len > 3 && (l.ptr[0] == '1' || l.ptr[0] == '2')) {
                        char x = l.ptr[2], y = l.ptr[3];
                        if (l.ptr[0] == '2') g.renamed++;
                        else if (x != '.' && x != 'D') g.staged++;
                        if (x == 'D' || y == 'D') g.deleted++;
                        if (y == 'M' || y == 'T') g.modified++;
                    } else if (l.len && l.ptr[0] == 'u') g.conflicted++;
                    else if (l.len && l.ptr[0] == '?') g.untracked++;
                }
                if (!g.branch[0] && oid[0] && strcmp(oid, "(initial")) snprintf(g.branch, sizeof g.branch, "%s", oid);
                Buf s2 = {0};
                static const char *const stash[] = {GIT_SAFE, "stash", "list", NULL};
                if (run_out(stash, &s2))
                    for (size_t k = 0; k < s2.len; k++) g.stashed += s2.ptr[k] == '\n';
                buf_free(&s2);
            }
            buf_free(&o);
            pthread_mutex_lock(&g_git_lock);
            g_git = g;
            pthread_mutex_unlock(&g_git_lock);
            atomic_store(&g_git_changed, true);
            int wfd = atomic_load(&g_git_wake);
            if (wfd >= 0) (void)!write(wfd, "g", 1);
        } while (atomic_load(&g_git_again) && !atomic_load(&g_git_quit));
        atomic_store(&g_git_running, false);
        // a refresh asked between the check above and the store: this thread takes it again, unless
        // a new thread already did (it set running itself)
        if (!atomic_load(&g_git_again) || atomic_load(&g_git_quit) || atomic_exchange(&g_git_running, true)) break;
    }
    return NULL;
}

// Another look at the repository (a refresh asked while one runs is done right after it).
static void git_refresh(void) {
    if (atomic_load(&g_git_quit)) return;
    pthread_mutex_lock(&g_git_lock);
    snprintf(g_git_next, sizeof g_git_next, "%s", U->cfg->cwd);
    pthread_mutex_unlock(&g_git_lock);
    if (atomic_exchange(&g_git_running, true)) {
        atomic_store(&g_git_again, true);
        return;
    }
    if (g_git_started) pthread_join(g_git_th, NULL); // the last one ended (running was false)
    g_git_started = false;
    atomic_store(&g_git_wake, g_wake[1]);
    if (pthread_create(&g_git_th, NULL, git_main, NULL)) atomic_store(&g_git_running, false);
    else g_git_started = true;
}

static uint64_t g_turn_start_ns;
static double g_last_turn_s = -1;

// starship's duration: 13s, 1m20s, 1h2m
static void fmt_took(Buf *b, double s) {
    long t = (long)s;
    if (t < 60) buf_printf(b, "%lds", t);
    else if (t < 3600) buf_printf(b, "%ldm%lds", t / 60, t % 60);
    else buf_printf(b, "%ldh%ldm", t / 3600, (t % 3600) / 60);
}

static void prompt_line(Buf *out) {
    int ap = th_appearance();
    bool saetta = U->theme && !strcmp(U->theme, "saetta");
    uint8_t amber[3], purple[3];
    theme_brand(ap == 2, amber, purple);
    // the folder as pi shows it: the working directory with the home folder as ~
    const char *cwd = U->cfg->cwd, *home = getenv("HOME");
    size_t hl = home ? strlen(home) : 0;
    char where[4096];
    char real_home[PATH_MAX];
    const char *rh = home && realpath(home, real_home) ? real_home : NULL;
    size_t rl = rh ? strlen(rh) : 0;
    // the home folder as given, or its real path (a /tmp or /var home is /private/... once resolved)
    if (hl > 1 && !strncmp(cwd, home, hl) && (cwd[hl] == '/' || cwd[hl] == 0)) snprintf(where, sizeof where, "~%s", cwd + hl);
    else if (rl > 1 && !strncmp(cwd, rh, rl) && (cwd[rl] == '/' || cwd[rl] == 0)) snprintf(where, sizeof where, "~%s", cwd + rl);
    else snprintf(where, sizeof where, "%s", cwd);
    const char *base = where;
    // the folder in bold cyan
    if (saetta) buf_adds(out, ap == 2 ? "\x1b[1m\x1b[38;2;0;122;164m" : "\x1b[1m\x1b[38;2;100;210;255m");
    else buf_adds(out, "\x1b[1m\x1b[36m");
    buf_adds(out, base);
    buf_adds(out, "\x1b[22m\x1b[39m");
    GitState g;
    pthread_mutex_lock(&g_git_lock);
    g = g_git;
    pthread_mutex_unlock(&g_git_lock);
    if (g.repo && g.branch[0]) {
        buf_adds(out, " on ");
        Buf br = {0};
        buf_printf(&br, "\xe2\x8e\x87 %s", g.branch);
        if (saetta) {
            buf_printf(out, "\x1b[1m\x1b[38;2;%d;%d;%dm", purple[0], purple[1], purple[2]);
            buf_addstr(out, buf_str(&br));
            buf_adds(out, "\x1b[22m\x1b[39m");
        } else {
            buf_adds(out, "\x1b[1m");
            th_fg(out, T_ACCENT, buf_str(&br));
            buf_adds(out, "\x1b[22m");
        }
        buf_free(&br);
        Buf st = {0};
        if (g.conflicted) buf_addc(&st, '=');
        if (g.stashed) buf_addc(&st, '$');
        if (g.modified) buf_addc(&st, '!');
        if (g.staged) buf_addc(&st, '+');
        if (g.renamed) buf_adds(&st, "\xc2\xbb");
        if (g.deleted) buf_adds(&st, "\xe2\x9c\x98");
        if (g.untracked) buf_addc(&st, '?');
        if (g.ahead) buf_printf(&st, "\xe2\x87\xa1%d", g.ahead);
        if (g.behind) buf_printf(&st, "\xe2\x87\xa3%d", g.behind);
        if (st.len) {
            Buf br2 = {0};
            buf_printf(&br2, "[%s]", st.ptr);
            buf_addc(out, ' ');
            buf_adds(out, "\x1b[1m");
            th_fg(out, T_ERROR, buf_str(&br2));
            buf_adds(out, "\x1b[22m");
            buf_free(&br2);
        }
        buf_free(&st);
    }
    sess_ui_footer_name(out);
    if (g_last_turn_s >= 2) {
        Buf d = {0};
        fmt_took(&d, g_last_turn_s);
        buf_adds(out, " took ");
        if (saetta) {
            buf_printf(out, "\x1b[1m\x1b[38;2;%d;%d;%dm", amber[0], amber[1], amber[2]);
            buf_addstr(out, buf_str(&d));
            buf_adds(out, "\x1b[22m\x1b[39m");
        } else {
            buf_adds(out, "\x1b[1m");
            th_fg(out, T_WARNING, buf_str(&d));
            buf_adds(out, "\x1b[22m");
        }
        buf_free(&d);
    }
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
    bool ctx_unknown = false;
    sess_ui_footer(&in, &outp, &cr, &cwr, &cost, &ctx_unknown);
    double window = (double)U->model.context_window;
    double pct = window > 0 && !ctx_unknown ? tokens / window * 100 : 0;
    (void)cfg;
    Buf pwd = {0};
    prompt_line(&pwd);
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
    if (ctx_unknown) buf_printf(&ctxs, "?/%s%s", t.ptr, U->auto_compact ? " (auto)" : "");
    else buf_printf(&ctxs, "%.1f%%/%s%s", pct, t.ptr, U->auto_compact ? " (auto)" : "");
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
    // a pack: its name and quantization, shortened to fit (the "(mixed ...)" part, then the
    // quantization dropped) before the line is cut; another model: its id, as pi shows it
    const char *names[] = {U->model.label, U->model.label_brief, U->model.label ? U->model.name : NULL,
                           U->model.id ? U->model.id : "no-model"};
    int rw = 0;
    for (size_t c = 0; c < COUNT(names); c++) {
        const char *model = names[c];
        if (!model) continue;
        right.len = 0;
        if (U->model.reasoning) {
            const char *lv = U->thinking ? U->thinking : "off";
            if (!strcmp(lv, "off")) buf_printf(&right, "%s \xe2\x80\xa2 thinking off", model);
            else buf_printf(&right, "%s \xe2\x80\xa2 %s", model, lv);
        } else buf_adds(&right, model);
        rw = tui_width(buf_str(&right));
        if (sw + 2 + rw <= width || !U->model.label || c >= 2) break;
    }
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
    buf_addstr(&p, buf_str(&pwd));
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
    if (!v->ptr) buf_adds(v, ""); // the value is read as a C string: always terminated
    if (k->kind == K_CHAR && k->ch >= 0x20) {
        char enc[4];
        int n = tui_utf8(enc, k->ch);
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

static size_t available_levels(void) { return U->model.reasoning ? 5 : 1; }

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
    if (U->sel_data && U->sel_free) U->sel_free(U->sel_data);
    U->sel_data = NULL;
    U->sel_free = NULL;
    U->sel = SEL_NONE;
    sel_free(&U->list);
    buf_free(&U->search.value);
    U->search.cursor = 0;
}

// ---- calls from the feature files (code_res_ui.h) ------------------------------------------------------

static CodeUiPanel g_panel;

void code_ui_panel_open(CodeUiPanel p) {
    close_selector();
    g_panel = p;
    U->sel = SEL_PANEL;
    U->sel_data = p.ctx;
    U->sel_free = p.destroy;
    request_render();
}

void code_ui_panel_close(void) {
    if (U->sel == SEL_PANEL) close_selector();
    request_render();
}

void code_ui_status(const char *msg) { show_status(msg); }
void code_ui_warning(const char *msg) { show_warning(msg); }
void code_ui_chat_add(Comp *c) { chat_add(c); }
size_t code_ui_chat_len(void) { return comp_children(U->chat)->len; }
bool code_ui_expanded(void) { return U && U->tools_expanded; }
bool code_ui_busy(void) { return U->busy; }
const CodeUiConfig *code_ui_config(void) { return U->cfg; }

static void set_thinking(const char *level) {
    free(U->thinking);
    U->thinking = dupz(level);
    editor_set_border(U->ed, th_thinking_color(level));
    if (U->busy) request_thinking(level); // the agent thread applies it at its next turn
    else if (U->ops->set_thinking) U->ops->set_thinking(U->host, level);
}

static void open_thinking(void) {
    close_selector();
    U->sel = SEL_THINKING;
    thinking_build(NULL, NULL);
}

// ---- the model selector (model-selector.ts) ---------------------------------------------------------

typedef struct ProvDlg ProvDlg;

typedef struct {
    size_t *idx; // indexes into cfg->models, sorted (current, default, then by provider), then MODEL_ADD / MODEL_EDIT
    size_t n, all;
    size_t selected;
    ProvDlg *dlg; // adding, editing or removing an OpenAI-compatible provider, in the selector's place
} ModelSel;

// the entries after the models (Saetta Code's own: pi adds providers with /login)
#define MODEL_ADD SIZE_MAX
#define MODEL_EDIT (SIZE_MAX - 1)
#define MODEL_ADD_LABEL "Add an OpenAI-compatible model..."
#define MODEL_EDIT_LABEL "Edit or remove an OpenAI-compatible provider..."

static bool is_current(const CodeUiModel *m) {
    return U->model.provider && U->model.id && !strcmp(m->provider, U->model.provider) && !strcmp(m->id, U->model.id);
}

static bool is_default_model(const CodeUiModel *m) {
    return U->default_provider && U->default_model && !strcmp(m->provider, U->default_provider) && !strcmp(m->id, U->default_model);
}

static const CodeUiModel *g_sort_models;
static int model_cmp(const void *a, const void *b) {
    const CodeUiModel *x = &g_sort_models[*(const size_t *)a], *y = &g_sort_models[*(const size_t *)b];
    bool xc = is_current(x), yc = is_current(y);
    if (xc != yc) return xc ? -1 : 1;
    bool xd = is_default_model(x), yd = is_default_model(y);
    if (xd != yd) return xd ? -1 : 1;
    int c = strcasecmp(x->provider, y->provider);
    if (c) return c;
    return *(const size_t *)a < *(const size_t *)b ? -1 : 1;
}

// getDistinctModelName: the name when it has words the id does not.
static bool name_distinct(const char *id, const char *name) {
    if (!name || !*name) return false;
    Buf words = {0};
    for (const char *p = id; *p; p++) buf_addc(&words, isalnum((unsigned char)*p) ? (char)tolower((unsigned char)*p) : ' ');
    Buf compact = {0};
    for (const char *p = id; *p; p++)
        if (isalnum((unsigned char)*p)) buf_addc(&compact, (char)tolower((unsigned char)*p));
    bool all = true;
    for (const char *p = name; *p && all;) {
        while (*p && !isalnum((unsigned char)*p)) p++;
        const char *e = p;
        while (*e && isalnum((unsigned char)*e)) e++;
        if (e > p) {
            char w[128];
            size_t n = MIN((size_t)(e - p), sizeof w - 1);
            for (size_t k = 0; k < n; k++) w[k] = (char)tolower((unsigned char)p[k]);
            w[n] = 0;
            // a whole id token, or (longer than one character) inside the compact id
            bool tok = false;
            for (const char *q = words.ptr ? words.ptr : ""; *q && !tok;) {
                while (*q == ' ') q++;
                const char *qe = q;
                while (*qe && *qe != ' ') qe++;
                if ((size_t)(qe - q) == n && !strncmp(q, w, n)) tok = true;
                q = qe;
            }
            if (!tok && !(n > 1 && compact.ptr && strstr(compact.ptr, w))) all = false;
        }
        p = e;
    }
    buf_free(&words);
    buf_free(&compact);
    return !all;
}

static void prov_dlg_free(ProvDlg *d);

static void model_sel_free(void *p) {
    prov_dlg_free(((ModelSel *)p)->dlg);
    free(((ModelSel *)p)->idx);
    free(p);
}

// ---- OpenAI-compatible providers from /model (Saetta Code's own; models.json in pi's format) ------------
//
// Adding asks, one input at a time (pi's ExtensionInputComponent, extension-input.ts), for the
// provider's name, base URL, API key and model ids, and writes the provider into the agent folder's
// models.json (code_cfg.h cfg_openai_provider), so pi lists the same models. Editing picks one of
// models.json's providers (pi's ExtensionSelectorComponent, extension-selector.ts), then Edit (the
// same inputs, filled in) or Remove (confirmed).

enum { DLG_NAME, DLG_URL, DLG_KEY, DLG_IDS, DLG_PICK, DLG_ACTION, DLG_REMOVE };

struct ProvDlg {
    int step;
    char *edit;    // the provider being edited or removed, NULL when adding
    char *val[4];  // name, base URL, API key, model ids
    Input in;
    char *error;   // why the last value was refused
    char **opts;   // the choices of a list step
    size_t nopts, sel;
};

static void prov_dlg_free(ProvDlg *d) {
    if (!d) return;
    free(d->edit);
    for (int i = 0; i < 4; i++) free(d->val[i]);
    buf_free(&d->in.value);
    free(d->error);
    for (size_t i = 0; i < d->nopts; i++) free(d->opts[i]);
    free(d->opts);
    free(d);
}

static void dlg_opts(ProvDlg *d, const char *const *opts, size_t n) {
    for (size_t i = 0; i < d->nopts; i++) free(d->opts[i]);
    free(d->opts);
    d->opts = xcalloc(n + 1, sizeof *d->opts);
    for (size_t i = 0; i < n; i++) d->opts[i] = strdup(opts[i]);
    d->nopts = n;
    d->sel = 0;
}

// The input of a step, holding what was given for it before.
static void dlg_step(ProvDlg *d, int step) {
    d->step = step;
    free(d->error);
    d->error = NULL;
    d->in.value.len = 0;
    d->in.cursor = 0;
    if (step <= DLG_IDS && d->val[step]) {
        buf_adds(&d->in.value, d->val[step]);
        d->in.cursor = d->in.value.len;
    }
}

// The providers /model edits: models.json's openai-completions ones, saetta's own left out.
static size_t editable_providers(Arena *a, const char ***out) {
    JValue *m = cfg_read_models(a, U->cfg->agent_dir);
    const JValue *pv = json_get(m, "providers");
    size_t n = 0;
    *out = arena_alloc(a, (json_is(pv, J_OBJECT) ? pv->obj.len : 0) * sizeof **out + sizeof **out);
    for (size_t i = 0; json_is(pv, J_OBJECT) && i < pv->obj.len; i++) {
        const JMember *p = &pv->obj.items[i];
        const char *api = json_str(json_get(p->val, "api"));
        if (str_eqz(p->key, "saetta") || (api && strcmp(api, "openai-completions"))) continue;
        (*out)[n++] = str_z(a, p->key);
    }
    return n;
}

static void open_model(const char *search);

static void dlg_start(ModelSel *ms, bool add) {
    ProvDlg *d = xcalloc(1, sizeof *d);
    ms->dlg = d;
    if (add) {
        dlg_step(d, DLG_NAME);
        return;
    }
    Arena a = {0};
    const char **names;
    size_t n = editable_providers(&a, &names);
    dlg_opts(d, names, n);
    arena_reset(&a);
    d->step = DLG_PICK;
}

// The provider's current values into the inputs (editing).
static void dlg_load(ProvDlg *d) {
    Arena a = {0};
    const JValue *p = json_get(json_get(cfg_read_models(&a, U->cfg->agent_dir), "providers"), d->edit);
    d->val[0] = strdup(d->edit);
    d->val[1] = dupz(json_str(json_get(p, "baseUrl")));
    const char *key = json_str(json_get(p, "apiKey"));
    d->val[2] = dupz(key && strcmp(key, "none") ? key : NULL);
    Buf ids = {0};
    const JValue *ms = json_get(p, "models");
    for (size_t i = 0; json_is(ms, J_ARRAY) && i < ms->arr.len; i++) {
        const char *id = json_str(json_get(ms->arr.items[i], "id"));
        if (id) buf_printf(&ids, "%s%s", ids.len ? ", " : "", id);
    }
    d->val[3] = ids.len ? strndup(ids.ptr, ids.len) : NULL;
    buf_free(&ids);
    arena_reset(&a);
}

// The list again, from the models.json just written, filtered to that provider.
static void dlg_done(const char *search) {
    if (U->ops->reload_models) {
        size_t n = 0;
        const CodeUiModel *models = U->ops->reload_models(U->host, &n);
        if (models) {
            CodeUiConfig *cfg = (CodeUiConfig *)U->cfg; // code_main.c's own, not const there
            cfg->models = models;
            cfg->nmodels = n;
        }
    }
    char *s = dupz(search);
    open_model(s);
    free(s);
}

static void dlg_save(ProvDlg *d) {
    Arena a = {0};
    const JValue *old = d->edit ? json_get(json_get(cfg_read_models(&a, U->cfg->agent_dir), "providers"), d->edit) : NULL;
    JValue *e = cfg_openai_provider(&a, old, d->val[1], d->val[2], d->val[3]);
    Buf why = {0};
    if (!cfg_put_provider(U->cfg->agent_dir, d->val[0], d->edit, e, &why)) {
        free(d->error);
        d->error = strndup(why.ptr ? why.ptr : "", why.len);
        buf_free(&why);
        arena_reset(&a);
        return;
    }
    const JValue *ms = json_get(e, "models");
    Buf st = {0};
    size_t nm = json_is(ms, J_ARRAY) ? ms->arr.len : 0;
    buf_printf(&st, "%s provider %s (%zu model%s) in models.json", d->edit ? "Saved" : "Added", d->val[0], nm, nm == 1 ? "" : "s");
    char *name = strdup(d->val[0]);
    arena_reset(&a);
    dlg_done(name);
    show_status(st.ptr);
    buf_free(&st);
    free(name);
}

// Enter on an input: the value checked, then the next step.
static void dlg_submit(ProvDlg *d) {
    Str v = str_trim(buf_str(&d->in.value));
    const char *err = NULL;
    Buf eb = {0};
    if (d->step == DLG_NAME) {
        bool bad = false;
        for (size_t i = 0; i < v.len; i++) bad |= v.ptr[i] == '/' || v.ptr[i] == ' ' || v.ptr[i] == '\t';
        Arena a = {0};
        JValue *pv = json_get(cfg_read_models(&a, U->cfg->agent_dir), "providers");
        char *nm = str_z(&a, v);
        if (!v.len) err = "Enter a name for the provider";
        else if (bad) err = "The name cannot hold spaces or \"/\"";
        else if (!strcmp(nm, "saetta")) err = "saetta is Saetta Code's own provider; choose another name";
        else if (json_get(pv, nm) && (!d->edit || strcmp(d->edit, nm))) {
            buf_printf(&eb, "models.json has a provider named %s already; edit it from /model", nm);
            err = eb.ptr;
        }
        arena_reset(&a);
    } else if (d->step == DLG_URL) {
        if (!str_starts(v, S("http://")) && !str_starts(v, S("https://"))) err = "The base URL starts with http:// or https://";
    } else if (d->step == DLG_IDS) {
        if (!v.len) err = "Enter at least one model id";
    }
    if (err) {
        free(d->error);
        d->error = strdup(err);
        buf_free(&eb);
        return;
    }
    free(d->val[d->step]);
    d->val[d->step] = strndup(v.ptr, v.len);
    if (d->step == DLG_IDS) dlg_save(d);
    else dlg_step(d, d->step + 1);
}

static void dlg_render(ProvDlg *d, int width, Lines *out) {
    Buf t = {0};
    border_line(out, width, T_BORDER);
    lines_addz(out, "");
    if (d->step >= DLG_PICK) {
        // ExtensionSelectorComponent
        Buf title = {0};
        if (d->step == DLG_PICK) buf_adds(&title, "Edit or remove an OpenAI-compatible provider");
        else if (d->step == DLG_ACTION) buf_printf(&title, "Provider %s", d->edit);
        else buf_printf(&title, "Remove provider %s from models.json?", d->edit);
        buf_adds(&t, "\x1b[1m");
        th_fg(&t, T_ACCENT, buf_str(&title));
        buf_adds(&t, "\x1b[22m");
        text_lines(buf_str(&t), 1, width, out);
        buf_free(&title);
        lines_addz(out, "");
        if (!d->nopts) {
            t.len = 0;
            th_fgz(&t, T_MUTED, "No OpenAI-compatible providers in models.json");
            text_lines(buf_str(&t), 1, width, out);
        }
        const size_t maxv = 10;
        size_t start = d->sel > maxv / 2 ? d->sel - maxv / 2 : 0;
        if (d->nopts > maxv && start > d->nopts - maxv) start = d->nopts - maxv;
        size_t end = MIN(start + maxv, d->nopts);
        for (size_t i = start; i < end; i++) {
            t.len = 0;
            if (i == d->sel) {
                th_fgz(&t, T_ACCENT, "\xe2\x86\x92 ");
                th_fgz(&t, T_ACCENT, d->opts[i]);
            } else {
                buf_adds(&t, "  ");
                th_fgz(&t, T_TEXT, d->opts[i]);
            }
            text_lines(buf_str(&t), 1, width, out);
        }
        if (start > 0 || end < d->nopts) {
            Buf si = {0};
            buf_printf(&si, "  (%zu/%zu)", d->sel + 1, d->nopts);
            t.len = 0;
            th_fg(&t, T_MUTED, buf_str(&si));
            text_lines(buf_str(&t), 0, width, out);
            buf_free(&si);
        }
        lines_addz(out, "");
        t.len = 0;
        key_hint(&t, "\xe2\x86\x91\xe2\x86\x93", "navigate");
        buf_adds(&t, "  ");
        key_hint(&t, "enter", "select");
        buf_adds(&t, "  ");
        key_hint(&t, "escape/ctrl+c", "cancel");
        text_lines(buf_str(&t), 1, width, out);
    } else {
        // ExtensionInputComponent
        static const char *const titles[] = {"provider name", "base URL", "API key", "model ids"};
        static const char *const descs[] = {
            "The provider's name in models.json, such as ollama or lmstudio.",
            "The server's OpenAI-compatible API, such as http://127.0.0.1:11434/v1 (chat completions).",
            "Optional: the key, or $NAME to read it from that environment variable. Empty: none.",
            "One or more ids the server knows the models by, separated by commas.",
        };
        Buf title = {0};
        buf_printf(&title, "%s (%d/4): %s", d->edit ? "Edit an OpenAI-compatible provider" : "Add an OpenAI-compatible model",
                   d->step + 1, titles[d->step]);
        th_fg(&t, T_ACCENT, buf_str(&title));
        text_lines(buf_str(&t), 1, width, out);
        buf_free(&title);
        lines_addz(out, "");
        t.len = 0;
        th_fgz(&t, T_TEXT, descs[d->step]);
        text_lines(buf_str(&t), 1, width, out);
        lines_addz(out, "");
        input_render(&d->in, width, out);
        lines_addz(out, "");
        t.len = 0;
        key_hint(&t, "enter", "submit");
        buf_adds(&t, "  ");
        key_hint(&t, "escape/ctrl+c", "cancel");
        text_lines(buf_str(&t), 1, width, out);
    }
    if (d->error) {
        lines_addz(out, "");
        t.len = 0;
        th_fgz(&t, T_ERROR, d->error);
        text_lines(buf_str(&t), 1, width, out);
    }
    lines_addz(out, "");
    border_line(out, width, T_BORDER);
    buf_free(&t);
}

static void dlg_key(ModelSel *ms, const Key *k) {
    ProvDlg *d = ms->dlg;
    bool cancel = k->kind == K_ESC || (k->kind == K_CTRL && k->ch == 'c');
    if (cancel) { // back to the list
        prov_dlg_free(d);
        ms->dlg = NULL;
        return;
    }
    if (d->step < DLG_PICK) {
        if (k->kind == K_ENTER) dlg_submit(d);
        else if (input_key(&d->in, k)) {
            free(d->error); // the refused value is being changed
            d->error = NULL;
        }
        return;
    }
    if ((k->kind == K_UP || (k->kind == K_CHAR && k->ch == 'k')) && d->sel > 0) d->sel--;
    else if ((k->kind == K_DOWN || (k->kind == K_CHAR && k->ch == 'j')) && d->sel + 1 < d->nopts) d->sel++;
    else if (k->kind == K_ENTER && d->sel < d->nopts) {
        const char *o = d->opts[d->sel];
        if (d->step == DLG_PICK) {
            free(d->edit);
            d->edit = strdup(o);
            d->step = DLG_ACTION;
            dlg_opts(d, (const char *const[]){"Edit", "Remove"}, 2);
        } else if (d->step == DLG_ACTION && !strcmp(o, "Edit")) {
            dlg_load(d);
            dlg_step(d, DLG_NAME);
        } else if (d->step == DLG_ACTION) {
            d->step = DLG_REMOVE;
            dlg_opts(d, (const char *const[]){"Yes", "No"}, 2);
        } else if (!strcmp(o, "Yes")) {
            Buf why = {0};
            if (!cfg_put_provider(U->cfg->agent_dir, d->edit, NULL, NULL, &why)) {
                free(d->error);
                d->error = strndup(why.ptr ? why.ptr : "", why.len);
            } else {
                Buf st = {0};
                buf_printf(&st, "Removed provider %s from models.json", d->edit);
                dlg_done(NULL);
                show_status(st.ptr);
                buf_free(&st);
            }
            buf_free(&why);
        } else {
            prov_dlg_free(d);
            ms->dlg = NULL;
        }
    }
}

// The add and edit entries after the models, when the query matches their words (fuzzyMatch).
static void model_filter_extra(ModelSel *ms, const char *query) {
    Arena a = {0};
    const char **names;
    bool any = editable_providers(&a, &names) > 0;
    arena_reset(&a);
    const char *labels[] = {MODEL_ADD_LABEL, any ? MODEL_EDIT_LABEL : NULL};
    const size_t ids[] = {MODEL_ADD, MODEL_EDIT};
    for (size_t e = 0; e < 2; e++) {
        if (!labels[e]) continue;
        bool ok = true;
        const char *h = labels[e];
        for (const char *q = query ? query : ""; *q && ok; q++) {
            if (*q == ' ') continue;
            while (*h && tolower((unsigned char)*h) != tolower((unsigned char)*q)) h++;
            if (!*h) ok = false;
            else h++;
        }
        if (ok) ms->idx[ms->n++] = ids[e];
    }
}

static void model_filter(ModelSel *ms, const char *query) {
    const CodeUiModel *ms_all = U->cfg->models;
    size_t n = U->cfg->nmodels;
    size_t *order = xmalloc((n + 1) * sizeof *order);
    for (size_t i = 0; i < n; i++) order[i] = i;
    g_sort_models = ms_all;
    qsort(order, n, sizeof *order, model_cmp);
    free(ms->idx);
    ms->idx = xmalloc((n + 3) * sizeof *ms->idx);
    ms->n = 0;
    for (size_t k = 0; k < n; k++) {
        const CodeUiModel *m = &ms_all[order[k]];
        bool ok = true;
        if (query && *query) {
            Buf hay = {0};
            buf_printf(&hay, "%s %s/%s %s %s%s%s%s", m->provider, m->provider, m->id, m->provider, m->id, m->name ? " " : "",
                       m->name ? m->name : "", is_default_model(m) ? " default" : "");
            for (const char *q = query; *q && ok;) {
                while (*q == ' ') q++;
                const char *e = q;
                while (*e && *e != ' ') e++;
                const char *h = hay.ptr;
                for (const char *c = q; c < e && ok; c++) {
                    while (*h && tolower((unsigned char)*h) != tolower((unsigned char)*c)) h++;
                    if (!*h) ok = false;
                    else h++;
                }
                q = e;
            }
            buf_free(&hay);
        }
        if (ok) ms->idx[ms->n++] = order[k];
    }
    free(order);
    model_filter_extra(ms, query);
    ms->selected = 0;
    if (!query || !*query)
        for (size_t i = 0; i < ms->n; i++)
            if (ms->idx[i] < n && is_current(&ms_all[ms->idx[i]])) ms->selected = i;
}

static void model_render(int width, Lines *out) {
    ModelSel *ms = U->sel_data;
    if (ms->dlg) {
        dlg_render(ms->dlg, width, out);
        return;
    }
    border_line(out, width, T_BORDER);
    lines_addz(out, "");
    Buf w = {0};
    th_fgz(&w, T_WARNING, "Only showing models from configured providers. Use /login to add providers.");
    text_lines(buf_str(&w), 0, width, out);
    buf_free(&w);
    lines_addz(out, "");
    input_render(&U->search, width, out);
    lines_addz(out, "");
    const int maxv = 10;
    long start = (long)ms->selected - maxv / 2;
    if (start > (long)ms->n - maxv) start = (long)ms->n - maxv;
    if (start < 0) start = 0;
    size_t end = MIN((size_t)start + maxv, ms->n);
    for (size_t i = (size_t)start; i < end; i++) {
        bool sel = i == ms->selected;
        if (ms->idx[i] == MODEL_ADD || ms->idx[i] == MODEL_EDIT) {
            Buf l = {0};
            const char *label = ms->idx[i] == MODEL_ADD ? MODEL_ADD_LABEL : MODEL_EDIT_LABEL;
            if (sel) {
                th_fgz(&l, T_ACCENT, "\xe2\x86\x92   ");
                th_fgz(&l, T_ACCENT, label);
            } else {
                buf_adds(&l, "    ");
                buf_adds(&l, label);
            }
            text_lines(buf_str(&l), 0, width, out);
            buf_free(&l);
            continue;
        }
        const CodeUiModel *m = &U->cfg->models[ms->idx[i]];
        Buf l = {0};
        if (sel) th_fgz(&l, T_ACCENT, "\xe2\x86\x92 ");
        else buf_adds(&l, "  ");
        if (is_current(m)) th_fgz(&l, T_ACCENT, "\xe2\x9c\x93 ");
        else buf_adds(&l, "  ");
        if (m->pack) {
            // a model run on this Mac: its name, where it stands
            const char *nm = m->name ? m->name : m->id;
            if (m->too_big) th_fgz(&l, T_DIM, nm);
            else if (sel) th_fgz(&l, T_ACCENT, nm);
            else buf_adds(&l, nm);
            Buf st = {0};
            if (m->too_big) buf_printf(&st, " [needs %u GB memory]", m->min_ram_gb);
            else if (!m->unavailable) buf_adds(&st, " [on this Mac]");
            else buf_printf(&st, " [not downloaded, %u GB]", m->disk_gb);
            th_fg(&l, m->too_big ? T_DIM : T_MUTED, buf_str(&st));
            buf_free(&st);
            text_lines(buf_str(&l), 0, width, out);
            buf_free(&l);
            continue;
        }
        if (sel) th_fgz(&l, T_ACCENT, m->id);
        else buf_adds(&l, m->id);
        if (name_distinct(m->id, m->name)) {
            Buf nm = {0};
            buf_printf(&nm, " (%s)", m->name);
            th_fg(&l, T_MUTED, buf_str(&nm));
            buf_free(&nm);
        }
        buf_addc(&l, ' ');
        Buf pb = {0};
        buf_printf(&pb, "[%s]", m->provider);
        th_fg(&l, T_MUTED, buf_str(&pb));
        buf_free(&pb);
        if (is_default_model(m)) th_fgz(&l, T_MUTED, " \xc2\xb7 default");
        text_lines(buf_str(&l), 0, width, out);
        buf_free(&l);
    }
    if (start > 0 || end < ms->n) {
        Buf si = {0}, t = {0};
        buf_printf(&si, "  (%zu/%zu)", ms->selected + 1, ms->n);
        th_fg(&t, T_MUTED, buf_str(&si));
        text_lines(buf_str(&t), 0, width, out);
        buf_free(&si);
        buf_free(&t);
    }
    Buf t = {0};
    if (!ms->n) th_fgz(&t, T_MUTED, "  No matching models");
    else {
        lines_addz(out, "");
        Buf nm = {0};
        size_t sx = ms->idx[ms->selected];
        const CodeUiModel *m = sx < U->cfg->nmodels ? &U->cfg->models[sx] : NULL;
        if (m) buf_printf(&nm, "  Model Name: %s", m->name ? m->name : m->id);
        else if (sx == MODEL_ADD) buf_adds(&nm, "  A server with an OpenAI-compatible API, added to models.json (pi lists it too)");
        else buf_adds(&nm, "  Change or remove a provider of models.json");
        th_fg(&t, T_MUTED, buf_str(&nm));
        buf_free(&nm);
    }
    text_lines(buf_str(&t), 0, width, out);
    buf_free(&t);
    lines_addz(out, "");
    th_fgz(&t, T_SUCCESS, "  Model catalogs refreshed.");
    text_lines(buf_str(&t), 0, width, out);
    buf_free(&t);
    lines_addz(out, "");
    th_fgz(&t, T_DIM, "  Enter to select \xc2\xb7 Ctrl+S to set as default \xc2\xb7 Escape/Ctrl+C to cancel");
    text_lines(buf_str(&t), 0, width, out);
    buf_free(&t);
    border_line(out, width, T_BORDER);
}

static void close_selector(void);
static void show_status(const char *msg);

// ---- downloading a pack from /model (saetta download in a child, its progress in the status) ------------

static uint64_t free_disk_bytes(const char *dir) {
    struct statfs st;
    char probe[4096];
    snprintf(probe, sizeof probe, "%s", dir);
    // the nearest folder that exists
    while (statfs(probe, &st) != 0) {
        char *sl = strrchr(probe, '/');
        if (!sl || sl == probe) return 0;
        *sl = 0;
    }
    return (uint64_t)st.f_bavail * st.f_bsize;
}

static void confirm_download(const CodeUiModel *m) {
    char *mdir = NULL;
    Arena a = {0};
    uint64_t freeb = download_models_dir(&a, &mdir) ? 0 : free_disk_bytes(mdir);
    if (getenv("SAETTA_CODE_FREE_GB")) freeb = (uint64_t)atoll(getenv("SAETTA_CODE_FREE_GB")) * 1000000000ull; // tests
    free(U->dl_name);
    free(U->dl_title);
    free(U->dl_models_dir);
    U->dl_name = strdup(m->id);
    U->dl_title = strdup(m->name ? m->name : m->id);
    U->dl_disk = m->disk_gb;
    U->dl_free_gb = (unsigned)(freeb / 1000000000ull);
    U->dl_models_dir = dupz(mdir);
    arena_reset(&a);
    U->sel = SEL_CONFIRM;
}

static void confirm_render(int width, Lines *out) {
    if (U->confirm_switch) {
        border_line(out, width, T_BORDER);
        lines_addz(out, "");
        Buf t = {0};
        buf_printf(&t, "\x1b[1mSwitch to %s?\x1b[22m", U->sw_title);
        text_lines(buf_str(&t), 1, width, out);
        lines_addz(out, "");
        t.len = 0;
        th_fgz(&t, T_MUTED, "Saetta Code starts again on it; this conversation goes on there.");
        text_lines(buf_str(&t), 1, width, out);
        size_t others = U->sw_others; // counted once, when the dialog opened
        if (others) {
            t.len = 0;
            Buf w = {0};
            buf_printf(&w, "%zu other session%s use%s this model: switching disconnects %s.", others, others == 1 ? "" : "s",
                       others == 1 ? "s" : "", others == 1 ? "it" : "them");
            th_fg(&t, T_WARNING, buf_str(&w));
            buf_free(&w);
            text_lines(buf_str(&t), 1, width, out);
        }
        lines_addz(out, "");
        t.len = 0;
        th_fgz(&t, T_DIM, "Enter to switch \xc2\xb7 Escape to cancel");
        text_lines(buf_str(&t), 1, width, out);
        border_line(out, width, T_BORDER);
        buf_free(&t);
        return;
    }
    border_line(out, width, T_BORDER);
    lines_addz(out, "");
    Buf t = {0};
    buf_printf(&t, "\x1b[1mDownload %s?\x1b[22m", U->dl_title);
    text_lines(buf_str(&t), 1, width, out);
    t.len = 0;
    lines_addz(out, "");
    Buf d = {0};
    buf_printf(&d, "About %u GB (%u GB free on that disk). A cancelled download resumes where it stopped.", U->dl_disk, U->dl_free_gb);
    th_fg(&t, U->dl_free_gb < U->dl_disk ? T_ERROR : T_MUTED, buf_str(&d));
    text_lines(buf_str(&t), 1, width, out);
    lines_addz(out, "");
    t.len = 0;
    th_fgz(&t, T_DIM, U->dl_free_gb < U->dl_disk ? "Not enough free disk \xc2\xb7 Escape to cancel" : "Enter to download \xc2\xb7 Escape to cancel");
    text_lines(buf_str(&t), 1, width, out);
    border_line(out, width, T_BORDER);
    buf_free(&t);
    buf_free(&d);
}

static void start_download(void) {
    const char *cmd = getenv("SAETTA_CODE_DOWNLOAD"); // a stand-in for the tests
    int pfd[2];
    if (tui_pipe(pfd)) {
        show_error("Could not start the download");
        return;
    }
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_adddup2(&fa, pfd[1], 2);
    const char *argv[4];
    if (cmd && *cmd) {
        argv[0] = cmd;
        argv[1] = U->dl_name;
        argv[2] = NULL;
    } else {
        argv[0] = U->cfg->self_exe ? U->cfg->self_exe : "saetta";
        argv[1] = "download";
        argv[2] = U->dl_name;
        argv[3] = NULL;
    }
    pid_t pid;
    // its own process group: a cancel stops it and whatever it started
    int rc = tui_spawn(&pid, (char *const *)argv, &fa, false, true);
    posix_spawn_file_actions_destroy(&fa);
    close(pfd[1]);
    if (rc) {
        close(pfd[0]);
        show_error("Could not start the download");
        return;
    }
    fcntl(pfd[0], F_SETFL, O_NONBLOCK);
    U->dl_pid = pid;
    U->dl_fd = pfd[0];
    U->dl_line.len = 0;
    Buf m = {0};
    buf_printf(&m, "Downloading %s... (escape to cancel)", U->dl_title);
    editor_set_working(U->ed, m.ptr);
    if (!U->working) g_last_spin = now_ns();
    U->working = true;
    buf_free(&m);
}

static void restart_model(const char *provider, const char *id);

// The child's progress lines (\r-separated: "  12.3 / 76.0 GB  95 MB/s  3/40 files  ETA 12m03s").
static void download_poll(void) {
    if (U->dl_fd < 0) return;
    if (U->dl_cancelled && U->dl_pid > 0 && now_ns() - U->dl_cancel_ns > 2000000000ull) kill(-U->dl_pid, SIGKILL);
    char chunk[4096];
    ssize_t n;
    while ((n = read(U->dl_fd, chunk, sizeof chunk)) > 0) {
        for (ssize_t i = 0; i < n; i++) {
            char c = chunk[i];
            if (c == '\r' || c == '\n') {
                Str l = str_trim(buf_str(&U->dl_line));
                if (l.len && (str_find(l, S(" GB"), 0) >= 0)) {
                    Buf m = {0}, cl = {0};
                    tui_strip(&cl, l);
                    buf_printf(&m, "Downloading %s: %.*s (escape to cancel)", U->dl_title, (int)cl.len, cl.ptr ? cl.ptr : "");
                    editor_set_working(U->ed, m.ptr);
                    buf_free(&m);
                    buf_free(&cl);
                    request_render();
                }
                U->dl_line.len = 0;
            } else buf_addc(&U->dl_line, c);
        }
        // the line being written (download.c rewrites it after each \r)
        Str l = str_trim(buf_str(&U->dl_line));
        if (l.len && str_find(l, S(" GB"), 0) >= 0) {
            Buf m = {0}, cl = {0};
            tui_strip(&cl, l);
            buf_printf(&m, "Downloading %s: %.*s (escape to cancel)", U->dl_title, (int)cl.len, cl.ptr ? cl.ptr : "");
            editor_set_working(U->ed, m.ptr);
            buf_free(&m);
            buf_free(&cl);
            request_render();
        }
    }
    if (n != 0) return; // more to come
    close(U->dl_fd);
    U->dl_fd = -1;
    int st = 0;
    while (waitpid(U->dl_pid, &st, 0) < 0 && errno == EINTR) {}
    U->dl_pid = 0;
    U->working = U->busy;
    editor_set_working(U->ed, U->busy ? "Working" : NULL);
    if (WIFEXITED(st) && WEXITSTATUS(st) == 0) { // finished, even if a cancel came too late
        for (size_t i = 0; i < U->cfg->nmodels; i++) {
            CodeUiModel *mm = (CodeUiModel *)&U->cfg->models[i];
            if (mm->pack && !strcmp(mm->id, U->dl_name)) mm->unavailable = false;
        }
        Buf m = {0};
        buf_printf(&m, "Downloaded %s", U->dl_title);
        show_status(m.ptr);
        buf_free(&m);
        restart_model("saetta", U->dl_name);
    } else if (U->dl_cancelled) {
        show_status("Download stopped; choose the model again to resume it");
    } else {
        Buf m = {0};
        buf_printf(&m, "The download of %s failed; choose it again to resume it (or run saetta download %s)", U->dl_title, U->dl_name);
        show_error(m.ptr);
        buf_free(&m);
    }
    U->dl_cancelled = false;
    request_render();
}

// SIGTERM to the download's process group (download.c keeps its .part files: resumable). The reap
// happens in download_poll once its pipe closes; SIGKILL follows if it does not stop.
static void cancel_download(void) {
    if (U->dl_pid > 0) {
        U->dl_cancelled = true;
        U->dl_cancel_ns = now_ns();
        kill(-U->dl_pid, SIGTERM);
    }
}

// The engine is replaced by starting again on the other model (code_main.c), the session kept.
static void restart_model(const char *provider, const char *id) {
    if (!U->ops->restart) {
        show_error("Could not switch the model");
        return;
    }
    free(U->re_provider);
    free(U->re_id);
    U->re_provider = strdup(provider);
    U->re_id = strdup(id);
    if (!U->busy) U->quit = true;
    else show_status("Switching when the current turn ends");
}

static void choose_model(const CodeUiModel *m, bool persist) {
    if (m->pack && (m->unavailable || m->too_big)) {
        close_selector();
        if (m->too_big) {
            Buf b = {0};
            buf_printf(&b, "%s needs a Mac with %u GB of memory", m->name ? m->name : m->id, m->min_ram_gb);
            show_status(b.ptr);
            buf_free(&b);
        } else if (U->dl_pid > 0) show_status("A download is running already");
        else confirm_download(m);
        return;
    }
    if (is_current(m)) {
        close_selector();
        Buf b = {0};
        buf_printf(&b, "Model: %s", m->id);
        show_status(b.ptr);
        buf_free(&b);
        return;
    }
    if (m->pack || U->cfg->engine) {
        // another pack, or away from the engine in this process: a restart on it, once confirmed
        close_selector();
        free(U->sw_provider);
        free(U->sw_id);
        free(U->sw_title);
        U->sw_provider = strdup(m->provider);
        U->sw_id = strdup(m->id);
        U->sw_title = strdup(m->name ? m->name : m->id);
        U->sw_others = code_share_clients(); // once: the dialog's drawing never counts (a lock, a folder scan)
        U->confirm_switch = true;
        U->sel = SEL_CONFIRM;
        return;
    }
    char *prov = strdup(m->provider), *id = strdup(m->id);
    CodeUiModel copy = *m;
    close_selector();
    bool ok = true;
    if (U->busy) request_model(prov, id); // pi applies it to the next request: the next turn
    else if (U->ops->set_model) ok = U->ops->set_model(U->host, prov, id);
    if (!ok) {
        show_error("Could not switch the model");
    } else {
        free((char *)U->model.provider);
        free((char *)U->model.id);
        free((char *)U->model.name);
        free((char *)U->model.label);
        free((char *)U->model.label_brief);
        U->model = (CodeUiModel){.provider = strdup(prov), .id = strdup(id), .name = dupz(copy.name), .label = dupz(copy.label),
                                 .label_brief = dupz(copy.label_brief), .reasoning = copy.reasoning,
                                 .context_window = copy.context_window};
        if (!copy.reasoning) {
            free(U->thinking);
            U->thinking = strdup("off");
        }
        editor_set_border(U->ed, th_thinking_color(U->thinking));
        Buf b = {0};
        if (persist) {
            free(U->default_provider);
            free(U->default_model);
            U->default_provider = strdup(prov);
            U->default_model = strdup(id);
            buf_printf(&b, "Default model: %s/%s", prov, id);
        } else buf_printf(&b, "Model: %s", id);
        show_status(b.ptr);
        buf_free(&b);
    }
    free(prov);
    free(id);
}

static void open_model(const char *search) {
    close_selector();
    U->sel = SEL_MODEL;
    ModelSel *ms = xcalloc(1, sizeof *ms);
    U->sel_data = ms;
    U->sel_free = model_sel_free;
    if (search) {
        buf_adds(&U->search.value, search);
        U->search.cursor = U->search.value.len;
    }
    model_filter(ms, search);
}

static void model_key(const Key *k) {
    ModelSel *ms = U->sel_data;
    if (ms->dlg) {
        dlg_key(ms, k);
        return;
    }
    size_t sx = ms->n ? ms->idx[ms->selected] : 0;
    if (ms->n && (sx == MODEL_ADD || sx == MODEL_EDIT) && (k->kind == K_ENTER || (k->kind == K_CTRL && k->ch == 's'))) {
        if (k->kind == K_ENTER) dlg_start(ms, sx == MODEL_ADD);
        return;
    }
    if (k->kind == K_UP && ms->n) ms->selected = ms->selected ? ms->selected - 1 : ms->n - 1;
    else if (k->kind == K_DOWN && ms->n) ms->selected = ms->selected + 1 >= ms->n ? 0 : ms->selected + 1;
    else if (k->kind == K_ENTER) {
        if (ms->n) choose_model(&U->cfg->models[ms->idx[ms->selected]], false);
    } else if (k->kind == K_ESC || (k->kind == K_CTRL && k->ch == 'c')) close_selector();
    else if (k->kind == K_CTRL && k->ch == 's') {
        if (ms->n) choose_model(&U->cfg->models[ms->idx[ms->selected]], true);
    } else if (k->kind == K_TAB) {
    } else if (input_key(&U->search, k)) model_filter(ms, U->search.value.ptr ? U->search.value.ptr : "");
}

// ---- the settings selector (settings-selector.ts, settings-list.ts) ---------------------------------------

typedef struct {
    const char *id, *label, *desc;
    const char *values[8];
    char cur[64];
} SetItem;

typedef struct {
    SetItem items[40];
    size_t n;
    size_t *shown; // the filtered items
    size_t nshown;
    size_t selected;
} SetSel;

static const char *setting_str(const char *key, const char *dflt) {
    const JValue *v = json_get(U->cfg->settings, key);
    return json_str(v) ? json_str(v) : dflt;
}

static const char *setting_bool(const char *key, bool dflt) {
    const JValue *v = json_get(U->cfg->settings, key);
    return (json_is(v, J_BOOL) ? v->b : dflt) ? "true" : "false";
}

static void setting_num(char *out, size_t cap, const char *key, int dflt) {
    const JValue *v = json_get(U->cfg->settings, key);
    snprintf(out, cap, "%d", json_is(v, J_INTEGER) ? (int)v->i : dflt);
}

static void set_add(SetSel *s, const char *id, const char *label, const char *desc, const char *cur, const char *const *values) {
    SetItem *it = &s->items[s->n++];
    it->id = id;
    it->label = label;
    it->desc = desc;
    snprintf(it->cur, sizeof it->cur, "%s", cur);
    for (size_t k = 0; values && values[k] && k < 7; k++) it->values[k] = values[k];
}

static void settings_build(SetSel *s) {
    static const char *const tf[] = {"true", "false", NULL};
    static const char *const mode[] = {"one-at-a-time", "all", NULL};
    char num[16];
    set_add(s, "autocompact", "Auto-compact", "Automatically compact context when it gets too large", U->auto_compact ? "true" : "false", tf);
    set_add(s, "auto-resize-images", "Auto-resize images", "Resize large images to 2000x2000 max for better model compatibility",
            setting_bool("autoResizeImages", true), tf);
    set_add(s, "block-images", "Block images", "Prevent images from being sent to LLM providers", setting_bool("blockImages", false), tf);
    set_add(s, "skill-commands", "Skill commands", "Register skills as /skill:name commands", setting_bool("enableSkillCommands", true), tf);
    set_add(s, "show-hardware-cursor", "Show hardware cursor", "Use the terminal cursor instead of Pi's drawn cursor",
            setting_bool("showHardwareCursor", false), tf);
    setting_num(num, sizeof num, "editorPaddingX", 0);
    set_add(s, "editor-padding", "Editor padding", "Horizontal padding for input editor (0-3)", num, (const char *const[]){"0", "1", "2", "3", NULL});
    setting_num(num, sizeof num, "outputPad", 1);
    set_add(s, "output-padding", "Output padding", "Horizontal padding for messages, tool output, and command output", num,
            (const char *const[]){"0", "1", NULL});
    setting_num(num, sizeof num, "autocompleteMaxVisible", 5);
    set_add(s, "autocomplete-max-visible", "Autocomplete max items", "Max visible items in autocomplete dropdown (3-20)", num,
            (const char *const[]){"3", "5", "7", "10", "15", "20", NULL});
    set_add(s, "clear-on-shrink", "Clear on shrink", "Clear empty rows when content shrinks (may cause flicker)",
            setting_bool("clearOnShrink", false), tf);
    set_add(s, "terminal-progress", "Terminal progress", "Show OSC 9;4 progress indicators in the terminal tab bar",
            setting_bool("showTerminalProgress", false), tf);
    set_add(s, "steering-mode", "Steering mode",
            "Enter while streaming queues steering messages. 'one-at-a-time': deliver one, wait for response. 'all': deliver all at once.",
            setting_str("steeringMode", "one-at-a-time"), mode);
    set_add(s, "follow-up-mode", "Follow-up mode",
            "Option+Enter queues follow-up messages until agent stops. 'one-at-a-time': deliver one, wait for response. 'all': deliver all at once.",
            setting_str("followUpMode", "one-at-a-time"), mode);
    set_add(s, "transport", "Transport", "Preferred transport for providers that support multiple transports", setting_str("transport", "auto"),
            (const char *const[]){"sse", "websocket", "websocket-cached", "auto", NULL});
    set_add(s, "share-destination", "Session share destination",
            "Where to upload sessions with /share (auto prefers Radius when authenticated, otherwise GitHub Gist)",
            setting_str("sessionShareDestination", "auto"), (const char *const[]){"auto", "gist", "radius", NULL});
    set_add(s, "http-idle-timeout", "HTTP idle timeout",
            "Maximum idle gap while waiting for HTTP headers or body chunks. Disable for local models that pause longer than five minutes.",
            "5 min", (const char *const[]){"30 sec", "1 min", "2 min", "5 min", "disabled", NULL});
    set_add(s, "cache-warming-mode", "Cache warming",
            "off; streaming while the agent runs; idle also between runs while continuation stays profitable",
            setting_str("cacheWarmingMode", "idle"), (const char *const[]){"off", "streaming", "idle", NULL});
    set_add(s, "hide-thinking", "Hide thinking", "Hide thinking blocks in assistant responses", U->hide_thinking ? "true" : "false", tf);
    set_add(s, "mermaid-rendering", "Mermaid diagrams", "Render Mermaid code blocks as Unicode diagrams",
            setting_str("mermaidRendering", "streaming"), (const char *const[]){"off", "final", "streaming", NULL});
    set_add(s, "cache-miss-notices", "Cache miss notices", "Show transcript notices for cache costs and provider recovery diagnostics",
            setting_bool("showCacheMissNotices", false), tf);
    set_add(s, "collapse-changelog", "Collapse changelog", "Show condensed changelog after updates", setting_bool("collapseChangelog", false), tf);
    set_add(s, "quiet-startup", "Quiet startup", "Disable verbose printing at startup (header: keep only the startup header)",
            setting_bool("quietStartup", false), (const char *const[]){"true", "header", "false", NULL});
    set_add(s, "install-telemetry", "Install telemetry", "Send an anonymous version/update ping after changelog-detected updates",
            setting_bool("enableInstallTelemetry", true), tf);
    set_add(s, "default-project-trust", "Default project trust",
            "Fallback behavior when no extension or saved trust decision decides project trust", "Ask",
            (const char *const[]){"Ask", "Always trust", "Never trust", NULL});
    set_add(s, "double-escape-action", "Double-escape action", "Action when pressing Escape twice with empty editor",
            setting_str("doubleEscapeAction", "tree"), (const char *const[]){"tree", "fork", "none", NULL});
    set_add(s, "tree-filter-mode", "Tree filter mode", "Default filter when opening /tree", setting_str("treeFilterMode", "default"),
            (const char *const[]){"default", "no-tools", "user-only", "labeled-only", "all", NULL});
    set_add(s, "warnings", "Warnings", "Enable or disable individual warnings", "configure", NULL);
    set_add(s, "model-thinking", "Default thinking level per model",
            "Override the default thinking level for specific models. Shift+Tab cycles in-session.", "none", NULL);
    set_add(s, "tui-mode", "TUI mode", "Interface layout; regular mode uses the terminal's normal scrollback", "fullscreen",
            (const char *const[]){"regular", "fullscreen", NULL});
    set_add(s, "fullscreen-exit-output", "Fullscreen exit output",
            "Print the transcript or only a session resume hint when exiting fullscreen mode", setting_str("fullscreenExitOutput", "transcript"),
            (const char *const[]){"transcript", "resume-hint", NULL});
    set_add(s, "fullscreen-scrollbar", "Fullscreen scrollbar", "Scrollbar behavior in fullscreen mode; has no effect in regular mode",
            setting_str("fullscreenScrollbar", "auto"), (const char *const[]){"auto", "always", "hidden", NULL});
    set_add(s, "fullscreen-copy-on-select", "Fullscreen copy on select",
            "Automatically copy selected text in fullscreen mode; disable to copy selections with Ctrl+X",
            setting_bool("fullscreenCopyOnSelect", true), tf);
    set_add(s, "fullscreen-wheel-scroll-lines", "Fullscreen wheel scrolling",
            "Lines per mouse-wheel event in fullscreen mode; 'auto' speeds up fast wheel spins where the terminal does not", "auto",
            (const char *const[]){"auto", "1", "2", "3", "5", "10", NULL});
    set_add(s, "theme", "Theme", "Color theme for the interface", U->theme ? U->theme : "saetta", NULL);
}

static void set_sel_free(void *p) {
    free(((SetSel *)p)->shown);
    free(p);
}

static void settings_filter(SetSel *s, const char *query) {
    free(s->shown);
    s->shown = xmalloc((s->n + 1) * sizeof *s->shown);
    s->nshown = 0;
    for (size_t i = 0; i < s->n; i++) {
        bool ok = true;
        const char *h = s->items[i].label;
        for (const char *q = query ? query : ""; *q && ok; q++) {
            if (*q == ' ') continue;
            while (*h && tolower((unsigned char)*h) != tolower((unsigned char)*q)) h++;
            if (!*h) ok = false;
            else h++;
        }
        if (ok) s->shown[s->nshown++] = i;
    }
    s->selected = 0;
}

static void settings_render(int width, Lines *out) {
    SetSel *s = U->sel_data;
    border_line(out, width, T_BORDER);
    input_render(&U->search, width, out);
    lines_addz(out, "");
    if (!s->nshown) {
        Buf t = {0}, c = {0};
        th_fgz(&t, T_DIM, "  No matching settings");
        tui_truncate(&c, buf_str(&t), width, "...", false);
        lines_addb(out, &c);
        buf_free(&t);
    } else {
        int maxlabel = 0;
        for (size_t i = 0; i < s->n; i++) maxlabel = MAX(maxlabel, tui_width(str(s->items[i].label)));
        maxlabel = MIN(36, maxlabel);
        const int maxv = 10;
        long start = (long)s->selected - maxv / 2;
        if (start > (long)s->nshown - maxv) start = (long)s->nshown - maxv;
        if (start < 0) start = 0;
        size_t end = MIN((size_t)start + maxv, s->nshown);
        for (size_t i = (size_t)start; i < end; i++) {
            const SetItem *it = &s->items[s->shown[i]];
            bool sel = i == s->selected;
            Buf l = {0}, lab = {0}, val = {0}, cut = {0};
            if (sel) th_fgz(&l, T_ACCENT, "\xe2\x86\x92 ");
            else buf_adds(&l, "  ");
            buf_adds(&lab, it->label);
            tui_spaces(&lab, maxlabel - tui_width(str(it->label)));
            if (sel) th_fg(&l, T_ACCENT, buf_str(&lab));
            else buf_addstr(&l, buf_str(&lab));
            buf_adds(&l, "  ");
            tui_truncate(&val, str(it->cur), width - (2 + maxlabel + 2) - 2, "", false);
            th_fg(&l, sel ? T_ACCENT : T_MUTED, buf_str(&val));
            tui_truncate(&cut, buf_str(&l), width, "...", false);
            lines_addb(out, &cut);
            buf_free(&l);
            buf_free(&lab);
            buf_free(&val);
        }
        if (start > 0 || end < s->nshown) {
            char info[64];
            snprintf(info, sizeof info, "  (%zu/%zu)", s->selected + 1, s->nshown);
            Buf t = {0}, c = {0};
            tui_truncate(&c, str(info), width - 2, "", false);
            th_fg(&t, T_DIM, buf_str(&c));
            lines_addb(out, &t);
            buf_free(&c);
        }
        const SetItem *cur = &s->items[s->shown[s->selected]];
        if (cur->desc) {
            lines_addz(out, "");
            Lines w = {0};
            tui_wrap(str(cur->desc), width - 4, &w);
            for (size_t k = 0; k < w.n; k++) {
                Buf t = {0}, d = {0};
                buf_printf(&d, "  %s", w.v[k]);
                th_fg(&t, T_DIM, buf_str(&d));
                lines_addb(out, &t);
                buf_free(&d);
            }
            lines_free(&w);
        }
    }
    lines_addz(out, "");
    Buf h = {0}, c = {0};
    th_fgz(&h, T_DIM, "  Type to search \xc2\xb7 Enter/Space to change \xc2\xb7 Esc to cancel");
    tui_truncate(&c, buf_str(&h), width, "...", false);
    lines_addb(out, &c);
    buf_free(&h);
    border_line(out, width, T_BORDER);
}

static void setting_changed(const char *id, const char *v) {
    if (!strcmp(id, "autocompact")) {
        U->auto_compact = !strcmp(v, "true");
        code_compaction_set_enabled(U->cfg->agent_dir, U->auto_compact); // setAutoCompactionEnabled
    }
    else if (!strcmp(id, "hide-thinking")) {
        U->hide_thinking = !strcmp(v, "true");
        invalidate_kind(&asst_vt);
    }
}

// ---- themes (/theme, theme-selector.ts; the first-start choice) --------------------------------------------

static void invalidate_all(void);

static void apply_theme(const char *name) {
    // the saetta theme follows the terminal's background when it reports one, else macOS's appearance
    int ap = U->have_colors && U->colors.has_bg ? 0 : U->mac_appearance;
    theme_use(name ? name : "system", U->have_colors ? &U->colors : NULL, ap);
    invalidate_all();
    editor_set_border(U->ed, th_thinking_color(U->thinking));
    request_render();
}

// settings.json with "theme" set (written back as pi writes it: two-space JSON)
static bool put_theme(Arena *a, JValue *root, void *ctx) {
    json_put(a, root, S("theme"), json_string(a, str((const char *)ctx)));
    return true;
}

// pi's settings write: under proper-lockfile's lock, the file read again, the field set, written
// whole (code_cfg.c cfg_json_update_ex). A file that does not parse is left alone (pi skips
// saving after a load error).
static void save_theme_setting(const char *name) {
    Buf path = {0}, why = {0};
    buf_printf(&path, "%s/settings.json", U->cfg->agent_dir);
    CfgUpdateOpts o = {.parse_numbers = true,
                       .lock_tries = 20,
                       .locked = "settings.json is locked by another program; the theme is not saved",
                       .unreadable = "settings.json could not be read; the theme is not saved",
                       .unparsable = "settings.json does not parse; the theme is not saved (fix the file, then choose it again)",
                       .unwritable = "Could not save the theme in settings.json"};
    if (!cfg_json_update_ex(path.ptr, &o, put_theme, (void *)name, &why)) show_warning(why.ptr ? why.ptr : "");
    buf_free(&path);
    buf_free(&why);
}

static void open_theme(bool first) {
    close_selector();
    U->sel = SEL_THEME;
    U->theme_first = first;
    free(U->theme_before);
    U->theme_before = dupz(U->theme ? U->theme : "system");
    const char *const *names;
    size_t n = theme_names(&names);
    sel_free(&U->list);
    sel_init(&U->list, 10, 12, 32);
    const char *cur = first ? "saetta" : U->theme ? U->theme : "system";
    for (size_t i = 0; i < n; i++) {
        const char *d = NULL;
        if (first)
            d = !strcmp(names[i], "saetta")   ? "macOS colors, light or dark as this Mac is"
                : !strcmp(names[i], "system") ? "pi's colors, matched to this terminal's"
                : !strcmp(names[i], "dark")   ? "pi's dark theme"
                                              : "pi's light theme";
        else if (!strcmp(names[i], cur)) d = "(current)";
        sel_add(&U->list, names[i], names[i], d);
        if (!strcmp(names[i], cur)) U->list.selected = i;
    }
    if (first) apply_theme(cur);
}

static void theme_render(int width, Lines *out) {
    border_line(out, width, T_BORDER);
    if (U->theme_first) {
        Buf t = {0};
        buf_adds(&t, "\x1b[1m Choose a theme\x1b[22m");
        text_lines(buf_str(&t), 0, width, out);
        t.len = 0;
        th_fgz(&t, T_MUTED, " \xe2\x86\x91/\xe2\x86\x93 to preview \xc2\xb7 Enter to keep it \xc2\xb7 /theme changes it later");
        text_lines(buf_str(&t), 0, width, out);
        buf_free(&t);
    }
    sel_render(&U->list, width, out);
    border_line(out, width, T_BORDER);
}

static void theme_key(const Key *k) {
    if (k->kind == K_UP || k->kind == K_DOWN) {
        if (k->kind == K_UP) sel_up(&U->list);
        else sel_down(&U->list);
        const SelItem *it = sel_current(&U->list);
        if (it) apply_theme(it->value); // live preview
    } else if (k->kind == K_ENTER || (U->theme_first && (k->kind == K_ESC || (k->kind == K_CTRL && k->ch == 'c')))) {
        const SelItem *it = sel_current(&U->list);
        char *name = strdup(it ? it->value : "saetta");
        close_selector();
        free(U->theme);
        U->theme = name;
        apply_theme(name);
        save_theme_setting(name);
        Buf m = {0};
        buf_printf(&m, "Theme: %s", name);
        show_status(m.ptr);
        buf_free(&m);
    } else if (k->kind == K_ESC || (k->kind == K_CTRL && k->ch == 'c')) {
        char *back = U->theme_before ? strdup(U->theme_before) : NULL;
        close_selector();
        apply_theme(back);
        free(back);
    }
}

static void open_settings(void) {
    close_selector();
    U->sel = SEL_SETTINGS;
    SetSel *s = xcalloc(1, sizeof *s);
    U->sel_data = s;
    U->sel_free = set_sel_free;
    settings_build(s);
    settings_filter(s, "");
}

static void settings_key(const Key *k) {
    SetSel *s = U->sel_data;
    if (k->kind == K_UP && s->nshown) s->selected = s->selected ? s->selected - 1 : s->nshown - 1;
    else if (k->kind == K_DOWN && s->nshown) s->selected = s->selected + 1 >= s->nshown ? 0 : s->selected + 1;
    else if (k->kind == K_ENTER || (k->kind == K_CHAR && k->ch == ' ' && !U->search.value.len)) {
        if (!s->nshown) return;
        SetItem *it = &s->items[s->shown[s->selected]];
        if (!strcmp(it->id, "theme")) {
            open_theme(false);
            return;
        }
        if (!it->values[0]) {
            Buf b = {0};
            buf_printf(&b, "%s is not available in Saetta Code yet.", it->label);
            show_status(b.ptr);
            buf_free(&b);
            return;
        }
        size_t nv = 0, ci = 0;
        while (nv < 7 && it->values[nv]) {
            if (!strcmp(it->values[nv], it->cur)) ci = nv;
            nv++;
        }
        snprintf(it->cur, sizeof it->cur, "%s", it->values[(ci + 1) % nv]);
        setting_changed(it->id, it->cur);
    } else if (k->kind == K_ESC || (k->kind == K_CTRL && k->ch == 'c')) close_selector();
    else if (input_key(&U->search, k)) settings_filter(s, U->search.value.ptr ? U->search.value.ptr : "");
}

// ---- the session tree (tree-selector.ts) ------------------------------------------------------------------

typedef struct {
    const JValue *entry;
    int indent;
    bool connector, last;
} TreeRow;

typedef struct {
    Arena a;
    JValue *entries;
    const char *leaf;
    VEC(TreeRow) flat;
    VEC(size_t) shown;
    size_t selected;
    StrMap active; // ids on the path from the root to the leaf
    Buf query;
    int max_visible;
} TreeSel;

static void add_user(const JValue *m);

static bool contains_ci(Str hay, Str needle) {
    if (!needle.len) return true;
    for (size_t i = 0; i + needle.len <= hay.len; i++)
        if (!strncasecmp(hay.ptr + i, needle.ptr, needle.len)) return true;
    return false;
}

static const JValue *entry_by_id(TreeSel *t, Str id) {
    for (size_t i = 1; i < t->entries->arr.len; i++)
        if (str_eq(json_strs(json_get(t->entries->arr.items[i], "id")), id)) return t->entries->arr.items[i];
    return NULL;
}

static bool has_text(const JValue *content) {
    if (json_is(content, J_STRING)) return str_trim(content->s).len > 0;
    for (size_t i = 0; content && json_is(content, J_ARRAY) && i < content->arr.len; i++)
        if (str_eqz(json_strs(json_get(content->arr.items[i], "type")), "text") && str_trim(json_strs(json_get(content->arr.items[i], "text"))).len)
            return true;
    return false;
}

static void content_text(Buf *b, const JValue *content, size_t max) {
    if (json_is(content, J_STRING)) buf_addstr(b, content->s);
    else
        for (size_t i = 0; content && json_is(content, J_ARRAY) && i < content->arr.len; i++)
            if (str_eqz(json_strs(json_get(content->arr.items[i], "type")), "text")) buf_addstr(b, json_strs(json_get(content->arr.items[i], "text")));
    // the first 200 characters, newlines and tabs as spaces, trimmed
    size_t cut = 0, chars = 0;
    while (cut < b->len && chars < max) {
        size_t k = cut;
        tui_cp(buf_str(b), &k);
        cut = k;
        chars++;
    }
    b->len = cut;
    for (size_t i = 0; i < b->len; i++)
        if (b->ptr[i] == '\n' || b->ptr[i] == '\t') b->ptr[i] = ' ';
    Str t = str_trim(buf_str(b));
    if (b->ptr) {
        memmove(b->ptr, t.ptr, t.len);
        b->len = t.len;
        b->ptr[b->len] = 0;
    }
}

static void tool_call_label(TreeSel *t, Buf *b, Str call_id, const char *tool_name) {
    // the call in an assistant message
    const JValue *args = NULL;
    Str name = str(tool_name ? tool_name : "tool");
    bool found = false;
    for (size_t i = 1; i < t->entries->arr.len && !found; i++) {
        const JValue *m = json_get(t->entries->arr.items[i], "message");
        const JValue *c = json_get(m, "content");
        for (size_t k = 0; c && json_is(c, J_ARRAY) && k < c->arr.len; k++)
            if (str_eqz(json_strs(json_get(c->arr.items[k], "type")), "toolCall") && str_eq(json_strs(json_get(c->arr.items[k], "id")), call_id)) {
                args = json_get(c->arr.items[k], "arguments");
                name = json_strs(json_get(c->arr.items[k], "name"));
                found = true;
                break;
            }
    }
    if (!found) {
        buf_printf(b, "[%.*s]", (int)name.len, name.ptr);
        return;
    }
    Buf p = {0};
    const JValue *pv = json_get(args, "path");
    if (!pv) pv = json_get(args, "file_path");
    shorten_path(&p, json_strs(pv).len ? json_strs(pv) : (str_eqz(name, "grep") || str_eqz(name, "find") || str_eqz(name, "ls") ? S(".") : S("")));
    if (str_eqz(name, "read") || str_eqz(name, "write") || str_eqz(name, "edit") || str_eqz(name, "ls"))
        buf_printf(b, "[%.*s: %s]", (int)name.len, name.ptr, p.ptr ? p.ptr : "");
    else if (str_eqz(name, "bash")) {
        Buf c = {0};
        Str raw = json_strs(json_get(args, "command"));
        for (size_t i = 0; i < raw.len; i++) buf_addc(&c, raw.ptr[i] == '\n' || raw.ptr[i] == '\t' ? ' ' : raw.ptr[i]);
        Str tr = str_trim(buf_str(&c));
        buf_printf(b, "[bash: %.*s%s]", (int)MIN(tr.len, 50), tr.ptr, raw.len > 50 ? "..." : "");
        buf_free(&c);
    } else if (str_eqz(name, "grep"))
        buf_printf(b, "[grep: /%s/ in %s]", json_str(json_get(args, "pattern")) ? json_str(json_get(args, "pattern")) : "", p.ptr);
    else if (str_eqz(name, "find"))
        buf_printf(b, "[find: %s in %s]", json_str(json_get(args, "pattern")) ? json_str(json_get(args, "pattern")) : "", p.ptr);
    else {
        Buf j = {0};
        json_write(&j, args);
        buf_printf(b, "[%.*s: %.*s%s]", (int)name.len, name.ptr, (int)MIN(j.len, 40), j.ptr ? j.ptr : "", j.len > 40 ? "..." : "");
        buf_free(&j);
    }
    buf_free(&p);
}

// getEntryDisplayText
static void entry_text(TreeSel *t, const JValue *e, bool selected, Buf *out) {
    Buf r = {0};
    Str type = json_strs(json_get(e, "type"));
    if (str_eqz(type, "message")) {
        const JValue *m = json_get(e, "message");
        Str role = json_strs(json_get(m, "role"));
        Buf c = {0};
        if (str_eqz(role, "user")) {
            content_text(&c, json_get(m, "content"), 200);
            th_fgz(&r, T_ACCENT, "user: ");
            buf_addstr(&r, buf_str(&c));
        } else if (str_eqz(role, "assistant")) {
            content_text(&c, json_get(m, "content"), 200);
            th_fgz(&r, T_SUCCESS, "assistant: ");
            if (c.len) buf_addstr(&r, buf_str(&c));
            else if (str_eqz(json_strs(json_get(m, "stopReason")), "aborted")) th_fgz(&r, T_MUTED, "(aborted)");
            else if (json_str(json_get(m, "errorMessage"))) {
                Buf em = {0};
                Str es = json_strs(json_get(m, "errorMessage"));
                for (size_t i = 0; i < es.len; i++) buf_addc(&em, es.ptr[i] == '\n' || es.ptr[i] == '\t' ? ' ' : es.ptr[i]);
                Str tr = str_trim(buf_str(&em));
                th_fg(&r, T_ERROR, str_n(tr.ptr, MIN(tr.len, 80)));
                buf_free(&em);
            } else th_fgz(&r, T_MUTED, "(no content)");
        } else if (str_eqz(role, "toolResult")) {
            Buf l = {0};
            tool_call_label(t, &l, json_strs(json_get(m, "toolCallId")), json_str(json_get(m, "toolName")));
            th_fg(&r, T_MUTED, buf_str(&l));
            buf_free(&l);
        } else {
            Buf l = {0};
            buf_printf(&l, "[%.*s]", (int)role.len, role.ptr);
            th_fg(&r, T_DIM, buf_str(&l));
            buf_free(&l);
        }
        buf_free(&c);
    } else if (str_eqz(type, "compaction")) {
        Buf l = {0};
        buf_printf(&l, "[compaction: %.0fk tokens]", round(jnum(json_get(e, "tokensBefore")) / 1000));
        th_fg(&r, T_BORDER_ACCENT, buf_str(&l));
        buf_free(&l);
    } else if (str_eqz(type, "branch_summary")) {
        th_fgz(&r, T_WARNING, "[branch summary]: ");
        buf_addstr(&r, str_trim(json_strs(json_get(e, "summary"))));
    } else {
        Buf l = {0};
        if (str_eqz(type, "model_change")) buf_printf(&l, "[model: %s]", json_str(json_get(e, "modelId")) ? json_str(json_get(e, "modelId")) : "");
        else if (str_eqz(type, "thinking_level_change"))
            buf_printf(&l, "[thinking: %s]", json_str(json_get(e, "thinkingLevel")) ? json_str(json_get(e, "thinkingLevel")) : "");
        else if (str_eqz(type, "context_edit"))
            buf_printf(&l, "[context %s: %s]", json_is(json_get(e, "replacement"), J_NULL) ? "omit" : "replace",
                       json_str(json_get(e, "targetId")) ? json_str(json_get(e, "targetId")) : "");
        else if (str_eqz(type, "custom")) buf_printf(&l, "[custom: %s]", json_str(json_get(e, "customType")) ? json_str(json_get(e, "customType")) : "");
        else if (str_eqz(type, "label")) buf_printf(&l, "[label: %s]", json_str(json_get(e, "label")) ? json_str(json_get(e, "label")) : "(cleared)");
        if (l.len) th_fg(&r, T_DIM, buf_str(&l));
        buf_free(&l);
    }
    if (selected) {
        buf_adds(out, "\x1b[1m");
        buf_addstr(out, buf_str(&r));
        buf_adds(out, "\x1b[22m");
    } else buf_addstr(out, buf_str(&r));
    buf_free(&r);
}

static bool tree_hidden(TreeSel *t, const JValue *e) {
    Str type = json_strs(json_get(e, "type"));
    if (str_eqz(type, "usage")) return true;
    bool leaf = t->leaf && str_eqz(json_strs(json_get(e, "id")), t->leaf);
    if (str_eqz(type, "message") && !leaf) {
        const JValue *m = json_get(e, "message");
        if (str_eqz(json_strs(json_get(m, "role")), "assistant")) {
            Str stop = json_strs(json_get(m, "stopReason"));
            bool err = stop.len && !str_eqz(stop, "stop") && !str_eqz(stop, "toolUse");
            if (!has_text(json_get(m, "content")) && !err) return true;
        }
    }
    return str_eqz(type, "label") || str_eqz(type, "context_edit") || str_eqz(type, "custom") || str_eqz(type, "model_change") ||
           str_eqz(type, "thinking_level_change") || str_eqz(type, "session_info");
}

// flattenTree: depth first, the branch holding the leaf first, children of a branching node indented.
static void tree_flatten(TreeSel *t) {
    size_t n = t->entries->arr.len;
    // children lists
    VEC(size_t) *kids = xcalloc(n + 1, sizeof *kids);
    VEC(size_t) roots = {0};
    for (size_t i = 1; i < n; i++) {
        const JValue *p = json_get(t->entries->arr.items[i], "parentId");
        long pi = -1;
        if (json_is(p, J_STRING))
            for (size_t k = 1; k < n; k++)
                if (str_eq(json_strs(json_get(t->entries->arr.items[k], "id")), p->s)) {
                    pi = (long)k;
                    break;
                }
        if (pi < 0) vec_push(&roots, i);
        else vec_push(&kids[pi], i);
    }
    bool *active = xcalloc(n + 1, sizeof *active);
    for (size_t i = 1; i < n; i++) active[i] = strmap_has(&t->active, json_strs(json_get(t->entries->arr.items[i], "id")));
    typedef struct {
        size_t node;
        int indent;
        bool just_branched, connector, last;
    } Item;
    VEC(Item) stack = {0};
    bool multi = roots.len > 1;
    for (size_t i = roots.len; i-- > 0;) vec_push(&stack, ((Item){roots.ptr[i], multi ? 1 : 0, multi, multi, i == roots.len - 1}));
    while (stack.len) {
        Item it = stack.ptr[--stack.len];
        vec_push(&t->flat, ((TreeRow){t->entries->arr.items[it.node], multi ? MAX(0, it.indent - 1) : it.indent, it.connector, it.last}));
        size_t nk = kids[it.node].len;
        bool many = nk > 1;
        int ci = many ? it.indent + 1 : (it.just_branched && it.indent > 0) ? it.indent + 1 : it.indent;
        // the active child first
        VEC(size_t) ord = {0};
        for (size_t k = 0; k < nk; k++)
            if (active[kids[it.node].ptr[k]]) vec_push(&ord, kids[it.node].ptr[k]);
        for (size_t k = 0; k < nk; k++)
            if (!active[kids[it.node].ptr[k]]) vec_push(&ord, kids[it.node].ptr[k]);
        for (size_t k = ord.len; k-- > 0;) vec_push(&stack, ((Item){ord.ptr[k], ci, many, many, k == ord.len - 1}));
        vec_free(&ord);
    }
    for (size_t i = 0; i <= n; i++) vec_free(&kids[i]);
    free(kids);
    free(active);
    vec_free(&roots);
    vec_free(&stack);
}

static void tree_filter(TreeSel *t, bool keep) {
    const JValue *prev = keep && t->shown.len ? t->flat.ptr[t->shown.ptr[t->selected]].entry : NULL;
    t->shown.len = 0;
    for (size_t i = 0; i < t->flat.len; i++) {
        const JValue *e = t->flat.ptr[i].entry;
        if (tree_hidden(t, e)) continue;
        if (t->query.len) {
            Buf hay = {0};
            entry_text(t, e, false, &hay);
            Buf plain = {0};
            tui_strip(&plain, buf_str(&hay));
            bool ok = true;
            Str q = buf_str(&t->query);
            for (size_t p = 0; p < q.len && ok;) {
                while (p < q.len && q.ptr[p] == ' ') p++;
                size_t s0 = p;
                while (p < q.len && q.ptr[p] != ' ') p++;
                if (p > s0 && !contains_ci(buf_str(&plain), str_sub(q, s0, p))) ok = false;
            }
            buf_free(&hay);
            buf_free(&plain);
            if (!ok) continue;
        }
        vec_push(&t->shown, i);
    }
    t->selected = t->shown.len ? t->shown.len - 1 : 0;
    // the leaf (or the nearest visible entry above it), else the one kept
    const char *target = prev ? json_str(json_get(prev, "id")) : t->leaf;
    for (const char *id = target; id;) {
        bool found = false;
        for (size_t k = 0; k < t->shown.len; k++)
            if (!strcmp(json_str(json_get(t->flat.ptr[t->shown.ptr[k]].entry, "id")), id)) {
                t->selected = k;
                found = true;
                break;
            }
        if (found) break;
        const JValue *e = entry_by_id(t, str(id));
        const JValue *p = e ? json_get(e, "parentId") : NULL;
        id = json_is(p, J_STRING) ? p->s.ptr : NULL;
    }
}

static void tree_render(int width, Lines *out) {
    TreeSel *t = U->sel_data;
    lines_addz(out, "");
    border_line(out, width, T_BORDER);
    Buf b = {0};
    buf_adds(&b, "\x1b[1m  Session Tree\x1b[22m");
    text_lines(buf_str(&b), 1, width, out);
    b.len = 0;
    // the key help, wrapped on item boundaries (TreeHelp)
    static const char *const help[] = {"\xe2\x86\x91/\xe2\x86\x93 move", "\xe2\x86\x90/\xe2\x86\x92 page", "option+\xe2\x86\x90/\xe2\x86\x92 branch",
                                       "ctrl+x copy", "shift+l label", "shift+t label time", "filters ctrl+d/t/u/l/a",
                                       "cycle ctrl+o/shift+ctrl+o"};
    Buf cur = {0};
    for (size_t i = 0; i < COUNT(help); i++) {
        Buf cand = {0};
        if (cur.len) buf_printf(&cand, "%s \xc2\xb7 %s", cur.ptr, help[i]);
        else buf_printf(&cand, "  %s", help[i]);
        if (!cur.len || tui_width(buf_str(&cand)) <= width) {
            buf_free(&cur);
            cur = cand;
            continue;
        }
        Lines w = {0};
        tui_wrap(buf_str(&cur), width, &w);
        for (size_t k = 0; k < w.n; k++) {
            Buf l = {0};
            th_fgz(&l, T_MUTED, w.v[k]);
            lines_addb(out, &l);
        }
        lines_free(&w);
        buf_free(&cur);
        buf_printf(&cur, "  %s", help[i]);
        buf_free(&cand);
    }
    if (cur.len) {
        Lines w = {0};
        tui_wrap(buf_str(&cur), width, &w);
        for (size_t k = 0; k < w.n; k++) {
            Buf l = {0};
            th_fgz(&l, T_MUTED, w.v[k]);
            lines_addb(out, &l);
        }
        lines_free(&w);
    }
    buf_free(&cur);
    // the search line, focused
    buf_adds(&b, "  ");
    th_fgz(&b, T_MUTED, "Type to search:");
    buf_addc(&b, ' ');
    if (t->query.len) th_fg(&b, T_ACCENT, buf_str(&t->query));
    Buf c = {0};
    tui_truncate(&c, buf_str(&b), MAX(0, width - 1), "...", false);
    buf_adds(&c, "\x1b[7m \x1b[27m");
    lines_addb(out, &c);
    buf_free(&b);
    border_line(out, width, T_BORDER);
    lines_addz(out, "");
    if (!t->shown.len) {
        Buf l = {0};
        th_fgz(&l, T_MUTED, "  No entries found");
        lines_addb(out, &l);
        th_fgz(&l, T_MUTED, "  (0/0)");
        lines_addb(out, &l);
    } else {
        long start = (long)t->selected - t->max_visible / 2;
        if (start > (long)t->shown.len - t->max_visible) start = (long)t->shown.len - t->max_visible;
        if (start < 0) start = 0;
        size_t end = MIN((size_t)start + (size_t)t->max_visible, t->shown.len);
        for (size_t i = (size_t)start; i < end; i++) {
            const TreeRow *row = &t->flat.ptr[t->shown.ptr[i]];
            bool sel = i == t->selected;
            Buf l = {0}, prefix = {0};
            if (sel) th_fgz(&l, T_ACCENT, "\xe2\x80\xba ");
            else buf_adds(&l, "  ");
            for (int k = 0; k < row->indent; k++) {
                if (k == row->indent - 1 && row->connector) buf_adds(&prefix, row->last ? "\xe2\x94\x94\xe2\x94\x80 " : "\xe2\x94\x9c\xe2\x94\x80 ");
                else buf_adds(&prefix, "   ");
            }
            th_fg(&l, T_DIM, buf_str(&prefix));
            if (strmap_has(&t->active, json_strs(json_get(row->entry, "id")))) th_fgz(&l, T_ACCENT, "\xe2\x80\xa2 ");
            entry_text(t, row->entry, sel, &l);
            Buf cut = {0};
            tui_truncate(&cut, buf_str(&l), width, "", false);
            lines_addb(out, &cut);
            buf_free(&l);
            buf_free(&prefix);
        }
        Buf info = {0}, l = {0};
        buf_printf(&info, "  (%zu/%zu)", t->selected + 1, t->shown.len);
        th_fg(&l, T_MUTED, buf_str(&info));
        lines_addb(out, &l);
        buf_free(&info);
    }
    lines_addz(out, "");
    border_line(out, width, T_BORDER);
}

static void render_history(void);
static void tree_free_any(void *p);

static void open_tree(void) {
    const char *leaf = NULL;
    const JValue *entries = U->ops->entries ? U->ops->entries(U->host, &leaf) : NULL;
    if (!entries || !json_is(entries, J_ARRAY) || entries->arr.len <= 1) {
        show_status("No entries in session");
        return;
    }
    close_selector();
    TreeSel *t = xcalloc(1, sizeof *t);
    t->entries = jcopy(&t->a, entries);
    t->leaf = leaf ? arena_strdup(&t->a, leaf) : NULL;
    t->max_visible = MAX(5, U->scr.rows / 2);
    // the active path
    for (const char *id = t->leaf; id;) {
        const JValue *e = entry_by_id(t, str(id));
        if (!e) break;
        strmap_put(&t->active, json_strs(json_get(e, "id")), (void *)e);
        const JValue *p = json_get(e, "parentId");
        id = json_is(p, J_STRING) ? p->s.ptr : NULL;
    }
    tree_flatten(t);
    tree_filter(t, false);
    U->sel = SEL_TREE;
    U->sel_data = t;
    U->sel_free = tree_free_any;
}

static void tree_free(TreeSel *t);
static void tree_free_any(void *p) { tree_free(p); }
static void tree_free(TreeSel *t) {
    vec_free(&t->flat);
    vec_free(&t->shown);
    strmap_free(&t->active);
    buf_free(&t->query);
    arena_reset(&t->a);
    free(t);
}

static void tree_key(const Key *k) {
    TreeSel *t = U->sel_data;
    size_t n = t->shown.len;
    if (k->kind == K_UP && n) t->selected = t->selected ? t->selected - 1 : n - 1;
    else if (k->kind == K_DOWN && n) t->selected = t->selected + 1 >= n ? 0 : t->selected + 1;
    else if ((k->kind == K_LEFT || k->kind == K_PGUP) && n) t->selected = t->selected > (size_t)t->max_visible ? t->selected - (size_t)t->max_visible : 0;
    else if ((k->kind == K_RIGHT || k->kind == K_PGDN) && n) t->selected = MIN(n - 1, t->selected + (size_t)t->max_visible);
    else if (k->kind == K_ESC || (k->kind == K_CTRL && k->ch == 'c')) {
        if (t->query.len) {
            t->query.len = 0;
            tree_filter(t, true);
        } else close_selector();
    } else if (k->kind == K_ENTER && n) {
        const JValue *e = t->flat.ptr[t->shown.ptr[t->selected]].entry;
        const char *id = json_str(json_get(e, "id"));
        if (t->leaf && id && !strcmp(id, t->leaf)) {
            close_selector();
            show_status("Already at this point");
            return;
        }
        // navigateTree without a summary: a user message goes back to the editor and the leaf moves
        // to its parent; any other entry becomes the leaf
        const JValue *m = json_get(e, "message");
        bool user = str_eqz(json_strs(json_get(e, "type")), "message") && str_eqz(json_strs(json_get(m, "role")), "user");
        Buf text = {0};
        if (user) {
            const JValue *c = json_get(m, "content");
            if (json_is(c, J_STRING)) buf_addstr(&text, c->s);
            else
                for (size_t i = 0; c && json_is(c, J_ARRAY) && i < c->arr.len; i++)
                    if (str_eqz(json_strs(json_get(c->arr.items[i], "type")), "text")) buf_addstr(&text, json_strs(json_get(c->arr.items[i], "text")));
        }
        const JValue *p = json_get(e, "parentId");
        char *target = user ? (json_is(p, J_STRING) ? strndup(p->s.ptr, p->s.len) : NULL) : strdup(id);
        close_selector();
        if (U->ops->set_leaf) U->ops->set_leaf(U->host, target);
        free(target);
        render_history();
        code_trust_warning(); // code_trust.c
        if (user) editor_set_text(U->ed, buf_str(&text));
        buf_free(&text);
    } else if (k->kind == K_BACKSPACE && t->query.len) {
        size_t p = t->query.len - 1;
        while (p > 0 && ((unsigned char)t->query.ptr[p] & 0xc0) == 0x80) p--;
        t->query.len = p;
        t->query.ptr[p] = 0;
        tree_filter(t, true);
    } else if (k->kind == K_CHAR && k->ch >= 0x20) {
        char enc[4];
        buf_add(&t->query, enc, (size_t)tui_utf8(enc, k->ch));
        tree_filter(t, true);
    }
}

// ---- the transcript of a session (renderSessionItems) ------------------------------------------------------

static void render_history(void) {
    comp_clear(U->chat);
    U->stats.len = 0;
    U->last_status_text = U->last_status_spacer = NULL;
    U->pending_tools.len = 0;
    Arena ha = {0};
    const JValue *msgs = sess_ui_history(&ha);
    VEC(ToolView *) open = {0};
    for (size_t i = 0; msgs && json_is(msgs, J_ARRAY) && i < msgs->arr.len; i++) {
        const JValue *m = msgs->arr.items[i];
        account(m);
        Str role = json_strs(json_get(m, "role"));
        if (str_eqz(role, "user")) add_user(m);
        else if (str_eqz(role, "compactionSummary")) sess_ui_add_compaction(m);
        else if (str_eqz(role, "assistant")) {
            AsstView *v = asst_new(m);
            chat_add(&v->base);
            const JValue *c = json_get(m, "content");
            for (size_t k = 0; c && json_is(c, J_ARRAY) && k < c->arr.len; k++) {
                const JValue *b = c->arr.items[k];
                if (!str_eqz(json_strs(json_get(b, "type")), "toolCall")) continue;
                ToolView *t = tool_new(json_str(json_get(b, "name")), json_str(json_get(b, "id")), json_get(b, "arguments"));
                t->started = t->args_complete = true;
                chat_add(&t->base);
                vec_push(&open, t);
            }
            Str stop = json_strs(json_get(m, "stopReason"));
            if (str_eqz(stop, "aborted") || str_eqz(stop, "error")) {
                const char *em = json_str(json_get(m, "errorMessage"));
                Arena ra = {0};
                JValue *res = json_new(&ra, J_OBJECT), *content = json_new(&ra, J_ARRAY), *blk = json_new(&ra, J_OBJECT);
                json_put(&ra, blk, S("type"), json_string(&ra, S("text")));
                json_put(&ra, blk, S("text"), json_string(&ra, str(em ? em : "Error")));
                json_push(&ra, content, blk);
                json_put(&ra, res, S("content"), content);
                for (size_t k = 0; k < open.len; k++) tool_set_result(open.ptr[k], res, true, false, NULL);
                open.len = 0;
                arena_reset(&ra);
            }
        } else if (str_eqz(role, "toolResult")) {
            Str id = json_strs(json_get(m, "toolCallId"));
            for (size_t k = 0; k < open.len; k++)
                if (str_eqz(id, open.ptr[k]->id)) {
                    tool_set_result(open.ptr[k], m, jtrue(json_get(m, "isError")), false, json_get(m, "durationMs"));
                    memmove(open.ptr + k, open.ptr + k + 1, (open.len - k - 1) * sizeof(ToolView *));
                    open.len--;
                    break;
                }
        }
    }
    vec_free(&open);
    arena_reset(&ha);
    sess_ui_history_done();
}

// ---- layout and drawing --------------------------------------------------------------------------------

static void (*g_ext_render)(int width, Lines *out); // SEL_EXT: a selector kept in its own file
static void (*g_ext_key)(const Key *k);

static void render_dock(int width, Lines *out) {
    comp_render(U->pending, width, out);
    comp_render(U->status, width, out);
    lines_addz(out, ""); // the widgets above the editor: a spacer when there are none
    if (U->sel == SEL_THINKING) thinking_render(width, out);
    else if (U->sel == SEL_MODEL) model_render(width, out);
    else if (U->sel == SEL_SETTINGS) settings_render(width, out);
    else if (U->sel == SEL_TREE) tree_render(width, out);
    else if (U->sel == SEL_CONFIRM) confirm_render(width, out);
    else if (U->sel == SEL_THEME) theme_render(width, out);
    else if (U->sel == SEL_PANEL) g_panel.render(g_panel.ctx, width, out);
    else if (cmds_dock_active()) cmds_dock_render(width, out);
    else if (U->sel == SEL_EXT) g_ext_render(width, out);
    else if (code_mcp_ui_open()) code_mcp_ui_render(width, out); // /mcp (code_mcp_ui.c)
    else editor_render(U->ed, width, out);
    footer_render(NULL, width, out);
}

// ---- the transcript viewport (TuiAltScreen, ScrollView follow "end", layout.ts scrollbar) --------

#define PAGE_SCROLL_OVERLAP 4

typedef struct {
    int top;          // the first transcript row shown
    bool follow;      // pinned to the end
    int total, view;  // content and viewport heights of the last frame
    uint64_t bar_until; // the auto scrollbar shows until then (1 s after scrolling)
    // the wheel accelerator (wheel-scroll.ts)
    uint64_t wheel_last;
    int wheel_dir;
    double wheel_gap, wheel_carry;
    bool wheel_accel;
    // mouse selection in transcript rows (document coordinates)
    bool sel_on, sel_drag, sel_press;
    int sel_r0, sel_c0, sel_r1, sel_c1;
    int jump_row, jump_col, jump_w; // the jump-to-latest label of the last frame (jump_w 0: none)
} View;
static View g_view = {.follow = true};

static void view_scroll(int lines) {
    View *v = &g_view;
    int max_top = MAX(0, v->total - v->view);
    int start = v->follow ? max_top : v->top;
    int next = MAX(0, MIN(max_top, start + lines));
    if (next != start) v->bar_until = now_ns() + 1000000000ull;
    v->top = next;
    v->follow = next == max_top;
    request_render();
}

static void view_top(void) {
    g_view.top = 0;
    g_view.follow = g_view.total <= g_view.view;
    g_view.bar_until = now_ns() + 1000000000ull;
    request_render();
}

static void view_end(void) {
    g_view.follow = true;
    g_view.bar_until = now_ns() + 1000000000ull;
    request_render();
}

// The components whose first row pi marks with OSC 133;A: a user message, and an assistant
// message without tool calls (user-message.ts, assistant-message.ts render).
static bool prompt_zone(const Comp *c) {
    if (comp_markdown_is_user(c)) return true;
    if (c->vt != &asst_vt) return false;
    const JValue *content = json_get(((const AsstView *)c)->msg, "content");
    for (size_t i = 0; json_is(content, J_ARRAY) && i < content->arr.len; i++)
        if (str_eqz(json_strs(json_get(content->arr.items[i], "type")), "toolCall")) return false;
    return true;
}

// scrollToPrompt (tui-alt-screen.ts): the view's top to the previous (dir -1) or next prompt
// zone's first row; nothing when there is none.
static void view_prompt(int dir) {
    View *v = &g_view;
    int w = U->scr.cols;
    int row = (int)comp_lines(U->header, w)->n;
    VEC(int) starts = {0};
    CompVec *kids = comp_children(U->chat);
    for (size_t k = 0; k < kids->len; k++) {
        const Lines *l = comp_lines(kids->ptr[k], w);
        if (l->n && prompt_zone(kids->ptr[k])) vec_push(&starts, row);
        row += (int)l->n;
    }
    int max_top = MAX(0, v->total - v->view);
    int top = v->follow ? max_top : v->top;
    for (int r = top + dir; r >= 0 && r < row; r += dir) {
        bool start = false;
        for (size_t i = 0; i < starts.len && !start; i++) start = starts.ptr[i] == r;
        if (!start) continue;
        // ScrollView.scrollTo: clamped; the end follows again when it lands there
        int next = MAX(0, MIN(max_top, r));
        if (next != top) v->bar_until = now_ns() + 1000000000ull;
        v->top = next;
        v->follow = next == max_top;
        request_render();
        break;
    }
    vec_free(&starts);
}

// WheelScrollAccelerator("auto"): one line per event where the terminal accelerates (macOS, not
// over SSH), else faster spins move up to six.
static int wheel_lines(int dir) {
    View *v = &g_view;
    uint64_t now = now_ns();
    if (!v->wheel_accel) return 1;
    double gap = v->wheel_last ? (double)(now - v->wheel_last) / 1e6 : 1e9;
    bool same = dir == v->wheel_dir && gap <= 200;
    v->wheel_last = now;
    v->wheel_dir = dir;
    if (!same) {
        v->wheel_gap = 0;
        v->wheel_carry = 0;
        return 1;
    }
    if (gap < 5) return 1;
    v->wheel_gap = v->wheel_gap == 0 ? gap : (v->wheel_gap + gap) / 2;
    double lines = MIN(6.0, MAX(1.0, 100.0 / v->wheel_gap)) + v->wheel_carry;
    int whole = (int)floor(lines);
    v->wheel_carry = lines - whole;
    return whole;
}

// The plain text of a styled row's columns [c0, c1).
static void plain_cols(Buf *out, const char *line, int c0, int c1) {
    Buf cut = {0}, plain = {0};
    tui_slice(&cut, str(line), c0, MAX(0, c1 - c0), true);
    tui_strip(&plain, buf_str(&cut));
    buf_addstr(out, buf_str(&plain));
    buf_free(&cut);
    buf_free(&plain);
}

static void sel_bounds(int *r0, int *c0, int *r1, int *c1) {
    View *v = &g_view;
    bool fwd = v->sel_r0 < v->sel_r1 || (v->sel_r0 == v->sel_r1 && v->sel_c0 <= v->sel_c1);
    *r0 = fwd ? v->sel_r0 : v->sel_r1;
    *c0 = fwd ? v->sel_c0 : v->sel_c1;
    *r1 = fwd ? v->sel_r1 : v->sel_r0;
    *c1 = fwd ? v->sel_c1 : v->sel_c0;
}

// The transcript (header and chat) rows [from, to) into out; returns the total height. The
// components keep their lines, so a frame only copies the rows it shows.
static int doc_rows(int width, int from, int to, Lines *out) {
    int row = 0;
    const Lines *h = comp_lines(U->header, width);
    for (size_t i = 0; i < h->n; i++, row++)
        if (row >= from && row < to) lines_addz(out, h->v[i]);
    CompVec *kids = comp_children(U->chat);
    for (size_t k = 0; k < kids->len; k++) {
        const Lines *l = comp_lines(kids->ptr[k], width);
        if (row + (int)l->n <= from || row >= to) {
            row += (int)l->n;
            continue;
        }
        for (size_t i = 0; i < l->n; i++, row++)
            if (row >= from && row < to) lines_addz(out, l->v[i]);
    }
    return row;
}

// settings fullscreenCopyOnSelect (default true): a selection is copied when the mouse is released;
// off, Ctrl+X copies it (tui-alt-screen.ts getCopyOnSelect)
static bool copy_on_select(void) {
    const JValue *v = json_get(U->cfg->settings, "fullscreenCopyOnSelect");
    return json_is(v, J_BOOL) ? v->b : true;
}

static void copy_selection(int width) {
    int r0, c0, r1, c1;
    sel_bounds(&r0, &c0, &r1, &c1);
    Lines rows = {0};
    doc_rows(width, r0, r1 + 1, &rows);
    Buf text = {0};
    for (size_t i = 0; i < rows.n; i++) {
        int r = r0 + (int)i;
        Buf l = {0};
        plain_cols(&l, rows.v[i], r == r0 ? c0 : 0, r == r1 ? c1 : width);
        while (l.len && l.ptr[l.len - 1] == ' ') l.len--;
        if (i) buf_addc(&text, '\n');
        buf_add(&text, l.ptr ? l.ptr : "", l.len);
        buf_free(&l);
    }
    lines_free(&rows);
    if (text.len) {
        // pbcopy, else an OSC 52 write
        static const char *const pb[] = {"pbcopy", NULL};
        bool ok = tui_run((char *const *)pb, NULL, buf_str(&text), NULL, 2000);
        if (ok) uik_flash("Copied!", 1000); // copyTextToClipboard
        if (!ok) {
            static const char b64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            Buf o = {0};
            buf_adds(&o, "\x1b]52;c;");
            for (size_t i = 0; i < text.len; i += 3) {
                uint32_t v = (uint32_t)(uint8_t)text.ptr[i] << 16;
                if (i + 1 < text.len) v |= (uint32_t)(uint8_t)text.ptr[i + 1] << 8;
                if (i + 2 < text.len) v |= (uint8_t)text.ptr[i + 2];
                buf_addc(&o, b64[v >> 18 & 63]);
                buf_addc(&o, b64[v >> 12 & 63]);
                buf_addc(&o, i + 1 < text.len ? b64[v >> 6 & 63] : '=');
                buf_addc(&o, i + 2 < text.len ? b64[v & 63] : '=');
            }
            buf_adds(&o, "\x07");
            (void)!write(1, o.ptr, o.len);
            buf_free(&o);
        }
    }
    buf_free(&text);
}

// A styled row with columns [c0, c1) replaced by repl (layout.ts compositeTuiLine).
static char *overlay(const char *line, int c0, int c1, Str repl, int width) {
    Buf b = {0}, x = {0};
    tui_slice(&x, str(line), 0, c0, true);
    buf_addstr(&b, buf_str(&x));
    tui_spaces(&b, c0 - tui_width(buf_str(&x)));
    buf_adds(&b, "\x1b[0m");
    buf_addstr(&b, repl);
    buf_adds(&b, "\x1b[0m");
    x.len = 0;
    tui_slice(&x, str(line), c1, MAX(0, width - c1), true);
    buf_addstr(&b, buf_str(&x));
    buf_free(&x);
    return b.ptr ? b.ptr : strdup("");
}

static void draw(bool force) {
    Screen *s = &U->scr;
    int w = s->cols, h = s->rows;
    View *v = &g_view;
    if (code_monitor_is_open()) { // /monitor: the whole screen
        Lines mf = {0};
        code_monitor_render(w, h, &mf);
        screen_draw(s, &mf, force);
        lines_free(&mf);
        return;
    }
    editor_set_rows(U->ed, h);
    Lines dock = {0}, frame = {0};
    render_dock(w, &dock);
    int th = MAX(1, h - (int)dock.n);
    // the heights first (cached), then only the rows on screen
    int total = doc_rows(w, 0, 0, &frame);
    v->total = total;
    v->view = th;
    int max_top = MAX(0, total - th);
    if (v->follow) v->top = max_top;
    v->top = MAX(0, MIN(v->top, max_top));
    if (v->top == max_top) v->follow = true;
    doc_rows(w, v->top, v->top + th, &frame);
    while (frame.n < (size_t)th) lines_addz(&frame, "");
    // the selection, in inverse
    if (v->sel_on) {
        int r0, c0, r1, c1;
        sel_bounds(&r0, &c0, &r1, &c1);
        for (int r = MAX(r0, v->top); r <= r1 && r < v->top + th; r++) {
            int a = r == r0 ? c0 : 0, b = r == r1 ? c1 : w;
            if (b <= a) continue;
            char **line = &frame.v[r - v->top];
            Buf mid = {0}, inv = {0};
            plain_cols(&mid, *line, a, b);
            buf_adds(&inv, "\x1b[7m");
            buf_addstr(&inv, buf_str(&mid));
            tui_spaces(&inv, (b - a) - tui_width(buf_str(&mid)));
            buf_adds(&inv, "\x1b[27m");
            char *nl = overlay(*line, a, b, buf_str(&inv), w);
            free(*line);
            *line = nl;
            buf_free(&mid);
            buf_free(&inv);
        }
    }
    // the scrollbar ("auto": a moment after scrolling, when the content overflows)
    uint64_t now = now_ns();
    if (total > th && now < v->bar_until && w > 1) {
        int thumb_h = MAX(MIN(2, th), MIN(th, (int)lround((double)th * th / total)));
        int thumb_top = max_top ? (int)lround((double)v->top / max_top * (th - thumb_h)) : 0;
        for (int r = 0; r < th; r++) {
            bool thumb = r >= thumb_top && r < thumb_top + thumb_h;
            Buf g = {0};
            th_fgz(&g, thumb ? T_SCROLL_THUMB : T_SCROLL_TRACK, thumb ? "\xe2\x94\x83" : "\xe2\x94\x82");
            char *nl = overlay(frame.v[r], w - 1, w, buf_str(&g), w);
            free(frame.v[r]);
            frame.v[r] = nl;
            buf_free(&g);
        }
    }
    // scrolled away from the end: the jump-to-latest label on the last transcript row
    v->jump_w = 0;
    if (!v->follow) {
        Buf l = {0}, t = {0};
        th_fgz(&l, T_TEXT, " \xe2\x86\x93 Jump to latest message \xc2\xb7 Ctrl+End ");
        tui_truncate(&t, buf_str(&l), w, "", false);
        int lw = tui_width(buf_str(&t));
        int col = (w - lw) / 2;
        if (lw > 0) {
            char *nl = overlay(frame.v[th - 1], col, col + lw, buf_str(&t), w);
            free(frame.v[th - 1]);
            frame.v[th - 1] = nl;
            v->jump_row = th - 1;
            v->jump_col = col;
            v->jump_w = lw;
        }
        buf_free(&l);
        buf_free(&t);
    }
    for (size_t i = 0; i < dock.n && frame.n < (size_t)h; i++) {
        lines_push(&frame, dock.v[i]);
        dock.v[i] = NULL;
    }
    // compositeFlashes: the flash messages at the top right (code_ui_keys.c)
    Lines fl = {0};
    uik_flash_render(w, &fl);
    for (size_t i = fl.n > (size_t)h ? fl.n - (size_t)h : 0, row = 0; i < fl.n && row < frame.n; i++, row++) {
        int fw = tui_width(str(fl.v[i]));
        if (!fw) continue;
        char *nl = overlay(frame.v[row], w - fw, w, str(fl.v[i]), w);
        free(frame.v[row]);
        frame.v[row] = nl;
    }
    lines_free(&fl);
    screen_draw(s, &frame, force);
    lines_free(&dock);
    lines_free(&frame);
}

// A mouse event: the wheel scrolls the transcript, a drag selects its text (copied on release),
// a click on the jump label goes to the end.
static void handle_mouse(const Key *k) {
    View *v = &g_view;
    int b = k->button;
    if (b & 64) { // wheel: 64 up, 65 down; bit 3 is Alt (five times as far)
        int dir = (b & 3) == 0 ? -1 : (b & 3) == 1 ? 1 : 0;
        if (!dir) return;
        int n = wheel_lines(dir);
        view_scroll(dir * ((b & 8) ? n * 5 : n));
        return;
    }
    bool motion = b & 32;
    int btn = b & 3;
    if (k->my >= v->view) { // the dock: no selection there
        if (!motion && !k->release && v->sel_on) {
            v->sel_on = false;
            request_render();
        }
        return;
    }
    int row = v->top + k->my, col = MAX(0, MIN(U->scr.cols, k->mx));
    if (!motion && !k->release && btn == 0) {
        if (v->jump_w && k->my == v->jump_row && k->mx >= v->jump_col && k->mx < v->jump_col + v->jump_w) {
            view_end();
            return;
        }
        v->sel_press = true;
        v->sel_drag = false;
        v->sel_on = false;
        v->sel_r0 = v->sel_r1 = row;
        v->sel_c0 = v->sel_c1 = col;
        request_render();
    } else if (motion && btn == 0 && v->sel_press) {
        v->sel_r1 = row;
        v->sel_c1 = col + 1;
        v->sel_drag = v->sel_on = true;
        request_render();
    } else if (k->release && v->sel_press) {
        v->sel_press = false;
        if (v->sel_drag && copy_on_select()) copy_selection(U->scr.cols);
    }
}

static bool g_dirty;
static void request_render(void) { g_dirty = true; }

// ---- the agent thread -------------------------------------------------------------------------------------

// The thread's handoff lives outside the UI's state, so a thread still running after the UI
// returns (see code_ui_run) touches nothing that was freed.
static pthread_mutex_t g_tlock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_tcond = PTHREAD_COND_INITIALIZER;
static char *g_job;
static bool g_tquit;
static const CodeUiOps *g_tops;
static CodeUiHost *g_thost;

static void *agent_main(void *arg) {
    (void)arg;
    const CodeUiOps *ops = g_tops;
    CodeUiHost *host = g_thost;
    pthread_mutex_lock(&g_tlock);
    for (;;) {
        while (!g_job && !g_tquit) pthread_cond_wait(&g_tcond, &g_tlock);
        if (g_tquit) break;
        char *job = g_job;
        g_job = NULL;
        pthread_mutex_unlock(&g_tlock);
        if (job[0] == '\x01' && ops->compact) ops->compact(host, job + 1); // /compact (uix_start_compact)
        else ops->prompt(host, job);
        free(job);
        q_post("{\"type\":\"ui_idle\"}");
        pthread_mutex_lock(&g_tlock);
    }
    pthread_mutex_unlock(&g_tlock);
    return NULL;
}

static void start_prompt(const char *text) {
    U->busy = true;
    g_turn_start_ns = now_ns();
    pthread_mutex_lock(&g_tlock);
    free(g_job);
    g_job = strdup(text);
    pthread_cond_signal(&g_tcond);
    pthread_mutex_unlock(&g_tlock);
}

// ---- the streamed message, rebuilt from the ring's deltas ------------------------------------------------

typedef struct {
    char type; // 't' text, 'h' thinking, 'c' tool call
    Buf text;
    char *id, *name;
    CodeArgsParser *args;
} SBlock;

static struct {
    bool on, dirty, tools_dirty;
    Arena base_a;
    JValue *base; // the message as message_start gave it
    VEC(SBlock) blocks;
} g_stream;

static void stream_reset(void) {
    for (size_t i = 0; i < g_stream.blocks.len; i++) {
        SBlock *b = &g_stream.blocks.ptr[i];
        buf_free(&b->text);
        free(b->id);
        free(b->name);
        code_args_parser_free(b->args);
    }
    vec_free(&g_stream.blocks);
    arena_reset(&g_stream.base_a);
    g_stream.base = NULL;
    g_stream.on = g_stream.dirty = g_stream.tools_dirty = false;
}

static SBlock *stream_block(int index, char type) {
    if (index < 0 || index > 100000) return NULL;
    while (g_stream.blocks.len <= (size_t)index) {
        SBlock nb = {.type = type};
        vec_push(&g_stream.blocks, nb);
    }
    SBlock *b = &g_stream.blocks.ptr[index];
    if (!b->type) b->type = type;
    return b;
}

static void stream_begin(const JValue *m) {
    stream_reset();
    g_stream.on = true;
    g_stream.base = jcopy(&g_stream.base_a, m);
    const JValue *c = json_get(m, "content");
    for (size_t i = 0; json_is(c, J_ARRAY) && i < c->arr.len; i++) {
        const JValue *bl = c->arr.items[i];
        Str t = json_strs(json_get(bl, "type"));
        SBlock *b = stream_block((int)i, str_eqz(t, "thinking") ? 'h' : str_eqz(t, "toolCall") ? 'c' : 't');
        if (!b) continue;
        if (b->type == 'c') {
            b->id = strndup(json_strs(json_get(bl, "id")).ptr, json_strs(json_get(bl, "id")).len);
            b->name = strndup(json_strs(json_get(bl, "name")).ptr, json_strs(json_get(bl, "name")).len);
        } else buf_addstr(&b->text, json_strs(json_get(bl, b->type == 'h' ? "thinking" : "text")));
    }
}

static void stream_rec(CodeUiStreamKind k, int index, Str a, Str b) {
    if (!g_stream.on) return;
    SBlock *bl = NULL;
    switch (k) {
    case UI_S_TEXT_START: bl = stream_block(index, 't'); break;
    case UI_S_THINK_START: bl = stream_block(index, 'h'); break;
    case UI_S_TEXT:
    case UI_S_THINK:
        bl = stream_block(index, k == UI_S_TEXT ? 't' : 'h');
        if (bl) buf_addstr(&bl->text, a);
        break;
    case UI_S_TOOL_START:
        bl = stream_block(index, 'c');
        if (bl) {
            free(bl->id);
            free(bl->name);
            bl->id = strndup(a.ptr, a.len);
            bl->name = strndup(b.ptr, b.len);
        }
        g_stream.tools_dirty = true;
        break;
    case UI_S_TOOL_ARGS:
        bl = stream_block(index, 'c');
        if (bl) {
            if (!bl->args) bl->args = code_args_parser_new();
            code_args_parser_feed(bl->args, a);
        }
        g_stream.tools_dirty = true;
        break;
    }
    g_stream.dirty = true;
}

static void sync_tools(const JValue *msg);

static bool stream_live(const AsstView *v) { return g_stream.on && U && v == U->streaming; }

// The streamed message as asst_render draws a message (text blocks, thinking blocks joined), read
// from the blocks as they grow: no copy of the message per update.
static void stream_render(int width, Lines *out) {
    bool need_spacer = true;
    size_t n = g_stream.blocks.len;
    for (size_t i = 0; i < n; i++) {
        SBlock *b = &g_stream.blocks.ptr[i];
        if (b->type == 't') {
            Str t = str_trim(buf_str(&b->text));
            if (!t.len) continue;
            if (need_spacer) lines_addz(out, "");
            need_spacer = false;
            md_render(t, 1, 0, &md_plain, width, out);
        } else if (b->type == 'h') {
            Buf joined = {0};
            for (; i < n && g_stream.blocks.ptr[i].type == 'h'; i++) {
                Str t = str_trim(buf_str(&g_stream.blocks.ptr[i].text));
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
}

// After a batch of deltas: the view drawn again; the tool views get their calls' arguments (a
// message of the tool-call blocks only, built when a call changed).
static void stream_flush(void) {
    if (!g_stream.on || !g_stream.dirty || !U->streaming) return;
    g_stream.dirty = false;
    comp_invalidate(&U->streaming->base);
    if (!g_stream.tools_dirty) return;
    g_stream.tools_dirty = false;
    Arena a = {0};
    JValue *m = json_new(&a, J_OBJECT);
    JValue *content = json_new(&a, J_ARRAY);
    for (size_t i = 0; i < g_stream.blocks.len; i++) {
        SBlock *b = &g_stream.blocks.ptr[i];
        if (b->type != 'c') continue;
        JValue *o = json_new(&a, J_OBJECT);
        json_put(&a, o, S("type"), json_string(&a, S("toolCall")));
        json_put(&a, o, S("id"), json_string(&a, str(b->id ? b->id : "")));
        json_put(&a, o, S("name"), json_string(&a, str(b->name ? b->name : "")));
        json_put(&a, o, S("arguments"), b->args ? (JValue *)code_args_parser_value(b->args) : json_new(&a, J_OBJECT));
        json_push(&a, content, o);
    }
    json_put(&a, m, S("content"), content);
    sync_tools(m);
    arena_reset(&a);
}

static void handle_event(const JValue *ev);

// Everything the agent thread produced so far, in order.
static void ring_drain(bool handle) {
    static Buf ra, rb;
    for (;;) {
        uint64_t rt = atomic_load_explicit(&g_rtail, memory_order_relaxed);
        uint64_t rh = atomic_load_explicit(&g_rhead, memory_order_acquire);
        if (rt == rh) break;
        Rec r = g_recs[rt % RING_RECS];
        if (r.kind == R_JSON) {
            if (handle) {
                stream_flush();
                Arena a = {0};
                JValue *ev = NULL;
                if (!json_parse(&a, str(r.json), true, &ev) && ev) handle_event(ev);
                arena_reset(&a);
            } else if (strstr(r.json, "\"ui_idle\"") && U) U->busy = false;
            free(r.json);
        } else if (handle) {
            ra.len = rb.len = 0;
            ring_copy_out(r.off, r.len, &ra);
            ring_copy_out(r.off + r.len, r.len2, &rb);
            stream_rec((CodeUiStreamKind)r.skind, r.index, buf_str(&ra), buf_str(&rb));
        }
        atomic_store_explicit(&g_btail, r.off + r.len + r.len2, memory_order_release);
        atomic_store_explicit(&g_rtail, rt + 1, memory_order_release);
    }
    if (handle) stream_flush();
}

// ---- events (handleEvent) -------------------------------------------------------------------------------------


static void set_working(bool on) {
    if (on && !U->working) g_last_spin = now_ns(); // the spinner starts at its first frame
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
        Str rest;
        Comp *skill = code_skill_block_comp(buf_str(&t), &rest); // a /skill:name block (code_skills.c)
        if (skill) {
            chat_add(skill);
            if (rest.len) {
                chat_add(comp_spacer(1));
                chat_add(comp_markdown(rest, 1, 1, &md_user));
            }
        } else
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


static void pending_display(void);

static void handle_event(const JValue *ev) {
    sess_ui_event(ev);
    Str type = json_strs(json_get(ev, "type"));
    if (str_eqz(type, "ui_idle")) {
        U->busy = false;
        g_last_turn_s = g_turn_start_ns ? (double)(now_ns() - g_turn_start_ns) / 1e9 : -1;
        git_refresh();
        set_working(false);
        // what was chosen while the run went on and its last turn did not take
        char *rp, *rm;
        if (code_ui_take_model(&rp, &rm)) {
            if (U->ops->set_model && !U->ops->set_model(U->host, rp, rm)) show_error("Could not switch the model");
            free(rp);
            free(rm);
        }
        char *rt = code_ui_take_thinking();
        if (rt) {
            if (U->ops->set_thinking) U->ops->set_thinking(U->host, rt);
            free(rt);
        }
        if (sess_ui_idle()) return;
        if (U->re_id) {
            U->quit = true;
            return;
        }
        // the prompts from the command line, one after another (pi awaits each)
        if (U->ninitial_next < U->ninitial) {
            start_prompt(U->initial[U->ninitial_next++]);
            return;
        }
        char *next = code_ui_take_queued(false);
        if (next) {
            start_prompt(next);
            free(next);
        }
        return;
    }
    if (str_eqz(type, "ui_warning")) { // a notice of the agent's side (code_share.c's, for one)
        show_warning(json_str(json_get(ev, "message")) ? json_str(json_get(ev, "message")) : "");
        return;
    }
    if (str_eqz(type, "ui_error")) {
        show_error(json_str(json_get(ev, "message")) ? json_str(json_get(ev, "message")) : "error");
        return;
    }
    if (str_eqz(type, "queue_update")) {
        pending_display();
    } else if (str_eqz(type, "agent_start")) {
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
            stream_begin(m);
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
            } else
                for (size_t i = 0; i < U->pending_tools.len; i++) U->pending_tools.ptr[i]->args_complete = true;
            arena_reset(&tmp);
            U->streaming = NULL;
            stream_reset();
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
        Str tn = json_strs(json_get(ev, "toolName"));
        if (str_eqz(tn, "bash") || str_eqz(tn, "edit") || str_eqz(tn, "write")) git_refresh();
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
        stream_reset();
        U->pending_tools.len = 0;
    } else if (str_eqz(type, "auto_retry_start")) {
        // RetryStatusIndicator: the retry countdown in the editor's border
        Buf m = {0};
        buf_printf(&m, "Retrying (%g/%g) in %.0fs... (escape to cancel)", jnum(json_get(ev, "attempt")), jnum(json_get(ev, "maxAttempts")),
                   ceil(jnum(json_get(ev, "delayMs")) / 1000));
        editor_set_working(U->ed, m.ptr);
        if (!U->working) g_last_spin = now_ns();
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
    {"share", "Share session as a secret GitHub gist", NULL},
    {"copy", "Copy last agent message to clipboard", NULL},
    {"name", "Set session display name", NULL},
    {"session", "Show session info and stats", NULL},
    {"changelog", "Show changelog entries", NULL},
    {"hotkeys", "Show all keyboard shortcuts", NULL},
    {"fork", "Create a new fork from a previous user message", NULL},
    {"clone", "Duplicate the current session at the current position", NULL},
    {"trust", "Save project trust decision for future sessions", NULL},
    {"new", "Start a new session", NULL},
    {"compact", "Manually compact the session context", NULL},
    {"resume", "Resume another session", NULL},
    {"reload", "Reload keybindings, extensions, skills, prompts, themes, and context files", NULL},
    {"quit", "Quit Saetta Code", NULL},
    {"mcp", "Manage MCP servers: sign in, reconnect, enable or disable, and change exposure", NULL},
    {"theme", "Select color theme", NULL},
    {"monitor", "Show the model's live statistics (saetta monitor)", NULL},
};

static bool arg_items(void *ctx, const char *cmd, SelectList *all, Lines *search) {
    (void)ctx;
    if (!strcmp(cmd, "mcp")) return code_mcp_ui_arg_items(all, search); // code_mcp_ui.c
    if (!strcmp(cmd, "thinking")) {
        for (size_t i = 0; i < available_levels(); i++) {
            sel_add(all, level_names[i], level_names[i], NULL);
            lines_addz(search, level_names[i]);
        }
        return true;
    }
    if (!strcmp(cmd, "model")) {
        for (size_t i = 0; i < U->cfg->nmodels; i++) {
            const CodeUiModel *m = &U->cfg->models[i];
            Buf v = {0}, h = {0};
            buf_printf(&v, "%s/%s", m->provider, m->id);
            buf_printf(&h, "%s %s %s/%s %s %s%s%s", m->id, m->provider, m->provider, m->id, m->provider, m->id, m->name ? " " : "",
                       m->name ? m->name : "");
            sel_add(all, v.ptr, m->id, m->provider);
            lines_addb(search, &h);
            buf_free(&v);
        }
        return true;
    }
    return false;
}

void code_ui_refresh_commands(void) {
    size_t n;
    const SlashCmd *cmds = code_skill_commands(builtin_cmds, COUNT(builtin_cmds), &n); // with skill:<name> (code_skills.c)
    editor_set_commands(U->ed, cmds, n);
}

void code_ui_rebuild_chat(void) { render_history(); }

static void new_session(void) {
    if (U->busy) return;
    if (U->ops->new_session) U->ops->new_session(U->host);
    sess_ui_reset();
    comp_clear(U->chat);
    U->stats.len = 0;
    U->bug_hint_shown = false;
    U->last_status_text = U->last_status_spacer = NULL;
    chat_add(comp_spacer(1));
    chat_add(themed(T_ACCENT, "\xe2\x9c\x93 New session started", 1, 1));
}

// MCP (code_mcp.c): its notices as pi's ctx.ui.notify shows them, and the wake-up when they or the
// /mcp manager's state change
static void mcp_notice(int kind, const char *msg) {
    if (kind == 2) show_error(msg);
    else if (kind == 1) show_warning(msg);
    else show_status(msg);
}

static void mcp_wake(void) {
    if (g_wake[1] >= 0) (void)!write(g_wake[1], "x", 1);
}

static bool g_follow_up; // the text being submitted is a follow-up (Alt+Enter), not a steering message

// updatePendingMessagesDisplay: the queued messages above the editor.
static void pending_display(void) {
    comp_clear(U->pending);
    pthread_mutex_lock(&g_mlock);
    if (g_msgq.len) {
        comp_add(U->pending, comp_spacer(1));
        for (int pass = 0; pass < 2; pass++)
            for (size_t i = 0; i < g_msgq.len; i++) {
                if (g_msgq.ptr[i].steer != (pass == 0)) continue;
                Buf m = {0}, l = {0};
                buf_printf(&m, "%s: %s", pass == 0 ? "Steering" : "Follow-up", g_msgq.ptr[i].text);
                th_fg(&l, T_DIM, buf_str(&m));
                comp_add(U->pending, comp_truncated(buf_str(&l), 1));
                buf_free(&m);
                buf_free(&l);
            }
        Buf h = {0};
        th_fgz(&h, T_DIM, "\xe2\x86\xb3 alt+up to edit all queued messages");
        comp_add(U->pending, comp_truncated(buf_str(&h), 1));
        buf_free(&h);
    }
    pthread_mutex_unlock(&g_mlock);
}

// restoreQueuedMessagesToEditor: every queued message back in the editor, before what it holds.
static size_t restore_queued(void) {
    pthread_mutex_lock(&g_mlock);
    Buf all = {0};
    size_t n = g_msgq.len;
    for (int pass = 0; pass < 2; pass++)
        for (size_t i = 0; i < g_msgq.len; i++) {
            if (g_msgq.ptr[i].steer != (pass == 0)) continue;
            if (all.len) buf_adds(&all, "\n\n");
            buf_adds(&all, g_msgq.ptr[i].text);
            free(g_msgq.ptr[i].text);
        }
    g_msgq.len = 0;
    pthread_mutex_unlock(&g_mlock);
    if (n) {
        Str cur = editor_text(U->ed);
        if (str_trim(cur).len) {
            if (all.len) buf_adds(&all, "\n\n");
            buf_addstr(&all, cur);
        }
        editor_set_text(U->ed, buf_str(&all));
    }
    buf_free(&all);
    pending_display();
    return n;
}

static void submit(char *text) {
    Str t = str_trim(str(text));
    if (!t.len) return;
    char *s = strndup(t.ptr, t.len);
    if (s[0] == '/') {
        const char *c = s + 1;
        if (cmds_submit(s)) {
            free(s);
            return;
        }
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
        if (!strcmp(c, "theme")) {
            open_theme(false);
            free(s);
            return;
        }
        if (!strcmp(c, "settings")) {
            open_settings();
            free(s);
            return;
        }
        if (!strcmp(c, "tree")) {
            if (!U->busy) open_tree();
            free(s);
            return;
        }
        if (!strcmp(c, "model") || !strncmp(c, "model ", 6)) {
            const char *arg = !strncmp(c, "model ", 6) ? c + 6 : NULL;
            while (arg && *arg == ' ') arg++;
            if (arg && *arg) {
                // an exact provider/id or id
                const CodeUiModel *hit = NULL;
                for (size_t i = 0; i < U->cfg->nmodels && !hit; i++) {
                    const CodeUiModel *m = &U->cfg->models[i];
                    char full[512];
                    snprintf(full, sizeof full, "%s/%s", m->provider, m->id);
                    if (!strcasecmp(arg, full) || !strcasecmp(arg, m->id)) hit = m;
                }
                if (hit) choose_model(hit, false);
                else open_model(arg);
            } else open_model(NULL);
            free(s);
            return;
        }
        if (!strcmp(c, "monitor")) {
            code_monitor_open();
            free(s);
            return;
        }
        if (!strcmp(c, "quit")) {
            U->quit = true;
            free(s);
            return;
        }
        if (!strcmp(c, "trust")) { // code_trust.c
            code_trust_command();
            free(s);
            return;
        }
        if (!strcmp(c, "changelog")) { // code_changelog.c
            code_changelog_command();
            free(s);
            return;
        }
        if (!strcmp(c, "reload")) { // code_skills.c
            code_reload_command();
            free(s);
            return;
        }
        if (sess_ui_command(c)) { // code_sessions_ui.c
            free(s);
            return;
        }
        if (!strcmp(c, "mcp") || !strncmp(c, "mcp ", 4)) {
            code_mcp_ui_command(c + 3); // code_mcp_ui.c
            free(s);
            return;
        }
    }
    editor_add_history(U->ed, str(s));
    if (U->busy) {
        pthread_mutex_lock(&g_mlock);
        vec_push(&g_msgq, ((Queued){strdup(s), !g_follow_up}));
        pthread_mutex_unlock(&g_mlock);
        pending_display();
    } else start_prompt(s);
    free(s);
}

static void toggle_tools(void) {
    U->tools_expanded = !U->tools_expanded;
    U->header_expanded = U->tools_expanded;
    CompVec *kids = comp_children(U->chat);
    for (size_t i = 0; i < kids->len; i++)
        if (kids->ptr[i]->vt == &tool_vt) {
            ((ToolView *)kids->ptr[i])->expanded = U->tools_expanded;
            comp_invalidate(kids->ptr[i]);
        }
    sess_ui_set_expanded(U->tools_expanded);
    show_status(U->tools_expanded ? "Tool output: expanded" : "Tool output: collapsed");
}

static void cycle_model(int dir);

static void cycle_thinking(void) {
    if (!U->model.reasoning) {
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
    if (U->sel == SEL_PANEL) {
        g_panel.key(g_panel.ctx, k);
        return;
    }
    if (U->sel == SEL_CONFIRM && U->confirm_switch) {
        if (k->kind == K_ENTER) {
            U->sel = SEL_NONE;
            U->confirm_switch = false;
            restart_model(U->sw_provider, U->sw_id);
        } else if (k->kind == K_ESC || (k->kind == K_CTRL && k->ch == 'c')) {
            U->sel = SEL_NONE;
            U->confirm_switch = false;
        }
        return;
    }
    if (U->sel == SEL_CONFIRM) {
        if (k->kind == K_ENTER && U->dl_free_gb >= U->dl_disk) {
            U->sel = SEL_NONE;
            start_download();
        } else if (k->kind == K_ESC || (k->kind == K_CTRL && k->ch == 'c')) U->sel = SEL_NONE;
        return;
    }
    if (U->sel == SEL_THEME) {
        theme_key(k);
        return;
    }
    if (U->sel == SEL_EXT) {
        g_ext_key(k);
        return;
    }
    if (U->sel == SEL_MODEL) model_key(k);
    else if (U->sel == SEL_SETTINGS) settings_key(k);
    else if (U->sel == SEL_TREE) tree_key(k);
    else if (U->sel == SEL_THINKING) {
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
    if (code_mcp_ui_open()) { // /mcp takes the keys (code_mcp_ui.c)
        code_mcp_ui_key(k);
        return;
    }
    if (cmds_dock_active()) {
        cmds_dock_key(k);
        return;
    }
    // app actions (custom-editor.ts)
    if (k->kind == K_ESC && !editor_autocomplete_open(U->ed)) {
        if (!U->busy && U->dl_pid > 0) {
            cancel_download();
            return;
        }
        if (U->busy) {
            // during a compaction escape cancels it only (compaction_start's onEscape:
            // abortCompaction); else restoreQueuedMessagesToEditor({abort: true})
            if (!sess_ui_compacting()) restore_queued();
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
    if (k->kind == K_CTRL && k->ch == 'l') {
        open_model(NULL);
        return;
    }
    if (k->kind == K_CTRL && k->ch == 't') {
        U->hide_thinking = !U->hide_thinking;
        invalidate_kind(&asst_vt);
        show_status(U->hide_thinking ? "Thinking blocks: hidden" : "Thinking blocks: visible");
        return;
    }
    if (k->kind == K_CTRL && k->ch == 'p') {
        cycle_model(k->shift ? -1 : 1);
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
            g_follow_up = true;
            submit(s);
            g_follow_up = false;
            free(s);
        }
        return;
    }
    if (k->kind == K_ALT_UP) { // app.message.dequeue
        size_t n = restore_queued();
        if (!n) show_status("No queued messages to restore");
        else {
            Buf m = {0};
            buf_printf(&m, "Restored %zu queued message%s to editor", n, n > 1 ? "s" : "");
            show_status(m.ptr);
            buf_free(&m);
        }
        return;
    }
    if (uik_key(k)) return; // app.clipboard.pasteImage, app.message.copy (code_ui_keys.c)
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

// app.editor.external (external-editor.ts): the text in $VISUAL / $EDITOR (settings
// "externalEditor" first, nano without either), the screen given back meanwhile.
static void external_editor(void) {
    const char *cmd = json_str(json_get(U->cfg->settings, "externalEditor"));
    if (!cmd || !*cmd) cmd = getenv("VISUAL");
    if (!cmd || !*cmd) cmd = getenv("EDITOR");
    if (!cmd || !*cmd) cmd = "nano";
    char dir[] = "/tmp/pi-editor-XXXXXX";
    if (!mkdtemp(dir)) {
        show_error("Failed to create a temporary file for the external editor");
        return;
    }
    char path[64];
    snprintf(path, sizeof path, "%s/prompt.md", dir);
    Str text = editor_text(U->ed);
    FILE *f = fopen(path, "w");
    if (f) {
        fwrite(text.ptr, 1, text.len, f);
        fclose(f);
    }
    screen_stop(&U->scr);
    printf("Launching external editor: %s\nSaetta Code will resume when the editor exits.\n", cmd);
    fflush(stdout);
    // the command split on spaces, the file last
    char *copy = strdup(cmd);
    const char *argv[64];
    int n = 0;
    for (char *tok = strtok(copy, " "); tok && n < 62; tok = strtok(NULL, " ")) argv[n++] = tok;
    argv[n++] = path;
    argv[n] = NULL;
    pid_t pid;
    int rc = tui_spawn(&pid, (char *const *)argv, NULL, true, false);
    int st = 0;
    if (rc == 0)
        while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
    screen_start(&U->scr);
    if (rc) {
        Buf m = {0};
        buf_printf(&m, "Failed to launch external editor: %s", cmd);
        show_status(m.ptr);
        buf_free(&m);
    } else if (!WIFEXITED(st) || WEXITSTATUS(st)) {
        Buf m = {0};
        if (WIFEXITED(st)) buf_printf(&m, "External editor exited with code %d; prompt unchanged", WEXITSTATUS(st));
        else buf_printf(&m, "External editor exited with %s; prompt unchanged", strsignal(WTERMSIG(st)));
        show_status(m.ptr);
        buf_free(&m);
    } else {
        Buf b = {0};
        f = fopen(path, "r");
        if (f) {
            char chunk[8192];
            size_t got;
            while ((got = fread(chunk, 1, sizeof chunk, f)) > 0) buf_add(&b, chunk, got);
            fclose(f);
        }
        Str t = buf_str(&b);
        if (str_starts(t, S("\xef\xbb\xbf"))) t = str_sub(t, 3, t.len);
        if (t.len && t.ptr[t.len - 1] == '\n') t.len--;
        editor_set_text(U->ed, t);
        buf_free(&b);
    }
    unlink(path);
    rmdir(dir);
    free(copy);
}

// app.model.cycleForward / Backward over the models /model lists that run now (downloaded).
static void cycle_model(int dir) {
    if (cmds_cycle_model(dir)) return; // the scoped models (/scoped-models)
    size_t n = U->cfg->nmodels, cur = n;
    size_t runnable = 0;
    // only the models this process switches to in place: the servers of models.json (packs, and
    // anything while the engine runs here, need a confirmed restart: /model)
#define CYCLABLE(m) (!(m)->pack && !U->cfg->engine)
    for (size_t i = 0; i < n; i++) {
        if (CYCLABLE(&U->cfg->models[i])) runnable++;
        if (is_current(&U->cfg->models[i])) cur = i;
    }
    if (runnable < 2) {
        show_status("Only one model available");
        return;
    }
    size_t i = cur == n ? 0 : cur;
    do i = (i + n + (size_t)(dir > 0 ? 1 : n - 1)) % n;
    while (!CYCLABLE(&U->cfg->models[i]));
#undef CYCLABLE
    const CodeUiModel *m = &U->cfg->models[i];
    choose_model(m, false);
    Buf b = {0};
    const char *lv = U->thinking ? U->thinking : "off";
    buf_printf(&b, "Switched to %s", m->name && *m->name ? m->name : m->id);
    if (m->reasoning && strcmp(lv, "off")) buf_printf(&b, " (thinking: %s)", lv);
    if (!m->pack) show_status(b.ptr);
    buf_free(&b);
}

// The keys the fullscreen viewport takes before the editor (tui.altScreen.*).
static bool viewport_key(const Key *k) {
    if ((U->sel == SEL_TREE || U->sel == SEL_EXT) && (k->kind == K_PGUP || k->kind == K_PGDN)) return false;
    View *v = &g_view;
    switch (k->kind) {
    case K_PGUP: view_scroll(-MAX(1, v->view - PAGE_SCROLL_OVERLAP)); return true;
    case K_PGDN: view_scroll(MAX(1, v->view - PAGE_SCROLL_OVERLAP)); return true;
    case K_CTRL_HOME: view_top(); return true;
    case K_CTRL_END: view_end(); return true;
    case K_CTRL_UP: view_prompt(-1); return true;  // tui.altScreen.previousPrompt
    case K_CTRL_DOWN: view_prompt(1); return true; // tui.altScreen.nextPrompt
    default: return false;
    }
}

// What code_cmds.c reads and calls here.
static bool cmds_busy(void) { return U->busy; }
static const CodeUiModel *cmds_model(void) { return &U->model; }
static void cmds_choose(const CodeUiModel *m) { choose_model(m, false); }
static const char *cmds_thinking(void) { return U->thinking; }
static const char *cmds_theme(void) { return U->theme ? U->theme : "system"; }
static const TermColors *cmds_colors(void) { return U->have_colors ? &U->colors : NULL; }
static int cmds_appearance(void) { return U->have_colors && U->colors.has_bg ? 0 : U->mac_appearance; }
static void cmds_set_cwd(const char *cwd) {
    // the caller's own config; the copy made here is freed when the next one replaces it
    static char *owned;
    char *prev = owned;
    owned = strdup(cwd);
    ((CodeUiConfig *)U->cfg)->cwd = owned;
    free(prev);
    git_refresh();
}

Err code_ui_run(const CodeUiConfig *cfg, const CodeUiOps *ops, CodeUiHost *host, const char *const *initial, size_t ninitial) {
    Ui ui = {0};
    U = &ui;
    g_ui_thread = pthread_self();
    g_view = (View){.follow = true};
    {
        bool local_mac = !getenv("SSH_CONNECTION") && !getenv("SSH_CLIENT") && !getenv("SSH_TTY");
        g_view.wheel_accel = !local_mac;
    }
    ui.cfg = cfg;
    ui.dl_fd = -1;
    ui.ops = ops;
    ui.host = host;
    ui.thinking = dupz(cfg->thinking ? cfg->thinking : "off");
    ui.model = (CodeUiModel){.provider = dupz(cfg->model.provider), .id = dupz(cfg->model.id), .name = dupz(cfg->model.name),
                             .label = dupz(cfg->model.label), .label_brief = dupz(cfg->model.label_brief),
                             .reasoning = cfg->model.reasoning, .context_window = cfg->model.context_window};
    ui.auto_compact = cfg->auto_compact;
    ui.default_provider = dupz(cfg->default_provider);
    ui.default_model = dupz(cfg->default_model);
    {
        const char *dt = json_str(json_get(cfg->settings, "defaultThinkingLevel"));
        free(ui.default_thinking);
        ui.default_thinking = dupz(dt ? dt : "medium");
        ui.hide_thinking = json_is(json_get(cfg->settings, "hideThinkingBlock"), J_BOOL) && json_get(cfg->settings, "hideThinkingBlock")->b;
    }
    ui.header = comp_fn(header_render, NULL);
    ui.chat = comp_container();
    ui.pending = comp_container();
    ui.status = comp_container();
    ui.ed = editor_new(cfg->cwd, cfg->fd_path);
    code_ui_refresh_commands();
    editor_set_arg_completer(ui.ed, arg_items, NULL);
    editor_set_border(ui.ed, th_thinking_color(ui.thinking));
    sel_init(&ui.list, 5, 32, 32);
    ring_init();
    if (pipe(g_wake) == 0) {
        fcntl(g_wake[0], F_SETFL, O_NONBLOCK);
        fcntl(g_wake[1], F_SETFL, O_NONBLOCK);
        fcntl(g_wake[0], F_SETFD, FD_CLOEXEC);
        fcntl(g_wake[1], F_SETFD, FD_CLOEXEC);
    }
    editor_set_waker(ui.ed, g_wake[1]);
    code_mcp_set_waker(mcp_wake);
    git_refresh();
    g_tops = ops;
    g_thost = host;
    g_tquit = false;
    pthread_create(&ui.thread, NULL, agent_main, NULL);
    struct sigaction sa = {0};
    sa.sa_handler = on_winch;
    sigaction(SIGWINCH, &sa, NULL);
    {
        const char *t = json_str(json_get(cfg->settings, "theme"));
        ui.theme = t ? strdup(t) : NULL;
        static const char *const dr[] = {"defaults", "read", "-g", "AppleInterfaceStyle", NULL};
        Buf v = {0};
        bool dark = tui_run((char *const *)dr, NULL, S(""), &v, 1000) && v.len >= 4 && !strncmp(v.ptr, "Dark", 4);
        buf_free(&v);
        ui.mac_appearance = dark ? 1 : 2;
    }
    // pi's system theme is grayscale until the terminal reports its colors; the others draw at once
    if (!ui.theme || !strcmp(ui.theme, "system")) theme_set(NULL, true);
    else theme_use(ui.theme, NULL, ui.mac_appearance);
    cmds_init(&(CmdsUi){.cfg = cfg, .ops = ops, .host = host, .show_status = show_status, .show_error = show_error,
                        .chat_add = chat_add, .request_render = request_render, .render_history = render_history,
                        .busy = cmds_busy, .model = cmds_model, .choose_model = cmds_choose, .thinking = cmds_thinking,
                        .set_thinking = set_thinking, .theme_name = cmds_theme, .term_colors = cmds_colors,
                        .appearance = cmds_appearance, .toggle_tools = toggle_tools, .set_cwd = cmds_set_cwd});
    screen_start(&ui.scr);
    if (cfg->resumed) render_history();
    code_startup_notices(cfg->resumed); // the changelog notice and the trust warning (code_changelog.c)
    code_update_check_start(g_wake[1]); // the version check, off the UI thread (code_update.c)
    draw(true);
    if (ninitial) {
        ui.initial = xcalloc(ninitial, sizeof *ui.initial);
        for (size_t i = 0; i < ninitial; i++) ui.initial[i] = strdup(initial[i]);
        ui.ninitial = ninitial;
        // the first now, the others as each run ends (pi runs them in turn, none as steering)
        while (ui.ninitial_next < ui.ninitial && !ui.busy) {
            char *t = ui.initial[ui.ninitial_next++];
            submit(t);
        }
    }
    KeyIn kin = {0};
    uint64_t last_draw = 0;
    bool first_asked = false;
    g_dirty = true;
    while (!ui.quit) {
        uint64_t now = now_ns();
        int timeout = ui.working ? 80 : 1000;
        int kt = keyin_timeout_ms(&kin, now);
        if (kt >= 0) timeout = MIN(timeout, kt);
        if (g_dirty) timeout = MIN(timeout, 8);
        if (editor_busy(ui.ed)) timeout = MIN(timeout, 10);
        if (g_view.bar_until > now) timeout = MIN(timeout, (int)((g_view.bar_until - now) / 1000000) + 1);
        int flash_ms = uik_flash_wait_ms(now);
        if (flash_ms >= 0) timeout = MIN(timeout, flash_ms);
        int sess_ms = sess_ui_tick(now);
        if (sess_ms >= 0) timeout = MIN(timeout, sess_ms);
        if (ui.scr.color_query) timeout = MIN(timeout, 10);
        if (code_monitor_wait_ms() >= 0) timeout = MIN(timeout, code_monitor_wait_ms());
        if (cmds_dock_active()) timeout = MIN(timeout, 80);
        // asleep only with the ring empty (the producer wakes a sleeping consumer once)
        atomic_store(&g_ui_sleeping, 1);
        if (atomic_load(&g_rhead) != atomic_load(&g_rtail)) timeout = 0;
        struct pollfd fds[3] = {{0, POLLIN, 0}, {g_wake[0], POLLIN, 0}, {ui.dl_fd, POLLIN, 0}};
        int r = poll(fds, ui.dl_fd >= 0 ? 3 : 2, timeout);
        atomic_store(&g_ui_sleeping, 0);
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
        ring_drain(true);
        if (g_queue_dirty) {
            g_queue_dirty = false;
            pending_display();
            request_render();
        }
        if (editor_tick(ui.ed)) request_render();
        if (code_monitor_tick()) request_render();
        // MCP notices and state (code_mcp.c), once the theme is known (the terminal's colors came)
        if (!ui.scr.color_query && code_mcp_ui_tick(mcp_notice)) request_render();
        if (atomic_exchange(&g_git_changed, false)) request_render();
        if (code_update_check_poll()) request_render();
        download_poll();
        if (cmds_tick()) request_render();
        // keys
        if (r > 0 && (fds[0].revents & (POLLIN | POLLHUP))) {
            char in[65536];
            ssize_t n = read(0, in, sizeof in);
            if (n == 0) break;
            if (n > 0) keyin_feed(&kin, in, (size_t)n);
        }
        Key k;
        while (!ui.quit && keyin_next(&kin, now_ns(), &k)) {
            if (k.kind == K_REPLY_KITTY || k.kind == K_REPLY_DA || k.kind == K_REPLY_COLOR) {
                bool late = !ui.scr.color_query && k.kind == K_REPLY_COLOR;
                if (late) ui.scr.color_query = true; // a reply after the timeout still applies (onLateReply)
                screen_reply(&ui.scr, &k);
                if (late) ui.scr.color_done = true;
            }
            else if (k.kind == K_MOUSE) { if (!code_monitor_is_open()) handle_mouse(&k); }
            else if (k.kind != K_NONE && code_monitor_key(&k)) request_render();
            else if (k.kind != K_NONE) {
                if (g_view.sel_on) {
                    g_view.sel_on = false;
                    request_render();
                }
                if (k.kind == K_CTRL && k.ch == 'z' && ui.sel == SEL_NONE) {
                    screen_suspend(&ui.scr);
                    screen_size(&ui.scr);
                    draw(true);
                } else if (k.kind == K_CTRL && k.ch == 'g' && ui.sel == SEL_NONE) {
                    external_editor();
                    draw(true);
                } else if (!viewport_key(&k)) handle_key(&k);
                request_render();
            }
            free(k.paste);
        }
        now = now_ns();
        // the terminal's colors: the theme once they all came, the DA1 after them, or 100 ms passed
        // (TERMINAL_QUERY_TIMEOUT_MS)
        if (ui.scr.color_query && (ui.scr.color_done || now - ui.scr.color_query_ns >= 100000000ull)) {
            ui.scr.color_query = ui.scr.color_done = false;
            ui.colors = ui.scr.colors;
            ui.have_colors = ui.colors.has_fg || ui.colors.has_bg || ui.colors.has_palette;
            const SelItem *it = ui.sel == SEL_THEME ? sel_current(&ui.list) : NULL;
            apply_theme(it ? it->value : ui.theme);
            // the first start: the theme choice, before the first prompt
            if (!ui.theme && ui.sel == SEL_NONE && !first_asked) {
                first_asked = true;
                open_theme(true);
            }
            g_dirty = true;
        }
        if (ui.working && now - g_last_spin >= 80000000ull) {
            editor_spin(ui.ed);
            g_last_spin = now;
            g_dirty = true;
        }
        if (uik_flash_expire(now)) g_dirty = true;
        if (g_view.bar_until && now >= g_view.bar_until) {
            g_view.bar_until = 0;
            g_dirty = true;
        }
        if (g_dirty && now - last_draw >= 8000000ull) {
            draw(false);
            last_draw = now;
            g_dirty = false;
        }
    }
    keyin_free(&kin);
    screen_stop(&ui.scr);
    Err result = OK;
    if (ui.busy) {
        // quitting during a run: stop it and wait for it to end (the ring kept drained, so the
        // agent never blocks on it); nothing it uses is freed before
        ops->abort(host);
        uint64_t until = now_ns() + 5000000000ull;
        while (ui.busy && now_ns() < until) {
            struct pollfd pf = {g_wake[0], POLLIN, 0};
            atomic_store(&g_ui_sleeping, 1);
            if (atomic_load(&g_rhead) == atomic_load(&g_rtail)) poll(&pf, 1, 50);
            atomic_store(&g_ui_sleeping, 0);
            char junk[256];
            while (read(g_wake[0], junk, sizeof junk) > 0) {}
            ring_drain(false);
        }
    }
    if (ui.busy) {
        // still running (a read that ignores the abort): the caller exits the process at once
        // without freeing what the run uses
        U = NULL;
        return "AgentBusy";
    }
    pthread_mutex_lock(&g_tlock);
    g_tquit = true;
    pthread_cond_signal(&g_tcond);
    pthread_mutex_unlock(&g_tlock);
    pthread_join(ui.thread, NULL);
    ring_drain(false);
    if (ui.re_id && ops->restart) ops->restart(host, ui.re_provider, ui.re_id);
    if (ui.dl_pid > 0) tui_stop_child(ui.dl_pid, true, 2000, NULL); // bounded: SIGKILL after 2 s
    if (ui.dl_fd >= 0) close(ui.dl_fd);
    free(ui.dl_name);
    free(ui.dl_title);
    free(ui.dl_models_dir);
    buf_free(&ui.dl_line);
    free(ui.re_provider);
    stream_reset();
    editor_free(ui.ed);
    comp_free(ui.header);
    comp_free(ui.chat);
    comp_free(ui.pending);
    comp_free(ui.status);
    sel_free(&ui.list);
    buf_free(&ui.search.value);
    vec_free(&ui.stats);
    vec_free(&ui.pending_tools);
    pthread_mutex_lock(&g_mlock);
    for (size_t i = 0; i < g_msgq.len; i++) free(g_msgq.ptr[i].text);
    vec_free(&g_msgq);
    pthread_mutex_unlock(&g_mlock);
    for (size_t i = 0; i < ui.ninitial; i++) free(ui.initial[i]);
    free(ui.initial);
    free(ui.re_id);
    free(ui.sw_provider);
    free(ui.sw_id);
    free(ui.sw_title);
    free(ui.theme);
    free(ui.theme_before);
    free(ui.thinking);
    free(ui.default_thinking);
    free((char *)ui.model.provider);
    free((char *)ui.model.id);
    free((char *)ui.model.name);
    free((char *)ui.model.label);
    free((char *)ui.model.label_brief);
    free(ui.default_provider);
    free(ui.default_model);
    free(g_job);
    g_job = NULL;
    // the git look may still run (its commands are bounded, 1 s each): joined before the wake pipe
    // it writes to is closed
    atomic_store(&g_git_quit, true);
    if (g_git_started) pthread_join(g_git_th, NULL);
    g_git_started = false;
    atomic_store(&g_git_wake, -1);
    code_update_check_stop();
    code_mcp_set_waker(NULL);
    close(g_wake[0]);
    close(g_wake[1]);
    g_wake[0] = g_wake[1] = -1;
    U = NULL;
    return result;
}

// ---- for the features kept in their own files (code_ui_ext.h) --------------------------------------------

const CodeUiConfig *uix_cfg(void) { return U->cfg; }
const CodeUiOps *uix_ops(void) { return U->ops; }
CodeUiHost *uix_host(void) { return U->host; }
Comp *uix_chat(void) { return U->chat; }
Editor *uix_editor(void) { return U->ed; }
bool uix_busy(void) { return U->busy; }
bool uix_tools_expanded(void) { return U->tools_expanded; }
void uix_show_status(const char *msg) { show_status(msg); }
void uix_show_error(const char *msg) { show_error(msg); }
void uix_show_warning(const char *msg) { show_warning(msg); }
void uix_render_history(void) { render_history(); }
void uix_request_render(void) { request_render(); }

void uix_open_selector(void (*render)(int width, Lines *out), void (*key)(const Key *k), void (*free_data)(void *), void *data) {
    close_selector();
    g_ext_render = render;
    g_ext_key = key;
    U->sel = SEL_EXT;
    U->sel_data = data;
    U->sel_free = free_data;
}

void *uix_selector_data(void) { return U->sel == SEL_EXT ? U->sel_data : NULL; }

void uix_close_selector(void) { close_selector(); }

void uix_show_thinking(const char *level) {
    free(U->thinking);
    U->thinking = dupz(level);
    editor_set_border(U->ed, th_thinking_color(level));
}

void uix_start_compact(const char *instructions) {
    U->busy = true;
    pthread_mutex_lock(&g_tlock);
    free(g_job);
    Buf b = {0};
    buf_addc(&b, '\x01');
    buf_adds(&b, instructions ? instructions : "");
    g_job = strdup(b.ptr);
    buf_free(&b);
    pthread_cond_signal(&g_tcond);
    pthread_mutex_unlock(&g_tlock);
}

void uix_working(const char *message) {
    if (!message) {
        set_working(false);
        return;
    }
    editor_set_working(U->ed, message);
    if (!U->working) g_last_spin = now_ns();
    U->working = true;
}

// handleCopyCommand's preferSelection: !getCopyOnSelect() && hasActiveSelection() copies the
// selection (copyActiveSelectionToClipboard); false when there is none to copy.
bool uix_copy_selection(void) {
    if (copy_on_select() || !g_view.sel_on) return false;
    copy_selection(U->scr.cols);
    return true;
}

void uix_abort(void) {
    if (U->busy) U->ops->abort(U->host);
}
