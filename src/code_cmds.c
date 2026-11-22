// Saetta Code's /copy, /hotkeys, /export, /import, /share and /scoped-models (see code_cmds.h).
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at v1.5.2,
// b4afddd): packages/coding-agent/src/modes/interactive/interactive-mode.ts (handleCopyCommand,
// handleHotkeysCommand, handleExportCommand, getPathCommandArgument, handleImportCommand,
// promptForMissingSessionCwd, switchToForkedSession, showExtensionConfirm, showModelsSelector),
// modes/interactive/session-share.ts (shareSession, shareViaGist), components/
// (extension-selector.ts, scoped-models-selector.ts, bordered-loader.ts, keybinding-hints.ts,
// dynamic-border.ts), core/agent-session.ts (getLastAssistantText, cycleModel,
// _cycleScopedModel), core/agent-session-runtime.ts (importFromJsonl), core/session-cwd.ts,
// core/session-manager.ts (SessionManager.open, forkFrom), core/model-resolver.ts
// (resolveModelScopeFromModels), core/keybindings.ts and packages/tui/src/keybindings.ts (the
// default keys), utils/clipboard.ts and clipboard-command.ts (copyToClipboard on macOS),
// packages/tui/src/fuzzy.ts (fuzzyFilter), components/loader.ts, input.ts.
#include "code_cmds.h"
#include "code_cfg.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <fnmatch.h>
#include <limits.h>
#include <signal.h>
#include <spawn.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static CmdsUi G;
static char *g_proc_cwd; // process.cwd(): paths typed in the commands are relative to it

// ---- small views (pi-tui Text, Spacer, DynamicBorder) -----------------------------------------------

static void text_lines(Str text, int px, int width, Lines *out) {
    Comp *t = comp_text(text, px, 0);
    comp_render(t, width, out);
    comp_free(t);
}

static void border_line(Lines *out, int width) {
    Buf b = {0}, r = {0};
    tui_repeat(&r, "\xe2\x94\x80", MAX(1, width));
    th_fg(&b, T_BORDER, buf_str(&r));
    lines_addb(out, &b);
    buf_free(&r);
}

static void border_render(void *ctx, int width, Lines *out) {
    (void)ctx;
    border_line(out, width);
}

// ---- keys (keybinding-hints.ts formatKeyText, the default keybindings) ---------------------------------

// formatKeyText(key, {capitalize}): each part of each "/" alternative capitalized, alt shown as
// option (macOS).
static void key_text(Buf *b, const char *keys, bool capitalize) {
    for (const char *p = keys; *p;) {
        const char *e = p;
        while (*e && *e != '/' && *e != '+') e++;
        bool alt = e - p == 3 && !strncasecmp(p, "alt", 3);
        const char *part = alt ? "option" : p;
        size_t n = alt ? 6 : (size_t)(e - p);
        if (n) {
            buf_addc(b, capitalize ? (char)toupper((unsigned char)part[0]) : part[0]);
            buf_add(b, part + 1, n - 1);
        }
        if (*e) buf_addc(b, *e++);
        p = e;
    }
}

static const char *kd(Buf *scratch, const char *keys) {
    scratch->len = 0;
    key_text(scratch, keys, true);
    return strdup(scratch->ptr ? scratch->ptr : "");
}

// keyHint(keybinding, description) for a default key
static void key_hint(Buf *b, const char *keys, const char *desc) {
    Buf k = {0}, d = {0};
    key_text(&k, keys, false);
    th_fg(b, T_DIM, buf_str(&k));
    buf_printf(&d, " %s", desc);
    th_fg(b, T_MUTED, buf_str(&d));
    buf_free(&k);
    buf_free(&d);
}

// ---- the dialogs in place of the editor -------------------------------------------------------------

typedef enum { D_NONE, D_SELECT, D_SCOPED, D_LOADER } DialogKind;

static struct {
    DialogKind kind;
    // ExtensionSelectorComponent
    char *title;
    char **options;
    size_t noptions, selected;
    void (*done)(int choice); // the option's index, -1 when cancelled
    // BorderedLoader
    char *message;
    int frame;
    uint64_t last_frame_ns;
    void (*abort)(void);
} D;

static void dialog_close(void) {
    free(D.title);
    for (size_t i = 0; i < D.noptions; i++) free(D.options[i]);
    free(D.options);
    free(D.message);
    memset(&D, 0, sizeof D);
    G.request_render();
}

bool cmds_dock_active(void) { return D.kind != D_NONE; }

// showExtensionSelector (an ExtensionSelectorComponent in the editor's place)
static void open_select(const char *title, const char *const *options, size_t n, void (*done)(int)) {
    dialog_close();
    D.kind = D_SELECT;
    D.title = strdup(title);
    D.options = xcalloc(n + 1, sizeof *D.options);
    for (size_t i = 0; i < n; i++) D.options[i] = strdup(options[i]);
    D.noptions = n;
    D.done = done;
    G.request_render();
}

static void select_render(int width, Lines *out) {
    border_line(out, width);
    lines_addz(out, "");
    Buf t = {0}, bold = {0};
    // theme.bold is chalk's: closed before each line break and opened again after it
    for (Str rest = str(D.title);;) {
        ptrdiff_t nl = str_find_char(rest, '\n', 0);
        th_bold(&bold, nl >= 0 ? str_sub(rest, 0, (size_t)nl) : rest);
        if (nl < 0) break;
        buf_addc(&bold, '\n');
        rest = str_sub(rest, (size_t)nl + 1, rest.len);
    }
    th_fg(&t, T_ACCENT, buf_str(&bold));
    text_lines(buf_str(&t), 1, width, out);
    buf_free(&t);
    buf_free(&bold);
    lines_addz(out, "");
    const size_t maxv = 10;
    long start = (long)D.selected - (long)(maxv / 2);
    if (start > (long)D.noptions - (long)maxv) start = (long)D.noptions - (long)maxv;
    if (start < 0) start = 0;
    size_t end = MIN((size_t)start + maxv, D.noptions);
    for (size_t i = (size_t)start; i < end; i++) {
        Buf l = {0};
        if (i == D.selected) {
            th_fgz(&l, T_ACCENT, "\xe2\x86\x92 ");
            th_fgz(&l, T_ACCENT, D.options[i]);
        } else {
            buf_adds(&l, "  ");
            th_fgz(&l, T_TEXT, D.options[i]);
        }
        text_lines(buf_str(&l), 1, width, out);
        buf_free(&l);
    }
    if (start > 0 || end < D.noptions) {
        Buf s = {0}, l = {0};
        buf_printf(&s, "  (%zu/%zu)", D.selected + 1, D.noptions);
        th_fg(&l, T_MUTED, buf_str(&s));
        text_lines(buf_str(&l), 0, width, out);
        buf_free(&s);
        buf_free(&l);
    }
    lines_addz(out, "");
    Buf h = {0};
    // rawKeyHint("↑↓", "navigate") + "  " + keyHint(confirm) + "  " + keyHint(cancel)
    key_hint(&h, "\xe2\x86\x91\xe2\x86\x93", "navigate");
    buf_adds(&h, "  ");
    key_hint(&h, "enter", "select");
    buf_adds(&h, "  ");
    key_hint(&h, "escape/ctrl+c", "cancel");
    text_lines(buf_str(&h), 1, width, out);
    buf_free(&h);
    lines_addz(out, "");
    border_line(out, width);
}

static void select_key(const Key *k) {
    if (k->kind == K_CTRL && k->ch == 'o') {
        if (G.toggle_tools) G.toggle_tools();
    } else if (k->kind == K_UP || (k->kind == K_CHAR && k->ch == 'k')) {
        if (D.selected > 0) D.selected--;
    } else if (k->kind == K_DOWN || (k->kind == K_CHAR && k->ch == 'j')) {
        if (D.selected + 1 < D.noptions) D.selected++;
    } else if (k->kind == K_ENTER || (k->kind == K_CTRL && k->ch == 'j')) {
        void (*done)(int) = D.done;
        int c = (int)D.selected;
        dialog_close();
        if (done) done(c);
    } else if (k->kind == K_ESC || (k->kind == K_CTRL && k->ch == 'c')) {
        void (*done)(int) = D.done;
        dialog_close();
        if (done) done(-1);
    }
}

// BorderedLoader(tui, theme, message): a cancellable Loader between two borders.
static const char *const SPIN[] = {"\xe2\xa0\x8b", "\xe2\xa0\x99", "\xe2\xa0\xb9", "\xe2\xa0\xb8", "\xe2\xa0\xbc",
                                   "\xe2\xa0\xb4", "\xe2\xa0\xa6", "\xe2\xa0\xa7", "\xe2\xa0\x87", "\xe2\xa0\x8f"};

static void open_loader(const char *message, void (*abort)(void)) {
    dialog_close();
    D.kind = D_LOADER;
    D.message = strdup(message);
    D.abort = abort;
    D.last_frame_ns = now_ns();
    G.request_render();
}

static void loader_render(int width, Lines *out) {
    border_line(out, width);
    lines_addz(out, ""); // Loader.render: a blank line, then its text
    Buf l = {0};
    th_fgz(&l, T_ACCENT, SPIN[D.frame % 10]);
    buf_addc(&l, ' ');
    th_fgz(&l, T_MUTED, D.message);
    text_lines(buf_str(&l), 1, width, out);
    buf_free(&l);
    lines_addz(out, "");
    Buf h = {0};
    key_hint(&h, "escape/ctrl+c", "cancel");
    text_lines(buf_str(&h), 1, width, out);
    buf_free(&h);
    lines_addz(out, "");
    border_line(out, width);
}

// ---- the session (what the commands read from the agent's side) -----------------------------------------

static CmdsSession session_now(void) {
    CmdsSession s = {0};
    const char *leaf = NULL;
    s.entries = G.ops->entries ? G.ops->entries(G.host, &leaf) : NULL;
    s.leaf = leaf;
    s.session_file = G.ops->session_file ? G.ops->session_file(G.host) : NULL;
    s.system_prompt = G.ops->system_prompt ? G.ops->system_prompt(G.host) : NULL;
    s.tools = G.ops->tool_decls ? G.ops->tool_decls(G.host) : NULL;
    return s;
}

static CmdsTheme theme_now(void) {
    return (CmdsTheme){.name = G.theme_name ? G.theme_name() : "system", .colors = G.term_colors ? G.term_colors() : NULL,
                       .appearance = G.appearance ? G.appearance() : 0};
}

static const char *home_dir(void) { return getenv("HOME") ? getenv("HOME") : "/"; }

// ---- /copy (handleCopyCommand, getLastAssistantText, copyToClipboard) -----------------------------------

// getLastAssistantText: the text blocks of the last assistant message (an aborted one without
// content skipped), trimmed; NULL when there is none.
static char *last_assistant_text(void) {
    const JValue *msgs = G.ops->messages ? G.ops->messages(G.host) : NULL;
    for (size_t i = msgs && json_is(msgs, J_ARRAY) ? msgs->arr.len : 0; i-- > 0;) {
        const JValue *m = msgs->arr.items[i];
        if (!str_eqz(json_strs(json_get(m, "role")), "assistant")) continue;
        const JValue *c = json_get(m, "content");
        size_t nc = json_is(c, J_ARRAY) ? c->arr.len : 0;
        if (str_eqz(json_strs(json_get(m, "stopReason")), "aborted") && nc == 0) continue;
        Buf t = {0};
        for (size_t k = 0; k < nc; k++)
            if (str_eqz(json_strs(json_get(c->arr.items[k], "type")), "text"))
                buf_addstr(&t, json_strs(json_get(c->arr.items[k], "text")));
        Str tr = str_trim(buf_str(&t));
        char *r = tr.len ? strndup(tr.ptr, tr.len) : NULL;
        buf_free(&t);
        return r;
    }
    return NULL;
}

// runClipboardCommand: the command from /, detached into its own process group, stdin fed, its
// output ignored, killed after the timeout; true on exit status 0.
static bool clipboard_command(const char *cmd, Str input, int timeout_ms) {
    int pi[2];
    if (tui_pipe(pi)) return false;
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, pi[0], 0);
    posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addchdir(&fa, "/");
    char *argv[] = {(char *)cmd, NULL};
    pid_t pid;
    int rc = tui_spawn(&pid, argv, &fa, false, true);
    posix_spawn_file_actions_destroy(&fa);
    close(pi[0]);
    if (rc) {
        close(pi[1]);
        return false;
    }
    void (*old)(int) = signal(SIGPIPE, SIG_IGN);
    for (size_t off = 0; off < input.len;) {
        ssize_t w = write(pi[1], input.ptr + off, input.len - off);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) break;
        off += (size_t)w;
    }
    close(pi[1]);
    signal(SIGPIPE, old);
    uint64_t until = now_ns() + (uint64_t)timeout_ms * 1000000ull;
    int st = 0;
    for (;;) {
        pid_t r = waitpid(pid, &st, WNOHANG);
        if (r == pid) return WIFEXITED(st) && WEXITSTATUS(st) == 0;
        if (r < 0 && errno != EINTR) return false;
        if (now_ns() >= until) {
            kill(pid, SIGKILL);
            while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
            return false;
        }
        struct timespec ts = {0, 5000000};
        nanosleep(&ts, NULL);
    }
}

#define MAX_OSC52_ENCODED_LENGTH 100000

// emitOsc52: false when the encoded text is over the limit.
static bool emit_osc52(Str text) {
    static const char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    if ((text.len + 2) / 3 * 4 > MAX_OSC52_ENCODED_LENGTH) return false;
    Buf o = {0};
    buf_adds(&o, "\x1b]52;c;");
    for (size_t i = 0; i < text.len; i += 3) {
        uint32_t v = (uint32_t)(uint8_t)text.ptr[i] << 16;
        if (i + 1 < text.len) v |= (uint32_t)(uint8_t)text.ptr[i + 1] << 8;
        if (i + 2 < text.len) v |= (uint8_t)text.ptr[i + 2];
        buf_addc(&o, tbl[v >> 18 & 63]);
        buf_addc(&o, tbl[v >> 12 & 63]);
        buf_addc(&o, i + 1 < text.len ? tbl[v >> 6 & 63] : '=');
        buf_addc(&o, i + 2 < text.len ? tbl[v & 63] : '=');
    }
    buf_adds(&o, "\x07");
    for (size_t off = 0; off < o.len;) {
        ssize_t w = write(1, o.ptr + off, o.len - off);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) break;
        off += (size_t)w;
    }
    buf_free(&o);
    return true;
}

// copyToClipboard on macOS: pbcopy (pi tries its native helper first; Saetta Code has none), then
// OSC 52 for a remote session (SSH, mosh, PI_CLIPBOARD_OSC52=1). NULL, or the error message.
const char *cmds_copy_text(Str text) {
    bool copied = clipboard_command("pbcopy", text, 5000);
    const char *e;
    bool remote = ((e = getenv("SSH_CONNECTION")) && *e) || ((e = getenv("SSH_CLIENT")) && *e) ||
                  ((e = getenv("MOSH_CONNECTION")) && *e) || ((e = getenv("PI_CLIPBOARD_OSC52")) && !strcmp(e, "1"));
    bool oversized = false;
    if (remote) {
        if (emit_osc52(text)) copied = true;
        else oversized = true;
    }
    if (copied) return NULL;
    if (oversized) return "Clipboard unavailable: text exceeds the OSC 52 size limit";
    return "Clipboard unavailable";
}

// handleCopyCommand: flash (app.message.copy's flashConfirmation) confirms with it instead of the status.
void cmds_copy(void (*flash)(const char *msg)) {
    char *text = last_assistant_text();
    if (!text) {
        G.show_error("No agent messages to copy yet.");
        return;
    }
    const char *err = cmds_copy_text(str(text));
    if (err) G.show_error(err);
    else if (flash) flash("Copied!");
    else G.show_status("Copied last agent message to clipboard");
    free(text);
}

static void cmd_copy(void) { cmds_copy(NULL); }

// ---- /hotkeys (handleHotkeysCommand) ---------------------------------------------------------------------

static void cmd_hotkeys(void) {
    Buf s = {0};
    // the default keys (packages/tui/src/keybindings.ts, core/keybindings.ts on macOS)
    const char *cursorUp = kd(&s, "up"), *cursorDown = kd(&s, "down"), *cursorLeft = kd(&s, "left/ctrl+b"),
               *cursorRight = kd(&s, "right/ctrl+f"), *cursorWordLeft = kd(&s, "alt+left/ctrl+left/alt+b"),
               *cursorWordRight = kd(&s, "alt+right/ctrl+right/alt+f"), *cursorLineStart = kd(&s, "home/ctrl+a"),
               *cursorLineEnd = kd(&s, "end/ctrl+e"), *jumpForward = kd(&s, "ctrl+]"), *jumpBackward = kd(&s, "ctrl+alt+]"),
               *pageUp = kd(&s, "pageUp/ctrl+pageUp"), *pageDown = kd(&s, "pageDown/ctrl+pageDown");
    const char *submit = kd(&s, "enter"), *newLine = kd(&s, "shift+enter/ctrl+j"),
               *deleteWordBackward = kd(&s, "ctrl+w/alt+backspace"), *deleteWordForward = kd(&s, "alt+d/alt+delete"),
               *deleteToLineStart = kd(&s, "ctrl+u"), *deleteToLineEnd = kd(&s, "ctrl+k"), *yank = kd(&s, "ctrl+y"),
               *yankPop = kd(&s, "alt+y"), *undo = kd(&s, "ctrl+-"), *tab = kd(&s, "tab");
    const char *interrupt = kd(&s, "escape"), *clear = kd(&s, "ctrl+c"), *exitk = kd(&s, "ctrl+d"), *suspend = kd(&s, "ctrl+z"),
               *cycleThinkingLevel = kd(&s, "shift+tab"), *cycleModelForward = kd(&s, "ctrl+p"), *selectModel = kd(&s, "ctrl+l"),
               *expandTools = kd(&s, "ctrl+o"), *toggleThinking = kd(&s, "ctrl+t"), *externalEditor = kd(&s, "ctrl+g"),
               *cycleModelBackward = kd(&s, "shift+ctrl+p"), *copyMessage = kd(&s, "ctrl+x"), *followUp = kd(&s, "alt+enter"),
               *dequeue = kd(&s, "alt+up"), *pasteImage = kd(&s, "ctrl+v");
    buf_free(&s);
    Buf h = {0};
    buf_printf(&h,
               "\n**Navigation**\n| Key | Action |\n|-----|--------|\n"
               "| `%s` / `%s` / `%s` / `%s` | Move cursor / browse history |\n"
               "| `%s` / `%s` | Move by word |\n"
               "| `%s` | Start of line |\n"
               "| `%s` | End of line |\n"
               "| `%s` | Jump forward to character |\n"
               "| `%s` | Jump backward to character |\n"
               "| `%s` / `%s` | Scroll by page |\n",
               cursorUp, cursorDown, cursorLeft, cursorRight, cursorWordLeft, cursorWordRight, cursorLineStart, cursorLineEnd,
               jumpForward, jumpBackward, pageUp, pageDown);
    buf_printf(&h,
               "\n**Editing**\n| Key | Action |\n|-----|--------|\n"
               "| `%s` | Send message |\n"
               "| `%s` | New line |\n"
               "| `%s` | Delete word backwards |\n"
               "| `%s` | Delete word forwards |\n"
               "| `%s` | Delete to start of line |\n"
               "| `%s` | Delete to end of line |\n"
               "| `%s` | Paste the most-recently-deleted text |\n"
               "| `%s` | Cycle through the deleted text after pasting |\n"
               "| `%s` | Undo |\n",
               submit, newLine, deleteWordBackward, deleteWordForward, deleteToLineStart, deleteToLineEnd, yank, yankPop, undo);
    buf_printf(&h,
               "\n**Other**\n| Key | Action |\n|-----|--------|\n"
               "| `%s` | Path completion / accept autocomplete |\n"
               "| `%s` | Cancel autocomplete / abort streaming |\n"
               "| `%s` | Clear editor (first) / exit (second) |\n"
               "| `%s` | Exit (when editor is empty) |\n"
               "| `%s` | Suspend to background |\n"
               "| `%s` | Cycle thinking level |\n"
               "| `%s` / `%s` | Cycle models |\n"
               "| `%s` | Open model selector |\n"
               "| `%s` | Toggle tool output expansion |\n"
               "| `%s` | Toggle thinking block visibility |\n"
               "| `%s` | Edit message in external editor |\n"
               "| `%s` | Copy selection or last assistant message |\n"
               "| `%s` | Queue follow-up message |\n"
               "| `%s` | Restore queued messages |\n"
               "| `%s` | Paste files on macOS, images, or text from clipboard |\n"
               "| `/` | Slash commands |\n"
               "| `!` | Run bash command |\n"
               "| `!!` | Run bash command (excluded from context) |\n",
               tab, interrupt, clear, exitk, suspend, cycleThinkingLevel, cycleModelForward, cycleModelBackward, selectModel,
               expandTools, toggleThinking, externalEditor, copyMessage, followUp, dequeue, pasteImage);
    const char *all[] = {cursorUp, cursorDown, cursorLeft, cursorRight, cursorWordLeft, cursorWordRight, cursorLineStart,
                         cursorLineEnd, jumpForward, jumpBackward, pageUp, pageDown, submit, newLine, deleteWordBackward,
                         deleteWordForward, deleteToLineStart, deleteToLineEnd, yank, yankPop, undo, tab, interrupt, clear,
                         exitk, suspend, cycleThinkingLevel, cycleModelForward, selectModel, expandTools, toggleThinking,
                         externalEditor, cycleModelBackward, copyMessage, followUp, dequeue, pasteImage};
    for (size_t i = 0; i < COUNT(all); i++) free((void *)all[i]);
    G.chat_add(comp_spacer(1));
    G.chat_add(comp_fn(border_render, NULL));
    Buf t = {0}, bold = {0};
    th_fgz(&t, T_ACCENT, "Keyboard Shortcuts");
    th_bold(&bold, buf_str(&t));
    G.chat_add(comp_text(buf_str(&bold), 1, 0));
    buf_free(&t);
    buf_free(&bold);
    G.chat_add(comp_spacer(1));
    static const MdStyle md_plain = {0};
    G.chat_add(comp_markdown(str_trim(buf_str(&h)), 1, 1, &md_plain));
    G.chat_add(comp_fn(border_render, NULL));
    buf_free(&h);
    G.request_render();
}

// ---- /export (handleExportCommand, getPathCommandArgument) -----------------------------------------------

// getPathCommandArgument: the first argument, or a quoted one; NULL when there is none (malloc'd).
static char *path_argument(const char *text, const char *command) {
    size_t cl = strlen(command);
    if (strncmp(text, command, cl) || text[cl] != ' ') return NULL;
    const char *a = text + cl + 1;
    while (*a && isspace((unsigned char)*a)) a++;
    if (!*a) return NULL;
    if (*a == '"' || *a == '\'') {
        const char *close = strchr(a + 1, *a);
        return close ? strndup(a + 1, (size_t)(close - a - 1)) : NULL;
    }
    const char *e = a;
    while (*e && !isspace((unsigned char)*e)) e++;
    return strndup(a, (size_t)(e - a));
}

static bool ends_with(const char *s, const char *suf) {
    size_t n = strlen(s), m = strlen(suf);
    return n >= m && !strcmp(s + n - m, suf);
}

static void cmd_export(const char *text) {
    char *out = path_argument(text, "/export");
    CmdsSession s = session_now();
    char *written = NULL, *err = NULL;
    bool ok;
    if (out && ends_with(out, ".jsonl")) ok = cmds_export_jsonl(&s, out, g_proc_cwd, &written, &err);
    else {
        CmdsTheme t = theme_now();
        ok = cmds_export_html(&s, &t, out, g_proc_cwd, &written, &err);
    }
    Buf m = {0};
    if (ok) {
        buf_printf(&m, "Session exported to: %s", written);
        G.show_status(m.ptr);
    } else {
        buf_printf(&m, "Failed to export session: %s", err ? err : "Unknown error");
        G.show_error(m.ptr);
    }
    buf_free(&m);
    free(written);
    free(err);
    free(out);
}

// ---- /import (handleImportCommand, importFromJsonl, promptForMissingSessionCwd) ----------------------------

static char *g_import_path;   // the argument as typed
static char *g_missing_cwd;   // the stored cwd that does not exist
static char *g_fallback_cwd;  // where the session would run instead

static char *session_cwd_of(const char *path) {
    char *data = NULL;
    size_t n = 0;
    if (read_file(path, (size_t)1 << 30, &data, &n)) return NULL;
    char *nl = memchr(data, '\n', n);
    Arena a = {0};
    JValue *h = NULL;
    char *cwd = NULL;
    if (!json_parse(&a, str_n(data, nl ? (size_t)(nl - data) : n), true, &h) && str_eqz(json_strs(json_get(h, "type")), "session") &&
        json_str(json_get(h, "cwd")))
        cwd = strdup(json_str(json_get(h, "cwd")));
    arena_reset(&a);
    free(data);
    return cwd;
}

static bool path_exists(const char *p) {
    struct stat st;
    return stat(p, &st) == 0;
}

// copyFileSync(src, dst, COPYFILE_EXCL)
static bool copy_excl(const char *src, const char *dst) {
    char *data = NULL;
    size_t n = 0;
    if (read_file(src, (size_t)1 << 31, &data, &n)) return false;
    int fd = open(dst, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    bool ok = fd >= 0;
    for (size_t off = 0; ok && off < n;) {
        ssize_t w = write(fd, data + off, n - off);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) ok = false;
        else off += (size_t)w;
    }
    if (fd >= 0) close(fd);
    free(data);
    return ok;
}

// The agent switched to the session file (createRuntime with its cwd), the transcript drawn again.
static bool set_session(const char *path, const char *cwd) {
    if (G.ops->open_session(G.host, path, cwd)) {
        Buf m = {0};
        buf_printf(&m, "Failed to import session: cannot open %s", path);
        G.show_error(m.ptr);
        buf_free(&m);
        return false;
    }
    if (G.set_cwd) G.set_cwd(cwd);
    // the thinking level the session runs at, on screen
    const char *leaf = NULL;
    const JValue *all = G.ops->entries ? G.ops->entries(G.host, &leaf) : NULL;
    const char *lv = NULL;
    for (size_t i = 0; all && json_is(all, J_ARRAY) && i < all->arr.len; i++)
        if (str_eqz(json_strs(json_get(all->arr.items[i], "type")), "thinking_level_change"))
            lv = json_str(json_get(all->arr.items[i], "thinkingLevel"));
    const CodeUiModel *m = G.model();
    if (lv && m && m->reasoning && (!G.thinking() || strcmp(lv, G.thinking()))) G.set_thinking(lv);
    G.render_history();
    return true;
}

static void import_cwd_choice(int choice);

typedef enum { IMPORTED, IMPORT_FAILED, IMPORT_ASKED } ImportResult;

// importFromJsonl(inputPath, cwdOverride): the file copied into this session's folder (a free name
// beside the others), opened there and run in its stored cwd; a cwd that no longer exists asks
// first (MissingSessionCwdError).
static ImportResult import_jsonl(const char *input, const char *cwd_override) {
    char *resolved = cmds_resolve_path(input, g_proc_cwd, home_dir());
    if (!path_exists(resolved)) {
        Buf m = {0};
        buf_printf(&m, "Failed to import session: File not found: %s", resolved);
        G.show_error(m.ptr);
        buf_free(&m);
        free(resolved);
        return IMPORT_FAILED;
    }
    const char *dir = G.ops->session_dir(G.host);
    (void)mkdir_p(dir);
    const char *base = strrchr(resolved, '/') ? strrchr(resolved, '/') + 1 : resolved;
    Buf dest = {0};
    buf_printf(&dest, "%s/%s", dir, base);
    char *dest_res = cmds_resolve_path(dest.ptr, "/", home_dir());
    bool stored = !strcmp(dest_res, resolved);
    free(dest_res);
    if (!stored) {
        const char *dot = strrchr(base, '.');
        size_t nl = dot && dot != base ? (size_t)(dot - base) : strlen(base);
        for (int suffix = 1; path_exists(dest.ptr); suffix++) {
            dest.len = 0;
            buf_printf(&dest, "%s/%.*s-%d%s", dir, (int)nl, base, suffix, dot && dot != base ? dot : "");
        }
        if (!copy_excl(resolved, dest.ptr)) {
            Buf m = {0};
            buf_printf(&m, "Failed to import session: cannot copy %s", resolved);
            G.show_error(m.ptr);
            buf_free(&m);
            buf_free(&dest);
            free(resolved);
            return IMPORT_FAILED;
        }
    }
    char *cwd = cwd_override ? strdup(cwd_override) : session_cwd_of(dest.ptr);
    if (!cwd_override && cwd && *cwd && !path_exists(cwd)) {
        // assertSessionCwdExists: the copy dropped, the choice asked
        if (!stored) unlink(dest.ptr);
        free(g_missing_cwd);
        free(g_fallback_cwd);
        g_missing_cwd = cwd;
        const char *cur = G.ops->session_cwd ? G.ops->session_cwd(G.host) : g_proc_cwd;
        g_fallback_cwd = strdup(cur);
        Buf title = {0}, o1 = {0}, o2 = {0};
        buf_printf(&title, "Session cwd does not exist: %s", g_missing_cwd);
        buf_printf(&o1, "Continue in %s (asks again next time)", g_fallback_cwd);
        buf_printf(&o2, "Fork into %s (new session ID, original kept)", g_fallback_cwd);
        const char *opts[] = {o1.ptr, o2.ptr, "Cancel"};
        open_select(title.ptr, opts, 3, import_cwd_choice);
        buf_free(&title);
        buf_free(&o1);
        buf_free(&o2);
        buf_free(&dest);
        free(resolved);
        return IMPORT_ASKED;
    }
    if (!cwd || !*cwd) {
        free(cwd);
        cwd = strdup(g_proc_cwd);
    }
    bool ok = set_session(dest.ptr, cwd);
    free(cwd);
    buf_free(&dest);
    free(resolved);
    return ok ? IMPORTED : IMPORT_FAILED;
}

// uuidv7 (createSessionId)
static void uuid_v7(char out[37]) {
    uint8_t b[16];
    arc4random_buf(b, sizeof b);
    uint64_t ms = wall_ns() / 1000000;
    for (int i = 0; i < 6; i++) b[i] = (uint8_t)(ms >> (40 - 8 * i));
    b[6] = (uint8_t)(0x70 | (b[6] & 0x0F));
    b[8] = (uint8_t)(0x80 | (b[8] & 0x3F));
    snprintf(out, 37, "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", b[0], b[1], b[2], b[3], b[4], b[5],
             b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
}

// SessionManager.forkFrom(source, cwd, sessionDir), then switchToForkedSession's switch: a new file
// with a new id whose header points at the source, the source's entries after it.
static void fork_into(const char *input, const char *cwd) {
    char *src = cmds_resolve_path(input, g_proc_cwd, home_dir());
    char *target = cmds_resolve_path(cwd, g_proc_cwd, home_dir());
    char *data = NULL;
    size_t n = 0;
    Arena a = {0};
    Buf out = {0}, path = {0}, m = {0};
    if (read_file(src, (size_t)1 << 30, &data, &n)) {
        buf_printf(&m, "Cannot fork: source session file is empty or invalid: %s", src);
        G.show_error(m.ptr);
        goto done;
    }
    char id[37];
    uuid_v7(id);
    uint64_t ms = wall_ns() / 1000000;
    time_t sec = (time_t)(ms / 1000);
    struct tm tm;
    gmtime_r(&sec, &tm);
    char ts[32];
    snprintf(ts, sizeof ts, "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour,
             tm.tm_min, tm.tm_sec, (int)(ms % 1000));
    char fts[32];
    snprintf(fts, sizeof fts, "%s", ts);
    for (char *p = fts; *p; p++)
        if (*p == ':' || *p == '.') *p = '-';
    const char *dir = G.ops->session_dir(G.host);
    (void)mkdir_p(dir);
    buf_printf(&path, "%s/%s_%s.jsonl", dir, fts, id);
    buf_adds(&out, "{\"type\":\"session\",\"version\":3,\"id\":");
    json_write_string(&out, str(id));
    buf_adds(&out, ",\"timestamp\":");
    json_write_string(&out, str(ts));
    buf_adds(&out, ",\"cwd\":");
    json_write_string(&out, str(target));
    buf_adds(&out, ",\"parentSession\":");
    json_write_string(&out, str(src));
    buf_adds(&out, "}\n");
    bool any = false;
    for (Str rest = str_n(data, n); rest.len;) {
        ptrdiff_t nl = str_find_char(rest, '\n', 0);
        Str line = nl >= 0 ? str_sub(rest, 0, (size_t)nl) : rest;
        rest = nl >= 0 ? str_sub(rest, (size_t)nl + 1, rest.len) : S("");
        JValue *e;
        if (!str_trim(line).len || json_parse(&a, line, true, &e) || !json_is(e, J_OBJECT)) continue;
        any = true;
        if (str_eqz(json_strs(json_get(e, "type")), "session")) continue;
        json_write(&out, e);
        buf_addc(&out, '\n');
    }
    if (!any) {
        buf_printf(&m, "Cannot fork: source session file is empty or invalid: %s", src);
        G.show_error(m.ptr);
        goto done;
    }
    int fd = open(path.ptr, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    bool ok = fd >= 0;
    for (size_t off = 0; ok && off < out.len;) {
        ssize_t w = write(fd, out.ptr + off, out.len - off);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) ok = false;
        else off += (size_t)w;
    }
    if (fd >= 0) close(fd);
    if (!ok) {
        buf_printf(&m, "Failed to import session: cannot write %s", path.ptr);
        G.show_error(m.ptr);
        goto done;
    }
    if (set_session(path.ptr, target)) {
        buf_printf(&m, "Forked into new session %s in %s", id, target);
        G.show_status(m.ptr);
    }
done:
    free(data);
    free(src);
    free(target);
    arena_reset(&a);
    buf_free(&out);
    buf_free(&path);
    buf_free(&m);
}

static void import_cwd_choice(int choice) {
    char *input = g_import_path;
    g_import_path = NULL;
    if (choice == 0) {
        if (import_jsonl(input, g_fallback_cwd) == IMPORTED) {
            Buf m = {0};
            buf_printf(&m, "Session imported from: %s", input);
            G.show_status(m.ptr);
            buf_free(&m);
        }
    } else if (choice == 1) fork_into(input, g_fallback_cwd);
    else G.show_status("Import cancelled");
    free(input);
}

static void import_confirmed(int choice) {
    if (choice != 0) {
        free(g_import_path);
        g_import_path = NULL;
        G.show_status("Import cancelled");
        return;
    }
    ImportResult r = import_jsonl(g_import_path, NULL);
    if (r == IMPORT_ASKED) return; // the path stays for the choice
    if (r == IMPORTED) {
        Buf m = {0};
        buf_printf(&m, "Session imported from: %s", g_import_path);
        G.show_status(m.ptr);
        buf_free(&m);
    }
    free(g_import_path);
    g_import_path = NULL;
}

static void cmd_import(const char *text) {
    char *in = path_argument(text, "/import");
    if (!in) {
        G.show_error("Usage: /import <path.jsonl>");
        return;
    }
    free(g_import_path);
    g_import_path = in;
    Buf t = {0};
    buf_printf(&t, "Import session\nReplace current session with %s?", in);
    const char *opts[] = {"Yes", "No"};
    open_select(t.ptr, opts, 2, import_confirmed);
    buf_free(&t);
}

// ---- /share (shareSession, shareViaGist) --------------------------------------------------------------

static struct {
    pid_t pid;
    int out_fd, err_fd;
    Buf out, err;
    char *dir, *html, *jsonl;
} g_share = {.pid = -1, .out_fd = -1, .err_fd = -1};

static void share_cleanup(void) {
    if (g_share.html) unlink(g_share.html);
    if (g_share.jsonl) unlink(g_share.jsonl);
    if (g_share.dir) rmdir(g_share.dir);
    free(g_share.html);
    free(g_share.jsonl);
    free(g_share.dir);
    if (g_share.out_fd >= 0) close(g_share.out_fd);
    if (g_share.err_fd >= 0) close(g_share.err_fd);
    buf_free(&g_share.out);
    buf_free(&g_share.err);
    g_share.html = g_share.jsonl = g_share.dir = NULL;
    g_share.out_fd = g_share.err_fd = -1;
    g_share.pid = -1;
}

static void share_abort(void) {
    if (g_share.pid > 0) tui_stop_child(g_share.pid, false, 1000, NULL);
    g_share.pid = -1;
    dialog_close();
    G.show_status("Share cancelled");
    share_cleanup();
}

static void share_done(int status) {
    dialog_close();
    Buf m = {0};
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        Str e = str_trim(buf_str(&g_share.err));
        buf_printf(&m, "Failed to create gist: %.*s", (int)(e.len ? e.len : 13), e.len ? e.ptr : "Unknown error");
        G.show_error(m.ptr);
    } else {
        Str url = str_trim(buf_str(&g_share.out));
        size_t s = url.len;
        while (s > 0 && url.ptr[s - 1] != '/') s--;
        Str id = str_sub(url, s, url.len);
        if (!url.len || !id.len) G.show_error("Failed to parse gist ID from gh output");
        else {
            // the gist alone: pi also prints a link to its session viewer at pi.dev, which Saetta Code
            // has no counterpart for (accepted difference)
            buf_printf(&m, "Gist: \x1b]8;;%.*s\x1b\\%.*s\x1b]8;;\x1b\\", (int)url.len, url.ptr, (int)url.len, url.ptr);
            G.show_status(m.ptr);
        }
    }
    buf_free(&m);
    share_cleanup();
}

static bool drain_fd(int *fd, Buf *into) {
    if (*fd < 0) return false;
    char chunk[4096];
    bool any = false;
    for (;;) {
        ssize_t n = read(*fd, chunk, sizeof chunk);
        if (n > 0) {
            buf_add(into, chunk, (size_t)n);
            any = true;
        } else if (n == 0) {
            close(*fd);
            *fd = -1;
            break;
        } else break; // EAGAIN
    }
    return any;
}

bool cmds_tick(void) {
    if (D.kind != D_LOADER) return false;
    bool changed = false;
    uint64_t now = now_ns();
    if (now - D.last_frame_ns >= 80000000ull) {
        D.frame = (D.frame + 1) % 10;
        D.last_frame_ns = now;
        changed = true;
    }
    if (g_share.pid > 0) {
        drain_fd(&g_share.out_fd, &g_share.out);
        drain_fd(&g_share.err_fd, &g_share.err);
        int st = 0;
        if (waitpid(g_share.pid, &st, WNOHANG) == g_share.pid) {
            g_share.pid = -1;
            // what is still in the pipes
            for (int i = 0; i < 50 && (g_share.out_fd >= 0 || g_share.err_fd >= 0); i++) {
                drain_fd(&g_share.out_fd, &g_share.out);
                drain_fd(&g_share.err_fd, &g_share.err);
                if (g_share.out_fd >= 0 || g_share.err_fd >= 0) {
                    struct timespec ts = {0, 2000000};
                    nanosleep(&ts, NULL);
                }
            }
            share_done(st);
            changed = true;
        }
    }
    return changed;
}

// shareViaGist: `gh gist create --public=false <file>` while the loader turns
static void share_via_gist(void) {
    int po[2], pe[2];
    if (tui_pipe(po)) {
        G.show_error("Failed to create gist: Unknown error");
        share_cleanup();
        return;
    }
    if (tui_pipe(pe)) {
        close(po[0]);
        close(po[1]);
        G.show_error("Failed to create gist: Unknown error");
        share_cleanup();
        return;
    }
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_adddup2(&fa, po[1], 1);
    posix_spawn_file_actions_adddup2(&fa, pe[1], 2);
    char *argv[] = {"gh", "gist", "create", "--public=false", g_share.html, NULL};
    pid_t pid;
    int rc = tui_spawn(&pid, argv, &fa, false, false);
    posix_spawn_file_actions_destroy(&fa);
    close(po[1]);
    close(pe[1]);
    if (rc) {
        close(po[0]);
        close(pe[0]);
        Buf m = {0};
        buf_printf(&m, "Failed to create gist: spawn gh %s", strerror(rc));
        G.show_error(m.ptr);
        buf_free(&m);
        share_cleanup();
        return;
    }
    fcntl(po[0], F_SETFL, O_NONBLOCK);
    fcntl(pe[0], F_SETFL, O_NONBLOCK);
    g_share.pid = pid;
    g_share.out_fd = po[0];
    g_share.err_fd = pe[0];
    open_loader("Creating gist...", share_abort);
}

static void cmd_share(void) {
    if (g_share.pid > 0) return;
    // mkdtempSync(join(os.tmpdir(), "pi-share-")): TMPDIR without its trailing slash
    char tmpl[PATH_MAX];
    const char *tmp = getenv("TMPDIR");
    if (!tmp || !*tmp) tmp = "/tmp";
    size_t tl = strlen(tmp);
    while (tl > 1 && tmp[tl - 1] == '/') tl--;
    snprintf(tmpl, sizeof tmpl, "%.*s/pi-share-XXXXXX", (int)tl, tmp);
    if (!mkdtemp(tmpl)) {
        G.show_error("Failed to export session: cannot create a temporary folder");
        return;
    }
    g_share.dir = strdup(tmpl);
    Buf j = {0}, h = {0};
    buf_printf(&j, "%s/session.jsonl", tmpl);
    buf_printf(&h, "%s/session.html", tmpl);
    g_share.jsonl = j.ptr;
    g_share.html = h.ptr;
    CmdsSession s = session_now();
    char *written = NULL, *err = NULL;
    // exportSessionForShare (the branch Radius would take; written and removed as pi does)
    if (!cmds_export_jsonl(&s, g_share.jsonl, g_proc_cwd, &written, &err)) {
        Buf m = {0};
        buf_printf(&m, "Failed to export session: %s", err ? err : "Unknown error");
        G.show_error(m.ptr);
        buf_free(&m);
        free(err);
        share_cleanup();
        return;
    }
    free(written);
    // spawnSync("gh", ["auth", "status"]): a failure to start is a non-zero status too
    static const char *const auth[] = {"gh", "auth", "status", NULL};
    Buf ignored = {0};
    bool logged_in = tui_run((char *const *)auth, NULL, S(""), &ignored, 30000);
    buf_free(&ignored);
    if (!logged_in) {
        G.show_error("GitHub CLI is not logged in. Run 'gh auth login' first.");
        share_cleanup();
        return;
    }
    CmdsTheme t = theme_now();
    if (!cmds_export_html(&s, &t, g_share.html, g_proc_cwd, &written, &err)) {
        Buf m = {0};
        buf_printf(&m, "Failed to export session: %s", err ? err : "Unknown error");
        G.show_error(m.ptr);
        buf_free(&m);
        free(err);
        share_cleanup();
        return;
    }
    free(written);
    share_via_gist();
}

// ---- fuzzy matching (fuzzy.ts) ------------------------------------------------------------------------

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

static void lower_ascii(Buf *b, Str s) {
    for (size_t i = 0; i < s.len; i++) buf_addc(b, (char)tolower((unsigned char)s.ptr[i]));
    if (!b->ptr) buf_adds(b, "");
}

static bool fuzzy_match(Str query, Str text, double *score) {
    Buf q = {0}, t = {0};
    lower_ascii(&q, query);
    lower_ascii(&t, text);
    bool ok = match_query(buf_str(&q), buf_str(&t), score);
    if (!ok) {
        // letters then digits, or digits then letters, swapped
        Str qs = buf_str(&q);
        size_t i = 0;
        bool alpha = qs.len && qs.ptr[0] >= 'a' && qs.ptr[0] <= 'z', digit = qs.len && qs.ptr[0] >= '0' && qs.ptr[0] <= '9';
        if (alpha) while (i < qs.len && qs.ptr[i] >= 'a' && qs.ptr[i] <= 'z') i++;
        else if (digit) while (i < qs.len && qs.ptr[i] >= '0' && qs.ptr[i] <= '9') i++;
        size_t j = i;
        if (alpha) while (j < qs.len && qs.ptr[j] >= '0' && qs.ptr[j] <= '9') j++;
        else if (digit) while (j < qs.len && qs.ptr[j] >= 'a' && qs.ptr[j] <= 'z') j++;
        if ((alpha || digit) && i > 0 && i < qs.len && j == qs.len) {
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
    return x->idx < y->idx ? -1 : x->idx > y->idx;
}

// fuzzyFilter: the indexes of the matching texts, best first; tokens split on whitespace and "/".
static size_t fuzzy_filter(char *const *texts, size_t n, Str query, size_t *out) {
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
            while (p < q.len && (isspace((unsigned char)q.ptr[p]) || q.ptr[p] == '/')) p++;
            size_t s = p;
            while (p < q.len && !(isspace((unsigned char)q.ptr[p]) || q.ptr[p] == '/')) p++;
            if (p == s) break;
            any = true;
            double s1;
            if (fuzzy_match(str_n(q.ptr + s, p - s), str(texts[i]), &s1)) total += s1;
            else all = false;
        }
        if (all) sc[m++] = (Scored){i, any ? total : 0};
    }
    qsort(sc, m, sizeof *sc, scored_cmp);
    for (size_t i = 0; i < m; i++) out[i] = sc[i].idx;
    free(sc);
    return m;
}

// ---- the search line (pi-tui Input) ---------------------------------------------------------------------

typedef struct {
    Buf value;
    size_t cursor;
} Input;

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

static void input_set(Input *in, const char *s) {
    in->value.len = 0;
    buf_adds(&in->value, s);
    in->cursor = in->value.len;
}

static void input_key(Input *in, const Key *k) {
    Buf *v = &in->value;
    if (!v->ptr) buf_adds(v, "");
    if (k->kind == K_CHAR && k->ch >= 0x20) {
        char enc[4];
        int n = tui_utf8(enc, k->ch);
        buf_reserve(v, (size_t)n);
        memmove(v->ptr + in->cursor + n, v->ptr + in->cursor, v->len - in->cursor + 1);
        memcpy(v->ptr + in->cursor, enc, (size_t)n);
        v->len += (size_t)n;
        in->cursor += (size_t)n;
    } else if (k->kind == K_PASTE && k->paste) {
        for (const char *p = k->paste; *p; p++)
            if ((unsigned char)*p >= 0x20) {
                buf_reserve(v, 1);
                memmove(v->ptr + in->cursor + 1, v->ptr + in->cursor, v->len - in->cursor + 1);
                v->ptr[in->cursor++] = *p;
                v->len++;
            }
    } else if (k->kind == K_BACKSPACE && in->cursor > 0) {
        size_t p = in->cursor - 1;
        while (p > 0 && ((unsigned char)v->ptr[p] & 0xc0) == 0x80) p--;
        memmove(v->ptr + p, v->ptr + in->cursor, v->len - in->cursor + 1);
        v->len -= in->cursor - p;
        in->cursor = p;
    } else if (k->kind == K_LEFT && in->cursor > 0) {
        in->cursor--;
        while (in->cursor > 0 && ((unsigned char)v->ptr[in->cursor] & 0xc0) == 0x80) in->cursor--;
    } else if (k->kind == K_RIGHT && in->cursor < v->len) in->cursor += tui_grapheme(buf_str(v), in->cursor);
    else if (k->kind == K_HOME) in->cursor = 0;
    else if (k->kind == K_END || (k->kind == K_CTRL && k->ch == 'e')) in->cursor = v->len;
    else if (k->kind == K_CTRL && k->ch == 'u') {
        v->len = 0;
        in->cursor = 0;
        v->ptr[0] = 0;
    }
}

// ---- model scope (resolveModelScopeFromModels, _cycleScopedModel) ---------------------------------------

static const char *const LEVELS[] = {"off", "minimal", "low", "medium", "high", "xhigh", "max"};

typedef struct {
    const CodeUiModel *model;
    const char *thinking; // a level from the pattern, or NULL
} Scoped;
typedef VEC(Scoped) ScopedVec;
typedef VEC(char *) NameVec;
static ScopedVec g_scope;  // the session's scoped models (empty: every model cycles)
static NameVec g_patterns; // settings.json enabledModels as this session knows it

// The models Ctrl+P cycles through and /scoped-models lists: those of models.json (a pack switch
// needs a confirmed restart, /model).
static size_t available(const CodeUiModel **out) {
    size_t n = 0;
    for (size_t i = 0; i < G.cfg->nmodels; i++)
        if (!G.cfg->models[i].pack) {
            if (out) out[n] = &G.cfg->models[i];
            n++;
        }
    return n;
}

static char *full_id(const CodeUiModel *m) {
    Buf b = {0};
    buf_printf(&b, "%s/%s", m->provider, m->id);
    return b.ptr;
}

static bool has_glob(const char *p) { return strchr(p, '*') || strchr(p, '?') || strchr(p, '['); }

// A trailing ":<level>" (splitThinkingSuffix), or NULL.
static const char *thinking_suffix(const char *p, size_t *base_len) {
    const char *c = strrchr(p, ':');
    if (!c) return NULL;
    for (size_t i = 0; i < COUNT(LEVELS); i++)
        if (!strcmp(c + 1, LEVELS[i])) {
            *base_len = (size_t)(c - p);
            return LEVELS[i];
        }
    return NULL;
}

// The model an exact reference names: provider/id, else the id (case-insensitive); NULL if none or
// several providers share the id.
static const CodeUiModel *exact_model(const char *ref, size_t len) {
    const CodeUiModel *avail[512];
    size_t n = MIN(available(NULL), COUNT(avail));
    available(avail);
    for (size_t i = 0; i < n; i++) {
        char *f = full_id(avail[i]);
        bool eq = strlen(f) == len && !strncasecmp(f, ref, len);
        free(f);
        if (eq) return avail[i];
    }
    const CodeUiModel *hit = NULL;
    for (size_t i = 0; i < n; i++)
        if (strlen(avail[i]->id) == len && !strncasecmp(avail[i]->id, ref, len)) {
            if (hit) return NULL;
            hit = avail[i];
        }
    return hit;
}

static void scope_add(ScopedVec *out, const CodeUiModel *m, const char *thinking) {
    for (size_t i = 0; i < out->len; i++)
        if (out->ptr[i].model == m) return;
    vec_push(out, ((Scoped){m, thinking}));
}

// resolveModelScopeFromModels(patterns, models): globs match provider/id or id; other patterns name
// one model, with an optional thinking level. Patterns that match nothing go to no_match (may be NULL).
static void resolve_scope(char *const *patterns, size_t np, ScopedVec *out, NameVec *no_match) {
    const CodeUiModel *avail[512];
    size_t n = MIN(available(NULL), COUNT(avail));
    available(avail);
    for (size_t k = 0; k < np; k++) {
        const char *p = patterns[k];
        size_t bl = strlen(p);
        const char *lv = thinking_suffix(p, &bl);
        if (!lv) bl = strlen(p);
        char *base = strndup(p, bl);
        bool any = false;
        if (has_glob(p)) {
            const CodeUiModel *ex = exact_model(base, bl);
            if (ex) {
                scope_add(out, ex, lv);
                any = true;
            } else
                for (size_t i = 0; i < n; i++) {
                    char *f = full_id(avail[i]);
                    if (!fnmatch(base, f, FNM_CASEFOLD) || !fnmatch(base, avail[i]->id, FNM_CASEFOLD)) {
                        scope_add(out, avail[i], lv);
                        any = true;
                    }
                    free(f);
                }
        } else {
            const CodeUiModel *ex = exact_model(p, strlen(p));
            if (ex) scope_add(out, ex, NULL);
            else if ((ex = exact_model(base, bl))) scope_add(out, ex, lv);
            any = ex != NULL;
        }
        if (!any && no_match) vec_push(no_match, strdup(p));
        free(base);
    }
}

static const char *display_name(const CodeUiModel *m) { return m->name && *m->name ? m->name : m->id; }

bool cmds_cycle_model(int dir) {
    if (!g_scope.len) return false;
    // the scoped models that are available, the current one's place among them
    ScopedVec sc = {0};
    for (size_t i = 0; i < g_scope.len; i++)
        if (!G.cfg->engine) vec_push(&sc, g_scope.ptr[i]);
    const CodeUiModel *cur = G.model();
    long ci = -1;
    for (size_t i = 0; i < sc.len; i++)
        if (cur && cur->provider && cur->id && !strcmp(sc.ptr[i].model->provider, cur->provider) && !strcmp(sc.ptr[i].model->id, cur->id))
            ci = (long)i;
    // getNextCycleIndex
    long next = -1;
    if (sc.len) {
        if (ci < 0) next = dir > 0 ? 0 : (long)sc.len - 1;
        else if (sc.len > 1) next = dir > 0 ? (ci + 1) % (long)sc.len : (ci - 1 + (long)sc.len) % (long)sc.len;
    }
    if (next < 0) {
        G.show_status("Only one model in scope");
        vec_free(&sc);
        return true;
    }
    Scoped s = sc.ptr[next];
    vec_free(&sc);
    G.choose_model(s.model);
    if (s.thinking && s.model->reasoning) G.set_thinking(s.thinking);
    Buf b = {0};
    const char *lv = G.thinking() ? G.thinking() : "off";
    buf_printf(&b, "Switched to %s", display_name(s.model));
    if (s.model->reasoning && strcmp(lv, "off")) buf_printf(&b, " (thinking: %s)", lv);
    G.show_status(b.ptr);
    buf_free(&b);
    return true;
}

static bool g_scope_notice; // the session started with a scope (showFullscreenModelScopeNotice)

// getModelScopeNotice: the scoped ids (with their thinking levels), and the cycle key
void cmds_header_notice(int width, Lines *out) {
    if (!g_scope_notice) return;
    if (g_scope.len) {
        Buf list = {0}, t = {0};
        buf_adds(&list, "Model scope: ");
        for (size_t i = 0; i < g_scope.len; i++) {
            if (i) buf_adds(&list, ", ");
            buf_adds(&list, g_scope.ptr[i].model->id);
            if (g_scope.ptr[i].thinking) buf_printf(&list, ":%s", g_scope.ptr[i].thinking);
        }
        th_fgz(&list, T_MUTED, " (Ctrl+P to cycle)");
        th_fg(&t, T_DIM, buf_str(&list));
        text_lines(buf_str(&t), 1, width, out);
        buf_free(&list);
        buf_free(&t);
    }
    lines_addz(out, "");
}

// ---- settings.json (SettingsManager.setEnabledModels: the file read again under its lock, the field
// set or dropped, written whole as JSON.stringify(settings, null, 2)) -------------------------------------

typedef struct {
    char *const *ids;
    size_t n;
    bool drop;
} EnabledModels;

// enabledModels dropped, then (unless dropping) set again at the end
static bool put_enabled_models(Arena *a, JValue *root, void *ctx) {
    const EnabledModels *e = ctx;
    JValue *out = json_new(a, J_OBJECT);
    for (size_t i = 0; i < root->obj.len; i++)
        if (!e->drop || !str_eqz(root->obj.items[i].key, "enabledModels")) json_put(a, out, root->obj.items[i].key, root->obj.items[i].val);
    if (!e->drop) {
        JValue *arr = json_new(a, J_ARRAY);
        for (size_t i = 0; i < e->n; i++) json_push(a, arr, json_string(a, str(e->ids[i])));
        json_put(a, out, S("enabledModels"), arr);
    }
    *root = *out;
    return true;
}

static void save_enabled_models(char *const *ids, size_t n, bool drop) {
    Buf path = {0}, why = {0};
    buf_printf(&path, "%s/settings.json", G.cfg->agent_dir);
    EnabledModels e = {ids, n, drop};
    CfgUpdateOpts o = {.parse_numbers = true,
                       .lock_tries = 20,
                       .locked = "settings.json is locked by another program; the model selection is not saved",
                       .unreadable = "settings.json could not be read; the model selection is not saved",
                       .unparsable = "settings.json does not parse; the model selection is not saved",
                       .unwritable = "Could not save the model selection in settings.json"};
    if (!cfg_json_update_ex(path.ptr, &o, put_enabled_models, &e, &why)) G.show_error(why.ptr ? why.ptr : "");
    buf_free(&path);
    buf_free(&why);
}

// ---- /scoped-models (showModelsSelector, ScopedModelsSelectorComponent) -----------------------------------

// getDistinctModelName: the name when it has words the id does not.
static bool name_distinct(const char *id, const char *name) {
    if (!name || !*name) return false;
    Buf words = {0}, compact = {0};
    for (const char *p = id; *p; p++) buf_addc(&words, isalnum((unsigned char)*p) ? (char)tolower((unsigned char)*p) : ' ');
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

typedef struct {
    const CodeUiModel **models; // the available models
    char **ids;                 // their provider/id
    size_t nmodels;
    bool all;            // enabledIds === null
    NameVec enabled;     // the explicit ordered list
    char **items;        // the rows shown (ids), filtered
    size_t nitems, selected;
    Input search;
    bool dirty;
} ScopedSel;
static ScopedSel *g_ss;

static const CodeUiModel *ss_model(ScopedSel *s, const char *id) {
    for (size_t i = 0; i < s->nmodels; i++)
        if (!strcmp(s->ids[i], id)) return s->models[i];
    return NULL;
}

static long ss_index(ScopedSel *s, const char *id) {
    for (size_t i = 0; i < s->enabled.len; i++)
        if (!strcmp(s->enabled.ptr[i], id)) return (long)i;
    return -1;
}

static bool ss_enabled(ScopedSel *s, const char *id) { return s->all || ss_index(s, id) >= 0; }

static void ss_set_list(ScopedSel *s, char **ids, size_t n) {
    for (size_t i = 0; i < s->enabled.len; i++) free(s->enabled.ptr[i]);
    s->enabled.len = 0;
    for (size_t i = 0; i < n; i++) vec_push(&s->enabled, strdup(ids[i]));
}

// normalizeEnabled: an explicit list that covers every available model is "all" again
static void ss_normalize(ScopedSel *s) {
    if (s->all || s->enabled.len != s->nmodels) return;
    for (size_t i = 0; i < s->nmodels; i++)
        if (ss_index(s, s->ids[i]) < 0) return;
    s->all = true;
    ss_set_list(s, NULL, 0);
}

static void ss_toggle(ScopedSel *s, const char *id) {
    if (s->all) {
        s->all = false;
        ss_set_list(s, NULL, 0);
        for (size_t i = 0; i < s->nmodels; i++)
            if (strcmp(s->ids[i], id)) vec_push(&s->enabled, strdup(s->ids[i]));
        return;
    }
    long k = ss_index(s, id);
    if (k >= 0) {
        free(s->enabled.ptr[k]);
        memmove(s->enabled.ptr + k, s->enabled.ptr + k + 1, (s->enabled.len - (size_t)k - 1) * sizeof(char *));
        s->enabled.len--;
        return;
    }
    vec_push(&s->enabled, strdup(id));
    ss_normalize(s);
}

static bool in_targets(char *const *t, size_t n, const char *id) {
    for (size_t i = 0; i < n; i++)
        if (!strcmp(t[i], id)) return true;
    return false;
}

static void ss_enable_all(ScopedSel *s, char *const *targets, size_t nt) {
    if (s->all) return;
    if (!targets) {
        targets = s->ids;
        nt = s->nmodels;
    }
    for (size_t i = 0; i < nt; i++)
        if (ss_index(s, targets[i]) < 0) vec_push(&s->enabled, strdup(targets[i]));
    ss_normalize(s);
}

static void ss_clear_all(ScopedSel *s, char *const *targets, size_t nt) {
    if (s->all) {
        s->all = false;
        ss_set_list(s, NULL, 0);
        if (targets)
            for (size_t i = 0; i < s->nmodels; i++)
                if (!in_targets(targets, nt, s->ids[i])) vec_push(&s->enabled, strdup(s->ids[i]));
        return;
    }
    if (!targets) {
        ss_set_list(s, NULL, 0);
        return;
    }
    size_t w = 0;
    for (size_t i = 0; i < s->enabled.len; i++) {
        if (in_targets(targets, nt, s->enabled.ptr[i])) free(s->enabled.ptr[i]);
        else s->enabled.ptr[w++] = s->enabled.ptr[i];
    }
    s->enabled.len = w;
}

// refresh: getSortedIds (the enabled ones in their order, then the rest), filtered by the search
static void ss_refresh(ScopedSel *s) {
    free(s->items);
    size_t cap = s->nmodels + s->enabled.len + 1;
    char **sorted = xmalloc(cap * sizeof *sorted);
    size_t n = 0;
    if (s->all)
        for (size_t i = 0; i < s->nmodels; i++) sorted[n++] = s->ids[i];
    else {
        for (size_t i = 0; i < s->enabled.len; i++) sorted[n++] = s->enabled.ptr[i];
        for (size_t i = 0; i < s->nmodels; i++)
            if (ss_index(s, s->ids[i]) < 0) sorted[n++] = s->ids[i];
    }
    const char *q = s->search.value.ptr ? s->search.value.ptr : "";
    if (*q) {
        char **texts = xmalloc((n + 1) * sizeof *texts);
        for (size_t i = 0; i < n; i++) {
            const CodeUiModel *m = ss_model(s, sorted[i]);
            Buf t = {0};
            if (m) buf_printf(&t, "%s %s %s/%s %s %s%s%s", m->id, m->provider, m->provider, m->id, m->provider, m->id, m->name ? " " : "",
                              m->name ? m->name : "");
            else buf_adds(&t, sorted[i]);
            texts[i] = t.ptr;
        }
        size_t *idx = xmalloc((n + 1) * sizeof *idx);
        size_t m = fuzzy_filter(texts, n, str(q), idx);
        s->items = xmalloc((m + 1) * sizeof *s->items);
        for (size_t i = 0; i < m; i++) s->items[i] = sorted[idx[i]];
        s->nitems = m;
        for (size_t i = 0; i < n; i++) free(texts[i]);
        free(texts);
        free(idx);
        free(sorted);
    } else {
        s->items = sorted;
        s->nitems = n;
    }
    if (s->selected >= s->nitems) s->selected = s->nitems ? s->nitems - 1 : 0;
}

// updateSessionModels: the session's scope from the selection (none when it covers every model or
// names none available)
static void ss_apply(ScopedSel *s) {
    g_scope.len = 0;
    if (s->all) return;
    bool any = false, every = true;
    for (size_t i = 0; i < s->enabled.len; i++)
        if (ss_model(s, s->enabled.ptr[i])) any = true;
    for (size_t i = 0; i < s->nmodels; i++)
        if (ss_index(s, s->ids[i]) < 0) every = false;
    if (any && !every) resolve_scope(s->enabled.ptr, s->enabled.len, &g_scope, NULL);
}

static void ss_footer(ScopedSel *s, Buf *out) {
    size_t en = 0, un = 0;
    for (size_t i = 0; i < s->enabled.len; i++)
        if (ss_model(s, s->enabled.ptr[i])) en++;
        else un++;
    Buf parts = {0};
    buf_adds(&parts, "  Enter toggle \xc2\xb7 Ctrl+A all \xc2\xb7 Ctrl+X clear \xc2\xb7 Ctrl+P provider \xc2\xb7 Option+Up/Option+Down reorder \xc2\xb7 Ctrl+S save \xc2\xb7 ");
    if (s->all) buf_adds(&parts, "all enabled");
    else {
        buf_printf(&parts, "%zu/%zu enabled", en, s->nmodels);
        if (un) buf_printf(&parts, " \xc2\xb7 %zu unavailable", un);
    }
    if (s->dirty) {
        buf_addc(&parts, ' ');
        th_fg(out, T_DIM, buf_str(&parts));
        th_fgz(out, T_WARNING, "(unsaved)");
    } else th_fg(out, T_DIM, buf_str(&parts));
    buf_free(&parts);
}

static void scoped_render(int width, Lines *out) {
    ScopedSel *s = g_ss;
    border_line(out, width);
    lines_addz(out, "");
    Buf t = {0}, b = {0};
    th_bold(&b, S("Model Configuration"));
    th_fg(&t, T_ACCENT, buf_str(&b));
    text_lines(buf_str(&t), 0, width, out);
    t.len = 0;
    th_fgz(&t, T_MUTED, "Session-only. Ctrl+S to save to settings.");
    text_lines(buf_str(&t), 0, width, out);
    lines_addz(out, "");
    input_render(&s->search, width, out);
    lines_addz(out, "");
    if (!s->nitems) {
        t.len = 0;
        th_fgz(&t, T_MUTED, "  No matching models");
        text_lines(buf_str(&t), 0, width, out);
    } else {
        const long maxv = 8;
        long start = (long)s->selected - maxv / 2;
        if (start > (long)s->nitems - maxv) start = (long)s->nitems - maxv;
        if (start < 0) start = 0;
        size_t end = MIN((size_t)(start + maxv), s->nitems);
        for (size_t i = (size_t)start; i < end; i++) {
            const char *id = s->items[i];
            const CodeUiModel *m = ss_model(s, id);
            bool sel = i == s->selected;
            Buf l = {0}, styled = {0};
            if (sel) th_fgz(&l, T_ACCENT, "\xe2\x86\x92 ");
            else buf_adds(&l, "  ");
            if (m && ss_enabled(s, id)) th_fgz(&l, T_ACCENT, "\xe2\x9c\x93 ");
            else buf_adds(&l, "  ");
            if (m) buf_adds(&styled, m->id);
            else th_strike(&styled, str(id));
            if (sel) th_fg(&l, T_ACCENT, buf_str(&styled));
            else buf_addstr(&l, buf_str(&styled));
            if (m && name_distinct(m->id, m->name)) {
                Buf nm = {0};
                buf_printf(&nm, " (%s)", m->name);
                th_fg(&l, T_MUTED, buf_str(&nm));
                buf_free(&nm);
            }
            Buf badge = {0};
            if (m) buf_printf(&badge, " [%s]", m->provider);
            else buf_adds(&badge, " [unavailable]");
            th_fg(&l, T_MUTED, buf_str(&badge));
            text_lines(buf_str(&l), 0, width, out);
            buf_free(&badge);
            buf_free(&l);
            buf_free(&styled);
        }
        if (start > 0 || end < s->nitems) {
            Buf si = {0};
            buf_printf(&si, "  (%zu/%zu)", s->selected + 1, s->nitems);
            t.len = 0;
            th_fg(&t, T_MUTED, buf_str(&si));
            text_lines(buf_str(&t), 0, width, out);
            buf_free(&si);
        }
        lines_addz(out, "");
        const CodeUiModel *m = ss_model(s, s->items[s->selected]);
        Buf nm = {0};
        if (m) buf_printf(&nm, "  Model Name: %s", m->name ? m->name : m->id);
        else buf_adds(&nm, "  Model unavailable");
        t.len = 0;
        th_fg(&t, T_MUTED, buf_str(&nm));
        text_lines(buf_str(&t), 0, width, out);
        buf_free(&nm);
    }
    lines_addz(out, "");
    // the catalogs: Saetta Code's models.json is read at start, nothing to refresh
    t.len = 0;
    th_fgz(&t, T_SUCCESS, "  Model catalogs refreshed.");
    text_lines(buf_str(&t), 0, width, out);
    t.len = 0;
    ss_footer(s, &t);
    text_lines(buf_str(&t), 0, width, out);
    border_line(out, width);
    buf_free(&t);
    buf_free(&b);
}

static void scoped_free(void) {
    ScopedSel *s = g_ss;
    if (!s) return;
    for (size_t i = 0; i < s->nmodels; i++) free(s->ids[i]);
    free(s->ids);
    free(s->models);
    for (size_t i = 0; i < s->enabled.len; i++) free(s->enabled.ptr[i]);
    vec_free(&s->enabled);
    free(s->items);
    buf_free(&s->search.value);
    free(s);
    g_ss = NULL;
}

static void scoped_changed(ScopedSel *s) {
    s->dirty = true;
    ss_refresh(s);
    ss_apply(s);
}

static void scoped_key(const Key *k) {
    ScopedSel *s = g_ss;
    if (k->kind == K_UP) {
        if (s->nitems) s->selected = s->selected == 0 ? s->nitems - 1 : s->selected - 1;
    } else if (k->kind == K_DOWN) {
        if (s->nitems) s->selected = s->selected + 1 >= s->nitems ? 0 : s->selected + 1;
    } else if (k->kind == K_PGUP) { // getSelectPageIndex
        if (s->nitems) s->selected = s->selected >= 8 ? s->selected - 8 : 0;
    } else if (k->kind == K_PGDN) {
        if (s->nitems) s->selected = MIN(s->selected + 8, s->nitems - 1);
    } else if (k->kind == K_ALT_UP || k->kind == K_ALT_DOWN) {
        if (s->all || !s->nitems) return;
        const char *id = s->items[s->selected];
        long cur = ss_index(s, id);
        long delta = k->kind == K_ALT_UP ? -1 : 1, to = cur + delta;
        if (cur >= 0 && to >= 0 && to < (long)s->enabled.len) {
            char *x = s->enabled.ptr[cur];
            s->enabled.ptr[cur] = s->enabled.ptr[to];
            s->enabled.ptr[to] = x;
            s->selected = (size_t)((long)s->selected + delta);
            scoped_changed(s);
        }
    } else if (k->kind == K_ENTER) {
        if (!s->nitems) return;
        char *id = strdup(s->items[s->selected]);
        ss_toggle(s, id);
        free(id);
        scoped_changed(s);
    } else if (k->kind == K_CTRL && (k->ch == 'a' || k->ch == 'x')) {
        bool q = s->search.value.len > 0;
        char **targets = q ? xmalloc((s->nitems + 1) * sizeof *targets) : NULL;
        for (size_t i = 0; q && i < s->nitems; i++) targets[i] = strdup(s->items[i]);
        size_t nt = q ? s->nitems : 0;
        if (k->ch == 'a') ss_enable_all(s, targets, nt);
        else ss_clear_all(s, targets, nt);
        for (size_t i = 0; i < nt; i++) free(targets[i]);
        free(targets);
        scoped_changed(s);
    } else if (k->kind == K_CTRL && k->ch == 'p') {
        if (!s->nitems) return;
        const CodeUiModel *m = ss_model(s, s->items[s->selected]);
        if (!m) return;
        char **pids = xmalloc((s->nmodels + 1) * sizeof *pids);
        size_t np = 0;
        bool every = true;
        for (size_t i = 0; i < s->nmodels; i++)
            if (!strcmp(s->models[i]->provider, m->provider)) {
                pids[np++] = s->ids[i];
                if (!ss_enabled(s, s->ids[i])) every = false;
            }
        if (every) ss_clear_all(s, pids, np);
        else ss_enable_all(s, pids, np);
        free(pids);
        scoped_changed(s);
    } else if (k->kind == K_CTRL && k->ch == 's') {
        // onPersist: every available model enabled is no setting at all
        bool all = s->all;
        if (!all && s->enabled.len == s->nmodels) {
            all = true;
            for (size_t i = 0; i < s->enabled.len; i++)
                if (!ss_model(s, s->enabled.ptr[i])) all = false;
        }
        save_enabled_models(s->enabled.ptr, s->enabled.len, all);
        for (size_t i = 0; i < g_patterns.len; i++) free(g_patterns.ptr[i]);
        g_patterns.len = 0;
        if (!all)
            for (size_t i = 0; i < s->enabled.len; i++) vec_push(&g_patterns, strdup(s->enabled.ptr[i]));
        G.show_status("Model selection saved to settings");
        s->dirty = false;
    } else if (k->kind == K_CTRL && k->ch == 'c') {
        if (s->search.value.len) {
            input_set(&s->search, "");
            ss_refresh(s);
        } else {
            scoped_free();
            dialog_close();
        }
    } else if (k->kind == K_ESC) {
        scoped_free();
        dialog_close();
    } else {
        input_key(&s->search, k);
        ss_refresh(s);
    }
}

static void cmd_scoped_models(void) {
    scoped_free();
    ScopedSel *s = xcalloc(1, sizeof *s);
    size_t n = available(NULL);
    s->models = xcalloc(n + 1, sizeof *s->models);
    s->ids = xcalloc(n + 1, sizeof *s->ids);
    available(s->models);
    s->nmodels = n;
    for (size_t i = 0; i < n; i++) s->ids[i] = full_id(s->models[i]);
    // the session's scope, else the settings' patterns resolved (with the ones matching nothing)
    if (g_scope.len) {
        s->all = false;
        for (size_t i = 0; i < g_scope.len; i++) vec_push(&s->enabled, full_id(g_scope.ptr[i].model));
    } else if (g_patterns.len) {
        ScopedVec r = {0};
        NameVec nm = {0};
        resolve_scope(g_patterns.ptr, g_patterns.len, &r, &nm);
        s->all = false;
        for (size_t i = 0; i < r.len; i++) vec_push(&s->enabled, full_id(r.ptr[i].model));
        for (size_t i = 0; i < nm.len; i++)
            if (ss_index(s, nm.ptr[i]) < 0) vec_push(&s->enabled, nm.ptr[i]);
            else free(nm.ptr[i]);
        vec_free(&r);
        vec_free(&nm);
    } else s->all = true;
    buf_adds(&s->search.value, "");
    ss_refresh(s);
    g_ss = s;
    dialog_close();
    D.kind = D_SCOPED;
    G.request_render();
}

// ---- the hooks ---------------------------------------------------------------------------------------------

void cmds_init(const CmdsUi *ui) {
    G = *ui;
    free(g_proc_cwd);
    g_proc_cwd = strdup(G.cfg->cwd);
    // the scope from settings.json enabledModels (main.ts: resolveModelScope at start)
    const JValue *em = json_get(G.cfg->settings, "enabledModels");
    for (size_t i = 0; json_is(em, J_ARRAY) && i < em->arr.len; i++)
        if (json_str(em->arr.items[i])) vec_push(&g_patterns, strdup(json_str(em->arr.items[i])));
    if (g_patterns.len) resolve_scope(g_patterns.ptr, g_patterns.len, &g_scope, NULL);
    g_scope_notice = g_scope.len > 0;
}

void cmds_dock_render(int width, Lines *out) {
    if (D.kind == D_SELECT) select_render(width, out);
    else if (D.kind == D_LOADER) loader_render(width, out);
    else if (D.kind == D_SCOPED && g_ss) scoped_render(width, out);
}

void cmds_dock_key(const Key *k) {
    if (D.kind == D_SELECT) select_key(k);
    else if (D.kind == D_LOADER) {
        if (k->kind == K_ESC || (k->kind == K_CTRL && k->ch == 'c')) {
            void (*abort)(void) = D.abort;
            if (abort) abort();
            else dialog_close();
        }
    } else if (D.kind == D_SCOPED) scoped_key(k);
    G.request_render();
}

static bool is_cmd(const char *text, const char *cmd) {
    size_t n = strlen(cmd);
    return !strncmp(text, cmd, n) && (text[n] == 0 || text[n] == ' ');
}

bool cmds_submit(const char *text) {
    if (!strcmp(text, "/scoped-models")) {
        cmd_scoped_models();
        return true;
    }
    if (is_cmd(text, "/export")) {
        if (!G.busy()) cmd_export(text);
        return true;
    }
    if (is_cmd(text, "/import")) {
        if (!G.busy()) cmd_import(text);
        return true;
    }
    if (!strcmp(text, "/share")) {
        if (!G.busy()) cmd_share();
        return true;
    }
    if (!strcmp(text, "/copy")) {
        cmd_copy();
        return true;
    }
    if (!strcmp(text, "/hotkeys")) {
        cmd_hotkeys();
        return true;
    }
    return false;
}
