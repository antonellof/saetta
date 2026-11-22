// Saetta Code's terminal UI: a port of pi's TUI (packages/tui) and of its interactive mode
// (packages/coding-agent/src/modes/interactive), so `saetta code` draws the screens pi 1.5.2 draws.
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at v1.5.2,
// b4afddd).
//
// Components render to lines of text with ANSI styles, as pi's do (Component.render(width)); the
// renderer composes them into a full screen frame on the alternate screen (pi's TuiAltScreen:
// a transcript that follows its end, and a dock with the editor and the footer below it), and
// writes the rows that changed.
#ifndef SAETTA_CODE_TUI_H
#define SAETTA_CODE_TUI_H

#include "base.h"
#include "code_tui_theme.h"
#include "json.h"

// ---- child processes (code_tui_util.c) -------------------------------------------------------------

#include <spawn.h>
#include <sys/types.h>

// pipe(2) with both ends close-on-exec.
int tui_pipe(int fds[2]);
// posix_spawnp with every signal back to its default (the UI ignores SIGINT; a child must not), an
// empty mask, only the descriptors fa names (close-on-exec by default; inherit_stdio keeps 0..2,
// the terminal), and in its own process group when group. Returns 0 or the errno.
int tui_spawn(pid_t *pid, char *const argv[], posix_spawn_file_actions_t *fa, bool inherit_stdio, bool group);
// SIGTERM to the child (its group), at most ms waited, then SIGKILL; reaped.
bool tui_stop_child(pid_t pid, bool group, int ms, int *status);
// argv's stdout into out (stdin fed from in), in cwd, at most ms: false if it failed or timed out
// (the child is then stopped).
bool tui_run(char *const argv[], const char *cwd, Str in, Buf *out, int ms);

// ---- lines (code_tui_util.c) ------------------------------------------------------------------

// A component's output: rows of text with ANSI escapes, each malloc'd.
typedef struct {
    char **v;
    size_t n, cap;
} Lines;
void lines_push(Lines *l, char *owned);
void lines_add(Lines *l, Str s);   // a copy
void lines_addz(Lines *l, const char *s);
void lines_addb(Lines *l, Buf *b); // takes the buffer's text (b is emptied)
void lines_append(Lines *dst, Lines *src); // moves every line of src
void lines_free(Lines *l);

// ---- text measurement and wrapping (pi-tui utils.ts) ------------------------------------------

// Length of the escape sequence (CSI, OSC, APC) at s[i], or 0 (ansiCodeLength).
size_t tui_ansi_len(Str s, size_t i);
// One code point at s[*i] (U+FFFD for an ill-formed byte); advances *i.
uint32_t tui_cp(Str s, size_t *i);
// Terminal cells of a code point: 0 for marks and format characters, 2 for wide ones.
int tui_cp_width(uint32_t cp);
// Byte length of the grapheme at s[i] (a code point and the marks after it).
size_t tui_grapheme(Str s, size_t i);
// Visible width in cells, escapes skipped, a tab counting 3 (visibleWidth).
int tui_width(Str s);
// wrapTextWithAnsi: word wrap to width, styles carried across the breaks.
void tui_wrap(Str text, int width, Lines *out);
// truncateToWidth: text cut to width with the ellipsis, optionally padded with spaces.
void tui_truncate(Buf *out, Str text, int width, const char *ellipsis, bool pad);
char *tui_truncatez(Str text, int width, const char *ellipsis, bool pad);
// sliceByColumn(line, start, len, strict).
void tui_slice(Buf *out, Str line, int start, int len, bool strict);
// Escapes and control characters dropped (stripTerminalSequences).
void tui_strip(Buf *out, Str s);
void tui_spaces(Buf *b, int n);
void tui_repeat(Buf *b, const char *s, int n);
// JavaScript's string length (UTF-16 code units), for pi's estimates.
size_t tui_js_len(Str s);
// A code point as UTF-8 (out holds 4 bytes); returns its length.
int tui_utf8(char *out, uint32_t cp);
// \s as JavaScript defines it, for one code point.
bool tui_is_space(uint32_t cp);

// The changed words of one edited line in inverse (pi's renderIntraLineDiff over jsdiff's
// diffWords; code_ui_diff.c).
void tui_intra_line(Str old_line, Str new_line, Buf *removed, Buf *added);

// ---- theme (pi's "system" theme as a terminal that reports no colors gets it) --------------------

typedef enum {
    T_ACCENT, T_BORDER, T_BORDER_ACCENT, T_BORDER_MUTED, T_SUCCESS, T_ERROR, T_WARNING, T_MUTED, T_DIM, T_TEXT,
    T_THINKING_TEXT, T_USER_TEXT, T_TOOL_TITLE, T_TOOL_OUTPUT, T_MD_HEADING, T_MD_LINK, T_MD_LINK_URL, T_MD_CODE,
    T_MD_CODE_BLOCK, T_MD_CODE_BLOCK_BORDER, T_MD_QUOTE, T_MD_QUOTE_BORDER, T_MD_HR, T_MD_LIST_BULLET,
    T_DIFF_ADDED, T_DIFF_REMOVED, T_DIFF_CONTEXT, T_SYN_COMMENT, T_SYN_KEYWORD, T_SYN_FUNCTION, T_SYN_VARIABLE,
    T_SYN_STRING, T_SYN_NUMBER, T_SYN_TYPE, T_SYN_OPERATOR, T_SYN_PUNCTUATION, T_THINK_OFF, T_THINK_MINIMAL,
    T_THINK_LOW, T_THINK_MEDIUM, T_THINK_HIGH, T_THINK_XHIGH, T_THINK_MAX, T_BASH_MODE, T_CUSTOM_LABEL,
    T_CUSTOM_TEXT, T_SCROLL_TRACK, T_SCROLL_THUMB, T_SEARCH_TEXT,
    T_COUNT
} ThemeColor;

// theme.fg(color, text) appended to b.
void th_fg(Buf *b, ThemeColor c, Str text);
void th_fgz(Buf *b, ThemeColor c, const char *text);
// The opening sequence of a color (getFgAnsi).
const char *th_open(ThemeColor c);
void th_bold(Buf *b, Str text);
void th_italic(Buf *b, Str text);
void th_underline(Buf *b, Str text);
void th_inverse(Buf *b, Str text);
void th_strike(Buf *b, Str text);
// A panel background (theme.bg), and a line padded to width on it (applyBackgroundToLine).
void th_bg(Buf *b, ThemeBgToken t, Str text);
void th_bg_line(Buf *out, ThemeBgToken t, Str line, int width);
// The theme for the colors the terminal reported (NULL: none); pending: grayscale until they come.
void theme_set(const TermColors *tc, bool pending);
// A named theme ("saetta", "system", "dark", "light"); appearance for saetta: 1 dark, 2 light, 0
// from the terminal's background. False for an unknown name (nothing changes).
bool theme_use(const char *name, const TermColors *tc, int appearance);
// What the terminal reported: 0 no colors, 1 a dark background, 2 a light one.
int th_appearance(void);
// The editor border color for a thinking level (getThinkingBorderColor).
ThemeColor th_thinking_color(const char *level);

// ---- components (pi-tui Component) --------------------------------------------------------------

typedef struct Comp Comp;
typedef struct {
    void (*render)(Comp *c, int width, Lines *out);
    void (*destroy)(Comp *c); // frees c
    bool live; // renders from state kept elsewhere: never cached
} CompVt;
// A component keeps its rendered lines until it changes (comp_invalidate) or the width does, as
// pi's components cache theirs.
struct Comp {
    const CompVt *vt;
    Lines cache;
    int cache_w;
    bool cached;
};
void comp_render(Comp *c, int width, Lines *out); // appends copies of the lines
const Lines *comp_lines(Comp *c, int width);      // the cached lines (valid until the next change)
void comp_invalidate(Comp *c);
void comp_free(Comp *c);

// Text(text, paddingX, paddingY): wrapped, padded to the width.
Comp *comp_text(Str text, int pad_x, int pad_y);
void comp_text_set(Comp *c, Str text);
// TruncatedText(text, paddingX, 0): one line, cut to the width.
Comp *comp_truncated(Str text, int pad_x);
// Spacer(n)
Comp *comp_spacer(int n);
// Container: children in order.
typedef VEC(Comp *) CompVec;
Comp *comp_container(void);
CompVec *comp_children(Comp *c);
void comp_add(Comp *container, Comp *child);
void comp_clear(Comp *container); // frees the children
// A component that renders its own lines through a callback (for views kept elsewhere).
Comp *comp_fn(void (*render)(void *ctx, int width, Lines *out), void *ctx);

// ---- markdown (code_tui_md.c; pi-tui components/markdown.ts) -----------------------------------

typedef struct {
    // the default text style (DefaultTextStyle): a color, italic
    bool has_color;
    ThemeColor color;
    bool italic;
    bool user; // a user message: list markers and backslash escapes kept as written
    bool has_bg; // every line on this panel background (bgColor)
    ThemeBgToken bg;
} MdStyle;
// Markdown(text, paddingX, paddingY, theme, defaultStyle)
void md_render(Str text, int pad_x, int pad_y, const MdStyle *style, int width, Lines *out);
// getCapabilities().hyperlinks: the terminal shows OSC 8 hyperlinks (PI_HYPERLINKS, then the terminal).
bool tui_hyperlinks(void);
Comp *comp_markdown(Str text, int pad_x, int pad_y, const MdStyle *style);
// highlight(code, lang): one styled line per source line (utils/syntax-highlight.ts).
void md_highlight(Str code, Str lang, Lines *out);
bool md_lang_supported(Str lang);

// ---- select list (pi-tui components/select-list.ts) ----------------------------------------------

typedef struct {
    char *value, *label, *description;
} SelItem;
typedef struct {
    SelItem *items;
    size_t n;
    size_t selected;
    int max_visible;
    int min_primary, max_primary; // the primary column's bounds
} SelectList;
void sel_init(SelectList *s, int max_visible, int min_primary, int max_primary);
void sel_add(SelectList *s, const char *value, const char *label, const char *description);
void sel_free(SelectList *s);
void sel_render(SelectList *s, int width, Lines *out);
const SelItem *sel_current(const SelectList *s);
void sel_up(SelectList *s);
void sel_down(SelectList *s);

// ---- keys (pi-tui keys.ts, stdin-buffer.ts, keybindings.ts) --------------------------------------

typedef enum {
    K_NONE, K_CHAR, K_ENTER, K_TAB, K_SHIFT_TAB, K_BACKSPACE, K_DELETE, K_ESC, K_UP, K_DOWN, K_LEFT, K_RIGHT,
    K_HOME, K_END, K_PGUP, K_PGDN, K_CTRL, K_ALT, K_PASTE, K_ALT_LEFT, K_ALT_RIGHT, K_CTRL_LEFT, K_CTRL_RIGHT,
    K_SHIFT_ENTER, K_ALT_ENTER, K_ALT_BACKSPACE, K_CTRL_BACKSPACE, K_ALT_UP, K_CTRL_UP, K_CTRL_DOWN, K_CTRL_HOME,
    K_CTRL_END,
    K_MOUSE,       // button (SGR code: 64/65 wheel, bit 5 motion), mx, my (0-based cells), release
    K_REPLY_KITTY, // the terminal's Kitty keyboard flags (ch)
    K_REPLY_DA,    // device attributes (the query's sentinel)
    K_REPLY_COLOR, // an OSC 10/11/4 color reply: ch 10, 11 or 100 + palette index; rgb in button, mx, my (release: unparsable)
    K_ALT_DOWN,
} KeyKind;
typedef struct {
    KeyKind kind;
    uint32_t ch;   // K_CHAR: the code point; K_CTRL / K_ALT: the letter ('a'..'z', ...)
    bool shift;    // K_CTRL: with shift (Kitty, modifyOtherKeys)
    bool kitty_text; // K_CHAR from an unmodified Kitty CSI u (its text copy is dropped)
    char *paste;   // K_PASTE: the pasted text (malloc'd, the caller frees it)
    int button, mx, my;
    bool release;
} Key;
// One key at p. Returns the bytes consumed, 0 for an incomplete sequence (unless flush: a lone
// ESC is then the Escape key, any other split sequence is dropped).
size_t key_parse(const char *p, size_t n, bool flush, Key *out);
// What stdin delivered, split into keys (pi's StdinBuffer): a bracketed paste of any size, split
// sequences and split UTF-8 kept until the rest arrives or their timeout (10 ms for a lone ESC,
// 100 over SSH, PI_TUI_ESC_TIMEOUT; 50 ms for other sequences; terminal replies wait for their end).
typedef struct {
    Buf buf;
    bool paste;
    Buf paste_buf;
    uint64_t pending_ns; // when an incomplete sequence was first seen
    uint32_t last_kitty_cp;
} KeyIn;
void keyin_feed(KeyIn *in, const char *p, size_t n);
bool keyin_next(KeyIn *in, uint64_t now_ns, Key *out);
int keyin_timeout_ms(const KeyIn *in, uint64_t now_ns); // -1: nothing pending
void keyin_free(KeyIn *in);

// ---- editor (code_tui_editor.c; pi-tui components/editor.ts with autocomplete.ts) ---------------

typedef struct Editor Editor;
typedef struct {
    const char *name, *description, *hint;
} SlashCmd;
Editor *editor_new(const char *cwd, const char *fd_path);
void editor_free(Editor *e);
void editor_set_commands(Editor *e, const SlashCmd *cmds, size_t n);
// Argument completions of a "/" command ("/thinking hi", "/model gpt"): fill all with every
// candidate and search with one text per candidate to match the argument against; return false
// when the command takes none.
typedef bool (*EditorArgFn)(void *ctx, const char *command, SelectList *all, Lines *search);
void editor_set_arg_completer(Editor *e, EditorArgFn fn, void *ctx);
void editor_set_border(Editor *e, ThemeColor c);
void editor_set_rows(Editor *e, int terminal_rows);
void editor_render(Editor *e, int width, Lines *out);
// Handles a key. Returns true when the text was submitted (*submitted gets it, malloc'd).
bool editor_key(Editor *e, const Key *k, char **submitted);
// Autocomplete timers (debounce) and the file lookup running off the UI thread: call on every
// wake-up; returns true if it changed the editor. editor_busy: a timer or a lookup is pending.
bool editor_tick(Editor *e);
bool editor_busy(const Editor *e);
// The pipe the lookup writes to when it finishes (wakes the UI's poll).
void editor_set_waker(Editor *e, int fd);
bool editor_autocomplete_open(const Editor *e);
Str editor_text(const Editor *e); // the text, lines joined with "\n" (valid until the next edit)
void editor_set_text(Editor *e, Str text);
bool editor_empty(const Editor *e);
void editor_add_history(Editor *e, Str text);
Comp *editor_comp(Editor *e);
// The working status in the top border (NULL hides it), and its spinner's next frame.
void editor_set_working(Editor *e, const char *message);
void editor_spin(Editor *e);

// ---- terminal and renderer (code_tui.c; pi-tui terminal.ts, tui-alt-screen.ts) -----------------

typedef struct {
    int cols, rows;
    Lines prev;    // the frame on screen
    bool started;
    bool kitty;    // the terminal confirmed the Kitty keyboard protocol
    int pending_da;
    // the color query (the theme): replies so far, and done once DA1 or every reply came
    bool color_query, color_done;
    uint64_t color_query_ns;
    TermColors colors;
    uint32_t palette_got;
} Screen;
// The alternate screen with the hardware cursor hidden, raw input, bracketed paste, mouse
// reporting, the Kitty keyboard protocol (modifyOtherKeys when the terminal has none). Crash
// signals and exit restore the terminal.
void screen_start(Screen *s);
void screen_stop(Screen *s);
// Ctrl+Z: the terminal restored, the process group stopped; returns once continued, the screen
// started again (the caller redraws it).
bool screen_suspend(Screen *s);
// A terminal reply (K_REPLY_KITTY, K_REPLY_DA): the keyboard protocol negotiation.
void screen_reply(Screen *s, const Key *k);
void screen_size(Screen *s);
// Writes frame (rows lines; shorter is padded with blank rows) where it differs from the last one.
// The frame's lines move into the screen (frame is left empty).
void screen_draw(Screen *s, Lines *frame, bool force);
// The loader's terminal: no echo, with the restore handlers (crash signals, Ctrl+C, SIGTERM, exit)
// in place; inline_rows > 0: it draws that many rows under the cursor, which a signal erases (CSI
// <rows> F, CSI J, the cursor shown); 0: the alternate screen. Stopped by screen_quiet_stop.
void screen_quiet_start(int inline_rows);
void screen_quiet_stop(void);
// The terminal back as it was (async-signal-safe), for signal handlers.
void screen_restore_signal_safe(void);

// ---- the loader (code_ui_splash.c) -------------------------------------------------------------------

// The turning bolt on the alternate screen with label under it, drawn from its own thread until
// splash_stop; NULL (nothing drawn) without a terminal, with NO_COLOR or "Reduce motion".
typedef struct Splash Splash;
Splash *splash_start(const char *label);
void splash_stop(Splash *s);

// ---- interactive mode (code_ui.c) ----------------------------------------------------------------

// The agent's side, run on its own thread: code_main.c provides these.
typedef struct CodeUiHost CodeUiHost;
typedef struct {
    const char *provider, *id, *name;
    bool reasoning;
    uint64_t context_window;
    // a model pack run in this process (provider "saetta", id the download name)
    bool pack;
    bool unavailable;   // a pack not downloaded yet (choosing it offers the download)
    bool too_big;       // its memory need exceeds this Mac's
    unsigned disk_gb, min_ram_gb;
    // a pack's name and quantization for the footer (code_cfg.h cfg_pack_label), the brief one
    // without "(mixed ...)"; NULL: the footer shows the id, as pi does
    const char *label, *label_brief;
} CodeUiModel;
typedef struct {
    // Runs one prompt to its end (on the agent thread); events arrive through code_ui_event.
    void (*prompt)(CodeUiHost *h, const char *text);
    // The calls below come from the UI thread while the agent is idle.
    void (*new_session)(CodeUiHost *h);
    void (*set_thinking)(CodeUiHost *h, const char *level);
    // Every session entry, the header first (pi's session file, for /tree), and the leaf's id.
    const JValue *(*entries)(CodeUiHost *h, const char **leaf);
    // The messages of the current branch, in order (for drawing the transcript).
    const JValue *(*messages)(CodeUiHost *h);
    // Moves the session's leaf (/tree): the next message is appended under it.
    void (*set_leaf)(CodeUiHost *h, const char *id);
    // Switches the model (/model); false when it cannot.
    bool (*set_model)(CodeUiHost *h, const char *provider, const char *id);
    // Abort the run in flight (any thread).
    void (*abort)(CodeUiHost *h);
    // A model this process cannot switch to in place (another pack, or away from the in-process
    // engine): recorded, and once the UI has ended the program starts again on it, the session
    // continued. Called from the UI thread with the agent idle.
    void (*restart)(CodeUiHost *h, const char *provider, const char *id);
    // /model's list again, after /model changed models.json (they stay valid to the end of the
    // run); NULL keeps the list the UI started with.
    const CodeUiModel *(*reload_models)(CodeUiHost *h, size_t *n);
    // For /export, /share and /import (code_cmds.c): the session file (NULL in memory), the folder
    // sessions are kept in, the working directory the session runs in, the current system prompt
    // and the tools' declarations ([{name, description, parameters}]), and a switch to another
    // session file run in cwd (its id and entries kept; non-zero when it cannot be opened).
    const char *(*session_file)(CodeUiHost *h);
    const char *(*session_dir)(CodeUiHost *h);
    const char *(*session_cwd)(CodeUiHost *h);
    const char *(*system_prompt)(CodeUiHost *h);
    const JValue *(*tool_decls)(CodeUiHost *h);
    Err (*open_session)(CodeUiHost *h, const char *path, const char *cwd);
    // The session commands (code_sessions_ui.c), from the UI thread with the agent idle: the
    // session; a fork of it made current (/fork before a user message, /clone at an entry: pi's
    // runtime fork, *selected the message's text); another session file made current (/resume),
    // in its own folder or (stay, its folder gone) in this one. NULL, or pi's error text.
    struct CodeSession *(*session)(CodeUiHost *h);
    const char *(*fork)(CodeUiHost *h, const char *entry_id, bool at, char **selected);
    const char *(*resume)(CodeUiHost *h, const char *path, bool stay);
    // /compact, run on the agent thread; compaction_start and compaction_end arrive as events.
    void (*compact)(CodeUiHost *h, const char *instructions);
} CodeUiOps;
typedef struct {
    const char *cwd, *home, *agent_dir;
    CodeUiModel model;
    const char *thinking;
    bool auto_compact;
    const char *fd_path; // fd for "@" completion, or NULL
    const CodeUiModel *models; // every model of models.json
    size_t nmodels;
    const char *default_provider, *default_model; // settings.json
    const JValue *settings; // settings.json, or NULL
    bool resumed; // a continued session: its messages are drawn at start
    const char *version; // saetta's version, for the header
    bool engine;          // the model runs in this process (a pack)
    const char *self_exe; // this program, for `saetta download` from /model
} CodeUiConfig;

// Runs pi's interactive mode until the user quits. initial: prompts given on the command line.
// "AgentBusy": the run did not stop after the abort; the caller exits the process at once.
Err code_ui_run(const CodeUiConfig *cfg, const CodeUiOps *ops, CodeUiHost *host, const char *const *initial, size_t ninitial);
// An agent event (pi's AgentSessionEvent as JSON, json mode's shape; message_update carries the
// partial message as "message"), from the agent thread.
void code_ui_event(const JValue *event);
// A streamed delta of the assistant message in flight (the per-token path: no allocation, lock or
// system call; the UI thread rebuilds the partial message). index: the content block; a: the text
// (for UI_S_TOOL_START the call id, b its name).
typedef enum { UI_S_TEXT_START, UI_S_TEXT, UI_S_THINK_START, UI_S_THINK, UI_S_TOOL_START, UI_S_TOOL_ARGS } CodeUiStreamKind;
void code_ui_stream(CodeUiStreamKind kind, int index, Str a, Str b);
// A model (malloc'd provider and id) or a thinking level the user chose while the run went on,
// for the agent thread to apply at its next turn boundary; false / NULL when none.
bool code_ui_take_model(char **provider, char **id);
char *code_ui_take_thinking(void);
// A message the user queued while the run goes on (malloc'd), or NULL: a steering message only
// when steering_only (between tool calls), else a follow-up too (when the run would end).
char *code_ui_take_queued(bool steering_only);

#endif
