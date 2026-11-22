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
#include "json.h"

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
// \s as JavaScript defines it, for one code point.
bool tui_is_space(uint32_t cp);

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
// The editor border color for a thinking level (getThinkingBorderColor).
ThemeColor th_thinking_color(const char *level);

// ---- components (pi-tui Component) --------------------------------------------------------------

typedef struct Comp Comp;
typedef struct {
    void (*render)(Comp *c, int width, Lines *out);
    void (*destroy)(Comp *c); // frees c
} CompVt;
struct Comp {
    const CompVt *vt;
};
void comp_render(Comp *c, int width, Lines *out);
void comp_free(Comp *c);

// Text(text, paddingX, paddingY): wrapped, padded to the width.
Comp *comp_text(Str text, int pad_x, int pad_y);
void comp_text_set(Comp *c, Str text);
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
} MdStyle;
// Markdown(text, paddingX, paddingY, theme, defaultStyle)
void md_render(Str text, int pad_x, int pad_y, const MdStyle *style, int width, Lines *out);
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
    K_SHIFT_ENTER, K_ALT_ENTER, K_ALT_BACKSPACE, K_CTRL_BACKSPACE,
} KeyKind;
typedef struct {
    KeyKind kind;
    uint32_t ch;   // K_CHAR: the code point; K_CTRL / K_ALT: the letter ('a'..'z', ...)
    char *paste;   // K_PASTE: the pasted text (malloc'd, the caller frees it)
} Key;
// Splits what stdin delivered into keys. Returns the bytes consumed; an incomplete sequence at
// the end stays (unless flush, which takes a lone ESC as the Escape key).
size_t key_parse(const char *p, size_t n, bool flush, Key *out);

// ---- editor (code_tui_editor.c; pi-tui components/editor.ts with autocomplete.ts) ---------------

typedef struct Editor Editor;
typedef struct {
    const char *name, *description, *hint;
} SlashCmd;
Editor *editor_new(const char *cwd, const char *fd_path);
void editor_free(Editor *e);
void editor_set_commands(Editor *e, const SlashCmd *cmds, size_t n);
void editor_set_border(Editor *e, ThemeColor c);
void editor_set_rows(Editor *e, int terminal_rows);
void editor_render(Editor *e, int width, Lines *out);
// Handles a key. Returns true when the text was submitted (*submitted gets it, malloc'd).
bool editor_key(Editor *e, const Key *k, char **submitted);
// Autocomplete timers (debounce): call when idle; returns true if it changed the editor.
bool editor_tick(Editor *e);
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
} Screen;
// The alternate screen with the hardware cursor hidden, raw input, bracketed paste.
void screen_start(Screen *s);
void screen_stop(Screen *s);
void screen_size(Screen *s);
// Writes frame (rows lines; shorter is padded with blank rows) where it differs from the last one.
void screen_draw(Screen *s, Lines *frame, bool force);

// ---- interactive mode (code_ui.c) ----------------------------------------------------------------

// The agent's side, run on its own thread: code_main.c provides these.
typedef struct CodeUiHost CodeUiHost;
typedef struct {
    // Runs one prompt to its end (on the agent thread); events arrive through code_ui_event.
    void (*prompt)(CodeUiHost *h, const char *text);
    // A new session (called from the UI thread while the agent is idle).
    void (*new_session)(CodeUiHost *h);
    // The thinking level changed (UI thread, agent idle).
    void (*set_thinking)(CodeUiHost *h, const char *level);
    // The session's entries on the current branch, as {"type":"message",...} objects (UI thread,
    // agent idle), for /tree.
    const JValue *(*entries)(CodeUiHost *h);
    // Abort the run in flight (any thread).
    void (*abort)(CodeUiHost *h);
} CodeUiOps;
typedef struct {
    const char *cwd, *home, *agent_dir;
    const char *provider, *model_id, *model_name, *thinking;
    bool reasoning;
    uint64_t context_window;
    bool auto_compact;
    const char *fd_path; // fd for "@" completion, or NULL
    const char *const *models; // "provider/id" of every model in models.json
    size_t nmodels;
} CodeUiConfig;

// Runs pi's interactive mode until the user quits. initial: prompts given on the command line.
Err code_ui_run(const CodeUiConfig *cfg, const CodeUiOps *ops, CodeUiHost *host, const char *const *initial, size_t ninitial);
// An agent event (pi's AgentSessionEvent as JSON, json mode's shape; message_update carries the
// partial message as "message"), from the agent thread.
void code_ui_event(const JValue *event);

#endif
