// The interactive mode's pieces that a feature kept in its own file uses (uix_*, at the end of
// code_ui.c), and the hooks code_ui.c calls in the session commands (sess_ui_*,
// code_sessions_ui.c).
#ifndef SAETTA_CODE_UI_EXT_H
#define SAETTA_CODE_UI_EXT_H

#include "code_tui.h"

// ---- code_ui.c --------------------------------------------------------------------------------

const CodeUiConfig *uix_cfg(void);
const CodeUiOps *uix_ops(void);
CodeUiHost *uix_host(void);
Comp *uix_chat(void);     // the transcript container
Editor *uix_editor(void);
bool uix_busy(void);      // a run (or a compaction) is going on the agent thread
bool uix_tools_expanded(void);
void uix_show_status(const char *msg);  // showStatus
void uix_show_error(const char *msg);   // showError ("Error: ...")
void uix_show_warning(const char *msg); // showWarning ("Warning: ...")
void uix_render_history(void);          // the transcript drawn again from the session
void uix_request_render(void);
// A selector in place of the editor (showSelector): drawn by render, keys to key; data is freed
// with free_data when it closes.
void uix_open_selector(void (*render)(int width, Lines *out), void (*key)(const Key *k), void (*free_data)(void *), void *data);
void *uix_selector_data(void);
void uix_close_selector(void);
// The thinking level shown (footer, editor border) after the session changed.
void uix_show_thinking(const char *level);
// Starts /compact on the agent thread (busy until its ui_idle).
void uix_start_compact(const char *instructions);
// The working status in the editor's top border (a status indicator), NULL to clear it.
void uix_working(const char *message);
void uix_abort(void); // the run in flight stopped
// Ctrl+X's selection first: true when a transcript selection (not copied on select) was copied.
bool uix_copy_selection(void);

// ---- code_sessions_ui.c -------------------------------------------------------------------------

// The session commands: true when cmd (the text after "/") is one of them and was handled.
bool sess_ui_command(const char *cmd);
// Every agent event, before the interactive mode handles it.
void sess_ui_event(const JValue *ev);
// A compaction is running (compaction_start seen, compaction_end not yet).
bool sess_ui_compacting(void);
// At ui_idle: true when a session command waiting for the run to stop took over (the rest of
// the idle handling is skipped).
bool sess_ui_idle(void);
// renderInitialMessages: the messages to draw (pi's context entries as messages, the compaction
// summary among them), the compaction summary's view, and the end of the drawing (the "Session
// compacted" status, the footer's totals).
const JValue *sess_ui_history(Arena *a);
void sess_ui_add_compaction(const JValue *m);
void sess_ui_history_done(void);
// Ctrl+O: the compaction summaries expanded or collapsed with the tool output.
void sess_ui_set_expanded(bool expanded);
// The footer (footer.ts getSessionStats): the totals over every session entry and whether the
// context use is unknown ("?"), from the sums over the drawn messages.
void sess_ui_footer(double *in, double *out, double *cache_read, double *cache_write, double *cost, bool *unknown);
// " • <name>" after the footer's folder when the session has a name.
void sess_ui_footer_name(Buf *out);
// The main loop's timer: ms until the next deadline (a selector status that expires), -1 none.
int sess_ui_tick(uint64_t now_ns);
// A new session (/new): the totals start again.
void sess_ui_reset(void);

#endif
