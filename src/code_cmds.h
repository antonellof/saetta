// Saetta Code's session and clipboard commands: /copy, /hotkeys, /export, /import, /share and
// /scoped-models, with the dialogs they open in place of the editor (code_cmds.c), and the HTML and
// JSONL exports (code_export.c).
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at v1.5.2,
// b4afddd); the files each function ports are named at the function.
#ifndef SAETTA_CODE_CMDS_H
#define SAETTA_CODE_CMDS_H

#include "code_tui.h"

// What the commands need from the interactive mode (code_ui.c fills it once, at start).
typedef struct {
    const CodeUiConfig *cfg;
    const CodeUiOps *ops;
    CodeUiHost *host;
    void (*show_status)(const char *msg);
    void (*show_error)(const char *msg);
    void (*chat_add)(Comp *c);
    void (*request_render)(void);
    // the transcript drawn again from the session's messages (a session switch)
    void (*render_history)(void);
    bool (*busy)(void);
    // the model in use, and a switch to another (pi's cycle: no persist)
    const CodeUiModel *(*model)(void);
    void (*choose_model)(const CodeUiModel *m);
    const char *(*thinking)(void);
    void (*set_thinking)(const char *level);
    // the theme on screen: its name ("saetta", "system", "dark", "light"), the terminal's colors
    // (NULL: none reported) and the appearance the saetta theme follows (1 dark, 2 light, 0 terminal)
    const char *(*theme_name)(void);
    const TermColors *(*term_colors)(void);
    int (*appearance)(void);
    void (*toggle_tools)(void); // app.tools.expand in a dialog
    void (*set_cwd)(const char *cwd); // the session now runs there (/import): the footer follows
} CmdsUi;

void cmds_init(const CmdsUi *ui);
// copyToClipboard: pbcopy, then OSC 52 in a remote session. NULL, or the error message.
const char *cmds_copy_text(Str text);
// handleCopyCommand: the last agent message to the clipboard; flash (NULL: the status line) says so.
void cmds_copy(void (*flash)(const char *msg));
// A submitted line: true when it was one of these commands (handled).
bool cmds_submit(const char *text);
// A dialog of these commands in place of the editor: it takes the keys and draws there.
bool cmds_dock_active(void);
void cmds_dock_render(int width, Lines *out);
void cmds_dock_key(const Key *k);
// Wake-ups while a dialog waits on a child (/share): true when the screen changed.
bool cmds_tick(void);
// app.model.cycleForward / Backward over the scoped models (/scoped-models, settings
// enabledModels): false when no scope is set (the caller cycles every model).
bool cmds_cycle_model(int dir);
// Below the header: the model scope notice when the session started with one (settings
// enabledModels), then a blank line.
void cmds_header_notice(int width, Lines *out);

// ---- exports (code_export.c; pi: core/export-html/index.ts, core/session-export.ts) ------------

// The session as the exports read it.
typedef struct {
    const JValue *entries;    // every entry, the header first
    const char *leaf;         // the current leaf id, or NULL
    const char *session_file; // NULL for an in-memory session
    const char *system_prompt; // the current system prompt text, or NULL
    const JValue *tools;       // [{name, description, parameters}], or NULL
} CmdsSession;

// The theme an export is drawn in.
typedef struct {
    const char *name;
    const TermColors *colors; // NULL: none reported
    int appearance;
} CmdsTheme;

// exportSessionToHtml: the page at out_path (NULL: pi-session-<session file name>.html in cwd);
// *written gets the path written (malloc'd). A failure is a message (malloc'd) in *err.
bool cmds_export_html(const CmdsSession *s, const CmdsTheme *t, const char *out_path, const char *cwd, char **written,
                      char **err);
// exportSessionToJsonl: the header and the current branch, at out_path resolved against cwd.
bool cmds_export_jsonl(const CmdsSession *s, const char *out_path, const char *cwd, char **written, char **err);

// resolvePath (utils/paths.ts): ~ expanded, relative to base, "." and ".." resolved (malloc'd).
char *cmds_resolve_path(const char *input, const char *base, const char *home);

#endif
