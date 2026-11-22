// The interactive mode's side of skills, trust and the changelog (code_skills.c, code_trust.c,
// code_changelog.c), and the few calls into code_ui.c they use.
#ifndef SAETTA_CODE_RES_UI_H
#define SAETTA_CODE_RES_UI_H

#include "code_res.h"
#include "code_tui.h"

// ---- provided by code_ui.c ---------------------------------------------------------------------

// A selector drawn in place of the editor (pi's showSelector): render draws it, key gets every
// key while it is open, destroy frees ctx when it closes.
typedef struct {
    void (*render)(void *ctx, int width, Lines *out);
    void (*key)(void *ctx, const Key *k);
    void (*destroy)(void *ctx);
    void *ctx;
} CodeUiPanel;
void code_ui_panel_open(CodeUiPanel p);
void code_ui_panel_close(void);
void code_ui_status(const char *msg);  // showStatus
void code_ui_warning(const char *msg); // showWarning
void code_ui_chat_add(Comp *c);        // chatContainer.addChild
size_t code_ui_chat_len(void);         // chatContainer.children.length
bool code_ui_expanded(void);           // toolOutputExpanded
bool code_ui_busy(void);               // a run in flight
void code_ui_rebuild_chat(void);       // rebuildChatFromMessages
void code_ui_refresh_commands(void);   // setupAutocompleteProvider
const CodeUiConfig *code_ui_config(void);

// ---- provided by the feature files ------------------------------------------------------------

// The loaded resources under the header ([Context], [Skills], [Skill warnings]), collapsed or
// expanded (ctrl+o).
void code_res_render(int width, bool expanded, Lines *out);
// A user message holding a skill block (parseSkillBlock): its [skill] component, and the user's
// own text after it in *user_message (empty when none); NULL for any other text.
Comp *code_skill_block_comp(Str text, Str *user_message);
// The "/" list: the built-in commands, then skill:<name> for each skill (enableSkillCommands).
const SlashCmd *code_skill_commands(const SlashCmd *builtin, size_t n, size_t *out_n);
void code_reload_command(void);    // /reload
void code_trust_command(void);     // /trust
void code_changelog_command(void); // /changelog
// At startup, after the transcript: the changelog notice (a new session only), then the project
// trust warning.
void code_startup_notices(bool resumed);
// renderProjectTrustWarningIfNeeded
void code_trust_warning(void);
// Whether cwd's project is trusted: the current runtime's decision for its folder, else the one
// made for cwd this session, else what the store and settings say (never a prompt). For what the
// UI itself runs in a folder (the footer's git).
bool code_trust_folder(const char *cwd);
// maybeSaveImplicitProjectTrustAfterReload: true when it saved the trust.
bool code_trust_after_reload(const CodeEnv *env);

#endif
