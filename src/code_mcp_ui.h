// Saetta Code's /mcp manager and the views of MCP tool calls (code_mcp_ui.c), for the interactive
// mode (code_ui.c). Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner;
// github.com/earendil-works/pi at v1.5.2, b4afddd): extensions/mcp/ui.ts, index.ts, tools.ts.
#ifndef SAETTA_CODE_MCP_UI_H
#define SAETTA_CODE_MCP_UI_H

#include "code_mcp.h"
#include "code_tui.h"

// /mcp with its arguments (the text after "/mcp"): opens the manager, or runs a subcommand.
void code_mcp_ui_command(const char *args);
// Whether the manager is open (it replaces the editor and takes the keys).
bool code_mcp_ui_open(void);
void code_mcp_ui_render(int width, Lines *out);
void code_mcp_ui_key(const Key *k);
// /mcp argument completions (login, logout, reconnect).
bool code_mcp_ui_arg_items(SelectList *all, Lines *search);
// The view of a call to an MCP tool (createMcpToolRenderers): the call line and the result, for
// the tool box (width: its content width). text: the result's text output (getTextOutput).
void code_mcp_ui_render_tool(const char *name, const JValue *args, const JValue *result, Str text, bool has_result,
                             bool is_error, bool expanded, int width, Lines *out);

#endif
