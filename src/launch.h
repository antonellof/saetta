// `saetta start <claude|pi|opencode> [--pack coder|full|<dir>] [--port N] [--ctx N] [--keep] [-- <agent args>]`
// Runs a coding agent against saetta. One server runs per machine (src/instance.h): a
// session joins the running one, whatever its port, or starts `saetta serve` (its own process
// group, so the agent's Ctrl-C does not reach it; log in ~/Library/Logs/saetta/), waits until
// it is ready, then runs the agent with its settings pointed at the server. A server started
// this way is stopped when the last session using it exits (--keep leaves it running); one
// started by hand with `saetta serve` is left alone. The user's own agent configuration is
// not edited:
//   claude    environment variables plus `--settings` with the same env block (it takes
//             precedence over ~/.claude/settings.json); Anthropic API at /v1/messages
//   pi        its own config folder ~/.saetta/pi (PI_CODING_AGENT_DIR), OpenAI API
//   opencode  its own config file ~/.saetta/opencode/opencode.json (OPENCODE_CONFIG)
#ifndef SAETTA_LAUNCH_H
#define SAETTA_LAUNCH_H

#include "base.h"

const char *launch_usage(void);
// args: the arguments after "start". Sets *code: the agent's exit code, 64 on a usage
// error, 127 when the agent is not installed, 1 on other failures it reports itself.
// Errors (printed by the caller as "saetta start: <err>"): parse errors of --port/--ctx,
// "NoHome", file errors.
Err launch_run(const char *const *args, size_t n, int *code);

// The official install command of an agent ("claude", "pi", "opencode"), NULL for others.
const char *launch_install_command(const char *agent);

#endif
