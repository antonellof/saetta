# Saetta Code

Saetta Code is the coding agent inside the `saetta` binary. It is written in C and runs the model in its own process, with no server and no HTTP between the agent and the model. The binary is 2.3 MB and needs neither Node.js nor Python.

```sh
saetta start code                       # the model inside the agent (or the running server, if one is up)
saetta code --pack coder                # the same with a chosen pack; Escape stops a turn
saetta code -c                          # continue this folder's last session (pi's too)
saetta code -p "fix the failing test"   # print mode, as pi -p
saetta code --mode json -p "..."        # pi's JSON event stream
saetta code --mode rpc                  # pi's rpc protocol: JSON commands on stdin, events on stdout
```

## A port of pi

Saetta Code is a port of the [pi](https://pi.dev) coding agent 1.5.2 (MIT). It has pi's tools, its system prompt, its terminal UI and its file formats:

- Tools: `read`, `bash`, `edit` and `write` by default; `grep`, `find` and `ls` with `--tools`.
- Context files (`AGENTS.md`, `CLAUDE.md`), `SYSTEM.md` and `APPEND_SYSTEM.md`, skills and `/skill:name`.
- Session files, in the same folder and format as pi's, so either agent continues a session the other started, compacted ones included.
- `settings.json`, `models.json`, `trust.json` and `mcp.json`, read and written as pi does.
- Print, JSON and rpc modes.

Its requests and session files match pi's byte for byte on 54 recorded print and JSON scenarios (tool errors, edits, truncation, sessions, compaction, retries, skills), apart from its own system prompt preamble. Its interactive screens are compared with pi's on 61 recorded scenarios, and its rpc mode with pi's, driven by pi's own rpc client, on 12.

pi's TypeScript extensions and packages are not supported, and there is no image input.

## The configuration folder

Saetta Code keeps its configuration and sessions in pi's agent folder: `$PI_CODING_AGENT_DIR`, else `~/.pi/agent`. Run as `saetta code`, it shares that folder with a pi you run yourself.

`saetta start code` and `saetta start pi` set `PI_CODING_AGENT_DIR` to `~/.saetta/pi`, so the two agents started through Saetta share one setup there and leave `~/.pi/agent` alone. In that folder they update only their own entries (the `saetta` provider in `models.json`, `defaultProvider` and `defaultModel` in `settings.json`) and keep everything else.

| file | what it holds |
|---|---|
| `settings.json` | default model and thinking level, theme, compaction, retry, skills and other settings ([Themes and settings](themes-and-settings.md)) |
| `models.json` | OpenAI-compatible providers and their models ([Models and providers](models-and-providers.md)) |
| `trust.json` | which project folders are trusted ([Context and skills](context-and-skills.md)) |
| `mcp.json`, `mcp-auth.json` | MCP servers and their OAuth sign-ins ([MCP](mcp.md)) |
| `sessions/` | one folder per working directory, one JSONL file per session ([Sessions](sessions.md)) |
| `skills/`, `SYSTEM.md`, `APPEND_SYSTEM.md`, `AGENTS.md` | global skills and prompt files |

The agent writes its own log to `~/.saetta/code.log`.

## Where the model runs

By default the model runs inside the agent. It can also use a running Saetta server, or any OpenAI-compatible chat completions server listed in `models.json`; `/model` adds one. LoRA adapters need the server: `saetta start code --lora name=adapter` starts one. Only one model runs on the Mac at a time: further Saetta Code sessions use the model the first one loaded ([Parallel agents](parallel-agents.md)), while `saetta start` for another agent asks you to close it first, or to start a server that both use.

## Options

```text
saetta code [options] [prompt ...]

  -p, --print              answer the prompt and exit (text output)
  --mode json              print pi's JSON event stream
  --mode rpc               pi's rpc protocol: JSON commands on stdin, responses and events on stdout
  -c, --continue           continue the most recent session in this folder
  --session FILE           continue this session file
  --no-session             do not save the session
  --provider P --model M   the model (M may be provider/id and end in :thinking)
  --thinking LEVEL         off, minimal, low, medium, high, xhigh, max
  --tools a,b,c | --no-tools   the tool set
  --no-mcp                 no MCP servers (pi's mcp.json) and no MCP tools
  --subagents              offer the subagent tool: tasks handed to child agents that run in
                           parallel (also "subagents": true in settings.json)
  --system-prompt T, --append-system-prompt T   text or a file
  --skill PATH             load a skill file or folder (can be repeated)
  --no-skills, -ns         no skill discovery (--skill paths still load)
  --approve, -a            trust this project for the session (--no-approve, -na: do not)
  --pack P                 run the model in this process (a download name such as coder, or a
                           folder): no server, no HTTP; the requests are the same
  --ctx N                  with --pack: the context in tokens (default 204800, as saetta start;
                           less when this Mac's memory holds less)
```

Input piped into `saetta code` is read as pi reads it: it is joined to the prompt and makes the run a print run.

```sh
git diff | saetta code "review this change"
```

The agent checks for a new Saetta release at start. `SAETTA_SKIP_VERSION_CHECK=1` or `PI_OFFLINE=1` turns the check off.
