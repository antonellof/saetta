# Commands

`saetta help` lists every command. `<pack>` is a download name, such as `coder` or `27b-turbo`, or a pack folder.

```text
usage: saetta <command> [arguments]

  (no command)                 in a terminal: pick a model and a coding agent, then start (also: saetta auto)

Agents and server (<pack> is a download name, such as coder or 27b-turbo, or a folder):
  start <claude|pi|opencode|code> [P | --pack P] [--lora-modules name=adapter ...] [--port N] [--ctx N] [--keep] [-- agent args]
  code [-p] [-c] [--mode json] [prompt]   Saetta Code, the built-in coding agent (pi-compatible; saetta code --help)
  serve <pack> [--port N] [--lora-modules name=path ...] [--default-chat-template-kwargs JSON]
  monitor | status | stop      live view, state, stop the running server

Models:
  download [name] [--jobs N]   fetch a pack into ~/Models (SAETTA_MODELS); default: what fits this Mac
  import <file.gguf> <dir> [--like dir]   a Qwen3.8-27B GGUF into a pack
  info <pack>                  what the pack holds

Tools:
  generate <pack> <n> <token ids...>   n greedy tokens after the ids (SAETTA_LORA=adapter)
  tokenize <pack> <file>       token ids of a text file
  render <request.json>        the prompt a chat request renders to
  verify [golden-dir]          the engine against reference outputs (development)
  update | version | help
```

## Agents and server

| command | page |
|---|---|
| `saetta`, `saetta auto` | [Install](install.md#the-guided-start) |
| `saetta start <agent>` | [pi](pi.md#saetta-start), [Claude Code](claude-code.md), [opencode](opencode.md) |
| `saetta start code`, `saetta code` | [Saetta Code](saetta-code.md) |
| `saetta serve <pack>` | [HTTP API](http-api.md) |
| `saetta monitor`, `status`, `stop` | [monitor, status, stop](server.md) |

## Models

| command | page |
|---|---|
| `saetta download [name] [--jobs N]` | [Choosing a model](choosing-a-model.md#downloading) |
| `saetta import <file.gguf> <dir> [--like dir]` | [GGUF import](gguf-import.md) |
| `saetta info <pack>` | prints what a pack holds |

## Tools

| command | what it does |
|---|---|
| `saetta generate <pack> <n> <token ids...>` | n greedy tokens after the given ids; `SAETTA_LORA=<folder>` applies an adapter |
| `saetta tokenize <pack> <file>` | the token ids of a text file |
| `saetta render <request.json>` | the prompt a chat completions request renders to |
| `saetta verify [golden-dir]` | the engine against reference outputs (development) |
| `saetta update` | install the latest release |
| `saetta version` | print the version |

## saetta code

`saetta code --help` prints the agent's options; they are listed on [Saetta Code](saetta-code.md#options).
