# Saetta documentation

Saetta runs Qwen3.8 coding models on any Apple Silicon Mac with 32 GB of memory or more. It is one binary, `saetta`, with three parts:

- Saetta Code, a coding agent in C that runs the model in its own process. It is a port of [pi](https://pi.dev) 1.5.2 and shares pi's settings, sessions, skills and MCP servers.
- An inference engine in C and hand-written Metal, tuned for coding agents: long shared-prefix prompts (20K to 200K tokens), short answers, tool calls.
- A local server with OpenAI- and Anthropic-compatible APIs, which `saetta start` points Claude Code, pi or opencode at.

It is tuned and measured on an M5 Max with 128 GB, whose neural accelerators read prompts fastest. M1 to M4 run the same kernels on their GPU cores (not yet tested on those chips: reports welcome).

## Start here

```sh
curl -fsSL https://saetta.dev/install.sh | sh
saetta start code
```

The second command picks the model that fits this Mac, downloads it and opens the agent on it. [Install](install.md) has the details and [Quickstart](quickstart.md) walks through a first session.

## Where to go next

- [Choosing a model](choosing-a-model.md): which pack fits how much memory, and the published benchmark scores.
- [Saetta Code](saetta-code.md): the agent, its commands, sessions, skills and MCP servers.
- [Claude Code](claude-code.md), [pi](pi.md) and [opencode](opencode.md) on the local model.
- [HTTP API](http-api.md): endpoints for any other client.
- [Commands](commands.md) and [Environment variables](environment.md) for reference.

## Requirements

An Apple Silicon Mac (M1 or later) with 32 GB of memory or more, and macOS 26 or later. The model runs on the Mac: prompts and code stay local. The network is used to download the weights from Hugging Face, for the agent's update check at start, and for what you ask for (`/share`, a remote model, an MCP server you configure).

Saetta is open source under the [MIT license](../LICENSE). The models keep their own licenses.
