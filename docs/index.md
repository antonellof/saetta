# Saetta documentation

Saetta Code is a local coding agent for Apple Silicon Macs with the model built in. It is a C port of [pi](https://pi.dev) 1.5.2 that runs Qwen3.8 and Inkling 2 Mini in its own process, on Saetta's inference engine (C and hand-written Metal). There is no Python, no Node and no server between the agent and the model. It shares pi's settings, sessions, skills and MCP servers.

The same engine also runs as a local server with OpenAI- and Anthropic-compatible APIs, which `saetta start` points Claude Code, pi or opencode at.

## Start here

```sh
curl -fsSL https://saetta.dev/install.sh | sh
saetta start code
```

The second command picks the model that fits this Mac, downloads it and opens the agent on it. `saetta start code --pack inkling` opens it on Inkling 2 Mini instead, and `/model` inside the agent switches packs. [Install](install.md) has the details and [Quickstart](quickstart.md) walks through a first session.

## Where to go next

- [Choosing a model](choosing-a-model.md): which pack fits how much memory, and the published benchmark scores.
- [Saetta Code](saetta-code.md): the agent, its commands, sessions, skills and MCP servers.
- [LoRA adapters](lora.md): adapters on Qwen3.8-27B, in the agent or the server.
- [Claude Code](claude-code.md), [pi](pi.md) and [opencode](opencode.md) on the local model.
- [HTTP API](http-api.md): endpoints for any other client.
- [Commands](commands.md) and [Environment variables](environment.md) for reference.

## Requirements

An Apple Silicon Mac (M1 or later) with 32 GB of memory or more, and macOS 26 or later. Saetta is tuned and measured on an M5 Max with 128 GB, whose neural accelerators read prompts fastest; M1 to M4 run kernels of their own on their GPU cores (not yet tested on those chips: reports welcome).

The model runs on the Mac: prompts and code stay local. The network is used to download the weights from Hugging Face, for the agent's update check at start, and for what you ask for (`/share`, a remote model, an MCP server you configure).

Saetta is open source under the [MIT license](../LICENSE). The models keep their own licenses.
