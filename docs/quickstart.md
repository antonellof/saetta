# Quickstart

This page takes you from an installed `saetta` to a first task done by the built-in agent.

## Open the agent

Run it in the folder of the project you want to work on:

```sh
cd ~/src/my-project
saetta start code
```

On the first run, with no model downloaded, Saetta Code shows the model that fits this Mac and downloads it after you confirm. Then the model loads inside the agent's process, with a lightning loader on screen. There is no server and no HTTP between the agent and the model.

If a Saetta server is already running on this Mac, `saetta start code` joins it instead of loading a second copy of the model.

## Pick a model

By default `saetta start code` picks by memory: `coder` (Qwen3.8-Flash-Next) with 96 GB or more, `27b-turbo` with 36 GB, `27b-4bit` with 32 GB. To use another model, download its pack and name it:

```sh
saetta download inkling                 # Inkling 2 Mini, 17 GB, from 32 GB of memory
saetta start code --pack inkling
saetta download 27b-qwopus              # Qwopus, a Qwen3.8-27B coding fine-tune, 16 GB
saetta start code --pack 27b-qwopus
```

Inside the agent, `/model` lists the packs on this Mac and the ones you can download (with their size and the free disk), downloads the one you pick and switches to it in the running session. Inkling 2 Mini is Thinking Machines' 26B mixture of experts with 3.4B active per token; it is not picked automatically. See [Choosing a model](choosing-a-model.md).

## Every model in one place

```sh
# Qwen3.8-Flash-Next (96 GB+) and Qwen3.8-27B (32 GB+)
saetta download coder && saetta start code --pack coder
saetta download 27b-4bit && saetta start code --pack 27b-4bit

# Qwen3.8-27B coding fine-tunes (converted from their GGUF on download, with a draft head)
saetta download 27b-turbo && saetta start code --pack 27b-turbo
saetta download 27b-qwopus && saetta start code --pack 27b-qwopus

# LoRA adapters on the 27B 8-bit, in the agent's own process
saetta download 27b-8bit && saetta download 27b-pi
saetta start code --pack 27b-8bit --lora-modules qwen3.8-pi=27b-pi

# Inkling 2 Mini (32 GB+), optionally with its DSpark drafter
saetta download inkling && saetta start code --pack inkling
saetta download inkling-dspark && SAETTA_DSPARK=1 saetta start code --pack inkling

# Any Qwen3.8-27B GGUF, imported into a pack (the folder holds the model's config.json and tokenizer)
saetta import my-finetune.gguf ~/Models/my-finetune && saetta start code --pack ~/Models/my-finetune

# A remote OpenAI-compatible model: add it in /model, or in ~/.pi/agent/models.json, then
saetta code --model myprovider/my-model
```

See [Choosing a model](choosing-a-model.md), [LoRA adapters](lora.md), [GGUF import](gguf-import.md) and [Models and providers](models-and-providers.md).

## Trust the project

When the folder has project resources (a `.pi` folder with settings, skills or MCP servers), the agent asks whether to trust it before loading them. Context files such as `AGENTS.md` load either way. See [Context and skills](context-and-skills.md).

## Ask for something

Type a request and press Enter:

```text
the parser test in tests/test_parser.py fails, find out why and fix it
```

The agent reads files, runs commands and edits code with its tools (`read`, `bash`, `edit`, `write`). Edits show as diffs and bash output appears as it runs. Escape stops a turn. While the agent works, Enter queues a message that steers the current turn and Option+Enter queues a follow-up for after it.

A few things to try next:

- `@` followed by part of a file name completes paths, to point the agent at a file.
- `/` lists the commands, such as `/model`, `/tree` and `/compact`. See [Interactive mode](interactive-mode.md).
- `!ls` runs a shell command yourself and puts its output in the context; `!!ls` keeps it out of the context.
- Shift+Tab cycles the thinking level.

## Come back later

Every conversation is saved as a session file. Continue the last one in this folder with:

```sh
saetta code -c
```

or pick any earlier one with `/resume`. See [Sessions](sessions.md).

## Without the terminal UI

For a single answer in a script:

```sh
saetta code -p "list the TODO comments in src/"
```

See [Print, JSON and RPC modes](print-json-rpc.md).
