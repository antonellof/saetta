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

## Try Inkling 2 Mini

Inkling 2 Mini is Thinking Machines' 26B mixture of experts with 3.4B active per token. It runs from 32 GB of memory and is not picked automatically, so download it by name and open the agent on it:

```sh
curl -fsSL https://saetta.dev/install.sh | sh     # or: saetta update
saetta download inkling                           # Inkling 2 Mini, 17 GB
saetta start code --pack inkling
```

Inside the agent, `/model` switches between Inkling and the Qwen packs you have downloaded. See [Choosing a model](choosing-a-model.md).

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

## Choosing the model

`saetta start code --pack 27b-turbo` picks a pack by name. `/model` switches packs in the running agent and can download one. See [Choosing a model](choosing-a-model.md).
