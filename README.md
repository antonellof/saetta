<p align="center">
  <img src="docs/assets/saetta-banner.png" alt="saetta" width="900">
</p>

<h3 align="center">Qwen3.8-Flash-Next at full speed on an M5 Max.</h3>
<p align="center">A single-model inference engine in C and hand-written Metal, tuned for coding agents.</p>

<p align="center">
  <a href="#install"><b>Install</b></a> · <a href="#models">Models</a> · <a href="#run-the-server">Server</a> ·
  <a href="#coding-agents">Coding agents</a> · <a href="#monitor">Monitor</a> · <a href="#performance">Performance</a> · <a href="#build-from-source">Build from source</a>
</p>

<p align="center">
  <a href="https://github.com/antonellof/saetta/releases/latest"><img src="https://img.shields.io/github/v/release/antonellof/saetta?label=download&color=8a3ffc" alt="Download saetta"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-f28c28" alt="MIT license"></a>
  <img src="https://img.shields.io/badge/Apple%20Silicon-M5%20Max%20128%20GB-111" alt="Apple M5 Max 128 GB">
</p>

<p align="center">
  <img src="docs/assets/demo.gif" width="900" alt="saetta start claude: Claude Code on saetta writes a snake game in Python, checks it, and the game runs">
  <br><sub><code>saetta start claude</code> on an M5 Max: Claude Code, running on the local model, writes a snake game and checks it. Real time, no cloud.</sub>
</p>

**One model, one machine, no compromises.** saetta does not try to run everything everywhere. It runs Qwen3.8-Flash-Next on an Apple M5 Max with 128 GB, and every kernel is written for that pair: long shared-prefix prompts (20K to 200K tokens), short answers and XML tool calls, served through an OpenAI-compatible API.

- **Fast prompt reading:** about 1,650 to 1,950 tokens/s from 4K to 70K tokens, on the GPU's neural accelerators.
- **Fast answers:** 63 to 82 tokens/s, with speculative decoding from the model's own MTP head, also when the agent samples; edits that repeat code go faster.
- **Agent turns that do not re-read:** a prompt that extends the previous one reads only its new tokens (a follow-up turn on a 16K conversation answers in 0.2 s), several agent sessions keep their own conversations, and a new session reuses the prefix it shares with the others.
- **Checked:** every layer is checked against a numpy reference of the model, and speculation keeps the model's own choices (greedy output matches decoding without it up to rounding at near ties).
- **One command to code:** `saetta start claude` (or `pi`, `opencode`) starts the server and your coding agent together. Claude Code connects through saetta's Anthropic-compatible API.
- **See it work:** `saetta monitor` shows the model's state, speed, GPU, memory and prompt-cache reuse live, like btop.
- **Small:** one binary that also downloads the model, no Python, no runtime dependencies beyond macOS.

## Install

```sh
curl -fsSL https://github.com/antonellof/saetta/releases/latest/download/install.sh | sh
```

Or with Homebrew:

```sh
brew install antonellof/saetta/saetta    # one line
```

or, to use the short name (Homebrew asks you to trust a third-party tap first):

```sh
brew tap antonellof/saetta && brew trust antonellof/saetta    # once
brew install saetta                                           # later: brew upgrade saetta
```

Installed with the script, `saetta update` installs a newer release over the binary.

The installer checks your Mac (Apple Silicon, macOS 27.2 or later), verifies the download's SHA-256 and puts the single `saetta` binary in `~/.local/bin` (`SAETTA_PREFIX` to change it, `SAETTA_VERSION=0.7.2` to pin a release). Then download a model and start coding:

```sh
saetta download coder           # 76 GB into ~/Models, resumable, every file hash-checked
saetta start claude             # or: saetta start pi, saetta start opencode
```

`HF_TOKEN` (or a token saved by `hf auth login`) raises Hugging Face's rate limits. Files: `saetta-macos-arm64.tar.gz` and its `.sha256` on the [releases page](https://github.com/antonellof/saetta/releases/latest).

| | |
|---|---|
| Machine | Apple M5 Max, 128 GB (the only target) |
| macOS | 27.2 or later |
| Disk | 76 GB for the Coder pack, 105 GB for the full pack |

## Models

Two MLX packs of Qwen3.8-Flash-Next run on the same binary. `saetta download coder|full|all` fetches them at pinned revisions into `~/Models/<org>/<repo>` (`SAETTA_MODELS` to change it), so other apps can share the files. It downloads four files at a time (`--jobs N`), resumes an interrupted run, and checks every file against the hash Hugging Face publishes before using it.

| pack | folder under `~/Models` | on disk | notes |
|---|---|---|---|
| **Coder · recommended for agents** | `mbrown-uno/Qwen3.8-Flash-Next-GSQ-RCO-Coder-MLX` | 76 GB | 256 experts (pruned for code by ISTA-DASLab), mixed 4/5/6/8 bits |
| full | `mlx-community/Qwen3.8-Flash-Next-4bit-mtp` | 105 GB | 512 experts, the original model, 4-bit |

Weights are under the Qwen Community License 1.0 and are not part of this repository.

### Qwen3.8-27B

The same binary runs the dense Qwen3.8-27B: give any command a 27B pack. It reads LoRA adapters too (see [LoRA adapters](#lora-adapters)).

| pack | `saetta download` | folder under `~/Models` | on disk |
|---|---|---|---|
| 8-bit · recommended with adapters | `27b-8bit` | `mlx-community/Qwen3.8-27B-8bit` | 29 GB |
| 4-bit | `27b-4bit` | `mlx-community/Qwen3.8-27B-4bit` | 16 GB |
| pi adapter (LoRA, r 32) | `27b-pi` | `harness-hub/Qwen3.8-27B-pi-adapted` | 0.6 GB |

Qwen3.8-27B weights and the pi adapter (harness-hub/Qwen3.8-27B-pi-adapted, card metadata) are under Apache-2.0.

### LoRA adapters

PEFT LoRA adapters (`adapter_config.json`, `adapter_model.safetensors`) run on 27B packs, applied at run time; the base model stays available next to them. The flags follow vLLM's:

```sh
saetta download 27b-8bit && saetta download 27b-pi
saetta serve ~/Models/mlx-community/Qwen3.8-27B-8bit \
    --lora-modules qwen3.8-pi=harness-hub/Qwen3.8-27B-pi-adapted \
    --default-chat-template-kwargs '{"enable_thinking": false}'
```

A request whose `model` is an adapter's name (`qwen3.8-pi`) runs with that adapter, any other name with the base model (`qwen3.8-27b`); `/v1/models` lists both. A path is a folder, or a Hugging Face id under `~/Models`. Prompt caches are kept per adapter. `--default-chat-template-kwargs` fills in chat template settings a request leaves out: the pi adapter was trained with thinking off.

For pi in one command (thinking off, pi pointed at the adapter):

```sh
saetta start pi --pack 27b-8bit --lora qwen3.8-pi=harness-hub/Qwen3.8-27B-pi-adapted
```

`SAETTA_LORA=<adapter folder>` applies an adapter to `saetta generate`. Supported: LoRA on the attention and MLP projections, any rank that is a multiple of 8 up to 128, rsLoRA; not DoRA, biases or saved modules. Each adapter adds about 0.3 GB of reads per generated token. Flash-Next packs take no adapters.

## Run the server

`saetta start` runs it for you (see [Coding agents](#coding-agents)). To run it by hand:

```sh
M=~/Models/mbrown-uno/Qwen3.8-Flash-Next-GSQ-RCO-Coder-MLX   # or the full pack
SAETTA_CTX=204800 saetta serve $M --port 8080                # loads in about 5 to 10 s
```

The server listens on 127.0.0.1 only and prints one `request:` line per request (prompt tokens, reused tokens, prompt and decode time, tok/s). Stop it with Ctrl-C.

One server runs per machine: the model takes most of the memory, so a second `saetta serve` refuses to start and names the one already running (its pid and port, recorded in `~/.saetta/run/serve.lock`).

### Settings

| variable | default | effect |
|---|---|---|
| `SAETTA_CTX` | 32768 | context capacity in tokens, preallocated (about 35 KB per token, 7.2 GB at 204800); use 204800 for agents |
| `SAETTA_SPEC` | 1 | `0` turns speculative decoding off (greedy output is identical either way) |
| `SAETTA_MTP` | 1 | `0` drafts with prompt lookup only, not the MTP head |
| `SAETTA_KV_F32` | unset | f32 instead of f16 KV cache (twice the memory) |

### Examples with curl

The model id is `qwen3.8-flash-next` for both packs (`GET /v1/models`).

```sh
curl -s localhost:8080/health
curl -s localhost:8080/v1/models

# one answer, thinking off
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "qwen3.8-flash-next", "temperature": 0, "max_tokens": 300,
  "chat_template_kwargs": {"enable_thinking": false},
  "messages": [{"role": "user", "content": "Write a Python function that checks if a string is a palindrome."}]
}' | python3 -m json.tool

# streaming (server-sent events)
curl -sN localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "qwen3.8-flash-next", "temperature": 0, "max_tokens": 300, "stream": true,
  "messages": [{"role": "user", "content": "Explain what a mutex is in two sentences."}]
}'

# a tool call, as a coding agent sends it
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "qwen3.8-flash-next", "temperature": 0, "max_tokens": 200,
  "messages": [{"role": "user", "content": "Read the file src/main.c and tell me what it does."}],
  "tools": [{"type": "function", "function": {"name": "read_file", "description": "Read a file",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
}' | python3 -m json.tool
```

The last one answers with `"finish_reason": "tool_calls"` and `{"name": "read_file", "arguments": "{\"path\":\"src/main.c\"}"}`.

Endpoints: `POST /v1/messages` and `POST /v1/messages/count_tokens` (Anthropic Messages API: system blocks, thinking, `tool_use` and `tool_result`, streaming events), `POST /v1/chat/completions` (OpenAI: JSON or `"stream": true` SSE; `reasoning_content`, `tool_calls`, `tool_choice` (`none`, `required`, a named function), `temperature`, `top_p`, `top_k`, `seed`, `stop`, `max_tokens`; template options `enable_thinking`, `reasoning_effort`, `preserve_thinking` in `chat_template_kwargs`), `GET /v1/models`, `GET /health`. One request at a time. A request whose prompt extends the previous one reuses its state and reads only the new tokens (`usage.prompt_tokens_details.cached_tokens`); the state after each reply is kept too, so the next turn does not re-read the reply. Every request decodes with speculation: the model's MTP head drafts 1 to 4 tokens (depth chosen by measured acceptance and timing), or prompt lookup up to 7 when the text repeats. Greedy requests (temperature 0) give the same output as without it up to rounding: verifying drafts in chunks rounds slightly differently from single steps, so where two tokens are near tied the choice can differ; sampled requests use speculative sampling, so their tokens follow the same distribution as without it.

<a id="coding-agents"></a>

## Coding agents

```sh
saetta start claude      # Claude Code
saetta start pi          # pi
saetta start opencode    # opencode
saetta start claude -- -p "fix the failing test"   # anything after -- goes to the agent
```

Guides: [Claude Code](docs/claude-code.md) · [opencode](docs/opencode.md)

<p align="center">
  <img src="docs/assets/demo-pi.gif" width="900" alt="saetta start pi: pi on saetta writes a Mandelbrot renderer, which then draws the set in color">
  <br><sub><code>saetta start pi</code>: pi writes a Mandelbrot renderer on the local model, then it runs. Real time.</sub>
</p>

`saetta start` joins the saetta server already running on this machine or starts one (with a 204,800-token context, logging to `~/Library/Logs/saetta/`), waits until the model is loaded (showing the seconds elapsed, and stopping with the server's error if it fails), and runs the agent pointed at it. Your own agent configuration is not edited: Claude Code gets its settings through environment variables and `--settings`, pi a config folder of its own (`~/.saetta/pi`), opencode a config file of its own (`~/.saetta/opencode`). When the last session exits, the server it started stops; `--keep` leaves it running so the next session starts at once.

| option | effect |
|---|---|
| `--pack coder\|full\|<dir>` | the model to load (default: the first downloaded of coder, full) |
| `--port N` | server port (default 8080, or `SAETTA_PORT`); if another program holds the default port, the next free one is used with a warning, a port you chose is an error |
| `--ctx N` | context capacity in tokens (default 204800, or `SAETTA_CTX`) |
| `--keep` | leave the server running when the agent exits |

Several sessions can run at once, in any mix of agents: they share the one server and its requests take turns. The server keeps several conversations (up to 4, or `SAETTA_CONVERSATIONS`), each with only the memory it needs: a conversation starts at 32K tokens (about 1.3 GB) and grows as it does, up to `SAETTA_CTX`. With the Coder pack all 4 fit; the full pack's weights fill the 128 GB, so it keeps one. Each request goes to the conversation it continues, so sessions that take turns keep their fast follow-up turns: two Claude Code sessions working at once reused about 19K of their 19K-token prompts on every turn. A new session also starts from what it shares with the others (an agent's system prompt and tool list): a second Claude Code session read its first prompt in 3.7 s instead of 11. A server started by `saetta start` stops when its last session ends, also if a terminal is closed or the launcher is killed (after 20 s without a live session).

### Without saetta start

Start the server as in [Run the server](#run-the-server), then:

- **Claude Code** ([guide](docs/claude-code.md)): pass the settings in [`config/claude-code/settings.json`](config/claude-code/settings.json): `claude --settings config/claude-code/settings.json --model qwen3.8-flash-next`. They point it at `http://127.0.0.1:8080` with a local token and route every model tier to saetta.
- **pi:** copy [`config/pi/models.json`](config/pi/models.json) into `~/.pi/agent/models.json` (or merge its provider), then `pi --provider saetta --model qwen3.8-flash-next`.
- **opencode:** put [`config/opencode/saetta-coder.json`](config/opencode/saetta-coder.json) in `opencode.json` at the root of a project and run `opencode` there. More in [docs/opencode.md](docs/opencode.md).
- **Anything else:** an OpenAI-compatible base URL `http://127.0.0.1:8080/v1` or an Anthropic one `http://127.0.0.1:8080`, model `qwen3.8-flash-next`, any API key.

Keep the agent's context limit at or below the server's `SAETTA_CTX`.

<a id="monitor"></a>

## Monitor

```sh
saetta monitor          # live view: r requests, c conversations, l server log, space freezes, q quits
saetta monitor --once   # one frame, for scripts (--conversations or --log for the other views)
saetta status           # the server on this machine: pid, port, model, state, sessions
saetta stop             # stop it
```

<p align="center">
  <img src="docs/assets/monitor.png" width="900" alt="saetta monitor: model state, speed, GPU load, GPU and system memory, prompt cache, speculation, context, sessions and recent requests">
</p>

`saetta monitor` shows the server on this machine while your agents work: what the model is doing (reading a prompt, with progress, or generating), decode speed, GPU load and the thermal state, the memory saetta holds on the GPU and the system's RAM and swap, how much of each prompt was reused, tokens per pass from speculation, the context kept for the next turn, the connected agent sessions, and the last requests with their prompt, reuse, output, speed and finish reason. It is read-only: quitting leaves the server running. The server writes its numbers twice a second to `~/.saetta/run/metrics.json`, so the view never waits behind a request.

## Performance

Measured on an M5 Max with 128 GB and macOS 27.2, on a six-request coding-agent workload: three cold prompts, two turns of one conversation and a tool-call edit. Time to first token and total time per request, in seconds.

| request | prompt tokens | full: first token | full: total | Coder: first token | Coder: total |
|---|---:|---:|---:|---:|---:|
| cold 4K | 3,334 | 2.02 | 3.54 | 1.96 | 3.94 |
| cold 16K | 15,868 | 8.72 | 10.36 | 8.13 | 10.09 |
| cold 64K | 69,925 | 39.81 | 41.89 | 36.84 | 39.29 |
| turn 1 | 15,864 | 8.63 | 9.46 | 8.01 | 8.82 |
| turn 2 (reuses turn 1) | 15,948 | 0.20 | 0.90 | 0.20 | 1.01 |
| edit (tool call, 400 tokens out) | 18,245 | 10.10 | 12.93 | 9.32 | 12.19 |

Prompt reading 1,650 to 1,820 tokens/s (full) and 1,700 to 1,950 (Coder); answers 72 to 82 and 63 to 70 tokens/s. On the edit, speculative decoding reaches 141 tokens/s on the full pack, by the server's own count.

## Command line

```sh
saetta download coder|full|all          # fetch a pack (pinned revision, resumable, hash-checked)
saetta download 27b-8bit|27b-4bit|27b-pi  # Qwen3.8-27B packs and the pi LoRA adapter
saetta start claude|pi|opencode         # a coding agent on saetta (starts the server)
  [--pack 27b-8bit] [--lora name=path]  # on Qwen3.8-27B, with a LoRA adapter
saetta monitor                          # live view of the server, GPU, memory, prompt cache
saetta status | stop                    # show or stop the server on this machine
saetta update                           # install the latest release over this binary
saetta serve $M [--port N]              # the API server
  [--lora-modules name=path ...] [--default-chat-template-kwargs JSON]   # 27B: LoRA adapters
saetta info $M                          # check the pack and the GPU path
saetta render request.json              # chat prompt for an OpenAI request body
saetta tokenize $M prompt.txt           # token ids, one per line
saetta generate $M 8 <token ids...>     # greedy, prints new ids and timing (@file.ids reads ids from a file)
saetta version
```

## Build from source

| | |
|---|---|
| Xcode | full Xcode with the Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`) |
| Compiler | Apple clang 21 or later (C23), part of Xcode |

```sh
git clone https://github.com/antonellof/saetta.git && cd saetta
make                                # ./build/bin/saetta
./build/bin/saetta download coder   # or full, or all
```

`make test` runs the unit tests (kernels against scalar references; `SAETTA_PACK=<pack>` for the tests that read real tensors).

| path | what |
|---|---|
| `src/` | the engine in C23: `model.c` Flash-Next forward pass, `dense.c` Qwen3.8-27B forward pass, `lora.c` LoRA kernels, `kernels/` Metal, `server.c` API, `tokenizer.c`, `chat.c`, `toolcall.c`, `metal/shim.*` |
| `scripts/` | curl installer |
| `config/` | agent configurations: Claude Code, pi, opencode |
| `docs/` | guides: [Claude Code](docs/claude-code.md), [opencode](docs/opencode.md) |

## License

[MIT](LICENSE). Parts are adapted from the MIT-licensed engines sushi, qwental and Strata; each adapted file names its source in its header, and their notices are in [LICENSE](LICENSE). Qwen3.8-Flash-Next weights are under the Qwen Community License 1.0; Qwen3.8-27B weights under Apache-2.0.
