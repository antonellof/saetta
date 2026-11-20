<p align="center">
  <img src="docs/assets/saetta-banner.png" alt="saetta" width="900">
</p>

<h3 align="center">Qwen3.8-Flash-Next and Qwen3.8-27B at full speed on an M5 Max.</h3>
<p align="center">An inference engine in C and hand-written Metal, tuned for coding agents, with LoRA adapters.</p>

<p align="center">
  <a href="#install"><b>Install</b></a> · <a href="#models">Models</a> · <a href="#lora-adapters">LoRA adapters</a> · <a href="#coding-agents">Coding agents</a> ·
  <a href="#run-the-server">Server</a> · <a href="#monitor">Monitor</a> · <a href="#performance">Performance</a> · <a href="#build-from-source">Build from source</a>
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

**Two models, one machine, no compromises.** saetta runs Qwen3.8-Flash-Next and Qwen3.8-27B on an Apple M5 Max with 128 GB, and nothing else: every kernel is written for that machine and for coding agents (long shared-prefix prompts of 20K to 200K tokens, short answers, XML tool calls), served through OpenAI- and Anthropic-compatible APIs.

- **Fast:** on Flash-Next, prompts read at 1,650 to 1,950 tokens/s from 4K to 70K tokens and answers come at 63 to 82 tokens/s, with speculative decoding from the model's MTP head.
- **Agent turns that do not re-read:** a prompt that extends the previous one reads only its new tokens (a follow-up turn on a 16K conversation answers in 0.2 s), and new sessions reuse the prefix they share with others.
- **LoRA adapters** on Qwen3.8-27B, per request, next to the base model: for example the [pi agent adapter](#lora-adapters).
- **One command to code:** `saetta start claude` (or `pi`, `opencode`) starts the server and the agent.
- **Checked:** every layer of both models, and the LoRA path, against a numpy reference.
- **Small:** one binary that also downloads the models; no Python, no dependencies beyond macOS.

## Install

```sh
curl -fsSL https://github.com/antonellof/saetta/releases/latest/download/install.sh | sh   # into ~/.local/bin
brew install antonellof/saetta/saetta                                                     # or with Homebrew
```

Keep one of the two: the installer stops if Homebrew already has saetta. Update with `saetta update` (curl install) or `brew upgrade saetta`. Then:

```sh
saetta download coder    # 76 GB into ~/Models, resumable, hash-checked
saetta start claude      # or: saetta start pi, saetta start opencode
```

Needs an Apple M5 Max with 128 GB and macOS 27.2 or later. `SAETTA_PREFIX` changes the install folder, `SAETTA_VERSION=0.8.0` pins a release, `HF_TOKEN` raises Hugging Face's download limits.

## Models

`saetta download <name>` fetches a pack at a pinned revision into `~/Models/<org>/<repo>` (`SAETTA_MODELS` to change it), four files at a time, resuming and checking every file's hash. `all` fetches both Flash-Next packs.

| name | model | folder under `~/Models` | on disk |
|---|---|---|---|
| **`coder`** · recommended for agents | Flash-Next, 256 experts pruned for code (ISTA-DASLab), mixed 4/5/6/8-bit | `mbrown-uno/Qwen3.8-Flash-Next-GSQ-RCO-Coder-MLX` | 76 GB |
| `full` | Flash-Next, the original 512 experts, 4-bit | `mlx-community/Qwen3.8-Flash-Next-4bit-mtp` | 105 GB |
| `27b-8bit` · recommended with adapters | Qwen3.8-27B, dense, 8-bit | `mlx-community/Qwen3.8-27B-8bit` | 29 GB |
| `27b-4bit` | Qwen3.8-27B, dense, 4-bit | `mlx-community/Qwen3.8-27B-4bit` | 16 GB |
| `27b-pi` | the pi LoRA adapter for Qwen3.8-27B (r 32) | `harness-hub/Qwen3.8-27B-pi-adapted` | 0.6 GB |

Every command takes either model's pack; the model is read from the pack. Flash-Next weights are under the Qwen Community License 1.0, [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) and the pi adapter under Apache-2.0; none are part of this repository.

### LoRA adapters

PEFT LoRA adapters run on Qwen3.8-27B, applied at run time with vLLM's flags. [harness-hub/Qwen3.8-27B-pi-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted) is trained with RL to drive the [pi](https://pi.dev) coding agent with thinking off; its card reports SWE-bench Pro (hard 51) 39.2% to 64.7% and Terminal-Bench 2.1 39.5% to 55.8% over the base model (harness-hub's numbers). pi on it, in one command:

```sh
saetta download 27b-8bit && saetta download 27b-pi
saetta start pi --pack 27b-8bit --lora qwen3.8-pi=harness-hub/Qwen3.8-27B-pi-adapted
```

Or serve it for any client:

```sh
saetta serve ~/Models/mlx-community/Qwen3.8-27B-8bit \
    --lora-modules qwen3.8-pi=harness-hub/Qwen3.8-27B-pi-adapted \
    --default-chat-template-kwargs '{"enable_thinking": false}'
```

A request whose `model` is an adapter's name (`qwen3.8-pi`) runs with it, any other with the base model (`qwen3.8-27b`); `/v1/models` lists both, and prompt caches are kept per adapter. A path is a folder or a Hugging Face id under `~/Models`. Supported: LoRA on the attention and MLP projections, rank a multiple of 8 up to 128, rsLoRA (not DoRA, biases or saved modules). An adapter adds about 0.3 GB of reads per generated token. `SAETTA_LORA=<folder>` applies one to `saetta generate`.

<a id="coding-agents"></a>

## Coding agents

```sh
saetta start claude      # Claude Code
saetta start pi          # pi
saetta start opencode    # opencode
saetta start claude -- -p "fix the failing test"   # anything after -- goes to the agent
```

<p align="center">
  <img src="docs/assets/demo-pi.gif" width="900" alt="saetta start pi: pi on saetta writes a Mandelbrot renderer, which then draws the set in color">
  <br><sub><code>saetta start pi</code>: pi writes a Mandelbrot renderer on the local model, then it runs. Real time.</sub>
</p>

`saetta start` joins the server running on this machine or starts one (logs in `~/Library/Logs/saetta/`), waits for the model to load and runs the agent pointed at it, without editing your own agent configuration. The server it started stops when the last session exits.

| option | effect |
|---|---|
| `--pack coder\|full\|27b-8bit\|27b-4bit\|<dir>` | the model (default: the first downloaded of coder, full) |
| `--lora name=path` | a LoRA adapter on a 27B pack, thinking off, the agent pointed at it |
| `--port N` | server port (default 8080, or `SAETTA_PORT`; a taken default moves to the next free port) |
| `--ctx N` | context in tokens (default 204800, or `SAETTA_CTX`) |
| `--keep` | leave the server running when the agent exits |

Several sessions, of any agents, share the one server and take turns; it keeps up to 4 conversations (`SAETTA_CONVERSATIONS`), each growing from 32K tokens as needed, so each session keeps its fast follow-up turns.

Without `saetta start`: run the server, then point the agent at `http://127.0.0.1:8080/v1` (OpenAI) or `http://127.0.0.1:8080` (Anthropic) with any API key and the model id from `/v1/models`. Ready configs: [`config/claude-code/settings.json`](config/claude-code/settings.json) ([guide](docs/claude-code.md)), [`config/pi/models.json`](config/pi/models.json), [`config/opencode/saetta-coder.json`](config/opencode/saetta-coder.json) ([guide](docs/opencode.md)).

## Run the server

```sh
M=~/Models/mbrown-uno/Qwen3.8-Flash-Next-GSQ-RCO-Coder-MLX   # or any pack
SAETTA_CTX=204800 saetta serve $M --port 8080                # loads in about 5 to 10 s
```

It listens on 127.0.0.1, prints one `request:` line per request, and stops with Ctrl-C. One server runs per machine (`~/.saetta/run/serve.lock`). The model id is `qwen3.8-flash-next` for the Flash-Next packs and `qwen3.8-27b` for 27B.

| variable | default | effect |
|---|---|---|
| `SAETTA_CTX` | 32768 | context capacity in tokens (Flash-Next about 35 KB per token, 27B 64 KB); use 204800 for agents |
| `SAETTA_SPEC` | 1 | `0` turns speculative decoding off |
| `SAETTA_MTP` | 1 | `0` drafts with prompt lookup only, not the MTP head (Flash-Next) |
| `SAETTA_KV_F32` | unset | f32 instead of f16 KV cache |

```sh
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "qwen3.8-flash-next", "temperature": 0, "max_tokens": 300,
  "chat_template_kwargs": {"enable_thinking": false},
  "messages": [{"role": "user", "content": "Write a Python function that checks if a string is a palindrome."}]
}' | python3 -m json.tool
```

Endpoints: `POST /v1/chat/completions` (OpenAI, JSON or SSE streaming, `tool_calls`, `tool_choice`, `reasoning_content`, sampling options; `enable_thinking`, `reasoning_effort` in `chat_template_kwargs`), `POST /v1/messages` and `/v1/messages/count_tokens` (Anthropic), `GET /v1/models`, `GET /health`. One request at a time. Speculation drafts with the MTP head (Flash-Next) or prompt lookup; greedy output matches decoding without it up to rounding at near ties, and sampled output follows the same distribution.

## Monitor

```sh
saetta monitor          # live view: r requests, c conversations, l server log, q quits
saetta status           # the server on this machine: pid, port, model, sessions
saetta stop             # stop it
```

<p align="center">
  <img src="docs/assets/monitor.png" width="900" alt="saetta monitor: model state, speed, GPU load, GPU and system memory, prompt cache, speculation, context, sessions and recent requests">
</p>

What the model is doing, decode speed, GPU load and memory, prompt reuse, speculation, sessions and the last requests, live; quitting leaves the server running.

## Performance

Flash-Next on an M5 Max with 128 GB and macOS 27.2, a six-request coding-agent workload: time to first token and total per request, in seconds.

| request | prompt tokens | full: first token | full: total | Coder: first token | Coder: total |
|---|---:|---:|---:|---:|---:|
| cold 4K | 3,334 | 2.02 | 3.54 | 1.96 | 3.94 |
| cold 16K | 15,868 | 8.72 | 10.36 | 8.13 | 10.09 |
| cold 64K | 69,925 | 39.81 | 41.89 | 36.84 | 39.29 |
| turn 1 | 15,864 | 8.63 | 9.46 | 8.01 | 8.82 |
| turn 2 (reuses turn 1) | 15,948 | 0.20 | 0.90 | 0.20 | 1.01 |
| edit (tool call, 400 tokens out) | 18,245 | 10.10 | 12.93 | 9.32 | 12.19 |

Prompt reading 1,650 to 1,820 tokens/s (full) and 1,700 to 1,950 (Coder); answers 72 to 82 and 63 to 70 tokens/s; speculative decoding reaches 141 tokens/s on the edit (full). Qwen3.8-27B is not benchmarked yet; long 27B prompts (past about 32K tokens) read slowly for now.

## Command line

```sh
saetta download <name>                  # coder | full | all | 27b-8bit | 27b-4bit | 27b-pi
saetta start claude|pi|opencode         # an agent on saetta [--pack ...] [--lora name=path]
saetta serve $M [--port N]              # the API server [--lora-modules name=path ...]
                                        #   [--default-chat-template-kwargs JSON]
saetta monitor | status | stop          # watch, show or stop the server
saetta update                           # newer release over a curl install
saetta info $M                          # check a pack and the GPU path
saetta render request.json              # chat prompt for an OpenAI request body
saetta tokenize $M prompt.txt           # token ids, one per line
saetta generate $M 8 <token ids...>     # greedy, new ids and timing (@file.ids reads a file)
saetta version
```

## Build from source

Needs full Xcode with the Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`) and Apple clang 21 or later (C23).

```sh
git clone https://github.com/antonellof/saetta.git && cd saetta
make                                # ./build/bin/saetta
make test                           # unit tests (SAETTA_PACK=<pack> for those reading real tensors)
```

`src/` is the engine: `model.c` (Flash-Next) and `dense.c` (Qwen3.8-27B) forward passes, `model_api.c` picks one per pack, `lora.c` and `kernels/` Metal, `server.c` the API, `tokenizer.c`, `chat.c`, `toolcall.c`. `scripts/` holds the installer, `config/` the agent configurations, `docs/` the guides.

## License

[MIT](LICENSE). Parts are adapted from the MIT-licensed engines sushi, qwental and Strata; each adapted file names its source in its header, and their notices are in [LICENSE](LICENSE).
