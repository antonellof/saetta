# Changelog

All notable changes to saetta and Saetta Code (`saetta code`), newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions are saetta's.

## [0.23.0] - 2026-11-27

### Added

- Saetta Code reads a running command's output into the conversation while the command runs (in process, Qwen3.8-Flash-Next and Qwen3.8-27B): when it ends, the next request finds most of the result already read and only reads the rest. The request still decides: what was read ahead and does not match is read again from the last point that does, and the answer is the same token for token. On a five-turn Coder session whose commands stream for 1 to 3 s, the turns take 38.6 s instead of 47.8, and the first token after the tool results comes 8.5 s sooner in all. A request from another session or agent goes first. `SAETTA_AHEAD=0` turns it off. Not yet for Inkling 2 Mini or over the HTTP API.

## [0.22.0] - 2026-11-27

### Added

- Inkling 2 Mini reads prompts on the M5's tensor ops: routed experts, dense projections, prompt attention and the loop's message gates run with half-precision activations and f32 sums, in 2048-token chunks. A 4K prompt reads at 2,396 tokens/s (473 in 0.21), 32K at 1,984 (424). Answering is unchanged. M1 to M4, `SAETTA_NO_ACCEL=1` and `SAETTA_PREFILL_F32=1` keep the f32 kernels; `make verify` checks the half path by KL against the reference.
- Checkpoints at message boundaries: the engine keeps restart points inside a conversation (at message starts at least 1024 tokens apart, and at each request's start), so a fork, an edit or a retry of an earlier message reads only from the nearest one. On the Coder session benchmark a fork at turn 13 takes 1.96 s instead of 7.77 s, and a retry's first token comes in 0.73 s instead of 1.29 s. Checkpoints stay inside the memory fit; HTTP clients each get their own conversation.

### Fixed

- Flash-Next: prompt lookup's n-gram history could be read while the next chunk was being prepared.

## [0.21.0] - 2026-11-26

### Added

- Durable runs (issue #2, stage A): Saetta Code keeps a journal beside each session while a turn runs, so a run cut off by a crash, a kill or a closed terminal is offered for resuming at the next start. Finished tool calls keep their result, read-only tools run again, and a command, an edit, an MCP tool or a subagent that was running is never run again without asking. Queued messages survive too.

## [0.20.0] - 2026-11-26

### Added

- Saetta Code saves the engine's state with the session on the SSD and restores it on resume (`saetta code -c`, `/resume`): a resumed session reads only its new message instead of the whole conversation. Checked against the uninterrupted run (exact, or recomputed when anything differs: pack, build, chip, switches); `SAETTA_STATE=0` turns it off. Not yet for Inkling 2 Mini.
- Inkling 2 Mini: context up to 1M tokens where memory allows (half-precision cache for its global layers), and Thinking Machines' DSpark drafter, opt-in with `saetta download inkling-dspark` and `SAETTA_DSPARK=1` (prompt lookup alone is faster at 4K and stays the default). Speculative answers are checked bit for bit against plain decoding.
- README, docs and site: "Model families" and an "All packs" table with each pack's options.

## [0.19.1] - 2026-11-26

### Added

- Saetta Code's footer shows the last turn's generation speed after the context size (`· 17.9 tok/s`), and `/session` shows generation and prompt reading speed for the model in process.

## [0.19.0] - 2026-11-26

### Added

- `saetta download 27b-qwopus`: Jackrong's Qwopus3.8-27B Coder-Flash, a coding fine-tune of Qwen3.8-27B distilled from Qwen3.8-Flash-Max (mlx-community's 4-bit, 16 GB, Macs with 32 GB or more). `27b-qwopus-8bit`: 29 GB, 48 GB. Same architecture and quantization as `27b-4bit` and `27b-8bit`, no MTP draft head.
- LoRA adapters in Saetta Code without a server: `saetta start code 27b-8bit --lora-modules qwen3.8-pi=27b-pi` (and `saetta code --pack P --lora-modules ...`) loads them into the engine in process, as `saetta serve` does, with `--default-chat-template-kwargs`; `--lora name=path` stays as a shorter form. The session runs on the first adapter, `--model NAME` picks another or the base model.
- `/model` lists LoRA adapters under each downloaded base model they fit ("adapter of Qwen3.8-27B 8-bit"): switching between the base model and the adapters loaded with it happens in place, another adapter starts Saetta Code again on its base, one not downloaded downloads from the list, and one without its base model on this Mac is greyed with the reason. Adapters are found in the download list and in any `org/repo` folder of the models folder that holds an `adapter_config.json`.
- `saetta start` takes `--lora-modules` (several adapters) and `--default-chat-template-kwargs` for every agent, and a download name right after the agent (`saetta start code 27b-8bit`).

### Changed

- An adapter that does not fit the base model (the base its `adapter_config.json` names, its rank or target modules) is refused before the load with the reason, by `saetta serve`, `saetta start` and Saetta Code alike.

### Fixed

- A Saetta Code session no longer waits forever for a shared model whose session has exited: a record counts only while its holder still holds the lock, and a session exiting slowly is waited for at most 30 s.
- `/model` filtering ranks matches the way pi does (best first), so the highlighted model is the one typed, not the current one.
- The README and site lead with Saetta Code and its models; a new ASCII-art banner.

## [0.18.0] - 2026-11-23

### Added

- Inkling 2 Mini (Thinking Machines, 26B mixture of experts, 3.4B active), first version: `saetta download inkling` (17 GB, from 32 GB of memory), served with its chat template and JSON tool calls. Checked layer by layer against a numpy reference built from the same weights. Not yet: speculative decoding, the M5's neural accelerators, reuse of a conversation that does not simply extend the last one.
- M1 to M4: prompt reading with half-precision kernels of their own (simdgroup matrices, from MLX's and llama.cpp's designs) and prompt attention that shares each key and value block across tokens (from sushi's). Measured on an M5 Max forced onto that path: the 27B reads prompts 68% faster, the Coder 52%; not yet measured on an M1 to M4.

### Fixed

- A kernel asking for more threads than the GPU allows now fails with its name instead of returning garbage (Metal skips such a dispatch silently; M1 and M2 allow fewer threads for kernels with many registers).

## [0.17.1] - 2026-11-23

### Fixed

- M1 to M4: replies were garbage. The prompt kernels written for the M5's neural accelerators now run only on GPUs that have them; M1 to M4 take the plain GPU kernels automatically (`SAETTA_NO_ACCEL=1` is no longer needed). `SAETTA_ACCEL=1` forces the accelerator kernels for testing.

## [0.17.0] - 2026-11-23

### Added

- Parallel agents: several Saetta Code sessions share one model. A `saetta start code` in another terminal uses the model the first one loaded, each session in its own conversation, over a private Unix socket. `--subagents` gives the model a tool that hands self-contained tasks to up to 3 child agents running at once.
- RPC mode (`saetta code --mode rpc`): pi's JSON protocol on stdin and stdout, every command of pi's rpc client, compared with pi on 12 recorded scenarios.
- OAuth sign-in for MCP servers over HTTP, as pi does it (discovery, dynamic registration, PKCE with a loopback callback).
- Documentation on saetta.dev/docs.

### Changed

- macOS 26 or later is enough (it was 27.2): nothing saetta uses needs a newer release.

### Notes

- Requests, session files and output match pi's byte for byte on 63 recorded print and JSON scenarios; the interactive screens on 75.

## [0.16.0] - 2026-11-22

### Added

- Saetta Code has pi's full terminal UI, ported to C: the editor with "/" commands and "@" file completion, markdown with code highlighting, tool views with diffs and live bash output, selectors, scrolling and mouse selection, pi's keys (kill ring, undo, character jump, Ctrl+X copy, Ctrl+V paste). Its screens match pi's on 69 recorded scenarios.
- Saetta Code's own look: a theme with macOS's system colors (chosen at the first start, `/theme` to change it; pi's themes stay available), a footer with the folder, git branch, time per turn and the model's full name and quantization, and a 3D lightning loader while the model loads.
- `/model` lists every saetta pack, downloads one from the list, switches packs in process, and adds, edits or removes OpenAI-compatible models in models.json.
- `/monitor`: the engine's live statistics inside the agent (speed, GPU, memory, prompt-cache reuse, speculation, recent requests).
- pi's session commands: `/compact` and automatic compaction, `/resume`, `/fork`, `/clone`, `/tree`, `/name`, `/session`, `/export` (pi's HTML page, or JSONL), `/import`, `/share` (a secret GitHub gist), `/copy`, `/hotkeys`, `/scoped-models`.
- Skills controls (`--skill`, `--no-skills`, settings entries, `/skill:name`, the startup list of loaded skills and context files, `/reload`), project trust shared with pi (`/trust`, the prompt for untrusted projects), `/changelog`.
- MCP: stdio and Streamable HTTP servers from pi's mcp.json, their tools offered to the model, the `/mcp` manager, `--no-mcp`.
- An update check at start (off with `SAETTA_SKIP_VERSION_CHECK=1` or `PI_OFFLINE=1`).

### Changed

- `saetta start code` and `saetta start pi` update only their own entries in settings.json and models.json and keep everything else.

### Notes

- Requests, session files and output match pi's byte for byte on 58 recorded scenarios.
- Not yet: OAuth sign-in for MCP servers, pi's TypeScript extensions and packages, RPC and SDK modes, image input.

## [0.15.0] - 2026-11-22

### Added

- Saetta Code, a minimal coding agent built into `saetta`, in C: a port of pi 1.5.2 that behaves as pi does (the same tools, system prompt, AGENTS.md / CLAUDE.md files, skills and session files, so pi and Saetta Code continue each other's sessions).
- `saetta start code` runs the model inside Saetta Code, with no server and no HTTP in between (or joins a running Saetta server). `saetta code --pack coder` does the same by hand.
- `saetta code -p "..."` and `--mode json` are pi's print and JSON modes; `-c` continues the folder's last session; piped input is read as pi reads it.
- The guided start (`saetta`) offers Saetta Code next to Claude Code, pi and opencode.
- The models table (README, saetta.dev) lists each model's published SWE-bench and Terminal-Bench scores.

### Notes

- Saetta Code's requests, session files and output match pi's byte for byte on 47 recorded scenarios.
- pi's TypeScript extensions are not supported.

## [0.14.0] - 2026-11-22

### Added

- `saetta download 27b-coder390`: nerkyor's Coder390 EfficientThink fine-tune of Qwen3.8-27B (SFT and RL on code), converted from its GGUF file with its MTP draft head, 23 GB, Macs with 48 GB or more. `27b-coder390-8bit`: 31 GB, 48 GB.
- `saetta download 27b-fable5`: khazarai's Fable-5 Coding Distilled fine-tune of Qwen3.8-27B (mradermacher's imatrix Q4_K_M), with its MTP draft head, 20 GB, Macs with 36 GB or more. `27b-fable5-8bit`: 31 GB, 48 GB.
- The guided start (`saetta`) lists the new models.

## [0.13.2] - 2026-11-21

### Fixed

- The installer (`curl -fsSL https://saetta.dev/install.sh | sh`) no longer ends with "SAETTA_MODELS: unbound variable" after a successful install.

### Changed

- The installer's closing message leads with `saetta`, the guided start.
- Install from saetta.dev; the README and the GitHub page point there.

## [0.13.1] - 2026-11-21

### Fixed

- The guided start (`saetta`, `saetta auto`) in a terminal narrower than a model row: rows are cut to the terminal's width, the list scrolls inside a short terminal, and each model's status (downloaded, recommended, or why it does not fit) has its own column.

## [0.13.0] - 2026-11-21

### Added

- `saetta` (or `saetta auto`) in a terminal is a guided start: it shows this Mac's chip, memory and free disk, lists the models best first, downloads the one you pick, finds Claude Code, pi and opencode, offers the official installer for a missing one, and starts it.

### Changed

- Qwen3.8-27B reads prompts a little faster (16K prompts about 2% faster; the long-prompt attention kernel 7 to 8% faster per layer), with the same output.
- The KV cache keeps a small tail so the attention kernel never reads past it.

## [0.12.0] - 2026-11-21

### Added

- Runs on any Apple Silicon Mac (M1 or later) with 32 GB of memory or more. Tuned and measured on an M5 Max; M1 to M4 run the same kernels on their GPU cores and are not yet tested on those chips.
- `saetta download` with no name picks the model that fits this Mac: `coder` with 96 GB or more, `27b-turbo` with 36 GB, `27b-4bit` with 32 GB.
- `SAETTA_NO_ACCEL=1` runs without the tensor-op (neural accelerator) kernels: same output, slower prompt reading.

### Changed

- A pack too large for this Mac's memory is refused (`SAETTA_IGNORE_MEMORY=1` overrides).
- The server caps the context to what memory holds and reports it (`max_model_len` in `/v1/models`); `saetta start` tells the agent the real context and picks a downloaded pack that fits.
- The binary is built for every Apple Silicon CPU (`-mcpu=apple-m1`).

## [0.11.0] - 2026-11-21

### Added

- `saetta help` lists every command.

### Changed

- `SAETTA_CTX` and `--ctx` are rounded to a multiple of 128 and checked against the model's limit (1,024 to 262,144).
- Requests with `max_tokens` below 1 or above 2^30, or `top_p` above 1, are rejected with a 400.
- Qwen3.8-27B: faster long-prompt attention (64K prompts 758 -> 787 tokens/s), same output.

### Fixed

- Deeply nested JSON, out-of-range numbers, oversized or malformed requests and slow clients get an error instead of crashing or blocking the server; errors after a stream has started end it with an SSE error event; a client that disconnects from a non-streamed request stops it; replies are always valid UTF-8; streamed output no longer leaks the start of a stop string; a `count_tokens` request whose body arrives after its headers is answered.

## [0.9.0] - 2026-11-20

### Added

- DeepSeek-V4-Flash-0731 (preview), from its official checkpoint read in place, with `saetta info`, `saetta generate` and the checks (`make verify-dsv4`).
- Streamed experts: a model whose routed experts do not fit in memory keeps the hot ones in a GPU-visible cache and reads the rest from the SSD on demand (`SAETTA_EXPERT_CACHE_GB` for Flash-Next, as an experiment).

### Changed

- Flash-Next and Qwen3.8-27B behave exactly as in 0.8.1.

## [0.8.1] - 2026-11-20

### Added

- Packs and adapters by name: `saetta serve coder`, `saetta serve 27b-8bit --lora-modules qwen3.8-pi=27b-pi`, `saetta start pi --pack 27b-8bit --lora qwen3.8-pi=27b-pi`. Names resolve under your models folder (`~/Models`, or `SAETTA_MODELS`).

### Changed

- The curl installer stops when Homebrew already has saetta and names any other copy on the PATH; `saetta update` leaves a Homebrew install to `brew upgrade saetta`.

## [0.8.0] - 2026-11-20

### Added

- Qwen3.8-27B (dense) in the same `saetta` binary: `saetta download 27b-8bit` or `27b-4bit`.
- LoRA adapters at run time, with vLLM's flags (`--lora-modules`, `--default-chat-template-kwargs`); prompt caches are kept per adapter.
- pi with the harness-hub pi adapter in one command: `saetta start pi --pack 27b-8bit --lora qwen3.8-pi=harness-hub/Qwen3.8-27B-pi-adapted`.
