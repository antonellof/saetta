# Environment variables

## Install

| variable | default | effect |
|---|---|---|
| `SAETTA_PREFIX` | `~/.local` | the install script puts the binary in `$SAETTA_PREFIX/bin` |
| `SAETTA_VERSION` | latest | the install script installs this release, such as `0.16.0` |

## Models and memory

| variable | default | effect |
|---|---|---|
| `SAETTA_MODELS` | `~/Models` | where packs are downloaded and looked up by name |
| `HF_TOKEN` | unset | a Hugging Face token for downloads, which raises Hugging Face's limits (else `~/.cache/huggingface/token`) |
| `SAETTA_IGNORE_MEMORY` | unset | `1` loads or downloads a pack this Mac's memory is too small for, and keeps the asked context |

## Server and engine

| variable | default | effect |
|---|---|---|
| `SAETTA_CTX` | 32768 (`saetta serve`), 204800 (`saetta start`) | context in tokens, 1024 to 262144 (Inkling 2 Mini 1048576), rounded down to a multiple of 128 (Flash-Next about 35 KB per token, 27B 64 KB) |
| `SAETTA_PORT` | 8080 | the port `saetta start` uses |
| `SAETTA_CONVERSATIONS` | 4 | conversations the server keeps between turns |
| `SAETTA_SPEC` / `SAETTA_MTP` | 1 / 1 | `0` turns speculative decoding / the MTP head (Inkling: the DSpark drafter) off |
| `SAETTA_DSPARK` | unset | Inkling 2 Mini: `1` loads the downloaded DSpark drafter (or a path to its `mtp.safetensors`); off by default |
| `SAETTA_KV_F32` | unset | f32 KV cache |
| `SAETTA_NO_ACCEL` | unset | `1` runs without the tensor-op (neural accelerator) kernels, for a Mac where they misbehave (slower prompt reading, same output) |
| `SAETTA_EXPERT_CACHE_GB` | unset | experimental: keep this many GB of experts in memory, read the rest from the SSD (Flash-Next: slower than resident, same output) |
| `SAETTA_LORA` | unset | an adapter folder for `saetta generate` |

## Saetta Code

Saetta Code reads these variables of pi's as pi does, along with its own.

| variable | default | effect |
|---|---|---|
| `PI_CODING_AGENT_DIR` | `~/.pi/agent` | the agent folder (settings, models, sessions, skills, MCP); `saetta start code` sets `~/.saetta/pi` |
| `SAETTA_SKIP_VERSION_CHECK`, `PI_SKIP_VERSION_CHECK` | unset | `1` skips the update check at start |
| `PI_OFFLINE` | unset | `1` skips the update check |
| `PI_TRUE_COLOR` | detected | `1` forces 24-bit color, `0` 256 colors |
| `PI_HYPERLINKS` | detected | `1` or `0` turns clickable links on or off |
| `PI_CLIPBOARD_OSC52` | unset | `1` copies through the terminal (OSC 52), as over SSH |
| `PI_TUI_ESC_TIMEOUT` | 10 (100 over SSH) | milliseconds to tell Escape from an escape sequence |
| `PI_CACHE_RETENTION` | unset | `long` asks providers that support it for 24-hour prompt caching |
| `SAETTA_DURABLE` | `1` | `0` keeps no journal of the run, so an interrupted run cannot be resumed ([Interrupted runs](sessions.md#interrupted-runs)) |
| `SAETTA_DURABLE_SYNC` | `barrier` | how a journal record reaches the disk before a tool that is never run again starts: `barrier` (F_BARRIERFSYNC), `full` (F_FULLFSYNC), `none` (survives a crash of the process, not a power loss) |
| `VISUAL`, `EDITOR` | nano | the editor Ctrl+G opens, when settings.json has no `externalEditor` |
| `SAETTA_STATE` | 1 | `0` turns off the conversation state on disk (`~/.saetta/state`): a resumed session then reads its history again |
