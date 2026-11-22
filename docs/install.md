# Install

Saetta is one binary for Apple Silicon Macs. The install script puts it in `~/.local/bin`.

```sh
curl -fsSL https://saetta.dev/install.sh | sh
```

It needs an Apple Silicon Mac (M1 or later) with 32 GB of memory or more, and macOS 27.2 or later. Make sure `~/.local/bin` is on your `PATH`.

## Options

The script reads two environment variables:

| variable | effect |
|---|---|
| `SAETTA_PREFIX` | install into `$SAETTA_PREFIX/bin` (default `~/.local`) |
| `SAETTA_VERSION` | pin a release, such as `SAETTA_VERSION=0.16.0` (default: the latest) |

Model downloads read `HF_TOKEN` (or `~/.cache/huggingface/token`), a Hugging Face token that raises Hugging Face's limits.

```sh
curl -fsSL https://saetta.dev/install.sh | SAETTA_PREFIX=/opt/saetta sh
```

## The guided start

`saetta` alone in a terminal (or `saetta auto`) shows this Mac's chip, memory and free disk, lists the models best first (the ones that do not fit are greyed out), downloads the one you pick, offers the built-in agent or the Claude Code, pi and opencode it finds (with the official installer for a missing one), and starts it.

```sh
saetta
```

## By hand

```sh
saetta download          # the model that fits this Mac, into ~/Models, resumable, hash-checked
saetta start code        # or: saetta start claude, saetta start pi, saetta start opencode
```

`saetta download` picks by memory: `coder` (Flash-Next, 76 GB) with 96 GB or more, `27b-turbo` (Qwen3.8-27B fine-tune, 20 GB, converted from its GGUF: about 40 GB free disk while it runs) with 36 GB, `27b-4bit` (16 GB) with 32 GB. Any pack can be named instead; see [Choosing a model](choosing-a-model.md). Packs go to `~/Models`, or to `SAETTA_MODELS` when it is set.

## Update and uninstall

```sh
saetta update            # the latest release
saetta version
```

To uninstall, delete the binary (`~/.local/bin/saetta`), the packs you downloaded under `~/Models`, `~/.saetta` (server state, the agent log, and the configuration `saetta start` writes for the agents) and `~/Library/Logs/saetta` (server logs).

To build it yourself, see [Build from source](build-from-source.md).
