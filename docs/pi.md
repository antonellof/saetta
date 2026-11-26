# pi

[pi](https://pi.dev) runs on Saetta's OpenAI-compatible API. `saetta start pi` starts the server and pi together.

```sh
saetta start pi                                     # the first downloaded pack that fits this Mac
saetta start pi --pack 27b-turbo
saetta start pi -- -p "fix the failing test"        # anything after -- goes to pi
```

## What it sets

`saetta start pi` runs pi with `PI_CODING_AGENT_DIR=~/.saetta/pi`, so your own `~/.pi/agent` is left alone. In that folder it writes only its own entries and keeps everything else:

- in `models.json`, the `saetta` provider (the server's URL, the model, its context window);
- in `settings.json`, `defaultProvider` and `defaultModel`.

It also sets `PI_SKIP_VERSION_CHECK=1` and passes `--provider saetta --model <id>`. Saetta Code started with `saetta start code` uses the same folder, so the two share settings, models, sessions and sign-ins: either one continues the other's sessions.

To use Saetta from your usual pi setup instead, copy the provider from [`config/pi/models.json`](../config/pi/models.json) into `~/.pi/agent/models.json` and start a server (`saetta serve coder`). The entry is shown on [Models and providers](models-and-providers.md#modelsjson).

## The pi adapter

[harness-hub's pi adapter](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted) is a LoRA adapter for Qwen3.8-27B trained to drive pi (its card: SWE-bench Pro hard 39.2% to 64.7% over the base model).

```sh
saetta download 27b-8bit && saetta download 27b-pi
saetta start pi --pack 27b-8bit --lora-modules qwen3.8-pi=27b-pi
```

With an adapter, thinking is off by default, as the adapter was trained. See [LoRA adapters](lora.md).

## saetta start

`saetta start <claude|pi|opencode|code>` joins the server on this machine or starts one, waits for the model and runs the agent pointed at it, without touching your agent configuration; the server it started stops with the last session. Several sessions share one server (up to 4 conversations, `SAETTA_CONVERSATIONS`).

| option | effect |
|---|---|
| `<pack>`, `--pack <name or dir>` | the model (default: the first downloaded pack that fits this Mac: coder, full, then the 27B packs); a download name right after the agent works too (`saetta start code 27b-8bit`) |
| `--lora-modules name=adapter ...` | LoRA adapters on a 27B pack, as `saetta serve` takes them (`--lora name=adapter` for one) |
| `--default-chat-template-kwargs JSON` | chat template values a request leaves out (with adapters, `{"enable_thinking": false}` unless given) |
| `--port N`, `--ctx N` | server port (8080) and context (204800 tokens) |
| `--keep` | leave the server running when the agent exits |

`SAETTA_PORT` and `SAETTA_CTX` set the same defaults. When port 8080 is taken by another program, `saetta start` moves to the next free port; a port you chose is an error instead. A missing agent is reported with its official install command.
