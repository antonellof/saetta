# opencode

Saetta serves an OpenAI-compatible API (`/v1/chat/completions` with streaming and tool calls, `/v1/models`, `/health`), which [opencode](https://opencode.ai) reaches through its `@ai-sdk/openai-compatible` provider.

## Quick start

```sh
saetta start opencode
```

`saetta start opencode` starts the server and opencode together. It writes its own config file, `~/.saetta/opencode/opencode.json`, and points opencode at it with `OPENCODE_CONFIG`, so your opencode configuration is not edited. The options are those of [saetta start](pi.md#saetta-start).

## By hand

Start the server with a context large enough for an agent session:

```sh
SAETTA_CTX=204800 saetta serve coder --port 8080
```

`SAETTA_CTX` is the context in tokens (the KV cache is allocated for it, about 35 KB per token with Flash-Next; the default 32768 is too small for an agent session). The model loads in 5 to 10 s.

Then copy [`config/opencode/saetta-coder.json`](../config/opencode/saetta-coder.json) to `opencode.json` in the project (or merge its `provider` and `model` into `~/.config/opencode/opencode.json`) and run `opencode` there:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "saetta/qwen3.8-flash-next-coder",
  "small_model": "saetta/qwen3.8-flash-next-coder",
  "provider": {
    "saetta": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "saetta (local, M5 Max)",
      "options": { "baseURL": "http://127.0.0.1:8080/v1", "timeout": false },
      "models": {
        "qwen3.8-flash-next-coder": {
          "name": "Qwen3.8-Flash-Next Coder (saetta)",
          "limit": { "context": 204800, "output": 8192 },
          "options": { "temperature": 0 }
        }
      }
    }
  }
}
```

Keep `limit.context` at or below the server's `SAETTA_CTX`. `"temperature": 0` makes requests greedy, which turns on Saetta's speculative decoding (MTP head and prompt lookup); the output is the same as without it.

## Check it by hand

```sh
curl -s localhost:8080/v1/models     # {"object":"list","data":[{"id":"qwen3.8-flash-next",...}]}
```

## Notes

- The first opencode run in a fresh config folder installs the provider package, which can take minutes; later runs start at once.
- `opencode run --auto "<task>"` runs one turn non-interactively.
- The server log is `~/Library/Logs/saetta/serve-<port>.log` when `saetta start` runs the server, or the terminal for `saetta serve`. Each request prints a `request:` line with prompt tokens, reused tokens, prefill and decode times.
