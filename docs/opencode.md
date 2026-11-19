# Using saetta with opencode

saetta serves an OpenAI-compatible API (`/v1/chat/completions` with streaming and tool calls,
`/v1/models`, `/health`), which opencode reaches through its `@ai-sdk/openai-compatible`
provider. Checked 2026-11-15 with opencode 1.19.5 and the Coder pack: an
agent turn (run a test, fix the bug, rerun) completed; opencode's ~12K-token system prompt is
read once (7.1 s) and every later step reuses it (0.24-0.38 s), decode 67-99 tok/s.

The quickest way: `saetta start opencode` starts the server and opencode together, with its own
config file (`~/.saetta/opencode/opencode.json`). The steps below do the same by hand.

## 1. Start the server

    # Coder variant (256 experts, 76 GB on disk); or the full pack (105 GB):
    #   ~/Models/mlx-community/Qwen3.8-Flash-Next-4bit-mtp
    SAETTA_CTX=204800 build/bin/saetta serve ~/Models/mbrown-uno/Qwen3.8-Flash-Next-GSQ-RCO-Coder-MLX --port 8080

`SAETTA_CTX` is the context capacity in tokens (KV cache preallocated, about 35 KB per token;
the default 32768 is too small for an agent session). Keep opencode's `limit.context` at or
below it.
Load takes about 7 s. One request runs at a time; one conversation's state is reused across
its turns, so run one opencode session per server.

## 2. Point opencode at it

Copy `config/opencode/saetta-coder.json` to `opencode.json` in the project (or merge its
`provider` and `model` into `~/.config/opencode/opencode.json`), then run `opencode` there.
`"temperature": 0` makes requests greedy, which turns on saetta's speculative decoding (MTP
head and prompt lookup); output is the same as without it.

## Check it by hand

    curl -s localhost:8080/v1/models     # {"data":[{"id":"qwen3.8-flash-next",...}]}

More curl examples (streaming, tool calls) are in the README under "Running the server".

## Notes

- The first opencode run in a fresh config folder installs the provider package (minutes);
  later runs start at once.
- `opencode run --auto "<task>"` runs one turn non-interactively.
- Server log: stderr, or wherever you redirect it; each request prints a `request:` line with prompt
  tokens, reused tokens, prefill and decode times.
