# Using saetta with Claude Code

saetta serves the Anthropic Messages API (`POST /v1/messages`, streaming, thinking, tool use,
`/v1/messages/count_tokens`), so Claude Code runs on Qwen3.8-Flash-Next locally, with no
account and nothing leaving the machine. Checked with Claude Code 2.1.325 and the Coder pack:
a read, edit and verify task completed end to end, and every turn after the first reused
about 20K of its roughly 20K prompt tokens (about 0.2 s to read each new turn).

## Quick start

```sh
saetta download coder      # once: 76 GB into ~/Models
saetta start claude        # starts the server, then Claude Code pointed at it
```

Run it in your project folder, like `claude`. The first start loads the model (about 5 to 10 s);
Claude Code opens when the server is ready. When you quit Claude Code, the server stops,
unless another session still uses it.

Everything after `--` goes to Claude Code:

```sh
saetta start claude -- -p "fix the failing test in tests/test_parser.py"   # one turn, no interface
saetta start claude -- --continue                                          # resume the last conversation
saetta start claude --keep                                                 # leave the server running afterwards
```

## What saetta start sets

Your `~/.claude/settings.json` is not edited. saetta passes these as environment variables and
again through `--settings`, which takes precedence over your user settings for this session only:

| variable | value | why |
|---|---|---|
| `ANTHROPIC_BASE_URL` | `http://127.0.0.1:<port>` | requests go to saetta |
| `ANTHROPIC_AUTH_TOKEN` | `saetta` | any token works; this one needs no login or approval |
| `ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU,FABLE}_MODEL`, `CLAUDE_CODE_SUBAGENT_MODEL` | `qwen3.8-flash-next` | every model tier and every subagent uses saetta |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | the server's context (204800) | Claude Code would otherwise assume a 1M window and compact too late |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | 32000 | |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`, `DISABLE_TELEMETRY`, `DISABLE_AUTOUPDATER` | 1 | no telemetry, update checks or feature-flag fetches |
| `CLAUDE_CODE_DISABLE_TERMINAL_TITLE` | 1 | no extra model request to title the session |
| `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS` | 1 | no pre-release request fields |
| `CLAUDE_CODE_ATTRIBUTION_HEADER` | 0 | no per-request stamp at the start of the system prompt |

`ANTHROPIC_API_KEY` is removed from the environment of that session, so a key you have set for
Anthropic's API is not sent to saetta.

## By hand

Start the server, then run Claude Code with the same settings from
[`config/claude-code/settings.json`](../config/claude-code/settings.json):

```sh
SAETTA_CTX=204800 saetta serve ~/Models/mbrown-uno/Qwen3.8-Flash-Next-GSQ-RCO-Coder-MLX &
claude --settings config/claude-code/settings.json --model qwen3.8-flash-next
```

Change the port in the file if the server is not on 8080, and keep
`CLAUDE_CODE_MAX_CONTEXT_TOKENS` at or below the server's `SAETTA_CTX`.

## Good to know

- **Prompt reuse.** Each turn sends the whole conversation; saetta reads only what is new. Claude
  Code's per-request billing stamp is neutralized server-side too, so reuse also works when you
  connect without `saetta start`.
- **Thinking** is on (Claude Code's effort setting maps to the model's reasoning effort) and is
  streamed as thinking blocks.
- **Speed.** Claude Code samples (it does not ask for greedy decoding); saetta speculates with
  speculative sampling, which keeps the same output distribution: 70 to 110 tokens/s on code
  and tool calls on the Coder pack, about 63 on prose. Watch it with `saetta monitor`.
- **Several sessions** (Claude Code, pi or opencode, in any mix) share the one server and take
  turns. The server keeps up to 3 conversations with the Coder pack (1 with the full pack), so
  each session keeps its fast follow-up turns, and a new session starts from the system prompt
  and tool list it shares with earlier ones (first prompt in about 3.7 s instead of 11).
- **Tool choice:** `none` leaves the tools out of the prompt; `any`, or a named tool, starts the
  answer with the tool call's opening, so the model calls a tool (with one tool declared, `any`
  calls that one).
- **Not supported:** image input (an image or PDF in a message becomes a short note saying it
  was omitted). Web search and other Anthropic-hosted tools are not available.
- A warning that claude.ai connectors are disabled is expected: the local token takes the place
  of your claude.ai login for that session.

## Troubleshooting

| symptom | what to do |
|---|---|
| `'claude' is not installed or not on PATH` | install Claude Code: `curl -fsSL https://claude.ai/install.sh \| bash` |
| `no model downloaded yet` | `saetta download coder` |
| the server does not come up | read `~/Library/Logs/saetta/serve-8080.log` |
| `a server is already running on this machine` | only one runs at a time; `saetta start` joins it, `saetta serve` refuses |
| "There's an issue with the selected model" | the server is not reachable: check the port in your settings |
