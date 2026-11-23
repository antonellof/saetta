# Parallel agents

Saetta Code runs several agents on one model in two ways: several Saetta Code sessions share the model one of them loaded, and one session can hand tasks to subagents that run in parallel.

## Sessions that share one model

The first interactive `saetta code --pack` (or `saetta start code`) loads the model in its process. Every Saetta Code started after it on the same Mac uses that model instead of refusing, with the same pack:

```sh
saetta start code          # terminal 1: loads the model
saetta start code          # terminal 2: uses the model of terminal 1
```

- Each session gets its own conversation in the engine (four by default; `SAETTA_CONVERSATIONS` sets up to eight), so switching between them does not reread their prompts. The engine answers one request at a time, in the order they arrive.
- A session started while the first is still loading waits for it. A session that asks for another pack is refused: one model runs at a time.
- When the first session exits while others still use the model, it keeps serving them until the last one exits (Ctrl-C stops it at once). A session whose model went away loads it itself on its next request, and the sessions started after it use it there.
- `/model` in the session that holds the model asks first when other sessions use it: switching disconnects them. The model stays reserved while it starts again; the other sessions then use the new model, or load theirs if it runs no model.
- A print, JSON or rpc run that loads the model keeps it to itself: it ends on its own, so it never waits for others.

The model is served to the other sessions on a Unix socket, `~/.saetta/run/inproc.sock`, in a folder only you can open, so other users of the Mac cannot connect to it at all. Every request also carries a random token from the holder's record (`~/.saetta/run/inproc.rec`, readable by you only), and requests from web pages are refused. Before each request a session checks that the session it joined is still there and that the program on the socket knows the token (it answers a challenge with an HMAC of it), so the conversation and the token never go to another program.

## Subagents

With `--subagents` (or `"subagents": true` in `settings.json`) the model gets a `subagent` tool. It hands a self-contained task to a child agent:

| argument | what it is |
|---|---|
| `description` | a short name for the task (3 to 5 words) |
| `prompt` | the task, with what the subagent should report back |
| `tools` | the tools the subagent may use (default: all of yours except `subagent`) |

- Several `subagent` calls in one answer run at the same time, up to three at once; the others wait for a free slot.
- A subagent sees the conversation so far, has the same tools and works on its own; it cannot start subagents. Its final message (at most 2000 lines and 50 KB) is the tool's result.
- The interactive screen shows each subagent's tool calls as they happen; Ctrl+O expands them, with the task and the answer. Escape stops the subagents too.
- In JSON mode each subagent's progress is a `tool_execution_update` event of its call.
- The session file keeps each subagent's messages in its result, so `/resume` and `/export` show them.

Subagents are off by default: pi has no built-in subagents, and with them off Saetta Code's prompt and tools stay pi's.

### How they use the engine

Each subagent's requests start with its parent's: the same system prompt, tools and messages, then its task. The engine reuses the part of the parent's prompt it has kept (in steps of 4,096 tokens, up to 32,768), so a subagent reads mostly its own task. The engine still decodes one request at a time: subagents overlap their tool calls and their waits, not their decoding.
