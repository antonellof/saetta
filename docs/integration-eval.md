# Integration evaluation

Saetta runs the model and the coding agent in one process: Saetta Code calls the engine directly, instead of sending text to a server over HTTP. This page measures what that buys, against the usual setup where an agent talks to an inference server through an OpenAI-compatible API (GitHub issue #5).

Measured on an M5 Max with 128 GB, macOS 27.2, Saetta 0.26.0, greedy decoding, on AC power with the machine otherwise idle, one run at a time. Cells marked "1 run" are single runs; the others are medians of 2 or 3 runs after a discarded warm-up, and their runs agreed within 1%.

## Summary

- **Where integration pays:** restarts, forks and retries. On the Coder session workload, the integrated agent takes 89.8 s against 166.0 s for the same agent over HTTP to a warm `saetta serve`, and 255.3 s to a cold one. On the 27B, with the same session token for token, it is 279.0 s against 373.9 s.
- **What each feature is worth:**
  - State on the SSD: 252.9 to 89.4 s on the session workload.
  - Read ahead: 44.5 to 35.3 s on the stream workload. The first token after a tool result comes in 0.77 s instead of 2.95.
  - Fork point: 95.1 to 93.1 s with subagents.
  - Feed: no measured gain on this build.
- **Where it does not:** single-process sessions with no restart, fork or retry. That covers the task set (30 of 30 passed in every setup; seconds within 5% of in process for Saetta Code over HTTP, the answers differing) and Inkling, which has none of the features yet (in process and over HTTP within 1%).
- **Against other engines:**
  - mlx-lm on the 27B (same weights): 289.9 s on the session against 279.0 integrated, with 96.6 GB of memory against 27.5.
  - Strata on the full Flash-Next (different weights): 130.3 s against 118.1 on the session. On the edit workload Strata is faster, 68.5 s against 86.4.
- **Gaps found:**
  - On a fork, `saetta serve` copies only the start of the shared history into the new conversation (1,345 of 50,191 tokens on the Coder) and reads the rest again. With one conversation (the full pack) it rereads the whole prompt.
  - Saetta's Inkling path restarts some requests from zero.

## Setups

| setup | agent | model reached | what it has |
|---|---|---|---|
| integrated | Saetta Code | in process (`saetta code --pack`) | everything below |
| feature off | Saetta Code | in process | one of `SAETTA_STATE=0`, `SAETTA_AHEAD=0`, `SAETTA_FEED=0`, `SAETTA_FORK_POINT=0`, or all four |
| HTTP, warm server | Saetta Code or pi 1.5.2 | `saetta serve` over HTTP, one server for the whole session | the server's own prefix reuse and checkpoints |
| HTTP, cold server | Saetta Code | `saetta serve`, restarted whenever the agent restarts | the same, but nothing survives a restart |
| other engines | pi or Saetta Code | mlx-lm's server, llama.cpp's server, Strata's server | their own prompt caches |

The integration features:

- **State on the SSD** (`SAETTA_STATE`): the conversation's KV cache and recurrent state are saved at each turn end, and a resumed session reads only its new message.
- **Checkpoints at message boundaries**: a fork or a retry restarts at the nearest checkpoint, in the same conversation. They are on in all setups, but over HTTP a fork goes to another conversation (see the Coder results).
- **Read ahead** (`SAETTA_AHEAD`): a running command's output is read into the conversation while the command runs.
- **Feed** (`SAETTA_FEED`): files the agent reads, edits or writes feed speculative decoding.
- **Fork point** (`SAETTA_FORK_POINT`): with subagents, the engine keeps a checkpoint where the children's prompts leave the parent's.

## Workloads

- **session**: 15 turns over a C source tree, reaching about 80K tokens. It has reads, a grep, an edit, three agent restarts that resume the session, a `/fork` from an earlier message and a retry of the last turn.
- **stream**: 5 turns whose bash commands print their output over 1 to 3 seconds, as a build or a test run does.
- **edit**: 6 turns of renames, whole-function replacements and a file written by copying functions.
- **subagent**: 5 turns, 3 of them handing reads to two or three subagents at once.
- **tasks**: 30 small repository tasks (Python and C) with hidden checks, one print-mode run each.

The prompts name the exact tool and range, so every setup reads the same files. The answers are the model's own: with greedy decoding a setup repeats itself, but two setups can still word an answer differently (see the caveats). "Same session" in the tables means the setup wrote, token for token, the session the integrated setup wrote.

## Qwen3.8-Flash-Next Coder

### Integrated against HTTP (session workload)

| setup | seconds | prompt tokens read | first token after the 4 restarts, summed (s) | fork turn (s) | retry turn (s) | engine peak memory (GB) |
|---|---:|---:|---:|---:|---:|---:|
| integrated (1 run) | 89.8 | 77,589 | 2.6 | 1.9 | 4.7 | 20.7 |
| Saetta Code, warm `saetta serve` (1 run) | 166.0 | 216,558 | 28.0 | 34.5 | 51.4 | 36.2 |
| pi, warm `saetta serve` (1 run) | 145.1 | 153,724 | 27.8 | 34.2 | 17.5 | 32.5 |
| Saetta Code, cold `saetta serve` (1 run) | 255.3 | 387,213 | 126.7 | 35.2 | 49.7 | 22.6 |

Turns 1 to 12 read the same prompts in the integrated and warm-server runs. From turn 7 the answers differ by a few tokens, and the warm server is 4.5% faster over those 12 turns (73.1 against 76.6 s). The gap is the fork and the retry:

- The integrated engine forks and retries from a checkpoint in the same conversation, reading 40 and 39 tokens.
- The server sends the fork to a new conversation. It copies only the first 1,345 of the 50,191 shared tokens there and reads the rest again: 59,217 tokens, 34.5 s instead of 1.9 s.
- The retry then reads 79,676 tokens: 51.4 s instead of 4.7 s.

The 4 restarts are the three resumes and the process that opens the fork; the fork's first token is most of the warm servers' 28 s. A warm server resumes as fast as the integrated engine, because it keeps the conversation in memory between agent restarts. A cold server reads the whole history again at each of the three resumes.

Seconds are the sum of the 15 turns. The integrated agent also loads the model at each of its 5 starts, 3.1 to 7.1 s each, which a warm server pays once.

### Each feature switched off

| workload | integrated | feature off | all four off | what changes |
|---|---:|---:|---:|---|
| session | 89.4 s | no state: 252.9 s | 252.9 s | first token of a resume, fork or retry 0.5 to 1.6 s instead of 17.6 to 42.3 s; 77,589 prompt tokens read over the session instead of 387,213 |
| stream | 35.3 s | no read ahead: 44.5 s | 44.4 s | first token after a tool result 0.77 s instead of 2.95 s (median); 12,244 prompt tokens read at request time instead of 29,136 |
| edit | 99.1 s | no feed: 84.3 s | 84.1 s | different answers (7,034 against 5,864 generated tokens), same decode speed (81.3 and 81.2 tok/s) |
| subagent | 93.1 s | no fork point: 95.1 s | 94.3 s | the same session; 2 s less |

These are medians of 3 runs. In the stream and subagent workloads, the setup with a feature off wrote the same session as integrated, so the difference in seconds is that feature alone. In the session workload, the answers after the restarts differ a little: 2,113 generated tokens with the state off against 2,588. Most of the 163 s difference is the 309,624 extra prompt tokens. In the edit workload the answers differ more.

Feed changes which drafts are tried. Greedy decoding with speculation can then settle a near tie differently, so the answers differ. On this workload and build, feed measured no decode gain (2.05 against 2.12 tokens per forward pass); it bought 81.7 to 87.6 tok/s when it was added in 0.24.0.

Saving the state writes 7.7 GB to the SSD per session. It costs no memory: peak 20.7 GB with it, 22.6 GB without (reading the history again needs larger prompt buffers), time average 12.0 against 14.2 GB.

### Task set (30 tasks, 1 run each)

| setup | passed | seconds | malformed tool calls | failed edits |
|---|---:|---:|---:|---:|
| integrated | 30 | 806 | 6 | 0 |
| integrated, all four off | 30 | 823 | 2 | 0 |
| Saetta Code over HTTP | 30 | 767 | 2 | 2 |
| pi over HTTP | 30 | 873 | 4 | 0 |

The tasks are short (about 2,500 prompt tokens read per task) and run in one process with no restarts or forks, so the integration features have little to act on. The spread in seconds comes from the answers, which differ between setups.

## Qwen3.8-27B 4-bit

Here mlx-lm's server serves the same weights (the mlx-community 4-bit pack) and can be compared directly.

| workload | integrated | Saetta Code, warm `saetta serve` | pi, warm `saetta serve` | pi, mlx-lm | Saetta Code, mlx-lm |
|---|---:|---:|---:|---:|---:|
| session | 279.0 s | 373.9 s (same session) | 366.6 s | 289.9 s | 310.2 s |
| fork turn | 7.8 s | 82.1 s | 81.2 s | 23.9 s | 24.0 s |
| retry turn | 31.9 s | 53.4 s | 50.8 s | 18.3 s | 26.4 s |
| stream | 70.2 s | 80.3 s (same session) | 174.2 s | 113.4 s | 85.7 s |
| edit | 367.2 s | 389.3 s | 255.5 s | 423.7 s | 394.8 s |
| subagent | 189.9 s | 195.2 s (same session) | | | 220.4 s |
| engine peak memory, session | 27.5 GB | 50.1 GB | 50.3 GB | 96.6 GB | 91.1 GB |
| decode, session | 27.5 tok/s | 27.5 tok/s | 28.3 tok/s | 26.0 tok/s | 25.8 tok/s |

Medians of 3 runs for session and stream, and for edit integrated and Saetta Code over `saetta serve`; 2 for the other edit cells and for subagent.

Over HTTP, Saetta Code writes the same session as in process, so the difference is integration alone: 95 s on the session (the fork 74 s, the retry 22 s), 10 s on the stream workload (read ahead), 5 s with subagents.

mlx-lm keeps a prompt cache that finds the fork's prefix (10K tokens read, against the 60K `saetta serve` reads) and the retry's exact prompt. On the fork and retry turns it is closer to the integrated engine than `saetta serve` is. On the stream workload it is not: 85.7 s with Saetta Code against 80.3 s on `saetta serve` and 70.2 s integrated. It decodes a little slower, and its memory grows with the prompts it caches: a 96.6 GB peak on the session, against 27.5 GB integrated.

The pi rows and the edit rows differ in their answers (pi's requests are not byte for byte Saetta Code's, and the answers' lengths differ, 3.3 times on the 27B stream workload). Compare them by tokens read and decode speed, not seconds.

## Inkling 2 Mini (1 run each)

None of the integration features works for Inkling yet: no state on the SSD, no read ahead, no feed. This is the control.

| workload | integrated | Saetta Code over HTTP | pi over HTTP | pi, llama.cpp (UD-Q4_K_XL GGUF) |
|---|---:|---:|---:|---:|
| session | 617.2 s | 613.5 s (same session) | 679.2 s | 789.8 s |
| edit | 184.1 s | 184.1 s (same session) | 180.3 s | 230.1 s |
| prompt tokens read, session | 805,513 | 798,040 | 903,681 | 135,261 |
| decode, session | 54.9 tok/s | 54.3 tok/s | 54.5 tok/s | 36.9 tok/s |

With no feature to use, in process and over HTTP take the same time (within 1%), which is what integration predicts.

Saetta reads six times more prompt tokens than llama.cpp on this session. Some requests that extend the conversation start a new one from zero. Inkling has no checkpoints to restart from, so they read the whole prompt again. This is a gap in Saetta's Inkling path, separate from integration. Saetta is still faster in time, because it reads prompts faster.

## Qwen3.8-Flash-Next (full pack) against Strata

Strata is the established engine for Flash-Next on Apple Silicon. It runs Unsloth's UD-Q4_K_XL GGUF with its MTP head, through its own OpenAI-compatible server. Saetta runs the mlx-community 4-bit pack with its MTP head. The weights and quantization differ, so this compares engines and setups, not integration alone. 1 run each.

| | integrated (Saetta) | pi, warm `saetta serve` | pi, Strata |
|---|---:|---:|---:|
| session | 118.1 s | 244.3 s | 130.3 s |
| prompt tokens read, session | 81,818 | 300,908 | 134,362 |
| fork turn | 2.8 s | 38.5 s | 34.1 s |
| retry turn | 12.0 s | 57.3 s | 9.6 s |
| first token after the 4 restarts, summed | 7.5 s | 78.3 s | 27.0 s |
| decode, session | 85.6 tok/s | 82.8 tok/s | 85.1 tok/s |
| edit | 86.4 s | 102.9 s | 68.5 s |
| decode, edit | 93.4 tok/s | 86.6 tok/s | 103.6 tok/s |
| engine peak memory, session | 20.6 GB | 21.1 GB | 25.9 GB |

On the session, the integrated engine is 12 s ahead of Strata. The gain is the fork (2.8 against 34.1 s) and the restarts. Strata's prompt cache finds the retry's prompt, as mlx-lm's does.

On the edit workload, Strata is faster: it decodes at 103.6 tok/s against 93.4, and its answers were 4% shorter (5,744 against 6,011 tokens).

`saetta serve` does worse than both here. With the full pack and a 204,800-token context, it fits one conversation on this Mac. A fork or a retry then finds no conversation holding the shared start and reads the whole prompt again: 61,697 tokens for the fork, 82,053 for the retry. That is a gap in the server, not in the integrated path.

## What is not integration

- **Agents.** pi (Node) and Saetta Code (C) are different programs, and they do not send identical requests: pi's answers differ from Saetta Code's on the same engine. Saetta Code over HTTP is the like-for-like decoupled setup. pi's own process takes about 0.1 GB, Saetta Code's 0.01 GB.
- **Answers.** Greedy decoding with speculation is not identical across setups that draft differently (feed on or off, another engine), so sessions can differ. The tables mark which setups wrote the same session.
- **Model loads.** The integrated agent loads the model at every start (3 to 4 s on the Coder, 1.4 s on the 27B from the page cache). These are left out of the turn seconds, and a warm server pays them once.
- **Server context.** `saetta serve` alone defaults to a 32,768-token context, where these sessions would compact. Every HTTP session run here uses 204,800, the context `saetta start` and Saetta Code with `--pack` use. The Coder task-set HTTP runs used the 32,768 default; none of those sessions compacted.
- **Other engines run other files.** mlx-lm runs the same 27B pack. llama.cpp runs Unsloth's UD-Q4_K_XL GGUF of Inkling, not the MLX 4-bit pack (17.6 against 17.3 GB). Strata runs the UD-Q4_K_XL GGUF of the full Flash-Next, not the MLX 4-bit pack.
- **Memory.** Engine memory is the process's physical footprint, sampled every 0.5 s. Weights an engine maps from a file are not all counted, so compare engines by trend, not by absolute GB.
- **GPU time.** The GPU busy seconds in the raw results come from the driver's utilisation counter, sampled once a second over the whole GPU. They run 0 to 25% below the turn seconds (lowest where a turn waits on a running command) and are not listed above.

## Not measured

- The Flash-Next Coder and full packs on mlx-lm: mlx-lm 0.32.2 has no model file for their architecture.
- llama.cpp on the Qwen3.8 packs: no GGUF of the same weights is on this machine.
- Strata on Inkling: the installed Strata build for Inkling has no server mode.
- Strata on the Coder pack: Strata has no conversion of the Coder variant here.
- The 27B task set, and the HTTP runs of the Coder stream workload: left out to keep the run short.
- A 32 GB Mac.
