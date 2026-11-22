# HTTP API

`saetta serve` runs the engine as a local server with OpenAI- and Anthropic-compatible APIs, for any client the `saetta start` agents do not cover.

```sh
SAETTA_CTX=204800 saetta serve coder     # loads in 5 to 10 s, one request at a time, Ctrl-C stops
```

The server listens on `127.0.0.1`, port 8080 (`--port N`). Any API key is accepted. One request runs at a time; up to 4 conversations keep their state between turns (`SAETTA_CONVERSATIONS`), so a follow-up turn reads only what is new.

## Endpoints

| method and path | API |
|---|---|
| `POST /v1/chat/completions` | OpenAI chat completions: streaming, `tool_calls`, `reasoning_content` |
| `POST /v1/messages` | Anthropic Messages: streaming, thinking, tool use |
| `POST /v1/messages/count_tokens` | Anthropic token count |
| `GET /v1/models` | the model id (with `max_model_len`, the context) and the LoRA adapters loaded |
| `GET /health` | `{"status":"ok"}` once the model is loaded |

Base URLs for clients: `http://127.0.0.1:8080/v1` (OpenAI) and `http://127.0.0.1:8080` (Anthropic).

## Model ids

| pack | model id |
|---|---|
| `coder`, `full` | `qwen3.8-flash-next` |
| the 27B packs | `qwen3.8-27b` |
| a LoRA adapter | the name given to `--lora-modules` |

A request whose `model` is an adapter's name runs with that adapter; any other name runs the base model.

## Chat completions

```sh
curl -sN localhost:8080/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "Write is_prime in Python"}],
  "stream": true
}'
```

Sampling fields and their defaults: `max_tokens` or `max_completion_tokens` (4096), `temperature` (1.0), `top_p` (0.95), `top_k` (20), `seed`, `stop`. `temperature: 0` is greedy decoding, with speculative decoding (the MTP head and prompt lookup) and the same output as without it. Sampled requests use speculative sampling, which keeps the same output distribution.

`tool_choice`: `none` leaves the tools out of the prompt; `required`, or a named function, starts the answer with a tool call. The usage block reports reused prompt tokens as `prompt_tokens_details.cached_tokens`.

### chat_template_kwargs

The chat template's options go in `chat_template_kwargs` (or at the top level of the request):

| key | values | effect |
|---|---|---|
| `enable_thinking` | `true` (default), `false` | whether the model thinks before answering |
| `reasoning_effort` | `low`, `medium`, `xhigh` (default) | how long it thinks |

```sh
curl -s localhost:8080/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "Rename foo to bar in this function: ..."}],
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

`saetta serve --default-chat-template-kwargs '{"enable_thinking": false}'` sets defaults for every request; a key the request sets wins.

## Anthropic Messages

`/v1/messages` takes Anthropic's request format, including `system`, tools, `tool_choice` (`auto`, `any`, `tool`, `none`) and streaming events. `thinking: {"type": "disabled"}` turns thinking off, and `output_config.effort` maps to `reasoning_effort` (`low`, `medium`, anything else `xhigh`). This is the API [Claude Code](claude-code.md) uses.

```sh
curl -s localhost:8080/v1/messages -H 'content-type: application/json' -d '{
  "model": "qwen3.8-flash-next",
  "max_tokens": 1024,
  "messages": [{"role": "user", "content": "What does git rebase --onto do?"}]
}'
```

## Not supported

Image input (on `/v1/messages`, an image or PDF becomes a short note saying it was omitted), Anthropic-hosted tools such as web search, and embeddings.

## Server options

| option or variable | effect |
|---|---|
| `--port N` | the port (8080) |
| `--lora-modules name=path ...` | LoRA adapters to serve ([LoRA adapters](lora.md)) |
| `--default-chat-template-kwargs JSON` | template defaults for every request |
| `SAETTA_CTX` | context in tokens (32768), 1024 to 262144 |
| `SAETTA_CONVERSATIONS` | conversations kept between turns (4) |

All variables are on [Environment variables](environment.md). `saetta render request.json` prints the prompt a chat request renders to, to check a client's requests.
