# LoRA adapters

PEFT LoRA adapters run on Qwen3.8-27B at run time, with vLLM's flags. The base pack stays as it is; the adapter is applied per request.

```sh
saetta download 27b-8bit && saetta download 27b-pi
saetta start pi --pack 27b-8bit --lora qwen3.8-pi=27b-pi                    # pi on it
saetta serve 27b-8bit --lora-modules qwen3.8-pi=27b-pi \
    --default-chat-template-kwargs '{"enable_thinking": false}'            # or any client
```

A request whose `model` is the adapter's name runs with it, any other with the base model. `GET /v1/models` lists the adapters next to the base model.

The adapter is a name and a pack name or folder: `--lora name=adapter` for `saetta start`, `--lora-modules name=path ...` for `saetta serve`. `saetta start` with `--lora` turns thinking off by default, as the pi adapter was trained. In Saetta Code an adapter needs the server: `saetta start code --lora name=adapter` starts one.

## What is supported

LoRA on attention and MLP projections, rank a multiple of 8 up to 128, and rsLoRA. Adapters apply to the 27B packs; the `27b-8bit` pack is the one meant for them.

## The pi adapter

The [pi adapter](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted) (`27b-pi`, 0.6 GB) is trained to drive the [pi](https://pi.dev) agent (its card: SWE-bench Pro hard 39.2% to 64.7% over the base model).

## Testing an adapter

`SAETTA_LORA=<folder>` applies an adapter to `saetta generate`, which prints greedy tokens after a list of token ids:

```sh
saetta tokenize 27b-8bit prompt.txt                                   # the token ids of a text
SAETTA_LORA=~/Models/my-adapter saetta generate 27b-8bit 64 <token ids...>
```
