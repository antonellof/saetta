# LoRA adapters

PEFT LoRA adapters run on Qwen3.8-27B at run time, with vLLM's flags. The base pack stays as it is; the adapter is applied per request.

```sh
saetta download 27b-8bit && saetta download 27b-pi
saetta start code 27b-8bit --lora-modules qwen3.8-pi=27b-pi                # Saetta Code on it, no server
saetta start pi --pack 27b-8bit --lora-modules qwen3.8-pi=27b-pi           # pi on it
saetta serve 27b-8bit --lora-modules qwen3.8-pi=27b-pi \
    --default-chat-template-kwargs '{"enable_thinking": false}'            # or any client
```

A request whose `model` is the adapter's name runs with it, any other with the base model. `GET /v1/models` lists the adapters next to the base model.

An adapter is a name and a path: `--lora-modules name=path [name=path ...]` (also `name=path,name=path`) for `saetta serve`, `saetta start` and `saetta code`, and `--lora name=path` as a shorter form of the same. The path is a folder, a download name (`27b-pi`) or a Hugging Face id (`org/repo`) under the models folder, so an adapter fetched with `saetta download` works by its name everywhere. `saetta start` turns thinking off by default when it is given adapters, as the pi adapter was trained; `--default-chat-template-kwargs` overrides that.

Before loading, saetta checks each adapter against the base model and refuses one that does not fit, with the reason: the base model its `adapter_config.json` names, the rank, the target modules, DoRA, bias and saved modules. The tensors' shapes are checked when it loads.

## In Saetta Code

With the model in process (`saetta start code` or `saetta code --pack P`), the adapters load into Saetta Code's own engine; no server is started. The session runs on the first adapter; `--model NAME` picks another one, or the base model by its id (`qwen3.8-27b`).

```sh
saetta code --pack 27b-8bit --lora-modules qwen3.8-pi=27b-pi -p "say hi"
saetta start code 27b-8bit --lora-modules qwen3.8-pi=27b-pi \
    --default-chat-template-kwargs '{"enable_thinking": false}'
```

`{"enable_thinking": false}` starts the session with thinking off (`/thinking` changes it). `/model` lists the downloaded adapters under each base model they fit, marked "adapter of ..." in the list:

- an adapter loaded with the model, or its base model, is switched to in place, without loading anything;
- another one starts Saetta Code again on its base model with that adapter (the conversation goes on);
- one not downloaded shows its size and downloads from the list, then Saetta Code switches to it;
- one whose base model is not on this Mac is greyed, with the reason.

Adapters are found in the download list (`27b-pi`) and in any `org/repo` folder of the models folder that holds an `adapter_config.json`. Adapters load with the model, so a Saetta Code started with adapters does not join one that already runs the model.

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
