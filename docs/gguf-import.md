# GGUF import

Saetta runs Qwen3.8-27B fine-tunes published as GGUF files by converting them into a Saetta pack once.

## Fine-tunes in the download list

```sh
saetta download 27b-turbo          # or 27b-turbo-8bit, 27b-coder390, 27b-coder390-8bit, 27b-fable5, 27b-fable5-8bit
```

`saetta download 27b-turbo` (or `27b-coder390`, `27b-fable5`) fetches the GGUF, converts it into a Saetta pack and deletes it. The conversion needs free disk for both while it runs: about 40 GB for `27b-turbo`. The packs and their sizes are on [Choosing a model](choosing-a-model.md).

## Any Qwen3.8-27B GGUF

`saetta import <file.gguf> <folder>` does the same for any Qwen3.8-27B GGUF (llama.cpp `qwen35`; Q8_0, Q4_K, Q5_K, Q6_K, F16 or BF16 tensors), with the model's `config.json` and tokenizer files in the folder.

```sh
mkdir -p ~/Models/my-27b
cp config.json tokenizer.json tokenizer_config.json ~/Models/my-27b/    # from the fine-tune's (or the base model's) repository
saetta import my-27b-Q4_K_M.gguf ~/Models/my-27b
saetta start code --pack ~/Models/my-27b
```

`--like <folder>` reads `config.json` from another folder (a pack of the same model) instead of the output folder:

```sh
saetta import my-27b-Q8_0.gguf ~/Models/my-27b-8bit --like ~/Models/my-27b
```

## The MTP head

A GGUF's MTP layer becomes the draft head: Qwen3.8-27B speculates with it as Flash-Next does. `SAETTA_MTP=0` turns it off.
