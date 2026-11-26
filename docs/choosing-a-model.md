# Choosing a model

A pack is a model folder Saetta runs. `saetta download <name>` fetches one at a pinned revision into `~/Models` (or `SAETTA_MODELS`). Commands take a pack by name (`saetta serve coder`) or by folder.

## By memory

`saetta download` with no name picks by memory: `coder` (Flash-Next, 76 GB) with 96 GB or more, `27b-turbo` (Qwen3.8-27B fine-tune, 20 GB, converted from its GGUF: about 40 GB free disk while it runs) with 36 GB, `27b-4bit` (16 GB) with 32 GB.

A pack that cannot run on this Mac is refused, and the context shrinks to what memory holds: about 37K tokens with `27b-4bit` on 32 GB. `SAETTA_IGNORE_MEMORY=1` skips the memory checks.

When no pack is named, `saetta start` uses the first downloaded pack that fits this Mac, in this order: coder, full, then the 27B packs.

## The packs

| name | weights | on disk | Mac memory | options | SWE-bench | Terminal-Bench 2.1 |
|---|---|---|---|---|---|---|
| **Qwen3.8-Flash-Next** | | | | | | |
| **`coder`** | [Flash-Next Coder](https://huggingface.co/mbrown-uno/Qwen3.8-Flash-Next-GSQ-RCO-Coder-MLX), 256 experts, mixed 4/5/6/8-bit | 76 GB | 96 GB | draft head (MTP) | Verified 79.2 ¹ | n/p |
| `full` | [Flash-Next](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit-mtp), 512 experts, 4-bit | 105 GB | 128 GB | draft head (MTP) | Pro 62.5 ², Verified 82.8 ¹ | n/p |
| **Qwen3.8-27B** | | | | | | |
| **`27b-4bit`** | [Qwen3.8-27B](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit), 4-bit | 16 GB | 32 GB | | Pro 61.7 ² | 73.0 ² |
| `27b-8bit` | [Qwen3.8-27B](https://huggingface.co/mlx-community/Qwen3.8-27B-8bit), 8-bit | 29 GB | 48 GB | LoRA adapters | Pro 61.7 ² | 73.0 ² |
| `27b-pi` | [pi adapter](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted) (LoRA, for `27b-8bit`) | 0.6 GB | | trained to drive pi | Pro hard-51 64.7 ³ | 55.8 ³ |
| **`27b-turbo`** | [TURBO Fable Cold Fusion](https://huggingface.co/DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF), shorter thinking, from Q4_K_M | 20 GB | 36 GB | draft head (MTP) | n/p | n/p |
| `27b-turbo-8bit` | the same, from Q8_0 | 30 GB | 48 GB | draft head (MTP) | n/p | n/p |
| `27b-coder390` | [Coder390 EfficientThink](https://huggingface.co/nerkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-GGUF-NInfer), SFT and RL on code, from Q4 | 23 GB | 48 GB | draft head (MTP) | n/p | n/p |
| `27b-coder390-8bit` | the same, from Q8_0 | 31 GB | 48 GB | draft head (MTP) | n/p | n/p |
| `27b-fable5` | [Fable-5 Coding Distilled](https://huggingface.co/khazarai/Qwen3.8-27B-Fable-5-Coding-Distilled), from [imatrix Q4_K_M](https://huggingface.co/mradermacher/Qwen3.8-27B-Fable-5-Coding-Distilled-i1-GGUF) | 20 GB | 36 GB | draft head (MTP) | n/p | n/p |
| `27b-fable5-8bit` | the same, from Q8_0 | 31 GB | 48 GB | draft head (MTP) | n/p | n/p |
| `27b-qwopus` | [Qwopus3.8-27B Coder-Flash](https://huggingface.co/mlx-community/Qwopus3.8-27B-Coder-Flash-4bit), distilled from Qwen3.8-Flash-Max, 4-bit | 16 GB | 32 GB | | n/p ⁵ | n/p ⁵ |
| `27b-qwopus-8bit` | [the same](https://huggingface.co/mlx-community/Qwopus3.8-27B-Coder-Flash-8bit), 8-bit | 29 GB | 48 GB | | n/p ⁵ | n/p ⁵ |
| **Inkling 2 Mini** | | | | | | |
| `inkling` | [Inkling 2 Mini](https://huggingface.co/mlx-community/Inkling-2-Mini-4bit), dense 4-bit, experts NVFP4 / MXFP8 | 17 GB | 32 GB | prompt lookup; 1M context | Verified 79.2 ⁴ | 71.6 ⁴ |
| `inkling-dspark` | [DSpark drafter](https://huggingface.co/thinkingmachines/Inkling-2-Mini) for `inkling` | 0.6 GB | | opt-in, `SAETTA_DSPARK=1` | | |

SWE-bench and Terminal-Bench are the publishers' scores for the full-precision model, each with its own harness, not measured by Saetta on these packs:

1. [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) (SWE-bench Verified, xhigh reasoning; the Coder against the full model).
2. [Qwen](https://huggingface.co/Qwen/Qwen3.8-27B) ([Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next); SWE-bench Pro with Claude Code, Terminal-Bench 2.1 with Terminus).
3. [harness-hub](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted) (the adapter under pi, SWE-bench Pro's hard 51 tasks; the base model scores 39.2 and 39.5 there).
4. [Thinking Machines](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Inkling 2 Mini: SWE-bench Verified with bash only, Terminal-Bench 2.1 with pi).
5. [Jackrong](https://huggingface.co/Jackrong/Qwopus3.8-27B-Coder-Flash) publishes only its own local tests for Qwopus (an MMLU-Pro subset of 1,500 questions, 93.87; 9 of 9 runs of a 3-task coding-agent test), not comparable with these.

n/p: not published (Qwen publishes no Terminal-Bench score for Flash-Next; the fine-tunes publish none for either).

## Downloading

```sh
saetta download coder            # one pack
saetta download all              # coder and full
saetta download 27b-turbo --jobs 8
saetta info coder                # what a downloaded pack holds
```

Downloads are resumable and hash-checked, with 4 parallel file downloads by default (`--jobs N`). `27b-turbo`, `27b-coder390` and `27b-fable5` (and their 8-bit variants) fetch a GGUF, convert it into a Saetta pack and delete it; see [GGUF import](gguf-import.md).

Inside Saetta Code, `/model` lists every pack, shows its size and the free disk, and downloads the one you pick.

## Licenses

Flash-Next: Qwen Community License 1.0. Qwen3.8-27B, the pi adapter, TURBO, Coder390, Fable-5, Qwopus and Inkling 2 Mini: Apache-2.0. No weights are part of the Saetta repository.
