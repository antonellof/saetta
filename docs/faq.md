# FAQ

## Does Saetta work with my pi sessions and settings?

Yes. It reads and writes pi's `settings.json`, `models.json`, `trust.json` and `mcp.json`, and keeps its session files in the same folder and format as pi 1.5.2, so either one continues a session the other started. It also reads `AGENTS.md` and `CLAUDE.md` files, skills and prompt files as pi does. Run as `saetta code` it uses pi's folder (`~/.pi/agent`); `saetta start code` and `saetta start pi` use `~/.saetta/pi`. pi's TypeScript extensions are not supported. See [Saetta Code](saetta-code.md).

## Can Saetta use a model other than the local one?

Yes. By default the model runs inside the agent. It can also use a running Saetta server, or any OpenAI-compatible chat completions server listed in pi's `models.json`. `/model` switches between them and adds new ones. See [Models and providers](models-and-providers.md).

## Does Saetta support MCP servers?

Yes. It reads pi's `mcp.json` (global, or `.pi/mcp.json` in a trusted project) and connects to stdio and Streamable HTTP servers, with OAuth sign-in for HTTP servers. Their tools are offered to the model, and `/mcp` manages the servers. See [MCP](mcp.md).

## Can I use Claude Code with a local model?

Yes. `saetta start claude` runs Claude Code against Saetta's Anthropic-compatible API on your own Mac, without editing your Claude Code configuration. pi and opencode work the same way with `saetta start pi` and `saetta start opencode`. See [Claude Code](claude-code.md).

## Which Mac do I need?

Any Apple Silicon Mac (M1 or later) with 32 GB of memory or more and macOS 26 or later. 32 GB runs Qwen3.8-27B in 4-bit, 36 GB the 27B TURBO or Fable-5 coding fine-tunes, 48 GB the Coder390 fine-tune, 96 GB Qwen3.8-Flash-Next Coder and 128 GB the full Flash-Next. Saetta picks the one that fits. See [Choosing a model](choosing-a-model.md).

## Does my code leave my machine?

No. The model runs on your Mac, inside the agent or in a server that listens only on 127.0.0.1, so prompts and code stay local. The network is used to download the model weights from Hugging Face, for the agent's update check at start (`SAETTA_SKIP_VERSION_CHECK=1` turns it off), and for what you ask for: `/share`, a remote model or an MCP server you configure.

## Is Saetta free?

Yes. Saetta is open source under the MIT license. The models keep their own licenses: Qwen3.8-27B and its TURBO, Coder390 and Fable-5 fine-tunes are Apache-2.0, Qwen3.8-Flash-Next uses the Qwen Community License.

## How fast is it?

On an M5 Max with 128 GB, Qwen3.8-Flash-Next reads prompts at 1,600 to 1,950 tokens per second and answers at 66 to 85. Earlier chips run the same kernels on their GPU cores and are slower; they are not yet measured. See [Performance](performance.md).

## Which models does it run?

Qwen3.8-Flash-Next (the pruned Coder variant and the full model) and Qwen3.8-27B in 4-bit and 8-bit, with LoRA adapters and the TURBO, Coder390 and Fable-5 coding fine-tunes. Any Qwen3.8-27B GGUF can be converted with `saetta import`. See [GGUF import](gguf-import.md).

## Can two agents use the model at once?

Through the server, yes: several sessions of Claude Code, pi, opencode or Saetta Code share one server and take turns, and up to 4 conversations keep their state. A model loaded inside Saetta Code's own process serves only that agent; to share it, start a server first (`saetta serve coder`) and `saetta start code` joins it.

## Where are the logs?

Saetta Code: `~/.saetta/code.log`. A server started by `saetta start`: `~/Library/Logs/saetta/serve-<port>.log`. `saetta monitor` shows the server log with `l`.
