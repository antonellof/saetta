# Build from source

Saetta is C23 with hand-written Metal kernels and a small Objective-C shim. It builds with the Makefile, with no other dependency than Apple's frameworks.

## Requirements

Full Xcode with the Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`) and Apple clang 21 or later (C23).

```sh
xcodebuild -downloadComponent MetalToolchain
```

## Build

```sh
git clone https://github.com/antonellof/saetta.git && cd saetta
make                                    # ./build/bin/saetta
make test                               # unit tests (SAETTA_PACK=<pack> for real tensors)
make verify GOLDEN=<dir>/golden         # development: against reference outputs (not in this repository)
```

The binary is `./build/bin/saetta`; copy it anywhere on your `PATH`, or run it in place:

```sh
./build/bin/saetta start code
```

## Source layout

`src/`: `model.c` (Flash-Next), `dense.c` (27B), `model_api.c` picks one per pack; `kernels/` Metal; `server.c`, `tokenizer.c`, `chat.c`, `toolcall.c`; `code*.c` Saetta Code. `saetta help` lists every command.

## The website and these docs

The site is the static folder `web/`. These pages are generated from the Markdown in `docs/` by `web/docs-src/build.mjs`; see [docs/README.md](README.md).

## License

[MIT](../LICENSE). The engine adapts parts of the MIT-licensed projects sushi, qwental, Strata, ds4 and llama.cpp. Saetta Code is a port of pi (MIT) and adapts parts of the libraries pi builds on (jsdiff, TypeBox, openai-node, highlight.js, marked, OpenTUI). Each adapted file names its source, and every notice is in [LICENSE](../LICENSE).
