// What fits on this Mac: saetta runs on any Apple Silicon Mac, and the model packs need from
// 32 GB of memory (Qwen3.8-27B 4-bit) to 128 GB (Flash-Next with all 512 experts). The rule is
// physical memory: the weights (mapped from the files, no copy) plus FIT_RESERVE_GB for macOS,
// the GPU scratch and the smallest context. Metal's recommended working set is not the limit:
// on a 128 GB M5 Max it is 107.5 GiB and the 105.5 GiB full pack runs.
#ifndef SAETTA_FIT_H
#define SAETTA_FIT_H

#include "base.h"
#include "weights.h"

#define FIT_RESERVE_GB 12 // macOS, other programs, GPU scratch, the smallest context
#define FIT_KV_RESERVE_GB 8 // what the context may not take: RAM - this - weights is its budget

// Physical memory in bytes (hw.memsize; SAETTA_RAM_GB=N acts as a Mac with N GB).
uint64_t fit_ram_bytes(void);

// Bytes of the pack's mapped weight files.
uint64_t fit_pack_bytes(const Pack *pack);

// Prints the download packs that fit a Mac with `ram` bytes.
void fit_suggest(uint64_t ram);

// "NotEnoughMemory" (after a message naming the memory it needs) when the pack's weights plus
// FIT_RESERVE_GB exceed this Mac's memory; SAETTA_IGNORE_MEMORY=1 loads it anyway.
Err fit_check_pack(const Pack *pack);

// The download name to suggest on this Mac: "coder" with 96 GB or more, else "27b-turbo"
// (32 GB or more), else NULL.
const char *fit_default_pack(void);

#endif
