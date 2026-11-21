// What fits on this Mac (see fit.h).
#include "fit.h"

#include "download.h"

#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>

uint64_t fit_ram_bytes(void) {
    // SAETTA_RAM_GB: act as a Mac with that much memory (checks the rules on a larger one)
    const char *sim = getenv("SAETTA_RAM_GB");
    if (sim && atoi(sim) > 0) return (uint64_t)atoi(sim) << 30;
    uint64_t total = 0;
    size_t len = sizeof total;
    if (sysctlbyname("hw.memsize", &total, &len, NULL, 0) != 0) return 0;
    return total;
}

uint64_t fit_pack_bytes(const Pack *pack) {
    uint64_t n = 0;
    for (size_t i = 0; i < pack->nfiles; i++) n += pack->files[i].len;
    return n;
}

static bool ignore_memory(void) {
    const char *v = getenv("SAETTA_IGNORE_MEMORY");
    return v != NULL && strcmp(v, "0") != 0;
}

void fit_suggest(uint64_t ram) {
    Buf b = {0};
    for (size_t i = 0; i < DOWNLOAD_NPACKS; i++) {
        const DownloadPack *p = &download_packs[i];
        if (p->min_ram_gb == 0 || ((uint64_t)p->min_ram_gb << 30) > ram) continue;
        buf_printf(&b, "%s%s", b.len ? ", " : "", p->name);
    }
    if (b.len)
        dbg("        Packs that fit this Mac: %.*s (saetta download <name>).\n", (int)b.len, b.ptr);
    else
        dbg("        No saetta model fits this Mac: the smallest, 27b-4bit, needs 32 GB.\n");
    buf_free(&b);
}

Err fit_check_pack(const Pack *pack) {
    const uint64_t gib = 1ull << 30, ram = fit_ram_bytes(), w = fit_pack_bytes(pack);
    if (ram == 0 || ignore_memory()) return OK;
    if (w + (uint64_t)FIT_RESERVE_GB * gib <= ram) return OK;
    // the smallest Apple Silicon memory size that holds it
    static const uint64_t sizes[] = {16, 18, 24, 32, 36, 48, 64, 96, 128, 192, 256, 512};
    uint64_t need = (w + (uint64_t)FIT_RESERVE_GB * gib + gib - 1) / gib, mac = need;
    for (size_t i = 0; i < COUNT(sizes); i++)
        if (sizes[i] >= need) {
            mac = sizes[i];
            break;
        }
    dbg("saetta: this model's weights are %.1f GB and need a Mac with %llu GB of memory or more; this one has %llu GB.\n",
        (double)w / (double)gib, (unsigned long long)mac, (unsigned long long)(ram / gib));
    fit_suggest(ram);
    dbg("        SAETTA_IGNORE_MEMORY=1 loads it anyway.\n");
    return "NotEnoughMemory";
}

const char *fit_default_pack(void) {
    const uint64_t gb = fit_ram_bytes() >> 30;
    if (gb >= 96) return "coder";
    if (gb >= 32) return "27b-turbo";
    return NULL;
}
