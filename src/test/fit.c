#include "../test.h"
#include "../download.h"
#include "../fit.h"

#include <stdlib.h>
#include <string.h>

// The memory rules (fit.h) on simulated Macs (SAETTA_RAM_GB), against the download table.
TEST(fit_rules_by_mac_memory) {
    const char *old = getenv("SAETTA_RAM_GB"), *oldi0 = getenv("SAETTA_IGNORE_MEMORY");
    char *keep = old ? strdup(old) : NULL, *oldi = oldi0 ? strdup(oldi0) : NULL;
    unsetenv("SAETTA_IGNORE_MEMORY");
    static const struct {
        const char *gb, *pick;
    } cases[] = {{"16", NULL}, {"24", NULL}, {"32", "27b-4bit"}, {"36", "27b-turbo"}, {"64", "27b-turbo"}, {"96", "coder"}, {"128", "coder"}};
    for (size_t i = 0; i < COUNT(cases); i++) {
        setenv("SAETTA_RAM_GB", cases[i].gb, 1);
        const char *p = fit_default_pack();
        EXPECT((p == NULL) == (cases[i].pick == NULL));
        if (p && cases[i].pick) EXPECT(strcmp(p, cases[i].pick) == 0);
    }
    // a pack of 15 GiB of weights (27b-4bit) fits 32 GB, not 24; 105.5 GiB (full) fits 128, not 96
    MappedFile f[1];
    Pack pk = {.files = f, .nfiles = 1};
    f[0].len = 15ull << 30;
    setenv("SAETTA_RAM_GB", "32", 1);
    EXPECT_OK(fit_check_pack(&pk));
    setenv("SAETTA_RAM_GB", "24", 1);
    EXPECT(fit_check_pack(&pk) != NULL);
    setenv("SAETTA_IGNORE_MEMORY", "1", 1);
    EXPECT_OK(fit_check_pack(&pk));
    unsetenv("SAETTA_IGNORE_MEMORY");
    f[0].len = (1055ull << 30) / 10;
    setenv("SAETTA_RAM_GB", "128", 1);
    EXPECT_OK(fit_check_pack(&pk));
    setenv("SAETTA_RAM_GB", "96", 1);
    EXPECT(fit_check_pack(&pk) != NULL);
    // each download pack's min_ram_gb agrees with the rule for its weights (sizes in GiB, 2026-11-21; the fine-tunes 2026-11-22)
    static const struct {
        const char *name;
        double gib;
    } sizes[] = {{"coder", 75.7}, {"full", 105.5}, {"27b-8bit", 27.5}, {"27b-4bit", 15.0}, {"27b-turbo", 18.6}, {"27b-turbo-8bit", 28.6},
                 {"27b-coder390", 21.8}, {"27b-coder390-8bit", 28.6}, {"27b-fable5", 18.5}, {"27b-fable5-8bit", 28.6},
                 {"27b-qwopus", 15.0}, {"27b-qwopus-8bit", 27.5}};
    for (size_t i = 0; i < COUNT(sizes); i++) {
        const DownloadPack *d = download_find(sizes[i].name);
        EXPECT(d != NULL);
        if (!d) continue;
        EXPECT(sizes[i].gib + FIT_RESERVE_GB <= (double)d->min_ram_gb);
    }
    if (keep) {
        setenv("SAETTA_RAM_GB", keep, 1);
        free(keep);
    } else
        unsetenv("SAETTA_RAM_GB");
    if (oldi) {
        setenv("SAETTA_IGNORE_MEMORY", oldi, 1);
        free(oldi);
    }
}
