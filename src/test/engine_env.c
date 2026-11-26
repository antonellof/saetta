// Every environment switch the engine's files read (getenv("...") and dense.c's env_on("...")
// with a literal name) is in src/engine_env.h, either as one that changes computed rows (hashed
// into a saved state's shape) or as one that does not. Reads the sources from src/ (make test runs
// from the repository's root).
#include "../engine_env.h"
#include "../test.h"

#include <dirent.h>
#include <stdlib.h>

static bool listed(const char *name) {
    for (size_t i = 0; i < COUNT(engine_env_rows); i++)
        if (!strcmp(engine_env_rows[i], name)) return true;
    for (size_t i = 0; i < COUNT(engine_env_other); i++)
        if (!strcmp(engine_env_other[i], name)) return true;
    return false;
}

// Saetta Code's own files and the tests are not the engine.
static bool engine_file(const char *name) {
    size_t n = strlen(name);
    if (n < 3 || strncmp(name, "code_", 5) == 0) return false;
    return !strcmp(name + n - 2, ".c");
}

// The literal names read in text: getenv("X") and env_on("X"); *calls counts every getenv( and
// env_on( call, *named those with a literal name.
static int unlisted_in(const char *path, const char *text, size_t *calls, size_t *named) {
    static const char *const pats[] = {"getenv(", "env_on(", "env_or("};
    int bad = 0;
    for (size_t p = 0; p < COUNT(pats); p++) {
        const char *s = text;
        while ((s = strstr(s, pats[p])) != NULL) {
            const char *q = s + strlen(pats[p]);
            // a definition or declaration of env_on itself, not a call
            if (p > 0 && strncmp(q, "const char", 10) == 0) {
                s = q;
                continue;
            }
            *calls += 1;
            if (*q == '"') {
                const char *e = strchr(q + 1, '"');
                if (e) {
                    char name[128];
                    size_t n = (size_t)(e - q - 1);
                    if (n < sizeof name) {
                        memcpy(name, q + 1, n);
                        name[n] = 0;
                        *named += 1;
                        if (!listed(name)) {
                            dbg("%s: %s is in neither list of src/engine_env.h\n", path, name);
                            bad++;
                        }
                    }
                }
            }
            s = q;
        }
    }
    return bad;
}

TEST(engine_env_every_switch_is_listed) {
    DIR *d = opendir("src");
    if (!d) SKIP("no src/ in the working directory");
    int bad = 0;
    size_t files = 0, calls = 0, named = 0;
    struct dirent *de;
    while ((de = readdir(d))) {
        if (!engine_file(de->d_name)) continue;
        char path[512];
        snprintf(path, sizeof path, "src/%s", de->d_name);
        char *text;
        size_t len;
        if (read_file(path, 64u << 20, &text, &len)) continue;
        files++;
        bad += unlisted_in(path, text, &calls, &named);
        free(text);
    }
    closedir(d);
    char *shim;
    size_t sl;
    if (!read_file("src/metal/shim.m", 64u << 20, &shim, &sl)) {
        files++;
        bad += unlisted_in("src/metal/shim.m", shim, &calls, &named);
        free(shim);
    }
    EXPECT(files > 20);
    EXPECT(named > 30);
    EXPECT_EQ(bad, 0);
    // the calls without a literal name are the helpers (base.c env_or, dense.c env_on) and the
    // state shape's own loop over engine_env_rows (server.c): no more, or a switch could hide there
    EXPECT_EQ(calls - named, 3);
    // no switch in both lists
    for (size_t i = 0; i < COUNT(engine_env_rows); i++)
        for (size_t j = 0; j < COUNT(engine_env_other); j++) EXPECT(strcmp(engine_env_rows[i], engine_env_other[j]) != 0);
}
