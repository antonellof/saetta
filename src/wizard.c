// Guided start (see wizard.h).
#include "wizard.h"

#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <termios.h>
#include <unistd.h>

#include "download.h"
#include "fit.h"
#include "launch.h"

// ---- terminal --------------------------------------------------------------------------------

static struct termios saved;
static volatile sig_atomic_t raw_on;

static void raw_off(void) {
    if (raw_on) {
        tcsetattr(0, TCSAFLUSH, &saved);
        raw_on = 0;
    }
    fputs("\x1b[?25h\x1b[?7h", stdout); // cursor and line wrap back
    fflush(stdout);
}

static void on_signal(int sig) {
    if (raw_on) {
        tcsetattr(0, TCSAFLUSH, &saved);
        raw_on = 0;
    }
    (void)!write(1, "\x1b[?25h\x1b[?7h\n", 13);
    signal(sig, SIG_DFL);
    raise(sig);
}

static void raw_enter(void) {
    struct termios t = saved;
    t.c_lflag &= ~(tcflag_t)(ICANON | ECHO);
    t.c_cc[VMIN] = 1;
    t.c_cc[VTIME] = 0;
    tcsetattr(0, TCSAFLUSH, &t);
    raw_on = 1;
    // hide the cursor and stop line wrap while a menu is up: a wrapped row would break the
    // cursor-up redraw
    fputs("\x1b[?25l\x1b[?7l", stdout);
    fflush(stdout);
}

enum { K_UP, K_DOWN, K_ENTER, K_QUIT, K_YES, K_NO, K_OTHER };

static int read_key(void) {
    unsigned char c;
    if (read(0, &c, 1) != 1) return K_QUIT;
    switch (c) {
    case '\r':
    case '\n': return K_ENTER;
    case 'q':
    case 'Q': return K_QUIT;
    case 'k': return K_UP;
    case 'j': return K_DOWN;
    case 'y':
    case 'Y': return K_YES;
    case 'n':
    case 'N': return K_NO;
    case 27: {
        // an arrow is ESC [ A / B; a lone ESC quits
        struct pollfd p = {.fd = 0, .events = POLLIN};
        unsigned char s[2];
        if (poll(&p, 1, 50) <= 0 || read(0, &s[0], 1) != 1) return K_QUIT;
        if (poll(&p, 1, 50) <= 0 || read(0, &s[1], 1) != 1) return K_OTHER;
        if (s[0] == '[' && s[1] == 'A') return K_UP;
        if (s[0] == '[' && s[1] == 'B') return K_DOWN;
        return K_OTHER;
    }
    default: return K_OTHER;
    }
}

typedef struct {
    const char *label, *status, *detail; // status: a short column ("downloaded", "needs 96 GB")
    bool off;                            // shown dimmed, not selectable
} Item;

static size_t term_cols(void) {
    struct winsize w;
    if (ioctl(1, TIOCGWINSZ, &w) == 0 && w.ws_col >= 20) return w.ws_col;
    return 80;
}

// One row, never wider than the terminal (the redraw counts one row per item): label (16),
// status (24), then as much of the detail as fits.
static void draw_item(const Item *it, bool sel) {
    const size_t cols = term_cols() - 1, lw = 16, sw = 24;
    char row[512];
    int n = snprintf(row, sizeof row, "%-*s%-*s%s", (int)lw, it->label, (int)sw, it->status, it->detail);
    size_t len = n < 0 ? 0 : (size_t)n < sizeof row ? (size_t)n : sizeof row - 1;
    size_t room = cols > 4 ? cols - 4 : 0;
    if (len > room) len = room;
    // the label bright, the rest dim; a dimmed row all dim
    size_t cut = len < lw ? len : lw;
    if (it->off)
        printf("\r\x1b[2K    \x1b[2m%.*s\x1b[0m\n", (int)len, row);
    else if (sel)
        printf("\r\x1b[2K  \x1b[1;36m> %.*s\x1b[0m\x1b[2m%.*s\x1b[0m\n", (int)cut, row, (int)(len - cut), row + cut);
    else
        printf("\r\x1b[2K    %.*s\x1b[2m%.*s\x1b[0m\n", (int)cut, row, (int)(len - cut), row + cut);
}

static size_t term_rows(void) {
    struct winsize w;
    if (ioctl(1, TIOCGWINSZ, &w) == 0 && w.ws_row >= 3) return w.ws_row;
    return 24;
}

// Items top .. top + shown - 1, then the footer line (with "more" marks when the list scrolls).
static void draw_window(const Item *it, size_t n, size_t top, size_t shown, size_t cur) {
    for (size_t i = top; i < top + shown; i++) draw_item(&it[i], i == cur);
    const bool up = top > 0, down = top + shown < n;
    printf("\r\x1b[2K\x1b[2m  up/down to move, enter to choose, q to quit%s\x1b[0m\n",
           up && down ? "  (more above and below)" : up ? "  (more above)" : down ? "  (more below)" : "");
}

// The index chosen, -1 when the user quit. The list scrolls inside the rows the terminal has, so
// every redraw moves the cursor up exactly the rows it drew.
static int menu(const char *title, const Item *it, size_t n, size_t def) {
    size_t cur = def < n ? def : 0;
    while (cur < n && it[cur].off) cur++;
    if (cur == n) return -1;
    printf("\n\x1b[1m%s\x1b[0m\n", title);
    // rows for the items: the terminal less the title and the footer, at least one
    const size_t rows = term_rows(), shown = n < (rows > 3 ? rows - 3 : 1) ? n : (rows > 3 ? rows - 3 : 1);
    size_t top = cur >= shown ? cur - shown + 1 : 0;
    draw_window(it, n, top, shown, cur);
    fflush(stdout);
    raw_enter();
    int ret = -1;
    for (;;) {
        int k = read_key();
        if (k == K_ENTER) {
            ret = (int)cur;
            break;
        }
        if (k == K_QUIT) break;
        size_t nc = cur;
        if (k == K_UP)
            do nc = nc == 0 ? cur : nc - 1;
            while (nc != cur && it[nc].off);
        else if (k == K_DOWN)
            do nc = nc + 1 >= n ? cur : nc + 1;
            while (nc != cur && it[nc].off);
        if (nc == cur) continue;
        cur = nc;
        if (cur < top) top = cur;
        if (cur >= top + shown) top = cur - shown + 1;
        printf("\x1b[%zuA", shown + 1); // back to the first row drawn
        draw_window(it, n, top, shown, cur);
        fflush(stdout);
    }
    raw_off();
    return ret;
}

// y / n, Enter takes the default.
static bool ask(const char *q, bool def) {
    printf("\n%s %s ", q, def ? "[Y/n]" : "[y/N]");
    fflush(stdout);
    raw_enter();
    bool r = def;
    for (;;) {
        int k = read_key();
        if (k == K_YES) r = true;
        else if (k == K_NO || k == K_QUIT) r = false;
        else if (k != K_ENTER) continue;
        break;
    }
    raw_off();
    printf("%s\n", r ? "yes" : "no");
    return r;
}

// ---- this Mac --------------------------------------------------------------------------------

static void chip_name(char *b, size_t cap) {
    size_t len = cap;
    if (sysctlbyname("machdep.cpu.brand_string", b, &len, NULL, 0) != 0) snprintf(b, cap, "Apple Silicon");
}

// Free bytes on the volume holding `dir` (or its nearest existing parent).
static uint64_t free_disk(const char *dir) {
    char p[1024];
    snprintf(p, sizeof p, "%s", dir);
    struct statfs s;
    while (statfs(p, &s) != 0) {
        char *sl = strrchr(p, '/');
        if (!sl || sl == p) return 0;
        *sl = 0;
    }
    return (uint64_t)s.f_bavail * (uint64_t)s.f_bsize;
}

static bool exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

// A pack folder that looks complete: its config and its weights (one file or an index).
static bool downloaded(Arena *a, const DownloadPack *p) {
    char *dir;
    if (download_pack_dir(a, p, &dir)) return false;
    return exists(path_join(a, dir, "config.json")) &&
           (exists(path_join(a, dir, "model.safetensors")) || exists(path_join(a, dir, "model.safetensors.index.json")));
}

// Whether `name` runs from PATH.
static bool on_path(Arena *a, const char *name) {
    const char *path = getenv("PATH");
    if (!path) return false;
    const char *s = path;
    while (*s) {
        const char *e = strchr(s, ':');
        size_t n = e ? (size_t)(e - s) : strlen(s);
        if (n) {
            char *dir = arena_printf(a, "%.*s", (int)n, s);
            if (access(path_join(a, dir, name), X_OK) == 0) return true;
        }
        if (!e) break;
        s = e + 1;
    }
    return false;
}

// ---- the steps -------------------------------------------------------------------------------

typedef struct {
    const char *name, *what;
} Choice;

// Best first: Flash-Next where it fits, then the 27B packs.
static const Choice models[] = {
    {"coder", "Qwen3.8-Flash-Next Coder, fastest for coding agents"},
    {"full", "Qwen3.8-Flash-Next, all 512 experts"},
    {"27b-turbo", "Qwen3.8-27B TURBO fine-tune, shorter thinking, MTP drafts"},
    {"27b-turbo-8bit", "Qwen3.8-27B TURBO fine-tune, 8-bit"},
    {"27b-coder390", "Qwen3.8-27B Coder390 fine-tune, SFT + RL on code, MTP drafts"},
    {"27b-coder390-8bit", "Qwen3.8-27B Coder390 fine-tune, 8-bit"},
    {"27b-fable5", "Qwen3.8-27B Fable-5 coding distill, MTP drafts"},
    {"27b-fable5-8bit", "Qwen3.8-27B Fable-5 coding distill, 8-bit"},
    {"27b-8bit", "Qwen3.8-27B, 8-bit, takes LoRA adapters"},
    {"27b-4bit", "Qwen3.8-27B, 4-bit, the smallest"},
};

static const struct {
    const char *cmd, *label;
} agents[] = {{"claude", "Claude Code"}, {"pi", "pi"}, {"opencode", "opencode"}};

Err wizard_run(int *code) {
    *code = 1;
    if (!isatty(0) || !isatty(1)) return "NotATerminal";
    if (tcgetattr(0, &saved) != 0) return "NotATerminal";
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    Arena a = {0};

    // this Mac
    char chip[128];
    chip_name(chip, sizeof chip);
    const uint64_t gib = 1ull << 30, ram = fit_ram_bytes();
    char *mdir = NULL;
    if (download_models_dir(&a, &mdir)) mdir = arena_printf(&a, "%s", getenv("HOME") ? getenv("HOME") : "/");
    const uint64_t disk = free_disk(mdir);
    printf("\x1b[1msaetta\x1b[0m: local coding models for this Mac\n\n");
    printf("  %s, %llu GB memory, %llu GB free disk\n", chip, (unsigned long long)(ram / gib),
           (unsigned long long)(disk / 1000000000ull));

    // the model
    const char *rec = fit_default_pack();
    Item mi[COUNT(models)];
    size_t def = 0;
    bool any = false;
    for (size_t i = 0; i < COUNT(models); i++) {
        const DownloadPack *p = download_find(models[i].name);
        const bool have = p && downloaded(&a, p);
        const bool fits = p && ((uint64_t)p->min_ram_gb << 30) <= ram;
        const bool room = have || (p && (uint64_t)p->disk_gb * 1000000000ull <= disk);
        const char *size = p ? arena_printf(&a, "%u GB", p->min_ram_gb ? p->disk_gb : 0) : "";
        const char *status;
        if (!fits)
            status = arena_printf(&a, "needs %u GB memory", p ? p->min_ram_gb : 0);
        else if (!room)
            status = arena_printf(&a, "needs %u GB free disk", p->disk_gb);
        else
            status = arena_printf(&a, "%s%s", have ? "downloaded" : size,
                                  rec && strcmp(rec, models[i].name) == 0 ? ", recommended" : "");
        mi[i] = (Item){models[i].name, status, models[i].what, !fits || !room};
        if (rec && strcmp(rec, models[i].name) == 0 && !mi[i].off) def = i;
        if (!mi[i].off) any = true;
    }
    if (!any) {
        printf("\n");
        fit_suggest(ram);
        printf("Nothing to run here: no saetta model fits this Mac's memory and free disk.\n");
        arena_reset(&a);
        return OK;
    }
    int m = menu("Choose a model", mi, COUNT(models), def);
    if (m < 0) {
        *code = 0;
        arena_reset(&a);
        return OK;
    }
    const DownloadPack *pk = download_find(models[m].name);
    if (!downloaded(&a, pk)) {
        if (!ask(arena_printf(&a, "Download %s now (about %u GB, resumable)?", pk->name, pk->disk_gb), true)) {
            printf("Later: saetta download %s\n", pk->name);
            *code = 0;
            arena_reset(&a);
            return OK;
        }
    }
    // also completes a partial download; a finished one is left as it is
    const char *dargs[] = {pk->name};
    Err de = download_run(dargs, 1);
    if (de) {
        printf("saetta: download failed (%s); run saetta download %s to resume\n", de, pk->name);
        arena_reset(&a);
        return OK;
    }

    // the agent
    for (;;) {
        Item ai[COUNT(agents)];
        size_t adef = COUNT(agents);
        for (size_t i = 0; i < COUNT(agents); i++) {
            const bool inst = on_path(&a, agents[i].cmd);
            ai[i] = (Item){agents[i].label, inst ? "installed" : "not installed", inst ? "" : "choose it to install", false};
            if (inst && adef == COUNT(agents)) adef = i;
        }
        int g = menu("Choose a coding agent", ai, COUNT(agents), adef == COUNT(agents) ? 0 : adef);
        if (g < 0) {
            printf("Later: saetta start <claude|pi|opencode> --pack %s\n", pk->name);
            *code = 0;
            arena_reset(&a);
            return OK;
        }
        if (!on_path(&a, agents[g].cmd)) {
            const char *cmd = launch_install_command(agents[g].cmd);
            printf("\n%s is not installed. Its official installer:\n\n    %s\n", agents[g].label, cmd);
            if (!ask("Run it now?", false)) continue;
            int rc = system(cmd);
            if (rc != 0 || !on_path(&a, agents[g].cmd)) {
                printf("\n%s is not on PATH yet%s. Open a new terminal (or add its folder to PATH) and run saetta again.\n",
                       agents[g].label, rc != 0 ? " (the installer failed)" : "");
                arena_reset(&a);
                return OK;
            }
        }
        printf("\nStarting %s on %s (next time: saetta start %s --pack %s)\n\n", agents[g].label, pk->name, agents[g].cmd,
               pk->name);
        fflush(stdout);
        const char *largs[] = {agents[g].cmd, "--pack", pk->name};
        Err le = launch_run(largs, COUNT(largs), code);
        arena_reset(&a);
        return le;
    }
}
