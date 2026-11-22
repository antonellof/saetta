// Saetta Code's update check: at the start of the interactive mode, the latest release (the one
// `saetta update` installs, src/control.c control_latest_release) is read on a thread of its own;
// a newer version than this binary is announced once in the chat, with `saetta update` to run.
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at v1.5.2,
// b4afddd): packages/coding-agent/src/utils/version-check.ts (getLatestPiRelease,
// checkForNewPiVersion, comparePackageVersions, isNewerPackageVersion),
// modes/interactive/interactive-mode.ts (run: the asynchronous start; showNewVersionNotification),
// modes/interactive/components/dynamic-border.ts (DynamicBorder with a color).
//
// The source is the GitHub release of antonellof/saetta (pi asks pi.dev/api/latest-version): its
// tag is the version and the first paragraph of its notes is the note. A failure of any kind (no
// network, a timeout, a private repository's 404, a reply that does not parse) shows nothing, as
// in pi. SAETTA_CODE_LATEST_VERSION (and SAETTA_CODE_LATEST_NOTES, the release notes) replace the
// network's answer, for the scenarios.
#include "code_update.h"

#include "code_res_ui.h"
#include "control.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHANGELOG_URL "https://github.com/antonellof/saetta/releases"
#define VERSION_CHECK_TIMEOUT_S 10.0 // DEFAULT_VERSION_CHECK_TIMEOUT_MS

typedef struct {
    char *version;
    char *note; // NULL when none
} Release;

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_wake = -1;           // under g_lock
static Release g_rel;             // written by the thread before g_ready
static _Atomic bool g_ready;      // the thread found a newer release
static bool g_started, g_shown;   // UI thread only
static char g_current[64];

// process.env.X: set and not empty (JavaScript's truthiness of a string).
static bool env_on(const char *name) {
    const char *v = getenv(name);
    return v && v[0];
}

// comparePackageVersions: semver.compare of two valid versions (semver.valid takes a leading v).
static bool compare_versions(Str l, Str r, int *out) {
    l = str_trim(l);
    r = str_trim(r);
    if (l.len && l.ptr[0] == 'v') l = str_sub(l, 1, l.len);
    if (r.len && r.ptr[0] == 'v') r = str_sub(r, 1, r.len);
    SemanticVersion a, b;
    if (semver_parse(l, &a) || semver_parse(r, &b)) return false;
    *out = semver_order(&a, &b);
    return true;
}

// isNewerPackageVersion: newer by semver, or just different when either is not a version.
static bool is_newer(Str candidate, Str current) {
    int c;
    if (compare_versions(candidate, current, &c)) return c > 0;
    return !str_eq(str_trim(candidate), str_trim(current));
}

// The note: the first paragraph of the release notes (pi's release.note is a line of its own).
static char *first_paragraph(const char *notes) {
    Buf b = {0};
    for (const char *p = notes; *p; p++)
        if (*p != '\r') buf_addc(&b, *p);
    Str t = code_js_trim(buf_str(&b));
    size_t end = t.len;
    for (size_t i = 0; i < t.len; i++) {
        if (t.ptr[i] != '\n') continue;
        size_t j = i + 1;
        while (j < t.len && (t.ptr[j] == ' ' || t.ptr[j] == '\t')) j++;
        if (j >= t.len || t.ptr[j] == '\n') {
            end = i;
            break;
        }
    }
    Str para = code_js_trim(str_sub(t, 0, end));
    char *r = para.len ? strndup(para.ptr, para.len) : NULL;
    buf_free(&b);
    return r;
}

// getLatestPiRelease: the version (trimmed, not empty) and the note, or false. No retry.
static bool latest_release(Release *out) {
    if (env_on("PI_OFFLINE")) return false;
    const char *tv = getenv("SAETTA_CODE_LATEST_VERSION");
    Arena a = {0};
    ControlRelease r = {.status = 200, .version = tv, .notes = getenv("SAETTA_CODE_LATEST_NOTES")};
    if (!tv && control_latest_release(&a, VERSION_CHECK_TIMEOUT_S, &r)) r.status = 0;
    bool ok = false;
    if (r.status == 200) { // !response.ok: nothing
        Str v = str_trim(str(r.version ? r.version : ""));
        if (v.len) {
            out->version = strndup(v.ptr, v.len);
            // no note: saetta's release notes open with install steps, not a one-line summary
            // (pi shows release.note only when its API sends one)
            out->note = NULL;
            (void)first_paragraph;
            ok = true;
        }
    }
    arena_reset(&a);
    return ok;
}

static void *check_main(void *arg) {
    (void)arg;
    Release rel = {0};
    // checkForNewPiVersion
    if (latest_release(&rel) && is_newer(str(rel.version), str(g_current))) {
        g_rel = rel;
        atomic_store(&g_ready, true);
        pthread_mutex_lock(&g_lock);
        if (g_wake >= 0) (void)!write(g_wake, "u", 1);
        pthread_mutex_unlock(&g_lock);
    } else {
        free(rel.version);
        free(rel.note);
    }
    return NULL;
}

void code_update_check_start(int wake_fd) {
    if (g_started) return;
    g_started = true;
    if (env_on("PI_SKIP_VERSION_CHECK") || env_on("SAETTA_SKIP_VERSION_CHECK")) return;
    const char *v = code_ui_config()->version;
    snprintf(g_current, sizeof g_current, "%s", v ? v : "0.0.0");
    pthread_mutex_lock(&g_lock);
    g_wake = wake_fd;
    pthread_mutex_unlock(&g_lock);
    pthread_t th;
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    if (pthread_create(&th, &at, check_main, NULL)) {
        pthread_mutex_lock(&g_lock);
        g_wake = -1;
        pthread_mutex_unlock(&g_lock);
    }
    pthread_attr_destroy(&at);
}

void code_update_check_stop(void) {
    pthread_mutex_lock(&g_lock);
    g_wake = -1;
    pthread_mutex_unlock(&g_lock);
}

// ---- showNewVersionNotification -------------------------------------------------------------------

// new DynamicBorder((text) => theme.fg("warning", text))
static void border_render(void *ctx, int width, Lines *out) {
    (void)ctx;
    Buf b = {0}, r = {0};
    tui_repeat(&r, "\xe2\x94\x80", MAX(1, width));
    th_fg(&b, T_WARNING, buf_str(&r));
    lines_addb(out, &b);
    buf_free(&r);
}

static void text_render(Buf *t, int width, Lines *out) {
    Comp *c = comp_text(buf_str(t), 1, 0);
    comp_render(c, width, out);
    comp_free(c);
}

// ThemedText: bold warning "Update Available", then updateInstruction()
static void title_render(void *ctx, int width, Lines *out) {
    const Release *rel = ctx;
    Buf w = {0}, t = {0}, m = {0};
    th_fgz(&w, T_WARNING, "Update Available");
    th_bold(&t, buf_str(&w));
    buf_addc(&t, '\n');
    buf_printf(&m, "New version %s is available. Run ", rel->version);
    th_fg(&t, T_MUTED, buf_str(&m));
    th_fgz(&t, T_ACCENT, "saetta update");
    text_render(&t, width, out);
    buf_free(&w);
    buf_free(&t);
    buf_free(&m);
}

// changelogLine: muted "Changelog: ", the address in accent, an OSC 8 link when the terminal shows them
static void changelog_render(void *ctx, int width, Lines *out) {
    (void)ctx;
    Buf t = {0}, u = {0};
    th_fgz(&t, T_MUTED, "Changelog: ");
    th_fgz(&u, T_ACCENT, CHANGELOG_URL);
    if (tui_hyperlinks()) {
        // hyperlink(text, url): ESC ] 8 ; ; url ESC \ text ESC ] 8 ; ; ESC \ (terminal-image.ts)
        buf_adds(&t, "\x1b]8;;" CHANGELOG_URL "\x1b\\");
        buf_addstr(&t, buf_str(&u));
        buf_adds(&t, "\x1b]8;;\x1b\\");
    } else
        buf_addstr(&t, buf_str(&u));
    text_render(&t, width, out);
    buf_free(&t);
    buf_free(&u);
}

bool code_update_check_poll(void) {
    if (g_shown || !atomic_load(&g_ready)) return false;
    g_shown = true;
    static const MdStyle muted = {.has_color = true, .color = T_MUTED};
    code_ui_chat_add(comp_spacer(1));
    code_ui_chat_add(comp_fn(border_render, NULL));
    code_ui_chat_add(comp_fn(title_render, &g_rel));
    if (g_rel.note) {
        code_ui_chat_add(comp_spacer(1));
        code_ui_chat_add(comp_markdown(str(g_rel.note), 1, 0, &muted));
        code_ui_chat_add(comp_spacer(1));
    }
    code_ui_chat_add(comp_fn(changelog_render, NULL));
    code_ui_chat_add(comp_fn(border_render, NULL));
    return true;
}
