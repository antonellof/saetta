// Saetta Code's changelog: CHANGELOG.md (the repository's root, embedded at build time), the
// startup notice of the versions since the last one seen (settings lastChangelogVersion), and
// /changelog.
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at v1.5.2,
// b4afddd): packages/coding-agent/src/utils/changelog.ts (parseChangelog, compareVersions,
// getNewEntries), modes/interactive/interactive-mode.ts (getChangelogForDisplay,
// showStartupNoticesIfNeeded, handleChangelogCommand), core/settings-manager.ts
// (getLastChangelogVersion, setLastChangelogVersion, getCollapseChangelog, the locked write).
//
// The content is Saetta Code's own (an accepted difference). normalizeChangelogLinks is not
// ported: it points relative links at pi's repository, and this changelog has none.
#include "code_res.h"
#include "code_res_ui.h"
#include "code_cfg.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static const char CHANGELOG[] = {
#embed "../CHANGELOG.md"
    , 0};

typedef struct {
    int major, minor, patch;
    Str content;
} Entry;
typedef VEC(Entry) VecEntry;

// "##\s+\[?(\d+)\.(\d+)\.(\d+)\]?" at the start of a line.
static bool heading_version(Str line, int v[3]) {
    size_t i = 2;
    if (!str_starts(line, S("##"))) return false;
    size_t ws = i;
    while (i < line.len && (line.ptr[i] == ' ' || line.ptr[i] == '\t')) i++;
    if (i == ws) return false;
    if (i < line.len && line.ptr[i] == '[') i++;
    for (int k = 0; k < 3; k++) {
        size_t d0 = i;
        long n = 0;
        while (i < line.len && line.ptr[i] >= '0' && line.ptr[i] <= '9') n = n * 10 + (line.ptr[i++] - '0');
        if (i == d0) return false;
        v[k] = (int)n;
        if (k < 2) {
            if (i >= line.len || line.ptr[i] != '.') return false;
            i++;
        }
    }
    return true;
}

// parseChangelog: an entry per "## x.y.z" heading, its lines up to the next "## " heading, trimmed.
static void parse_changelog(Arena *a, Str text, VecEntry *out) {
    Buf cur = {0};
    bool have = false;
    int v[3] = {0};
    for (size_t i = 0; i <= text.len;) {
        ptrdiff_t e = str_find_char(text, '\n', i);
        size_t end = e < 0 ? text.len : (size_t)e;
        Str line = str_sub(text, i, end);
        i = end + 1;
        if (str_starts(line, S("## "))) {
            if (have) {
                Str c = code_js_trim(buf_str(&cur));
                vec_push(out, ((Entry){v[0], v[1], v[2], {arena_strndup(a, c.ptr, c.len), c.len}}));
            }
            cur.len = 0;
            have = heading_version(line, v);
            if (have) buf_addstr(&cur, line);
        } else if (have) {
            buf_addc(&cur, '\n');
            buf_addstr(&cur, line);
        }
        if (e < 0) break;
    }
    if (have) {
        Str c = code_js_trim(buf_str(&cur));
        vec_push(out, ((Entry){v[0], v[1], v[2], {arena_strndup(a, c.ptr, c.len), c.len}}));
    }
    buf_free(&cur);
}

static int compare_versions(const Entry *x, const int y[3]) {
    if (x->major != y[0]) return x->major - y[0];
    if (x->minor != y[1]) return x->minor - y[1];
    return x->patch - y[2];
}

// getNewEntries: false when last has no leading major.minor.patch.
static bool new_entries(const VecEntry *all, const char *last, VecEntry *out) {
    Str t = code_js_trim(str(last));
    size_t i = 0;
    if (i < t.len && t.ptr[i] == 'v') i++;
    int v[3];
    for (int k = 0; k < 3; k++) {
        size_t d0 = i;
        long n = 0;
        while (i < t.len && t.ptr[i] >= '0' && t.ptr[i] <= '9') n = n * 10 + (t.ptr[i++] - '0');
        if (i == d0) return false;
        v[k] = (int)n;
        if (k < 2) {
            if (i >= t.len || t.ptr[i] != '.') return false;
            i++;
        }
    }
    for (size_t k = 0; k < all->len; k++)
        if (compare_versions(&all->ptr[k], v) > 0) vec_push(out, all->ptr[k]);
    return true;
}

// ---- settings.json (settings-manager.ts: the global file, written whole under its lock) ------------

static bool put_last_version(Arena *a, JValue *root, void *ctx) {
    json_put(a, root, S("saettaLastChangelogVersion"), json_string(a, str((const char *)ctx)));
    return true;
}

// setLastChangelogVersion and the flush: under proper-lockfile's lock, the file read again, the
// field set, written whole (code_cfg.c cfg_json_update_ex). A file that does not parse is left
// alone, silently.
static void save_last_version(const char *agent_dir, const char *version) {
    Buf path = {0};
    buf_printf(&path, "%s/settings.json", agent_dir);
    CfgUpdateOpts o = {.parse_numbers = true, .lock_tries = 20};
    (void)cfg_json_update_ex(path.ptr, &o, put_last_version, (void *)version, NULL);
    buf_free(&path);
}

// ---- the views --------------------------------------------------------------------------------------

static void border_render(void *ctx, int width, Lines *out) {
    (void)ctx;
    Buf b = {0}, r = {0};
    tui_repeat(&r, "\xe2\x94\x80", MAX(1, width));
    th_fg(&b, T_BORDER, buf_str(&r));
    lines_addb(out, &b);
    buf_free(&r);
}

static Comp *dynamic_border(void) { return comp_fn(border_render, NULL); }

static void whats_new_render(void *ctx, int width, Lines *out) {
    (void)ctx;
    Buf b = {0}, bold = {0};
    th_fgz(&b, T_ACCENT, "What's New");
    th_bold(&bold, buf_str(&b));
    Comp *t = comp_text(buf_str(&bold), 1, 0);
    comp_render(t, width, out);
    comp_free(t);
    buf_free(&b);
    buf_free(&bold);
}

// ThemedText(() => theme.bold(theme.fg("accent", "What's New")), 1, 0): drawn with the theme of the moment.
static Comp *whats_new(void) { return comp_fn(whats_new_render, NULL); }

// getChangelogForDisplay and showStartupNoticesIfNeeded
static void changelog_notice(void) {
    const CodeUiConfig *cfg = code_ui_config();
    Arena a = {0};
    const char *version = cfg->version ? cfg->version : "0.0.0";
    JValue *settings = code_read_json_object(&a, code_path_join(&a, cfg->agent_dir, "settings.json"));
    const char *last = json_str(json_get(settings, "saettaLastChangelogVersion"));
    VecEntry all = {0}, fresh = {0};
    parse_changelog(&a, str(CHANGELOG), &all);
    Buf md = {0};
    if (!last) save_last_version(cfg->agent_dir, version); // a fresh install: recorded, nothing shown
    else if (!new_entries(&all, last, &fresh)) save_last_version(cfg->agent_dir, version);
    else if (fresh.len) {
        save_last_version(cfg->agent_dir, version);
        for (size_t i = 0; i < fresh.len; i++) {
            if (i) buf_adds(&md, "\n\n");
            buf_addstr(&md, fresh.ptr[i].content);
        }
    }
    if (md.len) {
        if (code_ui_chat_len()) code_ui_chat_add(comp_spacer(1));
        code_ui_chat_add(dynamic_border());
        const JValue *collapse = json_get(settings, "collapseChangelog");
        if (json_is(collapse, J_BOOL) && collapse->b) {
            int v[3];
            const char *latest = version;
            // the first "## [x.y.z]" of the notice
            for (size_t i = 0; i + 1 < md.len; i++)
                if (md.ptr[i] == '#' && md.ptr[i + 1] == '#' && heading_version(str_sub(buf_str(&md), i, md.len), v)) {
                    latest = arena_printf(&a, "%d.%d.%d", v[0], v[1], v[2]);
                    break;
                }
            Buf t = {0}, bold = {0};
            th_bold(&bold, S("/changelog"));
            buf_printf(&t, "Updated to v%s. Use %.*s to view full changelog.", latest, (int)bold.len, bold.ptr);
            code_ui_chat_add(comp_text(buf_str(&t), 1, 0));
            buf_free(&t);
            buf_free(&bold);
        } else {
            code_ui_chat_add(whats_new());
            code_ui_chat_add(comp_spacer(1));
            code_ui_chat_add(comp_markdown(code_js_trim(buf_str(&md)), 1, 0, NULL));
            code_ui_chat_add(comp_spacer(1));
        }
        code_ui_chat_add(dynamic_border());
    }
    buf_free(&md);
    vec_free(&all);
    vec_free(&fresh);
    arena_reset(&a);
}

void code_startup_notices(bool resumed) {
    if (!resumed) changelog_notice(); // resumed sessions already have messages
    code_trust_warning();
}

// handleChangelogCommand: every entry, oldest first.
void code_changelog_command(void) {
    Arena a = {0};
    VecEntry all = {0};
    parse_changelog(&a, str(CHANGELOG), &all);
    Buf md = {0};
    for (size_t i = all.len; i-- > 0;) {
        if (md.len) buf_adds(&md, "\n\n");
        buf_addstr(&md, all.ptr[i].content);
    }
    if (!all.len) buf_adds(&md, "No changelog entries found.");
    code_ui_chat_add(comp_spacer(1));
    code_ui_chat_add(dynamic_border());
    code_ui_chat_add(whats_new());
    code_ui_chat_add(comp_spacer(1));
    code_ui_chat_add(comp_markdown(buf_str(&md), 1, 1, NULL));
    code_ui_chat_add(dynamic_border());
    buf_free(&md);
    vec_free(&all);
    arena_reset(&a);
}
