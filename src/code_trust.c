// Saetta Code's project trust (see code_res.h): the trust store shared with pi (agent_dir/
// trust.json), the session's trust decision, the startup prompt for a project with project
// resources, /trust, and the warning for an untrusted project.
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at v1.5.2,
// b4afddd): packages/coding-agent/src/core/trust-manager.ts (ProjectTrustStore, readTrustFile,
// writeTrustFile, withTrustFileLock, findNearestTrustEntry, getProjectTrustOptions,
// getProjectTrustParentPath, hasTrustRequiringProjectResources), core/project-trust.ts
// (resolveProjectTrusted, formatProjectTrustPrompt, selectProjectTrustOption), main.ts
// (createRuntime's trust, autoTrustOnReloadCwd), cli/args.ts (--approve, --no-approve),
// cli/startup-ui.ts (showStartupSelector), cli/project-trust.ts, components/
// extension-selector.ts, components/trust-selector.ts, modes/interactive/interactive-mode.ts
// (showTrustSelector, renderProjectTrustWarningIfNeeded,
// maybeSaveImplicitProjectTrustAfterReload).
//
// The trust.json lock is proper-lockfile's: a "trust.json.lock" folder, stale after 10 s.
// Not ported: extensions' project_trust event (Saetta Code has no extensions).
#include "code_res.h"
#include "code_res_ui.h"
#include "code_cfg.h"

#include <errno.h>
#include <poll.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static int g_override = -1; // --approve / --no-approve
static char *g_auto_cwd;    // autoTrustOnReloadCwd

// main.ts projectTrustByCwd: the decisions resolved (asked, or read from the store) for a folder
// with project resources, one per cwd, for this process. The current runtime's trust
// (settingsManager.isProjectTrusted) is g_trusted for g_cur_cwd; a new cwd (a resumed or imported
// session's folder) resolves its own, never inherits it.
typedef struct {
    char *cwd;
    bool trusted;
} CwdTrust;
static pthread_mutex_t g_tmu = PTHREAD_MUTEX_INITIALIZER;
static VEC(CwdTrust) g_by_cwd;
static char *g_cur_cwd;
static int g_trusted = -1;
// The folders (real paths) the user trusted in this session through a prompt (or the restart's
// hand-off of one): with the trust store's own "trusted" entries, the folders whose repository the
// footer may look into (code_trust_explicit).
static VEC(char *) g_approved;

// g_tmu held
static void approve_l(const char *real) {
    for (size_t i = 0; i < g_approved.len; i++)
        if (!strcmp(g_approved.ptr[i], real)) return;
    vec_push(&g_approved, strdup(real));
}

static bool approved_l(const char *real) {
    for (size_t i = 0; i < g_approved.len; i++)
        if (!strcmp(g_approved.ptr[i], real)) return true;
    return false;
}

// projectTrustByCwd.get: -1 for none (g_tmu held)
static int cached_l(const char *cwd) {
    for (size_t i = 0; i < g_by_cwd.len; i++)
        if (!strcmp(g_by_cwd.ptr[i].cwd, cwd)) return g_by_cwd.ptr[i].trusted;
    return -1;
}

// projectTrustByCwd.set (g_tmu held)
static void cache_l(const char *cwd, bool trusted) {
    for (size_t i = 0; i < g_by_cwd.len; i++)
        if (!strcmp(g_by_cwd.ptr[i].cwd, cwd)) {
            g_by_cwd.ptr[i].trusted = trusted;
            return;
        }
    vec_push(&g_by_cwd, ((CwdTrust){strdup(cwd), trusted}));
}

void code_trust_override(int trusted) { g_override = trusted; }

// ---- trust-manager.ts ---------------------------------------------------------------------------

bool code_trust_has_project_resources(Arena *a, const char *cwd, const char *home) {
    static const char *const entries[] = {"settings.json", "mcp.json", "extensions", "skills",
                                          "prompts", "themes", "SYSTEM.md", "APPEND_SYSTEM.md"};
    char *user_agents = code_path_join(a, code_path_real(a, code_path_resolve(a, cwd, home)), ".agents/skills");
    char *dir = code_path_real(a, code_path_resolve(a, "/", cwd));
    char *cfg = code_path_join(a, dir, ".pi");
    for (size_t i = 0; i < COUNT(entries); i++)
        if (code_path_exists(code_path_join(a, cfg, entries[i]))) return true;
    for (;;) {
        char *sd = code_path_join(a, dir, ".agents/skills");
        if (strcmp(sd, user_agents) && code_path_exists(sd)) return true;
        char *p = code_path_dirname(a, dir);
        if (!strcmp(p, dir)) return false;
        dir = p;
    }
}

// normalizeCwd
static char *norm_cwd(Arena *a, const char *cwd) { return code_path_real(a, code_path_resolve(a, "/", cwd)); }

static char *trust_path(Arena *a, const char *agent_dir) { return code_path_join(a, code_path_resolve(a, "/", agent_dir), "trust.json"); }

// readTrustFile: the entries (values true, false or null), or an error.
static const char *read_trust(Arena *a, const char *path, JValue **out) {
    *out = json_new(a, J_OBJECT);
    if (!code_path_exists(path)) return NULL;
    Str t;
    JValue *v;
    if (!code_read_text(a, path, &t) || json_parse(a, t, false, &v))
        return arena_printf(a, "Failed to read trust store %s", path);
    if (!json_is(v, J_OBJECT)) return arena_printf(a, "Invalid trust store %s: expected an object", path);
    for (size_t i = 0; i < v->obj.len; i++) {
        JValue *x = v->obj.items[i].val;
        if (!json_is(x, J_BOOL) && !json_is(x, J_NULL)) {
            Buf k = {0};
            json_write_string(&k, v->obj.items[i].key);
            const char *m = arena_printf(a, "Invalid trust store %s: value for %.*s must be true, false, or null", path, (int)k.len, k.ptr);
            buf_free(&k);
            return m;
        }
    }
    *out = v;
    return NULL;
}

// proper-lockfile's lockSync on the folder, lockfilePath "<path>.lock": 10 tries, 20 ms apart;
// a lock older than 10 s is stale.
static bool trust_lock(Arena *a, const char *path) {
    mkdir(code_path_dirname(a, path), 0755);
    return cfg_lock(path, 10, 20); // code_cfg.c
}

static void trust_unlock(Arena *a, const char *path) {
    (void)a;
    cfg_unlock(path);
}

// findNearestTrustEntry: the nearest entry with a decision, at or above the real cwd.
static bool nearest_entry(Arena *a, const JValue *data, const char *cwd, const char **path, bool *decision) {
    for (char *d = norm_cwd(a, cwd);;) {
        for (size_t i = 0; i < data->obj.len; i++) {
            JValue *v = data->obj.items[i].val;
            if (json_is(v, J_BOOL) && !strcmp(code_path_resolve(a, "/", data->obj.items[i].key.ptr), d)) {
                *path = data->obj.items[i].key.ptr;
                *decision = v->b;
                return true;
            }
        }
        char *p = code_path_dirname(a, d);
        if (!strcmp(p, d)) return false;
        d = p;
    }
}

// ProjectTrustStore.getEntry: false when none (or the store cannot be read: *error).
static bool store_entry(Arena *a, const char *agent_dir, const char *cwd, const char **path, bool *decision, const char **error) {
    const char *tp = trust_path(a, agent_dir);
    *error = NULL;
    if (!trust_lock(a, tp)) {
        *error = "Failed to acquire trust store lock";
        return false;
    }
    JValue *data;
    *error = read_trust(a, tp, &data);
    bool found = !*error && nearest_entry(a, data, cwd, path, decision);
    trust_unlock(a, tp);
    return found;
}

// ProjectTrustStore.get: -1 for none.
static int store_get(Arena *a, const char *agent_dir, const char *cwd) {
    const char *p, *err;
    bool d;
    return store_entry(a, agent_dir, cwd, &p, &d, &err) ? d : -1;
}

typedef struct {
    const char *path;
    int decision; // 1, 0, or -1 for null (the entry removed)
} TrustUpdate;

// ProjectTrustStore.setMany, and writeTrustFile: keys sorted, JSON.stringify(data, null, 2) + "\n".
static const char *store_set_many(Arena *a, const char *agent_dir, const TrustUpdate *u, size_t n) {
    const char *tp = trust_path(a, agent_dir);
    if (!trust_lock(a, tp)) return "Failed to acquire trust store lock";
    JValue *data;
    const char *err = read_trust(a, tp, &data);
    if (!err) {
        for (size_t i = 0; i < n; i++) {
            const char *key = norm_cwd(a, u[i].path);
            JValue *next = json_new(a, J_OBJECT);
            for (size_t k = 0; k < data->obj.len; k++)
                if (strcmp(code_path_resolve(a, "/", data->obj.items[k].key.ptr), code_path_resolve(a, "/", key)))
                    json_put(a, next, data->obj.items[k].key, data->obj.items[k].val);
            if (u[i].decision >= 0) json_put(a, next, str(key), json_bool(a, u[i].decision));
            data = next;
        }
        // sorted keys (Array.prototype.sort: code unit order), booleans and nulls only
        size_t m = data->obj.len;
        for (size_t i = 1; i < m; i++)
            for (size_t k = i; k > 0 && strcmp(data->obj.items[k - 1].key.ptr, data->obj.items[k].key.ptr) > 0; k--) {
                typeof(data->obj.items[0]) t = data->obj.items[k];
                data->obj.items[k] = data->obj.items[k - 1];
                data->obj.items[k - 1] = t;
            }
        Buf b = {0};
        size_t shown = 0;
        buf_adds(&b, "{");
        for (size_t i = 0; i < m; i++) {
            JValue *v = data->obj.items[i].val;
            if (!json_is(v, J_BOOL) && !json_is(v, J_NULL)) continue;
            buf_adds(&b, shown++ ? ",\n  " : "\n  ");
            json_write_string(&b, data->obj.items[i].key);
            buf_adds(&b, ": ");
            json_write(&b, v);
        }
        buf_adds(&b, shown ? "\n}\n" : "}\n");
        // the shared writer (code_cfg.c): fsync'd, through a symlink, the file's mode kept
        if (!cfg_write_file(tp, b.ptr, b.len, NULL)) err = arena_printf(a, "Could not write %s", tp);
        buf_free(&b);
    }
    trust_unlock(a, tp);
    return err;
}

typedef struct {
    const char *label;
    bool trusted;
    TrustUpdate updates[2];
    size_t nupdates;
    const char *saved_path;
} TrustOption;

// getProjectTrustOptions
static size_t trust_options(Arena *a, const char *cwd, bool session_only, TrustOption *out) {
    char *tp = norm_cwd(a, cwd);
    char *parent = code_path_dirname(a, tp);
    size_t n = 0;
    out[n++] = (TrustOption){"Trust", true, {{tp, 1}}, 1, tp};
    if (strcmp(parent, tp))
        out[n++] = (TrustOption){arena_printf(a, "Trust parent folder (%s)", parent), true, {{parent, 1}, {tp, -1}}, 2, parent};
    if (session_only) out[n++] = (TrustOption){"Trust (this session only)", true, {{0}}, 0, NULL};
    out[n++] = (TrustOption){"Do not trust", false, {{tp, 0}}, 1, tp};
    if (session_only) out[n++] = (TrustOption){"Do not trust (this session only)", false, {{0}}, 0, NULL};
    return n;
}

// ---- drawing helpers -------------------------------------------------------------------------

static void border(Lines *out, int width) {
    Buf b = {0}, r = {0};
    tui_repeat(&r, "\xe2\x94\x80", MAX(1, width));
    th_fg(&b, T_BORDER, buf_str(&r));
    lines_addb(out, &b);
    buf_free(&r);
}

static void text_into(Str text, int px, int width, Lines *out) {
    Comp *c = comp_text(text, px, 0);
    comp_render(c, width, out);
    comp_free(c);
}

// keyHint / rawKeyHint
static void key_hint(Buf *b, const char *key, const char *desc) {
    th_fgz(b, T_DIM, key);
    Buf d = {0};
    buf_printf(&d, " %s", desc);
    th_fg(b, T_MUTED, buf_str(&d));
    buf_free(&d);
}

static void hints(Lines *out, int width, const char *confirm) {
    Buf h = {0};
    key_hint(&h, "\xe2\x86\x91\xe2\x86\x93", "navigate");
    buf_adds(&h, "  ");
    key_hint(&h, "enter", confirm);
    buf_adds(&h, "  ");
    key_hint(&h, "escape/ctrl+c", "cancel");
    text_into(buf_str(&h), 1, width, out);
    buf_free(&h);
}

static bool key_up(const Key *k) { return k->kind == K_UP || (k->kind == K_CHAR && k->ch == 'k'); }
static bool key_down(const Key *k) { return k->kind == K_DOWN || (k->kind == K_CHAR && k->ch == 'j'); }
static bool key_cancel(const Key *k) { return k->kind == K_ESC || (k->kind == K_CTRL && k->ch == 'c'); }

// ---- the startup prompt (showStartupSelector with ExtensionSelectorComponent) ----------------------

// chalk's bold: closed before each line break and opened again after it.
static void chalk_bold(Buf *b, Str text) {
    buf_adds(b, "\x1b[1m");
    for (size_t i = 0; i < text.len; i++) {
        if (text.ptr[i] == '\n') buf_adds(b, "\x1b[22m\n\x1b[1m");
        else buf_addc(b, text.ptr[i]);
    }
    buf_adds(b, "\x1b[22m");
}

// The component's texts keep the theme of the moment they were built (pi's Text holds a string):
// the title and the hints from the start, the list from its last change; the borders draw with
// the current theme.
typedef struct {
    const char *title;
    TrustOption opts[5];
    size_t n, selected;
    Buf title_text, hint_text, list_text[6];
    size_t nlist;
} Prompt;

// updateList: at most 10 rows, centered on the selection, and the position when they do not all fit
static void prompt_list(Prompt *p) {
    for (size_t i = 0; i < p->nlist; i++) buf_free(&p->list_text[i]);
    p->nlist = 0;
    size_t max_visible = 10;
    size_t start = p->selected > max_visible / 2 ? p->selected - max_visible / 2 : 0;
    start = p->n > max_visible ? MIN(start, p->n - max_visible) : 0;
    size_t end = MIN(start + max_visible, p->n);
    for (size_t i = start; i < end && p->nlist < COUNT(p->list_text); i++) {
        Buf *l = &p->list_text[p->nlist++];
        *l = (Buf){0};
        if (i == p->selected) {
            th_fgz(l, T_ACCENT, "\xe2\x86\x92 ");
            th_fgz(l, T_ACCENT, p->opts[i].label);
        } else {
            buf_adds(l, "  ");
            th_fgz(l, T_TEXT, p->opts[i].label);
        }
    }
}

static void prompt_build(Prompt *p) {
    Buf bold = {0};
    chalk_bold(&bold, str(p->title));
    th_fg(&p->title_text, T_ACCENT, buf_str(&bold));
    buf_free(&bold);
    key_hint(&p->hint_text, "\xe2\x86\x91\xe2\x86\x93", "navigate");
    buf_adds(&p->hint_text, "  ");
    key_hint(&p->hint_text, "enter", "select");
    buf_adds(&p->hint_text, "  ");
    key_hint(&p->hint_text, "escape/ctrl+c", "cancel");
    prompt_list(p);
}

static void prompt_free(Prompt *p) {
    buf_free(&p->title_text);
    buf_free(&p->hint_text);
    for (size_t i = 0; i < p->nlist; i++) buf_free(&p->list_text[i]);
}

static void prompt_render(const Prompt *p, int width, Lines *out) {
    border(out, width);
    lines_addz(out, "");
    text_into(buf_str(&p->title_text), 1, width, out);
    lines_addz(out, "");
    for (size_t i = 0; i < p->nlist; i++) text_into(buf_str(&p->list_text[i]), 1, width, out);
    if (p->n > 10) {
        Buf l = {0};
        char c[48];
        snprintf(c, sizeof c, "  (%zu/%zu)", p->selected + 1, p->n);
        th_fgz(&l, T_MUTED, c);
        text_into(buf_str(&l), 0, width, out);
        buf_free(&l);
    }
    lines_addz(out, "");
    text_into(buf_str(&p->hint_text), 1, width, out);
    lines_addz(out, "");
    border(out, width);
}

// The macOS appearance for the saetta theme: 1 dark, 2 light.
static int mac_appearance(void) {
    static const char *const dr[] = {"defaults", "read", "-g", "AppleInterfaceStyle", NULL};
    Buf v = {0};
    bool dark = tui_run((char *const *)dr, NULL, S(""), &v, 1000) && v.len >= 4 && !strncmp(v.ptr, "Dark", 4);
    buf_free(&v);
    return dark ? 1 : 2;
}

// The selection's index, or -1 when cancelled.
static int run_prompt(Prompt *p, const char *theme) {
    int appearance = mac_appearance();
    if (!theme || !strcmp(theme, "system")) theme_set(NULL, true);
    else theme_use(theme, NULL, appearance);
    prompt_build(p);
    Screen scr = {0};
    screen_start(&scr);
    KeyIn kin = {0};
    int result = -1;
    bool done = false, dirty = true;
    while (!done) {
        if (dirty) {
            Lines frame = {0};
            prompt_render(p, scr.cols, &frame);
            screen_draw(&scr, &frame, false);
            lines_free(&frame);
            dirty = false;
        }
        int timeout = scr.color_query ? 10 : 1000;
        int kt = keyin_timeout_ms(&kin, now_ns());
        if (kt >= 0) timeout = MIN(timeout, kt);
        struct pollfd pf = {0, POLLIN, 0};
        int r = poll(&pf, 1, timeout);
        if (r < 0 && errno != EINTR) break;
        if (r > 0 && (pf.revents & (POLLIN | POLLHUP))) {
            char in[4096];
            ssize_t n = read(0, in, sizeof in);
            if (n <= 0) break;
            keyin_feed(&kin, in, (size_t)n);
        }
        Key k;
        while (!done && keyin_next(&kin, now_ns(), &k)) {
            if (k.kind == K_REPLY_KITTY || k.kind == K_REPLY_DA || k.kind == K_REPLY_COLOR) screen_reply(&scr, &k);
            else if (key_up(&k)) {
                if (p->selected > 0) p->selected--;
                prompt_list(p);
                dirty = true;
            } else if (key_down(&k)) {
                if (p->selected + 1 < p->n) p->selected++;
                prompt_list(p);
                dirty = true;
            } else if (k.kind == K_ENTER || (k.kind == K_CHAR && k.ch == '\n')) {
                result = (int)p->selected;
                done = true;
            } else if (key_cancel(&k)) done = true;
            free(k.paste);
        }
        // the terminal's colors (queryStartupTerminalColors): the theme set again when the query ends
        // (what came, possibly nothing), the request render redraws what is drawn on demand
        if (scr.color_query && (scr.color_done || now_ns() - scr.color_query_ns >= 100000000ull)) {
            scr.color_query = scr.color_done = false;
            bool have = scr.colors.has_fg || scr.colors.has_bg || scr.colors.has_palette;
            theme_use(theme ? theme : "system", have ? &scr.colors : NULL, have && scr.colors.has_bg ? 0 : appearance);
            dirty = true;
        }
    }
    keyin_free(&kin);
    Lines empty = {0};
    screen_draw(&scr, &empty, true); // clearStartupTui
    screen_stop(&scr);
    lines_free(&scr.prev);
    prompt_free(p);
    return result;
}

// ---- the session's trust (main.ts createRuntime, project-trust.ts resolveProjectTrusted) -------------

static const char *trust_title(Arena *a, const char *cwd) {
    // formatProjectTrustPrompt
    return arena_printf(a,
                        "Trust project folder?\n%s\n\nThis allows Saetta Code to load .pi settings and resources, install "
                        "missing project packages, and execute project extensions.",
                        cwd);
}

// resolveProjectTrusted for a cwd with project resources and no override: the store, then
// defaultProjectTrust, then the prompt when interactive (selectProjectTrustOption). *ask (when
// given) is set when it comes to the prompt.
static bool resolve(Arena *a, const CodeEnv *env, const char *cwd, bool interactive, bool *ask) {
    const char *agent_dir = code_path_resolve(a, cwd, env->agent_dir);
    int stored = store_get(a, agent_dir, cwd);
    if (stored >= 0) return stored;
    JValue *settings = code_read_json_object(a, code_path_join(a, agent_dir, "settings.json"));
    const char *def = json_str(json_get(settings, "defaultProjectTrust"));
    if (def && !strcmp(def, "always")) return true;
    if (def && !strcmp(def, "never")) return false;
    if (ask) *ask = true;
    if (!interactive) return false; // no UI to ask
    // selectProjectTrustOption
    Prompt p = {.title = trust_title(a, cwd)};
    p.n = trust_options(a, cwd, true, p.opts);
    int sel = run_prompt(&p, json_str(json_get(settings, "theme")));
    if (sel < 0) return false;
    if (p.opts[sel].trusted) {
        pthread_mutex_lock(&g_tmu);
        approve_l(norm_cwd(a, cwd));
        pthread_mutex_unlock(&g_tmu);
    }
    if (p.opts[sel].nupdates) {
        const char *err = store_set_many(a, agent_dir, p.opts[sel].updates, p.opts[sel].nupdates);
        if (err) fprintf(stderr, "%s\n", err);
    }
    return p.opts[sel].trusted;
}

// createRuntime's projectTrusted for cwd: projectTrustByCwd's decision, else --approve /
// --no-approve, else (project resources) resolveProjectTrusted, remembered for the cwd; else the
// store, else trusted.
static bool runtime_trust(Arena *a, const CodeEnv *env, const char *cwd, bool interactive) {
    bool has = code_trust_has_project_resources(a, cwd, env->home);
    pthread_mutex_lock(&g_tmu);
    int cached = cached_l(cwd);
    pthread_mutex_unlock(&g_tmu);
    if (cached >= 0) return cached;
    if (g_override >= 0) return g_override;
    if (!has) {
        int stored = store_get(a, code_path_resolve(a, cwd, env->agent_dir), cwd);
        return stored >= 0 ? stored : true;
    }
    bool trusted = resolve(a, env, cwd, interactive, NULL);
    pthread_mutex_lock(&g_tmu);
    cache_l(cwd, trusted);
    pthread_mutex_unlock(&g_tmu);
    return trusted;
}

static void set_current(const char *cwd, bool trusted) {
    pthread_mutex_lock(&g_tmu);
    free(g_cur_cwd);
    g_cur_cwd = strdup(cwd);
    g_trusted = trusted;
    pthread_mutex_unlock(&g_tmu);
}

// The hand-off of a restart on another model (code_main.c restart_on): the current folder's
// decision, for that folder only ("<0|1>:<path>"), taken out of the environment at once so no tool
// inherits it. --approve would trust every folder of the next process.
#define TRUST_ENV "SAETTA_CODE_PROJECT_TRUST"

// "2:" is a trust the user approved in this session (a prompt): the next process keeps it as such.
void code_trust_export_restart(const CodeEnv *env) {
    Arena a = {0};
    bool t = code_trust_trusted(env);
    const char *cwd = code_path_resolve(&a, "/", env->cwd);
    pthread_mutex_lock(&g_tmu);
    bool approved = t && approved_l(norm_cwd(&a, cwd));
    pthread_mutex_unlock(&g_tmu);
    setenv(TRUST_ENV, arena_printf(&a, "%d:%s", approved ? 2 : t, cwd), 1);
    arena_reset(&a);
}

void code_trust_cancel_restart(void) { unsetenv(TRUST_ENV); }

void code_trust_resolve(const CodeEnv *env, bool interactive) {
    Arena a = {0};
    const char *cwd = code_path_resolve(&a, "/", env->cwd);
    const char *handed = getenv(TRUST_ENV);
    // the hand-off applies to its own folder only: the real paths compared (a symbolic link, or
    // a ".." in either, names the same folder; another folder never inherits it)
    if (handed && handed[0] >= '0' && handed[0] <= '2' && handed[1] == ':' && handed[2] == '/') {
        const char *real = norm_cwd(&a, cwd);
        if (!strcmp(norm_cwd(&a, handed + 2), real)) {
            pthread_mutex_lock(&g_tmu);
            cache_l(cwd, handed[0] != '0');
            if (handed[0] == '2') approve_l(real);
            pthread_mutex_unlock(&g_tmu);
        }
    }
    unsetenv(TRUST_ENV);
    // autoTrustOnReloadCwd: the session's first folder, when it has no project resources
    if (g_override < 0 && !code_trust_has_project_resources(&a, cwd, env->home)) {
        free(g_auto_cwd);
        g_auto_cwd = strdup(cwd);
    }
    set_current(cwd, runtime_trust(&a, env, cwd, interactive));
    arena_reset(&a);
}

bool code_trust_trusted(const CodeEnv *env) {
    Arena a = {0};
    const char *cwd = code_path_resolve(&a, "/", env->cwd);
    pthread_mutex_lock(&g_tmu);
    int t = g_cur_cwd && !strcmp(g_cur_cwd, cwd) ? g_trusted : -1;
    pthread_mutex_unlock(&g_tmu);
    if (t < 0) {
        // a runtime in another folder (a resumed or imported session's): its own trust, with no UI
        // here (the interactive mode asked first: code_trust_switch_ask); never the previous folder's
        t = runtime_trust(&a, env, cwd, false);
        set_current(cwd, t);
    }
    arena_reset(&a);
    return t;
}

// The current runtime's folder (the last one resolved), else the configuration's.
static const char *current_cwd(Arena *a) {
    pthread_mutex_lock(&g_tmu);
    const char *c = g_cur_cwd ? arena_strdup(a, g_cur_cwd) : NULL;
    pthread_mutex_unlock(&g_tmu);
    return c ? c : code_path_resolve(a, "/", code_ui_config()->cwd);
}

static bool current_trusted(void) {
    pthread_mutex_lock(&g_tmu);
    bool t = g_trusted > 0;
    pthread_mutex_unlock(&g_tmu);
    return t;
}

// Whether the UI may run git's index refresh in cwd (the footer's git status): the trust store's
// nearest decision for it is "trusted", the user trusted it in this session through a prompt, or
// --approve. Never by default: a folder without project resources is trusted for pi's resources,
// yet its .git/config (filter drivers, ...) is the repository's own code.
bool code_trust_explicit(const char *cwd) {
    const CodeUiConfig *cfg = code_ui_config();
    if (g_override >= 0) return g_override;
    Arena a = {0};
    const char *c = code_path_resolve(&a, "/", cwd);
    const char *real = norm_cwd(&a, c);
    pthread_mutex_lock(&g_tmu);
    bool ok = approved_l(real);
    pthread_mutex_unlock(&g_tmu);
    if (!ok) ok = store_get(&a, code_path_resolve(&a, c, cfg->agent_dir), c) == 1;
    arena_reset(&a);
    return ok;
}

// ---- the question on a session switch (AgentSessionRuntime.switchSession with
// projectTrustContextFactory: interactive-mode.ts createProjectTrustContext, showExtensionSelector) --

bool code_trust_switch_asks(const char *cwd) {
    const CodeUiConfig *cfg = code_ui_config();
    Arena a = {0};
    const char *c = code_path_resolve(&a, "/", cwd);
    pthread_mutex_lock(&g_tmu);
    int cached = cached_l(c);
    pthread_mutex_unlock(&g_tmu);
    bool ask = false;
    if (g_override < 0 && cached < 0 && code_trust_has_project_resources(&a, c, cfg->home)) {
        CodeEnv env = {.cwd = c, .agent_dir = cfg->agent_dir, .home = cfg->home};
        (void)resolve(&a, &env, c, false, &ask);
    }
    arena_reset(&a);
    return ask;
}

typedef struct {
    Arena a;
    Prompt p;
    const char *cwd, *agent_dir;
    void (*done)(void *);
    void (*dropped)(void *); // the panel closed without an answer (replaced, or the UI ending)
    void *ctx;
    bool answered;
} SwitchAsk;

static void ask_render(void *ctx, int width, Lines *out) {
    SwitchAsk *s = ctx;
    prompt_render(&s->p, width, out);
}

static void ask_destroy(void *ctx) {
    SwitchAsk *s = ctx;
    // no answer: the switch is not made, its context freed
    if (!s->answered && s->dropped) s->dropped(s->ctx);
    prompt_free(&s->p);
    arena_reset(&s->a);
    free(s);
}

static void ask_key(void *ctx, const Key *k) {
    SwitchAsk *s = ctx;
    Prompt *p = &s->p;
    if (key_up(k)) {
        if (p->selected > 0) p->selected--;
        prompt_list(p);
    } else if (key_down(k)) {
        if (p->selected + 1 < p->n) p->selected++;
        prompt_list(p);
    } else if (k->kind == K_ENTER || (k->kind == K_CHAR && k->ch == '\n') || key_cancel(k)) {
        // resolveProjectTrusted's end: the choice saved (saveProjectTrustPromptResult), cancel is
        // untrusted; projectTrustByCwd.set either way
        bool trusted = false;
        const char *err = NULL;
        if (!key_cancel(k)) {
            const TrustOption *o = &p->opts[p->selected];
            trusted = o->trusted;
            if (o->nupdates) err = store_set_many(&s->a, s->agent_dir, o->updates, o->nupdates);
        }
        pthread_mutex_lock(&g_tmu);
        cache_l(s->cwd, trusted);
        if (trusted) approve_l(norm_cwd(&s->a, s->cwd));
        pthread_mutex_unlock(&g_tmu);
        char *errc = err ? strdup(err) : NULL;
        void (*done)(void *) = s->done;
        void *dctx = s->ctx;
        s->answered = true;
        code_ui_panel_close(); // frees s
        if (errc) code_ui_warning(errc);
        free(errc);
        done(dctx);
    }
}

void code_trust_switch_ask(const char *cwd, void (*done)(void *), void (*dropped)(void *), void *ctx) {
    const CodeUiConfig *cfg = code_ui_config();
    SwitchAsk *s = xcalloc(1, sizeof *s);
    Arena *a = &s->a;
    s->cwd = code_path_resolve(a, "/", cwd);
    s->agent_dir = code_path_resolve(a, s->cwd, cfg->agent_dir);
    s->done = done;
    s->dropped = dropped;
    s->ctx = ctx;
    s->p.title = trust_title(a, s->cwd);
    s->p.n = trust_options(a, s->cwd, true, s->p.opts);
    prompt_build(&s->p);
    code_ui_panel_open((CodeUiPanel){ask_render, ask_key, ask_destroy, s});
}

// ---- /trust (showTrustSelector, TrustSelectorComponent) ------------------------------------------

typedef struct {
    Arena a;
    const char *cwd, *agent_dir;
    TrustOption opts[3];
    size_t n, selected;
    bool has_saved, saved_decision;
    const char *saved_path;
} TrustSel;

static bool is_saved(const TrustSel *t, const TrustOption *o) {
    return o->saved_path && t->has_saved && t->saved_decision == o->trusted && !strcmp(t->saved_path, o->saved_path);
}

static void trust_render(void *ctx, int width, Lines *out) {
    TrustSel *t = ctx;
    Arena *a = &t->a;
    border(out, width);
    lines_addz(out, "");
    Buf b = {0}, bold = {0};
    th_bold(&bold, S("Project trust"));
    th_fg(&b, T_ACCENT, buf_str(&bold));
    text_into(buf_str(&b), 1, width, out);
    b.len = 0;
    th_fgz(&b, T_MUTED, t->cwd);
    text_into(buf_str(&b), 1, width, out);
    lines_addz(out, "");
    // formatDecision
    const char *decision = "none";
    if (t->has_saved) {
        const char *label = t->saved_decision ? "trusted" : "untrusted";
        const char *tp = t->n ? t->opts[0].saved_path : NULL;
        decision = tp && strcmp(t->saved_path, tp) ? arena_printf(a, "%s (inherited from %s)", label, t->saved_path)
                                                   : arena_printf(a, "%s (%s)", label, t->saved_path);
    }
    b.len = 0;
    th_fgz(&b, T_MUTED, arena_printf(a, "Saved decision: %s", decision));
    text_into(buf_str(&b), 1, width, out);
    b.len = 0;
    th_fgz(&b, T_MUTED, arena_printf(a, "Current session: %s", current_trusted() ? "trusted" : "untrusted"));
    text_into(buf_str(&b), 1, width, out);
    lines_addz(out, "");
    for (size_t i = 0; i < t->n; i++) {
        bool sel = i == t->selected;
        Buf l = {0};
        if (sel) th_fgz(&l, T_ACCENT, "\xe2\x86\x92 ");
        else buf_adds(&l, "  ");
        if (is_saved(t, &t->opts[i])) th_fgz(&l, T_ACCENT, "\xe2\x9c\x93 ");
        else buf_adds(&l, "  ");
        th_fgz(&l, sel ? T_ACCENT : T_TEXT, t->opts[i].label);
        text_into(buf_str(&l), 1, width, out);
        buf_free(&l);
    }
    lines_addz(out, "");
    hints(out, width, "save");
    lines_addz(out, "");
    border(out, width);
    buf_free(&b);
    buf_free(&bold);
}

static void trust_key(void *ctx, const Key *k) {
    TrustSel *t = ctx;
    if (key_up(k)) {
        if (t->selected > 0) t->selected--;
    } else if (key_down(k)) {
        if (t->selected + 1 < t->n) t->selected++;
    } else if (k->kind == K_ENTER || (k->kind == K_CHAR && k->ch == '\n')) {
        const TrustOption *o = &t->opts[t->selected];
        bool trusted = o->trusted;
        const char *err = store_set_many(&t->a, t->agent_dir, o->updates, o->nupdates);
        code_ui_panel_close(); // frees t
        if (err) code_ui_warning(err);
        else
            code_ui_status(trusted ? "Saved trust decision: trusted. Restart Saetta Code for this to take effect."
                                   : "Saved trust decision: untrusted. Restart Saetta Code for this to take effect.");
    } else if (key_cancel(k))
        code_ui_panel_close();
}

static void trust_destroy(void *ctx) {
    TrustSel *t = ctx;
    arena_reset(&t->a);
    free(t);
}

void code_trust_command(void) {
    const CodeUiConfig *cfg = code_ui_config();
    TrustSel *t = xcalloc(1, sizeof *t);
    Arena *a = &t->a;
    t->cwd = current_cwd(a); // sessionManager.getCwd()
    t->agent_dir = code_path_resolve(a, t->cwd, cfg->agent_dir);
    const char *err;
    t->has_saved = store_entry(a, t->agent_dir, t->cwd, &t->saved_path, &t->saved_decision, &err);
    if (t->has_saved) t->saved_path = arena_strdup(a, t->saved_path);
    if (err) {
        code_ui_warning(err);
        trust_destroy(t);
        return;
    }
    t->n = trust_options(a, t->cwd, false, t->opts);
    for (size_t i = 0; i < t->n; i++)
        if (is_saved(t, &t->opts[i])) {
            t->selected = i;
            break;
        }
    code_ui_panel_open((CodeUiPanel){trust_render, trust_key, trust_destroy, t});
}

// ---- the warning and the reload ---------------------------------------------------------------------

// ThemedText(() => theme.fg("warning", ...), 1, 0): drawn with the theme of the moment.
static void warning_render(void *ctx, int width, Lines *out) {
    (void)ctx;
    Buf b = {0};
    th_fgz(&b, T_WARNING,
           "This project is not trusted. Project .pi resources and packages are ignored. Use /trust to save a trust decision, then "
           "restart Saetta Code.");
    text_into(buf_str(&b), 1, width, out);
    buf_free(&b);
}

void code_trust_warning(void) {
    const CodeUiConfig *cfg = code_ui_config();
    Arena a = {0};
    bool has = code_trust_has_project_resources(&a, current_cwd(&a), cfg->home);
    arena_reset(&a);
    if (current_trusted() || !has) return;
    if (code_ui_chat_len()) code_ui_chat_add(comp_spacer(1));
    code_ui_chat_add(comp_fn(warning_render, NULL));
}

bool code_trust_after_reload(const CodeEnv *env) {
    Arena a = {0};
    const char *cwd = code_path_resolve(&a, "/", env->cwd);
    bool saved = false;
    if (g_auto_cwd && !strcmp(g_auto_cwd, cwd) && code_trust_trusted(env) && code_trust_has_project_resources(&a, cwd, env->home)) {
        const char *agent_dir = code_path_resolve(&a, cwd, env->agent_dir);
        if (store_get(&a, agent_dir, cwd) < 0) {
            TrustUpdate u = {cwd, 1};
            const char *err = store_set_many(&a, agent_dir, &u, 1);
            if (err) code_ui_warning(arena_printf(&a, "Could not save project trust after reload: %s", err));
            else saved = true;
        }
        if (saved || store_get(&a, agent_dir, cwd) >= 0) {
            free(g_auto_cwd);
            g_auto_cwd = NULL;
        }
    }
    arena_reset(&a);
    return saved;
}
