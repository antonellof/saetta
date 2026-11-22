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

#include <errno.h>
#include <poll.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static int g_override = -1; // --approve / --no-approve
static int g_trusted = -1;  // the session's decision, once resolved
static char *g_auto_cwd;    // autoTrustOnReloadCwd

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
    const char *lock = arena_printf(a, "%s.lock", path);
    char *dir = code_path_dirname(a, path);
    mkdir(dir, 0755);
    for (int i = 0; i < 10; i++) {
        if (mkdir(lock, 0755) == 0) return true;
        if (errno != EEXIST) return false;
        struct stat st;
        if (stat(lock, &st) == 0 && time(NULL) - st.st_mtime > 10) {
            rmdir(lock);
            continue;
        }
        struct timespec ts = {0, 20000000};
        nanosleep(&ts, NULL);
    }
    return false;
}

static void trust_unlock(Arena *a, const char *path) { rmdir(arena_printf(a, "%s.lock", path)); }

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
        const char *tmp = arena_printf(a, "%s.tmp-%d", tp, (int)getpid());
        FILE *f = fopen(tmp, "w");
        bool ok = f && fwrite(b.ptr, 1, b.len, f) == b.len;
        if (f && fclose(f)) ok = false;
        if (ok && rename(tmp, tp)) ok = false;
        if (!ok) {
            unlink(tmp);
            err = arena_printf(a, "Could not write %s", tp);
        }
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

static bool resolve(Arena *a, const CodeEnv *env, bool interactive) {
    const char *cwd = code_path_resolve(a, "/", env->cwd);
    const char *agent_dir = code_path_resolve(a, cwd, env->agent_dir);
    bool has = code_trust_has_project_resources(a, cwd, env->home);
    if (g_override < 0 && !has) {
        free(g_auto_cwd);
        g_auto_cwd = strdup(cwd);
    }
    if (g_override >= 0) return g_override;
    int stored = store_get(a, agent_dir, cwd);
    if (!has) return stored != 0;
    if (stored >= 0) return stored;
    JValue *settings = code_read_json_object(a, code_path_join(a, agent_dir, "settings.json"));
    const char *def = json_str(json_get(settings, "defaultProjectTrust"));
    if (def && !strcmp(def, "always")) return true;
    if (def && !strcmp(def, "never")) return false;
    if (!interactive) return false; // no UI to ask
    // selectProjectTrustOption
    Prompt p = {.title = arena_printf(a,
                                      "Trust project folder?\n%s\n\nThis allows Saetta Code to load .pi settings and resources, install "
                                      "missing project packages, and execute project extensions.",
                                      cwd)};
    p.n = trust_options(a, cwd, true, p.opts);
    int sel = run_prompt(&p, json_str(json_get(settings, "theme")));
    if (sel < 0) return false;
    if (p.opts[sel].nupdates) {
        const char *err = store_set_many(a, agent_dir, p.opts[sel].updates, p.opts[sel].nupdates);
        if (err) fprintf(stderr, "%s\n", err);
    }
    return p.opts[sel].trusted;
}

void code_trust_resolve(const CodeEnv *env, bool interactive) {
    Arena a = {0};
    g_trusted = resolve(&a, env, interactive);
    arena_reset(&a);
}

bool code_trust_trusted(const CodeEnv *env) {
    if (g_trusted < 0) code_trust_resolve(env, false);
    return g_trusted;
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
    th_fgz(&b, T_MUTED, arena_printf(a, "Current session: %s", g_trusted > 0 ? "trusted" : "untrusted"));
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
    t->cwd = code_path_resolve(a, "/", cfg->cwd);
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
    bool has = code_trust_has_project_resources(&a, code_path_resolve(&a, "/", cfg->cwd), cfg->home);
    arena_reset(&a);
    if (g_trusted > 0 || !has) return;
    if (code_ui_chat_len()) code_ui_chat_add(comp_spacer(1));
    code_ui_chat_add(comp_fn(warning_render, NULL));
}

bool code_trust_after_reload(const CodeEnv *env) {
    Arena a = {0};
    const char *cwd = code_path_resolve(&a, "/", env->cwd);
    bool saved = false;
    if (g_auto_cwd && !strcmp(g_auto_cwd, cwd) && g_trusted > 0 && code_trust_has_project_resources(&a, cwd, env->home)) {
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
