// Saetta Code's skills (see code_res.h): which skills load and from where, the /skill:name
// commands, the loaded-resources listing at startup, and /reload.
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at v1.5.2,
// b4afddd): packages/coding-agent/src/core/package-manager.ts (resolve, resolveLocalEntries,
// addAutoDiscoveredResources, applyPatterns, isEnabledByOverrides, matchesAnyPattern,
// matchesAnyExactPattern, toResolvedPaths, resourcePrecedenceRank), core/resource-loader.ts
// (reload's skill part, mergePaths, mapSkillPath, updateSkillsFromPaths, findSourceInfoForPath,
// getDefaultSourceInfoForPath), core/skills.ts (loadSkills, loadSkillFromFile, validateName,
// validateDescription, createSkillSourceInfo), core/source-info.ts, core/settings-manager.ts
// (migrateSettings' skills part, getEnableSkillCommands), core/agent-session.ts
// (_expandSkillCommand, parseSkillBlock), modes/interactive/interactive-mode.ts
// (showLoadedResources, buildScopeGroups, formatScopeGroups, formatDiagnostics,
// formatPathWithSource, getDisplaySourceInfo, getScopeGroup, formatDisplayPath,
// formatContextPath, getAutocompleteSourceTag, prefixAutocompleteDescription, the skill
// commands of createBaseAutocompleteProvider, handleReloadCommand), components/
// skill-invocation-message.ts, utils/paths.ts (resolvePath, shortenHomePath).
//
// Not ported: settings "packages" entries (npm and git packages and their skills), extension
// skill paths, the [Prompts], [Extensions] and [Theme conflicts] sections (Saetta Code has no
// prompt templates, extensions or theme files), minimatch's full glob syntax (fnmatch, with "**"
// across folders, is used), a YAML parse error's own message (a generic one is shown), and the
// reload box pi draws for the moment a reload takes.
#include "code_res.h"
#include "code_res_ui.h"

#include <fnmatch.h>
#include <stdlib.h>
#include <sys/stat.h>

// ---- options and state ----------------------------------------------------------------------------

static bool g_no_skills;
static const char *const *g_skill_paths;
static size_t g_nskill_paths;
static CodeResources g_res;
static Arena *g_res_arena; // the current load's strings
static const CodeEnv *g_env;

void code_skills_options(bool no_skills, const char *const *paths, size_t n) {
    g_no_skills = no_skills;
    g_skill_paths = paths;
    g_nskill_paths = n;
}

// ---- paths (utils/paths.ts) -------------------------------------------------------------------------

// normalizePath(input, {trim, expandTilde}) then resolvePath against base.
static char *resolve_input(Arena *a, const char *input, const char *base, const char *home) {
    Str t = code_js_trim(str(input));
    char *p = arena_strndup(a, t.ptr, t.len);
    if (!strcmp(p, "~")) p = arena_strdup(a, home);
    else if (!strncmp(p, "~/", 2)) p = code_path_join(a, home, p + 2);
    else if (!strncmp(p, "file://", 7)) p = p + 7;
    return code_path_resolve(a, base, p);
}

// path.relative(from, to) for absolute, resolved paths.
static char *path_relative(Arena *a, const char *from, const char *to) {
    // the common leading folders, then ".." for each of from's other ones, then to's rest
    const char *f = from, *t = to;
    size_t fi = 0, ti = 0, fcommon = 0, tcommon = 0;
    for (;;) {
        while (f[fi] == '/') fi++;
        while (t[ti] == '/') ti++;
        size_t fe = fi, te = ti;
        while (f[fe] && f[fe] != '/') fe++;
        while (t[te] && t[te] != '/') te++;
        if (fe == fi || te == ti || fe - fi != te - ti || memcmp(f + fi, t + ti, fe - fi)) break;
        fi = fcommon = fe;
        ti = tcommon = te;
    }
    Buf b = {0};
    for (const char *p = f + fcommon; *p;) {
        while (*p == '/') p++;
        if (!*p) break;
        buf_adds(&b, b.len ? "/.." : "..");
        while (*p && *p != '/') p++;
    }
    const char *tail = t + tcommon;
    while (*tail == '/') tail++;
    if (*tail) {
        if (b.len) buf_addc(&b, '/');
        buf_adds(&b, tail);
    }
    char *out = arena_strndup(a, b.ptr ? b.ptr : "", b.len);
    buf_free(&b);
    return out;
}

// shortenHomePath: a leading home folder as "~".
static const char *display_path(Arena *a, const char *p, const char *home) {
    if (!home || !home[0] || p[0] != '/' || !code_path_within(home, p)) return p;
    const char *rel = path_relative(a, code_path_resolve(a, "/", home), p);
    return rel[0] ? arena_printf(a, "~/%s", rel) : "~";
}

// formatContextPath: relative to cwd inside it, else the display path.
static const char *context_path(Arena *a, const char *p, const char *cwd, const char *home) {
    const char *abs = code_path_resolve(a, cwd, p);
    const char *rel = path_relative(a, cwd, abs);
    bool inside = !rel[0] || (strcmp(rel, "..") && strncmp(rel, "../", 3) && rel[0] != '/');
    if (inside) return rel[0] ? rel : ".";
    return display_path(a, abs, home);
}

// getResourcePathKey: the real path.
static char *path_key(Arena *a, const char *p) { return code_path_real(a, code_path_resolve(a, "/", p)); }

// ---- settings (settings-manager.ts) ----------------------------------------------------------------

typedef struct {
    JValue *global, *project; // project only when trusted
} Settings;

// migrateSettings: the old {enableSkillCommands, customDirectories} skills object.
static void migrate_skills(Arena *a, JValue *s) {
    JValue *sk = (JValue *)json_get(s, "skills");
    if (!json_is(sk, J_OBJECT)) return;
    const JValue *esc = json_get(sk, "enableSkillCommands");
    if (esc && !json_get(s, "enableSkillCommands")) json_put(a, s, S("enableSkillCommands"), (JValue *)esc);
    const JValue *dirs = json_get(sk, "customDirectories");
    if (json_is(dirs, J_ARRAY) && dirs->arr.len) json_put(a, s, S("skills"), (JValue *)dirs);
    else json_put(a, s, S("skills"), NULL);
}

static Settings load_settings(Arena *a, const char *cwd, const char *agent_dir, bool trusted) {
    Settings st = {code_read_json_object(a, code_path_join(a, agent_dir, "settings.json")), NULL};
    if (trusted) st.project = code_read_json_object(a, code_path_join(a, cwd, ".pi/settings.json"));
    if (st.global) migrate_skills(a, st.global);
    if (st.project) migrate_skills(a, st.project);
    return st;
}

// isMergeableObject, deepMergeObjects (settings-manager.ts): base's keys, each override's value
// in place (both objects: merged), into a new object.
static JValue *deep_merge(Arena *a, const JValue *base, const JValue *over) {
    JValue *r = json_new(a, J_OBJECT);
    for (size_t i = 0; json_is(base, J_OBJECT) && i < base->obj.len; i++) json_put(a, r, base->obj.items[i].key, base->obj.items[i].val);
    for (size_t i = 0; json_is(over, J_OBJECT) && i < over->obj.len; i++) {
        JValue *ov = over->obj.items[i].val;
        const JValue *bv = json_get(base, over->obj.items[i].key.ptr);
        json_put(a, r, over->obj.items[i].key, json_is(bv, J_OBJECT) && json_is(ov, J_OBJECT) ? deep_merge(a, bv, ov) : ov);
    }
    return r;
}

// SettingsManager.create's settings: deepMergeSettings(global, project), the project's only when
// the folder is trusted.
JValue *code_settings_merged(Arena *a, const CodeEnv *env) {
    const char *cwd = code_path_resolve(a, "/", env->cwd);
    Settings st = load_settings(a, cwd, code_path_resolve(a, cwd, env->agent_dir), code_trust_trusted(env));
    return deep_merge(a, st.global, st.project);
}

// the string entries of a settings array ("skills")
static void string_entries(const JValue *arr, VecStr *out) {
    for (size_t i = 0; json_is(arr, J_ARRAY) && i < arr->arr.len; i++)
        if (json_is(arr->arr.items[i], J_STRING)) vec_push(out, arr->arr.items[i]->s);
}

// ---- patterns (package-manager.ts) -------------------------------------------------------------

// minimatch(path, pattern) with fnmatch: "*" stays in one folder, "**" crosses folders, a leading
// dot is matched only by a pattern that names it.
static bool mm(const char *path, Str pattern) {
    char pat[4096];
    if (pattern.len >= sizeof pat) return false;
    memcpy(pat, pattern.ptr, pattern.len);
    pat[pattern.len] = 0;
    // normalizePath, then "./" dropped
    const char *p = pat;
    if (!strncmp(p, "./", 2)) p += 2;
    int flags = strstr(p, "**") ? FNM_PERIOD : FNM_PATHNAME | FNM_PERIOD;
    return fnmatch(p, path, flags) == 0;
}

// matchesAnyPattern: the relative path, the name, the full path, and for a SKILL.md its folder's.
static bool matches_any(Arena *a, const char *file, const Str *pats, size_t n, const char *base) {
    const char *rel = path_relative(a, base, file), *name = code_path_basename(file);
    bool skill = !strcmp(name, "SKILL.md");
    const char *pdir = skill ? code_path_dirname(a, file) : NULL;
    const char *prel = skill ? path_relative(a, base, pdir) : NULL, *pname = skill ? code_path_basename(pdir) : NULL;
    for (size_t i = 0; i < n; i++) {
        if (mm(rel, pats[i]) || mm(name, pats[i]) || mm(file, pats[i])) return true;
        if (skill && (mm(prel, pats[i]) || mm(pname, pats[i]) || mm(pdir, pats[i]))) return true;
    }
    return false;
}

// matchesAnyExactPattern
static bool matches_exact(Arena *a, const char *file, const Str *pats, size_t n, const char *base) {
    if (!n) return false;
    const char *rel = path_relative(a, base, file), *name = code_path_basename(file);
    bool skill = !strcmp(name, "SKILL.md");
    const char *pdir = skill ? code_path_dirname(a, file) : NULL;
    const char *prel = skill ? path_relative(a, base, pdir) : NULL;
    for (size_t i = 0; i < n; i++) {
        Str p = pats[i];
        if (str_starts(p, S("./"))) p = str_sub(p, 2, p.len);
        if (str_eqz(p, rel) || str_eqz(p, file)) return true;
        if (skill && (str_eqz(p, prel) || str_eqz(p, pdir))) return true;
    }
    return false;
}

typedef struct {
    VecStr inc, exc, force_inc, force_exc;
} Patterns;

static void split_patterns(const VecStr *entries, Patterns *p, VecStr *plain) {
    for (size_t i = 0; i < entries->len; i++) {
        Str e = entries->ptr[i];
        bool pattern = str_starts(e, S("!")) || str_starts(e, S("+")) || str_starts(e, S("-")) || str_find_char(e, '*', 0) >= 0 ||
                       str_find_char(e, '?', 0) >= 0;
        if (!pattern) {
            if (plain) vec_push(plain, e);
            continue;
        }
        if (str_starts(e, S("+"))) vec_push(&p->force_inc, str_sub(e, 1, e.len));
        else if (str_starts(e, S("-"))) vec_push(&p->force_exc, str_sub(e, 1, e.len));
        else if (str_starts(e, S("!"))) vec_push(&p->exc, str_sub(e, 1, e.len));
        else vec_push(&p->inc, e);
    }
}

static void patterns_free(Patterns *p) {
    vec_free(&p->inc);
    vec_free(&p->exc);
    vec_free(&p->force_inc);
    vec_free(&p->force_exc);
}

// isEnabledByOverrides: only the "!", "+" and "-" entries count.
static bool enabled_by_overrides(Arena *a, const char *file, const VecStr *entries, const char *base) {
    Patterns p = {0};
    split_patterns(entries, &p, NULL);
    bool on = true;
    if (p.exc.len && matches_any(a, file, p.exc.ptr, p.exc.len, base)) on = false;
    if (p.force_inc.len && matches_exact(a, file, p.force_inc.ptr, p.force_inc.len, base)) on = true;
    if (p.force_exc.len && matches_exact(a, file, p.force_exc.ptr, p.force_exc.len, base)) on = false;
    patterns_free(&p);
    return on;
}

// ---- the package manager's skill resolution (resolve) ----------------------------------------------

typedef struct {
    const char *path;
    CodeSourceInfo meta; // PathMetadata: source, scope, origin, baseDir
    bool enabled;
} Res;
typedef VEC(Res) VecRes;

// addResource: the first of a path wins.
static void add_resource(VecRes *acc, const char *path, CodeSourceInfo meta, bool enabled) {
    if (!path || !path[0]) return;
    for (size_t i = 0; i < acc->len; i++)
        if (!strcmp(acc->ptr[i].path, path)) return;
    vec_push(acc, ((Res){path, meta, enabled}));
}

// collectFilesFromPaths(paths, "skills")
static void collect_files(Arena *a, const char *const *paths, size_t n, CodePaths *out) {
    for (size_t i = 0; i < n; i++) {
        if (!code_path_exists(paths[i])) continue;
        if (code_path_isfile(paths[i])) vec_push(out, (char *)paths[i]);
        else if (code_path_isdir(paths[i])) code_collect_skill_entries(a, paths[i], false, out);
    }
}

// resolveLocalEntries: the settings entries of one scope, relative to its folder.
static void resolve_local(Arena *a, const VecStr *entries, CodeSourceInfo meta, const char *base, const char *home, VecRes *acc) {
    if (!entries->len) return;
    VecStr plain = {0};
    Patterns p = {0};
    split_patterns(entries, &p, &plain);
    CodePaths resolved = {0}, files = {0};
    for (size_t i = 0; i < plain.len; i++) vec_push(&resolved, resolve_input(a, arena_strndup(a, plain.ptr[i].ptr, plain.ptr[i].len), base, home));
    collect_files(a, (const char *const *)resolved.ptr, resolved.len, &files);
    // applyPatterns: includes (all without any), excludes, force-includes, force-excludes
    for (size_t i = 0; i < files.len; i++) {
        const char *f = files.ptr[i];
        bool on = !p.inc.len || matches_any(a, f, p.inc.ptr, p.inc.len, base);
        if (on && p.exc.len && matches_any(a, f, p.exc.ptr, p.exc.len, base)) on = false;
        if (!on && p.force_inc.len && matches_exact(a, f, p.force_inc.ptr, p.force_inc.len, base)) on = true;
        if (on && p.force_exc.len && matches_exact(a, f, p.force_exc.ptr, p.force_exc.len, base)) on = false;
        add_resource(acc, f, meta, on);
    }
    vec_free(&plain);
    patterns_free(&p);
    vec_free(&resolved);
    vec_free(&files);
}

static void add_auto(Arena *a, const char *dir, bool mode_agents, CodeSourceInfo meta, const VecStr *overrides, VecRes *acc) {
    CodePaths files = {0};
    code_collect_skill_entries(a, dir, mode_agents, &files);
    for (size_t i = 0; i < files.len; i++) add_resource(acc, files.ptr[i], meta, enabled_by_overrides(a, files.ptr[i], overrides, meta.base_dir));
    vec_free(&files);
}

// resourcePrecedenceRank: project settings, project auto, user settings, user auto.
static int rank(const CodeSourceInfo *m) {
    return (!strcmp(m->scope, "project") ? 0 : 2) + (!strcmp(m->source, "local") ? 0 : 1);
}

// DefaultPackageManager.resolve, the skills: the resources in precedence order, the first of a
// real path kept.
static void resolve_skills(Arena *a, const char *cwd, const char *agent_dir, const char *home, bool trusted, const Settings *st,
                           VecRes *out) {
    VecRes acc = {0};
    VecStr g = {0}, p = {0};
    string_entries(json_get(st->global, "skills"), &g);
    string_entries(json_get(st->project, "skills"), &p);
    const char *project_base = code_path_join(a, cwd, ".pi");
    resolve_local(a, &p, (CodeSourceInfo){"local", "project", "top-level", NULL}, project_base, home, &acc);
    resolve_local(a, &g, (CodeSourceInfo){"local", "user", "top-level", NULL}, agent_dir, home, &acc);
    // addAutoDiscoveredResources
    const char *user_agents = code_path_join(a, home, ".agents/skills");
    if (trusted) add_auto(a, code_path_join(a, project_base, "skills"), false, (CodeSourceInfo){"auto", "project", "top-level", project_base}, &p, &acc);
    if (trusted) {
        // collectAncestorAgentsSkillDirs: cwd up to the git repository's root (or /)
        const char *git_root = NULL;
        for (const char *d = cwd;;) {
            if (code_path_exists(code_path_join(a, d, ".git"))) {
                git_root = d;
                break;
            }
            const char *up = code_path_dirname(a, d);
            if (!strcmp(up, d)) break;
            d = up;
        }
        for (const char *d = cwd;;) {
            const char *sd = code_path_join(a, d, ".agents/skills");
            if (strcmp(sd, user_agents)) {
                const char *base = code_path_dirname(a, sd);
                add_auto(a, sd, true, (CodeSourceInfo){"auto", "project", "top-level", base}, &p, &acc);
            }
            if (git_root && !strcmp(d, git_root)) break;
            const char *up = code_path_dirname(a, d);
            if (!strcmp(up, d)) break;
            d = up;
        }
    }
    add_auto(a, code_path_join(a, agent_dir, "skills"), false, (CodeSourceInfo){"auto", "user", "top-level", agent_dir}, &g, &acc);
    add_auto(a, user_agents, true, (CodeSourceInfo){"auto", "user", "top-level", code_path_dirname(a, user_agents)}, &g, &acc);
    // toResolvedPaths: a stable sort by rank, then the first of each real path
    Res *v = acc.ptr;
    for (size_t i = 1; i < acc.len; i++) // insertion sort keeps the order of equal ranks
        for (size_t k = i; k > 0 && rank(&v[k - 1].meta) > rank(&v[k].meta); k--) {
            Res t = v[k];
            v[k] = v[k - 1];
            v[k - 1] = t;
        }
    CodePaths seen = {0};
    for (size_t i = 0; i < acc.len; i++) {
        char *key = path_key(a, v[i].path);
        bool dup = false;
        for (size_t k = 0; k < seen.len && !dup; k++) dup = !strcmp(seen.ptr[k], key);
        if (dup) continue;
        vec_push(&seen, key);
        vec_push(out, v[i]);
    }
    vec_free(&seen);
    vec_free(&acc);
    vec_free(&g);
    vec_free(&p);
}

// ---- loading the skills (skills.ts loadSkills, includeDefaults false) -----------------------------

typedef VEC(CodeSkill) VecSkill;
typedef VEC(CodeDiag) VecDiag;

static void diag(Arena *a, VecDiag *d, const char *type, const char *message, const char *path) {
    vec_push(d, ((CodeDiag){.type = type, .message = arena_strdup(a, message), .path = path}));
}

// validateName
static void validate_name(Arena *a, Str name, const char *path, VecDiag *d) {
    if (name.len > 64) diag(a, d, "warning", arena_printf(a, "name exceeds 64 characters (%zu)", tui_js_len(name)), path);
    bool ok = name.len > 0;
    for (size_t i = 0; i < name.len; i++) {
        char c = name.ptr[i];
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-')) ok = false;
    }
    if (!ok) diag(a, d, "warning", "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)", path);
    if (str_starts(name, S("-")) || str_ends(name, S("-"))) diag(a, d, "warning", "name must not start or end with a hyphen", path);
    if (str_find(name, S("--"), 0) >= 0) diag(a, d, "warning", "name must not contain consecutive hyphens", path);
}

// createSkillSourceInfo
static CodeSourceInfo skill_source(const char *source, const char *base_dir) {
    if (!strcmp(source, "user")) return (CodeSourceInfo){"local", "user", "top-level", base_dir};
    if (!strcmp(source, "project")) return (CodeSourceInfo){"local", "project", "top-level", base_dir};
    return (CodeSourceInfo){"local", "temporary", "top-level", base_dir};
}

// loadSkillFromFile: a skill, or false (with its diagnostics).
static bool load_skill_file(Arena *a, const char *path, const char *source, CodeSkill *out, VecDiag *d) {
    bool declared = !strcmp(code_path_basename(path), "SKILL.md");
    Str raw;
    if (!code_read_text(a, path, &raw)) {
        diag(a, d, "warning", "failed to read skill file", path);
        return false;
    }
    CodeFrontmatter fm;
    if (!code_frontmatter(a, raw, &fm)) {
        if (declared) diag(a, d, "warning", "failed to parse skill file", path);
        return false;
    }
    bool has_desc = fm.has_description && !code_js_blank(fm.description);
    if (!declared && !has_desc) return false;
    char *dir = code_path_dirname(a, path);
    // validateDescription
    if (!has_desc) diag(a, d, "warning", "description is required", path);
    else if (tui_js_len(fm.description) > 1024)
        diag(a, d, "warning", arena_printf(a, "description exceeds 1024 characters (%zu)", tui_js_len(fm.description)), path);
    Str name = fm.has_name && fm.name.len ? fm.name : str(code_path_basename(dir));
    validate_name(a, name, path, d);
    if (!has_desc) return false;
    *out = (CodeSkill){name, fm.description, path, dir, fm.disable, skill_source(source, dir)};
    return true;
}

typedef struct {
    VecSkill skills;
    VecDiag diags, collisions;
    CodePaths reals;
} Loaded;

// loadSkills' addSkills: the first of a real path, the first of a name (later ones collide).
static void add_skill(Arena *a, Loaded *l, CodeSkill s) {
    char *real = path_key(a, s.file);
    for (size_t i = 0; i < l->reals.len; i++)
        if (!strcmp(l->reals.ptr[i], real)) return;
    for (size_t i = 0; i < l->skills.len; i++)
        if (str_eq(l->skills.ptr[i].name, s.name)) {
            vec_push(&l->collisions, ((CodeDiag){"collision", arena_printf(a, "name \"%.*s\" collision", (int)s.name.len, s.name.ptr),
                                                 s.file, s.name, l->skills.ptr[i].file, s.file}));
            return;
        }
    vec_push(&l->skills, s);
    vec_push(&l->reals, real);
}

static void load_skills(Arena *a, const char *cwd, const char *agent_dir, const char *home, const CodePaths *paths, Loaded *l) {
    const char *user_dir = code_path_join(a, agent_dir, "skills"), *project_dir = code_path_join(a, cwd, ".pi/skills");
    for (size_t i = 0; i < paths->len; i++) {
        const char *p = resolve_input(a, paths->ptr[i], cwd, home);
        if (!code_path_exists(p)) {
            diag(a, &l->diags, "warning", "skill path does not exist", p);
            continue;
        }
        // getSource (includeDefaults false)
        const char *source = code_path_within(user_dir, p) ? "user" : code_path_within(project_dir, p) ? "project" : "path";
        if (code_path_isdir(p)) {
            CodePaths files = {0};
            code_collect_skill_entries(a, p, false, &files);
            for (size_t k = 0; k < files.len; k++) {
                CodeSkill s;
                if (load_skill_file(a, files.ptr[k], source, &s, &l->diags)) add_skill(a, l, s);
            }
            vec_free(&files);
        } else if (code_path_isfile(p) && str_ends(str(p), S(".md"))) {
            CodeSkill s;
            if (load_skill_file(a, p, source, &s, &l->diags)) add_skill(a, l, s);
        } else
            diag(a, &l->diags, "warning", "skill path is not a markdown file", p);
    }
}

// resource-loader.ts findSourceInfoForPath over the resolved resources' metadata, then the
// skill's own, then getDefaultSourceInfoForPath.
static CodeSourceInfo source_for(Arena *a, const CodeSkill *s, const VecRes *meta) {
    const char *norm = code_path_resolve(a, "/", s->file);
    for (size_t i = 0; i < meta->len; i++)
        if (!strcmp(meta->ptr[i].path, norm) || !strcmp(meta->ptr[i].path, s->file)) return meta->ptr[i].meta;
    for (size_t i = 0; i < meta->len; i++)
        if (code_path_within(meta->ptr[i].path, norm)) return meta->ptr[i].meta;
    return s->si;
}

// resource-loader.ts reload, the skills and the context files.
static void load_all(Arena *a, const CodeEnv *env, CodeResources *res) {
    const char *cwd = code_path_resolve(a, "/", env->cwd);
    const char *agent_dir = code_path_resolve(a, cwd, env->agent_dir);
    const char *home = env->home;
    bool trusted = code_trust_trusted(env);
    Settings st = load_settings(a, cwd, agent_dir, trusted);
    VecRes resolved = {0};
    resolve_skills(a, cwd, agent_dir, home, trusted, &st, &resolved);
    // mergePaths(enabled, --skill paths): resolved, the first of a real path
    CodePaths paths = {0}, keys = {0};
    for (size_t pass = 0; pass < 2; pass++) {
        size_t n = pass ? g_nskill_paths : (g_no_skills ? 0 : resolved.len);
        for (size_t i = 0; i < n; i++) {
            if (!pass && !resolved.ptr[i].enabled) continue;
            const char *p = pass ? resolve_input(a, g_skill_paths[i], cwd, home) : resolved.ptr[i].path;
            char *key = path_key(a, p);
            bool dup = false;
            for (size_t k = 0; k < keys.len && !dup; k++) dup = !strcmp(keys.ptr[k], key);
            if (dup) continue;
            vec_push(&keys, key);
            vec_push(&paths, (char *)p);
        }
    }
    Loaded l = {0};
    if (!(g_no_skills && !paths.len)) load_skills(a, cwd, agent_dir, home, &paths, &l);
    // --skill paths that do not exist: an error unless already a diagnostic
    for (size_t i = 0; i < g_nskill_paths; i++) {
        const char *p = resolve_input(a, g_skill_paths[i], cwd, home);
        if (code_path_exists(p)) continue;
        bool known = false;
        for (size_t k = 0; k < l.diags.len && !known; k++) known = l.diags.ptr[k].path && !strcmp(l.diags.ptr[k].path, p);
        for (size_t k = 0; k < l.collisions.len && !known; k++) known = !strcmp(l.collisions.ptr[k].path, p);
        if (!known) vec_push(&l.collisions, ((CodeDiag){.type = "error", .message = "Skill path does not exist", .path = p}));
    }
    res->nskills = l.skills.len;
    res->skills = ARENA_ARRAY(a, CodeSkill, l.skills.len + 1);
    for (size_t i = 0; i < l.skills.len; i++) {
        res->skills[i] = l.skills.ptr[i];
        res->skills[i].si = source_for(a, &l.skills.ptr[i], &resolved);
    }
    res->ndiags = l.diags.len + l.collisions.len;
    res->diags = ARENA_ARRAY(a, CodeDiag, res->ndiags + 1);
    for (size_t i = 0; i < l.diags.len; i++) res->diags[i] = l.diags.ptr[i];
    for (size_t i = 0; i < l.collisions.len; i++) res->diags[l.diags.len + i] = l.collisions.ptr[i];
    res->context = (CodePaths){0};
    code_prompt_context_paths(a, env, trusted, &res->context);
    res->loaded = true;
    vec_free(&l.skills);
    vec_free(&l.diags);
    vec_free(&l.collisions);
    vec_free(&l.reals);
    vec_free(&paths);
    vec_free(&keys);
    vec_free(&resolved);
}

const CodeResources *code_resources(const CodeEnv *env) {
    if (!g_res.loaded) {
        g_env = env;
        g_res_arena = xcalloc(1, sizeof *g_res_arena);
        load_all(g_res_arena, env, &g_res);
    }
    return &g_res;
}

void code_resources_reload(const CodeEnv *env) {
    Arena *old = g_res_arena;
    CodePaths old_ctx = g_res.context;
    g_res = (CodeResources){0};
    g_env = env;
    g_res_arena = xcalloc(1, sizeof *g_res_arena);
    load_all(g_res_arena, env, &g_res);
    vec_free(&old_ctx);
    // the previous load's strings stay valid until the next reload (the autocomplete list
    // pointed at them until it was rebuilt)
    static Arena *older;
    if (older) {
        arena_reset(older);
        free(older);
    }
    older = old;
}

// ---- /skill:name (agent-session.ts) ----------------------------------------------------------------

const char *code_skill_expand(Arena *a, const CodeEnv *env, const char *text, const char **error) {
    if (error) *error = NULL;
    if (strncmp(text, "/skill:", 7)) return text;
    const char *sp = strchr(text, ' ');
    Str name = sp ? str_n(text + 7, (size_t)(sp - text - 7)) : str(text + 7);
    Str args = sp ? code_js_trim(str(sp + 1)) : S("");
    const CodeResources *res = code_resources(env);
    const CodeSkill *skill = NULL;
    for (size_t i = 0; i < res->nskills && !skill; i++)
        if (str_eq(res->skills[i].name, name)) skill = &res->skills[i];
    if (!skill) return text; // an unknown skill passes through
    Str content, body;
    if (!code_read_text(a, skill->file, &content)) {
        if (error) *error = "failed to read skill file";
        return text;
    }
    if (!code_frontmatter_body(a, content, &body)) {
        if (error) *error = "failed to parse skill file";
        return text;
    }
    body = code_js_trim(body);
    char *block = arena_printf(a, "<skill name=\"%.*s\" location=\"%s\">\nReferences are relative to %s.\n\n%.*s\n</skill>", (int)skill->name.len,
                               skill->name.ptr, skill->file, skill->base_dir, (int)body.len, body.ptr);
    return args.len ? arena_printf(a, "%s\n\n%.*s", block, (int)args.len, args.ptr) : block;
}

// parseSkillBlock: the regex
// ^<skill name="([^"]+)" location="([^"]+)">\n([\s\S]*?)\n<\/skill>(?:\n\n([\s\S]+))?$
static bool parse_skill_block(Str t, Str *name, Str *content, Str *user) {
    if (!str_starts(t, S("<skill name=\""))) return false;
    size_t i = 13, n0 = i;
    while (i < t.len && t.ptr[i] != '"') i++;
    if (i == n0 || i >= t.len) return false;
    *name = str_sub(t, n0, i);
    if (!str_starts(str_sub(t, i, t.len), S("\" location=\""))) return false;
    i += 12;
    size_t l0 = i;
    while (i < t.len && t.ptr[i] != '"') i++;
    if (i == l0 || i >= t.len || !str_starts(str_sub(t, i, t.len), S("\">\n"))) return false;
    size_t c0 = i + 3;
    // the lazy content: the first "\n</skill>" the rest of the text accepts
    for (ptrdiff_t e = str_find(t, S("\n</skill>"), c0 > 0 ? c0 - 1 : 0); e >= 0; e = str_find(t, S("\n</skill>"), (size_t)e + 1)) {
        if ((size_t)e < c0 - 1) continue;
        size_t after = (size_t)e + 9;
        Str rest = str_sub(t, after, t.len);
        *content = (size_t)e >= c0 ? str_sub(t, c0, (size_t)e) : S("");
        if (!rest.len) {
            *user = S("");
            return true;
        }
        if (str_starts(rest, S("\n\n")) && rest.len > 2) {
            *user = code_js_trim(str_sub(rest, 2, rest.len));
            return true;
        }
    }
    return false;
}

// ---- the skill block in the transcript (SkillInvocationMessageComponent) ---------------------------

typedef struct {
    Comp base;
    Buf name, content;
    int cache_w;
    int cache_exp;
    char *cache_theme;
    Lines lines;
} SkillView;

static void skill_box(Lines *kids, int width, Lines *out) {
    // Box(outputPad 1, 1, customMessageBg)
    for (int k = 0; k < 1; k++) {
        Buf l = {0};
        th_bg_line(&l, TH_BG_CUSTOM_MESSAGE, S(""), width);
        lines_addb(out, &l);
    }
    for (size_t i = 0; i < kids->n; i++) {
        Buf p = {0}, l = {0};
        tui_spaces(&p, 1);
        buf_adds(&p, kids->v[i]);
        th_bg_line(&l, TH_BG_CUSTOM_MESSAGE, buf_str(&p), width);
        lines_addb(out, &l);
        buf_free(&p);
    }
    Buf l = {0};
    th_bg_line(&l, TH_BG_CUSTOM_MESSAGE, S(""), width);
    lines_addb(out, &l);
}

static void text_into(Str text, int px, int width, Lines *out) {
    Comp *c = comp_text(text, px, 0);
    comp_render(c, width, out);
    comp_free(c);
}

static void skill_render(Comp *c, int width, Lines *out) {
    SkillView *v = (SkillView *)c;
    bool expanded = code_ui_expanded();
    const char *theme_key = th_open(T_CUSTOM_LABEL);
    if (v->cache_w != width || v->cache_exp != (int)expanded || !v->cache_theme || strcmp(v->cache_theme, theme_key)) {
        lines_free(&v->lines);
        free(v->cache_theme);
        v->cache_theme = strdup(theme_key);
        v->cache_w = width;
        v->cache_exp = expanded;
        int cw = MAX(1, width - 2);
        Lines kids = {0};
        Buf t = {0};
        if (expanded) {
            th_fgz(&t, T_CUSTOM_LABEL, "\x1b[1m[skill]\x1b[22m");
            text_into(buf_str(&t), 0, cw, &kids);
            Buf md = {0};
            buf_printf(&md, "**%.*s**\n\n", (int)v->name.len, v->name.ptr);
            buf_addstr(&md, buf_str(&v->content));
            MdStyle st = {.has_color = true, .color = T_CUSTOM_TEXT};
            md_render(buf_str(&md), 0, 0, &st, cw, &kids);
            buf_free(&md);
        } else {
            th_fgz(&t, T_CUSTOM_LABEL, "\x1b[1m[skill]\x1b[22m ");
            th_fg(&t, T_CUSTOM_TEXT, buf_str(&v->name));
            th_fgz(&t, T_DIM, " (ctrl+o to expand)");
            text_into(buf_str(&t), 0, cw, &kids);
        }
        buf_free(&t);
        skill_box(&kids, width, &v->lines);
        lines_free(&kids);
    }
    for (size_t i = 0; i < v->lines.n; i++) lines_addz(out, v->lines.v[i]);
}

static void skill_destroy(Comp *c) {
    SkillView *v = (SkillView *)c;
    buf_free(&v->name);
    buf_free(&v->content);
    free(v->cache_theme);
    lines_free(&v->lines);
    free(v);
}

static const CompVt skill_vt = {skill_render, skill_destroy, true};

Comp *code_skill_block_comp(Str text, Str *user_message) {
    Str name, content, user;
    if (!parse_skill_block(text, &name, &content, &user)) return NULL;
    SkillView *v = xcalloc(1, sizeof *v);
    v->base.vt = &skill_vt;
    v->cache_w = -1;
    buf_addstr(&v->name, name);
    buf_addstr(&v->content, content);
    *user_message = user;
    return &v->base;
}

// ---- the "/" list (createBaseAutocompleteProvider) -------------------------------------------------

// getAutocompleteSourceTag for a skill: its scope's letter.
static const char *source_tag(const CodeSourceInfo *si) {
    if (!si->source || !strcmp(si->source, "builtin")) return NULL;
    return !strcmp(si->scope, "user") ? "u" : !strcmp(si->scope, "project") ? "p" : "t";
}

static bool enable_skill_commands(Arena *a, const CodeEnv *env) {
    const char *cwd = code_path_resolve(a, "/", env->cwd);
    Settings st = load_settings(a, cwd, code_path_resolve(a, cwd, env->agent_dir), code_trust_trusted(env));
    const JValue *v = json_get(st.project, "enableSkillCommands");
    if (!json_is(v, J_BOOL)) v = json_get(st.global, "enableSkillCommands");
    return json_is(v, J_BOOL) ? v->b : true;
}

static Arena g_cmd_arena;
static SlashCmd *g_cmds;

const SlashCmd *code_skill_commands(const SlashCmd *builtin, size_t n, size_t *out_n) {
    const CodeEnv *env = g_env;
    Arena *a = &g_cmd_arena;
    Arena tmp = {0};
    bool on = env && enable_skill_commands(&tmp, env);
    arena_reset(&tmp);
    const CodeResources *res = env ? code_resources(env) : NULL;
    size_t k = on && res ? res->nskills : 0;
    free(g_cmds);
    arena_reset(a);
    g_cmds = xcalloc(n + k + 1, sizeof *g_cmds);
    memcpy(g_cmds, builtin, n * sizeof *g_cmds);
    for (size_t i = 0; i < k; i++) {
        const CodeSkill *s = &res->skills[i];
        const char *tag = source_tag(&s->si);
        g_cmds[n + i].name = arena_printf(a, "skill:%.*s", (int)s->name.len, s->name.ptr);
        g_cmds[n + i].description = tag ? arena_printf(a, "[%s] %.*s", tag, (int)s->description.len, s->description.ptr)
                                        : arena_strndup(a, s->description.ptr, s->description.len);
    }
    *out_n = n + k;
    return g_cmds;
}

// ---- the loaded resources (showLoadedResources) -----------------------------------------------------

// String.prototype.localeCompare with ICU's root collation, close enough for paths and names:
// punctuation and symbols first (in ICU's order), then digits, then letters case-insensitively;
// ties go to lowercase first.
static int coll_key(unsigned char c) {
    static const char order[] = "_-,;:!?.'\"()[]{}@*/\\&#%`^+<=>|~$";
    if (c == ' ' || c == '\t' || c == '\n') return 0;
    const char *p = c ? strchr(order, c) : NULL;
    if (p) return 1 + (int)(p - order);
    if (c >= '0' && c <= '9') return 100 + (c - '0');
    if (c >= 'a' && c <= 'z') return 200 + (c - 'a');
    if (c >= 'A' && c <= 'Z') return 200 + (c - 'A');
    return 300 + c;
}

static int locale_cmp(const char *x, const char *y) {
    for (size_t i = 0;; i++) {
        unsigned char a = (unsigned char)x[i], b = (unsigned char)y[i];
        if (!a || !b) {
            if (a || b) return a ? 1 : -1;
            break;
        }
        int d = coll_key(a) - coll_key(b);
        if (d) return d;
    }
    for (size_t i = 0; x[i] && y[i]; i++) {
        unsigned char a = (unsigned char)x[i], b = (unsigned char)y[i];
        if (a != b) {
            bool la = a >= 'a' && a <= 'z', lb = b >= 'a' && b <= 'z';
            if (la != lb) return la ? -1 : 1;
            return a < b ? -1 : 1;
        }
    }
    return 0;
}

static int cmp_strs(const void *x, const void *y) { return locale_cmp(*(const char *const *)x, *(const char *const *)y); }

// getScopeGroup
static const char *scope_group(const CodeSourceInfo *si) {
    const char *source = si->source ? si->source : "local", *scope = si->scope ? si->scope : "project";
    if (!strcmp(source, "cli") || !strcmp(scope, "temporary")) return "path";
    if (!strcmp(scope, "user")) return "user";
    if (!strcmp(scope, "project")) return "project";
    return "path";
}

// formatPathWithSource with getDisplaySourceInfo (no package sources here)
static const char *path_with_source(Arena *a, const char *p, const CodeSourceInfo *si, const char *home) {
    if (!si) return display_path(a, p, home);
    const char *source = si->source ? si->source : "local", *scope = si->scope ? si->scope : "project";
    const char *label, *scope_label = NULL;
    if (!strcmp(source, "local")) {
        label = !strcmp(scope, "user") ? "user" : !strcmp(scope, "project") ? "project" : "path";
        if (!strcmp(scope, "temporary")) scope_label = "temp";
    } else if (!strcmp(source, "cli")) {
        label = "path";
        if (!strcmp(scope, "temporary")) scope_label = "temp";
    } else {
        label = source;
        scope_label = !strcmp(scope, "user") ? "user" : !strcmp(scope, "project") ? "project" : !strcmp(scope, "temporary") ? "temp" : NULL;
    }
    const char *shown = display_path(a, p, home);
    return scope_label ? arena_printf(a, "%s (%s) %s", label, scope_label, shown) : arena_printf(a, "%s %s", label, shown);
}

// interactive-mode.ts findSourceInfoForPath: the exact path, else the nearest parent.
static const CodeSourceInfo *diag_source(Arena *a, const CodeResources *res, const char *p) {
    for (char *cur = arena_strdup(a, p);;) {
        for (size_t i = 0; i < res->nskills; i++)
            if (!strcmp(res->skills[i].file, cur)) return &res->skills[i].si;
        char *s = strrchr(cur, '/');
        if (!s) return NULL;
        *s = 0;
    }
}

static void fg_line(Buf *b, ThemeColor c, const char *text) {
    if (b->len) buf_addc(b, '\n');
    th_fgz(b, c, text);
}

// formatDiagnostics
static void format_diags(Arena *a, const CodeResources *res, const char *home, Buf *b) {
    // the collisions, by name in order of first appearance
    for (size_t i = 0; i < res->ndiags; i++) {
        const CodeDiag *d = &res->diags[i];
        if (strcmp(d->type, "collision")) continue;
        bool first = true;
        for (size_t k = 0; k < i; k++)
            if (!strcmp(res->diags[k].type, "collision") && str_eq(res->diags[k].name, d->name)) first = false;
        if (!first) continue;
        fg_line(b, T_WARNING, arena_printf(a, "  \"%.*s\" collision:", (int)d->name.len, d->name.ptr));
        Buf w = {0};
        buf_adds(&w, "    ");
        th_fgz(&w, T_SUCCESS, "\xe2\x9c\x93");
        buf_printf(&w, " %s", path_with_source(a, d->winner, diag_source(a, res, d->winner), home));
        fg_line(b, T_DIM, buf_str(&w).ptr);
        buf_free(&w);
        for (size_t k = i; k < res->ndiags; k++) {
            const CodeDiag *e = &res->diags[k];
            if (strcmp(e->type, "collision") || !str_eq(e->name, d->name)) continue;
            Buf l = {0};
            buf_adds(&l, "    ");
            th_fgz(&l, T_WARNING, "\xe2\x9c\x97");
            buf_printf(&l, " %s (skipped)", path_with_source(a, e->loser, diag_source(a, res, e->loser), home));
            fg_line(b, T_DIM, buf_str(&l).ptr);
            buf_free(&l);
        }
    }
    for (size_t i = 0; i < res->ndiags; i++) {
        const CodeDiag *d = &res->diags[i];
        if (!strcmp(d->type, "collision")) continue;
        ThemeColor c = !strcmp(d->type, "error") ? T_ERROR : T_WARNING;
        if (d->path) {
            fg_line(b, c, arena_printf(a, "  %s", path_with_source(a, d->path, diag_source(a, res, d->path), home)));
            fg_line(b, c, arena_printf(a, "    %s", d->message));
        } else
            fg_line(b, c, arena_printf(a, "  %s", d->message));
    }
}

static void section(Lines *out, int width, const char *name, ThemeColor color, Str body) {
    Buf t = {0};
    Buf h = {0};
    buf_printf(&h, "[%s]", name);
    th_fg(&t, color, buf_str(&h));
    buf_addc(&t, '\n');
    buf_addstr(&t, body);
    text_into(buf_str(&t), 0, width, out);
    lines_addz(out, ""); // Spacer(1)
    buf_free(&t);
    buf_free(&h);
}

static Lines g_list_lines;
static int g_list_w = -1, g_list_exp = -1;
static char *g_list_theme;
static const CodeResources *g_list_res;

void code_res_invalidate(void) { g_list_w = -1; }

// showLoadedResources({force: false, showDiagnosticsWhenQuiet: true}) with quietStartup false.
static void build_listing(int width, bool expanded, Lines *out) {
    const CodeEnv *env = g_env;
    if (!env) return;
    const CodeResources *res = code_resources(env);
    Arena a = {0};
    const char *cwd = code_path_resolve(&a, "/", env->cwd);
    const char *home = env->home;
    Buf body = {0};
    if (res->context.len) {
        lines_addz(out, "");
        if (expanded) {
            for (size_t i = 0; i < res->context.len; i++) {
                if (i) buf_addc(&body, '\n');
                th_fgz(&body, T_DIM, arena_printf(&a, "  %s", display_path(&a, res->context.ptr[i], home)));
            }
        } else {
            Buf l = {0};
            for (size_t i = 0; i < res->context.len; i++) {
                Str t = code_js_trim(str(context_path(&a, res->context.ptr[i], cwd, home)));
                if (!t.len) continue;
                if (l.len) buf_adds(&l, ", ");
                buf_addstr(&l, t);
            }
            th_fgz(&body, T_DIM, arena_printf(&a, "  %.*s", (int)l.len, l.ptr ? l.ptr : ""));
            buf_free(&l);
        }
        section(out, width, "Context", T_MD_HEADING, buf_str(&body));
        body.len = 0;
    }
    if (res->nskills) {
        if (expanded) {
            // buildScopeGroups and formatScopeGroups: project, user, path; paths sorted
            static const char *const groups[] = {"project", "user", "path"};
            for (size_t g = 0; g < COUNT(groups); g++) {
                const char **paths = ARENA_ARRAY(&a, const char *, res->nskills);
                size_t n = 0;
                for (size_t i = 0; i < res->nskills; i++)
                    if (!strcmp(scope_group(&res->skills[i].si), groups[g])) paths[n++] = res->skills[i].file;
                if (!n) continue;
                if (body.len) buf_addc(&body, '\n');
                buf_adds(&body, "  ");
                th_fgz(&body, T_ACCENT, groups[g]);
                qsort(paths, n, sizeof *paths, cmp_strs);
                for (size_t i = 0; i < n; i++) fg_line(&body, T_DIM, arena_printf(&a, "    %s", display_path(&a, paths[i], home)));
            }
        } else {
            const char **names = ARENA_ARRAY(&a, const char *, res->nskills);
            size_t n = 0;
            for (size_t i = 0; i < res->nskills; i++) {
                Str t = code_js_trim(res->skills[i].name);
                if (t.len) names[n++] = arena_strndup(&a, t.ptr, t.len);
            }
            qsort(names, n, sizeof *names, cmp_strs);
            Buf l = {0};
            for (size_t i = 0; i < n; i++) {
                if (i) buf_adds(&l, ", ");
                buf_adds(&l, names[i]);
            }
            th_fgz(&body, T_DIM, arena_printf(&a, "  %.*s", (int)l.len, l.ptr ? l.ptr : ""));
            buf_free(&l);
        }
        section(out, width, "Skills", T_MD_HEADING, buf_str(&body));
        body.len = 0;
    }
    if (res->ndiags) {
        format_diags(&a, res, home, &body);
        section(out, width, "Skill warnings", T_WARNING, buf_str(&body));
        body.len = 0;
    }
    buf_free(&body);
    arena_reset(&a);
}

void code_res_render(int width, bool expanded, Lines *out) {
    const char *theme_key = th_open(T_MD_HEADING);
    const CodeResources *res = g_env ? code_resources(g_env) : NULL;
    if (g_list_w != width || g_list_exp != (int)expanded || g_list_res != res || !g_list_theme || strcmp(g_list_theme, theme_key)) {
        lines_free(&g_list_lines);
        free(g_list_theme);
        g_list_theme = strdup(theme_key);
        g_list_w = width;
        g_list_exp = expanded;
        g_list_res = res;
        build_listing(width, expanded, &g_list_lines);
    }
    for (size_t i = 0; i < g_list_lines.n; i++) lines_addz(out, g_list_lines.v[i]);
}

// ---- /reload (handleReloadCommand) ------------------------------------------------------------------

void code_reload_command(void) {
    if (code_ui_busy()) {
        code_ui_warning("Wait for the current response to finish before reloading.");
        return;
    }
    if (!g_env) return;
    code_resources_reload(g_env);
    code_res_invalidate();
    code_ui_rebuild_chat();
    code_ui_refresh_commands();
    bool saved = code_trust_after_reload(g_env);
    code_ui_status(saved ? "Reloaded keybindings, extensions, skills, prompts, themes, and context files; saved project trust"
                         : "Reloaded keybindings, extensions, skills, prompts, themes, and context files");
}
