// Saetta Code system prompt (see code.h): the sections pi builds for a session, from the tool
// set, the context files (AGENTS.md / CLAUDE.md), skills, SYSTEM.md and APPEND_SYSTEM.md.
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at
// v1.5.2, b4afddd): packages/coding-agent/src/core/system-prompt.ts, resource-loader.ts
// (loadProjectContextFiles, SYSTEM.md discovery), skills.ts, package-manager.ts (skill
// discovery), trust-manager.ts and project-trust.ts (project trust), tools/*.ts (prompt
// snippets and guidelines), utils/frontmatter.ts; packages/ai/src/utils/text.ts
// (getSystemMessageText).
//
// Behaviour kept from pi, in print mode:
// - Context files: the agent dir's own, then one per directory from the filesystem root down to
//   cwd (first of AGENTS.override.md, AGENTS.md, AGENTS.MD, CLAUDE.md, CLAUDE.MD), deduped by
//   path, skipping the main repo's file when cwd is in a git worktree nested inside it. They
//   load whether or not the project is trusted.
// - Skills: the ones code_skills.c loads (settings "skills" entries, --skill, --no-skills, the
//   auto-discovered folders, project ones only when the project is trusted).
// - Trust: code_trust.c (trust.json, defaultProjectTrust, --approve, the interactive prompt).
// Not ported: settings "packages" entries and package skills, --no-context-files, extensions
// (their prompt sections, tools, project_trust event, before_agent_start forced prompts). The
// YAML frontmatter parser covers top-level scalars (plain, quoted, block) and treats
// collections as non-strings; the ignore
// files (.gitignore, .ignore, .fdignore) in skill folders are matched with fnmatch, close to
// but not exactly the `ignore` package.
#include "code_res.h"

#include <dirent.h>
#include <fnmatch.h>
#include <limits.h>
#include <stdlib.h>
#include <sys/stat.h>

// ---- paths (node:path semantics on POSIX) ----------------------------------------------

// path.resolve(base, p): absolute, without ".", "..", "//" or a trailing "/".
static char *path_resolve(Arena *a, const char *base, const char *p) {
    Buf in = {0};
    if (p[0] != '/') {
        buf_adds(&in, base ? base : "/");
        buf_addc(&in, '/');
    }
    buf_adds(&in, p);
    size_t n = in.len;
    char *out = arena_alloc(a, n + 2);
    size_t o = 0;
    for (size_t i = 0; i < n;) {
        while (i < n && in.ptr[i] == '/') i++;
        size_t j = i;
        while (j < n && in.ptr[j] != '/') j++;
        size_t len = j - i;
        if (len == 0) break;
        if (len == 1 && in.ptr[i] == '.') {
        } else if (len == 2 && in.ptr[i] == '.' && in.ptr[i + 1] == '.') {
            while (o > 0 && out[o - 1] != '/') o--;
            if (o > 0) o--;
        } else {
            out[o++] = '/';
            memcpy(out + o, in.ptr + i, len);
            o += len;
        }
        i = j;
    }
    if (o == 0) out[o++] = '/';
    out[o] = 0;
    buf_free(&in);
    return out;
}

static char *pjoin(Arena *a, const char *x, const char *y) {
    size_t n = strlen(x);
    return n > 0 && x[n - 1] == '/' ? arena_printf(a, "%s%s", x, y) : arena_printf(a, "%s/%s", x, y);
}

static char *pdirname(Arena *a, const char *p) {
    const char *s = strrchr(p, '/');
    if (!s || s == p) return arena_strdup(a, "/");
    return arena_strndup(a, p, (size_t)(s - p));
}

static const char *pbasename(const char *p) {
    const char *s = strrchr(p, '/');
    return s ? s + 1 : p;
}

static bool p_exists(const char *p) {
    struct stat st;
    return stat(p, &st) == 0;
}

static bool p_isfile(const char *p) {
    struct stat st;
    return stat(p, &st) == 0 && S_ISREG(st.st_mode);
}

// canonicalizePath: realpath, or the path itself when that fails.
static char *p_real(Arena *a, const char *p) {
    char buf[PATH_MAX];
    return realpath(p, buf) ? arena_strdup(a, buf) : arena_strdup(a, p);
}

// isPathWithin(parent, child)
static bool p_within(const char *parent, const char *child) {
    size_t n = strlen(parent);
    if (strcmp(parent, child) == 0) return true;
    if (strncmp(parent, child, n) != 0) return false;
    return (n > 0 && parent[n - 1] == '/') || child[n] == '/';
}

// readFileSync(p, "utf-8") then stripBom. false when the file cannot be read.
static bool read_text(Arena *a, const char *p, Str *out) {
    char *data;
    size_t len;
    if (read_file(p, (size_t)1 << 30, &data, &len)) return false;
    size_t off = len >= 3 && memcmp(data, "\xEF\xBB\xBF", 3) == 0 ? 3 : 0;
    // readFileSync(..., "utf8"): ill-formed UTF-8 becomes U+FFFD
    *out = code_utf8_clean(a, str_n(arena_strndup(a, data + off, len - off), len - off));
    free(data);
    return true;
}

// Directory entries in byte order (what APFS returns to readdirSync), without "." and "..".
typedef struct {
    char **names;
    size_t n;
} DirList;

static int cmp_names(const void *x, const void *y) { return strcmp(*(char *const *)x, *(char *const *)y); }

static bool list_dir(Arena *a, const char *dir, DirList *out) {
    DIR *d = opendir(dir);
    if (!d) return false;
    VEC(char *) v = {0};
    struct dirent *e;
    while ((e = readdir(d))) {
        if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0) continue;
        vec_push(&v, arena_strdup(a, e->d_name));
    }
    closedir(d);
    out->names = ARENA_ARRAY(a, char *, v.len ? v.len : 1);
    if (v.len) memcpy(out->names, v.ptr, v.len * sizeof(char *));
    out->n = v.len;
    vec_free(&v);
    qsort(out->names, out->n, sizeof(char *), cmp_names);
    return true;
}

// Dirent.isFile / isDirectory, following a symlink as pi does (false on a broken one).
static void entry_kind(const char *full, bool *is_file, bool *is_dir, bool *ok) {
    struct stat st;
    *ok = stat(full, &st) == 0;
    *is_file = *ok && S_ISREG(st.st_mode);
    *is_dir = *ok && S_ISDIR(st.st_mode);
}

// ---- context files (resource-loader.ts loadProjectContextFiles) ------------------------

typedef struct {
    const char *path;
    Str content;
} CtxFile;
typedef VEC(CtxFile) VecCtx;

static bool ctx_from_dir(Arena *a, const char *dir, CtxFile *out) {
    static const char *const names[] = {"AGENTS.override.md", "AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD"};
    for (size_t i = 0; i < COUNT(names); i++) {
        char *p = pjoin(a, dir, names[i]);
        if (!p_exists(p) || !p_isfile(p)) continue;
        Str c;
        if (!read_text(a, p, &c)) continue; // pi warns and tries the next name
        *out = (CtxFile){p, c};
        return true;
    }
    return false;
}

static Str file_trim(Arena *a, const char *p, bool *ok) {
    Str s = {0};
    *ok = read_text(a, p, &s);
    return str_trim(s);
}

// footer-data-provider.ts findGitPaths
static bool find_git_paths(Arena *a, const char *cwd, char **repo_dir, char **common_git_dir) {
    char *dir = arena_strdup(a, cwd);
    for (;;) {
        char *git = pjoin(a, dir, ".git");
        struct stat st;
        if (stat(git, &st) == 0) {
            if (S_ISREG(st.st_mode)) {
                bool ok;
                Str c = file_trim(a, git, &ok);
                if (!ok) return false;
                if (str_starts(c, S("gitdir: "))) {
                    char *gd = path_resolve(a, dir, str_z(a, str_trim(str_sub(c, 8, c.len))));
                    if (!p_exists(pjoin(a, gd, "HEAD"))) return false;
                    char *cd = pjoin(a, gd, "commondir");
                    char *common = gd;
                    if (p_exists(cd)) {
                        Str cc = file_trim(a, cd, &ok);
                        if (!ok) return false;
                        common = path_resolve(a, gd, str_z(a, cc));
                    }
                    *repo_dir = dir;
                    *common_git_dir = common;
                    return true;
                }
            } else if (S_ISDIR(st.st_mode)) {
                if (!p_exists(pjoin(a, git, "HEAD"))) return false;
                *repo_dir = dir;
                *common_git_dir = git;
                return true;
            }
        }
        char *parent = pdirname(a, dir);
        if (strcmp(parent, dir) == 0) return false;
        dir = parent;
    }
}

// findShadowedContextFile: the main repo's context file a nested linked worktree's own shadows.
static const char *shadowed_ctx(Arena *a, const char *cwd) {
    char *repo, *common;
    if (!find_git_paths(a, cwd, &repo, &common)) return NULL;
    char *common_real = p_real(a, common), *wt_root = p_real(a, repo);
    char *main_root = pdirname(a, common_real);
    if (!p_within(main_root, wt_root) || strcmp(wt_root, main_root) == 0) return NULL;
    if (strcmp(p_real(a, pjoin(a, main_root, ".git")), common_real) != 0) return NULL;
    CtxFile f;
    if (!ctx_from_dir(a, wt_root, &f)) return NULL;
    return pjoin(a, main_root, pbasename(f.path));
}

static void load_context_files(Arena *a, const char *cwd, const char *agent_dir, VecCtx *out) {
    CtxFile g;
    if (ctx_from_dir(a, agent_dir, &g)) vec_push(out, g);
    const char *shadow = shadowed_ctx(a, cwd);
    VecCtx anc = {0};
    char *dir = arena_strdup(a, cwd);
    for (;;) {
        CtxFile f;
        if (ctx_from_dir(a, dir, &f)) {
            bool shadowed = shadow && strcmp(pjoin(a, p_real(a, dir), pbasename(f.path)), shadow) == 0;
            bool seen = false;
            for (size_t i = 0; i < out->len; i++) seen |= strcmp(out->ptr[i].path, f.path) == 0;
            for (size_t i = 0; i < anc.len; i++) seen |= strcmp(anc.ptr[i].path, f.path) == 0;
            if (!shadowed && !seen) vec_push(&anc, f);
        }
        char *parent = pdirname(a, dir);
        if (strcmp(parent, dir) == 0) break;
        dir = parent;
    }
    for (size_t i = anc.len; i-- > 0;) vec_push(out, anc.ptr[i]); // root first (unshift)
    vec_free(&anc);
}

// ---- ignore files in skill folders (skills.ts / package-manager.ts addIgnoreRules) ------

typedef struct {
    const char *pat;
    bool neg, dir_only, anchored;
} IgRule;
typedef VEC(IgRule) Ignore;

static void ig_add_dir(Arena *a, Ignore *ig, const char *dir, const char *root) {
    static const char *const files[] = {".gitignore", ".ignore", ".fdignore"};
    // relative(root, dir) as a "prefix/"
    const char *rel = strcmp(dir, root) == 0 ? "" : dir + strlen(root) + (root[strlen(root) - 1] == '/' ? 0 : 1);
    char *prefix = rel[0] ? arena_printf(a, "%s/", rel) : "";
    for (size_t f = 0; f < COUNT(files); f++) {
        Str text;
        char *p = pjoin(a, dir, files[f]);
        if (!p_exists(p) || !read_text(a, p, &text)) continue;
        for (size_t i = 0; i <= text.len;) {
            ptrdiff_t e = str_find_char(text, '\n', i);
            size_t end = e < 0 ? text.len : (size_t)e;
            Str line = str_sub(text, i, end);
            if (line.len && line.ptr[line.len - 1] == '\r') line.len--;
            i = end + 1;
            Str t = str_trim(line);
            if (!t.len || (t.ptr[0] == '#')) continue;
            bool neg = false;
            Str pat = line;
            if (str_starts(pat, S("!"))) {
                neg = true;
                pat = str_sub(pat, 1, pat.len);
            } else if (str_starts(pat, S("\\!"))) {
                pat = str_sub(pat, 1, pat.len);
            }
            if (str_starts(pat, S("/"))) pat = str_sub(pat, 1, pat.len);
            while (pat.len && (pat.ptr[pat.len - 1] == ' ' || pat.ptr[pat.len - 1] == '\t')) pat.len--;
            char *full = arena_printf(a, "%s%.*s", prefix, (int)pat.len, pat.ptr);
            size_t fl = strlen(full);
            IgRule r = {.neg = neg};
            if (fl && full[fl - 1] == '/') {
                r.dir_only = true;
                full[--fl] = 0;
            }
            if (!fl) continue;
            r.anchored = strchr(full, '/') != NULL;
            r.pat = full;
            vec_push(ig, r);
        }
    }
}

// ig.ignores(rel) for a file, or rel + "/" for a directory
static bool ig_ignores(const Ignore *ig, const char *rel, bool is_dir) {
    bool ignored = false;
    const char *base = pbasename(rel);
    for (size_t i = 0; i < ig->len; i++) {
        const IgRule *r = &ig->ptr[i];
        if (r->dir_only && !is_dir) continue;
        int flags = strstr(r->pat, "**") ? 0 : FNM_PATHNAME;
        bool m = r->anchored ? fnmatch(r->pat, rel, flags) == 0 : fnmatch(r->pat, base, 0) == 0;
        if (m) ignored = !r->neg;
    }
    return ignored;
}

// ---- skill discovery (package-manager.ts collectSkillEntries) --------------------------

typedef CodePaths VecPath;

static const char *rel_to(const char *root, const char *full) {
    size_t n = strlen(root);
    return full + n + (root[n - 1] == '/' ? 0 : 1);
}

// mode_agents: ~/.agents/skills style (loose .md files only below the root) instead of pi's
// (loose .md files only in the root).
static void collect_skill_entries(Arena *a, const char *dir, bool mode_agents, Ignore *ig, const char *root, VecPath *out) {
    if (!p_exists(dir)) return;
    Ignore own = {0};
    if (!ig) ig = &own;
    if (!root) root = dir;
    ig_add_dir(a, ig, dir, root);
    DirList dl;
    if (!list_dir(a, dir, &dl)) goto done;
    for (size_t i = 0; i < dl.n; i++) {
        if (strcmp(dl.names[i], "SKILL.md") != 0) continue;
        char *full = pjoin(a, dir, dl.names[i]);
        bool f, d, ok;
        entry_kind(full, &f, &d, &ok);
        if (!ok) continue;
        if (f && !ig_ignores(ig, rel_to(root, full), false)) {
            vec_push(out, full);
            goto done;
        }
    }
    for (size_t i = 0; i < dl.n; i++) {
        const char *name = dl.names[i];
        if (name[0] == '.' || strcmp(name, "node_modules") == 0) continue;
        char *full = pjoin(a, dir, name);
        bool f, d, ok;
        entry_kind(full, &f, &d, &ok);
        if (!ok) continue;
        const char *rel = rel_to(root, full);
        size_t nl = strlen(name);
        bool at_root = strcmp(dir, root) == 0;
        if (f && nl >= 3 && strcmp(name + nl - 3, ".md") == 0 && !ig_ignores(ig, rel, false) &&
            (mode_agents ? !at_root : at_root)) {
            vec_push(out, full);
            continue;
        }
        if (!d || ig_ignores(ig, rel, true)) continue;
        collect_skill_entries(a, full, mode_agents, ig, root, out);
    }
done:
    if (ig == &own) vec_free(&own);
}

// ---- YAML frontmatter (the subset skills use) ------------------------------------------

typedef enum { FM_ABSENT, FM_STRING, FM_OTHER } FmKind;
typedef struct {
    FmKind kind;
    Str s;     // FM_STRING
    bool btrue; // FM_OTHER: the boolean true
} FmVal;
typedef struct {
    FmVal name, description, disable;
} Frontmatter;

typedef struct {
    Str *l;
    size_t n;
} Lines;

static size_t indent_of(Str l) {
    size_t i = 0;
    while (i < l.len && l.ptr[i] == ' ') i++;
    return i;
}

static bool blank(Str l) { return str_trim(l).len == 0; }

static void put_utf8(Buf *b, uint32_t c) {
    if (c < 0x80) buf_addc(b, (char)c);
    else if (c < 0x800) {
        buf_addc(b, (char)(0xC0 | (c >> 6)));
        buf_addc(b, (char)(0x80 | (c & 0x3F)));
    } else if (c < 0x10000) {
        buf_addc(b, (char)(0xE0 | (c >> 12)));
        buf_addc(b, (char)(0x80 | ((c >> 6) & 0x3F)));
        buf_addc(b, (char)(0x80 | (c & 0x3F)));
    } else {
        buf_addc(b, (char)(0xF0 | (c >> 18)));
        buf_addc(b, (char)(0x80 | ((c >> 12) & 0x3F)));
        buf_addc(b, (char)(0x80 | ((c >> 6) & 0x3F)));
        buf_addc(b, (char)(0x80 | (c & 0x3F)));
    }
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static bool is_ws(char c) { return c == ' ' || c == '\t'; }

// A quoted scalar starting at text[0] (the quote), across the joined lines in text ("\n"
// between them). Folds line breaks as YAML does. *end is the index after the closing quote.
static bool parse_quoted(Arena *a, Str text, size_t *end, Str *out) {
    char q = text.ptr[0];
    Buf b = {0};
    size_t i = 1;
    for (;;) {
        if (i >= text.len) {
            buf_free(&b);
            return false;
        }
        char c = text.ptr[i];
        if (c == q) {
            if (q == '\'' && i + 1 < text.len && text.ptr[i + 1] == '\'') {
                buf_addc(&b, '\'');
                i += 2;
                continue;
            }
            i++;
            break;
        }
        if (c == '\n') {
            // fold: trailing spaces before, leading after; one break -> space, more -> newlines
            while (b.len && is_ws(b.ptr[b.len - 1])) b.len--;
            size_t breaks = 0;
            while (i < text.len && (text.ptr[i] == '\n' || is_ws(text.ptr[i]))) {
                if (text.ptr[i] == '\n') breaks++;
                i++;
            }
            if (breaks == 1) buf_addc(&b, ' ');
            else
                for (size_t k = 1; k < breaks; k++) buf_addc(&b, '\n');
            continue;
        }
        if (q == '"' && c == '\\') {
            if (i + 1 >= text.len) {
                buf_free(&b);
                return false;
            }
            char e = text.ptr[i + 1];
            i += 2;
            uint32_t cp = 0;
            int nhex = 0;
            switch (e) {
            case '0': buf_addc(&b, 0); break;
            case 'a': buf_addc(&b, '\a'); break;
            case 'b': buf_addc(&b, '\b'); break;
            case 't': case '\t': buf_addc(&b, '\t'); break;
            case 'n': buf_addc(&b, '\n'); break;
            case 'v': buf_addc(&b, '\v'); break;
            case 'f': buf_addc(&b, '\f'); break;
            case 'r': buf_addc(&b, '\r'); break;
            case 'e': buf_addc(&b, 0x1b); break;
            case ' ': buf_addc(&b, ' '); break;
            case '"': buf_addc(&b, '"'); break;
            case '/': buf_addc(&b, '/'); break;
            case '\\': buf_addc(&b, '\\'); break;
            case 'N': put_utf8(&b, 0x85); break;
            case '_': put_utf8(&b, 0xA0); break;
            case 'L': put_utf8(&b, 0x2028); break;
            case 'P': put_utf8(&b, 0x2029); break;
            case 'x': nhex = 2; break;
            case 'u': nhex = 4; break;
            case 'U': nhex = 8; break;
            case '\n': // escaped line break: joins, dropping the next line's indentation
                while (i < text.len && is_ws(text.ptr[i])) i++;
                break;
            default: buf_free(&b); return false;
            }
            if (nhex) {
                for (int k = 0; k < nhex; k++) {
                    int h = i < text.len ? hexval(text.ptr[i]) : -1;
                    if (h < 0) {
                        buf_free(&b);
                        return false;
                    }
                    cp = cp * 16 + (uint32_t)h;
                    i++;
                }
                put_utf8(&b, cp);
            }
            continue;
        }
        buf_addc(&b, c);
        i++;
    }
    *end = i;
    *out = (Str){arena_strndup(a, b.ptr ? b.ptr : "", b.len), b.len};
    buf_free(&b);
    return true;
}

static bool all_digits(Str s, size_t from, const char *set) {
    if (from >= s.len) return false;
    for (size_t i = from; i < s.len; i++)
        if (!strchr(set, s.ptr[i])) return false;
    return true;
}

// A plain scalar the core schema resolves to null, a bool or a number (not a string).
static bool plain_not_string(Str s, bool *btrue) {
    *btrue = str_eqz(s, "true") || str_eqz(s, "True") || str_eqz(s, "TRUE");
    if (*btrue || str_eqz(s, "false") || str_eqz(s, "False") || str_eqz(s, "FALSE")) return true;
    if (!s.len || str_eqz(s, "~") || str_eqz(s, "null") || str_eqz(s, "Null") || str_eqz(s, "NULL")) return true;
    if (str_starts(s, S("0o")) && all_digits(s, 2, "01234567")) return true;
    if (str_starts(s, S("0x")) && all_digits(s, 2, "0123456789abcdefABCDEF")) return true;
    size_t i = 0;
    if (s.ptr[0] == '+' || s.ptr[0] == '-') i++;
    Str r = str_sub(s, i, s.len);
    if (str_eqz(r, ".inf") || str_eqz(r, ".Inf") || str_eqz(r, ".INF")) return true;
    if (!i && (str_eqz(s, ".nan") || str_eqz(s, ".NaN") || str_eqz(s, ".NAN"))) return true;
    // [0-9]+(\.[0-9]*)? | \.[0-9]+, then an optional exponent
    size_t d0 = i;
    while (i < s.len && s.ptr[i] >= '0' && s.ptr[i] <= '9') i++;
    size_t ndig = i - d0;
    if (i < s.len && s.ptr[i] == '.') {
        i++;
        size_t f0 = i;
        while (i < s.len && s.ptr[i] >= '0' && s.ptr[i] <= '9') i++;
        if (!ndig && i == f0) return false;
    } else if (!ndig)
        return false;
    if (i < s.len && (s.ptr[i] == 'e' || s.ptr[i] == 'E')) {
        i++;
        if (i < s.len && (s.ptr[i] == '+' || s.ptr[i] == '-')) i++;
        size_t e0 = i;
        while (i < s.len && s.ptr[i] >= '0' && s.ptr[i] <= '9') i++;
        if (i == e0) return false;
    }
    return i == s.len;
}

// A block scalar (| or >) from its header and the indented lines under its key.
static bool parse_block(Arena *a, Str header, Lines body, size_t parent_indent, Str *out) {
    bool folded = header.ptr[0] == '>';
    char chomp = 'c';
    size_t explicit_ind = 0;
    size_t i = 1;
    for (; i < header.len && !is_ws(header.ptr[i]) && header.ptr[i] != '#'; i++) {
        char c = header.ptr[i];
        if (c == '-' || c == '+') chomp = c;
        else if (c >= '1' && c <= '9') explicit_ind = (size_t)(c - '0');
        else return false;
    }
    Str rest = str_trim(str_sub(header, i, header.len));
    if (rest.len && rest.ptr[0] != '#') return false;
    size_t ind = parent_indent + explicit_ind;
    if (!explicit_ind) {
        ind = 0;
        for (size_t k = 0; k < body.n; k++)
            if (!blank(body.l[k])) {
                ind = indent_of(body.l[k]);
                break;
            }
    }
    Buf b = {0};
    size_t empties = 0;
    int prev = 0; // 0 none, 1 normal, 2 more indented
    for (size_t k = 0; k < body.n; k++) {
        Str l = body.l[k];
        if (blank(l) && l.len <= ind) {
            empties++;
            continue;
        }
        if (blank(l) && !folded) {
            empties++;
            continue;
        }
        Str c = str_sub(l, MIN(ind, l.len), l.len);
        int kind = c.len && is_ws(c.ptr[0]) ? 2 : 1;
        if (!folded) {
            if (prev) buf_addc(&b, '\n');
            for (size_t e = 0; e < empties; e++) buf_addc(&b, '\n');
        } else if (!prev) {
            for (size_t e = 0; e < empties; e++) buf_addc(&b, '\n');
        } else if (prev == 1 && kind == 1) {
            if (empties)
                for (size_t e = 0; e < empties; e++) buf_addc(&b, '\n');
            else
                buf_addc(&b, ' ');
        } else {
            for (size_t e = 0; e < empties + 1; e++) buf_addc(&b, '\n');
        }
        buf_addstr(&b, c);
        prev = kind;
        empties = 0;
    }
    if (prev) {
        if (chomp == 'c') buf_addc(&b, '\n');
        else if (chomp == '+')
            for (size_t e = 0; e < empties + 1; e++) buf_addc(&b, '\n');
    } else if (chomp == '+') {
        for (size_t e = 0; e < empties; e++) buf_addc(&b, '\n');
    }
    *out = (Str){arena_strndup(a, b.ptr ? b.ptr : "", b.len), b.len};
    buf_free(&b);
    return true;
}

// The value of one top-level key: text after "key:" plus the lines indented under it.
static bool parse_value(Arena *a, Str v, Lines body, FmVal *out) {
    v = str_trim(v);
    // anchors are names only; a tag or an alias makes it something we do not model
    if (str_starts(v, S("&"))) {
        size_t k = 0;
        while (k < v.len && !is_ws(v.ptr[k])) k++;
        v = str_trim(str_sub(v, k, v.len));
    }
    if (str_starts(v, S("#"))) v = (Str){"", 0};
    if (!v.len) {
        size_t k = 0;
        while (k < body.n && blank(body.l[k])) k++;
        if (k == body.n) {
            *out = (FmVal){.kind = FM_OTHER}; // null
            return true;
        }
        Str first = str_trim(body.l[k]);
        if (str_starts(first, S("- ")) || str_eqz(first, "-") || str_find(first, S(": "), 0) >= 0 ||
            str_ends(first, S(":")) || first.ptr[0] == '[' || first.ptr[0] == '{') {
            *out = (FmVal){.kind = FM_OTHER}; // a nested collection
            return true;
        }
        if (first.ptr[0] == '"' || first.ptr[0] == '\'' || first.ptr[0] == '|' || first.ptr[0] == '>') {
            v = first;
            body.l += k + 1;
            body.n -= k + 1;
        }
    }
    if (v.len && (v.ptr[0] == '|' || v.ptr[0] == '>')) {
        out->kind = FM_STRING;
        return parse_block(a, v, body, 0, &out->s);
    }
    if (v.len && (v.ptr[0] == '"' || v.ptr[0] == '\'')) {
        Buf t = {0};
        buf_addstr(&t, v);
        for (size_t k = 0; k < body.n; k++) {
            buf_addc(&t, '\n');
            buf_addstr(&t, body.l[k]);
        }
        size_t end;
        Str s;
        bool ok = parse_quoted(a, buf_str(&t), &end, &s);
        if (ok) {
            Str rest = str_trim(str_sub(buf_str(&t), end, t.len));
            ok = !rest.len || rest.ptr[0] == '#';
        }
        buf_free(&t);
        if (!ok) return false;
        *out = (FmVal){.kind = FM_STRING, .s = s};
        return true;
    }
    if (v.len && (v.ptr[0] == '[' || v.ptr[0] == '{' || v.ptr[0] == '*' || v.ptr[0] == '!')) {
        *out = (FmVal){.kind = FM_OTHER};
        return true;
    }
    // plain scalar, possibly over several lines
    Buf b = {0};
    size_t breaks = 0;
    bool ended = false;
    for (size_t k = 0; k <= body.n && !ended; k++) {
        Str l = k == 0 ? v : str_trim(body.l[k - 1]);
        if (k > 0 && !l.len) {
            breaks++;
            continue;
        }
        ptrdiff_t h = str_find(l, S(" #"), 0);
        if (h >= 0 || (k > 0 && l.ptr[0] == '#')) {
            l = str_trim(str_sub(l, 0, h >= 0 ? (size_t)h : 0));
            ended = true;
        }
        if (str_find(l, S(": "), 0) >= 0 || str_ends(l, S(":"))) {
            buf_free(&b);
            return false; // a mapping inside a plain scalar: the yaml package throws
        }
        if (!l.len) continue;
        if (b.len) {
            if (breaks)
                for (size_t e = 0; e < breaks; e++) buf_addc(&b, '\n');
            else
                buf_addc(&b, ' ');
        }
        breaks = 0;
        buf_addstr(&b, l);
    }
    Str s = {arena_strndup(a, b.ptr ? b.ptr : "", b.len), b.len};
    buf_free(&b);
    bool btrue;
    if (plain_not_string(s, &btrue)) *out = (FmVal){.kind = FM_OTHER, .btrue = btrue};
    else *out = (FmVal){.kind = FM_STRING, .s = s};
    return true;
}

// parseFrontmatter: false when the yaml package would throw.
static bool parse_frontmatter(Arena *a, Str content, Frontmatter *fm) {
    *fm = (Frontmatter){0};
    // normalizeNewlines(stripBom(content)); the BOM is already gone
    Buf n = {0};
    for (size_t i = 0; i < content.len; i++) {
        char c = content.ptr[i];
        if (c == '\r') {
            buf_addc(&n, '\n');
            if (i + 1 < content.len && content.ptr[i + 1] == '\n') i++;
        } else
            buf_addc(&n, c);
    }
    Str t = {arena_strndup(a, n.ptr ? n.ptr : "", n.len), n.len};
    buf_free(&n);
    if (!str_starts(t, S("---"))) return true;
    ptrdiff_t end = str_find(t, S("\n---"), 3);
    if (end < 0) return true;
    Str y = str_sub(t, MIN((size_t)4, (size_t)end), (size_t)end);
    if (!y.len) return true;
    VEC(Str) ls = {0};
    for (size_t i = 0; i <= y.len;) {
        ptrdiff_t e = str_find_char(y, '\n', i);
        size_t stop = e < 0 ? y.len : (size_t)e;
        vec_push(&ls, str_sub(y, i, stop));
        i = stop + 1;
    }
    bool ok = true, mapping = false;
    Str seen[64];
    size_t nseen = 0;
    for (size_t i = 0; i < ls.len && ok;) {
        Str l = ls.ptr[i];
        Str tl = str_trim(l);
        if (!tl.len || tl.ptr[0] == '#') {
            i++;
            continue;
        }
        if (indent_of(l) > 0 || l.ptr[0] == '\t') {
            ok = mapping ? false : true; // stray indentation
            if (!mapping) break;
            continue;
        }
        if (str_starts(l, S("- ")) || str_eqz(l, "-") || str_eqz(l, "---") || str_eqz(l, "...")) {
            ok = !str_eqz(l, "---");
            break; // a sequence or a document marker: no mapping keys for us
        }
        Str key;
        size_t after;
        if (l.ptr[0] == '"' || l.ptr[0] == '\'') {
            size_t qe;
            if (!parse_quoted(a, l, &qe, &key) || qe >= l.len || l.ptr[qe] != ':') {
                ok = false;
                break;
            }
            after = qe + 1;
        } else {
            ptrdiff_t c = -1;
            for (size_t k = 0; k < l.len; k++)
                if (l.ptr[k] == ':' && (k + 1 == l.len || is_ws(l.ptr[k + 1]))) {
                    c = (ptrdiff_t)k;
                    break;
                }
            if (c < 0) {
                ok = mapping ? false : true; // a bare scalar document
                break;
            }
            key = str_trim(str_sub(l, 0, (size_t)c));
            after = (size_t)c + 1;
        }
        if (after < l.len && !is_ws(l.ptr[after])) {
            ok = false;
            break;
        }
        mapping = true;
        size_t j = i + 1;
        while (j < ls.len && (blank(ls.ptr[j]) || indent_of(ls.ptr[j]) > 0 || ls.ptr[j].ptr[0] == '\t')) j++;
        // trailing blank lines are not part of the value
        size_t jb = j;
        while (jb > i + 1 && blank(ls.ptr[jb - 1])) jb--;
        Lines body = {ls.ptr + i + 1, jb - i - 1};
        for (size_t k = 0; k < nseen; k++)
            if (str_eq(seen[k], key)) ok = false; // duplicate key
        if (!ok) break;
        if (nseen < COUNT(seen)) seen[nseen++] = key;
        FmVal val;
        if (!parse_value(a, str_sub(l, after, l.len), body, &val)) {
            ok = false;
            break;
        }
        if (str_eqz(key, "name")) fm->name = val;
        else if (str_eqz(key, "description")) fm->description = val;
        else if (str_eqz(key, "disable-model-invocation")) fm->disable = val;
        i = j;
    }
    vec_free(&ls);
    if (!mapping) *fm = (Frontmatter){0};
    return ok;
}

// ---- skills (skills.ts) ----------------------------------------------------------------

typedef struct {
    Str name, description;
    const char *file;
    bool disable;
} Skill;
typedef VEC(Skill) VecSkill;

static bool js_blank(Str s) {
    // String.prototype.trim() === "" for the whitespace that occurs in practice
    for (size_t i = 0; i < s.len; i++) {
        unsigned char c = (unsigned char)s.ptr[i];
        if (c == ' ' || (c >= '\t' && c <= '\r')) continue;
        if (c == 0xC2 && i + 1 < s.len && (unsigned char)s.ptr[i + 1] == 0xA0) {
            i++;
            continue;
        }
        if (c == 0xEF && i + 2 < s.len && (unsigned char)s.ptr[i + 1] == 0xBB && (unsigned char)s.ptr[i + 2] == 0xBF) {
            i += 2;
            continue;
        }
        return false;
    }
    return true;
}

// The skills pi loads (code_skills.c, the resource loader's), in its order.
static void load_skills(const CodeEnv *env, VecSkill *out) {
    const CodeResources *res = code_resources(env);
    for (size_t i = 0; i < res->nskills; i++) {
        const CodeSkill *k = &res->skills[i];
        vec_push(out, ((Skill){k->name, k->description, k->file, k->disable}));
    }
}

static void xml_escape(Buf *b, Str s) {
    for (size_t i = 0; i < s.len; i++) {
        switch (s.ptr[i]) {
        case '&': buf_adds(b, "&amp;"); break;
        case '<': buf_adds(b, "&lt;"); break;
        case '>': buf_adds(b, "&gt;"); break;
        case '"': buf_adds(b, "&quot;"); break;
        case '\'': buf_adds(b, "&apos;"); break;
        default: buf_addc(b, s.ptr[i]);
        }
    }
}

// formatSkillsForPrompt(...).trim(); empty when no skill is visible
static void format_skills(Buf *b, const VecSkill *skills, const char *reader) {
    size_t visible = 0;
    for (size_t i = 0; i < skills->len; i++) visible += !skills->ptr[i].disable;
    if (!visible) return;
    buf_adds(b, "The following skills provide specialized instructions for specific tasks.\n");
    buf_adds(b, strcmp(reader, "read") == 0   ? "Use the read tool to load a skill's file when the task matches its description.\n"
                : strcmp(reader, "bash") == 0 ? "Use bash to load a skill's file when the task matches its description.\n"
                                              : "Load a skill's file when the task matches its description.\n");
    buf_adds(b, "Resolve relative paths against the skill's directory (parent of SKILL.md). Use absolute paths in tool calls.\n\n");
    buf_adds(b, "<available_skills>\n");
    for (size_t i = 0; i < skills->len; i++) {
        const Skill *s = &skills->ptr[i];
        if (s->disable) continue;
        buf_adds(b, "  <skill>\n    <name>");
        xml_escape(b, s->name);
        buf_adds(b, "</name>\n    <description>");
        xml_escape(b, s->description);
        buf_adds(b, "</description>\n    <location>");
        xml_escape(b, str(s->file));
        buf_adds(b, "</location>\n  </skill>\n");
    }
    buf_adds(b, "</available_skills>");
}

static JValue *read_json(Arena *a, const char *path) {
    Str t;
    JValue *v;
    if (!p_exists(path) || !read_text(a, path, &t) || json_parse(a, t, false, &v)) return NULL;
    return json_is(v, J_OBJECT) ? v : NULL;
}

// resolvePromptInput: a file's text when the input names one, else the input; NULL for none
static const char *prompt_input(Arena *a, const char *cwd, const char *input, size_t *len) {
    if (!input || !input[0]) return NULL;
    char *p = path_resolve(a, cwd, input);
    Str t;
    if (p_exists(p) && read_text(a, p, &t)) {
        *len = t.len;
        return t.ptr;
    }
    *len = strlen(input);
    return input;
}

// ---- tools' prompt contributions (core/tools/*.ts) -------------------------------------

typedef struct {
    const char *name, *snippet, *guidelines[3];
} ToolPrompt;

static const ToolPrompt TOOL_PROMPTS[] = {
    {"read", "Read file contents", {"Use read to examine files instead of cat or sed."}},
    {"bash", "Execute bash commands (ls, grep, find, etc.)",
     {"You can inspect PI_* environment variables for current model and session details."}},
    {"edit", "Make precise file edits with exact text replacement, including multiple disjoint edits in one call",
     {"Use edit for targeted changes. Batch changes to one file in a single call.",
      "Each edits[].oldText must exactly and uniquely match the original file. Keep matches short and "
      "non-overlapping; merge nearby changes without including large unchanged regions."}},
    {"write", "Create or overwrite files", {"Use write only for new files or complete rewrites."}},
    {"grep", "Search file contents for patterns (respects .gitignore)", {0}},
    {"find", "Find files by glob pattern (respects .gitignore)", {0}},
    {"ls", "List directory contents", {0}},
    {"powershell", "Execute PowerShell commands",
     {"You can inspect PI_* environment variables for current model and session details."}},
};

static const ToolPrompt *tool_prompt(const char *name) {
    for (size_t i = 0; i < COUNT(TOOL_PROMPTS); i++)
        if (strcmp(TOOL_PROMPTS[i].name, name) == 0) return &TOOL_PROMPTS[i];
    return NULL;
}

static bool has_tool(const char *const *tools, size_t n, const char *name) {
    for (size_t i = 0; i < n; i++)
        if (strcmp(tools[i], name) == 0) return true;
    return false;
}

// buildRules: deduped, trimmed rules as "- rule" lines
static void add_rule(VecStr *rules, Str r) {
    r = str_trim(r);
    if (!r.len) return;
    for (size_t i = 0; i < rules->len; i++)
        if (str_eq(rules->ptr[i], r)) return;
    vec_push(rules, r);
}

static void build_rules(Buf *b, const char *const *tools, size_t n) {
    VecStr rules = {0};
    bool bash = has_tool(tools, n, "bash"), ps = has_tool(tools, n, "powershell");
    if ((bash || ps) && !has_tool(tools, n, "grep") && !has_tool(tools, n, "find") && !has_tool(tools, n, "ls")) {
        add_rule(&rules, bash && ps ? S("Use bash or PowerShell for file operations like listing, searching, and finding files")
                         : ps       ? S("Use PowerShell for file operations like listing, searching, and finding files")
                                    : S("Use bash for file operations like ls, rg, find"));
    }
    for (size_t i = 0; i < n; i++) {
        const ToolPrompt *tp = tool_prompt(tools[i]);
        for (size_t k = 0; tp && k < COUNT(tp->guidelines) && tp->guidelines[k]; k++) add_rule(&rules, str(tp->guidelines[k]));
    }
    add_rule(&rules, S("Be concise in your responses"));
    add_rule(&rules, S("Show file paths clearly when working with files"));
    for (size_t i = 0; i < rules.len; i++) {
        if (i) buf_addc(b, '\n');
        buf_adds(b, "- ");
        buf_addstr(b, rules.ptr[i]);
    }
    vec_free(&rules);
}

// ---- the system message ----------------------------------------------------------------

static void put_section(Arena *a, JValue *sections, const char *name, Str body) {
    if (strcmp(name, "preamble") == 0) {
        json_put(a, sections, S("preamble"), json_string(a, body));
        return;
    }
    Buf b = {0};
    buf_printf(&b, "<%s>\n", name);
    buf_addstr(&b, body);
    buf_printf(&b, "\n</%s>", name);
    json_put(a, sections, str(name), json_string(a, buf_str(&b)));
    buf_free(&b);
}

Err code_system_message(Arena *a, const CodeEnv *env, const char *const *tools, size_t ntools, JValue **out) {
    const char *cwd = path_resolve(a, "/", env->cwd);
    const char *agent_dir = path_resolve(a, cwd, env->agent_dir);
    bool trusted = code_trust_trusted(env);

    VecCtx ctx = {0};
    load_context_files(a, cwd, agent_dir, &ctx);
    VecSkill skills = {0};
    load_skills(env, &skills);

    // the custom prompt: --system-prompt, else .pi/SYSTEM.md (trusted), else agent_dir/SYSTEM.md
    const char *sys_src = env->system_prompt;
    if (!sys_src) {
        char *pp = pjoin(a, cwd, ".pi/SYSTEM.md"), *gp = pjoin(a, agent_dir, "SYSTEM.md");
        sys_src = trusted && p_exists(pp) ? pp : p_exists(gp) ? gp : NULL;
    }
    size_t custom_len = 0;
    const char *custom = prompt_input(a, cwd, sys_src, &custom_len);
    if (custom) { // argv text and files reach pi decoded: ill-formed UTF-8 as U+FFFD
        Str c = code_utf8_clean(a, str_n(custom, custom_len));
        custom = c.ptr;
        custom_len = c.len;
    }

    // the addendum: each --append-system-prompt, else .pi/APPEND_SYSTEM.md (trusted) or the agent dir's
    Buf append = {0};
    const char *const *srcs = env->append_system_prompt;
    size_t nsrc = env->nappend_system_prompt;
    const char *found = NULL;
    if (!nsrc) {
        char *pp = pjoin(a, cwd, ".pi/APPEND_SYSTEM.md"), *gp = pjoin(a, agent_dir, "APPEND_SYSTEM.md");
        found = trusted && p_exists(pp) ? pp : p_exists(gp) ? gp : NULL;
        srcs = &found;
        nsrc = found ? 1 : 0;
    }
    size_t parts = 0;
    for (size_t i = 0; i < nsrc; i++) {
        size_t len;
        const char *t = prompt_input(a, cwd, srcs[i], &len);
        if (!t) continue;
        Str c = code_utf8_clean(a, str_n(t, len));
        if (parts++) buf_adds(&append, "\n\n");
        buf_add(&append, c.ptr, c.len);
    }

    JValue *sections = json_new(a, J_OBJECT);
    Buf b = {0};
    if (custom && custom_len) {
        put_section(a, sections, "preamble", str_n(custom, custom_len));
    } else {
        put_section(a, sections, "preamble",
                    S("You are a coding assistant running inside Saetta Code, a coding agent harness. Help users inspect "
                      "files, run commands, and modify code."));
        size_t listed = 0;
        for (size_t i = 0; i < ntools; i++) {
            const ToolPrompt *tp = tool_prompt(tools[i]);
            if (!tp) continue; // tools without a snippet are not listed
            if (listed++) buf_addc(&b, '\n');
            buf_printf(&b, "- %s: %s", tools[i], tp->snippet);
        }
        if (!listed) buf_adds(&b, "(none)");
        buf_adds(&b, "\n\nAdditional tools may be available.");
        put_section(a, sections, "tools", buf_str(&b));
        b.len = 0;
        build_rules(&b, tools, ntools);
        put_section(a, sections, "rules", buf_str(&b));
        b.len = 0;
        // no <docs> section: pi's points at pi's own install (Saetta Code is its own product)
    }
    if (append.len) put_section(a, sections, "addendum", buf_str(&append));
    if (ctx.len) {
        buf_adds(&b, "Project-specific instructions and guidelines:");
        for (size_t i = 0; i < ctx.len; i++) {
            buf_printf(&b, "\n\n<project_instructions path=\"%s\">\n", ctx.ptr[i].path);
            buf_addstr(&b, ctx.ptr[i].content);
            buf_adds(&b, "\n</project_instructions>");
        }
        put_section(a, sections, "project_context", buf_str(&b));
        b.len = 0;
    }
    // the skills' reader: read, else bash (no hidden declarations without extensions)
    const char *reader = has_tool(tools, ntools, "read") ? "read" : has_tool(tools, ntools, "bash") ? "bash" : NULL;
    if (reader && skills.len) {
        format_skills(&b, &skills, reader);
        if (b.len) put_section(a, sections, "skills", buf_str(&b));
        b.len = 0;
    }
    for (const char *p = env->cwd; *p; p++) buf_addc(&b, *p == '\\' ? '/' : *p);
    put_section(a, sections, "cwd", buf_str(&b));

    JValue *msg = json_new(a, J_OBJECT);
    json_put(a, msg, S("role"), json_string(a, S("system")));
    json_put(a, msg, S("content"), json_string(a, S("")));
    json_put(a, msg, S("sections"), sections);
    *out = msg;
    buf_free(&b);
    buf_free(&append);
    vec_free(&ctx);
    vec_free(&skills);
    return OK;
}

// getSystemMessageText: the content's text, then every non-null section, the non-empty ones
// joined by a blank line.
Str code_system_text(Arena *a, const JValue *system) {
    Buf b = {0};
    Buf content = {0};
    const JValue *c = json_get(system, "content");
    if (json_is(c, J_STRING)) buf_addstr(&content, c->s);
    else if (json_is(c, J_ARRAY)) {
        size_t k = 0;
        for (size_t i = 0; i < c->arr.len; i++) {
            const JValue *blk = c->arr.items[i];
            const char *type = json_str(json_get(blk, "type"));
            if (!type || strcmp(type, "text") != 0) continue;
            if (k++) buf_addc(&content, '\n');
            buf_addstr(&content, json_strs(json_get(blk, "text")));
        }
    }
    size_t parts = 0;
    if (content.len) {
        buf_addstr(&b, buf_str(&content));
        parts++;
    }
    const JValue *sections = json_get(system, "sections");
    for (size_t i = 0; json_is(sections, J_OBJECT) && i < sections->obj.len; i++) {
        const JValue *v = sections->obj.items[i].val;
        if (!json_is(v, J_STRING) || !v->s.len) continue;
        if (parts++) buf_adds(&b, "\n\n");
        buf_addstr(&b, v->s);
    }
    Str out = {arena_strndup(a, b.ptr ? b.ptr : "", b.len), b.len};
    buf_free(&b);
    buf_free(&content);
    return out;
}

// ---- shared with code_skills.c, code_trust.c and code_changelog.c (code_res.h) ----------------

char *code_path_resolve(Arena *a, const char *base, const char *p) { return path_resolve(a, base, p); }
char *code_path_join(Arena *a, const char *x, const char *y) { return pjoin(a, x, y); }
char *code_path_dirname(Arena *a, const char *p) { return pdirname(a, p); }
const char *code_path_basename(const char *p) { return pbasename(p); }
bool code_path_exists(const char *p) { return p_exists(p); }
bool code_path_isfile(const char *p) { return p_isfile(p); }
bool code_path_isdir(const char *p) {
    struct stat st;
    return stat(p, &st) == 0 && S_ISDIR(st.st_mode);
}
char *code_path_real(Arena *a, const char *p) { return p_real(a, p); }
bool code_path_within(const char *parent, const char *child) { return p_within(parent, child); }
bool code_read_text(Arena *a, const char *p, Str *out) { return read_text(a, p, out); }
JValue *code_read_json_object(Arena *a, const char *p) { return read_json(a, p); }
bool code_js_blank(Str s) { return js_blank(s); }

// String.prototype.trim(): JavaScript's whitespace (ASCII, NBSP, the BOM, U+1680, U+2000..U+200A,
// U+2028, U+2029, U+202F, U+205F, U+3000) cut from both ends.
static size_t js_ws_len(Str s, size_t i, bool back) {
    static const char *const ws[] = {"\xC2\xA0", "\xEF\xBB\xBF", "\xE1\x9A\x80", "\xE2\x80\xA8", "\xE2\x80\xA9", "\xE2\x80\xAF",
                                     "\xE2\x81\x9F", "\xE3\x80\x80"};
    unsigned char c = (unsigned char)(back ? s.ptr[i - 1] : s.ptr[i]);
    if (c == ' ' || (c >= '\t' && c <= '\r')) return 1;
    for (size_t k = 0; k < COUNT(ws); k++) {
        size_t n = strlen(ws[k]);
        if (back ? i >= n && !memcmp(s.ptr + i - n, ws[k], n) : i + n <= s.len && !memcmp(s.ptr + i, ws[k], n)) return n;
    }
    // U+2000..U+200A: E2 80 80..8A
    if (back ? i >= 3 && (unsigned char)s.ptr[i - 3] == 0xE2 && (unsigned char)s.ptr[i - 2] == 0x80 && (unsigned char)s.ptr[i - 1] <= 0x8A
             : i + 3 <= s.len && (unsigned char)s.ptr[i] == 0xE2 && (unsigned char)s.ptr[i + 1] == 0x80 && (unsigned char)s.ptr[i + 2] >= 0x80 &&
                   (unsigned char)s.ptr[i + 2] <= 0x8A)
        return 3;
    return 0;
}

Str code_js_trim(Str s) {
    size_t a = 0, b = s.len, n;
    while (a < b && (n = js_ws_len(s, a, false))) a += n;
    while (b > a && (n = js_ws_len(s, b, true))) b -= n;
    return str_sub(s, a, b);
}

void code_collect_skill_entries(Arena *a, const char *dir, bool mode_agents, CodePaths *out) {
    VecPath v = {0};
    collect_skill_entries(a, dir, mode_agents, NULL, NULL, &v);
    for (size_t i = 0; i < v.len; i++) vec_push(out, v.ptr[i]);
    vec_free(&v);
}

bool code_frontmatter(Arena *a, Str content, CodeFrontmatter *out) {
    Frontmatter fm;
    *out = (CodeFrontmatter){0};
    if (!parse_frontmatter(a, content, &fm)) return false;
    out->has_name = fm.name.kind == FM_STRING;
    out->name = fm.name.s;
    out->has_description = fm.description.kind == FM_STRING;
    out->description = fm.description.s;
    out->disable = fm.disable.kind == FM_OTHER && fm.disable.btrue;
    return true;
}

// frontmatter.ts extractFrontmatter's body: after the closing "---" line, trimmed; the whole
// (normalized) text without a frontmatter.
bool code_frontmatter_body(Arena *a, Str content, Str *body) {
    Frontmatter fm;
    if (!parse_frontmatter(a, content, &fm)) return false;
    Buf n = {0};
    size_t from = str_starts(content, S("\xEF\xBB\xBF")) ? 3 : 0;
    for (size_t i = from; i < content.len; i++) {
        char c = content.ptr[i];
        if (c == '\r') {
            buf_addc(&n, '\n');
            if (i + 1 < content.len && content.ptr[i + 1] == '\n') i++;
        } else
            buf_addc(&n, c);
    }
    Str t = {arena_strndup(a, n.ptr ? n.ptr : "", n.len), n.len};
    buf_free(&n);
    ptrdiff_t end = str_starts(t, S("---")) ? str_find(t, S("\n---"), 3) : -1;
    if (end < 0) *body = t;
    else *body = code_js_trim(str_sub(t, (size_t)end + 4, t.len));
    return true;
}

// resource-loader.ts: getSystemPromptSource (an existing file only), getAppendSystemPromptSources
// (the existing ones), then the context files, as the startup listing shows them.
void code_prompt_context_paths(Arena *a, const CodeEnv *env, bool trusted, CodePaths *out) {
    const char *cwd = path_resolve(a, "/", env->cwd);
    const char *agent_dir = path_resolve(a, cwd, env->agent_dir);
    const char *sys_src = env->system_prompt;
    if (!sys_src) {
        char *pp = pjoin(a, cwd, ".pi/SYSTEM.md"), *gp = pjoin(a, agent_dir, "SYSTEM.md");
        sys_src = trusted && p_exists(pp) ? pp : p_exists(gp) ? gp : NULL;
    }
    if (sys_src && p_exists(path_resolve(a, cwd, sys_src))) vec_push(out, path_resolve(a, cwd, sys_src));
    const char *const *srcs = env->append_system_prompt;
    size_t nsrc = env->nappend_system_prompt;
    const char *found = NULL;
    if (!nsrc) {
        char *pp = pjoin(a, cwd, ".pi/APPEND_SYSTEM.md"), *gp = pjoin(a, agent_dir, "APPEND_SYSTEM.md");
        found = trusted && p_exists(pp) ? pp : p_exists(gp) ? gp : NULL;
        srcs = &found;
        nsrc = found ? 1 : 0;
    }
    for (size_t i = 0; i < nsrc; i++)
        if (p_exists(path_resolve(a, cwd, srcs[i]))) vec_push(out, path_resolve(a, cwd, srcs[i]));
    VecCtx ctx = {0};
    load_context_files(a, cwd, agent_dir, &ctx);
    for (size_t i = 0; i < ctx.len; i++) vec_push(out, (char *)ctx.ptr[i].path);
    vec_free(&ctx);
}
