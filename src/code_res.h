// Saetta Code's resources: the skills, context files and prompt files pi's resource loader holds,
// project trust, and the changelog (code_skills.c, code_trust.c, code_changelog.c), with the path
// and frontmatter helpers of code_prompt.c they share.
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at v1.5.2,
// b4afddd): packages/coding-agent/src/core/resource-loader.ts, package-manager.ts, skills.ts,
// source-info.ts, settings-manager.ts, trust-manager.ts, project-trust.ts, utils/changelog.ts.
#ifndef SAETTA_CODE_RES_H
#define SAETTA_CODE_RES_H

#include "code.h"

// ---- helpers from code_prompt.c (node:path and node:fs semantics on POSIX) ----------------------

typedef VEC(char *) CodePaths;
char *code_path_resolve(Arena *a, const char *base, const char *p); // path.resolve(base, p)
char *code_path_join(Arena *a, const char *x, const char *y);       // path.join
char *code_path_dirname(Arena *a, const char *p);
const char *code_path_basename(const char *p);
bool code_path_exists(const char *p);
bool code_path_isfile(const char *p);
bool code_path_isdir(const char *p);
char *code_path_real(Arena *a, const char *p);                // canonicalizePath
bool code_path_within(const char *parent, const char *child); // isPathWithin
// readFileSync(p, "utf-8") with stripBom; false when it cannot be read.
bool code_read_text(Arena *a, const char *p, Str *out);
// A JSON object file, or NULL (missing, unreadable, not an object).
JValue *code_read_json_object(Arena *a, const char *p);
// package-manager.ts collectSkillEntries(dir, "pi" | "agents"), appended to out.
void code_collect_skill_entries(Arena *a, const char *dir, bool mode_agents, CodePaths *out);
// The frontmatter fields skills read (parseFrontmatter); false when the yaml package would throw.
typedef struct {
    bool has_name, has_description; // the value is a string
    Str name, description;
    bool disable; // "disable-model-invocation": true
} CodeFrontmatter;
bool code_frontmatter(Arena *a, Str content, CodeFrontmatter *out);
// stripFrontmatter(content); false when the yaml package would throw.
bool code_frontmatter_body(Arena *a, Str content, Str *body);
// String.prototype.trim() === "" for the whitespace that occurs in practice.
bool code_js_blank(Str s);
Str code_js_trim(Str s); // String.prototype.trim()
// The context files the prompt uses, in pi's order (getSystemPromptSource,
// getAppendSystemPromptSources, getAgentsFiles), for the startup listing.
void code_prompt_context_paths(Arena *a, const CodeEnv *env, bool trusted, CodePaths *out);

// ---- skills (code_skills.c) ------------------------------------------------------------------------

// SourceInfo: where a resource comes from (source "auto", "local", "cli"; scope "user",
// "project", "temporary"; origin "top-level").
typedef struct {
    const char *source, *scope, *origin, *base_dir;
} CodeSourceInfo;

typedef struct {
    Str name, description;
    const char *file, *base_dir;
    bool disable; // disableModelInvocation
    CodeSourceInfo si;
} CodeSkill;

// ResourceDiagnostic: "warning", "error" or "collision" (then name, winner and loser).
typedef struct {
    const char *type, *message, *path;
    Str name;
    const char *winner, *loser;
} CodeDiag;

typedef struct {
    CodeSkill *skills;
    size_t nskills;
    CodeDiag *diags;
    size_t ndiags;
    CodePaths context; // the context files (code_prompt_context_paths)
    // the --skill paths that do not exist (resource-loader.ts: errors unless already a diagnostic)
    bool loaded;
} CodeResources;

// The command line's skill controls: --no-skills and each --skill path, as given.
void code_skills_options(bool no_skills, const char *const *paths, size_t n);
// The resources as loaded at startup or by the last /reload (loaded on first use).
const CodeResources *code_resources(const CodeEnv *env);
// Loads them again (/reload): settings, skills, context files.
void code_resources_reload(const CodeEnv *env);
// AgentSession._expandSkillCommand: "/skill:name args" as the skill block, else text itself.
// error (may be NULL) gets the read or parse error of a known skill.
const char *code_skill_expand(Arena *a, const CodeEnv *env, const char *text, const char **error);

// ---- project trust (code_trust.c) ----------------------------------------------------------------

// hasTrustRequiringProjectResources
bool code_trust_has_project_resources(Arena *a, const char *cwd, const char *home);
// The session's project trust (main.ts createRuntime and resolveProjectTrusted): --approve /
// --no-approve, the trust store, settings defaultProjectTrust, and in the interactive mode pi's
// startup prompt ("Trust project folder?"). Resolved once per session.
void code_trust_override(int trusted); // -1 none, 0 --no-approve, 1 --approve
void code_trust_resolve(const CodeEnv *env, bool interactive);
bool code_trust_trusted(const CodeEnv *env);

#endif
