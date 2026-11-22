// Tests for src/code_prompt.c against pi 1.5.2's own prompts: with SAETTA_CODE_GOLDEN at the
// captures of pi runs (one folder per scenario, the scenario files beside it), every scenario's
// workspace and home are recreated in a temporary folder as the captures were made, and the
// system message's
// sections and the prompt text are compared byte for byte (after the same normalization) with
// the golden session's system message and the golden first request's messages[0].content.
#include "../code.h"
#include "../test.h"

#include <dirent.h>
#include <ftw.h>
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>

static const char *PI_PACKAGE = "releases/1.5.2/node_modules/@earendil-works/pi-coding-agent";

// Prints where two texts first differ.
static void show_diff(const char *name, const char *what, Str got, Str want) {
    size_t i = 0;
    while (i < got.len && i < want.len && got.ptr[i] == want.ptr[i]) i++;
    size_t from = i > 80 ? i - 80 : 0;
    dbg("%s: %s differ at byte %zu\n  got:  ...%.*s\n  want: ...%.*s\n", name, what, i, (int)(MIN(got.len, i + 80) - from),
        got.ptr + from, (int)(MIN(want.len, i + 80) - from), want.ptr + from);
}

static int rm_one(const char *p, const struct stat *st, int flag, struct FTW *f) { return remove(p); }

static char *pdir(Arena *a, const char *p) {
    const char *s = strrchr(p, '/');
    return arena_strndup(a, p, s ? (size_t)(s - p) : 0);
}

static void replace_all(Buf *b, const char *from, const char *to) {
    size_t fl = strlen(from);
    if (!fl) return;
    Buf o = {0};
    for (size_t i = 0; i < b->len;) {
        if (i + fl <= b->len && memcmp(b->ptr + i, from, fl) == 0) {
            buf_adds(&o, to);
            i += fl;
        } else
            buf_addc(&o, b->ptr[i++]);
    }
    buf_free(b);
    *b = o;
}

// One scenario: 0 when it matches, 1 when it does not (printed), -1 when it cannot run.
static int check_scenario(const char *golden, const char *name, const char *install) {
    Arena a = {0};
    int rc = -1;
    char *sc_path = arena_printf(&a, "%s/../scenarios/%s.json", golden, name);
    char *text;
    size_t len;
    if (read_file(sc_path, 1 << 24, &text, &len)) return -1;
    JValue *sc;
    Err e = json_parse(&a, str_n(text, len), false, &sc);
    free(text);
    if (e) return -1;

    char tmpl[PATH_MAX];
    snprintf(tmpl, sizeof tmpl, "%s/sc-prompt-XXXXXX", env_or("TMPDIR", "/tmp"));
    if (!mkdtemp(tmpl)) return -1;
    char tmp[PATH_MAX];
    if (!realpath(tmpl, tmp)) return -1;
    char *home = arena_printf(&a, "%s/home", tmp), *ws = arena_printf(&a, "%s/ws", tmp);
    char *agent = arena_printf(&a, "%s/.pi/agent", home);
    mkdir_p(agent);
    mkdir_p(ws);
    JValue *files = json_get(sc, "files");
    for (size_t i = 0; files && i < files->obj.len; i++) {
        Str k = files->obj.items[i].key;
        char *dst = str_starts(k, S("~/")) ? arena_printf(&a, "%s/%s", home, k.ptr + 2) : arena_printf(&a, "%s/%s", ws, k.ptr);
        mkdir_p(pdir(&a, dst));
        Str v = json_strs(files->obj.items[i].val);
        write_file(dst, v.ptr, v.len);
    }
    JValue *settings = json_get(sc, "settings");
    if (settings) {
        Buf b = {0};
        json_write(&b, settings);
        write_file(arena_printf(&a, "%s/settings.json", agent), b.ptr, b.len);
        buf_free(&b);
    }

    // run 0's arguments
    const char *tools[16] = {"read", "bash", "edit", "write"};
    size_t ntools = 4;
    const char *append[8];
    CodeEnv env = {.cwd = ws, .home = home, .agent_dir = agent,
                   .pi_docs = arena_printf(&a, "%s/%s", install, PI_PACKAGE), .append_system_prompt = append};
    JValue *run = json_get(sc, "runs")->arr.items[0];
    JValue *args = json_get(run, "args");
    for (size_t i = 0; args && i < args->arr.len; i++) {
        const char *s = json_str(args->arr.items[i]);
        const char *next = i + 1 < args->arr.len ? json_str(args->arr.items[i + 1]) : NULL;
        if (strcmp(s, "--no-tools") == 0) ntools = 0;
        else if (strcmp(s, "--tools") == 0 && next) {
            ntools = 0;
            for (char *t = arena_strdup(&a, next), *tok; (tok = strsep(&t, ",")) && ntools < COUNT(tools);)
                if (*tok) tools[ntools++] = tok;
            i++;
        } else if (strcmp(s, "--system-prompt") == 0 && next) {
            env.system_prompt = next;
            i++;
        } else if (strcmp(s, "--append-system-prompt") == 0 && next && env.nappend_system_prompt < COUNT(append)) {
            append[env.nappend_system_prompt++] = next;
            i++;
        }
    }

    // the golden: the session's system message and the first request's system content
    char *sess_path = arena_printf(&a, "%s/%s/sessions/0.jsonl", golden, name);
    char *req_path = arena_printf(&a, "%s/%s/requests.jsonl", golden, name);
    char *sess, *req;
    size_t sl, rl;
    if (read_file(sess_path, 1 << 26, &sess, &sl)) goto out;
    if (read_file(req_path, 1 << 26, &req, &rl)) {
        free(sess);
        goto out;
    }
    JValue *gold_sections = NULL;
    for (char *line = sess, *nl; line < sess + sl; line = nl + 1) {
        nl = memchr(line, '\n', (size_t)(sess + sl - line));
        if (!nl) nl = sess + sl;
        JValue *ent;
        if (json_parse(&a, str_n(line, (size_t)(nl - line)), false, &ent)) continue;
        JValue *m = json_get(ent, "message");
        const char *role = json_str(json_get(m, "role"));
        if (role && strcmp(role, "system") == 0) {
            gold_sections = json_get(m, "sections");
            break;
        }
    }
    char *rnl = memchr(req, '\n', rl);
    JValue *r0;
    Str gold_text = {0};
    if (!json_parse(&a, str_n(req, rnl ? (size_t)(rnl - req) : rl), false, &r0))
        gold_text = json_strs(json_get(json_get(r0, "messages")->arr.items[0], "content"));
    free(sess);
    free(req);
    if (!gold_sections || !gold_text.ptr) goto out;

    JValue *msg;
    if (code_system_message(&a, &env, tools, ntools, &msg)) goto out;
    Buf got = {0}, want = {0}, gtext = {0};
    json_write(&got, json_get(msg, "sections"));
    json_write(&want, gold_sections);
    buf_addstr(&gtext, code_system_text(&a, msg));
    Buf *norm[] = {&got, &gtext};
    for (size_t i = 0; i < COUNT(norm); i++) {
        replace_all(norm[i], ws, "$WS");
        replace_all(norm[i], home, "$HOME");
        replace_all(norm[i], install, "$PI_INSTALL");
    }
    // the shape pi stores before the agent loop adds timestamp and toolsAdded
    bool shape = msg->obj.len == 3 && str_eqz(msg->obj.items[0].key, "role") && str_eqz(msg->obj.items[1].key, "content") &&
                 str_eqz(json_strs(json_get(msg, "content")), "");
    rc = 0;
    if (!shape || !str_eq(buf_str(&got), buf_str(&want))) {
        show_diff(name, "sections", buf_str(&got), buf_str(&want));
        rc = 1;
    }
    if (!str_eq(buf_str(&gtext), gold_text)) {
        show_diff(name, "prompt texts", buf_str(&gtext), gold_text);
        rc = 1;
    }
    buf_free(&got);
    buf_free(&want);
    buf_free(&gtext);
out:
    nftw(tmp, rm_one, 16, FTW_DEPTH | FTW_PHYS);
    arena_reset(&a);
    return rc;
}

TEST(code_prompt_golden) {
    const char *golden = getenv("SAETTA_CODE_GOLDEN");
    if (!golden) SKIP("no $SAETTA_CODE_GOLDEN");
    char install[PATH_MAX];
    char pi[PATH_MAX];
    snprintf(pi, sizeof pi, "%s/.pi/agent/install", env_or("HOME", ""));
    if (!realpath(pi, install)) SKIP("no pi install in ~/.pi/agent/install");
    DIR *d = opendir(golden);
    EXPECT(d != NULL);
    int checked = 0, failed = 0;
    for (struct dirent *e; (e = readdir(d));) {
        if (e->d_name[0] == '.' || e->d_type != DT_DIR) continue;
        int r = check_scenario(golden, e->d_name, install);
        if (r < 0) continue;
        checked++;
        failed += r;
    }
    closedir(d);
    dbg("code_prompt: %d scenarios, %d differ\n", checked, failed);
    EXPECT(checked > 0);
    EXPECT_EQ(failed, 0);
}

// getSystemMessageText on hand-made messages: content first, null and empty parts dropped
TEST(code_prompt_text) {
    Arena a = {0};
    JValue *m;
    EXPECT_OK(json_parse(&a, S("{\"role\":\"system\",\"content\":[{\"type\":\"text\",\"text\":\"a\"},{\"type\":\"image\"},"
                               "{\"type\":\"text\",\"text\":\"b\"}],\"sections\":{\"x\":\"\",\"y\":null,\"z\":\"<z>\\nq\\n</z>\"}}"),
                         false, &m));
    EXPECT_STR(code_system_text(&a, m), S("a\nb\n\n<z>\nq\n</z>"));
    EXPECT_OK(json_parse(&a, S("{\"role\":\"system\",\"content\":\"forced\"}"), false, &m));
    EXPECT_STR(code_system_text(&a, m), S("forced"));
    arena_reset(&a);
}

static void put(Arena *a, const char *root, const char *rel, const char *text) {
    char *p = arena_printf(a, "%s/%s", root, rel);
    mkdir_p(pdir(a, p));
    write_file(p, text, strlen(text));
}

// Lookup above cwd (root first, after the agent dir's), a nested git worktree shadowing the
// main repo's file, and no <docs> without pi_docs: pi's loadProjectContextFiles and
// buildSystemPromptSections, checked by hand (the captures keep cwd at the workspace root).
TEST(code_prompt_lookup) {
    Arena a = {0};
    char tmpl[PATH_MAX], tmp[PATH_MAX];
    snprintf(tmpl, sizeof tmpl, "%s/sc-prompt-XXXXXX", env_or("TMPDIR", "/tmp"));
    EXPECT(mkdtemp(tmpl) && realpath(tmpl, tmp));
    put(&a, tmp, "home/.pi/agent/CLAUDE.md", "G");
    put(&a, tmp, "AGENTS.md", "T");
    put(&a, tmp, "main/AGENTS.md", "M");
    put(&a, tmp, "main/.git/HEAD", "ref: refs/heads/main\n");
    put(&a, tmp, "main/.git/worktrees/wt/HEAD", "ref: refs/heads/wt\n");
    put(&a, tmp, "main/.git/worktrees/wt/commondir", "../..\n");
    put(&a, tmp, "main/wt/.git", arena_printf(&a, "gitdir: %s/main/.git/worktrees/wt\n", tmp));
    put(&a, tmp, "main/wt/AGENTS.md", "W"); // same name as the main repo's, as a checkout has
    mkdir_p(arena_printf(&a, "%s/main/wt/src", tmp));
    CodeEnv env = {.cwd = arena_printf(&a, "%s/main/wt/src", tmp), .home = arena_printf(&a, "%s/home", tmp),
                   .agent_dir = arena_printf(&a, "%s/home/.pi/agent", tmp)};
    const char *tools[] = {"read"};
    JValue *m;
    EXPECT_OK(code_system_message(&a, &env, tools, 1, &m));
    JValue *s = json_get(m, "sections");
    EXPECT(json_get(s, "docs") == NULL);
    EXPECT_STR(json_strs(json_get(s, "project_context")),
               str(arena_printf(&a,
                                "<project_context>\nProject-specific instructions and guidelines:\n\n"
                                "<project_instructions path=\"%s/home/.pi/agent/CLAUDE.md\">\nG\n</project_instructions>\n\n"
                                "<project_instructions path=\"%s/AGENTS.md\">\nT\n</project_instructions>\n\n"
                                "<project_instructions path=\"%s/main/wt/AGENTS.md\">\nW\n</project_instructions>\n"
                                "</project_context>",
                                tmp, tmp, tmp)));
    // the worktree beside the main repo, not inside it: both files load
    put(&a, tmp, "side/.git", arena_printf(&a, "gitdir: %s/main/.git/worktrees/wt\n", tmp));
    put(&a, tmp, "side/AGENTS.md", "S");
    env.cwd = arena_printf(&a, "%s/side", tmp);
    EXPECT_OK(code_system_message(&a, &env, tools, 1, &m));
    Str pc = json_strs(json_get(json_get(m, "sections"), "project_context"));
    EXPECT(str_find(pc, S(">\nS\n<"), 0) > 0 && str_find(pc, S(">\nT\n<"), 0) > 0);
    nftw(tmp, rm_one, 16, FTW_DEPTH | FTW_PHYS);
    arena_reset(&a);
}
