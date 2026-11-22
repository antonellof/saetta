// Saetta Code's session commands in the interactive mode: /compact, /resume, /fork, /clone, /name
// and /session, with pi's screens (the compaction summary, the session selector, the user message
// selector, the session info), and the footer's session totals. The session files are handled in
// code_sessions.c, the agent's side in code_main.c.
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi v1.5.2,
// b4afddd): packages/coding-agent/src/modes/interactive/interactive-mode.ts (handleNameCommand,
// handleSessionCommand, showUserMessageSelector, handleCloneCommand, showSessionSelector,
// handleResumeSession, handleCompactCommand, the compaction_start / compaction_end events,
// renderInitialMessages, addMessageToChat for compactionSummary), components/
// (compaction-summary-message.ts, user-message-selector.ts, session-selector.ts,
// session-selector-search.ts, footer.ts getSessionStats, keybinding-hints.ts), core/cache-warmer.ts
// (the status /session shows), packages/tui/src/fuzzy.ts and components/input.ts.
#include "code_sessions.h"
#include "code_ui_ext.h"
#include "code_res_ui.h"

#include <errno.h>
#include <math.h>
#include <regex.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

// ---- small helpers ------------------------------------------------------------------------------

static double wall_ms(void) { return (double)(wall_ns() / 1000000); }

static double jnum(const JValue *v) {
    if (json_is(v, J_INTEGER)) return (double)v->i;
    if (json_is(v, J_FLOAT)) return v->f;
    if (json_is(v, J_NUMBER_STRING)) return strtod(v->s.ptr, NULL);
    return 0;
}

static const JValue *entries_now(const char **leaf) {
    const CodeUiOps *ops = uix_ops();
    *leaf = NULL;
    return ops->entries ? ops->entries(uix_host(), leaf) : NULL;
}

// Text(text, 1, 0) rendered into out.
static void text_lines(Str text, int px, int width, Lines *out) {
    Comp *t = comp_text(text, px, 0);
    comp_render(t, width, out);
    comp_free(t);
}

// new ThemedText(() => theme.fg(color, text), 1, 0) after a Spacer(1)
static void chat_themed(ThemeColor c, const char *text) {
    Buf b = {0};
    th_fgz(&b, c, text);
    comp_add(uix_chat(), comp_spacer(1));
    comp_add(uix_chat(), comp_text(buf_str(&b), 1, 0));
    buf_free(&b);
}

// DynamicBorder: a rule across the width in a color.
static void border(Lines *out, int width, ThemeColor c) {
    Buf b = {0}, r = {0};
    tui_repeat(&r, "\xe2\x94\x80", width > 0 ? width : 1);
    th_fg(&b, c, buf_str(&r));
    lines_addb(out, &b);
    buf_free(&r);
}

// keyHint: the key dim, the description muted.
static void key_hint(Buf *b, const char *key, const char *desc) {
    th_fgz(b, T_DIM, key);
    Buf d = {0};
    buf_printf(&d, " %s", desc);
    th_fg(b, T_MUTED, buf_str(&d));
    buf_free(&d);
}

// Number.prototype.toLocaleString() in en-US for whole numbers.
static void locale_int(Buf *b, double v) {
    char tmp[64];
    long long n = llround(v);
    bool neg = n < 0;
    unsigned long long u = neg ? (unsigned long long)(-n) : (unsigned long long)n;
    snprintf(tmp, sizeof tmp, "%llu", u);
    size_t len = strlen(tmp);
    if (neg) buf_addc(b, '-');
    for (size_t i = 0; i < len; i++) {
        if (i && (len - i) % 3 == 0) buf_addc(b, ',');
        buf_addc(b, tmp[i]);
    }
}

// formatTokens (footer.ts / format.ts)
static void fmt_tokens(Buf *b, double n) {
    if (n < 1000) buf_printf(b, "%.0f", n);
    else if (n < 10000) buf_printf(b, "%.1fk", n / 1000);
    else if (n < 1000000) buf_printf(b, "%.0fk", round(n / 1000));
    else if (n < 10000000) buf_printf(b, "%.1fM", n / 1000000);
    else buf_printf(b, "%.0fM", round(n / 1000000));
}

// ---- the session-wide state the footer and /session read (kept current at idle times) -----------

static struct {
    char *name;              // getSessionName
    CodeSessionStats base;   // the totals over every entry when the transcript was drawn
    double at[5];            // the drawn messages' sums then (footer_render's)
    bool capture;            // take at[] at the next footer
    bool unknown;            // the context use is unknown (a compaction without an answer after it)
    bool requested;          // a request went out in this session (the cache warmer's first request)
    VEC(char *) deferred;    // commands waiting for the run to stop
    bool named;              // name read once (a session continued at start)
    bool tail_compaction;    // draw the latest compaction after the kept messages (compaction_end)
    char *tail_summary;
    double tail_tokens;
    bool compacting;         // between compaction_start and compaction_end
} G;

bool sess_ui_compacting(void) { return G.compacting; }

static void refresh_name(void) {
    const char *leaf;
    const JValue *entries = entries_now(&leaf);
    Arena a = {0};
    const char *n = entries ? code_sessions_name(&a, entries) : NULL;
    free(G.name);
    G.name = n ? strdup(n) : NULL;
    G.named = true;
    arena_reset(&a);
}

void sess_ui_reset(void) {
    memset(&G.base, 0, sizeof G.base);
    memset(G.at, 0, sizeof G.at);
    G.capture = false;
    G.unknown = false;
    G.requested = false;
    refresh_name();
}

void sess_ui_footer_name(Buf *out) {
    if (!G.named && !uix_busy()) refresh_name();
    if (!G.name) return;
    buf_adds(out, " \xe2\x80\xa2 ");
    buf_adds(out, G.name);
}

void sess_ui_footer(double *in, double *out, double *cache_read, double *cache_write, double *cost, bool *unknown) {
    double cur[5] = {*in, *out, *cache_read, *cache_write, *cost};
    if (G.capture) {
        memcpy(G.at, cur, sizeof cur);
        G.capture = false;
    }
    *in = G.base.input + cur[0] - G.at[0];
    *out = G.base.output + cur[1] - G.at[1];
    *cache_read = G.base.cache_read + cur[2] - G.at[2];
    *cache_write = G.base.cache_write + cur[3] - G.at[3];
    *cost = G.base.cost + cur[4] - G.at[4];
    *unknown = G.unknown;
}

// ---- the compaction summary (CompactionSummaryMessageComponent) ---------------------------------------

typedef struct {
    Comp base;
    char *summary;
    double tokens;
    bool expanded;
} CompactionView;

static void compaction_render(Comp *c, int width, Lines *out) {
    CompactionView *v = (CompactionView *)c;
    int cw = width - 2 > 1 ? width - 2 : 1;
    Lines kids = {0};
    Buf b = {0}, tok = {0};
    locale_int(&tok, v->tokens);
    th_fgz(&b, T_CUSTOM_LABEL, "\x1b[1m[compaction]\x1b[22m");
    text_lines(buf_str(&b), 0, cw, &kids);
    lines_addz(&kids, "");
    b.len = 0;
    if (v->expanded) {
        buf_printf(&b, "**Compacted from %s tokens**\n\n%s", tok.ptr, v->summary);
        MdStyle st = {.has_color = true, .color = T_CUSTOM_TEXT};
        md_render(buf_str(&b), 0, 0, &st, cw, &kids);
    } else {
        Buf t = {0};
        buf_printf(&t, "Compacted from %s tokens (", tok.ptr);
        th_fg(&b, T_CUSTOM_TEXT, buf_str(&t));
        th_fgz(&b, T_DIM, "ctrl+o");
        th_fgz(&b, T_CUSTOM_TEXT, " to expand)");
        buf_free(&t);
        text_lines(buf_str(&b), 0, cw, &kids);
    }
    buf_free(&b);
    buf_free(&tok);
    // Box(1, 1, customMessageBg)
    for (int k = 0; k < 1; k++) {
        Buf l = {0};
        th_bg_line(&l, TH_BG_CUSTOM_MESSAGE, S(""), width);
        lines_addb(out, &l);
    }
    for (size_t i = 0; i < kids.n; i++) {
        Buf p = {0}, l = {0};
        buf_addc(&p, ' ');
        buf_adds(&p, kids.v[i]);
        th_bg_line(&l, TH_BG_CUSTOM_MESSAGE, buf_str(&p), width);
        lines_addb(out, &l);
        buf_free(&p);
    }
    Buf l = {0};
    th_bg_line(&l, TH_BG_CUSTOM_MESSAGE, S(""), width);
    lines_addb(out, &l);
    lines_free(&kids);
}

static void compaction_destroy(Comp *c) {
    free(((CompactionView *)c)->summary);
    free(c);
}

static const CompVt compaction_vt = {compaction_render, compaction_destroy, false};

// addMessageToChat for a compactionSummary message: a Spacer, then the summary.
static void add_compaction(const char *summary, double tokens) {
    CompactionView *v = xcalloc(1, sizeof *v);
    v->base.vt = &compaction_vt;
    v->summary = strdup(summary ? summary : "");
    v->tokens = tokens;
    v->expanded = uix_tools_expanded();
    comp_add(uix_chat(), comp_spacer(1));
    comp_add(uix_chat(), &v->base);
}

void sess_ui_add_compaction(const JValue *m) { add_compaction(json_str(json_get(m, "summary")), jnum(json_get(m, "tokensBefore"))); }

void sess_ui_set_expanded(bool expanded) {
    CompVec *kids = comp_children(uix_chat());
    for (size_t i = 0; i < kids->len; i++)
        if (kids->ptr[i]->vt == &compaction_vt) {
            ((CompactionView *)kids->ptr[i])->expanded = expanded;
            comp_invalidate(kids->ptr[i]);
        }
}

// ---- the transcript of the session (renderInitialMessages, renderSessionEntries) -----------------

const JValue *sess_ui_history(Arena *a) {
    const char *leaf;
    const JValue *entries = entries_now(&leaf);
    JValue *out = json_new(a, J_ARRAY);
    if (!entries) return out;
    JValue *ctx = code_sessions_context_entries(a, entries, leaf);
    size_t from = 0;
    if (G.tail_compaction && ctx->arr.len && !strcmp(json_str(json_get(ctx->arr.items[0], "type")) ?: "", "compaction")) {
        // compaction_end: the latest compaction is drawn below the messages it kept
        free(G.tail_summary);
        G.tail_summary = strdup(json_str(json_get(ctx->arr.items[0], "summary")) ?: "");
        G.tail_tokens = jnum(json_get(ctx->arr.items[0], "tokensBefore"));
        from = 1;
    }
    for (size_t i = from; i < ctx->arr.len; i++) code_sessions_entry_messages(a, ctx->arr.items[i], out);
    return out;
}

void sess_ui_history_done(void) {
    const char *leaf;
    const JValue *entries = entries_now(&leaf);
    if (G.tail_compaction) {
        G.tail_compaction = false;
        if (G.tail_summary) add_compaction(G.tail_summary, G.tail_tokens);
    } else if (entries) {
        // renderInitialMessages: how often the session was compacted
        size_t n = 0;
        for (size_t i = 0; i < entries->arr.len; i++) n += !strcmp(json_str(json_get(entries->arr.items[i], "type")) ?: "", "compaction");
        if (n) {
            char m[64];
            if (n == 1) snprintf(m, sizeof m, "Session compacted 1 time");
            else snprintf(m, sizeof m, "Session compacted %zu times", n);
            uix_show_status(m);
        }
    }
    if (entries) {
        code_sessions_stats(entries, &G.base);
        G.capture = true;
        Arena a = {0};
        G.unknown = code_sessions_context_unknown(&a, entries, leaf);
        arena_reset(&a);
    }
    refresh_name();
}

// ---- events -------------------------------------------------------------------------------------------

static bool valid_usage(const JValue *m) {
    const JValue *u = json_get(m, "usage");
    const char *stop = json_str(json_get(m, "stopReason"));
    if (stop && (!strcmp(stop, "aborted") || !strcmp(stop, "error"))) return false;
    double t = jnum(json_get(u, "totalTokens"));
    if (!t) t = jnum(json_get(u, "input")) + jnum(json_get(u, "output")) + jnum(json_get(u, "cacheRead")) + jnum(json_get(u, "cacheWrite"));
    return t > 0;
}

void sess_ui_event(const JValue *ev) {
    const char *type = json_str(json_get(ev, "type"));
    if (!type) return;
    if (!strcmp(type, "message_end")) {
        const JValue *m = json_get(ev, "message");
        if (!strcmp(json_str(json_get(m, "role")) ?: "", "assistant")) {
            G.requested = true;
            if (valid_usage(m)) G.unknown = false;
        }
    } else if (!strcmp(type, "compaction_start")) {
        G.compacting = true;
        // CompactionStatusIndicator in the editor's border (manual, threshold or overflow)
        const char *reason = json_str(json_get(ev, "reason"));
        if (!reason || !strcmp(reason, "manual")) uix_working("Compacting context... (escape to cancel)");
        else uix_working(!strcmp(reason, "overflow") ? "Context overflow detected, Auto-compacting... (escape to cancel)" : "Auto-compacting... (escape to cancel)");
    } else if (!strcmp(type, "compaction_end")) {
        G.compacting = false;
        uix_working(NULL);
        const char *reason = json_str(json_get(ev, "reason"));
        bool manual = !reason || !strcmp(reason, "manual");
        bool aborted = json_is(json_get(ev, "aborted"), J_BOOL) && json_get(ev, "aborted")->b;
        if (aborted) {
            if (manual) uix_show_error("Compaction cancelled");
            else uix_show_status("Auto-compaction cancelled");
        } else if (json_get(ev, "result")) {
            // the transcript again: the kept messages, then the summary at its place in time
            G.tail_compaction = true;
            uix_render_history();
        } else if (json_str(json_get(ev, "errorMessage"))) {
            if (manual) uix_show_error(json_str(json_get(ev, "errorMessage")));
            else chat_themed(T_ERROR, json_str(json_get(ev, "errorMessage"))); // Spacer + ThemedText("error")
        }
        uix_request_render();
    }
}

// ---- /name (handleNameCommand) --------------------------------------------------------------------------

static void name_command(const char *text) {
    // text.replace(/^\/name\s*/, "").trim()
    const char *p = text + 4;
    Str name = str_trim(str(p));
    CodeSession *s = uix_ops()->session ? uix_ops()->session(uix_host()) : NULL;
    if (!s) return;
    Arena a = {0};
    if (!name.len) {
        const char *cur = code_sessions_name(&a, code_session_entries(s));
        if (cur) {
            Buf m = {0};
            buf_printf(&m, "Session name: %s", cur);
            chat_themed(T_DIM, m.ptr);
            buf_free(&m);
        } else uix_show_warning("Usage: /name <name>");
        arena_reset(&a);
        return;
    }
    char *want = str_z(&a, name);
    (void)code_sessions_set_name(s, want);
    const char *got = code_sessions_name(&a, code_session_entries(s));
    if (!got || strcmp(got, want)) {
        Buf m = {0};
        buf_adds(&m, "Session name was normalized from ");
        json_write_string(&m, str(want));
        buf_adds(&m, " to ");
        if (got) json_write_string(&m, str(got));
        else buf_adds(&m, "undefined");
        uix_show_warning(m.ptr);
        buf_free(&m);
    }
    Buf m = {0};
    buf_printf(&m, "Session name set: %s", got ? got : want);
    chat_themed(T_DIM, m.ptr);
    buf_free(&m);
    arena_reset(&a);
    refresh_name();
}

// ---- /session (handleSessionCommand) -----------------------------------------------------------------

static void dim_label(Buf *b, const char *label) { th_fgz(b, T_DIM, label); }

static void session_command(void) {
    CodeSession *s = uix_ops()->session ? uix_ops()->session(uix_host()) : NULL;
    if (!s) return;
    const JValue *entries = code_session_entries(s);
    CodeSessionStats st;
    code_sessions_stats(entries, &st);
    Arena a = {0};
    const char *name = code_sessions_name(&a, entries);
    const char *file = code_session_path(s);
    const char *id = json_str(json_get(code_session_header(s), "id"));
    Buf b = {0};
    th_bold(&b, S("Session Info"));
    buf_adds(&b, "\n\n");
    if (name) {
        dim_label(&b, "Name:");
        buf_printf(&b, " %s\n", name);
    }
    dim_label(&b, "File:");
    buf_printf(&b, " %s\n", file ? file : "In-memory");
    dim_label(&b, "ID:");
    buf_printf(&b, " %s\n\n", id ? id : "");
    th_bold(&b, S("Messages"));
    buf_adds(&b, "\n");
    dim_label(&b, "Total:");
    buf_printf(&b, " %zu\n", st.total);
    dim_label(&b, "User:");
    buf_printf(&b, " %zu\n", st.user);
    dim_label(&b, "Assistant:");
    buf_printf(&b, " %zu\n", st.assistant);
    dim_label(&b, "Tools:");
    buf_printf(&b, " %zu calls, %zu results\n\n", st.tool_calls, st.tool_results);
    th_bold(&b, S("Tokens"));
    buf_adds(&b, "\n");
    double prompt = st.input + st.cache_read + st.cache_write;
    dim_label(&b, "Input:");
    buf_addc(&b, ' ');
    locale_int(&b, prompt);
    buf_adds(&b, "\n");
    if (prompt > 0 && (st.cache_read > 0 || st.cache_write > 0)) {
        Buf r = {0};
        buf_printf(&r, "(%.1f%%)", st.cache_read / prompt * 100);
        buf_adds(&b, "  ");
        dim_label(&b, "Cached:");
        buf_addc(&b, ' ');
        locale_int(&b, st.cache_read);
        buf_addc(&b, ' ');
        th_fg(&b, T_DIM, buf_str(&r));
        buf_adds(&b, "\n  ");
        dim_label(&b, "Uncached:");
        buf_addc(&b, ' ');
        locale_int(&b, st.input + st.cache_write);
        if (st.cache_write > 0) {
            Buf w = {0};
            buf_adds(&w, "(");
            locale_int(&w, st.cache_write);
            buf_adds(&w, " written to cache)");
            buf_addc(&b, ' ');
            th_fg(&b, T_DIM, buf_str(&w));
            buf_free(&w);
        }
        buf_adds(&b, "\n");
        buf_free(&r);
    }
    dim_label(&b, "Output:");
    buf_addc(&b, ' ');
    locale_int(&b, st.output);
    buf_adds(&b, "\n");
    dim_label(&b, "Total:");
    buf_addc(&b, ' ');
    locale_int(&b, st.input + st.output + st.cache_read + st.cache_write);
    buf_adds(&b, "\n");
    // Cache Warming: Saetta Code's providers report no cache lifetime, so the warmer never starts
    const JValue *settings = uix_cfg()->settings;
    const char *mode = json_str(json_get(settings, "cacheWarming"));
    if (!mode || (strcmp(mode, "off") && strcmp(mode, "streaming") && strcmp(mode, "idle"))) mode = "idle";
    const char *reason = !strcmp(mode, "off") ? "cache warming disabled" : G.requested ? "cache lifetime unavailable" : "waiting for first request";
    buf_adds(&b, "\n");
    th_bold(&b, S("Cache Warming"));
    buf_adds(&b, "\n");
    dim_label(&b, "Mode:");
    buf_printf(&b, " %s\n", mode);
    dim_label(&b, "Status:");
    buf_printf(&b, " Inactive (%s)\n", reason);
    if (st.cost > 0) {
        buf_adds(&b, "\n");
        th_bold(&b, S("Cost"));
        buf_adds(&b, "\n");
        dim_label(&b, "Total:");
        buf_printf(&b, " $%.3f", st.cost);
        // getUsageCostBreakdown: a single entry repeats the total unless it names another model
        typedef struct {
            char *key;
            double cost, tokens;
        } Row;
        VEC(Row) rows = {0};
        for (size_t i = 0; i < entries->arr.len; i++) {
            const JValue *e = entries->arr.items[i];
            const char *t = json_str(json_get(e, "type")) ?: "";
            const JValue *m = json_get(e, "message"), *u = NULL;
            char *key = NULL;
            if (!strcmp(t, "message") && !strcmp(json_str(json_get(m, "role")) ?: "", "assistant")) {
                const char *rm = json_str(json_get(m, "responseModel"));
                key = arena_printf(&a, "%s/%s", json_str(json_get(m, "provider")) ?: "", rm ? rm : json_str(json_get(m, "model")) ?: "");
                u = json_get(m, "usage");
            } else if (!strcmp(t, "usage")) {
                key = arena_printf(&a, "%s/%s", json_str(json_get(e, "provider")) ?: "", json_str(json_get(e, "model")) ?: "");
                u = json_get(e, "usage");
            } else if (!strcmp(t, "message") && !strcmp(json_str(json_get(m, "role")) ?: "", "toolResult") && json_get(m, "usage")) {
                key = "Tools/summaries";
                u = json_get(m, "usage");
            } else if ((!strcmp(t, "branch_summary") || !strcmp(t, "compaction")) && json_get(e, "usage")) {
                key = "Tools/summaries";
                u = json_get(e, "usage");
            }
            if (!key || !u) continue;
            size_t k = 0;
            while (k < rows.len && strcmp(rows.ptr[k].key, key)) k++;
            if (k == rows.len) vec_push(&rows, ((Row){key, 0, 0}));
            rows.ptr[k].cost += jnum(json_get(json_get(u, "cost"), "total"));
            rows.ptr[k].tokens += jnum(json_get(u, "input")) + jnum(json_get(u, "output")) + jnum(json_get(u, "cacheRead")) + jnum(json_get(u, "cacheWrite"));
        }
        size_t kept = 0;
        for (size_t k = 0; k < rows.len; k++)
            if (rows.ptr[k].cost > 0 || rows.ptr[k].tokens > 0) rows.ptr[kept++] = rows.ptr[k];
        rows.len = kept;
        const CodeUiModel *cm = &uix_cfg()->model;
        char *selected = arena_printf(&a, "%s/%s", cm->provider ? cm->provider : "", cm->id ? cm->id : "");
        if (rows.len > 1 || (rows.len && strcmp(rows.ptr[0].key, selected)))
            for (size_t k = 0; k < rows.len; k++) {
                Buf kk = {0}, tk = {0};
                buf_printf(&kk, "%s:", rows.ptr[k].key);
                buf_adds(&b, "\n  ");
                th_fg(&b, T_DIM, buf_str(&kk));
                buf_printf(&b, " $%.3f ", rows.ptr[k].cost);
                buf_adds(&tk, "(");
                fmt_tokens(&tk, rows.ptr[k].tokens);
                buf_adds(&tk, " tokens)");
                th_fg(&b, T_DIM, buf_str(&tk));
                buf_free(&kk);
                buf_free(&tk);
            }
        vec_free(&rows);
    }
    comp_add(uix_chat(), comp_spacer(1));
    comp_add(uix_chat(), comp_text(buf_str(&b), 1, 0));
    buf_free(&b);
    arena_reset(&a);
}

// ---- a one-line text input (pi-tui Input) -----------------------------------------------------------

typedef struct {
    Buf v;
    size_t cur;
} Inp;

static Str inp_str(const Inp *in) { return buf_str(&in->v); }

static void inp_set(Inp *in, const char *s) {
    in->v.len = 0;
    buf_adds(&in->v, s ? s : "");
    if (!in->v.ptr) buf_adds(&in->v, "");
    if (in->cur > in->v.len) in->cur = in->v.len; // setValue keeps the cursor where it can
}

static void inp_free(Inp *in) {
    buf_free(&in->v);
    in->cur = 0;
}

static void inp_render(const Inp *in, int width, Lines *out) {
    Str v = inp_str(in);
    int avail = width - 2;
    Buf b = {0};
    buf_adds(&b, "> ");
    if (avail <= 0) {
        lines_addb(out, &b);
        return;
    }
    size_t start = 0, end = v.len, cur = in->cur;
    int total = tui_width(v);
    if (total >= avail) {
        // horizontal scrolling: the cursor kept in view
        int scroll = in->cur == v.len ? avail - 1 : avail;
        int ccol = tui_width(str_n(v.ptr, in->cur));
        int half = scroll / 2, startcol = 0;
        if (ccol < half) startcol = 0;
        else if (ccol > total - half) startcol = total - scroll > 0 ? total - scroll : 0;
        else startcol = ccol - half > 0 ? ccol - half : 0;
        // the bytes of the columns startcol .. startcol + scroll
        int col = 0;
        size_t i = 0;
        while (i < v.len && col < startcol) {
            size_t g = tui_grapheme(v, i);
            col += tui_width(str_n(v.ptr + i, g));
            i += g;
        }
        start = i;
        while (i < v.len) {
            size_t g = tui_grapheme(v, i);
            int w = tui_width(str_n(v.ptr + i, g));
            if (col + w > startcol + scroll) break;
            col += w;
            i += g;
        }
        end = i;
        cur = in->cur < start ? start : in->cur > end ? end : in->cur;
    }
    buf_add(&b, v.ptr + start, cur - start);
    if (cur < end) {
        size_t g = tui_grapheme(v, cur);
        buf_adds(&b, "\x1b[7m");
        buf_add(&b, v.ptr + cur, g);
        buf_adds(&b, "\x1b[27m");
        buf_add(&b, v.ptr + cur + g, end - cur - g);
    } else buf_adds(&b, "\x1b[7m \x1b[27m");
    int w = tui_width(buf_str(&b)) - 2;
    tui_spaces(&b, avail - w);
    lines_addb(out, &b);
}

static bool word_char(char c) { return c != ' ' && c != '\t'; }

// handleInput for the keys that edit; true when the value changed or the cursor moved.
static bool inp_key(Inp *in, const Key *k) {
    Buf *v = &in->v;
    if (!v->ptr) buf_adds(v, "");
    if ((k->kind == K_CHAR && k->ch >= 0x20 && k->ch != 0x7f && !(k->ch >= 0x80 && k->ch <= 0x9f)) || k->kind == K_PASTE) {
        char enc[4];
        const char *ins = enc;
        size_t n;
        Buf clean = {0};
        if (k->kind == K_PASTE) {
            // handlePaste: line breaks dropped, tabs as spaces
            for (const char *p = k->paste ? k->paste : ""; *p; p++)
                if (*p == '\t') buf_addc(&clean, ' ');
                else if (*p != '\r' && *p != '\n') buf_addc(&clean, *p);
            ins = clean.ptr ? clean.ptr : "";
            n = clean.len;
        } else n = (size_t)tui_utf8(enc, k->ch);
        buf_reserve(v, n);
        memmove(v->ptr + in->cur + n, v->ptr + in->cur, v->len - in->cur + 1);
        memcpy(v->ptr + in->cur, ins, n);
        v->len += n;
        in->cur += n;
        buf_free(&clean);
        return true;
    }
    if (k->kind == K_BACKSPACE || (k->kind == K_CTRL && k->ch == 'h')) {
        if (!in->cur) return false;
        size_t p = in->cur - 1;
        while (p > 0 && ((unsigned char)v->ptr[p] & 0xc0) == 0x80) p--;
        memmove(v->ptr + p, v->ptr + in->cur, v->len - in->cur + 1);
        v->len -= in->cur - p;
        in->cur = p;
        return true;
    }
    if (k->kind == K_DELETE) {
        if (in->cur >= v->len) return false;
        size_t g = tui_grapheme(buf_str(v), in->cur);
        memmove(v->ptr + in->cur, v->ptr + in->cur + g, v->len - in->cur - g + 1);
        v->len -= g;
        return true;
    }
    if ((k->kind == K_CTRL && k->ch == 'w') || k->kind == K_ALT_BACKSPACE || k->kind == K_CTRL_BACKSPACE) {
        size_t p = in->cur;
        while (p > 0 && !word_char(v->ptr[p - 1])) p--;
        while (p > 0 && word_char(v->ptr[p - 1])) p--;
        memmove(v->ptr + p, v->ptr + in->cur, v->len - in->cur + 1);
        v->len -= in->cur - p;
        in->cur = p;
        return true;
    }
    if (k->kind == K_CTRL && k->ch == 'u') {
        memmove(v->ptr, v->ptr + in->cur, v->len - in->cur + 1);
        v->len -= in->cur;
        in->cur = 0;
        return true;
    }
    if (k->kind == K_CTRL && k->ch == 'k') {
        v->len = in->cur;
        v->ptr[v->len] = 0;
        return true;
    }
    if (k->kind == K_LEFT || (k->kind == K_CTRL && k->ch == 'b')) {
        if (in->cur > 0) {
            in->cur--;
            while (in->cur > 0 && ((unsigned char)v->ptr[in->cur] & 0xc0) == 0x80) in->cur--;
        }
        return true;
    }
    if (k->kind == K_RIGHT || (k->kind == K_CTRL && k->ch == 'f')) {
        if (in->cur < v->len) in->cur += tui_grapheme(buf_str(v), in->cur);
        return true;
    }
    if (k->kind == K_HOME || (k->kind == K_CTRL && k->ch == 'a')) {
        in->cur = 0;
        return true;
    }
    if (k->kind == K_END || (k->kind == K_CTRL && k->ch == 'e')) {
        in->cur = v->len;
        return true;
    }
    if (k->kind == K_ALT_LEFT || k->kind == K_CTRL_LEFT) {
        while (in->cur > 0 && !word_char(v->ptr[in->cur - 1])) in->cur--;
        while (in->cur > 0 && word_char(v->ptr[in->cur - 1])) in->cur--;
        return true;
    }
    if (k->kind == K_ALT_RIGHT || k->kind == K_CTRL_RIGHT) {
        while (in->cur < v->len && !word_char(v->ptr[in->cur])) in->cur++;
        while (in->cur < v->len && word_char(v->ptr[in->cur])) in->cur++;
        return true;
    }
    return false;
}

static bool is_cancel(const Key *k) { return k->kind == K_ESC || (k->kind == K_CTRL && k->ch == 'c'); }

// ---- /fork (showUserMessageSelector, UserMessageSelectorComponent) -----------------------------------

typedef struct {
    Arena a;
    JValue *msgs; // {"entryId","text"}
    size_t sel;
} ForkSel;

static void fork_free(void *p) {
    ForkSel *f = p;
    arena_reset(&f->a);
    free(f);
}

static void fork_render(int width, Lines *out) {
    ForkSel *f = uix_selector_data();
    Buf b = {0};
    lines_addz(out, "");
    th_bold(&b, S("Fork from Message"));
    text_lines(buf_str(&b), 1, width, out);
    b.len = 0;
    th_fgz(&b, T_MUTED, "Select a user message to copy the active path up to that point into a new session");
    text_lines(buf_str(&b), 1, width, out);
    lines_addz(out, "");
    border(out, width, T_BORDER);
    lines_addz(out, "");
    size_t n = f->msgs->arr.len;
    const size_t max_visible = 10;
    if (!n) {
        b.len = 0;
        th_fgz(&b, T_MUTED, "  No user messages found");
        lines_addb(out, &b);
    } else {
        ptrdiff_t st = (ptrdiff_t)f->sel - (ptrdiff_t)(max_visible / 2);
        if (st > (ptrdiff_t)n - (ptrdiff_t)max_visible) st = (ptrdiff_t)n - (ptrdiff_t)max_visible;
        size_t start = st > 0 ? (size_t)st : 0;
        size_t end = start + max_visible < n ? start + max_visible : n;
        for (size_t i = start; i < end; i++) {
            Str text = json_strs(json_get(f->msgs->arr.items[i], "text"));
            // message.text.replace(/\n/g, " ").trim()
            Buf norm = {0};
            for (size_t k = 0; k < text.len; k++) buf_addc(&norm, text.ptr[k] == '\n' ? ' ' : text.ptr[k]);
            Str t = str_trim(buf_str(&norm));
            char *cut = tui_truncatez(t, width - 2, "...", false);
            Buf l = {0};
            if (i == f->sel) {
                th_fgz(&l, T_ACCENT, "\xe2\x80\xba ");
                th_bold(&l, str(cut));
            } else {
                buf_adds(&l, "  ");
                buf_adds(&l, cut);
            }
            lines_addb(out, &l);
            free(cut);
            buf_free(&norm);
            Buf meta = {0}, ml = {0};
            buf_printf(&meta, "  Message %zu of %zu", i + 1, n);
            th_fg(&ml, T_MUTED, buf_str(&meta));
            lines_addb(out, &ml);
            buf_free(&meta);
            lines_addz(out, "");
        }
        if (start > 0 || end < n) {
            Buf s = {0}, sl = {0};
            buf_printf(&s, "  (%zu/%zu)", f->sel + 1, n);
            th_fg(&sl, T_MUTED, buf_str(&s));
            lines_addb(out, &sl);
            buf_free(&s);
        }
    }
    buf_free(&b);
    lines_addz(out, "");
    border(out, width, T_BORDER);
}

// The session drawn again after it was replaced (rebindCurrentSession: renderCurrentSessionState).
static void after_switch(void) {
    CodeSession *s = uix_ops()->session(uix_host());
    const char *level = json_str(code_session_last(s, "thinking_level_change", "thinkingLevel"));
    if (level) uix_show_thinking(level);
    sess_ui_reset();
    uix_render_history();
    code_trust_warning(); // renderInitialMessages' renderProjectTrustWarningIfNeeded (code_trust.c)
}

static void fork_at(const char *entry_id) {
    char *selected = NULL;
    const char *err = uix_ops()->fork(uix_host(), entry_id, false, &selected);
    if (err) {
        uix_show_error(err);
        return;
    }
    after_switch();
    editor_set_text(uix_editor(), str(selected ? selected : ""));
    uix_show_status("Forked to new session");
}

static void fork_key(const Key *k) {
    ForkSel *f = uix_selector_data();
    size_t n = f->msgs->arr.len;
    if (k->kind == K_UP) f->sel = f->sel == 0 ? (n ? n - 1 : 0) : f->sel - 1;
    else if (k->kind == K_DOWN) f->sel = n && f->sel == n - 1 ? 0 : f->sel + 1;
    else if (k->kind == K_PGUP || k->kind == K_PGDN) {
        if (!n) return;
        long t = (long)f->sel + (k->kind == K_PGUP ? -10 : 10);
        f->sel = t < 0 ? 0 : t > (long)n - 1 ? n - 1 : (size_t)t;
    } else if (k->kind == K_ENTER) {
        if (f->sel >= n) return;
        char *id = strdup(json_str(json_get(f->msgs->arr.items[f->sel], "entryId")) ?: "");
        uix_close_selector();
        fork_at(id);
        free(id);
    } else if (is_cancel(k)) uix_close_selector();
}

static void fork_command(void) {
    CodeSession *s = uix_ops()->session(uix_host());
    ForkSel *f = xcalloc(1, sizeof *f);
    f->msgs = code_sessions_fork_messages(&f->a, code_session_entries(s));
    if (!f->msgs->arr.len) {
        fork_free(f);
        uix_show_status("No messages to fork from");
        return;
    }
    f->sel = f->msgs->arr.len - 1; // the most recent
    uix_open_selector(fork_render, fork_key, fork_free, f);
}

// ---- /clone (handleCloneCommand) ------------------------------------------------------------------------

static void clone_command(void) {
    CodeSession *s = uix_ops()->session(uix_host());
    const char *leaf = code_session_leaf(s);
    if (!leaf) {
        uix_show_status("Nothing to clone yet");
        return;
    }
    char *id = strdup(leaf), *selected = NULL;
    const char *err = uix_ops()->fork(uix_host(), id, true, &selected);
    free(id);
    if (err) {
        uix_show_error(err);
        return;
    }
    after_switch();
    editor_set_text(uix_editor(), S(""));
    uix_show_status("Cloned to new session");
}

// ---- /resume (showSessionSelector, SessionSelectorComponent) -------------------------------------------

static double now_wall(void) { return wall_ms(); }

// formatSessionDate
static void session_age(Buf *b, double modified) {
    double diff = now_wall() - modified;
    double mins = floor(diff / 60000), hours = floor(diff / 3600000), days = floor(diff / 86400000);
    if (mins < 1) buf_adds(b, "now");
    else if (mins < 60) buf_printf(b, "%.0fm", mins);
    else if (hours < 24) buf_printf(b, "%.0fh", hours);
    else if (days < 7) buf_printf(b, "%.0fd", days);
    else if (days < 30) buf_printf(b, "%.0fw", floor(days / 7));
    else if (days < 365) buf_printf(b, "%.0fmo", floor(days / 30));
    else buf_printf(b, "%.0fy", floor(days / 365));
}

// shortenHomePath
static void short_home(Buf *b, const char *path) {
    const char *home = getenv("HOME");
    size_t hl = home ? strlen(home) : 0;
    while (hl > 1 && home[hl - 1] == '/') hl--;
    if (hl && path[0] == '/' && !strncmp(path, home, hl) && (path[hl] == '/' || !path[hl])) {
        const char *rest = path + hl;
        while (*rest == '/') rest++;
        if (*rest) buf_printf(b, "~/%s", rest);
        else buf_adds(b, "~");
    } else buf_adds(b, path);
}

// getSessionPathKey: the path with links resolved
static char *path_key(const char *p) {
    char *r = realpath(p, NULL);
    return r ? r : strdup(p);
}

typedef struct {
    const CodeSessionInfo *s;
    int depth;
    bool last;
    VEC(bool) cont; // ancestorContinues
} Flat;

typedef struct {
    Arena a;
    int scope; // 0 current folder, 1 all
    int sort;  // 0 threaded, 1 recent, 2 relevance (Fuzzy)
    bool named;
    CodeSessionInfo *cur, *all;
    size_t ncur, nall;
    bool have_cur, have_all;
    // the list
    const CodeSessionInfo *shown; // the sessions of the scope (setSessions)
    size_t nshown;
    bool show_cwd;
    VEC(Flat) flat;
    size_t sel;
    bool touched;
    Inp search;
    bool show_path;
    char *confirm; // the session being deleted
    char *current_key;
    // the header's status line
    char *status;
    bool status_error;
    uint64_t status_until;
    // rename
    bool renaming;
    Inp rename;
    char *rename_target;
} ResumeSel;

static ResumeSel *g_resume; // the open selector (for its status timer)

static void flat_clear(ResumeSel *r) {
    for (size_t i = 0; i < r->flat.len; i++) vec_free(&r->flat.ptr[i].cont);
    r->flat.len = 0;
}

static void resume_free(void *p) {
    ResumeSel *r = p;
    flat_clear(r);
    vec_free(&r->flat);
    inp_free(&r->search);
    inp_free(&r->rename);
    free(r->confirm);
    free(r->current_key);
    free(r->status);
    free(r->rename_target);
    arena_reset(&r->a);
    if (g_resume == r) g_resume = NULL;
    free(r);
}

static void set_status(ResumeSel *r, const char *msg, bool error, int ms) {
    free(r->status);
    r->status = msg ? strdup(msg) : NULL;
    r->status_error = error;
    r->status_until = msg && ms ? now_ns() + (uint64_t)ms * 1000000ull : 0;
}

// ---- search (session-selector-search.ts) ----

static double score_positions(const char *lower, const size_t *pos, size_t n) {
    double score = 0;
    long last = -1;
    int consec = 0;
    for (size_t k = 0; k < n; k++) {
        long i = (long)pos[k];
        if (last == i - 1) {
            consec++;
            score -= consec * 5;
        } else {
            consec = 0;
            if (last >= 0) score += (double)(i - last - 1) * 2;
        }
        if (i == 0 || strchr(" \t\n\r\f\v-_./:", lower[i - 1])) score -= 10;
        score += (double)i * 0.1;
        last = i;
    }
    return score;
}

static char *lower(const char *s) {
    char *r = strdup(s ? s : "");
    for (char *p = r; *p; p++)
        if (*p >= 'A' && *p <= 'Z') *p = (char)(*p + 32);
    return r;
}

static bool match_query(const char *q, const char *t, double *score) {
    size_t ql = strlen(q), tl = strlen(t);
    if (!ql) {
        *score = 0;
        return true;
    }
    if (ql > tl) return false;
    size_t *pos = xmalloc(ql * sizeof *pos);
    long last = -1;
    for (size_t k = 0; k < ql; k++) {
        const char *f = memchr(t + last + 1, q[k], tl - (size_t)(last + 1));
        if (!f) {
            free(pos);
            return false;
        }
        pos[k] = (size_t)(f - t);
        last = (long)pos[k];
    }
    double s = score_positions(t, pos, ql);
    for (const char *st = strstr(t, q); st; st = strstr(st + 1, q)) {
        for (size_t k = 0; k < ql; k++) pos[k] = (size_t)(st - t) + k;
        double s2 = score_positions(t, pos, ql);
        if (s2 < s) s = s2;
    }
    if (!strcmp(q, t)) s -= 100;
    free(pos);
    *score = s;
    return true;
}

// fuzzyMatch (pi-tui fuzzy.ts), with the letters/digits swap
static bool fuzzy(const char *query, const char *text, double *score) {
    char *q = lower(query), *t = lower(text);
    bool ok = match_query(q, t, score);
    if (!ok) {
        size_t n = strlen(q), i = 0;
        bool alpha = n && q[0] >= 'a' && q[0] <= 'z', digit = n && q[0] >= '0' && q[0] <= '9';
        if (alpha) while (i < n && q[i] >= 'a' && q[i] <= 'z') i++;
        else if (digit) while (i < n && q[i] >= '0' && q[i] <= '9') i++;
        size_t j = i;
        if (alpha) while (j < n && q[j] >= '0' && q[j] <= '9') j++;
        else if (digit) while (j < n && q[j] >= 'a' && q[j] <= 'z') j++;
        if ((alpha || digit) && i > 0 && i < n && j == n) {
            char *sw = xmalloc(n + 1);
            memcpy(sw, q + i, n - i);
            memcpy(sw + n - i, q, i);
            sw[n] = 0;
            double s2;
            if (match_query(sw, t, &s2)) {
                ok = true;
                *score = s2 + 5;
            }
            free(sw);
        }
    }
    free(q);
    free(t);
    return ok;
}

enum { TIER_NAME = 0, TIER_TEXT = 1, TIER_FUZZY = 2 };
#define TIER_WEIGHT 1000000.0

// normalizeWhitespaceLower
static char *norm_ws_lower(const char *s) {
    Buf b = {0};
    bool sp = false;
    for (const char *p = s ? s : ""; *p; p++) {
        char c = *p;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v') {
            sp = true;
            continue;
        }
        if (sp && b.len) buf_addc(&b, ' ');
        sp = false;
        buf_addc(&b, c >= 'A' && c <= 'Z' ? (char)(c + 32) : c);
    }
    char *r = strdup(b.ptr ? b.ptr : "");
    buf_free(&b);
    return r;
}

static bool contains_lower(const char *hay, const char *needle) {
    char *h = lower(hay);
    bool r = strstr(h, needle) != NULL;
    free(h);
    return r;
}

// matchLiteralToken (value lowercased)
static bool literal_token(const CodeSessionInfo *s, const char *value, double *score) {
    char *name = lower(s->name ? s->name : "");
    if (strstr(name, value)) {
        double f = 0;
        fuzzy(value, name, &f);
        *score = TIER_NAME * TIER_WEIGHT + f;
        free(name);
        return true;
    }
    free(name);
    if (contains_lower(s->id, value) || contains_lower(s->all_text, value) || contains_lower(s->cwd, value)) {
        *score = TIER_TEXT * TIER_WEIGHT;
        return true;
    }
    return false;
}

static bool fuzzy_token(const CodeSessionInfo *s, const char *value, double *score) {
    char *lv = lower(value);
    bool lit = literal_token(s, lv, score);
    free(lv);
    if (lit) return true;
    bool any = false;
    double best = 0;
    const char *fields[2] = {s->name ? s->name : "", s->cwd};
    for (int i = 0; i < 2; i++) {
        double sc;
        if (fuzzy(value, fields[i], &sc) && (!any || sc < best)) {
            best = sc;
            any = true;
        }
    }
    if (any) *score = TIER_FUZZY * TIER_WEIGHT + best;
    return any;
}

static bool phrase_token(const CodeSessionInfo *s, const char *phrase, double *score) {
    char *name = norm_ws_lower(s->name ? s->name : "");
    bool r = false;
    if (strstr(name, phrase)) {
        double f = 0;
        fuzzy(phrase, name, &f);
        *score = TIER_NAME * TIER_WEIGHT + f;
        r = true;
    } else {
        char *t = norm_ws_lower(s->all_text), *c = norm_ws_lower(s->cwd);
        if (contains_lower(s->id, phrase) || strstr(t, phrase) || strstr(c, phrase)) {
            *score = TIER_TEXT * TIER_WEIGHT;
            r = true;
        }
        free(t);
        free(c);
    }
    free(name);
    return r;
}

typedef struct {
    bool regex_mode, error;
    regex_t re;
    bool have_re;
    VEC(char *) tokens;
    VEC(bool) phrase;
} Query;

static void query_free(Query *q) {
    for (size_t i = 0; i < q->tokens.len; i++) free(q->tokens.ptr[i]);
    vec_free(&q->tokens);
    vec_free(&q->phrase);
    if (q->have_re) regfree(&q->re);
}

static void push_token(Query *q, Buf *b, bool phrase) {
    Str t = str_trim(buf_str(b));
    if (t.len) {
        vec_push(&q->tokens, strndup(t.ptr, t.len));
        vec_push(&q->phrase, phrase);
    }
    b->len = 0;
}

// parseSearchQuery (POSIX extended regular expressions stand in for JavaScript's)
static void parse_query(const char *query, Query *q) {
    *q = (Query){0};
    Str t = str_trim(str(query));
    if (!t.len) return;
    if (str_starts(t, S("re:"))) {
        q->regex_mode = true;
        Str pat = str_trim(str_sub(t, 3, t.len));
        if (!pat.len) {
            q->error = true;
            return;
        }
        char *p = strndup(pat.ptr, pat.len);
        if (regcomp(&q->re, p, REG_EXTENDED | REG_ICASE | REG_NOSUB) == 0) q->have_re = true;
        else q->error = true;
        free(p);
        return;
    }
    Buf b = {0};
    bool inq = false;
    for (size_t i = 0; i < t.len; i++) {
        char c = t.ptr[i];
        if (c == '"') {
            if (inq) {
                push_token(q, &b, true);
                inq = false;
            } else {
                push_token(q, &b, false);
                inq = true;
            }
            continue;
        }
        if (!inq && (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v')) {
            push_token(q, &b, false);
            continue;
        }
        buf_addc(&b, c);
    }
    if (inq) {
        // unbalanced quotes: plain whitespace tokens
        for (size_t i = 0; i < q->tokens.len; i++) free(q->tokens.ptr[i]);
        q->tokens.len = q->phrase.len = 0;
        b.len = 0;
        for (size_t i = 0; i <= t.len; i++) {
            char c = i < t.len ? t.ptr[i] : ' ';
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v') push_token(q, &b, false);
            else buf_addc(&b, c);
        }
    } else push_token(q, &b, false);
    buf_free(&b);
}

// matchSession
static bool match_session(const CodeSessionInfo *s, const Query *q, double *score) {
    if (q->regex_mode) {
        if (!q->have_re) return false;
        if (s->name && regexec(&q->re, s->name, 0, NULL, 0) == 0) {
            *score = TIER_NAME * TIER_WEIGHT;
            return true;
        }
        if (regexec(&q->re, s->id, 0, NULL, 0) == 0 || regexec(&q->re, s->all_text, 0, NULL, 0) == 0 || regexec(&q->re, s->cwd, 0, NULL, 0) == 0) {
            *score = TIER_TEXT * TIER_WEIGHT;
            return true;
        }
        return false;
    }
    double total = 0;
    for (size_t i = 0; i < q->tokens.len; i++) {
        double sc;
        bool ok;
        if (q->phrase.ptr[i]) {
            char *ph = norm_ws_lower(q->tokens.ptr[i]);
            if (!*ph) {
                free(ph);
                continue;
            }
            ok = phrase_token(s, ph, &sc);
            free(ph);
        } else ok = fuzzy_token(s, q->tokens.ptr[i], &sc);
        if (!ok) return false;
        total += sc;
    }
    *score = total;
    return true;
}

// ---- the list (SessionList) ----

typedef struct {
    const CodeSessionInfo *s;
    char *key, *parent;
    double latest;
    VEC(size_t) kids;
} Node;

static double node_latest(Node *nodes, size_t i) {
    double l = nodes[i].s->modified;
    for (size_t k = 0; k < nodes[i].kids.len; k++) {
        double c = node_latest(nodes, nodes[i].kids.ptr[k]);
        if (c > l) l = c;
    }
    nodes[i].latest = l;
    return l;
}

static void sort_by_latest(Node *nodes, size_t *v, size_t n) {
    for (size_t i = 1; i < n; i++) { // stable insertion sort, the latest first
        size_t x = v[i], j = i;
        while (j > 0 && nodes[v[j - 1]].latest < nodes[x].latest) {
            v[j] = v[j - 1];
            j--;
        }
        v[j] = x;
    }
    for (size_t i = 0; i < n; i++) sort_by_latest(nodes, nodes[v[i]].kids.ptr, nodes[v[i]].kids.len);
}

static void flat_walk(ResumeSel *r, Node *nodes, size_t i, int depth, const bool *cont, size_t ncont, bool last) {
    Flat f = {.s = nodes[i].s, .depth = depth, .last = last};
    for (size_t k = 0; k < ncont; k++) vec_push(&f.cont, cont[k]);
    vec_push(&r->flat, f);
    for (size_t k = 0; k < nodes[i].kids.len; k++) {
        bool child_last = k == nodes[i].kids.len - 1;
        bool c = depth > 0 ? !last : false;
        bool *nc = xmalloc((ncont + 1) * sizeof *nc);
        memcpy(nc, cont, ncont * sizeof *nc);
        nc[ncont] = c;
        flat_walk(r, nodes, nodes[i].kids.ptr[k], depth + 1, nc, ncont + 1, child_last);
        free(nc);
    }
}

// buildSessionTree and flattenSessionTree
static void threaded(ResumeSel *r, const CodeSessionInfo **v, size_t n) {
    Node *nodes = xcalloc(n + 1, sizeof *nodes);
    for (size_t i = 0; i < n; i++) {
        nodes[i].s = v[i];
        nodes[i].key = path_key(v[i]->path);
        nodes[i].parent = v[i]->parent ? path_key(v[i]->parent) : NULL;
        nodes[i].latest = v[i]->modified;
    }
    VEC(size_t) roots = {0};
    for (size_t i = 0; i < n; i++) {
        size_t p = n;
        // byPath keeps the last session of a path
        for (size_t k = 0; nodes[i].parent && k < n; k++)
            if (!strcmp(nodes[k].key, nodes[i].parent)) p = k;
        if (p < n) vec_push(&nodes[p].kids, i);
        else vec_push(&roots, i);
    }
    for (size_t i = 0; i < roots.len; i++) node_latest(nodes, roots.ptr[i]);
    sort_by_latest(nodes, roots.ptr, roots.len);
    for (size_t i = 0; i < roots.len; i++) flat_walk(r, nodes, roots.ptr[i], 0, NULL, 0, i == roots.len - 1);
    for (size_t i = 0; i < n; i++) {
        free(nodes[i].key);
        free(nodes[i].parent);
        vec_free(&nodes[i].kids);
    }
    free(nodes);
    vec_free(&roots);
}

// filterSessions
static void filter_sessions(ResumeSel *r) {
    flat_clear(r);
    const CodeSessionInfo **v = xcalloc(r->nshown + 1, sizeof *v);
    size_t n = 0;
    for (size_t i = 0; i < r->nshown; i++)
        if (!r->named || (r->shown[i].name && str_trim(str(r->shown[i].name)).len)) v[n++] = &r->shown[i];
    Str q = str_trim(inp_str(&r->search));
    if (r->sort == 0 && !q.len) threaded(r, v, n);
    else if (q.len) {
        // filterAndSortSessions
        Query pq;
        parse_query(r->search.v.ptr ? r->search.v.ptr : "", &pq);
        if (!pq.error) {
            typedef struct {
                const CodeSessionInfo *s;
                double score;
            } Sc;
            VEC(Sc) sc = {0};
            for (size_t i = 0; i < n; i++) {
                double s;
                if (match_session(v[i], &pq, &s)) vec_push(&sc, ((Sc){v[i], s}));
            }
            if (r->sort != 1) // relevance: by score, then the newest
                for (size_t i = 1; i < sc.len; i++) {
                    Sc x = sc.ptr[i];
                    size_t j = i;
                    while (j > 0 && (sc.ptr[j - 1].score > x.score || (sc.ptr[j - 1].score == x.score && sc.ptr[j - 1].s->modified < x.s->modified))) {
                        sc.ptr[j] = sc.ptr[j - 1];
                        j--;
                    }
                    sc.ptr[j] = x;
                }
            for (size_t i = 0; i < sc.len; i++) vec_push(&r->flat, ((Flat){.s = sc.ptr[i].s, .depth = 0, .last = true}));
            vec_free(&sc);
        }
        query_free(&pq);
    } else
        for (size_t i = 0; i < n; i++) vec_push(&r->flat, ((Flat){.s = v[i], .depth = 0, .last = true}));
    free(v);
    size_t maxi = r->flat.len ? r->flat.len - 1 : 0;
    if (r->sel > maxi) r->sel = maxi;
}

// setSessions
static void set_sessions(ResumeSel *r, const CodeSessionInfo *s, size_t n, bool show_cwd) {
    char *selected = r->touched && r->sel < r->flat.len ? strdup(r->flat.ptr[r->sel].s->path) : NULL;
    r->shown = s;
    r->nshown = n;
    r->show_cwd = show_cwd;
    filter_sessions(r);
    if (!r->touched) r->sel = 0;
    else if (selected)
        for (size_t i = 0; i < r->flat.len; i++)
            if (!strcmp(r->flat.ptr[i].s->path, selected)) {
                r->sel = i;
                break;
            }
    free(selected);
}

typedef struct {
    ResumeSel *r;
    int scope;
} LoadCtx;

static void on_progress(void *ctx, size_t loaded, size_t total, const CodeSessionInfo *partial, size_t npartial) {
    LoadCtx *c = ctx;
    (void)loaded;
    (void)total;
    if (!partial) return;
    ResumeSel *r = c->r;
    // a copy: the loader reuses its partial array
    CodeSessionInfo *copy = arena_alloc(&r->a, (npartial + 1) * sizeof *copy);
    memcpy(copy, partial, npartial * sizeof *copy);
    if (c->scope == 0) {
        r->cur = copy;
        r->ncur = npartial;
    } else {
        r->all = copy;
        r->nall = npartial;
    }
    if (c->scope == r->scope) set_sessions(r, copy, npartial, c->scope == 1);
}

// loadScope (the loads finish before the next frame: pi's progress steps run in order)
static void load_scope(ResumeSel *r, int scope) {
    const CodeUiConfig *cfg = uix_cfg();
    LoadCtx c = {r, scope};
    CodeSessionInfo *list;
    size_t n;
    if (scope == 0) n = code_sessions_list(&r->a, code_sessions_dir(&r->a, cfg->agent_dir, cfg->cwd), on_progress, &c, &list);
    else n = code_sessions_list_all(&r->a, cfg->agent_dir, on_progress, &c, &list);
    if (scope == 0) {
        r->cur = list;
        r->ncur = n;
        r->have_cur = true;
    } else {
        r->all = list;
        r->nall = n;
        r->have_all = true;
    }
    if (scope == r->scope) set_sessions(r, list, n, scope == 1);
}

static void refresh_after_mutation(ResumeSel *r) {
    r->have_cur = r->have_all = false;
    r->cur = r->all = NULL;
    r->ncur = r->nall = 0;
    load_scope(r, r->scope);
}

static bool is_current(const ResumeSel *r, const char *path) {
    if (!r->current_key) return false;
    char *k = path_key(path);
    bool eq = !strcmp(k, r->current_key);
    free(k);
    return eq;
}

static void header_render(ResumeSel *r, int width, Lines *out) {
    Buf left = {0}, right = {0};
    th_bold(&left, str(r->scope == 0 ? "Resume Session (Current Folder)" : "Resume Session (All)"));
    Buf scope = {0};
    if (r->scope == 0) {
        th_fgz(&scope, T_ACCENT, "\xe2\x97\x89 Current Folder");
        th_fgz(&scope, T_MUTED, " | \xe2\x97\x8b All");
    } else {
        th_fgz(&scope, T_MUTED, "\xe2\x97\x8b Current Folder | ");
        th_fgz(&scope, T_ACCENT, "\xe2\x97\x89 All");
    }
    buf_addstr(&right, buf_str(&scope));
    buf_adds(&right, "  ");
    th_fgz(&right, T_MUTED, "Name: ");
    th_fgz(&right, T_ACCENT, r->named ? "Named" : "All");
    buf_adds(&right, "  ");
    th_fgz(&right, T_MUTED, "Sort: ");
    th_fgz(&right, T_ACCENT, r->sort == 0 ? "Threaded" : r->sort == 1 ? "Recent" : "Fuzzy");
    char *rt = tui_truncatez(buf_str(&right), width, "", false);
    int avail = width - tui_width(str(rt)) - 1;
    char *lt = tui_truncatez(buf_str(&left), avail > 0 ? avail : 0, "", false);
    int spacing = width - tui_width(str(lt)) - tui_width(str(rt));
    Buf l1 = {0};
    buf_adds(&l1, lt);
    tui_spaces(&l1, spacing > 0 ? spacing : 0);
    buf_adds(&l1, rt);
    lines_addb(out, &l1);
    free(rt);
    free(lt);
    buf_free(&left);
    buf_free(&right);
    buf_free(&scope);
    if (r->confirm) {
        Buf h = {0};
        buf_adds(&h, "Delete session? ");
        key_hint(&h, "enter", "confirm");
        buf_adds(&h, " \xc2\xb7 ");
        key_hint(&h, "escape/ctrl+c", "cancel");
        char *c = tui_truncatez(buf_str(&h), width, "\xe2\x80\xa6", false);
        Buf l = {0};
        th_fgz(&l, T_ERROR, c);
        lines_addb(out, &l);
        lines_addz(out, "");
        free(c);
        buf_free(&h);
    } else if (r->status) {
        char *c = tui_truncatez(str(r->status), width, "\xe2\x80\xa6", false);
        Buf l = {0};
        th_fgz(&l, r->status_error ? T_ERROR : T_ACCENT, c);
        lines_addb(out, &l);
        lines_addz(out, "");
        free(c);
    } else {
        Buf sep = {0}, h1 = {0}, h2 = {0};
        th_fgz(&sep, T_MUTED, " \xc2\xb7 ");
        key_hint(&h1, "tab", "scope");
        buf_addstr(&h1, buf_str(&sep));
        th_fgz(&h1, T_MUTED, "re:<pattern> regex \xc2\xb7 \"phrase\" exact");
        key_hint(&h2, "ctrl+s", "sort");
        buf_addstr(&h2, buf_str(&sep));
        key_hint(&h2, "ctrl+n", "named");
        buf_addstr(&h2, buf_str(&sep));
        key_hint(&h2, "ctrl+d", "delete");
        buf_addstr(&h2, buf_str(&sep));
        key_hint(&h2, "ctrl+p", r->show_path ? "path (on)" : "path (off)");
        buf_addstr(&h2, buf_str(&sep));
        key_hint(&h2, "ctrl+r", "rename");
        lines_push(out, tui_truncatez(buf_str(&h1), width, "\xe2\x80\xa6", false));
        lines_push(out, tui_truncatez(buf_str(&h2), width, "\xe2\x80\xa6", false));
        buf_free(&sep);
        buf_free(&h1);
        buf_free(&h2);
    }
}

static void list_render(ResumeSel *r, int width, Lines *out) {
    inp_render(&r->search, width, out);
    lines_addz(out, "");
    if (!r->flat.len) {
        const char *msg;
        if (r->named)
            msg = r->show_cwd ? "  No named sessions found. Press ctrl+n to show all."
                              : "  No named sessions in current folder. Press ctrl+n to show all, or Tab to view all.";
        else msg = r->show_cwd ? "  No sessions found" : "  No sessions in current folder. Press Tab to view all.";
        char *c = tui_truncatez(str(msg), width, "\xe2\x80\xa6", false);
        Buf l = {0};
        th_fgz(&l, T_MUTED, c);
        lines_addb(out, &l);
        free(c);
        return;
    }
    const size_t max_visible = 10;
    size_t n = r->flat.len;
    ptrdiff_t st = (ptrdiff_t)r->sel - (ptrdiff_t)(max_visible / 2);
    if (st > (ptrdiff_t)n - (ptrdiff_t)max_visible) st = (ptrdiff_t)n - (ptrdiff_t)max_visible;
    size_t start = st > 0 ? (size_t)st : 0, end = start + max_visible < n ? start + max_visible : n;
    for (size_t i = start; i < end; i++) {
        const Flat *f = &r->flat.ptr[i];
        const CodeSessionInfo *s = f->s;
        bool selected = i == r->sel, deleting = r->confirm && !strcmp(s->path, r->confirm), current = is_current(r, s->path);
        Buf prefix = {0};
        if (f->depth > 0) {
            for (size_t k = 0; k < f->cont.len; k++) buf_adds(&prefix, f->cont.ptr[k] ? "\xe2\x94\x82  " : "   ");
            buf_adds(&prefix, f->last ? "\xe2\x94\x94\xe2\x94\x80 " : "\xe2\x94\x9c\xe2\x94\x80 ");
        }
        // the name or the first message, control characters as spaces
        Buf msg = {0};
        for (const char *p = s->name ? s->name : s->first_message; *p; p++)
            buf_addc(&msg, ((unsigned char)*p < 0x20 || *p == 0x7f) ? ' ' : *p);
        Str norm = str_trim(buf_str(&msg));
        Buf rp = {0};
        if (r->show_path) {
            short_home(&rp, s->path);
            buf_addc(&rp, ' ');
        }
        if (r->show_cwd && s->cwd[0]) {
            short_home(&rp, s->cwd);
            buf_addc(&rp, ' ');
        }
        buf_printf(&rp, "%zu ", s->message_count);
        session_age(&rp, s->modified);
        int avail = width - 2 - tui_width(buf_str(&prefix)) - (tui_width(buf_str(&rp)) + 2);
        char *cut = tui_truncatez(norm, avail > 10 ? avail : 10, "\xe2\x80\xa6", false);
        Buf styled = {0};
        ThemeColor color = T_TEXT;
        bool colored = deleting || current || s->name;
        if (deleting) color = T_ERROR;
        else if (current) color = T_ACCENT;
        else if (s->name) color = T_WARNING;
        if (colored) th_fgz(&styled, color, cut);
        else buf_adds(&styled, cut);
        if (selected) {
            Buf bold = {0};
            th_bold(&bold, buf_str(&styled));
            buf_free(&styled);
            styled = bold;
        }
        Buf line = {0};
        if (selected) th_fgz(&line, T_ACCENT, "\xe2\x80\xba ");
        else buf_adds(&line, "  ");
        th_fg(&line, T_DIM, buf_str(&prefix));
        buf_addstr(&line, buf_str(&styled));
        int lw = tui_width(buf_str(&line));
        int spacing = width - lw - tui_width(buf_str(&rp));
        tui_spaces(&line, spacing > 1 ? spacing : 1);
        th_fg(&line, deleting ? T_ERROR : T_DIM, buf_str(&rp));
        if (selected) {
            Buf bg = {0};
            th_bg(&bg, TH_BG_SELECTED, buf_str(&line));
            buf_free(&line);
            line = bg;
        }
        lines_push(out, tui_truncatez(buf_str(&line), width, "...", false));
        free(cut);
        buf_free(&prefix);
        buf_free(&msg);
        buf_free(&rp);
        buf_free(&styled);
        buf_free(&line);
    }
    if (start > 0 || end < n) {
        Buf s = {0}, l = {0};
        buf_printf(&s, "  (%zu/%zu)", r->sel + 1, n);
        char *c = tui_truncatez(buf_str(&s), width, "", false);
        th_fgz(&l, T_MUTED, c);
        lines_addb(out, &l);
        free(c);
        buf_free(&s);
    }
}

static void resume_render(int width, Lines *out) {
    ResumeSel *r = uix_selector_data();
    if (r->status && r->status_until && now_ns() >= r->status_until) set_status(r, NULL, false, 0);
    lines_addz(out, "");
    border(out, width, T_ACCENT);
    lines_addz(out, "");
    if (r->renaming) {
        Buf b = {0};
        th_bold(&b, S("Rename Session"));
        text_lines(buf_str(&b), 1, width, out);
        lines_addz(out, "");
        inp_render(&r->rename, width, out);
        lines_addz(out, "");
        b.len = 0;
        th_fgz(&b, T_MUTED, "enter to save \xc2\xb7 escape/ctrl+c to cancel");
        text_lines(buf_str(&b), 1, width, out);
        buf_free(&b);
    } else {
        header_render(r, width, out);
        lines_addz(out, "");
        list_render(r, width, out);
    }
    lines_addz(out, "");
    border(out, width, T_ACCENT);
}

// deleteSessionFile: the trash command first, then unlink
static const char *delete_file(const char *path, bool *trashed) {
    char *argv[4] = {"trash", NULL, NULL, NULL};
    int k = 1;
    if (path[0] == '-') argv[k++] = "--";
    argv[k] = (char *)path;
    Buf out = {0};
    bool ok = tui_run(argv, NULL, S(""), &out, 10000);
    buf_free(&out);
    if (ok || access(path, F_OK) != 0) {
        *trashed = true;
        return NULL;
    }
    *trashed = false;
    if (unlink(path) == 0) return NULL;
    return strerror(errno);
}

static void toggle_scope(ResumeSel *r) {
    r->scope = 1 - r->scope;
    bool have = r->scope == 0 ? r->have_cur : r->have_all;
    if (have) set_sessions(r, r->scope == 0 ? r->cur : r->all, r->scope == 0 ? r->ncur : r->nall, r->scope == 1);
    else {
        set_sessions(r, NULL, 0, r->scope == 1);
        load_scope(r, r->scope);
    }
}

// The session at path made current, then the status.
static void switch_now(const char *path, bool stay, const char *status) {
    const char *err = uix_ops()->resume(uix_host(), path, stay);
    if (err) {
        uix_show_error(err);
        return;
    }
    after_switch();
    uix_show_status(status);
}

typedef struct {
    char *path, *status;
    bool stay;
} PendingSwitch;

static void switch_trusted(void *p) {
    PendingSwitch *w = p;
    switch_now(w->path, w->stay, w->status);
    free(w->path);
    free(w->status);
    free(w);
}

// switchSession with projectTrustContextFactory: the new folder's project trust asked first when
// it needs a decision (code_trust.c), never the previous folder's carried over.
static void switch_to(const char *path, bool stay, const char *status) {
    Arena a = {0};
    CodeSession *s;
    const char *cwd = NULL;
    if (stay) {
        char *here = getcwd(NULL, 0);
        if (here) cwd = arena_strdup(&a, here);
        free(here);
    } else if (!code_session_open_path(&a, path, &s))
        cwd = json_str(json_get(code_session_header(s), "cwd"));
    bool ask = cwd && *cwd && code_trust_switch_asks(cwd);
    if (ask) {
        PendingSwitch *w = xcalloc(1, sizeof *w);
        w->path = strdup(path);
        w->status = strdup(status);
        w->stay = stay;
        code_trust_switch_ask(cwd, switch_trusted, w);
    }
    arena_reset(&a);
    if (!ask) switch_now(path, stay, status);
}

// promptForMissingSessionCwd: an ExtensionSelectorComponent with getMissingSessionCwdOptions.
typedef struct {
    char *title, *path, *cwd;
    char *options[3];
    size_t sel;
} CwdSel;

static void cwd_free(void *p) {
    CwdSel *c = p;
    free(c->title);
    free(c->path);
    free(c->cwd);
    for (int i = 0; i < 3; i++) free(c->options[i]);
    free(c);
}

static void cwd_render(int width, Lines *out) {
    CwdSel *c = uix_selector_data();
    border(out, width, T_BORDER);
    lines_addz(out, "");
    Buf b = {0}, t = {0};
    th_bold(&t, str(c->title));
    th_fg(&b, T_ACCENT, buf_str(&t));
    text_lines(buf_str(&b), 1, width, out);
    lines_addz(out, "");
    for (size_t i = 0; i < 3; i++) {
        b.len = 0;
        if (i == c->sel) {
            th_fgz(&b, T_ACCENT, "\xe2\x86\x92 ");
            th_fgz(&b, T_ACCENT, c->options[i]);
        } else {
            buf_adds(&b, "  ");
            th_fgz(&b, T_TEXT, c->options[i]);
        }
        text_lines(buf_str(&b), 1, width, out);
    }
    lines_addz(out, "");
    b.len = 0;
    key_hint(&b, "\xe2\x86\x91\xe2\x86\x93", "navigate");
    buf_adds(&b, "  ");
    key_hint(&b, "enter", "select");
    buf_adds(&b, "  ");
    key_hint(&b, "escape/ctrl+c", "cancel");
    text_lines(buf_str(&b), 1, width, out);
    lines_addz(out, "");
    border(out, width, T_BORDER);
    buf_free(&b);
    buf_free(&t);
}

static void cwd_key(const Key *k) {
    CwdSel *c = uix_selector_data();
    if (k->kind == K_CTRL && k->ch == 'o') return; // the tool output toggle stays with the transcript
    if (k->kind == K_UP || (k->kind == K_CHAR && k->ch == 'k')) c->sel = c->sel ? c->sel - 1 : 0;
    else if (k->kind == K_DOWN || (k->kind == K_CHAR && k->ch == 'j')) c->sel = c->sel < 2 ? c->sel + 1 : 2;
    else if (k->kind == K_ENTER || is_cancel(k)) {
        int choice = k->kind == K_ENTER ? (int)c->sel : 2;
        char *path = strdup(c->path), *here = getcwd(NULL, 0);
        uix_close_selector();
        if (choice == 0) {
            Buf m = {0};
            buf_printf(&m, "Resumed session in %s", here ? here : "");
            switch_to(path, true, m.ptr);
            buf_free(&m);
        } else if (choice == 1) {
            // switchToForkedSession: SessionManager.forkFrom into this folder's sessions
            Arena a = {0};
            const CodeUiConfig *cfg = uix_cfg();
            char *fork = NULL;
            const char *err = code_sessions_fork_from(&a, path, here ? here : cfg->cwd, code_sessions_dir(&a, cfg->agent_dir, here ? here : cfg->cwd), &fork);
            if (err) uix_show_error(err);
            else {
                CodeSession *fs;
                Buf m = {0};
                if (!code_session_open_path(&a, fork, &fs))
                    buf_printf(&m, "Forked into new session %s in %s", json_str(json_get(code_session_header(fs), "id")) ?: "", here ? here : "");
                switch_to(fork, false, m.ptr ? m.ptr : "");
                buf_free(&m);
            }
            arena_reset(&a);
        } else uix_show_status("Resume cancelled");
        free(path);
        free(here);
    }
}

static void resume_selected(const char *path) {
    // assertSessionCwdExists: a stored folder that is gone asks what to do first
    Arena a = {0};
    CodeSession *s;
    const char *cwd = !code_session_open_path(&a, path, &s) ? json_str(json_get(code_session_header(s), "cwd")) : NULL;
    struct stat st;
    if (cwd && *cwd && stat(cwd, &st) != 0) {
        CwdSel *c = xcalloc(1, sizeof *c);
        char *here = getcwd(NULL, 0);
        Buf b = {0};
        buf_printf(&b, "Session cwd does not exist: %s", cwd);
        c->title = strdup(b.ptr);
        b.len = 0;
        buf_printf(&b, "Continue in %s (asks again next time)", here ? here : "");
        c->options[0] = strdup(b.ptr);
        b.len = 0;
        buf_printf(&b, "Fork into %s (new session ID, original kept)", here ? here : "");
        c->options[1] = strdup(b.ptr);
        c->options[2] = strdup("Cancel");
        c->path = strdup(path);
        c->cwd = strdup(cwd);
        buf_free(&b);
        free(here);
        arena_reset(&a);
        uix_open_selector(cwd_render, cwd_key, cwd_free, c);
        return;
    }
    arena_reset(&a);
    switch_to(path, false, "Resumed session");
}

static void start_delete(ResumeSel *r) {
    if (r->sel >= r->flat.len) return;
    const char *p = r->flat.ptr[r->sel].s->path;
    if (is_current(r, p)) {
        set_status(r, "Cannot delete the currently active session", true, 3000);
        return;
    }
    free(r->confirm);
    r->confirm = strdup(p);
}

static void resume_key(const Key *k) {
    ResumeSel *r = uix_selector_data();
    if (r->renaming) {
        if (is_cancel(k)) {
            r->renaming = false;
            free(r->rename_target);
            r->rename_target = NULL;
        } else if (k->kind == K_ENTER) {
            // confirmRename
            Str next = str_trim(inp_str(&r->rename));
            if (!next.len) return;
            char *name = strndup(next.ptr, next.len);
            if (r->rename_target) (void)code_sessions_rename_file(r->rename_target, name);
            free(name);
            if (r->rename_target) refresh_after_mutation(r);
            r->renaming = false;
            free(r->rename_target);
            r->rename_target = NULL;
        } else inp_key(&r->rename, k);
        return;
    }
    if (r->confirm) {
        if (k->kind == K_ENTER) {
            char *path = r->confirm;
            r->confirm = NULL;
            bool trashed;
            const char *err = delete_file(path, &trashed);
            if (!err) {
                size_t w = 0;
                for (size_t i = 0; i < r->ncur; i++)
                    if (strcmp(r->cur[i].path, path)) r->cur[w++] = r->cur[i];
                r->ncur = w;
                w = 0;
                for (size_t i = 0; i < r->nall; i++)
                    if (strcmp(r->all[i].path, path)) r->all[w++] = r->all[i];
                r->nall = w;
                set_sessions(r, r->scope == 1 ? r->all : r->cur, r->scope == 1 ? r->nall : r->ncur, r->scope == 1);
                set_status(r, trashed ? "Session moved to trash" : "Session deleted", false, 2000);
                refresh_after_mutation(r);
            } else {
                Buf m = {0};
                buf_printf(&m, "Failed to delete: %s", err);
                set_status(r, m.ptr, true, 3000);
                buf_free(&m);
            }
            free(path);
        } else if (is_cancel(k)) {
            free(r->confirm);
            r->confirm = NULL;
        }
        return;
    }
    if (k->kind == K_TAB) {
        toggle_scope(r);
        return;
    }
    if (k->kind == K_CTRL && k->ch == 's') {
        r->sort = (r->sort + 1) % 3;
        filter_sessions(r);
        return;
    }
    if (k->kind == K_CTRL && k->ch == 'n') {
        r->named = !r->named;
        filter_sessions(r);
        return;
    }
    if (k->kind == K_CTRL && k->ch == 'p') {
        r->show_path = !r->show_path;
        return;
    }
    if (k->kind == K_CTRL && k->ch == 'd') {
        start_delete(r);
        return;
    }
    if (k->kind == K_CTRL && k->ch == 'r') {
        if (r->sel >= r->flat.len) return;
        const CodeSessionInfo *s = r->flat.ptr[r->sel].s;
        r->renaming = true;
        free(r->rename_target);
        r->rename_target = strdup(s->path);
        inp_set(&r->rename, s->name ? s->name : "");
        return;
    }
    if (k->kind == K_CTRL_BACKSPACE) {
        if (r->search.v.len) {
            inp_key(&r->search, k);
            filter_sessions(r);
        } else start_delete(r);
        return;
    }
    r->touched = true;
    size_t n = r->flat.len;
    if (k->kind == K_UP) r->sel = r->sel > 0 ? r->sel - 1 : 0;
    else if (k->kind == K_DOWN) r->sel = n && r->sel + 1 < n ? r->sel + 1 : (n ? n - 1 : 0);
    else if (k->kind == K_PGUP) r->sel = r->sel > 10 ? r->sel - 10 : 0;
    else if (k->kind == K_PGDN) r->sel = n ? (r->sel + 10 < n ? r->sel + 10 : n - 1) : 0;
    else if (k->kind == K_ENTER) {
        if (r->sel >= n) return;
        char *path = strdup(r->flat.ptr[r->sel].s->path);
        uix_close_selector();
        resume_selected(path);
        free(path);
    } else if (is_cancel(k)) uix_close_selector();
    else if (inp_key(&r->search, k)) filter_sessions(r);
}

static void resume_command(void) {
    ResumeSel *r = xcalloc(1, sizeof *r);
    CodeSession *s = uix_ops()->session(uix_host());
    const char *file = code_session_path(s);
    r->current_key = file ? path_key(file) : NULL;
    inp_set(&r->search, "");
    inp_set(&r->rename, "");
    load_scope(r, 0);
    g_resume = r;
    uix_open_selector(resume_render, resume_key, resume_free, r);
}

// ---- /compact (handleCompactCommand) ------------------------------------------------------------------

static void compact_command(const char *instructions) { uix_start_compact(instructions); }

// ---- the commands ------------------------------------------------------------------------------------------

// The commands that change the session wait for the run in flight, which they stop first (pi's
// runtime tears the session down, compact() aborts it); /name and /session wait without stopping it.
static bool command_now(const char *c) {
    if (!strcmp(c, "name") || !strncmp(c, "name ", 5) || !strncmp(c, "name\t", 5)) name_command(c);
    else if (!strcmp(c, "session")) session_command();
    else if (!strcmp(c, "fork")) fork_command();
    else if (!strcmp(c, "clone")) clone_command();
    else if (!strcmp(c, "compact")) compact_command(NULL);
    else if (!strncmp(c, "compact ", 8)) {
        Str t = str_trim(str(c + 8));
        char *i = strndup(t.ptr, t.len);
        compact_command(i);
        free(i);
    } else if (!strcmp(c, "resume")) resume_command();
    else return false;
    return true;
}

static bool is_command(const char *c) {
    return !strcmp(c, "name") || !strncmp(c, "name ", 5) || !strncmp(c, "name\t", 5) || !strcmp(c, "session") || !strcmp(c, "fork") ||
           !strcmp(c, "clone") || !strcmp(c, "compact") || !strncmp(c, "compact ", 8) || !strcmp(c, "resume");
}

bool sess_ui_command(const char *c) {
    if (!is_command(c)) return false;
    if (!uix_ops()->session || !uix_ops()->fork || !uix_ops()->resume || !uix_ops()->compact) return false;
    if (uix_busy()) {
        vec_push(&G.deferred, strdup(c));
        bool waits = !strcmp(c, "name") || !strncmp(c, "name", 4) || !strcmp(c, "session");
        if (!waits) uix_abort();
        return true;
    }
    return command_now(c);
}

bool sess_ui_idle(void) {
    while (G.deferred.len && !uix_busy()) {
        char *c = G.deferred.ptr[0];
        memmove(G.deferred.ptr, G.deferred.ptr + 1, (G.deferred.len - 1) * sizeof *G.deferred.ptr);
        G.deferred.len--;
        command_now(c);
        free(c);
    }
    return uix_busy();
}

int sess_ui_tick(uint64_t now) {
    ResumeSel *r = g_resume;
    if (!r || !r->status || !r->status_until) return -1;
    if (now >= r->status_until) {
        set_status(r, NULL, false, 0);
        uix_request_render();
        return -1;
    }
    return (int)((r->status_until - now) / 1000000) + 1;
}
