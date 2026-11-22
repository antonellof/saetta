// Saetta Code's session exports (see code_cmds.h): the HTML page /export and /share write, and the
// JSONL branch /export writes for a .jsonl path.
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at v1.5.2,
// b4afddd): packages/coding-agent/src/core/export-html/index.ts (generateThemeVars,
// deriveExportColors, generateHtml, exportSessionToHtml), core/session-export.ts
// (serializeSessionBranch, exportSessionToJsonl), utils/paths.ts (normalizePath, resolvePath),
// modes/interactive/theme/theme.ts (getResolvedThemeColors, getThemeExportColors, Theme.colors),
// packages/tui/src/colors.ts and oklab.ts (colorToHex, indexedToRgb, mixColors, the OKLCH
// conversions). The page itself is code_export_tpl.c.
#include "code_cmds.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

extern const char *const code_export_tpl[6];
extern const size_t code_export_tpl_len[6];

// ---- paths (utils/paths.ts) ------------------------------------------------------------------------

// path.resolve for POSIX: base joined with p unless p is absolute, "." and ".." folded, no trailing
// slash.
static char *posix_resolve(const char *base, const char *p) {
    Buf in = {0};
    if (p[0] != '/') {
        buf_adds(&in, base);
        buf_addc(&in, '/');
    }
    buf_adds(&in, p);
    VEC(Str) parts = {0};
    Str s = buf_str(&in);
    for (size_t i = 0; i < s.len;) {
        while (i < s.len && s.ptr[i] == '/') i++;
        size_t j = i;
        while (j < s.len && s.ptr[j] != '/') j++;
        Str seg = str_n(s.ptr + i, j - i);
        if (seg.len == 0 || str_eqz(seg, ".")) {
        } else if (str_eqz(seg, "..")) {
            if (parts.len) parts.len--;
        } else vec_push(&parts, seg);
        i = j;
    }
    Buf out = {0};
    for (size_t i = 0; i < parts.len; i++) {
        buf_addc(&out, '/');
        buf_addstr(&out, parts.ptr[i]);
    }
    if (!out.len) buf_addc(&out, '/');
    vec_free(&parts);
    buf_free(&in);
    return out.ptr;
}

// normalizePath: a leading ~ expanded (malloc'd).
static char *normalize_path(const char *input, const char *home) {
    Buf b = {0};
    if (!strcmp(input, "~")) buf_adds(&b, home);
    else if (!strncmp(input, "~/", 2)) {
        buf_adds(&b, home);
        if (b.len && b.ptr[b.len - 1] != '/') buf_addc(&b, '/');
        buf_adds(&b, input + 2);
    } else buf_adds(&b, input);
    return b.ptr ? b.ptr : strdup("");
}

char *cmds_resolve_path(const char *input, const char *base, const char *home) {
    char *n = normalize_path(input, home);
    char *nb = normalize_path(base, home);
    char *r = posix_resolve(nb, n);
    free(n);
    free(nb);
    return r;
}

// Node's message for a failed open or write ("ENOENT: no such file or directory, open 'x'").
static char *node_error(int e, const char *syscall, const char *path) {
    const char *code = e == ENOENT ? "ENOENT" : e == EACCES ? "EACCES" : e == EISDIR ? "EISDIR" : e == ENOTDIR ? "ENOTDIR"
                       : e == EEXIST ? "EEXIST" : e == ENOSPC ? "ENOSPC" : e == EROFS ? "EROFS" : "EIO";
    const char *what = e == ENOENT ? "no such file or directory" : e == EACCES ? "permission denied"
                       : e == EISDIR ? "illegal operation on a directory" : e == ENOTDIR ? "not a directory"
                       : e == EEXIST ? "file already exists" : e == ENOSPC ? "no space left on device"
                       : e == EROFS ? "read-only file system" : strerror(e);
    Buf b = {0};
    buf_printf(&b, "%s: %s, %s '%s'", code, what, syscall, path);
    return b.ptr;
}

// writeFileSync(path, data): created or truncated, 0666 under the umask.
static bool write_whole(const char *path, const char *shown, Str data, char **err) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
    if (fd < 0) {
        *err = node_error(errno, "open", shown);
        return false;
    }
    for (size_t off = 0; off < data.len;) {
        ssize_t w = write(fd, data.ptr + off, data.len - off);
        if (w < 0 && errno == EINTR) continue;
        if (w < 0) {
            *err = node_error(errno, "write", shown);
            close(fd);
            return false;
        }
        off += (size_t)w;
    }
    close(fd);
    return true;
}

// ---- colors (pi-tui colors.ts, oklab.ts) --------------------------------------------------------------

typedef struct {
    double r, g, b;
} Rgb;

static const Rgb BASIC_COLORS[16] = {{0, 0, 0},       {128, 0, 0},   {0, 128, 0},   {128, 128, 0}, {0, 0, 128},   {128, 0, 128},
                                     {0, 128, 128},   {192, 192, 192}, {128, 128, 128}, {255, 0, 0}, {0, 255, 0},   {255, 255, 0},
                                     {0, 0, 255},     {255, 0, 255}, {0, 255, 255}, {255, 255, 255}};

// indexedToRgb
static Rgb indexed_rgb(int index) {
    static const double cube[6] = {0, 95, 135, 175, 215, 255};
    if (index < 16) return BASIC_COLORS[index];
    if (index < 232) {
        int c = index - 16;
        return (Rgb){cube[c / 36], cube[(c % 36) / 6], cube[c % 6]};
    }
    double g = 8 + (index - 232) * 10;
    return (Rgb){g, g, g};
}

static const double LINEAR_SRGB_TO_LMS[3][3] = {
    {0.4122214694707629, 0.5363325372617349, 0.0514459932675022},
    {0.2119034958178251, 0.6806995506452344, 0.1073969535369405},
    {0.0883024591900564, 0.2817188391361215, 0.6299787016738222},
};
static const double LMS_TO_LAB[3][3] = {
    {0.210454268309314, 0.793617774702305, -0.0040720430116193},
    {1.9779985324311684, -2.42859224204858, 0.450593709617411},
    {0.0259040424655478, 0.7827717124575296, -0.8086757549230774},
};
static const double LAB_TO_LMS[3][3] = {
    {1, 0.3963377773761749, 0.2158037573099136},
    {1, -0.1055613458156586, -0.0638541728258133},
    {1, -0.0894841775298119, -1.2914855480194092},
};
static const double LMS_TO_LINEAR_SRGB[3][3] = {
    {4.0767416360759583, -3.3077115392580629, 0.2309699031821043},
    {-1.2684379732850315, 2.6097573492876882, -0.341319376002657},
    {-0.0041960761386756, -0.7034186179359362, 1.7076146940746117},
};

static void mul3(const double m[3][3], const double v[3], double out[3]) {
    for (int i = 0; i < 3; i++) out[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
}

static double js_round(double x) {
    double r = floor(x);
    return x - r >= 0.5 ? r + 1 : r;
}

// V8's Math.hypot for two arguments.
static double js_hypot(double x, double y) {
    double ax = fabs(x), ay = fabs(y), max = ax > ay ? ax : ay;
    if (max == 0) return 0;
    double sum = 0, comp = 0, v[2] = {ax, ay};
    for (int i = 0; i < 2; i++) {
        double n = v[i] / max, summand = n * n - comp, pre = sum + summand;
        comp = (pre - sum) - summand;
        sum = pre;
    }
    return sqrt(sum) * max;
}

static double srgb_to_linear(double v) { return v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4); }
static double linear_to_srgb(double v) { return v > 0.0031308 ? 1.055 * pow(v, 1 / 2.4) - 0.055 : 12.92 * v; }

static void rgb_to_oklab(Rgb c, double lab[3]) {
    double lin[3] = {srgb_to_linear(c.r / 255), srgb_to_linear(c.g / 255), srgb_to_linear(c.b / 255)}, lms[3];
    mul3(LINEAR_SRGB_TO_LMS, lin, lms);
    for (int i = 0; i < 3; i++) lms[i] = cbrt(lms[i]);
    mul3(LMS_TO_LAB, lms, lab);
}

static void oklab_to_linear(const double lab[3], double out[3]) {
    double lms[3];
    mul3(LAB_TO_LMS, lab, lms);
    for (int i = 0; i < 3; i++) lms[i] = lms[i] * lms[i] * lms[i];
    mul3(LMS_TO_LINEAR_SRGB, lms, out);
}

static Rgb linear_to_rgb(const double lin[3]) {
    double c[3];
    for (int i = 0; i < 3; i++) c[i] = js_round(fmin(1, fmax(0, linear_to_srgb(lin[i]))) * 255);
    return (Rgb){c[0], c[1], c[2]};
}

static bool in_gamut(const double lin[3]) {
    const double eps = 1e-7;
    for (int i = 0; i < 3; i++)
        if (lin[i] < -eps || lin[i] > 1 + eps) return false;
    return true;
}

// oklchToRgb: the chroma reduced (bisection) until the color fits sRGB.
static Rgb oklch_to_rgb(double l, double c, double h) {
    double rad = h * M_PI / 180, cs = cos(rad), sn = sin(rad);
    double lab[3] = {l, c * cs, c * sn}, lin[3];
    oklab_to_linear(lab, lin);
    if (in_gamut(lin)) return linear_to_rgb(lin);
    double best[3];
    double lab0[3] = {l, 0, 0};
    oklab_to_linear(lab0, best);
    double low = 0, high = c;
    for (int i = 0; i < 20; i++) {
        double ch = (low + high) / 2;
        double labc[3] = {l, ch * cs, ch * sn}, cand[3];
        oklab_to_linear(labc, cand);
        if (in_gamut(cand)) {
            low = ch;
            memcpy(best, cand, sizeof best);
        } else high = ch;
    }
    return linear_to_rgb(best);
}

// mixColors(first, second, amount, "oklch")
static Rgb mix_oklch(Rgb x, Rgb y, double amount) {
    double a[3], b[3];
    rgb_to_oklab(x, a);
    rgb_to_oklab(y, b);
    double ac = js_hypot(a[1], a[2]), ah = fmod((atan2(a[2], a[1]) * 180) / M_PI + 360, 360);
    double bc = js_hypot(b[1], b[2]), bh = fmod((atan2(b[2], b[1]) * 180) / M_PI + 360, 360);
    double first = ac < 1e-7 ? bh : ah, second = bc < 1e-7 ? first : bh;
    double delta = fmod(second - first + 540, 360) - 180;
    return oklch_to_rgb(a[0] + (b[0] - a[0]) * amount, ac + (bc - ac) * amount, first + delta * amount);
}

static void hex(Buf *b, Rgb c) { buf_printf(b, "#%02x%02x%02x", (int)js_round(c.r), (int)js_round(c.g), (int)js_round(c.b)); }

// ---- the theme's colors (theme.ts getResolvedThemeColors) ----------------------------------------------

// A token's value, read back from its SGR sequence as pi's Theme holds it: a color, or "" (the
// terminal's default), and whether it is drawn faint.
typedef struct {
    bool is_default, dim;
    Rgb rgb;
} TokVal;

static TokVal sgr_value(const char *open, bool bg) {
    TokVal v = {.is_default = true};
    const char *p = open;
    while ((p = strstr(p, "\x1b["))) {
        p += 2;
        int params[16], n = 0;
        while (n < 16) {
            char *e;
            long x = strtol(p, &e, 10);
            params[n++] = (int)x;
            p = e;
            if (*p != ';') break;
            p++;
        }
        for (int i = 0; i < n; i++) {
            int x = params[i];
            int ext = bg ? 48 : 38;
            if (x == ext && i + 1 < n && params[i + 1] == 2 && i + 4 < n) {
                v = (TokVal){.rgb = {params[i + 2], params[i + 3], params[i + 4]}, .dim = v.dim};
                i += 4;
            } else if (x == ext && i + 1 < n && params[i + 1] == 5 && i + 2 < n) {
                v = (TokVal){.rgb = indexed_rgb(params[i + 2]), .dim = v.dim};
                i += 2;
            } else if (!bg && x >= 30 && x <= 37) v = (TokVal){.rgb = BASIC_COLORS[x - 30], .dim = v.dim};
            else if (!bg && x >= 90 && x <= 97) v = (TokVal){.rgb = BASIC_COLORS[x - 90 + 8], .dim = v.dim};
            else if (bg && x >= 40 && x <= 47) v = (TokVal){.rgb = BASIC_COLORS[x - 40], .dim = v.dim};
            else if (bg && x >= 100 && x <= 107) v = (TokVal){.rgb = BASIC_COLORS[x - 100 + 8], .dim = v.dim};
            else if (!bg && x == 2) v.dim = true;
        }
    }
    return v;
}

// The token names (theme-tokens.ts), by ThemeFg and ThemeBgToken.
static const char *const FG_NAMES[THEME_NFG] = {
    "accent", "border", "borderAccent", "borderMuted", "success", "error", "warning", "muted", "dim", "text", "thinkingText",
    "scrollbarTrack", "scrollbarThumb", "searchMatchText", "userMessageText", "customMessageText", "customMessageLabel",
    "toolTitle", "toolOutput", "mdHeading", "mdLink", "mdLinkUrl", "mdCode", "mdCodeBlock", "mdCodeBlockBorder", "mdQuote",
    "mdQuoteBorder", "mdHr", "mdListBullet", "toolDiffAdded", "toolDiffRemoved", "toolDiffContext", "syntaxComment",
    "syntaxKeyword", "syntaxFunction", "syntaxVariable", "syntaxString", "syntaxNumber", "syntaxType", "syntaxOperator",
    "syntaxPunctuation", "thinkingOff", "thinkingMinimal", "thinkingLow", "thinkingMedium", "thinkingHigh", "thinkingXhigh",
    "thinkingMax", "bashMode"};
static const char *const BG_NAMES[THEME_NBG] = {"selectedBg",    "searchMatchBg", "userMessageBg", "customMessageBg",
                                                "toolPendingBg", "toolSuccessBg", "toolErrorBg"};

// The order pi's Theme holds the tokens in: the system theme's (TOKEN_FAMILIES in system-theme.ts),
// and the built-in themes' (dark.json and light.json), as names.
static const char *const FAMILY_ORDER[] = {
    "selectedBg", "searchMatchBg", "userMessageBg", "customMessageBg", "toolPendingBg", "toolSuccessBg", "toolErrorBg", "text",
    "userMessageText", "customMessageText", "toolTitle", "syntaxOperator", "syntaxPunctuation", "muted", "dim", "thinkingText",
    "toolOutput", "mdLinkUrl", "mdQuote", "mdQuoteBorder", "mdHr", "mdCodeBlockBorder", "toolDiffContext", "syntaxComment",
    "scrollbarTrack", "scrollbarThumb", "searchMatchText", "borderMuted", "accent", "borderAccent", "customMessageLabel",
    "mdCode", "mdListBullet", "syntaxType", "border", "mdLink", "syntaxKeyword", "syntaxVariable", "success", "mdCodeBlock",
    "toolDiffAdded", "bashMode", "syntaxNumber", "error", "toolDiffRemoved", "warning", "mdHeading", "syntaxFunction",
    "syntaxString", "thinkingOff", "thinkingMinimal", "thinkingLow", "thinkingMedium", "thinkingHigh", "thinkingXhigh",
    "thinkingMax"};
static const char *const JSON_ORDER[] = {
    "accent", "border", "borderAccent", "borderMuted", "success", "error", "warning", "muted", "dim", "text", "thinkingText",
    "selectedBg", "scrollbarTrack", "scrollbarThumb", "searchMatchBg", "searchMatchText", "userMessageBg", "userMessageText",
    "customMessageBg", "customMessageText", "customMessageLabel", "toolPendingBg", "toolSuccessBg", "toolErrorBg", "toolTitle",
    "toolOutput", "mdHeading", "mdLink", "mdLinkUrl", "mdCode", "mdCodeBlock", "mdCodeBlockBorder", "mdQuote", "mdQuoteBorder",
    "mdHr", "mdListBullet", "toolDiffAdded", "toolDiffRemoved", "toolDiffContext", "syntaxComment", "syntaxKeyword",
    "syntaxFunction", "syntaxVariable", "syntaxString", "syntaxNumber", "syntaxType", "syntaxOperator", "syntaxPunctuation",
    "thinkingOff", "thinkingMinimal", "thinkingLow", "thinkingMedium", "thinkingHigh", "thinkingXhigh", "thinkingMax",
    "bashMode"};

typedef struct {
    const char *name;
    Rgb rgb;
} NamedColor;

// getResolvedThemeColors: every token's color, in the Theme's order (the concrete colors, then the
// tokens on the terminal's default colors: its reported ones, else pi's guess for the appearance);
// faint tokens mixed 40% toward the background. Returns the count; *user_bg gets userMessageBg.
static size_t resolved_colors(const CmdsTheme *t, NamedColor out[THEME_NFG + THEME_NBG], Rgb *user_bg) {
    ThemeSgr fg[THEME_NFG], bg[THEME_NBG];
    bool light = false;
    const char *name = t->name ? t->name : "system";
    if (!theme_named(name, t->colors, TC_TRUECOLOR, t->appearance, fg, bg, &light)) {
        name = "system";
        theme_named(name, t->colors, TC_TRUECOLOR, t->appearance, fg, bg, &light);
    }
    bool json = !strcmp(name, "dark") || !strcmp(name, "light");
    const char *const *order = json ? JSON_ORDER : FAMILY_ORDER;
    size_t norder = json ? COUNT(JSON_ORDER) : COUNT(FAMILY_ORDER);
    Rgb dfg = light ? (Rgb){0, 0, 0} : (Rgb){0xe5, 0xe5, 0xe7}, dbg = light ? (Rgb){255, 255, 255} : (Rgb){0, 0, 0};
    if (t->colors && t->colors->has_fg) dfg = (Rgb){t->colors->fg[0], t->colors->fg[1], t->colors->fg[2]};
    if (t->colors && t->colors->has_bg) dbg = (Rgb){t->colors->bg[0], t->colors->bg[1], t->colors->bg[2]};
    // the Theme's insertion order: the foregrounds then the backgrounds, concrete ones first
    TokVal vals[THEME_NFG + THEME_NBG];
    int slot[THEME_NFG + THEME_NBG];
    size_t n = 0;
    for (int pass = 0; pass < 4; pass++) { // concrete fg, concrete bg, default fg, default bg
        bool want_bg = pass & 1, want_default = pass >= 2;
        for (size_t i = 0; i < norder; i++) {
            int k = -1;
            bool isbg = false;
            for (int f = 0; f < THEME_NFG && k < 0; f++)
                if (!strcmp(order[i], FG_NAMES[f])) k = f;
            for (int b = 0; b < THEME_NBG && k < 0; b++)
                if (!strcmp(order[i], BG_NAMES[b])) k = b, isbg = true;
            if (k < 0 || isbg != want_bg) continue;
            TokVal v = sgr_value(isbg ? bg[k].open : fg[k].open, isbg);
            if (v.is_default != want_default) continue;
            if (v.is_default) v.rgb = isbg ? dbg : dfg;
            vals[n] = v;
            slot[n] = isbg ? THEME_NFG + k : k;
            out[n].name = order[i];
            n++;
        }
    }
    for (size_t i = 0; i < n; i++) {
        out[i].rgb = vals[i].dim && slot[i] < THEME_NFG ? mix_oklch(vals[i].rgb, dbg, 0.4) : vals[i].rgb;
        if (slot[i] == THEME_NFG + TH_BG_USER_MESSAGE) *user_bg = out[i].rgb;
    }
    return n;
}

static double luminance(Rgb c) {
    double v[3] = {c.r, c.g, c.b};
    for (int i = 0; i < 3; i++) {
        double s = v[i] / 255;
        v[i] = s <= 0.03928 ? s / 12.92 : pow((s + 0.055) / 1.055, 2.4);
    }
    return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
}

static void adjust(Buf *b, Rgb c, double f) {
    double v[3] = {c.r, c.g, c.b};
    for (int i = 0; i < 3; i++) v[i] = fmin(255, fmax(0, js_round(v[i] * f)));
    buf_printf(b, "rgb(%d, %d, %d)", (int)v[0], (int)v[1], (int)v[2]);
}

// deriveExportColors (the base color is userMessageBg), or the theme's own export colors
// (getThemeExportColors: dark.json and light.json declare them).
static void export_colors(const char *name, Rgb base, Buf *page, Buf *card, Buf *info) {
    if (name && !strcmp(name, "dark")) {
        buf_adds(page, "#21252c");
        buf_adds(card, "#282c34");
        buf_adds(info, "#4e2f1b");
        return;
    }
    if (name && !strcmp(name, "light")) {
        buf_adds(page, "#efeeee");
        buf_adds(card, "#f7f6f6");
        buf_adds(info, "#ede3dd");
        return;
    }
    if (luminance(base) > 0.5) {
        adjust(page, base, 0.96);
        hex(card, base);
        buf_printf(info, "rgb(%d, %d, %d)", (int)fmin(255, base.r + 10), (int)fmin(255, base.g + 5), (int)fmax(0, base.b - 20));
    } else {
        adjust(page, base, 0.7);
        adjust(card, base, 0.85);
        buf_printf(info, "rgb(%d, %d, %d)", (int)fmin(255, base.r + 20), (int)fmin(255, base.g + 15), (int)base.b);
    }
}

// ---- the page (generateHtml) -------------------------------------------------------------------------

static void base64(Buf *out, Str in) {
    static const char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    buf_reserve(out, (in.len + 2) / 3 * 4 + 1);
    for (size_t i = 0; i < in.len; i += 3) {
        uint32_t v = (uint32_t)(uint8_t)in.ptr[i] << 16;
        if (i + 1 < in.len) v |= (uint32_t)(uint8_t)in.ptr[i + 1] << 8;
        if (i + 2 < in.len) v |= (uint8_t)in.ptr[i + 2];
        buf_addc(out, tbl[v >> 18 & 63]);
        buf_addc(out, tbl[v >> 12 & 63]);
        buf_addc(out, i + 1 < in.len ? tbl[v >> 6 & 63] : '=');
        buf_addc(out, i + 2 < in.len ? tbl[v & 63] : '=');
    }
}

// JSON.stringify(sessionData): header, entries (without the header), leafId, then systemPrompt and
// tools when known (an undefined member is left out).
static void session_data(Buf *b, const CmdsSession *s) {
    const JValue *all = s->entries;
    buf_adds(b, "{\"header\":");
    if (all && all->arr.len) json_write(b, all->arr.items[0]);
    else buf_adds(b, "null");
    buf_adds(b, ",\"entries\":[");
    for (size_t i = 1; all && i < all->arr.len; i++) {
        if (i > 1) buf_addc(b, ',');
        json_write(b, all->arr.items[i]);
    }
    buf_adds(b, "],\"leafId\":");
    if (s->leaf) json_write_string(b, str(s->leaf));
    else buf_adds(b, "null");
    if (s->system_prompt) {
        buf_adds(b, ",\"systemPrompt\":");
        json_write_string(b, str(s->system_prompt));
    }
    if (s->tools) {
        buf_adds(b, ",\"tools\":");
        json_write(b, s->tools);
    }
    buf_addc(b, '}');
}

static void generate_html(Buf *html, const CmdsSession *s, const CmdsTheme *t) {
    NamedColor colors[THEME_NFG + THEME_NBG];
    Rgb user_bg = {0x34, 0x35, 0x41};
    size_t n = resolved_colors(t, colors, &user_bg);
    Buf vars = {0}, page = {0}, card = {0}, info = {0};
    for (size_t i = 0; i < n; i++) {
        buf_printf(&vars, "--%s: ", colors[i].name);
        hex(&vars, colors[i].rgb);
        buf_adds(&vars, ";\n      ");
    }
    export_colors(t->name, user_bg, &page, &card, &info);
    buf_printf(&vars, "--exportPageBg: %s;\n      --exportCardBg: %s;\n      --exportInfoBg: %s;", page.ptr, card.ptr, info.ptr);
    Buf json = {0}, b64 = {0};
    session_data(&json, s);
    base64(&b64, buf_str(&json));
    Str dyn[5] = {buf_str(&vars), buf_str(&page), buf_str(&card), buf_str(&info), buf_str(&b64)};
    for (int i = 0; i < 6; i++) {
        buf_add(html, code_export_tpl[i], code_export_tpl_len[i]);
        if (i < 5) buf_addstr(html, dyn[i]);
    }
    buf_free(&vars);
    buf_free(&page);
    buf_free(&card);
    buf_free(&info);
    buf_free(&json);
    buf_free(&b64);
}

bool cmds_export_html(const CmdsSession *s, const CmdsTheme *t, const char *out_path, const char *cwd, char **written,
                      char **err) {
    *written = *err = NULL;
    if (!s->session_file) {
        *err = strdup("Cannot export in-memory session to HTML");
        return false;
    }
    if (access(s->session_file, F_OK) != 0) {
        *err = strdup("Nothing to export yet - start a conversation first");
        return false;
    }
    const char *home = getenv("HOME") ? getenv("HOME") : "/";
    char *path;
    if (out_path && *out_path) path = normalize_path(out_path, home);
    else {
        const char *base = strrchr(s->session_file, '/');
        base = base ? base + 1 : s->session_file;
        size_t bl = strlen(base);
        if (bl >= 6 && !strcmp(base + bl - 6, ".jsonl")) bl -= 6;
        Buf p = {0};
        buf_printf(&p, "pi-session-%.*s.html", (int)bl, base);
        path = p.ptr;
    }
    Buf html = {0};
    generate_html(&html, s, t);
    // a relative path is the process's working directory's (writeFileSync)
    char *abs = path[0] == '/' ? strdup(path) : posix_resolve(cwd, path);
    bool ok = write_whole(abs, path, buf_str(&html), err); // Node names the path as given
    free(abs);
    buf_free(&html);
    if (ok) *written = path;
    else free(path);
    return ok;
}

// new Date().toISOString()
static void iso_now(Buf *b) {
    uint64_t ms = wall_ns() / 1000000;
    time_t sec = (time_t)(ms / 1000);
    struct tm tm;
    gmtime_r(&sec, &tm);
    buf_printf(b, "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min,
               tm.tm_sec, (int)(ms % 1000));
}

// serializeSessionBranch: a fresh header (this session's id and cwd, the time now) and the entries
// from the root to the leaf, each with its parentId set to the entry before it.
static void serialize_branch(Buf *out, const CmdsSession *s) {
    const JValue *all = s->entries;
    const JValue *h = all && all->arr.len ? all->arr.items[0] : NULL;
    Buf ts = {0};
    iso_now(&ts);
    buf_adds(out, "{\"type\":\"session\",\"version\":3,\"id\":");
    json_write_string(out, json_strs(json_get(h, "id")));
    buf_adds(out, ",\"timestamp\":");
    json_write_string(out, buf_str(&ts));
    buf_adds(out, ",\"cwd\":");
    json_write_string(out, json_strs(json_get(h, "cwd")));
    buf_adds(out, "}\n");
    buf_free(&ts);
    // the branch: up from the leaf, then reversed
    VEC(const JValue *) up = {0};
    for (const char *id = s->leaf; id && up.len <= (all ? all->arr.len : 0);) {
        const JValue *found = NULL;
        for (size_t i = all->arr.len; i-- > 1 && !found;)
            if (str_eqz(json_strs(json_get(all->arr.items[i], "id")), id)) found = all->arr.items[i];
        if (!found) break;
        vec_push(&up, found);
        const JValue *p = json_get(found, "parentId");
        id = json_is(p, J_STRING) ? p->s.ptr : NULL;
    }
    const char *parent = NULL;
    for (size_t k = up.len; k-- > 0;) {
        const JValue *e = up.ptr[k];
        // { ...entry, parentId }: the members in order, parentId replaced where it stands
        bool had = false;
        buf_addc(out, '{');
        for (size_t m = 0; m < e->obj.len; m++) {
            if (m) buf_addc(out, ',');
            json_write_string(out, e->obj.items[m].key);
            buf_addc(out, ':');
            if (str_eqz(e->obj.items[m].key, "parentId")) {
                had = true;
                if (parent) json_write_string(out, str(parent));
                else buf_adds(out, "null");
            } else json_write(out, e->obj.items[m].val);
        }
        if (!had) {
            if (e->obj.len) buf_addc(out, ',');
            buf_adds(out, "\"parentId\":");
            if (parent) json_write_string(out, str(parent));
            else buf_adds(out, "null");
        }
        buf_adds(out, "}\n");
        parent = json_str(json_get(e, "id"));
    }
    vec_free(&up);
}

bool cmds_export_jsonl(const CmdsSession *s, const char *out_path, const char *cwd, char **written, char **err) {
    *written = *err = NULL;
    const char *home = getenv("HOME") ? getenv("HOME") : "/";
    char *path;
    if (out_path && *out_path) path = cmds_resolve_path(out_path, cwd, home);
    else {
        Buf ts = {0}, p = {0};
        iso_now(&ts);
        for (size_t i = 0; i < ts.len; i++)
            if (ts.ptr[i] == ':' || ts.ptr[i] == '.') ts.ptr[i] = '-';
        buf_printf(&p, "session-%s.jsonl", ts.ptr);
        path = cmds_resolve_path(p.ptr, cwd, home);
        buf_free(&ts);
        buf_free(&p);
    }
    char *slash = strrchr(path, '/');
    if (slash && slash != path) {
        *slash = 0;
        Err e = mkdir_p(path);
        if (e) {
            *err = node_error(errno ? errno : EIO, "mkdir", path);
            *slash = '/';
            free(path);
            return false;
        }
        *slash = '/';
    }
    Buf text = {0};
    serialize_branch(&text, s);
    bool ok = write_whole(path, path, buf_str(&text), err);
    buf_free(&text);
    if (ok) *written = path;
    else free(path);
    return ok;
}
