// Saetta Code TUI: pi's "system" theme (see code_tui_theme.h).
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at v1.5.2,
// b4afddd): packages/coding-agent/src/modes/interactive/theme/system-theme.ts, theme.ts;
// packages/tui/src/oklab.ts, colors.ts. oklab.ts is itself a port of Bjorn Ottosson's OKHSL
// reference implementation (MIT, Copyright (c) 2021 Bjorn Ottosson).
//
// The generator solves every token's OKLab lightness from contrast curves fitted per level and
// appearance, on the background and on the panels solved before it, relaxing the levels on mid-gray
// backgrounds where they cannot all be reached; it paints each token in OKHSL with its family's hue
// (or its palette color's, capped at that color's chroma), keeps panels readable under white or
// black text and body text at WCAG 4.5:1, and uses the terminal's own foreground for body text when
// it is strong enough. Each expression keeps JavaScript's evaluation order: x ** y is pow, a + b + c
// is left to right, Math.round rounds half up, and Math.hypot is V8's (scaled, Kahan summed).
#include "code_tui_theme.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---- JavaScript numerics ----------------------------------------------------------------------------

// Math.round: the nearest integer, halves toward +infinity.
static double js_round(double x) {
    double r = floor(x);
    return x - r >= 0.5 ? r + 1 : r;
}

// V8's Math.hypot for two arguments (builtins/math.tq MathHypot).
static double js_hypot(double x, double y) {
    double ax = fabs(x), ay = fabs(y);
    if (isnan(x) || isnan(y)) return isinf(ax) || isinf(ay) ? INFINITY : NAN;
    double max = ax > ay ? ax : ay;
    if (max == INFINITY) return INFINITY;
    if (max == 0) return 0;
    double sum = 0, comp = 0, v[2] = {ax, ay};
    for (int i = 0; i < 2; i++) {
        double n = v[i] / max;
        double summand = n * n - comp;
        double pre = sum + summand;
        comp = (pre - sum) - summand;
        sum = pre;
    }
    return sqrt(sum) * max;
}

static double js_max(double a, double b) { return isnan(a) || isnan(b) ? NAN : a > b ? a : b; }
static double js_min(double a, double b) { return isnan(a) || isnan(b) ? NAN : a < b ? a : b; }
static double js_max3(const double v[3]) { return js_max(js_max(v[0], v[1]), v[2]); }

// ---- OKLab and OKHSL (pi-tui oklab.ts) ------------------------------------------------------------

typedef struct {
    double r, g, b; // 0-255, integers
} Rgb;

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
static const double SATURATION_FIT_DIR[3][2] = {
    {-1.8817031, -0.80936501}, {1.8144408, -1.19445267}, {0.13110758, 1.81333971}};
static const double SATURATION_FIT_K[3][5] = {
    {1.19086277, 1.76576728, 0.59662641, 0.75515197, 0.56771245},
    {0.73956515, -0.45954404, 0.08285427, 0.12541073, -0.14503204},
    {1.35733652, -0.00915799, -1.1513021, -0.50559606, 0.00692167},
};
#define K1 0.206
#define K2 0.03
#define K3 ((1 + K1) / (1 + K2))

static void multiply(const double m[3][3], const double v[3], double out[3]) {
    double x = v[0], y = v[1], z = v[2];
    for (int i = 0; i < 3; i++) out[i] = m[i][0] * x + m[i][1] * y + m[i][2] * z;
}

static double oklab_to_okhsl_l(double x) {
    return 0.5 * (K3 * x - K1 + sqrt(pow(K3 * x - K1, 2) + 4 * K2 * K3 * x));
}
static double okhsl_to_oklab_l(double x) { return (x * x + K1 * x) / (K3 * (x + K2)); }

static double linear_to_srgb(double v) { return v > 0.0031308 ? 1.055 * pow(v, 1 / 2.4) - 0.055 : 12.92 * v; }
static double srgb_to_linear(double v) { return v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4); }

static void oklab_to_linear(const double lab[3], double out[3]) {
    double lms[3];
    multiply(LAB_TO_LMS, lab, lms);
    for (int i = 0; i < 3; i++) lms[i] = pow(lms[i], 3);
    multiply(LMS_TO_LINEAR_SRGB, lms, out);
}

static void rgb_to_oklab(Rgb c, double out[3]) {
    double lin[3] = {srgb_to_linear(c.r / 255), srgb_to_linear(c.g / 255), srgb_to_linear(c.b / 255)}, lms[3];
    multiply(LINEAR_SRGB_TO_LMS, lin, lms);
    for (int i = 0; i < 3; i++) lms[i] = cbrt(lms[i]);
    multiply(LMS_TO_LAB, lms, out);
}

static double to_channel(double v) { return js_round(js_min(1, js_max(0, linear_to_srgb(v))) * 255); }

static Rgb linear_to_rgb(const double lin[3]) { return (Rgb){to_channel(lin[0]), to_channel(lin[1]), to_channel(lin[2])}; }

static void lms_slopes(double a, double b, double out[3]) {
    for (int i = 0; i < 3; i++) out[i] = LAB_TO_LMS[i][1] * a + LAB_TO_LMS[i][2] * b;
}

static double max_saturation(double a, double b) {
    int ch = 2;
    for (int i = 0; i < 2; i++)
        if (SATURATION_FIT_DIR[i][0] * a + SATURATION_FIT_DIR[i][1] * b > 1) {
            ch = i;
            break;
        }
    const double *k = SATURATION_FIT_K[ch], *w = LMS_TO_LINEAR_SRGB[ch];
    double s = k[0] + k[1] * a + k[2] * b + k[3] * a * a + k[4] * a * b;
    double sl[3], base[3];
    lms_slopes(a, b, sl);
    for (int i = 0; i < 3; i++) base[i] = 1 + s * sl[i];
    double f = 0, f1 = 0, f2 = 0;
    for (int i = 0; i < 3; i++) f = f + w[i] * pow(base[i], 3);
    for (int i = 0; i < 3; i++) f1 = f1 + w[i] * (3 * sl[i] * pow(base[i], 2));
    for (int i = 0; i < 3; i++) f2 = f2 + w[i] * (6 * pow(sl[i], 2) * base[i]);
    return s - (f * f1) / (f1 * f1 - 0.5 * f * f2);
}

static void cusp(double a, double b, double out[2]) {
    double s = max_saturation(a, b), lab[3] = {1, s * a, s * b}, lin[3];
    oklab_to_linear(lab, lin);
    double l = cbrt(1 / js_max3(lin));
    out[0] = l;
    out[1] = l * s;
}

static double max_chroma(double a, double b, double l, const double cs[2]) {
    if (l <= cs[0]) return (cs[1] * l) / cs[0];
    double t = (cs[1] * (l - 1)) / (cs[0] - 1);
    double sl[3], lms[3], cubes[3], first[3], second[3];
    lms_slopes(a, b, sl);
    for (int i = 0; i < 3; i++) {
        lms[i] = l + t * sl[i];
        cubes[i] = pow(lms[i], 3);
        first[i] = 3 * sl[i] * pow(lms[i], 2);
        second[i] = 6 * pow(sl[i], 2) * lms[i];
    }
    double best = 0;
    for (int i = 0; i < 3; i++) {
        const double *row = LMS_TO_LINEAR_SRGB[i];
        double f = row[0] * cubes[0] + row[1] * cubes[1] + row[2] * cubes[2] - 1;
        double f1 = row[0] * first[0] + row[1] * first[1] + row[2] * first[2];
        double f2 = row[0] * second[0] + row[1] * second[1] + row[2] * second[2];
        double u = f1 / (f1 * f1 - 0.5 * f * f2);
        double step = u >= 0 ? -f * u : DBL_MAX;
        best = i == 0 ? step : js_min(best, step);
    }
    return t + best;
}

static void chroma_stops(double L, double a, double b, double out[3]) {
    double peak[2];
    cusp(a, b, peak);
    double cmax = max_chroma(a, b, L, peak);
    double k = cmax / js_min(L * (peak[1] / peak[0]), (1 - L) * (peak[1] / (1 - peak[0])));
    double mids = 0.11516993 +
                  1 / (7.4477897 + 4.1590124 * b +
                       a * (-2.19557347 + 1.75198401 * b +
                            a * (-2.13704948 - 10.02301043 * b + a * (-4.24894561 + 5.38770819 * b + 4.69891013 * a))));
    double midt = 0.11239642 +
                  1 / (1.6132032 - 0.68124379 * b +
                       a * (0.40370612 + 0.90148123 * b +
                            a * (-0.27087943 + 0.6122399 * b + a * (0.00299215 - 0.45399568 * b - 0.14661872 * a))));
    double cmid = 0.9 * k * sqrt(sqrt(1 / (1 / pow(L * mids, 4) + 1 / pow((1 - L) * midt, 4))));
    double c0 = sqrt(1 / (1 / pow(L * 0.4, 2) + 1 / pow((1 - L) * 0.8, 2)));
    out[0] = c0;
    out[1] = cmid;
    out[2] = cmax;
}

// okhslColor(h, s, l): OKHSL to sRGB, rounded and clipped.
static Rgb okhsl_rgb(double hue, double s, double light) {
    double L = okhsl_to_oklab_l(light), lab[3] = {L, 0, 0}, lin[3];
    if (L > 0 && L < 1 && s > 0) {
        double angle = (2 * M_PI * fmod(fmod(hue, 360) + 360, 360)) / 360;
        double a = cos(angle), b = sin(angle), st[3], chroma;
        chroma_stops(L, a, b, st);
        double c0 = st[0], cmid = st[1], cmax = st[2];
        if (s < 0.8) {
            double t = 1.25 * s, k1 = 0.8 * c0;
            chroma = (t * k1) / (1 - (1 - k1 / cmid) * t);
        } else {
            double t = 5 * (s - 0.8), k1 = (0.2 * pow(cmid, 2) * pow(1.25, 2)) / c0;
            chroma = cmid + (t * k1) / (1 - (1 - k1 / (cmax - cmid)) * t);
        }
        lab[1] = chroma * a;
        lab[2] = chroma * b;
    }
    oklab_to_linear(lab, lin);
    return linear_to_rgb(lin);
}

typedef struct {
    double h, s, l, chroma; // OKHSL channels and the OKLCH chroma (sourceOf)
} Source;

static double hue_of(double a, double b) { return fmod((atan2(b, a) * 180) / M_PI + 360, 360); }

// rgbToOkhsl.
static void rgb_okhsl(Rgb c, double *h, double *s, double *l) {
    double lab[3];
    rgb_to_oklab(c, lab);
    double L = lab[0], chroma = js_hypot(lab[1], lab[2]), light = oklab_to_okhsl_l(L);
    *l = light;
    if (chroma < 1e-9 || light <= 0 || light >= 1) {
        *h = 0;
        *s = 0;
        return;
    }
    *h = hue_of(lab[1], lab[2]);
    double st[3], sat;
    chroma_stops(L, lab[1] / chroma, lab[2] / chroma, st);
    double c0 = st[0], cmid = st[1], cmax = st[2];
    if (chroma < cmid) {
        double k1 = 0.8 * c0;
        sat = 0.8 * (chroma / (k1 + (1 - k1 / cmid) * chroma));
    } else {
        double k1 = (0.2 * pow(cmid, 2) * pow(1.25, 2)) / c0, off = chroma - cmid;
        sat = 0.8 + 0.2 * (off / (k1 + (1 - k1 / (cmax - cmid)) * off));
    }
    *s = js_min(1, js_max(0, sat));
}

// colorToOklch for an sRGB color.
static void rgb_oklch(Rgb c, double *l, double *ch, double *h) {
    double lab[3];
    rgb_to_oklab(c, lab);
    *l = lab[0];
    *ch = js_hypot(lab[1], lab[2]);
    if (h) *h = hue_of(lab[1], lab[2]);
}

static double oklab_l(Rgb c) {
    double l, ch;
    rgb_oklch(c, &l, &ch, NULL);
    return l;
}

static bool in_gamut(const double lin[3]) {
    const double eps = 1e-7;
    for (int i = 0; i < 3; i++)
        if (!(lin[i] >= -eps && lin[i] <= 1 + eps)) return false;
    return true;
}

// colorToRgb(oklchColor(l, c, h)): chroma reduced by bisection until the color fits sRGB.
static Rgb oklch_rgb(double l, double c, double h) {
    h = fmod(fmod(h, 360) + 360, 360);
    double rad = (h * M_PI) / 180, cs = cos(rad), sn = sin(rad), lin[3];
    double lab[3] = {l, c * cs, c * sn};
    oklab_to_linear(lab, lin);
    if (in_gamut(lin)) return linear_to_rgb(lin);
    double best[3], lo = 0, hi = c;
    lab[1] = 0 * cs;
    lab[2] = 0 * sn;
    oklab_to_linear(lab, best);
    for (int i = 0; i < 20; i++) {
        double m = (lo + hi) / 2, cand[3];
        lab[1] = m * cs;
        lab[2] = m * sn;
        oklab_to_linear(lab, cand);
        if (in_gamut(cand)) {
            lo = m;
            memcpy(best, cand, sizeof best);
        } else hi = m;
    }
    return linear_to_rgb(best);
}

// ---- tokens, families, levels and rules (system-theme.ts) -------------------------------------------

// A token is a foreground (0..THEME_NFG-1) or a background (THEME_NFG + ThemeBgToken).
#define NTOK (THEME_NFG + THEME_NBG)
#define BG(x) (THEME_NFG + (x))
#define SURF_BACKGROUND (-1)

typedef struct {
    double hue, smin, smax;
    int slot;
} Family;

enum {
    F_NEUTRAL, F_BLUE, F_GREEN, F_RED, F_YELLOW, F_ORANGE, F_VIOLET, F_CALAMINE, F_T_SLATE, F_T_BLUE, F_T_PERI,
    F_T_VIOLET, F_T_MAGENTA, F_T_RED
};
static const Family FAMILIES[] = {
    [F_NEUTRAL] = {231.49, 0.02, 0.08, 8},   [F_BLUE] = {231.49, 0.1, 0.68, 4},
    [F_GREEN] = {158.68, 0.1, 0.76, 2},      [F_RED] = {20, 0.1, 0.92, 1},
    [F_YELLOW] = {82.36, 0.5, 1, 3},         [F_ORANGE] = {52, 0.12, 0.85, 3},
    [F_VIOLET] = {295, 0.2, 0.6, 5},         [F_CALAMINE] = {202.43, 0.1, 0.74, 6},
    [F_T_SLATE] = {231.49, 0.08, 0.2, 4},    [F_T_BLUE] = {231.49, 0.2, 0.45, 4},
    [F_T_PERI] = {263.25, 0.3, 0.6, 6},      [F_T_VIOLET] = {295, 0.4, 0.75, 5},
    [F_T_MAGENTA] = {337.5, 0.5, 0.85, 13},  [F_T_RED] = {20, 0.95, 1, 1},
};

static const uint8_t TOKEN_FAMILY[NTOK] = {
    [BG(TH_BG_SELECTED)] = F_BLUE, [BG(TH_BG_SEARCH_MATCH)] = F_ORANGE, [BG(TH_BG_USER_MESSAGE)] = F_BLUE,
    [BG(TH_BG_CUSTOM_MESSAGE)] = F_VIOLET, [BG(TH_BG_TOOL_PENDING)] = F_NEUTRAL,
    [BG(TH_BG_TOOL_SUCCESS)] = F_GREEN, [BG(TH_BG_TOOL_ERROR)] = F_RED,

    [TH_FG_TEXT] = F_NEUTRAL, [TH_FG_USER_MESSAGE_TEXT] = F_NEUTRAL, [TH_FG_CUSTOM_MESSAGE_TEXT] = F_NEUTRAL,
    [TH_FG_TOOL_TITLE] = F_NEUTRAL, [TH_FG_SYNTAX_OPERATOR] = F_NEUTRAL, [TH_FG_SYNTAX_PUNCTUATION] = F_NEUTRAL,
    [TH_FG_MUTED] = F_NEUTRAL, [TH_FG_DIM] = F_NEUTRAL, [TH_FG_THINKING_TEXT] = F_NEUTRAL,
    [TH_FG_TOOL_OUTPUT] = F_NEUTRAL, [TH_FG_MD_LINK_URL] = F_NEUTRAL, [TH_FG_MD_QUOTE] = F_NEUTRAL,
    [TH_FG_MD_QUOTE_BORDER] = F_NEUTRAL, [TH_FG_MD_HR] = F_NEUTRAL, [TH_FG_MD_CODE_BLOCK_BORDER] = F_NEUTRAL,
    [TH_FG_TOOL_DIFF_CONTEXT] = F_NEUTRAL, [TH_FG_SYNTAX_COMMENT] = F_NEUTRAL,
    [TH_FG_SCROLLBAR_TRACK] = F_NEUTRAL, [TH_FG_SCROLLBAR_THUMB] = F_NEUTRAL,
    [TH_FG_SEARCH_MATCH_TEXT] = F_NEUTRAL, [TH_FG_BORDER_MUTED] = F_NEUTRAL,

    [TH_FG_ACCENT] = F_VIOLET, [TH_FG_BORDER_ACCENT] = F_VIOLET, [TH_FG_CUSTOM_MESSAGE_LABEL] = F_VIOLET,
    [TH_FG_MD_CODE] = F_VIOLET, [TH_FG_MD_LIST_BULLET] = F_VIOLET, [TH_FG_SYNTAX_TYPE] = F_VIOLET,
    [TH_FG_BORDER] = F_BLUE, [TH_FG_MD_LINK] = F_BLUE, [TH_FG_SYNTAX_KEYWORD] = F_BLUE,
    [TH_FG_SYNTAX_VARIABLE] = F_CALAMINE, [TH_FG_SUCCESS] = F_GREEN, [TH_FG_MD_CODE_BLOCK] = F_GREEN,
    [TH_FG_TOOL_DIFF_ADDED] = F_GREEN, [TH_FG_BASH_MODE] = F_GREEN, [TH_FG_SYNTAX_NUMBER] = F_GREEN,
    [TH_FG_ERROR] = F_RED, [TH_FG_TOOL_DIFF_REMOVED] = F_RED, [TH_FG_WARNING] = F_YELLOW,
    [TH_FG_MD_HEADING] = F_YELLOW, [TH_FG_SYNTAX_FUNCTION] = F_YELLOW, [TH_FG_SYNTAX_STRING] = F_ORANGE,

    [TH_FG_THINKING_OFF] = F_NEUTRAL, [TH_FG_THINKING_MINIMAL] = F_T_SLATE, [TH_FG_THINKING_LOW] = F_T_BLUE,
    [TH_FG_THINKING_MEDIUM] = F_T_PERI, [TH_FG_THINKING_HIGH] = F_T_VIOLET,
    [TH_FG_THINKING_XHIGH] = F_T_MAGENTA, [TH_FG_THINKING_MAX] = F_T_RED,
};

// TOKEN_SLOTS: palette slots for tokens that would share a hue with a similar token.
static int token_slot(int tok) {
    if (tok == TH_FG_SYNTAX_STRING) return 2;
    if (tok == TH_FG_SYNTAX_NUMBER) return 5;
    if (tok == BG(TH_BG_SEARCH_MATCH)) return 3;
    return FAMILIES[TOKEN_FAMILY[tok]].slot;
}

static bool is_panel(int tok) { return tok >= THEME_NFG; }
static bool is_foreground_token(int tok) {
    return tok == TH_FG_TEXT || tok == TH_FG_USER_MESSAGE_TEXT || tok == TH_FG_TOOL_TITLE;
}

typedef struct {
    double c[6], lo, hi;
} Curve;

enum {
    L_PANEL, L_TRACK, L_THINK0, L_THINK1, L_THINK2, L_THINK3, L_THINK4, L_THINK5, L_THINK6, L_SUBTLE, L_THUMB,
    L_READABLE, L_EMPHASIS, L_TEXT_ON_PANEL, L_TEXT, L_COUNT
};
// [level][0 dark, 1 light]
static const Curve LEVELS[L_COUNT][2] = {
    [L_PANEL] = {{{0.29131, -0.39746, 2.33185, -0.85524, -1.2076, 0.86276}, 0, 0.979},
                 {{-3.74073, 27.94549, -78.44258, 112.6798, -79.60015, 22.11277}, 0.348, 1}},
    [L_TRACK] = {{{0.39028, -0.23015, 0.83573, 2.43829, -4.38292, 2.01582}, 0, 0.946},
                 {{-5.24921, 38.37322, -107.28833, 152.10005, -106.17127, 29.18061}, 0.368, 1}},
    [L_THINK0] = {{{0.52988, -0.05809, -0.30924, 4.63567, -6.52933, 2.89108}, 0, 0.873},
                  {{-28.27749, 182.85284, -469.62416, 603.15916, -384.59976, 97.35147}, 0.51, 1}},
    [L_THINK1] = {{{0.55278, -0.03667, -0.45659, 4.95347, -6.90265, 3.0706}, 0, 0.858},
                  {{-37.10484, 235.86282, -596.62344, 754.3633, -474.00763, 118.3551}, 0.535, 1}},
    [L_THINK2] = {{{0.57486, -0.01765, -0.58987, 5.25227, -7.27175, 3.25532}, 0, 0.842},
                  {{-59.89653, 377.05024, -945.07843, 1182.03145, -734.96375, 181.68658}, 0.556, 1}},
    [L_THINK3] = {{{0.59621, -0.00062, -0.71148, 5.53588, -7.6392, 3.44606}, 0, 0.827},
                  {{-72.07122, 445.84082, -1099.57352, 1353.88793, -829.53392, 202.26164}, 0.58, 1}},
    [L_THINK4] = {{{0.61691, 0.01462, -0.82288, 5.80651, -8.00641, 3.64333}, 0, 0.811},
                  {{-110.14338, 674.21488, -1645.75941, 2004.32367, -1215.15899, 293.3183}, 0.6, 1}},
    [L_THINK5] = {{{0.63702, 0.02826, -0.92498, 6.06465, -8.37246, 3.84651}, 0, 0.795},
                  {{-175.47701, 1063.54495, -2570.70594, 3098.80776, -1860.15527, 444.76392}, 0.62, 1}},
    [L_THINK6] = {{{0.65658, 0.04044, -1.01835, 6.30989, -8.73529, 4.05439}, 0, 0.779},
                  {{-183.81712, 1094.70055, -2602.68539, 3088.71276, -1826.91131, 430.75931}, 0.643, 1}},
    [L_SUBTLE] = {{{0.56762, -0.02475, -0.5383, 5.12628, -7.10931, 3.17324}, 0, 0.848},
                  {{-232.85459, 1376.54473, -3249.11801, 3827.91186, -2248.29472, 526.55751}, 0.657, 1}},
    [L_THUMB] = {{{0.60323, 0.00278, -0.73328, 5.57157, -7.68067, 3.46933}, 0, 0.823},
                 {{-82.89897, 511.01355, -1255.98095, 1540.76821, -940.68087, 228.58523}, 0.586, 1}},
    [L_READABLE] = {{{0.66937, 0.04704, -1.06871, 6.43941, -8.9332, 4.17229}, 0, 0.77},
                    {{-1554.52576, 8733.56817, -19604.93507, 21977.72696, -12300.99599, 2749.81288}, 0.751, 1}},
    [L_EMPHASIS] = {{{0.7303, 0.07695, -1.31626, 7.1681, -10.14436, 4.92846}, 0, 0.712},
                    {{-4948.31942, 26870.91986, -58334.48399, 63280.17197, -34298.01053, 7430.30146}, 0.811, 1}},
    [L_TEXT_ON_PANEL] = {{{0.86713, 0.05232, -0.89428, 4.79014, -5.5432, 1.75023}, 0, 0.542},
                         {{-8570.89457, 43954.60805, -90084.00702, 92220.6791, -47152.15802, 9632.27113}, 0.867, 1}},
    [L_TEXT] = {{{0.89242, 0.02311, -0.44862, 2.34417, -0.06084, -2.63844}, 0, 0.5},
                {{-2004.67048, 6664.47299, -6060.70202, -1792.61209, 5133.82359, -1939.85583}, 0.894, 1}},
};

typedef struct {
    int8_t tok, level, n, on[8];
} Rule;

#define R(tok, lvl, ...) {tok, lvl, (int8_t)sizeof((int8_t[]){__VA_ARGS__}), {__VA_ARGS__}}
#define BACK SURF_BACKGROUND
#define TOOLS BG(TH_BG_TOOL_PENDING), BG(TH_BG_TOOL_SUCCESS), BG(TH_BG_TOOL_ERROR)
#define MSGS BG(TH_BG_USER_MESSAGE), BG(TH_BG_CUSTOM_MESSAGE)
#define SEL BG(TH_BG_SELECTED)
#define CUSTOM BG(TH_BG_CUSTOM_MESSAGE)

static const Rule RULES[] = {
    // PANELS on the background
    R(BG(TH_BG_USER_MESSAGE), L_PANEL, BACK), R(BG(TH_BG_TOOL_PENDING), L_PANEL, BACK),
    R(BG(TH_BG_TOOL_SUCCESS), L_PANEL, BACK), R(BG(TH_BG_TOOL_ERROR), L_PANEL, BACK),
    R(BG(TH_BG_SELECTED), L_PANEL, BACK), R(BG(TH_BG_SEARCH_MATCH), L_PANEL, BACK),
    R(BG(TH_BG_CUSTOM_MESSAGE), L_PANEL, BACK),
    R(TH_FG_TEXT, L_TEXT, BACK),
    R(TH_FG_TEXT, L_TEXT_ON_PANEL, SEL),
    R(TH_FG_USER_MESSAGE_TEXT, L_TEXT_ON_PANEL, BG(TH_BG_USER_MESSAGE)),
    R(TH_FG_TOOL_TITLE, L_TEXT_ON_PANEL, TOOLS),
    R(TH_FG_ACCENT, L_READABLE, BACK, SEL, TOOLS), R(TH_FG_SUCCESS, L_READABLE, BACK, SEL, TOOLS),
    R(TH_FG_ERROR, L_READABLE, BACK, SEL, TOOLS), R(TH_FG_WARNING, L_READABLE, BACK, SEL, TOOLS),
    R(TH_FG_MUTED, L_READABLE, BACK, SEL, CUSTOM, TOOLS),
    R(TH_FG_DIM, L_SUBTLE, BACK, SEL, CUSTOM, TOOLS),
    R(TH_FG_THINKING_TEXT, L_READABLE, BACK),
    R(TH_FG_CUSTOM_MESSAGE_TEXT, L_READABLE, CUSTOM, TOOLS),
    R(TH_FG_CUSTOM_MESSAGE_LABEL, L_READABLE, BACK, CUSTOM, SEL, TOOLS),
    R(TH_FG_TOOL_OUTPUT, L_READABLE, BACK, TOOLS),
    R(TH_FG_MD_HEADING, L_READABLE, BACK, MSGS), R(TH_FG_MD_LINK, L_READABLE, BACK, MSGS),
    R(TH_FG_MD_LINK_URL, L_READABLE, BACK, MSGS), R(TH_FG_MD_CODE, L_READABLE, BACK, MSGS),
    R(TH_FG_MD_QUOTE, L_READABLE, BACK, MSGS), R(TH_FG_MD_CODE_BLOCK_BORDER, L_READABLE, BACK, MSGS),
    R(TH_FG_MD_LIST_BULLET, L_READABLE, BACK, MSGS),
    R(TH_FG_MD_CODE_BLOCK, L_READABLE, BACK, MSGS, TOOLS),
    R(TH_FG_TOOL_DIFF_ADDED, L_READABLE, BACK, TOOLS), R(TH_FG_TOOL_DIFF_REMOVED, L_READABLE, BACK, TOOLS),
    R(TH_FG_TOOL_DIFF_CONTEXT, L_READABLE, BACK, TOOLS),
    R(TH_FG_SYNTAX_COMMENT, L_READABLE, BACK, MSGS, TOOLS), R(TH_FG_SYNTAX_KEYWORD, L_READABLE, BACK, MSGS, TOOLS),
    R(TH_FG_SYNTAX_FUNCTION, L_READABLE, BACK, MSGS, TOOLS), R(TH_FG_SYNTAX_VARIABLE, L_READABLE, BACK, MSGS, TOOLS),
    R(TH_FG_SYNTAX_STRING, L_READABLE, BACK, MSGS, TOOLS), R(TH_FG_SYNTAX_NUMBER, L_READABLE, BACK, MSGS, TOOLS),
    R(TH_FG_SYNTAX_TYPE, L_READABLE, BACK, MSGS, TOOLS), R(TH_FG_SYNTAX_OPERATOR, L_READABLE, BACK, MSGS, TOOLS),
    R(TH_FG_SYNTAX_PUNCTUATION, L_READABLE, BACK, MSGS, TOOLS),
    R(TH_FG_SEARCH_MATCH_TEXT, L_READABLE, BG(TH_BG_SEARCH_MATCH)),
    R(TH_FG_BASH_MODE, L_READABLE, BACK), R(TH_FG_BORDER, L_READABLE, BACK), R(TH_FG_BORDER_ACCENT, L_READABLE, BACK),
    R(TH_FG_BORDER_MUTED, L_SUBTLE, BACK),
    R(TH_FG_MD_QUOTE_BORDER, L_READABLE, BACK, MSGS, TOOLS), R(TH_FG_MD_HR, L_READABLE, BACK, MSGS, TOOLS),
    R(TH_FG_SCROLLBAR_TRACK, L_TRACK, BACK),
    R(TH_FG_SCROLLBAR_THUMB, L_THUMB, TH_FG_SCROLLBAR_TRACK),
    R(TH_FG_THINKING_OFF, L_THINK0, BACK), R(TH_FG_THINKING_MINIMAL, L_THINK1, BACK),
    R(TH_FG_THINKING_LOW, L_THINK2, BACK), R(TH_FG_THINKING_MEDIUM, L_THINK3, BACK),
    R(TH_FG_THINKING_HIGH, L_THINK4, BACK), R(TH_FG_THINKING_XHIGH, L_THINK5, BACK),
    R(TH_FG_THINKING_MAX, L_THINK6, BACK),
};
#define NRULES ((int)(sizeof RULES / sizeof RULES[0]))

#define READABLE_FLOOR(light) ((light) ? L_SUBTLE : L_READABLE)
#define FOREGROUND_LEVEL L_EMPHASIS
#define TEXT_MINIMUM_WCAG_CONTRAST 4.5

// SOLVE_ORDER: tokens in dependency order, every surface before the tokens drawn on it.
static int solve_order[NTOK], solve_n;
static bool in_order(int tok) {
    for (int i = 0; i < solve_n; i++)
        if (solve_order[i] == tok) return true;
    return false;
}
static void visit(int tok) {
    if (in_order(tok)) return;
    for (int r = 0; r < NRULES; r++) {
        if (RULES[r].tok != tok) continue;
        for (int j = 0; j < RULES[r].n; j++)
            if (RULES[r].on[j] != SURF_BACKGROUND) visit(RULES[r].on[j]);
    }
    solve_order[solve_n++] = tok;
}
static void build_solve_order(void) {
    if (solve_n) return;
    for (int r = 0; r < NRULES; r++) visit(RULES[r].tok);
}

// ---- generation -----------------------------------------------------------------------------------

static double relative_luminance(Rgb c) {
    double v[3] = {c.r / 255, c.g / 255, c.b / 255};
    for (int i = 0; i < 3; i++) v[i] = v[i] <= 0.04045 ? v[i] / 12.92 : pow((v[i] + 0.055) / 1.055, 2.4);
    return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
}

static double wcag_contrast(Rgb x, Rgb y) {
    double a = relative_luminance(x), b = relative_luminance(y);
    return (js_max(a, b) + 0.05) / (js_min(a, b) + 0.05);
}

// terminalAppearance: true for light.
static bool terminal_is_light(Rgb bg, const Rgb *fg) {
    Rgb white = {255, 255, 255}, black = {0, 0, 0};
    double wc = wcag_contrast(white, bg), bc = wcag_contrast(black, bg);
    if (fg) {
        double fl = oklab_l(*fg), bl = oklab_l(bg);
        if (fabs(fl - bl) > 0.05) {
            bool light = !(fl > bl);
            double best = light ? bc : wc;
            if (best >= TEXT_MINIMUM_WCAG_CONTRAST) return light;
        }
    }
    return !(wc >= bc);
}

static double bell_weight(double l) {
    double g0 = exp(-pow(0 - 0.5, 2) / (2 * pow(0.25, 2)));
    double gl = exp(-pow(l - 0.5, 2) / (2 * pow(0.25, 2)));
    return (gl - g0) / (1 - g0);
}

static double saturation_curve(const Family *f, double l) {
    double floor_ = f->smax > 0 ? f->smin / f->smax : 1;
    return floor_ + (1 - floor_) * bell_weight(l);
}

static Source source_of(Rgb c) {
    Source s;
    rgb_okhsl(c, &s.h, &s.s, &s.l);
    double l, ch;
    rgb_oklch(c, &l, &ch, NULL);
    s.chroma = ch;
    return s;
}

static Rgb anchored(const Source *src, const Family *fam, double lightness, double sat) {
    double anchor = saturation_curve(fam, src->l);
    double falloff = anchor > 0 ? js_min(1, saturation_curve(fam, lightness) / anchor) : 1;
    Rgb color = okhsl_rgb(src->h, src->s * falloff * sat, lightness);
    double cap = src->chroma * falloff * sat, l, c;
    rgb_oklch(color, &l, &c, NULL);
    return c <= cap ? color : oklch_rgb(l, cap, src->h);
}

static Rgb with_text_contrast(Rgb color, const Rgb *surf, int n, bool lighter) {
#define MEETS(cand, ok)                                                                                     \
    do {                                                                                                    \
        ok = true;                                                                                          \
        for (int i_ = 0; i_ < n; i_++)                                                                      \
            if (!(wcag_contrast(cand, surf[i_]) >= TEXT_MINIMUM_WCAG_CONTRAST)) {                           \
                ok = false;                                                                                 \
                break;                                                                                      \
            }                                                                                               \
    } while (0)
    bool ok;
    MEETS(color, ok);
    if (ok) return color;
    double h, s, l;
    rgb_okhsl(color, &h, &s, &l);
    double extreme = lighter ? 1 : 0;
    Rgb at = okhsl_rgb(h, s, extreme);
    MEETS(at, ok);
    if (!ok) return at;
    double lo = l, hi = extreme;
    for (int i = 0; i < 20; i++) {
        double m = (lo + hi) / 2;
        Rgb c = okhsl_rgb(h, s, m);
        MEETS(c, ok);
        if (ok) hi = m;
        else lo = m;
    }
    return okhsl_rgb(h, s, hi);
#undef MEETS
}

typedef struct {
    bool light, lighter, has_palette, has_fg;
    double extreme, background_l, saturation;
    Rgb background, foreground;
    Source palette[16];
} Gen;

static bool level_target(const Gen *g, int level, double sl, double *out) {
    const Curve *c = &LEVELS[level][g->light];
    if (sl < c->lo || sl > c->hi) return false;
    double sum = 0;
    for (int p = 0; p < 6; p++) sum = sum + c->c[p] * pow(sl, p);
    *out = sum;
    return true;
}

// target(level, surfaceL, t); false for undefined.
static bool target(const Gen *g, int level, double sl, double t, double *out) {
    double reached;
    bool has = level_target(g, level, sl, &reached);
    if (!has && t == 0) return false;
    double distance = (has ? reached : g->extreme) - sl;
    double fl;
    bool hf = level_target(g, READABLE_FLOOR(g->light), sl, &fl);
    double floor_ = (hf ? fl : g->extreme) - sl;
    double compressed = fabs(distance) > fabs(floor_) ? distance - (distance - floor_) * js_min(t, 1) : distance;
    *out = sl + compressed * (1 - js_max(0, t - 1));
    return true;
}

static Rgb paint(const Gen *g, int tok, double oklab_lightness) {
    double lightness = oklab_to_okhsl_l(oklab_lightness);
    const Family *f = &FAMILIES[TOKEN_FAMILY[tok]];
    if (!g->has_palette)
        return okhsl_rgb(f->hue, (f->smin + (f->smax - f->smin) * bell_weight(lightness)) * g->saturation, lightness);
    return anchored(&g->palette[token_slot(tok)], f, lightness, g->saturation);
}

static bool panel_readable(const Gen *g, Rgb c) {
    Rgb ext = g->lighter ? (Rgb){255, 255, 255} : (Rgb){0, 0, 0};
    return wcag_contrast(ext, c) >= TEXT_MINIMUM_WCAG_CONTRAST;
}

static Rgb limit_panel(const Gen *g, int tok, double l) {
    Rgb c = paint(g, tok, l);
    if (panel_readable(g, c)) return c;
    double lo = g->background_l, hi = l;
    for (int i = 0; i < 20; i++) {
        double m = (lo + hi) / 2;
        if (panel_readable(g, paint(g, tok, m))) lo = m;
        else hi = m;
    }
    return paint(g, tok, lo);
}

static Rgb surface_color(const Gen *g, const Rgb *colors, const bool *set, int s) {
    return s == SURF_BACKGROUND || !set[s] ? g->background : colors[s];
}

static bool solve(const Gen *g, double t, Rgb colors[NTOK], bool set[NTOK]) {
    memset(set, 0, NTOK * sizeof set[0]);
    for (int o = 0; o < solve_n; o++) {
        int tok = solve_order[o];
        double best = 0;
        int nt = 0;
        for (int r = 0; r < NRULES; r++) {
            if (RULES[r].tok != tok) continue;
            for (int j = 0; j < RULES[r].n; j++) {
                double v;
                double sl = oklab_l(surface_color(g, colors, set, RULES[r].on[j]));
                if (!target(g, RULES[r].level, sl, t, &v) || v < 0 || v > 1) return false;
                best = nt++ == 0 ? v : g->lighter ? js_max(best, v) : js_min(best, v);
            }
        }
        double l = nt ? best : (g->lighter ? -INFINITY : INFINITY);
        colors[tok] = is_panel(tok) ? limit_panel(g, tok, l) : paint(g, tok, l);
        set[tok] = true;
    }
    return true;
}

static void set_rgb(ThemeValue *v, Rgb c) {
    v->kind = TH_VALUE_RGB;
    v->rgb[0] = (uint8_t)js_round(c.r);
    v->rgb[1] = (uint8_t)js_round(c.g);
    v->rgb[2] = (uint8_t)js_round(c.b);
}

static void generate(const Gen *g, ThemeValue vals[NTOK]) {
    Rgb colors[NTOK], tmp[NTOK];
    bool set[NTOK], tset[NTOK];
    double relaxation = 0;
    if (!solve(g, 0, colors, set)) {
        double lo = 0, hi = 2;
        bool got = solve(g, hi, colors, set);
        for (int i = 0; i < 20; i++) {
            double m = (lo + hi) / 2;
            if (solve(g, m, tmp, tset)) {
                hi = m;
                memcpy(colors, tmp, sizeof colors);
                memcpy(set, tset, sizeof set);
                got = true;
            } else lo = m;
        }
        relaxation = hi;
        if (!got) memset(set, 0, sizeof set);
    }
    for (int k = 0; k < NTOK; k++) {
        memset(&vals[k], 0, sizeof vals[k]);
        if (set[k]) set_rgb(&vals[k], colors[k]);
    }
    for (int tok = 0; tok < THEME_NFG; tok++) {
        if (!is_foreground_token(tok)) continue;
        Rgb surf[8];
        int ns = 0;
        for (int r = 0; r < NRULES; r++)
            if (RULES[r].tok == tok)
                for (int j = 0; j < RULES[r].n; j++) surf[ns++] = surface_color(g, colors, set, RULES[r].on[j]);
        bool has_text = set[tok];
        Rgb text = colors[tok];
        if (g->has_fg) {
            bool all = true;
            double needed = 0;
            for (int i = 0; i < ns; i++) {
                double v;
                if (!target(g, FOREGROUND_LEVEL, oklab_l(surf[i]), relaxation, &v) || !(v >= 0 && v <= 1)) {
                    all = false;
                    break;
                }
                needed = i == 0 ? v : g->lighter ? js_max(needed, v) : js_min(needed, v);
            }
            if (all) {
                if (ns == 0) needed = g->lighter ? -INFINITY : INFINITY;
                double fl = oklab_l(g->foreground);
                if (g->lighter ? fl >= needed : fl <= needed) {
                    memset(&vals[tok], 0, sizeof vals[tok]);
                    continue;
                }
                Source src = source_of(g->foreground);
                text = anchored(&src, &FAMILIES[F_NEUTRAL], oklab_to_okhsl_l(needed), g->saturation);
                has_text = true;
            }
        }
        if (has_text) set_rgb(&vals[tok], with_text_contrast(text, surf, ns, g->lighter));
    }
}

// indexedColors: ANSI indices for the families, default colors for neutral tokens (faint below body
// text) and for panels.
static void indexed(double saturation, ThemeValue vals[NTOK]) {
    for (int tok = 0; tok < NTOK; tok++) {
        memset(&vals[tok], 0, sizeof vals[tok]);
        if (is_panel(tok)) continue;
        bool neutral = TOKEN_FAMILY[tok] == F_NEUTRAL;
        if (!neutral && saturation > 0) {
            vals[tok].kind = TH_VALUE_INDEX;
            vals[tok].index = (uint8_t)token_slot(tok);
        }
        if (neutral && !is_foreground_token(tok)) vals[tok].dim = true;
    }
}

// ---- escape sequences (colors.ts foregroundAnsi / backgroundAnsi, theme.ts Theme) -------------------

static const double CUBE[6] = {0, 95, 135, 175, 215, 255};

static int closest(const double *v, int n, double t) {
    int best = 0;
    double bd = INFINITY;
    for (int i = 0; i < n; i++) {
        double d = fabs(t - v[i]);
        if (d < bd) {
            best = i;
            bd = d;
        }
    }
    return best;
}

static double color_distance(const double a[3], const double b[3]) {
    double dr = a[0] - b[0], dg = a[1] - b[1], db = a[2] - b[2];
    return dr * dr * 0.299 + dg * dg * 0.587 + db * db * 0.114;
}

static int rgb_to_ansi256(const uint8_t c8[3]) {
    double c[3] = {c8[0], c8[1], c8[2]}, gray_values[24];
    for (int i = 0; i < 24; i++) gray_values[i] = 8 + i * 10;
    int ri = closest(CUBE, 6, c[0]), gi = closest(CUBE, 6, c[1]), bi = closest(CUBE, 6, c[2]);
    double cube[3] = {CUBE[ri], CUBE[gi], CUBE[bi]};
    int cube_index = 16 + 36 * ri + 6 * gi + bi;
    double gray = js_round(0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]);
    int go = closest(gray_values, 24, gray);
    double gv[3] = {gray_values[go], gray_values[go], gray_values[go]};
    double mx = js_max3(c), mn = js_min(js_min(c[0], c[1]), c[2]);
    if (mx - mn < 10 && color_distance(c, gv) < color_distance(c, cube)) return 232 + go;
    return cube_index;
}

static void sgr_of(const ThemeValue *v, ColorMode mode, bool background, char *out, size_t cap) {
    int base = background ? 48 : 38;
    if (v->kind == TH_VALUE_DEFAULT) snprintf(out, cap, "\x1b[%dm", background ? 49 : 39);
    else if (v->kind == TH_VALUE_INDEX) snprintf(out, cap, "\x1b[%d;5;%dm", base, v->index);
    else if (mode == TC_TRUECOLOR) snprintf(out, cap, "\x1b[%d;2;%d;%d;%dm", base, v->rgb[0], v->rgb[1], v->rgb[2]);
    else snprintf(out, cap, "\x1b[%d;5;%dm", base, rgb_to_ansi256(v->rgb));
}

// ---- public API -----------------------------------------------------------------------------------

static bool env_is(const char *name, const char *value) {
    const char *v = getenv(name);
    return v && !strcmp(v, value);
}

static bool env_set(const char *name) {
    const char *v = getenv(name);
    return v && *v;
}

// The env value lowercased into buf (detectCapabilitiesFromEnvironment lowercases each).
static const char *env_lower(const char *name, char *buf, size_t cap) {
    const char *v = getenv(name);
    size_t i = 0;
    for (; v && v[i] && i + 1 < cap; i++) buf[i] = (char)(v[i] >= 'A' && v[i] <= 'Z' ? v[i] + 32 : v[i]);
    buf[i] = 0;
    return buf;
}

ColorMode theme_color_mode(void) {
    if (env_is("PI_TRUE_COLOR", "1")) return TC_TRUECOLOR;
    if (env_is("PI_TRUE_COLOR", "0")) return TC_256;
    char tp[64], te[64], tm[64], ct[64];
    const char *prog = env_lower("TERM_PROGRAM", tp, sizeof tp), *emu = env_lower("TERMINAL_EMULATOR", te, sizeof te);
    const char *term = env_lower("TERM", tm, sizeof tm), *colorterm = env_lower("COLORTERM", ct, sizeof ct);
    size_t tl = strlen(term);
    bool hint = !strcmp(colorterm, "truecolor") || !strcmp(colorterm, "24bit") ||
                (tl >= 7 && !strcmp(term + tl - 7, "-direct"));
    bool truecolor;
    if (env_set("TMUX") || !strncmp(term, "tmux", 4)) truecolor = hint;
    else if (!strncmp(term, "screen", 6)) truecolor = hint;
    else if (!strcmp(prog, "herdr")) truecolor = hint;
    else if (!strcmp(prog, "rex")) truecolor = true;
    else if (!strcmp(prog, "orca") && env_is("ORCA_IMAGE_PROTOCOL", "kitty")) truecolor = true;
    else if (env_set("KITTY_WINDOW_ID") || !strcmp(prog, "kitty")) truecolor = true;
    else if (!strcmp(prog, "ghostty") || strstr(term, "ghostty") || env_set("GHOSTTY_RESOURCES_DIR")) truecolor = true;
    else if (env_set("WEZTERM_PANE") || !strcmp(prog, "wezterm")) truecolor = true;
    else if (!strcmp(prog, "warpterminal") || env_set("WARP_SESSION_ID") || env_set("WARP_TERMINAL_SESSION_UUID"))
        truecolor = true;
    else if (env_set("ITERM_SESSION_ID") || !strcmp(prog, "iterm.app")) truecolor = true;
    else if (env_set("WT_SESSION")) truecolor = true;
    else if (!strcmp(prog, "alacritty") || !strncmp(term, "alacritty", 9) || env_set("ALACRITTY_WINDOW_ID") ||
             !strcmp(prog, "vscode") || !strcmp(prog, "zed"))
        truecolor = true;
    else if (!strcmp(emu, "jetbrains-jediterm")) truecolor = true;
    else truecolor = hint;
    return truecolor ? TC_TRUECOLOR : TC_256;
}

// detectTerminalTheme without a background: the mode 2031 report, then COLORFGBG, then dark.
static bool hint_is_light(const TermColors *tc) {
    if (tc && tc->scheme) return tc->scheme == 2;
    const char *v = getenv("COLORFGBG");
    if (!v) return false;
    const char *last = strrchr(v, ';');
    last = last ? last + 1 : v;
    while (*last == ' ' || *last == '\t') last++;
    size_t n = strlen(last);
    while (n && (last[n - 1] == ' ' || last[n - 1] == '\t' || last[n - 1] == '\n' || last[n - 1] == '\r')) n--;
    if (n < 1 || n > 2) return false;
    for (size_t i = 0; i < n; i++)
        if (last[i] < '0' || last[i] > '9') return false;
    int idx = n == 1 ? last[0] - '0' : (last[0] - '0') * 10 + (last[1] - '0');
    if (idx > 15) return false;
    return !(idx <= 6 || idx == 8);
}

static Rgb rgb_of(const uint8_t c[3]) { return (Rgb){c[0], c[1], c[2]}; }

// Theme.fg / Theme.bg: each token's opening and closing sequences.
static void emit_sgr(const ThemeValue vals[NTOK], ColorMode mode, ThemeSgr fg[THEME_NFG], ThemeSgr bg[THEME_NBG],
                     ThemeValue fg_values[THEME_NFG], ThemeValue bg_values[THEME_NBG]) {
    for (int tok = 0; tok < NTOK; tok++) {
        bool back = is_panel(tok);
        ThemeSgr *s = back ? &bg[tok - THEME_NFG] : &fg[tok];
        sgr_of(&vals[tok], mode, back, s->open, sizeof s->open);
        if (back) snprintf(s->close, sizeof s->close, "\x1b[49m");
        else if (vals[tok].dim) {
            strcat(s->open, "\x1b[2m");
            snprintf(s->close, sizeof s->close, "\x1b[22;39m");
        } else snprintf(s->close, sizeof s->close, "\x1b[39m");
        if (back && bg_values) bg_values[tok - THEME_NFG] = vals[tok];
        if (!back && fg_values) fg_values[tok] = vals[tok];
    }
}

void theme_system_ex(const TermColors *tc, ColorMode mode, bool pending, ThemeSgr fg[THEME_NFG],
                     ThemeSgr bg[THEME_NBG], ThemeValue fg_values[THEME_NFG], ThemeValue bg_values[THEME_NBG],
                     bool *is_light) {
    ThemeValue vals[NTOK];
    double saturation = pending ? 0 : 1;
    bool light;
    if (!tc || !tc->has_bg) {
        indexed(saturation, vals);
        light = hint_is_light(tc);
    } else {
        build_solve_order();
        Gen g = {0};
        g.background = rgb_of(tc->bg);
        g.has_fg = tc->has_fg;
        if (g.has_fg) g.foreground = rgb_of(tc->fg);
        g.has_palette = tc->has_palette;
        if (g.has_palette)
            for (int i = 0; i < 16; i++) g.palette[i] = source_of(rgb_of(tc->palette[i]));
        g.saturation = saturation;
        g.light = terminal_is_light(g.background, g.has_fg ? &g.foreground : NULL);
        g.lighter = !g.light;
        g.extreme = g.lighter ? 1 : 0;
        g.background_l = oklab_l(g.background);
        generate(&g, vals);
        light = g.light;
    }
    emit_sgr(vals, mode, fg, bg, fg_values, bg_values);
    if (is_light) *is_light = light;
}

void theme_system(const TermColors *tc, ColorMode mode, ThemeSgr fg[THEME_NFG], ThemeSgr bg[THEME_NBG],
                  bool *is_light) {
    theme_system_ex(tc, mode, false, fg, bg, NULL, NULL, is_light);
}

// ---- named themes ---------------------------------------------------------------------------------

// pi's built-in dark.json and light.json, their var references resolved: okhsl(h s% l%) per token,
// parsed as pi's parseColor does (percentages divided by 100). They declare their appearance and
// set no token to "" (terminal default), so the terminal's reported colors play no part.
typedef struct {
    double h, s, l;
} Okhsl;

static const Okhsl PI_DARK[NTOK] = {
    [TH_FG_ACCENT] = {295, 50, 67}, [TH_FG_BORDER] = {231, 57, 65}, [TH_FG_BORDER_ACCENT] = {295, 53, 64},
    [TH_FG_BORDER_MUTED] = {229, 8, 53}, [TH_FG_SUCCESS] = {159, 59, 67}, [TH_FG_ERROR] = {20, 72, 67},
    [TH_FG_WARNING] = {83, 88, 67}, [TH_FG_MUTED] = {229, 6, 67}, [TH_FG_DIM] = {229, 8, 56},
    [TH_FG_TEXT] = {234, 3, 89}, [TH_FG_THINKING_TEXT] = {226, 7, 65}, [BG(TH_BG_SELECTED)] = {233, 41, 24},
    [TH_FG_SCROLLBAR_TRACK] = {237, 7, 33}, [TH_FG_SCROLLBAR_THUMB] = {232, 7, 65},
    [BG(TH_BG_SEARCH_MATCH)] = {53, 51, 24}, [TH_FG_SEARCH_MATCH_TEXT] = {229, 6, 67},
    [BG(TH_BG_USER_MESSAGE)] = {233, 41, 24}, [TH_FG_USER_MESSAGE_TEXT] = {234, 3, 89},
    [BG(TH_BG_CUSTOM_MESSAGE)] = {295, 42, 24}, [TH_FG_CUSTOM_MESSAGE_TEXT] = {229, 6, 67},
    [TH_FG_CUSTOM_MESSAGE_LABEL] = {295, 50, 67}, [BG(TH_BG_TOOL_PENDING)] = {229, 5, 24},
    [BG(TH_BG_TOOL_SUCCESS)] = {158, 46, 25}, [BG(TH_BG_TOOL_ERROR)] = {19, 54, 25},
    [TH_FG_TOOL_TITLE] = {234, 3, 89}, [TH_FG_TOOL_OUTPUT] = {229, 6, 67}, [TH_FG_MD_HEADING] = {83, 88, 67},
    [TH_FG_MD_LINK] = {232, 54, 67}, [TH_FG_MD_LINK_URL] = {229, 6, 67}, [TH_FG_MD_CODE] = {295, 50, 67},
    [TH_FG_MD_CODE_BLOCK] = {159, 59, 67}, [TH_FG_MD_CODE_BLOCK_BORDER] = {229, 6, 67},
    [TH_FG_MD_QUOTE] = {229, 6, 67}, [TH_FG_MD_QUOTE_BORDER] = {229, 6, 67}, [TH_FG_MD_HR] = {229, 6, 67},
    [TH_FG_MD_LIST_BULLET] = {295, 50, 67}, [TH_FG_TOOL_DIFF_ADDED] = {159, 59, 67},
    [TH_FG_TOOL_DIFF_REMOVED] = {20, 72, 67}, [TH_FG_TOOL_DIFF_CONTEXT] = {229, 6, 67},
    [TH_FG_SYNTAX_COMMENT] = {229, 6, 67}, [TH_FG_SYNTAX_KEYWORD] = {232, 54, 67},
    [TH_FG_SYNTAX_FUNCTION] = {83, 88, 67}, [TH_FG_SYNTAX_VARIABLE] = {202, 58, 67},
    [TH_FG_SYNTAX_STRING] = {52, 67, 67}, [TH_FG_SYNTAX_NUMBER] = {159, 59, 67}, [TH_FG_SYNTAX_TYPE] = {295, 50, 67},
    [TH_FG_SYNTAX_OPERATOR] = {229, 6, 67}, [TH_FG_SYNTAX_PUNCTUATION] = {229, 6, 67},
    [TH_FG_THINKING_OFF] = {229, 8, 49}, [TH_FG_THINKING_MINIMAL] = {232, 20, 52}, [TH_FG_THINKING_LOW] = {232, 45, 54},
    [TH_FG_THINKING_MEDIUM] = {263, 59, 56}, [TH_FG_THINKING_HIGH] = {295, 73, 59},
    [TH_FG_THINKING_XHIGH] = {337, 81, 61}, [TH_FG_THINKING_MAX] = {20, 99, 63}, [TH_FG_BASH_MODE] = {159, 64, 65},
};

static const Okhsl PI_LIGHT[NTOK] = {
    [TH_FG_ACCENT] = {295, 60, 46}, [TH_FG_BORDER] = {231, 67, 55}, [TH_FG_BORDER_ACCENT] = {295, 59, 55},
    [TH_FG_BORDER_MUTED] = {235, 7, 66}, [TH_FG_SUCCESS] = {159, 75, 46}, [TH_FG_ERROR] = {20, 91, 47},
    [TH_FG_WARNING] = {83, 99, 47}, [TH_FG_MUTED] = {229, 8, 47}, [TH_FG_DIM] = {229, 7, 59},
    [TH_FG_TEXT] = {225, 5, 27}, [TH_FG_THINKING_TEXT] = {234, 8, 55}, [BG(TH_BG_SELECTED)] = {235, 19, 91},
    [TH_FG_SCROLLBAR_TRACK] = {248, 3, 90}, [TH_FG_SCROLLBAR_THUMB] = {226, 7, 65},
    [BG(TH_BG_SEARCH_MATCH)] = {56, 22, 91}, [TH_FG_SEARCH_MATCH_TEXT] = {229, 8, 47},
    [BG(TH_BG_USER_MESSAGE)] = {235, 19, 91}, [TH_FG_USER_MESSAGE_TEXT] = {225, 5, 27},
    [BG(TH_BG_CUSTOM_MESSAGE)] = {295, 25, 91}, [TH_FG_CUSTOM_MESSAGE_TEXT] = {229, 8, 47},
    [TH_FG_CUSTOM_MESSAGE_LABEL] = {295, 60, 46}, [BG(TH_BG_TOOL_PENDING)] = {248, 3, 91},
    [BG(TH_BG_TOOL_SUCCESS)] = {156, 21, 91}, [BG(TH_BG_TOOL_ERROR)] = {24, 23, 91},
    [TH_FG_TOOL_TITLE] = {225, 5, 27}, [TH_FG_TOOL_OUTPUT] = {229, 8, 47}, [TH_FG_MD_HEADING] = {83, 99, 47},
    [TH_FG_MD_LINK] = {231, 68, 47}, [TH_FG_MD_LINK_URL] = {229, 8, 47}, [TH_FG_MD_CODE] = {295, 60, 46},
    [TH_FG_MD_CODE_BLOCK] = {159, 75, 46}, [TH_FG_MD_CODE_BLOCK_BORDER] = {229, 8, 47},
    [TH_FG_MD_QUOTE] = {229, 8, 47}, [TH_FG_MD_QUOTE_BORDER] = {229, 8, 47}, [TH_FG_MD_HR] = {229, 8, 47},
    [TH_FG_MD_LIST_BULLET] = {295, 60, 46}, [TH_FG_TOOL_DIFF_ADDED] = {159, 75, 46},
    [TH_FG_TOOL_DIFF_REMOVED] = {20, 91, 47}, [TH_FG_TOOL_DIFF_CONTEXT] = {229, 8, 47},
    [TH_FG_SYNTAX_COMMENT] = {229, 8, 47}, [TH_FG_SYNTAX_KEYWORD] = {231, 68, 47},
    [TH_FG_SYNTAX_FUNCTION] = {83, 99, 47}, [TH_FG_SYNTAX_VARIABLE] = {203, 73, 46},
    [TH_FG_SYNTAX_STRING] = {52, 84, 46}, [TH_FG_SYNTAX_NUMBER] = {159, 75, 46}, [TH_FG_SYNTAX_TYPE] = {295, 60, 46},
    [TH_FG_SYNTAX_OPERATOR] = {229, 8, 47}, [TH_FG_SYNTAX_PUNCTUATION] = {229, 8, 47},
    [TH_FG_THINKING_OFF] = {223, 5, 80}, [TH_FG_THINKING_MINIMAL] = {229, 14, 78}, [TH_FG_THINKING_LOW] = {232, 33, 76},
    [TH_FG_THINKING_MEDIUM] = {264, 48, 74}, [TH_FG_THINKING_HIGH] = {295, 62, 72},
    [TH_FG_THINKING_XHIGH] = {337, 74, 70}, [TH_FG_THINKING_MAX] = {20, 98, 68}, [TH_FG_BASH_MODE] = {159, 74, 55},
};

static void pi_builtin(const Okhsl *t, ThemeValue vals[NTOK]) {
    for (int tok = 0; tok < NTOK; tok++) {
        memset(&vals[tok], 0, sizeof vals[tok]);
        set_rgb(&vals[tok], okhsl_rgb(t[tok].h, t[tok].s / 100, t[tok].l / 100));
    }
}

// Saetta Code's own theme: macOS system colors (dark and light), Xcode's default syntax colors.
// SAETTA_DEFAULT marks the terminal's default color.
#define SAETTA_DEFAULT (-1)
#define BORDERS_D 0x3A3A3C
#define BORDERS_L 0xD1D1D6
#define MUTED_D 0x98989D
#define MUTED_L 0x8E8E93

static const int32_t SAETTA_DARK[NTOK] = {
    [TH_FG_ACCENT] = 0x0A84FF, [TH_FG_BORDER] = BORDERS_D, [TH_FG_BORDER_ACCENT] = 0x0A84FF,
    [TH_FG_BORDER_MUTED] = 0x636366, [TH_FG_SUCCESS] = 0x30D158, [TH_FG_ERROR] = 0xFF453A,
    [TH_FG_WARNING] = 0xFF9F0A, [TH_FG_MUTED] = MUTED_D, [TH_FG_DIM] = 0x7C7C80, [TH_FG_TEXT] = SAETTA_DEFAULT,
    [TH_FG_THINKING_TEXT] = MUTED_D, [TH_FG_SCROLLBAR_TRACK] = BORDERS_D, [TH_FG_SCROLLBAR_THUMB] = 0x636366,
    [TH_FG_SEARCH_MATCH_TEXT] = SAETTA_DEFAULT, [TH_FG_USER_MESSAGE_TEXT] = SAETTA_DEFAULT,
    [TH_FG_CUSTOM_MESSAGE_TEXT] = MUTED_D, [TH_FG_CUSTOM_MESSAGE_LABEL] = 0xBF5AF2,
    [TH_FG_TOOL_TITLE] = SAETTA_DEFAULT, [TH_FG_TOOL_OUTPUT] = MUTED_D, [TH_FG_MD_HEADING] = 0x0A84FF,
    [TH_FG_MD_LINK] = 0x0A84FF, [TH_FG_MD_LINK_URL] = MUTED_D, [TH_FG_MD_CODE] = 0xE5E5EA,
    [TH_FG_MD_CODE_BLOCK] = 0x30D158, [TH_FG_MD_CODE_BLOCK_BORDER] = BORDERS_D, [TH_FG_MD_QUOTE] = MUTED_D,
    [TH_FG_MD_QUOTE_BORDER] = BORDERS_D, [TH_FG_MD_HR] = BORDERS_D, [TH_FG_MD_LIST_BULLET] = 0x0A84FF,
    [TH_FG_TOOL_DIFF_ADDED] = 0x30D158, [TH_FG_TOOL_DIFF_REMOVED] = 0xFF453A, [TH_FG_TOOL_DIFF_CONTEXT] = MUTED_D,
    [TH_FG_SYNTAX_COMMENT] = 0x7F8C98, [TH_FG_SYNTAX_KEYWORD] = 0xFF7AB2, [TH_FG_SYNTAX_FUNCTION] = 0x67B7A4,
    [TH_FG_SYNTAX_VARIABLE] = 0xA167E6, [TH_FG_SYNTAX_STRING] = 0xFC6A5D, [TH_FG_SYNTAX_NUMBER] = 0xD9C97C,
    [TH_FG_SYNTAX_TYPE] = 0x5DD8FF, [TH_FG_SYNTAX_OPERATOR] = MUTED_D, [TH_FG_SYNTAX_PUNCTUATION] = MUTED_D,
    [TH_FG_THINKING_OFF] = 0x636366, [TH_FG_THINKING_MINIMAL] = 0x5E7CE2, [TH_FG_THINKING_LOW] = 0x0A84FF,
    [TH_FG_THINKING_MEDIUM] = 0x5E5CE6, [TH_FG_THINKING_HIGH] = 0xBF5AF2, [TH_FG_THINKING_XHIGH] = 0xFF375F,
    [TH_FG_THINKING_MAX] = 0xFF453A, [TH_FG_BASH_MODE] = 0x30D158,
    [BG(TH_BG_SELECTED)] = 0x1C3A5E, [BG(TH_BG_SEARCH_MATCH)] = 0x3D2E14, [BG(TH_BG_USER_MESSAGE)] = 0x2C2C2E,
    [BG(TH_BG_CUSTOM_MESSAGE)] = 0x2A1F33, [BG(TH_BG_TOOL_PENDING)] = 0x242426,
    [BG(TH_BG_TOOL_SUCCESS)] = 0x1E2B22, [BG(TH_BG_TOOL_ERROR)] = 0x2E1F1F,
};

static const int32_t SAETTA_LIGHT[NTOK] = {
    [TH_FG_ACCENT] = 0x007AFF, [TH_FG_BORDER] = BORDERS_L, [TH_FG_BORDER_ACCENT] = 0x007AFF,
    [TH_FG_BORDER_MUTED] = 0xAEAEB2, [TH_FG_SUCCESS] = 0x34C759, [TH_FG_ERROR] = 0xFF3B30,
    [TH_FG_WARNING] = 0xFF9500, [TH_FG_MUTED] = MUTED_L, [TH_FG_DIM] = 0xAEAEB2, [TH_FG_TEXT] = SAETTA_DEFAULT,
    [TH_FG_THINKING_TEXT] = MUTED_L, [TH_FG_SCROLLBAR_TRACK] = BORDERS_L, [TH_FG_SCROLLBAR_THUMB] = 0xAEAEB2,
    [TH_FG_SEARCH_MATCH_TEXT] = SAETTA_DEFAULT, [TH_FG_USER_MESSAGE_TEXT] = SAETTA_DEFAULT,
    [TH_FG_CUSTOM_MESSAGE_TEXT] = MUTED_L, [TH_FG_CUSTOM_MESSAGE_LABEL] = 0xAF52DE,
    [TH_FG_TOOL_TITLE] = SAETTA_DEFAULT, [TH_FG_TOOL_OUTPUT] = MUTED_L, [TH_FG_MD_HEADING] = 0x007AFF,
    [TH_FG_MD_LINK] = 0x007AFF, [TH_FG_MD_LINK_URL] = MUTED_L, [TH_FG_MD_CODE] = 0x1C1C1E,
    [TH_FG_MD_CODE_BLOCK] = 0x34C759, [TH_FG_MD_CODE_BLOCK_BORDER] = BORDERS_L, [TH_FG_MD_QUOTE] = MUTED_L,
    [TH_FG_MD_QUOTE_BORDER] = BORDERS_L, [TH_FG_MD_HR] = BORDERS_L, [TH_FG_MD_LIST_BULLET] = 0x007AFF,
    [TH_FG_TOOL_DIFF_ADDED] = 0x34C759, [TH_FG_TOOL_DIFF_REMOVED] = 0xFF3B30, [TH_FG_TOOL_DIFF_CONTEXT] = MUTED_L,
    [TH_FG_SYNTAX_COMMENT] = 0x5D6C79, [TH_FG_SYNTAX_KEYWORD] = 0xAD3DA4, [TH_FG_SYNTAX_FUNCTION] = 0x326D74,
    [TH_FG_SYNTAX_VARIABLE] = 0x6C36A9, [TH_FG_SYNTAX_STRING] = 0xD12F1B, [TH_FG_SYNTAX_NUMBER] = 0x272AD8,
    [TH_FG_SYNTAX_TYPE] = 0x3900A0, [TH_FG_SYNTAX_OPERATOR] = MUTED_L, [TH_FG_SYNTAX_PUNCTUATION] = MUTED_L,
    [TH_FG_THINKING_OFF] = 0xAEAEB2, [TH_FG_THINKING_MINIMAL] = 0x4A63C9, [TH_FG_THINKING_LOW] = 0x007AFF,
    [TH_FG_THINKING_MEDIUM] = 0x5856D6, [TH_FG_THINKING_HIGH] = 0xAF52DE, [TH_FG_THINKING_XHIGH] = 0xFF2D55,
    [TH_FG_THINKING_MAX] = 0xFF3B30, [TH_FG_BASH_MODE] = 0x34C759,
    [BG(TH_BG_SELECTED)] = 0xDCEBFF, [BG(TH_BG_SEARCH_MATCH)] = 0xFFF0D9, [BG(TH_BG_USER_MESSAGE)] = 0xF2F2F7,
    [BG(TH_BG_CUSTOM_MESSAGE)] = 0xF5EEFA, [BG(TH_BG_TOOL_PENDING)] = 0xF5F5F7,
    [BG(TH_BG_TOOL_SUCCESS)] = 0xEEF6F0, [BG(TH_BG_TOOL_ERROR)] = 0xFBEFEF,
};

static void saetta_values(const int32_t *t, ThemeValue vals[NTOK]) {
    for (int tok = 0; tok < NTOK; tok++) {
        memset(&vals[tok], 0, sizeof vals[tok]);
        if (t[tok] == SAETTA_DEFAULT) continue;
        vals[tok].kind = TH_VALUE_RGB;
        vals[tok].rgb[0] = (uint8_t)(t[tok] >> 16);
        vals[tok].rgb[1] = (uint8_t)(t[tok] >> 8);
        vals[tok].rgb[2] = (uint8_t)t[tok];
    }
}

void theme_brand(bool light, uint8_t amber[3], uint8_t purple[3]) {
    uint32_t a = light ? 0xB36400 : 0xFFB020, p = light ? 0xAF52DE : 0xBF5AF2;
    for (int i = 0; i < 3; i++) {
        amber[i] = (uint8_t)(a >> (16 - 8 * i));
        purple[i] = (uint8_t)(p >> (16 - 8 * i));
    }
}

static const char *const THEME_NAMES[] = {"saetta", "system", "dark", "light"};

size_t theme_names(const char *const **names) {
    *names = THEME_NAMES;
    return sizeof THEME_NAMES / sizeof THEME_NAMES[0];
}

bool theme_named(const char *name, const TermColors *tc, ColorMode mode, int appearance, ThemeSgr fg[THEME_NFG],
                 ThemeSgr bg[THEME_NBG], bool *is_light) {
    ThemeValue vals[NTOK];
    bool light;
    if (!strcmp(name, "system")) {
        theme_system(tc, mode, fg, bg, is_light);
        return true;
    } else if (!strcmp(name, "dark") || !strcmp(name, "light")) {
        light = name[0] == 'l';
        pi_builtin(light ? PI_LIGHT : PI_DARK, vals);
    } else if (!strcmp(name, "saetta")) {
        if (appearance == 1 || appearance == 2) light = appearance == 2;
        else if (tc && tc->has_bg) {
            Rgb f = rgb_of(tc->fg);
            light = terminal_is_light(rgb_of(tc->bg), tc->has_fg ? &f : NULL);
        } else light = hint_is_light(tc);
        saetta_values(light ? SAETTA_LIGHT : SAETTA_DARK, vals);
    } else return false;
    emit_sgr(vals, mode, fg, bg, NULL, NULL);
    if (is_light) *is_light = light;
    return true;
}
