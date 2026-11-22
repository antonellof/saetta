// Saetta Code TUI: pi's "system" theme, generated from the colors the terminal reports.
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at v1.5.2,
// b4afddd): packages/coding-agent/src/modes/interactive/theme/system-theme.ts, theme.ts;
// packages/tui/src/oklab.ts, colors.ts.
//
// Every token belongs to a color family (a hue, and an ANSI palette slot it can take its hue from)
// and must reach a contrast level, a target-lightness curve, on the background and on the panels it
// is drawn on. Three tiers, as pi: background and palette (hues from the palette), background only
// (the families' own hues), nothing (ANSI palette indices and the terminal's default colors, faint
// neutral tokens, panels without a background). The arithmetic follows pi's JavaScript double for
// double (OKHSL and OKLab conversions, Math.round, V8's Math.hypot), so the colors are pi's exactly.
#ifndef SAETTA_CODE_TUI_THEME_H
#define SAETTA_CODE_TUI_THEME_H

#include <stdbool.h>
#include <stdint.h>

// Foreground tokens, in pi's THEME_TOKENS order (theme-tokens.ts).
typedef enum {
    TH_FG_ACCENT, TH_FG_BORDER, TH_FG_BORDER_ACCENT, TH_FG_BORDER_MUTED, TH_FG_SUCCESS, TH_FG_ERROR,
    TH_FG_WARNING, TH_FG_MUTED, TH_FG_DIM, TH_FG_TEXT, TH_FG_THINKING_TEXT, TH_FG_SCROLLBAR_TRACK,
    TH_FG_SCROLLBAR_THUMB, TH_FG_SEARCH_MATCH_TEXT, TH_FG_USER_MESSAGE_TEXT, TH_FG_CUSTOM_MESSAGE_TEXT,
    TH_FG_CUSTOM_MESSAGE_LABEL, TH_FG_TOOL_TITLE, TH_FG_TOOL_OUTPUT, TH_FG_MD_HEADING, TH_FG_MD_LINK,
    TH_FG_MD_LINK_URL, TH_FG_MD_CODE, TH_FG_MD_CODE_BLOCK, TH_FG_MD_CODE_BLOCK_BORDER, TH_FG_MD_QUOTE,
    TH_FG_MD_QUOTE_BORDER, TH_FG_MD_HR, TH_FG_MD_LIST_BULLET, TH_FG_TOOL_DIFF_ADDED, TH_FG_TOOL_DIFF_REMOVED,
    TH_FG_TOOL_DIFF_CONTEXT, TH_FG_SYNTAX_COMMENT, TH_FG_SYNTAX_KEYWORD, TH_FG_SYNTAX_FUNCTION,
    TH_FG_SYNTAX_VARIABLE, TH_FG_SYNTAX_STRING, TH_FG_SYNTAX_NUMBER, TH_FG_SYNTAX_TYPE, TH_FG_SYNTAX_OPERATOR,
    TH_FG_SYNTAX_PUNCTUATION, TH_FG_THINKING_OFF, TH_FG_THINKING_MINIMAL, TH_FG_THINKING_LOW,
    TH_FG_THINKING_MEDIUM, TH_FG_THINKING_HIGH, TH_FG_THINKING_XHIGH, TH_FG_THINKING_MAX, TH_FG_BASH_MODE,
    THEME_NFG
} ThemeFg;

// Background tokens (panels), in the same order.
typedef enum {
    TH_BG_SELECTED, TH_BG_SEARCH_MATCH, TH_BG_USER_MESSAGE, TH_BG_CUSTOM_MESSAGE, TH_BG_TOOL_PENDING,
    TH_BG_TOOL_SUCCESS, TH_BG_TOOL_ERROR,
    THEME_NBG
} ThemeBgToken;

// What the terminal reported: OSC 10 (foreground), OSC 11 (background), all 16 colors of OSC 4
// (palette), and its light/dark report (mode 2031: 0 none, 1 dark, 2 light), which only decides the
// appearance while no background was reported.
typedef struct {
    bool has_fg, has_bg, has_palette;
    uint8_t fg[3], bg[3], palette[16][3];
    int8_t scheme;
} TermColors;

// pi's color mode (getTerminalColorMode): 24-bit colors, or the nearest of the 256-color cube and
// gray ramp.
typedef enum { TC_TRUECOLOR, TC_256 } ColorMode;

// The color mode pi picks from the environment (terminal-image.ts detectCapabilities: COLORTERM
// truecolor or 24bit, TERM ending in -direct, the terminals known to support it, PI_TRUE_COLOR=0/1).
ColorMode theme_color_mode(void);

// The opening and closing sequences pi's theme.fg(token, text) / theme.bg(token, text) wrap text
// in: open + text + close. A faint token's open ends in SGR 2 (getFgAnsi) and its close is
// "\x1b[22;39m".
typedef struct {
    char open[48], close[24];
} ThemeSgr;

// pi's system theme for these colors (tc NULL, or no background reported: the indexed tier), with
// is_light the appearance pi detects (terminalAppearance, or without a background: the mode 2031
// report, then COLORFGBG, then dark). is_light may be NULL.
void theme_system(const TermColors *tc, ColorMode mode, ThemeSgr fg[THEME_NFG], ThemeSgr bg[THEME_NBG],
                  bool *is_light);

// A token's generated value: the terminal's default color, an ANSI palette index, or a color.
typedef enum { TH_VALUE_DEFAULT, TH_VALUE_INDEX, TH_VALUE_RGB } ThemeValueKind;
typedef struct {
    uint8_t kind, index, rgb[3];
    bool dim; // rendered faint (SGR 2)
} ThemeValue;

// theme_system with pi's pending state: while the color query is in flight (pending), pi draws the
// theme at saturation 0 (grayscale: the generated tiers in gray, the indexed tier all default
// colors). fg_values and bg_values (may be NULL) receive the generated values.
void theme_system_ex(const TermColors *tc, ColorMode mode, bool pending, ThemeSgr fg[THEME_NFG],
                     ThemeSgr bg[THEME_NBG], ThemeValue fg_values[THEME_NFG], ThemeValue bg_values[THEME_NBG],
                     bool *is_light);

#endif
