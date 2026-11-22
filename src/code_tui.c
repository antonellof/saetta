// Saetta Code TUI: the terminal (raw input, the alternate screen, the keyboard protocols, the
// mouse), the frame renderer and the key decoder (see code_tui.h).
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at v1.5.2,
// b4afddd): packages/tui/src/terminal.ts (ProcessTerminal: bracketed paste, the Kitty keyboard
// protocol query with the device-attributes sentinel and the modifyOtherKeys fallback),
// tui-alt-screen.ts (the fullscreen frame: rows written with CUP and erase-line, autowrap off, the
// hardware cursor hidden, mouse reporting), keys.ts and stdin-buffer.ts (legacy, xterm and Kitty
// sequences, bracketed paste, the timeouts for split sequences). stdin-buffer.ts is itself based
// on OpenTUI (MIT, Copyright (c) 2025 opentui).
#include "code_tui.h"

#include <math.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

static struct termios g_saved;
static volatile sig_atomic_t g_raw, g_alt, g_kitty_pushed, g_mok, g_mouse, g_inline;
static char g_inline_off[32]; // erases the loader's rows (prepared before they are drawn)
static size_t g_inline_off_len;

static void out_all(const char *p, size_t n) {
    while (n) {
        ssize_t w = write(1, p, n);
        if (w <= 0) return;
        p += w;
        n -= (size_t)w;
    }
}

#define OUTS(s) out_all(s, sizeof(s) - 1)

static const char k_disable_mouse[] = "\x1b[?1006l\x1b[?1004l\x1b[?1003l\x1b[?1002l\x1b[?1000l";

// Async-signal-safe: only write(2) and tcsetattr(3) on state set up beforehand.
void screen_restore_signal_safe(void) {
    if (g_inline) out_all(g_inline_off, g_inline_off_len);
    g_inline = 0;
    if (g_alt) {
        OUTS("\x1b[?2026l");
        if (g_mouse) OUTS(k_disable_mouse);
        // Kitty keyboard flags are per screen: popped before the alternate screen is left
        if (g_kitty_pushed) OUTS("\x1b[<u");
        if (g_mok) OUTS("\x1b[>4;0m");
        OUTS("\x1b[?2004l\x1b[0m\x1b[?7h\x1b[?25h\x1b[?1049l");
    }
    if (g_raw) tcsetattr(0, TCSANOW, &g_saved);
    g_alt = g_raw = g_kitty_pushed = g_mok = g_mouse = 0;
}

static void on_crash(int sig) {
    screen_restore_signal_safe();
    signal(sig, SIG_DFL);
    raise(sig);
}

static void at_exit(void) { screen_restore_signal_safe(); }

static void install_handlers(void) {
    static bool done;
    if (done) return;
    done = true;
    struct sigaction sa = {0};
    sa.sa_handler = on_crash;
    sa.sa_flags = (int)SA_RESETHAND;
    sigemptyset(&sa.sa_mask);
    static const int sigs[] = {SIGSEGV, SIGBUS, SIGABRT, SIGFPE, SIGILL, SIGTRAP};
    for (size_t i = 0; i < COUNT(sigs); i++) sigaction(sigs[i], &sa, NULL);
    atexit(at_exit);
}

static void on_quiet_int(int sig) {
    screen_restore_signal_safe();
    signal(sig, SIG_DFL);
    raise(sig);
}

void screen_quiet_start(int inline_rows) {
    install_handlers();
    if (tcgetattr(0, &g_saved) == 0) {
        struct termios t = g_saved;
        t.c_lflag &= ~(tcflag_t)(ICANON | ECHO); // typed keys do not echo over the loader; Ctrl+C still signals
        tcsetattr(0, TCSANOW, &t);
        g_raw = 1;
    }
    struct sigaction cur;
    if (sigaction(SIGINT, NULL, &cur) == 0 && cur.sa_handler == SIG_DFL) {
        struct sigaction sa = {0};
        sa.sa_handler = on_quiet_int;
        sa.sa_flags = (int)SA_RESETHAND;
        sigaction(SIGINT, &sa, NULL);
    }
    if (inline_rows > 0) { // drawn under the cursor: a signal erases those rows (no alternate screen)
        int n = snprintf(g_inline_off, sizeof g_inline_off, "\x1b[0m\x1b[%dF\x1b[J\x1b[?25h", inline_rows);
        g_inline_off_len = n > 0 ? (size_t)n : 0;
        g_inline = 1;
        return;
    }
    OUTS("\x1b[?1049h\x1b[?25l\x1b[2J");
    g_alt = 1;
}

void screen_quiet_stop(void) {
    if (g_alt) OUTS("\x1b[0m\x1b[2J\x1b[?1049l\x1b[?25h");
    g_alt = 0;
    g_inline = 0; // the loader erased its rows itself
    if (g_raw) tcsetattr(0, TCSAFLUSH, &g_saved); // keys typed meanwhile are dropped
    g_raw = 0;
    struct sigaction cur;
    if (sigaction(SIGINT, NULL, &cur) == 0 && cur.sa_handler == on_quiet_int) signal(SIGINT, SIG_DFL);
}

void screen_size(Screen *s) {
    struct winsize ws;
    if (ioctl(1, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0 && ws.ws_row > 0) {
        s->cols = ws.ws_col;
        s->rows = ws.ws_row;
    } else {
        s->cols = 80;
        s->rows = 24;
    }
}

void screen_start(Screen *s) {
    install_handlers();
    if (tcgetattr(0, &g_saved) == 0) {
        struct termios t = g_saved;
        t.c_lflag &= ~(tcflag_t)(ICANON | ECHO | ISIG | IEXTEN);
        t.c_iflag &= ~(tcflag_t)(IXON | ICRNL | INLCR | IGNCR);
        t.c_oflag &= ~(tcflag_t)OPOST;
        t.c_cc[VMIN] = 1;
        t.c_cc[VTIME] = 0;
        tcsetattr(0, TCSANOW, &t);
        g_raw = 1;
    }
    screen_size(s);
    // TuiAltScreen: the alternate screen, autowrap off, mouse reporting (button motion only under
    // a multiplexer), the screen cleared, the cursor hidden
    const char *term = getenv("TERM");
    bool mux = getenv("TMUX") || getenv("ZELLIJ") || getenv("STY") || (term && (!strncasecmp(term, "tmux", 4) || !strncasecmp(term, "screen", 6)));
    OUTS("\x1b[?1049h\x1b[?7l");
    g_alt = 1;
    if (mux) OUTS("\x1b[?1000h\x1b[?1002h\x1b[?1004h\x1b[?1006h");
    else OUTS("\x1b[?1000h\x1b[?1002h\x1b[?1003h\x1b[?1004h\x1b[?1006h");
    g_mouse = 1;
    OUTS("\x1b[2J\x1b[H\x1b[?25l");
    // ProcessTerminal.start: bracketed paste, then the Kitty flags pushed and queried with the
    // device attributes as the sentinel (no Kitty answer before it: modifyOtherKeys)
    OUTS("\x1b[?2004h");
    OUTS("\x1b[>7u\x1b[?u\x1b[c");
    g_kitty_pushed = 1;
    s->kitty = false;
    s->pending_da++;
    // the theme's colors (tui.ts TERMINAL_COLOR_QUERY): OSC 10, 11, 4;0..15, then DA1 to end it
    OUTS("\x1b]10;?\x07\x1b]11;?\x07");
    for (int i = 0; i < 16; i++) {
        char q[16];
        int n = snprintf(q, sizeof q, "\x1b]4;%d;?\x07", i);
        out_all(q, (size_t)n);
    }
    OUTS("\x1b[c");
    s->color_query = true;
    s->color_query_ns = now_ns();
    s->colors = (TermColors){0};
    s->palette_got = 0;
    s->started = true;
    lines_free(&s->prev);
}

void screen_reply(Screen *s, const Key *k) {
    if (k->kind == K_REPLY_KITTY) {
        if (k->ch) {
            if (g_mok) OUTS("\x1b[>4;0m");
            g_mok = 0;
            s->kitty = true;
        } else if (!s->kitty && !g_mok) {
            OUTS("\x1b[>4;2m");
            g_mok = 1;
        }
    } else if (k->kind == K_REPLY_DA) {
        // the first owed DA ends the keyboard query, the next one the color query
        if (s->pending_da > 0) {
            s->pending_da--;
            if (!s->kitty && !g_mok) {
                OUTS("\x1b[>4;2m");
                g_mok = 1;
            }
        } else if (s->color_query) s->color_done = true;
    } else if (k->kind == K_REPLY_COLOR && !k->release) {
        uint8_t rgb[3] = {(uint8_t)k->button, (uint8_t)k->mx, (uint8_t)k->my};
        if (k->ch == 10) {
            s->colors.has_fg = true;
            memcpy(s->colors.fg, rgb, 3);
        } else if (k->ch == 11) {
            s->colors.has_bg = true;
            memcpy(s->colors.bg, rgb, 3);
        } else if (k->ch >= 100 && k->ch < 116) {
            memcpy(s->colors.palette[k->ch - 100], rgb, 3);
            s->palette_got |= 1u << (k->ch - 100);
            s->colors.has_palette = s->palette_got == 0xffffu;
        }
        if (s->colors.has_fg && s->colors.has_bg && s->colors.has_palette) s->color_done = true;
    }
}

void screen_stop(Screen *s) {
    if (!s->started) return;
    screen_restore_signal_safe();
    s->started = false;
    s->kitty = false;
    lines_free(&s->prev);
}

bool screen_suspend(Screen *s) {
    screen_stop(s);
    // pi's handleCtrlZ: the terminal restored, SIGTSTP to the process group; the kill returns
    // once the shell continues the job
    struct sigaction dfl = {0}, old;
    dfl.sa_handler = SIG_DFL;
    sigaction(SIGTSTP, &dfl, &old);
    kill(0, SIGTSTP);
    sigaction(SIGTSTP, &old, NULL);
    screen_start(s);
    return true;
}


void screen_draw(Screen *s, Lines *frame, bool force) {
    Buf b = {0};
    buf_adds(&b, "\x1b[?2026h");
    bool any = false;
    for (int row = 0; row < s->rows; row++) {
        const char *line = (size_t)row < frame->n ? frame->v[row] : "";
        const char *old = (size_t)row < s->prev.n ? s->prev.v[row] : NULL;
        if (!force && old && !strcmp(old, line)) continue;
        any = true;
        buf_printf(&b, "\x1b[%d;1H\x1b[2K\x1b[0m", row + 1);
        Str l = str(line);
        if (tui_width(l) > s->cols) {
            Buf cut = {0};
            tui_slice(&cut, l, 0, s->cols, true);
            buf_addstr(&b, buf_str(&cut));
            buf_free(&cut);
        } else buf_addstr(&b, l);
        buf_adds(&b, "\x1b[0m");
    }
    // the hardware cursor stays hidden (the editor draws its own), every frame, as pi's does
    buf_adds(&b, "\x1b[?25l\x1b[?2026l");
    if (any) out_all(b.ptr, b.len);
    buf_free(&b);
    // the frame is kept as the screen's copy (its lines move over)
    lines_free(&s->prev);
    for (int row = 0; row < s->rows; row++) {
        if ((size_t)row < frame->n) {
            lines_push(&s->prev, frame->v[row]);
            frame->v[row] = NULL;
        } else lines_addz(&s->prev, "");
    }
    for (size_t row = (size_t)s->rows; row < frame->n; row++) free(frame->v[row]);
    frame->n = 0;
}

// ---- keys ------------------------------------------------------------------------------------------

typedef struct {
    int v[8][4]; // parameters and their ':' sub-parameters (-1: absent)
    int n;
    char prefix; // '<', '=', '>', '?' or 0
    char final;
} Csi;

static void csi_parse(const char *p, size_t len, Csi *c) {
    memset(c, 0, sizeof *c);
    for (int i = 0; i < 8; i++)
        for (int k = 0; k < 4; k++) c->v[i][k] = -1;
    size_t q = 0;
    if (q < len && strchr("<=>?", p[q])) c->prefix = p[q++];
    int pi = 0, sub = 0;
    bool any = false;
    for (; q < len; q++) {
        char ch = p[q];
        if (ch >= '0' && ch <= '9') {
            if (pi < 8 && sub < 4) c->v[pi][sub] = (c->v[pi][sub] < 0 ? 0 : c->v[pi][sub]) * 10 + (ch - '0');
            any = true;
        } else if (ch == ':') sub++;
        else if (ch == ';') {
            pi++;
            sub = 0;
            any = true;
        }
    }
    c->n = any ? MIN(pi + 1, 8) : 0;
}

static int csi_num(const Csi *c, int i, int dflt) { return i < c->n && c->v[i][0] >= 0 ? c->v[i][0] : dflt; }

// The modifier bits (shift 1, alt 2, ctrl 4, super 8) of an xterm / Kitty modifier parameter,
// the lock keys masked off.
static int mod_bits(int m) { return m > 0 ? (m - 1) & ~(64 | 128) : 0; }

static KeyKind csi_key(char final, int p1, int mods) {
    bool ctrl = mods & 4, alt = mods & 2;
    switch (final) {
    case 'A': return ctrl ? K_CTRL_UP : alt ? K_ALT_UP : K_UP;
    case 'B': return ctrl ? K_CTRL_DOWN : alt ? K_ALT_DOWN : K_DOWN;
    case 'C': return ctrl ? K_CTRL_RIGHT : alt ? K_ALT_RIGHT : K_RIGHT;
    case 'D': return ctrl ? K_CTRL_LEFT : alt ? K_ALT_LEFT : K_LEFT;
    case 'H': return ctrl ? K_CTRL_HOME : K_HOME;
    case 'F': return ctrl ? K_CTRL_END : K_END;
    case 'Z': return K_SHIFT_TAB;
    case '~':
        switch (p1) {
        case 1:
        case 7: return ctrl ? K_CTRL_HOME : K_HOME;
        case 4:
        case 8: return ctrl ? K_CTRL_END : K_END;
        case 3: return K_DELETE;
        case 5: return K_PGUP;
        case 6: return K_PGDN;
        default: return K_NONE;
        }
    default: return K_NONE;
    }
}

// A key given by its code point and modifiers (Kitty CSI u, xterm modifyOtherKeys).
static void code_key(Key *k, int code, int shifted, int mods) {
    bool shift = mods & 1, alt = mods & 2, ctrl = mods & 4;
    if (mods & 8) return; // super: the terminal's own shortcuts
    switch (code) {
    case 13:
    case 57414: k->kind = shift ? K_SHIFT_ENTER : alt ? K_ALT_ENTER : K_ENTER; return;
    case 27: k->kind = K_ESC; return;
    case 9: k->kind = shift ? K_SHIFT_TAB : K_TAB; return;
    case 127:
    case 8: k->kind = alt ? K_ALT_BACKSPACE : ctrl ? K_CTRL_BACKSPACE : K_BACKSPACE; return;
    default: break;
    }
    if (code >= 57344 && code <= 63743) return; // Kitty's functional keys (keypad, media, lone modifiers)
    if (ctrl && code < 0x80 && code > ' ') {
        k->kind = K_CTRL;
        k->ch = (uint32_t)(code >= 'A' && code <= 'Z' ? code + 32 : code);
        k->shift = shift;
        return;
    }
    if (alt && code < 0x80) {
        k->kind = code == 'b' ? K_ALT_LEFT : code == 'f' ? K_ALT_RIGHT : K_ALT;
        k->ch = (uint32_t)code;
        return;
    }
    if (code < ' ') return;
    k->kind = K_CHAR;
    k->ch = (uint32_t)(shift && shifted > 0 ? shifted : shift && code >= 'a' && code <= 'z' ? code - 32 : code);
}

// The end of an OSC / DCS / APC string at p (BEL or ESC \), or 0 while it may go on. As pi's
// stdin buffer: CAN or SUB cancels it (consumed), and an ESC not followed by \ ends it there (the
// ESC starts the next key), so an unterminated reply never swallows the keys after it.
static size_t string_end(const char *p, size_t n) {
    for (size_t j = 2; j < n; j++) {
        if (p[j] == 0x07) return j + 1;
        if (p[j] == 0x18 || p[j] == 0x1a) return j + 1;
        if (p[j] == 0x1b) {
            if (j + 1 >= n) return 0;
            return p[j + 1] == '\\' ? j + 2 : j;
        }
    }
    return 0;
}

// A hex channel of an OSC color ("ff", "ffff"): scaled to 0..255 as pi does (Math.round).
static int osc_channel(const char *p, size_t n) {
    if (!n || n > 8) return -1;
    double v = 0, max = 1;
    for (size_t i = 0; i < n; i++) {
        char c = p[i];
        int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
        if (d < 0) return -1;
        v = v * 16 + d;
        max *= 16;
    }
    return (int)floor(v / (max - 1) * 255 + 0.5);
}

// OSC 10 / 11 / 4;n color replies (terminal-colors.ts parseOscColorResponse).
static void osc_color(const char *p, size_t e, Key *k) {
    size_t end = p[e - 1] == 0x07 ? e - 1 : e - 2;
    Str body = str_n(p + 2, end - 2);
    int target;
    size_t i = 0;
    if (str_starts(body, S("10;"))) target = 10, i = 3;
    else if (str_starts(body, S("11;"))) target = 11, i = 3;
    else if (str_starts(body, S("4;"))) {
        i = 2;
        int idx = 0, nd = 0;
        while (i < body.len && body.ptr[i] >= '0' && body.ptr[i] <= '9' && nd < 3) idx = idx * 10 + (body.ptr[i++] - '0'), nd++;
        if (!nd || i >= body.len || body.ptr[i] != ';') return;
        i++;
        target = 100 + idx;
    } else return;
    Str v = str_trim(str_sub(body, i, body.len));
    k->kind = K_REPLY_COLOR;
    k->ch = (uint32_t)target;
    k->release = true; // unparsable until shown otherwise
    int rgb[3] = {-1, -1, -1};
    if (v.len && v.ptr[0] == '#') {
        if (v.len == 7)
            for (int c = 0; c < 3; c++) rgb[c] = osc_channel(v.ptr + 1 + 2 * c, 2);
        else if (v.len == 13)
            for (int c = 0; c < 3; c++) rgb[c] = osc_channel(v.ptr + 1 + 4 * c, 4);
    } else {
        if (str_starts(v, S("rgba:")) || str_starts(v, S("RGBA:"))) v = str_sub(v, 5, v.len);
        else if (str_starts(v, S("rgb:")) || str_starts(v, S("RGB:"))) v = str_sub(v, 4, v.len);
        size_t q = 0;
        for (int c = 0; c < 3; c++) {
            size_t st = q;
            while (q < v.len && v.ptr[q] != '/') q++;
            rgb[c] = osc_channel(v.ptr + st, q - st);
            q++;
        }
    }
    if (rgb[0] < 0 || rgb[1] < 0 || rgb[2] < 0) return;
    k->release = false;
    k->button = rgb[0];
    k->mx = rgb[1];
    k->my = rgb[2];
}

size_t key_parse(const char *p, size_t n, bool flush, Key *k) {
    memset(k, 0, sizeof *k);
    if (!n) return 0;
    unsigned char c = (unsigned char)p[0];
    if (c == 0x1b) {
        if (n == 1) {
            if (!flush) return 0;
            k->kind = K_ESC;
            return 1;
        }
        unsigned char c1 = (unsigned char)p[1];
        if (c1 == '[') {
            if (n >= 3 && p[2] == 'M') { // X10 mouse: ESC [ M b x y
                if (n < 6) return flush ? n : 0;
                k->kind = K_MOUSE;
                k->button = (unsigned char)p[3] - 32;
                k->mx = (unsigned char)p[4] - 33;
                k->my = (unsigned char)p[5] - 33;
                k->release = (k->button & 3) == 3;
                return 6;
            }
            size_t j = 2;
            while (j < n && !((unsigned char)p[j] >= 0x40 && (unsigned char)p[j] <= 0x7e)) j++;
            if (j >= n) return flush ? n : 0; // split: wait for the rest (a stale one is dropped)
            Csi cs;
            csi_parse(p + 2, j - 2, &cs);
            cs.final = p[j];
            size_t used = j + 1;
            if (cs.final == '~' && csi_num(&cs, 0, 0) == 200) { // a paste start the reader did not take
                k->kind = K_NONE;
                return used;
            }
            if (cs.prefix == '<' && (cs.final == 'M' || cs.final == 'm')) { // SGR mouse
                k->kind = K_MOUSE;
                k->button = csi_num(&cs, 0, 0);
                k->mx = csi_num(&cs, 1, 1) - 1;
                k->my = csi_num(&cs, 2, 1) - 1;
                k->release = cs.final == 'm';
                return used;
            }
            if (cs.prefix == '?' && cs.final == 'u') {
                k->kind = K_REPLY_KITTY;
                k->ch = (uint32_t)csi_num(&cs, 0, 0);
                return used;
            }
            if (cs.prefix == '?' && cs.final == 'c') {
                k->kind = K_REPLY_DA;
                return used;
            }
            if (cs.prefix) return used; // other replies (cursor position, modes)
            if (!cs.n && (cs.final == 'I' || cs.final == 'O')) return used; // focus in / out
            // the event type (Kitty flag 2): 3 is a release
            if (cs.n >= 2 && cs.v[1][1] == 3) return used;
            int mods = mod_bits(csi_num(&cs, 1, 1));
            if (cs.final == 'u') {
                int code = csi_num(&cs, 0, 0);
                code_key(k, code, cs.v[0][1], mods);
                if (k->kind == K_CHAR && !mods) k->kitty_text = true;
                return used;
            }
            if (cs.final == '~' && csi_num(&cs, 0, 0) == 27 && cs.n >= 3) { // modifyOtherKeys: 27;mod;code~
                code_key(k, csi_num(&cs, 2, 0), -1, mod_bits(csi_num(&cs, 1, 1)));
                return used;
            }
            if (cs.final == '~' && csi_num(&cs, 0, 0) == 13 && (mods & 1)) {
                k->kind = K_SHIFT_ENTER;
                return used;
            }
            k->kind = csi_key(cs.final, csi_num(&cs, 0, 1), mods);
            return used;
        }
        if (c1 == 'O') { // SS3
            if (n < 3) return flush ? n : 0;
            k->kind = csi_key(p[2], 1, 0);
            return 3;
        }
        if (c1 == ']' || c1 == 'P' || c1 == '_') { // OSC, DCS, APC: terminal replies
            size_t e = string_end(p, n);
            if (e && c1 == ']' && (p[e - 1] == 0x07 || p[e - 1] == '\\')) osc_color(p, e, k);
            if (e) return e;
            if (!flush) return 0;
            // a lone Alt+] (and the like): the key, then what follows it as input
            k->kind = K_ALT;
            k->ch = c1;
            return 2;
        }
        // ESC + key: alt
        if (c1 == 0x7f || c1 == 0x08) {
            k->kind = K_ALT_BACKSPACE;
            return 2;
        }
        if (c1 == '\r') {
            k->kind = K_ALT_ENTER;
            return 2;
        }
        if (c1 == 0x1b) {
            k->kind = K_ESC;
            return 1;
        }
        if (c1 == 'b') {
            k->kind = K_ALT_LEFT;
            return 2;
        }
        if (c1 == 'f') {
            k->kind = K_ALT_RIGHT;
            return 2;
        }
        if (c1 >= 0x80) {
            size_t len = c1 >= 0xf0 ? 4 : c1 >= 0xe0 ? 3 : c1 >= 0xc0 ? 2 : 1;
            if (1 + len > n) return flush ? n : 0;
            size_t i = 0;
            k->kind = K_ALT;
            k->ch = tui_cp(str_n(p + 1, len), &i);
            return 1 + len;
        }
        k->kind = K_ALT;
        k->ch = c1;
        return 2;
    }
    if (c == '\r') {
        k->kind = K_ENTER;
        return 1;
    }
    if (c == '\t') {
        k->kind = K_TAB;
        return 1;
    }
    if (c == 0x7f || c == 0x08) {
        k->kind = K_BACKSPACE;
        return 1;
    }
    if (c < 0x20) {
        k->kind = K_CTRL;
        k->ch = c == 0 ? ' ' : c <= 26 ? 'a' + c - 1 : c == 0x1c ? '\\' : c == 0x1d ? ']' : c == 0x1e ? '^' : '_';
        return 1;
    }
    size_t len = c < 0x80 ? 1 : c >= 0xf0 ? 4 : c >= 0xe0 ? 3 : c >= 0xc0 ? 2 : 1;
    if (len > n) return flush ? n : 0; // split UTF-8: kept for the next read
    size_t i = 0;
    k->kind = K_CHAR;
    k->ch = tui_cp(str_n(p, len), &i);
    return len;
}

// ---- the input buffer (stdin-buffer.ts) -------------------------------------------------------------

static const char k_paste_start[] = "\x1b[200~", k_paste_end[] = "\x1b[201~";

static int esc_timeout_ms(void) {
    static int ms = -1;
    if (ms < 0) {
        const char *e = getenv("PI_TUI_ESC_TIMEOUT");
        int v = e ? atoi(e) : 0;
        ms = v > 0 ? v : (getenv("SSH_CONNECTION") || getenv("SSH_TTY")) ? 100 : 10;
    }
    return ms;
}

// A recognized terminal reply header, exactly pi's TERMINAL_REPLY_HEADER_PATTERN:
// ESC ] (1[01]; | 4;\d{1,3}; | 7501;?) | ESC P >| | ESC _ G ([a-z]=\d+,)* [a-z]=\d+;
static bool reply_header(Str b) {
    if (b.len < 3 || b.ptr[0] != 0x1b) return false;
    if (b.ptr[1] == ']') {
        Str r = str_sub(b, 2, b.len);
        if (str_starts(r, S("10;")) || str_starts(r, S("11;")) || str_starts(r, S("7501;?"))) return true;
        if (!str_starts(r, S("4;"))) return false;
        size_t i = 2, d = 0;
        while (i < r.len && r.ptr[i] >= '0' && r.ptr[i] <= '9' && d < 4) i++, d++;
        return d >= 1 && d <= 3 && i < r.len && r.ptr[i] == ';';
    }
    if (b.ptr[1] == 'P') return b.len >= 4 && b.ptr[2] == '>' && b.ptr[3] == '|';
    if (b.ptr[1] == '_') {
        if (b.ptr[2] != 'G') return false;
        size_t i = 3;
        for (;;) { // [a-z]=\d+ then ',' (more) or ';' (the end of the header)
            if (i >= b.len || b.ptr[i] < 'a' || b.ptr[i] > 'z') return false;
            if (++i >= b.len || b.ptr[i] != '=') return false;
            size_t d = 0;
            while (++i < b.len && b.ptr[i] >= '0' && b.ptr[i] <= '9') d++;
            if (!d || i >= b.len) return false;
            if (b.ptr[i] == ';') return true;
            if (b.ptr[i] != ',') return false;
            i++;
        }
    }
    return false;
}

void keyin_feed(KeyIn *in, const char *p, size_t n) {
    buf_add(&in->buf, p, n);
    in->pending_ns = 0;
}

int keyin_timeout_ms(const KeyIn *in, uint64_t now) {
    if (!in->buf.len || in->paste) return -1;
    Str b = buf_str(&in->buf);
    // a terminal reply waits for its end, up to a second (a stuck one cannot hold the keys forever)
    int ms = reply_header(b) ? 1000 : b.len == 1 && b.ptr[0] == 0x1b ? esc_timeout_ms() : 50;
    if (!in->pending_ns) return ms;
    uint64_t due = in->pending_ns + (uint64_t)ms * 1000000ull;
    return now >= due ? 0 : (int)((due - now + 999999) / 1000000);
}

static void consume(KeyIn *in, size_t n) {
    memmove(in->buf.ptr, in->buf.ptr + n, in->buf.len - n);
    in->buf.len -= n;
    if (in->buf.ptr) in->buf.ptr[in->buf.len] = 0;
}

bool keyin_next(KeyIn *in, uint64_t now, Key *k) {
    memset(k, 0, sizeof *k);
    for (;;) {
        if (in->paste) {
            // everything up to the end marker, however long
            Str b = buf_str(&in->buf);
            ptrdiff_t e = str_find(b, S(k_paste_end), 0);
            if (e < 0) {
                size_t keep = MIN(b.len, sizeof k_paste_end - 2); // a split end marker
                buf_add(&in->paste_buf, b.ptr, b.len - keep);
                consume(in, b.len - keep);
                return false;
            }
            buf_add(&in->paste_buf, b.ptr, (size_t)e);
            consume(in, (size_t)e + sizeof k_paste_end - 1);
            in->paste = false;
            in->last_kitty_cp = 0;
            k->kind = K_PASTE;
            k->paste = in->paste_buf.ptr ? in->paste_buf.ptr : strdup("");
            in->paste_buf = (Buf){0};
            return true;
        }
        if (!in->buf.len) return false;
        Str b = buf_str(&in->buf);
        if (b.len >= sizeof k_paste_start - 1 && !memcmp(b.ptr, k_paste_start, sizeof k_paste_start - 1)) {
            consume(in, sizeof k_paste_start - 1);
            in->paste = true;
            continue;
        }
        if (b.len < sizeof k_paste_start - 1 && !memcmp(b.ptr, k_paste_start, b.len)) {
            // maybe a paste start split across reads
            if (!in->pending_ns) in->pending_ns = now;
            if (keyin_timeout_ms(in, now) != 0) return false;
        }
        size_t used = key_parse(b.ptr, b.len, false, k);
        if (!used) {
            if (!in->pending_ns) in->pending_ns = now;
            if (keyin_timeout_ms(in, now) != 0) return false;
            used = key_parse(b.ptr, b.len, true, k);
            if (!used) used = b.len;
        }
        in->pending_ns = 0;
        // a Kitty printable key the terminal also sent as text is taken once
        bool dup = k->kind == K_CHAR && !k->kitty_text && in->last_kitty_cp && k->ch == in->last_kitty_cp;
        in->last_kitty_cp = k->kind == K_CHAR && k->kitty_text ? k->ch : 0;
        consume(in, used);
        if (dup) continue;
        return true;
    }
}

void keyin_free(KeyIn *in) {
    buf_free(&in->buf);
    buf_free(&in->paste_buf);
}
