// Saetta Code TUI: the terminal (raw input, the alternate screen), the frame renderer and the key
// decoder (see code_tui.h).
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi at v1.5.2,
// b4afddd): packages/tui/src/terminal.ts (ProcessTerminal), tui-alt-screen.ts (the fullscreen
// frame: rows written with CUP and erase-line, autowrap off, the hardware cursor hidden while the
// editor draws its own), keys.ts and stdin-buffer.ts (legacy and xterm sequences, bracketed paste).
#include "code_tui.h"

#include <stdlib.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

static struct termios g_saved;
static bool g_raw;

static void out_all(const char *p, size_t n) {
    while (n) {
        ssize_t w = write(1, p, n);
        if (w <= 0) return;
        p += w;
        n -= (size_t)w;
    }
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
    if (tcgetattr(0, &g_saved) == 0) {
        struct termios t = g_saved;
        t.c_lflag &= ~(tcflag_t)(ICANON | ECHO | ISIG | IEXTEN);
        t.c_iflag &= ~(tcflag_t)(IXON | ICRNL | INLCR | IGNCR);
        t.c_oflag &= ~(tcflag_t)OPOST;
        t.c_cc[VMIN] = 1;
        t.c_cc[VTIME] = 0;
        tcsetattr(0, TCSANOW, &t);
        g_raw = true;
    }
    screen_size(s);
    // alternate screen, autowrap off, the screen cleared, the cursor hidden, bracketed paste
    static const char on[] = "\x1b[?1049h\x1b[?7l\x1b[2J\x1b[H\x1b[?25l\x1b[?2004h";
    out_all(on, sizeof on - 1);
    s->started = true;
    lines_free(&s->prev);
}

void screen_stop(Screen *s) {
    if (!s->started) return;
    static const char off[] = "\x1b[?2004l\x1b[0m\x1b[?7h\x1b[?25h\x1b[?1049l";
    out_all(off, sizeof off - 1);
    if (g_raw) tcsetattr(0, TCSANOW, &g_saved);
    g_raw = false;
    s->started = false;
    lines_free(&s->prev);
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
    buf_adds(&b, "\x1b[?2026l");
    if (any) out_all(b.ptr, b.len);
    buf_free(&b);
    lines_free(&s->prev);
    for (int row = 0; row < s->rows; row++) lines_addz(&s->prev, (size_t)row < frame->n ? frame->v[row] : "");
}

// ---- keys ------------------------------------------------------------------------------------------

static KeyKind csi_key(char final, int p1, int mod) {
    bool ctrl = mod > 0 && ((mod - 1) & 4), alt = mod > 0 && ((mod - 1) & 2);
    switch (final) {
    case 'A': return K_UP;
    case 'B': return K_DOWN;
    case 'C': return ctrl ? K_CTRL_RIGHT : alt ? K_ALT_RIGHT : K_RIGHT;
    case 'D': return ctrl ? K_CTRL_LEFT : alt ? K_ALT_LEFT : K_LEFT;
    case 'H': return K_HOME;
    case 'F': return K_END;
    case 'Z': return K_SHIFT_TAB;
    case '~':
        switch (p1) {
        case 1:
        case 7: return K_HOME;
        case 4:
        case 8: return K_END;
        case 3: return K_DELETE;
        case 5: return K_PGUP;
        case 6: return K_PGDN;
        default: return K_NONE;
        }
    default: return K_NONE;
    }
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
        if (n >= 6 && !memcmp(p, "\x1b[200~", 6)) {
            // bracketed paste: everything to the end marker
            for (size_t j = 6; j + 6 <= n; j++)
                if (!memcmp(p + j, "\x1b[201~", 6)) {
                    k->kind = K_PASTE;
                    k->paste = xmalloc(j - 6 + 1);
                    memcpy(k->paste, p + 6, j - 6);
                    k->paste[j - 6] = 0;
                    return j + 6;
                }
            return 0;
        }
        if (p[1] == '[' || p[1] == 'O') {
            size_t j = 2;
            while (j < n && !((unsigned char)p[j] >= 0x40 && (unsigned char)p[j] <= 0x7e)) j++;
            if (j >= n) {
                if (!flush) return 0;
                k->kind = K_ESC;
                return 1;
            }
            int params[4] = {0}, np = 0;
            for (size_t q = 2; q < j && np < 4; q++) {
                if (p[q] >= '0' && p[q] <= '9') params[np] = params[np] * 10 + (p[q] - '0');
                else if (p[q] == ';') np++;
            }
            char final = p[j];
            if (p[1] == '[' && final == 'u') {
                // CSI u (kitty / modifyOtherKeys): codepoint;modifier
                int mod = params[1];
                if (params[0] == 13) k->kind = mod == 2 ? K_SHIFT_ENTER : mod == 3 ? K_ALT_ENTER : K_ENTER;
                else if (params[0] == 27) k->kind = K_ESC;
                else if (params[0] == 9) k->kind = mod == 2 ? K_SHIFT_TAB : K_TAB;
                else if (params[0] == 127) k->kind = mod == 3 ? K_ALT_BACKSPACE : K_BACKSPACE;
                else if (mod > 0 && ((mod - 1) & 4) && params[0] >= 'a' && params[0] <= 'z') {
                    k->kind = K_CTRL;
                    k->ch = (uint32_t)params[0];
                } else {
                    k->kind = K_CHAR;
                    k->ch = (uint32_t)params[0];
                }
                return j + 1;
            }
            if (p[1] == '[' && final == '~' && params[0] == 27 && np >= 2) { // modifyOtherKeys: 27;mod;code~
                if (params[2] == 13) k->kind = params[1] == 2 ? K_SHIFT_ENTER : K_ALT_ENTER;
                return j + 1;
            }
            int mod = np >= 1 ? params[1] : 0;
            if (final == '~' && params[0] == 13 && mod == 2) k->kind = K_SHIFT_ENTER;
            else k->kind = csi_key(final, params[0], mod);
            return j + 1;
        }
        // ESC + key: alt
        if (p[1] == 0x7f || p[1] == 0x08) {
            k->kind = K_ALT_BACKSPACE;
            return 2;
        }
        if (p[1] == '\r') {
            k->kind = K_ALT_ENTER;
            return 2;
        }
        if (p[1] == 0x1b) {
            k->kind = K_ESC;
            return 1;
        }
        if (p[1] == 'b') {
            k->kind = K_ALT_LEFT;
            return 2;
        }
        if (p[1] == 'f') {
            k->kind = K_ALT_RIGHT;
            return 2;
        }
        k->kind = K_ALT;
        k->ch = (unsigned char)p[1];
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
    if (len > n) return flush ? n : 0;
    size_t i = 0;
    k->kind = K_CHAR;
    k->ch = tui_cp(str_n(p, len), &i);
    return len;
}
