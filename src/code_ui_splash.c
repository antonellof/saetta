// Saetta Code's loader: while the model loads (or a download or a server is awaited), Saetta's
// bolt turns in 3D, inline under the command (a few rows, not the whole screen): the logo's
// outline extruded, sampled surface points with normals, rotated, projected into a grid of
// half-cell pixels with a depth buffer, each pixel colored by the light it catches (ambient,
// diffuse, a highlight) and drawn with half blocks; a flash every few seconds; the label and the elapsed time beside it. It draws
// from its own thread at 25 frames a second, so the load runs at full speed, and erases its rows
// when it stops.
// The math is ported from saetta's web header (web/hero.js, the same project).
#include "code_tui.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

struct Splash {
    pthread_t th;
    _Atomic bool stop;
    char label[256];
    uint64_t t0;
    float *pts; // x, y, z, nx, ny, nz
    size_t npts;
    bool truecolor;
};

// The favicon path (32-unit box), centred, y up.
static const double P0[6][2] = {{18.5, 3}, {7, 18}, {14.5, 18}, {12, 29}, {25, 13}, {17.2, 13}};
#define NP 6

static bool inside(const double P[NP][2], double x, double y) {
    bool c = false;
    for (int i = 0, j = NP - 1; i < NP; j = i++) {
        double xi = P[i][0], yi = P[i][1], xj = P[j][0], yj = P[j][1];
        if ((yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c;
    }
    return c;
}

static void push(Splash *s, size_t *cap, double x, double y, double z, double nx, double ny, double nz) {
    if (s->npts + 1 > *cap) {
        *cap = *cap ? *cap * 2 : 4096;
        s->pts = xrealloc(s->pts, *cap * 6 * sizeof(float));
    }
    float *p = s->pts + s->npts * 6;
    p[0] = (float)x, p[1] = (float)y, p[2] = (float)z, p[3] = (float)nx, p[4] = (float)ny, p[5] = (float)nz;
    s->npts++;
}

static void build(Splash *s) {
    double P[NP][2];
    for (int i = 0; i < NP; i++) {
        P[i][0] = (P0[i][0] - 16) / 13;
        P[i][1] = (16 - P0[i][1]) / 13;
    }
    const double D = 0.2;
    size_t cap = 0;
    // coarser than the web page: a terminal cell holds many samples
    for (double x = -1.3; x <= 1.3; x += 0.015)
        for (double y = -1.3; y <= 1.3; y += 0.015)
            if (inside(P, x, y)) {
                push(s, &cap, x, y, -D, 0, 0, -1);
                push(s, &cap, x, y, D, 0, 0, 1);
            }
    double area = 0;
    for (int i = 0, j = NP - 1; i < NP; j = i++) area += P[j][0] * P[i][1] - P[i][0] * P[j][1];
    double orient = area > 0 ? 1 : -1;
    for (int i = 0; i < NP; i++) {
        double ax = P[i][0], ay = P[i][1], bx = P[(i + 1) % NP][0], by = P[(i + 1) % NP][1];
        double len = hypot(bx - ax, by - ay);
        double nx = orient * (by - ay) / len, ny = orient * -(bx - ax) / len;
        for (double t = 0; t <= len; t += 0.012)
            for (double z = -D; z <= D; z += 0.012) push(s, &cap, ax + (bx - ax) * t / len, ay + (by - ay) * t / len, z, nx, ny, 0);
    }
}

static void color(Buf *b, bool tc, bool bg, int r, int g, int bl) {
    if (tc) buf_printf(b, "\x1b[%d;2;%d;%d;%dm", bg ? 48 : 38, r, g, bl);
    else {
        int q = 16 + 36 * ((r * 5 + 127) / 255) + 6 * ((g * 5 + 127) / 255) + (bl * 5 + 127) / 255;
        buf_printf(b, "\x1b[%d;5;%dm", bg ? 48 : 38, q);
    }
}

static void out(const char *p, size_t n) {
    while (n) {
        ssize_t w = write(1, p, n);
        if (w <= 0) return;
        p += w;
        n -= (size_t)w;
    }
}

#define SROWS 13
#define SCOLS 30
#define PH (SROWS * 2)

// The light a surface catches, 0..1.3, to a color: deep amber in the shade, orange, gold, a white
// highlight. Pixels are half cells (square), drawn with upper and lower half blocks.
static uint32_t shade(double l) {
    static const double stop[5] = {0, 0.35, 0.65, 0.95, 1.3};
    static const int col[5][3] = {{0x5a, 0x1e, 0x04}, {0xc4, 0x4c, 0x0a}, {0xff, 0x8c, 0x1a}, {0xff, 0xd2, 0x3c}, {0xff, 0xfa, 0xe0}};
    l = MAX(0.0, MIN(1.3, l));
    int i = 0;
    while (i < 3 && l > stop[i + 1]) i++;
    double f = (l - stop[i]) / (stop[i + 1] - stop[i]);
    int r = (int)(col[i][0] + (col[i + 1][0] - col[i][0]) * f), g = (int)(col[i][1] + (col[i + 1][1] - col[i][1]) * f),
        bl = (int)(col[i][2] + (col[i + 1][2] - col[i][2]) * f);
    return 0x1000000u | (uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)bl;
}

// One frame drawn over the SROWS rows reserved under the cursor: up to their first row, each row
// cleared and drawn, ending where it started (the row after them).
static void frame(Splash *s, double t, double flash, Buf *b) {
    struct winsize ws;
    int tw = 80;
    if (ioctl(1, TIOCGWINSZ, &ws) == 0 && ws.ws_col) tw = ws.ws_col;
    const int W = SCOLS, H = PH;
    float zb[PH * SCOLS];
    uint32_t px[PH * SCOLS]; // 0: empty, else 0x1RRGGBB
    for (int i = 0; i < W * H; i++) zb[i] = 0, px[i] = 0;
    // a slow turn to and fro, wide enough to show the sides, with a little tilt
    double A = sin(t * 0.9) * 1.05 + sin(t * 0.37) * 0.2, B = 0.22 + sin(t * 0.53) * 0.12;
    double cA = cos(A), sA = sin(A), cB = cos(B), sB = sin(B);
    const double K2 = 4.0;
    double K1 = H * 0.98 * K2 / 2.35;
    double cx = W / 2.0, cy = H / 2.0;
    // light from the upper left, in front; the viewer along -z
    double L[3] = {-0.55, 0.6, -0.58};
    double ll = sqrt(L[0] * L[0] + L[1] * L[1] + L[2] * L[2]);
    for (int k = 0; k < 3; k++) L[k] /= ll;
    for (size_t i = 0; i < s->npts; i++) {
        const float *o = s->pts + i * 6;
        double x = o[0], y = o[1], z = o[2];
        double x1 = x * cA + z * sA, z1 = -x * sA + z * cA;
        double y2 = y * cB - z1 * sB, z2 = y * sB + z1 * cB;
        double nx = o[3], ny = o[4], nz = o[5];
        double nx1 = nx * cA + nz * sA, nz1 = -nx * sA + nz * cA;
        double ny2 = ny * cB - nz1 * sB, nz2 = ny * sB + nz1 * cB;
        double ooz = 1 / (K2 + z2);
        int c = (int)floor(cx + K1 * x1 * ooz), r = (int)floor(cy - K1 * y2 * ooz);
        if (c < 0 || r < 0 || c >= W || r >= H) continue;
        int k = r * W + c;
        if (ooz <= zb[k]) continue;
        zb[k] = (float)ooz;
        double d = nx1 * L[0] + ny2 * L[1] + nz2 * L[2];
        // Blinn highlight: half vector between the light and the viewer (0, 0, -1)
        double hx = L[0], hy = L[1], hz = L[2] - 1, hl = sqrt(hx * hx + hy * hy + hz * hz);
        double sp = MAX(0.0, (nx1 * hx + ny2 * hy + nz2 * hz) / hl);
        double l = 0.14 + 0.08 * y + 0.78 * MAX(0.0, d) + 0.55 * pow(sp, 18) + flash * 0.45;
        px[k] = shade(l);
    }
    double el = (double)(now_ns() - s->t0) / 1e9;
    char line[320];
    int n = snprintf(line, sizeof line, "%s  %.1fs", s->label, el);
    int room = tw - W - 5;
    if (n > room) n = MAX(0, room);
    buf_adds(b, "\x1b[?2026h");
    buf_printf(b, "\x1b[%dF", SROWS);
    for (int r = 0; r < SROWS; r++) {
        buf_adds(b, "\x1b[2K ");
        uint32_t fg = 0, bg = 0; // the colors set so far on this row (0: default)
        int pend = 0;
        for (int c = 0; c < W && c + 2 < tw; c++) {
            uint32_t up = px[(2 * r) * W + c], dn = px[(2 * r + 1) * W + c];
            if (!up && !dn) {
                pend++;
                continue;
            }
            if (pend) {
                if (bg) buf_adds(b, "\x1b[49m"), bg = 0;
                for (; pend; pend--) buf_addc(b, ' ');
            }
            // the upper half block in the top pixel's color over the bottom pixel's (or the
            // terminal's background), or the lower half block when only the bottom one is set
            uint32_t f = up ? up : dn, g = up ? dn : 0;
            if (f != fg) color(b, s->truecolor, false, (f >> 16) & 255, (f >> 8) & 255, f & 255), fg = f;
            if (g != bg) {
                if (g) color(b, s->truecolor, true, (g >> 16) & 255, (g >> 8) & 255, g & 255);
                else buf_adds(b, "\x1b[49m");
                bg = g;
            }
            buf_adds(b, up ? "\xe2\x96\x80" : "\xe2\x96\x84");
        }
        buf_adds(b, "\x1b[0m");
        if (r == SROWS / 2 && n > 0) buf_printf(b, "\x1b[%dG\x1b[2m%.*s\x1b[0m", W + 5, n, line);
        buf_adds(b, "\r\n");
    }
    buf_adds(b, "\x1b[?2026l");
}

static void *splash_main(void *arg) {
    Splash *s = arg;
    double t = 0.9, flash = 0, next_flash = 2.5; // a three-quarter view first (hero.js lastT)
    uint64_t prev = now_ns();
    Buf b = {0};
    while (!atomic_load(&s->stop)) {
        uint64_t now = now_ns();
        double dt = MIN(0.1, (double)(now - prev) / 1e9);
        prev = now;
        t += dt;
        if (t > next_flash) {
            flash = 1;
            next_flash = t + 2 + (double)(arc4random() % 1000) / 1000.0 * 1.5;
        }
        flash = MAX(0.0, flash - dt * 2.2);
        b.len = 0;
        frame(s, t, flash, &b);
        out(b.ptr, b.len);
        struct timespec ts = {0, 40000000};
        nanosleep(&ts, NULL);
    }
    buf_free(&b);
    return NULL;
}

// No animation without a terminal, with NO_COLOR, or with macOS "Reduce motion".
static bool splash_allowed(void) {
    if (!isatty(1) || !isatty(0)) return false;
    if (getenv("NO_COLOR") && *getenv("NO_COLOR")) return false;
    const char *term = getenv("TERM");
    if (term && !strcmp(term, "dumb")) return false;
    static const char *const rm[] = {"defaults", "read", "com.apple.universalaccess", "reduceMotion", NULL};
    Buf rv = {0};
    bool reduce = tui_run((char *const *)rm, NULL, S(""), &rv, 1000) && rv.len && rv.ptr[0] == '1';
    buf_free(&rv);
    if (reduce) return false;
    return true;
}

Splash *splash_start(const char *label) {
    if (!splash_allowed()) return NULL;
    Splash *s = xcalloc(1, sizeof *s);
    snprintf(s->label, sizeof s->label, "%s", label ? label : "Loading ...");
    s->t0 = now_ns();
    const char *ct = getenv("COLORTERM");
    s->truecolor = ct && (!strcmp(ct, "truecolor") || !strcmp(ct, "24bit"));
    build(s);
    // no echo while it draws, and its rows erased on a crash, Ctrl+C or SIGTERM
    screen_quiet_start(SROWS);
    // the rows it draws on, reserved under the cursor (scrolling the screen if needed)
    char on[64];
    int k = snprintf(on, sizeof on, "\x1b[?25l\r\x1b[J");
    for (int i = 0; i < SROWS; i++) on[k++] = '\n';
    out(on, (size_t)k);
    if (pthread_create(&s->th, NULL, splash_main, s)) {
        char off[32];
        int m = snprintf(off, sizeof off, "\x1b[%dF\x1b[J\x1b[?25h", SROWS);
        out(off, (size_t)m);
        screen_quiet_stop();
        free(s->pts);
        free(s);
        return NULL;
    }
    return s;
}

void splash_label(Splash *s, const char *label) {
    if (s) snprintf(s->label, sizeof s->label, "%s", label);
}

void splash_stop(Splash *s) {
    if (!s) return;
    atomic_store(&s->stop, true);
    pthread_join(s->th, NULL);
    // its rows erased, the cursor back where the loader started
    char off[40];
    int m = snprintf(off, sizeof off, "\x1b[0m\x1b[%dF\x1b[J\x1b[?25h", SROWS);
    out(off, (size_t)m);
    screen_quiet_stop();
    free(s->pts);
    free(s);
}
