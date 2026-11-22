// Saetta Code's loader: while the model loads (or a download or a server is awaited), Saetta's
// bolt turns on the alternate screen in ASCII, shaded with a character ramp like donut.c: the
// logo's outline extruded, sampled surface points with normals, rotated, projected into the cell
// grid with a depth buffer, each cell's character picked by the light it catches; a faint field
// of drifting ASCII noise around it; a flash every few seconds. It draws from its own thread at
// 25 frames a second, so the load runs at full speed, and leaves the screen as it found it.
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
    const double D = 0.13;
    size_t cap = 0;
    // coarser than the web page: a terminal cell holds many samples
    for (double x = -1.3; x <= 1.3; x += 0.03)
        for (double y = -1.3; y <= 1.3; y += 0.03)
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
        for (double t = 0; t <= len; t += 0.02)
            for (double z = -D; z <= D; z += 0.04) push(s, &cap, ax + (bx - ax) * t / len, ay + (by - ay) * t / len, z, nx, ny, 0);
    }
}

// value noise for the field
static uint8_t perm[512];
static void noise_init(void) {
    uint8_t p[256];
    for (int i = 0; i < 256; i++) p[i] = (uint8_t)i;
    long long seed = 9;
    for (int i = 255; i > 0; i--) {
        seed = seed * 16807 % 2147483647;
        int j = (int)(seed % (i + 1));
        uint8_t t = p[i];
        p[i] = p[j];
        p[j] = t;
    }
    for (int i = 0; i < 512; i++) perm[i] = p[i & 255];
}
static double hash2(int x, int y) { return perm[(perm[x & 255] + y) & 511] / 255.0; }
static double noise(double x, double y) {
    int xi = (int)floor(x), yi = (int)floor(y);
    double xf = x - xi, yf = y - yi, u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
    double a = hash2(xi, yi), b = hash2(xi + 1, yi), c = hash2(xi, yi + 1), d = hash2(xi + 1, yi + 1);
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}

static const char RAMP[] = ".,-~:;=!*#$@";
static const char FIELD[] = " .:-=+";

static void color(Buf *b, bool tc, int r, int g, int bl) {
    if (tc) buf_printf(b, "\x1b[38;2;%d;%d;%dm", r, g, bl);
    else {
        int q = 16 + 36 * (r * 5 / 255) + 6 * (g * 5 / 255) + bl * 5 / 255;
        buf_printf(b, "\x1b[38;5;%dm", q);
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

static void frame(Splash *s, double t, double flash, Buf *b) {
    struct winsize ws;
    int tw = 80, thh = 24;
    if (ioctl(1, TIOCGWINSZ, &ws) == 0 && ws.ws_col && ws.ws_row) tw = ws.ws_col, thh = ws.ws_row;
    int cols = MIN(46, tw - 2), rows = MIN(22, thh - 3);
    if (cols < 12 || rows < 6) return;
    // a cell is about twice as tall as it is wide
    double cw = 1, ch = 2.05;
    float zb[24 * 48];
    float lum[24 * 48];
    for (int i = 0; i < rows * cols; i++) zb[i] = 0, lum[i] = -1;
    // turning to and fro, never edge-on (the cell grid is coarse)
    double A = sin(t * 0.55) * 0.75 + sin(t * 0.17) * 0.25, B = 0.18 + sin(t * 0.29) * 0.12 + sin(t * 0.71) * 0.05;
    double cA = cos(A), sA = sin(A), cB = cos(B), sB = sin(B);
    const double K2 = 4.2;
    double H = rows * ch, W = cols * cw;
    double boltH = H * 0.92;
    double K1 = boltH * K2 / 2.3;
    double cx = W / 2, cy = H / 2;
    static const double LP[3] = {-1.9, 1.7, -2.4};
    for (size_t i = 0; i < s->npts; i++) {
        const float *o = s->pts + i * 6;
        double x = o[0], y = o[1], z = o[2];
        double x1 = x * cA + z * sA, z1 = -x * sA + z * cA;
        double y2 = y * cB - z1 * sB, z2 = y * sB + z1 * cB;
        double nx = o[3], ny = o[4], nz = o[5];
        double nx1 = nx * cA + nz * sA, nz1 = -nx * sA + nz * cA;
        double ny2 = ny * cB - nz1 * sB, nz2 = ny * sB + nz1 * cB;
        double ooz = 1 / (K2 + z2);
        int col = (int)floor((cx + K1 * x1 * ooz) / cw), row = (int)floor((cy - K1 * y2 * ooz) / ch);
        if (col < 0 || row < 0 || col >= cols || row >= rows) continue;
        int k = row * cols + col;
        if (ooz > zb[k]) {
            zb[k] = (float)ooz;
            double lx = LP[0] - x1, ly = LP[1] - y2, lz = LP[2] - z2, ln = sqrt(lx * lx + ly * ly + lz * lz);
            double d = (nx1 * lx + ny2 * ly + nz2 * lz) / ln;
            lum[k] = (float)(MAX(0.0, d) * MIN(1.7, 11 / (ln * ln)) - 0.04);
        }
    }
    int top = (thh - rows - 2) / 2, left = (tw - cols) / 2;
    buf_adds(b, "\x1b[?2026h");
    for (int r = 0; r < rows; r++) {
        buf_printf(b, "\x1b[%d;%dH", top + r + 1, left + 1);
        int last = -1; // the color state: -1 none, 0 field, 1.. bolt band
        for (int c = 0; c < cols; c++) {
            int k = r * cols + c;
            double l = lum[k];
            if (l > -1) {
                double v = MAX(0.0, MIN(0.999, l));
                char g = RAMP[(int)(v * (sizeof RAMP - 1))];
                int band = 1 + MIN(4, (int)(v * 5));
                if (band != last) {
                    // the warm gradient from the dark side (orange) to the lit one (yellow); white-yellow in a flash
                    static const int warm[5][3] = {{0xd9, 0x5a, 0x10}, {0xff, 0x7a, 0x1a}, {0xff, 0x96, 0x1e}, {0xff, 0xb0, 0x20}, {0xff, 0xe1, 0x4d}};
                    const int *w = warm[band - 1];
                    if (flash > 0.05 && band >= 4) color(b, s->truecolor, 0xff, 0xf6, 0xc8);
                    else color(b, s->truecolor, w[0], w[1], w[2]);
                    buf_adds(b, "\x1b[22m");
                    if (band == 5) buf_adds(b, "\x1b[1m");
                    last = band;
                }
                buf_addc(b, g);
            } else {
                double n = noise(c * 0.11 + t * 0.12, r * 0.44 - t * 0.05) * 0.7 + noise(c * 0.31 - t * 0.08, r * 1.2) * 0.3;
                double v = 0.05 + n * 0.85 * (1 + flash * 0.5);
                char g = ' ';
                if (v > 0.62) g = FIELD[MIN((int)sizeof FIELD - 2, 1 + (int)((v - 0.62) * 2.6 * (sizeof FIELD - 2)))];
                if (g != ' ' && last != 0) {
                    buf_adds(b, "\x1b[22m\x1b[2m");
                    color(b, s->truecolor, 0x8e, 0x8e, 0x93);
                    last = 0;
                }
                buf_addc(b, g);
            }
        }
        buf_adds(b, "\x1b[0m");
    }
    // the line under it
    double el = (double)(now_ns() - s->t0) / 1e9;
    char line[320];
    int n = snprintf(line, sizeof line, "%s  %.1fs", s->label, el);
    if (n > tw) n = tw;
    buf_printf(b, "\x1b[%d;1H\x1b[2K\x1b[%dG\x1b[2m%.*s\x1b[0m", top + rows + 2, MAX(1, (tw - n) / 2 + 1), n, line);
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
    Buf v = {0};
    bool reduce = tui_run((char *const *)rm, NULL, S(""), &v, 1000) && v.len && v.ptr[0] == '1';
    buf_free(&v);
    if (reduce) return false;
    return true;
}

Splash *splash_start(const char *label) {
    if (!splash_allowed()) return NULL;
    static bool inited;
    if (!inited) {
        noise_init();
        inited = true;
    }
    Splash *s = xcalloc(1, sizeof *s);
    snprintf(s->label, sizeof s->label, "%s", label ? label : "Loading ...");
    s->t0 = now_ns();
    const char *ct = getenv("COLORTERM");
    s->truecolor = ct && (!strcmp(ct, "truecolor") || !strcmp(ct, "24bit"));
    build(s);
    screen_quiet_start(0);
    if (pthread_create(&s->th, NULL, splash_main, s)) {
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
    // the screen as it was: the alternate screen cleared and left, the cursor and echo back
    screen_quiet_stop();
    free(s->pts);
    free(s);
}
