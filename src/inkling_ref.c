// Scalar references for the Inkling 2 Mini kernels (see inkling_ref.h).
#include "inkling_ref.h"

#include <math.h>
#include <string.h>

static float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
static float bf16v(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

void ink_ref_mix(const float *states, size_t sstride, size_t n, const float *partial, const float *query, size_t nq, size_t H,
                 float eps, float *out) {
    const size_t ns = n + 1;
    double r[8];
    for (size_t j = 0; j < ns; j++) {
        const float *x = j < n ? states + j * sstride : partial;
        double ss = 0;
        for (size_t i = 0; i < H; i++) ss += (double)x[i] * x[i];
        r[j] = 1.0 / sqrt(ss / (double)H + eps);
    }
    for (size_t q = 0; q < nq; q++) {
        double s[8], mx = -INFINITY, z = 0;
        for (size_t j = 0; j < ns; j++) {
            const float *x = j < n ? states + j * sstride : partial;
            double d = 0;
            for (size_t i = 0; i < H; i++) d += (double)query[q * H + i] * x[i];
            s[j] = d * r[j] / sqrt((double)H);
            mx = fmax(mx, s[j]);
        }
        for (size_t j = 0; j < ns; j++) z += s[j] = exp(s[j] - mx);
        for (size_t i = 0; i < H; i++) {
            double a = 0;
            for (size_t j = 0; j < ns; j++) a += s[j] / z * r[j] * (j < n ? states[j * sstride + i] : partial[i]);
            out[q * H + i] = (float)a;
        }
    }
}

void ink_ref_rmsnorm(const float *x, const float *w, size_t n, float eps, float *out) {
    double ss = 0;
    for (size_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const double r = 1.0 / sqrt(ss / (double)n + eps);
    for (size_t i = 0; i < n; i++) out[i] = (float)(x[i] * r * (w ? w[i] : 1.0f));
}

void ink_ref_msg_gate(const float *x, const float *m, const uint16_t *D, const float *F, const float *G, size_t H, size_t R,
                      float eps, float *out) {
    double sx = 0, sm = 0;
    for (size_t i = 0; i < H; i++) sx += (double)x[i] * x[i], sm += (double)m[i] * m[i];
    const double rx = 1.0 / sqrt(sx / (double)H + eps), rm = 1.0 / sqrt(sm / (double)H + eps);
    double d[256];
    for (size_t r = 0; r < R; r++) {
        double a = 0;
        for (size_t i = 0; i < H; i++) a += bf16v(D[r * 2 * H + i]) * x[i] * rx + bf16v(D[r * 2 * H + H + i]) * m[i] * rm;
        d[r] = a;
    }
    for (size_t c = 0; c < H; c++) {
        double f = 0, g = 0;
        for (size_t k = 0; k < R; k++) f += F[c * R + k] * d[k], g += G[c * R + k] * d[k];
        out[c] = (float)((1 - tanh(f)) * x[c] + tanh(g) * m[c]);
    }
}

static void clip(float *v, size_t n, const float *scale, float thr, float tau) {
    double ss = 0;
    for (size_t i = 0; i < n; i++) {
        v[i] *= scale[i];
        ss += (double)v[i] * v[i];
    }
    const double f = fmin(thr / (sqrt(ss) + 1e-6), 1.0);
    for (size_t i = 0; i < n; i++) v[i] = (float)(v[i] * f * tau);
}

void ink_ref_qr(float *q, float *r, const float *qscale, const float *rscale, size_t NH, size_t HD, size_t DREL, size_t pos,
                int logscale, float thr_q, float thr_r, float alpha, float nfloor) {
    const double tau = logscale ? 1.0 + alpha * log(fmax((double)(pos + 1) / nfloor, 1.0)) : 1.0;
    for (size_t h = 0; h < NH; h++) {
        clip(q + h * HD, HD, qscale + h * HD, thr_q, (float)tau);
        clip(r + h * DREL, DREL, rscale + h * DREL, thr_r, (float)tau);
    }
}

static float lagv(const float *raw, const float *state, size_t C, size_t c, long t, long l) {
    const long s = t - l;
    return s >= 0 ? raw[(size_t)s * C + c] : state[(size_t)(3 + s) * C + c];
}

void ink_ref_kv_conv(const float *raw, const float *state, const float *w, const float *dyn, size_t T, size_t C, size_t HD,
                     int mode, float den, float thr, float eps, float *out) {
    const size_t nkv = C / HD;
    for (size_t t = 0; t < T; t++)
        for (size_t h = 0; h < nkv; h++) {
            float y[512];
            double ss = 0;
            for (size_t d = 0; d < HD; d++) {
                const size_t c = h * HD + d;
                double a = 0;
                for (long l = 0; l < 4; l++) a += (w[c * 4 + l] + (dyn ? dyn[(t * nkv + h) * 4 + l] : 0.0f)) * lagv(raw, state, C, c, (long)t, l);
                y[d] = (float)((a + raw[t * C + c]) / den);
                ss += (double)y[d] * y[d];
            }
            const double f = mode == 0 ? fmin(thr / (sqrt(ss) + 1e-6), 1.0) : 1.0 / sqrt(ss / (double)HD + eps);
            for (size_t d = 0; d < HD; d++) out[t * C + h * HD + d] = (float)(y[d] * f);
        }
}

void ink_ref_conv_res(const float *a, const float *state, const float *w, const float *hin, size_t T, size_t C, int boundary,
                      float den, float *r) {
    for (size_t t = 0; t < T; t++)
        for (size_t c = 0; c < C; c++) {
            double y = 0;
            for (long l = 0; l < 4; l++) y += w[c * 4 + l] * lagv(a, state, C, c, (long)t, l);
            y = (y + a[t * C + c]) / den;
            r[t * C + c] = (float)(boundary ? y : hin[t * C + c] + y);
        }
}

void ink_ref_router(const float *x, const float *W, const float *bias, size_t E, size_t H, size_t K, float scale, uint32_t *idx,
                    float *wts) {
    double sc[256], b[256];
    for (size_t e = 0; e < E; e++) {
        double a = 0;
        for (size_t i = 0; i < H; i++) a += (double)W[e * H + i] * x[i];
        sc[e] = 1.0 / (1.0 + exp(-a));
        b[e] = sc[e] + bias[e];
    }
    double sum = 0;
    for (size_t k = 0; k < K; k++) {
        size_t best = 0;
        for (size_t e = 1; e < E; e++)
            if (b[e] > b[best]) best = e;
        b[best] = -INFINITY;
        idx[k] = (uint32_t)best;
        sum += sc[best];
    }
    for (size_t k = 0; k < K; k++) wts[k] = (float)(sc[idx[k]] / (sum + 1e-20) * scale);
}

void ink_ref_attn(const float *q, const float *r, const float *proj, const float *K, const float *V, const float *sink,
                  const float *gate, size_t NH, size_t NKV, size_t HD, size_t DREL, size_t EXT, size_t pq, size_t win, float *out) {
    const size_t lo = win && pq + 1 > win ? pq + 1 - win : 0, ld = NKV * HD, rep = NH / NKV;
    for (size_t h = 0; h < NH; h++) {
        const size_t kh = h / rep;
        double mx = sink[h], s[8192];
        for (size_t j = lo; j <= pq; j++) {
            double d = 0;
            for (size_t i = 0; i < HD; i++) d += (double)q[h * HD + i] * K[j * ld + kh * HD + i];
            d /= sqrt((double)HD);
            const size_t dist = pq - j;
            if (dist < EXT)
                for (size_t i = 0; i < DREL; i++) d += (double)r[h * DREL + i] * proj[i * EXT + dist];
            s[j - lo] = d;
            mx = fmax(mx, d);
        }
        double z = exp(sink[h] - mx);
        for (size_t j = lo; j <= pq; j++) z += s[j - lo] = exp(s[j - lo] - mx);
        for (size_t i = 0; i < HD; i++) {
            double a = 0;
            for (size_t j = lo; j <= pq; j++) a += s[j - lo] * V[j * ld + kh * HD + i];
            out[h * HD + i] = (float)(a / z * sigm(gate[h * HD + i]));
        }
    }
}

static const float E2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

float ink_ref_e4m3(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const float v = e == 0 ? ldexpf((float)m / 8.0f, -6) : ldexpf(1.0f + (float)m / 8.0f, e - 7);
    return (b & 0x80) ? -v : v;
}

void ink_ref_row_nv(const uint8_t *w, const uint8_t *s, float gfac, size_t in, size_t row, float *out) {
    for (size_t i = 0; i < in; i++) {
        const uint8_t b = w[row * (in / 2) + i / 2];
        out[i] = E2M1[(i & 1) ? b >> 4 : b & 15] * (ink_ref_e4m3(s[row * (in / 16) + i / 16]) * gfac);
    }
}

void ink_ref_row_mx(const uint8_t *w, const uint8_t *s, size_t in, size_t row, float *out) {
    for (size_t i = 0; i < in; i++) out[i] = ink_ref_e4m3(w[row * in + i]) * ldexpf(1.0f, (int)s[row * (in / 32) + i / 32] - 127);
}

float ink_ref_swiglu(float g, float u, float gcap, float ucap) {
    return tanhf(g / gcap) * gcap * sigm(g) * (tanhf(u / ucap) * ucap);
}
