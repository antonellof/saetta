// Run-time LoRA: t = A x, then y += scale * B t, per target linear, after its base matvec.
// A f16 [Rt][in] (the A rows of the linears sharing x, concatenated), B f16 [out][R] (one
// linear's B, applied to the slice t[r0 .. r0+R] of a t row of Rt values). f32 inputs,
// accumulation and outputs; plain multiplies and adds in a fixed order (no fma, no fast-math
// intrinsics). Shapes come in a constant buffer at the end of each kernel (see src/lora.h).
//   lora_down_small  T <= 8 (decode, verification chunks): one threadgroup per row of A.
//   lora_up_small    T <= 8: one thread per output, up to 3 linears sharing t in one dispatch.
//   lora_mm          any T: C = alpha * X W^T (+ C), f32 simdgroup matrices, both directions.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint LORA_TMAX = 8;   // tokens of the small kernels
constant constexpr uint LORA_DN_TG = 256; // threads of lora_down_small
constant constexpr uint LORA_UP_TG = 64;  // threads of lora_up_small
constant constexpr uint LORA_RMAX = 128;

struct LoraDownP {
    uint in, rt, T, ldx; // ldx: floats between rows of x
};

// grid (Rt) x 256. t[tok][r] = sum_k A[r][k] * x[tok][k]. in % 4 == 0, T <= 8.
kernel void lora_down_small(device const half *A [[buffer(0)]], device const float *x [[buffer(1)]],
                            device float *t [[buffer(2)]], constant LoraDownP &p [[buffer(3)]],
                            uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[LORA_DN_TG / 32][LORA_TMAX];
    float acc[LORA_TMAX];
    for (uint j = 0; j < LORA_TMAX; j++) acc[j] = 0.0f;
    device const half4 *a4 = (device const half4 *)(A + (ulong)r * p.in);
    const uint n4 = p.in / 4;
    for (uint i = tid; i < n4; i += LORA_DN_TG) {
        const float4 a = float4(a4[i]);
        for (uint j = 0; j < LORA_TMAX; j++) {
            if (j < p.T) {
                const float4 v = ((device const float4 *)(x + (ulong)j * p.ldx))[i];
                acc[j] += (a.x * v.x + a.y * v.y) + (a.z * v.z + a.w * v.w);
            }
        }
    }
    for (uint j = 0; j < LORA_TMAX; j++) {
        const float s = simd_sum(acc[j]);
        if (lane == 0) part[sg][j] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < p.T) {
        float s = 0.0f;
        for (uint k = 0; k < LORA_DN_TG / 32; k++) s += part[k][tid];
        t[(ulong)tid * p.rt + r] = s;
    }
}

struct LoraUpP {
    uint T, rt, R, n;      // tokens, t row stride, slice length, targets (1..3)
    uint out[3], r0[3], ldy[3];
    uint tg0[4];           // first threadgroup of each target, tg0[n] = grid size
    float scale;
};

// grid (tg0[n]) x 64. y_j[tok][o] += scale * sum_r B_j[o][r] * t[tok][r0_j + r], for target j
// of the threadgroup. R % 8 == 0, R <= 128, T <= 8. Unused B/y slots are bound to any buffer.
kernel void lora_up_small(device const half *B0 [[buffer(0)]], device const half *B1 [[buffer(1)]],
                          device const half *B2 [[buffer(2)]], device float *y0 [[buffer(3)]],
                          device float *y1 [[buffer(4)]], device float *y2 [[buffer(5)]],
                          device const float *t [[buffer(6)]], constant LoraUpP &p [[buffer(7)]],
                          uint g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float ts[LORA_TMAX * LORA_RMAX];
    const uint j = g >= p.tg0[2] ? 2 : (g >= p.tg0[1] ? 1 : 0);
    device const half *B = j == 0 ? B0 : (j == 1 ? B1 : B2);
    device float *y = j == 0 ? y0 : (j == 1 ? y1 : y2);
    const uint R = p.R, r0 = p.r0[j], out = p.out[j], ldy = p.ldy[j];
    for (uint e = tid; e < p.T * R; e += LORA_UP_TG) {
        const uint tok = e / R, r = e % R;
        ts[tok * R + r] = t[(ulong)tok * p.rt + r0 + r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint o = (g - p.tg0[j]) * LORA_UP_TG + tid;
    if (o >= out) return;
    float acc[LORA_TMAX];
    for (uint k = 0; k < LORA_TMAX; k++) acc[k] = 0.0f;
    device const half4 *b4 = (device const half4 *)(B + (ulong)o * R);
    for (uint i = 0; i < R / 4; i++) {
        const float4 b = float4(b4[i]);
        for (uint k = 0; k < LORA_TMAX; k++) {
            if (k < p.T) {
                const threadgroup float *v = ts + k * R + i * 4;
                acc[k] += (b.x * v[0] + b.y * v[1]) + (b.z * v[2] + b.w * v[3]);
            }
        }
    }
    for (uint k = 0; k < p.T; k++) {
        device float *dst = y + (ulong)k * ldy + o;
        *dst = *dst + p.scale * acc[k];
    }
}

struct LoraMmP {
    uint M, N, K, ldx, ldc, add; // add: 1 accumulates into C
    float alpha;
};

// grid (ceil(N / 32), ceil(M / 32)) x 128. C[m][n] = alpha * sum_k X[m][k] * W[n][k] (+ C[m][n]),
// X f32 rows ldx apart, W f16 [N][K], C rows ldc apart. K % 8 == 0 (rows of W read as given).
// Down: X = x, W = A, C = t. Up: X = t + r0 (rows Rt apart), W = B, C = y, add = 1.
kernel void lora_mm(device const half *W [[buffer(0)]], device const float *X [[buffer(1)]],
                    device float *C [[buffer(2)]], constant LoraMmP &p [[buffer(3)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
                    uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint BM = 32, BN = 32, BK = 32;
    threadgroup float xs[BM][BK + 1];
    threadgroup float ws[BN][BK + 1];
    threadgroup float cs[BM][BN + 1];
    const uint n0 = tg.x * BN, m0 = tg.y * BM;
    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; i++) for (uint j = 0; j < 2; j++) acc[i][j] = simdgroup_float8x8(0.0f);
    const uint sm = sgid / 2, sn = sgid % 2;
    for (uint k0 = 0; k0 < p.K; k0 += BK) {
        for (uint e = tid; e < BM * BK; e += 128) {
            const uint m = e / BK, k = e % BK;
            xs[m][k] = (m0 + m < p.M && k0 + k < p.K) ? X[(ulong)(m0 + m) * p.ldx + k0 + k] : 0.0f;
        }
        { // 8 consecutive weights per thread (K % 8 == 0: a run is all in or all out)
            const uint nn = tid / 4, k = (tid % 4) * 8, n = n0 + nn;
            if (n < p.N && k0 + k < p.K) {
                device const half4 *w4 = (device const half4 *)(W + (ulong)n * p.K + k0 + k);
                const float4 a = float4(w4[0]), b = float4(w4[1]);
                for (uint v = 0; v < 4; v++) { ws[nn][k + v] = a[v]; ws[nn][k + 4 + v] = b[v]; }
            } else {
                for (uint v = 0; v < 8; v++) ws[nn][k + v] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < BK; k += 8) {
            simdgroup_float8x8 a[2], b[2];
            for (uint i = 0; i < 2; i++) simdgroup_load(a[i], &xs[sm * 16 + i * 8][k], BK + 1);
            for (uint j = 0; j < 2; j++) simdgroup_load(b[j], &ws[sn * 16 + j * 8][k], BK + 1, ulong2(0, 0), true);
            for (uint i = 0; i < 2; i++) for (uint j = 0; j < 2; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 2; i++) for (uint j = 0; j < 2; j++) simdgroup_store(acc[i][j], &cs[sm * 16 + i * 8][sn * 16 + j * 8], BN + 1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < BM * BN; e += 128) {
        const uint m = e / BN, n = e % BN;
        if (m0 + m < p.M && n0 + n < p.N) {
            device float *dst = C + (ulong)(m0 + m) * p.ldc + n0 + n;
            const float v = p.alpha * cs[m][n];
            *dst = p.add ? *dst + v : v;
        }
    }
}
