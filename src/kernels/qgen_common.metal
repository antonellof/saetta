// Shared helpers of the run-time quantization kernels (qgen.metal, qgen_mpp.metal): MLX affine
// packs at any bits in {4, 5, 6, 8} and group in {32, 64, 128}, both chosen at run time.
// Packing (checked against mx.dequantize; dequantRow in src/model.zig): a little-endian bitstream per row,
// value j at bit j * bits, so 5- and 6-bit values cross 32-bit words; w = q * scale + bias with
// bf16 scales and biases [rows][in / group]. Rows start at multiples of 4 bytes from the tensor
// start when in % 32 == 0 (in * bits / 8 = 4 * bits * in / 32), as do 32-value chunks; tensors
// themselves are not aligned, so words are read aligned and funnel-shifted.
// Bits is a template parameter inside: each kernel switches once on the run-time value (uniform
// over the dispatch) into a body compiled for that width, so the unpacking shifts are constants.
// Unsupported widths write NaN, so a wrong dispatch cannot pass a test silently.
// Helper names carry a qg_ prefix so this file can be concatenated with the other kernel sources.
#include <metal_stdlib>
using namespace metal;

inline float qg_bf16(device const uchar *p, uint i) {
    const uint lo = p[2 * i], hi = p[2 * i + 1];
    return as_type<float>(((hi << 8) | lo) << 16);
}

// N consecutive 32-bit words from byte address p (any alignment): aligned loads, funnel shift.
template <int N>
inline void qg_words(device const uchar *p, thread uint *wd) {
    const ulong addr = (ulong)p;
    const uint sh = uint(addr & 3);
    device const uint *q = (device const uint *)(addr - sh);
    if (sh == 0) {
#pragma clang loop unroll(full)
        for (int i = 0; i < N; i++) wd[i] = q[i];
    } else {
        uint prev = q[0];
#pragma clang loop unroll(full)
        for (int i = 0; i < N; i++) {
            const uint nx = q[i + 1];
            wd[i] = (prev >> (8 * sh)) | (nx << (32 - 8 * sh));
            prev = nx;
        }
    }
}

// Value j (compile-time) of a BITS-bit stream held in words wd.
template <int BITS, int J>
inline uint qg_val(thread const uint *wd) {
    constexpr uint MASK = (1u << BITS) - 1u;
    constexpr uint bit = J * BITS, wi = bit / 32, off = bit % 32;
    uint x = wd[wi] >> off;
    if (off + BITS > 32) x |= wd[wi + 1] << (32 - off);
    return x & MASK;
}

template <int BITS, int J>
inline float4 qg_val4(thread const uint *wd) {
    return float4(float(qg_val<BITS, J>(wd)), float(qg_val<BITS, J + 1>(wd)), float(qg_val<BITS, J + 2>(wd)), float(qg_val<BITS, J + 3>(wd)));
}

// 32 consecutive values (one chunk, BITS words, starting at byte p) as 8 float4.
template <int BITS>
inline void qg_unpack32(device const uchar *p, thread float4 *qv) {
    uint wd[BITS];
    qg_words<BITS>(p, wd);
    qv[0] = qg_val4<BITS, 0>(wd);
    qv[1] = qg_val4<BITS, 4>(wd);
    qv[2] = qg_val4<BITS, 8>(wd);
    qv[3] = qg_val4<BITS, 12>(wd);
    qv[4] = qg_val4<BITS, 16>(wd);
    qv[5] = qg_val4<BITS, 20>(wd);
    qv[6] = qg_val4<BITS, 24>(wd);
    qv[7] = qg_val4<BITS, 28>(wd);
}

// 8 consecutive values starting at byte p (BITS bytes, any alignment): a 64-bit window from at
// most 3 aligned words (only the words the window touches are read).
template <int BITS>
inline void qg_unpack8(device const uchar *p, thread float4 &qa, thread float4 &qb) {
    const ulong addr = (ulong)p;
    const uint sh = uint(addr & 3);
    device const uint *q = (device const uint *)(addr - sh);
    uint wd[2];
    if (BITS == 4 || BITS == 8) {
        // the 8 values are whole words from a word-relative start (4 or 8 bytes)
        qg_words<BITS / 4>(p, wd);
    } else {
        const uint w0 = q[0];
        const uint w1 = (sh + BITS > 4) ? q[1] : 0u;
        const uint w2 = (sh + BITS > 8) ? q[2] : 0u;
        wd[0] = sh == 0 ? w0 : (w0 >> (8 * sh)) | (w1 << (32 - 8 * sh));
        wd[1] = sh == 0 ? w1 : (w1 >> (8 * sh)) | (w2 << (32 - 8 * sh));
    }
    qa = qg_val4<BITS, 0>(wd);
    qb = qg_val4<BITS, 4>(wd);
}

// Run-time bits -> compile-time B, then STMT (which uses B); FAIL for unsupported widths.
#define QG_BITS(bits, STMT, FAIL)                                \
    switch (bits) {                                              \
    case 4: { constexpr int B = 4; STMT; } break;                \
    case 5: { constexpr int B = 5; STMT; } break;                \
    case 6: { constexpr int B = 6; STMT; } break;                \
    case 8: { constexpr int B = 8; STMT; } break;                \
    default: { FAIL; } break;                                    \
    }
