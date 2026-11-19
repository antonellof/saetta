// Tests of the server's sampling (src/server.h).
#include <math.h>

#include "../server.h"
#include "../test.h"

TEST(server_speculative_sampling_keeps_the_sampling_distribution) {
    // a fixed draft token: accepted with its probability, else drawn from the rest; the
    // frequencies must match softmax(logits / T) for every token
    QgenPrng rng;
    qgen_prng_init(&rng, 12345);
    const float logits[5] = {1.0f, 2.5f, 0.3f, 1.8f, -0.5f};
    uint32_t cand[256];
    const float temps[2] = {1.0f, 0.6f};
    for (int ti = 0; ti < 2; ti++) {
        float temp = temps[ti];
        SampleChooser ch = {&rng, {temp, 1.0f, 20}, cand, 5};
        double want[5], z = 0;
        for (int i = 0; i < 5; i++) {
            want[i] = exp((double)logits[i] / temp);
            z += want[i];
        }
        const int trials = 400000;
        double got[5] = {0};
        for (int t = 0; t < trials; t++) got[server_sample_pick(&ch, logits, true, 3)] += 1;
        for (int i = 0; i < 5; i++) EXPECT(fabs(want[i] / z - got[i] / trials) <= 0.004);
    }
}

// std.Random.uintLessThan (Lemire) on 32 bits, for intRangeAtMost.
static uint32_t less_than(QgenPrng *r, uint32_t bound) {
    uint32_t x = (uint32_t)qgen_prng_next(r);
    uint64_t m = (uint64_t)x * bound;
    uint32_t l = (uint32_t)m;
    if (l < bound) {
        uint32_t t = (uint32_t)-bound;
        if (t >= bound) {
            t -= bound;
            if (t >= bound) t %= bound;
        }
        while (l < t) {
            x = (uint32_t)qgen_prng_next(r);
            m = (uint64_t)x * bound;
            l = (uint32_t)m;
        }
    }
    return (uint32_t)(m >> 32);
}

// A standard normal value (Box-Muller; the test only needs varied values with ties absent).
static float normal(QgenPrng *r) {
    double u = 1.0 - (double)qgen_prng_float(r), v = (double)qgen_prng_float(r);
    return (float)(sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v));
}

TEST(server_top_k_with_group_skipping_matches_the_plain_loop) {
    QgenPrng rng;
    qgen_prng_init(&rng, 7);
    static float logits[5003];
    uint32_t a[256], b[256];
    for (size_t round = 0; round < 20; round++) {
        for (size_t i = 0; i < 5003; i++)
            logits[i] = round % 3 == 0 ? (float)less_than(&rng, 10) : normal(&rng) * 3;
        size_t k = 1 + round % 25;
        size_t na = server_top_k(logits, 5003, k, a);
        // the plain loop
        size_t nb = 0;
        for (size_t i = 0; i < 5003; i++) {
            float v = logits[i];
            if (nb < k) {
                b[nb] = (uint32_t)i;
                nb += 1;
                for (size_t j = nb - 1; j > 0 && logits[b[j]] > logits[b[j - 1]]; j--) {
                    uint32_t t = b[j];
                    b[j] = b[j - 1];
                    b[j - 1] = t;
                }
            } else if (v > logits[b[k - 1]]) {
                b[k - 1] = (uint32_t)i;
                for (size_t j = k - 1; j > 0 && logits[b[j]] > logits[b[j - 1]]; j--) {
                    uint32_t t = b[j];
                    b[j] = b[j - 1];
                    b[j - 1] = t;
                }
            }
        }
        EXPECT_EQ(na, nb);
        for (size_t i = 0; i < nb; i++) EXPECT_EQ(a[i], b[i]);
    }
}
