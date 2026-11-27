// Prompt-lookup drafting for speculative decoding: guesses that the tokens after the most
// recent earlier occurrence of the last N tokens follow again (copying code, file contents,
// tool output). The idea is the prompt-lookup / suffix drafter in sushi and qwental;
// this is a plain trigram index, O(1) per token.
#ifndef SAETTA_SPEC_H
#define SAETTA_SPEC_H

#include "base.h"

#define SPEC_N 3

// History of consumed tokens and a trigram -> position map (open addressing). Zero
// initialize (`Lookup lk = {0};`); memory is malloc'd, release it with lookup_free (the
// Zig engine took an arena; callers that used one free it where the arena ended).
typedef struct {
    VecU32 hist;
    // trigram -> position just after its most recent occurrence
    uint64_t *keys;
    uint32_t *vals;
    uint8_t *used;
    size_t cap, len;
} Lookup;

// Clears the history and the index, keeping their memory.
void lookup_reset(Lookup *l);
// Appends a consumed token; indexes the trigram that ends before it.
void lookup_push(Lookup *l, uint32_t tok);
// Up to n guessed tokens into out, after the history, from the most recent earlier
// occurrence of its last N tokens; 0 when there is none.
size_t lookup_draft(const Lookup *l, uint32_t *out, size_t n);
void lookup_free(Lookup *l);

// Drafting from another source (src/server.c, speculation fed by the agent): up to n tokens into
// out from src, after the most recent occurrence in src of the last SPEC_N tokens of ctx[0..nctx),
// stopping before a SPEC_SEP; *match: how many tokens of ctx end there, SPEC_N up to SPEC_CTX (an
// occurrence with a longer common context is the likelier copy). 0 when there is none. With src
// the history itself (ctx = src->hist) it drafts what lookup_draft drafts.
#define SPEC_CTX 32
// Pushed between the texts of one source (not a token id: vocabularies stay under 2^21, which the
// trigram key packs exactly), so no draft runs from one text into the next.
#define SPEC_SEP ((uint32_t)((1u << 21) - 1))
size_t lookup_draft_in(const Lookup *src, const uint32_t *ctx, size_t nctx, uint32_t *out, size_t n, size_t *match);
// Appends SPEC_SEP without indexing the trigram that ends before it (the last trigram of a text
// keeps pointing at its earlier occurrences, which have tokens after them).
void lookup_push_sep(Lookup *l);

// Speculates only while it pays: a verification pass costs about `break_even` decode steps
// (measured 37 ms against 14 ms: 2.6), so drafting continues while the running average of
// tokens per pass stays above it, and after a poor stretch pauses for `cooldown` steps.
typedef struct {
    float avg;
    uint32_t pause;
} Gate;
#define GATE_BREAK_EVEN 2.6f
#define GATE_COOLDOWN 16u
#define GATE_INIT ((Gate){.avg = 4.0f, .pause = 0})

bool gate_allow(Gate *g);
// Drafts to ask prompt lookup for: about 1.5x the recent tokens per pass, 2..max (long
// copies get deep drafts, fresh text short ones: KMAX 8 helped an edit 9.22 -> 8.41 ms/token
// but cost fresh code 11.12 -> 11.88 at a fixed depth).
size_t gate_depth(const Gate *g, size_t max);
// tokens: produced by a speculative pass (accepted drafts + 1).
void gate_record(Gate *g, size_t tokens);

// MTP draft depth (after sushi's expected-value controller, src/round_cost.zig, MIT): keeps
// the acceptance rate p[i] of draft i given the ones before it and the measured wall time of
// a round at each depth, and picks the depth k that maximizes expected tokens per round,
// 1 + p0 + p0 p1 + ..., over its time (k = 0: a plain step). Times start from measurements
// on this machine (`SAETTA_CHUNK_COST=1 saetta generate`, step 13.1 ms;
// verify 1..5 tokens 13.6/17.7/23.2/27.9/33.2; a head pass about 1.9) and then follow what
// rounds really cost (server rounds include streaming).
typedef struct {
    float p[4];
    float ms[5];
    uint32_t rounds;
} Depth;
// ms[k] = verify(k + 1 tokens) + k head passes: 17.7 + 1.9, 23.2 + 3.8, 27.9 + 5.7, 33.2 + 7.6
// (written as their f32 sums, which is what the comptime arithmetic rounds to).
#define DEPTH_INIT ((Depth){.p = {0.7f, 0.6f, 0.5f, 0.4f}, .ms = {13.1f, 19.6f, 27.0f, 33.6f, 40.8f}, .rounds = 0})

size_t depth_pick(Depth *d, size_t kmax);
// A round at depth k (0: plain step) took `ms` and accepted a drafts.
void depth_record(Depth *d, size_t k, size_t a, float ms);

#endif
