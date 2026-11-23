// Incremental split of a reply's generated text into reasoning (before </think>) and answer,
// for streaming: split_poll gives the bytes that no later token can change. Held back: a
// partial UTF-8 character, trailing whitespace, a tail that could start </think> or
// <tool_call>, and a tail that could start one of the request's stop strings (a stop
// string cuts the answer where it begins, so none of its bytes may go out early).
#ifndef SAETTA_SPLIT_H
#define SAETTA_SPLIT_H

#include "base.h"
#include "toolcall.h"

typedef struct {
    Buf text;      // the generated text so far (with the forced opening, if any)
    bool thinking; // the reply starts inside a reasoning block
    // An Inkling 2 reply (special-token blocks): reasoning and content come from an incremental
    // toolcall_inkling scan over the new bytes of each token (no allocation per token; the
    // arena, set by the caller, holds the tool calls' strings), instead of the </think> split.
    bool inkling;
    Arena *arena;
    InkScan ink;
    size_t stop_from; // content bytes already searched for stop strings
    size_t sent_reasoning, sent_content;
    const Str *stops; // the request's stop strings (searched in the answer)
    size_t n_stops;
} Splitter;

// The reasoning (trimmed) and the answer (everything after </think>) of the text so far.
Str split_reasoning(const Splitter *sp);
Str split_answer(const Splitter *sp);

typedef struct {
    Str reasoning, content;
} SplitPoll;

// New reasoning and content bytes that can be sent now; final: the reply has ended, send
// the rest (trailing whitespace trimmed, content up to a tool call).
SplitPoll split_poll(Splitter *sp, bool final);

// Inkling: resolves the rest of the reply (the final scan); the calls are then sp->ink.calls.
void split_finish(Splitter *sp);
// Frees the splitter's buffers.
void split_free(Splitter *sp);

// When the answer contains one of the stop strings (the first in stops order that occurs),
// cuts the text where it begins and returns it; else NULL.
const Str *split_cut_stop(Splitter *sp);

// Stable prefix length of s: drops a tail that could start tag or any of stops[0..n_stops),
// then a partial UTF-8 character, then trailing whitespace.
size_t split_stable(Str s, Str tag, const Str *stops, size_t n_stops);

#endif
