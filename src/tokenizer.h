// Byte-level BPE tokenizer for Qwen3.8-Flash-Next, read from the pack's tokenizer.json.
// Adapted from sushi src/tokenizer.zig (MIT, Copyright (c) 2026 Theinruj Toranavikrai, David Dalcu; d5e876e):
// the hand-written pre-tokenizer state machine and the heap BPE with versioned
// candidates. Narrowed to the one grammar this model uses, which is checked at load
// instead of chosen at run time, and with generated Unicode classes (src/unicode.h).
// NFC normalization (tokenizer.json `normalizer`) goes through Foundation, only for
// text that has a non-ASCII byte.
#ifndef SAETTA_TOKENIZER_H
#define SAETTA_TOKENIZER_H

#include "base.h"

// An added token (tokenizer.json added_tokens): its literal text and id.
typedef struct {
    Str text;
    uint32_t id;
} Special;

typedef struct {
    StrMap vocab;          // token text -> id (value is the id cast to a pointer)
    // "left right" -> rank. Byte-level tokens never contain a space (it maps to U+0120).
    StrMap merges;
    size_t max_merge_key;
    Str *id_text;          // id -> token text ("" for unused ids)
    size_t n_ids;
    Special *specials;     // sorted longest first (stable)
    size_t n_specials;
    uint32_t byte_cp[256]; // byte -> code point (GPT-2 table)
    int16_t cp_byte[512];  // code point -> byte, -1 when the code point is not in the table
} Tokenizer;

// Loads <dir>/tokenizer.json into t. Strings and tables live in the arena (the two maps
// are malloc'd and live as long as the process). Errors: file errors, "BadTokenizer",
// "UnsupportedPreTokenizer", "UnsupportedModel", JSON errors.
Err tokenizer_load(Arena *a, const char *dir, Tokenizer *t);
// Same from the bytes of a tokenizer.json (they must outlive t).
Err tokenizer_from_json(Arena *a, Str bytes, Tokenizer *t);

// Id of a special or vocab token whose text is exactly `text`; false when none.
bool tokenizer_token_id(const Tokenizer *t, Str text, uint32_t *id);

// Appends the ids of `text` to out; added tokens written literally in text become their
// ids. Error: "TokenNotInVocab".
Err tokenizer_encode(const Tokenizer *t, Str text, VecU32 *out);
// Text without added tokens: pre-tokenize, map bytes to code points, BPE each piece.
Err tokenizer_encode_text(const Tokenizer *t, Str raw, VecU32 *out);

// Appends the bytes of token `id` to out (nothing for an id out of range).
void tokenizer_decode_append(const Tokenizer *t, uint32_t id, Buf *out);

// End of the pre-tokenizer piece that starts at byte i < s.len (exposed for tests).
size_t tokenizer_next_piece(Str s, size_t i);

#endif
