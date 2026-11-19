// Parses the model's XML tool calls into OpenAI `tool_calls`:
//   <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>
// Rules adapted from sushi src/tool_calls.zig (MIT, Copyright (c) 2026 Theinruj Toranavikrai, David Dalcu; d5e876e),
// learned there from live agent traffic: exactly one framing newline is removed on each
// side of a value (an `old_string` must keep its indentation), a value ends at the last
// `</parameter>` before the next opener, a repeated parameter name keeps the first value,
// a malformed name is skipped. Changed here: values are typed from the request's tool
// schema (a "string" parameter stays a string even when it reads "123"), and only by
// spelling when the schema says nothing.
#ifndef SAETTA_TOOLCALL_H
#define SAETTA_TOOLCALL_H

#include "base.h"
#include "json.h"

typedef struct {
    Str name;
    Str arguments; // a JSON object text, as OpenAI's function.arguments
} Call;

typedef struct {
    Str content;   // the text before the first call, trailing whitespace removed
    Call *calls;
    size_t n_calls;
} Parsed;

// Byte index of the first tool call opener ("<tool_call>") in text, or -1.
ptrdiff_t toolcall_start(Str text);

// Splits text (the answer after reasoning) into content and tool calls. tools is the
// request's "tools" array (or NULL) for the parameter types. Results live in the arena;
// content and names point into text.
Parsed toolcall_parse(Arena *a, Str text, const JValue *tools);

#endif
