// Parses the model's tool calls into OpenAI `tool_calls`. Qwen writes XML:
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

// ---- Inkling 2 ---------------------------------------------------------------------------
// An Inkling 2 reply is a run of blocks framed by special tokens (tokenizer_config.json
// response_template; vLLM's unified Inkling parser, rust/src/parser/src/unified/inkling.rs at
// ef687f6, reads them the same way):
//   [<|message_model|>HEADER]<|content_thinking|>TEXT<|end_message|>          reasoning
//   [<|message_model|>HEADER]<|content_text|>TEXT<|end_message|>              content (also
//       <|content_tool_error|> and <|content_invoke_tool_text|>)
//   [<|message_model|>HEADER]<|content_invoke_tool_json|>{"name":N,"args":{..}}<|end_message|>
// and <|content_model_end_sampling|> (the end of the reply, also a block end). HEADER (the
// author name, the tool's name before a call) is dropped when a content kind follows; one
// that is closed without content, and text outside any block, are content. Blocks keep their
// text as written (no trimming); reasoning blocks are joined, and so are content blocks.

typedef struct {
    Str reasoning, content; // stable parts (see toolcall_inkling)
    Call *calls;            // complete tool calls, in order; arguments is the args object
    size_t n_calls;
    bool in_call;           // not final: the text ends inside a tool call block
    bool bad_call;          // a tool call block was not {"name": string, "args": object}
                            // (or, final, was cut off): its raw text went to content instead
    bool ended;             // <|content_model_end_sampling|> was seen
} InkParsed;

// Parses text, the reply so far (special tokens decoded to their text). Streaming-safe when
// final is false: reasoning and content leave out what a later token could still change (a
// tail that may begin a marker, a partial UTF-8 character, a header not yet resolved), so
// the results for a growing text extend each other and the caller sends the new bytes. With
// final, everything is resolved. Results live in the arena.
InkParsed toolcall_inkling(Arena *a, Str text, bool final);

// The same parse, incremental: each call consumes the bytes of text (the reply so far, which
// only grows) from where the last stopped, appending to reasoning and content (whole bytes; a
// caller trims a trailing partial UTF-8 character itself); final resolves the rest and ends the
// scan. The arena holds only the calls' strings (one parse per call), so streaming a reply
// allocates per tool call, not per token. toolcall_inkling is one scan on a fresh state.
typedef struct {
    int mode;
    size_t hdr, p;
    Buf reasoning, content;
    Call *calls;
    size_t n_calls, cap_calls;
    bool in_call, bad_call, ended, done;
} InkScan;
void toolcall_inkling_scan(Arena *a, InkScan *s, Str text, bool final);
void toolcall_inkling_free(InkScan *s);
// Length of s without a trailing partial UTF-8 character.
size_t toolcall_utf8_whole(Str s);

#endif
