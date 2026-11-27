// Tool-call grammar for the Qwen chat format (SAETTA_GRAMMAR=1; off by default): a character-level
// automaton over the reply that says, for each candidate token, whether emitting it keeps the
// reply a well-formed answer for the request's tools. The sampler masks with it (src/server.c):
// the model's own token is taken whenever it is valid, and only an invalid one is replaced by the
// best valid token, so a greedy reply whose tool calls are already well formed is unchanged.
//
// The reply it accepts (the format of the chat template's tools section, src/chat.c tools_tail):
//   [reasoning </think>] text [<tool_call> ws <function=NAME> ws (<parameter=KEY> VALUE </parameter> ws)*
//   </function> ws </tool_call> (ws <tool_call> ...)*] then the end of the reply
// - reasoning (when the prompt opened <think>) may not contain <tool_call>: a call written there is
//   read as reasoning and nothing runs (the model has to close </think> first); the reply may end
//   inside it (refusing that end made the model loop in its reasoning to the token limit);
// - NAME is one of the request's tools (a tool whose name has a newline, a '>' or padding at an end
//   cannot be called and is left out); KEY any name the parser takes (a key the schema does not
//   list is allowed, as JSON Schema allows it; spaces, tabs and CRs around a name are trimmed as the
//   parser trims them, a newline in a name is refused); every required key comes before
//   </function>, and </function> may not even start before (a prefix of it would leave the reply
//   nothing valid to write); a required key no call can name (empty, padded, with one of < > = "
//   or a newline, longer than GR_NAME) is not required here;
// - VALUE of a parameter the parser reads as text (schema type "string", a union with "string",
//   or no type: src/toolcall.c param_type) is any text without </parameter> followed by anything
//   but another <parameter= or </function>, and without <parameter= (the parser would split it
//   there); of any other type it is one JSON value that json_parse takes (escapes, surrogate pairs,
//   UTF-8, no repeated key of the schema's, keys compared after their escapes), of the declared
//   type or types (a union without "string" takes its kinds only), with the schema's object keys,
//   item types and required keys checked as far as they are declared (nested up to GR_DEPTH; the
//   required and repeated checks on an object's first 64 properties). enum, const and the
//   combinators (anyOf, oneOf, allOf) are not checked: a value keeps the node's "type"; parameters
//   with no object type take any keys;
// - after the last </tool_call> only whitespace or another call (no text after a call).
// Not checked: a key the schema does not list (or past its 64th property, or longer than GR_NAME)
// written twice in one JSON object (json_parse would refuse it: the parser then keeps the value as
// a string); the value of a key longer than GR_NAME (typed as any JSON); a parameter whose type is a
// union with "string": its text is free, and the parser emits text that reads as JSON as that JSON,
// whatever the union's other kinds (5 for ["string","null"]); schemas of more than 65,535 nodes or
// properties (the grammar is then not used).
//
// The state (GState) is a small value: a check copies it and feeds the token's bytes, with no
// allocation; the Grammar is built once per request in the request's arena.
#ifndef SAETTA_GRAMMAR_H
#define SAETTA_GRAMMAR_H

#include "base.h"
#include "json.h"
#include "tokenizer.h"

// GR_NAME: the longest tool, parameter or key name matched (the parser takes parameter names up
// to 128 bytes); GR_DEPTH: the deepest JSON nesting accepted
enum { GR_NAME = 128, GR_DEPTH = 16 };

typedef struct {
    uint8_t kinds;     // JSON kinds a value may be (GK_* bits; GK_ANY: any)
    uint8_t text;      // a parameter the parser reads as text (no JSON check)
    uint16_t items;    // array: node of the items (0: any)
    uint16_t prop0;    // object: first property in Grammar.props
    uint16_t nprops;
    uint64_t required; // object: bit i = property prop0 + i is required (first 64)
} GNode;

typedef struct {
    Str name;
    uint16_t node;
} GProp;

typedef struct {
    Str name;
    uint16_t node; // object node of its parameters
} GTool;

typedef struct {
    const Tokenizer *tok;
    const uint32_t *eos;
    size_t n_eos;
    GNode *nodes; // nodes[0] is "any"
    size_t n_nodes;
    GProp *props;
    size_t n_props;
    GTool *tools; // the request's tools with names of 1 to GR_NAME bytes
    size_t n_tools;
} Grammar;

typedef struct {
    uint8_t mode, next; // GM_* and, in a literal mode, the mode after it
    uint8_t lit, k;     // the literal being matched and its progress
    uint8_t alive;      // a choice between literals: bit per literal still matching
    uint8_t kc, kt;     // progress of the literals watched in the current mode
    uint8_t trail;      // a name's trailing padding stored in name[] (trimmed at its '>')
    uint8_t vpad;       // padding after a name, not stored (only padding or '>' may follow)
    int16_t tool;       // the call's tool
    int16_t param;      // its current parameter (index into the tool's properties, or -1)
    uint64_t seen;      // the call's parameters given so far (first 64)
    uint16_t nlen;      // bytes of name[] (a function, parameter or object key name); GR_NAME + 1: longer
    char name[GR_NAME];
    // JSON value of a typed parameter
    uint8_t jmode, depth, jk, jflag;
    uint8_t jstr;             // the string is a key
    uint8_t u8need, u8lo, u8hi; // UTF-8 inside a string: continuation bytes due, range of the next
    uint16_t uval, uhi;       // a \u escape's value so far; a high surrogate before its low half
    uint16_t jexp;            // node expected for the next value
    uint16_t jnode[GR_DEPTH]; // per open container: its node
    uint8_t jkind[GR_DEPTH];  // '[' or '{'
    uint64_t jseen[GR_DEPTH]; // per open object: properties given
    int16_t jkey;             // the property of the key just read (-1: not in the schema)
} GState;

// Builds the grammar of a request's tools array (OpenAI "tools"). eos: the model's end tokens.
// Lives in the arena. With n_tools 0 no call is valid: the caller does not use it then.
Grammar *grammar_build(Arena *a, const JValue *tools, const Tokenizer *tok, const uint32_t *eos, size_t n_eos);

// The state at the start of a reply: in reasoning or not, then forced (the reply's forced
// opening, may be empty) fed.
GState grammar_start(const Grammar *g, bool thinking, Str forced);

// Whether emitting token tok in state s keeps the reply valid. A token with no bytes (an id past
// the tokenizer's, which a logits row may hold) is valid only when s is dead: it would not move s.
bool grammar_ok(const Grammar *g, const GState *s, uint32_t tok);
// The same for raw bytes (tests; a token's text).
bool grammar_ok_bytes(const Grammar *g, const GState *s, Str bytes);

// Advances s by token tok (whatever its validity: an invalid token leaves s dead, and then
// every token is valid again, so a reply the mask could not save is not blocked). An end token
// also leaves s dead: the tokens a verification round reads past the reply's end are not emitted.
void grammar_feed(const Grammar *g, GState *s, uint32_t tok);
void grammar_feed_bytes(const Grammar *g, GState *s, Str bytes);

// Whether the reply may end in s (an end token).
bool grammar_may_end(const GState *s);
// Whether s is inside a tool call (tests, logs).
bool grammar_in_call(const GState *s);

#endif
