// Chat prompt rendering: the official Qwen3.8-Flash-Next chat_template.jinja (HF 812d96e)
// written out in C for this one template, instead of a general Jinja engine (sushi ships
// one, src/template/). It reads an OpenAI chat request body; a development check compares
// it byte for byte with the template rendered by jinja2 the way transformers does it.
// Template kwargs come from `chat_template_kwargs` (enable_thinking, reasoning_effort,
// preserve_thinking) or a top-level `reasoning_effort`.
#ifndef SAETTA_CHAT_H
#define SAETTA_CHAT_H

#include "base.h"
#include "json.h"

// Renders the prompt for req (an OpenAI chat completion request, parsed with
// parse_numbers false so numbers keep their spelling) with the generation prompt.
// *out lives in the arena. Errors: "NoMessages", "NoUserQuery", "SystemNotFirst",
// "BadRole", "BadContent", "VisionUnsupported", "BadReasoningEffort", "BadToolCall".
Err chat_render(Arena *a, const JValue *req, Str *out);

// Python json.dumps(x, ensure_ascii=False): ", " and ": " separators, keys in order.
// Numbers are written as the request spelled them (parse with parse_numbers false).
void chat_to_json(Buf *o, const JValue *v);

// A quoted JSON string: \" \\ \n \r \t \b \f escaped, other controls as \u00xx
// (lowercase hex), everything else (DEL, non-ASCII) raw, except ill-formed UTF-8 (a reply
// cut inside a character, stray bytes): each maximal ill-formed subpart becomes U+FFFD, so
// the output is always valid UTF-8. Used for every JSON string the server writes.
void chat_json_string(Buf *o, Str s);

#endif
