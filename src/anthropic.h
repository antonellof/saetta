// Anthropic Messages API requests (POST /v1/messages, as Claude Code sends them) turned into
// the OpenAI chat request that chat_render reads, so both APIs share one prompt and one
// generation path. Mapping after Strata serve/frontend.py anthropic_to_messages and
// pin_billing_stamp (MIT, Copyright (c) 2026 Niko1221 and the Strata contributors; 13eedf5):
//   system (string or text blocks)   -> the system message, billing stamp pinned (below)
//   text / thinking / tool_use       -> content / reasoning_content / tool_calls
//   tool_result                      -> a "tool" message each, in the order of the calls
//   a system message inside messages -> a user message in place (the prompt start stays put)
//   tools {name, description, input_schema} -> {type: function, function: {..., parameters}}
//   thinking {type: disabled}        -> enable_thinking false; output_config.effort -> reasoning_effort
// Images and documents have no input path in this engine: they become a short text note.
#ifndef SAETTA_ANTHROPIC_H
#define SAETTA_ANTHROPIC_H

#include "base.h"
#include "json.h"

// Claude Code starts its system prompt with `x-anthropic-billing-header: cc_version=A.B.C.xyz;
// cc_entrypoint=...; cch=12345;`. cch changes on every request and the version's 4th part on
// every session, which would change the prompt at its start and defeat prefix reuse on every
// turn. Both are overwritten with 'f' (in a copy in the arena, same length), only inside the
// first 160 bytes of a system text that starts with the header. Other text is returned as is.
Str anthropic_pin_billing_stamp(Arena *a, Str sys);

// The OpenAI chat request (*out, in the arena; it shares values with r) for Anthropic
// request r, streaming and sampling fields carried over. Errors: "BadRequest", "NoMessages".
Err anthropic_to_openai(Arena *a, const JValue *r, JValue **out);

#endif
