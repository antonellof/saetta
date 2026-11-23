#include "../test.h"
#include "../json.h"
#include "../toolcall.h"

TEST(tool_calls_framing_typing_duplicates_content) {
    Arena a = {0};
    JValue *tools;
    EXPECT_OK(json_parse(&a, S("[{\"type\":\"function\",\"function\":{\"name\":\"edit\",\"parameters\":{\"type\":\"object\",\"properties\":{\"old\":{\"type\":\"string\"},\"n\":{\"type\":\"integer\"},\"code\":{\"type\":\"string\"}}}}}]"), true, &tools));
    Str text = S("I will edit.\n\n<tool_call>\n<function=edit>\n<parameter=old>\n    indented\n\n</parameter>\n<parameter=n>\n 3 \n</parameter>\n<parameter=code>\n123\n</parameter>\n<parameter=old>\nsecond\n</parameter>\n<parameter=html>\n<p>a</parameter> b</p>\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=ls>\n</function>\n</tool_call>");
    Parsed r = toolcall_parse(&a, text, tools);
    EXPECT_STR(r.content, S("I will edit."));
    EXPECT_EQ(r.n_calls, 2);
    EXPECT_STR(r.calls[0].name, S("edit"));
    EXPECT_STR(r.calls[0].arguments, S("{\"old\":\"    indented\\n\",\"n\":3,\"code\":\"123\",\"html\":\"<p>a</parameter> b</p>\"}"));
    EXPECT_STR(r.calls[1].name, S("ls"));
    EXPECT_STR(r.calls[1].arguments, S("{}"));
    arena_reset(&a);
}

// ---- Inkling 2 ---------------------------------------------------------------------------

#define INK_THINK "<|message_model|><|content_thinking|>"
#define INK_TEXT "<|message_model|><|content_text|>"
#define INK_END "<|end_message|>"
#define INK_EOS "<|content_model_end_sampling|>"

TEST(inkling_reply_reasoning_content_calls) {
    Arena a = {0};
    Str text = S(INK_THINK "plan:\n  ls first" INK_END INK_TEXT " Checking. " INK_END
                 "<|message_model|>ls<|content_invoke_tool_json|>{\"name\":\"ls\",\"args\":{}}" INK_END
                 "<|message_model|>edit<|content_invoke_tool_json|>{\"name\":\"edit\",\"args\":{\"path\":\"a.c\",\"n\":1.50,"
                 "\"s\":\"x <|end_message|> }{\\\"\",\"d\":{\"k\":[1,true,null]}}}" INK_END INK_EOS);
    InkParsed r = toolcall_inkling(&a, text, true);
    EXPECT_STR(r.reasoning, S("plan:\n  ls first"));
    EXPECT_STR(r.content, S(" Checking. "));
    EXPECT_EQ(r.n_calls, 2);
    EXPECT_STR(r.calls[0].name, S("ls"));
    EXPECT_STR(r.calls[0].arguments, S("{}"));
    EXPECT_STR(r.calls[1].name, S("edit"));
    EXPECT_STR(r.calls[1].arguments, S("{\"path\":\"a.c\",\"n\":1.50,\"s\":\"x <|end_message|> }{\\\"\",\"d\":{\"k\":[1,true,null]}}"));
    EXPECT(r.ended && !r.in_call && !r.bad_call);
    arena_reset(&a);
}

// Every prefix of a reply, parsed as a stream: reasoning and content only grow, calls only
// appear when complete, and the last result is the final parse.
TEST(inkling_stream_prefixes_extend) {
    Arena a = {0};
    Str text = S(INK_THINK "r\xc3\xa9" "ason <|x" INK_END INK_TEXT "caf\xc3\xa9 <b>" INK_END
                 "<|message_model|>get_weather<|content_invoke_tool_json|> {\"name\":\"get_weather\",\"args\":{\"city\":\"S\\\"F}\"}} "
                 INK_END "<|message_model|>bot" INK_END INK_TEXT "tail" INK_END INK_EOS);
    InkParsed full = toolcall_inkling(&a, text, true);
    EXPECT_STR(full.reasoning, S("r\xc3\xa9" "ason <|x"));
    EXPECT_STR(full.content, S("caf\xc3\xa9 <b>bottail"));
    EXPECT_EQ(full.n_calls, 1);
    EXPECT_STR(full.calls[0].arguments, S("{\"city\":\"S\\\"F}\"}"));
    Str prev_r = S(""), prev_c = S("");
    size_t prev_n = 0;
    for (size_t n = 0; n <= text.len; n++) {
        InkParsed r = toolcall_inkling(&a, str_sub(text, 0, n), false);
        EXPECT(str_starts(r.reasoning, prev_r));
        EXPECT(str_starts(r.content, prev_c));
        EXPECT(str_starts(full.reasoning, r.reasoning));
        EXPECT(str_starts(full.content, r.content));
        EXPECT(r.n_calls >= prev_n && r.n_calls <= full.n_calls);
        EXPECT(!r.bad_call);
        prev_r = r.reasoning;
        prev_c = r.content;
        prev_n = r.n_calls;
    }
    EXPECT_STR(prev_r, full.reasoning);
    EXPECT_STR(prev_c, full.content);
    EXPECT_EQ(prev_n, full.n_calls);
    arena_reset(&a);
}

TEST(inkling_stream_holds_markers_headers_and_partial_utf8) {
    Arena a = {0};
    InkParsed r = toolcall_inkling(&a, S(INK_TEXT "abc<|end_mes"), false);
    EXPECT_STR(r.content, S("abc"));
    r = toolcall_inkling(&a, S(INK_TEXT "caf\xc3"), false);
    EXPECT_STR(r.content, S("caf"));
    r = toolcall_inkling(&a, S("<|message_model|>plain answer"), false);
    EXPECT_STR(r.content, S(""));
    r = toolcall_inkling(&a, S("<|message_model|>plain answer"), true);
    EXPECT_STR(r.content, S("plain answer"));
    r = toolcall_inkling(&a, S("<|message_model|>plain" INK_END INK_EOS), false);
    EXPECT_STR(r.content, S("plain"));
    EXPECT(r.ended);
    r = toolcall_inkling(&a, S("<|content_invoke_tool_json|>{\"name\":\"ls\",\"args\":{\"a\":\"<|end_message|>"), false);
    EXPECT(r.in_call && r.n_calls == 0 && r.content.len == 0);
    arena_reset(&a);
}

TEST(inkling_plain_text_raw_kinds_and_bad_calls) {
    Arena a = {0};
    InkParsed r = toolcall_inkling(&a, S("no markers at all"), true);
    EXPECT_STR(r.content, S("no markers at all"));
    r = toolcall_inkling(&a, S("<|content_invoke_tool_text|>search SF" INK_END "<|content_tool_error|>failed" INK_END), true);
    EXPECT_STR(r.content, S("search SFfailed"));
    EXPECT_EQ(r.n_calls, 0);
    // args not an object: the raw payload is content
    r = toolcall_inkling(&a, S("<|content_invoke_tool_json|>{\"name\":\"ls\",\"args\":42}" INK_END INK_TEXT "after" INK_END), false);
    EXPECT(r.bad_call && r.n_calls == 0);
    EXPECT_STR(r.content, S("{\"name\":\"ls\",\"args\":42}after"));
    // not JSON
    r = toolcall_inkling(&a, S("<|content_invoke_tool_json|>ls -la" INK_END), true);
    EXPECT(r.bad_call);
    EXPECT_STR(r.content, S("ls -la"));
    // cut off at the end of the reply
    r = toolcall_inkling(&a, S("<|content_invoke_tool_json|>{\"name\":\"ls\",\"args\":{\"p\""), true);
    EXPECT(r.bad_call && r.n_calls == 0);
    EXPECT_STR(r.content, S("{\"name\":\"ls\",\"args\":{\"p\""));
    // a whole call with its block end missing, at the end of the reply
    r = toolcall_inkling(&a, S("<|content_invoke_tool_json|>{\"name\":\"ls\"}"), true);
    EXPECT(!r.bad_call && r.n_calls == 1);
    EXPECT_STR(r.calls[0].arguments, S("{}"));
    arena_reset(&a);
}
