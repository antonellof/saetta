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
