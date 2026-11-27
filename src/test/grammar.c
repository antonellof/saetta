// The tool-call grammar (src/grammar.h) on reply texts, byte by byte: well-formed replies stay
// valid at every prefix, may end, and parse (src/toolcall.c) into the expected number of calls with
// arguments of the schema's types; each kind of malformed reply (seen from the models, or found in
// review) is refused at the byte where it goes wrong; and no accepted prefix is a dead end (some
// byte keeps it valid, or the reply may end there), which a refused token would otherwise turn
// into a loop.
#include "../test.h"
#include "../grammar.h"
#include "../json.h"
#include "../toolcall.h"

static const char pi_tools[] =
    "[{\"type\":\"function\",\"function\":{\"name\":\"read\",\"parameters\":{\"type\":\"object\",\"required\":[\"path\"],"
    "\"properties\":{\"path\":{\"type\":\"string\"},\"offset\":{\"type\":\"number\"},\"limit\":{\"type\":\"number\"}}}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"bash\",\"parameters\":{\"type\":\"object\",\"required\":[\"command\"],"
    "\"properties\":{\"command\":{\"type\":\"string\"},\"timeout\":{\"type\":\"number\"}}}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"edit\",\"parameters\":{\"type\":\"object\",\"required\":[\"path\",\"edits\"],"
    "\"properties\":{\"path\":{\"type\":\"string\"},\"edits\":{\"type\":\"array\",\"items\":{\"type\":\"object\","
    "\"required\":[\"oldText\",\"newText\"],\"properties\":{\"oldText\":{\"type\":\"string\"},\"newText\":{\"type\":\"string\"}}}}}}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"write\",\"parameters\":{\"type\":\"object\",\"required\":[\"path\",\"content\"],"
    "\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}}}}},"
    // typed parameters with an enum and a union, and a 64-byte name (OpenAI's longest)
    "{\"type\":\"function\",\"function\":{\"name\":\"mode\",\"parameters\":{\"type\":\"object\",\"required\":[\"level\",\" padded\"],"
    "\"properties\":{\"level\":{\"type\":\"integer\",\"enum\":[1,2]},\"either\":{\"type\":[\"number\",\"boolean\"]},"
    "\"note\":{\"type\":[\"string\",\"null\"]},\"opts\":{\"type\":\"object\",\"properties\":{\"n\":{\"type\":\"integer\"}}},"
    "\"pick\":{\"type\":[\"number\",\"array\"]},\" padded\":{\"type\":\"string\"}}}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"mcp__server_with_a_long_name__and_a_tool_name_that_fills_64bytes\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]";

static Grammar *pi_grammar(Arena *a, JValue **tools) {
    if (json_parse(a, str(pi_tools), true, tools)) die("pi_tools: bad JSON\n");
    static const uint32_t eos[1] = {0};
    return grammar_build(a, *tools, NULL, eos, 1);
}

// The first byte offset at which text stops being valid, or -1.
static ptrdiff_t first_bad(const Grammar *g, bool thinking, Str text) {
    GState s = grammar_start(g, thinking, S(""));
    for (size_t i = 0; i < text.len; i++) {
        if (!grammar_ok_bytes(g, &s, str_n(text.ptr + i, 1))) return (ptrdiff_t)i;
        grammar_feed_bytes(g, &s, str_n(text.ptr + i, 1));
    }
    return -1;
}

// The first prefix of text (up to its first refused byte) at which no byte is valid and the reply
// may not end, or -1.
static ptrdiff_t first_dead_end(const Grammar *g, bool thinking, Str text) {
    GState s = grammar_start(g, thinking, S(""));
    for (size_t i = 0; i <= text.len; i++) {
        bool any = grammar_may_end(&s);
        for (int b = 1; b < 256 && !any; b++) {
            char c = (char)b;
            any = grammar_ok_bytes(g, &s, str_n(&c, 1));
        }
        if (!any) return (ptrdiff_t)i;
        if (i == text.len || !grammar_ok_bytes(g, &s, str_n(text.ptr + i, 1))) break;
        grammar_feed_bytes(g, &s, str_n(text.ptr + i, 1));
    }
    return -1;
}

static bool may_end_after(const Grammar *g, bool thinking, Str text) {
    GState s = grammar_start(g, thinking, S(""));
    grammar_feed_bytes(g, &s, text);
    return grammar_may_end(&s);
}

#define CALL(name, params) "<tool_call>\n<function=" name ">\n" params "</function>\n</tool_call>"
#define P(k, v) "<parameter=" k ">\n" v "\n</parameter>\n"
#define LONG_NAME "mcp__server_with_a_long_name__and_a_tool_name_that_fills_64bytes"

typedef struct {
    const char *text;
    size_t calls;
} Good;

static const Good good[] = {
    {"Plain answer, no call.", 0},
    {"Let me read it.\n\n" CALL("read", P("path", "mathkit/stats.py")), 1},
    {CALL("read", P("path", "a.py") P("offset", "10") P("limit", "2.5e1")), 1},
    {CALL("bash", P("command", "cd /w && python3 -c \"print(1)\"\nls -la | head")) "\n" CALL("bash", P("command", "ls")), 2},
    {CALL("edit", P("path", "s.py") P("edits", "[{\"oldText\": \"    return s[mid]\\n\", \"newText\": \"    return (a + b) / 2\\n\"}]")), 1},
    {CALL("edit", P("path", "s.py") P("edits", "[\n  {\"newText\": \"x\", \"oldText\": \"y \\\"q\\\" \\u00e9 \\ud83d\\ude00 caf\xc3\xa9\", "
                                                 "\"note\": [1, true, null]},\n  {\"oldText\": \"a\", \"newText\": \"\"}\n]")), 1},
    {CALL("write", P("path", "u/i.py") P("content", "def f():\n    return \"<b>\" + '</p>'\n")), 1},
    {CALL("write", P("content", "x") P("path", "y") P("extra", "kept")), 1},
    {"text\n" CALL("read", P("path", "a")) "\n\n", 1},
    // padding around names, trimmed by the parser; \v between tags
    {"<tool_call>\n<function= read >\n<parameter= offset\t>\n3\n</parameter>\n\v<parameter=path >\na\n</parameter>\n</function>\n</tool_call>", 1},
    {CALL("mode", P("level", "2") P("either", "true") P("note", "free text")), 1},
    {CALL(LONG_NAME, ""), 1},
    {"<tool_call>\n<function=read   >\n" P("path", "a") "</function>\n</tool_call>", 1},          // trailing padding
    {"<tool_call>\n<function=read>\n\f" P("path", "a") "</function>\n</tool_call>", 1},          // \f between tags
    {CALL("mode", P("level", "1") P("opts", "{\"\\u006e\": 3}") P("pick", "[1]")), 1},            // an escaped key, decoded
    {CALL("edit", P("path", "a") P("edits", "[{\"oldText\": \"\xe0\xa0\x80\xed\x9f\xbf\xf0\x90\x80\x80\xf4\x8f\xbf\xbf\", "
                                           "\"newText\": \"\"}]")), 1},                                     // UTF-8 edges
};

static const char *const good_thinking[] = {
    "Plan: read the file.\n</think>\n\n" CALL("read", P("path", "a.py")),
    "I could write </tool_call> or <tool here.\n</think>\n\nDone.",
    "Now edit cli.py.", // the reply may end inside the reasoning
};

typedef struct {
    bool thinking;
    const char *text;
    const char *bad_at; // the text up to the first refused byte
} Bad;

#define PAD10 "          "
#define PAD123 PAD10 PAD10 PAD10 PAD10 PAD10 PAD10 PAD10 PAD10 PAD10 PAD10 PAD10 PAD10 "   "
#define EDIT0 "<tool_call>\n<function=edit>\n<parameter=path>\na\n</parameter>\n<parameter=edits>\n"

static const Bad bad[] = {
    {true, "Let me look.\n<tool_call>\n<function=read>", "Let me look.\n<tool_call"},                      // a call in reasoning
    {false, CALL("read", P("path", "a.py") "</parameter>\n"), "<tool_call>\n<function=read>\n<parameter=path>\na.py\n</parameter>\n</"},
    {false, EDIT0 "[{\"oldText\": \"a\nb\"", EDIT0 "[{\"oldText\": \"a"},                                    // a raw newline
    {false, "<tool_call>\n<function=grep>", "<tool_call>\n<function="},                                       // an unknown tool
    // edits missing: </function> may not even start (its prefix would be a dead end)
    {false, "<tool_call>\n<function=edit>\n" P("path", "a") "</function>", "<tool_call>\n<function=edit>\n<parameter=path>\na\n</parameter>\n<"},
    {false, CALL("read", P("path", "a")) "\nDone.", CALL("read", P("path", "a")) "\n"},                        // text after a call
    {false, "<tool_call>\n<function=edit>\n<parameter=edits>\n{\"oldText\"", "<tool_call>\n<function=edit>\n<parameter=edits>\n"},
    {false, "<tool_call>\n<function=read>\n<parameter=limit>\nfifty", "<tool_call>\n<function=read>\n<parameter=limit>\n"},
    {false, EDIT0 "[{\"oldText\": \"a\"}]", EDIT0 "[{\"oldText\": \"a\""},                                       // newText missing
    {false, "<tool_call>\n<function=write>\n<parameter=content>\nsee <parameter=x>", "<tool_call>\n<function=write>\n<parameter=content>\nsee <parameter"},
    // json_parse refuses these, and the parser would then keep the value as a string
    {false, EDIT0 "[{\"oldText\": \"a\", \"oldText\": \"b\"", EDIT0 "[{\"oldText\": \"a\", \"oldText"},        // a repeated key (at its closing quote)
    {false, EDIT0 "[{\"oldText\": \"\\udc00\"", EDIT0 "[{\"oldText\": \"\\ud"},                                  // a lone low surrogate
    {false, EDIT0 "[{\"oldText\": \"\\ud800x\"", EDIT0 "[{\"oldText\": \"\\ud800"},                              // a high one alone
    {false, EDIT0 "[{\"oldText\": \"\\ud800\\ud8", EDIT0 "[{\"oldText\": \"\\ud800\\ud"},                     // a high one, then another
    {false, EDIT0 "[{\"oldText\": \"\xff\"", EDIT0 "[{\"oldText\": \""},                                         // not UTF-8
    {false, EDIT0 "[{\"oldText\": \"\xc3(\"", EDIT0 "[{\"oldText\": \"\xc3"},
    // a padded name is the schema's parameter (the parser trims it): typed as it is
    {false, "<tool_call>\n<function=read>\n<parameter=offset >\nabc", "<tool_call>\n<function=read>\n<parameter=offset >\n"},
    {false, "<tool_call>\n<function=read>\n<parameter=of\nfset>", "<tool_call>\n<function=read>\n<parameter=of"}, // a newline in a name
    {false, "<tool_call>\n<function=mode>\n<parameter=level>\ntwo", "<tool_call>\n<function=mode>\n<parameter=level>\n"}, // an enum keeps its type
    {false, "<tool_call>\n<function=mode>\n<parameter=either>\nmaybe", "<tool_call>\n<function=mode>\n<parameter=either>\n"}, // a union without string
    {false, "<tool_call>\n<function=mode>\n<parameter=level>\n1.5", "<tool_call>\n<function=mode>\n<parameter=level>\n1"},     // an integer
    {false, "<tool_call>\n<function=rea x", "<tool_call>\n<function=rea"},                    // padding after a partial name
    {false, "<tool_call>\n<function=read  x", "<tool_call>\n<function=read  "},              // text after a name's padding
    // padding past GR_NAME: still the schema's offset, typed as a number
    {false, "<tool_call>\n<function=read>\n<parameter=offset" PAD123 ">\nabc", "<tool_call>\n<function=read>\n<parameter=offset" PAD123 ">\n"},
    {false, EDIT0 "[{\"oldText\": \"a\", \"\\u006fldText\"", EDIT0 "[{\"oldText\": \"a\", \"\\u006fldText"}, // escaped, repeated
    {false, "<tool_call>\n<function=mode>\n<parameter=opts>\n{\"\\u006e\": \"x\"", "<tool_call>\n<function=mode>\n<parameter=opts>\n{\"\\u006e\": "},
    {false, "<tool_call>\n<function=mode>\n<parameter=pick>\n\"k\"", "<tool_call>\n<function=mode>\n<parameter=pick>\n"}, // no string in the union
    {false, EDIT0 "[{\"oldText\": \"\xe0\x80", EDIT0 "[{\"oldText\": \"\xe0"},            // an overlong form
    {false, EDIT0 "[{\"oldText\": \"\xed\xa0", EDIT0 "[{\"oldText\": \"\xed"},            // a surrogate in UTF-8
    {false, EDIT0 "[{\"oldText\": \"\xf0\x8f", EDIT0 "[{\"oldText\": \"\xf0"},
    {false, EDIT0 "[{\"oldText\": \"\xf4\x90", EDIT0 "[{\"oldText\": \"\xf4"},            // past U+10FFFF
};

TEST(grammar_good_replies_valid_at_every_prefix_and_parse) {
    Arena a = {0};
    JValue *tools;
    Grammar *g = pi_grammar(&a, &tools);
    EXPECT_EQ(g->n_tools, 6);
    // a NUL in a required name is a name a call can write: required
    {
        JValue *t2;
        EXPECT_OK(json_parse(&a, S("[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\","
                                   "\"required\":[\"a\\u0000b\"],\"properties\":{\"a\\u0000b\":{\"type\":\"string\"}}}}}]"),
                             true, &t2));
        Grammar *g2 = grammar_build(&a, t2, NULL, NULL, 0);
        EXPECT(!may_end_after(g2, false, S("<tool_call>\n<function=f>\n<")));
        GState s2 = grammar_start(g2, false, S("<tool_call>\n<function=f>\n<"));
        EXPECT(!grammar_ok_bytes(g2, &s2, S("/")));
    }
    // " padded" is required but no call can name it (the parser trims the name): not required here
    EXPECT(may_end_after(g, false, S(CALL("mode", P("level", "1")))));
    for (size_t i = 0; i < COUNT(good); i++) {
        Str t = str(good[i].text);
        EXPECT_EQ(first_bad(g, false, t), -1);
        EXPECT_EQ(first_dead_end(g, false, t), -1);
        EXPECT(may_end_after(g, false, t));
        Parsed p = toolcall_parse(&a, t, tools);
        EXPECT_EQ(p.n_calls, good[i].calls);
        for (size_t k = 0; k < p.n_calls; k++) {
            JValue *args;
            EXPECT_OK(json_parse(&a, p.calls[k].arguments, true, &args));
            Str n = p.calls[k].name;
            if (str_eq(n, S("edit"))) EXPECT(json_is(json_get(args, "edits"), J_ARRAY));
            if (str_eq(n, S("read"))) {
                EXPECT(json_is(json_get(args, "path"), J_STRING));
                if (json_get(args, "offset")) EXPECT(json_is(json_get(args, "offset"), J_INTEGER));
                if (json_get(args, "limit")) EXPECT(json_is(json_get(args, "limit"), J_FLOAT));
            }
            if (str_eq(n, S("bash"))) EXPECT(json_is(json_get(args, "command"), J_STRING));
            if (str_eq(n, S("write"))) EXPECT(json_is(json_get(args, "content"), J_STRING));
            if (str_eq(n, S("mode"))) {
                EXPECT(json_is(json_get(args, "level"), J_INTEGER));
                if (json_get(args, "either")) EXPECT(json_is(json_get(args, "either"), J_BOOL));
                if (json_get(args, "note")) EXPECT(json_is(json_get(args, "note"), J_STRING)); // a union with string: text
                if (json_get(args, "opts")) EXPECT(json_is(json_get(json_get(args, "opts"), "n"), J_INTEGER));
            }
        }
    }
    for (size_t i = 0; i < COUNT(good_thinking); i++) {
        EXPECT_EQ(first_bad(g, true, str(good_thinking[i])), -1);
        EXPECT_EQ(first_dead_end(g, true, str(good_thinking[i])), -1);
        EXPECT(may_end_after(g, true, str(good_thinking[i])));
    }
    // a reply cut inside a call may not end there
    EXPECT(!may_end_after(g, false, S("<tool_call>\n<function=read>\n<parameter=path>\na")));
    EXPECT(may_end_after(g, false, S("<tool_c"))); // still text: may end
    arena_reset(&a);
}

TEST(grammar_malformed_refused_at_the_first_wrong_byte_no_dead_end) {
    Arena a = {0};
    JValue *tools;
    Grammar *g = pi_grammar(&a, &tools);
    for (size_t i = 0; i < COUNT(bad); i++) {
        Str t = str(bad[i].text);
        ptrdiff_t at = first_bad(g, bad[i].thinking, t);
        EXPECT_EQ(at, (ptrdiff_t)strlen(bad[i].bad_at));
        // every prefix it accepted can still be completed
        EXPECT_EQ(first_dead_end(g, bad[i].thinking, t), -1);
    }
    arena_reset(&a);
}

TEST(grammar_forced_opening_empty_tokens_and_the_end) {
    Arena a = {0};
    JValue *tools;
    Grammar *g = pi_grammar(&a, &tools);
    GState s = grammar_start(g, false, S("<tool_call>\n<function=bash>\n"));
    EXPECT(grammar_in_call(&s));
    EXPECT(grammar_ok_bytes(g, &s, S("<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>")));
    EXPECT(!grammar_ok_bytes(g, &s, S("</"))); // command is required: </function> may not start
    // a token with no bytes (an id past the tokenizer's) would not move the reply: refused
    EXPECT(!grammar_ok_bytes(g, &s, S("")));
    // without a tool a call could name, the grammar has none (the server then does not use it)
    Grammar *none = grammar_build(&a, NULL, NULL, NULL, 0);
    EXPECT_EQ(none->n_tools, 0);
    // after an end token the reply is over: nothing past it is checked
    GState e = grammar_start(g, false, S(CALL("read", P("path", "a"))));
    grammar_feed(g, &e, 0); // eos (pi_grammar's end token)
    EXPECT(grammar_ok_bytes(g, &e, S("<|im_start|>")));
    EXPECT(grammar_ok_bytes(g, &e, S("")));
    // a dead state accepts everything (the mask gives up rather than block the reply)
    GState d = grammar_start(g, false, S("<tool_call>\n<function=nope"));
    EXPECT(!grammar_in_call(&d));
    EXPECT(grammar_ok_bytes(g, &d, S("anything")));
    arena_reset(&a);
}

// A tool with more than 64 parameters: every one is typed (the required and repeated checks hold
// for the first 64).
TEST(grammar_many_parameters_typed) {
    Arena a = {0};
    Buf b = {0};
    buf_adds(&b, "[{\"type\":\"function\",\"function\":{\"name\":\"many\",\"parameters\":{\"type\":\"object\",\"properties\":{");
    for (int i = 0; i < 70; i++) buf_printf(&b, "%s\"p%d\":{\"type\":\"integer\"}", i ? "," : "", i);
    buf_adds(&b, "}}}}]");
    JValue *tools;
    EXPECT_OK(json_parse(&a, buf_str(&b), true, &tools));
    Grammar *g = grammar_build(&a, tools, NULL, NULL, 0);
    EXPECT_EQ(first_bad(g, false, S("<tool_call>\n<function=many>\n<parameter=p69>\nabc")),
              (ptrdiff_t)strlen("<tool_call>\n<function=many>\n<parameter=p69>\n"));
    EXPECT_EQ(first_bad(g, false, S(CALL("many", P("p69", "42")))), -1);
    buf_free(&b);
    arena_reset(&a);
}

// Random walks through the grammar, by bytes and by pieces of the format: at every prefix some byte
// is valid or the reply may end, and every complete reply parses into calls of known tools with
// their required arguments.
TEST(grammar_random_walks_no_dead_end) {
    Arena a = {0};
    JValue *tools;
    Grammar *g = pi_grammar(&a, &tools);
    static const char *const pieces[] = {
        "<parameter=", "</parameter>", "</function>", "</tool_call>", "<tool_call>", "<function=", "\n", " ", "\"",
        "{", "}", "[", "]", ",", ":", "\\", "\\u", "\\ud83d\\ude00", "\\u00e9", "\xc3\xa9", "0", "-1.5e3", "true", "null",
        "read", "bash", "edit", "write", "mode", "path", "edits", "oldText", "newText", "command", "level", "opts",
        "pick", "offset", "content", ">", "</think>", "x", "\t", "\r", LONG_NAME,
    };
    uint64_t r = 0x9e3779b97f4a7c15ull;
    size_t dead_ends = 0, completed = 0, bad_parse = 0;
    for (int walk = 0; walk < 20000; walk++) {
        bool thinking = walk % 4 == 0;
        Buf text = {0};
        if (walk % 2) buf_adds(&text, "<tool_call>\n<function=");
        GState s = grammar_start(g, thinking, buf_str(&text));
        for (int step = 0; step < 400; step++) {
            r = r * 6364136223846793005ull + 1442695040888963407ull;
            uint32_t x = (uint32_t)(r >> 33);
            char one;
            Str piece;
            if (x % 3) {
                piece = str(pieces[(x >> 2) % COUNT(pieces)]);
            } else {
                one = (char)(1 + (x >> 2) % 255);
                piece = str_n(&one, 1);
            }
            if (!grammar_ok_bytes(g, &s, piece)) continue;
            for (size_t i = 0; i < piece.len; i++) {
                grammar_feed_bytes(g, &s, str_n(piece.ptr + i, 1));
                buf_add(&text, piece.ptr + i, 1);
                bool any = grammar_may_end(&s);
                for (int c = 1; c < 256 && !any; c++) {
                    char cc = (char)c;
                    any = grammar_ok_bytes(g, &s, str_n(&cc, 1));
                }
                if (!any) dead_ends++;
            }
        }
        if (grammar_may_end(&s) && grammar_in_call(&s)) { // ended after a call: check what it parses to
            completed++;
            Str t = buf_str(&text);
            Str ans = t;
            ptrdiff_t th = str_find(t, S("</think>"), 0);
            if (thinking && th >= 0) ans = str_sub(t, (size_t)th + 8, t.len);
            Parsed p = toolcall_parse(&a, ans, tools);
            bool ok = p.n_calls > 0;
            for (size_t k = 0; k < p.n_calls && ok; k++) {
                JValue *args;
                if (json_parse(&a, p.calls[k].arguments, true, &args)) ok = false;
                else if (str_eq(p.calls[k].name, S("edit"))) ok = json_is(json_get(args, "edits"), J_ARRAY) && json_get(args, "path");
                else if (str_eq(p.calls[k].name, S("read"))) ok = json_is(json_get(args, "path"), J_STRING);
                else if (str_eq(p.calls[k].name, S("mode"))) ok = json_is(json_get(args, "level"), J_INTEGER);
            }
            bad_parse += !ok;
        }
        buf_free(&text);
    }
    dbg("grammar walks: %zu dead ends, %zu replies ended after a call, %zu parsed wrong\n", dead_ends, completed, bad_parse);
    EXPECT_EQ(dead_ends, 0);
    EXPECT_EQ(bad_parse, 0);
    EXPECT(completed > 0);
    arena_reset(&a);
}
