// Tests for src/json.c. Expected output is std.json.Stringify's.
#include "../json.h"
#include "../test.h"

#include <math.h>

static Str roundtrip(Arena *a, Buf *b, const char *text, bool nums) {
    JValue *v;
    if (json_parse(a, str(text), nums, &v)) return S("<error>");
    b->len = 0;
    json_write(b, v);
    return buf_str(b);
}

TEST(json_roundtrip) {
    Arena a = {0};
    Buf b = {0};
    EXPECT_STR(roundtrip(&a, &b, " { \"a\" : [1, 2.50, -0, 1e3, true, false, null], \"b\":{} , \"c\":[]} ", false),
               S("{\"a\":[1,2.50,-0,1e3,true,false,null],\"b\":{},\"c\":[]}"));
    EXPECT_STR(roundtrip(&a, &b, "{\"a\":[1,2.50,-0,1e3,1.5E-7,12345678901234567890,1e999]}", true),
               S("{\"a\":[1,2.5,-0,1000,0.00000015,12345678901234567890,1e999]}"));
    EXPECT_STR(roundtrip(&a, &b, "{\"z\":1,\"a\":2,\"m\":3}", true), S("{\"z\":1,\"a\":2,\"m\":3}"));
    EXPECT_STR(roundtrip(&a, &b, "\"\\u00e9\\ud83d\\ude00\\/\\\"\"", true), S("\"\xc3\xa9\xf0\x9f\x98\x80/\\\"\""));
    EXPECT_STR(roundtrip(&a, &b, "[[[[[\"deep\"]]]]]", true), S("[[[[[\"deep\"]]]]]"));
    EXPECT_STR(roundtrip(&a, &b, "-9223372036854775808", true), S("-9223372036854775808"));
    EXPECT_STR(roundtrip(&a, &b, "9223372036854775808", true), S("9223372036854775808"));
    buf_free(&b);
    arena_reset(&a);
}

TEST(json_numbers) {
    Arena a = {0};
    JValue *v;
    EXPECT_OK(json_parse(&a, S("[1, -0, 0.5, 1e2, 99999999999999999999]"), true, &v));
    EXPECT_EQ(v->arr.items[0]->kind, J_INTEGER);
    EXPECT_EQ(v->arr.items[0]->i, 1);
    EXPECT_EQ(v->arr.items[1]->kind, J_FLOAT);
    EXPECT(signbit(v->arr.items[1]->f));
    EXPECT_EQ(v->arr.items[2]->kind, J_FLOAT);
    EXPECT(v->arr.items[3]->f == 100.0);
    EXPECT_EQ(v->arr.items[4]->kind, J_NUMBER_STRING);
    EXPECT_OK(json_parse(&a, S("[1, 0.5]"), false, &v));
    EXPECT_EQ(v->arr.items[0]->kind, J_NUMBER_STRING);
    EXPECT_STR(v->arr.items[1]->s, S("0.5"));
    arena_reset(&a);
}

TEST(json_escaping) {
    Buf b = {0};
    json_write_string(&b, S("a\x00\x1f\x7f\"\\/\b\f\n\r\t\xc3\xa9"));
    EXPECT_STR(buf_str(&b), S("\"a\\u0000\\u001f\x7f\\\"\\\\/\\b\\f\\n\\r\\t\xc3\xa9\""));
    Arena a = {0};
    JValue *o = json_new(&a, J_OBJECT);
    JValue *f = json_new(&a, J_FLOAT);
    f->f = NAN;
    json_put(&a, o, S("nan"), f);
    JValue *g = json_new(&a, J_FLOAT);
    g->f = INFINITY;
    json_put(&a, o, S("inf"), g);
    JValue *h = json_new(&a, J_FLOAT);
    h->f = 1.0;
    json_put(&a, o, S("one"), h);
    json_put(&a, o, S("k\n"), json_string(&a, S("v")));
    json_put(&a, o, S("nan"), json_int(&a, -3)); // replaces in place
    JValue *arr = json_new(&a, J_ARRAY);
    json_push(&a, arr, json_bool(&a, true));
    json_push(&a, arr, json_new(&a, J_NULL));
    json_put(&a, o, S("arr"), arr);
    b.len = 0;
    json_write(&b, o);
    EXPECT_STR(buf_str(&b), S("{\"nan\":-3,\"inf\":inf,\"one\":1,\"k\\n\":\"v\",\"arr\":[true,null]}"));
    b.len = 0;
    json_write(&b, f);
    EXPECT_STR(buf_str(&b), S("\"nan\""));
    EXPECT_STR(json_strs(json_get(o, "k\n")), S("v"));
    EXPECT(json_get(o, "missing") == NULL && json_str(json_get(o, "one")) == NULL);
    EXPECT(json_is(json_get(o, "arr"), J_ARRAY));
    buf_free(&b);
    arena_reset(&a);
}

TEST(json_errors) {
    Arena a = {0};
    JValue *v;
    const char *bad[] = {
        "", " ", "{", "[1,]", "{\"a\":1,}", "{\"a\" 1}", "{a:1}", "[1 2]", "01", "1.", ".5", "1e", "-",
        "+1", "tru", "nul", "\"abc", "\"\\x\"", "\"\x01\"", "\"\\ud800\"", "\"\\udc00\"", "\"\\ud800\\u0041\"",
        "\"\xc0\x80\"", "\"\xed\xa0\x80\"", "\"\xf4\x90\x80\x80\"", "\"\xff\"", "[1]x", "{} {}", "NaN",
        "{\"a\":1,\"a\":2}", "[\"\\u12g4\"]", "\"\\u12\"",
    };
    for (size_t i = 0; i < COUNT(bad); i++) {
        Err e = json_parse(&a, str(bad[i]), true, &v);
        if (!e) dbg("accepted: %s\n", bad[i]);
        EXPECT(e != OK);
    }
    EXPECT_ERR(json_parse(&a, S("{\"a\":1,\"a\":2}"), true, &v), "DuplicateField");
    EXPECT_ERR(json_parse(&a, S("[1,"), true, &v), "UnexpectedEndOfInput");
    // Duplicates in a large object are found too.
    Buf b = {0};
    buf_adds(&b, "{");
    for (int k = 0; k < 100; k++) buf_printf(&b, "\"k%d\":%d,", k, k);
    buf_adds(&b, "\"k42\":0}");
    EXPECT_ERR(json_parse(&a, buf_str(&b), true, &v), "DuplicateField");
    // No nesting limit: 2000 levels parse (std.json has none).
    b.len = 0;
    for (int k = 0; k < 2000; k++) buf_addc(&b, '[');
    for (int k = 0; k < 2000; k++) buf_addc(&b, ']');
    EXPECT_OK(json_parse(&a, buf_str(&b), true, &v));
    b.len = 0;
    for (int k = 0; k < 1000; k++) buf_addc(&b, '[');
    for (int k = 0; k < 1000; k++) buf_addc(&b, ']');
    EXPECT_OK(json_parse(&a, buf_str(&b), true, &v));
    // A NUL inside the input is a syntax error, an escaped one is kept.
    EXPECT(json_parse(&a, str_n("[1]\0", 4), true, &v) != OK);
    EXPECT_OK(json_parse(&a, S("\"a\\u0000b\""), true, &v));
    EXPECT_EQ(v->s.len, 3);
    buf_free(&b);
    arena_reset(&a);
}

// std.json reports a duplicate key when the duplicate's value is complete, before any
// later syntax error, in small and large objects alike.
TEST(json_duplicate_reported_at_value) {
    Arena a = {0};
    JValue *v;
    EXPECT_ERR(json_parse(&a, S("{\"a\":1,\"a\":2,"), true, &v), "DuplicateField");
    Buf big = {0};
    buf_adds(&big, "{");
    for (int i = 0; i < 40; i++) buf_printf(&big, "\"k%d\":%d,", i, i);
    buf_adds(&big, "\"k7\":0 ,]");
    EXPECT_ERR(json_parse(&a, buf_str(&big), true, &v), "DuplicateField");
    big.len = 0;
    buf_adds(&big, "{");
    for (int i = 0; i < 40; i++) buf_printf(&big, "%s\"k%d\":{\"x\":%d}", i ? "," : "", i, i);
    buf_adds(&big, "}");
    EXPECT_OK(json_parse(&a, buf_str(&big), true, &v));
    EXPECT_EQ(v->obj.len, 40);
    buf_free(&big);
    arena_reset(&a);
}

// No nesting limit, as std.json: 5000 nested arrays and objects parse.
TEST(json_deep_nesting) {
    Arena a = {0};
    Buf b = {0};
    for (int i = 0; i < 5000; i++) buf_adds(&b, i % 2 ? "[" : "{\"k\":");
    buf_adds(&b, "1");
    for (int i = 4999; i >= 0; i--) buf_adds(&b, i % 2 ? "]" : "}");
    JValue *v;
    EXPECT_OK(json_parse(&a, buf_str(&b), true, &v));
    EXPECT(v->kind == J_OBJECT);
    buf_free(&b);
    arena_reset(&a);
}
