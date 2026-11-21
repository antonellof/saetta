// Tests of the JSON string writer of src/chat.h.
#include "../chat.h"
#include "../test.h"

static Str js(Buf *b, Str s) {
    b->len = 0;
    chat_json_string(b, s);
    return buf_str(b);
}

// Well-formed UTF-8 is written raw; each maximal ill-formed subpart becomes one U+FFFD.
TEST(chat_json_string_replaces_invalid_utf8) {
    Buf b = {0};
    EXPECT_STR(js(&b, S("caf\xc3\xa9 \xf0\x9f\x98\x80 \xef\xbf\xbf\x7f")), S("\"caf\xc3\xa9 \xf0\x9f\x98\x80 \xef\xbf\xbf\x7f\""));
    EXPECT_STR(js(&b, S("a\xc3")), S("\"a\xef\xbf\xbd\""));                   // cut after the lead byte
    EXPECT_STR(js(&b, S("a\xf0\x9f\x98")), S("\"a\xef\xbf\xbd\""));           // cut inside a 4-byte char
    EXPECT_STR(js(&b, S("\x80" "b")), S("\"\xef\xbf\xbd" "b\""));              // stray continuation
    EXPECT_STR(js(&b, S("\xc0\xaf")), S("\"\xef\xbf\xbd\xef\xbf\xbd\""));     // overlong: two bad bytes
    EXPECT_STR(js(&b, S("\xed\xa0\x80")), S("\"\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd\"")); // a surrogate
    EXPECT_STR(js(&b, S("\xf4\x90\x80\x80")), S("\"\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd\"")); // > U+10FFFF
    EXPECT_STR(js(&b, S("\xe2\x9c\"")), S("\"\xef\xbf\xbd\\\"\""));            // a quote after a cut char
    EXPECT_STR(js(&b, S("\xff\n")), S("\"\xef\xbf\xbd\\n\""));
    buf_free(&b);
}
