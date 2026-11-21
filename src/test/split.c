// Tests of the streamed reply split (src/split.h): what goes out early never changes and
// never includes a byte a later token could take back.
#include "../split.h"
#include "../test.h"

typedef struct {
    Buf reasoning, content;
} Sent;

// Appends piece to the splitter and collects what split_poll lets out.
static void feed(Splitter *sp, Sent *out, Str piece) {
    buf_addstr(&sp->text, piece);
    SplitPoll d = split_poll(sp, false);
    buf_addstr(&out->reasoning, d.reasoning);
    buf_addstr(&out->content, d.content);
}

static void finish(Splitter *sp, Sent *out) {
    SplitPoll d = split_poll(sp, true);
    buf_addstr(&out->reasoning, d.reasoning);
    buf_addstr(&out->content, d.content);
}

static void sent_free(Sent *s, Splitter *sp) {
    buf_free(&s->reasoning);
    buf_free(&s->content);
    buf_free(&sp->text);
}

// Every way of cutting the text in three pieces streams the same reasoning and content as
// the whole text at once.
TEST(split_reasoning_and_answer_across_chunk_boundaries) {
    Str text = S("\n  Let me think: caf\xc3\xa9 \xe2\x9c\x93.\n</think>\n\nThe answer is 42 \xf0\x9f\x98\x80.  \n");
    for (size_t i = 0; i <= text.len; i++)
        for (size_t j = i; j <= text.len; j++) {
            Splitter sp = {.thinking = true};
            Sent s = {0};
            feed(&sp, &s, str_sub(text, 0, i));
            feed(&sp, &s, str_sub(text, i, j));
            feed(&sp, &s, str_sub(text, j, text.len));
            finish(&sp, &s);
            if (!str_eq(buf_str(&s.content), S("The answer is 42 \xf0\x9f\x98\x80.")))
                dbg("cut at %zu, %zu: \"%.*s\"\n", i, j, (int)s.content.len, s.content.ptr);
            EXPECT_STR(buf_str(&s.reasoning), S("Let me think: caf\xc3\xa9 \xe2\x9c\x93."));
            EXPECT_STR(buf_str(&s.content), S("The answer is 42 \xf0\x9f\x98\x80."));
            sent_free(&s, &sp);
        }
}

TEST(split_holds_back_a_partial_think_close) {
    Splitter sp = {.thinking = true};
    Sent s = {0};
    feed(&sp, &s, S("abc</thi"));
    EXPECT_STR(buf_str(&s.reasoning), S("abc"));
    feed(&sp, &s, S("nk>x"));
    EXPECT_STR(buf_str(&s.reasoning), S("abc"));
    finish(&sp, &s);
    EXPECT_STR(buf_str(&s.content), S("x"));
    sent_free(&s, &sp);
}

TEST(split_holds_back_a_tool_call_start) {
    Splitter sp = {0};
    Sent s = {0};
    feed(&sp, &s, S("Sure.\n<tool_"));
    EXPECT_STR(buf_str(&s.content), S("Sure."));
    feed(&sp, &s, S("call>\n<function=f>\n"));
    EXPECT_STR(buf_str(&s.content), S("Sure."));
    finish(&sp, &s);
    EXPECT_STR(buf_str(&s.content), S("Sure."));
    // a tail that turns out not to be a tag goes out
    Splitter sp2 = {0};
    Sent s2 = {0};
    feed(&sp2, &s2, S("a <tool"));
    EXPECT_STR(buf_str(&s2.content), S("a"));
    feed(&sp2, &s2, S("box>!"));
    EXPECT_STR(buf_str(&s2.content), S("a <toolbox>!"));
    sent_free(&s, &sp);
    sent_free(&s2, &sp2);
}

TEST(split_holds_back_a_partial_utf8_character) {
    Splitter sp = {0};
    Sent s = {0};
    feed(&sp, &s, S("caf\xc3"));
    EXPECT_STR(buf_str(&s.content), S("caf"));
    feed(&sp, &s, S("\xa9 \xf0\x9f"));
    EXPECT_STR(buf_str(&s.content), S("caf\xc3\xa9"));
    feed(&sp, &s, S("\x98"));
    EXPECT_STR(buf_str(&s.content), S("caf\xc3\xa9"));
    feed(&sp, &s, S("\x80!"));
    EXPECT_STR(buf_str(&s.content), S("caf\xc3\xa9 \xf0\x9f\x98\x80!"));
    // a reply that ends inside a character sends its bytes at the final flush (the JSON
    // writer turns them into U+FFFD)
    feed(&sp, &s, S(" \xe2\x9c"));
    EXPECT_STR(buf_str(&s.content), S("caf\xc3\xa9 \xf0\x9f\x98\x80!"));
    finish(&sp, &s);
    EXPECT_STR(buf_str(&s.content), S("caf\xc3\xa9 \xf0\x9f\x98\x80! \xe2\x9c"));
    sent_free(&s, &sp);
}

TEST(split_holds_back_a_stop_string_prefix) {
    const Str stops[2] = {S("STOP"), S("\nUser:")};
    Splitter sp = {.stops = stops, .n_stops = 2};
    Sent s = {0};
    feed(&sp, &s, S("hello ST"));
    EXPECT_STR(buf_str(&s.content), S("hello"));
    EXPECT(split_cut_stop(&sp) == NULL);
    feed(&sp, &s, S("ART and\nUs"));
    EXPECT_STR(buf_str(&s.content), S("hello START and"));
    // the full stop string arrives: the text is cut where it begins, nothing of it was sent
    buf_adds(&sp.text, "er: hi");
    const Str *st = split_cut_stop(&sp);
    EXPECT(st == &stops[1]);
    EXPECT_STR(buf_str(&sp.text), S("hello START and"));
    finish(&sp, &s);
    EXPECT_STR(buf_str(&s.content), S("hello START and"));
    sent_free(&s, &sp);
}

// The stop string arrives split across tokens at every position: the streamed content is
// the text before it, and no byte of it goes out early.
TEST(split_stop_string_across_chunk_boundaries) {
    const Str stops[1] = {S("<|end|>")};
    Str text = S("one two<|end|>three");
    for (size_t i = 1; i < text.len; i++) {
        Splitter sp = {.stops = stops, .n_stops = 1};
        Sent s = {0};
        const Str *hit = NULL;
        for (size_t k = 0; k < text.len && !hit; k += i) {
            // as the server's decode loop: append the token, cut at a stop string, then poll
            buf_addstr(&sp.text, str_sub(text, k, MIN(k + i, text.len)));
            hit = split_cut_stop(&sp);
            feed(&sp, &s, S(""));
            EXPECT(str_find(buf_str(&s.content), S("<"), 0) < 0);
        }
        EXPECT(hit != NULL);
        finish(&sp, &s);
        EXPECT_STR(buf_str(&s.content), S("one two"));
        sent_free(&s, &sp);
    }
}

TEST(split_stable_prefix) {
    const Str stops[1] = {S("abcd")};
    EXPECT_EQ(split_stable(S("x abc"), S("</think>"), stops, 1), 1);
    EXPECT_EQ(split_stable(S("x ab"), S("</think>"), NULL, 0), 4);
    EXPECT_EQ(split_stable(S("x </"), S("</think>"), stops, 1), 1);
    EXPECT_EQ(split_stable(S("x \xe2\x9c"), S("</think>"), NULL, 0), 1);
    EXPECT_EQ(split_stable(S("x \xe2\x9c\x93"), S("</think>"), NULL, 0), 5);
}

// Stop strings that begin with whitespace (which the streamed content leaves out at its
// start): across every chunk boundary the stream sends exactly the non-streamed answer.
TEST(split_whitespace_leading_stop_strings) {
    const Str stops[2] = {S("\nObservation"), S("  END")};
    static const char *const texts[] = {"\nObservation: x", "Hi.\nObservation: x", "\n\nHi  ENDING", "  END", "a\nObs b"};
    static const char *const want[] = {"", "Hi.", "\n\nHi", "", "a\nObs b"};
    for (size_t t = 0; t < COUNT(texts); t++) {
        Str text = str(texts[t]);
        for (size_t i = 1; i <= text.len; i++) {
            Splitter sp = {.stops = stops, .n_stops = 2};
            Sent s = {0};
            for (size_t k = 0; k < text.len; k += i) {
                buf_addstr(&sp.text, str_sub(text, k, MIN(k + i, text.len)));
                if (split_cut_stop(&sp)) break;
                feed(&sp, &s, S(""));
            }
            finish(&sp, &s);
            // the non-streamed answer: the cut text, trimmed
            Str whole = str_trim(buf_str(&sp.text));
            EXPECT_STR(buf_str(&s.content), whole);
            Str w = str_trim(str(want[t]));
            EXPECT_STR(buf_str(&s.content), w);
            sent_free(&s, &sp);
        }
    }
}
