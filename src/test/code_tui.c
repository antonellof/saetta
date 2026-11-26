// Tests for src/code_tui_util.c: fuzzyMatch and fuzzyFilter's scoring (pi-tui fuzzy.ts), which
// orders the /model selector's matches.
#include "../code_tui.h"
#include "../test.h"

#include <math.h>

static double score(const char *q, const char *t) {
    double s = 1e9;
    return tui_fuzzy_score(str(q), str(t), &s) ? s : 1e9;
}

TEST(code_tui_fuzzy_match_in_order) {
    double s;
    EXPECT(tui_fuzzy_match(S("omd"), S("other-model"), &s));
    EXPECT(tui_fuzzy_match(S("OTHER"), S("other-model"), &s));
    EXPECT(!tui_fuzzy_match(S("dmo"), S("other-model"), &s));
    EXPECT(!tui_fuzzy_match(S("other-models"), S("other-model"), &s));
    EXPECT(tui_fuzzy_match(S(""), S("x"), &s) && s == 0);
    // letters and digits swapped, at a cost of 5
    double a, b;
    EXPECT(tui_fuzzy_match(S("gpt5"), S("5gpt"), &a) && tui_fuzzy_match(S("5gpt"), S("5gpt"), &b) && a == b + 5);
}

TEST(code_tui_fuzzy_score_values) {
    // as fuzzy.ts computes them: "ab" in "ab": a run from index -1 (-5) and a word start (-10) at 0,
    // the run (-10) and 0.1 at 1, exact (-100)
    EXPECT(fabs(score("ab", "ab") - -124.9) < 1e-9);
    // "ac" in "abc": -15 at 0, a gap of one (+2) and 0.2 at 2
    EXPECT(fabs(score("ac", "abc") - -12.8) < 1e-9);
    // tokens split at spaces and slashes, every one must match, scores summed
    EXPECT(score("ab/c", "ab c") == score("ab", "ab c") + score("c", "ab c"));
    EXPECT(score("ab zz", "ab c") == 1e9);
}

// The /model regression: a pack whose id is a long path matched "other-model" letter by letter
// and, sorted first as the current model, was selected; the whole id must rank first.
TEST(code_tui_fuzzy_whole_id_ranks_first) {
    const char *pack = "saetta saetta//private/var/folders/ld/5jyl_hyd0_72d6q_p5r67slm0000gn/T/share-d08jibe3/pack saetta "
                       "/private/var/folders/ld/5jyl_hyd0_72d6q_p5r67slm0000gn/T/share-d08jibe3/pack mock-model";
    const char *other = "other other/other-model other other-model Other";
    EXPECT(score("other-model", pack) < 1e9); // it matches, scattered
    EXPECT(score("other-model", other) < score("other-model", pack));
    EXPECT(score("mock", pack) < score("mock", other));
}
