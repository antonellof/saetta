// Byte-level BPE tokenizer (see tokenizer.h).
// Adapted from sushi src/tokenizer.zig (MIT, Copyright (c) 2026 Theinruj Toranavikrai, David Dalcu; d5e876e).
#include "tokenizer.h"

#include <stdlib.h>

#include "json.h"
#include "metal/shim.h"
#include "unicode.h"

// The Split regex this pre-tokenizer implements: the official tokenizer.json at HF
// commit 812d96e. The mlx-community pack ships the older Qwen2 regex (`\p{L}+`, no
// `\p{M}`) with the same vocab and merges; it is accepted and the official grammar is
// applied.
static const char qwen_split_mlx[] = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
static const char qwen_split[] = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
// The o200k split regex (GPT-4o), as Inkling 2's tokenizer.json (HF thinkingmachines/Inkling-2-Mini
// b85c3cf) spells it; llama.cpp calls it LLAMA_VOCAB_PRE_TYPE_GPT4O.
static const char o200k_split[] = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

// ---- UTF-8 helpers, as std.unicode ------------------------------------------------

// utf8ByteSequenceLength: 1..4, or 0 for an invalid start byte.
static int utf8_seq_len(uint8_t b) {
    if (b <= 0x7F) return 1;
    if (b >= 0xC0 && b <= 0xDF) return 2;
    if (b >= 0xE0 && b <= 0xEF) return 3;
    if (b >= 0xF0 && b <= 0xF7) return 4;
    return 0;
}

// utf8Decode of exactly n bytes (n = the sequence length of s[0]); false on error.
static bool utf8_decode(const uint8_t *s, int n, uint32_t *cp) {
    uint32_t v;
    switch (n) {
    case 1: *cp = s[0]; return true;
    case 2:
        v = s[0] & 0x1F;
        if ((s[1] & 0xC0) != 0x80) return false;
        v = v << 6 | (s[1] & 0x3F);
        if (v < 0x80) return false;
        break;
    case 3:
        v = s[0] & 0x0F;
        for (int k = 1; k < 3; k++) {
            if ((s[k] & 0xC0) != 0x80) return false;
            v = v << 6 | (s[k] & 0x3F);
        }
        if (v < 0x800 || (v >= 0xD800 && v <= 0xDFFF)) return false;
        break;
    case 4:
        v = s[0] & 0x07;
        for (int k = 1; k < 4; k++) {
            if ((s[k] & 0xC0) != 0x80) return false;
            v = v << 6 | (s[k] & 0x3F);
        }
        if (v < 0x10000 || v > 0x10FFFF) return false;
        break;
    default: return false;
    }
    *cp = v;
    return true;
}

// utf8Encode for the code points of the byte table (all below U+0800).
static int utf8_encode(uint32_t c, char *out) {
    if (c < 0x80) {
        out[0] = (char)c;
        return 1;
    }
    out[0] = (char)(0xC0 | (c >> 6));
    out[1] = (char)(0x80 | (c & 0x3F));
    return 2;
}

// GPT-2 byte to code point table: printable bytes map to themselves, the rest from U+0100 up.
static void bytes_to_unicode(uint32_t table[256]) {
    uint32_t n = 256;
    for (uint32_t b = 0; b < 256; b++) {
        if ((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)) {
            table[b] = b;
        } else {
            table[b] = n++;
        }
    }
}

static void *id_val(uint32_t id) { return (void *)(uintptr_t)id; }

static bool vocab_get(const StrMap *m, Str key, uint32_t *id) {
    void **v = strmap_get(m, key);
    if (!v) return false;
    *id = (uint32_t)(uintptr_t)*v;
    return true;
}

static bool json_index(const JValue *v, size_t *out) {
    if (!v || v->kind != J_INTEGER || v->i < 0) return false;
    *out = (size_t)v->i;
    return true;
}

// ---- load -----------------------------------------------------------------------

Err tokenizer_load(Arena *a, const char *dir, Tokenizer *t) {
    char *path = path_join(a, dir, "tokenizer.json");
    char *bytes;
    size_t len;
    TRY(read_file(path, (size_t)64 << 20, &bytes, &len));
    // keep the bytes for the tokenizer's lifetime (they are not freed)
    return tokenizer_from_json(a, str_n(bytes, len), t);
}

Err tokenizer_from_json(Arena *a, Str bytes, Tokenizer *t) {
    JValue *root;
    TRY(json_parse(a, bytes, true, &root));
    JValue *pre = json_get(root, "pre_tokenizer");
    if (!pre) return "BadTokenizer";
    JValue *seqv = json_get(pre, "pretokenizers");
    if (!seqv || seqv->kind != J_ARRAY || seqv->arr.len == 0) return "BadTokenizer";
    JValue *pat = json_get(seqv->arr.items[0], "pattern");
    JValue *rxv = json_get(pat, "Regex");
    if (!rxv || rxv->kind != J_STRING) return "BadTokenizer";
    Str rx = rxv->s;
    if (seqv->arr.len != 2) return "UnsupportedPreTokenizer";
    PreTokenizer kind;
    if (str_eq(rx, S(qwen_split)) || str_eq(rx, S(qwen_split_mlx))) kind = PRE_QWEN;
    else if (str_eq(rx, S(o200k_split))) kind = PRE_O200K;
    else return "UnsupportedPreTokenizer";
    // the second step maps bytes to code points and nothing else
    JValue *bl = seqv->arr.items[1];
    if (!str_eqz(json_strs(json_get(bl, "type")), "ByteLevel")) return "UnsupportedPreTokenizer";
    JValue *ur = json_get(bl, "use_regex"), *aps = json_get(bl, "add_prefix_space");
    // HF's defaults: use_regex true (a second GPT-2 split) and add_prefix_space true when missing
    if (!(ur && ur->kind == J_BOOL && !ur->b) || (aps && aps->kind == J_BOOL && aps->b)) return "UnsupportedPreTokenizer";
    JValue *norm = json_get(root, "normalizer");
    bool nfc = false;
    if (norm && norm->kind != J_NULL) {
        if (!str_eqz(json_strs(json_get(norm, "type")), "NFC")) return "UnsupportedNormalizer";
        nfc = true;
    }
    JValue *model = json_get(root, "model");
    if (!model || model->kind != J_OBJECT) return "BadTokenizer";
    if (!str_eqz(json_strs(json_get(model, "type")), "BPE")) return "UnsupportedModel";
    JValue *im = json_get(model, "ignore_merges");

    memset(t, 0, sizeof *t);
    t->pre = kind;
    t->nfc = nfc;
    t->ignore_merges = im && im->kind == J_BOOL && im->b;
    bytes_to_unicode(t->byte_cp);
    for (int k = 0; k < 512; k++) t->cp_byte[k] = -1;
    for (int b = 0; b < 256; b++) t->cp_byte[t->byte_cp[b]] = (int16_t)b;

    JValue *vocab = json_get(model, "vocab");
    JValue *addedv = json_get(root, "added_tokens");
    if (!vocab || vocab->kind != J_OBJECT || !addedv || addedv->kind != J_ARRAY) return "BadTokenizer";
    size_t max_id = 0, id;
    for (size_t k = 0; k < vocab->obj.len; k++) {
        if (!json_index(vocab->obj.items[k].val, &id)) return "BadTokenizer";
        max_id = MAX(max_id, id);
    }
    for (size_t k = 0; k < addedv->arr.len; k++) {
        if (!json_index(json_get(addedv->arr.items[k], "id"), &id)) return "BadTokenizer";
        max_id = MAX(max_id, id);
    }
    t->n_ids = max_id + 1;
    t->id_text = ARENA_ARRAY(a, Str, t->n_ids);
    for (size_t k = 0; k < t->n_ids; k++) t->id_text[k] = S("");
    for (size_t k = 0; k < vocab->obj.len; k++) {
        JMember *e = &vocab->obj.items[k];
        id = (size_t)e->val->i;
        strmap_put(&t->vocab, e->key, id_val((uint32_t)id));
        t->id_text[id] = e->key;
    }
    JValue *merges = json_get(model, "merges");
    if (!merges || merges->kind != J_ARRAY) return "BadTokenizer";
    for (size_t r = 0; r < merges->arr.len; r++) {
        JValue *m = merges->arr.items[r];
        Str key;
        if (m->kind == J_STRING) {
            key = m->s;
        } else if (m->kind == J_ARRAY && m->arr.len >= 2 && m->arr.items[0]->kind == J_STRING && m->arr.items[1]->kind == J_STRING) {
            Str l = m->arr.items[0]->s, rt = m->arr.items[1]->s;
            char *k = arena_alloc(a, l.len + 1 + rt.len + 1);
            memcpy(k, l.ptr, l.len);
            k[l.len] = ' ';
            memcpy(k + l.len + 1, rt.ptr, rt.len);
            key = str_n(k, l.len + 1 + rt.len);
        } else {
            return "BadTokenizer";
        }
        strmap_put(&t->merges, key, id_val((uint32_t)r));
        t->max_merge_key = MAX(t->max_merge_key, key.len);
    }
    t->n_specials = addedv->arr.len;
    t->specials = ARENA_ARRAY(a, Special, t->n_specials);
    for (size_t k = 0; k < t->n_specials; k++) {
        JValue *ad = addedv->arr.items[k];
        JValue *c = json_get(ad, "content");
        if (!c || c->kind != J_STRING) return "BadTokenizer";
        t->specials[k] = (Special){c->s, (uint32_t)json_get(ad, "id")->i};
        t->id_text[t->specials[k].id] = c->s;
    }
    // Longest first, so a special that is a prefix of another never wins (stable sort).
    for (size_t k = 1; k < t->n_specials; k++) {
        Special x = t->specials[k];
        size_t j = k;
        while (j > 0 && x.text.len > t->specials[j - 1].text.len) {
            t->specials[j] = t->specials[j - 1];
            j--;
        }
        t->specials[j] = x;
    }
    return OK;
}

bool tokenizer_token_id(const Tokenizer *t, Str text, uint32_t *id) {
    for (size_t k = 0; k < t->n_specials; k++)
        if (str_eq(t->specials[k].text, text)) {
            *id = t->specials[k].id;
            return true;
        }
    return vocab_get(&t->vocab, text, id);
}

static const Special *special_at(const Tokenizer *t, Str s) {
    for (size_t k = 0; k < t->n_specials; k++)
        if (str_starts(s, t->specials[k].text)) return &t->specials[k];
    return NULL;
}

Err tokenizer_encode(const Tokenizer *t, Str text, VecU32 *out) {
    size_t seg_start = 0, i = 0;
    while (i < text.len) {
        if (text.ptr[i] == '<') {
            const Special *s = special_at(t, str_sub(text, i, text.len));
            if (s) {
                TRY(tokenizer_encode_text(t, str_sub(text, seg_start, i), out));
                vec_push(out, s->id);
                i += s->text.len;
                seg_start = i;
                continue;
            }
        }
        i += 1;
    }
    return tokenizer_encode_text(t, str_sub(text, seg_start, text.len), out);
}

// ---- BPE --------------------------------------------------------------------------

typedef struct {
    uint32_t start, end;
    int32_t prev, next;
    uint32_t ver;
} Node;

typedef struct {
    uint32_t rank, left, right, lver, rver;
} Cand;

typedef VEC(Node) VecNode;
typedef VEC(Cand) VecCand;

static bool cand_lt(Cand a, Cand b) {
    if (a.rank != b.rank) return a.rank < b.rank;
    return a.left < b.left;
}

// Binary min-heap (std.PriorityQueue order: rank, then left).
static void heap_push(VecCand *h, Cand c) {
    vec_push(h, c);
    size_t i = h->len - 1;
    while (i > 0) {
        size_t p = (i - 1) / 2;
        if (!cand_lt(h->ptr[i], h->ptr[p])) break;
        Cand x = h->ptr[i];
        h->ptr[i] = h->ptr[p];
        h->ptr[p] = x;
        i = p;
    }
}

static bool heap_pop(VecCand *h, Cand *out) {
    if (h->len == 0) return false;
    *out = h->ptr[0];
    h->ptr[0] = h->ptr[--h->len];
    size_t i = 0;
    for (;;) {
        size_t l = 2 * i + 1, r = l + 1, m = i;
        if (l < h->len && cand_lt(h->ptr[l], h->ptr[m])) m = l;
        if (r < h->len && cand_lt(h->ptr[r], h->ptr[m])) m = r;
        if (m == i) break;
        Cand x = h->ptr[i];
        h->ptr[i] = h->ptr[m];
        h->ptr[m] = x;
        i = m;
    }
    return true;
}

static bool rank_of(const Tokenizer *t, Str w, const Node *nodes, uint32_t l, uint32_t r, uint32_t *rk) {
    size_t al = nodes[l].end - nodes[l].start, bl = nodes[r].end - nodes[r].start;
    if (al + bl + 1 > t->max_merge_key) return false;
    char buf[1024];
    memcpy(buf, w.ptr + nodes[l].start, al);
    buf[al] = ' ';
    memcpy(buf + al + 1, w.ptr + nodes[r].start, bl);
    return vocab_get(&t->merges, str_n(buf, al + 1 + bl), rk);
}

static void push(const Tokenizer *t, VecCand *heap, Str w, const Node *nodes, uint32_t l, uint32_t r) {
    uint32_t rk;
    if (rank_of(t, w, nodes, l, r, &rk)) heap_push(heap, (Cand){rk, l, r, nodes[l].ver, nodes[r].ver});
}

// Lowest-rank adjacent pair first, leftmost on ties, until no pair has a rank.
static Err bpe(const Tokenizer *t, Str w, VecNode *nodes, VecCand *heap, VecU32 *out) {
    uint32_t whole;
    if (t->ignore_merges && vocab_get(&t->vocab, w, &whole)) {
        vec_push(out, whole);
        return OK;
    }
    nodes->len = 0;
    heap->len = 0;
    size_t i = 0;
    while (i < w.len) {
        int n = utf8_seq_len((uint8_t)w.ptr[i]);
        if (n == 0) n = 1;
        int32_t k = (int32_t)nodes->len;
        vec_push(nodes, ((Node){(uint32_t)i, (uint32_t)(i + n), k - 1, i + n < w.len ? k + 1 : -1, 0}));
        i += n;
    }
    for (size_t j = 1; j < nodes->len; j++) push(t, heap, w, nodes->ptr, (uint32_t)(j - 1), (uint32_t)j);
    Cand c;
    while (heap_pop(heap, &c)) {
        Node *ns = nodes->ptr;
        if (ns[c.left].ver != c.lver || ns[c.right].ver != c.rver) continue;
        ns[c.left].end = ns[c.right].end;
        ns[c.left].ver += 1;
        ns[c.right].ver += 1;
        ns[c.left].next = ns[c.right].next;
        if (ns[c.right].next >= 0) ns[ns[c.right].next].prev = (int32_t)c.left;
        if (ns[c.left].prev >= 0) push(t, heap, w, ns, (uint32_t)ns[c.left].prev, c.left);
        if (ns[c.left].next >= 0) push(t, heap, w, ns, c.left, (uint32_t)ns[c.left].next);
    }
    for (int32_t cur = 0; cur >= 0; cur = nodes->ptr[cur].next) {
        Node n = nodes->ptr[cur];
        uint32_t id;
        if (!vocab_get(&t->vocab, str_n(w.ptr + n.start, n.end - n.start), &id)) return "TokenNotInVocab";
        vec_push(out, id);
    }
    return OK;
}

static bool all_ascii(Str s) {
    for (size_t i = 0; i < s.len; i++)
        if ((uint8_t)s.ptr[i] >= 0x80) return false;
    return true;
}

Err tokenizer_encode_text(const Tokenizer *t, Str raw, VecU32 *out) {
    char *nfc_buf = NULL;
    Str text = raw;
    if (t->nfc && !all_ascii(raw)) {
        // NFC can grow text by up to 3x in UTF-8; ask for the length, then fill.
        size_t n = sx_nfc(raw.ptr, raw.len, NULL, 0);
        if (n != SIZE_MAX) { // invalid UTF-8: encode the bytes as they are
            nfc_buf = xmalloc(n ? n : 1);
            sx_nfc(raw.ptr, raw.len, nfc_buf, n);
            text = str_n(nfc_buf, n);
        }
    }
    Buf word = {0};
    VecNode nodes = {0};
    VecCand heap = {0};
    Err err = OK;
    size_t pos = 0;
    while (pos < text.len) {
        size_t end = t->pre == PRE_O200K ? tokenizer_next_piece_o200k(text, pos) : tokenizer_next_piece(text, pos);
        word.len = 0;
        for (size_t k = pos; k < end; k++) {
            char b[4];
            int n = utf8_encode(t->byte_cp[(uint8_t)text.ptr[k]], b);
            buf_add(&word, b, (size_t)n);
        }
        err = bpe(t, buf_str(&word), &nodes, &heap, out);
        if (err) break;
        pos = end;
    }
    buf_free(&word);
    vec_free(&nodes);
    vec_free(&heap);
    free(nfc_buf);
    return err;
}

void tokenizer_decode_append(const Tokenizer *t, uint32_t id, Buf *out) {
    if (id >= t->n_ids) return;
    Str s = t->id_text[id];
    for (size_t k = 0; k < t->n_specials; k++)
        if (t->specials[k].id == id) {
            buf_addstr(out, s);
            return;
        }
    // Utf8View.initUnchecked: the text is valid UTF-8 (it came from JSON)
    size_t i = 0;
    while (i < s.len) {
        int n = utf8_seq_len((uint8_t)s.ptr[i]);
        if (n == 0 || i + (size_t)n > s.len) n = 1;
        uint32_t cp = (uint8_t)s.ptr[i];
        if (n > 1) {
            cp = (uint8_t)s.ptr[i] & (0x7F >> n);
            for (int k = 1; k < n; k++) cp = cp << 6 | ((uint8_t)s.ptr[i + k] & 0x3F);
        }
        buf_addc(out, cp < 512 && t->cp_byte[cp] >= 0 ? (char)t->cp_byte[cp] : '?');
        i += (size_t)n;
    }
}

// ---- pre-tokenizer: qwen_split as a state machine, alternatives tried in order --------

typedef struct {
    uint32_t cp;
    size_t len;
} Cp;

static bool cp_at(Str s, size_t i, Cp *c) {
    if (i >= s.len) return false;
    const uint8_t *p = (const uint8_t *)s.ptr + i;
    int n = utf8_seq_len(p[0]);
    uint32_t cp;
    if (n == 0 || i + (size_t)n > s.len || !utf8_decode(p, n, &cp)) {
        *c = (Cp){p[0], 1};
        return true;
    }
    *c = (Cp){cp, (size_t)n};
    return true;
}

static bool is_l(uint32_t cp) { return unicode_class(cp) == UNICODE_LETTER; }
static bool is_n(uint32_t cp) { return unicode_class(cp) == UNICODE_NUMBER; }
static bool is_lm(uint32_t cp) {
    Class c = unicode_class(cp);
    return c == UNICODE_LETTER || c == UNICODE_MARK;
}
static bool is_nl(uint32_t cp) { return cp == '\r' || cp == '\n'; }
static bool is_punct(uint32_t cp) { return !unicode_space(cp) && unicode_class(cp) == UNICODE_OTHER; }

static inline size_t run_of(Str s, size_t from, bool (*pred)(uint32_t)) {
    size_t i = from;
    Cp c;
    while (cp_at(s, i, &c)) {
        if (!pred(c.cp)) break;
        i += c.len;
    }
    return i;
}

static char to_lower(char ch) { return ch >= 'A' && ch <= 'Z' ? (char)(ch + 32) : ch; }

size_t tokenizer_next_piece(Str s, size_t i) {
    Cp c0, c;
    cp_at(s, i, &c0);
    // 1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (s.ptr[i] == '\'' && i + 1 < s.len) {
        char a = to_lower(s.ptr[i + 1]);
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
        if (i + 2 < s.len) {
            char b = to_lower(s.ptr[i + 2]);
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
        }
    }
    // 2: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
    if (!is_nl(c0.cp) && !is_l(c0.cp) && !is_n(c0.cp)) {
        if (cp_at(s, i + c0.len, &c) && is_lm(c.cp)) return run_of(s, i + c0.len, is_lm);
    }
    if (is_lm(c0.cp)) return run_of(s, i, is_lm);
    // 3: \p{N}
    if (is_n(c0.cp)) return i + c0.len;
    // 4: ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*`
    {
        size_t p = s.ptr[i] == ' ' ? i + 1 : i;
        if (cp_at(s, p, &c) && is_punct(c.cp)) {
            size_t e = run_of(s, p, is_punct);
            while (e < s.len && (s.ptr[e] == '\r' || s.ptr[e] == '\n')) e += 1;
            return e;
        }
    }
    // 5: \s*[\r\n]+ (the regex backtracks to the last newline of the space run)
    {
        size_t e = i;
        bool have_nl = false;
        size_t last_nl = 0;
        while (cp_at(s, e, &c)) {
            if (!unicode_space(c.cp)) break;
            e += c.len;
            if (is_nl(c.cp)) {
                last_nl = e;
                have_nl = true;
            }
        }
        if (have_nl) return last_nl;
    }
    // 6: \s+(?!\S), then 7: \s+
    if (unicode_space(c0.cp)) {
        size_t e = i, prev = i;
        while (cp_at(s, e, &c)) {
            if (!unicode_space(c.cp)) break;
            prev = e;
            e += c.len;
        }
        if (e == s.len) return e;
        if (prev > i) return prev;
        return e;
    }
    return i + c0.len;
}

// ---- pre-tokenizer: o200k_split as a state machine ---------------------------------------
// The alternatives, tried in order at i (a backtracking engine takes the first that matches;
// some alternative always matches, so the pieces tile the text):
//   A1 [^\r\n\p{L}\p{N}]? U* W+ C?    A2 [^\r\n\p{L}\p{N}]? U+ W* C?    A3 \p{N}{1,3}
//   A4 ` ?[^\s\p{L}\p{N}]+[\r\n/]*`   A5 \s*[\r\n]+   A6 \s+(?!\S)   A7 \s+
// with U = [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}], W = [\p{Ll}\p{Lm}\p{Lo}\p{M}] and
// C = (?i:'s|'t|'re|'ve|'m|'ll|'d). Under (?i) `s` also matches U+017F (long s), as the
// tokenizers library does; no other letter of C has a case fold outside ASCII.

static bool o_upper(uint32_t cp) {
    Gc g = unicode_gc(cp);
    return g == GC_LU || g == GC_LT || g == GC_LM || g == GC_LO || g == GC_M;
}
static bool o_lower(uint32_t cp) {
    Gc g = unicode_gc(cp);
    return g == GC_LL || g == GC_LM || g == GC_LO || g == GC_M;
}
// [^\r\n\p{L}\p{N}]: the optional first code point of A1 and A2
static bool o_lead(uint32_t cp) { return !is_nl(cp) && !is_l(cp) && !is_n(cp); }
// [^\s\p{L}\p{N}] (marks included, unlike Qwen's class)
static bool o_punct(uint32_t cp) { return !unicode_space(cp) && !is_l(cp) && !is_n(cp); }

// End of C at i, or i when no contraction starts there.
static size_t o_contraction(Str s, size_t i) {
    if (i + 1 >= s.len || s.ptr[i] != '\'') return i;
    char a = to_lower(s.ptr[i + 1]);
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
    if ((uint8_t)s.ptr[i + 1] == 0xC5 && i + 2 < s.len && (uint8_t)s.ptr[i + 2] == 0xBF) return i + 3; // 'ſ
    if (i + 2 < s.len) {
        char b = to_lower(s.ptr[i + 2]);
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
    }
    return i;
}

// U* W+ C? at p: the end, or 0 when W+ cannot match. U* is greedy and gives back code points
// until a W follows: either the code point after the U run (then an Ll, the only W outside U),
// or the last code point of the run that is also a W (Lm, Lo, M), which W+ then takes alone
// (the U run continues with Lu or Lt after it, which W excludes).
static size_t o_a1(Str s, size_t p) {
    size_t e = p, last_w = SIZE_MAX, last_w_end = 0;
    Cp c;
    while (cp_at(s, e, &c) && o_upper(c.cp)) {
        if (o_lower(c.cp)) {
            last_w = e;
            last_w_end = e + c.len;
        }
        e += c.len;
    }
    size_t w_end;
    if (cp_at(s, e, &c) && o_lower(c.cp)) w_end = run_of(s, e, o_lower);
    else if (last_w != SIZE_MAX) w_end = last_w_end;
    else return 0;
    return o_contraction(s, w_end);
}

// U+ W* C? at p: the end, or 0 when no U starts there.
static size_t o_a2(Str s, size_t p) {
    Cp c;
    if (!cp_at(s, p, &c) || !o_upper(c.cp)) return 0;
    size_t e = run_of(s, p, o_upper);
    e = run_of(s, e, o_lower);
    return o_contraction(s, e);
}

size_t tokenizer_next_piece_o200k(Str s, size_t i) {
    Cp c0, c;
    cp_at(s, i, &c0);
    bool lead = o_lead(c0.cp);
    size_t e;
    // A1, A2: with the leading code point first, then without
    if (lead && (e = o_a1(s, i + c0.len)) > 0) return e;
    if ((e = o_a1(s, i)) > 0) return e;
    if (lead && (e = o_a2(s, i + c0.len)) > 0) return e;
    if ((e = o_a2(s, i)) > 0) return e;
    // A3: \p{N}{1,3}
    if (is_n(c0.cp)) {
        e = i + c0.len;
        for (int k = 1; k < 3 && cp_at(s, e, &c) && is_n(c.cp); k++) e += c.len;
        return e;
    }
    // A4: ` ?[^\s\p{L}\p{N}]+[\r\n/]*`
    {
        size_t p = s.ptr[i] == ' ' ? i + 1 : i;
        if (cp_at(s, p, &c) && o_punct(c.cp)) {
            e = run_of(s, p, o_punct);
            while (e < s.len && (s.ptr[e] == '\r' || s.ptr[e] == '\n' || s.ptr[e] == '/')) e += 1;
            return e;
        }
    }
    // A5: \s*[\r\n]+ (the regex backtracks to the last newline of the space run)
    {
        size_t last_nl = 0;
        bool have_nl = false;
        e = i;
        while (cp_at(s, e, &c) && unicode_space(c.cp)) {
            e += c.len;
            if (is_nl(c.cp)) {
                last_nl = e;
                have_nl = true;
            }
        }
        if (have_nl) return last_nl;
    }
    // A6: \s+(?!\S), then A7: \s+
    if (unicode_space(c0.cp)) {
        size_t prev = i;
        e = i;
        while (cp_at(s, e, &c) && unicode_space(c.cp)) {
            prev = e;
            e += c.len;
        }
        if (e == s.len) return e;
        if (prev > i) return prev;
        return e;
    }
    return i + c0.len;
}
