// Tool-call grammar for the Qwen chat format (see grammar.h).
#include "grammar.h"

#include <stdlib.h>
#include <string.h>

// JSON kinds a value may take (GNode.kinds); GK_INT alone is an integer, with GK_NUM any number
enum { GK_STR = 1, GK_NUM = 2, GK_INT = 4, GK_BOOL = 8, GK_NULL = 16, GK_ARR = 32, GK_OBJ = 64, GK_ANY = 127 };

typedef enum {
    GM_DEAD,
    GM_REASON,   // inside <think>: watch </think>; <tool_call> is not allowed
    GM_TEXT,     // the answer's text before any call: watch <tool_call>
    GM_LIT,      // literal lit from k, then mode next
    GM_WS_LIT,   // whitespace, then literal lit, then mode next
    GM_FNAME,    // a tool name up to '>'
    GM_PCHOICE,  // whitespace, then <parameter= or </function>
    GM_PNAME,    // a parameter name up to '>'
    GM_VSTR,     // a text parameter's value up to </parameter>
    GM_VJSON,    // a typed parameter's JSON value
    GM_AFTER,    // after </tool_call>: whitespace, then <tool_call> or the end
} GMode;

enum { L_TOOL_CALL, L_FUNCTION, L_PARAM, L_FN_CLOSE, L_PCLOSE, L_CALL_CLOSE, L_THINK_CLOSE };
static const Str lits[] = {S("<tool_call>"), S("<function="), S("<parameter="), S("</function>"),
                           S("</parameter>"), S("</tool_call>"), S("</think>")};

// JSON sub-modes (GState.jmode); a string (J_STR..J_STR_LO) is a value or, with jstr, a key
enum {
    J_VALUE,      // a value (whitespace first)
    J_STR,        // inside a string
    J_STR_ESC,    // after '\'
    J_STR_U,      // \u: jk hex digits read
    J_STR_LO_BS,  // after a high surrogate: '\' of its low half
    J_STR_LO_U,   // its 'u'
    J_STR_LO,     // its digits
    J_NUM,
    J_LIT,        // true, false, null (jflag: which, jk: bytes matched)
    J_AFTER,      // after an item or property value, or an array just opened (jflag 1)
    J_KEY_OR_END, // an object's key, or '}' just after '{' (jflag 1)
    J_COLON,
    J_DONE,       // the top value ended: whitespace, then </parameter>
};
// number states (jk while J_NUM)
enum { N_SIGN, N_ZERO, N_INT, N_DOT, N_FRAC, N_E, N_ESIGN, N_EXP };

// ---- building ---------------------------------------------------------------------------

typedef struct {
    Arena *a;
    VEC(GNode) nodes;
    VEC(GProp) props;
} Build;

static uint8_t kind_named(Str n) {
    return str_eq(n, S("string")) ? GK_STR : str_eq(n, S("number")) ? GK_NUM | GK_INT : str_eq(n, S("integer")) ? GK_INT
         : str_eq(n, S("boolean")) ? GK_BOOL : str_eq(n, S("null")) ? GK_NULL : str_eq(n, S("array")) ? GK_ARR
         : str_eq(n, S("object")) ? GK_OBJ : GK_ANY;
}

// The kinds of schema s: its "type" (a name or a union of names), else its shape (properties,
// items), else any.
static uint8_t kinds_of(const JValue *s) {
    const JValue *t = json_get(s, "type");
    if (t && t->kind == J_STRING) return kind_named(t->s);
    if (t && t->kind == J_ARRAY && t->arr.len > 0) {
        uint8_t k = 0;
        for (size_t i = 0; i < t->arr.len; i++) k |= t->arr.items[i]->kind == J_STRING ? kind_named(t->arr.items[i]->s) : GK_ANY;
        return k;
    }
    if (t) return GK_ANY;
    return json_get(s, "properties") ? GK_OBJ : json_get(s, "items") ? GK_ARR : GK_ANY;
}

// Whether the parser reads parameter schema s as text (src/toolcall.c param_type: type "string",
// a union with "string", or no type; any other type is JSON).
static bool read_as_text(const JValue *s) {
    const JValue *t = s ? json_get(s, "type") : NULL;
    if (!t) return true;
    if (t->kind == J_STRING) return str_eq(t->s, S("string"));
    if (t->kind == J_ARRAY) {
        for (size_t i = 0; i < t->arr.len; i++)
            if (t->arr.items[i]->kind == J_STRING && str_eq(t->arr.items[i]->s, S("string"))) return true;
        return false;
    }
    return true;
}

static bool name_pad(char c);

// Whether a call can name parameter n (GM_PNAME: the parser trims " \t\r\n" and refuses < > = " and
// newlines; names longer than GR_NAME are not matched).
static bool param_name_ok(Str n) {
    if (n.len == 0 || n.len > GR_NAME || name_pad(n.ptr[0]) || name_pad(n.ptr[n.len - 1])) return false;
    for (size_t i = 0; i < n.len; i++) {
        char c = n.ptr[i];
        if (c == '<' || c == '>' || c == '=' || c == '"' || c == '\n') return false;
    }
    return true;
}

static uint16_t node_of(Build *b, const JValue *s, int depth, bool top);

// The object node of schema s's properties and required list; top: a tool's parameters (named
// in <parameter=...> tags, typed as the parser types them), else a JSON object's keys.
static GNode object_node(Build *b, const JValue *s, int depth, bool top) {
    GNode n = {.kinds = GK_OBJ};
    const JValue *ps = json_get(s, "properties");
    size_t np = ps && ps->kind == J_OBJECT ? MIN(ps->obj.len, (size_t)UINT16_MAX) : 0;
    uint16_t *sub = np ? ARENA_ARRAY(b->a, uint16_t, np) : NULL;
    for (size_t i = 0; i < np; i++) sub[i] = node_of(b, ps->obj.items[i].val, depth + 1, top);
    n.prop0 = (uint16_t)b->props.len;
    n.nprops = (uint16_t)np;
    for (size_t i = 0; i < np; i++) vec_push(&b->props, ((GProp){ps->obj.items[i].key, sub[i]}));
    const JValue *rq = json_get(s, "required");
    // only names a reply can write are required (else the call or object could never close)
    if (rq && rq->kind == J_ARRAY)
        for (size_t r = 0; r < rq->arr.len; r++)
            for (size_t i = 0; i < np && i < 64; i++) {
                Str k = ps->obj.items[i].key;
                if (rq->arr.items[r]->kind == J_STRING && str_eq(rq->arr.items[r]->s, k) &&
                    (top ? param_name_ok(k) : k.len <= GR_NAME))
                    n.required |= 1ull << i;
            }
    return n;
}

// top: a parameter's own schema (text or a JSON value, as the parser types it)
static uint16_t node_of(Build *b, const JValue *s, int depth, bool top) {
    if (!s || s->kind != J_OBJECT || depth > GR_DEPTH) return 0;
    GNode n = {.kinds = kinds_of(s), .text = top && read_as_text(s)};
    if ((n.kinds & GK_OBJ) && n.kinds != GK_ANY) {
        GNode o = object_node(b, s, depth, false);
        n.prop0 = o.prop0, n.nprops = o.nprops, n.required = o.required;
    }
    if ((n.kinds & GK_ARR) && n.kinds != GK_ANY) n.items = node_of(b, json_get(s, "items"), depth + 1, false);
    vec_push(&b->nodes, n);
    return (uint16_t)(b->nodes.len - 1);
}

// Whether a call can name tool n (GM_FNAME: the parser ends a name at '>' or a newline and trims
// " \t\r").
static bool tool_name_ok(Str n) {
    if (n.len == 0 || n.len > GR_NAME || name_pad(n.ptr[0]) || name_pad(n.ptr[n.len - 1])) return false;
    for (size_t i = 0; i < n.len; i++)
        if (n.ptr[i] == '>' || n.ptr[i] == '\n') return false;
    return true;
}

Grammar *grammar_build(Arena *a, const JValue *tools, const Tokenizer *tok, const uint32_t *eos, size_t n_eos) {
    Grammar *g = ARENA_NEW(a, Grammar);
    g->em_tool = -1;
    g->tok = tok;
    g->eos = eos;
    g->n_eos = n_eos;
    Build b = {.a = a};
    vec_push(&b.nodes, ((GNode){.kinds = GK_ANY, .text = 1}));
    VEC(GTool) ts = {0};
    if (tools && tools->kind == J_ARRAY)
        for (size_t i = 0; i < tools->arr.len; i++) {
            const JValue *t = tools->arr.items[i];
            const JValue *f = json_get(t, "function");
            if (!f) f = t;
            const JValue *nm = json_get(f, "name");
            if (!nm || nm->kind != J_STRING || !tool_name_ok(nm->s)) continue;
            const JValue *ps = json_get(f, "parameters");
            // the parameters' object (no schema: any keys, nothing required)
            GNode on = ps && ps->kind == J_OBJECT ? object_node(&b, ps, 0, true) : (GNode){.kinds = GK_OBJ};
            vec_push(&b.nodes, on);
            vec_push(&ts, ((GTool){nm->s, (uint16_t)(b.nodes.len - 1)}));
        }
    // node and property indexes are 16 bits: a schema with more is not held to (no tools: the
    // server then decodes without the grammar)
    if (b.nodes.len > UINT16_MAX || b.props.len > UINT16_MAX) ts.len = 0;
    g->nodes = arena_dup(a, b.nodes.ptr, b.nodes.len * sizeof(GNode));
    g->n_nodes = b.nodes.len;
    if (b.props.len) g->props = arena_dup(a, b.props.ptr, b.props.len * sizeof(GProp));
    g->n_props = b.props.len;
    if (ts.len) g->tools = arena_dup(a, ts.ptr, ts.len * sizeof(GTool));
    g->n_tools = ts.len;
    vec_free(&b.nodes);
    vec_free(&b.props);
    vec_free(&ts);
    return g;
}

// ---- edit match: the files -----------------------------------------------------------------

// The suffix array of t (prefix doubling with counting sorts, O(n log n); a suffix that is a prefix
// of another sorts first). NULL when out of memory.
static uint32_t *suffix_array(Arena *a, const unsigned char *t, size_t n) {
    uint32_t *sa = ARENA_ARRAY(a, uint32_t, n);
    size_t m = MAX(n, (size_t)256) + 1;
    uint32_t *rk = malloc(n * sizeof *rk), *tmp = malloc(n * sizeof *tmp), *cnt = malloc(m * sizeof *cnt);
    if (!rk || !tmp || !cnt) {
        free(rk), free(tmp), free(cnt);
        return NULL;
    }
    memset(cnt, 0, m * sizeof *cnt);
    for (size_t i = 0; i < n; i++) cnt[t[i]]++;
    for (size_t c = 1; c < 256; c++) cnt[c] += cnt[c - 1];
    for (size_t i = n; i-- > 0;) sa[--cnt[t[i]]] = (uint32_t)i;
    rk[sa[0]] = 0;
    for (size_t i = 1; i < n; i++) rk[sa[i]] = rk[sa[i - 1]] + (t[sa[i]] != t[sa[i - 1]]);
    for (size_t k = 1; rk[sa[n - 1]] < n - 1 && k < n; k <<= 1) {
        // by the rank of the second half (past the end first), then stably by the first half's
        size_t p = 0;
        for (size_t i = n - k; i < n; i++) tmp[p++] = (uint32_t)i;
        for (size_t j = 0; j < n; j++)
            if (sa[j] >= k) tmp[p++] = sa[j] - (uint32_t)k;
        size_t nr = (size_t)rk[sa[n - 1]] + 1;
        memset(cnt, 0, nr * sizeof *cnt);
        for (size_t i = 0; i < n; i++) cnt[rk[i]]++;
        for (size_t c = 1; c < nr; c++) cnt[c] += cnt[c - 1];
        for (size_t j = n; j-- > 0;) sa[--cnt[rk[tmp[j]]]] = tmp[j];
        // new ranks (tmp): equal when both halves are
        tmp[sa[0]] = 0;
        for (size_t j = 1; j < n; j++) {
            uint32_t x = sa[j - 1], y = sa[j];
            int64_t xs = x + k < n ? (int64_t)rk[x + k] : -1, ys = y + k < n ? (int64_t)rk[y + k] : -1;
            tmp[y] = tmp[x] + (rk[x] != rk[y] || xs != ys);
        }
        memcpy(rk, tmp, n * sizeof *rk);
    }
    free(rk), free(tmp), free(cnt);
    return sa;
}

// UTF-8 as json_parse takes it (the rules of string_byte)
static bool utf8_ok(Str t) {
    for (size_t i = 0; i < t.len;) {
        unsigned char u = (unsigned char)t.ptr[i];
        size_t need = 0;
        unsigned char lo = 0x80, hi = 0xBF;
        if (u < 0x80) need = 0;
        else if (u >= 0xC2 && u <= 0xDF) need = 1;
        else if (u == 0xE0) need = 2, lo = 0xA0;
        else if ((u >= 0xE1 && u <= 0xEC) || u == 0xEE || u == 0xEF) need = 2;
        else if (u == 0xED) need = 2, hi = 0x9F;
        else if (u == 0xF0) need = 3, lo = 0x90;
        else if (u >= 0xF1 && u <= 0xF3) need = 3;
        else if (u == 0xF4) need = 3, hi = 0x8F;
        else return false;
        i++;
        for (size_t k = 0; k < need; k++, i++) {
            if (i >= t.len) return false;
            unsigned char v = (unsigned char)t.ptr[i];
            if (v < lo || v > hi) return false;
            lo = 0x80, hi = 0xBF;
        }
    }
    return true;
}

// CRLF and CR as LF (normalizeToLF), as the edit tool matches (the caller sends the text without the
// BOM the tool drops)
static Str edit_text(Arena *a, Str t) {
    if (!memchr(t.ptr, '\r', t.len)) return t;
    char *o = arena_alloc(a, t.len);
    size_t n = 0;
    for (size_t i = 0; i < t.len; i++) {
        if (t.ptr[i] == '\r') {
            o[n++] = '\n';
            if (i + 1 < t.len && t.ptr[i + 1] == '\n') i++;
        } else o[n++] = t.ptr[i];
    }
    return str_n(o, n);
}

static int prop_index(const Grammar *g, uint16_t node, Str name);
static bool fold_path(Str base, Str p, char *out, size_t cap, size_t *len);

enum { EM_FILE_MAX = 1 << 20, EM_TOTAL_MAX = 4 << 20, EM_FILES_MAX = 256 };

void grammar_edit_files(Grammar *g, Arena *a, const JValue *spec) {
    if (!spec || spec->kind != J_OBJECT) return;
    const JValue *cwd = json_get(spec, "cwd"), *home = json_get(spec, "home"), *fs = json_get(spec, "files");
    if (!cwd || cwd->kind != J_STRING || !cwd->s.len || cwd->s.ptr[0] != '/' || !fs || fs->kind != J_ARRAY) return;
    // pi's edit tool: path (text), edits (an array of objects with an oldText string)
    int16_t tool = -1;
    for (size_t i = 0; i < g->n_tools; i++)
        if (str_eq(g->tools[i].name, S("edit"))) tool = (int16_t)i;
    if (tool < 0) return;
    uint16_t tn = g->tools[tool].node;
    int pp = prop_index(g, tn, S("path")), pe = prop_index(g, tn, S("edits"));
    if (pp < 0 || pe < 0 || pp >= 64 || pe >= 64) return;
    const GNode *pn = &g->nodes[g->props[g->nodes[tn].prop0 + (size_t)pp].node];
    const GNode *en = &g->nodes[g->props[g->nodes[tn].prop0 + (size_t)pe].node];
    if (!pn->text || en->text || en->kinds != GK_ARR || en->items == 0) return;
    const GNode *in = &g->nodes[en->items];
    int po = in->kinds == GK_OBJ ? prop_index(g, en->items, S("oldText")) : -1;
    if (po < 0 || g->nodes[g->props[in->prop0 + (size_t)po].node].kinds != GK_STR) return;
    Str home_s = home && home->kind == J_STRING && home->s.len && home->s.ptr[0] == '/' ? home->s : S("/");
    GFile *files = ARENA_ARRAY(a, GFile, MIN(fs->arr.len, (size_t)EM_FILES_MAX));
    size_t nf = 0, total = 0;
    for (size_t i = 0; i < fs->arr.len && nf < EM_FILES_MAX; i++) {
        const JValue *f = fs->arr.items[i];
        const JValue *p = json_get(f, "path"), *t = json_get(f, "text");
        if (!p || p->kind != J_STRING || !t || t->kind != J_STRING) continue;
        Str text = edit_text(a, t->s);
        if (text.len == 0 || text.len > EM_FILE_MAX || total + text.len > EM_TOTAL_MAX || !utf8_ok(text)) continue;
        char out[4096];
        size_t n;
        if (!p->s.len || p->s.ptr[0] != '/' || !fold_path(S("/"), p->s, out, sizeof out, &n)) continue;
        uint32_t *sa = suffix_array(a, (const unsigned char *)text.ptr, text.len);
        if (!sa) continue;
        files[nf++] = (GFile){str(arena_strndup(a, out, n)), text, sa};
        total += text.len;
    }
    if (nf == 0) return;
    g->em_tool = tool;
    g->em_path = (uint16_t)pp;
    g->em_edits = (uint16_t)pe;
    g->em_item = en->items;
    g->em_old = (uint16_t)po;
    g->files = files;
    g->n_files = nf;
    g->cwd = cwd->s;
    g->home = home_s;
}

// ---- the automaton ----------------------------------------------------------------------

// whitespace between tags, as the parser skips it (std.ascii.isWhitespace, src/toolcall.c is_ws)
static bool tag_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x0b || c == 0x0c; }
// JSON whitespace (json_parse; typed_value trims " \t\r\n" around the value)
static bool json_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
// spaces the parser trims around a function or parameter name (trim_set " \t\r")
static bool name_pad(char c) { return c == ' ' || c == '\t' || c == '\r'; }

static void dead(GState *s) { s->mode = GM_DEAD; }

static void lit_then(GState *s, uint8_t mode, uint8_t lit, uint8_t next) {
    s->mode = mode;
    s->lit = lit;
    s->k = 0;
    s->next = next;
}

static int prop_index(const Grammar *g, uint16_t node, Str name) {
    const GNode *n = &g->nodes[node];
    for (size_t i = 0; i < n->nprops; i++)
        if (str_eq(g->props[n->prop0 + i].name, name)) return (int)i;
    return -1;
}

// progress of literal lit after k matched bytes and c: literals have '<' only at their start
static uint8_t lit_step(uint8_t lit, uint8_t k, char c) {
    if (c == lits[lit].ptr[k]) return (uint8_t)(k + 1);
    return c == '<' ? 1 : 0;
}

// a value of node n may start with c (JSON)
static bool json_type_ok(const GNode *n, char c) {
    uint8_t k = c == '"' ? GK_STR : c == '-' || (c >= '0' && c <= '9') ? GK_NUM | GK_INT : c == 't' || c == 'f' ? GK_BOOL
              : c == 'n' ? GK_NULL : c == '[' ? GK_ARR : c == '{' ? GK_OBJ : 0;
    return k == 0 || (n->kinds & k) != 0; // another byte: refused by the JSON rules
}

static void json_value_done(GState *s) {
    s->jmode = s->depth == 0 ? J_DONE : J_AFTER;
    s->jflag = 0;
}

static int hexval(char c) {
    return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
}

// the node of the next item or property value in the innermost container
static uint16_t child_node(const Grammar *g, const GState *s) {
    const GNode *n = &g->nodes[s->jnode[s->depth - 1]];
    if (s->jkind[s->depth - 1] == '[') return n->items;
    if (s->jkey < 0) return 0;
    return g->props[n->prop0 + (size_t)s->jkey].node;
}

// ---- edit match: suffixes of the file's text ---------------------------------------------

// the byte at depth d of suffix i of the text, -1 past its end (shorter suffixes sort first)
static int em_key(const GFile *f, uint32_t i, uint32_t d) {
    size_t p = (size_t)f->sa[i] + d;
    return p < f->text.len ? (unsigned char)f->text.ptr[p] : -1;
}

// the first index in [lo, hi) whose byte at depth d is at least c
static uint32_t em_lower(const GFile *f, uint32_t lo, uint32_t hi, uint32_t d, int c) {
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (em_key(f, mid, d) < c) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// whether a substring of the text continues the bytes so far with a byte in [lo, hi]
static bool em_can(const Grammar *g, const GState *s, int lo, int hi) {
    const GFile *f = &g->files[s->efile];
    uint32_t i = em_lower(f, s->elo, s->ehi, s->edep, lo);
    return i < s->ehi && em_key(f, i, s->edep) <= hi;
}

// a decoded byte of the oldText: the suffixes that continue with it (none: refused; a CR is refused,
// the tool reads it as LF)
static void em_byte(const Grammar *g, GState *s, char ch) {
    if (s->mode == GM_DEAD) return;
    int c = (unsigned char)ch;
    if (c == '\r') return dead(s);
    const GFile *f = &g->files[s->efile];
    uint32_t lo = em_lower(f, s->elo, s->ehi, s->edep, c), hi = em_lower(f, lo, s->ehi, s->edep, c + 1);
    if (lo == hi) return dead(s);
    s->elo = lo;
    s->ehi = hi;
    s->edep++;
}

// Whether the oldText so far (not empty) occurs once as the tool counts (count_occurrences: from the
// left, each match past the previous one's end). More than EM_ONCE_MAX overlapping occurrences: no.
enum { EM_ONCE_MAX = 64 };
static bool em_once(const Grammar *g, const GState *s) {
    size_t k = s->ehi - s->elo;
    if (s->edep == 0 || k == 0 || k > EM_ONCE_MAX) return false;
    if (k == 1) return true;
    uint32_t at[EM_ONCE_MAX];
    const GFile *f = &g->files[s->efile];
    for (size_t i = 0; i < k; i++) {
        uint32_t x = f->sa[s->elo + i];
        size_t j = i;
        for (; j > 0 && at[j - 1] > x; j--) at[j] = at[j - 1];
        at[j] = x;
    }
    size_t n = 1;
    for (size_t i = 1, end = (size_t)at[0] + s->edep; i < k; i++)
        if (at[i] >= end) n++, end = (size_t)at[i] + s->edep;
    return n == 1;
}

static void open_string(const Grammar *g, GState *s, bool key) {
    s->jmode = J_STR;
    s->jstr = key;
    s->u8need = 0;
    if (key) s->nlen = 0;
    // an edit's oldText (depth 2: the edits array, then one edit's object) in the call to a known file
    s->em = 0;
    if (!key && s->efile >= 0 && s->tool == g->em_tool && s->param == (int16_t)g->em_edits && s->depth == 2 &&
        s->jkind[0] == '[' && s->jkind[1] == '{' && s->jnode[1] == g->em_item && s->jkey == (int16_t)g->em_old) {
        s->em = 1;
        s->elo = 0;
        s->ehi = (uint32_t)g->files[s->efile].text.len;
        s->edep = 0;
    }
}

// a string's byte (decoded): a key's, for matching it against the schema's names as json_parse reads
// it; an oldText's, held to the file
static void key_byte(const Grammar *g, GState *s, char c) {
    if (s->em) return em_byte(g, s, c);
    if (!s->jstr) return;
    if (s->nlen < GR_NAME) s->name[s->nlen] = c;
    if (s->nlen <= GR_NAME) s->nlen++;
}

// a key's code point from a \u escape, as UTF-8 (json.c put_utf8)
static void key_cp(const Grammar *g, GState *s, uint32_t u) {
    if (u < 0x80) key_byte(g, s, (char)u);
    else if (u < 0x800) key_byte(g, s, (char)(0xC0 | u >> 6)), key_byte(g, s, (char)(0x80 | (u & 0x3F)));
    else if (u < 0x10000) {
        key_byte(g, s, (char)(0xE0 | u >> 12));
        key_byte(g, s, (char)(0x80 | (u >> 6 & 0x3F)));
        key_byte(g, s, (char)(0x80 | (u & 0x3F)));
    } else {
        key_byte(g, s, (char)(0xF0 | u >> 18));
        key_byte(g, s, (char)(0x80 | (u >> 12 & 0x3F)));
        key_byte(g, s, (char)(0x80 | (u >> 6 & 0x3F)));
        key_byte(g, s, (char)(0x80 | (u & 0x3F)));
    }
}

// the string's closing quote
static void close_string(const Grammar *g, GState *s) {
    if (!s->jstr) return json_value_done(s);
    s->jkey = s->nlen <= GR_NAME ? (int16_t)prop_index(g, s->jnode[s->depth - 1], str_n(s->name, s->nlen)) : (int16_t)-1;
    if (s->jkey >= 0 && s->jkey < 64) {
        uint64_t bit = 1ull << s->jkey;
        if (s->jseen[s->depth - 1] & bit) return dead(s); // json_parse refuses a repeated key
        s->jseen[s->depth - 1] |= bit;
    }
    s->jmode = J_COLON;
}

// a byte inside a string: UTF-8 checked as json_parse checks it (utf8_len, src/json.c)
static void string_byte(const Grammar *g, GState *s, char c) {
    unsigned char u = (unsigned char)c;
    if (s->u8need) {
        if (u < s->u8lo || u > s->u8hi) return dead(s);
        s->u8need--;
        s->u8lo = 0x80;
        s->u8hi = 0xBF;
        key_byte(g, s, c);
        return;
    }
    if (c == '"') {
        // an oldText closes when it is not empty and occurs once in the file
        if (s->em && !em_once(g, s)) return dead(s);
        s->em = 0;
        return close_string(g, s);
    }
    if (c == '\\') {
        // in an oldText an escape decodes to ASCII: some ASCII byte must continue it
        if (s->em && !em_can(g, s, 0, 0x7f)) return dead(s);
        s->jmode = J_STR_ESC;
        return;
    }
    if (u < 0x20) return dead(s);
    if (u >= 0x80) {
        s->u8lo = 0x80;
        s->u8hi = 0xBF;
        if (u >= 0xC2 && u <= 0xDF) s->u8need = 1;
        else if (u == 0xE0) s->u8need = 2, s->u8lo = 0xA0;
        else if ((u >= 0xE1 && u <= 0xEC) || u == 0xEE || u == 0xEF) s->u8need = 2;
        else if (u == 0xED) s->u8need = 2, s->u8hi = 0x9F;
        else if (u == 0xF0) s->u8need = 3, s->u8lo = 0x90;
        else if (u >= 0xF1 && u <= 0xF3) s->u8need = 3;
        else if (u == 0xF4) s->u8need = 3, s->u8hi = 0x8F;
        else return dead(s);
    }
    key_byte(g, s, c);
}

static void json_char(const Grammar *g, GState *s, char c) {
    switch (s->jmode) {
    case J_DONE:
        if (json_ws(c)) return;
        if (c == '<') {
            lit_then(s, GM_LIT, L_PCLOSE, GM_PCHOICE);
            s->k = 1;
            return;
        }
        return dead(s);
    case J_VALUE: {
        if (json_ws(c)) return;
        const GNode *n = &g->nodes[s->jexp];
        if (!json_type_ok(n, c)) return dead(s);
        if (c == '"') {
            open_string(g, s, false);
        } else if (c == '-' || (c >= '0' && c <= '9')) {
            s->jmode = J_NUM;
            s->jk = c == '-' ? N_SIGN : c == '0' ? N_ZERO : N_INT;
            s->jflag = !(n->kinds & GK_NUM); // an integer: no fraction, no exponent
        } else if (c == 't' || c == 'f' || c == 'n') {
            s->jmode = J_LIT;
            s->jflag = c == 't' ? 0 : c == 'f' ? 1 : 2;
            s->jk = 1;
        } else if (c == '[' || c == '{') {
            if (s->depth >= GR_DEPTH) return dead(s);
            s->jnode[s->depth] = s->jexp;
            s->jkind[s->depth] = (uint8_t)c;
            s->jseen[s->depth] = 0;
            s->depth++;
            // just opened (jflag 1): an array takes ']' or its first item (J_AFTER), an object
            // '}' or its first key
            s->jflag = 1;
            s->jmode = c == '[' ? J_AFTER : J_KEY_OR_END;
        } else {
            dead(s);
        }
        return;
    }
    case J_STR: return string_byte(g, s, c);
    case J_STR_ESC:
        if (c == 'u') {
            s->jmode = J_STR_U;
            s->jk = 0;
            s->uval = 0;
        } else if (c && strchr("\"\\/bfnrt", c)) {
            static const char from[] = "\"\\/bfnrt", to[] = "\"\\/\b\f\n\r\t";
            key_byte(g, s, to[strchr(from, c) - from]);
            s->jmode = J_STR;
        } else {
            dead(s);
        }
        return;
    case J_STR_U:
    case J_STR_LO: {
        int h = hexval(c);
        if (h < 0) return dead(s);
        s->uval = (uint16_t)(s->uval << 4 | (unsigned)h);
        ++s->jk;
        // in an oldText only \u0000 to \u007f, digit by digit along the bytes the text continues with
        if (s->em && s->jmode == J_STR_U && ((s->jk <= 2 && s->uval != 0) || (s->jk == 3 && (s->uval > 7 || !em_can(g, s, s->uval << 4, (s->uval << 4) | 15)))))
            return dead(s);
        // checked digit by digit, so no prefix is a dead end: \uDC.. to \uDF.. alone is a lone low
        // surrogate, and a high surrogate's low half is \uDC.. to \uDF..
        if (s->jmode == J_STR_U && s->jk == 2 && s->uval >= 0xDC && s->uval <= 0xDF) return dead(s);
        if (s->jmode == J_STR_LO && ((s->jk == 1 && s->uval != 0xD) || (s->jk == 2 && s->uval < 0xDC))) return dead(s);
        if (s->jk < 4) return;
        if (s->jmode == J_STR_LO) {
            if (s->uval < 0xDC00 || s->uval > 0xDFFF) return dead(s);
            key_cp(g, s, 0x10000 + ((uint32_t)(s->uhi - 0xD800) << 10) + (uint32_t)(s->uval - 0xDC00));
            s->jmode = J_STR;
        } else if (s->uval >= 0xDC00 && s->uval <= 0xDFFF) {
            return dead(s); // a lone low surrogate
        } else if (s->uval >= 0xD800 && s->uval <= 0xDBFF) {
            s->uhi = s->uval;
            s->jmode = J_STR_LO_BS; // a high surrogate: its low half must follow
        } else {
            key_cp(g, s, s->uval);
            s->jmode = J_STR;
        }
        return;
    }
    case J_STR_LO_BS:
        if (c != '\\') return dead(s);
        s->jmode = J_STR_LO_U;
        return;
    case J_STR_LO_U:
        if (c != 'u') return dead(s);
        s->jmode = J_STR_LO;
        s->jk = 0;
        s->uval = 0;
        return;
    case J_NUM: {
        bool digit = c >= '0' && c <= '9';
        switch (s->jk) {
        case N_SIGN:
            if (!digit) return dead(s);
            s->jk = c == '0' ? N_ZERO : N_INT;
            return;
        case N_ZERO:
        case N_INT:
            if (digit && s->jk == N_INT) return;
            if (digit) return dead(s); // a leading zero
            if (c == '.' && !s->jflag) {
                s->jk = N_DOT;
                return;
            }
            if ((c == 'e' || c == 'E') && !s->jflag) {
                s->jk = N_E;
                return;
            }
            break;
        case N_DOT:
            if (!digit) return dead(s);
            s->jk = N_FRAC;
            return;
        case N_FRAC:
            if (digit) return;
            if (c == 'e' || c == 'E') {
                s->jk = N_E;
                return;
            }
            break;
        case N_E:
            if (c == '+' || c == '-') {
                s->jk = N_ESIGN;
                return;
            }
            if (!digit) return dead(s);
            s->jk = N_EXP;
            return;
        case N_ESIGN:
            if (!digit) return dead(s);
            s->jk = N_EXP;
            return;
        case N_EXP:
            if (digit) return;
            break;
        }
        // the number ended at a complete state: c belongs to what follows
        json_value_done(s);
        json_char(g, s, c);
        return;
    }
    case J_LIT: {
        static const char *const words[] = {"true", "false", "null"};
        const char *w = words[s->jflag];
        if (c != w[s->jk]) return dead(s);
        if (w[++s->jk] == 0) json_value_done(s);
        return;
    }
    case J_AFTER: {
        if (json_ws(c)) return;
        uint8_t kind = s->jkind[s->depth - 1];
        bool just = s->jflag == 1;
        if ((kind == '[' && c == ']') || (kind == '{' && c == '}')) {
            if (kind == '{') {
                const GNode *n = &g->nodes[s->jnode[s->depth - 1]];
                if ((s->jseen[s->depth - 1] & n->required) != n->required) return dead(s);
            }
            s->depth--;
            json_value_done(s);
            return;
        }
        if (kind == '[' && just) {
            // the first item of an array just opened
            s->jflag = 0;
            s->jexp = child_node(g, s);
            s->jmode = J_VALUE;
            json_char(g, s, c);
            return;
        }
        if (c != ',' || just) return dead(s);
        s->jflag = 0;
        if (kind == '[') {
            s->jexp = child_node(g, s);
            s->jmode = J_VALUE;
        } else {
            s->jmode = J_KEY_OR_END;
            s->jflag = 2; // after a comma: a key must follow
        }
        return;
    }
    case J_KEY_OR_END:
        if (json_ws(c)) return;
        if (c == '}' && s->jflag == 1) {
            s->jmode = J_AFTER;
            json_char(g, s, c);
            return;
        }
        if (c != '"') return dead(s);
        s->jflag = 0;
        open_string(g, s, true);
        return;
    case J_COLON:
        if (json_ws(c)) return;
        if (c != ':') return dead(s);
        s->jexp = child_node(g, s);
        s->jmode = J_VALUE;
        return;
    }
}

// Padding around names: the parser trims it. Before a name it is skipped; after a name it is
// stored while there is room (trail counts it, trimmed at the '>') and counted past that (vpad,
// then only padding or '>' may follow a function name, and a parameter name continued past
// GR_NAME is longer than any the schema has).
static void name_store(GState *s, char c) {
    if (s->nlen < GR_NAME) s->name[s->nlen] = c;
    if (s->nlen <= GR_NAME) s->nlen++;
}

// the name without its trailing padding
static size_t name_len(const GState *s) { return s->nlen > GR_NAME ? s->nlen : s->nlen - MIN(s->nlen, s->trail); }

static bool tool_prefix(const Grammar *g, Str p) {
    for (size_t i = 0; i < g->n_tools; i++)
        if (str_starts(g->tools[i].name, p)) return true;
    return false;
}

static bool tool_named(const Grammar *g, Str n, size_t *out) {
    for (size_t i = 0; i < g->n_tools; i++)
        if (str_eq(g->tools[i].name, n)) {
            *out = i;
            return true;
        }
    return false;
}

// a function name's byte (not '>'): only along a tool's name, then padding; false: refused
static bool fname_char(const Grammar *g, GState *s, char c) {
    if (c == '\n') return false;
    bool pad = name_pad(c);
    if (pad && s->nlen == 0) return true;
    if (s->vpad) {
        if (!pad) return false;
        if (s->vpad < 255) s->vpad++;
        return true;
    }
    if (s->nlen < GR_NAME) {
        s->name[s->nlen] = c;
        if (tool_prefix(g, str_n(s->name, s->nlen + 1))) {
            s->nlen++;
            s->trail = pad ? (uint8_t)MIN(s->trail + 1, 255) : 0;
            return true;
        }
    }
    size_t t;
    if (pad && tool_named(g, str_n(s->name, name_len(s)), &t)) {
        s->vpad = 1;
        return true;
    }
    return false;
}

// a parameter name's byte (not '>'); false: refused (the parser drops a name with < = " or a
// newline: plausible_name)
static bool pname_char(GState *s, char c) {
    if (c == '<' || c == '=' || c == '\n' || c == '"') return false;
    bool pad = name_pad(c);
    if (pad && s->nlen == 0) return true;
    if (pad && s->nlen >= GR_NAME) {
        if (s->vpad < 255) s->vpad++;
        return true;
    }
    if (!pad && s->vpad) {
        s->nlen = GR_NAME + 1; // padding inside a name past GR_NAME: longer than any of the schema's
        s->vpad = 0;
        s->trail = 0;
        return true;
    }
    name_store(s, c);
    s->trail = pad ? (uint8_t)MIN(s->trail + 1, 255) : 0;
    return true;
}

static bool required_given(const Grammar *g, const GState *s) {
    const GNode *n = &g->nodes[g->tools[s->tool].node];
    return (s->seen & n->required) == n->required;
}

// path.resolve as the edit tool resolves (src/code_tools.c path_resolve): segments "." and empty
// dropped, ".." folded; false when it does not fit
static bool fold_path(Str base, Str p, char *out, size_t cap, size_t *len) {
    char in[4096];
    size_t n = 0;
    if (!(p.len && p.ptr[0] == '/')) {
        if (base.len + 1 > sizeof in) return false;
        memcpy(in, base.ptr, base.len);
        n = base.len;
        in[n++] = '/';
    }
    if (n + p.len > sizeof in) return false;
    memcpy(in + n, p.ptr, p.len);
    n += p.len;
    size_t o = 0, i = 0;
    while (i <= n) {
        size_t j = i;
        while (j < n && in[j] != '/') j++;
        size_t sl = j - i;
        if (sl == 0 || (sl == 1 && in[i] == '.')) {
        } else if (sl == 2 && in[i] == '.' && in[i + 1] == '.') {
            while (o > 0 && out[o - 1] != '/') o--;
            if (o > 0) o--;
        } else {
            if (o + 1 + sl > cap) return false;
            out[o++] = '/';
            memcpy(out + o, in + i, sl);
            o += sl;
        }
        i = j + 1;
    }
    if (o == 0) out[o++] = '/';
    *len = o;
    return true;
}

// The listed file the edit's path names (the value in name[], its closer included), or -1.
static int16_t edit_file(const Grammar *g, const GState *s) {
    size_t cl = lits[L_PCLOSE].len;
    if (s->nlen > GR_NAME || s->nlen < cl) return -1;
    Str v = str_n(s->name, s->nlen - cl);
    // the parser's framing (src/toolcall.c unframe), then the tool's (resolve_to_cwd)
    if (str_starts(v, S("\r\n"))) v = str_n(v.ptr + 2, v.len - 2);
    else if (str_starts(v, S("\n"))) v = str_n(v.ptr + 1, v.len - 1);
    if (str_ends(v, S("\r\n"))) v.len -= 2;
    else if (str_ends(v, S("\n"))) v.len -= 1;
    // the tool maps Unicode spaces to ' ' and reads file:// URLs: such a path is left unconstrained
    for (size_t i = 0; i < v.len; i++)
        if ((unsigned char)v.ptr[i] >= 0x80) return -1;
    if (v.len && v.ptr[0] == '@') v = str_n(v.ptr + 1, v.len - 1);
    if (str_starts(v, S("file:"))) return -1;
    char out[4096];
    size_t n;
    bool ok;
    if (str_eq(v, S("~"))) ok = fold_path(S("/"), g->home, out, sizeof out, &n);
    else if (str_starts(v, S("~/"))) ok = fold_path(g->home, str_n(v.ptr + 2, v.len - 2), out, sizeof out, &n);
    else ok = fold_path(g->cwd, v, out, sizeof out, &n);
    if (!ok) return -1;
    for (size_t i = 0; i < g->n_files; i++)
        if (str_eq(g->files[i].path, str_n(out, n))) return (int16_t)i;
    return -1;
}

static void feed_char(const Grammar *g, GState *s, char c) {
    switch ((GMode)s->mode) {
    case GM_DEAD: return;
    case GM_REASON:
        s->kt = lit_step(L_TOOL_CALL, s->kt, c);
        if (s->kt == lits[L_TOOL_CALL].len) return dead(s);
        s->kc = lit_step(L_THINK_CLOSE, s->kc, c);
        if (s->kc == lits[L_THINK_CLOSE].len) {
            s->mode = GM_TEXT;
            s->kt = 0;
        }
        return;
    case GM_TEXT:
        s->kt = lit_step(L_TOOL_CALL, s->kt, c);
        if (s->kt == lits[L_TOOL_CALL].len) lit_then(s, GM_WS_LIT, L_FUNCTION, GM_FNAME), s->nlen = 0, s->trail = 0, s->vpad = 0;
        return;
    case GM_WS_LIT:
        if (s->k == 0 && tag_ws(c)) return;
        s->mode = GM_LIT;
        [[fallthrough]];
    case GM_LIT: {
        Str l = lits[s->lit];
        if (c != l.ptr[s->k]) return dead(s);
        if (++s->k < l.len) return;
        s->mode = s->next;
        s->k = 0;
        s->alive = 0;
        s->nlen = 0;
        s->trail = 0;
        s->vpad = 0;
        return;
    }
    case GM_FNAME: {
        if (c == '>') {
            size_t t;
            if (!tool_named(g, str_n(s->name, name_len(s)), &t)) return dead(s);
            s->tool = (int16_t)t;
            s->seen = 0;
            s->efile = -1;
            s->mode = GM_PCHOICE;
            s->k = 0;
            s->alive = 0;
            return;
        }
        if (!fname_char(g, s, c)) return dead(s);
        return;
    }
    case GM_PCHOICE:
        if (s->k == 0) {
            if (tag_ws(c)) return;
            if (c != '<') return dead(s);
            s->k = 1;
            // </function> only once the required parameters are given: a prefix of it would leave
            // nothing valid to write
            s->alive = (uint8_t)(1 | (required_given(g, s) ? 2 : 0)); // <parameter=, </function>
            return;
        }
        {
            uint8_t al = 0;
            if ((s->alive & 1) && lits[L_PARAM].ptr[s->k] == c) al |= 1;
            if ((s->alive & 2) && lits[L_FN_CLOSE].ptr[s->k] == c) al |= 2;
            if (!al) return dead(s);
            s->alive = al;
            s->k++;
            if ((al & 1) && s->k == lits[L_PARAM].len) {
                s->mode = GM_PNAME;
                s->nlen = 0;
                s->trail = 0;
                s->vpad = 0;
                s->k = 0;
            } else if ((al & 2) && s->k == lits[L_FN_CLOSE].len) {
                lit_then(s, GM_WS_LIT, L_CALL_CLOSE, GM_AFTER);
            }
        }
        return;
    case GM_PNAME: {
        if (c == '>') {
            size_t nl = name_len(s);
            if (nl == 0) return dead(s);
            uint16_t tn = g->tools[s->tool].node;
            int pi = nl <= GR_NAME ? prop_index(g, tn, str_n(s->name, nl)) : -1;
            s->param = (int16_t)pi;
            uint16_t vn = 0;
            // the parser keeps a parameter's first value and skips repeats (src/toolcall.c)
            bool first = pi >= 0 && pi < 64 && !(s->seen & 1ull << pi);
            if (pi >= 0) {
                if (pi < 64) s->seen |= 1ull << pi;
                vn = g->props[g->nodes[tn].prop0 + (size_t)pi].node;
            }
            s->kc = s->kt = 0;
            if (g->nodes[vn].text) {
                s->mode = GM_VSTR;
                s->k = 0;
                // the edit's path: its bytes go to name[] (free while a text value is read)
                s->ecap = g->n_files > 0 && s->tool == g->em_tool && pi == (int)g->em_path && first;
                if (s->ecap) s->efile = -1, s->nlen = 0;
            } else {
                s->mode = GM_VJSON;
                s->jmode = J_VALUE;
                s->depth = 0;
                s->jexp = vn;
                s->jflag = 0;
            }
            return;
        }
        if (!pname_char(s, c)) return dead(s);
        return;
    }
    case GM_VSTR:
        // <parameter= inside a value would end it for the parser (src/toolcall.c value_end)
        s->kt = lit_step(L_PARAM, s->kt, c);
        if (s->kt == lits[L_PARAM].len) return dead(s);
        s->k = lit_step(L_PCLOSE, s->k, c);
        if (s->ecap) name_store(s, c);
        if (s->k == lits[L_PCLOSE].len) {
            if (s->ecap) {
                s->ecap = 0;
                s->efile = edit_file(g, s);
            }
            s->mode = GM_PCHOICE;
            s->k = 0;
            s->alive = 0;
        }
        return;
    case GM_VJSON:
        // the parser splits values at these, inside a JSON string too
        s->kt = lit_step(L_PARAM, s->kt, c);
        s->kc = lit_step(L_PCLOSE, s->kc, c);
        if (s->jmode != J_DONE && (s->kt == lits[L_PARAM].len || s->kc == lits[L_PCLOSE].len)) return dead(s);
        json_char(g, s, c);
        return;
    case GM_AFTER:
        if (s->k == 0 && tag_ws(c)) return;
        if (c != lits[L_TOOL_CALL].ptr[s->k]) return dead(s);
        if (++s->k == lits[L_TOOL_CALL].len) lit_then(s, GM_WS_LIT, L_FUNCTION, GM_FNAME), s->nlen = 0, s->trail = 0, s->vpad = 0;
        return;
    }
}

GState grammar_start(const Grammar *g, bool thinking, Str forced) {
    GState s = {.mode = thinking ? GM_REASON : GM_TEXT, .tool = -1, .param = -1, .efile = -1};
    grammar_feed_bytes(g, &s, forced);
    return s;
}

void grammar_feed_bytes(const Grammar *g, GState *s, Str b) {
    for (size_t i = 0; i < b.len && s->mode != GM_DEAD; i++) feed_char(g, s, b.ptr[i]);
}

bool grammar_ok_bytes(const Grammar *g, const GState *s, Str b) {
    if (s->mode == GM_DEAD) return true;
    if (b.len == 0) return false; // would not move the reply: a loop of empty tokens
    GState t = *s;
    grammar_feed_bytes(g, &t, b);
    return t.mode != GM_DEAD;
}

bool grammar_may_end(const GState *s) {
    switch ((GMode)s->mode) {
    case GM_DEAD:
    case GM_REASON:
    case GM_TEXT: return true;
    case GM_AFTER: return s->k == 0;
    default: return false;
    }
}

bool grammar_in_edit(const GState *s) { return s->mode == GM_VJSON && s->em; }

bool grammar_in_call(const GState *s) { return s->mode != GM_DEAD && s->mode != GM_REASON && s->mode != GM_TEXT; }

static bool is_eos(const Grammar *g, uint32_t tok) {
    for (size_t i = 0; i < g->n_eos; i++)
        if (g->eos[i] == tok) return true;
    return false;
}

bool grammar_ok(const Grammar *g, const GState *s, uint32_t tok) {
    if (s->mode == GM_DEAD) return true;
    if (is_eos(g, tok)) return grammar_may_end(s);
    char b[512];
    size_t n = tokenizer_token_bytes(g->tok, tok, b, sizeof b);
    if (n > sizeof b) return false;
    return grammar_ok_bytes(g, s, str_n(b, n));
}

void grammar_feed(const Grammar *g, GState *s, uint32_t tok) {
    if (s->mode == GM_DEAD) return;
    // the reply ended: what a verification round reads past its end is never emitted, so it is not checked
    if (is_eos(g, tok)) return dead(s);
    char b[512];
    size_t n = tokenizer_token_bytes(g->tok, tok, b, sizeof b);
    if (n > sizeof b) return dead(s);
    grammar_feed_bytes(g, s, str_n(b, n));
}
