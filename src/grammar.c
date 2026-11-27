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

static void open_string(GState *s, bool key) {
    s->jmode = J_STR;
    s->jstr = key;
    s->u8need = 0;
    if (key) s->nlen = 0;
}

// a key's byte (decoded), for matching it against the schema's names as json_parse reads it
static void key_byte(GState *s, char c) {
    if (!s->jstr) return;
    if (s->nlen < GR_NAME) s->name[s->nlen] = c;
    if (s->nlen <= GR_NAME) s->nlen++;
}

// a key's code point from a \u escape, as UTF-8 (json.c put_utf8)
static void key_cp(GState *s, uint32_t u) {
    if (u < 0x80) key_byte(s, (char)u);
    else if (u < 0x800) key_byte(s, (char)(0xC0 | u >> 6)), key_byte(s, (char)(0x80 | (u & 0x3F)));
    else if (u < 0x10000) {
        key_byte(s, (char)(0xE0 | u >> 12));
        key_byte(s, (char)(0x80 | (u >> 6 & 0x3F)));
        key_byte(s, (char)(0x80 | (u & 0x3F)));
    } else {
        key_byte(s, (char)(0xF0 | u >> 18));
        key_byte(s, (char)(0x80 | (u >> 12 & 0x3F)));
        key_byte(s, (char)(0x80 | (u >> 6 & 0x3F)));
        key_byte(s, (char)(0x80 | (u & 0x3F)));
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
        key_byte(s, c);
        return;
    }
    if (c == '"') return close_string(g, s);
    if (c == '\\') {
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
    key_byte(s, c);
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
            open_string(s, false);
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
            key_byte(s, to[strchr(from, c) - from]);
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
        // checked digit by digit, so no prefix is a dead end: \uDC.. to \uDF.. alone is a lone low
        // surrogate, and a high surrogate's low half is \uDC.. to \uDF..
        if (s->jmode == J_STR_U && s->jk == 2 && s->uval >= 0xDC && s->uval <= 0xDF) return dead(s);
        if (s->jmode == J_STR_LO && ((s->jk == 1 && s->uval != 0xD) || (s->jk == 2 && s->uval < 0xDC))) return dead(s);
        if (s->jk < 4) return;
        if (s->jmode == J_STR_LO) {
            if (s->uval < 0xDC00 || s->uval > 0xDFFF) return dead(s);
            key_cp(s, 0x10000 + ((uint32_t)(s->uhi - 0xD800) << 10) + (uint32_t)(s->uval - 0xDC00));
            s->jmode = J_STR;
        } else if (s->uval >= 0xDC00 && s->uval <= 0xDFFF) {
            return dead(s); // a lone low surrogate
        } else if (s->uval >= 0xD800 && s->uval <= 0xDBFF) {
            s->uhi = s->uval;
            s->jmode = J_STR_LO_BS; // a high surrogate: its low half must follow
        } else {
            key_cp(s, s->uval);
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
        open_string(s, true);
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
            if (pi >= 0) {
                if (pi < 64) s->seen |= 1ull << pi;
                vn = g->props[g->nodes[tn].prop0 + (size_t)pi].node;
            }
            s->kc = s->kt = 0;
            if (g->nodes[vn].text) {
                s->mode = GM_VSTR;
                s->k = 0;
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
        if (s->k == lits[L_PCLOSE].len) {
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
    GState s = {.mode = thinking ? GM_REASON : GM_TEXT, .tool = -1, .param = -1};
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
