// Chat prompt rendering for the Qwen3.8-Flash-Next template (see chat.h).
#include "chat.h"

#include <inttypes.h>
#include <stdlib.h>

#include "unicode.h"

static const char tools_head[] = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
static const char tools_tail[] = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";
static const char effort_xhigh[] = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
static const char effort_low[] = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";

// Member of an object, or NULL when absent, null, or v is not an object.
static const JValue *get(const JValue *v, const char *key) {
    if (!v || v->kind != J_OBJECT) return NULL;
    const JValue *r = json_get(v, key);
    return r && r->kind != J_NULL ? r : NULL;
}

static bool contains(const JValue *obj, const char *key) { return json_get(obj, key) != NULL; }

static const JValue *kwarg(const JValue *req, const char *key) {
    const JValue *k = get(req, "chat_template_kwargs");
    if (k) {
        const JValue *v = get(k, key);
        if (v) return v;
    }
    return get(req, key);
}

static bool is_bool(const JValue *v, bool b) { return v && v->kind == J_BOOL && v->b == b; }

// Python str.strip() whitespace.
static bool is_py_space(uint32_t cp) { return unicode_space(cp) || (cp >= 0x1c && cp <= 0x1f); }

static int seq_len(uint8_t b) {
    if (b <= 0x7F) return 1;
    if (b >= 0xC0 && b <= 0xDF) return 2;
    if (b >= 0xE0 && b <= 0xEF) return 3;
    if (b >= 0xF0 && b <= 0xF7) return 4;
    return 0;
}

// std.unicode.utf8Decode of the n bytes at s (the length picks the form, as in Zig).
static bool decode(const uint8_t *s, size_t n, uint32_t *cp) {
    if (n == 1) {
        *cp = s[0];
        return true;
    }
    if (n < 2 || n > 4) return false;
    static const uint8_t lead_mask[5] = {0, 0, 0x1F, 0x0F, 0x07};
    static const uint32_t min_cp[5] = {0, 0, 0x80, 0x800, 0x10000};
    uint32_t v = s[0] & lead_mask[n];
    for (size_t k = 1; k < n; k++) {
        if ((s[k] & 0xC0) != 0x80) return false;
        v = v << 6 | (s[k] & 0x3F);
    }
    if (v < min_cp[n]) return false;
    if (n == 3 && v >= 0xD800 && v <= 0xDFFF) return false;
    if (n == 4 && v > 0x10FFFF) return false;
    *cp = v;
    return true;
}

static Str trim(Str s) {
    const uint8_t *p = (const uint8_t *)s.ptr;
    size_t a = 0, b = s.len;
    uint32_t cp;
    while (a < b) {
        int n = seq_len(p[a]);
        if (n == 0) n = 1;
        if (!decode(p + a, MIN(a + (size_t)n, b) - a, &cp)) break;
        if (!is_py_space(cp)) break;
        a += (size_t)n;
    }
    while (b > a) {
        size_t st = b - 1;
        while (st > a && (p[st] & 0xC0) == 0x80) st -= 1;
        if (!decode(p + st, b - st, &cp)) break;
        if (!is_py_space(cp)) break;
        b = st;
    }
    return str_sub(s, a, b);
}

// render_content(content): string, list of text parts, or none. Images and video are out of scope.
static Err content(Arena *a, const JValue *c, Str *out) {
    if (!c) {
        *out = S("");
        return OK;
    }
    if (c->kind == J_STRING) {
        *out = c->s;
        return OK;
    }
    if (c->kind != J_ARRAY) return "BadContent";
    Buf o = {0};
    Err err = OK;
    for (size_t i = 0; i < c->arr.len; i++) {
        const JValue *item = c->arr.items[i];
        if (item->kind != J_OBJECT) { err = "BadContent"; break; }
        const JValue *t = get(item, "type");
        Str ty = t && t->kind == J_STRING ? t->s : S("");
        if (contains(item, "image") || contains(item, "image_url") || str_eqz(ty, "image") ||
            contains(item, "video") || str_eqz(ty, "video")) { err = "VisionUnsupported"; break; }
        const JValue *tx = json_get(item, "text");
        if (!tx || tx->kind != J_STRING) { err = "BadContent"; break; }
        buf_addstr(&o, tx->s);
    }
    if (!err) *out = str_n(arena_dup(a, o.ptr ? o.ptr : "", o.len), o.len);
    buf_free(&o);
    return err;
}

static Str role(const JValue *m) {
    const JValue *r = get(m, "role");
    return r && r->kind == J_STRING ? r->s : S("");
}

static Err trimmed_content(Arena *a, const JValue *m, Str *out) {
    TRY(content(a, get(m, "content"), out));
    *out = trim(*out);
    return OK;
}

static Err render(Arena *a, const JValue *req, Buf *o) {
    const JValue *msgs = get(req, "messages");
    if (!msgs || msgs->kind != J_ARRAY || msgs->arr.len == 0) return "NoMessages";
    JValue *const *M = msgs->arr.items;
    size_t nm = msgs->arr.len;

    const JValue *thinking = kwarg(req, "enable_thinking");
    bool thinking_on = thinking == NULL || is_bool(thinking, true);
    const char *reasoning = "";
    if (thinking_on) {
        const JValue *e = kwarg(req, "reasoning_effort");
        Str eff = S("xhigh");
        if (e) {
            if (e->kind != J_STRING) return "BadReasoningEffort";
            eff = e->s;
        }
        if (str_eqz(eff, "xhigh")) reasoning = effort_xhigh;
        else if (str_eqz(eff, "low")) reasoning = effort_low;
        else if (!str_eqz(eff, "medium")) return "BadReasoningEffort";
    }

    const JValue *tools = get(req, "tools");
    bool first_system = str_eqz(role(M[0]), "system");
    if (tools && tools->kind == J_ARRAY && tools->arr.len > 0) {
        buf_adds(o, "<|im_start|>system\n");
        if (reasoning[0]) buf_printf(o, "%s\n\n", reasoning);
        buf_adds(o, tools_head);
        for (size_t i = 0; i < tools->arr.len; i++) {
            buf_addc(o, '\n');
            chat_to_json(o, tools->arr.items[i]);
        }
        buf_adds(o, "\n</tools>");
        buf_adds(o, tools_tail);
        if (first_system) {
            Str sc;
            TRY(trimmed_content(a, M[0], &sc));
            if (sc.len > 0) {
                buf_adds(o, "\n\n");
                buf_addstr(o, sc);
            }
        }
        buf_adds(o, "<|im_end|>\n");
    } else {
        Str sc = S("");
        if (first_system) TRY(trimmed_content(a, M[0], &sc));
        if (sc.len > 0) {
            buf_adds(o, "<|im_start|>system\n");
            if (reasoning[0]) buf_printf(o, "%s\n\n", reasoning);
            buf_addstr(o, sc);
            buf_adds(o, "<|im_end|>\n");
        } else if (reasoning[0]) {
            buf_printf(o, "<|im_start|>system\n%s<|im_end|>\n", reasoning);
        }
    }

    // last real user query (not a wrapped tool response)
    bool found = false;
    size_t lq = 0;
    for (size_t k = nm; k > 0;) {
        k -= 1;
        if (!str_eqz(role(M[k]), "user")) continue;
        Str c;
        TRY(trimmed_content(a, M[k], &c));
        if (!(str_starts(c, S("<tool_response>")) && str_ends(c, S("</tool_response>")))) {
            lq = k;
            found = true;
            break;
        }
    }
    if (!found) return "NoUserQuery";
    const JValue *pt = kwarg(req, "preserve_thinking");
    bool preserve = pt == NULL || is_bool(pt, true);

    for (size_t i = 0; i < nm; i++) {
        const JValue *m = M[i];
        Str r = role(m);
        Str c;
        TRY(trimmed_content(a, m, &c));
        if (str_eqz(r, "system")) {
            if (i != 0) return "SystemNotFirst";
        } else if (str_eqz(r, "user")) {
            buf_adds(o, "<|im_start|>user\n");
            buf_addstr(o, c);
            buf_adds(o, "<|im_end|>\n");
        } else if (str_eqz(r, "assistant")) {
            const JValue *x = get(m, "reasoning_content");
            Str rc = x && x->kind == J_STRING ? trim(x->s) : S("");
            buf_adds(o, "<|im_start|>assistant\n");
            if (preserve || i > lq) {
                buf_adds(o, "<think>\n");
                buf_addstr(o, rc);
                buf_adds(o, "\n</think>\n\n");
                buf_addstr(o, c);
            } else {
                buf_addstr(o, c);
            }
            const JValue *tcs = get(m, "tool_calls");
            if (tcs && tcs->kind == J_ARRAY) {
                for (size_t j = 0; j < tcs->arr.len; j++) {
                    const JValue *tc0 = tcs->arr.items[j];
                    const JValue *tc = get(tc0, "function");
                    if (!tc) tc = tc0;
                    const JValue *n = get(tc, "name");
                    if (!n || n->kind != J_STRING) return "BadToolCall";
                    if (j == 0) {
                        if (trim(c).len > 0) buf_adds(o, "\n\n");
                    } else {
                        buf_addc(o, '\n');
                    }
                    buf_adds(o, "<tool_call>\n<function=");
                    buf_addstr(o, n->s);
                    buf_adds(o, ">\n");
                    const JValue *args0 = get(tc, "arguments");
                    if (args0) {
                        // OpenAI sends arguments as a JSON string; the template iterates a mapping.
                        const JValue *av = args0;
                        if (args0->kind == J_STRING) {
                            if (args0->s.len == 0) {
                                av = NULL;
                            } else {
                                JValue *p;
                                if (json_parse(a, args0->s, false, &p)) return "BadToolCall";
                                av = p;
                            }
                        }
                        if (av) {
                            if (av->kind != J_OBJECT) return "BadToolCall";
                            for (size_t k = 0; k < av->obj.len; k++) {
                                const JMember *e = &av->obj.items[k];
                                buf_adds(o, "<parameter=");
                                buf_addstr(o, e->key);
                                buf_adds(o, ">\n");
                                if (e->val->kind == J_STRING) buf_addstr(o, e->val->s);
                                else chat_to_json(o, e->val);
                                buf_adds(o, "\n</parameter>\n");
                            }
                        }
                    }
                    buf_adds(o, "</function>\n</tool_call>");
                }
            }
            buf_adds(o, "<|im_end|>\n");
        } else if (str_eqz(r, "tool")) {
            if (i > 0 && !str_eqz(role(M[i - 1]), "tool")) buf_adds(o, "<|im_start|>user");
            buf_adds(o, "\n<tool_response>\n");
            buf_addstr(o, c);
            buf_adds(o, "\n</tool_response>");
            if (i + 1 == nm || !str_eqz(role(M[i + 1]), "tool")) buf_adds(o, "<|im_end|>\n");
        } else {
            return "BadRole";
        }
    }
    buf_adds(o, "<|im_start|>assistant\n");
    if (is_bool(thinking, false)) buf_adds(o, "<think>\n\n</think>\n\n");
    else buf_adds(o, "<think>\n");
    return OK;
}

Err chat_render(Arena *a, const JValue *req, Str *out) {
    Buf o = {0};
    Err err = render(a, req, &o);
    if (!err) *out = str_n(arena_dup(a, o.ptr ? o.ptr : "", o.len + 1), o.len);
    buf_free(&o);
    return err;
}

void chat_to_json(Buf *o, const JValue *v) {
    switch (v->kind) {
    case J_NULL: buf_adds(o, "null"); break;
    case J_BOOL: buf_adds(o, v->b ? "true" : "false"); break;
    case J_INTEGER: buf_printf(o, "%" PRId64, v->i); break;
    case J_FLOAT: {
        char tmp[512]; // "{d}" writes no exponent: up to ~330 digits for f64
        int n = fmt_f64(tmp, sizeof tmp, v->f);
        buf_add(o, tmp, MIN((size_t)n, sizeof tmp - 1));
        break;
    }
    case J_NUMBER_STRING: buf_addstr(o, v->s); break;
    case J_STRING: chat_json_string(o, v->s); break;
    case J_ARRAY:
        buf_addc(o, '[');
        for (size_t i = 0; i < v->arr.len; i++) {
            if (i > 0) buf_adds(o, ", ");
            chat_to_json(o, v->arr.items[i]);
        }
        buf_addc(o, ']');
        break;
    case J_OBJECT:
        buf_addc(o, '{');
        for (size_t i = 0; i < v->obj.len; i++) {
            if (i > 0) buf_adds(o, ", ");
            chat_json_string(o, v->obj.items[i].key);
            buf_adds(o, ": ");
            chat_to_json(o, v->obj.items[i].val);
        }
        buf_addc(o, '}');
        break;
    }
}

// Length of the well-formed UTF-8 sequence at p[0..n) (Unicode Table 3-7), p[0] >= 0x80;
// 0 when ill-formed, with *bad the length of its maximal subpart (replaced by one U+FFFD).
static size_t utf8_seq(const uint8_t *p, size_t n, size_t *bad) {
    uint8_t b = p[0], lo = 0x80, hi = 0xBF;
    size_t need;
    if (b >= 0xC2 && b <= 0xDF) need = 1;
    else if (b >= 0xE0 && b <= 0xEF) {
        need = 2;
        if (b == 0xE0) lo = 0xA0;
        if (b == 0xED) hi = 0x9F;
    } else if (b >= 0xF0 && b <= 0xF4) {
        need = 3;
        if (b == 0xF0) lo = 0x90;
        if (b == 0xF4) hi = 0x8F;
    } else {
        *bad = 1;
        return 0;
    }
    for (size_t i = 1; i <= need; i++) {
        if (i >= n || p[i] < lo || p[i] > hi) {
            *bad = i;
            return 0;
        }
        lo = 0x80;
        hi = 0xBF;
    }
    return need + 1;
}

void chat_json_string(Buf *o, Str s) {
    buf_reserve(o, s.len + 2);
    buf_addc(o, '"');
    size_t run = 0; // start of the pending run of bytes written as they are
    for (size_t i = 0; i < s.len; i++) {
        uint8_t ch = (uint8_t)s.ptr[i];
        const char *esc = NULL;
        if (ch >= 0x80) {
            size_t bad = 0, n = utf8_seq((const uint8_t *)s.ptr + i, s.len - i, &bad);
            if (n > 0) {
                i += n - 1;
                continue;
            }
            buf_add(o, s.ptr + run, i - run);
            buf_adds(o, "\xef\xbf\xbd"); // U+FFFD
            i += bad - 1;
            run = i + 1;
            continue;
        }
        switch (ch) {
        case '"': esc = "\\\""; break;
        case '\\': esc = "\\\\"; break;
        case '\n': esc = "\\n"; break;
        case '\r': esc = "\\r"; break;
        case '\t': esc = "\\t"; break;
        case 0x08: esc = "\\b"; break;
        case 0x0c: esc = "\\f"; break;
        default:
            if (ch >= 0x20) continue;
        }
        buf_add(o, s.ptr + run, i - run);
        run = i + 1;
        if (esc) buf_adds(o, esc);
        else buf_printf(o, "\\u%04x", ch);
    }
    buf_add(o, s.ptr + run, s.len - run);
    buf_addc(o, '"');
}
