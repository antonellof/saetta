// Chat prompt rendering for the Qwen3.8-Flash-Next and Inkling 2 templates (see chat.h).
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

// ---- Inkling 2 -------------------------------------------------------------------------
// The template (HF thinkingmachines/Inkling-2-Mini b85c3cf chat_template.jinja) as jinja2
// renders it the way transformers does. Its values follow Python: truthiness, str.strip(),
// json.dumps(ensure_ascii=False) on what json.loads made of the request (so a float literal
// is written as Python's repr of its double: 1.50 as 1.5, 1e2 as 100.0).

// Python truthiness of a JSON value (absent is false).
static bool truthy(const JValue *v) {
    if (!v) return false;
    switch (v->kind) {
    case J_NULL: return false;
    case J_BOOL: return v->b;
    case J_INTEGER: return v->i != 0;
    case J_FLOAT: return v->f != 0;
    case J_NUMBER_STRING: return strtod(v->s.ptr, NULL) != 0;
    case J_STRING: return v->s.len > 0;
    case J_ARRAY: return v->arr.len > 0;
    case J_OBJECT: return v->obj.len > 0;
    }
    return false;
}

static bool is_str_v(const JValue *v) { return v && v->kind == J_STRING; }

static bool number_value(const JValue *v, double *x) {
    if (v->kind == J_INTEGER) *x = (double)v->i;
    else if (v->kind == J_FLOAT) *x = v->f;
    else if (v->kind == J_NUMBER_STRING) *x = strtod(v->s.ptr, NULL);
    else return false;
    return true;
}

// Python repr(float) (json.dumps writes floats with it, and inf / nan as Infinity / NaN):
// the shortest digits that round-trip, positional when the decimal exponent is in [-4, 16),
// else d.ddde+XX.
static void py_float(Buf *o, double x) {
    if (x != x) {
        buf_adds(o, "NaN");
        return;
    }
    if (x == __builtin_inf() || x == -__builtin_inf()) {
        buf_adds(o, x > 0 ? "Infinity" : "-Infinity");
        return;
    }
    char e[48];
    for (int prec = 1; prec <= 17; prec++) {
        snprintf(e, sizeof e, "%.*e", prec - 1, x);
        if (strtod(e, NULL) == x) break;
    }
    // e is [-]d[.ddd]e(+|-)XX
    const char *p = e;
    bool neg = *p == '-';
    if (neg) p++;
    char dig[24];
    size_t nd = 0;
    for (; *p && *p != 'e'; p++)
        if (*p != '.') dig[nd++] = *p;
    int exp = atoi(p + 1);
    while (nd > 1 && dig[nd - 1] == '0') nd--;
    if (neg) buf_addc(o, '-');
    if (exp >= -4 && exp < 16) {
        if (exp < 0) {
            buf_adds(o, "0.");
            for (int k = 0; k < -exp - 1; k++) buf_addc(o, '0');
            buf_add(o, dig, nd);
        } else {
            size_t ip = (size_t)exp + 1;
            for (size_t k = 0; k < ip; k++) buf_addc(o, k < nd ? dig[k] : '0');
            buf_addc(o, '.');
            if (nd > ip) buf_add(o, dig + ip, nd - ip);
            else buf_addc(o, '0');
        }
    } else {
        buf_addc(o, dig[0]);
        if (nd > 1) {
            buf_addc(o, '.');
            buf_add(o, dig + 1, nd - 1);
        }
        buf_printf(o, "e%c%02d", exp < 0 ? '-' : '+', exp < 0 ? -exp : exp);
    }
}

// A JSON number as Python writes what json.loads made of it: integers as spelled (-0 is 0),
// floats as repr.
static void py_number(Buf *o, const JValue *v) {
    if (v->kind == J_INTEGER) {
        buf_printf(o, "%" PRId64, v->i);
        return;
    }
    if (v->kind == J_FLOAT) {
        py_float(o, v->f);
        return;
    }
    Str s = v->s;
    bool is_float = false;
    for (size_t i = 0; i < s.len; i++)
        if (s.ptr[i] == '.' || s.ptr[i] == 'e' || s.ptr[i] == 'E') is_float = true;
    if (is_float) py_float(o, strtod(s.ptr, NULL));
    else if (str_eqz(s, "-0")) buf_addc(o, '0');
    else buf_addstr(o, s);
}

// Byte order of two keys (UTF-8 byte order is code point order, as Python sorts str).
static int key_cmp(Str x, Str y) {
    int c = memcmp(x.ptr, y.ptr, MIN(x.len, y.len));
    return c ? c : (x.len > y.len) - (x.len < y.len);
}

// The members of object v, sorted by key when sort (stable insertion sort); free() it.
static const JMember **members(const JValue *v, bool sort) {
    size_t n = v->obj.len;
    const JMember **ms = xmalloc((n ? n : 1) * sizeof *ms);
    for (size_t i = 0; i < n; i++) ms[i] = &v->obj.items[i];
    for (size_t i = 1; sort && i < n; i++) {
        const JMember *x = ms[i];
        size_t j = i;
        while (j > 0 && key_cmp(ms[j - 1]->key, x->key) > 0) {
            ms[j] = ms[j - 1];
            j--;
        }
        ms[j] = x;
    }
    return ms;
}

// json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=sort).
static void compact_json(Buf *o, const JValue *v, bool sort) {
    switch (v->kind) {
    case J_NULL: buf_adds(o, "null"); break;
    case J_BOOL: buf_adds(o, v->b ? "true" : "false"); break;
    case J_INTEGER:
    case J_FLOAT:
    case J_NUMBER_STRING: py_number(o, v); break;
    case J_STRING: chat_json_string(o, v->s); break;
    case J_ARRAY:
        buf_addc(o, '[');
        for (size_t i = 0; i < v->arr.len; i++) {
            if (i > 0) buf_addc(o, ',');
            compact_json(o, v->arr.items[i], sort);
        }
        buf_addc(o, ']');
        break;
    case J_OBJECT: {
        size_t n = v->obj.len;
        const JMember **ms = members(v, sort);
        buf_addc(o, '{');
        for (size_t i = 0; i < n; i++) {
            if (i > 0) buf_addc(o, ',');
            chat_json_string(o, ms[i]->key);
            buf_addc(o, ':');
            compact_json(o, ms[i]->val, sort);
        }
        buf_addc(o, '}');
        free(ms);
        break;
    }
    }
}

static const struct {
    const char *name;
    double value;
} ink_efforts[] = {{"none", 0.0}, {"minimal", 0.1}, {"low", 0.2}, {"medium", 0.7}, {"high", 0.9}, {"xhigh", 0.99}, {"max", 0.99}};

// The kwarg as the template sees it: chat_template_kwargs first (null included), then the
// request's own field.
static const JValue *kwarg_raw(const JValue *req, const char *key) {
    const JValue *k = get(req, "chat_template_kwargs");
    const JValue *v = k ? json_get(k, key) : NULL;
    return v ? v : json_get(req, key);
}

// emit_thinking_effort(): default 0.9; a name from the map (stripped) or a number (jinja's
// float filter: true is 1.0, anything else that is no number 0.0); in [0, 0.99].
static Err ink_effort(const JValue *req, Buf *o) {
    const JValue *e = kwarg_raw(req, "reasoning_effort");
    double num = 0.9;
    if (e && e->kind == J_STRING) {
        Str key = trim(e->s);
        size_t k = 0;
        while (k < COUNT(ink_efforts) && !str_eqz(key, ink_efforts[k].name)) k++;
        if (k == COUNT(ink_efforts)) return "BadReasoningEffort";
        num = ink_efforts[k].value;
    } else if (e && e->kind != J_NULL) {
        if (!number_value(e, &num)) num = e->kind == J_BOOL && e->b ? 1.0 : 0.0;
    }
    if (!(num >= 0.0 && num <= 0.99)) return "BadReasoningEffort";
    buf_adds(o, "<|message_system|><|content_text|>Thinking effort level: ");
    if (num == 0.0) buf_addc(o, '0');
    else py_float(o, num);
    buf_adds(o, "<|end_message|>");
    return OK;
}

static Err ink_tools(const JValue *tools, Buf *o) {
    if (!truthy(tools)) return OK;
    if (tools->kind != J_ARRAY) return "BadTool";
    buf_adds(o, "<|message_system|>tool_declare<|content_xml|>[");
    for (size_t i = 0; i < tools->arr.len; i++) {
        const JValue *tool = tools->arr.items[i];
        if (tool->kind != J_OBJECT) return "BadTool";
        const JValue *fn = json_get(tool, "function");
        if (!fn) fn = tool;
        if (fn->kind != J_OBJECT) return "BadTool";
        const JValue *d = json_get(fn, "description"), *n = json_get(fn, "name"), *p = json_get(fn, "parameters"),
                     *t = json_get(tool, "type");
        if (!n) return "BadTool";
        // the spec's keys, already in sorted order: description, name, parameters, type
        if (i > 0) buf_addc(o, ',');
        buf_adds(o, "{\"description\":");
        if (truthy(d)) compact_json(o, d, true);
        else buf_adds(o, "\"\"");
        buf_adds(o, ",\"name\":");
        compact_json(o, n, true);
        buf_adds(o, ",\"parameters\":");
        if (truthy(p)) compact_json(o, p, true);
        else buf_adds(o, "{}");
        buf_adds(o, ",\"type\":");
        if (truthy(t)) compact_json(o, t, true);
        else buf_adds(o, "\"function\"");
        buf_addc(o, '}');
    }
    buf_adds(o, "]<|end_message|>");
    return OK;
}

static void ink_text_block(Buf *o, Str rtok, Str text) {
    buf_addstr(o, rtok);
    buf_adds(o, "<|content_text|>");
    buf_addstr(o, text);
    buf_adds(o, "<|end_message|>");
}

// The message's content blocks (string, or a list of parts).
static Err ink_content(const JValue *m, bool assistant, Str rtok, Buf *o) {
    const JValue *c = json_get(m, "content");
    if (is_str_v(c)) {
        bool skip_empty = assistant && (truthy(json_get(m, "tool_calls")) || truthy(json_get(m, "reasoning_content")));
        if (c->s.len > 0 || !skip_empty) ink_text_block(o, rtok, c->s);
        return OK;
    }
    if (!truthy(c)) return OK;
    if (c->kind != J_ARRAY) return "BadContent";
    for (size_t i = 0; i < c->arr.len; i++) {
        const JValue *part = c->arr.items[i];
        if (is_str_v(part)) {
            ink_text_block(o, rtok, part->s);
            continue;
        }
        if (part->kind != J_OBJECT) return "BadContent";
        const JValue *ty = json_get(part, "type");
        Str t = json_strs(ty);
        if (!ty || (is_str_v(ty) && (str_eqz(t, "text") || str_eqz(t, "input_text")))) {
            const JValue *tx = json_get(part, "text");
            ink_text_block(o, rtok, is_str_v(tx) ? tx->s : S(""));
        } else if (is_str_v(ty) && (str_eqz(t, "image") || str_eqz(t, "input_image") || str_eqz(t, "image_url"))) {
            return "VisionUnsupported"; // the template writes <|content_image|><|unused_200054|> here
        } else {
            return "BadContent"; // audio and unknown part types are template errors
        }
    }
    return OK;
}

static Err ink_tool_calls(Arena *a, const JValue *m, Buf *o) {
    const JValue *tcs = json_get(m, "tool_calls");
    if (!truthy(tcs)) return OK;
    if (tcs->kind != J_ARRAY) return "BadToolCall";
    const JValue *mname = json_get(m, "name");
    for (size_t j = 0; j < tcs->arr.len; j++) {
        const JValue *fn = json_get(tcs->arr.items[j], "function");
        const JValue *n = json_get(fn, "name");
        if (!is_str_v(n)) return "BadToolCall";
        if (truthy(mname) && !(is_str_v(mname) && str_eq(mname->s, n->s))) return "BadToolCall";
        const JValue *args = json_get(fn, "arguments");
        // OpenAI sends arguments as a JSON string; the template wants the parsed object
        if (is_str_v(args) && args->s.len > 0) {
            JValue *p;
            if (json_parse(a, args->s, false, &p)) return "BadToolCall";
            args = p;
        }
        if (truthy(args) && args->kind != J_OBJECT) return "BadToolCall";
        buf_adds(o, "<|message_model|>");
        buf_addstr(o, n->s);
        buf_adds(o, "<|content_invoke_tool_json|>{\"name\":");
        chat_json_string(o, n->s);
        buf_adds(o, ",\"args\":{");
        if (truthy(args)) {
            // args | dictsort(case_sensitive=true): top-level keys sorted, values compact as given
            size_t na = args->obj.len;
            const JMember **ms = members(args, true);
            for (size_t i = 0; i < na; i++) {
                if (i > 0) buf_addc(o, ',');
                chat_json_string(o, ms[i]->key);
                buf_addc(o, ':');
                compact_json(o, ms[i]->val, false);
            }
            free(ms);
        }
        buf_adds(o, "}}<|end_message|>");
    }
    return OK;
}

static const struct {
    const char *role, *token;
} ink_roles[] = {{"user", "<|message_user|>"}, {"assistant", "<|message_model|>"}, {"system", "<|message_system|>"}, {"tool", "<|message_tool|>"}};

static Err render_inkling(Arena *a, const JValue *req, Buf *o) {
    const JValue *msgs = get(req, "messages");
    if (!msgs || msgs->kind != J_ARRAY) return "NoMessages";
    TRY(ink_tools(get(req, "tools"), o));
    bool effort_done = false;
    Buf hdr = {0};
    Err err = OK;
    for (size_t i = 0; i < msgs->arr.len && !err; i++) {
        const JValue *m = msgs->arr.items[i];
        Str r = role(m);
        size_t k = 0;
        while (k < COUNT(ink_roles) && !str_eqz(r, ink_roles[k].role)) k++;
        if (k == COUNT(ink_roles)) {
            err = "BadRole";
            break;
        }
        if (!effort_done && !str_eqz(r, "system")) {
            effort_done = true;
            if ((err = ink_effort(req, o))) break;
        }
        bool assistant = str_eqz(r, "assistant"), tool = str_eqz(r, "tool");
        hdr.len = 0;
        buf_adds(&hdr, ink_roles[k].token);
        const JValue *name = json_get(m, "name");
        if ((tool || assistant) && truthy(name)) {
            if (!is_str_v(name)) {
                err = "BadRole";
                break;
            }
            buf_addstr(&hdr, name->s);
        } else if (tool) {
            // the name of every earlier or later assistant tool call with this id
            const JValue *id = json_get(m, "tool_call_id");
            for (size_t p = 0; truthy(id) && p < msgs->arr.len; p++) {
                const JValue *pm = msgs->arr.items[p];
                if (!str_eqz(role(pm), "assistant")) continue;
                const JValue *tcs = json_get(pm, "tool_calls");
                if (!truthy(tcs) || tcs->kind != J_ARRAY) continue;
                for (size_t q = 0; q < tcs->arr.len; q++) {
                    const JValue *tc = tcs->arr.items[q];
                    const JValue *tid = json_get(tc, "id");
                    const JValue *tn = json_get(json_get(tc, "function"), "name");
                    if (is_str_v(tid) && is_str_v(id) && str_eq(tid->s, id->s) && is_str_v(tn)) buf_addstr(&hdr, tn->s);
                }
            }
        }
        Str rtok = buf_str(&hdr);
        const JValue *rc = json_get(m, "reasoning_content");
        if (assistant && truthy(rc)) {
            if (!is_str_v(rc)) {
                err = "BadContent";
                break;
            }
            buf_addstr(o, rtok);
            buf_adds(o, "<|content_thinking|>");
            buf_addstr(o, rc->s);
            buf_adds(o, "<|end_message|>");
        }
        if ((err = ink_content(m, assistant, rtok, o))) break;
        if (assistant) {
            if ((err = ink_tool_calls(a, m, o))) break;
            buf_adds(o, "<|content_model_end_sampling|>");
        }
    }
    buf_free(&hdr);
    if (!err && !effort_done) err = ink_effort(req, o);
    return err;
}

Err chat_render_as(Arena *a, ChatFormat f, const JValue *req, Str *out) {
    Buf o = {0};
    Err err = f == CHAT_INKLING ? render_inkling(a, req, &o) : render(a, req, &o);
    if (!err) *out = str_n(arena_dup(a, o.ptr ? o.ptr : "", o.len + 1), o.len);
    buf_free(&o);
    return err;
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
