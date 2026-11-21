// JSON values, parser and writer: the subset of std.json the engine used.
// Objects keep insertion order (std.json.ObjectMap is an array hash map). A duplicate key
// in parsed input is an error ("DuplicateField"), std.json's default for Value.
#ifndef SAETTA_JSON_H
#define SAETTA_JSON_H

#include "base.h"

typedef enum { J_NULL, J_BOOL, J_INTEGER, J_FLOAT, J_NUMBER_STRING, J_STRING, J_ARRAY, J_OBJECT } JKind;

typedef struct JValue JValue;
typedef struct {
    Str key;
    JValue *val;
} JMember;

struct JValue {
    JKind kind;
    union {
        bool b;
        int64_t i;
        double f;
        Str s; // J_STRING (decoded, NUL terminated) and J_NUMBER_STRING (the literal text)
        struct {
            JValue **items;
            size_t len, cap;
        } arr;
        struct {
            JMember *items;
            size_t len, cap;
        } obj;
    };
};

// Parses a whole document into the arena. parse_numbers false keeps every number as
// J_NUMBER_STRING (std.json .parse_numbers = false); true gives J_INTEGER when the literal
// is an integer that fits i64, else J_FLOAT (std.json's default; an integer that overflows
// i64 or a float that overflows f64 stays J_NUMBER_STRING). Errors: "SyntaxError",
// "UnexpectedEndOfInput", "DuplicateField", "NestingTooDeep" (more than JSON_MAX_DEPTH
// nested arrays and objects; std.json has no limit, but the writers recurse); callers
// only test for failure. Strings, keys and number literals are copied into the arena.
enum { JSON_MAX_DEPTH = 128 };
Err json_parse(Arena *a, Str text, bool parse_numbers, JValue **out);

// Lookup and accessors (NULL / defaults when absent or of another kind).
JValue *json_get(const JValue *obj, const char *key); // NULL unless obj is an object with key
const char *json_str(const JValue *v);  // J_STRING contents or NULL
Str json_strs(const JValue *v);          // J_STRING contents or empty
bool json_is(const JValue *v, JKind k);

// Builders (arena owned).
JValue *json_new(Arena *a, JKind k);
JValue *json_string(Arena *a, Str s);
JValue *json_bool(Arena *a, bool b);
JValue *json_int(Arena *a, int64_t i);
void json_push(Arena *a, JValue *arr, JValue *v);
void json_put(Arena *a, JValue *obj, Str key, JValue *v); // replaces in place if present

// Writer, byte for byte as std.json.Stringify with default options (minified: no spaces;
// strings escaped as std.json does: \" \\ \n \r \t \b \f, other controls \u00XX, non-ASCII
// UTF-8 written raw, '/' not escaped; floats as std.json writes f64; integers in decimal;
// number strings verbatim).
void json_write(Buf *out, const JValue *v);
void json_write_string(Buf *out, Str s); // a quoted, escaped string

#endif
