// Common C helpers for saetta: errors, arenas, strings, growable arrays, a string map,
// files, time, formatting. Replaces the parts of the Zig standard library the engine used.
#ifndef SAETTA_BASE_H
#define SAETTA_BASE_H

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// --- errors ----------------------------------------------------------------
// A function that can fail returns Err: NULL on success, else a static string naming the
// error (the Zig error name it replaces, e.g. "FileNotFound"), so messages stay the same.
typedef const char *Err;
#define OK ((Err)0)
#define TRY(expr) do { Err e_ = (expr); if (e_) return e_; } while (0)
// Maps an errno value to the Zig-style name ("FileNotFound", "AccessDenied", ...).
Err err_errno(int e);

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLAMP(x, lo, hi) MIN(MAX(x, lo), hi)
#define ALIGN_UP(x, a) (((x) + (a) - 1) / (a) * (a))
#define COUNT(a) (sizeof(a) / sizeof((a)[0]))

// Prints to stderr (std.debug.print).
void dbg(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
// Where dbg writes (stderr when NULL): Saetta Code points it at a log file while the engine runs
// in its process, so the server's lines stay out of the agent's output.
extern FILE *dbg_out;
// Prints to stderr and exits with code 1.
_Noreturn void die(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

// --- memory ----------------------------------------------------------------
// xmalloc/xcalloc/xrealloc abort on out of memory (the engine treats it as fatal).
void *xmalloc(size_t n);
void *xcalloc(size_t n, size_t size);
void *xrealloc(void *p, size_t n);

// Bump allocator in chunks; everything is freed at once. Not thread safe.
typedef struct ArenaChunk ArenaChunk;
typedef struct {
    ArenaChunk *head;
} Arena;
void *arena_alloc(Arena *a, size_t n);            // 16-byte aligned, zero filled
void *arena_dup(Arena *a, const void *p, size_t n);
char *arena_strdup(Arena *a, const char *s);
char *arena_strndup(Arena *a, const char *s, size_t n); // NUL terminated copy
char *arena_printf(Arena *a, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
char *arena_vprintf(Arena *a, const char *fmt, va_list ap);
void arena_reset(Arena *a); // frees every chunk
#define ARENA_NEW(a, T) ((T *)arena_alloc((a), sizeof(T)))
#define ARENA_ARRAY(a, T, n) ((T *)arena_alloc((a), sizeof(T) * (size_t)(n)))

// --- strings ---------------------------------------------------------------
// A byte slice, not necessarily NUL terminated.
typedef struct {
    const char *ptr;
    size_t len;
} Str;
#define S(lit) ((Str){(lit), sizeof(lit) - 1})
static inline Str str(const char *s) { return (Str){s, s ? strlen(s) : 0}; }
static inline Str str_n(const char *s, size_t n) { return (Str){s, n}; }
bool str_eq(Str a, Str b);
bool str_eqz(Str a, const char *b);
bool str_starts(Str s, Str prefix);
bool str_ends(Str s, Str suffix);
// Index of the first occurrence of needle at or after pos, or -1.
ptrdiff_t str_find(Str hay, Str needle, size_t pos);
ptrdiff_t str_find_char(Str hay, char c, size_t pos);
Str str_trim(Str s);  // ASCII whitespace " \t\r\n" both ends
Str str_sub(Str s, size_t from, size_t to);
char *str_z(Arena *a, Str s); // NUL terminated copy in the arena
// Parses a whole decimal integer (optional leading '-' for the signed one). Errors:
// "InvalidCharacter", "Overflow".
Err parse_u64(Str s, uint64_t *out);
Err parse_i64(Str s, int64_t *out);

// Growable byte buffer, always NUL terminated after len (ptr may be NULL while empty).
typedef struct {
    char *ptr;
    size_t len, cap;
} Buf;
void buf_reserve(Buf *b, size_t extra);
void buf_add(Buf *b, const void *p, size_t n);
void buf_adds(Buf *b, const char *s);
void buf_addstr(Buf *b, Str s);
void buf_addc(Buf *b, char c);
void buf_printf(Buf *b, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
void buf_vprintf(Buf *b, const char *fmt, va_list ap);
void buf_free(Buf *b);
static inline Str buf_str(const Buf *b) { return (Str){b->ptr ? b->ptr : "", b->len}; }

// Growable typed array: VEC(T) declares an anonymous struct {T *ptr; size_t len, cap;}.
// Use typedef VEC(uint32_t) VecU32; then vec_push(&v, x). Memory is malloc'd; vec_free.
#define VEC(T) struct { T *ptr; size_t len, cap; }
#define vec_reserve(v, extra) \
    do { size_t need_ = (v)->len + (extra); if (need_ > (v)->cap) { size_t nc_ = (v)->cap ? (v)->cap * 2 : 8; \
        while (nc_ < need_) nc_ *= 2; (v)->ptr = xrealloc((v)->ptr, nc_ * sizeof(*(v)->ptr)); (v)->cap = nc_; } } while (0)
#define vec_push(v, x) do { vec_reserve((v), 1); (v)->ptr[(v)->len++] = (x); } while (0)
#define vec_free(v) do { free((v)->ptr); (v)->ptr = NULL; (v)->len = (v)->cap = 0; } while (0)
typedef VEC(uint32_t) VecU32;
typedef VEC(Str) VecStr;

// String-keyed hash map (open addressing) to a pointer-sized value. Keys are not copied:
// they must outlive the map. Iteration order is unspecified.
typedef struct {
    Str *keys;
    void **vals;
    uint64_t *hashes; // 0 = empty slot
    size_t cap, len;
} StrMap;
void strmap_put(StrMap *m, Str key, void *val); // replaces an existing key
void **strmap_get(const StrMap *m, Str key);    // pointer to the value slot, or NULL
bool strmap_has(const StrMap *m, Str key);
void strmap_free(StrMap *m);
// Iterates: for (size_t i = 0; strmap_next(m, &i, &k, &v);) {...}
bool strmap_next(const StrMap *m, size_t *it, Str *key, void **val);
uint64_t hash_bytes(const void *p, size_t n); // wyhash-like 64-bit, never 0

// --- files and environment -------------------------------------------------
// Reads a whole file (NUL terminated after len). Errors: errno names, "StreamTooLong" when the
// file reaches limit bytes (Zig readFileAlloc .limited: a file of exactly limit bytes fails).
Err read_file(const char *path, size_t limit, char **out, size_t *len);
Err write_file(const char *path, const void *p, size_t n); // create/truncate
bool file_exists(const char *path);
Err mkdir_p(const char *path);            // like mkdir -p
char *path_join(Arena *a, const char *x, const char *y); // "x/y"
const char *env_or(const char *name, const char *fallback);

// --- time ------------------------------------------------------------------
uint64_t now_ns(void);   // monotonic (CLOCK_UPTIME_RAW: the awake clock)
uint64_t wall_ns(void);  // CLOCK_REALTIME
double now_s(void);

// --- formatting ------------------------------------------------------------
// Zig's "{d}" for floats: the shortest decimal that round-trips, no exponent
// (e.g. 0.1, 1, 1e-7 as 0.0000001). Writes into out (cap >= 64 is enough for f32, 330
// for any f64), truncating if shorter; returns the full length.
int fmt_f64(char *out, size_t cap, double x);
int fmt_f32(char *out, size_t cap, float x);
// Zig's "{d:.N}": not printf "%.Nf". Zig rounds the shortest digits half up (0.5 at .0 is
// "1", 2.25 at .1 is "2.3"), and an f32 argument rounds its own shortest digits (0.35f at
// .1 is "0.4", the widened double gives "0.3"), so f32 values use fmt_f32 / fmt_fixed_f32.
// Output wider than Zig's 347-byte buffer is "(float)", as in Zig.
int fmt_fixed(char *out, size_t cap, double x, int prec);
int fmt_fixed_f32(char *out, size_t cap, float x, int prec);

#endif
