// Implementation of base.h.
#include "base.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>

// --- errors ----------------------------------------------------------------

// Names follow the pinned Zig std (Io/Threaded.zig) mapping of each errno.
Err err_errno(int e) {
    switch (e) {
    case 0: return OK;
    case ENOENT: return "FileNotFound";
    case EACCES: return "AccessDenied";
    case EPERM: return "PermissionDenied";
    case EISDIR: return "IsDir";
    case ENOTDIR: return "NotDir";
    case EEXIST: return "PathAlreadyExists";
    case ENOSPC: return "NoSpaceLeft";
    case EDQUOT: return "DiskQuota";
    case EROFS: return "ReadOnlyFileSystem";
    case ENAMETOOLONG: return "NameTooLong";
    case ELOOP: return "SymLinkLoop";
    case EMFILE: return "ProcessFdQuotaExceeded";
    case ENFILE: return "SystemFdQuotaExceeded";
    case ENOMEM: return "SystemResources";
    case ENOBUFS: return "SystemResources";
    case EFBIG: return "FileTooBig";
    case EOVERFLOW: return "FileTooBig";
    case EBUSY: return "DeviceBusy";
    case ETXTBSY: return "FileBusy";
    case EAGAIN: return "WouldBlock";
    case EIO: return "InputOutput";
    case ENOTEMPTY: return "DirNotEmpty";
    case EXDEV: return "CrossDevice";
    case EMLINK: return "LinkQuotaExceeded";
    case ENODEV: return "NoDevice";
    case ENXIO: return "NoDevice";
    case EINVAL: return "BadPathName";
    case EILSEQ: return "BadPathName";
    case EPIPE: return "BrokenPipe";
    case ECONNRESET: return "ConnectionResetByPeer";
    case ECONNREFUSED: return "ConnectionRefused";
    case ETIMEDOUT: return "ConnectionTimedOut";
    case ENOTCONN: return "SocketUnconnected";
    case EADDRINUSE: return "AddressInUse";
    case EADDRNOTAVAIL: return "AddressUnavailable";
    case EAFNOSUPPORT: return "AddressFamilyUnsupported";
    case ENETUNREACH: return "NetworkUnreachable";
    case EHOSTUNREACH: return "HostUnreachable";
    case ECONNABORTED: return "ConnectionAborted";
    case ECANCELED: return "Canceled";
    default: return "Unexpected";
    }
}

void dbg(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

_Noreturn void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    exit(1);
}

// --- memory ----------------------------------------------------------------

static _Noreturn void oom(void) {
    fputs("out of memory\n", stderr);
    abort();
}

void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) oom();
    return p;
}

void *xcalloc(size_t n, size_t size) {
    void *p = calloc(n ? n : 1, size ? size : 1);
    if (!p) oom();
    return p;
}

void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) oom();
    return q;
}

struct ArenaChunk {
    ArenaChunk *next;
    size_t used, cap;
    _Alignas(16) unsigned char data[];
};

#define ARENA_CHUNK (64u << 10)

void *arena_alloc(Arena *a, size_t n) {
    n = ALIGN_UP(n ? n : 1, 16);
    ArenaChunk *c = a->head;
    if (!c || c->cap - c->used < n) {
        size_t cap = n > ARENA_CHUNK / 4 ? n : ARENA_CHUNK;
        ArenaChunk *nc = xmalloc(sizeof(ArenaChunk) + cap);
        nc->used = 0;
        nc->cap = cap;
        if (c && cap == n) {
            // A large block goes behind the current chunk so its free space stays usable.
            nc->next = c->next;
            c->next = nc;
            nc->used = n;
            memset(nc->data, 0, n);
            return nc->data;
        }
        nc->next = c;
        a->head = c = nc;
    }
    void *p = c->data + c->used;
    c->used += n;
    memset(p, 0, n);
    return p;
}

void *arena_dup(Arena *a, const void *p, size_t n) {
    void *q = arena_alloc(a, n);
    if (n) memcpy(q, p, n);
    return q;
}

char *arena_strdup(Arena *a, const char *s) { return arena_strndup(a, s, strlen(s)); }

char *arena_strndup(Arena *a, const char *s, size_t n) {
    char *q = arena_alloc(a, n + 1);
    if (n) memcpy(q, s, n);
    q[n] = 0;
    return q;
}

char *arena_vprintf(Arena *a, const char *fmt, va_list ap) {
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap2);
    va_end(ap2);
    if (n < 0) n = 0;
    char *q = arena_alloc(a, (size_t)n + 1);
    vsnprintf(q, (size_t)n + 1, fmt, ap);
    return q;
}

char *arena_printf(Arena *a, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char *q = arena_vprintf(a, fmt, ap);
    va_end(ap);
    return q;
}

void arena_reset(Arena *a) {
    ArenaChunk *c = a->head;
    while (c) {
        ArenaChunk *n = c->next;
        free(c);
        c = n;
    }
    a->head = NULL;
}

// --- strings ---------------------------------------------------------------

bool str_eq(Str a, Str b) { return a.len == b.len && (a.len == 0 || memcmp(a.ptr, b.ptr, a.len) == 0); }
bool str_eqz(Str a, const char *b) { return str_eq(a, str(b)); }
bool str_starts(Str s, Str p) { return s.len >= p.len && (p.len == 0 || memcmp(s.ptr, p.ptr, p.len) == 0); }
bool str_ends(Str s, Str p) {
    return s.len >= p.len && (p.len == 0 || memcmp(s.ptr + s.len - p.len, p.ptr, p.len) == 0);
}

ptrdiff_t str_find(Str hay, Str needle, size_t pos) {
    if (pos > hay.len) return -1;
    if (needle.len == 0) return (ptrdiff_t)pos;
    if (needle.len > hay.len - pos) return -1;
    const char *p = memmem(hay.ptr + pos, hay.len - pos, needle.ptr, needle.len);
    return p ? p - hay.ptr : -1;
}

ptrdiff_t str_find_char(Str hay, char c, size_t pos) {
    if (pos >= hay.len) return -1;
    const char *p = memchr(hay.ptr + pos, c, hay.len - pos);
    return p ? p - hay.ptr : -1;
}

static bool is_ws(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

Str str_trim(Str s) {
    size_t b = 0, e = s.len;
    while (b < e && is_ws(s.ptr[b])) b++;
    while (e > b && is_ws(s.ptr[e - 1])) e--;
    return (Str){s.ptr + b, e - b};
}

Str str_sub(Str s, size_t from, size_t to) {
    if (to > s.len) to = s.len;
    if (from > to) from = to;
    return (Str){s.ptr + from, to - from};
}

char *str_z(Arena *a, Str s) { return arena_strndup(a, s.ptr ? s.ptr : "", s.len); }

// std.fmt.parseInt(T, s, 10): optional '+' or '-', underscores allowed between digits.
static Err parse_int(Str s, bool is_signed, uint64_t *u, int64_t *i) {
    if (s.len == 0) return "InvalidCharacter";
    bool neg = false;
    if (s.ptr[0] == '+' || s.ptr[0] == '-') {
        neg = s.ptr[0] == '-';
        s.ptr++;
        s.len--;
    }
    if (s.len == 0) return "InvalidCharacter";
    if (s.ptr[0] == '_' || s.ptr[s.len - 1] == '_') return "InvalidCharacter";
    uint64_t au = 0;
    int64_t ai = 0;
    for (size_t k = 0; k < s.len; k++) {
        char c = s.ptr[k];
        if (c == '_') continue;
        if (c < '0' || c > '9') return "InvalidCharacter";
        int d = c - '0';
        if (is_signed) {
            if (__builtin_mul_overflow(ai, 10, &ai)) return "Overflow";
            if (neg ? __builtin_sub_overflow(ai, d, &ai) : __builtin_add_overflow(ai, d, &ai)) return "Overflow";
        } else {
            if (neg) {
                if (d != 0 || au != 0) return "Overflow";
                continue;
            }
            if (__builtin_mul_overflow(au, 10, &au)) return "Overflow";
            if (__builtin_add_overflow(au, (uint64_t)d, &au)) return "Overflow";
        }
    }
    if (u) *u = au;
    if (i) *i = ai;
    return OK;
}

Err parse_u64(Str s, uint64_t *out) { return parse_int(s, false, out, NULL); }
Err parse_i64(Str s, int64_t *out) { return parse_int(s, true, NULL, out); }

// --- Buf -------------------------------------------------------------------

void buf_reserve(Buf *b, size_t extra) {
    size_t need = b->len + extra + 1;
    if (need <= b->cap) return;
    size_t nc = b->cap ? b->cap * 2 : 64;
    while (nc < need) nc *= 2;
    b->ptr = xrealloc(b->ptr, nc);
    b->cap = nc;
}

void buf_add(Buf *b, const void *p, size_t n) {
    buf_reserve(b, n);
    if (n) memcpy(b->ptr + b->len, p, n);
    b->len += n;
    b->ptr[b->len] = 0;
}

void buf_adds(Buf *b, const char *s) { buf_add(b, s, strlen(s)); }
void buf_addstr(Buf *b, Str s) { buf_add(b, s.ptr, s.len); }
void buf_addc(Buf *b, char c) { buf_add(b, &c, 1); }

void buf_vprintf(Buf *b, const char *fmt, va_list ap) {
    va_list ap2;
    va_copy(ap2, ap);
    char tmp[256];
    int n = vsnprintf(tmp, sizeof tmp, fmt, ap2);
    va_end(ap2);
    if (n < 0) return;
    if ((size_t)n < sizeof tmp) {
        buf_add(b, tmp, (size_t)n);
        return;
    }
    buf_reserve(b, (size_t)n);
    vsnprintf(b->ptr + b->len, (size_t)n + 1, fmt, ap);
    b->len += (size_t)n;
}

void buf_printf(Buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    buf_vprintf(b, fmt, ap);
    va_end(ap);
}

void buf_free(Buf *b) {
    free(b->ptr);
    b->ptr = NULL;
    b->len = b->cap = 0;
}

// --- StrMap ----------------------------------------------------------------

static inline uint64_t mix64(uint64_t a, uint64_t b) {
    __uint128_t r = (__uint128_t)a * b;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
}

static inline uint64_t rd64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint64_t rd32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }

uint64_t hash_bytes(const void *data, size_t n) {
    const uint64_t s0 = 0xa0761d6478bd642full, s1 = 0xe7037ed1a0b428dbull, s2 = 0x8ebc6af09c88c6e3ull;
    const unsigned char *p = data;
    uint64_t seed = s0 ^ mix64(n, s1);
    size_t i = n;
    while (i > 16) {
        seed = mix64(rd64(p) ^ s1, rd64(p + 8) ^ seed);
        p += 16;
        i -= 16;
    }
    uint64_t a, b;
    if (i > 8) {
        a = rd64(p);
        b = rd64(p + i - 8);
    } else if (i >= 4) {
        a = (rd32(p) << 32) | rd32(p + ((i >> 3) << 2));
        b = (rd32(p + i - 4) << 32) | rd32(p + i - 4 - ((i >> 3) << 2));
    } else if (i > 0) {
        a = ((uint64_t)p[0] << 16) | ((uint64_t)p[i >> 1] << 8) | p[i - 1];
        b = 0;
    } else {
        a = b = 0;
    }
    uint64_t h = mix64(s1 ^ n, mix64(a ^ s1, b ^ seed) ^ s2);
    return h ? h : 1;
}

static void strmap_grow(StrMap *m) {
    size_t nc = m->cap ? m->cap * 2 : 16;
    Str *keys = xcalloc(nc, sizeof(Str));
    void **vals = xcalloc(nc, sizeof(void *));
    uint64_t *hs = xcalloc(nc, sizeof(uint64_t));
    for (size_t i = 0; i < m->cap; i++) {
        if (!m->hashes[i]) continue;
        size_t j = m->hashes[i] & (nc - 1);
        while (hs[j]) j = (j + 1) & (nc - 1);
        hs[j] = m->hashes[i];
        keys[j] = m->keys[i];
        vals[j] = m->vals[i];
    }
    free(m->keys);
    free(m->vals);
    free(m->hashes);
    m->keys = keys;
    m->vals = vals;
    m->hashes = hs;
    m->cap = nc;
}

static ptrdiff_t strmap_slot(const StrMap *m, Str key, uint64_t h) {
    if (!m->cap) return -1;
    size_t j = h & (m->cap - 1);
    while (m->hashes[j]) {
        if (m->hashes[j] == h && str_eq(m->keys[j], key)) return (ptrdiff_t)j;
        j = (j + 1) & (m->cap - 1);
    }
    return -1;
}

void strmap_put(StrMap *m, Str key, void *val) {
    uint64_t h = hash_bytes(key.ptr, key.len);
    ptrdiff_t s = strmap_slot(m, key, h);
    if (s >= 0) {
        m->vals[s] = val;
        return;
    }
    if ((m->len + 1) * 4 > m->cap * 3) strmap_grow(m);
    size_t j = h & (m->cap - 1);
    while (m->hashes[j]) j = (j + 1) & (m->cap - 1);
    m->hashes[j] = h;
    m->keys[j] = key;
    m->vals[j] = val;
    m->len++;
}

void **strmap_get(const StrMap *m, Str key) {
    ptrdiff_t s = strmap_slot(m, key, hash_bytes(key.ptr, key.len));
    return s >= 0 ? &m->vals[s] : NULL;
}

bool strmap_has(const StrMap *m, Str key) { return strmap_get(m, key) != NULL; }

void strmap_free(StrMap *m) {
    free(m->keys);
    free(m->vals);
    free(m->hashes);
    memset(m, 0, sizeof *m);
}

bool strmap_next(const StrMap *m, size_t *it, Str *key, void **val) {
    while (*it < m->cap) {
        size_t i = (*it)++;
        if (m->hashes[i]) {
            if (key) *key = m->keys[i];
            if (val) *val = m->vals[i];
            return true;
        }
    }
    return false;
}

// --- files and environment -------------------------------------------------

Err read_file(const char *path, size_t limit, char **out, size_t *len) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return err_errno(errno);
    struct stat st;
    if (fstat(fd, &st) != 0) {
        int e = errno;
        close(fd);
        return err_errno(e);
    }
    if (S_ISDIR(st.st_mode)) {
        close(fd);
        return "IsDir";
    }
    size_t cap = st.st_size > 0 ? (size_t)st.st_size : 4096;
    if (cap > limit) cap = limit;
    char *p = xmalloc(cap + 1);
    size_t n = 0;
    for (;;) {
        if (n == cap) {
            if (cap >= limit) {
                // Zig's readFileAlloc(.limited(n)): reaching the limit is StreamTooLong
                free(p);
                close(fd);
                return "StreamTooLong";
            }
            cap = cap * 2 > limit ? limit : cap * 2;
            p = xrealloc(p, cap + 1);
        }
        ssize_t r = read(fd, p + n, cap - n);
        if (r < 0) {
            if (errno == EINTR) continue;
            int e = errno;
            free(p);
            close(fd);
            return err_errno(e);
        }
        if (r == 0) break;
        n += (size_t)r;
    }
    close(fd);
    p[n] = 0;
    *out = p;
    *len = n;
    return OK;
}

Err write_file(const char *path, const void *data, size_t n) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
    if (fd < 0) return err_errno(errno);
    const char *p = data;
    while (n) {
        ssize_t w = write(fd, p, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            int e = errno;
            close(fd);
            return err_errno(e);
        }
        p += w;
        n -= (size_t)w;
    }
    if (close(fd) != 0) return err_errno(errno);
    return OK;
}

bool file_exists(const char *path) { return access(path, F_OK) == 0; }

Err mkdir_p(const char *path) {
    size_t n = strlen(path);
    if (n == 0) return OK;
    char *tmp = xmalloc(n + 1);
    memcpy(tmp, path, n + 1);
    for (size_t i = 1; i <= n; i++) {
        if (i < n && tmp[i] != '/') continue;
        char save = tmp[i];
        tmp[i] = 0;
        if (mkdir(tmp, 0777) != 0 && errno != EEXIST) {
            int e = errno;
            free(tmp);
            return err_errno(e);
        }
        tmp[i] = save;
    }
    struct stat st;
    int r = stat(path, &st);
    free(tmp);
    if (r != 0) return err_errno(errno);
    if (!S_ISDIR(st.st_mode)) return "NotDir";
    return OK;
}

// std.fs.path.join: empty parts are skipped, one separator at the junction.
char *path_join(Arena *a, const char *x, const char *y) {
    size_t nx = strlen(x), ny = strlen(y);
    if (nx == 0) return arena_strndup(a, y, ny);
    if (ny == 0) return arena_strndup(a, x, nx);
    bool xs = x[nx - 1] == '/', ys = y[0] == '/';
    if (xs && ys) return arena_printf(a, "%s%s", x, y + 1);
    if (xs || ys) return arena_printf(a, "%s%s", x, y);
    return arena_printf(a, "%s/%s", x, y);
}

const char *env_or(const char *name, const char *fallback) {
    const char *v = getenv(name);
    return v ? v : fallback;
}

// --- time ------------------------------------------------------------------

uint64_t now_ns(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

uint64_t wall_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

double now_s(void) { return (double)now_ns() / 1e9; }

// --- formatting ------------------------------------------------------------

// Shortest decimal digits that read back as x (as std.fmt.float's Ryu): digits into d
// (no leading or trailing zeros), returns the count; *e10 is the decimal exponent of the
// first digit. Among the shortest candidates the one closest to x is chosen.
static int shortest(double x, bool is_f32, char *d, int *e10) {
    char tmp[48];
    for (int p = 1; p <= 17; p++) {
        snprintf(tmp, sizeof tmp, "%.*e", p - 1, x);
        bool ok = is_f32 ? strtof(tmp, NULL) == (float)x : strtod(tmp, NULL) == x;
        if (!ok) {
            // The correctly rounded p-digit value can miss an asymmetric interval (x a
            // power of two); a neighbour on the wider side may still read back.
            char m[24];
            int k = 0;
            for (const char *s = tmp; *s && *s != 'e'; s++)
                if (*s >= '0' && *s <= '9') m[k++] = *s;
            m[k] = 0;
            int ex = atoi(strchr(tmp, 'e') + 1);
            for (int dir = 1; dir >= -1 && !ok; dir -= 2) {
                char c[24];
                memcpy(c, m, (size_t)k + 1);
                int ce = ex;
                int i = k - 1;
                if (dir > 0) {
                    while (i >= 0 && c[i] == '9') c[i--] = '0';
                    if (i < 0) { c[0] = '1'; ce++; } else c[i]++;
                } else {
                    while (i >= 0 && c[i] == '0') c[i--] = '9';
                    if (i < 0) continue;
                    c[i]--;
                    if (c[0] == '0') {
                        if (k == 1) continue;
                        memmove(c, c + 1, (size_t)k);
                        c[k - 1] = '9';
                        ce--;
                    }
                }
                char t2[48];
                snprintf(t2, sizeof t2, "%c.%se%d", c[0], c + 1, ce);
                ok = is_f32 ? strtof(t2, NULL) == (float)x : strtod(t2, NULL) == x;
                if (ok) memcpy(tmp, t2, strlen(t2) + 1);
            }
        }
        if (!ok) continue;
        int n = 0;
        const char *s = tmp;
        for (; *s && *s != 'e'; s++)
            if (*s >= '0' && *s <= '9') d[n++] = *s;
        *e10 = atoi(s + 1);
        while (n > 1 && d[n - 1] == '0') n--;
        d[n] = 0;
        return n;
    }
    return 0; // unreachable for finite x
}

static uint32_t dec_len(uint64_t v) {
    uint32_t n = 1;
    for (; v >= 10; v /= 10) n++;
    return n;
}

static bool is_pow10(uint64_t n) {
    for (; n != 0; n /= 10)
        if (n % 10 != 0) return false;
    return true;
}

static void write_dec(char *b, uint64_t v, size_t count) {
    for (size_t i = 0; i < count; i++) {
        b[count - i - 1] = (char)('0' + v % 10);
        v /= 10;
    }
}

// std.fmt.float decimal mode: the shortest digits, then round() (half up on those
// digits) and formatDecimal() when a precision is given (prec < 0: none).
static int fmt_float(char *out, size_t cap, double x, bool sign, bool is_f32, int prec_) {
    char b[400];
    size_t i = 0;
    bool has_prec = prec_ >= 0;
    size_t prec = has_prec ? (size_t)prec_ : 0;
    if (isnan(x) || isinf(x)) {
        if (sign) b[i++] = '-';
        memcpy(b + i, isnan(x) ? "nan" : "inf", 3);
        i += 3;
        goto done;
    }
    uint64_t mant = 0;
    int32_t exp = 0;
    if (x != 0) {
        char d[24];
        int e10;
        int n = shortest(fabs(x), is_f32, d, &e10);
        for (int k = 0; k < n; k++) mant = mant * 10 + (uint64_t)(d[k] - '0');
        exp = e10 - (n - 1);
    }
    uint32_t olength = dec_len(mant);
    if (has_prec) {
        size_t round_digit = 0;
        if (exp > 0) {
            round_digit = (olength - 1) + prec + (size_t)exp;
        } else {
            size_t min_exp = (size_t)(-(int64_t)exp);
            if (prec + olength > min_exp) round_digit = prec + olength - min_exp;
        }
        if (round_digit < olength) {
            for (size_t k = round_digit + 1; k < olength; k++) {
                mant /= 10;
                exp++;
            }
            if (mant % 10 >= 5) {
                mant = mant / 10 + 1;
                exp++;
                if (is_pow10(mant)) {
                    mant /= 10;
                    exp++;
                }
            }
        }
        olength = dec_len(mant);
    }
    // Zig renders into a 347-byte buffer and prints "(float)" when it does not fit.
    size_t aexp = (size_t)(exp < 0 ? -(int64_t)exp : exp);
    size_t req = exp >= 0 ? 2 + aexp + olength + prec : 2 + MAX(aexp + olength, prec);
    if (req > 347) {
        memcpy(b, "(float)", 7);
        i = 7;
        goto done;
    }
    if (sign) b[i++] = '-';
    int32_t dp_offset = exp + (int32_t)olength;
    if (dp_offset <= 0) {
        b[i++] = '0';
        b[i++] = '.';
        size_t dp_index = i;
        size_t z = (size_t)(-dp_offset);
        memset(b + i, '0', z);
        i += z;
        write_dec(b + i, mant, olength);
        i += olength;
        if (has_prec) {
            size_t written = i - dp_index;
            if (prec > written) memset(b + i, '0', prec - written);
            i = dp_index + prec - (prec == 0);
        }
    } else {
        size_t dp = (size_t)dp_offset;
        if (dp >= olength) {
            write_dec(b + i, mant, olength);
            i += olength;
            memset(b + i, '0', dp - olength);
            i += dp - olength;
            if (has_prec && prec != 0) {
                b[i++] = '.';
                memset(b + i, '0', prec);
                i += prec;
            }
        } else {
            uint64_t hi = mant, lo = mant;
            for (size_t k = 0; k < olength - dp; k++) hi /= 10;
            write_dec(b + i + dp + 1, lo, olength - dp);
            b[i + dp] = '.';
            size_t dp_index = i + dp + 1;
            write_dec(b + i, hi, dp);
            i += olength + 1;
            if (has_prec) {
                size_t written = olength - dp;
                if (prec > written) memset(b + i, '0', prec - written);
                i = dp_index + prec - (prec == 0);
            }
        }
    }
done:
    b[i] = 0;
    if (cap) {
        size_t m = i < cap - 1 ? i : cap - 1;
        memcpy(out, b, m);
        out[m] = 0;
    }
    return (int)i;
}

int fmt_f64(char *out, size_t cap, double x) { return fmt_float(out, cap, x, signbit(x) != 0, false, -1); }
int fmt_f32(char *out, size_t cap, float x) { return fmt_float(out, cap, (double)x, signbit(x) != 0, true, -1); }
int fmt_fixed(char *out, size_t cap, double x, int prec) {
    return fmt_float(out, cap, x, signbit(x) != 0, false, prec < 0 ? 0 : prec);
}
int fmt_fixed_f32(char *out, size_t cap, float x, int prec) {
    return fmt_float(out, cap, (double)x, signbit(x) != 0, true, prec < 0 ? 0 : prec);
}
