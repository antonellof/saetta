// HTTP/1.1 for the server (see http.h).
#include "http.h"

#include <errno.h>
#include <poll.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sys/socket.h>
#include <time.h>

// ---- the connection guard --------------------------------------------------------------------

// Drops a connection whose socket read or write has been blocked too long. The server serves
// one connection at a time, so a client that connects and sends nothing, or stops reading its
// stream, would otherwise hold up every other request forever. Each socket call counts seq
// up on entry and on exit (odd: inside a call); the guard thread samples it, and when it stays
// odd and unchanged past the limit it shuts the socket down, so the blocked call returns
// end of stream or a write error and the connection ends normally.
#define READ_LIMIT_MS 30000u  // head and body of a request: a client sends them at once
#define WRITE_LIMIT_MS 60000u // a streamed answer the client does not read
#define TICK_MS 500u

static struct {
    _Atomic uint64_t seq;
    _Atomic uint32_t limit_ms;
    _Atomic int32_t fd;
    _Atomic bool busy;
} guard = {0, READ_LIMIT_MS, -1, false};

static void guard_enter(uint32_t limit_ms) {
    atomic_store_explicit(&guard.limit_ms, limit_ms, memory_order_release);
    atomic_fetch_add_explicit(&guard.seq, 1, memory_order_acq_rel);
}
static void guard_leave(void) { atomic_fetch_add_explicit(&guard.seq, 1, memory_order_acq_rel); }

void http_guard_attach(int fd) { atomic_store(&guard.fd, fd); }

// Before the socket closes: the guard is not shutting down this descriptor, and never
// will (the number may be reused by the next accept).
void http_guard_detach(void) {
    atomic_store(&guard.fd, -1);
    while (atomic_load(&guard.busy)) __builtin_arm_yield();
}

static void *guard_loop(void *arg) {
    (void)arg;
    uint64_t last = 0;
    uint32_t stalled_ms = 0;
    for (;;) {
        struct timespec ts = {0, (long)TICK_MS * 1000000L};
        nanosleep(&ts, NULL);
        uint64_t s = atomic_load_explicit(&guard.seq, memory_order_acquire);
        stalled_ms = ((s & 1) == 1 && s == last) ? stalled_ms + TICK_MS : 0;
        last = s;
        uint32_t limit = atomic_load_explicit(&guard.limit_ms, memory_order_acquire);
        if (stalled_ms < limit) continue;
        stalled_ms = 0;
        // busy before fd: detach either sees busy and waits, or has cleared fd first
        atomic_store(&guard.busy, true);
        int fd = atomic_load(&guard.fd);
        if (fd >= 0 && atomic_load_explicit(&guard.seq, memory_order_acquire) == s) {
            shutdown(fd, SHUT_RDWR);
            dbg("connection: no %s for %u s, dropped\n", limit == READ_LIMIT_MS ? "request" : "reader", limit / 1000);
        }
        atomic_store(&guard.busy, false);
    }
    return NULL;
}

Err http_guard_spawn(void) {
    pthread_t th;
    if (pthread_create(&th, NULL, guard_loop, NULL) != 0) return "SystemResources";
    pthread_detach(th);
    return OK;
}

// ---- connection I/O ----------------------------------------------------------------------------

static const char *const method_names[] = {"GET", "HEAD", "POST", "PUT", "DELETE", "CONNECT", "OPTIONS", "TRACE", "PATCH", "QUERY"};

void conn_init(Conn *c, int fd) {
    c->fd = fd;
    c->rpos = c->rend = c->wlen = 0;
    c->wfail = false;
    c->deadline_ns = 0;
    c->timed_out = false;
}

// Reads more bytes into rbuf (compacting it first when full). 1: read, 0: end of stream,
// -1: read error or the deadline passed (timed_out set).
static int conn_fill(Conn *c) {
    if (c->rend == HTTP_RBUF) {
        memmove(c->rbuf, c->rbuf + c->rpos, c->rend - c->rpos);
        c->rend -= c->rpos;
        c->rpos = 0;
        if (c->rend == HTTP_RBUF) return -1;
    }
    for (;;) {
        if (c->deadline_ns) {
            // wait for bytes no later than the deadline (the guard still bounds the recv)
            uint64_t now = now_ns();
            if (now >= c->deadline_ns) {
                c->timed_out = true;
                return -1;
            }
            struct pollfd p = {c->fd, POLLIN, 0};
            int ms = (int)MIN((c->deadline_ns - now + 999999) / 1000000, (uint64_t)INT32_MAX);
            int r = poll(&p, 1, ms);
            if (r < 0 && errno == EINTR) continue;
            if (r < 0) return -1;
            if (r == 0) {
                c->timed_out = true;
                return -1;
            }
        }
        guard_enter(READ_LIMIT_MS);
        ssize_t n = recv(c->fd, c->rbuf + c->rend, HTTP_RBUF - c->rend, 0);
        guard_leave();
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) return -1;
        if (n == 0) return 0;
        c->rend += (size_t)n;
        return 1;
    }
}

Err conn_flush(Conn *c) {
    if (c->wfail) return "WriteFailed";
    size_t off = 0;
    while (off < c->wlen) {
        guard_enter(WRITE_LIMIT_MS);
        ssize_t n = send(c->fd, c->wbuf + off, c->wlen - off, 0);
        guard_leave();
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            c->wfail = true;
            c->wlen = 0;
            return "WriteFailed";
        }
        off += (size_t)n;
    }
    c->wlen = 0;
    return OK;
}

Err conn_write(Conn *c, const void *p, size_t n) {
    const char *s = p;
    while (n > 0) {
        if (c->wlen == HTTP_WBUF) TRY(conn_flush(c));
        size_t k = MIN(n, HTTP_WBUF - c->wlen);
        memcpy(c->wbuf + c->wlen, s, k);
        c->wlen += k;
        s += k;
        n -= k;
    }
    return c->wfail ? "WriteFailed" : OK;
}
Err conn_writes(Conn *c, const char *s) { return conn_write(c, s, strlen(s)); }

bool conn_peer_gone(Conn *c) {
    if (c->wfail) return true;
    char b;
    for (;;) {
        ssize_t n = recv(c->fd, &b, 1, MSG_PEEK | MSG_DONTWAIT);
        if (n > 0) return false;
        if (n == 0) return true;
        if (errno == EINTR) continue;
        return errno != EAGAIN && errno != EWOULDBLOCK;
    }
}

// ---- request head ------------------------------------------------------------------------------

static bool ieq(Str a, const char *b) {
    size_t n = strlen(b);
    if (a.len != n) return false;
    for (size_t i = 0; i < n; i++) {
        char x = a.ptr[i], y = b[i];
        if (x >= 'A' && x <= 'Z') x = (char)(x + 32);
        if (y >= 'A' && y <= 'Z') y = (char)(y + 32);
        if (x != y) return false;
    }
    return true;
}

static Str trim_set(Str s, const char *set) {
    size_t b = 0, e = s.len;
    while (b < e && strchr(set, s.ptr[b])) b++;
    while (e > b && strchr(set, s.ptr[e - 1])) e--;
    return (Str){s.ptr + b, e - b};
}

static bool content_encoding_known(Str s) {
    static const char *const names[] = {"zstd", "gzip", "x-gzip", "deflate", "compress", "x-compress", "identity"};
    for (size_t i = 0; i < COUNT(names); i++)
        if (str_eqz(s, names[i])) return true;
    return false;
}

// std.http.Server.Request.Head.parse.
Err http_parse_head(Str bytes, Req *h) {
    memset(h, 0, sizeof *h);
    size_t pos = 0;
    ptrdiff_t eol = str_find(bytes, S("\r\n"), 0);
    Str first = eol < 0 ? bytes : str_sub(bytes, 0, (size_t)eol);
    pos = eol < 0 ? bytes.len + 2 : (size_t)eol + 2;
    if (first.len < 10) return "HttpHeadersInvalid";
    ptrdiff_t method_end = str_find_char(first, ' ', 0);
    if (method_end < 0) return "HttpHeadersInvalid";
    Str mname = str_sub(first, 0, (size_t)method_end);
    bool found = false;
    for (size_t i = 0; i < COUNT(method_names); i++)
        if (str_eqz(mname, method_names[i])) {
            h->method = (Method)i;
            found = true;
        }
    if (!found) return "UnknownHttpMethod";
    ptrdiff_t version_start = -1;
    for (size_t i = first.len; i > 0; i--)
        if (first.ptr[i - 1] == ' ') {
            version_start = (ptrdiff_t)i - 1;
            break;
        }
    if (version_start == method_end) return "HttpHeadersInvalid";
    Str version = str_sub(first, (size_t)version_start + 1, first.len);
    if (version.len != 8) return "HttpHeadersInvalid";
    if (!str_eqz(version, "HTTP/1.0") && !str_eqz(version, "HTTP/1.1")) return "HttpHeadersInvalid";
    h->target = str_sub(first, (size_t)method_end + 1, (size_t)version_start);
    bool has_compression = false;
    while (pos <= bytes.len) {
        ptrdiff_t e = str_find(bytes, S("\r\n"), pos);
        Str line = e < 0 ? str_sub(bytes, pos, bytes.len) : str_sub(bytes, pos, (size_t)e);
        pos = e < 0 ? bytes.len + 1 : (size_t)e + 2;
        if (line.len == 0) return OK;
        if (line.ptr[0] == ' ' || line.ptr[0] == '\t') return "HttpHeaderContinuationsUnsupported";
        ptrdiff_t colon = str_find_char(line, ':', 0);
        Str name = colon < 0 ? line : str_sub(line, 0, (size_t)colon);
        Str value = colon < 0 ? S("") : trim_set(str_sub(line, (size_t)colon + 1, line.len), " \t");
        if (name.len == 0) return "HttpHeadersInvalid";
        if (ieq(name, "expect")) {
            h->has_expect = true;
            h->expect = value;
        } else if (ieq(name, "content-length")) {
            if (h->has_length) return "HttpHeadersInvalid";
            if (parse_u64(value, &h->content_length)) return "InvalidContentLength";
            h->has_length = true;
        } else if (ieq(name, "content-encoding")) {
            if (has_compression) return "HttpHeadersInvalid";
            Str t = trim_set(value, " ");
            if (!content_encoding_known(t)) return "HttpTransferEncodingUnsupported";
            has_compression = !str_eqz(t, "identity");
        } else if (ieq(name, "transfer-encoding")) {
            // Transfer-Encoding: second, first (read from the end)
            Str seg[3];
            size_t nseg = 0, end = value.len;
            for (size_t i = value.len + 1; i > 0 && nseg < 3; i--) {
                if (i - 1 == 0 || value.ptr[i - 2] == ',') {
                    seg[nseg++] = str_sub(value, i - 1, end);
                    end = i >= 2 ? i - 2 : 0;
                }
            }
            size_t si = 0;
            Str first = trim_set(seg[si], " ");
            bool has_next = true;
            if (str_eqz(first, "chunked") || str_eqz(first, "none")) {
                if (h->chunked) return "HttpHeadersInvalid";
                h->chunked = str_eqz(first, "chunked");
                si++;
                has_next = si < nseg;
            }
            if (has_next) {
                Str t = trim_set(seg[si], " ");
                if (!content_encoding_known(t)) return "HttpTransferEncodingUnsupported";
                if (has_compression) return "HttpHeadersInvalid";
                has_compression = !str_eqz(t, "identity");
                si++;
            }
            if (si < nseg) return "HttpTransferEncodingUnsupported";
        }
    }
    return "MissingFinalNewline";
}

// std.http.Server.receiveHead: up to HTTP_RBUF bytes, ending at an empty line.
Err http_receive_head(Conn *c, Req *h) {
    for (;;) {
        Str buf = {c->rbuf + c->rpos, c->rend - c->rpos};
        ptrdiff_t a = str_find(buf, S("\r\n\r\n"), 0), b = str_find(buf, S("\n\n"), 0);
        size_t end = 0;
        if (a >= 0 && (b < 0 || a + 2 <= b)) end = (size_t)a + 4;
        else if (b >= 0) end = (size_t)b + 2;
        if (end > 0) {
            memcpy(c->head, buf.ptr, end);
            c->rpos += end;
            return http_parse_head((Str){c->head, end}, h);
        }
        if (buf.len >= HTTP_RBUF) return "HttpHeadersOversize";
        int r = conn_fill(c);
        if (r == 0) return buf.len == 0 ? "HttpConnectionClosing" : "HttpRequestTruncated";
        if (r < 0) return c->timed_out ? "RequestTimeout" : "ReadFailed";
    }
}

// ---- request body ------------------------------------------------------------------------------

static Err short_read(const Conn *c) { return c->timed_out ? "RequestTimeout" : "HttpRequestTruncated"; }

// One byte of the body stream; false at end of stream or on a read error.
static bool conn_byte(Conn *c, char *out) {
    if (c->rpos == c->rend) {
        c->rpos = c->rend = 0;
        if (conn_fill(c) <= 0) return false;
    }
    *out = c->rbuf[c->rpos++];
    return true;
}

static bool conn_take(Conn *c, Buf *out, uint64_t n) {
    while (n > 0) {
        if (c->rpos == c->rend) {
            c->rpos = c->rend = 0;
            if (conn_fill(c) <= 0) return false;
        }
        size_t k = (size_t)MIN((uint64_t)(c->rend - c->rpos), n);
        buf_add(out, c->rbuf + c->rpos, k);
        c->rpos += k;
        n -= k;
    }
    return true;
}

static int hex_digit(char ch) {
    if (ch >= '0' && ch <= '9') return ch - '0';
    if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
    if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
    return -1;
}

// A chunk head: hex size (at least one digit), optional whitespace, an optional extension
// from ';', then CRLF (or LF).
static Err chunk_head(Conn *c, uint64_t *len) {
    uint64_t n = 0;
    size_t digits = 0;
    char ch;
    enum { SIZE, WS, EXT, CR } state = SIZE;
    for (;;) {
        if (!conn_byte(c, &ch)) return short_read(c);
        if (state == SIZE) {
            int d = hex_digit(ch);
            if (d >= 0) {
                if (n > (UINT64_MAX >> 4)) return "BadChunk";
                n = n * 16 + (uint64_t)d;
                digits++;
                continue;
            }
            if (digits == 0) return "BadChunk";
            state = WS;
        }
        if (state == WS) {
            if (ch == ' ' || ch == '\t') continue;
            if (ch == ';') state = EXT;
            else if (ch == '\r') state = CR;
            else if (ch == '\n') break;
            else return "BadChunk";
        } else if (state == EXT) {
            if (ch == '\r') state = CR;
            else if (ch == '\n') break;
        } else {
            if (ch != '\n') return "BadChunk";
            break;
        }
    }
    *len = n;
    return OK;
}

Err http_read_body(Conn *c, const Req *h, uint64_t limit, Buf *out) {
    if (h->method == M_HEAD) return OK;
    if (h->chunked) {
        for (;;) {
            uint64_t len;
            TRY(chunk_head(c, &len));
            char ch;
            if (len == 0) {
                // trailers until an empty line
                size_t line = 0;
                for (;;) {
                    if (!conn_byte(c, &ch)) return short_read(c);
                    if (ch == '\n') {
                        if (line == 0) return OK;
                        line = 0;
                    } else if (ch != '\r') line++;
                }
            }
            if (len > limit || out->len > limit - len) return "BodyTooLarge"; // no wrap on a huge chunk size
            if (!conn_take(c, out, len)) return short_read(c);
            char r1, r2;
            if (!conn_byte(c, &r1) || !conn_byte(c, &r2)) return short_read(c);
            if (r1 != '\r' || r2 != '\n') return "BadChunk";
        }
    }
    if (!h->has_length) return OK;
    if (h->content_length > limit) return "BodyTooLarge";
    return conn_take(c, out, h->content_length) ? OK : short_read(c);
}

int http_error_status(Err e) {
    if (!e) return 0;
    if (!strcmp(e, "HttpConnectionClosing") || !strcmp(e, "ReadFailed") || !strcmp(e, "WriteFailed")) return 0;
    if (!strcmp(e, "BodyTooLarge")) return 413;
    if (!strcmp(e, "RequestTimeout")) return 408;
    if (!strcmp(e, "HttpHeadersOversize")) return 431;
    return 400;
}

// ---- responses ---------------------------------------------------------------------------------

Err http_write_expect_continue(Conn *c, Req *h) {
    if (!h->has_expect) return OK;
    if (!str_eqz(h->expect, "100-continue")) return "HttpExpectationFailed";
    TRY(conn_writes(c, "HTTP/1.1 100 Continue\r\n\r\n"));
    h->has_expect = false;
    return OK;
}

static const char *phrase(int status) {
    switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 408: return "Request Timeout";
    case 413: return "Content Too Large";
    case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error";
    default: return "";
    }
}

// Request.respond with keep_alive false: the whole response, then a flush.
Err http_respond(Conn *c, Req *h, int status, Str body, const char *content_type) {
    TRY(http_write_expect_continue(c, h));
    char head[256];
    int n = snprintf(head, sizeof head, "HTTP/1.1 %d %s\r\nconnection: close\r\ncontent-length: %zu\r\n", status,
                     phrase(status), body.len);
    TRY(conn_write(c, head, (size_t)n));
    if (content_type) {
        TRY(conn_writes(c, "content-type: "));
        TRY(conn_writes(c, content_type));
        TRY(conn_writes(c, "\r\n"));
    }
    TRY(conn_writes(c, "\r\n"));
    if (h->method != M_HEAD && body.len > 0) TRY(conn_write(c, body.ptr, body.len));
    return conn_flush(c);
}
