// HTTP/1.1 over a Unix domain socket (see unix_http.h): one request per connection.
#include "unix_http.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

static int fail(char *err, size_t cap, const char *what) {
    snprintf(err, cap, "%s", what);
    return 1;
}

// The request out on the non-blocking socket: waits for room 100 ms at a time, as the reads wait for
// bytes, so a server that does not read is given up on Esc (*cancel) or after the idle timeout.
// 0 sent, else the error.
static const char *send_all(int fd, const void *p, size_t n, int limit_ms, volatile int *cancel) {
    int idle = 0;
    for (size_t w = 0; w < n;) {
        if (cancel && *cancel) return "cancelled";
        ssize_t x = send(fd, (const char *)p + w, n - w, 0);
        if (x > 0) {
            w += (size_t)x;
            idle = 0;
            continue;
        }
        if (x < 0 && errno == EINTR) continue;
        if (x < 0 && errno != EAGAIN) return "the connection was lost while sending the request";
        struct pollfd pf = {fd, POLLOUT, 0};
        int pr = poll(&pf, 1, 100);
        if (pr == 0 && limit_ms >= 0 && (idle += 100) >= limit_ms) return "The request timed out.";
    }
    return NULL;
}

typedef struct {
    sx_http_sink on_data;
    void *ctx;
    int status;
    bool chunked, stop, bad, ended;
    int64_t length; // Content-Length of a body that is not chunked, -1: none (to the close)
    // chunked: bytes left in the current chunk; -1: a size line is expected; -2: its CRLF after
    int64_t left;
    Buf line;
} Body;

// Body bytes: straight to the sink, or through the chunk framing. False once the sink stops it or
// the framing ends (the last chunk).
static bool feed(Body *b, const char *p, size_t n) {
    if (!b->chunked) {
        if (b->length >= 0) n = (size_t)MIN((int64_t)n, b->length); // nothing past the length
        if (n && b->on_data(b->ctx, b->status, p, n)) b->stop = true;
        if (b->length >= 0 && (b->length -= (int64_t)n) == 0) b->ended = true;
        return !b->stop && !b->ended;
    }
    while (n && !b->stop) {
        if (b->left > 0) {
            size_t k = (size_t)MIN((int64_t)n, b->left);
            if (b->on_data(b->ctx, b->status, p, k)) b->stop = true;
            p += k, n -= k, b->left -= (int64_t)k;
            if (!b->left) b->left = -2;
            continue;
        }
        // a size line, or the CRLF after a chunk's data
        char c = *p++;
        n--;
        if (b->left == -2) { // exactly CRLF after the data
            if (c == '\r' && !b->line.len) {
                buf_addc(&b->line, c);
                continue;
            }
            if (c == '\n' && b->line.len == 1) {
                b->left = -1;
                b->line.len = 0;
                continue;
            }
            b->bad = true;
            return false;
        }
        if (c != '\n') {
            if (b->line.len >= 8192) { // a size line that never ends
                b->bad = true;
                return false;
            }
            buf_addc(&b->line, c);
            continue;
        }
        // hex digits, then optional extensions (";..."), then CR
        size_t len = b->line.len;
        if (len && b->line.ptr[len - 1] == '\r') len--;
        size_t k = 0;
        unsigned long long size = 0;
        while (k < len && k < 16 && isxdigit((unsigned char)b->line.ptr[k])) {
            char d = b->line.ptr[k++];
            size = size * 16 + (unsigned)(d <= '9' ? d - '0' : (d | 0x20) - 'a' + 10);
        }
        bool ok = k > 0 && (k == len || b->line.ptr[k] == ';' || b->line.ptr[k] == ' ' || b->line.ptr[k] == '\t');
        b->line.len = 0;
        if (!ok) {
            b->bad = true;
            return false;
        }
        if (size == 0) { // the last chunk (its trailers are not read)
            b->ended = true;
            return false;
        }
        b->left = (int64_t)size;
    }
    return !b->stop;
}

int unix_http(const char *sock, const char *method, const char *target, const sx_header *headers, size_t nheaders,
              const void *body, size_t body_len, double timeout_s, sx_http_sink on_data, void *ctx, int *status, char *err,
              size_t err_cap, volatile int *cancel) {
    struct sockaddr_un sa = {.sun_family = AF_UNIX};
    if (strlen(sock) >= sizeof sa.sun_path) return fail(err, err_cap, "socket path too long");
    snprintf(sa.sun_path, sizeof sa.sun_path, "%s", sock);
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return fail(err, err_cap, strerror(errno));
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
    if (connect(fd, (struct sockaddr *)&sa, sizeof sa) != 0) {
        int e = errno;
        close(fd);
        return fail(err, err_cap, strerror(e));
    }
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK); // sends and reads wait in poll, cancellable
    Buf req = {0};
    buf_printf(&req, "%s %s HTTP/1.1\r\nHost: saetta\r\nConnection: close\r\n", method, target);
    for (size_t i = 0; i < nheaders; i++) buf_printf(&req, "%s: %s\r\n", headers[i].name, headers[i].value);
    if (body) buf_printf(&req, "Content-Length: %zu\r\n", body_len);
    buf_adds(&req, "\r\n");
    int limit_ms = timeout_s > 0 ? (int)(timeout_s * 1000) : -1;
    const char *se = send_all(fd, req.ptr, req.len, limit_ms, cancel);
    if (!se && body_len) se = send_all(fd, body, body_len, limit_ms, cancel);
    buf_free(&req);
    if (se) {
        close(fd);
        return fail(err, err_cap, se);
    }
    Buf head = {0};
    Body b = {.on_data = on_data, .ctx = ctx, .left = -1, .length = -1};
    bool in_body = false, done = false;
    int rc = 0, idle_ms = 0;
    char buf[65536];
    while (!done) {
        if (cancel && *cancel) {
            rc = fail(err, err_cap, "cancelled");
            break;
        }
        struct pollfd pf = {fd, POLLIN, 0};
        int pr = poll(&pf, 1, 100);
        if (pr < 0 && errno == EINTR) continue;
        if (pr == 0) {
            idle_ms += 100;
            if (limit_ms >= 0 && idle_ms >= limit_ms) {
                rc = fail(err, err_cap, "The request timed out.");
                break;
            }
            continue;
        }
        idle_ms = 0;
        ssize_t n = recv(fd, buf, sizeof buf, 0);
        if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
        if (n < 0) {
            rc = fail(err, err_cap, strerror(errno));
            break;
        }
        if (n == 0) {
            // the end: a whole body ends here (connection: close); a head or a chunked body cut here
            // did not arrive whole
            if (!in_body) rc = fail(err, err_cap, "the server closed the connection before answering");
            else if (b.chunked || b.length > 0) rc = fail(err, err_cap, "the connection was lost during the response");
            break;
        }
        if (in_body) {
            if (!feed(&b, buf, (size_t)n)) done = true;
            if (b.bad) rc = fail(err, err_cap, "a malformed chunked response");
            continue;
        }
        buf_add(&head, buf, (size_t)n);
        ptrdiff_t e = str_find(buf_str(&head), S("\r\n\r\n"), 0);
        if (e < 0) {
            if (head.len > 65536) {
                rc = fail(err, err_cap, "the response head is too long");
                break;
            }
            continue;
        }
        Str h = str_sub(buf_str(&head), 0, (size_t)e + 2);
        if (!str_starts(h, S("HTTP/1."))) {
            rc = fail(err, err_cap, "not an HTTP response");
            break;
        }
        b.status = atoi(head.ptr + 9);
        *status = b.status;
        for (size_t i = 0; i < h.len;) {
            ptrdiff_t nl = str_find_char(h, '\n', i);
            size_t j = nl >= 0 ? (size_t)nl : h.len;
            Str l = str_sub(h, i, j);
            if (l.len > 18 && !strncasecmp(l.ptr, "transfer-encoding:", 18)) {
                for (size_t q = 18; q + 7 <= l.len && !b.chunked; q++) b.chunked = !strncasecmp(l.ptr + q, "chunked", 7);
            } else if (l.len > 15 && !strncasecmp(l.ptr, "content-length:", 15)) b.length = strtoll(l.ptr + 15, NULL, 10);
            i = j + 1;
        }
        in_body = true;
        if (b.chunked) b.length = -1;
        if (b.length == 0) b.ended = done = true;
        size_t rest = head.len - ((size_t)e + 4);
        if (rest && !done && !feed(&b, head.ptr + e + 4, rest)) done = true;
        if (b.bad) rc = fail(err, err_cap, "a malformed chunked response");
    }
    if (!rc && b.stop) rc = fail(err, err_cap, "cancelled");
    buf_free(&head);
    buf_free(&b.line);
    close(fd);
    return rc;
}
