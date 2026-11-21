// Tests of the server's HTTP request reading (src/http.h) over a socketpair: the test writes
// a request into one end and reads it from the other.
#include "../http.h"
#include "../test.h"

#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>

typedef struct {
    int peer; // the client's end
    Conn *c;  // the server's end
} Pair;

// A connected pair with bytes already sent; close_after: the client closes its side.
static bool pair_open(Pair *p, Str bytes, bool close_after) {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) return false;
    int one = 1;
    setsockopt(sv[0], SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
    setsockopt(sv[1], SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
    // the socket buffer holds the small requests of these tests at once
    size_t off = 0;
    while (off < bytes.len) {
        ssize_t n = write(sv[0], bytes.ptr + off, bytes.len - off);
        if (n <= 0) return false;
        off += (size_t)n;
    }
    if (close_after) shutdown(sv[0], SHUT_WR);
    p->peer = sv[0];
    p->c = xmalloc(sizeof(Conn));
    conn_init(p->c, sv[1]);
    return true;
}

static void pair_close(Pair *p) {
    close(p->peer);
    close(p->c->fd);
    free(p->c);
}

// Head and body of a request; the body error (OK on success) in *body_err.
static Err read_request(Str bytes, bool close_after, uint64_t limit, Req *h, Buf *body, Err *body_err) {
    Pair p;
    if (!pair_open(&p, bytes, close_after)) return "SocketPair";
    Err e = http_receive_head(p.c, h);
    if (!e) *body_err = http_read_body(p.c, h, limit, body);
    pair_close(&p);
    return e;
}

TEST(http_content_length_body) {
    Req h;
    Buf b = {0};
    Err be = "unset";
    EXPECT_OK(read_request(S("POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n\r\nhello"), true, 1 << 20, &h,
                           &b, &be));
    EXPECT_OK(be);
    EXPECT_EQ(h.method, M_POST);
    EXPECT_STR(h.target, S("/v1/messages"));
    EXPECT_STR(buf_str(&b), S("hello"));
    buf_free(&b);
}

TEST(http_chunked_body) {
    Req h;
    Buf b = {0};
    Err be = "unset";
    EXPECT_OK(read_request(S("POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"
                             "5\r\nhello\r\nA;ext=1\r\n, chunked!\r\n6 \r\n world\r\n0\r\nTrailer: t\r\n\r\n"),
                           true, 1 << 20, &h, &b, &be));
    EXPECT_OK(be);
    EXPECT(h.chunked);
    EXPECT_STR(buf_str(&b), S("hello, chunked! world"));
    buf_free(&b);
}

TEST(http_chunked_rejects_bad_sizes) {
    static const char *const bad[] = {
        "5x\r\nhello\r\n0\r\n\r\n",                  // a non-hex character after the digits
        "g\r\nhello\r\n0\r\n\r\n",                   // 'g' is not a hex digit
        "\r\nhello\r\n0\r\n\r\n",                    // no digits
        "Z\r\nhello\r\n0\r\n\r\n",                   // the old parser took G-Z as digits
        "10000000000000000\r\nx\r\n0\r\n\r\n",       // 17 hex digits overflow u64
        "5\r\nhelloXY0\r\n\r\n",                     // no CRLF after the data
    };
    for (size_t i = 0; i < COUNT(bad); i++) {
        Req h;
        Buf b = {0};
        Err be = OK;
        Buf req = {0};
        buf_adds(&req, "POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n");
        buf_adds(&req, bad[i]);
        EXPECT_OK(read_request(buf_str(&req), true, 1 << 20, &h, &b, &be));
        buf_free(&req);
        buf_free(&b);
        if (!be || strcmp(be, "BadChunk") != 0) dbg("chunk case %zu: %s\n", i, be ? be : "success");
        EXPECT_ERR(be, "BadChunk");
        EXPECT_EQ(http_error_status(be), 400);
    }
}

TEST(http_oversize_and_truncated_bodies) {
    Req h;
    Buf b = {0};
    Err be = OK;
    EXPECT_OK(read_request(S("POST / HTTP/1.1\r\nContent-Length: 100\r\n\r\nabc"), true, 10, &h, &b, &be));
    EXPECT_ERR(be, "BodyTooLarge");
    EXPECT_EQ(http_error_status(be), 413);
    b.len = 0;
    EXPECT_OK(read_request(S("POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n8\r\n12345678\r\n8\r\n12345678\r\n0\r\n\r\n"),
                           true, 10, &h, &b, &be));
    EXPECT_ERR(be, "BodyTooLarge");
    b.len = 0;
    EXPECT_OK(read_request(S("POST / HTTP/1.1\r\nContent-Length: 100\r\n\r\nabc"), true, 1 << 20, &h, &b, &be));
    EXPECT_ERR(be, "HttpRequestTruncated");
    EXPECT_EQ(http_error_status(be), 400);
    b.len = 0;
    EXPECT_OK(read_request(S("POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n8\r\n1234"), true, 1 << 20, &h, &b, &be));
    EXPECT_ERR(be, "HttpRequestTruncated");
    buf_free(&b);
}

TEST(http_bad_heads) {
    Req h;
    Buf b = {0};
    Err be = OK;
    EXPECT_ERR(read_request(S("FETCH / HTTP/1.1\r\n\r\n"), true, 10, &h, &b, &be), "UnknownHttpMethod");
    EXPECT_ERR(read_request(S("GET / HTTP/2.0\r\n\r\n"), true, 10, &h, &b, &be), "HttpHeadersInvalid");
    EXPECT_ERR(read_request(S("GET / HTTP/1.1\r\nContent-Length: x\r\n\r\n"), true, 10, &h, &b, &be), "InvalidContentLength");
    EXPECT_ERR(read_request(S("GET / HTTP/1.1\r\nHost"), true, 10, &h, &b, &be), "HttpRequestTruncated");
    EXPECT_ERR(read_request(S(""), true, 10, &h, &b, &be), "HttpConnectionClosing");
    EXPECT_EQ(http_error_status("HttpConnectionClosing"), 0);
    EXPECT_EQ(http_error_status("HttpHeadersOversize"), 431);
    buf_free(&b);
}

// A client that sends part of its request and then nothing is cut off at the deadline.
TEST(http_request_deadline) {
    Pair p;
    EXPECT(pair_open(&p, S("POST / HTTP/1.1\r\nContent-Length: 10\r\n\r\nabc"), false));
    p.c->deadline_ns = now_ns() + 200ull * 1000000ull;
    Req h;
    Err e = http_receive_head(p.c, &h);
    Buf b = {0};
    Err be = e ? e : http_read_body(p.c, &h, 1 << 20, &b);
    double waited = (double)(now_ns() - (p.c->deadline_ns - 200ull * 1000000ull)) / 1e9;
    EXPECT(!conn_peer_gone(p.c)); // the client is still there
    pair_close(&p);
    buf_free(&b);
    EXPECT_OK(e);
    EXPECT_ERR(be, "RequestTimeout");
    EXPECT_EQ(http_error_status(be), 408);
    EXPECT(waited < 5.0);
}

TEST(http_peer_gone) {
    Pair p;
    EXPECT(pair_open(&p, S(""), false));
    EXPECT(!conn_peer_gone(p.c));
    close(p.peer);
    p.peer = -1;
    EXPECT(conn_peer_gone(p.c));
    close(p.c->fd);
    free(p.c);
}

static bool send_all(int fd, Str s) {
    size_t off = 0;
    while (off < s.len) {
        ssize_t n = write(fd, s.ptr + off, s.len - off);
        if (n <= 0) return false;
        off += (size_t)n;
    }
    return true;
}

// The body arrives after the head was read, and a body larger than the read buffer: the
// target and the Expect value parsed from the head survive the body reads, which refill
// the read buffer from its start.
TEST(http_head_survives_the_body_read) {
    for (int big = 0; big < 2; big++) {
        size_t blen = big ? 3 * HTTP_RBUF + 17 : 40;
        Buf req = {0};
        buf_printf(&req, "POST /v1/messages/count_tokens HTTP/1.1\r\nExpect: 100-continue\r\nContent-Length: %zu\r\n\r\n", blen);
        Pair p;
        EXPECT(pair_open(&p, buf_str(&req), false));
        buf_free(&req);
        Req h;
        Err e = http_receive_head(p.c, &h);
        EXPECT_OK(e);
        Buf body = {0}, b = {0};
        for (size_t i = 0; i < blen; i++) buf_addc(&body, (char)('a' + i % 26));
        // a writer thread is not needed for the small body; the big one is written in
        // pieces between reads, as the socket buffer is smaller than the body
        Err be = OK;
        if (!big) {
            EXPECT(send_all(p.peer, buf_str(&body)));
            be = http_read_body(p.c, &h, 1 << 26, &b);
        } else {
            Req hh = h;
            size_t sent = 0;
            hh.content_length = 0;
            while (sent < blen && !be) {
                size_t k = MIN((size_t)8192, blen - sent);
                EXPECT(send_all(p.peer, str_sub(buf_str(&body), sent, sent + k)));
                hh.content_length = k;
                be = http_read_body(p.c, &hh, 1 << 26, &b);
                sent += k;
            }
        }
        EXPECT_OK(be);
        EXPECT_STR(buf_str(&b), buf_str(&body));
        EXPECT_STR(h.target, S("/v1/messages/count_tokens"));
        EXPECT_STR(h.expect, S("100-continue"));
        buf_free(&body);
        buf_free(&b);
        pair_close(&p);
    }
}
