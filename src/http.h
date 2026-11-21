// HTTP/1.1 for the server (src/server.h): the parts of std.http.Server the engine used, one
// request per connection. A request's head and body are read under a stall guard (a socket
// call blocked too long drops the connection) and a whole-request deadline.
#ifndef SAETTA_HTTP_H
#define SAETTA_HTTP_H

#include "base.h"

typedef enum { M_GET, M_HEAD, M_POST, M_PUT, M_DELETE, M_CONNECT, M_OPTIONS, M_TRACE, M_PATCH, M_QUERY } Method;

#define HTTP_RBUF (1 << 16)
#define HTTP_WBUF (1 << 16)
// Head and body of a request must arrive within this time of the connection's start.
#define HTTP_REQUEST_DEADLINE_MS 120000u

typedef struct {
    int fd;
    char rbuf[HTTP_RBUF];
    size_t rpos, rend;
    // the request head, copied out of rbuf (which the body reads refill): Req's strings
    // point here and stay valid for the whole request
    char head[HTTP_RBUF];
    char wbuf[HTTP_WBUF];
    size_t wlen;
    bool wfail;
    uint64_t deadline_ns; // now_ns() by which reads must be done; 0: none
    bool timed_out;       // a read passed deadline_ns
} Conn;

typedef struct {
    Method method;
    Str target;
    bool has_expect;
    Str expect;
    bool has_length;
    uint64_t content_length;
    bool chunked;
} Req;

// The connection guard: a thread that shuts down the attached socket when a read has been
// blocked READ_LIMIT (30 s) or a write WRITE_LIMIT (60 s). Errors: "SystemResources".
Err http_guard_spawn(void);
void http_guard_attach(int fd);
void http_guard_detach(void); // before the socket closes

void conn_init(Conn *c, int fd); // no deadline

// Parses a request head (the bytes up to and including the empty line). Errors:
// "HttpHeadersInvalid", "UnknownHttpMethod", "HttpHeaderContinuationsUnsupported",
// "InvalidContentLength", "HttpTransferEncodingUnsupported", "MissingFinalNewline".
Err http_parse_head(Str bytes, Req *h);
// Reads and parses one request head (at most HTTP_RBUF bytes) into c->head, so h's strings
// survive the body reads. Errors: parse errors,
// "HttpHeadersOversize", "HttpConnectionClosing" (closed before any byte),
// "HttpRequestTruncated", "ReadFailed", "RequestTimeout".
Err http_receive_head(Conn *c, Req *h);
// The request body (content-length or chunked; none without either), at most limit bytes,
// appended to out. Errors: "BodyTooLarge", "HttpRequestTruncated" (end of stream or read
// error inside the body), "BadChunk" (a chunk size that is not hex, overflows, or a chunk
// not followed by CRLF), "RequestTimeout".
Err http_read_body(Conn *c, const Req *h, uint64_t limit, Buf *out);
// The status a request read error is answered with (400, 408, 413, 431), 0: none (the
// client is gone).
int http_error_status(Err e);

// Buffered writes; a failed send makes every later write fail ("WriteFailed").
Err conn_write(Conn *c, const void *p, size_t n);
Err conn_writes(Conn *c, const char *s);
Err conn_flush(Conn *c);
// True when the peer has closed the connection or it failed: one non-blocking recv peek.
// Limitation: a client that half-closes (shutdown(SHUT_WR)) after sending its request reads
// exactly like one that left (recv returns 0 either way), so it also counts as gone; the
// server peeks only for non-streamed requests.
bool conn_peer_gone(Conn *c);

// After an error response to a request whose body was not read: ends the write side and
// reads and drops up to max_bytes of what the client still sends, for at most ms, so the
// close does not reset the connection before the client reads the response.
void http_drain(Conn *c, size_t max_bytes, uint32_t ms);

// 100 Continue for a request that expects it. Errors: "HttpExpectationFailed" (another
// expectation), write errors.
Err http_write_expect_continue(Conn *c, Req *h);
// A whole response with connection: close and content-length, then a flush.
Err http_respond(Conn *c, Req *h, int status, Str body, const char *content_type);

#endif
