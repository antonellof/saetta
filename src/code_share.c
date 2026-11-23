// One model for several Saetta Codes (see code_share.h). Saetta's own: pi runs no model.
#include "code_share.h"
#include "hash.h"
#include "instance.h"
#include "server.h"
#include "turns.h"
#include "unix_http.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <limits.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

// Connections served at once (each on its own thread; one runs on the engine at a time, the
// others read their request or wait for their turn with keep-alives): room for 6 sessions with 3
// subagents each and more. Only this user can connect (the socket is 0600 in a 0700 folder).
#define MAX_CONNS 32

static struct {
    bool serving, joined, priv;
    int ls;                // the listening socket
    int wake[2];           // stops the accept thread
    pthread_t accept_th;
    struct Engine *engine; // NULL with the stub
    uint16_t stub_port;    // the stub's upstream
    char token[80];        // "Bearer <token>" (the stub checks it itself)
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int active;            // connection threads running
    int fds[MAX_CONNS];    // their sockets (-1: free), shut down on a forced stop
    char sock[PATH_MAX];   // the socket this holder listens on (unlinked at the end)
    void (*notice)(const char *msg); // while the screen is up (code_share_set_notice); else stderr
    char holder[40];       // the joined holder's identity, and its token
    char jtoken[80];
    char jsock[PATH_MAX];  // the joined holder's socket
    char dir[PATH_MAX];    // the pack, for a joined session that takes over
} S = {.ls = -1, .wake = {-1, -1}, .mu = PTHREAD_MUTEX_INITIALIZER, .cv = PTHREAD_COND_INITIALIZER};

static volatile sig_atomic_t g_stop; // Ctrl-C while the holder serves the others at its exit

void code_share_describe(Arena *a, CodeModel *m, const char *id, size_t ctx, const char *base_url) {
    m->provider = "saetta";
    m->sock = NULL; // a fresh description: not the socket or the check of a model joined before
    m->verify = NULL;
    m->id = id;
    m->name = id;
    m->base_url = base_url;
    m->api_key = "saetta";
    m->reasoning = true;
    m->context_window = ctx;
    m->max_tokens = 32768;
    m->supports_developer_role = m->supports_store = m->supports_reasoning_effort = false;
    m->max_tokens_field = "max_tokens";
    m->thinking_format = "qwen-chat-template";
    JValue *compat = json_new(a, J_OBJECT);
    json_put(a, compat, S("supportsDeveloperRole"), json_bool(a, false));
    json_put(a, compat, S("supportsStore"), json_bool(a, false));
    json_put(a, compat, S("supportsReasoningEffort"), json_bool(a, false));
    json_put(a, compat, S("maxTokensField"), json_string(a, S("max_tokens")));
    json_put(a, compat, S("thinkingFormat"), json_string(a, S("qwen-chat-template")));
    m->compat = compat;
}

void code_share_set_notice(void (*fn)(const char *msg)) { S.notice = fn; }

static void (*g_wait_begin)(void), (*g_wait_end)(void);
void code_share_set_wait_hooks(void (*begin)(void), void (*end)(void)) {
    g_wait_begin = begin;
    g_wait_end = end;
}

// A message for the user: the screen's when it is up (never stderr under it), else stderr.
static void say(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void say(const char *fmt, ...) {
    char b[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    if (S.notice) S.notice(b);
    else fprintf(stderr, "saetta code: %s\n", b);
}

static void sleep_ms(int ms) {
    struct timespec ts = {ms / 1000, (long)(ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
}

static bool same_dir(const char *x, const char *y) {
    char rx[PATH_MAX], ry[PATH_MAX];
    if (!realpath(x, rx)) snprintf(rx, sizeof rx, "%s", x);
    if (!realpath(y, ry)) snprintf(ry, sizeof ry, "%s", y);
    return !strcmp(rx, ry);
}

int code_share_join(Arena *a, const char *dir, CodeModel *m) {
    return code_share_join_until(a, dir, m, now_ns() + 600ull * 1000000000ull, NULL);
}

int code_share_join_until(Arena *a, const char *dir, CodeModel *m, uint64_t deadline_ns, volatile int *abort) {
    snprintf(S.dir, sizeof S.dir, "%s", dir);
    bool told = false;
    for (;;) {
        if (abort && *abort) return -1;
        Inproc in;
        if (!instance_inproc_info(&in)) return 0; // nobody holds the model: this process loads it
        if (in.pid == getpid()) return 0;           // this image after a /model restart: it holds the lock
        if (in.state == INPROC_PRIVATE) {
            say("another Saetta Code (pid %d) runs the model for itself (print, json or rpc mode); one model runs at a time.", in.pid);
            return -1;
        }
        if (in.dir[0] && !same_dir(in.dir, dir)) {
            say("another Saetta Code (pid %d) runs %s; one model runs at a time.\n"
                "Run this one with that pack (saetta start code joins it), or close the other first.", in.pid, in.dir);
            return -1;
        }
        if (in.state == INPROC_SERVING && in.sock[0] && instance_inproc_join(&in)) {
            code_share_describe(a, m, arena_strdup(a, in.model), in.ctx, arena_printf(a, "unix:%s", in.sock));
            m->api_key = arena_strdup(a, in.token);
            m->engine = NULL;
            m->sock = arena_strdup(a, in.sock);
            m->verify = code_share_verify; // the holder checked before every request (code_oai.c)
            snprintf(S.holder, sizeof S.holder, "%s", in.holder);
            snprintf(S.jtoken, sizeof S.jtoken, "%s", in.token);
            snprintf(S.jsock, sizeof S.jsock, "%s", in.sock);
            S.joined = true;
            if (code_share_verify(m) == 0) {
                // the program on the holder's port does not know the token: nothing is sent to it
                instance_inproc_leave();
                S.joined = false;
                say("the Saetta Code that holds the model (pid %d) does not answer on its socket; restart to load the model here", in.pid);
                return -1;
            }
            dbg("saetta code: using the model of Saetta Code pid %d at %s\n", in.pid, in.sock);
            return 1;
        }
        // loading or restarting (or exiting: then the lock frees and this process loads the model)
        if (!told && (in.state == INPROC_LOADING || in.state == INPROC_RESTARTING) && (S.notice || isatty(2)))
            say("waiting for the model that Saetta Code (pid %d) is loading ...", in.pid);
        told = true;
        if (now_ns() > deadline_ns) {
            say("Saetta Code (pid %d) did not have the model ready in time", in.pid);
            return -1;
        }
        // the wait for a holder: the caller's lock (the rpc mode's) let go meanwhile
        if (g_wait_begin) g_wait_begin();
        sleep_ms(100);
        if (g_wait_end) g_wait_end();
    }
}

bool code_share_joined(void) { return S.joined; }

typedef struct {
    Buf b;
} Collect;

static int collect(void *ctx, int status, const void *p, size_t n) {
    (void)status;
    Collect *c = ctx;
    if (c->b.len + n > 4096) return 1;
    buf_add(&c->b, p, n);
    return 0;
}

// One GET on the holder's socket: its HTTP status and body (at most cap - 1 bytes), -1 on failure.
static int get(const char *sock, const char *path, char *body, size_t cap) {
    Collect c = {0};
    int status = 0;
    char err[256];
    int rc = unix_http(sock, "GET", path, NULL, 0, NULL, 0, 5, collect, &c, &status, err, sizeof err, NULL);
    snprintf(body, cap, "%.*s", (int)c.b.len, c.b.ptr ? c.b.ptr : "");
    buf_free(&c.b);
    return rc ? -1 : status;
}

int code_share_verify(const CodeModel *m) {
    (void)m;
    if (!S.joined) return 1;
    // (a) the record still names the holder joined, alive (its pid and start time), on the same socket
    Inproc in;
    if (!instance_inproc_info(&in) || in.state != INPROC_SERVING || strcmp(in.holder, S.holder) || strcmp(in.sock, S.jsock)) return 0;
    // (b) the program on that port knows the token: HMAC-SHA256(token, nonce) for a fresh nonce,
    // asked without the token, right before the request that carries it
    uint8_t raw[32];
    arc4random_buf(raw, sizeof raw);
    char nonce[65], path[128], body[512], want[65];
    hash_hex(raw, sizeof raw, nonce);
    snprintf(path, sizeof path, "/saetta/whoami?n=%s", nonce);
    uint8_t d[SHA256_LEN];
    hmac_sha256(S.jtoken, strlen(S.jtoken), nonce, strlen(nonce), d);
    hash_hex(d, SHA256_LEN, want);
    int st = get(S.jsock, path, body, sizeof body);
    if (st != 200) return st == 503 ? 2 : 0; // 503: the holder is busy (its slots full), not gone
    const char *h = strstr(body, "\"hmac\":\"");
    if (!h || strlen(h + 8) < 64) return 0;
    unsigned char diff = 0;
    for (int i = 0; i < 64; i++) diff |= (unsigned char)(h[8 + i] ^ want[i]);
    return diff == 0;
}

int code_share_rejoin(Arena *a, CodeModel *m, uint64_t deadline_ns, volatile int *abort) {
    if (S.joined) instance_inproc_leave();
    S.joined = false;
    return code_share_join_until(a, S.dir, m, deadline_ns, abort);
}

// ---- the holder's listener ------------------------------------------------------------------------

static bool same_secret(const char *a, size_t an, const char *b, size_t bn) {
    unsigned char d = an != bn;
    for (size_t i = 0; i < an && i < bn; i++) d |= (unsigned char)(a[i] ^ b[i]);
    return d == 0;
}

// A header's value in the head bytes [p, p+n), NULL when absent.
static const char *find_header(const char *p, size_t n, const char *name, size_t *len) {
    size_t nl = strlen(name);
    for (size_t i = 0; i < n;) {
        const char *e = memchr(p + i, '\n', n - i);
        size_t j = e ? (size_t)(e - p) : n;
        if (j - i > nl && p[i + nl] == ':' && !strncasecmp(p + i, name, nl)) {
            size_t s = i + nl + 1;
            while (s < j && (p[s] == ' ' || p[s] == '\t')) s++;
            size_t t = j;
            while (t > s && (p[t - 1] == '\r' || p[t - 1] == ' ')) t--;
            *len = t - s;
            return p + s;
        }
        i = j + 1;
    }
    return NULL;
}

static void refuse(int fd, int status, const char *msg) {
    char b[256];
    int bl = snprintf(b, sizeof b, "{\"error\":{\"message\":\"%s\",\"type\":\"invalid_request_error\"}}", msg);
    char h[256];
    int hl = snprintf(h, sizeof h, "HTTP/1.1 %d %s\r\nconnection: close\r\ncontent-type: application/json\r\ncontent-length: %d\r\n\r\n",
                      status, msg, bl);
    (void)!write(fd, h, (size_t)hl);
    (void)!write(fd, b, (size_t)bl);
}

static bool write_all(int fd, const void *p, size_t n) {
    for (size_t w = 0; w < n;) {
        ssize_t x = write(fd, (const char *)p + w, n - w);
        if (x <= 0) return false;
        w += (size_t)x;
    }
    return true;
}

static bool write_chunk(int fd, const char *p, size_t n) {
    char hd[24];
    int hl = snprintf(hd, sizeof hd, "%zx\r\n", n);
    return write_all(fd, hd, (size_t)hl) && write_all(fd, p, n) && write_all(fd, "\r\n", 2);
}

static Turns g_relay; // the stub's requests take turns as the engine's do (server.c)

// The stub's relay: the joined session's request read whole and its token checked (as server.c
// does), then, in its turn, sent to the stub's server and the answer relayed back. While it waits
// for its turn a streamed request gets its head and keep-alive comments, as server.c sends them.
static void relay(int fd) {
    struct timeval t5 = {5, 0}, t30 = {30, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &t5, sizeof t5); // the head within 5 s, as server.c
    Buf req = {0};
    char buf[65536];
    size_t hl = 0;
    while (!hl) {
        ssize_t r = read(fd, buf, sizeof buf);
        if (r <= 0 || req.len > (1u << 20)) goto out;
        buf_add(&req, buf, (size_t)r);
        for (size_t i = 3; i < req.len && !hl; i++)
            if (!memcmp(req.ptr + i - 3, "\r\n\r\n", 4)) hl = i + 1;
    }
    if (!strncmp(req.ptr, "GET /saetta/whoami?n=", 21)) { // the challenge, as server.c answers it
        const char *n = req.ptr + 21;
        size_t nl = strcspn(n, " \r\n");
        uint8_t d[SHA256_LEN];
        char hex[65], body[96], head[256];
        hmac_sha256(S.token + 7, strlen(S.token + 7), n, nl, d);
        hash_hex(d, SHA256_LEN, hex);
        int bl = snprintf(body, sizeof body, "{\"hmac\":\"%s\"}", hex);
        int hl2 = snprintf(head, sizeof head, "HTTP/1.1 200 OK\r\nconnection: close\r\ncontent-type: application/json\r\ncontent-length: %d\r\n\r\n", bl);
        if (write_all(fd, head, (size_t)hl2)) write_all(fd, body, (size_t)bl);
        goto out;
    }
    size_t al = 0, ol = 0, cl = 0;
    const char *auth = find_header(req.ptr, hl, "authorization", &al);
    if (find_header(req.ptr, hl, "origin", &ol)) {
        refuse(fd, 403, "Forbidden");
        goto out;
    }
    if (!auth || !same_secret(auth, al, S.token, strlen(S.token))) {
        refuse(fd, 401, "Unauthorized");
        goto out;
    }
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &t30, sizeof t30);
    const char *clen = find_header(req.ptr, hl, "content-length", &cl);
    size_t want = hl + (clen ? (size_t)strtoull(clen, NULL, 10) : 0);
    while (req.len < want) {
        ssize_t r = read(fd, buf, sizeof buf);
        if (r <= 0) goto out;
        buf_add(&req, buf, (size_t)r);
    }
    bool stream = memmem(req.ptr + hl, req.len - hl, "\"stream\":true", 13) != NULL;
    uint64_t t = turns_take(&g_relay);
    bool head_sent = false;
    while (!turns_wait(&g_relay, t, 5000)) {
        bool ok;
        if (stream) {
            static const char head[] = "HTTP/1.1 200 OK\r\nconnection: close\r\ntransfer-encoding: chunked\r\ncontent-type: "
                                       "text/event-stream\r\ncache-control: no-cache\r\n\r\n";
            ok = (head_sent || write_all(fd, head, sizeof head - 1)) && write_chunk(fd, ": queued\n\n", 10);
            head_sent = true;
        } else {
            char c;
            ok = recv(fd, &c, 1, MSG_PEEK | MSG_DONTWAIT) != 0;
        }
        if (!ok) {
            turns_leave(&g_relay, t);
            goto out;
        }
    }
    int up = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    if (up >= 0) setsockopt(up, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
    struct sockaddr_in sa = {.sin_family = AF_INET, .sin_port = htons(S.stub_port), .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
    if (up >= 0 && connect(up, (struct sockaddr *)&sa, sizeof sa) == 0 && write_all(up, req.ptr, req.len)) {
        // the answer: as it comes, or (the head already sent) its body as chunks
        Buf uh = {0};
        bool body = !head_sent;
        for (;;) {
            // the answer until the server closes, or the client leaves (a keep-alive answer of
            // the server ends when the client, having read it, closes)
            struct pollfd pf[2] = {{up, POLLIN, 0}, {fd, POLLIN, 0}};
            if (poll(pf, 2, -1) < 0) {
                if (errno == EINTR) continue;
                break;
            }
            if (pf[1].revents && !(pf[0].revents & POLLIN)) {
                char c;
                if (recv(fd, &c, 1, MSG_PEEK | MSG_DONTWAIT) <= 0) break;
            }
            if (!(pf[0].revents & (POLLIN | POLLHUP | POLLERR))) continue;
            ssize_t r = read(up, buf, sizeof buf);
            if (r <= 0) break;
            if (body) {
                if (!(head_sent ? write_chunk(fd, buf, (size_t)r) : write_all(fd, buf, (size_t)r))) break;
                continue;
            }
            buf_add(&uh, buf, (size_t)r);
            for (size_t i = 3; i < uh.len && !body; i++)
                if (!memcmp(uh.ptr + i - 3, "\r\n\r\n", 4)) {
                    body = true;
                    if (i + 1 < uh.len && !write_chunk(fd, uh.ptr + i + 1, uh.len - i - 1)) break;
                }
        }
        if (head_sent) write_all(fd, "0\r\n\r\n", 5);
        buf_free(&uh);
    }
    if (up >= 0) close(up);
    turns_done(&g_relay);
out:
    buf_free(&req);
}

static void *conn_main(void *arg) {
    int slot = (int)(intptr_t)arg, fd = S.fds[slot];
    if (S.engine) {
        Err e = server_engine_connection(S.engine, fd, NULL, NULL);
        if (e) dbg("saetta code: shared connection: %s\n", e);
    } else relay(fd);
    pthread_mutex_lock(&S.mu);
    close(fd);
    S.fds[slot] = -1;
    S.active--;
    pthread_cond_broadcast(&S.cv);
    pthread_mutex_unlock(&S.mu);
    return NULL;
}

static void *accept_loop(void *arg) {
    (void)arg;
    int one = 1;
    for (;;) {
        struct pollfd pf[2] = {{S.ls, POLLIN, 0}, {S.wake[0], POLLIN, 0}};
        if (poll(pf, 2, -1) < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (pf[1].revents) break; // code_share_end
        int fd = accept(S.ls, NULL, NULL);
        if (fd < 0) {
            if (errno == EMFILE || errno == ENFILE || errno == ENOBUFS || errno == ENOMEM) sleep_ms(100); // no spinning
            continue; // EAGAIN (the listening socket does not block), ECONNABORTED, EINTR
        }
        int fl = fcntl(fd, F_GETFL);
        if (fl >= 0) fcntl(fd, F_SETFL, fl & ~O_NONBLOCK);
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
        // no connection guard thread here (one socket at a time): reads and writes time out instead
        struct timeval rt = {30, 0}, wt = {60, 0};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rt, sizeof rt);
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &wt, sizeof wt);
        pthread_mutex_lock(&S.mu);
        int slot = -1;
        for (int i = 0; i < MAX_CONNS && slot < 0; i++)
            if (S.fds[i] < 0) slot = i;
        if (slot >= 0) {
            S.fds[slot] = fd;
            S.active++;
        }
        pthread_mutex_unlock(&S.mu);
        if (slot < 0) {
            refuse(fd, 503, "Service Unavailable");
            close(fd);
            continue;
        }
        pthread_t th;
        if (pthread_create(&th, NULL, conn_main, (void *)(intptr_t)slot) != 0) conn_main((void *)(intptr_t)slot);
        else pthread_detach(th);
    }
    return NULL;
}

// The socket path made ready for bind: nothing there, or a socket of this user that is left over
// (removed); anything else (a file, a symbolic link, someone else's) is refused, never followed.
static Err socket_path(const char *path) {
    struct stat st;
    if (lstat(path, &st) != 0) return errno == ENOENT ? OK : err_errno(errno);
    if (!S_ISSOCK(st.st_mode) || st.st_uid != getuid()) {
        dbg("saetta code: %s exists and is not a socket of this user: the model is not shared\n", path);
        return "SocketPathTaken";
    }
    return unlink(path) == 0 ? OK : err_errno(errno);
}

static Err listen_and_publish(const CodeModel *m) {
    for (int i = 0; i < MAX_CONNS; i++) S.fds[i] = -1;
    const char *home = getenv("HOME");
    if (!home) return "NoHome";
    snprintf(S.sock, sizeof S.sock, "%s/.saetta/run/inproc.sock", home);
    struct sockaddr_un sa = {.sun_family = AF_UNIX};
    if (strlen(S.sock) >= sizeof sa.sun_path) return "SocketPathTooLong";
    snprintf(sa.sun_path, sizeof sa.sun_path, "%s", S.sock);
    TRY(socket_path(S.sock));
    int ls = socket(AF_UNIX, SOCK_STREAM, 0);
    if (ls < 0) return err_errno(errno);
    mode_t old = umask(077); // the socket is born 0600 (its folder is 0700)
    int br = bind(ls, (struct sockaddr *)&sa, sizeof sa);
    umask(old);
    if (!br) chmod(S.sock, 0600);
    if (br != 0 || listen(ls, 64) != 0 || pipe(S.wake) != 0) {
        Err e = err_errno(errno);
        close(ls);
        return e;
    }
    S.ls = ls;
    // accept never blocks: the accept thread returns to its poll (and sees a stop) whatever happens
    fcntl(ls, F_SETFL, fcntl(ls, F_GETFL) | O_NONBLOCK);
    turns_init(&g_relay);
    const char *tok = instance_inproc_token();
    snprintf(S.token, sizeof S.token, "Bearer %s", tok ? tok : "");
    if (S.engine) server_engine_set_token(S.engine, tok);
    if (pthread_create(&S.accept_th, NULL, accept_loop, NULL) != 0) {
        close(ls);
        S.ls = -1;
        return "ThreadFailed";
    }
    S.serving = true;
    TRY(instance_inproc_publish(INPROC_SERVING, (uint32_t)m->context_window, m->id, S.sock));
    dbg("saetta code: sharing the model with other Saetta Codes at %s\n", S.sock);
    return OK;
}

Err code_share_serve(const CodeModel *m, bool interactive) {
    S.engine = m->engine;
    if (!interactive) {
        // a print, json or rpc run ends on its own: it keeps the model to itself and never waits
        S.priv = true;
        return instance_inproc_publish(INPROC_PRIVATE, (uint32_t)m->context_window, m->id, "");
    }
    Err e = listen_and_publish(m);
    if (e) {
        // not shared (the socket's path is taken, say): the model stays this session's own
        S.priv = true;
        (void)instance_inproc_publish(INPROC_PRIVATE, (uint32_t)m->context_window, m->id, "");
    }
    return e;
}

bool code_share_stub(Arena *a, const char *dir, CodeModel *m, bool interactive, Err *err) {
    const char *stub = getenv("SAETTA_CODE_SHARE_STUB");
    if (!stub || !*stub) return false;
    *err = OK;
    const char *colon = strrchr(stub, ':');
    S.stub_port = colon ? (uint16_t)atoi(colon + 1) : 0;
    int fd;
    if ((*err = instance_inproc_acquire(dir, &fd))) {
        dbg("saetta code: another Saetta Code is running the model\n"); // a takeover lost to another session: it retries
        return true;
    }
    const char *id = getenv("SAETTA_CODE_SHARE_STUB_MODEL");
    code_share_describe(a, m, id && *id ? id : "mock-model", 131072, arena_printf(a, "%s/v1", stub));
    m->engine = NULL;
    // the load a real engine takes, for the test of a session that joins while it loads
    const char *ms = getenv("SAETTA_CODE_SHARE_STUB_LOAD_MS");
    if (ms && atoi(ms) > 0) sleep_ms(atoi(ms));
    Err se = code_share_serve(m, interactive);
    if (se) dbg("saetta code: the model is not shared with other sessions: %s\n", se); // as open_engine
    return true;
}

bool code_share_holding(void) { return instance_inproc_token() != NULL; }

size_t code_share_clients(void) { return S.serving ? instance_inproc_clients() : 0; }

static void on_int(int sig) {
    (void)sig;
    g_stop = 1;
}

// Stops accepting and waits for the connections in flight (all of them end before the engine and
// its arena go); forced: they are cut first.
static void stop_listener(bool forced) {
    if (S.ls < 0) return;
    (void)!write(S.wake[1], "x", 1);
    pthread_join(S.accept_th, NULL);
    close(S.ls);
    S.ls = -1;
    unlink(S.sock);
    pthread_mutex_lock(&S.mu);
    // the connections in flight end on their own (their sessions are gone), at most 10 s; then,
    // or at once when forced, they are cut: a stuck one never keeps this process alive
    struct timespec dl;
    clock_gettime(CLOCK_REALTIME, &dl);
    dl.tv_sec += forced ? 0 : 10;
    while (S.active > 0 && !forced)
        if (pthread_cond_timedwait(&S.cv, &S.mu, &dl) != 0) break;
    for (int i = 0; i < MAX_CONNS; i++)
        if (S.fds[i] >= 0) shutdown(S.fds[i], SHUT_RDWR);
    while (S.active > 0) pthread_cond_wait(&S.cv, &S.mu);
    pthread_mutex_unlock(&S.mu);
    close(S.wake[0]);
    close(S.wake[1]);
    S.wake[0] = S.wake[1] = -1;
}

void code_share_end(bool now) {
    if (S.joined) {
        instance_inproc_leave();
        S.joined = false;
    }
    if (S.priv) {
        instance_inproc_withdraw(true);
        S.priv = false;
    }
    if (!S.serving) return;
    // the count and the record's withdrawal in one step: no session joins after the last count
    size_t n = instance_inproc_withdraw(now);
    bool forced = now;
    if (n) {
        struct sigaction sa = {0}, old;
        sa.sa_handler = on_int;
        sigaction(SIGINT, &sa, &old);
        fprintf(stderr, "saetta code: %zu other Saetta Code session%s use%s the model in this process; serving %s until %s exit%s "
                        "(Ctrl-C stops now)\n",
                n, n == 1 ? "" : "s", n == 1 ? "s" : "", n == 1 ? "it" : "them", n == 1 ? "it" : "they", n == 1 ? "s" : "");
        while (instance_inproc_withdraw(false) > 0) {
            if (g_stop) {
                forced = true;
                instance_inproc_withdraw(true);
                fprintf(stderr, "saetta code: stopped; the other sessions load the model again on their next request\n");
                break;
            }
            sleep_ms(100);
        }
        sigaction(SIGINT, &old, NULL);
    }
    stop_listener(forced);
    S.serving = false;
    dbg("saetta code: the model is no longer shared\n");
}
