// The machine-wide server lock and the session registry (see instance.h).
#include "instance.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/file.h>
#include <unistd.h>

static Err run_dir(Arena *a, char **out) {
    const char *home = getenv("HOME");
    if (!home) return "NoHome";
    *out = path_join(a, path_join(a, home, ".saetta"), "run");
    return OK;
}

static Err lock_path(Arena *a, char **out) {
    char *d;
    TRY(run_dir(a, &d));
    *out = path_join(a, d, "serve.lock");
    return OK;
}

static Err clients_dir(Arena *a, char **out) {
    char *d;
    TRY(run_dir(a, &d));
    *out = path_join(a, d, "clients");
    return OK;
}

static Err pwrite_all(int fd, const void *p, size_t n, off_t off) {
    const char *c = p;
    while (n) {
        ssize_t w = pwrite(fd, c, n, off);
        if (w < 0) {
            if (errno == EINTR) continue;
            return err_errno(errno);
        }
        c += w;
        n -= (size_t)w;
        off += w;
    }
    return OK;
}

static Err acquire_in(Arena *a, uint16_t port, int *out) {
    char *dir, *path;
    TRY(run_dir(a, &dir));
    TRY(mkdir_p(dir));
    TRY(lock_path(a, &path));
    int fd = open(path, O_WRONLY | O_CREAT | O_CLOEXEC, 0666);
    if (fd < 0) return err_errno(errno);
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        int e = errno;
        close(fd);
        if (e == EWOULDBLOCK) {
            Server s;
            if (instance_running(&s)) {
                dbg("saetta: a server is already running on this machine (pid %d) at http://127.0.0.1:%u.\n"
                    "Only one runs at a time: use it (saetta start <agent> joins it), or stop it first.\n",
                    s.pid, s.port);
            } else dbg("saetta: another saetta server holds %s\n", path);
            return "AlreadyRunning";
        }
        return err_errno(e);
    }
    const char *v = getenv("SAETTA_AUTOSTOP");
    bool autostop = v && strcmp(v, "1") == 0;
    // fixed width, so a longer old record never leaves a tail behind
    char head[64], rec[64];
    snprintf(head, sizeof head, "%d %u %s", (int)getpid(), port, autostop ? "auto" : "manual");
    int n = snprintf(rec, sizeof rec, "%-48s", head); // padded to exactly 48 bytes
    if (n != 48) {
        close(fd);
        return "NoSpaceLeft";
    }
    Err e = pwrite_all(fd, rec, 48, 0);
    if (e) {
        close(fd);
        return e;
    }
    *out = fd;
    return OK;
}

Err instance_acquire(uint16_t port, int *fd) {
    Arena a = {0};
    Err e = acquire_in(&a, port, fd);
    arena_reset(&a);
    return e;
}

static bool parse_i32(Str s, int32_t *out) {
    int64_t v;
    if (parse_i64(s, &v) || v < INT32_MIN || v > INT32_MAX) return false;
    *out = (int32_t)v;
    return true;
}

// Next token separated by spaces (std.mem.tokenizeScalar).
static bool next_tok(Str s, size_t *pos, Str *tok) {
    size_t i = *pos;
    while (i < s.len && s.ptr[i] == ' ') i++;
    if (i == s.len) return false;
    size_t j = i;
    while (j < s.len && s.ptr[j] != ' ') j++;
    *tok = str_n(s.ptr + i, j - i);
    *pos = j;
    return true;
}

static bool running_in(Arena *a, Server *out) {
    char *path;
    if (lock_path(a, &path)) return false;
    // a shared lock we can take means nobody holds the exclusive one
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    if (flock(fd, LOCK_SH | LOCK_NB) == 0) {
        close(fd);
        return false;
    }
    int e = errno;
    close(fd);
    if (e != EWOULDBLOCK) return false;
    fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    char buf[64];
    size_t n = 0;
    while (n < sizeof buf) {
        ssize_t r = pread(fd, buf + n, sizeof buf - n, (off_t)n);
        if (r < 0 && errno == EINTR) continue;
        if (r < 0) {
            close(fd);
            return false;
        }
        if (r == 0) break;
        n += (size_t)r;
    }
    close(fd);
    Str s = str_n(buf, n), t;
    size_t pos = 0;
    int32_t pid;
    uint64_t port;
    if (!next_tok(s, &pos, &t) || !parse_i32(t, &pid)) return false;
    if (!next_tok(s, &pos, &t) || parse_u64(t, &port) || port > UINT16_MAX) return false;
    Str mode = S("manual");
    if (next_tok(s, &pos, &t)) mode = t;
    *out = (Server){.pid = pid, .port = (uint16_t)port, .autostop = str_starts(mode, S("auto"))};
    return true;
}

bool instance_running(Server *out) {
    Arena a = {0};
    Server s;
    bool r = running_in(&a, &s);
    arena_reset(&a);
    if (r && out) *out = s;
    return r;
}

static Err register_in(Arena *a, const char *agent) {
    char *dir;
    TRY(clients_dir(a, &dir));
    TRY(mkdir_p(dir));
    char *p = arena_printf(a, "%s/%d", dir, (int)getpid());
    int fd = open(p, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
    if (fd < 0) return err_errno(errno);
    Err e = pwrite_all(fd, agent, strlen(agent), 0);
    close(fd);
    return e;
}

Err instance_register(const char *agent) {
    Arena a = {0};
    Err e = register_in(&a, agent);
    arena_reset(&a);
    return e;
}

Err instance_keep_path(Arena *a, char **out) {
    char *d;
    TRY(run_dir(a, &d));
    *out = path_join(a, d, "keep");
    return OK;
}

bool instance_kept(void) {
    Arena a = {0};
    char *p;
    bool r = !instance_keep_path(&a, &p) && access(p, F_OK) == 0;
    arena_reset(&a);
    return r;
}

void instance_set_keep(bool on) {
    Arena a = {0};
    char *p;
    if (!instance_keep_path(&a, &p)) {
        if (on) {
            int fd = open(p, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
            if (fd >= 0) close(fd);
        } else unlink(p);
    }
    arena_reset(&a);
}

size_t instance_clients(Arena *a, Client **out) {
    VEC(Client) v = {0};
    *out = NULL;
    char *dir_path;
    if (clients_dir(a, &dir_path)) return 0;
    DIR *dir = opendir(dir_path);
    if (!dir) return 0;
    struct dirent *e;
    while ((e = readdir(dir))) {
        if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0) continue;
        int32_t pid;
        if (!parse_i32(str(e->d_name), &pid)) continue;
        char *p = path_join(a, dir_path, e->d_name);
        if (kill(pid, 0) != 0) {
            unlink(p);
            continue;
        }
        char *text = NULL;
        size_t len = 0;
        const char *name = "";
        if (!read_file(p, 64, &text, &len)) {
            name = arena_strndup(a, text, len);
            free(text);
        }
        vec_push(&v, ((Client){.pid = pid, .agent = name[0] ? name : "agent"}));
    }
    closedir(dir);
    size_t n = v.len;
    if (n) *out = arena_dup(a, v.ptr, n * sizeof(Client));
    vec_free(&v);
    return n;
}

size_t instance_count_clients(void) {
    Arena a = {0};
    Client *c;
    size_t n = instance_clients(&a, &c);
    arena_reset(&a);
    return n;
}

size_t instance_unregister(void) {
    Arena a = {0};
    char *dir_path;
    size_t others = 0;
    if (clients_dir(&a, &dir_path)) goto done;
    int32_t me = (int32_t)getpid();
    DIR *dir = opendir(dir_path);
    if (!dir) goto done;
    struct dirent *e;
    while ((e = readdir(dir))) {
        if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0) continue;
        int32_t pid;
        if (!parse_i32(str(e->d_name), &pid)) continue;
        bool alive = pid != me && kill(pid, 0) == 0;
        if (alive) others += 1;
        else unlink(path_join(&a, dir_path, e->d_name));
    }
    closedir(dir);
done:
    arena_reset(&a);
    return others;
}
