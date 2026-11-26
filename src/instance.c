// The machine-wide server lock and the session registry (see instance.h).
#include "instance.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/file.h>
#include <libproc.h>
#include <limits.h>
#include <sys/stat.h>
#include <time.h>
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

// ---- the in-process engine (Saetta Code with --pack): its own lock, so a second engine is not
// loaded beside it -----------------------------------------------------------------------------

static Err inproc_clients_dir(Arena *a, char **out);

// This process as the in-process engine's holder (instance_inproc_acquire).
static struct {
    bool on, restarting;
    int lock_fd;
    char id[40], token[65], dir[1024];
} g_holder = {.lock_fd = -1};

static Err inproc_path(Arena *a, char **out) {
    char *d;
    TRY(run_dir(a, &d));
    *out = path_join(a, d, "inproc.lock");
    return OK;
}

// A process's start time in microseconds (0: no such process): with the pid, the holder's
// identity, so a recycled pid is never taken for it (and no lock is probed to find it).
static uint64_t proc_start_us(int pid) {
    struct proc_bsdinfo bi;
    if (pid <= 0 || proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &bi, sizeof bi) != (int)sizeof bi) return 0;
    return (uint64_t)bi.pbi_start_tvsec * 1000000ull + bi.pbi_start_tvusec;
}

static bool read_record(Inproc *out);

// The lock this image inherited from the one that exec'd it (a /model restart): its descriptor,
// accepted only when it is that lock: the same file as ~/.saetta/run/inproc.lock (not a link),
// held by it (a non-blocking exclusive flock succeeds only for the holder of that open file), and
// the record says restarting, by this pid. Anything else is ignored (never written or closed).
static int inherited_lock(void) {
    const char *v = getenv("SAETTA_INPROC_LOCK_FD");
    if (!v) return -1;
    int fd = atoi(v);
    unsetenv("SAETTA_INPROC_LOCK_FD");
    struct stat fs, ps;
    Arena a = {0};
    char *path;
    bool ok = fd >= 3 && fcntl(fd, F_GETFD) >= 0 && fstat(fd, &fs) == 0 && !inproc_path(&a, &path) && lstat(path, &ps) == 0 &&
              S_ISREG(ps.st_mode) && fs.st_dev == ps.st_dev && fs.st_ino == ps.st_ino && flock(fd, LOCK_EX | LOCK_NB) == 0;
    arena_reset(&a);
    Inproc in;
    ok = ok && read_record(&in) && in.state == INPROC_RESTARTING && in.pid == getpid();
    if (!ok) return -1;
    fcntl(fd, F_SETFD, FD_CLOEXEC);
    return fd;
}
static int g_inherited = -2; // -2: not looked at yet

void instance_inproc_take_inherited(void) {
    if (g_inherited == -2) g_inherited = inherited_lock();
}

static int take_inherited(void) {
    instance_inproc_take_inherited();
    int fd = g_inherited;
    g_inherited = -1;
    return fd;
}

Err instance_inproc_acquire(const char *dir, int *out) {
    Arena a = {0};
    char *rdir, *path;
    Err e = run_dir(&a, &rdir);
    if (!e) e = mkdir_p(rdir);
    if (!e) e = inproc_path(&a, &path);
    if (e) {
        arena_reset(&a);
        return e;
    }
    chmod(rdir, 0700); // the shared engine's record holds its token, and its socket is there
    // a /model restart hands its lock over: this image is still the holder
    int fd = take_inherited();
    if (fd < 0) fd = open(path, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
    arena_reset(&a);
    if (fd < 0) return err_errno(errno);
    // a reader's liveness probe (lock_held) holds the lock for an instant: tried again for 100 ms
    // before another holder is taken to have it
    for (int tries = 0; flock(fd, LOCK_EX | LOCK_NB) != 0; tries++) {
        int er = errno;
        if (er == EWOULDBLOCK && tries < 20) {
            struct timespec ts = {0, 5000000L};
            nanosleep(&ts, NULL);
            continue;
        }
        close(fd);
        return er == EWOULDBLOCK ? "EngineBusy" : err_errno(er);
    }
    char rec[32];
    int n = snprintf(rec, sizeof rec, "%-16d", (int)getpid());
    if (ftruncate(fd, 0) != 0 || pwrite_all(fd, rec, (size_t)n, 0)) {
        close(fd);
        return "WriteFailed";
    }
    // a new holder: its identity and token, and the record (loading, with its pack)
    uint8_t raw[32];
    arc4random_buf(raw, sizeof raw);
    for (size_t i = 0; i < sizeof raw; i++) snprintf(g_holder.token + 2 * i, 3, "%02x", raw[i]);
    snprintf(g_holder.id, sizeof g_holder.id, "%d-%llu", (int)getpid(), (unsigned long long)wall_ns());
    char rp[PATH_MAX];
    if (dir && realpath(dir, rp)) dir = rp;
    snprintf(g_holder.dir, sizeof g_holder.dir, "%s", dir ? dir : "");
    g_holder.on = true;
    g_holder.lock_fd = fd;
    Arena ca = {0};
    char *cdir;
    (void)inproc_clients_dir(&ca, &cdir); // made private before any session registers
    arena_reset(&ca);
    Err pe = instance_inproc_publish(INPROC_LOADING, 0, "", "");
    if (pe) {
        close(fd);
        g_holder.on = false;
        return pe;
    }
    *out = fd;
    return OK;
}

void instance_inproc_release_inherited(void) {
    int fd = take_inherited();
    if (fd < 0) return;
    // restarted away from the model (a verified lock: inherited_lock): not a holder any more. The
    // record goes first, then the lock: a session that looks in between finds no holder, tries to
    // take the lock, and tries again a moment later
    Arena a = {0};
    char *p;
    if (!run_dir(&a, &p)) unlink(path_join(&a, p, "inproc.rec"));
    arena_reset(&a);
    close(fd);
}

Err instance_inproc_restarting(const char *dir) {
    if (!g_holder.on || g_holder.lock_fd < 0) return "NotHolder";
    g_holder.restarting = true; // withdraw leaves this record as it is
    snprintf(g_holder.dir, sizeof g_holder.dir, "%s", dir ? dir : "");
    TRY(instance_inproc_publish(INPROC_RESTARTING, 0, "", ""));
    // the lock goes on into the next image (it takes it from SAETTA_INPROC_LOCK_FD)
    fcntl(g_holder.lock_fd, F_SETFD, 0);
    char v[16];
    snprintf(v, sizeof v, "%d", g_holder.lock_fd);
    setenv("SAETTA_INPROC_LOCK_FD", v, 1);
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


// ---- sharing the in-process engine (code_share.c) ---------------------------------------------------
// The record ~/.saetta/run/inproc.rec (0600, replaced by rename): "pid state port ctx holder\ntoken\n
// model\ndir\n". Registration (a joiner's entry, the holder's last count and the record's
// withdrawal) happens under the flock of ~/.saetta/run/inproc.reg, so no session joins after the
// holder's final count.

static Err rec_path(Arena *a, const char *name, char **out) {
    char *d;
    TRY(run_dir(a, &d));
    *out = path_join(a, d, name);
    return OK;
}

static int reg_lock(void) {
    Arena a = {0};
    char *p;
    int fd = -1;
    if (!rec_path(&a, "inproc.reg", &p)) {
        fd = open(p, O_RDWR | O_CREAT | O_CLOEXEC, 0600);
        if (fd >= 0) flock(fd, LOCK_EX);
    }
    arena_reset(&a);
    return fd;
}

static void reg_unlock(int fd) {
    if (fd >= 0) close(fd); // drops the flock
}

static const char *const state_names[] = {"loading", "serving", "private", "closing", "restarting"};

Err instance_inproc_publish(InprocState st, uint32_t ctx, const char *model, const char *sock) {
    if (!g_holder.on) return "NotHolder";
    Arena a = {0};
    char *path, *tmp;
    Err e = rec_path(&a, "inproc.rec", &path);
    if (!e) e = rec_path(&a, arena_printf(&a, "inproc.rec.%d", (int)getpid()), &tmp);
    if (e) {
        arena_reset(&a);
        return e;
    }
    char rec[4096];
    int n = snprintf(rec, sizeof rec, "%d %s %u %s %llu\n%s\n%s\n%s\n%s\n", (int)getpid(), state_names[st], ctx, g_holder.id,
                     (unsigned long long)proc_start_us(getpid()), g_holder.token, model ? model : "", g_holder.dir, sock ? sock : "");
    if (n < 0 || (size_t)n >= sizeof rec) {
        arena_reset(&a);
        return "NoSpaceLeft";
    }
    unlink(tmp);
    int fd = open(tmp, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) e = err_errno(errno);
    else {
        e = pwrite_all(fd, rec, (size_t)n, 0);
        close(fd);
        if (!e && rename(tmp, path) != 0) e = err_errno(errno);
        if (e) unlink(tmp);
    }
    arena_reset(&a);
    return e;
}

const char *instance_inproc_token(void) { return g_holder.on ? g_holder.token : NULL; }

// Whether a process holds ~/.saetta/run/inproc.lock: a non-blocking shared flock fails only while
// someone holds it exclusively (this process's own lock too: a lock is per open file). The probe
// holds the lock for an instant (instance_inproc_acquire tries again meanwhile). When the probe gets
// it, nobody holds the lock, so nobody writes the record (a holder writes it only under its lock):
// the record is stale and is removed under the probe (drop).
static bool lock_held(bool drop) {
    Arena a = {0};
    char *path;
    bool held = true; // an error other than no file: taken as held (never a live holder ignored)
    if (!inproc_path(&a, &path)) {
        int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0) held = errno != ENOENT; // no lock file: no holder (the record is left: one may be taking it now)
        else {
            held = flock(fd, LOCK_SH | LOCK_NB) != 0;
            char *rec;
            if (!held && drop && !rec_path(&a, "inproc.rec", &rec)) unlink(rec);
            close(fd); // drops the probe's lock
        }
    }
    arena_reset(&a);
    return held;
}

// The record, if its holder is alive: it still holds the lock (lock_held), and the pid the record
// names runs and started when the record says. A record of any state that fails either is stale:
// ignored, and removed when the lock is free (the next holder writes its own).
static bool read_record(Inproc *out) {
    Arena a = {0};
    char *path, *text = NULL;
    size_t len = 0;
    bool ok = false;
    if (!rec_path(&a, "inproc.rec", &path) && !read_file(path, 8192, &text, &len)) {
        Str s = str_n(text, len), line[5] = {0};
        for (size_t k = 0, i = 0; k < 5 && i <= s.len; k++) {
            ptrdiff_t nl = str_find_char(s, '\n', i);
            size_t j = nl >= 0 ? (size_t)nl : s.len;
            line[k] = str_sub(s, i, j);
            i = j + 1;
        }
        Str t;
        size_t pos = 0;
        uint64_t v, start = 0;
        int32_t pid;
        *out = (Inproc){.state = INPROC_CLOSING};
        if (next_tok(line[0], &pos, &t) && parse_i32(t, &pid) && next_tok(line[0], &pos, &t)) {
            out->pid = pid;
            for (int k = 0; k < 5; k++)
                if (str_eqz(t, state_names[k])) out->state = (InprocState)k;
            if (next_tok(line[0], &pos, &t) && !parse_u64(t, &v) && v <= UINT32_MAX) out->ctx = (uint32_t)v;
            if (next_tok(line[0], &pos, &t)) snprintf(out->holder, sizeof out->holder, "%.*s", (int)t.len, t.ptr);
            if (next_tok(line[0], &pos, &t)) (void)parse_u64(t, &start);
            snprintf(out->token, sizeof out->token, "%.*s", (int)line[1].len, line[1].ptr);
            snprintf(out->model, sizeof out->model, "%.*s", (int)line[2].len, line[2].ptr);
            snprintf(out->dir, sizeof out->dir, "%.*s", (int)line[3].len, line[3].ptr);
            snprintf(out->sock, sizeof out->sock, "%.*s", (int)line[4].len, line[4].ptr);
            ok = start && proc_start_us(pid) == start;
        }
        free(text);
        if (!lock_held(true)) ok = false;
    }
    arena_reset(&a);
    return ok;
}

bool instance_inproc_running(int *pid) {
    Inproc in;
    if (!read_record(&in)) return false;
    if (pid) *pid = in.pid;
    return true;
}

bool instance_inproc_info(Inproc *out) { return read_record(out); }

static Err inproc_clients_dir(Arena *a, char **out) {
    TRY(rec_path(a, "inproc-clients", out));
    TRY(mkdir_p(*out));
    chmod(*out, 0700);
    return OK;
}

// The live entries of holder id (NULL: any), others' and dead ones removed.
static size_t count_clients(const char *holder) {
    Arena a = {0};
    char *dir_path;
    size_t n = 0;
    if (inproc_clients_dir(&a, &dir_path)) goto done;
    DIR *dir = opendir(dir_path);
    if (!dir) goto done;
    struct dirent *e;
    while ((e = readdir(dir))) {
        int32_t pid;
        if (!parse_i32(str(e->d_name), &pid)) continue;
        char *p = path_join(&a, dir_path, e->d_name), *text = NULL;
        size_t len = 0;
        // "<holder id> <the session's start time>": a session of this holder, alive, not a recycled pid
        bool mine = false;
        if (!read_file(p, 128, &text, &len) && holder) {
            Str t = str_trim(str_n(text, len));
            ptrdiff_t sp = str_find_char(t, ' ', 0);
            uint64_t start = 0;
            mine = sp > 0 && str_eq(str_sub(t, 0, (size_t)sp), str(holder)) && !parse_u64(str_sub(t, (size_t)sp + 1, t.len), &start) &&
                   start && proc_start_us(pid) == start;
        }
        free(text);
        if (mine) n++;
        else unlink(p); // a session that died, or one of an earlier holder
    }
    closedir(dir);
done:
    arena_reset(&a);
    return n;
}

bool instance_inproc_join(const Inproc *h) {
    int lk = reg_lock();
    Inproc now;
    bool ok = instance_inproc_info(&now) && now.pid == h->pid && now.state == INPROC_SERVING && !strcmp(now.holder, h->holder);
    if (ok) {
        Arena a = {0};
        char *dir;
        ok = !inproc_clients_dir(&a, &dir);
        if (ok) {
            // written whole, then renamed into place (a count never sees half an entry)
            char *p = arena_printf(&a, "%s/%d", dir, (int)getpid()), *tmp = arena_printf(&a, "%s/.%d.tmp", dir, (int)getpid());
            unlink(tmp);
            int fd = open(tmp, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
            char ent[96];
            int el = snprintf(ent, sizeof ent, "%s %llu", h->holder, (unsigned long long)proc_start_us(getpid()));
            ok = fd >= 0 && !pwrite_all(fd, ent, (size_t)el, 0);
            if (fd >= 0) close(fd);
            ok = ok && rename(tmp, p) == 0;
            if (!ok) unlink(tmp);
        }
        arena_reset(&a);
    }
    reg_unlock(lk);
    return ok;
}

void instance_inproc_leave(void) {
    int lk = reg_lock();
    Arena a = {0};
    char *dir;
    if (!inproc_clients_dir(&a, &dir)) unlink(arena_printf(&a, "%s/%d", dir, (int)getpid()));
    arena_reset(&a);
    reg_unlock(lk);
}

size_t instance_inproc_clients(void) {
    if (!g_holder.on) return 0;
    int lk = reg_lock(); // no entry is written or removed meanwhile
    size_t n = count_clients(g_holder.id);
    reg_unlock(lk);
    return n;
}

size_t instance_inproc_withdraw(bool force) {
    if (!g_holder.on) return 0;
    int lk = reg_lock();
    size_t n = force ? 0 : count_clients(g_holder.id);
    if (!n) {
        // the record says closing until the lock goes with this process: no session joins, and none
        // tries to load the model while this one still holds it (a restart keeps its own record)
        if (!g_holder.restarting) (void)instance_inproc_publish(INPROC_CLOSING, 0, "", "");
        if (force) count_clients(g_holder.id);
    }
    reg_unlock(lk);
    return n;
}
