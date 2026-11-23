// One model for several Saetta Codes (`saetta code --pack`, `saetta start code`): the first
// loads the engine in its process and also serves the server's HTTP API on a Unix socket,
// ~/.saetta/run/inproc.sock (0600 in a 0700 folder: no other user connects; src/server.h
// server_engine_connection), named in the record ~/.saetta/run/inproc.rec with a token
// (instance_inproc_publish); every later one finds that record and talks to the engine over
// HTTP on that socket (src/unix_http.h) with the same model description, so its requests and the tokens the engine reads are
// those of the first. The engine routes each session to its own conversation (src/server.c,
// up to SAETTA_CONVERSATIONS) and takes one request at a time. When the first exits while
// others still use it, it keeps serving until the last one leaves, as `saetta start` keeps
// its server.
#ifndef SAETTA_CODE_SHARE_H
#define SAETTA_CODE_SHARE_H

#include "code.h"

// The model as Saetta Code describes saetta's engine (launch.c pi_setup's provider): id,
// context, the compat flags; base_url "in-process" (engine set by the caller) or the URL.
void code_share_describe(Arena *a, CodeModel *m, const char *id, size_t ctx, const char *base_url);

// Before loading the pack in dir: a Saetta Code that already runs the engine here. 1: joined
// (*m talks to it over HTTP with its token, this process registered), 0: none (load the engine;
// also when the holder exits meanwhile), -1: refused (another pack, or a holder that keeps the
// model to itself; the reason printed). Waits while the holder loads.
int code_share_join(Arena *a, const char *dir, CodeModel *m);
bool code_share_joined(void);
// Before each request of a joined session (CodeModel.verify): 1 when the holder joined is still
// there (its record, alive) and the program on its socket proves it knows the token (an HMAC of a
// fresh nonce: GET /saetta/whoami, without the token); 0 when it is gone or another program has
// the socket (the request is not sent); 2 when it is there but busy (503). The window left: the
// holder dying between this check and the request, which then fails as a lost connection.
int code_share_verify(const CodeModel *m);
// Messages while the interactive screen is up go to fn (NULL: stderr).
void code_share_set_notice(void (*fn)(const char *msg));
// Called around each wait of a join for a loading or restarting holder (the rpc mode lets its lock
// go there); NULL: none.
void code_share_set_wait_hooks(void (*begin)(void), void (*end)(void));
// code_share_join, waiting for a loading or restarting holder until deadline_ns at most, and
// returning -1 as soon as *abort is set (may be NULL).
int code_share_join_until(Arena *a, const char *dir, CodeModel *m, uint64_t deadline_ns, volatile int *abort);
// A joined session whose holder stopped answering: leaves it and joins again (another session may
// hold the model by now), as code_share_join_until. 0: the caller loads the model.
int code_share_rejoin(Arena *a, CodeModel *m, uint64_t deadline_ns, volatile int *abort);

// The engine of this process loaded (m->engine set). Interactive: serve it to other Saetta Codes
// (a Unix socket, a token; published in the record). Otherwise (print, json, rpc) the model is
// kept to this process, which then never waits for others at its exit.
Err code_share_serve(const CodeModel *m, bool interactive);

// A stand-in for the engine in the tests (no GPU): with $SAETTA_CODE_SHARE_STUB set to an
// OpenAI-compatible server's address (http://127.0.0.1:PORT), --pack takes the in-process lock
// without loading anything, talks to that server itself, and relays the connections of the
// sessions that join it there (checking their token as the engine does). True when the stub is on
// (*err set on failure).
bool code_share_stub(Arena *a, const char *dir, CodeModel *m, bool interactive, Err *err);

// The holder: the sessions that use its engine now. code_share_holding: this process holds the model.
size_t code_share_clients(void);
bool code_share_holding(void);

// At exit: a joined session unregisters; the holder withdraws its record and, while others still
// use its engine, serves them (Ctrl-C stops at once), then stops accepting and waits for the
// connections in flight, before the engine is freed. now: at once (/model switching away, the
// others confirmed): their connections are cut and they load or join the model again.
void code_share_end(bool now);

#endif
