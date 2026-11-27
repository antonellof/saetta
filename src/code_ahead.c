// Prefill ahead of the request in Saetta Code (see code_ahead.h).
#include "code_ahead.h"

#include <pthread.h>
#include <stdlib.h>

#include "server.h"

struct CodeAhead {
    Engine *e;
    pthread_t th;
    bool started, quit, off;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    // the snapshot not yet read (NULL: none)
    char *body;
    size_t len;
    char session[96], id[128];
    volatile int stop;
    size_t total;
};

static void *ahead_main(void *arg) {
    CodeAhead *ah = arg;
    pthread_mutex_lock(&ah->mu);
    for (;;) {
        while (!ah->body && !ah->quit) pthread_cond_wait(&ah->cv, &ah->mu);
        if (ah->quit) break;
        char *body = ah->body;
        size_t len = ah->len;
        char session[96], id[128];
        memcpy(session, ah->session, sizeof session);
        memcpy(id, ah->id, sizeof id);
        ah->body = NULL;
        // this read's stop flag: set by a cancel from now on (a post does not clear it for a read
        // already under way, which then stops at its next chunk)
        ah->stop = 0;
        pthread_mutex_unlock(&ah->mu);
        size_t got = 0;
        Err err = server_engine_ahead(ah->e, str_n(body, len), session, id, &ah->stop, &got);
        free(body);
        pthread_mutex_lock(&ah->mu);
        ah->total += got;
        if (err) {
            // the model reads nothing ahead (Inkling 2 Mini): no more posts
            dbg("saetta code: no prefill ahead (%s)\n", err);
            ah->off = true;
        }
    }
    pthread_mutex_unlock(&ah->mu);
    return NULL;
}

CodeAhead *code_ahead_new(Engine *e) {
    const char *v = getenv("SAETTA_AHEAD");
    if (!e || (v && v[0] == '0')) return NULL;
    CodeAhead *ah = xcalloc(1, sizeof *ah);
    ah->e = e;
    pthread_mutex_init(&ah->mu, NULL);
    pthread_cond_init(&ah->cv, NULL);
    return ah;
}

void code_ahead_post(CodeAhead *ah, Str body, const char *session, const char *partial_id) {
    if (!ah || !session || !partial_id || strlen(session) >= sizeof ah->session || strlen(partial_id) >= sizeof ah->id) return;
    pthread_mutex_lock(&ah->mu);
    if (ah->off || ah->quit) {
        pthread_mutex_unlock(&ah->mu);
        return;
    }
    if (!ah->started) {
        ah->started = pthread_create(&ah->th, NULL, ahead_main, ah) == 0;
        if (!ah->started) ah->off = true;
    }
    free(ah->body);
    ah->body = xmalloc(body.len + 1);
    memcpy(ah->body, body.ptr, body.len);
    ah->body[body.len] = 0;
    ah->len = body.len;
    snprintf(ah->session, sizeof ah->session, "%s", session);
    snprintf(ah->id, sizeof ah->id, "%s", partial_id);
    pthread_cond_signal(&ah->cv);
    pthread_mutex_unlock(&ah->mu);
}

void code_ahead_cancel(CodeAhead *ah) {
    if (!ah) return;
    pthread_mutex_lock(&ah->mu);
    free(ah->body);
    ah->body = NULL;
    ah->stop = 1;
    pthread_mutex_unlock(&ah->mu);
}

size_t code_ahead_total(CodeAhead *ah) {
    if (!ah) return 0;
    pthread_mutex_lock(&ah->mu);
    size_t t = ah->total;
    pthread_mutex_unlock(&ah->mu);
    return t;
}

void code_ahead_free(CodeAhead *ah) {
    if (!ah) return;
    pthread_mutex_lock(&ah->mu);
    ah->quit = true;
    ah->stop = 1;
    free(ah->body);
    ah->body = NULL;
    pthread_cond_signal(&ah->cv);
    bool started = ah->started;
    pthread_mutex_unlock(&ah->mu);
    if (started) pthread_join(ah->th, NULL);
    pthread_mutex_destroy(&ah->mu);
    pthread_cond_destroy(&ah->cv);
    free(ah);
}
