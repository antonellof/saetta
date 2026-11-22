// The editor's kill ring and undo stack (see code_tui_ring.h).
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi v1.5.2,
// b4afddd): packages/tui/src/kill-ring.ts, undo-stack.ts.
#include "code_tui_ring.h"

#include <stdlib.h>
#include <string.h>

// KillRing.push
void kill_ring_push(KillRing *r, Str text, bool prepend, bool accumulate) {
    if (!text.len) return;
    if (accumulate && r->n > 0) {
        char *last = r->v[r->n - 1];
        size_t ll = strlen(last);
        char *m = xmalloc(ll + text.len + 1);
        if (prepend) {
            memcpy(m, text.ptr, text.len);
            memcpy(m + text.len, last, ll);
        } else {
            memcpy(m, last, ll);
            memcpy(m + ll, text.ptr, text.len);
        }
        m[ll + text.len] = 0;
        free(last);
        r->v[r->n - 1] = m;
        return;
    }
    if (r->n == r->cap) {
        r->cap = r->cap ? r->cap * 2 : 8;
        r->v = xrealloc(r->v, r->cap * sizeof *r->v);
    }
    r->v[r->n++] = strndup(text.ptr, text.len);
}

// KillRing.peek
const char *kill_ring_peek(const KillRing *r) { return r->n ? r->v[r->n - 1] : NULL; }

// KillRing.rotate
void kill_ring_rotate(KillRing *r) {
    if (r->n <= 1) return;
    char *last = r->v[r->n - 1];
    memmove(r->v + 1, r->v, (r->n - 1) * sizeof *r->v);
    r->v[0] = last;
}

void kill_ring_free(KillRing *r) {
    for (size_t i = 0; i < r->n; i++) free(r->v[i]);
    free(r->v);
    *r = (KillRing){0};
}

// UndoStack.push (structuredClone of the snapshot)
void undo_push(UndoStack *s, Str text, size_t cl, size_t cc, char *const *pastes, size_t npastes) {
    if (s->n == s->cap) {
        s->cap = s->cap ? s->cap * 2 : 16;
        s->v = xrealloc(s->v, s->cap * sizeof *s->v);
    }
    UndoSnap *u = &s->v[s->n++];
    u->text = strndup(text.ptr ? text.ptr : "", text.len);
    u->cl = cl;
    u->cc = cc;
    u->npastes = npastes;
    u->pastes = npastes ? xmalloc(npastes * sizeof *u->pastes) : NULL;
    for (size_t i = 0; i < npastes; i++) u->pastes[i] = strdup(pastes[i]);
}

// UndoStack.pop
bool undo_pop(UndoStack *s, UndoSnap *out) {
    if (!s->n) return false;
    *out = s->v[--s->n];
    return true;
}

void undo_snap_free(UndoSnap *u) {
    free(u->text);
    for (size_t i = 0; i < u->npastes; i++) free(u->pastes[i]);
    free(u->pastes);
    *u = (UndoSnap){0};
}

// UndoStack.clear
void undo_clear(UndoStack *s) {
    for (size_t i = 0; i < s->n; i++) undo_snap_free(&s->v[i]);
    s->n = 0;
}
