// The editor's kill ring and undo stack (code_tui_ring.c).
// Adapted from pi (MIT, Copyright (c) 2025 Mario Zechner; github.com/earendil-works/pi v1.5.2,
// b4afddd): packages/tui/src/kill-ring.ts (KillRing), undo-stack.ts (UndoStack).
#ifndef SAETTA_CODE_TUI_RING_H
#define SAETTA_CODE_TUI_RING_H

#include "base.h"

// KillRing: killed text, the most recent last.
typedef struct {
    char **v;
    size_t n, cap;
} KillRing;
// push: text added, or (accumulate) merged with the most recent entry, before it (prepend: a
// backward deletion) or after it.
void kill_ring_push(KillRing *r, Str text, bool prepend, bool accumulate);
const char *kill_ring_peek(const KillRing *r); // NULL when empty
void kill_ring_rotate(KillRing *r);            // the last entry moved to the front (yank-pop)
void kill_ring_free(KillRing *r);

// UndoStack of editor snapshots (EditorSnapshot: the state, the pastes and their counter).
typedef struct {
    char *text;     // the lines joined with "\n"
    size_t cl, cc;  // cursor line and byte column
    char **pastes;  // the paste contents by id - 1
    size_t npastes;
} UndoSnap;
typedef struct {
    UndoSnap *v;
    size_t n, cap;
} UndoStack;
void undo_push(UndoStack *s, Str text, size_t cl, size_t cc, char *const *pastes, size_t npastes); // copies
bool undo_pop(UndoStack *s, UndoSnap *out); // the caller owns out (undo_snap_free)
void undo_clear(UndoStack *s);
void undo_snap_free(UndoSnap *s);

#endif
