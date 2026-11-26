// Conversation state on the SSD: what a resumed conversation needs to continue without
// reading its history again (the KV cache rows, the recurrent state, the token ids), kept in
// ~/.saetta/state/<pack identity>/<state shape>/ by Saetta Code and read back when a new
// process gets a prompt that extends a saved state. libc only: the model describes its buffers
// (StLayout) and the server decides when to save and load.
//
// The shape (the caller's hash of the engine build, the state format and the buffers' shapes)
// names the folder, so states and chunks of another build or layout never share a file with
// this one's; every file also carries it and is checked before it is read or reused.
//
// Files (every number little endian, every section at a multiple of ST_PAGE; folders 0700,
// files 0600):
//   chunks/<32 hex>.kv  rows [c * ST_CHUNK, (c + 1) * ST_CHUNK) of every plane of every region,
//                       named by its key: the sha256 of the lineage that wrote it (one
//                       computation) and the token ids before its end. A state lists the key of
//                       each chunk it uses: a conversation restored from a state reads that
//                       state's chunks below the restore point by their keys and writes every
//                       chunk past it under a lineage of its own, so a state never mixes rows
//                       computed elsewhere with its own rows and snapshot
//   states/<32 hex>.st  one state, named by the sha256 of its lineage and its n ids: a header
//                       page, the ids, the chunk keys it uses, the tail (rows past its last full
//                       chunk, in the chunk layout) and one recurrent snapshot (the restore point
//                       at n), with the caller's metadata (lineage, scalars, an opaque blob)
// Every section carries CRC32C values (four per slice, computed in parallel), checked at every
// load; the header pages carry their own. A file is written under a temporary name (O_EXCL,
// O_NOFOLLOW), synced (F_FULLFSYNC) and renamed, so a crash leaves either the old state or the
// new one; temporary files of dead processes are removed when the store opens. Writes and
// reads bypass the page cache (F_NOCACHE): a buffered 2-4 GB write would push the mapped
// weights out of memory.
//
// Eviction (after each save and when the store opens, under the tree's flock root/.lock, which
// writes take too): across the whole tree (every pack's and shape's store), files unused for
// ST_MAX_AGE_S go; over ST_MAX_BYTES in all, this pack's other shapes go first (older engine
// builds, oldest first), then the oldest of the other packs' files and this store's states, each
// state with the chunks no remaining state uses. A load
// holds a shared flock on its state file, so no process removes it (or its chunks) meanwhile; a
// state that cannot be read in a scan keeps every chunk that round. A state under ST_MIN_TOKENS
// tokens is not written.
#ifndef SAETTA_STATE_STORE_H
#define SAETTA_STATE_STORE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "base.h"

#define ST_PAGE ((size_t)16384)
#define ST_CHUNK ((size_t)4096)
#define ST_MIN_TOKENS ((size_t)4096)
#define ST_MAX_BYTES ((uint64_t)20 << 30)
#define ST_MAX_AGE_S ((int64_t)7 * 24 * 3600)
#define ST_MAX_REGIONS 4
// recurrent buffers and the few rows the model reads at n beyond the rows below it
#define ST_MAX_REC 8
// slices of a chunk: one per plane of every region
#define ST_MAX_SLICES 128
#define ST_EXTRA_MAX 1024
// The file format's version (bump when a header or a section changes).
#define ST_FORMAT 3u

// A buffer of per-token rows in planes: plane p's row r at base + p * plane_stride + r * row_bytes.
// per_row tokens make one row (1 for KV and indexer keys; BLOCK for pooled block keys, of which
// only full blocks are written, so a state at n keeps n / per_row rows).
typedef struct {
    uint8_t *base;
    size_t planes, plane_stride, row_bytes, per_row;
} StRegion;

// What a conversation's state is made of: its row buffers (at their current capacity) and the
// buffers of one restore point's recurrent state.
typedef struct {
    size_t nreg, nrec;
    StRegion reg[ST_MAX_REGIONS];
    uint8_t *rec[ST_MAX_REC];
    size_t rec_bytes[ST_MAX_REC];
} StLayout;

// What a state carries besides rows and recurrent buffers: the lineage that writes its new chunks
// (the conversation's own: picked at its first save, never a restored state's), the restore
// point's scalars (Ckpt.prev, Ckpt.since_eos), and an opaque blob
// (the engine's speculation controller).
typedef struct {
    uint8_t lineage[16];
    uint32_t prev[2];
    uint64_t since_eos;
    size_t extra_len;
    uint8_t extra[ST_EXTRA_MAX];
} StMeta;

typedef struct StateStore StateStore;

// CRC32C (Castagnoli, reflected 0x82F63B78) with the ARM CRC instructions: crc32c(0, p, n) is
// the checksum of p[0..n); crc32c(c, q, m) continues c over q.
uint32_t crc32c(uint32_t crc, const void *p, size_t n);
// The checksums of the four parts of p[0..n) (each of (n / 4) & ~7 bytes, the last one also
// the rest), as four independent chains in one pass.
void crc32c_quad(const void *p, size_t n, uint32_t out[4]);

// The store in root/<16 hex of ident>/<16 hex of shape>/ (created when missing, 0700). ident:
// the pack's identity; shape: the identity of the engine build, the state format and the
// buffers' shapes. Errors: "StateDirFailed", "StateWriterFailed" (no writer thread).
Err state_store_open(const char *root, const uint8_t ident[32], const uint8_t shape[32], StateStore **out);
// Waits for the writer, then frees the store.
void state_store_close(StateStore *st);
// The store's folder.
const char *state_store_dir(const StateStore *st);

// Saves the state of a conversation at n tokens (ids[0..n)) with meta. Chunks c < ninh are the
// ones it was restored from, named by inh[c] (rows identical to theirs: reused when present,
// else written under that key); the others are keyed by meta->lineage. The recurrent snapshot
// (lay.rec), the ids and meta are copied now, the rows below n are read later by the writer
// thread from lay.reg, which must not change below n until the write ends (state_store_fence).
// hold[0..nhold) are released by release(hold[i]) on the writer when it is done (the caller
// retains the row buffers, so they outlive a reallocation). Waits for a previous write first.
// Returns false when nothing is written (n under ST_MIN_TOKENS, disk short, this state is
// already there and sound); release is then called at once.
bool state_store_save(StateStore *st, const StLayout *lay, const uint32_t *ids, size_t n, const StMeta *meta,
                      const uint8_t (*inh)[32], size_t ninh, void *const *hold, size_t nhold, void (*release)(void *));
// Before rows from position from on of the buffer at base are rewritten: waits for a write
// that reads that buffer below its n past from.
void state_store_fence(StateStore *st, const void *base, size_t from);
// Waits for the writer.
void state_store_wait(StateStore *st);

// The longest saved state whose ids are a prefix of prompt[0..np) with more than min_n and fewer
// than np tokens: its length (0 if none), and its name in name (33 bytes).
size_t state_store_find(StateStore *st, const uint32_t *prompt, size_t np, size_t min_n, char name[33]);
// Reads state name into lay (its regions with room for its n tokens) and *meta, with 4 threads,
// checking every header, size and CRC and that its ids are prompt[0..n) with n < np; *bytes
// gets the bytes read, *keys (malloc'd, the caller frees it) its chunks' keys, *nkeys their
// count (n / ST_CHUNK). *touched: whether lay's buffers were written (a failure before the
// reads leaves them as they were). A state that fails a check is deleted, and so is a chunk
// that fails one. Errors: "StateMissing", "StateCorrupt", "StateMismatch", "StateTooLarge".
Err state_store_load(StateStore *st, const char *name, const uint32_t *prompt, size_t np, const StLayout *lay, size_t *n,
                     StMeta *meta, uint8_t (**keys)[32], size_t *nkeys, uint64_t *bytes, bool *touched);

// For tests: the store's total bytes on disk and its states' count (after a scan).
void state_store_usage(StateStore *st, uint64_t *bytes, size_t *states);
// Eviction with explicit limits (the defaults are ST_MAX_BYTES, ST_MAX_AGE_S), now in seconds.
void state_store_gc(StateStore *st, uint64_t max_bytes, int64_t max_age_s, int64_t now);

#endif
