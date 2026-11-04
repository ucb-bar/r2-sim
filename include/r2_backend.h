/* r2_backend.h: the interface of a functional backend plug-in of r2-sim.
 *
 * A plug-in is a shared library that exports one function,
 * r2_backend_open. r2-sim opens the library with dlopen when the
 * configuration selects the backend, calls r2_backend_open, checks abi,
 * then calls configure once, apply once per task in the order of seq, and
 * close after the last task. All calls come from one thread at a time.
 */
#ifndef R2_BACKEND_H
#define R2_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#define R2_BACKEND_ABI 1

/* One memory block of the store. */
typedef struct {
    uint64_t base;    /* the global byte address of the first byte */
    uint64_t size;    /* the bytes of the block */
    uint8_t *ptr;     /* the bytes of the block in the r2-sim process */
    const char *name; /* for example "root.gpc0.hbm" */
} r2_block;

/* A global byte range [base, bound). */
typedef struct {
    uint64_t base;
    uint64_t bound;
} r2_range;

/* The store: blocks sorted by base, with no overlap. A write into the
   sync-only space has no effect. */
typedef struct {
    const r2_block *blocks;
    size_t count;
    uint64_t granule;          /* the Neutrino granule in bytes */
    const r2_range *sync_only; /* the sync-only space */
    size_t sync_only_count;
} r2_memory;

/* The bytes [addr, addr + len), or NULL when they are not in one block. */
static inline uint8_t *r2_at(const r2_memory *m, uint64_t addr, uint64_t len) {
    size_t lo = 0, hi = m->count;
    while (lo < hi) { /* the first block with base > addr */
        size_t mid = lo + (hi - lo) / 2;
        if (m->blocks[mid].base <= addr) lo = mid + 1; else hi = mid;
    }
    if (lo == 0) return NULL;
    const r2_block *b = &m->blocks[lo - 1];
    uint64_t off = addr - b->base;
    if (b->ptr == NULL || off > b->size || len > b->size - off) return NULL;
    return b->ptr + off;
}

/* One task. */
typedef struct {
    uint64_t seq;         /* 1, 2, 3, ... in the order of emission */
    uint64_t cycle;       /* the cycle of the issue */
    uint64_t gid;         /* the task id */
    uint32_t neutrino;    /* the Neutrino that owns the task */
    uint32_t pe;          /* the executing PE: an index of the PE table;
                             UINT32_MAX for a task without a PE */
    const uint8_t *desc;  /* the descriptor, Neutrino spec layout (ABI 1 of
                             the RV programs: header, keys, arguments) */
    size_t desc_bytes;
    const uint8_t *args;  /* the argument block of a Muon task, or NULL */
    size_t args_bytes;
} r2_task;

typedef struct {
    uint32_t abi;         /* R2_BACKEND_ABI */
    void *self;
    /* Before the run. pes: the PE names by index. config: the backend's
       entry of the configuration, as JSON, with the key kernels: the Muon
       kernel registry (per kernel id, name, bytes, flags, and fields with
       name, type and offset). Returns 0, or a value that is not 0 and a
       message in err. */
    int (*configure)(void *self, const r2_memory *memory,
                     const char *const *pes, size_t pe_count,
                     const char *config, char *err, size_t err_len);
    /* One task, in the order of seq. Returns 0, or a value that is not 0
       and a message in err. */
    int (*apply)(void *self, const r2_task *task, char *err, size_t err_len);
    /* After the last task. It frees the backend. The blocks of the store
       are still mapped during close: a backend can write back its state. */
    void (*close)(void *self);
} r2_backend;

/* The one exported symbol. NULL and a message in err on an error. */
const r2_backend *r2_backend_open(char *err, size_t err_len);

#endif
