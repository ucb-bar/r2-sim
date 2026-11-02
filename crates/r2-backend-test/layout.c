/* The sizes and field offsets of the types of r2_backend.h, for the
   layout test of the Rust types. */
#include <stddef.h>
#include "r2_backend.h"

static const size_t LAYOUT[] = {
    sizeof(r2_block), offsetof(r2_block, base), offsetof(r2_block, size),
    offsetof(r2_block, ptr), offsetof(r2_block, name),
    sizeof(r2_range), offsetof(r2_range, base), offsetof(r2_range, bound),
    sizeof(r2_memory), offsetof(r2_memory, blocks), offsetof(r2_memory, count),
    offsetof(r2_memory, granule), offsetof(r2_memory, sync_only),
    offsetof(r2_memory, sync_only_count),
    sizeof(r2_task), offsetof(r2_task, seq), offsetof(r2_task, cycle),
    offsetof(r2_task, gid), offsetof(r2_task, neutrino), offsetof(r2_task, pe),
    offsetof(r2_task, desc), offsetof(r2_task, desc_bytes),
    offsetof(r2_task, args), offsetof(r2_task, args_bytes),
    sizeof(r2_backend), offsetof(r2_backend, abi), offsetof(r2_backend, self),
    offsetof(r2_backend, configure), offsetof(r2_backend, apply),
    offsetof(r2_backend, close),
};

size_t r2_test_layout(size_t *out, size_t n) {
    size_t count = sizeof(LAYOUT) / sizeof(LAYOUT[0]);
    for (size_t i = 0; i < count && i < n; i++) out[i] = LAYOUT[i];
    return count;
}

/* r2_at, for the test of the Rust lookup. */
uint8_t *r2_test_at(const r2_memory *m, uint64_t addr, uint64_t len) {
    return r2_at(m, addr, len);
}
