"""The descriptor bytes of a task (ABI version 3 of the RV programs, as the
Neutrino reads them from its data scratch) as the fields of `ops`.

Layout, little-endian: a 24-byte header (opcode u16, pe u16, flags u16,
ndeps u16, dst base u64 in granules, bound u32 in granules, mask u32),
`ndeps` keys of 8 bytes, then the arguments of the opcode. Addresses in the
arguments are byte addresses.

- DMA (32 B): src u64, dst u64, row_bytes u32, rows u32 (1 or more),
  src_stride u32, dst_stride u32; the bytes are rows x row_bytes.
- Matmul (40 B): a u64, b u64, a2 i32 and b2 i32 (byte offsets of the high
  halves from a and b; 0: not split), lda, ldb, ldc, m, n, k u16, types u16
  (A | B << 4 | C << 8), flags u16 (ACCUMULATE 1, KEEP 2, TRANSPOSE_B 4).
  The output tile is the Atlas's (the backend adds it).
- Vector (16 B): kernel u16, args_bytes u16, reserved u32, args u64 (the
  block itself comes with the task)."""

from __future__ import annotations

import struct

from r2sim.func.memory import FuncError

HEADER = 24
KINDS = ["join", "spawn", "dma", "matmul", "vector", "imem_copy"]
ARGS_BYTES = [0, 16, 32, 40, 16, 24]
DTYPES = ["bf16", "f32", "fp8", "mxfp4"]
MM_ACCUMULATE, MM_KEEP, MM_TRANSPOSE_B = 1, 2, 4


def decode(data: bytes, granule: int, pe: int | None) -> dict:
    """The fields of the descriptor `data`; `pe`: the global index of the
    task's PE (the key of the Atlas accumulators)."""
    if len(data) < HEADER:
        raise FuncError(f"a descriptor of {len(data)} bytes, shorter than its header")
    opcode, _, _, ndeps, base, bound, mask = struct.unpack_from("<HHHHQII", data, 0)
    if opcode >= len(KINDS):
        raise FuncError(f"unknown opcode {opcode}")
    at = HEADER + 8 * ndeps
    if len(data) < at + ARGS_BYTES[opcode]:
        raise FuncError(f"a descriptor of opcode {opcode} with {len(data)} bytes")
    kind = KINDS[opcode]
    d = {"kind": kind, "pe": pe, "dst": [base * granule, (base + bound) * granule], "mask": mask}
    if kind == "dma":
        src, to, row_bytes, rows, src_stride, dst_stride = struct.unpack_from(
            "<QQIIII", data, at)
        if rows == 0:
            raise FuncError("a DMA of 0 rows")
        d.update(src=src, to=to, bytes=rows * row_bytes, rows=rows, row_bytes=row_bytes,
                 src_stride=src_stride, dst_stride=dst_stride)
    elif kind == "matmul":
        a, b, a2, b2, lda, ldb, ldc, m, n, k, types, flags = struct.unpack_from(
            "<QQiiHHHHHHHH", data, at)
        dtype = lambda c: DTYPES[c] if c < len(DTYPES) else f"type {c}"  # noqa: E731
        d.update(
            a=a, b=b, a2=a + a2 if a2 else None, b2=b + b2 if b2 else None, lda=lda, ldb=ldb,
            ldc=ldc, m=m, n=n, k=k, a_dtype=dtype(types & 0xF), b_dtype=dtype((types >> 4) & 0xF),
            out_dtype=dtype((types >> 8) & 0xF),
            accumulate=bool(flags & MM_ACCUMULATE), keep=bool(flags & MM_KEEP),
            transpose_b=bool(flags & MM_TRANSPOSE_B),
        )
    elif kind == "vector":
        kernel, args_bytes, _, args = struct.unpack_from("<HHIQ", data, at)
        d.update(kernel=kernel, args_bytes=args_bytes, args=args)
    return d
