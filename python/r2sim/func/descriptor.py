"""The descriptor bytes of a task (ABI version 1 of the RV programs, as the
Neutrino reads them from its data scratch) as the fields of `ops`.

Layout, little-endian: a 24-byte header (opcode u16, pe u16, flags u16,
ndeps u16, dst base u64 in granules, bound u32 in granules, mask u32),
`ndeps` keys of 8 bytes, then the arguments of the opcode. Addresses in the
arguments are byte addresses."""

from __future__ import annotations

import struct

from r2sim.func.memory import FuncError

HEADER = 24
KINDS = ["join", "spawn", "dma", "matmul", "vector", "imem_copy"]
ARGS_BYTES = [0, 16, 32, 72, 32, 24]
DTYPES = ["bf16", "f32", "fp8", "mxfp4"]
MM_ACCUMULATE, MM_KEEP = 1, 2


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
        src, to, n, tile = struct.unpack_from("<QQQQ", data, at)
        d.update(src=src, to=to, bytes=n, tile_bytes=tile)
    elif kind == "matmul":
        a, b, a2, b2, lda, ldb, ldc, m, n, k, types, flags, t0, t1 = struct.unpack_from(
            "<QQQQIIIIIIIIHH", data, at)
        dtype = lambda c: DTYPES[c] if c < len(DTYPES) else f"type {c}"  # noqa: E731
        d.update(
            a=a, b=b, a2=a2 or None, b2=b2 or None, lda=lda, ldb=ldb, ldc=ldc, m=m, n=n, k=k,
            a_dtype=dtype(types & 0xFF), b_dtype=dtype((types >> 8) & 0xFF),
            out_dtype=dtype((types >> 16) & 0xFF),
            accumulate=bool(flags & MM_ACCUMULATE), keep=bool(flags & MM_KEEP), tile=[t0, t1],
        )
    elif kind == "vector":
        ops, bytes_in, bytes_out, src = struct.unpack_from("<QQQQ", data, at)
        d.update(ops=ops, bytes_in=bytes_in, bytes_out=bytes_out, src=src or None)
    return d
