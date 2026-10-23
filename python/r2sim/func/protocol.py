"""Frames of the process-backend protocol (version 1; see the Rust module
func::process): u32 n, then u8 type, u32 j, j bytes of JSON and n - 5 - j
raw bytes, all little-endian."""

from __future__ import annotations

import json
import struct
from typing import BinaryIO

PROTOCOL = 1
HELLO, CONFIGURE, TASK, STORE, WRITE, READ, FINISH = 1, 2, 3, 4, 5, 6, 7
OK, DATA, ERROR = 64, 65, 66


def read_frame(f: BinaryIO) -> tuple[int, object, bytes] | None:
    """The next frame, or None at end of file."""
    head = f.read(4)
    if len(head) < 4:
        return None
    (n,) = struct.unpack("<I", head)
    body = f.read(n)
    if len(body) < n or n < 5:
        return None
    kind = body[0]
    (j,) = struct.unpack("<I", body[1:5])
    obj = json.loads(body[5:5 + j]) if j else None
    return kind, obj, bytes(body[5 + j:])


def write_frame(f: BinaryIO, kind: int, obj: object = None, blob: bytes = b"") -> None:
    j = b"" if obj is None else json.dumps(obj).encode()
    f.write(struct.pack("<IBI", 1 + 4 + len(j) + len(blob), kind, len(j)))
    f.write(j)
    f.write(blob)
    f.flush()
