"""A misbehaving functional process for the lifecycle tests of the
process backend. It writes its pid to the file R2SIM_CHILD_PID, answers
HELLO, and then, by its argument:
  hang: never answers CONFIGURE;
  deaf: answers nothing more and does not exit at end of file;
  ok: answers OK to every frame and zeros to a READ;
  fail: as ok, but ERROR to the first TASK."""

import json
import os
import struct
import sys
import time


def read_frame(f):
    head = f.read(4)
    if len(head) < 4:
        return None
    body = f.read(struct.unpack("<I", head)[0])
    j = struct.unpack("<I", body[1:5])[0]
    return body[0], json.loads(body[5:5 + j]) if j else None


def write_frame(f, kind, obj=None):
    j = b"" if obj is None else json.dumps(obj).encode()
    f.write(struct.pack("<IBI", 5 + len(j), kind, len(j)) + j)
    f.flush()


def main():
    mode = sys.argv[1]
    with open(os.environ["R2SIM_CHILD_PID"], "w") as f:
        f.write(str(os.getpid()))
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    read_frame(inp)
    write_frame(out, 64, {"protocol": 1})
    if mode == "hang":
        while True:
            time.sleep(1)
    if mode == "deaf":
        while inp.read(65536):
            pass
        while True:
            time.sleep(1)
    while (frame := read_frame(inp)) is not None:
        kind, obj = frame
        if kind == 7:
            return
        if kind == 3 and mode == "fail":
            write_frame(out, 66, {"seq": obj["seq"], "msg": "refused by the test"})
        elif kind == 6:
            out.write(struct.pack("<IBI", 5 + obj["len"], 65, 0) + bytes(obj["len"]))
            out.flush()
        else:
            write_frame(out, 64)


main()
