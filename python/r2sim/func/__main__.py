"""`python -m r2sim.func`: the functional plane as a child process of r2sim
(`r2sim run ... --functional python`). It reads frames on stdin, answers on
stdout, and exits at FINISH or at end of file on stdin."""

from __future__ import annotations

import sys

from r2sim.func import protocol as p
from r2sim.func.memory import FuncError
from r2sim.func.plane import Plane


def serve(inp, out) -> int:
    plane = Plane()
    while True:
        frame = p.read_frame(inp)
        if frame is None:
            return 0
        kind, obj, blob = frame
        seq = obj.get("seq", 0) if isinstance(obj, dict) else 0
        try:
            if kind == p.FINISH:
                return 0
            if kind == p.HELLO:
                if obj.get("protocol") != p.PROTOCOL:
                    raise FuncError(f"protocol {obj.get('protocol')} (this plane speaks {p.PROTOCOL})")
                p.write_frame(out, p.OK, {"protocol": p.PROTOCOL, "backend": "r2sim.func"})
            elif kind == p.CONFIGURE:
                plane.configure(obj["memories"], obj["sync_only"], obj["granule"])
                p.write_frame(out, p.OK)
            elif kind == p.TASK:
                plane.task(obj["desc"])
                p.write_frame(out, p.OK)
            elif kind == p.STORE:
                plane.store(obj["addr"], blob)
                p.write_frame(out, p.OK)
            elif kind == p.WRITE:
                plane.memory.write(obj["addr"], blob)
                p.write_frame(out, p.OK)
            elif kind == p.READ:
                p.write_frame(out, p.DATA, None, plane.memory.read(obj["addr"], obj["len"]))
            else:
                raise FuncError(f"unknown frame type {kind}")
        except FuncError as e:
            p.write_frame(out, p.ERROR, {"seq": seq, "msg": str(e)})
        except Exception as e:  # noqa: BLE001 - any failure of an op stops the run with its seq
            p.write_frame(out, p.ERROR, {"seq": seq, "msg": f"{type(e).__name__}: {e}"})


def main() -> int:
    return serve(sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    sys.exit(main())
