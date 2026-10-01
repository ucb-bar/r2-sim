"""Write Trace backend files for the AdaMoE program.

The trace file format (little-endian), read by `func::load_trace`:
the magic b"R2TRACE1", a u64 record count, then per record a u64 label,
a u32 length and that many bytes.

Router records are keyed by the program's label for (step, layer):
    label = (1 << 56) | (step << 48) | (layer << 40)
and hold one expert id (uint8) per suffix token.

Usage:
    python -m r2sim.trace_convert --synthetic --seed 7 -o trace.bin
    python -m r2sim.trace_convert --routing FILE --frame N -o trace.bin

The --routing input (the AdaMoE routing trace of workload 2) is not wired
yet: its location and on-disk layout are still to be confirmed.
"""

import argparse
import struct
import sys

import numpy as np

KIND_ROUTER = 1
MAGIC = b"R2TRACE1"


def router_label(step: int, layer: int) -> int:
    return (KIND_ROUTER << 56) | ((step & 0xFF) << 48) | ((layer & 0xFF) << 40)


def write_trace(path: str, records: dict) -> None:
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", len(records)))
        for label in sorted(records):
            data = bytes(records[label])
            f.write(struct.pack("<QI", label, len(data)))
            f.write(data)


def read_trace(path: str) -> dict:
    with open(path, "rb") as f:
        buf = f.read()
    if buf[:8] != MAGIC:
        raise ValueError("not a trace file")
    (count,) = struct.unpack_from("<Q", buf, 8)
    at, out = 16, {}
    for _ in range(count):
        label, n = struct.unpack_from("<QI", buf, at)
        at += 12
        out[label] = buf[at : at + n]
        at += n
    if at != len(buf):
        raise ValueError("trailing bytes")
    return out


def synthetic(steps: int, layers: int, tokens: int, experts: int, seed: int,
              alpha: float = 4.0, step_agree: float = 0.9) -> dict:
    """Router records with the measured statistics (workload 2): per layer a
    Dirichlet(alpha) share over the experts (alpha 4: mean max share 0.41 of
    51 tokens, all experts live in 99% of cases; measured 0.41 and 97%), and
    each later denoise step keeping a token's expert with probability
    `step_agree` (measured 0.91). Layers are independent."""
    rng = np.random.default_rng(seed)
    out = {}
    for layer in range(layers):
        shares = rng.dirichlet([alpha] * experts)
        prev = None
        for step in range(steps):
            fresh = rng.choice(experts, size=tokens, p=shares).astype(np.uint8)
            if prev is None:
                ids = fresh
            else:
                keep = rng.random(tokens) < step_agree
                ids = np.where(keep, prev, fresh).astype(np.uint8)
            out[router_label(step, layer)] = ids.tobytes()
            prev = ids
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--routing", help="AdaMoE routing trace (not wired yet)")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--layers", type=int, default=18)
    ap.add_argument("--tokens", type=int, default=51)
    ap.add_argument("--experts", type=int, default=4)
    a = ap.parse_args(argv)
    if a.synthetic == (a.routing is not None):
        ap.error("give exactly one of --synthetic and --routing")
    if a.routing is not None:
        print("r2sim.trace_convert: --routing is not wired yet "
              "(trace location and layout pending)", file=sys.stderr)
        return 2
    records = synthetic(a.steps, a.layers, a.tokens, a.experts, a.seed)
    write_trace(a.output, records)
    print(f"wrote {len(records)} records to {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
