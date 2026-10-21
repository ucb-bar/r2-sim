"""Check a raw trace dump (`r2sim run ... --trace-raw DIR`).

The format is r2-vis's `trace-raw.md` (version 1.9; older versions read
as that document says). The checks are the ones of its Section 8, plus
the file layout:

* the manifest's record layouts have the record sizes of the format (tasks:
  112 B with `flags`, 104 B without), every binary file holds whole
  records, and the counts match `summary.json`;
* untracked tasks (version 1.8, `tasks.flags` bit 0): `version`, `queued`
  and `sent` are null, and no dependency record names one as producer;
* destination ranges (version 1.9): `dst_base` and `dst_len` are
  multiples of the run's granule and `dst_len` is at least one granule;
  `dst_mask` is the mask in granules (a 1.8 or older dump reads as all
  ones: plain ranges); the number of masked (not all ones) ranges is
  printed;
* in a complete dump (`filters.partial` false), every gid in chunks,
  deps, charges and programs.spawn has a task record;
* on a resource with a rate, the slots held at any cycle (version 1.7: a
  record holds usage / 65536 slots over [start, start + span), at most
  floor(capacity x 65536) / 65536 in all; 1.1 to 1.6: cycles / span
  lanes, at most `lanes`) fit its capacity; per resource the slot-cycles
  match the `--stats` JSON (1.7: `busy_slot_cycles` within the records'
  spans / 65536; before: `busy_cycles` exactly); a charge's `sub` on a
  resource with `rates` is one of its classes;
* entries (version 1.3): end >= start; per Neutrino and table the open
  entries at any cycle are at most the table's capacity in the manifest;
  in a complete dump every gid has a task record, a pending entry that
  left has the gid of its adopting task, an inbound invoke entry of an
  Invoke ends at its task's invoke and an outbound one (not a
  VersionReset) starts at its task's `queued`; a VersionReset record
  (flags bit 0) in inbound_invokes (version 1.8) or outbound_invokes has
  gid null;
* hash links (version 1.4): the rule's fields are consistent (equal
  range sizes, granule = the lowest bit of any mask, slice_bytes = size /
  2^masks, one drop bit per mask), and every write port (`wport`) has
  the role mem_port with "write" true;
* write channels (version 1.5): every write channel resource (links
  `ab_wr`/`ba_wr`, attachments `in_wr`/`out_wr`, a bus's resource with a
  write role) has the role of its element with "write" true and capacity
  = wr_bytes / beat_bytes x clock (lanes = ceil(wr_bytes / beat_bytes)
  before 1.7); charges on it are writes, and no write
  charge lands on the read channel of an element that has a write
  channel;
* charge owners (version 1.6): every non-null `gid` is a task gid (with a
  task record, in a complete dump) or a program reference (bit 31 set:
  the root program 0xFFFFFFFE, or 0x80000000 | a spawn gid of
  programs.bin); a message's owner fits its kind (Invoke: a task;
  InvokeAck, RetryAck, Reject: their request's owner, a task or the
  program of a VersionReset; Await, AwaitAck, Resolve: a task or a
  program; Start, Return: a spawn task; the others: a program). Null
  owners are counted by message kind or resource role;
* every resource id in the records is in the manifest, every link of the
  topology tables has a resource, and records on unlimited links have
  span >= 1 and usage = 65536 (1.7; span = cycles before);
* with `--reference FULL` (a complete dump of the same run): the dump
  holds exactly the records of FULL that the filters of its manifest keep
  (r2sim spec Section 11.5, trace-raw.md Section 1.1).

Usage, from the repository root in the pixi environment:

    PYTHONPATH=python python -m r2sim.traceraw DIR [--stats STATS.json] [--reference FULL]

Exit status 1 when a check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SIZES = {"tasks": 112, "chunks": 32, "deps": 24, "charges": 32, "programs": 32, "marks": 16,
         "entries": 48}
# The record types added after version 1.0: (stem, version).
SINCE = {"entries": [1, 3]}
TABLES = ["pending", "versions", "inbound_refs", "inbound_invokes", "outbound_deps",
          "outbound_invokes"]
NULL64 = 0xFFFFFFFFFFFFFFFF
VRESET = 1
UNTRACKED = 1
TYPES = {"u8": "<u1", "u16": "<u2", "u32": "<u4", "u64": "<u8"}
NULL32 = 0xFFFFFFFF


def load(dump: Path, manifest: dict) -> dict[str, np.ndarray]:
    """The records of every binary file, as structured arrays."""
    out = {}
    for stem, layout in manifest["records"].items():
        dtype = np.dtype([(name, TYPES[t]) for name, t in layout])
        raw = dump / f"{stem}.bin"
        zst = dump / f"{stem}.bin.zst"
        if raw.exists():
            data = raw.read_bytes()
        elif zst.exists():
            data = unzstd(zst)
        else:
            continue
        if len(data) % dtype.itemsize:
            raise ValueError(f"{stem}: {len(data)} B is not whole records of {dtype.itemsize} B")
        out[stem] = np.frombuffer(data, dtype=dtype)
    return out


def unzstd(path: Path) -> bytes:
    """The decompressed content of a zstd file (the zstandard module, or
    the zstd command)."""
    try:
        import zstandard
    except ImportError:
        import subprocess

        return subprocess.run(["zstd", "-dc", str(path)], check=True,
                              capture_output=True).stdout
    with path.open("rb") as f:
        return zstandard.ZstdDecompressor().stream_reader(f).read()


def strings(dump: Path) -> list[str]:
    return (dump / "strings.txt").read_text().splitlines()


def comparable(stem: str, recs: np.ndarray, names: list[str]) -> list:
    """Records as sortable tuples; a string id becomes its string (the
    ids of two dumps can differ)."""
    rows = recs.tolist()
    if stem in ("programs", "marks"):
        i = recs.dtype.names.index("name")
        rows = [r[:i] + (names[r[i]],) + r[i + 1:] for r in rows]
    return sorted(rows)


class Report:
    def __init__(self) -> None:
        self.failed = 0

    def check(self, ok: bool, what: str) -> None:
        print(("ok    " if ok else "FAIL  ") + what)
        self.failed += not ok


def check_layouts(r: Report, manifest: dict, recs: dict, summary: dict | None) -> None:
    for stem, size in SIZES.items():
        layout = manifest["records"].get(stem)
        if layout is None and manifest["version"] < SINCE.get(stem, [1, 0]):
            continue
        got = sum(np.dtype(TYPES[t]).itemsize for _, t in layout) if layout else None
        if stem == "tasks" and layout and "flags" not in [n for n, _ in layout]:
            size = 104  # 1.7, or 1.8 before untracked tasks
        r.check(got == size, f"{stem} layout is {got} B (format: {size} B)")
    if summary is not None:
        for stem, n in summary["counts"].items():
            have = len(recs[stem]) if stem in recs else 0
            r.check(have == n, f"{stem}: {have} records, summary {n}")


def check_gids(r: Report, manifest: dict, recs: dict) -> None:
    if manifest["filters"]["partial"]:
        print("skip  gid references (partial dump)")
        return
    tasks = recs["tasks"]["gid"]
    r.check(len(np.unique(tasks)) == len(tasks), "one task record per gid")
    known = np.sort(tasks)

    def missing(gids: np.ndarray) -> int:
        # Program references (version 1.6, bit 31 set) are checked with
        # the owners.
        gids = gids[gids < 0x80000000]
        pos = np.searchsorted(known, gids)
        pos = np.minimum(pos, len(known) - 1)
        return int(np.count_nonzero(known[pos] != gids)) if len(known) else len(gids)

    for stem, field in [("chunks", "gid"), ("deps", "consumer"), ("deps", "producer"),
                        ("charges", "gid"), ("programs", "spawn")]:
        if stem in recs:
            n = missing(recs[stem][field].astype(np.uint64))
            r.check(n == 0, f"{stem}.{field}: {n} gids without a task record")


def dst_mask(t: np.ndarray) -> np.ndarray:
    """The tasks' `dst_mask` (1.9); all ones before."""
    if "dst_mask" in t.dtype.names:
        return t["dst_mask"]
    return np.full(len(t), NULL32, dtype=np.uint32)


def check_dst(r: Report, manifest: dict, recs: dict) -> None:
    t = recs["tasks"]
    if manifest["version"] < [1, 9]:
        print("skip  destination ranges in granules (before 1.9)")
        return
    g = int(manifest["run"]["granule"])
    ok = (t["dst_base"] % g == 0) & (t["dst_len"] % g == 0) & (t["dst_len"] >= g)
    r.check(bool(ok.all()),
            f"tasks: {int(np.count_nonzero(~ok))} of {len(t)} dst not whole granules of {g} B")
    masked = int(np.count_nonzero(dst_mask(t) != NULL32))
    print(f"info  tasks: {masked} masked destination ranges")


def check_untracked(r: Report, recs: dict) -> None:
    t = recs["tasks"]
    if "flags" not in t.dtype.names:
        print("skip  untracked tasks (no tasks.flags)")
        return
    u = t[(t["flags"] & UNTRACKED) != 0]
    r.check(bool((u["version"] == NULL32).all() and (u["queued"] == NULL64).all()
                 and (u["sent"] == NULL64).all()),
            f"untracked tasks: {len(u)}, version, queued and sent null")
    if "deps" in recs:
        named = np.isin(recs["deps"]["producer"], u["gid"])
        r.check(not named.any(),
                f"untracked tasks: {int(np.count_nonzero(named))} deps name one as producer")


USAGE_ONE = 65536


def v17(manifest: dict) -> bool:
    return manifest["version"] >= [1, 7]


def capacity(x: dict) -> float:
    """Slots per core cycle (1.7), or the lanes of an older dump."""
    if x.get("capacity") is not None:
        num, den = x["capacity"]
        return num / den
    return float(x.get("lanes") or 1)


def sub_of(c: np.ndarray) -> np.ndarray:
    """The `sub` field (1.7), `msg` before."""
    return c["sub"] if "sub" in c.dtype.names else c["msg"]


def is_message(manifest: dict, c: np.ndarray) -> np.ndarray:
    """Records of protocol messages: a `sub` on a resource without rate
    classes (on one with `rates`, `sub` is the class)."""
    res = manifest["resources"]
    classed = np.array([bool(x.get("rates")) for x in res] + [False])
    ids = np.minimum(c["resource"].astype(np.int64), len(res))
    return (sub_of(c) != 0xFF) & ~classed[ids]


def check_charges(r: Report, manifest: dict, recs: dict, stats: dict | None) -> None:
    c = recs.get("charges")
    if c is None:
        print("skip  charges (not recorded)")
        return
    res = manifest["resources"]
    rated = np.array([x["rate"] is not None for x in res])
    ids = c["resource"].astype(np.int64)
    r.check(bool((ids < len(res)).all()), "every charged resource id is in the manifest")
    ids = np.minimum(ids, len(res) - 1)
    on_rated = rated[ids]
    u = c[~on_rated]
    if v17(manifest):
        r.check(bool(((u["span"] >= 1) & (u["usage"] == USAGE_ONE)).all()),
                f"{len(u)} unlimited-link records have span >= 1 and usage = one slot")
    else:
        r.check(bool(((u["span"] >= 1) & (u["span"] == u["cycles"])).all()),
                f"{len(u)} unlimited-link records have span = cycles >= 1")
    # Capacity on rated resources: sweep the records of nonzero span in
    # time order per resource (an end before a start at the same cycle);
    # the slots held never exceed the resource's capacity.
    if v17(manifest):
        lanes = np.array([np.floor(capacity(x) * USAGE_ONE) / USAGE_ONE for x in res])
    else:
        lanes = np.array([capacity(x) for x in res], dtype=np.float64)
    k = c[on_rated & (c["span"] > 0)]
    if v17(manifest):
        held = k["usage"].astype(np.float64) / USAGE_ONE
    else:
        held = k["cycles"].astype(np.float64) / k["span"].astype(np.float64)
    rid = k["resource"].astype(np.int64)
    t = np.concatenate([k["start"], k["start"] + k["span"].astype(np.uint64)])
    dl = np.concatenate([held, -held])
    rr = np.concatenate([rid, rid])
    order = np.lexsort((dl, t, rr))
    t, dl, rr = t[order], dl[order], rr[order]
    level = np.cumsum(dl)
    # Restart the running sum at each resource.
    first = np.r_[True, rr[1:] != rr[:-1]]
    base = np.maximum.accumulate(np.where(first, np.arange(len(rr)), 0))
    offset = np.r_[0.0, level][base]
    level = level - offset
    over = level > lanes[rr] + 1e-6
    n = int(np.count_nonzero(over))
    r.check(n == 0, f"rated resources: {n} cycles over capacity")
    if n:
        i = int(np.flatnonzero(over)[0])
        print(f"      first: resource {rr[i]} at {t[i]}: {level[i]:.4f} of {lanes[rr[i]]:.4f} slots")
    if v17(manifest):
        classes = np.array([len(x.get("rates") or []) for x in res] + [0])
        sub = sub_of(c).astype(np.int64)
        on_classed = classes[np.minimum(c["resource"].astype(np.int64), len(res))] > 0
        bad = int(np.count_nonzero(on_classed & (sub != 0xFF)
                                   & (sub >= classes[np.minimum(c["resource"].astype(np.int64), len(res))])))
        r.check(bad == 0, f"rate classes: {int(on_classed.sum())} records, {bad} with no such class")
    if stats is not None and manifest["filters"]["partial"]:
        print("skip  per-resource cycles against busy_cycles (partial dump)")
    elif stats is not None and v17(manifest):
        busy = {x["name"]: x["busy_slot_cycles"] for x in stats["resources"]}
        k = c[on_rated]
        sc = np.bincount(k["resource"], weights=k["usage"].astype(np.float64) / USAGE_ONE
                         * k["span"].astype(np.float64), minlength=len(res))
        tol = np.bincount(k["resource"], weights=k["span"].astype(np.float64) / USAGE_ONE,
                          minlength=len(res))
        bad = [(x["name"], sc[x["id"]], busy[x["name"]]) for x in res
               if x["rate"] is not None and x["name"] in busy
               and abs(sc[x["id"]] - busy[x["name"]]) > tol[x["id"]] + 1e-6]
        r.check(not bad, f"per-resource slot-cycles match busy_slot_cycles ({len(bad)} differ)")
        for b in bad[:5]:
            print(f"      {b[0]}: charges {b[1]:.3f}, stats {b[2]:.3f}")
    elif stats is not None:
        busy = {x["name"]: x["busy_cycles"] for x in stats["resources"]}
        sums = np.bincount(c["resource"], weights=c["cycles"].astype(np.float64),
                           minlength=len(res))
        bad = [(x["name"], int(sums[x["id"]]), busy[x["name"]]) for x in res
               if x["rate"] is not None and x["name"] in busy
               and int(sums[x["id"]]) != busy[x["name"]]]
        r.check(not bad, f"per-resource cycles equal busy_cycles ({len(bad)} differ)")
        for b in bad[:5]:
            print(f"      {b[0]}: charges {b[1]}, stats {b[2]}")


def check_entries(r: Report, manifest: dict, recs: dict) -> None:
    e = recs.get("entries")
    if e is None:
        print("skip  entries (not recorded)")
        return
    closed = e["end"] != NULL64
    r.check(bool((e["end"][closed] >= e["start"][closed]).all()),
            f"{len(e)} entries: end >= start")
    # Occupancy per (Neutrino, table): an end before a start at one cycle.
    cap = np.array([[n["neutrino"][t] if n.get("neutrino") else 0 for t in TABLES]
                    for n in manifest["nodes"]], dtype=np.int64)
    key = e["nu"].astype(np.int64) * len(TABLES) + e["table"].astype(np.int64)
    t = np.concatenate([e["start"], e["end"][closed]])
    d = np.concatenate([np.ones(len(e), np.int64), -np.ones(int(closed.sum()), np.int64)])
    k = np.concatenate([key, key[closed]])
    order = np.lexsort((d, t, k))
    t, d, k = t[order], d[order], k[order]
    level = np.cumsum(d)
    first = np.r_[True, k[1:] != k[:-1]]
    base = np.maximum.accumulate(np.where(first, np.arange(len(k)), 0))
    level = level - np.r_[0, level][base]
    over = level > cap.reshape(-1)[k]
    for i, name in enumerate(TABLES):
        sel = k % len(TABLES) == i
        n = int(np.count_nonzero(over & sel))
        peak = int(level[sel].max()) if sel.any() else 0
        r.check(n == 0, f"entries {name}: {int(np.count_nonzero(e['table'] == i))} records, "
                f"peak {peak}, {n} cycles over capacity")
        if n:
            j = int(np.flatnonzero(over & sel)[0])
            print(f"      first: node {k[j] // len(TABLES)} at {t[j]}: {level[j]}")
    if manifest["filters"]["partial"]:
        print("skip  entries against tasks (partial dump)")
        return
    tasks = recs["tasks"]
    order = np.argsort(tasks["gid"])
    gids = tasks["gid"][order]
    named = e["gid"] != NULL32
    pos = np.minimum(np.searchsorted(gids, e["gid"]), len(gids) - 1)
    found = gids[pos] == e["gid"]
    r.check(bool(found[named].all()),
            f"entries.gid: {int(np.count_nonzero(named & ~found))} gids without a task record")
    task = tasks[order][pos]
    p = (e["table"] == 0) & closed
    r.check(bool(named[p].all()), f"pending: {int(np.count_nonzero(p & ~named))} "
            "entries that left without an adopting task")
    vr = (e["flags"] & VRESET) != 0
    i = (e["table"] == 3) & ~vr & closed & named & found
    r.check(bool((e["end"][i] == task["invoke"][i]).all()),
            f"inbound_invokes: {int(np.count_nonzero(i))} end at their task's invoke")
    v = ((e["table"] == 3) | (e["table"] == 5)) & vr
    r.check(bool((~named[v]).all()),
            f"VersionReset entries: {int(np.count_nonzero(v & (e['table'] == 3)))} inbound, "
            f"{int(np.count_nonzero(v & (e['table'] == 5)))} outbound, "
            f"{int(np.count_nonzero(v & named))} with a gid")
    o = (e["table"] == 5) & (e["flags"] & VRESET == 0) & named & found
    r.check(bool((e["start"][o] == task["queued"][o]).all()),
            f"outbound_invokes: {int(np.count_nonzero(o))} start at their task's queued")


def check_hashes(r: Report, manifest: dict) -> None:
    hashes = [l for l in manifest.get("links", []) if l.get("hash")]
    wports = [m for m in manifest["memories"] if m.get("wport") is not None]
    if not hashes and not wports:
        print("skip  hash links and write ports (none)")
        return
    bad = []
    for l in hashes:
        h = l["hash"]
        (lo, hi), (tlo, thi) = [[int(x, 16) for x in p] for p in (h["range"], h["to_range"])]
        masks = [int(x, 16) for x in h["index"]]
        lowest = min((m & -m) for m in masks)
        k = len(masks)
        if hi - lo != thi - tlo:
            bad.append(f"{l['path']}: range and to_range sizes differ")
        if h["granule"] != lowest:
            bad.append(f"{l['path']}: granule {h['granule']}, lowest mask bit {lowest}")
        if h["slice_bytes"] * 2**k != hi - lo:
            bad.append(f"{l['path']}: slice_bytes {h['slice_bytes']} x 2^{k} != {hi - lo:#x}")
        if len(h["drop"]) != k:
            bad.append(f"{l['path']}: {len(h['drop'])} drop bits for {k} masks")
    roles = {x["id"]: x["role"] for x in manifest["resources"]}
    for m in wports:
        role = roles.get(m["wport"], {})
        if not (role.get("type") == "mem_port" and role.get("memory") == m["id"] and role.get("write")):
            bad.append(f"{m['path']}: wport {m['wport']} has role {role}")
    r.check(not bad, f"{len(hashes)} hash links, {len(wports)} write ports consistent")
    for b in bad[:5]:
        print(f"      {b}")


def check_channels(r: Report, manifest: dict, recs: dict) -> None:
    res = manifest["resources"]
    pairs = []  # (read id, write id)
    for l in manifest.get("links", []):
        for d in ("ab", "ba"):
            if l.get(f"{d}_wr") is not None:
                pairs.append((l[d], l[f"{d}_wr"]))
    for a in manifest["attachments"]:
        for d in ("in", "out"):
            if a.get(f"{d}_wr") is not None:
                pairs.append((a[d], a[f"{d}_wr"]))
    bus_wr = {x["role"]["bus"]: x["id"] for x in res
              if x["role"]["type"] == "bus" and x["role"].get("write")}
    for x in res:
        if x["role"]["type"] == "bus" and not x["role"].get("write") and x["role"]["bus"] in bus_wr:
            pairs.append((x["id"], bus_wr[x["role"]["bus"]]))
    if not pairs:
        print("skip  write channels (none)")
        return
    bad = []
    for rd, wr in pairs:
        w, rr = res[wr], res[rd]
        if not w["role"].get("write") or w["role"]["type"] != rr["role"]["type"]:
            bad.append(f"{w['name']}: role {w['role']}")
        num, den = w["rate"]
        if v17(manifest):
            # `rate` is per core cycle: capacity = rate / beat.
            beat = w["beat_bytes"] or 1
            ok = abs(capacity(w) - num / (den * beat)) < 1e-9
            what = f"capacity {w.get('capacity')}"
        else:
            want = -(-num // (den * w["beat_bytes"])) if w["beat_bytes"] else 1
            ok = w["lanes"] == max(want, 1)
            what = f"lanes {w['lanes']}"
        if not ok or w["beat_bytes"] != rr["beat_bytes"]:
            bad.append(f"{w['name']}: {what} beat {w['beat_bytes']}")
    r.check(not bad, f"{len(pairs)} write channels: roles and capacity")
    for b in bad[:5]:
        print(f"      {b}")
    c = recs.get("charges")
    if c is None:
        return
    reads = np.array(sorted(p[0] for p in pairs), dtype=np.int64)
    writes = np.array(sorted(p[1] for p in pairs), dtype=np.int64)
    on_wr = np.isin(c["resource"], writes)
    on_rd = np.isin(c["resource"], reads)
    n1 = int(np.count_nonzero(on_wr & (c["dir"] != 1)))
    n2 = int(np.count_nonzero(on_rd & (c["dir"] == 1)))
    r.check(n1 == 0 and n2 == 0,
            f"{int(on_wr.sum())} charges on write channels: {n1} not writes, "
            f"{n2} writes on a split element's read channel")


PROGRAM = 0x80000000
ROOT_PROGRAM = 0xFFFFFFFE


def check_owners(r: Report, manifest: dict, recs: dict) -> None:
    c = recs.get("charges")
    if c is None or manifest["version"] < [1, 6]:
        print("skip  charge owners (not recorded, or before version 1.6)")
        return
    gid = c["gid"].astype(np.int64)
    null = gid == NULL32
    prog = ~null & (gid >= PROGRAM)
    spawns = set()
    if "programs" in recs:
        spawns = {int(x) for x in recs["programs"]["spawn"] if x != NULL32}
    p_ok = (gid == ROOT_PROGRAM) | np.isin(gid & 0x7FFFFFFF, list(spawns))
    bad_prog = int(np.count_nonzero(prog & ~p_ok))
    r.check(bad_prog == 0, f"program owners: {int(prog.sum())} records, {bad_prog} "
            "not the root program or a program of programs.bin")
    task = ~null & ~prog
    if not manifest["filters"]["partial"]:
        known = np.sort(recs["tasks"]["gid"].astype(np.int64))
        g = gid[task]
        pos = np.minimum(np.searchsorted(known, g), len(known) - 1)
        miss = int(np.count_nonzero(known[pos] != g)) if len(known) else len(g)
        r.check(miss == 0, f"task owners: {int(task.sum())} records, {miss} without a task record")
    kinds = manifest["message_kinds"]
    msg = sub_of(c).astype(np.int64)
    is_msg = is_message(manifest, c)
    # Allowed owners per kind: t task, p program, s spawn task.
    allowed = {"Invoke": "t", "InvokeAck": "tp", "RetryAck": "tp", "Reject": "tp",
               "Await": "tp", "AwaitAck": "tp", "Resolve": "tp", "Start": "s", "Return": "s"}
    bad = []
    for k, name in enumerate(kinds):
        sel = is_msg & (msg == k) & ~null
        if not sel.any():
            continue
        a = allowed.get(name, "p")
        ok = np.zeros(int(sel.sum()), bool)
        if "t" in a or "s" in a:
            t = task[sel]
            if "s" in a:
                t &= np.isin(gid[sel], list(spawns))
            ok |= t
        if "p" in a:
            ok |= prog[sel]
        n = int(np.count_nonzero(~ok))
        if n:
            bad.append(f"{name}: {n} of {int(sel.sum())}")
    r.check(not bad, "message owners fit their kind")
    for b in bad[:5]:
        print(f"      {b}")
    roles = [x["role"]["type"] for x in manifest["resources"]]
    counts = {}
    for i in np.flatnonzero(null):
        key = kinds[msg[i]] if is_msg[i] else roles[c["resource"][i]] if c["resource"][i] < len(roles) else "?"
        counts[key] = counts.get(key, 0) + 1
    print(f"info  null owners: {int(null.sum())} of {len(c)} records"
          + (": " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) if counts else ""))


def check_links(r: Report, manifest: dict) -> None:
    n = len(manifest["resources"])
    buses = manifest["buses"]
    ok = all(x["id"] == i for i, x in enumerate(manifest["resources"]))
    r.check(ok, "resource ids equal their index")
    miss = []
    for a in manifest["attachments"]:
        if buses[a["bus"]]["topology"] == "xbar":
            for d in ("in", "out"):
                if a[d] is None or not 0 <= a[d] < n:
                    miss.append(f"attachment {a['id']} {d}")
    # Version 1.2 has `links`; 1.0 and 1.1 have `bridges`.
    for b in manifest.get("links", manifest.get("bridges", [])):
        for d in ("ab", "ba"):
            if b[d] is None or not 0 <= b[d] < n:
                miss.append(f"link {b['id']} {d}")
    roles = {(x["role"]["type"], x["role"].get("bus")) for x in manifest["resources"]}
    for b in buses:
        if b["topology"] == "bus" and ("bus", b["id"]) not in roles:
            miss.append(f"bus {b['id']}")
    r.check(not miss, f"every topology link has a resource ({len(miss)} missing)")
    for m in miss[:5]:
        print(f"      {m}")


def overlaps(start: np.ndarray, end: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """`[start, end)` overlaps `[lo, hi)`; an empty interval when its cycle is in it."""
    return (start < hi) & ((end > lo) | ((end == start) & (start >= lo)))


def kept(manifest: dict, full: dict) -> dict[str, np.ndarray]:
    """The records of the complete dump `full` that the filters of `manifest` keep."""
    f = manifest["filters"]
    paths = [n["path"] for n in manifest["nodes"]]
    if f["nodes"] is None:
        keep_nu = np.ones(len(paths), bool)
    else:
        keep_nu = np.array([any(p == n or p.startswith(n + ".") for n in f["nodes"])
                            for p in paths])
    owners = [x["owner"] for x in manifest["resources"]]
    keep_res = np.array([(keep_nu[o] if o is not None else f["nodes"] is None) for o in owners])
    lo, hi = f["window"] if f["window"] is not None else (0, 2**64 - 1)
    lo, hi = np.uint64(lo), np.uint64(hi)
    files = f["records"]
    out = {}
    t = full["tasks"]
    task_keep = keep_nu[t["nu"]] & overlaps(t["invoke"], t["retire"], lo, hi)
    kept_gids = np.sort(t["gid"][task_keep])
    gid_nu = dict(zip(t["gid"].tolist(), t["nu"].tolist()))

    def member(g: np.ndarray, sorted_set: np.ndarray) -> np.ndarray:
        pos = np.minimum(np.searchsorted(sorted_set, g), max(len(sorted_set) - 1, 0))
        return (sorted_set[pos] == g) if len(sorted_set) else np.zeros(len(g), bool)

    def want(stem: str) -> bool:
        return files is None or stem in files

    if want("tasks"):
        out["tasks"] = t[task_keep]
    if want("chunks"):
        c = full["chunks"]
        nus = np.array([gid_nu[g] for g in c["gid"].tolist()], dtype=np.int64)
        out["chunks"] = c[keep_nu[nus] & overlaps(c["start"], c["end"], lo, hi)]
    if want("deps"):
        d = full["deps"]
        out["deps"] = d[member(d["consumer"], kept_gids)]
    if want("charges"):
        c = full["charges"]
        end = c["start"] + c["span"].astype(np.uint64)
        types = f.get("types")
        is_msg = is_message(manifest, c)
        kind = np.where(is_msg, types is None or "messages" in types,
                        types is None or "charges" in types)
        out["charges"] = c[keep_res[c["resource"]] & overlaps(c["start"], end, lo, hi) & kind]
    if want("programs"):
        p = full["programs"]
        out["programs"] = p[keep_nu[p["nu"]] & overlaps(p["start"], p["end"], lo, hi)]
    if want("marks"):
        m = full["marks"]
        nu_ok = np.where(m["nu"] == 0xFFFF, True, keep_nu[np.minimum(m["nu"], len(paths) - 1)])
        out["marks"] = m[nu_ok & overlaps(m["time"], m["time"], lo, hi)]
    if want("entries") and "entries" in full:
        e = full["entries"]
        out["entries"] = e[keep_nu[e["nu"]] & overlaps(e["start"], e["end"], lo, hi)]
    return out


def check_reference(r: Report, dump: Path, manifest: dict, recs: dict, ref: Path) -> None:
    full_manifest = json.loads((ref / "manifest.json").read_text())
    r.check(not full_manifest["filters"]["partial"], "the reference is a complete dump")
    want = kept(manifest, load(ref, full_manifest))
    names, ref_names = strings(dump), strings(ref)
    for stem in SIZES:
        if stem not in want:
            r.check(stem not in recs, f"{stem}: not recorded, no file")
            continue
        got = comparable(stem, recs[stem], names) if stem in recs else []
        exp = comparable(stem, want[stem], ref_names)
        r.check(got == exp, f"{stem}: {len(got)} records, the filters keep {len(exp)} of the reference")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dump", type=Path)
    ap.add_argument("--stats", type=Path, help="the --stats JSON of the same run")
    ap.add_argument("--reference", type=Path, help="a complete dump of the same run")
    a = ap.parse_args(argv)
    manifest = json.loads((a.dump / "manifest.json").read_text())
    summary_path = a.dump / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    stats = json.loads(a.stats.read_text()) if a.stats else None
    recs = load(a.dump, manifest)
    r = Report()
    check_layouts(r, manifest, recs, summary)
    check_gids(r, manifest, recs)
    check_untracked(r, recs)
    check_dst(r, manifest, recs)
    check_charges(r, manifest, recs, stats)
    check_entries(r, manifest, recs)
    check_hashes(r, manifest)
    check_channels(r, manifest, recs)
    check_owners(r, manifest, recs)
    check_links(r, manifest)
    if a.reference:
        check_reference(r, a.dump, manifest, recs, a.reference)
    print(f"{r.failed} failed")
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
