"""Check a raw trace dump (`r2sim run ... --trace-raw DIR`).

The format is r2-vis's `trace-raw.md` (version 1.0). The checks are the
ones of its Section 8, plus the file layout:

* the manifest's record layouts have the record sizes of the format, every
  binary file holds whole records, and the counts match `summary.json`;
* in a complete dump (`filters.partial` false), every gid in chunks,
  deps, charges and programs.spawn has a task record;
* on a resource with a rate, two charges overlap only when they are the
  direction records of one charge (same start and span); per resource the
  cycles sum to `busy_cycles` of the `--stats` JSON (with `--stats`);
* every resource id in the records is in the manifest, every link of the
  topology tables has a resource, and records on unlimited links have
  span = cycles >= 1;
* with `--reference FULL` (a complete dump of the same run): the dump
  holds exactly the records of FULL that the filters of its manifest keep
  (architecture Section 7.3, trace-raw.md Section 1.1).

Usage, from r2-sim/ in the workspace environment:

    PYTHONPATH=python python -m r2sim.traceraw DIR [--stats STATS.json] [--reference FULL]

Exit status 1 when a check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SIZES = {"tasks": 104, "chunks": 32, "deps": 24, "charges": 32, "programs": 32, "marks": 16}
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
        got = sum(np.dtype(TYPES[t]).itemsize for _, t in layout) if layout else None
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
        gids = gids[gids != NULL32]
        pos = np.searchsorted(known, gids)
        pos = np.minimum(pos, len(known) - 1)
        return int(np.count_nonzero(known[pos] != gids)) if len(known) else len(gids)

    for stem, field in [("chunks", "gid"), ("deps", "consumer"), ("deps", "producer"),
                        ("charges", "gid"), ("programs", "spawn")]:
        if stem in recs:
            n = missing(recs[stem][field].astype(np.uint64))
            r.check(n == 0, f"{stem}.{field}: {n} gids without a task record")


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
    r.check(bool(((u["span"] >= 1) & (u["span"] == u["cycles"])).all()),
            f"{len(u)} unlimited-link records have span = cycles >= 1")
    # Overlap on rated resources: unique (resource, start, span) intervals
    # of nonzero span, sorted, must not overlap.
    k = c[on_rated & (c["span"] > 0)]
    key = np.unique(np.stack([k["resource"].astype(np.uint64), k["start"],
                              k["span"].astype(np.uint64)], axis=1), axis=0)
    same = key[1:, 0] == key[:-1, 0]
    overlap = same & (key[1:, 1] < key[:-1, 1] + key[:-1, 2])
    n = int(np.count_nonzero(overlap))
    r.check(n == 0, f"rated resources: {n} overlapping charges")
    if n:
        i = int(np.flatnonzero(overlap)[0])
        print(f"      first: resource {key[i, 0]} {key[i, 1:]} and {key[i + 1, 1:]}")
    if stats is not None:
        busy = {x["name"]: x["busy_cycles"] for x in stats["resources"]}
        sums = np.bincount(c["resource"], weights=c["cycles"].astype(np.float64),
                           minlength=len(res))
        bad = [(x["name"], int(sums[x["id"]]), busy[x["name"]]) for x in res
               if x["rate"] is not None and x["name"] in busy
               and int(sums[x["id"]]) != busy[x["name"]]]
        r.check(not bad, f"per-resource cycles equal busy_cycles ({len(bad)} differ)")
        for b in bad[:5]:
            print(f"      {b[0]}: charges {b[1]}, stats {b[2]}")


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
    for b in manifest["bridges"]:
        for d in ("ab", "ba"):
            if b[d] is None or not 0 <= b[d] < n:
                miss.append(f"bridge {b['id']} {d}")
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
        is_msg = c["msg"] != 0xFF
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
    check_charges(r, manifest, recs, stats)
    check_links(r, manifest)
    if a.reference:
        check_reference(r, a.dump, manifest, recs, a.reference)
    print(f"{r.failed} failed")
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
