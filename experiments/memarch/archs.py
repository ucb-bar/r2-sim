"""Design points of the three SM memory architectures: capacity, macros,
area, analytic bandwidth bound, and the simulator rates.

Options:
  A  TMEM-fed Atlas: operands and accumulators in TMEM (wide macros, fixed
     lane wiring); a small smem with subbanks and a crossbar serves Muon.
  B  Unified smem: one R x S smem feeds Atlas (row port of W_a B/cycle) and
     Muon (crossbar); Atlas keeps 2 x 32x32 accumulators (1 in the array, 1
     drain buffer) and spills partials / drains C to smem.
  C  Blackwell-like: A/B in an R x S smem (row port W_a); accumulators in a
     TMEM of N_acc 32x32 fp32 tiles; Muon reaches TMEM by lane partition.

Assumptions (see ASSUMPTIONS for the list printed with results) are module
constants so they are easy to swap.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import area as A
import muonxbar
import simconfig

# ---------------------------------------------------------------- assumptions
LANES = 16                # lanes per Muon core
MUON_CORES = 2
MUON_OPS = 128            # lane-ops per cycle per core (256 per SM, as configs/sm.yaml)
WORD = 4                  # bytes per lane access
ACC_ARRAY = 8192          # bytes of accumulator in/at the array (2 x 32x32 fp32)
MUON_SCRATCH = 32 << 10   # Muon private data in smem
TMEM_W = 256              # TMEM macro width (bits): "large, wide macros"
OPBUF_W = 256             # Atlas-local operand buffer macro width
D = 128                   # attention head dim
KV = 128                  # attention key block
SM_OPS = 8                # softmax lane-ops per element of S
MAX_MACROS = 128          # hard limit: total macros < 128
ASSUMPTIONS = [
    f"Muon: {MUON_CORES} cores x {LANES} lanes, {WORD} B per lane access, {MUON_OPS} lane-ops/cycle per core",
    f"Atlas in-array accumulator {ACC_ARRAY} B (2 x 32x32 fp32) in every option, not counted in area",
    f"fp8 runs at the same MAC rate as bf16 (bytes halve)",
    f"TMEM macros {TMEM_W} b wide; Atlas-local operand buffers {OPBUF_W} b wide",
    f"Muon scratch {MUON_SCRATCH // 1024} KiB in smem; attention head dim {D}, key block {KV}, softmax {SM_OPS} lane-ops/element",
    "smem bandwidth = R x S macros x width/8 per cycle, read and write separate (1R1W)",
    "Atlas row port reads one bank row: W_a <= S x width/8",
    "Muon TMEM access: one partition per core per cycle: min(lanes x 4, TMEM/partitions)",
    "Atlas operand reuse over a block > 32x32 needs an Atlas-local double-buffered operand buffer 2 (bm+bn) kc b",
    "HBM ideal (no latency, unlimited); DMA writes still charge the on-SM write ports",
    "Arbitration: first-come per task chunk (r2-sim flow model); no Atlas priority / round robin",
]


@dataclass(frozen=True)
class Design:
    option: str
    macs: int = 1024          # P
    dtype: str = "bf16"
    banks: int = 2            # R
    subbanks: int = 16        # S
    width: int = 32           # smem macro width (bits)
    radix: int | None = None  # Muon crossbar radix; None = full
    kc: int = 64
    block: int = 32           # Atlas operand-reuse block bm = bn
    atlas_port: int = 128     # W_a (B, C)
    n_acc: int = 0            # TMEM accumulator tiles (A, C)
    partitions: int = 2       # TMEM lane partitions (A, C)
    tmem_cols: int = 0        # TMEM macros side by side (A, C)
    qr: int = 64              # attention query rows per block

    @property
    def b(self) -> int:
        return 1 if self.dtype == "fp8" else 2

    def label(self) -> str:
        r = "full" if self.radix is None else f"r{self.radix}"
        s = f"{self.option} P{self.macs} {self.dtype} {self.banks}x{self.subbanks}x{self.width}b {r} kc{self.kc} blk{self.block} q{self.qr}"
        if self.option in "BC":
            s += f" Wa{self.atlas_port}"
        if self.option in "AC":
            s += f" acc{self.n_acc} p{self.partitions} tc{self.tmem_cols}"
        return s


# ------------------------------------------------------------ workload shapes
def attn_rows(d: Design) -> int | None:
    """Query rows per block (the knob qr): S (x2) and O must fit the
    accumulator memory (TMEM for A and C; B's smem is sized for it)."""
    if d.qr < d.block:
        return None
    if d.option in "AC" and d.qr * (2 * KV + D) * 4 > d.n_acc * 4096:
        return None
    return d.qr


def gemm_tile(d: Design) -> int | None:
    """Square-ish output tile (rows = cols) of the GEMM: A and C hold two
    tiles' partials in TMEM; B: 32 (block-major, no spills) or 128."""
    if d.option == "B":
        return None  # chosen per workload (gemm_tiles_b)
    for t in (128, 64, 32):
        if 2 * t * t * 4 <= d.n_acc * 4096 and t >= d.block:
            return t
    return None


# ----------------------------------------------------------------- capacities
def capacities(d: Design) -> dict:
    """Bytes each memory must hold (max over the attention and GEMM working sets)."""
    b, kc = d.b, d.kc
    qr = d.qr
    t = (gemm_tile(d) or 128) if d.option in "AC" else gemm_plan_b(d, "square")
    att_ops = qr * D * b + 4 * KV * D * b + 2 * qr * KV * b       # Q, K^T x2, V x2, P x2
    att_acc = (2 * qr * KV + qr * D) * 4                         # S x2, O (fp32)
    gemm_ops = 2 * (t + t) * kc * b                              # two stages of A and B chunks
    gemm_acc = 2 * t * t * 4
    gemm_out = 2 * t * t * 2
    opbuf = 2 * (2 * d.block) * kc * b if d.block > 32 else 0
    if d.option == "A":
        return {"tmem": max(att_ops, gemm_ops) + d.n_acc * 4096, "smem": MUON_SCRATCH + gemm_out, "opbuf": opbuf}
    if d.option == "C":
        return {"smem": max(att_ops, gemm_ops) + gemm_out + MUON_SCRATCH, "tmem": d.n_acc * 4096, "opbuf": opbuf}
    # B: everything in smem; the GEMM tile of its plan.
    return {"smem": max(att_ops + att_acc, gemm_ops + gemm_acc + gemm_out) + MUON_SCRATCH,
            "drain": 4096, "opbuf": opbuf}


def columns(bytes_: int, cols: int, width: int) -> tuple[int, int, int] | None:
    """Macros for `bytes_` over `cols` macro columns of `width` bits:
    (count, depth, width) or None if illegal."""
    if not A.MIN_W <= width <= A.MAX_W:
        return None
    bits_col = math.ceil(bytes_ * 8 / cols)
    depth = 1 << max(0, math.ceil(math.log2(max(1, math.ceil(bits_col / width)))))
    depth = max(depth, A.MIN_BITS // width)
    max_depth = A.MAX_BITS // width
    per_col = math.ceil(depth / max_depth)
    return cols * per_col, min(depth, max_depth), width


def macros(d: Design) -> dict | None:
    """Macro inventory: name -> (count, depth, width)."""
    cap = capacities(d)
    out = {}
    out["smem"] = columns(cap["smem"], d.banks * d.subbanks, d.width)
    if "tmem" in cap:
        out["tmem"] = columns(cap["tmem"], d.tmem_cols, TMEM_W)
    if cap.get("opbuf"):
        cols = max(1, math.ceil(atlas_operand_rate(d, d.block) / (OPBUF_W // 8)))
        out["opbuf"] = columns(cap["opbuf"], cols, OPBUF_W)
    if "drain" in cap:
        out["drain"] = columns(cap["drain"], max(1, math.ceil(4096 * 8 / 2048 / OPBUF_W)), OPBUF_W)
    if any(v is None for v in out.values()):
        return None
    return out


def atlas_operand_rate(d: Design, block: int) -> float:
    """B/cycle of A and B the array reads at full rate with a block x block reuse block."""
    return 2 * block * d.b * d.macs / (block * block)


# ------------------------------------------------------------------- area
_FIT = None
_LOGIC = None


def fit():
    global _FIT, _LOGIC
    if _FIT is None:
        _FIT, _ = A.fit_sram()
        _LOGIC = A.load_logic()
    return _FIT, _LOGIC


def area(d: Design) -> dict | None:
    f, lg = fit()
    inv = macros(d)
    if inv is None:
        return None
    mac = {k: n * f.area(depth * w, w) for k, (n, depth, w) in inv.items()}
    count = sum(n for n, _, _ in inv.values())
    ncols = d.banks * d.subbanks
    lanes = LANES * MUON_CORES
    xbar = A.xbar_area(lanes, ncols, 32, d.radix, lg)
    other = {"arbiters": A.arbiter_area(3, ncols, lg)}
    if d.option in "BC":
        bits = d.atlas_port * 8
        other["atlas_port_mux"] = lg["k_xp"] * d.banks * bits
        other["swizzle"] = A.rotator_area(d.atlas_port // 4, 32, lg)
        other["dma_port_mux"] = lg["k_xp"] * d.banks * bits
    if d.option in "AC":
        other["muon_tmem_wiring"] = lg["k_xp"] * lanes * 32 * d.partitions
        other["tmem_arbiters"] = A.arbiter_area(3, d.tmem_cols, lg)
    total = sum(mac.values()) + xbar + sum(other.values())
    return {"total": total, "macros": mac, "macro_count": count, "xbar": xbar, "other": other,
            "inventory": {k: list(v) for k, v in inv.items()}}


# ------------------------------------------------------------ bandwidth caps
@lru_cache(maxsize=None)
def muon_eff(pattern: str, banks: int, subbanks: int, width: int, radix: int | None) -> float:
    return muonxbar.efficiency(pattern, banks, subbanks, width, radix, lanes=LANES, cores=MUON_CORES,
                               accesses=128)


def caps(d: Design, pattern: str = "row") -> dict:
    smem = d.banks * d.subbanks * d.width / 8
    tmem = d.tmem_cols * TMEM_W / 8 if d.option in "AC" else 0
    c = {"smem_rd": smem, "smem_wr": smem, "tmem_rd": tmem, "tmem_wr": tmem,
         "muon_smem": LANES * WORD * muon_eff(pattern, d.banks, d.subbanks, d.width, d.radix),
         "muon_tmem": min(LANES * WORD, tmem / d.partitions) if tmem else 0}
    if d.option in "BC":
        c["atlas_port"] = d.atlas_port
    return c


def valid(d: Design) -> str | None:
    """Why the design cannot be built (None: it can)."""
    if d.option in "BC" and d.atlas_port > d.subbanks * d.width / 8:
        return "atlas port wider than a bank row"
    if d.option in "AC" and d.tmem_cols % d.partitions:
        return "tmem columns not a multiple of partitions"
    if d.option == "B" and d.block != 32:
        return "B keeps 2 tiles: block 32"
    if attn_rows(d) is None:
        return "attention block does not fit (TMEM) or is below the reuse block"
    if d.option in "AC" and gemm_tile(d) is None:
        return "TMEM too small for the GEMM working set"
    if D % d.kc or KV % d.kc:
        return "kc must divide 128 for attention"
    return None


# ------------------------------------------------------------ analytic bound
def demands_attn(d: Design) -> tuple[dict, float, float]:
    """Bytes per Atlas-cycle on each resource for one attention step at full
    Atlas rate; (demand per cycle, Atlas cycles per step, Muon cycles per step)."""
    qr, b, bl = attn_rows(d), d.b, d.block
    macs = 2 * qr * KV * D
    t = macs / d.macs
    opr = macs * b * (2 / bl)                      # operand reads
    # Partials: S over d in kc chunks (kept in the array when <= 2 blocks fit);
    # O over every chunk of every step.
    s_blocks = (qr // bl) * (KV // bl)
    s_part = 0 if (D // d.kc == 1 or (s_blocks <= 2 and bl * bl * 4 * s_blocks <= ACC_ARRAY)) else (D // d.kc - 1) * qr * KV * 4
    o_part = (KV // d.kc) * qr * D * 4
    s_out, s_in_muon = qr * KV * 4, qr * KV * 4
    p_out, dma = qr * KV * b, 2 * KV * D * b
    opm = "tmem" if d.option == "A" else "smem"
    acm = "smem" if d.option == "B" else "tmem"
    dem = {f"{opm}_rd": opr, f"{opm}_wr": p_out + dma}
    dem[f"{acm}_rd"] = dem.get(f"{acm}_rd", 0) + s_part + o_part + s_in_muon
    dem[f"{acm}_wr"] = dem.get(f"{acm}_wr", 0) + s_part + o_part + s_out
    if d.option in "BC":
        dem["atlas_port"] = opr
    # Muon per core: half of S read, half of P written.
    link_r = caps(d)["muon_tmem" if acm == "tmem" else "muon_smem"]
    link_w = caps(d)["muon_tmem" if opm == "tmem" else "muon_smem"]
    t_muon = 50 + max(SM_OPS * qr * KV / 2 / MUON_OPS, s_in_muon / 2 / max(link_r, 1e-9),
                      p_out / 2 / max(link_w, 1e-9))
    return {k: v / t for k, v in dem.items()}, t, t_muon


def gemm_plan_b(d: Design, kind: str) -> int:
    """B's GEMM tile: the one with the lower peak demand (32: block-major,
    DMA traffic = Atlas traffic; 128: chunk-major, partial spills)."""
    best = None
    for t in (32, 64, 128):
        dem = demands_gemm(d, kind, tile=t)
        peak = max(dem[k] / caps(d)[k] for k in dem if caps(d).get(k))
        if peak <= 1 / 0.95:
            return t  # the smallest tile that keeps Atlas fed: least capacity
        if best is None or peak < best[0]:
            best = (peak, t)
    return best[1]


GEMMS = {"square": (1024, 1024, 1024), "skinny": (64, 4096, 4096), "kheavy": (256, 256, 8192)}


def gemm_shape(d: Design, kind: str, tile: int | None = None) -> tuple[int, int, int]:
    """(tm, tn, bm) of a GEMM workload."""
    m, n, k = GEMMS[kind]
    t = tile or (gemm_tile(d) if d.option in "AC" else gemm_plan_b(d, kind))
    tm = min(t, m)
    tn = min(t * t // tm, n, 128)
    bl = min(d.block, tm, tn)
    return tm, tn, bl


def demands_gemm(d: Design, kind: str, tile: int | None = None) -> dict:
    m, n, k = GEMMS[kind]
    tm, tn, bl = gemm_shape(d, kind, tile)
    b = d.b
    per_mac = 1 / d.macs                            # cycles per MAC
    opr = b * (2 / bl) / per_mac                     # bytes per cycle
    blocks = (tm // bl) * (tn // bl)
    spill = 0 if (k // d.kc == 1 or (blocks <= 2 and blocks * bl * bl * 4 <= ACC_ARRAY)) else 4 / d.kc / per_mac
    dma = b * (tm + tn) / (tm * tn) / per_mac
    out = 2 / k / per_mac
    opm = "tmem" if d.option == "A" else "smem"
    acm = "smem" if d.option == "B" else "tmem"
    dem = {f"{opm}_rd": opr, f"{opm}_wr": dma}
    dem[f"{acm}_rd"] = dem.get(f"{acm}_rd", 0) + spill
    dem[f"{acm}_wr"] = dem.get(f"{acm}_wr", 0) + spill + out
    if d.option in "BC":
        dem["atlas_port"] = opr
    return dem


def bound(d: Design) -> dict:
    """Analytic utilization bound: min over resources of cap / demand, and
    Atlas / Muon overlap, for attention and the square GEMM."""
    c = caps(d)
    dem, t, tm = demands_attn(d)
    att = min([1.0] + [c[k] / v for k, v in dem.items() if v > 0] + [t / tm])
    sq = demands_gemm(d, "square")
    gem = min([1.0] + [c[k] / v for k, v in sq.items() if v > 0])
    return {"attn": att, "gemm": gem, "muon_cycles": tm, "atlas_cycles": t}


# ------------------------------------------------------------- enumeration
KNOBS = {
    "A": dict(banks=[1, 2, 4], subbanks=[8, 16, 32], n_acc=[4, 8, 16, 32], partitions=[2, 4],
              tmem_mult=[1, 2, 4, 8]),
    "B": dict(banks=[2, 4, 8], subbanks=[8, 16, 32], atlas_port=[64, 128, 256]),
    "C": dict(banks=[2, 4, 8], subbanks=[8, 16, 32], atlas_port=[64, 128, 256], n_acc=[2, 4, 8, 16],
              partitions=[2, 4], tmem_mult=[1, 2, 4, 8]),
}
WIDTHS = [32, 64, 128, 256]
RADIX = [2, 4, 8, None]
KCS = [64, 128]           # kc must divide 128 (attention)
BLOCKS = [32, 64, 128]
QRS = [32, 64, 128]       # attention query rows per block


def enumerate_designs(option: str, macs: int, dtype: str):
    k = KNOBS[option]
    for banks, sub, w, radix, kc, blk, qr in itertools.product(k["banks"], k["subbanks"], WIDTHS, RADIX, KCS,
                                                               BLOCKS, QRS):
        if option == "B":
            for wa in k["atlas_port"]:
                yield Design("B", macs, dtype, banks, sub, w, radix, kc, blk, atlas_port=wa, qr=qr)
        else:
            for nacc, part, mult in itertools.product(k["n_acc"], k["partitions"], k["tmem_mult"]):
                ports = k.get("atlas_port", [0])
                for wa in ports:
                    yield Design(option, macs, dtype, banks, sub, w, radix, kc, blk, atlas_port=wa or 0,
                                 n_acc=nacc, partitions=part, tmem_cols=part * mult, qr=qr)


def evaluate(d: Design) -> dict:
    """Validity, area and the analytic bound of one design."""
    why = valid(d)
    if why:
        return {"design": d, "pruned": why}
    a = area(d)
    if a is None:
        return {"design": d, "pruned": "illegal macro shape"}
    if a["macro_count"] >= MAX_MACROS:
        return {"design": d, "pruned": f"{a['macro_count']} macros", "area": a}
    bd = bound(d)
    out = {"design": d, "area": a, "bound": bd}
    if min(bd["attn"], bd["gemm"]) < 0.95:
        out["pruned"] = f"bandwidth bound attn {bd['attn']:.2f} gemm {bd['gemm']:.2f}"
    return out


# ------------------------------------------------------------- sim mapping
def sim_point(d: Design, pattern: str = "row", ideal: bool = False) -> simconfig.SimPoint:
    c = caps(d, pattern)
    if ideal:
        big = 1 << 20
        return simconfig.SimPoint(macs=d.macs, atlas_port=None, smem_rd=big, smem_wr=big, tmem_rd=big,
                                  tmem_wr=big, muons=MUON_CORES, muon_ops=MUON_OPS,
                                  muon_smem={w: big for w in range(MUON_CORES)},
                                  muon_tmem={w: big for w in range(MUON_CORES)})
    return simconfig.SimPoint(
        macs=d.macs, atlas_port=c.get("atlas_port"), acc_buffer=ACC_ARRAY,
        smem_rd=c["smem_rd"], smem_wr=c["smem_wr"], tmem_rd=c["tmem_rd"], tmem_wr=c["tmem_wr"],
        muons=MUON_CORES, muon_ops=MUON_OPS,
        muon_smem={w: c["muon_smem"] for w in range(MUON_CORES)},
        muon_tmem={w: max(c["muon_tmem"], 1) for w in range(MUON_CORES)})


def kernel_point(d: Design, workload: str, scale: float = 1.0) -> dict:
    """The kernel fields of a workload on a design."""
    opm = "tmem" if d.option == "A" else "smem"
    acm = "smem" if d.option == "B" else "tmem"
    if workload == "attn":
        qr = attn_rows(d)
        return dict(mode="attn", dtype=d.dtype, tm=qr, bm=min(d.block, qr), bn=d.block, kc=d.kc,
                    opmem=opm, accmem=acm, qblocks=2, kvblocks=max(2, int(16 * scale)), d=D, sm_ops=SM_OPS)
    if workload == "muon_alone":
        qr = attn_rows(d)
        src = acm
        return dict(mode="muon", tasks=max(2, int(16 * scale)) * 2, ops=SM_OPS * qr * KV // 2,
                    bytes_in=qr * KV * 4 // 2, bytes_out=qr * KV * d.b // 2, mmem=src, cores=2)
    m, n, k = GEMMS[workload]
    tm, tn, bl = gemm_shape(d, workload)
    return dict(mode="gemm", dtype=d.dtype, m=m, n=n, k=k, tm=tm, tn=tn, bm=bl, bn=bl,
                kc=min(d.kc, k), opmem=opm, accmem=acm)


def macs_of(kp: dict) -> int:
    if kp["mode"] == "gemm":
        return kp["m"] * kp["n"] * kp["k"]
    if kp["mode"] == "attn":
        return kp["qblocks"] * kp["kvblocks"] * 2 * kp["tm"] * KV * kp["d"]
    return 0


def to_json(d: Design) -> dict:
    return asdict(d)
