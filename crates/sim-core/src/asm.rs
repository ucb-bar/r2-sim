//! An assembly-style listing of what programs say to their Neutrinos: one
//! instruction per request (`NuCtx` call), in program order, with each
//! task's descriptor as operands. The input is the request stream, so a
//! listing exists for any program; [`crate::Sim::set_asm`] records it
//! (off by default) and [`Stats::asm`](crate::Stats) holds one
//! [`Listing`] per program instance.
//!
//! Syntax (one instruction per line; `;` starts a comment):
//!
//! ```text
//! .program <neutrino> <name>          ; parent <neutrino>, started at cycle <t>
//! <mnemonic> <operand> ...            ; t=<cycle the program made the request>
//! ```
//!
//! An operand is `key=value` or, for the one-address instructions (`wait`,
//! `version`, `vreset`, `load`, `store`), a leading address. Values:
//! decimal numbers; `0x` hex offsets and lengths; names (dotted paths, no
//! spaces); an address `<symbol>+0x<offset>` (or a bare `0x` address
//! outside every symbol); a range `<address>:0x<len>`; a versioned key
//! `<address>@v<n>` (`@v?` when the program did not infer the version);
//! lists `[a,b]`; quoted strings for mark names.
//!
//! Instructions:
//!
//! * a task, by its opcode's mnemonic (`dma`, `mm`, `vec`, `join`, or the
//!   registered name): `dst=<range>[@v<n>] [pe=<path>] deps=[...]
//!   [label=<n>]` and the opcode's arguments:
//!   - `dma from=<address> to=<address> bytes=<n> tile=<n>`;
//!   - `mm a=<address> b=<address> [a2=..] [b2=..] lda= ldb= ldc= m= n=
//!     k= types=<a>,<b>,<c> acc=0|1 keep=0|1 tile=<rows>x<cols>`;
//!   - `vec ops=<lane-ops> in=<bytes> out=<bytes> [src=<address>]`;
//! * `spawn child=<neutrino> done=<range> deps=[...]`;
//! * `wait <address>[@v<n>]`, `version <address> v=<n>`, `vreset_all`,
//!   `vreset <address>`, `cycles <n>`, `load <address> len=<n>`,
//!   `store <address> len=<n>`, `mark "<name>"`, `cache_flush`.
//!
//! Symbols ([`Symbols`]): an address prints relative to the narrowest
//! named range that holds it: a memory by its name (a hashed memory at the
//! linear range programs address), a bank outside its memory's range by
//! the bank's name, a hash link's linear range not covered by a memory by
//! the link's name, a sync-only range by its name.

use std::collections::HashMap;

use nu_model::{Addr, Dep, NeutrinoId, Range};

use crate::config::Built;
use crate::profile::{Args, Descriptor, Dtype, OpcodeId};
use crate::program::Request;
use crate::time::Cycle;

/// One request of a program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    Invoke(Descriptor),
    Wait(Dep),
    Version(u64, u64),
    VersionReset,
    VReset(u64),
    Cycles(u64),
    Load {
        addr: u64,
        len: u64,
    },
    Store {
        addr: u64,
        len: u64,
    },
    Mark(String),
    CacheFlush,
    Spawn {
        child: NeutrinoId,
        done: Range,
        deps: Vec<Dep>,
    },
}

impl Op {
    pub(crate) fn from_request(req: &Request) -> Self {
        match req {
            Request::Invoke(d) => Op::Invoke(d.clone()),
            Request::Wait(d) => Op::Wait(*d),
            Request::Version(a, v) => Op::Version(*a, *v),
            Request::VersionReset => Op::VersionReset,
            Request::VReset(a) => Op::VReset(*a),
            Request::Cycles(n) => Op::Cycles(*n),
            Request::Load { addr, len } => Op::Load {
                addr: addr.0,
                len: *len,
            },
            Request::Store { addr, bytes } => Op::Store {
                addr: addr.0,
                len: bytes.len() as u64,
            },
            Request::Mark(n) => Op::Mark(n.clone()),
            Request::CacheFlush => Op::CacheFlush,
            Request::Spawn { child, done, deps } => Op::Spawn {
                child: *child,
                done: *done,
                deps: deps.clone(),
            },
        }
    }
}

/// One program instance's requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Listing {
    pub nu: NeutrinoId,
    pub name: &'static str,
    /// The Neutrino whose spawn started it (`None`: started by the host).
    pub parent: Option<NeutrinoId>,
    pub start: Cycle,
    /// (the cycle of the request, the request).
    pub ops: Vec<(Cycle, Op)>,
}

/// Records listings while on (held by the simulation).
#[derive(Debug, Default)]
pub(crate) struct Recorder {
    pub listings: Vec<Listing>,
    /// The current program instance of each Neutrino.
    current: HashMap<NeutrinoId, usize>,
}

impl Recorder {
    pub fn start(
        &mut self,
        nu: NeutrinoId,
        name: &'static str,
        parent: Option<NeutrinoId>,
        now: Cycle,
    ) {
        self.current.insert(nu, self.listings.len());
        self.listings.push(Listing {
            nu,
            name,
            parent,
            start: now,
            ops: Vec::new(),
        });
    }

    pub fn op(&mut self, nu: NeutrinoId, now: Cycle, req: &Request) {
        if let Some(&i) = self.current.get(&nu) {
            self.listings[i].ops.push((now, Op::from_request(req)));
        }
    }
}

/// The names a listing prints: address ranges, PEs, Neutrinos and opcode
/// mnemonics.
#[derive(Debug, Clone, Default)]
pub struct Symbols {
    /// Named ranges; an address takes the narrowest that holds it.
    ranges: Vec<(Range, String)>,
    pes: Vec<String>,
    nus: Vec<String>,
    opcodes: Vec<String>,
}

/// The mnemonic of a registered opcode name.
fn mnemonic(name: &str) -> String {
    match name {
        "atlas.matmul" => "mm".into(),
        "muon.vector" => "vec".into(),
        n => n.replace('.', "_"),
    }
}

/// The name of a memory block: its memory's for a bank (`<memory>.b<i>`).
fn memory_of(path: &str) -> &str {
    match path.rsplit_once(".b") {
        Some((m, i)) if !i.is_empty() && i.bytes().all(|c| c.is_ascii_digit()) => m,
        _ => path,
    }
}

impl Symbols {
    /// The symbols of a built configuration.
    pub fn from_built(b: &Built) -> Self {
        let mut ranges: Vec<(Range, String)> = Vec::new();
        let covered =
            |ranges: &[(Range, String)], r: &Range| ranges.iter().any(|x| x.0.overlaps(r));
        // Memories, at the range programs address (a hashed memory's linear
        // range), then banks outside them, then sync ranges.
        let mut seen: Vec<&str> = Vec::new();
        for m in &b.layout.memories {
            if m.kind == "sync" {
                continue;
            }
            let name = memory_of(&m.path);
            if seen.contains(&name) {
                continue;
            }
            seen.push(name);
            if let Some(r) = b.try_memory(name) {
                ranges.push((r, name.to_string()));
            }
        }
        for m in &b.layout.memories {
            if m.kind == "sync" {
                continue;
            }
            for r in &m.ranges {
                if !covered(&ranges, r) {
                    ranges.push((*r, m.path.clone()));
                }
            }
        }
        for l in &b.layout.links {
            if let Some(h) = &l.hash {
                if !covered(&ranges, &h.range) {
                    ranges.push((h.range, l.path.clone()));
                }
            }
        }
        for m in b.layout.memories.iter().filter(|m| m.kind == "sync") {
            for r in &m.ranges {
                ranges.push((*r, m.path.clone()));
            }
        }
        let mut pes: Vec<(u32, String)> =
            b.pe_paths().into_iter().map(|(p, id)| (id.0, p)).collect();
        pes.sort();
        let topo = b.sim.topology();
        Self {
            ranges,
            pes: pes.into_iter().map(|x| x.1).collect(),
            nus: (0..topo.len())
                .map(|i| topo.path_name(NeutrinoId(i as u32)))
                .collect(),
            opcodes: b.sim.profiles().names().map(mnemonic).collect(),
        }
    }

    /// `<symbol>+0x<offset>`, or the bare address in no symbol.
    pub fn addr(&self, a: u64) -> String {
        match self
            .ranges
            .iter()
            .filter(|(r, _)| r.contains(Addr(a)))
            .min_by_key(|(r, _)| r.len())
        {
            Some((r, n)) => format!("{n}+{:#x}", a - r.base),
            None => format!("{a:#x}"),
        }
    }

    pub fn range(&self, r: Range) -> String {
        format!("{}:{:#x}", self.addr(r.base), r.len())
    }

    pub fn dep(&self, d: &Dep) -> String {
        match d {
            Dep::Versioned { addr, version } => format!("{}@v{version}", self.addr(*addr)),
            Dep::Unversioned(a) => self.addr(*a),
        }
    }

    fn deps(&self, deps: &[Dep]) -> String {
        let v: Vec<String> = deps.iter().map(|d| self.dep(d)).collect();
        format!("[{}]", v.join(","))
    }

    pub fn nu(&self, nu: NeutrinoId) -> String {
        self.nus
            .get(nu.0 as usize)
            .cloned()
            .unwrap_or_else(|| format!("nu{}", nu.0))
    }

    fn opcode(&self, op: OpcodeId) -> String {
        self.opcodes
            .get(op.0 as usize)
            .cloned()
            .unwrap_or_else(|| format!("op{}", op.0))
    }

    /// One task's instruction.
    pub fn task(&self, d: &Descriptor) -> String {
        let mut s = format!("{} dst={}", self.opcode(d.opcode), self.range(d.dst));
        if d.versioned {
            match d.inferred_version {
                Some(v) => s += &format!("@v{v}"),
                None => s += "@v?",
            }
        }
        if let Some(pe) = d.pe {
            let p = self
                .pes
                .get(pe.0 as usize)
                .cloned()
                .unwrap_or_else(|| format!("pe{}", pe.0));
            s += &format!(" pe={p}");
        }
        s += &format!(" deps={}", self.deps(&d.deps));
        match &d.args {
            Args::Dma {
                src,
                dst,
                bytes,
                tile_bytes,
            } => {
                s += &format!(
                    " from={} to={} bytes={bytes} tile={tile_bytes}",
                    self.addr(src.0),
                    self.addr(dst.0)
                )
            }
            Args::Matmul {
                a,
                b,
                a2,
                b2,
                lda,
                ldb,
                ldc,
                m,
                n,
                k,
                in_dtype,
                b_dtype,
                out_dtype,
                accumulate,
                keep,
                tile,
            } => {
                s += &format!(" a={} b={}", self.addr(a.0), self.addr(b.0));
                if let Some(x) = a2 {
                    s += &format!(" a2={}", self.addr(x.0));
                }
                if let Some(x) = b2 {
                    s += &format!(" b2={}", self.addr(x.0));
                }
                s += &format!(
                    " lda={lda} ldb={ldb} ldc={ldc} m={m} n={n} k={k} types={},{},{} acc={} keep={} tile={}x{}",
                    dtype(*in_dtype),
                    dtype(*b_dtype),
                    dtype(*out_dtype),
                    *accumulate as u8,
                    *keep as u8,
                    tile[0],
                    tile[1]
                );
            }
            Args::Vector {
                ops,
                bytes_in,
                bytes_out,
                src,
            } => {
                s += &format!(" ops={ops} in={bytes_in} out={bytes_out}");
                if let Some(a) = src {
                    s += &format!(" src={}", self.addr(a.0));
                }
            }
            Args::Join => {}
            Args::Spawn { child } => s += &format!(" child={}", self.nu(*child)),
        }
        if let Some(l) = d.label {
            s += &format!(" label={l}");
        }
        s
    }

    /// One request's instruction.
    pub fn op(&self, op: &Op) -> String {
        match op {
            Op::Invoke(d) => self.task(d),
            Op::Wait(d) => format!("wait {}", self.dep(d)),
            Op::Version(a, v) => format!("version {} v={v}", self.addr(*a)),
            Op::VersionReset => "vreset_all".into(),
            Op::VReset(a) => format!("vreset {}", self.addr(*a)),
            Op::Cycles(n) => format!("cycles {n}"),
            Op::Load { addr, len } => format!("load {} len={len}", self.addr(*addr)),
            Op::Store { addr, len } => format!("store {} len={len}", self.addr(*addr)),
            Op::Mark(n) => format!("mark {n:?}"),
            Op::CacheFlush => "cache_flush".into(),
            Op::Spawn { child, done, deps } => format!(
                "spawn child={} done={} deps={}",
                self.nu(*child),
                self.range(*done),
                self.deps(deps)
            ),
        }
    }

    /// A program instance's listing: its header, then one instruction per
    /// request with its cycle in a trailing comment.
    pub fn listing(&self, l: &Listing) -> String {
        let mut out = format!(".program {} {}", self.nu(l.nu), l.name);
        out += &match l.parent {
            Some(p) => format!(
                "    ; parent {}, started at cycle {}\n",
                self.nu(p),
                l.start
            ),
            None => format!("    ; started by the host at cycle {}\n", l.start),
        };
        for (t, op) in &l.ops {
            out += &format!("    {}    ; t={t}\n", self.op(op));
        }
        out
    }
}

fn dtype(d: Dtype) -> &'static str {
    match d {
        Dtype::Bf16 => "bf16",
        Dtype::F32 => "f32",
        Dtype::Fp8 => "fp8",
        Dtype::Mxfp4 => "mxfp4",
    }
}
