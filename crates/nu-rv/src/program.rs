//! The interpreter: an RV program on the scalar core of a Neutrino.
//!
//! At its start the program writes the start mailbox `{entry, args,
//! done}` to offset 0 of its data scratch and starts at IMEM offset 4,
//! with CSR 0xFC0 the node's prefix and CSR 0xFC1 its epoch (that of the
//! spawn that started it, until the program writes it). It then runs
//! instructions in batches: a batch ends at a `nu.*` instruction, at a
//! write of CSR 0xFC1, or at a load or store outside the IMEM and the own
//! data scratch. The batch's instruction
//! cycles are charged on the scalar resource (the `scalar` costs of the
//! Neutrino), then the request is made:
//!
//! * `nu.invoke rd, rs1`: the descriptor at `rs1`, decoded; `rd` gets the
//!   version of a local versioned invoke, else 0;
//! * `nu.wait rs1`: the key, 0 for the fence;
//! * `nu.version`, `nu.vreset` (granule addresses), `nu.vreset_all`;
//! * a CSR instruction that writes CSR 0xFC1: the instruction `epoch` with
//!   the value written;
//! * `nu.exit`: the program returns;
//! * a SPAWN descriptor: the RV program at `entry` on the direct child of
//!   that index, with the arguments at `args`; the tracer names it by the
//!   record in the child's IMEM when it starts;
//! * an IMEM_COPY descriptor: a task on the instruction DMA of the node;
//! * a load or store elsewhere: `NuCtx::load` or `NuCtx::store` (the
//!   memory path); the instruction then runs again with the value at
//!   hand. A faulting step changes nothing, so the repeat is exact.
//!
//! An illegal instruction, an unknown CSR, `ecall`, `ebreak`, a fetch
//! outside the IMEM, a store into the IMEM, a bad descriptor and the
//! instruction limit stop the run with an error that names the node and
//! the PC.

use std::rc::Rc;

use nu_iss::{decode as decode_insn, Bus, Event, Fault, Hart, Inst};
use nu_model::{Addr, NeutrinoId, Range};
use sim_core::{abi, Args, Built, CoreMem, Descriptor, NuCtx, OpcodeId, ProgramName, ScalarCosts};

use crate::decode::{decode, Decoded};
use crate::Abi;

/// Instructions after which a program stops the run.
pub const DEFAULT_LIMIT: u64 = 1_000_000_000;

/// A program record in an IMEM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    /// Its IMEM offset.
    pub at: u64,
    /// The IMEM offset of the entry function.
    pub entry: u64,
    pub args_bytes: u32,
    pub stack_bytes: u32,
    pub name: String,
    /// The node type it runs on.
    pub node: String,
}

/// A NUL-padded text field, if it is one: printable ASCII, then NULs.
fn padded(b: &[u8]) -> Option<String> {
    let n = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    let ok = n > 0 && b[..n].iter().all(|c| c.is_ascii_graphic()) && b[n..].iter().all(|&c| c == 0);
    ok.then(|| String::from_utf8_lossy(&b[..n]).into_owned())
}

/// The program records of `imem` for node type `node`.
pub fn find_record(imem: &[u8], node: &str) -> Vec<Record> {
    records(imem)
        .into_iter()
        .filter(|r| r.node == node)
        .collect()
}

/// The program records of `imem`, for any node type: 72 bytes at an
/// 8-byte aligned offset, ABI version 1, reserved 0, an entry inside the
/// image, and NUL-padded names.
pub fn records(imem: &[u8]) -> Vec<Record> {
    let n = abi::RECORD_BYTES as usize;
    let mut out = Vec::new();
    let mut at = 0;
    while at + n <= imem.len() {
        let r = &imem[at..at + n];
        let word = |o: usize| u32::from_le_bytes(r[o..o + 4].try_into().expect("4 bytes"));
        let entry = u64::from_le_bytes(r[..8].try_into().expect("8 bytes"));
        if word(8) == abi::VERSION && word(20) == 0 && entry < imem.len() as u64 {
            if let (Some(name), Some(t)) = (padded(&r[24..56]), padded(&r[56..72])) {
                out.push(Record {
                    at: at as u64,
                    entry,
                    args_bytes: word(12),
                    stack_bytes: word(16),
                    name,
                    node: t,
                });
            }
        }
        at += 8;
    }
    out
}

/// An RV program on one Neutrino.
pub struct RvProgram {
    abi: Rc<Abi>,
    /// The IMEM offset of the entry; `None`: the one record of the node's
    /// type in its IMEM.
    entry: Option<u64>,
    /// The global address of the arguments.
    args: u64,
    /// The base granule of the done range of the spawn (0 for the root).
    done: u64,
    /// Instructions after which the run stops.
    limit: u64,
}

/// The value at hand for the repeat of a load or store that left the
/// core.
#[derive(Debug, Clone, Copy)]
enum Resume {
    Load { addr: u64, value: u64 },
    Store { addr: u64 },
}

/// The bus of the core: its IMEM at `[0, imem_bytes)` and its data
/// scratch; everything else faults.
struct CoreBus<'a> {
    mem: &'a mut CoreMem,
    scratch: Range,
    imem_bytes: u64,
    resume: Option<Resume>,
    /// The last access was the repeat of one that left the core.
    resumed: bool,
}

impl CoreBus<'_> {
    fn local(&mut self, addr: u64, n: u64) -> Option<&mut [u8]> {
        let end = addr.checked_add(n)?;
        if end <= self.imem_bytes {
            return Some(&mut self.mem.imem()[addr as usize..end as usize]);
        }
        if self.scratch.base <= addr && end <= self.scratch.bound {
            let o = (addr - self.scratch.base) as usize;
            return Some(&mut self.mem.scratch()[o..o + n as usize]);
        }
        None
    }
}

impl Bus for CoreBus<'_> {
    fn fetch(&mut self, pc: u64) -> Result<u32, Fault> {
        if pc.checked_add(2).is_none_or(|e| e > self.imem_bytes) {
            return Err(Fault::Access {
                addr: pc,
                write: false,
            });
        }
        let imem = self.mem.imem();
        let half = |i: usize| {
            imem.get(i..i + 2)
                .map_or(0, |b| u16::from_le_bytes([b[0], b[1]]) as u32)
        };
        Ok(half(pc as usize) | half(pc as usize + 2) << 16)
    }

    fn load(&mut self, addr: u64, bytes: u8) -> Result<(u64, u64), Fault> {
        if let Some(Resume::Load { addr: a, value }) = self.resume {
            if a == addr {
                self.resume = None;
                self.resumed = true;
                return Ok((value, 0));
            }
        }
        match self.local(addr, bytes as u64) {
            Some(b) => {
                let mut v = [0u8; 8];
                v[..b.len()].copy_from_slice(b);
                Ok((u64::from_le_bytes(v), 0))
            }
            None => Err(Fault::Access { addr, write: false }),
        }
    }

    fn store(&mut self, addr: u64, bytes: u8, value: u64, _taint: u64) -> Result<(), Fault> {
        if let Some(Resume::Store { addr: a }) = self.resume {
            if a == addr {
                self.resume = None;
                self.resumed = true;
                return Ok(());
            }
        }
        // The IMEM window is not writable by the core.
        if addr < self.imem_bytes {
            return Err(Fault::Access { addr, write: true });
        }
        match self.local(addr, bytes as u64) {
            Some(b) => {
                b.copy_from_slice(&value.to_le_bytes()[..bytes as usize]);
                Ok(())
            }
            None => Err(Fault::Access { addr, write: true }),
        }
    }
}

/// Why a batch ended.
enum Stop {
    Event(Event),
    /// A load or store outside the core: the instruction at `pc` repeats
    /// once it is done.
    Remote {
        write: bool,
        addr: u64,
        bytes: u8,
        value: u64,
    },
    /// An error at `pc`.
    Fault {
        pc: u64,
        msg: String,
    },
}

/// The cycles of instruction `inst` (length `len`) at `pc` that went on to
/// `next`; `resumed`: a load or store that left the core (its request
/// pays for it).
fn cost(c: &ScalarCosts, inst: &Inst, pc: u64, len: u64, next: u64, resumed: bool) -> u64 {
    match inst {
        Inst::Jal { .. } | Inst::Jalr { .. } => c.branch_taken,
        Inst::Branch { .. } if next != pc.wrapping_add(len) => c.branch_taken,
        Inst::Branch { .. } => c.branch,
        Inst::Load { .. } | Inst::Store { .. } if resumed => 0,
        Inst::Load { .. } => c.load,
        Inst::Store { .. } => c.store,
        Inst::Op { m: true, f, .. } if f & 7 < 4 => c.mul,
        Inst::Op { m: true, .. } => c.div,
        Inst::Custom { .. } | Inst::Ecall | Inst::Ebreak => 0,
        _ => c.alu,
    }
}

/// Runs instructions until an event, a remote access, a fault or the
/// limit; adds their cycles to `cycles`.
fn batch(
    hart: &mut Hart,
    bus: &mut CoreBus,
    c: &ScalarCosts,
    cycles: &mut u64,
    limit: u64,
) -> Stop {
    loop {
        let pc = hart.pc;
        if hart.retired >= limit {
            return Stop::Fault {
                pc,
                msg: format!("the instruction limit of {limit} is reached"),
            };
        }
        let word = match bus.fetch(pc) {
            Ok(w) => w,
            Err(_) => {
                return Stop::Fault {
                    pc,
                    msg: "fetch outside the IMEM".into(),
                }
            }
        };
        let Some((inst, len)) = decode_insn(word) else {
            return Stop::Fault {
                pc,
                msg: format!("illegal instruction {word:#010x}"),
            };
        };
        bus.resumed = false;
        match hart.step(bus) {
            Ok(ev) => {
                *cycles += cost(c, &inst, pc, len, hart.pc, bus.resumed);
                if let Some(ev) = ev {
                    return Stop::Event(ev);
                }
            }
            Err(Fault::Access { addr, write }) => {
                if write && addr < bus.imem_bytes {
                    return Stop::Fault {
                        pc,
                        msg: format!("store into the IMEM at {addr:#x}"),
                    };
                }
                let (bytes, value) = match inst {
                    Inst::Load { bytes, .. } => (bytes, 0),
                    Inst::Store { bytes, rs2, .. } => (bytes, hart.get(rs2).0),
                    _ => unreachable!("only loads and stores access memory"),
                };
                return Stop::Remote {
                    write,
                    addr,
                    bytes,
                    value,
                };
            }
            Err(Fault::Illegal { word, .. }) => {
                return Stop::Fault {
                    pc,
                    msg: format!("illegal instruction {word:#010x}"),
                }
            }
            Err(Fault::Csr { csr, .. }) => {
                return Stop::Fault {
                    pc,
                    msg: format!("unknown CSR {csr:#x}"),
                }
            }
        }
    }
}

impl RvProgram {
    /// The program of the record of the node's type in its IMEM, with the
    /// arguments at `args`.
    pub fn root(abi: Rc<Abi>, args: u64, limit: u64) -> Self {
        Self {
            abi,
            entry: None,
            args,
            done: 0,
            limit,
        }
    }

    /// Runs the program on `ctx`'s Neutrino.
    pub async fn run(self, ctx: NuCtx) {
        let me = ctx.id();
        let node = self.abi.node(me).clone();
        let fail = |msg: String| ctx.fail(format!("{}: {msg}", node.path));
        let core = ctx.core().clone();
        let scratch = match &node.scratch {
            Ok(r) => *r,
            Err(e) => return fail(e.clone()).await,
        };
        let entry = match self.entry {
            Some(e) => e,
            None => {
                let t = abi::node_type(&node.path);
                let found = loaded_records(&mut core.mem.borrow_mut(), t);
                match found.as_slice() {
                    [r] => r.entry,
                    rs => {
                        return fail(format!(
                            "{} program records of node type {t} in the loaded image, not one",
                            rs.len()
                        ))
                        .await
                    }
                }
            }
        };
        {
            let mut mem = core.mem.borrow_mut();
            let s = mem.scratch();
            for (i, v) in [entry, self.args, self.done].into_iter().enumerate() {
                s[8 * i..8 * i + 8].copy_from_slice(&v.to_le_bytes());
            }
        }
        let costs = node.scalar;
        let mut hart = Hart::new(abi::START_VECTOR, core.prefix);
        hart.epoch = ctx.epoch();
        let mut resume = None;
        loop {
            let mut cycles = 0;
            let stop = {
                let mut mem = core.mem.borrow_mut();
                let mut bus = CoreBus {
                    mem: &mut mem,
                    scratch,
                    imem_bytes: node.imem_bytes,
                    resume: resume.take(),
                    resumed: false,
                };
                batch(&mut hart, &mut bus, &costs, &mut cycles, self.limit)
            };
            if cycles > 0 {
                ctx.cycles(cycles).await;
            }
            let at = |pc: u64, msg: String| format!("pc {pc:#x}: {msg}");
            match stop {
                Stop::Fault { pc, msg } => return fail(at(pc, msg)).await,
                Stop::Remote {
                    write: false,
                    addr,
                    bytes,
                    ..
                } => {
                    let b = ctx.load(Addr(addr), bytes as u64).await;
                    let mut v = [0u8; 8];
                    v[..b.len()].copy_from_slice(&b);
                    resume = Some(Resume::Load {
                        addr,
                        value: u64::from_le_bytes(v),
                    });
                }
                Stop::Remote {
                    write: true,
                    addr,
                    bytes,
                    value,
                } => {
                    let b = value.to_le_bytes()[..bytes as usize].to_vec();
                    ctx.store(Addr(addr), b).await;
                    resume = Some(Resume::Store { addr });
                }
                Stop::Event(Event::Exit { .. }) => return,
                Stop::Event(Event::Break { pc }) => {
                    return fail(at(pc, "ecall or ebreak".into())).await
                }
                Stop::Event(Event::Invoke { pc, rd, desc, .. }) => {
                    let d = {
                        let mut mem = core.mem.borrow_mut();
                        decode(mem.scratch(), scratch, desc, me, pc, &self.abi)
                    };
                    match d {
                        Err(e) => return fail(at(pc, e)).await,
                        Ok(Decoded::Task(d)) if d.untracked => {
                            ctx.invoke_untracked(d).await;
                            hart.set(rd, 0, 0);
                        }
                        Ok(Decoded::Task(d)) => {
                            let h = ctx.invoke(d).await;
                            // A forwarded invoke returns 0: the owner
                            // assigns the version.
                            let v = if h.task.is_some() { h.version } else { 0 };
                            hart.set(rd, v, 0);
                        }
                        Ok(Decoded::Spawn {
                            done,
                            deps,
                            child,
                            epoch,
                            entry,
                            args,
                        }) => {
                            let Some(&nu) = ctx.children().get(child as usize) else {
                                return fail(at(pc, self.past_children(&ctx, child))).await;
                            };
                            let program = RvProgram {
                                abi: self.abi.clone(),
                                entry: Some(entry as u64),
                                args,
                                done: done.base >> self.abi.granule_shift,
                                limit: self.limit,
                            };
                            let name = ProgramName::AtStart(Box::new(record_name(
                                &self.abi,
                                nu,
                                Some(entry as u64),
                            )));
                            ctx.spawn_in_epoch(nu, name, done, deps, Some(pc), epoch, move |c| {
                                program.run(c)
                            })
                            .await;
                            hart.set(rd, 0, 0);
                        }
                        Ok(Decoded::ImemCopy {
                            record,
                            deps,
                            child,
                            src,
                            offset,
                            bytes,
                        }) => {
                            let Some(&nu) = ctx.children().get(child as usize) else {
                                return fail(at(pc, self.past_children(&ctx, child))).await;
                            };
                            let d = Descriptor {
                                opcode: OpcodeId::IMEM_COPY,
                                pe: None,
                                dst: record,
                                versioned: false,
                                untracked: false,
                                dst_mask: nu_model::FULL_MASK,
                                inferred_version: None,
                                deps,
                                label: None,
                                args: Args::ImemCopy {
                                    child: nu,
                                    src,
                                    offset: offset as u64,
                                    bytes,
                                },
                                pc: Some(pc),
                            };
                            ctx.invoke(d).await;
                            hart.set(rd, 0, 0);
                        }
                    }
                }
                Stop::Event(Event::Wait { key: 0, .. }) => ctx.fence().await,
                Stop::Event(Event::Wait { key, .. }) => ctx.wait_key(self.abi.dep(key)).await,
                Stop::Event(Event::Version { base, counter, .. }) => {
                    ctx.version(base << self.abi.granule_shift, counter).await
                }
                Stop::Event(Event::Vreset { base, .. }) => {
                    ctx.vreset(base << self.abi.granule_shift).await
                }
                Stop::Event(Event::VresetAll { .. }) => ctx.vreset_all().await,
                Stop::Event(Event::Epoch { epoch, .. }) => ctx.set_epoch(epoch).await,
            }
        }
    }
}

impl RvProgram {
    fn past_children(&self, ctx: &NuCtx, child: u16) -> String {
        format!(
            "child index {child} past the {} direct children",
            ctx.children().len()
        )
    }
}

/// The records of node type `node` in the image loaded into `mem`: the
/// bytes of the last image load or IMEM copy at offset 0
/// (`CoreMem::image`), so that records left over from a larger image
/// loaded before do not count. Without a recorded load (an IMEM written
/// directly), the whole IMEM.
fn loaded_records(mem: &mut CoreMem, node: &str) -> Vec<Record> {
    let n = mem.image.as_ref().map_or(usize::MAX, |i| i.bytes as usize);
    let imem = mem.imem();
    find_record(&imem[..n.min(imem.len())], node)
}

/// The name of the program that starts on `nu`: the record of the node's
/// type in its loaded image, and with `entry` (a spawn's) the record with
/// that entry; "rv" when there is not exactly one. Read when the program
/// starts: the IMEM holds the image only after its copy.
fn record_name(abi: &Abi, nu: NeutrinoId, entry: Option<u64>) -> impl FnOnce() -> &'static str {
    let (core, path) = (abi.node(nu).core.clone(), abi.node(nu).path.clone());
    move || {
        let found: Vec<Record> = loaded_records(&mut core.mem.borrow_mut(), abi::node_type(&path))
            .into_iter()
            .filter(|r| entry.is_none_or(|e| r.entry == e))
            .collect();
        match found.as_slice() {
            [r] => intern(&r.name),
            _ => "rv",
        }
    }
}

/// A program name for the tracer, which keeps `&'static str` names: one
/// copy per distinct name.
fn intern(name: &str) -> &'static str {
    use std::cell::RefCell;
    use std::collections::HashMap;
    thread_local! {
        static NAMES: RefCell<HashMap<String, &'static str>> = RefCell::new(HashMap::new());
    }
    NAMES.with(|m| {
        *m.borrow_mut()
            .entry(name.to_string())
            .or_insert_with(|| Box::leak(name.to_string().into_boxed_str()))
    })
}

/// `--program rv`: the root program, the one record of the root's node
/// type in the root's IMEM, with the arguments at `args`; the run stops
/// after `limit` instructions of the program. The tracer names it by that
/// record when it starts (`Launch::start_name`).
pub fn launch(b: &Built, args: u64, limit: u64) -> Result<nu_prog::Launch, String> {
    let abi = Rc::new(Abi::new(b));
    let root = NeutrinoId(0);
    abi.node(root).scratch.as_ref().map_err(Clone::clone)?;
    let name = ProgramName::AtStart(Box::new(record_name(&abi, root, None)));
    let p = RvProgram::root(abi, args, limit);
    let mut l = nu_prog::Launch::new(root, "rv", move |ctx| p.run(ctx));
    l.start_name = Some(name);
    Ok(l)
}
