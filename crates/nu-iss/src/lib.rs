//! An instruction-set simulator of the Neutrino scalar core: RV64I, M, C
//! and Zicsr, plus the `nu.*` instructions (custom-0 major opcode).
//!
//! [`Hart::step`] runs one instruction against a [`Bus`]. A `nu.*`
//! instruction does not change the hart: `step` returns it as an
//! [`Event`] for the caller, the Neutrino model, which then writes the
//! result register with [`Hart::set`].
//!
//! Every bit of a register carries a taint bit, and a [`Bus`] keeps taint
//! for the bytes of its memory. A load from a tainted byte taints the bits
//! it loads, and each instruction taints the bits of its result that depend
//! on tainted bits of its operands (an add from the lowest tainted bit up,
//! an and only where no operand bit is a known zero, a shift by a known
//! amount shifts the taint). A caller uses taint to mark values that depend
//! on unknown input (for example the arguments of a program in a dry run).
//! A branch or jump on a tainted value sets [`Hart::uncertain`].

mod decode;

pub use decode::{decode, Inst};

/// A memory access the bus refuses, or an instruction the hart cannot run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fault {
    /// No memory at this address, or a misaligned access.
    Access { addr: u64, write: bool },
    /// An instruction this simulator does not implement (for example an
    /// atomic or a floating-point instruction).
    Illegal { pc: u64, word: u32 },
    /// A CSR that does not exist.
    Csr { pc: u64, csr: u16 },
}

/// The memory of a hart: instruction fetch, data loads and stores. A value
/// comes with its taint: a mask of its tainted bits.
pub trait Bus {
    /// The 32 bits at `pc` (an instruction may be 16 bits long; the upper
    /// half is then the next parcel or zero).
    fn fetch(&mut self, pc: u64) -> Result<u32, Fault>;
    /// `bytes` (1, 2, 4 or 8) little-endian bytes at `addr`, zero-extended,
    /// and their taint.
    fn load(&mut self, addr: u64, bytes: u8) -> Result<(u64, u64), Fault>;
    fn store(&mut self, addr: u64, bytes: u8, value: u64, taint: u64) -> Result<(), Fault>;
}

/// A `nu.*` instruction, for the Neutrino model. `taint` is the tainted
/// bits of the operand(s).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    /// `nu.invoke`: the descriptor is at `desc`; the version goes to `rd`.
    Invoke {
        pc: u64,
        rd: u8,
        desc: u64,
        taint: u64,
    },
    /// `nu.wait`: `key`, 0 for the fence.
    Wait { pc: u64, key: u64, taint: u64 },
    /// `nu.version`: sets the counter of base granule `base`.
    Version {
        pc: u64,
        base: u64,
        counter: u64,
        taint: u64,
    },
    /// `nu.vreset`.
    Vreset { pc: u64, base: u64, taint: u64 },
    /// `nu.vreset_all`.
    VresetAll { pc: u64 },
    /// `nu.exit`: the program ends.
    Exit { pc: u64 },
    /// `ebreak` or `ecall`.
    Break { pc: u64 },
}

/// One hart: program counter, integer registers with their taint, the node
/// prefix that CSR 0xFC0 returns and the epoch that CSR 0xFC1 returns.
#[derive(Debug, Clone)]
pub struct Hart {
    pub pc: u64,
    pub x: [u64; 32],
    /// The tainted bits of each register.
    pub t: [u64; 32],
    /// CSR 0xFC0: the global prefix of the node.
    pub prefix: u64,
    /// CSR 0xFC1: the epoch of the node (0 unless the client sets it).
    pub epoch: u64,
    /// Instructions retired (CSRs cycle and instret return it).
    pub retired: u64,
    /// Set by the first branch or jump that depends on a tainted value.
    pub uncertain: bool,
}

/// CSR numbers.
pub const CSR_PREFIX: u16 = 0xfc0;
pub const CSR_EPOCH: u16 = 0xfc1;
const CSR_CYCLE: u16 = 0xc00;
const CSR_INSTRET: u16 = 0xc02;
const CSR_MHARTID: u16 = 0xf14;

impl Hart {
    pub fn new(pc: u64, prefix: u64) -> Self {
        Hart {
            pc,
            x: [0; 32],
            t: [0; 32],
            prefix,
            epoch: 0,
            retired: 0,
            uncertain: false,
        }
    }

    /// Register `r` and its taint.
    pub fn get(&self, r: u8) -> (u64, u64) {
        (self.x[r as usize], self.t[r as usize])
    }

    /// Writes register `r` (x0 stays 0).
    pub fn set(&mut self, r: u8, v: u64, taint: u64) {
        if r != 0 {
            self.x[r as usize] = v;
            self.t[r as usize] = taint;
        }
    }

    fn csr(&self, pc: u64, csr: u16) -> Result<u64, Fault> {
        match csr {
            CSR_PREFIX => Ok(self.prefix),
            CSR_EPOCH => Ok(self.epoch),
            CSR_CYCLE | CSR_INSTRET => Ok(self.retired),
            CSR_MHARTID => Ok(0),
            _ => Err(Fault::Csr { pc, csr }),
        }
    }

    /// Runs one instruction. Returns the `nu.*` event it raised, if any.
    pub fn step<B: Bus>(&mut self, bus: &mut B) -> Result<Option<Event>, Fault> {
        let pc = self.pc;
        let word = bus.fetch(pc)?;
        let (inst, len) = decode(word).ok_or(Fault::Illegal { pc, word })?;
        let mut next = pc.wrapping_add(len);
        let mut event = None;
        use Inst::*;
        match inst {
            Lui { rd, imm } => self.set(rd, imm as u64, 0),
            Auipc { rd, imm } => self.set(rd, pc.wrapping_add(imm as u64), 0),
            Jal { rd, imm } => {
                self.set(rd, next, 0);
                next = pc.wrapping_add(imm as u64);
            }
            Jalr { rd, rs1, imm } => {
                let (b, tb) = self.get(rs1);
                self.uncertain |= tb != 0;
                self.set(rd, next, 0);
                next = b.wrapping_add(imm as u64) & !1;
            }
            Branch { f, rs1, rs2, imm } => {
                let ((a, ta), (b, tb)) = (self.get(rs1), self.get(rs2));
                self.uncertain |= (ta | tb) != 0;
                let take = match f {
                    0 => a == b,
                    1 => a != b,
                    4 => (a as i64) < (b as i64),
                    5 => (a as i64) >= (b as i64),
                    6 => a < b,
                    7 => a >= b,
                    _ => return Err(Fault::Illegal { pc, word }),
                };
                if take {
                    next = pc.wrapping_add(imm as u64);
                }
            }
            Load {
                bytes,
                signed,
                rd,
                rs1,
                imm,
            } => {
                let (b, tb) = self.get(rs1);
                let addr = b.wrapping_add(imm as u64);
                let (v, tv) = bus.load(addr, bytes)?;
                let n = bytes as u32 * 8;
                let (v, tv) = if signed {
                    (sext(v, n), sext(tv, n))
                } else {
                    (v, tv)
                };
                // A load from an address that the build cannot know.
                self.set(rd, v, if tb != 0 { !0 } else { tv });
            }
            Store {
                bytes,
                rs1,
                rs2,
                imm,
            } => {
                let ((b, _), (v, tv)) = (self.get(rs1), self.get(rs2));
                bus.store(b.wrapping_add(imm as u64), bytes, v, tv)?;
            }
            OpImm {
                f,
                word: w,
                rd,
                rs1,
                imm,
            } => {
                let (a, ta) = self.get(rs1);
                let b = imm as u64;
                let v = alu(f, a, b, w, true).ok_or(Fault::Illegal { pc, word })?;
                self.set(rd, v, alu_taint(f, (a, ta), (b, 0), w));
            }
            Op {
                f,
                m,
                word: w,
                rd,
                rs1,
                rs2,
            } => {
                let ((a, ta), (b, tb)) = (self.get(rs1), self.get(rs2));
                let v = if m {
                    mul(f, a, b, w)
                } else {
                    alu(f, a, b, w, false)
                }
                .ok_or(Fault::Illegal { pc, word })?;
                let t = if m {
                    if (ta | tb) != 0 {
                        !0
                    } else {
                        0
                    }
                } else {
                    alu_taint(f, (a, ta), (b, tb), w)
                };
                self.set(rd, v, t);
            }
            Csr {
                f,
                rd,
                rs1,
                uimm,
                csr,
            } => {
                let old = self.csr(pc, csr)?;
                // The CSRs here are read-only: a write is ignored.
                let _ = (f, rs1, uimm);
                self.set(rd, old, 0);
            }
            Fence => {}
            Ecall | Ebreak => event = Some(Event::Break { pc }),
            Custom { f3, rd, rs1, rs2 } => {
                let ((a, ta), (b, tb)) = (self.get(rs1), self.get(rs2));
                event = Some(match f3 {
                    0 => Event::Invoke {
                        pc,
                        rd,
                        desc: a,
                        taint: ta,
                    },
                    1 => Event::Wait {
                        pc,
                        key: a,
                        taint: ta,
                    },
                    2 => Event::Version {
                        pc,
                        base: a,
                        counter: b,
                        taint: ta | tb,
                    },
                    3 => Event::Vreset {
                        pc,
                        base: a,
                        taint: ta,
                    },
                    4 => Event::VresetAll { pc },
                    5 => Event::Exit { pc },
                    _ => return Err(Fault::Illegal { pc, word }),
                });
            }
        }
        self.pc = next;
        self.retired += 1;
        Ok(event)
    }
}

/// Sign-extends the low `bits` bits of `v`.
pub fn sext(v: u64, bits: u32) -> u64 {
    let s = 64 - bits;
    (((v << s) as i64) >> s) as u64
}

/// Which bits of the result of an OP or OP-IMM instruction (`alu`) depend
/// on tainted bits of its operands, given as value and taint.
fn alu_taint(f: u8, (a, ta): (u64, u64), (b, tb): (u64, u64), word: bool) -> u64 {
    let (op, alt) = (f & 7, f & 8 != 0);
    let low = if word { 0xffff_ffff } else { !0 };
    let (ta, tb) = (ta & low, tb & low);
    let shamt = if word { 31 } else { 63 };
    let sh = (b & shamt) as u32;
    // A carry (or borrow) carries taint from the lowest tainted bit up.
    let carry = |t: u64| {
        if t == 0 {
            0
        } else {
            !0u64 << t.trailing_zeros()
        }
    };
    let t = match op {
        0 => carry(ta | tb),
        1 if tb & shamt != 0 => !0,
        1 => ta << sh,
        2 | 3 => ((ta | tb) != 0) as u64,
        4 => ta | tb,
        5 if tb & shamt != 0 => !0,
        5 if alt && word => ((ta as u32 as i32) >> sh) as u32 as u64,
        5 if alt => ((ta as i64) >> sh) as u64,
        5 => ta >> sh,
        // A known 1 decides an or, a known 0 an and.
        6 => (ta | tb) & !((a & !ta) | (b & !tb)),
        _ => (ta | tb) & !((!a & !ta) | (!b & !tb)),
    };
    if word {
        sext(t & 0xffff_ffff, 32)
    } else {
        t
    }
}

/// funct3 of OP and OP-IMM; `alt` is funct7 bit 5 (SUB, SRA).
fn alu(f: u8, a: u64, b: u64, word: bool, imm: bool) -> Option<u64> {
    let (op, alt) = (f & 7, f & 8 != 0);
    let sh = if word { b & 31 } else { b & 63 };
    let v = match op {
        0 if alt && !imm => a.wrapping_sub(b),
        0 => a.wrapping_add(b),
        1 => {
            if word {
                ((a as u32) << sh) as u64
            } else {
                a << sh
            }
        }
        2 if !word => ((a as i64) < (b as i64)) as u64,
        3 if !word => (a < b) as u64,
        4 if !word => a ^ b,
        5 if alt => {
            if word {
                ((a as i32) >> sh) as u64
            } else {
                ((a as i64) >> sh) as u64
            }
        }
        5 => {
            if word {
                ((a as u32) >> sh) as u64
            } else {
                a >> sh
            }
        }
        6 if !word => a | b,
        7 if !word => a & b,
        _ => return None,
    };
    Some(if word { sext(v, 32) } else { v })
}

/// The M extension: funct3 of OP with funct7 = 1.
fn mul(f: u8, a: u64, b: u64, word: bool) -> Option<u64> {
    if word {
        let (x, y) = (a as i32, b as i32);
        let v: i32 = match f & 7 {
            0 => x.wrapping_mul(y),
            4 => match y {
                0 => -1,
                -1 if x == i32::MIN => x,
                _ => x / y,
            },
            5 => match b as u32 {
                0 => -1,
                d => ((a as u32) / d) as i32,
            },
            6 => match y {
                0 => x,
                -1 => 0,
                _ => x % y,
            },
            7 => match b as u32 {
                0 => a as u32 as i32,
                d => ((a as u32) % d) as i32,
            },
            _ => return None,
        };
        return Some(v as i64 as u64);
    }
    let (x, y) = (a as i64, b as i64);
    Some(match f & 7 {
        0 => a.wrapping_mul(b),
        1 => ((x as i128 * y as i128) >> 64) as u64,
        2 => ((x as i128 * b as i128) >> 64) as u64,
        3 => ((a as u128 * b as u128) >> 64) as u64,
        4 => match y {
            0 => u64::MAX,
            -1 if x == i64::MIN => a,
            _ => (x / y) as u64,
        },
        5 => match b {
            0 => u64::MAX,
            _ => a / b,
        },
        6 => match y {
            0 => a,
            -1 => 0,
            _ => (x % y) as u64,
        },
        7 => match b {
            0 => a,
            _ => a % b,
        },
        _ => return None,
    })
}
