//! Decoding of RV64I, M, Zicsr, the compressed C extension (integer
//! instructions only) and the `nu.*` instructions into [`Inst`].

/// One decoded instruction. Immediates are sign-extended to 64 bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inst {
    Lui {
        rd: u8,
        imm: i64,
    },
    Auipc {
        rd: u8,
        imm: i64,
    },
    Jal {
        rd: u8,
        imm: i64,
    },
    Jalr {
        rd: u8,
        rs1: u8,
        imm: i64,
    },
    /// funct3: 0 beq, 1 bne, 4 blt, 5 bge, 6 bltu, 7 bgeu.
    Branch {
        f: u8,
        rs1: u8,
        rs2: u8,
        imm: i64,
    },
    Load {
        bytes: u8,
        signed: bool,
        rd: u8,
        rs1: u8,
        imm: i64,
    },
    Store {
        bytes: u8,
        rs1: u8,
        rs2: u8,
        imm: i64,
    },
    /// `f`: funct3, plus 8 for SRAI.
    OpImm {
        f: u8,
        word: bool,
        rd: u8,
        rs1: u8,
        imm: i64,
    },
    /// `f`: funct3, plus 8 for SUB and SRA; `m`: the M extension.
    Op {
        f: u8,
        m: bool,
        word: bool,
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
    /// CSRRW/S/C (`f` 1, 2, 3) and their immediate forms (`f` 5, 6, 7).
    Csr {
        f: u8,
        rd: u8,
        rs1: u8,
        uimm: u8,
        csr: u16,
    },
    Fence,
    Ecall,
    Ebreak,
    /// custom-0: a `nu.*` instruction.
    Custom {
        f3: u8,
        rd: u8,
        rs1: u8,
        rs2: u8,
    },
}

/// Bits `hi..=lo` of `w`.
fn bits(w: u32, hi: u32, lo: u32) -> u32 {
    (w >> lo) & ((1 << (hi - lo + 1)) - 1)
}

fn sx(v: u32, width: u32) -> i64 {
    crate::sext(v as u64, width) as i64
}

/// Decodes the instruction in the low bits of `w` and gives its length in
/// bytes (2 or 4). `None` for an instruction outside the subset.
pub fn decode(w: u32) -> Option<(Inst, u64)> {
    if w & 3 != 3 {
        return decode_c(w as u16).map(|i| (i, 2));
    }
    decode_32(w).map(|i| (i, 4))
}

fn decode_32(w: u32) -> Option<Inst> {
    let rd = bits(w, 11, 7) as u8;
    let rs1 = bits(w, 19, 15) as u8;
    let rs2 = bits(w, 24, 20) as u8;
    let f3 = bits(w, 14, 12) as u8;
    let f7 = bits(w, 31, 25);
    let imm_i = sx(bits(w, 31, 20), 12);
    let imm_s = sx(bits(w, 31, 25) << 5 | bits(w, 11, 7), 12);
    let imm_b = sx(
        bits(w, 31, 31) << 12 | bits(w, 7, 7) << 11 | bits(w, 30, 25) << 5 | bits(w, 11, 8) << 1,
        13,
    );
    let imm_u = sx(w & 0xffff_f000, 32);
    let imm_j = sx(
        bits(w, 31, 31) << 20
            | bits(w, 19, 12) << 12
            | bits(w, 20, 20) << 11
            | bits(w, 30, 21) << 1,
        21,
    );
    use Inst::*;
    Some(match w & 0x7f {
        0x37 => Lui { rd, imm: imm_u },
        0x17 => Auipc { rd, imm: imm_u },
        0x6f => Jal { rd, imm: imm_j },
        0x67 if f3 == 0 => Jalr {
            rd,
            rs1,
            imm: imm_i,
        },
        0x63 if f3 != 2 && f3 != 3 => Branch {
            f: f3,
            rs1,
            rs2,
            imm: imm_b,
        },
        0x03 => {
            let (bytes, signed) = match f3 {
                0 => (1, true),
                1 => (2, true),
                2 => (4, true),
                3 => (8, true),
                4 => (1, false),
                5 => (2, false),
                6 => (4, false),
                _ => return None,
            };
            Load {
                bytes,
                signed,
                rd,
                rs1,
                imm: imm_i,
            }
        }
        0x23 if f3 <= 3 => Store {
            bytes: 1 << f3,
            rs1,
            rs2,
            imm: imm_s,
        },
        0x13 | 0x1b => {
            let word = w & 0x7f == 0x1b;
            match f3 {
                1 | 5 => {
                    let (shamt, top) = if word {
                        (bits(w, 24, 20), bits(w, 31, 25))
                    } else {
                        (bits(w, 25, 20), bits(w, 31, 26) << 1)
                    };
                    let alt = top == 0x20;
                    if top != 0 && !alt || (f3 == 1 && alt) {
                        return None;
                    }
                    OpImm {
                        f: f3 | (alt as u8) << 3,
                        word,
                        rd,
                        rs1,
                        imm: shamt as i64,
                    }
                }
                0 => OpImm {
                    f: 0,
                    word,
                    rd,
                    rs1,
                    imm: imm_i,
                },
                _ if word => return None,
                _ => OpImm {
                    f: f3,
                    word,
                    rd,
                    rs1,
                    imm: imm_i,
                },
            }
        }
        0x33 | 0x3b => {
            let word = w & 0x7f == 0x3b;
            match f7 {
                0 => Op {
                    f: f3,
                    m: false,
                    word,
                    rd,
                    rs1,
                    rs2,
                },
                0x20 if f3 == 0 || f3 == 5 => Op {
                    f: f3 | 8,
                    m: false,
                    word,
                    rd,
                    rs1,
                    rs2,
                },
                1 => Op {
                    f: f3,
                    m: true,
                    word,
                    rd,
                    rs1,
                    rs2,
                },
                _ => return None,
            }
        }
        0x0f => Fence,
        0x73 => match (f3, w >> 7) {
            (0, 0) => Ecall,
            (0, 0x2000) => Ebreak,
            (1..=3 | 5..=7, _) => Csr {
                f: f3,
                rd,
                rs1,
                uimm: rs1,
                csr: bits(w, 31, 20) as u16,
            },
            _ => return None,
        },
        0x0b => Custom { f3, rd, rs1, rs2 },
        _ => return None,
    })
}

/// A 16-bit instruction of RV64C (integer instructions only).
fn decode_c(h: u16) -> Option<Inst> {
    let w = h as u32;
    if w == 0 {
        return None;
    }
    let op = bits(w, 1, 0);
    let f3 = bits(w, 15, 13);
    // Full and compressed register fields.
    let rd = bits(w, 11, 7) as u8;
    let rs2 = bits(w, 6, 2) as u8;
    let rdp = bits(w, 4, 2) as u8 + 8;
    let rs1p = bits(w, 9, 7) as u8 + 8;
    let imm6 = sx(bits(w, 12, 12) << 5 | bits(w, 6, 2), 6);
    let uimm_lw = (bits(w, 12, 10) << 3 | bits(w, 6, 6) << 2 | bits(w, 5, 5) << 6) as i64;
    let uimm_ld = (bits(w, 12, 10) << 3 | bits(w, 6, 5) << 6) as i64;
    use Inst::*;
    Some(match (op, f3) {
        (0, 0) => {
            let nz = bits(w, 12, 11) << 4
                | bits(w, 10, 7) << 6
                | bits(w, 6, 6) << 2
                | bits(w, 5, 5) << 3;
            if nz == 0 {
                return None;
            }
            OpImm {
                f: 0,
                word: false,
                rd: rdp,
                rs1: 2,
                imm: nz as i64,
            }
        }
        (0, 2) => Load {
            bytes: 4,
            signed: true,
            rd: rdp,
            rs1: rs1p,
            imm: uimm_lw,
        },
        (0, 3) => Load {
            bytes: 8,
            signed: true,
            rd: rdp,
            rs1: rs1p,
            imm: uimm_ld,
        },
        (0, 6) => Store {
            bytes: 4,
            rs1: rs1p,
            rs2: rdp,
            imm: uimm_lw,
        },
        (0, 7) => Store {
            bytes: 8,
            rs1: rs1p,
            rs2: rdp,
            imm: uimm_ld,
        },
        (1, 0) => OpImm {
            f: 0,
            word: false,
            rd,
            rs1: rd,
            imm: imm6,
        },
        (1, 1) if rd != 0 => OpImm {
            f: 0,
            word: true,
            rd,
            rs1: rd,
            imm: imm6,
        },
        (1, 2) => OpImm {
            f: 0,
            word: false,
            rd,
            rs1: 0,
            imm: imm6,
        },
        (1, 3) if rd == 2 => {
            let nz = bits(w, 12, 12) << 9
                | bits(w, 6, 6) << 4
                | bits(w, 5, 5) << 6
                | bits(w, 4, 3) << 7
                | bits(w, 2, 2) << 5;
            if nz == 0 {
                return None;
            }
            OpImm {
                f: 0,
                word: false,
                rd: 2,
                rs1: 2,
                imm: sx(nz, 10),
            }
        }
        (1, 3) => {
            let nz = bits(w, 12, 12) << 17 | bits(w, 6, 2) << 12;
            if nz == 0 {
                return None;
            }
            Lui {
                rd,
                imm: sx(nz, 18),
            }
        }
        (1, 4) => {
            let shamt = (bits(w, 12, 12) << 5 | bits(w, 6, 2)) as i64;
            match bits(w, 11, 10) {
                0 => OpImm {
                    f: 5,
                    word: false,
                    rd: rs1p,
                    rs1: rs1p,
                    imm: shamt,
                },
                1 => OpImm {
                    f: 5 | 8,
                    word: false,
                    rd: rs1p,
                    rs1: rs1p,
                    imm: shamt,
                },
                2 => OpImm {
                    f: 7,
                    word: false,
                    rd: rs1p,
                    rs1: rs1p,
                    imm: imm6,
                },
                _ => {
                    let (word, f) = match (bits(w, 12, 12), bits(w, 6, 5)) {
                        (0, 0) => (false, 8),
                        (0, 1) => (false, 4),
                        (0, 2) => (false, 6),
                        (0, 3) => (false, 7),
                        (1, 0) => (true, 8),
                        (1, 1) => (true, 0),
                        _ => return None,
                    };
                    Op {
                        f,
                        m: false,
                        word,
                        rd: rs1p,
                        rs1: rs1p,
                        rs2: rdp,
                    }
                }
            }
        }
        (1, 5) => {
            let i = bits(w, 12, 12) << 11
                | bits(w, 11, 11) << 4
                | bits(w, 10, 9) << 8
                | bits(w, 8, 8) << 10
                | bits(w, 7, 7) << 6
                | bits(w, 6, 6) << 7
                | bits(w, 5, 3) << 1
                | bits(w, 2, 2) << 5;
            Jal {
                rd: 0,
                imm: sx(i, 12),
            }
        }
        (1, 6) | (1, 7) => {
            let i = bits(w, 12, 12) << 8
                | bits(w, 11, 10) << 3
                | bits(w, 6, 5) << 6
                | bits(w, 4, 3) << 1
                | bits(w, 2, 2) << 5;
            Branch {
                f: (f3 == 7) as u8,
                rs1: rs1p,
                rs2: 0,
                imm: sx(i, 9),
            }
        }
        (2, 0) => OpImm {
            f: 1,
            word: false,
            rd,
            rs1: rd,
            imm: (bits(w, 12, 12) << 5 | bits(w, 6, 2)) as i64,
        },
        (2, 2) if rd != 0 => {
            let u = bits(w, 12, 12) << 5 | bits(w, 6, 4) << 2 | bits(w, 3, 2) << 6;
            Load {
                bytes: 4,
                signed: true,
                rd,
                rs1: 2,
                imm: u as i64,
            }
        }
        (2, 3) if rd != 0 => {
            let u = bits(w, 12, 12) << 5 | bits(w, 6, 5) << 3 | bits(w, 4, 2) << 6;
            Load {
                bytes: 8,
                signed: true,
                rd,
                rs1: 2,
                imm: u as i64,
            }
        }
        (2, 4) => match (bits(w, 12, 12), rd, rs2) {
            (0, 0, _) => return None,
            (0, _, 0) => Jalr {
                rd: 0,
                rs1: rd,
                imm: 0,
            },
            (0, _, _) => Op {
                f: 0,
                m: false,
                word: false,
                rd,
                rs1: 0,
                rs2,
            },
            (1, 0, 0) => Ebreak,
            (1, _, 0) => Jalr {
                rd: 1,
                rs1: rd,
                imm: 0,
            },
            (1, _, _) => Op {
                f: 0,
                m: false,
                word: false,
                rd,
                rs1: rd,
                rs2,
            },
            _ => return None,
        },
        (2, 6) => {
            let u = bits(w, 12, 9) << 2 | bits(w, 8, 7) << 6;
            Store {
                bytes: 4,
                rs1: 2,
                rs2,
                imm: u as i64,
            }
        }
        (2, 7) => {
            let u = bits(w, 12, 10) << 3 | bits(w, 9, 7) << 6;
            Store {
                bytes: 8,
                rs1: 2,
                rs2,
                imm: u as i64,
            }
        }
        _ => return None,
    })
}
