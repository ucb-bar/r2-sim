//! The shadow call stack of the hart (`Hart::calls`): the return-address
//! hints of the RISC-V unprivileged spec, x1 and x5 the link registers.

use nu_iss::{Bus, CallStack, Fault, Hart};

/// A memory of one instruction at every address; no data.
struct One(u32);

impl Bus for One {
    fn fetch(&mut self, _pc: u64) -> Result<u32, Fault> {
        Ok(self.0)
    }

    fn load(&mut self, addr: u64, _bytes: u8) -> Result<(u64, u64), Fault> {
        Err(Fault::Access { addr, write: false })
    }

    fn store(&mut self, addr: u64, _: u8, _: u64, _: u64) -> Result<(), Fault> {
        Err(Fault::Access { addr, write: true })
    }
}

/// `jal rd, 8`.
fn jal(rd: u32) -> u32 {
    (8 >> 1) << 21 | rd << 7 | 0x6f
}

/// `jalr rd, 0(rs1)`.
fn jalr(rd: u32, rs1: u32) -> u32 {
    rs1 << 15 | rd << 7 | 0x67
}

/// Runs `insn` at `pc` on `h`.
fn at(h: &mut Hart, pc: u64, insn: u32) {
    h.pc = pc;
    h.step(&mut One(insn)).unwrap();
}

#[test]
fn calls_push_and_returns_pop() {
    let mut h = Hart::new(0, 0);
    h.set(6, 0x100, 0);
    at(&mut h, 0x10, jal(1));
    at(&mut h, 0x20, jalr(5, 6));
    assert_eq!(
        h.calls.frames,
        [0x10, 0x20],
        "jal ra and jalr t0, (t6) push"
    );
    at(&mut h, 0x30, jal(0));
    at(&mut h, 0x34, jalr(0, 6));
    at(&mut h, 0x38, jalr(7, 6));
    assert_eq!(h.calls.frames, [0x10, 0x20], "no link register: no change");
    at(&mut h, 0x40, jalr(0, 5));
    assert_eq!(h.calls.frames, [0x10], "jr t0 pops");
    at(&mut h, 0x44, jalr(0, 1));
    assert!(h.calls.frames.is_empty(), "ret pops");
    assert_eq!(h.calls.underflows, 0);
}

#[test]
fn two_link_registers_pop_then_push_or_push() {
    let mut s = CallStack::default();
    s.jump(0x10, 1, None);
    // rd and rs1 both link registers and different: a coroutine swap.
    s.jump(0x20, 5, Some(1));
    assert_eq!(s.frames, [0x20]);
    // The same link register: push.
    s.jump(0x30, 1, Some(1));
    assert_eq!(s.frames, [0x20, 0x30]);
}

#[test]
fn compressed_calls_and_returns_follow_the_same_rules() {
    let mut h = Hart::new(0, 0);
    h.set(6, 0x100, 0);
    // c.jalr t1 (funct4 1001, rs1 6, rs2 0, op 10): pushes.
    at(&mut h, 0x10, 0b1001 << 12 | 6 << 7 | 0b10);
    assert_eq!(h.calls.frames, [0x10]);
    // c.jr ra (ret: funct4 1000, rs1 1, rs2 0, op 10): pops.
    at(&mut h, 0x12, 0b1000 << 12 | 1 << 7 | 0b10);
    assert!(h.calls.frames.is_empty());
}

#[test]
fn a_pop_on_an_empty_stack_is_ignored_and_counted() {
    let mut h = Hart::new(0, 0);
    at(&mut h, 0x10, jalr(0, 1));
    at(&mut h, 0x14, jalr(0, 5));
    assert!(h.calls.frames.is_empty());
    assert_eq!(h.calls.underflows, 2);
    at(&mut h, 0x18, jal(1));
    assert_eq!(h.calls.frames, [0x18]);
}
