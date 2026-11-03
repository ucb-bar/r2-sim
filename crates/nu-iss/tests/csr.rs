//! The CSRs of the node: CSR 0xFC0 returns the prefix and CSR 0xFC1 the
//! epoch, both untainted. A write of CSR 0xFC1 sets the epoch and is the
//! instruction `epoch`; a write of CSR 0xFC0 is ignored. The simulator's
//! write-only CSRs 0x8C0 (mark) and 0x8C1 (cache flush).

use nu_iss::{Bus, Event, Fault, Hart, CSR_CACHE_FLUSH, CSR_EPOCH, CSR_MARK, CSR_PREFIX};

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

/// A CSR instruction of funct3 `f` (1 to 3: csrrw, csrrs, csrrc; 5 to 7:
/// their immediate forms, `rs1` the immediate).
fn csr_insn(f: u32, rd: u32, csr: u16, rs1: u32) -> u32 {
    (csr as u32) << 20 | rs1 << 15 | f << 12 | rd << 7 | 0x73
}

/// `csrrs rd, csr, rs1` (with rs1 = x0: `csrr rd, csr`).
fn csrrs(rd: u32, csr: u16, rs1: u32) -> u32 {
    csr_insn(2, rd, csr, rs1)
}

#[test]
fn the_epoch_csr_returns_the_epoch_untainted() {
    let mut h = Hart::new(0, 0x2000);
    assert_eq!(h.epoch, 0, "a new hart is in epoch 0");
    h.epoch = 3;
    h.step(&mut One(csrrs(5, CSR_EPOCH, 0))).unwrap();
    assert_eq!(h.get(5), (3, 0));
    h.step(&mut One(csrrs(6, CSR_PREFIX, 0))).unwrap();
    assert_eq!(h.get(6), (0x2000, 0));
    // A write of the prefix is ignored.
    h.set(7, 9, 0);
    assert_eq!(h.step(&mut One(csrrs(8, CSR_PREFIX, 7))).unwrap(), None);
    assert_eq!((h.get(8), h.prefix), ((0x2000, 0), 0x2000));
}

#[test]
fn a_write_of_the_epoch_csr_is_the_instruction_epoch() {
    // (funct3, source register or immediate, value of the source, the
    // epoch after it from epoch 0b1100).
    let (rw, rs, rc, rwi, rsi, rci) = (1, 2, 3, 5, 6, 7);
    let cases = [
        (rw, 7, 5, Some(5)),
        (rw, 0, 0, Some(0)),
        (rs, 7, 0b0011, Some(0b1111)),
        (rs, 0, 0, None),
        (rc, 7, 0b0100, Some(0b1000)),
        (rc, 0, 0, None),
        (rwi, 9, 9, Some(9)),
        (rwi, 0, 0, Some(0)),
        (rsi, 1, 1, Some(0b1101)),
        (rsi, 0, 0, None),
        (rci, 8, 8, Some(0b0100)),
        (rci, 0, 0, None),
    ];
    for (f, src, value, after) in cases {
        let mut h = Hart::new(0x40, 0x2000);
        h.epoch = 0b1100;
        if f < 5 {
            h.set(src as u8, value, 0);
        }
        let ev = h.step(&mut One(csr_insn(f, 5, CSR_EPOCH, src))).unwrap();
        let what = format!("funct3 {f}, source {src}");
        assert_eq!(h.get(5), (0b1100, 0), "{what}: the old value goes to rd");
        match after {
            Some(e) => {
                assert_eq!(
                    ev,
                    Some(Event::Epoch {
                        pc: 0x40,
                        epoch: e,
                        taint: 0
                    }),
                    "{what}"
                );
                assert_eq!(h.epoch, e, "{what}");
            }
            None => assert_eq!((ev, h.epoch), (None, 0b1100), "{what}: a read only"),
        }
        // A later read returns the new epoch.
        h.step(&mut One(csrrs(6, CSR_EPOCH, 0))).unwrap();
        assert_eq!(h.get(6), (h.epoch, 0), "{what}");
    }
}

#[test]
fn a_write_of_a_simulator_csr_is_an_event_and_a_read_is_a_fault() {
    // csrw 0x8c0, x7 with x7 = 42 (tainted bit 0): a mark of id 42.
    let mut h = Hart::new(0x40, 0x2000);
    h.set(7, 42, 1);
    let ev = h.step(&mut One(csr_insn(1, 0, CSR_MARK, 7))).unwrap();
    assert_eq!(
        ev,
        Some(Event::Mark {
            pc: 0x40,
            id: 42,
            taint: 1
        })
    );
    // csrwi 0x8c0, 9 and csrwi 0x8c1, 0.
    let ev = h.step(&mut One(csr_insn(5, 0, CSR_MARK, 9))).unwrap();
    assert_eq!(
        ev,
        Some(Event::Mark {
            pc: 0x44,
            id: 9,
            taint: 0
        })
    );
    let ev = h
        .step(&mut One(csr_insn(5, 0, CSR_CACHE_FLUSH, 0)))
        .unwrap();
    assert_eq!(ev, Some(Event::CacheFlush { pc: 0x48 }));
    // A read: csrr, csrrs and csrrc with a source, csrrw with rd != x0.
    for (f, rd, rs1) in [(2, 5, 0), (2, 0, 7), (3, 0, 7), (1, 5, 7), (6, 0, 1)] {
        for csr in [CSR_MARK, CSR_CACHE_FLUSH] {
            let e = h.step(&mut One(csr_insn(f, rd, csr, rs1))).unwrap_err();
            assert_eq!(
                e,
                Fault::Csr { pc: h.pc, csr },
                "funct3 {f} rd {rd} rs1 {rs1}"
            );
        }
    }
}
