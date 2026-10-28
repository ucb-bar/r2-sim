//! The CSRs of the node: CSR 0xFC0 returns the prefix and CSR 0xFC1 the
//! epoch, both untainted; a write is ignored.

use nu_iss::{Bus, Fault, Hart, CSR_EPOCH, CSR_PREFIX};

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

/// `csrrs rd, csr, rs1` (with rs1 = x0: `csrr rd, csr`).
fn csrrs(rd: u32, csr: u16, rs1: u32) -> u32 {
    (csr as u32) << 20 | rs1 << 15 | 2 << 12 | rd << 7 | 0x73
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
    // A write (rs1 != x0) leaves the epoch as it is.
    h.set(7, 9, 0);
    h.step(&mut One(csrrs(8, CSR_EPOCH, 7))).unwrap();
    assert_eq!((h.get(8), h.epoch), ((3, 0), 3));
}
