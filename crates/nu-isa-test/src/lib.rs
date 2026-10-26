//! The ISA tests of the Neutrino scalar core (RV64IMC, Zicsr and the
//! `nu.*` instructions), from a checkout of r2-isa-tests: their discovery,
//! their build with clang and lld, and their run on nu-iss or in r2-sim.
//!
//! Suites ([`SUITES`]):
//! - `rv64ui`, `rv64um`, `rv64uc`: riscv-tests (a pinned submodule of
//!   r2-isa-tests), every test but rv64ui `fence_i`. That one tests
//!   Zifencei with code that changes itself, and the IMEM of a Neutrino is
//!   read-only for its core.
//! - `nu`: the `nu.*` instructions and the prefix CSR.
//! - `sim`: the `nu.*` instructions in r2-sim (`r2sim run --program rv`
//!   on the root of a configuration), against the Neutrino model of
//!   r2-sim. It needs the SoC constants of that configuration as a C
//!   header, `soc.h`.
//!
//! On nu-iss ([`run_iss`]) a test runs in one flat memory of
//! [`MEM_BYTES`] from address 0, as in the "p" environment of
//! riscv-tests. It passes when it reaches `nu.exit` with gp = 1, and its
//! test case n fails at `ebreak` with gp = 2n + 1. The `nu.*` instructions
//! run against a model of one Neutrino's version table: a versioned invoke
//! (flag bit 0 set, bit 1 clear) returns the next version of its base
//! granule, and any other invoke returns 0; `nu.version` sets a counter,
//! `nu.vreset` and `nu.vreset_all` clear counters; a `nu.wait` on a
//! version that no invoke made is an error, since on a Neutrino the
//! program would block forever.
//!
//! In r2-sim ([`run_sim`]) a test passes when the run ends without an
//! error. Its test case n stops the run at the `ebreak` at `nu_fail_table
//! + 4n`, and the PC of the error gives n.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use nu_iss::{Bus, Event, Fault, Hart};
use object::{Object, ObjectSegment, ObjectSymbol};

/// Bytes of the flat memory of a test on nu-iss.
pub const MEM_BYTES: usize = 64 * 1024;
/// CSR 0xFC0 of the test hart on nu-iss (`NU_TEST_PREFIX` in
/// env/riscv_test.h of r2-isa-tests).
pub const PREFIX: u64 = 0x12_0000_0000;
/// The granule address bits of a key in the model (`NU_TEST_KEY_BITS` in
/// env/riscv_test.h of r2-isa-tests).
pub const KEY_BITS: u32 = 29;
/// A test on nu-iss that runs longer than this stops with an error.
pub const LIMIT: u64 = 10_000_000;

/// The suites, in their order.
pub const SUITES: [&str; 5] = ["rv64ui", "rv64um", "rv64uc", "nu", "sim"];
/// Tests that do not apply to the Neutrino scalar core.
pub const EXCLUDED: [(&str, &str); 1] = [("rv64ui", "fence_i")];

/// One test: its suite, its name and its source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Test {
    pub suite: String,
    pub name: String,
    pub src: PathBuf,
}

/// How a test ended; `detail` is what it took (instructions on nu-iss,
/// cycles in r2-sim).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Pass {
        detail: String,
    },
    /// Test case `case` failed.
    Fail {
        case: u64,
        detail: String,
    },
    /// The test did not reach a pass or a fail.
    Error(String),
}

/// The tests of `suites`, each sorted by name, from the checkout
/// `isa_tests` of r2-isa-tests.
pub fn tests(isa_tests: &Path, suites: &[&str]) -> Result<Vec<Test>, String> {
    let mut out = Vec::new();
    for &suite in suites {
        let dir = match suite {
            "nu" | "sim" => isa_tests.join(suite),
            "rv64ui" | "rv64um" | "rv64uc" => {
                isa_tests.join("third_party/riscv-tests/isa").join(suite)
            }
            s => return Err(format!("unknown suite {s} (known: {SUITES:?})")),
        };
        let entries = std::fs::read_dir(&dir).map_err(|e| {
            format!(
                "{}: {e} (the tests: git submodule update --init --recursive isa-tests)",
                dir.display()
            )
        })?;
        let mut names: Vec<(String, PathBuf)> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "S"))
            .filter_map(|p| Some((p.file_stem()?.to_string_lossy().into_owned(), p)))
            .filter(|(n, _)| !EXCLUDED.contains(&(suite, n.as_str())))
            .collect();
        names.sort();
        out.extend(names.into_iter().map(|(name, src)| Test {
            suite: suite.to_string(),
            name,
            src,
        }));
    }
    Ok(out)
}

/// Builds `t` with `clang` into `out/<suite>-<name>.elf`; returns its
/// path. A `sim` test needs `soc_dir`, the directory of `soc.h`: it gets
/// `-DNU_SIM`, the include directories `sim` and `soc_dir`, and the linker
/// script env/sim.ld instead of env/link.ld.
pub fn build(
    isa_tests: &Path,
    clang: &str,
    t: &Test,
    out: &Path,
    soc_dir: Option<&Path>,
) -> Result<PathBuf, String> {
    let elf = out.join(format!("{}-{}.elf", t.suite, t.name));
    let env = isa_tests.join("env");
    let sim = t.suite == "sim";
    let script = env.join(if sim { "sim.ld" } else { "link.ld" });
    let mut cmd = Command::new(clang);
    if sim {
        let soc = soc_dir.ok_or("a sim test needs the directory of soc.h")?;
        cmd.arg("-DNU_SIM")
            .arg(format!("-I{}", isa_tests.join("sim").display()))
            .arg(format!("-I{}", soc.display()));
    }
    let output = cmd
        .args([
            "--target=riscv64-unknown-elf",
            "-march=rv64imc_zicsr",
            "-mabi=lp64",
            "-mcmodel=medany",
            "-nostdlib",
            "-static",
            "-fuse-ld=lld",
        ])
        .arg(format!("-Wl,-T,{}", script.display()))
        .arg(format!("-I{}", env.display()))
        .arg(format!(
            "-I{}",
            isa_tests
                .join("third_party/riscv-tests/isa/macros/scalar")
                .display()
        ))
        .arg("-o")
        .arg(&elf)
        .arg(&t.src)
        .output()
        .map_err(|e| format!("{clang}: {e} (run under `pixi run`)"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    Ok(elf)
}

/// The flat memory of a test: code and data from address 0.
struct Flat(Vec<u8>);

impl Flat {
    fn range(&self, addr: u64, n: u64, write: bool) -> Result<std::ops::Range<usize>, Fault> {
        match addr.checked_add(n) {
            Some(end) if end <= self.0.len() as u64 => Ok(addr as usize..end as usize),
            _ => Err(Fault::Access { addr, write }),
        }
    }
}

impl Bus for Flat {
    fn fetch(&mut self, pc: u64) -> Result<u32, Fault> {
        let r = self.range(pc, 2, false)?;
        let lo = u16::from_le_bytes([self.0[r.start], self.0[r.start + 1]]) as u32;
        let hi = self
            .0
            .get(r.end..r.end + 2)
            .map_or(0, |b| u16::from_le_bytes([b[0], b[1]]) as u32);
        Ok(lo | hi << 16)
    }

    fn load(&mut self, addr: u64, bytes: u8) -> Result<(u64, u64), Fault> {
        let r = self.range(addr, bytes as u64, false)?;
        let mut v = [0u8; 8];
        v[..bytes as usize].copy_from_slice(&self.0[r]);
        Ok((u64::from_le_bytes(v), 0))
    }

    fn store(&mut self, addr: u64, bytes: u8, value: u64, _taint: u64) -> Result<(), Fault> {
        let r = self.range(addr, bytes as u64, true)?;
        self.0[r].copy_from_slice(&value.to_le_bytes()[..bytes as usize]);
        Ok(())
    }
}

/// The version table of the model Neutrino, by base granule.
#[derive(Default)]
struct Versions(HashMap<u64, u64>);

impl Versions {
    /// The result of `nu.invoke` on the descriptor at `desc`.
    fn invoke(&mut self, mem: &Flat, desc: u64) -> Result<u64, String> {
        let r = mem
            .range(desc, 24, false)
            .map_err(|_| format!("descriptor at {desc:#x} outside the memory"))?;
        let h = &mem.0[r];
        let flags = u16::from_le_bytes([h[4], h[5]]);
        let base = u64::from_le_bytes(h[8..16].try_into().expect("8 bytes"));
        let versioned = flags & 1 != 0 && flags & 2 == 0;
        Ok(if versioned {
            let v = self.0.entry(base).or_insert(0);
            *v += 1;
            *v
        } else {
            0
        })
    }

    /// Checks that `nu.wait` on `key` can return.
    fn wait(&self, key: u64) -> Result<(), String> {
        let (version, granule) = (key >> KEY_BITS, key & ((1 << KEY_BITS) - 1));
        let made = self.0.get(&granule).copied().unwrap_or(0);
        if version > made {
            return Err(format!(
                "nu.wait on version {version} of granule {granule:#x}, of which no invoke made more than {made}: the program would block"
            ));
        }
        Ok(())
    }
}

/// Runs the image `mem` from `entry` on nu-iss; the outcome and the hart
/// at the end.
fn execute(mut mem: Flat, entry: u64) -> (Outcome, Hart) {
    let mut hart = Hart::new(entry, PREFIX);
    let mut versions = Versions::default();
    loop {
        if hart.retired >= LIMIT {
            return (
                Outcome::Error(format!("no end after {LIMIT} instructions")),
                hart,
            );
        }
        let pc = hart.pc;
        let ev = match hart.step(&mut mem) {
            Ok(ev) => ev,
            Err(f) => return (Outcome::Error(format!("{f:?} at pc {pc:#x}")), hart),
        };
        let result = match ev {
            None => Ok(()),
            Some(Event::Exit { .. }) => {
                let o = match hart.get(3).0 {
                    1 => Outcome::Pass {
                        detail: format!("{} instructions", hart.retired),
                    },
                    gp => Outcome::Error(format!("nu.exit at pc {pc:#x} with gp = {gp}")),
                };
                return (o, hart);
            }
            Some(Event::Break { .. }) => {
                let o = Outcome::Fail {
                    case: hart.get(3).0 >> 1,
                    detail: format!("{} instructions", hart.retired),
                };
                return (o, hart);
            }
            Some(Event::Invoke { rd, desc, .. }) => {
                versions.invoke(&mem, desc).map(|v| hart.set(rd, v, 0))
            }
            Some(Event::Wait { key, .. }) => versions.wait(key),
            Some(Event::Version { base, counter, .. }) => {
                versions.0.insert(base, counter);
                Ok(())
            }
            Some(Event::Vreset { base, .. }) => {
                versions.0.remove(&base);
                Ok(())
            }
            Some(Event::VresetAll { .. }) => {
                versions.0.clear();
                Ok(())
            }
        };
        if let Err(e) = result {
            return (Outcome::Error(format!("pc {pc:#x}: {e}")), hart);
        }
    }
}

/// The flat memory of the ELF at `path`, its PT_LOAD segments at their
/// addresses, and its entry.
fn load(path: &Path) -> Result<(Flat, u64), String> {
    let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let elf = object::File::parse(&*raw).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut mem = vec![0u8; MEM_BYTES];
    for seg in elf.segments() {
        let data = seg.data().map_err(|e| e.to_string())?;
        let at = seg.address() as usize;
        let end = at + seg.size() as usize;
        if end > MEM_BYTES {
            return Err(format!(
                "{}: a segment ends at {end:#x}, past the {MEM_BYTES} B memory",
                path.display()
            ));
        }
        mem[at..at + data.len()].copy_from_slice(data);
    }
    Ok((Flat(mem), elf.entry()))
}

/// Runs the test ELF `elf` on nu-iss.
pub fn run_iss(elf: &Path) -> Outcome {
    match load(elf) {
        Ok((mem, entry)) => execute(mem, entry).0,
        Err(e) => Outcome::Error(e),
    }
}

/// Runs the `sim` test ELF `elf` in r2-sim (the binary `r2sim`) on
/// configuration `config`, with its image file and working directory in
/// `out`.
pub fn run_sim(r2sim: &Path, config: &Path, elf: &Path, out: &Path) -> Outcome {
    let elf = match std::fs::canonicalize(elf) {
        Ok(p) => p,
        Err(e) => return Outcome::Error(format!("{}: {e}", elf.display())),
    };
    let toml = elf.with_extension("toml");
    let entry = format!("[[entry]]\nimem = \"root\"\nelf = \"{}\"\n", elf.display());
    if let Err(e) = std::fs::write(&toml, entry) {
        return Outcome::Error(format!("{}: {e}", toml.display()));
    }
    let output = Command::new(r2sim)
        .args(["run", "--config"])
        .arg(config)
        .args(["--program", "rv", "--image"])
        .arg(&toml)
        .current_dir(out)
        .output();
    let output = match output {
        Ok(o) => o,
        Err(e) => return Outcome::Error(format!("{}: {e}", r2sim.display())),
    };
    let text = String::from_utf8_lossy(&output.stdout).into_owned()
        + &String::from_utf8_lossy(&output.stderr);
    sim_outcome(output.status.success(), &text, fail_table(&elf))
}

/// The outcome of an r2-sim run that printed `text`, for a test whose
/// fail table is at `table`.
fn sim_outcome(success: bool, text: &str, table: Option<u64>) -> Outcome {
    if success && !text.contains("FATAL") {
        // The summary line: "end N cycles ...".
        let cycles = text
            .split("end ")
            .nth(1)
            .and_then(|r| r.split_whitespace().next())
            .unwrap_or("?");
        return Outcome::Pass {
            detail: format!("{cycles} cycles"),
        };
    }
    let pc = text.split("pc 0x").nth(1).and_then(|r| {
        let hex: String = r.chars().take_while(|c| c.is_ascii_hexdigit()).collect();
        u64::from_str_radix(&hex, 16).ok()
    });
    match (pc, table) {
        (Some(pc), Some(t)) if pc >= t && (pc - t) % 4 == 0 && (pc - t) / 4 < 256 => {
            Outcome::Fail {
                case: (pc - t) / 4,
                detail: format!("at pc {pc:#x}"),
            }
        }
        _ => Outcome::Error(
            text.lines()
                .find(|l| l.contains("FATAL") || l.contains("error"))
                .unwrap_or("r2sim failed")
                .trim()
                .to_string(),
        ),
    }
}

/// The address of `nu_fail_table` in the ELF at `path`.
fn fail_table(path: &Path) -> Option<u64> {
    let raw = std::fs::read(path).ok()?;
    let elf = object::File::parse(&*raw).ok()?;
    elf.symbols()
        .find(|s| s.name() == Ok("nu_fail_table"))
        .map(|s| s.address())
}

#[cfg(test)]
mod tests {
    use super::*;

    const GP: u32 = 3;
    const A0: u32 = 10;
    const A1: u32 = 11;

    fn addi(rd: u32, rs1: u32, imm: i32) -> u32 {
        ((imm as u32 & 0xfff) << 20) | (rs1 << 15) | (rd << 7) | 0x13
    }
    fn lui(rd: u32, imm20: u32) -> u32 {
        (imm20 << 12) | (rd << 7) | 0x37
    }
    fn nu(f3: u32, rd: u32, rs1: u32) -> u32 {
        (rs1 << 15) | (f3 << 12) | (rd << 7) | 0x0b
    }
    const EBREAK: u32 = 0x0010_0073;
    const EXIT: u32 = 5 << 12 | 0x0b;

    /// Runs `code` from address 0, with a descriptor header at 0x1000 of
    /// flags `flags` and base granule 0x40.
    fn exec(code: &[u32], flags: u16) -> (Outcome, Hart) {
        let mut m = vec![0u8; MEM_BYTES];
        for (i, w) in code.iter().enumerate() {
            m[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        m[0x1004..0x1006].copy_from_slice(&flags.to_le_bytes());
        m[0x1008..0x1010].copy_from_slice(&0x40u64.to_le_bytes());
        execute(Flat(m), 0)
    }

    #[test]
    fn a_wait_on_a_version_no_invoke_made_is_an_error() {
        // Version 1 of granule 0x40, with no invoke.
        let (o, _) = exec(
            &[
                lui(A0, 1 << (KEY_BITS - 12)),
                addi(A0, A0, 0x40),
                nu(1, 0, A0),
            ],
            1,
        );
        match o {
            Outcome::Error(e) => assert!(e.contains("no invoke made more than 0"), "{e}"),
            o => panic!("{o:?}"),
        }
    }

    #[test]
    fn an_untracked_invoke_returns_0_and_a_versioned_one_its_version() {
        let code = [lui(A0, 1), nu(0, A1, A0), addi(GP, 0, 1), EXIT];
        for (flags, want) in [(3u16, 0u64), (1, 1)] {
            let (o, hart) = exec(&code, flags);
            assert!(matches!(o, Outcome::Pass { .. }), "{o:?}");
            assert_eq!(hart.get(A1 as u8).0, want, "flags {flags}");
        }
    }

    #[test]
    fn a_fail_at_ebreak_gives_the_case_from_gp() {
        let (o, _) = exec(&[addi(GP, 0, 7), EBREAK], 0);
        assert!(matches!(o, Outcome::Fail { case: 3, .. }), "{o:?}");
    }

    #[test]
    fn nu_exit_with_gp_not_1_is_an_error() {
        let (o, _) = exec(&[addi(GP, 0, 5), EXIT], 0);
        match o {
            Outcome::Error(e) => assert!(e.contains("with gp = 5"), "{e}"),
            o => panic!("{o:?}"),
        }
    }

    #[test]
    fn an_r2sim_error_at_the_fail_table_gives_the_case() {
        let text = "FATAL: Fatal { error: Invalid(\"root: pc 0x10c: ecall or ebreak\") }";
        assert!(matches!(
            sim_outcome(false, text, Some(0x100)),
            Outcome::Fail { case: 3, .. }
        ));
        assert!(matches!(
            sim_outcome(false, text, Some(0x200)),
            Outcome::Error(_)
        ));
        let ok = sim_outcome(true, "end 1234 cycles (0 steps)", None);
        assert_eq!(
            ok,
            Outcome::Pass {
                detail: "1234 cycles".into()
            }
        );
    }
}
