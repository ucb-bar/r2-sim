//! The `Numeric` backend.

use nu_model::{Addr, Range};
use sim_core::{
    split_row, Args, Descriptor, Dtype, FuncError, FuncOp, FunctionalBackend, MemBlock, OpKind,
};

use crate::dtype::{bf16_to_f32, f32_to_bf16};
use crate::memory::{check_in_dst, err, Memory};

/// Executes ops on sparse byte memory: DMA copies, GEMM, stores. Muon
/// vector tasks are opaque in phase 1 and write nothing.
#[derive(Default)]
pub struct Numeric {
    mem: Memory,
}

impl Numeric {
    /// Builds a backend with empty memory.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reads a row-major `rows x cols` matrix with leading dimension `ld`.
    fn read_matrix(
        &mut self,
        base: u64,
        rows: u64,
        cols: u64,
        ld: u64,
        dt: Dtype,
    ) -> Result<Vec<f32>, FuncError> {
        let es = dt.bytes();
        let mut out = Vec::with_capacity((rows * cols) as usize);
        for r in 0..rows {
            let row = self.mem.read(base + r * ld * es, cols * es)?;
            out.extend(decode(&row, dt));
        }
        Ok(out)
    }

    fn write_matrix(
        &mut self,
        base: u64,
        rows: u64,
        cols: u64,
        ld: u64,
        dt: Dtype,
        v: &[f32],
    ) -> Result<(), FuncError> {
        let es = dt.bytes();
        for r in 0..rows {
            let row = &v[(r * cols) as usize..((r + 1) * cols) as usize];
            self.mem.write(base + r * ld * es, &encode(row, dt))?;
        }
        Ok(())
    }

    fn task(&mut self, desc: &Descriptor) -> Result<(), FuncError> {
        match &desc.args {
            Args::Join | Args::Spawn { .. } | Args::Vector { .. } => Ok(()),
            &Args::Dma {
                src, dst, bytes, ..
            } => self.mem.dma(&desc.held(), src.0, dst.0, bytes),
            &Args::Matmul {
                a,
                b,
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
                a2,
                b2,
                tile,
                ..
            } => {
                for dt in [in_dtype, b_dtype, out_dtype] {
                    if !matches!(dt, Dtype::Bf16 | Dtype::F32) {
                        return err(format!("numeric matmul on {dt:?} operands"));
                    }
                }
                let c = desc.dst.base;
                let es = out_dtype.bytes();
                // Each row of C, against the masked dst.
                for r in 0..m {
                    let row = Range::from_len(c + r * ldc * es, n * es);
                    check_in_dst(&desc.held(), self.mem.granule(), &row)?;
                }
                let es_a = in_dtype.bytes();
                let av = match a2 {
                    None => self.read_matrix(a.0, m, k, lda, in_dtype)?,
                    Some(a2) => {
                        let mut v = Vec::with_capacity((m * k) as usize);
                        for r in 0..m {
                            let (hi, i) = split_row(r, tile[0]);
                            let base = if hi { a2.0 } else { a.0 };
                            v.extend(self.read_matrix(
                                base + i * lda * es_a,
                                1,
                                k,
                                lda,
                                in_dtype,
                            )?);
                        }
                        v
                    }
                };
                let bv = match b2 {
                    None => self.read_matrix(b.0, k, n, ldb, b_dtype)?,
                    Some(b2) => {
                        let es_b = b_dtype.bytes();
                        let mut v = vec![0.0; (k * n) as usize];
                        for c in 0..n {
                            let (hi, i) = split_row(c, tile[1]);
                            let base = if hi { b2.0 } else { b.0 } + i * es_b;
                            let col = self.read_matrix(base, k, 1, ldb, b_dtype)?;
                            for (p, x) in col.into_iter().enumerate() {
                                v[p * n as usize + c as usize] = x;
                            }
                        }
                        v
                    }
                };
                let mut cv = if accumulate {
                    self.read_matrix(c, m, n, ldc, out_dtype)?
                } else {
                    vec![0.0; (m * n) as usize]
                };
                gemm(&av, &bv, &mut cv, m as usize, n as usize, k as usize);
                self.write_matrix(c, m, n, ldc, out_dtype, &cv)
            }
        }
    }
}

/// `c[i][j] += sum over p in 0..k, in order, of a[i][p] * b[p][j]`, in f32.
fn gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = c[i * n + j];
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

fn decode(bytes: &[u8], dt: Dtype) -> impl Iterator<Item = f32> + '_ {
    let es = dt.bytes() as usize;
    bytes.chunks_exact(es).map(move |c| match dt {
        Dtype::Bf16 => bf16_to_f32(u16::from_le_bytes([c[0], c[1]])),
        Dtype::F32 => f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
        Dtype::Fp8 | Dtype::Mxfp4 => unreachable!("rejected before decode"),
    })
}

fn encode(v: &[f32], dt: Dtype) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * dt.bytes() as usize);
    for &x in v {
        match dt {
            Dtype::Bf16 => out.extend_from_slice(&f32_to_bf16(x).to_le_bytes()),
            Dtype::F32 => out.extend_from_slice(&x.to_le_bytes()),
            Dtype::Fp8 | Dtype::Mxfp4 => unreachable!("rejected before encode"),
        }
    }
    out
}

impl FunctionalBackend for Numeric {
    fn configure(&mut self, memories: &[MemBlock], sync_only: &[Range], granule: u64) {
        self.mem.configure(memories, sync_only, granule);
    }

    fn has_data(&self) -> bool {
        true
    }

    fn apply(&mut self, op: &FuncOp) -> Result<(), FuncError> {
        match &op.kind {
            OpKind::Task(desc) => self.task(desc),
            OpKind::Store { addr, bytes } => self.mem.write(addr.0, bytes),
        }
    }

    fn read(&mut self, addr: Addr, len: u64) -> Result<Vec<u8>, FuncError> {
        self.mem.read(addr.0, len)
    }

    fn write(&mut self, addr: Addr, bytes: &[u8]) -> Result<(), FuncError> {
        self.mem.write(addr.0, bytes)
    }
}
