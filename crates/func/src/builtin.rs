//! The `builtin` backend.

use std::collections::HashMap;

use nu_model::Range;
use sim_core::store::check_in_dst;
use sim_core::{
    split_row, Args, Descriptor, Dtype, FuncError, FuncOp, FunctionalBackend, OpKind, PeId, Store,
};

use crate::dtype::{bf16_to_f32, f32_to_bf16};

fn err<T>(msg: String) -> Result<T, FuncError> {
    Err(FuncError(msg))
}

/// Executes tasks on the store: DMA copies and GEMM. A Muon task of the
/// opaque kernel writes nothing; the other kernels are not built in.
///
/// A GEMM is output stationary, as in the Python plane: the partial C of a
/// chain stays in the accumulator of its Atlas (fp32, keyed by the PE and
/// the C base) while the tasks keep it, and is written once.
#[derive(Default)]
pub struct Builtin {
    /// The partials held in the Atlas accumulators, by (PE, C base):
    /// `m x n` row-major fp32.
    acc: HashMap<(Option<PeId>, u64), Vec<f32>>,
}

impl Builtin {
    /// Builds a backend with no partial sums.
    pub fn new() -> Self {
        Self::default()
    }

    /// One task; `tile`: the output tile of a matmul's Atlas (the halves of
    /// split operands).
    fn task(
        &mut self,
        desc: &Descriptor,
        tile: [u64; 2],
        mem: &mut Store,
    ) -> Result<(), FuncError> {
        match &desc.args {
            // An IMEM_COPY moves bytes of the control plane: no data here.
            Args::Join | Args::Spawn { .. } | Args::ImemCopy { .. } => Ok(()),
            Args::Vector { kernel, .. } => opaque(*kernel),
            &Args::Dma {
                src,
                dst,
                bytes,
                rows,
                ..
            } => mem.dma(&desc.held(), src.0, dst.0, bytes, rows.as_ref()),
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
                keep,
                a2,
                b2,
                transpose_b,
            } => {
                for dt in [in_dtype, b_dtype, out_dtype] {
                    if !matches!(dt, Dtype::Bf16 | Dtype::F32) {
                        return err(format!("builtin matmul on {dt:?} operands"));
                    }
                }
                let c = desc.dst.base;
                let es = out_dtype.bytes();
                // Each row of C, against the masked dst.
                for r in 0..m {
                    let row = Range::from_len(c + r * ldc * es, n * es);
                    check_in_dst(&desc.held(), mem.granule(), &row)?;
                }
                let es_a = in_dtype.bytes();
                let av = match a2 {
                    None => read_matrix(mem, a.0, m, k, lda, in_dtype)?,
                    Some(a2) => {
                        let mut v = Vec::with_capacity((m * k) as usize);
                        for r in 0..m {
                            let (hi, i) = split_row(r, tile[0]);
                            let base = if hi { a2.0 } else { a.0 };
                            v.extend(read_matrix(
                                mem,
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
                    // B stored as n x k: its transpose.
                    None if transpose_b => {
                        let t = read_matrix(mem, b.0, n, k, ldb, b_dtype)?;
                        let (n, k) = (n as usize, k as usize);
                        let mut v = vec![0.0; k * n];
                        for j in 0..n {
                            for p in 0..k {
                                v[p * n + j] = t[j * k + p];
                            }
                        }
                        v
                    }
                    None => read_matrix(mem, b.0, k, n, ldb, b_dtype)?,
                    Some(b2) => {
                        let es_b = b_dtype.bytes();
                        let mut v = vec![0.0; (k * n) as usize];
                        for c in 0..n {
                            let (hi, i) = split_row(c, tile[1]);
                            let base = if hi { b2.0 } else { b.0 } + i * es_b;
                            let col = read_matrix(mem, base, k, 1, ldb, b_dtype)?;
                            for (p, x) in col.into_iter().enumerate() {
                                v[p * n as usize + c as usize] = x;
                            }
                        }
                        v
                    }
                };
                // A task that accumulates starts from the held partial of
                // its C, else from C in memory; any task drops the partial.
                let partial = self.acc.remove(&(desc.pe, c));
                let start = match (accumulate, partial) {
                    (false, _) => None,
                    (true, Some(p)) if p.len() as u64 == m * n => Some(p),
                    (true, Some(p)) => {
                        return err(format!(
                            "matmul on C {c:#x} of {m} x {n}: the held partial has {} elements",
                            p.len()
                        ))
                    }
                    (true, None) => Some(read_matrix(mem, c, m, n, ldc, out_dtype)?),
                };
                let mut cv = gemm(&av, &bv, m as usize, n as usize, k as usize);
                if let Some(start) = start {
                    for (x, s) in cv.iter_mut().zip(start) {
                        *x += s;
                    }
                }
                if keep {
                    self.acc.insert((desc.pe, c), cv);
                    Ok(())
                } else {
                    write_matrix(mem, c, m, n, ldc, out_dtype, &cv)
                }
            }
        }
    }
}

/// A Muon task: the opaque kernel 0 writes nothing; any other kernel is
/// an error.
fn opaque(kernel: u32) -> Result<(), FuncError> {
    match kernel {
        0 => Ok(()),
        k => err(format!(
            "Muon kernel {k}: the builtin backend has no Muon kernels"
        )),
    }
}

/// Reads a row-major `rows x cols` matrix with leading dimension `ld`.
fn read_matrix(
    mem: &Store,
    base: u64,
    rows: u64,
    cols: u64,
    ld: u64,
    dt: Dtype,
) -> Result<Vec<f32>, FuncError> {
    let es = dt.bytes();
    let mut out = Vec::with_capacity((rows * cols) as usize);
    for r in 0..rows {
        out.extend(decode(mem.bytes(base + r * ld * es, cols * es)?, dt));
    }
    Ok(out)
}

/// Writes a row-major `rows x cols` matrix with leading dimension `ld`.
fn write_matrix(
    mem: &mut Store,
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
        mem.write(base + r * ld * es, &encode(row, dt))?;
    }
    Ok(())
}

/// The product A B (`m x k` times `k x n`, row-major) in f32: each
/// `c[i][j]` is the sum over p in 0..k, in order, of `a[i][p] * b[p][j]`.
/// The loop order i-p-j keeps the inner loop contiguous over a row of B
/// and of C, so that the compiler vectorizes it; the sum order per element
/// is that of the plain triple loop, so the result is deterministic.
fn gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    if n == 0 || k == 0 {
        return c;
    }
    for (row, ar) in c.chunks_exact_mut(n).zip(a.chunks_exact(k)) {
        for (&x, br) in ar.iter().zip(b.chunks_exact(n)) {
            for (o, &y) in row.iter_mut().zip(br) {
                *o += x * y;
            }
        }
    }
    c
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

impl FunctionalBackend for Builtin {
    fn name(&self) -> &'static str {
        "builtin"
    }

    fn apply(&mut self, op: &FuncOp, mem: &mut Store) -> Result<(), FuncError> {
        match &op.kind {
            OpKind::Task(desc) => self.task(desc, op.tile, mem),
            OpKind::Store { addr, bytes } => mem.write(addr.0, bytes),
        }
    }
}
