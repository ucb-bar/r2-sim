//! The `Process` backend: a functional plane in a child process.
//!
//! The backend forwards everything the functional worker does to a child
//! command over its stdin and stdout, and the child keeps the memory and
//! executes the ops (for example `python -m r2sim.func`, the Python plane;
//! later a PyTorch, cyclotron or RTL plane).
//!
//! Protocol, version 1. Every message is one frame:
//!
//! ```text
//! u32 n (little-endian): the bytes after this field
//! u8  type
//! u32 j: the bytes of the JSON part
//! j bytes: JSON (UTF-8)
//! n - 5 - j bytes: raw bytes (memory data), never text-encoded
//! ```
//!
//! The parent sends HELLO `{"protocol": 1}`, then CONFIGURE (memory blocks,
//! sync-only ranges, granule), then in the worker's order: TASK (an op with
//! its descriptor), STORE (an op with its bytes), WRITE (a host write), READ
//! (a load or the final readback), and FINISH. The child answers each frame
//! with OK, DATA (the bytes of a READ) or ERROR `{"seq": .., "msg": ..}`; an
//! error stops the run as a worker error does. With `"op_time": true` in
//! CONFIGURE, the OK of a TASK or STORE carries `{"ns": ..}`, the child's
//! time on the op (for the thread timeline); a child may ignore it.
//!
//! Lifecycle: the child dies with r2-sim. It exits at end of file on stdin;
//! it starts with the parent-death signal set to SIGKILL (`prctl` in
//! `pre_exec`; the signal follows the thread that spawned it, so spawn it on
//! the main thread); and [`Drop`] kills and reaps it on a normal exit, an
//! error or a panic. Its stderr goes to r2-sim's stderr with a prefix.

use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::os::unix::process::CommandExt;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

use nu_model::Range;
use serde_json::{json, Value};
use sim_core::{Args, Descriptor, Dtype, FuncError, FuncOp, FunctionalBackend, MemBlock, OpKind};

/// The protocol version of the handshake.
pub const PROTOCOL: u64 = 1;

/// Frame types, parent to child.
pub mod frame {
    pub const HELLO: u8 = 1;
    pub const CONFIGURE: u8 = 2;
    pub const TASK: u8 = 3;
    pub const STORE: u8 = 4;
    pub const WRITE: u8 = 5;
    pub const READ: u8 = 6;
    pub const FINISH: u8 = 7;
    /// Child to parent.
    pub const OK: u8 = 64;
    pub const DATA: u8 = 65;
    pub const ERROR: u8 = 66;
}

/// A functional backend in a child process.
pub struct Process {
    child: Child,
    to: Option<BufWriter<ChildStdin>>,
    from: BufReader<ChildStdout>,
    /// An error from `configure`, returned by the next op.
    failed: Option<FuncError>,
    /// Ask the child for the time of each op (`op_time` in CONFIGURE).
    op_time: bool,
    /// The child's time of the last op (`ns` in its OK), in ns.
    last_ns: Option<u64>,
}

impl Process {
    /// Starts `command` (program and arguments) and checks the protocol
    /// version. `prefix` marks the child's stderr lines.
    pub fn spawn(command: &[String], prefix: &str) -> Result<Self, FuncError> {
        Self::spawn_with_env(command, &[], prefix)
    }

    /// Like [`Process::spawn`], with extra environment variables.
    pub fn spawn_with_env(
        command: &[String],
        env: &[(String, String)],
        prefix: &str,
    ) -> Result<Self, FuncError> {
        let (prog, args) = command
            .split_first()
            .ok_or_else(|| FuncError("an empty functional command".into()))?;
        let mut cmd = Command::new(prog);
        cmd.args(args)
            .envs(env.iter().map(|(k, v)| (k, v)))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let parent = std::process::id() as libc::pid_t;
        // SAFETY: the closure runs in the child between fork and exec and
        // calls only async-signal-safe functions (prctl, getppid, _exit).
        unsafe {
            cmd.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                // The parent died before the prctl: do not outlive it.
                if libc::getppid() != parent {
                    libc::_exit(1);
                }
                Ok(())
            });
        }
        let mut child = cmd.spawn().map_err(|e| FuncError(format!("{prog}: {e}")))?;
        let to = BufWriter::new(child.stdin.take().expect("piped stdin"));
        let from = BufReader::new(child.stdout.take().expect("piped stdout"));
        let err = child.stderr.take().expect("piped stderr");
        let prefix = prefix.to_string();
        std::thread::spawn(move || {
            for line in BufReader::new(err).lines().map_while(Result::ok) {
                eprintln!("{prefix}{line}");
            }
        });
        let mut p = Self {
            child,
            to: Some(to),
            from,
            failed: None,
            op_time: false,
            last_ns: None,
        };
        let (_, hello, _) = p.call(frame::HELLO, &json!({ "protocol": PROTOCOL }), &[], 0)?;
        if hello.get("protocol").and_then(Value::as_u64) != Some(PROTOCOL) {
            return Err(FuncError(format!(
                "{prog}: protocol {hello} (r2-sim speaks {PROTOCOL})"
            )));
        }
        Ok(p)
    }

    /// The child's process id.
    pub fn id(&self) -> u32 {
        self.child.id()
    }

    /// Asks the child (at CONFIGURE) to report the time it spends on each
    /// TASK and STORE: the field `ns` of its OK, for the thread timeline.
    pub fn set_op_time(&mut self, on: bool) {
        self.op_time = on;
    }

    /// Sends one frame and reads the answer: (type, JSON, bytes). An ERROR
    /// answer is an `Err` carrying the child's message; `seq` names the op.
    fn call(
        &mut self,
        kind: u8,
        head: &Value,
        blob: &[u8],
        seq: u64,
    ) -> Result<(u8, Value, Vec<u8>), FuncError> {
        let lost = |e: std::io::Error| match e.kind() {
            std::io::ErrorKind::UnexpectedEof | std::io::ErrorKind::BrokenPipe => {
                FuncError("functional process: the child closed its pipes (it exited)".into())
            }
            _ => FuncError(format!("functional process: {e}")),
        };
        let to = self
            .to
            .as_mut()
            .ok_or_else(|| FuncError("functional process: finished".into()))?;
        write_frame(to, kind, head, blob).map_err(lost)?;
        to.flush().map_err(lost)?;
        let (kind, head, blob) = read_frame(&mut self.from).map_err(lost)?;
        match kind {
            frame::OK | frame::DATA => Ok((kind, head, blob)),
            frame::ERROR => {
                let msg = head.get("msg").and_then(Value::as_str).unwrap_or("error");
                let at = head.get("seq").and_then(Value::as_u64).unwrap_or(seq);
                Err(FuncError(format!("op {at}: {msg}")))
            }
            k => Err(FuncError(format!("functional process: answer type {k}"))),
        }
    }
}

/// Writes one frame.
pub fn write_frame(w: &mut impl Write, kind: u8, head: &Value, blob: &[u8]) -> std::io::Result<()> {
    let j = serde_json::to_vec(head).expect("JSON");
    let n = 1 + 4 + j.len() + blob.len();
    w.write_all(&(n as u32).to_le_bytes())?;
    w.write_all(&[kind])?;
    w.write_all(&(j.len() as u32).to_le_bytes())?;
    w.write_all(&j)?;
    w.write_all(blob)
}

/// Reads one frame: (type, JSON, bytes).
pub fn read_frame(r: &mut impl Read) -> std::io::Result<(u8, Value, Vec<u8>)> {
    let mut b4 = [0u8; 4];
    r.read_exact(&mut b4)?;
    let n = u32::from_le_bytes(b4) as usize;
    let mut body = vec![0u8; n];
    r.read_exact(&mut body)?;
    let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
    if n < 5 {
        return Err(bad("short frame"));
    }
    let j = u32::from_le_bytes(body[1..5].try_into().expect("4 bytes")) as usize;
    if 5 + j > n {
        return Err(bad("frame JSON past its end"));
    }
    let head = if j == 0 {
        Value::Null
    } else {
        serde_json::from_slice(&body[5..5 + j]).map_err(|e| bad(&e.to_string()))?
    };
    Ok((body[0], head, body[5 + j..].to_vec()))
}

fn dtype(d: Dtype) -> &'static str {
    match d {
        Dtype::Bf16 => "bf16",
        Dtype::F32 => "f32",
        Dtype::Fp8 => "fp8",
        Dtype::Mxfp4 => "mxfp4",
    }
}

fn range(r: &Range) -> Value {
    json!([r.base, r.bound])
}

/// A task descriptor as JSON: `kind` and the opcode's arguments, the PE,
/// the destination range and its mask.
pub fn descriptor(d: &Descriptor) -> Value {
    let mut v = json!({
        "pe": d.pe.map(|p| p.0),
        "dst": range(&d.dst),
        "mask": d.dst_mask,
        "versioned": d.versioned,
        "untracked": d.untracked,
        "label": d.label,
    });
    let args = match &d.args {
        Args::Join => json!({ "kind": "join" }),
        Args::Spawn { child } => json!({ "kind": "spawn", "child": child.0 }),
        Args::ImemCopy {
            child,
            src,
            offset,
            bytes,
        } => json!({
            "kind": "imem_copy", "child": child.0, "src": src, "offset": offset, "bytes": bytes,
        }),
        Args::Dma {
            src,
            dst,
            bytes,
            tile_bytes,
        } => json!({
            "kind": "dma", "src": src.0, "to": dst.0, "bytes": bytes, "tile_bytes": tile_bytes,
        }),
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
        } => json!({
            "kind": "matmul", "a": a.0, "b": b.0, "a2": a2.map(|x| x.0), "b2": b2.map(|x| x.0),
            "lda": lda, "ldb": ldb, "ldc": ldc, "m": m, "n": n, "k": k,
            "a_dtype": dtype(*in_dtype), "b_dtype": dtype(*b_dtype), "out_dtype": dtype(*out_dtype),
            "accumulate": accumulate, "keep": keep, "tile": tile,
        }),
        Args::Vector {
            ops,
            bytes_in,
            bytes_out,
            src,
        } => json!({
            "kind": "vector", "ops": ops, "bytes_in": bytes_in, "bytes_out": bytes_out,
            "src": src.map(|x| x.0),
        }),
    };
    for (k, x) in args.as_object().expect("object") {
        v[k] = x.clone();
    }
    v
}

impl FunctionalBackend for Process {
    fn configure(&mut self, memories: &[MemBlock], sync_only: &[Range], granule: u64) {
        let blocks: Vec<Value> = memories
            .iter()
            .map(|m| {
                json!({
                    "name": m.name, "range": range(&m.range), "kind": m.kind,
                    "numeric": m.numeric,
                })
            })
            .collect();
        let sync: Vec<Value> = sync_only.iter().map(range).collect();
        let mut head = json!({ "memories": blocks, "sync_only": sync, "granule": granule });
        if self.op_time {
            head["op_time"] = json!(true);
        }
        if let Err(e) = self.call(frame::CONFIGURE, &head, &[], 0) {
            self.failed = Some(e);
        }
    }

    fn has_data(&self) -> bool {
        true
    }

    fn apply(&mut self, op: &FuncOp) -> Result<(), FuncError> {
        if let Some(e) = self.failed.take() {
            return Err(e);
        }
        let base = json!({ "seq": op.seq, "time": op.time, "nu": op.nu.0, "gid": op.gid });
        let (_, answer, _) = match &op.kind {
            OpKind::Task(d) => {
                let mut head = base;
                head["desc"] = descriptor(d);
                self.call(frame::TASK, &head, &[], op.seq)?
            }
            OpKind::Store { addr, bytes } => {
                let mut head = base;
                head["addr"] = json!(addr.0);
                self.call(frame::STORE, &head, bytes, op.seq)?
            }
        };
        self.last_ns = answer.get("ns").and_then(Value::as_u64);
        Ok(())
    }

    fn child_ns(&self) -> Option<u64> {
        self.last_ns
    }

    fn read(&mut self, addr: nu_model::Addr, len: u64) -> Result<Vec<u8>, FuncError> {
        let head = json!({ "addr": addr.0, "len": len });
        let (_, _, blob) = self.call(frame::READ, &head, &[], 0)?;
        if blob.len() as u64 != len {
            return Err(FuncError(format!(
                "functional process: {} bytes for a read of {len}",
                blob.len()
            )));
        }
        Ok(blob)
    }

    fn write(&mut self, addr: nu_model::Addr, bytes: &[u8]) -> Result<(), FuncError> {
        let head = json!({ "addr": addr.0 });
        self.call(frame::WRITE, &head, bytes, 0).map(|_| ())
    }
}

impl Drop for Process {
    /// Asks the child to finish, closes its stdin (it exits at end of
    /// file), and kills it if it has not exited after a short wait; always
    /// reaps it.
    fn drop(&mut self) {
        if let Some(mut to) = self.to.take() {
            let _ = write_frame(&mut to, frame::FINISH, &Value::Null, &[]);
            let _ = to.flush();
        }
        let deadline = Instant::now() + Duration::from_millis(500);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = self.child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
