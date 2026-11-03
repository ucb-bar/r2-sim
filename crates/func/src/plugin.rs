//! A backend in a plug-in library (include/r2_backend.h): the loader.
//!
//! [`Plugin::open`] opens the library with `dlopen`, finds
//! `r2_backend_open` and checks the ABI version. `configure` passes the
//! block table of the store, the sync-only space, the PE names by index
//! and the backend's entry of the configuration as JSON, with the keys
//! `kernels` (the Muon kernel registry: id, name, bytes, flags, and the
//! fields with name, type and offset) and `tiles` (per PE index, the
//! output tile `[rows, columns]` of an Atlas, which gives the halves of a
//! matmul's split operands; [128, 128] for another PE). `apply` passes
//! each task with its descriptor bytes (a task of an RV program); a task
//! without bytes (a task of a Rust program) is an error. `close` runs when
//! the backend is dropped, at the end of the run. The library stays
//! loaded until the process ends: a plug-in that runs an interpreter
//! cannot be unloaded.
//!
//! Every error names the backend and the library.

use std::ffi::{c_char, CStr, CString};
use std::path::{Path, PathBuf};

use r2_backend::{Backend, ByteRange, Memory, Task, ABI, NO_PE, OPEN};
use sim_core::{FuncError, FuncOp, FunctionalBackend, OpKind, Store};

const ERR_LEN: usize = 1024;

/// A functional backend in a plug-in library.
pub struct Plugin {
    name: &'static str,
    lib: String,
    backend: *const Backend,
    config: CString,
    pes: Vec<CString>,
    /// The block table and the sync-only space that `configure` passed;
    /// they live as long as the backend.
    blocks: Vec<r2_backend::Block>,
    names: Vec<CString>,
    sync_only: Vec<ByteRange>,
    memory: Option<Box<Memory>>,
}

// SAFETY: the plug-in's calls come from one thread at a time (the
// configure of the run, then the worker thread).
unsafe impl Send for Plugin {}

/// The library `lib` of a configuration: a file name is looked up next to
/// the running binary, a path is used as it is.
pub fn lib_path(lib: &str) -> PathBuf {
    let p = Path::new(lib);
    if p.components().count() > 1 {
        return p.to_path_buf();
    }
    std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|d| d.join(lib)))
        .unwrap_or_else(|| p.to_path_buf())
}

/// The last `dlerror` message.
fn dl_error() -> String {
    // SAFETY: dlerror returns null or a NUL-terminated string.
    let e = unsafe { libc::dlerror() };
    if e.is_null() {
        "unknown error".into()
    } else {
        unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned()
    }
}

impl Plugin {
    /// Opens the backend `name` in the library `lib` (a file name next to
    /// the binary, or a path). `config`: the backend's entry as JSON;
    /// `pes`: the PE names by index; `tiles`: the Atlas tile of each PE
    /// (the key `tiles`).
    pub fn open(
        name: &str,
        lib: &str,
        config: &str,
        pes: Vec<String>,
        tiles: &[[u64; 2]],
    ) -> Result<Self, String> {
        let at = |msg: String| format!("backend {name} ({lib}): {msg}");
        let path = lib_path(lib);
        let cpath =
            CString::new(path.as_os_str().as_encoded_bytes()).map_err(|e| at(e.to_string()))?;
        // SAFETY: dlopen of a path; the library's initializers run.
        let h = unsafe { libc::dlopen(cpath.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if h.is_null() {
            return Err(at(dl_error()));
        }
        // SAFETY: a lookup in the handle.
        let sym = unsafe { libc::dlsym(h, OPEN.as_ptr().cast()) };
        if sym.is_null() {
            return Err(at(format!(
                "no symbol r2_backend_open in {}",
                path.display()
            )));
        }
        // SAFETY: the symbol has the type of the interface.
        let open: r2_backend::OpenFn = unsafe { std::mem::transmute(sym) };
        let mut err = [0 as c_char; ERR_LEN];
        let backend = unsafe { open(err.as_mut_ptr(), ERR_LEN) };
        if backend.is_null() {
            return Err(at(format!("r2_backend_open: {}", message(&err))));
        }
        let abi = unsafe { (*backend).abi };
        if abi != ABI {
            return Err(at(format!("ABI version {abi}, r2-sim has {ABI}")));
        }
        Ok(Self {
            name: Box::leak(name.to_string().into_boxed_str()),
            lib: lib.to_string(),
            backend,
            config: CString::new(with_keys(config, tiles).map_err(at)?)
                .map_err(|e| at(e.to_string()))?,
            pes: pes
                .into_iter()
                .map(|p| CString::new(p).expect("a PE name without NUL"))
                .collect(),
            blocks: Vec::new(),
            names: Vec::new(),
            sync_only: Vec::new(),
            memory: None,
        })
    }

    fn error(&self, msg: String) -> FuncError {
        FuncError(format!("backend {} ({}): {msg}", self.name, self.lib))
    }
}

/// The backend's entry `config` (a JSON object) with the keys `kernels`
/// and `tiles`.
fn with_keys(config: &str, tiles: &[[u64; 2]]) -> Result<String, String> {
    let mut v: serde_json::Value =
        serde_json::from_str(config).map_err(|e| format!("the configuration: {e}"))?;
    let Some(entry) = v.as_object_mut() else {
        return Err("the configuration is not a JSON object".into());
    };
    for key in ["kernels", "tiles"] {
        if entry.contains_key(key) {
            return Err(format!(
                "the configuration has the key {key}, which r2-sim sets"
            ));
        }
    }
    let kernels = serde_json::to_value(sim_core::kernels::registry()).expect("JSON");
    entry.insert("kernels".into(), kernels);
    entry.insert("tiles".into(), serde_json::json!(tiles));
    Ok(v.to_string())
}

/// The NUL-terminated message in an error buffer.
fn message(err: &[c_char]) -> String {
    let bytes: Vec<u8> = err
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

impl FunctionalBackend for Plugin {
    fn name(&self) -> &'static str {
        self.name
    }

    fn configure(&mut self, store: &Store) -> Result<(), FuncError> {
        self.names = store
            .blocks()
            .iter()
            .map(|b| CString::new(b.name.as_str()).expect("a block name without NUL"))
            .collect();
        self.blocks = store
            .blocks()
            .iter()
            .zip(&self.names)
            .map(|(b, n)| r2_backend::Block {
                base: b.base,
                size: b.size,
                ptr: b.ptr,
                name: n.as_ptr(),
            })
            .collect();
        self.sync_only = store
            .sync_only()
            .iter()
            .map(|r| ByteRange {
                base: r.base,
                bound: r.bound,
            })
            .collect();
        let memory = Box::new(Memory {
            blocks: self.blocks.as_ptr(),
            count: self.blocks.len(),
            granule: store.granule(),
            sync_only: self.sync_only.as_ptr(),
            sync_only_count: self.sync_only.len(),
        });
        let pes: Vec<*const c_char> = self.pes.iter().map(|p| p.as_ptr()).collect();
        let mut err = [0 as c_char; ERR_LEN];
        // SAFETY: the tables live in `self` for the rest of the run.
        let r = unsafe {
            let b = &*self.backend;
            (b.configure)(
                b.this,
                &*memory,
                pes.as_ptr(),
                pes.len(),
                self.config.as_ptr(),
                err.as_mut_ptr(),
                ERR_LEN,
            )
        };
        self.memory = Some(memory);
        if r != 0 {
            return Err(self.error(format!("configure: {}", message(&err))));
        }
        Ok(())
    }

    fn apply(&mut self, op: &FuncOp, _store: &mut Store) -> Result<(), FuncError> {
        let OpKind::Task(desc) = &op.kind else {
            return Ok(());
        };
        if desc.bytes.is_empty() {
            return Err(self.error(format!(
                "task {} has no descriptor bytes: it is a task of a Rust program, which a plug-in cannot apply",
                op.gid.map_or("-".into(), |g| g.to_string())
            )));
        }
        // The argument block of a Muon task.
        let block: &[u8] = match &desc.args {
            sim_core::Args::Vector { block, .. } => block,
            _ => &[],
        };
        let task = Task {
            seq: op.seq,
            cycle: op.time,
            gid: op.gid.unwrap_or(u64::MAX),
            neutrino: op.nu.0,
            pe: desc.pe.map_or(NO_PE, |p| p.0),
            desc: desc.bytes.as_ptr(),
            desc_bytes: desc.bytes.len(),
            args: if block.is_empty() {
                std::ptr::null()
            } else {
                block.as_ptr()
            },
            args_bytes: block.len(),
        };
        let mut err = [0 as c_char; ERR_LEN];
        // SAFETY: the task's bytes live for the call.
        let r = unsafe {
            let b = &*self.backend;
            (b.apply)(b.this, &task, err.as_mut_ptr(), ERR_LEN)
        };
        if r != 0 {
            return Err(self.error(message(&err)));
        }
        Ok(())
    }
}

impl Drop for Plugin {
    fn drop(&mut self) {
        // SAFETY: the backend came from r2_backend_open; close frees it.
        unsafe {
            let b = &*self.backend;
            (b.close)(b.this);
        }
    }
}
