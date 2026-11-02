//! libr2_python.so: the Python backend of the functional plane, a plug-in
//! (include/r2_backend.h) that runs CPython inside the r2-sim process.
//!
//! At `r2_backend_open` the plug-in loads the shared libpython of the
//! environment it was built against, with global symbol visibility (the
//! C extension modules of NumPy need the interpreter's symbols), and
//! starts the interpreter with that environment's prefix as its home (its
//! standard library) and the source tree's `python` directory on the
//! module path. It calls the C API through `dlsym`, so the build needs no
//! Python headers and no link to libpython.
//!
//! `configure` calls `r2sim.func.backend.create` with the block table, the
//! sync-only space, the granule, the PE names and the backend's entry of
//! the configuration (JSON); `apply` calls the backend's `apply` with the
//! task's fields and its descriptor bytes. Each call takes the interpreter
//! lock (`PyGILState_Ensure`), so the worker thread can call it. The
//! interpreter stays until the process ends.

use std::ffi::{c_char, c_int, c_long, c_void, CStr, CString};
use std::sync::OnceLock;

use r2_backend::{set_err, Backend, Memory, Task, ABI};

type Obj = *mut c_void;

/// The functions of the CPython C API that the plug-in calls.
struct Api {
    is_initialized: unsafe extern "C" fn() -> c_int,
    set_python_home: unsafe extern "C" fn(*const libc::wchar_t),
    decode_locale: unsafe extern "C" fn(*const c_char, *mut usize) -> *mut libc::wchar_t,
    initialize_ex: unsafe extern "C" fn(c_int),
    save_thread: unsafe extern "C" fn() -> *mut c_void,
    gil_ensure: unsafe extern "C" fn() -> c_int,
    gil_release: unsafe extern "C" fn(c_int),
    run_simple_string: unsafe extern "C" fn(*const c_char) -> c_int,
    import_module: unsafe extern "C" fn(*const c_char) -> Obj,
    get_attr_string: unsafe extern "C" fn(Obj, *const c_char) -> Obj,
    call_object: unsafe extern "C" fn(Obj, Obj) -> Obj,
    tuple_new: unsafe extern "C" fn(isize) -> Obj,
    tuple_set_item: unsafe extern "C" fn(Obj, isize, Obj) -> c_int,
    unicode_from_string: unsafe extern "C" fn(*const c_char) -> Obj,
    unicode_as_utf8: unsafe extern "C" fn(Obj) -> *const c_char,
    long_from_u64: unsafe extern "C" fn(u64) -> Obj,
    long_from_long: unsafe extern "C" fn(c_long) -> Obj,
    bytes_from: unsafe extern "C" fn(*const c_char, isize) -> Obj,
    err_fetch: unsafe extern "C" fn(*mut Obj, *mut Obj, *mut Obj),
    err_normalize: unsafe extern "C" fn(*mut Obj, *mut Obj, *mut Obj),
    object_str: unsafe extern "C" fn(Obj) -> Obj,
    dec_ref: unsafe extern "C" fn(Obj),
}

/// The libpython of the build environment.
const LIB: &str = env!("R2_PYTHON_LIB");
/// The prefix of the build environment: the interpreter's home.
const HOME: &str = env!("R2_PYTHON_HOME");
/// The source tree's `python` directory (the package `r2sim`).
const PATH: &str = env!("R2_PYTHON_PATH");

static API: OnceLock<Result<Api, String>> = OnceLock::new();

/// Loads libpython and its functions, and starts the interpreter once per
/// process.
fn api() -> Result<&'static Api, String> {
    API.get_or_init(|| unsafe { load() })
        .as_ref()
        .map_err(Clone::clone)
}

/// The function at `p`, of the type of the field it fills.
///
/// # Safety
/// `p` must be a function of type `F`.
unsafe fn fn_of<F: Copy>(p: *mut c_void) -> F {
    debug_assert_eq!(std::mem::size_of::<F>(), std::mem::size_of::<*mut c_void>());
    std::mem::transmute_copy(&p)
}

unsafe fn load() -> Result<Api, String> {
    let lib = CString::new(LIB).expect("a path without NUL");
    let h = libc::dlopen(lib.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL);
    if h.is_null() {
        let e = CStr::from_ptr(libc::dlerror()).to_string_lossy();
        return Err(format!("loading {LIB}: {e}"));
    }
    macro_rules! sym {
        ($name:literal) => {{
            let p = libc::dlsym(h, concat!($name, "\0").as_ptr().cast());
            if p.is_null() {
                return Err(format!("{LIB} has no {}", $name));
            }
            fn_of(p)
        }};
    }
    let api = Api {
        is_initialized: sym!("Py_IsInitialized"),
        set_python_home: sym!("Py_SetPythonHome"),
        decode_locale: sym!("Py_DecodeLocale"),
        initialize_ex: sym!("Py_InitializeEx"),
        save_thread: sym!("PyEval_SaveThread"),
        gil_ensure: sym!("PyGILState_Ensure"),
        gil_release: sym!("PyGILState_Release"),
        run_simple_string: sym!("PyRun_SimpleString"),
        import_module: sym!("PyImport_ImportModule"),
        get_attr_string: sym!("PyObject_GetAttrString"),
        call_object: sym!("PyObject_CallObject"),
        tuple_new: sym!("PyTuple_New"),
        tuple_set_item: sym!("PyTuple_SetItem"),
        unicode_from_string: sym!("PyUnicode_FromString"),
        unicode_as_utf8: sym!("PyUnicode_AsUTF8"),
        long_from_u64: sym!("PyLong_FromUnsignedLongLong"),
        long_from_long: sym!("PyLong_FromLong"),
        bytes_from: sym!("PyBytes_FromStringAndSize"),
        err_fetch: sym!("PyErr_Fetch"),
        err_normalize: sym!("PyErr_NormalizeException"),
        object_str: sym!("PyObject_Str"),
        dec_ref: sym!("Py_DecRef"),
    };
    if (api.is_initialized)() == 0 {
        // The home must outlive the interpreter: it is never freed.
        let home = CString::new(HOME).expect("a path without NUL");
        let w = (api.decode_locale)(home.as_ptr(), std::ptr::null_mut());
        (api.set_python_home)(w);
        (api.initialize_ex)(0);
        let path =
            format!("import sys\nif {PATH:?} not in sys.path: sys.path.insert(0, {PATH:?})\n");
        let path = CString::new(path).expect("a path without NUL");
        if (api.run_simple_string)(path.as_ptr()) != 0 {
            return Err("setting the module path of the interpreter".into());
        }
        // Release the interpreter lock: every call takes it.
        (api.save_thread)();
    }
    Ok(api)
}

/// The interpreter lock, for the life of the value.
struct Gil<'a>(&'a Api, c_int);

impl<'a> Gil<'a> {
    fn take(api: &'a Api) -> Self {
        Gil(api, unsafe { (api.gil_ensure)() })
    }
}

impl Drop for Gil<'_> {
    fn drop(&mut self) {
        unsafe { (self.0.gil_release)(self.1) };
    }
}

/// The pending Python exception as `Type: message`, and clears it.
unsafe fn exception(api: &Api) -> String {
    let (mut t, mut v, mut tb) = (
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    );
    (api.err_fetch)(&mut t, &mut v, &mut tb);
    if t.is_null() {
        return "an error without a Python exception".into();
    }
    (api.err_normalize)(&mut t, &mut v, &mut tb);
    let text = |o: Obj| -> String {
        if o.is_null() {
            return String::new();
        }
        let s = (api.object_str)(o);
        if s.is_null() {
            return String::new();
        }
        let c = (api.unicode_as_utf8)(s);
        let r = if c.is_null() {
            String::new()
        } else {
            CStr::from_ptr(c).to_string_lossy().into_owned()
        };
        (api.dec_ref)(s);
        r
    };
    let name = (api.get_attr_string)(t, c"__name__".as_ptr());
    let msg = format!("{}: {}", text(name), text(v));
    for o in [name, t, v, tb] {
        if !o.is_null() {
            (api.dec_ref)(o);
        }
    }
    msg
}

/// The state of one backend: the bound method `apply` of the Python
/// backend.
struct State {
    apply: Obj,
}

/// # Safety
/// `err` must point to `err_len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn r2_backend_open(err: *mut c_char, err_len: usize) -> *const Backend {
    if let Err(e) = api() {
        set_err(err, err_len, &e);
        return std::ptr::null();
    }
    let state = Box::new(State {
        apply: std::ptr::null_mut(),
    });
    Box::into_raw(Box::new(Backend {
        abi: ABI,
        this: Box::into_raw(state).cast(),
        configure,
        apply,
        close,
    }))
}

/// The configuration of `create` (JSON).
unsafe fn create_text(memory: &Memory, pes: &[String], config: &str) -> Result<String, String> {
    let blocks: Vec<serde_json::Value> = std::slice::from_raw_parts(memory.blocks, memory.count)
        .iter()
        .map(|b| {
            serde_json::json!({
                "base": b.base, "size": b.size, "ptr": b.ptr as usize,
                "name": CStr::from_ptr(b.name).to_string_lossy(),
            })
        })
        .collect();
    let sync: Vec<[u64; 2]> = if memory.sync_only.is_null() {
        Vec::new()
    } else {
        std::slice::from_raw_parts(memory.sync_only, memory.sync_only_count)
            .iter()
            .map(|r| [r.base, r.bound])
            .collect()
    };
    let config: serde_json::Value =
        serde_json::from_str(config).map_err(|e| format!("the configuration: {e}"))?;
    let all = serde_json::json!({
        "blocks": blocks, "sync_only": sync, "granule": memory.granule, "pes": pes, "config": config,
    });
    Ok(all.to_string())
}

unsafe extern "C" fn configure(
    this: *mut c_void,
    memory: *const Memory,
    pes: *const *const c_char,
    pe_count: usize,
    config: *const c_char,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    let s = &mut *this.cast::<State>();
    let api = api().expect("loaded at open");
    let names: Vec<String> = (0..pe_count)
        .map(|i| CStr::from_ptr(*pes.add(i)).to_string_lossy().into_owned())
        .collect();
    let text = match create_text(&*memory, &names, &CStr::from_ptr(config).to_string_lossy()) {
        Ok(t) => t,
        Err(e) => {
            set_err(err, err_len, &e);
            return 1;
        }
    };
    let _gil = Gil::take(api);
    let module = (api.import_module)(c"r2sim.func.backend".as_ptr());
    if module.is_null() {
        set_err(
            err,
            err_len,
            &format!("import r2sim.func.backend: {}", exception(api)),
        );
        return 1;
    }
    let create = (api.get_attr_string)(module, c"create".as_ptr());
    (api.dec_ref)(module);
    if create.is_null() {
        set_err(err, err_len, &exception(api));
        return 1;
    }
    let args = (api.tuple_new)(1);
    let ctext = CString::new(text).expect("JSON without NUL");
    (api.tuple_set_item)(args, 0, (api.unicode_from_string)(ctext.as_ptr()));
    let backend = (api.call_object)(create, args);
    (api.dec_ref)(args);
    (api.dec_ref)(create);
    if backend.is_null() {
        set_err(err, err_len, &exception(api));
        return 1;
    }
    s.apply = (api.get_attr_string)(backend, c"apply".as_ptr());
    (api.dec_ref)(backend);
    if s.apply.is_null() {
        set_err(err, err_len, &exception(api));
        return 1;
    }
    0
}

unsafe extern "C" fn apply(
    this: *mut c_void,
    task: *const Task,
    err: *mut c_char,
    err_len: usize,
) -> i32 {
    let s = &*this.cast::<State>();
    let api = api().expect("loaded at open");
    let t = &*task;
    let _gil = Gil::take(api);
    let args = (api.tuple_new)(6);
    let items = [
        (api.long_from_u64)(t.seq),
        (api.long_from_u64)(t.cycle),
        (api.long_from_u64)(t.gid),
        (api.long_from_long)(t.neutrino as c_long),
        (api.long_from_u64)(t.pe as u64),
        (api.bytes_from)(t.desc.cast(), t.desc_bytes as isize),
    ];
    for (i, o) in items.into_iter().enumerate() {
        (api.tuple_set_item)(args, i as isize, o);
    }
    let r = (api.call_object)(s.apply, args);
    (api.dec_ref)(args);
    if r.is_null() {
        set_err(err, err_len, &exception(api));
        return 1;
    }
    (api.dec_ref)(r);
    0
}

unsafe extern "C" fn close(this: *mut c_void) {
    let s = Box::from_raw(this.cast::<State>());
    if !s.apply.is_null() {
        let api = api().expect("loaded at open");
        let _gil = Gil::take(api);
        (api.dec_ref)(s.apply);
    }
}
