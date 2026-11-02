//! A plug-in of another ABI version is an error. (Its own test binary: it
//! changes the version that the test plug-in returns.)

use func::Plugin;

#[test]
fn another_abi_version_is_an_error() {
    let exe = std::env::current_exe().unwrap();
    let deps = exe.parent().unwrap();
    let lib = [deps, deps.parent().unwrap()]
        .iter()
        .map(|d| d.join("libr2_backend_test.so"))
        .find(|p| p.exists())
        .expect("libr2_backend_test.so");
    let path = std::ffi::CString::new(lib.to_str().unwrap()).unwrap();
    // SAFETY: the same library that the loader opens; its handle stays.
    unsafe {
        let h = libc::dlopen(path.as_ptr(), libc::RTLD_NOW);
        assert!(!h.is_null());
        let f = libc::dlsym(h, c"r2_test_set_abi".as_ptr());
        let set: extern "C" fn(u32) = std::mem::transmute(f);
        set(2);
    }
    let e = Plugin::open("test", lib.to_str().unwrap(), "{}", vec![])
        .err()
        .unwrap();
    assert!(e.ends_with("ABI version 2, r2-sim has 1"), "{e}");
}
