//! Records the Python that the plug-in runs: the shared libpython (named
//! by the version) and the prefix of the environment of the `python3` on PATH at build time (the
//! pixi environment), and the source tree's `python` directory.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=PATH");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    let out = Command::new("python3")
        .args([
            "-c",
            "import sys, sysconfig; print(sysconfig.get_config_var('LIBDIR')); \
             print('libpython' + sysconfig.get_config_var('VERSION') + '.so'); \
             print(sys.base_prefix)",
        ])
        .output()
        .expect("python3, to find libpython (run in the pixi environment)");
    assert!(
        out.status.success(),
        "python3: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).expect("UTF-8");
    let mut lines = text.lines();
    let (libdir, library, prefix) = (
        lines.next().expect("LIBDIR"),
        lines.next().expect("LDLIBRARY"),
        lines.next().expect("prefix"),
    );
    // The shared library (conda's Python links its own interpreter
    // statically, and ships the shared one next to it).
    let lib = format!("{libdir}/{library}");
    assert!(
        std::path::Path::new(&lib).exists(),
        "no shared libpython: {lib}"
    );
    println!("cargo:rustc-env=R2_PYTHON_LIB={lib}");
    println!("cargo:rustc-env=R2_PYTHON_HOME={prefix}");
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let dir = std::path::Path::new(&manifest).join("../../python");
    let dir = dir
        .canonicalize()
        .expect("the python directory of the source tree");
    println!("cargo:rustc-env=R2_PYTHON_PATH={}", dir.display());
}
