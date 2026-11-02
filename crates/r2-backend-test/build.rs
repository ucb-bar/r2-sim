//! Compiles layout.c, which reports the layout of the C types of
//! include/r2_backend.h.

fn main() {
    println!("cargo:rerun-if-changed=layout.c");
    println!("cargo:rerun-if-changed=../../include/r2_backend.h");
    cc::Build::new()
        .file("layout.c")
        .include("../../include")
        .compile("r2_backend_layout");
}
