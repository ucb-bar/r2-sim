//! Sets `R2SIM_COMMIT` to the git HEAD at build time ("-dirty" with
//! uncommitted changes) for the raw trace manifest.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    let head = git(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty =
        git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
    let suffix = if dirty { "-dirty" } else { "" };
    println!("cargo:rustc-env=R2SIM_COMMIT={head}{suffix}");
    let root = git(&["rev-parse", "--show-toplevel"]).unwrap_or_default();
    if !root.is_empty() {
        println!("cargo:rerun-if-changed={root}/.git/HEAD");
        println!("cargo:rerun-if-changed={root}/.git/index");
    }
}
