//! The plug-in interface and its loader, with the test plug-in of
//! crates/r2-backend-test: the layout of the Rust types against the C
//! header, the tasks that reach a plug-in, and the errors of the loader.

use std::path::PathBuf;

use func::Plugin;
use nu_model::{Addr, NeutrinoId, Range};
use r2_backend::{Backend, Block, ByteRange, Memory, Task};
use sim_core::{
    Args, Descriptor, FuncOp, FunctionalBackend, MemBlock, OpKind, OpcodeId, PeId, ResourceId,
    Store,
};

/// The test plug-in, which cargo builds next to this test.
fn test_lib() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let deps = exe.parent().unwrap();
    [deps, deps.parent().unwrap()]
        .iter()
        .map(|d| d.join("libr2_backend_test.so"))
        .find(|p| p.exists())
        .expect("libr2_backend_test.so")
}

#[test]
fn the_rust_types_have_the_layout_of_the_header() {
    use std::mem::{offset_of, size_of};
    let rust = [
        size_of::<Block>(),
        offset_of!(Block, base),
        offset_of!(Block, size),
        offset_of!(Block, ptr),
        offset_of!(Block, name),
        size_of::<ByteRange>(),
        offset_of!(ByteRange, base),
        offset_of!(ByteRange, bound),
        size_of::<Memory>(),
        offset_of!(Memory, blocks),
        offset_of!(Memory, count),
        offset_of!(Memory, granule),
        offset_of!(Memory, sync_only),
        offset_of!(Memory, sync_only_count),
        size_of::<Task>(),
        offset_of!(Task, seq),
        offset_of!(Task, cycle),
        offset_of!(Task, gid),
        offset_of!(Task, neutrino),
        offset_of!(Task, pe),
        offset_of!(Task, desc),
        offset_of!(Task, desc_bytes),
        offset_of!(Task, args),
        offset_of!(Task, args_bytes),
        size_of::<Backend>(),
        offset_of!(Backend, abi),
        offset_of!(Backend, this),
        offset_of!(Backend, configure),
        offset_of!(Backend, apply),
        offset_of!(Backend, close),
    ];
    assert_eq!(r2_backend_test::c_layout(), rust);
}

#[test]
fn the_rust_lookup_is_r2_at_of_the_header() {
    let mut a = vec![0u8; 0x100];
    let mut c = vec![0u8; 0x100];
    let blocks = [
        Block {
            base: 0x1000,
            size: 0x100,
            ptr: a.as_mut_ptr(),
            name: c"a".as_ptr(),
        },
        Block {
            base: 0x2000,
            size: 0x100,
            ptr: std::ptr::null_mut(),
            name: c"b".as_ptr(),
        },
        Block {
            base: 0x3000,
            size: 0x100,
            ptr: c.as_mut_ptr(),
            name: c"c".as_ptr(),
        },
    ];
    let m = Memory {
        blocks: blocks.as_ptr(),
        count: 3,
        granule: 512,
        sync_only: std::ptr::null(),
        sync_only_count: 0,
    };
    for (addr, len) in [
        (0x1000, 4),
        (0x10ff, 1),
        (0x10ff, 2),
        (0xfff, 4),
        (0x2010, 4),
        (0x3080, 0x80),
        (0x3080, 0x81),
        (0x4000, 1),
    ] {
        let (rust, c) = unsafe {
            (
                r2_backend::at(&m, addr, len),
                r2_backend_test::c_at(&m, addr, len),
            )
        };
        assert_eq!(rust, c, "[{addr:#x}, +{len})");
    }
}

/// A store of one block, [1 MiB, 2 MiB).
fn store() -> Store {
    let b = MemBlock::plain(
        "root.mem".into(),
        NeutrinoId(0),
        Range::new(0x10_0000, 0x20_0000),
        ResourceId(0),
        0,
    );
    Store::new(&[b], &[], 512).unwrap()
}

/// The descriptor bytes of a DMA of `n` bytes from `src` to `dst` (ABI
/// version 1: a 24-byte header, no keys, 32 bytes of arguments).
fn dma_bytes(src: u64, dst: u64, n: u64) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend(2u16.to_le_bytes()); // opcode
    b.extend(0u16.to_le_bytes()); // pe
    b.extend(1u16.to_le_bytes()); // flags: versioned
    b.extend(0u16.to_le_bytes()); // keys
    b.extend((dst / 512).to_le_bytes());
    b.extend(1u32.to_le_bytes()); // bound in granules
    b.extend(u32::MAX.to_le_bytes()); // mask
    for x in [src, dst, n, n] {
        b.extend(x.to_le_bytes());
    }
    b
}

/// A DMA op with `bytes` (empty: a task of a Rust program).
fn dma_op(seq: u64, src: u64, dst: u64, n: u64, bytes: Vec<u8>) -> FuncOp {
    FuncOp {
        seq,
        time: 100 + seq,
        nu: NeutrinoId(3),
        gid: Some(40 + seq),
        backend: 0,
        kind: OpKind::Task(Descriptor {
            opcode: OpcodeId::DMA,
            pe: Some(PeId(7)),
            dst: Range::from_len(dst, 512),
            versioned: true,
            untracked: false,
            dst_mask: nu_model::FULL_MASK,
            inferred_version: None,
            pc: Some(0x40),
            call_path: Vec::new(),
            bytes,
            deps: vec![],
            label: None,
            args: Args::Dma {
                src: Addr(src),
                dst: Addr(dst),
                bytes: n,
                tile_bytes: n,
                rows: None,
            },
        }),
    }
}

/// The test plug-in with the entry `config`.
fn plugin(config: &str) -> Result<Plugin, String> {
    let lib = test_lib();
    let pes = vec!["root.dma".to_string(), "root.atlas".to_string()];
    Plugin::open("test", lib.to_str().unwrap(), config, pes)
}

fn temp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("r2sim-plugin-{}-{name}", std::process::id()))
}

#[test]
fn a_task_reaches_the_plugin_with_its_bytes_and_the_store() {
    let log = temp("log");
    let mut p = plugin(&format!("{{\"log\": {log:?}}}")).unwrap();
    let mut m = store();
    m.write(0x10_0000, b"abcdefgh").unwrap();
    p.configure(&m).unwrap();
    let bytes = dma_bytes(0x10_0000, 0x10_0200, 8);
    p.apply(&dma_op(1, 0x10_0000, 0x10_0200, 8, bytes.clone()), &mut m)
        .unwrap();
    drop(p);
    let text = std::fs::read_to_string(&log).unwrap();
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(
        text,
        format!("pes root.dma root.atlas\n1 101 41 3 7 {hex}\n")
    );
    assert_eq!(
        m.read(0x10_0200, 8).unwrap(),
        b"abcdefgh",
        "the plug-in copied on the store"
    );
    std::fs::remove_file(log).unwrap();
}

#[test]
fn configure_passes_the_kernel_registry() {
    let out = temp("config");
    let mut p = plugin(&format!("{{\"config_out\": {out:?}}}")).unwrap();
    p.configure(&store()).unwrap();
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    std::fs::remove_file(out).unwrap();
    let k = &cfg["kernels"];
    assert_eq!(k.as_array().unwrap().len(), 4);
    let e = &k[1];
    assert_eq!(
        (&e["id"], &e["name"], &e["bytes"]),
        (&2.into(), &"epilogue".into(), &80.into())
    );
    assert_eq!(e["flags"]["GELU_TANH"], 8);
    assert_eq!(
        e["fields"][12],
        serde_json::json!({"name": "eps", "type": "f32", "offset": 72})
    );
    // The key is r2-sim's.
    let e = plugin(r#"{"kernels": []}"#).err().unwrap();
    assert!(
        e.ends_with("the configuration has the key kernels, which r2-sim sets"),
        "{e}"
    );
}

#[test]
fn a_task_without_bytes_is_an_error() {
    let mut p = plugin("{}").unwrap();
    let mut m = store();
    p.configure(&m).unwrap();
    let e = p
        .apply(&dma_op(1, 0x10_0000, 0x10_0200, 8, Vec::new()), &mut m)
        .unwrap_err();
    assert!(e.0.starts_with("backend test ("), "{}", e.0);
    assert!(e.0.contains("task 41 has no descriptor bytes"), "{}", e.0);
}

#[test]
fn errors_of_configure_and_apply_name_the_backend_and_the_library() {
    let lib = test_lib();
    let lib = lib.to_str().unwrap();
    let mut m = store();
    let mut p = plugin(r#"{"fail": "configure"}"#).unwrap();
    let e = p.configure(&m).unwrap_err();
    assert_eq!(
        e.0,
        format!("backend test ({lib}): configure: the test backend fails at configure")
    );
    let mut p = plugin(r#"{"fail": "apply"}"#).unwrap();
    p.configure(&m).unwrap();
    let e = p
        .apply(
            &dma_op(
                5,
                0x10_0000,
                0x10_0200,
                8,
                dma_bytes(0x10_0000, 0x10_0200, 8),
            ),
            &mut m,
        )
        .unwrap_err();
    assert_eq!(
        e.0,
        format!("backend test ({lib}): the test backend fails at seq 5")
    );
}

#[test]
fn a_missing_library_or_symbol_is_an_error() {
    let e = Plugin::open("py", "/nonexistent/libx.so", "{}", vec![])
        .err()
        .unwrap();
    assert!(e.starts_with("backend py (/nonexistent/libx.so): "), "{e}");
    assert!(e.contains("No such file"), "{e}");
    let e = Plugin::open("m", "libm.so.6", "{}", vec![]).err();
    // A file name is looked up next to the binary: not there.
    assert!(e.is_some());
    // The C library, which has no r2_backend_open: its path from dladdr.
    let libc_path = unsafe {
        let mut info: libc::Dl_info = std::mem::zeroed();
        assert_ne!(libc::dladdr(libc::malloc as *const _, &mut info), 0);
        std::ffi::CStr::from_ptr(info.dli_fname)
            .to_string_lossy()
            .into_owned()
    };
    let e = Plugin::open("c", &libc_path, "{}", vec![]).err().unwrap();
    assert!(e.contains("no symbol r2_backend_open"), "{e}");
}

#[test]
fn the_argument_block_of_a_muon_task_reaches_the_plugin() {
    let log = temp("block");
    let mut p = plugin(&format!("{{\"log\": {log:?}}}")).unwrap();
    let mut m = store();
    p.configure(&m).unwrap();
    // A vector descriptor (ABI version 2) with kernel 7 and a 24-byte block.
    let block: Vec<u8> = (0..24).map(|i| 0xa0 + i).collect();
    let mut desc_bytes = vec![4, 0, 0, 0, 1, 0, 0, 0];
    desc_bytes.extend((0x10_0000u64 / 512).to_le_bytes());
    desc_bytes.extend(1u32.to_le_bytes());
    desc_bytes.extend(u32::MAX.to_le_bytes());
    for w in [1u64, 64, 64, 0] {
        desc_bytes.extend(w.to_le_bytes());
    }
    desc_bytes.extend(7u32.to_le_bytes());
    desc_bytes.extend(24u32.to_le_bytes());
    desc_bytes.extend(0x200u64.to_le_bytes());
    let mut d = Descriptor::vector(PeId(2), Range::from_len(0x10_0000, 512), 1, 64, 64);
    if let Args::Vector {
        kernel, block: b, ..
    } = &mut d.args
    {
        (*kernel, *b) = (7, block.clone());
    }
    d.bytes = desc_bytes.clone();
    let op = FuncOp {
        seq: 9,
        time: 50,
        nu: NeutrinoId(1),
        gid: Some(77),
        backend: 0,
        kind: OpKind::Task(d),
    };
    p.apply(&op, &mut m).unwrap();
    drop(p);
    let text = std::fs::read_to_string(&log).unwrap();
    std::fs::remove_file(log).unwrap();
    let hex = |b: &[u8]| -> String { b.iter().map(|x| format!("{x:02x}")).collect() };
    let line = text.lines().nth(1).unwrap();
    assert_eq!(
        line,
        format!("9 50 77 1 2 {} args {}", hex(&desc_bytes), hex(&block))
    );
}
