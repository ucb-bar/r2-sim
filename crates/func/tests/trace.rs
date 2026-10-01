use func::{decode_trace, encode_trace, Records, Trace};
use nu_model::{Addr, NeutrinoId, Range};
use sim_core::{
    Args, Descriptor, FuncOp, FunctionalBackend, MemBlock, OpKind, OpcodeId, ResourceId,
};

fn trace(records: Records) -> Trace {
    let mut t = Trace::new(records);
    t.configure(
        &[MemBlock {
            name: "m".into(),
            owner: NeutrinoId(0),
            attach: Vec::new(),
            numeric: true,
            range: Range::new(0, 0x10_0000),
            port: ResourceId(0),
            load_latency: 0,
        }],
        &[],
    );
    t
}

fn op(desc: Descriptor) -> FuncOp {
    FuncOp {
        seq: 1,
        time: 0,
        nu: NeutrinoId(0),
        gid: Some(0),
        kind: OpKind::Task(desc),
    }
}

fn task(label: Option<u64>, dst: Range) -> Descriptor {
    Descriptor {
        opcode: OpcodeId(3),
        pe: None,
        dst,
        deps: vec![],
        label,
        versioned: true,
        inferred_version: None,
        args: Args::Vector {
            ops: 1,
            bytes_in: 0,
            bytes_out: 0,
            src: None,
        },
    }
}

#[test]
fn file_format_round_trips() {
    let mut r = Records::new();
    r.insert(7, vec![1, 2, 3]);
    r.insert(0x0100_0203, vec![]);
    let bytes = encode_trace(&r);
    assert_eq!(&bytes[..8], b"R2TRACE1");
    assert_eq!(decode_trace(&bytes).unwrap(), r);
    assert!(decode_trace(&bytes[..bytes.len() - 1]).is_err());
}

#[test]
fn labeled_task_writes_its_record_and_others_write_nothing() {
    let mut r = Records::new();
    r.insert(7, vec![9; 51]);
    let mut t = trace(r);
    let dst = Range::from_len(0x1000, 512);
    t.apply(&op(task(Some(7), dst))).unwrap();
    assert_eq!(t.read(Addr(0x1000), 51).unwrap(), vec![9; 51]);
    t.apply(&op(task(Some(8), Range::from_len(0x2000, 512))))
        .unwrap();
    t.apply(&op(task(None, Range::from_len(0x2000, 512))))
        .unwrap();
    assert_eq!(t.read(Addr(0x2000), 4).unwrap(), vec![0; 4]);
    assert_eq!(t.hits(), 1);
    // A record larger than dst is an error.
    let mut r = Records::new();
    r.insert(1, vec![0; 1024]);
    let mut t = trace(r);
    assert!(t.apply(&op(task(Some(1), dst))).is_err());
}

#[test]
fn dma_copies_bytes() {
    let mut t = trace(Records::new());
    t.write(Addr(0x100), &[5, 6, 7]).unwrap();
    let d = Descriptor {
        opcode: OpcodeId(2),
        pe: None,
        dst: Range::from_len(0x8000, 512),
        deps: vec![],
        label: None,
        versioned: true,
        inferred_version: None,
        args: Args::Dma {
            src: Addr(0x100),
            dst: Addr(0x8000),
            bytes: 3,
            tile_bytes: 3,
        },
    };
    t.apply(&op(d)).unwrap();
    assert_eq!(t.read(Addr(0x8000), 3).unwrap(), vec![5, 6, 7]);
}
