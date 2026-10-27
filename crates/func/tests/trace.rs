use func::Trace;
use nu_model::{Addr, NeutrinoId, Range};
use sim_core::{
    Args, Descriptor, FuncOp, FunctionalBackend, MemBlock, OpKind, OpcodeId, ResourceId,
};

fn trace() -> Trace {
    let mut t = Trace::new();
    t.configure(
        &[MemBlock::plain(
            "m".into(),
            NeutrinoId(0),
            Range::new(0, 0x10_0000),
            ResourceId(0),
            0,
        )],
        &[],
        512,
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
        untracked: false,
        dst_mask: nu_model::FULL_MASK,
        inferred_version: None,
        pc: None,
        args: Args::Vector {
            ops: 1,
            bytes_in: 0,
            bytes_out: 0,
            src: None,
        },
    }
}

#[test]
fn a_compute_task_writes_nothing() {
    let mut t = trace();
    let dst = Range::from_len(0x1000, 512);
    t.write(Addr(0x1000), &[1, 2, 3, 4]).unwrap();
    t.apply(&op(task(Some(7), dst))).unwrap();
    t.apply(&op(task(None, dst))).unwrap();
    assert_eq!(t.read(Addr(0x1000), 4).unwrap(), vec![1, 2, 3, 4]);
}

#[test]
fn dma_copies_bytes() {
    let mut t = trace();
    t.write(Addr(0x100), &[5, 6, 7]).unwrap();
    let d = Descriptor {
        opcode: OpcodeId(2),
        pe: None,
        dst: Range::from_len(0x8000, 512),
        deps: vec![],
        label: None,
        versioned: true,
        untracked: false,
        dst_mask: nu_model::FULL_MASK,
        inferred_version: None,
        pc: None,
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
