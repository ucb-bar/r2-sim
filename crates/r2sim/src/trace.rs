//! Tracer export (architecture Section 7): per-resource utilization over
//! time, per-task and per-chunk timelines, program marks and phases, and
//! the culprit tree, as CSV.gz and JSON files in one directory.
//!
//! Files:
//! * `resources.csv.gz`: resource, kind, node, bin_start, busy (fraction of
//!   the bin the resource was charged; bins with nothing charged are left
//!   out).
//! * `tasks.csv.gz`: gid, node, pe, opcode, kind, step, layer, unit, index,
//!   invoke, ready, issue, complete, retire, version (empty for none).
//!   A versioned task retires at completion; an unversioned one when a
//!   new producer of its range supersedes it or its program returns.
//! * `chunks.csv.gz`: gid, index, start, end, culprit.
//! * `marks.csv`: time, name (program marks).
//! * `phases.csv`: stage, name, layer, start, end, estimate, roofline,
//!   bound_by (prefill phases; action denoise steps).
//! * `tree.json`: per node, the culprit cycles of its own resources (from
//!   the chunks in the window) and the largest one in its subtree.
//!
//! Filters: `window` keeps the tasks invoked in [a, b) with all their
//! chunks, and the bins that overlap it (the culprit tree counts only the
//! chunk time inside it); `node` keeps the tasks of the nodes whose path
//! starts with it, and the resources that such nodes own or that have no
//! owner.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use flate2::write::GzEncoder;
use flate2::Compression;
use nu_model::NeutrinoId;
use nu_prog::label;
use serde::Serialize;
use sim_core::{Built, ResourceId, Stats};

pub struct Opts {
    pub dir: PathBuf,
    pub bin: u64,
    pub window: Option<(u64, u64)>,
    pub node: Option<String>,
}

/// What the export needs from the configuration, taken before the run
/// consumes the simulation.
pub struct Names {
    /// Dotted node name by Neutrino id.
    pub nodes: Vec<String>,
    /// Parent by Neutrino id.
    pub parents: Vec<Option<usize>>,
    /// PE path by PE id.
    pub pes: Vec<String>,
    /// Owner node of each resource, by resource id.
    pub res_owner: Vec<Option<usize>>,
}

/// One phase row.
pub struct Phase {
    pub stage: String,
    pub name: String,
    pub layer: u32,
    pub start: u64,
    pub end: u64,
    pub estimate: Option<f64>,
    pub roofline: Option<f64>,
    pub bound_by: String,
}

fn gz(path: &Path) -> std::io::Result<BufWriter<GzEncoder<File>>> {
    Ok(BufWriter::new(GzEncoder::new(
        File::create(path)?,
        Compression::fast(),
    )))
}

fn finish(w: BufWriter<GzEncoder<File>>) -> std::io::Result<()> {
    w.into_inner().map_err(|e| e.into_error())?.finish()?;
    Ok(())
}

fn opt(v: Option<u64>) -> String {
    v.map_or(String::new(), |x| x.to_string())
}

/// The names of the nodes, PEs and resource owners of `b`.
pub fn names(b: &Built) -> Names {
    let topo = b.sim.topology();
    let (mut nodes, mut parents) = (vec![], vec![]);
    for i in 0..topo.len() {
        let id = NeutrinoId(i as u32);
        nodes.push(topo.path_name(id));
        parents.push(topo.node(id).parent.map(|p| p.0 as usize));
    }
    let mut pes = Vec::new();
    for (path, id) in b.pe_paths() {
        let i = id.0 as usize;
        if pes.len() <= i {
            pes.resize(i + 1, String::new());
        }
        pes[i] = path;
    }
    let res = b.sim.resources();
    let res_owner = (0..res.len())
        .map(|i| owner_of(&res.get(ResourceId(i as u32)).name, &nodes))
        .collect();
    Names {
        nodes,
        parents,
        pes,
        res_owner,
    }
}

/// The owner node of resource `name`: the longest node name it starts
/// with ("root.gpc0.sm1.smem.port"; a bridge belongs to its child node).
fn owner_of(name: &str, nodes: &[String]) -> Option<usize> {
    (0..nodes.len())
        .filter(|&i| name.starts_with(&format!("{}.", nodes[i])))
        .max_by_key(|&i| nodes[i].len())
}

pub fn write(o: &Opts, names: &Names, stats: &Stats, phases: &[Phase]) -> std::io::Result<()> {
    std::fs::create_dir_all(&o.dir)?;
    let (w0, w1) = o.window.unwrap_or((0, u64::MAX));
    let node_ok = |n: usize| match &o.node {
        None => true,
        Some(p) => names.nodes[n].starts_with(p.as_str()),
    };
    // Resources, binned.
    let mut f = gz(&o.dir.join("resources.csv.gz"))?;
    writeln!(f, "resource,kind,node,bin_start,busy,read,write,work")?;
    let bin = o.bin.max(1);
    for (i, r) in stats.resources.iter().enumerate() {
        let owner = names.res_owner.get(i).copied().flatten();
        if let (Some(_), Some(n)) = (&o.node, owner) {
            if !node_ok(n) {
                continue;
            }
        }
        let Some(log) = &r.log else { continue };
        // Per bin: busy cycles, and their read, write and work parts (an
        // interval's direction split spread evenly over it).
        let mut bins: BTreeMap<u64, (u64, [f64; 3])> = BTreeMap::new();
        for x in log {
            let len = (x.end - x.start).max(1) as f64;
            let (s, e) = (x.start.max(w0), x.end.min(w1));
            let mut t = s;
            while t < e {
                let b = t / bin * bin;
                let end = (b + bin).min(e);
                let entry = bins.entry(b).or_default();
                entry.0 += end - t;
                for (k, part) in x.dir.iter().enumerate() {
                    entry.1[k] += *part as f64 * (end - t) as f64 / len;
                }
                t = end;
            }
        }
        let node = owner.map_or("", |n| names.nodes[n].as_str());
        for (b, (busy, d)) in bins {
            let w = bin as f64;
            writeln!(
                f,
                "{},{:?},{},{},{:.4},{:.4},{:.4},{:.4}",
                r.name,
                r.kind,
                node,
                b,
                busy as f64 / w,
                d[0] / w,
                d[1] / w,
                d[2] / w
            )?;
        }
    }
    finish(f)?;
    // Tasks and chunks.
    let mut ft = gz(&o.dir.join("tasks.csv.gz"))?;
    writeln!(
        ft,
        "gid,node,pe,opcode,kind,step,layer,unit,index,invoke,ready,issue,complete,retire,version"
    )?;
    let mut fc = gz(&o.dir.join("chunks.csv.gz"))?;
    writeln!(fc, "gid,index,start,end,culprit")?;
    let mut culprit: Vec<u64> = vec![0; stats.resources.len()];
    for (gid, t) in stats.tasks.iter().enumerate() {
        let n = t.nu.0 as usize;
        if t.invoke < w0 || t.invoke >= w1 || !node_ok(n) {
            continue;
        }
        let (k, st, ly, un, ix) = t.label.map_or((0, 0, 0, 0, 0), nu_prog::label::split);
        let pe = t.pe.map_or("", |p| {
            names.pes.get(p.0 as usize).map_or("", |s| s.as_str())
        });
        writeln!(
            ft,
            "{gid},{},{pe},{},{},{st},{ly},{un},{ix},{},{},{},{},{},{}",
            names.nodes[n],
            t.opcode.0,
            label::kind_name(k),
            t.invoke,
            opt(t.ready),
            opt(t.issue),
            opt(t.complete),
            opt(t.retire),
            opt(t.version)
        )?;
        for (ci, c) in t.chunks.iter().enumerate() {
            let cul = c.culprit.map_or(String::new(), |r| {
                stats.resources[r.0 as usize].name.clone()
            });
            writeln!(fc, "{gid},{ci},{},{},{cul}", c.start, c.end)?;
            if let Some(r) = c.culprit {
                culprit[r.0 as usize] += c.end.min(w1).saturating_sub(c.start.max(w0));
            }
        }
    }
    finish(ft)?;
    finish(fc)?;
    // Marks and phases.
    let mut fm = BufWriter::new(File::create(o.dir.join("marks.csv"))?);
    writeln!(fm, "time,name")?;
    for m in &stats.marks {
        writeln!(fm, "{},{}", m.time, m.name)?;
    }
    let mut fp = BufWriter::new(File::create(o.dir.join("phases.csv"))?);
    writeln!(fp, "stage,name,layer,start,end,estimate,roofline,bound_by")?;
    for p in phases {
        let f64s = |v: Option<f64>| v.map_or(String::new(), |x| format!("{x:.0}"));
        writeln!(
            fp,
            "{},{},{},{},{},{},{},{}",
            p.stage,
            p.name,
            p.layer,
            p.start,
            p.end,
            f64s(p.estimate),
            f64s(p.roofline),
            p.bound_by
        )?;
    }
    // Culprit tree.
    let n = names.nodes.len();
    let mut own: Vec<(u64, String)> = vec![(0, String::new()); n];
    for (i, &c) in culprit.iter().enumerate() {
        if let Some(o) = names.res_owner.get(i).copied().flatten() {
            if c > own[o].0 {
                own[o] = (c, stats.resources[i].name.clone());
            }
        }
    }
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut roots = Vec::new();
    for (i, p) in names.parents.iter().enumerate() {
        match p {
            Some(p) => children[*p].push(i),
            None => roots.push(i),
        }
    }
    #[derive(Serialize)]
    struct Node {
        node: String,
        own_culprit: String,
        own_cycles: u64,
        /// The largest culprit in the subtree (the parent takes the max).
        subtree_culprit: String,
        subtree_cycles: u64,
        children: Vec<Node>,
    }
    fn build(i: usize, names: &Names, own: &[(u64, String)], ch: &[Vec<usize>]) -> Node {
        let kids: Vec<Node> = ch[i].iter().map(|&c| build(c, names, own, ch)).collect();
        let mut best = own[i].clone();
        for k in &kids {
            if k.subtree_cycles > best.0 {
                best = (k.subtree_cycles, k.subtree_culprit.clone());
            }
        }
        Node {
            node: names.nodes[i].clone(),
            own_culprit: own[i].1.clone(),
            own_cycles: own[i].0,
            subtree_culprit: best.1,
            subtree_cycles: best.0,
            children: kids,
        }
    }
    let tree: Vec<Node> = roots
        .iter()
        .map(|&r| build(r, names, &own, &children))
        .collect();
    serde_json::to_writer_pretty(File::create(o.dir.join("tree.json"))?, &tree)?;
    Ok(())
}
