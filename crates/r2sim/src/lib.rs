//! The Radiance 2 simulator's run and its outputs: run a program on a
//! configuration in timing mode with the Trace backend, print a summary,
//! and write the JSON stats (`stats`), the tracer files (`trace_out`,
//! [`trace`]) and the raw trace dump for r2-vis (`trace_raw`,
//! [`traceraw`]). The `r2sim` command line parses its
//! arguments into [`Options`] and calls [`run`].

pub mod image;
mod sha256;
pub mod trace;
pub mod traceraw;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use nu_model::{epoch_of, Dep, NeutrinoId, Warning, COUNT_MASK};
use serde::Serialize;
use sim_core::{build_config_file, Built, FatalError, ProfileRegistry, SimWarning, Stats};

/// The command line.
pub const USAGE: &str = "usage: r2sim run --config C --program P [--params F] [--param k=v]... \
[--functional trace|numeric|python|null [--functional-cmd CMD]] [--image FILE.toml] [--image-out FILE.safetensors] [--stats OUT] [--timeline] [--thread-timeline FILE.jsonl] [--quantize tile|none] \
[--cocharge longest|own] [--flow readyvalid|ticket] [--phase-stats] [--operand-reuse none|row_panel] [--acc-buffer BYTES] [--set PATH=VALUE ...] [--trace-out DIR [--bin CYCLES] [--window A:B] [--node PATH]] [--trace-raw DIR [--trace-zstd] [--trace-config FILE]] [--no-diagnostics]";

/// A functional backend of the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Functional {
    /// Memory without compute: DMA copies and stores (`func::Trace`).
    Trace,
    /// Memory and numerics in Rust (`func::Numeric`).
    Numeric,
    /// A child process (`func::Process`), by default the Python plane.
    Python,
    /// No data (`sim_core::NullBackend`).
    Null,
}

impl std::str::FromStr for Functional {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "trace" => Ok(Self::Trace),
            "numeric" => Ok(Self::Numeric),
            "python" => Ok(Self::Python),
            "null" => Ok(Self::Null),
            _ => Err(format!("--functional {s} (trace, numeric, python or null)")),
        }
    }
}

/// The default command of the process backend.
pub const PYTHON_PLANE: &str = "python -m r2sim.func";

/// Starts the process backend: `cmd` (white-space separated) or the
/// Python plane. The Python plane finds the package `r2sim` in the
/// source tree's `python` directory, appended to PYTHONPATH.
pub fn spawn_process(cmd: Option<&str>) -> Result<func::Process, String> {
    let words: Vec<String> = cmd
        .unwrap_or(PYTHON_PLANE)
        .split_whitespace()
        .map(String::from)
        .collect();
    let mut env = Vec::new();
    if cmd.is_none() {
        let tree = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python");
        if tree.join("r2sim/func").is_dir() {
            // After the user's own entries, which keep precedence.
            let path = match std::env::var("PYTHONPATH") {
                Ok(p) if !p.is_empty() => format!("{p}:{}", tree.display()),
                _ => tree.display().to_string(),
            };
            env.push(("PYTHONPATH".to_string(), path));
        }
    }
    func::Process::spawn_with_env(&words, &env, "r2sim.func: ").map_err(|e| e.0)
}

/// What to run and what to write: the command line's options.
#[derive(Default)]
pub struct Options {
    /// `--config`: the configuration file.
    pub config: Option<PathBuf>,
    /// `--program`: `rv` or a program of `nu_prog::launch` ([`run`]).
    pub program: Option<String>,
    /// `--params`: the program's parameter file (TOML).
    pub params: Option<PathBuf>,
    /// `--param key=value`: parameter overrides.
    pub overrides: Vec<String>,
    /// `--functional`: the functional backend (default: Null for the
    /// kernels program, Trace for the others).
    pub functional: Option<Functional>,
    /// `--functional-cmd`: the command of the process backend, split at
    /// white space (default `python -m r2sim.func`).
    pub functional_cmd: Option<String>,
    /// `--image`: the memory image loaded before the run ([`image`]).
    pub image: Option<PathBuf>,
    /// `--image-out`: the named image entries, read back after the run.
    pub image_out: Option<PathBuf>,
    /// `--stats`: the JSON stats file.
    pub stats: Option<PathBuf>,
    /// `--timeline`: per-task records in the stats.
    pub timeline: bool,
    /// `--thread-timeline`: the wall-clock timeline of the timing thread
    /// and the functional worker, as JSON lines.
    pub thread_timeline: Option<PathBuf>,
    /// `--phase-stats`: statistics per interval between program marks.
    pub phase_stats: bool,
    /// `--trace-out DIR`: the tracer files ([`trace`]).
    pub trace_out: Option<PathBuf>,
    /// `--trace-raw DIR`: the raw trace dump ([`traceraw`]).
    pub trace_raw: Option<PathBuf>,
    /// `--trace-zstd`: compress the raw dump.
    pub trace_zstd: bool,
    /// `--trace-config`: the raw dump's configuration file.
    pub trace_config: Option<PathBuf>,
    /// `--set trace.KEY=VALUE`: trace configuration overrides.
    pub trace_sets: Vec<String>,
    /// `--set PATH=VALUE`: configuration overrides.
    pub sets: Vec<String>,
    /// `--bin`: the tracer files' bin in cycles.
    pub bin: Option<u64>,
    /// `--window A:B`: the tracer files' window.
    pub window: Option<(u64, u64)>,
    /// `--node`: the tracer files' subtree.
    pub node: Option<String>,
    /// `--operand-reuse`: row_panel (true) or none.
    pub row_panel: Option<bool>,
    /// `--acc-buffer`: every Atlas's accumulator buffer.
    pub acc_buffer: Option<u64>,
    /// `--quantize`: tile (true) or none.
    pub quantize: Option<bool>,
    /// `--cocharge`.
    pub cocharge: Option<sim_core::CoCharge>,
    /// `--flow`.
    pub flow: Option<sim_core::Flow>,
    /// `--no-diagnostics`: no open-pending report and no late-consumer
    /// check.
    pub no_diagnostics: bool,
    /// The flags that change the model besides `--set` (`--acc-buffer`,
    /// `--operand-reuse`, `--quantize`, `--cocharge`, `--flow`), as
    /// `"--flag value"` in command-line order, for the raw manifest.
    pub model_flags: Vec<String>,
}

/// Resource shares in one interval between program marks.
#[derive(Serialize)]
struct PhaseStats {
    name: String,
    start: u64,
    end: u64,
    /// Top resources by share of chunk culprit cycles, of start delay, and
    /// by utilization (busy / interval).
    culprits: Vec<(String, f64)>,
    delays: Vec<(String, f64)>,
    busiest: Vec<(String, f64)>,
}

fn phase_stats(stats: &Stats) -> Vec<PhaseStats> {
    let slots: Vec<f64> = stats.resources.iter().map(|r| r.slots.max(1e-9)).collect();
    let names: Vec<&str> = stats.resources.iter().map(|r| r.name.as_str()).collect();
    let top = |v: Vec<(usize, f64)>| -> Vec<(String, f64)> {
        let mut v: Vec<(usize, f64)> = v.into_iter().filter(|x| x.1 > 0.0).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v.into_iter()
            .take(5)
            .map(|(i, x)| (names[i].to_string(), x))
            .collect()
    };
    stats
        .marks
        .windows(2)
        .map(|w| {
            let (a, b) = (&w[0], &w[1]);
            let d: Vec<[u64; 3]> = b
                .res
                .iter()
                .zip(&a.res)
                .map(|(x, y)| [x[0] - y[0], x[1] - y[1], x[2] - y[2]])
                .collect();
            let len = (b.time - a.time).max(1) as f64;
            let sum = |k: usize| d.iter().map(|x| x[k]).sum::<u64>().max(1) as f64;
            let share = |k: usize, den: f64| {
                top(d
                    .iter()
                    .enumerate()
                    .map(|(i, x)| (i, x[k] as f64 / den))
                    .collect())
            };
            // Busy lane-cycles over all lanes of the interval.
            let busiest = top(d
                .iter()
                .enumerate()
                .map(|(i, x)| (i, x[0] as f64 / (len * slots[i])))
                .collect());
            PhaseStats {
                name: b.name.clone(),
                start: a.time,
                end: b.time,
                culprits: share(1, sum(1)),
                delays: share(2, sum(2)),
                busiest,
            }
        })
        .collect()
}

/// SMEM port utilization over the SMs other than the driver's.
#[derive(Serialize)]
struct Spread {
    min: f64,
    max: f64,
    mean: f64,
    /// The driver SM's own port.
    home: f64,
    per_sm: Vec<(String, f64)>,
}

#[derive(Serialize, Clone)]
struct ResourceOut {
    name: String,
    kind: String,
    /// Slots per core cycle (r2sim spec Section 5.2); `busy_cycles` counts
    /// slot-cycles, rounded.
    slots: f64,
    busy_cycles: u64,
    busy_slot_cycles: f64,
    /// Bytes charged, and their share of `rate x end`.
    bytes: u64,
    byte_utilization: f64,
    /// Occupancy resources: the size and the peak; an in-flight pool also
    /// its mean occupancy (held byte-cycles over the end time) and, in
    /// `delay_cycles`, the cycles chunks waited for it.
    size: u64,
    peak: u64,
    mean_occupancy: f64,
    /// `busy_cycles` by direction (they sum to it).
    busy_read: u64,
    busy_write: u64,
    busy_work: u64,
    utilization: f64,
    culprit_cycles: u64,
    culprit_share: f64,
    delay_cycles: u64,
    delay_share: f64,
}

#[derive(Serialize)]
struct Out<'a> {
    config: String,
    program: String,
    quantize: Option<bool>,
    /// The `--cocharge` and `--flow` overrides ("default": the
    /// configuration's).
    cocharge: &'static str,
    flow: &'static str,
    end_time: u64,
    /// The scalar utilization (busy / end) of the Neutrino that runs the
    /// program (the root for an RV run): at 1 its program is the
    /// bottleneck.
    driver_scalar_utilization: f64,
    wall_seconds: f64,
    events: u64,
    events_per_second: f64,
    tasks: u64,
    messages: u64,
    rejects: u64,
    /// Program cycles stalled on full outbound invoke tables, summed.
    outbound_stall_total: u64,
    resends: u64,
    /// The most version table entries held at once, over all Neutrinos.
    version_peak_max: usize,
    warnings: BTreeMap<String, u64>,
    /// All warnings; `stats.warnings` keeps the first 1000.
    warnings_total: usize,
    smem_port_util: Spread,
    /// Atlas knobs of the run (accumulator buffer, row panel).
    acc_buffer: u64,
    row_panel: bool,
    /// Per interval between program marks (--phase-stats).
    phase_stats: Vec<PhaseStats>,
    culprits: Vec<ResourceOut>,
    /// Resources chunks queued on, by share of start delay.
    delays: Vec<(String, f64)>,
    resources: Vec<ResourceOut>,
    /// A micro-benchmark's measured quantities (absent for the other
    /// programs, so their JSON does not change).
    #[serde(skip_serializing_if = "Option::is_none")]
    micro: Option<serde_json::Value>,
    /// A kernel sequence's per-kernel record (absent for the other
    /// programs).
    #[serde(skip_serializing_if = "Option::is_none")]
    kernels: Option<serde_json::Value>,
    stats: &'a Stats,
}

/// Runs the program `o.program` (a program of `nu_prog::launch`) on the
/// configuration `o.config` and writes the outputs that `o` asks for.
pub fn run(o: &Options) -> Result<(), String> {
    let a = o;
    let cfg_path = a.config.clone().ok_or(USAGE)?;
    let program = a.program.clone().ok_or(USAGE)?;
    let params_text = match &a.params {
        Some(p) => std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?,
        None => String::new(),
    };
    let mut b = build_config_file(&cfg_path, &a.sets, ProfileRegistry::with_builtins())
        .map_err(|e| format!("{}: {e}", cfg_path.display()))?;
    for w in &b.warnings {
        eprintln!("r2sim: warning: {}: {w}", cfg_path.display());
    }
    // The Neutrino granule of the configuration (512 B on the 2p family,
    // 128 B on rtx6000 since 2026-10-21).
    let granule = b
        .sim
        .neutrino_config(nu_model::NeutrinoId(0))
        .tables
        .granule;
    // Before the launch: the kernel sequences read the Atlas knobs.
    if let Some(on) = a.row_panel {
        b.sim.set_atlas_row_panel(on);
    }
    if let Some(bytes) = a.acc_buffer {
        b.sim.set_atlas_acc_buffer(bytes);
    }
    let launch = if program == "rv" {
        rv_launch(a, &b)?
    } else {
        nu_prog::launch(&program, &b, &params_text, &a.overrides)?
    };
    let cfg = cfg_path.display().to_string();
    execute(a, b, launch, &cfg, program, granule)
}

/// `--program rv` ([`nu_rv::launch`]): the root program of the root's
/// IMEM. `--param args=NAME`: the arguments are at the image entry NAME
/// (default address 0); `--param limit=N`: the run stops after N
/// instructions of the program (default [`nu_rv::DEFAULT_LIMIT`]).
fn rv_launch(a: &Options, b: &Built) -> Result<nu_prog::Launch, String> {
    let (mut args, mut limit) = (0, nu_rv::DEFAULT_LIMIT);
    for o in &a.overrides {
        match o.split_once('=') {
            Some(("args", name)) => {
                let path = a.image.as_ref().ok_or("--param args=NAME needs --image")?;
                let img = image::Image::load(path, b).map_err(|e| format!("--image: {e}"))?;
                args = img
                    .address_of(name)
                    .ok_or_else(|| format!("--param args={name}: no image entry {name:?}"))?;
            }
            Some(("limit", n)) => {
                limit = n.parse().map_err(|e| format!("--param limit={n}: {e}"))?;
            }
            _ => return Err(format!("--program rv: unknown --param {o} (args, limit)")),
        }
    }
    nu_rv::launch(b, args, limit)
}

/// Runs `launch` on `b` and writes the outputs (the tail of [`run`]).
fn execute(
    a: &Options,
    mut b: Built,
    mut launch: nu_prog::Launch,
    cfg: &str,
    program: String,
    granule: u64,
) -> Result<(), String> {
    let home_path = b.sim.topology().path_name(launch.nu);
    let driver_scalar = format!("{home_path}.scalar");
    // The tracer export needs the timeline, the marks and the resource logs.
    let tracing = a.trace_out.is_some();
    // A micro-benchmark measures from the task timeline.
    b.sim
        .set_timeline(a.timeline || tracing || launch.micro.is_some());
    b.sim.set_marks(a.phase_stats || tracing);
    b.sim.set_log(tracing);
    if a.no_diagnostics {
        b.sim.set_open_pending_report(false);
        b.sim.set_late_consumer_check(false);
    }
    if let Some(q) = a.quantize {
        b.sim.set_atlas_quantize(q);
    }
    if let Some(c) = a.cocharge {
        b.sim.resources_mut().set_cocharge(c);
    }
    if let Some(f) = a.flow {
        b.sim.set_flow(f);
    }
    // The kernel sequences carry no data (no loads), and their output
    // stores name a store record as their range, not the memory they
    // write (a correlation-only convention, nu_prog::kernels): no data
    // plane by default.
    let functional = a.functional.unwrap_or(if launch.kernels.is_some() {
        Functional::Null
    } else {
        Functional::Trace
    });
    let backend: Box<dyn sim_core::FunctionalBackend> = match functional {
        Functional::Null => Box::new(sim_core::NullBackend),
        Functional::Trace => Box::new(func::Trace::new()),
        Functional::Numeric => Box::new(func::Numeric::new()),
        // Spawned here, on the main thread: the child's parent-death
        // signal follows the thread that spawns it.
        Functional::Python => {
            let mut p = spawn_process(a.functional_cmd.as_deref())?;
            p.set_op_time(a.thread_timeline.is_some());
            Box::new(p)
        }
    };
    b.sim.set_thread_timeline(a.thread_timeline.is_some());
    let image = match &a.image {
        Some(p) => {
            if !backend.has_data() {
                return Err("--image needs a functional backend with data".into());
            }
            let img = image::Image::load(p, &b).map_err(|e| format!("--image: {e}"))?;
            img.install(&mut b).map_err(|e| format!("--image: {e}"))?;
            Some(img)
        }
        None if a.image_out.is_some() => return Err("--image-out needs --image".into()),
        None => None,
    };
    // An RV run makes its lazy resources (instruction DMAs, data scratch
    // ports) now, so that the raw trace lists them with fixed ids.
    if program == "rv" {
        b.sim.make_core_resources();
    }
    let (blocks, sync_only) = (b.sim.functional_blocks(), b.sim.sync_only_ranges().to_vec());
    b.sim.set_functional(backend);
    // The names are taken now, before the run consumes the simulation.
    let names = tracing.then(|| trace::names(&b));
    let atlas = b
        .pe_paths()
        .into_iter()
        .find_map(|(_, id)| match b.sim.pe_config(id).params {
            sim_core::PeParams::Atlas {
                acc_buffer,
                row_panel,
                ..
            } => Some((acc_buffer, row_panel)),
            _ => None,
        })
        .unwrap_or((0, false));
    // The raw dump writes its manifest now, from the configuration.
    let raw = match &a.trace_raw {
        Some(dir) => {
            let opts = traceraw::opts(
                dir.clone(),
                a.trace_config.as_deref(),
                &a.trace_sets,
                a.trace_zstd,
            )
            .map_err(|e| format!("--trace-raw: {e}"))?;
            let run = traceraw::RunInfo {
                config: cfg.to_string(),
                program: program.clone(),
                params_file: a.params.as_ref().map(|p| p.display().to_string()),
                params: a.overrides.clone(),
                overrides: a.sets.iter().chain(&a.model_flags).cloned().collect(),
                granule,
            };
            let (t, out) = traceraw::RawTracer::create(&opts, &b, &run)
                .map_err(|e| format!("--trace-raw: {e}"))?;
            b.sim.set_tracer(Box::new(t));
            Some((dir.clone(), out))
        }
        None => None,
    };
    let mut sim = b.sim;
    let nodes: Vec<String> = (0..sim.topology().len())
        .map(|i| sim.topology().path_name(NeutrinoId(i as u32)))
        .collect();
    match launch.start_name.take() {
        Some(name) => sim.spawn_program(launch.nu, name, launch.program),
        None => sim.spawn_named(launch.nu, launch.name, launch.program),
    }
    let t0 = Instant::now();
    let (stats, mut backend) = sim.run_with_backend();
    let wall = t0.elapsed().as_secs_f64();
    if let (Some(img), Some(out), Some(be), None) =
        (&image, &a.image_out, backend.as_deref_mut(), &stats.fatal)
    {
        img.write_out(be, &blocks, &sync_only, out)
            .map_err(|e| format!("--image-out: {e}"))?;
    }
    // The process backend's child exits here.
    drop(backend);
    let micro = launch
        .micro
        .as_ref()
        .map(|r| nu_prog::micro::result(&r.borrow(), &stats));
    let kernels = launch
        .kernels
        .as_ref()
        .map(|r| nu_prog::kernels::result(&r.borrow(), &stats));
    // No program reports phases: the phase rows of the raw summary and
    // of the tracer files are empty.
    if let Some((dir, out)) = &raw {
        let out = out.borrow();
        traceraw::write_summary(dir, &stats, wall, &out, &[])
            .map_err(|e| format!("--trace-raw: {e}"))?;
        if let Some(e) = &out.error {
            return Err(format!("--trace-raw: {e}"));
        }
    }
    if let (Some(dir), Some(names)) = (&a.trace_out, &names) {
        let opts = trace::Opts {
            dir: dir.clone(),
            bin: a.bin.unwrap_or(10_000),
            window: a.window,
            node: a.node.clone(),
        };
        trace::write(&opts, names, &stats, &[]).map_err(|e| format!("--trace-out: {e}"))?;
    }
    // The warning list can be long (one record per pipelined slot reuse);
    // the JSON keeps the first ones and the per-kind counts.
    const KEEP_WARNINGS: usize = 1000;
    let warnings_total = stats.warnings.len();

    let res = resource_outs(&stats);
    // Per SM: its SMEM read port, or on a banked SMEM the mean over its
    // banks' read ports (named `<sm>.smem.b*.port`).
    let smem_key = |name: &str| -> Option<String> {
        if name.ends_with(".smem.port") {
            return Some(name.to_string());
        }
        let (head, bank) = name.strip_suffix(".port")?.rsplit_once(".smem.b")?;
        bank.parse::<u32>()
            .ok()
            .map(|_| format!("{head}.smem.b*.port"))
    };
    let mut grouped: Vec<(String, f64, u32)> = Vec::new();
    for r in &res {
        let Some(k) = smem_key(&r.name) else { continue };
        match grouped.iter_mut().find(|g| g.0 == k) {
            Some(g) => {
                g.1 += r.utilization;
                g.2 += 1;
            }
            None => grouped.push((k, r.utilization, 1)),
        }
    }
    let smem_util: Vec<(String, f64)> = grouped
        .into_iter()
        .map(|(k, u, n)| (k, u / n as f64))
        .collect();
    let home_port = [
        format!("{}.smem.port", home_path),
        format!("{}.smem.b*.port", home_path),
    ];
    let per_sm: Vec<(String, f64)> = smem_util
        .iter()
        .filter(|r| !home_port.contains(&r.0))
        .cloned()
        .collect();
    let us = || per_sm.iter().map(|x| x.1);
    let smem_port_util = Spread {
        min: us().fold(f64::INFINITY, f64::min),
        max: us().fold(0.0, f64::max),
        mean: us().sum::<f64>() / per_sm.len().max(1) as f64,
        home: smem_util
            .iter()
            .find(|r| home_port.contains(&r.0))
            .map_or(0.0, |r| r.1),
        per_sm: per_sm.clone(),
    };
    let mut culprits: Vec<ResourceOut> = res
        .iter()
        .filter(|r| r.culprit_cycles > 0)
        .cloned()
        .collect();
    culprits.sort_by_key(|r| std::cmp::Reverse(r.culprit_cycles));
    let mut delays: Vec<&ResourceOut> = res.iter().filter(|r| r.delay_cycles > 0).collect();
    delays.sort_by_key(|r| std::cmp::Reverse(r.delay_cycles));
    let delays: Vec<(String, f64)> = delays
        .iter()
        .take(8)
        .map(|r| (r.name.clone(), r.delay_share))
        .collect();
    let warnings = warning_counts(&stats);
    let phase_stats = phase_stats(&stats);
    let mut stats_out = stats.clone();
    stats_out.marks.clear();
    // The interval logs go to the tracer files, not into the JSON.
    for r in &mut stats_out.resources {
        r.log = None;
    }
    stats_out.warnings.truncate(KEEP_WARNINGS);
    let out = Out {
        config: cfg.to_string(),
        program,
        quantize: a.quantize,
        cocharge: match a.cocharge {
            None => "default",
            Some(sim_core::CoCharge::Longest) => "longest",
            Some(sim_core::CoCharge::Own) => "own",
        },
        flow: match a.flow {
            None => "default",
            Some(sim_core::Flow::ReadyValid) => "readyvalid",
            Some(sim_core::Flow::Ticket) => "ticket",
        },
        end_time: stats.end_time,
        driver_scalar_utilization: res
            .iter()
            .filter(|r| r.name == driver_scalar)
            .map(|r| r.utilization)
            .fold(0.0, f64::max),
        wall_seconds: wall,
        events: stats.events,
        events_per_second: stats.events as f64 / wall.max(1e-9),
        tasks: stats.invoked,
        messages: stats.messages,
        rejects: stats.rejects,
        outbound_stall_total: stats.outbound_stall.iter().sum(),
        resends: stats.resends,
        version_peak_max: stats.version_peak.iter().copied().max().unwrap_or(0),
        warnings,
        warnings_total,
        smem_port_util,
        acc_buffer: atlas.0,
        row_panel: atlas.1,
        phase_stats,
        culprits,
        delays,
        resources: res,
        micro,
        kernels,
        stats: &stats_out,
    };
    print_summary(&out, &stats, &nodes);
    if let Some(p) = &a.thread_timeline {
        write_thread_timeline(p, &stats.thread_timeline)
            .map_err(|e| format!("--thread-timeline {}: {e}", p.display()))?;
    }
    if let Some(p) = &a.stats {
        let json = serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?;
        std::fs::write(p, json).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    if stats.fatal.is_some() {
        return Err("the run stopped on a fatal error".into());
    }
    Ok(())
}

/// Writes the thread timeline as JSON lines, one record per line.
fn write_thread_timeline(
    path: &std::path::Path,
    records: &[sim_core::ThreadRecord],
) -> std::io::Result<()> {
    use std::io::Write;
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    for r in records {
        serde_json::to_writer(&mut w, r)?;
        w.write_all(b"\n")?;
    }
    w.flush()
}

/// Per-resource totals of the run, with utilization and shares.
fn resource_outs(stats: &Stats) -> Vec<ResourceOut> {
    let end = stats.end_time.max(1);
    let total_culprit: u64 = stats.resources.iter().map(|r| r.culprit_cycles).sum();
    let total_delay: u64 = stats.resources.iter().map(|r| r.delay_cycles).sum();
    stats
        .resources
        .iter()
        .map(|r| ResourceOut {
            name: r.name.clone(),
            kind: format!("{:?}", r.kind),
            slots: r.slots,
            busy_cycles: r.busy_cycles,
            busy_slot_cycles: r.busy_slot_cycles,
            bytes: r.bytes,
            byte_utilization: r.bytes as f64 / (end as f64 * r.rate.max(1e-9)),
            size: r.size,
            peak: r.peak,
            mean_occupancy: r.held_cycles / end as f64,
            busy_read: r.busy_read,
            busy_write: r.busy_write,
            busy_work: r.busy_work,
            utilization: r.busy_slot_cycles / (end as f64 * r.slots.max(1e-9)),
            culprit_cycles: r.culprit_cycles,
            culprit_share: r.culprit_cycles as f64 / total_culprit.max(1) as f64,
            delay_cycles: r.delay_cycles,
            delay_share: r.delay_cycles as f64 / total_delay.max(1) as f64,
        })
        .collect()
}

/// Warning counts by kind.
fn warning_counts(stats: &Stats) -> BTreeMap<String, u64> {
    let mut warnings = BTreeMap::new();
    for w in &stats.warnings {
        let k = match w.warning {
            SimWarning::Dep(d) => format!("{d:?}")
                .split_whitespace()
                .next()
                .unwrap_or("Dep")
                .to_string(),
            SimWarning::Race { .. } => "Race".into(),
            SimWarning::LateConsumer { .. } => "LateConsumer".into(),
        };
        *warnings.entry(k).or_insert(0) += 1;
    }
    warnings
}

/// The one-line summary with the SMEM spread, the top culprits and queues;
/// the first warning of a kind that names a likely hang, and the report
/// of a deadlock. `nodes`: the node names by Neutrino.
fn print_summary(out: &Out, stats: &Stats, nodes: &[String]) {
    let s = &out.smem_port_util;
    let spread = format!(
        "SMEM port utilization over workers min {:.3} mean {:.3} max {:.3}, home {:.3}",
        s.min, s.mean, s.max, s.home
    );
    println!(
        "end {} cycles, driver scalar {:.3}, wall {:.2} s, {} events",
        out.end_time, out.driver_scalar_utilization, out.wall_seconds, out.events
    );
    println!("  {spread}");
    for c in out.culprits.iter().take(5) {
        println!(
            "  culprit {:<24} {:>6.1}% of chunk cycles, utilization {:.3}",
            c.name,
            100.0 * c.culprit_share,
            c.utilization
        );
    }
    for (n, sh) in out.delays.iter().take(3) {
        println!("  queued on {:<24} {:>6.1}% of start delay", n, 100.0 * sh);
    }
    let first = stats.warnings.iter().find_map(|w| match w.warning {
        SimWarning::Dep(Warning::UnversionedWait { addr, versioned }) => Some((w, addr, versioned)),
        _ => None,
    });
    if let Some((w, addr, versioned)) = first {
        println!(
            "warning UnversionedWait ({} in all; the first at cycle {} on {}): the unversioned key {addr:#x} meets the versioned key {}",
            out.warnings.get("UnversionedWait").copied().unwrap_or(0),
            w.time,
            nodes[w.nu.0 as usize],
            key_name(&versioned)
        );
    }
    match &stats.fatal {
        Some(f) if matches!(f.error, FatalError::Deadlock { .. }) => print_deadlock(stats, nodes),
        Some(f) => {
            println!("FATAL: {f:?}");
            if !stats.unfinished.is_empty() {
                println!("unfinished: {:?}", stats.unfinished);
            }
        }
        None => {}
    }
}

/// `addr` and the epoch and count of a versioned key.
fn key_name(k: &Dep) -> String {
    match *k {
        Dep::Versioned { addr, version } => format!(
            "({addr:#x}, epoch {} count {})",
            epoch_of(version),
            version & COUNT_MASK
        ),
        Dep::Unversioned(addr) => format!("({addr:#x}, unversioned)"),
    }
}

/// The deadlock report: the programs that have not returned and what they
/// wait on, the open pending entries and the channel stages that hold
/// messages.
fn print_deadlock(stats: &Stats, nodes: &[String]) {
    let Some(f) = &stats.fatal else { return };
    let FatalError::Deadlock {
        idle,
        programs,
        stages,
        ..
    } = &f.error
    else {
        return;
    };
    let name = |nu: NeutrinoId| nodes[nu.0 as usize].as_str();
    println!(
        "FATAL: Deadlock at cycle {} on {} (no progress for {idle} cycles)",
        f.time,
        name(f.nu)
    );
    for p in programs {
        let on = match (p.key, p.owner) {
            (Some(k), Some(o)) => format!(" on {} owned by {}", key_name(&k), name(o)),
            (Some(k), None) => format!(" on {}", key_name(&k)),
            _ => String::new(),
        };
        println!(
            "  program on {}: {:?} in {}{on}",
            name(p.nu),
            p.blocked,
            p.call
        );
    }
    for p in &stats.open_pending {
        let key = match p.pending.version {
            Some(version) => Dep::Versioned {
                addr: p.pending.addr,
                version,
            },
            None => Dep::Unversioned(p.pending.addr),
        };
        println!(
            "  open pending entry {} at its owner {}: {} waiters",
            key_name(&key),
            name(p.nu),
            p.pending.waiters
        );
    }
    for s in stages {
        println!("  stage {}: {} messages", s.stage, s.messages);
    }
}
