//! `r2sim run --config C --program P [options]` (see `USAGE`).
//!
//! Runs program P on configuration C in timing mode with the Trace backend.
//! It prints a summary to stdout and, with `--stats OUT`, writes the run as
//! JSON to OUT. `--trace-out DIR` writes the tracer files (`trace.rs`);
//! `--trace-raw DIR` writes the raw trace dump for r2-vis (`traceraw.rs`).

mod trace;
mod traceraw;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use serde::Serialize;
use sim_core::{build_config_file, ProfileRegistry, SimWarning, Stats};

const USAGE: &str = "usage: r2sim run --config C --program P [--params F] [--param k=v]... \
[--trace FILE | --synthetic SEED] [--stats OUT] [--timeline] [--quantize tile|none] \
[--cocharge longest|own] [--flow readyvalid|ticket] [--phase-stats] [--operand-reuse none|row_panel] [--acc-buffer BYTES] [--set PATH=VALUE ...] [--trace-out DIR [--bin CYCLES] [--window A:B] [--node PATH]] [--trace-raw DIR [--trace-zstd] [--trace-config FILE]] [--no-diagnostics]";

#[derive(Default)]
struct Args {
    config: Option<PathBuf>,
    program: Option<String>,
    params: Option<PathBuf>,
    overrides: Vec<String>,
    trace: Option<PathBuf>,
    synthetic: Option<u64>,
    stats: Option<PathBuf>,
    timeline: bool,
    phase_stats: bool,
    trace_out: Option<PathBuf>,
    trace_raw: Option<PathBuf>,
    trace_zstd: bool,
    trace_config: Option<PathBuf>,
    /// `--set trace.KEY=VALUE`: trace configuration overrides.
    trace_sets: Vec<String>,
    sets: Vec<String>,
    bin: Option<u64>,
    window: Option<(u64, u64)>,
    node: Option<String>,
    row_panel: Option<bool>,
    acc_buffer: Option<u64>,
    quantize: Option<bool>,
    cocharge: Option<sim_core::CoCharge>,
    flow: Option<sim_core::Flow>,
    diagnostics: bool,
    /// The flags that change the model besides `--set` (`--acc-buffer`,
    /// `--operand-reuse`, `--quantize`, `--cocharge`, `--flow`), as
    /// `"--flag value"` in command-line order, for the raw manifest.
    model_flags: Vec<String>,
}

fn parse(argv: &[String]) -> Result<Args, String> {
    let mut a = Args {
        diagnostics: true,
        ..Args::default()
    };
    let mut it = argv.iter();
    match it.next().map(String::as_str) {
        Some("run") => {}
        _ => return Err(USAGE.into()),
    }
    const MODEL_FLAGS: [&str; 5] = [
        "--acc-buffer",
        "--operand-reuse",
        "--quantize",
        "--cocharge",
        "--flow",
    ];
    while let Some(k) = it.next() {
        if MODEL_FLAGS.contains(&k.as_str()) {
            if let Some(v) = it.clone().next() {
                a.model_flags.push(format!("{k} {v}"));
            }
        }
        let mut val = || it.next().cloned().ok_or(format!("{k} needs a value"));
        match k.as_str() {
            "--config" => a.config = Some(val()?.into()),
            "--program" => a.program = Some(val()?),
            "--params" => a.params = Some(val()?.into()),
            "--param" => a.overrides.push(val()?),
            "--trace" => a.trace = Some(val()?.into()),
            "--synthetic" => a.synthetic = Some(val()?.parse().map_err(|e| format!("{e}"))?),
            "--stats" => a.stats = Some(val()?.into()),
            "--timeline" => a.timeline = true,
            "--phase-stats" => a.phase_stats = true,
            "--trace-out" => a.trace_out = Some(val()?.into()),
            "--trace-raw" => a.trace_raw = Some(val()?.into()),
            "--trace-zstd" => a.trace_zstd = true,
            "--trace-config" => a.trace_config = Some(val()?.into()),
            "--set" => {
                // Tracing is not part of the SoC configuration.
                let v = val()?;
                if v.starts_with("trace.") {
                    a.trace_sets.push(v);
                } else {
                    a.sets.push(v);
                }
            }
            "--bin" => a.bin = Some(val()?.parse().map_err(|e| format!("--bin: {e}"))?),
            "--node" => a.node = Some(val()?),
            "--window" => {
                let v = val()?;
                let (x, y) = v.split_once(':').ok_or("--window wants A:B")?;
                let p = |t: &str| t.parse::<u64>().map_err(|e| format!("--window: {e}"));
                a.window = Some((p(x)?, p(y)?));
            }
            "--acc-buffer" => {
                a.acc_buffer = Some(val()?.parse().map_err(|e| format!("--acc-buffer: {e}"))?)
            }
            "--operand-reuse" => {
                a.row_panel = Some(match val()?.as_str() {
                    "none" => false,
                    "row_panel" => true,
                    r => return Err(format!("unknown operand reuse {r:?}")),
                })
            }
            "--no-diagnostics" => a.diagnostics = false,
            "--cocharge" => {
                a.cocharge = Some(match val()?.as_str() {
                    "longest" => sim_core::CoCharge::Longest,
                    "own" => sim_core::CoCharge::Own,
                    c => return Err(format!("--cocharge {c}")),
                })
            }
            "--flow" => {
                a.flow = Some(match val()?.as_str() {
                    "readyvalid" => sim_core::Flow::ReadyValid,
                    "ticket" => sim_core::Flow::Ticket,
                    f => return Err(format!("--flow {f}")),
                })
            }
            "--quantize" => {
                a.quantize = Some(match val()?.as_str() {
                    "tile" => true,
                    "none" => false,
                    q => return Err(format!("--quantize {q}")),
                })
            }
            _ => return Err(format!("unknown argument {k}\n{USAGE}")),
        }
    }
    if a.trace_raw.is_none()
        && (a.trace_zstd || a.trace_config.is_some() || !a.trace_sets.is_empty())
    {
        return Err("--trace-zstd, --trace-config and --set trace.* need --trace-raw".into());
    }
    Ok(a)
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
    /// Slots per core cycle (Section 5.2); `busy_cycles` counts
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
    step_end: Vec<u64>,
    step_time: Vec<u64>,
    hbm_bytes: u64,
    /// HBM bytes loaded again because every attention SM computes the MQA
    /// K/V projection and streams the prefix KV itself.
    mqa_dup_bytes: u64,
    hbm_rate_bytes_per_cycle: f64,
    hbm_bound_cycles: u64,
    end_over_hbm_bound: f64,
    /// The central driver's instruction throughput used (busy / end): at 1
    /// the driver is the bottleneck.
    /// The busiest driver's scalar utilization.
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
    /// Atlas knobs of the run (H3 accumulator buffer, H1 row panel).
    acc_buffer: u64,
    row_panel: bool,
    /// Per interval between program marks (--phase-stats).
    phase_stats: Vec<PhaseStats>,
    culprits: Vec<ResourceOut>,
    /// Resources chunks queued on, by share of start delay.
    delays: Vec<(String, f64)>,
    resources: Vec<ResourceOut>,
    report: &'a nu_prog::adamoe::Report,
    prefill: Option<&'a nu_prog::prefill::Report>,
    forward: Option<&'a nu_prog::forward::Report>,
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

/// The action expert's parameters of `program` (for the trace).
fn action_params(
    program: &str,
    text: &str,
    overrides: &[String],
) -> Result<nu_prog::adamoe::Params, String> {
    if program == "forward" {
        let p: nu_prog::forward::Params = nu_prog::params(text, overrides)?;
        Ok(p.action)
    } else {
        nu_prog::params(text, overrides)
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&argv) {
        eprintln!("r2sim: {e}");
        std::process::exit(2);
    }
}

fn run(argv: &[String]) -> Result<(), String> {
    let a = parse(argv)?;
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
    let granule = 512;
    let records = match (&a.trace, a.synthetic) {
        (Some(p), None) => func::load_trace(p).map_err(|e| e.0)?,
        (None, Some(seed)) => {
            let p = action_params(&program, &params_text, &a.overrides)?;
            nu_prog::synthetic_trace(&p, seed, 4.0, 0.9)
        }
        (None, None) if program != "adamoe" => Default::default(),
        (None, None) => return Err("adamoe needs --trace or --synthetic".into()),
        _ => return Err("give only one of --trace and --synthetic".into()),
    };
    // Activation probabilities for cmem placement, from the trace.
    let mut overrides = a.overrides.clone();
    if (program == "adamoe" || program == "forward")
        && !overrides.iter().any(|o| o.starts_with("expert_prob"))
    {
        let p = action_params(&program, &params_text, &overrides)?;
        let probs = nu_prog::adamoe::expert_prob(&records, &p);
        overrides.push(format!("expert_prob={probs:?}"));
    }
    // Before the launch: the prefill cost model reads the Atlas knob.
    if let Some(on) = a.row_panel {
        b.sim.set_atlas_row_panel(on);
    }
    if let Some(bytes) = a.acc_buffer {
        b.sim.set_atlas_acc_buffer(bytes);
    }
    let launch = nu_prog::launch(&program, &b, granule, &params_text, &overrides)?;
    // HBM rate for the bound: every slice.
    let rate_f: f64 = b
        .sim
        .memories()
        .iter()
        .filter(|m| m.memory_name().ends_with(".hbm"))
        .map(|m| {
            let r = b.sim.resources().flow(m.port).rate();
            r.num as f64 / r.den as f64
        })
        .sum();
    if rate_f == 0.0 && launch.micro.is_none() && launch.kernels.is_none() {
        return Err("no memory named hbm".into());
    }

    let driver_scalars: Vec<String> = launch
        .drivers
        .iter()
        .map(|&n| format!("{}.scalar", b.sim.topology().path_name(n)))
        .collect();
    let home_path = b.sim.topology().path_name(launch.home);
    // The tracer export needs the timeline, the marks and the resource logs.
    let tracing = a.trace_out.is_some();
    // A micro-benchmark measures from the task timeline.
    b.sim
        .set_timeline(a.timeline || tracing || launch.micro.is_some());
    b.sim.set_marks(a.phase_stats || tracing);
    b.sim.set_log(tracing);
    if !a.diagnostics {
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
    if launch.kernels.is_some() {
        // The kernel sequences carry no data (no loads), and their output
        // stores name a store record as their range, not the memory they
        // write (a correlation-only convention, nu_prog::kernels): no data
        // plane.
        b.sim.set_functional(Box::new(sim_core::NullBackend));
    } else {
        b.sim.set_functional(Box::new(func::Trace::new(records)));
    }
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
                config: cfg_path.display().to_string(),
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
    sim.spawn_named(launch.nu, launch.name, launch.program);
    let t0 = Instant::now();
    let stats = sim.run();
    let wall = t0.elapsed().as_secs_f64();
    let micro = launch
        .micro
        .as_ref()
        .map(|r| nu_prog::micro::result(&r.borrow(), &stats));
    let kernels = launch
        .kernels
        .as_ref()
        .map(|r| nu_prog::kernels::result(&r.borrow()));
    let prefill = launch.prefill.as_ref().map(|r| r.borrow().clone());
    let forward = launch.forward.as_ref().map(|r| r.borrow().clone());
    if let Some((dir, out)) = &raw {
        let phases = trace_phases(prefill.as_ref(), forward.as_ref(), &launch.report.borrow());
        let rows: Vec<traceraw::PhaseRow> = phases
            .iter()
            .map(|p| traceraw::PhaseRow {
                stage: &p.stage,
                name: &p.name,
                layer: p.layer,
                start: p.start,
                end: p.end,
                estimate: p.estimate,
                roofline: p.roofline,
                bound_by: &p.bound_by,
            })
            .collect();
        let out = out.borrow();
        traceraw::write_summary(dir, &stats, wall, &out, &rows)
            .map_err(|e| format!("--trace-raw: {e}"))?;
        if let Some(e) = &out.error {
            return Err(format!("--trace-raw: {e}"));
        }
    }
    if let (Some(dir), Some(names)) = (&a.trace_out, &names) {
        let phases = trace_phases(prefill.as_ref(), forward.as_ref(), &launch.report.borrow());
        let opts = trace::Opts {
            dir: dir.clone(),
            bin: a.bin.unwrap_or(10_000),
            window: a.window,
            node: a.node.clone(),
        };
        trace::write(&opts, names, &stats, &phases).map_err(|e| format!("--trace-out: {e}"))?;
    }
    let mut report = launch.report.borrow().clone();
    if let Some(p) = &prefill {
        report.step_end = p.layer_end.clone();
        report.hbm_bytes = p.hbm_bytes;
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
    let mut step_time = Vec::new();
    let mut prev = 0;
    for &e in &report.step_end {
        step_time.push(e - prev);
        prev = e;
    }
    let bound = (report.hbm_bytes as f64 / rate_f).ceil() as u64;
    let phase_stats = phase_stats(&stats);
    let mut stats_out = stats.clone();
    stats_out.marks.clear();
    // The interval logs go to the tracer files, not into the JSON.
    for r in &mut stats_out.resources {
        r.log = None;
    }
    stats_out.warnings.truncate(KEEP_WARNINGS);
    let out = Out {
        config: cfg_path.display().to_string(),
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
        step_end: report.step_end.clone(),
        step_time,
        hbm_bytes: report.hbm_bytes,
        mqa_dup_bytes: report.mqa_dup_bytes,
        hbm_rate_bytes_per_cycle: rate_f,
        hbm_bound_cycles: bound,
        end_over_hbm_bound: stats.end_time as f64 / bound.max(1) as f64,
        driver_scalar_utilization: res
            .iter()
            .filter(|r| driver_scalars.contains(&r.name))
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
        report: &report,
        prefill: prefill.as_ref(),
        forward: forward.as_ref(),
        micro,
        kernels,
        stats: &stats_out,
    };
    print_summary(&out, &stats);
    if let Some(p) = &a.stats {
        let json = serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?;
        std::fs::write(p, json).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    if stats.fatal.is_some() {
        return Err("the run stopped on a fatal error".into());
    }
    Ok(())
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

/// The phase rows of the tracer: the prefill phases of each prefill stage,
/// and the denoise steps of the action expert when there is no prefill
/// program.
fn trace_phases(
    prefill: Option<&nu_prog::prefill::Report>,
    forward: Option<&nu_prog::forward::Report>,
    action: &nu_prog::adamoe::Report,
) -> Vec<trace::Phase> {
    let mut phases = Vec::new();
    let mut add = |stage: &str, r: &nu_prog::prefill::Report| {
        for p in &r.phases {
            phases.push(trace::Phase {
                stage: stage.into(),
                name: p.name.clone(),
                layer: p.layer,
                start: p.start,
                end: p.end,
                estimate: p.grid.map(|g| g.cost),
                roofline: p.roofline.map(|x| x.bound),
                bound_by: p.roofline.map_or("", |x| x.bound_by).to_string(),
            });
        }
    };
    if let Some(p) = prefill {
        add("prefill", p);
    }
    if let Some(f) = forward {
        if let Some(v) = &f.vision {
            add("vision", v);
        }
        if let Some(v) = &f.vlm {
            add("vlm", v);
        }
    }
    if prefill.is_none() {
        let mut prev = forward
            .and_then(|f| {
                f.stages
                    .iter()
                    .find(|s| s.name == "action")
                    .map(|s| s.start)
            })
            .unwrap_or(0);
        for (i, &e) in action.step_end.iter().enumerate() {
            phases.push(trace::Phase {
                stage: "action".into(),
                name: format!("step {i}"),
                layer: 0,
                start: prev,
                end: e,
                estimate: None,
                roofline: None,
                bound_by: String::new(),
            });
            prev = e;
        }
    }
    phases
}

/// The one-line summary with the SMEM spread, the top culprits and queues.
fn print_summary(out: &Out, stats: &Stats) {
    let s = &out.smem_port_util;
    let spread = format!(
        "SMEM port utilization over workers min {:.3} mean {:.3} max {:.3}, home {:.3}",
        s.min, s.mean, s.max, s.home
    );
    println!(
        "end {} cycles ({} steps), HBM {} B (MQA dup {}), bound {} cycles, ratio {:.3}, \
    driver scalar {:.3}, wall {:.2} s, {} events",
        out.end_time,
        out.step_end.len(),
        out.hbm_bytes,
        out.mqa_dup_bytes,
        out.hbm_bound_cycles,
        out.end_over_hbm_bound,
        out.driver_scalar_utilization,
        out.wall_seconds,
        out.events
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
    if let Some(f) = &stats.fatal {
        println!("FATAL: {f:?}");
    }
    if !stats.unfinished.is_empty() {
        println!("unfinished: {:?}", stats.unfinished);
    }
}
