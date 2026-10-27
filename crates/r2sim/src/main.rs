//! `r2sim run --config C --program P [options]` (see `USAGE`).
//!
//! Runs program P on configuration C in timing mode with the Trace backend
//! (`--functional` selects another; `--image` loads a memory image).
//! It prints a summary to stdout and, with `--stats OUT`, writes the run as
//! JSON to OUT. `--trace-out DIR` writes the tracer files;
//! `--trace-raw DIR` writes the raw trace dump for r2-vis. The run itself
//! is the library's (`r2sim::run`); this file parses the arguments.

use r2sim::{Options, USAGE};

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = parse(&argv).and_then(|o| r2sim::run(&o)) {
        eprintln!("r2sim: {e}");
        std::process::exit(2);
    }
}

fn parse(argv: &[String]) -> Result<Options, String> {
    let mut a = Options::default();
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
            "--functional" => a.functional = Some(val()?.parse()?),
            "--functional-cmd" => a.functional_cmd = Some(val()?),
            "--image" => a.image = Some(val()?.into()),
            "--image-out" => a.image_out = Some(val()?.into()),
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
            "--no-diagnostics" => a.no_diagnostics = true,
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
    if a.functional_cmd.is_some() && a.functional != Some(r2sim::Functional::Python) {
        return Err("--functional-cmd needs --functional python".into());
    }
    Ok(a)
}
