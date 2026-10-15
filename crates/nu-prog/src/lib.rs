//! Neutrino program library (architecture Section 6).
//!
//! * [`adamoe`]: the AdaMoE action-expert workload.
//! * [`prefill`]: the vision encoder and the VLM prefill.
//! * [`forward`]: one inference step (vision, VLM, action expert) in one
//!   HBM map.
//! * [`machine`]: the SMs, memories and allocators of a built
//!   configuration.
//! * [`micro`]: micro-benchmarks, one mode per run, to compare the model
//!   with measurements on a card.
//! * [`kernels`]: a sequence of GPU-style kernels (GEMM, softmax,
//!   layernorm, elementwise, attention) with the card's tiling.
//! * [`kv`]: the layout of the prefix KV cache in HBM.
//! * [`label`]: the task label scheme.
//! * [`synthetic`]: synthetic router traces.
//! * `graph` (private): a window of tasks built before it is invoked, so
//!   that every producer carries its expected-consumer count.
//!
//! Programs are registered by name in [`launch`].

pub mod adamoe;
pub mod forward;
mod graph;
pub mod kernels;
pub mod kv;
pub mod label;
pub mod machine;
pub mod micro;
pub mod prefill;
pub mod synthetic;
pub mod versions;

use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

use nu_model::NeutrinoId;
use sim_core::{Built, NuCtx};

pub use machine::Machine;
pub use synthetic::synthetic_trace;

/// A boxed program value.
pub type ProgramFn = Box<dyn FnOnce(NuCtx) -> Pin<Box<dyn Future<Output = ()>>>>;

/// A program ready to spawn.
pub struct Launch {
    /// The Neutrino that runs the program (the home SM, or the device
    /// with per-gpc drivers).
    pub nu: NeutrinoId,
    /// The Neutrinos whose programs invoke the tasks (the home SM, or the
    /// gpcs).
    pub drivers: Vec<NeutrinoId>,
    /// The home SM (it runs no GEMM work).
    pub home: NeutrinoId,
    /// The program to spawn on `nu`.
    pub program: ProgramFn,
    /// The program name (for a tracer).
    pub name: &'static str,
    /// The action expert's report (`adamoe`; the action stage of `forward`).
    pub report: Rc<RefCell<adamoe::Report>>,
    /// The prefill programs' report (their `report` stays empty).
    pub prefill: Option<Rc<RefCell<prefill::Report>>>,
    /// The forward pass's per-stage report (`report` is its action stage).
    pub forward: Option<Rc<RefCell<forward::Report>>>,
    /// A micro-benchmark's record, for [`micro::result`] after the run.
    pub micro: Option<Rc<RefCell<micro::Report>>>,
    /// A kernel sequence's record, for [`kernels::result`] after the run.
    pub kernels: Option<Rc<RefCell<kernels::Report>>>,
}

/// Parses program parameters: TOML text, then `key=value` overrides
/// (values in TOML syntax: `steps=2`, `expert_sms=[[1,2],[3]]`).
pub fn params<T: serde::de::DeserializeOwned>(
    text: &str,
    overrides: &[String],
) -> Result<T, String> {
    let mut table: toml::Table = toml::from_str(text).map_err(|e| e.to_string())?;
    for o in overrides {
        let (k, v) = o
            .split_once('=')
            .ok_or_else(|| format!("override {o:?} is not key=value"))?;
        let parsed: toml::Table = toml::from_str(&format!("v = {v}"))
            .or_else(|_| toml::from_str(&format!("v = {v:?}")))
            .map_err(|e| format!("override {o:?}: {e}"))?;
        table.insert(k.trim().to_string(), parsed["v"].clone());
    }
    toml::Value::Table(table)
        .try_into()
        .map_err(|e: toml::de::Error| e.to_string())
}

/// The programs known by name: `adamoe`, `prefill`, `forward`, `micro`,
/// `kernels`.
pub fn launch(
    name: &str,
    built: &Built,
    granule: u64,
    params_text: &str,
    overrides: &[String],
) -> Result<Launch, String> {
    match name {
        "adamoe" => {
            let p: adamoe::Params = params(params_text, overrides)?;
            let m = Machine::from_built(built, granule)?;
            let d = adamoe::Driver::new(m, p)?;
            let report = d.report.clone();
            let nu = d.home_nu();
            Ok(Launch {
                nu,
                drivers: vec![nu],
                home: nu,
                program: Box::new(move |ctx| Box::pin(d.run(ctx))),
                name: "adamoe",
                report,
                prefill: None,
                forward: None,
                micro: None,
                kernels: None,
            })
        }
        "forward" => {
            let p: forward::Params = params(params_text, overrides)?;
            let m = Machine::from_built(built, granule)?;
            let d = forward::Driver::new(m, p)?;
            let (report, fwd) = (d.action_report.clone(), d.report.clone());
            let (nu, drivers, home) = (d.home_nu(), d.drivers(), d.home_sm());
            Ok(Launch {
                nu,
                drivers,
                home,
                program: Box::new(move |ctx| Box::pin(d.run(ctx))),
                name: "forward",
                report,
                prefill: None,
                forward: Some(fwd),
                micro: None,
                kernels: None,
            })
        }
        "prefill" => {
            let p: prefill::Params = params(params_text, overrides)?;
            let m = Machine::from_built(built, granule)?;
            let d = prefill::Driver::new(m, p)?;
            let report = d.report.clone();
            let nu = d.home_nu();
            Ok(Launch {
                nu,
                drivers: vec![nu],
                home: nu,
                program: Box::new(move |ctx| Box::pin(d.run(ctx))),
                name: "prefill",
                report: Default::default(),
                prefill: Some(report),
                forward: None,
                micro: None,
                kernels: None,
            })
        }
        "micro" => {
            let p: micro::Params = params(params_text, overrides)?;
            let run = micro::build(built, p)?;
            let (nu, report) = (run.driver, run.report.clone());
            Ok(Launch {
                nu,
                drivers: vec![nu],
                home: nu,
                program: Box::new(move |ctx| Box::pin(run.run(ctx))),
                name: "micro",
                report: Default::default(),
                prefill: None,
                forward: None,
                micro: Some(report),
                kernels: None,
            })
        }
        "kernels" => {
            let p: kernels::Params = params(params_text, overrides)?;
            let run = kernels::build(built, p)?;
            let report = run.report.clone();
            let nu = NeutrinoId(0);
            Ok(Launch {
                nu,
                drivers: vec![nu],
                home: nu,
                program: Box::new(move |ctx| Box::pin(run.run(ctx))),
                name: "kernels",
                report: Default::default(),
                prefill: None,
                forward: None,
                micro: None,
                kernels: Some(report),
            })
        }
        _ => Err(format!(
            "unknown program {name:?} (known: adamoe, prefill, forward, micro, kernels)"
        )),
    }
}
