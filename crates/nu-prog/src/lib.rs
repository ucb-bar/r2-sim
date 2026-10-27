//! Neutrino program library (r2sim spec Section 12.3): the calibration
//! programs, which compare the model with measurements on a card.
//!
//! * [`micro`]: micro-benchmarks, one mode per run.
//! * [`kernels`]: a sequence of GPU-style kernels (GEMM, softmax,
//!   layernorm, elementwise, attention) with the card's tiling.
//! * [`label`]: the task label scheme.
//!
//! Programs are registered by name in [`launch`]. The RV programs of the
//! Neutrino ABI run through `nu_rv::launch`.

pub mod kernels;
pub mod label;
pub mod micro;

use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

use nu_model::NeutrinoId;
use sim_core::{Built, NuCtx};

/// A boxed program value.
pub type ProgramFn = Box<dyn FnOnce(NuCtx) -> Pin<Box<dyn Future<Output = ()>>>>;

/// A program ready to spawn.
pub struct Launch {
    /// The Neutrino that runs the program.
    pub nu: NeutrinoId,
    /// The program to spawn on `nu`.
    pub program: ProgramFn,
    /// The program name (for a tracer).
    pub name: &'static str,
    /// A name found when the program starts, for a tracer instead of
    /// `name` (an RV program names itself by the record in its IMEM).
    pub start_name: Option<sim_core::ProgramName>,
    /// A micro-benchmark's record, for [`micro::result`] after the run.
    pub micro: Option<Rc<RefCell<micro::Report>>>,
    /// A kernel sequence's record, for [`kernels::result`] after the run.
    pub kernels: Option<Rc<RefCell<kernels::Report>>>,
}

impl Launch {
    /// The program `name` on `nu`, with no program-specific report.
    pub fn new(
        nu: NeutrinoId,
        name: &'static str,
        program: impl sim_core::NeutrinoProgram + 'static,
    ) -> Self {
        Self {
            nu,
            program: Box::new(move |ctx| Box::pin(program.run(ctx))),
            name,
            micro: None,
            kernels: None,
            start_name: None,
        }
    }
}

/// Parses program parameters: TOML text, then `key=value` overrides
/// (values in TOML syntax: `sms=2`, `mode="stream"`).
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

/// The programs known by name: `micro`, `kernels`.
pub fn launch(
    name: &str,
    built: &Built,
    params_text: &str,
    overrides: &[String],
) -> Result<Launch, String> {
    match name {
        "micro" => {
            let p: micro::Params = params(params_text, overrides)?;
            let run = micro::build(built, p)?;
            let (nu, report) = (run.driver, run.report.clone());
            Ok(Launch {
                nu,
                program: Box::new(move |ctx| Box::pin(run.run(ctx))),
                name: "micro",
                micro: Some(report),
                kernels: None,
                start_name: None,
            })
        }
        "kernels" => {
            let p: kernels::Params = params(params_text, overrides)?;
            let run = kernels::build(built, p)?;
            let report = run.report.clone();
            let nu = NeutrinoId(0);
            Ok(Launch {
                nu,
                program: Box::new(move |ctx| Box::pin(run.run(ctx))),
                name: "kernels",
                micro: None,
                kernels: Some(report),
                start_name: None,
            })
        }
        _ => Err(format!("unknown program {name:?} (known: micro, kernels)")),
    }
}
