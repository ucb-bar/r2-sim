//! Neutrino program library: a program ready to spawn ([`Launch`]) and the
//! task label scheme of the tracers ([`label`]).
//!
//! The programs of a run are RV programs of r2-toolchain (`nu_rv::launch`).
//! The calibration programs are r2-toolchain's examples calib-micro and
//! calib-kernels.

pub mod label;

use std::future::Future;
use std::pin::Pin;

use nu_model::NeutrinoId;
use sim_core::NuCtx;

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
}

impl Launch {
    /// The program `name` on `nu`.
    pub fn new(
        nu: NeutrinoId,
        name: &'static str,
        program: impl sim_core::NeutrinoProgram + 'static,
    ) -> Self {
        Self {
            nu,
            program: Box::new(move |ctx| Box::pin(program.run(ctx))),
            name,
            start_name: None,
        }
    }
}
