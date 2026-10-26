# r2-sim

r2-sim is a performance simulator of Radiance 2, a hierarchical
accelerator. Each node of Radiance 2 has a Neutrino: a scalar core that
runs programs, with tables that track their tasks and the dependencies
between them. The tasks run on DMA engines, tensor units (Atlas) and SIMT
cores (Muon). The specifications are in [docs/](docs/): the simulator
(`r2sim.pdf`), the Neutrino (`neutrino.pdf`) and the architecture
(`radiance2.pdf`).

## Quick start

Install [pixi](https://pixi.sh), then from the repository root (with Rust
1.85 or later already installed, `cargo build --release` replaces
`pixi run build`):

```bash
pixi run build
mkdir -p out
target/release/r2sim run --config configs/2p.yaml --program hello \
  --stats out/hello.json --trace-out out/hello
```

The `hello` program ([crates/nu-prog/src/hello.rs](crates/nu-prog/src/hello.rs))
copies two 128 x 128 bf16 tiles from the DRAM of a cluster into the SMEM
of an SM, multiplies them on the Atlas of that SM, and copies the product
back. The run prints its end cycle and writes:

- `out/hello.json`: the statistics of the run;
- `out/hello/`: the trace, as CSV and JSON files (`tasks.csv.gz` has one
  row per task, `resources.csv.gz` the use of each resource over time).

To write a program, start from `hello.rs` and register it in
`nu_prog::launch`, or define it outside the crates as in
[crates/r2sim/examples/custom.rs](crates/r2sim/examples/custom.rs):

```bash
pixi run cargo run --release --example custom
```

`pixi run test` runs the tests.

The tests include the ISA tests of the Neutrino scalar core (RV64IMC, Zicsr
and the `nu.*` instructions) on its instruction-set simulator, `nu-iss`.
They come from the r2-isa-tests repository, the submodule `isa-tests`:
`git submodule update --init --recursive isa-tests` fetches them. The
crate `nu-isa-test` builds them with clang and lld, which the pixi
environment provides, so run these tests under `pixi run`.
