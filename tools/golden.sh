#!/bin/bash
# Golden check of the headline: runs the 10 forward-pass rows of
# docs/results.md Section 3 and compares them with a reference run.
#
#   tools/golden.sh record DIR        # run the rows into DIR (the reference)
#   tools/golden.sh check REF DIR     # run into DIR and compare with REF
#
# Run from r2-sim/ after a release build. R2SIM overrides the binary
# (default ../build/target/release/r2sim); GOLDEN_ARGS adds arguments to
# every run (for example GOLDEN_ARGS="--set global.endpoint_charging=false").
# The rows run the default hierarchical program (configs/forward.toml);
# GOLDEN_ARGS="--param drivers=single" runs the single driver. References:
# scratch/fwd15 (hierarchical), scratch/fwd15-single.
# "check" exits with 1 when a summary line or a JSON file differs; the
# comparison leaves out the wall time and the functional worker's progress
# at each load (thread timing).
set -euo pipefail
R2SIM=${R2SIM:-../build/target/release/r2sim}

run_all() {
  local out=$1
  mkdir -p "$out"
  run() { # name config weight_dtype acc_buffer
    "$R2SIM" run --config "configs/$2" --program forward --params configs/forward.toml \
      --param "weight_dtype=$3" --acc-buffer "$4" --synthetic 7 \
      --stats "$out/$1.json" ${GOLDEN_ARGS:-} > "$out/$1.log" 2>&1
  }
  run 2p_bf16_a0 2p.yaml bf16 0 &
  run 2p_bf16_a1048576 2p.yaml bf16 1048576 &
  run 2p_mx_a0 2p.yaml mxfp4 0 &
  run 2p_mx_a1048576 2p.yaml mxfp4 1048576 &
  run c64_mx_a0 2p-cmem64.yaml mxfp4 0 &
  run c64_mx_a1048576 2p-cmem64.yaml mxfp4 1048576 &
  run cl_mx_a0 2p-cmem-large.yaml mxfp4 0 &
  run cl_mx_a1048576 2p-cmem-large.yaml mxfp4 1048576 &
  run s2m_mx_a1048576 2p-smem2m.yaml mxfp4 1048576 &
  run s512_mx_a1048576 2p-smem512.yaml mxfp4 1048576 &
  wait
}

case "${1:-}" in
  record) run_all "$2" ;;
  check)
    run_all "$3"
    python3 "$(dirname "$0")/golden_diff.py" "$2" "$3"
    ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
