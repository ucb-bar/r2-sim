#!/bin/bash
# Golden check: runs the RV image rows below (the encoder on
# configs/2p.yaml, the others on configs/2p-smem.yaml) and compares them
# with a reference run.
#
#   tools/golden.sh record DIR        # run the rows into DIR (the reference)
#   tools/golden.sh check REF DIR     # run into DIR and compare with REF
#
# Run from the repository root after a release build. R2SIM overrides the
# binary (default ${CARGO_TARGET_DIR:-target}/release/r2sim); GOLDEN_ARGS adds arguments to
# every run (for example GOLDEN_ARGS="--set global.endpoint_charging=false").
# The images are the test fixtures in crates/nu-rv/tests/fixtures/rv. REF
# is a directory that `record` wrote from a known-good build, with the
# same GOLDEN_ARGS.
# "check" exits with 1 when a summary line or a JSON file differs; the
# comparison leaves out the wall time and the functional worker's progress
# at each load (thread timing).
set -euo pipefail
R2SIM=${R2SIM:-${CARGO_TARGET_DIR:-target}/release/r2sim}
FIXTURES=crates/nu-rv/tests/fixtures/rv

run_all() {
  local out=$1
  mkdir -p "$out"
  run() { # name config image [arguments]
    local name=$1 config=$2 image=$3
    shift 3
    "$R2SIM" run --config "configs/$config.yaml" --program rv --image "$FIXTURES/$image.toml" \
      "$@" --stats "$out/$name.json" ${GOLDEN_ARGS:-} > "$out/$name.log" 2>&1
  }
  run encoder_a0 2p encoder --acc-buffer 0 &
  run encoder_a1048576 2p encoder --acc-buffer 1048576 &
  run three_level 2p-smem three-level &
  run hello 2p-smem hello &
  wait
}

case "${1:-}" in
  record) run_all "$2" ;;
  check)
    run_all "$3"
    python3 "$(dirname "$0")/golden_diff.py" "$2" "$3"
    ;;
  *) sed -n '2,17p' "$0"; exit 2 ;;
esac
