#!/usr/bin/env bash
set -euo pipefail
ROOT='/home/heinrich/robolab-pi05-stage2-prep'
BASE='/media/heinrich/SANDISK SSD/pi05-stage2/plug_rlt_afixed_20261103'
export PI05_STAGE2_ROOT="$ROOT"
export PI05_STAGE2_SESSION="$BASE/comparisons/rlinf_core_20261103/baseline_round10_evaluation"
exec bash "$ROOT/scripts/run_pi05_stage2.sh" eval \
  --runtime-config "$BASE/runtime_afixed.yaml" \
  --policy "$BASE/rounds/round-000010/latest/policy.pt" --execute "$@"
