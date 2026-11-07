#!/usr/bin/env bash
set -euo pipefail
export PI05_STAGE2_ROOT='/home/heinrich/robolab-pi05-gello-corrections-20261103'
export PI05_STAGE2_SESSION='/home/heinrich/pi05-corrections-20261103'
exec bash "$PI05_STAGE2_ROOT/scripts/run_pi05_stage2.sh" eval \
 --runtime-config "$PI05_STAGE2_ROOT/configs/runtime_gello_corrections.yaml" --execute "$@"
