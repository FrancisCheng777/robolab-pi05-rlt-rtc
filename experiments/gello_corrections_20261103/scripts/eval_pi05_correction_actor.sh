#!/usr/bin/env bash
set -euo pipefail
if (( $# < 1 )); then echo '用法: bash ~/eval_pi05_correction_actor.sh /完整路径/policy.pt'; exit 2; fi
CORRECTION_POLICY="$1"
shift
export PI05_STAGE2_ROOT='/home/heinrich/robolab-pi05-gello-corrections-20261103'
export PI05_STAGE2_SESSION='/home/heinrich/pi05-corrections-20261103/actor_evaluation'
exec bash "$PI05_STAGE2_ROOT/scripts/run_pi05_stage2.sh" eval \
 --runtime-config "$PI05_STAGE2_ROOT/configs/runtime_correction_actor_eval.yaml" \
 --policy "$CORRECTION_POLICY" --execute "$@"
