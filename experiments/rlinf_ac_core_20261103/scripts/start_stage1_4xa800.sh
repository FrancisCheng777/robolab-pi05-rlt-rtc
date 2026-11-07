#!/usr/bin/env bash
# Four-card real-world SFT profile. No robot processes are launched here.
set -euo pipefail
PI05_RUNTIME_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PI05_CONFIG_NAME=stage1_4xa800_realworld
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
if [[ "${1:-inspect}" == train ]]; then
  : "${PI05_DATASET:?Set PI05_DATASET to the complete reviewed LeRobot export}"
  : "${PI05_BASE_MODEL:?Set PI05_BASE_MODEL to the pinned pi05_base checkpoint}"
  : "${PI05_RUN_NAME:?Set a unique PI05_RUN_NAME (or explicitly resume)}"
fi
if [[ $# == 0 ]]; then set -- inspect; fi
exec bash "$PI05_RUNTIME_ROOT/scripts/start_stage1_integrated.sh" "$@"
