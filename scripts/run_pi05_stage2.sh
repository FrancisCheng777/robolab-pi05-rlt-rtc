#!/usr/bin/env bash
set -euo pipefail
ROOT="${PI05_STAGE2_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="$ROOT:$ROOT/stage1_integration${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$ROOT"
MODE="${1:-check}"
if (( $# )); then shift; fi
exec "$ROOT/.venv-runtime/bin/python" -m pi05_rlt_rtc.deployment.stage2_session "$MODE" \
 --runtime-config "$ROOT/configs/runtime_franka_stage2.yaml" \
 --session-root "${PI05_STAGE2_SESSION:-/media/heinrich/SANDISK SSD/pi05-stage2/plug_rlt_20261102}" "$@"
