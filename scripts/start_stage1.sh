#!/usr/bin/env bash
# Register the physical profile first, then launch RLinf's unchanged SFT runner.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${RLINF_ROOT:?set RLINF_ROOT to the isolated, pinned RLinf checkout}"
: "${PI05_DATASET_ROOT:?set PI05_DATASET_ROOT to the real export containing train/ and norm_stats.json}"
: "${PI05_BASE_CHECKPOINT:?set PI05_BASE_CHECKPOINT to compatible pi05 base weights}"
export EMBODIED_PATH="${RLINF_ROOT}/examples/embodiment"
export PYTHONPATH="${REPO_ROOT}:${RLINF_ROOT}:${PYTHONPATH:-}"
python -m pi05_rlt_rtc.adapters.upstream_profile --rlinf-root "$RLINF_ROOT"
exec python "$RLINF_ROOT/examples/sft/train_vla_sft.py" \
  --config-path "$REPO_ROOT/configs" --config-name stage1_panda "$@"
