#!/usr/bin/env bash
# Compatibility entry: use the same canonical Stage-1 config and isolated runtime.
set -euo pipefail
PI05_RUNTIME_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PI05_DATASET="${PI05_DATASET:-${PI05_DATASET_ROOT:-}}"
export PI05_BASE_MODEL="${PI05_BASE_MODEL:-${PI05_BASE_CHECKPOINT:-}}"
# Empty variables must not override the canonical config defaults.
[[ -n "$PI05_DATASET" ]] || unset PI05_DATASET
[[ -n "$PI05_BASE_MODEL" ]] || unset PI05_BASE_MODEL
if [[ $# == 0 ]]; then set -- inspect; fi
exec bash "$PI05_RUNTIME_ROOT/scripts/start_stage1_integrated.sh" "$@"
