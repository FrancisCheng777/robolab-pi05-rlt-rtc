#!/usr/bin/env bash
set -euo pipefail
PI05_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PI05_PROJECT_ROOT
export PYTHONPATH="$(dirname "$PI05_PROJECT_ROOT"):$PI05_PROJECT_ROOT:$PI05_PROJECT_ROOT/vendor/rlinf_stage1${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export RLINF_NODE_RANK=0
PI05_TRAIN_PYTHON="${PI05_TRAIN_PYTHON:-$PI05_PROJECT_ROOT/.venv-stage1/bin/python}"
exec "$PI05_TRAIN_PYTHON" -m stage1.launch "$@"
