#!/usr/bin/env bash
set -euo pipefail
PI05_EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PI05_EVAL_ROOT:$PI05_EVAL_ROOT/stage1_integration${PYTHONPATH:+:$PYTHONPATH}"
exec "${PI05_EVAL_PYTHON:-python}" -m stage1.evaluate "$@"
