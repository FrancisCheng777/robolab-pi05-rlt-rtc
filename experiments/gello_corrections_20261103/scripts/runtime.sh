#!/usr/bin/env bash
# Planner/controller CLI only; invoking this wrapper does not start a service.
set -euo pipefail
PI05_RUNTIME_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PI05_RUNTIME_ROOT:$PI05_RUNTIME_ROOT/stage1_integration${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec "${PI05_RUNTIME_PYTHON:-$PI05_RUNTIME_ROOT/.venv-runtime/bin/python}" -m pi05_rlt_rtc.tools.cli "$@"
