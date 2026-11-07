#!/usr/bin/env bash
# Unified data entrypoint; real collection remains explicitly controlled by its CLI.
set -euo pipefail
PI05_DATA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PI05_DATA_ROOT:$PI05_DATA_ROOT/vendor/robolab-collecting-pipeline${PYTHONPATH:+:$PYTHONPATH}"
case "${1:---help}" in
  collect|review|export|preflight)
    PI05_DATA_COMMAND="$1"
    shift
    exec "${PI05_DATA_PYTHON:-python3}" -m "pi05_demo.$PI05_DATA_COMMAND" "$@"
    ;;
  *) echo "Usage: PI05_DATA_PYTHON=/path/to/python $0 {collect|review|export|preflight} [arguments]" ;;
esac
