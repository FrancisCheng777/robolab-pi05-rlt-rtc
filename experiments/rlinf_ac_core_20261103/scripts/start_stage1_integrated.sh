#!/usr/bin/env bash
# Use the isolated Stage-1 snapshot and the common physical-window package.
set -euo pipefail
PI05_RUNTIME_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$PI05_RUNTIME_ROOT/stage1_integration/scripts/stage1.sh" "$@"
