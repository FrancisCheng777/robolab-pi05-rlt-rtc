#!/usr/bin/env bash
set -euo pipefail
PI05_INTEGRATION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$PI05_INTEGRATION_ROOT/scripts/runtime.sh" "$@"
