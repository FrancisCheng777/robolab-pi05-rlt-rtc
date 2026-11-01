#!/usr/bin/env bash
# Start only an explicitly selected NEW bridge; never reconnect an old service.
set -euo pipefail
PI05_BRIDGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PI05_BRIDGE_SOCKET="${PI05_BRIDGE_SOCKET:-/tmp/pi05-rlt-rtc-integrated.sock}"
case "${1:-}" in
  --mock)
    exec "$PI05_BRIDGE_ROOT/build/mock/pi05_franka_bridge" --mock \
      --config "$PI05_BRIDGE_ROOT/stage1_integration/configs/bridge.mock.json" --socket "$PI05_BRIDGE_SOCKET"
    ;;
  --execute)
    : "${PI05_REAL_BRIDGE_BIN:?Set the separately compiled libfranka bridge binary}"
    : "${PI05_DATA_PYTHON:?Set the collection Python interpreter}"
    bash "$PI05_BRIDGE_ROOT/scripts/data_pipeline.sh" preflight --hardware-config
    exec "$PI05_REAL_BRIDGE_BIN" --config "$PI05_BRIDGE_ROOT/stage1_integration/configs/bridge.real.json" \
      --robot 172.16.0.40 --socket "$PI05_BRIDGE_SOCKET"
    ;;
  *) echo "Usage: $0 --mock | --execute"; exit 2 ;;
esac
