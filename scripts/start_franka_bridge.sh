#!/usr/bin/env bash
# Explicit opt-in; always uses the pinned RoboLab bridge and accepted profile.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TASK_CONFIG="$TASK_ROOT/vendor/robolab-collecting-pipeline/configs/latency/responsive_recording/bridge.trial.json"
TASK_SOCKET="${PI05_BRIDGE_SOCKET:-/tmp/robolab-rlt.sock}"
case "${1:-}" in
 --mock) exec "$TASK_ROOT/build/bridge-mock/pi05_franka_bridge" --config "$TASK_CONFIG" --socket "$TASK_SOCKET" --mock;;
 --execute) exec "$TASK_ROOT/build/bridge-real/pi05_franka_bridge" --config "$TASK_CONFIG" --socket "$TASK_SOCKET" --robot "${PI05_ROBOT_IP:-172.16.0.40}";;
 *) echo 'Usage: start_franka_bridge.sh --mock | --execute (default: no device opened)'; exit 2;;
esac
