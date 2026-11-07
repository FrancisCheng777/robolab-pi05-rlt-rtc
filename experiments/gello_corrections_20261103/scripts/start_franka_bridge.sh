#!/usr/bin/env bash
# Explicit opt-in; always uses the pinned RoboLab bridge and accepted profile.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TASK_CONFIG="$TASK_ROOT/vendor/robolab-collecting-pipeline/configs/latency/responsive_recording/bridge.trial.json"
TASK_SOCKET="${PI05_BRIDGE_SOCKET:-/tmp/robolab-rlt.sock}"
# Interactive ROS shells may expose an ABI-incompatible libruckig.so.
# Scope the matching build libraries to this launcher, ahead of inherited paths.
TASK_BUILD="$TASK_ROOT/build/bridge-real"
if [[ "${1:-}" == --mock ]]; then TASK_BUILD="$TASK_ROOT/build/bridge-mock"; fi
export LD_LIBRARY_PATH="$TASK_BUILD/_deps/ruckig-build:$TASK_BUILD/robotiq_sdk${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ "${1:-}" == --execute || "${1:-}" == --mock ]]; then
 export PYTHONPATH="$TASK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
 "${PI05_RUNTIME_PYTHON:-python3}" -m pi05_rlt_rtc.deployment.bridge_preflight prepare "$TASK_SOCKET"
fi
case "${1:-}" in
 --check-libraries)
  TASK_LINK_REPORT="$(ldd -r "$TASK_BUILD/pi05_franka_bridge" 2>&1)"
  printf '%s\n' "$TASK_LINK_REPORT"
  if [[ "$TASK_LINK_REPORT" == *"undefined symbol:"* || "$TASK_LINK_REPORT" == *"not found"* ]]; then exit 1; fi
  ;;
 --mock) exec "$TASK_ROOT/build/bridge-mock/pi05_franka_bridge" --config "$TASK_CONFIG" --socket "$TASK_SOCKET" --mock;;
 --execute) exec "$TASK_ROOT/build/bridge-real/pi05_franka_bridge" --config "$TASK_CONFIG" --socket "$TASK_SOCKET" --robot "${PI05_ROBOT_IP:-172.16.0.40}";;
 *) echo 'Usage: start_franka_bridge.sh --mock | --execute | --check-libraries (default: no device opened)'; exit 2;;
esac
