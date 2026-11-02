#!/usr/bin/env bash
set -euo pipefail
ROOT="${PI05_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="$ROOT/configs/runtime_franka_stage1_8000.yaml"
export PI05_RUNTIME_PYTHON="$ROOT/.venv-runtime/bin/python"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH="$ROOT:$ROOT/stage1_integration${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$ROOT"
case "${1:-check}" in
 stop-model|status)
  exec "$PI05_RUNTIME_PYTHON" -m pi05_rlt_rtc.deployment.managed_eval "$1" --runtime-config "$CONFIG";;
esac
"$PI05_RUNTIME_PYTHON" - "$CONFIG" <<'PY'
import json,sys
from pathlib import Path
from pi05_rlt_rtc.config.runtime_config import load_runtime_config
cfg=load_runtime_config(sys.argv[1])
report=Path('outputs/stage1_8000_offline/report.json')
if not report.exists():raise SystemExit('尚未完成权重传输/完整 RTC 验收，请等待准备完成。')
d=json.loads(report.read_text())
if not d.get('passed'):raise SystemExit('完整模型离线检查未通过，请先处理 outputs/stage1_8000_offline/report.json。')
if cfg.rtc_mode!='full_vjp' or cfg.raw['policy']['kind']!='teacher':raise SystemExit('必须使用 full_vjp RTC 和 Stage-1 teacher。')
if d['metadata']['runtime_contract']!=cfg.wire_contract():raise SystemExit('配置已变化，需要重新测量 RTC 延迟。')
if not d['modes']['full_vjp']['within_measured_budget']:raise SystemExit('完整 RTC 推理未达到当前动作队列的时限，尚不可启动连续控制。')
clock=d.get('wall_clock_recorded_inputs', {})
if clock.get('ticks_run', 0)<60 or any(clock.get(k, 1)!=0 for k in ('deadline_misses','rejected_plans','safety_stops','planner_errors','underruns')):
    raise SystemExit('真实模型动作队列离线验收未通过。')
print('8000-step Stage-1 teacher; inference-time full_vjp RTC; eval only; no Stage-2 updates.')
print('Execution:', cfg.raw['franka'].get('joint_command_mode', 'measured_delta'), '; H/C/D=', cfg.contract.horizon, cfg.contract.chunk, cfg.contract.delay, '; lead cap=', cfg.raw['franka'].get('reference_lead_rad'), 'rad')
PY
case "${1:-check}" in
 check)
  echo '启动：bash ~/run_pi05_stage1_8000.sh eval'
  echo 'r 开始；e 停止并回采集初始位姿/张开夹爪；x 保持；q 退出硬件会话'
  echo '模型独立常驻，stop-model 释放显存。首次预热/代码模型变化后需重新加载。'
  ;;
 bridge)
  exec bash scripts/start_franka_bridge.sh --execute
  ;;
 eval|prewarm)
  if [[ "$1" == eval ]]; then
   "$PI05_RUNTIME_PYTHON" - "$CONFIG" <<'VERIFY'
import json,sys
from pathlib import Path
from pi05_rlt_rtc.config.runtime_config import load_runtime_config
cfg=load_runtime_config(sys.argv[1])
p=Path('profile_output/grasp_diagnosis/resident_acceptance.json')
if not p.exists():raise SystemExit('常驻模型 RTC 验收尚未完成。')
r=json.loads(p.read_text())
if not r.get('passed') or r['runtime_contract']!=cfg.wire_contract():raise SystemExit('常驻 RTC 验收未通过或配置已变化。')
VERIFY
  fi
  TASK_BRIDGE_LOG="$HOME/.cache/pi05-stage1-8000/bridge.log"
  TASK_LOG_OFFSET=0
  if [[ -f "$TASK_BRIDGE_LOG" ]]; then TASK_LOG_OFFSET=$(wc -c < "$TASK_BRIDGE_LOG"); fi
  if "$PI05_RUNTIME_PYTHON" -m pi05_rlt_rtc.deployment.managed_eval "$1" --runtime-config "$CONFIG"; then
   exit 0
  else
   TASK_EXIT=$?
   if [[ "$1" == eval && -f "$TASK_BRIDGE_LOG" ]]; then
    "$PI05_RUNTIME_PYTHON" - "$TASK_BRIDGE_LOG" "$TASK_LOG_OFFSET" <<'FAILURE'
from pathlib import Path
import sys
p=Path(sys.argv[1]);offset=int(sys.argv[2]);size=p.stat().st_size
with p.open('rb') as stream:
    stream.seek(max(offset, size-8192))
    details=stream.read().decode(errors='replace').strip()
if details:
    print('本次 bridge 日志：', file=sys.stderr)
    print(details, file=sys.stderr)
if 'command not possible in the current mode' in details:
    print('Panda 当前模式拒绝初始化。请在 Franka Desk 检查并按提示恢复到 Idle 后再启动；程序没有自动清错或启动运动。', file=sys.stderr)
FAILURE
   fi
   exit "$TASK_EXIT"
  fi
  ;;
 *) echo 'Usage: bash ~/run_pi05_stage1_8000.sh [check|eval|prewarm|stop-model|status|bridge]'; exit 2;;
esac
