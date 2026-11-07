#!/usr/bin/env bash
# Restore the existing collection station's temporary settings. No robot motion.
set -euo pipefail
BASE=/home/heinrich/robolab-pi05-rlt-rtc/vendor/robolab-collecting-pipeline
python3 - <<'PY'
from pathlib import Path
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:argv=(proc/'cmdline').read_bytes().split(b'\0')
 except OSError:continue
 if argv and Path(argv[0].decode(errors='replace')).name=='pi05_franka_bridge':
  raise SystemExit('请先在推理窗口按 q，确认 bridge 已退出，再执行本脚本。')
PY
sudo -v
sudo /usr/bin/python3 "$BASE/scripts/fix_fci_network_priority.py" --apply
sudo /usr/bin/python3 "$BASE/scripts/set_teleop_usb_latency.py" --apply
printf '%s\n' '站点设置已恢复。没有连接机械臂或发送运动命令；接下来仍需操作员验证回位/通信稳定性。'
