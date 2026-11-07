#!/usr/bin/env bash
set -euo pipefail
ROOT='/home/heinrich/robolab-pi05-rlinf-core-20261103'
OUT='/media/heinrich/SANDISK SSD/pi05-stage2/plug_rlt_afixed_20261103/comparisons/rlinf_core_20261103'
if [[ ! -f "$OUT/status.json" ]] || ! "$ROOT/.venv-runtime/bin/python" - "$OUT/status.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get('state')=='complete' else 1)
PY
then
  echo 'RLinf-core 对照权重尚未训练并验证完成。'
  exit 2
fi
export PI05_STAGE2_ROOT="$ROOT"
export PI05_STAGE2_SESSION="$OUT/evaluation"
echo 'RLinf-core 离线对照；成功 +1 / 失败 -1，gamma=0.96。r 开始，b 后新 Actor 接管。'
exec bash "$ROOT/scripts/run_pi05_stage2.sh" eval \
  --runtime-config "$OUT/runtime_rlinf_core.yaml" --policy "$OUT/latest/policy.pt" --execute "$@"
