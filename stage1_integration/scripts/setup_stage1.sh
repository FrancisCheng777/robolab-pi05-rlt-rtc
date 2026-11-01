#!/usr/bin/env bash
# Run on the rented GPU machine, inside the transferred independent project.
set -euo pipefail
PI05_SETUP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PI05_BOOTSTRAP_PYTHON="${PI05_BOOTSTRAP_PYTHON:-python3}"
"$PI05_BOOTSTRAP_PYTHON" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10-3.12"'
if [[ ! -e "$PI05_SETUP_ROOT/.venv-stage1" ]]; then
  "$PI05_BOOTSTRAP_PYTHON" -m venv "$PI05_SETUP_ROOT/.venv-stage1"
fi
PI05_SETUP_PYTHON="$PI05_SETUP_ROOT/.venv-stage1/bin/python"
"$PI05_SETUP_PYTHON" -m pip install --upgrade pip
"$PI05_SETUP_PYTHON" -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
"$PI05_SETUP_PYTHON" -m pip install -r "$PI05_SETUP_ROOT/stage1/requirements.txt"
"$PI05_SETUP_PYTHON" -m pip check
PYTHONPATH="$(dirname "$PI05_SETUP_ROOT"):$PI05_SETUP_ROOT:$PI05_SETUP_ROOT/vendor/rlinf_stage1" \
  "$PI05_SETUP_PYTHON" -c 'from stage1.worker import PlugSftWorker; import torch; print(PlugSftWorker.__name__, torch.__version__, torch.version.cuda)'
"$PI05_SETUP_ROOT/scripts/stage1.sh" inspect
