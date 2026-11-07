#!/usr/bin/env bash
# Extend a provisioned pinned OpenPI environment without modifying that environment.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${OPENPI_PYTHON:?set OPENPI_PYTHON to Python 3.11 in a provisioned OpenPI environment (commit 215abfb2)}"
"$OPENPI_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,11); import openpi.transforms'
"$OPENPI_PYTHON" -m venv --system-site-packages "$TASK_ROOT/.venv-runtime"
TASK_PYTHON="$TASK_ROOT/.venv-runtime/bin/python"
"$TASK_PYTHON" -m pip install --upgrade torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128
"$TASK_PYTHON" -m pip install 'numpy==1.26.4' 'transformers==4.53.2' 'sentencepiece==0.2.2' \
 'omegaconf==2.3.0' 'hydra-core==1.3.2' 'ray==2.47.1' 'h5py>=3.10' 'pytest>=7.4' \
 'pyrealsense2==2.59.1.11053' 'torchdata==0.11.0' einops gymnasium scipy sortedcontainers icmplib lz4 zstandard
"$TASK_PYTHON" -m pip install --no-deps -e "$TASK_ROOT"
JAX_PLATFORMS=cpu "$TASK_PYTHON" "$TASK_ROOT/scripts/validate_cuda.py"
