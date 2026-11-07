#!/usr/bin/env bash
set -euo pipefail
cd '/home/heinrich/robolab-pi05-gello-corrections-20261103'
export PYTHONPATH='/home/heinrich/robolab-pi05-gello-corrections-20261103:/home/heinrich/robolab-pi05-gello-corrections-20261103/stage1_integration'
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=''
exec .venv-runtime/bin/python -m pi05_rlt_rtc.deployment.correction_training \
 --runtime-config configs/runtime_gello_corrections.yaml --root '/home/heinrich/pi05-corrections-20261103' "$@"
