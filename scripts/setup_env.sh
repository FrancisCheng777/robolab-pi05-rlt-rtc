#!/usr/bin/env bash
# Create the project virtualenv and install test dependencies.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[test,torch,yaml]"
echo "venv ready: $(python -c 'import torch,numpy; print("torch", torch.__version__, "numpy", numpy.__version__)')"
