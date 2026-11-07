#!/usr/bin/env bash
# Run the full CPU acceptance suite and regenerate test_status.json + tests/RESULTS.md.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
"$PYTHON" -m pi05_rlt_rtc.tools.test_ledger \
  --out test_status.json \
  --log runs/cpu_tests/pytest.log
