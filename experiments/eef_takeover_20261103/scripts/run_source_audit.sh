#!/usr/bin/env bash
# Clone the pinned RLinf commit into a scratch dir (if needed) and audit it.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
RLINF_DIR="${RLINF_DIR:-/tmp/pi05-rlinf-audit}"
PINNED=c70606f08cdca259b8dec03d4430926b5b8fac9d
if [ ! -d "$RLINF_DIR/.git" ]; then
  mkdir -p "$RLINF_DIR"
  git -C "$RLINF_DIR" init -q
  git -C "$RLINF_DIR" remote add origin https://github.com/RLinf/RLinf.git 2>/dev/null || true
  git -C "$RLINF_DIR" fetch --depth 1 origin "$PINNED"
  git -C "$RLINF_DIR" checkout -q FETCH_HEAD
fi
"$PYTHON" -m pi05_rlt_rtc.adapters.source_audit \
  --rlinf "$RLINF_DIR" \
  --out runs/p0_audit/rlinf_source_audit.json \
  --environment runs/p0_audit/environment_versions.json
