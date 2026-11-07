#!/usr/bin/env bash
# Build the pinned accepted bridge without changing its source/configuration.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TASK_DEP="$TASK_ROOT/vendor/robolab-collecting-pipeline"
TASK_MODE="${1:---mock}"
case "$TASK_MODE" in --mock) TASK_REAL=OFF; TASK_BUILD="$TASK_ROOT/build/bridge-mock";; --real) TASK_REAL=ON; TASK_BUILD="$TASK_ROOT/build/bridge-real";; *) echo 'Usage: build_bridge.sh --mock | --real'; exit 2;; esac
cmake -S "$TASK_DEP/cpp/bridge" -B "$TASK_BUILD" -DPI05_WITH_FRANKA="$TASK_REAL" -DPI05_FETCH_DEPS=ON \
  -DFETCHCONTENT_SOURCE_DIR_RUCKIG="$TASK_DEP/vendor/cppdeps/ruckig-0.15.3" \
  -DFETCHCONTENT_SOURCE_DIR_JSON="$TASK_DEP/vendor/cppdeps/json-3.11.3" -DCMAKE_BUILD_TYPE=Release
cmake --build "$TASK_BUILD" -j 4
cd "$TASK_BUILD"
ctest --output-on-failure
