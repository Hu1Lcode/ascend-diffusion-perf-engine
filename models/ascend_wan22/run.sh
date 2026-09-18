#!/usr/bin/env bash
set -euo pipefail
: "${OUT_DIR:?Sol-Engine launcher must set OUT_DIR}"
: "${AUTOVIDEO_REPO_ROOT:?Sol-Engine launcher must set AUTOVIDEO_REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${AUTOVIDEO_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m ascend_engine.run_manifest 2>&1 | tee "${OUT_DIR}/run.log"
exit "${PIPESTATUS[0]}"
