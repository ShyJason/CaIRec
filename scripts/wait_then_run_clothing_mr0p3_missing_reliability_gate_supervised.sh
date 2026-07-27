#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

WAIT_PATTERN="${WAIT_PATTERN:-missing_reliability_gate_weightedfusion_20260630_222727}"
CHECK_INTERVAL="${CHECK_INTERVAL:-120}"
RUN_TAG="${RUN_TAG:-missing_reliability_gate_supervised_afterwf_$(date +%Y%m%d_%H%M%S)}"

echo "$(date '+%F %T') waiting for python jobs matching: ${WAIT_PATTERN}"
while pgrep -af ".venv/bin/python -u main.py" | grep -F "${WAIT_PATTERN}" >/dev/null; do
  sleep "${CHECK_INTERVAL}"
done

echo "$(date '+%F %T') launching supervised reliability gate: ${RUN_TAG}"
GPUS="${GPUS:-0 7}" MAX_PARALLEL="${MAX_PARALLEL:-2}" RUN_TAG="${RUN_TAG}" \
  bash scripts/run_clothing_mr0p3_missing_reliability_gate_supervised.sh
rc=$?
echo "$(date '+%F %T') supervised reliability gate exited rc=${rc}"
exit "${rc}"
