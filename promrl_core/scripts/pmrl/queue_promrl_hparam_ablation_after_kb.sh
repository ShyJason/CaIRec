#!/usr/bin/env bash
set -euo pipefail

WAIT_PATTERN="${WAIT_PATTERN:-retrieval_all_zeroshot_variants_20260330_015204_promrl_kb}"
SLEEP_SECONDS="${SLEEP_SECONDS:-120}"

while pgrep -f "${WAIT_PATTERN}" > /dev/null; do
    echo "[queue] waiting for ${WAIT_PATTERN} to finish..."
    sleep "${SLEEP_SECONDS}"
done

echo "[queue] start promrl hparam ablation"
bash scripts/pmrl/run_promrl_hparam_ablation.sh
