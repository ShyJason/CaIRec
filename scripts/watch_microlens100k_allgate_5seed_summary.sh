#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

eval_dir="exp_report/fusion_norm_ablation/late_score_eval_microlens100k_allgate_score_5seed_rest_20260529"
log_path="exp_report/fusion_norm_ablation/microlens100k_allgate_5seed_summary_watcher.log"

while true; do
  ok=1
  for seed in 1 1234 12345; do
    eval_log="${eval_dir}/microlens100k_seed${seed}_test.log"
    if [[ ! -f "${eval_log}" ]] || ! grep -q '^1\.0000' "${eval_log}"; then
      ok=0
    fi
  done

  if [[ "${ok}" -eq 1 ]]; then
    .venv/bin/python scripts/summarize_microlens100k_allgate_5seed.py > "${log_path}" 2>&1
    break
  fi

  sleep 300
done
