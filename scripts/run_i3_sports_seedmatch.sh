#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PATH="${ROOT_DIR}/.venv/bin:${PATH}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

RUN_TAG="${RUN_TAG:-i3_sports_seedmatch_$(date +%Y%m%d_%H%M%S)}"
DEVICE_ID="${DEVICE_ID:-6}"
TARGET_R="${TARGET_R:-0.08301}"
TARGET_N="${TARGET_N:-0.03699}"
TOL_R="${TOL_R:-0.003}"
TOL_N="${TOL_N:-0.002}"
STOP_AFTER="${STOP_AFTER:-3}"
SEEDS_STR="${SEEDS:-2 3 4 5 6 7 8 9 10 11 21 22 33 42 77 101 202 333 777 999}"

read -r -a SEEDS_ARR <<< "${SEEDS_STR}"

OUT_DIR="${ROOT_DIR}/exp_report/sports/i3_seed_search/${RUN_TAG}"
mkdir -p "${OUT_DIR}/logs"

SUMMARY="${OUT_DIR}/summary.tsv"
RANKED="${OUT_DIR}/ranked.tsv"
QUEUE_LOG="${OUT_DIR}/queue.log"

printf "seed\tbest_epoch\tr20\tndcg20\tdiff_r\tdiff_ndcg\tclose\tlog\n" > "${SUMMARY}"

log() {
  date +"[seedmatch] %Y-%m-%d %H:%M:%S $*" | tee -a "${QUEUE_LOG}"
}

parse_one() {
  local seed="$1"
  local log_path="$2"
  python - "${log_path}" "${seed}" "${TARGET_R}" "${TARGET_N}" "${TOL_R}" "${TOL_N}" <<'PY'
import pathlib
import re
import sys

log_path = pathlib.Path(sys.argv[1])
seed = sys.argv[2]
target_r = float(sys.argv[3])
target_n = float(sys.argv[4])
tol_r = float(sys.argv[5])
tol_n = float(sys.argv[6])

ansi = re.compile(r"\x1b\[[0-9;]*m")
text = ansi.sub("", log_path.read_text(errors="ignore"))
best = re.findall(r"best epoch\s+(\d+)", text)
metrics = re.findall(
    r"hr@20\s*=\s*([0-9.]+),\s*recall@20\s*=\s*[0-9.]+,\s*ndcg@20\s*=\s*([0-9.]+)",
    text,
)
if not metrics:
    print(f"{seed}\tNA\tNA\tNA\tNA\tNA\t0\t{log_path}")
    raise SystemExit
r20, ndcg20 = map(float, metrics[-1])
diff_r = abs(r20 - target_r)
diff_n = abs(ndcg20 - target_n)
close = int(diff_r <= tol_r and diff_n <= tol_n)
best_epoch = best[-1] if best else "NA"
print(
    f"{seed}\t{best_epoch}\t{r20:.5f}\t{ndcg20:.5f}\t"
    f"{diff_r:.5f}\t{diff_n:.5f}\t{close}\t{log_path}"
)
PY
}

rank_summary() {
  python - "${SUMMARY}" "${RANKED}" "${TARGET_R}" "${TARGET_N}" <<'PY'
import csv
import sys

summary_path, ranked_path = sys.argv[1], sys.argv[2]
target_r, target_n = float(sys.argv[3]), float(sys.argv[4])

rows = []
with open(summary_path, newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["r20"] == "NA":
            continue
        score = abs(float(row["r20"]) - target_r) + 2 * abs(float(row["ndcg20"]) - target_n)
        row["score"] = f"{score:.5f}"
        rows.append(row)
rows.sort(key=lambda row: float(row["score"]))

with open(ranked_path, "w", newline="") as f:
    fields = ["seed", "best_epoch", "r20", "ndcg20", "diff_r", "diff_ndcg", "close", "score", "log"]
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
PY
}

close_count=0
log "run_tag=${RUN_TAG} gpu=${DEVICE_ID} target=${TARGET_R}/${TARGET_N} tol=${TOL_R}/${TOL_N}"
log "seeds=${SEEDS_STR}"

for seed in "${SEEDS_ARR[@]}"; do
  log_path="${OUT_DIR}/logs/seed${seed}.log"
  suffix="i3_sports_mm_mr0.3_seed${seed}_${RUN_TAG}"
  log "start seed=${seed}"

  ./run_i3clear.sh \
    --dataset sports \
    --exp_mode mm \
    --missing_rate 0.3 \
    --seed "${seed}" \
    --device_id "${DEVICE_ID}" \
    --use_gpu 1 \
    --epoch 200 \
    --eva_interval 10 \
    --early_stop 20 \
    --batch_size 2048 \
    --lr 1e-3 \
    --reg_coeff 1e-3 \
    --penalty_coeff 300 \
    --max_info_coeff 1e-3 \
    --min_info_coeff 1e-5 \
    --tensorboard 0 \
    --save 1 \
    --suffix "${suffix}" 2>&1 | tee "${log_path}"

  row="$(parse_one "${seed}" "${log_path}")"
  echo "${row}" >> "${SUMMARY}"
  log "result ${row}"
  rank_summary

  close="$(awk -F '\t' '{print $7}' <<< "${row}")"
  if [[ "${close}" == "1" ]]; then
    close_count=$((close_count + 1))
  fi
  if (( close_count >= STOP_AFTER )); then
    log "stop: found ${close_count} close seeds"
    break
  fi
done

log "done"
