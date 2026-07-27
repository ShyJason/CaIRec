#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATASET="${DATASET:-clothing}"
EXP_MODE="${EXP_MODE:-mm}"
GPU="${GPU:-0}"
SEED="${SEED:-2023}"
DATASET_SEED="${DATASET_SEED:-0}"
MISSING_RATE="${MISSING_RATE:-0.3}"
EVAL_MISSING_RATE="${EVAL_MISSING_RATE:-0.5}"
STAGE12_EPOCHS="${STAGE12_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"
IMPUTATION_SELECTION_SPLIT="${IMPUTATION_SELECTION_SPLIT:-val}"
IMPUTATION_SELECTION_METRIC="${IMPUTATION_SELECTION_METRIC:-cosine}"

RUN_TAG="${RUN_TAG:-observed_stage1_core_mr0p3_$(date +%Y%m%d_%H%M%S)}"
BASE_DIR="${BASE_DIR:-exp_report/clothing/observed_stage1_core_search/${RUN_TAG}}"
LOG_DIR="${BASE_DIR}/logs"
METRIC_DIR="${BASE_DIR}/metrics"
SUMMARY_FILE="${BASE_DIR}/summary.tsv"

STAGE11_CKPT="${STAGE11_CKPT:-}"

mkdir -p "${LOG_DIR}" "${METRIC_DIR}"

if [[ ! -f "${STAGE11_CKPT}" ]]; then
  echo "missing STAGE11_CKPT; pass a non-projection Stage 1.1 checkpoint explicitly" >&2
  exit 1
fi

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "${BASE_DIR}/run.log"
}

# tag alpha_rec alpha_decode alpha_intra alpha_inter alpha_itm lr_imp lr_decoder
# alpha_rec is searched first because observed NLL is negative and dominates
# the current stage1.2 loss scale.
CANDIDATES=(
  "rec1_dec1_base 1.0 1.0 1.0 1.0 1.0 0.0005 0.0002"
  "rec0p5_dec1 0.5 1.0 1.0 1.0 1.0 0.0005 0.0002"
  "rec0p25_dec1 0.25 1.0 1.0 1.0 1.0 0.0005 0.0002"
  "rec0p5_dec2 0.5 2.0 1.0 1.0 1.0 0.0005 0.0002"
  "rec0p25_dec2 0.25 2.0 1.0 1.0 1.0 0.0005 0.0002"
  "rec0p5_dec1_noitm 0.5 1.0 1.0 1.0 0.0 0.0005 0.0002"
  "rec0p5_dec1_lrlow 0.5 1.0 1.0 1.0 1.0 0.0002 0.0001"
)

stage12_ckpt() {
  local suffix="$1"
  local epoch="$2"
  printf 'exp_report/%s/%s/ckpt/%s_imputer_backprop_%s_epoch%s.pth' \
    "${DATASET}" "${suffix}" "${suffix}" "${STAGE12_EPOCHS}" "${epoch}"
}

selected_epoch_from_log() {
  local log_file="$1" final_epoch="$2"
  .venv/bin/python - "${log_file}" "${final_epoch}" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
fallback = int(sys.argv[2])
current = None
best = None
for line in path.read_text(errors="ignore").splitlines():
    m = re.search(r"TRAIN:stage = imputer_backprop, epoch = ([0-9]+)/", line)
    if m:
        current = int(m.group(1))
    if "save ckpt (best" in line and current is not None:
        best = current
print(best if best is not None else fallback)
PY
}

evaluate_ckpt() {
  local metric_tag="$1" ckpt="$2"
  log "evaluate imputation ${metric_tag}"
  .venv/bin/python tools/evaluate_imputation_metrics.py \
    --config "configs/${DATASET}/stage1_2_decoder_v2.yaml" \
    --device_id "${GPU}" \
    --dataset "${DATASET}" \
    --exp_mode "${EXP_MODE}" \
    --seed "${SEED}" \
    --dataset_seed "${DATASET_SEED}" \
    --missing_rate "${MISSING_RATE}" \
    --eval_missing_rate "${EVAL_MISSING_RATE}" \
    --missing_mask_protocol i3 \
    --imputer_ckpt "${ckpt}" \
    --feature_bridge_mode raw_decoder \
    --gcn_frontend_mode original_linear \
    --stage1_profile v2 \
    --stage1_v2_loss_preset balanced \
    --metric_space both \
    --split both \
    --include_random_baseline 1 \
    --json_output "${METRIC_DIR}/${metric_tag}.json" \
    --save 0 \
    --tensorboard 0 \
    > "${LOG_DIR}/metric_${metric_tag}.log" 2>&1
}

run_stage12_candidate() {
  local tag="$1" alpha_rec="$2" alpha_decode="$3" alpha_intra="$4" alpha_inter="$5" alpha_itm="$6" lr_imp="$7" lr_decoder="$8"
  local suffix="stage1_2_clothing_mr0p3_observed_core_${tag}_${RUN_TAG}"
  local log_file="${LOG_DIR}/stage12_${tag}.log"
  local final_epoch=$((STAGE12_EPOCHS - 1))
  local final_ckpt
  final_ckpt="$(stage12_ckpt "${suffix}" "${final_epoch}")"

  if [[ -f "${final_ckpt}" && "${FORCE_RERUN:-0}" != "1" ]]; then
    log "skip existing stage1.2 ${tag}: ${final_ckpt}"
  else
    log "stage1.2 observed start ${tag}: alpha_rec=${alpha_rec}, alpha_decode=${alpha_decode}, alpha_itm=${alpha_itm}, lr_imp=${lr_imp}, lr_decoder=${lr_decoder}"
    CONFIG="configs/${DATASET}/stage1_2_decoder_v2.yaml" \
    DATASET="${DATASET}" \
    EXP_MODE="${EXP_MODE}" \
    STAGE1_2_MODE=observed \
    EPOCHS="${STAGE12_EPOCHS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    LR="${lr_imp}" \
    LR_IMP="${lr_imp}" \
    LR_DECODER="${lr_decoder}" \
    ALPHA_REC="${alpha_rec}" \
    ALPHA_DECODE="${alpha_decode}" \
    ALPHA_INTRA="${alpha_intra}" \
    ALPHA_INTER="${alpha_inter}" \
    ALPHA_ITM="${alpha_itm}" \
    IMPUTATION_SELECTION_POLICY=stage1_default \
    IMPUTATION_SELECTION_SPLIT="${IMPUTATION_SELECTION_SPLIT}" \
    IMPUTATION_SELECTION_METRIC="${IMPUTATION_SELECTION_METRIC}" \
    SAVE_ALL_EPOCHS=1 \
    SUFFIX="${suffix}" \
    MISSING_RATE="${MISSING_RATE}" \
    IMPUTER_CKPT="${STAGE11_CKPT}" \
    ./run_stage1_2_baby_imputer_backprop_decoder_v2.sh \
      --imputer_ckpt "${STAGE11_CKPT}" \
      --device_id "${GPU}" \
      --seed "${SEED}" \
      --dataset_seed "${DATASET_SEED}" \
      --missing_mask_protocol i3 \
      --tensorboard 0 \
      > "${log_file}" 2>&1
    log "stage1.2 observed done ${tag}"
  fi

  if [[ ! -f "${final_ckpt}" ]]; then
    echo "missing final stage1.2 checkpoint for ${tag}: ${final_ckpt}" >&2
    exit 1
  fi

  local best_epoch best_ckpt
  best_epoch="$(selected_epoch_from_log "${log_file}" "${final_epoch}")"
  best_ckpt="$(stage12_ckpt "${suffix}" "${best_epoch}")"
  if [[ ! -f "${best_ckpt}" ]]; then
    echo "missing selected stage1.2 checkpoint for ${tag}: ${best_ckpt}" >&2
    exit 1
  fi

  evaluate_ckpt "${tag}_best_e${best_epoch}" "${best_ckpt}"
  if [[ "${best_epoch}" != "${final_epoch}" ]]; then
    evaluate_ckpt "${tag}_final_e${final_epoch}" "${final_ckpt}"
  fi
}

summarize() {
  .venv/bin/python - "${METRIC_DIR}" "${SUMMARY_FILE}" <<'PY'
import json
import sys
from pathlib import Path

metric_dir = Path(sys.argv[1])
summary_file = Path(sys.argv[2])

rows = []
for path in sorted(metric_dir.glob("*.json")):
    data = json.loads(path.read_text())
    row = {"tag": path.stem}
    for split in ("train", "test"):
        for space in ("shared", "decode"):
            overall = data.get(split, {}).get(space, {}).get("_overall", {})
            row[f"{split}_{space}_mse"] = overall.get("mse", "")
            row[f"{split}_{space}_cosine"] = overall.get("cosine", "")
    rows.append(row)

fields = [
    "tag",
    "train_shared_mse", "train_shared_cosine",
    "test_shared_mse", "test_shared_cosine",
    "train_decode_mse", "train_decode_cosine",
    "test_decode_mse", "test_decode_cosine",
]
with summary_file.open("w") as f:
    f.write("\t".join(fields) + "\n")
    for row in sorted(rows, key=lambda r: float(r.get("test_shared_cosine") or -1), reverse=True):
        f.write("\t".join(str(row.get(field, "")) for field in fields) + "\n")
print(summary_file)
PY
}

log "run_tag=${RUN_TAG}"
log "base_dir=${BASE_DIR}"
log "stage11_ckpt=${STAGE11_CKPT}"
for cand in "${CANDIDATES[@]}"; do
  read -r tag alpha_rec alpha_decode alpha_intra alpha_inter alpha_itm lr_imp lr_decoder <<<"${cand}"
  run_stage12_candidate "${tag}" "${alpha_rec}" "${alpha_decode}" "${alpha_intra}" "${alpha_inter}" "${alpha_itm}" "${lr_imp}" "${lr_decoder}"
  summarize | tee -a "${BASE_DIR}/run.log"
done
log "done. summary=${SUMMARY_FILE}"
