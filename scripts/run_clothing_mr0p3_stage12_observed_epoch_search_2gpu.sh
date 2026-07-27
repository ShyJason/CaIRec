#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="${DATASET:-clothing}"
EXP_MODE="${EXP_MODE:-mm}"
MISSING_RATE="${MISSING_RATE:-0.3}"
EVAL_MISSING_RATE="${EVAL_MISSING_RATE:-0.5}"
SEED="${SEED:-2023}"
DATASET_SEED="${DATASET_SEED:-0}"
GPUS="${GPUS:-3 4}"
RUN_TAG="${RUN_TAG:-clothing_mr0p3_stage12_observed_epoch_search_$(date +%Y%m%d_%H%M%S)}"
BASE_DIR="${BASE_DIR:-exp_report/clothing/stage12_observed_epoch_search/${RUN_TAG}}"

STAGE11_EPOCHS="${STAGE11_EPOCHS:-5}"
STAGE11_CANDIDATE_EPOCHS="${STAGE11_CANDIDATE_EPOCHS:-0 1 2 3 4}"
STAGE12_EPOCHS="${STAGE12_EPOCHS:-50}"
STAGE12_CANDIDATE_EPOCHS="${STAGE12_CANDIDATE_EPOCHS:-0 5 10 20 35 49}"

STAGE2_CONFIG="${STAGE2_CONFIG:-configs/clothing/mainline_mr0p1.yaml}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-200}"
STAGE2_EARLY_STOP="${STAGE2_EARLY_STOP:-20}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-2048}"

STAGE11_SUFFIX="${STAGE11_SUFFIX:-stage1_1_clothing_mr0p3_observed_epoch_search_${RUN_TAG}}"
STAGE12_SUFFIX_PREFIX="${STAGE12_SUFFIX_PREFIX:-stage1_2_clothing_mr0p3_observed_epoch_search_${RUN_TAG}}"
STAGE2_SUFFIX_PREFIX="${STAGE2_SUFFIX_PREFIX:-stage2_clothing_mr0p3_stage12_observed_epoch_search_${RUN_TAG}}"

LOG_DIR="${BASE_DIR}/logs"
STATE_DIR="${BASE_DIR}/state"
STAGE12_TASKS="${BASE_DIR}/stage12_tasks.tsv"
STAGE2_CANDIDATES="${BASE_DIR}/stage2_candidates.tsv"
SUMMARY_FILE="${BASE_DIR}/summary.tsv"
SUMMARY_TOPK_FILE="${BASE_DIR}/summary_topk.tsv"
LAUNCHER_LOG="${BASE_DIR}/launcher.log"
mkdir -p "${LOG_DIR}" "${STATE_DIR}"

log() {
  echo "[$(date -Is)] $*" | tee -a "${LAUNCHER_LOG}"
}

first_gpu() {
  for gpu in ${GPUS}; do
    echo "${gpu}"
    return 0
  done
}

stage11_ckpt() {
  local epoch="$1"
  find "exp_report/${DATASET}/${STAGE11_SUFFIX}/ckpt" -maxdepth 1 -type f \
    -name "${STAGE11_SUFFIX}_imputer_param_*_epoch${epoch}.pth" 2>/dev/null \
    | sort | tail -1 || true
}

stage12_suffix() {
  local stage11_epoch="$1"
  printf "%s_s11e%s" "${STAGE12_SUFFIX_PREFIX}" "${stage11_epoch}"
}

stage12_ckpt() {
  local stage11_epoch="$1"
  local stage12_epoch="$2"
  local suffix
  suffix="$(stage12_suffix "${stage11_epoch}")"
  find "exp_report/${DATASET}/${suffix}/ckpt" -maxdepth 1 -type f \
    -name "${suffix}_imputer_backprop_*_epoch${stage12_epoch}.pth" 2>/dev/null \
    | sort | tail -1 || true
}

train_stage11() {
  local missing=0
  local ckpt
  for epoch in ${STAGE11_CANDIDATE_EPOCHS}; do
    ckpt="$(stage11_ckpt "${epoch}")"
    if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
      missing=1
      break
    fi
  done

  if [[ "${missing}" == "0" ]]; then
    log "Stage1.1 checkpoints already exist: ${STAGE11_SUFFIX}"
    return 0
  fi

  local gpu
  gpu="$(first_gpu)"
  log "Stage1.1 start on GPU ${gpu}: suffix=${STAGE11_SUFFIX}, epochs=${STAGE11_EPOCHS}"
  DATASET="${DATASET}" \
  EXP_MODE="${EXP_MODE}" \
  MISSING_RATE="${MISSING_RATE}" \
  CONFIG="configs/${DATASET}/stage1_1_imputer_param.yaml" \
  DEVICE_ID="${gpu}" \
  SUFFIX="${STAGE11_SUFFIX}" \
  EPOCHS="${STAGE11_EPOCHS}" \
  SAVE=1 \
  SAVE_ALL_EPOCHS=1 \
  TENSORBOARD=0 \
  ./run_stage1_1_baby_imputer_param.sh \
    --save_all_epochs 1 \
    --tensorboard 0 \
    --seed "${SEED}" \
    --dataset_seed "${DATASET_SEED}" \
    2>&1 | tee -a "${LOG_DIR}/${STAGE11_SUFFIX}.log"

  for epoch in ${STAGE11_CANDIDATE_EPOCHS}; do
    ckpt="$(stage11_ckpt "${epoch}")"
    if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
      echo "missing Stage1.1 checkpoint for epoch ${epoch}" >&2
      exit 1
    fi
  done
  log "Stage1.1 complete"
}

write_stage12_tasks() {
  {
    echo -e "stage11_epoch\tsuffix"
    for epoch in ${STAGE11_CANDIDATE_EPOCHS}; do
      echo -e "${epoch}\t$(stage12_suffix "${epoch}")"
    done
  } > "${STAGE12_TASKS}"
}

stage12_worker() {
  local gpu="$1"
  local next_file="${STATE_DIR}/stage12_next"
  local lock_file="${STATE_DIR}/stage12.lock"
  touch "${next_file}"
  while true; do
    local index line stage11_epoch suffix ckpt final_ckpt
    index="$(
      flock "${lock_file}" bash -c '
        next="$(cat "$1" 2>/dev/null || true)"
        next="${next:-1}"
        echo "$((next + 1))" > "$1"
        echo "$next"
      ' _ "${next_file}"
    )"
    line="$(awk -F '\t' -v n="$((index + 1))" 'NR == n { print }' "${STAGE12_TASKS}")"
    [[ -n "${line}" ]] || break
    stage11_epoch="$(cut -f1 <<<"${line}")"
    suffix="$(cut -f2 <<<"${line}")"
    final_ckpt="$(stage12_ckpt "${stage11_epoch}" "$((STAGE12_EPOCHS - 1))")"
    if [[ -n "${final_ckpt}" && -f "${final_ckpt}" ]]; then
      log "Stage1.2 skip existing: s11e${stage11_epoch}"
      continue
    fi
    ckpt="$(stage11_ckpt "${stage11_epoch}")"
    if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
      echo "missing Stage1.1 ckpt for epoch ${stage11_epoch}" >&2
      exit 1
    fi

    log "Stage1.2 observed start on GPU ${gpu}: s11e${stage11_epoch}, suffix=${suffix}"
    DATASET="${DATASET}" \
    EXP_MODE="${EXP_MODE}" \
    MISSING_RATE="${MISSING_RATE}" \
    CONFIG="configs/${DATASET}/stage1_2_decoder_v2.yaml" \
    DEVICE_ID="${gpu}" \
    SUFFIX="${suffix}" \
    IMPUTER_CKPT="${ckpt}" \
    STAGE1_2_MODE=observed \
    EPOCHS="${STAGE12_EPOCHS}" \
    SAVE=1 \
    SAVE_ALL_EPOCHS=1 \
    TENSORBOARD=0 \
    ./run_stage1_2_baby_imputer_backprop_decoder_v2.sh \
      --save_all_epochs 1 \
      --tensorboard 0 \
      --seed "${SEED}" \
      --dataset_seed "${DATASET_SEED}" \
      > "${LOG_DIR}/${suffix}.log" 2>&1
    log "Stage1.2 observed done: s11e${stage11_epoch}"
  done
}

train_stage12_all() {
  write_stage12_tasks
  : > "${STATE_DIR}/stage12_next"
  log "Stage1.2 observed queue start on GPUs: ${GPUS}"
  for gpu in ${GPUS}; do
    stage12_worker "${gpu}" &
  done
  wait

  for stage11_epoch in ${STAGE11_CANDIDATE_EPOCHS}; do
    for stage12_epoch in ${STAGE12_CANDIDATE_EPOCHS}; do
      local ckpt
      ckpt="$(stage12_ckpt "${stage11_epoch}" "${stage12_epoch}")"
      if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
        echo "missing Stage1.2 ckpt for s11e${stage11_epoch}, s12e${stage12_epoch}" >&2
        exit 1
      fi
    done
  done
  log "Stage1.2 observed checkpoints ready"
}

write_stage2_candidates() {
  {
    echo -e "tag\tstage11_epoch\tstage12_epoch\timputer_ckpt"
    for stage11_epoch in ${STAGE11_CANDIDATE_EPOCHS}; do
      for stage12_epoch in ${STAGE12_CANDIDATE_EPOCHS}; do
        local tag ckpt
        tag="s11e${stage11_epoch}_s12e${stage12_epoch}"
        ckpt="$(stage12_ckpt "${stage11_epoch}" "${stage12_epoch}")"
        echo -e "${tag}\t${stage11_epoch}\t${stage12_epoch}\t${ckpt}"
      done
    done
  } > "${STAGE2_CANDIDATES}"
}

stage2_worker() {
  local gpu="$1"
  local next_file="${STATE_DIR}/stage2_next"
  local lock_file="${STATE_DIR}/stage2.lock"
  touch "${next_file}"
  while true; do
    local index line tag stage11_epoch stage12_epoch ckpt suffix log_file
    index="$(
      flock "${lock_file}" bash -c '
        next="$(cat "$1" 2>/dev/null || true)"
        next="${next:-1}"
        echo "$((next + 1))" > "$1"
        echo "$next"
      ' _ "${next_file}"
    )"
    line="$(awk -F '\t' -v n="$((index + 1))" 'NR == n { print }' "${STAGE2_CANDIDATES}")"
    [[ -n "${line}" ]] || break
    tag="$(cut -f1 <<<"${line}")"
    stage11_epoch="$(cut -f2 <<<"${line}")"
    stage12_epoch="$(cut -f3 <<<"${line}")"
    ckpt="$(cut -f4 <<<"${line}")"
    suffix="${STAGE2_SUFFIX_PREFIX}_${tag}"
    log_file="${LOG_DIR}/${suffix}.log"
    if [[ -f "${log_file}" ]] && grep -q "final strict test hr@20" "${log_file}"; then
      log "Stage2 skip existing: ${tag}"
      continue
    fi
    if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
      echo "missing Stage1.2 ckpt for ${tag}: ${ckpt}" >&2
      exit 1
    fi

    log "Stage2 start on GPU ${gpu}: ${tag} (s11=${stage11_epoch}, s12=${stage12_epoch})"
    PYTHONUNBUFFERED=1 .venv/bin/python -u main.py \
      --config "${STAGE2_CONFIG}" \
      --device_id "${gpu}" \
      --dataset "${DATASET}" \
      --exp_mode "${EXP_MODE}" \
      --train_stage recommender \
      --missing_rate "${MISSING_RATE}" \
      --eval_missing_rate "${EVAL_MISSING_RATE}" \
      --seed "${SEED}" \
      --dataset_seed "${DATASET_SEED}" \
      --imputer_ckpt "${ckpt}" \
      --suffix "${suffix}" \
      --epoch "${STAGE2_EPOCHS}" \
      --early_stop "${STAGE2_EARLY_STOP}" \
      --eva_interval 1 \
      --batch_size "${STAGE2_BATCH_SIZE}" \
      --lr 0.01 \
      --lr_rec 0.01 \
      --lr_imp 0.0002 \
      --lr_decoder 0.00005 \
      --freeze_imputer 1 \
      --freeze_decoder 1 \
      --recommender_allow_modal_grad 0 \
      --feature_bridge_mode raw_decoder \
      --gcn_frontend_mode original_linear \
      --disable_imputation 0 \
      --modality_bpr_coeff 1.0 \
      --reg_coeff 0.01 \
      --evaluation_protocol strict \
      --selection_mode val \
      --strict_probe_test_interval 0 \
      --recommendation_selection_metric recall \
      --recommendation_selection_topk 20 \
      --rec_neighbor_cl_weight 0.01 \
      --rec_neighbor_cl_temp 0.2 \
      --rec_neighbor_cl_bank_size 256 \
      --item_graph_kind fused_completed \
      --item_graph_topk 8 \
      --item_graph_norm rw \
      --item_graph_cf_weight 0.25 \
      --item_graph_image_weight 0.375 \
      --item_graph_text_weight 0.375 \
      --item_graph_audio_weight 0.0 \
      --item_graph_modal_alpha 0.25 \
      --item_graph_modal_layers 1 \
      --item_graph_modal_target all \
      --tensorboard 0 \
      --save 1 \
      --topk "[10, 20, 30, 40, 50]" \
      > "${log_file}" 2>&1
    log "Stage2 done: ${tag}"
  done
}

summarize() {
  .venv/bin/python - "${LOG_DIR}" "${STAGE2_CANDIDATES}" "${SUMMARY_FILE}" "${SUMMARY_TOPK_FILE}" <<'PY'
import csv
import re
import sys
from pathlib import Path

log_dir, candidates_file, summary_file, summary_topk_file = map(Path, sys.argv[1:])

with candidates_file.open(newline="") as f:
    candidates = {row["tag"]: row for row in csv.DictReader(f, delimiter="\t")}

def final_at_k(text, k):
    vals = re.findall(
        rf"final strict test hr@{k}\s*=\s*([0-9.]+),\s*recall@{k}\s*=\s*([0-9.]+),\s*ndcg@{k}\s*=\s*([0-9.]+)",
        text,
    )
    return tuple(map(float, vals[-1])) if vals else None

def best_epoch(text):
    vals = re.findall(r"best epoch\s+([0-9]+)", text)
    return int(vals[-1]) if vals else -1

rows = []
topk_rows = []
for tag, cand in candidates.items():
    logs = sorted(log_dir.glob(f"*_{tag}.log"))
    if not logs:
        continue
    text = logs[-1].read_text(errors="ignore")
    if "final strict test hr@20" not in text:
        continue
    f20 = final_at_k(text, 20)
    if not f20:
        continue
    _, recall20, ndcg20 = f20
    row = {
        "tag": tag,
        "stage11_epoch": cand["stage11_epoch"],
        "stage12_epoch": cand["stage12_epoch"],
        "recall20": f"{recall20:.5f}",
        "ndcg20": f"{ndcg20:.5f}",
        "best_stage2_epoch": str(best_epoch(text)),
        "log": logs[-1].name,
    }
    rows.append(row)
    top = dict(row)
    for k in (10, 20, 30, 40, 50):
        vals = final_at_k(text, k)
        if vals:
            hr, rec, ndcg = vals
            top[f"hr@{k}"] = f"{hr:.5f}"
            top[f"recall@{k}"] = f"{rec:.5f}"
            top[f"ndcg@{k}"] = f"{ndcg:.5f}"
    topk_rows.append(top)

rows.sort(key=lambda r: (float(r["recall20"]), float(r["ndcg20"])), reverse=True)
topk_rows.sort(key=lambda r: (float(r["recall20"]), float(r["ndcg20"])), reverse=True)

with summary_file.open("w", newline="") as f:
    fields = ["tag", "stage11_epoch", "stage12_epoch", "recall20", "ndcg20", "best_stage2_epoch", "log"]
    w = csv.DictWriter(f, delimiter="\t", fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)

with summary_topk_file.open("w", newline="") as f:
    fields = [
        "tag", "stage11_epoch", "stage12_epoch",
        "hr@10", "recall@10", "ndcg@10",
        "hr@20", "recall@20", "ndcg@20",
        "hr@30", "recall@30", "ndcg@30",
        "hr@40", "recall@40", "ndcg@40",
        "hr@50", "recall@50", "ndcg@50",
        "best_stage2_epoch", "log",
    ]
    w = csv.DictWriter(f, delimiter="\t", fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    w.writerows(topk_rows)
PY
}

run_stage2_all() {
  write_stage2_candidates
  : > "${STATE_DIR}/stage2_next"
  log "Stage2 queue start on GPUs: ${GPUS}"
  for gpu in ${GPUS}; do
    stage2_worker "${gpu}" &
  done
  wait
  summarize
  log "Stage2 queue complete; summary=${SUMMARY_FILE}"
}

log "run_tag=${RUN_TAG}"
log "base_dir=${BASE_DIR}"
log "stage11_epochs=${STAGE11_CANDIDATE_EPOCHS}"
log "stage12_epochs=${STAGE12_CANDIDATE_EPOCHS}"
train_stage11
train_stage12_all
run_stage2_all
