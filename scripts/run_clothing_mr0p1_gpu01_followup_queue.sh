#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="${STAMP:-clothing_mr0p1_hparam_20260624_094823}"
OUT_DIR="exp_report/clothing/hparam_search_mr0p1_${STAMP}"
LOG_DIR="${OUT_DIR}/logs"
STATE_DIR="${OUT_DIR}/state"
mkdir -p "${LOG_DIR}" "${STATE_DIR}"

NEXT_FILE="${STATE_DIR}/gpu01_followup_next_index"
LOCK_FILE="${STATE_DIR}/gpu01_followup.lock"
if [[ ! -f "${NEXT_FILE}" ]]; then
  echo 0 > "${NEXT_FILE}"
fi

GPUS="${GPUS:-0 1}"
MEM_FREE_THRESHOLD="${MEM_FREE_THRESHOLD:-1000}"
POLL_SECONDS="${POLL_SECONDS:-60}"

CONFIG="${CONFIG:-configs/clothing/stage2_decoder_mm_itemgraph_completed.yaml}"
STAGE12_DIR="${STAGE12_DIR:-exp_report/clothing/stage1_2_clothing_seed2023_completion_mm_v2_mmrec_clothing_mm_mr0p1_seed2023_20260623_004452}"
STAGE12_PREFIX="${STAGE12_PREFIX:-stage1_2_clothing_seed2023_completion_mm_v2_mmrec_clothing_mm_mr0p1_seed2023_20260623_004452}"

# tag ckpt_epoch modality_bpr_coeff modal_alpha reg_coeff lr_rec cf image text
CANDIDATES=(
  "e14_mbpr2p0_default 14 2.0 0.25 0.01 0.01 0.3 0.35 0.35"
  "e14_mbpr2p5_default 14 2.5 0.25 0.01 0.01 0.3 0.35 0.35"
  "e15_mbpr2p0_reg0p02 15 2.0 0.25 0.02 0.01 0.3 0.35 0.35"
  "e15_mbpr2p0_lrrec0p02 15 2.0 0.25 0.01 0.02 0.3 0.35 0.35"
  "e15_mbpr2p0_malpha0p5 15 2.0 0.5 0.01 0.01 0.3 0.35 0.35"
  "e15_mbpr2p0_malpha0p75 15 2.0 0.75 0.01 0.01 0.3 0.35 0.35"
  "e16_mbpr2p5_default 16 2.5 0.25 0.01 0.01 0.3 0.35 0.35"
  "e16_mbpr1p5_default 16 1.5 0.25 0.01 0.01 0.3 0.35 0.35"
)

is_gpu_free() {
  local gpu="$1"
  local used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')"
  [[ "${used}" =~ ^[0-9]+$ ]] && (( used < MEM_FREE_THRESHOLD ))
}

summarize_results() {
  python3 - "${LOG_DIR}" <<'PY'
import glob
import os
import re
import sys

rows = []
for path in glob.glob(os.path.join(sys.argv[1], "*.log")):
    text = open(path, "r", errors="ignore").read()
    vals = re.findall(
        r"final strict test hr@20 = ([0-9.]+), recall@20 = ([0-9.]+), ndcg@20 = ([0-9.]+)",
        text,
    )
    best = re.findall(r"best epoch ([0-9]+)", text)
    if vals:
        _, rec, ndcg = map(float, vals[-1])
        rows.append((rec, ndcg, int(best[-1]) if best else -1, os.path.basename(path)))
for rec, ndcg, best, name in sorted(rows, reverse=True):
    print(f"{rec:.5f}\tndcg20={ndcg:.5f}\tbest={best}\t{name}")
PY
}

claim_next_candidate() {
  (
    flock -x 9
    local idx tag ckpt_epoch mbpr modal_alpha reg lr_rec cf_w image_w text_w suffix log_path
    idx="$(cat "${NEXT_FILE}")"
    while (( idx < ${#CANDIDATES[@]} )); do
      read -r tag ckpt_epoch mbpr modal_alpha reg lr_rec cf_w image_w text_w <<<"${CANDIDATES[$idx]}"
      suffix="stage2_clothing_mr0p1_${tag}_gpu01_followup_${STAMP}"
      log_path="${LOG_DIR}/${suffix}.log"
      if [[ -f "${log_path}" ]] && grep -q "final strict test hr@20" "${log_path}"; then
        idx=$((idx + 1))
        echo "${idx}" > "${NEXT_FILE}"
        continue
      fi
      echo $((idx + 1)) > "${NEXT_FILE}"
      printf '%s\n' "${CANDIDATES[$idx]}"
      return 0
    done
    return 1
  ) 9>"${LOCK_FILE}"
}

run_candidate() {
  local gpu="$1"
  local tag="$2"
  local ckpt_epoch="$3"
  local mbpr="$4"
  local modal_alpha="$5"
  local reg="$6"
  local lr_rec="$7"
  local cf_w="$8"
  local image_w="$9"
  local text_w="${10}"
  local suffix="stage2_clothing_mr0p1_${tag}_gpu01_followup_${STAMP}"
  local log_path="${LOG_DIR}/${suffix}.log"
  local imputer_ckpt="${STAGE12_DIR}/ckpt/${STAGE12_PREFIX}_imputer_backprop_50_epoch${ckpt_epoch}.pth"

  if [[ ! -f "${imputer_ckpt}" ]]; then
    echo "missing imputer checkpoint: ${imputer_ckpt}" >&2
    exit 1
  fi

  echo "[$(date -Is)] launch ${tag} on GPU ${gpu}" | tee -a "${OUT_DIR}/gpu01_followup_launcher.log"
  .venv/bin/python -u main.py \
    --config "${CONFIG}" \
    --device_id "${gpu}" \
    --dataset clothing \
    --exp_mode mm \
    --train_stage recommender \
    --missing_rate 0.1 \
    --seed 2023 \
    --dataset_seed 0 \
    --imputer_ckpt "${imputer_ckpt}" \
    --suffix "${suffix}" \
    --epoch 200 \
    --early_stop 10000 \
    --eva_interval 1 \
    --batch_size 2048 \
    --lr 0.01 \
    --lr_rec "${lr_rec}" \
    --lr_imp 0.0002 \
    --lr_decoder 0.00005 \
    --freeze_imputer 1 \
    --freeze_decoder 1 \
    --recommender_allow_modal_grad 0 \
    --feature_bridge_mode raw_decoder \
    --gcn_frontend_mode original_linear \
    --disable_imputation 0 \
    --modality_bpr_coeff "${mbpr}" \
    --reg_coeff "${reg}" \
    --evaluation_protocol strict \
    --selection_mode val \
    --recommendation_selection_metric recall \
    --recommendation_selection_topk 20 \
    --rec_neighbor_cl_weight "${REC_NEIGHBOR_CL_WEIGHT:-0.005}" \
    --rec_neighbor_cl_temp "${REC_NEIGHBOR_CL_TEMP:-0.2}" \
    --rec_neighbor_cl_bank_size "${REC_NEIGHBOR_CL_BANK_SIZE:-256}" \
    --item_graph_kind fused_completed \
    --item_graph_topk 20 \
    --item_graph_norm rw \
    --item_graph_cf_weight "${cf_w}" \
    --item_graph_image_weight "${image_w}" \
    --item_graph_text_weight "${text_w}" \
    --item_graph_audio_weight 0.0 \
    --item_graph_modal_alpha "${modal_alpha}" \
    --item_graph_modal_layers 1 \
    --item_graph_modal_target all \
    --tensorboard 0 \
    --save 1 \
    --topk "[10, 20, 30, 40, 50]" \
    > "${log_path}" 2>&1
  echo "[$(date -Is)] done ${tag} on GPU ${gpu}" | tee -a "${OUT_DIR}/gpu01_followup_launcher.log"
}

worker_loop() {
  local gpu="$1"
  local cand tag ckpt_epoch mbpr modal_alpha reg lr_rec cf_w image_w text_w
  while true; do
    while ! is_gpu_free "${gpu}"; do
      echo "[$(date -Is)] gpu=${gpu} waiting for free GPU" | tee -a "${OUT_DIR}/gpu01_followup_launcher.log"
      sleep "${POLL_SECONDS}"
    done
    if ! cand="$(claim_next_candidate)"; then
      echo "[$(date -Is)] gpu=${gpu} no follow-up candidates left" | tee -a "${OUT_DIR}/gpu01_followup_launcher.log"
      return 0
    fi
    read -r tag ckpt_epoch mbpr modal_alpha reg lr_rec cf_w image_w text_w <<<"${cand}"
    run_candidate "${gpu}" "${tag}" "${ckpt_epoch}" "${mbpr}" "${modal_alpha}" "${reg}" "${lr_rec}" "${cf_w}" "${image_w}" "${text_w}"
    summarize_results | tee "${OUT_DIR}/summary.tsv"
  done
}

echo "[$(date -Is)] start gpu01 follow-up queue" | tee -a "${OUT_DIR}/gpu01_followup_launcher.log"
read -r -a GPU_LIST <<<"${GPUS}"
for gpu in "${GPU_LIST[@]}"; do
  worker_loop "${gpu}" &
done
wait
