#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="${STAMP:-clothing_mr0p1_hparam_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="exp_report/clothing/hparam_search_mr0p1_${STAMP}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
STATE_DIR="${OUT_DIR}/state"
mkdir -p "${STATE_DIR}"
NEXT_FILE="${STATE_DIR}/next_index"
LOCK_FILE="${STATE_DIR}/queue.lock"
if [[ ! -f "${NEXT_FILE}" ]]; then
  echo 0 > "${NEXT_FILE}"
fi

GPUS="${GPUS:-5 6}"
MEM_FREE_THRESHOLD="${MEM_FREE_THRESHOLD:-1000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"

CONFIG="${CONFIG:-configs/clothing/stage2_decoder_mm_itemgraph_completed.yaml}"
STAGE12_DIR="${STAGE12_DIR:-exp_report/clothing/stage1_2_clothing_seed2023_completion_mm_v2_mmrec_clothing_mm_mr0p1_seed2023_20260623_004452}"
STAGE12_PREFIX="${STAGE12_PREFIX:-stage1_2_clothing_seed2023_completion_mm_v2_mmrec_clothing_mm_mr0p1_seed2023_20260623_004452}"
DEFAULT_IMPUTER_CKPT="${STAGE12_DIR}/ckpt/${STAGE12_PREFIX}_imputer_backprop_50_epoch18.pth"
IMPUTER_CKPT="${IMPUTER_CKPT:-${DEFAULT_IMPUTER_CKPT}}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "missing config: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${IMPUTER_CKPT}" ]]; then
  echo "missing imputer checkpoint: ${IMPUTER_CKPT}" >&2
  exit 1
fi

# tag modality_bpr_coeff item_graph_modal_alpha cf image text ckpt_epoch
CANDIDATES=(
  "e10_mbpr2p0_default 2.0 0.25 0.3 0.35 0.35 10"
  "e10_mbpr2p5_default 2.5 0.25 0.3 0.35 0.35 10"
  "e11_mbpr2p0_default 2.0 0.25 0.3 0.35 0.35 11"
  "e15_mbpr2p0_default 2.0 0.25 0.3 0.35 0.35 15"
  "e18_mbpr1p5_default 1.5 0.25 0.3 0.35 0.35 18"
  "e18_mbpr2p5_default 2.5 0.25 0.3 0.35 0.35 18"
  "e18_mbpr3p0_default 3.0 0.25 0.3 0.35 0.35 18"
  "e18_mbpr2p0_img60 2.0 0.25 0.1 0.60 0.30 18"
  "e18_mbpr2p5_img60 2.5 0.25 0.1 0.60 0.30 18"
  "e18_mbpr2p0_img55 2.0 0.25 0.1 0.55 0.35 18"
  "e18_mbpr2p0_img60_malpha0p5 2.0 0.5 0.1 0.60 0.30 18"
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
        hr, rec, ndcg = map(float, vals[-1])
        rows.append((rec, ndcg, int(best[-1]) if best else -1, os.path.basename(path)))
for rec, ndcg, best, name in sorted(rows, reverse=True):
    print(f"{rec:.5f}\tndcg20={ndcg:.5f}\tbest={best}\t{name}")
PY
}

echo "[$(date -Is)] start ${STAMP}" | tee -a "${OUT_DIR}/launcher.log"
echo "CONFIG=${CONFIG}" | tee -a "${OUT_DIR}/launcher.log"
echo "IMPUTER_CKPT=${IMPUTER_CKPT}" | tee -a "${OUT_DIR}/launcher.log"

run_candidate() {
  local tag="$1"
  local mbpr="$2"
  local modal_alpha="$3"
  local cf_w="$4"
  local image_w="$5"
  local text_w="$6"
  local ckpt_epoch="$7"
  local gpu="$8"
  local suffix="stage2_clothing_mr0p1_${tag}_${STAMP}"
  local log_path="${LOG_DIR}/${suffix}.log"
  local run_imputer_ckpt="${STAGE12_DIR}/ckpt/${STAGE12_PREFIX}_imputer_backprop_50_epoch${ckpt_epoch}.pth"

  echo "[$(date -Is)] launch ${tag} on GPU ${gpu} ckpt_epoch=${ckpt_epoch}" | tee -a "${OUT_DIR}/launcher.log"

  .venv/bin/python -u main.py \
    --config "${CONFIG}" \
    --device_id "${gpu}" \
    --dataset clothing \
    --exp_mode mm \
    --train_stage recommender \
    --missing_rate 0.1 \
    --seed 2023 \
    --dataset_seed 0 \
    --imputer_ckpt "${run_imputer_ckpt}" \
    --suffix "${suffix}" \
    --epoch 200 \
    --early_stop 10000 \
    --eva_interval 1 \
    --batch_size 2048 \
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
    --modality_bpr_coeff "${mbpr}" \
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

  echo "[$(date -Is)] done ${tag} on GPU ${gpu}" | tee -a "${OUT_DIR}/launcher.log"
}

wait_for_gpu() {
  local gpu="$1"
  while true; do
    if is_gpu_free "${gpu}"; then
      return 0
    fi
    echo "[$(date -Is)] worker gpu=${gpu} waiting for free GPU" | tee -a "${OUT_DIR}/launcher.log" >&2
    sleep "${POLL_SECONDS}"
  done
}

claim_next_candidate() {
  (
    flock -x 9
    local idx tag mbpr modal_alpha cf_w image_w text_w ckpt_epoch suffix log_path
    idx="$(cat "${NEXT_FILE}")"
    while (( idx < ${#CANDIDATES[@]} )); do
      read -r tag mbpr modal_alpha cf_w image_w text_w ckpt_epoch <<<"${CANDIDATES[$idx]}"
      suffix="stage2_clothing_mr0p1_${tag}_${STAMP}"
      log_path="${LOG_DIR}/${suffix}.log"
      if [[ -f "${log_path}" ]] && grep -q "final strict test hr@20" "${log_path}"; then
        echo "[$(date -Is)] skip finished ${tag}" >> "${OUT_DIR}/launcher.log"
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

worker_loop() {
  local worker_idx="$1"
  local gpu="$2"
  local cand

  while true; do
    wait_for_gpu "${gpu}"
    if ! cand="$(claim_next_candidate)"; then
      echo "[$(date -Is)] worker ${worker_idx} gpu=${gpu} no candidates left" | tee -a "${OUT_DIR}/launcher.log"
      return 0
    fi

    read -r tag mbpr modal_alpha cf_w image_w text_w ckpt_epoch <<<"${cand}"
    run_imputer_ckpt="${STAGE12_DIR}/ckpt/${STAGE12_PREFIX}_imputer_backprop_50_epoch${ckpt_epoch}.pth"

    if [[ ! -f "${run_imputer_ckpt}" ]]; then
      echo "missing candidate imputer checkpoint: ${run_imputer_ckpt}" >&2
      exit 1
    fi

    run_candidate "${tag}" "${mbpr}" "${modal_alpha}" "${cf_w}" "${image_w}" "${text_w}" "${ckpt_epoch}" "${gpu}"
    summarize_results | tee "${OUT_DIR}/summary.tsv"
  done
}

read -r -a GPU_LIST <<<"${GPUS}"
WORKER_COUNT="${#GPU_LIST[@]}"
if (( WORKER_COUNT > MAX_PARALLEL )); then
  WORKER_COUNT="${MAX_PARALLEL}"
fi
if (( WORKER_COUNT < 1 )); then
  echo "no GPUs configured" >&2
  exit 1
fi

for ((worker_idx = 0; worker_idx < WORKER_COUNT; worker_idx++)); do
  worker_loop "${worker_idx}" "${GPU_LIST[$worker_idx]}" &
done

wait

echo "[$(date -Is)] all done" | tee -a "${OUT_DIR}/launcher.log"
summarize_results | tee "${OUT_DIR}/summary.tsv"
