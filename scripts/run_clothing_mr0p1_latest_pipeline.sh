#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
MEM_FREE_THRESHOLD="${MEM_FREE_THRESHOLD:-1000}"
POLL_SECONDS="${POLL_SECONDS:-60}"

DATASET=clothing
EXP_MODE=mm
MISSING_RATE="${MISSING_RATE:-0.1}"
MR_TAG="${MISSING_RATE//./p}"
STAMP="${STAMP:-clothing_mr${MR_TAG}_latest_$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-2023}"
DATASET_SEED="${DATASET_SEED:-0}"

STAGE11_SUFFIX="stage1_1_${DATASET}_${EXP_MODE}_mr${MR_TAG}_${STAMP}"
STAGE12_SUFFIX="stage1_2_${DATASET}_${EXP_MODE}_mr${MR_TAG}_beststyle_nocl_${STAMP}"
STAGE2_SUFFIX="stage2_${DATASET}_${EXP_MODE}_mr${MR_TAG}_gcncl_malpha025_${STAMP}"

log() {
  echo "[$(date -Is)] $*"
}

is_gpu_free() {
  local gpu="$1"
  local used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')"
  [[ "${used}" =~ ^[0-9]+$ ]] && (( used < MEM_FREE_THRESHOLD ))
}

wait_for_gpu() {
  while true; do
    local gpu
    for gpu in ${GPUS}; do
      if is_gpu_free "${gpu}"; then
        echo "${gpu}"
        return 0
      fi
    done
    log "no free GPU below ${MEM_FREE_THRESHOLD} MiB; polling again in ${POLL_SECONDS}s" >&2
    sleep "${POLL_SECONDS}"
  done
}

latest_ckpt_for_suffix() {
  local suffix="$1"
  local ckpt_dir="exp_report/${DATASET}/${suffix}/ckpt"
  find "${ckpt_dir}" -maxdepth 1 -type f -name "*.pth" -printf "%T@ %p\n" 2>/dev/null \
    | sort -nr \
    | awk 'NR == 1 { $1=""; sub(/^ /, ""); print }'
}

log "run_tag=${STAMP}"
log "stage1.1 suffix=${STAGE11_SUFFIX}"
log "stage1.2 suffix=${STAGE12_SUFFIX}"
log "stage2 suffix=${STAGE2_SUFFIX}"
log "GPUS=${GPUS} missing_rate=${MISSING_RATE} seed=${SEED}"

GPU="$(wait_for_gpu)"
log "stage1.1 start gpu=${GPU}"
CUDA_VISIBLE_DEVICES="${GPU}" \
DATASET="${DATASET}" \
EXP_MODE="${EXP_MODE}" \
MISSING_RATE="${MISSING_RATE}" \
SEED="${SEED}" \
DATASET_SEED="${DATASET_SEED}" \
DEVICE_ID=0 \
SUFFIX="${STAGE11_SUFFIX}" \
SAVE=1 \
TENSORBOARD=0 \
./run_stage1_1_baby_imputer_param.sh

STAGE11_CKPT="$(latest_ckpt_for_suffix "${STAGE11_SUFFIX}")"
if [[ -z "${STAGE11_CKPT}" || ! -f "${STAGE11_CKPT}" ]]; then
  log "failed to find stage1.1 checkpoint for ${STAGE11_SUFFIX}"
  exit 1
fi
log "stage1.1 ckpt=${STAGE11_CKPT}"

GPU="$(wait_for_gpu)"
log "stage1.2 start gpu=${GPU}"
CUDA_VISIBLE_DEVICES="${GPU}" \
DATASET="${DATASET}" \
EXP_MODE="${EXP_MODE}" \
MISSING_RATE="${MISSING_RATE}" \
SEED="${SEED}" \
DATASET_SEED="${DATASET_SEED}" \
DEVICE_ID=0 \
SUFFIX="${STAGE12_SUFFIX}" \
IMPUTER_CKPT="${STAGE11_CKPT}" \
SAVE=1 \
TENSORBOARD=1 \
./run_stage1_2_baby_imputer_backprop_decoder_v2.sh

STAGE12_CKPT="$(latest_ckpt_for_suffix "${STAGE12_SUFFIX}")"
if [[ -z "${STAGE12_CKPT}" || ! -f "${STAGE12_CKPT}" ]]; then
  log "failed to find stage1.2 checkpoint for ${STAGE12_SUFFIX}"
  exit 1
fi
log "stage1.2 ckpt=${STAGE12_CKPT}"

GPU="$(wait_for_gpu)"
log "stage2 start gpu=${GPU}"
CUDA_VISIBLE_DEVICES="${GPU}" .venv/bin/python -u main.py \
  --config configs/clothing/stage2_decoder_mm_itemgraph_completed.yaml \
  --device_id 0 \
  --dataset "${DATASET}" \
  --exp_mode "${EXP_MODE}" \
  --train_stage recommender \
  --missing_rate "${MISSING_RATE}" \
  --seed "${SEED}" \
  --dataset_seed "${DATASET_SEED}" \
  --imputer_ckpt "${STAGE12_CKPT}" \
  --suffix "${STAGE2_SUFFIX}" \
  --tensorboard 0 \
  --save 1 \
  --topk "[10, 20, 30, 40, 50]"

log "done"
