#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

DATASET="${DATASET:-clothing}"
EXP_MODE="${EXP_MODE:-mm}"
MISSING_RATE="${MISSING_RATE:-0.1}"
SEED="${SEED:-2023}"
DATASET_SEED="${DATASET_SEED:-0}"
DEVICE_ID="${DEVICE_ID:-0}"

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
MEM_FREE_THRESHOLD="${MEM_FREE_THRESHOLD:-1000}"
POLL_SECONDS="${POLL_SECONDS:-60}"

STAMP="${STAMP:-clothing_mmrec_mr0p1_best_20260624}"
STAGE11_SUFFIX="${STAGE11_SUFFIX:-stage1_1_clothing_mm_mr0p1_${STAMP}}"
STAGE12_SUFFIX="${STAGE12_SUFFIX:-stage1_2_clothing_mm_mr0p1_beststyle_nocl_${STAMP}}"
STAGE2_SUFFIX="${STAGE2_SUFFIX:-stage2_clothing_mm_mr0p1_gcncl_malpha025_${STAMP}}"

STAGE1_1_EPOCHS="${STAGE1_1_EPOCHS:-5}"
STAGE1_2_EPOCHS="${STAGE1_2_EPOCHS:-50}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-200}"

run1_log="exp_report/clothing/${STAGE11_SUFFIX}_mr0p1_repro.log"
run2_log="exp_report/clothing/${STAGE12_SUFFIX}_mr0p1_repro.log"
run3_log="exp_report/clothing/${STAGE2_SUFFIX}_mr0p1_repro.log"

mkdir -p exp_report/clothing

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
    echo "[run] no free GPU (threshold ${MEM_FREE_THRESHOLD} MiB), retry in ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
  done
}

latest_ckpt_for_suffix() {
  local suffix="$1"
  local ckpt_dir="exp_report/clothing/${suffix}/ckpt"
  find "${ckpt_dir}" -maxdepth 1 -type f -name "*.pth" -printf "%T@ %p\n" 2>/dev/null \
    | sort -nr \
    | awk 'NR == 1 { $1=""; sub(/^ /, ""); print }'
}

log() {
  echo "[run $(date -Is)] $*"
}

log "Repro pipeline for clothing missing_rate=${MISSING_RATE}, seed=${SEED}, stamp=${STAMP}"
log "Stage1.1 suffix=${STAGE11_SUFFIX}"
log "Stage1.2 suffix=${STAGE12_SUFFIX}"
log "Stage2 suffix=${STAGE2_SUFFIX}"

log "Stage1.1 start"
GPU="$(wait_for_gpu)"
log "using gpu=${GPU}"
(
  echo "[stage1.1] start"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  DEVICE_ID=0 \
  DATASET="${DATASET}" \
  EXP_MODE="${EXP_MODE}" \
  MISSING_RATE="${MISSING_RATE}" \
  SEED="${SEED}" \
  DATASET_SEED="${DATASET_SEED}" \
  EPOCHS="${STAGE1_1_EPOCHS}" \
  SUFFIX="${STAGE11_SUFFIX}" \
  ./run_stage1_1_baby_imputer_param.sh
) 2>&1 | tee "${run1_log}"

STAGE11_CKPT="$(latest_ckpt_for_suffix "${STAGE11_SUFFIX}")"
if [[ -z "${STAGE11_CKPT}" || ! -f "${STAGE11_CKPT}" ]]; then
  log "failed to find stage1.1 checkpoint for ${STAGE11_SUFFIX}"
  exit 1
fi
log "stage1.1 ckpt=${STAGE11_CKPT}"

log "Stage1.2 start"
GPU="$(wait_for_gpu)"
log "using gpu=${GPU}"
(
  echo "[stage1.2] start"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  CONFIG="configs/${DATASET}/stage1_2_decoder_v2.yaml" \
  DATASET="${DATASET}" \
  EXP_MODE="${EXP_MODE}" \
  MISSING_RATE="${MISSING_RATE}" \
  SEED="${SEED}" \
  DATASET_SEED="${DATASET_SEED}" \
  SUFFIX="${STAGE12_SUFFIX}" \
  IMPUTER_CKPT="${STAGE11_CKPT}" \
  EPOCHS="${STAGE1_2_EPOCHS}" \
  STAGE1_PROFILE=v2 \
  STAGE1_V2_LOSS_PRESET=balanced \
  GENERATIVE_UPDATE_MODE=fixed \
  STAGE1_REC_LOSS_MODE=observed \
  DECODE_LOSS_GRAD_MODE=detached \
  DECODE_LOSS_TARGET_MODE=observed \
  DEVICE_ID=0 \
  TENSORBOARD=0 \
  ./run_stage1_2_baby_imputer_backprop_decoder_v2.sh
) 2>&1 | tee "${run2_log}"

STAGE12_CKPT="$(latest_ckpt_for_suffix "${STAGE12_SUFFIX}")"
if [[ -z "${STAGE12_CKPT}" || ! -f "${STAGE12_CKPT}" ]]; then
  log "failed to find stage1.2 checkpoint for ${STAGE12_SUFFIX}"
  exit 1
fi
log "stage1.2 ckpt=${STAGE12_CKPT}"

log "Stage2 start"
GPU="$(wait_for_gpu)"
log "using gpu=${GPU}"
(
  echo "[stage2] start"
  CUDA_VISIBLE_DEVICES="${GPU}" .venv/bin/python -u main.py \
    --config configs/clothing/stage2_decoder_mm_itemgraph_completed.yaml \
    --device_id 0 \
    --seed "${SEED}" \
    --dataset_seed "${DATASET_SEED}" \
    --dataset "${DATASET}" \
    --exp_mode "${EXP_MODE}" \
    --train_stage recommender \
    --missing_rate "${MISSING_RATE}" \
    --imputer_ckpt "${STAGE12_CKPT}" \
    --suffix "${STAGE2_SUFFIX}" \
    --batch_size 2048 \
    --epoch "${STAGE2_EPOCHS}" \
    --early_stop 10000 \
    --lr 0.01 \
    --lr_rec 0.01 \
    --lr_imp 0.0002 \
    --lr_decoder 0.00005 \
    --freeze_imputer 1 \
    --freeze_recommender -1 \
    --freeze_decoder 1 \
    --topk "[10, 20, 30, 40, 50]" \
    --tensorboard 0 \
    --save 1 \
) 2>&1 | tee "${run3_log}"

log "done"
