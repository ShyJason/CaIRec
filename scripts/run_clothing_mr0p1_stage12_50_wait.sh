#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPUS="${GPUS:-3 4 5 6}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MEM_FREE_THRESHOLD="${MEM_FREE_THRESHOLD:-1000}"

STAMP="${STAMP:-20260624_stage12_50}"
SUFFIX="${SUFFIX:-stage1_2_clothing_seed2023_completion_mm_v2_mr0p1_epoch50_${STAMP}}"
CONFIG="${CONFIG:-configs/clothing/stage1_2_decoder_v2.yaml}"
IMPUTER_CKPT="${IMPUTER_CKPT:-/home/ruiyuliu/projects/MMRec/exp_report/clothing/stage1_1_clothing_seed2023_completion_mm_mmrec_clothing_mm_mr0p1_seed2023_20260623_004452/ckpt/stage1_1_clothing_seed2023_completion_mm_mmrec_clothing_mm_mr0p1_seed2023_20260623_004452_imputer_param_50_epoch1.pth}"

is_gpu_free() {
  local gpu="$1"
  local used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')"
  [[ "${used}" =~ ^[0-9]+$ ]] && (( used < MEM_FREE_THRESHOLD ))
}

pick_gpu() {
  local gpu
  for gpu in ${GPUS}; do
    if is_gpu_free "${gpu}"; then
      printf '%s\n' "${gpu}"
      return 0
    fi
  done
  return 1
}

if [[ ! -f "${CONFIG}" ]]; then
  echo "missing config: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${IMPUTER_CKPT}" ]]; then
  echo "missing imputer checkpoint: ${IMPUTER_CKPT}" >&2
  exit 1
fi

echo "[$(date -Is)] waiting for free GPU among: ${GPUS}"
while true; do
  if DEVICE_ID="$(pick_gpu)"; then
    break
  fi
  echo "[$(date -Is)] no free GPU yet"
  sleep "${POLL_SECONDS}"
done

echo "[$(date -Is)] launch stage1.2 50 epochs on GPU ${DEVICE_ID}"
exec env \
  DATASET=clothing \
  CONFIG="${CONFIG}" \
  SUFFIX="${SUFFIX}" \
  IMPUTER_CKPT="${IMPUTER_CKPT}" \
  MISSING_RATE=0.1 \
  EPOCHS=50 \
  DEVICE_ID="${DEVICE_ID}" \
  EXP_MODE=mm \
  FEATURE_BRIDGE_MODE=raw_decoder \
  GCN_FRONTEND_MODE=original_linear \
  STAGE1_PROFILE=v2 \
  STAGE1_V2_LOSS_PRESET=balanced \
  IMPUTATION_SELECTION_POLICY=stage1_default \
  IMPUTATION_VAL_RATE=0.1 \
  SAVE_ALL_EPOCHS=1 \
  TENSORBOARD=0 \
  ./run_stage1_2_baby_imputer_backprop_decoder_v2.sh \
  --tensorboard 0
