#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DEVICE_ID="${DEVICE_ID:-4}"
SAVE="${SAVE:-1}"
EXP_MODE="${EXP_MODE:-mm}"
SUFFIX="${SUFFIX:-stage2_baby_recommender_decoder_${EXP_MODE}_no_impute}"
IMPUTER_CKPT="${IMPUTER_CKPT:-}"

cmd=(
  env
  "TRAIN_STAGE=recommender"
  "DATASET=baby"
  "EXP_MODE=${EXP_MODE}"
  "FEATURE_BRIDGE_MODE=raw_decoder"
  "GCN_FRONTEND_MODE=original_linear"
  "${ROOT_DIR}/run_demo_itm.sh"
  --disable_imputation 1
  --epoch "${EPOCHS:-20}"
  --batch_size "${BATCH_SIZE:-256}"
  --lr "${LR:-1e-3}"
  --lr_rec "${LR_REC:-1e-3}"
  --freeze_decoder "${FREEZE_DECODER:-1}"
  --save "${SAVE}"
  --suffix "${SUFFIX}"
  --device_id "${DEVICE_ID}"
)

if [[ -n "${IMPUTER_CKPT}" ]]; then
  cmd+=(--imputer_ckpt "${IMPUTER_CKPT}")
fi

cmd+=("$@")
exec "${cmd[@]}"
