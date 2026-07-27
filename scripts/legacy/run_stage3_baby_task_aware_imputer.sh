#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DEVICE_ID="${DEVICE_ID:-4}"
SAVE="${SAVE:-1}"
EXP_MODE="${EXP_MODE:-mm}"
SUFFIX="${SUFFIX:-stage3_baby_task_aware_imputer_${EXP_MODE}}"
CKPT="${CKPT:?Set CKPT to the trained recommender checkpoint path}"
CONFIG="${CONFIG:-configs/baby/stage2_decoder_${EXP_MODE}.yaml}"

exec env CONFIG="${CONFIG}" \
  TRAIN_STAGE=joint \
  DATASET=baby \
  EXP_MODE="${EXP_MODE}" \
  MISSING_RATE="${MISSING_RATE:-0.3}" \
  FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE:-raw_decoder}" \
  GCN_FRONTEND_MODE=original_linear \
  "${ROOT_DIR}/run_demo_itm.sh" \
  --ckpt "${CKPT}" \
  --freeze_imputer "${FREEZE_IMPUTER:-0}" \
  --freeze_recommender "${FREEZE_RECOMMENDER:-1}" \
  --freeze_decoder "${FREEZE_DECODER:-0}" \
  --epoch "${EPOCHS:-3}" \
  --batch_size "${BATCH_SIZE:-256}" \
  --lr "${LR:-5e-5}" \
  --lr_rec "${LR_REC:-5e-5}" \
  --lr_imp "${LR_IMP:-2e-5}" \
  --lr_decoder "${LR_DECODER:-1e-5}" \
  --beta_intra "${BETA_INTRA:-0.005}" \
  --beta_inter "${BETA_INTER:-0.005}" \
  --beta_itm "${BETA_ITM:-0.005}" \
  --beta_rec "${BETA_REC:-0.001}" \
  --beta_decode "${BETA_DECODE:-0.003}" \
  --gamma_align "${GAMMA_ALIGN:-0.05}" \
  --gamma_distill "${GAMMA_DISTILL:-0.1}" \
  --save "${SAVE}" \
  --suffix "${SUFFIX}" \
  --device_id "${DEVICE_ID}" \
  "$@"
