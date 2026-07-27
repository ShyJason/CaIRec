#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DEVICE_ID="${DEVICE_ID:-4}"
SAVE="${SAVE:-1}"
EXP_MODE="${EXP_MODE:-mm}"
SUFFIX="${SUFFIX:-stage2_baby_joint_decoder_${EXP_MODE}}"
IMPUTER_CKPT="${IMPUTER_CKPT:?Set IMPUTER_CKPT to the stage1.2 checkpoint path}"
CONFIG="${CONFIG:-configs/baby/stage2_decoder_${EXP_MODE}.yaml}"

exec env CONFIG="${CONFIG}" \
  TRAIN_STAGE=joint \
  DATASET=baby \
  EXP_MODE="${EXP_MODE}" \
  MISSING_RATE="${MISSING_RATE:-0.3}" \
  FEATURE_BRIDGE_MODE=raw_decoder \
  GCN_FRONTEND_MODE=original_linear \
  "${ROOT_DIR}/run_demo_itm.sh" \
  --imputer_ckpt "${IMPUTER_CKPT}" \
  --freeze_imputer "${FREEZE_IMPUTER:-0}" \
  --freeze_decoder "${FREEZE_DECODER:-0}" \
  --epoch "${EPOCHS:-30}" \
  --batch_size "${BATCH_SIZE:-256}" \
  --lr "${LR:-5e-4}" \
  --lr_rec "${LR_REC:-5e-4}" \
  --lr_imp "${LR_IMP:-1e-4}" \
  --lr_decoder "${LR_DECODER:-5e-5}" \
  --beta_intra "${BETA_INTRA:-0.02}" \
  --beta_inter "${BETA_INTER:-0.02}" \
  --beta_itm "${BETA_ITM:-0.02}" \
  --beta_rec "${BETA_REC:-0.005}" \
  --beta_decode "${BETA_DECODE:-0.01}" \
  --save "${SAVE}" \
  --suffix "${SUFFIX}" \
  --device_id "${DEVICE_ID}" \
  "$@"
