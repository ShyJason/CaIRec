#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

DEVICE_ID="${DEVICE_ID:-4}"
SAVE="${SAVE:-1}"
DATASET="${DATASET:-baby}"
SUFFIX="${SUFFIX:-stage1_align_${DATASET}_promrl}"
IMPUTER_CKPT="${IMPUTER_CKPT:?Set IMPUTER_CKPT to the stage1 init checkpoint path}"
CONFIG="${CONFIG:-configs/${DATASET}/stage1_align_promrl.yaml}"

exec env CONFIG="${CONFIG}" TRAIN_STAGE=imputer_align DATASET="${DATASET}" EXP_MODE="${EXP_MODE:-mm}" MISSING_RATE="${MISSING_RATE:-0.3}" ./run_demo_itm.sh \
  --train_stage imputer_align \
  --imputer_ckpt "${IMPUTER_CKPT}" \
  --epoch "${EPOCHS:-20}" \
  --batch_size "${BATCH_SIZE:-256}" \
  --lr "${LR:-5e-4}" \
  --lr_imp "${LR_IMP:-5e-4}" \
  --lr_decoder "${LR_DECODER:-2e-4}" \
  --generative_update_mode "${GENERATIVE_UPDATE_MODE:-fixed}" \
  --stage1_masking_policy "${STAGE1_MASKING_POLICY:-dynamic}" \
  --imputation_val_rate "${IMPUTATION_VAL_RATE:-0.1}" \
  --imputation_selection_policy "${IMPUTATION_SELECTION_POLICY:-promrl_shared}" \
  --alpha_intra "${ALPHA_INTRA:-1.0}" \
  --alpha_inter "${ALPHA_INTER:-1.0}" \
  --alpha_itm "${ALPHA_ITM:-1.0}" \
  --alpha_rec "${ALPHA_REC:-0.1}" \
  --alpha_decode "${ALPHA_DECODE:-0.0}" \
  --freeze_imputer "${FREEZE_IMPUTER:--1}" \
  --freeze_recommender "${FREEZE_RECOMMENDER:--1}" \
  --freeze_decoder "${FREEZE_DECODER:--1}" \
  --save "${SAVE}" \
  --suffix "${SUFFIX}" \
  --device_id "${DEVICE_ID}" \
  "$@"
