#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

DEVICE_ID="${DEVICE_ID:-4}"
SAVE="${SAVE:-1}"
DATASET="${DATASET:-baby}"
SUFFIX="${SUFFIX:-stage1_adapter_${DATASET}_mmrec}"
IMPUTER_CKPT="${IMPUTER_CKPT:?Set IMPUTER_CKPT to the stage1 align checkpoint path}"
CONFIG="${CONFIG:-configs/${DATASET}/stage1_adapter_mmrec.yaml}"

exec env CONFIG="${CONFIG}" TRAIN_STAGE=imputer_adapter DATASET="${DATASET}" EXP_MODE="${EXP_MODE:-mm}" MISSING_RATE="${MISSING_RATE:-0.3}" ./run_demo_itm.sh \
  --train_stage imputer_adapter \
  --imputer_ckpt "${IMPUTER_CKPT}" \
  --epoch "${EPOCHS:-10}" \
  --batch_size "${BATCH_SIZE:-256}" \
  --lr "${LR:-5e-4}" \
  --lr_imp "${LR_IMP:-5e-4}" \
  --lr_decoder "${LR_DECODER:-2e-4}" \
  --generative_update_mode "${GENERATIVE_UPDATE_MODE:-fixed}" \
  --stage1_masking_policy "${STAGE1_MASKING_POLICY:-fixed}" \
  --imputation_val_rate "${IMPUTATION_VAL_RATE:-0.1}" \
  --imputation_selection_policy "${IMPUTATION_SELECTION_POLICY:-adapter_default}" \
  --alpha_intra "${ALPHA_INTRA:-0.0}" \
  --alpha_inter "${ALPHA_INTER:-0.0}" \
  --alpha_itm "${ALPHA_ITM:-0.0}" \
  --alpha_rec "${ALPHA_REC:-0.0}" \
  --alpha_decode "${ALPHA_DECODE:-1.0}" \
  --freeze_imputer "${FREEZE_IMPUTER:--1}" \
  --freeze_recommender "${FREEZE_RECOMMENDER:--1}" \
  --freeze_decoder "${FREEZE_DECODER:-0}" \
  --save "${SAVE}" \
  --suffix "${SUFFIX}" \
  --device_id "${DEVICE_ID}" \
  "$@"
