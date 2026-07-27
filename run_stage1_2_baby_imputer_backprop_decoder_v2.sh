#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

DATASET="${DATASET:-baby}"
CONFIG="${CONFIG:-configs/${DATASET}/stage1_2_decoder_v2.yaml}"
SUFFIX="${SUFFIX:-stage1_2_${DATASET}_beststyle_nocl_decoder_v2}"
IMPUTER_CKPT="${IMPUTER_CKPT:?Set IMPUTER_CKPT to the stage1.1 checkpoint path}"
DEFAULT_EPOCHS=50
DEFAULT_ALPHA_REC=1.0
DEFAULT_GENERATIVE_UPDATE_MODE=fixed
DEFAULT_DECODE_LOSS_GRAD_MODE=detached
STAGE1_2_MODE=observed
DEFAULT_STAGE1_REC_LOSS_MODE=observed
DEFAULT_DECODE_LOSS_TARGET_MODE=observed

exec env \
  CONFIG="${CONFIG}" \
  SUFFIX="${SUFFIX}" \
  STAGE1_2_MODE="${STAGE1_2_MODE}" \
  TRAIN_STAGE=imputer_backprop \
  DATASET="${DATASET}" \
  EXP_MODE="${EXP_MODE:-mm}" \
  MISSING_RATE="${MISSING_RATE:-0.3}" \
  FEATURE_BRIDGE_MODE=raw_decoder \
  GCN_FRONTEND_MODE=original_linear \
  EPOCHS="${EPOCHS:-${DEFAULT_EPOCHS}}" \
  STAGE1_PROFILE="${STAGE1_PROFILE:-v2}" \
  STAGE1_V2_LOSS_PRESET="${STAGE1_V2_LOSS_PRESET:-balanced}" \
  GENERATIVE_UPDATE_MODE="${GENERATIVE_UPDATE_MODE:-${DEFAULT_GENERATIVE_UPDATE_MODE}}" \
  STAGE1_REC_LOSS_MODE="${STAGE1_REC_LOSS_MODE:-${DEFAULT_STAGE1_REC_LOSS_MODE}}" \
  DECODE_LOSS_GRAD_MODE="${DECODE_LOSS_GRAD_MODE:-${DEFAULT_DECODE_LOSS_GRAD_MODE}}" \
  DECODE_LOSS_TARGET_MODE="${DECODE_LOSS_TARGET_MODE:-${DEFAULT_DECODE_LOSS_TARGET_MODE}}" \
  IMPUTATION_SELECTION_POLICY="${IMPUTATION_SELECTION_POLICY:-stage1_default}" \
  IMPUTATION_VAL_RATE="${IMPUTATION_VAL_RATE:-0.1}" \
  SAVE_ALL_EPOCHS="${SAVE_ALL_EPOCHS:-1}" \
  ./run_demo_itm.sh \
  --imputer_ckpt "${IMPUTER_CKPT}" \
  --epoch "${EPOCHS:-${DEFAULT_EPOCHS}}" \
  --batch_size "${BATCH_SIZE:-256}" \
  --lr "${LR:-5e-4}" \
  --lr_imp "${LR_IMP:-5e-4}" \
  --lr_decoder "${LR_DECODER:-2e-4}" \
  --stage1_2_mode "${STAGE1_2_MODE}" \
  --stage1_profile "${STAGE1_PROFILE:-v2}" \
  --stage1_v2_loss_preset "${STAGE1_V2_LOSS_PRESET:-balanced}" \
  --alpha_intra "${ALPHA_INTRA:-1.0}" \
  --alpha_inter "${ALPHA_INTER:-1.0}" \
  --alpha_itm "${ALPHA_ITM:-1.0}" \
  --alpha_rec "${ALPHA_REC:-${DEFAULT_ALPHA_REC}}" \
  --alpha_decode "${ALPHA_DECODE:-1.0}" \
  --alpha_decode_kl "${ALPHA_DECODE_KL:-0.0}" \
  --decode_kl_temp "${DECODE_KL_TEMP:-0.2}" \
  --decode_loss_grad_mode "${DECODE_LOSS_GRAD_MODE:-${DEFAULT_DECODE_LOSS_GRAD_MODE}}" \
  --decode_loss_target_mode "${DECODE_LOSS_TARGET_MODE:-${DEFAULT_DECODE_LOSS_TARGET_MODE}}" \
  --generative_update_mode "${GENERATIVE_UPDATE_MODE:-${DEFAULT_GENERATIVE_UPDATE_MODE}}" \
  --stage1_rec_loss_mode "${STAGE1_REC_LOSS_MODE:-${DEFAULT_STAGE1_REC_LOSS_MODE}}" \
  --imputation_val_rate "${IMPUTATION_VAL_RATE:-0.1}" \
  --imputation_selection_policy "${IMPUTATION_SELECTION_POLICY:-stage1_default}" \
  --imputation_selection_split "${IMPUTATION_SELECTION_SPLIT:-train}" \
  --imputation_selection_metric "${IMPUTATION_SELECTION_METRIC:-mse}" \
  --save "${SAVE:-1}" \
  --save_all_epochs "${SAVE_ALL_EPOCHS:-1}" \
  --suffix "${SUFFIX}" \
  --device_id "${DEVICE_ID:-4}" \
  "$@"
