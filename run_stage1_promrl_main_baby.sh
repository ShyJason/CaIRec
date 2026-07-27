#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

DEVICE_ID="${DEVICE_ID:-4}"
SAVE="${SAVE:-1}"
DATASET="${DATASET:-baby}"
CONFIG="${CONFIG:-configs/${DATASET}/stage1_promrl_main.yaml}"
SUFFIX="${SUFFIX:-stage1_promrl_main_${DATASET}_promrl}"
IMPUTER_CKPT="${IMPUTER_CKPT:?Set IMPUTER_CKPT to the stage1 init checkpoint path}"

exec env CONFIG="${CONFIG}" TRAIN_STAGE=imputer_promrl_main DATASET="${DATASET}" EXP_MODE="${EXP_MODE:-mm}" MISSING_RATE="${MISSING_RATE:-0.3}" ./run_demo_itm.sh \
  --train_stage imputer_promrl_main \
  --imputer_ckpt "${IMPUTER_CKPT}" \
  --freeze_imputer "${FREEZE_IMPUTER:--1}" \
  --freeze_recommender "${FREEZE_RECOMMENDER:--1}" \
  --freeze_decoder "${FREEZE_DECODER:--1}" \
  --save "${SAVE}" \
  --suffix "${SUFFIX}" \
  --device_id "${DEVICE_ID}" \
  "$@"
