#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

DEVICE_ID="${DEVICE_ID:-4}"
SAVE="${SAVE:-1}"
DATASET="${DATASET:-baby}"
EXP_MODE="${EXP_MODE:-mm}"
MISSING_RATE="${MISSING_RATE:-0.3}"
SUFFIX="${SUFFIX:-stage1_1_${DATASET}_imputer_param}"
CONFIG="${CONFIG:-configs/${DATASET}/stage1_1_imputer_param.yaml}"

exec env CONFIG="${CONFIG}" TRAIN_STAGE=imputer_param DATASET="${DATASET}" EXP_MODE="${EXP_MODE}" MISSING_RATE="${MISSING_RATE}" FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE:-raw_decoder}" ./run_demo_itm.sh \
  --epoch "${EPOCHS:-5}" \
  --batch_size "${BATCH_SIZE:-256}" \
  --lr "${LR:-1e-3}" \
  --alpha_rec "${ALPHA_REC:-1.0}" \
  --save "${SAVE}" \
  --suffix "${SUFFIX}" \
  --device_id "${DEVICE_ID}" \
  "$@"
