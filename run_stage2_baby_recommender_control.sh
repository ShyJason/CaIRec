#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

DEVICE_ID="${DEVICE_ID:-4}"
SAVE="${SAVE:-1}"
DATASET="${DATASET:-baby}"
EXP_MODE="${EXP_MODE:-mm}"
SUFFIX="${SUFFIX:-stage2_${DATASET}_recommender_control_${EXP_MODE}}"

exec env TRAIN_STAGE=recommender DATASET="${DATASET}" EXP_MODE="${EXP_MODE}" MISSING_RATE="${MISSING_RATE:-0.3}" FEATURE_BRIDGE_MODE=raw_decoder GCN_FRONTEND_MODE=original_linear DISABLE_IMPUTATION=1 FREEZE_DECODER=1 ./run_demo_itm.sh \
  --disable_imputation 1 \
  --freeze_decoder 1 \
  --epoch "${EPOCHS:-200}" \
  --early_stop "${EARLY_STOP:-20}" \
  --batch_size "${BATCH_SIZE:-256}" \
  --lr "${LR:-5e-4}" \
  --lr_rec "${LR_REC:-5e-4}" \
  --strict_probe_test_interval "${STRICT_PROBE_TEST_INTERVAL:-10}" \
  --save "${SAVE}" \
  --suffix "${SUFFIX}" \
  --device_id "${DEVICE_ID}" \
  "$@"
