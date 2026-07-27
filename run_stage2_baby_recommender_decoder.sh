#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

DEVICE_ID="${DEVICE_ID:-4}"
SAVE="${SAVE:-1}"
DATASET="${DATASET:-baby}"
EXP_MODE="${EXP_MODE:-mm}"
SUFFIX="${SUFFIX:-stage2_${DATASET}_recommender_decoder_${EXP_MODE}}"
CKPT="${CKPT:-}"
IMPUTER_CKPT="${IMPUTER_CKPT:-}"
if [[ -z "${IMPUTER_CKPT}" && -z "${CKPT}" ]]; then
  echo "Set IMPUTER_CKPT to the decoder stage1.2 checkpoint path, or CKPT to resume a full checkpoint" >&2
  exit 1
fi
CONFIG="${CONFIG:-configs/${DATASET}/stage2_decoder_${EXP_MODE}.yaml}"
if [[ ! -f "${CONFIG}" ]]; then
  FALLBACK_CONFIG="configs/${DATASET}/stage2_decoder_mm.yaml"
  if [[ -f "${FALLBACK_CONFIG}" ]]; then
    echo "[stage2-decoder] missing ${CONFIG}; falling back to ${FALLBACK_CONFIG} and overriding exp_mode=${EXP_MODE}" >&2
    CONFIG="${FALLBACK_CONFIG}"
  else
    echo "[stage2-decoder] missing config ${CONFIG} and fallback ${FALLBACK_CONFIG}" >&2
    exit 1
  fi
fi

cmd=(env CONFIG="${CONFIG}" TRAIN_STAGE="${TRAIN_STAGE:-recommender}" DATASET="${DATASET}" EXP_MODE="${EXP_MODE}" MISSING_RATE="${MISSING_RATE:-0.3}" FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE:-raw_decoder}" GCN_FRONTEND_MODE=original_linear ./run_demo_itm.sh)
if [[ -n "${IMPUTER_CKPT}" ]]; then
  cmd+=(--imputer_ckpt "${IMPUTER_CKPT}")
fi
exec "${cmd[@]}" \
  --freeze_imputer "${FREEZE_IMPUTER:-1}" \
  --freeze_decoder "${FREEZE_DECODER:-1}" \
  --epoch "${EPOCHS:-200}" \
  --early_stop "${EARLY_STOP:-20}" \
  --batch_size "${BATCH_SIZE:-256}" \
  --lr "${LR:-1e-3}" \
  --lr_rec "${LR_REC:-1e-3}" \
  --lr_imp "${LR_IMP:-2e-4}" \
  --lr_decoder "${LR_DECODER:-5e-5}" \
  --strict_probe_test_interval "${STRICT_PROBE_TEST_INTERVAL:-10}" \
  --save "${SAVE}" \
  --suffix "${SUFFIX}" \
  --device_id "${DEVICE_ID}" \
  "$@"
