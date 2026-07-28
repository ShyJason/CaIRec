#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi

latest_ckpt_for_suffix() {
  local suffix="$1"
  local ckpt_dir="${ROOT_DIR}/exp_report/${DATASET}/${suffix}/ckpt"
  find "${ckpt_dir}" -maxdepth 1 -type f -name "*.pth" -printf "%T@ %p\n" 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

ckpt_for_suffix_epoch() {
  local suffix="$1"
  local epoch="$2"
  local ckpt_dir="${ROOT_DIR}/exp_report/${DATASET}/${suffix}/ckpt"
  find "${ckpt_dir}" -maxdepth 1 -type f -name "*_epoch${epoch}.pth" -printf "%T@ %p\n" 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

DATASET="${DATASET:-clothing}"
EXP_MODE="${EXP_MODE:-mm}"
DEVICE_ID="${DEVICE_ID:-4}"
SEED="${SEED:-2023}"
FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE:-decoupled_latent}"
GCN_FRONTEND_MODE="${GCN_FRONTEND_MODE:-original_linear}"
PROMRL_PROJECTION_MODE="${PROMRL_PROJECTION_MODE:-learned}"

if [[ "${DATASET}" == "clothing" ]]; then
  DEFAULT_MISSING_RATE="0.3"
  DEFAULT_STAGE12_EPOCHS="50"
  DEFAULT_STAGE2_EPOCHS="400"
  DEFAULT_STAGE2_BATCH_SIZE="2048"
  DEFAULT_STAGE2_LR="0.005"
  DEFAULT_STAGE2_LR_REC="0.005"
  DEFAULT_STAGE2_GAMMA_ALIGN="0.00125"
else
  DEFAULT_MISSING_RATE="0.3"
  DEFAULT_STAGE12_EPOCHS="5"
  DEFAULT_STAGE2_EPOCHS="200"
  DEFAULT_STAGE2_BATCH_SIZE="256"
  DEFAULT_STAGE2_LR="0.001"
  DEFAULT_STAGE2_LR_REC="0.001"
  DEFAULT_STAGE2_GAMMA_ALIGN="0.0"
fi
MISSING_RATE="${MISSING_RATE:-${DEFAULT_MISSING_RATE}}"
TRAIN_MISSING_MODALITY="${TRAIN_MISSING_MODALITY:-random}"
EVAL_MISSING_RATE="${EVAL_MISSING_RATE:-0.5}"
MISSING_MASK_PROTOCOL="${MISSING_MASK_PROTOCOL:-i3}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
STAGE11_EPOCHS="${STAGE11_EPOCHS:-5}"
STAGE12_EPOCHS="${STAGE12_EPOCHS:-${DEFAULT_STAGE12_EPOCHS}}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-${DEFAULT_STAGE2_EPOCHS}}"
STAGE2_EARLY_STOP="${STAGE2_EARLY_STOP:-}"
STAGE11_BATCH_SIZE="${STAGE11_BATCH_SIZE:-256}"
STAGE12_BATCH_SIZE="${STAGE12_BATCH_SIZE:-256}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-${DEFAULT_STAGE2_BATCH_SIZE}}"
STAGE11_LR="${STAGE11_LR:-0.001}"
STAGE12_LR="${STAGE12_LR:-0.0005}"
STAGE12_LR_IMP="${STAGE12_LR_IMP:-0.0005}"
STAGE12_GENERATIVE_UPDATE_MODE="${STAGE12_GENERATIVE_UPDATE_MODE:-fixed}"
STAGE11_IMPUTATION_VAL_RATE="${STAGE11_IMPUTATION_VAL_RATE:-0.0}"
STAGE12_IMPUTATION_VAL_RATE="${STAGE12_IMPUTATION_VAL_RATE:-0.1}"
STAGE2_LR="${STAGE2_LR:-${DEFAULT_STAGE2_LR}}"
STAGE2_LR_REC="${STAGE2_LR_REC:-${DEFAULT_STAGE2_LR_REC}}"
STAGE2_LR_IMP="${STAGE2_LR_IMP:-0.0002}"
STAGE2_GAMMA_ALIGN="${STAGE2_GAMMA_ALIGN:-${DEFAULT_STAGE2_GAMMA_ALIGN}}"
STAGE2_ADAPTER_ALIGN_PSEUDO_RATIO="${STAGE2_ADAPTER_ALIGN_PSEUDO_RATIO:-1.0}"
RUN_STAGE2="${RUN_STAGE2:-1}"
PROJECTION_CKPT="${PROJECTION_CKPT:-${ROOT_DIR}/pretrained_projections/${DATASET}.pth}"
STAGE11_SUFFIX="${STAGE11_SUFFIX:-three_stage_${DATASET}_${EXP_MODE}_${FEATURE_BRIDGE_MODE}_${RUN_TAG}_stage1_1_param}"
STAGE12_SUFFIX="${STAGE12_SUFFIX:-three_stage_${DATASET}_${EXP_MODE}_${FEATURE_BRIDGE_MODE}_${RUN_TAG}_stage1_2_completion}"
STAGE2_SUFFIX="${STAGE2_SUFFIX:-three_stage_${DATASET}_${EXP_MODE}_${FEATURE_BRIDGE_MODE}_${RUN_TAG}_stage2_recommender}"

STAGE11_CONFIG="${STAGE11_CONFIG:-configs/${DATASET}/stage1_1_imputer_param.yaml}"
STAGE12_CONFIG="${STAGE12_CONFIG:-configs/${DATASET}/stage1_2_decoder_v2.yaml}"
if [[ "${DATASET}" == "clothing" && -f "configs/clothing/mainline_mr0p1.yaml" ]]; then
  DEFAULT_STAGE2_CONFIG="configs/clothing/mainline_mr0p1.yaml"
else
  DEFAULT_STAGE2_CONFIG="configs/${DATASET}/stage2_decoder_${EXP_MODE}.yaml"
fi
STAGE2_CONFIG="${STAGE2_CONFIG:-${DEFAULT_STAGE2_CONFIG}}"
if [[ ! -f "${STAGE2_CONFIG}" ]]; then
  STAGE2_FALLBACK_CONFIG="configs/${DATASET}/stage2_decoder_mm.yaml"
  if [[ -f "${STAGE2_FALLBACK_CONFIG}" ]]; then
    echo "[mainline] missing ${STAGE2_CONFIG}; falling back to ${STAGE2_FALLBACK_CONFIG}" >&2
    STAGE2_CONFIG="${STAGE2_FALLBACK_CONFIG}"
  else
    echo "[mainline] missing stage2 config ${STAGE2_CONFIG} and fallback ${STAGE2_FALLBACK_CONFIG}" >&2
    exit 1
  fi
fi

echo "[mainline] dataset=${DATASET} exp_mode=${EXP_MODE} bridge=${FEATURE_BRIDGE_MODE} train_missing_modality=${TRAIN_MISSING_MODALITY} missing_rate=${MISSING_RATE} eval_missing_rate=${EVAL_MISSING_RATE}"

if [[ ! -f "${PROJECTION_CKPT}" ]]; then
  echo "[mainline] projection checkpoint not found: ${PROJECTION_CKPT}" >&2
  exit 1
fi
echo "[mainline] pretrained projection=${PROJECTION_CKPT}"

echo "[mainline] stage1.1: imputer_param epochs=${STAGE11_EPOCHS} suffix=${STAGE11_SUFFIX}"

env \
  CONFIG="${STAGE11_CONFIG}" \
  DATASET="${DATASET}" \
  EXP_MODE="${EXP_MODE}" \
  TRAIN_STAGE=imputer_param \
  FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE}" \
  GCN_FRONTEND_MODE="${GCN_FRONTEND_MODE}" \
  PROMRL_PROJECTION_MODE="${PROMRL_PROJECTION_MODE}" \
  MISSING_RATE="${MISSING_RATE}" \
  TRAIN_MISSING_MODALITY="${TRAIN_MISSING_MODALITY}" \
  EVAL_MISSING_RATE="${EVAL_MISSING_RATE}" \
  MISSING_MASK_PROTOCOL="${MISSING_MASK_PROTOCOL}" \
  DEVICE_ID="${DEVICE_ID}" \
  SEED="${SEED}" \
  EPOCHS="${STAGE11_EPOCHS}" \
  BATCH_SIZE="${STAGE11_BATCH_SIZE}" \
  LR="${STAGE11_LR}" \
  IMPUTATION_VAL_RATE="${STAGE11_IMPUTATION_VAL_RATE}" \
  PROJECTION_CKPT="${PROJECTION_CKPT}" \
  IMPUTER_CKPT= \
  ALPHA_REC="${STAGE11_ALPHA_REC:-1.0}" \
  ALPHA_DECODE=0.0 \
  SAVE=1 \
  SAVE_ALL_EPOCHS=1 \
  SUFFIX="${STAGE11_SUFFIX}" \
  ./run_demo_itm.sh \
    --freeze_recommender 1 \
    --freeze_imputer -1 \
    --freeze_decoder 1 \
    "$@"

STAGE11_FINAL_EPOCH=$((STAGE11_EPOCHS - 1))
STAGE11_CKPT="${STAGE11_CKPT:-$(ckpt_for_suffix_epoch "${STAGE11_SUFFIX}" "${STAGE11_FINAL_EPOCH}")}"
if [[ -z "${STAGE11_CKPT}" || ! -f "${STAGE11_CKPT}" ]]; then
  echo "[mainline] failed to find final stage1.1 checkpoint epoch=${STAGE11_FINAL_EPOCH} for suffix=${STAGE11_SUFFIX}" >&2
  exit 1
fi

echo "[mainline] stage1.1 final checkpoint=${STAGE11_CKPT}"
echo "[mainline] stage1.2: completion/imputer epochs=${STAGE12_EPOCHS} suffix=${STAGE12_SUFFIX}"

env \
  CONFIG="${STAGE12_CONFIG}" \
  DATASET="${DATASET}" \
  EXP_MODE="${EXP_MODE}" \
  TRAIN_STAGE=imputer_backprop \
  FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE}" \
  GCN_FRONTEND_MODE="${GCN_FRONTEND_MODE}" \
  PROMRL_PROJECTION_MODE="${PROMRL_PROJECTION_MODE}" \
  MISSING_RATE="${MISSING_RATE}" \
  TRAIN_MISSING_MODALITY="${TRAIN_MISSING_MODALITY}" \
  EVAL_MISSING_RATE="${EVAL_MISSING_RATE}" \
  MISSING_MASK_PROTOCOL="${MISSING_MASK_PROTOCOL}" \
  DEVICE_ID="${DEVICE_ID}" \
  SEED="${SEED}" \
  EPOCHS="${STAGE12_EPOCHS}" \
  BATCH_SIZE="${STAGE12_BATCH_SIZE}" \
  LR="${STAGE12_LR}" \
  LR_IMP="${STAGE12_LR_IMP}" \
  GENERATIVE_UPDATE_MODE="${STAGE12_GENERATIVE_UPDATE_MODE}" \
  IMPUTATION_VAL_RATE="${STAGE12_IMPUTATION_VAL_RATE}" \
  IMPUTER_CKPT="${STAGE11_CKPT}" \
  ALPHA_REC="${STAGE12_ALPHA_REC:-1.0}" \
  ALPHA_INTRA="${STAGE12_ALPHA_INTRA:-1.0}" \
  ALPHA_INTER="${STAGE12_ALPHA_INTER:-1.0}" \
  ALPHA_ITM="${STAGE12_ALPHA_ITM:-1.0}" \
  ALPHA_DECODE=0.0 \
  SAVE=1 \
  SAVE_ALL_EPOCHS=1 \
  SUFFIX="${STAGE12_SUFFIX}" \
  ./run_demo_itm.sh \
    --imputer_ckpt "${STAGE11_CKPT}" \
    --freeze_recommender 1 \
    --freeze_imputer -1 \
    --freeze_decoder 1 \
    "$@"

STAGE12_FINAL_EPOCH=$((STAGE12_EPOCHS - 1))
STAGE12_CKPT="${STAGE12_CKPT:-$(ckpt_for_suffix_epoch "${STAGE12_SUFFIX}" "${STAGE12_FINAL_EPOCH}")}"
if [[ -z "${STAGE12_CKPT}" || ! -f "${STAGE12_CKPT}" ]]; then
  echo "[mainline] failed to find final stage1.2 checkpoint epoch=${STAGE12_FINAL_EPOCH} for suffix=${STAGE12_SUFFIX}" >&2
  exit 1
fi

echo "[mainline] stage1.2 final checkpoint=${STAGE12_CKPT}"
if [[ "${RUN_STAGE2}" != "1" ]]; then
  echo "[mainline] RUN_STAGE2=${RUN_STAGE2}; stopping after stage1.2"
  exit 0
fi
echo "[mainline] stage2: recommender epochs=${STAGE2_EPOCHS} suffix=${STAGE2_SUFFIX}"

env \
  CONFIG="${STAGE2_CONFIG}" \
  DATASET="${DATASET}" \
  EXP_MODE="${EXP_MODE}" \
  TRAIN_STAGE=recommender \
  FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE}" \
  GCN_FRONTEND_MODE="${GCN_FRONTEND_MODE}" \
  PROMRL_PROJECTION_MODE="${PROMRL_PROJECTION_MODE}" \
  MISSING_RATE="${MISSING_RATE}" \
  TRAIN_MISSING_MODALITY="${TRAIN_MISSING_MODALITY}" \
  EVAL_MISSING_RATE="${EVAL_MISSING_RATE}" \
  MISSING_MASK_PROTOCOL="${MISSING_MASK_PROTOCOL}" \
  DEVICE_ID="${DEVICE_ID}" \
  SEED="${SEED}" \
  EPOCHS="${STAGE2_EPOCHS}" \
  EARLY_STOP="${STAGE2_EARLY_STOP}" \
  BATCH_SIZE="${STAGE2_BATCH_SIZE}" \
  LR="${STAGE2_LR}" \
  LR_REC="${STAGE2_LR_REC}" \
  LR_IMP="${STAGE2_LR_IMP}" \
  IMPUTATION_VAL_RATE=0.0 \
  GAMMA_ALIGN="${STAGE2_GAMMA_ALIGN}" \
  ADAPTER_ALIGN_PSEUDO_RATIO="${STAGE2_ADAPTER_ALIGN_PSEUDO_RATIO}" \
  IMPUTER_CKPT="${STAGE12_CKPT}" \
  ALPHA_DECODE=0.0 \
  ALPHA_DECODE_KL=0.0 \
  SAVE="${SAVE:-1}" \
  SUFFIX="${STAGE2_SUFFIX}" \
  ./run_demo_itm.sh \
    --imputer_ckpt "${STAGE12_CKPT}" \
    --freeze_imputer 1 \
    --freeze_decoder 1 \
    --freeze_recommender -1 \
    "$@"
