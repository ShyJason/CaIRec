#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATASET="${DATASET:-baby}"
DEVICE_ID="${DEVICE_ID:-4}"
MISSING_RATE="${MISSING_RATE:-0.3}"
FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE:-raw_decoder}"
EVALUATION_PROTOCOL="${EVALUATION_PROTOCOL:-strict}"
SAVE="${SAVE:-1}"
TENSORBOARD="${TENSORBOARD:-0}"
EXP_MODE="${EXP_MODE:-mm}"

STAGE11_EPOCHS="${STAGE11_EPOCHS:-5}"
STAGE12_EPOCHS="${STAGE12_EPOCHS:-5}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-30}"

LR="${LR:-1e-3}"
LR_REC="${LR_REC:-1e-3}"
LR_IMP="${LR_IMP:-2e-4}"
LR_DECODER="${LR_DECODER:-5e-5}"
MODALITY_BPR_COEFF="${MODALITY_BPR_COEFF:-0.2}"
BETA_INTRA="${BETA_INTRA:-0.05}"
BETA_INTER="${BETA_INTER:-0.05}"
BETA_ITM="${BETA_ITM:-0.05}"
BETA_REC="${BETA_REC:-0.01}"
BETA_DECODE="${BETA_DECODE:-0.01}"

SEEDS=("$@")
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  SEEDS=(1 12 123 1234 12345)
fi

find_latest_ckpt() {
  local dataset="$1"
  local suffix="$2"
  ls -t "${ROOT_DIR}/exp_report/${dataset}/${suffix}/ckpt/"*.pth 2>/dev/null | head -n 1
}

for seed in "${SEEDS[@]}"; do
  echo "[joint-seed-sweep] running seed=${seed}"
  s11="stage1_1_${DATASET}_seed${seed}_completion_${EXP_MODE}_joint"
  s12="stage1_2_${DATASET}_seed${seed}_completion_${EXP_MODE}_joint"
  s2="stage2_${DATASET}_seed${seed}_completion_${EXP_MODE}_joint"

  rm -rf \
    "${ROOT_DIR}/exp_report/${DATASET}/${s11}" \
    "${ROOT_DIR}/exp_report/${DATASET}/${s12}" \
    "${ROOT_DIR}/exp_report/${DATASET}/${s2}"

  DEVICE_ID="${DEVICE_ID}" SAVE="${SAVE}" TENSORBOARD="${TENSORBOARD}" \
  EXP_MODE="${EXP_MODE}" MISSING_RATE="${MISSING_RATE}" FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE}" \
  EVALUATION_PROTOCOL="${EVALUATION_PROTOCOL}" EPOCHS="${STAGE11_EPOCHS}" SUFFIX="${s11}" \
  ./run_stage1_1_baby_imputer_param.sh --seed "${seed}"

  ckpt11="$(find_latest_ckpt "${DATASET}" "${s11}")"
  if [[ -z "${ckpt11}" ]]; then
    echo "[joint-seed-sweep] missing stage1.1 ckpt for seed=${seed}" >&2
    exit 1
  fi

  DEVICE_ID="${DEVICE_ID}" SAVE="${SAVE}" TENSORBOARD="${TENSORBOARD}" \
  EXP_MODE="${EXP_MODE}" MISSING_RATE="${MISSING_RATE}" FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE}" \
  EVALUATION_PROTOCOL="${EVALUATION_PROTOCOL}" EPOCHS="${STAGE12_EPOCHS}" SUFFIX="${s12}" \
  IMPUTER_CKPT="${ckpt11}" ./run_stage1_2_baby_imputer_backprop_decoder_v2.sh --seed "${seed}"

  ckpt12="$(find_latest_ckpt "${DATASET}" "${s12}")"
  if [[ -z "${ckpt12}" ]]; then
    echo "[joint-seed-sweep] missing stage1.2 ckpt for seed=${seed}" >&2
    exit 1
  fi

  DEVICE_ID="${DEVICE_ID}" SAVE="${SAVE}" TENSORBOARD="${TENSORBOARD}" \
  EXP_MODE="${EXP_MODE}" MISSING_RATE="${MISSING_RATE}" FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE}" \
  EVALUATION_PROTOCOL="${EVALUATION_PROTOCOL}" EPOCHS="${STAGE2_EPOCHS}" SUFFIX="${s2}" \
  IMPUTER_CKPT="${ckpt12}" LR="${LR}" LR_REC="${LR_REC}" LR_IMP="${LR_IMP}" LR_DECODER="${LR_DECODER}" \
  MODALITY_BPR_COEFF="${MODALITY_BPR_COEFF}" BETA_INTRA="${BETA_INTRA}" BETA_INTER="${BETA_INTER}" \
  BETA_ITM="${BETA_ITM}" BETA_REC="${BETA_REC}" BETA_DECODE="${BETA_DECODE}" \
  "${ROOT_DIR}/scripts/legacy/run_stage2_baby_joint_decoder.sh" \
    --seed "${seed}" \
    --modality_bpr_coeff "${MODALITY_BPR_COEFF}"

  echo "[joint-seed-sweep] completed seed=${seed}"
done
