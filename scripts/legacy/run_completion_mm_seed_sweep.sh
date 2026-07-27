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
if [[ "${DATASET}" == "tiktok" ]]; then
  STAGE12_EPOCHS="${STAGE12_EPOCHS:-20}"
  STAGE2_EPOCHS="${STAGE2_EPOCHS:-200}"
else
  STAGE12_EPOCHS="${STAGE12_EPOCHS:-5}"
  STAGE2_EPOCHS="${STAGE2_EPOCHS:-30}"
fi
SEEDS=("$@")
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  SEEDS=(1 12 123 1234 12345)
fi
find_best_ckpt() {
  local dataset="$1"
  local suffix="$2"
  local exp_dir="${ROOT_DIR}/exp_report/${dataset}/${suffix}"
  local best_epoch=""
  local best_ckpt=""

  best_epoch="$(grep -hEo 'best epoch [0-9]+' "${exp_dir}/log/"*.log 2>/dev/null | tail -n 1 | awk '{print $3}')"
  if [[ -n "${best_epoch}" ]]; then
    best_ckpt="$(ls -t "${exp_dir}/ckpt/"*_epoch"${best_epoch}".pth 2>/dev/null | head -n 1)"
    if [[ -n "${best_ckpt}" ]]; then
      echo "${best_ckpt}"
      return 0
    fi
  fi

  ls -t "${exp_dir}/ckpt/"*.pth 2>/dev/null | head -n 1
}
for seed in "${SEEDS[@]}"; do
  echo "[seed-sweep] running seed=${seed}"
  s11="stage1_1_${DATASET}_seed${seed}_completion_${EXP_MODE}"
  s12="stage1_2_${DATASET}_seed${seed}_completion_${EXP_MODE}"
  s2="stage2_${DATASET}_seed${seed}_completion_${EXP_MODE}"
  rm -rf "${ROOT_DIR}/exp_report/${DATASET}/${s11}" "${ROOT_DIR}/exp_report/${DATASET}/${s12}" "${ROOT_DIR}/exp_report/${DATASET}/${s2}"
  DEVICE_ID="${DEVICE_ID}" SAVE="${SAVE}" TENSORBOARD="${TENSORBOARD}" EXP_MODE="${EXP_MODE}" MISSING_RATE="${MISSING_RATE}" FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE}" EVALUATION_PROTOCOL="${EVALUATION_PROTOCOL}" EPOCHS="${STAGE11_EPOCHS}" SUFFIX="${s11}" ./run_stage1_1_baby_imputer_param.sh --seed "${seed}"
  ckpt11="$(find_best_ckpt "${DATASET}" "${s11}")"
  if [[ -z "${ckpt11}" ]]; then
    echo "[seed-sweep] missing stage1.1 ckpt for seed=${seed}" >&2
    exit 1
  fi
  DEVICE_ID="${DEVICE_ID}" SAVE="${SAVE}" TENSORBOARD="${TENSORBOARD}" EXP_MODE="${EXP_MODE}" MISSING_RATE="${MISSING_RATE}" FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE}" EVALUATION_PROTOCOL="${EVALUATION_PROTOCOL}" EPOCHS="${STAGE12_EPOCHS}" SUFFIX="${s12}" IMPUTER_CKPT="${ckpt11}" ./run_stage1_2_baby_imputer_backprop_decoder_v2.sh --seed "${seed}"
  ckpt12="$(find_best_ckpt "${DATASET}" "${s12}")"
  if [[ -z "${ckpt12}" ]]; then
    echo "[seed-sweep] missing stage1.2 ckpt for seed=${seed}" >&2
    exit 1
  fi
  DEVICE_ID="${DEVICE_ID}" SAVE="${SAVE}" TENSORBOARD="${TENSORBOARD}" EXP_MODE="${EXP_MODE}" MISSING_RATE="${MISSING_RATE}" FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE}" EVALUATION_PROTOCOL="${EVALUATION_PROTOCOL}" EPOCHS="${STAGE2_EPOCHS}" SUFFIX="${s2}" IMPUTER_CKPT="${ckpt12}" ./run_stage2_baby_recommender_decoder.sh --seed "${seed}"
  echo "[seed-sweep] completed seed=${seed}"
done
