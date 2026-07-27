#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CKPT="${CKPT:?Set CKPT to the trained recommender checkpoint path}"
DEVICE_ID="${DEVICE_ID:-4}"
EXP_MODE="${EXP_MODE:-mm}"
MISSING_RATE="${MISSING_RATE:-0.3}"
EPOCHS="${EPOCHS:-5}"
SAVE="${SAVE:-1}"
FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE:-raw_decoder}"
RUN_TAG="${RUN_TAG:-stage3_grid_$(date +%Y%m%d_%H%M%S)}"

declare -a SEARCH_SPACE=(
  "A 0.0 0.0"
  "B 0.0 0.001"
  "C 0.01 0.0"
  "D 0.01 0.001"
)

for spec in "${SEARCH_SPACE[@]}"; do
  read -r label gamma_align beta_decode <<< "${spec}"
  suffix="${RUN_TAG}_${label}"
  echo "[grid] running ${label}: gamma_align=${gamma_align}, beta_decode=${beta_decode}, suffix=${suffix}"
  FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE}" \
  CKPT="${CKPT}" \
  DEVICE_ID="${DEVICE_ID}" \
  EXP_MODE="${EXP_MODE}" \
  MISSING_RATE="${MISSING_RATE}" \
  EPOCHS="${EPOCHS}" \
  SAVE="${SAVE}" \
  GAMMA_ALIGN="${gamma_align}" \
  BETA_DECODE="${beta_decode}" \
  SUFFIX="${suffix}" \
  "${ROOT_DIR}/scripts/legacy/run_stage3_baby_task_aware_imputer.sh"
done
