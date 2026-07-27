#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

exec env \
  DATASET=clothing \
  EXP_MODE=mm \
  SEED="${SEED:-2023}" \
  DEVICE_ID="${DEVICE_ID:-4}" \
  RUN_TAG="${RUN_TAG}" \
  STAGE12_SCRIPT="./run_stage1_2_baby_imputer_backprop_decoder_v2.sh" \
  STAGE11_SUFFIX="${STAGE11_SUFFIX:-stage1_1_clothing_seed${SEED:-2023}_completion_mm_${RUN_TAG}}" \
  STAGE12_SUFFIX="${STAGE12_SUFFIX:-stage1_2_clothing_seed${SEED:-2023}_completion_mm_v2_${RUN_TAG}}" \
  STAGE12_EPOCHS="${STAGE12_EPOCHS:-20}" \
  STAGE2_SUFFIX="${STAGE2_SUFFIX:-stage2_clothing_seed${SEED:-2023}_completion_mm_v2_${RUN_TAG}}" \
  STAGE2_EPOCHS="${STAGE2_EPOCHS:-200}" \
  STAGE2_EARLY_STOP="${STAGE2_EARLY_STOP:-20}" \
  STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-2048}" \
  STAGE2_LR="${STAGE2_LR:-0.01}" \
  STAGE2_LR_REC="${STAGE2_LR_REC:-0.01}" \
  TENSORBOARD="${TENSORBOARD:-0}" \
  ./run_baby_three_stage_report.sh
