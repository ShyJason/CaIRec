#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

DEVICE_ID="${DEVICE_ID:-4}"
SAVE="${SAVE:-1}"
TENSORBOARD="${TENSORBOARD:-0}"
DATASET="${DATASET:-baby}"
EXP_MODE="${EXP_MODE:-mm}"
FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE:-raw_decoder}"
MISSING_RATE="${MISSING_RATE:-0.3}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${ROOT_DIR}/exp_report/${DATASET}/pipeline_reports/${RUN_TAG}_${FEATURE_BRIDGE_MODE}_${EXP_MODE}"

mkdir -p "${REPORT_DIR}"

find_latest_ckpt() {
  local suffix="$1"
  ls -t "${ROOT_DIR}/exp_report/${DATASET}/${suffix}/ckpt/"*.pth 2>/dev/null | head -n 1
}

find_best_ckpt() {
  local suffix="$1"
  local exp_dir="${ROOT_DIR}/exp_report/${DATASET}/${suffix}"
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

  find_latest_ckpt "${suffix}"
}

reset_stage_output() {
  local suffix="$1"
  rm -rf "${ROOT_DIR}/exp_report/${DATASET}/${suffix}"
}

extract_last_train_line() {
  local log_file="$1"
  grep 'TRAIN:stage' "${log_file}" | tail -n 1 || true
}

extract_best_block() {
  local log_file="$1"
  grep -A5 'best epoch' "${log_file}" || true
}

if [[ -z "${STAGE12_SCRIPT:-}" ]]; then
  STAGE12_SCRIPT="./run_stage1_2_baby_imputer_backprop_decoder_v2.sh"
fi

if [[ -z "${STAGE2_SCRIPT:-}" ]]; then
  STAGE2_SCRIPT="./run_stage2_baby_recommender_decoder.sh"
fi

if [[ -z "${STAGE12_SUFFIX:-}" ]]; then
  if [[ "${FEATURE_BRIDGE_MODE}" == "raw_decoder" ]]; then
    STAGE12_SUFFIX="stage1_2_${DATASET}_imputer_backprop_decoder_v2_${RUN_TAG}"
  else
    STAGE12_SUFFIX="stage1_2_${DATASET}_imputer_backprop_${FEATURE_BRIDGE_MODE}_${RUN_TAG}"
  fi
fi

if [[ -z "${STAGE2_SUFFIX:-}" ]]; then
  if [[ "${FEATURE_BRIDGE_MODE}" == "raw_decoder" ]]; then
    STAGE2_SUFFIX="stage2_${DATASET}_recommender_decoder_${EXP_MODE}_${RUN_TAG}"
  else
    STAGE2_SUFFIX="stage2_${DATASET}_recommender_${FEATURE_BRIDGE_MODE}_${EXP_MODE}_${RUN_TAG}"
  fi
fi
DEFAULT_STAGE2_FREEZE_IMPUTER=1

STAGE11_SUFFIX="${STAGE11_SUFFIX:-stage1_1_${DATASET}_imputer_param_${RUN_TAG}}"

STAGE11_LOG="${REPORT_DIR}/stage1_1.log"
STAGE12_LOG="${REPORT_DIR}/stage1_2.log"
STAGE2_LOG="${REPORT_DIR}/stage2_${EXP_MODE}.log"
SUMMARY_FILE="${REPORT_DIR}/summary.md"

echo "[pipeline] report_dir=${REPORT_DIR}"
echo "[pipeline] dataset=${DATASET}"
echo "[pipeline] feature_bridge_mode=${FEATURE_BRIDGE_MODE} exp_mode=${EXP_MODE}"
echo "[pipeline] missing_rate=${MISSING_RATE}"

echo "[pipeline] Stage 1.1"
reset_stage_output "${STAGE11_SUFFIX}"
{
  DEVICE_ID="${DEVICE_ID}" \
  SEED="${SEED:-2023}" \
  SAVE="${SAVE}" \
  DATASET="${DATASET}" \
  EXP_MODE="${STAGE11_EXP_MODE:-mm}" \
  MISSING_RATE="${MISSING_RATE}" \
  TENSORBOARD="${TENSORBOARD}" \
  HF_TENSORBOARD_REPO="${HF_TENSORBOARD_REPO:-}" \
  HF_TOKEN="${HF_TOKEN:-}" \
  HF_COMMIT_EVERY="${HF_COMMIT_EVERY:-5}" \
  SUFFIX="${STAGE11_SUFFIX}" \
  EPOCHS="${STAGE11_EPOCHS:-5}" \
  BATCH_SIZE="${BATCH_SIZE:-256}" \
  LR="${STAGE11_LR:-1e-3}" \
  ALPHA_REC="${STAGE11_ALPHA_REC:-1.0}" \
  ./run_stage1_1_baby_imputer_param.sh
} 2>&1 | tee "${STAGE11_LOG}"

STAGE11_CKPT="$(find_best_ckpt "${STAGE11_SUFFIX}")"
if [[ -z "${STAGE11_CKPT}" ]]; then
  echo "[pipeline] Failed to locate stage 1.1 checkpoint" >&2
  exit 1
fi
echo "[pipeline] stage1.1 checkpoint=${STAGE11_CKPT}"

echo "[pipeline] Stage 1.2"
reset_stage_output "${STAGE12_SUFFIX}"
{
  DEVICE_ID="${DEVICE_ID}" \
  SEED="${SEED:-2023}" \
  SAVE="${SAVE}" \
  DATASET="${DATASET}" \
  EXP_MODE="${STAGE12_EXP_MODE:-mm}" \
  MISSING_RATE="${MISSING_RATE}" \
  TENSORBOARD="${TENSORBOARD}" \
  HF_TENSORBOARD_REPO="${HF_TENSORBOARD_REPO:-}" \
  HF_TOKEN="${HF_TOKEN:-}" \
  HF_COMMIT_EVERY="${HF_COMMIT_EVERY:-5}" \
  SUFFIX="${STAGE12_SUFFIX}" \
  IMPUTER_CKPT="${STAGE11_CKPT}" \
  EPOCHS="${STAGE12_EPOCHS:-20}" \
  BATCH_SIZE="${BATCH_SIZE:-256}" \
  LR="${STAGE12_LR:-5e-4}" \
  LR_IMP="${STAGE12_LR_IMP:-5e-4}" \
  LR_DECODER="${STAGE12_LR_DECODER:-2e-4}" \
  ALPHA_INTRA="${STAGE12_ALPHA_INTRA:-1.0}" \
  ALPHA_INTER="${STAGE12_ALPHA_INTER:-1.0}" \
  ALPHA_ITM="${STAGE12_ALPHA_ITM:-1.0}" \
  ALPHA_REC="${STAGE12_ALPHA_REC:-0.1}" \
  ALPHA_DECODE="${STAGE12_ALPHA_DECODE:-1.0}" \
  FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE}" \
  "${STAGE12_SCRIPT}"
} 2>&1 | tee "${STAGE12_LOG}"

STAGE12_CKPT="$(find_best_ckpt "${STAGE12_SUFFIX}")"
if [[ -z "${STAGE12_CKPT}" ]]; then
  echo "[pipeline] Failed to locate stage 1.2 checkpoint" >&2
  exit 1
fi
echo "[pipeline] stage1.2 checkpoint=${STAGE12_CKPT}"

echo "[pipeline] Stage 2 (${EXP_MODE})"
reset_stage_output "${STAGE2_SUFFIX}"
{
  DEVICE_ID="${DEVICE_ID}" \
  SEED="${SEED:-2023}" \
  SAVE="${SAVE}" \
  DATASET="${DATASET}" \
  EXP_MODE="${EXP_MODE}" \
  MISSING_RATE="${MISSING_RATE}" \
  SUFFIX="${STAGE2_SUFFIX}" \
  IMPUTER_CKPT="${STAGE12_CKPT}" \
  FREEZE_IMPUTER="${STAGE2_FREEZE_IMPUTER:-${DEFAULT_STAGE2_FREEZE_IMPUTER}}" \
  FREEZE_DECODER="${STAGE2_FREEZE_DECODER:-1}" \
  EPOCHS="${STAGE2_EPOCHS:-200}" \
  EARLY_STOP="${STAGE2_EARLY_STOP:-20}" \
  BATCH_SIZE="${STAGE2_BATCH_SIZE:-${BATCH_SIZE:-256}}" \
  LR="${STAGE2_LR:-1e-3}" \
  LR_REC="${STAGE2_LR_REC:-1e-3}" \
  LR_IMP="${STAGE2_LR_IMP:-2e-4}" \
  LR_DECODER="${STAGE2_LR_DECODER:-5e-5}" \
  FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE}" \
  TENSORBOARD="${TENSORBOARD}" \
  HF_TENSORBOARD_REPO="${HF_TENSORBOARD_REPO:-}" \
  HF_TOKEN="${HF_TOKEN:-}" \
  HF_COMMIT_EVERY="${HF_COMMIT_EVERY:-5}" \
  "${STAGE2_SCRIPT}"
} 2>&1 | tee "${STAGE2_LOG}"

cat > "${SUMMARY_FILE}" <<EOF
# Three-Stage Pipeline Results

- Run tag: \`${RUN_TAG}\`
- Dataset: \`${DATASET}\`
- Feature bridge mode: \`${FEATURE_BRIDGE_MODE}\`
- Stage2 exp_mode: \`${EXP_MODE}\`
- Missing rate: \`${MISSING_RATE}\`
- Report dir: \`${REPORT_DIR}\`

## Stage 1.1

- Suffix: \`${STAGE11_SUFFIX}\`
- Checkpoint: \`${STAGE11_CKPT}\`
- Final train line:

\`\`\`text
$(extract_last_train_line "${STAGE11_LOG}")
\`\`\`

## Stage 1.2

- Suffix: \`${STAGE12_SUFFIX}\`
- Checkpoint: \`${STAGE12_CKPT}\`
- Final train line:

\`\`\`text
$(extract_last_train_line "${STAGE12_LOG}")
\`\`\`

## Stage 2

- Suffix: \`${STAGE2_SUFFIX}\`
- Final best block:

\`\`\`text
$(extract_best_block "${STAGE2_LOG}")
\`\`\`
EOF

echo "[pipeline] summary=${SUMMARY_FILE}"
echo "[pipeline] done"
