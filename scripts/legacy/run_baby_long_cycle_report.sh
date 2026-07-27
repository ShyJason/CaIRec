#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DEVICE_ID="${DEVICE_ID:-4}"
SAVE="${SAVE:-1}"
EXP_MODE="${EXP_MODE:-mm}"
FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE:-raw_decoder}"
MISSING_RATE="${MISSING_RATE:-0.3}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${ROOT_DIR}/exp_report/baby/long_cycle_reports/${RUN_TAG}_${FEATURE_BRIDGE_MODE}_${EXP_MODE}"

mkdir -p "${REPORT_DIR}"

find_latest_ckpt() {
  local suffix="$1"
  ls -t "${ROOT_DIR}/exp_report/baby/${suffix}/ckpt/"*.pth 2>/dev/null | head -n 1
}

reset_stage_output() {
  local suffix="$1"
  rm -rf "${ROOT_DIR}/exp_report/baby/${suffix}"
}

extract_last_train_line() {
  local log_file="$1"
  grep 'TRAIN:stage' "${log_file}" | tail -n 1 || true
}

extract_best_block() {
  local log_file="$1"
  grep -A5 'best epoch' "${log_file}" || true
}

run_imputation_eval() {
  local ckpt_path="$1"
  local output_json="$2"
  MKL_THREADING_LAYER=GNU python tools/evaluate_imputation_metrics.py \
    --dataset baby \
    --exp_mode "${EXP_MODE}" \
    --device_id "${DEVICE_ID}" \
    --use_gpu 1 \
    --feature_bridge_mode "${FEATURE_BRIDGE_MODE}" \
    --gcn_frontend_mode original_linear \
    --contra_dim 64 \
    --d_beta 32 \
    --missing_rate "${MISSING_RATE}" \
    --ckpt "${ckpt_path}" \
    --split both \
    --json_output "${output_json}"
}

STAGE11_SUFFIX="${STAGE11_SUFFIX:-stage1_1_baby_imputer_param_${RUN_TAG}}"
STAGE12_SUFFIX="${STAGE12_SUFFIX:-stage1_2_baby_imputer_backprop_decoder_${RUN_TAG}}"
STAGE2_SUFFIX="${STAGE2_SUFFIX:-stage2_baby_recommender_decoder_${EXP_MODE}_${RUN_TAG}}"
STAGE3_SUFFIX="${STAGE3_SUFFIX:-stage3_baby_task_aware_imputer_${EXP_MODE}_${RUN_TAG}}"

STAGE11_LOG="${REPORT_DIR}/stage1_1.log"
STAGE12_LOG="${REPORT_DIR}/stage1_2.log"
STAGE2_LOG="${REPORT_DIR}/stage2_${EXP_MODE}.log"
STAGE3_LOG="${REPORT_DIR}/stage3_${EXP_MODE}.log"
SUMMARY_FILE="${REPORT_DIR}/summary.md"

STAGE12_IMPUTE_JSON="${REPORT_DIR}/stage1_2_imputation_metrics.json"
STAGE2_IMPUTE_JSON="${REPORT_DIR}/stage2_imputation_metrics.json"
STAGE3_IMPUTE_JSON="${REPORT_DIR}/stage3_imputation_metrics.json"

echo "[long-cycle] report_dir=${REPORT_DIR}"
echo "[long-cycle] feature_bridge_mode=${FEATURE_BRIDGE_MODE} exp_mode=${EXP_MODE}"
echo "[long-cycle] missing_rate=${MISSING_RATE}"

echo "[long-cycle] Stage 1.1"
reset_stage_output "${STAGE11_SUFFIX}"
{
  DEVICE_ID="${DEVICE_ID}" \
  SAVE="${SAVE}" \
  EXP_MODE="${STAGE11_EXP_MODE:-mm}" \
  MISSING_RATE="${MISSING_RATE}" \
  SUFFIX="${STAGE11_SUFFIX}" \
  EPOCHS="${STAGE11_EPOCHS:-20}" \
  BATCH_SIZE="${BATCH_SIZE:-256}" \
  LR="${STAGE11_LR:-1e-3}" \
  ALPHA_REC="${STAGE11_ALPHA_REC:-1.0}" \
  "${ROOT_DIR}/run_stage1_1_baby_imputer_param.sh"
} 2>&1 | tee "${STAGE11_LOG}"

STAGE11_CKPT="$(find_latest_ckpt "${STAGE11_SUFFIX}")"
if [[ -z "${STAGE11_CKPT}" ]]; then
  echo "[long-cycle] Failed to locate stage 1.1 checkpoint" >&2
  exit 1
fi

echo "[long-cycle] Stage 1.2"
reset_stage_output "${STAGE12_SUFFIX}"
{
  DEVICE_ID="${DEVICE_ID}" \
  SAVE="${SAVE}" \
  EXP_MODE="${STAGE12_EXP_MODE:-mm}" \
  MISSING_RATE="${MISSING_RATE}" \
  SUFFIX="${STAGE12_SUFFIX}" \
  IMPUTER_CKPT="${STAGE11_CKPT}" \
  EPOCHS="${STAGE12_EPOCHS:-30}" \
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
  "${ROOT_DIR}/run_stage1_2_baby_imputer_backprop_decoder_v2.sh"
} 2>&1 | tee "${STAGE12_LOG}"

STAGE12_CKPT="$(find_latest_ckpt "${STAGE12_SUFFIX}")"
if [[ -z "${STAGE12_CKPT}" ]]; then
  echo "[long-cycle] Failed to locate stage 1.2 checkpoint" >&2
  exit 1
fi
run_imputation_eval "${STAGE12_CKPT}" "${STAGE12_IMPUTE_JSON}" > "${REPORT_DIR}/stage1_2_imputation_eval.log"

echo "[long-cycle] Stage 2 (${EXP_MODE})"
reset_stage_output "${STAGE2_SUFFIX}"
{
  DEVICE_ID="${DEVICE_ID}" \
  SAVE="${SAVE}" \
  EXP_MODE="${EXP_MODE}" \
  MISSING_RATE="${MISSING_RATE}" \
  SUFFIX="${STAGE2_SUFFIX}" \
  IMPUTER_CKPT="${STAGE12_CKPT}" \
  FREEZE_IMPUTER="${STAGE2_FREEZE_IMPUTER:-1}" \
  FREEZE_DECODER="${STAGE2_FREEZE_DECODER:-1}" \
  EPOCHS="${STAGE2_EPOCHS:-60}" \
  BATCH_SIZE="${BATCH_SIZE:-256}" \
  LR="${STAGE2_LR:-1e-3}" \
  LR_REC="${STAGE2_LR_REC:-1e-3}" \
  LR_IMP="${STAGE2_LR_IMP:-2e-4}" \
  LR_DECODER="${STAGE2_LR_DECODER:-5e-5}" \
  FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE}" \
  "${ROOT_DIR}/run_stage2_baby_recommender_decoder.sh"
} 2>&1 | tee "${STAGE2_LOG}"

STAGE2_CKPT="$(find_latest_ckpt "${STAGE2_SUFFIX}")"
if [[ -z "${STAGE2_CKPT}" ]]; then
  echo "[long-cycle] Failed to locate stage 2 checkpoint" >&2
  exit 1
fi
run_imputation_eval "${STAGE2_CKPT}" "${STAGE2_IMPUTE_JSON}" > "${REPORT_DIR}/stage2_imputation_eval.log"

echo "[long-cycle] Stage 3 (${EXP_MODE})"
reset_stage_output "${STAGE3_SUFFIX}"
{
  DEVICE_ID="${DEVICE_ID}" \
  SAVE="${SAVE}" \
  EXP_MODE="${EXP_MODE}" \
  MISSING_RATE="${MISSING_RATE}" \
  SUFFIX="${STAGE3_SUFFIX}" \
  CKPT="${STAGE2_CKPT}" \
  EPOCHS="${STAGE3_EPOCHS:-20}" \
  BATCH_SIZE="${BATCH_SIZE:-256}" \
  LR="${STAGE3_LR:-5e-5}" \
  LR_REC="${STAGE3_LR_REC:-5e-5}" \
  LR_IMP="${STAGE3_LR_IMP:-2e-5}" \
  LR_DECODER="${STAGE3_LR_DECODER:-1e-5}" \
  BETA_INTRA="${STAGE3_BETA_INTRA:-0.005}" \
  BETA_INTER="${STAGE3_BETA_INTER:-0.005}" \
  BETA_ITM="${STAGE3_BETA_ITM:-0.005}" \
  BETA_REC="${STAGE3_BETA_REC:-0.001}" \
  BETA_DECODE="${STAGE3_BETA_DECODE:-0.001}" \
  GAMMA_ALIGN="${STAGE3_GAMMA_ALIGN:-0.0}" \
  GAMMA_DISTILL="${STAGE3_GAMMA_DISTILL:-0.1}" \
  FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE}" \
  "${ROOT_DIR}/scripts/legacy/run_stage3_baby_task_aware_imputer.sh"
} 2>&1 | tee "${STAGE3_LOG}"

STAGE3_CKPT="$(find_latest_ckpt "${STAGE3_SUFFIX}")"
if [[ -z "${STAGE3_CKPT}" ]]; then
  echo "[long-cycle] Failed to locate stage 3 checkpoint" >&2
  exit 1
fi
run_imputation_eval "${STAGE3_CKPT}" "${STAGE3_IMPUTE_JSON}" > "${REPORT_DIR}/stage3_imputation_eval.log"

cat > "${SUMMARY_FILE}" <<EOF
# Long-Cycle Pipeline Results

- Run tag: \`${RUN_TAG}\`
- Feature bridge mode: \`${FEATURE_BRIDGE_MODE}\`
- Stage exp_mode: \`${EXP_MODE}\`
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

- Imputation metrics JSON: \`${STAGE12_IMPUTE_JSON}\`

## Stage 2

- Suffix: \`${STAGE2_SUFFIX}\`
- Checkpoint: \`${STAGE2_CKPT}\`
- Final best block:

\`\`\`text
$(extract_best_block "${STAGE2_LOG}")
\`\`\`

- Imputation metrics JSON: \`${STAGE2_IMPUTE_JSON}\`

## Stage 3

- Suffix: \`${STAGE3_SUFFIX}\`
- Checkpoint: \`${STAGE3_CKPT}\`
- Final best block:

\`\`\`text
$(extract_best_block "${STAGE3_LOG}")
\`\`\`

- Imputation metrics JSON: \`${STAGE3_IMPUTE_JSON}\`
EOF

echo "[long-cycle] summary=${SUMMARY_FILE}"
echo "[long-cycle] done"
