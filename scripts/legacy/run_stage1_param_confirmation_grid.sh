#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DEVICE_ID="${DEVICE_ID:-4}"
SAVE="${SAVE:-1}"
PIPELINE_SAVE=1
EXP_MODE="${EXP_MODE:-mm}"
FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE:-raw_decoder}"
MISSING_RATE="${MISSING_RATE:-0.3}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${ROOT_DIR}/exp_report/baby/stage1_param_grid_reports/${RUN_TAG}_${FEATURE_BRIDGE_MODE}_${EXP_MODE}"

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

extract_stage1_metrics() {
  local log_file="$1"
  python - "$log_file" <<'PY'
import re, sys
path = sys.argv[1]
text = open(path, 'r', encoding='utf-8').read()
matches = re.findall(
    r'imputation_train_mse = ([0-9.]+), imputation_train_cosine = ([0-9.]+), '
    r'imputation_test_mse = ([0-9.]+), imputation_test_cosine = ([0-9.]+)',
    text,
)
if matches:
    print("\t".join(matches[-1]))
else:
    print("\t".join(["NA"] * 4))
PY
}

extract_stage2_metrics() {
  local log_file="$1"
  python - "$log_file" <<'PY'
import re, sys
path = sys.argv[1]
text = open(path, 'r', encoding='utf-8').read()
best_idx = text.rfind('best epoch')
if best_idx == -1:
    print("\t".join(["NA"] * 4))
    raise SystemExit
block = text[best_idx:]
def get(metric, k):
    m = re.search(rf'{metric}@{k} = ([0-9.]+)', block)
    return m.group(1) if m else "NA"
print("\t".join([
    get('recall', 20),
    get('ndcg', 20),
    get('recall', 10),
    get('ndcg', 10),
]))
PY
}

append_summary_row() {
  local name="$1"
  local alpha_rec="$2"
  local alpha_decode="$3"
  local s12_log="$4"
  local s2_log="$5"

  local stage1_metrics
  local stage2_metrics
  stage1_metrics="$(extract_stage1_metrics "${s12_log}")"
  stage2_metrics="$(extract_stage2_metrics "${s2_log}")"

  IFS=$'\t' read -r train_mse train_cosine test_mse test_cosine <<<"${stage1_metrics}"
  IFS=$'\t' read -r recall20 ndcg20 recall10 ndcg10 <<<"${stage2_metrics}"

  cat >> "${SUMMARY_FILE}" <<EOF
| ${name} | ${alpha_rec} | ${alpha_decode} | ${train_mse} | ${test_mse} | ${train_cosine} | ${test_cosine} | ${recall20} | ${ndcg20} | ${recall10} | ${ndcg10} |
EOF
}

STAGE11_SUFFIX="${STAGE11_SUFFIX:-stage1_1_baby_imputer_param_${RUN_TAG}_base}"
STAGE11_LOG="${REPORT_DIR}/stage1_1.log"
SUMMARY_FILE="${REPORT_DIR}/summary.md"

echo "[grid] report_dir=${REPORT_DIR}"
echo "[grid] feature_bridge_mode=${FEATURE_BRIDGE_MODE} exp_mode=${EXP_MODE}"
echo "[grid] missing_rate=${MISSING_RATE}"
if [[ "${SAVE}" != "1" ]]; then
  echo "[grid] overriding SAVE=${SAVE} -> 1 because checkpoints are required between stages"
fi

echo "[grid] Stage 1.1 (shared base)"
reset_stage_output "${STAGE11_SUFFIX}"
{
  DEVICE_ID="${DEVICE_ID}" \
  SAVE="${PIPELINE_SAVE}" \
  EXP_MODE="${EXP_MODE}" \
  MISSING_RATE="${MISSING_RATE}" \
  SUFFIX="${STAGE11_SUFFIX}" \
  EPOCHS="${STAGE11_EPOCHS:-5}" \
  BATCH_SIZE="${BATCH_SIZE:-256}" \
  LR="${STAGE11_LR:-1e-3}" \
  ALPHA_REC="${STAGE11_ALPHA_REC:-1.0}" \
  "${ROOT_DIR}/run_stage1_1_baby_imputer_param.sh"
} 2>&1 | tee "${STAGE11_LOG}"

STAGE11_CKPT="$(find_latest_ckpt "${STAGE11_SUFFIX}")"
if [[ -z "${STAGE11_CKPT}" ]]; then
  echo "[grid] Failed to locate stage 1.1 checkpoint" >&2
  exit 1
fi

cat > "${SUMMARY_FILE}" <<EOF
# Stage1 Parameter Confirmation Grid

- Run tag: \`${RUN_TAG}\`
- Feature bridge mode: \`${FEATURE_BRIDGE_MODE}\`
- Exp mode: \`${EXP_MODE}\`
- Missing rate: \`${MISSING_RATE}\`
- Shared stage1.1 checkpoint: \`${STAGE11_CKPT}\`

## Shared Stage 1.1

\`\`\`text
$(extract_last_train_line "${STAGE11_LOG}")
\`\`\`

## Results

| Combo | alpha_rec | alpha_decode | train_mse | test_mse | train_cosine | test_cosine | Recall@20 | NDCG@20 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
EOF

COMBO_NAMES=("A" "B" "C" "D")
COMBO_ALPHA_REC=("0.1" "0.3" "0.5" "0.5")
COMBO_ALPHA_DECODE=("1.0" "0.5" "0.2" "0.1")

for idx in "${!COMBO_NAMES[@]}"; do
  name="${COMBO_NAMES[$idx]}"
  alpha_rec="${COMBO_ALPHA_REC[$idx]}"
  alpha_decode="${COMBO_ALPHA_DECODE[$idx]}"

  stage12_suffix="stage1_2_baby_imputer_backprop_${RUN_TAG}_${name}"
  stage2_suffix="stage2_baby_recommender_decoder_${EXP_MODE}_${RUN_TAG}_${name}"
  stage12_log="${REPORT_DIR}/stage1_2_${name}.log"
  stage2_log="${REPORT_DIR}/stage2_${name}.log"

  echo "[grid] Combo ${name}: alpha_rec=${alpha_rec}, alpha_decode=${alpha_decode}"

  reset_stage_output "${stage12_suffix}"
  {
    DEVICE_ID="${DEVICE_ID}" \
    SAVE="${PIPELINE_SAVE}" \
    EXP_MODE="${EXP_MODE}" \
    MISSING_RATE="${MISSING_RATE}" \
    SUFFIX="${stage12_suffix}" \
    IMPUTER_CKPT="${STAGE11_CKPT}" \
    EPOCHS="${STAGE12_EPOCHS:-5}" \
    BATCH_SIZE="${BATCH_SIZE:-256}" \
    LR="${STAGE12_LR:-5e-4}" \
    LR_IMP="${STAGE12_LR_IMP:-5e-4}" \
    LR_DECODER="${STAGE12_LR_DECODER:-2e-4}" \
    ALPHA_INTRA="${STAGE12_ALPHA_INTRA:-1.0}" \
    ALPHA_INTER="${STAGE12_ALPHA_INTER:-1.0}" \
    ALPHA_ITM="${STAGE12_ALPHA_ITM:-1.0}" \
    ALPHA_REC="${alpha_rec}" \
    ALPHA_DECODE="${alpha_decode}" \
    FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE}" \
    "${ROOT_DIR}/run_stage1_2_baby_imputer_backprop_decoder_v2.sh"
  } 2>&1 | tee "${stage12_log}"

  stage12_ckpt="$(find_latest_ckpt "${stage12_suffix}")"
  if [[ -z "${stage12_ckpt}" ]]; then
    echo "[grid] Failed to locate stage 1.2 checkpoint for combo ${name}" >&2
    exit 1
  fi

  reset_stage_output "${stage2_suffix}"
  {
    DEVICE_ID="${DEVICE_ID}" \
    SAVE="${PIPELINE_SAVE}" \
    EXP_MODE="${EXP_MODE}" \
    MISSING_RATE="${MISSING_RATE}" \
    SUFFIX="${stage2_suffix}" \
    IMPUTER_CKPT="${stage12_ckpt}" \
    FREEZE_IMPUTER="${STAGE2_FREEZE_IMPUTER:-1}" \
    FREEZE_DECODER="${STAGE2_FREEZE_DECODER:-1}" \
    EPOCHS="${STAGE2_EPOCHS:-30}" \
    BATCH_SIZE="${BATCH_SIZE:-256}" \
    LR="${STAGE2_LR:-1e-3}" \
    LR_REC="${STAGE2_LR_REC:-1e-3}" \
    LR_IMP="${STAGE2_LR_IMP:-2e-4}" \
    LR_DECODER="${STAGE2_LR_DECODER:-5e-5}" \
    FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE}" \
    "${ROOT_DIR}/run_stage2_baby_recommender_decoder.sh"
  } 2>&1 | tee "${stage2_log}"

  append_summary_row "${name}" "${alpha_rec}" "${alpha_decode}" "${stage12_log}" "${stage2_log}"
done

echo "[grid] summary=${SUMMARY_FILE}"
echo "[grid] done"
