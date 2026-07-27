#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

DATASET="${DATASET:-baby}"
EXP_MODE="${EXP_MODE:-mm}"
DATASET_SEED="${DATASET_SEED:-2023}"
MISSING_RATE="${MISSING_RATE:-0.3}"
SEEDS="${SEEDS:-2023}"
METHODS="${METHODS:-mean reliability rum_user rum_full}"
GPUS_CSV="${GPUS_CSV:-4}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
EPOCHS="${EPOCHS:-200}"
EVA_INTERVAL="${EVA_INTERVAL:-1}"
EARLY_STOP="${EARLY_STOP:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-3}"
LR_REC="${LR_REC:-1e-3}"
LR_IMP="${LR_IMP:-2e-4}"
LR_DECODER="${LR_DECODER:-5e-5}"
REG_COEFF="${REG_COEFF:-1e-4}"
MODALITY_BPR_COEFF="${MODALITY_BPR_COEFF:-0.2}"
SAVE="${SAVE:-1}"
TENSORBOARD="${TENSORBOARD:-0}"
STRICT_PROBE_TEST_INTERVAL="${STRICT_PROBE_TEST_INTERVAL:-0}"
RUN_TAG="${RUN_TAG:-baby_fusion_effect_$(date +%Y%m%d_%H%M%S)}"
IMPUTER_CKPT="${IMPUTER_CKPT:-exp_report/baby/stage1_2_baby_seed2023_completion_mm_v2_seed2023_20260426_155227/ckpt/stage1_2_baby_seed2023_completion_mm_v2_seed2023_20260426_155227_imputer_backprop_50_epoch19.pth}"

if [[ ! -f "${IMPUTER_CKPT}" ]]; then
  echo "[fusion-effect] missing imputer checkpoint: ${IMPUTER_CKPT}" >&2
  exit 1
fi

IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "[fusion-effect] GPUS_CSV must contain at least one GPU id" >&2
  exit 1
fi

LOG_DIR="${ROOT_DIR}/exp_report/${DATASET}/fusion_effect_logs"
mkdir -p "${LOG_DIR}"

ACTIVE_JOBS=0
FAILED_JOBS=0
LAUNCH_COUNT=0

wait_for_slot() {
  while (( ACTIVE_JOBS >= MAX_PARALLEL )); do
    if ! wait -n; then
      FAILED_JOBS=$((FAILED_JOBS + 1))
    fi
    ACTIVE_JOBS=$((ACTIVE_JOBS - 1))
  done
}

wait_for_all() {
  while (( ACTIVE_JOBS > 0 )); do
    if ! wait -n; then
      FAILED_JOBS=$((FAILED_JOBS + 1))
    fi
    ACTIVE_JOBS=$((ACTIVE_JOBS - 1))
  done
}

launch_one() {
  local method="$1"
  local seed="$2"
  local gpu="$3"
  local config="configs/baby/stage2_decoder_mm.yaml"
  local suffix="fusion_${method}_${DATASET}_${EXP_MODE}_dseed${DATASET_SEED}_seed${seed}_${RUN_TAG}"
  local log="${LOG_DIR}/${suffix}.launch.log"
  local extra_args=()

  case "${method}" in
    mean)
      config="configs/baby/stage2_decoder_mm.yaml"
      extra_args+=(--fusion_mode mean --completion_gate_mode off)
      ;;
    reliability)
      config="configs/baby/stage2_decoder_mm.yaml"
      extra_args+=(
        --fusion_mode mean
        --completion_gate_mode alignment
        --completion_gate_floor 0.7
        --completion_gate_alignment_center 0.0
        --completion_gate_alignment_temp 0.2
      )
      ;;
    rum_user)
      config="configs/baby/stage2_decoder_rum_mm.yaml"
      extra_args+=(--fusion_mode rum --rum_reliability_coeff 0.0 --rum_match_coeff 1.0)
      ;;
    rum_full)
      config="configs/baby/stage2_decoder_rum_mm.yaml"
      extra_args+=(--fusion_mode rum --rum_reliability_coeff 1.0 --rum_match_coeff 1.0)
      ;;
    *)
      echo "[fusion-effect] unsupported method: ${method}" >&2
      return 2
      ;;
  esac

  echo "[fusion-effect] launch method=${method} seed=${seed} dataset_seed=${DATASET_SEED} gpu=${gpu} suffix=${suffix}"
  env \
    CONFIG="${config}" \
    DATASET="${DATASET}" \
    EXP_MODE="${EXP_MODE}" \
    TRAIN_STAGE=recommender \
    FEATURE_BRIDGE_MODE=raw_decoder \
    GCN_FRONTEND_MODE=original_linear \
    SEED="${seed}" \
    DATASET_SEED="${DATASET_SEED}" \
    DEVICE_ID="${gpu}" \
    EPOCHS="${EPOCHS}" \
    EVA_INTERVAL="${EVA_INTERVAL}" \
    EARLY_STOP="${EARLY_STOP}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    LR="${LR}" \
    LR_REC="${LR_REC}" \
    LR_IMP="${LR_IMP}" \
    LR_DECODER="${LR_DECODER}" \
    MISSING_RATE="${MISSING_RATE}" \
    SAVE="${SAVE}" \
    TENSORBOARD="${TENSORBOARD}" \
    HF_TENSORBOARD_REPO= \
    STRICT_PROBE_TEST_INTERVAL="${STRICT_PROBE_TEST_INTERVAL}" \
    IMPUTER_CKPT="${IMPUTER_CKPT}" \
    SUFFIX="${suffix}" \
    ./run_stage2_baby_recommender_decoder.sh \
      --dataset_seed "${DATASET_SEED}" \
      --reg_coeff "${REG_COEFF}" \
      --modality_bpr_coeff "${MODALITY_BPR_COEFF}" \
      --selection_mode val \
      --recommendation_selection_metric recall \
      --recommendation_selection_topk 20 \
      "${extra_args[@]}" \
      > "${log}" 2>&1
}

echo "[fusion-effect] run_tag=${RUN_TAG}"
echo "[fusion-effect] dataset=${DATASET} exp_mode=${EXP_MODE} missing_rate=${MISSING_RATE}"
echo "[fusion-effect] dataset_seed=${DATASET_SEED} seeds=${SEEDS}"
echo "[fusion-effect] methods=${METHODS}"
echo "[fusion-effect] imputer_ckpt=${IMPUTER_CKPT}"
echo "[fusion-effect] logs=${LOG_DIR}"

for seed in ${SEEDS}; do
  for method in ${METHODS}; do
    wait_for_slot
    gpu="${GPUS[$((LAUNCH_COUNT % ${#GPUS[@]}))]}"
    launch_one "${method}" "${seed}" "${gpu}" &
    ACTIVE_JOBS=$((ACTIVE_JOBS + 1))
    LAUNCH_COUNT=$((LAUNCH_COUNT + 1))
  done
done

wait_for_all
if (( FAILED_JOBS > 0 )); then
  echo "[fusion-effect] ${FAILED_JOBS} jobs failed" >&2
  exit 1
fi

python scripts/summarize_fusion_effect.py \
  --log-dir "${LOG_DIR}" \
  --run-tag "${RUN_TAG}"

echo "[fusion-effect] completed"
