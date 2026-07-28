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

export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"

DATASET="${DATASET:-baby}"
EXP_MODE="${EXP_MODE:-ff}"
DEVICE_ID="${DEVICE_ID:-4}"
USE_GPU="${USE_GPU:-1}"
SEED="${SEED:-2023}"
DATASET_SEED="${DATASET_SEED:-0}"
EVA_INTERVAL="${EVA_INTERVAL:-1}"
EARLY_STOP="${EARLY_STOP:-}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-3}"
LR_REC="${LR_REC:-}"
LR_IMP="${LR_IMP:-}"
LR_DECODER="${LR_DECODER:-}"
TRAIN_STAGE="${TRAIN_STAGE:-joint}"
if [[ "${DATASET}" == "clothing" && "${TRAIN_STAGE}" == "imputer_backprop" ]]; then
  EPOCHS="${EPOCHS:-50}"
else
  EPOCHS="${EPOCHS:-1}"
fi
CKPT="${CKPT:-}"
CKPT_START_EPOCH="${CKPT_START_EPOCH:-}"
PROJECTION_CKPT="${PROJECTION_CKPT:-}"
IMPUTER_CKPT="${IMPUTER_CKPT:-}"
FREEZE_IMPUTER="${FREEZE_IMPUTER:--1}"
FREEZE_RECOMMENDER="${FREEZE_RECOMMENDER:--1}"
FREEZE_DECODER="${FREEZE_DECODER:-0}"
CONTRA_DIM="${CONTRA_DIM:-64}"
D_BETA="${D_BETA:-32}"
TAU1="${TAU1:-0.1}"
TAU2="${TAU2:-0.1}"
LAMBDA_ITM="${LAMBDA_ITM:-0.1}"
ITM_TEMP="${ITM_TEMP:-0.07}"
ITM_NUM_HEADS="${ITM_NUM_HEADS:-4}"
EMA_ETA="${EMA_ETA:-0.01}"
DISABLE_IMPUTATION="${DISABLE_IMPUTATION:-0}"
FEATURE_BRIDGE_MODE="${FEATURE_BRIDGE_MODE:-raw_decoder}"
GCN_FRONTEND_MODE="${GCN_FRONTEND_MODE:-original_linear}"
PROMRL_PROJECTION_MODE="${PROMRL_PROJECTION_MODE:-learned}"
GENERATIVE_UPDATE_MODE="${GENERATIVE_UPDATE_MODE:-em}"
MODAL_FEATURE_OVERRIDE_DIR="${MODAL_FEATURE_OVERRIDE_DIR:-${FEATURE_DIR:-}}"
MODAL_FEATURE_TRAIN_DIR="${MODAL_FEATURE_TRAIN_DIR:-}"
MODAL_FEATURE_EVAL_DIR="${MODAL_FEATURE_EVAL_DIR:-}"
MODAL_FEATURE_IMAGE_FILE="${MODAL_FEATURE_IMAGE_FILE:-agg_image_items.npy}"
MODAL_FEATURE_TEXT_FILE="${MODAL_FEATURE_TEXT_FILE:-agg_text_items.npy}"
MODAL_FEATURE_AUDIO_FILE="${MODAL_FEATURE_AUDIO_FILE:-agg_audio_items.npy}"
MODAL_FEATURE_VIDEO_FILE="${MODAL_FEATURE_VIDEO_FILE:-agg_video_items.npy}"
MODAL_FEATURE_MASK_SOURCE="${MODAL_FEATURE_MASK_SOURCE:-nonzero}"
MODAL_FEATURE_IMAGE_MASK_FILE="${MODAL_FEATURE_IMAGE_MASK_FILE:-image_observed_mask.npy}"
MODAL_FEATURE_TEXT_MASK_FILE="${MODAL_FEATURE_TEXT_MASK_FILE:-text_observed_mask.npy}"
MODAL_FEATURE_AUDIO_MASK_FILE="${MODAL_FEATURE_AUDIO_MASK_FILE:-audio_observed_mask.npy}"
MODAL_FEATURE_VIDEO_MASK_FILE="${MODAL_FEATURE_VIDEO_MASK_FILE:-video_observed_mask.npy}"
MODAL_FEATURE_OVERRIDE_IS_COMPLETED="${MODAL_FEATURE_OVERRIDE_IS_COMPLETED:-0}"
ALPHA_INTRA="${ALPHA_INTRA:-1.0}"
ALPHA_INTER="${ALPHA_INTER:-1.0}"
ALPHA_ITM="${ALPHA_ITM:-1.0}"
STRUCTURE_LOSS_VARIANT="${STRUCTURE_LOSS_VARIANT:-original}"
if [[ "${DATASET}" == "clothing" && "${TRAIN_STAGE}" == "imputer_backprop" ]]; then
  ALPHA_REC="${ALPHA_REC:-1.0}"
else
  ALPHA_REC="${ALPHA_REC:-0.1}"
fi
ALPHA_DECODE="${ALPHA_DECODE:-0.0}"
BETA_INTRA="${BETA_INTRA:-0.05}"
BETA_INTER="${BETA_INTER:-0.05}"
BETA_ITM="${BETA_ITM:-0.05}"
BETA_REC="${BETA_REC:-0.01}"
BETA_DECODE="${BETA_DECODE:-0.01}"
GAMMA_ALIGN="${GAMMA_ALIGN:-0.0}"
ADAPTER_ALIGN_PSEUDO_RATIO="${ADAPTER_ALIGN_PSEUDO_RATIO:-1.0}"
GAMMA_DISTILL="${GAMMA_DISTILL:-0.0}"
RECOMMENDER_ALLOW_MODAL_GRAD="${RECOMMENDER_ALLOW_MODAL_GRAD:-0}"
MLP_IMPUTE_TRAIN_PROJECTION="${MLP_IMPUTE_TRAIN_PROJECTION:-0}"
SEMANTIC_IMPUTE_CONFIDENCE_FLOOR="${SEMANTIC_IMPUTE_CONFIDENCE_FLOOR:-0.2}"
ALPHA_MISSING_SHARED="${ALPHA_MISSING_SHARED:-}"
ALPHA_MISSING_DECODE="${ALPHA_MISSING_DECODE:-}"
BETA_MISSING_SHARED="${BETA_MISSING_SHARED:-0.0}"
BETA_MISSING_DECODE="${BETA_MISSING_DECODE:-0.0}"
MISSING_RATE="${MISSING_RATE:-0.3}"
TRAIN_MISSING_MODALITY="${TRAIN_MISSING_MODALITY:-random}"
EVAL_MISSING_RATE="${EVAL_MISSING_RATE:-}"
MISSING_MASK_PROTOCOL="${MISSING_MASK_PROTOCOL:-i3}"
IMPUTATION_VAL_RATE="${IMPUTATION_VAL_RATE:-0.0}"
IMPUTATION_SELECTION_POLICY="${IMPUTATION_SELECTION_POLICY:-legacy}"
IMPUTATION_SELECTION_SPLIT="${IMPUTATION_SELECTION_SPLIT:-train}"
IMPUTATION_SELECTION_METRIC="${IMPUTATION_SELECTION_METRIC:-mse}"
ORACLE_MISSING_SUPERVISION="${ORACLE_MISSING_SUPERVISION:-0}"
SAVE="${SAVE:-0}"
SAVE_ALL_EPOCHS="${SAVE_ALL_EPOCHS:-0}"
SUFFIX="${SUFFIX:-demo_${DATASET}_${EXP_MODE}_${TRAIN_STAGE}}"
SELECTION_MODE="${SELECTION_MODE:-val}"
EVALUATION_PROTOCOL="${EVALUATION_PROTOCOL:-strict}"
STRICT_PROBE_TEST_INTERVAL="${STRICT_PROBE_TEST_INTERVAL:-0}"
AUTO_LOG_FILE="${AUTO_LOG_FILE:-1}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/exp_report/${DATASET}/${SUFFIX}/log}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log}"
TENSORBOARD="${TENSORBOARD:-1}"
HF_TENSORBOARD_REPO="${HF_TENSORBOARD_REPO:-}"
HF_TOKEN="${HF_TOKEN:-}"
HF_COMMIT_EVERY="${HF_COMMIT_EVERY:-5}"
CONFIG="${CONFIG:-}"

if [[ "${AUTO_LOG_FILE}" == "1" ]]; then
  mkdir -p "${LOG_DIR}"
  exec > >(tee -a "${LOG_FILE}") 2>&1
fi

echo "[demo] root=${ROOT_DIR}"
echo "[demo] dataset=${DATASET} exp_mode=${EXP_MODE} device_id=${DEVICE_ID}"
echo "[demo] seed=${SEED} dataset_seed=${DATASET_SEED}"
echo "[demo] stage=${TRAIN_STAGE} epochs=${EPOCHS} batch_size=${BATCH_SIZE} lambda_itm=${LAMBDA_ITM} structure_loss_variant=${STRUCTURE_LOSS_VARIANT} disable_imputation=${DISABLE_IMPUTATION} bridge=${FEATURE_BRIDGE_MODE} frontend=${GCN_FRONTEND_MODE} train_missing_modality=${TRAIN_MISSING_MODALITY} missing_rate=${MISSING_RATE} eval_missing_rate=${EVAL_MISSING_RATE:-config/default} selection=${SELECTION_MODE} eval_protocol=${EVALUATION_PROTOCOL} strict_probe_test_interval=${STRICT_PROBE_TEST_INTERVAL}"
echo "[demo] log_file=${LOG_FILE}"
echo "[demo] tensorboard=${TENSORBOARD} hf_tensorboard_repo=${HF_TENSORBOARD_REPO:-<none>}"
echo "[demo] config=${CONFIG:-<none>}"

cmd=(
  "${PYTHON_BIN}" main.py
  --dataset "${DATASET}"
  --exp_mode "${EXP_MODE}"
  --use_gpu "${USE_GPU}"
  --device_id "${DEVICE_ID}"
  --seed "${SEED}"
  --dataset_seed "${DATASET_SEED}"
  --epoch "${EPOCHS}"
  --eva_interval "${EVA_INTERVAL}"
  --batch_size "${BATCH_SIZE}"
  --lr "${LR}"
  --train_stage "${TRAIN_STAGE}"
  --freeze_imputer "${FREEZE_IMPUTER}"
  --freeze_recommender "${FREEZE_RECOMMENDER}"
  --freeze_decoder "${FREEZE_DECODER}"
  --contra_dim "${CONTRA_DIM}"
  --d_beta "${D_BETA}"
  --tau1 "${TAU1}"
  --tau2 "${TAU2}"
  --lambda_itm "${LAMBDA_ITM}"
  --itm_temp "${ITM_TEMP}"
  --itm_num_heads "${ITM_NUM_HEADS}"
  --ema_eta "${EMA_ETA}"
  --disable_imputation "${DISABLE_IMPUTATION}"
  --feature_bridge_mode "${FEATURE_BRIDGE_MODE}"
  --gcn_frontend_mode "${GCN_FRONTEND_MODE}"
  --promrl_projection_mode "${PROMRL_PROJECTION_MODE}"
  --generative_update_mode "${GENERATIVE_UPDATE_MODE}"
  --alpha_intra "${ALPHA_INTRA}"
  --alpha_inter "${ALPHA_INTER}"
  --alpha_itm "${ALPHA_ITM}"
  --structure_loss_variant "${STRUCTURE_LOSS_VARIANT}"
  --alpha_rec "${ALPHA_REC}"
  --alpha_decode "${ALPHA_DECODE}"
  --beta_intra "${BETA_INTRA}"
  --beta_inter "${BETA_INTER}"
  --beta_itm "${BETA_ITM}"
  --beta_rec "${BETA_REC}"
  --beta_decode "${BETA_DECODE}"
  --gamma_align "${GAMMA_ALIGN}"
  --adapter_align_pseudo_ratio "${ADAPTER_ALIGN_PSEUDO_RATIO}"
  --gamma_distill "${GAMMA_DISTILL}"
  --recommender_allow_modal_grad "${RECOMMENDER_ALLOW_MODAL_GRAD}"
  --train_missing_modality "${TRAIN_MISSING_MODALITY}"
  --missing_rate "${MISSING_RATE}"
  --missing_mask_protocol "${MISSING_MASK_PROTOCOL}"
  --imputation_val_rate "${IMPUTATION_VAL_RATE}"
  --imputation_selection_policy "${IMPUTATION_SELECTION_POLICY}"
  --imputation_selection_split "${IMPUTATION_SELECTION_SPLIT}"
  --imputation_selection_metric "${IMPUTATION_SELECTION_METRIC}"
  --selection_mode "${SELECTION_MODE}"
  --evaluation_protocol "${EVALUATION_PROTOCOL}"
  --strict_probe_test_interval "${STRICT_PROBE_TEST_INTERVAL}"
  --tensorboard "${TENSORBOARD}"
  --save "${SAVE}"
  --save_all_epochs "${SAVE_ALL_EPOCHS}"
  --suffix "${SUFFIX}"
)

if [[ -n "${CONFIG}" ]]; then
  cmd+=(--config "${CONFIG}")
fi

if [[ -n "${LR_REC}" ]]; then
  cmd+=(--lr_rec "${LR_REC}")
fi

if [[ -n "${LR_IMP}" ]]; then
  cmd+=(--lr_imp "${LR_IMP}")
fi

if [[ -n "${LR_DECODER}" ]]; then
  cmd+=(--lr_decoder "${LR_DECODER}")
fi

if [[ -n "${CKPT}" ]]; then
  cmd+=(--ckpt "${CKPT}")
fi

if [[ -n "${CKPT_START_EPOCH}" ]]; then
  cmd+=(--ckpt_start_epoch "${CKPT_START_EPOCH}")
fi

if [[ -n "${EARLY_STOP}" ]]; then
  cmd+=(--early_stop "${EARLY_STOP}")
fi

if [[ -n "${IMPUTER_CKPT}" ]]; then
  cmd+=(--imputer_ckpt "${IMPUTER_CKPT}")
fi

if [[ -n "${PROJECTION_CKPT}" ]]; then
  cmd+=(--projection_ckpt "${PROJECTION_CKPT}")
fi

if [[ -n "${EVAL_MISSING_RATE}" ]]; then
  cmd+=(--eval_missing_rate "${EVAL_MISSING_RATE}")
fi

if [[ -n "${MODAL_FEATURE_OVERRIDE_DIR}" ]]; then
  cmd+=(
    --modal_feature_override_dir "${MODAL_FEATURE_OVERRIDE_DIR}"
    --modal_feature_image_file "${MODAL_FEATURE_IMAGE_FILE}"
    --modal_feature_text_file "${MODAL_FEATURE_TEXT_FILE}"
    --modal_feature_audio_file "${MODAL_FEATURE_AUDIO_FILE}"
    --modal_feature_video_file "${MODAL_FEATURE_VIDEO_FILE}"
    --modal_feature_mask_source "${MODAL_FEATURE_MASK_SOURCE}"
    --modal_feature_image_mask_file "${MODAL_FEATURE_IMAGE_MASK_FILE}"
    --modal_feature_text_mask_file "${MODAL_FEATURE_TEXT_MASK_FILE}"
    --modal_feature_audio_mask_file "${MODAL_FEATURE_AUDIO_MASK_FILE}"
    --modal_feature_video_mask_file "${MODAL_FEATURE_VIDEO_MASK_FILE}"
    --modal_feature_override_is_completed "${MODAL_FEATURE_OVERRIDE_IS_COMPLETED}"
  )
fi

if [[ -n "${MODAL_FEATURE_TRAIN_DIR}" ]]; then
  cmd+=(--modal_feature_train_dir "${MODAL_FEATURE_TRAIN_DIR}")
fi

if [[ -n "${MODAL_FEATURE_EVAL_DIR}" ]]; then
  cmd+=(--modal_feature_eval_dir "${MODAL_FEATURE_EVAL_DIR}")
fi

if [[ -n "${HF_TENSORBOARD_REPO}" ]]; then
  cmd+=(--hf_tensorboard_repo "${HF_TENSORBOARD_REPO}")
fi

if [[ -n "${HF_TOKEN}" ]]; then
  cmd+=(--hf_token "${HF_TOKEN}")
fi

if [[ -n "${HF_COMMIT_EVERY}" ]]; then
  cmd+=(--hf_commit_every "${HF_COMMIT_EVERY}")
fi

cmd+=("$@")
"${cmd[@]}"
