#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PHYSICAL_GPU="${PHYSICAL_GPU:-${1:-0}}"
CHECK_ONLY="${CHECK_ONLY:-0}"
RUN_TAG="${RUN_TAG:-repro_best_sports_20260719_$(date -u +%Y%m%d_%H%M%S)}"
CONFIG="configs/clothing/mainline_mr0p1.yaml"
IMPUTER_CKPT="exp_report/sports/sports_unified_mr0p5_stage12_fixed_repro_20260718_stage1_2_completion/ckpt/sports_unified_mr0p5_stage12_fixed_repro_20260718_stage1_2_completion_imputer_backprop_50_epoch49.pth"
PAYLOAD="Data/sports/unified_missing_items_mr0.5_seed2023.npy"
OUT_DIR="exp_report/sports/reproduce_best_20260719/${RUN_TAG}"
LOG_FILE="${OUT_DIR}/${RUN_TAG}.launch.log"

verify_sha256() {
  local expected="$1"
  local path="$2"
  test -f "${path}" || { echo "missing required file: ${path}" >&2; exit 1; }
  echo "${expected}  ${path}" | sha256sum --check --status || {
    echo "sha256 mismatch: ${path}" >&2
    exit 1
  }
}

verify_sha256 ef652d196efbe267b574f47ff9154da59c992e6c20a0d570804e20fe65114772 "${CONFIG}"
verify_sha256 95e388b214f75fd159de7b28c9902a60604eca2c804370324c8738d9d7db6086 "${IMPUTER_CKPT}"
verify_sha256 421816fbeaa65cb6323f9f42e209a52f5688401525ba75bb6c902789580aaabe "${PAYLOAD}"
test -f Data/sports/sports.inter
test -f Data/sports/image_feat.npy
test -f Data/sports/text_feat.npy

echo "dataset=sports protocol=unified_static train_mr=0.5 eval_mr=0.5 seed=2023 payload_seed=2023"
echo "reliability=both scale=50 fusion=posterior_reliability"
echo "checkpoint=${IMPUTER_CKPT}"
if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "preflight passed; no training started"
  exit 0
fi

mkdir -p "${OUT_DIR}"
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONUNBUFFERED=1
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"

env \
  CONFIG="${CONFIG}" DATASET=sports EXP_MODE=mm DEVICE_ID=0 \
  SEED=2023 DATASET_SEED=0 \
  EPOCHS=400 EVA_INTERVAL=1 EARLY_STOP=30 BATCH_SIZE=2048 \
  LR=0.005 LR_REC=0.005 LR_IMP=0.0002 \
  TRAIN_STAGE=recommender FEATURE_BRIDGE_MODE=decoupled_latent \
  GCN_FRONTEND_MODE=original_linear PROMRL_PROJECTION_MODE=learned \
  GENERATIVE_UPDATE_MODE=em \
  MISSING_RATE=0.5 EVAL_MISSING_RATE=0.5 \
  MISSING_MASK_PROTOCOL=unified_static TRAIN_MISSING_MODALITY=random \
  IMPUTATION_VAL_RATE=0.0 GAMMA_ALIGN=0.00125 ADAPTER_ALIGN_PSEUDO_RATIO=1.0 \
  DISABLE_IMPUTATION=0 IMPUTER_CKPT="${IMPUTER_CKPT}" CKPT= \
  FREEZE_IMPUTER=1 FREEZE_DECODER=1 FREEZE_RECOMMENDER=-1 \
  ALPHA_DECODE=0.0 SELECTION_MODE=val EVALUATION_PROTOCOL=strict \
  STRICT_PROBE_TEST_INTERVAL=0 TENSORBOARD=0 SAVE=1 SAVE_ALL_EPOCHS=0 \
  AUTO_LOG_FILE=0 SUFFIX="${RUN_TAG}" \
  ./run_demo_itm.sh \
    --unified_payload_seed 2023 \
    --imputer_ckpt "${IMPUTER_CKPT}" \
    --item_graph_kind modality_completed \
    --item_graph_missing_scope train \
    --item_graph_feature_space shared \
    --item_graph_topk 10 --item_graph_norm rw \
    --item_graph_cf_weight 0.4 --item_graph_cf_scale raw \
    --item_graph_cf_power 0.5 --item_graph_cf_clip 3.0 \
    --item_graph_image_weight 0.6 --item_graph_text_weight 0.6 \
    --item_graph_audio_weight 0.0 --item_graph_video_weight 0.0 \
    --item_graph_modal_alpha 0.25 --item_graph_modal_layers 1 \
    --item_graph_modal_target all \
    --rec_neighbor_cl_weight 0.005 --rec_neighbor_cl_temp 0.2 \
    --rec_neighbor_cl_bank_size 256 \
    --reg_coeff 0.01 --modality_bpr_coeff 1.0 \
    --fusion_mode posterior_reliability \
    --posterior_reliability_scope both \
    --posterior_reliability_scale 50 \
    --posterior_reliability_floor 0.0 \
    --freeze_imputer 1 --freeze_decoder 1 --freeze_recommender -1 \
    --selection_mode val --recommendation_selection_metric recall \
    --recommendation_selection_topk 20 --evaluation_protocol strict \
    --strict_probe_test_interval 0 --topk "[10, 20, 30, 40, 50]" \
    2>&1 | tee "${LOG_FILE}"
