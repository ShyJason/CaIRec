#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
RUN_TAG="${RUN_TAG:-clothing_mr0p3_smore_native_missing_item_embeds_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-exp_report/clothing/smore_native_missing_item_embeds/${RUN_TAG}}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

CONFIG="${CONFIG:-configs/clothing/stage2_decoder_mm_itemgraph_completed.yaml}"
FEATURE_DIR="${FEATURE_DIR:-/home/ruiyuliu/projects/baselines/SMORE/experiment_logs/smore_native_missing_input_compact_clothing_mr0p3_20260702/native_missing_mr0p3}"
EPOCHS="${EPOCHS:-200}"
EARLY_STOP="${EARLY_STOP:-10000}"
MEM_FREE_THRESHOLD="${MEM_FREE_THRESHOLD:-45000}"
POLL_SECONDS="${POLL_SECONDS:-60}"

log() {
  echo "[$(date -Is)] $*" | tee -a "${OUT_DIR}/launcher.log"
}

gpu_has_room() {
  local used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" | tr -d ' ')"
  [[ "${used}" =~ ^[0-9]+$ ]] && (( used < MEM_FREE_THRESHOLD ))
}

run_frontend() {
  local frontend="$1"
  local suffix
  local log_path

  suffix="stage2_clothing_mr0p3_smore_native_missing_item_embeds_${frontend}_no_completion_no_ii_${RUN_TAG}"
  log_path="${LOG_DIR}/native_missing_item_embeds_${frontend}.log"

  log "start native missing item_embeds ${frontend}; feature_dir=${FEATURE_DIR}"
  .venv/bin/python -u main.py \
    --config "${CONFIG}" \
    --device_id "${GPU}" \
    --dataset clothing \
    --exp_mode mm \
    --train_stage recommender \
    --missing_rate 0.3 \
    --eval_missing_rate 0.5 \
    --seed 2023 \
    --dataset_seed 0 \
    --suffix "${suffix}" \
    --epoch "${EPOCHS}" \
    --early_stop "${EARLY_STOP}" \
    --eva_interval 1 \
    --batch_size 2048 \
    --lr 0.01 \
    --lr_rec 0.01 \
    --lr_imp 0.0002 \
    --lr_decoder 0.00005 \
    --freeze_imputer 1 \
    --freeze_decoder 1 \
    --recommender_allow_modal_grad 0 \
    --feature_bridge_mode raw_decoder \
    --gcn_frontend_mode "${frontend}" \
    --disable_imputation 1 \
    --modality_bpr_coeff 1.0 \
    --reg_coeff 0.01 \
    --evaluation_protocol strict \
    --selection_mode val \
    --recommendation_selection_metric recall \
    --recommendation_selection_topk 20 \
    --rec_neighbor_cl_weight 0.005 \
    --rec_neighbor_cl_temp 0.2 \
    --rec_neighbor_cl_bank_size 256 \
    --item_graph_kind none \
    --item_graph_topk 20 \
    --item_graph_norm rw \
    --item_graph_cf_weight 0.0 \
    --item_graph_image_weight 0.0 \
    --item_graph_text_weight 0.0 \
    --item_graph_audio_weight 0.0 \
    --item_graph_modal_alpha 0.0 \
    --item_graph_modal_layers 1 \
    --item_graph_modal_target all \
    --modal_feature_override_dir "${FEATURE_DIR}" \
    --modal_feature_image_file image_item_embeds.npy \
    --modal_feature_text_file text_item_embeds.npy \
    --tensorboard 0 \
    --save 1 \
    --topk "[10, 20, 30, 40, 50]" \
    > "${log_path}" 2>&1
  log "done native missing item_embeds ${frontend}; log=${log_path}"
}

log "run_tag=${RUN_TAG}"
log "out_dir=${OUT_DIR}"
log "feature_dir=${FEATURE_DIR}"
while ! gpu_has_room; do
  log "GPU ${GPU} above memory threshold; waiting ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done

run_frontend identity
run_frontend original_linear
log "all frontends finished"
