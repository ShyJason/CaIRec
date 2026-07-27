#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
RUN_TAG="${RUN_TAG:-clothing_mr0p3_smore_agg_identity_no_completion_no_ii_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-exp_report/clothing/smore_agg_identity_no_completion_no_ii/${RUN_TAG}}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

CONFIG="${CONFIG:-configs/clothing/stage2_decoder_mm_itemgraph_completed.yaml}"
SMORE_DIR="${SMORE_DIR:-/home/ruiyuliu/projects/baselines/SMORE/experiment_logs/smore_pretrain_agg_compact_clothing_mr0p3_20260702}"
EPOCHS="${EPOCHS:-200}"
EARLY_STOP="${EARLY_STOP:-10000}"
MEM_FREE_THRESHOLD="${MEM_FREE_THRESHOLD:-1000}"
POLL_SECONDS="${POLL_SECONDS:-60}"

log() {
  echo "[$(date -Is)] $*" | tee -a "${OUT_DIR}/launcher.log"
}

gpu_is_free() {
  local used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" | tr -d ' ')"
  [[ "${used}" =~ ^[0-9]+$ ]] && (( used < MEM_FREE_THRESHOLD ))
}

run_variant() {
  local name="$1"
  local feature_dir="$2"
  local suffix="stage2_clothing_mr0p3_smore_${name}_agg_identity_no_completion_no_ii_${RUN_TAG}"
  local log_path="${LOG_DIR}/${name}.log"

  log "start ${name} on GPU ${GPU}; feature_dir=${feature_dir}"
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
    --gcn_frontend_mode identity \
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
    --modal_feature_override_dir "${feature_dir}" \
    --tensorboard 0 \
    --save 1 \
    --topk "[10, 20, 30, 40, 50]" \
    > "${log_path}" 2>&1
  log "done ${name}; log=${log_path}"
}

log "run_tag=${RUN_TAG}"
log "out_dir=${OUT_DIR}"
while ! gpu_is_free; do
  log "GPU ${GPU} busy; waiting ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done

run_variant "observed" "${SMORE_DIR}/observed_e49_smore_baseline"
run_variant "pseudo" "${SMORE_DIR}/pseudo_mr0p3_e49_smore_baseline"
log "all variants finished"
