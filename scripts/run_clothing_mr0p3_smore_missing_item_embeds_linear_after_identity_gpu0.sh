#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
RUN_TAG="${RUN_TAG:-clothing_mr0p3_smore_missing_item_embeds_linear_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-exp_report/clothing/smore_missing_input_item_embeds_original_linear/${RUN_TAG}}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

CONFIG="${CONFIG:-configs/clothing/stage2_decoder_mm_itemgraph_completed.yaml}"
SMORE_DIR="${SMORE_DIR:-/home/ruiyuliu/projects/baselines/SMORE/experiment_logs/smore_pretrain_missing_input_compact_clothing_mr0p3_20260702}"
EPOCHS="${EPOCHS:-200}"
EARLY_STOP="${EARLY_STOP:-10000}"
WAIT_PIDS="${WAIT_PIDS:-97404}"
POLL_SECONDS="${POLL_SECONDS:-60}"

log() {
  echo "[$(date -Is)] $*" | tee -a "${OUT_DIR}/launcher.log"
}

wait_for_pid() {
  local pid="$1"
  while kill -0 "${pid}" 2>/dev/null; do
    log "waiting for PID ${pid}"
    sleep "${POLL_SECONDS}"
  done
  log "PID ${pid} finished"
}

run_variant() {
  local variant="$1"
  local feature_dir
  local suffix
  local log_path

  case "${variant}" in
    observed) feature_dir="${SMORE_DIR}/observed_e49_smore_missing_input" ;;
    pseudo) feature_dir="${SMORE_DIR}/pseudo_mr0p3_e49_smore_missing_input" ;;
    *) echo "unknown variant: ${variant}" >&2; exit 1 ;;
  esac

  suffix="stage2_clothing_mr0p3_smore_missing_${variant}_item_embeds_original_linear_no_completion_no_ii_${RUN_TAG}"
  log_path="${LOG_DIR}/${variant}_item_embeds_original_linear.log"

  log "start ${variant} missing item_embeds original_linear; feature_dir=${feature_dir}"
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
    --gcn_frontend_mode original_linear \
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
    --modal_feature_image_file image_item_embeds.npy \
    --modal_feature_text_file text_item_embeds.npy \
    --tensorboard 0 \
    --save 1 \
    --topk "[10, 20, 30, 40, 50]" \
    > "${log_path}" 2>&1
  log "done ${variant} missing item_embeds original_linear; log=${log_path}"
}

log "run_tag=${RUN_TAG}"
log "out_dir=${OUT_DIR}"
log "wait_pids=${WAIT_PIDS}"
for pid in ${WAIT_PIDS}; do
  wait_for_pid "${pid}"
done

run_variant observed
run_variant pseudo
log "all variants finished"
