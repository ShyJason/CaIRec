#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
RUN_TAG="${RUN_TAG:-clothing_mr0p3_smore_other_reps_after_agg_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-exp_report/clothing/smore_other_reps_no_completion_no_ii/${RUN_TAG}}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

CONFIG="${CONFIG:-configs/clothing/stage2_decoder_mm_itemgraph_completed.yaml}"
SMORE_DIR="${SMORE_DIR:-/home/ruiyuliu/projects/baselines/SMORE/experiment_logs/smore_pretrain_agg_compact_clothing_mr0p3_20260702}"
EPOCHS="${EPOCHS:-200}"
EARLY_STOP="${EARLY_STOP:-10000}"
WAIT_PIDS="${WAIT_PIDS:-3557806 3575617}"
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

run_case() {
  local rep_tag="$1"
  local frontend="$2"
  local variant="$3"
  local image_file="$4"
  local text_file="$5"
  local variant_dir
  local suffix
  local log_path

  case "${variant}" in
    observed) variant_dir="${SMORE_DIR}/observed_e49_smore_baseline" ;;
    pseudo) variant_dir="${SMORE_DIR}/pseudo_mr0p3_e49_smore_baseline" ;;
    *) echo "unknown variant: ${variant}" >&2; exit 1 ;;
  esac

  suffix="stage2_clothing_mr0p3_smore_${variant}_${rep_tag}_${frontend}_no_completion_no_ii_${RUN_TAG}"
  log_path="${LOG_DIR}/${variant}_${rep_tag}_${frontend}.log"

  log "start ${variant} ${rep_tag} ${frontend}; image=${image_file}; text=${text_file}"
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
    --modal_feature_override_dir "${variant_dir}" \
    --modal_feature_image_file "${image_file}" \
    --modal_feature_text_file "${text_file}" \
    --tensorboard 0 \
    --save 1 \
    --topk "[10, 20, 30, 40, 50]" \
    > "${log_path}" 2>&1
  log "done ${variant} ${rep_tag} ${frontend}; log=${log_path}"
}

log "run_tag=${RUN_TAG}"
log "out_dir=${OUT_DIR}"
log "wait_pids=${WAIT_PIDS}"
for pid in ${WAIT_PIDS}; do
  wait_for_pid "${pid}"
done

# True image/text modality pairs.
PAIR_CASES=(
  "item_embeds image_item_embeds.npy text_item_embeds.npy"
  "projected image_projected.npy text_projected.npy"
)

# Single item-level SMORE representations, duplicated into image/text branches.
SHARED_CASES=(
  "all_embeddings all_embeddings_items.npy all_embeddings_items.npy"
  "content_embeds content_embeds_items.npy content_embeds_items.npy"
  "fusion_item fusion_item_embeds.npy fusion_item_embeds.npy"
  "side_embeds side_embeds_items.npy side_embeds_items.npy"
)

for frontend in identity original_linear; do
  for variant in observed pseudo; do
    for case_desc in "${PAIR_CASES[@]}"; do
      read -r rep_tag image_file text_file <<<"${case_desc}"
      run_case "${rep_tag}" "${frontend}" "${variant}" "${image_file}" "${text_file}"
    done
    for case_desc in "${SHARED_CASES[@]}"; do
      read -r rep_tag image_file text_file <<<"${case_desc}"
      run_case "${rep_tag}" "${frontend}" "${variant}" "${image_file}" "${text_file}"
    done
  done
done

log "all SMORE representation cases finished"
