#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
GPU="${GPU:-1}"
RUN_TAG="${RUN_TAG:-vae_completion_clothing_mr0p3_smoke_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-exp_report/clothing/${RUN_TAG}}"
VAE_DIR="${OUT_DIR}/vae_features"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

VAE_EPOCHS="${VAE_EPOCHS:-2}"
REC_EPOCHS="${REC_EPOCHS:-1}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-512}"
REC_BATCH_SIZE="${REC_BATCH_SIZE:-2048}"

log() {
  echo "[$(date -Is)] $*" | tee -a "${OUT_DIR}/pipeline.log"
}

log "run_tag=${RUN_TAG}"
log "output=${OUT_DIR}"
log "gpu=${GPU}"

log "stage A: train/export VAE completed features"
"${PYTHON_BIN}" -u variants/vae_completion/train_vae_imputer.py \
  --dataset clothing \
  --data_dir Data/clothing \
  --output_dir "${VAE_DIR}" \
  --device "cuda:${GPU}" \
  --seed 2023 \
  --train_missing_rate 0.3 \
  --eval_missing_rate 0.5 \
  --missing_seed 2023 \
  --epochs "${VAE_EPOCHS}" \
  --batch_size "${VAE_BATCH_SIZE}" \
  --lr 0.0005 \
  --weight_decay 0.000001 \
  --latent_dim 64 \
  --hidden_dim 512 \
  --modal_hidden_dim 256 \
  --dropout 0.1 \
  --beta_kl 0.001 \
  --kl_warmup_epochs 20 \
  --normalize_features 1 \
  --normalize_outputs 1 \
  --input_dropout 0.0 \
  --eval_interval 1 \
  --early_stop 0 \
  > "${LOG_DIR}/vae.log" 2>&1

log "stage B: run recommender on VAE completed features"
"${PYTHON_BIN}" -u main.py \
  --config configs/clothing/stage2_decoder_mm.yaml \
  --device_id "${GPU}" \
  --dataset clothing \
  --exp_mode ff \
  --train_stage recommender \
  --missing_rate 0.3 \
  --eval_missing_rate 0.5 \
  --seed 2023 \
  --dataset_seed 0 \
  --suffix "${RUN_TAG}_stage2_iigraph" \
  --epoch "${REC_EPOCHS}" \
  --early_stop 10000 \
  --eva_interval 1 \
  --batch_size "${REC_BATCH_SIZE}" \
  --lr 0.01 \
  --lr_rec 0.01 \
  --freeze_imputer 1 \
  --freeze_decoder 1 \
  --feature_bridge_mode raw_decoder \
  --gcn_frontend_mode original_linear \
  --disable_imputation 1 \
  --item_graph_kind fused_completed \
  --item_graph_feature_source external_completed \
  --item_graph_feature_space raw_decoder \
  --item_graph_feature_dir "${VAE_DIR}/phase_graph" \
  --item_graph_topk 20 \
  --item_graph_norm rw \
  --item_graph_cf_weight 0.3 \
  --item_graph_image_weight 0.35 \
  --item_graph_text_weight 0.35 \
  --item_graph_audio_weight 0.0 \
  --item_graph_modal_alpha 0.25 \
  --item_graph_modal_layers 1 \
  --item_graph_modal_target all \
  --rec_neighbor_cl_weight 0.0 \
  --evaluation_protocol strict \
  --selection_mode val \
  --recommendation_selection_metric recall \
  --recommendation_selection_topk 20 \
  --modal_feature_override_dir "${VAE_DIR}" \
  --modal_feature_image_file image_feat.npy \
  --modal_feature_text_file text_feat.npy \
  --tensorboard 0 \
  --save 1 \
  --topk "[10, 20, 30, 40, 50]" \
  > "${LOG_DIR}/stage2.log" 2>&1

log "done. logs=${LOG_DIR}"
