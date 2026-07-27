#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-7}"
MAX_CONCURRENT="${MAX_CONCURRENT:-1}"
LR_REC="${LR_REC:-0.01}"
CONF_REG_COEFF="${CONF_REG_COEFF:-0.1}"
SCORE_BLEND="${SCORE_BLEND:-1.0}"
ITEM_GRAPH_TOPK="${ITEM_GRAPH_TOPK:-5}"
ITEM_GRAPH_MODAL_ALPHA="${ITEM_GRAPH_MODAL_ALPHA:-0.15}"
ITEM_GRAPH_CF_WEIGHT="${ITEM_GRAPH_CF_WEIGHT:-0.25}"
ITEM_GRAPH_IMAGE_WEIGHT="${ITEM_GRAPH_IMAGE_WEIGHT:-0.375}"
ITEM_GRAPH_TEXT_WEIGHT="${ITEM_GRAPH_TEXT_WEIGHT:-0.375}"
ITEM_GRAPH_IMAGE_CF_WEIGHT="${ITEM_GRAPH_IMAGE_CF_WEIGHT:-${ITEM_GRAPH_CF_WEIGHT}}"
ITEM_GRAPH_IMAGE_SEMANTIC_WEIGHT="${ITEM_GRAPH_IMAGE_SEMANTIC_WEIGHT:-${ITEM_GRAPH_IMAGE_WEIGHT}}"
ITEM_GRAPH_TEXT_CF_WEIGHT="${ITEM_GRAPH_TEXT_CF_WEIGHT:-${ITEM_GRAPH_CF_WEIGHT}}"
ITEM_GRAPH_TEXT_SEMANTIC_WEIGHT="${ITEM_GRAPH_TEXT_SEMANTIC_WEIGHT:-${ITEM_GRAPH_TEXT_WEIGHT}}"
REC_NEIGHBOR_CL_WEIGHT="${REC_NEIGHBOR_CL_WEIGHT:-0.01}"
RUN_TAG="${RUN_TAG:-range_compare_gpu${GPU}_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-exp_report/clothing/modality_dynamic_modspec_range_compare_mr0p3/${RUN_TAG}}"
mkdir -p "${OUT_DIR}"

CONFIG="${CONFIG:-configs/clothing/mainline_mr0p1.yaml}"
IMPUTER_CKPT="${IMPUTER_CKPT:-exp_report/clothing/stage1_2_clothing_mm_mr0p3_beststyle_nocl_clothing_mr0p3_latest_20260624_174840/ckpt/stage1_2_clothing_mm_mr0p3_beststyle_nocl_clothing_mr0p3_latest_20260624_174840_imputer_backprop_50_epoch49.pth}"

BASE_ARGS=(
  --config "${CONFIG}"
  --device_id "${GPU}"
  --dataset clothing
  --exp_mode mm
  --train_stage recommender
  --missing_rate 0.3
  --eval_missing_rate 0.5
  --seed 2023
  --dataset_seed 0
  --imputer_ckpt "${IMPUTER_CKPT}"
  --epoch 200
  --early_stop 20
  --eva_interval 1
  --batch_size 2048
  --lr 0.01
  --lr_rec "${LR_REC}"
  --lr_imp 0.0002
  --lr_decoder 0.00005
  --freeze_imputer 1
  --freeze_decoder 1
  --recommender_allow_modal_grad 0
  --feature_bridge_mode raw_decoder
  --gcn_frontend_mode original_linear
  --disable_imputation 0
  --modality_bpr_coeff 1.0
  --reg_coeff 0.01
  --evaluation_protocol strict
  --selection_mode val
  --strict_probe_test_interval 0
  --recommendation_selection_metric recall
  --recommendation_selection_topk 20
  --rec_neighbor_cl_weight "${REC_NEIGHBOR_CL_WEIGHT}"
  --rec_neighbor_cl_temp 0.2
  --rec_neighbor_cl_bank_size 256
  --item_graph_kind modality_completed_dynamic_confidence
  --item_graph_topk "${ITEM_GRAPH_TOPK}"
  --item_graph_norm rw
  --item_graph_cf_weight "${ITEM_GRAPH_CF_WEIGHT}"
  --item_graph_cf_scale raw
  --item_graph_image_weight "${ITEM_GRAPH_IMAGE_WEIGHT}"
  --item_graph_text_weight "${ITEM_GRAPH_TEXT_WEIGHT}"
  --item_graph_image_cf_weight "${ITEM_GRAPH_IMAGE_CF_WEIGHT}"
  --item_graph_image_semantic_weight "${ITEM_GRAPH_IMAGE_SEMANTIC_WEIGHT}"
  --item_graph_text_cf_weight "${ITEM_GRAPH_TEXT_CF_WEIGHT}"
  --item_graph_text_semantic_weight "${ITEM_GRAPH_TEXT_SEMANTIC_WEIGHT}"
  --item_graph_audio_weight 0.0
  --item_graph_modal_alpha "${ITEM_GRAPH_MODAL_ALPHA}"
  --item_graph_modal_layers 1
  --item_graph_modal_target all
  --item_graph_modality_specific_confidence 1
  --item_graph_confidence_transform sigmoid
  --item_graph_dynamic_score_blend "${SCORE_BLEND}"
  --item_graph_rr_confidence_init 1.0
  --item_graph_ri_confidence_init 1.0
  --item_graph_ii_confidence_init 1.0
  --item_graph_confidence_reg_coeff "${CONF_REG_COEFF}"
  --item_graph_confidence_reg_start_epoch 0
  --item_graph_confidence_log_interval 10
  --tensorboard 0
  --save 1
  --topk "[10, 20, 30, 40, 50]"
)

if [[ $# -gt 0 ]]; then
  RANGES=("$@")
else
  RANGES=("0.9:1.1:1.05:1.0:0.95" "0.5:1.5:1.0:0.9:0.8")
fi

running_jobs=0
for range in "${RANGES[@]}"; do
  IFS=":" read -r min max rr_target ri_target ii_target <<< "${range}"
  rr_target="${rr_target:-1.0}"
  ri_target="${ri_target:-0.9}"
  ii_target="${ii_target:-0.8}"
  min_tag="${min//./p}"
  max_tag="${max//./p}"
  rr_tag="${rr_target//./p}"
  ri_tag="${ri_target//./p}"
  ii_tag="${ii_target//./p}"
  topk_tag="${ITEM_GRAPH_TOPK}"
  alpha_tag="${ITEM_GRAPH_MODAL_ALPHA//./p}"
  cf_tag="${ITEM_GRAPH_CF_WEIGHT//./p}"
  img_tag="${ITEM_GRAPH_IMAGE_WEIGHT//./p}"
  txt_tag="${ITEM_GRAPH_TEXT_WEIGHT//./p}"
  cl_tag="${REC_NEIGHBOR_CL_WEIGHT//./p}"
  name="rng${min_tag}_${max_tag}_rr${rr_tag}_ri${ri_tag}_ii${ii_tag}_typedreg_nowarm_topk${topk_tag}_a${alpha_tag}_w${cf_tag}_${img_tag}_${txt_tag}_cl${cl_tag}"
  run_dir="${OUT_DIR}/${name}"
  log_dir="${run_dir}/logs"
  mkdir -p "${log_dir}"
  suffix="stage2_clothing_mr0p3_${name}_${RUN_TAG}"
  log_path="${log_dir}/${suffix}.log"
  cmd_path="${run_dir}/run.cmd"

  if [[ -f "${log_path}" ]] && grep -qi "final strict test hr@20" "${log_path}"; then
    echo "skip completed ${name}"
    continue
  fi

  cmd=(
    .venv/bin/python -u main.py
    "${BASE_ARGS[@]}"
    --suffix "${suffix}"
    --item_graph_confidence_min "${min}"
    --item_graph_confidence_max "${max}"
    --item_graph_confidence_reg_target 1.0
    --item_graph_rr_confidence_reg_target "${rr_target}"
    --item_graph_ri_confidence_reg_target "${ri_target}"
    --item_graph_ii_confidence_reg_target "${ii_target}"
  )

  printf "%q " "${cmd[@]}" > "${cmd_path}"
  printf "\n" >> "${cmd_path}"
  echo "start ${name} on GPU ${GPU}"
  "${cmd[@]}" > "${log_path}" 2>&1 &
  echo "$!" > "${run_dir}/pid"
  running_jobs=$((running_jobs + 1))

  if (( running_jobs >= MAX_CONCURRENT )); then
    wait -n
    running_jobs=$((running_jobs - 1))
  fi
done

wait
