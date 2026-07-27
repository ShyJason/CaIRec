#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-7}"
RUN_TAG="${RUN_TAG:-modspec_finalpush_gpu${GPU}_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-exp_report/clothing/modality_dynamic_modspec_finalpush_mr0p3/${RUN_TAG}}"
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
  --rec_neighbor_cl_temp 0.2
  --rec_neighbor_cl_bank_size 256
  --item_graph_kind modality_completed_dynamic_confidence
  --item_graph_norm rw
  --item_graph_cf_scale raw
  --item_graph_audio_weight 0.0
  --item_graph_modal_layers 1
  --item_graph_modal_target all
  --item_graph_modality_specific_confidence 1
  --item_graph_confidence_transform sigmoid
  --item_graph_dynamic_score_blend 1.0
  --item_graph_rr_confidence_init 1.0
  --item_graph_ri_confidence_init 1.0
  --item_graph_ii_confidence_init 1.0
  --item_graph_confidence_reg_start_epoch 0
  --item_graph_confidence_log_interval 10
  --item_graph_confidence_min 0.0
  --item_graph_confidence_max 1.0
  --item_graph_confidence_reg_target 1.0
  --item_graph_rr_confidence_reg_target 1.0
  --item_graph_ri_confidence_reg_target 1.0
  --item_graph_ii_confidence_reg_target 1.0
  --tensorboard 0
  --save 1
  --topk "[10, 20, 30, 40, 50]"
)

# name|topk|alpha|cf|image|text|reg|text_rr|text_ri|text_ii|image_rr|image_ri|image_ii|cl|lr_rec
CANDIDATES=(
  "cf015_soft_lr005|8|0.25|0.15|0.425|0.425|0.02|0.70|0.10|0.02|0.05|0.05|0.70|0.01|0.005"
  "cf015_learnedtarget_lr005|8|0.25|0.15|0.425|0.425|0.02|0.45|0.10|0.02|0.05|0.05|0.55|0.01|0.005"
  "cf015_soft_cl005|8|0.25|0.15|0.425|0.425|0.02|0.70|0.10|0.02|0.05|0.05|0.70|0.005|0.01"
  "cf015_soft_topk10|10|0.25|0.15|0.425|0.425|0.02|0.70|0.10|0.02|0.05|0.05|0.70|0.01|0.01"
)

pids=()
names=()

for candidate in "${CANDIDATES[@]}"; do
  IFS="|" read -r name topk alpha cf_weight image_weight text_weight reg_coeff \
    text_rr text_ri text_ii image_rr image_ri image_ii cl_weight lr_rec <<< "${candidate}"

  run_dir="${OUT_DIR}/${name}"
  log_dir="${run_dir}/logs"
  mkdir -p "${log_dir}"
  suffix="stage2_clothing_mr0p3_${name}_${RUN_TAG}"
  log_path="${log_dir}/${suffix}.log"
  cmd_path="${run_dir}/run.cmd"
  status_path="${run_dir}/exit.status"

  if [[ -f "${log_path}" ]] && grep -qi "final strict test hr@20" "${log_path}"; then
    echo "skip completed ${name}"
    continue
  fi

  cmd=(
    .venv/bin/python -u main.py
    "${BASE_ARGS[@]}"
    --suffix "${suffix}"
    --lr_rec "${lr_rec}"
    --rec_neighbor_cl_weight "${cl_weight}"
    --item_graph_topk "${topk}"
    --item_graph_modal_alpha "${alpha}"
    --item_graph_cf_weight "${cf_weight}"
    --item_graph_image_weight "${image_weight}"
    --item_graph_text_weight "${text_weight}"
    --item_graph_confidence_reg_coeff "${reg_coeff}"
    --item_graph_text_rr_confidence_reg_target "${text_rr}"
    --item_graph_text_ri_confidence_reg_target "${text_ri}"
    --item_graph_text_ii_confidence_reg_target "${text_ii}"
    --item_graph_image_rr_confidence_reg_target "${image_rr}"
    --item_graph_image_ri_confidence_reg_target "${image_ri}"
    --item_graph_image_ii_confidence_reg_target "${image_ii}"
  )

  printf "%q " "${cmd[@]}" > "${cmd_path}"
  printf "\n" >> "${cmd_path}"
  echo "start ${name} on GPU ${GPU}"
  (
    "${cmd[@]}" > "${log_path}" 2>&1
    rc=$?
    echo "EXIT_CODE=${rc}" > "${status_path}"
    exit "${rc}"
  ) &
  pids+=("$!")
  names+=("${name}")
  echo "$!" > "${run_dir}/pid"
done

overall=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  name="${names[$i]}"
  if wait "${pid}"; then
    echo "done ${name}"
  else
    rc=$?
    echo "failed ${name} exit=${rc}"
    overall=1
  fi
done

exit "${overall}"
