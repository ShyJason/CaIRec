#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-0}"
WAIT_MAX_MEMORY_MB="${WAIT_MAX_MEMORY_MB:-36000}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"
QUEUE_TAG="${QUEUE_TAG:-modspec_nextwave_gpu${GPU}_$(date +%Y%m%d_%H%M%S)}"
QUEUE_DIR="${QUEUE_DIR:-exp_report/clothing/modality_dynamic_modspec_nextwave_mr0p3/${QUEUE_TAG}}"
mkdir -p "${QUEUE_DIR}"

CONFIG="${CONFIG:-configs/clothing/mainline_mr0p1.yaml}"
IMPUTER_CKPT="${IMPUTER_CKPT:-exp_report/clothing/stage1_2_clothing_mm_mr0p3_beststyle_nocl_clothing_mr0p3_latest_20260624_174840/ckpt/stage1_2_clothing_mm_mr0p3_beststyle_nocl_clothing_mr0p3_latest_20260624_174840_imputer_backprop_50_epoch49.pth}"

if [[ -n "${CANDIDATES:-}" ]]; then
  read -r -a candidate_list <<< "${CANDIDATES}"
else
  candidate_list=(
    topk10_reg005_init_target
    topk10_reg001_init_target
    topk10_warm_score30
  )
fi

wait_for_slot() {
  local used_mb
  while true; do
    used_mb="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" | tr -d ' ')"
    if [[ "${used_mb}" =~ ^[0-9]+$ ]] && (( used_mb <= WAIT_MAX_MEMORY_MB )); then
      echo "$(date '+%F %T') slot available on GPU ${GPU}: used=${used_mb}MB <= threshold=${WAIT_MAX_MEMORY_MB}MB"
      return 0
    fi
    echo "$(date '+%F %T') wait GPU ${GPU}: used=${used_mb}MB > threshold=${WAIT_MAX_MEMORY_MB}MB"
    sleep "${WAIT_POLL_SECONDS}"
  done
}

run_candidate() {
  local candidate="$1"
  local topk alpha cf_weight image_weight text_weight conf_reg cl_weight lr_rec
  local score_blend score_blend_start score_blend_warmup neigh_blend neigh_blend_start neigh_blend_warmup
  local text_rr text_ri text_ii image_rr image_ri image_ii

  case "${candidate}" in
    topk10_reg005_init_target)
      topk=10; alpha=0.25; cf_weight=0.15; image_weight=0.425; text_weight=0.425
      conf_reg=0.05; cl_weight=0.01; lr_rec=0.01
      score_blend=1.0; score_blend_start=-1.0; score_blend_warmup=0
      neigh_blend=1.0; neigh_blend_start=-1.0; neigh_blend_warmup=0
      text_rr=0.70; text_ri=0.10; text_ii=0.02
      image_rr=0.05; image_ri=0.05; image_ii=0.70
      ;;
    topk10_reg001_init_target)
      topk=10; alpha=0.25; cf_weight=0.15; image_weight=0.425; text_weight=0.425
      conf_reg=0.01; cl_weight=0.01; lr_rec=0.01
      score_blend=1.0; score_blend_start=-1.0; score_blend_warmup=0
      neigh_blend=1.0; neigh_blend_start=-1.0; neigh_blend_warmup=0
      text_rr=0.70; text_ri=0.10; text_ii=0.02
      image_rr=0.05; image_ri=0.05; image_ii=0.70
      ;;
    topk10_warm_score30)
      topk=10; alpha=0.25; cf_weight=0.15; image_weight=0.425; text_weight=0.425
      conf_reg=0.02; cl_weight=0.01; lr_rec=0.01
      score_blend=1.0; score_blend_start=0.0; score_blend_warmup=30
      neigh_blend=1.0; neigh_blend_start=-1.0; neigh_blend_warmup=0
      text_rr=0.70; text_ri=0.10; text_ii=0.02
      image_rr=0.05; image_ri=0.05; image_ii=0.70
      ;;
    topk10_warm_neigh30)
      topk=10; alpha=0.25; cf_weight=0.15; image_weight=0.425; text_weight=0.425
      conf_reg=0.02; cl_weight=0.01; lr_rec=0.01
      score_blend=1.0; score_blend_start=-1.0; score_blend_warmup=0
      neigh_blend=1.0; neigh_blend_start=0.0; neigh_blend_warmup=30
      text_rr=0.70; text_ri=0.10; text_ii=0.02
      image_rr=0.05; image_ri=0.05; image_ii=0.70
      ;;
    topk8_cf0125_reg005)
      topk=8; alpha=0.25; cf_weight=0.125; image_weight=0.4375; text_weight=0.4375
      conf_reg=0.05; cl_weight=0.01; lr_rec=0.01
      score_blend=1.0; score_blend_start=-1.0; score_blend_warmup=0
      neigh_blend=1.0; neigh_blend_start=-1.0; neigh_blend_warmup=0
      text_rr=0.45; text_ri=0.10; text_ii=0.02
      image_rr=0.05; image_ri=0.05; image_ii=0.55
      ;;
    topk10_alpha015_reg002)
      topk=10; alpha=0.15; cf_weight=0.15; image_weight=0.425; text_weight=0.425
      conf_reg=0.02; cl_weight=0.01; lr_rec=0.01
      score_blend=1.0; score_blend_start=-1.0; score_blend_warmup=0
      neigh_blend=1.0; neigh_blend_start=-1.0; neigh_blend_warmup=0
      text_rr=0.70; text_ri=0.10; text_ii=0.02
      image_rr=0.05; image_ri=0.05; image_ii=0.70
      ;;
    *)
      echo "unknown CANDIDATE=${candidate}" >&2
      return 2
      ;;
  esac

  wait_for_slot

  local run_dir="${QUEUE_DIR}/${candidate}"
  local log_dir="${run_dir}/logs"
  mkdir -p "${log_dir}"
  local suffix="stage2_clothing_mr0p3_${candidate}_${QUEUE_TAG}"
  local log_path="${log_dir}/${suffix}.log"
  local cmd_path="${run_dir}/run.cmd"
  local status_path="${run_dir}/exit.status"

  if [[ -f "${log_path}" ]] && grep -qi "final strict test hr@20" "${log_path}"; then
    echo "$(date '+%F %T') skip completed ${candidate}"
    return 0
  fi

  local cmd=(
    .venv/bin/python -u main.py
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
    --lr_rec "${lr_rec}"
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
    --rec_neighbor_cl_weight "${cl_weight}"
    --rec_neighbor_cl_temp 0.2
    --rec_neighbor_cl_bank_size 256
    --item_graph_kind modality_completed_dynamic_confidence
    --item_graph_topk "${topk}"
    --item_graph_norm rw
    --item_graph_cf_scale raw
    --item_graph_cf_weight "${cf_weight}"
    --item_graph_image_weight "${image_weight}"
    --item_graph_text_weight "${text_weight}"
    --item_graph_audio_weight 0.0
    --item_graph_modal_alpha "${alpha}"
    --item_graph_modal_layers 1
    --item_graph_modal_target all
    --item_graph_modality_specific_confidence 1
    --item_graph_confidence_transform sigmoid
    --item_graph_dynamic_score_blend "${score_blend}"
    --item_graph_dynamic_score_blend_start "${score_blend_start}"
    --item_graph_dynamic_score_blend_warmup_epochs "${score_blend_warmup}"
    --item_graph_dynamic_neighbor_blend "${neigh_blend}"
    --item_graph_dynamic_neighbor_blend_start "${neigh_blend_start}"
    --item_graph_dynamic_neighbor_blend_warmup_epochs "${neigh_blend_warmup}"
    --item_graph_confidence_min 0.0
    --item_graph_confidence_max 1.0
    --item_graph_confidence_reg_coeff "${conf_reg}"
    --item_graph_confidence_reg_start_epoch 0
    --item_graph_confidence_log_interval 10
    --item_graph_rr_confidence_reg_target 1.0
    --item_graph_ri_confidence_reg_target 1.0
    --item_graph_ii_confidence_reg_target 1.0
    --item_graph_text_rr_confidence_init "${text_rr}"
    --item_graph_text_ri_confidence_init "${text_ri}"
    --item_graph_text_ii_confidence_init "${text_ii}"
    --item_graph_image_rr_confidence_init "${image_rr}"
    --item_graph_image_ri_confidence_init "${image_ri}"
    --item_graph_image_ii_confidence_init "${image_ii}"
    --item_graph_text_rr_confidence_reg_target "${text_rr}"
    --item_graph_text_ri_confidence_reg_target "${text_ri}"
    --item_graph_text_ii_confidence_reg_target "${text_ii}"
    --item_graph_image_rr_confidence_reg_target "${image_rr}"
    --item_graph_image_ri_confidence_reg_target "${image_ri}"
    --item_graph_image_ii_confidence_reg_target "${image_ii}"
    --tensorboard 0
    --save 1
    --topk "[10, 20, 30, 40, 50]"
    --suffix "${suffix}"
  )

  printf "%q " "${cmd[@]}" > "${cmd_path}"
  printf "\n" >> "${cmd_path}"
  echo "$(date '+%F %T') start ${candidate} on GPU ${GPU}; log=${log_path}"
  "${cmd[@]}" > "${log_path}" 2>&1
  local rc=$?
  echo "EXIT_CODE=${rc}" > "${status_path}"
  return "${rc}"
}

overall=0
for candidate in "${candidate_list[@]}"; do
  if run_candidate "${candidate}"; then
    echo "$(date '+%F %T') done ${candidate}"
  else
    rc=$?
    echo "$(date '+%F %T') failed ${candidate} exit=${rc}"
    overall=1
  fi
done

exit "${overall}"
