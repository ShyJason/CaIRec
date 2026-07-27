#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPUS="${GPUS:-0 7}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
RUN_TAG="${RUN_TAG:-missing_reliability_gate_supervised_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-exp_report/clothing/missing_reliability_gate_mr0p3/${RUN_TAG}}"
mkdir -p "${OUT_DIR}"

CONFIG="${CONFIG:-configs/clothing/mainline_mr0p1.yaml}"
IMPUTER_CKPT="${IMPUTER_CKPT:-exp_report/clothing/stage1_2_clothing_mm_mr0p3_beststyle_nocl_clothing_mr0p3_latest_20260624_174840/ckpt/stage1_2_clothing_mm_mr0p3_beststyle_nocl_clothing_mr0p3_latest_20260624_174840_imputer_backprop_50_epoch49.pth}"

# name|floor|gate_reg|target_mean|context|hidden|dropout|init_logit|gate_lr|stats_norm|sup_coeff|obs_target|cf_coeff|cf_ratio|cf_mse_temp
CANDIDATES=(
  "base_floor07_reg01_t090_lrm5_noaux|0.70|0.10|0.90|id_embedding|64|0.10|0.0|0.0005|1|0.00|1.00|0.00|0.50|0.10"
  "base_floor05_reg05_t085_lrm5_noaux|0.50|0.50|0.85|id_embedding|64|0.10|0.0|0.0005|1|0.00|1.00|0.00|0.50|0.10"
  "cf_floor05_reg05_t085_lrm5_c002|0.50|0.50|0.85|id_embedding|64|0.10|0.0|0.0005|1|0.00|1.00|0.02|0.50|0.10"
  "cf_floor05_reg05_t085_lrm5_c005|0.50|0.50|0.85|id_embedding|64|0.10|0.0|0.0005|1|0.00|1.00|0.05|0.50|0.10"
  "supcf_floor05_reg05_t085_lrm5_s002_c002|0.50|0.50|0.85|id_embedding|64|0.10|0.0|0.0005|1|0.02|1.00|0.02|0.50|0.10"
  "supcf_floor07_reg01_t090_lrm5_s002_c002|0.70|0.10|0.90|id_embedding|64|0.10|0.0|0.0005|1|0.02|1.00|0.02|0.50|0.10"
  "supcf_floor07_reg05_t090_lrm5_s002_c002|0.70|0.50|0.90|id_embedding|64|0.10|0.0|0.0005|1|0.02|1.00|0.02|0.50|0.10"
  "supcf_floor05_reg05_t085_lrm2_s002_c002|0.50|0.50|0.85|id_embedding|64|0.10|0.0|0.0002|1|0.02|1.00|0.02|0.50|0.10"
)

read -r -a gpu_list <<< "${GPUS}"
if [[ "${#gpu_list[@]}" -eq 0 ]]; then
  echo "GPUS is empty" >&2
  exit 2
fi
if [[ "${MAX_PARALLEL}" -lt 1 ]]; then
  echo "MAX_PARALLEL must be >= 1" >&2
  exit 2
fi

pids=()
names=()
slot=0

launch_candidate() {
  local candidate="$1"
  local gpu="$2"
  IFS="|" read -r name gate_floor gate_reg target_mean context hidden dropout init_logit gate_lr stats_norm sup_coeff obs_target cf_coeff cf_ratio cf_mse_temp <<< "${candidate}"

  local run_dir="${OUT_DIR}/${name}"
  local log_dir="${run_dir}/logs"
  mkdir -p "${log_dir}"
  local suffix="stage2_clothing_mr0p3_${name}_${RUN_TAG}"
  local log_path="${log_dir}/${suffix}.log"
  local cmd_path="${run_dir}/run.cmd"
  local status_path="${run_dir}/exit.status"

  if [[ -f "${log_path}" ]] && grep -qi "final strict test hr@20" "${log_path}"; then
    echo "skip completed ${name}"
    return 0
  fi

  local cmd=(
    .venv/bin/python -u main.py
    --config "${CONFIG}"
    --device_id "${gpu}"
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
    --lr_rec 0.01
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
    --rec_neighbor_cl_weight 0.01
    --rec_neighbor_cl_temp 0.2
    --rec_neighbor_cl_bank_size 256
    --item_graph_kind fused_completed
    --item_graph_topk 8
    --item_graph_norm rw
    --item_graph_cf_scale raw
    --item_graph_cf_weight 0.25
    --item_graph_image_weight 0.375
    --item_graph_text_weight 0.375
    --item_graph_audio_weight 0.0
    --item_graph_modal_alpha 0.25
    --item_graph_modal_layers 1
    --item_graph_modal_target all
    --completion_gate_mode missing_reliability
    --completion_gate_floor "${gate_floor}"
    --completion_gate_reg_coeff "${gate_reg}"
    --completion_gate_target_mean "${target_mean}"
    --completion_gate_use_item_context 1
    --completion_gate_item_context_source "${context}"
    --completion_gate_hidden_dim "${hidden}"
    --completion_gate_dropout "${dropout}"
    --completion_gate_init_logit "${init_logit}"
    --completion_gate_lr "${gate_lr}"
    --completion_gate_stats_norm "${stats_norm}"
    --completion_gate_detach_inputs 1
    --completion_gate_supervision_coeff "${sup_coeff}"
    --completion_gate_supervision_observed_target "${obs_target}"
    --completion_gate_counterfactual_coeff "${cf_coeff}"
    --completion_gate_counterfactual_ratio "${cf_ratio}"
    --completion_gate_counterfactual_mse_temp "${cf_mse_temp}"
    --completion_gate_apply_observed 0
    --fusion_mode mean
    --tensorboard 0
    --save 1
    --topk "[10, 20, 30, 40, 50]"
    --suffix "${suffix}"
  )

  printf "%q " "${cmd[@]}" > "${cmd_path}"
  printf "\n" >> "${cmd_path}"
  echo "$(date '+%F %T') start ${name} on GPU ${gpu}; log=${log_path}"
  (
    "${cmd[@]}" > "${log_path}" 2>&1
    rc=$?
    echo "EXIT_CODE=${rc}" > "${status_path}"
    exit "${rc}"
  ) &
  pids+=("$!")
  names+=("${name}")
  echo "$!" > "${run_dir}/pid"
}

overall=0

wait_for_batch() {
  for i in "${!pids[@]}"; do
    local pid="${pids[$i]}"
    local name="${names[$i]}"
    if wait "${pid}"; then
      echo "$(date '+%F %T') done ${name}"
    else
      local rc=$?
      echo "$(date '+%F %T') failed ${name} exit=${rc}"
      overall=1
    fi
  done
  pids=()
  names=()
}

for candidate in "${CANDIDATES[@]}"; do
  gpu_index=$(( slot % ${#gpu_list[@]} ))
  gpu="${gpu_list[$gpu_index]}"
  launch_candidate "${candidate}" "${gpu}"
  slot=$((slot + 1))
  if [[ "${#pids[@]}" -ge "${MAX_PARALLEL}" ]]; then
    wait_for_batch
  fi
done

if [[ "${#pids[@]}" -gt 0 ]]; then
  wait_for_batch
fi

exit "${overall}"
