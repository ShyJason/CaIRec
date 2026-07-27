#!/usr/bin/env bash
set -euo pipefail

seed="${1:?seed required}"
gpu="${2:?gpu id required}"
stamp="${3:-allgate_2seed_20260528}"

cd "$(dirname "$0")/.."

run_one() {
  local dataset="$1"
  local config="$2"
  local imputer_ckpt="$3"
  local gate_reg="$4"
  local suffix="stage2_${dataset}_rankres_allgate_seed${seed}_${stamp}"
  local outer_dir="exp_report/fusion_norm_ablation/${stamp}"

  mkdir -p "${outer_dir}"
  echo "[allgate] dataset=${dataset} seed=${seed} gpu=${gpu} suffix=${suffix}"

  CUDA_VISIBLE_DEVICES="${gpu}" .venv/bin/python main.py \
    --config "${config}" \
    --suffix "${suffix}" \
    --dataset "${dataset}" \
    --exp_mode mm \
    --device_id 0 \
    --seed "${seed}" \
    --dataset_seed 0 \
    --train_stage recommender \
    --freeze_imputer 1 \
    --freeze_decoder 1 \
    --disable_imputation 0 \
    --feature_bridge_mode raw_decoder \
    --gcn_frontend_mode original_linear \
    --imputer_ckpt "${imputer_ckpt}" \
    --epoch 200 \
    --early_stop 20 \
    --eva_interval 1 \
    --evaluation_protocol strict \
    --selection_mode val \
    --strict_probe_test_interval 10 \
    --save 1 \
    --completion_gate_mode rank_residual_allgate \
    --completion_gate_hidden_dim 64 \
    --completion_gate_dropout 0.1 \
    --completion_gate_init_logit 0.0 \
    --completion_gate_detach_inputs 1 \
    --completion_gate_use_item_context 1 \
    --completion_gate_item_context_source shared_mean \
    --completion_gate_residual_alpha 0.18 \
    --completion_gate_mix_alpha 0.35 \
    --completion_gate_identity_coeff 0.05 \
    --completion_gate_balance_coeff 0.01 \
    --completion_gate_reg_coeff "${gate_reg}" \
    --recommender_allow_modal_grad 0 \
    2>&1 | tee "${outer_dir}/${dataset}_seed${seed}.log"
}

run_one "clothing" \
  "configs/clothing/stage2_decoder_mm.yaml" \
  "exp_report/clothing/stage1_2_clothing_mmrec_fixed_seed${seed}_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage1_2_clothing_mmrec_fixed_seed${seed}_mmrec_clothing_mm_fixedmissing_20260521_052129_imputer_backprop_50_epoch19.pth" \
  "1.0"

run_one "sports" \
  "configs/sports/stage2_decoder_mm.yaml" \
  "exp_report/sports/stage1_2_sports_imputer_backprop_decoder_v2_mmrec_sports_mm_mr0.3_seed${seed}_mmrec_sports_mm_fixedmissing_20260524_165817/ckpt/stage1_2_sports_imputer_backprop_decoder_v2_mmrec_sports_mm_mr0.3_seed${seed}_mmrec_sports_mm_fixedmissing_20260524_165817_imputer_backprop_50_epoch19.pth" \
  "0.1"
