#!/usr/bin/env bash
set -euo pipefail

seed="${1:?seed required}"
gpu="${2:?gpu id required}"
stamp="${3:-microlens_allgate_2seed_20260528}"

cd "$(dirname "$0")/.."

case "${seed}" in
  12)
    imputer_ckpt="exp_report/microlens/stage1_2_microlens_imputer_backprop_decoder_v2_mmrec_microlens_mm_2seed_20260521_153006_seed12/ckpt/stage1_2_microlens_imputer_backprop_decoder_v2_mmrec_microlens_mm_2seed_20260521_153006_seed12_imputer_backprop_50_epoch8.pth"
    ;;
  123)
    imputer_ckpt="exp_report/microlens/stage1_2_microlens_imputer_backprop_decoder_v2_mmrec_microlens_tuned_5seed_20260522_023007_seed123/ckpt/stage1_2_microlens_imputer_backprop_decoder_v2_mmrec_microlens_tuned_5seed_20260522_023007_seed123_imputer_backprop_50_epoch19.pth"
    ;;
  *)
    echo "Unsupported seed: ${seed}" >&2
    exit 2
    ;;
esac

suffix="stage2_microlens_rankres_allgate_seed${seed}_${stamp}"
outer_dir="exp_report/fusion_norm_ablation/${stamp}"
mkdir -p "${outer_dir}"

echo "[microlens-allgate] seed=${seed} gpu=${gpu} suffix=${suffix}"

CUDA_VISIBLE_DEVICES="${gpu}" .venv/bin/python main.py \
  --config configs/microlens/stage2_decoder_mm.yaml \
  --suffix "${suffix}" \
  --dataset microlens \
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
  --completion_gate_reg_coeff 1.0 \
  --recommender_allow_modal_grad 0 \
  2>&1 | tee "${outer_dir}/microlens_seed${seed}.log"
