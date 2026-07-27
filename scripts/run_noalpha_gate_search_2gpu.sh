#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PATH="${ROOT_DIR}/.venv/bin:${PATH}"
export PYTHONUNBUFFERED=1
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

RUN_TAG="${RUN_TAG:-noalpha_gate_search_$(date +%Y%m%d_%H%M%S)}"
GPUS_STR="${GPUS:-2 3}"
DATASETS_STR="${DATASETS:-clothing sports microlens100k}"
SEARCH_SEEDS_STR="${SEARCH_SEEDS:-12 123}"
FULL_SEEDS_STR="${FULL_SEEDS:-1 12 123 1234 12345}"
DATASET_SEED="${DATASET_SEED:-0}"
ASSEMBLE_ALPHAS="${ASSEMBLE_ALPHAS:-0,0.2,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,1}"
PRIMARY_ALPHA="${PRIMARY_ALPHA:-0.4}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a GPUS <<< "${GPUS_STR}"
read -r -a DATASETS <<< "${DATASETS_STR}"
read -r -a SEARCH_SEEDS <<< "${SEARCH_SEEDS_STR}"
read -r -a FULL_SEEDS <<< "${FULL_SEEDS_STR}"

if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "GPUS must contain at least one GPU id" >&2
  exit 2
fi

OUT_DIR="${ROOT_DIR}/exp_report/noalpha_gate_search/${RUN_TAG}"
mkdir -p "${OUT_DIR}/tasks" "${OUT_DIR}/logs" "${OUT_DIR}/eval/search" "${OUT_DIR}/eval/full"

log() {
  date +"[noalpha-search] %Y-%m-%d %H:%M:%S $*"
}

candidate_table() {
  cat <<'EOF'
na_m010_reg01_i005_b001_h64_d01	--completion_gate_mode rank_residual_allgate --completion_gate_no_residual_alpha 1 --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_mix_alpha 0.10 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 0.1 --recommender_allow_modal_grad 0
na_m020_reg01_i005_b001_h64_d01	--completion_gate_mode rank_residual_allgate --completion_gate_no_residual_alpha 1 --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_mix_alpha 0.20 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 0.1 --recommender_allow_modal_grad 0
na_m035_reg01_i005_b001_h64_d01	--completion_gate_mode rank_residual_allgate --completion_gate_no_residual_alpha 1 --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_mix_alpha 0.35 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 0.1 --recommender_allow_modal_grad 0
na_m050_reg01_i005_b001_h64_d01	--completion_gate_mode rank_residual_allgate --completion_gate_no_residual_alpha 1 --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_mix_alpha 0.50 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 0.1 --recommender_allow_modal_grad 0
na_m070_reg01_i005_b001_h64_d01	--completion_gate_mode rank_residual_allgate --completion_gate_no_residual_alpha 1 --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_mix_alpha 0.70 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 0.1 --recommender_allow_modal_grad 0
na_m035_reg001_i005_b001_h64_d01	--completion_gate_mode rank_residual_allgate --completion_gate_no_residual_alpha 1 --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_mix_alpha 0.35 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 0.01 --recommender_allow_modal_grad 0
na_m035_reg1_i005_b001_h64_d01	--completion_gate_mode rank_residual_allgate --completion_gate_no_residual_alpha 1 --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_mix_alpha 0.35 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 1.0 --recommender_allow_modal_grad 0
na_m050_reg1_i005_b001_h64_d01	--completion_gate_mode rank_residual_allgate --completion_gate_no_residual_alpha 1 --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_mix_alpha 0.50 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 1.0 --recommender_allow_modal_grad 0
na_m035_reg01_i001_b001_h64_d01	--completion_gate_mode rank_residual_allgate --completion_gate_no_residual_alpha 1 --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_mix_alpha 0.35 --completion_gate_identity_coeff 0.01 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 0.1 --recommender_allow_modal_grad 0
na_m035_reg01_i010_b001_h64_d01	--completion_gate_mode rank_residual_allgate --completion_gate_no_residual_alpha 1 --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_mix_alpha 0.35 --completion_gate_identity_coeff 0.10 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 0.1 --recommender_allow_modal_grad 0
na_m035_reg01_i005_b000_h64_d01	--completion_gate_mode rank_residual_allgate --completion_gate_no_residual_alpha 1 --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_mix_alpha 0.35 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.00 --completion_gate_reg_coeff 0.1 --recommender_allow_modal_grad 0
na_m035_reg01_i005_b005_h64_d01	--completion_gate_mode rank_residual_allgate --completion_gate_no_residual_alpha 1 --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_mix_alpha 0.35 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.05 --completion_gate_reg_coeff 0.1 --recommender_allow_modal_grad 0
na_m035_reg01_i005_b001_h32_d01	--completion_gate_mode rank_residual_allgate --completion_gate_no_residual_alpha 1 --completion_gate_hidden_dim 32 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_mix_alpha 0.35 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 0.1 --recommender_allow_modal_grad 0
na_m035_reg01_i005_b001_h128_d01	--completion_gate_mode rank_residual_allgate --completion_gate_no_residual_alpha 1 --completion_gate_hidden_dim 128 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_mix_alpha 0.35 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 0.1 --recommender_allow_modal_grad 0
na_m035_reg01_i005_b001_h64_d00	--completion_gate_mode rank_residual_allgate --completion_gate_no_residual_alpha 1 --completion_gate_hidden_dim 64 --completion_gate_dropout 0.0 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_mix_alpha 0.35 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 0.1 --recommender_allow_modal_grad 0
na_m035_reg01_i005_b001_h64_ctxoff	--completion_gate_mode rank_residual_allgate --completion_gate_no_residual_alpha 1 --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 0 --completion_gate_item_context_source off --completion_gate_mix_alpha 0.35 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 0.1 --recommender_allow_modal_grad 0
na_m035_reg01_i005_b001_h64_idctx	--completion_gate_mode rank_residual_allgate --completion_gate_no_residual_alpha 1 --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source id_embedding --completion_gate_mix_alpha 0.35 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 0.1 --recommender_allow_modal_grad 0
na_learnmix_cap05_reg01_i005_b001_h64	--completion_gate_mode rank_residual_allgate --completion_gate_no_residual_alpha 1 --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_mix_alpha 0.35 --completion_gate_learn_mix 1 --completion_gate_mix_max 0.5 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 0.1 --recommender_allow_modal_grad 0
EOF
}

config_for_dataset() {
  echo "configs/${1}/stage2_decoder_mm.yaml"
}

runtime_args_for_dataset() {
  local dataset="$1"
  case "${dataset}" in
    clothing)
      echo "--epoch 200 --early_stop 20 --eva_interval 1 --batch_size 2048 --lr 0.01 --lr_rec 0.01 --lr_imp 0.0002 --lr_decoder 0.00005 --reg_coeff 0.01 --penalty_coeff 1.0 --max_info_coeff 0.01 --min_info_coeff 0.000001 --strict_probe_test_interval 10"
      ;;
    sports)
      echo "--epoch 200 --early_stop 20 --eva_interval 1 --batch_size 256 --lr 0.001 --lr_rec 0.001 --lr_imp 0.0002 --lr_decoder 0.00005 --reg_coeff 0.0001 --penalty_coeff 50 --max_info_coeff 0.05 --min_info_coeff 0.05 --strict_probe_test_interval 10"
      ;;
    microlens100k)
      echo "--epoch 200 --early_stop 200 --eva_interval 10 --batch_size 2048 --lr 0.01 --lr_rec 0.01 --lr_imp 0.0002 --lr_decoder 0.00005 --reg_coeff 0.01 --penalty_coeff 1.0 --max_info_coeff 0.01 --min_info_coeff 0.000001 --modality_bpr_coeff 2.5 --strict_probe_test_interval 0"
      ;;
    *)
      echo "unsupported dataset: ${dataset}" >&2
      return 2
      ;;
  esac
}

imputer_ckpt_for_seed() {
  local dataset="$1"
  local seed="$2"
  case "${dataset}:${seed}" in
    clothing:*) echo "exp_report/clothing/stage1_2_clothing_mmrec_fixed_seed${seed}_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage1_2_clothing_mmrec_fixed_seed${seed}_mmrec_clothing_mm_fixedmissing_20260521_052129_imputer_backprop_50_epoch19.pth" ;;
    sports:*) echo "exp_report/sports/stage1_2_sports_imputer_backprop_decoder_v2_mmrec_sports_mm_mr0.3_seed${seed}_mmrec_sports_mm_fixedmissing_20260524_165817/ckpt/stage1_2_sports_imputer_backprop_decoder_v2_mmrec_sports_mm_mr0.3_seed${seed}_mmrec_sports_mm_fixedmissing_20260524_165817_imputer_backprop_50_epoch19.pth" ;;
    microlens100k:1) echo "exp_report/microlens100k/stage1_2_microlens100k_imputer_backprop_decoder_v2_mmrec_microlens100k_3modal_seed1_base_20260523_012808/ckpt/stage1_2_microlens100k_imputer_backprop_decoder_v2_mmrec_microlens100k_3modal_seed1_base_20260523_012808_imputer_backprop_50_epoch11.pth" ;;
    microlens100k:12) echo "exp_report/microlens100k/stage1_2_microlens100k_imputer_backprop_decoder_v2_mmrec_microlens100k_3modal_seed12_mbpr25_20260523_033613/ckpt/stage1_2_microlens100k_imputer_backprop_decoder_v2_mmrec_microlens100k_3modal_seed12_mbpr25_20260523_033613_imputer_backprop_50_epoch18.pth" ;;
    microlens100k:123) echo "exp_report/microlens100k/stage1_2_microlens100k_imputer_backprop_decoder_v2_mmrec_microlens100k_3modal_seed123_mbpr25_20260523_033613/ckpt/stage1_2_microlens100k_imputer_backprop_decoder_v2_mmrec_microlens100k_3modal_seed123_mbpr25_20260523_033613_imputer_backprop_50_epoch13.pth" ;;
    microlens100k:1234) echo "exp_report/microlens100k/stage1_2_microlens100k_imputer_backprop_decoder_v2_mmrec_microlens100k_3modal_seed1234_mbpr25_20260523_033613/ckpt/stage1_2_microlens100k_imputer_backprop_decoder_v2_mmrec_microlens100k_3modal_seed1234_mbpr25_20260523_033613_imputer_backprop_50_epoch15.pth" ;;
    microlens100k:12345) echo "exp_report/microlens100k/stage1_2_microlens100k_imputer_backprop_decoder_v2_mmrec_microlens100k_3modal_seed12345_mbpr25_20260523_033613/ckpt/stage1_2_microlens100k_imputer_backprop_decoder_v2_mmrec_microlens100k_3modal_seed12345_mbpr25_20260523_033613_imputer_backprop_50_epoch11.pth" ;;
    *) echo "missing imputer mapping for ${dataset} seed ${seed}" >&2; return 2 ;;
  esac
}

base_log_for_seed() {
  local dataset="$1"
  local seed="$2"
  case "${dataset}:${seed}" in
    clothing:1) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed1_mmrec_clothing_mm_fixedmissing_20260521_052129/log/run_20260521_052413.log" ;;
    clothing:12) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed12_mmrec_clothing_mm_fixedmissing_20260521_052129/log/run_20260521_052411.log" ;;
    clothing:123) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed123_mmrec_clothing_mm_fixedmissing_20260521_052129/log/run_20260521_060217.log" ;;
    clothing:1234) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed1234_mmrec_clothing_mm_fixedmissing_20260521_052129/log/run_20260521_055340.log" ;;
    clothing:12345) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed12345_mmrec_clothing_mm_fixedmissing_20260521_052129/log/run_20260521_063053.log" ;;
    sports:1) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed1_mmrec_sports_mm_fixedmissing_20260524_165817/log/run_20260524_165955.log" ;;
    sports:12) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed12_mmrec_sports_mm_fixedmissing_20260524_165817/log/run_20260524_190838.log" ;;
    sports:123) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed123_mmrec_sports_mm_fixedmissing_20260524_165817/log/run_20260524_211736.log" ;;
    sports:1234) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed1234_mmrec_sports_mm_fixedmissing_20260524_165817/log/run_20260524_232628.log" ;;
    sports:12345) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed12345_mmrec_sports_mm_fixedmissing_20260524_165817/log/run_20260525_012732.log" ;;
    microlens100k:1) echo "exp_report/fusion_norm_ablation/microlens100k_allgate_score_5seed_rest_20260529/microlens100k_base_seed1.log" ;;
    microlens100k:12) echo "exp_report/fusion_norm_ablation/microlens100k_allgate_score_2seed_20260528/microlens100k_base_seed12.log" ;;
    microlens100k:123) echo "exp_report/fusion_norm_ablation/microlens100k_allgate_score_2seed_20260528/microlens100k_base_seed123.log" ;;
    microlens100k:1234) echo "exp_report/fusion_norm_ablation/microlens100k_allgate_score_5seed_rest_20260529/microlens100k_base_seed1234.log" ;;
    microlens100k:12345) echo "exp_report/fusion_norm_ablation/microlens100k_allgate_score_5seed_rest_20260529/microlens100k_base_seed12345.log" ;;
    *) echo "missing base log mapping for ${dataset} seed ${seed}" >&2; return 2 ;;
  esac
}

base_ckpt_for_seed() {
  local dataset="$1"
  local seed="$2"
  case "${dataset}:${seed}" in
    clothing:1) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed1_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage2_clothing_mmrec_fixed_seed1_mmrec_clothing_mm_fixedmissing_20260521_052129_recommender_1.0_epoch169.pth" ;;
    clothing:12) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed12_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage2_clothing_mmrec_fixed_seed12_mmrec_clothing_mm_fixedmissing_20260521_052129_recommender_1.0_epoch122.pth" ;;
    clothing:123) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed123_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage2_clothing_mmrec_fixed_seed123_mmrec_clothing_mm_fixedmissing_20260521_052129_recommender_1.0_epoch121.pth" ;;
    clothing:1234) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed1234_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage2_clothing_mmrec_fixed_seed1234_mmrec_clothing_mm_fixedmissing_20260521_052129_recommender_1.0_epoch137.pth" ;;
    clothing:12345) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed12345_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage2_clothing_mmrec_fixed_seed12345_mmrec_clothing_mm_fixedmissing_20260521_052129_recommender_1.0_epoch121.pth" ;;
    sports:1) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed1_mmrec_sports_mm_fixedmissing_20260524_165817/ckpt/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed1_mmrec_sports_mm_fixedmissing_20260524_165817_recommender_50_epoch158.pth" ;;
    sports:12) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed12_mmrec_sports_mm_fixedmissing_20260524_165817/ckpt/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed12_mmrec_sports_mm_fixedmissing_20260524_165817_recommender_50_epoch160.pth" ;;
    sports:123) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed123_mmrec_sports_mm_fixedmissing_20260524_165817/ckpt/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed123_mmrec_sports_mm_fixedmissing_20260524_165817_recommender_50_epoch158.pth" ;;
    sports:1234) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed1234_mmrec_sports_mm_fixedmissing_20260524_165817/ckpt/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed1234_mmrec_sports_mm_fixedmissing_20260524_165817_recommender_50_epoch148.pth" ;;
    sports:12345) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed12345_mmrec_sports_mm_fixedmissing_20260524_165817/ckpt/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed12345_mmrec_sports_mm_fixedmissing_20260524_165817_recommender_50_epoch196.pth" ;;
    microlens100k:*) find "exp_report/microlens100k/stage2_microlens100k_base_seed${seed}_microlens100k_allgate_score_"*/ckpt -maxdepth 1 -name '*.pth' -print 2>/dev/null | sort -V | tail -1 ;;
    *) echo "missing base ckpt mapping for ${dataset} seed ${seed}" >&2; return 2 ;;
  esac
}

latest_ckpt_for_suffix() {
  local dataset="$1"
  local suffix="$2"
  find "exp_report/${dataset}/${suffix}/ckpt" -maxdepth 1 -name '*.pth' -print | sort -V | tail -1
}

run_task() {
  local gpu="$1"
  local dataset="$2"
  local phase="$3"
  local candidate="$4"
  local seed="$5"
  local args="$6"

  local config suffix train_dir train_log eval_dir eval_log imputer_ckpt base_log base_ckpt fusion_ckpt
  config="$(config_for_dataset "${dataset}")"
  suffix="stage2_${dataset}_noalpha_${phase}_${candidate}_seed${seed}_${RUN_TAG}"
  train_dir="${OUT_DIR}/logs/${phase}/${dataset}/${candidate}"
  train_log="${train_dir}/seed${seed}.log"
  eval_dir="${OUT_DIR}/eval/${phase}/${dataset}/${candidate}"
  eval_log="${eval_dir}/seed${seed}_test.log"
  mkdir -p "${train_dir}" "${eval_dir}"

  imputer_ckpt="$(imputer_ckpt_for_seed "${dataset}" "${seed}")"
  base_log="$(base_log_for_seed "${dataset}" "${seed}")"
  base_ckpt="$(base_ckpt_for_seed "${dataset}" "${seed}")"
  for required in "${config}" "${imputer_ckpt}" "${base_log}" "${base_ckpt}"; do
    if [[ ! -f "${required}" ]]; then
      echo "missing input: ${required}" >&2
      return 1
    fi
  done

  if [[ -f "${eval_log}" ]] && grep -q '^1\.0000' "${eval_log}" 2>/dev/null; then
    log "skip finished dataset=${dataset} phase=${phase} candidate=${candidate} seed=${seed}"
    return
  fi

  if [[ ! -f "${train_log}" ]] || ! grep -q "best epoch" "${train_log}" 2>/dev/null; then
    log "train dataset=${dataset} phase=${phase} candidate=${candidate} seed=${seed} gpu=${gpu}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "CUDA_VISIBLE_DEVICES=${gpu} main.py --config ${config} --suffix ${suffix} ${args}" | tee "${train_log}"
    else
      read -r -a runtime_args <<< "$(runtime_args_for_dataset "${dataset}")"
      read -r -a gate_args <<< "${args}"
      CUDA_VISIBLE_DEVICES="${gpu}" .venv/bin/python main.py \
        --config "${config}" \
        --suffix "${suffix}" \
        --dataset "${dataset}" \
        --exp_mode mm \
        --device_id 0 \
        --seed "${seed}" \
        --dataset_seed "${DATASET_SEED}" \
        --train_stage recommender \
        --freeze_imputer 1 \
        --freeze_decoder 1 \
        --disable_imputation 0 \
        --feature_bridge_mode raw_decoder \
        --gcn_frontend_mode original_linear \
        --imputer_ckpt "${imputer_ckpt}" \
        --evaluation_protocol strict \
        --selection_mode val \
        --save 1 \
        "${runtime_args[@]}" \
        "${gate_args[@]}" \
        2>&1 | tee "${train_log}"
    fi
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "0.4000 hr@20 0.00000 0.00000" > "${eval_log}"
    echo "1.0000 hr@20 0.00000 0.00000" >> "${eval_log}"
    return
  fi

  fusion_ckpt="$(latest_ckpt_for_suffix "${dataset}" "${suffix}")"
  if [[ ! -f "${fusion_ckpt}" ]]; then
    echo "missing fusion checkpoint for ${suffix}" >&2
    return 1
  fi

  log "eval dataset=${dataset} phase=${phase} candidate=${candidate} seed=${seed} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" .venv/bin/python tools/evaluate_assemble_score_fusion.py \
    --base-log "${base_log}" \
    --base-ckpt "${base_ckpt}" \
    --fusion-log "${train_log}" \
    --fusion-ckpt "${fusion_ckpt}" \
    --device-id 0 \
    --split test \
    --normalize zscore \
    --alphas "${ASSEMBLE_ALPHAS}" \
    > "${eval_log}" 2>&1
}

write_task_files() {
  local phase="$1"
  local tasks_file="${OUT_DIR}/tasks/${phase}.tsv"
  : > "${tasks_file}"
  if [[ "${phase}" == "search" ]]; then
    for dataset in "${DATASETS[@]}"; do
      while IFS=$'\t' read -r candidate args; do
        for seed in "${SEARCH_SEEDS[@]}"; do
          printf '%s\t%s\t%s\t%s\t%s\n' "${dataset}" "${phase}" "${candidate}" "${seed}" "${args}" >> "${tasks_file}"
        done
      done < <(candidate_table)
    done
  else
    for dataset in "${DATASETS[@]}"; do
      local selected="${OUT_DIR}/${dataset}_selected_candidate.txt"
      if [[ ! -f "${selected}" ]]; then
        echo "missing selected candidate for ${dataset}: ${selected}" >&2
        exit 1
      fi
      local candidate args
      candidate="$(< "${selected}")"
      args="$(candidate_table | awk -F '\t' -v n="${candidate}" '$1 == n {print $2}')"
      if [[ -z "${args}" ]]; then
        echo "failed to resolve args for ${candidate}" >&2
        exit 1
      fi
      for seed in "${FULL_SEEDS[@]}"; do
        printf '%s\t%s\t%s\t%s\t%s\n' "${dataset}" "${phase}" "${candidate}" "${seed}" "${args}" >> "${tasks_file}"
      done
    done
  fi
}

run_task_file_on_gpu() {
  local gpu="$1"
  local file="$2"
  while IFS=$'\t' read -r dataset phase candidate seed args; do
    [[ -n "${dataset}" ]] || continue
    run_task "${gpu}" "${dataset}" "${phase}" "${candidate}" "${seed}" "${args}"
  done < "${file}"
}

run_phase() {
  local phase="$1"
  local tasks_file="${OUT_DIR}/tasks/${phase}.tsv"
  write_task_files "${phase}"
  for idx in "${!GPUS[@]}"; do
    : > "${OUT_DIR}/tasks/${phase}_gpu${idx}.tsv"
  done
  local line_no=0
  while IFS= read -r line; do
    local slot=$((line_no % ${#GPUS[@]}))
    printf '%s\n' "${line}" >> "${OUT_DIR}/tasks/${phase}_gpu${slot}.tsv"
    line_no=$((line_no + 1))
  done < "${tasks_file}"

  log "${phase} phase queued ${line_no} tasks on GPUs: ${GPUS_STR}"
  local pids=()
  for idx in "${!GPUS[@]}"; do
    run_task_file_on_gpu "${GPUS[$idx]}" "${OUT_DIR}/tasks/${phase}_gpu${idx}.tsv" \
      > "${OUT_DIR}/tasks/${phase}_gpu${idx}.worker.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
}

summarize_phase() {
  local phase="$1"
  .venv/bin/python - "${OUT_DIR}" "${phase}" "${PRIMARY_ALPHA}" "${DATASETS[@]}" <<'PY'
from pathlib import Path
import statistics
import sys

out_dir = Path(sys.argv[1])
phase = sys.argv[2]
primary_alpha = float(sys.argv[3])
datasets = sys.argv[4:]
targets = {
    "clothing": (0.07638, 0.03425),
    "sports": (0.09918, 0.04460),
    "microlens100k": (0.04257, 0.01552),
}

def parse_eval(path):
    rows = {}
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) == 4:
            try:
                alpha = float(parts[0])
                rows[alpha] = (float(parts[2]), float(parts[3]))
            except ValueError:
                pass
    return rows

for dataset in datasets:
    base = out_dir / "eval" / phase / dataset
    rows = []
    for log_path in sorted(base.glob("*/seed*_test.log")):
        candidate = log_path.parent.name
        seed = log_path.stem.replace("seed", "").replace("_test", "")
        parsed = parse_eval(log_path)
        if primary_alpha not in parsed or not parsed:
            rows.append([candidate, seed, "", "", "", "", "", "", "missing_eval"])
            continue
        pr, pn = parsed[primary_alpha]
        ar, an = parsed.get(1.0, ("", ""))
        best_alpha, (best_r, best_n) = max(parsed.items(), key=lambda item: (item[1][0], item[1][1]))
        rows.append([candidate, seed, pr, pn, ar, an, best_alpha, best_r, best_n, "ok"])

    summary = out_dir / f"{dataset}_{phase}_summary.tsv"
    with summary.open("w") as f:
        f.write("candidate\tseed\tprimary_R20\tprimary_N20\taux_R20\taux_N20\tbest_alpha\tbest_R20\tbest_N20\tstatus\n")
        for row in rows:
            f.write("\t".join(map(str, row)) + "\n")

    agg = {}
    for row in rows:
        if row[-1] != "ok":
            continue
        agg.setdefault(row[0], []).append(row)
    ranked = []
    target_r, target_n = targets.get(dataset, ("", ""))
    for candidate, vals in agg.items():
        primary_r = [float(v[2]) for v in vals]
        primary_n = [float(v[3]) for v in vals]
        best_r = [float(v[7]) for v in vals]
        best_n = [float(v[8]) for v in vals]
        ranked.append((
            statistics.mean(primary_r),
            statistics.mean(primary_n),
            statistics.mean(best_r),
            statistics.mean(best_n),
            candidate,
            len(vals),
        ))
    ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)

    agg_path = out_dir / f"{dataset}_{phase}_agg.tsv"
    with agg_path.open("w") as f:
        f.write("rank\tcandidate\tok_count\tmean_primary_R20\tmean_primary_N20\tmean_best_R20\tmean_best_N20\ttarget_R20\ttarget_N20\tprimary_R_gap\tprimary_N_gap\n")
        for rank, (mean_pr, mean_pn, mean_br, mean_bn, candidate, count) in enumerate(ranked, 1):
            gap_r = "" if target_r == "" else f"{mean_pr - target_r:.5f}"
            gap_n = "" if target_n == "" else f"{mean_pn - target_n:.5f}"
            f.write(
                f"{rank}\t{candidate}\t{count}\t{mean_pr:.5f}\t{mean_pn:.5f}\t"
                f"{mean_br:.5f}\t{mean_bn:.5f}\t{target_r}\t{target_n}\t{gap_r}\t{gap_n}\n"
            )

    if ranked and phase == "search":
        (out_dir / f"{dataset}_selected_candidate.txt").write_text(ranked[0][4] + "\n")
PY
}

log "run_tag=${RUN_TAG} out=${OUT_DIR}"
run_phase search
summarize_phase search
run_phase full
summarize_phase full
log "done run_tag=${RUN_TAG}"
