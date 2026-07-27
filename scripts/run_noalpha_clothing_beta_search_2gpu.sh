#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PATH="${ROOT_DIR}/.venv/bin:${PATH}"
export PYTHONUNBUFFERED=1
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

RUN_TAG="${RUN_TAG:-noalpha_clothing_beta_$(date +%Y%m%d_%H%M%S)}"
GPUS_STR="${GPUS:-2 3}"
SEARCH_SEEDS_STR="${SEARCH_SEEDS:-12 123}"
FULL_SEEDS_STR="${FULL_SEEDS:-1 12 123 1234 12345}"
ASSEMBLE_ALPHAS="${ASSEMBLE_ALPHAS:-0,0.2,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,1}"
PRIMARY_ALPHA="${PRIMARY_ALPHA:-0.4}"
DRY_RUN="${DRY_RUN:-0}"
SEARCH_EPOCHS="${SEARCH_EPOCHS:-50}"
SEARCH_EARLY_STOP="${SEARCH_EARLY_STOP:-8}"
SEARCH_EVA_INTERVAL="${SEARCH_EVA_INTERVAL:-1}"
FULL_EPOCHS="${FULL_EPOCHS:-200}"
FULL_EARLY_STOP="${FULL_EARLY_STOP:-20}"
FULL_EVA_INTERVAL="${FULL_EVA_INTERVAL:-1}"
CANDIDATE_SET="${CANDIDATE_SET:-full}"
FULL_CANDIDATES_STR="${FULL_CANDIDATES:-}"
SKIP_SEARCH="${SKIP_SEARCH:-0}"

read -r -a GPUS <<< "${GPUS_STR}"
read -r -a SEARCH_SEEDS <<< "${SEARCH_SEEDS_STR}"
read -r -a FULL_SEEDS <<< "${FULL_SEEDS_STR}"

OUT_DIR="${ROOT_DIR}/exp_report/noalpha_clothing_beta_search/${RUN_TAG}"
mkdir -p "${OUT_DIR}/tasks" "${OUT_DIR}/logs/search" "${OUT_DIR}/logs/full" "${OUT_DIR}/eval/search" "${OUT_DIR}/eval/full"

log() {
  date +"[noalpha-clothing-beta] %Y-%m-%d %H:%M:%S $*"
}

make_args() {
  local beta="$1"
  local reg="$2"
  local identity="$3"
  local balance="$4"
  local hidden="$5"
  local dropout="$6"
  local ctx="$7"
  local learn_mix="$8"
  local mix_max="$9"
  local ctx_args=("--completion_gate_use_item_context" "1" "--completion_gate_item_context_source" "shared_mean")
  if [[ "${ctx}" == "off" ]]; then
    ctx_args=("--completion_gate_use_item_context" "0" "--completion_gate_item_context_source" "off")
  elif [[ "${ctx}" == "id" ]]; then
    ctx_args=("--completion_gate_use_item_context" "1" "--completion_gate_item_context_source" "id_embedding")
  fi

  local args=(
    --completion_gate_mode rank_residual_allgate
    --completion_gate_no_residual_alpha 1
    --completion_gate_hidden_dim "${hidden}"
    --completion_gate_dropout "${dropout}"
    --completion_gate_init_logit 0.0
    --completion_gate_detach_inputs 1
    "${ctx_args[@]}"
    --completion_gate_mix_alpha "${beta}"
    --completion_gate_identity_coeff "${identity}"
    --completion_gate_balance_coeff "${balance}"
    --completion_gate_reg_coeff "${reg}"
    --recommender_allow_modal_grad 0
  )
  if [[ "${learn_mix}" == "1" ]]; then
    args+=(--completion_gate_learn_mix 1 --completion_gate_mix_max "${mix_max}")
  fi
  printf '%q ' "${args[@]}"
}

candidate_table() {
  local beta beta_tag

  if [[ "${CANDIDATE_SET}" == "train_reg_beta_refine" ]]; then
    local gate_name gate_args profile_name profile_args
    while IFS=$'\t' read -r gate_name gate_args; do
      while IFS=$'\t' read -r profile_name profile_args; do
        printf '%s_%s\t%s %s\n' "${gate_name}" "${profile_name}" "${gate_args}" "${profile_args}"
      done <<'EOF'
reg0015	--reg_coeff 0.015
reg0020	--reg_coeff 0.020
reg0025	--reg_coeff 0.025
reg0030	--reg_coeff 0.030
reg0040	--reg_coeff 0.040
reg0050	--reg_coeff 0.050
reg0070	--reg_coeff 0.070
reg0030_mbpr050	--reg_coeff 0.030 --modality_bpr_coeff 0.5
EOF
    done <<EOF
b0p60_reg000_i005_bal001_h64_d01_shared	$(make_args 0.60 0.0 0.05 0.01 64 0.1 shared 0 1.0)
b0p70_reg000_i005_bal001_h64_d01_shared	$(make_args 0.70 0.0 0.05 0.01 64 0.1 shared 0 1.0)
b0p80_reg000_i005_bal001_h64_d01_shared	$(make_args 0.80 0.0 0.05 0.01 64 0.1 shared 0 1.0)
b0p85_reg000_i005_bal001_h64_d01_shared	$(make_args 0.85 0.0 0.05 0.01 64 0.1 shared 0 1.0)
b1p00_reg000_i005_bal001_h64_d01_shared	$(make_args 1.00 0.0 0.05 0.01 64 0.1 shared 0 1.0)
b0p70_reg003_i005_bal001_h64_d01_shared	$(make_args 0.70 0.03 0.05 0.01 64 0.1 shared 0 1.0)
b0p85_reg003_i005_bal001_h64_d01_shared	$(make_args 0.85 0.03 0.05 0.01 64 0.1 shared 0 1.0)
b0p85_reg01_i001_bal001_h64_d01_shared	$(make_args 0.85 0.1 0.01 0.01 64 0.1 shared 0 1.0)
EOF
    return
  fi

  if [[ "${CANDIDATE_SET}" == "best_local_refine" ]]; then
    local gate_name gate_args profile_name profile_args
    while IFS=$'\t' read -r gate_name gate_args; do
      while IFS=$'\t' read -r profile_name profile_args; do
        printf '%s_%s\t%s %s\n' "${gate_name}" "${profile_name}" "${gate_args}" "${profile_args}"
      done <<'EOF'
reg0010	--reg_coeff 0.010
reg0012	--reg_coeff 0.012
reg0015	--reg_coeff 0.015
reg0018	--reg_coeff 0.018
reg0020	--reg_coeff 0.020
reg0025	--reg_coeff 0.025
EOF
    done <<EOF
b0p65_reg001_i005_bal001_h64_d01_shared	$(make_args 0.65 0.01 0.05 0.01 64 0.1 shared 0 1.0)
b0p65_reg003_i005_bal001_h64_d01_shared	$(make_args 0.65 0.03 0.05 0.01 64 0.1 shared 0 1.0)
b0p65_reg005_i005_bal001_h64_d01_shared	$(make_args 0.65 0.05 0.05 0.01 64 0.1 shared 0 1.0)
b0p70_reg001_i005_bal001_h64_d01_shared	$(make_args 0.70 0.01 0.05 0.01 64 0.1 shared 0 1.0)
b0p70_reg003_i005_bal001_h64_d01_shared	$(make_args 0.70 0.03 0.05 0.01 64 0.1 shared 0 1.0)
b0p70_reg005_i005_bal001_h64_d01_shared	$(make_args 0.70 0.05 0.05 0.01 64 0.1 shared 0 1.0)
b0p75_reg001_i005_bal001_h64_d01_shared	$(make_args 0.75 0.01 0.05 0.01 64 0.1 shared 0 1.0)
b0p75_reg003_i005_bal001_h64_d01_shared	$(make_args 0.75 0.03 0.05 0.01 64 0.1 shared 0 1.0)
b0p75_reg005_i005_bal001_h64_d01_shared	$(make_args 0.75 0.05 0.05 0.01 64 0.1 shared 0 1.0)
EOF
    return
  fi

  if [[ "${CANDIDATE_SET}" == "priority_beta_refine" ]]; then
    local gate_name gate_args profile_name profile_args
    while IFS=$'\t' read -r gate_name gate_args; do
      while IFS=$'\t' read -r profile_name profile_args; do
        printf '%s_%s\t%s %s\n' "${gate_name}" "${profile_name}" "${gate_args}" "${profile_args}"
      done <<'EOF'
reg0012	--reg_coeff 0.012
reg0015	--reg_coeff 0.015
reg0018	--reg_coeff 0.018
reg0020	--reg_coeff 0.020
reg0025	--reg_coeff 0.025
EOF
    done <<EOF
b0p70_g003_i005_bal001_h64_d01_shared	$(make_args 0.70 0.03 0.05 0.01 64 0.1 shared 0 1.0)
b0p70_g005_i005_bal001_h64_d01_shared	$(make_args 0.70 0.05 0.05 0.01 64 0.1 shared 0 1.0)
b0p75_g003_i005_bal001_h64_d01_shared	$(make_args 0.75 0.03 0.05 0.01 64 0.1 shared 0 1.0)
b0p75_g005_i005_bal001_h64_d01_shared	$(make_args 0.75 0.05 0.05 0.01 64 0.1 shared 0 1.0)
b0p70_g001_i005_bal001_h64_d01_shared	$(make_args 0.70 0.01 0.05 0.01 64 0.1 shared 0 1.0)
b0p75_g001_i005_bal001_h64_d01_shared	$(make_args 0.75 0.01 0.05 0.01 64 0.1 shared 0 1.0)
b0p80_g003_i005_bal001_h64_d01_shared	$(make_args 0.80 0.03 0.05 0.01 64 0.1 shared 0 1.0)
b0p80_g005_i005_bal001_h64_d01_shared	$(make_args 0.80 0.05 0.05 0.01 64 0.1 shared 0 1.0)
b0p70_g003_i002_bal001_h64_d01_shared	$(make_args 0.70 0.03 0.02 0.01 64 0.1 shared 0 1.0)
b0p75_g005_i002_bal001_h64_d01_shared	$(make_args 0.75 0.05 0.02 0.01 64 0.1 shared 0 1.0)
b0p70_g003_i005_bal000_h64_d01_shared	$(make_args 0.70 0.03 0.05 0.00 64 0.1 shared 0 1.0)
b0p75_g005_i005_bal000_h64_d01_shared	$(make_args 0.75 0.05 0.05 0.00 64 0.1 shared 0 1.0)
b0p70_g003_i005_bal001_h128_d01_shared	$(make_args 0.70 0.03 0.05 0.01 128 0.1 shared 0 1.0)
b0p75_g005_i005_bal001_h128_d01_shared	$(make_args 0.75 0.05 0.05 0.01 128 0.1 shared 0 1.0)
b0p70_g003_i005_bal001_h64_d005_shared	$(make_args 0.70 0.03 0.05 0.01 64 0.05 shared 0 1.0)
b0p75_g005_i005_bal001_h64_d005_shared	$(make_args 0.75 0.05 0.05 0.01 64 0.05 shared 0 1.0)
EOF
    return
  fi

  if [[ "${CANDIDATE_SET}" == "capacity_reg_refine" ]]; then
    local gate_name gate_args profile_name profile_args
    while IFS=$'\t' read -r gate_name gate_args; do
      while IFS=$'\t' read -r profile_name profile_args; do
        printf '%s_%s\t%s %s\n' "${gate_name}" "${profile_name}" "${gate_args}" "${profile_args}"
      done <<'EOF'
reg0012	--reg_coeff 0.012
reg0015	--reg_coeff 0.015
reg0018	--reg_coeff 0.018
EOF
    done < <(
      local beta beta_tag gate_reg gate_reg_tag identity identity_tag hidden dropout dropout_tag

      for beta in 0.65 0.70 0.75 0.80; do
        beta_tag="${beta/./p}"
        for gate_reg in 0.02 0.03 0.05; do
          gate_reg_tag="${gate_reg/./p}"
          for identity in 0.02 0.05; do
            identity_tag="${identity/./p}"
            printf 'b%s_g%s_i%s_bal001_h64_d01_shared\t%s\n' \
              "${beta_tag}" "${gate_reg_tag}" "${identity_tag}" \
              "$(make_args "${beta}" "${gate_reg}" "${identity}" 0.01 64 0.1 shared 0 1.0)"
          done
        done
      done

      for beta in 0.70 0.75; do
        beta_tag="${beta/./p}"
        for gate_reg in 0.03 0.05; do
          gate_reg_tag="${gate_reg/./p}"
          printf 'b%s_g%s_i005_bal000_h64_d01_shared\t%s\n' \
            "${beta_tag}" "${gate_reg_tag}" \
            "$(make_args "${beta}" "${gate_reg}" 0.05 0.00 64 0.1 shared 0 1.0)"
          printf 'b%s_g%s_i008_bal001_h64_d01_shared\t%s\n' \
            "${beta_tag}" "${gate_reg_tag}" \
            "$(make_args "${beta}" "${gate_reg}" 0.08 0.01 64 0.1 shared 0 1.0)"
        done
      done

      for beta in 0.70 0.75; do
        beta_tag="${beta/./p}"
        for gate_reg in 0.03 0.05; do
          gate_reg_tag="${gate_reg/./p}"
          for hidden in 64 128; do
            for dropout in 0.0 0.05; do
              dropout_tag="${dropout/./p}"
              printf 'b%s_g%s_i005_bal001_h%s_d%s_shared\t%s\n' \
                "${beta_tag}" "${gate_reg_tag}" "${hidden}" "${dropout_tag}" \
                "$(make_args "${beta}" "${gate_reg}" 0.05 0.01 "${hidden}" "${dropout}" shared 0 1.0)"
            done
          done
          printf 'b%s_g%s_i005_bal001_h128_d01_shared\t%s\n' \
            "${beta_tag}" "${gate_reg_tag}" \
            "$(make_args "${beta}" "${gate_reg}" 0.05 0.01 128 0.1 shared 0 1.0)"
        done
      done
    )
    return
  fi

  if [[ "${CANDIDATE_SET}" == "train_refine" ]]; then
    local gate_name gate_args profile_name profile_args
    while IFS=$'\t' read -r gate_name gate_args; do
      while IFS=$'\t' read -r profile_name profile_args; do
        printf '%s_%s\t%s %s\n' "${gate_name}" "${profile_name}" "${gate_args}" "${profile_args}"
      done <<'EOF'
base
lr005	--lr 0.005 --lr_rec 0.005
lr015	--lr 0.015 --lr_rec 0.015
lr020	--lr 0.02 --lr_rec 0.02
reg0001	--reg_coeff 0.001
reg0003	--reg_coeff 0.003
reg003	--reg_coeff 0.03
batch1024	--batch_size 1024
batch4096	--batch_size 4096
mbpr000	--modality_bpr_coeff 0.0
mbpr050	--modality_bpr_coeff 0.5
mbpr100	--modality_bpr_coeff 1.0
mbpr250	--modality_bpr_coeff 2.5
alpharec000	--alpha_rec 0.0
alpharec020	--alpha_rec 0.2
modalgrad1	--recommender_allow_modal_grad 1
EOF
    done <<EOF
b0p85_reg01_i001_bal001_h64_d01_shared	$(make_args 0.85 0.1 0.01 0.01 64 0.1 shared 0 1.0)
b0p85_reg01_i005_bal001_h64_d00_shared	$(make_args 0.85 0.1 0.05 0.01 64 0.0 shared 0 1.0)
b1p00_reg003_i005_bal001_h64_d01_shared	$(make_args 1.00 0.03 0.05 0.01 64 0.1 shared 0 1.0)
b0p70_reg000_i005_bal001_h64_d01_shared	$(make_args 0.70 0.0 0.05 0.01 64 0.1 shared 0 1.0)
EOF
    return
  fi

  if [[ "${CANDIDATE_SET}" == "reg_refine" ]]; then
    for beta in 0.50 0.60 0.70 0.85 1.00; do
      beta_tag="${beta/./p}"
      printf 'b%s_reg01_i005_bal001_h64_d01_shared\t%s\n' "${beta_tag}" "$(make_args "${beta}" 0.1 0.05 0.01 64 0.1 shared 0 1.0)"
    done

    for beta in 0.50 0.70 0.85 1.00; do
      beta_tag="${beta/./p}"
      for reg in 0.00 0.003 0.01 0.03 0.30 1.00; do
        reg_tag="${reg/./p}"
        printf 'b%s_reg%s_i005_bal001_h64_d01_shared\t%s\n' "${beta_tag}" "${reg_tag}" "$(make_args "${beta}" "${reg}" 0.05 0.01 64 0.1 shared 0 1.0)"
      done
    done

    for beta in 0.50 0.70 0.85; do
      beta_tag="${beta/./p}"
      for identity in 0.00 0.01 0.10 0.20; do
        identity_tag="${identity/./p}"
        printf 'b%s_reg01_i%s_bal001_h64_d01_shared\t%s\n' "${beta_tag}" "${identity_tag}" "$(make_args "${beta}" 0.1 "${identity}" 0.01 64 0.1 shared 0 1.0)"
      done
    done

    for beta in 0.50 0.70 0.85; do
      beta_tag="${beta/./p}"
      for balance in 0.00 0.005 0.05 0.10; do
        balance_tag="${balance/./p}"
        printf 'b%s_reg01_i005_bal%s_h64_d01_shared\t%s\n' "${beta_tag}" "${balance_tag}" "$(make_args "${beta}" 0.1 0.05 "${balance}" 64 0.1 shared 0 1.0)"
      done
    done

    for beta in 0.50 0.70 0.85; do
      beta_tag="${beta/./p}"
      printf 'b%s_reg01_i005_bal001_h32_d01_shared\t%s\n' "${beta_tag}" "$(make_args "${beta}" 0.1 0.05 0.01 32 0.1 shared 0 1.0)"
      printf 'b%s_reg01_i005_bal001_h128_d01_shared\t%s\n' "${beta_tag}" "$(make_args "${beta}" 0.1 0.05 0.01 128 0.1 shared 0 1.0)"
      printf 'b%s_reg01_i005_bal001_h64_d00_shared\t%s\n' "${beta_tag}" "$(make_args "${beta}" 0.1 0.05 0.01 64 0.0 shared 0 1.0)"
      printf 'b%s_reg01_i005_bal001_h64_d01_ctxoff\t%s\n' "${beta_tag}" "$(make_args "${beta}" 0.1 0.05 0.01 64 0.1 off 0 1.0)"
      printf 'b%s_reg01_i005_bal001_h64_d01_idctx\t%s\n' "${beta_tag}" "$(make_args "${beta}" 0.1 0.05 0.01 64 0.1 id 0 1.0)"
    done

    printf 'learnbeta_init050_cap10_reg01_i005_bal001_h64\t%s\n' "$(make_args 0.50 0.1 0.05 0.01 64 0.1 shared 1 1.0)"
    printf 'learnbeta_init070_cap10_reg01_i005_bal001_h64\t%s\n' "$(make_args 0.70 0.1 0.05 0.01 64 0.1 shared 1 1.0)"
    printf 'learnbeta_init085_cap10_reg01_i005_bal001_h64\t%s\n' "$(make_args 0.85 0.1 0.05 0.01 64 0.1 shared 1 1.0)"
    return
  fi

  for beta in 0.00 0.02 0.05 0.08 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.50 0.60 0.70 0.85 1.00; do
    beta_tag="${beta/./p}"
    printf 'b%s_reg01_i005_bal001_h64_d01_shared\t%s\n' "${beta_tag}" "$(make_args "${beta}" 0.1 0.05 0.01 64 0.1 shared 0 1.0)"
  done
  if [[ "${CANDIDATE_SET}" == "beta_dense" ]]; then
    return
  fi

  for beta in 0.10 0.20 0.35 0.50 0.70 1.00; do
    beta_tag="${beta/./p}"
    printf 'b%s_reg001_i005_bal001_h64_d01_shared\t%s\n' "${beta_tag}" "$(make_args "${beta}" 0.01 0.05 0.01 64 0.1 shared 0 1.0)"
    printf 'b%s_reg1_i005_bal001_h64_d01_shared\t%s\n' "${beta_tag}" "$(make_args "${beta}" 1.0 0.05 0.01 64 0.1 shared 0 1.0)"
  done

  for beta in 0.20 0.35 0.50 0.70; do
    beta_tag="${beta/./p}"
    printf 'b%s_reg01_i001_bal001_h64_d01_shared\t%s\n' "${beta_tag}" "$(make_args "${beta}" 0.1 0.01 0.01 64 0.1 shared 0 1.0)"
    printf 'b%s_reg01_i010_bal001_h64_d01_shared\t%s\n' "${beta_tag}" "$(make_args "${beta}" 0.1 0.10 0.01 64 0.1 shared 0 1.0)"
    printf 'b%s_reg01_i005_bal000_h64_d01_shared\t%s\n' "${beta_tag}" "$(make_args "${beta}" 0.1 0.05 0.00 64 0.1 shared 0 1.0)"
    printf 'b%s_reg01_i005_bal005_h64_d01_shared\t%s\n' "${beta_tag}" "$(make_args "${beta}" 0.1 0.05 0.05 64 0.1 shared 0 1.0)"
  done

  for beta in 0.20 0.35 0.50; do
    beta_tag="${beta/./p}"
    printf 'b%s_reg01_i005_bal001_h32_d01_shared\t%s\n' "${beta_tag}" "$(make_args "${beta}" 0.1 0.05 0.01 32 0.1 shared 0 1.0)"
    printf 'b%s_reg01_i005_bal001_h128_d01_shared\t%s\n' "${beta_tag}" "$(make_args "${beta}" 0.1 0.05 0.01 128 0.1 shared 0 1.0)"
    printf 'b%s_reg01_i005_bal001_h64_d00_shared\t%s\n' "${beta_tag}" "$(make_args "${beta}" 0.1 0.05 0.01 64 0.0 shared 0 1.0)"
    printf 'b%s_reg01_i005_bal001_h64_d01_ctxoff\t%s\n' "${beta_tag}" "$(make_args "${beta}" 0.1 0.05 0.01 64 0.1 off 0 1.0)"
    printf 'b%s_reg01_i005_bal001_h64_d01_idctx\t%s\n' "${beta_tag}" "$(make_args "${beta}" 0.1 0.05 0.01 64 0.1 id 0 1.0)"
  done

  printf 'learnbeta_init020_cap05_reg01_i005_bal001_h64\t%s\n' "$(make_args 0.20 0.1 0.05 0.01 64 0.1 shared 1 0.5)"
  printf 'learnbeta_init035_cap05_reg01_i005_bal001_h64\t%s\n' "$(make_args 0.35 0.1 0.05 0.01 64 0.1 shared 1 0.5)"
  printf 'learnbeta_init050_cap10_reg01_i005_bal001_h64\t%s\n' "$(make_args 0.50 0.1 0.05 0.01 64 0.1 shared 1 1.0)"
}

imputer_ckpt_for_seed() {
  local seed="$1"
  echo "exp_report/clothing/stage1_2_clothing_mmrec_fixed_seed${seed}_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage1_2_clothing_mmrec_fixed_seed${seed}_mmrec_clothing_mm_fixedmissing_20260521_052129_imputer_backprop_50_epoch19.pth"
}

base_log_for_seed() {
  local seed="$1"
  case "${seed}" in
    1) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed1_mmrec_clothing_mm_fixedmissing_20260521_052129/log/run_20260521_052413.log" ;;
    12) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed12_mmrec_clothing_mm_fixedmissing_20260521_052129/log/run_20260521_052411.log" ;;
    123) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed123_mmrec_clothing_mm_fixedmissing_20260521_052129/log/run_20260521_060217.log" ;;
    1234) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed1234_mmrec_clothing_mm_fixedmissing_20260521_052129/log/run_20260521_055340.log" ;;
    12345) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed12345_mmrec_clothing_mm_fixedmissing_20260521_052129/log/run_20260521_063053.log" ;;
    *) echo "missing base log mapping for clothing seed ${seed}" >&2; return 2 ;;
  esac
}

base_ckpt_for_seed() {
  local seed="$1"
  case "${seed}" in
    1) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed1_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage2_clothing_mmrec_fixed_seed1_mmrec_clothing_mm_fixedmissing_20260521_052129_recommender_1.0_epoch169.pth" ;;
    12) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed12_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage2_clothing_mmrec_fixed_seed12_mmrec_clothing_mm_fixedmissing_20260521_052129_recommender_1.0_epoch122.pth" ;;
    123) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed123_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage2_clothing_mmrec_fixed_seed123_mmrec_clothing_mm_fixedmissing_20260521_052129_recommender_1.0_epoch121.pth" ;;
    1234) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed1234_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage2_clothing_mmrec_fixed_seed1234_mmrec_clothing_mm_fixedmissing_20260521_052129_recommender_1.0_epoch137.pth" ;;
    12345) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed12345_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage2_clothing_mmrec_fixed_seed12345_mmrec_clothing_mm_fixedmissing_20260521_052129_recommender_1.0_epoch121.pth" ;;
    *) echo "missing base ckpt mapping for clothing seed ${seed}" >&2; return 2 ;;
  esac
}

latest_ckpt_for_suffix() {
  local suffix="$1"
  find "exp_report/clothing/${suffix}/ckpt" -maxdepth 1 -name '*.pth' -print | sort -V | tail -1
}

run_task() {
  local gpu="$1"
  local phase="$2"
  local candidate="$3"
  local seed="$4"
  local args="$5"
  local suffix="stage2_clothing_noalpha_${phase}_${candidate}_seed${seed}_${RUN_TAG}"
  local train_dir="${OUT_DIR}/logs/${phase}/${candidate}"
  local train_log="${train_dir}/seed${seed}.log"
  local eval_dir="${OUT_DIR}/eval/${phase}/${candidate}"
  local eval_log="${eval_dir}/seed${seed}_test.log"
  mkdir -p "${train_dir}" "${eval_dir}"
  local epochs early_stop eva_interval strict_probe_interval
  if [[ "${phase}" == "search" ]]; then
    epochs="${SEARCH_EPOCHS}"
    early_stop="${SEARCH_EARLY_STOP}"
    eva_interval="${SEARCH_EVA_INTERVAL}"
    strict_probe_interval=0
  else
    epochs="${FULL_EPOCHS}"
    early_stop="${FULL_EARLY_STOP}"
    eva_interval="${FULL_EVA_INTERVAL}"
    strict_probe_interval=10
  fi

  local imputer_ckpt base_log base_ckpt fusion_ckpt
  imputer_ckpt="$(imputer_ckpt_for_seed "${seed}")"
  base_log="$(base_log_for_seed "${seed}")"
  base_ckpt="$(base_ckpt_for_seed "${seed}")"
  for required in "configs/clothing/stage2_decoder_mm.yaml" "${imputer_ckpt}" "${base_log}" "${base_ckpt}"; do
    [[ -f "${required}" ]] || { echo "missing input: ${required}" >&2; return 1; }
  done

  if [[ ! -f "${train_log}" ]] || ! grep -q "best epoch" "${train_log}" 2>/dev/null; then
    log "train phase=${phase} candidate=${candidate} seed=${seed} gpu=${gpu}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "dry train ${candidate} seed=${seed}" > "${train_log}"
      echo "best epoch 0" >> "${train_log}"
    else
      read -r -a gate_args <<< "${args}"
      CUDA_VISIBLE_DEVICES="${gpu}" .venv/bin/python main.py \
        --config configs/clothing/stage2_decoder_mm.yaml \
        --suffix "${suffix}" \
        --dataset clothing \
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
        --epoch "${epochs}" \
        --early_stop "${early_stop}" \
        --eva_interval "${eva_interval}" \
        --batch_size 2048 \
        --lr 0.01 \
        --lr_rec 0.01 \
        --lr_imp 0.0002 \
        --lr_decoder 0.00005 \
        --reg_coeff 0.01 \
        --penalty_coeff 1.0 \
        --max_info_coeff 0.01 \
        --min_info_coeff 0.000001 \
        --evaluation_protocol strict \
        --selection_mode val \
        --strict_probe_test_interval "${strict_probe_interval}" \
        --save 1 \
        "${gate_args[@]}" \
        2>&1 | tee "${train_log}"
    fi
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '0.4000 hr@20 0.00000 0.00000\n1.0000 hr@20 0.00000 0.00000\n' > "${eval_log}"
    return
  fi

  fusion_ckpt="$(latest_ckpt_for_suffix "${suffix}")"
  [[ -f "${fusion_ckpt}" ]] || { echo "missing fusion checkpoint for ${suffix}" >&2; return 1; }

  if [[ -f "${eval_log}" ]] && grep -q '^1\.0000' "${eval_log}" 2>/dev/null; then
    log "skip eval phase=${phase} candidate=${candidate} seed=${seed}"
    return
  fi

  log "eval phase=${phase} candidate=${candidate} seed=${seed} gpu=${gpu}"
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

write_tasks() {
  local phase="$1"
  local tasks="${OUT_DIR}/tasks/${phase}.tsv"
  : > "${tasks}"
  if [[ "${phase}" == "search" ]]; then
    while IFS=$'\t' read -r candidate args; do
      for seed in "${SEARCH_SEEDS[@]}"; do
        printf '%s\t%s\t%s\t%s\n' "${phase}" "${candidate}" "${seed}" "${args}" >> "${tasks}"
      done
    done < <(candidate_table)
else
    local selected="${OUT_DIR}/selected_candidate.txt"
    local full_candidates=()
    if [[ -n "${FULL_CANDIDATES_STR}" ]]; then
      read -r -a full_candidates <<< "${FULL_CANDIDATES_STR}"
    else
      [[ -f "${selected}" ]] || { echo "missing selected candidate: ${selected}" >&2; return 1; }
      full_candidates=("$(< "${selected}")")
    fi

    local candidate args
    for candidate in "${full_candidates[@]}"; do
      args="$(candidate_table | awk -F '\t' -v n="${candidate}" '$1 == n {print $2}')"
      [[ -n "${args}" ]] || { echo "failed to resolve selected args for ${candidate}" >&2; return 1; }
      for seed in "${FULL_SEEDS[@]}"; do
        printf '%s\t%s\t%s\t%s\n' "${phase}" "${candidate}" "${seed}" "${args}" >> "${tasks}"
      done
    done
  fi
}

worker() {
  local gpu="$1"
  local file="$2"
  while IFS=$'\t' read -r phase candidate seed args; do
    [[ -n "${phase}" ]] || continue
    run_task "${gpu}" "${phase}" "${candidate}" "${seed}" "${args}"
  done < "${file}"
}

run_phase() {
  local phase="$1"
  local tasks="${OUT_DIR}/tasks/${phase}.tsv"
  write_tasks "${phase}"
  for idx in "${!GPUS[@]}"; do
    : > "${OUT_DIR}/tasks/${phase}_gpu${idx}.tsv"
  done

  local line_no=0
  while IFS= read -r line || [[ -n "${line}" ]]; do
    local slot=$((line_no % ${#GPUS[@]}))
    printf '%s\n' "${line}" >> "${OUT_DIR}/tasks/${phase}_gpu${slot}.tsv"
    line_no=$((line_no + 1))
  done < "${tasks}"

  log "${phase} queued ${line_no} tasks on GPUs: ${GPUS_STR}"
  local pids=()
  for idx in "${!GPUS[@]}"; do
    worker "${GPUS[$idx]}" "${OUT_DIR}/tasks/${phase}_gpu${idx}.tsv" \
      > "${OUT_DIR}/tasks/${phase}_gpu${idx}.worker.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
}

summarize_phase() {
  local phase="$1"
  .venv/bin/python - "${OUT_DIR}" "${phase}" "${PRIMARY_ALPHA}" <<'PY'
from pathlib import Path
import statistics
import sys

out_dir = Path(sys.argv[1])
phase = sys.argv[2]
primary_alpha = float(sys.argv[3])
target_r, target_n = 0.07638, 0.03425

def parse_eval(path):
    rows = {}
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) == 4:
            try:
                rows[float(parts[0])] = (float(parts[2]), float(parts[3]))
            except ValueError:
                pass
    return rows

rows = []
for log_path in sorted((out_dir / "eval" / phase).glob("*/seed*_test.log")):
    candidate = log_path.parent.name
    seed = log_path.stem.replace("seed", "").replace("_test", "")
    parsed = parse_eval(log_path)
    if primary_alpha not in parsed or not parsed:
        rows.append([candidate, seed, "", "", "", "", "", "", "", "missing_eval"])
        continue
    pr, pn = parsed[primary_alpha]
    ar, an = parsed.get(1.0, ("", ""))
    best_alpha, (best_r, best_n) = max(parsed.items(), key=lambda item: (item[1][0], item[1][1]))
    rows.append([candidate, seed, pr, pn, ar, an, best_alpha, best_r, best_n, "ok"])

summary = out_dir / f"{phase}_summary.tsv"
with summary.open("w") as f:
    f.write("candidate\tseed\tprimary_R20\tprimary_N20\taux_R20\taux_N20\tbest_alpha\tbest_R20\tbest_N20\tstatus\n")
    for row in rows:
        f.write("\t".join(map(str, row)) + "\n")

agg = {}
for row in rows:
    if row[-1] == "ok":
        agg.setdefault(row[0], []).append(row)

ranked = []
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

agg_path = out_dir / f"{phase}_agg.tsv"
with agg_path.open("w") as f:
    f.write("rank\tcandidate\tok_count\tmean_primary_R20\tmean_primary_N20\tmean_best_R20\tmean_best_N20\ttarget_R20\ttarget_N20\tprimary_R_gap\tprimary_N_gap\n")
    for rank, (mean_pr, mean_pn, mean_br, mean_bn, candidate, count) in enumerate(ranked, 1):
        f.write(
            f"{rank}\t{candidate}\t{count}\t{mean_pr:.5f}\t{mean_pn:.5f}\t"
            f"{mean_br:.5f}\t{mean_bn:.5f}\t{target_r:.5f}\t{target_n:.5f}\t"
            f"{mean_pr - target_r:.5f}\t{mean_pn - target_n:.5f}\n"
        )

if phase == "search" and ranked:
    (out_dir / "selected_candidate.txt").write_text(ranked[0][4] + "\n")
PY
}

log "run_tag=${RUN_TAG} out=${OUT_DIR}"
if [[ "${SKIP_SEARCH}" != "1" ]]; then
  run_phase search
  summarize_phase search
else
  log "skip search phase"
fi
run_phase full
summarize_phase full
log "done run_tag=${RUN_TAG}"
