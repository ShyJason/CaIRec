#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SEEDS_STR="${SEEDS:-1 12 123 1234 12345}"
DATASETS_STR="${DATASETS:-baby}"
EXP_MODES_STR="${EXP_MODES:-mm}"
TRAIN_MISSING_RATES_STR="${TRAIN_MISSING_RATES:-${MISSING_RATES:-0.1 0.3 0.5}}"
TEST_MISSING_RATE="0.5"
METHODS_STR="${METHODS:-mmrec i3 i3_noirm_noib}"

DEVICE_ID="${DEVICE_ID:-0}"
USE_GPU="${USE_GPU:-1}"
SAVE="${SAVE:-1}"
TENSORBOARD="${TENSORBOARD:-0}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
FAIL_FAST="${FAIL_FAST:-0}"

MMREC_FEATURE_BRIDGE_MODE="${MMREC_FEATURE_BRIDGE_MODE:-raw_decoder}"

I3_EPOCHS="${I3_EPOCHS:-200}"
I3_EVA_INTERVAL="${I3_EVA_INTERVAL:-10}"
I3_EARLY_STOP="${I3_EARLY_STOP:-20}"
I3_BATCH_SIZE="${I3_BATCH_SIZE:-2048}"
I3_SAVE="${I3_SAVE:-${SAVE}}"
I3_TENSORBOARD="${I3_TENSORBOARD:-0}"

read -r -a SEEDS <<< "${SEEDS_STR}"
read -r -a DATASETS <<< "${DATASETS_STR}"
read -r -a EXP_MODES <<< "${EXP_MODES_STR}"
read -r -a TRAIN_MISSING_RATES <<< "${TRAIN_MISSING_RATES_STR}"
read -r -a METHODS <<< "${METHODS_STR}"

run_cmd() {
  echo "[grid] $*"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

grep_existing_logs() {
  local pattern="$1"
  local needle="$2"
  compgen -G "${pattern}" >/dev/null || return 1
  grep -q "${needle}" ${pattern}
}

mmrec_done() {
  local stage2_dir="$1"
  [[ "${OVERWRITE}" != "1" ]] || return 1
  [[ -d "${stage2_dir}" ]] || return 1
  grep_existing_logs "${stage2_dir}/log/*.log" 'best epoch'
}

i3_done() {
  local out_dir="$1"
  local launch_log="$2"
  [[ "${OVERWRITE}" != "1" ]] || return 1
  [[ -d "${out_dir}" ]] || return 1
  [[ -f "${launch_log}" ]] && grep -q 'best epoch' "${launch_log}"
}

ACTIVE_JOBS=0
FAILED_JOBS=0

wait_for_slot() {
  if [[ "${MAX_PARALLEL}" -le 1 ]]; then
    return
  fi

  while [[ "${ACTIVE_JOBS}" -ge "${MAX_PARALLEL}" ]]; do
    if ! wait -n; then
      FAILED_JOBS=1
      if [[ "${FAIL_FAST}" == "1" ]]; then
        echo "[grid] a parallel task failed; stopping because FAIL_FAST=1" >&2
        exit 1
      fi
    fi
    ACTIVE_JOBS=$((ACTIVE_JOBS - 1))
  done
}

launch_task() {
  local label="$1"
  local log_file="$2"
  shift 2

  if [[ "${MAX_PARALLEL}" -le 1 || "${DRY_RUN}" == "1" ]]; then
    "$@"
    return
  fi

  wait_for_slot
  mkdir -p "$(dirname "${log_file}")"
  echo "[grid] launch ${label} -> ${log_file}"
  (
    echo "[grid-task] start ${label}"
    "$@"
    echo "[grid-task] done ${label}"
  ) > "${log_file}" 2>&1 &
  ACTIVE_JOBS=$((ACTIVE_JOBS + 1))
}

wait_for_all() {
  if [[ "${MAX_PARALLEL}" -le 1 ]]; then
    return
  fi

  while [[ "${ACTIVE_JOBS}" -gt 0 ]]; do
    if ! wait -n; then
      FAILED_JOBS=1
    fi
    ACTIVE_JOBS=$((ACTIVE_JOBS - 1))
  done

  if [[ "${FAILED_JOBS}" != "0" ]]; then
    echo "[grid] one or more parallel tasks failed" >&2
    exit 1
  fi
}

i3_params_for_dataset() {
  local dataset="$1"
  local exp_mode="$2"

  I3_LR="${I3_LR_OVERRIDE:-}"
  I3_REG_COEFF="${I3_REG_COEFF_OVERRIDE:-}"
  I3_PENALTY_COEFF="${I3_PENALTY_COEFF_OVERRIDE:-}"
  I3_MAX_INFO_COEFF="${I3_MAX_INFO_COEFF_OVERRIDE:-}"
  I3_MIN_INFO_COEFF="${I3_MIN_INFO_COEFF_OVERRIDE:-}"

  if [[ -n "${I3_LR}" && -n "${I3_REG_COEFF}" && -n "${I3_PENALTY_COEFF}" && -n "${I3_MAX_INFO_COEFF}" && -n "${I3_MIN_INFO_COEFF}" ]]; then
    return
  fi

  case "${dataset}" in
    baby)
      I3_LR="${I3_LR:-1e-3}"
      I3_REG_COEFF="${I3_REG_COEFF:-1e-3}"
      I3_PENALTY_COEFF="${I3_PENALTY_COEFF:-300}"
      I3_MAX_INFO_COEFF="${I3_MAX_INFO_COEFF:-1e-3}"
      I3_MIN_INFO_COEFF="${I3_MIN_INFO_COEFF:-1e-5}"
      ;;
    clothing)
      I3_LR="${I3_LR:-1e-2}"
      I3_REG_COEFF="${I3_REG_COEFF:-1e-2}"
      I3_PENALTY_COEFF="${I3_PENALTY_COEFF:-1}"
      I3_MAX_INFO_COEFF="${I3_MAX_INFO_COEFF:-1e-2}"
      if [[ "${exp_mode}" == "ff" ]]; then
        I3_MIN_INFO_COEFF="${I3_MIN_INFO_COEFF:-1e-5}"
      else
        I3_MIN_INFO_COEFF="${I3_MIN_INFO_COEFF:-1e-6}"
      fi
      ;;
    sports)
      # Sports is not listed in the I3 README, so use the I3 code defaults.
      I3_LR="${I3_LR:-1e-3}"
      I3_REG_COEFF="${I3_REG_COEFF:-1e-4}"
      I3_PENALTY_COEFF="${I3_PENALTY_COEFF:-50}"
      I3_MAX_INFO_COEFF="${I3_MAX_INFO_COEFF:-0.05}"
      I3_MIN_INFO_COEFF="${I3_MIN_INFO_COEFF:-0.05}"
      ;;
    tiktok)
      # TikTok is not listed in the I3 README, so use the I3 code defaults.
      I3_LR="${I3_LR:-1e-3}"
      I3_REG_COEFF="${I3_REG_COEFF:-1e-4}"
      I3_PENALTY_COEFF="${I3_PENALTY_COEFF:-50}"
      I3_MAX_INFO_COEFF="${I3_MAX_INFO_COEFF:-0.05}"
      I3_MIN_INFO_COEFF="${I3_MIN_INFO_COEFF:-0.05}"
      ;;
    *)
      echo "[grid] unsupported dataset for I3 defaults: ${dataset}" >&2
      exit 1
      ;;
  esac
}

run_mmrec() {
  local dataset="$1"
  local exp_mode="$2"
  local train_missing_rate="$3"
  local seed="$4"
  local suffix_tag="mmrec_${dataset}_${exp_mode}_mr${train_missing_rate}_seed${seed}_${RUN_TAG}"
  local stage2_suffix="stage2_${dataset}_recommender_decoder_${exp_mode}_${suffix_tag}"
  local stage2_dir="${ROOT_DIR}/exp_report/${dataset}/${stage2_suffix}"

  if mmrec_done "${stage2_dir}"; then
    echo "[grid] skip existing MMRec: ${stage2_dir}"
    return
  fi

  echo "[grid] MMRec dataset=${dataset} exp_mode=${exp_mode} train_missing_rate=${train_missing_rate} test_missing_rate=${TEST_MISSING_RATE} seed=${seed}"
  local env_args=(
    DEVICE_ID="${DEVICE_ID}" \
    USE_GPU="${USE_GPU}" \
    SAVE="${SAVE}" \
    TENSORBOARD="${TENSORBOARD}" \
    DATASET="${dataset}" \
    EXP_MODE="${exp_mode}" \
    MISSING_RATE="${train_missing_rate}" \
    SEED="${seed}" \
    RUN_TAG="${suffix_tag}" \
    FEATURE_BRIDGE_MODE="${MMREC_FEATURE_BRIDGE_MODE}"
  )

  for name in \
    STAGE11_EPOCHS STAGE11_LR STAGE11_ALPHA_REC \
    STAGE12_EPOCHS STAGE12_LR STAGE12_LR_IMP STAGE12_LR_DECODER \
    STAGE12_ALPHA_INTRA STAGE12_ALPHA_INTER STAGE12_ALPHA_ITM STAGE12_ALPHA_REC STAGE12_ALPHA_DECODE \
    STAGE2_EPOCHS STAGE2_EARLY_STOP STAGE2_BATCH_SIZE STAGE2_LR STAGE2_LR_REC STAGE2_LR_IMP STAGE2_LR_DECODER \
    BATCH_SIZE; do
    local override_name="MMREC_${name}"
    if [[ -n "${!override_name:-}" ]]; then
      env_args+=("${name}=${!override_name}")
    fi
  done

  run_cmd env "${env_args[@]}" ./run_baby_three_stage_report.sh
}

run_i3_variant() {
  local method="$1"
  local dataset="$2"
  local exp_mode="$3"
  local train_missing_rate="$4"
  local seed="$5"
  local disable_irm=0
  local disable_ib=0

  if [[ "${method}" == "i3_noirm_noib" ]]; then
    disable_irm=1
    disable_ib=1
  fi

  i3_params_for_dataset "${dataset}" "${exp_mode}"

  local suffix="${method}_${dataset}_${exp_mode}_mr${train_missing_rate}_seed${seed}_${RUN_TAG}"
  local out_dir="${ROOT_DIR}/exp_report/${dataset}/${suffix}"
  local launch_dir="${ROOT_DIR}/exp_report/${dataset}/comparison_launch_logs"
  local launch_log="${launch_dir}/${suffix}.launch.log"

  if i3_done "${out_dir}" "${launch_log}"; then
    echo "[grid] skip existing ${method}: ${out_dir}"
    return
  fi

  mkdir -p "${launch_dir}"
  echo "[grid] ${method} dataset=${dataset} exp_mode=${exp_mode} train_missing_rate=${train_missing_rate} test_missing_rate=${TEST_MISSING_RATE} seed=${seed} disable_irm=${disable_irm} disable_ib=${disable_ib}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo ./run_i3clear.sh \
      --dataset "${dataset}" --exp_mode "${exp_mode}" --missing_rate "${train_missing_rate}" \
      --seed "${seed}" --device_id "${DEVICE_ID}" --use_gpu "${USE_GPU}" \
      --epoch "${I3_EPOCHS}" --eva_interval "${I3_EVA_INTERVAL}" --early_stop "${I3_EARLY_STOP}" \
      --batch_size "${I3_BATCH_SIZE}" --lr "${I3_LR}" \
      --reg_coeff "${I3_REG_COEFF}" --penalty_coeff "${I3_PENALTY_COEFF}" \
      --max_info_coeff "${I3_MAX_INFO_COEFF}" --min_info_coeff "${I3_MIN_INFO_COEFF}" \
      --disable_irm "${disable_irm}" --disable_ib "${disable_ib}" \
      --tensorboard "${I3_TENSORBOARD}" --save "${I3_SAVE}" --suffix "${suffix}"
    return
  fi

  ./run_i3clear.sh \
    --dataset "${dataset}" \
    --exp_mode "${exp_mode}" \
    --missing_rate "${train_missing_rate}" \
    --seed "${seed}" \
    --device_id "${DEVICE_ID}" \
    --use_gpu "${USE_GPU}" \
    --epoch "${I3_EPOCHS}" \
    --eva_interval "${I3_EVA_INTERVAL}" \
    --early_stop "${I3_EARLY_STOP}" \
    --batch_size "${I3_BATCH_SIZE}" \
    --lr "${I3_LR}" \
    --reg_coeff "${I3_REG_COEFF}" \
    --penalty_coeff "${I3_PENALTY_COEFF}" \
    --max_info_coeff "${I3_MAX_INFO_COEFF}" \
    --min_info_coeff "${I3_MIN_INFO_COEFF}" \
    --disable_irm "${disable_irm}" \
    --disable_ib "${disable_ib}" \
    --tensorboard "${I3_TENSORBOARD}" \
    --save "${I3_SAVE}" \
    --suffix "${suffix}" \
    2>&1 | tee "${launch_log}"
}

echo "[grid] run_tag=${RUN_TAG}"
echo "[grid] methods=${METHODS[*]}"
echo "[grid] datasets=${DATASETS[*]}"
echo "[grid] exp_modes=${EXP_MODES[*]}"
echo "[grid] train_missing_rates=${TRAIN_MISSING_RATES[*]}"
echo "[grid] test_missing_rate=${TEST_MISSING_RATE}"
echo "[grid] seeds=${SEEDS[*]}"
echo "[grid] max_parallel=${MAX_PARALLEL}"

for dataset in "${DATASETS[@]}"; do
  for exp_mode in "${EXP_MODES[@]}"; do
    for train_missing_rate in "${TRAIN_MISSING_RATES[@]}"; do
      for seed in "${SEEDS[@]}"; do
        for method in "${METHODS[@]}"; do
          task_label="${method} dataset=${dataset} exp_mode=${exp_mode} train_missing_rate=${train_missing_rate} seed=${seed}"
          task_log="${ROOT_DIR}/exp_report/${dataset}/comparison_launch_logs/${method}_${dataset}_${exp_mode}_mr${train_missing_rate}_seed${seed}_${RUN_TAG}.grid_task.log"
          case "${method}" in
            mmrec)
              launch_task "${task_label}" "${task_log}" run_mmrec "${dataset}" "${exp_mode}" "${train_missing_rate}" "${seed}"
              ;;
            i3|i3_noirm_noib)
              launch_task "${task_label}" "${task_log}" run_i3_variant "${method}" "${dataset}" "${exp_mode}" "${train_missing_rate}" "${seed}"
              ;;
            *)
              echo "[grid] unknown method: ${method}" >&2
              exit 1
              ;;
          esac
        done
      done
    done
  done
done

wait_for_all
