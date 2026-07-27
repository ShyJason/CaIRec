#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PATH="${ROOT_DIR}/.venv/bin:${PATH}"
export PYTHONUNBUFFERED=1
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

RUN_TAG="${RUN_TAG:?Set RUN_TAG to the active microlens100k fusion search tag}"
DEVICE_ID="${DEVICE_ID:-1}"
DATASET_SEED="${DATASET_SEED:-0}"
SEARCH_SEEDS_STR="${SEARCH_SEEDS:-1 12}"
CANDIDATES_STR="${CANDIDATES:-rr_a018_m035_reg01 rr_a018_m050_reg1}"
WAIT_SPORTS_RUN_TAG="${WAIT_SPORTS_RUN_TAG:-}"

read -r -a SEARCH_SEEDS <<< "${SEARCH_SEEDS_STR}"
read -r -a CANDIDATES <<< "${CANDIDATES_STR}"

log() {
  date +"[microlens100k-tail] %Y-%m-%d %H:%M:%S $*"
}

if [[ -n "${WAIT_SPORTS_RUN_TAG}" ]]; then
  log "waiting for sports full ${WAIT_SPORTS_RUN_TAG}"
  while [[ ! -f "exp_report/sports/fusion_search/${WAIT_SPORTS_RUN_TAG}/full_summary.tsv" ]] \
    && ! grep -q "full done dataset=sports" "exp_report/fusion_search/${WAIT_SPORTS_RUN_TAG}/queue.log" 2>/dev/null; do
    sleep 120
  done
  log "sports full complete; starting tail companion on gpu=${DEVICE_ID}"
fi

candidate_args_by_name() {
  case "$1" in
    rr_a018_m050_reg1)
      echo "--fusion_mode mean --completion_gate_mode rank_residual --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_residual_alpha 0.18 --completion_gate_mix_alpha 0.50 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 1.0 --recommender_allow_modal_grad 0"
      ;;
    rr_a018_m035_reg01)
      echo "--fusion_mode mean --completion_gate_mode rank_residual --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_residual_alpha 0.18 --completion_gate_mix_alpha 0.35 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 0.1 --recommender_allow_modal_grad 0"
      ;;
    *)
      echo "unsupported companion candidate: $1" >&2
      return 1
      ;;
  esac
}

seed_tag() {
  local seed="$1"
  if [[ "${seed}" == "1" ]]; then
    echo "mmrec_microlens100k_3modal_seed1_base_20260523_012808"
  else
    echo "mmrec_microlens100k_3modal_seed${seed}_mbpr25_20260523_033613"
  fi
}

find_best_ckpt_from_log() {
  local log_path="$1"
  local stage_dir="$2"
  python - "$log_path" "$stage_dir" <<'PY'
import pathlib
import re
import sys

log_path = pathlib.Path(sys.argv[1])
stage_dir = pathlib.Path(sys.argv[2])
if not log_path.exists():
    raise SystemExit(f"missing stage1_2 log: {log_path}")
text = log_path.read_text(errors="ignore")
matches = re.findall(r"best epoch\s+(\d+)", text)
if not matches:
    raise SystemExit(f"no best epoch in {log_path}")
epoch = int(matches[-1])
ckpts = sorted((stage_dir / "ckpt").glob(f"*_epoch{epoch}.pth"))
if not ckpts:
    raise SystemExit(f"no checkpoint for epoch {epoch} in {stage_dir / 'ckpt'}")
print(ckpts[-1])
PY
}

ckpt_for_seed() {
  local seed="$1"
  local tag
  tag="$(seed_tag "${seed}")"
  local stage_dir="exp_report/microlens100k/stage1_2_microlens100k_imputer_backprop_decoder_v2_${tag}"
  local pipeline_log="exp_report/microlens100k/pipeline_reports/${tag}_raw_decoder_mm/stage1_2.log"
  local seed_log="exp_report/microlens100k/mmrec_seed_logs/mmrec_microlens100k_3modal_mbpr25_5seed_20260523_033613/logs/seed${seed}_stage1_2.log"
  if [[ -f "${pipeline_log}" ]]; then
    find_best_ckpt_from_log "${pipeline_log}" "${stage_dir}"
  else
    find_best_ckpt_from_log "${seed_log}" "${stage_dir}"
  fi
}

run_one_search() {
  local candidate="$1"
  local seed="$2"
  local ckpt="$3"
  local args="$4"
  local out_dir="${ROOT_DIR}/exp_report/microlens100k/fusion_search/${RUN_TAG}/search/${candidate}"
  local suffix="stage2_microlens100k_fusion_search_${candidate}_dseed${DATASET_SEED}_seed${seed}_${RUN_TAG}"
  local log_path="${out_dir}/seed${seed}.log"
  mkdir -p "${out_dir}"

  if [[ -f "${log_path}" ]] && grep -q 'best epoch' "${log_path}"; then
    log "skip existing candidate=${candidate} seed=${seed}"
    return
  fi

  log "run candidate=${candidate} seed=${seed} gpu=${DEVICE_ID}"
  read -r -a extra_args <<< "${args}"
  (
    CONFIG="configs/microlens100k/stage2_decoder_mm.yaml" \
    DATASET="microlens100k" \
    EXP_MODE=mm \
    DATASET_SEED="${DATASET_SEED}" \
    SEED="${seed}" \
    DEVICE_ID="${DEVICE_ID}" \
    USE_GPU=1 \
    TENSORBOARD=0 \
    SAVE=1 \
    IMPUTER_CKPT="${ckpt}" \
    SUFFIX="${suffix}" \
    EPOCHS=200 \
    EVA_INTERVAL=10 \
    EARLY_STOP=200 \
    BATCH_SIZE=2048 \
    LR=0.01 \
    LR_REC=0.01 \
    LR_IMP=0.0002 \
    LR_DECODER=0.00005 \
    STRICT_PROBE_TEST_INTERVAL=0 \
    ./run_stage2_baby_recommender_decoder.sh "${extra_args[@]}"
  ) 2>&1 | tee "${log_path}"
}

for candidate in "${CANDIDATES[@]}"; do
  args="$(candidate_args_by_name "${candidate}")"
  for seed in "${SEARCH_SEEDS[@]}"; do
    ckpt="$(ckpt_for_seed "${seed}")"
    run_one_search "${candidate}" "${seed}" "${ckpt}" "${args}"
  done
done

log "tail companion done run_tag=${RUN_TAG}"
