#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PATH="${ROOT_DIR}/.venv/bin:${PATH}"
export PYTHONUNBUFFERED=1
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

RUN_TAG="${RUN_TAG:?RUN_TAG is required}"
DEVICE_ID="${DEVICE_ID:-3}"
DATASET_SEED="${DATASET_SEED:-0}"
CANDIDATE="rr_a018_m035_reg01"
SEEDS_STR="${SEEDS:-1 12}"
read -r -a SEEDS_ARR <<< "${SEEDS_STR}"

ARGS=(
  --fusion_mode mean
  --completion_gate_mode rank_residual
  --completion_gate_hidden_dim 64
  --completion_gate_dropout 0.1
  --completion_gate_init_logit 0.0
  --completion_gate_detach_inputs 1
  --completion_gate_use_item_context 1
  --completion_gate_item_context_source shared_mean
  --completion_gate_residual_alpha 0.18
  --completion_gate_mix_alpha 0.35
  --completion_gate_identity_coeff 0.05
  --completion_gate_balance_coeff 0.01
  --completion_gate_reg_coeff 0.1
  --recommender_allow_modal_grad 0
)

log() {
  date +"[companion-reg01] %Y-%m-%d %H:%M:%S $*"
}

for seed in "${SEEDS_ARR[@]}"; do
  out_dir="exp_report/sports/fusion_search/${RUN_TAG}/search/${CANDIDATE}"
  mkdir -p "${out_dir}"
  log_path="${out_dir}/seed${seed}.log"
  if [[ -f "${log_path}" ]] && grep -q "best epoch" "${log_path}"; then
    log "skip completed candidate=${CANDIDATE} seed=${seed}"
    continue
  fi

  ckpt="exp_report/sports/stage1_2_sports_imputer_backprop_decoder_v2_mmrec_sports_mm_mr0.3_seed${seed}_mmrec_sports_mm_fixedmissing_20260524_165817/ckpt/stage1_2_sports_imputer_backprop_decoder_v2_mmrec_sports_mm_mr0.3_seed${seed}_mmrec_sports_mm_fixedmissing_20260524_165817_imputer_backprop_50_epoch19.pth"
  suffix="stage2_sports_fusion_search_${CANDIDATE}_dseed${DATASET_SEED}_seed${seed}_${RUN_TAG}"
  log "run candidate=${CANDIDATE} seed=${seed} gpu=${DEVICE_ID}"

  (
    CONFIG="configs/sports/stage2_decoder_mm.yaml" \
    DATASET=sports \
    EXP_MODE=mm \
    DATASET_SEED="${DATASET_SEED}" \
    SEED="${seed}" \
    DEVICE_ID="${DEVICE_ID}" \
    USE_GPU=1 \
    TENSORBOARD=0 \
    SAVE=1 \
    IMPUTER_CKPT="${ckpt}" \
    SUFFIX="${suffix}" \
    EPOCHS=80 \
    EVA_INTERVAL=1 \
    EARLY_STOP=15 \
    BATCH_SIZE=256 \
    LR=0.001 \
    LR_REC=0.001 \
    LR_IMP=0.0002 \
    LR_DECODER=0.00005 \
    STRICT_PROBE_TEST_INTERVAL=10 \
    ./run_stage2_baby_recommender_decoder.sh "${ARGS[@]}"
  ) 2>&1 | tee "${log_path}"
done

log "done"
