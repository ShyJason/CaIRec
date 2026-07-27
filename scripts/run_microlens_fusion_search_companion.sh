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
SEEDS_STR="${SEEDS:-1 12}"
WAIT_SESSION="${WAIT_SESSION:-}"

read -r -a SEEDS_ARR <<< "${SEEDS_STR}"

log() {
  date +"[microlens-companion] %Y-%m-%d %H:%M:%S $*"
}

if [[ -n "${WAIT_SESSION}" ]]; then
  log "waiting for tmux session ${WAIT_SESSION}"
  while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
    log "still waiting for ${WAIT_SESSION}"
    sleep 300
  done
  log "wait session finished"
fi

candidate_table() {
  cat <<'EOF'
rr_a018_m035_reg1	--fusion_mode mean --completion_gate_mode rank_residual --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_residual_alpha 0.18 --completion_gate_mix_alpha 0.35 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 1.0 --recommender_allow_modal_grad 0
rr_a010_m035_reg1	--fusion_mode mean --completion_gate_mode rank_residual --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_residual_alpha 0.10 --completion_gate_mix_alpha 0.35 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 1.0 --recommender_allow_modal_grad 0
rr_a025_m035_reg1	--fusion_mode mean --completion_gate_mode rank_residual --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_residual_alpha 0.25 --completion_gate_mix_alpha 0.35 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 1.0 --recommender_allow_modal_grad 0
rr_a018_m020_reg1	--fusion_mode mean --completion_gate_mode rank_residual --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_residual_alpha 0.18 --completion_gate_mix_alpha 0.20 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 1.0 --recommender_allow_modal_grad 0
rr_a018_m050_reg1	--fusion_mode mean --completion_gate_mode rank_residual --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_residual_alpha 0.18 --completion_gate_mix_alpha 0.50 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 1.0 --recommender_allow_modal_grad 0
rr_a018_m035_reg01	--fusion_mode mean --completion_gate_mode rank_residual --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_residual_alpha 0.18 --completion_gate_mix_alpha 0.35 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 0.1 --recommender_allow_modal_grad 0
EOF
}

find_best_ckpt() {
  local report_dir="$1"
  local stage_dir="$2"
  python - "$report_dir" "$stage_dir" <<'PY'
import pathlib
import re
import sys

report_dir = pathlib.Path(sys.argv[1])
stage_dir = pathlib.Path(sys.argv[2])
log_path = report_dir / "stage1_2.log"
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

seed_tag() {
  local seed="$1"
  if [[ "${seed}" == "1" || "${seed}" == "12" ]]; then
    echo "mmrec_microlens_mm_2seed_20260521_153006_seed${seed}"
  else
    echo "mmrec_microlens_tuned_5seed_20260522_023007_seed${seed}"
  fi
}

ckpt_for_seed() {
  local seed="$1"
  local tag
  tag="$(seed_tag "${seed}")"
  find_best_ckpt \
    "exp_report/microlens/pipeline_reports/${tag}_raw_decoder_mm" \
    "exp_report/microlens/stage1_2_microlens_imputer_backprop_decoder_v2_${tag}"
}

for seed in "${SEEDS_ARR[@]}"; do
  ckpt_for_seed "${seed}" >/dev/null
done

while IFS=$'\t' read -r candidate args; do
  for seed in "${SEEDS_ARR[@]}"; do
    out_dir="exp_report/microlens/fusion_search/${RUN_TAG}/search/${candidate}"
    mkdir -p "${out_dir}"
    log_path="${out_dir}/seed${seed}.log"
    if [[ -f "${log_path}" ]] && grep -q "best epoch" "${log_path}"; then
      log "skip completed candidate=${candidate} seed=${seed}"
      continue
    fi

    ckpt="$(ckpt_for_seed "${seed}")"
    suffix="stage2_microlens_fusion_search_${candidate}_dseed${DATASET_SEED}_seed${seed}_${RUN_TAG}"
    read -r -a extra_args <<< "${args}"
    log "run candidate=${candidate} seed=${seed} gpu=${DEVICE_ID}"
    (
      CONFIG="configs/microlens/stage2_decoder_mm.yaml" \
      DATASET=microlens \
      EXP_MODE=mm \
      DATASET_SEED="${DATASET_SEED}" \
      SEED="${seed}" \
      DEVICE_ID="${DEVICE_ID}" \
      USE_GPU=1 \
      TENSORBOARD=0 \
      SAVE=1 \
      IMPUTER_CKPT="${ckpt}" \
      SUFFIX="${suffix}" \
      EPOCHS=60 \
      EVA_INTERVAL=30 \
      EARLY_STOP=30 \
      BATCH_SIZE=2048 \
      LR=0.01 \
      LR_REC=0.01 \
      LR_IMP=0.0002 \
      LR_DECODER=0.00005 \
      STRICT_PROBE_TEST_INTERVAL=0 \
      ./run_stage2_baby_recommender_decoder.sh "${extra_args[@]}"
    ) 2>&1 | tee "${log_path}"
  done
done < <(candidate_table)

log "done"
