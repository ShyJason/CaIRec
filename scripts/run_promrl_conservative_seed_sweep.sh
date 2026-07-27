#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEVICE_ID="${DEVICE_ID:-4}"
USE_GPU="${USE_GPU:-1}"
TENSORBOARD="${TENSORBOARD:-0}"
HF_TENSORBOARD_REPO="${HF_TENSORBOARD_REPO:-}"
HF_TOKEN="${HF_TOKEN:-}"

INIT_EPOCHS="${INIT_EPOCHS:-5}"
PROMRL_EPOCHS="${PROMRL_EPOCHS:-10}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-100}"

SEEDS=("$@")
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  SEEDS=(1 12 123 1234 12345)
fi

find_only_ckpt() {
  local run_dir="$1"
  ls -t "${run_dir}/ckpt/"*.pth 2>/dev/null | head -n 1
}

run_seed() {
  local seed="$1"
  local init_suffix="stage1_init_baby_promrl_seed${seed}_latest"
  local promrl_suffix="stage1_promrl_main_baby_conservative_seed${seed}_latest"
  local stage2_suffix="stage2_baby_seed${seed}_mm_promrl_conservative_e100_full_latest"

  echo "[promrl-seed-sweep] seed=${seed} stage1_init"
  python main.py \
    --config configs/baby/stage1_init_promrl.yaml \
    --seed "${seed}" \
    --epoch "${INIT_EPOCHS}" \
    --device_id "${DEVICE_ID}" \
    --use_gpu "${USE_GPU}" \
    --tensorboard "${TENSORBOARD}" \
    --hf_tensorboard_repo "${HF_TENSORBOARD_REPO}" \
    --hf_token "${HF_TOKEN}" \
    --suffix "${init_suffix}"

  local init_ckpt
  init_ckpt="$(find_only_ckpt "${ROOT_DIR}/exp_report/baby/${init_suffix}")"
  if [[ -z "${init_ckpt}" ]]; then
    echo "[promrl-seed-sweep] missing init ckpt for seed=${seed}" >&2
    exit 1
  fi

  echo "[promrl-seed-sweep] seed=${seed} promrl_main"
  python main.py \
    --config configs/baby/stage1_promrl_main_conservative.yaml \
    --seed "${seed}" \
    --epoch "${PROMRL_EPOCHS}" \
    --device_id "${DEVICE_ID}" \
    --use_gpu "${USE_GPU}" \
    --tensorboard "${TENSORBOARD}" \
    --hf_tensorboard_repo "${HF_TENSORBOARD_REPO}" \
    --hf_token "${HF_TOKEN}" \
    --imputer_ckpt "${init_ckpt}" \
    --suffix "${promrl_suffix}"

  local promrl_ckpt
  promrl_ckpt="$(find_only_ckpt "${ROOT_DIR}/exp_report/baby/${promrl_suffix}")"
  if [[ -z "${promrl_ckpt}" ]]; then
    echo "[promrl-seed-sweep] missing promrl ckpt for seed=${seed}" >&2
    exit 1
  fi

  echo "[promrl-seed-sweep] seed=${seed} stage2_e100"
  python main.py \
    --config configs/baby/stage2_decoder_mm.yaml \
    --seed "${seed}" \
    --epoch "${STAGE2_EPOCHS}" \
    --device_id "${DEVICE_ID}" \
    --use_gpu "${USE_GPU}" \
    --tensorboard "${TENSORBOARD}" \
    --hf_tensorboard_repo "${HF_TENSORBOARD_REPO}" \
    --hf_token "${HF_TOKEN}" \
    --imputer_ckpt "${promrl_ckpt}" \
    --suffix "${stage2_suffix}"

  echo "[promrl-seed-sweep] completed seed=${seed}"
}

for seed in "${SEEDS[@]}"; do
  run_seed "${seed}"
done
