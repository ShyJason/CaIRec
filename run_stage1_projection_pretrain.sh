#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

DEVICE_ID="${DEVICE_ID:-4}"
DATASET="${DATASET:-clothing}"
SUFFIX="${SUFFIX:-stage1_projection_pretrain_${DATASET}_raw_decoder}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi

exec "${PYTHON_BIN}" tools/pretrain_completion_projection.py \
  --dataset "${DATASET}" \
  --suffix "${SUFFIX}" \
  --epoch "${EPOCHS:-20}" \
  --batch_size "${BATCH_SIZE:-1024}" \
  --lr "${LR:-1e-3}" \
  --latent_dim "${LATENT_DIM:-64}" \
  --temperature "${TEMPERATURE:-0.07}" \
  --base_ce_weight "${BASE_CE_WEIGHT:-1.0}" \
  --completion_ce_weight "${COMPLETION_CE_WEIGHT:-1.0}" \
  --mse_weight "${MSE_WEIGHT:-0.05}" \
  --cosine_weight "${COSINE_WEIGHT:-0.05}" \
  --val_rate "${VAL_RATE:-0.1}" \
  --seed "${SEED:-2023}" \
  --use_gpu "${USE_GPU:-1}" \
  --device_id "${DEVICE_ID}" \
  --checkpoint_key_style "${CHECKPOINT_KEY_STYLE:-raw_decoder}" \
  "$@"
