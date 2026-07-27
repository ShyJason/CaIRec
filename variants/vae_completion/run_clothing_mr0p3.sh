#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
DEVICE_ID="${DEVICE_ID:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-exp_report/clothing/vae_completion_mr0p3_seed2023}"

exec "${PYTHON_BIN}" variants/vae_completion/train_vae_imputer.py \
  --dataset clothing \
  --data_dir Data/clothing \
  --output_dir "${OUTPUT_DIR}" \
  --device "cuda:${DEVICE_ID}" \
  --seed 2023 \
  --train_missing_rate 0.3 \
  --eval_missing_rate 0.5 \
  --missing_seed 2023 \
  --epochs 100 \
  --batch_size 512 \
  --lr 0.0005 \
  --weight_decay 0.000001 \
  --latent_dim 64 \
  --hidden_dim 512 \
  --modal_hidden_dim 256 \
  --dropout 0.1 \
  --beta_kl 0.001 \
  --kl_warmup_epochs 20 \
  --normalize_features 1 \
  --normalize_outputs 1 \
  --input_dropout 0.0 \
  --eval_interval 1 \
  --early_stop 20 \
  "$@"
