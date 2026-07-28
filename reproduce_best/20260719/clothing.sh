#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then PYTHON_BIN="python3"; fi

PHYSICAL_GPU="${PHYSICAL_GPU:-${1:-0}}"
CHECK_ONLY="${CHECK_ONLY:-0}"
RUN_TAG="${RUN_TAG:-repro_best_clothing_20260719_$(date -u +%Y%m%d_%H%M%S)}"
CONFIG="configs/clothing/paper_stage2.yaml"
IMPUTER_CKPT="exp_report/clothing/three_stage_clothing_mm_decoupled_latent_clothing_unified_mr0p5_fixed_stage12_seed2023_20260718_stage1_2_completion/ckpt/three_stage_clothing_mm_decoupled_latent_clothing_unified_mr0p5_fixed_stage12_seed2023_20260718_stage1_2_completion_imputer_backprop_50_epoch49.pth"
PAYLOAD="Data/clothing/unified_missing_items_mr0.5_seed2023.npy"
OUT_DIR="exp_report/clothing/reproduce_best_20260719/${RUN_TAG}"
LOG_FILE="${OUT_DIR}/${RUN_TAG}.launch.log"

verify_sha256() {
  local expected="$1"
  local path="$2"
  test -f "${path}" || { echo "missing required file: ${path}" >&2; exit 1; }
  echo "${expected}  ${path}" | sha256sum --check --status || {
    echo "sha256 mismatch: ${path}" >&2
    exit 1
  }
}

verify_sha256 70dc101c7b403470dd95e199a75d241306c11d7cb7eaece39ee979281acaa726 "${CONFIG}"
verify_sha256 dfe03cc6dbab74a2f4d651fff603f8be57696ed18c6a861b51cd9c7e0975cd46 "${IMPUTER_CKPT}"
verify_sha256 34e09412a337e19906b16bb7bdb9e097d824e1e85a1b1908e501e5a29bc1873c "${PAYLOAD}"
test -f Data/clothing/clothing.inter
test -f Data/clothing/image_feat.npy
test -f Data/clothing/text_feat.npy
"${PYTHON_BIN}" main.py --config "${CONFIG}" --check_config

echo "dataset=clothing protocol=unified_static train_mr=0.5 eval_mr=0.5 seed=2023 payload_seed=2023"
echo "reliability=off fusion=mean"
echo "checkpoint=${IMPUTER_CKPT}"
if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "preflight passed; no training started"
  exit 0
fi

mkdir -p "${OUT_DIR}"
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONUNBUFFERED=1
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"

"${PYTHON_BIN}" main.py \
  --config "${CONFIG}" \
  --imputer_ckpt "${IMPUTER_CKPT}" \
  --suffix "${RUN_TAG}" \
  2>&1 | tee "${LOG_FILE}"
