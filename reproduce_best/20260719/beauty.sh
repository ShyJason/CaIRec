#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then PYTHON_BIN="python3"; fi

PHYSICAL_GPU="${PHYSICAL_GPU:-${1:-0}}"
CHECK_ONLY="${CHECK_ONLY:-0}"
RUN_TAG="${RUN_TAG:-repro_best_beauty_20260719_$(date -u +%Y%m%d_%H%M%S)}"
CONFIG="configs/beauty/paper_stage2.yaml"
IMPUTER_CKPT="exp_report/beauty/beauty_unified_mr0p5_stage1_loss_search_20260718_fixed_lr00080_stage1_2_completion/ckpt/beauty_unified_mr0p5_stage1_loss_search_20260718_fixed_lr00080_stage1_2_completion_imputer_backprop_50_epoch49.pth"
PAYLOAD="Data/beauty/unified_missing_items_mr0.5_seed2023.npy"
OUT_DIR="exp_report/beauty/reproduce_best_20260719/${RUN_TAG}"
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

verify_sha256 d89b7799819c1637c85330bc94a0bb136c52d0ba29f8e7fecb61db5cf0593fd4 "${CONFIG}"
verify_sha256 00f2960cd4eeef41af65ad9769b9d9af118240f57e7c37cfd398056e243a0569 "${IMPUTER_CKPT}"
verify_sha256 408e3c8bfffd8322412e63b77cc87b07ae4ce1f02329f0500d61c2aee95e0cf4 "${PAYLOAD}"
test -f Data/beauty/beauty.inter
test -f Data/beauty/image_feat.npy
test -f Data/beauty/text_feat.npy
"${PYTHON_BIN}" main.py --config "${CONFIG}" --check_config

echo "dataset=beauty protocol=unified_static train_mr=0.5 eval_mr=0.5 seed=2023 payload_seed=2023"
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
