#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then PYTHON_BIN="python3"; fi

PHYSICAL_GPU="${PHYSICAL_GPU:-${1:-0}}"
CHECK_ONLY="${CHECK_ONLY:-0}"
RUN_TAG="${RUN_TAG:-repro_best_sports_20260719_$(date -u +%Y%m%d_%H%M%S)}"
CONFIG="configs/sports/paper_stage2.yaml"
IMPUTER_CKPT="exp_report/sports/sports_unified_mr0p5_stage12_fixed_repro_20260718_stage1_2_completion/ckpt/sports_unified_mr0p5_stage12_fixed_repro_20260718_stage1_2_completion_imputer_backprop_50_epoch49.pth"
PAYLOAD="Data/sports/unified_missing_items_mr0.5_seed2023.npy"
OUT_DIR="exp_report/sports/reproduce_best_20260719/${RUN_TAG}"
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

verify_sha256 b782119d3ccba7e96f6898b85742716cc49ca616214b24dcea4a6fe21c2a4197 "${CONFIG}"
verify_sha256 95e388b214f75fd159de7b28c9902a60604eca2c804370324c8738d9d7db6086 "${IMPUTER_CKPT}"
verify_sha256 421816fbeaa65cb6323f9f42e209a52f5688401525ba75bb6c902789580aaabe "${PAYLOAD}"
test -f Data/sports/sports.inter
test -f Data/sports/image_feat.npy
test -f Data/sports/text_feat.npy
"${PYTHON_BIN}" main.py --config "${CONFIG}" --check_config

echo "dataset=sports protocol=unified_static train_mr=0.5 eval_mr=0.5 seed=2023 payload_seed=2023"
echo "reliability=both scale=50 fusion=posterior_reliability"
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
