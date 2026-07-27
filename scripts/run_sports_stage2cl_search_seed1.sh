#!/usr/bin/env bash
set -euo pipefail

cd /home/ruiyuliu/projects/MMRec

GPU="${GPU:-0}"
PY="/home/ruiyuliu/projects/MMRec/.venv/bin/python"
CKPT="/home/ruiyuliu/projects/MMRec/exp_report/sports/stage1_2_sports_imputer_backprop_decoder_v2_mmrec_sports_mm_mr0.3_seed1_mmrec_sports_mm_fixedmissing_20260524_165817/ckpt/stage1_2_sports_imputer_backprop_decoder_v2_mmrec_sports_mm_mr0.3_seed1_mmrec_sports_mm_fixedmissing_20260524_165817_imputer_backprop_50_epoch19.pth"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/home/ruiyuliu/projects/MMRec/rerun_logs"
mkdir -p "${LOG_DIR}"

run_one() {
  local weight="$1"
  local temp="$2"
  local tag="$3"
  local suffix="stage2_sports_gcncl_${tag}_seed1_dseed0_${STAMP}"
  local log="${LOG_DIR}/${suffix}.log"

  echo "### START ${suffix} $(date)"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" main.py \
    --config configs/sports/stage2_decoder_mm.yaml \
    --device_id 0 \
    --seed 1 \
    --dataset_seed 0 \
    --missing_mask_protocol i3 \
    --imputer_ckpt "${CKPT}" \
    --evaluation_protocol strict \
    --topk '[20]' \
    --epoch 200 \
    --eva_interval 1 \
    --early_stop 20 \
    --rec_neighbor_cl_weight "${weight}" \
    --rec_neighbor_cl_temp "${temp}" \
    --rec_neighbor_cl_bank_size 256 \
    --suffix "${suffix}" \
    --log 1 2>&1 | tee "${log}"
  echo "### END ${suffix} $(date)"
}

run_one 0.0 0.25 "w000_t025"
run_one 0.005 0.25 "w0005_t025"
run_one 0.02 0.25 "w002_t025"
run_one 0.01 0.15 "w001_t015"
