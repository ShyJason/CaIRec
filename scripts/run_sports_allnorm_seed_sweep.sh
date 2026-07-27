#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PATH="${ROOT_DIR}/.venv/bin:${PATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

RUN_TAG="${RUN_TAG:-sports_allnorm_seed_sweep_$(date +%Y%m%d_%H%M%S)}"
DEVICE_ID="${DEVICE_ID:-4}"
DATASET_SEED="${DATASET_SEED:-0}"
SEEDS_STR="${SEEDS:-12 123 1234 12345}"

read -r -a SEEDS_ARR <<< "${SEEDS_STR}"

OUT_DIR="${ROOT_DIR}/exp_report/fusion_norm_ablation/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

SUMMARY="${OUT_DIR}/summary.tsv"
QUEUE_LOG="${OUT_DIR}/queue.log"
printf "seed\tbest_epoch\tr20\tndcg20\tlog\n" > "${SUMMARY}"

log() {
  date +"[allnorm] %Y-%m-%d %H:%M:%S $*" | tee -a "${QUEUE_LOG}"
}

parse_one() {
  local seed="$1"
  local log_path="$2"
  python - "${log_path}" "${seed}" <<'PY'
import pathlib
import re
import sys

log_path = pathlib.Path(sys.argv[1])
seed = sys.argv[2]
ansi = re.compile(r"\x1b\[[0-9;]*m")
text = ansi.sub("", log_path.read_text(errors="ignore"))
best = re.findall(r"best epoch\s+(\d+)", text)
final = re.findall(
    r"final strict test hr@20\s*=\s*([0-9.]+),\s*recall@20\s*=\s*[0-9.]+,\s*ndcg@20\s*=\s*([0-9.]+)",
    text,
)
if final:
    r20, ndcg20 = final[-1]
else:
    vals = re.findall(
        r"^epoch\s*=\s*(\d+)\s+hr@20\s*=\s*([0-9.]+),\s*recall@20\s*=\s*[0-9.]+,\s*ndcg@20\s*=\s*([0-9.]+)",
        text,
        re.M,
    )
    r20, ndcg20 = (vals[-1][1], vals[-1][2]) if vals else ("NA", "NA")
print(f"{seed}\t{best[-1] if best else 'NA'}\t{r20}\t{ndcg20}\t{log_path}")
PY
}

log "run_tag=${RUN_TAG} gpu=${DEVICE_ID} seeds=${SEEDS_STR}"
for seed in "${SEEDS_ARR[@]}"; do
  ckpt="exp_report/sports/stage1_2_sports_imputer_backprop_decoder_v2_mmrec_sports_mm_mr0.3_seed${seed}_mmrec_sports_mm_fixedmissing_20260524_165817/ckpt/stage1_2_sports_imputer_backprop_decoder_v2_mmrec_sports_mm_mr0.3_seed${seed}_mmrec_sports_mm_fixedmissing_20260524_165817_imputer_backprop_50_epoch19.pth"
  if [[ ! -f "${ckpt}" ]]; then
    log "missing ckpt seed=${seed}: ${ckpt}"
    exit 1
  fi
  log_path="${OUT_DIR}/sports_seed${seed}.log"
  suffix="stage2_sports_rankres_allnorm_seed${seed}_${RUN_TAG}"
  log "start seed=${seed}"

  CONFIG=configs/sports/stage2_decoder_mm.yaml \
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
  EPOCHS=200 \
  EVA_INTERVAL=1 \
  EARLY_STOP=20 \
  BATCH_SIZE=256 \
  LR=0.001 \
  LR_REC=0.001 \
  LR_IMP=0.0002 \
  LR_DECODER=0.00005 \
  STRICT_PROBE_TEST_INTERVAL=10 \
  ./run_stage2_baby_recommender_decoder.sh \
    --fusion_mode mean \
    --completion_gate_mode rank_residual_allnorm \
    --completion_gate_hidden_dim 64 \
    --completion_gate_dropout 0.1 \
    --completion_gate_init_logit 0.0 \
    --completion_gate_detach_inputs 1 \
    --completion_gate_use_item_context 1 \
    --completion_gate_item_context_source shared_mean \
    --completion_gate_residual_alpha 0.18 \
    --completion_gate_mix_alpha 0.35 \
    --completion_gate_identity_coeff 0.05 \
    --completion_gate_balance_coeff 0.01 \
    --completion_gate_reg_coeff 0.1 \
    --recommender_allow_modal_grad 0 2>&1 | tee "${log_path}"

  row="$(parse_one "${seed}" "${log_path}")"
  echo "${row}" >> "${SUMMARY}"
  log "result ${row}"
done
log "done"
