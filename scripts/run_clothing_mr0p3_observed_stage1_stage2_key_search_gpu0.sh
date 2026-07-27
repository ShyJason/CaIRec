#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-0}"
DATASET="${DATASET:-clothing}"
EXP_MODE="${EXP_MODE:-mm}"
CONFIG="${CONFIG:-configs/clothing/mainline_mr0p1.yaml}"
MISSING_RATE="${MISSING_RATE:-0.3}"
EVAL_MISSING_RATE="${EVAL_MISSING_RATE:-0.5}"
SEED="${SEED:-2023}"
DATASET_SEED="${DATASET_SEED:-0}"
EPOCHS="${EPOCHS:-200}"
EARLY_STOP="${EARLY_STOP:-20}"
BATCH_SIZE="${BATCH_SIZE:-2048}"

RUN_TAG="${RUN_TAG:-observed_stage1_stage2_key_mr0p3_$(date +%Y%m%d_%H%M%S)}"
BASE_DIR="${BASE_DIR:-exp_report/clothing/observed_stage1_stage2_key_search/${RUN_TAG}}"
LOG_DIR="${BASE_DIR}/logs"
SUMMARY_FILE="${BASE_DIR}/summary.tsv"

IMPUTER_CKPT="${IMPUTER_CKPT:-exp_report/clothing/stage1_2_clothing_stage12_direction_mr0p3_noleak_20260701_185914_observed/ckpt/stage1_2_clothing_stage12_direction_mr0p3_noleak_20260701_185914_observed_imputer_backprop_50_epoch46.pth}"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${IMPUTER_CKPT}" ]]; then
  echo "missing IMPUTER_CKPT: ${IMPUTER_CKPT}" >&2
  exit 1
fi

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "${BASE_DIR}/run.log"
}

# tag lr_rec reg_coeff modality_bpr_coeff cl_weight cl_temp cf_w image_w text_w topk modal_alpha rr ri ii
CANDIDATES=(
  "lr005_all1 0.005 0.01 1.0 0.01 0.2 0.25 0.375 0.375 8 0.25 1.00 1.00 1.00"
  "mbpr15_all1 0.01 0.01 1.5 0.01 0.2 0.25 0.375 0.375 8 0.25 1.00 1.00 1.00"
  "reg005_all1 0.01 0.005 1.0 0.01 0.2 0.25 0.375 0.375 8 0.25 1.00 1.00 1.00"
  "cl005_temp01_all1 0.01 0.01 1.0 0.005 0.1 0.25 0.375 0.375 8 0.25 1.00 1.00 1.00"
  "conf100_090_070 0.01 0.01 1.0 0.01 0.2 0.25 0.375 0.375 8 0.25 1.00 0.90 0.70"
  "conf100_090_070_lr005 0.005 0.01 1.0 0.01 0.2 0.25 0.375 0.375 8 0.25 1.00 0.90 0.70"
  "w030_035_035_all1 0.01 0.01 1.0 0.01 0.2 0.30 0.35 0.35 8 0.25 1.00 1.00 1.00"
  "topk10_all1 0.01 0.01 1.0 0.01 0.2 0.25 0.375 0.375 10 0.25 1.00 1.00 1.00"
)

run_candidate() {
  local tag="$1" lr_rec="$2" reg="$3" mbpr="$4" clw="$5" clt="$6" cf_w="$7" image_w="$8" text_w="$9"
  local topk="${10}" modal_alpha="${11}" rr="${12}" ri="${13}" ii="${14}"
  local suffix="stage2_clothing_mr0p3_observed_stage1_${tag}_${RUN_TAG}"
  local log_file="${LOG_DIR}/${suffix}.log"

  if [[ -f "${log_file}" ]] && rg -q "final strict test hr@20" "${log_file}" && [[ "${FORCE_RERUN:-0}" != "1" ]]; then
    log "skip finished ${tag}"
    return
  fi

  log "start ${tag}: lr=${lr_rec}, reg=${reg}, mbpr=${mbpr}, cl=${clw}/${clt}, graph=${cf_w}/${image_w}/${text_w}, topk=${topk}, rr/ri/ii=${rr}/${ri}/${ii}"
  PYTHONUNBUFFERED=1 .venv/bin/python -u main.py \
    --config "${CONFIG}" \
    --device_id "${GPU}" \
    --dataset "${DATASET}" \
    --exp_mode "${EXP_MODE}" \
    --train_stage recommender \
    --missing_rate "${MISSING_RATE}" \
    --eval_missing_rate "${EVAL_MISSING_RATE}" \
    --seed "${SEED}" \
    --dataset_seed "${DATASET_SEED}" \
    --imputer_ckpt "${IMPUTER_CKPT}" \
    --suffix "${suffix}" \
    --epoch "${EPOCHS}" \
    --early_stop "${EARLY_STOP}" \
    --eva_interval 1 \
    --batch_size "${BATCH_SIZE}" \
    --lr 0.01 \
    --lr_rec "${lr_rec}" \
    --lr_imp 0.0002 \
    --lr_decoder 0.00005 \
    --freeze_imputer 1 \
    --freeze_decoder 1 \
    --recommender_allow_modal_grad 0 \
    --feature_bridge_mode raw_decoder \
    --gcn_frontend_mode original_linear \
    --disable_imputation 0 \
    --modality_bpr_coeff "${mbpr}" \
    --reg_coeff "${reg}" \
    --evaluation_protocol strict \
    --selection_mode val \
    --strict_probe_test_interval 0 \
    --recommendation_selection_metric recall \
    --recommendation_selection_topk 20 \
    --rec_neighbor_cl_weight "${clw}" \
    --rec_neighbor_cl_temp "${clt}" \
    --rec_neighbor_cl_bank_size 256 \
    --item_graph_kind fused_completed_confidence \
    --item_graph_topk "${topk}" \
    --item_graph_norm rw \
    --item_graph_cf_weight "${cf_w}" \
    --item_graph_image_weight "${image_w}" \
    --item_graph_text_weight "${text_w}" \
    --item_graph_audio_weight 0.0 \
    --item_graph_feature_space raw_decoder \
    --item_graph_modal_alpha "${modal_alpha}" \
    --item_graph_modal_layers 1 \
    --item_graph_modal_target all \
    --item_graph_confidence_transform sigmoid \
    --item_graph_confidence_min 0.0 \
    --item_graph_confidence_max 1.0 \
    --item_graph_rr_confidence_init "${rr}" \
    --item_graph_ri_confidence_init "${ri}" \
    --item_graph_ii_confidence_init "${ii}" \
    --tensorboard 0 \
    --save 1 \
    --topk "[10, 20, 30, 40, 50]" \
    > "${log_file}" 2>&1
  log "done ${tag}"
}

summarize() {
  .venv/bin/python - "${LOG_DIR}" "${SUMMARY_FILE}" <<'PY'
import re
import sys
from pathlib import Path

log_dir = Path(sys.argv[1])
summary = Path(sys.argv[2])

def final_at(text, k):
    vals = re.findall(rf"final strict test hr@{k}\s*=\s*([0-9.]+),\s*recall@{k}\s*=\s*([0-9.]+),\s*ndcg@{k}\s*=\s*([0-9.]+)", text)
    return tuple(map(float, vals[-1])) if vals else None

rows = []
for path in sorted(log_dir.glob("*.log")):
    text = path.read_text(errors="ignore")
    f20 = final_at(text, 20)
    if not f20:
        continue
    best = re.findall(r"best epoch\s+([0-9]+)", text)
    f50 = final_at(text, 50)
    rows.append({
        "tag": path.stem.replace("stage2_clothing_mr0p3_observed_stage1_", ""),
        "recall20": f"{f20[1]:.5f}",
        "ndcg20": f"{f20[2]:.5f}",
        "recall50": f"{f50[1]:.5f}" if f50 else "",
        "ndcg50": f"{f50[2]:.5f}" if f50 else "",
        "best_epoch": best[-1] if best else "",
        "log": str(path),
    })

rows.sort(key=lambda r: float(r["recall20"]), reverse=True)
fields = ["tag", "recall20", "ndcg20", "recall50", "ndcg50", "best_epoch", "log"]
summary.write_text(
    "\t".join(fields) + "\n" + "\n".join("\t".join(row[f] for f in fields) for row in rows) + ("\n" if rows else "")
)
print(summary)
PY
}

log "run_tag=${RUN_TAG}"
log "imputer_ckpt=${IMPUTER_CKPT}"
for cand in "${CANDIDATES[@]}"; do
  read -r tag lr_rec reg mbpr clw clt cf_w image_w text_w topk modal_alpha rr ri ii <<<"${cand}"
  run_candidate "${tag}" "${lr_rec}" "${reg}" "${mbpr}" "${clw}" "${clt}" "${cf_w}" "${image_w}" "${text_w}" "${topk}" "${modal_alpha}" "${rr}" "${ri}" "${ii}"
  summarize | tee -a "${BASE_DIR}/run.log"
done
log "done summary=${SUMMARY_FILE}"
