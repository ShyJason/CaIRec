#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
RUN_TAG="${RUN_TAG:-smore_native_imputed_stage2_search_$(date +%Y%m%d_%H%M%S)_gpu${GPU}}"
OUT_DIR="${OUT_DIR:-exp_report/clothing/smore_native_missing_item_embeds_iigraph_imputed_search/${RUN_TAG}}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

CONFIG="${CONFIG:-configs/clothing/mainline_mr0p1.yaml}"
FEATURE_DIR="${FEATURE_DIR:-/home/ruiyuliu/projects/baselines/SMORE/experiment_logs/smore_native_missing_raw_input_compact_clothing_mr0p3_20260702/native_missing_raw_mr0p3}"
IMPUTER_CKPT="${IMPUTER_CKPT:-exp_report/clothing/clothing_mr0p3_smore_native_missing_raw_identity_stage11_observed_5e_20260702_gpu0/ckpt/clothing_mr0p3_smore_native_missing_raw_identity_stage11_observed_5e_20260702_gpu0_imputer_param_50_epoch4.pth}"
EPOCHS="${EPOCHS:-200}"
EARLY_STOP="${EARLY_STOP:-20}"
MEM_FREE_THRESHOLD="${MEM_FREE_THRESHOLD:-45000}"
POLL_SECONDS="${POLL_SECONDS:-60}"

log() {
  echo "[$(date -Is)] $*" | tee -a "${OUT_DIR}/launcher.log"
}

gpu_has_room() {
  local used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" | tr -d ' ')"
  [[ "${used}" =~ ^[0-9]+$ ]] && (( used < MEM_FREE_THRESHOLD ))
}

write_candidates() {
  local path="${OUT_DIR}/candidates.tsv"
  if [[ -f "${path}" ]]; then
    return
  fi
  cat > "${path}" <<'EOF'
tag	topk	cf	image	text	modal_alpha	rec_cl	rec_temp	lr	reg	mod_bpr
topk12_base	12	0.2	0.4	0.4	0.25	0.005	0.2	0.01	0.01	1.0
topk15_base	15	0.2	0.4	0.4	0.25	0.005	0.2	0.01	0.01	1.0
topk8_base	8	0.2	0.4	0.4	0.25	0.005	0.2	0.01	0.01	1.0
sem425_topk10	10	0.15	0.425	0.425	0.25	0.005	0.2	0.01	0.01	1.0
cf25_topk10	10	0.25	0.375	0.375	0.25	0.005	0.2	0.01	0.01	1.0
alpha20_topk10	10	0.2	0.4	0.4	0.20	0.005	0.2	0.01	0.01	1.0
alpha30_topk10	10	0.2	0.4	0.4	0.30	0.005	0.2	0.01	0.01	1.0
reccl0075_topk10	10	0.2	0.4	0.4	0.25	0.0075	0.2	0.01	0.01	1.0
reccl003_topk10	10	0.2	0.4	0.4	0.25	0.003	0.2	0.01	0.01	1.0
img425_txt375	10	0.2	0.425	0.375	0.25	0.005	0.2	0.01	0.01	1.0
img375_txt425	10	0.2	0.375	0.425	0.25	0.005	0.2	0.01	0.01	1.0
reg0075_topk10	10	0.2	0.4	0.4	0.25	0.005	0.2	0.01	0.0075	1.0
EOF
}

summarize() {
  .venv/bin/python - "${LOG_DIR}" "${OUT_DIR}/summary.tsv" <<'PY'
import pathlib
import re
import sys

log_dir = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])
rows = []
for log in sorted(log_dir.glob("*.log")):
    text = log.read_text(errors="ignore")
    final = re.findall(
        r"final strict test hr@20\s*[:=]\s*([0-9.]+).*?ndcg@20\s*[:=]\s*([0-9.]+)",
        text,
        re.I | re.S,
    )
    final50 = re.findall(
        r"final strict test hr@50\s*[:=]\s*([0-9.]+).*?ndcg@50\s*[:=]\s*([0-9.]+)",
        text,
        re.I | re.S,
    )
    best_epoch = re.findall(r"best epoch\s+([0-9]+)", text)
    vals = re.findall(
        r"epoch = ([0-9]+) hr@20 = ([0-9.]+), recall@20 = ([0-9.]+), ndcg@20 = ([0-9.]+)",
        text,
    )
    best_val = max(vals, key=lambda row: float(row[2])) if vals else ("", "", "", "")
    status = "done" if final else ("running" if vals else "failed_or_pending")
    recall20, ndcg20 = final[-1] if final else ("", "")
    recall50, ndcg50 = final50[-1] if final50 else ("", "")
    tag = log.stem
    rows.append((
        status, tag, recall20, ndcg20, recall50, ndcg50,
        best_epoch[-1] if best_epoch else "",
        best_val[0], best_val[2], best_val[3],
        str(log),
    ))

with out_path.open("w") as f:
    f.write("status\ttag\trecall20\tndcg20\trecall50\tndcg50\tbest_epoch\tbest_val_epoch\tbest_val_recall20\tbest_val_ndcg20\tlog\n")
    for row in sorted(rows, key=lambda r: (r[0] != "done", -(float(r[2]) if r[2] else -1.0), r[1])):
        f.write("\t".join(row) + "\n")
PY
}

run_candidate() {
  local tag="$1" topk="$2" cf="$3" image="$4" text="$5" modal_alpha="$6" rec_cl="$7" rec_temp="$8" lr="$9" reg="${10}" mod_bpr="${11}"
  local suffix="stage2_clothing_mr0p3_smore_native_imputed_${tag}_${RUN_TAG}"
  local log_path="${LOG_DIR}/${tag}.log"
  local cmd_file="${log_path}.cmd"

  if rg -q "final strict test hr@20" "${log_path}" 2>/dev/null; then
    log "skip completed ${tag}"
    return
  fi

  while ! gpu_has_room; do
    log "GPU ${GPU} above memory threshold; waiting ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
  done

  log "start ${tag}: topk=${topk}, weights=${cf}/${image}/${text}, alpha=${modal_alpha}, rec_cl=${rec_cl}, lr=${lr}, reg=${reg}, mbpr=${mod_bpr}"
  local cmd=(
    .venv/bin/python -u main.py
    --config "${CONFIG}"
    --device_id "${GPU}"
    --dataset clothing
    --exp_mode mm
    --train_stage recommender
    --suffix "${suffix}"
    --missing_rate 0.3
    --eval_missing_rate 0.5
    --seed 2023
    --dataset_seed 0
    --epoch "${EPOCHS}"
    --early_stop "${EARLY_STOP}"
    --eva_interval 1
    --batch_size 2048
    --lr "${lr}"
    --lr_rec "${lr}"
    --lr_imp 0.0002
    --lr_decoder 0.00005
    --contra_dim 64
    --d_beta 32
    --imputer_ckpt "${IMPUTER_CKPT}"
    --disable_imputation 0
    --freeze_imputer 1
    --freeze_decoder 1
    --recommender_allow_modal_grad 0
    --promrl_projection_mode identity
    --feature_bridge_mode shared_identity
    --gcn_frontend_mode identity
    --modal_feature_override_dir "${FEATURE_DIR}"
    --modal_feature_image_file image_item_embeds.npy
    --modal_feature_text_file text_item_embeds.npy
    --item_graph_kind fused_completed
    --item_graph_feature_space shared
    --item_graph_topk "${topk}"
    --item_graph_cf_weight "${cf}"
    --item_graph_image_weight "${image}"
    --item_graph_text_weight "${text}"
    --item_graph_audio_weight 0.0
    --item_graph_modal_alpha "${modal_alpha}"
    --item_graph_modal_layers 1
    --item_graph_modal_target all
    --rec_neighbor_cl_weight "${rec_cl}"
    --rec_neighbor_cl_temp "${rec_temp}"
    --rec_neighbor_cl_bank_size 256
    --modality_bpr_coeff "${mod_bpr}"
    --reg_coeff "${reg}"
    --selection_mode val
    --recommendation_selection_metric recall
    --recommendation_selection_topk 20
    --evaluation_protocol strict
    --tensorboard 0
    --save 1
    --topk "[10, 20, 30, 40, 50]"
  )

  printf '%q ' "${cmd[@]}" > "${cmd_file}"
  printf '\n' >> "${cmd_file}"
  if "${cmd[@]}" > "${log_path}" 2>&1; then
    log "done ${tag}"
  else
    log "failed ${tag}; see ${log_path}"
  fi
  summarize
}

write_candidates
log "run_tag=${RUN_TAG}"
log "out_dir=${OUT_DIR}"
log "imputer_ckpt=${IMPUTER_CKPT}"
log "feature_dir=${FEATURE_DIR}"

tail -n +2 "${OUT_DIR}/candidates.tsv" | while IFS=$'\t' read -r tag topk cf image text modal_alpha rec_cl rec_temp lr reg mod_bpr; do
  run_candidate "${tag}" "${topk}" "${cf}" "${image}" "${text}" "${modal_alpha}" "${rec_cl}" "${rec_temp}" "${lr}" "${reg}" "${mod_bpr}"
done

summarize
log "all finished"
