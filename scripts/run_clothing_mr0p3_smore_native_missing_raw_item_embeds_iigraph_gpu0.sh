#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
RUN_TAG="${RUN_TAG:-clothing_mr0p3_smore_native_missing_raw_item_embeds_iigraph_$(date +%Y%m%d_%H%M%S)_gpu${GPU}}"
OUT_DIR="${OUT_DIR:-exp_report/clothing/smore_native_missing_item_embeds_iigraph/${RUN_TAG}}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

CONFIG="${CONFIG:-configs/clothing/mainline_mr0p1.yaml}"
FEATURE_DIR="${FEATURE_DIR:-/home/ruiyuliu/projects/baselines/SMORE/experiment_logs/smore_native_missing_raw_input_compact_clothing_mr0p3_20260702/native_missing_raw_mr0p3}"
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
    best_epoch = re.findall(r"best epoch\s+([0-9]+)", text)
    conf = re.findall(
        r"learned edge confidence:\s*rr=([0-9.eE+-]+),\s*ri=([0-9.eE+-]+),\s*ii=([0-9.eE+-]+)",
        text,
    )
    coeff = re.findall(
        r"learned edge coeff:\s*rr=([0-9.eE+-]+),\s*ri=([0-9.eE+-]+),\s*ii=([0-9.eE+-]+)",
        text,
    )
    status = "done" if final else "running_or_failed"
    recall, ndcg = final[-1] if final else ("", "")
    rr, ri, ii = conf[-1] if conf else ("", "", "")
    crr, cri, cii = coeff[-1] if coeff else ("", "", "")
    rows.append((status, recall, ndcg, best_epoch[-1] if best_epoch else "", rr, ri, ii, crr, cri, cii, str(log)))

with out_path.open("w") as f:
    f.write("status\trecall20\tndcg20\tbest_epoch\tconf_rr\tconf_ri\tconf_ii\tcoeff_rr\tcoeff_ri\tcoeff_ii\tlog\n")
    for row in sorted(rows, key=lambda r: (r[0] != "done", -(float(r[1]) if r[1] else -1.0), r[-1])):
        f.write("\t".join(row) + "\n")
PY
}

run_identity_iigraph() {
  local suffix="stage2_clothing_mr0p3_smore_native_missing_raw_item_embeds_identity_iigraph_${RUN_TAG}"
  local log_path="${LOG_DIR}/native_missing_raw_item_embeds_identity_iigraph.log"
  local cmd_file="${log_path}.cmd"

  log "start SMORE native missing raw image_item/text_item + mainline ii graph"
  log "feature_dir=${FEATURE_DIR}"

  local cmd=(
    .venv/bin/python -u main.py
    --config "${CONFIG}"
    --device_id "${GPU}"
    --dataset clothing
    --exp_mode mm
    --train_stage recommender
    --missing_rate 0.3
    --eval_missing_rate 0.5
    --seed 2023
    --dataset_seed 0
    --suffix "${suffix}"
    --epoch "${EPOCHS}"
    --early_stop "${EARLY_STOP}"
    --eva_interval 1
    --batch_size 2048
    --lr 0.01
    --lr_rec 0.01
    --lr_imp 0.0002
    --lr_decoder 0.00005
    --freeze_imputer 1
    --freeze_decoder 1
    --recommender_allow_modal_grad 0
    --feature_bridge_mode raw_decoder
    --gcn_frontend_mode identity
    --disable_imputation 1
    --modality_bpr_coeff 1.0
    --reg_coeff 0.01
    --evaluation_protocol strict
    --selection_mode val
    --strict_probe_test_interval 0
    --recommendation_selection_metric recall
    --recommendation_selection_topk 20
    --rec_neighbor_cl_weight 0.01
    --rec_neighbor_cl_temp 0.2
    --rec_neighbor_cl_bank_size 256
    --item_graph_kind fused_completed_confidence
    --item_graph_topk 8
    --item_graph_norm rw
    --item_graph_cf_weight 0.25
    --item_graph_image_weight 0.375
    --item_graph_text_weight 0.375
    --item_graph_audio_weight 0.0
    --item_graph_feature_space shared
    --item_graph_modal_alpha 0.25
    --item_graph_modal_layers 1
    --item_graph_modal_target all
    --item_graph_confidence_transform sigmoid
    --item_graph_confidence_min 0.0
    --item_graph_confidence_max 1.0
    --item_graph_rr_confidence_init 1.00
    --item_graph_ri_confidence_init 1.00
    --item_graph_ii_confidence_init 1.00
    --modal_feature_override_dir "${FEATURE_DIR}"
    --modal_feature_image_file image_item_embeds.npy
    --modal_feature_text_file text_item_embeds.npy
    --tensorboard 0
    --save 1
    --topk "[10, 20, 30, 40, 50]"
  )

  printf '%q ' "${cmd[@]}" > "${cmd_file}"
  printf '\n' >> "${cmd_file}"
  "${cmd[@]}" > "${log_path}" 2>&1
  summarize
  log "done; log=${log_path}"
}

log "run_tag=${RUN_TAG}"
log "out_dir=${OUT_DIR}"
while ! gpu_has_room; do
  log "GPU ${GPU} above memory threshold; waiting ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done

run_identity_iigraph
log "all finished"
