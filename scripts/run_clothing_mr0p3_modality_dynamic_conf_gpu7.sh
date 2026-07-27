#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-7}"
RUN_TAG="${RUN_TAG:-modality_dynamic_conf_gpu${GPU}_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-exp_report/clothing/modality_dynamic_conf_itemgraph_mr0p3/${RUN_TAG}}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

ITEM_GRAPH_CF_WEIGHT="${ITEM_GRAPH_CF_WEIGHT:-0.2}"
ITEM_GRAPH_IMAGE_WEIGHT="${ITEM_GRAPH_IMAGE_WEIGHT:-0.4}"
ITEM_GRAPH_TEXT_WEIGHT="${ITEM_GRAPH_TEXT_WEIGHT:-0.4}"
ITEM_GRAPH_IMAGE_CF_WEIGHT="${ITEM_GRAPH_IMAGE_CF_WEIGHT:-${ITEM_GRAPH_CF_WEIGHT}}"
ITEM_GRAPH_IMAGE_SEMANTIC_WEIGHT="${ITEM_GRAPH_IMAGE_SEMANTIC_WEIGHT:-${ITEM_GRAPH_IMAGE_WEIGHT}}"
ITEM_GRAPH_TEXT_CF_WEIGHT="${ITEM_GRAPH_TEXT_CF_WEIGHT:-${ITEM_GRAPH_CF_WEIGHT}}"
ITEM_GRAPH_TEXT_SEMANTIC_WEIGHT="${ITEM_GRAPH_TEXT_SEMANTIC_WEIGHT:-${ITEM_GRAPH_TEXT_WEIGHT}}"

CONFIG="${CONFIG:-configs/clothing/mainline_mr0p1.yaml}"
IMPUTER_CKPT="${IMPUTER_CKPT:-exp_report/clothing/stage1_2_clothing_mm_mr0p3_beststyle_nocl_clothing_mr0p3_latest_20260624_174840/ckpt/stage1_2_clothing_mm_mr0p3_beststyle_nocl_clothing_mr0p3_latest_20260624_174840_imputer_backprop_50_epoch49.pth}"

SUFFIX="stage2_clothing_mr0p3_modality_dynamic_conf_${RUN_TAG}"
LOG_PATH="${LOG_DIR}/${SUFFIX}.log"
CMD_PATH="${LOG_PATH}.cmd"

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
    final = re.findall(r"final strict test hr@20\s*=\s*([0-9.]+).*?ndcg@20\s*=\s*([0-9.]+)", text, re.S)
    best_epoch = re.findall(r"best epoch\s+([0-9]+)", text)
    conf = re.findall(r"learned edge confidence:\s*rr=([0-9.eE+-]+),\s*ri=([0-9.eE+-]+),\s*ii=([0-9.eE+-]+)", text)
    status = "done" if final else "running_or_failed"
    recall, ndcg = final[-1] if final else ("", "")
    rr, ri, ii = conf[-1] if conf else ("", "", "")
    rows.append((status, recall, ndcg, best_epoch[-1] if best_epoch else "", rr, ri, ii, str(log)))

with out_path.open("w") as f:
    f.write("status\trecall20\tndcg20\tbest_epoch\tconf_rr\tconf_ri\tconf_ii\tlog\n")
    for row in rows:
        f.write("\t".join(row) + "\n")
PY
}

if [[ -f "${LOG_PATH}" ]] && grep -qi "final strict test hr@20" "${LOG_PATH}"; then
  echo "skip completed ${SUFFIX}"
  summarize
  exit 0
fi

CMD=(
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
  --imputer_ckpt "${IMPUTER_CKPT}"
  --suffix "${SUFFIX}"
  --epoch 200
  --early_stop 20
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
  --gcn_frontend_mode original_linear
  --disable_imputation 0
  --modality_bpr_coeff 1.0
  --reg_coeff 0.01
  --evaluation_protocol strict
  --selection_mode val
  --strict_probe_test_interval 0
  --recommendation_selection_metric recall
  --recommendation_selection_topk 20
  --rec_neighbor_cl_weight 0.005
  --rec_neighbor_cl_temp 0.2
  --rec_neighbor_cl_bank_size 256
  --item_graph_kind modality_completed_dynamic_confidence
  --item_graph_topk 10
  --item_graph_norm rw
  --item_graph_cf_weight "${ITEM_GRAPH_CF_WEIGHT}"
  --item_graph_cf_scale raw
  --item_graph_image_weight "${ITEM_GRAPH_IMAGE_WEIGHT}"
  --item_graph_text_weight "${ITEM_GRAPH_TEXT_WEIGHT}"
  --item_graph_image_cf_weight "${ITEM_GRAPH_IMAGE_CF_WEIGHT}"
  --item_graph_image_semantic_weight "${ITEM_GRAPH_IMAGE_SEMANTIC_WEIGHT}"
  --item_graph_text_cf_weight "${ITEM_GRAPH_TEXT_CF_WEIGHT}"
  --item_graph_text_semantic_weight "${ITEM_GRAPH_TEXT_SEMANTIC_WEIGHT}"
  --item_graph_audio_weight 0.0
  --item_graph_modal_alpha 0.25
  --item_graph_modal_layers 1
  --item_graph_modal_target all
  --item_graph_confidence_transform sigmoid
  --item_graph_confidence_min 0.0
  --item_graph_confidence_max 1.0
  --item_graph_rr_confidence_init 1.0
  --item_graph_ri_confidence_init 1.0
  --item_graph_ii_confidence_init 1.0
  --tensorboard 0
  --save 1
  --topk "[10, 20, 30, 40, 50]"
)

printf "%q " "${CMD[@]}" > "${CMD_PATH}"
printf "\n" >> "${CMD_PATH}"
echo "start ${SUFFIX} on GPU ${GPU}"
"${CMD[@]}" > "${LOG_PATH}" 2>&1
status=$?
echo "RUN_EXIT_STATUS=${status}" >> "${LOG_PATH}"
summarize || true
exit "${status}"
