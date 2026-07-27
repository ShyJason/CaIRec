#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
DIRECT="${DIRECT:-0}"
RUN_TAG="${RUN_TAG:-fused_conf_sigmoid01_gpu0_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-exp_report/clothing/fused_conf_sigmoid01_mr0p3/${RUN_TAG}}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

CONFIG="${CONFIG:-configs/clothing/mainline_mr0p1.yaml}"
IMPUTER_CKPT="${IMPUTER_CKPT:-exp_report/clothing/stage1_2_clothing_mm_mr0p3_beststyle_nocl_clothing_mr0p3_latest_20260624_174840/ckpt/stage1_2_clothing_mm_mr0p3_beststyle_nocl_clothing_mr0p3_latest_20260624_174840_imputer_backprop_50_epoch49.pth}"

cat > "${OUT_DIR}/candidates.tsv" <<'EOF'
tag	rr	ri	ii
rr100_ri090_ii070	1.00	0.90	0.70
rr100_ri085_ii060	1.00	0.85	0.60
rr100_ri075_ii050	1.00	0.75	0.50
rr080_ri060_ii040	0.80	0.60	0.40
rr100_ri100_ii100	1.00	1.00	1.00
rr060_ri040_ii025	0.60	0.40	0.25
EOF

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
    final = re.findall(r"final strict test hr@20\s*[:=]\s*([0-9.]+).*?ndcg@20\s*[:=]\s*([0-9.]+)", text, re.I | re.S)
    if not final:
        final = re.findall(r"final strict test.*?recall@20\s*[:=]\s*([0-9.]+).*?ndcg@20\s*[:=]\s*([0-9.]+)", text, re.I | re.S)
    best_epoch = re.findall(r"best epoch\s+([0-9]+)", text)
    conf = re.findall(r"learned edge confidence:\s*rr=([0-9.eE+-]+),\s*ri=([0-9.eE+-]+),\s*ii=([0-9.eE+-]+)", text)
    coeff = re.findall(r"learned edge coeff:\s*rr=([0-9.eE+-]+),\s*ri=([0-9.eE+-]+),\s*ii=([0-9.eE+-]+)", text)
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

run_one() {
  local tag="$1"
  local rr="$2"
  local ri="$3"
  local ii="$4"
  local suffix="stage2_clothing_mr0p3_sigmoid01_${tag}_${RUN_TAG}"
  local log="${LOG_DIR}/${suffix}.log"
  local cmd_file="${log}.cmd"

  if [[ -f "${log}" ]] && grep -qi "final strict test hr@20" "${log}"; then
    echo "skip completed ${suffix}"
    return 0
  fi

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
    --imputer_ckpt "${IMPUTER_CKPT}"
    --suffix "${suffix}"
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
    --item_graph_feature_space raw_decoder
    --item_graph_modal_alpha 0.25
    --item_graph_modal_layers 1
    --item_graph_modal_target all
    --item_graph_confidence_transform sigmoid
    --item_graph_confidence_min 0.0
    --item_graph_confidence_max 1.0
    --item_graph_rr_confidence_init "${rr}"
    --item_graph_ri_confidence_init "${ri}"
    --item_graph_ii_confidence_init "${ii}"
    --tensorboard 0
    --save 1
    --topk "[10, 20, 30, 40, 50]"
  )

  printf "%q " "${cmd[@]}" > "${cmd_file}"
  printf "\n" >> "${cmd_file}"
  echo "start ${suffix} on GPU ${GPU}"
  "${cmd[@]}" > "${log}" 2>&1
  summarize
}

if [[ "${DIRECT}" == "1" ]]; then
  while IFS=$'\t' read -r tag rr ri ii; do
    [[ "${tag}" == "tag" ]] && continue
    run_one "${tag}" "${rr}" "${ri}" "${ii}"
  done < "${OUT_DIR}/candidates.tsv"
else
  while IFS=$'\t' read -r tag rr ri ii; do
    [[ "${tag}" == "tag" ]] && continue
    while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
      sleep 30
      summarize || true
    done
    run_one "${tag}" "${rr}" "${ri}" "${ii}" &
  done < "${OUT_DIR}/candidates.tsv"
  wait
fi
summarize
echo "done ${OUT_DIR}"
