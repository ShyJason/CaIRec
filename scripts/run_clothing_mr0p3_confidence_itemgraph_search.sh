#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="clothing"
EXP_MODE="mm"
CONFIG="${CONFIG:-configs/clothing/mainline_mr0p1.yaml}"
MISSING_RATE="${MISSING_RATE:-0.3}"
EVAL_MISSING_RATE="${EVAL_MISSING_RATE:-0.5}"
MR_TAG="${MISSING_RATE//./p}"
RUN_TAG="${RUN_TAG:-clothing_mr${MR_TAG}_confidence_itemgraph_$(date +%Y%m%d_%H%M%S)}"
BASE_DIR="${BASE_DIR:-exp_report/clothing/confidence_itemgraph_mr0p3_search/${RUN_TAG}}"

GPUS="${GPUS:-0 7}"
TASKS_PER_GPU="${TASKS_PER_GPU:-2}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
POLL_SECONDS="${POLL_SECONDS:-30}"
DRY_RUN="${DRY_RUN:-0}"
CANDIDATE_LIMIT="${CANDIDATE_LIMIT:-0}"

SEED="${SEED:-2023}"
DATASET_SEED="${DATASET_SEED:-0}"
EPOCHS="${EPOCHS:-200}"
EARLY_STOP="${EARLY_STOP:-20}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
CKPT_EPOCH="${CKPT_EPOCH:-49}"

LOG_DIR="${BASE_DIR}/logs"
STATE_DIR="${BASE_DIR}/state"
mkdir -p "${LOG_DIR}" "${STATE_DIR}"

NEXT_FILE="${STATE_DIR}/next_index"
LOCK_FILE="${STATE_DIR}/queue.lock"
LAUNCHER_LOG="${BASE_DIR}/launcher.log"
CANDIDATES_FILE="${BASE_DIR}/candidates.tsv"
SUMMARY_FILE="${BASE_DIR}/summary.tsv"
SUMMARY_VAL_FILE="${BASE_DIR}/summary_val.tsv"
SUMMARY_TOPK_FILE="${BASE_DIR}/summary_topk.tsv"

log() {
  echo "[$(date -Is)] $*" | tee -a "${LAUNCHER_LOG}"
}

discover_stage12_dir() {
  find exp_report/clothing -maxdepth 1 -type d \
    -name "stage1_2_clothing_mm_mr${MR_TAG}_beststyle_nocl_*" \
    -printf "%T@ %p\n" 2>/dev/null \
    | sort -nr \
    | awk 'NR == 1 { $1=""; sub(/^ /, ""); print }'
}

STAGE12_DIR="${STAGE12_DIR:-$(discover_stage12_dir)}"
if [[ -n "${STAGE12_DIR}" ]]; then
  STAGE12_PREFIX="${STAGE12_PREFIX:-$(basename "${STAGE12_DIR}")}"
else
  STAGE12_PREFIX="${STAGE12_PREFIX:-stage1_2_clothing_mm_mr${MR_TAG}_beststyle_nocl_<RUN_TAG>}"
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "missing config: ${CONFIG}" >&2
  exit 1
fi

if [[ "${DRY_RUN}" != "1" && ( -z "${STAGE12_DIR}" || ! -d "${STAGE12_DIR}" ) ]]; then
  echo "missing STAGE12_DIR; set STAGE12_DIR and STAGE12_PREFIX or finish stage1.2 first" >&2
  exit 1
fi

ckpt_path_for_epoch() {
  local ckpt_epoch="$1"
  printf "%s/ckpt/%s_imputer_backprop_50_epoch%s.pth" "${STAGE12_DIR}" "${STAGE12_PREFIX}" "${ckpt_epoch}"
}

slug() {
  printf "%s" "$1" | sed 's/\./p/g; s/-/m/g'
}

generate_candidates() {
  if [[ -f "${CANDIDATES_FILE}" ]]; then
    return
  fi
  {
    printf "tag\tblend\trr\tri\tii\tconf_min\tconf_max\talpha\tlr_rec\treg_coeff\tcl_weight\tcl_temp\tcf_w\timage_w\ttext_w\ttopk\tseed\tdataset_seed\n"
    add_row() {
      local tag="$1" blend="$2" rr="$3" ri="$4" ii="$5" alpha="$6" lr_rec="$7" reg="$8" clw="$9" clt="${10}"
      printf "%s\t%s\t%s\t%s\t%s\t0.50\t2.00\t%s\t%s\t%s\t%s\t%s\t0.25\t0.375\t0.375\t8\t%s\t%s\n" \
        "${tag}" "${blend}" "${rr}" "${ri}" "${ii}" "${alpha}" "${lr_rec}" "${reg}" "${clw}" "${clt}" "${SEED}" "${DATASET_SEED}"
    }
    add_row "b025_rr12_ri09_ii07" "0.25" "1.20" "0.90" "0.70" "0.25" "0.01" "0.01" "0.01" "0.2"
    add_row "b025_rr15_ri085_ii06" "0.25" "1.50" "0.85" "0.60" "0.25" "0.01" "0.01" "0.01" "0.2"
    add_row "b025_rr10_ri08_ii05" "0.25" "1.00" "0.80" "0.50" "0.25" "0.01" "0.01" "0.01" "0.2"
    add_row "b025_rr13_ri10_ii07" "0.25" "1.30" "1.00" "0.70" "0.25" "0.01" "0.01" "0.01" "0.2"
    add_row "b050_rr12_ri09_ii07" "0.50" "1.20" "0.90" "0.70" "0.25" "0.01" "0.01" "0.01" "0.2"
    add_row "b050_rr15_ri085_ii06" "0.50" "1.50" "0.85" "0.60" "0.25" "0.01" "0.01" "0.01" "0.2"
    add_row "b050_rr10_ri08_ii05" "0.50" "1.00" "0.80" "0.50" "0.25" "0.01" "0.01" "0.01" "0.2"
    add_row "b050_rr13_ri10_ii07" "0.50" "1.30" "1.00" "0.70" "0.25" "0.01" "0.01" "0.01" "0.2"
    add_row "b025_rr15_ri085_ii06_alpha010" "0.25" "1.50" "0.85" "0.60" "0.10" "0.01" "0.01" "0.01" "0.2"
    add_row "b050_rr12_ri09_ii07_alpha010" "0.50" "1.20" "0.90" "0.70" "0.10" "0.01" "0.01" "0.01" "0.2"
    add_row "b025_rr15_ri085_ii06_lr005" "0.25" "1.50" "0.85" "0.60" "0.25" "0.005" "0.01" "0.01" "0.2"
    add_row "b025_rr12_ri09_ii07_reg005" "0.25" "1.20" "0.90" "0.70" "0.25" "0.01" "0.005" "0.01" "0.2"
  } > "${CANDIDATES_FILE}"
}

summarize() {
  python3 - "${LOG_DIR}" "${CANDIDATES_FILE}" "${SUMMARY_FILE}" "${SUMMARY_VAL_FILE}" "${SUMMARY_TOPK_FILE}" <<'PY'
import csv
import re
import sys
from pathlib import Path

log_dir, candidates_file, summary_file, summary_val_file, summary_topk_file = map(Path, sys.argv[1:])
with candidates_file.open(newline="") as f:
    candidates = {row["tag"]: row for row in csv.DictReader(f, delimiter="\t")}

def tag_from_log(path):
    for tag in candidates:
        if f"_{tag}_" in path.name:
            return tag
    return path.stem

def best_epoch(text):
    vals = re.findall(r"best epoch\s+([0-9]+)", text)
    return int(vals[-1]) if vals else -1

def final_at_k(text, k):
    vals = re.findall(
        rf"final strict test hr@{k}\s*=\s*([0-9.]+),\s*recall@{k}\s*=\s*([0-9.]+),\s*ndcg@{k}\s*=\s*([0-9.]+)",
        text,
    )
    return tuple(map(float, vals[-1])) if vals else None

def val_at_20(text, epoch):
    vals = re.findall(
        rf"epoch\s*=\s*{epoch}\s+hr@20\s*=\s*([0-9.]+),\s*recall@20\s*=\s*([0-9.]+),\s*ndcg@20\s*=\s*([0-9.]+)",
        text,
    ) if epoch >= 0 else []
    if vals:
        return tuple(map(float, vals[-1]))
    vals = re.findall(r"hr@20\s*=\s*([0-9.]+),\s*recall@20\s*=\s*([0-9.]+),\s*ndcg@20\s*=\s*([0-9.]+)", text)
    return tuple(map(float, vals[-1])) if vals else None

rows_final = []
rows_val = []
rows_topk = []
for path in sorted(log_dir.glob("*.log")):
    text = path.read_text(errors="ignore")
    if "final strict test hr@20" not in text:
        continue
    tag = tag_from_log(path)
    cand = candidates.get(tag, {})
    epoch = best_epoch(text)
    row_base = {
        "tag": tag,
        "blend": cand.get("blend", ""),
        "rr": cand.get("rr", ""),
        "ri": cand.get("ri", ""),
        "ii": cand.get("ii", ""),
        "alpha": cand.get("alpha", ""),
        "lr_rec": cand.get("lr_rec", ""),
        "reg_coeff": cand.get("reg_coeff", ""),
        "best_epoch": str(epoch),
        "log": path.name,
    }
    f20 = final_at_k(text, 20)
    if f20:
        _, rec, ndcg = f20
        rows_final.append({**row_base, "recall20": f"{rec:.5f}", "ndcg20": f"{ndcg:.5f}"})
    v20 = val_at_20(text, epoch)
    if v20:
        _, rec, ndcg = v20
        rows_val.append({**row_base, "recall20": f"{rec:.5f}", "ndcg20": f"{ndcg:.5f}"})
    top = {"tag": tag, "best_epoch": str(epoch), "log": path.name}
    for k in (10, 20, 30, 40, 50):
        vals = final_at_k(text, k)
        if vals:
            hr, rec, ndcg = vals
            top[f"hr@{k}"] = f"{hr:.5f}"
            top[f"recall@{k}"] = f"{rec:.5f}"
            top[f"ndcg@{k}"] = f"{ndcg:.5f}"
    if any(key.startswith("recall@") for key in top):
        rows_topk.append(top)

def sort_key(row):
    return (float(row.get("recall20", 0.0)), float(row.get("ndcg20", 0.0)))

rows_final.sort(key=sort_key, reverse=True)
rows_val.sort(key=sort_key, reverse=True)

summary_fields = ["tag", "recall20", "ndcg20", "blend", "rr", "ri", "ii", "alpha", "lr_rec", "reg_coeff", "best_epoch", "log"]
for path, fields, rows in (
    (summary_file, summary_fields, rows_final),
    (summary_val_file, summary_fields, rows_val),
    (summary_topk_file, [
        "tag", "hr@10", "recall@10", "ndcg@10", "hr@20", "recall@20", "ndcg@20",
        "hr@30", "recall@30", "ndcg@30", "hr@40", "recall@40", "ndcg@40",
        "hr@50", "recall@50", "ndcg@50", "best_epoch", "log",
    ], rows_topk),
):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
PY
}

suffix_for_tag() {
  local tag="$1"
  printf "stage2_clothing_mr%s_conf_%s_%s" "${MR_TAG}" "${tag}" "${RUN_TAG}"
}

build_command() {
  local gpu="$1" tag="$2" blend="$3" rr="$4" ri="$5" ii="$6" conf_min="$7" conf_max="$8" alpha="$9"
  local lr_rec="${10}" reg="${11}" clw="${12}" clt="${13}" cf_w="${14}" image_w="${15}" text_w="${16}"
  local topk="${17}" seed="${18}" dataset_seed="${19}"
  local suffix imputer_ckpt
  suffix="$(suffix_for_tag "${tag}")"
  imputer_ckpt="$(ckpt_path_for_epoch "${CKPT_EPOCH}")"
  printf "%q " .venv/bin/python -u main.py \
    --config "${CONFIG}" \
    --device_id "${gpu}" \
    --dataset "${DATASET}" \
    --exp_mode "${EXP_MODE}" \
    --train_stage recommender \
    --missing_rate "${MISSING_RATE}" \
    --eval_missing_rate "${EVAL_MISSING_RATE}" \
    --seed "${seed}" \
    --dataset_seed "${dataset_seed}" \
    --imputer_ckpt "${imputer_ckpt}" \
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
    --modality_bpr_coeff 1.0 \
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
    --item_graph_modal_alpha "${alpha}" \
    --item_graph_modal_layers 1 \
    --item_graph_modal_target all \
    --item_graph_rr_confidence_init "${rr}" \
    --item_graph_ri_confidence_init "${ri}" \
    --item_graph_ii_confidence_init "${ii}" \
    --item_graph_confidence_blend "${blend}" \
    --item_graph_confidence_min "${conf_min}" \
    --item_graph_confidence_max "${conf_max}" \
    --tensorboard 0 \
    --save 1 \
    --topk "[10, 20, 30, 40, 50]"
  printf "\n"
}

run_candidate() {
  local gpu="$1" tag="$2" blend="$3" rr="$4" ri="$5" ii="$6" conf_min="$7" conf_max="$8" alpha="$9"
  local lr_rec="${10}" reg="${11}" clw="${12}" clt="${13}" cf_w="${14}" image_w="${15}" text_w="${16}"
  local topk="${17}" seed="${18}" dataset_seed="${19}"
  local suffix log_path imputer_ckpt
  suffix="$(suffix_for_tag "${tag}")"
  log_path="${LOG_DIR}/${suffix}.log"
  imputer_ckpt="$(ckpt_path_for_epoch "${CKPT_EPOCH}")"
  build_command "${gpu}" "${tag}" "${blend}" "${rr}" "${ri}" "${ii}" "${conf_min}" "${conf_max}" "${alpha}" \
    "${lr_rec}" "${reg}" "${clw}" "${clt}" "${cf_w}" "${image_w}" "${text_w}" "${topk}" "${seed}" "${dataset_seed}" \
    > "${log_path}.cmd"
  if [[ ! -f "${imputer_ckpt}" && "${DRY_RUN}" != "1" ]]; then
    log "skip ${tag}: missing imputer checkpoint ${imputer_ckpt}"
    return 0
  fi
  log "launch ${tag} on GPU ${gpu}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    cat "${log_path}.cmd"
    return 0
  fi
  bash -lc "$(cat "${log_path}.cmd")" > "${log_path}" 2>&1
  summarize
  log "done ${tag} on GPU ${gpu}"
}

claim_next_candidate() {
  (
    flock -x 9
    local idx total line tag suffix log_path
    idx="$(cat "${NEXT_FILE}")"
    total="${#CANDIDATES[@]}"
    while (( idx < total )); do
      line="${CANDIDATES[$idx]}"
      tag="$(printf "%s" "${line}" | cut -f1)"
      suffix="$(suffix_for_tag "${tag}")"
      log_path="${LOG_DIR}/${suffix}.log"
      if [[ -f "${log_path}" ]] && grep -q "final strict test hr@20" "${log_path}"; then
        echo "[$(date -Is)] skip finished ${tag}" >> "${LAUNCHER_LOG}"
        idx=$((idx + 1))
        echo "${idx}" > "${NEXT_FILE}"
        continue
      fi
      echo $((idx + 1)) > "${NEXT_FILE}"
      printf "%s\n" "${line}"
      return 0
    done
    return 1
  ) 9>"${LOCK_FILE}"
}

worker_loop() {
  local worker_idx="$1" gpu="$2" cand
  log "worker=${worker_idx} gpu=${gpu} started"
  while true; do
    if ! cand="$(claim_next_candidate)"; then
      log "worker=${worker_idx} gpu=${gpu} no candidates left"
      return 0
    fi
    IFS=$'\t' read -r tag blend rr ri ii conf_min conf_max alpha lr_rec reg clw clt cf_w image_w text_w topk seed dataset_seed <<<"${cand}"
    run_candidate "${gpu}" "${tag}" "${blend}" "${rr}" "${ri}" "${ii}" "${conf_min}" "${conf_max}" "${alpha}" \
      "${lr_rec}" "${reg}" "${clw}" "${clt}" "${cf_w}" "${image_w}" "${text_w}" "${topk}" "${seed}" "${dataset_seed}"
    if [[ "${DRY_RUN}" != "1" ]]; then
      sleep "${POLL_SECONDS}"
    fi
  done
}

generate_candidates
mapfile -t CANDIDATES < <(tail -n +2 "${CANDIDATES_FILE}")
if (( CANDIDATE_LIMIT > 0 && CANDIDATE_LIMIT < ${#CANDIDATES[@]} )); then
  CANDIDATES=("${CANDIDATES[@]:0:${CANDIDATE_LIMIT}}")
fi
if [[ ! -f "${NEXT_FILE}" ]]; then
  echo 0 > "${NEXT_FILE}"
fi

log "base_dir=${BASE_DIR}"
log "stage12_dir=${STAGE12_DIR}"
log "candidates=${#CANDIDATES[@]} gpus=${GPUS} tasks_per_gpu=${TASKS_PER_GPU} max_parallel=${MAX_PARALLEL} dry_run=${DRY_RUN}"

worker_idx=0
pids=()
for gpu in ${GPUS}; do
  for _ in $(seq 1 "${TASKS_PER_GPU}"); do
    if (( worker_idx >= MAX_PARALLEL )); then
      break 2
    fi
    worker_loop "${worker_idx}" "${gpu}" &
    pids+=("$!")
    worker_idx=$((worker_idx + 1))
  done
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
summarize || true
log "all workers finished status=${status}"
exit "${status}"
