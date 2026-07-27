#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="clothing"
EXP_MODE="mm"
CONFIG="${CONFIG:-configs/clothing/mainline_mr0p1.yaml}"
MISSING_RATE="${MISSING_RATE:-0.3}"
EVAL_MISSING_RATE="${EVAL_MISSING_RATE:-0.5}"
MR_TAG="${MISSING_RATE//./p}"
RUN_TAG="${RUN_TAG:-clothing_mr${MR_TAG}_compact_hparam_$(date +%Y%m%d_%H%M%S)}"
BASE_DIR="${BASE_DIR:-exp_report/clothing/mr0p3_compact_hparam_search/${RUN_TAG}}"

PHASE="${PHASE:-1}"
GPUS="${GPUS:-0 1}"
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

BASELINE_RECALL20="${BASELINE_RECALL20:-0.07738}"
TIE_EPS="${TIE_EPS:-0.0003}"

PHASE_DIR=""
case "${PHASE}" in
  1|phase1|phase1_graph) PHASE="1"; PHASE_DIR="${BASE_DIR}/phase1_graph" ;;
  2|phase2|phase2_cl) PHASE="2"; PHASE_DIR="${BASE_DIR}/phase2_cl" ;;
  3|phase3|phase3_train) PHASE="3"; PHASE_DIR="${BASE_DIR}/phase3_train" ;;
  4|phase4|phase4_confirm) PHASE="4"; PHASE_DIR="${BASE_DIR}/phase4_confirm" ;;
  all)
    for phase in 1 2 3 4; do
      PHASE="${phase}" BASE_DIR="${BASE_DIR}" RUN_TAG="${RUN_TAG}" GPUS="${GPUS}" \
      TASKS_PER_GPU="${TASKS_PER_GPU}" MAX_PARALLEL="${MAX_PARALLEL}" \
      MISSING_RATE="${MISSING_RATE}" EVAL_MISSING_RATE="${EVAL_MISSING_RATE}" \
      BATCH_SIZE="${BATCH_SIZE}" bash "$0"
    done
    exit 0
    ;;
  *) echo "PHASE must be 1, 2, 3, 4, or all; got ${PHASE}" >&2; exit 2 ;;
esac

LOG_DIR="${PHASE_DIR}/logs"
STATE_DIR="${PHASE_DIR}/state"
mkdir -p "${LOG_DIR}" "${STATE_DIR}"

NEXT_FILE="${STATE_DIR}/next_index"
LOCK_FILE="${STATE_DIR}/queue.lock"
LAUNCHER_LOG="${PHASE_DIR}/launcher.log"
CANDIDATES_FILE="${PHASE_DIR}/candidates.tsv"
SUMMARY_FILE="${PHASE_DIR}/summary.tsv"
SUMMARY_VAL_FILE="${PHASE_DIR}/summary_val.tsv"
SUMMARY_TOPK_FILE="${PHASE_DIR}/summary_topk.tsv"

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

generate_candidates() {
  python3 - "${PHASE}" "${BASE_DIR}" "${CANDIDATES_FILE}" "${CKPT_EPOCH}" "${SEED}" "${DATASET_SEED}" "${BASELINE_RECALL20}" "${TIE_EPS}" <<'PY'
import csv
import re
import sys
from pathlib import Path

phase, base_dir, out_path, ckpt_epoch, seed, dataset_seed, baseline, tie_eps = sys.argv[1:]
base = Path(base_dir)
out = Path(out_path)
baseline = float(baseline)
tie_eps = float(tie_eps)
header = [
    "tag", "ckpt_epoch", "modality_bpr_coeff", "lr_rec", "reg_coeff",
    "item_graph_modal_alpha", "rec_neighbor_cl_weight", "rec_neighbor_cl_temp",
    "rec_neighbor_cl_bank_size", "item_graph_cf_weight", "item_graph_image_weight",
    "item_graph_text_weight", "item_graph_topk", "seed", "dataset_seed",
]

def slug(value):
    return str(value).replace(".", "p").replace("-", "m")

def write(rows):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)

def read_candidates(path):
    with Path(path).open(newline="") as f:
        return {row["tag"]: row for row in csv.DictReader(f, delimiter="\t")}

def sorted_summary(summary_path):
    path = Path(summary_path)
    if not path.exists():
        raise SystemExit(f"missing prerequisite summary: {path}")
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if not row.get("tag"):
                continue
            rows.append(row)
    if not rows:
        raise SystemExit(f"no completed rows in prerequisite summary: {path}")
    rows.sort(
        key=lambda r: (
            float(r.get("recall20", "0") or 0),
            float(r.get("ndcg20", "0") or 0),
        ),
        reverse=True,
    )
    return rows

def prior_settings(phase_dir, n):
    summary = sorted_summary(base / phase_dir / "summary_val.tsv")[:n]
    cands = read_candidates(base / phase_dir / "candidates.tsv")
    settings = []
    for row in summary:
        tag = row["tag"]
        if tag not in cands:
            raise SystemExit(f"tag {tag} missing from {phase_dir}/candidates.tsv")
        settings.append(cands[tag])
    return settings

def as_row(tag, s, **updates):
    row = {k: s[k] for k in header if k in s}
    row.update(updates)
    row["tag"] = tag
    return [row[k] for k in header]

if phase == "1":
    rows = []
    weights = [
        ("w015_0425_0425", "0.15", "0.425", "0.425"),
        ("w020_040_040", "0.20", "0.40", "0.40"),
        ("w025_0375_0375", "0.25", "0.375", "0.375"),
        ("w030_035_035", "0.30", "0.35", "0.35"),
    ]
    for topk in ("8", "10", "12", "15"):
        for wtag, cf, image, text in weights:
            for alpha in ("0.10", "0.25"):
                tag = f"p1_topk{topk}_{wtag}_alpha{slug(alpha)}"
                rows.append([
                    tag, ckpt_epoch, "1.0", "0.01", "0.01", alpha,
                    "0.005", "0.2", "256", cf, image, text, topk, seed, dataset_seed,
                ])
    write(rows)
elif phase == "2":
    rows = []
    for i, s in enumerate(prior_settings("phase1_graph", 3), 1):
        rows.append(as_row(f"p2_s{i}_clw0", s, rec_neighbor_cl_weight="0"))
        rows.append(as_row(f"p2_s{i}_clw0p003", s, rec_neighbor_cl_weight="0.003"))
        rows.append(as_row(f"p2_s{i}_clw0p01", s, rec_neighbor_cl_weight="0.01"))
        rows.append(as_row(
            f"p2_s{i}_clw0p005_temp0p1",
            s,
            rec_neighbor_cl_weight="0.005",
            rec_neighbor_cl_temp="0.1",
        ))
    write(rows)
elif phase == "3":
    rows = []
    for i, s in enumerate(prior_settings("phase2_cl", 2), 1):
        rows.append(as_row(f"p3_s{i}_lrrec0p005", s, lr_rec="0.005"))
        rows.append(as_row(f"p3_s{i}_lrrec0p02", s, lr_rec="0.02"))
        rows.append(as_row(f"p3_s{i}_reg0p005", s, reg_coeff="0.005"))
        rows.append(as_row(f"p3_s{i}_reg0p015", s, reg_coeff="0.015"))
        rows.append(as_row(f"p3_s{i}_mbpr1p5", s, modality_bpr_coeff="1.5"))
    write(rows)
elif phase == "4":
    all_rows = []
    for phase_dir in ("phase1_graph", "phase2_cl", "phase3_train"):
        summary = base / phase_dir / "summary_val.tsv"
        if summary.exists():
            all_rows.extend((phase_dir, row) for row in sorted_summary(summary))
    if not all_rows:
        raise SystemExit("no phase summaries found for confirmation")
    cands_by_phase = {
        phase_dir: read_candidates(base / phase_dir / "candidates.tsv")
        for phase_dir in ("phase1_graph", "phase2_cl", "phase3_train")
        if (base / phase_dir / "candidates.tsv").exists()
    }
    all_rows.sort(
        key=lambda pr: (float(pr[1].get("recall20", "0") or 0), float(pr[1].get("ndcg20", "0") or 0)),
        reverse=True,
    )
    rows = []
    seen = set()
    for phase_dir, row in all_rows:
        src = cands_by_phase[phase_dir][row["tag"]]
        key = tuple(src[k] for k in header[1:13])
        if key in seen:
            continue
        seen.add(key)
        rank = len(rows) + 1
        rows.append(as_row(f"p4_top{rank}_confirm_seed2023", src, seed="2023", dataset_seed="0"))
        if len(rows) >= 2:
            break
    existing_confirm = []
    confirm_summary = base / "phase4_confirm" / "summary_val.tsv"
    if confirm_summary.exists():
        existing_confirm = sorted_summary(confirm_summary)
    add_seed1 = any(float(r.get("recall20", "0") or 0) > baseline for r in existing_confirm[:2])
    if add_seed1:
        seed1_rows = []
        for r in rows[:2]:
            vals = dict(zip(header, r))
            seed1_rows.append(as_row(vals["tag"].replace("seed2023", "seed1"), vals, seed="1", dataset_seed="0"))
        rows.extend(seed1_rows)
    write(rows)
else:
    raise SystemExit(f"unsupported phase: {phase}")
PY
}

summarize_phase() {
  python3 - "${LOG_DIR}" "${CANDIDATES_FILE}" "${SUMMARY_FILE}" "${SUMMARY_VAL_FILE}" "${SUMMARY_TOPK_FILE}" <<'PY'
import csv
import os
import re
import sys
from pathlib import Path

log_dir, candidates_file, summary_file, summary_val_file, summary_topk_file = map(Path, sys.argv[1:])

with candidates_file.open(newline="") as f:
    candidates = {row["tag"]: row for row in csv.DictReader(f, delimiter="\t")}

def tag_from_log(path):
    name = path.name
    for tag in candidates:
        if f"_{tag}_" in name or name.startswith(f"{tag}."):
            return tag
    m = re.search(r"stage2_clothing_mr[^_]+_(.+?)_", name)
    return m.group(1) if m else path.stem

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
    patterns = []
    if epoch >= 0:
        patterns.append(rf"epoch\s*=\s*{epoch}\s+hr@20\s*=\s*([0-9.]+),\s*recall@20\s*=\s*([0-9.]+),\s*ndcg@20\s*=\s*([0-9.]+)")
    patterns.append(r"hr@20\s*=\s*([0-9.]+),\s*recall@20\s*=\s*([0-9.]+),\s*ndcg@20\s*=\s*([0-9.]+)")
    for pattern in patterns:
        vals = re.findall(pattern, text)
        if vals:
            return tuple(map(float, vals[-1]))
    vals = re.findall(r"val_recall@20\s*=\s*([0-9.]+),\s*val_ndcg@20\s*=\s*([0-9.]+)", text)
    if vals:
        rec, ndcg = map(float, vals[-1])
        return (0.0, rec, ndcg)
    return None

rows_final = []
rows_val = []
rows_topk = []
for path in sorted(log_dir.glob("*.log")):
    text = path.read_text(errors="ignore")
    if "final strict test hr@20" not in text:
        continue
    tag = tag_from_log(path)
    epoch = best_epoch(text)
    f20 = final_at_k(text, 20)
    v20 = val_at_20(text, epoch)
    if f20:
        _, rec, ndcg = f20
        rows_final.append({
            "tag": tag, "recall20": f"{rec:.5f}", "ndcg20": f"{ndcg:.5f}",
            "best_epoch": str(epoch), "log": path.name,
        })
    if v20:
        _, rec, ndcg = v20
        rows_val.append({
            "tag": tag, "recall20": f"{rec:.5f}", "ndcg20": f"{ndcg:.5f}",
            "best_epoch": str(epoch), "log": path.name,
        })
    top = {"tag": tag, "best_epoch": str(epoch), "log": path.name}
    for k in (10, 20, 30, 40, 50):
        vals = final_at_k(text, k)
        if vals:
            hr, rec, ndcg = vals
            top[f"hr@{k}"] = f"{hr:.5f}"
            top[f"recall@{k}"] = f"{rec:.5f}"
            top[f"ndcg@{k}"] = f"{ndcg:.5f}"
    if any(k.startswith("recall@") for k in top):
        rows_topk.append(top)

def sort_key(row):
    return (float(row.get("recall20", 0.0)), float(row.get("ndcg20", 0.0)))

rows_final.sort(key=sort_key, reverse=True)
rows_val.sort(key=sort_key, reverse=True)

for path, fields, rows in (
    (summary_file, ["tag", "recall20", "ndcg20", "best_epoch", "log"], rows_final),
    (summary_val_file, ["tag", "recall20", "ndcg20", "best_epoch", "log"], rows_val),
    (summary_topk_file, [
        "tag", "hr@10", "recall@10", "ndcg@10", "hr@20", "recall@20", "ndcg@20",
        "hr@30", "recall@30", "ndcg@30", "hr@40", "recall@40", "ndcg@40",
        "hr@50", "recall@50", "ndcg@50", "best_epoch", "log",
    ], rows_topk),
):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
PY
}

suffix_for_tag() {
  local tag="$1"
  printf "stage2_clothing_mr%s_%s_%s" "${MR_TAG}" "${tag}" "${RUN_TAG}"
}

build_command() {
  local gpu="$1"
  local tag="$2"
  local ckpt_epoch="$3"
  local mbpr="$4"
  local lr_rec="$5"
  local reg="$6"
  local modal_alpha="$7"
  local cl_weight="$8"
  local cl_temp="$9"
  local cl_bank="${10}"
  local cf_w="${11}"
  local image_w="${12}"
  local text_w="${13}"
  local topk="${14}"
  local seed="${15}"
  local dataset_seed="${16}"
  local suffix imputer_ckpt

  suffix="$(suffix_for_tag "${tag}")"
  imputer_ckpt="$(ckpt_path_for_epoch "${ckpt_epoch}")"

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
    --modality_bpr_coeff "${mbpr}" \
    --reg_coeff "${reg}" \
    --evaluation_protocol strict \
    --selection_mode val \
    --strict_probe_test_interval 0 \
    --recommendation_selection_metric recall \
    --recommendation_selection_topk 20 \
    --rec_neighbor_cl_weight "${cl_weight}" \
    --rec_neighbor_cl_temp "${cl_temp}" \
    --rec_neighbor_cl_bank_size "${cl_bank}" \
    --item_graph_kind fused_completed \
    --item_graph_topk "${topk}" \
    --item_graph_norm rw \
    --item_graph_cf_weight "${cf_w}" \
    --item_graph_image_weight "${image_w}" \
    --item_graph_text_weight "${text_w}" \
    --item_graph_audio_weight 0.0 \
    --item_graph_modal_alpha "${modal_alpha}" \
    --item_graph_modal_layers 1 \
    --item_graph_modal_target all \
    --tensorboard 0 \
    --save 1 \
    --topk "[10, 20, 30, 40, 50]"
  printf "\n"
}

run_candidate() {
  local gpu="$1"
  local tag="$2"
  local ckpt_epoch="$3"
  local mbpr="$4"
  local lr_rec="$5"
  local reg="$6"
  local modal_alpha="$7"
  local cl_weight="$8"
  local cl_temp="$9"
  local cl_bank="${10}"
  local cf_w="${11}"
  local image_w="${12}"
  local text_w="${13}"
  local topk="${14}"
  local seed="${15}"
  local dataset_seed="${16}"
  local suffix log_path imputer_ckpt

  suffix="$(suffix_for_tag "${tag}")"
  log_path="${LOG_DIR}/${suffix}.log"
  imputer_ckpt="$(ckpt_path_for_epoch "${ckpt_epoch}")"

  build_command "${gpu}" "${tag}" "${ckpt_epoch}" "${mbpr}" "${lr_rec}" "${reg}" \
    "${modal_alpha}" "${cl_weight}" "${cl_temp}" "${cl_bank}" \
    "${cf_w}" "${image_w}" "${text_w}" "${topk}" "${seed}" "${dataset_seed}" \
    > "${log_path}.cmd"

  if [[ ! -f "${imputer_ckpt}" && "${DRY_RUN}" != "1" ]]; then
    log "skip ${tag}: missing imputer checkpoint ${imputer_ckpt}"
    return 0
  fi

  log "launch ${tag} on GPU ${gpu} seed=${seed} dseed=${dataset_seed}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    cat "${log_path}.cmd"
    return 0
  fi

  bash -lc "$(cat "${log_path}.cmd")" > "${log_path}" 2>&1
  log "done ${tag} on GPU ${gpu}"
}

claim_next_candidate() {
  (
    flock -x 9
    local idx total line tag ckpt_epoch mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank cf_w image_w text_w topk seed dataset_seed suffix log_path
    idx="$(cat "${NEXT_FILE}")"
    total="${#CANDIDATES[@]}"
    while (( idx < total )); do
      line="${CANDIDATES[$idx]}"
      IFS=$'\t' read -r tag ckpt_epoch mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank cf_w image_w text_w topk seed dataset_seed <<<"${line}"
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
  local worker_idx="$1"
  local gpu="$2"
  local cand tag ckpt_epoch mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank cf_w image_w text_w topk seed dataset_seed
  log "worker=${worker_idx} gpu=${gpu} started"
  while true; do
    if ! cand="$(claim_next_candidate)"; then
      log "worker=${worker_idx} gpu=${gpu} no candidates left"
      return 0
    fi
    IFS=$'\t' read -r tag ckpt_epoch mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank cf_w image_w text_w topk seed dataset_seed <<<"${cand}"
    run_candidate "${gpu}" "${tag}" "${ckpt_epoch}" "${mbpr}" "${lr_rec}" "${reg}" \
      "${modal_alpha}" "${cl_weight}" "${cl_temp}" "${cl_bank}" \
      "${cf_w}" "${image_w}" "${text_w}" "${topk}" "${seed}" "${dataset_seed}"
    if [[ "${DRY_RUN}" != "1" ]]; then
      summarize_phase
      sleep "${POLL_SECONDS}"
    fi
  done
}

generate_candidates
if [[ "${PHASE}" == "4" && -f "${SUMMARY_VAL_FILE}" && "${DRY_RUN}" != "1" ]]; then
  generate_candidates
fi

if [[ ! -f "${NEXT_FILE}" ]]; then
  echo 0 > "${NEXT_FILE}"
fi

mapfile -t CANDIDATES < <(tail -n +2 "${CANDIDATES_FILE}")
if (( CANDIDATE_LIMIT > 0 && CANDIDATE_LIMIT < ${#CANDIDATES[@]} )); then
  CANDIDATES=("${CANDIDATES[@]:0:${CANDIDATE_LIMIT}}")
fi

read -r -a GPU_LIST <<<"${GPUS}"
WORKER_GPUS=()
for gpu in "${GPU_LIST[@]}"; do
  for ((slot = 0; slot < TASKS_PER_GPU; slot++)); do
    WORKER_GPUS+=("${gpu}")
  done
done

if (( ${#WORKER_GPUS[@]} < 1 )); then
  echo "no GPU workers configured" >&2
  exit 1
fi
if (( ${#WORKER_GPUS[@]} > MAX_PARALLEL )); then
  WORKER_GPUS=("${WORKER_GPUS[@]:0:${MAX_PARALLEL}}")
fi

log_run_header() {
  log "run_tag=${RUN_TAG}"
  log "phase=${PHASE} phase_dir=${PHASE_DIR}"
  log "config=${CONFIG}"
  log "missing_rate=${MISSING_RATE} eval_missing_rate=${EVAL_MISSING_RATE} batch_size=${BATCH_SIZE}"
  log "stage12_dir=${STAGE12_DIR:-<not found>}"
  log "stage12_prefix=${STAGE12_PREFIX}"
  log "ckpt_epoch=${CKPT_EPOCH} ckpt=$(ckpt_path_for_epoch "${CKPT_EPOCH}")"
  log "gpus=${GPUS} tasks_per_gpu=${TASKS_PER_GPU} max_parallel=${MAX_PARALLEL} workers=${WORKER_GPUS[*]}"
  log "candidate_count=${#CANDIDATES[@]} dry_run=${DRY_RUN}"
}

run_workers() {
  local worker_idx
  for ((worker_idx = 0; worker_idx < ${#WORKER_GPUS[@]}; worker_idx++)); do
    worker_loop "${worker_idx}" "${WORKER_GPUS[$worker_idx]}" &
  done
  wait
}

log_run_header
run_workers

if [[ "${DRY_RUN}" != "1" ]]; then
  summarize_phase
  if [[ "${PHASE}" == "4" ]]; then
    old_count="${#CANDIDATES[@]}"
    generate_candidates
    mapfile -t CANDIDATES < <(tail -n +2 "${CANDIDATES_FILE}")
    if (( ${#CANDIDATES[@]} > old_count )); then
      log "phase4 added seed=1 confirmation candidates after baseline check"
      log "candidate_count=${#CANDIDATES[@]} dry_run=${DRY_RUN}"
      run_workers
      summarize_phase
    fi
  fi
fi
log "phase ${PHASE} done"
