#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="sports"
EXP_MODE="mm"
MISSING_RATE="${MISSING_RATE:-0.3}"
MR_TAG="${MISSING_RATE//./p}"
RUN_TAG="${RUN_TAG:-sports_multigraph_beststage12_mr${MR_TAG}_seed${SEED:-1}_$(date +%Y%m%d_%H%M%S)}"

OUT_DIR="${OUT_DIR:-exp_report/sports/multigraph_beststage12_search/${RUN_TAG}}"
LOG_DIR="${OUT_DIR}/logs"
STATE_DIR="${OUT_DIR}/state"
mkdir -p "${LOG_DIR}" "${STATE_DIR}"

NEXT_FILE="${STATE_DIR}/next_index"
LOCK_FILE="${STATE_DIR}/queue.lock"
SUMMARY_FILE="${OUT_DIR}/summary.tsv"
LAUNCHER_LOG="${OUT_DIR}/launcher.log"
if [[ ! -f "${NEXT_FILE}" ]]; then
  echo 0 > "${NEXT_FILE}"
fi

GPUS="${GPUS:-0 1}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
MEM_FREE_THRESHOLD="${MEM_FREE_THRESHOLD:-1000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
DRY_RUN="${DRY_RUN:-0}"
INCLUDE_SINGLE_GRAPH_ABLATIONS="${INCLUDE_SINGLE_GRAPH_ABLATIONS:-0}"

CONFIG="${CONFIG:-configs/sports/stage2_decoder_mm.yaml}"
SEED="${SEED:-1}"
DATASET_SEED="${DATASET_SEED:-0}"
EPOCHS="${EPOCHS:-1000000}"
EARLY_STOP="${EARLY_STOP:-20}"

log() {
  echo "[$(date -Is)] $*" | tee -a "${LAUNCHER_LOG}"
}

discover_stage12_dir() {
  find exp_report/sports -maxdepth 1 -type d \
    -name "stage1_2_sports*beststyle_nocl*" \
    -printf "%T@ %p\n" 2>/dev/null \
    | sort -nr \
    | awk 'NR == 1 { $1=""; sub(/^ /, ""); print }'
}

STAGE12_DIR="${STAGE12_DIR:-$(discover_stage12_dir)}"
if [[ -n "${STAGE12_DIR}" ]]; then
  STAGE12_PREFIX="${STAGE12_PREFIX:-$(basename "${STAGE12_DIR}")}"
else
  STAGE12_PREFIX="${STAGE12_PREFIX:-stage1_2_sports_beststyle_nocl_<RUN_TAG>}"
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "missing config: ${CONFIG}" >&2
  exit 1
fi

if [[ "${DRY_RUN}" != "1" && ( -z "${STAGE12_DIR}" || ! -d "${STAGE12_DIR}" ) ]]; then
  echo "missing STAGE12_DIR; set STAGE12_DIR and STAGE12_PREFIX or finish stage1.2 first" >&2
  exit 1
fi

discover_stage12_best_epoch() {
  local log_file
  log_file="$(find "${STAGE12_DIR}/log" -maxdepth 1 -type f -name '*.log' 2>/dev/null | sort | tail -1)"
  if [[ -z "${log_file}" ]]; then
    return 1
  fi
  python3 - "${log_file}" <<'PY'
import re
import sys

text = open(sys.argv[1], "r", errors="ignore").read()
matches = re.findall(r"best epoch\s+([0-9]+)", text)
if matches:
    print(matches[-1])
    raise SystemExit(0)

loss_rows = re.findall(r"TRAIN:stage = imputer_backprop, epoch = ([0-9]+)/[0-9]+ loss_s1 = ([\-0-9.]+)", text)
if not loss_rows:
    raise SystemExit(1)
best_epoch, _ = min(((int(epoch), float(loss)) for epoch, loss in loss_rows), key=lambda row: row[1])
print(best_epoch)
PY
}

STAGE12_BEST_EPOCH="${STAGE12_BEST_EPOCH:-$(discover_stage12_best_epoch)}"
if [[ -z "${STAGE12_BEST_EPOCH}" ]]; then
  echo "missing STAGE12_BEST_EPOCH; could not parse ${STAGE12_DIR}/log/*.log" >&2
  exit 1
fi

# tag batch_size mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank cf image text topk
#
# Current mainline policy:
# - MAINLINE_CANDIDATES always use a multi-source fused graph: at least CF,
#   image, and text weights are positive.
# - SINGLE_GRAPH_ABLATIONS are a side branch and are included only when
#   INCLUDE_SINGLE_GRAPH_ABLATIONS=1.
MAINLINE_CANDIDATES=(
  # Center: use the best Stage 1.2 checkpoint by Stage 1.2 completion metric.
  "center_topk10 256 1.0 0.001 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 10"

  # Graph neighborhood: topk and source weights, keeping all graph sources positive.
  "topk5 256 1.0 0.001 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 5"
  "topk8 256 1.0 0.001 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 8"
  "topk12 256 1.0 0.001 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 12"
  "topk15 256 1.0 0.001 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 15"
  "topk20 256 1.0 0.001 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 20"
  "graph_modal040_topk10 256 1.0 0.001 0.0001 0.25 0.005 0.2 256 0.2 0.4 0.4 10"
  "graph_modal0375_topk10 256 1.0 0.001 0.0001 0.25 0.005 0.2 256 0.25 0.375 0.375 10"
  "graph_cf040_topk10 256 1.0 0.001 0.0001 0.25 0.005 0.2 256 0.4 0.3 0.3 10"
  "graph_image045_text035_topk10 256 1.0 0.001 0.0001 0.25 0.005 0.2 256 0.2 0.45 0.35 10"
  "graph_image035_text045_topk10 256 1.0 0.001 0.0001 0.25 0.005 0.2 256 0.2 0.35 0.45 10"

  # Modal residual strength around the current mainline alpha=0.25.
  "malpha0p10 256 1.0 0.001 0.0001 0.10 0.005 0.2 256 0.3 0.35 0.35 10"
  "malpha0p20 256 1.0 0.001 0.0001 0.20 0.005 0.2 256 0.3 0.35 0.35 10"
  "malpha0p35 256 1.0 0.001 0.0001 0.35 0.005 0.2 256 0.3 0.35 0.35 10"
  "malpha0p50 256 1.0 0.001 0.0001 0.50 0.005 0.2 256 0.3 0.35 0.35 10"

  # Loss and optimizer neighborhood around the verified center.
  "mbpr0p5 256 0.5 0.001 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 10"
  "mbpr1p5 256 1.5 0.001 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 10"
  "mbpr2p0 256 2.0 0.001 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 10"
  "lr0p0005 256 1.0 0.0005 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 10"
  "lr0p002 256 1.0 0.002 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 10"
  "reg5em05 256 1.0 0.001 0.00005 0.25 0.005 0.2 256 0.3 0.35 0.35 10"
  "reg3em04 256 1.0 0.001 0.0003 0.25 0.005 0.2 256 0.3 0.35 0.35 10"
  "batch128 128 1.0 0.001 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 10"
  "batch512 512 1.0 0.001 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 10"

  # Retained post-GCN true-missing InfoNCE objective, searched locally.
  "reccl0 256 1.0 0.001 0.0001 0.25 0.0 0.2 256 0.3 0.35 0.35 10"
  "reccl0p003 256 1.0 0.001 0.0001 0.25 0.003 0.2 256 0.3 0.35 0.35 10"
  "reccl0p010 256 1.0 0.001 0.0001 0.25 0.010 0.2 256 0.3 0.35 0.35 10"
  "reccl_temp0p15 256 1.0 0.001 0.0001 0.25 0.005 0.15 256 0.3 0.35 0.35 10"
  "reccl_temp0p25 256 1.0 0.001 0.0001 0.25 0.005 0.25 256 0.3 0.35 0.35 10"
  "reccl_bank128 256 1.0 0.001 0.0001 0.25 0.005 0.2 128 0.3 0.35 0.35 10"
  "reccl_bank512 256 1.0 0.001 0.0001 0.25 0.005 0.2 512 0.3 0.35 0.35 10"

  # Compact interaction checks around the strongest graph settings.
  "topk8_modal040 256 1.0 0.001 0.0001 0.25 0.005 0.2 256 0.2 0.4 0.4 8"
  "topk12_modal040 256 1.0 0.001 0.0001 0.25 0.005 0.2 256 0.2 0.4 0.4 12"
  "modal040_malpha0p20 256 1.0 0.001 0.0001 0.20 0.005 0.2 256 0.2 0.4 0.4 10"
  "modal040_malpha0p35 256 1.0 0.001 0.0001 0.35 0.005 0.2 256 0.2 0.4 0.4 10"
)

SINGLE_GRAPH_ABLATIONS=(
  "single_cf_topk10 256 1.0 0.001 0.0001 0.25 0.005 0.2 256 1.0 0.0 0.0 10"
  "single_image_topk10 256 1.0 0.001 0.0001 0.25 0.005 0.2 256 0.0 1.0 0.0 10"
  "single_text_topk10 256 1.0 0.001 0.0001 0.25 0.005 0.2 256 0.0 0.0 1.0 10"
)

CANDIDATES=("${MAINLINE_CANDIDATES[@]}")
if [[ "${INCLUDE_SINGLE_GRAPH_ABLATIONS}" == "1" ]]; then
  CANDIDATES+=("${SINGLE_GRAPH_ABLATIONS[@]}")
fi

is_gpu_free() {
  local gpu="$1"
  local used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')"
  [[ "${used}" =~ ^[0-9]+$ ]] && (( used < MEM_FREE_THRESHOLD ))
}

ckpt_path_for_epoch() {
  printf "%s/ckpt/%s_imputer_backprop_50_epoch%s.pth" "${STAGE12_DIR}" "${STAGE12_PREFIX}" "${STAGE12_BEST_EPOCH}"
}

suffix_for_tag() {
  local tag="$1"
  printf "stage2_sports_mr%s_%s_%s" "${MR_TAG}" "${tag}" "${RUN_TAG}"
}

summarize_results() {
  python3 - "${LOG_DIR}" <<'PY'
import glob
import os
import re
import sys

rows = []
for path in glob.glob(os.path.join(sys.argv[1], "*.log")):
    text = open(path, "r", errors="ignore").read()
    vals = re.findall(
        r"final strict test hr@20 = ([0-9.]+), recall@20 = ([0-9.]+), ndcg@20 = ([0-9.]+)",
        text,
    )
    best = re.findall(r"best epoch ([0-9]+)", text)
    if vals:
        _, rec, ndcg = map(float, vals[-1])
        rows.append((rec, ndcg, int(best[-1]) if best else -1, os.path.basename(path)))
print("recall20\tndcg20\tbest_epoch\tlog")
for rec, ndcg, best, name in sorted(rows, reverse=True):
    print(f"{rec:.5f}\t{ndcg:.5f}\t{best}\t{name}")
PY
}

build_command() {
  local gpu="$1"
  local tag="$2"
  local batch_size="$3"
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
  local suffix
  local imputer_ckpt

  suffix="$(suffix_for_tag "${tag}")"
  imputer_ckpt="$(ckpt_path_for_epoch)"

  printf "%q " .venv/bin/python -u main.py \
    --config "${CONFIG}" \
    --device_id "${gpu}" \
    --dataset "${DATASET}" \
    --exp_mode "${EXP_MODE}" \
    --train_stage recommender \
    --missing_rate "${MISSING_RATE}" \
    --seed "${SEED}" \
    --dataset_seed "${DATASET_SEED}" \
    --missing_mask_protocol i3 \
    --imputer_ckpt "${imputer_ckpt}" \
    --suffix "${suffix}" \
    --epoch "${EPOCHS}" \
    --early_stop "${EARLY_STOP}" \
    --eva_interval 1 \
    --batch_size "${batch_size}" \
    --lr "${lr_rec}" \
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
  local batch_size="$3"
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
  local suffix
  local log_path
  local imputer_ckpt

  suffix="$(suffix_for_tag "${tag}")"
  log_path="${LOG_DIR}/${suffix}.log"
  imputer_ckpt="$(ckpt_path_for_epoch)"

  if [[ ! -f "${imputer_ckpt}" ]]; then
    log "skip ${tag}: missing imputer checkpoint ${imputer_ckpt}"
    return 0
  fi

  log "launch ${tag} on GPU ${gpu} stage12_best_epoch=${STAGE12_BEST_EPOCH}"
  build_command "${gpu}" "${tag}" "${batch_size}" "${mbpr}" "${lr_rec}" "${reg}" \
    "${modal_alpha}" "${cl_weight}" "${cl_temp}" "${cl_bank}" \
    "${cf_w}" "${image_w}" "${text_w}" "${topk}" \
    > "${log_path}.cmd"

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
    local idx tag batch_size mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank cf_w image_w text_w topk suffix log_path
    idx="$(cat "${NEXT_FILE}")"
    while (( idx < ${#CANDIDATES[@]} )); do
      read -r tag batch_size mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank cf_w image_w text_w topk <<<"${CANDIDATES[$idx]}"
      suffix="$(suffix_for_tag "${tag}")"
      log_path="${LOG_DIR}/${suffix}.log"
      if [[ -f "${log_path}" ]] && grep -q "final strict test hr@20" "${log_path}"; then
        echo "[$(date -Is)] skip finished ${tag}" >> "${LAUNCHER_LOG}"
        idx=$((idx + 1))
        echo "${idx}" > "${NEXT_FILE}"
        continue
      fi
      echo $((idx + 1)) > "${NEXT_FILE}"
      printf "%s\n" "${CANDIDATES[$idx]}"
      return 0
    done
    return 1
  ) 9>"${LOCK_FILE}"
}

worker_loop() {
  local worker_idx="$1"
  local gpu="$2"
  local cand tag batch_size mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank cf_w image_w text_w topk

  while true; do
    while [[ "${DRY_RUN}" != "1" ]] && ! is_gpu_free "${gpu}"; do
      log "worker=${worker_idx} gpu=${gpu} waiting for free GPU"
      sleep "${POLL_SECONDS}"
    done

    if ! cand="$(claim_next_candidate)"; then
      log "worker=${worker_idx} gpu=${gpu} no candidates left"
      return 0
    fi

    read -r tag batch_size mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank cf_w image_w text_w topk <<<"${cand}"
    run_candidate "${gpu}" "${tag}" "${batch_size}" "${mbpr}" "${lr_rec}" "${reg}" \
      "${modal_alpha}" "${cl_weight}" "${cl_temp}" "${cl_bank}" \
      "${cf_w}" "${image_w}" "${text_w}" "${topk}"
    if [[ "${DRY_RUN}" != "1" ]]; then
      summarize_results > "${SUMMARY_FILE}"
    fi
  done
}

log "run_tag=${RUN_TAG}"
log "config=${CONFIG}"
log "missing_rate=${MISSING_RATE}"
log "seed=${SEED}"
log "stage12_dir=${STAGE12_DIR:-<not found>}"
log "stage12_prefix=${STAGE12_PREFIX}"
log "stage12_best_epoch=${STAGE12_BEST_EPOCH}"
log "dry_run=${DRY_RUN}"
log "mainline_candidates=${#MAINLINE_CANDIDATES[@]}"
log "include_single_graph_ablations=${INCLUDE_SINGLE_GRAPH_ABLATIONS}"
log "total_candidates=${#CANDIDATES[@]}"

read -r -a GPU_LIST <<<"${GPUS}"
WORKER_COUNT="${#GPU_LIST[@]}"
if (( WORKER_COUNT > MAX_PARALLEL )); then
  WORKER_COUNT="${MAX_PARALLEL}"
fi
if (( WORKER_COUNT < 1 )); then
  echo "no GPUs configured" >&2
  exit 1
fi

for ((worker_idx = 0; worker_idx < WORKER_COUNT; worker_idx++)); do
  worker_loop "${worker_idx}" "${GPU_LIST[$worker_idx]}" &
done
wait

if [[ "${DRY_RUN}" != "1" ]]; then
  summarize_results > "${SUMMARY_FILE}"
fi
log "all done"
