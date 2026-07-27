#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="sports"
EXP_MODE="mm"
MISSING_RATE="${MISSING_RATE:-0.3}"
MR_TAG="${MISSING_RATE//./p}"
RUN_TAG="${RUN_TAG:-sports_mainline_hparam_mr${MR_TAG}_seed${SEED:-1}_$(date +%Y%m%d_%H%M%S)}"

OUT_DIR="${OUT_DIR:-exp_report/sports/mainline_hparam_search/${RUN_TAG}}"
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

CONFIG="${CONFIG:-configs/sports/stage2_decoder_mm.yaml}"
SEED="${SEED:-1}"
DATASET_SEED="${DATASET_SEED:-0}"
EPOCHS="${EPOCHS:-200}"
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

# tag ckpt_epoch batch_size mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank cf image text topk
CANDIDATES=(
  # Centers from previous Sports search, current Sports mainline, and Clothing topk10 transfer.
  "sports_prev_center 19 256 0.2 0.001 0.0001 0.5 0.01 0.25 256 0.3 0.35 0.35 20"
  "sports_current_mainline 19 256 0.2 0.001 0.0001 0.5 0.0 0.2 256 0.3 0.35 0.35 20"
  "clothing_topk10_transfer 19 256 1.0 0.001 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 10"

  # Stage1.2 checkpoint epoch sanity around the fixed-missing Sports checkpoint.
  "ckpt_e15 15 256 0.2 0.001 0.0001 0.5 0.01 0.25 256 0.3 0.35 0.35 20"
  "ckpt_e18 18 256 0.2 0.001 0.0001 0.5 0.01 0.25 256 0.3 0.35 0.35 20"
  "ckpt_e19 19 256 0.2 0.001 0.0001 0.5 0.01 0.25 256 0.3 0.35 0.35 20"

  # Previous Sports recommendation-side best neighborhood.
  "bs128 19 128 0.2 0.001 0.0001 0.5 0.01 0.25 256 0.3 0.35 0.35 20"
  "bs512 19 512 0.2 0.001 0.0001 0.5 0.01 0.25 256 0.3 0.35 0.35 20"
  "lr0p0005 19 256 0.2 0.0005 0.0001 0.5 0.01 0.25 256 0.3 0.35 0.35 20"
  "lr0p002 19 256 0.2 0.002 0.0001 0.5 0.01 0.25 256 0.3 0.35 0.35 20"
  "mbpr0 19 256 0.0 0.001 0.0001 0.5 0.01 0.25 256 0.3 0.35 0.35 20"
  "mbpr0p1 19 256 0.1 0.001 0.0001 0.5 0.01 0.25 256 0.3 0.35 0.35 20"
  "mbpr0p5 19 256 0.5 0.001 0.0001 0.5 0.01 0.25 256 0.3 0.35 0.35 20"
  "reg1em05 19 256 0.2 0.001 0.00001 0.5 0.01 0.25 256 0.3 0.35 0.35 20"
  "reg0p0003 19 256 0.2 0.001 0.0003 0.5 0.01 0.25 256 0.3 0.35 0.35 20"
  "reg0p001 19 256 0.2 0.001 0.001 0.5 0.01 0.25 256 0.3 0.35 0.35 20"

  # Rec-neighbor CL around previous Sports best and Clothing transfer.
  "clw0 19 256 0.2 0.001 0.0001 0.5 0.0 0.25 256 0.3 0.35 0.35 20"
  "clw0p003 19 256 0.2 0.001 0.0001 0.5 0.003 0.25 256 0.3 0.35 0.35 20"
  "clw0p005 19 256 0.2 0.001 0.0001 0.5 0.005 0.25 256 0.3 0.35 0.35 20"
  "clw0p015 19 256 0.2 0.001 0.0001 0.5 0.015 0.25 256 0.3 0.35 0.35 20"
  "cltemp0p1 19 256 0.2 0.001 0.0001 0.5 0.01 0.1 256 0.3 0.35 0.35 20"
  "cltemp0p15 19 256 0.2 0.001 0.0001 0.5 0.01 0.15 256 0.3 0.35 0.35 20"
  "cltemp0p2 19 256 0.2 0.001 0.0001 0.5 0.01 0.2 256 0.3 0.35 0.35 20"
  "clbank128 19 256 0.2 0.001 0.0001 0.5 0.01 0.25 128 0.3 0.35 0.35 20"
  "clbank512 19 256 0.2 0.001 0.0001 0.5 0.01 0.25 512 0.3 0.35 0.35 20"

  # Item-item graph parameters, informed by current mainline and Clothing topk10.
  "topk8 19 256 0.2 0.001 0.0001 0.5 0.01 0.25 256 0.3 0.35 0.35 8"
  "topk10 19 256 0.2 0.001 0.0001 0.5 0.01 0.25 256 0.3 0.35 0.35 10"
  "topk15 19 256 0.2 0.001 0.0001 0.5 0.01 0.25 256 0.3 0.35 0.35 15"
  "topk40 19 256 0.2 0.001 0.0001 0.5 0.01 0.25 256 0.3 0.35 0.35 40"
  "malpha0p1 19 256 0.2 0.001 0.0001 0.1 0.01 0.25 256 0.3 0.35 0.35 20"
  "malpha0p25 19 256 0.2 0.001 0.0001 0.25 0.01 0.25 256 0.3 0.35 0.35 20"
  "malpha0p75 19 256 0.2 0.001 0.0001 0.75 0.01 0.25 256 0.3 0.35 0.35 20"
  "graph_cf040 19 256 0.2 0.001 0.0001 0.5 0.01 0.25 256 0.4 0.3 0.3 20"
  "graph_modal040 19 256 0.2 0.001 0.0001 0.5 0.01 0.25 256 0.2 0.4 0.4 20"
  "graph_img060_txt030 19 256 0.2 0.001 0.0001 0.5 0.01 0.25 256 0.1 0.6 0.3 20"

  # Local combinations around the strongest priors.
  "topk10_cltemp0p1 19 256 0.2 0.001 0.0001 0.5 0.01 0.1 256 0.3 0.35 0.35 10"
  "topk10_malpha0p25 19 256 0.2 0.001 0.0001 0.25 0.01 0.25 256 0.3 0.35 0.35 10"
  "topk10_graph_modal040 19 256 0.2 0.001 0.0001 0.5 0.01 0.25 256 0.2 0.4 0.4 10"
  "mbpr0p5_topk10 19 256 0.5 0.001 0.0001 0.5 0.01 0.25 256 0.3 0.35 0.35 10"
  "lr0p002_topk10 19 256 0.2 0.002 0.0001 0.5 0.01 0.25 256 0.3 0.35 0.35 10"
  "clothing_transfer_cltemp0p1 19 256 1.0 0.001 0.0001 0.25 0.005 0.1 256 0.3 0.35 0.35 10"
)

is_gpu_free() {
  local gpu="$1"
  local used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')"
  [[ "${used}" =~ ^[0-9]+$ ]] && (( used < MEM_FREE_THRESHOLD ))
}

ckpt_path_for_epoch() {
  local ckpt_epoch="$1"
  printf "%s/ckpt/%s_imputer_backprop_50_epoch%s.pth" "${STAGE12_DIR}" "${STAGE12_PREFIX}" "${ckpt_epoch}"
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
  local ckpt_epoch="$3"
  local batch_size="$4"
  local mbpr="$5"
  local lr_rec="$6"
  local reg="$7"
  local modal_alpha="$8"
  local cl_weight="$9"
  local cl_temp="${10}"
  local cl_bank="${11}"
  local cf_w="${12}"
  local image_w="${13}"
  local text_w="${14}"
  local topk="${15}"
  local suffix
  local imputer_ckpt

  suffix="$(suffix_for_tag "${tag}")"
  imputer_ckpt="$(ckpt_path_for_epoch "${ckpt_epoch}")"

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
  local ckpt_epoch="$3"
  local batch_size="$4"
  local mbpr="$5"
  local lr_rec="$6"
  local reg="$7"
  local modal_alpha="$8"
  local cl_weight="$9"
  local cl_temp="${10}"
  local cl_bank="${11}"
  local cf_w="${12}"
  local image_w="${13}"
  local text_w="${14}"
  local topk="${15}"
  local suffix
  local log_path
  local imputer_ckpt

  suffix="$(suffix_for_tag "${tag}")"
  log_path="${LOG_DIR}/${suffix}.log"
  imputer_ckpt="$(ckpt_path_for_epoch "${ckpt_epoch}")"

  if [[ ! -f "${imputer_ckpt}" ]]; then
    log "skip ${tag}: missing imputer checkpoint ${imputer_ckpt}"
    return 0
  fi

  log "launch ${tag} on GPU ${gpu} ckpt_epoch=${ckpt_epoch}"
  build_command "${gpu}" "${tag}" "${ckpt_epoch}" "${batch_size}" "${mbpr}" "${lr_rec}" "${reg}" \
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
    local idx tag ckpt_epoch batch_size mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank cf_w image_w text_w topk suffix log_path
    idx="$(cat "${NEXT_FILE}")"
    while (( idx < ${#CANDIDATES[@]} )); do
      read -r tag ckpt_epoch batch_size mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank cf_w image_w text_w topk <<<"${CANDIDATES[$idx]}"
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
  local cand tag ckpt_epoch batch_size mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank cf_w image_w text_w topk

  while true; do
    while [[ "${DRY_RUN}" != "1" ]] && ! is_gpu_free "${gpu}"; do
      log "worker=${worker_idx} gpu=${gpu} waiting for free GPU"
      sleep "${POLL_SECONDS}"
    done

    if ! cand="$(claim_next_candidate)"; then
      log "worker=${worker_idx} gpu=${gpu} no candidates left"
      return 0
    fi

    read -r tag ckpt_epoch batch_size mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank cf_w image_w text_w topk <<<"${cand}"
    run_candidate "${gpu}" "${tag}" "${ckpt_epoch}" "${batch_size}" "${mbpr}" "${lr_rec}" "${reg}" \
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
log "dry_run=${DRY_RUN}"

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
