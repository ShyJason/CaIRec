#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="clothing"
EXP_MODE="mm"
MISSING_RATE="${MISSING_RATE:-0.3}"
MR_TAG="${MISSING_RATE//./p}"
RUN_TAG="${RUN_TAG:-clothing_mainline_hparam_mr${MR_TAG}_$(date +%Y%m%d_%H%M%S)}"

OUT_DIR="${OUT_DIR:-exp_report/clothing/mainline_hparam_search/${RUN_TAG}}"
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

CONFIG="${CONFIG:-configs/clothing/stage2_decoder_mm_itemgraph_completed.yaml}"
SEED="${SEED:-2023}"
DATASET_SEED="${DATASET_SEED:-0}"
EPOCHS="${EPOCHS:-200}"
EARLY_STOP="${EARLY_STOP:-10000}"

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

# tag ckpt_epoch mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank cf image text topk
CANDIDATES=(
  # Baseline and checkpoint epoch scan.
  "baseline_e47 47 1.0 0.01 0.01 0.25 0.005 0.2 256 0.3 0.35 0.35 20"
  "ckpt_e40 40 1.0 0.01 0.01 0.25 0.005 0.2 256 0.3 0.35 0.35 20"
  "ckpt_e45 45 1.0 0.01 0.01 0.25 0.005 0.2 256 0.3 0.35 0.35 20"
  "ckpt_e49 49 1.0 0.01 0.01 0.25 0.005 0.2 256 0.3 0.35 0.35 20"

  # Stage2 core training parameters.
  "mbpr0p5 47 0.5 0.01 0.01 0.25 0.005 0.2 256 0.3 0.35 0.35 20"
  "mbpr1p5 47 1.5 0.01 0.01 0.25 0.005 0.2 256 0.3 0.35 0.35 20"
  "mbpr2p0 47 2.0 0.01 0.01 0.25 0.005 0.2 256 0.3 0.35 0.35 20"
  "lrrec0p005 47 1.0 0.005 0.01 0.25 0.005 0.2 256 0.3 0.35 0.35 20"
  "lrrec0p02 47 1.0 0.02 0.01 0.25 0.005 0.2 256 0.3 0.35 0.35 20"
  "reg0p005 47 1.0 0.01 0.005 0.25 0.005 0.2 256 0.3 0.35 0.35 20"
  "reg0p015 47 1.0 0.01 0.015 0.25 0.005 0.2 256 0.3 0.35 0.35 20"
  "reg0p02 47 1.0 0.01 0.02 0.25 0.005 0.2 256 0.3 0.35 0.35 20"

  # Item-item graph residual parameters.
  "malpha0p1 47 1.0 0.01 0.01 0.1 0.005 0.2 256 0.3 0.35 0.35 20"
  "malpha0p5 47 1.0 0.01 0.01 0.5 0.005 0.2 256 0.3 0.35 0.35 20"
  "malpha0p75 47 1.0 0.01 0.01 0.75 0.005 0.2 256 0.3 0.35 0.35 20"
  "topk10 47 1.0 0.01 0.01 0.25 0.005 0.2 256 0.3 0.35 0.35 10"
  "topk40 47 1.0 0.01 0.01 0.25 0.005 0.2 256 0.3 0.35 0.35 40"
  "graph_cf040 47 1.0 0.01 0.01 0.25 0.005 0.2 256 0.4 0.3 0.3 20"
  "graph_modal040 47 1.0 0.01 0.01 0.25 0.005 0.2 256 0.2 0.4 0.4 20"
  "graph_cf050 47 1.0 0.01 0.01 0.25 0.005 0.2 256 0.5 0.25 0.25 20"

  # True-missing post-GCN InfoNCE parameters.
  "clw0 47 1.0 0.01 0.01 0.25 0.0 0.2 256 0.3 0.35 0.35 20"
  "clw0p001 47 1.0 0.01 0.01 0.25 0.001 0.2 256 0.3 0.35 0.35 20"
  "clw0p003 47 1.0 0.01 0.01 0.25 0.003 0.2 256 0.3 0.35 0.35 20"
  "clw0p01 47 1.0 0.01 0.01 0.25 0.01 0.2 256 0.3 0.35 0.35 20"
  "cltemp0p1 47 1.0 0.01 0.01 0.25 0.005 0.1 256 0.3 0.35 0.35 20"
  "cltemp0p5 47 1.0 0.01 0.01 0.25 0.005 0.5 256 0.3 0.35 0.35 20"
  "clbank128 47 1.0 0.01 0.01 0.25 0.005 0.2 128 0.3 0.35 0.35 20"
  "clbank512 47 1.0 0.01 0.01 0.25 0.005 0.2 512 0.3 0.35 0.35 20"

  # Local combinations around the current best-style center.
  "mbpr1p5_malpha0p25_clw0p005 47 1.5 0.01 0.01 0.25 0.005 0.2 256 0.3 0.35 0.35 20"
  "mbpr1p0_lrrec0p02_reg0p01 47 1.0 0.02 0.01 0.25 0.005 0.2 256 0.3 0.35 0.35 20"
  "mbpr1p5_lrrec0p02_reg0p015 47 1.5 0.02 0.015 0.25 0.005 0.2 256 0.3 0.35 0.35 20"
  "mbpr1p0_malpha0p5_graphcf040 47 1.0 0.01 0.01 0.5 0.005 0.2 256 0.4 0.3 0.3 20"
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
  printf "stage2_clothing_mr%s_%s_%s" "${MR_TAG}" "${tag}" "${RUN_TAG}"
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
    --imputer_ckpt "${imputer_ckpt}" \
    --suffix "${suffix}" \
    --epoch "${EPOCHS}" \
    --early_stop "${EARLY_STOP}" \
    --eva_interval 1 \
    --batch_size 2048 \
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
  build_command "${gpu}" "${tag}" "${ckpt_epoch}" "${mbpr}" "${lr_rec}" "${reg}" \
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
    local idx tag ckpt_epoch mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank cf_w image_w text_w topk suffix log_path
    idx="$(cat "${NEXT_FILE}")"
    while (( idx < ${#CANDIDATES[@]} )); do
      read -r tag ckpt_epoch mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank cf_w image_w text_w topk <<<"${CANDIDATES[$idx]}"
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
  local cand tag ckpt_epoch mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank cf_w image_w text_w topk

  while true; do
    while [[ "${DRY_RUN}" != "1" ]] && ! is_gpu_free "${gpu}"; do
      log "worker=${worker_idx} gpu=${gpu} waiting for free GPU"
      sleep "${POLL_SECONDS}"
    done

    if ! cand="$(claim_next_candidate)"; then
      log "worker=${worker_idx} gpu=${gpu} no candidates left"
      return 0
    fi

    read -r tag ckpt_epoch mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank cf_w image_w text_w topk <<<"${cand}"
    run_candidate "${gpu}" "${tag}" "${ckpt_epoch}" "${mbpr}" "${lr_rec}" "${reg}" \
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
