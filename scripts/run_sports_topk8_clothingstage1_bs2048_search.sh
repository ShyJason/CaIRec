#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="sports"
EXP_MODE="mm"
MISSING_RATE="${MISSING_RATE:-0.3}"
MR_TAG="${MISSING_RATE//./p}"
SEED="${SEED:-1}"
DATASET_SEED="${DATASET_SEED:-0}"
RUN_TAG="${RUN_TAG:-sports_topk8_clothingstage1_bs2048_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"

OUT_DIR="${OUT_DIR:-exp_report/sports/topk8_clothingstage1_bs2048_search/${RUN_TAG}}"
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

CONFIG="${CONFIG:-configs/sports/stage2_decoder_mm.yaml}"
IMPUTER_CKPT="${IMPUTER_CKPT:-exp_report/sports/stage1_2_sports_clothing_stage1style_mr0p3_seed1_20260627_004233/ckpt/stage1_2_sports_clothing_stage1style_mr0p3_seed1_20260627_004233_imputer_backprop_50_epoch49.pth}"
GPUS="${GPUS:-5 6}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-2}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
MEM_FREE_THRESHOLD="${MEM_FREE_THRESHOLD:-6000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
DRY_RUN="${DRY_RUN:-0}"
EPOCHS="${EPOCHS:-10000}"
EARLY_STOP="${EARLY_STOP:-20}"
BATCH_SIZE="${BATCH_SIZE:-2048}"

log() {
  echo "[$(date -Is)] $*" | tee -a "${LAUNCHER_LOG}"
}

if [[ ! -f "${CONFIG}" ]]; then
  echo "missing config: ${CONFIG}" >&2
  exit 1
fi
if [[ "${DRY_RUN}" != "1" && ! -f "${IMPUTER_CKPT}" ]]; then
  echo "missing imputer checkpoint: ${IMPUTER_CKPT}" >&2
  exit 1
fi

# tag mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank cf image text topk
CANDIDATES=(
  # Center and learning-rate scaling for the larger batch.
  "center_lr0p001 1.0 0.001 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 8"
  "lr0p0015 1.0 0.0015 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 8"
  "lr0p002 1.0 0.002 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 8"
  "lr0p003 1.0 0.003 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 8"
  "lr0p004 1.0 0.004 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 8"
  "lr0p006 1.0 0.006 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 8"

  # Regularization and modality-BPR around the center.
  "reg5em05 1.0 0.002 0.00005 0.25 0.005 0.2 256 0.3 0.35 0.35 8"
  "reg2em04 1.0 0.002 0.0002 0.25 0.005 0.2 256 0.3 0.35 0.35 8"
  "reg3em04 1.0 0.002 0.0003 0.25 0.005 0.2 256 0.3 0.35 0.35 8"
  "reg1em03 1.0 0.002 0.001 0.25 0.005 0.2 256 0.3 0.35 0.35 8"
  "mbpr0p5 0.5 0.002 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 8"
  "mbpr1p5 1.5 0.002 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 8"
  "mbpr2p0 2.0 0.002 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 8"

  # True-missing GCN InfoNCE strength, temperature, and bank size.
  "reccl0 1.0 0.002 0.0001 0.25 0.0 0.2 256 0.3 0.35 0.35 8"
  "reccl0p003 1.0 0.002 0.0001 0.25 0.003 0.2 256 0.3 0.35 0.35 8"
  "reccl0p010 1.0 0.002 0.0001 0.25 0.010 0.2 256 0.3 0.35 0.35 8"
  "reccl0p015 1.0 0.002 0.0001 0.25 0.015 0.2 256 0.3 0.35 0.35 8"
  "reccl_temp0p15 1.0 0.002 0.0001 0.25 0.005 0.15 256 0.3 0.35 0.35 8"
  "reccl_temp0p25 1.0 0.002 0.0001 0.25 0.005 0.25 256 0.3 0.35 0.35 8"
  "reccl_bank128 1.0 0.002 0.0001 0.25 0.005 0.2 128 0.3 0.35 0.35 8"
  "reccl_bank512 1.0 0.002 0.0001 0.25 0.005 0.2 512 0.3 0.35 0.35 8"
  "reccl_bank1024 1.0 0.002 0.0001 0.25 0.005 0.2 1024 0.3 0.35 0.35 8"

  # Multi-source item graph neighborhood. All graph sources remain positive.
  "topk5 1.0 0.002 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 5"
  "topk10 1.0 0.002 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 10"
  "topk12 1.0 0.002 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 12"
  "topk15 1.0 0.002 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 15"
  "topk20 1.0 0.002 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 20"
  "graph_modal040 1.0 0.002 0.0001 0.25 0.005 0.2 256 0.2 0.4 0.4 8"
  "graph_modal0375 1.0 0.002 0.0001 0.25 0.005 0.2 256 0.25 0.375 0.375 8"
  "graph_cf040 1.0 0.002 0.0001 0.25 0.005 0.2 256 0.4 0.3 0.3 8"
  "graph_img045_txt035 1.0 0.002 0.0001 0.25 0.005 0.2 256 0.2 0.45 0.35 8"
  "graph_img035_txt045 1.0 0.002 0.0001 0.25 0.005 0.2 256 0.2 0.35 0.45 8"

  # Modal residual strength.
  "malpha0p10 1.0 0.002 0.0001 0.10 0.005 0.2 256 0.3 0.35 0.35 8"
  "malpha0p20 1.0 0.002 0.0001 0.20 0.005 0.2 256 0.3 0.35 0.35 8"
  "malpha0p35 1.0 0.002 0.0001 0.35 0.005 0.2 256 0.3 0.35 0.35 8"
  "malpha0p50 1.0 0.002 0.0001 0.50 0.005 0.2 256 0.3 0.35 0.35 8"

  # Compact interactions around promising large-batch hypotheses.
  "lr0p003_reg2em04 1.0 0.003 0.0002 0.25 0.005 0.2 256 0.3 0.35 0.35 8"
  "lr0p003_reccl0p010 1.0 0.003 0.0001 0.25 0.010 0.2 256 0.3 0.35 0.35 8"
  "lr0p003_topk10 1.0 0.003 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 10"
  "lr0p003_modal040 1.0 0.003 0.0001 0.25 0.005 0.2 256 0.2 0.4 0.4 8"
  "lr0p004_reg2em04 1.0 0.004 0.0002 0.25 0.005 0.2 256 0.3 0.35 0.35 8"
  "lr0p004_reccl0p010 1.0 0.004 0.0001 0.25 0.010 0.2 256 0.3 0.35 0.35 8"
  "lr0p004_topk10 1.0 0.004 0.0001 0.25 0.005 0.2 256 0.3 0.35 0.35 10"
  "lr0p004_modal040 1.0 0.004 0.0001 0.25 0.005 0.2 256 0.2 0.4 0.4 8"
  "lr0p004_malpha0p35 1.0 0.004 0.0001 0.35 0.005 0.2 256 0.3 0.35 0.35 8"
  "lr0p006_reg3em04 1.0 0.006 0.0003 0.25 0.005 0.2 256 0.3 0.35 0.35 8"
)

is_gpu_available() {
  local gpu="$1"
  local used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')"
  [[ "${used}" =~ ^[0-9]+$ ]] && (( used < MEM_FREE_THRESHOLD ))
}

suffix_for_tag() {
  local tag="$1"
  printf "stage2_sports_mr%s_bs%s_%s_%s" "${MR_TAG}" "${BATCH_SIZE}" "${tag}" "${RUN_TAG}"
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
    print(f"{rec:.5f}\t{ndg if False else ndcg:.5f}\t{best}\t{name}")
PY
}

run_candidate() {
  local gpu="$1"
  local tag="$2"
  local mbpr="$3"
  local lr_rec="$4"
  local reg="$5"
  local modal_alpha="$6"
  local cl_weight="$7"
  local cl_temp="$8"
  local cl_bank="$9"
  local cf_w="${10}"
  local image_w="${11}"
  local text_w="${12}"
  local topk="${13}"
  local suffix log_path cmd_path

  suffix="$(suffix_for_tag "${tag}")"
  log_path="${LOG_DIR}/${suffix}.log"
  cmd_path="${LOG_DIR}/${suffix}.log.cmd"

  if [[ -f "${log_path}" ]] && grep -q "final strict test hr@20" "${log_path}"; then
    log "skip finished ${tag}"
    return 0
  fi

  log "launch ${tag} on GPU ${gpu}"
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
    --imputer_ckpt "${IMPUTER_CKPT}" \
    --suffix "${suffix}" \
    --epoch "${EPOCHS}" \
    --early_stop "${EARLY_STOP}" \
    --eva_interval 1 \
    --batch_size "${BATCH_SIZE}" \
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
    --topk "[10, 20, 30, 40, 50]" \
    > "${cmd_path}"
  printf "\n" >> "${cmd_path}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    cat "${cmd_path}"
    return 0
  fi

  .venv/bin/python -u main.py \
    --config "${CONFIG}" \
    --device_id "${gpu}" \
    --dataset "${DATASET}" \
    --exp_mode "${EXP_MODE}" \
    --train_stage recommender \
    --missing_rate "${MISSING_RATE}" \
    --seed "${SEED}" \
    --dataset_seed "${DATASET_SEED}" \
    --missing_mask_protocol i3 \
    --imputer_ckpt "${IMPUTER_CKPT}" \
    --suffix "${suffix}" \
    --epoch "${EPOCHS}" \
    --early_stop "${EARLY_STOP}" \
    --eva_interval 1 \
    --batch_size "${BATCH_SIZE}" \
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
    --topk "[10, 20, 30, 40, 50]" \
    > "${log_path}" 2>&1
  log "done ${tag} on GPU ${gpu}"
  summarize_results > "${SUMMARY_FILE}"
}

claim_next_candidate() {
  (
    flock -x 9
    local idx tag suffix log_path
    idx="$(cat "${NEXT_FILE}")"
    while (( idx < ${#CANDIDATES[@]} )); do
      read -r tag _ <<<"${CANDIDATES[$idx]}"
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
  local cand tag mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank cf_w image_w text_w topk

  while true; do
    while [[ "${DRY_RUN}" != "1" ]] && ! is_gpu_available "${gpu}"; do
      log "worker=${worker_idx} gpu=${gpu} waiting: memory used above threshold"
      sleep "${POLL_SECONDS}"
    done

    if ! cand="$(claim_next_candidate)"; then
      log "worker=${worker_idx} gpu=${gpu} no candidates left"
      return 0
    fi

    read -r tag mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank cf_w image_w text_w topk <<<"${cand}"
    run_candidate "${gpu}" "${tag}" "${mbpr}" "${lr_rec}" "${reg}" "${modal_alpha}" \
      "${cl_weight}" "${cl_temp}" "${cl_bank}" "${cf_w}" "${image_w}" "${text_w}" "${topk}"
  done
}

log "run_tag=${RUN_TAG}"
log "config=${CONFIG}"
log "imputer_ckpt=${IMPUTER_CKPT}"
log "missing_rate=${MISSING_RATE}"
log "seed=${SEED}"
log "batch_size=${BATCH_SIZE}"
log "gpus=${GPUS}"
log "max_jobs_per_gpu=${MAX_JOBS_PER_GPU}"
log "max_parallel=${MAX_PARALLEL}"
log "candidates=${#CANDIDATES[@]}"
log "dry_run=${DRY_RUN}"

read -r -a BASE_GPUS <<<"${GPUS}"
GPU_SLOTS=()
for gpu in "${BASE_GPUS[@]}"; do
  for ((slot = 0; slot < MAX_JOBS_PER_GPU; slot++)); do
    GPU_SLOTS+=("${gpu}")
  done
done

WORKER_COUNT="${#GPU_SLOTS[@]}"
if (( WORKER_COUNT > MAX_PARALLEL )); then
  WORKER_COUNT="${MAX_PARALLEL}"
fi
if (( WORKER_COUNT < 1 )); then
  echo "no GPU slots configured" >&2
  exit 1
fi

for ((worker_idx = 0; worker_idx < WORKER_COUNT; worker_idx++)); do
  worker_loop "${worker_idx}" "${GPU_SLOTS[$worker_idx]}" &
done
wait

if [[ "${DRY_RUN}" != "1" ]]; then
  summarize_results > "${SUMMARY_FILE}"
fi
log "all done"
