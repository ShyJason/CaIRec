#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="sports"
EXP_MODE="mm"
MISSING_RATE="${MISSING_RATE:-0.3}"
MR_TAG="${MISSING_RATE//./p}"
SEED="${SEED:-1}"
DATASET_SEED="${DATASET_SEED:-0}"
RUN_TAG="${RUN_TAG:-sports_modality_completed_bs2048_compact_mr${MR_TAG}_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"

OUT_DIR="${OUT_DIR:-exp_report/sports/modality_completed_bs2048_compact_search/${RUN_TAG}}"
LOG_DIR="${OUT_DIR}/logs"
STATE_DIR="${OUT_DIR}/state"
mkdir -p "${LOG_DIR}" "${STATE_DIR}"

NEXT_FILE="${STATE_DIR}/next_index"
LOCK_FILE="${STATE_DIR}/queue.lock"
SUMMARY_FILE="${OUT_DIR}/summary.tsv"
SUMMARY_VAL_FILE="${OUT_DIR}/summary_val.tsv"
LAUNCHER_LOG="${OUT_DIR}/launcher.log"
if [[ ! -f "${NEXT_FILE}" ]]; then
  echo 0 > "${NEXT_FILE}"
fi

CONFIG="${CONFIG:-configs/sports/stage2_decoder_mm.yaml}"
IMPUTER_CKPT="${IMPUTER_CKPT:-exp_report/sports/stage1_2_sports_clothing_stage1style_mr0p3_seed1_20260627_004233/ckpt/stage1_2_sports_clothing_stage1style_mr0p3_seed1_20260627_004233_imputer_backprop_50_epoch49.pth}"
GPUS="${GPUS:-0 7}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
MEM_FREE_MIN="${MEM_FREE_MIN:-6500}"
POLL_SECONDS="${POLL_SECONDS:-60}"
DRY_RUN="${DRY_RUN:-0}"
EPOCHS="${EPOCHS:-500}"
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

# tag mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank topk image_cf image_sem text_cf text_sem
CANDIDATES=(
  # Direct transfer from the strongest Sports bs=2048 fused run.
  "reg1em03_center 1.0 0.002 0.0010 0.25 0.005 0.20 256 8 0.30 0.35 0.30 0.35"

  # Best nearby bs=2048 fused regularization / BPR hypotheses.
  "reg3em04 1.0 0.002 0.0003 0.25 0.005 0.20 256 8 0.30 0.35 0.30 0.35"
  "reg2em04 1.0 0.002 0.0002 0.25 0.005 0.20 256 8 0.30 0.35 0.30 0.35"
  "mbpr1p5 1.5 0.002 0.0001 0.25 0.005 0.20 256 8 0.30 0.35 0.30 0.35"

  # CL variants that were competitive in the large-batch fused search.
  "reccl0p015 1.0 0.002 0.0001 0.25 0.015 0.20 256 8 0.30 0.35 0.30 0.35"
  "reccl_bank128 1.0 0.002 0.0001 0.25 0.005 0.20 128 8 0.30 0.35 0.30 0.35"

  # Graph neighborhood and modality-specific graph weights.
  "topk12_reg1em03 1.0 0.002 0.0010 0.25 0.005 0.20 256 12 0.30 0.35 0.30 0.35"
  "sem_high_sym_reg1em03 1.0 0.002 0.0010 0.25 0.005 0.20 256 8 0.20 0.45 0.20 0.45"
  "cf_high_sym_reg1em03 1.0 0.002 0.0010 0.25 0.005 0.20 256 8 0.40 0.30 0.40 0.30"
  "text_sem_high_reg1em03 1.0 0.002 0.0010 0.25 0.005 0.20 256 8 0.35 0.30 0.20 0.45"

  # Propagation strength check.
  "alpha0p35_reg1em03 1.0 0.002 0.0010 0.35 0.005 0.20 256 8 0.30 0.35 0.30 0.35"
  "alpha0p20_reg1em03 1.0 0.002 0.0010 0.20 0.005 0.20 256 8 0.30 0.35 0.30 0.35"
)

is_gpu_available() {
  local gpu="$1"
  local free
  free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')"
  [[ "${free}" =~ ^[0-9]+$ ]] && (( free >= MEM_FREE_MIN ))
}

suffix_for_tag() {
  local tag="$1"
  printf "stage2_sports_mr%s_modality_completed_bs%s_%s_%s" "${MR_TAG}" "${BATCH_SIZE}" "${tag}" "${RUN_TAG}"
}

summarize_results() {
  python3 - "${LOG_DIR}" "${SUMMARY_FILE}" "${SUMMARY_VAL_FILE}" <<'PY'
import glob
import os
import re
import sys

log_dir, summary_path, summary_val_path = sys.argv[1:4]
final_rows = []
val_rows = []
for path in glob.glob(os.path.join(log_dir, "*.log")):
    text = open(path, "r", errors="ignore").read()
    final = re.findall(
        r"final strict test hr@20 = ([0-9.]+), recall@20 = ([0-9.]+), ndcg@20 = ([0-9.]+)",
        text,
    )
    best_epoch = re.findall(r"best epoch\s+([0-9]+)", text)
    if final:
        _, rec, ndcg = map(float, final[-1])
        final_rows.append((rec, ndcg, int(best_epoch[-1]) if best_epoch else -1, os.path.basename(path)))

    val_matches = re.findall(
        r"epoch = ([0-9]+) hr@20 = ([0-9.]+), recall@20 = ([0-9.]+), ndcg@20 = ([0-9.]+)",
        text,
    )
    if val_matches:
        epoch, rec, ndcg = max(
            ((int(e), float(r), float(n)) for e, _hr, r, n in val_matches),
            key=lambda row: (row[1], row[2]),
        )
        val_rows.append((rec, ndcg, epoch, os.path.basename(path)))

with open(summary_path, "w") as f:
    f.write("recall20\tndcg20\tbest_epoch\tlog\n")
    for rec, ndcg, epoch, name in sorted(final_rows, reverse=True):
        f.write(f"{rec:.5f}\t{ndcg:.5f}\t{epoch}\t{name}\n")

with open(summary_val_path, "w") as f:
    f.write("val_recall20\tval_ndcg20\tbest_val_epoch\tlog\n")
    for rec, ndcg, epoch, name in sorted(val_rows, reverse=True):
        f.write(f"{rec:.5f}\t{ndcg:.5f}\t{epoch}\t{name}\n")
PY
}

build_command() {
  local gpu="$1"
  local tag="$2"
  local mbpr="$3"
  local lr_rec="$4"
  local reg="$5"
  local modal_alpha="$6"
  local cl_weight="$7"
  local cl_temp="$8"
  local cl_bank="$9"
  local topk="${10}"
  local image_cf="${11}"
  local image_sem="${12}"
  local text_cf="${13}"
  local text_sem="${14}"
  local suffix cf_global

  suffix="$(suffix_for_tag "${tag}")"
  cf_global="$(python3 - "${image_cf}" "${text_cf}" <<'PY'
import sys
print(f"{(float(sys.argv[1]) + float(sys.argv[2])) / 2.0:.6g}")
PY
)"
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
    --item_graph_kind modality_completed \
    --item_graph_topk "${topk}" \
    --item_graph_norm rw \
    --item_graph_cf_weight "${cf_global}" \
    --item_graph_image_weight 0.0 \
    --item_graph_text_weight 0.0 \
    --item_graph_image_cf_weight "${image_cf}" \
    --item_graph_image_semantic_weight "${image_sem}" \
    --item_graph_text_cf_weight "${text_cf}" \
    --item_graph_text_semantic_weight "${text_sem}" \
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
  local mbpr="$3"
  local lr_rec="$4"
  local reg="$5"
  local modal_alpha="$6"
  local cl_weight="$7"
  local cl_temp="$8"
  local cl_bank="$9"
  local topk="${10}"
  local image_cf="${11}"
  local image_sem="${12}"
  local text_cf="${13}"
  local text_sem="${14}"
  local suffix log_path status

  suffix="$(suffix_for_tag "${tag}")"
  log_path="${LOG_DIR}/${suffix}.log"
  if [[ -f "${log_path}" ]] && grep -qi "final strict test hr@20" "${log_path}"; then
    log "skip completed ${tag}"
    return 0
  fi

  log "launch ${tag} on GPU ${gpu} bs=${BATCH_SIZE} topk=${topk} alpha=${modal_alpha} image=cf${image_cf}/sem${image_sem} text=cf${text_cf}/sem${text_sem}"
  build_command "${gpu}" "${tag}" "${mbpr}" "${lr_rec}" "${reg}" \
    "${modal_alpha}" "${cl_weight}" "${cl_temp}" "${cl_bank}" "${topk}" \
    "${image_cf}" "${image_sem}" "${text_cf}" "${text_sem}" \
    > "${log_path}.cmd"

  if [[ "${DRY_RUN}" == "1" ]]; then
    cat "${log_path}.cmd"
    return 0
  fi

  set +e
  bash -lc "$(cat "${log_path}.cmd")" > "${log_path}" 2>&1
  status=$?
  set -e
  echo "RUN_EXIT_STATUS=${status}" >> "${log_path}"
  summarize_results || true
  log "done ${tag} on GPU ${gpu} status=${status}"
  return 0
}

claim_next_candidate() {
  (
    flock -x 9
    local idx tag mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank topk image_cf image_sem text_cf text_sem suffix log_path
    idx="$(cat "${NEXT_FILE}")"
    while (( idx < ${#CANDIDATES[@]} )); do
      read -r tag mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank topk image_cf image_sem text_cf text_sem <<<"${CANDIDATES[$idx]}"
      suffix="$(suffix_for_tag "${tag}")"
      log_path="${LOG_DIR}/${suffix}.log"
      if [[ -f "${log_path}" ]] && grep -qi "final strict test hr@20" "${log_path}"; then
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
  local cand tag mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank topk image_cf image_sem text_cf text_sem

  while true; do
    while [[ "${DRY_RUN}" != "1" ]] && ! is_gpu_available "${gpu}"; do
      log "worker=${worker_idx} gpu=${gpu} waiting: memory.free < ${MEM_FREE_MIN} MiB"
      sleep "${POLL_SECONDS}"
    done

    if ! cand="$(claim_next_candidate)"; then
      log "worker=${worker_idx} gpu=${gpu} no candidates left"
      return 0
    fi

    read -r tag mbpr lr_rec reg modal_alpha cl_weight cl_temp cl_bank topk image_cf image_sem text_cf text_sem <<<"${cand}"
    run_candidate "${gpu}" "${tag}" "${mbpr}" "${lr_rec}" "${reg}" \
      "${modal_alpha}" "${cl_weight}" "${cl_temp}" "${cl_bank}" "${topk}" \
      "${image_cf}" "${image_sem}" "${text_cf}" "${text_sem}"
  done
}

log "run_tag=${RUN_TAG}"
log "config=${CONFIG}"
log "missing_rate=${MISSING_RATE}"
log "seed=${SEED}, dataset_seed=${DATASET_SEED}"
log "imputer_ckpt=${IMPUTER_CKPT}"
log "gpus=${GPUS}, max_parallel=${MAX_PARALLEL}, mem_free_min=${MEM_FREE_MIN}"
log "batch_size=${BATCH_SIZE}, epochs=${EPOCHS}, early_stop=${EARLY_STOP}"
log "total_candidates=${#CANDIDATES[@]}, dry_run=${DRY_RUN}"

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
  summarize_results || true
fi
log "all done"
