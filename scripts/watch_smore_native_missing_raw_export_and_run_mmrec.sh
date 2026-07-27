#!/usr/bin/env bash
set -euo pipefail

SMORE_PID="${SMORE_PID:?set SMORE_PID to the running SMORE training process}"
GPU="${GPU:-0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
SMORE_ROOT="${SMORE_ROOT:-/home/ruiyuliu/projects/baselines/SMORE}"
MMREC_ROOT="${MMREC_ROOT:-/home/ruiyuliu/projects/MMRec}"
SMORE_LOG="${SMORE_LOG:-${SMORE_ROOT}/experiment_logs/smore_native_missing_raw_clothing_mr0p3_20260702/run.log}"
TAG="${TAG:-native_missing_raw_mr0p3}"
EXPORT_DIR="${EXPORT_DIR:-${SMORE_ROOT}/experiment_logs/smore_native_missing_raw_input_exports_clothing_mr0p3_20260702}"
COMPACT_DIR="${COMPACT_DIR:-${SMORE_ROOT}/experiment_logs/smore_native_missing_raw_input_compact_clothing_mr0p3_20260702/${TAG}}"
PYTHON="${PYTHON:-${MMREC_ROOT}/.venv/bin/python}"
PIPELINE_LOG="${PIPELINE_LOG:-${SMORE_ROOT}/experiment_logs/smore_native_missing_raw_clothing_mr0p3_20260702/postprocess.log}"

log() {
  echo "[$(date -Is)] $*" | tee -a "${PIPELINE_LOG}"
}

wait_for_smore() {
  log "waiting for SMORE pid ${SMORE_PID}"
  while kill -0 "${SMORE_PID}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
  done
  log "SMORE pid ${SMORE_PID} finished"
}

latest_checkpoint() {
  local ckpt
  ckpt="$(grep 'Saved best checkpoint to' "${SMORE_LOG}" | tail -1 | sed -E 's/.*Saved best checkpoint to //')"
  if [[ -z "${ckpt}" ]]; then
    log "no checkpoint found in ${SMORE_LOG}"
    exit 1
  fi
  if [[ "${ckpt}" != /* ]]; then
    ckpt="${SMORE_ROOT}/src/${ckpt}"
  fi
  if [[ ! -f "${ckpt}" ]]; then
    log "checkpoint does not exist: ${ckpt}"
    exit 1
  fi
  printf '%s\n' "${ckpt}"
}

export_representations() {
  local ckpt="$1"
  if [[ -e "${EXPORT_DIR}" ]]; then
    log "export dir already exists, skipping export: ${EXPORT_DIR}"
    return
  fi
  log "exporting SMORE missing representations from ${ckpt}"
  "${PYTHON}" "${SMORE_ROOT}/scripts/export_pretrain_reps_from_checkpoint.py" \
    --checkpoint "${ckpt}" \
    --tag "${TAG}" \
    --output_dir "${EXPORT_DIR}" \
    --dataset clothing \
    --model SMORE \
    --feature_mode missing
}

compact_representations() {
  local src="${EXPORT_DIR}/${TAG}"
  if [[ -e "${COMPACT_DIR}" ]]; then
    log "compact dir already exists, skipping compact: ${COMPACT_DIR}"
    return
  fi
  log "compacting exported representations into ${COMPACT_DIR}"
  mkdir -p "${COMPACT_DIR}/phase_train" "${COMPACT_DIR}/phase_eval"
  cp "${src}/image_embedding_weight.npy" "${COMPACT_DIR}/image_embedding_weight.npy"
  cp "${src}/text_embedding_weight.npy" "${COMPACT_DIR}/text_embedding_weight.npy"
  cp "${EXPORT_DIR}/manifest.json" "${COMPACT_DIR}/export_manifest.json"

  for phase in train eval; do
    local prefix="SMORE_${phase}_missing_${TAG}"
    for name in agg_image_items agg_text_items all_embeddings_items content_embeds_items fusion_item_embeds image_item_embeds text_item_embeds side_embeds_items image_projected text_projected; do
      cp "${src}/phase_${phase}/${prefix}_${name}.npy" "${COMPACT_DIR}/phase_${phase}/${name}.npy"
    done
    cp "${src}/phase_${phase}/${prefix}_image_observed_mask.npy" "${COMPACT_DIR}/phase_${phase}/image_observed_mask.npy"
    cp "${src}/phase_${phase}/${prefix}_text_observed_mask.npy" "${COMPACT_DIR}/phase_${phase}/text_observed_mask.npy"
    cp "${src}/phase_${phase}/${prefix}_image_observed_ids.npy" "${COMPACT_DIR}/phase_${phase}/image_observed_ids.npy"
    cp "${src}/phase_${phase}/${prefix}_text_observed_ids.npy" "${COMPACT_DIR}/phase_${phase}/text_observed_ids.npy"
    cp "${src}/phase_${phase}/${prefix}_manifest.json" "${COMPACT_DIR}/phase_${phase}/manifest.json"
  done
}

run_mmrec() {
  local run_tag="clothing_mr0p3_smore_native_missing_raw_item_embeds_$(date +%Y%m%d_%H%M%S)_gpu${GPU}"
  log "starting MMRec item_embeds identity/original_linear with FEATURE_DIR=${COMPACT_DIR}, run_tag=${run_tag}"
  cd "${MMREC_ROOT}"
  GPU="${GPU}" \
  RUN_TAG="${run_tag}" \
  FEATURE_DIR="${COMPACT_DIR}" \
  scripts/run_clothing_mr0p3_smore_native_missing_item_embeds_gpu0.sh
}

wait_for_smore
ckpt="$(latest_checkpoint)"
log "latest checkpoint: ${ckpt}"
export_representations "${ckpt}"
compact_representations
run_mmrec
log "pipeline finished"
