#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-7}"
WAIT_MAX_MEMORY_MB="${WAIT_MAX_MEMORY_MB:-36000}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"
QUEUE_TAG="${QUEUE_TAG:-modspec_initprobe_queue_gpu${GPU}_$(date +%Y%m%d_%H%M%S)}"
QUEUE_DIR="${QUEUE_DIR:-exp_report/clothing/modality_dynamic_modspec_initprobe_mr0p3/${QUEUE_TAG}}"
mkdir -p "${QUEUE_DIR}"

if [[ -n "${CANDIDATES:-}" ]]; then
  read -r -a candidate_list <<< "${CANDIDATES}"
else
  candidate_list=(
    soft_init_target
    learned_init_target
    soft_init_blend05
    topk5_a015_init_target
  )
fi

pids=()
names=()

wait_for_slot() {
  local used_mb
  while true; do
    used_mb="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" | tr -d ' ')"
    if [[ "${used_mb}" =~ ^[0-9]+$ ]] && (( used_mb <= WAIT_MAX_MEMORY_MB )); then
      echo "$(date '+%F %T') slot available on GPU ${GPU}: used=${used_mb}MB <= threshold=${WAIT_MAX_MEMORY_MB}MB"
      return 0
    fi
    echo "$(date '+%F %T') wait GPU ${GPU}: used=${used_mb}MB > threshold=${WAIT_MAX_MEMORY_MB}MB"
    sleep "${WAIT_POLL_SECONDS}"
  done
}

for candidate in "${candidate_list[@]}"; do
  wait_for_slot
  run_tag="${QUEUE_TAG}_${candidate}"
  out_file="${QUEUE_DIR}/${candidate}.launcher.out"
  echo "$(date '+%F %T') launch ${candidate} on GPU ${GPU}; out=${out_file}"
  (
    env \
      GPU="${GPU}" \
      CANDIDATE="${candidate}" \
      WAIT_MAX_MEMORY_MB=999999 \
      WAIT_POLL_SECONDS=1 \
      RUN_TAG="${run_tag}" \
      OUT_DIR="${QUEUE_DIR}/${candidate}" \
      bash scripts/run_clothing_mr0p3_modspec_initprobe_backfill.sh \
      > "${out_file}" 2>&1
  ) &
  pids+=("$!")
  names+=("${candidate}")
  sleep 20
done

overall=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  name="${names[$i]}"
  if wait "${pid}"; then
    echo "$(date '+%F %T') done ${name}"
  else
    rc=$?
    echo "$(date '+%F %T') failed ${name} exit=${rc}"
    overall=1
  fi
done

exit "${overall}"
