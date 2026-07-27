#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-0}"
SLOT_WORKERS="${SLOT_WORKERS:-4}"
WAIT_MAX_MEMORY_MB="${WAIT_MAX_MEMORY_MB:-36000}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"
LAUNCH_SETTLE_SECONDS="${LAUNCH_SETTLE_SECONDS:-25}"
QUEUE_TAG="${QUEUE_TAG:-modspec_nextwave_slotfill_gpu${GPU}_$(date +%Y%m%d_%H%M%S)}"
QUEUE_DIR="${QUEUE_DIR:-exp_report/clothing/modality_dynamic_modspec_nextwave_mr0p3/${QUEUE_TAG}}"
mkdir -p "${QUEUE_DIR}"

CONFIG="${CONFIG:-configs/clothing/mainline_mr0p1.yaml}"
IMPUTER_CKPT="${IMPUTER_CKPT:-exp_report/clothing/stage1_2_clothing_mm_mr0p3_beststyle_nocl_clothing_mr0p3_latest_20260624_174840/ckpt/stage1_2_clothing_mm_mr0p3_beststyle_nocl_clothing_mr0p3_latest_20260624_174840_imputer_backprop_50_epoch49.pth}"
QUEUE_FILE="${QUEUE_DIR}/candidates.queue"

if [[ ! -f "${QUEUE_FILE}" ]]; then
  if [[ -n "${CANDIDATES:-}" ]]; then
    for candidate in ${CANDIDATES}; do
      printf "%s\n" "${candidate}" >> "${QUEUE_FILE}"
    done
  else
    cat > "${QUEUE_FILE}" <<'EOF'
topk10_reg005_init_target
topk10_reg001_init_target
topk10_reg003_init_target
topk10_warm_score30
topk10_warm_neigh30
topk8_cf0125_reg005
topk12_reg005_init_target
topk5_alpha015_reg005
topk8_alpha015_reg005
topk10_cf020_reg005
EOF
  fi
fi

pop_candidate() {
  local selected=""
  exec 8>"${QUEUE_DIR}/queue.lock"
  flock 8
  if [[ -s "${QUEUE_FILE}" ]]; then
    selected="$(head -n 1 "${QUEUE_FILE}")"
    tail -n +2 "${QUEUE_FILE}" > "${QUEUE_FILE}.tmp"
    mv "${QUEUE_FILE}.tmp" "${QUEUE_FILE}"
  fi
  flock -u 8
  exec 8>&-
  printf "%s" "${selected}"
}

set_candidate_params() {
  local candidate="$1"
  case "${candidate}" in
    topk10_reg005_init_target)
      topk=10; alpha=0.25; cf_weight=0.15; image_weight=0.425; text_weight=0.425
      conf_reg=0.05; cl_weight=0.01; lr_rec=0.01; lr_confidence=""
      score_blend=1.0; score_blend_start=-1.0; score_blend_warmup=0
      neigh_blend=1.0; neigh_blend_start=-1.0; neigh_blend_warmup=0
      text_rr=0.70; text_ri=0.10; text_ii=0.02
      image_rr=0.05; image_ri=0.05; image_ii=0.70
      ;;
    topk10_reg003_init_target)
      topk=10; alpha=0.25; cf_weight=0.15; image_weight=0.425; text_weight=0.425
      conf_reg=0.03; cl_weight=0.01; lr_rec=0.01; lr_confidence=""
      score_blend=1.0; score_blend_start=-1.0; score_blend_warmup=0
      neigh_blend=1.0; neigh_blend_start=-1.0; neigh_blend_warmup=0
      text_rr=0.70; text_ri=0.10; text_ii=0.02
      image_rr=0.05; image_ri=0.05; image_ii=0.70
      ;;
    topk10_reg001_init_target)
      topk=10; alpha=0.25; cf_weight=0.15; image_weight=0.425; text_weight=0.425
      conf_reg=0.01; cl_weight=0.01; lr_rec=0.01; lr_confidence=""
      score_blend=1.0; score_blend_start=-1.0; score_blend_warmup=0
      neigh_blend=1.0; neigh_blend_start=-1.0; neigh_blend_warmup=0
      text_rr=0.70; text_ri=0.10; text_ii=0.02
      image_rr=0.05; image_ri=0.05; image_ii=0.70
      ;;
    topk10_warm_score30)
      topk=10; alpha=0.25; cf_weight=0.15; image_weight=0.425; text_weight=0.425
      conf_reg=0.02; cl_weight=0.01; lr_rec=0.01; lr_confidence=""
      score_blend=1.0; score_blend_start=0.0; score_blend_warmup=30
      neigh_blend=1.0; neigh_blend_start=-1.0; neigh_blend_warmup=0
      text_rr=0.70; text_ri=0.10; text_ii=0.02
      image_rr=0.05; image_ri=0.05; image_ii=0.70
      ;;
    topk10_warm_neigh30)
      topk=10; alpha=0.25; cf_weight=0.15; image_weight=0.425; text_weight=0.425
      conf_reg=0.02; cl_weight=0.01; lr_rec=0.01; lr_confidence=""
      score_blend=1.0; score_blend_start=-1.0; score_blend_warmup=0
      neigh_blend=1.0; neigh_blend_start=0.0; neigh_blend_warmup=30
      text_rr=0.70; text_ri=0.10; text_ii=0.02
      image_rr=0.05; image_ri=0.05; image_ii=0.70
      ;;
    topk8_cf0125_reg005)
      topk=8; alpha=0.25; cf_weight=0.125; image_weight=0.4375; text_weight=0.4375
      conf_reg=0.05; cl_weight=0.01; lr_rec=0.01; lr_confidence=""
      score_blend=1.0; score_blend_start=-1.0; score_blend_warmup=0
      neigh_blend=1.0; neigh_blend_start=-1.0; neigh_blend_warmup=0
      text_rr=0.45; text_ri=0.10; text_ii=0.02
      image_rr=0.05; image_ri=0.05; image_ii=0.55
      ;;
    topk12_reg005_init_target)
      topk=12; alpha=0.25; cf_weight=0.15; image_weight=0.425; text_weight=0.425
      conf_reg=0.05; cl_weight=0.01; lr_rec=0.01; lr_confidence=""
      score_blend=1.0; score_blend_start=-1.0; score_blend_warmup=0
      neigh_blend=1.0; neigh_blend_start=-1.0; neigh_blend_warmup=0
      text_rr=0.70; text_ri=0.10; text_ii=0.02
      image_rr=0.05; image_ri=0.05; image_ii=0.70
      ;;
    topk5_alpha015_reg005)
      topk=5; alpha=0.15; cf_weight=0.15; image_weight=0.425; text_weight=0.425
      conf_reg=0.05; cl_weight=0.01; lr_rec=0.01; lr_confidence=""
      score_blend=1.0; score_blend_start=-1.0; score_blend_warmup=0
      neigh_blend=1.0; neigh_blend_start=-1.0; neigh_blend_warmup=0
      text_rr=0.45; text_ri=0.10; text_ii=0.02
      image_rr=0.05; image_ri=0.05; image_ii=0.55
      ;;
    topk8_alpha015_reg005)
      topk=8; alpha=0.15; cf_weight=0.15; image_weight=0.425; text_weight=0.425
      conf_reg=0.05; cl_weight=0.01; lr_rec=0.01; lr_confidence=""
      score_blend=1.0; score_blend_start=-1.0; score_blend_warmup=0
      neigh_blend=1.0; neigh_blend_start=-1.0; neigh_blend_warmup=0
      text_rr=0.45; text_ri=0.10; text_ii=0.02
      image_rr=0.05; image_ri=0.05; image_ii=0.55
      ;;
    topk10_cf020_reg005)
      topk=10; alpha=0.25; cf_weight=0.20; image_weight=0.40; text_weight=0.40
      conf_reg=0.05; cl_weight=0.01; lr_rec=0.01; lr_confidence=""
      score_blend=1.0; score_blend_start=-1.0; score_blend_warmup=0
      neigh_blend=1.0; neigh_blend_start=-1.0; neigh_blend_warmup=0
      text_rr=0.70; text_ri=0.10; text_ii=0.02
      image_rr=0.05; image_ri=0.05; image_ii=0.70
      ;;
    topk10_conflr001_reg005)
      topk=10; alpha=0.25; cf_weight=0.15; image_weight=0.425; text_weight=0.425
      conf_reg=0.05; cl_weight=0.01; lr_rec=0.01; lr_confidence=0.001
      score_blend=1.0; score_blend_start=-1.0; score_blend_warmup=0
      neigh_blend=1.0; neigh_blend_start=-1.0; neigh_blend_warmup=0
      text_rr=0.70; text_ri=0.10; text_ii=0.02
      image_rr=0.05; image_ri=0.05; image_ii=0.70
      ;;
    topk10_conflr003_reg005)
      topk=10; alpha=0.25; cf_weight=0.15; image_weight=0.425; text_weight=0.425
      conf_reg=0.05; cl_weight=0.01; lr_rec=0.01; lr_confidence=0.003
      score_blend=1.0; score_blend_start=-1.0; score_blend_warmup=0
      neigh_blend=1.0; neigh_blend_start=-1.0; neigh_blend_warmup=0
      text_rr=0.70; text_ri=0.10; text_ii=0.02
      image_rr=0.05; image_ri=0.05; image_ii=0.70
      ;;
    topk10_conflr030_reg005)
      topk=10; alpha=0.25; cf_weight=0.15; image_weight=0.425; text_weight=0.425
      conf_reg=0.05; cl_weight=0.01; lr_rec=0.01; lr_confidence=0.03
      score_blend=1.0; score_blend_start=-1.0; score_blend_warmup=0
      neigh_blend=1.0; neigh_blend_start=-1.0; neigh_blend_warmup=0
      text_rr=0.70; text_ri=0.10; text_ii=0.02
      image_rr=0.05; image_ri=0.05; image_ii=0.70
      ;;
    *)
      echo "unknown candidate=${candidate}" >&2
      return 2
      ;;
  esac
}

run_candidate() {
  local candidate="$1"
  set_candidate_params "${candidate}" || return $?

  local run_dir="${QUEUE_DIR}/${candidate}"
  local log_dir="${run_dir}/logs"
  mkdir -p "${log_dir}"
  local suffix="stage2_clothing_mr0p3_${candidate}_${QUEUE_TAG}"
  local log_path="${log_dir}/${suffix}.log"
  local cmd_path="${run_dir}/run.cmd"
  local status_path="${run_dir}/exit.status"
  local pid_path="${run_dir}/pid"

  if [[ -f "${log_path}" ]] && grep -qi "final strict test hr@20" "${log_path}"; then
    echo "$(date '+%F %T') skip completed ${candidate}"
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
    --epoch 200
    --early_stop 20
    --eva_interval 1
    --batch_size 2048
    --lr 0.01
    --lr_rec "${lr_rec}"
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
    --rec_neighbor_cl_weight "${cl_weight}"
    --rec_neighbor_cl_temp 0.2
    --rec_neighbor_cl_bank_size 256
    --item_graph_kind modality_completed_dynamic_confidence
    --item_graph_topk "${topk}"
    --item_graph_norm rw
    --item_graph_cf_scale raw
    --item_graph_cf_weight "${cf_weight}"
    --item_graph_image_weight "${image_weight}"
    --item_graph_text_weight "${text_weight}"
    --item_graph_audio_weight 0.0
    --item_graph_modal_alpha "${alpha}"
    --item_graph_modal_layers 1
    --item_graph_modal_target all
    --item_graph_modality_specific_confidence 1
    --item_graph_confidence_transform sigmoid
    --item_graph_dynamic_score_blend "${score_blend}"
    --item_graph_dynamic_score_blend_start "${score_blend_start}"
    --item_graph_dynamic_score_blend_warmup_epochs "${score_blend_warmup}"
    --item_graph_dynamic_neighbor_blend "${neigh_blend}"
    --item_graph_dynamic_neighbor_blend_start "${neigh_blend_start}"
    --item_graph_dynamic_neighbor_blend_warmup_epochs "${neigh_blend_warmup}"
    --item_graph_confidence_min 0.0
    --item_graph_confidence_max 1.0
    --item_graph_confidence_reg_coeff "${conf_reg}"
    --item_graph_confidence_reg_start_epoch 0
    --item_graph_confidence_log_interval 10
    --item_graph_rr_confidence_reg_target 1.0
    --item_graph_ri_confidence_reg_target 1.0
    --item_graph_ii_confidence_reg_target 1.0
    --item_graph_text_rr_confidence_init "${text_rr}"
    --item_graph_text_ri_confidence_init "${text_ri}"
    --item_graph_text_ii_confidence_init "${text_ii}"
    --item_graph_image_rr_confidence_init "${image_rr}"
    --item_graph_image_ri_confidence_init "${image_ri}"
    --item_graph_image_ii_confidence_init "${image_ii}"
    --item_graph_text_rr_confidence_reg_target "${text_rr}"
    --item_graph_text_ri_confidence_reg_target "${text_ri}"
    --item_graph_text_ii_confidence_reg_target "${text_ii}"
    --item_graph_image_rr_confidence_reg_target "${image_rr}"
    --item_graph_image_ri_confidence_reg_target "${image_ri}"
    --item_graph_image_ii_confidence_reg_target "${image_ii}"
    --tensorboard 0
    --save 1
    --topk "[10, 20, 30, 40, 50]"
    --suffix "${suffix}"
  )

  if [[ -n "${lr_confidence}" ]]; then
    cmd+=(--item_graph_confidence_lr "${lr_confidence}")
  fi

  printf "%q " "${cmd[@]}" > "${cmd_path}"
  printf "\n" >> "${cmd_path}"

  local run_pid=""
  while true; do
    exec 9>"${QUEUE_DIR}/launch.lock"
    flock 9
    local used_mb
    used_mb="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" | tr -d ' ')"
    if [[ "${used_mb}" =~ ^[0-9]+$ ]] && (( used_mb <= WAIT_MAX_MEMORY_MB )); then
      echo "$(date '+%F %T') start ${candidate} on GPU ${GPU}; used=${used_mb}MB; log=${log_path}"
      "${cmd[@]}" > "${log_path}" 2>&1 &
      run_pid="$!"
      echo "${run_pid}" > "${pid_path}"
      sleep "${LAUNCH_SETTLE_SECONDS}"
      flock -u 9
      exec 9>&-
      break
    fi
    flock -u 9
    exec 9>&-
    echo "$(date '+%F %T') wait GPU ${GPU}: used=${used_mb}MB > threshold=${WAIT_MAX_MEMORY_MB}MB for ${candidate}"
    sleep "${WAIT_POLL_SECONDS}"
  done

  wait "${run_pid}"
  local rc=$?
  echo "EXIT_CODE=${rc}" > "${status_path}"
  return "${rc}"
}

worker_loop() {
  local worker_id="$1"
  while true; do
    local candidate
    candidate="$(pop_candidate)"
    if [[ -z "${candidate}" ]]; then
      echo "$(date '+%F %T') worker ${worker_id}: queue empty"
      return 0
    fi
    echo "$(date '+%F %T') worker ${worker_id}: claimed ${candidate}"
    if run_candidate "${candidate}"; then
      echo "$(date '+%F %T') worker ${worker_id}: done ${candidate}"
    else
      local rc=$?
      echo "$(date '+%F %T') worker ${worker_id}: failed ${candidate} exit=${rc}"
    fi
  done
}

pids=()
supervisor_id="${SUPERVISOR_ID:-$$}"
for worker_id in $(seq 1 "${SLOT_WORKERS}"); do
  worker_loop "${worker_id}" > "${QUEUE_DIR}/worker_${supervisor_id}_${worker_id}.out" 2>&1 &
  pids+=("$!")
done

overall=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    overall=1
  fi
done

exit "${overall}"
