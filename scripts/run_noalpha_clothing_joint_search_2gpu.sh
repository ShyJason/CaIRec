#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PATH="${ROOT_DIR}/.venv/bin:${PATH}"
export PYTHONUNBUFFERED=1
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

RUN_TAG="${RUN_TAG:-noalpha_clothing_joint_$(date +%Y%m%d_%H%M%S)}"
GPUS_STR="${GPUS:-2 3}"
SEARCH_SEEDS_STR="${SEARCH_SEEDS:-12 123}"
FULL_SEEDS_STR="${FULL_SEEDS:-1 12 123 1234 12345}"
ASSEMBLE_ALPHAS="${ASSEMBLE_ALPHAS:-1}"
PRIMARY_ALPHA="${PRIMARY_ALPHA:-1}"
DRY_RUN="${DRY_RUN:-0}"
SEARCH_EPOCHS="${SEARCH_EPOCHS:-80}"
SEARCH_EARLY_STOP="${SEARCH_EARLY_STOP:-10}"
SEARCH_EVA_INTERVAL="${SEARCH_EVA_INTERVAL:-1}"
FULL_EPOCHS="${FULL_EPOCHS:-200}"
FULL_EARLY_STOP="${FULL_EARLY_STOP:-20}"
FULL_EVA_INTERVAL="${FULL_EVA_INTERVAL:-1}"
CANDIDATE_SET="${CANDIDATE_SET:-joint_focus}"
FULL_CANDIDATES_STR="${FULL_CANDIDATES:-}"
SKIP_SEARCH="${SKIP_SEARCH:-0}"

read -r -a GPUS <<< "${GPUS_STR}"
read -r -a SEARCH_SEEDS <<< "${SEARCH_SEEDS_STR}"
read -r -a FULL_SEEDS <<< "${FULL_SEEDS_STR}"

OUT_DIR="${ROOT_DIR}/exp_report/noalpha_clothing_joint_search/${RUN_TAG}"
mkdir -p "${OUT_DIR}/tasks" "${OUT_DIR}/logs/search" "${OUT_DIR}/logs/full" "${OUT_DIR}/eval/search" "${OUT_DIR}/eval/full"

log() {
  date +"[noalpha-clothing-joint] %Y-%m-%d %H:%M:%S $*"
}

make_gate_args() {
  local beta="$1"
  local reg="$2"
  local identity="$3"
  local balance="$4"
  local hidden="$5"
  local dropout="$6"
  local ctx="$7"
  local ctx_args=("--completion_gate_use_item_context" "1" "--completion_gate_item_context_source" "shared_mean")
  if [[ "${ctx}" == "off" ]]; then
    ctx_args=("--completion_gate_use_item_context" "0" "--completion_gate_item_context_source" "off")
  elif [[ "${ctx}" == "id" ]]; then
    ctx_args=("--completion_gate_use_item_context" "1" "--completion_gate_item_context_source" "id_embedding")
  fi

  local args=(
    --completion_gate_mode rank_residual_allgate
    --completion_gate_no_residual_alpha 1
    --completion_gate_hidden_dim "${hidden}"
    --completion_gate_dropout "${dropout}"
    --completion_gate_init_logit 0.0
    --completion_gate_detach_inputs 1
    "${ctx_args[@]}"
    --completion_gate_mix_alpha "${beta}"
    --completion_gate_identity_coeff "${identity}"
    --completion_gate_balance_coeff "${balance}"
    --completion_gate_reg_coeff "${reg}"
    --recommender_allow_modal_grad 0
  )
  printf '%q ' "${args[@]}"
}

candidate_table() {
  if [[ "${CANDIDATE_SET}" == "joint_focus" ]]; then
    local -a gates=(
      "b0p70_g003_i005_bal000_h64_d01_shared|$(make_gate_args 0.70 0.03 0.05 0.00 64 0.1 shared)"
      "b0p70_g003_i005_bal001_h64_d01_shared|$(make_gate_args 0.70 0.03 0.05 0.01 64 0.1 shared)"
      "b0p75_g003_i005_bal000_h64_d01_shared|$(make_gate_args 0.75 0.03 0.05 0.00 64 0.1 shared)"
      "b0p75_g005_i005_bal000_h64_d01_shared|$(make_gate_args 0.75 0.05 0.05 0.00 64 0.1 shared)"
      "b0p80_g003_i005_bal001_h64_d01_shared|$(make_gate_args 0.80 0.03 0.05 0.01 64 0.1 shared)"
      "b0p85_g003_i005_bal001_h64_d01_shared|$(make_gate_args 0.85 0.03 0.05 0.01 64 0.1 shared)"
    )
    local -a profiles=(
      "joint_imp5e5_reg0015|--train_stage joint --freeze_imputer 0 --freeze_decoder 1 --batch_size 2048 --lr 0.01 --lr_rec 0.01 --lr_imp 0.00005 --lr_decoder 0.00005 --reg_coeff 0.015"
      "joint_imp1e4_reg0015|--train_stage joint --freeze_imputer 0 --freeze_decoder 1 --batch_size 2048 --lr 0.01 --lr_rec 0.01 --lr_imp 0.0001 --lr_decoder 0.00005 --reg_coeff 0.015"
      "joint_imp2e4_reg0010|--train_stage joint --freeze_imputer 0 --freeze_decoder 1 --batch_size 2048 --lr 0.01 --lr_rec 0.01 --lr_imp 0.0002 --lr_decoder 0.00005 --reg_coeff 0.010"
      "joint_lr5e3_imp5e5_reg0015_b1024|--train_stage joint --freeze_imputer 0 --freeze_decoder 1 --batch_size 1024 --lr 0.005 --lr_rec 0.005 --lr_imp 0.00005 --lr_decoder 0.00005 --reg_coeff 0.015"
      "joint_dec0_imp5e5_reg0015|--train_stage joint --freeze_imputer 0 --freeze_decoder 0 --batch_size 2048 --lr 0.01 --lr_rec 0.01 --lr_imp 0.00005 --lr_decoder 0.00005 --reg_coeff 0.015"
    )
    local gate_entry profile_entry gate_name gate_args profile_name profile_args
    for gate_entry in "${gates[@]}"; do
      gate_name="${gate_entry%%|*}"
      gate_args="${gate_entry#*|}"
      for profile_entry in "${profiles[@]}"; do
        profile_name="${profile_entry%%|*}"
        profile_args="${profile_entry#*|}"
        printf '%s_%s\t%s %s\n' "${gate_name}" "${profile_name}" "${gate_args}" "${profile_args}"
      done
    done
    return
  fi

  echo "unknown CANDIDATE_SET=${CANDIDATE_SET}" >&2
  return 1
}

imputer_ckpt_for_seed() {
  local seed="$1"
  echo "exp_report/clothing/stage1_2_clothing_mmrec_fixed_seed${seed}_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage1_2_clothing_mmrec_fixed_seed${seed}_mmrec_clothing_mm_fixedmissing_20260521_052129_imputer_backprop_50_epoch19.pth"
}

base_log_for_seed() {
  local seed="$1"
  case "${seed}" in
    1) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed1_mmrec_clothing_mm_fixedmissing_20260521_052129/log/run_20260521_052413.log" ;;
    12) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed12_mmrec_clothing_mm_fixedmissing_20260521_052129/log/run_20260521_052411.log" ;;
    123) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed123_mmrec_clothing_mm_fixedmissing_20260521_052129/log/run_20260521_060217.log" ;;
    1234) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed1234_mmrec_clothing_mm_fixedmissing_20260521_052129/log/run_20260521_055340.log" ;;
    12345) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed12345_mmrec_clothing_mm_fixedmissing_20260521_052129/log/run_20260521_063053.log" ;;
    *) echo "missing base log mapping for clothing seed ${seed}" >&2; return 2 ;;
  esac
}

base_ckpt_for_seed() {
  local seed="$1"
  case "${seed}" in
    1) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed1_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage2_clothing_mmrec_fixed_seed1_mmrec_clothing_mm_fixedmissing_20260521_052129_recommender_1.0_epoch169.pth" ;;
    12) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed12_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage2_clothing_mmrec_fixed_seed12_mmrec_clothing_mm_fixedmissing_20260521_052129_recommender_1.0_epoch122.pth" ;;
    123) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed123_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage2_clothing_mmrec_fixed_seed123_mmrec_clothing_mm_fixedmissing_20260521_052129_recommender_1.0_epoch121.pth" ;;
    1234) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed1234_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage2_clothing_mmrec_fixed_seed1234_mmrec_clothing_mm_fixedmissing_20260521_052129_recommender_1.0_epoch137.pth" ;;
    12345) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed12345_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage2_clothing_mmrec_fixed_seed12345_mmrec_clothing_mm_fixedmissing_20260521_052129_recommender_1.0_epoch121.pth" ;;
    *) echo "missing base ckpt mapping for clothing seed ${seed}" >&2; return 2 ;;
  esac
}

latest_ckpt_for_suffix() {
  local suffix="$1"
  find "exp_report/clothing/${suffix}/ckpt" -maxdepth 1 -name '*.pth' -print | sort -V | tail -1
}

run_task() {
  local gpu="$1"
  local phase="$2"
  local candidate="$3"
  local seed="$4"
  local args="$5"
  local suffix="stage2_clothing_noalpha_joint_${phase}_${candidate}_seed${seed}_${RUN_TAG}"
  local train_dir="${OUT_DIR}/logs/${phase}/${candidate}"
  local train_log="${train_dir}/seed${seed}.log"
  local eval_dir="${OUT_DIR}/eval/${phase}/${candidate}"
  local eval_log="${eval_dir}/seed${seed}_test.log"
  mkdir -p "${train_dir}" "${eval_dir}"

  local epochs early_stop eva_interval strict_probe_interval
  if [[ "${phase}" == "search" ]]; then
    epochs="${SEARCH_EPOCHS}"
    early_stop="${SEARCH_EARLY_STOP}"
    eva_interval="${SEARCH_EVA_INTERVAL}"
    strict_probe_interval=0
  else
    epochs="${FULL_EPOCHS}"
    early_stop="${FULL_EARLY_STOP}"
    eva_interval="${FULL_EVA_INTERVAL}"
    strict_probe_interval=10
  fi

  local imputer_ckpt base_log base_ckpt fusion_ckpt
  imputer_ckpt="$(imputer_ckpt_for_seed "${seed}")"
  base_log="$(base_log_for_seed "${seed}")"
  base_ckpt="$(base_ckpt_for_seed "${seed}")"
  for required in "configs/clothing/stage2_decoder_mm.yaml" "${imputer_ckpt}" "${base_log}" "${base_ckpt}"; do
    [[ -f "${required}" ]] || { echo "missing input: ${required}" >&2; return 1; }
  done

  if [[ ! -f "${train_log}" ]] || ! grep -q "best epoch" "${train_log}" 2>/dev/null; then
    log "train phase=${phase} candidate=${candidate} seed=${seed} gpu=${gpu}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "dry train ${candidate} seed=${seed}" > "${train_log}"
      echo "best epoch 0" >> "${train_log}"
    else
      read -r -a extra_args <<< "${args}"
      CUDA_VISIBLE_DEVICES="${gpu}" .venv/bin/python main.py \
        --config configs/clothing/stage2_decoder_mm.yaml \
        --suffix "${suffix}" \
        --dataset clothing \
        --exp_mode mm \
        --device_id 0 \
        --seed "${seed}" \
        --dataset_seed 0 \
        --disable_imputation 0 \
        --feature_bridge_mode raw_decoder \
        --gcn_frontend_mode original_linear \
        --imputer_ckpt "${imputer_ckpt}" \
        --epoch "${epochs}" \
        --early_stop "${early_stop}" \
        --eva_interval "${eva_interval}" \
        --penalty_coeff 1.0 \
        --max_info_coeff 0.01 \
        --min_info_coeff 0.000001 \
        --modality_bpr_coeff 0.2 \
        --evaluation_protocol strict \
        --selection_mode val \
        --strict_probe_test_interval "${strict_probe_interval}" \
        --save 1 \
        "${extra_args[@]}" \
        2>&1 | tee "${train_log}"
    fi
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'alpha\thr20\trecall20\tndcg20\n1.0000\t0.00000\t0.00000\t0.00000\n' > "${eval_log}"
    return
  fi

  fusion_ckpt="$(latest_ckpt_for_suffix "${suffix}")"
  [[ -f "${fusion_ckpt}" ]] || { echo "missing fusion checkpoint for ${suffix}" >&2; return 1; }

  if [[ -f "${eval_log}" ]] && grep -q '^1\.0000' "${eval_log}" 2>/dev/null; then
    log "skip eval phase=${phase} candidate=${candidate} seed=${seed}"
    return
  fi

  log "eval phase=${phase} candidate=${candidate} seed=${seed} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" .venv/bin/python tools/evaluate_assemble_score_fusion.py \
    --base-log "${base_log}" \
    --base-ckpt "${base_ckpt}" \
    --fusion-log "${train_log}" \
    --fusion-ckpt "${fusion_ckpt}" \
    --device-id 0 \
    --split test \
    --normalize zscore \
    --alphas "${ASSEMBLE_ALPHAS}" \
    > "${eval_log}" 2>&1
}

write_tasks() {
  local phase="$1"
  local tasks="${OUT_DIR}/tasks/${phase}.tsv"
  : > "${tasks}"
  if [[ "${phase}" == "search" ]]; then
    while IFS=$'\t' read -r candidate args; do
      for seed in "${SEARCH_SEEDS[@]}"; do
        printf '%s\t%s\t%s\t%s\n' "${phase}" "${candidate}" "${seed}" "${args}" >> "${tasks}"
      done
    done < <(candidate_table)
  else
    local selected="${OUT_DIR}/selected_candidate.txt"
    local full_candidates=()
    if [[ -n "${FULL_CANDIDATES_STR}" ]]; then
      read -r -a full_candidates <<< "${FULL_CANDIDATES_STR}"
    else
      [[ -f "${selected}" ]] || { echo "missing selected candidate: ${selected}" >&2; return 1; }
      full_candidates=("$(< "${selected}")")
    fi

    local candidate args
    for candidate in "${full_candidates[@]}"; do
      args="$(candidate_table | awk -F '\t' -v n="${candidate}" '$1 == n {print $2}')"
      [[ -n "${args}" ]] || { echo "failed to resolve selected args for ${candidate}" >&2; return 1; }
      for seed in "${FULL_SEEDS[@]}"; do
        printf '%s\t%s\t%s\t%s\n' "${phase}" "${candidate}" "${seed}" "${args}" >> "${tasks}"
      done
    done
  fi
}

worker() {
  local gpu="$1"
  local file="$2"
  while IFS=$'\t' read -r phase candidate seed args; do
    [[ -n "${phase}" ]] || continue
    run_task "${gpu}" "${phase}" "${candidate}" "${seed}" "${args}"
  done < "${file}"
}

run_phase() {
  local phase="$1"
  local tasks="${OUT_DIR}/tasks/${phase}.tsv"
  write_tasks "${phase}"
  for idx in "${!GPUS[@]}"; do
    : > "${OUT_DIR}/tasks/${phase}_gpu${idx}.tsv"
  done

  local line_no=0
  while IFS= read -r line || [[ -n "${line}" ]]; do
    local slot=$((line_no % ${#GPUS[@]}))
    printf '%s\n' "${line}" >> "${OUT_DIR}/tasks/${phase}_gpu${slot}.tsv"
    line_no=$((line_no + 1))
  done < "${tasks}"

  log "${phase} queued ${line_no} tasks on GPUs: ${GPUS_STR}"
  local pids=()
  for idx in "${!GPUS[@]}"; do
    worker "${GPUS[$idx]}" "${OUT_DIR}/tasks/${phase}_gpu${idx}.tsv" \
      > "${OUT_DIR}/tasks/${phase}_gpu${idx}.worker.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
}

summarize_phase() {
  local phase="$1"
  .venv/bin/python - "${OUT_DIR}" "${phase}" "${PRIMARY_ALPHA}" <<'PY'
from pathlib import Path
import statistics
import sys

out_dir = Path(sys.argv[1])
phase = sys.argv[2]
primary_alpha = float(sys.argv[3])
target_r = 0.073

def parse_eval(path):
    rows = {}
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) == 4:
            try:
                alpha = float(parts[0])
                recall = float(parts[2])
                ndcg = float(parts[3])
            except ValueError:
                continue
            rows[alpha] = (recall, ndcg)
    return rows

records = []
phase_dir = out_dir / "eval" / phase
for candidate_dir in sorted(phase_dir.iterdir() if phase_dir.exists() else []):
    if not candidate_dir.is_dir():
        continue
    vals = []
    details = []
    for eval_log in sorted(candidate_dir.glob("seed*_test.log")):
        rows = parse_eval(eval_log)
        if primary_alpha not in rows:
            continue
        recall, ndcg = rows[primary_alpha]
        vals.append((recall, ndcg))
        details.append(f"{eval_log.stem}:{recall:.5f}/{ndcg:.5f}")
    if vals:
        recalls = [x[0] for x in vals]
        ndcgs = [x[1] for x in vals]
        records.append(
            (
                statistics.mean(recalls),
                statistics.mean(ndcgs),
                candidate_dir.name,
                len(vals),
                ", ".join(details),
            )
        )

records.sort(key=lambda x: (x[0], x[1]), reverse=True)
summary_path = out_dir / f"{phase}_summary.tsv"
agg_path = out_dir / f"{phase}_agg.tsv"
with summary_path.open("w") as fh:
    fh.write("rank\tcandidate\tseeds\trecall20\tndcg20\trecall_gap\tseed_scores\n")
    for idx, (recall, ndcg, candidate, count, details) in enumerate(records, 1):
        fh.write(
            f"{idx}\t{candidate}\t{count}\t{recall:.5f}\t{ndcg:.5f}\t{recall-target_r:.5f}\t{details}\n"
        )
with agg_path.open("w") as fh:
    fh.write("candidate\trecall20\tndcg20\tseeds\n")
    for recall, ndcg, candidate, count, _details in records:
        fh.write(f"{candidate}\t{recall:.5f}\t{ndcg:.5f}\t{count}\n")

if records:
    best = records[0][2]
    (out_dir / "selected_candidate.txt").write_text(best + "\n")
    print(best)
PY
}

if [[ "${DRY_RUN}" == "1" ]]; then
  log "run_tag=${RUN_TAG} out=${OUT_DIR}"
  log "dry run only"
fi

if [[ "${SKIP_SEARCH}" != "1" ]]; then
  log "run_tag=${RUN_TAG} out=${OUT_DIR}"
  run_phase search
  summarize_phase search >/dev/null
else
  log "run_tag=${RUN_TAG} out=${OUT_DIR}"
  log "skip search phase"
fi

run_phase full
summarize_phase full >/dev/null
log "done run_tag=${RUN_TAG}"
