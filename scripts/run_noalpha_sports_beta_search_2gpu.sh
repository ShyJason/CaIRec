#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PATH="${ROOT_DIR}/.venv/bin:${PATH}"
export PYTHONUNBUFFERED=1
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

RUN_TAG="${RUN_TAG:-noalpha_sports_beta_$(date -u +%Y%m%d_%H%M%S)}"
GPUS_STR="${GPUS:-2 3}"
SEARCH_SEEDS_STR="${SEARCH_SEEDS:-12 123}"
FULL_SEEDS_STR="${FULL_SEEDS:-1 12 123 1234 12345}"
BETAS_STR="${BETAS:-0.00 0.02 0.05 0.08 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.50 0.60 0.70 0.85 1.00}"
DATASET_SEED="${DATASET_SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_SEARCH="${SKIP_SEARCH:-0}"
FULL_CANDIDATES_STR="${FULL_CANDIDATES:-}"

SEARCH_EPOCHS="${SEARCH_EPOCHS:-80}"
SEARCH_EARLY_STOP="${SEARCH_EARLY_STOP:-10}"
FULL_EPOCHS="${FULL_EPOCHS:-200}"
FULL_EARLY_STOP="${FULL_EARLY_STOP:-20}"
EVA_INTERVAL="${EVA_INTERVAL:-1}"

GATE_REG="${GATE_REG:-0.1}"
IDENTITY_COEFF="${IDENTITY_COEFF:-0.05}"
BALANCE_COEFF="${BALANCE_COEFF:-0.01}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
DROPOUT="${DROPOUT:-0.1}"
ITEM_CONTEXT_SOURCE="${ITEM_CONTEXT_SOURCE:-shared_mean}"

ASSEMBLE_ALPHAS="${ASSEMBLE_ALPHAS:-1}"
PRIMARY_ALPHA="${PRIMARY_ALPHA:-1}"
TARGET_RECALL="${TARGET_RECALL:-0.09918}"
TARGET_NDCG="${TARGET_NDCG:-0.04460}"

read -r -a GPUS <<< "${GPUS_STR}"
read -r -a SEARCH_SEEDS <<< "${SEARCH_SEEDS_STR}"
read -r -a FULL_SEEDS <<< "${FULL_SEEDS_STR}"
read -r -a BETAS <<< "${BETAS_STR}"

if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "GPUS must contain at least one GPU id" >&2
  exit 2
fi

OUT_DIR="${ROOT_DIR}/exp_report/noalpha_sports_beta_search/${RUN_TAG}"
mkdir -p "${OUT_DIR}/tasks" "${OUT_DIR}/logs/search" "${OUT_DIR}/logs/full" "${OUT_DIR}/eval/search" "${OUT_DIR}/eval/full"

log() {
  date +"[noalpha-sports-beta] %Y-%m-%d %H:%M:%S $*"
}

beta_tag() {
  printf '%s' "${1/./p}"
}

candidate_args() {
  local beta="$1"
  local ctx_args=("--completion_gate_use_item_context" "1" "--completion_gate_item_context_source" "${ITEM_CONTEXT_SOURCE}")
  if [[ "${ITEM_CONTEXT_SOURCE}" == "off" ]]; then
    ctx_args=("--completion_gate_use_item_context" "0" "--completion_gate_item_context_source" "off")
  fi

  local args=(
    --completion_gate_mode rank_residual_allgate
    --completion_gate_no_residual_alpha 1
    --completion_gate_hidden_dim "${HIDDEN_DIM}"
    --completion_gate_dropout "${DROPOUT}"
    --completion_gate_init_logit 0.0
    --completion_gate_detach_inputs 1
    "${ctx_args[@]}"
    --completion_gate_mix_alpha "${beta}"
    --completion_gate_identity_coeff "${IDENTITY_COEFF}"
    --completion_gate_balance_coeff "${BALANCE_COEFF}"
    --completion_gate_reg_coeff "${GATE_REG}"
    --recommender_allow_modal_grad 0
  )
  printf '%q ' "${args[@]}"
}

candidate_table() {
  local beta tag
  for beta in "${BETAS[@]}"; do
    tag="$(beta_tag "${beta}")"
    printf 'b%s_g%s_i%s_bal%s_h%s_d%s_%s\t%s\n' \
      "${tag}" \
      "$(beta_tag "${GATE_REG}")" \
      "$(beta_tag "${IDENTITY_COEFF}")" \
      "$(beta_tag "${BALANCE_COEFF}")" \
      "${HIDDEN_DIM}" \
      "$(beta_tag "${DROPOUT}")" \
      "${ITEM_CONTEXT_SOURCE}" \
      "$(candidate_args "${beta}")"
  done
}

imputer_ckpt_for_seed() {
  local seed="$1"
  echo "exp_report/sports/stage1_2_sports_imputer_backprop_decoder_v2_mmrec_sports_mm_mr0.3_seed${seed}_mmrec_sports_mm_fixedmissing_20260524_165817/ckpt/stage1_2_sports_imputer_backprop_decoder_v2_mmrec_sports_mm_mr0.3_seed${seed}_mmrec_sports_mm_fixedmissing_20260524_165817_imputer_backprop_50_epoch19.pth"
}

base_log_for_seed() {
  local seed="$1"
  case "${seed}" in
    1) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed1_mmrec_sports_mm_fixedmissing_20260524_165817/log/run_20260524_165955.log" ;;
    12) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed12_mmrec_sports_mm_fixedmissing_20260524_165817/log/run_20260524_190838.log" ;;
    123) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed123_mmrec_sports_mm_fixedmissing_20260524_165817/log/run_20260524_211736.log" ;;
    1234) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed1234_mmrec_sports_mm_fixedmissing_20260524_165817/log/run_20260524_232628.log" ;;
    12345) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed12345_mmrec_sports_mm_fixedmissing_20260524_165817/log/run_20260525_012732.log" ;;
    *) echo "missing base log mapping for sports seed ${seed}" >&2; return 2 ;;
  esac
}

base_ckpt_for_seed() {
  local seed="$1"
  case "${seed}" in
    1) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed1_mmrec_sports_mm_fixedmissing_20260524_165817/ckpt/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed1_mmrec_sports_mm_fixedmissing_20260524_165817_recommender_50_epoch158.pth" ;;
    12) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed12_mmrec_sports_mm_fixedmissing_20260524_165817/ckpt/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed12_mmrec_sports_mm_fixedmissing_20260524_165817_recommender_50_epoch160.pth" ;;
    123) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed123_mmrec_sports_mm_fixedmissing_20260524_165817/ckpt/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed123_mmrec_sports_mm_fixedmissing_20260524_165817_recommender_50_epoch158.pth" ;;
    1234) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed1234_mmrec_sports_mm_fixedmissing_20260524_165817/ckpt/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed1234_mmrec_sports_mm_fixedmissing_20260524_165817_recommender_50_epoch148.pth" ;;
    12345) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed12345_mmrec_sports_mm_fixedmissing_20260524_165817/ckpt/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed12345_mmrec_sports_mm_fixedmissing_20260524_165817_recommender_50_epoch196.pth" ;;
    *) echo "missing base ckpt mapping for sports seed ${seed}" >&2; return 2 ;;
  esac
}

latest_ckpt_for_suffix() {
  local suffix="$1"
  find "exp_report/sports/${suffix}/ckpt" -maxdepth 1 -name '*.pth' -print 2>/dev/null | sort -V | tail -1
}

run_task() {
  local gpu="$1"
  local phase="$2"
  local candidate="$3"
  local seed="$4"
  local args="$5"

  local suffix train_dir train_log eval_dir eval_log imputer_ckpt base_log base_ckpt fusion_ckpt
  suffix="stage2_sports_noalpha_beta_${phase}_${candidate}_seed${seed}_${RUN_TAG}"
  train_dir="${OUT_DIR}/logs/${phase}/${candidate}"
  train_log="${train_dir}/seed${seed}.log"
  eval_dir="${OUT_DIR}/eval/${phase}/${candidate}"
  eval_log="${eval_dir}/seed${seed}_test.log"
  mkdir -p "${train_dir}" "${eval_dir}"

  imputer_ckpt="$(imputer_ckpt_for_seed "${seed}")"
  base_log="$(base_log_for_seed "${seed}")"
  base_ckpt="$(base_ckpt_for_seed "${seed}")"
  for required in "configs/sports/stage2_decoder_mm.yaml" "${imputer_ckpt}" "${base_log}" "${base_ckpt}"; do
    [[ -f "${required}" ]] || { echo "missing input: ${required}" >&2; return 1; }
  done

  if [[ -f "${eval_log}" ]] && grep -q '^1\.0000' "${eval_log}" 2>/dev/null; then
    log "skip finished phase=${phase} candidate=${candidate} seed=${seed}"
    return
  fi

  local epochs early_stop
  if [[ "${phase}" == "search" ]]; then
    epochs="${SEARCH_EPOCHS}"
    early_stop="${SEARCH_EARLY_STOP}"
  else
    epochs="${FULL_EPOCHS}"
    early_stop="${FULL_EARLY_STOP}"
  fi

  if [[ ! -f "${train_log}" ]] || ! grep -q "best epoch" "${train_log}" 2>/dev/null; then
    log "train phase=${phase} candidate=${candidate} seed=${seed} beta_args='${args}' gpu=${gpu}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "dry train ${candidate} seed=${seed}" > "${train_log}"
    else
      read -r -a gate_args <<< "${args}"
      CUDA_VISIBLE_DEVICES="${gpu}" .venv/bin/python main.py \
        --config configs/sports/stage2_decoder_mm.yaml \
        --suffix "${suffix}" \
        --dataset sports \
        --exp_mode mm \
        --device_id 0 \
        --seed "${seed}" \
        --dataset_seed "${DATASET_SEED}" \
        --train_stage recommender \
        --freeze_imputer 1 \
        --freeze_decoder 1 \
        --disable_imputation 0 \
        --feature_bridge_mode raw_decoder \
        --gcn_frontend_mode original_linear \
        --imputer_ckpt "${imputer_ckpt}" \
        --epoch "${epochs}" \
        --early_stop "${early_stop}" \
        --eva_interval "${EVA_INTERVAL}" \
        --batch_size 256 \
        --lr 0.001 \
        --lr_rec 0.001 \
        --lr_imp 0.0002 \
        --lr_decoder 0.00005 \
        --reg_coeff 0.0001 \
        --penalty_coeff 50 \
        --max_info_coeff 0.05 \
        --min_info_coeff 0.05 \
        --evaluation_protocol strict \
        --selection_mode val \
        --strict_probe_test_interval 10 \
        --save 1 \
        "${gate_args[@]}" \
        2>&1 | tee "${train_log}"
    fi
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'alpha\thr20\trecall20\tndcg20\n1.0000\t0.00000\t0.00000\t0.00000\n' > "${eval_log}"
    return
  fi

  fusion_ckpt="$(latest_ckpt_for_suffix "${suffix}")"
  [[ -f "${fusion_ckpt}" ]] || { echo "missing fusion checkpoint for ${suffix}" >&2; return 1; }

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
    local candidates=()
    if [[ -n "${FULL_CANDIDATES_STR}" ]]; then
      read -r -a candidates <<< "${FULL_CANDIDATES_STR}"
    else
      local selected="${OUT_DIR}/selected_candidate.txt"
      [[ -f "${selected}" ]] || { echo "missing selected candidate: ${selected}" >&2; return 1; }
      candidates=("$(< "${selected}")")
    fi
    local candidate args
    for candidate in "${candidates[@]}"; do
      args="$(candidate_table | awk -F '\t' -v n="${candidate}" '$1 == n {print $2}')"
      [[ -n "${args}" ]] || { echo "failed to resolve args for ${candidate}" >&2; return 1; }
      for seed in "${FULL_SEEDS[@]}"; do
        printf '%s\t%s\t%s\t%s\n' "${phase}" "${candidate}" "${seed}" "${args}" >> "${tasks}"
      done
    done
  fi
}

worker() {
  local gpu="$1"
  local tasks="$2"
  while IFS=$'\t' read -r phase candidate seed args; do
    [[ -n "${phase}" ]] || continue
    run_task "${gpu}" "${phase}" "${candidate}" "${seed}" "${args}"
  done < "${tasks}"
}

run_phase() {
  local phase="$1"
  write_tasks "${phase}"
  for idx in "${!GPUS[@]}"; do
    : > "${OUT_DIR}/tasks/${phase}_gpu${idx}.tsv"
  done
  local line_no=0
  while IFS= read -r line; do
    local slot=$((line_no % ${#GPUS[@]}))
    printf '%s\n' "${line}" >> "${OUT_DIR}/tasks/${phase}_gpu${slot}.tsv"
    line_no=$((line_no + 1))
  done < "${OUT_DIR}/tasks/${phase}.tsv"

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
  .venv/bin/python - "${OUT_DIR}" "${phase}" "${PRIMARY_ALPHA}" "${TARGET_RECALL}" "${TARGET_NDCG}" <<'PY'
from pathlib import Path
import statistics
import sys

out_dir = Path(sys.argv[1])
phase = sys.argv[2]
primary_alpha = float(sys.argv[3])
target_r = float(sys.argv[4])
target_n = float(sys.argv[5])

def parse_eval(path: Path):
    rows = {}
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) != 4:
            continue
        try:
            alpha = float(parts[0])
            rows[alpha] = (float(parts[2]), float(parts[3]))
        except ValueError:
            pass
    return rows

rows = []
for log_path in sorted((out_dir / "eval" / phase).glob("*/seed*_test.log")):
    candidate = log_path.parent.name
    seed = log_path.stem.replace("seed", "").replace("_test", "")
    parsed = parse_eval(log_path)
    if primary_alpha not in parsed:
        rows.append([candidate, seed, "", "", "missing_eval"])
        continue
    recall, ndcg = parsed[primary_alpha]
    rows.append([candidate, seed, recall, ndcg, "ok"])

summary = out_dir / f"{phase}_summary.tsv"
with summary.open("w", encoding="utf-8") as f:
    f.write("candidate\tseed\tR20\tN20\tstatus\n")
    for row in rows:
        f.write("\t".join(map(str, row)) + "\n")

by_candidate = {}
for candidate, seed, recall, ndcg, status in rows:
    if status != "ok":
        continue
    by_candidate.setdefault(candidate, []).append((seed, float(recall), float(ndcg)))

ranked = []
for candidate, vals in by_candidate.items():
    recall = statistics.mean(v[1] for v in vals)
    ndcg = statistics.mean(v[2] for v in vals)
    ranked.append((recall, ndcg, candidate, vals))
ranked.sort(reverse=True)

agg_path = out_dir / f"{phase}_agg.tsv"
with agg_path.open("w", encoding="utf-8") as f:
    f.write("rank\tcandidate\tseeds\tR20\tN20\ttarget_R20\ttarget_N20\tR_gap\tN_gap\tseed_scores\n")
    for rank, (recall, ndcg, candidate, vals) in enumerate(ranked, 1):
        scores = ", ".join(f"{seed}:{r:.5f}/{n:.5f}" for seed, r, n in sorted(vals))
        f.write(
            f"{rank}\t{candidate}\t{len(vals)}\t{recall:.5f}\t{ndcg:.5f}\t"
            f"{target_r:.5f}\t{target_n:.5f}\t{recall - target_r:.5f}\t{ndcg - target_n:.5f}\t{scores}\n"
        )

if ranked and phase == "search":
    (out_dir / "selected_candidate.txt").write_text(ranked[0][2] + "\n", encoding="utf-8")
PY
}

log "run_tag=${RUN_TAG} out=${OUT_DIR}"
log "fixed: gate_reg=${GATE_REG} identity=${IDENTITY_COEFF} balance=${BALANCE_COEFF} hidden=${HIDDEN_DIM} dropout=${DROPOUT} ctx=${ITEM_CONTEXT_SOURCE}"
if [[ "${SKIP_SEARCH}" != "1" ]]; then
  run_phase search
  summarize_phase search
else
  log "skip search phase"
fi
run_phase full
summarize_phase full
log "done run_tag=${RUN_TAG}"
