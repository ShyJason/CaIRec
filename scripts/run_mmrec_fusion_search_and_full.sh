#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PATH="${ROOT_DIR}/.venv/bin:${PATH}"
export PYTHONUNBUFFERED=1
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

RUN_TAG="${RUN_TAG:-fusion_search_full_$(date +%Y%m%d_%H%M%S)}"
WAIT_SESSION="${WAIT_SESSION:-}"
DEVICE_ID="${DEVICE_ID:-1}"
DATASET_SEED="${DATASET_SEED:-0}"
SEARCH_SEEDS_STR="${SEARCH_SEEDS:-1 12}"
FULL_SEEDS_STR="${FULL_SEEDS:-1 12 123 1234 12345}"
DATASETS_STR="${DATASETS:-sports microlens}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a SEARCH_SEEDS <<< "${SEARCH_SEEDS_STR}"
read -r -a FULL_SEEDS <<< "${FULL_SEEDS_STR}"
read -r -a DATASETS <<< "${DATASETS_STR}"

QUEUE_DIR="${ROOT_DIR}/exp_report/fusion_search/${RUN_TAG}"
mkdir -p "${QUEUE_DIR}"

log() {
  date +"[fusion-search] %Y-%m-%d %H:%M:%S $*"
}

if [[ -n "${WAIT_SESSION}" ]]; then
  log "waiting for tmux session ${WAIT_SESSION}"
  while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
    log "still waiting for ${WAIT_SESSION}"
    sleep 300
  done
  log "wait session finished"
fi

dataset_runtime_params() {
  local dataset="$1"
  case "${dataset}" in
    sports)
      EPOCHS="${SPORTS_EPOCHS:-200}"
      EVA_INTERVAL="${SPORTS_EVA_INTERVAL:-1}"
      EARLY_STOP="${SPORTS_EARLY_STOP:-20}"
      BATCH_SIZE="${SPORTS_BATCH_SIZE:-256}"
      LR="${SPORTS_LR:-0.001}"
      LR_REC="${SPORTS_LR_REC:-0.001}"
      STRICT_PROBE_TEST_INTERVAL="${SPORTS_STRICT_PROBE_TEST_INTERVAL:-10}"
      ;;
    microlens)
      EPOCHS="${MICROLENS_EPOCHS:-60}"
      EVA_INTERVAL="${MICROLENS_EVA_INTERVAL:-30}"
      EARLY_STOP="${MICROLENS_EARLY_STOP:-60}"
      BATCH_SIZE="${MICROLENS_BATCH_SIZE:-2048}"
      LR="${MICROLENS_LR:-0.01}"
      LR_REC="${MICROLENS_LR_REC:-0.01}"
      STRICT_PROBE_TEST_INTERVAL="${MICROLENS_STRICT_PROBE_TEST_INTERVAL:-0}"
      ;;
    microlens100k)
      EPOCHS="${MICROLENS100K_EPOCHS:-200}"
      EVA_INTERVAL="${MICROLENS100K_EVA_INTERVAL:-10}"
      EARLY_STOP="${MICROLENS100K_EARLY_STOP:-200}"
      BATCH_SIZE="${MICROLENS100K_BATCH_SIZE:-2048}"
      LR="${MICROLENS100K_LR:-0.01}"
      LR_REC="${MICROLENS100K_LR_REC:-0.01}"
      STRICT_PROBE_TEST_INTERVAL="${MICROLENS100K_STRICT_PROBE_TEST_INTERVAL:-0}"
      ;;
    *)
      echo "unsupported dataset: ${dataset}" >&2
      return 1
      ;;
  esac
}

phase_runtime_params() {
  local dataset="$1"
  local phase="$2"
  dataset_runtime_params "${dataset}"

  if [[ "${phase}" == "search" ]]; then
    case "${dataset}" in
      sports)
        EPOCHS="${SPORTS_SEARCH_EPOCHS:-80}"
        EARLY_STOP="${SPORTS_SEARCH_EARLY_STOP:-15}"
        ;;
      microlens)
        EPOCHS="${MICROLENS_SEARCH_EPOCHS:-60}"
        EARLY_STOP="${MICROLENS_SEARCH_EARLY_STOP:-30}"
        ;;
      microlens100k)
        EPOCHS="${MICROLENS100K_SEARCH_EPOCHS:-200}"
        EARLY_STOP="${MICROLENS100K_SEARCH_EARLY_STOP:-200}"
        ;;
    esac
  fi
}

candidate_table() {
  cat <<'EOF'
rr_a018_m035_reg1	--fusion_mode mean --completion_gate_mode rank_residual --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_residual_alpha 0.18 --completion_gate_mix_alpha 0.35 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 1.0 --recommender_allow_modal_grad 0
rr_a010_m035_reg1	--fusion_mode mean --completion_gate_mode rank_residual --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_residual_alpha 0.10 --completion_gate_mix_alpha 0.35 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 1.0 --recommender_allow_modal_grad 0
rr_a025_m035_reg1	--fusion_mode mean --completion_gate_mode rank_residual --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_residual_alpha 0.25 --completion_gate_mix_alpha 0.35 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 1.0 --recommender_allow_modal_grad 0
rr_a018_m020_reg1	--fusion_mode mean --completion_gate_mode rank_residual --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_residual_alpha 0.18 --completion_gate_mix_alpha 0.20 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 1.0 --recommender_allow_modal_grad 0
rr_a018_m050_reg1	--fusion_mode mean --completion_gate_mode rank_residual --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_residual_alpha 0.18 --completion_gate_mix_alpha 0.50 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 1.0 --recommender_allow_modal_grad 0
rr_a018_m035_reg01	--fusion_mode mean --completion_gate_mode rank_residual --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_residual_alpha 0.18 --completion_gate_mix_alpha 0.35 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 0.1 --recommender_allow_modal_grad 0
EOF
}

find_best_ckpt() {
  local report_dir="$1"
  local stage_dir="$2"
  find_best_ckpt_from_log "${report_dir}/stage1_2.log" "${stage_dir}"
}

find_best_ckpt_from_log() {
  local log_path_arg="$1"
  local stage_dir="$2"
  python - "$log_path_arg" "$stage_dir" <<'PY'
import pathlib
import re
import sys

log_path = pathlib.Path(sys.argv[1])
stage_dir = pathlib.Path(sys.argv[2])
if not log_path.exists():
    raise SystemExit(f"missing stage1_2 log: {log_path}")
text = log_path.read_text(errors="ignore")
matches = re.findall(r"best epoch\s+(\d+)", text)
if not matches:
    raise SystemExit(f"no best epoch in {log_path}")
epoch = int(matches[-1])
ckpts = sorted((stage_dir / "ckpt").glob(f"*_epoch{epoch}.pth"))
if not ckpts:
    raise SystemExit(f"no checkpoint for epoch {epoch} in {stage_dir / 'ckpt'}")
print(ckpts[-1])
PY
}

seed_tag() {
  local dataset="$1"
  local seed="$2"
  case "${dataset}" in
    sports)
      echo "mmrec_sports_mm_mr0.3_seed${seed}_mmrec_sports_mm_fixedmissing_20260524_165817"
      ;;
    microlens)
      if [[ "${seed}" == "1" || "${seed}" == "12" ]]; then
        echo "mmrec_microlens_mm_2seed_20260521_153006_seed${seed}"
      else
        echo "mmrec_microlens_tuned_5seed_20260522_023007_seed${seed}"
      fi
      ;;
    microlens100k)
      if [[ "${seed}" == "1" ]]; then
        echo "mmrec_microlens100k_3modal_seed1_base_20260523_012808"
      else
        echo "mmrec_microlens100k_3modal_seed${seed}_mbpr25_20260523_033613"
      fi
      ;;
    *)
      return 1
      ;;
  esac
}

ckpt_for_seed() {
  local dataset="$1"
  local seed="$2"
  local tag
  tag="$(seed_tag "${dataset}" "${seed}")"
  if [[ "${dataset}" == "microlens100k" ]]; then
    local stage_dir="exp_report/${dataset}/stage1_2_${dataset}_imputer_backprop_decoder_v2_${tag}"
    local pipeline_log="exp_report/${dataset}/pipeline_reports/${tag}_raw_decoder_mm/stage1_2.log"
    local seed_log="exp_report/${dataset}/mmrec_seed_logs/mmrec_microlens100k_3modal_mbpr25_5seed_20260523_033613/logs/seed${seed}_stage1_2.log"
    if [[ -f "${pipeline_log}" ]]; then
      find_best_ckpt_from_log "${pipeline_log}" "${stage_dir}"
    else
      find_best_ckpt_from_log "${seed_log}" "${stage_dir}"
    fi
    return
  fi
  find_best_ckpt \
    "exp_report/${dataset}/pipeline_reports/${tag}_raw_decoder_mm" \
    "exp_report/${dataset}/stage1_2_${dataset}_imputer_backprop_decoder_v2_${tag}"
}

run_one() {
  local dataset="$1"
  local phase="$2"
  local candidate="$3"
  local seed="$4"
  local ckpt="$5"
  local args="$6"

  phase_runtime_params "${dataset}" "${phase}"
  local out_dir="${ROOT_DIR}/exp_report/${dataset}/fusion_search/${RUN_TAG}/${phase}/${candidate}"
  local suffix="stage2_${dataset}_fusion_${phase}_${candidate}_dseed${DATASET_SEED}_seed${seed}_${RUN_TAG}"
  local log_path="${out_dir}/seed${seed}.log"
  mkdir -p "${out_dir}"

  if [[ -f "${log_path}" ]] && grep -q 'best epoch' "${log_path}"; then
    log "skip existing dataset=${dataset} phase=${phase} candidate=${candidate} seed=${seed}"
    return
  fi

  log "run dataset=${dataset} phase=${phase} candidate=${candidate} seed=${seed} gpu=${DEVICE_ID}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "CONFIG=configs/${dataset}/stage2_decoder_mm.yaml DATASET=${dataset} DATASET_SEED=${DATASET_SEED} SEED=${seed} SUFFIX=${suffix} ${args}" | tee "${log_path}"
    return
  fi

  read -r -a extra_args <<< "${args}"
  (
    CONFIG="configs/${dataset}/stage2_decoder_mm.yaml" \
    DATASET="${dataset}" \
    EXP_MODE=mm \
    DATASET_SEED="${DATASET_SEED}" \
    SEED="${seed}" \
    DEVICE_ID="${DEVICE_ID}" \
    USE_GPU=1 \
    TENSORBOARD=0 \
    SAVE=1 \
    IMPUTER_CKPT="${ckpt}" \
    SUFFIX="${suffix}" \
    EPOCHS="${EPOCHS}" \
    EVA_INTERVAL="${EVA_INTERVAL}" \
    EARLY_STOP="${EARLY_STOP}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    LR="${LR}" \
    LR_REC="${LR_REC}" \
    LR_IMP=0.0002 \
    LR_DECODER=0.00005 \
    STRICT_PROBE_TEST_INTERVAL="${STRICT_PROBE_TEST_INTERVAL}" \
    ./run_stage2_baby_recommender_decoder.sh "${extra_args[@]}"
  ) 2>&1 | tee "${log_path}"
}

summarize_logs() {
  local dataset="$1"
  local phase="$2"
  local out_path="${ROOT_DIR}/exp_report/${dataset}/fusion_search/${RUN_TAG}/${phase}_summary.tsv"
  python - "$dataset" "$RUN_TAG" "$phase" "$out_path" <<'PY'
import pathlib
import re
import statistics
import sys

dataset, run_tag, phase, out_path = sys.argv[1], sys.argv[2], sys.argv[3], pathlib.Path(sys.argv[4])
base = pathlib.Path("exp_report") / dataset / "fusion_search" / run_tag / phase
ansi = re.compile(r"\x1b\[[0-9;]*m")
rows = []
for log_path in sorted(base.glob("*/seed*.log")):
    candidate = log_path.parent.name
    seed = re.search(r"seed(\d+)\.log$", log_path.name)
    seed = seed.group(1) if seed else "unknown"
    text = ansi.sub("", log_path.read_text(errors="ignore"))
    bests = re.findall(r"best epoch\s+(\d+)", text)
    if not bests:
        rows.append([candidate, seed, "", "", "", "", "", "missing_best"])
        continue
    best_epoch = bests[-1]
    val_re = re.compile(
        rf"epoch\s*=\s*{best_epoch}\s+hr@20\s*=\s*([0-9.]+),\s*recall@20\s*=\s*([0-9.]+),\s*ndcg@20\s*=\s*([0-9.]+)"
    )
    val_matches = val_re.findall(text)
    metric_matches = re.findall(
        r"hr@20\s*=\s*([0-9.]+),\s*recall@20\s*=\s*([0-9.]+),\s*ndcg@20\s*=\s*([0-9.]+)",
        text,
    )
    if val_matches:
        _, val_r20, val_n20 = val_matches[-1]
    elif metric_matches:
        _, val_r20, val_n20 = metric_matches[-1]
    else:
        rows.append([candidate, seed, best_epoch, "", "", "", "", "missing_metric"])
        continue
    test_r20 = test_n20 = ""
    final_matches = re.findall(
        r"final strict test hr@20\s*=\s*([0-9.]+),\s*recall@20\s*=\s*([0-9.]+),\s*ndcg@20\s*=\s*([0-9.]+)",
        text,
    )
    if final_matches:
        _, test_r20, test_n20 = final_matches[-1]
    rows.append([candidate, seed, best_epoch, val_r20, val_n20, test_r20, test_n20, "ok"])

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    f.write("candidate\tseed\tbest_epoch\tval_R20\tval_N20\ttest_R20\ttest_N20\tstatus\n")
    for row in rows:
        f.write("\t".join(map(str, row)) + "\n")

ok = [r for r in rows if r[-1] == "ok"]
agg = {}
for row in ok:
    agg.setdefault(row[0], []).append(row)
ranked = []
for candidate, vals in agg.items():
    val_r = [float(r[3]) for r in vals]
    val_n = [float(r[4]) for r in vals]
    test_r = [float(r[5]) for r in vals if r[5]]
    test_n = [float(r[6]) for r in vals if r[6]]
    ranked.append(
        (
            statistics.mean(val_r),
            statistics.mean(val_n),
            candidate,
            len(vals),
            statistics.mean(test_r) if test_r else None,
            statistics.mean(test_n) if test_n else None,
        )
    )
ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
agg_path = out_path.with_name(out_path.stem + "_agg.tsv")
with agg_path.open("w") as f:
    f.write("rank\tcandidate\tok_count\tmean_val_R20\tmean_val_N20\tmean_test_R20\tmean_test_N20\n")
    for idx, (mean_r, mean_n, candidate, count, mean_tr, mean_tn) in enumerate(ranked, 1):
        f.write(
            f"{idx}\t{candidate}\t{count}\t{mean_r:.5f}\t{mean_n:.5f}\t"
            f"{'' if mean_tr is None else f'{mean_tr:.5f}'}\t{'' if mean_tn is None else f'{mean_tn:.5f}'}\n"
        )
best_path = out_path.with_name(f"{phase}_best_candidate.txt")
if ranked:
    best_path.write_text(ranked[0][2] + "\n")
    print(ranked[0][2])
else:
    raise SystemExit(f"no completed logs for {dataset} {phase}")
PY
}

candidate_args_by_name() {
  local name="$1"
  candidate_table | awk -F '\t' -v n="${name}" '$1 == n {print $2}'
}

for dataset in "${DATASETS[@]}"; do
  log "search start dataset=${dataset} run_tag=${RUN_TAG}"
  while IFS=$'\t' read -r candidate args; do
    for seed in "${SEARCH_SEEDS[@]}"; do
      ckpt="$(ckpt_for_seed "${dataset}" "${seed}")"
      run_one "${dataset}" search "${candidate}" "${seed}" "${ckpt}" "${args}"
    done
  done < <(candidate_table)

  best_candidate="$(summarize_logs "${dataset}" search | tail -1)"
  best_args="$(candidate_args_by_name "${best_candidate}")"
  if [[ -z "${best_args}" ]]; then
    echo "failed to resolve args for best candidate ${best_candidate}" >&2
    exit 1
  fi
  echo "${best_args}" > "exp_report/${dataset}/fusion_search/${RUN_TAG}/selected_args.txt"
  log "search done dataset=${dataset} best=${best_candidate}"

  for seed in "${FULL_SEEDS[@]}"; do
    ckpt="$(ckpt_for_seed "${dataset}" "${seed}")"
    run_one "${dataset}" full "${best_candidate}" "${seed}" "${ckpt}" "${best_args}"
  done
  summarize_logs "${dataset}" full >/dev/null
  log "full done dataset=${dataset} best=${best_candidate}"
done

log "all done run_tag=${RUN_TAG}"
