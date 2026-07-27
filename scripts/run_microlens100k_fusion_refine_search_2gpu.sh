#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PATH="${ROOT_DIR}/.venv/bin:${PATH}"
export PYTHONUNBUFFERED=1
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

RUN_TAG="${RUN_TAG:-microlens100k_fusion_refine_$(date +%Y%m%d_%H%M%S)}"
GPU_IDS_STR="${GPU_IDS:-3 4}"
SEARCH_SEEDS_STR="${SEARCH_SEEDS:-1 12}"
DATASET_SEED="${DATASET_SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
read -r -a SEARCH_SEEDS <<< "${SEARCH_SEEDS_STR}"

if [[ "${#GPU_IDS[@]}" -lt 1 ]]; then
  echo "Set GPU_IDS to at least one GPU id" >&2
  exit 1
fi

log() {
  date +"[microlens100k-refine] %Y-%m-%d %H:%M:%S $*"
}

candidate_table() {
  cat <<'EOF'
rr_a022_m035_reg1	--fusion_mode mean --completion_gate_mode rank_residual --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_residual_alpha 0.22 --completion_gate_mix_alpha 0.35 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 1.0 --recommender_allow_modal_grad 0
rr_a025_m025_reg1	--fusion_mode mean --completion_gate_mode rank_residual --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_residual_alpha 0.25 --completion_gate_mix_alpha 0.25 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 1.0 --recommender_allow_modal_grad 0
rr_a025_m035_reg01	--fusion_mode mean --completion_gate_mode rank_residual --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_residual_alpha 0.25 --completion_gate_mix_alpha 0.35 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 0.1 --recommender_allow_modal_grad 0
rr_a025_m050_reg1	--fusion_mode mean --completion_gate_mode rank_residual --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_residual_alpha 0.25 --completion_gate_mix_alpha 0.50 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 1.0 --recommender_allow_modal_grad 0
rr_a030_m035_reg1	--fusion_mode mean --completion_gate_mode rank_residual --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_residual_alpha 0.30 --completion_gate_mix_alpha 0.35 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 1.0 --recommender_allow_modal_grad 0
rr_a030_m050_reg1	--fusion_mode mean --completion_gate_mode rank_residual --completion_gate_hidden_dim 64 --completion_gate_dropout 0.1 --completion_gate_init_logit 0.0 --completion_gate_detach_inputs 1 --completion_gate_use_item_context 1 --completion_gate_item_context_source shared_mean --completion_gate_residual_alpha 0.30 --completion_gate_mix_alpha 0.50 --completion_gate_identity_coeff 0.05 --completion_gate_balance_coeff 0.01 --completion_gate_reg_coeff 1.0 --recommender_allow_modal_grad 0
EOF
}

seed_tag() {
  local seed="$1"
  if [[ "${seed}" == "1" ]]; then
    echo "mmrec_microlens100k_3modal_seed1_base_20260523_012808"
  else
    echo "mmrec_microlens100k_3modal_seed${seed}_mbpr25_20260523_033613"
  fi
}

find_best_ckpt_from_log() {
  local log_path="$1"
  local stage_dir="$2"
  python - "$log_path" "$stage_dir" <<'PY'
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

ckpt_for_seed() {
  local seed="$1"
  local tag
  tag="$(seed_tag "${seed}")"
  local stage_dir="exp_report/microlens100k/stage1_2_microlens100k_imputer_backprop_decoder_v2_${tag}"
  local pipeline_log="exp_report/microlens100k/pipeline_reports/${tag}_raw_decoder_mm/stage1_2.log"
  local seed_log="exp_report/microlens100k/mmrec_seed_logs/mmrec_microlens100k_3modal_mbpr25_5seed_20260523_033613/logs/seed${seed}_stage1_2.log"
  if [[ -f "${pipeline_log}" ]]; then
    find_best_ckpt_from_log "${pipeline_log}" "${stage_dir}"
  else
    find_best_ckpt_from_log "${seed_log}" "${stage_dir}"
  fi
}

run_one() {
  local gpu="$1"
  local candidate="$2"
  local seed="$3"
  local args="$4"
  local ckpt
  ckpt="$(ckpt_for_seed "${seed}")"

  local out_dir="${ROOT_DIR}/exp_report/microlens100k/fusion_search/${RUN_TAG}/search/${candidate}"
  local suffix="stage2_microlens100k_fusion_search_${candidate}_dseed${DATASET_SEED}_seed${seed}_${RUN_TAG}"
  local log_path="${out_dir}/seed${seed}.log"
  mkdir -p "${out_dir}"

  if [[ -f "${log_path}" ]] && grep -q 'best epoch' "${log_path}"; then
    log "skip existing candidate=${candidate} seed=${seed}"
    return
  fi

  log "run candidate=${candidate} seed=${seed} gpu=${gpu}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DEVICE_ID=${gpu} SEED=${seed} SUFFIX=${suffix} ${args}" | tee "${log_path}"
    return
  fi

  read -r -a extra_args <<< "${args}"
  (
    CONFIG="configs/microlens100k/stage2_decoder_mm.yaml" \
    DATASET="microlens100k" \
    EXP_MODE=mm \
    DATASET_SEED="${DATASET_SEED}" \
    SEED="${seed}" \
    DEVICE_ID="${gpu}" \
    USE_GPU=1 \
    TENSORBOARD=0 \
    SAVE=1 \
    IMPUTER_CKPT="${ckpt}" \
    SUFFIX="${suffix}" \
    EPOCHS="${MICROLENS100K_SEARCH_EPOCHS:-200}" \
    EVA_INTERVAL="${MICROLENS100K_EVA_INTERVAL:-10}" \
    EARLY_STOP="${MICROLENS100K_SEARCH_EARLY_STOP:-200}" \
    BATCH_SIZE="${MICROLENS100K_BATCH_SIZE:-2048}" \
    LR="${MICROLENS100K_LR:-0.01}" \
    LR_REC="${MICROLENS100K_LR_REC:-0.01}" \
    LR_IMP=0.0002 \
    LR_DECODER=0.00005 \
    STRICT_PROBE_TEST_INTERVAL=0 \
    ./run_stage2_baby_recommender_decoder.sh "${extra_args[@]}"
  ) 2>&1 | tee "${log_path}"
}

summarize_logs() {
  local out_path="${ROOT_DIR}/exp_report/microlens100k/fusion_search/${RUN_TAG}/search_summary.tsv"
  python - "${RUN_TAG}" "${out_path}" <<'PY'
import pathlib
import re
import statistics
import sys

run_tag, out_path = sys.argv[1], pathlib.Path(sys.argv[2])
base = pathlib.Path("exp_report/microlens100k/fusion_search") / run_tag / "search"
ansi = re.compile(r"\x1b\[[0-9;]*m")
rows = []
for log_path in sorted(base.glob("*/seed*.log")):
    candidate = log_path.parent.name
    seed_match = re.search(r"seed(\d+)\.log$", log_path.name)
    seed = seed_match.group(1) if seed_match else "unknown"
    text = ansi.sub("", log_path.read_text(errors="ignore"))
    bests = re.findall(r"best epoch\s+(\d+)", text)
    if not bests:
        rows.append([candidate, seed, "", "", "", "", "", "missing_best"])
        continue
    best_epoch = bests[-1]
    metric_matches = re.findall(
        r"hr@20\s*=\s*([0-9.]+),\s*recall@20\s*=\s*([0-9.]+),\s*ndcg@20\s*=\s*([0-9.]+)",
        text,
    )
    if not metric_matches:
        rows.append([candidate, seed, best_epoch, "", "", "", "", "missing_metric"])
        continue
    _, val_r20, val_n20 = metric_matches[-1]
    final_matches = re.findall(
        r"final strict test hr@20\s*=\s*([0-9.]+),\s*recall@20\s*=\s*([0-9.]+),\s*ndcg@20\s*=\s*([0-9.]+)",
        text,
    )
    test_r20 = test_n20 = ""
    if final_matches:
        _, test_r20, test_n20 = final_matches[-1]
    rows.append([candidate, seed, best_epoch, val_r20, val_n20, test_r20, test_n20, "ok"])

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    f.write("candidate\tseed\tbest_epoch\tval_R20\tval_N20\ttest_R20\ttest_N20\tstatus\n")
    for row in rows:
        f.write("\t".join(map(str, row)) + "\n")

agg = {}
for row in rows:
    if row[-1] == "ok":
        agg.setdefault(row[0], []).append(row)
ranked = []
for candidate, vals in agg.items():
    val_r = [float(r[3]) for r in vals]
    val_n = [float(r[4]) for r in vals]
    test_r = [float(r[5]) for r in vals if r[5]]
    test_n = [float(r[6]) for r in vals if r[6]]
    ranked.append((
        statistics.mean(val_r),
        statistics.mean(val_n),
        candidate,
        len(vals),
        statistics.mean(test_r) if test_r else None,
        statistics.mean(test_n) if test_n else None,
    ))
ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
agg_path = out_path.with_name("search_summary_agg.tsv")
with agg_path.open("w") as f:
    f.write("rank\tcandidate\tok_count\tmean_val_R20\tmean_val_N20\tmean_test_R20\tmean_test_N20\n")
    for idx, (mean_r, mean_n, candidate, count, mean_tr, mean_tn) in enumerate(ranked, 1):
        f.write(
            f"{idx}\t{candidate}\t{count}\t{mean_r:.5f}\t{mean_n:.5f}\t"
            f"{'' if mean_tr is None else f'{mean_tr:.5f}'}\t{'' if mean_tn is None else f'{mean_tn:.5f}'}\n"
        )
if ranked:
    out_path.with_name("search_best_candidate.txt").write_text(ranked[0][2] + "\n")
    print(ranked[0][2])
else:
    print("no_completed_logs")
PY
}

declare -a TASK_CANDIDATES=()
declare -a TASK_SEEDS=()
declare -a TASK_ARGS=()

while IFS=$'\t' read -r candidate args; do
  for seed in "${SEARCH_SEEDS[@]}"; do
    TASK_CANDIDATES+=("${candidate}")
    TASK_SEEDS+=("${seed}")
    TASK_ARGS+=("${args}")
  done
done < <(candidate_table)

run_worker() {
  local worker_idx="$1"
  local gpu="$2"
  local idx
  for idx in "${!TASK_CANDIDATES[@]}"; do
    if (( idx % ${#GPU_IDS[@]} == worker_idx )); then
      run_one "${gpu}" "${TASK_CANDIDATES[$idx]}" "${TASK_SEEDS[$idx]}" "${TASK_ARGS[$idx]}"
    fi
  done
}

log "start run_tag=${RUN_TAG} gpus=${GPU_IDS_STR} seeds=${SEARCH_SEEDS_STR}"
for worker_idx in "${!GPU_IDS[@]}"; do
  run_worker "${worker_idx}" "${GPU_IDS[$worker_idx]}" &
done
wait

best="$(summarize_logs | tail -1)"
log "search done run_tag=${RUN_TAG} best=${best}"
log "summary=exp_report/microlens100k/fusion_search/${RUN_TAG}/search_summary_agg.tsv"
