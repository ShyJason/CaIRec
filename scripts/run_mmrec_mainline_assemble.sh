#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
Usage:
  scripts/run_mmrec_mainline_assemble.sh DATASET GPU [STAMP]

Legacy retired experiment:
  1. Train/evaluate the imputation-based base recommender.
  2. Train/evaluate the all-modal normalized gate auxiliary recommender.
  3. Assemble final scores:
       score_final = (1-alpha) * zscore(score_base) + alpha * zscore(score_aux)

This score-assembly path is no longer the MMRec mainline. The current mainline
is the raw-decoder stage2 recommender with the completed-feature item-item graph
modal adapter and no recommendation-side CL.
Set ALLOW_LEGACY_ASSEMBLE=1 only when intentionally reproducing old results.

Required for new runs:
  IMPUTER_CKPT_TEMPLATE   Path template for seed-specific imputer checkpoints.
                          Use {seed} as placeholder.

Optional:
  SEEDS                   Default: "1 12 123 1234 12345"
  CONFIG                  Default: configs/$DATASET/stage2_decoder_mm.yaml
  RUN_BASE                Default: 1. Set 0 to reuse existing base logs/ckpts.
  RUN_AUX                 Default: 1. Set 0 to reuse existing aux logs/ckpts.
  EVAL                    Default: 1.
  BASE_LOG_TEMPLATE       Required when RUN_BASE=0. Use {seed}.
  BASE_CKPT_TEMPLATE      Required when RUN_BASE=0. Use {seed}.
  AUX_LOG_TEMPLATE        Optional when RUN_AUX=0. Use {seed}.
  AUX_CKPT_TEMPLATE       Optional when RUN_AUX=0. Use {seed}.
  AUX_GATE_MODE           Default: rank_residual_allgate.
  ASSEMBLE_ALPHA          Default: 0.4.
  ASSEMBLE_NORMALIZE      Default: zscore.
  ALPHAS                  Default: 0,0.2,0.3,0.35,0.4,0.45,0.5,0.6,1.
  EXTRA_BASE_ARGS         Extra args appended to the base recommender command.
  EXTRA_AUX_ARGS          Extra args appended to the aux recommender command.

All fusion/gate variants should be passed explicitly through
AUX_GATE_MODE and EXTRA_AUX_ARGS, or run through a separate ablation script.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 2 ]]; then
  usage
  exit 0
fi

if [[ "${ALLOW_LEGACY_ASSEMBLE:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
[legacy-assemble] This score-assembly script is retired and is not part of the
current MMRec mainline. Use the raw-decoder stage2 recommender with
item_graph_kind=fused_completed, item_graph_modal_alpha=0.25, and
rec_neighbor_cl_weight=0.005 instead.

To intentionally reproduce old assemble results, rerun with:
  ALLOW_LEGACY_ASSEMBLE=1 scripts/run_mmrec_mainline_assemble.sh ...
EOF
  exit 2
fi

dataset="$1"
gpu="$2"
stamp="${3:-mmrec_legacy_assemble_${dataset}_$(date +%Y%m%d_%H%M%S)}"

seeds="${SEEDS:-1 12 123 1234 12345}"
config="${CONFIG:-configs/${dataset}/stage2_decoder_mm.yaml}"
run_base="${RUN_BASE:-1}"
run_aux="${RUN_AUX:-1}"
run_eval="${EVAL:-1}"
imputer_ckpt_template="${IMPUTER_CKPT_TEMPLATE:-}"
base_log_template="${BASE_LOG_TEMPLATE:-}"
base_ckpt_template="${BASE_CKPT_TEMPLATE:-}"
aux_log_template="${AUX_LOG_TEMPLATE:-}"
aux_ckpt_template="${AUX_CKPT_TEMPLATE:-}"

epoch="${EPOCH:-200}"
early_stop="${EARLY_STOP:-200}"
eva_interval="${EVA_INTERVAL:-10}"
batch_size="${BATCH_SIZE:-2048}"
lr="${LR:-0.01}"
lr_rec="${LR_REC:-${lr}}"
lr_imp="${LR_IMP:-0.0002}"
lr_decoder="${LR_DECODER:-0.00005}"
reg_coeff="${REG_COEFF:-0.01}"
penalty_coeff="${PENALTY_COEFF:-1.0}"
max_info_coeff="${MAX_INFO_COEFF:-0.01}"
min_info_coeff="${MIN_INFO_COEFF:-0.000001}"
modality_bpr_coeff="${MODALITY_BPR_COEFF:-2.5}"
dataset_seed="${DATASET_SEED:-0}"

aux_gate_mode="${AUX_GATE_MODE:-rank_residual_allgate}"
gate_hidden_dim="${GATE_HIDDEN_DIM:-64}"
gate_dropout="${GATE_DROPOUT:-0.1}"
gate_init_logit="${GATE_INIT_LOGIT:-0.0}"
gate_detach_inputs="${GATE_DETACH_INPUTS:-1}"
gate_use_item_context="${GATE_USE_ITEM_CONTEXT:-1}"
gate_item_context_source="${GATE_ITEM_CONTEXT_SOURCE:-shared_mean}"
gate_residual_alpha="${GATE_RESIDUAL_ALPHA:-0.18}"
gate_mix_alpha="${GATE_MIX_ALPHA:-0.35}"
gate_identity_coeff="${GATE_IDENTITY_COEFF:-0.05}"
gate_balance_coeff="${GATE_BALANCE_COEFF:-0.01}"
gate_reg_coeff="${GATE_REG_COEFF:-1.0}"
recommender_allow_modal_grad="${RECOMMENDER_ALLOW_MODAL_GRAD:-0}"

assemble_alpha="${ASSEMBLE_ALPHA:-0.4}"
assemble_normalize="${ASSEMBLE_NORMALIZE:-zscore}"
alphas="${ALPHAS:-0,0.2,0.3,0.35,0.4,0.45,0.5,0.6,1}"

outer_dir="exp_report/legacy_assemble/${stamp}"
eval_dir="${outer_dir}/eval"
mkdir -p "${outer_dir}" "${eval_dir}"

if [[ ! -f "${config}" ]]; then
  echo "missing config: ${config}" >&2
  exit 1
fi

render_template() {
  local template="$1"
  local seed="$2"
  if [[ -z "${template}" ]]; then
    return 1
  fi
  echo "${template//\{seed\}/${seed}}"
}

latest_ckpt_for_suffix() {
  local suffix="$1"
  find "exp_report/${dataset}/${suffix}/ckpt" -maxdepth 1 -name '*.pth' -print | sort -V | tail -1
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "missing ${label}: ${path}" >&2
    exit 1
  fi
}

run_recommender() {
  local seed="$1"
  local mode="$2"
  local imputer_ckpt="$3"
  local suffix="stage2_${dataset}_${mode}_seed${seed}_${stamp}"
  local log_path="${outer_dir}/${dataset}_${mode}_seed${seed}.log"
  local gate_args=()

  if [[ -f "${log_path}" ]] && grep -q "best epoch" "${log_path}"; then
    echo "[mainline-assemble] skip finished dataset=${dataset} seed=${seed} mode=${mode}"
    return
  fi

  if [[ "${mode}" == "aux" ]]; then
    gate_args=(
      --completion_gate_mode "${aux_gate_mode}"
      --completion_gate_hidden_dim "${gate_hidden_dim}"
      --completion_gate_dropout "${gate_dropout}"
      --completion_gate_init_logit "${gate_init_logit}"
      --completion_gate_detach_inputs "${gate_detach_inputs}"
      --completion_gate_use_item_context "${gate_use_item_context}"
      --completion_gate_item_context_source "${gate_item_context_source}"
      --completion_gate_residual_alpha "${gate_residual_alpha}"
      --completion_gate_mix_alpha "${gate_mix_alpha}"
      --completion_gate_identity_coeff "${gate_identity_coeff}"
      --completion_gate_balance_coeff "${gate_balance_coeff}"
      --completion_gate_reg_coeff "${gate_reg_coeff}"
      --recommender_allow_modal_grad "${recommender_allow_modal_grad}"
    )
  else
    gate_args=(--completion_gate_mode off)
  fi

  local extra_args=()
  if [[ "${mode}" == "aux" && -n "${EXTRA_AUX_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_args=(${EXTRA_AUX_ARGS})
  elif [[ "${mode}" == "base" && -n "${EXTRA_BASE_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_args=(${EXTRA_BASE_ARGS})
  fi

  echo "[mainline-assemble] train dataset=${dataset} seed=${seed} mode=${mode} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" .venv/bin/python main.py \
    --config "${config}" \
    --suffix "${suffix}" \
    --dataset "${dataset}" \
    --exp_mode mm \
    --device_id 0 \
    --seed "${seed}" \
    --dataset_seed "${dataset_seed}" \
    --train_stage recommender \
    --freeze_imputer 1 \
    --freeze_decoder 1 \
    --disable_imputation 0 \
    --feature_bridge_mode raw_decoder \
    --gcn_frontend_mode original_linear \
    --imputer_ckpt "${imputer_ckpt}" \
    --epoch "${epoch}" \
    --early_stop "${early_stop}" \
    --eva_interval "${eva_interval}" \
    --batch_size "${batch_size}" \
    --lr "${lr}" \
    --lr_rec "${lr_rec}" \
    --lr_imp "${lr_imp}" \
    --lr_decoder "${lr_decoder}" \
    --reg_coeff "${reg_coeff}" \
    --penalty_coeff "${penalty_coeff}" \
    --max_info_coeff "${max_info_coeff}" \
    --min_info_coeff "${min_info_coeff}" \
    --modality_bpr_coeff "${modality_bpr_coeff}" \
    --evaluation_protocol strict \
    --selection_mode val \
    --strict_probe_test_interval 0 \
    --save 1 \
    "${gate_args[@]}" \
    "${extra_args[@]}" \
    2>&1 | tee "${log_path}"
}

log_for_seed() {
  local seed="$1"
  local mode="$2"
  local suffix="stage2_${dataset}_${mode}_seed${seed}_${stamp}"
  if [[ "${mode}" == "base" && "${run_base}" == "0" ]]; then
    render_template "${base_log_template}" "${seed}"
  elif [[ "${mode}" == "aux" && "${run_aux}" == "0" && -n "${aux_log_template}" ]]; then
    render_template "${aux_log_template}" "${seed}"
  else
    echo "${outer_dir}/${dataset}_${mode}_seed${seed}.log"
  fi
}

ckpt_for_seed() {
  local seed="$1"
  local mode="$2"
  local suffix="stage2_${dataset}_${mode}_seed${seed}_${stamp}"
  if [[ "${mode}" == "base" && "${run_base}" == "0" ]]; then
    render_template "${base_ckpt_template}" "${seed}"
  elif [[ "${mode}" == "aux" && "${run_aux}" == "0" && -n "${aux_ckpt_template}" ]]; then
    render_template "${aux_ckpt_template}" "${seed}"
  else
    latest_ckpt_for_suffix "${suffix}"
  fi
}

eval_seed() {
  local seed="$1"
  local out_log="${eval_dir}/${dataset}_seed${seed}_test.log"
  if [[ -f "${out_log}" ]] && grep -q '^1\.0000' "${out_log}"; then
    echo "[mainline-assemble] skip finished eval dataset=${dataset} seed=${seed}"
    return
  fi

  local base_log base_ckpt aux_log aux_ckpt
  base_log="$(log_for_seed "${seed}" base)"
  base_ckpt="$(ckpt_for_seed "${seed}" base)"
  aux_log="$(log_for_seed "${seed}" aux)"
  aux_ckpt="$(ckpt_for_seed "${seed}" aux)"

  require_file "${base_log}" "base log"
  require_file "${base_ckpt}" "base checkpoint"
  require_file "${aux_log}" "aux log"
  require_file "${aux_ckpt}" "aux checkpoint"

  echo "[mainline-assemble] eval dataset=${dataset} seed=${seed} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" .venv/bin/python tools/evaluate_assemble_score_fusion.py \
    --base-log "${base_log}" \
    --base-ckpt "${base_ckpt}" \
    --fusion-log "${aux_log}" \
    --fusion-ckpt "${aux_ckpt}" \
    --device-id 0 \
    --split test \
    --normalize "${assemble_normalize}" \
    --alphas "${alphas}" \
    > "${out_log}" 2>&1
}

write_summary() {
  .venv/bin/python - "${eval_dir}" "${dataset}" "${assemble_alpha}" ${seeds} <<'PY'
from pathlib import Path
import statistics
import sys

out = Path(sys.argv[1])
dataset = sys.argv[2]
primary_alpha = float(sys.argv[3])
seeds = sys.argv[4:]

def parse(path):
    rows = {}
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) == 4:
            try:
                rows[float(parts[0])] = (float(parts[2]), float(parts[3]))
            except ValueError:
                pass
    for alpha in (0.0, primary_alpha, 1.0):
        if alpha not in rows:
            raise ValueError(f"{path} missing alpha={alpha}")
    return rows

base_r = []; base_n = []; aux_r = []; aux_n = []; fus_r = []; fus_n = []
lines = [
    f"# {dataset} MMRec legacy assemble summary",
    "",
    "Retired experiment: completion base recommender + all-modal gate auxiliary recommender + assemble score fusion.",
    "Aux mode: `rank_residual_allgate` unless `AUX_GATE_MODE` was overridden.",
    f"Score fusion: `score_final = (1-alpha) * zscore(score_base) + alpha * zscore(score_aux)`.",
    f"Primary comparison uses `alpha={primary_alpha:g}`.",
    "",
    "| seed | base R@20 | base N@20 | aux R@20 | aux N@20 | assemble R@20 | assemble N@20 | R gain | N gain |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for seed in seeds:
    rows = parse(out / f"{dataset}_seed{seed}_test.log")
    br, bn = rows[0.0]
    ar, an = rows[1.0]
    fr, fn = rows[primary_alpha]
    base_r.append(br); base_n.append(bn)
    aux_r.append(ar); aux_n.append(an)
    fus_r.append(fr); fus_n.append(fn)
    lines.append(
        f"| {seed} | {br:.5f} | {bn:.5f} | {ar:.5f} | {an:.5f} | {fr:.5f} | {fn:.5f} | "
        f"{(fr / br - 1) * 100:.2f}% | {(fn / bn - 1) * 100:.2f}% |"
    )
lines += [
    "",
    f"- Base mean: R@20 {statistics.mean(base_r):.5f}, N@20 {statistics.mean(base_n):.5f}",
    f"- Aux mean: R@20 {statistics.mean(aux_r):.5f}, N@20 {statistics.mean(aux_n):.5f}",
    f"- Assemble mean: R@20 {statistics.mean(fus_r):.5f}, N@20 {statistics.mean(fus_n):.5f}",
    f"- Mean gain: R@20 {(statistics.mean(fus_r) / statistics.mean(base_r) - 1) * 100:.2f}%, "
    f"N@20 {(statistics.mean(fus_n) / statistics.mean(base_n) - 1) * 100:.2f}%",
    "",
]
(out / "summary.md").write_text("\n".join(lines))
print(out / "summary.md")
PY
}

if [[ "${run_base}" != "0" || "${run_aux}" != "0" ]]; then
  if [[ -z "${imputer_ckpt_template}" ]]; then
    echo "IMPUTER_CKPT_TEMPLATE is required when RUN_BASE=1 or RUN_AUX=1" >&2
    exit 1
  fi
fi

if [[ "${run_base}" == "0" ]]; then
  [[ -n "${base_log_template}" && -n "${base_ckpt_template}" ]] || {
    echo "BASE_LOG_TEMPLATE and BASE_CKPT_TEMPLATE are required when RUN_BASE=0" >&2
    exit 1
  }
fi

for seed in ${seeds}; do
  imputer_ckpt=""
  if [[ -n "${imputer_ckpt_template}" ]]; then
    imputer_ckpt="$(render_template "${imputer_ckpt_template}" "${seed}")"
    require_file "${imputer_ckpt}" "imputer checkpoint"
  fi

  if [[ "${run_base}" != "0" ]]; then
    run_recommender "${seed}" base "${imputer_ckpt}"
  fi
  if [[ "${run_aux}" != "0" ]]; then
    run_recommender "${seed}" aux "${imputer_ckpt}"
  fi
  if [[ "${run_eval}" != "0" ]]; then
    eval_seed "${seed}"
  fi
done

if [[ "${run_eval}" != "0" ]]; then
  write_summary
fi
