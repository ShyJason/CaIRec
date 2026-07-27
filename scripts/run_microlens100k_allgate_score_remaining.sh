#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

gpu="${GPU:-2}"
stamp="${STAMP:-microlens100k_allgate_score_5seed_rest_$(date +%Y%m%d_%H%M%S)}"
seeds="${SEEDS:-1 1234 12345}"
outer_dir="exp_report/fusion_norm_ablation/${stamp}"
eval_dir="exp_report/fusion_norm_ablation/late_score_eval_${stamp}"
mkdir -p "${outer_dir}" "${eval_dir}"

imputer_ckpt_for_seed() {
  local seed="$1"
  case "${seed}" in
    1)
      echo "exp_report/microlens100k/stage1_2_microlens100k_imputer_backprop_decoder_v2_mmrec_microlens100k_3modal_seed1_base_20260523_012808/ckpt/stage1_2_microlens100k_imputer_backprop_decoder_v2_mmrec_microlens100k_3modal_seed1_base_20260523_012808_imputer_backprop_50_epoch11.pth"
      ;;
    1234)
      echo "exp_report/microlens100k/stage1_2_microlens100k_imputer_backprop_decoder_v2_mmrec_microlens100k_3modal_seed1234_mbpr25_20260523_033613/ckpt/stage1_2_microlens100k_imputer_backprop_decoder_v2_mmrec_microlens100k_3modal_seed1234_mbpr25_20260523_033613_imputer_backprop_50_epoch15.pth"
      ;;
    12345)
      echo "exp_report/microlens100k/stage1_2_microlens100k_imputer_backprop_decoder_v2_mmrec_microlens100k_3modal_seed12345_mbpr25_20260523_033613/ckpt/stage1_2_microlens100k_imputer_backprop_decoder_v2_mmrec_microlens100k_3modal_seed12345_mbpr25_20260523_033613_imputer_backprop_50_epoch11.pth"
      ;;
    *)
      echo "unsupported seed: ${seed}" >&2
      return 2
      ;;
  esac
}

run_stage2() {
  local seed="$1"
  local mode="$2"
  local imputer_ckpt="$3"
  local suffix="stage2_microlens100k_${mode}_seed${seed}_${stamp}"
  local log_path="${outer_dir}/microlens100k_${mode}_seed${seed}.log"

  if [[ -f "${log_path}" ]] && grep -q "best epoch" "${log_path}"; then
    echo "[microlens100k-rest] skip finished seed=${seed} mode=${mode}"
    return
  fi

  echo "[microlens100k-rest] run seed=${seed} mode=${mode} gpu=${gpu} suffix=${suffix}"
  local gate_args=()
  if [[ "${mode}" == "allgate" ]]; then
    gate_args=(
      --completion_gate_mode rank_residual_allgate
      --completion_gate_hidden_dim 64
      --completion_gate_dropout 0.1
      --completion_gate_init_logit 0.0
      --completion_gate_detach_inputs 1
      --completion_gate_use_item_context 1
      --completion_gate_item_context_source shared_mean
      --completion_gate_residual_alpha 0.18
      --completion_gate_mix_alpha 0.35
      --completion_gate_identity_coeff 0.05
      --completion_gate_balance_coeff 0.01
      --completion_gate_reg_coeff 1.0
      --recommender_allow_modal_grad 0
    )
  else
    gate_args=(--completion_gate_mode off)
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" .venv/bin/python main.py \
    --config configs/microlens100k/stage2_decoder_mm.yaml \
    --suffix "${suffix}" \
    --dataset microlens100k \
    --exp_mode mm \
    --device_id 0 \
    --seed "${seed}" \
    --dataset_seed 0 \
    --train_stage recommender \
    --freeze_imputer 1 \
    --freeze_decoder 1 \
    --disable_imputation 0 \
    --feature_bridge_mode raw_decoder \
    --gcn_frontend_mode original_linear \
    --imputer_ckpt "${imputer_ckpt}" \
    --epoch 200 \
    --early_stop 200 \
    --eva_interval 10 \
    --batch_size 2048 \
    --lr 0.01 \
    --lr_rec 0.01 \
    --lr_imp 0.0002 \
    --lr_decoder 0.00005 \
    --reg_coeff 0.01 \
    --penalty_coeff 1.0 \
    --max_info_coeff 0.01 \
    --min_info_coeff 0.000001 \
    --modality_bpr_coeff 2.5 \
    --evaluation_protocol strict \
    --selection_mode val \
    --strict_probe_test_interval 0 \
    --save 1 \
    "${gate_args[@]}" \
    2>&1 | tee "${log_path}"
}

latest_ckpt() {
  local suffix="$1"
  find "exp_report/microlens100k/${suffix}/ckpt" -maxdepth 1 -name '*.pth' -print | sort -V | tail -1
}

eval_seed() {
  local seed="$1"
  local out_log="${eval_dir}/microlens100k_seed${seed}_test.log"
  if [[ -f "${out_log}" ]] && grep -q '^1\.0000' "${out_log}"; then
    echo "[microlens100k-rest] skip finished eval seed=${seed}"
    return
  fi

  local base_suffix="stage2_microlens100k_base_seed${seed}_${stamp}"
  local gate_suffix="stage2_microlens100k_allgate_seed${seed}_${stamp}"
  local base_log="${outer_dir}/microlens100k_base_seed${seed}.log"
  local gate_log="${outer_dir}/microlens100k_allgate_seed${seed}.log"
  local base_ckpt gate_ckpt
  base_ckpt="$(latest_ckpt "${base_suffix}")"
  gate_ckpt="$(latest_ckpt "${gate_suffix}")"

  for required in "${base_log}" "${gate_log}" "${base_ckpt}" "${gate_ckpt}"; do
    if [[ ! -f "${required}" ]]; then
      echo "missing eval input: ${required}" >&2
      exit 1
    fi
  done

  echo "[microlens100k-rest] eval seed=${seed}"
  CUDA_VISIBLE_DEVICES="${gpu}" .venv/bin/python tools/evaluate_late_score_fusion.py \
    --base-log "${base_log}" \
    --base-ckpt "${base_ckpt}" \
    --fusion-log "${gate_log}" \
    --fusion-ckpt "${gate_ckpt}" \
    --device-id 0 \
    --split test \
    --normalize zscore \
    --alphas 0,0.2,0.3,0.35,0.4,0.45,0.5,0.6,1 \
    > "${out_log}" 2>&1
}

write_summary() {
  .venv/bin/python - "${eval_dir}" ${seeds} <<'PY'
from pathlib import Path
import statistics
import sys

out = Path(sys.argv[1])
seeds = sys.argv[2:]

def parse(path):
    rows = {}
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) == 4:
            try:
                rows[float(parts[0])] = (float(parts[2]), float(parts[3]))
            except ValueError:
                pass
    return rows

base_r, base_n, aux_r, aux_n, fus_r, fus_n = [], [], [], [], [], []
lines = [
    "# MicroLens-100k all-modal normalized gate remaining seeds",
    "",
    "Aux mode: `rank_residual_allgate`.",
    "Score fusion: `score_final = (1-alpha) * zscore(score_base) + alpha * zscore(score_aux)`.",
    "Primary comparison uses `alpha=0.4`.",
    "",
    "| seed | base R@20 | base N@20 | aux R@20 | aux N@20 | alpha0.4 R@20 | alpha0.4 N@20 | R gain | N gain |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for seed in seeds:
    rows = parse(out / f"microlens100k_seed{seed}_test.log")
    br, bn = rows[0.0]
    ar, an = rows[1.0]
    fr, fn = rows[0.4]
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
    f"- Alpha0.4 mean: R@20 {statistics.mean(fus_r):.5f}, N@20 {statistics.mean(fus_n):.5f}",
    f"- Mean gain: R@20 {(statistics.mean(fus_r) / statistics.mean(base_r) - 1) * 100:.2f}%, "
    f"N@20 {(statistics.mean(fus_n) / statistics.mean(base_n) - 1) * 100:.2f}%",
    "",
]
(out / "summary.md").write_text("\n".join(lines))
print(out / "summary.md")
PY
}

for seed in ${seeds}; do
  imputer_ckpt="$(imputer_ckpt_for_seed "${seed}")"
  if [[ ! -f "${imputer_ckpt}" ]]; then
    echo "missing imputer checkpoint: ${imputer_ckpt}" >&2
    exit 1
  fi
  run_stage2 "${seed}" base "${imputer_ckpt}"
  run_stage2 "${seed}" allgate "${imputer_ckpt}"
  eval_seed "${seed}"
done

write_summary
