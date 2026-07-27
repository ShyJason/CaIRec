#!/usr/bin/env bash
set -euo pipefail

dataset="${1:?dataset required: clothing|sports}"
gpu="${2:?gpu id required}"
stamp="${3:-allgate_5seed_rest_20260528}"
seeds="${SEEDS:-1 1234 12345}"

cd "$(dirname "$0")/.."

outer_dir="exp_report/fusion_norm_ablation/${stamp}"
eval_dir="exp_report/fusion_norm_ablation/late_score_eval_${stamp}"
mkdir -p "${outer_dir}" "${eval_dir}"

case "${dataset}" in
  clothing)
    config="configs/clothing/stage2_decoder_mm.yaml"
    gate_reg="1.0"
    ;;
  sports)
    config="configs/sports/stage2_decoder_mm.yaml"
    gate_reg="0.1"
    ;;
  *)
    echo "unsupported dataset: ${dataset}" >&2
    exit 2
    ;;
esac

imputer_ckpt_for_seed() {
  local seed="$1"
  case "${dataset}" in
    clothing)
      echo "exp_report/clothing/stage1_2_clothing_mmrec_fixed_seed${seed}_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage1_2_clothing_mmrec_fixed_seed${seed}_mmrec_clothing_mm_fixedmissing_20260521_052129_imputer_backprop_50_epoch19.pth"
      ;;
    sports)
      echo "exp_report/sports/stage1_2_sports_imputer_backprop_decoder_v2_mmrec_sports_mm_mr0.3_seed${seed}_mmrec_sports_mm_fixedmissing_20260524_165817/ckpt/stage1_2_sports_imputer_backprop_decoder_v2_mmrec_sports_mm_mr0.3_seed${seed}_mmrec_sports_mm_fixedmissing_20260524_165817_imputer_backprop_50_epoch19.pth"
      ;;
  esac
}

base_log_for_seed() {
  local seed="$1"
  case "${dataset}:${seed}" in
    clothing:1) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed1_mmrec_clothing_mm_fixedmissing_20260521_052129/log/run_20260521_052413.log" ;;
    clothing:1234) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed1234_mmrec_clothing_mm_fixedmissing_20260521_052129/log/run_20260521_055340.log" ;;
    clothing:12345) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed12345_mmrec_clothing_mm_fixedmissing_20260521_052129/log/run_20260521_063053.log" ;;
    sports:1) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed1_mmrec_sports_mm_fixedmissing_20260524_165817/log/run_20260524_165955.log" ;;
    sports:1234) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed1234_mmrec_sports_mm_fixedmissing_20260524_165817/log/run_20260524_232628.log" ;;
    sports:12345) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed12345_mmrec_sports_mm_fixedmissing_20260524_165817/log/run_20260525_012732.log" ;;
    *) echo "missing base log mapping for ${dataset} seed ${seed}" >&2; return 2 ;;
  esac
}

base_ckpt_for_seed() {
  local seed="$1"
  case "${dataset}:${seed}" in
    clothing:1) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed1_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage2_clothing_mmrec_fixed_seed1_mmrec_clothing_mm_fixedmissing_20260521_052129_recommender_1.0_epoch169.pth" ;;
    clothing:1234) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed1234_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage2_clothing_mmrec_fixed_seed1234_mmrec_clothing_mm_fixedmissing_20260521_052129_recommender_1.0_epoch137.pth" ;;
    clothing:12345) echo "exp_report/clothing/stage2_clothing_mmrec_fixed_seed12345_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage2_clothing_mmrec_fixed_seed12345_mmrec_clothing_mm_fixedmissing_20260521_052129_recommender_1.0_epoch121.pth" ;;
    sports:1) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed1_mmrec_sports_mm_fixedmissing_20260524_165817/ckpt/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed1_mmrec_sports_mm_fixedmissing_20260524_165817_recommender_50_epoch158.pth" ;;
    sports:1234) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed1234_mmrec_sports_mm_fixedmissing_20260524_165817/ckpt/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed1234_mmrec_sports_mm_fixedmissing_20260524_165817_recommender_50_epoch148.pth" ;;
    sports:12345) echo "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed12345_mmrec_sports_mm_fixedmissing_20260524_165817/ckpt/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed12345_mmrec_sports_mm_fixedmissing_20260524_165817_recommender_50_epoch196.pth" ;;
    *) echo "missing base ckpt mapping for ${dataset} seed ${seed}" >&2; return 2 ;;
  esac
}

run_train_seed() {
  local seed="$1"
  local suffix="stage2_${dataset}_rankres_allgate_seed${seed}_${stamp}"
  local log_path="${outer_dir}/${dataset}_seed${seed}.log"
  local imputer_ckpt
  imputer_ckpt="$(imputer_ckpt_for_seed "${seed}")"

  if [[ ! -f "${imputer_ckpt}" ]]; then
    echo "missing imputer checkpoint: ${imputer_ckpt}" >&2
    exit 1
  fi

  if [[ -f "${log_path}" ]] && grep -q "best epoch" "${log_path}"; then
    echo "[allgate-rest] skip finished train dataset=${dataset} seed=${seed}"
    return
  fi

  echo "[allgate-rest] train dataset=${dataset} seed=${seed} gpu=${gpu} suffix=${suffix}"
  CUDA_VISIBLE_DEVICES="${gpu}" .venv/bin/python main.py \
    --config "${config}" \
    --suffix "${suffix}" \
    --dataset "${dataset}" \
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
    --early_stop 20 \
    --eva_interval 1 \
    --evaluation_protocol strict \
    --selection_mode val \
    --strict_probe_test_interval 10 \
    --save 1 \
    --completion_gate_mode rank_residual_allgate \
    --completion_gate_hidden_dim 64 \
    --completion_gate_dropout 0.1 \
    --completion_gate_init_logit 0.0 \
    --completion_gate_detach_inputs 1 \
    --completion_gate_use_item_context 1 \
    --completion_gate_item_context_source shared_mean \
    --completion_gate_residual_alpha 0.18 \
    --completion_gate_mix_alpha 0.35 \
    --completion_gate_identity_coeff 0.05 \
    --completion_gate_balance_coeff 0.01 \
    --completion_gate_reg_coeff "${gate_reg}" \
    --recommender_allow_modal_grad 0 \
    2>&1 | tee "${log_path}"
}

latest_fusion_ckpt() {
  local seed="$1"
  find "exp_report/${dataset}/stage2_${dataset}_rankres_allgate_seed${seed}_${stamp}/ckpt" \
    -maxdepth 1 -name '*.pth' -print | sort -V | tail -1
}

eval_seed() {
  local seed="$1"
  local out_log="${eval_dir}/${dataset}_seed${seed}_test.log"
  if [[ -f "${out_log}" ]] && grep -q '^1\\.0000' "${out_log}"; then
    echo "[allgate-rest] skip finished eval dataset=${dataset} seed=${seed}"
    return
  fi

  local base_log base_ckpt fusion_log fusion_ckpt
  base_log="$(base_log_for_seed "${seed}")"
  base_ckpt="$(base_ckpt_for_seed "${seed}")"
  fusion_log="${outer_dir}/${dataset}_seed${seed}.log"
  fusion_ckpt="$(latest_fusion_ckpt "${seed}")"

  for required in "${base_log}" "${base_ckpt}" "${fusion_log}" "${fusion_ckpt}"; do
    if [[ ! -f "${required}" ]]; then
      echo "missing eval input: ${required}" >&2
      exit 1
    fi
  done

  echo "[allgate-rest] eval dataset=${dataset} seed=${seed} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" .venv/bin/python tools/evaluate_late_score_fusion.py \
    --base-log "${base_log}" \
    --base-ckpt "${base_ckpt}" \
    --fusion-log "${fusion_log}" \
    --fusion-ckpt "${fusion_ckpt}" \
    --device-id 0 \
    --split test \
    --normalize zscore \
    --alphas 0,0.2,0.3,0.35,0.4,0.45,0.5,0.6,1 \
    > "${out_log}" 2>&1
}

write_summary() {
  .venv/bin/python - "${eval_dir}" "${dataset}" ${seeds} <<'PY'
from pathlib import Path
import statistics
import sys

out = Path(sys.argv[1])
dataset = sys.argv[2]
seeds = sys.argv[3:]

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

base_r = []; base_n = []; aux_r = []; aux_n = []; fus_r = []; fus_n = []
lines = [
    f"# {dataset} all-modal normalized gate remaining seeds",
    "",
    "Aux mode: `rank_residual_allgate`.",
    "Score fusion: `score_final = (1-alpha) * zscore(score_base) + alpha * zscore(score_aux)`.",
    "Primary comparison uses `alpha=0.4`.",
    "",
    "| seed | base R@20 | base N@20 | aux R@20 | aux N@20 | alpha0.4 R@20 | alpha0.4 N@20 | R gain | N gain |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for seed in seeds:
    rows = parse(out / f"{dataset}_seed{seed}_test.log")
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
(out / f"{dataset}_summary.md").write_text("\n".join(lines))
print(out / f"{dataset}_summary.md")
PY
}

for seed in ${seeds}; do
  run_train_seed "${seed}"
  eval_seed "${seed}"
done

write_summary
