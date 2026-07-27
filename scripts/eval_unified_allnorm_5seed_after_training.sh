#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

stamp="unified_allnorm_5seed_20260528"
out="exp_report/fusion_norm_ablation/late_score_eval_20260528_unified_allnorm_5seed"
mkdir -p "${out}"

latest_ckpt() {
  local dataset="$1"
  local seed="$2"
  find "exp_report/${dataset}/stage2_${dataset}_rankres_allnorm_seed${seed}_${stamp}/ckpt" \
    -maxdepth 1 -name '*.pth' -print | sort -V | tail -1
}

eval_one() {
  local dataset="$1"
  local seed="$2"
  local base_log="$3"
  local base_ckpt="$4"
  local fusion_log="exp_report/fusion_norm_ablation/${stamp}/${dataset}_seed${seed}.log"
  local fusion_ckpt
  fusion_ckpt="$(latest_ckpt "${dataset}" "${seed}")"

  if [[ ! -f "${fusion_log}" ]]; then
    echo "[eval] missing fusion log: ${fusion_log}" >&2
    return 1
  fi
  if [[ -z "${fusion_ckpt}" || ! -f "${fusion_ckpt}" ]]; then
    echo "[eval] missing fusion ckpt for ${dataset} seed${seed}" >&2
    return 1
  fi

  echo "[eval] ${dataset} seed${seed} ckpt=${fusion_ckpt}"
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python tools/evaluate_late_score_fusion.py \
    --base-log "${base_log}" \
    --base-ckpt "${base_ckpt}" \
    --fusion-log "${fusion_log}" \
    --fusion-ckpt "${fusion_ckpt}" \
    --device-id 0 \
    --split test \
    --normalize zscore \
    --alphas 0,0.4 \
    > "${out}/${dataset}_seed${seed}_test.log" 2>&1
}

while tmux has-session -t allnorm_seed1234 2>/dev/null || tmux has-session -t allnorm_seed12345 2>/dev/null; do
  date '+[watch] %F %T waiting for allnorm training sessions'
  sleep 300
done

eval_one "clothing" "1234" \
  "exp_report/clothing/stage2_clothing_mmrec_fixed_seed1234_mmrec_clothing_mm_fixedmissing_20260521_052129/log/run_20260521_055340.log" \
  "exp_report/clothing/stage2_clothing_mmrec_fixed_seed1234_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage2_clothing_mmrec_fixed_seed1234_mmrec_clothing_mm_fixedmissing_20260521_052129_recommender_1.0_epoch137.pth"

eval_one "clothing" "12345" \
  "exp_report/clothing/stage2_clothing_mmrec_fixed_seed12345_mmrec_clothing_mm_fixedmissing_20260521_052129/log/run_20260521_063053.log" \
  "exp_report/clothing/stage2_clothing_mmrec_fixed_seed12345_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage2_clothing_mmrec_fixed_seed12345_mmrec_clothing_mm_fixedmissing_20260521_052129_recommender_1.0_epoch121.pth"

eval_one "sports" "1234" \
  "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed1234_mmrec_sports_mm_fixedmissing_20260524_165817/log/run_20260524_232628.log" \
  "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed1234_mmrec_sports_mm_fixedmissing_20260524_165817/ckpt/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed1234_mmrec_sports_mm_fixedmissing_20260524_165817_recommender_50_epoch148.pth"

eval_one "sports" "12345" \
  "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed12345_mmrec_sports_mm_fixedmissing_20260524_165817/log/run_20260525_012732.log" \
  "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed12345_mmrec_sports_mm_fixedmissing_20260524_165817/ckpt/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed12345_mmrec_sports_mm_fixedmissing_20260524_165817_recommender_50_epoch196.pth"

.venv/bin/python - <<'PY'
from pathlib import Path
import math
import statistics

try:
    from scipy import stats
except Exception:
    stats = None

root = Path("exp_report/fusion_norm_ablation")
out = root / "late_score_eval_20260528_unified_allnorm_5seed"
old = root / "late_score_eval_20260528_unified_allnorm"
seeds = ["1", "12", "123", "1234", "12345"]
datasets = ["clothing", "sports"]

def parse(path):
    rows = {}
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) == 4:
            try:
                alpha = float(parts[0])
                rows[alpha] = (float(parts[2]), float(parts[3]))
            except ValueError:
                pass
    return rows

def log_path(dataset, seed):
    current = out / f"{dataset}_seed{seed}_test.log"
    if current.exists():
        return current
    return old / f"{dataset}_seed{seed}_test.log"

lines = [
    "# Unified allnorm late-score fusion 5seed",
    "",
    "Scheme: score_final = 0.6 * zscore(score_base) + 0.4 * zscore(score_allnorm_aux).",
    "",
]

for dataset in datasets:
    base_r, base_n, fus_r, fus_n = [], [], [], []
    lines.extend([
        f"## {dataset}",
        "",
        "| seed | base R@20 | base N@20 | fusion R@20 | fusion N@20 | R gain | N gain |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for seed in seeds:
        rows = parse(log_path(dataset, seed))
        b_r, b_n = rows[0.0]
        f_r, f_n = rows[0.4]
        base_r.append(b_r); base_n.append(b_n); fus_r.append(f_r); fus_n.append(f_n)
        lines.append(
            f"| {seed} | {b_r:.5f} | {b_n:.5f} | {f_r:.5f} | {f_n:.5f} | "
            f"{(f_r / b_r - 1) * 100:.2f}% | {(f_n / b_n - 1) * 100:.2f}% |"
        )
    def mean_std(xs):
        return statistics.mean(xs), statistics.stdev(xs)
    br_m, br_s = mean_std(base_r); bn_m, bn_s = mean_std(base_n)
    fr_m, fr_s = mean_std(fus_r); fn_m, fn_s = mean_std(fus_n)
    lines.extend(["", "Summary:", ""])
    lines.append(f"- Base R@20: {br_m:.5f} +/- {br_s:.5f}; fusion R@20: {fr_m:.5f} +/- {fr_s:.5f}; gain {(fr_m / br_m - 1) * 100:.2f}%")
    lines.append(f"- Base N@20: {bn_m:.5f} +/- {bn_s:.5f}; fusion N@20: {fn_m:.5f} +/- {fn_s:.5f}; gain {(fn_m / bn_m - 1) * 100:.2f}%")
    if stats is not None:
        r_p = stats.ttest_rel(fus_r, base_r).pvalue
        n_p = stats.ttest_rel(fus_n, base_n).pvalue
        lines.append(f"- Paired t-test p-value: R@20={r_p:.6g}, N@20={n_p:.6g}")
    lines.append("")

(out / "summary.md").write_text("\n".join(lines) + "\n")
print(out / "summary.md")
PY
