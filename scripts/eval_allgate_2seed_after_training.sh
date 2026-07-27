#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

stamp="allgate_2seed_20260528"
out="exp_report/fusion_norm_ablation/late_score_eval_20260528_allgate_2seed"
mkdir -p "${out}"

latest_ckpt() {
  local dataset="$1"
  local seed="$2"
  find "exp_report/${dataset}/stage2_${dataset}_rankres_allgate_seed${seed}_${stamp}/ckpt" \
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

  echo "[eval] ${dataset} seed${seed} ckpt=${fusion_ckpt}"
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python tools/evaluate_late_score_fusion.py \
    --base-log "${base_log}" \
    --base-ckpt "${base_ckpt}" \
    --fusion-log "${fusion_log}" \
    --fusion-ckpt "${fusion_ckpt}" \
    --device-id 0 \
    --split test \
    --normalize zscore \
    --alphas 0,0.2,0.3,0.35,0.4,0.45,0.5,0.6,1 \
    > "${out}/${dataset}_seed${seed}_test.log" 2>&1
}

while tmux has-session -t allgate_seed12 2>/dev/null || tmux has-session -t allgate_seed123 2>/dev/null; do
  date '+[watch] %F %T waiting for allgate training sessions'
  sleep 300
done

eval_one "clothing" "12" \
  "exp_report/clothing/stage2_clothing_mmrec_fixed_seed12_mmrec_clothing_mm_fixedmissing_20260521_052129/log/run_20260521_052411.log" \
  "exp_report/clothing/stage2_clothing_mmrec_fixed_seed12_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage2_clothing_mmrec_fixed_seed12_mmrec_clothing_mm_fixedmissing_20260521_052129_recommender_1.0_epoch122.pth"

eval_one "clothing" "123" \
  "exp_report/clothing/stage2_clothing_mmrec_fixed_seed123_mmrec_clothing_mm_fixedmissing_20260521_052129/log/run_20260521_060217.log" \
  "exp_report/clothing/stage2_clothing_mmrec_fixed_seed123_mmrec_clothing_mm_fixedmissing_20260521_052129/ckpt/stage2_clothing_mmrec_fixed_seed123_mmrec_clothing_mm_fixedmissing_20260521_052129_recommender_1.0_epoch121.pth"

eval_one "sports" "12" \
  "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed12_mmrec_sports_mm_fixedmissing_20260524_165817/log/run_20260524_190838.log" \
  "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed12_mmrec_sports_mm_fixedmissing_20260524_165817/ckpt/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed12_mmrec_sports_mm_fixedmissing_20260524_165817_recommender_50_epoch160.pth"

eval_one "sports" "123" \
  "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed123_mmrec_sports_mm_fixedmissing_20260524_165817/log/run_20260524_211736.log" \
  "exp_report/sports/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed123_mmrec_sports_mm_fixedmissing_20260524_165817/ckpt/stage2_sports_recommender_decoder_mm_mmrec_sports_mm_mr0.3_seed123_mmrec_sports_mm_fixedmissing_20260524_165817_recommender_50_epoch158.pth"

.venv/bin/python - <<'PY'
from pathlib import Path
import statistics

out = Path("exp_report/fusion_norm_ablation/late_score_eval_20260528_allgate_2seed")
datasets = ["clothing", "sports"]
seeds = ["12", "123"]

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

lines = [
    "# All-modal normalized gate 2seed",
    "",
    "Aux mode: `rank_residual_allgate`.",
    "Score fusion: `score_final = (1-alpha) * zscore(score_base) + alpha * zscore(score_aux)`.",
    "Primary comparison uses `alpha=0.4`, same as the allnorm setting.",
    "",
]

for dataset in datasets:
    base_r = []; base_n = []; fus_r = []; fus_n = []; aux_r = []; aux_n = []
    lines += [
        f"## {dataset}",
        "",
        "| seed | base R@20 | base N@20 | aux R@20 | aux N@20 | alpha0.4 R@20 | alpha0.4 N@20 | R gain | N gain |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in seeds:
        rows = parse(out / f"{dataset}_seed{seed}_test.log")
        br, bn = rows[0.0]
        fr, fn = rows[0.4]
        ar, an = rows[1.0]
        base_r.append(br); base_n.append(bn); fus_r.append(fr); fus_n.append(fn); aux_r.append(ar); aux_n.append(an)
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

(out / "summary.md").write_text("\n".join(lines) + "\n")
print(out / "summary.md")
PY
