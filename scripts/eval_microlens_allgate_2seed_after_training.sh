#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

stamp="microlens_allgate_2seed_20260528"
out="exp_report/fusion_norm_ablation/late_score_eval_20260528_microlens_allgate_2seed"
mkdir -p "${out}"

latest_ckpt() {
  local seed="$1"
  find "exp_report/microlens/stage2_microlens_rankres_allgate_seed${seed}_${stamp}/ckpt" \
    -maxdepth 1 -name '*.pth' -print | sort -V | tail -1
}

eval_one() {
  local seed="$1"
  local base_log="$2"
  local base_ckpt="$3"
  local fusion_log="exp_report/fusion_norm_ablation/${stamp}/microlens_seed${seed}.log"
  local fusion_ckpt
  fusion_ckpt="$(latest_ckpt "${seed}")"

  echo "[eval] microlens seed${seed} ckpt=${fusion_ckpt}"
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python tools/evaluate_late_score_fusion.py \
    --base-log "${base_log}" \
    --base-ckpt "${base_ckpt}" \
    --fusion-log "${fusion_log}" \
    --fusion-ckpt "${fusion_ckpt}" \
    --device-id 0 \
    --split test \
    --normalize zscore \
    --alphas 0,0.2,0.3,0.35,0.4,0.45,0.5,0.6,1 \
    > "${out}/microlens_seed${seed}_test.log" 2>&1
}

while tmux has-session -t microlens_allgate_seed12 2>/dev/null || tmux has-session -t microlens_allgate_seed123 2>/dev/null; do
  date '+[watch] %F %T waiting for microlens allgate training sessions' | tee -a "${out}/watch.log"
  sleep 300
done

eval_one "12" \
  "exp_report/microlens/stage2_microlens_recommender_decoder_mm_mmrec_microlens_mm_2seed_20260521_153006_seed12/log/run_20260521_153329.log" \
  "exp_report/microlens/stage2_microlens_recommender_decoder_mm_mmrec_microlens_mm_2seed_20260521_153006_seed12/ckpt/stage2_microlens_recommender_decoder_mm_mmrec_microlens_mm_2seed_20260521_153006_seed12_recommender_1.0_epoch30.pth"

eval_one "123" \
  "exp_report/microlens/stage2_microlens_recommender_decoder_mm_mmrec_microlens_tuned_5seed_20260522_023007_seed123/log/run_20260522_023242.log" \
  "exp_report/microlens/stage2_microlens_recommender_decoder_mm_mmrec_microlens_tuned_5seed_20260522_023007_seed123/ckpt/stage2_microlens_recommender_decoder_mm_mmrec_microlens_tuned_5seed_20260522_023007_seed123_recommender_1.0_epoch30.pth"

.venv/bin/python - <<'PY'
from pathlib import Path
import statistics

out = Path("exp_report/fusion_norm_ablation/late_score_eval_20260528_microlens_allgate_2seed")
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

base_r = []; base_n = []; aux_r = []; aux_n = []; fus_r = []; fus_n = []
lines = [
    "# Microlens all-modal normalized gate 2seed",
    "",
    "Aux mode: `rank_residual_allgate`.",
    "Score fusion: `score_final = (1-alpha) * zscore(score_base) + alpha * zscore(score_aux)`.",
    "Primary comparison uses `alpha=0.4`.",
    "",
    "| seed | base R@20 | base N@20 | aux R@20 | aux N@20 | alpha0.4 R@20 | alpha0.4 N@20 | R gain | N gain |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for seed in seeds:
    rows = parse(out / f"microlens_seed{seed}_test.log")
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
(out / "summary.md").write_text("\n".join(lines))
print(out / "summary.md")
PY
