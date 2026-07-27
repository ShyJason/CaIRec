#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
from pathlib import Path


SEEDS = ("1", "12", "123", "1234", "12345")
REQUIRED_ALPHAS = (0.0, 0.4, 1.0)


def parse_eval_log(path: Path) -> dict[float, tuple[float, float]]:
    rows: dict[float, tuple[float, float]] = {}
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) != 4:
            continue
        try:
            alpha = float(parts[0])
            recall = float(parts[2])
            ndcg = float(parts[3])
        except ValueError:
            continue
        rows[alpha] = (recall, ndcg)
    missing = [alpha for alpha in REQUIRED_ALPHAS if alpha not in rows]
    if missing:
        raise ValueError(f"{path} missing alpha rows: {missing}")
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old-dir",
        default="exp_report/fusion_norm_ablation/late_score_eval_microlens100k_allgate_score_2seed_20260528",
        help="Directory containing seed12 and seed123 eval logs.",
    )
    parser.add_argument(
        "--new-dir",
        default="exp_report/fusion_norm_ablation/late_score_eval_microlens100k_allgate_score_5seed_rest_20260529",
        help="Directory containing seed1, seed1234 and seed12345 eval logs.",
    )
    parser.add_argument(
        "--out",
        default="exp_report/fusion_norm_ablation/microlens100k_allgate_5seed_summary.md",
    )
    args = parser.parse_args()

    old_dir = Path(args.old_dir)
    new_dir = Path(args.new_dir)
    out = Path(args.out)

    logs = {
        "1": new_dir / "microlens100k_seed1_test.log",
        "12": old_dir / "microlens100k_seed12_test.log",
        "123": old_dir / "microlens100k_seed123_test.log",
        "1234": new_dir / "microlens100k_seed1234_test.log",
        "12345": new_dir / "microlens100k_seed12345_test.log",
    }

    missing_logs = [str(path) for path in logs.values() if not path.is_file()]
    if missing_logs:
        raise FileNotFoundError("missing eval logs:\n" + "\n".join(missing_logs))

    base_r: list[float] = []
    base_n: list[float] = []
    aux_r: list[float] = []
    aux_n: list[float] = []
    fusion_r: list[float] = []
    fusion_n: list[float] = []

    lines = [
        "# MicroLens-100k all-modal normalized gate 5-seed summary",
        "",
        "Aux mode: `rank_residual_allgate`.",
        "Score fusion: `score_final = (1-alpha) * zscore(score_base) + alpha * zscore(score_aux)`.",
        "Primary comparison uses `alpha=0.4`.",
        "Seeds: `1, 12, 123, 1234, 12345`.",
        "",
        "| seed | base R@20 | base N@20 | aux R@20 | aux N@20 | alpha0.4 R@20 | alpha0.4 N@20 | R gain | N gain |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for seed in SEEDS:
        rows = parse_eval_log(logs[seed])
        br, bn = rows[0.0]
        ar, an = rows[1.0]
        fr, fn = rows[0.4]
        base_r.append(br)
        base_n.append(bn)
        aux_r.append(ar)
        aux_n.append(an)
        fusion_r.append(fr)
        fusion_n.append(fn)
        lines.append(
            f"| {seed} | {br:.5f} | {bn:.5f} | {ar:.5f} | {an:.5f} | "
            f"{fr:.5f} | {fn:.5f} | {(fr / br - 1) * 100:.2f}% | "
            f"{(fn / bn - 1) * 100:.2f}% |"
        )

    br_m, br_s = mean_std(base_r)
    bn_m, bn_s = mean_std(base_n)
    ar_m, ar_s = mean_std(aux_r)
    an_m, an_s = mean_std(aux_n)
    fr_m, fr_s = mean_std(fusion_r)
    fn_m, fn_s = mean_std(fusion_n)

    lines += [
        "",
        f"- Base mean: R@20 {br_m:.5f} +/- {br_s:.5f}, N@20 {bn_m:.5f} +/- {bn_s:.5f}",
        f"- Aux mean: R@20 {ar_m:.5f} +/- {ar_s:.5f}, N@20 {an_m:.5f} +/- {an_s:.5f}",
        f"- Alpha0.4 mean: R@20 {fr_m:.5f} +/- {fr_s:.5f}, N@20 {fn_m:.5f} +/- {fn_s:.5f}",
        f"- Mean gain: R@20 {(fr_m / br_m - 1) * 100:.2f}%, N@20 {(fn_m / bn_m - 1) * 100:.2f}%",
        "",
    ]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
