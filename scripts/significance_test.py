#!/usr/bin/env python3
"""Paired significance tests for MMRec/I3 seed sweeps."""

from __future__ import annotations

import argparse
import itertools
import math
import os
import re
import sys
from pathlib import Path


os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

REPO_ROOT = Path(__file__).resolve().parents[1]
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MMRec against an I3 baseline across paired seeds."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--exp-mode", required=True, help="ff, fm, mf, or mm")
    parser.add_argument(
        "--missing-rate",
        "--train-missing-rate",
        dest="missing_rate",
        default="0.3",
        help="Training missing rate encoded in run names; eval/test missing rate is fixed at 0.5.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 12, 123, 1234, 12345],
    )
    parser.add_argument(
        "--baseline-method",
        default="i3",
        choices=["i3", "i3_noirm_noib"],
        help="Which I3-code baseline to compare against MMRec.",
    )
    parser.add_argument(
        "--method",
        default="mmrec",
        help="Method to compare against the baseline. Default: mmrec.",
    )
    parser.add_argument(
        "--metric",
        default="recall",
        choices=["hr", "recall", "ndcg"],
    )
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional suffix tag used by scripts/run_mmrec_i3_seed_grid.sh.",
    )
    parser.add_argument(
        "--exp-root",
        type=Path,
        default=REPO_ROOT / "exp_report",
    )
    parser.add_argument(
        "--test",
        default="paired_t",
        choices=["paired_t", "wilcoxon", "both"],
    )
    parser.add_argument(
        "--alternative",
        default="two-sided",
        choices=["two-sided", "greater", "less"],
        help="'greater' tests whether method > baseline.",
    )
    return parser.parse_args()


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def metric_pattern(topk: int) -> re.Pattern[str]:
    number = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    return re.compile(
        rf"hr@{topk}\s*=\s*{number},\s*"
        rf"recall@{topk}\s*=\s*{number},\s*"
        rf"ndcg@{topk}\s*=\s*{number}"
    )


def find_run_dir(
    exp_root: Path,
    method: str,
    dataset: str,
    exp_mode: str,
    missing_rate: str,
    seed: int,
    run_tag: str | None,
) -> Path:
    dataset_root = exp_root / dataset
    if method == "mmrec":
        base = (
            f"stage2_{dataset}_*_{exp_mode}_"
            f"mmrec_{dataset}_{exp_mode}_mr{missing_rate}_seed{seed}_"
        )
    else:
        base = f"{method}_{dataset}_{exp_mode}_mr{missing_rate}_seed{seed}_"

    pattern = base + (f"{run_tag}*" if run_tag else "*")
    candidates = [path for path in dataset_root.glob(pattern) if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directory found for {method}, seed={seed}: {dataset_root / pattern}")

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def log_files_for_run(exp_root: Path, dataset: str, run_dir: Path) -> list[Path]:
    logs: list[Path] = []
    logs.extend(sorted((run_dir / "log").glob("*.log")))

    launch_log = exp_root / dataset / "comparison_launch_logs" / f"{run_dir.name}.launch.log"
    if launch_log.exists():
        logs.append(launch_log)

    return logs


def extract_metric(log_files: list[Path], metric: str, topk: int) -> float:
    pattern = metric_pattern(topk)
    index = {"hr": 0, "recall": 1, "ndcg": 2}[metric]
    values: list[float] = []

    for path in log_files:
        text = strip_ansi(path.read_text(encoding="utf-8", errors="replace"))
        for match in pattern.finditer(text):
            values.append(float(match.group(index + 1)))

    if not values:
        searched = ", ".join(str(path) for path in log_files) or "<no log files>"
        raise ValueError(f"No {metric}@{topk} value found in {searched}")
    return values[-1]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def paired_t_test(diffs: list[float], alternative: str) -> tuple[float, float]:
    if len(diffs) < 2:
        return float("nan"), float("nan")
    sd = sample_std(diffs)
    if sd == 0:
        if mean(diffs) == 0:
            return 0.0, 1.0
        return math.copysign(float("inf"), mean(diffs)), 0.0

    t_stat = mean(diffs) / (sd / math.sqrt(len(diffs)))
    try:
        from scipy import stats  # type: ignore
    except Exception as exc:
        raise RuntimeError("scipy is required for paired t-test p-values") from exc

    df = len(diffs) - 1
    if alternative == "greater":
        p_value = float(stats.t.sf(t_stat, df))
    elif alternative == "less":
        p_value = float(stats.t.cdf(t_stat, df))
    else:
        p_value = float(2 * stats.t.sf(abs(t_stat), df))
    return float(t_stat), p_value


def exact_wilcoxon_fallback(diffs: list[float], alternative: str) -> tuple[float, float]:
    nonzero = [diff for diff in diffs if diff != 0]
    if not nonzero:
        return 0.0, 1.0

    abs_values = [abs(diff) for diff in nonzero]
    order = sorted(range(len(abs_values)), key=lambda i: abs_values[i])
    ranks = [0.0] * len(abs_values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and abs_values[order[j]] == abs_values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j

    w_plus = sum(rank for rank, diff in zip(ranks, nonzero) if diff > 0)
    all_w_plus = []
    for signs in itertools.product([-1, 1], repeat=len(nonzero)):
        all_w_plus.append(sum(rank for rank, sign in zip(ranks, signs) if sign > 0))

    if alternative == "greater":
        p_value = sum(value >= w_plus for value in all_w_plus) / len(all_w_plus)
        statistic = w_plus
    elif alternative == "less":
        p_value = sum(value <= w_plus for value in all_w_plus) / len(all_w_plus)
        statistic = w_plus
    else:
        w_minus = sum(ranks) - w_plus
        statistic = min(w_plus, w_minus)
        p_value = sum(min(value, sum(ranks) - value) <= statistic for value in all_w_plus) / len(all_w_plus)
    return statistic, p_value


def wilcoxon_test(diffs: list[float], alternative: str) -> tuple[float, float]:
    try:
        from scipy import stats  # type: ignore
        result = stats.wilcoxon(diffs, alternative=alternative, zero_method="wilcox")
        return float(result.statistic), float(result.pvalue)
    except Exception:
        return exact_wilcoxon_fallback(diffs, alternative)


def collect_values(args: argparse.Namespace, method: str) -> tuple[list[float], list[Path]]:
    values: list[float] = []
    dirs: list[Path] = []
    for seed in args.seeds:
        run_dir = find_run_dir(
            args.exp_root,
            method,
            args.dataset,
            args.exp_mode,
            args.missing_rate,
            seed,
            args.run_tag,
        )
        logs = log_files_for_run(args.exp_root, args.dataset, run_dir)
        value = extract_metric(logs, args.metric, args.topk)
        values.append(value)
        dirs.append(run_dir)
    return values, dirs


def format_values(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.5f}" for value in values) + "]"


def main() -> None:
    args = parse_args()
    method_values, method_dirs = collect_values(args, args.method)
    baseline_values, baseline_dirs = collect_values(args, args.baseline_method)
    diffs = [left - right for left, right in zip(method_values, baseline_values)]

    print(f"dataset={args.dataset} exp_mode={args.exp_mode} train_missing_rate={args.missing_rate} test_missing_rate=0.5")
    print(f"metric={args.metric}@{args.topk} seeds={' '.join(str(seed) for seed in args.seeds)}")
    print(f"method={args.method} values={format_values(method_values)} mean={mean(method_values):.6f} std={sample_std(method_values):.6f}")
    print(f"baseline={args.baseline_method} values={format_values(baseline_values)} mean={mean(baseline_values):.6f} std={sample_std(baseline_values):.6f}")
    print(f"diff=method-baseline values={format_values(diffs)} mean={mean(diffs):.6f} std={sample_std(diffs):.6f}")

    if args.test in ("paired_t", "both"):
        statistic, p_value = paired_t_test(diffs, args.alternative)
        print(f"paired_t alternative={args.alternative} t={statistic:.6f} p={p_value:.8g}")
    if args.test in ("wilcoxon", "both"):
        statistic, p_value = wilcoxon_test(diffs, args.alternative)
        print(f"wilcoxon alternative={args.alternative} statistic={statistic:.6f} p={p_value:.8g}")

    print("\nmatched runs:")
    for seed, method_dir, baseline_dir in zip(args.seeds, method_dirs, baseline_dirs):
        print(f"  seed={seed} method={method_dir.name} baseline={baseline_dir.name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
