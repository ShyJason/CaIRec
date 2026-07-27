#!/usr/bin/env python3
"""Select top no-alpha candidates from eval/full logs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


def read_score(path: Path) -> tuple[float, float] | None:
    score = None
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("1.0000"):
            parts = line.split()
            score = (float(parts[2]), float(parts[3]))
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_root", type=Path)
    parser.add_argument("--min-seeds", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--format", choices=["names", "table"], default="names")
    args = parser.parse_args()

    rows: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for path in args.eval_root.glob("*/seed*_test.log"):
        score = read_score(path)
        if score is None:
            continue
        rows[path.parent.name].append((path.stem, score[0], score[1]))

    ranked = []
    for candidate, values in rows.items():
        if len(values) < args.min_seeds:
            continue
        recall = sum(value[1] for value in values) / len(values)
        ndcg = sum(value[2] for value in values) / len(values)
        ranked.append((recall, ndcg, candidate, sorted(values)))

    ranked.sort(reverse=True)
    selected = ranked[: args.top_k]

    if args.format == "names":
        print(" ".join(candidate for _, _, candidate, _ in selected))
        return

    for rank, (recall, ndcg, candidate, values) in enumerate(selected, start=1):
        seed_scores = ",".join(
            f"{seed}:{seed_recall:.5f}/{seed_ndcg:.5f}"
            for seed, seed_recall, seed_ndcg in values
        )
        print(f"{rank}\t{candidate}\t{len(values)}\t{recall:.5f}\t{ndcg:.5f}\t{seed_scores}")


if __name__ == "__main__":
    main()
