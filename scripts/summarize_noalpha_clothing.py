#!/usr/bin/env python3
"""Summarize clothing no-alpha no-assemble full evaluations."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


TARGET_RECALL = 0.07638
TARGET_NDCG = 0.03425


@dataclass(frozen=True)
class CandidateResult:
    run: str
    candidate: str
    seed_scores: tuple[tuple[str, float, float], ...]

    @property
    def seed_count(self) -> int:
        return len(self.seed_scores)

    @property
    def recall(self) -> float:
        return sum(score[1] for score in self.seed_scores) / self.seed_count

    @property
    def ndcg(self) -> float:
        return sum(score[2] for score in self.seed_scores) / self.seed_count

    @property
    def recall_gap(self) -> float:
        return self.recall - TARGET_RECALL

    @property
    def ndcg_gap(self) -> float:
        return self.ndcg - TARGET_NDCG


def read_score(path: Path) -> tuple[float, float] | None:
    score = None
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("1.0000"):
            parts = line.split()
            score = (float(parts[2]), float(parts[3]))
    return score


def collect_results(root: Path) -> list[CandidateResult]:
    results: list[CandidateResult] = []
    for run_dir in sorted(root.iterdir()):
        eval_root = run_dir / "eval" / "full"
        if not eval_root.exists():
            continue
        rows: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
        for test_log in eval_root.glob("*/seed*_test.log"):
            score = read_score(test_log)
            if score is None:
                continue
            rows[test_log.parent.name].append((test_log.stem, score[0], score[1]))
        for candidate, seed_scores in rows.items():
            seed_scores = sorted(seed_scores)
            if not seed_scores:
                continue
            recall = sum(score[1] for score in seed_scores) / len(seed_scores)
            ndcg = sum(score[2] for score in seed_scores) / len(seed_scores)
            if recall == 0.0 and ndcg == 0.0:
                continue
            results.append(CandidateResult(run_dir.name, candidate, tuple(seed_scores)))
    return results


def aggregate_by_candidate(results: list[CandidateResult]) -> list[CandidateResult]:
    grouped: dict[str, dict[str, tuple[str, float, float]]] = defaultdict(dict)
    run_names: dict[str, set[str]] = defaultdict(set)

    for result in results:
        run_names[result.candidate].add(result.run)
        for seed, recall, ndcg in result.seed_scores:
            previous = grouped[result.candidate].get(seed)
            if previous is None or result.run > previous[0]:
                grouped[result.candidate][seed] = (result.run, recall, ndcg)

    aggregated: list[CandidateResult] = []
    for candidate, seed_rows in grouped.items():
        seed_scores = tuple(
            (seed, recall, ndcg)
            for seed, (_run, recall, ndcg) in sorted(seed_rows.items())
        )
        run = "aggregate:" + "+".join(sorted(run_names[candidate]))
        aggregated.append(CandidateResult(run, candidate, seed_scores))

    return aggregated


def format_seed_scores(result: CandidateResult) -> str:
    return ", ".join(
        f"{seed}:{recall:.5f}/{ndcg:.5f}"
        for seed, recall, ndcg in result.seed_scores
    )


def write_tsv(path: Path, results: list[CandidateResult]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write(
            "rank\trun\tcandidate\tseeds\trecall@20\tndcg@20\t"
            "target_recall@20\ttarget_ndcg@20\trecall_gap\tndcg_gap\tseed_scores\n"
        )
        ranked = sorted(results, key=lambda x: (x.seed_count, x.recall, x.ndcg), reverse=True)
        for rank, result in enumerate(ranked, start=1):
            fh.write(
                f"{rank}\t{result.run}\t{result.candidate}\t{result.seed_count}\t"
                f"{result.recall:.5f}\t{result.ndcg:.5f}\t"
                f"{TARGET_RECALL:.5f}\t{TARGET_NDCG:.5f}\t"
                f"{result.recall_gap:.5f}\t{result.ndcg_gap:.5f}\t"
                f"{format_seed_scores(result)}\n"
            )


def write_markdown(path: Path, results: list[CandidateResult], aggregate_results: list[CandidateResult]) -> None:
    five_seed = [result for result in results if result.seed_count >= 5]
    two_seed = [result for result in results if result.seed_count >= 2]
    all_ranked = sorted(results, key=lambda x: (x.recall, x.ndcg), reverse=True)
    aggregate_five_seed = [result for result in aggregate_results if result.seed_count >= 5]
    aggregate_two_seed = [result for result in aggregate_results if result.seed_count >= 2]

    with path.open("w", encoding="utf-8") as fh:
        fh.write("# Clothing no-alpha no-assemble summary\n\n")
        fh.write(
            f"Target assemble baseline: R@20 {TARGET_RECALL:.5f}, "
            f"N@20 {TARGET_NDCG:.5f}.\n\n"
        )

        for title, subset, limit in [
            ("Best aggregated 5+ seed results by candidate", aggregate_five_seed, 10),
            ("Best aggregated 2+ seed results by candidate", aggregate_two_seed, 15),
        ]:
            fh.write(f"## {title}\n\n")
            fh.write(
                "| rank | seeds | R@20 | N@20 | R gap | N gap | candidate | sources |\n"
                "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |\n"
            )
            ranked = sorted(subset, key=lambda x: (x.recall, x.ndcg), reverse=True)
            for rank, result in enumerate(ranked[:limit], start=1):
                fh.write(
                    f"| {rank} | {result.seed_count} | {result.recall:.5f} | "
                    f"{result.ndcg:.5f} | {result.recall_gap:.5f} | "
                    f"{result.ndcg_gap:.5f} | `{result.candidate}` | "
                    f"`{result.run.removeprefix('aggregate:')}` |\n"
                )
            fh.write("\n")

        for title, subset, limit in [
            ("Best 5+ seed results", five_seed, 10),
            ("Best 2+ seed results", two_seed, 15),
            ("Best single/partial results", all_ranked, 15),
        ]:
            fh.write(f"## {title}\n\n")
            fh.write(
                "| rank | seeds | R@20 | N@20 | R gap | N gap | run | candidate |\n"
                "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |\n"
            )
            ranked = sorted(subset, key=lambda x: (x.recall, x.ndcg), reverse=True)
            for rank, result in enumerate(ranked[:limit], start=1):
                fh.write(
                    f"| {rank} | {result.seed_count} | {result.recall:.5f} | "
                    f"{result.ndcg:.5f} | {result.recall_gap:.5f} | "
                    f"{result.ndcg_gap:.5f} | `{result.run}` | "
                    f"`{result.candidate}` |\n"
                )
            fh.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("exp_report/noalpha_clothing_beta_search"),
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("exp_report/noalpha_clothing_beta_search/noalpha_clothing_current_summary"),
    )
    args = parser.parse_args()

    results = collect_results(args.root)
    aggregate_results = aggregate_by_candidate(results)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out_prefix.with_suffix(".tsv"), results)
    write_tsv(args.out_prefix.with_name(args.out_prefix.name + "_by_candidate").with_suffix(".tsv"), aggregate_results)
    write_markdown(args.out_prefix.with_suffix(".md"), results, aggregate_results)
    print(f"wrote {args.out_prefix.with_suffix('.tsv')}")
    print(f"wrote {args.out_prefix.with_name(args.out_prefix.name + '_by_candidate').with_suffix('.tsv')}")
    print(f"wrote {args.out_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
