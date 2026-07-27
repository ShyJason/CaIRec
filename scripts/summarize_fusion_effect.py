#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SUMMARY_RE = re.compile(
    r"-{5,}\s+(?P<suffix>.+?)\s+best epoch\s+(?P<epoch>\d+)-{5,}.*?"
    r"hr@20\s*=\s*(?P<hr>[-+]?(?:\d+(?:\.\d*)?|\.\d+)),\s*"
    r"recall@20\s*=\s*(?P<recall>[-+]?(?:\d+(?:\.\d*)?|\.\d+)),\s*"
    r"ndcg@20\s*=\s*(?P<ndcg>[-+]?(?:\d+(?:\.\d*)?|\.\d+))",
    re.S,
)


def parse_log(path):
    text = ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    matches = list(SUMMARY_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    suffix = match.group("suffix")
    parts = suffix.split("_")
    method = parts[1] if len(parts) > 1 else "unknown"
    seed = "unknown"
    dataset_seed = "unknown"
    for idx, part in enumerate(parts):
        if part.startswith("dseed"):
            dataset_seed = part.removeprefix("dseed")
        if part == "seed" and idx + 1 < len(parts):
            seed = parts[idx + 1]
        elif part.startswith("seed") and part.removeprefix("seed").isdigit():
            seed = part.removeprefix("seed")
    return {
        "method": method,
        "seed": seed,
        "dataset_seed": dataset_seed,
        "suffix": suffix,
        "best_epoch": int(match.group("epoch")),
        "hr20": float(match.group("hr")),
        "recall20": float(match.group("recall")),
        "ndcg20": float(match.group("ndcg")),
        "log": str(path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    rows = []
    for path in sorted(log_dir.glob(f"*_{args.run_tag}.launch.log")):
        row = parse_log(path)
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit(f"no completed fusion logs found for run_tag={args.run_tag} in {log_dir}")

    rows.sort(key=lambda row: (row["seed"], -row["recall20"], -row["ndcg20"], row["method"]))
    out_path = log_dir / f"{args.run_tag}.summary.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[fusion-summary] wrote {out_path}")
    for row in rows:
        print(
            "[fusion-summary] method={method} seed={seed} dseed={dataset_seed} "
            "epoch={best_epoch} recall@20={recall20:.5f} ndcg@20={ndcg20:.5f}".format(**row)
        )


if __name__ == "__main__":
    main()
