#!/usr/bin/env python3
import argparse
import pathlib
import re


VAL_RE = re.compile(
    r"epoch = (?P<epoch>\d+) hr@20 = (?P<hr>[0-9.]+), "
    r"recall@20 = (?P<recall>[0-9.]+), ndcg@20 = (?P<ndcg>[0-9.]+)"
)
FINAL_RE = re.compile(
    r"final strict test hr@20 = (?P<hr>[0-9.]+), "
    r"recall@20 = (?P<recall>[0-9.]+), ndcg@20 = (?P<ndcg>[0-9.]+)"
)
EDGE_RE = re.compile(
    r"learned edge confidence:\s*rr=(?P<rr>[0-9.eE+-]+),\s*"
    r"ri=(?P<ri>[0-9.eE+-]+),\s*ii=(?P<ii>[0-9.eE+-]+)"
)


def read_text(path):
    return pathlib.Path(path).read_text(errors="ignore")


def parse_val_curve(path):
    text = read_text(path)
    curve = {}
    for match in VAL_RE.finditer(text):
        epoch = int(match.group("epoch"))
        curve[epoch] = (
            float(match.group("recall")),
            float(match.group("ndcg")),
        )
    final_matches = list(FINAL_RE.finditer(text))
    final = None
    if final_matches:
        match = final_matches[-1]
        final = (float(match.group("recall")), float(match.group("ndcg")))
    edge_matches = list(EDGE_RE.finditer(text))
    edge = None
    if edge_matches:
        match = edge_matches[-1]
        edge = (
            float(match.group("rr")),
            float(match.group("ri")),
            float(match.group("ii")),
        )
    return curve, final, edge


def best_until(curve, epoch):
    values = [(e, *curve[e]) for e in sorted(curve) if e <= epoch]
    if not values:
        return None
    return max(values, key=lambda row: (row[1], row[2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current-name", default="current")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--tail", type=int, default=12)
    args = parser.parse_args()

    current, current_final, current_edge = parse_val_curve(args.current)
    baseline, baseline_final, baseline_edge = parse_val_curve(args.baseline)
    epochs = sorted(set(current) & set(baseline))

    print(f"current\t{args.current_name}\t{args.current}")
    print(f"baseline\t{args.baseline_name}\t{args.baseline}")
    if current_final:
        print(f"current_final\t{current_final[0]:.5f}\t{current_final[1]:.5f}")
    if baseline_final:
        print(f"baseline_final\t{baseline_final[0]:.5f}\t{baseline_final[1]:.5f}")
    if current_edge:
        print(
            "current_learned_edge_confidence\t"
            f"rr={current_edge[0]:.6f}\tri={current_edge[1]:.6f}\tii={current_edge[2]:.6f}"
        )
    if baseline_edge:
        print(
            "baseline_learned_edge_confidence\t"
            f"rr={baseline_edge[0]:.6f}\tri={baseline_edge[1]:.6f}\tii={baseline_edge[2]:.6f}"
        )
    print()
    print(
        "epoch\tcurrent_recall20\tcurrent_ndcg20\tbaseline_recall20\tbaseline_ndcg20\t"
        "delta_recall20\tdelta_ndcg20"
    )
    for epoch in epochs[-args.tail:]:
        cur_recall, cur_ndcg = current[epoch]
        base_recall, base_ndcg = baseline[epoch]
        print(
            f"{epoch}\t{cur_recall:.5f}\t{cur_ndcg:.5f}\t"
            f"{base_recall:.5f}\t{base_ndcg:.5f}\t"
            f"{cur_recall - base_recall:+.5f}\t{cur_ndcg - base_ndcg:+.5f}"
        )

    if current:
        latest_epoch = max(current)
        cur_best = best_until(current, latest_epoch)
        base_best = best_until(baseline, latest_epoch)
        if cur_best and base_best:
            print()
            print(
                "best_until_current_epoch\t"
                "name\tepoch\trecall20\tndcg20\tdelta_recall20\tdelta_ndcg20"
            )
            print(
                f"{args.current_name}\t{cur_best[0]}\t{cur_best[1]:.5f}\t{cur_best[2]:.5f}\t"
                f"{cur_best[1] - base_best[1]:+.5f}\t{cur_best[2] - base_best[2]:+.5f}"
            )
            print(
                f"{args.baseline_name}\t{base_best[0]}\t{base_best[1]:.5f}\t{base_best[2]:.5f}\t"
                "+0.00000\t+0.00000"
            )


if __name__ == "__main__":
    main()
