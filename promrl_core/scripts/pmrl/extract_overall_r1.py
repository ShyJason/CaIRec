#!/usr/bin/env python3

import ast
import json
import sys
from pathlib import Path


TARGET_DATASETS = [
    "msrvtt_ret",
    "activitynet_ret",
    "didemo_ret",
    "vatex_ret",
    "audiocaps_ret",
    "clothov2_ret",
]


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: extract_overall_r1.py <log_file>", file=sys.stderr)
        return 1

    log_file = Path(sys.argv[1])
    if not log_file.is_file():
        print(f"Log file not found: {log_file}", file=sys.stderr)
        return 1

    lines = log_file.read_text(encoding="utf-8").splitlines()
    per_dataset = {}

    for idx, line in enumerate(lines):
        if "evaluation--" not in line or "ret_itm" not in line:
            continue

        dataset = None
        for name in TARGET_DATASETS:
            if name in line:
                dataset = name
                break

        if dataset is None or idx + 2 >= len(lines):
            continue

        payload_line = lines[idx + 2]
        marker = "INFO - __main__ -   "
        if marker not in payload_line:
            continue

        payload = payload_line.split(marker, 1)[1]
        try:
            metrics = ast.literal_eval(payload)
        except Exception:
            continue

        if "video_r1" in metrics:
            per_dataset[dataset] = float(metrics["video_r1"])

    values = [per_dataset[name] for name in TARGET_DATASETS if name in per_dataset]
    overall = sum(values) / len(values) if values else None

    print(
        json.dumps(
            {
                "datasets": per_dataset,
                "overall_r1": overall,
                "num_datasets": len(values),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
