#!/usr/bin/env python3
"""Prepare the Amazon Sports BM3 files for this repo's MMRec loader."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import defaultdict
from pathlib import Path


SPLIT_NAMES = {
    "0": "train",
    "1": "val",
    "2": "test",
}


def link_or_copy(src: Path, dst: Path, overwrite: bool) -> str:
    if dst.exists():
        if not overwrite:
            return "exists"
        dst.unlink()

    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def read_splits(inter_file: Path) -> dict[str, dict[int, list[int]]]:
    splits: dict[str, dict[int, list[int]]] = {
        name: defaultdict(list) for name in SPLIT_NAMES.values()
    }

    with inter_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"userID", "itemID", "x_label"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{inter_file} misses required columns: {sorted(missing)}")

        for row in reader:
            split_name = SPLIT_NAMES.get(row["x_label"])
            if split_name is None:
                continue
            splits[split_name][int(row["userID"])].append(int(row["itemID"]))

    return splits


def write_split(path: Path, split: dict[int, list[int]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for user_id in sorted(split):
            items = split[user_id]
            if not items:
                f.write(f"{user_id}\n")
                continue
            f.write(f"{user_id} {' '.join(str(item_id) for item_id in items)}\n")
            count += len(items)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("Data/baby/BM3/sports"),
        help="Source directory containing sports.inter and feature files.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("Data/sports"),
        help="Target MMRec dataset directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the target directory.",
    )
    args = parser.parse_args()

    source = args.source
    target = args.target
    required_files = [
        "sports.inter",
        "image_feat.npy",
        "text_feat.npy",
        "u_id_mapping.csv",
        "i_id_mapping.csv",
    ]
    for name in required_files:
        src = source / name
        if not src.exists():
            raise FileNotFoundError(src)

    target.mkdir(parents=True, exist_ok=True)

    file_status = {}
    for name in required_files:
        file_status[name] = link_or_copy(source / name, target / name, args.overwrite)

    splits = read_splits(source / "sports.inter")
    split_counts = {
        split_name: write_split(target / f"{split_name}.txt", split_data)
        for split_name, split_data in splits.items()
    }

    user_ids = set()
    item_ids = set()
    for split_data in splits.values():
        user_ids.update(split_data.keys())
        for items in split_data.values():
            item_ids.update(items)

    print(f"source: {source}")
    print(f"target: {target}")
    for name, status in file_status.items():
        print(f"{name}: {status}")
    print(
        "splits: "
        + ", ".join(f"{name}={count}" for name, count in split_counts.items())
    )
    print(f"users={len(user_ids)} items={len(item_ids)}")


if __name__ == "__main__":
    main()
