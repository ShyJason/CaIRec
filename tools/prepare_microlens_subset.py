#!/usr/bin/env python3
"""Prepare a filtered MicroLens-1M subset for this MMRec repo."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


SPLIT_LABELS = {"train": 0, "val": 1, "test": 2}


def read_available_items(items_file: Path, title_file: Path) -> set[int]:
    listed_items: set[int] = set()
    with items_file.open("r", encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split("\t")
            if fields and fields[0]:
                listed_items.add(int(fields[0]))

    title_items: set[int] = set()
    with title_file.open("r", encoding="utf-8", newline="") as f:
        sample = f.readline()
        f.seek(0)
        if sample.startswith("item,"):
            reader = csv.DictReader(f)
            for row in reader:
                title_items.add(int(row["item"]))
            return listed_items.intersection(title_items)

        reader = csv.reader(f)
        for row in reader:
            if row:
                title_items.add(int(row[0]))

    return listed_items.intersection(title_items)


def iter_user_sequences(pairs_file: Path):
    with pairs_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            user_raw, seq_raw = line.split("\t", 1)
            yield int(user_raw), [int(item_id) for item_id in seq_raw.split()]


def count_top_items(
    pairs_file: Path,
    available_items: set[int],
    top_items: int,
) -> set[int]:
    counts: Counter[int] = Counter()
    for _, seq in iter_user_sequences(pairs_file):
        counts.update(item_id for item_id in seq if item_id in available_items)
    return {item_id for item_id, _ in counts.most_common(top_items)}


def select_users(
    pairs_file: Path,
    allowed_items: set[int],
    min_user_interactions: int,
    target_interactions: int,
    seed: int,
) -> dict[int, list[int]]:
    eligible: list[tuple[int, list[int]]] = []
    for user_id, seq in iter_user_sequences(pairs_file):
        kept = [item_id for item_id in seq if item_id in allowed_items]
        if len(kept) >= min_user_interactions:
            eligible.append((user_id, kept))

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(eligible))

    selected: dict[int, list[int]] = {}
    total = 0
    for idx in order:
        user_id, seq = eligible[int(idx)]
        selected[user_id] = seq
        total += len(seq)
        if total >= target_interactions:
            break

    return selected


def build_mappings(
    selected: dict[int, list[int]],
    top_item_order: list[int],
) -> tuple[dict[int, int], dict[int, int]]:
    user_map = {raw_id: idx for idx, raw_id in enumerate(sorted(selected))}
    selected_items = {item_id for seq in selected.values() for item_id in seq}
    item_order = [item_id for item_id in top_item_order if item_id in selected_items]
    item_map = {raw_id: idx for idx, raw_id in enumerate(item_order)}
    return user_map, item_map


def write_split(path: Path, split_data: dict[int, list[int]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for user_id in sorted(split_data):
            items = split_data[user_id]
            f.write(f"{user_id} {' '.join(str(item_id) for item_id in items)}\n")
            count += len(items)
    return count


def write_inter_and_splits(
    target_dir: Path,
    dataset_name: str,
    selected: dict[int, list[int]],
    user_map: dict[int, int],
    item_map: dict[int, int],
) -> dict[str, int]:
    splits: dict[str, dict[int, list[int]]] = {
        "train": {},
        "val": {},
        "test": {},
    }

    inter_file = target_dir / f"{dataset_name}.inter"
    with inter_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["userID", "itemID", "rating", "timestamp", "x_label"])
        for raw_user in sorted(selected):
            user_id = user_map[raw_user]
            mapped_seq = [item_map[item_id] for item_id in selected[raw_user]]
            split_parts = {
                "train": mapped_seq[:-2],
                "val": [mapped_seq[-2]],
                "test": [mapped_seq[-1]],
            }
            for split_name, items in split_parts.items():
                splits[split_name][user_id] = items
                label = SPLIT_LABELS[split_name]
                for pos, item_id in enumerate(items):
                    writer.writerow([user_id, item_id, "1.0", pos, label])

    counts = {
        split_name: write_split(target_dir / f"{split_name}.txt", split_data)
        for split_name, split_data in splits.items()
    }
    counts["total"] = sum(counts.values())
    return counts


def write_mappings(
    target_dir: Path,
    user_map: dict[int, int],
    item_map: dict[int, int],
) -> None:
    with (target_dir / "u_id_mapping.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["raw_userID", "userID"])
        for raw_id, mapped_id in sorted(user_map.items(), key=lambda x: x[1]):
            writer.writerow([raw_id, mapped_id])

    with (target_dir / "i_id_mapping.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["raw_itemID", "itemID"])
        for raw_id, mapped_id in sorted(item_map.items(), key=lambda x: x[1]):
            writer.writerow([raw_id, mapped_id])


def write_features(
    feature_file: Path,
    target_dir: Path,
    item_map: dict[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    feature_df = pd.read_parquet(
        feature_file,
        columns=["item_id", "txt_emb_BERT", "img_emb_CLIPRN50"],
    )
    feature_df = feature_df[feature_df["item_id"].isin(item_map)].copy()
    if len(feature_df) != len(item_map):
        found = set(int(item_id) for item_id in feature_df["item_id"])
        missing = sorted(set(item_map).difference(found))[:10]
        raise ValueError(
            f"{feature_file} misses {len(item_map) - len(found)} selected items; "
            f"first missing ids: {missing}"
        )

    feature_df["mapped_item_id"] = feature_df["item_id"].map(item_map)
    feature_df = feature_df.sort_values("mapped_item_id")

    text_feat = np.vstack(feature_df["txt_emb_BERT"].to_numpy()).astype(np.float32)
    image_feat = np.vstack(feature_df["img_emb_CLIPRN50"].to_numpy()).astype(np.float32)
    np.save(target_dir / "text_feat.npy", text_feat)
    np.save(target_dir / "image_feat.npy", image_feat)
    return image_feat.shape, text_feat.shape


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-file", type=Path, required=True)
    parser.add_argument("--items-file", type=Path, required=True)
    parser.add_argument("--title-file", type=Path, required=True)
    parser.add_argument("--feature-file", type=Path)
    parser.add_argument("--target-dir", type=Path, default=Path("Data/microlens"))
    parser.add_argument("--dataset-name", default="microlens")
    parser.add_argument("--top-items", type=int, default=30000)
    parser.add_argument("--min-user-interactions", type=int, default=5)
    parser.add_argument("--target-interactions", type=int, default=450000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    target_dir = args.target_dir
    if target_dir.exists() and any(target_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{target_dir} is not empty; pass --overwrite to replace files")
    target_dir.mkdir(parents=True, exist_ok=True)

    available_items = read_available_items(args.items_file, args.title_file)
    top_item_counts: Counter[int] = Counter()
    for _, seq in iter_user_sequences(args.pairs_file):
        top_item_counts.update(item_id for item_id in seq if item_id in available_items)
    top_item_order = [item_id for item_id, _ in top_item_counts.most_common(args.top_items)]
    allowed_items = set(top_item_order)

    selected = select_users(
        args.pairs_file,
        allowed_items,
        args.min_user_interactions,
        args.target_interactions,
        args.seed,
    )
    user_map, item_map = build_mappings(selected, top_item_order)

    split_counts = write_inter_and_splits(
        target_dir,
        args.dataset_name,
        selected,
        user_map,
        item_map,
    )
    write_mappings(target_dir, user_map, item_map)
    image_shape = None
    text_shape = None
    if args.feature_file is not None:
        image_shape, text_shape = write_features(args.feature_file, target_dir, item_map)

    seq_lengths = np.array([len(seq) for seq in selected.values()], dtype=np.int64)
    metadata = {
        "source": "MicroLens-1M",
        "dataset_name": args.dataset_name,
        "pairs_file": str(args.pairs_file),
        "items_file": str(args.items_file),
        "title_file": str(args.title_file),
        "feature_file": str(args.feature_file) if args.feature_file is not None else None,
        "filter": {
            "top_items": args.top_items,
            "min_user_interactions": args.min_user_interactions,
            "target_interactions": args.target_interactions,
            "seed": args.seed,
        },
        "stats": {
            "users": len(user_map),
            "items": len(item_map),
            "interactions": int(split_counts["total"]),
            "train": int(split_counts["train"]),
            "val": int(split_counts["val"]),
            "test": int(split_counts["test"]),
            "avg_interactions_per_user": float(seq_lengths.mean()),
            "median_interactions_per_user": float(np.median(seq_lengths)),
            "min_interactions_per_user": int(seq_lengths.min()),
            "max_interactions_per_user": int(seq_lengths.max()),
            "image_feat_shape": list(image_shape) if image_shape is not None else None,
            "text_feat_shape": list(text_shape) if text_shape is not None else None,
        },
    }
    with (target_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print(json.dumps(metadata["stats"], indent=2))


if __name__ == "__main__":
    main()
