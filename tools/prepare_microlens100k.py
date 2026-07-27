#!/usr/bin/env python3
"""Prepare MicroLens-100k for the local MMRec/I3 data format."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np


SPLIT_LABELS = {"train": 0, "val": 1, "test": 2}


def iter_user_sequences(pairs_file: Path):
    with pairs_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_user, seq = line.split("\t", 1)
            items = [int(item_id) for item_id in seq.split()]
            if len(items) >= 3:
                yield int(raw_user), items


def write_split(path: Path, split_data: dict[int, list[int]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for user_id in sorted(split_data):
            items = split_data[user_id]
            f.write(f"{user_id} {' '.join(str(item_id) for item_id in items)}\n")
            count += len(items)
    return count


def load_feature_from_zip(feature_zip: Path, name_pattern: str) -> np.ndarray:
    with zipfile.ZipFile(feature_zip) as zf:
        matches = [
            name
            for name in zf.namelist()
            if name.endswith(".npy") and name_pattern.lower() in Path(name).name.lower()
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one {name_pattern!r} .npy in {feature_zip}, found {matches}")
        with zf.open(matches[0]) as f:
            return np.load(f).astype(np.float32)


def prepare(args: argparse.Namespace) -> dict:
    target_dir: Path = args.target_dir
    if target_dir.exists() and any(target_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{target_dir} is not empty; pass --overwrite to replace files")
    target_dir.mkdir(parents=True, exist_ok=True)

    image_feat_all = load_feature_from_zip(args.feature_zip, "image")
    text_feat_all = load_feature_from_zip(args.feature_zip, "text")
    video_feat_all = load_feature_from_zip(args.feature_zip, "video")
    if not (len(image_feat_all) == len(text_feat_all) == len(video_feat_all)):
        raise ValueError(
            "Feature row counts differ: "
            f"image={len(image_feat_all)} text={len(text_feat_all)} video={len(video_feat_all)}"
        )

    # MicroLens-100k feature rows follow raw item ids 1..N in the official files.
    max_feature_item = len(image_feat_all)
    selected: dict[int, list[int]] = {}
    for raw_user, seq in iter_user_sequences(args.pairs_file):
        kept = [item for item in seq if 1 <= item <= max_feature_item]
        if len(kept) >= args.min_user_interactions:
            selected[raw_user] = kept

    if args.max_users > 0:
        selected = dict(sorted(selected.items())[: args.max_users])

    user_map = {raw_id: idx for idx, raw_id in enumerate(sorted(selected))}
    raw_items = sorted({item for seq in selected.values() for item in seq})
    item_map = {raw_id: idx for idx, raw_id in enumerate(raw_items)}

    splits: dict[str, dict[int, list[int]]] = {"train": {}, "val": {}, "test": {}}
    inter_file = target_dir / f"{args.dataset_name}.inter"
    with inter_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["userID", "itemID", "rating", "timestamp", "x_label"])
        for raw_user in sorted(selected):
            user_id = user_map[raw_user]
            mapped = [item_map[item_id] for item_id in selected[raw_user]]
            parts = {"train": mapped[:-2], "val": [mapped[-2]], "test": [mapped[-1]]}
            for split_name, items in parts.items():
                splits[split_name][user_id] = items
                for pos, item_id in enumerate(items):
                    writer.writerow([user_id, item_id, "1.0", pos, SPLIT_LABELS[split_name]])

    split_counts = {
        split_name: write_split(target_dir / f"{split_name}.txt", split_data)
        for split_name, split_data in splits.items()
    }
    split_counts["total"] = sum(split_counts.values())

    raw_item_indices = np.array([raw_id - 1 for raw_id in raw_items], dtype=np.int64)
    image_feat = image_feat_all[raw_item_indices]
    text_feat = text_feat_all[raw_item_indices]
    video_feat = video_feat_all[raw_item_indices]
    np.save(target_dir / "image_feat.npy", image_feat)
    np.save(target_dir / "text_feat.npy", text_feat)
    np.save(target_dir / "video_feat.npy", video_feat)
    # I3's third-modality path is named audio; keep a compatibility alias.
    np.save(target_dir / "audio_feat.npy", video_feat)

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

    seq_lengths = np.array([len(seq) for seq in selected.values()], dtype=np.int64)
    metadata = {
        "source": "MicroLens-100k",
        "dataset_name": args.dataset_name,
        "pairs_file": str(args.pairs_file),
        "feature_zip": str(args.feature_zip),
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
            "image_feat_shape": list(image_feat.shape),
            "text_feat_shape": list(text_feat.shape),
            "video_feat_shape": list(video_feat.shape),
        },
    }
    with (target_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-file", type=Path, required=True)
    parser.add_argument("--feature-zip", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, default=Path("Data/microlens100k"))
    parser.add_argument("--dataset-name", default="microlens100k")
    parser.add_argument("--min-user-interactions", type=int, default=3)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    metadata = prepare(args)
    print(json.dumps(metadata["stats"], indent=2))


if __name__ == "__main__":
    main()
