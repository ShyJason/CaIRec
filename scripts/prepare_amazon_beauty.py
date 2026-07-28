#!/usr/bin/env python3
"""Prepare the Amazon Beauty 5-core source data in CaIRec's dataset layout."""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from torchvision import models


RAW_FILES = {
    "reviews_Beauty_5.json.gz": (
        "e5925ec99023f1dc9c7d3dff7ef34cfeacd40d1b981b5525e1e4ba9c8abc18fe"
    ),
    "meta_Beauty.json.gz": (
        "c7977dc0e0ead14ac7df8c1c6de74da00e4bebfad87a0f0c5763d4c5bc9b53a0"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_raw_files(source: Path) -> None:
    for filename, expected in RAW_FILES.items():
        path = source / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing raw Beauty source: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for {path}: expected {expected}, got {actual}. "
                "Check that this is the 2014 Amazon Beauty source."
            )


def parse_record(line: str) -> dict:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)


def normalize_description(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(part) for part in value if part is not None)
    text = str(value).strip()
    if text.lower() in {"nan", "n/a", "null"}:
        return ""
    return text


def load_reviews(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = parse_record(line)
            user = record.get("reviewerID")
            asin = record.get("asin")
            rating = record.get("overall")
            timestamp = record.get("unixReviewTime", 0)
            if user and asin and rating is not None:
                rows.append(
                    {
                        "reviewerID": str(user),
                        "asin": str(asin),
                        "rating": float(rating),
                        "timestamp": int(timestamp or 0),
                    }
                )
    return rows


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = parse_record(line)
            asin = record.get("asin")
            if not asin or asin in metadata:
                continue
            description = normalize_description(record.get("description"))
            image_url = str(record.get("imUrl") or "").strip()
            if (
                description
                and image_url
                and image_url.lower() not in {"nan", "n/a", "null"}
            ):
                metadata[str(asin)] = {
                    "description": description,
                    "image_url": image_url,
                }
    return metadata


def download_one(
    asin: str, url: str, image_dir: Path, timeout: float
) -> tuple[str, bool]:
    output = image_dir / f"{asin}.jpg"
    if output.exists() and output.stat().st_size > 0:
        return asin, True
    temporary = output.with_suffix(".jpg.tmp")
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200 or not response.content:
            return asin, False
        temporary.write_bytes(response.content)
        with Image.open(temporary) as image:
            image.verify()
        temporary.replace(output)
        return asin, True
    except Exception:
        temporary.unlink(missing_ok=True)
        return asin, False


def download_images(
    metadata: dict[str, dict[str, str]],
    image_dir: Path,
    workers: int,
    timeout: float,
) -> set[str]:
    image_dir.mkdir(parents=True, exist_ok=True)
    downloaded: set[str] = set()
    items = sorted(metadata.items())
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_one, asin, info["image_url"], image_dir, timeout
            ): asin
            for asin, info in items
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            asin, success = future.result()
            if success:
                downloaded.add(asin)
            if completed % 500 == 0 or completed == len(items):
                print(
                    f"[download] {completed}/{len(items)} ok={len(downloaded)}",
                    flush=True,
                )
    return downloaded


def deduplicate_reviews(rows: list[dict], valid_asins: set[str]) -> list[dict]:
    latest: dict[tuple[str, str], dict] = {}
    for row in rows:
        if row["asin"] not in valid_asins:
            continue
        key = (row["reviewerID"], row["asin"])
        previous = latest.get(key)
        if previous is None or row["timestamp"] >= previous["timestamp"]:
            latest[key] = row
    return list(latest.values())


def filter_users(rows: list[dict], minimum: int) -> list[dict]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["reviewerID"]] += 1
    return [row for row in rows if counts[row["reviewerID"]] >= minimum]


def split_rows(rows: list[dict]) -> list[dict]:
    by_user: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_user[row["reviewerID"]].append(row)
    split = []
    for user_rows in by_user.values():
        ordered = sorted(user_rows, key=lambda row: (row["timestamp"], row["asin"]))
        for index, row in enumerate(ordered):
            output = dict(row)
            if index == len(ordered) - 1:
                output["x_label"] = 2
            elif index == len(ordered) - 2:
                output["x_label"] = 1
            else:
                output["x_label"] = 0
            split.append(output)
    return split


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def extract_image_features(
    asins: list[str], image_dir: Path, batch_size: int, device: str
) -> np.ndarray:
    weights = models.VGG16_Weights.DEFAULT
    vgg = models.vgg16(weights=weights)
    model = torch.nn.Sequential(
        vgg.features,
        vgg.avgpool,
        torch.nn.Flatten(),
        *list(vgg.classifier.children())[:-1],
    )
    model.eval().to(device)
    dataset = ImagePathDataset(
        [image_dir / f"{asin}.jpg" for asin in asins], weights.transforms()
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.startswith("cuda"),
    )
    features = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            output = model(batch.to(device, non_blocking=True))
            features.append(output.detach().cpu().numpy())
            if batch_index % 20 == 0:
                print(f"[image-feat] batches={batch_index}", flush=True)
    return np.concatenate(features, axis=0).astype(np.float64)


def write_grouped(path: Path, rows: list[dict], label: int) -> int:
    grouped: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        if row["x_label"] == label:
            grouped[row["userID"]].append(row["itemID"])
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for user in sorted(grouped):
            items = grouped[user]
            stream.write(f"{user} {' '.join(str(item) for item in items)}\n")
            count += len(items)
    return count


def write_mapping(
    path: Path, header: tuple[str, str], pairs: list[tuple[str, int]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(pairs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, default=Path("Data/beauty"))
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-user-interactions", type=int, default=3)
    parser.add_argument(
        "--skip-raw-checksums",
        action="store_true",
        help="Allow a different source revision. Exact reproduction is not guaranteed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    source = args.source_dir.resolve()
    target = args.target_dir.resolve()
    if not args.skip_raw_checksums:
        verify_raw_files(source)

    image_dir = target / "_raw" / "images"
    target.mkdir(parents=True, exist_ok=True)

    reviews = load_reviews(source / "reviews_Beauty_5.json.gz")
    metadata = load_metadata(source / "meta_Beauty.json.gz")
    candidate_asins = {row["asin"] for row in reviews}.intersection(metadata)
    metadata = {asin: metadata[asin] for asin in candidate_asins}
    print(
        f"[load] reviews={len(reviews)} meta_with_desc_image={len(metadata)}",
        flush=True,
    )

    image_asins = download_images(metadata, image_dir, args.workers, args.timeout)
    rows = filter_users(
        deduplicate_reviews(reviews, image_asins), args.min_user_interactions
    )
    rows = split_rows(rows)
    users = sorted({row["reviewerID"] for row in rows})
    items = sorted({row["asin"] for row in rows})
    user_map = {user: index for index, user in enumerate(users)}
    item_map = {asin: index for index, asin in enumerate(items)}

    indexed = []
    for row in rows:
        indexed.append(
            {
                "userID": user_map[row["reviewerID"]],
                "itemID": item_map[row["asin"]],
                "rating": row["rating"],
                "timestamp": row["timestamp"],
                "x_label": row["x_label"],
            }
        )
    indexed.sort(key=lambda row: (row["userID"], row["timestamp"], row["itemID"]))
    print(
        f"[index] users={len(users)} items={len(items)} "
        f"interactions={len(indexed)}",
        flush=True,
    )

    with (target / "beauty.inter").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["userID", "itemID", "rating", "timestamp", "x_label"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(indexed)

    split_counts = {
        "train": write_grouped(target / "train.txt", indexed, 0),
        "val": write_grouped(target / "val.txt", indexed, 1),
        "test": write_grouped(target / "test.txt", indexed, 2),
    }
    write_mapping(
        target / "u_id_mapping.csv",
        ("user_id", "userID"),
        [(user, user_map[user]) for user in users],
    )
    write_mapping(
        target / "i_id_mapping.csv",
        ("asin", "itemID"),
        [(asin, item_map[asin]) for asin in items],
    )

    descriptions = [metadata[asin]["description"] for asin in items]
    print(
        "[text-feat] loading sentence-transformers/all-MiniLM-L6-v2",
        flush=True,
    )
    text_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=args.device
    )
    text_features = text_model.encode(
        descriptions,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    np.save(target / "text_feat.npy", text_features)

    print("[image-feat] loading VGG16 and encoding images", flush=True)
    image_features = extract_image_features(
        items, image_dir, args.batch_size, args.device
    )
    np.save(target / "image_feat.npy", image_features)

    summary = {
        "dataset": "beauty",
        "users": len(users),
        "items": len(items),
        "interactions": len(indexed),
        "splits": split_counts,
        "image_success_items": len(image_asins),
        "image_feature_shape": list(image_features.shape),
        "text_feature_shape": list(text_features.shape),
    }
    (target / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
