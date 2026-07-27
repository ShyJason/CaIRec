#!/usr/bin/env python3
"""Prepare Amazon Beauty 5-core as an MMRec-style dataset."""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
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
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            rec = parse_record(line)
            user = rec.get("reviewerID")
            asin = rec.get("asin")
            rating = rec.get("overall")
            ts = rec.get("unixReviewTime", 0)
            if user and asin and rating is not None:
                rows.append(
                    {
                        "reviewerID": str(user),
                        "asin": str(asin),
                        "rating": float(rating),
                        "timestamp": int(ts or 0),
                    }
                )
    return rows


def load_meta(path: Path) -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            rec = parse_record(line)
            asin = rec.get("asin")
            if not asin or asin in meta:
                continue
            description = normalize_description(rec.get("description"))
            image_url = str(rec.get("imUrl") or "").strip()
            if description and image_url and image_url.lower() not in {"nan", "n/a", "null"}:
                meta[str(asin)] = {"description": description, "image_url": image_url}
    return meta


def download_one(asin: str, url: str, image_dir: Path, timeout: float) -> tuple[str, bool]:
    out = image_dir / f"{asin}.jpg"
    if out.exists() and out.stat().st_size > 0:
        return asin, True
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200 or not response.content:
            return asin, False
        tmp = out.with_suffix(".jpg.tmp")
        tmp.write_bytes(response.content)
        with Image.open(tmp) as img:
            img.verify()
        tmp.replace(out)
        return asin, True
    except Exception:
        try:
            out.with_suffix(".jpg.tmp").unlink()
        except FileNotFoundError:
            pass
        return asin, False


def download_images(meta: dict[str, dict[str, str]], image_dir: Path, workers: int, timeout: float) -> set[str]:
    image_dir.mkdir(parents=True, exist_ok=True)
    ok: set[str] = set()
    items = sorted(meta.items())
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(download_one, asin, info["image_url"], image_dir, timeout): asin
            for asin, info in items
        }
        done = 0
        for future in as_completed(futures):
            asin, success = future.result()
            done += 1
            if success:
                ok.add(asin)
            if done % 500 == 0 or done == len(items):
                print(f"[download] {done}/{len(items)} ok={len(ok)}", flush=True)
    return ok


def dedupe_reviews(rows: list[dict], valid_asins: set[str]) -> list[dict]:
    latest: dict[tuple[str, str], dict] = {}
    for row in rows:
        if row["asin"] not in valid_asins:
            continue
        key = (row["reviewerID"], row["asin"])
        prev = latest.get(key)
        if prev is None or row["timestamp"] >= prev["timestamp"]:
            latest[key] = row
    return list(latest.values())


def filter_users(rows: list[dict], min_user_interactions: int) -> list[dict]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["reviewerID"]] += 1
    return [row for row in rows if counts[row["reviewerID"]] >= min_user_interactions]


def split_rows(rows: list[dict]) -> list[dict]:
    by_user: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_user[row["reviewerID"]].append(row)
    split = []
    for user_rows in by_user.values():
        ordered = sorted(user_rows, key=lambda x: (x["timestamp"], x["asin"]))
        n = len(ordered)
        for idx, row in enumerate(ordered):
            out = dict(row)
            if idx == n - 1:
                out["x_label"] = 2
            elif idx == n - 2:
                out["x_label"] = 1
            else:
                out["x_label"] = 0
            split.append(out)
    return split


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        with Image.open(path) as img:
            return self.transform(img.convert("RGB"))


def extract_image_features(asins: list[str], image_dir: Path, batch_size: int, device: str) -> np.ndarray:
    weights = models.VGG16_Weights.DEFAULT
    vgg = models.vgg16(weights=weights)
    model = torch.nn.Sequential(vgg.features, vgg.avgpool, torch.nn.Flatten(), *list(vgg.classifier.children())[:-1])
    model.eval().to(device)
    dataset = ImagePathDataset([image_dir / f"{asin}.jpg" for asin in asins], weights.transforms())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=device.startswith("cuda"))
    feats = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            out = model(batch.to(device, non_blocking=True)).detach().cpu().numpy()
            feats.append(out)
            if batch_idx % 20 == 0:
                print(f"[image-feat] batches={batch_idx}", flush=True)
    return np.concatenate(feats, axis=0).astype(np.float64)


def write_grouped(path: Path, rows: list[dict], label: int) -> int:
    grouped: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        if row["x_label"] == label:
            grouped[row["userID"]].append(row["itemID"])
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for user in sorted(grouped):
            items = grouped[user]
            f.write(f"{user} {' '.join(str(item) for item in items)}\n")
            count += len(items)
    return count


def write_mapping(path: Path, header: tuple[str, str], pairs: list[tuple[str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("/home/ruiyuliu/projects/Graph-Missing-Modalities/data/Beauty"))
    parser.add_argument("--target-dir", type=Path, default=Path("/home/ruiyuliu/projects/MMRec/Data/beauty"))
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-user-interactions", type=int, default=3)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    source = args.source_dir
    target = args.target_dir
    cache = target / "_raw"
    image_dir = cache / "images"
    target.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    reviews = load_reviews(source / "reviews_Beauty_5.json.gz")
    meta = load_meta(source / "meta_Beauty.json.gz")
    candidate_asins = {row["asin"] for row in reviews}.intersection(meta)
    meta = {asin: meta[asin] for asin in candidate_asins}
    print(f"[load] reviews={len(reviews)} meta_with_desc_image={len(meta)}", flush=True)

    image_ok = download_images(meta, image_dir, args.workers, args.timeout)
    rows = filter_users(dedupe_reviews(reviews, image_ok), args.min_user_interactions)
    rows = split_rows(rows)
    users = sorted({row["reviewerID"] for row in rows})
    items = sorted({row["asin"] for row in rows})
    user_map = {user: idx for idx, user in enumerate(users)}
    item_map = {asin: idx for idx, asin in enumerate(items)}

    indexed = []
    for row in rows:
        if row["reviewerID"] in user_map and row["asin"] in item_map:
            indexed.append(
                {
                    "userID": user_map[row["reviewerID"]],
                    "itemID": item_map[row["asin"]],
                    "rating": row["rating"],
                    "timestamp": row["timestamp"],
                    "x_label": row["x_label"],
                }
            )
    indexed.sort(key=lambda x: (x["userID"], x["timestamp"], x["itemID"]))
    print(f"[index] users={len(users)} items={len(items)} interactions={len(indexed)}", flush=True)

    with (target / "beauty.inter").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["userID", "itemID", "rating", "timestamp", "x_label"], delimiter="\t")
        writer.writeheader()
        writer.writerows(indexed)
    split_counts = {
        "train": write_grouped(target / "train.txt", indexed, 0),
        "val": write_grouped(target / "val.txt", indexed, 1),
        "test": write_grouped(target / "test.txt", indexed, 2),
    }
    write_mapping(target / "u_id_mapping.csv", ("user_id", "userID"), [(user, user_map[user]) for user in users])
    write_mapping(target / "i_id_mapping.csv", ("asin", "itemID"), [(asin, item_map[asin]) for asin in items])

    ordered_asins = items
    descriptions = [meta[asin]["description"] for asin in ordered_asins]
    print("[text-feat] loading sentence-transformers/all-MiniLM-L6-v2", flush=True)
    text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=args.device)
    text_feat = text_model.encode(
        descriptions,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    np.save(target / "text_feat.npy", text_feat)

    print("[image-feat] loading VGG16 and encoding images", flush=True)
    image_feat = extract_image_features(ordered_asins, image_dir, args.batch_size, args.device)
    np.save(target / "image_feat.npy", image_feat)

    summary = {
        "dataset": "beauty",
        "users": len(users),
        "items": len(items),
        "interactions": len(indexed),
        "splits": split_counts,
        "image_success_items": len(image_ok),
        "image_feature_shape": list(image_feat.shape),
        "text_feature_shape": list(text_feat.shape),
    }
    (target / "prepare_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
