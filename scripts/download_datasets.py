#!/usr/bin/env python3
"""Download and prepare the datasets used by this project."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BABY_CLOTHING_URL = (
    "https://drive.google.com/drive/folders/"
    "13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing"
)
DEFAULT_TIKTOK_URL = (
    "https://drive.google.com/drive/folders/"
    "11wEn5k1Kzusj1GkdAlCcfS3GbBWGzFpX?usp=drive_link"
)

DATASET_REQUIREMENTS = {
    "baby": [
        "baby.inter",
        "train.txt",
        "val.txt",
        "test.txt",
        "image_feat.npy",
        "text_feat.npy",
        "u_id_mapping.csv",
        "i_id_mapping.csv",
    ],
    "clothing": [
        "clothing.inter",
        "train.txt",
        "val.txt",
        "test.txt",
        "image_feat.npy",
        "text_feat.npy",
        "u_id_mapping.csv",
        "i_id_mapping.csv",
    ],
    "sports": [
        "sports.inter",
        "train.txt",
        "val.txt",
        "test.txt",
        "image_feat.npy",
        "text_feat.npy",
        "u_id_mapping.csv",
        "i_id_mapping.csv",
    ],
    "tiktok": [
        "train.txt",
        "val.txt",
        "test.txt",
        "image_feat.npy",
        "text_feat.npy",
        "audio_feat.npy",
    ],
}

SPORTS_SOURCE_REQUIREMENTS = [
    "sports.inter",
    "image_feat.npy",
    "text_feat.npy",
    "u_id_mapping.csv",
    "i_id_mapping.csv",
]

SPLIT_NAMES = {
    "0": "train",
    "1": "val",
    "2": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Baby, Clothing, Sports, and TikTok datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", "baby", "clothing", "sports", "tiktok"],
        help="Datasets to download/prepare.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "Data",
        help="Target data directory. Defaults to Data under the repo root.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / ".dataset_downloads",
        help="Temporary download cache directory.",
    )
    parser.add_argument(
        "--baby-clothing-url",
        default=os.environ.get("MMREC_BABY_CLOTHING_URL", DEFAULT_BABY_CLOTHING_URL),
        help="Google Drive folder URL containing Baby/Clothing data.",
    )
    parser.add_argument(
        "--tiktok-url",
        default=os.environ.get("MMREC_TIKTOK_URL", DEFAULT_TIKTOK_URL),
        help="Google Drive folder URL containing TikTok data.",
    )
    parser.add_argument(
        "--sports-url",
        default=os.environ.get("MMREC_SPORTS_URL"),
        help=(
            "Optional Google Drive folder URL containing Sports data. If omitted, "
            "Sports is prepared from a BM3 sports directory in the Baby/Clothing download."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing dataset directories/files.",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep the temporary download cache after installation.",
    )
    parser.add_argument(
        "--no-cookies",
        action="store_true",
        help="Disable gdown cookies.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only install/prepare from existing cache and Data directories.",
    )
    return parser.parse_args()


def selected_datasets(names: list[str]) -> list[str]:
    if "all" in names:
        return ["baby", "clothing", "sports", "tiktok"]
    return list(dict.fromkeys(names))


def require_gdown():
    try:
        import gdown  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "gdown is required for Google Drive folder downloads. "
            "Install it with: pip install gdown"
        ) from exc
    return gdown


def has_files(directory: Path, filenames: list[str]) -> bool:
    return directory.is_dir() and all((directory / name).exists() for name in filenames)


def dataset_installed(data_dir: Path, dataset: str) -> bool:
    return has_files(data_dir / dataset, DATASET_REQUIREMENTS[dataset])


def download_folder(url: str, output: Path, use_cookies: bool, skip_download: bool) -> None:
    if skip_download:
        print(f"skip download: {output}")
        return

    gdown = require_gdown()
    output.mkdir(parents=True, exist_ok=True)
    print(f"download: {url}")
    print(f"output: {output}")
    files = gdown.download_folder(
        url=url,
        output=str(output),
        quiet=False,
        use_cookies=use_cookies,
        resume=True,
    )
    if not files:
        raise RuntimeError(f"No files were downloaded from {url}")


def iter_dirs(root: Path):
    if root.exists():
        yield root
        yield from (path for path in root.rglob("*") if path.is_dir())


def find_dataset_dir(roots: list[Path], dataset: str) -> Path | None:
    candidates: list[Path] = []
    for root in roots:
        for path in iter_dirs(root):
            if has_files(path, DATASET_REQUIREMENTS[dataset]):
                candidates.append(path)

    if not candidates:
        return None

    candidates.sort(key=lambda path: (path.name.lower() != dataset, len(path.parts)))
    return candidates[0]


def copy_dataset(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite and has_files(dst, DATASET_REQUIREMENTS[dst.name]):
            print(f"exists: {dst}")
            return
        if not overwrite:
            raise FileExistsError(f"{dst} exists but is incomplete; use --overwrite")
        shutil.rmtree(dst)

    print(f"install: {src} -> {dst}")
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
    shutil.copytree(src, dst, ignore=ignore)


def find_sports_source(roots: list[Path]) -> Path | None:
    candidates: list[Path] = []
    for root in roots:
        for path in iter_dirs(root):
            if has_files(path, SPORTS_SOURCE_REQUIREMENTS):
                candidates.append(path)

    if not candidates:
        return None

    candidates.sort(key=lambda path: (path.name.lower() != "sports", len(path.parts)))
    return candidates[0]


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


def read_sports_splits(inter_file: Path) -> dict[str, dict[int, list[int]]]:
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
            if split_name is not None:
                splits[split_name][int(row["userID"])].append(int(row["itemID"]))
    return splits


def write_split(path: Path, split: dict[int, list[int]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for user_id in sorted(split):
            items = split[user_id]
            if items:
                f.write(f"{user_id} {' '.join(str(item_id) for item_id in items)}\n")
                count += len(items)
            else:
                f.write(f"{user_id}\n")
    return count


def prepare_sports(source: Path, target: Path, overwrite: bool) -> None:
    if target.exists() and not overwrite and dataset_installed(target.parent, "sports"):
        print(f"exists: {target}")
        return

    target.mkdir(parents=True, exist_ok=True)
    print(f"prepare sports: {source} -> {target}")
    for name in SPORTS_SOURCE_REQUIREMENTS:
        status = link_or_copy(source / name, target / name, overwrite)
        print(f"  {name}: {status}")

    splits = read_sports_splits(source / "sports.inter")
    for split_name, split_data in splits.items():
        count = write_split(target / f"{split_name}.txt", split_data)
        print(f"  {split_name}.txt: {count} interactions")


def install_dataset_from_roots(
    dataset: str,
    roots: list[Path],
    data_dir: Path,
    overwrite: bool,
) -> None:
    src = find_dataset_dir(roots, dataset)
    if src is None:
        searched = ", ".join(str(root) for root in roots)
        raise FileNotFoundError(f"Could not find a prepared {dataset} directory in: {searched}")
    copy_dataset(src, data_dir / dataset, overwrite)


def main() -> None:
    args = parse_args()
    datasets = selected_datasets(args.datasets)
    data_dir = args.data_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    baby_cache = cache_dir / "baby_clothing"
    tiktok_cache = cache_dir / "tiktok"
    sports_cache = cache_dir / "sports"

    needs_baby_clothing = any(
        dataset in datasets and (args.overwrite or not dataset_installed(data_dir, dataset))
        for dataset in ("baby", "clothing")
    )
    needs_sports = "sports" in datasets and (
        args.overwrite or not dataset_installed(data_dir, "sports")
    )
    needs_tiktok = "tiktok" in datasets and (
        args.overwrite or not dataset_installed(data_dir, "tiktok")
    )

    if needs_baby_clothing or (needs_sports and not args.sports_url):
        download_folder(
            args.baby_clothing_url,
            baby_cache,
            use_cookies=not args.no_cookies,
            skip_download=args.skip_download,
        )
    if needs_sports and args.sports_url:
        download_folder(
            args.sports_url,
            sports_cache,
            use_cookies=not args.no_cookies,
            skip_download=args.skip_download,
        )
    if needs_tiktok:
        download_folder(
            args.tiktok_url,
            tiktok_cache,
            use_cookies=not args.no_cookies,
            skip_download=args.skip_download,
        )

    common_roots = [baby_cache, data_dir]
    if "baby" in datasets:
        install_dataset_from_roots("baby", common_roots, data_dir, args.overwrite)
    if "clothing" in datasets:
        install_dataset_from_roots("clothing", common_roots, data_dir, args.overwrite)
    if "sports" in datasets:
        sports_roots = [sports_cache, baby_cache, data_dir / "baby" / "BM3", data_dir]
        sports_source = find_sports_source(sports_roots)
        if sports_source is None:
            raise FileNotFoundError(
                "Could not find Sports source files. Provide --sports-url or make sure "
                "the Baby/Clothing download contains BM3/sports."
            )
        prepare_sports(sports_source, data_dir / "sports", args.overwrite)
    if "tiktok" in datasets:
        install_dataset_from_roots("tiktok", [tiktok_cache, data_dir], data_dir, args.overwrite)

    print("\nInstalled datasets:")
    for dataset in datasets:
        status = "ok" if dataset_installed(data_dir, dataset) else "missing"
        print(f"  {dataset}: {status} ({data_dir / dataset})")

    if not args.keep_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"removed cache: {cache_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
