#!/usr/bin/env python3
"""Download and verify the assets required to reproduce CaIRec."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORY = "ShyJason/CaIRec"
DEFAULT_RELEASE_TAG = "v1.0-assets"


@dataclass(frozen=True)
class Asset:
    filename: str
    sha256: str


DATASETS = {
    "clothing": Asset(
        "cair-data-clothing-v1.tar.gz",
        "cc74bf8c5545a592adb8e761c4f859abf93d3c8452fca32ff9e8943d133cb25a",
    ),
    "beauty": Asset(
        "cair-data-beauty-v1.tar.gz",
        "8882b52b39c6d4bc39e1fc959a487f80c74b056e4990c883eb01648962e186ec",
    ),
    "sports": Asset(
        "cair-data-sports-v1.tar.gz",
        "578360882d38c442222cc4ff6f49a43e41e87b97767ec70b1c7fb5929fd8e2b7",
    ),
}

CHECKPOINTS = Asset(
    "cair-projection-checkpoints-v1.tar.gz",
    "e5efd6d03e346a393857e7e09b9f1b46e779eaf59510a7cdc0eb0575584e6a17",
)

REQUIRED_DATA_FILES = (
    "train.txt",
    "val.txt",
    "test.txt",
    "image_feat.npy",
    "text_feat.npy",
    "u_id_mapping.csv",
    "i_id_mapping.csv",
    "unified_missing_items_mr0.5_seed2023.npy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CaIRec datasets and projection checkpoints."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", *DATASETS],
        help="Datasets to install. Defaults to all retained paper datasets.",
    )
    parser.add_argument(
        "--with-checkpoints",
        action="store_true",
        help="Also download the projection checkpoint bundle.",
    )
    parser.add_argument(
        "--repository",
        default=DEFAULT_REPOSITORY,
        help="GitHub owner/repository containing the release.",
    )
    parser.add_argument(
        "--release-tag",
        default=DEFAULT_RELEASE_TAG,
        help="GitHub Release tag to download.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=REPO_ROOT,
        help="CaIRec root. Data and checkpoints are installed below this directory.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / ".release_downloads",
        help="Download cache directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing installed dataset or checkpoint directory.",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep verified release archives after installation.",
    )
    return parser.parse_args()


def selected_datasets(names: list[str]) -> list[str]:
    if "all" in names:
        return list(DATASETS)
    return list(dict.fromkeys(names))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_url(repository: str, tag: str, filename: str) -> str:
    return f"https://github.com/{repository}/releases/download/{tag}/{filename}"


def download(asset: Asset, repository: str, tag: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / asset.filename

    if destination.is_file() and sha256_file(destination) == asset.sha256:
        print(f"verified cache: {destination}")
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    url = release_url(repository, tag, asset.filename)
    print(f"download: {url}")

    request = urllib.request.Request(url, headers={"User-Agent": "CaIRec-downloader"})
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    actual = sha256_file(partial)
    if actual != asset.sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {asset.filename}: expected {asset.sha256}, got {actual}"
        )
    os.replace(partial, destination)
    print(f"verified: {destination}")
    return destination


def safe_members(archive: tarfile.TarFile, expected_prefix: PurePosixPath):
    for member in archive.getmembers():
        member_path = PurePosixPath(member.name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise RuntimeError(f"unsafe archive path: {member.name}")
        if member.issym() or member.islnk() or member.isdev():
            raise RuntimeError(f"unsupported archive member: {member.name}")
        if member_path != expected_prefix and expected_prefix not in member_path.parents:
            raise RuntimeError(
                f"archive member is outside {expected_prefix}: {member.name}"
            )
        yield member


def extract_subtree(
    archive_path: Path,
    project_root: Path,
    relative_target: PurePosixPath,
    overwrite: bool,
) -> Path:
    target = project_root.joinpath(*relative_target.parts)
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass --overwrite to replace it")

    project_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".cair-extract-", dir=project_root
    ) as temporary:
        temporary_root = Path(temporary)
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(
                path=temporary_root,
                members=safe_members(archive, relative_target),
            )

        extracted = temporary_root.joinpath(*relative_target.parts)
        if not extracted.is_dir():
            raise RuntimeError(f"{archive_path} did not contain {relative_target}")
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted), str(target))
    print(f"installed: {target}")
    return target


def validate_dataset(path: Path, dataset: str) -> None:
    required = (f"{dataset}.inter", *REQUIRED_DATA_FILES)
    missing = [filename for filename in required if not (path / filename).is_file()]
    if missing:
        raise RuntimeError(f"{path} is missing required files: {', '.join(missing)}")


def install_dataset(
    dataset: str,
    archive_path: Path,
    project_root: Path,
    overwrite: bool,
) -> None:
    target = project_root / "Data" / dataset
    if target.is_dir() and not overwrite:
        try:
            validate_dataset(target, dataset)
        except RuntimeError:
            pass
        else:
            print(f"already installed: {target}")
            return
    installed = extract_subtree(
        archive_path,
        project_root,
        PurePosixPath("Data") / dataset,
        overwrite,
    )
    validate_dataset(installed, dataset)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    cache_dir = args.cache_dir.resolve()

    for dataset in selected_datasets(args.datasets):
        archive = download(
            DATASETS[dataset],
            args.repository,
            args.release_tag,
            cache_dir,
        )
        install_dataset(dataset, archive, project_root, args.overwrite)

    if args.with_checkpoints:
        archive = download(
            CHECKPOINTS,
            args.repository,
            args.release_tag,
            cache_dir,
        )
        extract_subtree(
            archive,
            project_root,
            PurePosixPath("projection_checkpoints"),
            args.overwrite,
        )

    if not args.keep_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"removed cache: {cache_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
