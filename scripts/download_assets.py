#!/usr/bin/env python3
"""Download CaIRec-owned auxiliary assets from a GitHub Release."""

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
DATASETS = ("clothing", "beauty", "sports")


@dataclass(frozen=True)
class Asset:
    filename: str
    sha256: str


PAYLOADS = Asset(
    "cair-missing-payloads-v1.tar.gz",
    "cbaa43381e5124d0f576ed3195bca21acfe6feef0c5fd4f2cd6d2499f9cd084a",
)
CHECKPOINTS = Asset(
    "cair-projection-checkpoints-v1.tar.gz",
    "ab8029835b33ad4ea9878e2e53df2726d6c60486f842c78c8cd5aebff8b23465",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download CaIRec missing-item payloads and optional projection "
            "checkpoints. Upstream datasets are not downloaded by this script."
        )
    )
    parser.add_argument(
        "--payloads",
        nargs="+",
        choices=["all", *DATASETS],
        help="Install fixed missing-item payloads for the selected datasets.",
    )
    parser.add_argument(
        "--with-checkpoints",
        action="store_true",
        help="Restore the projection checkpoint directory from the release mirror.",
    )
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--release-tag", default=DEFAULT_RELEASE_TAG)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--cache-dir", type=Path, default=REPO_ROOT / ".release_downloads"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing installed files."
    )
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()
    if not args.payloads and not args.with_checkpoints:
        parser.error("select --payloads DATASET... and/or --with-checkpoints")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    asset: Asset, repository: str, tag: str, cache_dir: Path
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / asset.filename
    if destination.is_file() and sha256_file(destination) == asset.sha256:
        print(f"verified cache: {destination}")
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    url = f"https://github.com/{repository}/releases/download/{tag}/{asset.filename}"
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
            f"SHA-256 mismatch for {asset.filename}: "
            f"expected {asset.sha256}, got {actual}"
        )
    os.replace(partial, destination)
    print(f"verified: {destination}")
    return destination


def selected_datasets(names: list[str]) -> set[str]:
    if "all" in names:
        return set(DATASETS)
    return set(names)


def payload_member(dataset: str) -> PurePosixPath:
    return (
        PurePosixPath("Data")
        / dataset
        / "unified_missing_items_mr0.5_seed2023.npy"
    )


def install_payloads(
    archive_path: Path,
    project_root: Path,
    datasets: set[str],
    overwrite: bool,
) -> None:
    expected = {str(payload_member(dataset)): dataset for dataset in datasets}
    found: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        unexpected = set(members).difference(
            str(payload_member(dataset)) for dataset in DATASETS
        )
        if unexpected:
            raise RuntimeError(
                "payload archive contains unexpected paths: "
                + ", ".join(sorted(unexpected))
            )
        for member_name, dataset in expected.items():
            member = members.get(member_name)
            if member is None or not member.isfile():
                raise RuntimeError(f"payload archive is missing {member_name}")
            target = project_root.joinpath(*PurePosixPath(member_name).parts)
            if not target.parent.is_dir():
                raise FileNotFoundError(
                    f"{target.parent} does not exist; install the upstream "
                    f"{dataset} dataset first"
                )
            if target.exists() and not overwrite:
                print(f"already installed: {target}")
                found.add(dataset)
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"could not read {member_name}")
            target.write_bytes(extracted.read())
            print(f"installed: {target}")
            found.add(dataset)
    if found != datasets:
        raise RuntimeError("not every requested payload was installed")


def install_checkpoints(
    archive_path: Path, project_root: Path, overwrite: bool
) -> None:
    prefix = PurePosixPath("ckpt")
    target = project_root / "ckpt"
    if target.exists() and not overwrite:
        print(f"already installed: {target}")
        return

    with tempfile.TemporaryDirectory(
        prefix=".cair-extract-", dir=project_root
    ) as temporary:
        temporary_root = Path(temporary)
        with tarfile.open(archive_path, "r:gz") as archive:
            members = []
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or (path != prefix and prefix not in path.parents)
                    or member.issym()
                    or member.islnk()
                    or member.isdev()
                ):
                    raise RuntimeError(f"unsafe archive member: {member.name}")
                members.append(member)
            archive.extractall(temporary_root, members=members)
        extracted = temporary_root / "ckpt"
        if not extracted.is_dir():
            raise RuntimeError("checkpoint archive has no ckpt/")
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(extracted), str(target))
        print(f"installed: {target}")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    cache_dir = args.cache_dir.resolve()

    if args.payloads:
        archive = download(
            PAYLOADS, args.repository, args.release_tag, cache_dir
        )
        install_payloads(
            archive,
            project_root,
            selected_datasets(args.payloads),
            args.overwrite,
        )

    if args.with_checkpoints:
        archive = download(
            CHECKPOINTS, args.repository, args.release_tag, cache_dir
        )
        install_checkpoints(archive, project_root, args.overwrite)

    if not args.keep_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"removed cache: {cache_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
