#!/usr/bin/env python3
"""Create phase-invariant payloads where every selected item misses one fixed modality."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


MODALITY_INDEX = {"image": 0, "text": 1}


def rate_tag(rate: float) -> str:
    return f"{rate:g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="Data")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--rates", nargs="+", type=float, required=True)
    parser.add_argument("--modalities", nargs="+", choices=sorted(MODALITY_INDEX), default=["image", "text"])
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    rates = sorted(set(args.rates))
    for rate in rates:
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"missing rate must be in [0,1], got {rate}")

    for dataset in args.datasets:
        data_dir = data_root / dataset
        image = np.load(data_dir / "image_feat.npy", mmap_mode="r")
        text = np.load(data_dir / "text_feat.npy", mmap_mode="r")
        if image.ndim != 2 or text.ndim != 2 or image.shape[0] != text.shape[0]:
            raise ValueError(f"invalid image/text feature shapes for {dataset}: {image.shape}, {text.shape}")
        n_items = int(image.shape[0])
        item_order = np.random.default_rng(args.seed).permutation(n_items).astype(np.int64)
        manifest_dir = data_dir / "missing_protocols"
        manifest_dir.mkdir(exist_ok=True)

        for rate in rates:
            tag = rate_tag(rate)
            selected = item_order[: int(n_items * rate)].copy()
            for modality in args.modalities:
                index = MODALITY_INDEX[modality]
                indicators = np.full(selected.shape, index, dtype=np.int64)
                mask = np.zeros(n_items, dtype=bool)
                mask[selected] = True
                payload_name = (
                    f"fixed_modality_missing_items_mr{tag}_seed{args.seed}_miss{modality}.npy"
                )
                payload = {
                    "protocol": "unified_single_modality",
                    "protocol_variant": "fixed_missing_modality",
                    "dataset": dataset,
                    "missing_rate": rate,
                    "seed": args.seed,
                    "modalities": ["image", "text"],
                    "fixed_missing_modality": modality,
                    "items": selected,
                    "indicator": indicators,
                    "shared_across_splits": True,
                }
                np.save(data_dir / payload_name, payload, allow_pickle=True)
                np.save(
                    data_dir / f"fixed_modality_mask_mr{tag}_seed{args.seed}_miss{modality}.npy",
                    mask,
                )
                manifest = {
                    "protocol": "unified_single_modality",
                    "protocol_variant": "fixed_missing_modality",
                    "dataset": dataset,
                    "missing_rate": rate,
                    "seed": args.seed,
                    "fixed_missing_modality": modality,
                    "num_items": n_items,
                    "num_selected_items": int(selected.size),
                    "shared_across_splits": True,
                    "exactly_one_missing_modality_per_selected_item": True,
                    "payload_file": payload_name,
                    "selected_items_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
                    "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                }
                manifest_path = manifest_dir / (
                    f"fixed_{modality}_unified_mr{tag}_seed{args.seed}.json"
                )
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"{dataset} mr={tag} miss={modality}: "
                    f"selected={selected.size}/{n_items} payload={payload_name}"
                )


if __name__ == "__main__":
    main()
