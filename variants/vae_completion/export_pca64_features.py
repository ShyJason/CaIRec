#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


PHASES = ("phase_train", "phase_eval", "phase_graph")
MODALITIES = ("image", "text")


def l2_normalize(x, eps=1e-12):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def load_phase(root, phase, modality):
    return np.load(root / phase / f"{modality}_feat.npy").astype(np.float32, copy=False)


def load_mask(root, phase, modality):
    path = root / phase / f"{modality}_observed_mask.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing observed mask: {path}")
    return np.load(path).astype(bool, copy=False)


def fit_pca(train_features, observed_mask, n_components, seed):
    fit_features = train_features[observed_mask]
    if fit_features.shape[0] <= n_components:
        raise ValueError(
            f"not enough observed rows for PCA: {fit_features.shape[0]} <= {n_components}"
        )
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    pca.fit(fit_features)
    return pca


def transform_and_save(pca, src_root, out_root, phase, modality):
    phase_dir = out_root / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    features = load_phase(src_root, phase, modality)
    projected = pca.transform(features).astype(np.float32, copy=False)
    projected = l2_normalize(projected).astype(np.float32, copy=False)
    np.save(phase_dir / f"{modality}_feat.npy", projected)
    completed_path = phase_dir / f"completed_{modality}_feat.npy"
    np.save(completed_path, projected)

    mask_path = src_root / phase / f"{modality}_observed_mask.npy"
    if mask_path.exists():
        shutil.copy2(mask_path, phase_dir / mask_path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--n_components", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "input_dir": str(args.input_dir),
        "n_components": args.n_components,
        "seed": args.seed,
        "modalities": {},
    }

    for modality in MODALITIES:
        train_features = load_phase(args.input_dir, "phase_train", modality)
        train_mask = load_mask(args.input_dir, "phase_train", modality)
        pca = fit_pca(train_features, train_mask, args.n_components, args.seed)
        metadata["modalities"][modality] = {
            "input_dim": int(train_features.shape[1]),
            "observed_fit_rows": int(train_mask.sum()),
            "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        }
        for phase in PHASES:
            transform_and_save(pca, args.input_dir, args.output_dir, phase, modality)

    with open(args.output_dir / "pca64_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
