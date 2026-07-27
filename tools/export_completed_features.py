import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

import tool
from enviroment import Env
from dataset_loader import Loader4MM
from model import MILK_model
from tools.evaluate_imputation_metrics import _build_parser, _load_config_file


def parse_args():
    parser = _build_parser()
    parser.description = "Export MMRec decoder-space completed raw features"
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--export_splits",
        type=str,
        default="train,test",
        help="Comma-separated splits to export: train,valid,val,test.",
    )
    parser.add_argument(
        "--external_missing_dir",
        type=str,
        default="",
        help="Optional directory containing VBPR-style *_missing_{split}_mask_mr*_seed*.npy files.",
    )
    parser.add_argument(
        "--external_missing_seed",
        type=int,
        default=None,
        help="Seed suffix for external missing masks. Defaults to --dataset_seed.",
    )
    parser.add_argument(
        "--vbpr_raw_feature_dir",
        type=str,
        default="",
        help="Optional VBPR raw feature directory. When set, observed rows are exported in VBPR raw space and missing rows are calibrated from train observed rows.",
    )
    parser.add_argument(
        "--export_space",
        type=str,
        default="raw",
        choices=["raw", "shared_completed"],
        help="raw exports decoder/raw modal features; shared_completed exports 64-d completed MMRec latent features.",
    )

    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        parser.set_defaults(**_load_config_file(pre_args.config))
    args = parser.parse_args()
    if args.config is not None:
        args.config = str(Path(args.config).expanduser().resolve())
    return args


def _phase_name(split_name):
    if split_name == "train":
        return "train"
    if split_name in {"val", "valid"}:
        return "valid"
    if split_name == "test":
        return "test"
    raise ValueError(f"Unsupported split: {split_name}")


def _model_split_name(split_name):
    return "val" if split_name == "valid" else split_name


def _rate_tag(rate):
    return f"{float(rate):g}"


def _to_numpy(tensor):
    return tensor.detach().cpu().numpy().astype(np.float32)


def _load_external_missing_mask(mask_dir, modality, split_name, rate, seed, device, n_items):
    prefix = "image_feat" if modality == "v" else "text_feat"
    path = (
        Path(mask_dir)
        / f"{prefix}_missing_{split_name}_mask_mr{_rate_tag(rate)}_seed{seed}.npy"
    )
    if not path.exists():
        raise FileNotFoundError(f"External missing mask not found: {path}")
    mask = torch.from_numpy(np.load(path).astype(bool)).to(device)
    if mask.ndim != 1 or mask.numel() != n_items:
        raise ValueError(f"Mask shape mismatch for {path}: expected ({n_items},), got {tuple(mask.shape)}")
    return mask


def _raw_features_from_external_masks(model, split_name, rate, mask_dir, mask_seed):
    full = model.get_split_raw_modal_features(split=_model_split_name(split_name), full=True)
    raw = {}
    missing_masks = {}
    for modality, feature in full.items():
        missing = _load_external_missing_mask(
            mask_dir,
            modality,
            split_name,
            rate,
            mask_seed,
            feature.device,
            feature.size(0),
        )
        masked = feature.clone()
        masked[missing] = 0
        raw[modality] = masked
        missing_masks[modality] = missing
    return raw, missing_masks


def _load_vbpr_raw_feature(raw_dir, modality, device, n_items):
    filename = "image_feat.npy" if modality == "v" else "text_feat.npy"
    path = Path(raw_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"VBPR raw feature not found: {path}")
    feature = torch.from_numpy(np.load(path).astype(np.float32, copy=False)).to(device)
    if feature.ndim != 2 or feature.size(0) != n_items:
        raise ValueError(f"Feature shape mismatch for {path}: expected first dim {n_items}, got {tuple(feature.shape)}")
    return feature


def _fit_raw_space_calibrators(model, args, mask_dir, mask_seed, raw_dir):
    raw, missing_masks = _raw_features_from_external_masks(
        model,
        "train",
        float(args.missing_rate),
        mask_dir,
        mask_seed,
    )
    completed = model.get_recommender_modal_features(raw_features=raw)
    raw_space = {}
    calibrators = {}
    eps = 1e-8
    for modality, feature in completed.items():
        raw_feature = _load_vbpr_raw_feature(raw_dir, modality, feature.device, feature.size(0))
        observed = ~missing_masks[modality]
        if not observed.any():
            raise ValueError(f"Cannot fit raw-space calibrator for {modality}: no observed train rows")
        source = feature[observed]
        target = raw_feature[observed]
        scale = (source * target).sum(dim=0) / source.pow(2).sum(dim=0).clamp_min(eps)
        raw_space[modality] = raw_feature
        calibrators[modality] = scale
    return raw_space, calibrators


def _calibrated_export_feature(completed, raw_space, calibrator, missing_mask):
    observed = ~missing_mask
    out = torch.empty_like(raw_space)
    out[observed] = raw_space[observed]
    out[missing_mask] = completed[missing_mask] * calibrator
    return out


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = Env(args)
    loader = Loader4MM(env)
    model = MILK_model(env, loader)
    if args.ckpt:
        loaded = model.load_full_checkpoint(args.ckpt)
        tool.cprint(f"Loaded full checkpoint from {args.ckpt} ({loaded} tensors)")
    if args.imputer_ckpt:
        loaded = model.load_imputer_checkpoint(args.imputer_ckpt)
        tool.cprint(f"Loaded imputer checkpoint from {args.imputer_ckpt} ({len(loaded)} tensors)")

    model.eval()
    manifest = {
        "dataset": args.dataset,
        "feature_kind": (
            "mmrec_completed_shared_features"
            if args.export_space == "shared_completed"
            else "mmrec_raw_decoder_completed_raw_features"
        ),
        "semantics": (
            "Observed modality rows keep MMRec projected features; missing modality rows are filled by MMRec completed shared-space outputs."
            if args.export_space == "shared_completed"
            else "Observed modality rows keep raw features; missing modality rows are filled by MMRec Stage1 raw_decoder outputs."
        ),
        "export_space": args.export_space,
        "missing_rate": float(args.missing_rate),
        "eval_missing_rate": float(getattr(args, "eval_missing_rate", args.missing_rate)),
        "dataset_seed": int(args.dataset_seed),
        "seed": int(args.seed),
        "ckpt": args.ckpt,
        "imputer_ckpt": args.imputer_ckpt,
        "splits": {},
    }

    external_missing_dir = str(Path(args.external_missing_dir).expanduser().resolve()) if args.external_missing_dir else ""
    external_missing_seed = int(args.dataset_seed if args.external_missing_seed is None else args.external_missing_seed)
    vbpr_raw_feature_dir = str(Path(args.vbpr_raw_feature_dir).expanduser().resolve()) if args.vbpr_raw_feature_dir else ""
    raw_space = {}
    raw_space_calibrators = {}
    if vbpr_raw_feature_dir:
        if args.export_space != "raw":
            raise ValueError("--vbpr_raw_feature_dir is only supported with --export_space raw")
        if not external_missing_dir:
            raise ValueError("--vbpr_raw_feature_dir requires --external_missing_dir")
        raw_space, raw_space_calibrators = _fit_raw_space_calibrators(
            model,
            args,
            external_missing_dir,
            external_missing_seed,
            vbpr_raw_feature_dir,
        )
        manifest["vbpr_raw_feature_dir"] = vbpr_raw_feature_dir
        manifest["raw_space_calibration"] = {
            "method": "diagonal_least_squares",
            "fit_split": "train",
            "fit_rows": "observed_only",
        }
    split_names = [split.strip() for split in args.export_splits.split(",") if split.strip()]
    with torch.no_grad():
        for split_name in split_names:
            rate = float(args.missing_rate if split_name == "train" else getattr(args, "eval_missing_rate", args.missing_rate))
            if external_missing_dir:
                raw, missing_masks = _raw_features_from_external_masks(
                    model,
                    split_name,
                    rate,
                    external_missing_dir,
                    external_missing_seed,
                )
                observed_masks = {
                    modality: ~mask
                    for modality, mask in missing_masks.items()
                }
            else:
                raw = model.get_split_raw_modal_features(split=_model_split_name(split_name), full=False)
                observed_masks = model._missing_masks(raw_features=raw)
            if args.export_space == "shared_completed":
                observed_masks = model._missing_masks(raw_features=raw)
                projected = model.project_features(raw_features=raw)
                completed = model._build_completed_features(
                    projected,
                    observed_masks,
                    detach_imputed=True,
                    item_ids=torch.arange(model.m_item, device=next(iter(raw.values())).device),
                    stage=args.train_stage,
                )
            else:
                completed = model.get_recommender_modal_features(raw_features=raw)
            phase = _phase_name(split_name)

            files = {}
            for modality, prefix in (("v", "image"), ("t", "text")):
                name_kind = "recspace" if args.export_space == "shared_completed" else "completed"
                path = output_dir / f"{prefix}_feat_mmrec_{name_kind}_{phase}.npy"
                export_feature = completed[modality]
                if vbpr_raw_feature_dir:
                    export_feature = _calibrated_export_feature(
                        completed[modality],
                        raw_space[modality],
                        raw_space_calibrators[modality],
                        missing_masks[modality],
                    )
                np.save(path, _to_numpy(export_feature))
                files[modality] = str(path)

            manifest["splits"][split_name] = {
                "phase": phase,
                "files": files,
                "shape": {
                    modality: list(completed[modality].shape)
                    for modality in ("v", "t")
                },
                "observed_count": {
                    modality: int(observed_masks[modality].sum().item())
                    for modality in ("v", "t")
                },
                "missing_count": {
                    modality: int((~observed_masks[modality]).sum().item())
                    for modality in ("v", "t")
                },
            }
            tool.cprint(
                f"Exported {split_name} completed features to {output_dir} "
                f"(v_missing={manifest['splits'][split_name]['missing_count']['v']}, "
                f"t_missing={manifest['splits'][split_name]['missing_count']['t']})"
            )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    tool.cprint(f"Saved manifest to {manifest_path}")
    env.close_env()


if __name__ == "__main__":
    main()
