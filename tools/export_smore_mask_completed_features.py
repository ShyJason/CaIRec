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
    parser.description = "Export MMRec completed raw features using externally supplied SMORE masks"
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--smore_data_dir", type=str, required=True)
    parser.add_argument("--smore_missing_seed", type=int, default=2023)
    parser.add_argument("--smore_train_missing_rate", type=float, default=None)
    parser.add_argument("--smore_eval_missing_rate", type=float, default=0.5)

    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        parser.set_defaults(**_load_config_file(pre_args.config))
    args = parser.parse_args()
    if args.config is not None:
        args.config = str(Path(args.config).expanduser().resolve())
    return args


def _rate_token(rate):
    return f"{float(rate):g}"


def _load_raw_features(data_dir, device):
    image = np.load(data_dir / "image_feat.npy").astype(np.float32)
    text = np.load(data_dir / "text_feat.npy").astype(np.float32)
    return {
        "v": torch.from_numpy(image).to(device),
        "t": torch.from_numpy(text).to(device),
    }


def _load_observed_masks(data_dir, phase, train_rate, eval_rate, seed, device):
    if phase == "train":
        rate = train_rate
        split = "train"
    elif phase == "eval":
        rate = eval_rate
        split = "test"
    else:
        raise ValueError(f"Unsupported phase: {phase}")

    masks = {}
    for modality, prefix in (("v", "image"), ("t", "text")):
        path = data_dir / f"{prefix}_feat_missing_{split}_mask_mr{_rate_token(rate)}_seed{seed}.npy"
        missing = np.load(path).astype(bool)
        masks[modality] = torch.from_numpy(~missing).to(device)
    return masks


def _to_numpy(tensor):
    return tensor.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def _complete_with_masks(model, raw, observed_masks):
    projected = model.project_features(raw_features=raw)
    completed_shared = model._build_completed_features(projected, observed_masks)
    decoded = model.decode_completed_to_raw(completed_shared)
    completed_raw = {}
    for modality in ("v", "t"):
        mask = observed_masks[modality].unsqueeze(1)
        completed_raw[modality] = torch.where(mask, raw[modality], decoded[modality])
    return completed_raw


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.smore_data_dir).expanduser().resolve()
    train_rate = float(args.smore_train_missing_rate if args.smore_train_missing_rate is not None else args.missing_rate)

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

    raw = _load_raw_features(data_dir, env.device)
    manifest = {
        "dataset": args.dataset,
        "missing_rate": float(args.missing_rate),
        "dataset_seed": int(args.dataset_seed),
        "seed": int(args.seed),
        "smore_missing_seed": int(args.smore_missing_seed),
        "smore_train_missing_rate": train_rate,
        "smore_eval_missing_rate": float(args.smore_eval_missing_rate),
        "ckpt": args.ckpt,
        "imputer_ckpt": args.imputer_ckpt,
        "exporter": "MMRec.tools.export_smore_mask_completed_features",
        "mask_source": str(data_dir),
        "splits": {},
    }

    for phase in ("train", "eval"):
        masks = _load_observed_masks(
            data_dir,
            phase,
            train_rate,
            float(args.smore_eval_missing_rate),
            int(args.smore_missing_seed),
            env.device,
        )
        completed = _complete_with_masks(model, raw, masks)

        files = {}
        for modality, prefix in (("v", "image"), ("t", "text")):
            path = output_dir / f"{prefix}_feat_mmrec_completed_{phase}.npy"
            np.save(path, _to_numpy(completed[modality]))
            files[modality] = str(path)

        split_name = "train" if phase == "train" else "test"
        manifest["splits"][split_name] = {
            "phase": phase,
            "files": files,
            "shape": {modality: list(completed[modality].shape) for modality in ("v", "t")},
            "observed_count": {modality: int(masks[modality].sum().item()) for modality in ("v", "t")},
            "missing_count": {modality: int((~masks[modality]).sum().item()) for modality in ("v", "t")},
        }
        tool.cprint(
            f"Exported {phase} completed features to {output_dir} "
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
