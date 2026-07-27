#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionCompletionPretrainer(nn.Module):
    def __init__(self, image_dim, text_dim, latent_dim):
        super().__init__()
        self.v = nn.Linear(image_dim, latent_dim)
        self.t = nn.Linear(text_dim, latent_dim)
        self.v_to_t = nn.Linear(latent_dim, latent_dim)
        self.t_to_v = nn.Linear(latent_dim, latent_dim)

    def forward(self, image, text):
        z_v_raw = self.v(image)
        z_t_raw = self.t(text)
        z_v = F.normalize(z_v_raw, dim=-1)
        z_t = F.normalize(z_t_raw, dim=-1)
        pred_t_from_v = F.normalize(self.v_to_t(z_v_raw), dim=-1)
        pred_v_from_t = F.normalize(self.t_to_v(z_t_raw), dim=-1)
        return z_v, z_t, pred_t_from_v, pred_v_from_t

    def projection_state_dict(self, key_style):
        state = {}
        if key_style in {"raw_decoder", "both"}:
            state.update(
                {
                    "contra_head_v.linear.weight": self.v.weight.detach().cpu(),
                    "contra_head_v.linear.bias": self.v.bias.detach().cpu(),
                    "contra_head_t.linear.weight": self.t.weight.detach().cpu(),
                    "contra_head_t.linear.bias": self.t.bias.detach().cpu(),
                }
            )
        if key_style in {"decoupled_latent", "both"}:
            state.update(
                {
                    "comp_proj_v.weight": self.v.weight.detach().cpu(),
                    "comp_proj_v.bias": self.v.bias.detach().cpu(),
                    "comp_proj_t.weight": self.t.weight.detach().cpu(),
                    "comp_proj_t.bias": self.t.bias.detach().cpu(),
                }
            )
        return state


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone completion-aware projection pretraining for MMRec Stage1."
    )
    parser.add_argument("--dataset", default="clothing")
    parser.add_argument("--data_root", default="Data")
    parser.add_argument("--suffix", default="stage0_completion_projection")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--image_file", default="image_feat.npy")
    parser.add_argument("--text_file", default="text_feat.npy")
    parser.add_argument("--train_file", default="train.txt")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--base_ce_weight", type=float, default=1.0)
    parser.add_argument("--completion_ce_weight", type=float, default=1.0)
    parser.add_argument("--mse_weight", type=float, default=0.05)
    parser.add_argument("--cosine_weight", type=float, default=0.05)
    parser.add_argument("--val_rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument(
        "--checkpoint_key_style",
        choices=["raw_decoder", "decoupled_latent", "both"],
        default="raw_decoder",
        help="raw_decoder writes contra_head_* keys that MMRec/raw_decoder Stage1 loads directly.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_train_items(path):
    items = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) < 2:
                continue
            items.update(int(item) for item in fields[1:])
    return np.array(sorted(items), dtype=np.int64)


def filter_nonzero_pair_items(item_ids, image_feat, text_feat):
    kept = []
    for start in range(0, len(item_ids), 8192):
        ids = item_ids[start : start + 8192]
        image = np.asarray(image_feat[ids], dtype=np.float32)
        text = np.asarray(text_feat[ids], dtype=np.float32)
        mask = (np.abs(image).sum(axis=1) > 0) & (np.abs(text).sum(axis=1) > 0)
        kept.append(ids[mask])
    if not kept:
        return np.array([], dtype=np.int64)
    return np.concatenate(kept).astype(np.int64, copy=False)


def make_batches(item_ids, batch_size, shuffle):
    if shuffle:
        order = np.random.permutation(len(item_ids))
        item_ids = item_ids[order]
    for start in range(0, len(item_ids), batch_size):
        batch = item_ids[start : start + batch_size]
        if len(batch) > 1:
            yield batch


def batch_to_tensor(feature, ids, device):
    array = np.asarray(feature[ids], dtype=np.float32)
    return torch.from_numpy(array).to(device=device, non_blocking=True)


def compute_loss(model, image, text, temperature, weights):
    z_v, z_t, pred_t_from_v, pred_v_from_t = model(image, text)
    labels = torch.arange(image.size(0), device=image.device)

    base_logits_vt = z_v @ z_t.t() / temperature
    base_logits_tv = z_t @ z_v.t() / temperature
    base_ce = 0.5 * (
        F.cross_entropy(base_logits_vt, labels)
        + F.cross_entropy(base_logits_tv, labels)
    )

    target_t = z_t.detach()
    target_v = z_v.detach()
    completion_logits_vt = pred_t_from_v @ target_t.t() / temperature
    completion_logits_tv = pred_v_from_t @ target_v.t() / temperature
    completion_ce = 0.5 * (
        F.cross_entropy(completion_logits_vt, labels)
        + F.cross_entropy(completion_logits_tv, labels)
    )

    mse = 0.5 * (
        F.mse_loss(pred_t_from_v, target_t)
        + F.mse_loss(pred_v_from_t, target_v)
    )
    cosine = 0.5 * (
        (1.0 - F.cosine_similarity(pred_t_from_v, target_t, dim=-1)).mean()
        + (1.0 - F.cosine_similarity(pred_v_from_t, target_v, dim=-1)).mean()
    )

    loss = (
        weights["base_ce"] * base_ce
        + weights["completion_ce"] * completion_ce
        + weights["mse"] * mse
        + weights["cosine"] * cosine
    )
    metrics = {
        "loss": float(loss.detach().cpu()),
        "base_ce": float(base_ce.detach().cpu()),
        "completion_ce": float(completion_ce.detach().cpu()),
        "mse": float(mse.detach().cpu()),
        "cosine": float(cosine.detach().cpu()),
    }
    return loss, metrics


@torch.no_grad()
def evaluate(model, image_feat, text_feat, item_ids, batch_size, device, temperature, weights):
    model.eval()
    totals = {key: 0.0 for key in ("loss", "base_ce", "completion_ce", "mse", "cosine")}
    count = 0
    for ids in make_batches(item_ids, batch_size, shuffle=False):
        image = batch_to_tensor(image_feat, ids, device)
        text = batch_to_tensor(text_feat, ids, device)
        _, metrics = compute_loss(model, image, text, temperature, weights)
        batch_count = len(ids)
        for key, value in metrics.items():
            totals[key] += value * batch_count
        count += batch_count
    return {key: value / max(count, 1) for key, value in totals.items()}


def save_projection_checkpoint(path, model, args, metrics, epoch):
    payload = {
        "model_state_dict": model.projection_state_dict(args.checkpoint_key_style),
        "epoch": epoch,
        "metrics": metrics,
        "meta": {
            "kind": "standalone_completion_projection_pretrain",
            "dataset": args.dataset,
            "latent_dim": args.latent_dim,
            "image_file": args.image_file,
            "text_file": args.text_file,
            "train_file": args.train_file,
            "checkpoint_key_style": args.checkpoint_key_style,
        },
    }
    torch.save(payload, path)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = (
        torch.device(f"cuda:{args.device_id}")
        if args.use_gpu and torch.cuda.is_available()
        else torch.device("cpu")
    )

    data_dir = Path(args.data_root) / args.dataset
    image_path = data_dir / args.image_file
    text_path = data_dir / args.text_file
    train_path = data_dir / args.train_file
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if not text_path.exists():
        raise FileNotFoundError(text_path)
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    output_dir = Path(args.output_dir) if args.output_dir else Path("exp_report") / args.dataset / args.suffix
    ckpt_dir = output_dir / "ckpt"
    log_dir = output_dir / "log"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    image_feat = np.load(image_path, mmap_mode="r")
    text_feat = np.load(text_path, mmap_mode="r")
    if image_feat.shape[0] != text_feat.shape[0]:
        raise ValueError(f"feature row mismatch: image={image_feat.shape}, text={text_feat.shape}")

    train_items = read_train_items(train_path)
    train_items = train_items[(train_items >= 0) & (train_items < image_feat.shape[0])]
    train_items = filter_nonzero_pair_items(train_items, image_feat, text_feat)
    if len(train_items) < 2:
        raise ValueError("not enough train items with both image and text features")

    rng = np.random.default_rng(args.seed)
    shuffled = train_items.copy()
    rng.shuffle(shuffled)
    val_size = int(len(shuffled) * args.val_rate)
    if args.val_rate > 0 and val_size < 2 and len(shuffled) >= 4:
        val_size = 2
    val_items = shuffled[:val_size]
    fit_items = shuffled[val_size:] if val_size else shuffled
    if len(fit_items) < 2:
        raise ValueError("not enough fit items after validation split")

    model = ProjectionCompletionPretrainer(
        image_dim=int(image_feat.shape[1]),
        text_dim=int(text_feat.shape[1]),
        latent_dim=args.latent_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = {
        "base_ce": args.base_ce_weight,
        "completion_ce": args.completion_ce_weight,
        "mse": args.mse_weight,
        "cosine": args.cosine_weight,
    }

    print(
        f"[projection-pretrain] dataset={args.dataset} device={device} "
        f"items={len(train_items)} fit={len(fit_items)} val={len(val_items)} "
        f"image_dim={image_feat.shape[1]} text_dim={text_feat.shape[1]} "
        f"latent_dim={args.latent_dim} key_style={args.checkpoint_key_style}"
    )

    best_val = float("inf")
    best_epoch = -1
    history = []
    for epoch in range(args.epoch):
        model.train()
        totals = {key: 0.0 for key in ("loss", "base_ce", "completion_ce", "mse", "cosine")}
        count = 0
        for ids in make_batches(fit_items, args.batch_size, shuffle=True):
            image = batch_to_tensor(image_feat, ids, device)
            text = batch_to_tensor(text_feat, ids, device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = compute_loss(model, image, text, args.temperature, weights)
            loss.backward()
            optimizer.step()
            batch_count = len(ids)
            for key, value in metrics.items():
                totals[key] += value * batch_count
            count += batch_count

        train_metrics = {key: value / max(count, 1) for key, value in totals.items()}
        val_metrics = (
            evaluate(model, image_feat, text_feat, val_items, args.batch_size, device, args.temperature, weights)
            if len(val_items) >= 2
            else train_metrics
        )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        print(
            "[projection-pretrain] "
            f"epoch={epoch}/{args.epoch} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_base_ce={train_metrics['base_ce']:.6f} "
            f"train_completion_ce={train_metrics['completion_ce']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_base_ce={val_metrics['base_ce']:.6f} "
            f"val_completion_ce={val_metrics['completion_ce']:.6f} "
            f"val_mse={val_metrics['mse']:.6f} "
            f"val_cosine_loss={val_metrics['cosine']:.6f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            best_path = ckpt_dir / f"{args.suffix}_projection_only_best_epoch{epoch}.pth"
            save_projection_checkpoint(best_path, model, args, val_metrics, epoch)
            save_projection_checkpoint(ckpt_dir / f"{args.suffix}_projection_only_best.pth", model, args, val_metrics, epoch)
            print(f"[projection-pretrain] save best checkpoint={best_path}")

    final_metrics = history[-1]["val"]
    final_path = ckpt_dir / f"{args.suffix}_projection_only_final_epoch{args.epoch - 1}.pth"
    save_projection_checkpoint(final_path, model, args, final_metrics, args.epoch - 1)
    stable_final_path = ckpt_dir / f"{args.suffix}_projection_only_final.pth"
    save_projection_checkpoint(stable_final_path, model, args, final_metrics, args.epoch - 1)
    with (log_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "history": history,
                "final_checkpoint": str(final_path),
                "stable_final_checkpoint": str(stable_final_path),
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"[projection-pretrain] done best_epoch={best_epoch} best_val_loss={best_val:.6f}")
    print(f"[projection-pretrain] final_checkpoint={final_path}")
    print(f"[projection-pretrain] stable_final_checkpoint={stable_final_path}")
    print(f"[projection-pretrain] best_checkpoint={ckpt_dir / f'{args.suffix}_projection_only_best.pth'}")


if __name__ == "__main__":
    main()
