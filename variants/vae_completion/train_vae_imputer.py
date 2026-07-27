import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train an isolated VAE modality imputer")
    parser.add_argument("--dataset", type=str, default="clothing")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--missing_seed", type=int, default=2023)
    parser.add_argument("--train_missing_rate", type=float, default=0.3)
    parser.add_argument("--eval_missing_rate", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--modal_hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--beta_kl", type=float, default=1e-3)
    parser.add_argument("--kl_warmup_epochs", type=int, default=20)
    parser.add_argument("--input_dropout", type=float, default=0.0)
    parser.add_argument("--normalize_features", type=int, default=1)
    parser.add_argument("--normalize_outputs", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--early_stop", type=int, default=20)
    parser.add_argument("--train_items", type=str, default="train_txt", choices=["train_txt", "all"])
    parser.add_argument("--cf_align_weight", type=float, default=0.0)
    parser.add_argument("--cf_align_dim", type=int, default=64)
    parser.add_argument("--cf_align_temp", type=float, default=0.2)
    parser.add_argument("--cf_max_neighbors", type=int, default=64)
    parser.add_argument("--cf_user_sample_size", type=int, default=8)
    parser.add_argument("--latent_consistency_weight", type=float, default=0.0)
    parser.add_argument("--latent_consistency_detach_anchor", type=int, default=1)
    parser.add_argument("--latent_consistency_start_epoch", type=int, default=1)
    parser.add_argument("--single_view_rec_weight", type=float, default=0.0)
    parser.add_argument("--single_view_rec_start_epoch", type=int, default=1)
    parser.add_argument("--fusion_consistency_weight", type=float, default=0.0)
    parser.add_argument("--fusion_consistency_start_epoch", type=int, default=1)
    parser.add_argument("--fusion_consistency_temp", type=float, default=0.2)
    parser.add_argument("--fusion_consistency_detach_target", type=int, default=1)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rate_token(rate):
    return f"{float(rate):g}"


def l2_normalize_np(array):
    denom = np.linalg.norm(array, axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return (array / denom).astype(np.float32, copy=False)


def load_feature(path, normalize):
    array = np.load(path).astype(np.float32, copy=False)
    if normalize:
        array = l2_normalize_np(array)
    return array


def load_split_arrays(data_dir, split, rate, missing_seed, normalize):
    image = load_feature(data_dir / f"image_feat_missing_{split}_mr{rate_token(rate)}_seed{missing_seed}.npy", normalize)
    text = load_feature(data_dir / f"text_feat_missing_{split}_mr{rate_token(rate)}_seed{missing_seed}.npy", normalize)
    image_missing = np.load(data_dir / f"image_feat_missing_{split}_mask_mr{rate_token(rate)}_seed{missing_seed}.npy").astype(bool)
    text_missing = np.load(data_dir / f"text_feat_missing_{split}_mask_mr{rate_token(rate)}_seed{missing_seed}.npy").astype(bool)
    return {
        "v": image,
        "t": text,
        "observed_v": ~image_missing,
        "observed_t": ~text_missing,
    }


def build_combined_graph_split(full_image, full_text, *splits):
    observed_v = np.ones(full_image.shape[0], dtype=bool)
    observed_t = np.ones(full_text.shape[0], dtype=bool)
    for split in splits:
        observed_v &= split["observed_v"]
        observed_t &= split["observed_t"]

    image = full_image.copy()
    text = full_text.copy()
    image[~observed_v] = 0.0
    text[~observed_t] = 0.0
    return {
        "v": image,
        "t": text,
        "observed_v": observed_v,
        "observed_t": observed_t,
    }


def load_train_item_ids(data_dir, n_items, mode):
    if mode == "all":
        return np.arange(n_items, dtype=np.int64)
    train_file = data_dir / "train.txt"
    if not train_file.exists():
        raise FileNotFoundError(f"train.txt not found: {train_file}")
    items = set()
    with train_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            for token in parts[1:]:
                item = int(token)
                if 0 <= item < n_items:
                    items.add(item)
    if not items:
        raise RuntimeError(f"No train items parsed from {train_file}")
    return np.array(sorted(items), dtype=np.int64)


def build_cf_neighbor_lists(data_dir, n_items, max_neighbors, user_sample_size, seed):
    train_file = data_dir / "train.txt"
    if not train_file.exists():
        raise FileNotFoundError(f"train.txt not found: {train_file}")

    rng = random.Random(seed)
    neighbors = [[] for _ in range(n_items)]
    with train_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) <= 2:
                continue
            items = []
            seen = set()
            for token in parts[1:]:
                item = int(token)
                if 0 <= item < n_items and item not in seen:
                    seen.add(item)
                    items.append(item)
            if len(items) <= 1:
                continue
            for item in items:
                candidates = [other for other in items if other != item]
                if len(candidates) > user_sample_size:
                    candidates = rng.sample(candidates, user_sample_size)
                bucket = neighbors[item]
                if len(bucket) < max_neighbors * 4:
                    bucket.extend(candidates)

    compact = []
    for bucket in neighbors:
        if not bucket:
            compact.append(np.empty(0, dtype=np.int64))
            continue
        unique = list(dict.fromkeys(bucket))
        if len(unique) > max_neighbors:
            unique = rng.sample(unique, max_neighbors)
        compact.append(np.asarray(unique, dtype=np.int64))
    return compact


class VAEImputer(nn.Module):
    def __init__(
        self,
        image_dim,
        text_dim,
        modal_hidden_dim,
        hidden_dim,
        latent_dim,
        dropout,
        normalize_outputs,
        cf_align_dim=64,
    ):
        super().__init__()
        self.normalize_outputs = bool(normalize_outputs)
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, modal_hidden_dim),
            nn.LayerNorm(modal_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, modal_hidden_dim),
            nn.LayerNorm(modal_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.encoder = nn.Sequential(
            nn.Linear(modal_hidden_dim * 2 + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)
        self.image_decoder = self._build_decoder(latent_dim, hidden_dim, image_dim, dropout)
        self.text_decoder = self._build_decoder(latent_dim, hidden_dim, text_dim, dropout)
        self.image_cf_head = self._build_cf_head(image_dim, hidden_dim, cf_align_dim, dropout)
        self.text_cf_head = self._build_cf_head(text_dim, hidden_dim, cf_align_dim, dropout)

    @staticmethod
    def _build_decoder(latent_dim, hidden_dim, output_dim, dropout):
        return nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    @staticmethod
    def _build_cf_head(input_dim, hidden_dim, output_dim, dropout):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode(self, image, text, observed):
        image_in = image * observed[:, 0:1]
        text_in = text * observed[:, 1:2]
        image_h = self.image_encoder(image_in)
        text_h = self.text_encoder(text_in)
        h = self.encoder(torch.cat([image_h, text_h, observed], dim=1))
        logvar = self.to_logvar(h).clamp(min=-8.0, max=6.0)
        return self.to_mu(h), logvar

    @staticmethod
    def reparameterize(mu, logvar):
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        image = self.image_decoder(z)
        text = self.text_decoder(z)
        if self.normalize_outputs:
            image = F.normalize(image, dim=-1)
            text = F.normalize(text, dim=-1)
        return image, text

    def complete(self, image, text, observed, image_rec, text_rec):
        image_out = torch.where(observed[:, 0:1] > 0.5, image, image_rec)
        text_out = torch.where(observed[:, 1:2] > 0.5, text, text_rec)
        return image_out, text_out

    def cf_embedding(self, image, text):
        image_emb = F.normalize(self.image_cf_head(image), dim=-1)
        text_emb = F.normalize(self.text_cf_head(text), dim=-1)
        return F.normalize((image_emb + text_emb) * 0.5, dim=-1)

    def forward(self, image, text, observed):
        mu, logvar = self.encode(image, text, observed)
        z = self.reparameterize(mu, logvar)
        image_rec, text_rec = self.decode(z)
        return image_rec, text_rec, mu, logvar


def maybe_apply_input_dropout(observed, dropout):
    if dropout <= 0:
        return observed
    dropped = observed.clone()
    both_observed = (observed[:, 0] > 0.5) & (observed[:, 1] > 0.5)
    selected = both_observed & (torch.rand(observed.size(0), device=observed.device) < dropout)
    if selected.any():
        choose_text = torch.rand(int(selected.sum().item()), device=observed.device) < 0.5
        selected_idx = torch.nonzero(selected, as_tuple=False).squeeze(1)
        dropped[selected_idx, 0] = choose_text.float()
        dropped[selected_idx, 1] = (~choose_text).float()
    return dropped


def observed_reconstruction_loss(image_rec, text_rec, image_target, text_target, observed):
    losses = []
    image_mask = observed[:, 0] > 0.5
    text_mask = observed[:, 1] > 0.5
    if image_mask.any():
        losses.append(F.mse_loss(image_rec[image_mask], image_target[image_mask], reduction="mean"))
    if text_mask.any():
        losses.append(F.mse_loss(text_rec[text_mask], text_target[text_mask], reduction="mean"))
    if not losses:
        return image_rec.sum() * 0.0
    return torch.stack(losses).mean()


def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def latent_consistency_loss(model, image, text, observed, detach_anchor):
    both_observed = (observed[:, 0] > 0.5) & (observed[:, 1] > 0.5)
    if not both_observed.any():
        return image.sum() * 0.0

    image_b = image[both_observed]
    text_b = text[both_observed]
    full_observed = torch.ones((image_b.size(0), 2), device=image.device, dtype=observed.dtype)
    image_only = full_observed.clone()
    image_only[:, 1] = 0.0
    text_only = full_observed.clone()
    text_only[:, 0] = 0.0

    full_mu, _ = model.encode(image_b, text_b, full_observed)
    if detach_anchor:
        full_mu = full_mu.detach()
    image_mu, _ = model.encode(image_b, text_b, image_only)
    text_mu, _ = model.encode(image_b, text_b, text_only)
    return 0.5 * (F.mse_loss(image_mu, full_mu, reduction="mean") + F.mse_loss(text_mu, full_mu, reduction="mean"))


def single_view_reconstruction_loss(model, image, text, observed):
    both_observed = (observed[:, 0] > 0.5) & (observed[:, 1] > 0.5)
    if not both_observed.any():
        return image.sum() * 0.0

    image_b = image[both_observed]
    text_b = text[both_observed]
    full_observed = torch.ones((image_b.size(0), 2), device=image.device, dtype=observed.dtype)
    image_only = full_observed.clone()
    image_only[:, 1] = 0.0
    text_only = full_observed.clone()
    text_only[:, 0] = 0.0

    image_from_image, text_from_image, _, _ = model(image_b, text_b, image_only)
    image_from_text, text_from_text, _, _ = model(image_b, text_b, text_only)
    image_loss = 0.5 * (
        F.mse_loss(image_from_image, image_b, reduction="mean")
        + F.mse_loss(image_from_text, image_b, reduction="mean")
    )
    text_loss = 0.5 * (
        F.mse_loss(text_from_image, text_b, reduction="mean")
        + F.mse_loss(text_from_text, text_b, reduction="mean")
    )
    return 0.5 * (image_loss + text_loss)


def completed_fusion_consistency_loss(model, image, text, observed, temp, detach_target):
    both_observed = (observed[:, 0] > 0.5) & (observed[:, 1] > 0.5)
    if not both_observed.any():
        return image.sum() * 0.0

    image_b = image[both_observed]
    text_b = text[both_observed]
    if image_b.size(0) <= 1:
        return image.sum() * 0.0

    full_observed = torch.ones((image_b.size(0), 2), device=image.device, dtype=observed.dtype)
    image_only = full_observed.clone()
    image_only[:, 1] = 0.0
    text_only = full_observed.clone()
    text_only[:, 0] = 0.0

    image_mu, _ = model.encode(image_b, text_b, image_only)
    text_mu, _ = model.encode(image_b, text_b, text_only)
    _, text_from_image = model.decode(image_mu)
    image_from_text, _ = model.decode(text_mu)

    full_fused = model.cf_embedding(image_b, text_b)
    if detach_target:
        full_fused = full_fused.detach()
    completed_from_image = model.cf_embedding(image_b, text_from_image)
    completed_from_text = model.cf_embedding(image_from_text, text_b)

    labels = torch.arange(full_fused.size(0), device=image.device)
    scale = max(float(temp), 1e-6)
    logits_i = completed_from_image @ full_fused.t() / scale
    logits_t = completed_from_text @ full_fused.t() / scale
    contrastive = 0.5 * (F.cross_entropy(logits_i, labels) + F.cross_entropy(logits_t, labels))

    cosine = 0.5 * (
        (1.0 - F.cosine_similarity(completed_from_image, full_fused, dim=-1)).mean()
        + (1.0 - F.cosine_similarity(completed_from_text, full_fused, dim=-1)).mean()
    )
    return contrastive + cosine


def sample_cf_neighbors(item_ids, neighbor_lists, rng):
    item_ids_np = item_ids.detach().cpu().numpy()
    neighbor_ids = np.empty_like(item_ids_np)
    valid = np.zeros(item_ids_np.shape[0], dtype=bool)
    for idx, item in enumerate(item_ids_np):
        choices = neighbor_lists[int(item)]
        if choices.size == 0:
            neighbor_ids[idx] = item
            continue
        neighbor_ids[idx] = choices[rng.integers(0, choices.size)]
        valid[idx] = True
    return neighbor_ids, valid


def cf_alignment_loss(
    model,
    image,
    text,
    observed,
    image_rec,
    text_rec,
    item_ids,
    neighbor_lists,
    train_split,
    device,
    rng,
    temp,
):
    neighbor_ids, valid = sample_cf_neighbors(item_ids, neighbor_lists, rng)
    if not valid.any():
        return image_rec.sum() * 0.0

    valid_t = torch.from_numpy(valid).to(device)
    image_completed, text_completed = model.complete(image, text, observed, image_rec, text_rec)
    anchor = model.cf_embedding(image_completed[valid_t], text_completed[valid_t])

    neighbor_ids = neighbor_ids[valid]
    neighbor_image = torch.from_numpy(train_split["v"][neighbor_ids]).to(device)
    neighbor_text = torch.from_numpy(train_split["t"][neighbor_ids]).to(device)
    neighbor_observed_np = np.stack(
        [train_split["observed_v"][neighbor_ids], train_split["observed_t"][neighbor_ids]],
        axis=1,
    )
    neighbor_observed = torch.from_numpy(neighbor_observed_np.astype(np.float32, copy=False)).to(device)
    neighbor_mu, _ = model.encode(neighbor_image, neighbor_text, neighbor_observed)
    neighbor_image_rec, neighbor_text_rec = model.decode(neighbor_mu)
    neighbor_image_completed, neighbor_text_completed = model.complete(
        neighbor_image,
        neighbor_text,
        neighbor_observed,
        neighbor_image_rec,
        neighbor_text_rec,
    )
    positive = model.cf_embedding(neighbor_image_completed, neighbor_text_completed)

    logits = anchor @ positive.t() / max(temp, 1e-6)
    labels = torch.arange(logits.size(0), device=device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def make_dataset(split_arrays, full_image, full_text, item_ids):
    ids = torch.from_numpy(item_ids.astype(np.int64, copy=False))
    image_in = torch.from_numpy(split_arrays["v"][item_ids])
    text_in = torch.from_numpy(split_arrays["t"][item_ids])
    observed = np.stack([split_arrays["observed_v"][item_ids], split_arrays["observed_t"][item_ids]], axis=1)
    observed = torch.from_numpy(observed.astype(np.float32, copy=False))
    image_target = torch.from_numpy(full_image[item_ids])
    text_target = torch.from_numpy(full_text[item_ids])
    return TensorDataset(ids, image_in, text_in, observed, image_target, text_target)


def batch_to_device(batch, device):
    return tuple(x.to(device, non_blocking=True) for x in batch)


@torch.no_grad()
def evaluate_missing(model, split_arrays, full_image, full_text, device, batch_size):
    model.eval()
    n_items = full_image.shape[0]
    ids = torch.arange(n_items, dtype=torch.long)
    loader = DataLoader(TensorDataset(ids), batch_size=batch_size, shuffle=False)
    metrics = {
        "v": {"count": 0, "mse_sum": 0.0, "cosine_sum": 0.0},
        "t": {"count": 0, "mse_sum": 0.0, "cosine_sum": 0.0},
    }
    missing_v = ~split_arrays["observed_v"]
    missing_t = ~split_arrays["observed_t"]
    for (item_ids,) in loader:
        np_ids = item_ids.numpy()
        image = torch.from_numpy(split_arrays["v"][np_ids]).to(device)
        text = torch.from_numpy(split_arrays["t"][np_ids]).to(device)
        observed_np = np.stack([split_arrays["observed_v"][np_ids], split_arrays["observed_t"][np_ids]], axis=1)
        observed = torch.from_numpy(observed_np.astype(np.float32, copy=False)).to(device)
        mu, logvar = model.encode(image, text, observed)
        image_rec, text_rec = model.decode(mu)

        for modality, rec, target_full, missing_mask in (
            ("v", image_rec, full_image, missing_v),
            ("t", text_rec, full_text, missing_t),
        ):
            local_missing = torch.from_numpy(missing_mask[np_ids]).to(device)
            if not local_missing.any():
                continue
            target = torch.from_numpy(target_full[np_ids]).to(device)[local_missing]
            pred = rec[local_missing]
            mse = (pred - target).pow(2).mean(dim=1)
            cosine = F.cosine_similarity(pred, target, dim=1)
            count = int(local_missing.sum().item())
            metrics[modality]["count"] += count
            metrics[modality]["mse_sum"] += float(mse.sum().cpu())
            metrics[modality]["cosine_sum"] += float(cosine.sum().cpu())

    total_count = 0
    total_mse = 0.0
    total_cosine = 0.0
    result = {}
    for modality, values in metrics.items():
        count = values["count"]
        mse = values["mse_sum"] / count if count else math.nan
        cosine = values["cosine_sum"] / count if count else math.nan
        result[modality] = {"count": count, "mse": mse, "cosine": cosine}
        total_count += count
        total_mse += values["mse_sum"]
        total_cosine += values["cosine_sum"]
    result["_overall"] = {
        "count": total_count,
        "mse": total_mse / total_count if total_count else math.nan,
        "cosine": total_cosine / total_count if total_count else math.nan,
    }
    return result


@torch.no_grad()
def export_completed(model, split_arrays, full_image, full_text, device, batch_size, output_dir):
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    n_items = full_image.shape[0]
    completed_image = np.empty_like(full_image, dtype=np.float32)
    completed_text = np.empty_like(full_text, dtype=np.float32)
    posterior_mu = None
    posterior_logvar = None
    ids = torch.arange(n_items, dtype=torch.long)
    loader = DataLoader(TensorDataset(ids), batch_size=batch_size, shuffle=False)
    for (item_ids,) in loader:
        np_ids = item_ids.numpy()
        image = torch.from_numpy(split_arrays["v"][np_ids]).to(device)
        text = torch.from_numpy(split_arrays["t"][np_ids]).to(device)
        observed_np = np.stack([split_arrays["observed_v"][np_ids], split_arrays["observed_t"][np_ids]], axis=1)
        observed = torch.from_numpy(observed_np.astype(np.float32, copy=False)).to(device)
        mu, logvar = model.encode(image, text, observed)
        image_rec, text_rec = model.decode(mu)

        image_out = image.clone()
        text_out = text.clone()
        image_missing = observed[:, 0] < 0.5
        text_missing = observed[:, 1] < 0.5
        image_out[image_missing] = image_rec[image_missing]
        text_out[text_missing] = text_rec[text_missing]

        completed_image[np_ids] = image_out.cpu().numpy().astype(np.float32)
        completed_text[np_ids] = text_out.cpu().numpy().astype(np.float32)
        if posterior_mu is None:
            posterior_mu = np.empty((n_items, mu.size(1)), dtype=np.float32)
            posterior_logvar = np.empty((n_items, logvar.size(1)), dtype=np.float32)
        posterior_mu[np_ids] = mu.cpu().numpy().astype(np.float32)
        posterior_logvar[np_ids] = logvar.cpu().numpy().astype(np.float32)

    for name, array in (
        ("completed_image_feat.npy", completed_image),
        ("completed_text_feat.npy", completed_text),
        ("image_feat.npy", completed_image),
        ("text_feat.npy", completed_text),
        ("image_observed_mask.npy", split_arrays["observed_v"].astype(bool)),
        ("text_observed_mask.npy", split_arrays["observed_t"].astype(bool)),
        ("posterior_mu.npy", posterior_mu),
        ("posterior_logvar.npy", posterior_logvar),
    ):
        np.save(output_dir / name, array)


def save_checkpoint(path, model, args, epoch, metrics):
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    normalize = bool(args.normalize_features)
    full_image = load_feature(data_dir / "image_feat.npy", normalize)
    full_text = load_feature(data_dir / "text_feat.npy", normalize)
    train_split = load_split_arrays(data_dir, "train", args.train_missing_rate, args.missing_seed, normalize)
    valid_split = load_split_arrays(data_dir, "valid", args.train_missing_rate, args.missing_seed, normalize)
    test_split = load_split_arrays(data_dir, "test", args.eval_missing_rate, args.missing_seed, normalize)
    graph_split = build_combined_graph_split(full_image, full_text, train_split, valid_split, test_split)

    train_ids = load_train_item_ids(data_dir, full_image.shape[0], args.train_items)
    train_dataset = make_dataset(train_split, full_image, full_text, train_ids)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = VAEImputer(
        image_dim=full_image.shape[1],
        text_dim=full_text.shape[1],
        modal_hidden_dim=args.modal_hidden_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        normalize_outputs=bool(args.normalize_outputs),
        cf_align_dim=args.cf_align_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cf_neighbor_lists = None
    cf_rng = np.random.default_rng(args.seed)
    if args.cf_align_weight > 0.0:
        cf_neighbor_lists = build_cf_neighbor_lists(
            data_dir,
            full_image.shape[0],
            args.cf_max_neighbors,
            args.cf_user_sample_size,
            args.seed,
        )
        cf_items = sum(1 for neighbors in cf_neighbor_lists if neighbors.size > 0)
        print(
            f"loaded CF alignment neighbors: items_with_neighbors={cf_items} "
            f"max_neighbors={args.cf_max_neighbors} user_sample_size={args.cf_user_sample_size}"
        )

    history = []
    best_score = -float("inf")
    best_epoch = -1
    no_improve = 0
    best_path = output_dir / "ckpt_best.pt"
    last_path = output_dir / "ckpt_last.pt"

    print(
        f"VAE completion dataset={args.dataset} device={device} "
        f"items={full_image.shape[0]} train_items={len(train_ids)} "
        f"image_dim={full_image.shape[1]} text_dim={full_text.shape[1]}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rec = 0.0
        total_kl = 0.0
        total_cf = 0.0
        total_latent_consistency = 0.0
        total_single_view_rec = 0.0
        total_fusion_consistency = 0.0
        total_count = 0
        kl_weight = args.beta_kl
        if args.kl_warmup_epochs > 0:
            kl_weight *= min(1.0, epoch / float(args.kl_warmup_epochs))

        for batch in train_loader:
            _, image_in, text_in, observed, image_target, text_target = batch_to_device(batch, device)
            observed_in = maybe_apply_input_dropout(observed, args.input_dropout)
            image_rec, text_rec, mu, logvar = model(image_in, text_in, observed_in)
            rec = observed_reconstruction_loss(image_rec, text_rec, image_target, text_target, observed)
            kl = kl_loss(mu, logvar)
            latent_consistency = image_rec.sum() * 0.0
            if args.latent_consistency_weight > 0.0 and epoch >= args.latent_consistency_start_epoch:
                latent_consistency = latent_consistency_loss(
                    model,
                    image_in,
                    text_in,
                    observed,
                    detach_anchor=bool(args.latent_consistency_detach_anchor),
                )
            single_view_rec = image_rec.sum() * 0.0
            if args.single_view_rec_weight > 0.0 and epoch >= args.single_view_rec_start_epoch:
                single_view_rec = single_view_reconstruction_loss(
                    model,
                    image_in,
                    text_in,
                    observed,
                )
            fusion_consistency = image_rec.sum() * 0.0
            if args.fusion_consistency_weight > 0.0 and epoch >= args.fusion_consistency_start_epoch:
                fusion_consistency = completed_fusion_consistency_loss(
                    model,
                    image_in,
                    text_in,
                    observed,
                    temp=args.fusion_consistency_temp,
                    detach_target=bool(args.fusion_consistency_detach_target),
                )
            cf = image_rec.sum() * 0.0
            if cf_neighbor_lists is not None:
                cf = cf_alignment_loss(
                    model,
                    image_in,
                    text_in,
                    observed_in,
                    image_rec,
                    text_rec,
                    item_ids=batch[0],
                    neighbor_lists=cf_neighbor_lists,
                    train_split=train_split,
                    device=device,
                    rng=cf_rng,
                    temp=args.cf_align_temp,
                )
            loss = (
                rec
                + kl_weight * kl
                + args.latent_consistency_weight * latent_consistency
                + args.single_view_rec_weight * single_view_rec
                + args.fusion_consistency_weight * fusion_consistency
                + args.cf_align_weight * cf
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size = image_in.size(0)
            total_loss += float(loss.detach().cpu()) * batch_size
            total_rec += float(rec.detach().cpu()) * batch_size
            total_kl += float(kl.detach().cpu()) * batch_size
            total_cf += float(cf.detach().cpu()) * batch_size
            total_latent_consistency += float(latent_consistency.detach().cpu()) * batch_size
            total_single_view_rec += float(single_view_rec.detach().cpu()) * batch_size
            total_fusion_consistency += float(fusion_consistency.detach().cpu()) * batch_size
            total_count += batch_size

        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_count,
            "train_rec": total_rec / total_count,
            "train_kl": total_kl / total_count,
            "train_cf": total_cf / total_count,
            "train_latent_consistency": total_latent_consistency / total_count,
            "train_single_view_rec": total_single_view_rec / total_count,
            "train_fusion_consistency": total_fusion_consistency / total_count,
            "kl_weight": kl_weight,
        }

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            valid_metrics = evaluate_missing(model, valid_split, full_image, full_text, device, args.batch_size)
            test_metrics = evaluate_missing(model, test_split, full_image, full_text, device, args.batch_size)
            row["valid"] = valid_metrics
            row["test"] = test_metrics
            score = valid_metrics["_overall"]["cosine"]
            print(
                f"epoch={epoch:03d} loss={row['train_loss']:.6f} rec={row['train_rec']:.6f} "
                f"kl={row['train_kl']:.6f} lc={row['train_latent_consistency']:.6f} "
                f"sv={row['train_single_view_rec']:.6f} "
                f"fc={row['train_fusion_consistency']:.6f} "
                f"cf={row['train_cf']:.6f} "
                f"valid_cos={valid_metrics['_overall']['cosine']:.6f} "
                f"valid_mse={valid_metrics['_overall']['mse']:.6f} "
                f"test_cos={test_metrics['_overall']['cosine']:.6f} "
                f"test_mse={test_metrics['_overall']['mse']:.6f}"
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                no_improve = 0
                save_checkpoint(best_path, model, args, epoch, row)
            else:
                no_improve += 1
        else:
            print(
                f"epoch={epoch:03d} loss={row['train_loss']:.6f} rec={row['train_rec']:.6f} "
                f"kl={row['train_kl']:.6f} lc={row['train_latent_consistency']:.6f} "
                f"sv={row['train_single_view_rec']:.6f} "
                f"fc={row['train_fusion_consistency']:.6f} "
                f"cf={row['train_cf']:.6f}"
            )
        history.append(row)
        if args.early_stop > 0 and no_improve >= args.early_stop:
            print(f"early_stop at epoch={epoch}, best_epoch={best_epoch}, best_valid_cos={best_score:.6f}")
            break

    save_checkpoint(last_path, model, args, epoch, history[-1])
    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model"])

    final_valid = evaluate_missing(model, valid_split, full_image, full_text, device, args.batch_size)
    final_test = evaluate_missing(model, test_split, full_image, full_text, device, args.batch_size)
    export_completed(model, train_split, full_image, full_text, device, args.batch_size, output_dir / "phase_train")
    export_completed(model, test_split, full_image, full_text, device, args.batch_size, output_dir / "phase_eval")
    export_completed(model, graph_split, full_image, full_text, device, args.batch_size, output_dir / "phase_graph")

    manifest = {
        "variant": "vae_completion",
        "dataset": args.dataset,
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "best_epoch": best_epoch,
        "best_valid_cosine": best_score,
        "train_missing_rate": args.train_missing_rate,
        "eval_missing_rate": args.eval_missing_rate,
        "missing_seed": args.missing_seed,
        "normalize_features": bool(args.normalize_features),
        "normalize_outputs": bool(args.normalize_outputs),
        "input_dropout": args.input_dropout,
        "cf_align_weight": args.cf_align_weight,
        "cf_align_dim": args.cf_align_dim,
        "cf_align_temp": args.cf_align_temp,
        "cf_max_neighbors": args.cf_max_neighbors,
        "cf_user_sample_size": args.cf_user_sample_size,
        "latent_consistency_weight": args.latent_consistency_weight,
        "latent_consistency_detach_anchor": bool(args.latent_consistency_detach_anchor),
        "latent_consistency_start_epoch": args.latent_consistency_start_epoch,
        "single_view_rec_weight": args.single_view_rec_weight,
        "single_view_rec_start_epoch": args.single_view_rec_start_epoch,
        "fusion_consistency_weight": args.fusion_consistency_weight,
        "fusion_consistency_start_epoch": args.fusion_consistency_start_epoch,
        "fusion_consistency_temp": args.fusion_consistency_temp,
        "fusion_consistency_detach_target": bool(args.fusion_consistency_detach_target),
        "files": {
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "phase_train": str(output_dir / "phase_train"),
            "phase_eval": str(output_dir / "phase_eval"),
            "phase_graph": str(output_dir / "phase_graph"),
        },
        "final_valid": final_valid,
        "final_test": final_test,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"history": history, "manifest": manifest}, f, indent=2)

    print(
        f"done best_epoch={best_epoch} valid_cos={final_valid['_overall']['cosine']:.6f} "
        f"test_cos={final_test['_overall']['cosine']:.6f} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
