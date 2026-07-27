import argparse
import ast
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from enviroment import Env
from dataset_loader import Loader4MM
from model import MILK_model


def _read_namespace_from_log(path):
    text = Path(path).read_text(errors="ignore")
    match = re.search(r"Namespace\((.*)\)", text)
    if not match:
        raise ValueError(f"Namespace(...) not found in {path}")
    pairs = {}
    for node in ast.parse(f"f({match.group(1)})", mode="eval").body.keywords:
        pairs[node.arg] = ast.literal_eval(node.value)
    return argparse.Namespace(**pairs)


def _fill_compat_defaults(args):
    defaults = {
        "completion_gate_learn_mix": 0,
        "completion_gate_mix_max": 1.0,
        "completion_gate_shrink_init_logit": -4.0,
        "completion_gate_tail_quantile": 1.0,
        "completion_gate_only_train": 0,
        "completion_gate_score_residual_alpha": 0.0,
        "completion_gate_no_residual_alpha": 0,
        "missing_fusion_imputed_weight": 0.7,
        "fusion_mode": "mean",
        "rum_tau": 1.0,
        "rum_reliability_coeff": 1.0,
        "rum_match_coeff": 1.0,
        "rum_eval_user_batch_size": 256,
        "rum_eval_item_chunk_size": 4096,
        "save_all_epochs": 0,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def _prepare_args(log_path, ckpt_path, device_id, suffix):
    args = _fill_compat_defaults(_read_namespace_from_log(log_path))
    args.ckpt = ckpt_path
    args.imputer_ckpt = None
    args.device_id = device_id
    args.use_gpu = 1
    args.tensorboard = 0
    args.log = 0
    args.save = 0
    args.suffix = suffix
    return args


def _load_model(args):
    env = Env(args)
    loader = Loader4MM(env)
    model = MILK_model(env, loader)
    if args.ckpt:
        model.load_full_checkpoint(args.ckpt)
    model.eval()
    return env, loader, model


@torch.no_grad()
def _embeddings(model, split):
    model.eval()
    model.set_missing_modality_via_env(eval_split=split)
    users, items = model()
    return users.detach(), items.detach()


def _normalize_rows(scores, mode):
    if mode == "none":
        return scores
    if mode == "zscore":
        mean = scores.mean(dim=1, keepdim=True)
        std = scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (scores - mean) / std
    if mode == "center":
        return scores - scores.mean(dim=1, keepdim=True)
    raise ValueError(f"unknown normalize mode: {mode}")


@torch.no_grad()
def evaluate(loader, base_model, fusion_model, split, alphas, normalize, user_batch_size):
    eval_data = loader.val_data if split == "val" else loader.test_data
    eval_users = list(eval_data.keys())
    topk = 20
    max_train_len = max((len(loader.train_data.get(u, [])) for u in eval_users), default=0)
    keepk = topk + max_train_len

    base_users, base_items = _embeddings(base_model, split)
    fusion_users, fusion_items = _embeddings(fusion_model, split)

    ranked = {alpha: {} for alpha in alphas}
    device = base_users.device
    all_items = torch.arange(loader.m_item, device=device)

    for start in range(0, len(eval_users), user_batch_size):
        batch_user_ids = eval_users[start:start + user_batch_size]
        batch_users = torch.as_tensor(batch_user_ids, dtype=torch.long, device=device)
        base_scores = base_users[batch_users] @ base_items.T
        fusion_scores = fusion_users[batch_users] @ fusion_items.T
        base_scores = _normalize_rows(base_scores, normalize)
        fusion_scores = _normalize_rows(fusion_scores, normalize)

        for alpha in alphas:
            scores = (1.0 - alpha) * base_scores + alpha * fusion_scores
            for row_idx, user_id in enumerate(batch_user_ids):
                train_items = loader.train_data.get(user_id, [])
                if train_items:
                    item_tensor = torch.as_tensor(train_items, dtype=torch.long, device=device)
                    scores[row_idx, item_tensor] = -float("inf")
            _, indices = torch.topk(scores, k=min(keepk, loader.m_item), dim=1)
            items = all_items[indices].cpu().numpy()
            for row_idx, user_id in enumerate(batch_user_ids):
                ranked[alpha][user_id] = items[row_idx].tolist()

    out = {}
    for alpha in alphas:
        hrs, recalls, ndcgs = [], [], []
        for user_id in eval_users:
            target = set(eval_data[user_id])
            if not target:
                continue
            pred = ranked[alpha][user_id][:topk]
            hit = 0
            dcg = 0.0
            for idx, item in enumerate(pred):
                if item in target:
                    hit += 1
                    dcg += np.log(2) / np.log(idx + 2)
            ideal_len = min(topk, len(target))
            idcg = sum(np.log(2) / np.log(idx + 2) for idx in range(ideal_len))
            hrs.append(hit / ideal_len if ideal_len else 0.0)
            recalls.append(hit / len(target))
            ndcgs.append(dcg / idcg if idcg else 0.0)
        out[alpha] = (
            float(np.mean(hrs)) if hrs else 0.0,
            float(np.mean(recalls)) if recalls else 0.0,
            float(np.mean(ndcgs)) if ndcgs else 0.0,
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-log", required=True)
    parser.add_argument("--base-ckpt", required=True)
    parser.add_argument("--fusion-log", required=True)
    parser.add_argument("--fusion-ckpt", required=True)
    parser.add_argument("--device-id", type=int, default=3)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--normalize", choices=["none", "center", "zscore"], default="none")
    parser.add_argument("--alphas", default="0,0.02,0.05,0.1,0.2,0.35,0.5,0.75,1")
    parser.add_argument("--user-batch-size", type=int, default=256)
    args = parser.parse_args()

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    base_args = _prepare_args(args.base_log, args.base_ckpt, args.device_id, "late_score_base_eval")
    fusion_args = _prepare_args(args.fusion_log, args.fusion_ckpt, args.device_id, "late_score_fusion_eval")
    if base_args.dataset != fusion_args.dataset or base_args.exp_mode != fusion_args.exp_mode:
        raise ValueError("base and fusion logs must use the same dataset/exp_mode")

    _, loader, base_model = _load_model(base_args)
    _, _, fusion_model = _load_model(fusion_args)
    metrics = evaluate(
        loader,
        base_model,
        fusion_model,
        split=args.split,
        alphas=alphas,
        normalize=args.normalize,
        user_batch_size=args.user_batch_size,
    )
    print("alpha\thr20\trecall20\tndcg20")
    for alpha in alphas:
        hr, recall, ndcg = metrics[alpha]
        print(f"{alpha:.4f}\t{hr:.5f}\t{recall:.5f}\t{ndcg:.5f}")


if __name__ == "__main__":
    main()
