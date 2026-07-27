import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import dataset_loader
import tool
from enviroment import Env
from dataset_loader import Loader4MM
from model import MILK_model


def _load_config_file(config_path):
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')

    suffix = path.suffix.lower()
    if suffix in {'.yaml', '.yml'}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError('PyYAML is required for yaml config files') from exc
        with path.open('r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    elif suffix == '.json':
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        raise ValueError(f'Unsupported config format: {path.suffix}')

    if not isinstance(data, dict):
        raise ValueError(f'Config file must contain a top-level mapping: {path}')
    if 'args' in data:
        data = data['args']
        if not isinstance(data, dict):
            raise ValueError(f'Config "args" must be a mapping: {path}')

    normalized = {}
    for key, value in data.items():
        normalized[key.replace('-', '_')] = value
    if isinstance(normalized.get('topk'), list):
        normalized['topk'] = str(normalized['topk'])
    return normalized


def _build_parser():
    parser = argparse.ArgumentParser(description='Diagnose imputed-vs-zero representations and score margins')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--suffix', type=str, default='diagnose_imputation_vs_zero')
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--dataset', type=str, default='baby')
    parser.add_argument('--exp_mode', type=str, default='mm')
    parser.add_argument('--model', type=str, default='MILK')
    parser.add_argument('--train_stage', type=str, default='recommender')
    parser.add_argument('--freeze_imputer', type=int, default=1)
    parser.add_argument('--freeze_recommender', type=int, default=-1)
    parser.add_argument('--freeze_decoder', type=int, default=1)
    parser.add_argument('--free_emb_dimension', type=int, default=64)
    parser.add_argument('--contra_dim', type=int, default=64)
    parser.add_argument('--d_beta', type=int, default=32)
    parser.add_argument('--missing_rate', type=float, default=0.3)
    parser.add_argument('--feature_bridge_mode', type=str, default='raw_decoder', choices=['raw_decoder'])
    parser.add_argument('--gcn_frontend_mode', type=str, default='original_linear', choices=['original_linear', 'deep_mlp'])
    parser.add_argument('--lambda_itm', type=float, default=0.1)
    parser.add_argument('--itm_temp', type=float, default=0.07)
    parser.add_argument('--itm_num_heads', type=int, default=4)
    parser.add_argument('--ema_eta', type=float, default=0.01)
    parser.add_argument('--disable_imputation', type=int, default=0)
    parser.add_argument('--alpha_intra', type=float, default=1.0)
    parser.add_argument('--alpha_inter', type=float, default=1.0)
    parser.add_argument('--alpha_itm', type=float, default=1.0)
    parser.add_argument('--alpha_rec', type=float, default=0.1)
    parser.add_argument('--alpha_decode', type=float, default=1.0)
    parser.add_argument('--beta_intra', type=float, default=0.05)
    parser.add_argument('--beta_inter', type=float, default=0.05)
    parser.add_argument('--beta_itm', type=float, default=0.05)
    parser.add_argument('--beta_rec', type=float, default=0.01)
    parser.add_argument('--beta_decode', type=float, default=0.01)
    parser.add_argument('--gamma_align', type=float, default=0.0)
    parser.add_argument('--gamma_distill', type=float, default=0.0)
    parser.add_argument('--reg_coeff', type=float, default=1e-4)
    parser.add_argument('--penalty_coeff', type=float, default=50)
    parser.add_argument('--max_info_coeff', type=float, default=0.05)
    parser.add_argument('--min_info_coeff', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_rec', type=float, default=None)
    parser.add_argument('--lr_imp', type=float, default=None)
    parser.add_argument('--lr_decoder', type=float, default=None)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--eva_interval', type=int, default=1)
    parser.add_argument('--neg_num', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--topk', type=str, default='[10, 20, 30, 40, 50]')
    parser.add_argument('--selection_mode', type=str, default='val')
    parser.add_argument('--imputation_selection_split', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--imputation_selection_metric', type=str, default='mse', choices=['mse', 'cosine'])
    parser.add_argument('--completion_ckpt', type=str, required=True)
    parser.add_argument('--control_ckpt', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--num_pairs', type=int, default=10000)
    parser.add_argument('--json_output', type=str, default='')
    parser.add_argument('--log', type=int, default=0)
    parser.add_argument('--tensorboard', type=int, default=0)
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--hf_tensorboard_repo', type=str, default='')
    parser.add_argument('--hf_token', type=str, default='')
    parser.add_argument('--hf_commit_every', type=int, default=5)
    return parser


def parse_args():
    parser = _build_parser()
    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        parser.set_defaults(**_load_config_file(pre_args.config))
    args = parser.parse_args()
    if args.config is not None:
        args.config = str(Path(args.config).expanduser().resolve())
    return args


def _build_model(env, loader, ckpt_path, disable_imputation):
    model = MILK_model(env, loader)
    loaded = model.load_full_checkpoint(ckpt_path)
    model.disable_imputation = bool(disable_imputation)
    model.eval()
    tool.cprint(f'Loaded checkpoint {ckpt_path} ({loaded} tensors), disable_imputation={int(model.disable_imputation)}')
    return model


@torch.no_grad()
def _collect_states(model, split):
    raw = model.get_split_raw_modal_features(split=split, full=False)
    full_raw = model.get_split_raw_modal_features(split=split, full=True)
    user_id_emb = model.user_emb.weight

    gcn_input = model.get_recommender_modal_features(raw_features=raw)
    gcn_input_full = model.get_recommender_modal_features(raw_features=full_raw)

    mlp = {}
    mlp_full = {}
    modal_gcn = {}
    modal_gcn_full = {}

    v_user_emb, v_item_emb = model.v_gcn(gcn_input["v"], user_id_emb)
    v_user_full, v_item_full = model.v_gcn(gcn_input_full["v"], user_id_emb)
    mlp["v"] = model.v_gcn.MLP(gcn_input["v"])
    mlp_full["v"] = model.v_gcn.MLP(gcn_input_full["v"])
    modal_gcn["v"] = v_item_emb
    modal_gcn_full["v"] = v_item_full

    t_user_emb, t_item_emb = model.t_gcn(gcn_input["t"], user_id_emb)
    t_user_full, t_item_full = model.t_gcn(gcn_input_full["t"], user_id_emb)
    mlp["t"] = model.t_gcn.MLP(gcn_input["t"])
    mlp_full["t"] = model.t_gcn.MLP(gcn_input_full["t"])
    modal_gcn["t"] = t_item_emb
    modal_gcn_full["t"] = t_item_full

    if "a" in model.modalities:
        a_user_emb, a_item_emb = model.a_gcn(gcn_input["a"], user_id_emb)
        a_user_full, a_item_full = model.a_gcn(gcn_input_full["a"], user_id_emb)
        mlp["a"] = model.a_gcn.MLP(gcn_input["a"])
        mlp_full["a"] = model.a_gcn.MLP(gcn_input_full["a"])
        modal_gcn["a"] = a_item_emb
        modal_gcn_full["a"] = a_item_full
        user_emb = user_id_emb + (v_user_emb + t_user_emb + a_user_emb) / 3
        user_emb_full = user_id_emb + (v_user_full + t_user_full + a_user_full) / 3
        item_source = (v_item_emb + t_item_emb + a_item_emb) / 3
        item_source_full = (v_item_full + t_item_full + a_item_full) / 3
    else:
        user_emb = user_id_emb + (v_user_emb + t_user_emb) / 2
        user_emb_full = user_id_emb + (v_user_full + t_user_full) / 2
        item_source = (v_item_emb + t_item_emb) / 2
        item_source_full = (v_item_full + t_item_full) / 2

    final_item = model._apply_fusion(item_source, deterministic=True)
    final_item_full = model._apply_fusion(item_source_full, deterministic=True)

    return {
        'gcn_input': gcn_input,
        'gcn_input_full': gcn_input_full,
        'mlp': mlp,
        'mlp_full': mlp_full,
        'modal_gcn': modal_gcn,
        'modal_gcn_full': modal_gcn_full,
        'user': user_emb,
        'user_full': user_emb_full,
        'final_item': final_item,
        'final_item_full': final_item_full,
    }


def _tensor_metrics(actual, teacher):
    if actual.numel() == 0:
        return {'count': 0, 'mse': 0.0, 'cosine': 0.0, 'actual_norm': 0.0, 'teacher_norm': 0.0}
    return {
        'count': int(actual.size(0)),
        'mse': float(F.mse_loss(actual, teacher, reduction='mean').item()),
        'cosine': float(F.cosine_similarity(actual, teacher, dim=-1).mean().item()),
        'actual_norm': float(actual.norm(dim=-1).mean().item()),
        'teacher_norm': float(teacher.norm(dim=-1).mean().item()),
    }


def _collect_layer_metrics(model, states, split):
    missing_items, missing_indicators = model.get_missing_item_metadata(split=split)
    results = {}
    overall_acc = {
        'gcn_input': {'count': 0, 'mse': 0.0, 'cosine': 0.0, 'actual_norm': 0.0, 'teacher_norm': 0.0},
        'mlp': {'count': 0, 'mse': 0.0, 'cosine': 0.0, 'actual_norm': 0.0, 'teacher_norm': 0.0},
        'modal_gcn': {'count': 0, 'mse': 0.0, 'cosine': 0.0, 'actual_norm': 0.0, 'teacher_norm': 0.0},
    }
    for modality_idx, modality in enumerate(model.modalities):
        selector = missing_indicators == modality_idx
        item_ids = missing_items[selector]
        if item_ids.numel() == 0:
            continue
        results.setdefault(modality, {})
        results[modality]['gcn_input'] = _tensor_metrics(states['gcn_input'][modality][item_ids], states['gcn_input_full'][modality][item_ids])
        results[modality]['mlp'] = _tensor_metrics(states['mlp'][modality][item_ids], states['mlp_full'][modality][item_ids])
        results[modality]['modal_gcn'] = _tensor_metrics(states['modal_gcn'][modality][item_ids], states['modal_gcn_full'][modality][item_ids])
        results[modality]['final_item'] = _tensor_metrics(states['final_item'][item_ids], states['final_item_full'][item_ids])
        for layer_name in ('gcn_input', 'mlp', 'modal_gcn'):
            metrics = results[modality][layer_name]
            count = metrics['count']
            overall_acc[layer_name]['count'] += count
            for key in ('mse', 'cosine', 'actual_norm', 'teacher_norm'):
                overall_acc[layer_name][key] += metrics[key] * count

    if missing_items.numel() > 0:
        overall = {}
        for layer_name, acc in overall_acc.items():
            count = acc['count']
            if count == 0:
                overall[layer_name] = {'count': 0, 'mse': 0.0, 'cosine': 0.0, 'actual_norm': 0.0, 'teacher_norm': 0.0}
            else:
                overall[layer_name] = {
                    'count': count,
                    'mse': acc['mse'] / count,
                    'cosine': acc['cosine'] / count,
                    'actual_norm': acc['actual_norm'] / count,
                    'teacher_norm': acc['teacher_norm'] / count,
                }
        results['_overall'] = {
            'gcn_input': overall['gcn_input'],
            'mlp': overall['mlp'],
            'modal_gcn': overall['modal_gcn'],
            'final_item': _tensor_metrics(states['final_item'][missing_items], states['final_item_full'][missing_items]),
        }
    return results


@torch.no_grad()
def _collect_margin_metrics(model, states, loader, num_pairs):
    S = dataset_loader.PairSample(loader)
    if len(S) == 0:
        return {}
    if num_pairs > 0 and len(S) > num_pairs:
        S = S[:num_pairs]
    users = torch.as_tensor(S[:, 0], dtype=torch.long, device=model.env.device)
    pos_items = torch.as_tensor(S[:, 1], dtype=torch.long, device=model.env.device)
    neg_items = torch.as_tensor(S[:, 2], dtype=torch.long, device=model.env.device)

    user_emb = states['user'][users]
    pos_emb = states['final_item'][pos_items]
    neg_emb = states['final_item'][neg_items]
    margin = torch.sum(user_emb * pos_emb, dim=1) - torch.sum(user_emb * neg_emb, dim=1)

    missing_items, _ = model.get_missing_item_metadata(split='train')
    missing_mask_lookup = torch.zeros(loader.m_item, dtype=torch.bool, device=model.env.device)
    if missing_items.numel() > 0:
        missing_mask_lookup[missing_items] = True
    pos_missing = missing_mask_lookup[pos_items]
    neg_missing = missing_mask_lookup[neg_items]
    any_missing = pos_missing | neg_missing

    def subset_stats(selector):
        if selector.numel() == 0 or selector.sum() == 0:
            return {'count': 0, 'mean_margin': 0.0, 'positive_ratio': 0.0}
        sub = margin[selector]
        return {
            'count': int(selector.sum().item()),
            'mean_margin': float(sub.mean().item()),
            'positive_ratio': float((sub > 0).float().mean().item()),
        }

    return {
        'all_pairs': subset_stats(torch.ones_like(any_missing, dtype=torch.bool)),
        'pos_missing_pairs': subset_stats(pos_missing),
        'any_missing_pairs': subset_stats(any_missing),
    }


def _compare_margin_metrics(completion_metrics, control_metrics):
    results = {}
    for key in completion_metrics.keys():
        comp = completion_metrics[key]
        ctrl = control_metrics.get(key, {})
        results[key] = {
            'completion': comp,
            'control': ctrl,
            'margin_gain': float(comp.get('mean_margin', 0.0) - ctrl.get('mean_margin', 0.0)),
            'positive_ratio_gain': float(comp.get('positive_ratio', 0.0) - ctrl.get('positive_ratio', 0.0)),
        }
    return results


def _print_layer_summary(title, metrics):
    tool.cprint(f'--- {title} ---')
    for scope, scope_metrics in metrics.items():
        print(f'[{scope}]')
        for layer_name, values in scope_metrics.items():
            print(
                f"  {layer_name}: count={values['count']}, mse={values['mse']:.6f}, cosine={values['cosine']:.6f}, "
                f"actual_norm={values['actual_norm']:.6f}, teacher_norm={values['teacher_norm']:.6f}"
            )


def _print_margin_summary(title, metrics):
    tool.cprint(f'--- {title} ---')
    for subset, values in metrics.items():
        print(
            f"{subset}: count={values.get('count', 0)}, "
            f"mean_margin={values.get('mean_margin', 0.0):.6f}, "
            f"positive_ratio={values.get('positive_ratio', 0.0):.6f}"
        )


def main():
    args = parse_args()
    env = Env(args)
    loader = Loader4MM(env)

    completion_model = _build_model(env, loader, args.completion_ckpt, disable_imputation=False)
    control_model = _build_model(env, loader, args.control_ckpt, disable_imputation=True)

    completion_states = _collect_states(completion_model, split=args.split)
    control_states = _collect_states(control_model, split=args.split)

    completion_layer_metrics = _collect_layer_metrics(completion_model, completion_states, split=args.split)
    control_layer_metrics = _collect_layer_metrics(control_model, control_states, split=args.split)

    completion_margin_metrics = _collect_margin_metrics(completion_model, completion_states, loader, args.num_pairs)
    control_margin_metrics = _collect_margin_metrics(control_model, control_states, loader, args.num_pairs)
    compared_margin = _compare_margin_metrics(completion_margin_metrics, control_margin_metrics)

    _print_layer_summary('completion model layer metrics', completion_layer_metrics)
    _print_layer_summary('control model layer metrics', control_layer_metrics)
    _print_margin_summary('completion model margins', completion_margin_metrics)
    _print_margin_summary('control model margins', control_margin_metrics)
    tool.cprint('--- margin comparison (completion - control) ---')
    for subset, values in compared_margin.items():
        print(
            f"{subset}: margin_gain={values['margin_gain']:.6f}, "
            f"positive_ratio_gain={values['positive_ratio_gain']:.6f}"
        )

    payload = {
        'split': args.split,
        'completion_ckpt': str(Path(args.completion_ckpt).expanduser().resolve()),
        'control_ckpt': str(Path(args.control_ckpt).expanduser().resolve()),
        'completion_layer_metrics': completion_layer_metrics,
        'control_layer_metrics': control_layer_metrics,
        'completion_margin_metrics': completion_margin_metrics,
        'control_margin_metrics': control_margin_metrics,
        'margin_comparison': compared_margin,
    }

    if args.json_output:
        output_path = Path(args.json_output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        tool.cprint(f'Saved diagnostics to {output_path}')

    env.close_env()


if __name__ == '__main__':
    main()
