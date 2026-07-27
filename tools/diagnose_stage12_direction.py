import argparse
import glob
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import tool
from enviroment import Env
from dataset_loader import Loader4MM
from tools.diagnose_imputation_vs_zero import (
    _build_model,
    _collect_states,
    _collect_layer_metrics,
    _collect_margin_metrics,
    _compare_margin_metrics,
    _load_config_file,
)


def _build_parser():
    parser = argparse.ArgumentParser(description='Diagnose whether stage1.2 training helps or hurts recommendation.')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--suffix', type=str, default='diagnose_stage12_direction')
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
    parser.add_argument('--feature_bridge_mode', type=str, default='raw_decoder')
    parser.add_argument('--gcn_frontend_mode', type=str, default='original_linear')
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
    parser.add_argument('--ckpt_glob', type=str, required=True)
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


def _principal_direction(tensor):
    if tensor.numel() == 0 or tensor.size(0) < 2:
        return None
    centered = tensor.float() - tensor.float().mean(dim=0, keepdim=True)
    if centered.abs().sum().item() == 0:
        return None
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return F.normalize(vh[0], dim=0)


def _anchor_shift(actual, teacher):
    actual_dir = _principal_direction(actual)
    teacher_dir = _principal_direction(teacher)
    if actual_dir is None or teacher_dir is None:
        return {'shift_l2': 0.0, 'anchor_cosine': 1.0}
    shift = min(
        torch.norm(actual_dir - teacher_dir).item(),
        torch.norm(actual_dir + teacher_dir).item(),
    )
    cosine = abs(torch.dot(actual_dir, teacher_dir).item())
    return {'shift_l2': float(shift), 'anchor_cosine': float(cosine)}


def _collect_anchor_metrics(model, states, split):
    missing_items, missing_indicators = model.get_missing_item_metadata(split=split)
    results = {}
    for modality_idx, modality in enumerate(model.modalities):
        selector = missing_indicators == modality_idx
        item_ids = missing_items[selector]
        if item_ids.numel() == 0:
            continue
        results[modality] = {
            'gcn_input': _anchor_shift(states['gcn_input'][modality][item_ids], states['gcn_input_full'][modality][item_ids]),
            'mlp': _anchor_shift(states['mlp'][modality][item_ids], states['mlp_full'][modality][item_ids]),
            'modal_gcn': _anchor_shift(states['modal_gcn'][modality][item_ids], states['modal_gcn_full'][modality][item_ids]),
            'final_item': _anchor_shift(states['final_item'][item_ids], states['final_item_full'][item_ids]),
        }
    if missing_items.numel() > 0:
        overall = {}
        for layer_name in ('gcn_input', 'mlp', 'modal_gcn'):
            total = 0
            shift_sum = 0.0
            cosine_sum = 0.0
            for modality_idx, modality in enumerate(model.modalities):
                selector = missing_indicators == modality_idx
                item_ids = missing_items[selector]
                if item_ids.numel() == 0:
                    continue
                metric = _anchor_shift(states[layer_name][modality][item_ids], states[f'{layer_name}_full'][modality][item_ids])
                count = int(item_ids.numel())
                total += count
                shift_sum += metric['shift_l2'] * count
                cosine_sum += metric['anchor_cosine'] * count
            if total > 0:
                overall[layer_name] = {
                    'shift_l2': float(shift_sum / total),
                    'anchor_cosine': float(cosine_sum / total),
                }
            else:
                overall[layer_name] = {'shift_l2': 0.0, 'anchor_cosine': 1.0}
        overall['final_item'] = _anchor_shift(states['final_item'][missing_items], states['final_item_full'][missing_items])
        results['_overall'] = overall
    return results


def _extract_epoch(path):
    stem = Path(path).stem
    marker = '_epoch'
    idx = stem.rfind(marker)
    if idx == -1:
        return -1
    try:
        return int(stem[idx + len(marker):])
    except ValueError:
        return -1


def _summarize_epoch(path, completion_layer_metrics, control_layer_metrics, completion_anchor_metrics, control_anchor_metrics, margin_comparison):
    comp_overall = completion_layer_metrics.get('_overall', {})
    ctrl_overall = control_layer_metrics.get('_overall', {})
    comp_anchor = completion_anchor_metrics.get('_overall', {})
    ctrl_anchor = control_anchor_metrics.get('_overall', {})
    any_missing = margin_comparison.get('any_missing_pairs', {})
    pos_missing = margin_comparison.get('pos_missing_pairs', {})
    return {
        'ckpt': str(Path(path).resolve()),
        'epoch': _extract_epoch(path),
        'completion_final_item_mse': comp_overall.get('final_item', {}).get('mse', 0.0),
        'completion_final_item_cosine': comp_overall.get('final_item', {}).get('cosine', 0.0),
        'control_final_item_mse': ctrl_overall.get('final_item', {}).get('mse', 0.0),
        'control_final_item_cosine': ctrl_overall.get('final_item', {}).get('cosine', 0.0),
        'completion_final_item_anchor_shift': comp_anchor.get('final_item', {}).get('shift_l2', 0.0),
        'completion_final_item_anchor_cosine': comp_anchor.get('final_item', {}).get('anchor_cosine', 1.0),
        'control_final_item_anchor_shift': ctrl_anchor.get('final_item', {}).get('shift_l2', 0.0),
        'control_final_item_anchor_cosine': ctrl_anchor.get('final_item', {}).get('anchor_cosine', 1.0),
        'completion_gcn_input_mse': comp_overall.get('gcn_input', {}).get('mse', 0.0),
        'completion_gcn_input_cosine': comp_overall.get('gcn_input', {}).get('cosine', 0.0),
        'completion_gcn_input_anchor_shift': comp_anchor.get('gcn_input', {}).get('shift_l2', 0.0),
        'control_gcn_input_anchor_shift': ctrl_anchor.get('gcn_input', {}).get('shift_l2', 0.0),
        'any_missing_margin_gain': any_missing.get('margin_gain', 0.0),
        'any_missing_positive_ratio_gain': any_missing.get('positive_ratio_gain', 0.0),
        'pos_missing_margin_gain': pos_missing.get('margin_gain', 0.0),
        'pos_missing_positive_ratio_gain': pos_missing.get('positive_ratio_gain', 0.0),
    }


def _print_table(rows):
    headers = [
        'epoch',
        'comp_final_mse',
        'comp_final_cos',
        'comp_final_shift',
        'ctrl_final_shift',
        'any_margin_gain',
        'pos_margin_gain',
        'any_pos_ratio_gain',
    ]
    print('\t'.join(headers))
    for row in rows:
        print(
            f"{row['epoch']}\t"
            f"{row['completion_final_item_mse']:.6f}\t"
            f"{row['completion_final_item_cosine']:.6f}\t"
            f"{row['completion_final_item_anchor_shift']:.6f}\t"
            f"{row['control_final_item_anchor_shift']:.6f}\t"
            f"{row['any_missing_margin_gain']:.6f}\t"
            f"{row['pos_missing_margin_gain']:.6f}\t"
            f"{row['any_missing_positive_ratio_gain']:.6f}"
        )


def main():
    args = parse_args()
    env = Env(args)
    loader = Loader4MM(env)

    ckpt_paths = sorted(glob.glob(args.ckpt_glob), key=_extract_epoch)
    if not ckpt_paths:
        raise FileNotFoundError(f'No checkpoints matched: {args.ckpt_glob}')

    summaries = []
    details = []
    for ckpt_path in ckpt_paths:
        tool.cprint(f'=== diagnose {ckpt_path} ===')
        completion_model = _build_model(env, loader, ckpt_path, disable_imputation=False)
        control_model = _build_model(env, loader, ckpt_path, disable_imputation=True)

        completion_states = _collect_states(completion_model, split=args.split)
        control_states = _collect_states(control_model, split=args.split)

        completion_layer_metrics = _collect_layer_metrics(completion_model, completion_states, split=args.split)
        control_layer_metrics = _collect_layer_metrics(control_model, control_states, split=args.split)
        completion_anchor_metrics = _collect_anchor_metrics(completion_model, completion_states, split=args.split)
        control_anchor_metrics = _collect_anchor_metrics(control_model, control_states, split=args.split)

        completion_margin_metrics = _collect_margin_metrics(completion_model, completion_states, loader, args.num_pairs)
        control_margin_metrics = _collect_margin_metrics(control_model, control_states, loader, args.num_pairs)
        compared_margin = _compare_margin_metrics(completion_margin_metrics, control_margin_metrics)

        summary = _summarize_epoch(
            ckpt_path,
            completion_layer_metrics,
            control_layer_metrics,
            completion_anchor_metrics,
            control_anchor_metrics,
            compared_margin,
        )
        summaries.append(summary)
        details.append({
            'summary': summary,
            'completion_layer_metrics': completion_layer_metrics,
            'control_layer_metrics': control_layer_metrics,
            'completion_anchor_metrics': completion_anchor_metrics,
            'control_anchor_metrics': control_anchor_metrics,
            'completion_margin_metrics': completion_margin_metrics,
            'control_margin_metrics': control_margin_metrics,
            'margin_comparison': compared_margin,
        })

    _print_table(summaries)

    if args.json_output:
        output_path = Path(args.json_output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump({'rows': summaries, 'details': details}, f, indent=2)
        tool.cprint(f'Saved diagnostics to {output_path}')

    env.close_env()


if __name__ == '__main__':
    main()
