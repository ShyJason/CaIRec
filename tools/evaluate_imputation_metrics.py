import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

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
    parser = argparse.ArgumentParser(description="Evaluate PROMRL imputation quality metrics")
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--suffix', type=str, default='imputation_metric_eval')
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--dataset_seed', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--imputer_ckpt', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='baby')
    parser.add_argument('--exp_mode', type=str, default='mm')
    parser.add_argument('--missing_mask_protocol', type=str, default='i3', choices=['i3', 'default_rng'])
    parser.add_argument('--model', type=str, default='MILK')
    parser.add_argument('--train_stage', type=str, default='recommender')
    parser.add_argument('--freeze_imputer', type=int, default=0)
    parser.add_argument('--freeze_recommender', type=int, default=1)
    parser.add_argument('--freeze_decoder', type=int, default=1)
    parser.add_argument('--free_emb_dimension', type=int, default=64)
    parser.add_argument('--contra_dim', type=int, default=64)
    parser.add_argument('--d_beta', type=int, default=32)
    parser.add_argument('--missing_rate', type=float, default=0.3)
    parser.add_argument('--eval_missing_rate', type=float, default=0.5)
    parser.add_argument('--feature_bridge_mode', type=str, default='raw_decoder', choices=['raw_decoder'])
    parser.add_argument('--gcn_frontend_mode', type=str, default='original_linear', choices=['original_linear', 'deep_mlp'])
    parser.add_argument('--lambda_itm', type=float, default=0.1)
    parser.add_argument('--itm_temp', type=float, default=0.07)
    parser.add_argument('--itm_num_heads', type=int, default=4)
    parser.add_argument('--ema_eta', type=float, default=0.01)
    parser.add_argument('--tau1', type=float, default=0.1)
    parser.add_argument('--tau2', type=float, default=0.1)
    parser.add_argument('--disable_imputation', type=int, default=0)
    parser.add_argument('--completion_gate_mode', type=str, default='off', choices=['off', 'alignment', 'reliability', 'rank_residual', 'rank_residual_norm', 'rank_residual_allnorm', 'rank_residual_allgate', 'rank_residual_softmax', 'rank_residual_global', 'rank_residual_shrink', 'rank_residual_centered', 'rank_residual_delta', 'rank_residual_centered_allgate'])
    parser.add_argument('--completion_gate_hidden_dim', type=int, default=64)
    parser.add_argument('--completion_gate_dropout', type=float, default=0.1)
    parser.add_argument('--completion_gate_init_logit', type=float, default=1.5)
    parser.add_argument('--completion_gate_detach_inputs', type=int, default=1)
    parser.add_argument('--completion_gate_use_item_context', type=int, default=1)
    parser.add_argument('--completion_gate_item_context_source', type=str, default='id_embedding', choices=['id_embedding', 'shared_mean', 'off'])
    parser.add_argument('--completion_gate_floor', type=float, default=0.7)
    parser.add_argument('--completion_gate_target_mean', type=float, default=0.95)
    parser.add_argument('--completion_gate_reg_coeff', type=float, default=0.0)
    parser.add_argument('--completion_gate_alignment_center', type=float, default=0.0)
    parser.add_argument('--completion_gate_alignment_temp', type=float, default=0.2)
    parser.add_argument('--completion_gate_residual_alpha', type=float, default=0.1)
    parser.add_argument('--completion_gate_mix_alpha', type=float, default=0.3)
    parser.add_argument('--completion_gate_identity_coeff', type=float, default=0.05)
    parser.add_argument('--completion_gate_balance_coeff', type=float, default=0.01)
    parser.add_argument('--completion_gate_softmax_temp', type=float, default=1.0)
    parser.add_argument('--completion_gate_advantage_coeff', type=float, default=0.0)
    parser.add_argument('--completion_gate_advantage_margin', type=float, default=0.0)
    parser.add_argument('--completion_gate_learn_mix', type=int, default=0)
    parser.add_argument('--completion_gate_mix_max', type=float, default=1.0)
    parser.add_argument('--completion_gate_shrink_init_logit', type=float, default=-4.0)
    parser.add_argument('--completion_gate_tail_quantile', type=float, default=1.0)
    parser.add_argument('--completion_gate_only_train', type=int, default=0)
    parser.add_argument('--fusion_mode', type=str, default='mean', choices=['mean', 'rum', 'missing_weighted_mean', 'global_weighted_mean'])
    parser.add_argument('--missing_fusion_imputed_weight', type=float, default=0.7)
    parser.add_argument('--rum_tau', type=float, default=1.0)
    parser.add_argument('--rum_reliability_coeff', type=float, default=1.0)
    parser.add_argument('--rum_match_coeff', type=float, default=1.0)
    parser.add_argument('--rum_eval_user_batch_size', type=int, default=256)
    parser.add_argument('--rum_eval_item_chunk_size', type=int, default=4096)
    parser.add_argument('--split', type=str, default='both', choices=['train', 'test', 'both'])
    parser.add_argument(
        '--metric_space',
        type=str,
        default='shared',
        choices=['shared', 'decode', 'both'],
        help='shared evaluates imputed projected features; decode evaluates decoded raw features against oracle raw targets.',
    )
    parser.add_argument('--include_random_baseline', type=int, default=1)
    parser.add_argument('--json_output', type=str, default='')
    parser.add_argument('--tensorboard', type=int, default=0)
    parser.add_argument('--log', type=int, default=0)
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--alpha_intra', type=float, default=1.0)
    parser.add_argument('--alpha_inter', type=float, default=1.0)
    parser.add_argument('--alpha_itm', type=float, default=1.0)
    parser.add_argument('--alpha_rec', type=float, default=0.1)
    parser.add_argument('--alpha_decode', type=float, default=1.0)
    parser.add_argument('--modality_bpr_coeff', type=float, default=0.2)
    parser.add_argument('--beta_intra', type=float, default=0.005)
    parser.add_argument('--beta_inter', type=float, default=0.005)
    parser.add_argument('--beta_itm', type=float, default=0.005)
    parser.add_argument('--beta_rec', type=float, default=0.001)
    parser.add_argument('--beta_decode', type=float, default=0.003)
    parser.add_argument('--gamma_align', type=float, default=0.0)
    parser.add_argument('--gamma_distill', type=float, default=0.0)
    parser.add_argument('--joint_allow_modal_grad', type=int, default=0)
    parser.add_argument('--recommender_allow_modal_grad', type=int, default=0)
    parser.add_argument('--stage1_profile', type=str, default='legacy', choices=['legacy', 'v2'])
    parser.add_argument('--stage1_v2_loss_preset', type=str, default='legacy', choices=['legacy', 'balanced'])
    parser.add_argument('--generative_update_mode', type=str, default='em', choices=['em', 'gradient'])
    parser.add_argument('--stage1_masking_policy', type=str, default='fixed', choices=['fixed', 'dynamic'])
    parser.add_argument('--imputation_val_rate', type=float, default=0.0)
    parser.add_argument('--imputation_selection_policy', type=str, default='legacy', choices=['legacy', 'stage1_default', 'promrl_shared', 'adapter_default'])
    parser.add_argument('--imputation_selection_split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--imputation_selection_metric', type=str, default='mse', choices=['mse', 'cosine'])
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


def _print_metrics(split_name, metrics, metric_space='shared'):
    tool.cprint(f'--- {split_name} {metric_space} imputation metrics ---')
    for modality, values in metrics.items():
        if modality == '_overall':
            continue
        line = (
            f"{modality}: count={values['count']}, "
            f"mse={values['mse']:.6f}, cosine={values['cosine']:.6f}"
        )
        if 'random_mse' in values:
            line += f", random_mse={values['random_mse']:.6f}, random_cosine={values['random_cosine']:.6f}"
        print(line)
    overall = metrics['_overall']
    line = (
        f"overall: count={overall['count']}, "
        f"mse={overall['mse']:.6f}, cosine={overall['cosine']:.6f}"
    )
    if 'random_mse' in overall:
        line += f", random_mse={overall['random_mse']:.6f}, random_cosine={overall['random_cosine']:.6f}"
    print(line)


def main():
    args = parse_args()
    env = Env(args)
    loader = Loader4MM(env)
    model = MILK_model(env, loader)
    if args.ckpt:
        loaded = model.load_full_checkpoint(args.ckpt)
        tool.cprint(f'Loaded full checkpoint from {args.ckpt} ({loaded} tensors)')
    if args.imputer_ckpt:
        loaded = model.load_imputer_checkpoint(args.imputer_ckpt)
        tool.cprint(f'Loaded imputer checkpoint from {args.imputer_ckpt} ({len(loaded)} tensors)')

    model.eval()

    split_names = ['train', 'test'] if args.split == 'both' else [args.split]
    payload = {}
    for split_name in split_names:
        payload[split_name] = {}
        if args.metric_space in {'shared', 'both'}:
            metrics = model.compute_imputation_representation_metrics(
                split=split_name,
                include_random_baseline=bool(args.include_random_baseline),
            )
            payload[split_name]['shared'] = metrics
            _print_metrics(split_name, metrics, metric_space='shared')
        if args.metric_space in {'decode', 'both'}:
            metrics = model.compute_missing_decode_metrics(
                split=split_name,
                include_random_baseline=bool(args.include_random_baseline),
            )
            payload[split_name]['decode'] = metrics
            _print_metrics(split_name, metrics, metric_space='decode')

    if args.json_output:
        output_path = Path(args.json_output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        tool.cprint(f'Saved imputation metrics to {output_path}')

    env.close_env()


if __name__ == '__main__':
    main()
