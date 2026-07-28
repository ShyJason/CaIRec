import argparse
import ast
import sys
import json
from pathlib import Path
import time
import torch
import tool
from enviroment import Env
from dataset_loader import Loader4MM
from model import MILK_model
from session import MILK_session


def _parse_topk(value):
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise argparse.ArgumentTypeError(f'invalid topk list: {value!r}') from exc
    if not isinstance(value, (list, tuple)) or not value:
        raise argparse.ArgumentTypeError('topk must be a non-empty list of positive integers')
    parsed = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise argparse.ArgumentTypeError('topk must contain only positive integers')
        parsed.append(item)
    if parsed != sorted(set(parsed)):
        raise argparse.ArgumentTypeError('topk values must be unique and increasing')
    return parsed


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

    return normalized, str(path)


def _build_parser():
    parser = argparse.ArgumentParser(description="MILK")
    parser.add_argument('--config', type=str, default=None, help='Path to a YAML/JSON config file')
    parser.add_argument(
        '--check_config',
        action='store_true',
        help='Validate configuration and exit before creating data or output directories.',
    )

    # ----------------------- File Identification
    parser.add_argument('--suffix', type=str, default='default')

    # ----------------------- Device Setting
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--dataset_seed', type=int, default=0)
    parser.add_argument(
        '--unified_payload_seed',
        type=int,
        default=-1,
        help='Seed of the pre-generated unified_static missing payload; defaults to --seed.',
    )
    parser.add_argument(
        '--unified_payload_file',
        type=str,
        default='',
        help=(
            'Optional unified_static payload filename. Relative paths are resolved '
            'inside Data/<dataset>; when omitted, the seed-based default is used.'
        ),
    )
    parser.add_argument(
        '--missing_mask_protocol',
        type=str,
        default='i3',
        choices=['i3', 'default_rng', 'unified_static'],
        help='unified_static loads the pre-generated, phase-invariant missing-item payload',
    )
    parser.add_argument(
        '--train_missing_modality',
        type=str,
        default='random',
        choices=['random', 'image', 'text'],
        help=(
            'Training-only missing-modality policy. Validation/test keep the '
            'random-modality protocol controlled by eval_missing_rate.'
        ),
    )
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--eval_only', type=int, default=0)
    parser.add_argument(
        '--projection_ckpt',
        type=str,
        default=None,
        help='Pretrained modality projection checkpoint loaded before Stage 1.',
    )
    parser.add_argument('--imputer_ckpt', type=str, default=None)
    parser.add_argument('--ckpt_start_epoch', type=int, default=0)

    # ------------------------ Training Setting
    parser.add_argument('--free_emb_dimension', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_rec', type=float, default=None)
    parser.add_argument('--lr_imp', type=float, default=None)
    parser.add_argument('--lr_decoder', type=float, default=None)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--eva_interval', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--topk', type=_parse_topk, default=[10, 20, 30, 40, 50])
    parser.add_argument('--selection_mode', type=str, default='val', choices=['val', 'test'])
    parser.add_argument(
        '--recommendation_selection_metric',
        type=str,
        default='recall',
        choices=['recall', 'ndcg', 'hr'],
    )
    parser.add_argument('--recommendation_selection_topk', type=int, default=20)
    parser.add_argument('--evaluation_protocol', type=str, default='strict', choices=['legacy', 'strict'])
    parser.add_argument('--strict_record_test_each_epoch', type=int, default=0)
    parser.add_argument(
        '--report_test_modality_subsets',
        type=int,
        default=0,
        help=(
            'Report test metrics separately for positive items with all modalities '
            'observed and positive items selected by the test missing-modality mask. '
            'The ranking candidate set remains unchanged.'
        ),
    )
    parser.add_argument(
        '--strict_probe_test_interval',
        type=int,
        default=0,
        help='Strict protocol only: log test metrics every N epochs without using them for selection. 0 disables.',
    )
    parser.add_argument('--dataset', type=str, default='clothing')
    parser.add_argument('--exp_mode', type=str, default='fm')
    parser.add_argument(
        '--train_stage',
        type=str,
        default='imputer_backprop',
        choices=[
            'imputer_param',
            'imputer_backprop',
            'recommender',
        ],
    )
    parser.add_argument('--freeze_imputer', type=int, default=-1)
    parser.add_argument('--freeze_recommender', type=int, default=-1)
    parser.add_argument('--freeze_decoder', type=int, default=0)

    # ----------------------- Regularizer coefficient
    parser.add_argument('--reg_coeff', type=float, default=1e-4)
    parser.add_argument('--penalty_coeff', type=float, default=50) # b 1000  c 50
    parser.add_argument('--missing_rate', type=float, default=0.3)
    parser.add_argument(
        '--eval_missing_rate',
        type=float,
        default=0.5,
        help='Validation/test missing-modality rate. Defaults to the historical fixed 0.5 protocol.',
    )

    parser.add_argument('--contra_dim', type=int, default=256)
    parser.add_argument('--d_beta', type=int, default=128)
    parser.add_argument('--tau1', type=float, default=0.1)
    parser.add_argument('--tau2', type=float, default=0.1)
    parser.add_argument('--lambda_itm', type=float, default=0.1)
    parser.add_argument('--itm_temp', type=float, default=0.07)
    parser.add_argument('--itm_num_heads', type=int, default=4)
    parser.add_argument('--ema_eta', type=float, default=0.01)
    parser.add_argument('--disable_imputation', type=int, default=0)
    parser.add_argument(
        '--feature_bridge_mode',
        type=str,
        default='raw_decoder',
        choices=['raw_decoder', 'latent_direct', 'decoupled_latent'],
    )
    parser.add_argument('--item_graph_topk', type=int, default=20)
    parser.add_argument(
        '--item_graph_fuse_before_topk',
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            'Experimental completed-item graph construction order. When enabled for '
            'modality_completed, compute weighted CF+semantic scores over all items first '
            'and apply top-k only once; the default keeps the legacy per-source top-k '
            'followed by fusion and a final top-k.'
        ),
    )
    parser.add_argument(
        '--item_graph_missing_scope',
        type=str,
        default='train',
        choices=['all', 'train'],
        help=(
            'Missing-mask scope used only when building completed item graphs. '
            'train excludes validation/test missing masks from the training-time item graph; '
            'all preserves the old transductive behavior for reproduction only.'
        ),
    )
    parser.add_argument(
        '--item_graph_kind',
        type=str,
        default='none',
        choices=[
            'none',
            'modality_masked',
            'modality_completed',
        ],
        help=(
            'Item-item graph adapter kind. modality_masked builds per-modality CF plus semantic '
            'graphs directly from masked raw features for strict Stage2-only no-completion ablations; '
            'modality_completed propagates each modality '
            'embedding with its own CF plus completed-feature modality graph.'
        ),
    )
    parser.add_argument(
        '--item_graph_norm',
        type=str,
        default='rw',
        choices=['rw', 'sym', 'none'],
    )
    parser.add_argument('--item_graph_cf_weight', type=float, default=0.5)
    parser.add_argument(
        '--item_graph_cf_scale',
        type=str,
        default='raw',
        choices=['raw', 'sqrt', 'power', 'clip', 'cosine', 'log1p', 'rowmax', 'log1p_rowmax'],
        help=(
            'Scale for the CF item-item graph before fusing with semantic graphs. '
            'raw keeps co-occurrence counts; sqrt/power/log1p compress count tails; '
            'clip caps counts; cosine normalizes by item popularity; rowmax scales each row to at most 1.'
        ),
    )
    parser.add_argument(
        '--item_graph_cf_power',
        type=float,
        default=0.5,
        help='Power exponent used when item_graph_cf_scale=power.',
    )
    parser.add_argument(
        '--item_graph_cf_clip',
        type=float,
        default=3.0,
        help='Maximum co-occurrence count used when item_graph_cf_scale=clip.',
    )
    parser.add_argument('--item_graph_image_weight', type=float, default=0.25)
    parser.add_argument('--item_graph_text_weight', type=float, default=0.25)
    parser.add_argument(
        '--item_graph_feature_space',
        type=str,
        default='shared',
        choices=['shared', 'raw_decoder'],
        help=(
            'Feature space used to build completed modality item graphs. '
            'shared uses observed projected completion latents plus imputed completion latents; '
            'raw_decoder uses observed raw features plus decoded imputed raw features.'
        ),
    )
    parser.add_argument(
        '--item_graph_audio_weight',
        type=float,
        default=0.0,
        help='Audio graph weight. Single-source graph runs are ablations: set exactly one item_graph_*_weight positive.',
    )
    parser.add_argument(
        '--item_graph_video_weight',
        type=float,
        default=0.0,
        help='Video graph weight. Single-source graph runs are ablations: set exactly one item_graph_*_weight positive.',
    )
    parser.add_argument('--item_graph_feature_chunk_size', type=int, default=1024)
    parser.add_argument(
        '--item_graph_modal_alpha',
        type=float,
        default=0.0,
        help='Residual weight for post-GCN, pre-fusion item-item graph propagation on each modality item embedding.',
    )
    parser.add_argument(
        '--item_graph_modal_layers',
        type=int,
        default=1,
        help='Number of post-GCN, pre-fusion modality item-item residual propagation layers.',
    )
    parser.add_argument(
        '--item_graph_modal_target',
        type=str,
        default='all',
        choices=['all', 'missing'],
        help='Apply post-GCN, pre-fusion modality item-item residual to all items or only items missing that modality.',
    )
    parser.add_argument('--alpha_intra', type=float, default=1.0)
    parser.add_argument('--alpha_inter', type=float, default=1.0)
    parser.add_argument('--alpha_itm', type=float, default=1.0)
    parser.add_argument('--alpha_rec', type=float, default=0.1)
    parser.add_argument('--alpha_decode', type=float, default=0.0)
    parser.add_argument(
        '--generative_update_mode',
        type=str,
        default='em',
        choices=['em', 'fixed'],
        help=(
            'Whether Stage 1 applies queued closed-form EM updates to W/mu/log_sigma. '
            'Stage 1.2 (imputer_backprop) is always forced to fixed; EM is only '
            'available to stages that estimate generative parameters.'
        ),
    )
    parser.add_argument('--modality_bpr_coeff', type=float, default=0.2)
    parser.add_argument('--gamma_align', type=float, default=0.0)
    parser.add_argument(
        '--adapter_align_pseudo_ratio',
        type=float,
        default=1.0,
        help='Fraction of eligible observed modalities to pseudo-mask for decoupled adapter alignment.',
    )
    parser.add_argument('--recommender_allow_modal_grad', type=int, default=0)
    parser.add_argument('--imputation_val_rate', type=float, default=0.0)
    parser.add_argument(
        '--imputation_selection_policy',
        type=str,
        default='legacy',
        choices=['legacy', 'stage1_default', 'decoder_default'],
    )
    parser.add_argument('--imputation_selection_split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--imputation_selection_metric', type=str, default='mse', choices=['mse', 'cosine'])
    # ----------------------- logger
    parser.add_argument('--log', type=int, default=0)
    parser.add_argument('--tensorboard', type=int, default=1)
    parser.add_argument('--hf_tensorboard_repo', type=str, default='')
    parser.add_argument('--hf_commit_every', type=int, default=5)
    parser.add_argument('--save', type=int, default=1)
    parser.add_argument('--save_all_epochs', type=int, default=0)
    return parser


def _apply_stage1_defaults(args):
    # Stage 1.2 optimizes the completion projection against the generative
    # model estimated in Stage 1.1. Keep that model fixed even if a stale
    # launcher explicitly requests EM.
    if args.train_stage == 'imputer_backprop':
        args.generative_update_mode = 'fixed'
    return args


def _validate_protocol_args(args):
    if args.evaluation_protocol == 'strict':
        if args.selection_mode == 'test':
            raise ValueError('Strict evaluation cannot use selection_mode=test; use validation selection.')
        if args.imputation_selection_split == 'test':
            raise ValueError('Strict evaluation cannot use imputation_selection_split=test; use train or val.')
    return args


def parse_args():
    parser = _build_parser()
    pre_args, _ = parser.parse_known_args()

    if pre_args.config:
        config_data, config_path = _load_config_file(pre_args.config)
        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(config_data.keys()) - valid_keys)
        if unknown_keys:
            raise ValueError(f'Unknown config keys in {config_path}: {unknown_keys}')
        parser.set_defaults(**config_data)

    args = parser.parse_args()
    args.topk = _parse_topk(args.topk)
    if args.config is not None:
        args.config = str(Path(args.config).expanduser().resolve())
    return _validate_protocol_args(_apply_stage1_defaults(args))



# ----------------------------------- Env Init -----------------------------------------------------------
tool.cprint('Init Env')
args = parse_args()
if args.check_config:
    print(f"configuration valid: {args.config or '<command line>'}")
    sys.exit(0)
# print(vars(args))
# exit()
my_env = Env(args)
tool.cprint(f'---------- {my_env.args.suffix} ----------')
print(my_env.format_public_args())

# ----------------------------------- Dataset Init -----------------------------------------------------------

my_loader = Loader4MM(my_env)

tool.cprint('Init Dataset')

# ----------------------------------- Model Init -----------------------------------------------------------

my_model = MILK_model(my_env, my_loader)
if args.ckpt is not None:
    loaded_count = my_model.load_full_checkpoint(args.ckpt)
    tool.cprint(f'Loaded full checkpoint from {args.ckpt} ({loaded_count} tensors)')
if args.projection_ckpt is not None:
    matched_keys = my_model.load_projection_checkpoint(args.projection_ckpt)
    tool.cprint(f'Loaded pretrained projection from {args.projection_ckpt} ({len(matched_keys)} tensors)')
if args.imputer_ckpt is not None:
    matched_keys = my_model.load_imputer_checkpoint(args.imputer_ckpt)
    tool.cprint(f'Loaded imputer checkpoint from {args.imputer_ckpt} ({len(matched_keys)} tensors)')
if getattr(args, 'item_graph_kind', None) in (
    'modality_masked',
    'modality_completed',
):
    my_model.build_completed_item_graph()
tool.cprint('Init Model')

# ----------------------------------- Session Init -----------------------------------------------------------

my_session = MILK_session(my_env, my_model, my_loader)
tool.cprint('Init Session')

if bool(args.eval_only):
    t = time.time()
    hr, recall, ndcg, test_time = my_session.test(
        mode='test', top_list=args.topk
    )
    for top_k in args.topk:
        message = (
            f'eval-only test hr@{top_k} = {hr[top_k]:.5f}, '
            f'recall@{top_k} = {recall[top_k]:.5f}, '
            f'ndcg@{top_k} = {ndcg[top_k]:.5f}, test_time = {test_time:.2f}'
        )
        tool.cprint(message)
        if args.log:
            my_env.test_logger.info(message)
    my_session.log_test_modality_subsets(prefix='eval-only')
    tool.cprint(f'eval-only stage cost time: {time.time() - t}')
    my_env.close_env()
    sys.exit(0)

# ---------------------------------------- Main -----------------------------------------------------------

t = time.time()
my_session.train(my_env.args.epoch)
# my_session.save_memory()
my_env.close_env()
tool.cprint(f'training stage cost time: {time.time() - t}')
if my_env.args.log:
    my_env.test_logger.info(f'--------- {my_env.args.suffix} best epoch {my_session.best_epoch}------------')

tool.cprint(f'--------- {my_env.args.suffix} best epoch {my_session.best_epoch}------------')
if my_env.args.train_stage in (
    'imputer_param',
    'imputer_backprop',
):
    final_metrics = my_session.last_train_metrics
    if final_metrics:
        summary = (
            f"final stage1 metrics: epoch = {final_metrics['epoch']}, "
            f"loss_s1 = {final_metrics['loss_s1']:.5f}, "
            f"promrl_rec = {final_metrics['promrl_rec']:.5f}, "
            f"promrl_decode = {final_metrics['promrl_decode']:.5f}, "
            f"promrl_decode_kl = {final_metrics.get('promrl_decode_kl', 0.0):.5f}, "
            f"promrl_intra = {final_metrics['promrl_intra']:.5f}, "
            f"promrl_inter = {final_metrics['promrl_inter']:.5f}, "
            f"promrl_itm = {final_metrics['promrl_itm']:.5f}"
        )
        if 'imputation_train_mse' in final_metrics:
            summary += (
                f", imputation_train_mse = {final_metrics['imputation_train_mse']:.6f}, "
                f"imputation_train_cosine = {final_metrics['imputation_train_cosine']:.6f}, "
                f"imputation_test_mse = {final_metrics.get('imputation_test_mse', 0.0):.6f}, "
                f"imputation_test_cosine = {final_metrics.get('imputation_test_cosine', 0.0):.6f}"
            )
        if 'imputation_val_mse' in final_metrics:
            summary += (
                f", imputation_val_mse = {final_metrics['imputation_val_mse']:.6f}, "
                f"imputation_val_cosine = {final_metrics['imputation_val_cosine']:.6f}"
            )
        if 'val_shared_cosine_gap' in final_metrics:
            summary += (
                f", val_shared_cosine_gap = {final_metrics['val_shared_cosine_gap']:.6f}, "
                f"val_missing_decode_cosine = {final_metrics.get('val_missing_decode_cosine', 0.0):.6f}, "
                f"val_shared_mse = {final_metrics.get('val_shared_mse', 0.0):.6f}"
            )
        tool.cprint(summary)
        if my_env.args.log:
            my_env.test_logger.info(summary)
else:
    for top_k in args.topk:
        tool.cprint(f'hr@{top_k} = {my_session.test_hr[top_k]:.5f}, recall@{top_k} = {my_session.test_recall[top_k]:.5f}, ndcg@{top_k} = {my_session.test_ndcg[top_k]:.5f}')
        if my_env.args.log:
            my_env.test_logger.info(f'hr@{top_k} = {my_session.test_hr[top_k]:.5f}, recall@{top_k} = {my_session.test_recall[top_k]:.5f}, ndcg@{top_k} = {my_session.test_ndcg[top_k]:.5f}')
