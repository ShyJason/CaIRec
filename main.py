import argparse
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
# from invRL_session import InvRL


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

    return normalized, str(path)


def _build_parser():
    parser = argparse.ArgumentParser(description="MILK")
    parser.add_argument('--config', type=str, default=None, help='Path to a YAML/JSON config file')

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
    parser.add_argument('--neg_num', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--topk', type=str, default='[10, 20, 30, 40, 50]')
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
    parser.add_argument(
        '--cold_start_protocol',
        type=str,
        default='none',
        choices=['none', 'milk'],
        help='milk loads a pre-generated 80/10/10 item-disjoint split.',
    )
    parser.add_argument('--cold_start_seed', type=int, default=2023)
    parser.add_argument('--cold_start_missing_seed', type=int, default=2023)
    parser.add_argument(
        '--cold_start_eval_candidates',
        type=str,
        default='milk_union',
        choices=['milk_union', 'split'],
        help='Use the official MILK cold union or split-specific cold candidates.',
    )
    parser.add_argument(
        '--cold_start_data_dir',
        type=str,
        default='',
        help='Optional split directory. Relative paths are resolved below Data/<dataset>.',
    )
    parser.add_argument('--exp_mode', type=str, default='fm')
    parser.add_argument('--model', type=str, default='MILK')
    parser.add_argument(
        '--train_stage',
        type=str,
        default='imputer_backprop',
        choices=[
            'imputer_param',
            'imputer_backprop',
            'projection_pretrain',
            'recommender',
            'joint',
        ],
    )
    parser.add_argument('--freeze_imputer', type=int, default=-1)
    parser.add_argument('--freeze_recommender', type=int, default=-1)
    parser.add_argument('--freeze_decoder', type=int, default=0)

    # ----------------------- Regularizer coefficient
    parser.add_argument('--reg_coeff', type=float, default=1e-4)
    parser.add_argument('--penalty_coeff', type=float, default=50) # b 1000  c 50
    parser.add_argument('--max_info_coeff', type=float, default=0.05) # b 0.1   c 0.05
    parser.add_argument('--min_info_coeff', type=float, default=0.05) # b 0.1   c 0.05

    parser.add_argument('--missing_rate', type=float, default=0.3)
    parser.add_argument(
        '--eval_missing_rate',
        type=float,
        default=0.5,
        help='Validation/test missing-modality rate. Defaults to the historical fixed 0.5 protocol.',
    )

    parser.add_argument('--alpha', type=float, default=0.1)
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
    parser.add_argument(
        '--enable_raw_completion_decoder',
        type=int,
        default=0,
        help=(
            'Attach a decoder from the completion space back to each raw modality '
            'space without changing feature_bridge_mode. This is intended for '
            'decoder-only training on a frozen decoupled-latent checkpoint.'
        ),
    )
    parser.add_argument(
        '--decoder_loss_mode',
        type=str,
        default='observed_projection',
        choices=['observed_projection', 'pseudo_missing'],
        help=(
            'observed_projection reconstructs raw features from their own projected '
            'representations; pseudo_missing reconstructs a held-out observed modality '
            'from the actual completion produced using the remaining modalities.'
        ),
    )
    parser.add_argument(
        '--decoder_pseudo_missing_ratio',
        type=float,
        default=1.0,
        help='Fraction of eligible batch items used for pseudo-missing decoder supervision.',
    )
    parser.add_argument(
        '--decoder_output_mode',
        type=str,
        default='normalized',
        choices=['normalized', 'native_direction_norm'],
        help=(
            'normalized preserves the historical unit-vector decoder; '
            'native_direction_norm predicts raw direction and item-specific magnitude '
            'with separate heads.'
        ),
    )
    parser.add_argument('--decoder_raw_loss_weight', type=float, default=1.0)
    parser.add_argument('--decoder_cosine_loss_weight', type=float, default=1.0)
    parser.add_argument('--decoder_norm_loss_weight', type=float, default=0.25)
    parser.add_argument('--decoder_relation_loss_weight', type=float, default=0.1)
    parser.add_argument('--decoder_relation_max_items', type=int, default=64)
    parser.add_argument(
        '--gcn_frontend_mode',
        type=str,
        default='original_linear',
        choices=['original_linear', 'deep_mlp', 'identity'],
    )
    parser.add_argument(
        '--promrl_projection_mode',
        type=str,
        default='learned',
        choices=['learned', 'identity'],
        help='Projection before completion/imputation. identity uses normalized modal features directly and requires feature_dim == promrl_dim.',
    )
    parser.add_argument(
        '--modal_feature_override_dir',
        type=str,
        default='',
        help='Optional directory with external image/text item features. If it contains phase_train/phase_eval, train/eval features are loaded separately.',
    )
    parser.add_argument(
        '--modal_feature_train_dir',
        type=str,
        default='',
        help='Optional explicit train-phase external feature directory.',
    )
    parser.add_argument(
        '--modal_feature_eval_dir',
        type=str,
        default='',
        help='Optional explicit eval/test-phase external feature directory.',
    )
    parser.add_argument(
        '--modal_feature_image_file',
        type=str,
        default='agg_image_items.npy',
        help='Image feature filename used with modal_feature_override_dir.',
    )
    parser.add_argument(
        '--modal_feature_text_file',
        type=str,
        default='agg_text_items.npy',
        help='Text feature filename used with modal_feature_override_dir.',
    )
    parser.add_argument(
        '--modal_feature_audio_file',
        type=str,
        default='agg_audio_items.npy',
        help='Optional audio feature filename used with modal_feature_override_dir when present.',
    )
    parser.add_argument(
        '--modal_feature_video_file',
        type=str,
        default='agg_video_items.npy',
        help='Optional video feature filename used with modal_feature_override_dir when present.',
    )
    parser.add_argument(
        '--modal_feature_mask_source',
        type=str,
        default='nonzero',
        choices=['nonzero', 'external_observed'],
        help=(
            'How to derive observed modality masks when using modal_feature_override_dir. '
            'external_observed reads exported observed-mask files so completed nonzero rows '
            'are still treated as originally missing.'
        ),
    )
    parser.add_argument(
        '--modal_feature_image_mask_file',
        type=str,
        default='image_observed_mask.npy',
        help='Image observed-mask filename used when modal_feature_mask_source=external_observed.',
    )
    parser.add_argument(
        '--modal_feature_text_mask_file',
        type=str,
        default='text_observed_mask.npy',
        help='Text observed-mask filename used when modal_feature_mask_source=external_observed.',
    )
    parser.add_argument(
        '--modal_feature_audio_mask_file',
        type=str,
        default='audio_observed_mask.npy',
        help='Audio observed-mask filename used when modal_feature_mask_source=external_observed.',
    )
    parser.add_argument(
        '--modal_feature_video_mask_file',
        type=str,
        default='video_observed_mask.npy',
        help='Video observed-mask filename used when modal_feature_mask_source=external_observed.',
    )
    parser.add_argument(
        '--modal_feature_override_is_completed',
        type=int,
        default=0,
        help='Set to 1 when external override files already contain completed features for missing modalities.',
    )
    parser.add_argument(
        '--completion_gate_mode',
        type=str,
        default='off',
        choices=[
            'off',
            'reliability',
            'alignment',
            'rank_residual',
            'rank_residual_norm',
            'rank_residual_allnorm',
            'rank_residual_allgate',
            'rank_residual_softmax',
            'rank_residual_global',
            'rank_residual_shrink',
            'rank_residual_centered',
            'rank_residual_delta',
            'rank_residual_centered_allgate',
        ],
    )
    parser.add_argument('--completion_gate_hidden_dim', type=int, default=64)
    parser.add_argument('--completion_gate_dropout', type=float, default=0.1)
    parser.add_argument('--completion_gate_init_logit', type=float, default=1.5)
    parser.add_argument('--completion_gate_detach_inputs', type=int, default=1)
    parser.add_argument('--completion_gate_use_item_context', type=int, default=1)
    parser.add_argument(
        '--completion_gate_item_context_source',
        type=str,
        default='id_embedding',
        choices=['id_embedding', 'shared_mean', 'off'],
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
            'fused_completed',
            'modality_completed',
            'modality_completed_inductive',
            'modality_completed_confidence',
            'modality_completed_dynamic_confidence',
            'fused_completed_confidence',
            'fused_completed_dynamic_confidence',
            'fused_completed_reliability',
            'fused_completed_reliability_topk',
        ],
        help=(
            'Item-item graph adapter kind. modality_masked builds per-modality CF plus semantic '
            'graphs directly from masked raw features for strict Stage2-only no-completion ablations; '
            'fused_completed fuses CF plus completed-feature '
            'modality graphs before propagation; modality_completed propagates each modality '
            'embedding with its own CF plus completed-feature modality graph; '
            'modality_completed_inductive restricts semantic references to warm items and '
            'attaches cold query rows only during evaluation; '
            'modality_completed_confidence adds learnable edge confidence (rr/ri/ii) for each modality completed graph; '
            'modality_completed_dynamic_confidence recomputes per-modality topk from the learned confidences during propagation; '
            'fused_completed_confidence learns edge confidence for real-real, real-imputed, '
            'and imputed-imputed completed-feature edges after a fixed initial topk; '
            'fused_completed_dynamic_confidence recomputes topk from the learned confidences during propagation; '
            'fused_completed_reliability downweights completed-feature graph edges using per-item completion reliability; '
            'fused_completed_reliability_topk applies the same reliability before top-k neighbor selection.'
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
    parser.add_argument('--item_graph_rr_confidence_init', type=float, default=1.0)
    parser.add_argument('--item_graph_ri_confidence_init', type=float, default=1.0)
    parser.add_argument('--item_graph_ii_confidence_init', type=float, default=1.0)
    parser.add_argument(
        '--item_graph_modality_specific_confidence',
        type=int,
        default=0,
        help='Use separate rr/ri/ii confidence parameters for each modality graph when item_graph_kind is modality_completed*_confidence.',
    )
    parser.add_argument(
        '--item_graph_confidence_blend',
        type=float,
        default=1.0,
        help='Blend strength for learned edge confidences. 0 keeps the unweighted fused graph; 1 fully applies learned confidences.',
    )
    parser.add_argument(
        '--item_graph_dynamic_score_blend',
        type=float,
        default=1.0,
        help='For dynamic confidence item graphs, blend raw candidate scores with learned-confidence scores before top-k. 1 fully uses learned scores.',
    )
    parser.add_argument(
        '--item_graph_dynamic_score_blend_start',
        type=float,
        default=-1.0,
        help='Optional starting score blend for dynamic confidence graphs. Negative keeps item_graph_dynamic_score_blend from epoch 0.',
    )
    parser.add_argument(
        '--item_graph_dynamic_score_blend_warmup_epochs',
        type=int,
        default=0,
        help='Linearly warm score blend from item_graph_dynamic_score_blend_start to item_graph_dynamic_score_blend over this many epochs.',
    )
    parser.add_argument(
        '--item_graph_confidence_log_interval',
        type=int,
        default=0,
        help='Print learned item graph edge confidences every N epochs when confidence item graphs are enabled. 0 disables periodic logging.',
    )
    parser.add_argument(
        '--item_graph_confidence_reg_coeff',
        type=float,
        default=0.0,
        help='L2 penalty coefficient that keeps active learned item graph edge confidences near item_graph_confidence_reg_target.',
    )
    parser.add_argument(
        '--item_graph_confidence_reg_target',
        type=float,
        default=1.0,
        help='Target edge confidence used by item_graph_confidence_reg_coeff.',
    )
    parser.add_argument('--item_graph_rr_confidence_reg_target', type=float, default=None)
    parser.add_argument('--item_graph_ri_confidence_reg_target', type=float, default=None)
    parser.add_argument('--item_graph_ii_confidence_reg_target', type=float, default=None)
    parser.add_argument(
        '--item_graph_confidence_reg_start_epoch',
        type=int,
        default=0,
        help='Epoch at which item graph edge confidence regularization starts.',
    )
    parser.add_argument(
        '--item_graph_dynamic_neighbor_blend',
        type=float,
        default=1.0,
        help='For dynamic confidence graphs, blend fixed-base neighbor propagation with dynamic learned neighbor propagation. 1 fully uses dynamic neighbors.',
    )
    parser.add_argument(
        '--item_graph_dynamic_neighbor_blend_start',
        type=float,
        default=-1.0,
        help='Optional starting neighbor blend for dynamic confidence graphs. Negative keeps item_graph_dynamic_neighbor_blend from epoch 0.',
    )
    parser.add_argument(
        '--item_graph_dynamic_neighbor_blend_warmup_epochs',
        type=int,
        default=0,
        help='Linearly warm neighbor blend from item_graph_dynamic_neighbor_blend_start to item_graph_dynamic_neighbor_blend over this many epochs.',
    )
    parser.add_argument(
        '--item_graph_confidence_transform',
        type=str,
        default='blend',
        choices=['blend', 'sigmoid'],
        help='blend uses 1 + blend * (confidence - 1); sigmoid directly uses a sigmoid-bounded learned edge weight.',
    )
    parser.add_argument('--item_graph_confidence_min', type=float, default=0.25)
    parser.add_argument('--item_graph_confidence_max', type=float, default=4.0)
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
        '--item_graph_reliability_floor',
        type=float,
        default=0.4,
        help='Minimum per-item reliability for missing completed modality features in fused_completed_reliability.',
    )
    parser.add_argument(
        '--item_graph_reliability_blend',
        type=float,
        default=1.0,
        help='Blend strength for reliability edge reweighting. 0 keeps original completed graphs; 1 fully applies q_i*q_j.',
    )
    parser.add_argument(
        '--item_graph_reliability_missing_penalty',
        type=float,
        default=1.0,
        help='Extra multiplier for missing-item reliability before clamping to [floor, 1].',
    )
    parser.add_argument(
        '--item_graph_reliability_missing_boost',
        type=float,
        default=0.0,
        help='Optional boost for high-consistency missing completed modality features; 0 keeps reliability in [floor, 1].',
    )
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
    parser.add_argument('--completion_gate_floor', type=float, default=0.7)
    parser.add_argument('--completion_gate_target_mean', type=float, default=0.95)
    parser.add_argument('--completion_gate_reg_coeff', type=float, default=0.0)
    parser.add_argument('--completion_gate_alignment_center', type=float, default=0.0)
    parser.add_argument('--completion_gate_alignment_temp', type=float, default=0.2)
    parser.add_argument('--completion_gate_residual_alpha', type=float, default=0.1)
    parser.add_argument(
        '--completion_gate_no_residual_alpha',
        type=int,
        default=0,
        help='If set, rank-residual gates use 1 + tanh(z) and ignore completion_gate_residual_alpha.',
    )
    parser.add_argument('--completion_gate_mix_alpha', type=float, default=0.3)
    parser.add_argument('--completion_gate_identity_coeff', type=float, default=0.05)
    parser.add_argument('--completion_gate_balance_coeff', type=float, default=0.01)
    parser.add_argument('--completion_gate_softmax_temp', type=float, default=1.0)
    parser.add_argument('--completion_gate_advantage_coeff', type=float, default=0.0)
    parser.add_argument('--completion_gate_advantage_margin', type=float, default=0.0)
    parser.add_argument('--completion_gate_score_residual_alpha', type=float, default=0.0)
    parser.add_argument('--completion_gate_learn_mix', type=int, default=0)
    parser.add_argument('--completion_gate_mix_max', type=float, default=1.0)
    parser.add_argument('--completion_gate_shrink_init_logit', type=float, default=-4.0)
    parser.add_argument('--completion_gate_tail_quantile', type=float, default=1.0)
    parser.add_argument('--completion_gate_only_train', type=int, default=0)
    parser.add_argument(
        '--fusion_mode',
        type=str,
        default='mean',
        choices=['mean', 'rum', 'global_weighted_mean', 'posterior_reliability'],
        help=(
            'Item modality fusion. posterior_reliability uses the linear-Gaussian '
            'completion posterior predictive variance in the components selected '
            'by posterior_reliability_scope.'
        ),
    )
    parser.add_argument(
        '--posterior_reliability_scope',
        type=str,
        default='both',
        choices=['both', 'graph', 'fusion'],
        help='Apply posterior reliability to semantic graph top-k, modality fusion, or both.',
    )
    parser.add_argument(
        '--posterior_reliability_scale',
        type=float,
        default=1.0,
        help='Lambda in c=exp(-lambda * mean posterior predictive variance).',
    )
    parser.add_argument(
        '--posterior_reliability_floor',
        type=float,
        default=0.0,
        help='Optional lower bound for missing-modality posterior reliability.',
    )
    parser.add_argument('--rum_tau', type=float, default=1.0)
    parser.add_argument('--rum_reliability_coeff', type=float, default=1.0)
    parser.add_argument('--rum_match_coeff', type=float, default=1.0)
    parser.add_argument('--rum_eval_user_batch_size', type=int, default=256)
    parser.add_argument('--rum_eval_item_chunk_size', type=int, default=4096)
    parser.add_argument('--alpha_intra', type=float, default=1.0)
    parser.add_argument('--alpha_inter', type=float, default=1.0)
    parser.add_argument('--alpha_itm', type=float, default=1.0)
    parser.add_argument('--alpha_rec', type=float, default=0.1)
    parser.add_argument('--alpha_decode', type=float, default=0.0)
    parser.add_argument(
        '--structure_loss_variant',
        type=str,
        default='original',
        choices=['original', 'shifted_lifted'],
        help=(
            'Structure loss used by Stage 1 contrastive training. shifted_lifted '
            'uses a shifted-cosine modality relation matrix and compares squared '
            'lifted semantic principal-direction similarities.'
        ),
    )
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
    parser.add_argument('--beta_intra', type=float, default=0.05)
    parser.add_argument('--beta_inter', type=float, default=0.05)
    parser.add_argument('--beta_itm', type=float, default=0.05)
    parser.add_argument('--beta_rec', type=float, default=0.01)
    parser.add_argument('--beta_decode', type=float, default=0.01)
    parser.add_argument(
        '--joint_completion_batch_size',
        type=int,
        default=0,
        help=(
            'Joint stage only: use an independent uniformly shuffled item batch of this '
            'size for completion losses. 0 preserves the legacy behavior that reuses '
            'unique positive/negative items from the interaction batch.'
        ),
    )
    parser.add_argument(
        '--joint_grad_audit',
        type=int,
        default=0,
        help='Joint stage only: print and validate first-batch imputer/recommender gradient norms.',
    )
    parser.add_argument(
        '--joint_item_graph_refresh_interval',
        type=int,
        default=0,
        help=(
            'Joint stage only: rebuild completed-feature item graphs every N epochs '
            'from the current completion module. 0 keeps the initialization-time graph.'
        ),
    )
    parser.add_argument(
        '--joint_log_sigma_min',
        type=float,
        default=-100.0,
        help='Joint stage only: lower bound applied to generative log_sigma after each update.',
    )
    parser.add_argument('--gamma_align', type=float, default=0.0)
    parser.add_argument(
        '--adapter_align_pseudo_ratio',
        type=float,
        default=1.0,
        help='Fraction of eligible observed modalities to pseudo-mask for decoupled adapter alignment.',
    )
    parser.add_argument(
        '--completion_adapter_mode',
        type=str,
        default='linear_ln',
        choices=['linear_ln', 'identity', 'residual_mlp'],
        help='Adapter from completion latent space to recommendation space in decoupled_latent mode.',
    )
    parser.add_argument(
        '--completion_adapter_hidden_dim',
        type=int,
        default=128,
        help='Hidden dimension for residual_mlp completion adapter.',
    )
    parser.add_argument(
        '--completion_adapter_dropout',
        type=float,
        default=0.0,
        help='Dropout for residual_mlp completion adapter.',
    )
    parser.add_argument('--gamma_distill', type=float, default=0.0)
    parser.add_argument('--joint_allow_modal_grad', type=int, default=0)
    parser.add_argument('--recommender_allow_modal_grad', type=int, default=0)
    parser.add_argument(
        '--rec_neighbor_cl_weight',
        type=float,
        default=0.0,
        help='Stage2 true-missing GCN InfoNCE loss weight.',
    )
    parser.add_argument(
        '--rec_neighbor_cl_temp',
        type=float,
        default=0.2,
        help='Temperature for stage2 true-missing GCN InfoNCE.',
    )
    parser.add_argument(
        '--rec_neighbor_cl_bank_size',
        type=int,
        default=256,
        help='Max number of in-batch items used for stage2 true-missing GCN InfoNCE.',
    )
    parser.add_argument(
        '--rec_neighbor_cl_stage',
        type=str,
        default='gcn',
        choices=['frontend', 'gcn', 'post_item_graph'],
        help=(
            'Representation stage used by true-missing CL. frontend applies CL before '
            'user-item and item-item propagation; gcn applies it after user-item GCN but '
            'before the II residual; post_item_graph matches final modality propagation.'
        ),
    )
    parser.add_argument(
        '--rec_neighbor_cl_anchor_weighting',
        type=str,
        default='uniform',
        choices=['uniform', 'posterior_reliability'],
        help='Optionally weight true-missing CL anchors by posterior completion reliability.',
    )
    parser.add_argument(
        '--rec_neighbor_cl_false_negative_threshold',
        type=float,
        default=1.1,
        help='Ignore negatives whose cosine similarity to the positive teacher is at least this value; >1 disables filtering.',
    )
    parser.add_argument(
        '--rec_neighbor_cl_objective',
        type=str,
        default='infonce',
        choices=['infonce', 'positive_cosine'],
        help='Use standard in-batch InfoNCE or a positive-only cosine consistency objective.',
    )
    parser.add_argument(
        '--rec_neighbor_cl_positive_source',
        type=str,
        default='cross_modal',
        choices=['cross_modal', 'cf_neighbor'],
        help=(
            'Positive teacher for true-missing recommendation CL. cross_modal uses '
            "the same item's observed modalities; cf_neighbor aggregates observed "
            'same-modality representations from the training CF item graph.'
        ),
    )
    parser.add_argument(
        '--rec_neighbor_cl_negative_source',
        type=str,
        default='same_modal',
        choices=['same_modal', 'cross_modal'],
        help=(
            'Key bank used by InfoNCE. same_modal preserves the original mixed-space '
            'objective; cross_modal draws negatives from the same observed other-modal '
            'space as the positive and therefore implements cross-modal retrieval CL.'
        ),
    )
    parser.add_argument(
        '--rec_neighbor_cl_similarity_space',
        type=str,
        default='embedding',
        choices=['embedding', 'user_preference'],
        help=(
            'Similarity space for recommendation InfoNCE. user_preference compares '
            'same-item modalities through detached user-score profiles and adds no '
            'CL-specific trainable parameters.'
        ),
    )
    parser.add_argument(
        '--rec_neighbor_cl_user_bank_size',
        type=int,
        default=256,
        help='Maximum number of detached batch-user embeddings in a CL preference profile.',
    )
    parser.add_argument(
        '--rec_neighbor_cl_start_epoch',
        type=int,
        default=0,
        help='First epoch to apply stage2 true-missing GCN InfoNCE. Earlier epochs use weight 0.',
    )
    parser.add_argument(
        '--rec_neighbor_cl_end_epoch',
        type=int,
        default=-1,
        help='Last epoch for stage2 true-missing GCN InfoNCE schedule. Negative keeps the base weight forever.',
    )
    parser.add_argument(
        '--rec_neighbor_cl_decay_start_epoch',
        type=int,
        default=-1,
        help='Epoch to start decaying stage2 true-missing GCN InfoNCE. Negative uses rec_neighbor_cl_start_epoch.',
    )
    parser.add_argument(
        '--rec_neighbor_cl_final_weight',
        type=float,
        default=-1.0,
        help='Final CL weight at rec_neighbor_cl_end_epoch. Negative keeps the base weight.',
    )
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
    parser.add_argument('--hf_token', type=str, default='')
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
    if args.cold_start_protocol == 'milk':
        if args.exp_mode != 'mm':
            raise ValueError('This MILK cold-start implementation is restricted to exp_mode=mm.')
        if args.evaluation_protocol != 'strict':
            raise ValueError('MILK cold-start requires evaluation_protocol=strict.')
        if getattr(args, 'item_graph_kind', 'none') not in ('none', 'modality_completed_inductive'):
            raise ValueError(
                'MILK cold-start requires item_graph_kind=none or modality_completed_inductive; '
                'the other semantic item graphs expose cold features during training.'
            )
        if getattr(args, 'item_graph_kind', 'none') == 'modality_completed_inductive':
            if args.train_stage != 'recommender':
                raise ValueError('modality_completed_inductive is only supported in cold-start Stage2 recommender training.')
            if int(getattr(args, 'freeze_imputer', -1)) == 0:
                raise ValueError('modality_completed_inductive requires a frozen Stage1 imputer.')
        if getattr(args, 'missing_mask_protocol', 'i3') == 'unified_static':
            raise ValueError(
                'MILK cold-start does not support unified_static masks because the shared payload can '
                'place cold items in the training missing-modality set.'
            )
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
    if args.config is not None:
        args.config = str(Path(args.config).expanduser().resolve())
    return _validate_protocol_args(_apply_stage1_defaults(args))



# ----------------------------------- Env Init -----------------------------------------------------------
tool.cprint('Init Env')
args = parse_args()
# print(vars(args))
# exit()
my_env = Env(args)
tool.cprint(f'---------- {my_env.args.suffix} ----------')
print(f'{my_env.args}')

# ----------------------------------- Dataset Init -----------------------------------------------------------

my_loader = Loader4MM(my_env)

tool.cprint('Init Dataset')

# ----------------------------------- Model Init -----------------------------------------------------------

my_model = MILK_model(my_env, my_loader)
if args.ckpt is not None:
    loaded_count = my_model.load_full_checkpoint(args.ckpt)
    tool.cprint(f'Loaded full checkpoint from {args.ckpt} ({loaded_count} tensors)')
if args.imputer_ckpt is not None:
    matched_keys = my_model.load_imputer_checkpoint(args.imputer_ckpt)
    tool.cprint(f'Loaded imputer checkpoint from {args.imputer_ckpt} ({len(matched_keys)} tensors)')
if getattr(args, 'item_graph_kind', None) in (
    'modality_masked',
    'fused_completed',
    'modality_completed',
    'modality_completed_inductive',
    'modality_completed_confidence',
    'modality_completed_dynamic_confidence',
    'fused_completed_confidence',
    'fused_completed_dynamic_confidence',
    'fused_completed_reliability',
    'fused_completed_reliability_topk',
):
    my_model.build_completed_item_graph()
tool.cprint('Init Model')

# ----------------------------------- Session Init -----------------------------------------------------------

my_session = MILK_session(my_env, my_model, my_loader)
tool.cprint('Init Session')

if bool(args.eval_only):
    t = time.time()
    hr, recall, ndcg, test_time = my_session.test(
        mode='test', top_list=eval(args.topk)
    )
    for top_k in eval(args.topk):
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
if getattr(my_model, 'use_item_graph_edge_confidence', False):
    with torch.no_grad():
        edge_conf = my_model._item_graph_edge_confidences()
        edge_coeff = my_model._item_graph_edge_confidence_coeffs()
    if isinstance(edge_conf, dict):
        for modality in sorted(edge_conf):
            conf = edge_conf[modality].detach().cpu().tolist()
            coeff = edge_coeff[modality].detach().cpu().tolist()
            edge_conf_msg = (
                f"learned edge confidence[{modality}]: rr={conf[0]:.6f}, "
                f"ri={conf[1]:.6f}, ii={conf[2]:.6f}"
            )
            edge_coeff_msg = (
                f"learned edge coeff[{modality}]: rr={coeff[0]:.6f}, "
                f"ri={coeff[1]:.6f}, ii={coeff[2]:.6f}"
            )
            tool.cprint(edge_conf_msg)
            tool.cprint(edge_coeff_msg)
            if my_env.args.log:
                my_env.test_logger.info(edge_conf_msg)
                my_env.test_logger.info(edge_coeff_msg)
    else:
        edge_conf = edge_conf.detach().cpu().tolist()
        edge_coeff = edge_coeff.detach().cpu().tolist()
        edge_conf_msg = (
            f"learned edge confidence: rr={edge_conf[0]:.6f}, "
            f"ri={edge_conf[1]:.6f}, ii={edge_conf[2]:.6f}"
        )
        edge_coeff_msg = (
            f"learned edge coeff: rr={edge_coeff[0]:.6f}, "
            f"ri={edge_coeff[1]:.6f}, ii={edge_coeff[2]:.6f}"
        )
        tool.cprint(edge_conf_msg)
        tool.cprint(edge_coeff_msg)
        if my_env.args.log:
            my_env.test_logger.info(edge_conf_msg)
            my_env.test_logger.info(edge_coeff_msg)
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
    for top_k in eval(args.topk):
        tool.cprint(f'hr@{top_k} = {my_session.test_hr[top_k]:.5f}, recall@{top_k} = {my_session.test_recall[top_k]:.5f}, ndcg@{top_k} = {my_session.test_ndcg[top_k]:.5f}')
        if my_env.args.log:
            my_env.test_logger.info(f'hr@{top_k} = {my_session.test_hr[top_k]:.5f}, recall@{top_k} = {my_session.test_recall[top_k]:.5f}, ndcg@{top_k} = {my_session.test_ndcg[top_k]:.5f}')
