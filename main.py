import argparse
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
    parser.add_argument('--missing_mask_protocol', type=str, default='i3', choices=['i3', 'default_rng'])
    parser.add_argument('--ckpt', type=str, default=None)
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
        '--strict_probe_test_interval',
        type=int,
        default=0,
        help='Strict protocol only: log test metrics every N epochs without using them for selection. 0 disables.',
    )
    parser.add_argument('--dataset', type=str, default='clothing')
    parser.add_argument('--exp_mode', type=str, default='fm')
    parser.add_argument('--model', type=str, default='MILK')
    parser.add_argument(
        '--train_stage',
        type=str,
        default='imputer_backprop',
        choices=[
            'imputer',
            'imputer_param',
            'imputer_backprop',
            'imputer_init',
            'imputer_align',
            'imputer_promrl_main',
            'imputer_adapter',
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
    parser.add_argument(
        '--promrl_projection_bias',
        type=int,
        default=1,
        help='Whether learned ProMRL projection heads use bias. Enabled to match I3 MGCN Linear projection heads.',
    )
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
        choices=['raw_decoder', 'shared_identity'],
    )
    parser.add_argument(
        '--promrl_projection_mode',
        type=str,
        default='learned',
        choices=['learned', 'identity'],
        help='Projection used before ProMRL/imputation. identity uses normalized modal features directly and requires feature_dim == contra_dim.',
    )
    parser.add_argument(
        '--gcn_frontend_mode',
        type=str,
        default='original_linear',
        choices=['original_linear', 'deep_mlp', 'identity'],
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
        '--modal_feature_mask_source',
        type=str,
        default='nonzero',
        choices=['nonzero', 'external_observed'],
        help=(
            'How to derive observed modality masks when using modal_feature_override_dir. '
            'nonzero preserves legacy behavior; external_observed loads exported observed-mask files.'
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
        '--smore_beta_prior_dir',
        type=str,
        default='',
        help='Optional directory with SMORE item embeddings used as a conditional beta prior.',
    )
    parser.add_argument(
        '--smore_beta_prior_train_dir',
        type=str,
        default='',
        help='Optional explicit train-phase SMORE prior feature directory.',
    )
    parser.add_argument(
        '--smore_beta_prior_eval_dir',
        type=str,
        default='',
        help='Optional explicit eval/test-phase SMORE prior feature directory.',
    )
    parser.add_argument(
        '--smore_beta_prior_image_file',
        type=str,
        default='image_item_embeds.npy',
        help='Image SMORE item embedding filename for beta prior.',
    )
    parser.add_argument(
        '--smore_beta_prior_text_file',
        type=str,
        default='text_item_embeds.npy',
        help='Text SMORE item embedding filename for beta prior.',
    )
    parser.add_argument(
        '--smore_beta_prior_lambda',
        type=float,
        default=0.0,
        help='Weight for SMORE conditional beta prior. Zero disables the prior.',
    )
    parser.add_argument(
        '--smore_beta_prior_rho',
        type=float,
        default=1.0,
        help='Deprecated for learned covariance prior; kept for old scripts.',
    )
    parser.add_argument('--smore_beta_prior_hidden_dim', type=int, default=128)
    parser.add_argument('--smore_beta_prior_dropout', type=float, default=0.0)
    parser.add_argument('--smore_beta_prior_normalize', type=int, default=0)
    parser.add_argument('--smore_beta_prior_var_min', type=float, default=0.1)
    parser.add_argument('--smore_beta_prior_var_max', type=float, default=2.0)
    parser.add_argument(
        '--smore_beta_prior_scope',
        type=str,
        default='stage12_recommender',
        choices=['stage12', 'stage12_recommender', 'all_nonparam'],
        help='Stages where the SMORE beta prior is active; imputer_param is always excluded.',
    )
    parser.add_argument(
        '--beta_completion_mode',
        type=str,
        default='linear',
        choices=['linear', 'decoder'],
        help=(
            'Mapping from closed-form beta posterior to shared modal features. '
            'linear keeps CalMRL W beta + mu; decoder uses an MLP D_m(beta).'
        ),
    )
    parser.add_argument('--beta_completion_decoder_hidden_dim', type=int, default=128)
    parser.add_argument('--beta_completion_decoder_dropout', type=float, default=0.0)
    parser.add_argument(
        '--beta_completion_rec_weight',
        type=float,
        default=1.0,
        help='Extra observed-modality reconstruction weight for beta_completion_mode != linear.',
    )
    parser.add_argument(
        '--beta_completion_rec_loss',
        type=str,
        default='mse_cosine',
        choices=['mse', 'cosine', 'mse_cosine'],
    )
    parser.add_argument(
        '--beta_completion_detach_beta',
        type=int,
        default=1,
        help='Detach closed-form beta before the beta completion decoder reconstruction loss.',
    )
    parser.add_argument(
        '--completion_gate_mode',
        type=str,
        default='off',
        choices=[
            'off',
            'reliability',
            'missing_reliability',
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
    parser.add_argument('--completion_gate_lr', type=float, default=None)
    parser.add_argument('--completion_gate_detach_inputs', type=int, default=1)
    parser.add_argument('--completion_gate_stats_norm', type=int, default=1)
    parser.add_argument('--completion_gate_use_item_context', type=int, default=1)
    parser.add_argument(
        '--completion_gate_item_context_source',
        type=str,
        default='id_embedding',
        choices=['id_embedding', 'shared_mean', 'off'],
    )
    parser.add_argument('--item_graph_topk', type=int, default=20)
    parser.add_argument(
        '--item_graph_kind',
        type=str,
        default='none',
        choices=[
            'none',
            'fused_completed',
            'modality_completed',
            'modality_completed_confidence',
            'modality_completed_dynamic_confidence',
            'fused_completed_confidence',
            'fused_completed_dynamic_confidence',
        ],
        help=(
            'Item-item graph adapter kind. fused_completed fuses CF plus completed-feature '
            'modality graphs before propagation; modality_completed propagates each modality '
            'embedding with its own CF plus completed-feature modality graph; '
            'modality_completed_confidence adds learnable edge confidence (rr/ri/ii) for each modality completed graph; '
            'modality_completed_dynamic_confidence recomputes per-modality topk from the learned confidences during propagation; '
            'fused_completed_confidence learns edge confidence for real-real, real-imputed, '
            'and imputed-imputed completed-feature edges after a fixed initial topk; '
            'fused_completed_dynamic_confidence recomputes topk from the learned confidences during propagation.'
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
        '--item_graph_image_cf_weight',
        type=float,
        default=None,
        help='CF graph weight used only for the image modality-specific item graph. Defaults to item_graph_cf_weight.',
    )
    parser.add_argument(
        '--item_graph_image_semantic_weight',
        type=float,
        default=None,
        help='Image semantic graph weight used only for the image modality-specific item graph. Defaults to item_graph_image_weight.',
    )
    parser.add_argument(
        '--item_graph_text_cf_weight',
        type=float,
        default=None,
        help='CF graph weight used only for the text modality-specific item graph. Defaults to item_graph_cf_weight.',
    )
    parser.add_argument(
        '--item_graph_text_semantic_weight',
        type=float,
        default=None,
        help='Text semantic graph weight used only for the text modality-specific item graph. Defaults to item_graph_text_weight.',
    )
    parser.add_argument(
        '--item_graph_audio_cf_weight',
        type=float,
        default=None,
        help='CF graph weight used only for the audio modality-specific item graph. Defaults to item_graph_cf_weight.',
    )
    parser.add_argument(
        '--item_graph_audio_semantic_weight',
        type=float,
        default=None,
        help='Audio semantic graph weight used only for the audio modality-specific item graph. Defaults to item_graph_audio_weight.',
    )
    parser.add_argument('--item_graph_rr_confidence_init', type=float, default=1.0)
    parser.add_argument('--item_graph_ri_confidence_init', type=float, default=1.0)
    parser.add_argument('--item_graph_ii_confidence_init', type=float, default=1.0)
    parser.add_argument('--item_graph_text_rr_confidence_init', type=float, default=None)
    parser.add_argument('--item_graph_text_ri_confidence_init', type=float, default=None)
    parser.add_argument('--item_graph_text_ii_confidence_init', type=float, default=None)
    parser.add_argument('--item_graph_image_rr_confidence_init', type=float, default=None)
    parser.add_argument('--item_graph_image_ri_confidence_init', type=float, default=None)
    parser.add_argument('--item_graph_image_ii_confidence_init', type=float, default=None)
    parser.add_argument('--item_graph_audio_rr_confidence_init', type=float, default=None)
    parser.add_argument('--item_graph_audio_ri_confidence_init', type=float, default=None)
    parser.add_argument('--item_graph_audio_ii_confidence_init', type=float, default=None)
    parser.add_argument(
        '--item_graph_confidence_lr',
        type=float,
        default=None,
        help='Optional optimizer learning rate for learnable item graph edge confidences. Defaults to lr_rec.',
    )
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
    parser.add_argument('--item_graph_text_rr_confidence_reg_target', type=float, default=None)
    parser.add_argument('--item_graph_text_ri_confidence_reg_target', type=float, default=None)
    parser.add_argument('--item_graph_text_ii_confidence_reg_target', type=float, default=None)
    parser.add_argument('--item_graph_image_rr_confidence_reg_target', type=float, default=None)
    parser.add_argument('--item_graph_image_ri_confidence_reg_target', type=float, default=None)
    parser.add_argument('--item_graph_image_ii_confidence_reg_target', type=float, default=None)
    parser.add_argument('--item_graph_audio_rr_confidence_reg_target', type=float, default=None)
    parser.add_argument('--item_graph_audio_ri_confidence_reg_target', type=float, default=None)
    parser.add_argument('--item_graph_audio_ii_confidence_reg_target', type=float, default=None)
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
        help='Feature space used to build completed modality item graphs. raw_decoder uses observed raw features plus decoded imputed raw features.',
    )
    parser.add_argument(
        '--item_graph_feature_source',
        type=str,
        default='internal_completion',
        choices=['internal_completion', 'external_completed'],
        help=(
            'Feature source used to build completed modality item graphs. '
            'internal_completion preserves the original MMRec path that masks raw features and runs the configured imputer. '
            'external_completed uses externally exported completed modal features; auto directory selection follows item_graph_mask_scope.'
        ),
    )
    parser.add_argument(
        '--item_graph_mask_scope',
        type=str,
        default='train',
        choices=['train', 'train_val', 'all'],
        help=(
            'Missing-mask scope used to build completed item graphs. '
            'train is strict and excludes recommendation val/test masks; '
            'all preserves the legacy all-split graph construction and is not valid under strict evaluation.'
        ),
    )
    parser.add_argument(
        '--item_graph_feature_dir',
        type=str,
        default='',
        help=(
            'Directory containing external completed item-graph features. '
            'If empty and item_graph_feature_source=external_completed, phase_train under modal_feature_override_dir is used by default.'
        ),
    )
    parser.add_argument(
        '--item_graph_feature_image_file',
        type=str,
        default='',
        help='Image feature filename for external completed item-graph features. Defaults to modal_feature_image_file.',
    )
    parser.add_argument(
        '--item_graph_feature_text_file',
        type=str,
        default='',
        help='Text feature filename for external completed item-graph features. Defaults to modal_feature_text_file.',
    )
    parser.add_argument(
        '--item_graph_feature_image_mask_file',
        type=str,
        default='image_observed_mask.npy',
        help='Optional observed-mask filename for external image item-graph features. True means observed.',
    )
    parser.add_argument(
        '--item_graph_feature_text_mask_file',
        type=str,
        default='text_observed_mask.npy',
        help='Optional observed-mask filename for external text item-graph features. True means observed.',
    )
    parser.add_argument(
        '--item_graph_audio_weight',
        type=float,
        default=0.0,
        help='Audio graph weight. Single-source graph runs are ablations: set exactly one item_graph_*_weight positive.',
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
    parser.add_argument('--completion_gate_supervision_coeff', type=float, default=0.0)
    parser.add_argument('--completion_gate_supervision_observed_target', type=float, default=1.0)
    parser.add_argument('--completion_gate_counterfactual_coeff', type=float, default=0.0)
    parser.add_argument('--completion_gate_counterfactual_ratio', type=float, default=0.5)
    parser.add_argument('--completion_gate_counterfactual_mse_temp', type=float, default=0.1)
    parser.add_argument(
        '--completion_gate_apply_observed',
        type=int,
        default=1,
        help='If 0, learned completion/reliability gates are applied only to originally missing modalities.',
    )
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
        choices=['mean', 'rum', 'missing_weighted_mean', 'global_weighted_mean', 'reliability_weighted_mean'],
    )
    parser.add_argument('--missing_fusion_imputed_weight', type=float, default=0.7)
    parser.add_argument('--rum_tau', type=float, default=1.0)
    parser.add_argument('--rum_reliability_coeff', type=float, default=1.0)
    parser.add_argument('--rum_match_coeff', type=float, default=1.0)
    parser.add_argument('--rum_eval_user_batch_size', type=int, default=256)
    parser.add_argument('--rum_eval_item_chunk_size', type=int, default=4096)
    parser.add_argument('--alpha_intra', type=float, default=1.0)
    parser.add_argument('--alpha_inter', type=float, default=1.0)
    parser.add_argument('--alpha_itm', type=float, default=1.0)
    parser.add_argument('--alpha_rec', type=float, default=0.1)
    parser.add_argument('--alpha_decode', type=float, default=1.0)
    parser.add_argument('--alpha_decode_kl', type=float, default=0.0)
    parser.add_argument('--decode_kl_temp', type=float, default=0.2)
    parser.add_argument(
        '--decode_loss_grad_mode',
        type=str,
        default='coupled',
        choices=['coupled', 'detached'],
        help='Whether decoder reconstruction loss updates completed/shared features.',
    )
    parser.add_argument(
        '--decode_loss_target_mode',
        type=str,
        default='observed',
        choices=['observed'],
        help='Stage1 decoder reconstruction target. Only observed targets are supported.',
    )
    parser.add_argument(
        '--decode_loss_pseudo_ratio',
        type=float,
        default=1.0,
        help='Deprecated; stage1 pseudo-missing decoder targets have been removed.',
    )
    parser.add_argument('--modality_bpr_coeff', type=float, default=0.2)
    parser.add_argument('--beta_intra', type=float, default=0.05)
    parser.add_argument('--beta_inter', type=float, default=0.05)
    parser.add_argument('--beta_itm', type=float, default=0.05)
    parser.add_argument('--beta_rec', type=float, default=0.01)
    parser.add_argument('--beta_decode', type=float, default=0.01)
    parser.add_argument('--beta_decode_kl', type=float, default=0.0)
    parser.add_argument('--gamma_align', type=float, default=0.0)
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
    parser.add_argument('--stage1_profile', type=str, default='legacy', choices=['legacy', 'v2'])
    parser.add_argument('--stage1_v2_loss_preset', type=str, default='legacy', choices=['legacy', 'balanced'])
    parser.add_argument(
        '--stage1_2_mode',
        type=str,
        default=None,
        choices=['observed'],
        help='Stage1.2 target mode. Only observed mode is supported.',
    )
    parser.add_argument('--generative_update_mode', type=str, default='em', choices=['em', 'fixed'])
    parser.add_argument('--stage1_masking_policy', type=str, default='fixed', choices=['fixed', 'dynamic'])
    parser.add_argument(
        '--stage1_rec_loss_mode',
        type=str,
        default='observed',
        choices=['observed'],
        help='How stage1 rec/NLL loss is computed. Only observed reconstruction is supported.',
    )
    parser.add_argument(
        '--stage1_rec_pseudo_ratio',
        type=float,
        default=1.0,
        help='Deprecated; stage1 pseudo-missing rec loss has been removed.',
    )
    parser.add_argument(
        '--alpha_stage1_rec_guidance',
        type=float,
        default=0.0,
        help='Extra frozen-recommender BPR guidance used during imputer_backprop.',
    )
    parser.add_argument('--imputation_val_rate', type=float, default=0.0)
    parser.add_argument(
        '--imputation_selection_policy',
        type=str,
        default='legacy',
        choices=['legacy', 'stage1_default', 'promrl_shared', 'adapter_default', 'recommender_probe'],
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
    parser.add_argument(
        '--alternating_stage12_stage2',
        type=int,
        default=0,
        help='Test mode: alternate one epoch of stage1.2 imputer_backprop and one epoch of stage2 recommender.',
    )
    parser.add_argument(
        '--alternating_cycles',
        type=int,
        default=-1,
        help='Number of stage1.2/stage2 pairs. Negative uses epoch.',
    )
    parser.add_argument('--alternating_stage1_batch_size', type=int, default=256)
    parser.add_argument('--alternating_stage2_batch_size', type=int, default=-1)
    parser.add_argument('--alternating_stage1_lr_imp', type=float, default=0.0005)
    parser.add_argument('--alternating_stage1_lr_decoder', type=float, default=0.0002)
    parser.add_argument('--alternating_stage1_freeze_decoder', type=int, default=0)
    parser.add_argument('--alternating_stage2_freeze_decoder', type=int, default=1)
    parser.add_argument('--alternating_stage1_alpha_intra', type=float, default=1.0)
    parser.add_argument('--alternating_stage1_alpha_inter', type=float, default=1.0)
    parser.add_argument('--alternating_stage1_alpha_itm', type=float, default=1.0)
    parser.add_argument('--alternating_stage1_alpha_rec', type=float, default=1.0)
    parser.add_argument('--alternating_stage1_alpha_decode', type=float, default=1.0)

    return parser


def _apply_stage1_defaults(args):
    if getattr(args, 'stage1_2_mode', None) is not None:
        if args.train_stage != 'imputer_backprop':
            raise ValueError('--stage1_2_mode is only valid for train_stage=imputer_backprop')
        args.stage1_rec_loss_mode = 'observed'
        args.decode_loss_target_mode = 'observed'
    return args


def _validate_protocol_args(args):
    if args.evaluation_protocol == 'strict':
        if args.selection_mode == 'test':
            raise ValueError('Strict evaluation cannot use selection_mode=test; use validation selection.')
        if args.imputation_selection_split == 'test':
            raise ValueError('Strict evaluation cannot use imputation_selection_split=test; use train or val.')
        if getattr(args, 'item_graph_mask_scope', 'train') == 'all':
            raise ValueError('Strict evaluation cannot use item_graph_mask_scope=all; use train or train_val.')
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
    'fused_completed',
    'modality_completed',
    'modality_completed_confidence',
    'modality_completed_dynamic_confidence',
    'fused_completed_confidence',
    'fused_completed_dynamic_confidence',
):
    my_model.build_completed_item_graph()
tool.cprint('Init Model')

# ----------------------------------- Session Init -----------------------------------------------------------

my_session = MILK_session(my_env, my_model, my_loader)
tool.cprint('Init Session')

# ---------------------------------------- Main -----------------------------------------------------------

t = time.time()
if my_env.args.alternating_stage12_stage2:
    my_session.train_alternating_stage12_stage2(my_env.args.alternating_cycles)
else:
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
    'imputer',
    'imputer_param',
    'imputer_backprop',
    'imputer_init',
    'imputer_align',
    'imputer_promrl_main',
    'imputer_adapter',
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
