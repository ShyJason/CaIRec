import copy
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from promrl_core.general_module import Contra_head, Match_head
from promrl_core.utils.impute import update_posterior, compute_nll_loss
from promrl_core.utils.eigen import (
    eigenvalue_computation_pmcl,
    shifted_relation_lifted_directions,
)
from promrl_core.promrl_variants import _build_mlp


class MGCN(torch.nn.Module):
    def __init__(self, edge_index, num_user, num_item, dim_feat, dim_latent, frontend_mode="deep_mlp"):
        super(MGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_feat = dim_feat
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.n_layers = 3
        self.frontend_mode = frontend_mode

        if self.frontend_mode == "identity":
            if self.dim_feat != self.dim_latent:
                raise ValueError(
                    f"gcn_frontend_mode='identity' requires dim_feat == dim_latent, "
                    f"got {self.dim_feat} and {self.dim_latent}"
                )
            self.MLP = nn.Identity()
        elif self.frontend_mode == "original_linear":
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
        elif self.frontend_mode == "deep_mlp":
            self.MLP = _build_mlp(self.dim_feat, self.dim_latent * 2, self.dim_latent, dropout=0.1)
        else:
            raise ValueError(f"Unsupported gcn_frontend_mode: {self.frontend_mode}")

    def forward(
        self,
        features,
        user_id_preference,
        skip_mlp=False,
    ):
        temp_features = features if skip_mlp else self.MLP(features)
        temp_features = torch.nan_to_num(temp_features, nan=0.0, posinf=0.0, neginf=0.0)
        all_emb = torch.cat((user_id_preference, temp_features), dim=0)
        all_emb = F.normalize(all_emb)
        all_emb = torch.nan_to_num(all_emb, nan=0.0, posinf=0.0, neginf=0.0)

        embs = [all_emb]
        g_droped = self.edge_index

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_user, self.num_item])

        return users, items


class ResidualCompletionAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.skip = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        return self.norm(self.skip(x) + self.mlp(x))


class DirectionNormDecoder(nn.Module):
    """Decode a completion latent into a native-scale raw feature.

    Direction and magnitude are deliberately predicted by separate heads.  A
    single normalized output cannot reproduce item-specific raw feature norms,
    while an unconstrained raw regression head tends to spend most of its
    capacity on the much larger visual-feature scale.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, initial_log_norm):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.direction_head = nn.Linear(hidden_dim, output_dim)
        self.log_norm_head = nn.Sequential(
            nn.Linear(hidden_dim, max(32, hidden_dim // 4)),
            nn.GELU(),
            nn.Linear(max(32, hidden_dim // 4), 1),
        )
        nn.init.zeros_(self.log_norm_head[-1].weight)
        nn.init.constant_(self.log_norm_head[-1].bias, float(initial_log_norm))

    def forward(self, x):
        hidden = self.trunk(x)
        direction = F.normalize(self.direction_head(hidden), dim=-1)
        # The clamp is only a numerical guard.  It spans norms from 6.7e-3 to
        # 3.0e3, comfortably beyond all modalities used in this repository.
        log_norm = self.log_norm_head(hidden).clamp(min=-5.0, max=8.0)
        return direction * log_norm.exp()


class MILK_model(torch.nn.Module):
    def __init__(self, env, dataset):
        super(MILK_model, self).__init__()
        self.env = env
        self.dataset = dataset
        self.n_layers = 3
        self.A_split = False
        self.n_user = dataset.n_user
        self.m_item = dataset.m_item
        self.Graph = dataset.getSparseGraph()
        self.ItemItemGraph = None
        self.ItemItemGraphs = {}
        self.ItemItemGraphComponents = {}
        self.InductiveItemItemGraphs = {}
        self._inductive_item_graph_split = None
        self.item_graph_dynamic_norm_type = "rw"
        self.ItemItemRawGraph = None
        self.free_emb_dimension = self.env.args.free_emb_dimension
        self.has_audio_modality = dataset.audio_feat is not None
        self.has_video_modality = getattr(dataset, "video_feat", None) is not None
        self.modalities = ["v", "t"]
        if self.has_audio_modality:
            self.modalities.append("a")
        if self.has_video_modality:
            self.modalities.append("d")

        self.train_missing_modality_items = dataset.train_missing_modality_items
        self.val_missing_modality_items = getattr(dataset, "val_missing_modality_items", {"items": [], "indicator": []})
        self.eval_val_missing_modality_items = getattr(
            dataset,
            "eval_val_missing_modality_items",
            {"items": [], "indicator": []},
        )
        self.test_missing_modality_items = dataset.test_missing_modality_items
        self.modal_feature_mask_source = getattr(self.env.args, "modal_feature_mask_source", "nonzero")
        self.modal_feature_override_is_completed = bool(
            getattr(self.env.args, "modal_feature_override_is_completed", 0)
        )
        self.train_external_modal_observed_masks = self._tensorize_external_modal_masks(
            getattr(dataset, "train_external_modal_observed_masks", None)
        )
        self.eval_external_modal_observed_masks = self._tensorize_external_modal_masks(
            getattr(dataset, "eval_external_modal_observed_masks", None)
        )

        self.audio_feat = None
        self.ori_audio_feat = None
        self.video_feat = None
        self.ori_video_feat = None

        native_image_feat = torch.tensor(dataset.image_feat, dtype=torch.float32).to(self.env.device)
        native_text_feat = torch.tensor(dataset.text_feat, dtype=torch.float32).to(self.env.device)
        self._register_native_raw_statistics("v", native_image_feat, dataset)
        self._register_native_raw_statistics("t", native_text_feat, dataset)
        self.ori_image_feat = F.normalize(native_image_feat)
        self.ori_text_feat = F.normalize(native_text_feat)

        self.eval_ori_image_feat = torch.tensor(
            getattr(dataset, "eval_image_feat", dataset.image_feat),
            dtype=torch.float32,
        ).to(self.env.device)
        self.eval_ori_image_feat = F.normalize(self.eval_ori_image_feat)

        self.eval_ori_text_feat = torch.tensor(
            getattr(dataset, "eval_text_feat", dataset.text_feat),
            dtype=torch.float32,
        ).to(self.env.device)
        self.eval_ori_text_feat = F.normalize(self.eval_ori_text_feat)

        if self.has_audio_modality:
            native_audio_feat = torch.tensor(dataset.audio_feat, dtype=torch.float32).to(self.env.device)
            self._register_native_raw_statistics("a", native_audio_feat, dataset)
            self.ori_audio_feat = F.normalize(native_audio_feat)
            self.eval_ori_audio_feat = torch.tensor(
                getattr(dataset, "eval_audio_feat", dataset.audio_feat),
                dtype=torch.float32,
            ).to(self.env.device)
            self.eval_ori_audio_feat = F.normalize(self.eval_ori_audio_feat)
        if self.has_video_modality:
            native_video_feat = torch.tensor(dataset.video_feat, dtype=torch.float32).to(self.env.device)
            self._register_native_raw_statistics("d", native_video_feat, dataset)
            self.ori_video_feat = F.normalize(native_video_feat)
            self.eval_ori_video_feat = torch.tensor(
                getattr(dataset, "eval_video_feat", dataset.video_feat),
                dtype=torch.float32,
            ).to(self.env.device)
            self.eval_ori_video_feat = F.normalize(self.eval_ori_video_feat)

        self.uses_split_modal_feature_override = bool(
            getattr(dataset, "uses_split_modal_feature_override", False)
        )

        self.contra_dim = self.env.args.contra_dim
        self.d_beta = self.env.args.d_beta
        self.ema_eta = self.env.args.ema_eta
        self.itm_temp = self.env.args.itm_temp
        self.lambda_itm = self.env.args.lambda_itm
        self.disable_imputation = bool(self.env.args.disable_imputation)
        self.feature_bridge_mode = self.env.args.feature_bridge_mode
        if self.feature_bridge_mode not in ("raw_decoder", "latent_direct", "decoupled_latent"):
            raise ValueError(f"Unsupported feature_bridge_mode: {self.feature_bridge_mode}")
        self.use_latent_direct_bridge = self.feature_bridge_mode == "latent_direct"
        self.use_decoupled_latent_bridge = self.feature_bridge_mode == "decoupled_latent"
        self.use_latent_completion_bridge = self.use_latent_direct_bridge or self.use_decoupled_latent_bridge
        self.use_decode_head = (
            not self.use_latent_completion_bridge
            or bool(getattr(self.env.args, "enable_raw_completion_decoder", 0))
        )
        self.decoder_output_mode = getattr(self.env.args, "decoder_output_mode", "normalized")
        if self.decoder_output_mode not in ("normalized", "native_direction_norm"):
            raise ValueError(f"Unsupported decoder_output_mode: {self.decoder_output_mode}")
        self.gcn_frontend_mode = self.env.args.gcn_frontend_mode
        self.promrl_projection_mode = getattr(self.env.args, "promrl_projection_mode", "learned")
        self.promrl_dim = self.free_emb_dimension if self.use_latent_completion_bridge else self.contra_dim
        self.completion_gate_mode = self.env.args.completion_gate_mode
        self.use_completion_gate = self.completion_gate_mode != "off"
        self.use_rank_residual_completion_gate = self.completion_gate_mode in (
            "rank_residual",
            "rank_residual_norm",
            "rank_residual_allnorm",
            "rank_residual_allgate",
            "rank_residual_softmax",
            "rank_residual_global",
            "rank_residual_shrink",
            "rank_residual_centered",
            "rank_residual_delta",
            "rank_residual_centered_allgate",
        )
        self.use_shrink_rank_residual_completion_gate = (
            self.completion_gate_mode == "rank_residual_shrink"
        )
        self.use_global_rank_residual_completion_gate = (
            self.completion_gate_mode == "rank_residual_global"
        )
        self.use_normalized_rank_residual_completion_gate = self.completion_gate_mode == "rank_residual_norm"
        self.use_all_normalized_rank_residual_completion_gate = (
            self.completion_gate_mode == "rank_residual_allnorm"
        )
        self.use_all_modal_rank_residual_completion_gate = (
            self.completion_gate_mode == "rank_residual_allgate"
        )
        self.use_centered_rank_residual_completion_gate = (
            self.completion_gate_mode == "rank_residual_centered"
        )
        self.use_delta_rank_residual_completion_gate = (
            self.completion_gate_mode == "rank_residual_delta"
        )
        self.use_centered_all_modal_rank_residual_completion_gate = (
            self.completion_gate_mode == "rank_residual_centered_allgate"
        )
        self.use_softmax_rank_residual_completion_gate = (
            self.completion_gate_mode == "rank_residual_softmax"
        )
        self.use_learned_completion_gate_mix = (
            bool(getattr(self.env.args, "completion_gate_learn_mix", 0))
            and self.use_rank_residual_completion_gate
        )
        self.use_learned_completion_gate = self.completion_gate_mode in (
            "reliability",
            "rank_residual",
            "rank_residual_norm",
            "rank_residual_allnorm",
            "rank_residual_allgate",
            "rank_residual_softmax",
            "rank_residual_shrink",
            "rank_residual_centered",
            "rank_residual_delta",
            "rank_residual_centered_allgate",
        )
        self.completion_gate_detach_inputs = bool(self.env.args.completion_gate_detach_inputs)
        self.completion_gate_use_item_context = bool(self.env.args.completion_gate_use_item_context)
        self.completion_gate_item_context_source = self.env.args.completion_gate_item_context_source
        if not self.completion_gate_use_item_context:
            self.completion_gate_item_context_source = "off"
        self.item_graph_modal_alpha = min(
            max(float(getattr(self.env.args, "item_graph_modal_alpha", 0.0)), 0.0),
            1.0,
        )
        self.item_graph_modal_layers = max(int(getattr(self.env.args, "item_graph_modal_layers", 1)), 0)
        self.item_graph_modal_target = getattr(self.env.args, "item_graph_modal_target", "all")
        self.item_graph_kind = getattr(self.env.args, "item_graph_kind", "none")
        if self.item_graph_kind == "modality_masked":
            if not self.disable_imputation:
                raise ValueError("item_graph_kind=modality_masked requires disable_imputation=1")
            if getattr(self.env.args, "imputer_ckpt", None):
                raise ValueError("item_graph_kind=modality_masked forbids imputer_ckpt")
            if getattr(self.env.args, "ckpt", None):
                raise ValueError("item_graph_kind=modality_masked forbids pretrained ckpt")
            if getattr(self.env.args, "train_stage", None) != "recommender":
                raise ValueError("item_graph_kind=modality_masked is only valid for train_stage=recommender")
        self.use_completed_item_graph = self.item_graph_kind in (
            "modality_masked",
            "fused_completed",
            "modality_completed",
            "modality_completed_inductive",
            "modality_completed_confidence",
            "modality_completed_dynamic_confidence",
            "fused_completed_confidence",
            "fused_completed_dynamic_confidence",
            "fused_completed_reliability",
            "fused_completed_reliability_topk",
        )
        self.use_item_graph_edge_confidence = self.item_graph_kind in (
            "fused_completed_confidence",
            "fused_completed_dynamic_confidence",
            "modality_completed_confidence",
            "modality_completed_dynamic_confidence",
        )
        self.use_dynamic_item_graph_edge_confidence = self.item_graph_kind in (
            "fused_completed_dynamic_confidence",
            "modality_completed_dynamic_confidence",
        )
        self.item_graph_confidence_transform = getattr(
            self.env.args,
            "item_graph_confidence_transform",
            "blend",
        )
        if self.item_graph_confidence_transform not in ("blend", "sigmoid"):
            raise ValueError(f"Unsupported item graph confidence transform: {self.item_graph_confidence_transform}")
        self.item_graph_confidence_blend = min(
            max(float(getattr(self.env.args, "item_graph_confidence_blend", 1.0)), 0.0),
            1.0,
        )
        self.item_graph_dynamic_score_blend = min(
            max(float(getattr(self.env.args, "item_graph_dynamic_score_blend", 1.0)), 0.0),
            1.0,
        )
        self.item_graph_dynamic_score_blend_start = float(
            getattr(self.env.args, "item_graph_dynamic_score_blend_start", -1.0)
        )
        self.item_graph_dynamic_score_blend_warmup_epochs = max(
            int(getattr(self.env.args, "item_graph_dynamic_score_blend_warmup_epochs", 0) or 0),
            0,
        )
        self.item_graph_dynamic_neighbor_blend = min(
            max(float(getattr(self.env.args, "item_graph_dynamic_neighbor_blend", 1.0)), 0.0),
            1.0,
        )
        self.item_graph_dynamic_neighbor_blend_start = float(
            getattr(self.env.args, "item_graph_dynamic_neighbor_blend_start", -1.0)
        )
        self.item_graph_dynamic_neighbor_blend_warmup_epochs = max(
            int(getattr(self.env.args, "item_graph_dynamic_neighbor_blend_warmup_epochs", 0) or 0),
            0,
        )
        self.item_graph_confidence_min = max(
            float(getattr(self.env.args, "item_graph_confidence_min", 0.25)),
            0.0,
        )
        self.item_graph_confidence_max = max(
            float(getattr(self.env.args, "item_graph_confidence_max", 4.0)),
            self.item_graph_confidence_min,
        )
        edge_conf_init = torch.tensor(
            [
                max(float(getattr(self.env.args, "item_graph_rr_confidence_init", 1.0)), 1e-6),
                max(float(getattr(self.env.args, "item_graph_ri_confidence_init", 1.0)), 1e-6),
                max(float(getattr(self.env.args, "item_graph_ii_confidence_init", 1.0)), 1e-6),
            ],
            dtype=torch.float32,
        )
        edge_conf_init = edge_conf_init.clamp(
            min=self.item_graph_confidence_min,
            max=self.item_graph_confidence_max,
        )
        if self.item_graph_confidence_transform == "sigmoid":
            span = max(self.item_graph_confidence_max - self.item_graph_confidence_min, 1e-6)
            normalized_conf = ((edge_conf_init - self.item_graph_confidence_min) / span).clamp(1e-4, 1.0 - 1e-4)
            edge_conf_param_init = torch.logit(normalized_conf)
        else:
            edge_conf_param_init = torch.log(edge_conf_init)
        self.use_item_graph_modality_specific_confidence = (
            bool(int(getattr(self.env.args, "item_graph_modality_specific_confidence", 0)))
            and self.item_graph_kind in ("modality_completed_confidence", "modality_completed_dynamic_confidence")
        )
        if self.use_item_graph_modality_specific_confidence:
            self.item_graph_edge_confidence_params = nn.ParameterDict(
                {
                    modality: nn.Parameter(edge_conf_param_init.clone())
                    for modality in self.modalities
                }
            )
        else:
            self.item_graph_edge_confidence_params = nn.ParameterList([nn.Parameter(edge_conf_param_init)])
        self.use_item_graph_modal_residual = (
            self.item_graph_modal_alpha > 0.0
            and self.item_graph_modal_layers > 0
        )
        self.completion_gate_floor = self.env.args.completion_gate_floor
        self.completion_gate_tail_quantile = min(
            max(float(getattr(self.env.args, "completion_gate_tail_quantile", 1.0)), 0.0),
            1.0,
        )
        self.fusion_mode = getattr(self.env.args, "fusion_mode", "mean")
        if self.fusion_mode not in {"mean", "rum", "global_weighted_mean", "posterior_reliability"}:
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")
        self.use_rum_fusion = self.fusion_mode == "rum"
        self.use_global_weighted_fusion = self.fusion_mode == "global_weighted_mean"
        self.use_posterior_reliability = self.fusion_mode == "posterior_reliability"
        self.posterior_reliability_scope = getattr(
            self.env.args, "posterior_reliability_scope", "both"
        )
        if self.posterior_reliability_scope not in {"both", "graph", "fusion"}:
            raise ValueError(
                "Unsupported posterior reliability scope: "
                f"{self.posterior_reliability_scope}"
            )
        self.posterior_reliability_scale = max(
            float(getattr(self.env.args, "posterior_reliability_scale", 1.0)),
            0.0,
        )
        self.posterior_reliability_floor = min(
            max(float(getattr(self.env.args, "posterior_reliability_floor", 0.0)), 0.0),
            1.0,
        )
        self.completion_gate_score_residual_alpha = min(
            max(float(getattr(self.env.args, "completion_gate_score_residual_alpha", 0.0)), 0.0),
            1.0,
        )
        self.use_score_residual_completion_gate = (
            self.completion_gate_score_residual_alpha > 0.0
            and self.use_rank_residual_completion_gate
            and not self.use_rum_fusion
        )

        train_items = np.asarray(getattr(dataset, "trainItem", []), dtype=np.int64)
        item_degree = np.bincount(train_items, minlength=self.m_item)[: self.m_item]
        item_degree_tensor = torch.tensor(item_degree, dtype=torch.float32, device=self.env.device)
        self.register_buffer("item_train_degree", item_degree_tensor)
        if self.completion_gate_tail_quantile < 1.0:
            threshold = torch.quantile(item_degree_tensor, self.completion_gate_tail_quantile)
            tail_mask = item_degree_tensor <= threshold
        else:
            threshold = item_degree_tensor.max() if item_degree_tensor.numel() else torch.tensor(0.0, device=self.env.device)
            tail_mask = torch.ones(self.m_item, dtype=torch.bool, device=self.env.device)
        self.completion_gate_tail_degree_threshold = float(threshold.detach().cpu())
        self.register_buffer("completion_gate_tail_mask", tail_mask.unsqueeze(1).float())

        if not self.use_latent_completion_bridge:
            self.contra_head_v = Contra_head(self.ori_image_feat.size(1), self.contra_dim)
            self.contra_head_t = Contra_head(self.ori_text_feat.size(1), self.contra_dim)
            if "a" in self.modalities:
                self.contra_head_a = Contra_head(self.ori_audio_feat.size(1), self.contra_dim)
            if "d" in self.modalities:
                self.contra_head_d = Contra_head(self.ori_video_feat.size(1), self.contra_dim)
        self.itm_cross_attn = nn.MultiheadAttention(
            self.promrl_dim,
            num_heads=self.env.args.itm_num_heads,
            batch_first=True,
        )
        self.itm_head = Match_head(self.promrl_dim)

        self.W = nn.ParameterDict({
            modality: nn.Parameter(torch.randn(self.promrl_dim, self.d_beta) * 0.02, requires_grad=False)
            for modality in self.modalities
        })
        self.mu = nn.ParameterDict({
            modality: nn.Parameter(torch.zeros(self.promrl_dim), requires_grad=False)
            for modality in self.modalities
        })
        self.log_sigma = nn.ParameterDict({
            modality: nn.Parameter(torch.zeros(1), requires_grad=False)
            for modality in self.modalities
        })

        if self.use_decode_head:
            self.decoder_v = self._build_modal_decoder("v", self.ori_image_feat.size(1))
            self.decoder_t = self._build_modal_decoder("t", self.ori_text_feat.size(1))
            if "a" in self.modalities:
                self.decoder_a = self._build_modal_decoder("a", self.ori_audio_feat.size(1))
            if "d" in self.modalities:
                self.decoder_d = self._build_modal_decoder("d", self.ori_video_feat.size(1))

        v_input_dim = self.ori_image_feat.size(1)
        t_input_dim = self.ori_text_feat.size(1)
        a_input_dim = self.ori_audio_feat.size(1) if "a" in self.modalities else None
        d_input_dim = self.ori_video_feat.size(1) if "d" in self.modalities else None

        self.v_gcn = MGCN(
            self.Graph,
            self.n_user,
            self.m_item,
            v_input_dim,
            self.free_emb_dimension,
            frontend_mode=self.gcn_frontend_mode,
        )
        self.t_gcn = MGCN(
            self.Graph,
            self.n_user,
            self.m_item,
            t_input_dim,
            self.free_emb_dimension,
            frontend_mode=self.gcn_frontend_mode,
        )
        if "a" in self.modalities:
            self.a_gcn = MGCN(
                self.Graph,
                self.n_user,
                self.m_item,
                a_input_dim,
                self.free_emb_dimension,
                frontend_mode=self.gcn_frontend_mode,
            )
        if "d" in self.modalities:
            self.d_gcn = MGCN(
                self.Graph,
                self.n_user,
                self.m_item,
                d_input_dim,
                self.free_emb_dimension,
                frontend_mode=self.gcn_frontend_mode,
            )

        if self.use_decoupled_latent_bridge:
            self.comp_proj_v = self._build_latent_projection_head(v_input_dim)
            self.comp_proj_t = self._build_latent_projection_head(t_input_dim)
            self.comp_to_rec_v = self._build_completion_adapter()
            self.comp_to_rec_t = self._build_completion_adapter()
            if "a" in self.modalities:
                self.comp_proj_a = self._build_latent_projection_head(a_input_dim)
                self.comp_to_rec_a = self._build_completion_adapter()
            if "d" in self.modalities:
                self.comp_proj_d = self._build_latent_projection_head(d_input_dim)
                self.comp_to_rec_d = self._build_completion_adapter()

        self.user_emb = torch.nn.Embedding(
            num_embeddings=self.n_user, embedding_dim=self.free_emb_dimension
        )
        self.item_emb = torch.nn.Embedding(
            num_embeddings=self.m_item, embedding_dim=self.free_emb_dimension
        )
        self.user_modality_pref = torch.nn.Embedding(
            num_embeddings=self.n_user, embedding_dim=len(self.modalities)
        )
        self.rum_modality_bias = nn.Parameter(torch.zeros(len(self.modalities)))
        self.rum_observed_bias = nn.Parameter(torch.zeros(len(self.modalities)))
        self.rum_biases = nn.ParameterList([
            self.rum_modality_bias,
            self.rum_observed_bias,
        ])
        self.global_fusion_params = nn.ParameterList()
        if self.use_global_weighted_fusion:
            self.global_fusion_params.append(
                nn.Parameter(torch.zeros(len(self.modalities)))
            )

        self.completion_gate_global_logits = nn.ParameterDict()
        if self.use_global_rank_residual_completion_gate:
            self.completion_gate_global_logits = nn.ParameterDict({
                modality: nn.Parameter(torch.zeros(()))
                for modality in self.modalities
            })

        if self.use_learned_completion_gate:
            if self.completion_gate_item_context_source == "id_embedding":
                item_context_dim = self.free_emb_dimension
            elif self.completion_gate_item_context_source == "shared_mean":
                item_context_dim = self.promrl_dim
            else:
                item_context_dim = 0
            gate_input_dim = self.promrl_dim + item_context_dim + len(self.modalities)
            self.completion_gates = nn.ModuleDict({
                modality: self._build_completion_gate(gate_input_dim)
                for modality in self.modalities
            })

        self.completion_gate_mix_params = nn.ParameterList()
        if self.use_learned_completion_gate_mix:
            mix_max = min(max(float(getattr(self.env.args, "completion_gate_mix_max", 1.0)), 1e-4), 1.0)
            init_mix = min(max(float(self.env.args.completion_gate_mix_alpha), 1e-4), mix_max - 1e-4)
            init_ratio = init_mix / mix_max
            init_logit = torch.logit(torch.tensor(init_ratio, dtype=torch.float32))
            self.completion_gate_mix_params.append(nn.Parameter(init_logit))

        self.fusion_linear = nn.Sequential(
            nn.Linear(self.free_emb_dimension, self.free_emb_dimension, bias=False),
            nn.Dropout(),
            nn.Tanh(),
        )

        self.final_item = None
        self.final_user = None
        self.activate = torch.nn.Sigmoid()
        self.latest_promrl_losses = {}
        self.latest_completion_gate_metrics = {}
        self.latest_rum_fusion_metrics = {}
        self.latest_completion_gate_regularizer = torch.zeros((), device=self.env.device)
        self.latest_completion_gate_ungated_item_emb = None
        self.latest_completion_gate_gated_item_emb = None
        self._gcn_cache = None
        self._imputer_updates_enabled = True
        self._pending_em_updates = []
        self._dynamic_stage1_refresh_counter = 0
        self._item_user_sets = self._build_item_user_sets(dataset)
        self._co_interact_positive_items = self._build_co_interact_positive_items(dataset)

        torch.nn.init.normal_(self.user_emb.weight, std=0.1)
        torch.nn.init.normal_(self.item_emb.weight, std=0.1)
        torch.nn.init.zeros_(self.user_modality_pref.weight)
        torch.nn.init.eye_(self.fusion_linear[0].weight)

        self.to(self.env.device)
        self.init_missing_modality_set()
        self.configure_training_stage(
            self.env.args.train_stage,
            freeze_imputer=self.env.args.freeze_imputer,
            freeze_recommender=self.env.args.freeze_recommender,
            freeze_decoder=self.env.args.freeze_decoder,
        )

    def _canonical_stage(self, train_stage=None):
        return train_stage or self.env.args.train_stage

    def _build_co_interact_positive_items(self, dataset):
        positive_items = torch.arange(self.m_item, dtype=torch.long)
        for items in getattr(dataset, "train_data", {}).values():
            if len(items) < 2:
                continue
            sorted_items = sorted(set(int(item) for item in items if 0 <= int(item) < self.m_item))
            if len(sorted_items) < 2:
                continue
            for idx, item in enumerate(sorted_items):
                positive_items[item] = sorted_items[(idx + 1) % len(sorted_items)]
        return positive_items.to(self.env.device)

    def _build_item_user_sets(self, dataset):
        item_user_sets = [set() for _ in range(self.m_item)]
        for user, items in getattr(dataset, "train_data", {}).items():
            for item in items:
                item = int(item)
                if 0 <= item < self.m_item:
                    item_user_sets[item].add(int(user))
        return item_user_sets

    def _uses_stage1_holdout_metrics(self, train_stage=None):
        canonical_stage = self._canonical_stage(train_stage)
        return canonical_stage == "imputer_backprop"

    def _training_observed_mask(self, modality, dataset):
        mask = torch.ones(self.m_item, dtype=torch.bool, device=self.env.device)
        metadata = getattr(dataset, "train_missing_modality_items", None) or {}
        items = np.asarray(metadata.get("items", []), dtype=np.int64)
        indicators = np.asarray(metadata.get("indicator", []), dtype=np.int64)
        modality_index = self.modalities.index(modality)
        missing_items = items[indicators == modality_index]
        if missing_items.size:
            mask[torch.as_tensor(missing_items, device=self.env.device)] = False
        return mask

    def _register_native_raw_statistics(self, modality, native_feature, dataset):
        observed = self._training_observed_mask(modality, dataset)
        observed_feature = native_feature[observed]
        observed_norm = observed_feature.norm(dim=-1).clamp_min(1e-8)
        feature_std = observed_feature.std(dim=0, unbiased=False)
        # Avoid exploding standardized errors on dimensions which are nearly
        # constant.  The floor is derived only from train-observed features.
        std_floor = feature_std.mean().clamp_min(1e-6) * 0.05
        feature_std = feature_std.clamp_min(std_floor)
        self.register_buffer(
            f"native_raw_norm_{modality}",
            native_feature.norm(dim=-1).clamp_min(1e-8),
            persistent=False,
        )
        self.register_buffer(
            f"native_raw_std_{modality}",
            feature_std,
            persistent=False,
        )
        self.register_buffer(
            f"native_raw_initial_log_norm_{modality}",
            observed_norm.log().mean(),
            persistent=False,
        )

    def _native_raw_target(self, modality, normalized_feature, item_ids=None):
        norms = getattr(self, f"native_raw_norm_{modality}")
        if item_ids is not None:
            norms = norms[item_ids]
        return normalized_feature * norms.unsqueeze(-1)

    def _decoder_hidden_dim(self, raw_dim):
        return min(1024, max(256, raw_dim // 2))

    def _build_modal_decoder(self, modality, raw_dim):
        hidden_dim = self._decoder_hidden_dim(raw_dim)
        if self.decoder_output_mode == "native_direction_norm":
            initial_log_norm = getattr(
                self, f"native_raw_initial_log_norm_{modality}"
            ).item()
            return DirectionNormDecoder(
                self.promrl_dim,
                hidden_dim,
                raw_dim,
                initial_log_norm,
            )
        return nn.Sequential(
            nn.Linear(self.promrl_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, raw_dim),
        )

    def _build_latent_projection_head(self, raw_dim):
        if self.promrl_projection_mode == "identity":
            if raw_dim != self.promrl_dim:
                raise ValueError(
                    "promrl_projection_mode='identity' requires modal feature dim "
                    f"to equal promrl_dim ({self.promrl_dim}); got {raw_dim}"
                )
            return nn.Identity()
        if self.gcn_frontend_mode == "original_linear":
            return nn.Linear(raw_dim, self.free_emb_dimension)
        return _build_mlp(raw_dim, self.free_emb_dimension * 2, self.free_emb_dimension, dropout=0.1)

    def _build_completion_adapter(self):
        mode = getattr(self.env.args, "completion_adapter_mode", "linear_ln")
        if mode == "identity":
            if self.promrl_dim != self.free_emb_dimension:
                raise ValueError(
                    "completion_adapter_mode=identity requires promrl_dim == free_emb_dimension"
                )
            return nn.Identity()
        if mode == "linear_ln":
            return nn.Sequential(
                nn.Linear(self.promrl_dim, self.free_emb_dimension),
                nn.LayerNorm(self.free_emb_dimension),
            )
        if mode == "residual_mlp":
            hidden_dim = int(getattr(self.env.args, "completion_adapter_hidden_dim", 128))
            if hidden_dim <= 0:
                raise ValueError("completion_adapter_hidden_dim must be positive")
            dropout = float(getattr(self.env.args, "completion_adapter_dropout", 0.0))
            return ResidualCompletionAdapter(
                self.promrl_dim,
                self.free_emb_dimension,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        raise ValueError(f"Unsupported completion_adapter_mode: {mode}")

    def _build_completion_gate(self, input_dim):
        hidden_dim = self.env.args.completion_gate_hidden_dim
        gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.env.args.completion_gate_dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(gate[-1].weight)
        if self.use_shrink_rank_residual_completion_gate:
            init_logit = self.env.args.completion_gate_shrink_init_logit
        elif self.use_rank_residual_completion_gate:
            init_logit = 0.0
        else:
            init_logit = self.env.args.completion_gate_init_logit
        nn.init.constant_(gate[-1].bias, init_logit)
        return gate

    def _projection_modules(self):
        if self.use_decoupled_latent_bridge:
            return [getattr(self, f"comp_proj_{modality}") for modality in self.modalities]

        if self.use_latent_direct_bridge:
            return [getattr(self, f"{modality}_gcn").MLP for modality in self.modalities]

        return [getattr(self, f"contra_head_{modality}") for modality in self.modalities]

    def _imputer_modules(self):
        modules = self._projection_modules()
        modules.extend([
            self.W,
            self.mu,
            self.log_sigma,
            self.itm_cross_attn,
            self.itm_head,
        ])
        return modules

    def _decoder_modules(self):
        if not self.use_decode_head:
            return []
        return [getattr(self, f"decoder_{modality}") for modality in self.modalities]

    def _recommender_modules(self):
        modules = [self.user_emb, self.item_emb, self.fusion_linear]
        if self.use_rum_fusion:
            modules.extend([self.user_modality_pref, self.rum_biases])
        if self.use_global_weighted_fusion:
            modules.append(self.global_fusion_params)
        modules.extend(getattr(self, f"{modality}_gcn") for modality in self.modalities)
        if self.use_decoupled_latent_bridge:
            modules.extend(getattr(self, f"comp_to_rec_{modality}") for modality in self.modalities)
        if self.use_learned_completion_gate:
            modules.append(self.completion_gates)
        if self.use_global_rank_residual_completion_gate:
            modules.append(self.completion_gate_global_logits)
        if self.use_learned_completion_gate_mix:
            modules.append(self.completion_gate_mix_params)
        if self.use_item_graph_edge_confidence:
            modules.append(self.item_graph_edge_confidence_params)
        return modules

    def _set_modules_trainable(self, modules, trainable):
        for module in modules:
            for param in module.parameters():
                param.requires_grad = trainable

    def configure_training_stage(self, train_stage, freeze_imputer=-1, freeze_recommender=-1, freeze_decoder=0):
        canonical_stage = self._canonical_stage(train_stage)

        auto_freeze_imputer = canonical_stage == "recommender"
        auto_freeze_recommender = canonical_stage in ("imputer_param", "imputer_backprop")
        auto_freeze_decoder = canonical_stage == "imputer_param"

        freeze_imputer = auto_freeze_imputer if freeze_imputer < 0 else bool(freeze_imputer)
        freeze_recommender = auto_freeze_recommender if freeze_recommender < 0 else bool(freeze_recommender)
        freeze_decoder = auto_freeze_decoder if freeze_decoder < 0 else bool(freeze_decoder)

        imputer_module_trainable = not freeze_imputer and canonical_stage != "imputer_param"
        self._set_modules_trainable(self._imputer_modules(), imputer_module_trainable)
        self._set_modules_trainable(self._recommender_modules(), not freeze_recommender)
        if self.use_latent_completion_bridge and imputer_module_trainable and freeze_recommender:
            self._set_modules_trainable(self._projection_modules(), True)
        decoder_trainable = (
            self.use_decode_head
            and not bool(freeze_decoder)
            and canonical_stage in ("imputer_backprop", "recommender", "joint")
        )
        self._set_modules_trainable(self._decoder_modules(), decoder_trainable)
        if (
            bool(getattr(self.env.args, "completion_gate_only_train", 0))
            and self.use_rank_residual_completion_gate
            and canonical_stage in ("recommender", "joint")
        ):
            self._set_modules_trainable(self._recommender_modules(), False)
            if self.use_learned_completion_gate:
                self._set_modules_trainable([self.completion_gates], True)
            if self.use_global_rank_residual_completion_gate:
                self._set_modules_trainable([self.completion_gate_global_logits], True)
            if self.use_learned_completion_gate_mix:
                self._set_modules_trainable([self.completion_gate_mix_params], True)
        self._imputer_updates_enabled = (
            getattr(self.env.args, "generative_update_mode", "em") == "em"
            and canonical_stage in (
                "imputer_param",
                "joint",
            )
        )
        self.clear_gcn_cache()

    def _module_param_ids(self, modules):
        return {id(param) for module in modules for param in module.parameters()}

    def _trainable_parameters(self, modules, exclude_param_ids=None):
        exclude_param_ids = exclude_param_ids or set()
        params = []
        seen = set()
        for module in modules:
            for param in module.parameters():
                param_id = id(param)
                if param_id in exclude_param_ids or param_id in seen or not param.requires_grad:
                    continue
                params.append(param)
                seen.add(param_id)
        return params

    def get_imputer_parameters(self):
        exclude = set()
        if self.use_latent_direct_bridge and self._canonical_stage() in ("recommender", "joint"):
            exclude = self._module_param_ids(self._projection_modules())
        return self._trainable_parameters(self._imputer_modules(), exclude)

    def get_decoder_parameters(self):
        return self._trainable_parameters(self._decoder_modules())

    def get_recommender_parameters(self):
        exclude = set()
        if self.use_latent_direct_bridge and self._canonical_stage() not in ("recommender", "joint"):
            exclude = self._module_param_ids(self._projection_modules())
        return self._trainable_parameters(self._recommender_modules(), exclude)

    def load_full_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.load_state_dict(state_dict, strict=False)
        return len(state_dict)

    def load_projection_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        projection_prefixes = []
        if self.use_decoupled_latent_bridge:
            projection_prefixes.extend(
                f"comp_proj_{modality}" for modality in self.modalities
            )
        elif self.use_latent_direct_bridge:
            projection_prefixes.extend(
                f"{modality}_gcn.MLP" for modality in self.modalities
            )
        else:
            projection_prefixes.extend(
                f"contra_head_{modality}" for modality in self.modalities
            )

        current_state = self.state_dict()
        matched_state = {
            key: value
            for key, value in state_dict.items()
            if (
                key in current_state
                and value.shape == current_state[key].shape
                and any(
                    key == prefix or key.startswith(f"{prefix}.")
                    for prefix in projection_prefixes
                )
            )
        }
        expected_keys = {
            key
            for key in current_state
            if any(
                key == prefix or key.startswith(f"{prefix}.")
                for prefix in projection_prefixes
            )
        }
        missing_keys = sorted(expected_keys - matched_state.keys())
        if missing_keys:
            raise ValueError(
                "Projection checkpoint is incomplete or incompatible; "
                f"missing tensors={missing_keys}: {ckpt_path}"
            )
        current_state.update(matched_state)
        self.load_state_dict(current_state, strict=False)
        return sorted(matched_state.keys())

    def load_imputer_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        prefixes = [
            "itm_cross_attn",
            "itm_head",
            "W",
            "mu",
            "log_sigma",
        ]
        if self.use_decoupled_latent_bridge:
            prefixes.extend(f"comp_proj_{modality}" for modality in self.modalities)
        elif self.use_latent_direct_bridge:
            prefixes.extend(f"{modality}_gcn.MLP" for modality in self.modalities)
        else:
            for modality in self.modalities:
                prefixes.extend([f"contra_head_{modality}", f"decoder_{modality}"])

        current_state = self.state_dict()
        matched_state = {
            key: value
            for key, value in state_dict.items()
            if (
                key in current_state
                and value.shape == current_state[key].shape
                and any(key == prefix or key.startswith(f"{prefix}.") for prefix in prefixes)
            )
        }
        current_state.update(matched_state)
        self.load_state_dict(current_state, strict=False)
        return sorted(matched_state.keys())

    def init_missing_modality_set(self):
        self.train_missing_modality_items = self.dataset.train_missing_modality_items
        self.val_missing_modality_items = getattr(
            self.dataset,
            "val_missing_modality_items",
            {"items": [], "indicator": []},
        )
        self.eval_val_missing_modality_items = getattr(
            self.dataset,
            "eval_val_missing_modality_items",
            {"items": [], "indicator": []},
        )
        self.test_missing_modality_items = self.dataset.test_missing_modality_items
        (
            self.miss_train_image_feature,
            self.miss_train_text_feature,
            self.miss_train_audio_feature,
            self.miss_train_video_feature,
        ) = self._build_missing_feature_view(self.train_missing_modality_items)
        (
            self.miss_val_image_feature,
            self.miss_val_text_feature,
            self.miss_val_audio_feature,
            self.miss_val_video_feature,
        ) = self._build_missing_feature_view(self.val_missing_modality_items)
        (
            self.miss_eval_val_image_feature,
            self.miss_eval_val_text_feature,
            self.miss_eval_val_audio_feature,
            self.miss_eval_val_video_feature,
        ) = self._build_missing_feature_view(
            self.eval_val_missing_modality_items,
            image_base=self.eval_ori_image_feat,
            text_base=self.eval_ori_text_feat,
            audio_base=self.eval_ori_audio_feat if "a" in self.modalities else None,
            video_base=self.eval_ori_video_feat if "d" in self.modalities else None,
        )
        (
            self.miss_test_image_feature,
            self.miss_test_text_feature,
            self.miss_test_audio_feature,
            self.miss_test_video_feature,
        ) = self._build_missing_feature_view(
            self.test_missing_modality_items,
            image_base=self.eval_ori_image_feat,
            text_base=self.eval_ori_text_feat,
            audio_base=self.eval_ori_audio_feat if "a" in self.modalities else None,
            video_base=self.eval_ori_video_feat if "d" in self.modalities else None,
        )
        missing_all_metadata = {
            "items": np.concatenate([
                np.asarray(self.train_missing_modality_items.get("items", []), dtype=np.int64),
                np.asarray(self.eval_val_missing_modality_items.get("items", []), dtype=np.int64),
                np.asarray(self.test_missing_modality_items.get("items", []), dtype=np.int64),
            ]),
            "indicator": np.concatenate([
                np.asarray(self.train_missing_modality_items.get("indicator", []), dtype=np.int64),
                np.asarray(self.eval_val_missing_modality_items.get("indicator", []), dtype=np.int64),
                np.asarray(self.test_missing_modality_items.get("indicator", []), dtype=np.int64),
            ]),
        }
        (
            self.miss_cold_union_image_feature,
            self.miss_cold_union_text_feature,
            self.miss_cold_union_audio_feature,
            self.miss_cold_union_video_feature,
        ) = self._build_missing_feature_view(
            missing_all_metadata,
            image_base=self.eval_ori_image_feat,
            text_base=self.eval_ori_text_feat,
            audio_base=self.eval_ori_audio_feat if "a" in self.modalities else None,
            video_base=self.eval_ori_video_feat if "d" in self.modalities else None,
        )

    def refresh_dynamic_stage1_missing_views(self):
        self._dynamic_stage1_refresh_counter += 1
        dataset_seed = int(getattr(self.env.args, "dataset_seed", 0))
        dynamic_seed = dataset_seed + self._dynamic_stage1_refresh_counter
        self.dataset.refresh_stage1_dynamic_train_missing_metadata(seed=dynamic_seed)
        self.init_missing_modality_set()

    def _build_missing_feature_view(
        self,
        missing_metadata,
        image_base=None,
        text_base=None,
        audio_base=None,
        video_base=None,
    ):
        image_base = self.ori_image_feat if image_base is None else image_base
        text_base = self.ori_text_feat if text_base is None else text_base
        audio_base = self.ori_audio_feat if audio_base is None and "a" in self.modalities else audio_base
        video_base = self.ori_video_feat if video_base is None and "d" in self.modalities else video_base
        miss_image_feature = copy.deepcopy(image_base)
        miss_text_feature = copy.deepcopy(text_base)
        miss_audio_feature = copy.deepcopy(audio_base) if "a" in self.modalities else None
        miss_video_feature = copy.deepcopy(video_base) if "d" in self.modalities else None

        if self.modal_feature_override_is_completed and self.modal_feature_mask_source == "external_observed":
            return miss_image_feature, miss_text_feature, miss_audio_feature, miss_video_feature

        selected_missing_items = np.array(missing_metadata["items"], dtype=np.int64)
        selected_missing_modality_indicator = np.array(missing_metadata["indicator"], dtype=np.int64)
        if selected_missing_items.size == 0:
            return miss_image_feature, miss_text_feature, miss_audio_feature, miss_video_feature

        image_missing_indicator = selected_missing_items[selected_missing_modality_indicator == 0]
        if image_missing_indicator.size > 0:
            miss_image_feature[image_missing_indicator] = 0

        text_missing_indicator = selected_missing_items[selected_missing_modality_indicator == 1]
        if text_missing_indicator.size > 0:
            miss_text_feature[text_missing_indicator] = 0

        if "a" in self.modalities:
            audio_missing_indicator = selected_missing_items[selected_missing_modality_indicator == 2]
            if audio_missing_indicator.size > 0:
                miss_audio_feature[audio_missing_indicator] = 0
        if "d" in self.modalities:
            video_missing_indicator = selected_missing_items[selected_missing_modality_indicator == 3]
            if video_missing_indicator.size > 0:
                miss_video_feature[video_missing_indicator] = 0

        return miss_image_feature, miss_text_feature, miss_audio_feature, miss_video_feature

    def set_missing_modality_via_env(self, eval_split=None):
        mode = self.env.args.exp_mode
        canonical_stage = self._canonical_stage()

        use_missing_train = mode in {"mm", "mf"}
        use_missing_test = mode in {"mm", "fm"}

        if self.training:
            if use_missing_train:
                self.image_feat = self.miss_train_image_feature
                self.text_feat = self.miss_train_text_feature
                if "a" in self.modalities:
                    self.audio_feat = self.miss_train_audio_feature
                if "d" in self.modalities:
                    self.video_feat = self.miss_train_video_feature
                print("set missing modality successfully for train setp")
            else:
                self.image_feat = self.ori_image_feat
                self.text_feat = self.ori_text_feat
                if "a" in self.modalities:
                    self.audio_feat = self.ori_audio_feat
                if "d" in self.modalities:
                    self.video_feat = self.ori_video_feat
                print("set complete modality successfully for train step")
        else:
            if use_missing_test:
                if (
                    getattr(self.dataset, "cold_start_protocol", "none") == "milk"
                    and getattr(self.env.args, "cold_start_eval_candidates", "milk_union") == "milk_union"
                ):
                    self.image_feat = self.miss_cold_union_image_feature
                    self.text_feat = self.miss_cold_union_text_feature
                    if "a" in self.modalities:
                        self.audio_feat = self.miss_cold_union_audio_feature
                    if "d" in self.modalities:
                        self.video_feat = self.miss_cold_union_video_feature
                    print("set MILK missing-all modality view for evaluation")
                    return
                if eval_split == "val":
                    self.image_feat = self.miss_eval_val_image_feature
                    self.text_feat = self.miss_eval_val_text_feature
                    if "a" in self.modalities:
                        self.audio_feat = self.miss_eval_val_audio_feature
                    if "d" in self.modalities:
                        self.video_feat = self.miss_eval_val_video_feature
                    print("set missing modality successfully for val setp")
                    return

                self.image_feat = self.miss_test_image_feature
                self.text_feat = self.miss_test_text_feature
                if "a" in self.modalities:
                    self.audio_feat = self.miss_test_audio_feature
                if "d" in self.modalities:
                    self.video_feat = self.miss_test_video_feature
                print("set missing modality successfully for test setp")
            else:
                self.image_feat = self.eval_ori_image_feat
                self.text_feat = self.eval_ori_text_feat
                if "a" in self.modalities:
                    self.audio_feat = self.eval_ori_audio_feat
                if "d" in self.modalities:
                    self.video_feat = self.eval_ori_video_feat
                print("set complete modality successfully for test step")

    def _current_raw_modal_features(self, full=False):
        if full:
            if not self.training and self.uses_split_modal_feature_override:
                features = {"v": self.eval_ori_image_feat, "t": self.eval_ori_text_feat}
            else:
                features = {"v": self.ori_image_feat, "t": self.ori_text_feat}
        else:
            features = {"v": self.image_feat, "t": self.text_feat}
        if "a" in self.modalities:
            if full:
                if not self.training and self.uses_split_modal_feature_override:
                    features["a"] = self.eval_ori_audio_feat
                else:
                    features["a"] = self.ori_audio_feat
            else:
                features["a"] = self.audio_feat
        if "d" in self.modalities:
            if full:
                if not self.training and self.uses_split_modal_feature_override:
                    features["d"] = self.eval_ori_video_feat
                else:
                    features["d"] = self.ori_video_feat
            else:
                features["d"] = self.video_feat
        return features

    def get_split_raw_modal_features(self, split="test", full=False):
        if full:
            use_eval_features = split in {"val", "test"} and self.uses_split_modal_feature_override
            features = {
                "v": self.eval_ori_image_feat if use_eval_features else self.ori_image_feat,
                "t": self.eval_ori_text_feat if use_eval_features else self.ori_text_feat,
            }
            if "a" in self.modalities:
                features["a"] = self.eval_ori_audio_feat if use_eval_features else self.ori_audio_feat
            if "d" in self.modalities:
                features["d"] = self.eval_ori_video_feat if use_eval_features else self.ori_video_feat
            return features

        if split == "train":
            features = {"v": self.miss_train_image_feature, "t": self.miss_train_text_feature}
            if "a" in self.modalities:
                features["a"] = self.miss_train_audio_feature
            if "d" in self.modalities:
                features["d"] = self.miss_train_video_feature
            return features
        if split == "val":
            features = {"v": self.miss_eval_val_image_feature, "t": self.miss_eval_val_text_feature}
            if "a" in self.modalities:
                features["a"] = self.miss_eval_val_audio_feature
            if "d" in self.modalities:
                features["d"] = self.miss_eval_val_video_feature
            return features
        if split == "imputation_val":
            features = {"v": self.miss_val_image_feature, "t": self.miss_val_text_feature}
            if "a" in self.modalities:
                features["a"] = self.miss_val_audio_feature
            if "d" in self.modalities:
                features["d"] = self.miss_val_video_feature
            return features
        if split == "test":
            features = {"v": self.miss_test_image_feature, "t": self.miss_test_text_feature}
            if "a" in self.modalities:
                features["a"] = self.miss_test_audio_feature
            if "d" in self.modalities:
                features["d"] = self.miss_test_video_feature
            return features
        raise ValueError(f"Unsupported split: {split}")

    def get_missing_item_metadata(self, split="test"):
        if split == "train":
            metadata = self.train_missing_modality_items
        elif split == "val":
            metadata = self.eval_val_missing_modality_items
        elif split == "imputation_val":
            metadata = self.val_missing_modality_items
        elif split == "test":
            metadata = self.test_missing_modality_items
        else:
            raise ValueError(f"Unsupported split: {split}")

        items = torch.as_tensor(metadata["items"], dtype=torch.long, device=self.env.device)
        indicators = torch.as_tensor(metadata["indicator"], dtype=torch.long, device=self.env.device)
        return items, indicators

    def _completed_item_graph_missing_metadata(self):
        scope = getattr(self.env.args, "item_graph_missing_scope", "train")
        if scope == "all":
            return [
                self.train_missing_modality_items,
                self.val_missing_modality_items,
                self.eval_val_missing_modality_items,
                self.test_missing_modality_items,
            ]
        if scope == "train":
            return [
                self.train_missing_modality_items,
                self.val_missing_modality_items,
            ]
        raise ValueError(f"Unsupported item_graph_missing_scope: {scope}")

    def _combined_missing_raw_modal_features(self):
        features = {
            "v": self.ori_image_feat.clone(),
            "t": self.ori_text_feat.clone(),
        }
        if "a" in self.modalities:
            features["a"] = self.ori_audio_feat.clone()
        if "d" in self.modalities:
            features["d"] = self.ori_video_feat.clone()

        for metadata in self._completed_item_graph_missing_metadata():
            if metadata is None:
                continue
            items = np.asarray(metadata.get("items", []), dtype=np.int64)
            indicators = np.asarray(metadata.get("indicator", []), dtype=np.int64)
            if items.size == 0:
                continue
            for idx, modality in enumerate(self.modalities):
                selected = items[indicators == idx]
                if selected.size > 0:
                    selected_tensor = torch.as_tensor(selected, dtype=torch.long, device=self.env.device)
                    features[modality][selected_tensor] = 0.0
        return features

    def build_completed_item_graph(self):
        if not self.use_completed_item_graph:
            return

        kind = getattr(self.env.args, "item_graph_kind", "cf")
        if kind not in (
            "modality_masked",
            "fused_completed",
            "modality_completed",
            "modality_completed_inductive",
            "modality_completed_confidence",
            "modality_completed_dynamic_confidence",
            "fused_completed_confidence",
            "fused_completed_dynamic_confidence",
            "fused_completed_reliability",
            "fused_completed_reliability_topk",
        ):
            raise ValueError(f"Unsupported completed item graph kind: {kind}")

        topk = int(getattr(self.env.args, "item_graph_topk", 20))
        norm_type = getattr(self.env.args, "item_graph_norm", "rw")
        chunk_size = int(getattr(self.env.args, "item_graph_feature_chunk_size", 1024))
        missing_scope = getattr(self.env.args, "item_graph_missing_scope", "train")
        graph_label = "masked" if kind == "modality_masked" else "completed"
        print(f"building {graph_label} item graph with missing_scope={missing_scope}")

        was_training = self.training
        self.eval()
        with torch.no_grad():
            raw_features = self._combined_missing_raw_modal_features()
            masks = self._missing_masks(raw_features=raw_features)
            graph_feature_space = getattr(self.env.args, "item_graph_feature_space", "shared")
            if kind == "modality_masked":
                graph_feature_space = "raw_masked"
                graph_features = raw_features
            elif graph_feature_space in ("shared", "raw_decoder"):
                projected = self.project_features(raw_features=raw_features)
                graph_features = self._build_completed_features(
                    projected,
                    masks,
                    detach_imputed=True,
                )
                if graph_feature_space == "raw_decoder":
                    if self.use_latent_completion_bridge:
                        graph_feature_space = "shared"
                    else:
                        decoded_raw = self.bridge_completed_to_recommendation_raw(graph_features)
                        graph_features = {
                            modality: torch.where(
                                masks[modality].unsqueeze(1),
                                raw_features[modality],
                                decoded_raw[modality],
                            )
                            for modality in self.modalities
                        }
            else:
                raise ValueError(f"Unsupported item graph feature space: {graph_feature_space}")
            graph_feature_np = {
                modality: graph_features[modality].detach().cpu().numpy().astype(np.float32)
                for modality in self.modalities
            }
        if was_training:
            self.train()

        weights = {
            "cf": float(getattr(self.env.args, "item_graph_cf_weight", 0.5)),
            "image": float(getattr(self.env.args, "item_graph_image_weight", 0.25)),
            "text": float(getattr(self.env.args, "item_graph_text_weight", 0.25)),
            "audio": float(getattr(self.env.args, "item_graph_audio_weight", 0.0)),
            "video": float(getattr(self.env.args, "item_graph_video_weight", 0.0)),
        }
        cf_scale = getattr(self.env.args, "item_graph_cf_scale", "raw")
        cf_power = float(getattr(self.env.args, "item_graph_cf_power", 0.5))
        cf_clip = float(getattr(self.env.args, "item_graph_cf_clip", 3.0))
        uses_reliability = kind in ("fused_completed_reliability", "fused_completed_reliability_topk")
        if uses_reliability:
            reliabilities = self._compute_item_graph_reliability_scores(graph_features, masks)
        elif self._uses_posterior_reliability_for("graph"):
            reliabilities = {
                modality: scores.detach().cpu().numpy().astype(np.float32)
                for modality, scores in self._posterior_completion_reliabilities(masks).items()
            }
        else:
            reliabilities = None
        reliability_blend = min(max(float(getattr(self.env.args, "item_graph_reliability_blend", 1.0)), 0.0), 1.0)
        reliability_topk = (
            kind == "fused_completed_reliability_topk"
            or self._uses_posterior_reliability_for("graph")
        )
        fuse_before_topk = bool(int(getattr(self.env.args, "item_graph_fuse_before_topk", 0)))
        if fuse_before_topk and kind != "modality_completed":
            raise ValueError(
                "item_graph_fuse_before_topk=1 currently requires item_graph_kind=modality_completed"
            )
        full_cf_graph = None
        if fuse_before_topk:
            full_cf_graph = self.dataset._build_cf_item_similarity(
                scale=cf_scale,
                power=cf_power,
                clip=cf_clip,
            )
            cf_graph = self.dataset._topk_sparse_rows(full_cf_graph, topk)
        else:
            cf_graph = self.dataset._build_cf_item_graph(
                topk,
                scale=cf_scale,
                power=cf_power,
                clip=cf_clip,
            )
        graphs = {"cf": cf_graph}
        inductive_kind = kind == "modality_completed_inductive"
        warm_items = np.asarray(getattr(self.dataset, "train_item_index", []), dtype=np.int64)

        def build_semantic_graph(feature, reliability):
            if inductive_kind:
                return self.dataset._build_inductive_feature_item_graph(
                    feature,
                    reference_items=warm_items,
                    query_items=warm_items,
                    topk=topk,
                    chunk_size=chunk_size,
                    reliability=reliability,
                    reliability_blend=reliability_blend,
                )
            return self.dataset._build_feature_item_graph(
                feature,
                topk,
                chunk_size,
                reliability=reliability,
                reliability_blend=reliability_blend,
            )

        if weights["image"] > 0.0 and not fuse_before_topk:
            graphs["image"] = build_semantic_graph(
                graph_feature_np["v"],
                reliabilities["v"] if reliability_topk else None,
            )
        if weights["text"] > 0.0 and not fuse_before_topk:
            graphs["text"] = build_semantic_graph(
                graph_feature_np["t"],
                reliabilities["t"] if reliability_topk else None,
            )
        if "a" in self.modalities and weights["audio"] > 0.0 and not fuse_before_topk:
            graphs["audio"] = build_semantic_graph(
                graph_feature_np["a"],
                reliabilities["a"] if reliability_topk else None,
            )
        if "d" in self.modalities and weights["video"] > 0.0 and not fuse_before_topk:
            graphs["video"] = build_semantic_graph(
                graph_feature_np["d"],
                reliabilities["d"] if reliability_topk else None,
            )
        if kind == "modality_masked":
            masked_graph_specs = (("image", "v"), ("text", "t"), ("audio", "a"), ("video", "d"))
            for graph_name, modality in masked_graph_specs:
                if graph_name not in graphs or modality not in masks:
                    continue
                missing = (~masks[modality]).detach().cpu().numpy()
                if graphs[graph_name][missing].nnz or graphs[graph_name][:, missing].nnz:
                    raise RuntimeError(
                        f"modality_masked graph contains semantic edges involving missing {modality} items"
                    )
                print(
                    f"strict masked graph modality={modality} missing_items={int(missing.sum())} "
                    f"semantic_edges={graphs[graph_name].nnz} missing_semantic_edges=0"
                )
        if uses_reliability:
            reliability_specs = (("image", "v"), ("text", "t"), ("audio", "a"), ("video", "d"))
            if not reliability_topk:
                for graph_name, modality in reliability_specs:
                    if graph_name not in graphs or modality not in reliabilities:
                        continue
                    graphs[graph_name] = self._apply_item_graph_reliability(
                        graphs[graph_name],
                        reliabilities[modality],
                    )
            graph = self._build_weighted_item_item_graph(
                graphs,
                weights,
                topk,
                norm_type,
                required_names=None,
                context="reliability completed item graph",
            )
            self.ItemItemGraph = self.dataset._convert_sp_mat_to_sp_tensor(graph).coalesce().to(self.env.device)
            self.ItemItemGraphs = {}
            self.ItemItemGraphComponents = {}
            rel_desc = ",".join(
                f"{modality}:mean={scores.mean():.4f}/min={scores.min():.4f}/missing_mean="
                f"{scores[(~masks[modality]).detach().cpu().numpy()].mean():.4f}"
                if np.any((~masks[modality]).detach().cpu().numpy())
                else f"{modality}:mean={scores.mean():.4f}/min={scores.min():.4f}/missing_mean=NA"
                for modality, scores in sorted(reliabilities.items())
            )
            print(
                f"built completed item-item graph kind={kind}, topk={topk}, norm={norm_type}, "
                f"feature_space={graph_feature_space}, "
                f"cf_scale={cf_scale},cf_power={cf_power},cf_clip={cf_clip}, "
                f"weights=cf:{weights['cf']},image:{weights['image']},text:{weights['text']},audio:{weights['audio']},video:{weights['video']}, "
                f"reliability_floor={float(getattr(self.env.args, 'item_graph_reliability_floor', 0.4))}, "
                f"reliability_blend={float(getattr(self.env.args, 'item_graph_reliability_blend', 1.0))}, "
                f"reliability_missing_penalty={float(getattr(self.env.args, 'item_graph_reliability_missing_penalty', 1.0))}, "
                f"reliability_missing_boost={float(getattr(self.env.args, 'item_graph_reliability_missing_boost', 0.0))}, "
                f"reliability_topk={int(reliability_topk)}, "
                f"reliability={rel_desc}"
            )
            return
        if kind == "fused_completed_dynamic_confidence":
            dynamic_candidates = self._build_dynamic_confidence_item_graph_candidates(
                graphs,
                weights,
                masks,
                topk,
            )
            self.ItemItemGraphDynamicCandidates = {
                name: value.to(self.env.device)
                for name, value in dynamic_candidates.items()
            }
            self.ItemItemGraphComponents = {}
            self.ItemItemGraph = None
            self.ItemItemGraphs = {}
            self.item_graph_dynamic_norm_type = norm_type
            self.item_graph_dynamic_topk = topk
            conf = self._item_graph_edge_confidences().detach().cpu().numpy()
            coeff = self._item_graph_edge_confidence_coeffs().detach().cpu().numpy()
            print(
                f"built completed item-item graph kind={kind}, topk={topk}, norm={norm_type}, "
                f"dynamic_topk=1, candidate_width={dynamic_candidates['cols'].shape[1]}, "
                f"feature_space={getattr(self.env.args, 'item_graph_feature_space', 'shared')}, "
                f"cf_scale={cf_scale},cf_power={cf_power},cf_clip={cf_clip}, "
                f"weights=cf:{weights['cf']},image:{weights['image']},text:{weights['text']},audio:{weights['audio']},video:{weights['video']}, "
                f"edge_conf_init=rr:{conf[0]:.4f},ri:{conf[1]:.4f},ii:{conf[2]:.4f}, "
                f"edge_conf_coeff=rr:{coeff[0]:.4f},ri:{coeff[1]:.4f},ii:{coeff[2]:.4f}, "
                f"transform={self.item_graph_confidence_transform}, "
                f"blend={self.item_graph_confidence_blend:.4f}, "
                f"score_blend={self.item_graph_dynamic_score_blend:.4f}, "
                f"neighbor_blend={self.item_graph_dynamic_neighbor_blend:.4f}, "
                f"range=[{self.item_graph_confidence_min:.4f},{self.item_graph_confidence_max:.4f}]"
            )
            return

        if kind == "fused_completed_confidence":
            component_graphs = self._build_confidence_item_item_components(
                graphs,
                weights,
                masks,
                topk,
                norm_type,
            )
            self.ItemItemGraphComponents = {
                name: self.dataset._convert_sp_mat_to_sp_tensor(graph).coalesce().to(self.env.device)
                for name, graph in component_graphs.items()
            }
            self.ItemItemGraph = None
            self.ItemItemGraphs = {}
            self.item_graph_dynamic_norm_type = norm_type
            conf = self._item_graph_edge_confidences().detach().cpu().numpy()
            coeff = self._item_graph_edge_confidence_coeffs().detach().cpu().numpy()
            print(
                f"built completed item-item graph kind={kind}, topk={topk}, norm={norm_type}, "
                f"components={','.join(sorted(self.ItemItemGraphComponents.keys()))}, "
                f"feature_space={getattr(self.env.args, 'item_graph_feature_space', 'shared')}, "
                f"cf_scale={cf_scale},cf_power={cf_power},cf_clip={cf_clip}, "
                f"weights=cf:{weights['cf']},image:{weights['image']},text:{weights['text']},audio:{weights['audio']},video:{weights['video']}, "
                f"edge_conf_init=rr:{conf[0]:.4f},ri:{conf[1]:.4f},ii:{conf[2]:.4f}, "
                f"edge_conf_coeff=rr:{coeff[0]:.4f},ri:{coeff[1]:.4f},ii:{coeff[2]:.4f}, "
                f"transform={self.item_graph_confidence_transform}, "
                f"blend={self.item_graph_confidence_blend:.4f}, "
                f"range=[{self.item_graph_confidence_min:.4f},{self.item_graph_confidence_max:.4f}]"
            )
            return

        if kind in ("modality_masked", "modality_completed", "modality_completed_inductive"):
            self.ItemItemGraphs = {}
            self.ItemItemGraphComponents = {}
            if fuse_before_topk:
                modality_specs = {
                    "v": ("v", weights["image"]),
                    "t": ("t", weights["text"]),
                }
                if "a" in self.modalities:
                    modality_specs["a"] = ("a", weights["audio"])
                if "d" in self.modalities:
                    modality_specs["d"] = ("d", weights["video"])

                cf_only = self._build_weighted_item_item_graph(
                    {"cf": graphs["cf"]},
                    {"cf": weights["cf"]},
                    topk,
                    norm_type,
                    required_names=["cf"],
                    context=f"{graph_label} cf item graph",
                )
                self.ItemItemGraphs["cf"] = (
                    self.dataset._convert_sp_mat_to_sp_tensor(cf_only).coalesce().to(self.env.device)
                )
                for key, (modality, feature_weight) in modality_specs.items():
                    if feature_weight <= 0.0:
                        continue
                    reliability = (
                        reliabilities[modality]
                        if reliability_topk and reliabilities is not None
                        else None
                    )
                    graph = self.dataset._build_fused_cf_feature_item_graph(
                        graph_feature_np[modality],
                        full_cf_graph,
                        topk,
                        chunk_size,
                        weights["cf"],
                        feature_weight,
                        reliability=reliability,
                        reliability_blend=reliability_blend,
                    )
                    graph = self.dataset._normalize_item_graph(graph, norm_type)
                    self.ItemItemGraphs[key] = (
                        self.dataset._convert_sp_mat_to_sp_tensor(graph).coalesce().to(self.env.device)
                    )
                self.ItemItemGraph = None
                print(
                    f"built {graph_label} item-item graph kind={kind}, topk={topk}, norm={norm_type}, "
                    f"graphs={','.join(sorted(self.ItemItemGraphs.keys()))}, "
                    f"feature_space={graph_feature_space}, fusion_order=fuse_then_topk, "
                    f"cf_scale={cf_scale},cf_power={cf_power},cf_clip={cf_clip}, "
                    f"weights=cf:{weights['cf']},image:{weights['image']},text:{weights['text']},"
                    f"audio:{weights['audio']},video:{weights['video']}"
                )
                return
            def modality_graph_parts(feature_graph_name):
                graph_parts = {"cf": graphs["cf"]}
                graph_weights = {"cf": weights["cf"]}
                if feature_graph_name in graphs and weights[feature_graph_name] > 0.0:
                    graph_parts[feature_graph_name] = graphs[feature_graph_name]
                    graph_weights[feature_graph_name] = weights[feature_graph_name]
                return graph_parts, graph_weights

            modality_graph_specs = {
                "cf": ({"cf": graphs["cf"]}, {"cf": weights["cf"]}),
                "v": modality_graph_parts("image"),
                "t": modality_graph_parts("text"),
            }
            if "a" in self.modalities and "audio" in graphs:
                modality_graph_specs["a"] = (
                    {"cf": graphs["cf"], "audio": graphs["audio"]},
                    {"cf": weights["cf"], "audio": weights["audio"]},
                )
            if "d" in self.modalities and "video" in graphs:
                modality_graph_specs["d"] = (
                    {"cf": graphs["cf"], "video": graphs["video"]},
                    {"cf": weights["cf"], "video": weights["video"]},
                )
            for key, (graph_parts, graph_weights) in modality_graph_specs.items():
                graph = self._build_weighted_item_item_graph(
                    graph_parts,
                    graph_weights,
                    topk,
                    norm_type,
                    required_names=list(graph_parts.keys()),
                    context=f"{graph_label} {key} item graph",
                )
                if kind == "modality_completed_inductive":
                    cold_items = np.asarray(self.dataset.cold_item_index, dtype=np.int64)
                    if graph[cold_items].nnz or graph[:, cold_items].nnz:
                        raise RuntimeError(
                            f"inductive training item graph {key} contains an edge involving a cold item"
                        )
                self.ItemItemGraphs[key] = self.dataset._convert_sp_mat_to_sp_tensor(graph).coalesce().to(self.env.device)
            self.ItemItemGraph = None
            if kind == "modality_completed_inductive":
                self.InductiveItemItemGraphs["train"] = self.ItemItemGraphs
                self._inductive_item_graph_split = "train"
            print(
                f"built {graph_label} item-item graph kind={kind}, topk={topk}, norm={norm_type}, "
                f"graphs={','.join(sorted(self.ItemItemGraphs.keys()))}, "
                f"feature_space={graph_feature_space}, "
                f"cf_scale={cf_scale},cf_power={cf_power},cf_clip={cf_clip}, "
                f"weights=cf:{weights['cf']},image:{weights['image']},text:{weights['text']},audio:{weights['audio']},video:{weights['video']}"
            )
            if self._uses_posterior_reliability_for("graph"):
                rel_desc = ",".join(
                    f"{modality}:mean={scores.mean():.4f}/min={scores.min():.4f}"
                    for modality, scores in sorted(reliabilities.items())
                )
                print(
                    "posterior reliability applied before semantic topk: "
                    f"scale={self.posterior_reliability_scale},floor={self.posterior_reliability_floor},"
                    f"{rel_desc}"
                )
            return

        if kind == "modality_completed_confidence":
            self.ItemItemGraphs = {}
            self.ItemItemGraphComponents = {}
            self.item_graph_dynamic_norm_type = norm_type
            modality_graph_specs = {
                "cf": ("image", "v"),
                "v": ("image", "v"),
                "t": ("text", "t"),
            }
            if "a" in self.modalities and "audio" in graphs:
                modality_graph_specs["a"] = ("audio", "a")
            if "d" in self.modalities and "video" in graphs:
                modality_graph_specs["d"] = ("video", "d")

            effective_weights = {}
            for key, (feature_graph_name, modality_name) in modality_graph_specs.items():
                if feature_graph_name not in graphs:
                    continue
                cf_weight = weights["cf"]
                feature_weight = weights[feature_graph_name]
                if cf_weight <= 0.0:
                    feature_weight = 1.0
                components = self._build_confidence_item_item_graph_for_modality(
                    graphs["cf"],
                    graphs[feature_graph_name],
                    masks[modality_name],
                    topk,
                    norm_type,
                    cf_weight,
                    feature_weight,
                    key,
                )
                effective_weights[key] = (cf_weight, feature_weight)
                self.ItemItemGraphs[key] = None
                self.ItemItemGraphComponents[key] = {
                    name: self.dataset._convert_sp_mat_to_sp_tensor(graph).coalesce().to(self.env.device)
                    for name, graph in components.items()
                }

            if not self.ItemItemGraphComponents:
                raise ValueError("modality_completed_confidence has no supported modality graphs")
            print(
                f"built completed item-item graph kind={kind}, topk={topk}, norm={norm_type}, "
                f"graphs={','.join(sorted(self.ItemItemGraphComponents.keys()))}, "
                f"feature_space={getattr(self.env.args, 'item_graph_feature_space', 'shared')}, "
                f"cf_scale={cf_scale},cf_power={cf_power},cf_clip={cf_clip}, "
                f"weights=cf:{weights['cf']},image:{weights['image']},text:{weights['text']},audio:{weights['audio']},video:{weights['video']}, "
                f"effective_modal_weights="
                f"{','.join(f'{name}:cf={vals[0]},semantic={vals[1]}' for name, vals in sorted(effective_weights.items()))}"
            )
            return

        if kind == "modality_completed_dynamic_confidence":
            self.ItemItemGraphs = {}
            self.ItemItemGraphComponents = {}
            self.ItemItemGraphDynamicCandidates = {}
            self.item_graph_dynamic_norm_type = norm_type
            self.item_graph_dynamic_topk = topk
            modality_graph_specs = {
                "v": ("image", "v"),
                "t": ("text", "t"),
            }
            if "a" in self.modalities and "audio" in graphs:
                modality_graph_specs["a"] = ("audio", "a")
            if "d" in self.modalities and "video" in graphs:
                modality_graph_specs["d"] = ("video", "d")

            effective_weights = {}
            candidate_widths = {}
            for key, (feature_graph_name, modality_name) in modality_graph_specs.items():
                if feature_graph_name not in graphs:
                    continue
                cf_weight = weights["cf"]
                feature_weight = weights[feature_graph_name]
                if cf_weight <= 0.0:
                    feature_weight = 1.0
                candidates = self._build_dynamic_confidence_item_graph_candidates_for_modality(
                    graphs["cf"],
                    graphs[feature_graph_name],
                    masks[modality_name],
                    cf_weight,
                    feature_weight,
                )
                effective_weights[key] = (cf_weight, feature_weight)
                candidate_widths[key] = candidates["cols"].shape[1]
                self.ItemItemGraphDynamicCandidates[key] = {
                    name: value.to(self.env.device)
                    for name, value in candidates.items()
                }

            if not self.ItemItemGraphDynamicCandidates:
                raise ValueError("modality_completed_dynamic_confidence has no supported modality graphs")
            conf = self._item_graph_edge_confidences()
            coeff = self._item_graph_edge_confidence_coeffs()
            if isinstance(conf, dict):
                edge_conf_desc = ",".join(
                    f"{name}:rr={conf[name][0].detach().cpu().item():.4f}/"
                    f"ri={conf[name][1].detach().cpu().item():.4f}/"
                    f"ii={conf[name][2].detach().cpu().item():.4f}"
                    for name in sorted(self.ItemItemGraphDynamicCandidates.keys())
                )
                edge_coeff_desc = ",".join(
                    f"{name}:rr={coeff[name][0].detach().cpu().item():.4f}/"
                    f"ri={coeff[name][1].detach().cpu().item():.4f}/"
                    f"ii={coeff[name][2].detach().cpu().item():.4f}"
                    for name in sorted(self.ItemItemGraphDynamicCandidates.keys())
                )
            else:
                conf_np = conf.detach().cpu().numpy()
                coeff_np = coeff.detach().cpu().numpy()
                edge_conf_desc = f"rr:{conf_np[0]:.4f},ri:{conf_np[1]:.4f},ii:{conf_np[2]:.4f}"
                edge_coeff_desc = f"rr:{coeff_np[0]:.4f},ri:{coeff_np[1]:.4f},ii:{coeff_np[2]:.4f}"
            print(
                f"built completed item-item graph kind={kind}, topk={topk}, norm={norm_type}, "
                f"dynamic_topk=1, graphs={','.join(sorted(self.ItemItemGraphDynamicCandidates.keys()))}, "
                f"candidate_widths="
                f"{','.join(f'{name}:{width}' for name, width in sorted(candidate_widths.items()))}, "
                f"feature_space={getattr(self.env.args, 'item_graph_feature_space', 'shared')}, "
                f"cf_scale={cf_scale},cf_power={cf_power},cf_clip={cf_clip}, "
                f"weights=cf:{weights['cf']},image:{weights['image']},text:{weights['text']},audio:{weights['audio']},video:{weights['video']}, "
                f"effective_modal_weights="
                f"{','.join(f'{name}:cf={vals[0]},semantic={vals[1]}' for name, vals in sorted(effective_weights.items()))}, "
                f"edge_conf_init={edge_conf_desc}, "
                f"edge_conf_coeff={edge_coeff_desc}, "
                f"transform={self.item_graph_confidence_transform}, "
                f"blend={self.item_graph_confidence_blend:.4f}, "
                f"score_blend={self.item_graph_dynamic_score_blend:.4f}, "
                f"neighbor_blend={self.item_graph_dynamic_neighbor_blend:.4f}, "
                f"range=[{self.item_graph_confidence_min:.4f},{self.item_graph_confidence_max:.4f}]"
            )
            return

        graph = self._build_weighted_item_item_graph(
            graphs,
            weights,
            topk,
            norm_type,
            required_names=None,
            context="completed item graph",
        )
        self.ItemItemGraph = self.dataset._convert_sp_mat_to_sp_tensor(graph).coalesce().to(self.env.device)
        self.ItemItemGraphs = {}
        self.ItemItemGraphComponents = {}
        print(
            f"built completed item-item graph kind={kind}, topk={topk}, norm={norm_type}, "
            f"feature_space={getattr(self.env.args, 'item_graph_feature_space', 'shared')}, "
            f"cf_scale={cf_scale},cf_power={cf_power},cf_clip={cf_clip}, "
            f"weights=cf:{weights['cf']},image:{weights['image']},text:{weights['text']},audio:{weights['audio']},video:{weights['video']}"
        )

    def set_inductive_item_graph_split(self, split):
        """Switch between the warm training graph and cold-query inference graph."""
        if self.item_graph_kind != "modality_completed_inductive":
            return
        if split == "train":
            self.ItemItemGraphs = self.InductiveItemItemGraphs["train"]
            self._inductive_item_graph_split = "train"
            self.clear_gcn_cache()
            return
        if split not in ("val", "test"):
            raise ValueError(f"Unsupported inductive item graph split: {split}")

        candidate_protocol = getattr(self.env.args, "cold_start_eval_candidates", "milk_union")
        cache_key = "milk_union" if candidate_protocol == "milk_union" else split
        if cache_key in self.InductiveItemItemGraphs:
            self.ItemItemGraphs = self.InductiveItemItemGraphs[cache_key]
            self._inductive_item_graph_split = cache_key
            self.clear_gcn_cache()
            return

        topk = int(getattr(self.env.args, "item_graph_topk", 20))
        norm_type = getattr(self.env.args, "item_graph_norm", "rw")
        chunk_size = int(getattr(self.env.args, "item_graph_feature_chunk_size", 1024))
        warm_items = np.asarray(self.dataset.train_item_index, dtype=np.int64)
        cold_queries = np.asarray(self.dataset.get_eval_candidate_items(split), dtype=np.int64)
        active_queries = np.unique(np.concatenate([warm_items, cold_queries]))
        raw_features = self._current_raw_modal_features()
        with torch.no_grad():
            masks = self._missing_masks(raw_features=raw_features)
            projected = self.project_features(raw_features=raw_features)
            graph_features = self._build_completed_features(
                projected,
                masks,
                detach_imputed=True,
            )
            graph_feature_space = getattr(self.env.args, "item_graph_feature_space", "shared")
            if graph_feature_space == "raw_decoder" and not self.use_latent_completion_bridge:
                decoded_raw = self.bridge_completed_to_recommendation_raw(graph_features)
                graph_features = {
                    modality: torch.where(
                        masks[modality].unsqueeze(1),
                        raw_features[modality],
                        decoded_raw[modality],
                    )
                    for modality in self.modalities
                }
            graph_feature_np = {
                modality: graph_features[modality].detach().cpu().numpy().astype(np.float32)
                for modality in self.modalities
            }
            if self.use_posterior_reliability:
                reliabilities = {
                    modality: scores.detach().cpu().numpy().astype(np.float32)
                    for modality, scores in self._posterior_completion_reliabilities(masks).items()
                }
            else:
                reliabilities = None

        weights = {
            "cf": float(getattr(self.env.args, "item_graph_cf_weight", 0.5)),
            "image": float(getattr(self.env.args, "item_graph_image_weight", 0.25)),
            "text": float(getattr(self.env.args, "item_graph_text_weight", 0.25)),
            "audio": float(getattr(self.env.args, "item_graph_audio_weight", 0.0)),
            "video": float(getattr(self.env.args, "item_graph_video_weight", 0.0)),
        }
        cf_graph = self.dataset._build_cf_item_graph(
            topk,
            scale=getattr(self.env.args, "item_graph_cf_scale", "raw"),
            power=float(getattr(self.env.args, "item_graph_cf_power", 0.5)),
            clip=float(getattr(self.env.args, "item_graph_cf_clip", 3.0)),
        )
        modality_specs = (("v", "image"), ("t", "text"), ("a", "audio"), ("d", "video"))
        item_graphs = {}
        reliability_blend = min(
            max(float(getattr(self.env.args, "item_graph_reliability_blend", 1.0)), 0.0),
            1.0,
        )
        for modality, graph_name in modality_specs:
            if modality not in self.modalities or weights[graph_name] <= 0.0:
                continue
            semantic_graph = self.dataset._build_inductive_feature_item_graph(
                graph_feature_np[modality],
                reference_items=warm_items,
                query_items=active_queries,
                topk=topk,
                chunk_size=chunk_size,
                reliability=reliabilities[modality] if reliabilities is not None else None,
                reliability_blend=reliability_blend,
            )
            graph = self._build_weighted_item_item_graph(
                {"cf": cf_graph, graph_name: semantic_graph},
                {"cf": weights["cf"], graph_name: weights[graph_name]},
                topk,
                norm_type,
                required_names=["cf", graph_name],
                context=f"inductive {split} {modality} item graph",
            )
            all_cold_items = np.asarray(self.dataset.cold_item_index, dtype=np.int64)
            if graph[:, all_cold_items].nnz:
                raise RuntimeError(
                    f"inductive evaluation item graph {modality} contains warm-to-cold or cold-to-cold edges"
                )
            item_graphs[modality] = self.dataset._convert_sp_mat_to_sp_tensor(graph).coalesce().to(self.env.device)
        item_graphs["cf"] = self.dataset._convert_sp_mat_to_sp_tensor(
            self._build_single_item_item_graph(cf_graph, topk, norm_type)
        ).coalesce().to(self.env.device)
        self.InductiveItemItemGraphs[cache_key] = item_graphs
        self.ItemItemGraphs = item_graphs
        self._inductive_item_graph_split = cache_key
        self.clear_gcn_cache()
        print(
            f"built inductive item graph split={split}, candidates={candidate_protocol}, "
            f"warm_references={len(warm_items)}, cold_queries={len(cold_queries)}"
        )

    def _build_single_item_item_graph(self, graph, topk, norm_type):
        graph = self.dataset._topk_sparse_rows(graph.tocsr(), topk)
        return self.dataset._normalize_item_graph(graph, norm_type)

    def _item_graph_edge_confidences_from_params(self, params):
        if self.item_graph_confidence_transform == "sigmoid":
            conf = self.item_graph_confidence_min + (
                self.item_graph_confidence_max - self.item_graph_confidence_min
            ) * torch.sigmoid(params)
            return conf
        conf = torch.exp(params)
        return conf.clamp(
            min=self.item_graph_confidence_min,
            max=self.item_graph_confidence_max,
        )

    def _item_graph_edge_confidences(self, modality=None):
        if self.use_item_graph_modality_specific_confidence:
            if modality is not None:
                return self._item_graph_edge_confidences_from_params(
                    self.item_graph_edge_confidence_params[modality]
                )
            return {
                name: self._item_graph_edge_confidences_from_params(params)
                for name, params in self.item_graph_edge_confidence_params.items()
            }
        return self._item_graph_edge_confidences_from_params(self.item_graph_edge_confidence_params[0])

    def _item_graph_edge_confidence_coeffs_from_conf(self, conf):
        if self.item_graph_confidence_transform == "sigmoid":
            return conf
        blend = self.item_graph_confidence_blend
        return 1.0 + blend * (conf - 1.0)

    def _item_graph_edge_confidence_coeffs(self, modality=None):
        conf = self._item_graph_edge_confidences(modality=modality)
        if isinstance(conf, dict):
            return {
                name: self._item_graph_edge_confidence_coeffs_from_conf(value)
                for name, value in conf.items()
            }
        return self._item_graph_edge_confidence_coeffs_from_conf(conf)

    def item_graph_confidence_regularization_loss(self):
        if not self.use_item_graph_edge_confidence:
            return self.item_emb.weight.new_zeros(())

        base_target = float(getattr(self.env.args, "item_graph_confidence_reg_target", 1.0))
        target = self.item_emb.weight.new_tensor([
            float(getattr(self.env.args, "item_graph_rr_confidence_reg_target", None) or base_target),
            float(getattr(self.env.args, "item_graph_ri_confidence_reg_target", None) or base_target),
            float(getattr(self.env.args, "item_graph_ii_confidence_reg_target", None) or base_target),
        ])
        conf = self._item_graph_edge_confidences()
        if isinstance(conf, dict):
            active_modalities = set()
            dynamic_candidates = getattr(self, "ItemItemGraphDynamicCandidates", None)
            if isinstance(dynamic_candidates, dict):
                active_modalities.update(dynamic_candidates.keys())
            components = getattr(self, "ItemItemGraphComponents", None)
            if isinstance(components, dict):
                active_modalities.update(components.keys())
            if not active_modalities:
                active_modalities.update(conf.keys())
            losses = [
                (value - target.to(dtype=value.dtype, device=value.device)).pow(2).mean()
                for name, value in conf.items()
                if name in active_modalities
            ]
        else:
            losses = [(conf - target.to(dtype=conf.dtype, device=conf.device)).pow(2).mean()]
        if not losses:
            return self.item_emb.weight.new_zeros(())
        return torch.stack(losses).mean()

    def _effective_item_graph_dynamic_score_blend(self):
        return self._effective_warmup_value(
            target=float(getattr(self, "item_graph_dynamic_score_blend", 1.0)),
            start=float(getattr(self, "item_graph_dynamic_score_blend_start", -1.0)),
            warmup_epochs=int(getattr(self, "item_graph_dynamic_score_blend_warmup_epochs", 0)),
        )

    def _effective_item_graph_dynamic_neighbor_blend(self):
        return self._effective_warmup_value(
            target=float(getattr(self, "item_graph_dynamic_neighbor_blend", 1.0)),
            start=float(getattr(self, "item_graph_dynamic_neighbor_blend_start", -1.0)),
            warmup_epochs=int(getattr(self, "item_graph_dynamic_neighbor_blend_warmup_epochs", 0)),
        )

    def _effective_warmup_value(self, target, start, warmup_epochs):
        target = min(max(float(target), 0.0), 1.0)
        warmup_epochs = max(int(warmup_epochs or 0), 0)
        if warmup_epochs <= 0 or start < 0.0:
            return target
        start = min(max(start, 0.0), 1.0)
        epoch = max(int(getattr(self, "current_epoch", warmup_epochs)), 0)
        progress = min(float(epoch) / float(max(warmup_epochs, 1)), 1.0)
        return start + progress * (target - start)

    def _split_item_graph_by_observation(self, graph, observed_mask):
        graph = graph.tocoo().astype(np.float32)
        observed = observed_mask.detach().cpu().numpy().astype(bool)
        src_observed = observed[graph.row]
        dst_observed = observed[graph.col]
        rr_mask = src_observed & dst_observed
        ii_mask = ~src_observed & ~dst_observed
        ri_mask = ~(rr_mask | ii_mask)

        pieces = {}
        for name, mask in (("rr", rr_mask), ("ri", ri_mask), ("ii", ii_mask)):
            pieces[name] = sp.csr_matrix(
                (graph.data[mask], (graph.row[mask], graph.col[mask])),
                shape=graph.shape,
                dtype=np.float32,
            )
        return pieces

    def _compute_item_graph_reliability_scores(self, graph_features, masks):
        floor = min(max(float(getattr(self.env.args, "item_graph_reliability_floor", 0.4)), 0.0), 1.0)
        missing_penalty = max(float(getattr(self.env.args, "item_graph_reliability_missing_penalty", 1.0)), 0.0)
        missing_boost = max(float(getattr(self.env.args, "item_graph_reliability_missing_boost", 0.0)), 0.0)
        reliabilities = {}
        for modality in self.modalities:
            feature = F.normalize(graph_features[modality], dim=-1)
            observed = masks[modality]
            ref_sum = torch.zeros_like(feature)
            ref_weight = torch.zeros(feature.size(0), 1, dtype=feature.dtype, device=feature.device)
            for other in self.modalities:
                if other == modality:
                    continue
                other_observed = masks[other].to(dtype=feature.dtype).unsqueeze(1)
                ref_sum = ref_sum + F.normalize(graph_features[other], dim=-1) * other_observed
                ref_weight = ref_weight + other_observed

            has_ref = ref_weight.squeeze(1) > 0
            ref = torch.zeros_like(feature)
            ref[has_ref] = ref_sum[has_ref] / ref_weight[has_ref].clamp_min(1.0)
            ref = F.normalize(ref, dim=-1)
            consistency = F.cosine_similarity(feature, ref, dim=-1, eps=1e-8)
            consistency = torch.where(has_ref, consistency, torch.zeros_like(consistency))
            consistency = ((consistency + 1.0) * 0.5).clamp(0.0, 1.0)
            missing_score = floor + (1.0 - floor) * consistency
            if missing_boost > 0.0:
                missing_score = missing_score + missing_boost * consistency
            missing_score = (missing_score * missing_penalty).clamp(floor, 1.0 + missing_boost)
            reliability = torch.where(observed, torch.ones_like(missing_score), missing_score)
            reliabilities[modality] = reliability.detach().cpu().numpy().astype(np.float32)
        return reliabilities

    def _posterior_completion_reliabilities(self, masks):
        """Return c_i^m from the linear-Gaussian completion predictive variance.

        Observed modalities have reliability one. For a missing modality m,
        c_i^m = exp(-lambda * tr(W_m V_i W_m^T + sigma_m^2 I) / d_c),
        where V_i is the latent posterior covariance given the observed modalities.
        With the current homoscedastic ProMRL model, items sharing an observation
        pattern also share the same posterior reliability.
        """
        first_mask = next(iter(masks.values()))
        reliabilities = {
            modality: torch.ones(first_mask.size(0), dtype=torch.float32, device=first_mask.device)
            for modality in self.modalities
        }
        pattern_keys = torch.stack([masks[modality] for modality in self.modalities], dim=1)
        eye = torch.eye(self.d_beta, dtype=self.W[self.modalities[0]].dtype, device=first_mask.device)

        with torch.no_grad():
            for pattern in torch.unique(pattern_keys, dim=0):
                selector = (pattern_keys == pattern.unsqueeze(0)).all(dim=1)
                if not bool(selector.any()):
                    continue
                observed = [
                    self.modalities[idx]
                    for idx in range(len(self.modalities))
                    if bool(pattern[idx])
                ]
                missing = [
                    self.modalities[idx]
                    for idx in range(len(self.modalities))
                    if not bool(pattern[idx])
                ]
                if not missing:
                    continue

                posterior_precision = eye.clone()
                for modality in observed:
                    W = self.W[modality]
                    sigma2 = torch.exp(2 * self.log_sigma[modality].squeeze())
                    posterior_precision = posterior_precision + (W.T @ W) / sigma2
                posterior_cov = torch.linalg.inv(posterior_precision + 1e-6 * eye)

                for modality in missing:
                    W = self.W[modality]
                    sigma2 = torch.exp(2 * self.log_sigma[modality].squeeze())
                    mean_predictive_variance = (
                        torch.trace(posterior_cov @ (W.T @ W)) / float(self.promrl_dim)
                        + sigma2
                    )
                    reliability = torch.exp(
                        -self.posterior_reliability_scale * mean_predictive_variance
                    ).clamp(min=self.posterior_reliability_floor, max=1.0)
                    reliabilities[modality][selector] = reliability.to(dtype=torch.float32)
        return reliabilities

    def _apply_item_graph_reliability(self, graph, reliability):
        blend = min(max(float(getattr(self.env.args, "item_graph_reliability_blend", 1.0)), 0.0), 1.0)
        if blend <= 0.0:
            return graph
        graph = graph.tocoo().astype(np.float32)
        edge_reliability = reliability[graph.row] * reliability[graph.col]
        if blend < 1.0:
            edge_reliability = 1.0 + blend * (edge_reliability - 1.0)
        data = graph.data * edge_reliability.astype(np.float32)
        reweighted = sp.csr_matrix((data, (graph.row, graph.col)), shape=graph.shape, dtype=np.float32)
        reweighted.setdiag(0.0)
        reweighted.eliminate_zeros()
        return reweighted

    def _build_confidence_item_item_components(self, graphs, weights, masks, topk, norm_type):
        del norm_type
        components = {
            "cf": graphs["cf"].multiply(max(float(weights.get("cf", 0.0)), 0.0)).tocsr()
        }
        modality_specs = (("image", "v"), ("text", "t"), ("audio", "a"))
        for graph_name, modality in modality_specs:
            if graph_name not in graphs or modality not in masks:
                continue
            weight = max(float(weights.get(graph_name, 0.0)), 0.0)
            if weight == 0.0:
                continue
            pieces = self._split_item_graph_by_observation(graphs[graph_name], masks[modality])
            for piece_name, piece in pieces.items():
                if piece.nnz == 0:
                    continue
                components[piece_name] = components.get(
                    piece_name,
                    sp.csr_matrix(graphs["cf"].shape, dtype=np.float32),
                ) + piece.multiply(weight)

        init_conf = self._item_graph_edge_confidence_coeffs().detach().cpu().numpy().astype(np.float32)
        base = components["cf"].copy()
        for idx, name in enumerate(("rr", "ri", "ii")):
            if name in components:
                base = base + components[name].multiply(float(init_conf[idx]))
        base = self.dataset._topk_sparse_rows(base.tocsr(), topk)
        topology = base.copy()
        topology.data = np.ones_like(topology.data, dtype=np.float32)
        topology = topology.tocsr()

        pruned = {}
        for name, component in components.items():
            component = component.multiply(topology).tocsr().astype(np.float32)
            component.setdiag(0.0)
            component.eliminate_zeros()
            if component.nnz > 0:
                pruned[name] = component
        if not pruned:
            raise ValueError("confidence completed item graph has no positive edges")
        return pruned

    def _build_dynamic_confidence_item_graph_candidates(self, graphs, weights, masks, topk):
        del topk
        components = {
            "cf": graphs["cf"].multiply(max(float(weights.get("cf", 0.0)), 0.0)).tocsr()
        }
        modality_specs = (("image", "v"), ("text", "t"), ("audio", "a"))
        for graph_name, modality in modality_specs:
            if graph_name not in graphs or modality not in masks:
                continue
            weight = max(float(weights.get(graph_name, 0.0)), 0.0)
            if weight == 0.0:
                continue
            pieces = self._split_item_graph_by_observation(graphs[graph_name], masks[modality])
            for piece_name, piece in pieces.items():
                if piece.nnz == 0:
                    continue
                components[piece_name] = components.get(
                    piece_name,
                    sp.csr_matrix(graphs["cf"].shape, dtype=np.float32),
                ) + piece.multiply(weight)

        union = None
        for component in components.values():
            component = component.tocsr()
            marker = component.copy()
            marker.data = np.ones_like(marker.data, dtype=np.float32)
            union = marker if union is None else union + marker
        if union is None or union.nnz == 0:
            raise ValueError("dynamic confidence item graph has no candidate edges")
        union = union.tocsr()
        union.setdiag(0.0)
        union.eliminate_zeros()
        if union.nnz == 0:
            raise ValueError("dynamic confidence item graph has no non-self candidate edges")

        n_rows = union.shape[0]
        row_widths = np.diff(union.indptr)
        max_width = int(row_widths.max(initial=0))
        if max_width <= 0:
            raise ValueError("dynamic confidence item graph has empty candidate rows")

        padded_cols = np.zeros((n_rows, max_width), dtype=np.int64)
        padded_mask = np.zeros((n_rows, max_width), dtype=np.bool_)
        padded_values = np.zeros((n_rows, max_width, 4), dtype=np.float32)
        component_order = ("cf", "rr", "ri", "ii")
        component_csrs = {
            name: components.get(name, sp.csr_matrix(union.shape, dtype=np.float32)).tocsr()
            for name in component_order
        }

        for row in range(n_rows):
            start, end = union.indptr[row], union.indptr[row + 1]
            if end <= start:
                continue
            cols = union.indices[start:end]
            width = cols.shape[0]
            padded_cols[row, :width] = cols
            padded_mask[row, :width] = True
            col_pos = {int(col): idx for idx, col in enumerate(cols)}
            for comp_idx, name in enumerate(component_order):
                comp = component_csrs[name]
                c_start, c_end = comp.indptr[row], comp.indptr[row + 1]
                if c_end <= c_start:
                    continue
                comp_cols = comp.indices[c_start:c_end]
                comp_data = comp.data[c_start:c_end]
                for col, val in zip(comp_cols, comp_data):
                    pos = col_pos.get(int(col))
                    if pos is not None:
                        padded_values[row, pos, comp_idx] = float(val)

        return {
            "cols": torch.from_numpy(padded_cols),
            "mask": torch.from_numpy(padded_mask),
            "values": torch.from_numpy(padded_values),
        }

    def _build_dynamic_confidence_item_graph_candidates_for_modality(
        self,
        cf_graph,
        feature_graph,
        observed_mask,
        cf_weight,
        feature_weight,
    ):
        components = {
            "cf": cf_graph.multiply(max(float(cf_weight), 0.0)).tocsr()
        }
        weight = max(float(feature_weight), 0.0)
        if weight != 0.0 and feature_graph is not None:
            pieces = self._split_item_graph_by_observation(feature_graph, observed_mask)
            for piece_name, piece in pieces.items():
                if piece.nnz == 0:
                    continue
                components[piece_name] = components.get(
                    piece_name,
                    sp.csr_matrix(cf_graph.shape, dtype=np.float32),
                ) + piece.multiply(weight)

        union = None
        for component in components.values():
            component = component.tocsr()
            marker = component.copy()
            marker.data = np.ones_like(marker.data, dtype=np.float32)
            union = marker if union is None else union + marker
        if union is None or union.nnz == 0:
            raise ValueError("dynamic modality confidence item graph has no candidate edges")
        union = union.tocsr()
        union.setdiag(0.0)
        union.eliminate_zeros()
        if union.nnz == 0:
            raise ValueError("dynamic modality confidence item graph has no non-self candidate edges")

        n_rows = union.shape[0]
        row_widths = np.diff(union.indptr)
        max_width = int(row_widths.max(initial=0))
        if max_width <= 0:
            raise ValueError("dynamic modality confidence item graph has empty candidate rows")

        padded_cols = np.zeros((n_rows, max_width), dtype=np.int64)
        padded_mask = np.zeros((n_rows, max_width), dtype=np.bool_)
        padded_values = np.zeros((n_rows, max_width, 4), dtype=np.float32)
        component_order = ("cf", "rr", "ri", "ii")
        component_csrs = {
            name: components.get(name, sp.csr_matrix(union.shape, dtype=np.float32)).tocsr()
            for name in component_order
        }

        for row in range(n_rows):
            start, end = union.indptr[row], union.indptr[row + 1]
            if end <= start:
                continue
            cols = union.indices[start:end]
            width = cols.shape[0]
            padded_cols[row, :width] = cols
            padded_mask[row, :width] = True
            col_pos = {int(col): idx for idx, col in enumerate(cols)}
            for comp_idx, name in enumerate(component_order):
                comp = component_csrs[name]
                c_start, c_end = comp.indptr[row], comp.indptr[row + 1]
                if c_end <= c_start:
                    continue
                comp_cols = comp.indices[c_start:c_end]
                comp_data = comp.data[c_start:c_end]
                for col, val in zip(comp_cols, comp_data):
                    pos = col_pos.get(int(col))
                    if pos is not None:
                        padded_values[row, pos, comp_idx] = float(val)

        return {
            "cols": torch.from_numpy(padded_cols),
            "mask": torch.from_numpy(padded_mask),
            "values": torch.from_numpy(padded_values),
        }

    def _build_confidence_item_item_graph_for_modality(
        self,
        cf_graph,
        feature_graph,
        observed_mask,
        topk,
        norm_type,
        cf_weight,
        feature_weight,
        modality=None,
    ):
        del norm_type
        components = {
            "cf": cf_graph.multiply(max(float(cf_weight), 0.0)).tocsr()
        }
        weight = max(float(feature_weight), 0.0)
        if weight != 0.0 and feature_graph is not None:
            pieces = self._split_item_graph_by_observation(feature_graph, observed_mask)
            for piece_name, piece in pieces.items():
                if piece.nnz == 0:
                    continue
                components[piece_name] = components.get(
                    piece_name,
                    sp.csr_matrix(cf_graph.shape, dtype=np.float32),
                ) + piece.multiply(weight)

        init_conf = self._item_graph_edge_confidence_coeffs(modality=modality).detach().cpu().numpy().astype(np.float32)
        base = components["cf"].copy()
        for idx, name in enumerate(("rr", "ri", "ii")):
            if name in components:
                base = base + components[name].multiply(float(init_conf[idx]))
        base = self.dataset._topk_sparse_rows(base.tocsr(), topk)
        topology = base.copy()
        topology.data = np.ones_like(topology.data, dtype=np.float32)
        topology = topology.tocsr()

        pruned = {}
        for name, component in components.items():
            component = component.multiply(topology).tocsr().astype(np.float32)
            component.setdiag(0.0)
            component.eliminate_zeros()
            if component.nnz > 0:
                pruned[name] = component
        if not pruned:
            raise ValueError("confidence completed item graph has no positive edges")
        return pruned

    def _build_weighted_item_item_graph(
        self,
        graphs,
        weights,
        topk,
        norm_type,
        required_names=None,
        context="item graph",
    ):
        if required_names is not None:
            missing = [name for name in required_names if name not in graphs]
            if missing:
                raise ValueError(f"{context} requires graph(s): {', '.join(missing)}")

        graph = None
        total_weight = 0.0
        for name, candidate in graphs.items():
            weight = max(float(weights.get(name, 0.0)), 0.0)
            if weight == 0.0:
                continue
            total_weight += weight
            graph = candidate.multiply(weight) if graph is None else graph + candidate.multiply(weight)
        if graph is None or total_weight == 0.0:
            raise ValueError(f"{context} requires at least one positive graph weight")
        graph = graph.multiply(1.0 / total_weight)
        graph = self.dataset._topk_sparse_rows(graph.tocsr(), topk)
        return self.dataset._normalize_item_graph(graph, norm_type)


    def _current_contra_heads(self):
        return {
            modality: getattr(self, f"contra_head_{modality}")
            for modality in self.modalities
        }

    def _current_projection_heads(self):
        if self.use_decoupled_latent_bridge:
            return {
                modality: getattr(self, f"comp_proj_{modality}")
                for modality in self.modalities
            }
        if self.use_latent_direct_bridge:
            return {
                modality: getattr(self, f"{modality}_gcn").MLP
                for modality in self.modalities
            }
        return self._current_contra_heads()

    def _current_recommendation_heads(self):
        return {
            modality: getattr(self, f"{modality}_gcn").MLP
            for modality in self.modalities
        }

    def _completion_adapters(self):
        return {
            modality: getattr(self, f"comp_to_rec_{modality}")
            for modality in self.modalities
        }

    def _gcn_skip_mlp(self):
        return self.use_latent_completion_bridge

    def _tensorize_external_modal_masks(self, masks):
        if masks is None:
            return None
        tensor_masks = {}
        for modality, mask in masks.items():
            tensor_masks[modality] = torch.as_tensor(mask, dtype=torch.bool, device=self.env.device)
        return tensor_masks

    def _current_external_modal_observed_masks(self):
        if self.modal_feature_mask_source != "external_observed":
            return None
        masks = self.train_external_modal_observed_masks if self.training else self.eval_external_modal_observed_masks
        if masks is None:
            raise ValueError(
                "modal_feature_mask_source=external_observed requires observed mask files "
                "in the external modal feature train/eval directories"
            )
        return masks

    def _missing_masks(self, raw_features=None):
        if raw_features is None:
            raw_features = self._current_raw_modal_features()
        external_masks = self._current_external_modal_observed_masks()
        if external_masks is not None:
            first = next(iter(raw_features.values()))
            if first.size(0) == self.m_item:
                return {
                    modality: external_masks[modality]
                    for modality in raw_features
                    if modality in external_masks
                }
        masks = {}
        for modality, feature in raw_features.items():
            masks[modality] = feature.abs().sum(dim=1) > 0
        return masks

    def project_features(self, item_ids=None, raw_features=None):
        raw_features = raw_features or self._current_raw_modal_features()
        if self.promrl_projection_mode == "identity":
            projected = {}
            for modality in self.modalities:
                source = raw_features[modality]
                if item_ids is not None:
                    source = source[item_ids]
                if source.size(1) != self.promrl_dim:
                    raise ValueError(
                        "promrl_projection_mode='identity' requires modal feature dim "
                        f"to equal promrl_dim ({self.promrl_dim}); modality {modality} has {source.size(1)}"
                    )
                projected[modality] = torch.nan_to_num(
                    F.normalize(source, dim=-1), nan=0.0, posinf=0.0, neginf=0.0
                )
            return projected

        heads = self._current_projection_heads()
        projected = {}
        for modality in self.modalities:
            source = raw_features[modality]
            if item_ids is not None:
                source = source[item_ids]
            projected_feature = torch.nan_to_num(heads[modality](source), nan=0.0, posinf=0.0, neginf=0.0)
            projected[modality] = torch.nan_to_num(
                F.normalize(projected_feature, dim=-1), nan=0.0, posinf=0.0, neginf=0.0
            )
        return projected

    def project_recommendation_features(self, item_ids=None, raw_features=None):
        raw_features = raw_features or self._current_raw_modal_features()
        heads = self._current_recommendation_heads()
        projected = {}
        for modality in self.modalities:
            source = raw_features[modality]
            if item_ids is not None:
                source = source[item_ids]
            projected_feature = torch.nan_to_num(heads[modality](source), nan=0.0, posinf=0.0, neginf=0.0)
            projected[modality] = torch.nan_to_num(
                F.normalize(projected_feature, dim=-1), nan=0.0, posinf=0.0, neginf=0.0
            )
        return projected

    def _posterior_mean_from_observed(
        self,
        obs_feats,
        observed_modalities,
        beta_prior=None,
        prior_precision=1.0,
        return_cov=False,
    ):
        N = next(iter(obs_feats.values())).size(0) if obs_feats else beta_prior.size(0)
        device = beta_prior.device if beta_prior is not None else next(iter(obs_feats.values())).device
        precision = float(prior_precision)
        V_inv = precision * torch.eye(self.d_beta, device=device)
        rhs = torch.zeros(N, self.d_beta, device=device)
        if beta_prior is not None:
            rhs = rhs + precision * beta_prior

        for modality in observed_modalities:
            if obs_feats.get(modality) is None:
                continue
            W = self.W[modality]
            sigma2 = torch.exp(2 * self.log_sigma[modality].squeeze())
            V_inv = V_inv + (1.0 / sigma2) * (W.T @ W)
            residual = obs_feats[modality] - self.mu[modality].unsqueeze(0)
            rhs = rhs + (1.0 / sigma2) * (residual @ W)

        V = torch.inverse(V_inv + 1e-6 * torch.eye(self.d_beta, device=device))
        mean = rhs @ V.T
        if return_cov:
            return mean, V
        return mean

    def _compute_standard_beta_means(self, projected, masks, detach_inputs=True):
        N = projected[self.modalities[0]].size(0)
        device = projected[self.modalities[0]].device
        beta = torch.zeros(N, self.d_beta, device=device)
        pattern_keys = torch.stack([masks[m] for m in self.modalities], dim=1)

        for pattern in torch.unique(pattern_keys, dim=0):
            selector = (pattern_keys == pattern.unsqueeze(0)).all(dim=1)
            if selector.sum() == 0:
                continue
            observed = [self.modalities[i] for i in range(len(self.modalities)) if bool(pattern[i])]
            if not observed:
                continue
            obs_feats = {}
            for modality in observed:
                feat = projected[modality][selector]
                obs_feats[modality] = feat.detach() if detach_inputs else feat
            beta[selector] = self._posterior_mean_from_observed(
                obs_feats,
                observed,
                beta_prior=None,
                prior_precision=1.0,
            )
        return beta

    def impute_modalities(self, projected_feats, masks=None, item_ids=None):
        if self.disable_imputation:
            return {modality: feat.clone() for modality, feat in projected_feats.items()}

        if masks is None:
            full_masks = self._missing_masks()
            masks = full_masks
        completed = {modality: feat.clone() for modality, feat in projected_feats.items()}

        if all(mask.all() for mask in masks.values()):
            return completed

        pattern_keys = torch.stack([masks[modality] for modality in self.modalities], dim=1)
        log_sigma_dict = {modality: self.log_sigma[modality].squeeze() for modality in self.modalities}
        W_dict = {modality: self.W[modality] for modality in self.modalities}
        mu_dict = {modality: self.mu[modality] for modality in self.modalities}
        unique_patterns = torch.unique(pattern_keys, dim=0)
        for pattern in unique_patterns:
            missing_modalities = [self.modalities[i] for i in range(len(self.modalities)) if not bool(pattern[i])]
            if not missing_modalities:
                continue

            selector = (pattern_keys == pattern.unsqueeze(0)).all(dim=1)
            if selector.sum() == 0:
                continue

            observed_modalities = [self.modalities[i] for i in range(len(self.modalities)) if bool(pattern[i])]
            if observed_modalities:
                obs_feats = {
                    modality: projected_feats[modality][selector]
                    for modality in observed_modalities
                }
                m, _ = update_posterior(
                    obs_feats,
                    W_dict,
                    mu_dict,
                    log_sigma_dict,
                    observed_modalities,
                    self.d_beta,
                )
            else:
                m = torch.zeros(selector.sum(), self.d_beta, device=self.env.device)

            for modality in missing_modalities:
                recon = m @ self.W[modality].transpose(0, 1) + self.mu[modality].unsqueeze(0)
                completed[modality][selector] = F.normalize(recon, dim=-1)

        return completed

    def get_gcn_modal_features(self):
        projected = self.project_features()
        if self.disable_imputation:
            return projected
        item_ids = torch.arange(self.m_item, device=self.env.device)
        completed = self.impute_modalities(projected, self._missing_masks(), item_ids=item_ids)
        return completed

    def decode_completed_to_raw(self, completed_shared):
        if not self.use_decode_head:
            raise RuntimeError(
                "decode_completed_to_raw is unavailable when feature_bridge_mode uses latent completion"
            )
        decoded = {
            modality: getattr(self, f"decoder_{modality}")(completed_shared[modality])
            for modality in self.modalities
        }
        if self.decoder_output_mode == "native_direction_norm":
            return decoded
        return {
            modality: F.normalize(feature, dim=-1)
            for modality, feature in decoded.items()
        }

    def bridge_completed_to_recommendation_raw(self, completed_shared):
        return self.decode_completed_to_raw(completed_shared)

    def _decoder_target(self, modality, normalized_target, item_ids=None):
        if self.decoder_output_mode == "native_direction_norm":
            return self._native_raw_target(modality, normalized_target, item_ids=item_ids)
        return normalized_target

    def _decoder_reconstruction_loss(self, modality, prediction, target):
        cosine_loss = 1.0 - F.cosine_similarity(prediction, target, dim=-1).mean()
        if self.decoder_output_mode != "native_direction_norm":
            return cosine_loss

        feature_std = getattr(self, f"native_raw_std_{modality}").unsqueeze(0)
        # Coordinate regression is useful for learning the native feature
        # direction, but its conditional-mean optimum shrinks magnitude when
        # the direction is uncertain.  Evaluate it at the target magnitude so
        # that this term cannot teach the norm head to mark completed items via
        # systematically smaller norms.  Magnitude is learned exclusively by
        # the log-norm objective below.
        target_norm = target.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        direction_scaled_prediction = F.normalize(prediction, dim=-1) * target_norm
        standardized_error = (direction_scaled_prediction - target) / feature_std
        raw_loss = F.smooth_l1_loss(
            standardized_error,
            torch.zeros_like(standardized_error),
            reduction="mean",
        )
        prediction_log_norm = prediction.norm(dim=-1).clamp_min(1e-8).log()
        target_log_norm = target.norm(dim=-1).clamp_min(1e-8).log()
        norm_loss = F.smooth_l1_loss(prediction_log_norm, target_log_norm)

        relation_loss = prediction.new_zeros(())
        relation_max_items = max(
            int(getattr(self.env.args, "decoder_relation_max_items", 64)),
            0,
        )
        if relation_max_items > 1 and prediction.size(0) > 1:
            relation_count = min(relation_max_items, prediction.size(0))
            pred_direction = F.normalize(prediction[:relation_count], dim=-1)
            target_direction = F.normalize(target[:relation_count], dim=-1)
            pred_relation = pred_direction @ pred_direction.transpose(0, 1)
            target_relation = target_direction @ target_direction.transpose(0, 1)
            relation_loss = F.smooth_l1_loss(pred_relation, target_relation)

        self.latest_decoder_loss_components = {
            "raw": float(raw_loss.detach().cpu()),
            "cosine": float(cosine_loss.detach().cpu()),
            "norm": float(norm_loss.detach().cpu()),
            "relation": float(relation_loss.detach().cpu()),
        }
        return (
            float(getattr(self.env.args, "decoder_raw_loss_weight", 1.0)) * raw_loss
            + float(getattr(self.env.args, "decoder_cosine_loss_weight", 1.0)) * cosine_loss
            + float(getattr(self.env.args, "decoder_norm_loss_weight", 0.25)) * norm_loss
            + float(getattr(self.env.args, "decoder_relation_loss_weight", 0.1)) * relation_loss
        )

    def adapt_completed_to_recommendation(self, completed_shared):
        adapters = self._completion_adapters()
        adapted = {}
        for modality in self.modalities:
            adapted_feature = torch.nan_to_num(
                adapters[modality](completed_shared[modality]),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            adapted[modality] = torch.nan_to_num(
                F.normalize(adapted_feature, dim=-1),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        return adapted

    def _apply_fusion(self, item_source, deterministic=False):
        item_hidden = self.fusion_linear[0](item_source)
        if not deterministic:
            item_hidden = self.fusion_linear[1](item_hidden)
        item_emb = self.fusion_linear[2](item_hidden)
        return item_emb

    def _item_graph_for_modality(self, modality=None):
        if self.item_graph_kind in (
            "modality_masked",
            "modality_completed",
            "modality_completed_inductive",
            "modality_completed_confidence",
        ):
            if not self.ItemItemGraphs:
                raise RuntimeError(
                    "modality graph item-item propagation requires modality item-item graphs"
                )
            key = modality if modality in self.ItemItemGraphs else None
            if key is None:
                key = "cf" if "cf" in self.ItemItemGraphs else next(iter(self.ItemItemGraphs))
            return self.ItemItemGraphs[key]
        return self.ItemItemGraph

    def _apply_item_graph_modal_residual(self, item_emb, observed_mask=None, modality=None):
        if not self.use_item_graph_modal_residual:
            return item_emb

        alpha = self.item_graph_modal_alpha
        out = item_emb
        for _ in range(self.item_graph_modal_layers):
            if self.item_graph_kind in ("fused_completed_dynamic_confidence", "modality_completed_dynamic_confidence"):
                neigh = self._dynamic_confidence_item_graph_mm(out, modality=modality)
            elif self.item_graph_kind in ("fused_completed_confidence", "modality_completed_confidence"):
                neigh = self._confidence_item_graph_mm(out, modality=modality)
            else:
                graph = self._item_graph_for_modality(modality)
                if graph is None:
                    raise RuntimeError("item_graph_modal_alpha > 0 requires an item-item graph")
                neigh = torch.sparse.mm(graph, out)
            out = (1.0 - alpha) * out + alpha * neigh
        if self.item_graph_modal_target == "missing" and observed_mask is not None:
            missing_mask = ~observed_mask
            out = torch.where(missing_mask.unsqueeze(1), out, item_emb)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _component_row_sum(self, component):
        component = component.coalesce()
        values = component.values()
        rows = component.indices()[0]
        row_sum = torch.zeros(component.shape[0], dtype=values.dtype, device=values.device)
        row_sum.scatter_add_(0, rows, values)
        return row_sum

    def _confidence_component_coeffs(self, modality=None):
        conf = self._item_graph_edge_confidence_coeffs(modality=modality)
        return {
            "cf": torch.ones((), dtype=conf.dtype, device=conf.device),
            "rr": conf[0],
            "ri": conf[1],
            "ii": conf[2],
        }

    def _dynamic_confidence_item_graph_mm(self, item_emb, modality=None):
        candidates = getattr(self, "ItemItemGraphDynamicCandidates", None)
        if not candidates:
            raise RuntimeError(
                "item_graph_kind=fused_completed_dynamic_confidence|modality_completed_dynamic_confidence "
                "requires dynamic graph candidates"
            )
        if self.item_graph_kind == "modality_completed_dynamic_confidence":
            if modality is None:
                raise RuntimeError("item_graph_kind=modality_completed_dynamic_confidence requires a modality key")
            if modality not in candidates:
                raise RuntimeError(f"missing dynamic confidence graph candidates for modality={modality}")
            candidates = candidates[modality]

        cols = candidates["cols"]
        mask = candidates["mask"]
        values = candidates["values"].to(dtype=item_emb.dtype)
        edge_coeffs = self._item_graph_edge_confidence_coeffs(modality=modality).to(dtype=item_emb.dtype)
        coeff = torch.stack(
            [
                torch.ones((), dtype=item_emb.dtype, device=item_emb.device),
                edge_coeffs[0],
                edge_coeffs[1],
                edge_coeffs[2],
            ]
        )
        learned_scores = torch.sum(values * coeff.view(1, 1, 4), dim=-1)
        score_blend = self._effective_item_graph_dynamic_score_blend()
        base_scores = torch.sum(values, dim=-1)
        if score_blend < 1.0:
            scores = base_scores + score_blend * (learned_scores - base_scores)
        else:
            scores = learned_scores
        neg_inf = torch.finfo(scores.dtype).min
        scores = torch.where(mask, scores, torch.full_like(scores, neg_inf))

        topk = min(int(getattr(self, "item_graph_dynamic_topk", scores.shape[1])), scores.shape[1])
        norm_type = getattr(self, "item_graph_dynamic_norm_type", "rw")

        def aggregate_neighbors(candidate_scores):
            top_scores, top_pos = torch.topk(candidate_scores, k=topk, dim=1)
            top_cols = torch.gather(cols, 1, top_pos)
            valid = torch.isfinite(top_scores) & (top_scores > 0)
            top_scores = torch.where(valid, top_scores, torch.zeros_like(top_scores))
            top_cols = torch.where(valid, top_cols, torch.zeros_like(top_cols))

            if norm_type == "sym":
                col_degree = torch.zeros(item_emb.size(0), dtype=item_emb.dtype, device=item_emb.device)
                col_degree.scatter_add_(0, top_cols.reshape(-1), top_scores.reshape(-1))
                col_inv_sqrt = torch.zeros_like(col_degree)
                col_nonzero = col_degree > 0
                col_inv_sqrt[col_nonzero] = torch.pow(col_degree[col_nonzero], -0.5)
                neighbor_emb = item_emb[top_cols] * col_inv_sqrt[top_cols].unsqueeze(-1)
                neigh = torch.sum(top_scores.unsqueeze(-1) * neighbor_emb, dim=1)
                row_degree = torch.sum(top_scores, dim=1)
                row_inv_sqrt = torch.zeros_like(row_degree)
                row_nonzero = row_degree > 0
                row_inv_sqrt[row_nonzero] = torch.pow(row_degree[row_nonzero], -0.5)
                return neigh * row_inv_sqrt.unsqueeze(1)

            neighbor_emb = item_emb[top_cols]
            neigh = torch.sum(top_scores.unsqueeze(-1) * neighbor_emb, dim=1)
            if norm_type == "none":
                return neigh
            if norm_type == "rw":
                degree = torch.sum(top_scores, dim=1)
                inv = torch.zeros_like(degree)
                nonzero = degree > 0
                inv[nonzero] = 1.0 / degree[nonzero]
                return neigh * inv.unsqueeze(1)
            raise ValueError(f"Unsupported item graph normalization: {norm_type}")

        dynamic_neigh = aggregate_neighbors(scores)
        neighbor_blend = self._effective_item_graph_dynamic_neighbor_blend()
        if neighbor_blend >= 1.0:
            return dynamic_neigh

        base_scores = torch.where(mask, base_scores, torch.full_like(base_scores, neg_inf))
        base_neigh = aggregate_neighbors(base_scores)
        if neighbor_blend <= 0.0:
            return base_neigh
        return base_neigh + neighbor_blend * (dynamic_neigh - base_neigh)

    def _confidence_item_graph_mm(self, item_emb, modality=None):
        if not self.ItemItemGraphComponents:
            raise RuntimeError(
                "item_graph_kind=fused_completed_confidence|modality_completed_confidence requires confidence graph components"
            )

        components = self.ItemItemGraphComponents
        if self.item_graph_kind == "modality_completed_confidence":
            if modality is None:
                raise RuntimeError("item_graph_kind=modality_completed_confidence requires a modality key")
            if modality not in components:
                raise RuntimeError(f"missing confidence graph components for modality={modality}")
            components = components[modality]

        coeffs = self._confidence_component_coeffs(modality=modality)
        norm_type = getattr(self, "item_graph_dynamic_norm_type", "rw")
        components = [
            (name, component, coeffs[name])
            for name, component in components.items()
            if name in coeffs
        ]
        if not components:
            raise RuntimeError("confidence graph components are empty")

        if norm_type == "none":
            neigh = torch.zeros_like(item_emb)
            for _, component, coeff in components:
                neigh = neigh + coeff * torch.sparse.mm(component, item_emb)
            return neigh

        degree = torch.zeros(item_emb.size(0), dtype=item_emb.dtype, device=item_emb.device)
        for _, component, coeff in components:
            degree = degree + coeff * self._component_row_sum(component).to(dtype=item_emb.dtype)

        if norm_type == "rw":
            neigh = torch.zeros_like(item_emb)
            for _, component, coeff in components:
                neigh = neigh + coeff * torch.sparse.mm(component, item_emb)
            inv = torch.zeros_like(degree)
            nonzero = degree > 0
            inv[nonzero] = 1.0 / degree[nonzero]
            return neigh * inv.unsqueeze(1)

        if norm_type == "sym":
            inv_sqrt = torch.zeros_like(degree)
            nonzero = degree > 0
            inv_sqrt[nonzero] = torch.pow(degree[nonzero], -0.5)
            scaled_item_emb = item_emb * inv_sqrt.unsqueeze(1)
            neigh = torch.zeros_like(item_emb)
            for _, component, coeff in components:
                neigh = neigh + coeff * torch.sparse.mm(component, scaled_item_emb)
            return neigh * inv_sqrt.unsqueeze(1)

        raise ValueError(f"Unsupported item graph normalization: {norm_type}")

    def _global_weighted_item_source(self, item_outputs):
        weights = F.softmax(self.global_fusion_params[0], dim=0)
        for idx, modality in enumerate(self.modalities):
            self.latest_rum_fusion_metrics[f"global_fusion_weight_{modality}"] = float(
                weights[idx].detach().cpu()
            )
        return sum(weights[idx] * item_outputs[modality] for idx, modality in enumerate(self.modalities))

    def _uses_posterior_reliability_for(self, component):
        if not self.use_posterior_reliability:
            return False
        return self.posterior_reliability_scope == "both" or self.posterior_reliability_scope == component

    def _fuse_item_sources(self, item_outputs, modal_features=None, raw_features=None, observed_masks=None):
        if self.use_global_weighted_fusion:
            return self._global_weighted_item_source(item_outputs)

        if self._uses_posterior_reliability_for("fusion"):
            if observed_masks is None:
                if raw_features is None:
                    raise ValueError(
                        "posterior_reliability fusion requires observed_masks or raw_features"
                    )
                observed_masks = self._missing_masks(raw_features=raw_features)
            reliabilities = self._posterior_completion_reliabilities(observed_masks)
            denominator = torch.stack(
                [reliabilities[modality] for modality in self.modalities],
                dim=0,
            ).sum(dim=0).clamp_min(1e-8)
            return sum(
                item_outputs[modality]
                * (reliabilities[modality] / denominator).unsqueeze(1)
                for modality in self.modalities
            )

        return sum(item_outputs[modality] for modality in self.modalities) / len(self.modalities)

    def _completion_gate_mix_alpha(self):
        if self.use_learned_completion_gate_mix:
            mix_max = min(max(float(getattr(self.env.args, "completion_gate_mix_max", 1.0)), 1e-4), 1.0)
            return mix_max * torch.sigmoid(self.completion_gate_mix_params[0])
        return min(max(float(self.env.args.completion_gate_mix_alpha), 0.0), 1.0)

    def _completion_gate_residual_alpha(self):
        if bool(getattr(self.env.args, "completion_gate_no_residual_alpha", 0)):
            return 1.0
        return max(float(self.env.args.completion_gate_residual_alpha), 0.0)

    def _record_completion_gate_mix_alpha(self, mix_alpha):
        if torch.is_tensor(mix_alpha):
            value = float(mix_alpha.detach().cpu())
        else:
            value = float(mix_alpha)
        self.latest_completion_gate_metrics["completion_gate_mix_alpha"] = value
        self.latest_completion_gate_metrics["completion_gate_learn_mix"] = float(
            self.use_learned_completion_gate_mix
        )
        self.latest_completion_gate_metrics["completion_gate_mix_max"] = float(
            getattr(self.env.args, "completion_gate_mix_max", 1.0)
        )

    def get_recommender_modal_features(self, raw_features=None, allow_imputer_grad=False):
        item_ids = None
        if raw_features is None:
            item_ids = torch.arange(self.m_item, device=self.env.device)
        raw_features = raw_features or self._current_raw_modal_features()

        if self.use_latent_direct_bridge:
            projected = self.project_features(raw_features=raw_features)
            if self.disable_imputation:
                return projected

            masks = self._missing_masks(raw_features)
            if all(mask.all() for mask in masks.values()):
                return projected

            return self._build_completed_features(
                projected,
                masks,
                detach_imputed=not allow_imputer_grad,
                item_ids=item_ids,
            )

        if self.use_decoupled_latent_bridge:
            recommendation_projected = self.project_recommendation_features(raw_features=raw_features)
            if self.disable_imputation:
                return recommendation_projected

            masks = self._missing_masks(raw_features)
            if all(mask.all() for mask in masks.values()):
                return recommendation_projected

            completion_projected = self.project_features(raw_features=raw_features)
            completed_shared = self._build_completed_features(
                completion_projected,
                masks,
                detach_imputed=not allow_imputer_grad,
                item_ids=item_ids,
            )
            adapted_completed = self.adapt_completed_to_recommendation(completed_shared)

            modal_features = {}
            for modality in self.modalities:
                mask = masks[modality].unsqueeze(1)
                modal_features[modality] = torch.where(
                    mask,
                    recommendation_projected[modality],
                    adapted_completed[modality],
                )
            return modal_features

        modal_features = {modality: feature.clone() for modality, feature in raw_features.items()}
        if self.disable_imputation:
            return modal_features

        masks = self._missing_masks(raw_features)
        if all(mask.all() for mask in masks.values()):
            return modal_features

        projected = self.project_features(raw_features=raw_features)
        completed_shared = self._build_completed_features(
            projected,
            masks,
            detach_imputed=not allow_imputer_grad,
            item_ids=item_ids,
        )
        bridged_raw = self.bridge_completed_to_recommendation_raw(completed_shared)

        for modality in self.modalities:
            mask = masks[modality].unsqueeze(1)
            modal_features[modality] = torch.where(mask, raw_features[modality], bridged_raw[modality])

        return modal_features

    def _completion_reliability_gates(self, raw_features=None):
        ones = {
            modality: torch.ones(self.m_item, 1, device=self.env.device)
            for modality in self.modalities
        }
        self.latest_completion_gate_metrics = {}
        self.latest_completion_gate_regularizer = torch.zeros((), device=self.env.device)
        if not self.use_completion_gate or self.disable_imputation:
            return ones

        raw_features = raw_features or self._current_raw_modal_features()
        masks = self._missing_masks(raw_features)
        if all(mask.all() for mask in masks.values()):
            return ones

        projected = self.project_features(raw_features=raw_features)
        item_ids = None
        if raw_features is None:
            item_ids = torch.arange(self.m_item, device=self.env.device)
        completed_shared = self._build_completed_features(projected, masks, item_ids=item_ids)
        pattern = torch.stack([masks[modality].float() for modality in self.modalities], dim=1)
        if self.completion_gate_item_context_source == "id_embedding":
            item_ids = torch.arange(self.m_item, device=self.env.device)
            item_context = self.item_emb(item_ids)
        elif self.completion_gate_item_context_source == "shared_mean":
            item_context = torch.stack(
                [
                    torch.where(masks[modality].unsqueeze(1), projected[modality], completed_shared[modality])
                    for modality in self.modalities
                ],
                dim=0,
            ).mean(dim=0)
        else:
            item_context = None

        gates = {}
        metrics = {}
        regularizers = []
        identity_regularizers = []
        missing_gate_means = []
        learned_gates = {}
        gate_logits = {}
        missing_masks = {}
        for modality in self.modalities:
            if self.completion_gate_mode == "alignment":
                observed_modalities = [m for m in self.modalities if m != modality]
                observed_stack = torch.stack(
                    [
                        torch.where(masks[m].unsqueeze(1), projected[m], completed_shared[m])
                        for m in observed_modalities
                    ],
                    dim=0,
                )
                reference = F.normalize(observed_stack.mean(dim=0), dim=-1)
                consistency = F.cosine_similarity(completed_shared[modality], reference, dim=-1, eps=1e-8).unsqueeze(1)
                centered = consistency - self.env.args.completion_gate_alignment_center
                temp = max(self.env.args.completion_gate_alignment_temp, 1e-6)
                learned_gate = torch.sigmoid(centered / temp)
                learned_gate = self.completion_gate_floor + (1.0 - self.completion_gate_floor) * learned_gate
            elif self.use_global_rank_residual_completion_gate:
                alpha = self._completion_gate_residual_alpha()
                global_logit = self.completion_gate_global_logits[modality]
                learned_gate = 1.0 + alpha * torch.tanh(global_logit)
                learned_gate = learned_gate.expand(self.m_item, 1)
                metrics[f"completion_gate_{modality}_global_logit"] = float(
                    global_logit.detach().cpu()
                )
            else:
                if (
                    self.use_all_modal_rank_residual_completion_gate
                    or self.use_centered_all_modal_rank_residual_completion_gate
                ):
                    modal_gate_source = torch.where(
                        masks[modality].unsqueeze(1),
                        projected[modality],
                        completed_shared[modality],
                    )
                else:
                    modal_gate_source = completed_shared[modality]
                if self.completion_gate_detach_inputs:
                    gate_modal_input = modal_gate_source.detach()
                    gate_pattern = pattern.detach()
                else:
                    gate_modal_input = modal_gate_source
                    gate_pattern = pattern
                gate_inputs = [gate_modal_input]
                if item_context is not None:
                    gate_item_context = item_context.detach() if self.completion_gate_detach_inputs else item_context
                    gate_inputs.append(gate_item_context)
                gate_inputs.append(gate_pattern)
                gate_input = torch.cat(gate_inputs, dim=1)
                gate_logit = self.completion_gates[modality](gate_input)
                gate_logits[modality] = gate_logit
                if (
                    self.use_softmax_rank_residual_completion_gate
                    or self.use_centered_rank_residual_completion_gate
                    or self.use_delta_rank_residual_completion_gate
                    or self.use_centered_all_modal_rank_residual_completion_gate
                ):
                    learned_gate = torch.ones_like(gate_logit)
                elif self.use_shrink_rank_residual_completion_gate:
                    alpha = self._completion_gate_residual_alpha()
                    learned_gate = 1.0 - alpha * torch.sigmoid(gate_logit)
                elif self.use_rank_residual_completion_gate:
                    alpha = self._completion_gate_residual_alpha()
                    learned_gate = 1.0 + alpha * torch.tanh(gate_logit)
                else:
                    learned_gate = torch.sigmoid(gate_logit)
                    learned_gate = self.completion_gate_floor + (1.0 - self.completion_gate_floor) * learned_gate
            learned_gates[modality] = learned_gate
            missing_masks[modality] = ~masks[modality]

        if self.use_softmax_rank_residual_completion_gate:
            temp = max(float(getattr(self.env.args, "completion_gate_softmax_temp", 1.0)), 1e-6)
            alpha = self._completion_gate_residual_alpha()
            logits = torch.stack([gate_logits[modality] for modality in self.modalities], dim=0)
            competitive = F.softmax(logits / temp, dim=0) * len(self.modalities)
            for idx, modality in enumerate(self.modalities):
                learned_gates[modality] = 1.0 + alpha * (competitive[idx] - 1.0)
            metrics["completion_gate_softmax_temp"] = temp
            metrics["completion_gate_softmax_mean_normalized"] = 1.0
        elif self.use_centered_rank_residual_completion_gate:
            alpha = self._completion_gate_residual_alpha()
            for modality in self.modalities:
                missing_mask = missing_masks[modality]
                gate_logit = gate_logits[modality]
                if missing_mask.any():
                    center = gate_logit[missing_mask].mean()
                else:
                    center = gate_logit.mean()
                learned_gates[modality] = 1.0 + alpha * torch.tanh(gate_logit - center)
            metrics["completion_gate_missing_logit_centered"] = 1.0
        elif self.use_delta_rank_residual_completion_gate:
            alpha = self._completion_gate_residual_alpha()
            for modality in self.modalities:
                missing_mask = missing_masks[modality]
                residual = torch.tanh(gate_logits[modality])
                if missing_mask.any():
                    center = residual[missing_mask].mean()
                else:
                    center = residual.mean()
                learned_gates[modality] = 1.0 + alpha * (residual - center)
            metrics["completion_gate_missing_residual_centered"] = 1.0
        elif self.use_centered_all_modal_rank_residual_completion_gate:
            alpha = self._completion_gate_residual_alpha()
            logits = torch.stack([gate_logits[modality] for modality in self.modalities], dim=0)
            centered_logits = logits - logits.mean(dim=0, keepdim=True)
            centered_gates = 1.0 + alpha * torch.tanh(centered_logits)
            centered_gates = centered_gates / centered_gates.mean(dim=0, keepdim=True).clamp_min(1e-6)
            for idx, modality in enumerate(self.modalities):
                learned_gates[modality] = centered_gates[idx]
            metrics["completion_gate_centered_all_modal_mean_normalized"] = 1.0
        elif self.use_normalized_rank_residual_completion_gate:
            missing_count = torch.stack(
                [missing_masks[modality].float() for modality in self.modalities],
                dim=0,
            ).sum(dim=0).clamp_min(1.0).unsqueeze(1)
            missing_gate_sum = torch.stack(
                [
                    torch.where(
                        missing_masks[modality].unsqueeze(1),
                        learned_gates[modality],
                        torch.zeros_like(learned_gates[modality]),
                    )
                    for modality in self.modalities
                ],
                dim=0,
            ).sum(dim=0)
            missing_gate_mean = missing_gate_sum / missing_count
            missing_gate_mean = missing_gate_mean.clamp_min(1e-6)
            for modality in self.modalities:
                learned_gates[modality] = torch.where(
                    missing_masks[modality].unsqueeze(1),
                    learned_gates[modality] / missing_gate_mean,
                    learned_gates[modality],
                )
            metrics["completion_gate_missing_mean_normalized"] = 1.0
        elif self.use_all_normalized_rank_residual_completion_gate:
            all_gate_mean = torch.stack(
                [
                    torch.where(
                        missing_masks[modality].unsqueeze(1),
                        learned_gates[modality],
                        torch.ones_like(learned_gates[modality]),
                    )
                    for modality in self.modalities
                ],
                dim=0,
            ).mean(dim=0).clamp_min(1e-6)
            for modality in self.modalities:
                learned_gates[modality] = torch.where(
                    missing_masks[modality].unsqueeze(1),
                    learned_gates[modality] / all_gate_mean,
                    learned_gates[modality],
                )
            metrics["completion_gate_all_mean_normalized"] = 1.0
        elif self.use_all_modal_rank_residual_completion_gate:
            all_gate_mean = torch.stack(
                [learned_gates[modality] for modality in self.modalities],
                dim=0,
            ).mean(dim=0).clamp_min(1e-6)
            for modality in self.modalities:
                learned_gates[modality] = learned_gates[modality] / all_gate_mean
            metrics["completion_gate_all_modal_mean_normalized"] = 1.0

        for modality in self.modalities:
            learned_gate = learned_gates[modality]
            observed_gate = torch.ones_like(learned_gate)
            missing_mask = missing_masks[modality]
            if (
                self.use_all_modal_rank_residual_completion_gate
                or self.use_softmax_rank_residual_completion_gate
                or self.use_centered_all_modal_rank_residual_completion_gate
            ):
                gate = learned_gate
            else:
                gate = torch.where(missing_mask.unsqueeze(1), learned_gate, observed_gate)
            if self.use_rank_residual_completion_gate and self.completion_gate_tail_quantile < 1.0:
                gate = 1.0 + self.completion_gate_tail_mask.to(gate.dtype) * (gate - 1.0)
            gates[modality] = gate
            if (
                self.use_all_modal_rank_residual_completion_gate
                or self.use_softmax_rank_residual_completion_gate
                or self.use_centered_all_modal_rank_residual_completion_gate
            ):
                identity_regularizers.append(torch.mean((gate - 1.0) ** 2))
                missing_gate_means.append(gate.mean())
                metrics[f"completion_gate_{modality}_all_mean"] = float(
                    gate.detach().mean().cpu()
                )
                metrics[f"completion_gate_{modality}_all_min"] = float(
                    gate.detach().min().cpu()
                )
                metrics[f"completion_gate_{modality}_all_max"] = float(
                    gate.detach().max().cpu()
                )
                observed_mask = masks[modality]
                if observed_mask.any():
                    observed_gate_values = gate[observed_mask]
                    metrics[f"completion_gate_{modality}_observed_mean"] = float(
                        observed_gate_values.detach().mean().cpu()
                    )
                    metrics[f"completion_gate_{modality}_observed_min"] = float(
                        observed_gate_values.detach().min().cpu()
                    )
                    metrics[f"completion_gate_{modality}_observed_max"] = float(
                        observed_gate_values.detach().max().cpu()
                    )
                else:
                    metrics[f"completion_gate_{modality}_observed_mean"] = 1.0
                    metrics[f"completion_gate_{modality}_observed_min"] = 1.0
                    metrics[f"completion_gate_{modality}_observed_max"] = 1.0
            if missing_mask.any():
                missing_gate = gate[missing_mask]
                if (
                    self.use_rank_residual_completion_gate
                    and not self.use_all_modal_rank_residual_completion_gate
                    and not self.use_softmax_rank_residual_completion_gate
                    and not self.use_centered_all_modal_rank_residual_completion_gate
                ):
                    identity_regularizers.append(torch.mean((missing_gate - 1.0) ** 2))
                    missing_gate_means.append(missing_gate.mean())
                elif not self.use_rank_residual_completion_gate:
                    target = torch.as_tensor(
                        self.env.args.completion_gate_target_mean,
                        dtype=missing_gate.dtype,
                        device=missing_gate.device,
                    )
                    regularizers.append(torch.mean((missing_gate - target) ** 2))
                metrics[f"completion_gate_{modality}_missing_mean"] = float(
                    missing_gate.detach().mean().cpu()
                )
                metrics[f"completion_gate_{modality}_missing_min"] = float(
                    missing_gate.detach().min().cpu()
                )
                metrics[f"completion_gate_{modality}_missing_max"] = float(
                    missing_gate.detach().max().cpu()
                )
            else:
                metrics[f"completion_gate_{modality}_missing_mean"] = 1.0
                metrics[f"completion_gate_{modality}_missing_min"] = 1.0
                metrics[f"completion_gate_{modality}_missing_max"] = 1.0

        if self.use_rank_residual_completion_gate:
            metrics["completion_gate_no_residual_alpha"] = float(
                bool(getattr(self.env.args, "completion_gate_no_residual_alpha", 0))
            )
            metrics["completion_gate_effective_residual_alpha"] = float(
                self._completion_gate_residual_alpha()
            )
            metrics["completion_gate_tail_quantile"] = self.completion_gate_tail_quantile
            metrics["completion_gate_tail_active_fraction"] = float(
                self.completion_gate_tail_mask.detach().mean().cpu()
            )
            metrics["completion_gate_tail_degree_threshold"] = self.completion_gate_tail_degree_threshold
            weighted_regularizers = []
            if identity_regularizers:
                identity_regularizer = torch.stack(identity_regularizers).mean()
                weighted_regularizers.append(
                    self.env.args.completion_gate_identity_coeff * identity_regularizer
                )
                metrics["completion_gate_identity_regularizer"] = float(
                    identity_regularizer.detach().cpu()
                )
            if len(missing_gate_means) > 1:
                stacked_means = torch.stack(missing_gate_means)
                balance_regularizer = torch.var(stacked_means, unbiased=False)
                weighted_regularizers.append(
                    self.env.args.completion_gate_balance_coeff * balance_regularizer
                )
                metrics["completion_gate_balance_regularizer"] = float(
                    balance_regularizer.detach().cpu()
                )
                metrics["completion_gate_modality_mean_gap"] = float(
                    (stacked_means.max() - stacked_means.min()).detach().cpu()
                )
            if weighted_regularizers:
                self.latest_completion_gate_regularizer = torch.stack(weighted_regularizers).sum()
                metrics["completion_gate_regularizer"] = float(
                    self.latest_completion_gate_regularizer.detach().cpu()
                )
        elif regularizers:
            self.latest_completion_gate_regularizer = torch.stack(regularizers).mean()
            metrics["completion_gate_regularizer"] = float(
                self.latest_completion_gate_regularizer.detach().cpu()
            )
        self.latest_completion_gate_metrics = metrics
        return gates

    def completion_gate_regularization_loss(self):
        if not self.use_completion_gate:
            return torch.zeros((), device=self.env.device)
        return self.latest_completion_gate_regularizer

    def compute_recommendation_embeddings(self, raw_features=None, allow_modal_grad=False, deterministic=False):
        raw_features = raw_features or self._current_raw_modal_features()
        user_id_emb = self.user_emb.weight
        modal_features = self.get_recommender_modal_features(
            raw_features=raw_features,
            allow_imputer_grad=allow_modal_grad,
        )
        reliability_gates = self._completion_reliability_gates(raw_features=raw_features)
        observed_masks = self._missing_masks(raw_features=raw_features)
        user_outputs = {}
        item_outputs = {}
        ungated_item_outputs = {}
        for modality in self.modalities:
            modal_user_emb, modal_item_emb = getattr(self, f"{modality}_gcn")(
                modal_features[modality],
                user_id_emb,
                skip_mlp=self._gcn_skip_mlp(),
            )
            modal_item_emb = self._apply_item_graph_modal_residual(
                modal_item_emb, observed_masks[modality], modality=modality
            )
            ungated_item_outputs[modality] = modal_item_emb
            if not self.use_rum_fusion:
                modal_item_emb = modal_item_emb * reliability_gates[modality]
            user_outputs[modality] = modal_user_emb
            item_outputs[modality] = modal_item_emb

        if self.use_rank_residual_completion_gate and not self.use_rum_fusion and not allow_modal_grad:
            ungated_item_outputs = {
                modality: item_emb.detach()
                for modality, item_emb in ungated_item_outputs.items()
            }
            item_outputs = {
                modality: item_emb * reliability_gates[modality]
                for modality, item_emb in ungated_item_outputs.items()
            }

        user_emb = user_id_emb + sum(user_outputs.values()) / len(user_outputs)
        if self.use_rank_residual_completion_gate and not self.use_rum_fusion:
            ungated_item_source = self._fuse_item_sources(
                ungated_item_outputs,
                modal_features=modal_features,
                raw_features=raw_features,
            )
            gated_item_source = self._fuse_item_sources(
                item_outputs,
                modal_features=modal_features,
                raw_features=raw_features,
            )
            mix_alpha = self._completion_gate_mix_alpha()
            item_source = (1.0 - mix_alpha) * ungated_item_source + mix_alpha * gated_item_source
            self._record_completion_gate_mix_alpha(mix_alpha)
        else:
            item_source = self._fuse_item_sources(
                item_outputs,
                modal_features=modal_features,
                raw_features=raw_features,
            )

        if not allow_modal_grad and not (self.use_rank_residual_completion_gate and not self.use_rum_fusion):
            item_source = item_source.detach()

        item_emb = self._apply_fusion(item_source, deterministic=deterministic)

        user_emb = torch.nan_to_num(user_emb, nan=0.0, posinf=0.0, neginf=0.0)
        item_emb = torch.nan_to_num(item_emb, nan=0.0, posinf=0.0, neginf=0.0)
        return user_emb, item_emb

    def compute_task_aware_distillation_losses(
        self,
        batch_users,
        batch_pos_items,
        batch_neg_items,
        student_user_emb=None,
        student_item_emb=None,
    ):
        zero = torch.zeros((), device=self.env.device)
        return zero, zero

    def compute_adapter_alignment_loss(self, item_ids, pseudo_ratio=1.0):
        """Align decoupled completed representations to the recommendation space.

        The target modality is pseudo-masked only when it is observed in the
        current training mask and at least one other modality remains observed.
        This keeps the loss within train-visible modalities and avoids using
        hidden raw features for truly missing train items.
        """
        zero = torch.zeros((), device=self.env.device)
        if (
            not self.use_decoupled_latent_bridge
            or self.disable_imputation
            or item_ids is None
            or item_ids.numel() == 0
        ):
            return zero

        item_ids = torch.unique(item_ids.detach()).long()
        item_ids = item_ids[(item_ids >= 0) & (item_ids < self.m_item)]
        if item_ids.numel() == 0:
            return zero

        raw_features = self._current_raw_modal_features()
        raw_batch = {modality: raw_features[modality][item_ids] for modality in self.modalities}
        masks = self._missing_masks(raw_features=raw_batch)
        if len(self.modalities) < 2:
            return zero

        completion_projected = self.project_features(raw_features=raw_batch)
        recommendation_projected = self.project_recommendation_features(raw_features=raw_batch)
        ratio = min(max(float(pseudo_ratio), 0.0), 1.0)
        losses = []

        for target_modality in self.modalities:
            eligible = masks[target_modality].clone()
            has_other_observed = torch.zeros_like(eligible)
            for source_modality in self.modalities:
                if source_modality == target_modality:
                    continue
                has_other_observed = has_other_observed | masks[source_modality]
            eligible = eligible & has_other_observed
            if ratio < 1.0:
                eligible = eligible & (torch.rand_like(eligible.float()) < ratio)
            if not eligible.any():
                continue

            pseudo_masks = {modality: mask.clone() for modality, mask in masks.items()}
            pseudo_masks[target_modality][eligible] = False
            completed = self._build_completed_features(
                completion_projected,
                pseudo_masks,
                detach_imputed=True,
            )
            adapted = self.adapt_completed_to_recommendation(completed)[target_modality][eligible]
            target = recommendation_projected[target_modality][eligible].detach()
            losses.append(1.0 - F.cosine_similarity(adapted, target, dim=-1).mean())

        if not losses:
            return zero
        return torch.stack(losses).mean()

    def _recommendation_frontend_item_embeddings(self, raw_features, item_ids, allow_modal_grad=False):
        modal_features = self.get_recommender_modal_features(
            raw_features=raw_features,
            allow_imputer_grad=allow_modal_grad,
        )
        item_outputs = {}
        modal_gcns = {
            modality: getattr(self, f"{modality}_gcn")
            for modality in self.modalities
        }
        observed_masks = self._missing_masks(raw_features)
        for modality in self.modalities:
            features = modal_features[modality][item_ids]
            item_output = features if self._gcn_skip_mlp() else modal_gcns[modality].MLP(features)
            item_outputs[modality] = F.normalize(
                torch.nan_to_num(item_output, nan=0.0, posinf=0.0, neginf=0.0),
                dim=-1,
            )
        modal_subset = {m: modal_features[m][item_ids] for m in self.modalities}
        raw_subset = {m: raw_features[m][item_ids] for m in self.modalities}
        observed_subset = {m: observed_masks[m][item_ids] for m in self.modalities}
        item_source = self._fuse_item_sources(
            item_outputs,
            modal_features=modal_subset,
            raw_features=raw_subset,
            observed_masks=observed_subset,
        )
        return torch.nan_to_num(self._apply_fusion(item_source, deterministic=True), nan=0.0, posinf=0.0, neginf=0.0)

    def _recommendation_gcn_modality_item_embeddings(
        self,
        raw_features=None,
        allow_modal_grad=False,
        apply_item_graph=False,
    ):
        raw_features = raw_features or self._current_raw_modal_features()
        modal_features = self.get_recommender_modal_features(
            raw_features=raw_features,
            allow_imputer_grad=allow_modal_grad,
        )
        user_id_emb = self.user_emb.weight
        modal_gcns = {
            modality: getattr(self, f"{modality}_gcn")
            for modality in self.modalities
        }

        item_outputs = {}
        observed_masks = self._missing_masks(raw_features)
        for modality in self.modalities:
            _, item_emb = modal_gcns[modality](
                modal_features[modality],
                user_id_emb,
                skip_mlp=self._gcn_skip_mlp(),
            )
            if apply_item_graph:
                item_emb = self._apply_item_graph_modal_residual(
                    item_emb,
                    observed_mask=observed_masks[modality],
                    modality=modality,
                )
            item_outputs[modality] = F.normalize(
                torch.nan_to_num(item_emb, nan=0.0, posinf=0.0, neginf=0.0),
                dim=-1,
            )
        return item_outputs

    def _recommendation_frontend_modality_item_embeddings(
        self,
        raw_features=None,
        allow_modal_grad=False,
    ):
        """Modality item representations before any graph propagation."""
        raw_features = raw_features or self._current_raw_modal_features()
        modal_features = self.get_recommender_modal_features(
            raw_features=raw_features,
            allow_imputer_grad=allow_modal_grad,
        )
        item_outputs = {}
        for modality in self.modalities:
            modal_gcn = getattr(self, f"{modality}_gcn")
            item_emb = (
                modal_features[modality]
                if self._gcn_skip_mlp()
                else modal_gcn.MLP(modal_features[modality])
            )
            item_outputs[modality] = F.normalize(
                torch.nan_to_num(item_emb, nan=0.0, posinf=0.0, neginf=0.0),
                dim=-1,
            )
        return item_outputs

    def compute_true_missing_gcn_infonce_loss(
        self,
        item_ids,
        user_ids=None,
        temperature=0.2,
        bank_size=256,
        user_bank_size=256,
        allow_modal_grad=False,
    ):
        """InfoNCE for real missing modalities after modality-specific GCNs.

        Anchor: imputed representation of a truly missing modality.
        Positive: same item's observed modality representations.
        Negatives: other items whose same modality is observed.
        """
        zero = torch.zeros((), device=self.env.device)
        if self.disable_imputation or item_ids is None or item_ids.numel() == 0:
            return zero

        raw_features = self._current_raw_modal_features()
        masks = self._missing_masks(raw_features)
        item_ids = torch.unique(item_ids.detach()).long()
        item_ids = item_ids[(item_ids >= 0) & (item_ids < self.m_item)]
        if item_ids.numel() < 2:
            return zero

        objective = getattr(self.env.args, "rec_neighbor_cl_objective", "infonce")
        max_bank_size = int(bank_size or 0)
        bank_ids = item_ids
        if (
            objective != "positive_cosine"
            and max_bank_size > 0
            and bank_ids.numel() > max_bank_size
        ):
            perm = torch.randperm(bank_ids.numel(), device=self.env.device)[:max_bank_size]
            bank_ids = bank_ids[perm]
        if bank_ids.numel() < 2:
            return zero

        similarity_space = getattr(
            self.env.args,
            "rec_neighbor_cl_similarity_space",
            "embedding",
        )
        user_basis = None
        if similarity_space == "user_preference":
            if user_ids is None or user_ids.numel() == 0:
                return zero
            profile_user_ids = torch.unique(user_ids.detach()).long()
            profile_user_ids = profile_user_ids[
                (profile_user_ids >= 0) & (profile_user_ids < self.n_user)
            ]
            max_user_bank_size = int(user_bank_size or 0)
            if (
                max_user_bank_size > 0
                and profile_user_ids.numel() > max_user_bank_size
            ):
                perm = torch.randperm(
                    profile_user_ids.numel(),
                    device=self.env.device,
                )[:max_user_bank_size]
                profile_user_ids = profile_user_ids[perm]
            if profile_user_ids.numel() < 2:
                return zero
            user_basis = F.normalize(
                self.user_emb.weight[profile_user_ids].detach(),
                dim=-1,
            )
        elif similarity_space != "embedding":
            raise ValueError(
                f"Unsupported rec_neighbor_cl_similarity_space: {similarity_space}"
            )

        def to_contrast_space(item_embeddings):
            if user_basis is not None:
                item_embeddings = torch.matmul(
                    item_embeddings,
                    user_basis.transpose(0, 1),
                )
            return F.normalize(
                torch.nan_to_num(
                    item_embeddings,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                dim=-1,
            )

        cl_stage = getattr(self.env.args, "rec_neighbor_cl_stage", "gcn")
        if cl_stage == "frontend":
            gcn_item_outputs = self._recommendation_frontend_modality_item_embeddings(
                raw_features=raw_features,
                allow_modal_grad=allow_modal_grad,
            )
        else:
            gcn_item_outputs = self._recommendation_gcn_modality_item_embeddings(
                raw_features=raw_features,
                allow_modal_grad=allow_modal_grad,
                apply_item_graph=(cl_stage == "post_item_graph"),
            )
        temp = max(float(temperature), 1e-6)
        losses = []
        loss_weights = []
        modality_count = len(self.modalities)
        anchor_weighting = getattr(
            self.env.args,
            "rec_neighbor_cl_anchor_weighting",
            "uniform",
        )
        reliabilities = None
        if anchor_weighting == "posterior_reliability":
            reliabilities = self._posterior_completion_reliabilities(masks)
        false_negative_threshold = float(
            getattr(self.env.args, "rec_neighbor_cl_false_negative_threshold", 1.1)
        )
        positive_source = getattr(
            self.env.args,
            "rec_neighbor_cl_positive_source",
            "cross_modal",
        )
        negative_source = getattr(
            self.env.args,
            "rec_neighbor_cl_negative_source",
            "same_modal",
        )
        if positive_source != "cross_modal" and negative_source == "cross_modal":
            raise ValueError(
                "rec_neighbor_cl_negative_source=cross_modal requires "
                "rec_neighbor_cl_positive_source=cross_modal"
            )
        for modality in self.modalities:
            anchor_mask = ~masks[modality][item_ids]
            if modality_count > 1:
                has_observed_other = torch.zeros_like(anchor_mask)
                for observed_modality in self.modalities:
                    if observed_modality == modality:
                        continue
                    has_observed_other = has_observed_other | masks[observed_modality][item_ids]
                anchor_mask = anchor_mask & has_observed_other
            anchor_ids = item_ids[anchor_mask]
            if anchor_ids.numel() == 0:
                continue

            if positive_source == "cf_neighbor":
                cf_graph = self.ItemItemGraphs.get("cf")
                if cf_graph is None:
                    raise RuntimeError(
                        "rec_neighbor_cl_positive_source=cf_neighbor requires "
                        "ItemItemGraphs['cf']"
                    )
                observed = masks[modality].to(gcn_item_outputs[modality].dtype).unsqueeze(1)
                positive_sum_all = torch.sparse.mm(
                    cf_graph,
                    gcn_item_outputs[modality] * observed,
                )
                positive_weight_all = torch.sparse.mm(cf_graph, observed)
                has_positive = positive_weight_all[anchor_ids, 0] > 0
                anchor_ids = anchor_ids[has_positive]
                if anchor_ids.numel() == 0:
                    continue
                positive_emb = F.normalize(
                    positive_sum_all[anchor_ids]
                    / positive_weight_all[anchor_ids].clamp_min(1e-8),
                    dim=-1,
                )
                positive_emb = to_contrast_space(positive_emb)
            else:
                positive_parts = []
                positive_weights = []
                for observed_modality in self.modalities:
                    if observed_modality == modality:
                        continue
                    observed_mask = masks[observed_modality][anchor_ids].float().unsqueeze(1)
                    positive_parts.append(
                        to_contrast_space(
                            gcn_item_outputs[observed_modality][anchor_ids]
                        )
                        * observed_mask
                    )
                    positive_weights.append(observed_mask)
                if not positive_parts:
                    continue
                positive_sum = torch.stack(positive_parts, dim=0).sum(dim=0)
                positive_count = torch.stack(positive_weights, dim=0).sum(dim=0).clamp_min(1.0)
                positive_emb = F.normalize(positive_sum / positive_count, dim=-1)

            anchor_emb = to_contrast_space(gcn_item_outputs[modality][anchor_ids])
            positive_target = positive_emb.detach()

            if objective == "positive_cosine":
                per_anchor_loss = 1.0 - (anchor_emb * positive_target).sum(dim=1)
            else:
                if negative_source == "cross_modal":
                    negative_parts = []
                    negative_weights = []
                    for observed_modality in self.modalities:
                        if observed_modality == modality:
                            continue
                        observed_mask = masks[observed_modality][bank_ids].float().unsqueeze(1)
                        negative_parts.append(
                            to_contrast_space(
                                gcn_item_outputs[observed_modality][bank_ids]
                            )
                            * observed_mask
                        )
                        negative_weights.append(observed_mask)
                    if not negative_parts:
                        continue
                    negative_sum = torch.stack(negative_parts, dim=0).sum(dim=0)
                    negative_count = torch.stack(negative_weights, dim=0).sum(dim=0)
                    valid_negative = negative_count[:, 0] > 0
                    negative_ids = bank_ids[valid_negative]
                    if negative_ids.numel() == 0:
                        continue
                    negative_emb = F.normalize(
                        negative_sum[valid_negative]
                        / negative_count[valid_negative].clamp_min(1.0),
                        dim=-1,
                    )
                else:
                    negative_ids = bank_ids[masks[modality][bank_ids]]
                    if negative_ids.numel() == 0:
                        continue
                    negative_emb = to_contrast_space(
                        gcn_item_outputs[modality][negative_ids]
                    )
                negative_target = negative_emb.detach()
                pos_logits = (anchor_emb * positive_target).sum(dim=1, keepdim=True) / temp
                neg_logits = torch.matmul(anchor_emb, negative_target.transpose(0, 1)) / temp
                same_item = anchor_ids.unsqueeze(1).eq(negative_ids.unsqueeze(0))
                neg_logits = neg_logits.masked_fill(same_item, -1e9)
                if false_negative_threshold <= 1.0:
                    teacher_negative_similarity = torch.matmul(
                        positive_target.detach(),
                        negative_target.detach().transpose(0, 1),
                    )
                    false_negatives = teacher_negative_similarity >= false_negative_threshold
                    neg_logits = neg_logits.masked_fill(false_negatives, -1e9)
                logits = torch.cat([pos_logits, neg_logits], dim=1)
                targets = torch.zeros(logits.size(0), dtype=torch.long, device=self.env.device)
                per_anchor_loss = F.cross_entropy(logits, targets, reduction="none")

            if reliabilities is None:
                anchor_weights = torch.ones_like(per_anchor_loss)
            else:
                anchor_weights = reliabilities[modality][anchor_ids].to(per_anchor_loss.dtype)
            anchor_weight_sum = anchor_weights.sum().clamp_min(1e-8)
            losses.append((per_anchor_loss * anchor_weights).sum() / anchor_weight_sum)
            if reliabilities is None:
                loss_weights.append(torch.ones((), device=self.env.device))
            else:
                loss_weights.append(anchor_weights.mean().clamp_min(1e-8))

        if not losses:
            return zero
        stacked_losses = torch.stack(losses)
        stacked_weights = torch.stack(loss_weights)
        return (stacked_losses * stacked_weights).sum() / stacked_weights.sum().clamp_min(1e-8)

    @torch.no_grad()
    def compute_imputation_representation_metrics(self, split="test", include_random_baseline=True):
        full_raw = self.get_split_raw_modal_features(split=split, full=True)
        missing_raw = self.get_split_raw_modal_features(split=split, full=False)
        missing_items, missing_indicators = self.get_missing_item_metadata(split=split)

        full_projected = self.project_features(raw_features=full_raw)
        missing_projected = self.project_features(raw_features=missing_raw)
        missing_masks = self._missing_masks(raw_features=missing_raw)
        completed_missing = self.impute_modalities(
            missing_projected,
            missing_masks,
            item_ids=torch.arange(missing_projected[self.modalities[0]].size(0), device=self.env.device),
        )

        results = {}
        total_count = 0
        weighted_mse = 0.0
        weighted_cosine = 0.0
        weighted_random_mse = 0.0
        weighted_random_cosine = 0.0

        for modality_idx, modality in enumerate(self.modalities):
            modality_selector = missing_indicators == modality_idx
            modality_items = missing_items[modality_selector]
            if modality_items.numel() == 0:
                continue

            real_repr = full_projected[modality][modality_items]
            imputed_repr = completed_missing[modality][modality_items]

            mse_value = F.mse_loss(imputed_repr, real_repr, reduction="mean").item()
            cosine_value = F.cosine_similarity(imputed_repr, real_repr, dim=-1).mean().item()

            metric = {
                "count": int(modality_items.numel()),
                "mse": float(mse_value),
                "cosine": float(cosine_value),
            }

            if include_random_baseline:
                random_ids = torch.randint(
                    0,
                    full_projected[modality].size(0),
                    (modality_items.numel(),),
                    device=self.env.device,
                )
                random_repr = full_projected[modality][random_ids]
                random_mse = F.mse_loss(random_repr, real_repr, reduction="mean").item()
                random_cosine = F.cosine_similarity(random_repr, real_repr, dim=-1).mean().item()
                metric["random_mse"] = float(random_mse)
                metric["random_cosine"] = float(random_cosine)
                weighted_random_mse += random_mse * modality_items.numel()
                weighted_random_cosine += random_cosine * modality_items.numel()

            results[modality] = metric
            total_count += int(modality_items.numel())
            weighted_mse += mse_value * modality_items.numel()
            weighted_cosine += cosine_value * modality_items.numel()

        if total_count == 0:
            results["_overall"] = {"count": 0, "mse": 0.0, "cosine": 0.0}
            if include_random_baseline:
                results["_overall"]["random_mse"] = 0.0
                results["_overall"]["random_cosine"] = 0.0
            return results

        results["_overall"] = {
            "count": total_count,
            "mse": float(weighted_mse / total_count),
            "cosine": float(weighted_cosine / total_count),
        }
        if include_random_baseline:
            results["_overall"]["random_mse"] = float(weighted_random_mse / total_count)
            results["_overall"]["random_cosine"] = float(weighted_random_cosine / total_count)
        return results

    @torch.no_grad()
    def compute_missing_decode_metrics(self, split="test", include_random_baseline=True):
        if not self.use_decode_head:
            results = {"_overall": {"count": 0, "mse": 0.0, "cosine": 0.0}}
            if include_random_baseline:
                results["_overall"]["random_mse"] = 0.0
                results["_overall"]["random_cosine"] = 0.0
            return results

        full_raw = self.get_split_raw_modal_features(split=split, full=True)
        missing_raw = self.get_split_raw_modal_features(split=split, full=False)
        missing_items, missing_indicators = self.get_missing_item_metadata(split=split)
        missing_masks = self._missing_masks(raw_features=missing_raw)
        missing_projected = self.project_features(raw_features=missing_raw)
        completed_missing = self.impute_modalities(
            missing_projected,
            missing_masks,
            item_ids=torch.arange(missing_projected[self.modalities[0]].size(0), device=self.env.device),
        )
        decoded_missing = self.decode_completed_to_raw(completed_missing)

        results = {}
        total_count = 0
        weighted_mse = 0.0
        weighted_cosine = 0.0
        weighted_random_mse = 0.0
        weighted_random_cosine = 0.0

        for modality_idx, modality in enumerate(self.modalities):
            modality_selector = missing_indicators == modality_idx
            modality_items = missing_items[modality_selector]
            if modality_items.numel() == 0:
                continue

            decoded = decoded_missing[modality][modality_items]
            real_raw = self._decoder_target(
                modality,
                full_raw[modality][modality_items],
                item_ids=modality_items,
            )

            mse_value = F.mse_loss(decoded, real_raw, reduction="mean").item()
            cosine_value = F.cosine_similarity(decoded, real_raw, dim=-1).mean().item()

            metric = {
                "count": int(modality_items.numel()),
                "mse": float(mse_value),
                "cosine": float(cosine_value),
            }

            if include_random_baseline:
                random_ids = torch.randint(
                    0,
                    full_raw[modality].size(0),
                    (modality_items.numel(),),
                    device=self.env.device,
                )
                random_raw = self._decoder_target(
                    modality,
                    full_raw[modality][random_ids],
                    item_ids=random_ids,
                )
                random_mse = F.mse_loss(random_raw, real_raw, reduction="mean").item()
                random_cosine = F.cosine_similarity(random_raw, real_raw, dim=-1).mean().item()
                metric["random_mse"] = float(random_mse)
                metric["random_cosine"] = float(random_cosine)
                weighted_random_mse += random_mse * modality_items.numel()
                weighted_random_cosine += random_cosine * modality_items.numel()

            results[modality] = metric
            total_count += int(modality_items.numel())
            weighted_mse += mse_value * modality_items.numel()
            weighted_cosine += cosine_value * modality_items.numel()

        if total_count == 0:
            results["_overall"] = {"count": 0, "mse": 0.0, "cosine": 0.0}
            if include_random_baseline:
                results["_overall"]["random_mse"] = 0.0
                results["_overall"]["random_cosine"] = 0.0
            return results

        results["_overall"] = {
            "count": total_count,
            "mse": float(weighted_mse / total_count),
            "cosine": float(weighted_cosine / total_count),
        }
        if include_random_baseline:
            results["_overall"]["random_mse"] = float(weighted_random_mse / total_count)
            results["_overall"]["random_cosine"] = float(weighted_random_cosine / total_count)
        return results

    @torch.no_grad()
    def compute_pseudo_decode_metrics(self, split="train", include_random_baseline=True, ratio=0.3):
        """Evaluate raw-space decoding without exposing truly missing raw targets."""
        if not self.use_decode_head:
            results = {"_overall": {"count": 0, "mse": 0.0, "cosine": 0.0}}
            if include_random_baseline:
                results["_overall"].update({"random_mse": 0.0, "random_cosine": 0.0})
            return results

        raw_features = self.get_split_raw_modal_features(split=split, full=False)
        masks = self._missing_masks(raw_features=raw_features)
        split_offset = {"train": 0, "val": 1, "test": 2, "imputation_val": 3}.get(split, 4)
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(int(getattr(self.env.args, "seed", 0)) + 104729 * (split_offset + 1))
        pseudo_masks, pseudo_selected = self._sample_pseudo_missing_masks(
            masks,
            ratio=float(ratio),
            generator=generator,
        )
        if pseudo_masks is None or pseudo_selected is None:
            results = {"_overall": {"count": 0, "mse": 0.0, "cosine": 0.0}}
            if include_random_baseline:
                results["_overall"].update({"random_mse": 0.0, "random_cosine": 0.0})
            return results

        projected = self.project_features(raw_features=raw_features)
        completed = self._build_completed_features(
            projected,
            pseudo_masks,
            detach_imputed=True,
        )
        decoded = self.decode_completed_to_raw(completed)

        results = {}
        total_count = 0
        weighted = {"mse": 0.0, "cosine": 0.0, "random_mse": 0.0, "random_cosine": 0.0}
        for modality in self.modalities:
            selected = pseudo_selected[modality]
            count = int(selected.sum().item())
            if count == 0:
                continue
            prediction = decoded[modality][selected]
            selected_item_ids = torch.arange(
                raw_features[modality].size(0), device=self.env.device
            )[selected]
            target = self._decoder_target(
                modality,
                raw_features[modality][selected],
                item_ids=selected_item_ids,
            )
            mse_value = F.mse_loss(prediction, target, reduction="mean").item()
            cosine_value = F.cosine_similarity(prediction, target, dim=-1).mean().item()
            metric = {"count": count, "mse": float(mse_value), "cosine": float(cosine_value)}
            weighted["mse"] += mse_value * count
            weighted["cosine"] += cosine_value * count
            if include_random_baseline:
                random_ids = torch.randint(
                    0,
                    raw_features[modality].size(0),
                    (count,),
                    device=self.env.device,
                    generator=generator,
                )
                random_target = self._decoder_target(
                    modality,
                    raw_features[modality][random_ids],
                    item_ids=random_ids,
                )
                random_mse = F.mse_loss(random_target, target, reduction="mean").item()
                random_cosine = F.cosine_similarity(random_target, target, dim=-1).mean().item()
                metric.update({"random_mse": float(random_mse), "random_cosine": float(random_cosine)})
                weighted["random_mse"] += random_mse * count
                weighted["random_cosine"] += random_cosine * count
            results[modality] = metric
            total_count += count

        results["_overall"] = {
            "count": total_count,
            "mse": float(weighted["mse"] / max(total_count, 1)),
            "cosine": float(weighted["cosine"] / max(total_count, 1)),
        }
        if include_random_baseline:
            results["_overall"].update({
                "random_mse": float(weighted["random_mse"] / max(total_count, 1)),
                "random_cosine": float(weighted["random_cosine"] / max(total_count, 1)),
            })
        return results

    @torch.no_grad()
    def compute_stage1_heldout_metrics(self, split="val", include_random_baseline=True):
        pseudo_shared_metrics = self.compute_pseudo_shared_metrics(
            split=split,
            include_random_baseline=include_random_baseline,
        )
        pseudo_shared_overall = pseudo_shared_metrics.get("_overall", {})
        pseudo_shared_cosine = float(pseudo_shared_overall.get("cosine", 0.0))
        random_pseudo_shared_cosine = float(pseudo_shared_overall.get("random_cosine", 0.0))

        metrics = {
            "split": split,
            "pseudo_shared_count": int(pseudo_shared_overall.get("count", 0)),
            "pseudo_shared_mse": float(pseudo_shared_overall.get("mse", 0.0)),
            "pseudo_shared_cosine": pseudo_shared_cosine,
            "pseudo_shared_cosine_gap": float(pseudo_shared_cosine - random_pseudo_shared_cosine),
            "pseudo_shared_random_mse": float(pseudo_shared_overall.get("random_mse", 0.0)),
            "pseudo_shared_random_cosine": random_pseudo_shared_cosine,
        }
        if self.use_decode_head:
            pseudo_decode_metrics = self.compute_pseudo_decode_metrics(
                split=split,
                include_random_baseline=include_random_baseline,
            ).get("_overall", {})
            decode_metrics = self.compute_missing_decode_metrics(
                split=split,
                include_random_baseline=include_random_baseline,
            ).get("_overall", {})
            metrics.update({
                "pseudo_decode_count": int(pseudo_decode_metrics.get("count", 0)),
                "pseudo_decode_mse": float(pseudo_decode_metrics.get("mse", 0.0)),
                "pseudo_decode_cosine": float(pseudo_decode_metrics.get("cosine", 0.0)),
                "missing_decode_count": int(decode_metrics.get("count", 0)),
                "missing_decode_mse": float(decode_metrics.get("mse", 0.0)),
                "missing_decode_cosine": float(decode_metrics.get("cosine", 0.0)),
                "missing_decode_random_mse": float(decode_metrics.get("random_mse", 0.0)),
                "missing_decode_random_cosine": float(decode_metrics.get("random_cosine", 0.0)),
            })
        return metrics

    def _collect_observed_projected(self, item_ids=None):
        projected = self.project_features(item_ids=item_ids)
        masks = self._missing_masks()
        if item_ids is not None:
            masks = {modality: mask[item_ids] for modality, mask in masks.items()}

        observed_feats = {}
        for modality in self.modalities:
            observed_feats[modality] = projected[modality][masks[modality]]
        return projected, masks, observed_feats

    def _em_update_generative_params(self, aligned_obs, m, V):
        """EM Step 2: Update generative params via paper Eq. (6).
        m: posterior mean [N, d_beta], V: posterior covariance [d_beta, d_beta].
        Must be called inside torch.no_grad().
        """
        if not self.training or not self._imputer_updates_enabled:
            return
        if not aligned_obs:
            return

        N = m.size(0)
        if N < 2:
            return

        modalities = list(aligned_obs.keys())

        # E[beta beta^T] summed over samples: m^T m + N * V  (corrected N-factor)
        E_beta_betaT_sum = m.T @ m + N * V

        for modality in modalities:
            z = aligned_obs[modality]  # [N, d]
            W_old = self.W[modality]   # [d, d_beta]

            # 1. Update mu (using old W)
            mu_new = z.mean(dim=0) - (W_old @ m.T).mean(dim=1)  # [d]

            # 2. Update W (using new mu)
            z_centered = z - mu_new.unsqueeze(0)  # [N, d]
            numerator = z_centered.T @ m  # [d, d_beta]
            W_new = numerator @ torch.inverse(E_beta_betaT_sum)  # [d, d_beta]

            # 3. Update sigma^2 (using new mu and new W)
            recon_error = z_centered - m @ W_new.T  # [N, d]
            recon_sq = (recon_error ** 2).sum()
            trace_term = torch.trace(W_new.T @ W_new @ V) * N
            sigma2_new = (recon_sq + trace_term) / (N * self.promrl_dim)

            # Direct replacement (no EMA)
            self.mu[modality].copy_(mu_new)
            self.W[modality].copy_(W_new)
            self.log_sigma[modality].fill_(0.5 * torch.log(sigma2_new.clamp(min=1e-6)))

    def queue_em_update(self, aligned_obs, m, V):
        if not self.training or not self._imputer_updates_enabled:
            return
        detached_obs = {k: v.detach() for k, v in aligned_obs.items()}
        self._pending_em_updates.append((detached_obs, m.detach(), V.detach()))

    def has_pending_em_updates(self):
        return bool(self._pending_em_updates)

    @torch.no_grad()
    def apply_pending_em_updates(self):
        if not self._pending_em_updates:
            return
        for aligned_obs, m, V in self._pending_em_updates:
            self._em_update_generative_params(aligned_obs, m, V)
        self._pending_em_updates.clear()

    def compute_contrastive_loss(self, completed_feats):
        """Contrastive loss on completed features (all items, like ProMRL).
        Observed features have grad_fn (trains Contra_head), imputed do not.
        """
        N = next(iter(completed_feats.values())).size(0)
        if N < 2:
            device = next(iter(completed_feats.values())).device
            zero = torch.zeros((), device=device)
            return zero, zero

        all_features = [completed_feats[modality] for modality in self.modalities]
        structure_loss_variant = getattr(
            self.env.args, "structure_loss_variant", "original"
        )
        if structure_loss_variant == "shifted_lifted":
            eigenvalues, item_directions = shifted_relation_lifted_directions(all_features)
            sim = (item_directions @ item_directions.T).pow(2) / self.env.args.tau2
        else:
            eigenvectors, S_V = eigenvalue_computation_pmcl(all_features)
            eigenvalues = S_V ** 2
            principal_eigenvector = eigenvectors[:, :, 0]
            sim = (
                principal_eigenvector @ principal_eigenvector.T
            ) / self.env.args.tau2

        targets = torch.zeros(eigenvalues.size(0), dtype=torch.long, device=eigenvalues.device)
        loss_intra = F.cross_entropy(eigenvalues / self.env.args.tau1, targets)

        bs = sim.size(0)
        targets = torch.arange(bs, device=sim.device)
        loss_inter = F.cross_entropy(sim, targets)

        return loss_intra, loss_inter

    def _compute_itm_logits(self, text_feats, vision_feats):
        query = text_feats.unsqueeze(1)
        context = vision_feats.unsqueeze(1)
        fused, _ = self.itm_cross_attn(query, context, context)
        return self.itm_head(fused.squeeze(1))

    def _stable_negative_sampling_weights(self, logits):
        weights = F.softmax(torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0), dim=1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        weights.fill_diagonal_(0)
        empty_rows = weights.sum(dim=1) <= 0
        if empty_rows.any():
            fallback = torch.ones_like(weights)
            fallback.fill_diagonal_(0)
            weights = torch.where(empty_rows.unsqueeze(1), fallback, weights)
        return weights

    def compute_itm_loss(self, item_ids, completed_feats=None):
        projected = self.project_features(item_ids=item_ids)
        if completed_feats is None:
            masks = self._missing_masks()
            masks = {modality: mask[item_ids] for modality, mask in masks.items()}
            completed_feats = self._build_completed_features(projected, masks, item_ids=item_ids)

        feat_t = completed_feats["t"]
        feat_v = completed_feats["v"]
        if feat_t.size(0) < 2:
            zero = projected["t"].new_zeros(())
            return zero, zero

        sim_t2v = feat_t @ feat_v.T / self.itm_temp
        sim_v2t = feat_v @ feat_t.T / self.itm_temp

        with torch.no_grad():
            weights_t2v = self._stable_negative_sampling_weights(sim_t2v)
            weights_v2t = self._stable_negative_sampling_weights(sim_v2t)

            neg_v_idx = torch.multinomial(weights_t2v, 1).squeeze(1)
            neg_t_idx = torch.multinomial(weights_v2t, 1).squeeze(1)

        pos_logits = self._compute_itm_logits(feat_t, feat_v)
        neg_v_logits = self._compute_itm_logits(feat_t, feat_v[neg_v_idx])
        neg_t_logits = self._compute_itm_logits(feat_t[neg_t_idx], feat_v)

        logits = torch.cat([pos_logits, neg_v_logits, neg_t_logits], dim=0)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        labels[: feat_t.size(0)] = 1
        loss_itm_raw = F.cross_entropy(logits, labels)
        return loss_itm_raw, self.lambda_itm * loss_itm_raw

    def _sample_pseudo_missing_masks(self, masks, ratio, generator=None):
        if ratio <= 0.0:
            return None, None

        obs_stack = torch.stack([masks[modality] for modality in self.modalities], dim=1)
        observed_count = obs_stack.sum(dim=1)
        eligible_items = observed_count > 1
        if not eligible_items.any():
            return None, None

        pseudo_masks = {
            modality: mask.clone()
            for modality, mask in masks.items()
        }
        pseudo_selected = {
            modality: torch.zeros_like(mask, dtype=torch.bool)
            for modality, mask in masks.items()
        }

        candidate_indices = torch.nonzero(eligible_items, as_tuple=False).squeeze(1)
        if candidate_indices.numel() == 0:
            return None, None

        candidate_keep = torch.rand(candidate_indices.size(0), device=self.env.device, generator=generator) < ratio
        selected_items = candidate_indices[candidate_keep]
        if selected_items.numel() == 0:
            return None, None

        selected_obs = obs_stack[selected_items]
        random_scores = torch.rand(selected_obs.size(), device=self.env.device, generator=generator)
        masked_scores = random_scores.masked_fill(~selected_obs, -1.0)
        chosen_modalities = masked_scores.argmax(dim=1)

        for modality_idx, modality in enumerate(self.modalities):
            item_selector = chosen_modalities == modality_idx
            if not item_selector.any():
                continue
            modality_items = selected_items[item_selector]
            pseudo_masks[modality][modality_items] = False
            pseudo_selected[modality][modality_items] = True

        return pseudo_masks, pseudo_selected

    def _compute_observed_rec_loss(self, projected, masks, stage, item_ids=None):
        pattern_keys = torch.stack([masks[modality] for modality in self.modalities], dim=1)
        rec_terms = []
        rec_weights = []
        for pattern in torch.unique(pattern_keys, dim=0):
            observed_modalities = [
                self.modalities[i]
                for i in range(len(self.modalities))
                if bool(pattern[i])
            ]
            if not observed_modalities:
                continue

            selector = (pattern_keys == pattern.unsqueeze(0)).all(dim=1)
            if selector.sum() == 0:
                continue

            batch_data = {
                modality: projected[modality][selector]
                for modality in observed_modalities
            }
            posterior_mean, posterior_cov = update_posterior(
                batch_data,
                {modality: self.W[modality] for modality in self.modalities},
                {modality: self.mu[modality] for modality in self.modalities},
                {modality: self.log_sigma[modality].squeeze() for modality in self.modalities},
                observed_modalities,
                self.d_beta,
            )
            if self._imputer_updates_enabled and stage in (
                "imputer_param",
                "joint",
            ):
                self.queue_em_update(batch_data, posterior_mean, posterior_cov)
            pattern_rec_loss, _ = compute_nll_loss(
                batch_data,
                posterior_mean.detach(),
                {modality: self.W[modality] for modality in self.modalities},
                {modality: self.mu[modality] for modality in self.modalities},
                {modality: self.log_sigma[modality].squeeze() for modality in self.modalities},
                observed_modalities,
            )
            rec_terms.append(pattern_rec_loss)
            rec_weights.append(selector.sum().float())

        if not rec_terms:
            return torch.zeros((), device=self.env.device)
        return (
            torch.stack(rec_terms) * torch.stack(rec_weights)
        ).sum() / torch.stack(rec_weights).sum().clamp_min(1.0)

    @torch.no_grad()
    def compute_pseudo_shared_metrics(self, split="val", include_random_baseline=True, ratio=None):
        raw_features = self.get_split_raw_modal_features(split=split, full=False)
        masks = self._missing_masks(raw_features=raw_features)
        if ratio is None:
            ratio = 0.3

        split_offset = {"train": 0, "val": 1, "test": 2, "imputation_val": 3}.get(split, 4)
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(int(getattr(self.env.args, "seed", 0)) + 7919 * (split_offset + 1))

        pseudo_masks, pseudo_selected = self._sample_pseudo_missing_masks(
            masks,
            ratio=ratio,
            generator=generator,
        )
        if pseudo_masks is None or pseudo_selected is None:
            results = {"_overall": {"count": 0, "mse": 0.0, "cosine": 0.0}}
            if include_random_baseline:
                results["_overall"]["random_mse"] = 0.0
                results["_overall"]["random_cosine"] = 0.0
            return results

        projected = self.project_features(raw_features=raw_features)
        pseudo_completed = self._build_completed_features(projected, pseudo_masks)

        results = {}
        total_count = 0
        weighted_mse = 0.0
        weighted_cosine = 0.0
        weighted_random_mse = 0.0
        weighted_random_cosine = 0.0

        for modality in self.modalities:
            selected_mask = pseudo_selected[modality]
            count = int(selected_mask.sum().item())
            if count == 0:
                continue

            target_repr = projected[modality][selected_mask]
            imputed_repr = pseudo_completed[modality][selected_mask]
            mse_value = F.mse_loss(imputed_repr, target_repr, reduction="mean").item()
            cosine_value = F.cosine_similarity(imputed_repr, target_repr, dim=-1).mean().item()

            metric = {
                "count": count,
                "mse": float(mse_value),
                "cosine": float(cosine_value),
            }

            if include_random_baseline:
                random_ids = torch.randint(
                    0,
                    projected[modality].size(0),
                    (count,),
                    device=self.env.device,
                    generator=generator,
                )
                random_repr = projected[modality][random_ids]
                random_mse = F.mse_loss(random_repr, target_repr, reduction="mean").item()
                random_cosine = F.cosine_similarity(random_repr, target_repr, dim=-1).mean().item()
                metric["random_mse"] = float(random_mse)
                metric["random_cosine"] = float(random_cosine)
                weighted_random_mse += random_mse * count
                weighted_random_cosine += random_cosine * count

            results[modality] = metric
            total_count += count
            weighted_mse += mse_value * count
            weighted_cosine += cosine_value * count

        if total_count == 0:
            results["_overall"] = {"count": 0, "mse": 0.0, "cosine": 0.0}
            if include_random_baseline:
                results["_overall"]["random_mse"] = 0.0
                results["_overall"]["random_cosine"] = 0.0
            return results

        results["_overall"] = {
            "count": total_count,
            "mse": float(weighted_mse / total_count),
            "cosine": float(weighted_cosine / total_count),
        }
        if include_random_baseline:
            results["_overall"]["random_mse"] = float(weighted_random_mse / total_count)
            results["_overall"]["random_cosine"] = float(weighted_random_cosine / total_count)
        return results

    def _build_completed_features(
        self,
        projected,
        masks,
        detach_imputed=True,
        item_ids=None,
    ):
        """Build completed features for ALL items (like ProMRL).
        Observed features keep grad_fn.
        Imputed features are detached by default to preserve the original
        ProMRL-style optimization path, but recommendation-time joint training
        can opt into gradient flow through the imputation branch.
        """
        if self.disable_imputation:
            return {modality: feat.clone() for modality, feat in projected.items()}

        if not detach_imputed:
            return self.impute_modalities(
                projected,
                masks,
                item_ids=item_ids,
            )

        N = projected[self.modalities[0]].size(0)
        device = projected[self.modalities[0]].device

        # Pre-compute imputed values for each modality (only filled for missing items)
        imputed = {
            modality: torch.zeros(N, self.promrl_dim, device=device)
            for modality in self.modalities
        }

        pattern_keys = torch.stack([masks[m] for m in self.modalities], dim=1)

        with torch.no_grad():
            for pattern in torch.unique(pattern_keys, dim=0):
                missing = [self.modalities[i] for i in range(len(self.modalities)) if not bool(pattern[i])]
                if not missing:
                    continue

                selector = (pattern_keys == pattern.unsqueeze(0)).all(dim=1)
                if selector.sum() == 0:
                    continue

                observed = [self.modalities[i] for i in range(len(self.modalities)) if bool(pattern[i])]
                obs_feats = {m: projected[m][selector].detach() for m in observed}
                W_dict = {m: self.W[m] for m in self.modalities}
                mu_dict = {m: self.mu[m] for m in self.modalities}
                log_sigma_dict = {m: self.log_sigma[m].squeeze() for m in self.modalities}

                if observed:
                    m, _ = update_posterior(
                        obs_feats, W_dict, mu_dict, log_sigma_dict, observed, self.d_beta,
                    )
                else:
                    m = torch.zeros(selector.sum(), self.d_beta, device=device)

                for modality in missing:
                    recon = F.normalize(
                        m @ self.W[modality].T + self.mu[modality].unsqueeze(0), dim=-1,
                    )
                    imputed[modality][selector] = recon

        # Blend: observed (with grad_fn) where mask=True, imputed (no grad) where mask=False
        completed = {}
        for modality in self.modalities:
            mask = masks[modality].unsqueeze(1)  # [N, 1]
            completed[modality] = torch.where(mask, projected[modality], imputed[modality])

        return completed

    def compute_promrl_losses(self, item_ids, stage=None):
        item_ids = torch.unique(item_ids)
        stage = self._canonical_stage(stage or self.env.args.train_stage)
        raw_features = self._current_raw_modal_features(full=False)
        raw_batch = {modality: raw_features[modality][item_ids] for modality in self.modalities}
        projected = self.project_features(item_ids=item_ids, raw_features=raw_features)
        full_masks = self._missing_masks(raw_features=raw_features)
        masks = {modality: full_masks[modality][item_ids] for modality in self.modalities}
        is_joint_stage = stage == "joint"

        if is_joint_stage:
            need_rec = float(getattr(self.env.args, "beta_rec", 0.0)) != 0.0
            need_contrastive = (
                float(getattr(self.env.args, "beta_intra", 0.0)) != 0.0
                or float(getattr(self.env.args, "beta_inter", 0.0)) != 0.0
            )
            need_itm = float(getattr(self.env.args, "beta_itm", 0.0)) != 0.0
            need_decode = (
                float(getattr(self.env.args, "beta_decode", 0.0)) != 0.0
            )
        else:
            need_rec = float(getattr(self.env.args, "alpha_rec", 0.0)) != 0.0
            need_contrastive = (
                float(getattr(self.env.args, "alpha_intra", 0.0)) != 0.0
                or float(getattr(self.env.args, "alpha_inter", 0.0)) != 0.0
            )
            need_itm = float(getattr(self.env.args, "alpha_itm", 0.0)) != 0.0
            need_decode = (
                float(getattr(self.env.args, "alpha_decode", 0.0)) != 0.0
            )

        # Step 2-3: Posterior inference + rec_loss for each observed-modality pattern.
        rec_loss = torch.zeros((), device=self.env.device)
        if not self.disable_imputation and need_rec:
            rec_loss = self._compute_observed_rec_loss(projected, masks, stage, item_ids=item_ids)

        zero = torch.zeros((), device=self.env.device)
        loss_intra = zero
        loss_inter = zero
        loss_itm_raw = zero
        loss_itm = zero
        loss_decode = zero
        loss_decode_kl = zero

        if stage != "imputer_param":
            # Step 4-5: Build completed features + contrastive loss (like ProMRL)
            need_completed = need_contrastive or need_itm or need_decode
            completed = self._build_completed_features(projected, masks, item_ids=item_ids) if need_completed else None
            if need_contrastive:
                loss_intra, loss_inter = self.compute_contrastive_loss(completed)
            if need_itm:
                loss_itm_raw, loss_itm = self.compute_itm_loss(item_ids, completed_feats=completed)
            if self.use_decode_head and need_decode:
                decode_completed = completed
                decode_selected = masks
                decode_targets = raw_batch

                if getattr(self.env.args, "decoder_loss_mode", "observed_projection") == "pseudo_missing":
                    pseudo_masks, pseudo_selected = self._sample_pseudo_missing_masks(
                        masks,
                        ratio=float(getattr(self.env.args, "decoder_pseudo_missing_ratio", 1.0)),
                    )
                    if pseudo_masks is None or pseudo_selected is None:
                        decode_completed = None
                    else:
                        # The completion module is normally frozen in this mode.  Detaching
                        # here also makes the decoder-only contract explicit and avoids
                        # retaining a graph through posterior inference.
                        decode_completed = self._build_completed_features(
                            projected,
                            pseudo_masks,
                            detach_imputed=True,
                            item_ids=item_ids,
                        )
                        decode_selected = pseudo_selected

                decode_losses = []
                if decode_completed is not None:
                    decoded_raw = self.decode_completed_to_raw(decode_completed)
                    for modality in self.modalities:
                        observed_mask = decode_selected[modality]
                        if need_decode and observed_mask.any():
                            decoded_obs = decoded_raw[modality][observed_mask]
                            normalized_target = decode_targets[modality][observed_mask]
                            selected_item_ids = item_ids[observed_mask]
                            raw_target = self._decoder_target(
                                modality,
                                normalized_target,
                                item_ids=selected_item_ids,
                            )
                            decode_losses.append(
                                self._decoder_reconstruction_loss(
                                    modality,
                                    decoded_obs,
                                    raw_target,
                                )
                            )
                    if decode_losses:
                        loss_decode = torch.stack(decode_losses).mean()

        total_contrastive = loss_intra + loss_inter
        self.latest_promrl_losses = {
            "loss_intra": loss_intra,
            "loss_inter": loss_inter,
            "contrastive": total_contrastive,
            "rec_loss": rec_loss,
            "loss_itm_raw": loss_itm_raw,
            "loss_itm": loss_itm,
            "loss_decode": loss_decode,
            "loss_decode_kl": loss_decode_kl,
        }
        return self.latest_promrl_losses

    def _run_modal_gcn(self):
        if self._gcn_cache is not None:
            return self._gcn_cache

        user_id_emb = self.user_emb.weight
        raw_features = self._current_raw_modal_features()
        modal_features = self.get_recommender_modal_features()
        reliability_gates = self._completion_reliability_gates(raw_features=raw_features)
        observed_masks = self._missing_masks(raw_features=raw_features)

        outputs = {
            "user_id": user_id_emb,
            "modal_inputs": modal_features,
            "reliability_gates": reliability_gates,
            "observed_masks": observed_masks,
            "ungated_item_outputs": {},
        }
        for modality in self.modalities:
            modal_user_emb, modal_item_emb = getattr(self, f"{modality}_gcn")(
                modal_features[modality], user_id_emb, skip_mlp=self._gcn_skip_mlp()
            )
            modal_item_emb = self._apply_item_graph_modal_residual(
                modal_item_emb, observed_masks[modality], modality=modality
            )
            outputs["ungated_item_outputs"][modality] = modal_item_emb
            if not self.use_rum_fusion:
                modal_item_emb = modal_item_emb * reliability_gates[modality]
            outputs[modality] = (modal_user_emb, modal_item_emb)

        self._gcn_cache = outputs
        return outputs

    def clear_gcn_cache(self):
        self._gcn_cache = None

    def forward(self, random=False):
        outputs = self._run_modal_gcn()
        user_id_emb = outputs["user_id"]
        user_outputs = {
            modality: outputs[modality][0]
            for modality in self.modalities
        }
        if self.use_rank_residual_completion_gate and not self.use_rum_fusion:
            reliability_gates = outputs["reliability_gates"]
            ungated_item_outputs = {
                modality: item_emb.detach()
                for modality, item_emb in outputs["ungated_item_outputs"].items()
            }
            item_outputs = {
                modality: item_emb * reliability_gates[modality]
                for modality, item_emb in ungated_item_outputs.items()
            }
        else:
            ungated_item_outputs = None
            item_outputs = {
                modality: outputs[modality][1].detach()
                for modality in self.modalities
            }

        user_emb = user_id_emb + sum(user_outputs.values()) / len(user_outputs)
        if self.use_rank_residual_completion_gate and not self.use_rum_fusion:
            ungated_item_source = self._fuse_item_sources(
                ungated_item_outputs,
                modal_features=outputs["modal_inputs"],
                observed_masks=outputs["observed_masks"],
            )
            gated_item_source = self._fuse_item_sources(
                item_outputs,
                modal_features=outputs["modal_inputs"],
                observed_masks=outputs["observed_masks"],
            )
            mix_alpha = self._completion_gate_mix_alpha()
            item_source = (1.0 - mix_alpha) * ungated_item_source + mix_alpha * gated_item_source
            self._record_completion_gate_mix_alpha(mix_alpha)
            self.latest_completion_gate_ungated_item_emb = self._apply_fusion(
                ungated_item_source,
                deterministic=True,
            ).detach()
            self.latest_completion_gate_gated_item_emb = self._apply_fusion(
                item_source,
                deterministic=True,
            )
        else:
            item_source = self._fuse_item_sources(
                item_outputs,
                modal_features=outputs["modal_inputs"],
                observed_masks=outputs["observed_masks"],
            )
            self.latest_completion_gate_ungated_item_emb = None
            self.latest_completion_gate_gated_item_emb = None
        item_emb = self.fusion_linear(item_source)

        user_emb = torch.nan_to_num(user_emb, nan=0.0, posinf=0.0, neginf=0.0)
        item_emb = torch.nan_to_num(item_emb, nan=0.0, posinf=0.0, neginf=0.0)
        self.final_user = user_emb
        self.final_item = item_emb

        return user_emb, item_emb

    def completion_gate_advantage_loss(self, batch_users, batch_pos_items, batch_neg_items):
        if not (
            self.use_rank_residual_completion_gate
            and not self.use_rum_fusion
            and self.latest_completion_gate_ungated_item_emb is not None
            and self.latest_completion_gate_gated_item_emb is not None
            and self.final_user is not None
        ):
            return torch.zeros((), device=self.env.device)

        local_users = torch.arange(batch_users.shape[0], device=batch_users.device)
        user_emb = self.final_user[batch_users].detach()
        ungated_loss = self._bpr_loss_from_embeddings(
            user_emb,
            self.latest_completion_gate_ungated_item_emb,
            local_users,
            batch_pos_items,
            batch_neg_items,
        ).detach()
        gated_loss = self._bpr_loss_from_embeddings(
            user_emb,
            self.latest_completion_gate_gated_item_emb,
            local_users,
            batch_pos_items,
            batch_neg_items,
        )
        margin = max(
            float(getattr(self.env.args, "completion_gate_advantage_margin", 0.0)),
            0.0,
        )
        return torch.relu(gated_loss - ungated_loss + margin)

    def _score_residual_embeddings(self, outputs, deterministic=False):
        user_id_emb = outputs["user_id"]
        user_outputs = {
            modality: outputs[modality][0]
            for modality in self.modalities
        }
        user_emb = user_id_emb + sum(user_outputs.values()) / len(user_outputs)

        base_item_outputs = {
            modality: outputs["ungated_item_outputs"][modality].detach()
            for modality in self.modalities
        }
        gated_item_outputs = {
            modality: base_item_outputs[modality] * outputs["reliability_gates"][modality]
            for modality in self.modalities
        }
        base_item_source = self._fuse_item_sources(
            base_item_outputs,
            modal_features=outputs["modal_inputs"],
            observed_masks=outputs["observed_masks"],
        )
        gated_item_source = self._fuse_item_sources(
            gated_item_outputs,
            modal_features=outputs["modal_inputs"],
            observed_masks=outputs["observed_masks"],
        )
        base_item_emb = self._apply_fusion(base_item_source, deterministic=deterministic)
        gated_item_emb = self._apply_fusion(gated_item_source, deterministic=deterministic)
        return user_emb, base_item_emb, gated_item_emb

    def _score_residual_alpha(self):
        return self.completion_gate_score_residual_alpha

    def score_residual_pair_scores(self, batch_users, batch_items, outputs=None):
        if outputs is None:
            outputs = self._run_modal_gcn()
        user_emb, base_item_emb, gated_item_emb = self._score_residual_embeddings(
            outputs,
            deterministic=True,
        )
        users = user_emb[batch_users]
        base_scores = torch.sum(users * base_item_emb[batch_items], dim=1)
        gated_scores = torch.sum(users * gated_item_emb[batch_items], dim=1)
        alpha = self._score_residual_alpha()
        self.latest_completion_gate_metrics["completion_gate_score_residual_alpha"] = alpha
        return base_scores + alpha * (gated_scores - base_scores)

    def score_residual_score_matrix(self, batch_users, batch_items, outputs=None):
        if outputs is None:
            outputs = self._run_modal_gcn()
        user_emb, base_item_emb, gated_item_emb = self._score_residual_embeddings(
            outputs,
            deterministic=True,
        )
        users = user_emb[batch_users]
        base_scores = users @ base_item_emb[batch_items].T
        gated_scores = users @ gated_item_emb[batch_items].T
        alpha = self._score_residual_alpha()
        return base_scores + alpha * (gated_scores - base_scores)

    def _score_residual_reg_loss(self, outputs, batch_users, batch_pos_items, batch_neg_items):
        user_emb, base_item_emb, _ = self._score_residual_embeddings(
            outputs,
            deterministic=True,
        )
        return self.calculate_reg_loss(
            batch_users,
            batch_pos_items,
            batch_neg_items,
            all_user_emb=user_emb,
            all_item_emb=base_item_emb,
        )

    def _rum_tau(self):
        return max(float(getattr(self.env.args, "rum_tau", 1.0)), 1e-6)

    def _rum_logit_terms(self, outputs, batch_users, batch_items, matrix=False, detach_items=False):
        reliability = outputs["reliability_gates"]
        observed = outputs["observed_masks"]
        pref = self.user_modality_pref(batch_users)
        scores = []
        logits = []
        rel_values = []
        obs_values = []
        user_id_emb = outputs["user_id"][batch_users]
        reliability_coeff = float(getattr(self.env.args, "rum_reliability_coeff", 1.0))
        match_coeff = float(getattr(self.env.args, "rum_match_coeff", 1.0))

        for modality_idx, modality in enumerate(self.modalities):
            modal_user_emb, modal_item_emb = outputs[modality]
            modal_users = user_id_emb + modal_user_emb[batch_users]
            modal_items = modal_item_emb[batch_items]
            if detach_items:
                modal_items = modal_items.detach()
            if matrix:
                modal_score = modal_users @ modal_items.T
                rel = reliability[modality][batch_items].squeeze(1).unsqueeze(0)
                obs = observed[modality][batch_items].float().unsqueeze(0)
                pref_term = pref[:, modality_idx].unsqueeze(1)
            else:
                modal_score = torch.sum(modal_users * modal_items, dim=1)
                rel = reliability[modality][batch_items].squeeze(1)
                obs = observed[modality][batch_items].float()
                pref_term = pref[:, modality_idx]

            logit = (
                pref_term
                + reliability_coeff * torch.log(rel.clamp_min(1e-6))
                + match_coeff * torch.tanh(modal_score)
                + self.rum_modality_bias[modality_idx]
                + self.rum_observed_bias[modality_idx] * obs
            )
            scores.append(modal_score)
            logits.append(logit)
            rel_values.append(rel.expand_as(modal_score))
            obs_values.append(obs.expand_as(modal_score))

        return (
            torch.stack(scores, dim=-1),
            torch.stack(logits, dim=-1),
            torch.stack(rel_values, dim=-1),
            torch.stack(obs_values, dim=-1),
        )

    def _update_rum_fusion_metrics(self, weights, reliability, observed):
        with torch.no_grad():
            weights = weights.detach()
            reliability = reliability.detach()
            observed = observed.detach().bool()
            metrics = {}
            entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
            metrics["rum_fusion_entropy"] = float(entropy.cpu())
            for modality_idx, modality in enumerate(self.modalities):
                modality_weights = weights[..., modality_idx]
                modality_reliability = reliability[..., modality_idx]
                modality_observed = observed[..., modality_idx]
                metrics[f"rum_weight_{modality}_mean"] = float(modality_weights.mean().cpu())
                metrics[f"rum_reliability_{modality}_mean"] = float(modality_reliability.mean().cpu())
                if modality_observed.any():
                    metrics[f"rum_weight_{modality}_observed_mean"] = float(
                        modality_weights[modality_observed].mean().cpu()
                    )
                if (~modality_observed).any():
                    metrics[f"rum_weight_{modality}_imputed_mean"] = float(
                        modality_weights[~modality_observed].mean().cpu()
                    )
            self.latest_rum_fusion_metrics = metrics

    def rum_pair_scores(self, batch_users, batch_items, outputs=None, allow_modal_grad=False, collect_metrics=True):
        if outputs is None:
            outputs = self._run_modal_gcn()
        scores, logits, reliability, observed = self._rum_logit_terms(
            outputs,
            batch_users,
            batch_items,
            matrix=False,
            detach_items=not allow_modal_grad,
        )
        weights = F.softmax(logits / self._rum_tau(), dim=-1)
        if collect_metrics:
            self._update_rum_fusion_metrics(weights, reliability, observed)
        return torch.sum(weights * scores, dim=-1)

    def rum_score_matrix(self, batch_users, batch_items, outputs=None):
        if outputs is None:
            outputs = self._run_modal_gcn()
        scores, logits, _, _ = self._rum_logit_terms(
            outputs,
            batch_users,
            batch_items,
            matrix=True,
        )
        weights = F.softmax(logits / self._rum_tau(), dim=-1)
        return torch.sum(weights * scores, dim=-1)

    def _rum_reg_loss(self, outputs, batch_users, batch_pos_items, batch_neg_items):
        user_terms = outputs["user_id"][batch_users].norm(2).pow(2)
        item_terms = []
        for modality in self.modalities:
            _, modal_item_emb = outputs[modality]
            item_terms.append(modal_item_emb[batch_pos_items].norm(2).pow(2))
            item_terms.append(modal_item_emb[batch_neg_items].norm(2).pow(2))
        pref_terms = self.user_modality_pref(batch_users).norm(2).pow(2)
        total = user_terms + pref_terms
        for term in item_terms:
            total = total + term
        return 0.5 * total.mean()

    def basic_recommendation_loss(
        self,
        batch_users,
        batch_pos_items,
        batch_neg_items,
        allow_modal_grad=False,
        return_embeddings=False,
    ):
        """Base recommender loss without the original I3 IRM/MI objectives."""
        if self.use_rum_fusion:
            outputs = self._run_modal_gcn()
            pos_scores = self.rum_pair_scores(
                batch_users,
                batch_pos_items,
                outputs=outputs,
                allow_modal_grad=allow_modal_grad,
                collect_metrics=True,
            )
            neg_scores = self.rum_pair_scores(
                batch_users,
                batch_neg_items,
                outputs=outputs,
                allow_modal_grad=allow_modal_grad,
                collect_metrics=True,
            )
            bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
            reg_loss = self._rum_reg_loss(outputs, batch_users, batch_pos_items, batch_neg_items)
            if return_embeddings:
                all_user_emb, all_item_emb = self.forward()
                return bpr_loss, reg_loss, all_user_emb, all_item_emb
            return bpr_loss, reg_loss

        if self.use_score_residual_completion_gate:
            outputs = self._run_modal_gcn()
            pos_scores = self.score_residual_pair_scores(
                batch_users,
                batch_pos_items,
                outputs=outputs,
            )
            neg_scores = self.score_residual_pair_scores(
                batch_users,
                batch_neg_items,
                outputs=outputs,
            )
            bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
            reg_loss = self._score_residual_reg_loss(
                outputs,
                batch_users,
                batch_pos_items,
                batch_neg_items,
            )
            if return_embeddings:
                all_user_emb, all_item_emb, _ = self._score_residual_embeddings(
                    outputs,
                    deterministic=True,
                )
                return bpr_loss, reg_loss, all_user_emb, all_item_emb
            return bpr_loss, reg_loss

        if allow_modal_grad:
            all_user_emb, all_item_emb = self.compute_recommendation_embeddings(
                raw_features=self._current_raw_modal_features(full=False),
                allow_modal_grad=True,
            )
        else:
            all_user_emb, all_item_emb = self.forward()

        user_emb = all_user_emb[batch_users]
        pos_items = all_item_emb[batch_pos_items]
        neg_items = all_item_emb[batch_neg_items]

        pos_scores = torch.sum(user_emb * pos_items, dim=1)
        neg_scores = torch.sum(user_emb * neg_items, dim=1)
        bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = self.calculate_reg_loss(
            batch_users,
            batch_pos_items,
            batch_neg_items,
            all_user_emb=all_user_emb,
            all_item_emb=all_item_emb,
        )
        if return_embeddings:
            return bpr_loss, reg_loss, all_user_emb, all_item_emb
        return bpr_loss, reg_loss

    def _bpr_loss_from_embeddings(self, all_user_emb, all_item_emb, batch_users, batch_pos_items, batch_neg_items):
        user_emb = all_user_emb[batch_users]
        pos_items = all_item_emb[batch_pos_items]
        neg_items = all_item_emb[batch_neg_items]

        pos_scores = torch.sum(user_emb * pos_items, dim=1)
        neg_scores = torch.sum(user_emb * neg_items, dim=1)
        return torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

    def get_env_emb(self, mix_ration, env):
        outputs = self._run_modal_gcn()
        user_id_emb = outputs["user_id"]
        item_parts = []
        user_parts = []
        for modality_idx, modality in enumerate(self.modalities):
            modal_user_emb, modal_item_emb = outputs[modality]
            item_parts.append(mix_ration[env][modality_idx] * modal_item_emb)
            user_parts.append(user_id_emb + modal_user_emb)
        item_emb = torch.cat(item_parts, dim=1)
        user_emb = torch.cat(user_parts, dim=1)

        assert torch.isnan(user_emb).sum() == 0
        assert torch.isnan(item_emb).sum() == 0
        return user_emb, item_emb

    def modality_bpr_loss(self, batch_users, batch_pos_items, batch_neg_items):
        mix_ration = [
            [1 if idx == env else 0 for idx in range(len(self.modalities))]
            for env in range(len(self.modalities))
        ]

        losses = []
        for env in range(len(mix_ration)):
            env_user_emb, env_item_emb = self.get_env_emb(mix_ration, env)
            losses.append(
                self._bpr_loss_from_embeddings(
                    env_user_emb,
                    env_item_emb,
                    batch_users,
                    batch_pos_items,
                    batch_neg_items,
                )
            )
        if not losses:
            return torch.zeros((), device=self.env.device)
        return torch.stack(losses).sum()

    def calculate_reg_loss(self, batch_users, batch_pos_items, batch_neg_items, all_user_emb=None, all_item_emb=None):
        if all_user_emb is None or all_item_emb is None:
            all_user_emb, all_item_emb = self.forward()
        reg_embedding_loss = (
            0.5
            * (
                all_user_emb[batch_users].norm(2).pow(2)
                + all_item_emb[batch_pos_items].norm(2).pow(2)
                + all_item_emb[batch_neg_items].norm(2).pow(2)
            )
            / float(len(batch_users))
        )

        return reg_embedding_loss
