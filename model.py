import copy
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from promrl_core.general_module import Contra_head, Match_head
from promrl_core.utils.impute import compute_nll_loss
from promrl_core.utils.eigen import eigenvalue_computation_pmcl
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
        self.item_graph_dynamic_norm_type = "rw"
        self.ItemItemRawGraph = None
        self.free_emb_dimension = self.env.args.free_emb_dimension
        self.has_audio_modality = dataset.audio_feat is not None
        self.modalities = ["v", "t"]
        if self.has_audio_modality:
            self.modalities.append("a")

        self.train_missing_modality_items = dataset.train_missing_modality_items
        self.val_missing_modality_items = getattr(dataset, "val_missing_modality_items", {"items": [], "indicator": []})
        self.eval_val_missing_modality_items = getattr(
            dataset,
            "eval_val_missing_modality_items",
            {"items": [], "indicator": []},
        )
        self.test_missing_modality_items = dataset.test_missing_modality_items

        self.audio_feat = None
        self.ori_audio_feat = None

        self.ori_image_feat = torch.tensor(dataset.image_feat, dtype=torch.float32).to(self.env.device)
        self.ori_image_feat = F.normalize(self.ori_image_feat)

        self.ori_text_feat = torch.tensor(dataset.text_feat, dtype=torch.float32).to(self.env.device)
        self.ori_text_feat = F.normalize(self.ori_text_feat)

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
            self.ori_audio_feat = torch.tensor(dataset.audio_feat, dtype=torch.float32).to(self.env.device)
            self.ori_audio_feat = F.normalize(self.ori_audio_feat)
            self.eval_ori_audio_feat = torch.tensor(
                getattr(dataset, "eval_audio_feat", dataset.audio_feat),
                dtype=torch.float32,
            ).to(self.env.device)
            self.eval_ori_audio_feat = F.normalize(self.eval_ori_audio_feat)

        self.uses_split_modal_feature_override = bool(
            getattr(dataset, "uses_split_modal_feature_override", False)
        )
        self.modal_feature_mask_source = getattr(self.env.args, "modal_feature_mask_source", "nonzero")
        self.train_external_modal_observed_masks = self._tensorize_external_modal_masks(
            getattr(dataset, "train_external_modal_observed_masks", None)
        )
        self.eval_external_modal_observed_masks = self._tensorize_external_modal_masks(
            getattr(dataset, "eval_external_modal_observed_masks", None)
        )

        self.contra_dim = self.env.args.contra_dim
        self.d_beta = self.env.args.d_beta
        self.ema_eta = self.env.args.ema_eta
        self.itm_temp = self.env.args.itm_temp
        self.lambda_itm = self.env.args.lambda_itm
        self.disable_imputation = bool(self.env.args.disable_imputation)
        self.feature_bridge_mode = self.env.args.feature_bridge_mode
        if self.feature_bridge_mode not in ("raw_decoder", "shared_identity"):
            raise ValueError("feature_bridge_mode must be 'raw_decoder' or 'shared_identity'")
        self.use_decode_head = True
        self.gcn_frontend_mode = self.env.args.gcn_frontend_mode
        self.promrl_projection_mode = getattr(self.env.args, "promrl_projection_mode", "learned")
        self.promrl_dim = self.contra_dim
        self.beta_completion_mode = getattr(self.env.args, "beta_completion_mode", "linear")
        if self.beta_completion_mode not in ("linear", "decoder"):
            raise ValueError(f"Unsupported beta_completion_mode: {self.beta_completion_mode}")
        self.use_beta_completion_decoder = self.beta_completion_mode != "linear"
        self.beta_completion_rec_weight = max(
            float(getattr(self.env.args, "beta_completion_rec_weight", 1.0)),
            0.0,
        )
        self.beta_completion_rec_loss = getattr(self.env.args, "beta_completion_rec_loss", "mse_cosine")
        self.beta_completion_detach_beta = bool(getattr(self.env.args, "beta_completion_detach_beta", 1))
        self.smore_beta_prior_lambda = max(float(getattr(self.env.args, "smore_beta_prior_lambda", 0.0)), 0.0)
        self.smore_beta_prior_rho = max(float(getattr(self.env.args, "smore_beta_prior_rho", 1.0)), 0.0)
        self.smore_beta_prior_var_min = max(float(getattr(self.env.args, "smore_beta_prior_var_min", 0.1)), 1e-8)
        self.smore_beta_prior_var_max = max(
            float(getattr(self.env.args, "smore_beta_prior_var_max", 2.0)),
            self.smore_beta_prior_var_min + 1e-8,
        )
        self.uses_split_smore_beta_prior_features = False
        self.smore_beta_prior_features = self._load_smore_beta_prior_features()
        self.smore_beta_prior_generators = nn.ModuleDict()
        if self.smore_beta_prior_lambda > 0.0:
            if not self.smore_beta_prior_features:
                raise ValueError(
                    "smore_beta_prior_lambda > 0 requires --smore_beta_prior_dir "
                    "or explicit SMORE prior train/eval directories"
                )
            hidden_dim = int(getattr(self.env.args, "smore_beta_prior_hidden_dim", 128))
            dropout = float(getattr(self.env.args, "smore_beta_prior_dropout", 0.0))
            train_features = self.smore_beta_prior_features["train"]
            for modality in ("v", "t"):
                if modality in self.modalities and modality in train_features:
                    self.smore_beta_prior_generators[modality] = _build_mlp(
                        train_features[modality].size(1),
                        hidden_dim,
                        self.d_beta * 2,
                        dropout=dropout,
                    )
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
        self.use_missing_aware_reliability_gate = (
            self.completion_gate_mode == "missing_reliability"
        )
        self.use_learned_completion_gate_mix = (
            bool(getattr(self.env.args, "completion_gate_learn_mix", 0))
            and self.use_rank_residual_completion_gate
        )
        self.use_learned_completion_gate = self.completion_gate_mode in (
            "reliability",
            "missing_reliability",
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
        self.completion_gate_stats_norm = bool(getattr(self.env.args, "completion_gate_stats_norm", 1))
        self.completion_gate_apply_observed = bool(
            getattr(self.env.args, "completion_gate_apply_observed", 1)
        )
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
        self.use_completed_item_graph = self.item_graph_kind in (
            "fused_completed",
            "modality_completed",
            "modality_completed_confidence",
            "modality_completed_dynamic_confidence",
            "fused_completed_confidence",
            "fused_completed_dynamic_confidence",
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
        shared_edge_conf_init_values = [
            max(float(getattr(self.env.args, "item_graph_rr_confidence_init", 1.0)), 1e-6),
            max(float(getattr(self.env.args, "item_graph_ri_confidence_init", 1.0)), 1e-6),
            max(float(getattr(self.env.args, "item_graph_ii_confidence_init", 1.0)), 1e-6),
        ]

        def build_edge_conf_init(modality=None):
            values = list(shared_edge_conf_init_values)
            if modality in ("t", "v", "a"):
                prefix = {"t": "text", "v": "image", "a": "audio"}[modality]
                for idx, edge_type in enumerate(("rr", "ri", "ii")):
                    override = getattr(
                        self.env.args,
                        f"item_graph_{prefix}_{edge_type}_confidence_init",
                        None,
                    )
                    if override is not None:
                        values[idx] = max(float(override), 1e-6)
            return torch.tensor(values, dtype=torch.float32).clamp(
                min=self.item_graph_confidence_min,
                max=self.item_graph_confidence_max,
            )

        def confidence_to_param(conf_init):
            if self.item_graph_confidence_transform == "sigmoid":
                span = max(self.item_graph_confidence_max - self.item_graph_confidence_min, 1e-6)
                normalized_conf = ((conf_init - self.item_graph_confidence_min) / span).clamp(1e-4, 1.0 - 1e-4)
                return torch.logit(normalized_conf)
            return torch.log(conf_init)

        edge_conf_param_init = confidence_to_param(build_edge_conf_init())
        self.use_item_graph_modality_specific_confidence = (
            bool(int(getattr(self.env.args, "item_graph_modality_specific_confidence", 0)))
            and self.item_graph_kind in ("modality_completed_confidence", "modality_completed_dynamic_confidence")
        )
        if self.use_item_graph_modality_specific_confidence:
            self.item_graph_edge_confidence_params = nn.ParameterDict(
                {
                    "v": nn.Parameter(confidence_to_param(build_edge_conf_init("v"))),
                    "t": nn.Parameter(confidence_to_param(build_edge_conf_init("t"))),
                    "a": nn.Parameter(confidence_to_param(build_edge_conf_init("a"))),
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
        self.use_rum_fusion = self.fusion_mode == "rum"
        self.use_missing_weighted_fusion = self.fusion_mode == "missing_weighted_mean"
        self.use_global_weighted_fusion = self.fusion_mode == "global_weighted_mean"
        self.use_reliability_weighted_fusion = self.fusion_mode == "reliability_weighted_mean"
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

        projection_bias = bool(getattr(self.env.args, "promrl_projection_bias", 1))
        self.contra_head_v = Contra_head(self.ori_image_feat.size(1), self.contra_dim, bias=projection_bias)
        self.contra_head_t = Contra_head(self.ori_text_feat.size(1), self.contra_dim, bias=projection_bias)
        if "a" in self.modalities:
            self.contra_head_a = Contra_head(self.ori_audio_feat.size(1), self.contra_dim, bias=projection_bias)
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

        self.decoder_v = self._build_modal_decoder(self.ori_image_feat.size(1))
        self.decoder_t = self._build_modal_decoder(self.ori_text_feat.size(1))
        if "a" in self.modalities:
            self.decoder_a = self._build_modal_decoder(self.ori_audio_feat.size(1))
        self.beta_completion_decoders = nn.ModuleDict()
        if self.use_beta_completion_decoder:
            for modality in self.modalities:
                self.beta_completion_decoders[modality] = self._build_beta_completion_decoder()

        v_input_dim = self.ori_image_feat.size(1)
        t_input_dim = self.ori_text_feat.size(1)
        a_input_dim = self.ori_audio_feat.size(1) if "a" in self.modalities else None

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
            gate_extra_dim = 6 if self.use_missing_aware_reliability_gate else 0
            gate_input_dim = self.promrl_dim + item_context_dim + len(self.modalities) + gate_extra_dim
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
        self.latest_completion_gate_supervision_loss = torch.zeros((), device=self.env.device)
        self.latest_completion_gate_counterfactual_loss = torch.zeros((), device=self.env.device)
        self.latest_reliability_gates = None
        self.latest_completion_gate_masks = None
        self.latest_completion_gate_stats = None
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
        canonical_stage = train_stage or self.env.args.train_stage
        if canonical_stage == "imputer":
            canonical_stage = "imputer_backprop"
        return canonical_stage

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

    def _uses_dynamic_stage1_masking(self, train_stage=None):
        canonical_stage = self._canonical_stage(train_stage)
        return (
            getattr(self.env.args, "stage1_masking_policy", "fixed") == "dynamic"
            and canonical_stage in ("imputer_init", "imputer_align", "imputer_promrl_main")
        )

    def _uses_stage1_holdout_metrics(self, train_stage=None):
        canonical_stage = self._canonical_stage(train_stage)
        return canonical_stage in (
            "imputer_backprop",
            "imputer_init",
            "imputer_align",
            "imputer_promrl_main",
            "imputer_adapter",
        )

    def _uses_smore_beta_prior_for_stage(self, train_stage=None):
        generators = getattr(self, "smore_beta_prior_generators", None)
        if (
            self.smore_beta_prior_lambda <= 0.0
            or generators is None
            or len(generators) == 0
        ):
            return False
        canonical_stage = self._canonical_stage(train_stage)
        if canonical_stage == "imputer_param":
            return False
        scope = getattr(self.env.args, "smore_beta_prior_scope", "stage12_recommender")
        if scope == "stage12":
            return canonical_stage == "imputer_backprop"
        if scope == "stage12_recommender":
            return canonical_stage in ("imputer_backprop", "recommender")
        if scope == "all_nonparam":
            return canonical_stage in (
                "imputer_backprop",
                "imputer_init",
                "imputer_align",
                "imputer_promrl_main",
                "imputer_adapter",
                "recommender",
                "joint",
            )
        return False

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

    def _decoder_hidden_dim(self, raw_dim):
        return min(1024, max(256, raw_dim // 2))

    def _build_modal_decoder(self, raw_dim):
        hidden_dim = self._decoder_hidden_dim(raw_dim)
        return nn.Sequential(
            nn.Linear(self.promrl_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, raw_dim),
        )

    def _build_beta_completion_decoder(self):
        hidden_dim = int(getattr(self.env.args, "beta_completion_decoder_hidden_dim", 128))
        dropout = float(getattr(self.env.args, "beta_completion_decoder_dropout", 0.0))
        layers = [
            nn.Linear(self.d_beta, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ])
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.extend([
            nn.Linear(hidden_dim, self.promrl_dim),
        ])
        return nn.Sequential(*layers)

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

    def _resolve_smore_beta_prior_dir(self, explicit_dir, base_dir, phase_name):
        if explicit_dir:
            return os.path.expanduser(explicit_dir)
        phase_dir = os.path.join(base_dir, phase_name)
        if os.path.isdir(phase_dir):
            return phase_dir
        return base_dir

    def _load_smore_beta_prior_pair(self, feature_dir, label):
        image_name = getattr(self.env.args, "smore_beta_prior_image_file", "image_item_embeds.npy")
        text_name = getattr(self.env.args, "smore_beta_prior_text_file", "text_item_embeds.npy")
        paths = {
            "v": os.path.join(feature_dir, image_name),
            "t": os.path.join(feature_dir, text_name),
        }
        features = {}
        for modality, path in paths.items():
            if modality not in self.modalities:
                continue
            if not os.path.exists(path):
                raise FileNotFoundError(f"{label} SMORE beta prior feature not found: {path}")
            array = np.load(path).astype(np.float32, copy=False)
            if array.shape[0] != self.m_item:
                raise ValueError(
                    f"{label} SMORE beta prior feature item count mismatch: "
                    f"feature has {array.shape[0]} rows, dataset has {self.m_item} items"
                )
            tensor = torch.tensor(array, dtype=torch.float32, device=self.env.device)
            if bool(getattr(self.env.args, "smore_beta_prior_normalize", 0)):
                tensor = F.normalize(tensor, dim=-1)
            features[modality] = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return features

    def _load_smore_beta_prior_features(self):
        base_dir = getattr(self.env.args, "smore_beta_prior_dir", "") or ""
        train_dir = getattr(self.env.args, "smore_beta_prior_train_dir", "") or ""
        eval_dir = getattr(self.env.args, "smore_beta_prior_eval_dir", "") or ""
        if not base_dir and not train_dir and not eval_dir:
            return {}

        if base_dir:
            base_dir = os.path.expanduser(base_dir)
            if not os.path.isdir(base_dir):
                raise FileNotFoundError(f"smore_beta_prior_dir not found: {base_dir}")
        elif not (train_dir and eval_dir):
            raise ValueError(
                "Set --smore_beta_prior_dir or both --smore_beta_prior_train_dir "
                "and --smore_beta_prior_eval_dir"
            )

        train_feature_dir = self._resolve_smore_beta_prior_dir(
            train_dir,
            base_dir,
            "phase_train",
        )
        eval_feature_dir = self._resolve_smore_beta_prior_dir(
            eval_dir,
            base_dir,
            "phase_eval",
        )
        train_features = self._load_smore_beta_prior_pair(train_feature_dir, "train")
        eval_features = self._load_smore_beta_prior_pair(eval_feature_dir, "eval")
        self.uses_split_smore_beta_prior_features = (
            os.path.abspath(train_feature_dir) != os.path.abspath(eval_feature_dir)
        )
        print(
            "loaded SMORE beta prior features: "
            f"train_dir={train_feature_dir}, eval_dir={eval_feature_dir}, "
            f"modalities={sorted(train_features.keys())}"
        )
        return {
            "train": train_features,
            "eval": eval_features,
        }

    def _projection_modules(self):
        modules = [
            self.contra_head_v,
            self.contra_head_t,
        ]
        if "a" in self.modalities:
            modules.append(self.contra_head_a)
        return modules

    def _imputer_modules(self):
        modules = self._projection_modules()
        modules.extend([self.itm_cross_attn, self.itm_head])
        if hasattr(self, "smore_beta_prior_generators") and len(self.smore_beta_prior_generators) > 0:
            modules.append(self.smore_beta_prior_generators)
        return modules

    def _decoder_modules(self):
        modules = [self.decoder_v, self.decoder_t]
        if "a" in self.modalities:
            modules.append(self.decoder_a)
        if hasattr(self, "beta_completion_decoders") and len(self.beta_completion_decoders) > 0:
            modules.append(self.beta_completion_decoders)
        return modules

    def _recommender_modules(self):
        modules = [self.user_emb, self.item_emb, self.fusion_linear]
        if self.use_rum_fusion:
            modules.extend([self.user_modality_pref, self.rum_biases])
        if self.use_global_weighted_fusion:
            modules.append(self.global_fusion_params)
        modules.extend([self.v_gcn, self.t_gcn])
        if "a" in self.modalities:
            modules.append(self.a_gcn)
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

        auto_freeze_imputer = canonical_stage in ("recommender", "imputer_init", "imputer_adapter")
        auto_freeze_recommender = canonical_stage in (
            "imputer_param",
            "imputer_backprop",
            "imputer_init",
            "imputer_align",
            "imputer_promrl_main",
            "imputer_adapter",
        )
        auto_freeze_decoder = canonical_stage in (
            "imputer_param",
            "imputer_init",
            "imputer_align",
            "imputer_promrl_main",
        )

        freeze_imputer = auto_freeze_imputer if freeze_imputer < 0 else bool(freeze_imputer)
        freeze_recommender = auto_freeze_recommender if freeze_recommender < 0 else bool(freeze_recommender)
        freeze_decoder = auto_freeze_decoder if freeze_decoder < 0 else bool(freeze_decoder)

        imputer_module_trainable = not freeze_imputer and canonical_stage != "imputer_param"
        if canonical_stage in ("imputer_init", "imputer_adapter"):
            imputer_module_trainable = False
        self._set_modules_trainable(self._imputer_modules(), imputer_module_trainable)
        self._set_modules_trainable(self._recommender_modules(), not freeze_recommender)
        decoder_trainable = (
            self.use_decode_head
            and not bool(freeze_decoder)
            and canonical_stage in ("imputer_backprop", "imputer_adapter", "recommender", "joint")
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
                "imputer_backprop",
                "imputer_init",
                "imputer_align",
                "imputer_promrl_main",
                "joint",
            )
            and not self._uses_smore_beta_prior_for_stage(canonical_stage)
        )
        self.clear_gcn_cache()

    def get_imputer_parameters(self):
        params = []
        for module in self._imputer_modules():
            params.extend([param for param in module.parameters() if param.requires_grad])
        return params

    def get_decoder_parameters(self):
        params = []
        for module in self._decoder_modules():
            params.extend([param for param in module.parameters() if param.requires_grad])
        return params

    def get_recommender_parameters(self):
        params = []
        for module in self._recommender_modules():
            params.extend([param for param in module.parameters() if param.requires_grad])
        return params

    def get_completion_gate_parameters(self):
        params = []
        if self.use_learned_completion_gate:
            params.extend([
                param for param in self.completion_gates.parameters()
                if param.requires_grad
            ])
        if self.use_global_rank_residual_completion_gate:
            params.extend([
                param for param in self.completion_gate_global_logits.parameters()
                if param.requires_grad
            ])
        if self.use_learned_completion_gate_mix:
            params.extend([
                param for param in self.completion_gate_mix_params.parameters()
                if param.requires_grad
            ])
        return params

    def get_item_graph_confidence_parameters(self):
        if not self.use_item_graph_edge_confidence:
            return []
        return [
            param
            for param in self.item_graph_edge_confidence_params.parameters()
            if param.requires_grad
        ]

    def load_full_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.load_state_dict(state_dict, strict=False)
        return len(state_dict)

    def load_imputer_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        aliased_state_dict = dict(state_dict)
        for modality in self.modalities:
            alias_pairs = (
                (
                    f"comp_proj_{modality}.weight",
                    f"contra_head_{modality}.linear.weight",
                ),
                (
                    f"comp_proj_{modality}.bias",
                    f"contra_head_{modality}.linear.bias",
                ),
            )
            for source_key, target_key in alias_pairs:
                if source_key in state_dict and target_key not in aliased_state_dict:
                    aliased_state_dict[target_key] = state_dict[source_key]
        prefixes = [
            "itm_cross_attn",
            "itm_head",
            "W",
            "mu",
            "log_sigma",
            "smore_beta_prior_generators",
        ]
        prefixes.extend(["contra_head_v", "contra_head_t", "decoder_v", "decoder_t"])
        if "a" in self.modalities:
            prefixes.extend(["contra_head_a", "decoder_a"])
        if hasattr(self, "beta_completion_decoders") and len(self.beta_completion_decoders) > 0:
            prefixes.append("beta_completion_decoders")

        current_state = self.state_dict()
        matched_state = {
            key: value
            for key, value in aliased_state_dict.items()
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
        ) = self._build_missing_feature_view(self.train_missing_modality_items)
        (
            self.miss_val_image_feature,
            self.miss_val_text_feature,
            self.miss_val_audio_feature,
        ) = self._build_missing_feature_view(self.val_missing_modality_items)
        (
            self.miss_eval_val_image_feature,
            self.miss_eval_val_text_feature,
            self.miss_eval_val_audio_feature,
        ) = self._build_missing_feature_view(
            self.eval_val_missing_modality_items,
            image_base=self.eval_ori_image_feat,
            text_base=self.eval_ori_text_feat,
            audio_base=self.eval_ori_audio_feat if "a" in self.modalities else None,
        )
        (
            self.miss_test_image_feature,
            self.miss_test_text_feature,
            self.miss_test_audio_feature,
        ) = self._build_missing_feature_view(
            self.test_missing_modality_items,
            image_base=self.eval_ori_image_feat,
            text_base=self.eval_ori_text_feat,
            audio_base=self.eval_ori_audio_feat if "a" in self.modalities else None,
        )

    def refresh_dynamic_stage1_missing_views(self):
        self._dynamic_stage1_refresh_counter += 1
        dataset_seed = int(getattr(self.env.args, "dataset_seed", 0))
        dynamic_seed = dataset_seed + self._dynamic_stage1_refresh_counter
        self.dataset.refresh_stage1_dynamic_train_missing_metadata(seed=dynamic_seed)
        self.init_missing_modality_set()

    def _build_missing_feature_view(self, missing_metadata, image_base=None, text_base=None, audio_base=None):
        image_base = self.ori_image_feat if image_base is None else image_base
        text_base = self.ori_text_feat if text_base is None else text_base
        audio_base = self.ori_audio_feat if audio_base is None and "a" in self.modalities else audio_base
        miss_image_feature = copy.deepcopy(image_base)
        miss_text_feature = copy.deepcopy(text_base)
        miss_audio_feature = copy.deepcopy(audio_base) if "a" in self.modalities else None

        selected_missing_items = np.array(missing_metadata["items"], dtype=np.int64)
        selected_missing_modality_indicator = np.array(missing_metadata["indicator"], dtype=np.int64)
        if selected_missing_items.size == 0:
            return miss_image_feature, miss_text_feature, miss_audio_feature

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

        return miss_image_feature, miss_text_feature, miss_audio_feature

    def set_missing_modality_via_env(self, eval_split=None):
        mode = self.env.args.exp_mode
        canonical_stage = self._canonical_stage()

        use_missing_train = mode in {"mm", "mf"}
        use_missing_test = mode in {"mm", "fm"}

        if self.training:
            if self._uses_dynamic_stage1_masking(canonical_stage):
                self.refresh_dynamic_stage1_missing_views()
                self.image_feat = self.miss_train_image_feature
                self.text_feat = self.miss_train_text_feature
                if "a" in self.modalities:
                    self.audio_feat = self.miss_train_audio_feature
                print("set dynamic missing modality successfully for train step")
                return
            if use_missing_train:
                self.image_feat = self.miss_train_image_feature
                self.text_feat = self.miss_train_text_feature
                if "a" in self.modalities:
                    self.audio_feat = self.miss_train_audio_feature
                print("set missing modality successfully for train setp")
            else:
                self.image_feat = self.ori_image_feat
                self.text_feat = self.ori_text_feat
                if "a" in self.modalities:
                    self.audio_feat = self.ori_audio_feat
                print("set complete modality successfully for train step")
        else:
            if use_missing_test:
                if eval_split == "val":
                    self.image_feat = self.miss_eval_val_image_feature
                    self.text_feat = self.miss_eval_val_text_feature
                    if "a" in self.modalities:
                        self.audio_feat = self.miss_eval_val_audio_feature
                    print("set missing modality successfully for val setp")
                    return

                self.image_feat = self.miss_test_image_feature
                self.text_feat = self.miss_test_text_feature
                if "a" in self.modalities:
                    self.audio_feat = self.miss_test_audio_feature
                print("set missing modality successfully for test setp")
            else:
                self.image_feat = self.eval_ori_image_feat
                self.text_feat = self.eval_ori_text_feat
                if "a" in self.modalities:
                    self.audio_feat = self.eval_ori_audio_feat
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
            return features

        if split == "train":
            features = {"v": self.miss_train_image_feature, "t": self.miss_train_text_feature}
            if "a" in self.modalities:
                features["a"] = self.miss_train_audio_feature
            return features
        if split == "val":
            features = {"v": self.miss_val_image_feature, "t": self.miss_val_text_feature}
            if "a" in self.modalities:
                features["a"] = self.miss_val_audio_feature
            return features
        if split == "test":
            features = {"v": self.miss_test_image_feature, "t": self.miss_test_text_feature}
            if "a" in self.modalities:
                features["a"] = self.miss_test_audio_feature
            return features
        raise ValueError(f"Unsupported split: {split}")

    def get_missing_item_metadata(self, split="test"):
        if split == "train":
            metadata = self.train_missing_modality_items
        elif split == "val":
            metadata = self.val_missing_modality_items
        elif split == "test":
            metadata = self.test_missing_modality_items
        else:
            raise ValueError(f"Unsupported split: {split}")

        items = torch.as_tensor(metadata["items"], dtype=torch.long, device=self.env.device)
        indicators = torch.as_tensor(metadata["indicator"], dtype=torch.long, device=self.env.device)
        return items, indicators

    def _item_graph_mask_scope(self):
        scope = getattr(self.env.args, "item_graph_mask_scope", "train")
        if scope not in {"train", "train_val", "all"}:
            raise ValueError(f"Unsupported item_graph_mask_scope: {scope}")
        return scope

    def _combined_missing_raw_modal_features(self, scope=None):
        scope = self._item_graph_mask_scope() if scope is None else scope
        features = {
            "v": self.ori_image_feat.clone(),
            "t": self.ori_text_feat.clone(),
        }
        if "a" in self.modalities:
            features["a"] = self.ori_audio_feat.clone()

        metadata_list = [self.train_missing_modality_items]
        if scope in {"train_val", "all"}:
            metadata_list.append(self.val_missing_modality_items)
        if scope == "all":
            metadata_list.extend([
                self.eval_val_missing_modality_items,
                self.test_missing_modality_items,
            ])
        for metadata in metadata_list:
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

    def _resolve_external_item_graph_feature_dir(self):
        explicit_dir = getattr(self.env.args, "item_graph_feature_dir", "") or ""
        if explicit_dir:
            return os.path.expanduser(explicit_dir)

        override_dir = getattr(self.env.args, "modal_feature_override_dir", "") or ""
        if not override_dir:
            raise ValueError(
                "item_graph_feature_source=external_completed requires --item_graph_feature_dir "
                "or --modal_feature_override_dir"
            )

        base_dir = os.path.expanduser(override_dir)
        scope = self._item_graph_mask_scope()
        phase_train_dir = os.path.join(base_dir, "phase_train")
        phase_graph_dir = os.path.join(base_dir, "phase_graph")
        explicit_train_dir = getattr(self.env.args, "modal_feature_train_dir", "") or ""
        if explicit_train_dir:
            return os.path.expanduser(explicit_train_dir)

        if scope == "all" and os.path.isdir(phase_graph_dir):
            return phase_graph_dir
        if os.path.isdir(phase_train_dir):
            return phase_train_dir
        if scope != "all" and os.path.isdir(phase_graph_dir):
            raise ValueError(
                "Refusing to auto-use phase_graph for strict item-graph construction because it may include "
                "validation/test missing masks. Provide --item_graph_feature_dir with a train-scope feature "
                "directory, or set --item_graph_mask_scope all to reproduce the legacy behavior."
            )

        return base_dir

    def _load_external_item_graph_features(self):
        feature_dir = self._resolve_external_item_graph_feature_dir()
        if not os.path.isdir(feature_dir):
            raise FileNotFoundError(f"external item graph feature dir not found: {feature_dir}")

        image_name = (
            getattr(self.env.args, "item_graph_feature_image_file", "")
            or getattr(self.env.args, "modal_feature_image_file", "agg_image_items.npy")
        )
        text_name = (
            getattr(self.env.args, "item_graph_feature_text_file", "")
            or getattr(self.env.args, "modal_feature_text_file", "agg_text_items.npy")
        )
        specs = {
            "v": (
                os.path.join(feature_dir, image_name),
                getattr(self.env.args, "item_graph_feature_image_mask_file", "image_observed_mask.npy"),
            ),
            "t": (
                os.path.join(feature_dir, text_name),
                getattr(self.env.args, "item_graph_feature_text_mask_file", "text_observed_mask.npy"),
            ),
        }
        raw_features = {}
        masks = {}
        for modality, (feature_path, mask_name) in specs.items():
            if not os.path.exists(feature_path):
                raise FileNotFoundError(f"external item graph feature file not found: {feature_path}")
            feature = np.load(feature_path).astype(np.float32, copy=False)
            if feature.shape[0] != self.m_item:
                raise ValueError(
                    f"external item graph feature item count mismatch for {modality}: "
                    f"feature has {feature.shape[0]} rows, model has {self.m_item} items"
                )
            feature_tensor = torch.as_tensor(feature, dtype=torch.float32, device=self.env.device)
            raw_features[modality] = torch.nan_to_num(
                F.normalize(feature_tensor, dim=-1),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            mask_path = os.path.join(feature_dir, mask_name)
            if os.path.exists(mask_path):
                mask = np.load(mask_path).astype(bool)
                if mask.shape[0] != self.m_item:
                    raise ValueError(
                        f"external item graph mask item count mismatch for {modality}: "
                        f"mask has {mask.shape[0]} rows, model has {self.m_item} items"
                    )
                masks[modality] = torch.as_tensor(mask, dtype=torch.bool, device=self.env.device)
            else:
                masks[modality] = raw_features[modality].abs().sum(dim=1) > 0

        if "a" in self.modalities:
            raise ValueError("external_completed item graph feature source currently supports image/text datasets only")

        print(
            "loaded external completed item graph features: "
            f"dir={feature_dir}, image={image_name}, text={text_name}, "
            f"v_observed={int(masks['v'].sum().item())}, t_observed={int(masks['t'].sum().item())}"
        )
        return raw_features, masks, feature_dir

    def _build_item_graph_feature_inputs(self):
        source = getattr(self.env.args, "item_graph_feature_source", "internal_completion")
        graph_feature_space = getattr(self.env.args, "item_graph_feature_space", "shared")

        if source == "external_completed":
            raw_features, masks, feature_dir = self._load_external_item_graph_features()
            if graph_feature_space == "raw_decoder":
                graph_features = raw_features
            elif graph_feature_space == "shared":
                graph_features = self.project_features(raw_features=raw_features)
            else:
                raise ValueError(f"Unsupported item graph feature space: {graph_feature_space}")
            return graph_features, masks, source, feature_dir

        if source != "internal_completion":
            raise ValueError(f"Unsupported item graph feature source: {source}")

        mask_scope = self._item_graph_mask_scope()
        raw_features = self._combined_missing_raw_modal_features(scope=mask_scope)
        projected = self.project_features(raw_features=raw_features)
        masks = self._missing_masks(raw_features=raw_features)
        graph_features = self._build_completed_features(
            projected,
            masks,
            detach_imputed=True,
            stage=self.env.args.train_stage,
        )
        if graph_feature_space == "raw_decoder":
            decoded_raw = self.bridge_completed_to_recommendation_raw(graph_features)
            graph_features = {
                modality: torch.where(
                    masks[modality].unsqueeze(1),
                    raw_features[modality],
                    decoded_raw[modality],
                )
                for modality in self.modalities
            }
        elif graph_feature_space != "shared":
            raise ValueError(f"Unsupported item graph feature space: {graph_feature_space}")
        return graph_features, masks, source, f"internal:{mask_scope}"

    def build_completed_item_graph(self):
        if not self.use_completed_item_graph:
            return

        kind = getattr(self.env.args, "item_graph_kind", "cf")
        if kind not in (
            "fused_completed",
            "modality_completed",
            "modality_completed_confidence",
            "modality_completed_dynamic_confidence",
            "fused_completed_confidence",
            "fused_completed_dynamic_confidence",
        ):
            raise ValueError(f"Unsupported completed item graph kind: {kind}")

        topk = int(getattr(self.env.args, "item_graph_topk", 20))
        norm_type = getattr(self.env.args, "item_graph_norm", "rw")
        chunk_size = int(getattr(self.env.args, "item_graph_feature_chunk_size", 1024))

        was_training = self.training
        self.eval()
        with torch.no_grad():
            graph_features, masks, graph_feature_source, graph_feature_dir = self._build_item_graph_feature_inputs()
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
        }
        modality_weights = self._item_graph_modality_weights(weights)
        cf_scale = getattr(self.env.args, "item_graph_cf_scale", "raw")
        cf_power = float(getattr(self.env.args, "item_graph_cf_power", 0.5))
        cf_clip = float(getattr(self.env.args, "item_graph_cf_clip", 3.0))
        graphs = {
            "cf": self.dataset._build_cf_item_graph(
                topk,
                scale=cf_scale,
                power=cf_power,
                clip=cf_clip,
            ),
        }
        if max(weights["image"], modality_weights["v"]["semantic"]) > 0.0:
            graphs["image"] = self.dataset._build_feature_item_graph(graph_feature_np["v"], topk, chunk_size)
        if max(weights["text"], modality_weights["t"]["semantic"]) > 0.0:
            graphs["text"] = self.dataset._build_feature_item_graph(graph_feature_np["t"], topk, chunk_size)
        if "a" in self.modalities and max(weights["audio"], modality_weights["a"]["semantic"]) > 0.0:
            graphs["audio"] = self.dataset._build_feature_item_graph(graph_feature_np["a"], topk, chunk_size)
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
                f"feature_source={graph_feature_source}, feature_dir={graph_feature_dir}, "
                f"cf_scale={cf_scale},cf_power={cf_power},cf_clip={cf_clip}, "
                f"weights=cf:{weights['cf']},image:{weights['image']},text:{weights['text']},audio:{weights['audio']}, "
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
                f"feature_source={graph_feature_source}, feature_dir={graph_feature_dir}, "
                f"cf_scale={cf_scale},cf_power={cf_power},cf_clip={cf_clip}, "
                f"weights=cf:{weights['cf']},image:{weights['image']},text:{weights['text']},audio:{weights['audio']}, "
                f"edge_conf_init=rr:{conf[0]:.4f},ri:{conf[1]:.4f},ii:{conf[2]:.4f}, "
                f"edge_conf_coeff=rr:{coeff[0]:.4f},ri:{coeff[1]:.4f},ii:{coeff[2]:.4f}, "
                f"transform={self.item_graph_confidence_transform}, "
                f"blend={self.item_graph_confidence_blend:.4f}, "
                f"range=[{self.item_graph_confidence_min:.4f},{self.item_graph_confidence_max:.4f}]"
            )
            return

        if kind == "modality_completed":
            self.ItemItemGraphs = {}
            self.ItemItemGraphComponents = {}
            modality_graph_specs = {
                "cf": ({"cf": graphs["cf"]}, {"cf": weights["cf"]}),
                "v": (
                    {"cf": graphs["cf"], "image": graphs["image"]},
                    {"cf": modality_weights["v"]["cf"], "image": modality_weights["v"]["semantic"]},
                ),
                "t": (
                    {"cf": graphs["cf"], "text": graphs["text"]},
                    {"cf": modality_weights["t"]["cf"], "text": modality_weights["t"]["semantic"]},
                ),
            }
            if "a" in self.modalities and "audio" in graphs:
                modality_graph_specs["a"] = (
                    {"cf": graphs["cf"], "audio": graphs["audio"]},
                    {"cf": modality_weights["a"]["cf"], "audio": modality_weights["a"]["semantic"]},
                )
            for key, (graph_parts, graph_weights) in modality_graph_specs.items():
                graph = self._build_weighted_item_item_graph(
                    graph_parts,
                    graph_weights,
                    topk,
                    norm_type,
                    required_names=list(graph_parts.keys()),
                    context=f"completed {key} item graph",
                )
                self.ItemItemGraphs[key] = self.dataset._convert_sp_mat_to_sp_tensor(graph).coalesce().to(self.env.device)
            self.ItemItemGraph = None
            print(
                f"built completed item-item graph kind={kind}, topk={topk}, norm={norm_type}, "
                f"graphs={','.join(sorted(self.ItemItemGraphs.keys()))}, "
                f"feature_space={getattr(self.env.args, 'item_graph_feature_space', 'shared')}, "
                f"feature_source={graph_feature_source}, feature_dir={graph_feature_dir}, "
                f"cf_scale={cf_scale},cf_power={cf_power},cf_clip={cf_clip}, "
                f"weights=cf:{weights['cf']},image:{weights['image']},text:{weights['text']},audio:{weights['audio']}, "
                f"modal_weights={self._format_item_graph_modality_weights(modality_weights)}"
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

            effective_weights = {}
            for key, (feature_graph_name, modality_name) in modality_graph_specs.items():
                if feature_graph_name not in graphs:
                    continue
                cf_weight = modality_weights[modality_name]["cf"]
                feature_weight = modality_weights[modality_name]["semantic"]
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
                f"feature_source={graph_feature_source}, feature_dir={graph_feature_dir}, "
                f"cf_scale={cf_scale},cf_power={cf_power},cf_clip={cf_clip}, "
                f"weights=cf:{weights['cf']},image:{weights['image']},text:{weights['text']},audio:{weights['audio']}, "
                f"modal_weights={self._format_item_graph_modality_weights(modality_weights)}, "
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

            effective_weights = {}
            candidate_widths = {}
            for key, (feature_graph_name, modality_name) in modality_graph_specs.items():
                if feature_graph_name not in graphs:
                    continue
                cf_weight = modality_weights[modality_name]["cf"]
                feature_weight = modality_weights[modality_name]["semantic"]
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
                f"feature_source={graph_feature_source}, feature_dir={graph_feature_dir}, "
                f"cf_scale={cf_scale},cf_power={cf_power},cf_clip={cf_clip}, "
                f"weights=cf:{weights['cf']},image:{weights['image']},text:{weights['text']},audio:{weights['audio']}, "
                f"modal_weights={self._format_item_graph_modality_weights(modality_weights)}, "
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
            f"feature_source={graph_feature_source}, feature_dir={graph_feature_dir}, "
            f"cf_scale={cf_scale},cf_power={cf_power},cf_clip={cf_clip}, "
            f"weights=cf:{weights['cf']},image:{weights['image']},text:{weights['text']},audio:{weights['audio']}"
        )

    def _item_graph_modality_weights(self, weights):
        args = self.env.args

        def override(name, fallback):
            value = getattr(args, name, None)
            return float(fallback if value is None else value)

        return {
            "v": {
                "cf": override("item_graph_image_cf_weight", weights["cf"]),
                "semantic": override("item_graph_image_semantic_weight", weights["image"]),
            },
            "t": {
                "cf": override("item_graph_text_cf_weight", weights["cf"]),
                "semantic": override("item_graph_text_semantic_weight", weights["text"]),
            },
            "a": {
                "cf": override("item_graph_audio_cf_weight", weights["cf"]),
                "semantic": override("item_graph_audio_semantic_weight", weights["audio"]),
            },
        }

    def _format_item_graph_modality_weights(self, modality_weights):
        names = (("v", "image"), ("t", "text"), ("a", "audio"))
        return ",".join(
            f"{label}:cf={modality_weights[key]['cf']},semantic={modality_weights[key]['semantic']}"
            for key, label in names
            if key in modality_weights
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
        shared_target_values = [
            float(getattr(self.env.args, "item_graph_rr_confidence_reg_target", None) or base_target),
            float(getattr(self.env.args, "item_graph_ri_confidence_reg_target", None) or base_target),
            float(getattr(self.env.args, "item_graph_ii_confidence_reg_target", None) or base_target),
        ]

        def build_target(modality=None, dtype=None, device=None):
            values = list(shared_target_values)
            if modality in ("t", "v", "a"):
                prefix = {"t": "text", "v": "image", "a": "audio"}[modality]
                for idx, edge_type in enumerate(("rr", "ri", "ii")):
                    override = getattr(
                        self.env.args,
                        f"item_graph_{prefix}_{edge_type}_confidence_reg_target",
                        None,
                    )
                    if override is not None:
                        values[idx] = float(override)
            return self.item_emb.weight.new_tensor(values, dtype=dtype, device=device)

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
                (value - build_target(name, dtype=value.dtype, device=value.device)).pow(2).mean()
                for name, value in conf.items()
                if name in active_modalities
            ]
        else:
            target = build_target(dtype=conf.dtype, device=conf.device)
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
        heads = {"v": self.contra_head_v, "t": self.contra_head_t}
        if "a" in self.modalities:
            heads["a"] = self.contra_head_a
        return heads

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
                        f"to equal contra_dim ({self.promrl_dim}); modality {modality} has {source.size(1)}"
                    )
                projected[modality] = torch.nan_to_num(
                    F.normalize(source, dim=-1), nan=0.0, posinf=0.0, neginf=0.0
                )
            return projected

        heads = self._current_contra_heads()
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
        eye = torch.eye(self.d_beta, device=device)
        prior_precision_tensor = None
        if torch.is_tensor(prior_precision):
            prior_precision_tensor = prior_precision.to(device=device, dtype=eye.dtype)
            V_inv = eye.unsqueeze(0) + torch.diag_embed(prior_precision_tensor)
        else:
            precision = float(prior_precision)
            V_inv = precision * eye
        rhs = torch.zeros(N, self.d_beta, device=device)
        if beta_prior is not None:
            if prior_precision_tensor is not None:
                rhs = rhs + prior_precision_tensor * beta_prior
            else:
                rhs = rhs + float(prior_precision) * beta_prior

        for modality in observed_modalities:
            if obs_feats.get(modality) is None:
                continue
            W = self.W[modality]
            sigma2 = torch.exp(2 * self.log_sigma[modality].squeeze())
            observed_precision = (1.0 / sigma2) * (W.T @ W)
            V_inv = V_inv + observed_precision
            residual = obs_feats[modality] - self.mu[modality].unsqueeze(0)
            rhs = rhs + (1.0 / sigma2) * (residual @ W)

        if prior_precision_tensor is not None:
            V = torch.inverse(V_inv + 1e-6 * eye.unsqueeze(0))
            mean = torch.bmm(V, rhs.unsqueeze(-1)).squeeze(-1)
        else:
            V = torch.inverse(V_inv + 1e-6 * eye)
            mean = rhs @ V.T
        if return_cov:
            return mean, V
        return mean

    def _linear_beta_to_shared(self, beta, modality):
        return beta @ self.W[modality].transpose(0, 1) + self.mu[modality].unsqueeze(0)

    def _beta_to_completed_shared(self, beta, modality):
        linear_recon = self._linear_beta_to_shared(beta, modality)
        if not self.use_beta_completion_decoder:
            recon = linear_recon
        elif self.beta_completion_mode == "decoder":
            recon = self.beta_completion_decoders[modality](beta)
        else:
            raise ValueError(f"Unsupported beta_completion_mode: {self.beta_completion_mode}")
        recon = torch.nan_to_num(recon, nan=0.0, posinf=0.0, neginf=0.0)
        return F.normalize(recon, dim=-1)

    def _beta_completion_decoder_rec_loss(self, beta, batch_data, observed_modalities):
        if not self.use_beta_completion_decoder or self.beta_completion_rec_weight <= 0.0:
            return torch.zeros((), device=self.env.device)

        beta_input = beta.detach() if self.beta_completion_detach_beta else beta
        losses = []
        for modality in observed_modalities:
            if batch_data.get(modality) is None:
                continue
            pred = self._beta_to_completed_shared(beta_input, modality)
            target = batch_data[modality]
            if self.beta_completion_rec_loss in ("mse", "mse_cosine"):
                losses.append(F.mse_loss(pred, target, reduction="mean"))
            if self.beta_completion_rec_loss in ("cosine", "mse_cosine"):
                cosine = F.cosine_similarity(pred, target, dim=-1)
                losses.append(1.0 - cosine.mean())

        if not losses:
            return torch.zeros((), device=self.env.device)
        return self.beta_completion_rec_weight * torch.stack(losses).mean()

    def _select_smore_beta_prior_features(self, split=None):
        if not self.smore_beta_prior_features:
            return None
        if split in ("val", "test"):
            return self.smore_beta_prior_features["eval"]
        if split == "train":
            return self.smore_beta_prior_features["train"]
        if not self.training and self.uses_split_smore_beta_prior_features:
            return self.smore_beta_prior_features["eval"]
        return self.smore_beta_prior_features["train"]

    def _smore_beta_prior_params(self, item_ids, masks, split=None, stage=None):
        if not self._uses_smore_beta_prior_for_stage(stage):
            return None, None, None
        features = self._select_smore_beta_prior_features(split=split)
        if not features:
            return None, None, None

        N = next(iter(masks.values())).size(0)
        device = next(iter(masks.values())).device
        mean_sum = torch.zeros(N, self.d_beta, device=device)
        var_sum = torch.zeros(N, self.d_beta, device=device)
        prior_count = torch.zeros(N, 1, device=device)

        for modality, generator in self.smore_beta_prior_generators.items():
            if modality not in masks or modality not in features:
                continue
            modality_features = features[modality]
            if item_ids is not None:
                modality_features = modality_features[item_ids]
            modality_features = modality_features.to(device)
            observed = masks[modality].bool()
            nonzero = modality_features.abs().sum(dim=1) > 0
            valid = observed & nonzero
            if not valid.any():
                continue
            generated = torch.nan_to_num(
                generator(modality_features),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            prior_mean, prior_raw_var = torch.chunk(generated, 2, dim=1)
            prior_var = (
                self.smore_beta_prior_var_min
                + (self.smore_beta_prior_var_max - self.smore_beta_prior_var_min)
                * torch.sigmoid(prior_raw_var)
            )
            mean_sum[valid] = mean_sum[valid] + prior_mean[valid]
            var_sum[valid] = var_sum[valid] + prior_var[valid]
            prior_count[valid] = prior_count[valid] + 1.0

        has_prior = prior_count.squeeze(1) > 0
        if not has_prior.any():
            return None, None, None
        prior_mean = mean_sum / prior_count.clamp_min(1.0)
        prior_var = var_sum / prior_count.clamp_min(1.0)
        prior_precision = self.smore_beta_prior_lambda / prior_var.clamp_min(1e-8)
        return prior_mean, prior_precision, has_prior

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

    def impute_modalities(self, projected_feats, masks=None, item_ids=None, smore_split=None, stage=None):
        if self.disable_imputation:
            return {modality: feat.clone() for modality, feat in projected_feats.items()}

        if masks is None:
            full_masks = self._missing_masks()
            masks = full_masks
        completed = {modality: feat.clone() for modality, feat in projected_feats.items()}

        if all(mask.all() for mask in masks.values()):
            return completed

        pattern_keys = torch.stack([masks[modality] for modality in self.modalities], dim=1)
        prior_mean_all, prior_precision_all, prior_has_all = self._smore_beta_prior_params(
            item_ids,
            masks,
            split=smore_split,
            stage=stage,
        )
        unique_patterns = torch.unique(pattern_keys, dim=0)
        for pattern in unique_patterns:
            missing_modalities = [self.modalities[i] for i in range(len(self.modalities)) if not bool(pattern[i])]
            if not missing_modalities:
                continue

            pattern_selector = (pattern_keys == pattern.unsqueeze(0)).all(dim=1)
            if pattern_selector.sum() == 0:
                continue

            observed_modalities = [self.modalities[i] for i in range(len(self.modalities)) if bool(pattern[i])]
            if prior_has_all is None:
                subgroup_specs = [(pattern_selector, False)]
            else:
                subgroup_specs = [
                    (pattern_selector & prior_has_all, True),
                    (pattern_selector & ~prior_has_all, False),
                ]

            for selector, use_prior in subgroup_specs:
                if selector.sum() == 0:
                    continue
                obs_feats = {
                    modality: projected_feats[modality][selector]
                    for modality in observed_modalities
                }
                if use_prior:
                    m = self._posterior_mean_from_observed(
                        obs_feats,
                        observed_modalities,
                        beta_prior=prior_mean_all[selector],
                        prior_precision=prior_precision_all[selector],
                    )
                elif observed_modalities:
                    m = self._posterior_mean_from_observed(
                        obs_feats,
                        observed_modalities,
                        beta_prior=None,
                        prior_precision=1.0,
                    )
                else:
                    m = torch.zeros(selector.sum(), self.d_beta, device=self.env.device)

                for modality in missing_modalities:
                    completed[modality][selector] = self._beta_to_completed_shared(m, modality)

        return completed

    def get_gcn_modal_features(self):
        projected = self.project_features()
        if self.disable_imputation:
            return projected
        item_ids = torch.arange(self.m_item, device=self.env.device)
        completed = self.impute_modalities(
            projected,
            self._missing_masks(),
            item_ids=item_ids,
            stage=self.env.args.train_stage,
        )
        return completed

    def decode_completed_to_raw(self, completed_shared):
        decoded = {
            "v": F.normalize(self.decoder_v(completed_shared["v"]), dim=-1),
            "t": F.normalize(self.decoder_t(completed_shared["t"]), dim=-1),
        }
        if "a" in self.modalities:
            decoded["a"] = F.normalize(self.decoder_a(completed_shared["a"]), dim=-1)
        return decoded

    def bridge_completed_to_recommendation_raw(self, completed_shared):
        if self.feature_bridge_mode == "shared_identity":
            return completed_shared
        return self.decode_completed_to_raw(completed_shared)

    def _apply_fusion(self, item_source, deterministic=False):
        item_hidden = self.fusion_linear[0](item_source)
        if not deterministic:
            item_hidden = self.fusion_linear[1](item_hidden)
        item_emb = self.fusion_linear[2](item_hidden)
        return item_emb

    def _item_graph_for_modality(self, modality=None):
        if self.item_graph_kind in ("modality_completed", "modality_completed_confidence"):
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

    def _missing_weighted_item_fusion_weights(self, item_outputs, raw_features=None, observed_masks=None):
        if observed_masks is None:
            if raw_features is None:
                return None
            observed_masks = self._missing_masks(raw_features=raw_features)

        imputed_weight = min(
            max(float(getattr(self.env.args, "missing_fusion_imputed_weight", 0.7)), 0.0),
            1.0,
        )
        weights = {}
        for modality in self.modalities:
            observed = observed_masks[modality].unsqueeze(1)
            weights[modality] = torch.where(
                observed,
                torch.ones_like(item_outputs[modality][..., :1]),
                torch.full_like(item_outputs[modality][..., :1], imputed_weight),
            )
        denom = torch.stack([weights[modality] for modality in self.modalities], dim=0).sum(dim=0)
        denom = denom.clamp_min(1e-8)
        return {modality: weights[modality] / denom for modality in self.modalities}

    def _global_weighted_item_source(self, item_outputs):
        weights = F.softmax(self.global_fusion_params[0], dim=0)
        for idx, modality in enumerate(self.modalities):
            self.latest_rum_fusion_metrics[f"global_fusion_weight_{modality}"] = float(
                weights[idx].detach().cpu()
            )
        return sum(weights[idx] * item_outputs[modality] for idx, modality in enumerate(self.modalities))

    def _reliability_weighted_item_source(self, item_outputs, reliability_gates=None):
        if reliability_gates is None:
            return sum(item_outputs[modality] for modality in self.modalities) / len(self.modalities)
        weights = {
            modality: reliability_gates[modality].to(dtype=item_outputs[modality].dtype)
            for modality in self.modalities
        }
        denom = torch.stack([weights[modality] for modality in self.modalities], dim=0).sum(dim=0)
        denom = denom.clamp_min(1e-8)
        weights = {modality: weights[modality] / denom for modality in self.modalities}
        for modality in self.modalities:
            self.latest_completion_gate_metrics[f"completion_gate_{modality}_fusion_weight_mean"] = float(
                weights[modality].detach().mean().cpu()
            )
        return sum(weights[modality] * item_outputs[modality] for modality in self.modalities)

    def _fuse_item_sources(
        self,
        item_outputs,
        modal_features=None,
        raw_features=None,
        observed_masks=None,
        reliability_gates=None,
    ):
        if self.use_global_weighted_fusion:
            return self._global_weighted_item_source(item_outputs)

        if self.use_reliability_weighted_fusion:
            return self._reliability_weighted_item_source(
                item_outputs,
                reliability_gates=reliability_gates,
            )

        if self.use_missing_weighted_fusion:
            weights = self._missing_weighted_item_fusion_weights(
                item_outputs,
                raw_features=raw_features,
                observed_masks=observed_masks,
            )
            if weights is not None:
                return sum(weights[modality] * item_outputs[modality] for modality in self.modalities)

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

    def _missing_aware_gate_stats(self, modality, raw_features, masks, completed_shared, decoded_raw=None):
        observed = masks[modality].float().unsqueeze(1)
        completed = completed_shared[modality]
        completed_norm = completed.norm(dim=1, keepdim=True)

        if decoded_raw is None:
            decoded_raw = self.decode_completed_to_raw(completed_shared)
        decoded = decoded_raw[modality]
        raw = raw_features[modality]
        decoded_norm = decoded.norm(dim=1, keepdim=True)

        recon_mse = torch.mean((decoded - raw) ** 2, dim=1, keepdim=True)
        recon_cos = F.cosine_similarity(decoded, raw, dim=1, eps=1e-8).unsqueeze(1)
        recon_mse = torch.where(masks[modality].unsqueeze(1), recon_mse, torch.zeros_like(recon_mse))
        recon_cos = torch.where(masks[modality].unsqueeze(1), recon_cos, torch.zeros_like(recon_cos))

        graph = self.ItemItemGraph
        if graph is not None:
            neigh = torch.sparse.mm(graph, completed)
            neigh = F.normalize(neigh, dim=1)
            neighbor_cos = F.cosine_similarity(
                F.normalize(completed, dim=1),
                neigh,
                dim=1,
                eps=1e-8,
            ).unsqueeze(1)
        else:
            neighbor_cos = torch.zeros_like(observed)

        if self.completion_gate_stats_norm:
            completed_norm = torch.log1p(completed_norm).clamp(max=5.0) / 5.0
            decoded_norm = torch.log1p(decoded_norm).clamp(max=5.0) / 5.0
            recon_mse = torch.log1p(recon_mse).clamp(max=5.0) / 5.0
            recon_cos = (recon_cos + 1.0) * 0.5
            neighbor_cos = (neighbor_cos + 1.0) * 0.5

        stats = torch.cat(
            [
                observed,
                completed_norm,
                decoded_norm,
                recon_mse,
                recon_cos,
                neighbor_cos,
            ],
            dim=1,
        )
        return torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)

    def get_recommender_modal_features(self, raw_features=None, allow_imputer_grad=False):
        item_ids = None
        if raw_features is None:
            item_ids = torch.arange(self.m_item, device=self.env.device)
        raw_features = raw_features or self._current_raw_modal_features()
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
            stage=self.env.args.train_stage,
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
        self.latest_completion_gate_supervision_loss = torch.zeros((), device=self.env.device)
        self.latest_completion_gate_counterfactual_loss = torch.zeros((), device=self.env.device)
        self.latest_reliability_gates = None
        self.latest_completion_gate_masks = None
        self.latest_completion_gate_stats = None
        external_completed_masks = (
            self.disable_imputation
            and self.modal_feature_mask_source == "external_observed"
        )
        if not self.use_completion_gate or (self.disable_imputation and not external_completed_masks):
            return ones

        raw_features = raw_features or self._current_raw_modal_features()
        masks = self._missing_masks(raw_features)
        if all(mask.all() for mask in masks.values()):
            return ones

        projected = self.project_features(raw_features=raw_features)
        if external_completed_masks:
            completed_shared = projected
        else:
            item_ids = None
            if raw_features is None:
                item_ids = torch.arange(self.m_item, device=self.env.device)
            completed_shared = self._build_completed_features(
                projected,
                masks,
                item_ids=item_ids,
                stage=self.env.args.train_stage,
            )
        decoded_raw_for_gate = (
            self.decode_completed_to_raw(completed_shared)
            if self.use_missing_aware_reliability_gate and not external_completed_masks
            else None
        )
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
        gate_stats_by_modality = {}
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
                if self.use_missing_aware_reliability_gate:
                    gate_stats = self._missing_aware_gate_stats(
                        modality,
                        raw_features,
                        masks,
                        completed_shared,
                        decoded_raw=decoded_raw_for_gate,
                    )
                    gate_stats_by_modality[modality] = gate_stats
                    if self.completion_gate_detach_inputs:
                        gate_stats = gate_stats.detach()
                    gate_inputs.append(gate_stats)
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
                or (
                    self.use_missing_aware_reliability_gate
                    and self.completion_gate_apply_observed
                )
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
                or self.use_missing_aware_reliability_gate
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
        self.latest_reliability_gates = gates
        self.latest_completion_gate_masks = masks
        self.latest_completion_gate_stats = gate_stats_by_modality
        return gates

    def completion_gate_supervision_loss(self):
        if not self.use_missing_aware_reliability_gate:
            return torch.zeros((), device=self.env.device)
        if (
            self.latest_reliability_gates is None
            or self.latest_completion_gate_masks is None
            or self.latest_completion_gate_stats is None
        ):
            return torch.zeros((), device=self.env.device)

        observed_target_value = float(
            getattr(self.env.args, "completion_gate_supervision_observed_target", 1.0)
        )
        losses = []
        for modality in self.modalities:
            gate = self.latest_reliability_gates.get(modality)
            stats = self.latest_completion_gate_stats.get(modality)
            observed_mask = self.latest_completion_gate_masks.get(modality)
            if gate is None or stats is None or observed_mask is None:
                continue

            observed = observed_mask.float().unsqueeze(1)
            observed_target = torch.full_like(gate, observed_target_value)
            # stats columns: observed, completed_norm, decoded_norm, recon_mse,
            # recon_cos, neighbor_cos. For true-missing items the reconstruction
            # target is unavailable, so use graph-neighbor consistency as the
            # reliability soft target.
            neighbor_score = stats[:, 5:6]
            if not self.completion_gate_stats_norm:
                neighbor_score = (neighbor_score + 1.0) * 0.5
            neighbor_score = neighbor_score.detach().clamp(0.0, 1.0)
            missing_target = (
                self.completion_gate_floor
                + (1.0 - self.completion_gate_floor) * neighbor_score
            )
            target = observed * observed_target + (1.0 - observed) * missing_target
            losses.append(torch.mean((gate - target.detach()) ** 2))

        if not losses:
            return torch.zeros((), device=self.env.device)
        self.latest_completion_gate_supervision_loss = torch.stack(losses).mean()
        self.latest_completion_gate_metrics["completion_gate_supervision_loss"] = float(
            self.latest_completion_gate_supervision_loss.detach().cpu()
        )
        return self.latest_completion_gate_supervision_loss

    def _missing_aware_gate_stats_local(self, modality, raw_features, masks, completed_shared, decoded_raw):
        observed = masks[modality].float().unsqueeze(1)
        completed = completed_shared[modality]
        decoded = decoded_raw[modality]
        raw = raw_features[modality]
        completed_norm = completed.norm(dim=1, keepdim=True)
        decoded_norm = decoded.norm(dim=1, keepdim=True)
        recon_mse = torch.mean((decoded - raw) ** 2, dim=1, keepdim=True)
        recon_cos = F.cosine_similarity(decoded, raw, dim=1, eps=1e-8).unsqueeze(1)
        # No full item graph is available for mini-batch pseudo missing; use a
        # neutral neighbor-consistency input so the decoder quality drives this
        # auxiliary target.
        neighbor_cos = torch.zeros_like(observed)

        if self.completion_gate_stats_norm:
            completed_norm = torch.log1p(completed_norm).clamp(max=5.0) / 5.0
            decoded_norm = torch.log1p(decoded_norm).clamp(max=5.0) / 5.0
            recon_mse = torch.log1p(recon_mse).clamp(max=5.0) / 5.0
            recon_cos = (recon_cos + 1.0) * 0.5
            neighbor_cos = torch.full_like(neighbor_cos, 0.5)

        stats = torch.cat(
            [
                observed,
                completed_norm,
                decoded_norm,
                recon_mse,
                recon_cos,
                neighbor_cos,
            ],
            dim=1,
        )
        return torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)

    def completion_gate_counterfactual_loss(self, item_ids):
        if not self.use_missing_aware_reliability_gate:
            return torch.zeros((), device=self.env.device)
        ratio = float(getattr(self.env.args, "completion_gate_counterfactual_ratio", 0.5))
        if ratio <= 0.0:
            return torch.zeros((), device=self.env.device)

        item_ids = torch.unique(item_ids)
        if item_ids.numel() == 0:
            return torch.zeros((), device=self.env.device)

        full_raw = self._current_raw_modal_features(full=True)
        raw_batch = {modality: full_raw[modality][item_ids] for modality in self.modalities}
        masks = self._missing_masks(raw_features=raw_batch)
        pseudo_masks, pseudo_selected = self._sample_pseudo_missing_masks(masks, ratio=ratio)
        if pseudo_masks is None or pseudo_selected is None:
            return torch.zeros((), device=self.env.device)

        projected = self.project_features(item_ids=item_ids, raw_features=full_raw)
        completed = self._build_completed_features(
            projected,
            pseudo_masks,
            item_ids=item_ids,
            stage=self.env.args.train_stage,
        )
        decoded_raw = self.decode_completed_to_raw(completed)
        pattern = torch.stack([pseudo_masks[modality].float() for modality in self.modalities], dim=1)

        if self.completion_gate_item_context_source == "id_embedding":
            item_context = self.item_emb(item_ids)
        elif self.completion_gate_item_context_source == "shared_mean":
            item_context = torch.stack(
                [
                    torch.where(pseudo_masks[modality].unsqueeze(1), projected[modality], completed[modality])
                    for modality in self.modalities
                ],
                dim=0,
            ).mean(dim=0)
        else:
            item_context = None

        mse_temp = max(float(getattr(self.env.args, "completion_gate_counterfactual_mse_temp", 0.1)), 1e-6)
        losses = []
        target_values = []
        gate_values = []
        for modality in self.modalities:
            selected = pseudo_selected[modality]
            if not selected.any():
                continue

            gate_source = completed[modality]
            gate_stats = self._missing_aware_gate_stats_local(
                modality,
                raw_batch,
                pseudo_masks,
                completed,
                decoded_raw,
            )
            if self.completion_gate_detach_inputs:
                gate_source = gate_source.detach()
                gate_pattern = pattern.detach()
                gate_stats = gate_stats.detach()
                gate_item_context = item_context.detach() if item_context is not None else None
            else:
                gate_pattern = pattern
                gate_item_context = item_context

            gate_inputs = [gate_source]
            if gate_item_context is not None:
                gate_inputs.append(gate_item_context)
            gate_inputs.extend([gate_stats, gate_pattern])
            gate = torch.sigmoid(self.completion_gates[modality](torch.cat(gate_inputs, dim=1)))
            gate = self.completion_gate_floor + (1.0 - self.completion_gate_floor) * gate

            decoded = decoded_raw[modality][selected]
            raw = raw_batch[modality][selected]
            cos01 = (F.cosine_similarity(decoded, raw, dim=1, eps=1e-8).unsqueeze(1) + 1.0) * 0.5
            mse = torch.mean((decoded - raw) ** 2, dim=1, keepdim=True)
            mse_quality = torch.exp(-mse / mse_temp)
            quality = (0.5 * cos01 + 0.5 * mse_quality).detach().clamp(0.0, 1.0)
            target = self.completion_gate_floor + (1.0 - self.completion_gate_floor) * quality

            selected_gate = gate[selected]
            losses.append(torch.mean((selected_gate - target) ** 2))
            target_values.append(target.mean())
            gate_values.append(selected_gate.mean())

        if not losses:
            return torch.zeros((), device=self.env.device)

        self.latest_completion_gate_counterfactual_loss = torch.stack(losses).mean()
        self.latest_completion_gate_metrics["completion_gate_counterfactual_loss"] = float(
            self.latest_completion_gate_counterfactual_loss.detach().cpu()
        )
        self.latest_completion_gate_metrics["completion_gate_counterfactual_target_mean"] = float(
            torch.stack(target_values).mean().detach().cpu()
        )
        self.latest_completion_gate_metrics["completion_gate_counterfactual_gate_mean"] = float(
            torch.stack(gate_values).mean().detach().cpu()
        )
        return self.latest_completion_gate_counterfactual_loss

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
        v_user_emb, v_item_emb = self.v_gcn(
            modal_features["v"],
            user_id_emb,
            skip_mlp=False,
        )
        t_user_emb, t_item_emb = self.t_gcn(
            modal_features["t"],
            user_id_emb,
            skip_mlp=False,
        )
        v_item_emb = self._apply_item_graph_modal_residual(v_item_emb, observed_masks["v"], modality="v")
        t_item_emb = self._apply_item_graph_modal_residual(t_item_emb, observed_masks["t"], modality="t")
        ungated_item_outputs = {"v": v_item_emb, "t": t_item_emb}
        if not self.use_rum_fusion and not self.use_reliability_weighted_fusion:
            v_item_emb = v_item_emb * reliability_gates["v"]
            t_item_emb = t_item_emb * reliability_gates["t"]
        user_outputs = {"v": v_user_emb, "t": t_user_emb}
        item_outputs = {"v": v_item_emb, "t": t_item_emb}

        if "a" in self.modalities:
            a_user_emb, a_item_emb = self.a_gcn(
                modal_features["a"],
                user_id_emb,
                skip_mlp=False,
            )
            a_item_emb = self._apply_item_graph_modal_residual(a_item_emb, observed_masks["a"], modality="a")
            ungated_item_outputs["a"] = a_item_emb
            if not self.use_rum_fusion and not self.use_reliability_weighted_fusion:
                a_item_emb = a_item_emb * reliability_gates["a"]
            user_outputs["a"] = a_user_emb
            item_outputs["a"] = a_item_emb

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
                reliability_gates=reliability_gates,
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

    def _recommendation_frontend_item_embeddings(self, raw_features, item_ids, allow_modal_grad=False):
        modal_features = self.get_recommender_modal_features(
            raw_features=raw_features,
            allow_imputer_grad=allow_modal_grad,
        )
        item_outputs = {}
        modal_gcns = {"v": self.v_gcn, "t": self.t_gcn}
        if "a" in self.modalities:
            modal_gcns["a"] = self.a_gcn
        observed_masks = self._missing_masks(raw_features)
        for modality in self.modalities:
            features = modal_features[modality][item_ids]
            item_output = modal_gcns[modality].MLP(features)
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

    def _recommendation_gcn_modality_item_embeddings(self, raw_features=None, allow_modal_grad=False):
        raw_features = raw_features or self._current_raw_modal_features()
        modal_features = self.get_recommender_modal_features(
            raw_features=raw_features,
            allow_imputer_grad=allow_modal_grad,
        )
        user_id_emb = self.user_emb.weight
        modal_gcns = {"v": self.v_gcn, "t": self.t_gcn}
        if "a" in self.modalities:
            modal_gcns["a"] = self.a_gcn

        item_outputs = {}
        for modality in self.modalities:
            _, item_emb = modal_gcns[modality](
                modal_features[modality],
                user_id_emb,
                skip_mlp=False,
            )
            item_outputs[modality] = F.normalize(
                torch.nan_to_num(item_emb, nan=0.0, posinf=0.0, neginf=0.0),
                dim=-1,
            )
        return item_outputs

    def compute_true_missing_gcn_infonce_loss(
        self,
        item_ids,
        temperature=0.2,
        bank_size=256,
        allow_modal_grad=False,
    ):
        """InfoNCE for real missing modalities after modality-specific GCNs.

        Anchor: imputed representation of a truly missing modality.
        Positive: same item's observed modality representations.
        Negatives: other items whose same modality is observed.
        """
        zero = torch.zeros((), device=self.env.device)
        external_completed_masks = self.modal_feature_mask_source == "external_observed"
        if (self.disable_imputation and not external_completed_masks) or item_ids is None or item_ids.numel() == 0:
            return zero

        raw_features = self._current_raw_modal_features()
        masks = self._missing_masks(raw_features)
        item_ids = torch.unique(item_ids.detach()).long()
        item_ids = item_ids[(item_ids >= 0) & (item_ids < self.m_item)]
        if item_ids.numel() < 2:
            return zero

        max_bank_size = int(bank_size or 0)
        bank_ids = item_ids
        if max_bank_size > 0 and bank_ids.numel() > max_bank_size:
            perm = torch.randperm(bank_ids.numel(), device=self.env.device)[:max_bank_size]
            bank_ids = bank_ids[perm]
        if bank_ids.numel() < 2:
            return zero

        gcn_item_outputs = self._recommendation_gcn_modality_item_embeddings(
            raw_features=raw_features,
            allow_modal_grad=allow_modal_grad,
        )
        temp = max(float(temperature), 1e-6)
        losses = []
        modality_count = len(self.modalities)

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

            negative_ids = bank_ids[masks[modality][bank_ids]]
            if negative_ids.numel() == 0:
                continue

            anchor_emb = gcn_item_outputs[modality][anchor_ids]
            positive_parts = []
            positive_weights = []
            for observed_modality in self.modalities:
                if observed_modality == modality:
                    continue
                observed_mask = masks[observed_modality][anchor_ids].float().unsqueeze(1)
                positive_parts.append(gcn_item_outputs[observed_modality][anchor_ids] * observed_mask)
                positive_weights.append(observed_mask)
            if not positive_parts:
                continue
            positive_sum = torch.stack(positive_parts, dim=0).sum(dim=0)
            positive_count = torch.stack(positive_weights, dim=0).sum(dim=0).clamp_min(1.0)
            positive_emb = F.normalize(positive_sum / positive_count, dim=-1)
            negative_emb = gcn_item_outputs[modality][negative_ids].detach()

            pos_logits = (anchor_emb * positive_emb.detach()).sum(dim=1, keepdim=True) / temp
            neg_logits = torch.matmul(anchor_emb, negative_emb.transpose(0, 1)) / temp
            same_item = anchor_ids.unsqueeze(1).eq(negative_ids.unsqueeze(0))
            neg_logits = neg_logits.masked_fill(same_item, -1e9)
            logits = torch.cat([pos_logits, neg_logits], dim=1)
            targets = torch.zeros(logits.size(0), dtype=torch.long, device=self.env.device)
            losses.append(F.cross_entropy(logits, targets))

        if not losses:
            return zero
        return torch.stack(losses).mean()

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
            smore_split=split,
            stage=self.env.args.train_stage,
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
            smore_split=split,
            stage=self.env.args.train_stage,
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
            real_raw = full_raw[modality][modality_items]

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
                random_raw = full_raw[modality][random_ids]
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
    def compute_stage1_heldout_metrics(self, split="val", include_random_baseline=True):
        pseudo_shared_metrics = self.compute_pseudo_shared_metrics(
            split=split,
            include_random_baseline=include_random_baseline,
        )
        pseudo_shared_overall = pseudo_shared_metrics.get("_overall", {})
        pseudo_shared_cosine = float(pseudo_shared_overall.get("cosine", 0.0))
        random_pseudo_shared_cosine = float(pseudo_shared_overall.get("random_cosine", 0.0))

        return {
            "split": split,
            "pseudo_shared_count": int(pseudo_shared_overall.get("count", 0)),
            "pseudo_shared_mse": float(pseudo_shared_overall.get("mse", 0.0)),
            "pseudo_shared_cosine": pseudo_shared_cosine,
            "pseudo_shared_cosine_gap": float(pseudo_shared_cosine - random_pseudo_shared_cosine),
            "pseudo_shared_random_mse": float(pseudo_shared_overall.get("random_mse", 0.0)),
            "pseudo_shared_random_cosine": random_pseudo_shared_cosine,
        }

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
        eigenvectors, S_V = eigenvalue_computation_pmcl(all_features)
        eigenvalues = S_V ** 2

        targets = torch.zeros(eigenvalues.size(0), dtype=torch.long, device=eigenvalues.device)
        loss_intra = F.cross_entropy(eigenvalues / self.env.args.tau1, targets)

        principal_eigenvector = eigenvectors[:, :, 0]
        bs = principal_eigenvector.size(0)
        sim = principal_eigenvector @ principal_eigenvector.T / self.env.args.tau2
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
            completed_feats = self._build_completed_features(
                projected,
                masks,
                item_ids=item_ids,
                stage=self.env.args.train_stage,
            )

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
        prior_mean_all, prior_precision_all, prior_has_all = self._smore_beta_prior_params(
            item_ids,
            masks,
            stage=stage,
        )
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

            pattern_selector = (pattern_keys == pattern.unsqueeze(0)).all(dim=1)
            if pattern_selector.sum() == 0:
                continue

            if prior_has_all is None:
                subgroup_specs = [(pattern_selector, False)]
            else:
                subgroup_specs = [
                    (pattern_selector & prior_has_all, True),
                    (pattern_selector & ~prior_has_all, False),
                ]

            for selector, use_prior in subgroup_specs:
                if selector.sum() == 0:
                    continue
                batch_data = {
                    modality: projected[modality][selector]
                    for modality in observed_modalities
                }
                if use_prior:
                    posterior_mean, posterior_cov = self._posterior_mean_from_observed(
                        batch_data,
                        observed_modalities,
                        beta_prior=prior_mean_all[selector],
                        prior_precision=prior_precision_all[selector],
                        return_cov=True,
                    )
                    nll_posterior = posterior_mean
                else:
                    posterior_mean, posterior_cov = self._posterior_mean_from_observed(
                        batch_data,
                        observed_modalities,
                        beta_prior=None,
                        prior_precision=1.0,
                        return_cov=True,
                    )
                    nll_posterior = posterior_mean.detach()

                if self._imputer_updates_enabled and stage in (
                    "imputer_param",
                    "imputer_backprop",
                    "imputer_init",
                    "imputer_align",
                    "imputer_promrl_main",
                    "joint",
                ):
                    self.queue_em_update(batch_data, posterior_mean, posterior_cov)
                pattern_rec_loss, _ = compute_nll_loss(
                    batch_data,
                    nll_posterior,
                    {modality: self.W[modality] for modality in self.modalities},
                    {modality: self.mu[modality] for modality in self.modalities},
                    {modality: self.log_sigma[modality].squeeze() for modality in self.modalities},
                    observed_modalities,
                )
                pattern_rec_loss = pattern_rec_loss + self._beta_completion_decoder_rec_loss(
                    posterior_mean,
                    batch_data,
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

        split_offset = {"train": 0, "val": 1, "test": 2}.get(split, 3)
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
        pseudo_completed = self._build_completed_features(
            projected,
            pseudo_masks,
            smore_split=split,
            stage=self.env.args.train_stage,
        )

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
        smore_split=None,
        stage=None,
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
                smore_split=smore_split,
                stage=stage,
            )

        N = projected[self.modalities[0]].size(0)
        device = projected[self.modalities[0]].device

        # Pre-compute imputed values for each modality (only filled for missing items)
        imputed = {
            modality: torch.zeros(N, self.promrl_dim, device=device)
            for modality in self.modalities
        }

        pattern_keys = torch.stack([masks[m] for m in self.modalities], dim=1)
        prior_mean_all, prior_precision_all, prior_has_all = self._smore_beta_prior_params(
            item_ids,
            masks,
            split=smore_split,
            stage=stage,
        )

        with torch.no_grad():
            for pattern in torch.unique(pattern_keys, dim=0):
                missing = [self.modalities[i] for i in range(len(self.modalities)) if not bool(pattern[i])]
                if not missing:
                    continue

                pattern_selector = (pattern_keys == pattern.unsqueeze(0)).all(dim=1)
                if pattern_selector.sum() == 0:
                    continue

                observed = [self.modalities[i] for i in range(len(self.modalities)) if bool(pattern[i])]
                if prior_has_all is None:
                    subgroup_specs = [(pattern_selector, False)]
                else:
                    subgroup_specs = [
                        (pattern_selector & prior_has_all, True),
                        (pattern_selector & ~prior_has_all, False),
                    ]

                for selector, use_prior in subgroup_specs:
                    if selector.sum() == 0:
                        continue
                    obs_feats = {m: projected[m][selector].detach() for m in observed}
                    if use_prior:
                        m = self._posterior_mean_from_observed(
                            obs_feats,
                            observed,
                            beta_prior=prior_mean_all[selector],
                            prior_precision=prior_precision_all[selector],
                        )
                    elif observed:
                        m = self._posterior_mean_from_observed(
                            obs_feats,
                            observed,
                            beta_prior=None,
                            prior_precision=1.0,
                        )
                    else:
                        m = torch.zeros(selector.sum(), self.d_beta, device=device)

                    for modality in missing:
                        imputed[modality][selector] = self._beta_to_completed_shared(m, modality)

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
                or float(getattr(self.env.args, "beta_decode_kl", 0.0)) != 0.0
            )
        else:
            need_rec = True
            need_contrastive = True
            need_itm = True
            need_decode = (
                float(getattr(self.env.args, "alpha_decode", 0.0)) != 0.0
                or float(getattr(self.env.args, "alpha_decode_kl", 0.0)) != 0.0
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

        if stage not in ("imputer_param", "imputer_init"):
            # Step 4-5: Build completed features + contrastive loss (like ProMRL)
            need_completed = need_contrastive or need_itm or need_decode
            completed = self._build_completed_features(
                projected,
                masks,
                item_ids=item_ids,
                stage=stage,
            ) if need_completed else None
            if need_contrastive:
                loss_intra, loss_inter = self.compute_contrastive_loss(completed)
            if need_itm:
                loss_itm_raw, loss_itm = self.compute_itm_loss(item_ids, completed_feats=completed)
            if self.use_decode_head and need_decode:
                decode_completed = completed
                decode_selected = masks
                decode_targets = raw_batch
                if (
                    decode_completed is not None
                    and getattr(self.env.args, "decode_loss_grad_mode", "coupled") == "detached"
                ):
                    decode_completed = {
                        modality: feat.detach()
                        for modality, feat in decode_completed.items()
                    }

                decode_losses = []
                decode_kl_losses = []
                if decode_completed is not None:
                    decoded_raw = self.decode_completed_to_raw(decode_completed)
                    for modality in self.modalities:
                        observed_mask = decode_selected[modality]
                        if need_decode and observed_mask.any():
                            decoded_obs = decoded_raw[modality][observed_mask]
                            raw_obs = decode_targets[modality][observed_mask]
                            cosine = F.cosine_similarity(decoded_obs, raw_obs, dim=-1)
                            decode_losses.append(1.0 - cosine.mean())
                            if observed_mask.sum() > 1:
                                temp = max(float(getattr(self.env.args, "decode_kl_temp", 0.2)), 1e-6)
                                decoded_norm = F.normalize(
                                    torch.nan_to_num(decoded_obs, nan=0.0, posinf=0.0, neginf=0.0),
                                    dim=-1,
                                )
                                raw_norm = F.normalize(
                                    torch.nan_to_num(raw_obs, nan=0.0, posinf=0.0, neginf=0.0),
                                    dim=-1,
                                )
                                teacher_logits = torch.matmul(raw_norm, raw_norm.transpose(0, 1)) / temp
                                student_logits = torch.matmul(decoded_norm, raw_norm.transpose(0, 1)) / temp
                                teacher = F.softmax(teacher_logits.detach(), dim=1)
                                student_log_prob = F.log_softmax(student_logits, dim=1)
                                decode_kl_losses.append(
                                    F.kl_div(student_log_prob, teacher, reduction="batchmean")
                                )
                    if decode_losses:
                        loss_decode = torch.stack(decode_losses).mean()
                    if decode_kl_losses:
                        loss_decode_kl = torch.stack(decode_kl_losses).mean()

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

        v_user_emb, v_item_emb = self.v_gcn(
            modal_features["v"], user_id_emb, skip_mlp=False
        )
        t_user_emb, t_item_emb = self.t_gcn(
            modal_features["t"], user_id_emb, skip_mlp=False
        )
        v_item_emb = self._apply_item_graph_modal_residual(v_item_emb, observed_masks["v"], modality="v")
        t_item_emb = self._apply_item_graph_modal_residual(t_item_emb, observed_masks["t"], modality="t")
        ungated_item_outputs = {"v": v_item_emb, "t": t_item_emb}
        if not self.use_rum_fusion:
            if not self.use_reliability_weighted_fusion:
                v_item_emb = v_item_emb * reliability_gates["v"]
                t_item_emb = t_item_emb * reliability_gates["t"]

        outputs = {
            "user_id": user_id_emb,
            "v": (v_user_emb, v_item_emb),
            "t": (t_user_emb, t_item_emb),
            "modal_inputs": modal_features,
            "reliability_gates": reliability_gates,
            "observed_masks": observed_masks,
            "ungated_item_outputs": ungated_item_outputs,
        }
        if "a" in self.modalities:
            a_user_emb, a_item_emb = self.a_gcn(
                modal_features["a"], user_id_emb, skip_mlp=False
            )
            a_item_emb = self._apply_item_graph_modal_residual(a_item_emb, observed_masks["a"], modality="a")
            ungated_item_outputs["a"] = a_item_emb
            if not self.use_rum_fusion:
                if not self.use_reliability_weighted_fusion:
                    a_item_emb = a_item_emb * reliability_gates["a"]
            outputs["a"] = (a_user_emb, a_item_emb)

        self._gcn_cache = outputs
        return outputs

    def clear_gcn_cache(self):
        self._gcn_cache = None

    def forward(self, random=False):
        outputs = self._run_modal_gcn()
        user_id_emb = outputs["user_id"]
        v_user_emb, v_item_emb = outputs["v"]
        t_user_emb, t_item_emb = outputs["t"]
        user_outputs = {"v": v_user_emb, "t": t_user_emb}
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
            item_outputs = {"v": v_item_emb.detach(), "t": t_item_emb.detach()}

        if "a" in self.modalities:
            a_user_emb, a_item_emb = outputs["a"]
            user_outputs["a"] = a_user_emb
            if not (self.use_rank_residual_completion_gate and not self.use_rum_fusion):
                item_outputs["a"] = a_item_emb.detach()

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
                reliability_gates=outputs["reliability_gates"],
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
        v_user_emb, v_item_emb = outputs["v"]
        t_user_emb, t_item_emb = outputs["t"]

        v_mm_emb = mix_ration[env][0] * v_item_emb
        t_mm_emb = mix_ration[env][1] * t_item_emb

        if "a" in self.modalities:
            a_user_emb, a_item_emb = outputs["a"]
            a_mm_emb = mix_ration[env][2] * a_item_emb
            item_emb = torch.cat([v_mm_emb, t_mm_emb, a_mm_emb], dim=1)
            user_emb = torch.cat(
                [
                    user_id_emb + v_user_emb,
                    user_id_emb + t_user_emb,
                    user_id_emb + a_user_emb,
                ],
                dim=1,
            )
        else:
            item_emb = torch.cat([v_mm_emb, t_mm_emb], dim=1)
            user_emb = torch.cat(
                [user_id_emb + v_user_emb, user_id_emb + t_user_emb],
                dim=1,
            )

        assert torch.isnan(user_emb).sum() == 0
        assert torch.isnan(item_emb).sum() == 0
        return user_emb, item_emb

    def modality_bpr_loss(self, batch_users, batch_pos_items, batch_neg_items):
        if "a" in self.modalities:
            mix_ration = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        else:
            mix_ration = [[1, 0], [0, 1]]

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
