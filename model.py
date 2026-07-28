import copy
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from promrl_core.layers import Contra_head, Match_head, _build_mlp
from promrl_core.utils.impute import update_posterior, compute_nll_loss
from promrl_core.utils.eigen import (
    eigenvalue_computation_pmcl,
    shifted_relation_lifted_directions,
)


def _load_tensor_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # PyTorch < 1.13 has no weights_only parameter. Public releases should
        # pin a newer PyTorch; this fallback keeps the development environment usable.
        return torch.load(path, map_location="cpu")


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

        self.audio_feat = None
        self.ori_audio_feat = None
        self.video_feat = None
        self.ori_video_feat = None

        native_image_feat = torch.tensor(dataset.image_feat, dtype=torch.float32).to(self.env.device)
        native_text_feat = torch.tensor(dataset.text_feat, dtype=torch.float32).to(self.env.device)
        self.ori_image_feat = F.normalize(native_image_feat)
        self.ori_text_feat = F.normalize(native_text_feat)

        self.eval_ori_image_feat = self.ori_image_feat
        self.eval_ori_text_feat = self.ori_text_feat

        if self.has_audio_modality:
            native_audio_feat = torch.tensor(dataset.audio_feat, dtype=torch.float32).to(self.env.device)
            self.ori_audio_feat = F.normalize(native_audio_feat)
            self.eval_ori_audio_feat = self.ori_audio_feat
        if self.has_video_modality:
            native_video_feat = torch.tensor(dataset.video_feat, dtype=torch.float32).to(self.env.device)
            self.ori_video_feat = F.normalize(native_video_feat)
            self.eval_ori_video_feat = self.ori_video_feat

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
        self.use_decode_head = not self.use_latent_completion_bridge
        self.gcn_frontend_mode = self.env.args.gcn_frontend_mode
        self.promrl_projection_mode = getattr(self.env.args, "promrl_projection_mode", "learned")
        self.promrl_dim = self.free_emb_dimension if self.use_latent_completion_bridge else self.contra_dim
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
            "modality_completed",
        )
        self.use_item_graph_modal_residual = (
            self.item_graph_modal_alpha > 0.0
            and self.item_graph_modal_layers > 0
        )
        self.fusion_mode = getattr(self.env.args, "fusion_mode", "mean")
        if self.fusion_mode not in {"mean", "posterior_reliability"}:
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")
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
        self.fusion_linear = nn.Sequential(
            nn.Linear(self.free_emb_dimension, self.free_emb_dimension, bias=False),
            nn.Dropout(),
            nn.Tanh(),
        )

        self.final_item = None
        self.final_user = None
        self.activate = torch.nn.Sigmoid()
        self.latest_promrl_losses = {}
        self._gcn_cache = None
        self._imputer_updates_enabled = True
        self._pending_em_updates = []
        self._dynamic_stage1_refresh_counter = 0
        self._item_user_sets = self._build_item_user_sets(dataset)
        self._co_interact_positive_items = self._build_co_interact_positive_items(dataset)

        torch.nn.init.normal_(self.user_emb.weight, std=0.1)
        torch.nn.init.normal_(self.item_emb.weight, std=0.1)
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

    def _decoder_hidden_dim(self, raw_dim):
        return min(1024, max(256, raw_dim // 2))

    def _build_modal_decoder(self, modality, raw_dim):
        hidden_dim = self._decoder_hidden_dim(raw_dim)
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
        modules.extend(getattr(self, f"{modality}_gcn") for modality in self.modalities)
        if self.use_decoupled_latent_bridge:
            modules.extend(getattr(self, f"comp_to_rec_{modality}") for modality in self.modalities)
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
            and canonical_stage in ("imputer_backprop", "recommender")
        )
        self._set_modules_trainable(self._decoder_modules(), decoder_trainable)
        self._imputer_updates_enabled = (
            getattr(self.env.args, "generative_update_mode", "em") == "em"
            and canonical_stage == "imputer_param"
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
        if self.use_latent_direct_bridge and self._canonical_stage() == "recommender":
            exclude = self._module_param_ids(self._projection_modules())
        return self._trainable_parameters(self._imputer_modules(), exclude)

    def get_decoder_parameters(self):
        return self._trainable_parameters(self._decoder_modules())

    def get_recommender_parameters(self):
        exclude = set()
        if self.use_latent_direct_bridge and self._canonical_stage() != "recommender":
            exclude = self._module_param_ids(self._projection_modules())
        return self._trainable_parameters(self._recommender_modules(), exclude)

    def load_full_checkpoint(self, ckpt_path):
        checkpoint = _load_tensor_checkpoint(ckpt_path)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.load_state_dict(state_dict, strict=True)
        return len(state_dict)

    def load_projection_checkpoint(self, ckpt_path):
        checkpoint = _load_tensor_checkpoint(ckpt_path)
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
        checkpoint = _load_tensor_checkpoint(ckpt_path)
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
            features = {"v": self.ori_image_feat, "t": self.ori_text_feat}
        else:
            features = {"v": self.image_feat, "t": self.text_feat}
        if "a" in self.modalities:
            if full:
                features["a"] = self.ori_audio_feat
            else:
                features["a"] = self.audio_feat
        if "d" in self.modalities:
            if full:
                features["d"] = self.ori_video_feat
            else:
                features["d"] = self.video_feat
        return features

    def get_split_raw_modal_features(self, split="test", full=False):
        if full:
            features = {
                "v": self.ori_image_feat,
                "t": self.ori_text_feat,
            }
            if "a" in self.modalities:
                features["a"] = self.ori_audio_feat
            if "d" in self.modalities:
                features["d"] = self.ori_video_feat
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
            "modality_completed",
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
        if self._uses_posterior_reliability_for("graph"):
            reliabilities = {
                modality: scores.detach().cpu().numpy().astype(np.float32)
                for modality, scores in self._posterior_completion_reliabilities(masks).items()
            }
        else:
            reliabilities = None
        reliability_blend = 1.0
        reliability_topk = self._uses_posterior_reliability_for("graph")
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
        def build_semantic_graph(feature, reliability):
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
        if kind in ("modality_masked", "modality_completed"):
            self.ItemItemGraphs = {}
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
                self.ItemItemGraphs[key] = self.dataset._convert_sp_mat_to_sp_tensor(graph).coalesce().to(self.env.device)
            self.ItemItemGraph = None
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

    def _build_single_item_item_graph(self, graph, topk, norm_type):
        graph = self.dataset._topk_sparse_rows(graph.tocsr(), topk)
        return self.dataset._normalize_item_graph(graph, norm_type)

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

    def _missing_masks(self, raw_features=None):
        if raw_features is None:
            raw_features = self._current_raw_modal_features()
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
        return {
            modality: F.normalize(feature, dim=-1)
            for modality, feature in decoded.items()
        }

    def bridge_completed_to_recommendation_raw(self, completed_shared):
        return self.decode_completed_to_raw(completed_shared)

    def _decoder_target(self, modality, normalized_target, item_ids=None):
        del modality, item_ids
        return normalized_target

    def _decoder_reconstruction_loss(self, modality, prediction, target):
        del modality
        return 1.0 - F.cosine_similarity(prediction, target, dim=-1).mean()

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
            graph = self._item_graph_for_modality(modality)
            if graph is None:
                raise RuntimeError("item_graph_modal_alpha > 0 requires an item-item graph")
            neigh = torch.sparse.mm(graph, out)
            out = (1.0 - alpha) * out + alpha * neigh
        if self.item_graph_modal_target == "missing" and observed_mask is not None:
            missing_mask = ~observed_mask
            out = torch.where(missing_mask.unsqueeze(1), out, item_emb)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _uses_posterior_reliability_for(self, component):
        if not self.use_posterior_reliability:
            return False
        return self.posterior_reliability_scope == "both" or self.posterior_reliability_scope == component

    def _fuse_item_sources(self, item_outputs, modal_features=None, raw_features=None, observed_masks=None):
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

    def compute_recommendation_embeddings(self, raw_features=None, allow_modal_grad=False, deterministic=False):
        raw_features = raw_features or self._current_raw_modal_features()
        user_id_emb = self.user_emb.weight
        modal_features = self.get_recommender_modal_features(
            raw_features=raw_features,
            allow_imputer_grad=allow_modal_grad,
        )
        observed_masks = self._missing_masks(raw_features=raw_features)
        user_outputs = {}
        item_outputs = {}
        for modality in self.modalities:
            modal_user_emb, modal_item_emb = getattr(self, f"{modality}_gcn")(
                modal_features[modality],
                user_id_emb,
                skip_mlp=self._gcn_skip_mlp(),
            )
            modal_item_emb = self._apply_item_graph_modal_residual(
                modal_item_emb, observed_masks[modality], modality=modality
            )
            user_outputs[modality] = modal_user_emb
            item_outputs[modality] = modal_item_emb

        user_emb = user_id_emb + sum(user_outputs.values()) / len(user_outputs)
        item_source = self._fuse_item_sources(
            item_outputs,
            modal_features=modal_features,
            raw_features=raw_features,
        )

        if not allow_modal_grad:
            item_source = item_source.detach()

        item_emb = self._apply_fusion(item_source, deterministic=deterministic)

        user_emb = torch.nan_to_num(user_emb, nan=0.0, posinf=0.0, neginf=0.0)
        item_emb = torch.nan_to_num(item_emb, nan=0.0, posinf=0.0, neginf=0.0)
        return user_emb, item_emb

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
        temperature=0.2,
        bank_size=256,
        allow_modal_grad=False,
    ):
        """Align imputed missing-modality GCN embeddings with observed modalities."""
        zero = torch.zeros((), device=self.env.device)
        if self.disable_imputation or item_ids is None or item_ids.numel() == 0:
            return zero
        raw_features = self._current_raw_modal_features()
        masks = self._missing_masks(raw_features)
        item_ids = torch.unique(item_ids.detach()).long()
        item_ids = item_ids[(item_ids >= 0) & (item_ids < self.m_item)]
        if item_ids.numel() < 2:
            return zero

        bank_ids = item_ids
        max_bank_size = int(bank_size or 0)
        if max_bank_size > 0 and bank_ids.numel() > max_bank_size:
            bank_ids = bank_ids[
                torch.randperm(bank_ids.numel(), device=self.env.device)[:max_bank_size]
            ]
        if bank_ids.numel() < 2:
            return zero

        item_outputs = self._recommendation_gcn_modality_item_embeddings(
            raw_features=raw_features,
            allow_modal_grad=allow_modal_grad,
            apply_item_graph=False,
        )
        temp = max(float(temperature), 1e-6)
        losses = []
        for modality in self.modalities:
            anchor_mask = ~masks[modality][item_ids]
            has_observed_other = torch.zeros_like(anchor_mask)
            for observed_modality in self.modalities:
                if observed_modality != modality:
                    has_observed_other |= masks[observed_modality][item_ids]
            anchor_ids = item_ids[anchor_mask & has_observed_other]
            if anchor_ids.numel() == 0:
                continue

            positive_parts = []
            positive_weights = []
            for observed_modality in self.modalities:
                if observed_modality == modality:
                    continue
                observed = masks[observed_modality][anchor_ids].float().unsqueeze(1)
                positive_parts.append(
                    F.normalize(item_outputs[observed_modality][anchor_ids], dim=-1)
                    * observed
                )
                positive_weights.append(observed)
            positive_sum = torch.stack(positive_parts).sum(dim=0)
            positive_count = torch.stack(positive_weights).sum(dim=0).clamp_min(1.0)
            positive_target = F.normalize(positive_sum / positive_count, dim=-1).detach()
            anchor_emb = F.normalize(item_outputs[modality][anchor_ids], dim=-1)

            negative_ids = bank_ids[masks[modality][bank_ids]]
            if negative_ids.numel() == 0:
                continue
            negative_target = F.normalize(
                item_outputs[modality][negative_ids], dim=-1
            ).detach()
            pos_logits = (anchor_emb * positive_target).sum(dim=1, keepdim=True) / temp
            neg_logits = torch.matmul(anchor_emb, negative_target.transpose(0, 1)) / temp
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
            if self._imputer_updates_enabled and stage == "imputer_param":
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
        ProMRL-style optimization path.
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
        need_rec = float(getattr(self.env.args, "alpha_rec", 0.0)) != 0.0
        need_contrastive = (
            float(getattr(self.env.args, "alpha_intra", 0.0)) != 0.0
            or float(getattr(self.env.args, "alpha_inter", 0.0)) != 0.0
        )
        need_itm = float(getattr(self.env.args, "alpha_itm", 0.0)) != 0.0
        need_decode = float(getattr(self.env.args, "alpha_decode", 0.0)) != 0.0

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
        observed_masks = self._missing_masks(raw_features=raw_features)

        outputs = {
            "user_id": user_id_emb,
            "modal_inputs": modal_features,
            "observed_masks": observed_masks,
        }
        for modality in self.modalities:
            modal_user_emb, modal_item_emb = getattr(self, f"{modality}_gcn")(
                modal_features[modality], user_id_emb, skip_mlp=self._gcn_skip_mlp()
            )
            modal_item_emb = self._apply_item_graph_modal_residual(
                modal_item_emb, observed_masks[modality], modality=modality
            )
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
        item_outputs = {
            modality: outputs[modality][1].detach()
            for modality in self.modalities
        }

        user_emb = user_id_emb + sum(user_outputs.values()) / len(user_outputs)
        item_source = self._fuse_item_sources(
            item_outputs,
            modal_features=outputs["modal_inputs"],
            observed_masks=outputs["observed_masks"],
        )
        item_emb = self.fusion_linear(item_source)

        user_emb = torch.nan_to_num(user_emb, nan=0.0, posinf=0.0, neginf=0.0)
        item_emb = torch.nan_to_num(item_emb, nan=0.0, posinf=0.0, neginf=0.0)
        self.final_user = user_emb
        self.final_item = item_emb

        return user_emb, item_emb

    def basic_recommendation_loss(
        self,
        batch_users,
        batch_pos_items,
        batch_neg_items,
        allow_modal_grad=False,
        return_embeddings=False,
    ):
        """Base recommender loss without the original I3 IRM/MI objectives."""
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
