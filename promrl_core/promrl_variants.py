import itertools
import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .promrl import ProMRL
from .utils.distributed import all_gather_with_grad, concat_all_gather
from .utils.eigen import eigenvalue_computation_pmcl


ALL_MODALITIES = ("v", "a", "s", "t")


def _build_mlp(input_dim, hidden_dim, output_dim, dropout=0.1):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


def _gaussian_poe(stats):
    precisions = []
    weighted_means = []
    for mu, logvar in stats:
        precision = torch.exp(-logvar).clamp(max=1e6)
        precisions.append(precision)
        weighted_means.append(mu * precision)
    precision_sum = torch.stack(precisions, dim=0).sum(dim=0).clamp_min(1e-6)
    mu = torch.stack(weighted_means, dim=0).sum(dim=0) / precision_sum
    logvar = torch.log(precision_sum.reciprocal())
    return mu, logvar


def _kl_standard_normal(mu, logvar):
    return 0.5 * (torch.exp(logvar) + mu.pow(2) - 1.0 - logvar).mean()


def _gaussian_nll(target, mean, logvar):
    var = torch.exp(logvar).clamp_min(1e-6)
    return 0.5 * (((target - mean) ** 2) / var + logvar + math.log(2 * math.pi)).mean()


def _all_non_empty_subsets(modalities):
    subsets = []
    for size in range(1, len(modalities) + 1):
        subsets.extend(itertools.combinations(modalities, size))
    return subsets


class _GaussianEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.1):
        super().__init__()
        self.backbone = _build_mlp(input_dim, hidden_dim, hidden_dim, dropout)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.backbone(x)
        return self.mu(h), self.logvar(h).clamp(min=-8.0, max=8.0)


class _GaussianDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.decoder = _build_mlp(latent_dim, hidden_dim, output_dim, dropout)

    def forward(self, z):
        return self.decoder(z)


class RepresentationMVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, modalities, dropout=0.1, kl_weight=1.0):
        super().__init__()
        self.modalities = tuple(modalities)
        self.kl_weight = kl_weight
        self.encoders = nn.ModuleDict(
            {m: _GaussianEncoder(input_dim, hidden_dim, latent_dim, dropout) for m in self.modalities}
        )
        self.decoders = nn.ModuleDict(
            {m: _GaussianDecoder(latent_dim, hidden_dim, input_dim, dropout) for m in self.modalities}
        )
        self.decoder_logvars = nn.ParameterDict(
            {m: nn.Parameter(torch.zeros(1)) for m in self.modalities}
        )
        self.latent_dim = latent_dim

    def _prior_stats(self, batch_size, device):
        mu = torch.zeros(batch_size, self.latent_dim, device=device)
        logvar = torch.zeros(batch_size, self.latent_dim, device=device)
        return mu, logvar

    def infer_posterior(self, obs_feats, observed_modalities):
        batch_size = next(iter(obs_feats.values())).size(0)
        device = next(iter(obs_feats.values())).device
        stats = [self._prior_stats(batch_size, device)]
        for modality in observed_modalities:
            stats.append(self.encoders[modality](obs_feats[modality]))
        return _gaussian_poe(stats)

    def reconstruct_from_stats(self, mu, logvar):
        z = mu
        if self.training:
            eps = torch.randn_like(mu)
            z = mu + torch.exp(0.5 * logvar) * eps
        return {m: self.decoders[m](z) for m in self.modalities}

    def forward(self, obs_feats, observed_modalities):
        mu, logvar = self.infer_posterior(obs_feats, observed_modalities)
        recon = self.reconstruct_from_stats(mu, logvar)
        rec_losses = [
            _gaussian_nll(obs_feats[m], recon[m], self.decoder_logvars[m])
            for m in observed_modalities
        ]
        aligned_feats = {
            m: obs_feats[m] if m in observed_modalities else recon[m]
            for m in self.modalities
        }
        losses = {
            "rec_loss": sum(rec_losses) / max(len(rec_losses), 1),
            "kl_loss": self.kl_weight * _kl_standard_normal(mu, logvar),
        }
        return aligned_feats, losses


class RepresentationMoPoE(RepresentationMVAE):
    def __init__(self, input_dim, hidden_dim, latent_dim, modalities, dropout=0.1, kl_weight=1.0, max_subsets=0):
        super().__init__(input_dim, hidden_dim, latent_dim, modalities, dropout=dropout, kl_weight=kl_weight)
        self.max_subsets = max_subsets

    def forward(self, obs_feats, observed_modalities):
        batch_size = next(iter(obs_feats.values())).size(0)
        device = next(iter(obs_feats.values())).device
        subsets = _all_non_empty_subsets(observed_modalities)
        if self.max_subsets and len(subsets) > self.max_subsets:
            full_subset = tuple(observed_modalities)
            subsets = [full_subset] + subsets[: self.max_subsets - 1]

        recon_per_subset = {m: [] for m in self.modalities}
        rec_losses = []
        kl_losses = []
        for subset in subsets:
            stats = [self._prior_stats(batch_size, device)]
            for modality in subset:
                stats.append(self.encoders[modality](obs_feats[modality]))
            mu, logvar = _gaussian_poe(stats)
            recon = self.reconstruct_from_stats(mu, logvar)
            for modality in self.modalities:
                recon_per_subset[modality].append(recon[modality])
            rec_losses.append(
                sum(
                    _gaussian_nll(obs_feats[m], recon[m], self.decoder_logvars[m])
                    for m in observed_modalities
                )
                / max(len(observed_modalities), 1)
            )
            kl_losses.append(_kl_standard_normal(mu, logvar))

        mean_recon = {
            m: torch.stack(outputs, dim=0).mean(dim=0)
            for m, outputs in recon_per_subset.items()
        }
        aligned_feats = {
            m: obs_feats[m] if m in observed_modalities else mean_recon[m]
            for m in self.modalities
        }
        losses = {
            "rec_loss": torch.stack(rec_losses).mean(),
            "kl_loss": self.kl_weight * torch.stack(kl_losses).mean(),
        }
        return aligned_feats, losses


class RepresentationSMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim, modalities, dropout=0.1, mc_samples=4):
        super().__init__()
        self.modalities = tuple(modalities)
        self.hidden_dim = hidden_dim
        self.mc_samples = mc_samples
        self.modality_embeddings = nn.ParameterDict(
            {m: nn.Parameter(torch.randn(hidden_dim) * 0.02) for m in self.modalities}
        )
        self.obs_proj = nn.ModuleDict(
            {m: _build_mlp(input_dim, hidden_dim, hidden_dim, dropout) for m in self.modalities}
        )
        self.predictors = nn.ModuleDict(
            {m: _build_mlp(hidden_dim * 2, hidden_dim, input_dim * 2, dropout) for m in self.modalities}
        )

    def _context(self, obs_feats, support_modalities):
        batch_size = next(iter(obs_feats.values())).size(0)
        device = next(iter(obs_feats.values())).device
        if not support_modalities:
            return torch.zeros(batch_size, self.hidden_dim, device=device)
        tokens = []
        for modality in support_modalities:
            token = self.obs_proj[modality](obs_feats[modality])
            token = token + self.modality_embeddings[modality].unsqueeze(0)
            tokens.append(token)
        return torch.stack(tokens, dim=1).mean(dim=1)

    def _predict(self, context, target_modality):
        target_embed = self.modality_embeddings[target_modality].unsqueeze(0).expand(context.size(0), -1)
        out = self.predictors[target_modality](torch.cat([context, target_embed], dim=-1))
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar.clamp(min=-8.0, max=8.0)

    def _predict_with_mc(self, context, target_modality):
        if not self.training or self.mc_samples <= 1:
            return self._predict(context, target_modality)
        mus = []
        logvars = []
        for _ in range(self.mc_samples):
            mu, logvar = self._predict(context, target_modality)
            mus.append(mu)
            logvars.append(logvar)
        return torch.stack(mus, dim=0).mean(dim=0), torch.stack(logvars, dim=0).mean(dim=0)

    def forward(self, obs_feats, observed_modalities):
        rec_losses = []
        aligned_feats = {}
        for modality in self.modalities:
            if modality in observed_modalities:
                aligned_feats[modality] = obs_feats[modality]
            else:
                context = self._context(obs_feats, observed_modalities)
                mu, _ = self._predict_with_mc(context, modality)
                aligned_feats[modality] = mu

        for target_modality in observed_modalities:
            support = [m for m in observed_modalities if m != target_modality]
            if not support:
                continue
            context = self._context(obs_feats, support)
            mu, logvar = self._predict_with_mc(context, target_modality)
            rec_losses.append(_gaussian_nll(obs_feats[target_modality], mu, logvar))

        rec_loss = torch.stack(rec_losses).mean() if rec_losses else next(iter(obs_feats.values())).new_zeros(())
        return aligned_feats, {"rec_loss": rec_loss}


class RepresentationKnowledgeBridger(nn.Module):
    def __init__(self, input_dim, hidden_dim, modalities, dropout=0.1, num_heads=4):
        super().__init__()
        self.modalities = tuple(modalities)
        self.hidden_dim = hidden_dim
        self.modality_embeddings = nn.ParameterDict(
            {m: nn.Parameter(torch.randn(hidden_dim) * 0.02) for m in self.modalities}
        )
        self.target_queries = nn.ParameterDict(
            {m: nn.Parameter(torch.randn(hidden_dim) * 0.02) for m in self.modalities}
        )
        self.input_proj = nn.ModuleDict(
            {m: nn.Linear(input_dim, hidden_dim) for m in self.modalities}
        )
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.output_heads = nn.ModuleDict(
            {m: _build_mlp(hidden_dim, hidden_dim, input_dim, dropout) for m in self.modalities}
        )
        self.output_logvars = nn.ParameterDict(
            {m: nn.Parameter(torch.zeros(1)) for m in self.modalities}
        )

    def _predict(self, obs_feats, support_modalities, target_modality):
        batch_size = next(iter(obs_feats.values())).size(0)
        device = next(iter(obs_feats.values())).device
        if not support_modalities:
            context = self.target_queries[target_modality].unsqueeze(0).expand(batch_size, -1)
            return self.output_heads[target_modality](context)

        tokens = []
        for modality in support_modalities:
            token = self.input_proj[modality](obs_feats[modality])
            token = token + self.modality_embeddings[modality].unsqueeze(0)
            tokens.append(token)
        tokens = torch.stack(tokens, dim=1)
        query = self.target_queries[target_modality].unsqueeze(0).unsqueeze(1).expand(batch_size, 1, -1)
        attn_output, _ = self.attn(query, tokens, tokens)
        return self.output_heads[target_modality](attn_output.squeeze(1))

    def forward(self, obs_feats, observed_modalities):
        aligned_feats = {}
        rec_losses = []
        for modality in self.modalities:
            if modality in observed_modalities:
                aligned_feats[modality] = obs_feats[modality]
            else:
                aligned_feats[modality] = self._predict(obs_feats, observed_modalities, modality)

        for target_modality in observed_modalities:
            support = [m for m in observed_modalities if m != target_modality]
            if not support:
                continue
            pred = self._predict(obs_feats, support, target_modality)
            rec_losses.append(_gaussian_nll(obs_feats[target_modality], pred, self.output_logvars[target_modality]))

        rec_loss = torch.stack(rec_losses).mean() if rec_losses else next(iter(obs_feats.values())).new_zeros(())
        return aligned_feats, {"rec_loss": rec_loss}


class ProMRLCompletionAdapter(ProMRL):
    def __init__(self, config):
        super().__init__(config)
        self.completion_loss_weight = getattr(config, "completion_loss_weight", 1.0)
        self.completion_model = self.build_completion_model(config)

    def build_completion_model(self, config):
        raise NotImplementedError

    def completion_forward(self, obs_feats, observed_modalities):
        aligned_feats, losses = self.completion_model(obs_feats, observed_modalities)
        return aligned_feats, {k: v * self.completion_loss_weight for k, v in losses.items()}

    def forward_ret(self, batch, task, compute_loss=True):
        if not compute_loss:
            return super().forward_ret(batch, task, compute_loss=False)

        if isinstance(batch.raw_captions[0], list):
            batch.raw_captions = [i for j in batch.raw_captions for i in j]

        subtasks = task.split("%")[1:]
        loss_dict = {}
        loss_intra = []
        loss_inter = []
        loss_itm = []

        obs_feats = {}
        observed_modalities = []
        feat_t_obs = None

        if "vision_pixels" in batch.keys():
            obs_feats["v"] = self.batch_get(batch, "feat_v")
            observed_modalities.append("v")
        if "audio_spectrograms" in batch.keys():
            obs_feats["a"] = self.batch_get(batch, "feat_a")
            observed_modalities.append("a")
        if "raw_subtitles" in batch.keys():
            obs_feats["s"] = self.batch_get(batch, "feat_s")
            observed_modalities.append("s")
        if "raw_captions" in batch.keys():
            feat_t_obs = self.batch_get(batch, "feat_t")
            obs_feats["t"] = feat_t_obs
            observed_modalities.append("t")
            feat_t_all = concat_all_gather(feat_t_obs)
            caption_tokens = self.batch_get(batch, "caption_tokens")
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            input_ids_collate = concat_all_gather(input_ids)
            attention_mask_collate = concat_all_gather(attention_mask)

        completed_feats, completion_losses = self.completion_forward(obs_feats, observed_modalities)
        loss_dict.update(completion_losses)

        feat_t = completed_feats["t"]
        feat_v = completed_feats["v"]
        feat_a = completed_feats["a"]

        for subtask in subtasks:
            assert subtask in ["tv", "ta", "tva", "tvs", "tvas", "va"]

            eigenvectors, eigenvalues = eigenvalue_computation_pmcl([feat_t, feat_v, feat_a])
            eigenvalues = eigenvalues / self.tau1
            loss_intra_ = F.cross_entropy(
                eigenvalues,
                torch.zeros(eigenvalues.shape[0], device=eigenvalues.device, dtype=torch.long),
            )
            principal_eigenvector = eigenvectors[:, :, 0]

            principal_eigenvector_all = concat_all_gather(principal_eigenvector.contiguous())
            rank = dist.get_rank()
            bs = principal_eigenvector.size(0)
            targets = torch.linspace(
                rank * bs,
                rank * bs + bs - 1,
                bs,
                dtype=torch.int64,
                device=principal_eigenvector.device,
            )
            loss_inter_ = F.cross_entropy(
                (principal_eigenvector @ principal_eigenvector_all.T) / self.tau2,
                targets,
            )

            loss_intra.append(loss_intra_)
            loss_inter.append(loss_inter_)

            if "t" in subtask and feat_t_obs is not None:
                feat_cond = self.batch_get(batch, f"feat_{subtask[1:]}")
                feat_cond_all = concat_all_gather(feat_cond)
                sim_cond2t = torch.matmul(feat_cond, feat_t_all.permute(1, 0)) / self.contra_temp
                sim_t2cond = torch.matmul(feat_t_obs, feat_cond_all.permute(1, 0)) / self.contra_temp
                rank = dist.get_rank()
                bs = feat_t_obs.size(0)

                condition_feats = self.batch_get(batch, f"condition_feats_{subtask[1:]}")
                condition_feats_collate = all_gather_with_grad(condition_feats)
                with torch.no_grad():
                    weights_t2cond = F.softmax(sim_t2cond, dim=1) + 1e-4
                    weights_t2cond[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
                    weights_cond2t = F.softmax(sim_cond2t, dim=1) + 1e-4
                    weights_cond2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

                condition_feats_neg = []
                for b in range(bs):
                    neg_idx = torch.multinomial(weights_t2cond[b], 1).item()
                    condition_feats_neg.append(condition_feats_collate[neg_idx])
                condition_feats_neg = torch.stack(condition_feats_neg, dim=0)

                text_ids_neg = []
                text_atts_neg = []
                for b in range(bs):
                    neg_idx = torch.multinomial(weights_cond2t[b], 1).item()
                    text_ids_neg.append(input_ids_collate[neg_idx])
                    text_atts_neg.append(attention_mask_collate[neg_idx])

                text_ids_neg = torch.stack(text_ids_neg, dim=0)
                text_atts_neg = torch.stack(text_atts_neg, dim=0)

                input_ids_1 = torch.cat((input_ids, input_ids, text_ids_neg), dim=0)
                attention_mask_1 = torch.cat((attention_mask, attention_mask, text_atts_neg), dim=0)
                condition_feats = torch.cat((condition_feats, condition_feats_neg, condition_feats), dim=0).detach()
                output = self.multimodal_encoder.bert(
                    input_ids=input_ids_1,
                    attention_mask=attention_mask_1,
                    encoder_hidden_states=condition_feats,
                ).last_hidden_state
                batch_size = condition_feats_neg.shape[0]
                logits = self.itm_head(output[:, 0].half())
                ground_truth = torch.zeros(batch_size * 3, dtype=torch.long, device=logits.device)
                ground_truth[:batch_size] = 1
                loss = F.cross_entropy(logits, ground_truth)
                loss_itm.append(self.lambda_itm * loss)
            else:
                loss_itm.append(feat_t.new_zeros(()))

        loss_dict["loss_intra"] = sum(loss_intra) / len(loss_intra)
        loss_dict["loss_inter"] = sum(loss_inter) / len(loss_inter)
        loss_dict["loss_itm"] = sum(loss_itm) / len(loss_itm)
        return loss_dict


class ProMRL_MVAE(ProMRLCompletionAdapter):
    def build_completion_model(self, config):
        return RepresentationMVAE(
            input_dim=config.contra_dim,
            hidden_dim=getattr(config, "completion_hidden_dim", 1024),
            latent_dim=getattr(config, "completion_latent_dim", config.contra_dim // 2),
            modalities=ALL_MODALITIES,
            dropout=getattr(config, "completion_dropout", 0.1),
            kl_weight=getattr(config, "completion_kl_weight", 1.0),
        )


class ProMRL_MoPoE(ProMRLCompletionAdapter):
    def build_completion_model(self, config):
        return RepresentationMoPoE(
            input_dim=config.contra_dim,
            hidden_dim=getattr(config, "completion_hidden_dim", 1024),
            latent_dim=getattr(config, "completion_latent_dim", config.contra_dim // 2),
            modalities=ALL_MODALITIES,
            dropout=getattr(config, "completion_dropout", 0.1),
            kl_weight=getattr(config, "completion_kl_weight", 1.0),
            max_subsets=getattr(config, "mopoe_max_subsets", 0),
        )


class ProMRL_SMIL(ProMRLCompletionAdapter):
    def build_completion_model(self, config):
        return RepresentationSMIL(
            input_dim=config.contra_dim,
            hidden_dim=getattr(config, "completion_hidden_dim", 1024),
            modalities=ALL_MODALITIES,
            dropout=getattr(config, "completion_dropout", 0.1),
            mc_samples=getattr(config, "smil_mc_samples", 4),
        )


class ProMRL_KnowledgeBridger(ProMRLCompletionAdapter):
    def build_completion_model(self, config):
        return RepresentationKnowledgeBridger(
            input_dim=config.contra_dim,
            hidden_dim=getattr(config, "completion_hidden_dim", 1024),
            modalities=ALL_MODALITIES,
            dropout=getattr(config, "completion_dropout", 0.1),
            num_heads=getattr(config, "kb_num_heads", 4),
        )
