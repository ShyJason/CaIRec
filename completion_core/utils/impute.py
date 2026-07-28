
import torch
import torch.nn as nn
import math


def update_posterior(Z_obs, W_dict, mu_dict, log_sigma_dict, modality_keys, d_beta):
    N = next(iter(Z_obs.values())).size(0)
    device = next(iter(Z_obs.values())).device

    V_inv = torch.eye(d_beta, device=device)
    for m in modality_keys:
        if Z_obs[m] is not None:
            W = W_dict[m]
            sigma2 = torch.exp(2 * log_sigma_dict[m])  # sigma = exp(log_sigma)
            V_inv += (1.0 / sigma2) * (W.T @ W)
    V = torch.inverse(V_inv + 1e-6 * torch.eye(d_beta, device=device))

    m = torch.zeros(N, d_beta, device=device)
    for m_key in modality_keys:
        if Z_obs[m_key] is not None:
            z = Z_obs[m_key]
            W = W_dict[m_key]
            mu_val = mu_dict[m_key]
            sigma2 = torch.exp(2 * log_sigma_dict[m_key])
            residual = z - mu_val.unsqueeze(0)
            m += (1.0 / sigma2) * (residual @ W)
    m = m @ V.T
    return m, V

def update_generative_params(Z_obs, m, V, W_dict, mu_dict, sigma_dict, modality_keys, d):
    """
    Update W^m, mu^m, sigma^m for each modality (only using observed data).
    """
    N = m.size(0)
    device = m.device
    W_dict_new = {}
    mu_dict_new = {}
    sigma_dict_new = {}

    E_beta_betaT = (m.T @ m) / N + V  # [d_beta, d_beta], approx average

    for m_key in modality_keys:
        if Z_obs[m_key] is not None:
            z = Z_obs[m_key]  # [N, d]
            # mu^m = mean(z - W m)
            mu = z.mean(dim=0) - (W_dict[m_key] @ m.T).mean(dim=1)  # rough init; better to iterate
            # But for closed-form update:
            # We'll compute W^m = (sum (z - mu) m^T) (sum E[beta beta^T])^{-1}
            z_centered = z - mu.unsqueeze(0)  # [N, d]
            numerator = z_centered.T @ m  # [d, d_beta]
            W = numerator @ torch.inverse(E_beta_betaT)  # [d, d_beta]

            # sigma^2
            recon_error = z_centered - m @ W.T  # [N, d]
            recon_sq = (recon_error ** 2).sum()
            trace_term = torch.trace(W.T @ W @ V) * N
            sigma2 = (recon_sq + trace_term) / (N * d)
            sigma = torch.sqrt(torch.clamp(sigma2, min=1e-6))

            W_dict_new[m_key] = W.detach()
            mu_dict_new[m_key] = mu.detach()
            sigma_dict_new[m_key] = sigma.detach()
        else:
            # Keep old or init (not updated)
            W_dict_new[m_key] = W_dict[m_key]
            mu_dict_new[m_key] = mu_dict[m_key]
            sigma_dict_new[m_key] = sigma_dict[m_key]

    return W_dict_new, mu_dict_new, sigma_dict_new

def impute_missing(Z_obs, m, W_dict, mu_dict, all_modalities):
    Z_full = {}
    for m_key in all_modalities:
        if Z_obs[m_key] is not None:
            Z_full[m_key] = Z_obs[m_key]
        else:
            # z^m' = W^{m'} m + mu^{m'}
            Z_full[m_key] = m @ W_dict[m_key].T + mu_dict[m_key].unsqueeze(0)
    return Z_full

def compute_nll_loss(Z_obs, m, W_dict, mu_dict, log_sigma_dict, modality_keys):
    total_loss = 0.0
    loss_dict = {}
    for k in modality_keys:
        if Z_obs[k] is not None:
            z = Z_obs[k]  # [N, d]
            recon_mean = m @ W_dict[k].T + mu_dict[k].unsqueeze(0)  # [N, d]
            sigma = torch.exp(log_sigma_dict[k])  # scalar
            # NLL for Gaussian: (z - recon)^2 / (2 sigma^2) + 0.5 * log(2 pi sigma^2)
            recon_error = (z - recon_mean) ** 2
            nll = recon_error / (2 * sigma**2) + 0.5 * torch.log(2 * math.pi * sigma**2)
            loss = nll.mean()  # scalar
            loss_dict[k] = loss.item()
            total_loss += loss
    return total_loss, loss_dict
