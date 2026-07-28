import torch
import numpy as np
import torch.nn.functional as F

def eigenvalue_computation_pmcl(modalities):

    stacked_reps = torch.stack(modalities, dim=-1) # [batch_size, dim, num_reps]

    # eigvals, _ = torch.linalg.eigh(G.float()) # [batch_size, num_reps]
    U_V, S_V, W_V = torch.linalg.svd(stacked_reps, full_matrices=True)

    return U_V, S_V


def shifted_relation_lifted_directions(modalities):
    """Return descending relation eigenvalues and lifted principal directions.

    Each modality is normalized before constructing the shifted-cosine relation
    matrix R = (Z^T Z + 1) / 2.  The leading relation eigenvector is then lifted
    from modality-index space back into the shared representation space.
    """
    normalized = [F.normalize(feature, p=2, dim=-1) for feature in modalities]
    stacked_reps = torch.stack(normalized, dim=-1)
    gram = stacked_reps.transpose(-1, -2) @ stacked_reps
    relation = 0.5 * (gram + torch.ones_like(gram))

    eigenvalues_ascending, eigenvectors = torch.linalg.eigh(relation)
    eigenvalues = eigenvalues_ascending.flip(dims=(-1,))
    principal_eigenvector = eigenvectors[:, :, -1]

    lifted = (stacked_reps @ principal_eigenvector.unsqueeze(-1)).squeeze(-1)
    lifted = F.normalize(lifted, p=2, dim=-1)
    return eigenvalues, lifted
