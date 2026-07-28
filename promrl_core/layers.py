import math

import torch
import torch.nn as nn


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GELU(nn.Module):
    def forward(self, input_):
        return gelu(input_)


class Contra_head(nn.Module):
    def __init__(self, input_dim, contra_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, contra_dim, bias=False)

    def forward(self, cls_token):
        return self.linear(cls_token)


class Match_head(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.activation = GELU()
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.linear2 = nn.Linear(hidden_size, 2)

    def forward(self, cls_token):
        return self.linear2(
            self.layernorm(self.activation(self.linear1(cls_token)))
        )


def _build_mlp(input_dim, hidden_dim, output_dim, dropout=0.1):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )
