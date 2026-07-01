# Copyright 2024 Wei Sun, Xiaoqing Pan, Yifan Yang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Neural-network building blocks and the ZINB distribution."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import NegativeBinomial, Bernoulli, Distribution


class ResidualBlock(nn.Module):
    """Residual block with optional dimension adaptation."""

    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float = 0.2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )
        self.shortcut = nn.Sequential()
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
            )
        self.final_activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.block(x)
        out += self.shortcut(identity)
        return self.final_activation(out)


class SelfAttention(nn.Module):
    """Lightweight multi-head self-attention for high-dimensional biological data."""

    def __init__(self, feature_dim: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if feature_dim % heads != 0:
            raise ValueError("feature_dim must be divisible by heads")

        self.heads = heads
        self.head_dim = feature_dim // heads

        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.fc_out = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        residual = x

        Q = self.query(x).view(batch_size, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(x).view(batch_size, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(x).view(batch_size, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        attention = F.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        out = torch.matmul(attention, V)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, -1, self.heads * self.head_dim).squeeze(1)

        out = self.fc_out(out)
        out = self.dropout(out)
        out += residual
        out = self.layer_norm(out)
        return out


class ZeroInflatedNegativeBinomial(Distribution):
    """Zero-inflated negative binomial (ZINB) distribution.

    Parameters
    ----------
    mu
        Mean parameter.
    theta
        Inverse dispersion (total_count).
    zi_logits
        Logits of the zero-inflation probability.
    """

    def __init__(
        self,
        mu: torch.Tensor,
        theta: torch.Tensor,
        zi_logits: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        super().__init__(validate_args=validate_args)
        self.mu, self.theta, self.zi_logits = torch.broadcast_tensors(mu, theta, zi_logits)
        self.nb = NegativeBinomial(
            total_count=self.theta,
            logits=(self.mu + 1e-8).log() - (self.theta + 1e-8).log(),
        )
        self.zi = Bernoulli(logits=self.zi_logits)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        log_nb = self.nb.log_prob(value)
        log_zi = -F.softplus(-self.zi_logits)
        log_not_zi = -F.softplus(self.zi_logits)

        return torch.where(
            torch.abs(value) < 1e-5,
            torch.logsumexp(torch.stack([log_zi, log_not_zi + log_nb]), dim=0),
            log_not_zi + log_nb,
        )

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        with torch.no_grad():
            samp = self.nb.sample(sample_shape)
            mask = torch.rand_like(samp.float()) <= self.zi.probs
            samp[mask] = 0
            return samp


class ConditionalAttention(nn.Module):
    """Cross-attention layer that fuses latent variables with condition vectors."""

    def __init__(
        self,
        latent_dim: int,
        batch_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.batch_dim = batch_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads

        self.q = nn.Linear(latent_dim, latent_dim)
        self.k = nn.Linear(batch_dim, latent_dim)
        self.v = nn.Linear(batch_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, z: torch.Tensor, batch_labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        z
            Latent variables, shape ``(batch_size, latent_dim)``.
        batch_labels
            Condition vectors, shape ``(batch_size, batch_dim)``.

        Returns
        -------
        output, attn_weights
            Fused latent representation and attention weights.
        """
        batch_size = z.size(0)

        Q = self.q(z).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k(batch_labels).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v(batch_labels).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.latent_dim)
        output = self.out_proj(attn_output.squeeze(1))
        return z + output, attn_weights
