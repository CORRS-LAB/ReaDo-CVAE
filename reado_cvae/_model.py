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

"""Core ReaDo-CVAE model architecture."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._modules import ResidualBlock, SelfAttention, ConditionalAttention, ZeroInflatedNegativeBinomial


class ReaDoCVAEModel(nn.Module):
    """Conditional variational auto-encoder for scRNA-seq doublet synthesis."""

    def __init__(
        self,
        gene_dim: int,
        batch_dim: int,
        latent_dim: int = 100,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.1,
        num_heads: int = 4,
    ) -> None:
        """Initialize the ReaDo-CVAE model.

        Parameters
        ----------
        gene_dim
            Number of genes (input/output dimensionality).
        batch_dim
            Dimensionality of the condition vector (e.g. one-hot cell types).
        latent_dim
            Dimensionality of the latent space.
        hidden_dims
            Encoder hidden-layer dimensions.
        dropout_rate
            Dropout probability.
        num_heads
            Number of attention heads in conditional and self-attention layers.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512, 512, 256, 256, 128]

        self.gene_dim = gene_dim
        self.batch_dim = batch_dim
        self.latent_dim = latent_dim

        self.condition_attention = ConditionalAttention(
            latent_dim=latent_dim,
            batch_dim=batch_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
        )

        # Encoder
        encoder_layers = []
        prev_dim = gene_dim
        for i, h_dim in enumerate(hidden_dims):
            if i > 0 and hidden_dims[i - 1] != h_dim:
                encoder_layers.append(ResidualBlock(prev_dim, h_dim, dropout_rate))
            else:
                encoder_layers.append(nn.Linear(prev_dim, h_dim))
                encoder_layers.append(nn.BatchNorm1d(h_dim))
                encoder_layers.append(nn.LeakyReLU(0.2))
                encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
            if i >= len(hidden_dims) - 1:
                encoder_layers.append(SelfAttention(h_dim, heads=num_heads, dropout=dropout_rate))

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Library-size estimator
        self.library_encoder = nn.Sequential(
            nn.Linear(gene_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus(),
        )

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim + batch_dim
        for i, h_dim in enumerate(reversed(hidden_dims)):
            if i > 0 and hidden_dims[len(hidden_dims) - i] != h_dim:
                decoder_layers.append(ResidualBlock(prev_dim, h_dim, dropout_rate))
            else:
                decoder_layers.append(nn.Linear(prev_dim, h_dim))
                decoder_layers.append(nn.BatchNorm1d(h_dim))
                decoder_layers.append(nn.LeakyReLU(0.2))
                decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
            if i < 1:
                decoder_layers.append(SelfAttention(h_dim, heads=num_heads, dropout=dropout_rate))

        self.decoder = nn.Sequential(*decoder_layers)

        # Output heads
        self.fc_mean = nn.Linear(hidden_dims[0], gene_dim)
        self.fc_disp = nn.Linear(hidden_dims[0], gene_dim)
        self.fc_zero = nn.Linear(hidden_dims[0], gene_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode gene-expression data into latent parameters.

        Returns
        -------
        mu, logvar, library
            Mean and log-variance of the latent distribution, plus estimated
            library size.
        """
        library = self.library_encoder(x)
        x_log = torch.log1p(x)
        h = self.encoder(x_log)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar, library

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterisation trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self,
        z: torch.Tensor,
        batch_labels: torch.Tensor,
        library: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode latent variables into gene-expression parameters.

        Returns
        -------
        mean, dispersion, zero_logits
            Parameters for the ZINB likelihood.
        """
        z_cond, _ = self.condition_attention(z, batch_labels)
        z_combined = torch.cat([z_cond, batch_labels], dim=1)
        h = self.decoder(z_combined)

        logits = self.fc_mean(h)
        dispersion = self.fc_disp(h)
        zero_logits = self.fc_zero(h)

        mean = library * F.softmax(logits, dim=-1)
        return mean, torch.exp(dispersion), zero_logits

    def forward(
        self,
        x: torch.Tensor,
        batch_labels: torch.Tensor,
        beta: float,
    ) -> tuple[torch.Tensor, ...]:
        """Forward pass.

        In training mode returns the ZINB parameters plus the decomposed
        loss (total, reconstruction, KL). In eval mode returns only the
        ZINB parameters.
        """
        mu, logvar, library = self.encode(x)
        z = self.reparameterize(mu, logvar)
        mean, dispersion, zero_logits = self.decode(z, batch_labels, library)

        if self.training:
            loss, recon_loss, kl_loss = self.loss_function(
                mean, dispersion, zero_logits, x, mu, logvar, beta
            )
            return mean, dispersion, zero_logits, loss, recon_loss, kl_loss

        return mean, dispersion, zero_logits

    def sample(
        self,
        batch_labels: torch.Tensor,
        n_samples: int = 1,
        library: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample from the prior and generate synthetic data.

        Parameters
        ----------
        batch_labels
            Condition vectors (e.g. one-hot cell types).
        n_samples
            Number of samples to draw. Must match ``batch_labels.size(0)``.
        library
            Optional library-size estimates.

        Returns
        -------
        torch.Tensor
            Generated gene-expression matrix.
        """
        z = torch.randn(n_samples, self.latent_dim, device=batch_labels.device)
        mean, dispersion, zero_logits = self.decode(z, batch_labels, library)
        zinb = ZeroInflatedNegativeBinomial(
            mu=mean,
            theta=dispersion,
            zi_logits=zero_logits,
        )
        return zinb.sample()

    def loss_function(
        self,
        mean: torch.Tensor,
        dispersion: torch.Tensor,
        zero_logits: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the ELBO loss.

        Returns
        -------
        total_loss, recon_loss, kl_div
        """
        zinb = ZeroInflatedNegativeBinomial(
            mu=mean,
            theta=dispersion,
            zi_logits=zero_logits,
        )
        recon_loss = -zinb.log_prob(x).sum()
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kl_div
        return total_loss, recon_loss, kl_div
