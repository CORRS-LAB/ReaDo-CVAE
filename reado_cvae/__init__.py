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

"""ReaDo-CVAE: conditional variational autoencoder for doublet synthesis."""

from __future__ import annotations

import torch
import numpy as np
import anndata
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ._model import ReaDoCVAEModel
from .active_learning import ActiveLearner

__version__ = "0.1.0"
__all__ = ["ReaDoCVAE", "ActiveLearner"]


def _resolve_device(device: str) -> str:
    """Resolve 'auto' to the best available accelerator."""
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ReaDoCVAE:
    """High-level interface for training and sampling with ReaDo-CVAE."""

    def __init__(
        self,
        adata: anndata.AnnData,
        cell_type_key: str = "cell_type",
        n_latent: int = 100,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.1,
        beta: float = 1.0,
        device: str = "auto",
    ) -> None:
        """Initialize ReaDoCVAE.

        Parameters
        ----------
        adata
            Preprocessed AnnData containing highly variable genes.
            ``obs`` must include the cell-type column.
        cell_type_key
            Column name in ``adata.obs`` that stores cell types.
        n_latent
            Dimensionality of the latent space.
        hidden_dims
            Hidden-layer dimensions for the encoder/decoder.
        dropout_rate
            Dropout probability.
        beta
            Weight for the KL-divergence term.
        device
            Compute device. ``"auto"`` selects ``mps`` > ``cuda`` > ``cpu``.
        """
        if hidden_dims is None:
            hidden_dims = [1024, 512, 512, 256, 256, 128]

        self.adata = adata
        self.cell_type_key = cell_type_key
        self.device = _resolve_device(device)
        self.beta = beta

        # Extract expression matrix and labels
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
        self.raw_X = X.astype(np.float32)

        self.le = LabelEncoder()
        y = self.le.fit_transform(adata.obs[cell_type_key])
        self.n_types = len(self.le.classes_)
        self.gene_dim = X.shape[1]

        # Compute per-cell-type library-size statistics for generation
        self._library_stats: dict[str, tuple[float, float]] = {}
        for ct in self.le.classes_:
            mask = adata.obs[cell_type_key] == ct
            counts = X[mask].sum(axis=1)
            key = str(ct)
            if len(counts) == 0:
                self._library_stats[key] = (7.0, 0.5)  # fallback default
            else:
                log_counts = np.log(np.maximum(counts, 1e-6))
                self._library_stats[key] = (
                    float(np.mean(log_counts)),
                    float(np.std(log_counts) + 1e-6),
                )

        self._X_tensor = torch.tensor(self.raw_X, dtype=torch.float32)
        self._onehot = torch.eye(self.n_types)[torch.tensor(y, dtype=torch.long)].float()

        self.model = ReaDoCVAEModel(
            gene_dim=self.gene_dim,
            batch_dim=self.n_types,
            latent_dim=n_latent,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
        ).to(self.device)

    def _get_library(self, cell_types: list[str], n_cells: int) -> torch.Tensor:
        """Sample library sizes [n_cells, 1] from log-normal distributions."""
        libraries = []
        for ct in cell_types:
            key = str(ct)
            if key not in self._library_stats:
                raise KeyError(f"Unknown cell type {ct!r}; known types: {list(self._library_stats.keys())}")
            mean, std = self._library_stats[key]
            lib = torch.distributions.LogNormal(mean, std).sample((n_cells, 1))
            libraries.append(lib)
        return torch.cat(libraries, dim=0).to(self.device)

    def _resolve_cell_type(self, cell_type: int | str) -> str:
        """Resolve a user-provided cell type to a value accepted by the label encoder."""
        candidates = [cell_type]
        if isinstance(cell_type, str):
            try:
                if float(cell_type).is_integer():
                    candidates.append(int(float(cell_type)))
                candidates.append(float(cell_type))
            except ValueError:
                pass
        else:
            candidates.append(str(cell_type))
            candidates.append(str(int(cell_type)) if float(cell_type).is_integer() else str(cell_type))
        for cand in candidates:
            try:
                self.le.transform([cand])
                return str(cand) if isinstance(cand, (int, float, np.floating, np.integer)) else cand
            except ValueError:
                continue
        raise ValueError(
            f"Cell type {cell_type!r} not recognized. Known types: {list(self.le.classes_)}"
        )

    def train(
        self,
        epochs: int = 200,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        verbose: bool = True,
    ) -> None:
        """Train the model.

        Parameters
        ----------
        epochs
            Number of training epochs.
        batch_size
            Mini-batch size.
        lr
            Learning rate.
        weight_decay
            AdamW weight decay.
        verbose
            Whether to display a tqdm progress bar.
        """
        dataset = TensorDataset(self._X_tensor, self._onehot)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.model.train()
        pbar = tqdm(range(epochs), desc="Training", disable=not verbose)
        for _ in pbar:
            total_l = total_r = total_k = 0.0
            for x, c in loader:
                x, c = x.to(self.device), c.to(self.device)
                opt.zero_grad()
                _, _, _, loss, recon, kl = self.model(x, c, self.beta)
                loss.backward()
                opt.step()
                total_l += loss.item()
                total_r += recon.item()
                total_k += kl.item()
            n_batches = len(loader)
            pbar.set_postfix(
                {
                    "loss": f"{total_l / n_batches:.2f}",
                    "recon": f"{total_r / n_batches:.2f}",
                    "KL": f"{total_k / n_batches:.2f}",
                }
            )

    def sample(
        self,
        cell_type: int | str | None = None,
        n_cells: int | None = None,
    ) -> anndata.AnnData:
        """Generate synthetic doublets.

        Parameters
        ----------
        cell_type
            Specific cell type to generate. If ``None`` (default), all cell
            types are generated with counts matching the original data.
        n_cells
            Number of cells to generate when ``cell_type`` is specified.

        Returns
        -------
        anndata.AnnData
            Synthetic doublets with matching ``var`` annotations.
        """
        self.model.eval()
        with torch.no_grad():
            if cell_type is None and n_cells is None:
                counts = self.adata.obs[self.cell_type_key].value_counts()
                all_samples: list[np.ndarray] = []
                all_types: list[str | int] = []
                for ct, n in counts.items():
                    ct_key = self._resolve_cell_type(ct)
                    c_idx = self.le.transform([ct_key])[0]
                    onehot = torch.zeros(n, self.n_types, device=self.device)
                    onehot[:, c_idx] = 1
                    library = self._get_library([ct_key], n)
                    samp = self.model.sample(onehot, n_samples=n, library=library).cpu().numpy()
                    all_samples.append(samp)
                    all_types.extend([str(ct)] * n)
                X_syn = np.concatenate(all_samples)
                obs_syn = pd.DataFrame({self.cell_type_key: all_types})
            elif cell_type is not None and n_cells is not None:
                # Handle string/float cell-type key mismatch robustly
                ct_key = self._resolve_cell_type(cell_type)
                c_idx = self.le.transform([ct_key])[0]
                onehot = torch.zeros(n_cells, self.n_types, device=self.device)
                onehot[:, c_idx] = 1
                library = self._get_library([ct_key], n_cells)
                X_syn = self.model.sample(onehot, n_samples=n_cells, library=library).cpu().numpy()
                obs_syn = pd.DataFrame({self.cell_type_key: [str(cell_type)] * n_cells})
            else:
                raise ValueError(
                    "Provide both cell_type and n_cells, or leave both as None."
                )
        return anndata.AnnData(X=X_syn, obs=obs_syn, var=self.adata.var)

    def save(self, path: str) -> None:
        """Serialize model state to ``path``."""
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "label_encoder": self.le,
                "gene_dim": self.gene_dim,
                "n_types": self.n_types,
                "library_stats": self._library_stats,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        adata: anndata.AnnData,
        device: str = "auto",
    ) -> ReaDoCVAE:
        """Load a saved model.

        Parameters
        ----------
        path
            File path written by :meth:`save`.
        adata
            AnnData used to re-instantiate the wrapper (must match original
            gene order and cell-type key).
        device
            Target compute device.

        Returns
        -------
        ReaDoCVAE
            Restored model instance.
        """
        device = _resolve_device(device)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(adata, device=device)
        model.model.load_state_dict(checkpoint["model_state"])
        model.le = checkpoint["label_encoder"]
        model._library_stats = checkpoint.get("library_stats", model._library_stats)
        return model
