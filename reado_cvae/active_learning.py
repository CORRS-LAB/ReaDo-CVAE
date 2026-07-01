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

"""Active-learning loop for iterative improvement of ReaDo-CVAE synthesis quality."""

from __future__ import annotations

import json
import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class ActiveLearner:
    """Orchestrate train → validate → acquire rounds for ReaDo-CVAE.

    Parameters
    ----------
    adata
        Full real AnnData (will be split internally if *train_adata* is not
        provided).
    train_adata
        Initial training pool. If ``None``, created by stratified split.
    val_adata
        Initial validation pool. If ``None``, created by stratified split.
    cell_type_key
        Column in ``obs`` holding cell-type labels.
    model_config
        Dict passed to ``ReaDoCVAE`` (n_latent, hidden_dims, dropout_rate,
        beta, device, ...).
    acquisition_budget
        Max number of cells to move from val → train per round.
    strategy
        Acquisition heuristic: ``"metric_gap"``, ``"uncertainty"``, or
        ``"random"``.
    random_state
        Seed for splits and sampling.
    results_dir
        Directory where round JSON logs and model checkpoints are written.
    """

    def __init__(
        self,
        adata: anndata.AnnData | None = None,
        train_adata: anndata.AnnData | None = None,
        val_adata: anndata.AnnData | None = None,
        cell_type_key: str = "cell_type",
        model_config: dict[str, Any] | None = None,
        acquisition_budget: int = 100,
        strategy: str = "metric_gap",
        random_state: int = 42,
        results_dir: str | Path = "activeLearning/results",
    ) -> None:
        if (train_adata is None or val_adata is None) and adata is None:
            raise ValueError("Provide either adata or both train_adata and val_adata.")

        self.cell_type_key = cell_type_key
        self.model_config = model_config or {}
        self.budget = acquisition_budget
        self.strategy = strategy
        self.random_state = random_state
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if train_adata is None or val_adata is None:
            self.train_adata, self.val_adata = self._stratified_split(adata)
        else:
            self.train_adata = train_adata.copy()
            self.val_adata = val_adata.copy()

        self.model: ReaDoCVAE | None = None
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Data management
    # ------------------------------------------------------------------
    def _stratified_split(
        self, adata: anndata.AnnData, train_size: float = 0.7
    ) -> tuple[anndata.AnnData, anndata.AnnData]:
        """Stratified split preserving cell-type proportions."""
        obs = adata.obs.copy()
        idx = np.arange(len(obs))
        y = obs[self.cell_type_key].values

        train_idx, val_idx = train_test_split(
            idx, train_size=train_size, stratify=y, random_state=self.random_state
        )
        return adata[train_idx].copy(), adata[val_idx].copy()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, epochs: int = 100, lr: float = 1e-5, batch_size: int = 256, warm_start_path: str | None = None) -> None:
        """Train (or fine-tune) the cVAE on the current training pool."""
        from reado_cvae import ReaDoCVAE

        cfg = deepcopy(self.model_config)
        cfg.setdefault("cell_type_key", self.cell_type_key)

        if warm_start_path is not None and Path(warm_start_path).exists():
            self.model = ReaDoCVAE.load(warm_start_path, self.train_adata, device=cfg.get("device", "auto"))
            print(f"[ActiveLearner] Warm-started from {warm_start_path}")
        else:
            self.model = ReaDoCVAE(self.train_adata, **cfg)

        self.model.train(epochs=epochs, batch_size=batch_size, lr=lr)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self, n_cells_per_type: int | None = None) -> dict[str, Any]:
        """Validate synthesis quality on every cell type in the validation set.

        Returns a dict with keys ``aggregate``, ``per_type``, ``recon_loss``.
        """
        if self.model is None:
            raise RuntimeError("Call train() before validate().")

        from evaluation.metrics import (
            calculate_centroid_distances,
            calculate_gcp,
            calculate_scc,
            calculate_MMD_paper,
            calculate_milisi,
        )

        val_types = self.val_adata.obs[self.cell_type_key].unique()
        per_type: dict[str, dict[str, float]] = {}
        recon_losses: dict[str, float] = {}

        for ct in val_types:
            real_subset = self.val_adata[self.val_adata.obs[self.cell_type_key] == ct]
            real_X = real_subset.X
            if hasattr(real_X, "toarray"):
                real_X = real_X.toarray()
            real_X = np.asarray(real_X, dtype=np.float32)

            n = n_cells_per_type or real_X.shape[0]
            gen_X = self.model.sample(cell_type=ct, n_cells=n).X

            # Ensure dense
            if hasattr(gen_X, "toarray"):
                gen_X = gen_X.toarray()
            gen_X = np.asarray(gen_X, dtype=np.float32)

            # Sub-sample real to match generated count for fair comparison
            if real_X.shape[0] > gen_X.shape[0]:
                rng = np.random.default_rng(self.random_state)
                idx = rng.choice(real_X.shape[0], gen_X.shape[0], replace=False)
                real_X = real_X[idx]
            elif real_X.shape[0] < gen_X.shape[0]:
                gen_X = gen_X[: real_X.shape[0]]

            # Compute metrics
            real_log = np.log1p(real_X)
            gen_log = np.log1p(gen_X)
            scaler = StandardScaler(with_mean=False, with_std=True)
            real_scaled = scaler.fit_transform(real_log)
            gen_scaled = scaler.transform(gen_log)

            _, cos_dist = calculate_centroid_distances(real_scaled, gen_scaled)
            mmd = calculate_MMD_paper(real_scaled, gen_scaled, n_samples=min(500, real_X.shape[0]))

            # miLISI on scaled log data
            scaler_pca = StandardScaler(with_mean=True, with_std=True)
            combined = np.vstack([scaler_pca.fit_transform(real_log), scaler_pca.transform(gen_log)])
            pca = PCA(n_components=min(50, combined.shape[0], combined.shape[1]), random_state=self.random_state)
            combined_pca = pca.fit_transform(combined)
            labels = np.concatenate([np.zeros(real_X.shape[0]), np.ones(gen_X.shape[0])])
            milisi = calculate_milisi(combined_pca, labels, n_neighbors=min(90, real_X.shape[0] // 2))

            gcp = calculate_gcp(real_scaled, gen_scaled)
            scc = calculate_scc(real_scaled, gen_scaled)

            # Simple RF AUC
            roc_auc = self._quick_rf_auc(real_X, gen_X)

            per_type[str(ct)] = {
                "cosine_distance": float(cos_dist),
                "mmd": float(mmd),
                "milisi": float(milisi),
                "roc_auc": float(roc_auc),
                "gcp": float(gcp),
                "scc": float(scc),
            }

            # Reconstruction loss (uncertainty proxy)
            recon_losses[str(ct)] = self._recon_loss_for_type(ct, real_X)

        # Aggregate means
        df = pd.DataFrame(per_type).T
        aggregate = {col: float(df[col].mean()) for col in df.columns}

        return {
            "aggregate": aggregate,
            "per_type": per_type,
            "recon_loss": recon_losses,
        }

    def _quick_rf_auc(self, real_X: np.ndarray, gen_X: np.ndarray, random_state: int = 42) -> float:
        """Train a quick RF classifier to distinguish real vs. generated."""
        X = np.vstack([real_X, gen_X])
        y = np.concatenate([np.zeros(real_X.shape[0]), np.ones(gen_X.shape[0])])
        if len(np.unique(y)) < 2:
            return 0.5

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        n_comp = min(50, X_train_s.shape[0], X_train_s.shape[1])
        pca = PCA(n_components=n_comp, random_state=random_state)
        X_train_p = pca.fit_transform(X_train_s)
        X_test_p = pca.transform(X_test_s)
        rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=1)
        rf.fit(X_train_p, y_train)
        y_pred = rf.predict_proba(X_test_p)[:, 1]
        return float(roc_auc_score(y_test, y_pred))

    def _recon_loss_for_type(self, cell_type, real_X: np.ndarray) -> float:
        """Mean per-cell reconstruction loss on real validation cells."""
        import torch

        was_training = self.model.model.training
        self.model.model.train()
        with torch.no_grad():
            # Build one-hot for this type
            ct_idx = self.model.le.transform([cell_type])[0]
            n = real_X.shape[0]
            onehot = torch.zeros(n, self.model.n_types, device=self.model.device)
            onehot[:, ct_idx] = 1
            x = torch.tensor(real_X, dtype=torch.float32, device=self.model.device)
            _, _, _, loss, recon, _ = self.model.model(x, onehot, self.model.beta)
        if not was_training:
            self.model.model.eval()
        return float(recon.item() / n)

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------
    def acquire(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Move selected cells from val → train according to strategy.

        Returns an acquisition report dict.
        """
        if self.val_adata is None or len(self.val_adata) == 0:
            return {"acquired_types": [], "n_acquired": 0, "reason": "val pool empty"}

        val_types = self.val_adata.obs[self.cell_type_key].unique()
        per_type = metrics["per_type"]
        recon_loss = metrics.get("recon_loss", {})

        # Build ranking
        if self.strategy == "metric_gap":
            # Higher cosine distance = worse; lower SCC = worse
            scores = {}
            for ct in val_types:
                ct_str = str(ct)
                m = per_type.get(ct_str, {})
                # Composite score: high cosine dist penalty, low scc penalty
                scores[ct] = m.get("cosine_distance", 0.0) * 2.0 + (1.0 - m.get("scc", 0.0))
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        elif self.strategy == "uncertainty":
            ranked = sorted(
                ((ct, recon_loss.get(str(ct), 0.0)) for ct in val_types),
                key=lambda kv: kv[1],
                reverse=True,
            )
        else:  # random
            rng = np.random.default_rng(self.random_state + len(self.history))
            shuffled = rng.permutation(list(val_types))
            ranked = [(ct, 0.0) for ct in shuffled]

        # Acquire cells from worst types until budget exhausted
        acquired_mask = np.zeros(len(self.val_adata), dtype=bool)
        acquired_types: list[str] = []
        budget_remaining = self.budget

        for ct, score in ranked:
            if budget_remaining <= 0:
                break
            type_mask = self.val_adata.obs[self.cell_type_key] == ct
            type_idx = np.where(type_mask)[0]
            n_take = min(budget_remaining, len(type_idx))
            if n_take > 0:
                take_idx = type_idx[:n_take]
                acquired_mask[take_idx] = True
                acquired_types.append(str(ct))
                budget_remaining -= n_take

        n_acquired = int(acquired_mask.sum())
        if n_acquired == 0:
            return {"acquired_types": [], "n_acquired": 0, "reason": "no cells matched criteria"}

        acquired = self.val_adata[acquired_mask].copy()
        remaining = self.val_adata[~acquired_mask].copy()

        # Concatenate acquired cells onto train
        self.train_adata = anndata.concat([self.train_adata, acquired], axis=0, merge="same")
        self.val_adata = remaining

        return {
            "acquired_types": acquired_types,
            "n_acquired": n_acquired,
            "budget": self.budget,
            "remaining_budget": budget_remaining,
            "strategy": self.strategy,
        }

    # ------------------------------------------------------------------
    # Round orchestration
    # ------------------------------------------------------------------
    def run_round(
        self,
        round_num: int,
        epochs: int = 100,
        lr: float = 1e-5,
        batch_size: int = 256,
        warm_start: bool = True,
    ) -> dict[str, Any]:
        """Execute one full active-learning round.

        Returns the round log dict.
        """
        print(f"\n{'='*60}")
        print(f"  Active Learning Round {round_num}  |  strategy={self.strategy}")
        print(f"  Train: {len(self.train_adata)}  |  Val: {len(self.val_adata)}")
        print(f"{'='*60}\n")

        # Checkpoint path for warm start
        ckpt_path = None
        if warm_start and round_num > 1:
            prev_ckpt = self.results_dir / f"round{round_num - 1}_model.pt"
            if prev_ckpt.exists():
                ckpt_path = str(prev_ckpt)

        # Train
        self.train(epochs=epochs, lr=lr, batch_size=batch_size, warm_start_path=ckpt_path)

        # Validate
        val_results = self.validate()

        # Save model checkpoint
        ckpt_out = self.results_dir / f"round{round_num}_model.pt"
        self.model.save(str(ckpt_out))

        # Acquire (skip on final round if desired, but we always acquire except val empty)
        acquisition = self.acquire(val_results)

        # Build log
        log = {
            "round": round_num,
            "strategy": self.strategy,
            "random_state": self.random_state,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "train_size": len(self.train_adata) - acquisition["n_acquired"],
            "val_size": len(self.val_adata) + acquisition["n_acquired"],
            "train_size_after": len(self.train_adata),
            "val_size_after": len(self.val_adata),
            "aggregate": val_results["aggregate"],
            "per_type": val_results["per_type"],
            "recon_loss": val_results["recon_loss"],
            "acquisition": acquisition,
        }

        # Write JSON
        out_path = self.results_dir / f"round{round_num}_kpi.json"
        with open(out_path, "w") as fh:
            json.dump(log, fh, indent=2)
        print(f"[ActiveLearner] Round {round_num} log written to {out_path}")

        self.history.append(log)
        return log

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_state(self, path: str | Path) -> None:
        """Save the current split state (not model weights)."""
        state = {
            "random_state": self.random_state,
            "strategy": self.strategy,
            "budget": self.budget,
            "history": self.history,
            "model_config": self.model_config,
        }
        with open(path, "w") as fh:
            json.dump(state, fh, indent=2)

    def load_state(self, path: str | Path) -> None:
        """Restore learner metadata (splits must be re-created or stored separately)."""
        with open(path) as fh:
            state = json.load(fh)
        self.random_state = state["random_state"]
        self.strategy = state["strategy"]
        self.budget = state["budget"]
        self.history = state["history"]
        self.model_config = state["model_config"]
