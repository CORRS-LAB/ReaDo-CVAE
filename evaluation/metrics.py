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

"""Distributional-similarity metrics for real vs. synthetic scRNA-seq data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib.pyplot as plt
import scanpy as sc

import warnings

warnings.filterwarnings("ignore")


def preprocess(bdata, n_top_genes: int = 3000):
    """Preprocess sparse RNA data: filter low-expression genes and select HVGs.

    Parameters
    ----------
    bdata
        AnnData object containing raw counts.
    n_top_genes
        Number of highly variable genes to retain.

    Returns
    -------
    np.ndarray
        Indices of highly variable genes.
    """
    total_cells = bdata.X.shape[0]
    min_cells = total_cells * 0.01
    counts = (bdata.X > 0).sum(axis=0)
    expressed_genes = np.where(counts >= min_cells)[0]

    gdata = bdata[:, expressed_genes].copy()
    sc.pp.highly_variable_genes(
        gdata,
        n_top_genes=n_top_genes,
        inplace=True,
        flavor="seurat_v3",
    )

    highly_variable_mask = gdata.var["highly_variable"]
    g_hvg_index = np.where(highly_variable_mask)[0]
    hvg_index = expressed_genes[g_hvg_index]
    return hvg_index


def calculate_gcp(real_data: np.ndarray, gen_data: np.ndarray, n_genes_sample: int = 100) -> float:
    """Gene Correlation Preservation (GCP).

    Computes the Spearman correlation between upper-triangular gene
    correlation matrices of the top ``n_genes_sample`` highest-expressed genes.
    """
    gene_means = np.mean(real_data, axis=0)
    top_indices = np.argsort(gene_means)[-n_genes_sample:]

    real_sub = real_data[:, top_indices]
    gen_sub = gen_data[:, top_indices]

    corr_real = np.corrcoef(real_sub.T)
    corr_gen = np.corrcoef(gen_sub.T)
    triu_idx = np.triu_indices_from(corr_real, k=1)
    real_flat = corr_real[triu_idx]
    gen_flat = corr_gen[triu_idx]
    spearman_corr, _ = stats.spearmanr(real_flat, gen_flat)
    return float(spearman_corr)


def calculate_scc(real_data: np.ndarray, gen_data: np.ndarray) -> float:
    """Spearman Correlation of average gene expression (SCC)."""
    real_mean = np.mean(real_data, axis=0)
    gen_mean = np.mean(gen_data, axis=0)
    scc, _ = stats.spearmanr(real_mean, gen_mean)
    return float(scc)


def calculate_centroid_distances(real_data: np.ndarray, gen_data: np.ndarray) -> tuple[float, float]:
    """Euclidean and cosine distance between cell-type centroids."""
    real_centroid = np.mean(real_data, axis=0)
    gen_centroid = np.mean(gen_data, axis=0)
    dot_product = np.dot(real_centroid, gen_centroid)
    norm_product = np.linalg.norm(real_centroid) * np.linalg.norm(gen_centroid)
    if norm_product < 1e-10:
        cosine_sim = 0.0
    else:
        cosine_sim = dot_product / norm_product
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    cosine_dist = 1.0 - cosine_sim
    euclidean_dist = np.linalg.norm(real_centroid - gen_centroid)
    return float(euclidean_dist), float(cosine_dist)


def gaussian_kernel_single(source: torch.Tensor, target: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """Compute a single Gaussian kernel matrix between source and target."""
    total = torch.cat([source, target], dim=0)
    L2_distance = torch.cdist(total, total, p=2) ** 2
    return torch.exp(-L2_distance / bandwidth)


def mmd_rbf_multiple_kernels(
    source: torch.Tensor,
    target: torch.Tensor,
    bandwidths: list[float],
    is_loss: bool = False,
) -> torch.Tensor | float:
    """Maximum Mean Discrepancy (MMD) with multiple RBF kernels."""
    n = source.size(0)
    m = target.size(0)
    total_mmd = 0.0

    for bandwidth in bandwidths:
        kernels = gaussian_kernel_single(source, target, bandwidth)
        XX = kernels[:n, :n]
        YY = kernels[n:, n:]
        XY = kernels[:n, n:]
        mmd = torch.mean(XX) + torch.mean(YY) - 2 * torch.mean(XY)
        total_mmd += mmd

    result = total_mmd / len(bandwidths)
    return result if is_loss else result.item()


def calculate_MMD_paper(
    real_data: np.ndarray,
    gen_data: np.ndarray,
    n_samples: int = 500,
    is_loss: bool = False,
) -> float:
    """MMD computed with three Gaussian kernels using the median heuristic."""
    n_real = min(n_samples, real_data.shape[0])
    n_gen = min(n_samples, gen_data.shape[0])

    real_idx = np.random.choice(real_data.shape[0], n_real, replace=False)
    gen_idx = np.random.choice(gen_data.shape[0], n_gen, replace=False)

    real_sample = real_data[real_idx]
    gen_sample = gen_data[gen_idx]

    combined_sample = np.vstack([real_sample, gen_sample])
    dists = pairwise_distances(combined_sample, metric="euclidean")
    upper_triangular = dists[np.triu_indices(len(dists), k=1)]
    median_distance = float(np.median(upper_triangular))

    if median_distance == 0 or np.isnan(median_distance):
        median_distance = 1.0

    bandwidths = [
        (median_distance ** 2) / 0.5,
        (median_distance ** 2) / 1.0,
        (median_distance ** 2) / 2.0,
    ]

    X = torch.tensor(real_sample, dtype=torch.float32)
    Y = torch.tensor(gen_sample, dtype=torch.float32)

    return mmd_rbf_multiple_kernels(X, Y, bandwidths, is_loss=is_loss)


def calculate_milisi(combined_pca: np.ndarray, labels: np.ndarray, n_neighbors: int = 90) -> float:
    """Mean inverse Simpson's index (miLISI) on a PCA embedding.

    Values near 1 indicate batch separation; values near 2 indicate mixing.
    """
    n_neighbors = min(n_neighbors, len(combined_pca) - 1)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(combined_pca)
    _, indices = nbrs.kneighbors(combined_pca)

    lisi_scores = []
    for i in range(len(indices)):
        neighbor_indices = indices[i][1:]
        neighbor_labels = labels[neighbor_indices]
        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
        proportions = counts / n_neighbors
        simpson_index = np.sum(proportions**2)
        lisi = 1.0 / simpson_index if simpson_index > 0 else 1.0
        lisi_scores.append(lisi)

    return float(np.mean(lisi_scores))


def evaluate_generated_data(
    real_data: np.ndarray,
    gen_data: np.ndarray,
    gen_method_name: str,
    n_pca_components: int = 500,
    n_rf_trees: int = 1000,
    random_state: int = 42,
) -> dict:
    """Evaluate synthetic data against real data using six metrics.

    Returns a dictionary with keys: ``method``, ``cosine_distance``,
    ``mmd``, ``milisi``, ``roc_auc``, ``gcp``, ``scc``.
    """
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    print(f"Evaluating: {gen_method_name}")
    print(f"Real shape: {real_data.shape}, Generated shape: {gen_data.shape}")

    real_log = np.log1p(real_data)
    gen_log = np.log1p(gen_data)

    scaler = StandardScaler(with_mean=False, with_std=True)
    real_scaled = scaler.fit_transform(real_log)
    gen_scaled = scaler.transform(gen_log)

    _, cosine_dist = calculate_centroid_distances(real_scaled, gen_scaled)

    scaler_pca = StandardScaler(with_mean=True, with_std=True)
    real_pca_input = scaler_pca.fit_transform(real_log)
    gen_pca_input = scaler_pca.transform(gen_log)
    combined = np.vstack([real_pca_input, gen_pca_input])
    pca = PCA(n_components=n_pca_components, random_state=random_state)
    combined_pca = pca.fit_transform(combined)
    real_pca = combined_pca[: real_data.shape[0]]
    gen_pca = combined_pca[real_data.shape[0] :]

    mmd_val = calculate_MMD_paper(real_pca, gen_pca)

    labels = np.concatenate([np.zeros(real_pca.shape[0]), np.ones(gen_pca.shape[0])])
    milisi_val = calculate_milisi(
        combined_pca, labels, n_neighbors=min(90, real_pca.shape[0] // 2)
    )

    X = np.vstack([real_data, gen_data])
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    scaler_rf = StandardScaler()
    X_train_scaled = scaler_rf.fit_transform(X_train)
    X_test_scaled = scaler_rf.transform(X_test)
    pca_rf = PCA(n_components=n_pca_components, random_state=random_state)
    X_train_pca = pca_rf.fit_transform(X_train_scaled)
    X_test_pca = pca_rf.transform(X_test_scaled)
    rf = RandomForestClassifier(n_estimators=n_rf_trees, random_state=random_state)
    rf.fit(X_train_pca, y_train)
    y_pred = rf.predict_proba(X_test_pca)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred)

    gcp_val = calculate_gcp(real_scaled, gen_scaled)
    scc_val = calculate_scc(real_scaled, gen_scaled)

    results_dict = {
        "method": gen_method_name,
        "cosine_distance": cosine_dist,
        "mmd": mmd_val,
        "milisi": milisi_val,
        "roc_auc": roc_auc,
        "gcp": gcp_val,
        "scc": scc_val,
    }

    print(f"Cosine distance: {cosine_dist:.6f}")
    print(f"MMD:            {mmd_val:.4f}")
    print(f"miLISI:         {milisi_val:.4f}")
    print(f"ROC AUC:        {roc_auc:.4f}")
    print(f"GCP:            {gcp_val:.4f}")
    print(f"SCC:            {scc_val:.4f}")
    print("=" * 50)

    return results_dict


if __name__ == "__main__":
    original_data_path = "data/demo_data.h5ad"
    generate_data_path = "data/ReaDo-CVAE_synthetic_all.h5ad"

    original_adata = sc.read_h5ad(original_data_path)
    original_data = original_adata[original_adata.obs["cell_type"] == 10].X
    if hasattr(original_data, "toarray"):
        original_data = original_data.toarray()

    generate_adata = sc.read_h5ad(generate_data_path)
    generate_data = generate_adata[generate_adata.obs["cell_type"] == 10].X
    if hasattr(generate_data, "toarray"):
        generate_data = generate_data.toarray()

    seed = 42
    n_samples = 500
    np.random.seed(seed)
    torch.manual_seed(seed)

    all_idx = np.random.choice(original_data.shape[0], n_samples, replace=False)
    ge_idx = np.random.choice(generate_data.shape[0], n_samples, replace=False)
    part1 = original_data[all_idx]
    part2 = generate_data[ge_idx]

    results = evaluate_generated_data(part1, part2, "ReaDo-CVAE")
