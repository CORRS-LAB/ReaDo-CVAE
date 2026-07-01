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

"""Cross-algorithm Consensus Score (CCS) and Aggregated Consensus Proportion (ACP)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_ccs_acp(
    df: pd.DataFrame,
    algorithm_col: str = "algorithm",
    score_col: str = "score",
    pred_col: str = "pred_label",
    name_col: str = "name",
    boundary_frac: float = 0.1,
    min_boundary: int = 30,
) -> pd.DataFrame:
    """Compute CCS and ACP for each doublet-detection algorithm.

    Parameters
    ----------
    df
        DataFrame with one row per (cell, algorithm) containing scores and
        binary predictions.
    algorithm_col
        Column holding algorithm names.
    score_col
        Column holding continuous scores.
    pred_col
        Column holding binary predictions (1 = doublet, 0 = singlet).
    name_col
        Column holding unique cell identifiers.
    boundary_frac
        Fraction of singlet cells to use as the boundary set.
    min_boundary
        Minimum number of boundary cells.

    Returns
    -------
    pd.DataFrame
        Table with columns ``algorithm``, ``CCS``, ``ACP``.
    """
    data = df.copy()
    algorithms = data[algorithm_col].unique()

    # Global standardisation per algorithm
    stats = {}
    for alg in algorithms:
        vals = data.loc[data[algorithm_col] == alg, score_col]
        mu = vals.mean()
        sigma = vals.std()
        if sigma == 0:
            sigma = 1.0
        stats[alg] = (mu, sigma)

    data["score_z"] = data.apply(
        lambda row: (row[score_col] - stats[row[algorithm_col]][0])
        / stats[row[algorithm_col]][1],
        axis=1,
    )

    total_cells = data[name_col].nunique()
    N = max(int(total_cells * boundary_frac), min_boundary)

    results = []
    for alg in algorithms:
        neg_mask = (data[algorithm_col] == alg) & (data[pred_col] == 0)
        neg_data = data[neg_mask].copy()
        if len(neg_data) == 0:
            results.append({"algorithm": alg, "CCS": np.nan, "ACP": np.nan})
            continue

        neg_data.sort_values("score_z", ascending=False, inplace=True)
        boundary = neg_data.iloc[:N] if len(neg_data) >= N else neg_data

        other_scores = []
        other_pos = []
        for _, row in boundary.iterrows():
            cell = row[name_col]
            others = data[(data[name_col] == cell) & (data[algorithm_col] != alg)]
            if len(others) == 0:
                continue
            other_scores.append(others["score_z"].mean())
            other_pos.append((others[pred_col] == 1).mean())

        if other_scores:
            ccs = np.mean(other_scores)
            acp = np.mean(other_pos)
        else:
            ccs, acp = np.nan, np.nan

        results.append({"algorithm": alg, "CCS": ccs, "ACP": acp})

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = pd.read_csv("data/demo_results.csv")
    result = compute_ccs_acp(
        df,
        algorithm_col="algorithm",
        score_col="score",
        pred_col="pred_label",
        name_col="name",
    )
    print(result)
