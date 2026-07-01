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

"""Quick-start demo: train ReaDo-CVAE and generate synthetic doublets."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import scanpy as sc
from reado_cvae import ReaDoCVAE

import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Auto-detect device: MPS on Apple Silicon, then CUDA, then CPU
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    device = "mps"
    print("Using MPS device")
elif torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA device")
else:
    device = "cpu"
    print("Using CPU device")

# ---------------------------------------------------------------------------
# 1. Load demo data (highly variable genes + cell-type annotations)
# ---------------------------------------------------------------------------
adata = sc.read_h5ad("data/demo_data.h5ad")

# ---------------------------------------------------------------------------
# 2. Initialize model
# ---------------------------------------------------------------------------
model = ReaDoCVAE(
    adata,
    cell_type_key="cell_type",
    n_latent=100,
    hidden_dims=[1024, 512, 512, 256, 256, 128],
    dropout_rate=0.1,
    beta=2.0,
    device=device,
)

# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------
model.train(epochs=100, batch_size=256, lr=1e-5)

# ---------------------------------------------------------------------------
# 4. Generate synthetic doublets
# ---------------------------------------------------------------------------
synthetic_all = model.sample()
synthetic_all.write("synthetic_all.h5ad")

synthetic_t = model.sample(cell_type=10, n_cells=2000)
synthetic_t.write("synthetic_10_cells.h5ad")
