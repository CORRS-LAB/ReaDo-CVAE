# ReaDo-CVAE

**ReaDo‑CVAE** is a conditional variational autoencoder for doublet synthesis in single-cell RNA-seq.

The model treats doublet synthesis as cell-type-conditioned nonlinear generation and learns directly from consensus doublets instead of relying on linear summation. In practice, this yields synthetic doublets that better match the real distribution and preserve downstream doublet-detection behavior.

---

## Platform support

- Supported: **Linux** and **macOS**
- Experimental: **Windows** may work in WSL or with a native setup, but Linux/macOS are the primary supported environments.

---

## Requirements

- Python **>= 3.12**
- [uv](https://docs.astral.sh/uv/) (recommended dependency manager)
- Git

---

## Repository structure

```text
ReaDo-CVAE/
├── reado_cvae/                       # Python package (ReaDoCVAE API)
│   ├── __init__.py
│   ├── _model.py                     # Core model implementation
│   └── _modules.py                   # Attention, ZINB, residual blocks
├── evaluation/
│   ├── metrics.py                    # Distributional similarity metrics (MMD, GCP, miLISI, RF-AUROC, etc.)
│   └── ccs_acp.py                    # CCS and ACP computation
├── data/
│   ├── demo_data.h5ad                # Small demo dataset (~1000 cells, 3000 HVGs)
│   ├── ReaDo-CVAE_synthetic_all.h5ad # Pre-generated synthetic doublets
│   └── demo_results.csv              # Pre-computed algorithm predictions (for CCS/ACP demo)
├── test.py                           # One-click demo: train → generate synthetic doublets
├── pyproject.toml                    # Project metadata and dependencies
├── walkthrough.qmd                   # Walkthrough notebook and reproducibility notes
└── README.md
```

---

## Installation

```bash
git clone https://github.com/CORRS-LAB/ReaDo-CVAE
cd ReaDo-CVAE
uv sync
```

If `uv` is not available, use a normal Python virtual environment and install dependencies with:

```bash
python -m pip install -e .
```

Then run commands with your environment’s `python`.

---

## Quick start

The demo dataset (`data/demo_data.h5ad`) contains ~1,000 cells with ~3,000 highly variable genes and cell-type annotations.

### 1) Train and generate synthetic doublets

```bash
uv run python test.py
```

This command loads the demo data, trains ReaDo‑CVAE, and saves:

- `synthetic_all.h5ad` (all cell types, matching original proportions)
- `synthetic_10_cells.h5ad` (example single-cell-type sample)

If you want to skip training, the pre-generated `data/ReaDo-CVAE_synthetic_all.h5ad` file is provided.

### 2) Evaluate distributional similarity

```bash
uv run python evaluation/metrics.py
```

Compares one real/synthetic pair (cell type 10 in the bundled demo) and reports six metrics:
cosine distance, MMD, miLISI, RF-AUROC, GCP, and SCC.

### 3) Compute CCS and ACP

```bash
uv run python evaluation/ccs_acp.py
```

Uses `data/demo_results.csv` and prints CCS (cross-algorithm consensus score) and ACP (aggregated consensus proportion) for each algorithm.

---

## Data and reproducibility

The demo files are intended for functional verification.
Full paper datasets are available from the corresponding author upon reasonable request.
Raw sequencing data have been deposited at GEO (accession number to be added upon publication).

The nine third-party doublet detection algorithms used in benchmarking are not included in this repository.
Their original repositories and exact versions are listed in the paper’s Supplementary Information.

---

## License

Apache-2.0 — see the [LICENSE](LICENSE) file.

---

## Contact

For questions, open an issue on GitHub or contact: **panxiaoqing@shnu.edu.cn**
