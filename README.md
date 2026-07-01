# ReaDo‑CVAE

**ReaDo‑CVAE** is a conditional variational autoencoder for doublet synthesis in single‑cell RNA sequencing data.

ReaDo‑CVAE reformulates doublet synthesis as cell‑type‑conditioned nonlinear generation, learning directly from consensus doublets without relying on linear summation.  
The model produces synthetic doublets that are distributionally realistic and faithfully reproduce algorithmic behaviour observed on real data.

---

## Repository structure

```
ReaDo-CVAE/
├── reado_cvae/                       # Python package (ReaDoCVAE API)
│   ├── __init__.py
│   ├── _model.py                     # core model
│   └── _modules.py                   # attention, ZINB, residual blocks
├── evaluation/
│   ├── metrics.py                    # distributional similarity metrics (MMD, GCP, miLISI, RF‑AUROC, etc.)
│   └── ccs_acp.py                    # CCS & ACP computation
├── data/
│   ├── demo_data.h5ad                # small example dataset (~1000 cells, 3000 HVGs)
│   ├── ReaDo-CVAE_synthetic_all.h5ad # pre‑generated synthetic doublets
│   └── demo_results.csv              # pre‑computed algorithm predictions (for CCS/ACP demo)
├── test.py                           # one‑click demo: train → generate
├── environment.yml
└── README.md
```

---

## Installation

```bash
git clone https://github.com/xxx/ReaDo-CVAE.git
cd ReaDo-CVAE
conda env create -f environment.yml
conda activate reado-cvae
```

---

## Quick start

The demo data (`data/demo_data.h5ad`) contains ~1000 cells with 3000 highly variable genes and cell‑type annotations.

### 1. Train & generate

```bash
python test.py
```

Loads the demo data, trains a ReaDo‑CVAE model, and generates synthetic doublets for all cell types and a single example type.  
Results are saved in `result/data/`.  
A pre‑generated dataset (`data/ReaDo-CVAE_synthetic_all.h5ad`) is also provided so you can skip training and directly evaluate.

### 2. Evaluate distributional similarity

```bash
python evaluation/metrics.py
```

Compares real and synthetic cells (type 10) and reports six metrics: cosine distance, MMD, miLISI, RF‑AUROC, GCP, and SCC.

### 3. Compute CCS & ACP

```bash
python evaluation/ccs_acp.py
```

Reads pre‑computed detection results from `data/demo_results.csv` and prints the cross‑algorithm consensus score (CCS) and aggregated consensus proportion (ACP) for each algorithm.

---

## Data & reproducibility

The demo data allow you to verify that the code runs correctly.  
Full datasets used in the paper are available from the corresponding author upon reasonable request.  
Raw sequencing data have been deposited at GEO (accession number will be provided upon publication).

The nine third‑party doublet detection algorithms used for benchmarking are **not** included in this repository. Their original repositories and exact versions are listed in the paper’s Supplementary Information.

---

## License

MIT License – see the [LICENSE](LICENSE) file.

---

## Contact

For questions, open an issue on GitHub or contact: **panxiaoqing@shnu.edu.cn**
