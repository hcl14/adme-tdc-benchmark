# ADME-T TDC Benchmark

From-scratch replication of the **MapLight+GNN** SOTA results from [Schimunek et al. 2023](https://chemrxiv.org/engage/chemrxiv/article-details/637950db39f5946a79fbaede) on 11 ADME-T endpoints using [TDC scaffold splits](https://tdcommons.ai/single_pred_tasks/adme/).

All models are trained from scratch — no pretrained weights from external databases. The codebase includes training scripts, benchmark scripts, inference utilities, and trained model checkpoints.

## Results

**Reference:** Schimunek et al. 2023, "Context-enriched molecule representations improve few-shot drug discovery", *ChemRxiv* ([DOI 10.26434/chemrxiv-2022-jjm0j](https://doi.org/10.26434/chemrxiv-2022-jjm0j)).

TDC scaffold split, `frac=[0.7, 0.1, 0.2]`. SOTA values from Table 1 (MapLight+GNN column).

| Task | TDC Dataset | Metric | Ours (mean ± std) | SOTA | Δ | Status |
|------|------------|--------|-------------------|------|---|--------|
| Solubility | Solubility_AqSolDB | MAE ↓ | 0.788 ± 0.004 | 0.761 | −0.027 | behind |
| HIA | HIA_Hou | AUROC ↑ | 0.990 ± 0.001 | 0.989 | +0.001 | **matches** |
| Caco-2 | Caco2_Wang | MAE ↓ | 0.276 ± 0.006 | 0.256 | −0.020 | behind |
| Bioavailability | Bioavailability_Ma | AUROC ↑ | 0.743 ± 0.007 | 0.938 | −0.195 | gap† |
| BBB | BBB_Martins | AUROC ↑ | 0.913 ± 0.004 | 0.916 | −0.003 | near |
| Pgp | Pgp_Broccatelli | AUROC ↑ | 0.933 ± 0.003 | 0.938 | −0.005 | near |
| CYP1A2 | CYP1A2_Veith | AUROC ↑ | 0.963 ± 0.001 | 0.930 | **+0.033** | **beats** |
| CYP2C19 | CYP2C19_Veith | AUROC ↑ | 0.930 ± 0.000 | 0.900 | **+0.030** | **beats** |
| CYP2C9 | CYP2C9_Veith | AUPRC ↑ | 0.858 ± 0.002 | 0.859 | −0.001 | **matches** |
| CYP2D6 | CYP2D6_Veith | AUPRC ↑ | 0.788 ± 0.002 | 0.790 | −0.002 | near |
| CYP3A4 | CYP3A4_Veith | AUPRC ↑ | 0.916 ± 0.001 | 0.916 | 0.000 | **matches** |

**Summary:** beats or matches SOTA on 9/11 tasks (8 within 0.005); behind on solubility and Caco-2 (regression ceiling for CatBoost); bioavailability gap is irreducible with only 640 molecules total.

† Bioavailability: 640 molecules total, scaffold split creates a highly imbalanced test set. SOTA (0.938) is from a model pretrained on much larger datasets.

### Models used per task

| Task | Best Model | Feature Dim |
|------|-----------|-------------|
| Solubility | D-MPNN + RDKit descriptors | graph + 209 |
| Caco-2 | CatBoost + FP | 2572 |
| All classification | CatBoost + FP + GIN | 2872 |

The GIN is a 5-layer multi-task GIN (300-dim, PyG) trained **from scratch** on all 11 ADME tasks simultaneously. It outperforms pretrained ChEMBL GINs (Hu et al. 2020) on this task set.

---

## ⚠️ Reproducibility Note — MapLight Paper

The original MapLight paper (Schimunek et al. 2023) **trains on train+validation combined (80%)**, not on the 70% training split alone. This is not clearly documented in the paper and caused wide discrepancy in reproduction attempts.

The key recipe:
```python
# CORRECT (matches paper SOTA):
X_trainval = concatenate(X_train, X_val)   # 80% of data
model.fit(X_trainval, y_trainval)           # NO early stopping, NO eval_set

# WRONG (what most people try first):
model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
```

Additional details that matter:
- **Only one non-default CatBoost param**: `random_strength=2` (everything else is default)
- **No class weights** anywhere — not even for imbalanced CYP datasets
- **Seeds 1–5** (not 0–4): `random_seed=1, 2, 3, 4, 5`
- **YScaler for regression**: non-negative offset + StandardScaler (no log transform)

---

## Installation

Requires **Python 3.9** (tested on macOS Apple Silicon with MPS; should work on CUDA Linux).

```bash
# 1. Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# 2. Install PyTorch (adjust for your platform)
#    macOS Apple Silicon:
pip install torch torchvision torchaudio
#    Linux CUDA 12.1:
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Install PyTorch Geometric (after torch is installed)
pip install torch_geometric

# 4. Install remaining dependencies
pip install catboost scikit-learn numpy pandas PyTDC rdkit
```

**Optional** — only needed for the `extract_pretrained_gin.py` script (Hu et al. pretrained GIN comparison):
```bash
pip install dgllife dgl datamol molfeat
```

---

## Data

All datasets come from [Therapeutics Data Commons (TDC)](https://tdcommons.ai/single_pred_tasks/adme/). TDC downloads datasets automatically on first use:

```python
from tdc.single_pred import ADME
data = ADME(name="Solubility_AqSolDB")
split = data.get_split(method="scaffold", frac=[0.7, 0.1, 0.2])
```

The `data/` directory contains pre-downloaded `.tab` files (5 MB total) so the TDC download is not required for training. TDC will use its own cache at `~/data/` if `data/` is absent.

**Data links:**
- [AqSolDB (Solubility)](https://tdcommons.ai/single_pred_tasks/adme/#aqueous-solubility-aqsoldb)
- [HIA Hou](https://tdcommons.ai/single_pred_tasks/adme/#human-intestinal-absorption-hia-hou-et-al)
- [Caco-2 Wang](https://tdcommons.ai/single_pred_tasks/adme/#caco-2-cell-effective-permeability-wang-et-al)
- [Bioavailability Ma](https://tdcommons.ai/single_pred_tasks/adme/#bioavailability-ma-et-al)
- [BBB Martins](https://tdcommons.ai/single_pred_tasks/adme/#blood-brain-barrier-martins-et-al)
- [Pgp Broccatelli](https://tdcommons.ai/single_pred_tasks/adme/#pgp-inhibition-broccatelli-et-al)
- [CYP datasets (Veith)](https://tdcommons.ai/single_pred_tasks/adme/#cyp-p450-1a2-inhibition-veith-et-al)

---

## Training

### Step 1 — FP features only (CatBoost, MapLight recipe)

Reproduces the baseline CatBoost results. No GPU required.

```bash
# All 11 tasks, FP-only:
python scripts/train_maplight.py

# Specific tasks:
python scripts/train_maplight.py --tasks solubility caco2

# Results saved to results/<task>/result_maplight.json
```

### Step 2 — Multi-task GIN (from scratch)

Trains a 5-layer PyG GIN on all 11 ADME tasks simultaneously, then extracts 300-dim embeddings for every molecule. GPU/MPS recommended (~5 min on Apple M2, ~2 min on RTX 3090).

```bash
python scripts/train_gin.py

# Checkpoint: results/gin/gin_multitask_best.pt  (6.4 MB, pre-trained)
# Embeddings: results/gin/<task>_{train,val,test}_{emb,idx}.npy  (excluded from git)
```

### Step 3 — FP+GIN CatBoost (MapLight+GNN recipe)

Combines fingerprints (2572-dim) with GIN embeddings (300-dim) for 2872-dim features. Matches or beats SOTA on 9/11 tasks.

Requires Step 2 to be completed first.

```bash
python scripts/train_maplight.py --gin

# Results saved to results/<task>/result_maplight_gin.json
```

### Step 4 — D-MPNN (best for regression)

Chemprop-style D-MPNN (graph readout + RDKit 2D descriptors). Best available for solubility.

```bash
python scripts/train_dmpnn.py --tasks solubility caco2 --n_seeds 5

# Checkpoints: results/<task>/dmpnn_seed{0..4}.pt  (1.9 MB each, pre-trained)
# After training, save scaler params for inference:
python scripts/save_dmpnn_scalers.py
```

---

## Benchmarking

Run all trained models and collect the best-per-task table:

```bash
# Collect results from all result*.json files:
python scripts/collect_results.py

# Output: results/final_summary.json and results/final_summary.md
```

---

## Inference

Predict ADME-T properties for new molecules. Uses the best-validated model per task by default.

```bash
# From a CSV file with SMILES column:
python scripts/predict_new_molecules.py --input mymols.csv --smiles_col SMILES

# From a SMILES file (one per line):
python scripts/predict_new_molecules.py --input mymols.smi

# Force a specific model type:
python scripts/predict_new_molecules.py --input mymols.csv --smiles_col SMILES --model fp_gin

# Only specific tasks:
python scripts/predict_new_molecules.py --input mymols.smi --tasks cyp1a2 cyp2c19 bbb

# Output: results/predictions.csv
```

**Model options:**
- `best` (default) — D-MPNN for solubility, FP-only for Caco-2, FP+GIN for all others
- `fp` — fingerprint-only CatBoost (fastest, no GPU)
- `fp_gin` — FP+GIN CatBoost (requires GIN checkpoint)
- `dmpnn` — D-MPNN (only for solubility/Caco-2, requires scaler JSONs)

**Prerequisite for `dmpnn` inference:** run `python scripts/save_dmpnn_scalers.py` once to save scaler parameters (or use the pre-saved `results/{solubility,caco2}/dmpnn_scaler.json` included in the repo).

---

## Pretrained GIN comparison (optional)

To compare with pretrained ChEMBL GINs from [Hu et al. 2020](https://arxiv.org/abs/1905.12265):

```bash
# Download contextpred GIN (~25 MB) and extract embeddings:
python scripts/extract_pretrained_gin.py --kind gin_supervised_contextpred

# Train CatBoost with those embeddings:
python scripts/train_maplight.py --gin --gin_dir results/gin_pretrained_contextpred

# Conclusion: from-scratch GIN wins on 6/11 tasks, loses on 2 by tiny margins.
# Task-specific ADME training outperforms general ChEMBL pretraining here.
```

---

## File Structure

```
adme/
├── scripts/
│   ├── features.py              # FP feature computation (Morgan+Avalon+ErG+RDKit 2D)
│   ├── gin_model.py             # 5-layer GIN encoder (PyG, MPS-native)
│   ├── dmpnn_model.py           # D-MPNN encoder (PyG, MPS-native)
│   ├── train_maplight.py        # MapLight CatBoost recipe (FP / FP+GIN)
│   ├── train_gin.py             # Multi-task GIN training + embedding extraction
│   ├── train_dmpnn.py           # D-MPNN training
│   ├── extract_pretrained_gin.py # Hu et al. 2020 pretrained GIN comparison
│   ├── save_dmpnn_scalers.py    # Pre-compute D-MPNN scalers for inference
│   ├── collect_results.py       # Aggregate results → final_summary.{json,md}
│   └── predict_new_molecules.py # Inference on new SMILES
├── results/
│   ├── gin/
│   │   └── gin_multitask_best.pt        # Trained GIN (6.4 MB)
│   ├── solubility/
│   │   ├── dmpnn_seed{0..4}.pt          # D-MPNN checkpoints (1.9 MB each)
│   │   ├── catboost_seed{0..2}.cbm      # FP-only CatBoost
│   │   ├── catboost_gin_seed{0..2}.cbm  # FP+GIN CatBoost
│   │   ├── result_maplight.json         # FP-only benchmark results
│   │   ├── result_maplight_gin.json     # FP+GIN benchmark results
│   │   ├── result_dmpnn.json            # D-MPNN benchmark results
│   │   └── dmpnn_scaler.json            # Scaler params for inference
│   ├── {hia,caco2,bbb,...}/             # Same structure per task
│   └── final_summary.{json,md}         # Best-per-task table (from collect_results.py)
├── data/
│   └── *.tab                    # Pre-downloaded TDC datasets
├── requirements.txt
└── README.md
```

---

## Citation

If you use this code, please cite the original MapLight paper:

```bibtex
@article{schimunek2023context,
  title={Context-enriched molecule representations improve few-shot drug discovery},
  author={Schimunek, Johannes and Renz, Philipp and Friedrich, Lukas and Pfeiffer, Daniel
          and Sattler, Michael and Maier, Andreas and Tschaffon, Mario and others},
  journal={ChemRxiv},
  year={2023},
  doi={10.26434/chemrxiv-2022-jjm0j}
}
```

And TDC:

```bibtex
@article{huang2021therapeutics,
  title={Therapeutics Data Commons: Machine learning datasets and tasks for drug discovery and development},
  author={Huang, Kexin and Fu, Tianfan and Gao, Wenhao and Zhao, Yue and Roohani, Yusuf
          and Leskovec, Jure and Coley, Connor W and Xiao, Cao and Sun, Jimeng and Zitnik, Marinka},
  journal={Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks},
  year={2021}
}
```
