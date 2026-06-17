# ADME-T Training Scripts

Lightweight ML pipeline to replicate SOTA scores on 11 ADME-T endpoints using:
- **Feature vector**: Morgan count FP (1024) + Avalon count FP (1024) + ErG FP (315) + RDKit 2D descriptors (~210) ≈ 2574 dims
- **Model**: CatBoost (Logloss for classification, MAE for regression)
- **Split**: TDC scaffold split (same as benchmark papers)

All scripts assume you're running from the repo root with the venv activated:

```bash
cd /path/to/adme
source venv/bin/activate
```

---

## Scripts

### `features.py`
Feature computation module. Not run directly — imported by other scripts.
```python
from scripts.features import smiles_to_features, smiles_list_to_matrix
```

### `train_all.py` — Main training script
Trains CatBoost on all 11 ADME-T tasks and prints results vs SOTA.

```bash
# Train all tasks (3 seeds each, ~30 min on M3 Max)
python scripts/train_all.py

# Train specific tasks
python scripts/train_all.py --tasks bbb pgp cyp2c9

# Faster: single seed
python scripts/train_all.py --n_seeds 1
```

Saves per-task models to `results/{task}/catboost_seed{n}.cbm` and metrics to `results/{task}/result.json`.

### `train_solubility_chemprop.py` — Chemprop regression
Trains a Chemprop D-MPNN specifically for aqueous solubility. Chemprop outperforms CatBoost on small regression tasks (MAE target: 0.761).

```bash
pip install chemprop   # if not installed
python scripts/train_solubility_chemprop.py
```

### `predict_new_molecules.py` — Apply trained models
Runs all trained models on new molecules (e.g. docking hits).

```bash
# Score the top-1000 docking hits
python scripts/predict_new_molecules.py \
    --input top1000_with_adme.csv \
    --smiles_col "Canonical SMILES"

# Score a custom SMILES file (one SMILES per line)
python scripts/predict_new_molecules.py \
    --input my_compounds.smi

# Score specific tasks only
python scripts/predict_new_molecules.py \
    --input top1000_with_adme.csv \
    --smiles_col "Canonical SMILES" \
    --tasks bbb pgp cyp2c9 cyp2d6 cyp3a4
```

Output: `results/predictions.csv`

---

## SOTA Targets

| Task           | Dataset (n)               | Metric | SOTA  | Model          |
|----------------|---------------------------|--------|-------|----------------|
| solubility     | AqSolDB (9,982)           | MAE↓   | 0.761 | Chemprop-RDKit |
| hia            | HIA_Hou (578)             | AUROC↑ | 0.989 | MapLight+GNN   |
| caco2          | Caco2_Wang (910)          | MAE↓   | 0.256 | CaliciBoost    |
| bioavailability| Bioavailability_Ma (640)  | AUROC↑ | 0.938 | MapLight+GNN   |
| bbb            | BBB_Martins (2,030)       | AUROC↑ | 0.916 | MapLight       |
| pgp            | Pgp_Broccatelli (1,218)   | AUROC↑ | 0.938 | MapLight+GNN   |
| cyp1a2         | CYP1A2_Veith (12,579)     | AUROC↑ | 0.930 | DEEPCYPs       |
| cyp2c19        | CYP2C19_Veith (12,665)    | AUROC↑ | 0.900 | DEEPCYPs       |
| cyp2c9         | CYP2C9_Veith (12,092)     | AUPRC↑ | 0.859 | MapLight+GNN   |
| cyp2d6         | CYP2D6_Veith (13,130)     | AUPRC↑ | 0.790 | MapLight+GNN   |
| cyp3a4         | CYP3A4_Veith (12,328)     | AUPRC↑ | 0.916 | MapLight+GNN   |

CYP2C9/2D6/3A4 use AUPRC because the positive class is ~17% — class-imbalanced.

---

## Notes

- **HuskinDB** (skin permeability, log Kp) is not in TDC; download from `huskindb.drug-design.de` and add `cyp` → `logkp` task entry manually.
- **B3DB** (extended BBB dataset, 7,807 mol) is in `datasets/b3db_classification.tsv`. A separate `train_b3db.py` script can train on this larger set for better BBB coverage.
- CYP1A2/2C19 are **not in the TDC ADMET benchmark** (22-dataset group). They are trained on Veith data via TDC's ADME task API, evaluated with AUROC matching DEEPCYPs paper.
