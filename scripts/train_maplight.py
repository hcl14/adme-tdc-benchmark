"""
Definitive MapLight replication (Schimunek et al. 2023). Matches the published
notebook recipe EXACTLY:

  * Train on train+valid combined (TDC `train_val` = 80%), test on 20% test.
  * CatBoost with DEFAULT params + only random_strength=2, random_seed,
    verbose=0, loss_function. NO early stopping, NO eval_set, NO class weights.
  * 5 seeds [1,2,3,4,5].
  * Regression: non-negative offset + StandardScaler (log=False for all 11 tasks).
  * Features: Morgan count(1024)+Avalon count(1024)+ErG(315)+RDKit descs.
  * +GNN variant: concatenate 300-d molfeat GIN (Hu et al. supervised masking).

Usage:
  python scripts/train_maplight.py                         # FP only, all tasks
  python scripts/train_maplight.py --gin                   # FP+GIN (MapLight+GNN)
  python scripts/train_maplight.py --gin --tasks cyp2c9 cyp2d6 cyp3a4
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
RESULTS_GIN = REPO_ROOT / "results" / "gin"  # overridden by --gin_dir
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from features import smiles_list_to_matrix
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error

# (tdc_name, task_type, metric, sota).  task_type: 'binary' or 'regression'
TASKS = {
    "solubility"      : ("Solubility_AqSolDB", "regression", "mae",   0.761),
    "hia"             : ("HIA_Hou",            "binary",     "auroc", 0.989),
    "caco2"           : ("Caco2_Wang",         "regression", "mae",   0.256),
    "bioavailability" : ("Bioavailability_Ma", "binary",     "auroc", 0.938),
    "bbb"             : ("BBB_Martins",        "binary",     "auroc", 0.916),
    "pgp"             : ("Pgp_Broccatelli",    "binary",     "auroc", 0.938),
    "cyp1a2"          : ("CYP1A2_Veith",       "binary",     "auroc", 0.930),
    "cyp2c19"         : ("CYP2C19_Veith",      "binary",     "auroc", 0.900),
    "cyp2c9"          : ("CYP2C9_Veith",       "binary",     "auprc", 0.859),
    "cyp2d6"          : ("CYP2D6_Veith",       "binary",     "auprc", 0.790),
    "cyp3a4"          : ("CYP3A4_Veith",       "binary",     "auprc", 0.916),
}
SEEDS = [1, 2, 3, 4, 5]


def get_split_full(tdc_name):
    """Return train, valid, test separately (TDC scaffold split)."""
    from tdc.single_pred import ADME
    data = ADME(name=tdc_name)
    split = data.get_split(method="scaffold", frac=[0.7, 0.1, 0.2])
    return split["train"], split["valid"], split["test"]


def _load_gin(task_key, split):
    p = RESULTS_GIN / f"{task_key}_{split}_emb.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}; run scripts/extract_pretrained_gin.py first")
    return np.load(p), np.load(str(RESULTS_GIN / f"{task_key}_{split}_idx.npy"))


class YScaler:
    """MapLight scaler: non-negative offset (+ optional log) + StandardScaler."""
    def __init__(self, log=False):
        self.log = log
        self.offset = None
        self.scaler = None

    def fit(self, y):
        self.offset = min(float(np.min(y)), 0.0)
        y = (np.asarray(y, dtype=np.float64).reshape(-1, 1) - self.offset)
        if self.log:
            y = np.log10(y + 1.0)
        self.scaler = preprocessing.StandardScaler().fit(y)

    def transform(self, y):
        y = (np.asarray(y, dtype=np.float64).reshape(-1, 1) - self.offset)
        if self.log:
            y = np.log10(y + 1.0)
        return self.scaler.transform(y)

    def inverse(self, y_scale):
        y = self.scaler.inverse_transform(np.asarray(y_scale, dtype=np.float64).reshape(-1, 1))
        if self.log:
            y = 10.0 ** y - 1.0
        return (y + self.offset).reshape(-1)


def feat_matrix(df, task_key=None, split=None, use_gin=False):
    """Return (X, y, valid_idx). Optionally concat precomputed molfeat GIN (300-d)."""
    smiles = df["Drug"].astype(str).tolist()
    X_fp, idx = smiles_list_to_matrix(smiles, verbose=False)
    y_all = df["Y"].values
    if use_gin:
        gin, gidx = _load_gin(task_key, split)
        common = sorted(set(idx) & set(gidx.tolist()))
        fpos = {v: i for i, v in enumerate(idx)}
        gpos = {v: i for i, v in enumerate(gidx.tolist())}
        rows_f = [fpos[c] for c in common]
        rows_g = [gpos[c] for c in common]
        X = np.concatenate([X_fp[rows_f], gin[rows_g]], axis=1)
        return X, y_all[common], common
    return X_fp, y_all[idx], idx


def metric(y_true, y_pred, m):
    if m == "auroc":
        return roc_auc_score(y_true, y_pred)
    if m == "auprc":
        return average_precision_score(y_true, y_pred)
    return mean_absolute_error(y_true, y_pred)


def run_task(task_key, use_gin=False):
    from catboost import CatBoostClassifier, CatBoostRegressor
    tdc_name, ttype, metric_name, sota = TASKS[task_key]
    label = "FP+GIN" if use_gin else "FP"
    print(f"\n{'='*64}\n{task_key} ({tdc_name}) [{label}]  {metric_name.upper()}"
          f"{'↓' if metric_name=='mae' else '↑'}  SOTA {sota:.3f}\n{'='*64}")

    train_df, val_df, test_df = get_split_full(tdc_name)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)} "
          f"(train_val={len(train_df)+len(val_df)})")

    X_tr, y_tr, _ = feat_matrix(train_df, task_key, "train", use_gin)
    X_va, y_va, _ = feat_matrix(val_df,   task_key, "val",   use_gin)
    X_te, y_te, _ = feat_matrix(test_df,  task_key, "test",  use_gin)
    X_trainval = np.concatenate([X_tr, X_va], axis=0)
    y_trainval = np.concatenate([y_tr, y_va], axis=0)
    print(f"feature dim: {X_trainval.shape[1]} | train_val rows: {X_trainval.shape[0]}")

    scores = []
    for seed in SEEDS:
        params = dict(random_strength=2, random_seed=seed, verbose=0)
        if ttype == "regression":
            ys = YScaler(log=False)
            ys.fit(y_trainval)
            params["loss_function"] = "MAE"
            model = CatBoostRegressor(**params)
            model.fit(X_trainval, ys.transform(y_trainval).reshape(-1))
            pred = ys.inverse(model.predict(X_te))
            sc = metric(y_te, pred, metric_name)
        else:
            params["loss_function"] = "Logloss"
            model = CatBoostClassifier(**params)
            model.fit(X_trainval, y_trainval)
            proba = model.predict_proba(X_te)[:, 1]
            sc = metric(y_te, proba, metric_name)
        scores.append(float(sc))
        print(f"  seed {seed}: {metric_name.upper()} = {sc:.4f}")

    mean = float(np.mean(scores)); std = float(np.std(scores))
    delta = (sota - mean) if metric_name == "mae" else (mean - sota)
    print(f"\n=> {mean:.4f} ± {std:.4f}  (SOTA {sota:.3f}, Δ {delta:+.4f})")

    out = RESULTS / task_key; out.mkdir(exist_ok=True)
    suffix = "_maplight_gin" if use_gin else "_maplight"
    res = {"task": task_key, "tdc_name": tdc_name, "task_type": ttype,
           "metric": metric_name, "sota": sota, "mean": mean, "std": std,
           "seeds": scores, "model": label, "n_train_val": len(X_trainval),
           "n_test": len(X_te), "feature_dim": int(X_trainval.shape[1])}
    with open(out / f"result{suffix}.json", "w") as f:
        json.dump(res, f, indent=2)
    return res


def main():
    global RESULTS_GIN
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=list(TASKS.keys()), choices=list(TASKS))
    p.add_argument("--gin", action="store_true")
    p.add_argument("--gin_dir", default=None, help="Override GIN embeddings directory")
    args = p.parse_args()
    if args.gin_dir:
        RESULTS_GIN = Path(args.gin_dir)

    print(f"MapLight replication | {'FP+GIN' if args.gin else 'FP'} | seeds {SEEDS}")
    all_res = [run_task(t, use_gin=args.gin) for t in args.tasks]

    print("\n" + "=" * 70)
    print(f"{'task':<14}{'metric':<7}{'mean':>9}{'std':>8}{'sota':>8}{'Δ':>9}")
    for r in all_res:
        m = r["metric"]; d = (r["sota"] - r["mean"]) if m == "mae" else (r["mean"] - r["sota"])
        print(f"  {r['task']:<12}{m:<7}{r['mean']:>9.4f}{r['std']:>8.4f}"
              f"{r['sota']:>8.3f}{d:>+9.4f}")


if __name__ == "__main__":
    main()
