"""
Train CatBoost models for all ADME-T tasks and report metrics matching TDC leaderboard.

Usage:
    # FP features only (baseline):
    python scripts/train_all.py [--tasks TASK1 TASK2 ...] [--n_seeds 3]

    # FP + GIN embeddings (MapLight+GNN replication):
    python scripts/train_all.py --gin [--tasks TASK1 TASK2 ...] [--n_seeds 3]

Each task uses:
  - Scaffold split from TDC (same as benchmark)
  - Feature vector: Morgan(1024) + Avalon(1024) + ErG(315) + RDKit descs [+ GIN 300]
  - CatBoost classifier (Logloss) or regressor (MAE)
  - SqrtBalanced class weights for AUPRC tasks (CYP2C9/2D6/3A4)
  - Results averaged over n_seeds random restarts
"""
from __future__ import annotations

import argparse
import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATASETS    = REPO_ROOT / "datasets"
RESULTS     = REPO_ROOT / "results"
RESULTS_GIN = REPO_ROOT / "results" / "gin"
RESULTS.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from features import smiles_list_to_matrix

# tasks that use AUPRC (class-imbalanced, ~17% positive)
AUPRC_TASKS = {"cyp2c9", "cyp2d6", "cyp3a4"}

# ── task registry ─────────────────────────────────────────────────────────────

TASKS = {
    # name            : (tdc_dataset_name,     task_type,      primary_metric, sota_score)
    "solubility"      : ("Solubility_AqSolDB", "regression",   "mae",          0.761),
    "hia"             : ("HIA_Hou",            "classification","auroc",        0.989),
    "caco2"           : ("Caco2_Wang",         "regression",   "mae",          0.256),
    "bioavailability" : ("Bioavailability_Ma", "classification","auroc",        0.938),
    "bbb"             : ("BBB_Martins",        "classification","auroc",        0.916),
    "pgp"             : ("Pgp_Broccatelli",    "classification","auroc",        0.938),
    "cyp1a2"          : ("CYP1A2_Veith",       "classification","auroc",        0.930),
    "cyp2c19"         : ("CYP2C19_Veith",      "classification","auroc",        0.900),
    "cyp2c9"          : ("CYP2C9_Veith",       "classification","auprc",        0.859),
    "cyp2d6"          : ("CYP2D6_Veith",       "classification","auprc",        0.790),
    "cyp3a4"          : ("CYP3A4_Veith",       "classification","auprc",        0.916),
}

# ── split helper ──────────────────────────────────────────────────────────────

def get_tdc_split(tdc_name: str):
    """Return train/val/test DataFrames using TDC scaffold split."""
    from tdc.single_pred import ADME
    data = ADME(name=tdc_name)
    split = data.get_split(method="scaffold", frac=[0.7, 0.1, 0.2])
    return split["train"], split["valid"], split["test"]

# ── metrics ───────────────────────────────────────────────────────────────────

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    mean_absolute_error,
)


def compute_metric(y_true, y_score, metric: str) -> float:
    if metric == "auroc":
        return roc_auc_score(y_true, y_score)
    if metric == "auprc":
        return average_precision_score(y_true, y_score)
    if metric == "mae":
        return mean_absolute_error(y_true, y_score)
    raise ValueError(f"Unknown metric: {metric}")

# ── feature computation ───────────────────────────────────────────────────────

def df_to_features(df: pd.DataFrame, verbose: bool = False):
    """Return (X, y) from a TDC-format DataFrame (FP features only)."""
    X, valid_idx = smiles_list_to_matrix(df["Drug"].tolist(), verbose=verbose)
    y = df["Y"].values[valid_idx]
    return X, y, valid_idx


def load_gin_embeddings(task_key: str, split: str) -> np.ndarray:
    """Load precomputed 300-dim GIN embeddings from train_gin.py output."""
    emb_path = RESULTS_GIN / f"{task_key}_{split}_emb.npy"
    idx_path = RESULTS_GIN / f"{task_key}_{split}_idx.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"GIN embeddings not found at {emb_path}\n"
            "Run `python scripts/train_gin.py` first."
        )
    emb = np.load(str(emb_path))
    idx = np.load(str(idx_path))
    return emb, idx


def df_to_features_gin(df: pd.DataFrame, task_key: str, split: str,
                       verbose: bool = False):
    """Return (X_combined, y) — FP features concatenated with GIN embeddings."""
    X_fp, y, fp_idx = df_to_features(df, verbose=verbose)

    gin_emb, gin_idx = load_gin_embeddings(task_key, split)

    # Both fp_idx and gin_idx should cover the same valid molecules,
    # but GIN may have failed on additional SMILES (very rare).
    # Build intersection using their common valid indices.
    fp_set  = set(fp_idx)
    gin_set = set(gin_idx.tolist())
    common  = sorted(fp_set & gin_set)

    fp_pos  = {v: i for i, v in enumerate(fp_idx)}
    gin_pos = {v: i for i, v in enumerate(gin_idx.tolist())}

    rows_fp  = [fp_pos[c]  for c in common]
    rows_gin = [gin_pos[c] for c in common]

    X_combined = np.concatenate([X_fp[rows_fp], gin_emb[rows_gin]], axis=1)
    y_combined = df["Y"].values[common]
    return X_combined, y_combined

# ── training ──────────────────────────────────────────────────────────────────

def train_catboost(X_train, y_train, X_val, y_val,
                   task_type: str, seed: int = 42,
                   use_class_weights: bool = False, metric: str = ""):
    """Train a single CatBoost model; returns fitted model.

    Early-stopping metric is chosen to match the leaderboard metric: PRAUC for
    the class-imbalanced AUPRC tasks (CYP2C9/2D6/3A4), AUC for the balanced
    AUROC tasks, MAE for regression.  Optimising the actual metric at early-stop
    time is a meaningful lift on the CYP endpoints.
    """
    from catboost import CatBoostClassifier, CatBoostRegressor

    is_auprc = metric == "auprc"
    if task_type == "classification":
        eval_metric = "PRAUC" if is_auprc else "AUC"
    else:
        eval_metric = "MAE"

    common = dict(
        random_seed=seed,
        random_strength=2,
        verbose=0,
        early_stopping_rounds=80,
        eval_metric=eval_metric,
    )

    # AUPRC tasks benefit from more iterations (PRAUC-convergent) + class weights
    iterations = 4000 if is_auprc else 2000

    if task_type == "classification":
        if use_class_weights:
            # SqrtBalanced helps with ~17% positive rate in CYP2C9/2D6/3A4
            common["auto_class_weights"] = "SqrtBalanced"
        model = CatBoostClassifier(
            loss_function="Logloss",
            iterations=iterations,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            **common,
        )
    else:
        model = CatBoostRegressor(
            loss_function="MAE",
            iterations=iterations,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            **common,
        )

    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    return model


def predict(model, X, task_type: str) -> np.ndarray:
    if task_type == "classification":
        return model.predict_proba(X)[:, 1]
    return model.predict(X)

# ── single task run ───────────────────────────────────────────────────────────

def run_task(task_key: str, n_seeds: int = 3,
             use_gin: bool = False) -> dict:
    tdc_name, task_type, metric, sota = TASKS[task_key]
    use_class_weights = task_key in AUPRC_TASKS
    feat_label = "FP+GIN" if use_gin else "FP"
    print(f"\n{'='*60}")
    print(f"Task: {task_key} ({tdc_name})  [{feat_label}]")
    print(f"Type: {task_type} | Metric: {metric.upper()}{'↓' if metric == 'mae' else '↑'}")
    if use_class_weights:
        print("Class weights: SqrtBalanced")
    print(f"SOTA target: {sota:.3f}")
    print(f"{'='*60}")

    print("Loading TDC scaffold split...")
    train_df, val_df, test_df = get_tdc_split(tdc_name)
    print(f"  train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    print("Computing features...")
    if use_gin:
        X_train, y_train = df_to_features_gin(train_df, task_key, "train")
        X_val,   y_val   = df_to_features_gin(val_df,   task_key, "val")
        X_test,  y_test  = df_to_features_gin(test_df,  task_key, "test")
    else:
        X_train, y_train, _ = df_to_features(train_df)
        X_val,   y_val,   _ = df_to_features(val_df)
        X_test,  y_test,  _ = df_to_features(test_df)
    print(f"  feature dim: {X_train.shape[1]}")

    suffix = "_gin" if use_gin else ""
    model_dir = RESULTS / task_key
    model_dir.mkdir(exist_ok=True)

    test_scores = []
    for seed in range(n_seeds):
        print(f"  Training seed {seed}...")
        model = train_catboost(X_train, y_train, X_val, y_val,
                               task_type, seed=seed,
                               use_class_weights=use_class_weights,
                               metric=metric)

        y_pred = predict(model, X_test, task_type)
        score  = compute_metric(y_test, y_pred, metric)
        test_scores.append(score)
        print(f"    seed {seed}: {metric.upper()} = {score:.4f}")

        model.save_model(str(model_dir / f"catboost{suffix}_seed{seed}.cbm"))

    mean_score = float(np.mean(test_scores))
    std_score  = float(np.std(test_scores))

    direction = "↓" if metric == "mae" else "↑"
    gap = mean_score - sota
    if metric == "mae":
        gap_str = f"+{sota - mean_score:+.4f}" if mean_score <= sota else f"{mean_score - sota:+.4f}"
    else:
        gap_str = f"{gap:+.4f}"

    print(f"\nResult: {mean_score:.4f} ± {std_score:.4f} {direction}")
    print(f"vs SOTA ({sota:.3f}): {gap_str}")

    result = {
        "task": task_key,
        "tdc_name": tdc_name,
        "task_type": task_type,
        "metric": metric,
        "sota": sota,
        "mean": mean_score,
        "std": std_score,
        "seeds": test_scores,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "feature_dim": int(X_train.shape[1]),
        "use_gin": use_gin,
    }
    result_file = model_dir / f"result{suffix}.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    return result

# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(results: list):
    use_gin = any(r.get("use_gin") for r in results)
    feat_label = "FP+GIN(300)" if use_gin else "FP only"
    print("\n" + "="*80)
    print(f"SUMMARY — CatBoost ({feat_label}) vs SOTA")
    print("="*80)
    print(f"{'Task':<18} {'Metric':<8} {'Ours':>8} {'±':>6} {'SOTA':>8} {'Delta':>8}")
    print("-"*60)
    for r in results:
        m     = r["metric"].upper()
        our   = r["mean"]
        std   = r["std"]
        sota  = r["sota"]
        delta = our - sota
        if r["metric"] == "mae":
            delta = sota - our  # positive = better than SOTA for MAE
        print(f"  {r['task']:<16} {m:<8} {our:>8.4f} {std:>6.4f} {sota:>8.4f} {delta:>+8.4f}")
    print("="*80)

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(TASKS.keys()),
                        choices=list(TASKS.keys()),
                        help="Which tasks to run (default: all)")
    parser.add_argument("--n_seeds", type=int, default=3,
                        help="Number of random seeds per task")
    parser.add_argument("--gin", action="store_true",
                        help="Concatenate precomputed GIN embeddings (run train_gin.py first)")
    args = parser.parse_args()

    all_results = []
    for task in args.tasks:
        result = run_task(task, n_seeds=args.n_seeds, use_gin=args.gin)
        all_results.append(result)

    print_summary(all_results)

    suffix = "_gin" if args.gin else ""
    summary_path = RESULTS / f"summary{suffix}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS}/")


if __name__ == "__main__":
    main()
