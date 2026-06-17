"""
Train Chemprop v1 D-MPNN for regression tasks (Solubility and Caco2).

Usage:
    python scripts/train_regression_chemprop.py [--tasks solubility caco2] [--epochs 50]

Outputs:
    results/<task>/chemprop_model/   — trained chemprop model directory
    results/<task>/result_chemprop.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS   = REPO_ROOT / "results"
RESULTS.mkdir(exist_ok=True)

REGRESSION_TASKS = {
    "solubility": ("Solubility_AqSolDB", 0.761),
    "caco2":      ("Caco2_Wang",         0.256),
}


def get_tdc_split(tdc_name: str):
    from tdc.single_pred import ADME
    data = ADME(name=tdc_name)
    split = data.get_split(method="scaffold", frac=[0.7, 0.1, 0.2])
    return split["train"], split["valid"], split["test"]


def write_csv(df: pd.DataFrame, path: Path):
    out = pd.DataFrame({"smiles": df["Drug"], "target": df["Y"]})
    out.to_csv(path, index=False)


def run_chemprop_train(train_csv: Path, val_csv: Path, test_csv: Path,
                       model_dir: Path, epochs: int = 50):
    model_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "chemprop_train",
        "--data_path",          str(train_csv),
        "--separate_val_path",  str(val_csv),
        "--separate_test_path", str(test_csv),
        "--dataset_type",       "regression",
        "--metric",             "mae",
        "--save_dir",           str(model_dir),
        "--epochs",             str(epochs),
        "--hidden_size",        "300",
        "--depth",              "3",
        "--ffn_num_layers",     "2",
        "--batch_size",         "50",
        "--no_cuda",
        "--quiet",
    ]
    print(f"Running chemprop_train (epochs={epochs})...")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"chemprop_train failed (exit {result.returncode})")


def run_chemprop_predict(test_csv: Path, model_dir: Path, preds_csv: Path):
    cmd = [
        "chemprop_predict",
        "--test_path",   str(test_csv),
        "--checkpoint_dir", str(model_dir),
        "--preds_path",  str(preds_csv),
        "--no_cuda",
    ]
    print("Running chemprop_predict...")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"chemprop_predict failed (exit {result.returncode})")


def evaluate_mae(test_csv: Path, preds_csv: Path) -> float:
    test_df  = pd.read_csv(test_csv)
    preds_df = pd.read_csv(preds_csv)
    y_true = test_df["target"].values
    # chemprop v1 preds CSV has header: smiles, target
    y_pred = preds_df.iloc[:, 1].values
    return float(np.mean(np.abs(y_true - y_pred)))


def run_task(task_key: str, epochs: int = 50) -> dict:
    tdc_name, sota = REGRESSION_TASKS[task_key]
    print(f"\n{'='*60}")
    print(f"Task: {task_key} ({tdc_name})  [Chemprop D-MPNN v1]")
    print(f"Metric: MAE↓  |  SOTA: {sota:.3f}")
    print(f"{'='*60}")

    train_df, val_df, test_df = get_tdc_split(tdc_name)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    model_dir = RESULTS / task_key / "chemprop_model"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        train_csv = tmp / "train.csv"
        val_csv   = tmp / "val.csv"
        test_csv  = tmp / "test.csv"
        preds_csv = tmp / "preds.csv"

        write_csv(train_df, train_csv)
        write_csv(val_df,   val_csv)
        write_csv(test_df,  test_csv)

        run_chemprop_train(train_csv, val_csv, test_csv, model_dir, epochs=epochs)
        run_chemprop_predict(test_csv, model_dir, preds_csv)

        mae = evaluate_mae(test_csv, preds_csv)

    direction = "better" if mae <= sota else "worse"
    print(f"\nResult: MAE = {mae:.4f} ↓  ({direction} than SOTA {sota:.3f})")

    result = {
        "task":      task_key,
        "tdc_name":  tdc_name,
        "task_type": "regression",
        "metric":    "mae",
        "sota":      sota,
        "mae":       mae,
        "model":     "chemprop_dmpnn_v1",
        "n_train":   len(train_df),
        "n_val":     len(val_df),
        "n_test":    len(test_df),
    }
    out_dir = RESULTS / task_key
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "result_chemprop.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+",
                        default=list(REGRESSION_TASKS.keys()),
                        choices=list(REGRESSION_TASKS.keys()))
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    all_results = []
    for task in args.tasks:
        result = run_task(task, epochs=args.epochs)
        all_results.append(result)

    print("\n" + "="*60)
    print("SUMMARY — Chemprop D-MPNN vs SOTA")
    print("="*60)
    print(f"{'Task':<16} {'MAE':>8} {'SOTA':>8} {'Delta':>8}")
    print("-"*48)
    for r in all_results:
        delta = r["sota"] - r["mae"]
        print(f"  {r['task']:<14} {r['mae']:>8.4f} {r['sota']:>8.4f} {delta:>+8.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
