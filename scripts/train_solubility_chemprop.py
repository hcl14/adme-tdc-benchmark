"""
Train a Chemprop D-MPNN for aqueous solubility (ESOL Log S).

Chemprop outperforms CatBoost on regression tasks with limited data because
D-MPNN learns directly from the molecular graph without explicit FP encoding.

SOTA target: MAE = 0.761 (Chemprop-RDKit, AqSolDB, scaffold split)

Usage:
    python scripts/train_solubility_chemprop.py

Requires chemprop:
    pip install chemprop
"""

import sys
import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS   = REPO_ROOT / "results" / "solubility_chemprop"
RESULTS.mkdir(parents=True, exist_ok=True)

try:
    import chemprop
except ImportError:
    print("chemprop not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "chemprop"])
    import chemprop


def get_split():
    from tdc.single_pred import ADME
    data = ADME(name="Solubility_AqSolDB")
    split = data.get_split(method="scaffold", frac=[0.7, 0.1, 0.2])
    return split["train"], split["valid"], split["test"]


def df_to_chemprop_csv(df: pd.DataFrame, path: Path, target_col: str = "Y"):
    out = df[["Drug", target_col]].rename(columns={"Drug": "smiles", target_col: "logS"})
    out.to_csv(path, index=False)


def train_and_evaluate(seed: int = 0) -> float:
    from sklearn.metrics import mean_absolute_error

    print(f"\nSeed {seed}: loading data...")
    train_df, val_df, test_df = get_split()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        train_csv = tmpdir / "train.csv"
        val_csv   = tmpdir / "val.csv"
        test_csv  = tmpdir / "test.csv"
        model_dir = tmpdir / "model"

        df_to_chemprop_csv(train_df, train_csv)
        df_to_chemprop_csv(val_df,   val_csv)
        df_to_chemprop_csv(test_df,  test_csv)

        # Train via chemprop CLI
        train_args = [
            sys.executable, "-m", "chemprop.cli.main", "train",
            "--data-path",          str(train_csv),
            "--separate-val-path",  str(val_csv),
            "--separate-test-path", str(test_csv),
            "--save-dir",           str(model_dir),
            "--target-columns",     "logS",
            "--task-type",          "regression",
            "--loss-function",      "mse",
            "--metric",             "mae",
            "--num-workers",        "0",
            "--epochs",             "50",
            "--seed",               str(seed),
            "--smiles-columns",     "smiles",
        ]
        print(f"  Training Chemprop (50 epochs)...")
        result = subprocess.run(train_args, capture_output=True, text=True)
        if result.returncode != 0:
            print("TRAIN STDERR:", result.stderr[-2000:])
            raise RuntimeError("Chemprop training failed")

        # Predict on test set
        pred_csv = tmpdir / "preds.csv"
        pred_args = [
            sys.executable, "-m", "chemprop.cli.main", "predict",
            "--test-path",      str(test_csv),
            "--model-path",     str(model_dir),
            "--preds-path",     str(pred_csv),
            "--smiles-columns", "smiles",
            "--num-workers",    "0",
        ]
        result = subprocess.run(pred_args, capture_output=True, text=True)
        if result.returncode != 0:
            print("PREDICT STDERR:", result.stderr[-2000:])
            raise RuntimeError("Chemprop prediction failed")

        preds_df = pd.read_csv(pred_csv)
        y_pred = preds_df["logS"].values
        y_true = test_df["Y"].values
        mae = mean_absolute_error(y_true[:len(y_pred)], y_pred)
        print(f"  Seed {seed} MAE: {mae:.4f}")
        return float(mae)


def main():
    SOTA = 0.761
    n_seeds = 3
    scores = []

    for seed in range(n_seeds):
        try:
            mae = train_and_evaluate(seed)
            scores.append(mae)
        except Exception as e:
            print(f"Seed {seed} failed: {e}")

    if scores:
        mean_mae = float(np.mean(scores))
        std_mae  = float(np.std(scores))
        delta    = SOTA - mean_mae   # positive = better than SOTA

        print(f"\n{'='*60}")
        print(f"Solubility (Chemprop D-MPNN)")
        print(f"MAE: {mean_mae:.4f} ± {std_mae:.4f} ↓")
        print(f"SOTA: {SOTA:.3f}  |  Delta: {delta:+.4f}")
        print(f"{'='*60}")

        result = {
            "task": "solubility",
            "model": "chemprop",
            "metric": "mae",
            "sota": SOTA,
            "mean": mean_mae,
            "std": std_mae,
            "seeds": scores,
        }
        with open(RESULTS / "result.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to {RESULTS}/result.json")


if __name__ == "__main__":
    main()
