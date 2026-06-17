"""
Run Chemprop v1 D-MPNN (the actual SOTA for solubility, MAE 0.761) on TDC
scaffold splits. Uses --features_generator rdkit (the Chemprop-RDKit variant
that achieves SOTA on AqSolDB).

  python scripts/run_chemprop.py --tasks solubility --epochs 50 --n_seeds 3

chemprop v1 CLI: chemprop_train / chemprop_predict (now working after the
RDKit PandasTools env patch). CPU-only (chemprop v1 has no MPS).
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
RESULTS = REPO_ROOT / "results"

TASKS = {
    "solubility": ("Solubility_AqSolDB", 0.761),
    "caco2":      ("Caco2_Wang",         0.256),
}


def get_split(tdc_name):
    from tdc.single_pred import ADME
    data = ADME(name=tdc_name)
    split = data.get_split(method="scaffold", frac=[0.7, 0.1, 0.2])
    return split["train"], split["valid"], split["test"]


def write_csv(df, path):
    out = pd.DataFrame({"smiles": df["Drug"].astype(str), "Y": df["Y"].values})
    out.to_csv(path, index=False)


def run_task(task_key, epochs=50, n_seeds=3, features=True, batch_size=64,
             extra_args=None):
    tdc_name, sota = TASKS[task_key]
    feat = ["--features_generator", "rdkit_2d"] if features else []
    print(f"\n{'='*60}\n{task_key} ({tdc_name}) [Chemprop{'+RDKit' if features else ''}]")
    print(f"MAE↓ | SOTA {sota:.3f} | seeds {n_seeds}\n{'='*60}")

    tr, va, te = get_split(tdc_name)
    print(f"train={len(tr)} val={len(va)} test={len(te)}")

    model_dir = RESULTS / task_key; model_dir.mkdir(exist_ok=True)
    scores = []
    ctrain = str(REPO_ROOT / "venv" / "bin" / "chemprop_train")
    cpredict = str(REPO_ROOT / "venv" / "bin" / "chemprop_predict")
    for seed in range(n_seeds):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_csv(tr, tmp / "train.csv")
            write_csv(va, tmp / "val.csv")
            write_csv(te, tmp / "test.csv")
            md = tmp / "model"
            cmd = [
                ctrain,
                "--data_path", str(tmp / "train.csv"),
                "--separate_val_path", str(tmp / "val.csv"),
                "--separate_test_path", str(tmp / "test.csv"),
                "--dataset_type", "regression",
                "--target_columns", "Y",
                "--save_dir", str(md),
                "--metric", "mae",
                "--loss_function", "mse",
                "--epochs", str(epochs),
                "--batch_size", str(batch_size),
                "--depth", "3", "--hidden_size", "300",
                "--ffn_num_layers", "2", "--dropout", "0.0",
                "--num_workers", "0",
                "--seed", str(seed),
            ] + feat
            if extra_args:
                cmd += extra_args
            print(f"  seed {seed} training...")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print("    FAILED:", r.stderr[-1500:])
                continue
            # chemprop writes test predictions to save_dir/test/test_predictions.csv
            pred_file = md / "test" / "test_predictions.csv"
            if not pred_file.exists():
                # fallback: explicit predict on best checkpoint
                import glob as _g
                ckpts = _g.glob(str(md / "fold_0" / "model_*.pt")) or \
                        _g.glob(str(md / "**" / "model*.pt"), recursive=True)
                pred_file = tmp / "preds.csv"
                pcmd = [cpredict,
                        "--test_path", str(tmp / "test.csv"),
                        "--checkpoint_path", ckpts[0] if ckpts else str(md),
                        "--preds_path", str(pred_file)]
                subprocess.run(pcmd, capture_output=True, text=True)
            preds = pd.read_csv(pred_file)
            y_pred = preds.iloc[:, 1].values
            y_true = te["Y"].values[:len(y_pred)]
            mae = float(np.mean(np.abs(y_pred - y_true)))
            scores.append(mae)
            print(f"    seed {seed} TEST MAE = {mae:.4f}")

    if not scores:
        print("  all seeds failed")
        return None
    mean = float(np.mean(scores)); std = float(np.std(scores))
    print(f"\nResult: MAE = {mean:.4f} ± {std:.4f} ↓  (SOTA {sota:.3f}, Δ {sota-mean:+.4f})")
    result = {"task": task_key, "tdc_name": tdc_name, "metric": "mae", "sota": sota,
              "mean": mean, "std": std, "seeds": scores,
              "model": "chemprop_rdkit" if features else "chemprop",
              "epochs": epochs, "n_train": len(tr), "n_test": len(te)}
    with open(model_dir / "result_chemprop.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["solubility"], choices=list(TASKS))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--no-features", action="store_true")
    args = p.parse_args()

    all_res = []
    for t in args.tasks:
        r = run_task(t, epochs=args.epochs, n_seeds=args.n_seeds,
                     features=not args.no_features, batch_size=args.batch_size)
        if r:
            all_res.append(r)
    print("\n" + "=" * 50)
    for r in all_res:
        print(f"  {r['task']:<12} {r['mean']:.4f}±{r['std']:.4f}  SOTA {r['sota']:.3f}")


if __name__ == "__main__":
    main()
