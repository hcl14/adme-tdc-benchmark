"""
Predict ADME-T properties for new molecules using trained models.

Three model types are supported:
  fp      — CatBoost on 2572-dim fingerprint/descriptor features (fastest)
  fp_gin  — CatBoost on FP + 300-dim GIN embeddings (best for classification)
  dmpnn   — D-MPNN graph neural network (best for solubility regression)

Usage examples:
  # Predict all tasks with best model per task (default):
  python scripts/predict_new_molecules.py --input molecules.csv --smiles_col SMILES

  # Use FP+GIN for all classification tasks:
  python scripts/predict_new_molecules.py --input molecules.csv --smiles_col SMILES --model fp_gin

  # Predict only specific tasks:
  python scripts/predict_new_molecules.py --input mols.smi --tasks cyp1a2 cyp2c19 bbb

  # Ensemble over all seeds (more robust, slower):
  python scripts/predict_new_molecules.py --input mols.smi --ensemble

Input format:
  CSV with a SMILES column, or a .smi file (one SMILES per line).

Output: results/predictions.csv with one prediction column per task.
  Classification: probability of active (0–1).
  Regression:     predicted value in original units (log mol/L for solubility).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS   = REPO_ROOT / "results"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from features import smiles_list_to_matrix

# ── task registry ─────────────────────────────────────────────────────────────
#  (task_key, task_type, metric, best_model)
TASKS = {
    "solubility"      : ("regression",      "mae",   "dmpnn"),
    "hia"             : ("classification",  "auroc", "fp_gin"),
    "caco2"           : ("regression",      "mae",   "fp"),
    "bioavailability" : ("classification",  "auroc", "fp_gin"),
    "bbb"             : ("classification",  "auroc", "fp"),
    "pgp"             : ("classification",  "auroc", "fp_gin"),
    "cyp1a2"          : ("classification",  "auroc", "fp_gin"),
    "cyp2c19"         : ("classification",  "auroc", "fp_gin"),
    "cyp2c9"          : ("classification",  "auprc", "fp_gin"),
    "cyp2d6"          : ("classification",  "auprc", "fp_gin"),
    "cyp3a4"          : ("classification",  "auprc", "fp_gin"),
}

# ── GIN embedding extraction ─────────────────────────────────────────────────

_gin_model_cache = None

def _get_gin_model(device=None):
    """Load and cache the from-scratch GIN model."""
    global _gin_model_cache
    if _gin_model_cache is not None:
        return _gin_model_cache

    import torch
    from gin_model import MultiTaskGIN

    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    ckpt = RESULTS / "gin" / "gin_multitask_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"GIN checkpoint not found: {ckpt}\n"
            "Run: python scripts/train_gin.py"
        )

    model = MultiTaskGIN(task_names=list(TASKS.keys()), hidden=300)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device).eval()
    _gin_model_cache = (model, device)
    return _gin_model_cache


def smiles_to_gin_embeddings(smiles_list: list) -> tuple[np.ndarray, list[int]]:
    """Extract 300-dim GIN embeddings for a list of SMILES.

    Returns:
        emb       : (n_valid, 300) float32 array
        valid_idx : original indices that parsed successfully
    """
    import torch
    from torch_geometric.loader import DataLoader
    from gin_model import smiles_to_pyg

    model, device = _get_gin_model()

    graphs, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        g = smiles_to_pyg(str(smi))
        if g is not None:
            graphs.append(g)
            valid_idx.append(i)

    if not graphs:
        return np.zeros((0, 300), dtype=np.float32), []

    loader = DataLoader(graphs, batch_size=256, shuffle=False, num_workers=0)
    embs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb, _ = model(batch)
            embs.append(emb.cpu().numpy().astype(np.float32))
    return np.vstack(embs), valid_idx


# ── CatBoost (FP-only) inference ──────────────────────────────────────────────

def predict_fp(smiles_list: list, tasks: list, seeds: list = None) -> pd.DataFrame:
    """Predict using CatBoost + fingerprint features."""
    from catboost import CatBoostClassifier, CatBoostRegressor

    print("Computing fingerprint features...")
    X, fp_idx = smiles_list_to_matrix(smiles_list, verbose=False)
    n = len(smiles_list)

    out = {"valid": [i in set(fp_idx) for i in range(n)]}
    for task in tasks:
        ttype = TASKS[task][0]
        model_files = sorted((RESULTS / task).glob("catboost_seed*.cbm"))
        if not model_files:
            print(f"  [skip] {task}: no catboost_seed*.cbm found")
            out[task] = [np.nan] * n
            continue
        if seeds:
            model_files = [f for f in model_files
                           if any(f.stem.endswith(f"seed{s}") for s in seeds)]

        preds_per_seed = []
        for mf in model_files:
            m = CatBoostClassifier() if ttype == "classification" else CatBoostRegressor()
            m.load_model(str(mf))
            if ttype == "classification":
                p = m.predict_proba(X)[:, 1]
            else:
                p = m.predict(X)
            preds_per_seed.append(p)

        p_mean = np.mean(preds_per_seed, axis=0)
        col = [np.nan] * n
        for j, i in enumerate(fp_idx):
            col[i] = float(p_mean[j])
        out[task] = col
        print(f"  {task:<16} [{len(preds_per_seed)} seed(s)]")
    return pd.DataFrame(out)


# ── CatBoost (FP+GIN) inference ───────────────────────────────────────────────

def predict_fp_gin(smiles_list: list, tasks: list, seeds: list = None) -> pd.DataFrame:
    """Predict using CatBoost + fingerprint + GIN embeddings."""
    from catboost import CatBoostClassifier, CatBoostRegressor

    print("Computing fingerprint features...")
    X_fp, fp_idx = smiles_list_to_matrix(smiles_list, verbose=False)
    fp_set = set(fp_idx)

    print("Extracting GIN embeddings...")
    X_gin, gin_idx = smiles_to_gin_embeddings(smiles_list)
    gin_set = set(gin_idx)

    # only molecules valid for BOTH fp and gin
    common = sorted(fp_set & gin_set)
    fp_pos  = {v: i for i, v in enumerate(fp_idx)}
    gin_pos = {v: i for i, v in enumerate(gin_idx)}
    X = np.concatenate([X_fp[[fp_pos[c] for c in common]],
                        X_gin[[gin_pos[c] for c in common]]], axis=1)
    print(f"  {len(common)}/{len(smiles_list)} molecules valid | feature dim: {X.shape[1]}")

    n = len(smiles_list)
    out = {"valid": [i in common for i in range(n)]}
    for task in tasks:
        ttype = TASKS[task][0]
        model_files = sorted((RESULTS / task).glob("catboost_gin_seed*.cbm"))
        if not model_files:
            # fall back to FP-only models
            model_files = sorted((RESULTS / task).glob("catboost_seed*.cbm"))
            X_use, idx_use = X_fp, fp_idx
            if not model_files:
                print(f"  [skip] {task}: no models found")
                out[task] = [np.nan] * n
                continue
            print(f"  {task:<16} [no gin models, using fp-only]")
        else:
            X_use, idx_use = X, common
        if seeds:
            model_files = [f for f in model_files
                           if any(f.stem.endswith(f"seed{s}") for s in seeds)]

        preds_per_seed = []
        for mf in model_files:
            m = CatBoostClassifier() if ttype == "classification" else CatBoostRegressor()
            m.load_model(str(mf))
            if ttype == "classification":
                p = m.predict_proba(X_use)[:, 1]
            else:
                p = m.predict(X_use)
            preds_per_seed.append(p)

        p_mean = np.mean(preds_per_seed, axis=0)
        col = [np.nan] * n
        for j, i in enumerate(idx_use):
            col[i] = float(p_mean[j])
        out[task] = col
        print(f"  {task:<16} [{len(preds_per_seed)} seed(s)]")
    return pd.DataFrame(out)


# ── D-MPNN inference ──────────────────────────────────────────────────────────

def predict_dmpnn(smiles_list: list, tasks: list, seeds: list = None) -> pd.DataFrame:
    """Predict using D-MPNN (only supports solubility and caco2)."""
    import torch
    from torch_geometric.loader import DataLoader
    from dmpnn_model import DMPNN, smiles_to_dmpnn, ATOM_FEAT_DIM, BOND_FEAT_DIM
    from features import _DESC_NAMES
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    def rdkit_extra(smis):
        rows, valid = [], []
        for i, smi in enumerate(smis):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                continue
            calc = Descriptors.CalcMolDescriptors(mol)
            arr = np.array([calc.get(nm, 0.0) for nm in _DESC_NAMES], dtype=np.float32)
            np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            rows.append(arr)
            valid.append(i)
        return (np.vstack(rows) if rows else np.empty((0, len(_DESC_NAMES)), dtype=np.float32),
                valid)

    n = len(smiles_list)
    print("Building D-MPNN graphs and RDKit descriptors...")
    graphs, g_idx = [], []
    for i, smi in enumerate(smiles_list):
        g = smiles_to_dmpnn(smi)
        if g is not None:
            graphs.append(g)
            g_idx.append(i)
    desc, d_idx = rdkit_extra(smiles_list)
    common = sorted(set(g_idx) & set(d_idx))
    g_pos = {v: i for i, v in enumerate(g_idx)}
    d_pos = {v: i for i, v in enumerate(d_idx)}
    graphs_c = [graphs[g_pos[c]] for c in common]
    desc_c   = desc[[d_pos[c] for c in common]]
    print(f"  {len(common)}/{n} molecules valid")

    out = {"valid": [i in set(common) for i in range(n)]}
    for task in tasks:
        if task not in ("solubility", "caco2"):
            print(f"  [skip] {task}: D-MPNN only trained for solubility / caco2")
            out[task] = [np.nan] * n
            continue

        # load scaler
        scaler_path = RESULTS / task / "dmpnn_scaler.json"
        if not scaler_path.exists():
            print(f"  [skip] {task}: scaler not found, run scripts/save_dmpnn_scalers.py first")
            out[task] = [np.nan] * n
            continue
        with open(scaler_path) as f:
            sc = json.load(f)
        desc_mean = np.array(sc["desc_mean"], dtype=np.float32)
        desc_std  = np.array(sc["desc_std"],  dtype=np.float32)
        ymean, ystd = sc["ymean"], sc["ystd"]

        desc_n = ((desc_c - desc_mean) / (desc_std + 1e-6)).astype(np.float32)

        model_files = sorted((RESULTS / task).glob("dmpnn_seed*.pt"))
        if not model_files:
            print(f"  [skip] {task}: no dmpnn_seed*.pt found")
            out[task] = [np.nan] * n
            continue
        if seeds:
            model_files = [f for f in model_files
                           if any(f.stem.endswith(f"seed{s}") for s in seeds)]

        preds_per_seed = []
        for mf in model_files:
            model = DMPNN(hidden=300, depth=3, dropout=0.0,
                          atom_dim=ATOM_FEAT_DIM, bond_dim=BOND_FEAT_DIM,
                          extra_dim=desc_n.shape[1]).to(device)
            model.load_state_dict(torch.load(str(mf), map_location=device, weights_only=True))
            model.eval()

            preds_norm = []
            batch_size = 128
            with torch.no_grad():
                for s in range(0, len(graphs_c), batch_size):
                    idxs = list(range(s, min(s + batch_size, len(graphs_c))))
                    loader = DataLoader([graphs_c[i] for i in idxs],
                                        batch_size=len(idxs), num_workers=0)
                    batch = next(iter(loader)).to(device)
                    xb = torch.from_numpy(desc_n[idxs]).to(device)
                    p = model(batch, xb).cpu().numpy()
                    preds_norm.append(p)
            pn = np.concatenate(preds_norm)
            preds_per_seed.append(pn * ystd + ymean)  # back to raw scale

        p_mean = np.mean(preds_per_seed, axis=0)
        col = [np.nan] * n
        for j, i in enumerate(common):
            col[i] = float(p_mean[j])
        out[task] = col
        print(f"  {task:<16} [{len(preds_per_seed)} seed(s)] "
              f"mean={np.nanmean(p_mean):.3f}")
    return pd.DataFrame(out)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predict ADME-T properties for new molecules.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",      required=True,
                        help="CSV or .smi file with molecules")
    parser.add_argument("--smiles_col", default="smiles",
                        help="Column name for SMILES (CSV only, default: smiles)")
    parser.add_argument("--tasks",      nargs="+", default=list(TASKS.keys()),
                        choices=list(TASKS), help="Tasks to predict")
    parser.add_argument("--model",      default="best",
                        choices=["best", "fp", "fp_gin", "dmpnn"],
                        help="Model to use: best (default, picks best per task), "
                             "fp (FP-only CatBoost), fp_gin (FP+GIN CatBoost), "
                             "dmpnn (D-MPNN, regression only)")
    parser.add_argument("--ensemble",   action="store_true",
                        help="Average over all trained seeds (more robust, slower)")
    parser.add_argument("--output",     default=None,
                        help="Output CSV path (default: results/predictions.csv)")
    args = parser.parse_args()

    # ── load input ────────────────────────────────────────────────────────────
    inp = Path(args.input)
    if inp.suffix == ".smi":
        df = pd.read_csv(inp, header=None, names=["smiles"])
        args.smiles_col = "smiles"
    else:
        df = pd.read_csv(inp)

    if args.smiles_col not in df.columns:
        raise ValueError(
            f"Column '{args.smiles_col}' not found in {inp}.\n"
            f"Available columns: {list(df.columns)}"
        )
    smiles_list = df[args.smiles_col].tolist()
    print(f"Loaded {len(smiles_list)} molecules from {inp}")

    seeds = None  # None = use all available seeds

    # ── dispatch ──────────────────────────────────────────────────────────────
    if args.model == "best":
        # Use best-validated model per task:
        #   solubility  → D-MPNN
        #   caco2       → FP-only CatBoost
        #   all others  → FP+GIN CatBoost
        tasks_dmpnn = [t for t in args.tasks if TASKS[t][2] == "dmpnn"]
        tasks_fp    = [t for t in args.tasks if TASKS[t][2] == "fp"]
        tasks_fpgin = [t for t in args.tasks if TASKS[t][2] == "fp_gin"]

        frames = []
        if tasks_dmpnn:
            print(f"\n[D-MPNN] tasks: {tasks_dmpnn}")
            frames.append(predict_dmpnn(smiles_list, tasks_dmpnn, seeds))
        if tasks_fp:
            print(f"\n[FP] tasks: {tasks_fp}")
            frames.append(predict_fp(smiles_list, tasks_fp, seeds))
        if tasks_fpgin:
            print(f"\n[FP+GIN] tasks: {tasks_fpgin}")
            frames.append(predict_fp_gin(smiles_list, tasks_fpgin, seeds))

        # merge: keep 'valid' from FP run (widest coverage)
        if not frames:
            raise RuntimeError("No predictions produced.")
        base = frames[0].copy()
        for f in frames[1:]:
            for col in f.columns:
                if col != "valid":
                    base[col] = f[col]
                else:
                    # mark valid if ANY model succeeded
                    base["valid"] = base["valid"] | f["valid"]
        preds_df = base

    elif args.model == "fp":
        print("\n[FP] all tasks")
        preds_df = predict_fp(smiles_list, args.tasks, seeds)

    elif args.model == "fp_gin":
        print("\n[FP+GIN] all tasks")
        preds_df = predict_fp_gin(smiles_list, args.tasks, seeds)

    elif args.model == "dmpnn":
        print("\n[D-MPNN] all tasks")
        preds_df = predict_dmpnn(smiles_list, args.tasks, seeds)

    # ── output ────────────────────────────────────────────────────────────────
    out_df = pd.concat([df.reset_index(drop=True), preds_df], axis=1)
    out_path = Path(args.output or str(RESULTS / "predictions.csv"))
    out_df.to_csv(out_path, index=False)
    print(f"\nPredictions saved to {out_path}")

    # print quick summary
    print(f"\n{'Task':<18}{'Type':<15}{'#predicted':>10}{'mean':>10}")
    print("-" * 55)
    for task in args.tasks:
        if task not in preds_df.columns:
            continue
        vals = preds_df[task].dropna()
        ttype = TASKS[task][0]
        print(f"  {task:<16}{ttype:<15}{len(vals):>10}{vals.mean():>10.4f}")


if __name__ == "__main__":
    main()
