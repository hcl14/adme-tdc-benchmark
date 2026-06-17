"""
Train a chemprop-style D-MPNN (reimplemented from scratch, see dmpnn_model.py)
on regression ADME-T tasks. SOTA path for solubility (MAE 0.761, Chemprop-RDKit).

  python scripts/train_dmpnn.py --tasks solubility --epochs 80 --n_seeds 3
  python scripts/train_dmpnn.py --tasks solubility caco2 --epochs 80

Architecture = Chemprop-RDKit:
  D-MPNN graph readout (300) + RDKit 2D descriptors (~209) -> FFN(2) -> y
  depth=3, hidden=300, dropout=0.0, AdamW, early stop on val MAE.

MPS-native. Run AFTER train_gin.py (to avoid GPU contention) or standalone.
"""
from __future__ import annotations

import argparse
import json
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dmpnn_model import DMPNN, smiles_to_dmpnn, ATOM_FEAT_DIM, BOND_FEAT_DIM
from features import _DESC_NAMES
from rdkit import Chem
from rdkit.Chem import Descriptors

RESULTS = REPO_ROOT / "results"
TASKS = {
    "solubility": ("Solubility_AqSolDB", 0.761),
    "caco2":      ("Caco2_Wang",         0.256),
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_tdc_split(tdc_name):
    from tdc.single_pred import ADME
    data = ADME(name=tdc_name)
    split = data.get_split(method="scaffold", frac=[0.7, 0.1, 0.2])
    return split["train"], split["valid"], split["test"]


def rdkit_extra(smiles_list):
    """Return (X_desc (n,~209), valid_idx) RDKit 2D descriptor matrix."""
    rows, valid = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        calc = Descriptors.CalcMolDescriptors(mol)
        arr = np.array([calc.get(n, 0.0) for n in _DESC_NAMES], dtype=np.float32)
        np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        rows.append(arr)
        valid.append(i)
    return np.vstack(rows), valid


class Scaler:
    def __init__(self):
        self.desc_mean = None; self.desc_std = None
        self.ymean = 0.0; self.ystd = 1.0


def build_split(df, scaler, fit=False):
    """graphs, desc(normalized), y_raw (raw scale). Graph .y stored normalized."""
    smis = df["Drug"].tolist()
    graphs, gidx = [], []
    for i, smi in enumerate(smis):
        g = smiles_to_dmpnn(smi)
        if g is not None:
            graphs.append(g); gidx.append(i)
    desc, didx = rdkit_extra(smis)
    common = sorted(set(gidx) & set(didx))
    gpos = {v: i for i, v in enumerate(gidx)}
    dpos = {v: i for i, v in enumerate(didx)}
    graphs = [graphs[gpos[c]] for c in common]
    desc = desc[[dpos[c] for c in common]]
    y_raw = df["Y"].values[common].astype(np.float32)

    if fit:
        scaler.desc_mean = desc.mean(0)
        scaler.desc_std = desc.std(0)
        scaler.ymean = float(y_raw.mean())
        scaler.ystd = float(y_raw.std()) or 1.0

    desc_n = ((desc - scaler.desc_mean) / (scaler.desc_std + 1e-6)).astype(np.float32)
    y_norm = (y_raw - scaler.ymean) / scaler.ystd
    for gg, yy in zip(graphs, y_norm):
        gg.y = torch.tensor([float(yy)], dtype=torch.float32)
    return graphs, desc_n, y_raw


def predict_raw(model, graphs, desc_n, scaler, device, batch_size=128):
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(graphs), batch_size):
            idxs = list(range(s, min(s + batch_size, len(graphs))))
            bg = DataLoader([graphs[i] for i in idxs], batch_size=len(idxs))
            batch = next(iter(bg)).to(device)
            xb = torch.from_numpy(desc_n[idxs]).to(device)
            p = model(batch, xb).cpu().numpy()
            preds.append(p)
    pn = np.concatenate(preds)
    return pn * scaler.ystd + scaler.ymean   # back to raw scale


def train_one(train_g, train_x, val_g, val_x, val_y_raw, scaler,
              epochs=80, hidden=300, depth=3, dropout=0.0, lr=1e-3,
              weight_decay=1e-6, seed=0, device=torch.device("cpu"),
              batch_size=64, extra_dim=0, patience=20, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    model = DMPNN(hidden=hidden, depth=depth, dropout=dropout,
                  atom_dim=ATOM_FEAT_DIM, bond_dim=BOND_FEAT_DIM,
                  extra_dim=extra_dim).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = ExponentialLR(opt, gamma=0.96)

    best_mae = float("inf"); best_state = None; bad = 0
    n = len(train_g)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n).tolist()
        running = 0.0; nb = 0
        for s in range(0, n, batch_size):
            idxs = perm[s:s + batch_size]
            bg = DataLoader([train_g[i] for i in idxs], batch_size=len(idxs))
            batch = next(iter(bg)).to(device)
            xb = torch.from_numpy(train_x[idxs]).to(device)
            yb = batch.y.squeeze(-1)
            opt.zero_grad()
            loss = F.l1_loss(model(batch, xb), yb)
            loss.backward(); opt.step()
            running += loss.item(); nb += 1
        sched.step()

        pred_raw = predict_raw(model, val_g, val_x, scaler, device)
        mae = float(np.mean(np.abs(pred_raw - val_y_raw)))
        if mae < best_mae - 1e-5:
            best_mae = mae; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"    epoch {epoch:3d}  train_l1={running/nb:.4f}  val_mae={mae:.4f}  best={best_mae:.4f}")
        if bad >= patience:
            if verbose:
                print(f"    early stop @ {epoch} (best val_mae={best_mae:.4f})")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_mae


def run_task(task_key, epochs=80, n_seeds=3, device=torch.device("cpu"),
             hidden=300, depth=3, dropout=0.0, lr=1e-3, batch_size=64,
             extra=True, patience=20):
    tdc_name, sota = TASKS[task_key]
    print(f"\n{'='*60}\nTask: {task_key} ({tdc_name})  [D-MPNN{'+RDKit' if extra else ''}]")
    print(f"MAE↓ | SOTA {sota:.3f}\n{'='*60}")

    tr, va, te = get_tdc_split(tdc_name)
    print(f"train={len(tr)} val={len(va)} test={len(te)}")

    scaler = Scaler()
    tg, tx, _ = build_split(tr, scaler, fit=True)
    vg, vx, vy_raw = build_split(va, scaler, fit=False)
    eg, ex, ey_raw = build_split(te, scaler, fit=False)
    extra_dim = tx.shape[1] if extra else 0
    print(f"graph feat: atom={ATOM_FEAT_DIM} bond={BOND_FEAT_DIM} extra_desc={extra_dim}")

    model_dir = RESULTS / task_key; model_dir.mkdir(exist_ok=True)
    scores = []
    for seed in range(n_seeds):
        print(f"  seed {seed}...")
        model, val_mae = train_one(
            tg, tx, vg, vx, vy_raw, scaler, epochs=epochs, hidden=hidden,
            depth=depth, dropout=dropout, lr=lr, seed=seed, device=device,
            batch_size=batch_size, extra_dim=extra_dim, patience=patience)
        pred = predict_raw(model, eg, ex, scaler, device)
        mae = float(np.mean(np.abs(pred - ey_raw)))
        scores.append(mae)
        print(f"    seed {seed} TEST MAE = {mae:.4f} (val {val_mae:.4f})")
        torch.save(model.state_dict(), model_dir / f"dmpnn_seed{seed}.pt")

    mean = float(np.mean(scores)); std = float(np.std(scores))
    delta = sota - mean
    print(f"\nResult: MAE = {mean:.4f} ± {std:.4f} ↓  (SOTA {sota:.3f}, delta {delta:+.4f})")
    result = {"task": task_key, "tdc_name": tdc_name, "task_type": "regression",
              "metric": "mae", "sota": sota, "mean": mean, "std": std, "seeds": scores,
              "model": "dmpnn_rdkit" if extra else "dmpnn", "n_train": len(tg),
              "n_val": len(vg), "n_test": len(eg)}
    with open(model_dir / "result_dmpnn.json", "w") as f:
        json.dump(result, f, indent=2)
    # save scaler params so inference works without TDC at prediction time
    scaler_data = {
        "desc_mean": scaler.desc_mean.tolist(),
        "desc_std":  scaler.desc_std.tolist(),
        "ymean":     scaler.ymean,
        "ystd":      scaler.ystd,
    }
    with open(model_dir / "dmpnn_scaler.json", "w") as f:
        json.dump(scaler_data, f)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["solubility"], choices=list(TASKS))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--hidden", type=int, default=300)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--no-extra", action="store_true", help="disable RDKit extra features")
    args = p.parse_args()

    device = get_device()
    print(f"Device: {device}")
    all_res = []
    for t in args.tasks:
        all_res.append(run_task(t, epochs=args.epochs, n_seeds=args.n_seeds,
                                device=device, hidden=args.hidden, depth=args.depth,
                                dropout=args.dropout, lr=args.lr,
                                batch_size=args.batch_size,
                                extra=not args.no_extra, patience=args.patience))
    print("\n" + "=" * 50)
    print(f"{'Task':<14}{'MAE':>8}{'±':>8}{'SOTA':>8}{'Δ':>8}")
    for r in all_res:
        print(f"  {r['task']:<12}{r['mean']:>8.4f}{r['std']:>8.4f}"
              f"{r['sota']:>8.3f}{r['sota']-r['mean']:>+8.4f}")


if __name__ == "__main__":
    main()
