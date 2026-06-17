"""
Train a multi-task GIN on all ADME-T training splits, then extract 300-dim
embeddings for every train/val/test molecule and save them as .npy files.

Memory-efficient: SMILES→graph conversion happens on-the-fly per batch
(lazy dataset), so we never hold all 200k+ PyG Data objects in RAM at once.

Usage:
    python scripts/train_gin.py [--epochs 100] [--batch_size 64] [--lr 1e-3]

Outputs:
    results/gin/gin_multitask.pt          — trained model weights
    results/gin/<task>_train_emb.npy      — (n_train, 300) float32
    results/gin/<task>_val_emb.npy
    results/gin/<task>_test_emb.npy
    results/gin/<task>_train_idx.npy      — valid row indices (into split df)
    results/gin/<task>_val_idx.npy
    results/gin/<task>_test_idx.npy
"""
from __future__ import annotations

import argparse
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gin_model import MultiTaskGIN, smiles_to_pyg

RESULTS_GIN = REPO_ROOT / "results" / "gin"
RESULTS_GIN.mkdir(parents=True, exist_ok=True)

# ── task registry ────────────────────────────────────────────────────────────

TASKS = {
    "solubility"      : ("Solubility_AqSolDB", "regression"),
    "hia"             : ("HIA_Hou",            "classification"),
    "caco2"           : ("Caco2_Wang",         "regression"),
    "bioavailability" : ("Bioavailability_Ma", "classification"),
    "bbb"             : ("BBB_Martins",        "classification"),
    "pgp"             : ("Pgp_Broccatelli",    "classification"),
    "cyp1a2"          : ("CYP1A2_Veith",       "classification"),
    "cyp2c19"         : ("CYP2C19_Veith",      "classification"),
    "cyp2c9"          : ("CYP2C9_Veith",       "classification"),
    "cyp2d6"          : ("CYP2D6_Veith",       "classification"),
    "cyp3a4"          : ("CYP3A4_Veith",       "classification"),
}

# ── device ───────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ── TDC split loading ────────────────────────────────────────────────────────

def get_tdc_split(tdc_name: str):
    from tdc.single_pred import ADME
    data = ADME(name=tdc_name)
    split = data.get_split(method="scaffold", frac=[0.7, 0.1, 0.2])
    return split["train"], split["valid"], split["test"]

# ── lazy dataset (SMILES → graph on demand) ──────────────────────────────────

class LazyMolDataset(TorchDataset):
    """
    Stores only SMILES strings and labels in RAM.
    Converts to PyG Data object per __getitem__ call.
    valid_indices maps positions in this dataset back to original df rows.
    """

    def __init__(self, smiles_list, labels, task_type: str):
        self.smiles     = []
        self.labels     = []
        self.valid_idx  = []   # index into original df

        scaler_mean, scaler_std = 0.0, 1.0
        if task_type == "regression":
            ys = np.array(labels, dtype=np.float32)
            scaler_mean = float(np.mean(ys))
            scaler_std  = float(np.std(ys)) or 1.0
        self.scaler_mean = scaler_mean
        self.scaler_std  = scaler_std

        for i, (smi, y) in enumerate(zip(smiles_list, labels)):
            mol_ok = smiles_to_pyg(str(smi)) is not None
            if mol_ok:
                self.smiles.append(str(smi))
                y_norm = (float(y) - scaler_mean) / scaler_std
                self.labels.append(y_norm)
                self.valid_idx.append(i)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        g = smiles_to_pyg(self.smiles[idx], y=self.labels[idx])
        return g

    def apply_scaler(self, other: "LazyMolDataset"):
        """Copy scaler from train set to val/test."""
        other.scaler_mean = self.scaler_mean
        other.scaler_std  = self.scaler_std
        other.labels = [
            (l * other.scaler_std + other.scaler_mean - self.scaler_mean) / self.scaler_std
            if other.scaler_std != 1.0 else l
            for l in other.labels
        ]
        # Re-normalise using train scaler
        other.labels = [
            (float(orig_y) - self.scaler_mean) / self.scaler_std
            for orig_y, smi in zip(
                [float(l) * other.scaler_std + other.scaler_mean for l in other.labels],
                other.smiles
            )
        ]


def df_to_lazy(df, task_type: str):
    return LazyMolDataset(df["Drug"].tolist(), df["Y"].tolist(), task_type)


def apply_train_scaler(train_ds: LazyMolDataset, other_ds: LazyMolDataset):
    """Re-normalise val/test labels using train set mean/std."""
    other_ds.scaler_mean = train_ds.scaler_mean
    other_ds.scaler_std  = train_ds.scaler_std
    raw_ys = [float(l) for l in other_ds.labels]   # currently unnormalised originals
    other_ds.labels = [(y - train_ds.scaler_mean) / train_ds.scaler_std for y in raw_ys]

# ── per-task loss ─────────────────────────────────────────────────────────────

def task_loss(pred, y, task_type: str):
    if task_type == "classification":
        return F.binary_cross_entropy_with_logits(pred, y)
    return F.l1_loss(pred, y)

# ── training / eval loops (per-task batches, shared encoder) ─────────────────

def train_epoch(model, loaders_types, optimiser, device):
    model.train()
    total, n = 0.0, 0
    for t_idx, (loader, ttype) in enumerate(loaders_types):
        for batch in loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            emb, preds = model(batch)
            p = preds[:, t_idx].squeeze(-1)
            y = batch.y.squeeze(-1)
            loss = task_loss(p, y, ttype)
            loss.backward()
            optimiser.step()
            total += loss.item()
            n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loaders_types, device):
    model.eval()
    total, n = 0.0, 0
    for t_idx, (loader, ttype) in enumerate(loaders_types):
        for batch in loader:
            batch = batch.to(device)
            emb, preds = model(batch)
            p = preds[:, t_idx].squeeze(-1)
            y = batch.y.squeeze(-1)
            loss = task_loss(p, y, ttype)
            total += loss.item()
            n += 1
    return total / max(n, 1)

# ── embedding extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, dataset, device, batch_size=256):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embs = []
    for batch in loader:
        batch = batch.to(device)
        emb, _ = model(batch)
        embs.append(emb.cpu().numpy())
    if embs:
        return np.vstack(embs)
    return np.zeros((0, model.encoder.hidden), dtype=np.float32)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=15)
    parser.add_argument("--hidden",     type=int,   default=300)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Device: {device}")

    task_names = list(TASKS.keys())
    task_types = [TASKS[k][1] for k in task_names]

    # ── load splits and build lazy datasets ───────────────────────────────────
    print("\nLoading TDC splits and validating SMILES (lazy — no graph tensors yet)...")
    train_sets, val_sets, test_sets = {}, {}, {}

    for tkey, (tname, ttype) in TASKS.items():
        print(f"  {tkey}...", end=" ", flush=True)
        train_df, val_df, test_df = get_tdc_split(tname)

        tr = df_to_lazy(train_df, ttype)
        va = LazyMolDataset(val_df["Drug"].tolist(),  val_df["Y"].tolist(),  ttype)
        te = LazyMolDataset(test_df["Drug"].tolist(), test_df["Y"].tolist(), ttype)

        # For regression: re-normalise val/test with train stats
        if ttype == "regression":
            # val/test were normalised with their own stats; redo with train's
            va_raw = val_df["Y"].tolist()
            te_raw = test_df["Y"].tolist()
            va.labels = [(float(y) - tr.scaler_mean) / tr.scaler_std for y in va_raw]
            te.labels = [(float(y) - tr.scaler_mean) / tr.scaler_std for y in te_raw]
            va.scaler_mean, va.scaler_std = tr.scaler_mean, tr.scaler_std
            te.scaler_mean, te.scaler_std = tr.scaler_mean, tr.scaler_std

        train_sets[tkey] = tr
        val_sets[tkey]   = va
        test_sets[tkey]  = te
        print(f"train={len(tr)} val={len(va)} test={len(te)}")

    # ── data loaders (num_workers=0 avoids MPS/fork issues) ──────────────────
    train_loaders = [
        DataLoader(train_sets[k], batch_size=args.batch_size, shuffle=True,  num_workers=0)
        for k in task_names
    ]
    val_loaders = [
        DataLoader(val_sets[k],   batch_size=args.batch_size, shuffle=False, num_workers=0)
        for k in task_names
    ]
    train_lt = list(zip(train_loaders, task_types))
    val_lt   = list(zip(val_loaders,   task_types))

    # ── model + optimiser ─────────────────────────────────────────────────────
    model     = MultiTaskGIN(task_names=task_names, hidden=args.hidden).to(device)
    optimiser = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimiser, mode="min", factor=0.5,
                                  patience=5, min_lr=1e-5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_params:,} parameters  |  up to {args.epochs} epochs  |  patience={args.patience}\n")

    best_val  = float("inf")
    best_ep   = 0
    no_imp    = 0
    history   = []

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_lt, optimiser, device)
        va_loss = eval_epoch(model, val_lt, device)
        scheduler.step(va_loss)
        lr_now  = optimiser.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train": tr_loss, "val": va_loss})

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs}  train={tr_loss:.4f}  val={va_loss:.4f}  lr={lr_now:.2e}")

        if va_loss < best_val - 1e-5:
            best_val = va_loss
            best_ep  = epoch
            no_imp   = 0
            torch.save(model.state_dict(), RESULTS_GIN / "gin_multitask_best.pt")
        else:
            no_imp += 1
            if no_imp >= args.patience:
                print(f"\nEarly stop at epoch {epoch}  (best val={best_val:.4f} @ ep {best_ep})")
                break

    torch.save(model.state_dict(), RESULTS_GIN / "gin_multitask.pt")
    with open(RESULTS_GIN / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nModel saved → {RESULTS_GIN}/gin_multitask.pt")

    # Reload best weights for embedding extraction
    model.load_state_dict(torch.load(RESULTS_GIN / "gin_multitask_best.pt",
                                      map_location=device))

    # ── extract and save embeddings ───────────────────────────────────────────
    print("\nExtracting embeddings...")
    for tkey in task_names:
        for split_name, ds in [
            ("train", train_sets[tkey]),
            ("val",   val_sets[tkey]),
            ("test",  test_sets[tkey]),
        ]:
            emb = extract_embeddings(model, ds, device)
            idx = np.array(ds.valid_idx)
            np.save(RESULTS_GIN / f"{tkey}_{split_name}_emb.npy", emb)
            np.save(RESULTS_GIN / f"{tkey}_{split_name}_idx.npy", idx)
            print(f"  {tkey}/{split_name}: {emb.shape}")

    print(f"\nAll embeddings saved to {RESULTS_GIN}/")
    print("Done.")


if __name__ == "__main__":
    main()
