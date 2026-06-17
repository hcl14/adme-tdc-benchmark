"""
Extract 300-dim GIN embeddings from a trained checkpoint into the .npy files
that train_all.py --gin expects:

    results/gin/<task>_<split>_emb.npy   (n, 300) float32
    results/gin/<task>_<split>_idx.npy   (n,)     int   valid row indices

Lets you produce FP+GIN features without waiting for train_gin.py to finish its
full loop — point it at the current best checkpoint.  Robust to the .pt being
rewritten by a still-running training process (retries torch.load).
"""
from __future__ import annotations

import argparse
import sys
import time
import numpy as np
import torch
from pathlib import Path
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gin_model import MultiTaskGIN, smiles_to_pyg

RESULTS_GIN = REPO_ROOT / "results" / "gin"
RESULTS_GIN.mkdir(parents=True, exist_ok=True)

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


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_split(tdc_name):
    from tdc.single_pred import ADME
    data = ADME(name=tdc_name)
    split = data.get_split(method="scaffold", frac=[0.7, 0.1, 0.2])
    return split["train"], split["valid"], split["test"]


def load_checkpoint(path, task_names, hidden, device, retries=5):
    for attempt in range(retries):
        try:
            sd = torch.load(path, map_location=device)
            return sd
        except Exception as e:  # file being rewritten by training
            print(f"  [retry {attempt+1}/{retries}] torch.load failed: {e}")
            time.sleep(2)
    raise RuntimeError(f"could not load {path}")


@torch.no_grad()
def extract(model, graphs, device, batch_size=256):
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    out = []
    for batch in loader:
        batch = batch.to(device)
        emb, _ = model(batch)
        out.append(emb.cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, model.encoder.hidden), dtype=np.float32)


def df_to_graphs(df):
    graphs, idx = [], []
    for i, row in enumerate(df.itertuples(index=False)):
        g = smiles_to_pyg(str(row.Drug), y=float(row.Y))
        if g is not None:
            graphs.append(g); idx.append(i)
    return graphs, idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(RESULTS_GIN / "gin_multitask_best.pt"))
    p.add_argument("--hidden", type=int, default=300)
    p.add_argument("--tasks", nargs="+", default=list(TASKS.keys()))
    args = p.parse_args()

    device = get_device()
    print(f"Device: {device}")
    task_names = [t for t in TASKS if t in args.tasks]
    print(f"Loading checkpoint: {args.ckpt}")
    sd = load_checkpoint(args.ckpt, task_names, args.hidden, device)
    # rebuild model with the FULL task set (heads are load-bearing for state_dict keys)
    all_names = list(TASKS.keys())
    model = MultiTaskGIN(task_names=all_names, hidden=args.hidden).to(device)
    model.load_state_dict(sd)
    print("Checkpoint loaded.")

    for tkey in task_names:
        tdc_name, _ = TASKS[tkey]
        tr, va, te = get_split(tdc_name)
        for split_name, df in [("train", tr), ("val", va), ("test", te)]:
            graphs, idx = df_to_graphs(df)
            emb = extract(model, graphs, device)
            np.save(RESULTS_GIN / f"{tkey}_{split_name}_emb.npy", emb)
            np.save(RESULTS_GIN / f"{tkey}_{split_name}_idx.npy", np.array(idx))
        print(f"  {tkey}: train/test emb saved "
              f"({RESULTS_GIN/tkey!s}_*_emb.npy)")
    print("Done.")


if __name__ == "__main__":
    main()
