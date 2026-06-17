"""
Save D-MPNN scaler parameters (desc_mean, desc_std, ymean, ystd) to JSON so
that inference works without needing TDC training data at prediction time.

Run once after training D-MPNN models:
    python scripts/save_dmpnn_scalers.py
"""
from __future__ import annotations

import json
import sys
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dmpnn_model import smiles_to_dmpnn
from features import _DESC_NAMES
from rdkit import Chem
from rdkit.Chem import Descriptors

TASKS = {
    "solubility": "Solubility_AqSolDB",
    "caco2":      "Caco2_Wang",
}


def rdkit_extra(smiles_list):
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
    return np.vstack(rows) if rows else np.empty((0, len(_DESC_NAMES))), valid


def compute_scaler(tdc_name: str) -> dict:
    from tdc.single_pred import ADME
    data = ADME(name=tdc_name)
    split = data.get_split(method="scaffold", frac=[0.7, 0.1, 0.2])
    train_df = split["train"]

    smis = train_df["Drug"].tolist()
    y_raw = train_df["Y"].values.astype(np.float32)

    # find molecules valid for both graph and descriptor computation
    graph_valid = [i for i, s in enumerate(smis) if smiles_to_dmpnn(s) is not None]
    desc, desc_valid = rdkit_extra(smis)
    common = sorted(set(graph_valid) & set(desc_valid))
    dpos = {v: i for i, v in enumerate(desc_valid)}
    desc_train = desc[[dpos[c] for c in common]]
    y_train = y_raw[common]

    return {
        "desc_mean": desc_train.mean(0).tolist(),
        "desc_std":  desc_train.std(0).tolist(),
        "ymean":     float(y_train.mean()),
        "ystd":      float(y_train.std()) or 1.0,
    }


def main():
    for task, tdc_name in TASKS.items():
        out = REPO_ROOT / "results" / task / "dmpnn_scaler.json"
        if out.exists():
            print(f"{task}: scaler already saved at {out}")
            continue
        print(f"{task}: computing scaler from TDC training split...")
        scaler = compute_scaler(tdc_name)
        with open(out, "w") as f:
            json.dump(scaler, f)
        print(f"  saved → {out}  (ymean={scaler['ymean']:.3f}, ystd={scaler['ystd']:.3f})")
    print("Done.")


if __name__ == "__main__":
    main()
