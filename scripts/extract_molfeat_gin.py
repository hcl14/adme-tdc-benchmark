"""
Compute the REAL MapLight+GNN GIN fingerprint — Hu et al. 2019 supervised
attribute masking, via molfeat's PretrainedDGLTransformer — for every
train/val/test molecule across the 11 ADME-T tasks, in the .npy format that
train_all.py --gin consumes:

    results/gin/<task>_<split>_emb.npy   (n, 300) float32
    results/gin/<task>_<split>_idx.npy   (n,)     int

This is the exact recipe from Schimunek et al. 2023 (MapLight): their "+GNN"
variant concatenates these 300-dim frozen embeddings to the FP features.

Requires the molfeat/DGL/RDKit env-patches (graphbolt disabled, PandasTools
guarded) so the pretrained GIN loads on this Py3.9 / pandas-2.3.3 / torch-2.8
stack.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit import RDLogger

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_GIN = REPO_ROOT / "results" / "gin"
RESULTS_GIN.mkdir(parents=True, exist_ok=True)

TASKS = {
    "solubility"      : "Solubility_AqSolDB",
    "hia"             : "HIA_Hou",
    "caco2"           : "Caco2_Wang",
    "bioavailability" : "Bioavailability_Ma",
    "bbb"             : "BBB_Martins",
    "pgp"             : "Pgp_Broccatelli",
    "cyp1a2"          : "CYP1A2_Veith",
    "cyp2c19"         : "CYP2C19_Veith",
    "cyp2c9"          : "CYP2C9_Veith",
    "cyp2d6"          : "CYP2D6_Veith",
    "cyp3a4"          : "CYP3A4_Veith",
}


def get_split(tdc_name):
    from tdc.single_pred import ADME
    data = ADME(name=tdc_name)
    split = data.get_split(method="scaffold", frac=[0.7, 0.1, 0.2])
    return split["train"], split["valid"], split["test"]


def embed_split(transformer, smiles_list, batch_size=512):
    """Return (emb (n_valid,300), valid_idx) skipping SMILES RDKit can't parse."""
    valid_idx, valid_smis = [], []
    for i, smi in enumerate(smiles_list):
        if Chem.MolFromSmiles(str(smi)) is not None:
            valid_idx.append(i)
            valid_smis.append(str(smi))
    embs = []
    for s in range(0, len(valid_smis), batch_size):
        chunk = valid_smis[s:s + batch_size]
        e = transformer.transform(chunk)
        embs.append(np.asarray(e, dtype=np.float32))
    emb = np.vstack(embs) if embs else np.zeros((0, 300), dtype=np.float32)
    emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
    return emb, valid_idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", default="gin_supervised_masking",
                   help="molfeat pretrained GIN variant "
                        "(masking=contextPred-style attr masking = MapLight choice)")
    p.add_argument("--pooling", default="mean")
    p.add_argument("--tasks", nargs="+", default=list(TASKS.keys()))
    args = p.parse_args()

    from molfeat.trans.pretrained import PretrainedDGLTransformer
    print(f"Loading pretrained GIN: kind={args.kind} pooling={args.pooling}")
    trans = PretrainedDGLTransformer(kind=args.kind, pooling=args.pooling,
                                     batch_size=512)
    # warm up (finalize weights)
    _ = trans.transform(["CCO"])
    print("GIN ready.\n")

    for tkey in args.tasks:
        tdc_name = TASKS[tkey]
        tr, va, te = get_split(tdc_name)
        for split_name, df in [("train", tr), ("val", va), ("test", te)]:
            emb, idx = embed_split(trans, df["Drug"].tolist())
            np.save(RESULTS_GIN / f"{tkey}_{split_name}_emb.npy", emb)
            np.save(RESULTS_GIN / f"{tkey}_{split_name}_idx.npy",
                    np.array(idx, dtype=np.int64))
        print(f"  {tkey:14s}: train={len(tr)} val={len(va)} test={len(te)} "
              f"(gin emb {emb.shape[1]}-d)")
    print("\nAll MapLight+GNN embeddings saved to", RESULTS_GIN)


if __name__ == "__main__":
    main()
