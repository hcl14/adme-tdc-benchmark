"""
Extract 300-dim embeddings from Hu et al. 2020 pretrained GINs for all 11
ADME-T tasks.  DGL core ops still work despite the graphbolt dylib warning.
Saves to results/gin_pretrained_<kind>/.

Usage:
  python scripts/extract_pretrained_gin.py                         # contextpred (default)
  python scripts/extract_pretrained_gin.py --kind gin_supervised_masking
  python scripts/extract_pretrained_gin.py --tasks solubility caco2
"""
from __future__ import annotations

import argparse
import pickle
import sys
import warnings
import numpy as np
import torch
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path.home() / "Library/Caches/molfeat/gin_supervised_masking/model.save"

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


def load_model(kind="gin_supervised_contextpred"):
    import dgl  # must import before pickle.load so dgllife classes resolve
    import dgllife

    if kind == "gin_supervised_masking" and MODEL_PATH.exists():
        # masking model is stored as a pickle (not a .pth zip)
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        model = dgllife.model.load_pretrained(kind)
    model.eval()
    return model


def featurize_batch(smiles_list):
    """Return (DGL batched graph, valid_indices) for a list of SMILES.
    Uses mol_to_bigraph with canonical_atom_order=False to match molfeat exactly."""
    import dgl
    import datamol as dm
    from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer, mol_to_bigraph

    atom_feat = PretrainAtomFeaturizer()
    bond_feat = PretrainBondFeaturizer()

    graphs, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        try:
            mol = dm.to_mol(str(smi))
            if mol is None:
                continue
            g = mol_to_bigraph(mol, add_self_loop=True, node_featurizer=atom_feat,
                               edge_featurizer=bond_feat, canonical_atom_order=False)
            if g is not None and g.num_nodes() > 0:
                graphs.append(g)
                valid_idx.append(i)
        except Exception:
            pass

    if not graphs:
        return None, []

    return dgl.batch(graphs), valid_idx


@torch.no_grad()
def extract_embeddings(model, smiles_list, batch_size=256):
    """Return (emb array (n, 300), valid_idx list)."""
    import dgl

    all_embs, all_idx = [], []
    offset = 0
    for start in range(0, len(smiles_list), batch_size):
        batch_smis = smiles_list[start:start + batch_size]
        bg, vidx = featurize_batch(batch_smis)
        if bg is None:
            offset += len(batch_smis)
            continue

        node_feats = model(
            bg,
            [bg.ndata["atomic_number"], bg.ndata["chirality_type"]],
            [bg.edata["bond_type"], bg.edata["bond_direction_type"]],
        )  # (total_nodes, 300)

        # mean-pool each graph
        num_nodes_per_graph = bg.batch_num_nodes().tolist()
        split = torch.split(node_feats, num_nodes_per_graph, dim=0)
        graph_embs = torch.stack([s.mean(0) for s in split], dim=0)  # (B, 300)

        all_embs.append(graph_embs.cpu().numpy().astype(np.float32))
        all_idx.extend([offset + i for i in vidx])
        offset += len(batch_smis)

    if not all_embs:
        return np.zeros((0, 300), dtype=np.float32), []
    return np.vstack(all_embs), all_idx


def get_split(tdc_name):
    from tdc.single_pred import ADME
    data = ADME(name=tdc_name)
    split = data.get_split(method="scaffold", frac=[0.7, 0.1, 0.2])
    return split["train"], split["valid"], split["test"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--kind", default="gin_supervised_contextpred",
                   choices=["gin_supervised_contextpred", "gin_supervised_masking",
                            "gin_supervised_infomax", "gin_supervised_edgepred"])
    args = p.parse_args()

    out_dir = REPO_ROOT / "results" / f"gin_pretrained_{args.kind.split('_')[-1]}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading pretrained GIN: {args.kind}")
    model = load_model(args.kind)
    print("Model loaded.")

    for tkey in args.tasks:
        tdc_name = TASKS[tkey]
        tr, va, te = get_split(tdc_name)
        print(f"\n{tkey} ({tdc_name}): train={len(tr)} val={len(va)} test={len(te)}")

        for split_name, df in [("train", tr), ("val", va), ("test", te)]:
            smiles = df["Drug"].astype(str).tolist()
            emb, idx = extract_embeddings(model, smiles, batch_size=args.batch_size)
            np.save(out_dir / f"{tkey}_{split_name}_emb.npy", emb)
            np.save(out_dir / f"{tkey}_{split_name}_idx.npy", np.array(idx, dtype=np.int64))
            print(f"  {split_name}: {emb.shape[0]}/{len(smiles)} valid → "
                  f"{out_dir/f'{tkey}_{split_name}_emb.npy'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
