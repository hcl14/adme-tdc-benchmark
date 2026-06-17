"""
Molecular feature computation for ADME-T modelling.

Feature vector (≈2574 dims, matches MapLight architecture):
  - Morgan count FP  : 1024 (radius=2, hashed)
  - Avalon count FP  : 1024
  - ErG FP           :  315
  - RDKit 2D descs   :  210
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors
from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint


# ── RDKit descriptor list (all 2D, no 3D) ────────────────────────────────────

_DESC_NAMES = [name for name, _ in Descriptors.descList
               if not name.startswith("Ipc")]   # Ipc can overflow


def _sparse_to_dense(fp, n_bits: int) -> np.ndarray:
    """Convert RDKit sparse int vect (Morgan/Avalon count FP) to dense array."""
    arr = np.zeros(n_bits, dtype=np.float32)
    for idx, cnt in fp.GetNonzeroElements().items():
        arr[idx] = cnt
    return arr


def _mol_to_morgan(mol, n_bits: int = 1024, radius: int = 2) -> np.ndarray:
    fp = rdMolDescriptors.GetHashedMorganFingerprint(mol, radius, nBits=n_bits)
    return _sparse_to_dense(fp, n_bits)


def _mol_to_avalon(mol, n_bits: int = 1024) -> np.ndarray:
    fp = GetAvalonCountFP(mol, nBits=n_bits)
    return _sparse_to_dense(fp, n_bits)


def _mol_to_erg(mol) -> np.ndarray:
    fp = GetErGFingerprint(mol)
    return np.array(fp, dtype=np.float32)


def _mol_to_rdkit_descs(mol) -> np.ndarray:
    calc = Descriptors.CalcMolDescriptors(mol)
    arr = np.array([calc.get(n, 0.0) for n in _DESC_NAMES], dtype=np.float32)
    np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return arr


def smiles_to_features(smiles: str) -> np.ndarray | None:
    """Return concatenated feature vector or None if SMILES is invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.concatenate([
        _mol_to_morgan(mol),
        _mol_to_avalon(mol),
        _mol_to_erg(mol),
        _mol_to_rdkit_descs(mol),
    ])


def smiles_list_to_matrix(smiles_list, verbose: bool = True) -> tuple[np.ndarray, list[int]]:
    """
    Compute features for a list of SMILES strings.

    Returns:
        X      : (n_valid, n_features) float32 array
        valid  : indices of rows with valid SMILES (same order as smiles_list)
    """
    results = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        feat = smiles_to_features(str(smi))
        if feat is not None:
            results.append(feat)
            valid_idx.append(i)
        elif verbose:
            print(f"  [warn] invalid SMILES at index {i}: {smi[:60]}")

    X = np.vstack(results) if results else np.empty((0, 0), dtype=np.float32)
    return X, valid_idx


def feature_dim() -> int:
    """Return the number of features per molecule."""
    smi = "c1ccccc1"   # benzene
    return len(smiles_to_features(smi))


if __name__ == "__main__":
    dim = feature_dim()
    print(f"Feature dimension: {dim}")
    test = ["c1ccccc1", "CC(=O)O", "invalid_smiles", "CCO"]
    X, idx = smiles_list_to_matrix(test)
    print(f"Matrix shape: {X.shape}, valid indices: {idx}")
