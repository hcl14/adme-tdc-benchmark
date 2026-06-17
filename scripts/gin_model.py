"""
GIN molecular encoder — PyTorch Geometric, MPS-native (no DGL).

Architecture matches Hu et al. 2019 "Strategies for Pre-training Graph Neural
Networks": 5-layer GIN, 300-dim hidden, batch-norm after each layer.

Atom features (75-dim):
  atom type one-hot  (44)
  degree one-hot     (12)
  H-count one-hot    (10)
  impl. valence      ( 8)
  aromaticity        ( 1)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data

from rdkit import Chem


# ── atom feature tables ──────────────────────────────────────────────────────

_ATOM_NUMS = [
    6, 7, 8, 16, 9, 14, 15, 17, 35, 12, 11, 20, 26, 33, 13,
    53, 5, 23, 19, 81, 70, 51, 50, 47, 46, 27, 34, 22, 30,
    1, 3, 32, 29, 79, 28, 48, 49, 25, 40, 24, 78, 80, 82,
]  # 43 explicit atoms + 1 "other" = 44-dim

_DEGREES   = list(range(11))  # 0-10 + other = 12-dim
_NUM_HS    = list(range(9))   # 0-8  + other = 10-dim
_VALENCES  = list(range(7))   # 0-6  + other =  8-dim


def _one_hot(val, choices: list) -> list:
    vec = [0.0] * (len(choices) + 1)
    try:
        vec[choices.index(val)] = 1.0
    except ValueError:
        vec[-1] = 1.0           # "other"
    return vec


ATOM_FEAT_DIM = 44 + 12 + 10 + 8 + 1   # = 75


def _atom_features(atom) -> list:
    return (
        _one_hot(atom.GetAtomicNum(),          _ATOM_NUMS) +
        _one_hot(atom.GetDegree(),             _DEGREES)   +
        _one_hot(atom.GetTotalNumHs(),         _NUM_HS)    +
        _one_hot(atom.GetImplicitValence(),    _VALENCES)  +
        [float(atom.GetIsAromatic())]
    )


# ── SMILES → PyG Data ────────────────────────────────────────────────────────

def smiles_to_pyg(smiles: str, y=None) -> Data | None:
    """Convert a SMILES string to a PyG Data object (node features only)."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()],
                     dtype=torch.float32)   # (n_atoms, 75)

    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, num_nodes=x.size(0))
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float32)
    return data


# ── GIN encoder ──────────────────────────────────────────────────────────────

class GINEncoder(nn.Module):
    """
    5-layer GIN with batch-norm, 300-dim output per molecule.
    MPS-compatible (pure PyTorch / PyG scatter ops).
    """

    def __init__(self, in_dim: int = ATOM_FEAT_DIM, hidden: int = 300,
                 n_layers: int = 5, dropout: float = 0.5):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        dims = [in_dim] + [hidden] * n_layers
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            mlp = nn.Sequential(
                nn.Linear(d_in, 2 * hidden),
                nn.ReLU(),
                nn.Linear(2 * hidden, d_out),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(d_out))

        self.dropout = dropout
        self.hidden  = hidden

    def forward(self, x, edge_index, batch):
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return global_mean_pool(h, batch)   # (B, hidden)


# ── multi-task predictor ─────────────────────────────────────────────────────

class MultiTaskGIN(nn.Module):
    """
    GIN encoder + one linear head per task.
    Supports mixed regression / classification tasks with masked loss.

    task_types: list of 'clf' or 'reg' per task
    """

    def __init__(self, task_names: list, hidden: int = 300):
        super().__init__()
        self.task_names = task_names
        self.encoder    = GINEncoder(hidden=hidden)
        self.heads      = nn.ModuleList(
            [nn.Linear(hidden, 1) for _ in task_names]
        )

    def forward(self, data):
        emb  = self.encoder(data.x, data.edge_index, data.batch)
        pred = torch.cat([h(emb) for h in self.heads], dim=1)   # (B, T)
        return emb, pred

    def embed(self, data):
        """Return (B, hidden) embeddings without computing predictions."""
        self.eval()
        with torch.no_grad():
            emb, _ = self.forward(data)
        return emb
