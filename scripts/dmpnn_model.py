"""
Directed Message-Passing Neural Network (D-MPNN), reimplemented from scratch
with PyTorch Geometric. Faithful to the chemprop / Yang et al. 2019 architecture:

  h^0_{uv} = ReLU(W_i x_u + W_j x_v + b)               # initial bond message
  m^{t+1}_{uv} = ReLU( h^0_{uv} + SUM_{w in N(u)\{v}} m^t_{wu} )
  m_v = SUM_{u in N(v)} m^T_{uv}                        # atom readout
  h_v = ReLU(W_a x_v + W_b m_v)
  c   = SUM_v h_v                                        # graph readout
  y   = FFN(c)

The "exclude reverse edge" trick is implemented as:
  agg[u] = sum of all incoming messages to u  (index_add)
  m_new[e=(u->v)] = ReLU(h0[e] + agg[u] - m[reverse(e)])

MPS-native (pure PyTorch index_add, no torch_scatter).  No DGL, no molfeat.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from rdkit import Chem


# ── atom feature tables (chemprop-style, richer than the GIN set) ────────────

# drug-like elements + "other"
_ATOM_NUMS = [1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 33, 34, 35, 53, 54]
_DEGREES = list(range(6))            # 0-5 + other
_FORMAL_CHARGE = [-2, -1, 0, 1, 2]   # + other
_NUM_HS = list(range(4))             # 0-3 + other
_CHIRAL = list(range(4))             # 0-3 + other
_HYBRID = [Chem.HybridizationType.S,
           Chem.HybridizationType.SP,
           Chem.HybridizationType.SP2,
           Chem.HybridizationType.SP3,
           Chem.HybridizationType.SP3D,
           Chem.HybridizationType.SP3D2]  # + other

ATOM_FEAT_DIM = (len(_ATOM_NUMS) + 1) + (len(_DEGREES) + 1) + (len(_FORMAL_CHARGE) + 1) \
              + (len(_NUM_HS) + 1) + (len(_CHIRAL) + 1) + (len(_HYBRID) + 1) + 2  # aromatic + mass


def _onehot(val, choices):
    vec = [0.0] * (len(choices) + 1)
    try:
        vec[choices.index(val)] = 1.0
    except ValueError:
        vec[-1] = 1.0
    return vec


def atom_features(atom):
    return (
        _onehot(atom.GetAtomicNum(), _ATOM_NUMS) +
        _onehot(atom.GetDegree(), _DEGREES) +
        _onehot(atom.GetFormalCharge(), _FORMAL_CHARGE) +
        _onehot(atom.GetTotalNumHs(), _NUM_HS) +
        _onehot(int(atom.GetChiralTag()), _CHIRAL) +
        _onehot(atom.GetHybridization(), _HYBRID) +
        [float(atom.GetIsAromatic()),
         float(atom.GetMass() * 0.01)]
    )


# bond features: bond type (4) + conjugated + in_ring + stereo (4) = ~11
_BOND_TYPES = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
               Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
BOND_FEAT_DIM = (len(_BOND_TYPES) + 1) + 1 + 1 + 4


def bond_features(bond):
    btype = _onehot(bond.GetBondType(), _BOND_TYPES)
    conj = [float(bond.GetIsConjugated())]
    ring = [float(bond.IsInRing())]
    stereo = [0.0] * 4
    try:
        stereo[int(bond.GetStereo())] = 1.0
    except (ValueError, RuntimeError):
        stereo[0] = 1.0
    return btype + conj + ring + stereo


# ── SMILES → PyG Data with reverse-edge map ─────────────────────────────────

def smiles_to_dmpnn(smiles, y=None):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)

    src, dst, bfeat = [], [], []
    rev = []
    # map (u,v) -> edge index for reverse lookup
    bond_pairs = {}  # (u,v) -> edge_id
    bonds = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        bonds.append((i, j, bf))
        bonds.append((j, i, bf))

    for eid, (i, j, bf) in enumerate(bonds):
        src.append(i); dst.append(j); bfeat.append(bf)
        bond_pairs[(i, j)] = eid

    for eid, (i, j, _) in enumerate(bonds):
        rev.append(bond_pairs[(j, i)])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(bfeat, dtype=torch.float32)
    edge_rev = torch.tensor(rev, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                edge_rev=edge_rev, num_nodes=x.size(0))
    if y is not None:
        data.y = torch.tensor([float(y)], dtype=torch.float32)
    return data


# ── D-MPNN encoder ──────────────────────────────────────────────────────────

def _index_add(values, index, n):
    """Sum `values` into rows [0, n) by `index`. MPS-safe (pure index_add_)."""
    out = values.new_zeros((n,) + values.shape[1:])
    out.index_add_(0, index, values)
    return out


class DMPNNEncoder(nn.Module):
    def __init__(self, atom_dim=ATOM_FEAT_DIM, bond_dim=BOND_FEAT_DIM,
                 hidden=300, depth=3, dropout=0.0):
        super().__init__()
        self.hidden = hidden
        self.depth = depth

        # W_i (source atom), W_j (dest atom + bond features)
        self.W_i = nn.Linear(atom_dim, hidden, bias=False)
        self.W_j = nn.Linear(atom_dim + bond_dim, hidden, bias=True)

        # message update (shared across steps like chemprop)
        self.W_h = nn.Linear(hidden, hidden, bias=True)

        # atom readout
        self.W_o = nn.Linear(atom_dim + hidden, hidden, bias=True)

        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_rev = data.edge_rev
        n = data.num_nodes
        src, dst = edge_index[0], edge_index[1]

        # initial bond message h0_{uv} = ReLU(W_i x_u + W_j [x_v ; bond_feat])
        h0 = F.relu(self.W_i(x[src]) + self.W_j(torch.cat([x[dst], edge_attr], dim=-1)))

        m = h0
        for _ in range(self.depth):
            # sum of messages into each node
            agg = _index_add(m, dst, n)            # (N, hidden)
            # for edge e=(u->v): incoming to u excluding reverse(e)
            incoming = agg[src] - m[edge_rev]
            m = F.relu(h0 + self.W_h(incoming))

        # atom readout
        atom_msg = _index_add(m, dst, n)           # (N, hidden)
        h_atom = F.relu(self.W_o(torch.cat([x, atom_msg], dim=-1)))
        h_atom = F.dropout(h_atom, p=self.dropout, training=self.training)

        # graph readout (sum)
        batch = data.batch
        graph_emb = _index_add(h_atom, batch, int(batch.max().item()) + 1)
        return graph_emb                            # (B, hidden)


class DMPNN(nn.Module):
    """D-MPNN encoder + FFN head (chemprop-style: 2 FFN layers w/ ReLU + dropout).

    Optional `extra_dim` concatenates global molecular descriptors (e.g. RDKit 2D)
    to the graph readout before the FFN — this is the "Chemprop-RDKit" variant that
    achieves SOTA (MAE 0.761) on AqSolDB solubility.
    """

    def __init__(self, hidden=300, ffn_layers=2, dropout=0.0,
                 atom_dim=ATOM_FEAT_DIM, bond_dim=BOND_FEAT_DIM, depth=3,
                 out_dim=1, extra_dim=0):
        super().__init__()
        self.encoder = DMPNNEncoder(atom_dim=atom_dim, bond_dim=bond_dim,
                                    hidden=hidden, depth=depth, dropout=dropout)
        self.extra_dim = extra_dim
        layers = []
        d = hidden + extra_dim
        for _ in range(ffn_layers - 1):
            layers += [nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(d, out_dim)]
        self.ffn = nn.Sequential(*layers)

    def forward(self, data, extra=None):
        emb = self.encoder(data)
        if self.extra_dim and extra is not None:
            emb = torch.cat([emb, extra], dim=-1)
        return self.ffn(emb).squeeze(-1)


if __name__ == "__main__":
    g = smiles_to_dmpnn("CC(=O)Oc1ccccc1C(=O)O", y=1.0)  # aspirin
    print("atom feat dim:", ATOM_FEAT_DIM, "bond feat dim:", BOND_FEAT_DIM)
    print("graph:", g, "| edges:", g.edge_index.shape[1], "| rev:", g.edge_rev.shape)
    from torch_geometric.loader import DataLoader
    m = DMPNN(hidden=64, dropout=0.1, extra_dim=10)
    b = DataLoader([g, smiles_to_dmpnn("CCO", y=0.5)], batch_size=2)
    bb = next(iter(b))
    extra = torch.randn(2, 10)
    print("pred:", m(bb, extra))
    print("params:", sum(p.numel() for p in m.parameters()))
