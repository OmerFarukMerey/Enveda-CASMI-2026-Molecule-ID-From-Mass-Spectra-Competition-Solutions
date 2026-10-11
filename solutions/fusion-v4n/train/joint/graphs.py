"""Molecular graphs for the GNN structure encoder: SMILES -> categorical atom / bond features (RDKit), and the batch collation.

No torch-geometric: a batch is one disjoint graph (node features [N, 7], directed edges [2, E], bond features [E, 2], graph id per node);
model.GraphEncoder passes messages with index_add_. An unparseable SMILES becomes a one-node dummy graph and is flagged in `g_ok`.
"""
import numpy as np

ELEM = {6: 1, 7: 2, 8: 3, 16: 4, 15: 5, 9: 6, 17: 7, 35: 8, 53: 9, 14: 10, 5: 11, 34: 12, 1: 13}      # 0 = other
A_SIZES = (14, 7, 5, 6, 2, 2, 5)          # element, degree, formal charge (-2..2), hydrogens, aromatic, in ring, hybridisation
B_SIZES = (5, 2)                          # bond type (single, double, triple, aromatic, other), in ring
_BT = {1.0: 0, 2.0: 1, 3.0: 2, 1.5: 3}


def featurize(smiles):
    """(atoms int8 [n, 7], edges int32 [2, 2m], bonds int8 [2m, 2]) or None."""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles) if smiles else None
    if m is None or m.GetNumAtoms() == 0:
        return None
    hy = Chem.rdchem.HybridizationType; H = {hy.SP: 1, hy.SP2: 2, hy.SP3: 3}
    a = np.array([(ELEM.get(x.GetAtomicNum(), 0), min(x.GetDegree(), 6), min(max(x.GetFormalCharge(), -2), 2) + 2, min(x.GetTotalNumHs(), 5), int(x.GetIsAromatic()),
                   int(x.IsInRing()), H.get(x.GetHybridization(), 0)) for x in m.GetAtoms()], np.int8)
    e, b = [], []
    for x in m.GetBonds():
        i, j = x.GetBeginAtomIdx(), x.GetEndAtomIdx(); f = (_BT.get(x.GetBondTypeAsDouble(), 4), int(x.IsInRing()))
        e += [(i, j), (j, i)]; b += [f, f]
    return a, (np.array(e, np.int32).T if e else np.zeros((2, 0), np.int32)), (np.array(b, np.int8) if b else np.zeros((0, 2), np.int8))


def collate(graphs):
    """List of featurize() results (None allowed) -> arrays of one disjoint batch graph."""
    A, E, Bd, gid, ok, off = [], [], [], [], [], 0
    for g, x in enumerate(graphs):
        if x is None:
            x = (np.zeros((1, len(A_SIZES)), np.int8), np.zeros((2, 0), np.int32), np.zeros((0, len(B_SIZES)), np.int8)); ok.append(False)
        else:
            ok.append(True)
        a, e, b = x; A.append(a); E.append(e.astype(np.int64) + off); Bd.append(b); gid.append(np.full(len(a), g, np.int64)); off += len(a)
    return dict(g_atoms=np.concatenate(A).astype(np.int64), g_edges=np.concatenate(E, 1), g_bonds=np.concatenate(Bd).astype(np.int64), g_id=np.concatenate(gid),
                g_n=np.int64(len(graphs)), g_ok=np.array(ok))
