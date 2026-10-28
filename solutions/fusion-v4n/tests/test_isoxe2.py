"""Tests of isoxe v2 (graph-aware featurisation in clean/isoxe_feat.py, train/isoxe2/{build_groups2,model2}.py); no competition data.

usage: python tests/test_isoxe2.py   (or pytest)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); SOL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SOL, 'clean')); sys.path.insert(0, os.path.join(SOL, 'train', 'isoxe2'))
sys.path.insert(0, os.path.join(SOL, 'train', 'isoxe')); sys.path.insert(0, os.path.join(SOL, 'train'))
import isoxe_feat as X  # noqa: E402

ETHYL_BENZOATE = 'CCOC(=O)c1ccccc1'


def _spec(mz, it, prec, adduct='[M+H]+', mode=1, peaks=X.PEAKS):
    return X.prep_spectrum(np.asarray(mz, float), np.asarray(it, float), prec, adduct, mode, [20.0], 0, peaks=peaks)


def test_v1_outputs_unchanged_by_defaults():
    g = X.cand_graph(ETHYL_BENZOATE); F = X.fragments(g)
    mh = 150.06808 + X.PROTON
    sp = _spec([105.03349, mh, 333.3333], [1.0, 0.5, 0.3], mh)
    a1 = X.annotate(F, g, sp); a2 = X.annotate(F, g, sp, bonds=True)
    assert len(a1) == 2 and len(a2) == 3
    assert np.array_equal(a1[0], a2[0]) and np.array_equal(a1[1], a2[1])
    r = X.featurize_candidate(ETHYL_BENZOATE, [sp])
    assert len(r) == 3 and r[1].shape == (1, X.PEAKS, X.N_INT)


def test_graph_features_and_broken_bond_index():
    from rdkit import Chem
    g = X.cand_graph(ETHYL_BENZOATE); F = X.fragments(g)
    m = Chem.MolFromSmiles(ETHYL_BENZOATE)
    assert g['afeat'].shape == (m.GetNumAtoms(), X.N_AF) and g['bfeat'].shape == (m.GetNumBonds(), X.N_BF)
    for j, s in enumerate(X.AF_SIZES):
        assert g['afeat'][:, j].min() >= 0 and g['afeat'][:, j].max() < s
    assert set(g['bfeat'][:, 0].tolist()) == {0, 1, 3}                     # single, double, aromatic
    mh = 150.06808 + X.PROTON
    sp = _spec([105.03349, mh], [1.0, 0.5], mh)
    ai, af, bi = X.annotate(F, g, sp, bonds=True)
    o = np.argsort(sp['mz'])
    b = int(bi[o[0], 0])                                                    # benzoyl: the ester C(=O)-O bond
    bond = m.GetBondWithIdx(b)
    assert {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()} == {'C', 'O'} and not bond.IsInRing()
    assert bond.GetBondTypeAsDouble() == 1.0 and bi[o[0], 1] == -1
    assert X.bond_type_id(bond) == ai[o[0], 3]
    assert (bi[o[1]] == -1).all()                                           # intact precursor: no broken bond


def test_peaks_parameter_and_bonds_featurize():
    mz = np.linspace(50, 140, 60); it = np.linspace(1, 0.01, 60)
    sp = _spec(mz, it, 151.0754, peaks=40)
    assert len(sp['mz']) == 40
    cg, ai, af, ab, gr = X.featurize_candidate(ETHYL_BENZOATE, [sp, sp], peaks=40, bonds=True)
    assert ai.shape == (2, 40, X.N_INT) and ab.shape == (2, 40, 3) and ab.dtype == np.int16
    assert ab.max() < len(gr['ea']) and ab.min() >= -1
    f, mask, meta, ce = X.spectra_block([sp], peaks=40)
    assert f.shape == (1, 40, X.N_SPF) and mask.sum() == 40


def _packed():
    import build_groups2 as B
    mh = 150.06808 + X.PROTON
    sps = [_spec([105.03349, 77.0386, mh], [1.0, 0.4, 0.5], mh, peaks=40), _spec([123.04, mh], [1.0, 0.3], mh, peaks=40)]
    groups = [dict(gid=0, c=[ETHYL_BENZOATE, 'CC(=O)Oc1ccccc1C', 'COC(=O)c1ccccc1C']), dict(gid=1, c=['CCOC(=O)c1ccccc1O', 'COC(=O)c1ccccc1OC'])]
    D = B.featurize_groups2(groups, lambda g: (g['c'], [0] * len(g['c']), [1] + [0] * (len(g['c']) - 1)), lambda g: sps)
    return D


def test_pack_store_gather_roundtrip(tmp=None):
    import tempfile
    import torch
    import build_groups2 as B
    import model2 as M
    D = _packed()
    assert D['g_ncand'].tolist() == [3, 2] and D['ann_b'].shape == D['ann_i'].shape[:2] + (3,)
    with tempfile.TemporaryDirectory() as d:
        B.save_chunk(os.path.join(d, 'chunk_00000.npz'), D)
        S = M.Store2(d)
        assert S.n == 2 and S.smiles[3] == 'CCOC(=O)c1ccccc1O'
        G, own, cmax = S.gather(np.array([1, 0]), [np.array([1, 0]), np.array([0])], torch.device('cpu'))
        assert cmax == 3 and G['n_cand'] == 5 and G['spf'].shape[0] == 2 * 2 + 3 * 1
        # first candidate of the batch = group 1's first candidate; its atoms come first
        na = Chem_atoms('CCOC(=O)c1ccccc1O')
        assert int((G['atom_cand'] == 0).sum()) == na
        # bond indices point into the candidate's own bond rows
        for s in range(G['ann_b'].shape[0]):
            c = int(G['seq_cand'][s]); b = G['ann_b'][s]; b = b[b >= 0]
            if len(b):
                ends = G['bond_e'][b]
                assert (G['atom_cand'][ends.reshape(-1)] == c).all()
        net = M.XE2(d=32, layers=1, heads=2, ff=64, dg=16, glayers=2)
        lg = net(**G)
        assert lg.shape == (7,) and torch.isfinite(lg).all()
        sc = M.group_scores(lg, own, 2, cmax)
        assert torch.isinf(sc[0, 2]) and torch.isfinite(sc[1]).all()
        net0 = M.XE2(d=32, layers=1, heads=2, ff=64, use_graph=False)
        assert net0(**G).shape == (7,)


def Chem_atoms(s):
    from rdkit import Chem
    return Chem.MolFromSmiles(s).GetNumAtoms()


def test_ranges_and_order_pool():
    import build_groups2 as B
    import model2 as M
    assert M.ranges([5, 0, 9], [2, 0, 3]).tolist() == [5, 6, 9, 10, 11]
    smi = {0: 'CCOC(=O)c1ccccc1', 1: 'CCCCCCCCC(=O)O', 2: 'COC(=O)c1ccccc1C', 3: 'CC(=O)OCc1ccccc1', 4: 'CCOC(=O)c1ccccc1'}
    o = B.order_pool('CCOC(=O)c1ccccc1', [1, 2, 3, 4], smi.get, np.random.default_rng(0), k_sim=1, k_rand=1, cap=4)
    assert o[0] == 4 and sorted(o) == [1, 2, 3, 4]


if __name__ == '__main__':
    for k, v in list(globals().items()):
        if k.startswith('test_'):
            v(); print('ok', k)
