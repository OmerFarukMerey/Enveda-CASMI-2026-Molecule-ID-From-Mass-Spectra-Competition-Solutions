"""Tests of the isomer cross-encoder pilot (clean/isoxe_feat.py, train/isoxe/model.py, build_groups packing); no competition data.

usage: python tests/test_isoxe.py   (or pytest)
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); SOL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SOL, 'clean')); sys.path.insert(0, os.path.join(SOL, 'train', 'isoxe')); sys.path.insert(0, os.path.join(SOL, 'train'))
import isoxe_feat as X  # noqa: E402

ETHYL_BENZOATE = 'CCOC(=O)c1ccccc1'          # C9H10O2, 150.0681


def _spec(mz, it, prec, adduct='[M+H]+', mode=1):
    return X.prep_spectrum(np.asarray(mz, float), np.asarray(it, float), prec, adduct, mode, [20.0], 0)


def test_graph_and_intact_mass():
    g = X.cand_graph(ETHYL_BENZOATE)
    F = X.fragments(g)
    assert abs(F['m'][F['k'] == 0][0] - 150.06808) < 1e-4
    assert F['depth'] == 3 and np.all(np.diff(F['m']) > 0)
    assert X.cand_graph('not a smiles') is None and X.cand_graph('') is None


def test_bond_type_ids_stable_and_in_range():
    from rdkit import Chem
    m = Chem.MolFromSmiles(ETHYL_BENZOATE)
    a = [X.bond_type_id(b) for b in m.GetBonds()]
    b = [X.bond_type_id(b) for b in Chem.MolFromSmiles('O=C(OCC)c1ccccc1').GetBonds()]
    assert all(1 <= x < X.NBT for x in a) and sorted(a) == sorted(b)
    g = X.cand_graph(ETHYL_BENZOATE)
    assert len(set(g['bt'].tolist())) >= 4            # aromatic CC, C-C(=O), C=O, C-O ester, O-CH2, CH2-CH3 differ


def test_precursor_and_benzoyl_annotation():
    g = X.cand_graph(ETHYL_BENZOATE); F = X.fragments(g)
    mh = 150.06808 + X.PROTON
    benzoyl = 105.03349                                 # C7H5O+ : ester C-O cleavage, one H fewer on the acyl piece
    junk = 333.3333
    sp = _spec([benzoyl, mh, junk], [1.0, 0.5, 0.3], mh)
    ai, af = X.annotate(F, g, sp)
    order = np.argsort(sp['mz'])                        # prep keeps m/z order: benzoyl, precursor, junk
    k = ai[order, 0]
    assert k[0] == 2 and ai[order[0], 1] == -1 + 3     # one cleavage, dh = -1
    assert k[1] == 1 and ai[order[1], 1] == 3          # intact molecule, dh = 0
    assert k[2] == 0 and ai[order[2], 1] == 7 and af[order[2], 7] == 0
    assert abs(af[order[1], 0]) < 0.1 and np.isclose(af[order[1], 1], 1.0)
    assert ai[order[0], 3] > 0 and ai[order[0], 4] == 0      # the broken bond's type id; no second bond
    assert np.isclose(af[order[0], 1], 8 / 11)


def test_negative_mode_and_adduct_carriers():
    g = X.cand_graph(ETHYL_BENZOATE); F = X.fragments(g)
    mna = 150.06808 + X.AMU['Na'] - X.E
    sp = _spec([mna], [1.0], mna, '[M+Na]+', 1)
    ai, _ = X.annotate(F, g, sp)
    assert ai[0, 0] == 1 and ai[0, 2] == 1             # explained by the intact molecule with the Na carrier
    mmh = 150.06808 - X.PROTON
    sp = _spec([mmh], [1.0], mmh, '[M-H]-', -1)
    ai, _ = X.annotate(F, g, sp)
    assert ai[0, 0] == 1 and ai[0, 1] == 3
    assert X.adduct_index('[M+H]+') == 0 and X.adduct_index('[M+Li]+') == len(X.ADDUCTS)


def test_prep_spectrum_top_peaks_and_padding():
    rng = np.random.default_rng(0)
    mz = np.sort(rng.uniform(50, 400, 100)); it = rng.uniform(0.01, 1, 100)
    sp = _spec(mz, it, 401.0)
    assert len(sp['mz']) == X.PEAKS and np.all(np.diff(sp['mz']) > 0)
    assert set(np.round(sp['it'] * it.max(), 9)) <= set(np.round(it, 9))
    assert np.isclose(sp['it'].max(), 1.0)
    assert X.prep_spectrum(np.zeros(0), np.zeros(0), 100.0, '[M+H]+', 1) is None
    f, mask, meta, ce = X.spectra_block([sp, _spec([100.0], [1.0], 200.0)])
    assert f.shape == (2, X.PEAKS, X.N_SPF) and mask[0].all() and mask[1].sum() == 1 and meta[1, 0] == 0
    r = X.featurize_candidate(ETHYL_BENZOATE, [sp, _spec([100.0], [1.0], 200.0)])
    assert r[1].shape == (2, X.PEAKS, X.N_INT) and (r[1][1, 1:, 1] == 7).all()


def test_clean_spectrum_matches_chemcore():
    import chemcore
    rng = np.random.default_rng(1)
    mz = rng.uniform(20, 600, 700); it = rng.uniform(0, 1, 700) ** 4
    a = X.clean_spectrum(mz, it, 450.0); b = chemcore.clean_peaks(mz, it, 450.0)
    assert np.allclose(a[0], b[0], atol=1e-4) and np.allclose(a[1], b[1], atol=1e-6)


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def test_model_forward_and_group_scores():
    if torch is None:
        return
    import model as M
    net = M.XE(d=32, layers=1, heads=2, ff=64)
    N = 5
    out = net(torch.randn(N, X.PEAKS, X.N_SPF), torch.ones(N, X.PEAKS, dtype=torch.bool), torch.zeros(N, X.PEAKS, X.N_INT, dtype=torch.int16),
              torch.randn(N, X.PEAKS, X.N_FLT), torch.zeros(N, 4, dtype=torch.int16), torch.zeros(N), torch.randn(N, X.N_CAND))
    assert out.shape == (N,)
    lg = torch.tensor([1.0, 3.0, 2.0, 5.0, 7.0])
    own = torch.tensor([0, 0, 1, 2, 2])                 # group 0: cand 0 (2 spectra), cand 1; group 1: cand 0 (2 spectra)
    sc = M.group_scores(lg, own, 2, 2)
    assert sc[0, 0] == 2.0 and sc[0, 1] == 2.0 and sc[1, 0] == 6.0 and torch.isinf(sc[1, 1])
    assert np.isclose(M.isomer_rr(np.array([1.0, 1.0, 0.5]), np.array([1, 0, 0])), 1 / 1.5)
    assert np.isclose(M.isomer_rr(np.array([0.2, 1.0, 0.5]), np.array([1, 0, 0])), 1 / 3)


def test_pack_and_gather_roundtrip():
    """build_groups.featurize_groups packing + model.Store.gather address the right (candidate, spectrum) rows."""
    if torch is None:
        return
    import build_groups as BG
    import model as M
    BG._W['X'] = X
    s1 = _spec([105.03349, 151.0754], [1.0, 0.4], 151.0754)
    s2 = _spec([123.04, 151.0754], [0.2, 1.0], 151.0754)
    groups = [dict(gid=0, c=[ETHYL_BENZOATE, 'CCC(=O)Oc1ccccc1', 'COC(=O)c1ccccc1C'], sp=[s1, s2]),
              dict(gid=1, c=['CC(=O)Oc1ccccc1C', 'bad smiles', ETHYL_BENZOATE], sp=[s2])]
    D = BG.featurize_groups(groups, lambda g: (g['c'], [0] * len(g['c']), [1] + [0] * (len(g['c']) - 1)), lambda g: g['sp'])
    assert list(D['g_ncand']) == [3, 2] and list(D['g_nspec']) == [2, 1]
    with tempfile.TemporaryDirectory() as d:
        np.savez(os.path.join(d, 'chunk_00000.npz'), **{k: v for k, v in D.items() if k != 'cand_smiles'}, cand_smiles=D['cand_smiles'].astype(str))
        S = M.Store(d)
        cr, sr, ar, own, cmax = S.seq_index(np.array([0, 1]), [np.array([1, 0]), np.array([0])])
        assert cmax == 3 and list(own) == [0, 0, 1, 1, 2, 2, 3, 4]
        assert list(sr) == [1, 0, 1, 0, 1, 0, 2, 2] and list(ar) == [1, 0, 3, 2, 5, 4, 6, 7]
        ref = X.featurize_candidate(groups[0]['c'][1], [s1, s2])[1]
        assert np.array_equal(S.ai[ar[2]].numpy(), ref[1]) and np.array_equal(S.ai[ar[3]].numpy(), ref[0])
        assert S.lab.tolist() == [1, 0, 0, 1, 0] and S.smiles[4] == ETHYL_BENZOATE


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print('ok', k)
