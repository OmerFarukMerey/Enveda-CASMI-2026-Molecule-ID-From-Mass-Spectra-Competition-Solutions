"""Tests of clean/isoxe2_infer.py (isoxe v2 delivery module) against the training path; no competition data.

usage: python tests/test_isoxe2_infer.py   (or pytest)
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); SOL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SOL, 'clean')); sys.path.insert(0, os.path.join(SOL, 'train', 'isoxe2'))
sys.path.insert(0, os.path.join(SOL, 'train', 'isoxe')); sys.path.insert(0, os.path.join(SOL, 'train'))
import isoxe_feat as X  # noqa: E402
import isoxe2_infer as XI2  # noqa: E402

SMILES = ['CCOC(=O)c1ccccc1', 'COC(=O)c1ccccc1C', 'CC(=O)OCc1ccccc1', 'COc1ccc(C(C)=O)cc1', 'not a smiles']


def _rows():
    mh = 150.06808 + X.PROTON
    return [dict(mz=[77.0386, 105.0335, 123.0441, mh], it=[0.2, 1.0, 0.3, 0.5], prec=mh, adduct='[M+H]+', ionization_mode='positive',
                 instrument='timsTOF Pro', ce=[20.0]),
            dict(mz=[105.0335, 133.0648, 173.0573], it=[1.0, 0.2, 0.6], prec=173.0573, adduct='[M+Na]+', ionization_mode='positive',
                 instrument='Orbitrap', ce=None),
            dict(mz=[93.0346, 121.0295, 149.0608], it=[0.4, 1.0, 0.3], prec=149.0608, adduct='[M-H]-', ionization_mode='negative',
                 instrument='qTOF', ce=[10.0, 30.0])]


def _nets(n, seed=0):
    import torch
    import model2 as M
    out = []
    for i in range(n):
        torch.manual_seed(seed + i)
        out.append(M.XE2(d=32, layers=2, heads=4, ff=64, dg=16, glayers=2).eval())
    return out


def test_network_classes_match_model2():
    import model2 as M
    a = M.XE2(d=32, layers=2, heads=4, ff=64, dg=16, glayers=2)
    b = XI2._classes()(d=32, layers=2, heads=4, ff=64, dg=16, glayers=2)
    assert list(a.state_dict()) == list(b.state_dict())
    b.load_state_dict(a.state_dict(), strict=True)


def test_scores_equal_training_path():
    """isoxe2_infer.score_candidates == build_groups2 packing (float16 storage) + Store2 + train_isoxe2.score_store (CPU)."""
    import torch
    import build_groups2 as B2
    import model2 as M
    import train_isoxe2 as TI
    B2._W['X'] = X
    sps = XI2.prep_spectra(_rows())
    assert len(sps) == 3 and len(sps[0]['mz']) <= XI2.PEAKS
    cands = SMILES[:4]
    D = B2.featurize_groups2([dict(gid=0)], lambda g: (cands, [0] * 4, [1, 0, 0, 0]), lambda g: sps)
    nets = _nets(2)
    with tempfile.TemporaryDirectory() as d:
        B2.save_chunk(os.path.join(d, 'chunk_00000.npz'), D)
        S = M.Store2(d)
        ref = np.mean([TI.score_store(n, S, torch.device('cpu')) for n in nets], 0)
    for chunk in (48, 3):                                    # candidate chunking does not change scores
        mine = XI2.score_candidates(nets, sps, cands, 'cpu', {}, cand_chunk=chunk)
        assert np.allclose(mine, ref, atol=1e-5), (mine, ref)


def test_export_load_nan_and_selftest():
    import torch
    net = _nets(1)[0]
    with tempfile.TemporaryDirectory() as d:
        args = dict(d=32, layers=2, heads=4, ff=64, dropout=0.1, no_graph=False, dg=16, glayers=2, seed=3)
        torch.save(dict(net=net.state_dict(), args=args, epoch=2, val=0.6), os.path.join(d, 'best.pt'))
        cfg = XI2.export_npz(os.path.join(d, 'best.pt'), os.path.join(d, 'm.npz'))
        assert cfg['dg'] == 16 and cfg['peaks'] == XI2.PEAKS
        back = XI2.load_model(os.path.join(d, 'm.npz'), 'cpu')
    sc = XI2.score_candidates([back], XI2.prep_spectra(_rows()), SMILES, 'cpu', {})
    assert np.isnan(sc[-1]) and np.isfinite(sc[:-1]).all()
    assert np.allclose(sc, XI2.score_candidates([net], XI2.prep_spectra(_rows()), SMILES, 'cpu', {}), equal_nan=True)
    exp = [None if v != v else float(v) for v in XI2.selftest_scores([back])]
    assert XI2.selftest([back], exp) == 'ok'
    bad = list(exp); bad[0] = bad[0] + 1.0
    try:
        XI2.selftest([back], bad); raise RuntimeError('selftest did not fail')
    except AssertionError:
        pass
    d = XI2.score_dict([back], _rows(), SMILES, 'cpu', {})
    assert set(d) == set(SMILES[:4])


if __name__ == '__main__':
    for k, v in list(globals().items()):
        if k.startswith('test_'):
            v(); print('ok', k)
