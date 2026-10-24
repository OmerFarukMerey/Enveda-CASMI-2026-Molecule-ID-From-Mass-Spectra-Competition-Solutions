"""Tests of clean/isoxe_infer.py (isoxe inference for the s1x frag-gap switch); no competition data, no trained weights.

Checks: cleaning / instrument rules = chemcore / specdata; the copied network = train/isoxe/model.py XE; checkpoint -> npz -> model
round trip; end-to-end equality with the training path (build_groups packing + Store + train_isoxe.score_store on CPU); spectrum
preparation details; NaN for unparsable candidates; self-test pass / fail.
usage: python tests/test_isoxe_infer.py   (or pytest)
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); SOL = os.path.dirname(HERE)
for p in (os.path.join(SOL, 'clean'), os.path.join(SOL, 'train', 'isoxe'), os.path.join(SOL, 'train')):
    sys.path.insert(0, p)
import isoxe_infer as XI  # noqa: E402
import isoxe_feat as X  # noqa: E402

SMILES = ['COC(=O)c1ccccc1O', 'COc1ccccc1C(=O)O', 'COc1ccc(C(=O)O)cc1', 'CC(=O)Oc1ccccc1', 'xx not smiles']


def _rows():
    return [dict(r) for r in XI.SELFTEST_ROWS]


def _nets(n=2, seed=0):
    import torch
    out = []
    for i in range(n):
        torch.manual_seed(seed + i)
        net = XI._xe_class()(d=32, layers=2, heads=4, ff=64, dropout=0.1)
        out.append(net.eval())
    return out


def test_clean_peaks_matches_chemcore():
    import chemcore
    rng = np.random.default_rng(0)
    for n in (0, 3, 40, 700):
        mz = np.sort(rng.uniform(20, 420, n)); it = rng.random(n); it[rng.random(n) < 0.1] = 0.0
        for prec in (300.12345, 450.0):
            a = XI.clean_peaks(mz, it, prec); b = chemcore.clean_peaks(mz, it, prec)
            assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[0].dtype == np.float32


def test_instrument_class_matches_specdata():
    import specdata
    for s in ('timsTOF Pro 2', 'Q Exactive HF', 'Orbitrap Fusion', 'qTOF', 'Maxis II HD Q-TOF', 'LTQ', 'QqQ', 'unknown', None, 'ESI-QFT'):
        assert XI.instrument_class(s) == specdata.instrument_class(s), s


def test_prep_row_fields():
    r = _rows()
    sp = XI.prep_row(r[0])
    assert sp['meta'].tolist() == [0, 1, 0, 1] and abs(float(sp['ce']) - 0.2) < 1e-6          # [M+H]+, positive, timsTOF, CE known
    sp = XI.prep_row(r[1])
    assert sp['meta'].tolist() == [6, 0, 1, 0] and float(sp['ce']) == 0.0                   # [M-H]-, negative, Orbitrap, CE unknown
    assert np.allclose(sp['shifts'][sp['sdh'] == 0], -X.PROTON)                              # negative carrier
    sp = XI.prep_row(r[2])
    assert sp['meta'].tolist() == [4, 1, 2, 1] and abs(float(sp['ce']) - 0.2) < 1e-6          # mean of 10 / 30 eV
    assert XI.prep_row(dict(r[0], ce=[np.nan, 40.0]))['ce'] == np.float32(0.4)
    assert XI.prep_row(dict(r[0], ce=[np.nan]))['meta'][3] == 0
    assert XI.prep_row(dict(r[0], mz=[], it=[])) is None
    assert XI.prep_row(dict(r[0], it=[0.0] * 5)) is None
    assert XI.prep_row(dict(r[1], ionization_mode='negative', adduct='[M-H]-'))['meta'][1] == 0
    assert XI.prep_row(dict(r[0], mode=-1))['meta'][1] == 0                                   # numeric mode wins
    assert XI.prep_row(dict(r[0], instrument=3))['meta'][2] == 3
    many = [r[0]] * 20 + [dict(r[0], mz=[], it=[])]
    assert len(XI.prep_spectra(many)) == XI.MAX_SPEC and len(XI.prep_spectra([dict(r[0], mz=[], it=[])] + r)) == 3


def test_network_equals_model_xe():
    import torch
    import model as M
    torch.manual_seed(1)
    a = M.XE(d=32, layers=2, heads=4, ff=64).eval()
    b = XI._xe_class()(d=32, layers=2, heads=4, ff=64).eval()
    assert list(a.state_dict()) == list(b.state_dict())
    b.load_state_dict(a.state_dict(), strict=True)
    sps = XI.prep_spectra(_rows()); idx, cg, ai, af = XI.featurize(sps, SMILES[:2], {})
    f, mask, meta, ce = X.spectra_block(sps)
    ci = np.repeat(np.arange(2), 3); si = np.tile(np.arange(3), 2)
    D = dict(spf=torch.from_numpy(f[si]), mask=torch.from_numpy(mask[si]), ai=torch.from_numpy(ai[ci, si].astype(np.int64)),
             af=torch.from_numpy(af[ci, si]), meta=torch.from_numpy(meta[si].astype(np.int64)), ce=torch.from_numpy(ce[si]),
             cg=torch.from_numpy(cg[ci]))
    with torch.no_grad():
        assert torch.allclose(a(**D), b(**D), atol=1e-6)


def test_export_and_load_roundtrip():
    import torch
    net = _nets(1)[0]
    with tempfile.TemporaryDirectory() as d:
        torch.save(dict(net=net.state_dict(), args=dict(d=32, layers=2, heads=4, ff=64, dropout=0.1, no_bt=False, seed=7), epoch=3, val=0.5),
                   os.path.join(d, 'best.pt'))
        cfg = XI.export_npz(os.path.join(d, 'best.pt'), os.path.join(d, 'm.npz'))
        assert cfg['d'] == 32 and cfg['seed'] == 7 and cfg['epoch'] == 3
        with np.load(os.path.join(d, 'm.npz'), allow_pickle=False) as z:              # loads without pickle
            assert '__config__' in z.files
        back = XI.load_model(os.path.join(d, 'm.npz'), 'cpu')
        sps = XI.prep_spectra(_rows())
        assert np.allclose(XI.score_candidates([net], sps, SMILES, 'cpu'), XI.score_candidates([back], sps, SMILES, 'cpu'), equal_nan=True)


def test_scores_equal_training_path():
    """isoxe_infer on prepared spectra == build_groups packing (float16 storage) + Store + train_isoxe.score_store (CPU)."""
    import torch
    import build_groups as BG
    import model as M
    import train_isoxe as TI
    BG._W['X'] = X
    sps = XI.prep_spectra(_rows())
    cands = SMILES[:4]
    D = BG.featurize_groups([dict(gid=0, e180=True)], lambda g: (cands, [0] * 4, [1, 0, 0, 0]), lambda g: sps)
    nets = _nets(2)
    with tempfile.TemporaryDirectory() as d:
        np.savez(os.path.join(d, 'chunk_00000.npz'), **{k: v for k, v in D.items() if k != 'cand_smiles'}, cand_smiles=D['cand_smiles'].astype(str))
        S = M.Store(d)
        ref = np.mean([TI.score_store(n, S, torch.device('cpu')) for n in nets], 0)
    mine = XI.score_candidates(nets, sps, cands, 'cpu', {})
    assert np.allclose(mine, ref, atol=1e-5), (mine, ref)


def test_nan_mean_over_spectra_and_seeds():
    nets = _nets(2)
    sps = XI.prep_spectra(_rows())
    s = XI.score_candidates(nets, sps, SMILES, 'cpu', {})
    assert s.shape == (5,) and np.isnan(s[4]) and np.isfinite(s[:4]).all()
    s0 = XI.score_candidates(nets[:1], sps, SMILES, 'cpu', {}); s1 = XI.score_candidates(nets[1:], sps, SMILES, 'cpu', {})
    assert np.allclose(s[:4], (s0[:4] + s1[:4]) / 2, atol=1e-6)
    per = [XI.score_candidates(nets[:1], [sp], SMILES, 'cpu', {}) for sp in sps]
    assert np.allclose(s0[:4], np.mean(per, 0)[:4], atol=1e-5)                                   # mean over spectra
    assert np.allclose(XI.score_candidates(nets, sps[::-1], SMILES, 'cpu', {})[:4], s[:4], atol=1e-5)   # order of spectra irrelevant
    assert np.allclose(XI.score_candidates(nets, sps, SMILES, 'cpu', {}, chunk=3)[:4], s[:4], atol=1e-6)  # chunking irrelevant
    assert np.isnan(XI.score_candidates(nets, [], SMILES, 'cpu')).all() and len(XI.score_candidates(nets, sps, [], 'cpu')) == 0
    d = XI.score_dict(nets, _rows(), SMILES, 'cpu', {})
    assert set(d) == set(SMILES[:4]) and all(np.isfinite(v) for v in d.values())


def test_selftest_pass_and_fail():
    nets = _nets(2)
    exp = [None if v != v else float(v) for v in XI.selftest_scores(nets, 'cpu')]
    assert exp[-1] is None
    assert XI.selftest(nets, exp, 'cpu') == 'ok'
    bad = list(exp); bad[0] += 0.1
    try:
        XI.selftest(nets, bad, 'cpu'); raise RuntimeError('should fail')
    except AssertionError:
        pass
    try:
        XI.selftest(nets, exp[:-1] + [0.0], 'cpu'); raise RuntimeError('should fail')
    except AssertionError:
        pass


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print('ok', k)
