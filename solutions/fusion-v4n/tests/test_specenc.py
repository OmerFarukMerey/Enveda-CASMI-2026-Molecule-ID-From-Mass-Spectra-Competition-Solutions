"""Subformula annotation and the configurable spectrum encoder (clean/subform.py, clean/specenc.py), without competition data.

Pinned: ion formulas of adducts, the annotation of known fragments of glucose and of a sodium dimer (formula, neutral
loss, error, no annotation for a shifted peak), the hydrogen rule, the token set 'mz' == fpnet.FPNet with the same
weights, merge_idx / prep_idx == the reference merge and peak preparation, every token configuration runs and does not
depend on padding or on the bucketed forward, the pairwise-difference features, hidden formulas, the checkpoint round trip.
usage: python tests/test_specenc.py   (or pytest)
"""
import os, sys, tempfile

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'clean'))
import numpy as np
import torch
import chemcore, fpnet, specdata, specenc, subform

E = chemcore.ELECTRON


def _mass(f):
    return float(subform.parse_counts(f) @ subform.MASS)


def test_ion_counts():
    c, s = subform.ion_counts('C6H12O6', '[M+H]+'); assert s == 1 and list(c[:4]) == [6, 13, 0, 6]
    c, s = subform.ion_counts('C6H12O6', '[2M+Na]+'); assert list(c[:4]) == [12, 24, 0, 12] and c[10] == 1
    c, s = subform.ion_counts('C6H12O6', '[M+CH2O2-H]-'); assert s == -1 and list(c[:4]) == [7, 13, 0, 8]
    c, s = subform.ion_counts('C6H12O6', '[M-H2O+H]+'); assert list(c[:4]) == [6, 11, 0, 5]
    assert subform.ion_counts('C6H12O6Si', '[M+H]+') is None and subform.ion_counts('C6H12O6', 'M+H') is None
    assert list(subform.parse_counts('C5H14NO+')[:4]) == [5, 14, 1, 1]


def test_annotate():
    A = subform.Annotator(); c, s = subform.ion_counts('C6H12O6', '[M+H]+')
    frags = ['C6H11O5', 'C6H9O4', 'C4H7O3', 'C3H5O2']
    mz = np.array([_mass(f) - E for f in frags] + [100.3371])
    cnt, nc, err, ok = A.annotate(mz * (1 + 3e-6), c, s)
    assert ok and list(nc[:4] > 0) == [True] * 4 and nc[4] == 0 and not cnt[4].any()
    for r, f in enumerate(frags):
        assert np.array_equal(cnt[r], subform.parse_counts(f)), (f, cnt[r])
        assert abs(err[r] + 3.0) < 0.2                                      # calculated minus observed, ppm
    c2, s2 = subform.ion_counts('C6H12O6', '[2M+Na]+')
    cnt, nc, _, _ = A.annotate(np.array([_mass('C6H12O6Na') - E]), c2, s2)
    assert cnt[0, 10] == 1 and list(cnt[0, :4]) == [6, 12, 0, 6]
    cn, sn = subform.ion_counts('C6H12O6', '[M-H]-')
    cnt, nc, _, _ = A.annotate(np.array([_mass('C6H11O6') + E, _mass('C3H5O3') + E]), cn, sn)
    assert list(cnt[0, :4]) == [6, 11, 0, 6] and list(cnt[1, :4]) == [3, 5, 0, 3]
    cnt, nc, _, _ = A.annotate(np.array([5 * subform.MASS[1] - E]), c, s)   # H5+: no heavy atom
    assert nc[0] == 0
    big = np.zeros(subform.NE, np.int64); big[[0, 2, 3, 5]] = 400
    assert A.annotate(np.array([100.0]), big, 1)[3] is False


def test_reference_parity():
    torch.manual_seed(0); ref = fpnet.FPNet(64, d=64, layers=2, heads=4).eval()
    net = specenc.SpecEnc(64, dict(tok='mz', d=64, layers=2, heads=4)).eval()
    assert {k: tuple(v.shape) for k, v in ref.state_dict().items()} == {k: tuple(v.shape) for k, v in net.state_dict().items()}
    net.load_state_dict(ref.state_dict()); rng = np.random.default_rng(0); items = []
    for _ in range(6):
        n = int(rng.integers(1, 40)); mz = np.sort(rng.uniform(50, 400, n)); it = rng.uniform(0.01, 1, n)
        items.append(fpnet.make_item(mz, it, 410.0, 3, 20.0, 2, 1.0))
    a = fpnet.collate(items); z0, e0 = ref(*a)
    b = dict(mz=a[0], it=a[1], pad=a[2], prec=a[3], ad=a[4], ce=a[5], ce_known=a[6], merged=a[7], mode=a[8])
    z1, e1 = net(b); assert torch.equal(z0, z1) and torch.equal(e0, e1)
    z2, _ = specenc.forward_bucketed(net, b, 3); assert torch.allclose(z0, z2, atol=1e-5)
    assert torch.equal(specenc.RefSignature(net)(*a)[0], z0)                # the reference's positional interface
    pn = specenc.SpecEnc(64, dict(tok='mz', pair=True, d=64, layers=2, heads=4)).eval()
    assert torch.equal(specenc.RefSignature(pn)(*a)[0], pn(b)[0]) and specenc.RefSignature(pn)(*a)[0].shape == (6, 64)


def test_merge_and_prep_idx():
    rng = np.random.default_rng(1)
    for _ in range(50):
        parts = []
        for _ in range(int(rng.integers(2, 6))):
            n = int(rng.integers(0, 30)); mz = np.sort(np.round(rng.uniform(50, 60, n), 3)).astype(np.float32); parts.append((mz, rng.uniform(0.01, 5, n).astype(np.float32)))
        m0, i0 = specdata.merge_spectra(parts); m1, i1, ix = specenc.merge_idx([p[0] for p in parts], [p[1] for p in parts])
        assert np.array_equal(m0, m1) and np.array_equal(i0, i1)
        assert np.array_equal(np.concatenate([p[0] for p in parts])[ix], m1)
        a, b = fpnet.prep_peaks(m1, i1, 58.0, max_peaks=20); c, d, k = specenc.prep_idx(m1, i1, 58.0, 20)
        assert np.array_equal(a, c) and np.array_equal(b, d) and np.array_equal(m1[k], c)
    assert np.allclose(specenc.rank_feature(np.array([0.2, 1.0, 0.5], np.float32)), [0.0, 1.0, 0.5])


class _Toy:
    """Three spectra of one structure; the annotation comes from an AnnFly with the true formula."""

    def __init__(self):
        f = ['C6H11O5', 'C6H9O4', 'C4H7O3', 'C3H5O2']
        self.mzs = [np.array(sorted(_mass(x) - E for x in f[k:]), np.float32) for k in range(3)]
        self.its = [np.linspace(0.2, 1, len(m)).astype(np.float32) for m in self.mzs]
        self.ce = np.array([10.0, np.nan, 40.0]); self.ce_n = np.array([1, 0, 1]); self.prec = np.full(3, _mass('C6H13O6') - E)
        self.adduct = np.array([0, 0, 0]); self.mode = np.array([1, 1, 1]); self.adducts = ['[M+H]+'] * 3

    def peaks(self, i):
        return self.mzs[i], self.its[i]


def test_items_and_configs():
    D = _Toy(); ann = specenc.AnnFly(D, D.adducts, {0: 'C6H12O6', 1: 'C6H12O6', 2: 'C6H12O6'})
    torch.manual_seed(0)
    for kw in (dict(tok='form'), dict(tok='both'), dict(tok='mz', pair=True), dict(tok='mz', multi=True), dict(tok='both', multi=True, pair=True),
               dict(tok='mz', rank=True), dict(tok='mz', dreams=16)):
        cfg = specenc.make_cfg(d=64, layers=2, heads=4, **kw); net = specenc.SpecEnc(32, cfg).eval()
        de = np.ones(16, np.float32) if cfg['dreams'] else None
        items = [specenc.build_item(D, e, cfg, ann, demb=de) for e in ([0], [1, 2], [0, 1, 2])]
        if cfg['tok'] != 'mz':
            x = items[0]; assert x['has_form'] == 1.0 and (x['nc'] > 0).all() and list(x['mcnt'][:4]) == [6, 12, 0, 6]
            assert np.array_equal(x['fcnt'].astype(int) + x['lcnt'], np.tile(subform.ion_counts('C6H12O6', '[M+H]+')[0], (len(x['mz']), 1)))
            h = specenc.build_item(D, [0], cfg, ann, hide_form=True); assert h['has_form'] == 0.0 and not h['fcnt'].any()
        if cfg['multi']:
            assert len(items[2]['mz']) == 4 + 3 + 2 and list(np.unique(items[2]['slot'])) == [0, 1, 2] and items[2]['pce_known'].min() == 0.0
        elif not cfg['rank']:
            assert len(items[2]['mz']) == 4                                # merged: the same four fragments
        b = specenc.collate(items, cfg); z, e = net(b); assert z.shape == (3, 32) and torch.isfinite(z).all()
        z1, _ = net(specenc.collate(items[:1], cfg)); assert torch.allclose(z[0], z1[0], atol=1e-5), kw   # padding does not matter
        zb, _ = specenc.forward_bucketed(net, b, 2); assert torch.allclose(z, zb, atol=1e-5)
        rng = np.random.default_rng(0)
        t = specenc.build_item(D, [0, 1], cfg, ann, rng, specdata.AUG_STD, demb=de); assert len(t['mz']) >= 1
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'x.pt'); specenc.save_ckpt(p, net, 7, half=True); m2 = specenc.load_ckpt(p)
        assert m2.cfg == net.cfg and torch.allclose(m2(b)[0], net(b)[0], atol=0.05)
        ref = fpnet.FPNet(32, d=64, layers=2, heads=4); q = os.path.join(td, 'r.pt'); fpnet.save_ckpt(q, ref, 1)
        assert specenc.load_ckpt(q).cfg['tok'] == 'mz'


def test_pair_features():
    net = specenc.SpecEnc(8, specenc.make_cfg(d=32, layers=1, heads=2, pair=True)); K = len(specenc.LOSSES); k = list(specenc.LOSSES).index('H2O')
    mz = torch.tensor([[100.0, 118.010565, 150.0, 0.0]]); it = torch.tensor([[0.5, 1.0, 0.3, 0.0]]); pad = torch.tensor([[False, False, False, True]])
    f = net.pair_feats(mz, it, pad, torch.tensor([[136.02113]]))
    assert f.shape == (1, 4, 3 * K) and f[0, 0, k] == 1.0 and f[0, 1, K + k] == 0.5 and f[0, 1, k] == 0 and f[0, 3].sum() == 0
    assert f[0, 1, 2 * K + k] == 1.0 and f[0, 0, 2 * K + list(specenc.LOSSES).index('2H2O')] == 1.0
    assert f[0, :3, :2 * K].sum() == 1.5
    assert all(abs(_mass('H4O2' if n == '2H2O' else n) - v) < 2e-5 for n, v in specenc.LOSSES.items())


if __name__ == '__main__':
    for k, f in sorted(globals().items()):
        if k.startswith('test_') and callable(f):
            f()
    print('specenc OK: ion formulas, subformula annotation, reference parity, merge / prep indices, token configurations, pair features, checkpoint')
