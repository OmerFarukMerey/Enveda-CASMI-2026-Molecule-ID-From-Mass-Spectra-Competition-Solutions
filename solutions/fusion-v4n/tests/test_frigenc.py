"""Data-free tests of clean/frigenc.py and the exclusion of clean/train_frigenc.py: FRIGID's fingerprint, the three-output model, bucketing, checkpoints."""
import os, sys, tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'clean'))
import frigenc  # noqa: E402
import specenc  # noqa: E402


def test_frigid_fp_is_morgan2_4096_bits():
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    smi = 'C[C@H](N)C(=O)O'
    a = frigenc.frigid_fp(smi)
    g = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)          # same bits as the legacy call, no chirality
    ref = np.zeros(4096, np.uint8); ref[list(g.GetFingerprint(Chem.MolFromSmiles(smi)).GetOnBits())] = 1
    assert a.dtype == np.uint8 and a.shape == (4096,) and (a == ref).all()
    assert (frigenc.frigid_fp('CC(N)C(=O)O') == a).all()                            # stereo does not change it
    assert frigenc.frigid_fp('not a smiles') is None
    assert frigenc.tanimoto(a[None], a[None])[0] == 1.0


def _items(n, rng, multi):
    out = []
    for _ in range(n):
        k = int(rng.integers(3, 30)); mz = np.sort(rng.uniform(50, 400, k)).astype(np.float32); s = rng.uniform(0.1, 1, k).astype(np.float32)
        it = dict(prec=450.0, adduct_ix=0, ce=20.0, ce_known=1.0, n_merged=1, mode=1.0, mz=mz, it=s)
        if multi:
            it.update(pprec=np.full(k, 450.0, np.float32), pce=np.full(k, 20.0, np.float32), pce_known=np.ones(k, np.float32),
                      pad_ix=np.zeros(k, np.int64), slot=(np.arange(k) % 2).astype(np.int64))
        out.append(it)
    return out


def test_three_outputs_bucketing_and_roundtrip():
    torch.manual_seed(0); rng = np.random.default_rng(0)
    cfg = specenc.make_cfg(pair=True, multi=True, d=32, layers=2, heads=4, multi_peaks=160)
    m = frigenc.FrigEnc(64, cfg).eval()
    b = specenc.collate(_items(6, rng, True), cfg)
    with torch.no_grad():
        z, e, f = m(b); z2, e2, f2 = frigenc.forward_bucketed(m, b, 3)
    assert z.shape == (6, 64) and e.shape[0] == 6 and f.shape == (6, frigenc.FRIG_BITS)
    assert torch.allclose(f, f2, atol=1e-5) and torch.allclose(z, z2, atol=1e-5)
    assert abs(torch.sigmoid(f).mean().item() - 46 / 4096) < 0.01                 # head starts at the mean bit density
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'x.pt'); frigenc.save_ckpt(p, m, 5); m2 = frigenc.load_ckpt(p)
        with torch.no_grad():
            assert torch.allclose(m2(b)[2], f, atol=1e-6)


def test_restrict_removes_excluded_structures():
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'train'))
    import train_frigenc as TF

    class D:
        sid = np.array([0, 0, 1, 2, 2, 3]); tr = np.arange(6)

        def restrict(self, idx):
            self.tr = np.asarray(idx)
    d = D(); n = TF.restrict(d, np.array([2]))
    assert n == 2 and d.tr.tolist() == [0, 1, 2, 5]
