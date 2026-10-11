"""Leak-safe data plumbing and the reference model of the joint spectrum-structure work (train/joint/data.py, graphs.py, model.py).

No competition data: tables and a JointData stand-in are built in memory. What is pinned down:
  * train_mask removes the held-out folds AND every structure sharing an InChIKey first block, metric key or raw key with them;
  * csr groups rows by code; spectra_arrays keeps the most intense peaks in m/z order and pads;
  * negatives: never the positive, same-formula pool isomers first, PubChem negatives next, near-mass rows of another formula, random
    rows last; `neg_hold=False` never returns a held-out pool row; the kinds are labelled;
  * unpack_bits is numpy.unpackbits; info_nce masks in-batch duplicates of a structure and invalid negatives; the spectrum encoder
    ignores padded peaks; the graph encoder accepts an unparseable SMILES and does not depend on the order of the graphs; a training
    step of both variants runs on the CPU and yields a finite loss.
usage: python tests/test_joint_data.py   (or pytest)
"""
import os, sys

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'train', 'joint'))
import numpy as np
import pandas as pd
import data as JD


def test_train_mask():
    st = pd.DataFrame(dict(sid=range(6), inchikey14=['A', 'B', 'C', 'D', 'E', 'F'], key=['A', 'B', 'C', 'A', 'E', 'F'], rawkey=['A', 'B', 'C', 'D', 'B', 'F']))
    sp = pd.DataFrame(dict(sid=[0, 1], inchikey14=['A', 'B'], key=['A', 'B'], fold=['hold', 'np']))
    ok, shared = JD.train_mask(st, sp)
    assert ok.tolist() == [False, False, True, False, False, True] and shared == 2           # 3 shares the metric key of 0, 4 the raw key of 1
    ok2, _ = JD.train_mask(st, sp, hold=('hold',)); assert ok2.tolist() == [False, True, True, False, True, True]


def test_csr_and_spectra_arrays():
    ptr, rows = JD.csr(np.array([2, 0, 2, 1, 0]), 3); assert [rows[ptr[c]:ptr[c + 1]].tolist() for c in range(3)] == [[1, 4], [3], [0, 2]]
    pk = [(np.array([10., 20., 30., 40.], np.float32), np.array([.1, 1., .5, .2], np.float32)), (np.array([5.], np.float32), np.array([1.], np.float32))]
    b = JD.spectra_arrays(pk, [100., 50.], [0, 1], [1, 0], [np.nan, 20.], [0, 1], [0, 0], max_peaks=2)
    assert b['mz'].shape == (2, 2) and b['mz'][0].tolist() == [20., 30.] and b['it'][0].tolist() == [1., .5] and b['n_peaks'].tolist() == [2, 1]
    assert b['mz'][1].tolist() == [5., 0.] and b['ce_known'].tolist() == [0., 1.] and b['ce'].tolist() == [0., 20.] and b['mode'].tolist() == [1, 0]


def _fake(neg_hold=True, pc=True):
    """Pool of 12 rows sorted by mass: rows 0-4 one formula (isomers), 5-6 another formula inside the same +-10 ppm window, 7-11 far away."""
    D = object.__new__(JD.JointData)
    D.pool_mass = np.array([100.0] * 5 + [100.0005] * 2 + [200.0, 300.0, 400.0, 500.0, 600.0]); D.pool_form = np.array([0] * 5 + [1] * 2 + [2, 3, 4, 5, 6], np.int32)
    D.form_ptr, D.form_rows = JD.csr(D.pool_form, 7); D.pool_held = np.zeros(12, bool); D.pool_held[[1, 5]] = True; D.neg_hold = neg_hold; D.pc = pc
    D.pc_of_pid = np.full(12, -1, np.int32); D.pc_of_pid[0] = 0; D.pc_hard = np.array([[7, 8, -1, -1]]); D.pc_rand = np.array([[3, 4, 5, -1]])
    return D


def test_negatives():
    rng = np.random.default_rng(0); D = _fake()
    for _ in range(20):
        s, i, k = D.negatives(0, 8, rng)
        assert not ((s == 0) & (i == 0)).any() and (k >= 0).all()
        assert sorted(i[k == 0].tolist()) == [1, 2, 3, 4] and set(i[k == 1]) <= {7, 8} and set(i[k == 2]) <= {3, 4, 5} and (s[k >= 1][k[k >= 1] <= 2] == 1).all()
        assert (k == 0).sum() == 4 and (k == 1).sum() + (k == 2).sum() == 4                  # half pool isomers, the rest PubChem
    s, i, k = D.negatives(0, 12, rng)                                                        # 4 isomers + 5 PubChem + 2 near-mass + 1 random
    assert (k == 0).sum() == 4 and (k == 1).sum() == 2 and (k == 2).sum() == 3 and sorted(i[k == 3].tolist()) == [5, 6] and (k == 4).sum() == 1
    D = _fake(neg_hold=False, pc=False)
    for _ in range(20):
        s, i, k = D.negatives(0, 6, rng); pool = i[(s == 0) & (k != 4)]
        assert not np.isin(pool, [0, 1, 5]).any() and sorted(i[k == 0].tolist()) == [2, 3, 4] and i[k == 3].tolist() == [6]
    s, i, k = _fake(pc=False).negatives(9, 3, rng); assert (k == 4).all() and not (i == 9).any()      # a structure without isomers: random rows, labelled


def test_model_pieces():
    import torch
    import graphs as G, model as JM
    torch.manual_seed(0)
    bits = np.random.default_rng(0).integers(0, 2, (3, 37)).astype(np.uint8); pk = np.packbits(bits, axis=1)
    assert (JM.unpack_bits(torch.as_tensor(pk), 37).numpy() == bits).all()
    # info_nce: perfect embeddings -> accuracy 1; a duplicate structure in the batch and an invalid negative are masked
    z = torch.eye(4); zn = torch.zeros(4, 2, 4); zn[0, 0] = z[0]                              # molecule 0's first "negative" equals its positive but is invalid
    ok = torch.ones(4, 2, dtype=torch.bool); ok[0, 0] = False
    loss, acc, n = JM.info_nce(z, z, zn, torch.arange(4), ok, torch.tensor(50.0)); assert float(acc) == 1.0 and float(loss) < 1e-3 and n == 6
    zp = z.clone(); zp[1] = z[0]; zs = z.clone(); zs[1] = z[0]                                # molecules 0 and 1 are the same structure
    loss2, acc2, _ = JM.info_nce(zs, zp, zn, torch.tensor([0, 0, 2, 3]), ok, torch.tensor(50.0)); assert float(acc2) == 1.0 and float(loss2) < 1e-3
    # spectrum encoder: padded peaks do not change the embedding
    enc = JM.SpecEncoder(d=32, layers=2, heads=4, ff=64, out=16).eval()
    b = dict(mz=torch.tensor([[50., 80., 0.], [60., 0., 0.]]), it=torch.tensor([[1., .5, 0.], [1., 0., 0.]]), n_peaks=torch.tensor([2, 1]), prec=torch.tensor([100., 90.]),
             add=torch.tensor([0, 1]), mode=torch.tensor([1, 0]), ce=torch.tensor([20., 0.]), ce_known=torch.tensor([1., 0.]), instr=torch.tensor([0, 1]), owner=torch.tensor([0, 0]), n_mol=1)
    b2 = dict(b, mz=torch.cat([b['mz'], torch.full((2, 2), 77.0)], 1), it=torch.cat([b['it'], torch.full((2, 2), 0.3)], 1))
    with torch.no_grad():
        e1, e2 = enc(b), enc(b2)
    assert e1.shape == (1, 16) and torch.allclose(e1, e2, atol=1e-5) and abs(float(e1.norm()) - 1) < 1e-5
    # graphs: an unparseable SMILES is flagged; the embedding of a molecule does not depend on its position in the batch
    g = [G.featurize(s) for s in ('CCO', 'not a smiles', 'c1ccccc1O')]; assert g[1] is None and g[0][0].shape == (3, 7) and g[0][1].shape == (2, 4)
    c = G.collate(g); assert c['g_ok'].tolist() == [True, False, True] and int(c['g_n']) == 3 and c['g_edges'].max() < len(c['g_atoms'])
    ge = JM.GraphEncoder(d=32, layers=2, out=16).eval(); t = lambda c: {k: torch.as_tensor(v) for k, v in c.items()}
    with torch.no_grad():
        a = ge(t(c)); r = ge(t(G.collate(g[::-1])))
    assert torch.allclose(a[0], r[2], atol=1e-5) and torch.allclose(a[2], r[0], atol=1e-5)


def test_training_step_both_variants():
    import torch
    import graphs as G, model as JM
    torch.manual_seed(0); B, K, nbits = 4, 3, 40; rng = np.random.default_rng(0); smi = ['CCO', 'CCN', 'CCC', 'CC=O'] + ['CCCl'] * (B * K)
    b = dict(mz=torch.rand(6, 5) * 100, it=torch.rand(6, 5), n_peaks=torch.tensor([5, 3, 1, 5, 2, 4]), prec=torch.full((6,), 120.0), add=torch.zeros(6, dtype=torch.long),
             mode=torch.ones(6, dtype=torch.long), ce=torch.full((6,), 30.0), ce_known=torch.ones(6), instr=torch.zeros(6, dtype=torch.long), owner=torch.tensor([0, 0, 1, 2, 3, 3]), n_mol=B,
             pid=torch.arange(B), neg_kind=torch.zeros(B, K, dtype=torch.long), pos_fp=torch.as_tensor(rng.integers(0, 256, (B, 5)).astype(np.uint8)),
             neg_fp=torch.as_tensor(rng.integers(0, 256, (B, K, 5)).astype(np.uint8)))
    b.update({k: torch.as_tensor(v) for k, v in G.collate([G.featurize(s) for s in smi]).items()})
    for variant in ('fp', 'gnn'):
        m = JM.Joint(variant, nbits=nbits, d=32, layers=1, out=16); loss, acc, n = m(b); loss.backward()
        assert torch.isfinite(loss) and n == B + K and all(p.grad is not None for p in m.spec.proj.parameters()) and JM.n_params(m) > 0


if __name__ == '__main__':
    for k, f in sorted(globals().items()):
        if k.startswith('test_') and callable(f):
            f()
    print('joint data OK: leak-safe training mask, negatives by kind, padding, InfoNCE masking, both structure encoders')
