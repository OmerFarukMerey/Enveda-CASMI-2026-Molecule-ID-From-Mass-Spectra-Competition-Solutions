"""Tests of prototypes/ranker_rows.py on synthetic rows (no competition data): ranks, groups, folds, a tiny ranker.

usage: python solutions/fusion-v4n/prototypes/test_ranker_rows.py
"""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ranker_rows as RR


def _write_run(d, n_mol=6, seed=0):
    """Two chunk files in the layout of scratch/ranker/run_engine.py; molecule 2 has no candidates."""
    rng = np.random.RandomState(seed)
    names_x = ['fz', 'lib_max', 'is_gen']; names_f = ['frag2__covw1']
    pool = pd.DataFrame(dict(key=[f'K{i}' for i in range(40)], smiles=['C' * (i + 1) for i in range(40)],
                             formula=[f'C{1 + i % 3}H4' for i in range(40)]))
    for c, mols in enumerate([range(0, 3), range(3, n_mol)]):
        X, F, rk, pid, y, g, meta, gen = [], [], [], [], [], [], [], []
        for li, m in enumerate(mols):
            n = 0 if m == 2 else 5 + m
            p = rng.choice(40, n, replace=False).astype(np.int32)
            if n:
                p[-1] = -1                                         # one generated candidate per molecule
                gen.append(dict(mid=f'm{m}', j=n - 1, smiles='CCO', key=f'G{m}', formula='C2H6O'))
            yy = np.zeros(n, bool)
            if n:
                yy[0] = True
            X.append(rng.rand(n, 3).astype(np.float32)); F.append(rng.rand(n, 1).astype(np.float32)); rk.append(rng.rand(n).astype(np.float32))
            pid.append(p); y.append(yy); g.append(np.full(n, li, np.int32))
            meta.append(dict(mid=f'm{m}', n_cand=n, lib_max=0.0))
        cat = lambda v, w: np.concatenate(v) if len(v) else np.zeros((0, w))
        np.savez(os.path.join(d, f'w0_c{c:04d}.npz'), X=np.concatenate(X), F=np.concatenate(F), rk=np.concatenate(rk), pid=np.concatenate(pid),
                 y=np.concatenate(y), g=np.concatenate(g), names_x=np.array(names_x), names_f=np.array(names_f), gen_json=np.array(json.dumps(gen)))
        pd.DataFrame(meta).to_parquet(os.path.join(d, f'w0_c{c:04d}.parquet'), index=False)
    return pool


def test_load_rows_and_subset():
    with tempfile.TemporaryDirectory() as d:
        pool = _write_run(d)
        R = RR.load_rows(d, pool)
        assert R.n_mol == 6 and R.start[3] - R.start[2] == 0 and len(R.y) == sum(5 + m for m in (0, 1, 3, 4, 5))
        assert R.y.sum() == 5 and (R.key[R.pid < 0] == np.array(['G0', 'G1', 'G3', 'G4', 'G5'], object)).all()
        assert len(np.unique(R.fg[R.start[0]:R.start[1]])) <= 4          # at most 3 pool formulas + the generated one
        assert R.col('frag2__covw1').shape == R.y.shape and R.matrix(['fz', 'frag2__covw1']).shape[1] == 2
        S = RR.subset(R, [4, 0])
        assert S.n_mol == 2 and list(S.meta.mid) == ['m4', 'm0'] and S.start.tolist() == [0, 9, 14] and S.y.sum() == 2
        E = RR.extra_features(R, pop=dict(lsid=np.arange(40, dtype=np.float16), lpmid=np.zeros(40, np.float16), np=np.ones(40, np.uint8)))
        assert len(E) == len(R.y) and np.isnan(E['pop'].values[R.pid < 0]).all() and np.isfinite(E['fz_fg_gap']).all()
        assert (E['fz_fg_gap'] <= 0).all() and (E['pop_fg_gap'] <= 0).all()


def test_ranks_and_ties():
    start = np.array([0, 4, 4, 7]); y = np.array([0, 1, 0, 0, 0, 0, 1], bool); fg = np.array([0, 0, 1, 0, 2, 2, 3])
    s = np.array([0.9, 0.5, 0.7, 0.5, 0.1, 0.2, 0.0])
    assert RR.list_rank(s, y, start).tolist()[0] == 3.5 and np.isinf(RR.list_rank(s, y, start)[1])     # tie with one decoy: 3 or 4
    rr = RR.list_rr(s, y, start)
    assert rr[0] == 0.25 and rr[1] == 0.0 and abs(rr[2] - 1 / 3) < 1e-12
    r, n = RR.ingroup_rank(s, y, start, fg)
    assert r[0] == 2.5 and n[0] == 3 and np.isnan(r[1]) and r[2] == 1.0 and n[2] == 1
    r2, n2 = RR.ingroup_rank(s, y, start, fg, restrict=s > 0.6)                                       # truth always kept
    assert r2[0] == 2.0 and n2[0] == 2
    summ = RR.summarize(r, n)
    assert summ['n'] == 1 and summ['top1'] == 0.0                                                     # the size-1 group is dropped
    p = RR.paired(np.array([1.0, 0.5, 0.0]), np.array([0.5, 0.5, 0.25]))
    assert abs(p['delta'] - 0.0833333) < 1e-6 and p['wins'] == 1 and p['losses'] == 1


def test_group_helpers_and_folds():
    v = np.array([1.0, 3.0, 2.0, 5.0, 5.0]); g = np.array([0, 0, 0, 1, 1])
    assert np.allclose(RR.group_rank01(v, g), [1.0, 0.0, 0.5, 0.5, 0.5])
    z = RR.group_z(v, g); assert abs(z[:3].mean()) < 1e-9 and (z[3:] == 0).all()
    f = RR.grouped_folds(['a', 'b', 'a', 'c', 'b', 'd', 'e', 'f'], n_fold=3)
    assert f[0] == f[2] and f[1] == f[4] and set(f) <= {0, 1, 2}
    assert RR.block_of('lib_max') == 'lib' and RR.block_of('fragnet__fn_score') == 'fragnet' and RR.block_of('nope') == 'other'


def test_cv_ranker_learns_a_planted_signal():
    rng = np.random.RandomState(0); n_mol, n = 120, 12
    y = np.zeros(n_mol * n, bool); y[::n] = True
    X = rng.randn(n_mol * n, 3); X[:, 0] += 2.5 * y                                                  # feature 0 carries the label
    start = np.arange(0, n_mol * n + 1, n); fold = RR.grouped_folds(np.arange(n_mol) // 2, 4)        # pairs share a fold
    pred = RR.cv_predict(X, y, start, fold, ['a', 'b', 'c'], rounds=30, params=dict(min_data_in_leaf=5, num_threads=1))
    rr_model = RR.list_rr(pred, y, start).mean(); rr_noise = RR.list_rr(X[:, 1], y, start).mean()
    assert rr_model > 0.6 > rr_noise, (rr_model, rr_noise)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
