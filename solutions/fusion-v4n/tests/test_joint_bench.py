"""The joint-model benchmark (train/joint/bench.py, bench_build.py): ranking, bootstrap, halves, and evaluate() on a synthetic benchmark.

No competition data: a tiny benchmark directory is written to a temporary folder. What is pinned down:
  * rank_answer deduplicates by key, ranks an unscored answer last, and breaks ties by list order or by the given tie-break;
  * rank_answer_avg follows fs_pass.rank_average (half weight to each order);
  * halves(): no scaffold in both halves, the halves balanced although one scaffold is large; deterministic;
  * evaluate(): a perfect score file gives top-1 = MRR = 1 and a non-negative paired difference; a candidate missing from the file ranks
    last; the best score of a key is used; strata and views are reported; score_features has one row per table row.
usage: python tests/test_joint_bench.py   (or pytest)
"""
import os, sys, tempfile

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'train', 'joint'))
import numpy as np
import pandas as pd
import bench
import bench_build


def test_rank_answer():
    key = np.array([0, 1, 1, 2, 3]); y = np.array([False, False, False, True, False])
    assert bench.rank_answer(np.array([5., 4., 3., 2., 1.]), y, key) == 3                    # the two rows of key 1 count once
    assert bench.rank_answer(np.array([1., 1., 1., 9., 1.]), y, key) == 1
    assert bench.rank_answer(np.array([1., 1., 1., -np.inf, 1.]), y, key) == 4               # unscored answer: last of four keys
    assert bench.rank_answer(np.zeros(5), y, key) == 3                                       # ties in list order
    assert bench.rank_answer(np.zeros(5), y, key, tie=np.array([5., 4., 3., 0., 1.])) == 1   # ties by the tie-break
    assert bench.rank_answer(np.zeros(5), np.zeros(5, bool), key) == 0                       # answer absent
    assert (bench.rr_of(np.array([0, 1, 2, 25, 26])) == np.array([0, 1, 0.5, 1 / 25, 0])).all()


def test_rank_average():
    key = np.arange(4); y = np.array([False, True, False, False])
    a = np.array([4., 3., 2., 1.]); b = np.array([1., 4., 3., 2.])                           # answer second in a, first in b
    assert bench.rank_answer_avg(a, b, y, key, w=0.5) == 1                                   # mean rank 1.5 beats key 0 (2.5)
    assert bench.rank_answer_avg(a, b, y, key, w=0.0) == 2 and bench.rank_answer_avg(a, b, y, key, w=1.0) == 1


def test_boot_se():
    v = np.r_[np.ones(500), np.zeros(500)]; se = bench.boot_se(v)
    assert abs(se - 0.5 / np.sqrt(1000)) < 0.003 and np.isnan(bench.boot_se([1.0]))


def test_halves():
    labs = ['big'] * 400 + [f's{i}' for i in range(600)]
    h = bench_build.halves(labs); assert h == bench_build.halves(list(reversed(labs)))
    n = pd.Series([h[x] for x in labs]).value_counts(); assert abs(n['dev'] - n['final']) <= 400 and set(h.values()) == {'dev', 'final'}
    labs2 = [f's{i}' for i in range(1000)]; n2 = pd.Series(list(bench_build.halves(labs2).values())).value_counts(); assert abs(n2['dev'] - n2['final']) <= 1


def _synthetic(root, n_mol=60, n_cand=12, seed=0):
    """Molecules with one c2J and one dropJ list each; candidate keys k<mol>_<j>, the answer is j == 0; a duplicate-key row per list."""
    rng = np.random.default_rng(seed); M, L, R, C = [], [], [], {}; start = 0
    for i in range(n_mol):
        mid = f'm{i}'; M.append(dict(mid=mid, key=f'k{i}_0', half='dev' if i % 2 == 0 else 'final', cvfold=i % 5, np_catalogue=i % 3 == 0, panel=i % 10 == 0,
                                     enveda=i % 4 == 0, nfree=i % 2 == 1, batch='old'))
        for v in ('c2J', 'dropJ'):
            keys = [f'k{i}_{j}' for j in range(n_cand)] + [f'k{i}_1']                         # the last row repeats key 1
            for k in keys:
                C.setdefault(k, len(C))
            L.append(dict(list=len(L), mid=mid, view=v, set='s_' + v, start=start, n=len(keys), n_run=n_mol, lib_max=0.5 if i % 2 else 0.9, n_isomers=3 + i,
                          half=M[-1]['half'], cvfold=i % 5, bench=True))
            for j, k in enumerate(keys):
                R.append(dict(list=len(L) - 1, cand=C[k], y=k == f'k{i}_0', inj=j > 6, fz=rng.normal(), fp_cos=rng.normal(), pub=rng.normal() + (k == f'k{i}_0'), ours=rng.normal() + 2 * (k == f'k{i}_0')))
            start += len(keys)
    os.makedirs(root, exist_ok=True)
    pd.DataFrame(M).to_parquet(root + '/mols.parquet'); pd.DataFrame(L).to_parquet(root + '/lists.parquet'); pd.DataFrame(R).to_parquet(root + '/rows.parquet')
    pd.DataFrame(dict(cand=list(C.values()), key=list(C.keys()), smiles='C', pid=-1, formula='CH4', pubchem_only=False)).to_parquet(root + '/cands.parquet')
    return pd.DataFrame(M), pd.DataFrame(L)


def test_evaluate_synthetic():
    with tempfile.TemporaryDirectory() as d:
        M, L = _synthetic(d); B = bench.Bench(d)
        P = B.pairs('dev'); assert set(P.mid) == set(M.mid[M.half == 'dev']) and not P.duplicated(['mid', 'cand']).any() and len(P) == 30 * 12
        perfect = pd.DataFrame(dict(molecule_id=P.mid, candidate_key=P.key, score=[1.0 if k.endswith('_0') else 0.0 for k in P.key]))
        t = bench.evaluate(perfect, baseline='ours', half='dev', bench=B, log=lambda *a: None)
        assert set(t.view) == {'c2J', 'dropJ'} and (t.top1 == 1.0).all() and (t.mrr == 1.0).all() and (t.d_mrr >= 0).all() and (t.losses == 0).all()
        assert int(t[(t.view == 'c2J') & (t.stratum == 'all')].n.iloc[0]) == 30 and 'nitrogen-free' not in set(t.stratum)      # dev = even molecules: none nitrogen-free
        # the answer missing from the file ranks last of the 12 keys; a duplicate row with a lower score does not change the rank
        miss = perfect[~perfect.candidate_key.str.endswith('_0')]
        t2 = bench.evaluate(miss, baseline='ours', half='dev', bench=B, log=lambda *a: None); assert (t2.top1 == 0).all() and np.allclose(t2.mrr, 1 / 12)
        dup = pd.concat([perfect, perfect.assign(score=-5.0)]); t3 = bench.evaluate(dup, baseline='ours', half='dev', bench=B, log=lambda *a: None); assert (t3.top1 == 1.0).all()
        # a random score file is far below the perfect one, and the baselines table has every stored baseline
        rnd = perfect.assign(score=np.random.default_rng(1).random(len(perfect))); t4 = bench.evaluate(rnd, baseline='random', half='dev', bench=B, log=lambda *a: None)
        assert t4[t4.stratum == 'all'].top1.max() < 0.5
        T, F = bench.baselines('dev', B); assert {'random', 'fz', 'pub', 'ours', 'avg'} <= set(T.columns) and F is None
        f = bench.score_features(B, B.row_scores(perfect)); assert f.shape == (len(B.R), 4)
        dev_rows = np.repeat((B.L.half == 'dev').values, B.n); assert np.isfinite(f[dev_rows]).all() and np.isnan(f[~dev_rows]).all()
        assert (f[dev_rows, 3] <= 0).all() and f[dev_rows & B.y, 2].max() == 0.0                # the answer is the best of its list
        assert 'd_mrr' in bench.fmt(t) or 'Δ MRR@25' in bench.fmt(t)


if __name__ == '__main__':
    for k, f in sorted(globals().items()):
        if k.startswith('test_') and callable(f):
            f()
    print('joint bench OK: ranking with key deduplication and ties, rank average, bootstrap, scaffold halves, evaluate() on a synthetic benchmark')
