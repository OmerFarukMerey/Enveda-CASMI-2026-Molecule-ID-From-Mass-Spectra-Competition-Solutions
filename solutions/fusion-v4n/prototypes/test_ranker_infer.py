"""Tests of prototypes/ranker_infer.py on synthetic candidates (no competition data): the inference features equal
the training-table features of ranker_rows / tie_order, and the three score modes behave as documented.

usage: python solutions/fusion-v4n/prototypes/test_ranker_infer.py
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ranker_infer as RI
import ranker_rows as RR
import tie_order as TO

NAMES_X = ['lib_max', 'lib_max_rank', 'ap', 'ap_rank', 'fz', 'fz_rank', 'el_sum', 'el_rank', 'fr_int', 'fr_rank', 'sf_int', 'sf_rank', 'mass_ppm',
           'mass_ppm_rank', 'target_mass', 'is_gen']
NAMES_F = ['frag2__covw1', 'fragnet__fn_score']


def _molecule(n=14, seed=0):
    rng = np.random.RandomState(seed)
    formulas = [['C6H6', 'C5H2O', 'C4H6N2'][i % 3] for i in range(n)]
    X = rng.rand(n, len(NAMES_X)).astype(np.float32)
    f_of = {f: k for k, f in enumerate(sorted(set(formulas)))}
    for src in ('el_sum', 'sf_int', 'mass_ppm'):                   # formula-level values: ties inside a formula
        X[:, NAMES_X.index(src)] = [0.1 * (1 + f_of[f]) for f in formulas]
    X[:, NAMES_X.index('lib_max')] = np.where(rng.rand(n) < 0.6, 0.0, X[:, NAMES_X.index('lib_max')])
    for f, (src, sgn, ab) in TO.RANK_SOURCES.items():              # the engine's stable ranks
        v = X[:, NAMES_X.index(src)].astype(np.float64)
        X[:, NAMES_X.index(f)] = TO.rank_norm_stable(sgn * (np.abs(v) if ab else v))
    F = rng.rand(n, len(NAMES_F)).astype(np.float32)
    return formulas, X, F


def test_order_free_equals_training_recompute_and_is_permutation_invariant():
    formulas, X, F = _molecule()
    start = np.array([0, len(X)])
    assert np.allclose(RI.order_free(X, NAMES_X), TO.rerank(X, NAMES_X, start, 'average'))
    p = np.random.RandomState(1).permutation(len(X))
    assert np.allclose(RI.order_free(X, NAMES_X)[p], RI.order_free(X[p], NAMES_X))
    j = NAMES_X.index('sf_rank'); g = np.array(formulas) == 'C6H6'
    assert len(set(X[g, j])) > 1 and len(set(RI.order_free(X, NAMES_X)[g, j])) == 1     # the row-order code is gone


def test_confidence_features_equal_training_table():
    formulas, X, F = _molecule(seed=3)
    Xo = RI.order_free(X, NAMES_X)
    E = RI.confidence_features(Xo, NAMES_X, formulas)
    n = len(X); mol = np.zeros(n, np.int64)
    R = RR.Rows(X=Xo, F=F, rk=np.zeros(n), pid=np.arange(n), y=np.zeros(n, bool), mol=mol, start=np.array([0, n]), names_x=NAMES_X, names_f=NAMES_F,
                meta=pd.DataFrame(dict(mid=['m'])), fg=RR.formula_groups(np.array(formulas, object), mol))
    T = RR.extra_features(R)
    assert set(E) == set(T.columns) and len(E) == 10
    for k in E:
        assert np.allclose(E[k], T[k].values, atol=1e-5), k
    one = RI.confidence_features(Xo[:1], NAMES_X, formulas[:1])
    assert one['fz_fg_z'][0] == 0 and one['fz_fg_margin'][0] == 0 and one['lib_margin'][0] == Xo[0, 0]


def test_ranker_roundtrip_and_modes():
    import lightgbm as lgb
    rng = np.random.RandomState(0)
    feats = ['x__fz', 'x__sf_rank', 'frag2__covw1', 'e__fz_fg_gap', 'e__fg_size']
    Xt = rng.rand(600, len(feats)).astype(np.float32); yt = (Xt[:, 0] > 0.8).astype(int)
    b = lgb.train(dict(objective='lambdarank', verbose=-1, min_data_in_leaf=5, num_threads=1, label_gain=[0, 1]),
                  lgb.Dataset(Xt, label=yt, group=[20] * 30, feature_name=feats), num_boost_round=10)
    with tempfile.TemporaryDirectory() as d:
        b.save_model(os.path.join(d, 'ranker_s0.txt')); b.save_model(os.path.join(d, 'ranker_s1.txt'))
        import json
        json.dump(feats, open(os.path.join(d, 'features.json'), 'w'))
        R = RI.OurRanker.load(d)
    formulas, X, F = _molecule(seed=5)
    M = R.matrix(formulas, X, F, NAMES_X, NAMES_F)
    assert M.shape == (len(X), 5) and np.allclose(M[:, 0], X[:, NAMES_X.index('fz')]) and np.allclose(M[:, 2], F[:, 0])
    assert np.allclose(M[:, 1], RI.order_free(X, NAMES_X)[:, NAMES_X.index('sf_rank')])
    s = R.score(formulas, X, F, NAMES_X, NAMES_F, threads=1)
    assert s.shape == (len(X),) and np.allclose(s, b.predict(M))
    seen = []

    def public(Mx, names):                                           # stands in for nb/05 rank_score
        seen.append(Mx[:, names.index('sf_rank')].copy()); return Mx[:, names.index('fz')] * 2.0
    stable = X[:, NAMES_X.index('sf_rank')]; free = RI.order_free(X, NAMES_X)[:, NAMES_X.index('sf_rank')]
    args = (formulas, X, F, NAMES_X, NAMES_F)
    RI.choose_score('public', None, *args, 0.3, public); assert np.allclose(seen[-1], stable)                    # default: unchanged
    RI.choose_score('public', None, *args, 0.3, public, lib_gate=0.7); assert np.allclose(seen[-1], stable)      # 'public' ignores the gate
    RI.choose_score('public_free', None, *args, 0.3, public); assert np.allclose(seen[-1], free)                 # order-free everywhere
    RI.choose_score('public_free', None, *args, 0.3, public, lib_gate=0.7); assert np.allclose(seen[-1], free)   # no library evidence
    RI.choose_score('public_free', None, *args, 0.9, public, lib_gate=0.7); assert np.allclose(seen[-1], stable) # library hit: as today
    assert np.allclose(RI.choose_score('ours', R, *args, 0.3, public), s)
    assert np.allclose(RI.choose_score('ours', R, *args, 0.9, public, lib_gate=0.7), X[:, NAMES_X.index('fz')] * 2.0)   # gated: public as today
    bl = RI.choose_score('blend', R, *args, 0.3, public)
    assert np.allclose(seen[-1], free) and abs(bl.mean()) < 1e-6 and len(bl) == len(X)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
