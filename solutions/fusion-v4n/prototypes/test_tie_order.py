"""Tests of prototypes/tie_order.py (no competition data): the stable rule reproduces the engine, the others move ties.

usage: python solutions/fusion-v4n/prototypes/test_tie_order.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tie_order as TO


def _engine_rank_norm(x):                 # verbatim casmi/engine.py
    o = np.argsort(-x, kind='stable'); r = np.empty(len(x)); r[o] = np.arange(len(x))
    return r / max(1, len(x) - 1)


def test_stable_matches_engine_and_average_is_order_free():
    rng = np.random.RandomState(0)
    x = rng.randint(0, 4, 30).astype(float)
    assert np.array_equal(TO.rank_norm_stable(x), _engine_rank_norm(x))
    p = rng.permutation(30)
    assert np.allclose(TO.rank_norm_average(x)[p], TO.rank_norm_average(x[p]))            # permutes with the candidates
    assert not np.allclose(TO.rank_norm_stable(x)[p], TO.rank_norm_stable(x[p]))          # the engine's rule does not
    assert abs(TO.rank_norm_average(x).mean() - 0.5) < 1e-12 and TO.rank_norm_average(np.zeros(1))[0] == 0.0


def test_rerank_and_class2_order():
    names = ['lib_max', 'lib_max_rank', 'mass_ppm', 'mass_ppm_rank', 'other']
    # one molecule, 5 candidates of one formula: rows 0-1 training structures, 2-4 COCONUT-only; truth = row 0, no library hit
    X = np.zeros((5, 5), np.float32); X[:, 2] = 1.5
    start = np.array([0, 5]); y = np.array([1, 0, 0, 0, 0], bool); pid = np.array([10, 11, 12, 13, 14]); fg = np.zeros(5, int)
    is_train = np.array([1, 1, 0, 0, 0], bool)
    X[:, 1] = _engine_rank_norm(X[:, 0].astype(float)); X[:, 3] = _engine_rank_norm(-np.abs(X[:, 2].astype(float)))
    assert np.array_equal(TO.rerank(X, names, start, 'stable'), X)                         # recomputation reproduces the stored ranks
    assert X[0, 1] == 0.0 and X[0, 3] == 0.0                                               # the leak: the truth is "best" by row order alone
    A = TO.rerank(X, names, start, 'average')
    assert np.allclose(A[:, 1], 0.5) and np.allclose(A[:, 3], 0.5) and np.array_equal(A[:, [0, 2, 4]], X[:, [0, 2, 4]])
    o = TO.class2_order(pid, y, start, fg, is_train, np.random.RandomState(1))
    assert o[0] > 2 and np.array_equal(o[1:], np.arange(1, 5))                             # truth moved behind a COCONUT-only row
    C = TO.rerank(X, names, start, 'stable', order=o)
    assert C[0, 1] >= 0.5 and C[1, 1] == 0.0                                               # the other training row is now first
    o2 = TO.class2_order(pid, y, start, fg, np.ones(5, bool), np.random.RandomState(1))    # no COCONUT-only row: after the last row
    assert o2[0] == 4.5


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
