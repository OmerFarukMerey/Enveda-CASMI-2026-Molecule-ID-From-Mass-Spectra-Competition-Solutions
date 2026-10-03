"""Tests for prototypes/mixed_regime.py (no competition data needed). Run: python prototypes/test_mixed_regime.py"""
import os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mixed_regime as MR


def test_base_from_rows():
    b = MR.base_from_rows(['a', 'b', 'c', 'd'], ['K1', 'K2', 'K1', 'K3'], ['F', 'F', 'F', 'G'], [5, -1, 7, 9], [0.1, 0.9, 0.5, 0.9], 0.3, topn=2)
    assert b[0] == ['b', 'd'] and b[1] == ['K2', 'K3'] and b[2] == 0.3 and b[3] == [0.9, 0.9] and b[4] == ['F', 'G'] and b[5] == [-1, 9]
    b = MR.base_from_rows(['a', 'b', 'c'], ['K1', 'K2', 'K1'], ['F'] * 3, [1, 2, 3], [0.1, 0.9, 0.5], 0.0)
    assert b[0] == ['b', 'c'], 'the better-scored duplicate of a key is kept'


def test_reciprocal_rank():
    assert MR.reciprocal_rank(['x', 'y', 'z'], 'z') == 1 / 3
    assert MR.reciprocal_rank(['x'] * 25 + ['z'], 'z') == 0.0
    assert MR.reciprocal_rank([], 'z') == 0.0 and MR.reciprocal_rank([None, 'z'], 'z') == 0.5


def test_mixture_mean():
    v = {'a': {'m1': 1.0, 'm2': 0.0}, 'b': {'m1': 0.5, 'm3': 0.5}}
    mu, se = MR.mixture_mean(v, {'a': 1, 'b': 1})
    assert abs(mu - 0.5) < 1e-12 and se > 0
    mu, se = MR.mixture_mean(v, {'a': 0.0, 'b': 2.0})
    assert abs(mu - 0.5) < 1e-12 and se == 0.0
    mu, _ = MR.mixture_mean(v, {'a': 0.5, 'zero': 0.5})          # an unreachable type contributes zeros
    assert abs(mu - 0.25) < 1e-12
    # independent types: the clustered error equals the stratified error
    rng = np.random.RandomState(0)
    a = {f'x{i}': float(x) for i, x in enumerate(rng.rand(200))}; b = {f'y{i}': float(x) for i, x in enumerate(rng.rand(100))}
    mu, se = MR.mixture_mean({'a': a, 'b': b}, {'a': 0.7, 'b': 0.3})
    xa, xb = np.array(list(a.values())), np.array(list(b.values()))
    ref = np.sqrt(0.49 * xa.var() / 200 + 0.09 * xb.var() / 100)
    assert abs(se - ref) / ref < 0.01, (se, ref)
    # the same molecules in two types with opposite values: the errors cancel, an independent treatment would not see it
    c = {m: 1.0 - x for m, x in a.items()}
    mu, se = MR.mixture_mean({'a': a, 'c': c}, {'a': 0.5, 'c': 0.5})
    assert abs(mu - 0.5) < 1e-9 and se < 1e-9


def test_mixture_delta():
    a = {'t': {'m1': 1.0, 'm2': 0.5, 'm3': 0.0}}; b = {'t': {'m1': 0.5, 'm2': 0.5, 'm3': 0.0}}
    d, se = MR.mixture_delta(a, b, {'t': 1.0})
    assert abs(d - 0.5 / 3) < 1e-12 and se > 0


def test_simplex_grid():
    g = MR.simplex_grid(['a', 'b', 'c'], 0.5)
    assert len(g) == 6 and all(abs(sum(w.values()) - 1) < 1e-9 for w in g)
    g = MR.simplex_grid(['a', 'b', 'c'], 0.25, fixed={'c': 0.5})
    assert len(g) == 3 and all(w['c'] == 0.5 for w in g)
    assert MR.simplex_grid(['a'], 0.1) == [{'a': 1.0}]


def test_agreement():
    lb = [0.399, 0.404, 0.390, 0.409]
    assert MR.pair_agreement(lb, [1, 2, 0, 3]) == (4, 4)          # pairs with a gap >= 0.008: (0,2) (1,2) (2,3) (0,3)
    assert MR.pair_agreement(lb, [3, 2, 0, 1]) == (3, 4)
    assert abs(MR.spearman(lb, [1, 2, 0, 3]) - 1.0) < 1e-12
    assert abs(MR.spearman([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-12
    assert np.isnan(MR.spearman([1, 1, 1], [1, 2, 3]))


def test_scan():
    names = ['x', 'y']; tm = {'in': [0.8, 0.9], 'out': [0.2, 0.1]}
    rows = MR.scan_mixtures(tm, names, [0.41, 0.40], MR.simplex_grid(['in', 'out', 'zero'], 0.5), checks={'y_minus_x': ('y', 'x')}, min_gap=0.005)
    by = {tuple(sorted(r['weights'].items())): r for r in rows}
    r_in = by[tuple(sorted({'in': 1.0, 'out': 0.0, 'zero': 0.0}.items()))]; r_out = by[tuple(sorted({'in': 0.0, 'out': 1.0, 'zero': 0.0}.items()))]
    assert r_in['y_minus_x'] > 0 and r_in['pairs_ok'] == 0 and r_out['y_minus_x'] < 0 and r_out['pairs_ok'] == 1
    extra = MR.scan_mixtures({'in': [0.8, 0.9, 0.5]}, ['x', 'y', 'probe'], [0.41, 0.40, float('nan')], [{'in': 1.0}], checks={'c': ('probe', 'x')}, min_gap=0.005)[0]
    assert extra['pairs'] == 1 and abs(extra['c'] + 0.3) < 1e-12, 'a config without an LB value only takes part in the checks'
    half = by[tuple(sorted({'in': 0.5, 'out': 0.0, 'zero': 0.5}.items()))]
    assert abs(half['local'][0] - 0.4) < 1e-12, 'the zero type scales the level and leaves the order alone'


if __name__ == '__main__':
    for n, f in sorted(globals().items()):
        if n.startswith('test_'):
            f(); print('ok', n)
