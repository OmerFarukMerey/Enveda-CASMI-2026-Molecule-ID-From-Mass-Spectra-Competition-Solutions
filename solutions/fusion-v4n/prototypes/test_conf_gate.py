"""Tests of conf_gate.py without competition data: list features, the built-in logistic model, the depth rule, the quantile guard.

    python solutions/fusion-v4n/prototypes/test_conf_gate.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conf_gate as CG

NAMES = ['fz', 'fz_z', 'fz_gap', 'fz_softmax', 'fz_norm', 'fp_tan', 'fp_cos', 'ap', 'best_tan', 'lib_max', 'fr_int', 'sf_int', 'formula_share', 'is_gen',
         'el_sum', 'agree_ap', 'ap_gap', 'fr_gap', 'n_cand', 'n_formula', 'fz_grp_ent', 'fz_grp_gap', 'ap_grp_max', 'top_sim', 'n_analog_hi', 'lib_grp_max',
         'n_query', 'n_gen', 'q_npeaks', 'n_pos', 'n_neg', 'mass_ppm_rank']


def fake_list(n=12, seed=0, s1=4.0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, len(NAMES)).astype(np.float32)
    score = np.sort(rng.rand(n))[::-1].copy() * 2.0; score[0] = s1
    perm = rng.permutation(n)
    return score[perm], X[perm], ['C6H12O6' if i % 3 else 'C7H8N2' for i in range(n)]


def test_features_follow_the_first_candidate():
    score, X, forms = fake_list()
    f = CG.list_features(score, X, NAMES, None, None, forms)
    top = int(np.argmax(score)); s = np.sort(score)[::-1]
    assert f['s1'] == s[0] == 4.0 and abs(f['m12'] - (s[0] - s[1])) < 1e-12 and abs(f['m1_10'] - (s[0] - s[9])) < 1e-12
    assert abs(f['m15'] - (s[0] - s[4])) < 1e-12 and abs(f['s_mean5'] - s[:5].mean()) < 1e-12
    for c in ('fp_cos', 'best_tan', 'lib_max', 'is_gen', 'formula_share'):
        assert f['t1_' + c] == float(X[top, NAMES.index(c)])
    assert f['n_formula'] == float(X[top, NAMES.index('n_formula')]) and f['q_npeaks'] == float(X[top, NAMES.index('q_npeaks')])
    assert np.isnan(f['t1_fragnet__fn_rank'])                       # no family matrix given
    assert set(CG.MODEL['features']) <= set(f)                      # everything the model reads is produced


def test_no_order_coded_rank_feature_is_read():
    """The seven *_rank features of the engine code the pool row order; shuffling them must not change anything."""
    score, X, forms = fake_list(seed=1)
    X2 = X.copy(); X2[:, NAMES.index('mass_ppm_rank')] = X2[::-1, NAMES.index('mass_ppm_rank')]
    a, b = CG.list_features(score, X, NAMES, None, None, forms), CG.list_features(score, X2, NAMES, None, None, forms)
    assert all((a[k] == b[k]) or (np.isnan(a[k]) and np.isnan(b[k])) for k in a)
    assert not any(c.endswith('_rank') for c in CG.TOP1_X + CG.MOL_X)


def test_row_order_does_not_matter_without_ties():
    score, X, forms = fake_list(seed=2)
    p = np.random.RandomState(5).permutation(len(score))
    a = CG.list_features(score, X, NAMES, None, None, forms); b = CG.list_features(score[p], X[p], NAMES, None, None, [forms[i] for i in p])
    for k in CG.MODEL['features'] + ['t1_fz_fg_z', 't1_fz_fg_margin', 't1_fg_size', 'z1']:
        assert abs(a[k] - b[k]) < 1e-9, k


def test_short_and_empty_lists():
    assert CG.list_features(np.zeros(0), np.zeros((0, len(NAMES)), np.float32), NAMES) == {}
    assert CG.p_present({}) == 0.0                                  # an empty list is never "confident"
    f = CG.list_features(np.array([1.5]), np.ones((1, len(NAMES)), np.float32), NAMES, None, None, ['C2H6O'])
    assert f['m12'] == 5.0 and f['m1_10'] == 0.0 and f['t1_fz_fg_z'] == 0.0 and f['t1_fg_size'] == 0.0
    assert 0.0 < CG.p_present(f) < 1.0


def test_family_features_and_group_statistics():
    score, X, forms = fake_list(seed=3)
    F = np.arange(len(score) * 2, dtype=np.float32).reshape(-1, 2)
    f = CG.list_features(score, X, NAMES, F, ['fragnet__fn_rank', 'other__x'], forms)
    top = int(np.argmax(score)); assert f['t1_fragnet__fn_rank'] == float(F[top, 0])
    g = [i for i in range(len(forms)) if forms[i] == forms[top]]; fz = X[g, NAMES.index('fz')].astype(np.float64) * 100
    assert abs(f['t1_fg_size'] - np.log(len(g))) < 1e-12
    assert abs(f['t1_fz_fg_margin'] - (np.sort(fz)[-1] - np.sort(fz)[-2]) / 100) < 1e-9


def test_model_is_consistent_and_monotone_in_the_top_score():
    M = CG.load_model(); n = len(M['features'])
    assert all(len(M[k]) == n for k in ('mean', 'scale', 'coef', 'fill', 'clip_lo', 'clip_hi')) and min(M['scale']) > 0
    assert all(lo <= fi <= hi for lo, fi, hi in zip(M['clip_lo'], M['fill'], M['clip_hi']))
    base = dict(zip(M['features'], M['fill']))
    ps = [CG.p_present(dict(base, s1=v)) for v in (-1.0, 0.5, 2.0, 3.5, 5.0)]
    assert all(a < b for a, b in zip(ps, ps[1:])) and ps[0] < 0.4 and ps[-1] > 0.9
    assert CG.p_present(dict(base, t1_is_gen=1.0)) < CG.p_present(base)          # a generated candidate on top: less likely present
    assert CG.p_present(dict(base, m1_10=9.0)) > CG.p_present(dict(base, m1_10=1.0))
    assert CG.p_present(dict(base, s1=50.0)) == CG.p_present(dict(base, s1=M['clip_hi'][0]))   # clipped
    miss = dict(base); miss.pop('q_npeaks'); miss['t1_fp_cos'] = float('nan')
    assert CG.p_present(miss) == CG.p_present(base)                               # missing / NaN -> fill value


def test_join_depth():
    assert CG.join_depth(0.9, 0.2, None, 100, 0.7) == 100                         # no confidence gate: today's behaviour
    assert CG.join_depth(0.1, 0.8, None, 100, 0.7) == 0                           # lib_max gate first
    assert CG.join_depth(0.6, 0.2, 0.5, 100, 0.7) == 0                            # confident list: untouched
    assert CG.join_depth(0.4, 0.2, 0.5, 100, 0.7) == 100
    assert CG.join_depth(0.4, 0.8, 0.5, 100, 0.7) == 0 and CG.join_depth(0.4, 0.8, 0.5, 100, None) == 100
    assert CG.join_depth(0.1, 0.2, 0.5, 100, 0.7, conf_deep=0.3, n_deep=200) == 200
    assert CG.join_depth(0.4, 0.2, 0.5, 100, 0.7, conf_deep=0.3, n_deep=200) == 100
    assert CG.join_depth(0.9, 0.2, 0.5, 100, 0.7, conf_deep=0.3, n_deep=200) == 0
    assert CG.join_depth(0.1, 0.2, 0.5, 100, 0.7, conf_deep=0.3, n_deep=0) == 100  # deep step off


def test_quantile_threshold():
    ps = [0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 0.3, 0.7]
    for share in (0.1, 0.3, 0.5, 1.0):
        t = CG.quantile_threshold(ps, share)
        assert sum(p < t for p in ps) == int(round(share * len(ps)))
    assert CG.quantile_threshold([], 0.5) == 0.0 and CG.quantile_threshold(ps, 0.0) == 0.0


if __name__ == '__main__':
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn(); n += 1
    print(f'{n} tests passed')
