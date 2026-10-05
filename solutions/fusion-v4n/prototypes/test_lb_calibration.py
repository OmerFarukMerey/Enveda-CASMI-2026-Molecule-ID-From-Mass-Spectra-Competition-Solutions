"""Tests for lb_calibration.py (no competition data): the constrained fit, the weight set, leave-one-out and predict()."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import lb_calibration as LC


def test_projection():
    assert np.allclose(LC._project(np.array([0.2, -0.1, 0.3])), [0.2, 0.0, 0.3])
    p = LC._project(np.array([0.9, 0.6, -0.2]))
    assert abs(p.sum() - 1.0) < 1e-12 and (p >= 0).all() and np.allclose(p, [0.65, 0.35, 0.0])


def test_fit_recovers_weights():
    r = np.random.RandomState(0); D = r.randn(12, 3) * 0.05; w = np.array([0.2, 0.5, 0.1])
    assert np.allclose(LC.fit_weights(D, D @ w), w, atol=1e-5)
    # the constraints bind: negative and oversized solutions are projected
    w2 = LC.fit_weights(D, D @ np.array([-0.3, 0.5, 0.1]))
    assert (w2 >= 0).all() and w2[0] < 1e-9
    w3 = LC.fit_weights(D, D @ np.array([1.5, 0.5, 0.1]))
    assert w3.sum() <= 1 + 1e-9


def test_weight_set_contains_optimum_and_grows_with_tol():
    r = np.random.RandomState(1); D = r.randn(10, 3) * 0.05; w = np.array([0.2, 0.5, 0.1]); y = D @ w
    a = LC.weight_set(D, y, 1e-8, 0.1); b = LC.weight_set(D, y, 1e-3, 0.1)
    assert any(np.allclose(x, w) for x in a) and len(b) > len(a)


def test_leave_one_out_and_skill():
    r = np.random.RandomState(2); D = r.randn(13, 3) * 0.05; w = np.array([0.2, 0.5, 0.1])
    cal = LC.calibrate(D, D @ w, [f's{i}' for i in range(13)], ['c1', 'c2', 'dB'])
    assert cal.rmse_loo < 1e-4 and cal.skill > 0.99 and len(cal.table()) == 13
    # leaderboard values unrelated to the local differences: no skill, and the module says so
    cal0 = LC.calibrate(D, r.randn(13) * 0.01, [f's{i}' for i in range(13)], ['c1', 'c2', 'dB'])
    assert cal0.skill < 0.5
    lo, hi = cal.ranges()['c2']
    assert lo <= 0.5 <= hi


def test_predict_interval():
    r = np.random.RandomState(3); D = r.randn(13, 2) * 0.05; w = np.array([0.3, 0.4])
    cal = LC.calibrate(D, D @ w, [f's{i}' for i in range(13)], ['c2', 'dB'])
    base = {'c2': {f'm{i}': 0.5 for i in range(100)}, 'dB': {f'm{i}': 0.0 for i in range(100)}}
    cfg = {'c2': {f'm{i}': 0.5 for i in range(100)}, 'dB': {f'm{i}': (1.0 if i < 20 else 0.0) for i in range(100)}}
    p = LC.predict(cal, cfg, base)
    assert abs(p['delta'] - 0.4 * 0.2) < 1e-4 and p['lo'] < p['delta'] < p['hi'] and p['reliable']
    # the public draw dominates: 130 molecules, 40% of them dB-like with a 20% chance of +1
    assert 0.02 < p['se_public'] < 0.03 and p['se_local'] < p['se_public']
    # the interval is never narrower than the leave-one-out error, and low skill is flagged
    cal.skill = 0.1; cal.rmse_loo = 0.5
    q = LC.predict(cal, cfg, base)
    assert not q['reliable'] and q['hi'] - q['delta'] >= 0.5 and q['delta'] - q['lo'] >= 0.5
    # a type missing from the config counts as no change
    assert abs(LC.predict(cal, {'c2': cfg['c2']}, base)['delta']) < 1e-9


def test_type_stats_and_public_z():
    st = LC.type_stats({'c2': {'a': 1.0, 'b': 0.5, 'c': 0.0}}, {'c2': {'a': 0.5, 'b': 0.5, 'x': 1.0}})
    assert st['c2']['n'] == 2 and abs(st['c2']['mean'] - 0.25) < 1e-12
    z = LC.public_z(-0.025, 0.0, [0.5], [0.1], [0.0], [0.005])
    assert -4.0 < z < -3.0


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print('ok', k)
