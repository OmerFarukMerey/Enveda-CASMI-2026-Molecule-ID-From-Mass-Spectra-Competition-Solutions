"""Tests of pc_window.py without competition data: offset table, shifted target, window bounds, loss share.

    python solutions/fusion-v4n/prototypes/test_pc_window.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pc_window as PW


def test_defaults_are_today():
    assert PW.DEFAULTS == dict(PC_PPM=10.0, PC_MASS_SHIFT=None, PC_N1=5000)
    nm = [300.0010, 300.0012, 300.0050]
    assert PW.shifted_target(nm, ['[M+H]+'] * 3, None) == float(np.median(nm))
    assert PW.shifted_target(nm, ['[M+H]+'] * 3, {}) == float(np.median(nm))


def test_shifted_target():
    nm = [300.00072, 300.00026, np.nan]
    t = PW.shifted_target(nm, ['[M+H]+', '[M-H]-', '[M+H]+'], PW.ENVEDA_SHIFT)
    assert abs(t - 300.0) < 1e-9
    assert PW.shifted_target([np.nan], ['[M+H]+'], PW.ENVEDA_SHIFT) is None
    assert abs(PW.shifted_target([300.001], ['[M+K]+'], PW.ENVEDA_SHIFT) - 300.001) < 1e-12      # unknown adduct: no correction


def test_estimate_shifts():
    rng = np.random.default_rng(0)
    d = np.r_[rng.normal(0.0007, 0.0003, 5000), rng.normal(0.0003, 0.0003, 3000), rng.normal(0.01, 0.001, 50), [np.nan] * 10]
    a = ['[M+H]+'] * 5000 + ['[M-H]-'] * 3000 + ['[M+K]+'] * 50 + ['[M+H]+'] * 10
    s = PW.estimate_shifts(d, a, min_n=2000)
    assert set(s) == {'[M+H]+', '[M-H]-'} and abs(s['[M+H]+'] - 0.0007) < 3e-5 and abs(s['[M-H]-'] - 0.0003) < 3e-5


def test_window():
    mass = np.sort(np.r_[np.linspace(299.99, 300.01, 2001), 300.0])
    a10, b10 = PW.window(mass, 300.0, 10.0); a5, b5 = PW.window(mass, 300.0, 5.0)
    assert a10 < a5 < b5 < b10
    assert np.all(np.abs(mass[a5:b5] - 300.0) <= 300.0 * 5e-6 + 1e-12)
    assert (a5 == 0 or abs(mass[a5 - 1] - 300.0) > 300.0 * 5e-6) and (b5 == len(mass) or abs(mass[b5] - 300.0) > 300.0 * 5e-6)
    assert abs((b5 - a5) / (b10 - a10) - 0.5) < 0.01
    assert PW.window(mass, 500.0, 5.0)[0] == PW.window(mass, 500.0, 5.0)[1] == len(mass)   # empty window


def test_loss_share():
    e = [0.5, 1.0, 2.0, 4.9, 5.1, 12.0, np.nan]
    assert abs(PW.loss_share(e, 5.0) - 2 / 6) < 1e-12 and abs(PW.loss_share(e, 10.0) - 1 / 6) < 1e-12
    assert np.isnan(PW.loss_share([], 5.0))


if __name__ == '__main__':
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn(); n += 1
    print(f'{n} tests passed')
