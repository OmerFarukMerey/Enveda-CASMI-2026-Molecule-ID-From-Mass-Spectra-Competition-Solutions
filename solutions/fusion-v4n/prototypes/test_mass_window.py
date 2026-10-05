"""Tests of mass_window.py (offset table, corrected target, windows); no competition data needed.

usage: python solutions/fusion-v4n/prototypes/test_mass_window.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mass_window as MW


def test_defaults_reproduce_today():
    assert MW.DEFAULTS == dict(MASS_OFFSET_BY_ADDUCT=None, PPM_WIN=10.0, ENG_POOL_PPM=10.0, PC_PPM=10.0)
    assert MW.offsets(MW.DEFAULTS['MASS_OFFSET_BY_ADDUCT']) == {}
    nm = [300.0010, 300.0004, np.nan, 300.0007]; ad = ['[M+H]+', '[M-H]-', '[M+H]+', '[M+Na]+']
    assert MW.target_mass(nm, ad, None) == float(np.median([300.0010, 300.0004, 300.0007]))
    assert MW.target_mass(nm, ad, {}) == MW.target_mass(nm, ad, None)
    assert MW.target_mass([np.nan], ['[M+H]+']) is None and MW.target_mass([], []) is None


def test_offsets_flag():
    assert MW.offsets('enveda') == MW.ENVEDA_PPM and MW.offsets('enveda') is not MW.ENVEDA_PPM
    assert MW.offsets('none') == {} and MW.offsets('') == {}
    assert MW.offsets({'[M+H]+': 2}) == {'[M+H]+': 2.0}
    try:
        MW.offsets('orbitrap'); raise AssertionError('unknown table accepted')
    except ValueError:
        pass
    assert set(MW.ENVEDA_PPM) <= {"[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+", "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"}
    assert all(abs(v) < 3.0 for v in MW.ENVEDA_PPM.values())            # an offset, not a mislabelled adduct


def test_correction_is_multiplicative_per_adduct():
    true = np.array([150.0, 400.0, 900.0]); off = {'[M+H]+': 2.22, '[M-H]-': 0.80}
    for ad in off:
        meas = true * (1 + off[ad] * 1e-6)
        c = MW.corrected(meas, [ad] * 3, off)
        assert np.allclose(c, true, rtol=0, atol=1e-9)                  # the same ppm at every mass
    c = MW.corrected([400.0], ['[M+K]+'], off)
    assert c[0] == 400.0                                                # adduct without an entry: unchanged
    src = np.array([400.0]); MW.corrected(src, ['[M+H]+'], off); assert src[0] == 400.0   # input not modified


def test_target_uses_each_spectrum_adduct():
    true = 354.1234; off = MW.offsets('enveda')
    ads = ['[M+H]+', '[M+H]+', '[M-H]-', '[M+Na]+', '[M+CH2O2-H]-']
    meas = [true * (1 + off[a] * 1e-6) for a in ads]
    plain = MW.target_mass(meas, ads); cor = MW.target_mass(meas, ads, off)
    assert abs(cor - true) / true * 1e6 < 1e-6
    assert 0.3 < (plain - true) / true * 1e6 < 2.3                       # the plain median keeps part of the offset


def test_window_matches_the_10_ppm_rule():
    rng = np.random.RandomState(0); m = np.sort(rng.uniform(100, 900, 200000)); t = 412.3456
    a, b = MW.window(m, t, 10.0); tol = t * 10e-6
    assert a == np.searchsorted(m, t - tol, 'left') and b == np.searchsorted(m, t + tol, 'right')
    assert np.all(np.abs(m[a:b] - t) <= tol) and (a == 0 or m[a - 1] < t - tol) and (b == len(m) or m[b] > t + tol)
    a5, b5 = MW.window(m, t, 5.0)
    assert a <= a5 <= b5 <= b                                            # a narrower window is a subset
    assert MW.window(m, 50.0, 5.0) == (0, 0)


def test_fallback_widens_stepwise():
    m = np.array([100.0, 200.0, 200.0015, 300.0])
    assert MW.window_with_fallback(m, 200.0, 5.0)[:2] == (1, 2) and MW.window_with_fallback(m, 200.0, 5.0)[2] == 5.0
    a, b, p = MW.window_with_fallback(m, 200.0016, 5.0)                   # 0.5 ppm from the second row: found at 5
    assert (a, b, p) == (2, 3, 5.0)
    a, b, p = MW.window_with_fallback(m, 200.0030, 5.0)                   # 7.5 ppm away: the 10 ppm step, not 30
    assert (a, b, p) == (2, 3, 10.0)
    a, b, p = MW.window_with_fallback(m, 200.0060, 5.0)                   # 22.5 ppm away from row 2, 30 ppm reaches rows 1 and 2
    assert p == 30.0 and (a, b) == (1, 3)
    a, b, p = MW.window_with_fallback(m, 250.0, 5.0)
    assert b == a and p == 30.0
    assert MW.window_with_fallback(m, 200.0030, 10.0)[2] == 10.0          # at 10 ppm the fallbacks are today's (30 only)


def test_fit_offsets_and_loss_share():
    rng = np.random.RandomState(1)
    d = np.r_[rng.normal(2.2, 0.9, 5000), rng.normal(0.8, 0.9, 3000), rng.normal(5.0, 1.0, 20), [np.nan]]
    a = np.array(['[M+H]+'] * 5000 + ['[M-H]-'] * 3000 + ['[M+K]+'] * 20 + ['[M+H]+'], object)
    f = MW.fit_offsets(d, a, min_n=1000)
    assert set(f) == {'[M+H]+', '[M-H]-'} and abs(f['[M+H]+'] - 2.2) < 0.05 and abs(f['[M-H]-'] - 0.8) < 0.05
    e = np.abs(np.r_[rng.normal(0, 0.9, 10000), [np.nan]])
    assert MW.loss_share(e, 10) == 0.0 and 0 < MW.loss_share(e, 3) < 0.01 and MW.loss_share(e, 3) > MW.loss_share(e, 5)
    assert np.isnan(MW.loss_share([], 5))


def test_correction_recentres_a_synthetic_library():
    rng = np.random.RandomState(2); n = 20000
    true = rng.uniform(150, 900, n); ads = rng.choice(['[M+H]+', '[M-H]-', '[M+Na]+'], n, p=[0.79, 0.19, 0.02])
    off = MW.offsets('enveda'); meas = true * (1 + 1e-6 * (np.array([off[a] for a in ads]) + rng.normal(0, 0.9, n)))
    raw = np.abs(meas - true) / true * 1e6; cor = np.abs(MW.corrected(meas, ads, off) - true) / true * 1e6
    assert MW.loss_share(raw, 3) > 0.1 and MW.loss_share(cor, 3) < 0.005   # 3 ppm is only usable after the correction
    assert MW.loss_share(cor, 5) <= MW.loss_share(raw, 5)


def test_pass2_rows():
    assert MW.pass2_rows(3000) == 3000 and MW.pass2_rows(12000) == 5000 and MW.pass2_rows(12000, 20000) == 12000


if __name__ == '__main__':
    n = 0
    for k, v in sorted(globals().items()):
        if k.startswith('test_') and callable(v):
            v(); n += 1
    print(f'{n} tests passed')
