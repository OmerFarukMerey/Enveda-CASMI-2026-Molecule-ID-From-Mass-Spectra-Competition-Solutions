"""Tests of pc_select.py without competition data: selection by f.z + prior, reserved share, defaults, the RDKit score.

    python solutions/fusion-v4n/prototypes/test_pc_select.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pc_select as PS


def test_default_is_first_n():
    fz = list(np.linspace(10, 0, 50)); pr = list(np.linspace(-2, 2, 50))
    assert PS.select(fz, pr, 10) == list(range(10))
    assert PS.select(fz, None, 10, lam=1.0) == list(range(10))
    assert PS.select(fz, pr, 100) == list(range(50)) and PS.select(fz, pr, 0) == [] and PS.select([], [], 5) == []


def test_lam_pulls_high_prior_in_and_keeps_fz_order():
    fz = list(np.linspace(10, 0, 100)); pr = np.zeros(100); pr[60] = 6.0
    s = PS.select(fz, pr, 20, lam=1.0)
    assert 60 in s and len(s) == 20 and s == sorted(s) and set(s) - {60} <= set(range(20))
    assert 60 not in PS.select(fz, pr, 20, lam=0.01)
    assert 60 not in PS.select(fz, pr, 20, lam=1.0, pool=50)               # outside the pool: cannot enter


def test_missing_prior_is_neutral():
    fz = list(np.linspace(10, 0, 40)); pr = np.full(40, np.nan)
    assert PS.select(fz, pr, 10, lam=2.0) == list(range(10))
    pr[:] = 1.0; pr[3] = np.nan
    assert PS.select(fz, pr, 10, lam=2.0) == list(range(10))               # constant prior: z = 0 everywhere


def test_reserve():
    fz = list(np.linspace(10, 0, 100)); pr = np.arange(100, dtype=float)   # the prior prefers the last candidates
    s = PS.select(fz, pr, 20, reserve=0.25)
    assert s[:15] == list(range(15)) and s[15:] == [95, 96, 97, 98, 99]
    pr2 = -np.arange(100, dtype=float)                                     # prior agrees with f.z: reserve picks the next ones
    assert PS.select(fz, pr2, 20, reserve=0.25) == list(range(20))


def test_np_likeness_orders_a_glycoside_above_a_sulfonamide():
    try:
        v = PS.np_likeness(['OCC1OC(Oc2ccc(C=CC(=O)O)cc2)C(O)C(O)C1O', 'CS(=O)(=O)Nc1ccc(F)cc1C(F)(F)F', 'not a smiles'])
    except Exception as e:                                                 # RDKit without the Contrib model
        print('NP_Score unavailable: skipped', repr(e)[:80]); return
    assert v[0] > 0 > v[1] and np.isnan(v[2])


if __name__ == '__main__':
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn(); n += 1
    print(f'{n} tests passed')
