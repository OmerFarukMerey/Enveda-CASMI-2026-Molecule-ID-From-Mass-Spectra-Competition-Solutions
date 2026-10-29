"""Tests for clean/isoxe_gate.py (isoxe3 list-aware gate): no competition data, no RDKit, no model."""
import os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'clean'))
import isoxe_gate as G


def fz_ref(smis, xe, frozen, formula, a=1.0, lock1=False):
    """Verbatim copy of scratch/land1027/fz_core.fz_list (the shipped rule) for the equivalence check."""
    smis = list(smis)
    groups = {}
    for i, s in enumerate(smis):
        if frozen[i] or formula[i] is None or (lock1 and i == 0):
            continue
        v = xe.get(s)
        if v is None or v != v:
            continue
        groups.setdefault(formula[i], []).append(i)
    out = list(smis)
    for f, idx in groups.items():
        if len(idx) < 2:
            continue
        order = sorted(idx, key=lambda i: -xe[smis[i]])
        xr = {i: r + 1 for r, i in enumerate(order)}
        new = sorted(idx, key=lambda i: (-(1.0 / (3 + i + 1) + a / (3 + xr[i])), i))
        for slot, i in zip(sorted(idx), new):
            out[slot] = smis[i]
    return out


def case(seed, n=12):
    rng = np.random.default_rng(seed)
    smis = [f'C{i}' for i in range(n)]
    formula = [rng.choice(['F1', 'F2', 'F3', None]) for _ in range(n)]
    frozen = [bool(rng.random() < 0.2) for _ in range(n)]
    xe = {s: float(rng.normal()) for s in smis if rng.random() < 0.85}
    ev = {s: dict(main_rank=int(rng.integers(0, 5)), engine2_rank=int(rng.integers(0, 3)), pubchem_rank=int(rng.integers(0, 4)),
                  gl=float(rng.random()) if rng.random() < 0.7 else None, ice=None) for s in smis}
    return smis, xe, frozen, formula, ev


def test_ungated_equals_fz_core():
    for seed in range(40):
        smis, xe, frozen, formula, ev = case(seed)
        for a in (0.5, 1.0, 2.0):
            for lock in (False, True):
                got, _ = G.gated_fz_list(smis, xe, frozen, formula, ev, 0.3, a=a, lock1=lock, gate=None)
                assert got == fz_ref(smis, xe, frozen, formula, a=a, lock1=lock)
                assert sorted(got) == sorted(smis)


def test_default_gate_never_moves():
    g = G.Gate()
    assert not g([0.0] * len(G.FEATURES)) and g.proba([0.0] * len(G.FEATURES)) == 0.0
    for seed in range(20):
        smis, xe, frozen, formula, ev = case(seed)
        got, info = G.gated_fz_list(smis, xe, frozen, formula, ev, 0.3, a=1.0, gate=g)
        assert got == smis and info['moved_groups'] == 0


def test_frozen_and_lock_never_move():
    for seed in range(40):
        smis, xe, frozen, formula, ev = case(seed)
        always = G.Gate(coef=np.zeros(len(G.FEATURES)), intercept=5.0)
        got, _ = G.gated_fz_list(smis, xe, frozen, formula, ev, 0.3, a=2.0, lock1=True, gate=always)
        assert got[0] == smis[0]
        assert all(x == y for x, y, f in zip(smis, got, frozen) if f)


def test_features():
    smis = ['A', 'B', 'C', 'D']
    xe = {'A': 0.0, 'B': 2.0, 'C': 1.0, 'D': 5.0}
    ev = {'A': dict(main_rank=1, gl=0.2, engine2_rank=1), 'B': dict(main_rank=0, pubchem_rank=3, gl=0.9)}
    x = dict(zip(G.FEATURES, G.group_features(smis, [0, 1, 2], xe, ev, 0.4)))
    assert x['top1'] == 1.0 and x['lpos_top'] == 0.0 and abs(x['lsize'] - np.log(3)) < 1e-12
    assert x['fav_hold'] == 0.0 and x['zmargin'] > 0 and x['lpos_fav'] == np.log(2)
    assert x['hold_main'] == 1.0 and x['fav_main'] == 0.0 and x['fav_pconly'] == 1.0
    assert x['gl_sign'] == 1.0 and x['ice_sign'] == 0.0 and x['eng2'] == -1.0 and x['lib_max'] == 0.4
    assert x['rho'] < 1.0
    y = dict(zip(G.FEATURES, G.group_features(smis, [1, 2], xe, ev, 0.0)))   # v2 agrees with list: B above C
    assert y['fav_hold'] == 1.0 and y['zmargin'] == 0.0 and abs(y['rho'] - 1.0) < 1e-9 and y['gl_sign'] == 0.0 and y['top1'] == 0.0


def test_gate_params_roundtrip():
    g = G.Gate(coef=np.arange(len(G.FEATURES)) * 0.1, intercept=-0.3, mean=np.ones(len(G.FEATURES)), scale=np.full(len(G.FEATURES), 2.0))
    g2 = G.Gate(**g.params())
    x = np.linspace(-1, 1, len(G.FEATURES))
    assert abs(g.proba(x) - g2.proba(x)) < 1e-12 and g(x) == g2(x)
