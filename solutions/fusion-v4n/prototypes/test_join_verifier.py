"""Tests of join_verifier.py without competition data: row features, the built-in logistic model, demotion (a joined row never moves up), promotion (one row, tagged -2), apply().

    python solutions/fusion-v4n/prototypes/test_join_verifier.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import join_verifier as JV


def fake(n_pool=8, n_join=6, seed=0, fwd=True):
    """A base entry with joined rows interleaved, the PubChem list they came from, and forward-score dicts."""
    rng = np.random.RandomState(seed); n = n_pool + n_join
    scs = np.sort(rng.randn(n) * 2.0)[::-1].tolist()
    pid = np.array([10 + i for i in range(n_pool)] + [-2] * n_join); pid = pid[rng.permutation(n)].tolist()
    smis = [f'C{i}' for i in range(n)]; keys = [f'K{i:02d}' for i in range(n)]
    forms = ['C6H12O6' if i % 3 else 'C7H8N2' for i in range(n)]
    jk = [keys[i] for i in range(n) if pid[i] == -2]; pc_keys = jk[::-1] + [f'X{i}' for i in range(120)]
    pc_fz = np.sort(rng.rand(len(pc_keys)) * 20.0)[::-1].tolist()
    ice = {s: (float(rng.rand()) if i % 5 else None) for i, s in enumerate(smis)} if fwd else {}
    gl = {s: float(rng.rand()) for s in smis} if fwd else None
    return [smis, keys, 0.3, scs, forms, pid], pc_keys, pc_fz, ice, gl


def feats(e, pc_keys, pc_fz, ice, gl, **kw):
    return JV.row_features(e[0], e[1], e[3], e[4], e[5], e[2], pc_keys, pc_fz, ice, gl, **kw)


def test_model_shape():
    n = len(JV.MODEL['features'])
    assert n == 56 and len(set(JV.MODEL['features'])) == n
    assert all(len(JV.MODEL[k]) == n for k in ('fill', 'clip_lo', 'clip_hi', 'mean', 'scale', 'coef'))
    assert all(lo <= hi for lo, hi in zip(JV.MODEL['clip_lo'], JV.MODEL['clip_hi'])) and min(JV.MODEL['scale']) > 0


def test_features_and_p():
    e, pk, pf, ice, gl = fake()
    f = feats(e, pk, pf, ice, gl, n_spec=3, n_adduct=2, has_mh=1, n_neg=0)
    nj = sum(p == -2 for p in e[5])
    assert set(JV.MODEL['features']) <= set(f) and all(len(f[c]) == nj for c in JV.MODEL['features'])
    assert [e[5][int(i)] for i in f['li']] == [-2] * nj
    assert np.allclose(f['rank_join'], np.arange(1, nj + 1)) and (f['gap_join'] <= 1e-12).all() and f['gap_join'][0] == 0.0
    assert (np.diff(f['pos']) > 0).all() and (f['pool_above'] == f['pos'] - f['rank_join']).all()
    p = JV.p_answer(f)
    assert p.shape == (nj,) and ((p > 0) & (p < 1)).all()


def test_no_joined_rows_and_missing_forward_scores():
    e, pk, pf, ice, gl = fake()
    e0 = [e[0], e[1], e[2], e[3], e[4], [abs(p) for p in e[5]]]
    assert feats(e0, pk, pf, ice, gl) == {} and len(JV.p_answer({})) == 0
    assert JV.apply(e0, {}) == e0
    f = feats(*fake(fwd=False)[:1], pk, pf, {}, None)
    assert (f['ice_has'] == 0).all() and (f['gl_has'] == 0).all() and np.isnan(f['fwd_z']).all()
    assert np.isfinite(JV.p_answer(f)).all()
    f2 = feats(e, [], [], ice, gl)                                  # no PubChem list: selection features are filled
    assert np.isnan(f2['sel_rank']).all() and np.isfinite(JV.p_answer(f2)).all()


def test_fz_scale_free():
    e, pk, pf, ice, gl = fake(seed=3)
    a = JV.p_answer(feats(e, pk, pf, ice, gl)); b = JV.p_answer(feats(e, pk, [30.0 * x for x in pf], ice, gl))
    assert np.allclose(a, b, atol=1e-9)


def test_demote_only_lowers_joined_rows():
    for seed in range(20):
        e, pk, pf, ice, gl = fake(n_pool=14, n_join=10, seed=seed)
        rng = np.random.RandomState(seed); p_of = {k: float(rng.rand()) for k, p in zip(e[1], e[5]) if p == -2}
        for k in (1, 3, 10, 50):
            d = JV.demote(e, p_of, 0.5, k)
            assert sorted(d[1]) == sorted(e[1]) and all(a >= b for a, b in zip(d[3][:-1], d[3][1:]))
            old = {key: i for i, key in enumerate(e[1])}; new = {key: i for i, key in enumerate(d[1])}
            low = [key for key, p in zip(e[1], e[5]) if p == -2 and p_of[key] < 0.5]
            assert all(new[key] >= old[key] for key in low)                                   # a demoted row never moves up
            nonj = [key for key, p in zip(e[1], e[5]) if p != -2]
            assert [key for key in d[1] if key in set(nonj)] == nonj                           # non-joined rows keep their order
            assert all(new[key] <= old[key] for key in nonj)
            first_k = [key for key in d[1] if key in set(nonj)][:k]
            if len(nonj) >= k:                                                                 # no low row is left above the k-th non-joined row
                assert all(new[key] > new[first_k[-1]] for key in low)
            assert dict(zip(d[1], d[5])) == dict(zip(e[1], e[5])) and dict(zip(d[1], d[0])) == dict(zip(e[1], e[0]))
        assert JV.demote(e, p_of, 0.0, 10) == e


def test_promote_one_row():
    e, pk, pf, ice, gl = fake(n_pool=10, n_join=5, seed=1)
    jk = [k for k, p in zip(e[1], e[5]) if p == -2]; p_of = {k: 0.1 for k in jk}; p_of[jk[3]] = 0.9
    u = JV.promote(e, p_of, 0.95)
    assert u[1] == e[1] and u[3] == e[3] and sorted(set(u[5]) - set(e[5])) == [-1] and -2 not in u[5]
    v = JV.promote(e, p_of, 0.6)
    assert v[1][0] == jk[3] and v[5][0] == -2 and v[5].count(-2) == 1 and sorted(v[1]) == sorted(e[1])
    assert all(a >= b for a, b in zip(v[3][:-1], v[3][1:])) and [k for k in v[1] if k != jk[3]] == [k for k in e[1] if k != jk[3]]


def test_apply():
    e, pk, pf, ice, gl = fake(n_pool=12, n_join=8, seed=5); snap = [list(x) if isinstance(x, list) else x for x in e]
    f = feats(e, pk, pf, ice, gl)
    out = JV.apply(e, f)
    assert e == snap and sorted(out[1]) == sorted(e[1]) and -2 not in out[5][1:]
    p = JV.p_answer(f); p_of = {e[1][int(i)]: float(v) for i, v in zip(f['li'], p)}
    assert out == JV.promote(e, p_of) == JV.apply(e, p_of)                      # default: promotion only
    assert JV.apply(e, f, demote_thr=0.03) == JV.promote(JV.demote(e, p_of, 0.03, 10), p_of)
    off = JV.apply(e, f, demote_thr=None, promote_thr=None)
    assert off == e


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn(); print('ok', name)
