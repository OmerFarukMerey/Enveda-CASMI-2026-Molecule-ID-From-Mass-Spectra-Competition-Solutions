"""Tests for pc_join.py (PubChem candidates joined into the v1 engine's candidate set); no competition data, no RDKit.

Run: python solutions/fusion-v4n/prototypes/test_pc_join.py   (or pytest)
"""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pc_join as PJ


def _pc(n=6):
    return dict(pc=[f'S{i}' for i in range(n)], pc_fz=[100.0 - 10 * i for i in range(n)], pc_keys=[f'K{i}' for i in range(n)],
                best_pool_fz=75.0, target=300.0, fz_top=None)


def test_select_injection_n_and_gates():
    p = _pc()
    assert PJ.select_injection(p, 0) == []
    assert PJ.select_injection(None, 5) == [] and PJ.select_injection(dict(pc=[]), 5) == []
    assert PJ.select_injection(p, 4) == [0, 1, 2, 3]
    assert PJ.select_injection(p, 50) == list(range(6))                           # N larger than the list
    assert PJ.select_injection(p, 4, lib_max=0.8, max_lib=0.7) == []             # library evidence: nothing injected
    assert PJ.select_injection(p, 4, lib_max=0.5, max_lib=0.7) == [0, 1, 2, 3]
    assert PJ.select_injection(p, 6, min_rel=0.0) == [0, 1, 2]                    # 100, 90, 80 beat the best pool f.z 75
    assert PJ.select_injection(p, 6, min_rel=20.0) == [0]
    assert PJ.select_injection(dict(p, best_pool_fz=None), 6, min_rel=0.0) == list(range(6))   # empty pool window
    assert PJ.select_injection(p, 4, absent=False) == [] and PJ.select_injection(p, 4, absent=True) == [0, 1, 2, 3]


def test_split_for_slots():
    p = _pc()
    a = PJ.split_for_slots(p, [0, 1, 2], 'all')
    assert a['pc'] == p['pc'] and a['fz_top'] == 100.0 and a['best_pool_fz'] == 75.0
    r = PJ.split_for_slots(p, [0, 1, 2], 'rest')
    assert r['pc'] == ['S3', 'S4', 'S5'] and r['pc_keys'] == ['K3', 'K4', 'K5'] and r['pc_fz'] == [70.0, 60.0, 50.0]
    assert r['fz_top'] == 100.0                                                   # the gate keeps the original top f.z
    assert PJ.split_for_slots(p, list(range(6)), 'rest') is None                  # everything was injected
    assert PJ.split_for_slots(p, [0], 'none') is None
    assert PJ.split_for_slots(p, [], 'rest', k=2)['pc'] == ['S0', 'S1']
    assert p['pc'] == [f'S{i}' for i in range(6)] and p['fz_top'] is None         # input not mutated


def test_inject_last_dedup():
    gen = [dict(key='G1'), dict(key='G2')]
    recs = [dict(key='K0'), dict(key='G2'), dict(key='P1'), dict(key='K0'), dict(key='K1')]
    out = PJ.inject_last(gen, recs, pool_keys={'P1', 'P2'})
    assert [x['key'] for x in out] == ['G1', 'G2', 'K0', 'K1']                    # generated twin, pool member and repeat dropped
    assert PJ.inject_last(None, [], set()) == []


class _Pool:
    def __init__(self):
        self.key = np.array(['P0', 'P1', 'P2'], object); self.mass = np.array([300.0, 300.0, 300.0])

    def window(self, target, ppm):
        return np.arange(3)


def _fake_engine():
    E = types.SimpleNamespace(pool=_Pool(), cfg=types.SimpleNamespace(ppm_win=10.0, ppm_fallback=30.0), calls=[])

    def generate(analogs, target, exclude_keys):
        E.calls.append(set(exclude_keys))
        return [dict(key=k, smiles=k) for k in ('G0', 'K1') if k not in exclude_keys]      # the generator can build K1

    def candidates(target, drop_pid=-1, generated=None):
        g = generated or []
        win = [c for c in range(3) if c != drop_pid]
        return types.SimpleNamespace(pid=np.array(win + [-1] * len(g)), key=[E.pool.key[c] for c in win] + [x['key'] for x in g],
                                     is_gen=np.array([0.0] * len(win) + [1.0] * len(g), np.float32))
    E.generate = generate; E.candidates = candidates
    return E


def _run(E, drop_pid=-1):
    win = [c for c in range(3) if c != drop_pid]
    gen = E.generate([], 300.0, set(E.pool.key[c] for c in win))
    return E.candidates(300.0, drop_pid, gen)


def test_install_without_injection_is_identity():
    E = _fake_engine(); PJ.install(E, None, 'first')
    C = _run(E)
    assert list(C.key) == ['P0', 'P1', 'P2', 'G0', 'K1'] and C.is_gen.tolist() == [0, 0, 0, 1, 1]
    assert not PJ.injected_mask(E, C).any() and PJ.injected_info(E) == []
    assert E.calls[-1] == {'P0', 'P1', 'P2'}


def test_install_order_first_and_after():
    recs = [dict(key='K0', smiles='K0'), dict(key='K1', smiles='K1'), dict(key='P1', smiles='x')]
    E = _fake_engine(); PJ.install(E, None, 'first'); PJ.set_injection(E, recs)
    C = _run(E)
    # 'first': the generator sees the injected keys as pool keys, so K1 is not generated; the pool member P1 is dropped
    assert list(C.key) == ['P0', 'P1', 'P2', 'G0', 'K0', 'K1'] and C.is_gen.tolist() == [0, 0, 0, 1, 0, 0]
    assert PJ.injected_mask(E, C).tolist() == [False] * 4 + [True, True] and [r['key'] for r in PJ.injected_info(E)] == ['K0', 'K1']
    assert E.calls[-1] == {'P0', 'P1', 'P2', 'K0', 'K1'}
    E = _fake_engine(); PJ.install(E, None, 'after'); PJ.set_injection(E, recs)
    C = _run(E)
    # 'after': the generator runs as today; its product K1 stays generated and the injected twin is dropped
    assert list(C.key) == ['P0', 'P1', 'P2', 'G0', 'K1', 'K0'] and C.is_gen.tolist() == [0, 0, 0, 1, 1, 0]
    assert PJ.injected_mask(E, C).tolist() == [False] * 5 + [True]
    PJ.set_injection(E, [])
    assert list(_run(E).key) == ['P0', 'P1', 'P2', 'G0', 'K1']                    # injection cleared: today's behaviour


def test_dropped_pool_row_can_be_injected():
    E = _fake_engine(); PJ.install(E, None, 'first'); PJ.set_injection(E, [dict(key='P1', smiles='truth')])
    C = _run(E, drop_pid=1)                                                       # class-3 simulation: P1 left the pool
    assert list(C.key) == ['P0', 'P2', 'G0', 'K1', 'P1'] and PJ.injected_mask(E, C).tolist() == [False] * 4 + [True]


def test_view_patch_marks_injected_as_pool_like():
    E = _fake_engine()
    V = types.SimpleNamespace(view=lambda rec, prep, target, C, X, f: types.SimpleNamespace(is_gen=(C.pid < 0).astype(np.float32)))
    PJ.install(E, V, 'first'); PJ.set_injection(E, [dict(key='K0', smiles='K0')])
    C = _run(E)
    v = V.view(None, None, 300.0, C, None, None)
    assert list(C.key) == ['P0', 'P1', 'P2', 'G0', 'K1', 'K0']
    assert v.is_gen.tolist() == [0, 0, 0, 1, 1, 0]                                # G0, K1 generated; the injected K0 is not
    PJ.set_injection(E, [])
    C = _run(E)
    assert V.view(None, None, 300.0, C, None, None).is_gen.tolist() == [0, 0, 0, 1, 1]


def test_candidate_record_with_stub_chem():
    chem = types.SimpleNamespace(standardize_smiles=lambda s: s.lower(), mol_props=lambda s: ('C2H6O', 46.04, 3),
                                 raw_fingerprint=lambda s: np.array([1, 0, 1, 1, 0], np.uint8), score_key=lambda s: 'KEY')
    frag = types.SimpleNamespace(fragments_for_smiles=lambda s: np.array([15.0, 31.0]))
    r = PJ.candidate_record('CCO', None, chem, frag, np.array([0, 2, 4]), rank=3, fz=12.5)
    assert r['smiles'] == 'cco' and r['pc_smiles'] == 'CCO' and r['key'] == 'KEY' and r['formula'] == 'C2H6O'
    assert r['fp'].dtype == np.float32 and r['fp'].tolist() == [1.0, 1.0, 0.0] and r['sim'] == 0.0 and r['parent'] == -1
    assert r['pc_rank'] == 3 and r['pc_fz'] == 12.5
    recs = PJ.build_records(['A', 'B'], ['KA', 'KB'], chem, frag, np.array([0, 1]), fz=[5.0, 4.0], cache={})
    assert [x['key'] for x in recs] == ['KA', 'KB'] and [x['pc_rank'] for x in recs] == [0, 1]
    bad = types.SimpleNamespace(standardize_smiles=lambda s: None, mol_props=lambda s: None, raw_fingerprint=lambda s: None, score_key=lambda s: None)
    assert PJ.candidate_record('??', 'K', bad, frag, np.array([0])) is None


def test_rank_two_tier():
    x = np.array([3.0, 1.0, 3.0, 1.0, 2.0])
    one = PJ.rank_two_tier(x, np.zeros(5, int))                                   # single tier: average ties
    assert np.allclose(one, np.array([0.5, 3.5, 0.5, 3.5, 2.0]) / 4)
    two = PJ.rank_two_tier(x, np.array([0, 0, 1, 1, 0]))                          # the injected copy of a tie comes second
    assert np.allclose(two, np.array([0.0, 3.0, 1.0, 4.0, 2.0]) / 4)
    assert len(PJ.rank_two_tier(np.zeros(0), np.zeros(0, int))) == 0 and PJ.rank_two_tier(np.array([5.0]), np.array([1])).tolist() == [0.0]
    perm = np.array([4, 2, 0, 3, 1])                                              # order invariance
    assert np.allclose(PJ.rank_two_tier(x[perm], np.array([0, 0, 1, 1, 0])[perm]), two[perm])


def _rows(n=9, seed=0):
    rs = np.random.RandomState(seed)
    names_x = ['lib_max', 'lib_max_rank', 'ap', 'ap_rank', 'fz', 'fz_rank', 'el_sum', 'el_rank', 'fr_int', 'fr_rank', 'sf_int', 'sf_rank', 'mass_ppm', 'mass_ppm_rank']
    X = rs.rand(n, len(names_x)).astype(np.float32)
    X[:, names_x.index('sf_int')] = np.repeat([0.5, 0.2, 0.9], 3)                 # formula-level ties
    X[:, names_x.index('lib_max')] = 0.0                                          # nobody has a library spectrum
    F = rs.rand(n, 2).astype(np.float32)
    return X, F, names_x, ['fam__a', 'fam__b']


def test_order_free_join_equals_ranker_infer_without_injection():
    import ranker_infer as RI
    X, F, nx, nf = _rows()
    assert np.array_equal(PJ.order_free_join(X, nx, None), RI.order_free(X, nx))
    assert np.array_equal(PJ.order_free_join(X, nx, np.zeros(len(X), bool)), RI.order_free(X, nx))
    inj = np.zeros(len(X), bool); inj[[2, 8]] = True
    Xj = PJ.order_free_join(X, nx, inj); Xa = RI.order_free(X, nx)
    j = nx.index('sf_rank')
    assert Xj[2, j] > Xa[2, j] and Xj[8, j] > Xa[8, j]                            # injected rows fall behind their ties
    assert Xj[0, j] < Xa[0, j] and Xj[1, j] == Xj[0, j]                           # the pool rows of the tie move up, still tied
    rank_cols = [nx.index(f) for f in PJ.RANK_SOURCES]
    assert np.array_equal(np.delete(Xj, rank_cols, 1), np.delete(X, rank_cols, 1))


class _Booster:
    def __init__(self, w):
        self.w = w

    def predict(self, M, num_threads=1):
        return M @ self.w[:M.shape[1]]


def test_choose_score_matches_ranker_infer_and_applies_tiers():
    import ranker_infer as RI
    X, F, nx, nf = _rows()
    rs = np.random.RandomState(1); w = rs.rand(40)
    pub = lambda M, names: M @ w[:M.shape[1]]
    forms = ['A'] * 3 + ['B'] * 3 + ['C'] * 3
    rk = RI.OurRanker(['x__fz', 'x__sf_rank', 'fam__a', 'e__fg_size', 'e__fz_fg_rank'], [_Booster(rs.rand(5)), _Booster(rs.rand(5))])
    none = np.zeros(len(X), bool)
    for mode in ('public', 'public_free', 'ours', 'blend'):
        for gate, lib in ((None, 0.2), (0.7, 0.2), (0.7, 0.8)):
            a = PJ.choose_score(mode, rk, forms, X, F, nx, nf, lib, pub, gate, none)
            b = RI.choose_score(mode, rk, forms, X, F, nx, nf, lib, pub, gate)
            assert np.allclose(a, b), (mode, gate, lib)
    inj = np.zeros(len(X), bool); inj[2] = True
    assert np.allclose(PJ.choose_score('public', rk, forms, X, F, nx, nf, 0.2, pub, None, inj), RI.choose_score('public', rk, forms, X, F, nx, nf, 0.2, pub))
    assert not np.allclose(PJ.choose_score('public_free', rk, forms, X, F, nx, nf, 0.2, pub, None, inj), RI.choose_score('public_free', rk, forms, X, F, nx, nf, 0.2, pub))
    s0 = PJ.choose_score('blend', rk, forms, X, F, nx, nf, 0.2, pub, None, inj)
    s1 = PJ.choose_score('blend', rk, forms, X, F, nx, nf, 0.2, pub, None, inj, bias=0.5)
    assert np.allclose(np.delete(s1, 2), np.delete(s0, 2)) and np.isclose(s0[2] - s1[2], 0.5 * np.std(s0, ddof=1))


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for f in fns:
        f()
    print(f'{len(fns)} tests passed')
