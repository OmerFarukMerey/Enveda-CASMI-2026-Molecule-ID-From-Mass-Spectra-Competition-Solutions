"""Tests of eng_join.py without competition data: outside-pool selection, the slot move in the fusion, record merging.

    python solutions/fusion-v4n/prototypes/test_eng_join.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nb', 'embed'))
import eng_join as EJ


def rrf(v_smis, e_smis, e_keys, score_key, alpha, krr, n):
    """fusion_core.fuse_rrf, copied so that the test also runs where pandas is missing."""
    sc, smi_of = {}, {}
    for r, s in enumerate(v_smis, 1):
        k = score_key(s) or s
        sc[k] = sc.get(k, 0.0) + 1.0 / (krr + r); smi_of.setdefault(k, s)
    for r, (s, k) in enumerate(zip(e_smis, e_keys), 1):
        if not k:
            continue
        sc[k] = sc.get(k, 0.0) + alpha / (krr + r); smi_of.setdefault(k, s)
    order = sorted(sc, key=lambda k: -sc[k])[:n]
    return [smi_of[k] for k in order], order, [sc[k] for k in order]


KEY = lambda s: 'K' + s
V = [f'v{i}' for i in range(1, 11)]                       # v1-side list, keys Kv1..Kv10
POOL = {KEY(s) for s in V} | {'Kp1', 'Kp2'}


def test_outside_pool():
    assert EJ.outside_pool(['Kv1', 'Kx', 'Ky', 'Kz'], POOL, top=3, n=1) == ['Kx']
    assert EJ.outside_pool(['Kv1', 'Kx', 'Ky', 'Kz'], POOL, top=3, n=2) == ['Kx', 'Ky']
    assert EJ.outside_pool(['Kv1', 'Kp1', 'Kp2', 'Kz'], POOL, top=3) == []          # Kz is beyond the first three
    assert EJ.outside_pool([None, '', 'Kx'], POOL, top=3) == ['Kx']
    assert EJ.outside_pool([], POOL) == []


def test_slot_off_is_identity():
    assert EJ.slot_fuse(rrf, POOL, None) is rrf


def test_slot_moves_only_outside_pool_candidates_up():
    e_smis = ['x', 'v2', 'v1']; e_keys = ['Kx', 'Kv2', 'Kv1']
    s0, o0, c0 = rrf(V, e_smis, e_keys, KEY, 0.6, 3.0, 40)
    assert o0.index('Kx') > 2                              # plain fusion: the engine-only candidate is not in the first three
    s2, o2, c2 = EJ.slot_fuse(rrf, POOL, 2)(V, e_smis, e_keys, KEY, 0.6, 3.0, 40)
    assert o2[1] == 'Kx' and s2[1] == 'x' and sorted(o2) == sorted(o0) and len(s2) == len(s0)
    assert [k for k in o2 if k != 'Kx'] == [k for k in o0 if k != 'Kx']            # everything else keeps its order
    assert c2 == sorted(c2, reverse=True)
    s1, o1, _ = EJ.slot_fuse(rrf, POOL, 1)(V, e_smis, e_keys, KEY, 0.6, 3.0, 40)
    assert o1[0] == 'Kx'


def test_slot_never_moves_down_and_ignores_pool_candidates():
    e_smis = ['x'] + V[:3]; e_keys = ['Kx'] + [KEY(s) for s in V[:3]]
    f = EJ.slot_fuse(rrf, POOL, 5)
    s, o, _ = f(['x'] + V, e_smis, e_keys, KEY, 0.6, 3.0, 40)                       # already first: stays first
    assert o[0] == 'Kx'
    e2 = ['v9', 'v8', 'v7']; k2 = [KEY(x) for x in e2]                              # engine-2's favourites are pool rows: no move
    assert EJ.slot_fuse(rrf, POOL, 1)(V, e2, k2, KEY, 0.6, 3.0, 40) == rrf(V, e2, k2, KEY, 0.6, 3.0, 40)


def test_slot_top_limit_and_two_candidates():
    e_smis = ['v1', 'v2', 'v3', 'x']; e_keys = ['Kv1', 'Kv2', 'Kv3', 'Kx']
    assert EJ.slot_fuse(rrf, POOL, 2, top=3)(V, e_smis, e_keys, KEY, 0.6, 3.0, 40)[1] == rrf(V, e_smis, e_keys, KEY, 0.6, 3.0, 40)[1]
    assert EJ.slot_fuse(rrf, POOL, 2, top=4)(V, e_smis, e_keys, KEY, 0.6, 3.0, 40)[1][1] == 'Kx'
    e_smis = ['x', 'y', 'v1']; e_keys = ['Kx', 'Ky', 'Kv1']
    o = EJ.slot_fuse(rrf, POOL, 2, top=3, n=2)(V, e_smis, e_keys, KEY, 0.6, 3.0, 40)[1]
    assert o[1:3] == ['Kx', 'Ky']


def test_outside_window():
    e = dict(smiles=['a', 'b', 'c', 'd', 'e'], keys=['K1', 'K2', None, 'K2', 'K3'])
    assert EJ.outside_window(e, {'K1'}, 5) == [1, 4]
    assert EJ.outside_window(e, {'K1'}, 1) == [1]
    assert EJ.outside_window(e, set(), 0) == [] and EJ.outside_window(None, set(), 5) == []


def test_merge_and_join_records():
    a = [dict(key='K1', mass=300.0), dict(key='K2', mass=300.001)]
    b = [dict(key='K2', mass=300.001, pc=True), dict(key='K3', mass=300.02), dict(key='K4', mass=299.999)]
    m = EJ.merge_records(a, b, target=300.0, ppm=30.0)
    assert [r['key'] for r in m] == ['K1', 'K2', 'K4'] and 'pc' not in m[1]         # first source wins; K3 is 67 ppm away
    e = dict(smiles=['s1', 's2', 's3'], keys=['Kw', 'K9', 'K2'])
    build = lambda smis, keys: [dict(key=k, mass=300.0, smiles=s) for s, k in zip(smis, keys)]
    j = EJ.join_records(e, {'Kw'}, 5, build, pc_records=b, target=300.0)
    assert [r['key'] for r in j] == ['K9', 'K2', 'K4'] and j[0]['src'] == 'eng' and 'pc' not in j[1]
    j = EJ.join_records(e, {'Kw'}, 5, build, pc_records=b, target=300.0, lib_max=0.8, max_lib=0.7)
    assert [r['key'] for r in j] == ['K2', 'K4'] and j[0].get('pc')                 # gated: PubChem records only
    assert EJ.join_records(None, set(), 5, build, pc_records=[], target=300.0) == []


if __name__ == '__main__':
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn(); n += 1
    print(f'{n} tests passed')
