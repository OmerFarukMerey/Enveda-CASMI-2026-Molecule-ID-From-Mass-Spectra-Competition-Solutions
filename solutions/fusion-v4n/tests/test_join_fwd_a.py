"""fusion_core JOIN_FWD_A: stage-A forward-model terms of joined PubChem rows (pool id -2) at a reduced weight.

Checks, on random lists and with small stand-ins for the ICEBERG / GLACIER re-rank modules (no competition data):
off (None) and 1.0 give the reference list; lists without joined rows never change; at 0 the joined rows of a
same-formula group keep the ranker's order among themselves while the other rows are re-ranked as before.
usage: python tests/test_join_fwd_a.py   (or pytest)
"""
import math, os, random, sys

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'nb', 'embed'))
import fusion_core


def _z(vals):
    xs = [float(v) for v in vals if v is not None]
    if len(xs) < 2:
        return [0.0] * len(vals)
    mu = sum(xs) / len(xs); sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))
    return [((float(v) - mu) / sd) if (v is not None and sd > 0) else 0.0 for v in vals]


class Fwd:
    """Stand-in for fuse.py / gl_fuse.py of the Kaggle packages: the same-formula z-sum re-rank."""

    @staticmethod
    def rerank_multi(smis, keys, scores, formulas, score_dicts, lams, top_n=60, min_covered=2, tie_eps=1e-9):
        n = len(smis); order = list(range(n)); use = [(d, float(l)) for d, l in zip(score_dicts, lams) if d]
        if n == 0 or not use:
            return order
        groups = {}
        for i in range(min(int(top_n), n)):
            groups.setdefault(formulas[i], []).append(i)
        for idx in groups.values():
            if len(idx) < 2:
                continue
            cols = [([d.get(smis[i]) for i in idx], lam) for d, lam in use]
            cols = [(v, lam) for v, lam in cols if sum(x is not None for x in v) >= 2]
            if not cols:
                continue
            fused = _z([scores[i] for i in idx])
            for v, lam in cols:
                fused = [a + lam * b for a, b in zip(fused, _z(v))]
            if len(cols) >= 2 and tie_eps > 0:
                fused = [round(x / tie_eps) for x in fused]
            new = [idx[k] for k in sorted(range(len(idx)), key=lambda k: (-fused[k], k))]
            for slot, src in zip(idx, new):
                order[slot] = src
        return order

    @staticmethod
    def rerank(smis, keys, scores, formulas, ice_scores, lam=0.5, top_n=60, min_covered=2):
        return Fwd.rerank_multi(smis, keys, scores, formulas, [ice_scores], [lam], top_n)


def case(rng, p_joined=0.4):
    n = rng.randint(2, 40); smis = ['s%d' % i for i in range(n)]
    forms = [rng.choice(['F0', 'F1', 'F2']) for _ in smis]
    pids = [(-2 if rng.random() < p_joined else rng.choice([-1, 3, 7, 11])) for _ in smis]
    sc = sorted((rng.gauss(0, 1) for _ in smis), reverse=True)
    ice = {s: (None if rng.random() < 0.15 else rng.random()) for s in smis} if rng.random() < 0.9 else {}
    gl = {s: (None if rng.random() < 0.2 else rng.random()) for s in smis} if rng.random() < 0.7 else {}
    return [smis, list(smis), rng.choice([0.2, 0.5, 0.95]), sc, forms, pids], ice, gl


def final(v, ice, gl, cfg):
    sub, st = fusion_core.build_submission({'m': v}, {}, {}, {'m': ice}, {'m': gl}, cfg, ['m'], ['m'], ice_fuse=Fwd, gl_fuse=Fwd,
                                           formula=lambda s: 'F', log=lambda *a: None)
    return sub.smiles[0].split(';'), st


def test_off_and_one_are_the_reference():
    rng = random.Random(0); fired = 0
    for _ in range(1500):
        v, ice, gl = case(rng)
        ref, _ = final(v, ice, gl, {})
        assert final(v, ice, gl, dict(JOIN_FWD_A=None))[0] == ref
        one, st = final(v, ice, gl, dict(JOIN_FWD_A=1.0))
        assert one == ref, (one, ref)
        fired += st['ice'].get('join_fwd_a', 0)
    assert fired > 500, fired                                  # the weighted path actually runs


def test_lists_without_joined_rows_do_not_change():
    rng = random.Random(1)
    for _ in range(500):
        v, ice, gl = case(rng, p_joined=0.0)
        ref, _ = final(v, ice, gl, {})
        out, st = final(v, ice, gl, dict(JOIN_FWD_A=0.0))
        assert out == ref and 'join_fwd_a' not in st['ice']


def test_zero_keeps_the_ranker_order_among_joined_rows():
    # one formula; the forward models prefer the last joined row and the last pool row
    smis = ['p0', 'j0', 'p1', 'j1']; pids = [5, -2, 6, -2]; forms = ['F|gen' if p < 0 else 'F' for p in pids]
    v = [smis, list(smis), 0.2, [4.0, 3.0, 2.0, 1.0], forms, pids]
    ice = dict(p0=0.1, j0=0.1, p1=0.9, j1=0.9); gl = dict(ice)
    ref, _ = final(v, ice, gl, {})
    assert ref == ['p1', 'j1', 'p0', 'j0'], ref                # both groups are turned over by the forward models
    out, _ = final(v, ice, gl, dict(JOIN_FWD_A=0.0))
    assert out == ['p1', 'j0', 'p0', 'j1'], out                # pool rows as before, joined rows in the ranker's order
    half, _ = final(v, ice, gl, dict(JOIN_FWD_A=0.25))
    assert half == ['p1', 'j0', 'p0', 'j1'], half              # z gap of the ranker 1.41 against 0.25 x 2 x 1.41
    # a joined row is not moved past a pool row of its own group by a lower weight: mixed group, weight 0
    v2 = [smis, list(smis), 0.2, [4.0, 3.0, 2.0, 1.0], ['F'] * 4, pids]
    out2, _ = final(v2, dict(p0=0.5, j0=0.9, p1=0.5, j1=0.1), {}, dict(JOIN_FWD_A=0.0))
    assert out2 == smis, out2


if __name__ == '__main__':
    test_off_and_one_are_the_reference(); test_lists_without_joined_rows_do_not_change(); test_zero_keeps_the_ranker_order_among_joined_rows()
    print('JOIN_FWD_A OK: off and 1.0 equal the reference, lists without joined rows are untouched, 0 keeps the ranker order among joined rows')
