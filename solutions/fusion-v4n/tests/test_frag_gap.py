"""fusion_core FRAG_LAM / FRAG_MODE 'gap': the MetFrag-parsimony term acts only on molecules the forward models do not cover.

Checks, on random lists and with a small stand-in for the ICEBERG / GLACIER re-rank modules (no competition data):
FRAG_LAM 0 (the default) ignores the frag scores; in 'gap' mode a molecule with ICEBERG scores is never touched by them at
any weight; a molecule without ICEBERG scores is re-ranked inside its same-formula groups with a strength that follows
FRAG_LAM (the ranker's order at a small weight, the frag order at a large one), and rows never leave their group's slots.
The preset configs/v26a-v24a-frag025.json lowers the weight from 0.5 to 0.25 (reviews/anatomy-2026-10-06.md).
usage: python tests/test_frag_gap.py   (or pytest)
"""
import json, math, os, random, sys

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
            new = [idx[k] for k in sorted(range(len(idx)), key=lambda k: (-fused[k], k))]
            for slot, src in zip(idx, new):
                order[slot] = src
        return order

    @staticmethod
    def rerank(smis, keys, scores, formulas, ice_scores, lam=0.5, top_n=60, min_covered=2):
        return Fwd.rerank_multi(smis, keys, scores, formulas, [ice_scores], [lam], top_n)


def case(rng):
    n = rng.randint(2, 25); smis = ['s%d' % i for i in range(n)]      # at most 25 rows: the final list holds every row
    forms = [rng.choice(['F0', 'F1', 'F2']) for _ in smis]
    sc = sorted((rng.gauss(0, 1) for _ in smis), reverse=True)
    frag = {s: (None if rng.random() < 0.1 else rng.random()) for s in smis}
    ice = {s: (None if rng.random() < 0.15 else rng.random()) for s in smis}
    return [smis, list(smis), rng.choice([0.2, 0.5, 0.95]), sc, forms, [1] * n], frag, ice


def final(v, frag, ice, cfg):
    sub, st = fusion_core.build_submission({'m': v}, {}, {}, {'m': ice} if ice else {}, {}, cfg, ['m'], ['m'], ice_fuse=Fwd, gl_fuse=Fwd,
                                           formula=lambda s: 'F', log=lambda *a: None, frag_scores={'m': frag})
    return sub.smiles[0].split(';'), st


def test_zero_weight_ignores_the_frag_scores():
    rng = random.Random(0)
    for _ in range(500):
        v, frag, ice = case(rng)
        for cov in (ice, {}):
            ref, _ = final(v, {}, cov, {})
            out, st = final(v, frag, cov, dict(FRAG_LAM=0.0, FRAG_MODE='gap'))
            assert out == ref and 'frag' not in st
            assert final(v, frag, cov, {})[0] == ref                 # the default is off


def test_gap_mode_leaves_covered_molecules_alone():
    rng = random.Random(1); used = 0
    for _ in range(500):
        v, frag, ice = case(rng)
        if not any(x is not None for x in ice.values()):
            continue
        ref, _ = final(v, {}, ice, {})
        for lam in (0.25, 0.5, 4.0):
            out, st = final(v, frag, ice, dict(FRAG_LAM=lam, FRAG_MODE='gap'))
            assert out == ref and 'frag' not in st, lam
        used += 1
    assert used > 400


def test_gap_mode_strength_follows_the_weight():
    # one formula, no forward scores: the ranker prefers a, the frag term prefers c
    v = [['a', 'b', 'c'], ['a', 'b', 'c'], 0.2, [3.0, 2.0, 1.0], ['F'] * 3, [1, 2, 3]]
    frag = dict(a=0.0, b=0.5, c=1.0)
    assert final(v, frag, {}, dict(FRAG_LAM=0.25, FRAG_MODE='gap'))[0] == ['a', 'b', 'c']
    assert final(v, frag, {}, dict(FRAG_LAM=0.5, FRAG_MODE='gap'))[0] == ['a', 'b', 'c']
    out, st = final(v, frag, {}, dict(FRAG_LAM=4.0, FRAG_MODE='gap'))
    assert out == ['c', 'b', 'a'] and st['frag']['molecules'] == 1
    # rows of another formula keep their slots
    v2 = [['a', 'x', 'c'], ['a', 'x', 'c'], 0.2, [3.0, 2.0, 1.0], ['F', 'G', 'F'], [1, 2, 3]]
    assert final(v2, dict(a=0.0, x=0.0, c=1.0), {}, dict(FRAG_LAM=4.0, FRAG_MODE='gap'))[0] == ['c', 'x', 'a']


def test_random_lists_keep_their_formula_slots():
    rng = random.Random(2); moved = {0.25: 0, 0.5: 0}
    for _ in range(500):
        v, frag, _ = case(rng)
        ref, _ = final(v, {}, {}, {})
        fo = dict(zip(v[0], v[4]))
        for lam in moved:
            out, _ = final(v, frag, {}, dict(FRAG_LAM=lam, FRAG_MODE='gap'))
            assert sorted(out) == sorted(ref) and [fo[s] for s in out] == [fo[s] for s in ref]
            moved[lam] += int(out != ref)
    assert 0 < moved[0.25] <= moved[0.5], moved                    # the lower weight re-orders fewer lists


def test_preset_is_v24a_with_the_lower_weight():
    cfg = lambda n: json.load(open(os.path.join(SOL, 'configs', n), encoding='utf-8'))
    a, b = cfg('v24a-v13a-fwd075.json'), cfg('v26a-v24a-frag025.json')
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {'VERSION', 'FRAG_LAM'}, diff
    assert b['FRAG_LAM'] == 0.25 and b['FRAG_MODE'] == 'gap' and a['FRAG_LAM'] == 0.5


if __name__ == '__main__':
    test_zero_weight_ignores_the_frag_scores(); test_gap_mode_leaves_covered_molecules_alone(); test_gap_mode_strength_follows_the_weight()
    test_random_lists_keep_their_formula_slots(); test_preset_is_v24a_with_the_lower_weight()
    print('FRAG gap OK: weight 0 ignores the scores, covered molecules are untouched, the strength follows FRAG_LAM, the preset differs from v24a in FRAG_LAM only')
