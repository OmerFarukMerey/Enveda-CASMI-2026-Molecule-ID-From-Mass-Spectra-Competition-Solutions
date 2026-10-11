"""fusion_core POP_GROUP_MU / POP_GROUP_MAX_LIB: the in-group popularity term of stage A (reviews/popgroup-2026-10-08.md).

Checks, on random lists with small stand-ins for the re-rank modules (no competition data): off (key absent, None or 0) gives the
reference submission bit for bit, also with a PubChem list and an engine-2 list and at any POP_MU; at or above POP_GROUP_MAX_LIB the
molecule is the reference; a list without two same-formula pool rows is untouched; with POP_MU 0 the stage-A order equals
gl_fuse.rerank_multi with the popularity as one more re-scorer (the form replayed in the review); a large weight puts the more
popular of two isomers first.
usage: python tests/test_pop_group.py   (or pytest)
"""
import os, random, sys

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'nb', 'embed')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import fusion_core
from test_join_fwd_a import Fwd

CFG = dict(ICE_LAM=0.75, GL_LAM=0.75, POP_MU=0.25)
_R7 = random.Random(7)
POP = np.array([_R7.uniform(0, 14) for _ in range(200)])


def case(rng, forms=('F0', 'F1', 'F2'), p_gen=0.25):
    n = rng.randint(2, 40); smis = ['s%d' % i for i in range(n)]
    fo = [rng.choice(forms) for _ in smis] if forms else ['G%d' % i for i in range(n)]
    pool = rng.sample(range(200), n)
    pids = [(rng.choice([-1, -2]) if rng.random() < p_gen else pool[i]) for i in range(n)]
    sc = sorted((rng.gauss(0, 1) for _ in smis), reverse=True)
    ice = {s: (None if rng.random() < 0.15 else rng.random()) for s in smis} if rng.random() < 0.8 else {}
    gl = {s: (None if rng.random() < 0.2 else rng.random()) for s in smis} if rng.random() < 0.7 else {}
    return [smis, list(smis), rng.choice([0.2, 0.5, 0.95]), sc, fo, pids], ice, gl


def run(v, ice, gl, cfg, pc=None, eng=None):
    sub, st = fusion_core.build_submission({'m': v}, {'m': pc} if pc else {}, {'m': eng} if eng else {}, {'m': ice}, {'m': gl}, dict(CFG, **cfg), ['m'], ['m'],
                                           ice_fuse=Fwd, gl_fuse=Fwd, formula=lambda s: 'F', score_key=lambda s: s, log=lambda *a: None, pool_pop=POP)
    return sub.smiles[0].split(';'), st


def extras(rng, v):
    pc = dict(pc=['q%d' % i for i in range(6)], pc_keys=['q%d' % i for i in range(6)], pc_fz=sorted((rng.random() for _ in range(6)), reverse=True), best_pool_fz=0.5, pc_form=['F'] * 6)
    es = rng.sample(v[0], min(len(v[0]), 5)) + ['e0', 'e1']
    return pc, dict(smiles=es, keys=list(es))


def test_off_is_the_reference_bit_for_bit():
    rng = random.Random(0); differ = 0
    for _ in range(600):
        v, ice, gl = case(rng); pc, eng = extras(rng, v)
        for kw in (dict(), dict(pc=pc), dict(pc=pc, eng=eng)):
            for mu in (0.0, 0.25, 0.35):
                ref = run(v, ice, gl, dict(POP_MU=mu), **kw)[0]
                for off in (dict(POP_GROUP_MU=None), dict(POP_GROUP_MU=0.0), dict(POP_GROUP_MU=None, POP_GROUP_MAX_LIB=0.7), dict(POP_GROUP_MU=1.5, POP_GROUP_MAX_LIB=0.0)):
                    assert run(v, ice, gl, dict(POP_MU=mu, **off), **kw)[0] == ref
                differ += int(run(v, ice, gl, dict(POP_MU=mu, POP_GROUP_MU=1.5), **kw)[0] != ref)
    assert differ > 900, differ                                  # the term does something when it is set
    assert 'POP_GROUP_MU' in fusion_core.DEFAULTS and fusion_core.DEFAULTS['POP_GROUP_MU'] is None and fusion_core.DEFAULTS['POP_GROUP_MAX_LIB'] is None


def test_library_gate():
    rng = random.Random(1); moved = 0
    for _ in range(600):
        v, ice, gl = case(rng); ref = run(v, ice, gl, {})[0]; got, st = run(v, ice, gl, dict(POP_GROUP_MU=1.5, POP_GROUP_MAX_LIB=0.7))
        if v[2] >= 0.7:
            assert got == ref and 'pop_group' not in st
        else:
            assert got == run(v, ice, gl, dict(POP_GROUP_MU=1.5))[0] and st['pop_group']['molecules'] == 1; moved += int(got != ref)
    assert moved > 150, moved


def test_list_without_same_formula_isomers_is_untouched():
    rng = random.Random(2)
    for _ in range(400):
        v, ice, gl = case(rng, forms=None)                         # every candidate has its own formula
        for mu in (0.0, 0.25):
            assert run(v, ice, gl, dict(POP_MU=mu, POP_GROUP_MU=1.5))[0] == run(v, ice, gl, dict(POP_MU=mu))[0]
    for _ in range(400):                                           # same formula, but at most one pool row per formula: nothing to compare
        v, ice, gl = case(rng); seen = set()
        for i, f in enumerate(v[4]):
            if v[5][i] >= 0 and f in seen:
                v[5][i] = -1
            seen.add(f) if v[5][i] >= 0 else None
        assert run(v, ice, gl, dict(POP_GROUP_MU=1.5))[0] == run(v, ice, gl, {})[0]


def test_equals_rerank_multi_with_popularity_as_a_rescorer():
    rng = random.Random(3); cov = lambda d: bool(d) and any(x is not None for x in d.values())
    for _ in range(800):
        v, ice, gl = case(rng); mu = rng.choice([0.5, 1.0, 1.5]); pm = rng.choice([0.0, 0.25])
        s, k, sc, fo, pid = fusion_core.popularity_reorder(v[0], v[1], v[3], v[4], v[5], POP, pm)
        pop = {x: float(POP[p]) for x, p in zip(s, pid) if p >= 0}
        terms = ([ice, gl if cov(gl) else {}] if cov(ice) else []) + [pop]; lams = ([0.75, 0.75] if cov(ice) else []) + [mu]
        o = Fwd.rerank_multi(s, k, sc, fo, terms, lams, top_n=60)
        assert run(v, ice, gl, dict(POP_MU=pm, POP_GROUP_MU=mu))[0] == [s[i] for i in o][:25]


def test_large_weight_puts_the_more_popular_isomer_first():
    v = [['a', 'b', 'c'], ['a', 'b', 'c'], 0.2, [3.0, 2.0, 1.0], ['F0', 'F0', 'F1'], [0, 1, 2]]
    pop = np.array([1.0, 9.0, 20.0])
    f = lambda cfg: fusion_core.build_submission({'m': v}, {}, {}, {}, {}, dict(CFG, POP_MU=0.0, **cfg), ['m'], ['m'], ice_fuse=Fwd, gl_fuse=Fwd, formula=lambda s: 'F',
                                                 score_key=lambda s: s, log=lambda *a: None, pool_pop=pop)[0].smiles[0].split(';')
    assert f({}) == ['a', 'b', 'c'] and f(dict(POP_GROUP_MU=10.0)) == ['b', 'a', 'c']      # c is far more popular but has another formula: it stays third
    assert f(dict(POP_GROUP_MU=0.1)) == ['a', 'b', 'c']


if __name__ == '__main__':
    test_off_is_the_reference_bit_for_bit(); test_library_gate(); test_list_without_same_formula_isomers_is_untouched()
    test_equals_rerank_multi_with_popularity_as_a_rescorer(); test_large_weight_puts_the_more_popular_isomer_first()
    print('POP_GROUP_MU OK: off == reference bit for bit, library gate, no isomers -> untouched, == rerank_multi with the popularity as a re-scorer, large weight promotes the popular isomer')
