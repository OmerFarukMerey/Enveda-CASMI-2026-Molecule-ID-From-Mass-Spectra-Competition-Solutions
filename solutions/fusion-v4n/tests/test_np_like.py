"""fusion_core NP_MU / NP_JOIN_MU: the natural-product-likeness term of stage A (reviews/nplike-2026-10-08.md).

Checks, on random lists with small stand-ins for the re-rank modules (no competition data): off (keys absent, 0, or no np_score
handed in) gives the reference submission bit for bit, also with a PubChem list and an engine-2 list and at any POP_MU;
np_reorder with both weights 0 equals popularity_reorder; NP_JOIN_MU moves joined rows (pool id -2) only among the joined rows'
slots and NP_MU moves pool rows only among the pool slots; a large weight orders the rows of a class by the term; a failing or
partial scorer falls back without an error.
usage: python tests/test_np_like.py   (or pytest)
"""
import os, random, sys

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'nb', 'embed')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion_core
from test_join_fwd_a import Fwd
from test_pop_group import case, extras, POP

CFG = dict(ICE_LAM=0.75, GL_LAM=0.75, POP_MU=0.25)


def npv(v, seed=1):
    r = random.Random(seed); return {s: r.gauss(0, 1) for s in v[0]}


def run(v, ice, gl, cfg, pc=None, eng=None, np_score=None, has_np=True):
    kw = dict(np_score=np_score) if has_np else {}
    sub, st = fusion_core.build_submission({'m': v}, {'m': pc} if pc else {}, {'m': eng} if eng else {}, {'m': ice}, {'m': gl}, dict(CFG, **cfg), ['m'], ['m'],
                                           ice_fuse=Fwd, gl_fuse=Fwd, formula=lambda s: 'F', score_key=lambda s: s, log=lambda *a: None, pool_pop=POP, **kw)
    return sub.smiles[0].split(';'), st


def test_off_is_the_reference_bit_for_bit():
    rng = random.Random(0); differ = 0
    for _ in range(400):
        v, ice, gl = case(rng); pc, eng = extras(rng, v); d = npv(v)
        for kw in (dict(), dict(pc=pc), dict(pc=pc, eng=eng)):
            for mu in (0.0, 0.25, 0.35):
                ref = run(v, ice, gl, dict(POP_MU=mu), has_np=False, **kw)[0]
                assert run(v, ice, gl, dict(POP_MU=mu, NP_MU=0.0, NP_JOIN_MU=0.0), np_score=d, **kw)[0] == ref          # weights 0, scorer present
                assert run(v, ice, gl, dict(POP_MU=mu), np_score=d, **kw)[0] == ref                                    # keys absent
                assert run(v, ice, gl, dict(POP_MU=mu, NP_MU=0.5, NP_JOIN_MU=1.0), np_score=None, **kw)[0] == ref      # no scorer
                differ += int(run(v, ice, gl, dict(POP_MU=mu, NP_MU=0.5, NP_JOIN_MU=1.0), np_score=d, **kw)[0] != ref)
    assert differ > 1500, differ                                 # the term does something when it is set
    assert fusion_core.DEFAULTS['NP_MU'] == 0.0 and fusion_core.DEFAULTS['NP_JOIN_MU'] == 0.0


def test_zero_weights_equal_popularity_reorder():
    rng = random.Random(1)
    for _ in range(500):
        v, _, _ = case(rng); s = [rng.gauss(0, 1) for _ in v[0]]
        for mu in (0.25, 0.0):
            assert fusion_core.np_reorder(v[0], v[1], v[3], v[4], v[5], POP, mu, s, 0.0, 0.0) == fusion_core.popularity_reorder(v[0], v[1], v[3], v[4], v[5], POP, mu)


def test_each_term_stays_inside_its_own_slots():
    rng = random.Random(2); moved_j = moved_p = 0
    for _ in range(500):
        v, _, _ = case(rng, p_gen=0.5); s = [rng.gauss(0, 1) for _ in v[0]]; pid = v[5]
        ref = fusion_core.popularity_reorder(v[0], v[1], v[3], v[4], v[5], POP, 0.25)
        a = fusion_core.np_reorder(v[0], v[1], v[3], v[4], v[5], POP, 0.25, s, 0.0, 2.0)          # joined term only
        assert [x for x, p in zip(a[0], a[4]) if p != -2] == [x for x, p in zip(ref[0], ref[4]) if p != -2]
        assert [i for i, p in enumerate(a[4]) if p == -2] == [i for i, p in enumerate(pid) if p == -2]           # joined slots unchanged
        assert [(i, x) for i, (x, p) in enumerate(zip(a[0], a[4])) if p == -1] == [(i, x) for i, (x, p) in enumerate(zip(v[0], pid)) if p == -1]
        moved_j += int(a[0] != ref[0])
        b = fusion_core.np_reorder(v[0], v[1], v[3], v[4], v[5], POP, 0.25, s, 1.0, 0.0)          # pool term only
        assert [(i, x) for i, (x, p) in enumerate(zip(b[0], b[4])) if p < 0] == [(i, x) for i, (x, p) in enumerate(zip(ref[0], ref[4])) if p < 0]
        assert all(f.endswith('|gen') == (p < 0) for f, p in zip(b[3], b[4])) and all(f.endswith('|gen') == (p < 0) for f, p in zip(a[3], a[4]))
        moved_p += int(b[0] != ref[0])
    assert moved_j > 150 and moved_p > 300, (moved_j, moved_p)


def test_large_weight_orders_by_the_term():
    rng = random.Random(3)
    for _ in range(200):
        v, _, _ = case(rng, p_gen=0.5); s = [rng.gauss(0, 1) for _ in v[0]]; val = dict(zip(v[0], s))
        a = fusion_core.np_reorder(v[0], v[1], v[3], v[4], v[5], None, 0.0, s, 1e6, 1e6)
        for cls in (lambda p: p == -2, lambda p: p >= 0):
            xs = [val[x] for x, p in zip(a[0], a[4]) if cls(p)]
            assert xs == sorted(xs, reverse=True)


def test_scorer_forms_and_failures():
    rng = random.Random(4)
    for _ in range(100):
        v, ice, gl = case(rng); d = npv(v); cfg = dict(NP_MU=0.5, NP_JOIN_MU=1.0)
        ref = run(v, ice, gl, {}, has_np=False)[0]; on = run(v, ice, gl, cfg, np_score=d)[0]
        assert run(v, ice, gl, cfg, np_score=lambda ss: [d[x] for x in ss])[0] == on                  # callable == dict
        part = {k: x for i, (k, x) in enumerate(d.items()) if i % 2}; full0 = {k: part.get(k, 0.0) for k in d}
        assert run(v, ice, gl, cfg, np_score=part)[0] == run(v, ice, gl, cfg, np_score=full0)[0]        # unknown = 0
        assert run(v, ice, gl, cfg, np_score={k: float('nan') for k in d})[0] == run(v, ice, gl, cfg, np_score={k: 0.0 for k in d})[0]

        def boom(ss):
            raise RuntimeError('no model')
        assert run(v, ice, gl, cfg, np_score=boom)[0] == ref                                         # a failing scorer = reference


if __name__ == '__main__':
    test_off_is_the_reference_bit_for_bit(); test_zero_weights_equal_popularity_reorder(); test_each_term_stays_inside_its_own_slots()
    test_large_weight_orders_by_the_term(); test_scorer_forms_and_failures()
    print('NP-likeness term OK: off == reference bit for bit, each term stays inside its own slots, failures fall back')
