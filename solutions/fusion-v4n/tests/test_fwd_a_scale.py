"""fusion_core FWD_A_SCALE: the stage-A re-rank of the base list with scaled ICEBERG / GLACIER weights (reviews/fsranker2-2026-10-07.md).

Checks, on random lists with small stand-ins for the re-rank modules (no competition data): off (None) and [1, 1] give the reference
submission, also with a PubChem list and an engine-2 list; with [0, 0] the base list keeps the ranker's order through stage A; the
PubChem list's re-rank and stage C still receive the full weights whatever the scale is.
usage: python tests/test_fwd_a_scale.py   (or pytest)
"""
import os, random, sys

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'nb', 'embed')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion_core
from test_join_fwd_a import Fwd, case

CFG = dict(ICE_LAM=0.75, GL_LAM=0.75, POP_MU=0.0)


class Spy:
    """Fwd that records the weights each call received and the first SMILES of the list it was called on."""
    calls = []

    @staticmethod
    def rerank_multi(smis, keys, scores, formulas, score_dicts, lams, top_n=60, min_covered=2, tie_eps=1e-9):
        Spy.calls.append((smis[0] if smis else '', tuple(lams))); return Fwd.rerank_multi(smis, keys, scores, formulas, score_dicts, lams, top_n, min_covered, tie_eps)

    @staticmethod
    def rerank(smis, keys, scores, formulas, ice_scores, lam=0.5, top_n=60, min_covered=2):
        Spy.calls.append((smis[0] if smis else '', (lam,))); return Fwd.rerank_multi(smis, keys, scores, formulas, [ice_scores], [lam], top_n)


def run(v, ice, gl, cfg, pc=None, eng=None, fuse=Fwd):
    sub, st = fusion_core.build_submission({'m': v}, {'m': pc} if pc else {}, {'m': eng} if eng else {}, {'m': ice}, {'m': gl}, dict(CFG, **cfg), ['m'], ['m'],
                                           ice_fuse=fuse, gl_fuse=fuse, formula=lambda s: 'F', score_key=lambda s: s, log=lambda *a: None)
    return sub.smiles[0].split(';')


def extras(rng, v):
    pc = dict(pc=['q%d' % i for i in range(6)], pc_keys=['q%d' % i for i in range(6)], pc_fz=sorted((rng.random() for _ in range(6)), reverse=True), best_pool_fz=0.5, pc_form=['F'] * 6)
    es = rng.sample(v[0], min(len(v[0]), 5)) + ['e0', 'e1']
    return pc, dict(smiles=es, keys=list(es))


def test_off_and_one_are_the_reference():
    rng = random.Random(0); differ = 0
    for _ in range(800):
        v, ice, gl = case(rng, p_joined=0.2); pc, eng = extras(rng, v)
        ice.update({s: rng.random() for s in pc['pc'] + eng['smiles'] if ice}); gl.update({s: rng.random() for s in pc['pc'] + eng['smiles'] if gl})
        for kw in (dict(), dict(pc=pc), dict(pc=pc, eng=eng)):
            ref = run(v, ice, gl, {}, **kw)
            assert run(v, ice, gl, dict(FWD_A_SCALE=None), **kw) == ref and run(v, ice, gl, dict(FWD_A_SCALE=[1.0, 1.0]), **kw) == ref
            differ += int(run(v, ice, gl, dict(FWD_A_SCALE=[0.0, 0.0]), **kw) != ref)
    assert differ > 300, differ                                 # the scale does something when it is set


def test_zero_keeps_the_ranker_order_in_stage_a():
    rng = random.Random(1)
    for _ in range(500):
        v, ice, gl = case(rng, p_joined=0.2)
        assert run(v, ice, gl, dict(FWD_A_SCALE=[0.0, 0.0])) == v[0][:25]                        # no PubChem list, no engine 2: the base list as the ranker ordered it
        assert run(v, ice, gl, dict(FWD_A_SCALE=[0.0, 0.0])) == run(v, {}, {}, {})


def test_pubchem_list_and_stage_c_keep_the_full_weights():
    rng = random.Random(2); seen_pc = seen_c = 0
    for _ in range(300):
        v, ice, gl = case(rng, p_joined=0.2); v[2] = 0.2; pc, eng = extras(rng, v)
        ice = {s: rng.random() for s in v[0] + pc['pc'] + eng['smiles']}; gl = dict(ice)
        Spy.calls = []; run(v, ice, gl, dict(FWD_A_SCALE=[0.0, 0.5]), pc=pc, eng=eng, fuse=Spy)
        base_calls = [l for s, l in Spy.calls[:2]]; assert base_calls == [(0.0,), (0.0, 0.375)], base_calls       # base list: ICEBERG, then both
        pc_calls = [l for s, l in Spy.calls if s.startswith('q') and l in ((0.75,), (0.75, 0.75))]; seen_pc += len(pc_calls)
        assert all(l in ((0.75,), (0.75, 0.75)) for s, l in Spy.calls[2:]), Spy.calls                              # everything after the base list: full weights
        seen_c += int(len(Spy.calls) >= 6)
    assert seen_pc >= 300 and seen_c > 200, (seen_pc, seen_c)


if __name__ == '__main__':
    test_off_and_one_are_the_reference(); test_zero_keeps_the_ranker_order_in_stage_a(); test_pubchem_list_and_stage_c_keep_the_full_weights()
    print('FWD_A_SCALE OK: off and [1, 1] equal the reference, [0, 0] keeps the ranker order in stage A, the PubChem list and stage C keep the full weights')
