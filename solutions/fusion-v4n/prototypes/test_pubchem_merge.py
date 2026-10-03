"""Tests of prototypes/pubchem_merge.py: equal to fusion_core stage B when the new evidence is off, tier logic, popularity
is promote-only, forward margin, inputs untouched. Needs no competition data.

usage: python prototypes/test_pubchem_merge.py   (or pytest)
"""
import copy, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'nb', 'embed'))
import pubchem_merge as PM

OFF = dict(FWD_T1=None, FWD_T2=None, POP_HI=None, SLOTS_GENTLE=[4, 8, 12, 16, 20])     # the reference rule


def stage_b_reference(smis, keys, lib_max, p, c):
    """The stage-B branch of fusion_core.build_submission, copied as a function (reference behaviour)."""
    import fusion_core
    if p is None or not p.get('pc'):
        return smis
    if lib_max >= c['LIB_TAU']:
        return smis
    bp = p.get('best_pool_fz')
    top = p['fz_top'] if p.get('fz_top') is not None else p['pc_fz'][0]
    rel = top - bp if bp is not None and bp == bp and abs(bp) != float('inf') else 1e9
    return fusion_core.merge(smis, keys, p['pc'], p['pc_keys'], c['SLOTS_AGG'] if rel > c['REL_TH'] else c['SLOTS_GENTLE'])


def random_case(rng):
    nb = rng.randint(0, 40); npc = rng.randint(0, 25)
    base = [f'b{i}' for i in range(nb)]; keys = [f'kb{i}' for i in range(nb)]
    pc = [f'p{i}' for i in range(npc)]
    pck = [rng.choice(keys) if (keys and rng.random() < 0.15) else f'kp{i}' for i in range(npc)]     # some proposals are already in base
    fz = sorted((rng.uniform(-500, 900) for _ in range(npc)), reverse=True)
    p = dict(pc=pc, pc_keys=pck, pc_fz=fz, best_pool_fz=rng.choice([rng.uniform(-600, 600), float('nan'), None]),
             fz_top=rng.choice([None, (fz[0] if fz else 0.0) + rng.uniform(0, 50)]))
    if rng.random() < 0.1:
        p = None
    return base, keys, rng.choice([0.0, 0.3, 0.89, 0.9, 0.97]), p


def test_equals_fusion_core_when_evidence_is_off():
    rng = random.Random(0)
    c = dict(LIB_TAU=0.9, REL_TH=600.0, SLOTS_AGG=[2, 4, 6, 8, 10], SLOTS_GENTLE=[4, 8, 12, 16, 20])
    n_tier = {0: 0, 1: 0, None: 0}
    for _ in range(2000):
        base, keys, lib, p = random_case(rng)
        ref = stage_b_reference(base, keys, lib, p, c)[:25]
        out, info = PM.merge_pubchem(base, keys, lib, p, dict(OFF, **c))
        assert out == ref, (base, p, out, ref)
        n_tier[info['tier']] += 1
    assert n_tier[0] > 100 and n_tier[1] > 100 and n_tier[None] > 100
    print('merge_pubchem == fusion_core stage B on 2000 random cases with the evidence switched off', n_tier)


def test_inputs_not_mutated_and_length():
    rng = random.Random(1)
    for _ in range(300):
        base, keys, lib, p = random_case(rng)
        b0, k0, p0 = list(base), list(keys), copy.deepcopy(p)
        pop = {s: rng.uniform(0, 12) for s in (p or {}).get('pc', [])}
        ice = {s: rng.random() for s in base + (p or {}).get('pc', [])}
        out, _ = PM.merge_pubchem(base, keys, lib, p, dict(FWD_T1=0.0, FWD_T2=0.05, POP_HI=8.0), pop=pop, ice=ice, gl=None, formula=lambda s: 'F')
        assert base == b0 and keys == k0 and repr(p) == repr(p0)
        assert len(out) <= 25 and len(set(out)) == len(out)
        assert set(out) <= set(base) | set((p or {}).get('pc', []))
    print('inputs untouched, <= 25 distinct structures')


def _case():
    base = [f'b{i}' for i in range(30)]; keys = [f'kb{i}' for i in range(30)]
    p = dict(pc=[f'p{i}' for i in range(6)], pc_keys=[f'kp{i}' for i in range(6)], pc_fz=[100.0, 90, 80, 70, 60, 50], best_pool_fz=120.0, fz_top=None)
    return base, keys, p


def test_tiers():
    base, keys, p = _case()
    form = lambda s: 'C10H10O' if s not in ('b0', 'b1') else 'C9H6O2'
    ice = {b: 0.30 for b in base}; ice.update(b0=0.95, b1=0.9, p0=0.50)       # b0, b1: another formula, not compared
    gl = {b: 0.40 for b in base}; gl.update(p0=0.41)
    out, info = PM.merge_pubchem(base, keys, 0.2, p, dict(FWD_T1=0.0, FWD_T2=0.05), ice=ice, gl=gl, formula=form)
    assert info['tier'] == 1 and abs(info['fwd'] - 0.01) < 1e-9 and info['fwd_models'] == 2      # min over models: GLACIER margin 0.01
    assert out[:4] == ['b0', 'p0', 'b1', 'p1']
    gl['p0'] = 0.60
    out, info = PM.merge_pubchem(base, keys, 0.2, p, dict(FWD_T1=0.0, FWD_T2=0.05), ice=ice, gl=gl, formula=form)
    assert info['tier'] == 2 and out[:3] == ['p0', 'b0', 'p1']
    out, info = PM.merge_pubchem(base, keys, 0.2, p, dict(FWD_T1=0.0, FWD_T2=None), ice=ice, gl=gl, formula=form)
    assert info['tier'] == 1                                                                    # rank-1 promotion is off by default
    out, info = PM.merge_pubchem(base, keys, 0.2, p, dict(FWD_T1=0.0, FWD_T2=0.05), ice=ice, gl=gl, formula=None)
    assert info['tier'] == 0 and out[2] == 'p0'                                                 # without formulas b0 (0.95) is compared and wins
    ice['p0'] = None
    out, info = PM.merge_pubchem(base, keys, 0.2, p, dict(FWD_T1=0.0, FWD_T2=0.05), ice=ice, gl=gl, formula=form)
    assert info['fwd_models'] == 1 and info['tier'] == 2                                        # one model left: GLACIER margin 0.2
    out, info = PM.merge_pubchem(base, keys, 0.2, p, dict(FWD_T1=0.0, FWD_T2=0.05, FWD_MODELS=2), ice=ice, gl=gl, formula=form)
    assert info['tier'] == 0
    out, info = PM.merge_pubchem(base, keys, 0.2, p, dict(FWD_T1=0.0), ice=None, gl=None)
    assert info['tier'] == 0 and info['fwd'] is None and [i + 1 for i, x in enumerate(out) if x.startswith('p')] == [3, 6, 9, 12, 15]
    p2 = dict(p, best_pool_fz=-600.0)                                                           # rel = 700 > REL_TH
    out, info = PM.merge_pubchem(base, keys, 0.2, p2, dict(FWD_T1=0.0))
    assert info['tier'] == 1 and out[1] == 'p0'
    out, info = PM.merge_pubchem(base, keys, 0.95, p2, dict(FWD_T1=0.0), ice=ice, gl=gl)
    assert out == base[:25] and info['tier'] is None                                            # library hit: untouched
    out, info = PM.merge_pubchem(base, keys, 0.6, p, dict(FWD_T1=0.0, FWD_T2=0.05, FWD_LIB_MAX=0.5), ice=dict(ice, p0=0.9), gl=gl, formula=form)
    assert info['tier'] == 0 and info['fwd'] is None
    print('tier logic ok')


def test_popularity_is_promote_only():
    base, keys, p = _case()
    pop = dict(p0=1.0, p1=2.0, p2=9.0, p3=12.0, p4=1.0, p5=15.0)
    out, info = PM.merge_pubchem(base, keys, 0.2, p, dict(OFF, POP_HI=8.0, POP_K=5), pop=pop)
    pcs = [x for x in out if x.startswith('p')]
    assert pcs == ['p2', 'p3', 'p0', 'p1', 'p4'] and info['pop_front'] and info['tier'] == 1 and out[1] == 'p2'   # fz order inside both groups; p5 is outside the first 5
    out, info = PM.merge_pubchem(base, keys, 0.2, p, dict(OFF, POP_HI=8.0, POP_K=5), pop=[1.0, 2.0, 3.0, 4.0, 5.0, 15.0])
    assert [x for x in out if x.startswith('p')] == ['p0', 'p1', 'p2', 'p3', 'p4'] and info['tier'] == 0 and not info['pop_front']
    out2, _ = PM.merge_pubchem(base, keys, 0.2, p, dict(OFF, POP_HI=None), pop=pop)
    assert [x for x in out2 if x.startswith('p')] == ['p0', 'p1', 'p2', 'p3', 'p4']               # off unless POP_HI is set
    out3, _ = PM.merge_pubchem(base, keys, 0.2, p, dict(OFF, POP_HI=8.0), pop=[pop[f'p{i}'] for i in range(6)])
    assert out3 == PM.merge_pubchem(base, keys, 0.2, p, dict(OFF, POP_HI=8.0), pop=pop)[0]       # list and dict forms agree
    # an obscure answer keeps its place unless a documented proposal sits in the first POP_K
    rng = random.Random(3)
    for _ in range(500):
        pop = {f'p{i}': rng.uniform(0, 7.9) for i in range(6)}
        assert PM.merge_pubchem(base, keys, 0.2, p, dict(OFF, POP_HI=8.0), pop=pop)[0] == PM.merge_pubchem(base, keys, 0.2, p, OFF)[0]
    print('popularity: promote-only, inside the first POP_K, inert without POP_HI')


def test_forward_margin():
    base = ['a', 'b', 'c']
    assert PM.forward_margin('x', base, [{'x': 0.5, 'a': 0.2, 'b': 0.4}, None]) == (0.5 - 0.4, 1)
    assert PM.forward_margin('x', base, [{'x': 0.5, 'a': 0.2}, {'x': 0.1, 'c': 0.3}])[0] == 0.1 - 0.3
    assert PM.forward_margin('x', base, [{'a': 0.2}, {'x': None, 'c': 0.3}]) == (None, 0)
    assert PM.forward_margin('x', base, [{'x': 0.5, 'c': 0.9}], n=2) == (None, 0)                 # c is outside the compared depth
    assert PM.forward_margin('x', base, [{'x': 0.5, 'a': 0.9, 'b': 0.1}], formula={'x': 'F1', 'a': 'F2', 'b': 'F1', 'c': 'F1'}.get) == (0.4, 1)
    assert PM.forward_margin('x', base, [{'x': float('nan'), 'a': 0.9}]) == (None, 0)
    print('forward_margin ok')


def test_defaults_are_the_recommended_rule():
    d = PM.DEFAULTS
    assert d['SLOTS_GENTLE'] == [3, 6, 9, 12, 15] and d['SLOTS_AGG'] == [2, 4, 6, 8, 10] and d['FWD_T1'] == 0.0 and d['FWD_T2'] is None
    assert d['POP_HI'] == 8.0 and d['POP_K'] == 5 and d['LIB_TAU'] == 0.9 and d['REL_TH'] == 600.0
    base, keys, p = _case()
    out, info = PM.merge_pubchem(base, keys, 0.2, p)                                              # no popularity passed: the step is inert
    assert not info['pop_front'] and info['tier'] == 0
    out, info = PM.merge_pubchem(base, keys, 0.2, p, pop=dict(p1=9.5))
    assert info['pop_front'] and info['tier'] == 1 and out[1] == 'p1' and out[3] == 'p0'
    print('defaults = recommended rule')


def test_proposals_already_in_base_are_skipped():
    base, keys, p = _case()
    p = dict(p, pc_keys=['kb3', 'kp1', 'kp2', 'kp3', 'kp4', 'kp5'])                               # p0 is base candidate b3
    ice = {b: 0.3 for b in base}; ice.update(p0=0.99, p1=0.2)
    out, info = PM.merge_pubchem(base, keys, 0.2, p, dict(FWD_T1=0.0, FWD_T2=0.05), ice=ice)
    assert 'p0' not in out and info['tier'] == 0 and out[2] == 'p1'                               # evidence is read for p1, the first new proposal
    print('duplicates of base candidates never take a slot')


if __name__ == '__main__':
    test_equals_fusion_core_when_evidence_is_off()
    test_inputs_not_mutated_and_length()
    test_tiers()
    test_popularity_is_promote_only()
    test_forward_margin()
    test_proposals_already_in_base_are_skipped()
    test_defaults_are_the_recommended_rule()
