"""fusion_core stage B with PC_FWD_T1 must equal prototypes/pubchem_merge.py (rule C of reviews/pubchem-2026-10-02.md).

Also checks that the knob is inert when off. No competition data needed.
usage: python tests/test_pc_fwd.py   (or pytest)
"""
import os, random, sys

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'nb', 'embed'))
sys.path.insert(0, os.path.join(SOL, 'prototypes'))
import fusion_core
import pubchem_merge


def case(rng):
    forms = ['F%d' % i for i in range(3)]
    n_base, n_pc = rng.randint(1, 40), rng.randint(0, 25)
    base = ['b%d' % i for i in range(n_base)]
    pc = ['p%d' % i for i in range(n_pc)]
    if pc and base and rng.random() < 0.3:                      # a PubChem proposal that is already in the base list
        pc[rng.randrange(len(pc))] = rng.choice(base)
    pc = list(dict.fromkeys(pc))
    fo = {s: rng.choice(forms) for s in base + pc}
    sc = lambda keys, p: {s: (None if rng.random() < p else rng.random()) for s in keys}
    ice = sc(base + pc, 0.2) if rng.random() < 0.85 else {}
    gl = sc(base + pc, 0.3) if rng.random() < 0.7 else {}
    v = [base, list(base), rng.choice([0.2, 0.5, 0.89, 0.95]), [float(n_base - i) for i in range(n_base)], [fo[s] for s in base]]
    p = dict(pc=pc, pc_keys=list(pc), pc_fz=[rng.uniform(0, 2000) for _ in pc],
             best_pool_fz=rng.choice([None, rng.uniform(0, 2000)]))
    if rng.random() < 0.3 and pc:
        p['fz_top'] = rng.uniform(0, 2000)
    return v, p, ice, gl, fo


def run(cfg, n=1500, seed=0):
    rng = random.Random(seed); diff = 0; early = 0
    for i in range(n):
        v, p, ice, gl, fo = case(rng)
        sub, st = fusion_core.build_submission({'m': v}, {'m': p}, {}, {'m': ice}, {'m': gl}, cfg, ['m'], ['m'],
                                               formula=fo.get, log=lambda *a: None)
        ours = sub.smiles[0].split(';')
        proto_cfg = dict(SLOTS_GENTLE=cfg.get('SLOTS_GENTLE', [4, 8, 12, 16, 20]), FWD_T1=cfg.get('PC_FWD_T1'), POP_HI=None)
        ref, info = pubchem_merge.merge_pubchem(v[0][:25], v[1][:25], v[2], p, proto_cfg, ice=ice,
                                                gl=gl if any(x is not None for x in gl.values()) else {}, formula=fo)
        diff += int(ours != ref); early += st['merge'].get('fwd_early', 0)
    return diff, early


def test_equals_prototype_rule_c():
    diff, early = run(dict(PC_FWD_T1=0.0, SLOTS_GENTLE=[3, 6, 9, 12, 15]))
    assert diff == 0, diff
    assert early > 50, early                                   # the tier actually fires in the random cases


def test_off_is_reference():
    diff, early = run(dict())
    assert diff == 0 and early == 0, (diff, early)


if __name__ == '__main__':
    test_equals_prototype_rule_c(); test_off_is_reference()
    print('PC_FWD_T1 stage B == prototype rule C on 1500 random cases; off == reference')
