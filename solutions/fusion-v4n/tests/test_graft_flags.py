"""fusion_core JOIN_POP_MU / JOIN_POP_FREE / PC_PROMOTE and nb/embed/pc_pop_side.py (reviews/graft-2026-10-10.md).

Checks on random cases with a stand-in for the two re-rank modules and a synthetic PubChem tier (no competition data):
the flags are off by default and off == the reference bit for bit even when the popularity inputs are handed over;
`popularity_reorder_join` without a joined-row term equals `popularity_reorder`; in slot mode a joined row never passes
a pool row and the joined rows are ordered by z + mu x popularity; in free mode a popular joined row passes pool rows;
the promotion rule 'e006' equals its direct expression, never fires at or above LIB_TAU, puts the first row of the
popularity list first before (stage B) or after (stage C) the engine fusion and leaves every other list untouched;
'observe' moves nothing and reports the features; 'strict' follows the logistic model; the tier lookup finds rows by
mass window and SMILES text; the presets v66a / v66b / v66c are v59a plus the flags.
usage: python tests/test_graft_flags.py   (or pytest)
"""
import json, os, random, sys, tempfile

import numpy as np

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'nb', 'embed')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion_core
import pc_pop_side
from test_frag_gap import Fwd

V59 = dict(POP_MU=0.25, FRAG_LAM=0.5, FRAG_MODE='gap', ICE_LAM_LIB=0.0, GL_LAM_LIB=0.0, ICE_LAM=0.75, GL_LAM=0.75)
FORM = {}
POOL_POP = np.asarray([0.0, 1.0, 2.5, 4.0, 6.0, 9.0, 12.0, 3.0])


def case(rng, n_mol=6):
    """Base lists with pool rows (pid >= 0), joined rows (-2) and generated rows (-1), PubChem fit lists, popularity lists, engine-2 lists, scores."""
    base, pc, eng, ice, gl, jp, pp = {}, {}, {}, {}, {}, {}, {}
    for j in range(n_mol):
        m = 'm%d' % j; n = rng.randint(2, 40); smis = ['%s_s%d' % (m, i) for i in range(n)]
        for s in smis:
            FORM[s] = rng.choice(['F0', 'F1'])
        sc = sorted((rng.gauss(0, 1) for _ in smis), reverse=True)
        pids = [rng.choice([-2, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]) for _ in smis]
        lib = rng.choice([0.2, 0.5, 0.75, 0.95])
        base[m] = [smis, list(smis), lib, sc, [FORM[s] for s in smis], pids]
        jp[m] = {s: (None if rng.random() < 0.1 else rng.uniform(0, 15)) for s, p in zip(smis, pids) if p == -2}
        fit = ['%s_p%d' % (m, i) for i in range(rng.randint(0, 6))] + rng.sample(smis, min(n, 2))
        popl = ['%s_q%d' % (m, i) for i in range(rng.randint(1, 6))] + rng.sample(smis, min(n, 2)); rng.shuffle(popl)
        for s in fit + popl:
            FORM.setdefault(s, rng.choice(['F0', 'F1']))
        if fit:
            pc[m] = dict(pc=fit, pc_keys=list(fit), pc_fz=sorted((rng.gauss(10, 3) for _ in fit), reverse=True), best_pool_fz=rng.choice([None, 5.0, 12.0]), target=300.0)
        fz = sorted((rng.gauss(4, 2) for _ in popl), reverse=True)
        pp[m] = dict(pc=popl, pc_keys=list(popl), pc_fz=fz, S=fz[0], top_pop=rng.choice([0.0, 3.0, 5.0, 9.0, 14.0]), fz_top=rng.gauss(10, 3),
                     best_pool_fz=rng.choice([None, float('nan'), 5.0, 12.0, -300.0]), target=300.0)
        es = rng.sample(smis, min(n, rng.randint(0, 8))) + ['%s_e%d' % (m, i) for i in range(rng.randint(0, 4))]
        for s in es:
            FORM.setdefault(s, rng.choice(['F0', 'F1']))
        if es:
            eng[m] = dict(smiles=es, keys=list(es))
        rows = smis + es + fit + popl
        if rng.random() < 0.8:
            ice[m] = {s: (None if rng.random() < 0.15 else rng.random()) for s in rows}
            gl[m] = {s: (None if rng.random() < 0.15 else rng.random()) for s in rows}
    return base, pc, eng, ice, gl, jp, pp


def run(c, cfg, give=True):
    base, pc, eng, ice, gl, jp, pp = c; mids = list(base)
    kw = dict(join_pop=jp, pc_pop=pp) if give else {}
    sub, st = fusion_core.build_submission(base, pc, eng, ice, gl, cfg, mids, mids, ice_fuse=Fwd, gl_fuse=Fwd, score_key=lambda s: s,
                                           formula=lambda s: FORM[s], log=lambda *a: None, pool_pop=POOL_POP, **kw)
    return dict(zip(sub.molecule_id, sub.smiles)), st


def test_default_off_and_off_is_the_reference():
    D = fusion_core.DEFAULTS
    assert D['JOIN_POP_MU'] == 0.0 and D['JOIN_POP_FREE'] is False and D['PC_PROMOTE'] is None and D['PROM_STAGE'] == 'B'
    rng = random.Random(0)
    for _ in range(120):
        c = case(rng)
        for cfg in ({}, V59):
            ref, st = run(c, cfg, give=False)
            got, st2 = run(c, cfg, give=True)
            assert got == ref and 'join_pop' not in st2 and 'promote' not in st2
            assert run(c, dict(cfg, JOIN_POP_MU=0.0, PC_PROMOTE=None))[0] == ref


def test_reorder_join_reduces_to_popularity_reorder():
    rng = random.Random(1)
    for _ in range(300):
        n = rng.randint(2, 40); smis = ['s%d' % i for i in range(n)]; sc = sorted((rng.gauss(0, 1) for _ in smis), reverse=True)
        pids = [rng.choice([-2, -1, 0, 1, 2, 5, 6]) for _ in smis]; forms = [rng.choice(['A', 'B']) for _ in smis]
        ref = fusion_core.popularity_reorder(smis, smis, sc, forms, pids, POOL_POP, 0.25)
        got = fusion_core.popularity_reorder_join(smis, smis, sc, forms, pids, POOL_POP, 0.25, [None] * n, 0.25, False)
        assert got == ref


def test_slots_and_free():
    rng = random.Random(2); n_pass = 0
    for _ in range(300):
        n = rng.randint(3, 40); smis = ['s%d' % i for i in range(n)]; sc = sorted((rng.gauss(0, 1) for _ in smis), reverse=True)
        pids = [rng.choice([-2, -2, -1, 0, 1, 2, 5, 6]) for _ in smis]; forms = ['A'] * n
        jp = [rng.uniform(0, 15) if p == -2 else None for p in pids]
        s2, _, f2, fo2, p2 = fusion_core.popularity_reorder_join(smis, smis, sc, forms, pids, POOL_POP, 0.25, jp, 0.25, False)
        assert [p == -2 for p in p2] == [p == -2 for p in pids] and [p == -1 for p in p2] == [p == -1 for p in pids]   # every kind keeps its slots
        assert [s for s, p in zip(s2, p2) if p == -1] == [s for s, p in zip(smis, pids) if p == -1]
        fj = [f for f, p in zip(f2, p2) if p == -2]; fp = [f for f, p in zip(f2, p2) if p >= 0]
        assert fj == sorted(fj, reverse=True) and fp == sorted(fp, reverse=True)
        z = (np.asarray(sc) - np.mean(sc)) / (np.std(sc) + 1e-9)
        want = {s: z[i] + 0.25 * jp[i] for i, (s, p) in enumerate(zip(smis, pids)) if p == -2}
        assert all(abs(f - want[s]) < 1e-9 for s, f, p in zip(s2, f2, p2) if p == -2)
        assert all(t.endswith('|gen') == (p < 0) for t, p in zip(fo2, p2))
        s3, _, f3, _, p3 = fusion_core.popularity_reorder_join(smis, smis, sc, forms, pids, POOL_POP, 0.25, jp, 0.25, True)
        fr = [f for f, p in zip(f3, p3) if p != -1]
        assert fr == sorted(fr, reverse=True) and [s for s, p in zip(s3, p3) if p == -1] == [s for s, p in zip(smis, pids) if p == -1]
        n_pass += int([p == -2 for p in p3] != [p == -2 for p in pids])
    assert n_pass > 100                                   # free mode does move joined rows across pool rows


def rule(q, lib):
    bp = q.get('best_pool_fz'); rel = (q['fz_top'] - bp) if (bp is not None and np.isfinite(bp)) else 1e9
    return lib < 0.9 and ((q['S'] > 6.0 and q['top_pop'] >= 5.0) or rel > 250.0)


def test_promotion_e006():
    rng = random.Random(3); n_fire = n_new = 0
    for _ in range(150):
        c = case(rng); base, pp = c[0], c[6]
        ref, _ = run(c, V59)
        obs, st = run(c, dict(V59, PC_PROMOTE='observe'))
        assert obs == ref and st['promote']['promoted'] == 0
        assert set(st['promote']['feat']) == {m for m in base if base[m][2] < 0.9}
        for stage in ('B', 'C'):
            got, st2 = run(c, dict(V59, PC_PROMOTE='e006', PROM_STAGE=stage))
            for m in base:
                fire = rule(pp[m], base[m][2])
                if not fire:
                    assert got[m] == ref[m], (m, stage)
                    continue
                row = st2['promote']['feat'][m]['row']; assert row in pp[m]['pc'] and st2['promote']['feat'][m]['fire'] == 1
                lst = got[m].split(';')
                if stage == 'C':
                    assert lst[0] == row and [x for x in lst[1:]] == [x for x in ref[m].split(';') if x != row][:24]
                else:
                    assert row in lst or len(lst) == 25                       # it entered the fusion as the first row of the first list
                n_fire += 1; n_new += int(row not in ref[m].split(';'))
            assert st2['promote']['promoted'] == sum(rule(pp[m], base[m][2]) for m in base)
    assert n_fire > 100 and n_new > 20


def test_promotion_strict_and_helpers():
    assert fusion_core.promote_first(['a', 'b', 'c'], 'x', lambda s: s) == ['x', 'a', 'b', 'c']
    assert fusion_core.promote_first(['a', 'b', 'c'], 'c', lambda s: s) == ['c', 'a', 'b']
    assert fusion_core.promote_first(['a', 'B', 'c'], 'b', lambda s: s.lower()) == ['B', 'a', 'c']      # same key: the list's own row moves
    assert len(fusion_core.promote_first(list('abcdefghijklmnopqrstuvwxy'), 'z', lambda s: s)) == 25
    M = dict(feat=['top_pop', 'marg'], mean=[10.0, 1.0], sd=[2.0, 1.0], coef=[1.0, 2.0], icpt=0.0, thr=0.5)
    f = dict(S=3.0, top_pop=12.0, rel=0.0, marg=1.5, gap=0.0, lib=0.3, in25=0)
    cfg = dict(fusion_core.DEFAULTS, PC_PROMOTE='strict', PROM_MODEL=M)
    assert fusion_core.promote_decision(f, cfg) is True
    assert fusion_core.promote_decision(dict(f, top_pop=4.9), cfg) is False                 # below PROM_POP_TAU: never
    assert fusion_core.promote_decision(dict(f, marg=-2.0), cfg) is False
    assert fusion_core.promote_decision(f, dict(cfg, PROM_MODEL=None)) is False
    assert fusion_core.promote_decision(f, dict(cfg, PC_PROMOTE='observe')) is False
    e = dict(fusion_core.DEFAULTS, PC_PROMOTE='e006')
    assert fusion_core.promote_decision(dict(f, S=6.01, top_pop=5.0), e) and not fusion_core.promote_decision(dict(f, S=6.0, top_pop=50.0), e)
    assert fusion_core.promote_decision(dict(f, S=0.0, rel=250.1), e) and not fusion_core.promote_decision(dict(f, S=9.0, top_pop=4.99, rel=250.0), e)


def test_join_pop_in_the_chain():
    rng = random.Random(4); n_ch = 0
    for _ in range(120):
        c = case(rng); base, jp = c[0], c[5]
        ref, _ = run(c, V59)
        for free in (False, True):
            got, st = run(c, dict(V59, JOIN_POP_MU=0.25, JOIN_POP_FREE=free))
            for m in base:
                if not any(p == -2 for p in base[m][5]) or len(base[m][0]) < 2:
                    assert got[m] == ref[m]
            assert st['join_pop']['rows'] >= st['join_pop']['rows_found']
            n_ch += int(got != ref)
    assert n_ch > 100


def test_tier_lookup():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:      # Windows: the memory maps keep the files open
        smis = ['CCO', 'CCN', 'c1ccccc1', 'CC(=O)O', 'CCO']; mass = np.asarray([100.0, 100.0005, 100.001, 200.0, 200.0005])
        raw = ''.join(smis).encode(); off = np.cumsum([0] + [len(s) for s in smis]).astype(np.int64)
        np.save(os.path.join(td, 'pc_mass.npy'), mass); np.save(os.path.join(td, 'pc_off.npy'), off); np.save(os.path.join(td, 'pc_smiles.npy'), np.frombuffer(raw, np.uint8))
        np.save(os.path.join(td, 'pc_lsid.npy'), np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], np.float32)); np.save(os.path.join(td, 'pc_lpmid.npy'), np.asarray([0.5, 0.0, 0.0, 1.0, 2.0], np.float32))
        T = pc_pop_side.TierPop(td, td)
        assert T.pops(['CCO', 'c1ccccc1', 'XX'], 100.0005) == {'CCO': 1.5, 'c1ccccc1': 3.0}
        assert T.pops(['CCO'], 200.0002) == {'CCO': 7.0} and T.pops(['CCO'], 150.0) == {} and T.pops([], 100.0) == {} and T.pops(['CCO'], None) == {}
        assert T.pops_wide(['CCO'], 100.003) == {'CCO': 1.5}                         # 30 ppm away: found by the wide look only
        base = dict(m1=[['CCO', 'CCN', 'q'], ['a', 'b', 'c'], 0.3, [2.0, 1.0, 0.0], ['F', 'F', 'F'], [-2, 5, -2]], m2=[['CCO'], ['a'], 0.3, [1.0], ['F'], [3]])
        jp, st = pc_pop_side.join_pop_table(base, dict(m1=100.0005, m2=100.0), T)
        assert jp == {'m1': {'CCO': 1.5}} and st == dict(molecules=1, rows=2, found=1)
        pp, st = pc_pop_side.prep_pc_pop(dict(m1=dict(pc=['CCN', 'CCO'], pc_fz=[6.5, 3.0], pc_keys=['b', 'a'], target=100.0005), m2=dict(pc=[], pc_fz=[], pc_keys=[])), T)
        assert set(pp) == {'m1'} and pp['m1']['S'] == 6.5 and pp['m1']['top_pop'] == 2.0 and st == dict(lists=1, first_found=1, s_gt6_pop5=0)
        del T


def test_presets():
    ref = json.load(open(os.path.join(SOL, 'configs', 'v59a-v33b-ccby.json')))
    seen = 0
    for name in sorted(os.listdir(os.path.join(SOL, 'configs'))):
        if not name.startswith('v66'):
            continue
        c = json.load(open(os.path.join(SOL, 'configs', name))); seen += 1
        extra = {k: v for k, v in c.items() if k != 'VERSION' and ref.get(k) != v}
        assert {k: c[k] for k in ref if k != 'VERSION'} == {k: v for k, v in ref.items() if k != 'VERSION'}, name
        assert extra and set(extra) <= {'JOIN_POP_MU', 'JOIN_POP_FREE', 'PC_PROMOTE', 'PROM_STAGE', 'PROM_S_TAU', 'PROM_POP_TAU', 'PROM_REL', 'PROM_MODEL'}, (name, extra)
        assert 'FWD_GL_ALONE' not in c
    return seen


if __name__ == '__main__':
    test_default_off_and_off_is_the_reference(); test_reorder_join_reduces_to_popularity_reorder(); test_slots_and_free()
    test_promotion_e006(); test_promotion_strict_and_helpers(); test_join_pop_in_the_chain(); test_tier_lookup(); n = test_presets()
    print(f'graft flags OK: off == reference; joined-row popularity (slots / free); promotion e006 / strict / observe; tier lookup; {n} v66 presets')
