"""fusion_core FWD_GL_ALONE: a molecule without an ICEBERG score but with GLACIER scores is re-ranked by GLACIER alone.

Background (docs/landscape/canqiang-e006-2026-10-10.md): the reference re-ranks a base list only when ICEBERG scored the
molecule; if ICEBERG crashed or stopped at its budget, the GLACIER scores of that molecule are not used and the fragment
'gap' term takes over. Checks on random cases with a stand-in for the two re-rank modules (no competition data):
the flag is off by default and off == the reference bit for bit, with and without ICEBERG scores; with the flag on,
every molecule that has an ICEBERG score keeps the reference list (so a healthy run does not change); a molecule
without ICEBERG scores gets the GLACIER-only same-formula order in the base list and after the engine fusion, and the
fragment 'gap' term stays out of it; a molecule with neither model still gets the fragment term. The preset
configs/v64a-v59a-glalone.json is v59a plus the flag.
usage: python tests/test_fwd_gl_alone.py   (or pytest)
"""
import json, os, random, sys

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'nb', 'embed')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion_core
from test_frag_gap import Fwd

V59 = dict(POP_MU=0.0, FRAG_LAM=0.5, FRAG_MODE='gap', ICE_LAM_LIB=0.0, GL_LAM_LIB=0.0, ICE_LAM=0.75, GL_LAM=0.75)
FORM = {}


def case(rng, n_mol=6):
    """Stage caches of n_mol molecules: base lists, engine-2 lists, ICEBERG / GLACIER / fragment scores with gaps."""
    base, eng, ice, gl, frag = {}, {}, {}, {}, {}
    for j in range(n_mol):
        m = 'm%d' % j; n = rng.randint(2, 30); smis = ['%s_s%d' % (m, i) for i in range(n)]
        for s in smis:
            FORM[s] = rng.choice(['F0', 'F1', 'F2'])
        sc = sorted((rng.gauss(0, 1) for _ in smis), reverse=True)
        base[m] = [smis, list(smis), rng.choice([0.2, 0.5, 0.75, 0.95]), sc, [FORM[s] for s in smis], [1] * n]
        es = rng.sample(smis, min(len(smis), rng.randint(0, 8))) + ['%s_e%d' % (m, i) for i in range(rng.randint(0, 6))]
        for s in es:
            FORM.setdefault(s, rng.choice(['F0', 'F1', 'F2']))
        if es:
            eng[m] = dict(smiles=es, keys=list(es))
        rows = smis + es
        kind = rng.choice(['both', 'both', 'gl_only', 'ice_null', 'none', 'ice_only'])
        if kind in ('both', 'ice_only'):
            ice[m] = {s: (None if rng.random() < 0.15 else rng.random()) for s in rows}
        elif kind == 'ice_null':                      # the runner's all-null shell after a crash or the budget
            ice[m] = {s: None for s in rows}
        if kind in ('both', 'gl_only', 'ice_null'):
            gl[m] = {s: (None if rng.random() < 0.15 else rng.random()) for s in rows}
        frag[m] = {s: rng.random() for s in rows}
    return base, eng, ice, gl, frag


def run(c, cfg, ice=None, gl=None, frag=True):
    base, eng, i0, g0, fr = c; mids = list(base)
    sub, st = fusion_core.build_submission(base, {}, eng, i0 if ice is None else ice, g0 if gl is None else gl, cfg, mids, mids,
                                           ice_fuse=Fwd, gl_fuse=Fwd, score_key=lambda s: s, formula=lambda s: FORM[s],
                                           log=lambda *a: None, frag_scores=fr if frag else None)
    return dict(zip(sub.molecule_id, sub.smiles)), st


def has(d, m):
    return any(v is not None for v in (d.get(m) or {}).values())


def test_default_is_off_and_off_is_the_reference():
    assert fusion_core.DEFAULTS['FWD_GL_ALONE'] is False
    rng = random.Random(0)
    for _ in range(150):
        c = case(rng)
        for cfg in ({}, V59):
            ref, st = run(c, cfg)
            assert run(c, dict(cfg, FWD_GL_ALONE=False))[0] == ref and 'gl_alone' not in st


def test_on_changes_nothing_where_iceberg_scored():
    rng = random.Random(1); n_same = n_gl = 0
    for _ in range(150):
        c = case(rng); ice, gl = c[2], c[3]
        ref, _ = run(c, V59); out, st = run(c, dict(V59, FWD_GL_ALONE=True))
        for m in ref:
            if has(ice, m) or not has(gl, m):
                assert out[m] == ref[m], m; n_same += 1
            else:
                n_gl += 1
        # a run in which ICEBERG scored every molecule that GLACIER scored is unchanged as a whole
        full = dict(c[2]); full.update({m: {s: 0.5 + 0.001 * i for i, s in enumerate(gl[m])} for m in gl if not has(ice, m)})
        r2, _ = run(c, V59, ice=full); o2, s2 = run(c, dict(V59, FWD_GL_ALONE=True), ice=full)
        assert o2 == r2 and 'gl_alone' not in s2
    assert n_same > 400 and n_gl > 150, (n_same, n_gl)


def test_glacier_alone_order_and_no_fragment_term():
    rng = random.Random(2); moved = 0
    for _ in range(150):
        c = case(rng); base, eng, ice, gl, fr = c
        out, st = run(c, dict(V59, FWD_GL_ALONE=True))
        nofr, _ = run(c, dict(V59, FWD_GL_ALONE=True), frag=False)
        ref, _ = run(c, V59)
        for m in base:
            if has(ice, m) or not has(gl, m):
                continue
            assert out[m] == nofr[m], m                                   # the fragment 'gap' term is not applied
            v = base[m]; lam = 0.0 if v[2] >= 0.9 else 0.75
            o = Fwd.rerank_multi(v[0], v[1], v[3], v[4], [gl[m]], [lam], top_n=60)
            a = [v[0][i] for i in o][:25]
            if m not in eng:
                assert out[m].split(';') == a, m                          # stage A alone: the GLACIER-only order
            else:
                f, fk, fs = fusion_core.fuse_rrf(a, eng[m]['smiles'], eng[m]['keys'], lambda s: s, 0.6, 3.0, 40)
                o2 = Fwd.rerank_multi(f, fk, fs, [FORM[s] for s in f], [gl[m]], [lam], top_n=len(f))
                assert out[m].split(';') == [f[i] for i in o2][:25], m    # stage C: the same on the fused list
            moved += int(out[m] != ref[m])
        if any((not has(ice, m)) and has(gl, m) for m in base):
            assert st['gl_alone']['molecules'] >= 1
    assert moved > 50, moved


def test_neither_model_keeps_the_fragment_term():
    rng = random.Random(3); n = 0
    for _ in range(100):
        c = case(rng)
        ref, _ = run(c, V59); out, _ = run(c, dict(V59, FWD_GL_ALONE=True))
        for m in ref:
            if not has(c[2], m) and not has(c[3], m):
                assert out[m] == ref[m]; n += 1
    assert n > 50


def test_iceberg_failure_keeps_glacier():
    """ICEBERG returns nothing for the whole run: off = no forward model at all, on = the lists of a GLACIER-only run."""
    rng = random.Random(4); diff = 0
    for _ in range(60):
        c = case(rng); gl = c[3]
        off, _ = run(c, V59, ice={}); on, st = run(c, dict(V59, FWD_GL_ALONE=True), ice={})
        glonly, _ = run(c, dict(V59, FWD_GL_ALONE=True, ICE_LAM=0.0, FRAG_LAM=0.5), ice={m: dict(gl[m]) for m in gl})
        for m in on:
            if has(gl, m):
                assert on[m] == glonly[m], m; diff += int(on[m] != off[m])
    assert diff > 30, diff


def test_preset_is_v59a_plus_the_flag():
    a = json.load(open(os.path.join(SOL, 'configs', 'v59a-v33b-ccby.json'))); b = json.load(open(os.path.join(SOL, 'configs', 'v64a-v59a-glalone.json')))
    a.pop('VERSION'); assert b.pop('VERSION') == 'v64a-v59a-glalone' and b.pop('FWD_GL_ALONE') is True and a == b


if __name__ == '__main__':
    for k, f in sorted(globals().items()):
        if k.startswith('test_'):
            f(); print('ok', k)
