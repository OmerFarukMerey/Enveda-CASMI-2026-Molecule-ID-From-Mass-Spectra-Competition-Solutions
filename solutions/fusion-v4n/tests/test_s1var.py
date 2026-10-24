"""Tests of the two s1 variants of debate round 4 (nb/embed/s1i_icegroup.py, nb/embed/s1m_shield.py, scratch/s1var/build_s1var.py); synthetic data only.

s1i: the formula-group rule (only groups with 0 finite GLACIER and >= 2 finite ICEBERG move, by the chain's ICE re-rank, on their own slots),
the self-test, switch-off / self-test-failure / per-molecule fallback, and s1's real cell 53 patched vs unpatched on random windows
(identical unless the switch is on, the molecule is not an N1 proxy and ice_fallback is True).
s1m: the row rule equals "shield on E006, then E2R2 again" on random lists, the file cell (applied / off / tau missing / per-molecule
fallback), the capture cell, and the builder (inserted cells, every other s1 cell unchanged).
The cell-53 and builder tests need s1's notebook (scratch/land1024/sub/casmi26-sub-s1, not in git) and are skipped without it.
usage: python solutions/fusion-v4n/tests/test_s1var.py   (or pytest)
"""
import io, json, math, os, random, sys, tempfile, textwrap

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(SOL))
EMBED = os.path.join(SOL, 'nb', 'embed')
S1_NB = os.path.join(ROOT, 'scratch', 'land1024', 'sub', 'casmi26-sub-s1', 'casmi26-sub-s1.ipynb')
HAVE_S1 = os.path.exists(S1_NB)


def load(name, **extra):
    g = {'__name__': 's1var_test'}
    g.update(extra)
    exec(compile(open(os.path.join(EMBED, name), encoding='utf-8').read(), name, 'exec'), g)
    return g


I = load('s1i_icegroup.py')
M = load('s1m_shield.py')


def _z(vals):
    ok = [v is not None and math.isfinite(float(v)) for v in vals]
    xs = [float(v) for v, o in zip(vals, ok) if o]
    if len(xs) < 2:
        return [0.0] * len(vals)
    mu = sum(xs) / len(xs); sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))
    return [((float(v) - mu) / sd) if (o and sd > 0) else 0.0 for v, o in zip(vals, ok)]


class Fwd:
    """Stand-in for fuse.py / gl_fuse.py (same-formula z-sum re-rank on the group's own slots; dict or aligned list)."""

    @staticmethod
    def _get(smis, keys, d, i):
        if isinstance(d, dict):
            v = d.get(smis[i])
            return d.get(keys[i]) if v is None and keys is not None else v
        return d[i] if i < len(d) else None

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
            cols = []
            for d, lam in use:
                v = [Fwd._get(smis, keys, d, i) for i in idx]
                v = [float(x) if x is not None and math.isfinite(float(x)) else None for x in v]
                if sum(x is not None for x in v) >= 2:
                    cols.append((v, lam))
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


def real_rerank():
    p = 'D:/casmi-data/input/datasets/ahmedberatozer/casmi26-iceberg/fuse.py'
    if not os.path.exists(p):
        return None
    g = {}
    exec(compile(open(p, encoding='utf-8').read(), 'fuse', 'exec'), g)
    return g['rerank']


def window(rng, n=None):
    n = n or rng.randint(2, 40)
    smis = ['s%d' % i for i in range(n)]
    forms = [rng.choice('ABCDE') for _ in smis]
    scores = sorted((rng.gauss(0, 1) for _ in smis), reverse=True)
    ice = {s: (None if rng.random() < 0.2 else rng.random()) for s in smis if rng.random() < 0.9}
    gl_forms = set(rng.sample('ABCDE', rng.randint(0, 5)))
    gl = {s: (None if rng.random() < 0.2 else rng.random()) for s, f in zip(smis, forms) if f in gl_forms}
    return smis, ['K' + s for s in smis], scores, forms, ice, gl


# ------------------------------------------------------------------------------------------------------------------------- s1i -----
def test_s1i_selftest():
    assert I['s1i_selftest'](Fwd.rerank) == 'ok'
    r = real_rerank()
    if r is not None:
        assert I['s1i_selftest'](r) == 'ok'


def test_s1i_only_eligible_groups_move():
    rng = random.Random(0)
    fin = I['s1i_finite']
    n_elig = 0
    for _ in range(600):
        smis, keys, scores, forms, ice, gl = window(rng)
        gl_order = Fwd.rerank_multi(smis, keys, scores, forms, [{}, gl], [1.0, 1.0], top_n=len(smis))
        new, info = I['s1i_group_order'](smis, keys, scores, forms, ice, gl, gl_order, Fwd.rerank, 1.0)
        assert sorted(new) == list(range(len(smis)))
        for f in set(forms):
            idx = [i for i in range(len(smis)) if forms[i] == f]
            n_gl = sum(fin(gl.get(smis[i])) for i in idx); n_ice = sum(fin(ice.get(smis[i])) for i in idx)
            if len(idx) >= 2 and n_gl == 0 and n_ice >= 2:
                n_elig += 1
                sub = Fwd.rerank([smis[i] for i in idx], None, [scores[i] for i in idx], [f] * len(idx), [ice.get(smis[i]) for i in idx], lam=1.0, top_n=len(idx))
                assert [new[i] for i in idx] == [idx[k] for k in sub]          # = the chain's ICE re-rank of that group alone
            else:
                assert [new[i] for i in idx] == [gl_order[i] for i in idx]     # GLACIER's order kept
        assert info['groups_eligible'] == sum(1 for f in set(forms) if len([i for i in range(len(smis)) if forms[i] == f]) >= 2
                                              and sum(fin(gl.get(s)) for s, g in zip(smis, forms) if g == f) == 0
                                              and sum(fin(ice.get(s)) for s, g in zip(smis, forms) if g == f) >= 2)
        assert I['s1i_group_order'](smis, keys, scores, forms, {}, gl, gl_order, Fwd.rerank, 1.0)[0] == gl_order
    assert n_elig > 100


def test_s1i_apply_switch_selftest_fallback():
    rng = random.Random(1)
    smis, keys, scores, forms, ice, gl = window(rng, 30)
    fm = dict(zip(smis, forms))
    order = Fwd.rerank_multi(smis, keys, scores, forms, [{}, gl], [1.0, 1.0], top_n=30)

    def scope(on=True, selftest='ok', form=lambda s: fm[s]):
        st = I['s1i_new_stats'](dict(on=on)); st['selftest'] = selftest
        return dict(S1I=dict(on=on), S1I_STATS=st, _form2=form, ice_fuse=Fwd, ICE_LAM=1.0)
    assert I['s1i_apply'](scope(on=False), 'm', smis, keys, scores, order, ice, gl) is order
    assert I['s1i_apply'](scope(selftest='FAILED x'), 'm', smis, keys, scores, order, ice, gl) is order

    def bad(s):
        raise ValueError('boom')
    sc = scope(form=bad)
    assert I['s1i_apply'](sc, 'm', smis, keys, scores, order, ice, gl) is order and sc['S1I_STATS']['fallback'] == 1
    sc = scope()
    new = I['s1i_apply'](sc, 'm', smis, keys, scores, order, ice, gl)
    ref, info = I['s1i_group_order'](smis, keys, scores, forms, ice, gl, order, Fwd.rerank, 1.0)
    assert new == ref and sc['S1I_STATS']['groups_eligible'] == info['groups_eligible'] and sc['S1I_STATS']['calls'] == 1


def _cell53_fns(text):
    a = text.index('    def _finite_coverage('); b = text.index('    def run_overlay(')
    ns = {'math': math}
    exec(compile(textwrap.dedent(text[a:b]), 'c53', 'exec'), ns)
    return ns['rerank_row']


def test_s1i_patched_cell53_identity_and_effect():
    if not HAVE_S1:
        print('skip (s1 notebook absent)'); return
    sys.path.insert(0, os.path.join(ROOT, 'scratch', 's1var'))
    import build_s1var as B
    c53 = ''.join(json.load(open(S1_NB, encoding='utf-8'))['cells'][B.I53]['source'])
    orig, patched = _cell53_fns(c53), _cell53_fns(B.patch_cell53(c53))
    rng = random.Random(2)
    n_changed = 0

    def mf(first, eng, identity, alpha, offset, limit):
        c = list(dict.fromkeys(first + eng))[:limit]
        return c, [float(len(c) - i) for i in range(len(c))], 0
    for t in range(300):
        smis, keys, scores, forms, ice, gl = window(rng, rng.randint(5, 45))
        fm = dict(zip(smis, forms))
        n1 = rng.random() < 0.2
        base = dict(ENG={'m': dict(smiles=smis[3:] + smis[:1])}, PREFUSION_LISTS={'m': smis[:25]}, metric_fusion=mf, metric_identity=None,
                    ALPHA=0.6, KRR=3.0, chem=type('C', (), dict(score_key=staticmethod(lambda s: 'K' + s))), ICE_SCORES={'m': ice},
                    gl_of=lambda m: gl if any(v is not None for v in gl.values()) else {}, gl_fuse=Fwd, ice_fuse=Fwd, _form2=lambda s: fm[s],
                    ICE_LAM=1.0, GL_LAM=1.0, FRAG_GAP={}, safe_extend=lambda a, b: list(dict.fromkeys(list(a) + list(b)))[:25],
                    N1_GL_MIDS={'m'} if n1 else set())
        st = I['s1i_new_stats'](dict(on=True)); st['selftest'] = 'ok'
        on = dict(base, S1I=dict(on=True), S1I_STATS=st, s1i_apply=I['s1i_apply'])
        off = dict(base, S1I=dict(on=False), S1I_STATS=I['s1i_new_stats'](dict(on=False)), s1i_apply=I['s1i_apply'])
        orig_list = smis[:25]
        for fb in (False, True):
            ref, au = orig(base, 'm', orig_list, window=40, ice_fallback=fb, processed=True)
            assert patched(base, 'm', orig_list, window=40, ice_fallback=fb, processed=True) == (ref, au)     # no s1i in scope
            assert patched(off, 'm', orig_list, window=40, ice_fallback=fb, processed=True)[0] == ref
            got, au2 = patched(on, 'm', orig_list, window=40, ice_fallback=fb, processed=True)
            if not fb or n1 or au['branch'] != 'glacier':
                assert got == ref
            else:
                cand, sc, _ = mf([s for s in smis[:25]], base['ENG']['m']['smiles'], None, 0.6, 3.0, 40)
                kk = ['K' + s for s in cand]; ff = [fm[s] for s in cand]
                g = base['gl_of']('m')
                o = Fwd.rerank_multi(cand, kk, sc, ff, [{}, g], [1.0, 1.0], top_n=len(cand))
                o2, _ = I['s1i_group_order'](cand, kk, sc, ff, ice, g, o, Fwd.rerank, 1.0)
                exp = base['safe_extend']([cand[i] for i in o2][:25], [cand[i] for i in o2][25:] + orig_list)
                assert got == exp
                n_changed += got != ref
    assert n_changed > 10, n_changed


# ------------------------------------------------------------------------------------------------------------------------- s1m -----
KEY = lambda s: s.lower() if isinstance(s, str) and s.strip() and s != 'CCO' else None


def rank2(toks, top):
    """cell 54's row logic (E2R2)."""
    toks = [t for t in toks if t]
    key = KEY(top) if top else None
    if toks and toks[0] != 'CCO' and key is not None:
        keys = [KEY(t) for t in toks]
        if key != keys[0]:
            rest = [t for t, k in zip(toks, keys) if k != key]
            toks = (rest[:1] + [top] + rest[1:])[:25]
    return toks


def shield(e006, b):
    kb = KEY(b)
    if KEY(e006[0]) == kb:
        return e006
    twin = next((t for t in e006 if KEY(t) == kb), b)
    return ([twin] + [t for t in e006 if KEY(t) != kb])[:25]


def test_s1m_selftest():
    assert M['s1m_selftest']() == 'ok'


def test_s1m_equals_shield_then_e2r2():
    rng = random.Random(3)
    pool = [chr(65 + i) for i in range(26)] + [chr(97 + i) for i in range(3)]       # 'a'..'c' are twins of 'A'..'C'
    n_shield = 0
    for _ in range(3000):
        e006 = list(dict.fromkeys(rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(rng.randint(1, 30))))[:25]
        b = rng.choice(pool); e2 = rng.choice(pool + [None])
        F = rank2(e006, e2)
        new, tag = M['s1m_row'](F, b, 2.0, 1.0, e2, KEY, dict(shield=True))
        assert new == rank2(shield(e006, b), e2) or [KEY(t) for t in new] == [KEY(t) for t in rank2(shield(e006, b), e2)], (e006, b, e2, F, new)
        assert len(new) <= 25 and len({KEY(t) for t in new}) == len(new)
        n_shield += tag == 'shield'
        assert M['s1m_row'](F, b, 0.5, 1.0, e2, KEY, dict(shield=True)) == (F, 'below')
        assert M['s1m_row'](F, b, 2.0, 1.0, e2, KEY, dict(shield=False)) == (F, 'off')
    assert n_shield > 1000


def _run_main(rows, E1, eng, sw, ident=KEY):
    import pandas as pd
    d = tempfile.mkdtemp()
    p = os.path.join(d, 'submission.csv')
    pd.DataFrame(rows, columns=['molecule_id', 'smiles']).to_csv(p, index=False)
    before = open(p, 'rb').read()
    out = io.StringIO(); old = sys.stdout; sys.stdout = out
    try:
        st = M['_s1m_main'](dict(S1M=sw, S1M_E1=E1, ENG=eng, metric_identity=ident), workdir=d)
    finally:
        sys.stdout = old
    return st, before, open(p, 'rb').read(), out.getvalue()


def test_s1m_file_cell():
    rows = [('m1', 'X;E;B;Y'), ('m2', 'B;E;X'), ('m3', 'CCO'), ('m4', 'X;Y')]
    E1 = dict(m1=dict(top='B', margin=3.0), m2=dict(top='B', margin=3.0), m3=dict(top='B', margin=3.0), m4=dict(top='Y', margin=0.1))
    eng = dict(m1=dict(smiles=['E']), m2=dict(smiles=['E']), m4=dict(smiles=['Y']))
    st, b0, b1, log = _run_main(rows, E1, eng, dict(shield=True, tau=1.0))
    assert st['status'] == 'applied' and st['shield'] == 1 and st['changed'] == 1 and st['fallback'] == 0, st
    assert b1.decode().splitlines()[1] == 'm1,B;E;X;Y' and b1.decode().splitlines()[2:] == b0.decode().splitlines()[2:]
    assert log.count('S1M_PATCH') == 1
    st, b0, b1, log = _run_main(rows, E1, eng, dict(shield=False, tau=1.0))
    assert st['status'] == 'off' and b0 == b1 and log.count('S1M_PATCH') == 1
    st, b0, b1, _ = _run_main(rows, E1, eng, dict(shield=True, tau=None))
    assert st['status'] == 'off (self-test)' and b0 == b1

    def flaky(s):
        if s == 'Y':
            raise RuntimeError('identity failed')
        return KEY(s)
    st, b0, b1, log = _run_main(rows, E1, eng, dict(shield=True, tau=1.0), ident=flaky)
    assert st['fallback'] >= 1 and 'S1M_FALLBACK' in log and st['status'] == 'applied'
    st, b0, b1, _ = _run_main(rows, {}, eng, dict(shield=True, tau=1.0))
    assert st['status'].startswith('FAILED') and b0 == b1                     # no capture -> s1's file kept


def test_s1m_capture_cell():
    if not HAVE_S1:
        print('skip (s1 notebook absent)'); return
    sys.path.insert(0, os.path.join(ROOT, 'scratch', 's1var'))
    import build_s1var as B
    BASE = {'m1': (['A', 'B', 'C'], ['k1', 'k2', 'k3'], 0.5, [2.0, 0.5, 0.1], ['F'] * 3), 'm2': (['A'], ['k1'], 0.2, [1.0], ['F']),
            'm3': ([], [], 0.0, [], [])}
    out = io.StringIO(); old = sys.stdout; sys.stdout = out
    try:
        g = dict(BASE=BASE, json=json); exec(B.capture_cell(0.75), g)
        g2 = dict(BASE=BASE, json=json); exec(B.capture_cell(0.75, shield=False), g2)
    finally:
        sys.stdout = old
    assert g['S1M'] == dict(shield=True, tau=0.75, rule=g['S1M']['rule'])
    assert g['S1M_E1'] == dict(m1=dict(top='A', margin=1.5), m2=dict(top='A', margin=None), m3=dict(top=None, margin=None))
    assert g2['S1M_E1'] == {} and 'S1M_CAPTURE' in out.getvalue()


def test_builder():
    if not HAVE_S1:
        print('skip (s1 notebook absent)'); return
    sys.path.insert(0, os.path.join(ROOT, 'scratch', 's1var'))
    import ast
    import build_s1var as B
    s1 = json.load(open(S1_NB, encoding='utf-8'))
    nb = B.build_i(s1)
    assert B.diff_vs_s1(s1, nb, [B.I53]) == [B.I53]
    assert B.src(nb['cells'][B.I53]) == open(os.path.join(EMBED, 's1i_icegroup.py'), encoding='utf-8').read()
    c53 = B.src(nb['cells'][B.I53 + 1]); ast.parse(c53)
    assert 'scope["s1i_apply"](scope, mid, candidates, keys, scores, order, ice, gl)' in c53 and "print('S1I_PATCH'" in c53
    assert 'S1I = dict(on=False)' in B.src(B.build_i(s1, on=False)['cells'][B.I53])
    nb = B.build_m(s1, 0.4321)
    ins = [B.I40 + 1, B.I55 + 2]
    assert B.diff_vs_s1(s1, nb, ins) == []
    assert "'tau': 0.4321" in B.src(nb['cells'][B.I40 + 1]) and B.src(nb['cells'][B.I55 + 2]).startswith('"""s1m cell')
    assert B.src(nb['cells'][B.I55 + 1]).startswith('V43_DATASET_SOURCES')         # the shield runs after the receipt
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            ast.parse(B.src(c))
    try:
        B.build_m(s1, None); raise SystemExit('built s1m without tau')
    except AssertionError:
        pass


if __name__ == '__main__':
    for n, f in list(globals().items()):
        if n.startswith('test_') and callable(f):
            f(); print('ok', n)
